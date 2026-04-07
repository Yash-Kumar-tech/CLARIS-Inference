import math
from typing import Any, Dict, List, Optional, Set, Tuple
import torch

from model import SpeechToUnitTransformer
from generate import Dictionary
from ensemble_model import EnsembleModel
from search import BeamSearch

class SequenceGenerator(torch.nn.Module):
    def __init__(
        self,
        model: SpeechToUnitTransformer,
        tgtDict: Dictionary,
        beamSize: int = 1,
        maxLenA: int = 0,
        maxLenB: int = 200,
        maxLen: int = 0,
        minLen: int = 1,
        normalizeScores: bool = True,
        lenPenalty: float = 1.0,
        unkPenalty: float = 0.0,
        temperature: float = 1.0,
        matchSourceLen: bool = False,
        noRepeatNgramSize: int = 0,
        searchStrategy: Any = None,
        eos: Optional[int] = None,
        symbolsToStripFromOutput: Optional[Set[str]] = None,
        tokensToSuppress: Tuple = ()
    ):
        super().__init__()
        self.model = EnsembleModel(model)
        self.tgtDict = tgtDict
        self.pad = tgtDict.pad()
        self.unk = tgtDict.unk()
        self.eos = eos or tgtDict.eos()
        self.symbolsToStripFromOutput = (
            symbolsToStripFromOutput.union({self.eos})
            if symbolsToStripFromOutput is not None else {self.eos}
        )
        
        self.tokenIndicesToSuppress: Optional[torch.Tensor] = None
        tokenIndicesToSuppress = []
        for tokenString in tokensToSuppress:
            tokenIndex = tgtDict.index(tokenString)
            assert tokenIndex != self.unk
            tokenIndicesToSuppress.append(tokenIndex)
        
        if len(tokenIndicesToSuppress) > 0:
            self.tokenIndicesToSuppress = torch.Tensor(tokenIndicesToSuppress).long()
        
        self.vocabSize = len(tgtDict)
        self.beamSize = beamSize
        self.beamSize = min(beamSize, self.vocabSize - 1)
        self.model.setDecoderBeamSize(self.beamSize)
        self.maxLenA = maxLenA
        self.maxLenB = maxLenB
        self.minLen = minLen
        self.maxLen = maxLen or self.model.maxDecoderPositions()
        
        self.normalizeScores = normalizeScores
        self.lenPenalty = lenPenalty
        self.unkPenalty = unkPenalty
        self.temperature = temperature
        self.matchSourceLen = matchSourceLen
        
        assert temperature > 0, "--temperature must be greater than 0"
        
        self.search: BeamSearch = BeamSearch(tgtDict) if searchStrategy is None else searchStrategy
        
        self.shouldSetSrcLengths = False # Check   
        self.model.eval()
            
    def cuda(self):
        self.model.cuda()
        return self
    
    @torch.no_grad()
    def forward(
        self,
        srcTokens: torch.Tensor,
        srcLengths: torch.Tensor,
        prefixTokens: Optional[torch.Tensor] = None,
        bosToken: Optional[int] = None
    ):
        return self._generate(
            srcTokens,
            srcLengths,
            prefixTokens,
            bosToken = bosToken,
        )
        
    @torch.no_grad()
    def generate(
        self,
        srcTokens: torch.Tensor,
        srcLengths: torch.Tensor,
        ids: torch.Tensor,
        **kwargs
    ):
        return self._generate(srcTokens, srcLengths, ids, **kwargs)
    
    def _generate(
        self,
        srcTokens: torch.Tensor,
        srcLengths: torch.Tensor,
        ids: torch.Tensor,
        prefixTokens: Optional[torch.Tensor] = None,
        constraints: Optional[torch.Tensor] = None,
        bosToken: Optional[int] = None,
    ):
        incrementalStates = torch.jit.annotate(
            List[Dict[str, Dict[str, Optional[torch.Tensor]]]],
            [
                torch.jit.annotate(Dict[str, Dict[str, Optional[torch.Tensor]]], {})
                for _ in range(self.model.modelsSize)
            ]
        )
        
        bsz, srcLen = srcTokens.size()[:2]
        beamSize = self.beamSize
        
        if constraints is not None and not self.search.supportsConstraints:
            raise NotImplementedError(
                f"Constraints were provided but search strategy {self.search.__class__.__name__} does not support constraints"
            )
        
        self.search.initConstraints(constraints, beamSize)
        
        maxLen: int = -1
        if self.matchSourceLen:
            maxLen = srcLengths.max().item()
        else:
            maxLen = min(
                int(self.maxLenA * srcLen + self.maxLenB),
                self.maxLen - 1
            )
            
        assert self.minLen <= maxLen, f"minLen ({self.minLen}) must be greater than maxLen ({maxLen})"
        
        encoderOut, encoderPaddingMask, encoderStates = self.model.forwardEncoder(
            srcTokens = srcTokens, 
            srcLengths = srcLengths
        )
        newOrder = torch.arange(bsz).view(-1, 1).repeat(1, beamSize).view(-1)
        newOrder = newOrder.to(srcTokens.device).long()
        encoderOut, encoderPaddingMask, _, encoderStates = self.model.reorderEncoderOut(
            encoderOut = encoderOut, 
            encoderPaddingMask = encoderPaddingMask,
            encoderStates = encoderStates,
            newOrder = newOrder
        )
        assert encoderOut is not None
        
        scores = torch.zeros(bsz * beamSize, maxLen + 1).to(srcTokens).float()
        # + 1 for eos; pad not chosen for scoring
        tokens = torch.zeros(bsz * beamSize, maxLen + 2).to(srcTokens).long().fill_(self.pad)
        # + 2 for eos and pad
        tokens[:, 0] = bosToken or self.eos
        attn: Optional[torch.Tensor] = None
        
        candsToIgnore = torch.zeros(bsz, beamSize).to(srcTokens).eq(-1)
        finalized: List[List[Dict[str, torch.Tensor]]] = torch.jit.annotate(
            List[List[Dict[str, torch.Tensor]]],
            [torch.jit.annotate(List[Dict[str, torch.Tensor]], []) for _ in range(bsz)]
        )
        
        finished: List[bool] = [False] * bsz
        numRemainingSent = bsz
        candSize = 2 * beamSize
        bbszOffsets = (torch.arange(0, bsz) * beamSize).unsqueeze(1).type_as(tokens).to(srcTokens.device)
        candOffsets = torch.arange(0, candSize).type_as(tokens).to(srcTokens.device)
        
        reorderState: Optional[torch.Tensor] = None
        batchIdxs: Optional[torch.Tensor] = None
        
        originalBatchIdxs: Optional[torch.Tensor] = ids
        
        for step in range(maxLen + 1): # 1 extra step for EOS marker
            if reorderState is not None:
                if batchIdxs is not None:
                    corr = batchIdxs - torch.arange(batchIdxs.numel()).type_as(batchIdxs)
                    reorderState.view(-1, beamSize).add_(corr.unsqueeze(-1) * beamSize)
                    originalBatchIdxs = originalBatchIdxs[batchIdxs]
                self.model.reorderIncrementalState(incrementalStates, reorderState)
                encoderOut, encoderPaddingMask, _, encoderStates = self.model.reorderEncoderOut(
                    encoderOut = encoderOut,
                    encoderPaddingMask = encoderPaddingMask,
                    encoderStates = encoderStates,
                    newOrder = reorderState
                )
            lprobs, avgAttnScores = self.model.forwardDecoder(
                tokens[:, : step + 1],
                encoderOut = encoderOut,
                encoderPaddingMask = encoderPaddingMask,
                incrementalStates = incrementalStates,
                temperature = self.temperature,
            )

            lprobs[lprobs != lprobs] = torch.tensor(-math.inf).to(lprobs)
            lprobs[:, self.pad] = -math.inf
            lprobs[:, self.unk] -= self.unkPenalty
            
            # Handling max length constraint
            if step >= maxLen:
                lprobs[:, : self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf
            
            if prefixTokens is not None and step < prefixTokens.size(1) and step < maxLen:
                lprobs, tokens, scores = self._prefixTokens(
                    step,
                    lprobs,
                    scores,
                    tokens,
                    prefixTokens,
                    beamSize,
                )
            else:
                if step < self.minLen:
                    lprobs[:, self.eos] = -math.inf
                if self.tokenIndicesToSuppress is not None:
                    lprobs[:, self.tokenIndicesToSuppress] = -math.inf
                    
            if avgAttnScores is not None:
                if attn is None:
                    attn = torch.empty(bsz * beamSize, avgAttnScores.size(1), maxLen + 2).to(scores)
                attn[:, :, step + 1].copy_(avgAttnScores)
            
            scores = scores.type_as(lprobs)
            eosBbszIdx = torch.empty(0).to(scores)
            # Indices of hypothesis ending with eos (finished sentences)
            eosScores = torch.empty(0).to(scores)
            # Scores of hypothesis ending with eos (finished sentences)
            
            if self.shouldSetSrcLengths:
                self.search.setSrcLengths(srcLengths)
            
            candScores, candIndices, candBeams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocabSize),
                scores.view(bsz, beamSize, -1)[:, :, :step],
                tokens[:, : step + 1],
                originalBatchIdxs
            )
            
            candBbszIdx = candBeams.add(bbszOffsets)
            eosMask = candIndices.eq(self.eos) & candScores.ne(-math.inf)
            eosMask[:, : beamSize][candsToIgnore] = torch.tensor(0).to(eosMask)
            
            eosBbszIdx = torch.masked_select(candBbszIdx[:, :beamSize], mask = eosMask[:, :beamSize])
            
            finalizedSents: List[int] = []
            if eosBbszIdx.numel() > 0:
                eosScores = torch.masked_select(candScores[:, :beamSize], mask = eosMask[:, :beamSize])
                
                finalizedSents = self.finalizeHypos(
                    step,
                    eosBbszIdx,
                    eosScores,
                    tokens,
                    scores,
                    finalized,
                    finished,
                    beamSize,
                    attn,
                    srcLengths,
                    maxLen,
                )
                
                numRemainingSent -= len(finalizedSents)
                
            assert numRemainingSent >= 0
            if numRemainingSent == 0:
                break
            if self.search.stopOnMaxLen and step >= maxLen:
                break
            assert step < maxLen, f"{step} should be < {maxLen}"
            
            if len(finalizedSents) > 0:
                newBsz = bsz - len(finalizedSents)
                
                batchMask = torch.ones(bsz, dtype = torch.bool, device = candIndices.device)
                batchMask[finalizedSents] = False
                batchIdxs = torch.arange(bsz, device = candIndices.device).masked_select(batchMask)
                
                self.search.pruneSentences(batchIdxs)
                                
                eosMask = eosMask[batchIdxs]
                candBeams = candBeams[batchIdxs]
                bbszOffsets.resize_(newBsz, 1)
                candBbszIdx = candBeams.add(bbszOffsets)
                candScores = candScores[batchIdxs]
                candIndices = candIndices[batchIdxs]
                
                if prefixTokens is not None:
                    prefixTokens = prefixTokens[batchIdxs]
                srcLengths = srcLengths[batchIdxs]
                candsToIgnore = candsToIgnore[batchIdxs]
                
                scores = scores.view(bsz, -1)[batchIdxs].view(newBsz * beamSize, -1)
                tokens = tokens.view(bsz, -1)[batchIdxs].view(newBsz * beamSize, -1)
                
                if attn is not None:
                    attn = attn.view(bsz, -1)[batchIdxs].view(newBsz * beamSize, attn.size(1), -1)
                bsz = newBsz
                
            else:
                batchIdxs = None
                
            eosMask[:, :beamSize] = ~((~candsToIgnore) & (~eosMask[:, :beamSize]))
            activeMask = torch.add(
                eosMask.type_as(candOffsets) * candSize,
                candOffsets[:eosMask.size(1)],
            )
            
            newCandsToIgnore, activeHypos = torch.topk(activeMask, k = beamSize, dim = 1, largest = False)
            
            candsToIgnore = newCandsToIgnore.ge(candSize)[:, :beamSize]
            assert (~candsToIgnore).any(dim = 1).all()
            
            activeBbszIdx = torch.gather(candBbszIdx, dim = 1, index = activeHypos).view(-1)
            activeScores = torch.gather(candScores, dim = 1, index = activeHypos).view(-1)
            
            tokens[:, :step + 1] = torch.index_select(
                tokens[:, :step + 1],
                dim = 0,
                index = activeBbszIdx
            )
            
            tokens.view(bsz, beamSize, -1)[:, :, step + 1] = torch.gather(
                candIndices, dim = 1, index = activeHypos
            )
            
            if step > 0:
                scores[:, :step] = torch.index_select(
                    scores[:, :step], 
                    dim = 0,
                    index = activeBbszIdx
                )
                
            scores.view(bsz, beamSize, -1)[:, :, step] = torch.gather(
                candScores,
                dim = 1,
                index = activeHypos
            )
            
            self.search.updateConstraints(activeHypos) 
            # TODO: Constraint update doesn't seem necessary, check if it can be removed
            
            if attn is not None:
                attn[:, :, :step + 2] = torch.index_select(
                    attn[:, :, :step + 2],
                    dim = 0,
                    index = activeBbszIdx
                )
                
            reorderState = activeBbszIdx
            
        for sent in range(len(finalized)):
            scores = torch.tensor([
                float(elem["score"].item()) for elem in finalized[sent]
            ])
            _, sortedScoresIndices = torch.sort(scores, descending = True)
            finalized[sent] = [finalized[sent][ssi] for ssi in sortedScoresIndices]
            finalized[sent] = torch.jit.annotate(
                List[Dict[str, torch.Tensor]], finalized[sent]
            )
        return finalized
    
    def _prefixTokens(
        self,
        step: int,
        lprobs: torch.Tensor,
        scores: torch.Tensor,
        tokens: torch.Tensor,
        prefixTokens: torch.Tensor,
        beamSize: int
    ):
        prefixToks = prefixTokens[:, step].unsqueeze(-1).repeat(1, beamSize).view(-1)
        prefixLprobs = lprobs.gather(-1, prefixToks.unsqueeze(-1))
        prefixMask = prefixToks.ne(self.pad)
        lprobs[prefixMask] = torch.tensor(-math.inf).to(lprobs)
        lprobs[prefixMask] = lprobs[prefixMask].scatter(
            -1, 
            prefixToks[prefixMask].unsqueeze(-1), 
            prefixLprobs[prefixMask]
        )
        
        eosMask = prefixToks.eq(self.eos)
        
        if eosMask.any():
            firstBeam = tokens[eosMask].view(-1, beamSize, tokens.size(-1))[:, 0, 1:step + 1]
            eosMaskBatchDim = eosMask.view(-1, beamSize)[:, 0]
            targetPrefix = prefixTokens[eosMaskBatchDim][:, :step]
            assert (firstBeam == targetPrefix).all()
            
            tokens = self.replicateFirstBeam(
                tokens, 
                eosMaskBatchDim, 
                beamSize
            )
            scores = self.replicateFirstBeam(
                scores, 
                eosMaskBatchDim, 
                beamSize
            )
            lprobs = self.replicateFirstBeam(
                lprobs, 
                eosMaskBatchDim, 
                beamSize
            )
            
        return lprobs, tokens, scores
    
    def replicateFirstBeam(
        self,
        tensor: torch.Tensor,
        mask: torch.Tensor,
        beamSize: int
    ) -> torch.Tensor:
        tensor = tensor.view(-1, beamSize, tensor.size(-1))
        tensor[mask] = tensor[mask][:, :1, :]
        return tensor.view(-1, tensor.size(-1))
    
    def finalizeHypos(
        self,
        step: int,
        bbszIdx: torch.Tensor,
        eosScores: torch.Tensor,
        tokens: torch.Tensor,
        scores: torch.Tensor,
        finalized: List[List[Dict[str, torch.Tensor]]],
        finished: List[bool],
        beamSize: int,
        attn: Optional[torch.Tensor],
        srcLengths: torch.Tensor,
        maxLen: int
    ):
        assert bbszIdx.numel() == eosScores.numel()
        
        tokensClone = tokens.index_select(0, bbszIdx)[:, 1:step + 2]
        tokensClone[:, step] = self.eos
        attnClone = (
            attn.index_select(0, bbszIdx)[:, :, 1:step + 2]
            if attn is not None else None
        )
        
        posScores = scores.index_select(0, bbszIdx)[:, :step + 1]
        posScores[:, step] = eosScores
        posScores[:, 1:] = posScores[:, 1:] - posScores[:, :-1]
        
        if self.normalizeScores:
            eosScores /= (step + 1) ** self.lenPenalty
            
        # Records which sentences in the batch are finished.
        # Helps with indexing between: a. the original sentences in the batch and
        # b. the current, possibly-reduced set of sentences
        cumUnfin: List[int] = []
        prev = 0
        for f in finished:
            if f:
                prev += 1
            else:
                cumUnfin.append(prev)
        
        cumFinTensor = torch.tensor(cumUnfin, dtype = torch.int).to(bbszIdx)
        unfinIdx = torch.div(bbszIdx, beamSize, rounding_mode='trunc')
        sent = unfinIdx + torch.index_select(cumFinTensor, 0, unfinIdx)
        
        seen = (sent << 32) + unfinIdx
        uniqueSeen: List[int] = torch.unique(seen).tolist()
        
        if self.matchSourceLen:
            condition = step > torch.index_select(srcLengths, 0, unfinIdx)
            eosScores = torch.where(condition, torch.tensor(-math.inf), eosScores)
        sentList: List[int] = sent.tolist()
        
        for i in range(bbszIdx.size()[0]):
            if len(finalized[sentList[i]]) < beamSize:
                if attnClone is not None:
                    hypoAttn = attnClone[i]
                else:
                    hypoAttn = torch.empty(0)
                
                finalized[sentList[i]].append({
                    "tokens": tokensClone[i],
                    "score": eosScores[i],
                    "attention": hypoAttn, # srcLen x tgtLen
                    "alignment": torch.empty(0),
                    "positionalScores": posScores[i],
                })
                
        newlyFinished: List[int] = []
        for uniqueS in uniqueSeen:
            uniqueSent: int = uniqueS >> 32
            uniqueUnfinIdx: int = uniqueS - (uniqueSent << 32)
            
            if not finished[uniqueSent] and self.isFinished(
                step,
                uniqueUnfinIdx,
                maxLen,
                len(finalized[uniqueSent]),
                beamSize
            ):
                finished[uniqueSent] = True
                newlyFinished.append(uniqueUnfinIdx)

        return newlyFinished
    
    def isFinished(
        self,
        step: int,
        unfinIdx: int,
        maxLen: int,
        finalizedSentLen: int,
        beamSize: int
    ):
        assert finalizedSentLen <= beamSize
        return finalizedSentLen == beamSize or step == maxLen