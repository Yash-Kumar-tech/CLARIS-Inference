from typing import Optional, Tuple
import torch

from utils import Dictionary

class Search(torch.nn.Module):
    def __init__(self, tgtDict: Dictionary):
        super().__init__()
        self.pad = tgtDict.pad()
        self.unk = tgtDict.unk()
        self.eos = tgtDict.eos()
        self.vocabSize = len(tgtDict)
        self.srcLengths = torch.tensor(-1)
        self.supportsConstraints = False
        self.stopOnMaxLen = False
        
    def step(
        self,
        step,
        lprobs,
        scores,
        prevOutputTokens = None,
        originalBatchIdxs = None,
    ):
        raise NotImplementedError
    
    @torch.jit.export
    def setSrcLengths(self, srcLengths):
        self.srcLengths = srcLengths
        
    @torch.jit.export
    def initConstraints(
        self,
        batchConstraints: Optional[torch.Tensor],
        beamSize: int
    ):
        pass
    
    def pruneSentences(self, batchIdxs: torch.Tensor):
        pass
    
    def updateConstraints(self, activeHypos: torch.Tensor):
        pass
    
class BeamSearch(Search):
    def __init__(self, tgtDict: Dictionary):
        super().__init__(tgtDict)
        self.constrantStates = None
    
    @torch.jit.export
    def step(
        self,
        step: int,
        lprobs: torch.Tensor,
        scores: Optional[torch.Tensor],
        prevOutputTokens: Optional[torch.Tensor] = None,
        originalBatchIdxs: Optional[torch.Tensor] = None,
        candidateMultiple: int = 2,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, beamSize, vocabSize = lprobs.size()
        
        if step == 0:
            lprobs = lprobs[:, ::beamSize, :].contiguous()
        else:
            assert scores is not None
            lprobs = lprobs + scores[:, :, step - 1].unsqueeze(-1)
        
        topPrediction = torch.topk(
            lprobs.view(bsz, -1),
            k = min(
                candidateMultiple * beamSize,
                lprobs.view(bsz, -1).size(1) - 1,
            ),
        )
        
        scoresBuf, indicesBuf = topPrediction[0], topPrediction[1]
        
        beamsBuf = torch.div(indicesBuf, vocabSize, rounding_mode = 'trunc')
        indicesBuf = indicesBuf.fmod(vocabSize)
        
        return scoresBuf, indicesBuf, beamsBuf