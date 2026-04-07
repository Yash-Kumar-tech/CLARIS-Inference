import torch
from typing import Optional, Dict, List, Tuple, Any

from modules.transformer_encoder import TransformerEncoder
from modules.stacked_embedding import StackedEmbedding
from modules.transformer_unit_decoder import TransformerUnitDecoder
from params import ModelParams
from utils import Dictionary

class SpeechToUnitTransformer(torch.nn.Module):
    def __init__(
        self,
        encoder: TransformerEncoder,
        decoder: TransformerUnitDecoder
    ):
        super().__init__()
        self._is_generation_fast = False
        self.encoder = encoder
        self.decoder = decoder
            
    def getTargets(self, sample, netOutput):
        return sample["target"]
    
    def getNormalizedProbs(
        self,
        netOutput: Tuple[torch.Tensor, Optional[Dict[str, List[Optional[torch.Tensor]]]]],
        logProbs: bool,
        sample: Optional[Dict[str, torch.Tensor]] = None
    ):
        return self.getNormalizedProbsScriptable(
            netOutput,
            logProbs,
            sample
        )
    
    def getNormalizedProbsScriptable(
        self,
        netOutput: Tuple[torch.Tensor, Optional[Dict[str, List[Optional[torch.Tensor]]]]],
        logProbs: bool,
        sample: Optional[Dict[str, torch.Tensor]] = None
    ):
        if hasattr(self, "decoder"):
            return self.decoder.getNormalizedProbs(netOutput[0], logProbs, sample)
        elif torch.is_tensor(netOutput):
            logits = netOutput.float()
            if logProbs:
                return torch.nn.functional.log_softmax(logits, dim = -1)
            return torch.nn.functional.softmax(logits, dim = -1)
        
        raise NotImplementedError
        
    def extractFeatures(self, *args, **kwargs):
        return self(*args, **kwargs)
    
    def setNumUpdates(self, numUpdates):
        for m in self.modules():
            if hasattr(m, "setNumUpdates") and m != self:
                m.setNumUpdates(numUpdates)
        
    def setEpoch(self, epoch):
        for m in self.modules():
            if hasattr(m, "setEpoch") and m != self:
                m.setEpoch(epoch)
    
    def prepareForInference_(self, cfg: ModelParams):
        kwargs = {}
        kwargs["beamableMmBeamSize"] = (
            None if getattr(cfg.generation, "noBeamableMm", False)
            else getattr(cfg.generation, "beam", 5)
        )
        kwargs["needAttn"] = getattr(cfg.generation, "printAlignment", False)
        
        if getattr(cfg.generation, "retainDropout", False):
            kwargs["retainDropout"] = cfg.generation.retainDropout
            kwargs["retainDropoutModules"] = cfg.generation.retainDropoutModules
            
        self.makeGenerationFast_(**kwargs)
        
    def makeGenerationFast_(self, **kwargs):
        if self._isGenerationFast:
            return
        self._isGenerationFast = True
        
        def applyRemoveWeightNorm(module):
            try:
                torch.nn.utils.remove_weight_norm(module)
            except (AttributeError, ValueError):
                return
            
        self.apply(applyRemoveWeightNorm)
        
        def applyMakeGenerationFast_(
            module: torch.nn.Module, 
            prefix: str
        ):
            if len(prefix) > 0:
                prefix += "."
            
            baseFunc = SpeechToUnitTransformer.makeGenerationFast_
            for n, m in module.named_modules():
                if (
                    m != self
                    and hasattr(m, "makeGenerationFast_")
                    and m.makeGenerationFast_.__func__ is not baseFunc
                ):
                    name = prefix + n
                    m.makeGenerationFast_(name = name, **kwargs)
        
        applyMakeGenerationFast_(self, "")
        self.eval()
        
    
    def prepareForOnnxExport_(self, **kwargs):
        seen = set()
        
        def applyPrepareForOnnxExport_(module: torch.nn.Module):
            if (
                module != self
                and hasattr(module, "prepareForOnnxExport_")
                and module not in seen
            ):
                seen.add(module)
                module.prepareForOnnxExport_(**kwargs)
        self.apply(applyPrepareForOnnxExport_)
        
    def forward(
        self,
        srcTokens: torch.LongTensor,
        srcLengths: torch.LongTensor,
        prevOutputTokens: torch.LongTensor,
        tgtSpeaker = None,
        returnAllHiddens = False
    ):
        encoderOut, encoderPaddingMask, _ = self.encoder(
            srcTokens,
            srcLengths = srcLengths,
            # tgtSpeaker = tgtSpeaker,
            returnAllHiddens = returnAllHiddens,
        )
        decoderOut = self.decoder(
            prevOutputTokens,
            encoderOut = encoderOut[0],
            encoderPaddingMask = encoderPaddingMask[0]
        )
        
        encOut = encoderOut[0]
        encOut = encOut.transpose(1, 0)
        
        if returnAllHiddens:
            decoderOut[-1]["encoderStates"] = encoderOut['encoderStates']
            decoderOut[-1]['encoderPaddingMask'] = encoderOut['encoderPaddingMask']
            
        return decoderOut
    
    def forwardDecoder(self, prevOutputTokens: torch.LongTensor, **kwargs: Any):
        return self.decoder(prevOutputTokens, **kwargs)
    
    def extractFeatures(
        self,
        srcTokens: torch.LongTensor,
        srcLengths: torch.LongTensor,
        prevOutputTokens: torch.LongTensor,
        **kwargs: Any
    ):
        encoderOut, encoderPaddingMask, encoderStates = self.encoder(srcTokens, srcLengths = srcLengths, **kwargs)
        features = self.decoder.extractFeatures(
            prevOutputTokens,
            encoderOut = encoderOut,
            encoderPaddingMask = encoderPaddingMask,
            **kwargs
        )
        return features
    
    def outputLayer(self, features: torch.Tensor, **kwargs: Any):
        return self.decoder.outputLayer(features, **kwargs)
    
    def maxPositions(self):
        return (self.encoder.maxPositions(), self.decoder.maxPositions())
    
    def maxDecoderPositions(self):
        return self.decoder.maxPositions()
    
    @classmethod
    def buildEncoder(cls, params: ModelParams):
        encoder = TransformerEncoder(params)
        return encoder
    
    @classmethod
    def buildModel(cls, params: ModelParams, tgtDict: Dictionary):
        encoder = cls.buildEncoder(params)
        decoder = cls.buildDecoder(params, tgtDict)
        
        baseModel = cls(encoder, decoder)
        
        return baseModel
    
    def forwardEncoder(
        self,
        srcTokens: torch.LongTensor,
        srcLengths: torch.LongTensor,
        **kwargs: Any
    ):
        return self.encoder(
            srcTokens,
            srcLengths = srcLengths,
            # tgtSpeaker = speaker,
            **kwargs
        )
        
    @classmethod
    def buildDecoder(cls, params: ModelParams, tgtDict: Dictionary):
        numEmbeddings = len(tgtDict)
        paddingIdx = tgtDict.pad()
        paddingIdx = 1
        embedTokens = StackedEmbedding(
            numEmbeddings,
            params.decoder.embedDim,
            paddingIdx,
            numStacked = params.nFramesPerStep
        )
        
        return TransformerUnitDecoder(
            params,
            tgtDict,
            embedTokens
        )
        
    def setBeamSize(self, beamSize: int):
        # Beam size will be handled by the SequenceGenerator, kept for compatibility but not required
        pass