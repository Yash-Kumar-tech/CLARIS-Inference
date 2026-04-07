import torch
from typing import Dict, Any, Optional, List, Tuple
import argparse

from modules.generator import Generator
from modules.variance_predictor import VariancePredictor
from params import CodeGeneratorParams
from utils import getVocoderStateDictFromPath

class CodeGenerator(Generator):
    def __init__(self, params: CodeGeneratorParams):
        super().__init__(params)
        self.dict = torch.nn.Embedding(
            params.numEmbeddings,
            params.embeddingDim
        )
        self.multispkr = None
        self.embedder = params.embedderParams
        
        if self.embedder:
            self.spkr = torch.nn.Linear(params.embeddingDim, params.embeddingDim)
        
        self.durPredictor = None
        if params.durPredictorParams is not None:
            self.durPredictor = VariancePredictor(params.durPredictorParams)
        
        self.f0 = params.f0
        nF0Bin = params.f0QuantNumBin
        self.f0QuantEmbed = (
            None if nF0Bin <= 0
            else torch.nn.Embedding(nF0Bin, params.embeddingDim)
        )
        
    @staticmethod
    def _upsample(signal: torch.Tensor, maxFrames: int):
        if signal.dim() == 3:
            bsz, channels, condLength = signal.size()
        elif signal.dim() == 2:
            signal = signal.unsqueeze(2)
            bsz, channels, condLength = signal.size()
        else:
            signal = signal.view(-1, 1, 1)
            bsz, channels, condLength = signal.size()
        
        signal = signal.unsqueeze(3).repeat(1, 1, 1, maxFrames // condLength)
        reminder = (maxFrames - signal.shape[2] * signal.shape[3]) // signal.shape[3]
        if reminder > 0:
            raise NotImplementedError("Padding condition signal - misalignment between condition features")
        
        signal = signal.view(bsz, channels, maxFrames)
        return signal
    
    def forward(self, code: torch.LongTensor, durPrediction: bool = False):
        x: torch.Tensor = self.dict(code).transpose(1, 2)
        
        if self.durPredictor and durPrediction:
            assert x.size(0), "only support signal sample"
            logDurPred = self.durPredictor(x.transpose(1, 2))
            durOut = torch.clamp(
                torch.round((torch.exp(logDurPred) - 1)).long(),
                min = 1
            )
            x = torch.repeat_interleave(x, durOut.view(-1), dim = 2)
        
        return super().forward(x)

class CodeHiFiGANVocoder(torch.nn.Module):
    def __init__(
        self,
        checkpointPath: str,
        params: CodeGeneratorParams,
        fp16: bool = False
    ):
        super().__init__()
        self.model = CodeGenerator(params = params)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        stateDict = torch.load(checkpointPath, map_location = device)
        
        self.model.load_state_dict(getVocoderStateDictFromPath(
            checkpointPath = checkpointPath,
            modelStateDict = self.model.state_dict()
        ))
        self.model.eval()
        
        if fp16:
            self.model.half()
            
        self.model.removeWeightNorm()
    
    def getTargets(self, sample: Dict[str, torch.Tensor], netOutput: Any):
        return sample["target"]
    
    def getNormalizedProbs(
        self,
        netOutput: Tuple[torch.Tensor, Optional[Dict[str, List[Optional[torch.Tensor]]]]],
        logProbs: bool,
        sample: Optional[Dict[str, torch.Tensor]] = None
    ):
        return self.getNormalizedProbsScriptable(netOutput, logProbs, sample)
    
    def getNormalizedProbsScriptable(
        self,
        netOutput: Tuple[torch.Tensor, Optional[Dict[str, List[Optional[torch.Tensor]]]]],
        logProbs: bool,
        sample: Optional[Dict[str, torch.Tensor]] = None
    ):
        if hasattr(self, "decoder"):
            return self.decoder.getNormalizedProbs(netOutput, logProbs, sample)
        elif torch.is_tensor(netOutput):
            logits = netOutput.float()
            if logProbs:
                return torch.nn.functional.log_softmax(logits, dim = -1)
            else:
                return torch.nn.functional.softmax(logits, dim = -1)
        raise NotImplementedError
    
    def extractFeatures(self, *args, **kwargs):
        return self(*args, **kwargs)
    
    def maxPositions(self):
        return None
    
    def forward(
        self,
        code: torch.LongTensor,
        durPrediction: bool = False
    ) -> torch.Tensor:
        
        mask = code >= 0
        code = code[mask].unsqueeze(dim = 0)
        
        # if "f0" in x:
        #     f0UpRatio = x["f0"].size(1) // code.size(1)
        #     mask = mask.unsqueeze(2).repeat(1, 1, f0UpRatio).view(-1, x["f0"].size(1))
        #     x["f0"] = x["f0"][mask].unsqueeze(dim = 0)
        return self.model.forward(code, durPrediction).detach().squeeze()
    
    