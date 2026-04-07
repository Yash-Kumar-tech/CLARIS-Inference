from typing import Dict, List, Optional, Any
import torch
import math

from modules.base_dropout import BaseDropout
from modules.base_transformer_decoder_layer import BaseTransformerDecoderLayer
from modules.layer_drop_module_list import LayerDropModuleList
from modules.layer_norm import LayerNorm
from modules.linear import Linear
from modules.positional_embedding import PositionalEmbedding
from modules.stacked_embedding import StackedEmbedding
from params import ModelParams
from utils import Dictionary, fillWithNegInf

class TransformerUnitDecoder(torch.nn.Module):
    def __init__(
        self,
        params: ModelParams,
        dictionary: Dictionary,
        embedTokens: Optional[StackedEmbedding],
        noEncoderAttn: bool = False,
        outputProjection: Optional[torch.nn.Module] = None
    ):
        super().__init__()
        self.params = params
        self.dictionary = dictionary
        self.onnxTrace = False
        self.adaptiveSoftmax = None
        self.register_buffer("version", torch.Tensor([3]))
        self._futureMask = torch.empty(0)
        
        self.decoderLayerdrop = params.decoder.layerdrop
        self.shareInputOutputEmbed = params.shareDecoderInputOutputEmbed # TODO: Check
        
        inputEmbedDim = embedTokens.embedding_dim
        self.embedDim = params.decoder.embedDim
        self.outputEmbedDim = params.decoder.outputDim
        
        self.paddingIdx = embedTokens.padding_idx
        self.maxTargetPositions = params.maxTargetPositions
        
        self.embedTokens = embedTokens
        
        self.embedScale = 1.0 if params.noScaleEmbedding else math.sqrt(self.embedDim)
        
        self.quantNoise = None
        self.projectInDim = (
            Linear(inputEmbedDim, self.embedDim, bias = False)
            if self.embedDim != inputEmbedDim else None
        )
        
        self.embedPositions = (
            PositionalEmbedding(
                self.maxTargetPositions,
                self.embedDim,
                self.paddingIdx,
                learned = params.decoder.learnedPos
            )
            if not params.noTokenPositionalEmbeddings else None
        )
        
        self.layernormEmbedding = None
        
        self.crossSelfAttention = params.crossSelfAttention
        
        if self.decoderLayerdrop > 0.0:
            self.layers = LayerDropModuleList(p = self.decoderLayerdrop)
        else:
            self.layers = torch.nn.ModuleList([])
            
        self.layers.extend([
            self.buildDecoderLayer(params, noEncoderAttn)
            for _ in range(params.decoder.layers)
        ])
        
        self.numLayers = len(self.layers)
        
        self.layerNorm = LayerNorm(self.embedDim, export = params.export)
        
        self.outputProjection = outputProjection
        
        if self.outputProjection is None:
            self.buildOutputProjection(embedTokens)
            
    def buildOutputProjection(
        self,
        embedTokens: StackedEmbedding
    ):
        self.outputProjection = torch.nn.Linear(
            embedTokens.weight.shape[1],
            embedTokens.weight.shape[0],
            bias = False
        )
        self.outputProjection.weight = embedTokens.weight
        
    def buildDecoderLayer(
        self,
        params,
        noEncoderAttn: bool = False
    ):
        layer = BaseTransformerDecoderLayer(params, noEncoderAttn)        
        return layer
    
    def getNormalizedProbs(
        self,
        netOutput: torch.Tensor,
        logProbs: bool,
        sample: Any = None
    ):
        return self.getNormalizedProbsScriptable(netOutput, logProbs, sample)
    
    def getNormalizedProbsScriptable(
        self,
        netOutput: torch.Tensor,
        logProbs: bool,
        sample: Any = None
    ):
        if logProbs:
            return torch.nn.functional.log_softmax(netOutput, dim = -1)
        return torch.nn.functional.softmax(netOutput, dim = -1)
    
    def maxPositions(self):
        return  int(1e6)
            
    def forward(
        self,
        prevOutputTokens: torch.LongTensor,
        encoderOut: Optional[torch.Tensor] = None,
        encoderPaddingMask: Optional[torch.Tensor] = None,
        incrementalState: Optional[Dict[str, Optional[Dict[str, Optional[torch.Tensor]]]]] = None,
        featuresOnly: bool = False,
        fullContextAlignment: bool = False,
        alignmentLayer: Optional[int] = None,
        alignmentHeads: Optional[int] = None,
        srcLengths: Optional[Any] = None,
        returnAllHiddens: bool = False
    ):
        # srcLengths and returnAllHiddens are used for debugging and ablations
        # returnAllHiddens will return the activations from all layers
        # This can be used to evaluate the CTC decoder
        # If required, can be turned on manually but does increase latency
        x, attn, innerStates = self.extractFeatures(
            prevOutputTokens = prevOutputTokens,
            encoderOut = encoderOut,
            encoderPaddingMask = encoderPaddingMask,
            incrementalState = incrementalState,
            fullContextAlignment = fullContextAlignment,
            alignmentLayer = alignmentLayer,
            alignmentHeads = alignmentHeads
        )
        
        if not featuresOnly:
            x = self.outputLayer(x)
        return x, attn, innerStates
    
    def extractFeatures(
        self,
        prevOutputTokens: torch.Tensor,
        encoderOut: Optional[torch.Tensor],
        encoderPaddingMask: Optional[torch.Tensor],
        incrementalState: Optional[Dict[str, Optional[Dict[str, Optional[torch.Tensor]]]]] = None,
        fullContextAlignment: bool = False,
        alignmentLayer: Optional[int] = None,
        alignmentHeads: Optional[int] = None,
    ):
        return self.extractFeaturesScriptable(
            prevOutputTokens,
            encoderOut,
            encoderPaddingMask,
            incrementalState,
            fullContextAlignment,
            alignmentLayer,
            alignmentHeads
        )
        
    def extractFeaturesScriptable(
        self,
        prevOutputTokens: torch.Tensor,
        encoderOut: Optional[torch.Tensor],
        encoderPaddingMask: Optional[torch.Tensor],
        incrementalState: Optional[Dict[str, Optional[Dict[str, Optional[torch.Tensor]]]]] = None,
        fullContextAlignment: bool = False,
        alignmentLayer: Optional[int] = None,
        alignmentHeads: Optional[int] = None,
    ):
        if alignmentLayer is None:
            alignmentLayer = self.numLayers - 1
            
        enc: Optional[torch.Tensor] = None
        paddingMask: Optional[torch.Tensor] = None
        if encoderOut is not None and encoderOut.shape[0] > 0:
            enc = encoderOut[0]
        if encoderPaddingMask is not None and encoderPaddingMask.shape[0] > 0:
            paddingMask = encoderPaddingMask[0]
            
        positions = None
        if self.embedPositions is not None:
            positions = self.embedPositions(prevOutputTokens, incrementalState is None)
            
        if incrementalState is  not None:
            prevOutputTokens = prevOutputTokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]
                
        prevOutputTokens = prevOutputTokens.contiguous()
        x = self.embedScale * self.embedTokens(prevOutputTokens)
        
        if self.projectInDim is not None:
            x = self.projectInDim(x)
            
        if positions is not None:
            x += positions
            
        # x = self.dropoutModule(x)
        
        x = x.transpose(0, 1) # B x T x C -> T x B x C
        
        selfAttnPaddingMask: Optional[torch.Tensor] = None
        if self.crossSelfAttention or prevOutputTokens.eq(self.paddingIdx).any():
            selfAttnPaddingMask = prevOutputTokens.eq(self.paddingIdx)
            
        attn: Optional[torch.Tensor] = None
        innerStates: Optional[torch.Tensor] = x.unsqueeze(0)
        
        for idx, layer in enumerate(self.layers):
            if incrementalState is None and not fullContextAlignment:
                selfAttnMask = self.bufferedFutureMask(x)
            else:
                selfAttnMask = None
            
            x, layerAttn, _ = layer(
                x,
                enc,
                paddingMask,
                incrementalState,
                selfAttnMask = selfAttnMask,
                selfAttnPaddingMask = selfAttnPaddingMask,
                needAttn = bool(idx == alignmentLayer),
                needHeadWeights = bool(idx == alignmentLayer)
            )
            
            innerStates = torch.cat((innerStates, x.unsqueeze(0)), dim = 0)
            
            if layerAttn is not None and idx == alignmentLayer:
                attn = layerAttn.float().to(x)
                
        if attn is not None:
            if alignmentHeads is not None:
                attn = attn[:alignmentHeads]
            attn = attn.mean(dim = 0) # Average probabilities over heads
        
        if self.layerNorm is not None:
            x = self.layerNorm(x)
            
        x = x.transpose(0, 1) # T x B x C -> B x T x C
        
        return x, attn.unsqueeze(0), innerStates
    
    def outputLayer(self, features):
        return self.outputProjection(features)
    
    def maxPositions(self):
        if self.embedPositions is None:
            return self.maxTargetPositions
        return min(self.maxTargetPositions, self.embedPositions.maxPositions)
    
    def bufferedFutureMask(self, tensor: torch.Tensor):
        dim = tensor.size(0)
        if (
            self._futureMask.size(0) == 0
            or (not self._futureMask.device == tensor.device)
            or self._futureMask.size(0) < dim
        ):
            self._futureMask = torch.triu(
                fillWithNegInf(torch.zeros([dim, dim])),  1
            )
        self._futureMask = self._futureMask.to(tensor)
        return self._futureMask[:dim, :dim]
    
    def reorderIncrementalState(
        self,
        incrementalState: Dict[str, Dict[str, Optional[torch.Tensor]]],
        newOrder: torch.Tensor
    ):
        pass
    
    def reorderIncrementalStateScripting(
        self,
        incrementalState: Dict[str, Dict[str, Optional[torch.Tensor]]],
        newOrder: torch.Tensor
    ):
        # None of the child modules have reorderIncrementalState method implemented
        # otherwise the following would have been useful
        for module in self.modules():
            if hasattr(module, "reorderIncrementalState"):
                result = module.reorderIncrementalState(incrementalState, newOrder)
                if result is not None:
                    incrementalState = result
        
    def setBeamSize(self, beamSize: int):
        if getattr(self, "_beamSize", -1) != beamSize:
            seen = set()
            
            def applySetBeamSize(module: torch.nn.Module):
                if (
                    module != self and hasattr(module, "setBeamSize")
                    and module not in seen
                ):
                    seen.add(module)
                    module.setBeamSize(beamSize)
            self.apply(applySetBeamSize)
            self._beamSize = beamSize