import uuid
import torch
from typing import List, Optional, Tuple, Dict

from modules.base_dropout import BaseDropout
from modules.quant_noise import quantNoise
from utils import evalStrDict


# Use of xFormers requires model training with xFormers components
# However, the released checkpoints do not use it 
# xFormers code provided for completeness but can be ignored
try:
    from xformers.components.attention import build_attention # type: ignore
    from xformers.components.attention.utils import maybe_merge_tasks # type: ignore
    
    _xformersAvailable = True
except ImportError:
    _xformersAvailable = False
    
def _maskForXFormers(mask: torch.Tensor, toDtype: Optional[torch.dtype] = None) -> torch.Tensor:
    """
    call to pytorch multihead accepts three mask types:
        - ByteTensor where non-zero means to mask
        - FloatTensor which is an additive mask
        - BoolTensor where True means to mask
    xFormers currently accepts boolean and additive maks. For boolean masks
    the values have opposite meaning. For a BoolTensor True mean to keep the value.
    """
    floatTypes = [torch.float, torch.float16]
    additive = mask.dtype in floatTypes
    toDtype = toDtype or mask.dtype
    toAdditive = toDtype in floatTypes
    
    if additive:
        if toAdditive:
            return mask.to(toDtype)
        mask = mask < 0
    
    if toAdditive:
        newMask = torch.zeros_like(mask, dtype = toDtype)
        newMask = newMask.masked_fill_(mask, float("-inf"))
    
    mask = ~mask.to(torch.bool)
    mask = mask.to(toDtype)
    return mask

class MultiheadAttention(torch.nn.Module):
    def __init__(
        self,
        embedDim: int,
        numHeads: int,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True,
        addBiasKv: bool = False,
        addZeroAttn: bool = False,
        selfAttention: bool = False,
        encoderDecoderAttention: bool = False,
        dictionary = None,
        qNoise: float = 0.0,
        qnBlockSize = 8,
        xformersAttnConfig: Optional[str] = None,
        xformersBlocksparseLayout: Optional[torch.Tensor] = None, # Probably can be made part of config
        xformersBlocksparseBlocksize: Optional[int] = 16 # Probably can be made part of config
    ):
        super().__init__()
        self.dictionary = dictionary
        
        # See xformersAttnConfig
        xformersAttnConfig = evalStrDict(xformersAttnConfig)
        self.useXformers = xformersAttnConfig is not None
        
        if self.useXformers:
            if not _xformersAvailable:
                raise ImportError("Please install xformers")
            raise NotImplementedError("xformers implementation not done")
        self.embedDim = embedDim
        self.kdim = kdim or embedDim
        self.vdim = vdim or embedDim
        self.qkvSameDim = self.kdim == embedDim and self.vdim == embedDim
        
        self.numHeads = numHeads
        self.dropoutModule = BaseDropout(
            dropout,
            moduleName = self.__class__.__name__
        )
        
        self.headDim = embedDim // numHeads
        assert (
            self.headDim * numHeads == self.embedDim
        ), f"embedDim ({self.embedDim}) must be divisible by numHeads ({numHeads})"
        
        self.scaling: float = self.headDim ** -0.5
        
        self.selfAttention = selfAttention
        self.encoderDecoderAttention = encoderDecoderAttention
        
        assert not self.selfAttention or self.qkvSameDim, (
            "Self-Attention requires q, k and v to be of the same size"
        )
        
        self.kProj = quantNoise(
            torch.nn.Linear(self.kdim, embedDim, bias = bias),
            qNoise,
            qnBlockSize
        )
        
        self.vProj = quantNoise(
            torch.nn.Linear(self.vdim, embedDim, bias = bias),
            qNoise,
            qnBlockSize   
        )
        
        self.qProj = quantNoise(
            torch.nn.Linear(embedDim, embedDim, bias = bias),
            qNoise,
            qnBlockSize
        )
        
        self.outProj = quantNoise(
            torch.nn.Linear(embedDim, embedDim, bias = bias),
            qNoise,
            qnBlockSize
        )
        
        if addBiasKv:
            self.biasK = torch.nn.Parameter(torch.Tensor(1, 1, embedDim))
            self.biasV = torch.nn.Parameter(torch.Tensor(1, 1, embedDim))
        else:
            self.biasK = self.biasV = None
            
        self.addZeroAttn = addZeroAttn
        self.beamSize = 1
        # self.resetParameters()
        
        if self.useXformers:
            xformersAttnConfig["dropout"] = xformersAttnConfig.get("dropout", dropout)
            xformersAttnConfig["numHeads"] = xformersAttnConfig.get("numHeads", numHeads)
            
            if xformersBlocksparseLayout is not None:
                xformersAttnConfig["blockSize"] = xformersBlocksparseBlocksize
                xformersAttnConfig["layout"] = xformersBlocksparseLayout
                xformersAttnConfig["name"] = "blocksparse"
            
            self.attention = build_attention(xformersAttnConfig)
            
        self.onnxTrace = False
        self.skipEmbedDimCheck = False
        self.initIncrementalState()
        
    def _padMasks(
        self,
        keyPaddingMask: Optional[torch.Tensor],
        attnMask: Optional[torch.Tensor]
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if attnMask is not None:
            shape = attnMask.size()[:-1] + torch.Size([1])
            attnMask = torch.cat([attnMask, attnMask.new_zeros(shape)], dim = -1)
        if keyPaddingMask is not None:
            shape = keyPaddingMask.size()[:-1] + torch.Size([1])
            keyPaddingMask = torch.cat([
                keyPaddingMask,
                keyPaddingMask.new_zeros(shape),
            ], dim = -1)
        
        return keyPaddingMask, attnMask
    
    def _addBias(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor],
        attnMask: Optional[torch.Tensor],
        bsz: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        assert self.biasK is not None
        assert self.biasV is not None
        k = torch.cat([k, self.biasK.repeat(1, bsz, 1)])
        v = torch.cat([v, self.biasV.repeat(1, bsz, 1)])
        keyPaddingMask, attnMask = self._padMasks(
            keyPaddingMask = keyPaddingMask,
            attnMask = attnMask,
        )
        
        return k, v, keyPaddingMask, attnMask
    
    def _appendZeroAttn(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        keyPaddingMask: Optional[torch.Tensor],
        attnMask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        zeroAttnShape = k.size()[:-2] + torch.Size([1]) + k.size()[-1:]
        k = torch.cat([k, torch.zeros(zeroAttnShape, dtype = k.dtype, device = k.device)], dim = -2)
        v = torch.cat([v, torch.zeros(zeroAttnShape, dtype = v.dtype, device = v.device)], dim = -2)
        
        keyPaddingMask, attnMask = self._padMasks(
            keyPaddingMask = keyPaddingMask,
            attnMask = attnMask
        )
        
        return k, v ,keyPaddingMask, attnMask
    
    def forward(
        self,
        query: torch.Tensor,
        key: Optional[torch.Tensor],
        value: Optional[torch.Tensor],
        keyPaddingMask: Optional[torch.Tensor] = None,
        incrementalState: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]] = None,
        needWeights: bool = True,
        staticKv: bool = False,
        attnMask: Optional[torch.Tensor] = None,
        beforeSoftmax: bool = False,
        needHeadWeights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if needHeadWeights:
            needWeights = True
        
        tgtLen, bsz, embedDim = query.size()
        srcLen = tgtLen
        
        if not self.skipEmbedDimCheck:
            assert embedDim == self.embedDim, f"query dim ({embedDim}) != {self.embedDim}"
        
        if key is not None:
            srcLen, keyBsz, _ = key.size()
            if not torch.jit.is_scripting():
                assert value is not None
                assert srcLen, keyBsz == value.shape[:2]
                
        if (
            not self.onnxTrace
            and incrementalState is None
            and not staticKv
            and not torch.jit.is_scripting()
            and not self.skipEmbedDimCheck
        ):
            assert key is not None and value is not None
            
            return torch.nn.functional.multi_head_attention_forward(
                query,
                key,
                value,
                self.embedDim,
                self.numHeads,
                torch.empty([0]),
                torch.cat((self.qProj.bias, self.kProj.bias, self.vProj.bias)),
                self.biasK,
                self.biasV,
                self.addZeroAttn,
                self.dropoutModule.p,
                self.outProj.weight,
                self.outProj.bias,
                self.training or self.dropoutModule.applyDuringInference,
                keyPaddingMask.bool() if keyPaddingMask is not None else None,
                needWeights,
                attnMask,
                use_separate_proj_weight = True,
                q_proj_weight = self.qProj.weight,
                k_proj_weight = self.kProj.weight,
                v_proj_weight = self.vProj.weight
            )
            
        
        if incrementalState is not None:
            savedState = self._getInputBuffer(incrementalState)
            if savedState is not None and "prevKey" in savedState:
                if staticKv:
                    assert self.encoderDecoderAttention and not self.selfAttention
                    key = value = None
        else:
            savedState = None
            
        if self.selfAttention:
            q = self.qProj(query)
            k = self.kProj(query)
            v = self.vProj(query)
        elif self.encoderDecoderAttention:
            q = self.qProj(query)
            
            if key is None:
                assert value is None
                k = v = None
            else:
                if self.beamSize > 1 and bsz == key.size(1):
                    key = key.view(key.size(0), -1, self.beamSize, key.size(2))[:, :, 0, :]
                    if keyPaddingMask is not None:
                        keyPaddingMask = keyPaddingMask.view(
                            -1, self.beamSize, keyPaddingMask.size(1)
                        )[:, 0, :]
                k = self.kProj(key)
                v = self.vProj(key)
        else:
            assert key is not None and value is not None
            q = self.qProj(query)
            k = self.kProj(key)
            v = self.vProj(value)
        
        q *= self.scaling
        
        if self.biasK is not None:
            assert self.biasV is not None
            k, v, attnMask, keyPaddingMask = self._addBias(
                k, v, attnMask, keyPaddingMask, bsz
            )
            
        q = (
            q.contiguous()
            .view(tgtLen, bsz * self.numHeads, self.headDim)
            .transpose(0, 1)
        )
        
        kvBsz = bsz # Default value required for scripting
        if k is not None:
            kvBsz = k.size(1)
            k = (
                k.contiguous()
                .view(-1, kvBsz * self.numHeads, self.headDim)
                .transpose(0, 1)
            )
        if v is not None:
            v = (
                v.contiguous()
                .view(-1, kvBsz * self.numHeads, self.headDim)
                .transpose(0, 1)
            )
        if savedState is not None:
            if "prevKey" in savedState:
                _prevKey = savedState["prevKey"]
                assert _prevKey is not None
                kvBsz = _prevKey.size(0)
                prevKey = _prevKey.view(kvBsz * self.numHeads, -1, self.headDim)
                if staticKv:
                    k = prevKey
                else:
                    assert k is not None
                    k = torch.cat([prevKey, k], dim = 1)
                srcLen = k.size(1)
            if "prevValue" in savedState:
                _prevValue = savedState["prevValue"]
                assert _prevValue is not None
                assert kvBsz == _prevValue.size(0)
                prevValue = _prevValue.view(kvBsz * self.numHeads, -1, self.headDim)
                if staticKv:
                    v = prevValue
                else:
                    assert v is not None
                    v = torch.cat([prevValue, v], dim = 1)
            prevKeyPaddingMask: Optional[torch.Tensor] = None
            if "prevKeyPaddingMask" in savedState:
                prevKeyPaddingMask = savedState["prevKeyPaddingMask"]
            assert k is not None and v is not None
            keyPaddingMask = MultiheadAttention._appendPrevKeyPaddingMask(
                keyPaddingMask = keyPaddingMask,
                prevKeyPaddingMask = prevKeyPaddingMask,
                batchSize = kvBsz,
                srcLen = k.size(1),
                staticKv = staticKv,
            )
            
            savedState["prevKey"] = k.view(kvBsz, self.numHeads, -1, self.headDim)
            savedState["prevValue"] = v.view(kvBsz, self.numHeads, -1, self.headDim)
            savedState["prevKeyPaddingMask"] = keyPaddingMask
            assert incrementalState is not None
            incrementalState = self._setInputBuffer(incrementalState, savedState)
        assert k is not None
        assert k.size(1) == srcLen
        if keyPaddingMask is not None and keyPaddingMask.dim() == 0:
            keyPaddingMask = None
        
        if keyPaddingMask is not None:
            assert keyPaddingMask.size(0) == kvBsz
            assert keyPaddingMask.size(1) == srcLen
        
        if self.addZeroAttn:
            assert v is not None
            srcLen += 1
            k, v, keyPaddingMask, attnMask = self._appendZeroAttn(
                k = k,
                v = v,
                keyPaddingMask = keyPaddingMask,
                attnMask = attnMask
            )
            
        if self.encoderDecoderAttention and bsz != kvBsz:
            attnWeights = torch.einsum(
                "bxhtd,bhsd->bxhts",
                q.view((kvBsz, -1, self.numHeads) + q.size()[1:]),
                k.view((kvBsz, self.numHeads) + k.size()[1:])
            )
            attnWeights = attnWeights.reshape((-1, ) + attnWeights.size()[-2:])
        else:
            attnWeights = torch.bmm(q, k.transpose(1, 2))
        attnWeights = self.applySparseMask(attnWeights, tgtLen, srcLen, bsz)
        
        if attnMask is not None:
            attnMask = attnMask.unsqueeze(0)
            if self.onnxTrace:
                attnMask = attnMask.repeat(attnWeights.size(0), 1, 1)
            attnWeights  += attnMask
            
        if keyPaddingMask is not None:
            attnWeights = attnWeights.view(kvBsz, -1, self.numHeads, tgtLen, srcLen)
            attnWeights = attnWeights.masked_fill(
                keyPaddingMask.unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .to(torch.bool),
                float("-inf")
            )
            attnWeights = attnWeights.view(bsz * self.numHeads, tgtLen, srcLen)
            
        if beforeSoftmax:
            return attnWeights, v
        
        attnWeightsFloat = torch.nn.functional.softmax(
            attnWeights, dim = -1
        )
        
        attnWeights = attnWeightsFloat.type_as(attnWeights)
        attnProbs = self.dropoutModule(attnWeights)
        
        assert v is not None
        attn: Optional[torch.Tensor] = None
        if self.encoderDecoderAttention and bsz != kvBsz:
            attn = torch.einsum(
                "bxhts,bhsd->bxhtd",
                attnProbs.view((kvBsz, -1, self.numHeads) + attnProbs.size()[1:]),
                v.view((kvBsz, self.numHeads), v.size()[1:])
            )
            attn = attn.reshape((-1, ) + attn.size()[-2:])
        else:
            attn = torch.bmm(attnProbs, v)
        
        attn = attn.transpose(0, 1).contiguous().view(tgtLen, bsz, self.embedDim)
        attn = self.outProj(attn)
        attnWeights: Optional[torch.Tensor] = None
        
        if needWeights:
            attnWeights = attnWeightsFloat.view(
                bsz, self.numHeads, tgtLen, srcLen
            ).transpose(1, 0)
            
            if not needHeadWeights:
                attnWeights = attnWeights.mean(dim = 0)
                
        return attn, attnWeights
    
    @staticmethod
    def _appendPrevKeyPaddingMask(
        keyPaddingMask: Optional[torch.Tensor],
        prevKeyPaddingMask: Optional[torch.Tensor],
        batchSize: int,
        srcLen: int,
        staticKv: bool
    ) -> Optional[torch.Tensor]:
        if prevKeyPaddingMask is not None and staticKv:
            newKeyPaddingMask = prevKeyPaddingMask
        elif prevKeyPaddingMask is not None and keyPaddingMask is not None:
            newKeyPaddingMask = torch.cat(
                [prevKeyPaddingMask.float(), keyPaddingMask.float()], dim = 1
            )
        elif prevKeyPaddingMask is not None:
            if srcLen > prevKeyPaddingMask.size(1):
                filler = torch.zeros(
                    (batchSize, srcLen - prevKeyPaddingMask.size(1)),
                    device = prevKeyPaddingMask.device
                )
                newKeyPaddingMask = torch.cat(
                    [prevKeyPaddingMask.float(), filler.float()], dim = 1
                )
            else:
                newKeyPaddingMask = prevKeyPaddingMask.float()
        elif keyPaddingMask is not None:
            if srcLen > keyPaddingMask.size(1):
                filler = torch.zeros(
                    (batchSize, srcLen - keyPaddingMask.size(1)),
                    device = keyPaddingMask.device
                )
                newKeyPaddingMask = torch.cat(
                    [filler.float(), keyPaddingMask.float()], dim = 1
                )
            else:
                newKeyPaddingMask = keyPaddingMask.float()
        else:
            newKeyPaddingMask = prevKeyPaddingMask
        return newKeyPaddingMask
    
    def _getInputBuffer(
        self,
        incrementalState: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]],
    ) -> Dict[str, Optional[torch.Tensor]]:
        result = self.getIncrementalState(incrementalState, "attnState")
        if result is not None:
            return result
        else:
            emptyResult: Dict[str, Optional[torch.Tensor]] = {}
            return emptyResult
    
    def _setInputBuffer(
        self,
        incrementalState: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]],
        buffer: Dict[str, Optional[torch.Tensor]],
    ):
        return self.setIncrementalState(incrementalState, "attnState", buffer)
    
    def applySparseMask(self, attnWeights, tgtLen: int, srcLen: int, bsz: int):
        return attnWeights
    
    def initIncrementalState(self):
        self._incrementalStateId = str(uuid.uuid4())
        
    def _getFullIncrementalStateKey(self, key: str) -> str:
        return f"{self._incrementalStateId}.{key}"
    
    def getIncrementalState(
        self,
        incrementalState: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]],
        key: str
    ) -> Optional[Dict[str, Optional[torch.Tensor]]]:
        fullKey = self._getFullIncrementalStateKey(key)
        if incrementalState is None or fullKey not in incrementalState:
            return None
        return incrementalState[fullKey]
    
    def setIncrementalState(
        self,
        incrementalState: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]],
        key: str,
        value: Dict[str, Optional[torch.Tensor]],
    ) -> Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]]:
        if incrementalState is not None:
            fullKey = self._getFullIncrementalStateKey(key)
            incrementalState[fullKey] = value
        return incrementalState