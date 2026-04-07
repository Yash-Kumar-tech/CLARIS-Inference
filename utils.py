from collections import Counter
import os
import re
import torch
from typing import BinaryIO, Dict, List, Optional, Set, Union
import argparse
from tqdm import tqdm

SPACE_NORMALIZER = re.compile(r"\s+")

def makePositions(
    tensor: torch.Tensor,
    paddingIdx: int,
    onnxTrace: bool = False
):
    mask = tensor.ne(paddingIdx).int()
    return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + paddingIdx

def getActivationFn(activation: str):
    if activation == "relu":
        return torch.nn.functional.relu
    elif activation == "relu_squared":
        return lambda x: torch.nn.functional.relu(x).pow(2)
    elif activation == "tanh":
        return torch.tanh
    elif activation == "linear":
        return lambda x: x
    elif activation == "swish":
        return torch.nn.SiLU
    else:
        raise NotImplementedError(f"Activation fn {activation} not implemented yet")
    
def lengthsToPaddingMask(lens):
    bsz, maxLens = lens.size(0), torch.max(lens).item()
    
    mask = torch.arange(maxLens).to(lens.device).view(1, maxLens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, maxLens)
    return mask

def fillWithNegInf(t: torch.Tensor):
    return t.float().fill_(float("-inf")).type_as(t)

def tokenizeLine(line: str) -> List[str]:
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()

class Dictionary:
    def __init__(
        self,
        *,
        bos: str = "<s>",
        pad: str = "<pad>",
        eos: str = "</s>",
        unk: str = "<unk>",
        extraSpecialSymbols: Optional[List[str]] = None,
        addSpecialSymbols: bool = True
    ):
        self.bosWord, self.unkWord, self.padWord, self.eosWord = bos, unk, pad, eos
        self.symbols = []
        self.count = []
        self.indices = {}
        
        if addSpecialSymbols:
            self.bosIndex: int = self.addSymbol(bos)
            self.padIndex: int = self.addSymbol(pad)
            self.eosIndex: int = self.addSymbol(eos)
            self.unkIndex: int = self.addSymbol(unk)
            
            if extraSpecialSymbols:
                for s in extraSpecialSymbols:
                    self.addSymbol(s)
            self.nSpecial = len(self.symbols)
            
    def __eq__(self, other: "Dictionary"):
        return self.indices == other.indices
    
    def __getitem__(self, idx):
        if idx < len(self.symbols):
            return self.symbols[idx]
        
        return self.unkWord
    
    def getCount(self, idx):
        return self.count[idx]
    
    def __len__(self):
        return len(self.symbols)
    
    def __contains__(self, sym: str):
        return sym in self.indices
    
    def index(self, sym: str):
        assert isinstance(sym, str)
        if sym in self.indices:
            return self.indices[sym]
        return self.unkIndex
    
    def string(
        self,
        tensor: torch.Tensor,
        bpeSymbol: Optional[str] = None,
        escapeUnk: bool = False,
        extraSymbolsToIgnore: Optional[Set[str]] = None,
        unkString: Optional[str] = None,
        includeEos: bool = False,
        separator: str = " "
    ):
        if torch.is_tensor(tensor) and tensor.dim() == 2:
            return "\n".join(
                self.string(
                    t,
                    bpeSymbol,
                    escapeUnk,
                    extraSymbolsToIgnore,
                    includeEos = includeEos,
                ) for t in tensor
            )
        extraSymbolsToIgnore = set(extraSymbolsToIgnore or [])
        if not includeEos:
            extraSymbolsToIgnore.add(self.eos())
            
        def tokenString(i: int) -> str:
            if i == self.unk():
                return unkString or self.unkString(escapeUnk)
            else:
                return self[i]
        
        extraSymbolsToIgnore.add(self.bosIndex)
        
        sent = separator.join(
            tokenString(i)
            for i in tensor if i not in extraSymbolsToIgnore and i >= 4
        )
        
        return postProcess(sent, bpeSymbol)
    
    def unkString(self, escape: bool = False) -> str:
        if escape:
            return f"<{self.unkWord}>"
        return self.unkWord
    
    def addSymbol(self, word: string, n: int = 1, overwrite: bool = False) -> int:
        if word in self.indices and not overwrite:
            idx = self.indices[word]
            self.count[idx] = self.count[idx] + n
            return idx
        else:
            idx = len(self.symbols)
            self.indices[word] = idx
            self.symbols.append(word)
            self.count.append(n)
            return idx
        
    def update(self, newDict: "Dictionary"):
        for word in newDict.symbols:
            idx2 = newDict.indices[word]
            if word in self.indices:
                idx = self.indices[word]
                self.count[idx] = self.count[idx] + newDict.count[idx]
            else:
                idx = len(self.symbols)
                self.indices[word] = idx
                self.symbols.append(word)
                self.count.append(newDict.count[idx2])
    
    def finalize(self, threshold: int = -1, nwords: int = -1, paddingFactor: int = 8):
        if nwords <= 0:
            nwords = len(self)
        newIndices = dict(zip(self.symbols[: self.nspecial], range(self.nSpecial)))
        newSymbols = self.symbols[: self.nSpecial]
        newCount = self.count[: self.nSpecial]
        
        c = Counter(dict(
            sorted(zip(self.symbols[self.nSpecial: ], self.count[self.nSpecial:]))
        ))
        
        for symbol, count in c.most_common(nwords - self.nSpecial):
            if count >= threshold:
                newIndices[symbol] = len(newSymbols)
                newSymbols.append(symbol)
                newCount.append(count)
            else:
                break
        assert len(newSymbols) == len(newIndices)
        
        self.count = list(newCount)
        self.symbols = list(newSymbols)
        self.indices = newIndices
        
        self.padToMultiple_(paddingFactor)
        
    def padToMultiple_(self, paddingFactor: int):
        if paddingFactor > 1:
            i = 0
            while len(self) % paddingFactor != 0:
                symbol = f"madeupword{i:04d}"
                self.addSymbol(symbol, n = 0)
                i += 1
                
    def bos(self) -> int:
        return self.bosIndex
    
    def pad(self) -> int:
        return self.padIndex
    
    def eos(self) -> int:
        return self.eosIndex
    
    def unk(self) -> int:
        return self.unkIndex
    
    @classmethod
    def load(cls, f, addSpecialSymbols = True):
        d = cls(addSpecialSymbols = addSpecialSymbols)
        d.addFromFile(f)
        return d
    
    def addFromFile(self, f: Union[str, BinaryIO]):
        if isinstance(f, str):
            try:
                with open(f, "r", encoding ="utf-8") as fd:
                    self.addFromFile(fd)
            except FileNotFoundError as fnfe:
                return fnfe
            except UnicodeError:
                raise Exception(
                    f"Incorrect encoding detected in {f}"
                )
            return
        lines = f.readlines()
        indicesStartLines = self._loadMeta(lines)
        
        for line in lines[indicesStartLines:]:
            try:
                line, field = line.rstrip().rstrip(" ", 1)
                if field == "#fairseq:overwrite":
                    overwrite = True
                    line, field = line.rsplit(" ", 1)
                else:
                    overwrite = False
                count = int(field)
                word = line
                if word in self and not overwrite:
                    raise RuntimeError(
                        f"Duplicate word found in dictionary: {word}"
                    )
                self.addSymbol(word, n = count, overwrite = overwrite)
                
            except ValueError:
                raise ValueError(
                    f"Incorrect dictionary format, expected '<token> <cnt> [flags]': \"{line}\""
                )
                
    def _save(self, f: Union[str, BinaryIO], kvIterator):
        if isinstance(f, str):
            os.makedirs(os.path.dirname(f))
            with open(f, "w", encoding = "utf-8") as fd:
                return self.save(fd)
        for k, v in kvIterator:
            print(f"{k} {v}", file = f)
            
    def _getMeta(self):
        return [], []
    
    def _loadMeta(self, lines):
        return 0
    
    def save(self, f):
        exKeys, exVals = self._getMeta()
        self._save(
            f,
            zip(
                exKeys + self.symbols[self.nSpecial:],
                exVals + self.count[self.nSpecial:]
            )
        )
        
    def dummySentence(self, length) -> torch.Tensor:
        t = torch.Tensor(length).uniform_(self.nSpecial + 1, len(self)).long()
        t[-1] = self.eos()
        return t
    
    def encodeLine(
        self,
        line: str,
        lineTokenizer = tokenizeLine,
        addIfNotExist: bool = True,
        consumer = None,
        appendEos = True,
        reverseOrder = False
    ) -> torch.IntTensor:
        words = lineTokenizer(line)
        if reverseOrder:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if appendEos else nwords)
        
        for i, word in enumerate(words):
            if addIfNotExist:
                idx = self.addSymbol(word)
            else:
                idx = self.index(word)
            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        
        if appendEos:
            ids[nwords] = self.eosIndex
            
        return ids
    
def postProcess(sentence: str, symbol: str) -> str:
    if symbol == "sentencepiece":
        sentence = sentence.replace(" ", "").replace("\u2581", " ").strip()
    elif symbol == "wordpiece":
        sentence = sentence.replace(" ", "").replace("_", " ").strip()
    elif symbol == "letter":
        sentence = sentence.replace(" ", "").replace("|", " ").strip()
    elif symbol == "silence":
        import re

        sentence = sentence.replace("<SIL>", "")
        sentence = re.sub(" +", " ", sentence).strip()
    elif symbol == "_EOW":
        sentence = sentence.replace(" ", "").replace("_EOW", " ").strip()
    elif symbol in {"subword_nmt", "@@ ", "@@"}:
        if symbol == "subword_nmt":
            symbol = "@@ "
        sentence = (sentence + " ").replace(symbol, "").rstrip()
    elif symbol == "none":
        pass
    elif symbol is not None:
        raise NotImplementedError(f"Unknown post_process option: {symbol}")
    return sentence

def replaceUnk(
    hypoStr,
    srcStr,
    alignment,
    alignDict,
    unk
):
    hypoTokens = tokenizeLine(hypoStr)
    srcTokens = tokenizeLine(srcStr) + ['<eos>']
    
    for i, ht in enumerate(hypoTokens):
        if ht == unk:
            srcToken = srcTokens[alignment[i]]
            hypoTokens[i] = alignDict.get(srcToken, srcToken)
    return " ".join(hypoTokens)

def postProcessPrediction(
    hypoTokens: torch.Tensor,
    srcStr,
    alignment,
    alignDict,
    tgtDict: Dictionary,
    removeBpe: Optional[str] = None,
    extraSymbolsToIgnore = None
):
    hypoStr = tgtDict.string(
        hypoTokens,
        removeBpe,
        extraSymbolsToIgnore = extraSymbolsToIgnore
    )
    hypoStr = replaceUnk(
        hypoStr,
        srcStr,
        alignment,
        alignDict,
        tgtDict.unkString(),
    )
    return hypoTokens, hypoStr
    
def convertKeys(name: str) -> str:
    # Insert underscore before uppercase letters and lowercase them
    return ''.join([f"_{ch.lower()}" if ch.isupper() else ch for ch in name])
    
def getModelStateDictFromPath(
    checkpointPath: str,
    modelStateDict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    newDict = {}
    stDict = torch.load(
        checkpointPath,
        weights_only = False
    )['model']
    
    for k in tqdm(modelStateDict.keys()):
        newName = convertKeys(k)
        
        if newName not in stDict:
            raise KeyError(f"Key {newName} not found in checkpoint for {k}")
        if modelStateDict[k].shape != stDict[newName].shape:
            raise ValueError(f"Shape mismatch for {k}: {modelStateDict[k].shape} vs {stDict[newName].shape}")
        
        newDict[k] = stDict[newName]
        
    return newDict

def getVocoderStateDictFromPath(
    checkpointPath: str,
    modelStateDict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    newDict = {}
    stDict = torch.load(checkpointPath, map_location = 'cpu', weights_only = False)['generator']

    for k in modelStateDict.keys():
        newName = convertKeys(k)
        
        if newName not in stDict: # This condition is not important but kept for safety
            raise KeyError(f"Key {newName} not found in checkpoint for {k}")
        if modelStateDict[k].shape != stDict[newName].shape:
            raise ValueError(f"Shape mismatch for {k}: {modelStateDict[k].shape} vs {stDict[newName].shape}")
        
        newDict[k] = stDict[newName]
    return newDict

def evalStrDict(x):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    return x

def getSymbolsToStripFromOutput(generator):
    return generator.symbolsToStripFromOutput or { generator.eos }

def stripPad(tensor, pad):
    return tensor[tensor.ne(pad)]

def makeVocoderParser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--inCodeFile',
        type = str,
        required = True,
        help = "Path to unit file",
    )
    parser.add_argument(
        '--vocoder',
        type = str,
        required = True,
        help = "path to the CodeHiFiGAN vocoder",
    )
    parser.add_argument(
        '--resultsPath',
        type = str,
        required = True
    )
    parser.add_argument(
        '--durPrediction',
        action = "store_true",
        help = "Enable duration prediction (for reduced/unique code sequences)",
    )
    parser.add_argument('--cpu', action = "store_true", help = "run on CPU")
    return parser

