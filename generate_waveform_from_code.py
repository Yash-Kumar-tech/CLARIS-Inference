import os
import torch
from pathlib import Path
from typing import Union, List, Dict
import soundfile as sf

from params import VocoderArgs, CodeGeneratorParams
from vocoders.code_hifigan import CodeHiFiGANVocoder
from utils import makeVocoderParser

def loadCode(path: Union[Path, str]) -> Dict[str, List[int]]:
    results: Dict[str, List[int]] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue 
            filename, *units = line.split()
            results[filename] = list(map(int, units))
    return results

def dumpResult(
    args: VocoderArgs,
    filename: str,
    predWav: torch.Tensor,
):
    sf.write(
        os.path.join(args.resultsPath, filename),
        predWav.detach().cpu().numpy(),
        16_000,
    )

def main():
    parser = makeVocoderParser()
    args: VocoderArgs = parser.parse_args()
    
    useCuda = torch.cuda.is_available() and not args.cpu
    
    vocoderParams = CodeGeneratorParams()
    vocoder = CodeHiFiGANVocoder(args.vocoder, vocoderParams)

    if useCuda:
        vocoder = vocoder.cuda()
        
    data = loadCode(args.inCodeFile)
    Path(args.resultsPath).mkdir(exist_ok = True, parents = True)
    
    for filename, code in data.items():
        codeTensor = torch.LongTensor(code).view(1, -1)
        wav = vocoder(code = codeTensor, durPrediction = args.durPrediction)
        dumpResult(args, filename, wav)
  
if __name__ == "__main__":
    main()