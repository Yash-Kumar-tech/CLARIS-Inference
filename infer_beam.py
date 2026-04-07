import torch
from model import SpeechToUnitTransformer
from params import ModelParams
from tqdm import tqdm

from data.dataset import Dataset
from sequence_generator import SequenceGenerator
from utils import (
    Dictionary, 
    getSymbolsToStripFromOutput,
    getModelStateDictFromPath, 
    postProcessPrediction, 
    stripPad
)

tgtDict = Dictionary()
for i in range(1000):
    # 1000 => target code size, can be changed for new models
    tgtDict.addSymbol(str(i))
    
# Build model
modelParams = ModelParams()
model = SpeechToUnitTransformer.buildModel(
    params = modelParams,
    tgtDict=tgtDict,
)

modelStDict = model.state_dict()

model.load_state_dict(getModelStateDictFromPath(
    './checkpoint.pt',
    modelStDict
))

model.eval()
generator = SequenceGenerator(
    model = model,
    tgtDict = tgtDict,
    beamSize = 20,
    maxLenA = 1,
    maxLen = 50_000, # Can be reduced to a much smaller value
)

ds = Dataset(
    root = "./whisper/",
    nFramesPerStep = 1,
)

dataLoader = torch.utils.data.DataLoader(
    dataset = ds,
    batch_size = 1,
    shuffle = False,
    collate_fn = ds.collater
)

out_path = "predictions_whisper.unit"
with open(out_path, "w", encoding="utf-8") as out_file:
    for sample in tqdm(dataLoader):
        hypos = generator.generate(
            srcTokens = sample['netInput']['srcTokens'],
            srcLengths = sample['netInput']['srcLengths'],
            ids = sample['id']
        )
        
        numGeneratedTokens = sum(len(h[0]["tokens"]) for h in hypos)
        
        for i, sampleId in enumerate(sample["id"].tolist()):
            srcTokens = stripPad(sample["netInput"]["srcTokens"], tgtDict.pad())
            srcStr = ""
            filename = sample["filenames"][i]
            for j, hypo in enumerate(hypos[i][:modelParams.generation.nbest]):
                hypoTokens, hypoStr = postProcessPrediction(
                    hypoTokens = hypo['tokens'].int().cpu(),
                    srcStr = srcStr,
                    alignment = hypo['alignment'],
                    alignDict = None,
                    tgtDict = tgtDict,
                    removeBpe = None,
                    extraSymbolsToIgnore = getSymbolsToStripFromOutput(generator),
                )
            
                units = hypoTokens.tolist()
                # Write filename + units to file
                out_file.write(f"{filename} {hypoStr}\n")

print(f"Units written to {out_path}")
