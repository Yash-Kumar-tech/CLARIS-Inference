import os
import torch

from data.dataset import Dataset
from model import SpeechToUnitTransformer
from params import ModelParams
from utils import Dictionary, getModelStateDictFromPath, postProcessPrediction

class EncoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, srcTokens, srcLengths):
        encOut, encPaddingMask, _ = self.model.forwardEncoder(
            srcTokens=srcTokens,
            srcLengths=srcLengths
        )
        return encOut, encPaddingMask


class DecoderWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, prevOutputTokens, encoderOut, encoderPaddingMask):
        # For simplicity, incrementalState is empty here
        # Use of incrementalState can help with KV-Cache to speed-up inference
        # But in some cases can cause incorrect outputs 
        # For accuracy, it is recommended to export without incrementalState
        # and maintain the KV-Cache manually through code
        curOut, attn, _ = self.model.forwardDecoder(
            prevOutputTokens=prevOutputTokens,
            encoderOut=encoderOut,
            encoderPaddingMask=encoderPaddingMask,
            incrementalState={}
        )
        return curOut, attn


def export_to_onnx(model, export_dir="onnx_exports"):
    os.makedirs(export_dir, exist_ok=True)

    # Dummy inputs: T=1000 frames, feature dim=80
    T = 1000
    dummy_srcTokens = torch.randn(1, T, 80).float()   # (batch=1, T, 80)
    dummy_srcLengths = torch.tensor([T]).long()       # (batch=1,) with value T
    dummy_prevOutputTokens = torch.randint(0, 1000, (1, 1)).long()

    # Encoder export
    encoder = EncoderWrapper(model)
    torch.onnx.export(
        encoder,
        (dummy_srcTokens, dummy_srcLengths),
        os.path.join(export_dir, "encoder.onnx"),
        input_names=["srcTokens", "srcLengths"],
        output_names=["encoderOut", "encoderPaddingMask"],
        dynamic_axes={
            "srcTokens": {1: "srcLen"},
            "encoderOut": {1: "srcLen"},
            "encoderPaddingMask": {1: "srcLen"},
        },
        opset_version=14,
        do_constant_folding=False,
    )
    print("Exported encoder to ONNX")

    # Run encoder once to get dummy outputs
    encOut, encPaddingMask = encoder(dummy_srcTokens, dummy_srcLengths)

    # Decoder export
    decoder = DecoderWrapper(model)
    torch.onnx.export(
        decoder,
        (dummy_prevOutputTokens, encOut, encPaddingMask),
        os.path.join(export_dir, "decoder.onnx"),
        input_names=["prevOutputTokens", "encoderOut", "encoderPaddingMask"],
        output_names=["decoderOut", "attn"],
        dynamic_axes={
            "prevOutputTokens": {1: "tgtLen"},
            "encoderOut": {1: "srcLen"},
            "encoderPaddingMask": {1: "srcLen"},
            "decoderOut": {1: "tgtLen"}
        },
        opset_version=14
    )
    print("Exported decoder to ONNX")

def main():
    ds = Dataset(
        root="path/to/whisper/files", 
        nFramesPerStep=1,
    )

    batchSize = 1
    dataloader = torch.utils.data.DataLoader(
        dataset=ds,
        batch_size=batchSize,
        shuffle=False,
        collate_fn=ds.collater,
    )

    # Build and load model
    model = SpeechToUnitTransformer.buildModel(
        params=ModelParams(),
        tgtDict={}
    )
    model.load_state_dict(
        getModelStateDictFromPath(
            'path/to/checkpoint.pt',
            model.state_dict(),
        )
    )
    model = model.eval()
    print("Loaded model state dict")

    # Build target dictionary
    tgtDict = Dictionary()
    for i in range(1000):
        tgtDict.addSymbol(str(i))

    # Export to ONNX
    export_to_onnx(model)

    # Run one batch through encoder+decoder (greedy) to test the export
    for batch in dataloader:
        print(f"SrcLengths: {batch['netInput']['srcLengths']}")
        srcTokens: torch.Tensor = batch['netInput']['srcTokens'][0].unsqueeze(0)  # (1, T, 80)
        srcLengths: torch.Tensor = batch['netInput']['srcLengths'][0].unsqueeze(0)  # (1,)
        filename: str = batch['filenames'][0]

        encEmbed = model.forwardEncoder(
            srcTokens=srcTokens,
            srcLengths=srcLengths
        )

        maxLen = 10  # generate up to 10 tokens
        prevOutputTokens = srcLengths.new_zeros((batchSize, 1)).long().fill_(tgtDict.eos())
        incrementalState = {}
        predOut = []

        for _ in range(maxLen):
            curOut, currAttn, _ = model.forwardDecoder(
                prevOutputTokens=prevOutputTokens,
                encoderOut=encEmbed[0],
                encoderPaddingMask=encEmbed[1],
                incrementalState=incrementalState
            )

            lprobs = model.getNormalizedProbs(curOut, logProbs=True)
            print(lprobs.shape)
            curPredLProb, curPredOut = torch.max(lprobs, dim=1)
            predOut.append(curPredOut)
            prevOutputTokens = torch.cat((prevOutputTokens, curPredOut.view(batchSize, 1)), dim=1)

            if torch.any(curPredOut.squeeze(0) == tgtDict.eos()):
                break
        predOut = torch.cat(predOut, dim=0).view(batchSize, -1)
        print(filename)
        print("Out: ", predOut)

        hypoTokens, hypoStr = postProcessPrediction(
            hypoTokens=predOut,
            tgtDict=tgtDict,
            removeBpe=None,
            extraSymbolsToIgnore=set()
        )
        print("Decoded string:", hypoStr)
        # Compare the decoded string with a similar inference loop through the pt checkpoint
        # to ensure the export is correct
        break  # just run one batch


if __name__ == "__main__":
    main()