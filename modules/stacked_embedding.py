import torch

from modules.linear import Linear

class StackedEmbedding(torch.nn.Embedding):
    def __init__(
        self,
        numEmbeddings: int,
        embedDim: int,
        paddingIdx: int,
        numStacked: int = 1,
    ):
        super().__init__(numEmbeddings, embedDim, paddingIdx)
        
        torch.nn.init.normal_(self.weight, mean = 0, std = embedDim ** -0.5)
        torch.nn.init.constant_(self.weight[paddingIdx], 0)
        
        self.offset = 4 # Skip <bos>, <pad>, <eos>, <unk>
        
        self.vocabSize = numEmbeddings - self.offset
        self.numStacked = numStacked
        
        if self.numStacked > 1:
            self.projectInDim = Linear(embedDim * numStacked, embedDim, bias = False)
            
    def forward(self, input: torch.Tensor):
        if self.numStacked == 1:
            return super().forward(input)
        raise NotImplementedError(f"Num Stacked > 1 not supported yet, you provided {self.numStacked}")