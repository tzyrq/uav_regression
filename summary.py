from torchsummary import summary
from seg_static import seg_static
from model import mainnet
import torch

if __name__ == "__main__":
    m = mainnet()
    #m = seg_static()
    summary(m, [(60, 100, 100), (1, 100, 100)], device="cpu")

