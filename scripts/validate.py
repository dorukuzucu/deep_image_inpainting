from datasets.inpainting import BsdDataset
from models.context_encoder import RedNet
from scripts.train import load_model
import torch
from PIL import Image
import numpy as np


def save_np(arr, name):
    x_np = arr.permute(1, 2, 0).numpy() * 255
    im = Image.fromarray(x_np.astype(np.uint8))
    im.save(name+".png")


mdl = RedNet(20, 3)
sd = torch.load(r"C:\Users\ABRA\PycharmProjects\WaveletInpainting\output\epoch99.pth")
mdl.load_state_dict(sd)
mdl.eval()


ds_p = r"D:\datasets\bsd500\images\test"
ds = BsdDataset(ds_p, size=(224, 224))
mask, label = ds[5]
with torch.no_grad():
    out = mdl(mask.unsqueeze(0))[0]

save_np(mask, "inp")
save_np(out, "out")