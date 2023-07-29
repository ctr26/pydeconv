# %%
from skimage.transform import rescale, resize, downscale_local_mean
from model import ResNetUNet
import torch
from skimage import data
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from scipy import special
from torchvision import transforms
from skimage import metrics
import os
from pathlib import Path

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl
import torch
# Note - you must have torchvision installed for this example
from torch.utils.data import DataLoader
import os
from pytorch_lightning import loggers as pl_loggers
from torchvision import transforms

from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import glob
# Note - you must have torchvision installed for this example
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from skimage.measure import regionprops
from torchvision.transforms.functional import crop
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from pytorch_lightning import loggers as pl_loggers
import torchvision
from sklearn.manifold import MDS
from sklearn.metrics.pairwise import euclidean_distances
from scipy.ndimage import convolve, sobel
from skimage.measure import find_contours
from scipy.interpolate import interp1d
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms, datasets, models


class BinaryGenerater(Dataset):
    def __init__(self, image, transform):
        self.image = image
        self.transform = transform

    def __len__(self):
        N = int(np.prod(self.image.shape))
        # return int(special.comb(N,2))
        return 200
        # return 10

    def __getitem__(self, idx):
        np.random.seed(idx)
        img_T = np.random.binomial(self.image, 0.5)
        img_V = self.image - img_T
        return (self.transform(img_T), self.transform(img_V))


class LitModel(pl.LightningModule):
    def __init__(self, model, batch_size=1, learning_rate=1e-4, params=None, gt=None):
        super().__init__()
        # self.autoencoder = AutoEncoder(batch_size, 1)
        self.model = model
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.MSELoss()
        self.params = params
        self.gt = gt
        # self.loss_fn = torch.nn.BCEWithLogitsLoss()
        # self.vae = VAE()
        # self.vae_flag = vae_flag
        # self.loss_fn = torch.nn.BCELoss()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.learning_rate)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        # self.curr_device = real_img.device
        img_T, img_V = batch
        output = self.forward(img_V)
        loss_T = self.loss_fn(img_T, output)
        loss_V = self.loss_fn(img_V, output)
        loss = loss_T
        gt_np = np.array(self.gt)
        output_np = np.array(output[0].cpu().detach())
        psnr = metrics.peak_signal_noise_ratio(gt_np, output_np)
        # loss = train_loss['loss']
        self.log("train_loss", loss)
        # tensorboard = self.logger.experiment
        self.log("psnr", psnr)
        self.log("loss_T", loss_T)
        self.log("loss_V", loss_V)

        # self.logger.experiment.add_scalar("Loss/train", loss, batch_idx)
        # self.logger.experiment.add_scalar("psnr", psnr, batch_idx)
        # self.logger.experiment.add_scalar("loss_T", loss_T, batch_idx)
        # self.logger.experiment.add_scalar("loss_V", loss_V, batch_idx)
        # torchvision.utils.make_grid(output)
        self.logger.experiment.add_image(
            "1_output", torchvision.utils.make_grid(output), batch_idx)
        self.logger.experiment.add_image(
            "img_V", torchvision.utils.make_grid(img_V), batch_idx)
        self.logger.experiment.add_image(
            "img_T", torchvision.utils.make_grid(img_T), batch_idx)

        # self.logger.experiment.add_embedding(
        #     "input_image", torchvision.utils.make_grid(transformer_image(inputs)), batch_idx)
        # self.logger.experiment.add_image(
        # "output", torchvision.utils.make_grid(self.model.output_from_results(*results)), batch_idx)
        # self.logger.experiment.add_embedding(
        #     "output_image", torchvision.utils.make_grid(transformer_image(output)), batch_idx)

        return loss


from skimage.util import crop
data_type = "SEM"

if (data_type == "astro"):
    work_dir = "data/astro"
    model_dir = work_dir+"model"


    image_rgb = data.astronaut()
    image_rgb_small = downscale_local_mean(
        image_rgb, (1, 1, 1)).astype('int32')
    image_rgb_small = image_rgb
    # image_rgb = resize(image,(256, 256,3)).astype('int32')

    grayscale = rescale(rgb2gray(image_rgb), 0.5)*255
    noise_small = np.random.poisson(grayscale).astype('int32')

    noise_rgb_small = (np.repeat(noise_small[..., np.newaxis], 3, -1))
    noise_rgb_small = np.random.poisson(image_rgb_small)
    
if data_type == "SEM":
    work_dir = "data/SEM"
    model_dir = work_dir+"model"
    grayscale = np.array(Image.open(work_dir+"/"+"validation.tif"))
    grayscale = ((grayscale/grayscale.max())*(2**8-1)).astype('int32')
    noise_rgb_small = (np.repeat(grayscale[..., np.newaxis], 3, -1))
    plt.imshow(noise_rgb_small)
    image_rgb_small = noise_rgb_small
    
    # noise_small = np.array(Image.open(work_dir+"/"+"train.tif"))[:457,457:457+457]
    # noise_rgb_small = (np.repeat(noise_small[..., np.newaxis], 3, -1))
# %%
transform = transforms.Compose(
    [
        # transforms.ToPILImage(),
        # transforms.ToTensor(),
        # transforms.Normalize(0, 1),
        # transforms.transforms.Lambda(lambda x :  np.repeat(x[..., np.newaxis], 3, -1)),
        # transforms.transforms.Lambda(lambda x : x.astype(np.uint16)),
        # transforms.ToTensor(),
        transforms.ToTensor(),
        transforms.CenterCrop(512),
        transforms.transforms.Lambda(lambda x: np.divide(x, 255*2)),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        # transforms.ToTensor(),
        # transforms.transforms.Lambda(lambda x : x.astype(np.float32)),
        # transforms.ToPILImage(),
        # transforms.ToTensor(),
        # transforms.transforms.Lambda(lambda x : x/255),
        # transforms.ConvertImageDtype(torch.float32),
        # transforms.ToTensor(),
        # transforms.ToPILImage(),
        # transforms.ToTensor(),
        # transforms.ConvertImageDtype(torch.float32),
        # transforms.ConvertImageDtype(torch.float32),
    ]
)
dataset = BinaryGenerater(noise_rgb_small, transform=transform)
a, b = dataset[0]
# %%

dataloader = DataLoader(dataset, batch_size=1,
                        shuffle=True, num_workers=8, pin_memory=True)

test_img = a.unsqueeze(dim=0)
# model(a.unsqueeze())
# %%
model = ResNetUNet(n_class=3)

# model = models.resnet18(pretrained=True)

model(test_img.float())

# %%
model = LitModel(model, gt=transform(image_rgb_small))

model(test_img.float())
# %%

tb_logger = pl_loggers.TensorBoardLogger("runs/")

model_dir = "out"
Path(f'{model_dir}/').mkdir(parents=True, exist_ok=True)

checkpoint_callback = ModelCheckpoint(
    dirpath=f'{model_dir}/',
    save_last=True
)

trainer = pl.Trainer(
    logger=tb_logger,
    enable_checkpointing=True,
    gpus=1,
    accumulate_grad_batches=1,
    callbacks=[checkpoint_callback],
    min_epochs=25,
    max_epochs=25,
)  # .from_argparse_args(args)


# try:
#     trainer.fit(model, dataloader,
#                 ckpt_path=f'{model_dir}/last.ckpt')
# except:
#     trainer.fit(model, dataloader)

trainer.fit(model, dataloader)

# %%
