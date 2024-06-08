import fastai
from fastai.data import transforms
from fastai.data.block import DataBlock, TransformBlock
from fastai.data.transforms import get_image_files
from fastai.optimizer import RMSProp
from fastai.vision.data import ImageBlock, ImageDataLoaders
from fastcore.imports import noop
from numpy import negative
import torch
import cv2
import PIL
from torchvision import transforms
from PIL import Image
from torch import nn
from fastai.vision import *
from fastai.vision.augment import *
from fastai.imports import *
from fastai.vision.gan import *
from fastai.data.block import *
from fastai.data.transforms import *
from fastai.callback.all import *
path = Path('pokemon/pokemon')

bs=100
size=64
dblock = DataBlock(blocks = (TransformBlock, ImageBlock),
                   get_x = generate_noise,
                   get_items = get_image_files,
                   splitter = IndexSplitter([]),
                   item_tfms=Resize(size, method=ResizeMethod.Crop), 
                   batch_tfms = Normalize.from_stats(torch.tensor([0.5,0.5,0.5]), torch.tensor([0.5,0.5,0.5])))
dls = dblock.dataloaders(path,path=path,bs=bs)
generator = basic_generator(64,3,n_extra_layers=1)
critic = basic_critic(64, 3, n_extra_layers=1,act_cls=partial(nn.LeakyReLU))
student = GANLearner.wgan(dls,generator,critic,opt_func = RMSProp)
student.recorder.train_metrics=True
student.recorder.valid_metrics=False
student.fit(1,2e-4,wd=0.)
#cv2.waitKey(0)
student.show_results(max_n=9,ds_idx=0)
student.gan_trainer.switch(gen_mode=True)
img = student.predict(generate_noise('pocheman',size=100))
print(img[0].size())
im =transforms.ToPILImage()(img[0]).convert('RGB')