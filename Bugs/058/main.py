from skimage.io import imread
from skimage.transform import resize
import imgaug.augmenters as iaa

file_name = "sample.jpeg"
resized_img = resize(imread(file_name), (224, 224))

aug = iaa.AdditiveGaussianNoise(scale=(0, 0.2*255))
augmented_image = aug(resized_img)