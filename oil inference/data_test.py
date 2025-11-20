import numpy as np
from PIL import Image
import glob

# # Load some training images
# train_images = glob.glob('data/0/*.jpg')[:100]  # Sample 100

# values = []
# for img_path in train_images:
#     img = np.array(Image.open(img_path))
#     values.extend(img.flatten())

# print(f"Training data range: [{np.min(values)}, {np.max(values)}]")
# print(f"Training data mean: {np.mean(values):.2f}")
# print(f"Training data std: {np.std(values):.2f}")

# img = Image.open('data/0/0_0_0_img_0bNt4tmcxCvG6liW_SFr_cls_0.jpg')
# print(f"Training image shape: {np.array(img).shape}")
# # Expected: (400, 400) for grayscale or (400, 400, 3) for RGB

sample = np.array(Image.open('sample_chip_inference_9900.jpg'))
print(sample)