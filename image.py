import cv2
import numpy as np
import matplotlib.pyplot as plt 
#Load the image 
kpop = cv2.imread('./Kpop_Faces/4minute/hyuna/95790_FBac.jpg')
image_rgb = cv2.cvtColor(kpop,cv2.COLOR_BGR2RGB)
#DEFINE THE SCALE FACTOR 
scale_factors = 3.0
scale_factor_2 = 1/3.0
heights, width = image_rgb.shape[:2]

heights_new = int(heights*scale_factors)
width_new = int(width*scale_factors)

# RESIZE THE IMAGE
zoomed = cv2.resize(src=image_rgb,
                    dsize=(width_new,heights_new),
                    interpolation=cv2.INTER_CUBIC)

# Calculate the new image dimension
new_heights2 = int(heights*scale_factor_2)
new_width2 = int(width*scale_factor_2) 

# Scaled image
scale_image = cv2.resize(src=image_rgb,
                         dsize=(width_new,heights_new),
                         interpolation=cv2.INTER_AREA)

# create subplots
fig, axs = plt.subplots(1,3,figsize=(10,4))

# original image plot
axs[0].imshow(image_rgb)
axs[0].set_title('Original Image Shape:'+str(image_rgb.shape))

# zoomed image plot
axs[1].imshow(zoomed)
axs[1].set_title('Zoomed Image Shape:'+str(zoomed.shape))

# scale image plot
axs[2].imshow(scale_image)
axs[2].set_title('Scaled Image Shape:'+str(scale_image.shape))

# Remove ticks from the subplots
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()