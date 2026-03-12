import os
import random
import matplotlib.pyplot as plt
from PIL import Image

folder = "analysis_output/gradcam"
images = random.sample(os.listdir(folder), 50)

fig, axes = plt.subplots(10, 5, figsize=(20, 20))
for ax, img_name in zip(axes.flat, images):
    ax.imshow(Image.open(os.path.join(folder, img_name)))
    ax.set_title(img_name.replace("_gradcam.jpg", ""), fontsize=7)
    ax.axis("off")

plt.tight_layout()
plt.savefig("gradcam_grid.png", dpi=150, bbox_inches="tight")
plt.show()