import platform
import torch
import cv2
import numpy as np
import matplotlib
import pandas
import sklearn
import plotly
import imageio
import lpips
import PIL
import yaml

print("=" * 70)
print("FedNeRF-Privacy3D Environment Verification")
print("=" * 70)

print(f"Python      : {platform.python_version()}")
print(f"PyTorch     : {torch.__version__}")
print(f"Torchvision : Available")
print(f"CUDA        : {torch.cuda.is_available()}")
print(f"OpenCV      : {cv2.__version__}")
print(f"NumPy       : {np.__version__}")
print(f"Matplotlib  : {matplotlib.__version__}")
print(f"Pandas      : {pandas.__version__}")
print(f"Scikit-Learn: {sklearn.__version__}")
print(f"Plotly      : {plotly.__version__}")
print(f"ImageIO     : {imageio.__version__}")
print(f"Pillow      : {PIL.__version__}")
print(f"PyYAML      : {yaml.__version__}")
print("=" * 70)

if torch.cuda.is_available():
    print("GPU Name    :", torch.cuda.get_device_name(0))
else:
    print("GPU         : Not available (Expected for local development)")
    print("             We will use Google Colab GPU for training.")

print("=" * 70)
print("Environment Ready!")
print("=" * 70)