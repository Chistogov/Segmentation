from PIL import Image, ImageOps
import glob, os
DATA_PATH="/home/maxim/Документы/py_projects/Segmentation/reflex/train/image_s"
n = 0
os.chdir(DATA_PATH)
cdn = 0

for file in glob.glob("*.png"):
    print (file)
    img = Image.open(file).convert('L')
    img = ImageOps.invert(img)
    os.remove(file)
    img.save(file)