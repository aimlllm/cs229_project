import os
import shutil
import random
from PIL import Image

count = 0
dir = 'CASIA2.0_revised/Tp'
dest = 'photoshopvsreal/Tp'
for filename in os.listdir(dir):
    f = os.path.join(dir, filename)
    if os.path.isfile(f) and filename.endswith('.jpg') or filename.endswith('.bmp') or filename.endswith('.tif'):
        image = Image.open(f)
        image = image.crop((image.width/2 - 112, image.height/2 - 112, image.width/2 + 112, image.height/2 + 112))
        image = image.convert('RGB')
        image.save(dest + '\Tp' + str(count) + '.jpg')
        image.close()
        count += 1


        # if random.random() < 0.1:
            # shutil.move(f, os.path.join(dest, filename))
        # image = Image.open(f)
        # image = image.resize((256,256))
        # image = image.convert('RGB')
        # image.save(f)

# dir = 'photoshopvsreal/train/Tp'
# dest = 'photoshopvsreal/test/Tp'
# for filename in os.listdir(dir):
#     f = os.path.join(dir, filename)
#     if os.path.isfile(f) and filename.endswith('.jpg') or filename.endswith('.bmp'):
#         if random.random() < 0.1:
#             shutil.move(f, os.path.join(dest, filename))
#         # image = Image.open(f)
#         # image = image.resize((256, 256))
#         # image = image.convert('RGB')
#         # image.save(f)