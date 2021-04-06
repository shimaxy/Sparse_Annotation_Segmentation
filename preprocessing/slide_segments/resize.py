from PIL import Image
from glob import glob 
Image.MAX_IMAGE_PIXELS = None

path = '/projects/patho1/melanoma_diagnosis/mpath_x2.5'

for filename in sorted(glob(path + '/*.tif')):
    print(filename)
    image = Image.open(filename)
    w,h = image.size    

    image_r = image.resize((int(w/4), int(h/4)))
    out_name = out_name.replace('_x10.tif','_x2.5.jpg')
    image_r.save(out_name)
    
    
