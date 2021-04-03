from PIL import Image
from glob import glob 
Image.MAX_IMAGE_PIXELS = 999999999

path = '/projects/patho1/melanoma_diagnosis/mpath_x2.5/ROI_box'

for filename in sorted(glob(path + '/*.tif')):
    print(filename)
    image = Image.open(filename)
    w,h = image.size    

    image_r = image.resize((int(w/4), int(h/4)))
    out_name = filename.replace('ROI_box','ROI_box_2.5')
    out_name = out_name.replace('_x10_z0.tif','_x2.5_z0.jpg')
    image_r.save(out_name)
    
    