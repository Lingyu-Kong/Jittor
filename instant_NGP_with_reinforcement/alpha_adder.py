from PIL import Image, ImageDraw, ImageFilter
import os

def find_all_file(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield f

category = "Scarf"

def main():
    base = r"/work/instant_NGP_with_reinforcement/my/data/"+category+"/train/"
    for i in find_all_file(base):
        if(category in i):
            im_rgb = Image.open(base+i)
            im_rgba = im_rgb.copy()
            im_rgba.putalpha(255)
            im_rgba.save(base+i)

if __name__ == '__main__':
    main()