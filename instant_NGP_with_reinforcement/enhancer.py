from aip import AipImageProcess
import base64, os

APP_ID = "26523757"
API_KEY = "mNmZIoBXbl8UuPweskWGFlWR"
SECRET_KEY = "HeoNakhM6GGZPxoHb7X2wCtIUI0G2pQy"

client = AipImageProcess(APP_ID, API_KEY, SECRET_KEY)

def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

def find_all_file(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            yield f

category = "Scarf"

def main():
    base = r"/work/instant_NGP_with_reinforcement/my/data/"+category+"/train/"
    for i in find_all_file(base):
        if category in i:
            image = get_file_content(base+i)
            image = client.imageDefinitionEnhance(image)["image"]
            image = image.replace('data:image/png;base64,', '')
            image = base64.b64decode(image)
            with open(base+i, 'wb+') as fp:
                fp.write(image)

if __name__ == '__main__':
    main()