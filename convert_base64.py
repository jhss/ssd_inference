import base64
import cv2
import argparse
import requests

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type=str)
args = parser.parse_args()

img = open(args.image_path, 'rb').read()

converted_img = base64.b64encode(img)
print(converted_img)
utf_img = converted_img.decode('utf-8')
#print(utf_img)
#r = requests.post('url',data=utf_img)
with open("converted_utf.txt", "w") as f:
    f.write(utf_img)
