from discord_webhook import DiscordWebhook, DiscordEmbed
from mss import mss
import mss
import time
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import sys
import os
import cv2
import pytesseract
import math, operator
def compare(im1,im2):
    img1 = Image.open(im1)
    img2 = Image.open(im2)
    dif = ImageChops.difference(img1, img2)
    return np.mean(np.array(dif))
def send(name,mess):
    url="https://discordapp.com/api/webhooks/978652726588235906/dZWsUlrgrpTy4eXo9BtHA0I7V5ArjS6tro3muxWru8mi0Qhk1ujClash3GUgdEkvCbLe"
    webhook = DiscordWebhook(url=url, username=name, content=mess,)
    with open("new_say.png", "rb") as f:
        webhook.add_file(file=f.read(), filename='example.jpg')
    response = webhook.execute()
def read(f):
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    image = cv2.imread(f)
    gray_im = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
    d = pytesseract.image_to_string(gray_im, lang='rus')
    return d
def partscreen(x, y, top, left,mode):
    with mss.mss() as sct:
        monitor_number = 2
        mon = sct.monitors[monitor_number]
        monitor = {
            "top": mon["top"] + top,  # 100px from the top
            "left": mon["left"] + left,  # 100px from the left
            "width": x,
            "height": y,
            "mon": 2,
        }
        sct_img = sct.grab(monitor)
        if mode ==1:
            # os.remove('file1.png')
            # os.rename('file.png', 'file1.png')
            mss.tools.to_png(sct_img.rgb, sct_img.size, output='file_out.png')
        else:
            mss.tools.to_png(sct_img.rgb, sct_img.size, output='new_say.png')


def find_ellement():
    sens=0.7
    img = cv2.imread('file_out.png')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('serch.jpg',cv2.IMREAD_GRAYSCALE)
    w, h = template.shape[::-1]
    result = cv2.matchTemplate(gray_img, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(result >= sens)
    if len(loc[0]) != 0:
        for pt in zip(*loc[::-1]):
            pt[0] + w
            pt[1] + h
        x = int((pt[0] * 2 + w) / 2)
        y = int((pt[1] * 2 + h) / 2)
        print("Found ", x, y)
        temp=0
        while y>60:
            partscreen(420, 80, 60+y-40, 740, 2)
            y-=110
            time.sleep(0.1)
            if temp==0:
                d=read("new_say.png")
                s = d.split('\n')
                name=s[0]
                print(name)
            im = Image.open("new_say.png")
            img = im.convert("RGB")
            pixdata = img.load()
            for u in range(img.size[1]):
                for x in range(img.size[0]):
                    if pixdata[x, u] <= (57, 57, 49) and pixdata[x, u] >= (0, 0, 0):
                        pixdata[x, u] = (255, 255, 255)
                    else:
                        pixdata[x, u] = (0, 0, 0)
            img.crop((270, 55, 390, 77)).save('f.jpg', quality=100)
            time.sleep(1)
            i = Image.open('f.jpg')
            p = pytesseract.image_to_string(i, config=r'--psm 6 -c tessedit_char_whitelist=0123456789')
            print(p)
            time.sleep(10000)
            mess=name+" "+p
            send(temp,mess)
            temp+=1
            print(s)
    else:
        print('хуй там')
if __name__ == '__main__':
    while True:
        partscreen(420, 940, 60, 740, 1)
        time.sleep(0.5)
        find_ellement()
        time.sleep(100)
