from discord_webhook import DiscordWebhook, DiscordEmbed
from mss import mss
import mss
import time
from PIL import Image, ImageChops
import numpy as np
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
    response = webhook.execute()
def partscreen(x, y, top, left,mode):
    with mss.mss() as sct:
        monitor = {"top": top, "left": left, "width": x, "height": y}
        sct_img = sct.grab(monitor)
        if mode ==1:
            os.remove('file1.png')
            os.rename('file.png', 'file1.png')
            mss.tools.to_png(sct_img.rgb, sct_img.size, output='file.png')
        else:
            mss.tools.to_png(sct_img.rgb, sct_img.size, output='new_say.png')


def find_ellement():
    sens=0.7
    img = cv2.imread('file.png')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.imread('serch.png',cv2.IMREAD_GRAYSCALE)
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
        time.sleep(0.5)
        partscreen(715, 570-y, 600+y-30,900 , 2)
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        time.sleep(0.5)
        image = cv2.imread("new_say.png")
        d=pytesseract.image_to_string(image,lang='eng')
        s = d.split('\n')
        name=s[0]
        mess=full_data = ' '.join(s)
        send(name,mess)
        print(s)
    else:
        print('хуй там')
if __name__ == '__main__':
    send(None,"Проверка")
    # while True:
    #     partscreen(715,570,600,900,1)
    #     if compare('file1.png','file.png') >3:
    #         time.sleep(0.5)
    #         find_ellement()
    #     time.sleep(1)
