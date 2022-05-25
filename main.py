from discord_webhook import DiscordWebhook, DiscordEmbed
import num_det
from mss import mss
import mss
import time
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
import sqlite3
import cv2
import pytesseract
import datetime
import dearpygui.dearpygui as dpg
import dearpygui.demo as demo
import math, operator
def Bd(name,cost,ammount,temp):
    name = name.replace("-", "_")
    cur.executescript("CREATE TABLE IF NOT EXISTS " + name + """(
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   cost int,
                   ammount int,
                   place int,
                   date TEXT,
                   time TEXT);""")
    conn.commit()
    cur.execute("INSERT INTO " + name + """ (cost, ammount, place, date,time)
          VALUES(?, ?, ?, ?, ?)""", (cost, ammount, temp, str(datetime.date.today()),str(datetime.datetime.now().strftime("%H:%M"))))
    print(name)
    # cur.execute('SELECT id FROM lots WHERE (lotname=?)', (name,))
    # entry = cur.fetchone()
    # if entry is None:
    #     cur.execute('INSERT INTO lots (lotname, selling) VALUES (?,?)', (name, 1))
    #     conn.commit()
    #     # cur.execute('SELECT * FROM lots')
    #     cur.execute('SELECT id FROM lots WHERE (lotname=?)', (name,))
    #     n = cur.fetchone()[0]
    #     bd = 'dada-dada'
    #     # bd=str(n)+'tab'
    #     conn.commit()
    #     cur.executescript("CREATE TABLE IF NOT EXISTS " + bd+ """(
    #               id INTEGER PRIMARY KEY AUTOINCREMENT,
    #               cost int,
    #               ammount int,
    #               place int,
    #               date TEXT);""")
    #     conn.commit()
    # else:
    #     cur.execute('SELECT id FROM lots WHERE (lotname=?)', (name,))
    #     n = cur.fetchone()[0]
    #     conn.commit()
    #     bd = 'dada-dada'
    #     # bd=str(n)+'tab'
    # cur.execute("INSERT INTO " + bd + """" (cost, ammount, place, date)
    #       VALUES(?, ?, ?, ?)""", (cost, ammount, temp, 'ddd'))
    # cursor.execute("SELECT `price` FROM `catalog` WHERE `id` = ? ", (id,))
    # conn.commit()
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
        temp=1
        while y>60:
            partscreen(420, 80, 60+y-40, 740, 2)
            y-=110
            time.sleep(0.1)
            if temp==1:
                d=read("new_say.png")
                s = d.split('\n')
                name=s[0]
                print(name)
            #Обработка
            im = Image.open("new_say.png")
            img = im.convert("RGB")
            pixdata = img.load()
            for u in range(img.size[1]):
                for x in range(img.size[0]):
                    if pixdata[x, u] <= (57, 57, 49) and pixdata[x, u] >= (0, 0, 0):
                        pixdata[x, u] = (255, 255, 255)
                    else:
                        pixdata[x, u] = (0, 0, 0)
            # функция стоймости
            img.crop((274, 55, 390, 75)).save('f.jpg', quality=100)
            time.sleep(0.1)
            'fin_num.jpg'
            # im = Image.open('fin_num.jpg')
            im = Image.open('f.jpg')
            p=num_det.detect(im,1)
            # обработка изображения колличества
            img.crop((25, 47, 75, 62)).save('numb.jpg', quality=100)
            img1 = Image.open('numb.jpg')
            pixdata = img1.load()
            gt=False
            for x in range(img1.size[0]):
                if gt is True:
                    break
                if pixdata[x, 1] != (255, 255, 255):
                    for u in range(img1.size[1]):
                        pixdata[x, u] = (255, 255, 255)
                else:
                    # for m in range(x+20):
                    #     for t in range(img1.size[1]):
                    #         pixdata[m, t] = (255, 255, 255)
                            gt=True
            img1.save('fin_num.jpg', quality=100)
            # функция получения значения
            img2 = Image.open('fin_num.jpg')
            allo = num_det.detect(img2,2)
            # Вывод в дискорд
            trigger=138800
            if p>=trigger:
                mess="Лот "+name+" достиг указанной вами цены верхнего порога - "+str(trigger) +" цена:"+str(p)+" колличество "+str(allo) + " строка товара "+str(9-temp)
                send(None,mess)
            Bd(name, p, allo, temp)
            temp+=1
    else:
        print('хуй там')
if __name__ == '__main__':
    conn = sqlite3.connect(r'bd.db')
    cur = conn.cursor()
    monitor_number = 2

    dpg.create_context()
    dpg.create_viewport(title='warspear bot', width=600, height=600)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.start_dearpygui()
    dpg.destroy_context()

    while True:
        partscreen(420, 940, 60, 740, 1)
        find_ellement()
        time.sleep(1000)





