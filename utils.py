<<<<<<< HEAD
<<<<<<< HEAD
# import pyautogui
import os, re, sys, time
from datetime import datetime, timedelta
=======
=======
>>>>>>> 996b3c85772bec68254d39e5f520dc2917e42bd0
import pyautogui
import time
from datetime import datetime
from datetime import timedelta
<<<<<<< HEAD
>>>>>>> 996b3c85772bec68254d39e5f520dc2917e42bd0
=======
>>>>>>> 996b3c85772bec68254d39e5f520dc2917e42bd0
import torch
# import torchvision as thv

import numpy as np
<<<<<<< HEAD
<<<<<<< HEAD
=======
import sys
import re
import os
>>>>>>> 996b3c85772bec68254d39e5f520dc2917e42bd0
=======
import sys
import re
import os
>>>>>>> 996b3c85772bec68254d39e5f520dc2917e42bd0
import shutil
import cv2



def screen_monitor(sec = 60, replace = True):

    if os.path.isdir(path):
        if replace == True:
            # os.remove('./sct')
            shutil.rmtree(path)
            print('Path removed')
            os.mkdir(path)
    else:
        os.mkdir(path)

    while True:

        pyautogui.PAUSE = 1
        filename = 'sct_{}'.format(str(datetime.now())).replace(
            '.', '_') + '.png'
        # print('filename: ', filename)
        pyautogui.screenshot(os.path.join(path, filename))
        # print(len(os.listdir(path)))
        for fn in os.listdir(path):
            # print(datetime.strptime(fn[4:].replace('_', '.').replace('.png',''), '%Y-%m-%d %H:%M:%S.%f'))
            if datetime.now() - datetime.strptime(fn[4:].replace('.png', ''), '%Y-%m-%d %H:%M:%S_%f') > timedelta(seconds=args.sec):
                os.remove(os.path.join(path, fn))


def dirrs(obj,start=""):
    if start != "":
        return [attr for attr in dir(obj) if not attr.startswith("_") and attr.startswith(start)]
    else:
        return [attr for attr in dir(obj) if not attr.startswith("_")]

def dirrm(obj):
    try:
        dicts = {attr: getattr(obj, attr) for attr in dir(obj) if not attr.startswith("_")}
        return dicts
    except:
        print("Not executable")


def close_ads(driver):
    try:
        closeitem = driver.find_element_by_id('feedback-header-close')
        closeitem.click()
        time.sleep(1)

        close = driver.find_element_by_link_text("关闭")
        close.click()
        time.sleep(1)
    except:
        pass


def violent_pyt():
    while True:
        close_ads()

        # <img src="/captcha" id="captcha_image" alt="Are you human?">
        captcha_ele = driver.find_element_by_id("captcha_image")

        cap_file = './captcha.png'
        png = captcha_ele.screenshot(cap_file)

        display(Image(filename=cap_file))

        # <a href="#reload_captcha" id="captcha_reload">看不清？换一张！</a>

        captcha_im = PIL.Image.open(cap_file)
        result = pytesseract.image_to_string(
            captcha_im,
            config='--psm 10 --oem 2 -c tessedit_char_whitelist=0123456789')
        result = result.replace(" ", "")

        print(result)
        driver.find_element_by_id("captcha_reload").click()
        time.sleep(0.3)

        #     pyautogui.hotkey('command','tab')

        try:
            if int(result) // 10e5 < 1 and int(result) // 10e3 > 1:
                driver.find_element_by_id("signup_verification").send_keys(
                    result)
                driver.find_element_by_id("signup_verification").send_keys(
                    Keys.RETURN)
                break

        except:
            continue

path = "webdriver.webkitgtk"

def getit(obj, path):
    # finite num iter

    if len(path.split()) > 1:
        sub = path.split()[0]
        obj = getattr(obj, sub)

        return getit(obj, sub, path)

    else:
        return obj


# getit(selenium, path)

def get_next_sib(element, driver):
    return driver.execute_script(
    """
    return arguments[0].nextElementSibling;
    """,
    element)

def get_attr_dict(element):
    return driver.execute_script(
        """var items = {}; 
        for (index = 0; index < arguments[0].attributes.length; ++index) 
        { items[arguments[0].attributes[index].name] = arguments[0].attributes[index].value }; 
        return items;""",
        element)


def rect2tup(rect):
    return tuple(
        map(lambda x: x * 2,
            [rect['x'], rect['y'], rect['width'], rect['height']]))



# cnt = 0
# for bh in gen(data, 100):
#     cnt += 1
#     print(bh)
#     if cnt > 3:
#         break

# o = gen(data, 100)

<<<<<<< HEAD
<<<<<<< HEAD
# next(o)
=======
# next(o)
>>>>>>> 996b3c85772bec68254d39e5f520dc2917e42bd0
=======
# next(o)
>>>>>>> 996b3c85772bec68254d39e5f520dc2917e42bd0
