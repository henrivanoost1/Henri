from pytimeextractor import ExtractionService, PySettingsBuilder
import cv2
import pytesseract
import os
import sys
from pytesseract import Output
import json
# import layoutparser as lp
import re
import ast
import numpy as np
from PIL import Image
from pythonRLSA import rlsa
# from pythonRLSA import rlsa
import math
pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\Henri Van Oost\\AppData\\Local\\Programs\\Tesseract-OCR\\tesseract.exe'

img = cv2.imread('hello.png')


def read_img():
    img = cv2.imread('static/upload.png')
    ConvertToText(img)
    test = ConvertToText(img)
    # print(test)
    return test


def test():
    test = "dit is een testje"
    return test


def ConvertToText(img):
    # grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # noise removal
    # img = cv2.medianBlur(img, 5)
    # thresholding
    # img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # dilation
    # kernel = np.ones((5, 5), np.uint8)
    # img = cv2.dilate(img, kernel, iterations=1)
    # erosion
    # kernel = np.ones((5, 5), np.uint8)
    # img = cv2.erode(img, kernel, iterations=1)
    # opening - erosion followed by dilation
    # kernel = np.ones((5, 5), np.uint8)
    # img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # canny edge detection
    # img = cv2.Canny(img, 100, 200)
    # skew correction

    # -----------------------------------------------------------
    # Apply dilation and erosion to remove some noise
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    cv2.imwrite("thres.png", img)
    # -----------------------------------------------------
    text = pytesseract.image_to_string(img, lang="eng")
    os.remove("thres.png")
    text = str(text)
    # website = FindWebsite(text)
    # date = FindDate(text)
    return text


def FindTitle():
    image = cv2.imread('static/upload.png')  # reading the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # convert2grayscale
    (thresh, binary) = cv2.threshold(gray, 150, 255,
                                     cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # convert2binary
    # cv2.imshow('binary', binary)
    cv2.imwrite('binary.png', binary)

    (contours, _) = cv2.findContours(
        ~binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # find contours
    for contour in contours:
        """
        draw a rectangle around those contours on main image
        """
        [x, y, w, h] = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
    # cv2.imshow('contour', image)
    cv2.imwrite('contours.png', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # create blank image of same dimension of the original image
    mask = np.ones(image.shape[:2], dtype="uint8") * 255
    (contours, _) = cv2.findContours(
        ~binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # collecting heights of each contour
    heights = [cv2.boundingRect(contour)[3] for contour in contours]
    avgheight = sum(heights)/len(heights)  # average height
    # finding the larger contours
    # Applying Height heuristic
    for c in contours:
        [x, y, w, h] = cv2.boundingRect(c)
        if h > 2*avgheight:
            cv2.drawContours(mask, [c], -1, 0, -1)
    # cv2.imshow('filter', mask)
    cv2.imwrite('filter.png', mask)
    text = pytesseract.image_to_string(
        "C:/Users/Henri Van Oost/Documents/MCT/Semester5/Research Project/Henri/filter.png", lang="eng")
    print("Titel is"+text)
    return text


def FindPhoneNumber():
    string = read_img()
    regex = r"[\d]{4} [\d]{3} [\d]{4}"
    tel = re.findall(regex, string)
    print("telefon: "+str(tel))
    # tel = [x[0] for x in tel]
    tel = str(tel)
    tel = tel[2: -2].lower()
    print("telefon: "+tel)
    # print(web)
    return tel


def FindWebsite():
    string = read_img()
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex, string)
    web = [x[0] for x in url]
    web = str(web)
    web = web[2: -2].lower()
    # print(web)
    return web


# website = FindWebsite(text)
# print(website)


# text2 = "from winter to summer"
# model = lp.Detectron2LayoutModel(
#     'lp://PrimaLayout/mask_rcnn_R_50_FPN_3x/config')
# layout = model.detect(img)

# text2 = "Catch the post-impressionist exhibit after 19pm! Free organ show every Sunday at 4!"

def FindDate():
    try:

        text = read_img()

        settings = (PySettingsBuilder()
                    .addRulesGroup("DateGroup")
                    .excludeRules("timeRule")
                    .build()
                    )
        test = str(ExtractionService.extract(text, settings))
        result = ast.literal_eval(test)
        print(test)
        dct_test_str = str(result[0])

        # result2 = dct_test_str.replace("'", '"')

        dct_test_str = dct_test_str[24:-1]
        date = dct_test_str.split("'")[0]
        return date.lower()
    except:
        return ""
    # d = pytesseract.image_to_data(img, output_type=Output.DICT)
    # n_boxes = len(d['level'])
    # for i in range(n_boxes):
    #     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
    #     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # imS = cv2.resize(img, (960, 540))

    # cv2.imshow('img', imS)
    # cv2.waitKey(0)

# print(test)

# print(test)


# result = json.loads(test)
# result = ast.literal_eval(test)

# print(result)
# print(test)
# print(result[0])
# dct_test_str = str(result[0])
# print(dct_test_str)
# result2 = dct_test_str.replace("'", '"')
# print(dct_test_str)
# result2 = json.loads(dct_test_str)
# print(str(result2))

# dct_test_str = dct_test_str[24:-1]
# print(dct_test_str.split("'")[0])

# d = pytesseract.image_to_data(img, output_type=Output.DICT)
# n_boxes = len(d['level'])
# for i in range(n_boxes):
#     (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
#     cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# imS = cv2.resize(img, (960, 540))

# cv2.imshow('img', imS)
# cv2.waitKey(0)
