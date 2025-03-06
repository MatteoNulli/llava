import os
import torch
import numpy as np
import torchvision
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as transforms


def padding(b, pad=336):
    width, height = b.size
    tar = int(np.ceil(height / pad) * pad)
    top_padding = 0 # int((tar - height)/2)
    bottom_padding = tar - height - top_padding
    left_padding = 0
    right_padding = 0
    b = transforms.functional.pad(b, [left_padding, top_padding, right_padding, bottom_padding], fill=[255,255,255])

    return b

def Image_transform(img, hd_num=25):
    width, height = img.size
    trans = False
    if width < height:
        img = img.transpose(Image.TRANSPOSE)
        trans = True
        width, height = img.size
    ratio = (width/ height)
    scale = 1
    while scale*np.ceil(scale/ratio) <= hd_num:
        scale += 1
    scale -= 1
    scale = min(np.ceil(width / 560), scale)
    new_w = int(scale * 560)
    new_h = int(new_w / ratio)
    #print (scale, f'{height}/{new_h}, {width}/{new_w}')

    img = transforms.functional.resize(img, [new_h, new_w],)
    img = padding(img, 560)
    width, height = img.size
    if trans:
        img = img.transpose(Image.TRANSPOSE)

    return img


def frame2img(imgs, font):
    new_imgs = []
    for img in imgs:
        w, h = img.size
        scale = w/h
        if w > h:
            new_w = 560 * 2
            new_h = int(560 * 2 / scale)
        else:
            new_w = int(560 * 2 * scale)
            new_h = 560 * 2
        img = transforms.functional.resize(img, [new_h, new_w],)
        new_imgs.append(img)
    imgs = new_imgs
    new_w = 0
    new_h = 0
    pad = 40
    if w > h:
        for im in imgs:
            w,h = im.size
            new_w = max(new_w, w)
            new_h += h + 10 + pad
        new_img = Image.new('RGB', (new_w, new_h), 'white')
        draw = ImageDraw.Draw(new_img)
        curr_h = 0
        for idx, im in enumerate(imgs):
            w,h = im.size
            new_img.paste(im, (0, pad + curr_h))
            draw.text((0, curr_h ), f'<IMAGE {idx}>', font=font, fill='black')
            if idx + 1 < len(imgs):
                draw.line([(0, pad +curr_h + h +5), (new_w, pad +curr_h + h +5)], fill = 'black', width=2)
            curr_h += h + 10 + pad
        #print (new_w, new_h)
    else:
        for im in imgs:
            w,h = im.size
            new_w += w + 10
            new_h = max(new_h, h)
        new_h += pad
        new_img = Image.new('RGB', (new_w, new_h), 'white')
        draw = ImageDraw.Draw(new_img)
        curr_w = 0
        for idx, im in enumerate(imgs):
            w,h = im.size
            new_img.paste(im, (curr_w, pad))
            draw.text((curr_w, 0), f'<IMAGE {idx}>', font=font, fill='black')
            if idx + 1 < len(imgs):
                draw.line([(curr_w + w + 5, 0), (curr_w + w + 5, new_h)], fill = 'black', width=2)
            curr_w += w + 10
    return new_img
