from PIL import Image
import sys
import os
import random

def openImage():
    return Image.open(sys.argv[2])

def operation():
    return sys.argv[1]

def seed(img):
    random.seed(hash(img.size))

def getPixels(img):
    w, h = img.size
    pxs = []
    for x in range(w):
        for y in range(h):
            pxs.append(img.getpixel((x, y)))
    return pxs

def scrambledIndex(pxs):
    idx = list(range(len(pxs)))
    random.shuffle(idx)
    return idx

def scramblePixels(img):
    seed(img)
    pxs = getPixels(img)
    idx = scrambledIndex(pxs)
    out = []
    for i in idx:
        out.append(pxs[i])
    return out

def unScramblePixels(img):
    seed(img)
    pxs = getPixels(img)
    idx = scrambledIndex(pxs)
    out = list(range(len(pxs)))
    cur = 0
    for i in idx:
        out[i] = pxs[cur]
        cur += 1
    return out

def storePixels(name, size, pxs):
    outImg = Image.new("RGB", size)
    w, h = size
    pxIter = iter(pxs)
    for x in range(w):
        for y in range(h):
            outImg.putpixel((x, y), next(pxIter))
    outImg.save(name)

def main():
    img = openImage()
    if operation() == "scramble":
        pxs = scramblePixels(img)
        storePixels("scrambled.png", img.size, pxs)
    elif operation() == "unscramble":
        pxs = unScramblePixels(img)
        storePixels("unscrambled.png", img.size, pxs)
    else:
        sys.exit("Unsupported operation: " + operation())

if __name__ == "__main__":
    main()