import numpy as np
import cv2
import sys

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    h, w = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / h
        dim = (int(w * r), height)
    else:
        r = width / w
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation = inter)
    return resized

def resize(img, shape):
    print(img.shape, shape)
    old_h, old_w = img.shape[:2]
    new_h, new_w = shape
    scale_w = new_w / old_w
    scale_h = new_h / old_h
    if scale_w > scale_h:
        img = image_resize(img, width=new_w)
        act_h, act_w = img.shape[:2]
        assert new_w == act_w
        off_h = (act_h - new_h) // 2
        return img[off_h:off_h+new_h, :]
    else:
        img = image_resize(img, height=new_h)
        act_h, act_w = img.shape[:2]
        assert new_h == act_h
        off_w = (act_w - new_w) // 2
        return img[:, off_w:off_w+new_w]

def addback(back, img, mask):
    assert img.shape[:2] == mask.shape
    fg = cv2.bitwise_and(img, img, mask=mask)
    mask = cv2.bitwise_not(mask)
    back = resize(back, img.shape[:2])
    # back = np.full(img.shape, 0, dtype=np.uint8)
    bk = cv2.bitwise_and(back, back, mask=mask)
    final = cv2.bitwise_or(fg, bk)
    return final

if __name__ == '__main__':
    back = cv2.imread(sys.argv[1])
    img = cv2.imread(sys.argv[2])
    mask = cv2.imread(sys.argv[3], 0)
    mask = np.where(mask < 128, 0, 255).astype(np.uint8)
    out = sys.argv[4]
    final = addback(back, img, mask)
    cv2.imwrite(out, final)

