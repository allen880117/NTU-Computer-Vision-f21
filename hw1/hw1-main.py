import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

lena = cv2.imread('lena.bmp')
plt.imshow(lena)
cv2.imwrite('lena.png',lena)

def my_flip(lena, tag='ud'):
    after = copy.deepcopy(lena)
    for r in range(lena.shape[0]):
        for c in range(lena.shape[1]):
            if (tag=='ud'): # upside-down
                after[r][c] = lena[lena.shape[0]-1-r][c]
            elif(tag=='rl'): # right-side-left
                after[r][c] = lena[r][lena.shape[1]-1-c]
            elif(tag=='dg'): # diagonally flip
                after[r][c] = lena[c][r]
            else: # no flip
                after[r][c] = lena[r][c]
    
    return after

lena_upside_down = my_flip(lena, tag='ud')
plt.imshow(lena_upside_down)
cv2.imwrite('upside-down lena.bmp',lena_upside_down)
cv2.imwrite('upside-down lena.png',lena_upside_down)

lena_rl = my_flip(lena, tag='rl')
plt.imshow(lena_rl)
cv2.imwrite('right-side-left lena.bmp',lena_rl)
cv2.imwrite('right-side-left lena.png',lena_rl)

diag_flip = my_flip(lena, tag='dg')
plt.imshow(diag_flip)
cv2.imwrite('diagonally flip lena.bmp',diag_flip)
cv2.imwrite('diagonally flip lena.png',diag_flip)

# Shrink
def shrink(lena, target_size=(256,256)):
    after = copy.deepcopy(lena)
    after = cv2.resize(lena, target_size)
    return after

shrink_lena = shrink(lena, (lena.shape[0]//2, lena.shape[1]//2))
plt.imshow(shrink_lena)
cv2.imwrite('shrink lena.bmp',shrink_lena)
cv2.imwrite('shrink lena.png',shrink_lena)

# Binarize
def binarize(lena, thr=128):
    after = copy.deepcopy(lena)
    for r in range(after.shape[0]):
        for c in range(after.shape[1]):
            if (np.sum(after[r][c]) < thr*3) : after[r][c] = [0,0,0]
            else: after[r][c] = [255,255,255]
    return after

bina = binarize(lena)
plt.imshow(bina)
cv2.imwrite('binarize lena.bmp',bina)
cv2.imwrite('binarize lena.png',bina)
