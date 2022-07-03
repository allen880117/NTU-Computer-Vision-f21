# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy


# %%
lena = cv2.imread('lena.bmp')
plt.imshow(lena)


# %%
def get_historgram(lena):
    hist = np.zeros(256)
    for r in range(lena.shape[0]):
        for c in range(lena.shape[1]):
            hist[lena[r][c]] += 1
    return hist

def draw_save_historgram(hist, name):
    plt.clf()
    plt.bar([i for i in range(256)], hist, width=1)
    plt.savefig(name)    

hist = get_historgram(lena)
draw_save_historgram(hist, "hist.png")


# %%
def divide_by_3(lena):
    after = copy.deepcopy(lena)
    for r in range(lena.shape[0]):
        for c in range(lena.shape[1]):
            tmp = lena[r][c] // 3
            after[r][c] = tmp
    return after
lena_d_3 = divide_by_3(lena)
plt.imshow(lena_d_3)
cv2.imwrite("lena_d_3.bmp", lena_d_3)
cv2.imwrite("lena_d_3.png", lena_d_3)


# %%
hist_d_3 = get_historgram(lena_d_3)
draw_save_historgram(hist_d_3, "hist_d_3.png")


# %%
def equalize_histogram(lena):
    after = copy.deepcopy(lena)

    hist = get_historgram(lena)
    sk = np.zeros(256)
    n = lena.shape[0] * lena.shape[1]

    tmp_sum = 0
    for k in range(256):
        tmp_sum += hist[k]
        tmp = 255 * tmp_sum // n
        sk[k] = tmp

    for r in range(lena.shape[0]):
        for c in range(lena.shape[1]):
            tmp = sk[lena[r][c][0]]
            after[r][c] = [tmp, tmp, tmp]

    return after
    
lena_eq = equalize_histogram(lena_d_3)
plt.imshow(lena_eq)
cv2.imwrite("lena_eq.bmp", lena_eq)
cv2.imwrite("lena_eq.png", lena_eq)


# %%
hist_eq = get_historgram(lena_eq)
draw_save_historgram(hist_eq, "hist_eq.png")


# %%



