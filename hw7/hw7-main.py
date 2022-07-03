import cv2
import numpy as np
import matplotlib.pyplot as plt
import copy

lena = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)
plt.imshow(lena, cmap='gray', vmin=0, vmax=255)

def binarize(lena, thr=128):
    after = copy.deepcopy(lena)
    for r in range(after.shape[0]):
        for c in range(after.shape[1]):
            if (after[r][c] < thr) : after[r][c] = 0
            else: after[r][c] = 255
    return after
bina = binarize(lena)
plt.imshow(bina, cmap='gray', vmin=0, vmax=255)

def downsample(bina):
    after = np.zeros((64, 64))
    for r in range(after.shape[0]):
        for c in range(after.shape[1]):
            after[r][c] = bina[r*8][c*8]
    return after
downa = downsample(bina)
plt.imshow(downa, cmap='gray', vmin=0, vmax=255)

def h(b, c, d, e):
    if (b == c) and (d != b or e != b):
        return 'q'
    elif (b == c) and (d == b and e == b):
        return 'r'
    elif (b != c):
        return 's'

def f(a_list):
    r_num = 0
    q_num = 0
    for a in a_list:
        if a == 'q': q_num +=1
        if a == 'r': r_num +=1
    if (r_num == 4): return 5
    else: return q_num

def get_yokoi(downa, r, c):
    blocks = [
        [(0, 0), ( 0,   1), (-1,  1), (-1,  0)], 
        [(0, 0), (-1,   0), (-1, -1), ( 0, -1)], 
        [(0, 0), ( 0,  -1), ( 1, -1), ( 1,  0)], 
        [(0, 0), ( 1,   0), ( 1,  1), ( 0,  1)], 
    ]
    a = []
    for blk in blocks:
        p = []
        for ofst in blk:
            if (r+ofst[0] >= 0 and r+ofst[0] < downa.shape[0]) and (c+ofst[1] >= 0 and c+ofst[1] < downa.shape[1]): 
                p.append(downa[r + ofst[0]][c + ofst[1]])
            else:
                p.append(0)
        a.append(h(p[0], p[1], p[2], p[3]))
    return f(a)

def yokoi(downa):
    after = np.zeros((64, 64))
    for r in range(downa.shape[0]):
        for c in range(downa.shape[1]):
            if downa[r][c] == 255:
                after[r][c] = get_yokoi(downa, r, c)
    return after

def pair_h(a, m):
    if (a == m): return 1
    else: return 0

def pair_relationship(yona):
    after = np.zeros((64, 64))
    blk = [(0, 1), (-1, 0), (0, -1), (1, 0)]
    for r in range(after.shape[0]):
        for c in range(after.shape[1]):
            ps = []
            for ofst in blk:
                if (r+ofst[0] >= 0 and r+ofst[0] < yona.shape[0]) and (c+ofst[1] >= 0 and c+ofst[1] < yona.shape[1]): 
                    ps.append(yona[r + ofst[0]][c + ofst[1]])
                else:
                    ps.append(0)
                
            ps = [ pair_h(p, 1) for p in ps] 
            if (ps.count(1) < 1 or yona[r][c] != 1):
                after[r][c] = 1 #q
            else:
                after[r][c] = 0 #p
    return after
    
def thinning(downa):
    original = copy.deepcopy(downa)
    while True:
        yona = yokoi(original)
        prna = pair_relationship(yona)
        thna = np.copy(original)
        for r in range(original.shape[0]):
            for c in range(original.shape[1]):
                if thna[r][c] == 255:
                    if get_yokoi(thna, r, c) == 1 and prna[r][c] == 0:
                        thna[r][c] = 0
        
        if (thna == original).all():
            break

        original = copy.deepcopy(thna)
    return original
    
thna = thinning(downa)
plt.imshow(thna, cmap='gray', vmin=0, vmax=255)
plt.imsave("thinning-lean.bmp", thna, cmap='gray', vmin=0, vmax=255)
plt.imsave("thinning-lean.png", thna, cmap='gray', vmin=0, vmax=255)