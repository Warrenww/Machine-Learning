import numpy as np
import matplotlib.pyplot as plt

def get_data(filename = ''):
    result = list()
    img = plt.imread(filename)
    for x,r in enumerate(img):
        for y,c in enumerate(r):
            result.append((x,y,tuple(c)))

    return np.array(result)

data_1 = get_data('image1.png')
data_2 = get_data('image2.png')

def kernel(x_i, x_j, g_s, g_c):
    return 
    pass

print(data_1[0])
