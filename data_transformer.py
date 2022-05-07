from dis import dis
from tkinter import image_types
from cv2 import getRotationMatrix2D, warpAffine
import numpy as np
import cv2
import matplotlib as plt
import math
from PIL import Image
import random

# path 
path = "/Users/andy/Documents/Data_Transformer/newdim.jpg"
window_name = "Output Frame"
HEIGHT = 720
WIDTH = 1280
  
def initialize(path):
    '''
    opens an image from specified file path. Note image[x,y] --> x is row, y is col
    '''
    return cv2.imread(path)

def make_black():
    '''
    creates a black image of dimensions 1280 x 720.
    '''
    image = np.zeros(shape=[HEIGHT, WIDTH, 3], dtype=np.uint8)
    return image

def display_image(image, window):
    '''
    prints input image to window.
    image is the input image.
    window is the name of the output window
    '''
    cv2.imshow(window, image) 
    cv2.waitKey(0)

def resize(image, hor, ver, center=1, method="factor"):
    '''
    resized an image and overlays onto a black image.
    image is the input image of dimension 1280 x 720.
    hor and ver are values between 0 and 1.
    they are the factors multiplied to the dimensions of the input image.
    center default makes the resized image overlayed onto the center, otherwise on the top-corner.
    method default makes hor and ver scaling factors, otherwise makes hor and ver the bottom-right coordinates of the resized image
    affects labels.
    '''
    if (method == "factor"):
        x = round(WIDTH * hor)
        y = round(HEIGHT * ver)
    else: 
        x = hor
        y = ver
    new_dim = (x,y)
    cpy = np.copy(image)
    resized = cv2.resize(cpy, new_dim, interpolation=cv2.INTER_LINEAR)
    ret = make_black()

    if (center == 1):
        for a in range(0, x):
            for b in range(0, y):
                ret[b + round((HEIGHT - y)/2),a + round((WIDTH - x)/2)] = resized[b,a]
    else:
        for a in range(0, x):
            for b in range(0, y):
                ret[b,a] = resized[b,a]

    return ret 

def pixel_swap(image, seed, swaps):
    '''
    randomly swaps pixels of an image to create noise. 
    image is the input image.
    seed is an integer that allows you to repreduce results. If called with the same seed twice, the output will be the same.
    swaps is a non-negative integer that determines the number of times pixels are swapped.
    does not affect labels.
    '''
    height, width, channels = image.shape
    ret = np.copy(image)
    for num in range(0, swaps):
        random.seed(seed + num)
        a = random.randint(0, height - 1)
        random.seed(seed + num + swaps)
        b = random.randint(0, width - 1)
        random.seed(seed + num + 2*swaps)
        c = random.randint(0, height - 1)
        random.seed(seed + num + 3*swaps)
        d = random.randint(0, width - 1)
        
        x, y, z = ret[a,b]
        ret[a,b] = ret[c,d]
        ret[c,d] = [x,y,z]

    return ret

def mosaic(image, seed, swaps):
    '''
    randomly swaps pixels of an image to create noise. 
    image is the input image.
    seed is an integer that allows you to repreduce results. If called with the same seed twice, the output will be the same.
    swaps is a non-negative integer that determines the number of times pixels are swapped.
    does not affect labels.
    '''
    height, width, channels = image.shape
    ret = np.copy(image)
    for num in range(0, swaps):
        random.seed(seed + num)
        a = random.randint(0, height - 1)
        random.seed(seed + num + swaps)
        b = random.randint(0, width - 1)
        random.seed(seed + num + 2*swaps)
        c = random.randint(0, height - 1)
        random.seed(seed + num + 3*swaps)
        d = random.randint(0, width - 1)
        
        x, y, z = ret[a,b]
        ret[a,b] = ret[c,d]
        ret[c,d] = [x,y,z]

    return ret

def uniform_mosaic(image, seed, x_box, y_box):
    '''
    randomly swaps pixels of an image to create noise. 
    image is the input image.
    seed is an integer that allows you to repreduce results. If called with the same seed twice, the output will be the same.
    Note: x_box and y_box should divide the dimensions of the input image for easy of swapping
    x_box is the number of boxes vertically. for 720: 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 
    30, 36, 40, 45, 48, 60, 72, 80, 90, 120, 144, 180, 240, 360, 720
    y_box is the number of boxes horizontally: for 1280: 1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 64, 80, 128, 160, 256, 320, 640, 1280
    does not affect labels.
    '''
    arr = np.arange(1, x_box * y_box + 1, dtype=int) # middle index is not swapped if odd
    np.random.seed(seed)
    np.random.shuffle(arr)

    rows = int(HEIGHT/x_box)
    cols = int(WIDTH/y_box)

    height, width, channels = image.shape
    ret = np.copy(image)
    for num in range(int((x_box*y_box)/2)):
        c1 = ((arr[2*num] - 1) % y_box) * cols
        r1 = int((arr[2*num]-1)/y_box) * rows
        c2 = ((arr[2*num + 1] - 1) % y_box) * cols
        r2 = int((arr[2*num + 1]-1)/y_box) * rows
        for i in range(rows):
            for j in range(cols):
                x, y, z = ret[r1 + i, c1 + j]
                ret[r1 + i, c1 + j] = ret[r2 + i, c2 + j]
                ret[r2 + i, c2 + j] = [x,y,z]

    return ret

def reflect(image, num):
    '''
    reflects input image along an axis.
    image is the input image of dimension 1280 x 720.
    num controls which axis the image is flipped along. 
    num can be: 0 (flip over y-axis), 1 (flip over x-axis), -1 (flip over both axes).
    affects labels.
    '''
    ret = np.copy(image)
    return cv2.flip(ret,num)

def gaussian(image, kernel_x=10, kernel_y=10):
    '''
    returns gaussian blur of image.
    image is the input image of dimension 1280 x 720.
    kernel_x and kernel_y are positive integers that determine the blurring.
    does not affect labels.
    '''
    ret = np.copy(image)
    return cv2.GaussianBlur(ret, (2*kernel_x + 1, 2*kernel_y + 1), 0)
 
def rotate(image, deg):
    '''
    rotates image by deg degrees, and resizes if necessary to preserve data from orignial image
    image is the input image of dimension 1280 x 720.
    deg is the rotation angle in degrees, between 180 and -180
    affects labels.
    '''
    
    val = abs(deg)
    if (val > 90):
        val = 180 - val

    hypo = math.sqrt(math.pow(HEIGHT/2, 2) + math.pow(WIDTH/2, 2))
    init_angle = math.atan(HEIGHT/WIDTH)
    factor = HEIGHT/(2*hypo*math.sin(init_angle + math.radians(val)))
    image = resize(image, factor, factor)
    
    rows, cols, dim = image.shape
    M = getRotationMatrix2D(center = (round(cols/2), round(rows/2)), angle=deg, scale=1)
    ret = np.copy(image)
    return cv2.warpAffine(ret, M, (int(cols),int(rows)))

def shear(image, sh_x, sh_y):
    '''
    image is the input image of dimension 1280 x 720.
    sh_x and sh_y are the shearing factors in the x and y axis. 
    Note: due to image dimensions, sh_y must be less than 0.5619 and sh_x must be less than 1.7763
    affects labels.
    '''
    M = np.float32([[1, sh_x, 0],
                    [sh_y, 1, 0],
                    [0, 0, 1]])
    inv = np.linalg.inv(M)
    col = np.float32([[1280], [720], [1]])
    res = np.dot(inv,col)

    size = resize(image, int(math.floor(res[0][0])), int(math.floor(res[1][0])), center=0,method="coordinates")
    display_image(size, window_name)
    rows, cols, dim = image.shape
    sheared_img = cv2.warpPerspective(size,M,(int(cols),int(rows)))

    return sheared_img

def wave(image, amplitude, stretch, shift, dir):
    '''
    creates a wave-like effect on image, based on a sinusoidal curve.
    dir is 0 for horizontal effect and 1 for vertical effect.
    amplitude is the amplitude of the sinusoidal curve.
    stretch is a value between 0 and 1 that stretches the sinusoid longer.
    shift is the translation of the curve.
    affects labels.
    '''
    if (dir is 0):
        factor = (WIDTH - 2*amplitude)/WIDTH
        ret = resize(image, factor, factor)

        for j in range (0, HEIGHT):
            for i in range (0, WIDTH):
                try:
                    num = round(amplitude*math.sin(stretch*(j + math.radians(shift))))
                    if (num < 0):
                        ret[j,i] = ret[j, i - num]
                    else:
                        if (WIDTH - i - num < 0):
                            ret[j,WIDTH - i] = [0,0,0]
                        else:
                            ret[j,WIDTH - i] = ret[j, WIDTH - i - num]
                except:
                    ret[j,i] = [0,0,0]
    elif (dir is 1):
        factor = (HEIGHT - 2*amplitude)/HEIGHT
        ret = resize(image, factor, factor)

        for j in range (0, WIDTH):
            for i in range (0, HEIGHT):
                try:
                    num = round(amplitude*math.sin(stretch*(j + math.radians(shift))))
                    if (num < 0):
                        ret[i,j] = ret[i - num, j]
                    else:
                        if (HEIGHT - i - num < 0):
                            ret[HEIGHT - i,j] = [0,0,0]
                        else:
                            ret[HEIGHT - i,j] = ret[HEIGHT - i - num, j]
                except:
                    ret[i,j] = [0,0,0]
    
    return ret

def shadow(image, num, seed):
    for k in range(num):
        random.seed(seed*k)
        size = random.randint(50,200)
        random.seed(seed*k + 100)
        a = random.randint(0, HEIGHT - size - 1)
        random.seed(seed*k + 200)
        b = random.randint(0, WIDTH - size - 1)
        random.seed(seed*k + 300)
        shade = random.randint(400, 700)
        for i in range(size):
            for j in range(size):
                image[a + i, b + j] = image[a + i, b + j] * shade / 1000

    return image

def colour(image, bf=1, gf=1, rf=1, channel="RGB"):
    if (channel == "gray"):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        ret = image * 0.5
        print(image)

        


        # for i in range(HEIGHT):
        #     for j in range(WIDTH):
        #         [x,y,z] = ret[i,j]
        #         x1 = x * bf
        #         y1 = y * gf
        #         z1 = z * rf
        #         if (x1 > 255):
        #             x1 = 255
        #         if (y1 > 255):
        #             y1 = 255
        #         if (z1 > 255):
        #             z1 = 255

        #         ret[i,j] = [x1, y1, z1]
        
        return ret

def raindrop(image, seed, num=40, kernel_x=100, kernel_y=100):
    '''
    overlay gaussian blur onto groups of pixels?
    shade groups of pixels?
    '''
    blur = gaussian(image, kernel_x, kernel_y)
    
    for k in range(num):
        random.seed(seed*k)
        size = random.randint(20, 50)
        random.seed(seed*k + 100)
        a = random.randint(0, HEIGHT - size - 1)
        random.seed(seed*k + 200)
        b = random.randint(0, WIDTH - size - 1)
        random.seed(seed*k + 300)
        shade = random.randint(400, 700)
        for i in range(size):
            for j in range(size):
                image[a + i, b + j] = blur[a + i, b + j]

    return image


if __name__ == "__main__":
    image = initialize(path)

    display_image(image, window_name)

    # black = make_black()
    # display_image(black, window_name)

    # resized = resize(image, 0.5, 0.25)
    # display_image(resized, window_name)

    # pixel = pixel_swap(image, 100, 100000)
    # display_image(pixel, window_name)

    # ref0 = reflect(image, 0)
    # display_image(ref0, window_name)

    # ref1 = reflect(image, 1)
    # display_image(ref1, window_name)

    # ref_1 = reflect(image, -1)
    # display_image(ref_1, window_name)

    # gs5 = gaussian(image, 5, 6)
    # display_image(gs5, window_name)

    # gs10 = gaussian(image, 10, 3)
    # display_image(gs10, window_name)

    # gs20 = gaussian(image, 20, 25)
    # display_image(gs20, window_name)

    # gs40 = gaussian(image, 40, 40)
    # display_image(gs40, window_name)

    # rot1 = rotate(image, -60)
    # display_image(rot1, window_name)

    # rot1 = rotate(image, 30)
    # display_image(rot1, window_name)

    sh1 = shear(image, 0.4, 0.2)
    display_image(sh1, window_name)

    # sh2 = shear(image, 0.7, 0.5)
    # display_image(sh2, window_name)

    # wv = wave(image, 25, 0.025, 60, 0)
    # display_image(wv, window_name)

    # wv = wave(image, 50, 0.025, 45, 1)
    # display_image(wv, window_name)

    # ms1 = mosaic(image, 0, 5, 5)
    # display_image(ms1, window_name)

    # ms2 = mosaic(image, 1, 16, 128)
    # display_image(ms2, window_name)

    # ms3 = mosaic(image, 0, 9, 10)
    # display_image(ms3, window_name)

    # gray = colour(image, channel="gray")
    # display_image(gray, window_name)
 
    # blue = colour (image, 0,1,1)
    # display_image(blue, window_name)

    # green = colour (image, 1, 0, 1)
    # display_image(green, window_name)

    # red = colour (image, 1 , 1 , 0)
    # display_image(red, window_name)

    allcl = colour(image)
    display_image(allcl, window_name)

    # sdw1 = shadow(image, 10, 1)
    # display_image(sdw1, window_name)

    # blur = raindrop(image, 1)
    # display_image(blur, window_name)

    cv2.destroyAllWindows()



# #Image grayscale

# def im_grayscale(image, scale):
#     return image

# # Load the input image
# image = cv2.imread('C:\\Documents\\full_path\\tomatoes.jpg')
# cv2.imshow('Original', image)
# cv2.waitKey(0)
# # Use the cvtColor() function to grayscale the image
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Grayscale', gray_image)
# cv2.waitKey(0) 
# # Window shown waits for any key pressing event
# cv2.destroyAllWindows()
 
 
# while(1):
#     _, frame = cap.read()
#     # It converts the BGR color space of image to HSV color space
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
     
#     # Threshold of blue in HSV space
#     lower_blue = np.array([60, 35, 140])
#     upper_blue = np.array([180, 255, 255])
 
#     # preparing the mask to overlay
#     mask = cv2.inRange(hsv, lower_blue, upper_blue)
     
#     # The black region in the mask has the value of 0,
#     # so when multiplied with original image removes all non-blue regions
#     result = cv2.bitwise_and(frame, frame, mask = mask)
 
#     cv2.imshow('frame', frame)
#     cv2.imshow('mask', mask)
#     cv2.imshow('result', result)
     
#     cv2.waitKey(0)
 
