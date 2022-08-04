import os

from data_transformer import *

# path, make sure images are in the same folder as
FILE_DIRECTORY = os.path.dirname(__file__)
FILE_DIRECTORY = os.path.join(FILE_DIRECTORY, "images")
DEFAULT_IMAGE_PATH = os.path.join(FILE_DIRECTORY, "newdim.jpg")
SHADOW_1_PATH = os.path.join(FILE_DIRECTORY, "shadow.png")
SHADOW_2_PATH = os.path.join(FILE_DIRECTORY, "shadow2.jpeg")
SHADOW_3_PATH = os.path.join(FILE_DIRECTORY, "cone.jpeg")
SHADOW_4_PATH = os.path.join(FILE_DIRECTORY, "barrel.png")

# output window name
WINDOW_NAME = "Output Frame"

# image dimensions
DEFAULT_HEIGHT = 720
DEFAULT_WIDTH = 1280

def main() -> None:
    image = initialize_image(DEFAULT_IMAGE_PATH)

    display_image(image)

    black = make_black()
    display_image(black)

    resized = resize_image(image, 0.5, 0.25)
    display_image(resized)

    cropped = crop_image(image, random_seed=10)
    display_image(cropped)

    sp1 = apply_saltpepper(image, 0)
    display_image(sp1)

    sp2 = apply_saltpepper(image, 1)
    display_image(sp2)

    sp3 = apply_saltpepper(image, 2)
    display_image(sp3)

    ms1 = apply_mosaic(image, 10, 20)
    display_image(ms1)

    ms2 = apply_mosaic(image, 12, 30)
    display_image(ms2)

    ref1 = reflect_image(image, 0)
    display_image(ref1)

    ref2 = reflect_image(image, 1)
    display_image(ref2)

    ref3 = reflect_image(image, -1)
    display_image(ref3)

    gs1 = apply_gaussian(image, 5, 6)
    display_image(gs1)

    gs2 = apply_gaussian(image, 10, 3)
    display_image(gs2)

    gs3 = apply_gaussian(image, 20, 25)
    display_image(gs3)

    gs4 = apply_gaussian(image, 40, 40)
    display_image(gs4)

    rot1 = rotate_image(image, -60)
    display_image(rot1)

    rot2 = rotate_image(image, 30)
    display_image(rot2)

    sh1 = shear_image(image, 0.4, 0.2)
    display_image(sh1)

    sh2 = shear_image(image, 0.7, 0.5)
    display_image(sh2)

    wv1 = apply_wave(image)
    display_image(wv1)

    wv2 = apply_wave(image, axis=0)
    display_image(wv2)

    ums1 = uniform_mosaic(image, 0, 5, 10)
    display_image(ums1)

    ums2 = uniform_mosaic(image, 1, 16, 128)
    display_image(ums2)

    ums3 = uniform_mosaic(image, 0, 9, 10)
    display_image(ums3)

    gray = colour(image, mode="gray")
    display_image(gray)

    blue = colour(image, 0, 1, 0)
    display_image(blue)

    green = colour(image, 1, 0, 1)
    display_image(green)

    red = colour(image, 1, 1, 0)
    display_image(red)

    boost_red = colour(image, 1, 1, 2)
    display_image(boost_red)

    shdr = shadow_round(image, 10, 1)
    display_image(shdr)

    rain1 = raindrop(image, 1, num_of_raindrops=20, kernel_x=300)
    display_image(rain1)

    rain2 = raindrop(image, 10)
    display_image(rain2)

    shdo1 = shadow_object(
        image, 40, shadow1_freq=2, shadow2_freq=1, shadow3_freq=0, shadow4_freq=0
    )
    display_image(shdo1)

    shdo2 = shadow_object(image, 2)
    display_image(shdo2)

    glare = lens_glare(image, 5)
    display_image(glare)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()