import cv2
import numpy as np
import math
import os


def set_roi(img_2):
    ori_height, ori_width, _ = img_2.shape
    img = img_2[int(ori_height / 3):int(ori_height * 3 / 5), :]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 100, 200)

    cropped_height, cropped_width = gray.shape

    for i in range(cropped_height):
        if edges[i, int(cropped_width / 3)] == 255:
            start_x_upper = i

        if edges[i, int(cropped_width * 2 / 3)] == 255:
            start_x_down = i

    vari_x = int(cropped_width * 2 / 3) - int(cropped_width / 3)

    if start_x_down - start_x_upper > 0:
        angle = math.atan2(vari_x, start_x_down - start_x_upper)
    else:
        angle = math.atan2(vari_x, start_x_upper - start_x_down)
        angle = -angle

    M = cv2.getRotationMatrix2D((ori_width / 2.0, ori_height / 2.0), angle / 4, 1)
    rotated_ori = cv2.warpAffine(img_2, M, (ori_width, ori_height))

    rotated_ori = rotated_ori[int(ori_height / 3):int(ori_height * 3 / 5), :]

    height, width, _ = rotated_ori.shape
    gray = cv2.cvtColor(rotated_ori, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(gray, 100, 200)

    for i in range(width):
        if edges[int(cropped_height / 2), i] == 255:
            flag_1 = i
            break

    for i in range(width):
        if edges[int(cropped_height / 2), i] == 255:
            flag_2 = i

    edges = rotated_ori[:, flag_1:flag_2]

    return edges


def grid(img, grid_img_path):
    height, width, _ = img.shape
    grid = int(width / 19)

    for j in range(3, 17):
        grid_img = img[:, grid * (j - 1):grid * (j)]
        grid_img = cv2.bitwise_not(grid_img)
        cv2.imwrite(grid_img_path + '_' + str(j) + '.png', grid_img)


if __name__ == "__main__":
    img_path = '../../data/panel_ori_ver1/'

    idx = 1

    for path in os.listdir(img_path):
        img = cv2.imread(img_path + path)
        rotated_ori = set_roi(img)

        cropped_image_path = '../../data/panel_cropped_ver1/cropped_' + str(idx) + '.png'
        grid_img_path = '../../data/panel_train_ver1/grid_' + str(idx) + '_'
        cv2.imwrite(cropped_image_path, rotated_ori)
        cropped_image = cv2.imread(cropped_image_path)
        grid(cropped_image, grid_img_path)

        idx = idx + 1

    print("Grid finish")

    # cropped_img_path = '../dataset/cropped_images/1.bmp'
    # grid_img_path = '../dataset/grid_images/1_'
    # img = cv2.imread(img_path)
    # rotated_ori = set_roi(img)
    # cv2.imwrite(cropped_img_path, rotated_ori)
    # copped_img = cv2.imread(cropped_img_path)
    # grid(copped_img, grid_img_path)
    # print("Finish!")
