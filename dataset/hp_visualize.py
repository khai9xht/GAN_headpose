import numpy as np
import math
from math import cos, sin
import cv2


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=80):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll)
                 * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch)
                 * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img

def draw_axis_from_vectors(img, Rotation_matrix, tdx=None, tdy=None, size=80):

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * Rotation_matrix[0, 0] + tdx
    y1 = size * Rotation_matrix[1, 0] + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * Rotation_matrix[0, 1] + tdx
    y2 = size * Rotation_matrix[1, 1] + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * Rotation_matrix[0, 2] + tdx
    y3 = size * Rotation_matrix[1, 2] + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img

def convertAngleToVector(yaw, pitch, roll):
    p, y, r = pitch, yaw, roll
    Rotate_matrix = np.array([
        [cos(y)*cos(r), -cos(y)*sin(r), sin(y)],
        [cos(p)*sin(r)+cos(r)*sin(p)*sin(y), cos(p)*cos(r)-sin(p)*sin(y)*sin(r), -cos(y)*sin(p)],
        [sin(p)*sin(r)-cos(p)*cos(r)*sin(y), cos(r)*sin(p)+cos(p)*sin(y)*sin(r), cos(p)*cos(y)]
    ], dtype=np.float32)
    Rotate_matrix = Rotate_matrix / np.linalg.norm(Rotate_matrix)
    Rotate_matrix = Rotate_matrix.flatten('F')

    return Rotate_matrix

def convertListAngleToVector(yaws, pitchs, rolls):
    Rotate_matrixs = []
    for yaw, pitch, roll in zip(yaws, pitchs, rolls):
        Rotate_matrix = convertAngleToVector(yaw, pitch, roll)
        Rotate_matrixs.append(Rotate_matrix)
    return np.array(Rotate_matrixs, dtype=np.float32)

def ConvertVectorToAngle(Rotate_matrix):
    R = Rotate_matrix
    sin_y = R[0, 2]
    tan_r = - R[0, 1] / R[0, 0]
    tan_p = - R[1, 2] / R[2, 2]

    r = np.arctan(tan_r)
    p = np.arctan(tan_p)

    cos_y = R[0, 0] / np.cos(r)
    y1 = np.arctan2(sin_y, cos_y)
    if cos_y > 0:
        y = y1
    elif y1 < 0:
        y = np.pi + y1
    else:
        y = -np.pi + y1

    return y, p, r # yaw, pitch, roll
