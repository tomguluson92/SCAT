import cv2
import numpy as np
from PIL import Image
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

def rotate_img(image,joints_2d,joints_3d,angle):
    image = np.asarray(image)
    # grab the dimensions of the image and then determine the
    # centre
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

    #joints_2d rotation
    joints_2d = np.hstack((joints_2d, np.ones((joints_2d.shape[0], 1), dtype=type(joints_2d[0][0]))))
    joints_2d=np.dot(M,joints_2d.T).T

    #joints_3d rotation

    joints_3d = np.hstack((joints_3d, np.ones((joints_3d.shape[0], 1), dtype=type(joints_3d[0][0]))))
    M_3d=np.eye(4)
    M_3d[0][0]=M[0,0]
    M_3d[0][1]=M[0,1]
    M_3d[1][0]=-M[0,1]
    M_3d[1][1]=M[0,0]

    joints_3d = np.dot(M_3d, joints_3d.T).T


    new_width, new_height = image.shape[:2]

    scale = 224 / new_width
    #joints_2d=joints_2d * scale
    image = Image.fromarray(np.uint8(image)).convert('RGB')
    #image = image.resize((224, 224))
    
    return image,joints_2d,joints_3d[:,:3]





def convert_rotvec_to_quat(pose):
    new_pose = R.from_rotvec(pose).as_quat()[:, [3, 0, 1, 2]]
    quat_array = np.zeros((pose.shape[0], 4))
    for i, cur_pose in enumerate(new_pose):
        quat = Quaternion(cur_pose)
        # log_map = Quaternion.log_map(Quaternion(),quat).q[1:]
        quat_array[i, :] = quat.q

    return quat_array


def convert_quat_to_rotvec(quat_pose):
    output = np.zeros((quat_pose.shape[0], 3))
    for i, quat in enumerate(quat_pose):
        rot = R.from_quat(quat[[1, 2, 3, 0]])
        rotvec = rot.as_rotvec()
        output[i, :] = rotvec

    return output



def rotate_pose_param(pose,angle):
    angle=360-angle
    angle=np.pi*angle/180
    quat_array=convert_rotvec_to_quat(pose[None,:3])
    q1 = Quaternion(axis=[0, 0, 1], angle=angle) #rotate at z-axis clockwise
    quat_array = q1 * quat_array[0]
    quat_array = quat_array.q[None, :]
    pose[:3]=convert_quat_to_rotvec(quat_array)[0]
    return pose