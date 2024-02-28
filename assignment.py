import random
import numpy as np
import cv2
import glm


block_size = 1.0


def reorder_cv_gl(array):
    x, y, z = array.T
    return np.stack([x, z, y]).T


def read_camera_properties(filename):
    fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)

    camera_properties = {
        "camera_matrix": fs.getNode("CameraMatrix").mat(),
        "distortion_coefficients": fs.getNode("DistortionCoeffs").mat(),
        "r_matrix": fs.getNode("RMatrix").mat(),
        "t_matrix": fs.getNode("TMatrix").mat(),
    }

    fs.release()
    return camera_properties


def generate_grid(width, depth):
    # Generates the floor grid locations
    # You don't need to edit this function
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append(
                [x * block_size - width / 2, -block_size, z * block_size - depth / 2]
            )
            colors.append([1.0, 1.0, 1.0] if (x + z) % 2 == 0 else [0, 0, 0])
    return data, colors


def set_voxel_positions(width, height, depth):
    voxel_data = np.load(r"..\data\vox_data.npz")
    true_vox, colors = (
        voxel_data["vox_coords"],
        voxel_data["colors"],
    )

    colors[:, 0], colors[:, 2] = colors[:, 2], colors[:, 0]
    return reorder_cv_gl(true_vox), colors / 255


def get_cam_pos(path):
    # https://stackoverflow.com/a/14693971

    voxel_data = np.load(r"..\data\vox_data.npz")
    xy_step, z_step = (
        voxel_data["_stepsize"],
        voxel_data["z_stepsize"],
    )

    cam_props = read_camera_properties(path)
    tvec = cam_props["t_matrix"]
    rvec = cam_props["r_matrix"]
    rmatrix = cv2.Rodrigues(rvec)[0]

    pos = -(rmatrix.T).dot(tvec).reshape(3)
    cam_vox = pos / np.array([xy_step, xy_step, z_step])

    return reorder_cv_gl(cam_vox), reorder_cv_gl(rvec.reshape(3))


def get_cam_positions():
    # Generates dummy camera locations at the 4 corners of the room
    # TODO: You need to input the estimated locations of the 4 cameras in the world coordinates.

    pos, rvec = zip(
        *[get_cam_pos(f"../data/cam{i}/camera_properties.xml") for i in range(1, 5)]
    )
    return pos, [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    _, cam_angles = zip(
        *[get_cam_pos(f"../data/cam{i}/camera_properties.xml") for i in range(1, 5)]
    )
    cam_rotations = [glm.mat4(1), glm.mat4(1), glm.mat4(1), glm.mat4(1)]
    cam_rotations = [np.ones((4, 4)), np.ones((4, 4)), np.ones((4, 4)), np.ones((4, 4))]
    for c in range(len(cam_rotations)):
        cam_rotations[c][:3, :3] = cv2.Rodrigues(cam_angles[c])[0]
        cam_rotations[c] = glm.mat4(cam_rotations[c])

    return cam_rotations
