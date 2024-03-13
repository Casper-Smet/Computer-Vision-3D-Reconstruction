import random
import numpy as np
import cv2
import glm
from scipy.spatial import distance


block_size = 1.0
# VOXEL_DATA = np.load(r"..\data\4persons\vox_data_video.npz")
# VOXEL_DATA = np.load(r"..\data\4persons\vox_data_video_multi.npz")
VOXEL_DATA = np.load(r"..\data\4persons\vox_data_video_multi_big.npz")

CENTER = None
RGB = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=np.float32)

from sklearn.cluster import KMeans



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
    for xyz in range(85):
        data.append([xyz, 0, 0])
        data.append([0, 0, xyz])
        colors.extend(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ]
        )
    for xyz in range(25):
        data.append([0, xyz, 0])
        colors.append(
            [1.0, 0.0, 0.0],
        )

    return data, colors

def set_centroid_pos():
    return CENTER, RGB


def set_voxel_positions(width, height, depth, i=0):
    # voxel_data = np.load(r"..\data\4persons\vox_data.npz")
    # voxel_data = np.load(r"..\data\vox_data.npz")
    voxel_data = VOXEL_DATA  # np.load(r"..\data\4persons\vox_data_video.npz")
    # i = 0
    true_vox, colors = (
        voxel_data[f"{i}_vox_coords"],
        np.median(voxel_data[f"{i}_vox_colors"], axis=0),
    )
    # true_vox, colors = (
    #     voxel_data["vox_coords"],
    #     voxel_data["colors"],
    # )
    global CENTER

    colors[:, 0], colors[:, 2] = colors[:, 2], colors[:, 0]
    XY = true_vox[:, [0, 1]].astype(np.float32)
    kmeans = KMeans(
        n_clusters=4, init=CENTER if (CENTER is not None and True) else "k-means++"
    )
    labels = kmeans.fit_predict(XY)
    CENTER = kmeans.cluster_centers_

    colors = (RGB[labels]) * 255

    vox = reorder_cv_gl(true_vox)
    non_occluded = np.ones(vox.shape[0], dtype=bool)
    # cam_pos, _ = get_cam_positions()
    # cam_pos = np.array(cam_pos)
    # print(vox.max(), vox.min(), cam_pos.max(), cam_pos.min())
    # dist = distance.cdist(cam_pos, vox.astype(cam_pos.dtype), "euclidean")
    # print(vox.shape)
    # for c in range(0, 4):
    #     srt = dist[c].argsort()
    #     non_occluded = raytrace_bresenham_3d(cam_pos[0].astype(int), vox[srt])
    #     break

    return vox[non_occluded], colors[non_occluded] / 255


def get_cam_pos(path):
    # https://stackoverflow.com/a/14693971

    voxel_data = VOXEL_DATA  # np.load(r"..\data\vox_data.npz")
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
        # *[get_cam_pos(f"../data/cam{i}/camera_properties.xml") for i in range(1, 5)]
        *[
            get_cam_pos(f"../data/4persons/{i}_camera_properties.xml")
            for i in range(1, 5)
        ]
    )
    return pos, [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


def get_cam_rotation_matrices():
    # Generates dummy camera rotation matrices, looking down 45 degrees towards the center of the room
    # TODO: You need to input the estimated camera rotation matrices (4x4) of the 4 cameras in the world coordinates.
    _, rvecs = zip(
        # *[get_cam_pos(f"../data/cam{i}/camera_properties.xml") for i in range(1, 5)]
        *[
            get_cam_pos(f"../data/4persons/{i}_camera_properties.xml")
            for i in range(1, 5)
        ]
    )
    cam_rotations = list()
    for i in range(1, 5):
        path = f"../data/4persons/{i}_camera_properties.xml"
        rvec = read_camera_properties(path)["r_matrix"]
        angle = np.linalg.norm(rvec)
        axis = rvec / angle

        # apply rotation to compensate for difference between OpenCV and OpenGL
        transform = glm.rotate(-0.5 * np.pi, [0, 0, 1]) * glm.rotate(
            -angle, glm.vec3(axis[0][0], axis[1][0], axis[2][0])
        )
        transform_to = glm.rotate(0.5 * np.pi, [1, 0, 0])
        transform_from = glm.rotate(-0.5 * np.pi, [1, 0, 0])
        cam_rotations.append(transform_to * transform * transform_from)

    return cam_rotations


def np_in(a, tgt) -> bool:
    """'Purest' numpy implementation for checking if subarray a is in array of arrays tgt.
    src: https://stackoverflow.com/questions/14766194/testing-whether-a-numpy-array-contains-a-given-row/14766816#14766816
    """
    return np.equal(a, tgt).all(1).any()


def raytrace_bresenham_3d(cam_pos, vox_pos):
    """Draws lines between voxel and camera. If that line intersects with a previous line, the voxel is occluded.

    Args:
        cam_pos (_type_): Position in 3D
        vox_pos (_type_): List of position in 3D, assumed to be sorted by distance to camera position

    Returns:
        _type_: Array of voxels that are not occluded
    """
    vox_occluded = np.zeros(vox_pos.shape[0])
    points = []
    from matplotlib import pyplot as plt
    from time import time
    from tqdm import trange

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    for i in trange(vox_pos.shape[0]):
        beam_pos = cam_pos.copy()
        # Initialize the list of points
        vox_point = vox_pos[i]
        # if not points:
        #     points = [vox_point.copy()]
        # elif np_in(vox_point, np.array(points)):
        #     continue
        # Calculate differences along each axis
        diff = vox_point - cam_pos
        abs_diff = np.abs(diff)
        # Determine the sign of increments along each axis
        signs_of_increments = np.ones((3), dtype=int)
        signs_of_increments[diff < 0] = -1

        # Choose the driving axis (axis with the greatest difference)
        driving_ax = np.argmax(abs_diff)
        axes = np.arange(3)
        ax1, ax2 = axes[axes != driving_ax]

        # ax.scatter(*cam_pos, c="red")

        err1, err2 = 2 * abs_diff[[ax1, ax2]] - abs_diff[driving_ax]
        while beam_pos[driving_ax] != vox_point[driving_ax]:
            # If the beam/ray intersects with a voxel, the voxel is occluded.
            if np_in(beam_pos, vox_pos):
                vox_occluded[i] = 1
                break
            if err1 >= 0:
                beam_pos[ax1] += signs_of_increments[ax1]
                err1 -= 2 * abs_diff[driving_ax]

            if err2 >= 0:
                beam_pos[ax2] += signs_of_increments[ax2]
                err2 -= 2 * abs_diff[driving_ax]

            err1 += 2 * abs_diff[ax1]
            err2 += 2 * abs_diff[ax2]
            beam_pos[driving_ax] += signs_of_increments[driving_ax]

            # ax.scatter(*beam_pos, c="green")

        # ax.scatter(vox_pos[i][ax1], vox_pos[i][ax2], c="blue")
        # ax.scatter(*vox_pos[i], c="blue")
        # plt.show()
        # break

    return vox_occluded == 0


def main():
    from scipy.spatial import distance

    vox, _ = set_voxel_positions(None, None, None)
    cam_pos, _ = get_cam_positions()
    cam_pos = np.array(cam_pos)
    print(vox.max(), vox.min(), cam_pos.max(), cam_pos.min())
    dist = distance.cdist(cam_pos, vox.astype(cam_pos.dtype), "euclidean")
    print(vox.shape)
    for c in range(0, 4):
        srt = dist[c].argsort()
        non_occluded = raytrace_bresenham_3d(cam_pos[0].astype(int), vox[srt])
        print(vox[non_occluded].shape)
        break


if __name__ == "__main__":
    main()
