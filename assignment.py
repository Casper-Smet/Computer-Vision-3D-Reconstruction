import random
import numpy as np
import cv2
import glm
from scipy.spatial import distance


block_size = 1.0
# VOXEL_DATA = np.load(r"..\data\4persons\vox_data_video.npz")
# VOXEL_DATA = np.load(r"..\data\4persons\vox_data_video_multi.npz")
VOXEL_DATA = np.load(r"..\data\4persons\vox_data_video_multi_big.npz")
HIST_DATA = np.load(r"..\data\4persons\cluster_hist.npz")
GMM_DATA = np.load(r"..\data\4persons\cluster_gmm.npz")

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
                # [x * block_size - width / 2, -block_size, z * block_size - depth / 2]
                [x * block_size, -block_size, z * block_size]
            )
            colors.append([1.0, 1.0, 1.0] if (x + z) % 2 == 0 else [0, 0, 0])

    return data, colors


def set_centroid_pos():
    return CENTER, RGB


def set_voxel_positions(width, height, depth, i=0, color_mode=0):
    # voxel_data = np.load(r"..\data\4persons\vox_data.npz")
    # voxel_data = np.load(r"..\data\vox_data.npz")
    voxel_data = VOXEL_DATA  # np.load(r"..\data\4persons\vox_data_video.npz")

    true_vox, colors = (
        voxel_data[f"{i}_vox_coords"],
        np.median(voxel_data[f"{i}_vox_colors"], axis=0),
    )

    global CENTER

    colors[:, 0], colors[:, 2] = colors[:, 2], colors[:, 0]
    XY = true_vox[:, [0, 1]].astype(np.float32)
    kmeans = KMeans(
        n_clusters=4, init=CENTER if (CENTER is not None and True) else "k-means++"
    )
    labels = kmeans.fit_predict(XY)
    CENTER = kmeans.cluster_centers_
    if color_mode == 1:
        colors = (RGB[labels]) * 255
    elif color_mode == 2:
        colors = RGB[HIST_DATA[f"{i}_label"]] * 255
    elif color_mode == 3:
        colors = RGB[GMM_DATA[f"{i}_label"]] * 255

    vox = reorder_cv_gl(true_vox)

    return vox, colors / 255


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
