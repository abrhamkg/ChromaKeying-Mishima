# Enter your code here
import os
from itertools import product, permutations

import cv2
import numpy as np
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from scipy import optimize
from scipy import spatial
from tqdm import tqdm


draw = False
frame_copy = None
MAX_POSSIBLE_DISTANCE = 441.67  # The maximum possible distance from the mean background color [255, 255, 255]

# A list of all the vectors that are used to define hexoctahedron in Mishima et al.
PYR_EDGE_VECTORS = np.array([
    (1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0),
    (0, -1, 0), (0, 0, - 1), (0.707, 0.707, 0), (0.707, -0.707, 0),
    (-0.707, 0.707, 0), (-0.707, -0.707, 0), (0, 0.707, 0.707), (0, 0.707, -0.707),
    (0, -0.707, 0.707), (0, -0.707, -0.707), (0.707, 0, 0.707), (-.707, 0, 0.707),
    (0.707, 0, -0.707), (-0.707, 0, -0.707), (0.577, 0.577, 0.577), (0.577, 0.577, -0.577),
    (.577, -0.577, 0.577), (0.577, -0.577, -0.577), (-0.577, 0.577, 0.577), (-0.577, 0.577, -0.577),
    (-0.577, -0.577, 0.577), (-0.577, -0.577, -0.577),
])

# A list of triangular pyramids edge vectors for the hexoctahedron in Mishima et al.
TRIANGULAR_PYRAMIDS = np.array([
    (18, 0, 6), (18, 6, 1), (18, 1, 10), (18, 10, 2), (18, 2, 14), (18, 14, 0), (22, 1, 8), (22, 8, 3), (22, 3, 15),
    (22, 15, 2), (22, 2, 10), (22, 10, 1),     (24, 3, 9), (24, 9, 4), (24, 4, 12), (24, 12, 2), (24, 2, 15),
    (24, 15, 3), (20, 4, 7), (20, 7, 0), (20, 0, 14), (20, 14, 2), (20, 2, 12), (20, 12, 4), (19, 0, 16), (19, 16, 5),
    (19, 5, 11), (19, 11, 1), (19, 1, 6), (19, 6, 0), (23, 1, 11), (23, 11, 5), (23, 5, 17), (23, 17, 3), (23, 3, 8),
    (23, 8, 1), (25, 3, 17), (25, 17, 5), (25, 5, 13), (25, 13, 4), (25, 4, 9), (25, 9, 3), (21, 4, 13), (21, 13, 5),
    (21, 5, 16), (21, 16, 0), (21, 0, 7), (21, 7, 4),
])

# A dictionary that maps a 9-bit integer to one of the 48-hexoctahedrons
LUT_dict = {
    471: 0, 503: 1, 501: 2, 509: 3, 477: 4, 479: 5, 244: 6, 228: 7,
    236: 8, 237: 9, 253: 10, 245: 11, 104: 12, 72: 13, 73: 14, 77: 15,
    109: 16, 108: 17, 331: 18, 347: 19, 351: 20, 349: 21, 333: 22, 329: 23,
    403: 24, 402: 25, 434: 26, 438: 27, 439: 28, 407: 29, 182: 30, 178: 31,
    162: 32, 160: 33, 164: 34, 180: 35, 32: 36, 34: 37, 2: 38, 10: 39,
    8: 40, 40: 41, 266: 42, 258: 43, 274: 44, 275: 45, 283: 46, 267: 47,
}

LUT = np.vectorize(LUT_dict.get)

LENGTHS_IN = np.array(
    [0.0] * len(PYR_EDGE_VECTORS)
)  # Holds the lengths of each edge vector for inner hexoctahedron

LENGTHS_OUT = np.array(
    [MAX_POSSIBLE_DISTANCE] * len(PYR_EDGE_VECTORS)
)  # Holds the lengths of each edge vector for inner hexoctahedron

PLANE_PARAMS_IN = np.zeros(
    (len(TRIANGULAR_PYRAMIDS), 5)
)  # Holds the hexoctahedron boundary planes for the inner hexoctahedron

PLANE_PARAMS_OUT = np.zeros(
    (len(TRIANGULAR_PYRAMIDS), 5)
)  # Holds the hexoctahedron boundary planes for the inner hexoctahedron


def identification_function_parameter_register(centered_data):
    """
    Takes data and finds a triangular pyramid the data belongs o
    :param centered_data: An (N, 3) numpy.ndarray for which we want to find the N triangular pyramids that contain each
    data point.
    :return: An (N,) numpy.ndarray of the indices of the triangular pyramid each data point belongs to
    """
    if centered_data.dtype != np.float32:
        raise TypeError(f"`centered_data` expected to have dtype np.float32")

    b, g, r = centered_data.T
    selector = np.zeros((centered_data.shape[0],), dtype=np.uint16)
    selector |= (((r >= 0) << 8).astype(np.uint16) |
                 ((g >= 0) << 7).astype(np.uint16) |
                 ((b >= 0) << 6).astype(np.uint16))
    # Change some of the comparisons to > instead of >= to ensure [0, 0, 0] doesn't get translated to 511
    diff_gr = (((g - r) > 0) << 5).astype(np.uint16)
    sum_gr = (((g + r) >= 0) << 4).astype(np.uint16)
    diff_bg = (((b - g) > 0) << 3).astype(np.uint16)
    sum_bg = (((b + g) >= 0) << 2).astype(np.uint16)
    diff_rb = (((r - b) >= 0) << 1).astype(np.uint16)
    sum_rb = ((r + b) >= 0).astype(np.uint16)

    selector |= (diff_gr | diff_rb | diff_bg | sum_gr | sum_rb | sum_bg)

    return LUT(selector)


def get_edge_vectors(centered_data):
    """
    Finds the edge vectors of the triangular pyramid each data point is contained in
    :param centered_data: An (N, 3) numpy.ndarray for which we want to find the edge vectors of the N triangular
    pyramids that contain each data point.
    :return: A tuple of the edge vectors for each datapoint, the indices of the edge of each data point, and the
    indices of the triangular pyramid each data point belongs to.
    """
    # Find the traingular pyramid each data point belongs to
    pyr_indices = identification_function_parameter_register(centered_data)

    # Find the unit vectors that define each triangular pyramid found above
    edge_indices = TRIANGULAR_PYRAMIDS[pyr_indices]
    edge_vectors = PYR_EDGE_VECTORS[edge_indices.flatten()]
    edge_vectors = edge_vectors.reshape((centered_data.shape[0], -1))
    return edge_vectors, edge_indices, pyr_indices


def get_new_edge_lengths(centered_data, fill_value=0.0):
    edge_vectors, edge_indices, pyr_indices = get_edge_vectors(centered_data)

    # Find a plane that minimizes sum of distances to each crossing point of edge vector
    outputs = Parallel(n_jobs=-1)(delayed(compute_plane)(d, e) for d, e in zip(centered_data, edge_vectors))
    planes, edge_vec_lengths = zip(*outputs)

    edge_vec_lengths = np.array(edge_vec_lengths).flatten()
    planes = np.vstack(planes)

    edge_indices = edge_indices.reshape((1, -1))
    unique = np.unique(edge_indices)
    matches = unique[:, np.newaxis] == edge_indices
    edge_vec_lengths = np.repeat(edge_vec_lengths[np.newaxis, :], matches.shape[0], axis=0)
    edge_vec_lengths = np.where(matches, edge_vec_lengths, fill_value)
    return edge_vec_lengths, unique, planes, pyr_indices


def determine_inner_hexoctahedron(centered_data):
    edge_vec_lengths, unique, planes, pyr_indices = get_new_edge_lengths(centered_data)
    max_edge_vec_len = edge_vec_lengths.max(axis=1)
    LENGTHS_IN[unique] = max_edge_vec_len


def determine_outer_hexoctahedron(centered_data):
    edge_vec_lengths, unique, planes, pyr_indices = get_new_edge_lengths(centered_data,
                                                                         fill_value=MAX_POSSIBLE_DISTANCE)
    min_edge_vec_len = edge_vec_lengths.min(axis=1)
    LENGTHS_OUT[unique] = min_edge_vec_len


def compute_plane(p, pyr_vecs, max_iter=1000, atol=2e-1, rtol=1e-3):
    """
    Uses gradient descent to compute a plane that passes through point `p` and minimizes the sum of lengths of the crossing
    points of the vectors (in `pyr_vecs`) with the plane.

    :param p: The point which the computed plane must pass through
    # :param pyr_vecs: A list of vectors that define a triangular pyramid
    :param pyr_vecs: A numpy.ndarray of nine values that correspond to the three vectors that define a
    triangular pyramid. First three values belong to the first vector and so on.
    :param max_iter:
    :param atol:
    :return:
    """
    r, g, b = p
    # v1 = pyr_vecs[0].astype(np.float32)
    # v2 = pyr_vecs[1].astype(np.float32)
    # v3 = pyr_vecs[2].astype(np.float32)
    v1, v2, v3 = [pyr_vecs[i:i + 3].astype(np.float32) for i in range(0, 7, 3)]

    # Initialize the plane normal to the unit vector whose heading is the sum of the vectors in pyr_vecs
    n = (v1 + v2 + v3)
    n /= np.linalg.norm(n)

    nr, ng, nb = n
    d = -(np.dot(n, p))

    t1 = -d / (np.dot(n, v1))
    t2 = -d / (np.dot(n, v2))
    t3 = -d / (np.dot(n, v3))

    r1, g1, b1 = v1
    r2, g2, b2 = v2
    r3, g3, b3 = v3

    lr = 1e-2
    best_param = n, d
    best_t = t1, t2, t3
    best_sum = t1 + t2 + t3
    prev_tsum = 0.0
    prev_n = n
    prev_t = t1, t2, t3
    for i in range(max_iter):
        tsum = t1 + t2 + t3

        if tsum < best_sum:
            best_param = n, d
            best_sum = tsum
            best_t = t1, t2, t3

        abs_diff = abs(tsum - prev_tsum)
        if tsum - prev_tsum < 0 and abs_diff < atol:
            break

        # if abs_diff / min(tsum, prev_tsum) < rtol:
        #     break

        prev_tsum = tsum
        # print(t1, t2, t3, nr, ng, nb)
        term1 = (b3 * nb + g3 * ng + nr * r3)
        term2 = (b2 * nb + g2 * ng + nr * r2)
        term3 = (b1 * nb + g1 * ng + nr * r1)
        term4 = (b * nb + g * ng + nr * r)

        dfdnr = r / (term1 + term2 + term3) - \
                term4 * (r1 / (term3) ** 2 - r2 / (term2) ** 2 - r3 / (term1) ** 2)

        dfdng = g / (term1 + term2 + term3) - \
                term4 * (g1 / (term3) ** 2 - g2 / (term2) ** 2 - g3 / (term1) ** 2)

        dfdnb = b / (term1 + term2 + term3) - \
                term4 * (b1 / (term3) ** 2 - b2 / (term2) ** 2 - b3  / (term1) ** 2)

        # print(dfdnr, dfdng, dfdnb)
        nr -= lr * dfdnr
        ng -= lr * dfdng
        nb -= lr * dfdnb
        n = np.array([nr, ng, nb])
        n /= np.linalg.norm(n)

        d = -(np.dot(n, p))

        # TODO: combine the following operations into one division and one dot product
        # t1, t2, t3 = -d / np.dot(n, np.array([v1, v2, v3]))
        t1 = -d / (np.dot(n, v1))
        t2 = -d / (np.dot(n, v2))
        t3 = -d / (np.dot(n, v3))

        # If any of the ts are less than zero roll back latest parameters and decrease the learning rate by an octave
        if t1 < 0 or t2 < 0 or t3 < 0:
            n = prev_n
            lr *= 0.5
            t1, t2, t3 = prev_t
        else:
            prev_n = n
            prev_t = t1, t2, t3
        # vp = t1 * v1 - t2 * v2
        # vpp = t3 * v3 - t2 * v2
        # nc = np.cross(vp, vpp)
        # nc /= np.linalg.norm(nc)

    bparam_list = best_param[0].tolist()
    bparam_list.append(best_param[1])
    best_param = np.array(bparam_list)

    return best_param, best_t


def compute_all_plane_params():
    """
    Uses the already computed edge lengths and plane orthogonal unit vector to determine the "intercept" of the
    48-planes that define each of the inner and outer hexoctahedrons.
    Note: Ensure the `determine_inner_hexoctahedron` and `determine_outer_hexoctahedron` are run before calling this
    function.
    :return:
    """
    all_edge_indices = TRIANGULAR_PYRAMIDS.flatten()
    all_edges_all_planes = PYR_EDGE_VECTORS[all_edge_indices]

    all_edges_all_planes_in = all_edges_all_planes * LENGTHS_IN[all_edge_indices][:, np.newaxis]
    all_edges_all_planes_out = all_edges_all_planes * LENGTHS_OUT[all_edge_indices][:, np.newaxis]

    all_edges_all_planes_in = all_edges_all_planes_in.reshape((-1, 3, 3))

    e1, e2, e3 = all_edges_all_planes_in[:, 0, :], all_edges_all_planes_in[:, 1, :], all_edges_all_planes_in[:, 2, :]

    v1 = e2 - e1
    v2 = e3 - e1
    n_in = np.cross(v1, v2)
    n_in /= np.linalg.norm(n_in, 2, axis=1, keepdims=True)

    all_edges_all_planes_out = all_edges_all_planes_out.reshape((-1, 3, 3))

    e1, e2, e3 = all_edges_all_planes_out[:, 0, :], all_edges_all_planes_out[:, 1, :], all_edges_all_planes_out[:, 2, :]
    v1 = e2 - e1
    v2 = e3 - e1
    n_out = np.cross(v1, v2)
    n_out /= np.linalg.norm(n_out, 2, axis=1, keepdims=True)

    first_edge_indices = TRIANGULAR_PYRAMIDS[:, 0]
    points_in = PYR_EDGE_VECTORS[first_edge_indices] * LENGTHS_IN[first_edge_indices][:, np.newaxis]
    points_out = PYR_EDGE_VECTORS[first_edge_indices] * LENGTHS_OUT[first_edge_indices][:, np.newaxis]

    d_in = (n_in * points_in).sum(axis=1)
    d_out = (n_out * points_out).sum(axis=1)

    PLANE_PARAMS_IN[:, :3] = n_in
    PLANE_PARAMS_OUT[:, :3] = n_out

    PLANE_PARAMS_IN[:, 3] = d_in
    PLANE_PARAMS_OUT[:, 3] = d_out


def compute_alpha(centered_data, eps=1e-8):
    """
    Assuming all plane parameters are computed this computes the alpha matte for a given set of pixels.
    :param vec:
    :return:
    """
    edge_vectors, edge_indices, pyr_indices = get_edge_vectors(centered_data)

    # When using constant background (e.g green screen) inner hexoctahedron will have a lot zero-length edges
    # Thus, handling NaNs is very important
    dprod_in = (np.nan_to_num(PLANE_PARAMS_IN[pyr_indices, :3]) * centered_data).sum(axis=1)
    dprod_out = (PLANE_PARAMS_OUT[pyr_indices, :3] * centered_data).sum(axis=1)

    tin = np.nan_to_num(PLANE_PARAMS_IN[pyr_indices, 3] / (dprod_in + eps))
    tout = (PLANE_PARAMS_OUT[pyr_indices, 3] / (dprod_out + eps))

    # Ensure data is in range [0, 1]
    alpha = np.clip((1 - tin) / (tout - tin), 0, 1)
    return alpha


def compute_vlahos_mask(bgr_image, tolerance_a, tolerance_b):
    """
    Computes an alpha matte from a given 2D array of BGR pixels.
    :param pixels:
    :return:
    """
    b, g, r = cv2.split(bgr_image)
    return 1 - tolerance_a * (g - tolerance_b * b)




