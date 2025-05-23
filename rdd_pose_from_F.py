import numpy as np
import cv2
from pathlib import Path
from scipy.io import loadmat
import torch
from time import time
from RDD.RDD import build
from RDD.RDD_helper import RDD_helper


# Assuming RDD_helper and other utilities are defined elsewhere
# from your_module import RDD_helper, build, load_intrinsics

def apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=(mask > 0).astype(np.uint8) * 255)

def load_intrinsics(path, img_shape):
    try:
        with open(path, 'r') as f:
            values = list(map(float, f.read().strip().split()))
        K = np.array(values).reshape(3, 3)
        if not (100 < K[0,0] < 5000 and 100 < K[1,1] < 5000 and
                0 < K[0,2] < img_shape[1] and 0 < K[1,2] < img_shape[0]):
            print("Warning: Invalid intrinsics, using default K")
            raise ValueError
    except:
        print("Warning: Failed to load intrinsics, using default K")
        K = np.array([[600, 0, img_shape[1]/2],
                      [0, 600, img_shape[0]/2],
                      [0, 0, 1]], dtype=np.float32)
    return K

def filter_correspondences(pts0, pts1, conf, mask0_bin, mask1_bin, conf_threshold=0.9):
    """
    Filter correspondences to keep only those within masks and above confidence threshold.
    
    Args:
        pts0, pts1: Nx2 arrays of corresponding points.
        conf: N array of confidence scores.
        mask0_bin, mask1_bin: Binary masks (H x W).
        conf_threshold: Minimum confidence for keeping a match.
    
    Returns:
        Filtered pts0, pts1, and conf arrays.
    """
    valid = []
    for i in range(len(pts0)):
        x0, y0 = int(pts0[i, 0]), int(pts0[i, 1])
        x1, y1 = int(pts1[i, 0]), int(pts1[i, 1])
        if (0 <= y0 < mask0_bin.shape[0] and 0 <= x0 < mask0_bin.shape[1] and
            0 <= y1 < mask1_bin.shape[0] and 0 <= x1 < mask1_bin.shape[1]):
            if (mask0_bin[y0, x0] > 0 and mask1_bin[y1, x1] > 0 and conf[i] > conf_threshold):
                valid.append(i)
    return pts0[valid], pts1[valid], conf[valid]

def estimate_pose(pts0, pts1, K):
    """
    Estimate relative pose (R, t) from correspondences and intrinsic matrix.
    
    Args:
        pts0, pts1: Nx2 arrays of corresponding points.
        K: 3x3 intrinsic matrix.
    
    Returns:
        R: 3x3 rotation matrix.
        t: 3x1 translation vector (up to scale).
    """
    # Compute Fundamental Matrix with RANSAC
    F, inlier_mask = cv2.findFundamentalMat(
        pts0, pts1, method=cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99
    )
    if F is None:
        raise ValueError("Failed to compute fundamental matrix")

    # Compute Essential Matrix
    E = K.T @ F @ K

    # Enforce Essential Matrix constraints
    U, S, Vt = np.linalg.svd(E)
    S = np.array([S[0], S[0], 0])  # Force two equal singular values and one zero
    E = U @ np.diag(S) @ Vt

    # Decompose Essential Matrix
    R1, R2, t = cv2.decomposeEssentialMat(E)

    # Disambiguate the correct (R, t)
    def check_pose(R, t, pts0, pts1, K, inlier_mask):
        P1 = K @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K @ np.hstack((R, t))
        pts0_inliers = pts0[inlier_mask.ravel() == 1]
        pts1_inliers = pts1[inlier_mask.ravel() == 1]
        points_4d = cv2.triangulatePoints(
            P1, P2,
            pts0_inliers.T[:2], pts1_inliers.T[:2]
        )
        points_3d = points_4d[:3] / points_4d[3]
        points_3d_cam2 = (R @ points_3d + t).T
        return np.sum(points_3d[2, :] > 0) + np.sum(points_3d_cam2[:, 2] > 0)

    best_score = 0
    best_R, best_t = None, None
    for R, t in [(R1, t), (R1, -t), (R2, t), (R2, -t)]:
        score = check_pose(R, t, pts0, pts1, K, inlier_mask)
        if score > best_score:
            best_score = score
            best_R, best_t = R, t

    return best_R, best_t

def main(args):
    """
    Main function to estimate relative pose between two frames and visualize results.

    Args:
        args: Command-line arguments with scene_dir and mode.
    """
    scene_dir = Path(args.scene_dir)
    mat_dir = scene_dir / "mat"
    rgb_dir = scene_dir / "rgb"
    mask_dir = scene_dir / "masks"
    depth_dir = scene_dir / "depth"
    K_path = scene_dir / "cam_K.txt"

    names = [('000000', '000001')]

    for _name0, _name1 in names:
        name0 = f"{_name0}.png"
        name1 = f"{_name1}.png"
        img0 = cv2.imread(str(rgb_dir / name0))
        img1 = cv2.imread(str(rgb_dir / name1))
        mask0 = cv2.imread(str(mask_dir / name0), cv2.IMREAD_UNCHANGED)
        mask1 = cv2.imread(str(mask_dir / name1), cv2.IMREAD_UNCHANGED)
        depth0 = cv2.imread(str(depth_dir / name0), cv2.IMREAD_UNCHANGED)
        depth1 = cv2.imread(str(depth_dir / name1), cv2.IMREAD_UNCHANGED)
        mat1 = loadmat(str(mat_dir / f"{_name0}.mat"))
        mat2 = loadmat(str(mat_dir / f"{_name1}.mat"))
        T_ca = list(mat1['obj_pose'])[0]
        T_cb = list(mat2['obj_pose'])[0]
        RT_cam1 = mat1['RT_camera']
        RT_cam2 = mat2['RT_camera']

        K = load_intrinsics(K_path, img0.shape)
        print("Camera Intrinsics K:\n", K)

        assert img0 is not None and img1 is not None, "Images not loaded"
        assert mask0 is not None and mask1 is not None, "Masks not loaded"
        assert depth0 is not None and depth1 is not None, "Depth images not loaded"

        mask0_bin = (mask0 > 0).astype(np.uint8)
        mask1_bin = (mask1 > 0).astype(np.uint8)
        img0_masked = apply_mask(img0, mask0_bin)
        img1_masked = apply_mask(img1, mask1_bin)

        RDD_model = build(weights='./weights/RDD-v2.pth')
        RDD_model.eval()
        RDD_wrap = RDD_helper(RDD_model)

        start = time()
        mkpts_0, mkpts_1, conf = RDD_wrap.match_dense(img0_masked, img1_masked, resize=1024)
        print(f"Found {len(mkpts_0)} matches in {time() - start:.2f} seconds")

        # Filter correspondences
        pts0, pts1, conf = filter_correspondences(mkpts_0, mkpts_1, conf, mask0_bin, mask1_bin, conf_threshold=0.9)
        print(f"Retained {len(pts0)} valid matches after filtering")

        # Ensure enough correspondences
        if len(pts0) < 8:
            print(f"Warning: Only {len(pts0)} valid correspondences. Need at least 8.")
            continue

        # Estimate relative pose
        R, t = estimate_pose(pts0, pts1, K)
        print("Estimated Rotation Matrix:\n", R)
        print("Estimated Translation Vector (up to scale):\n", t)

        # Optional: Visualize inlier matches
        inlier_mask = cv2.findFundamentalMat(
            pts0, pts1, method=cv2.FM_RANSAC, ransacReprojThreshold=1.0, confidence=0.99
        )[1]
        inlier_pts0 = pts0[inlier_mask.ravel() == 1]
        inlier_pts1 = pts1[inlier_mask.ravel() == 1]
        img_matches = cv2.drawMatches(
            img0, [cv2.KeyPoint(x, y, 1) for x, y in inlier_pts0],
            img1, [cv2.KeyPoint(x, y, 1) for x, y in inlier_pts1],
            [cv2.DMatch(i, i, 1) for i in range(len(inlier_pts0))],
            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )
        cv2.imwrite(str(scene_dir / f"matches_{_name0}_{_name1}.png"), img_matches)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, required=True, help="Path to scene directory")
    parser.add_argument("--mode", type=str, default="test", help="Mode (train/test)")
    args = parser.parse_args()
    main(args)