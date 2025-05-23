import argparse
import numpy as np
import cv2
from matplotlib import pyplot as plt
from pathlib import Path
from time import time
from RDD.RDD import build
from RDD.RDD_helper import RDD_helper

def draw_matches(ref_points, dst_points, img0, img1, inliers=None):
    keypoints0 = [cv2.KeyPoint(p[0], p[1], 1) for p in ref_points]
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 1) for p in dst_points]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(ref_points))]
    matches_mask = inliers.ravel() if inliers is not None else None
    return cv2.drawMatches(img0, keypoints0, img1, keypoints1, matches, None,
                           matchesMask=matches_mask, matchColor=(0, 255, 0), singlePointColor=(0, 0, 255), flags=2)

def apply_mask(img, mask):
    return cv2.bitwise_and(img, img, mask=(mask > 0).astype(np.uint8) * 255)

def load_intrinsics(path):
    with open(path, 'r') as f:
        values = list(map(float, f.read().strip().split()))
    return np.array(values).reshape(3, 3)

def get_3d_centroid_from_mask_and_depth(mask, depth, K):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        raise ValueError("Mask is empty, can't compute 3D centroid.")
    depths = depth[ys, xs].astype(np.float32)
    valid = depths > 0
    if np.sum(valid) == 0:
        raise ValueError("All depth values in mask are zero.")
    z = np.median(depths[valid]) / 1000.0
    cx_img = np.mean(xs[valid])
    cy_img = np.mean(ys[valid])
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (cx_img - cx) * z / fx
    y = (cy_img - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)

def draw_object_axis(img, K, R, t, axis_length=0.1, label_axes=True, prefix=""):
    t = np.array(t).flatten().astype(np.float32)
    origin = np.array([[0, 0, 0]], dtype=np.float32)
    axis = np.array([
        [axis_length, 0, 0],
        [0, axis_length, 0],
        [0, 0, axis_length]
    ], dtype=np.float32)
    rvec, _ = cv2.Rodrigues(R)
    pts_2d, _ = cv2.projectPoints(np.vstack((origin, axis)), rvec, t, K, None)
    origin_2d = tuple(pts_2d[0].ravel().astype(int))
    x_2d = tuple(pts_2d[1].ravel().astype(int))
    y_2d = tuple(pts_2d[2].ravel().astype(int))
    z_2d = tuple(pts_2d[3].ravel().astype(int))
    h, w = img.shape[:2]
    if not (0 <= origin_2d[0] < w and 0 <= origin_2d[1] < h):
        print(f"Warning: Origin {origin_2d} is outside image bounds ({w}, {h})")
    cv2.line(img, origin_2d, x_2d, (0, 0, 255), 2)
    cv2.line(img, origin_2d, y_2d, (0, 255, 0), 2)
    cv2.line(img, origin_2d, z_2d, (255, 0, 0), 2)
    if label_axes:
        cv2.putText(img, f'{prefix}X', x_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img, f'{prefix}Y', y_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f'{prefix}Z', z_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return img

def main(args):
    scene_dir = Path(args.scene_dir)
    rgb_dir = scene_dir / "rgb"
    mask_dir = scene_dir / "masks"
    depth_dir = scene_dir / "depth"
    K_path = scene_dir / "cam_K.txt"

    img0 = cv2.imread(str(rgb_dir / "000000.jpg"))
    img1 = cv2.imread(str(rgb_dir / "000001.jpg"))
    mask0 = cv2.imread(str(mask_dir / "000000.png"), cv2.IMREAD_UNCHANGED)
    mask1 = cv2.imread(str(mask_dir / "000001.png"), cv2.IMREAD_UNCHANGED)
    depth0 = cv2.imread(str(depth_dir / "000000.png"), cv2.IMREAD_UNCHANGED)
    depth1 = cv2.imread(str(depth_dir / "000001.png"), cv2.IMREAD_UNCHANGED)

    assert img0 is not None and img1 is not None, "Images not loaded"
    assert mask0 is not None and mask1 is not None, "Masks not loaded"
    assert depth0 is not None and depth1 is not None, "Depth images not loaded"

    mask0_bin = (mask0 > 0).astype(np.uint8)
    mask1_bin = (mask1 > 0).astype(np.uint8)
    img0_masked = apply_mask(img0, mask0_bin)
    img1_masked = apply_mask(img1, mask1_bin)
    K = load_intrinsics(K_path)

    RDD_model = build(weights='./weights/RDD-v2.pth')
    RDD_model.eval()
    RDD_wrap = RDD_helper(RDD_model)

    start = time()
    mkpts_0, mkpts_1, conf = RDD_wrap.match_dense(img0_masked, img1_masked, resize=1024)
    print(f"Found {len(mkpts_0)} matches in {time() - start:.2f} seconds")

    pts0 = mkpts_0.astype(np.float32)
    pts1 = mkpts_1.astype(np.float32)
    E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    # Refine matches using inlier mask
    inlier_mask = mask.ravel().astype(bool)
    pts0 = pts0[inlier_mask]
    pts1 = pts1[inlier_mask]
    # Replace recoverPose with SVD-based refinement
    def svd_pose_alignment(P1, P2):
        centroid_P1 = np.mean(P1, axis=0)
        centroid_P2 = np.mean(P2, axis=0)
        Q1 = P1 - centroid_P1
        Q2 = P2 - centroid_P2
        H = Q1.T @ Q2
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = centroid_P2 - R @ centroid_P1
        return R, t

    # Backproject 2D inlier keypoints to 3D using depth
    def backproject_2d_to_3d(pts, depth_img, K):
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        pts_3d = []
        for (u, v) in pts.astype(int):
            if 0 <= v < depth_img.shape[0] and 0 <= u < depth_img.shape[1]:
                z = depth_img[v, u] / 1000.0
                if z > 0:
                    x = (u - cx) * z / fx
                    y = (v - cy) * z / fy
                    pts_3d.append([x, y, z])
        return np.array(pts_3d, dtype=np.float32)

    pts3d_0 = backproject_2d_to_3d(pts0, depth0, K)
    pts3d_1 = backproject_2d_to_3d(pts1, depth1, K)
    min_pts = min(len(pts3d_0), len(pts3d_1))
    pts3d_0 = pts3d_0[:min_pts]
    pts3d_1 = pts3d_1[:min_pts]

    R, t = svd_pose_alignment(pts3d_0, pts3d_1)

    # Evaluate residual alignment error
    pts3d_0_transformed = (R @ pts3d_0.T).T + t
    residuals = np.linalg.norm(pts3d_1 - pts3d_0_transformed, axis=1)
    rmse = np.sqrt(np.mean(residuals ** 2))
    print("RMSE alignment error:", rmse)
    inliers = np.ones((min_pts, 1), dtype=np.uint8)  # assume all are inliers

    print("Relative Camera Rotation R:\n", R)
    print("Relative Camera Translation t:\n", t.T)
    print("Inlier count:", np.count_nonzero(inliers))

    # Save relative pose as 4x4 RT matrix
    RT = np.eye(4, dtype=np.float32)
    RT[:3, :3] = R
    RT[:3, 3] = t.flatten()
    np.savetxt(scene_dir / "relative_pose_rt.txt", RT, fmt="%.6f")

    t_obj_0 = get_3d_centroid_from_mask_and_depth(mask0_bin, depth0, K).reshape(1, 3)
    t_obj_1 = get_3d_centroid_from_mask_and_depth(mask1_bin, depth1, K).reshape(1, 3)

    import pdb; pdb.set_trace()
    t_est = t_obj_1 - t_obj_0
    
    canvas = draw_matches(mkpts_0, mkpts_1, img0_masked, img1_masked, inliers)

    

    axis_length = 0.1

    img0_axis = draw_object_axis(img0.copy(), K, np.eye(3), t_obj_0, axis_length=axis_length, label_axes=True, prefix="Ref")
    img1_axis = draw_object_axis(img1.copy(), K, R.T, t_obj_1, axis_length=axis_length, label_axes=True, prefix="Rel")

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs[0].imshow(img0_axis[..., ::-1])
    axs[0].set_title("Frame 0: Reference Pose")
    axs[1].imshow(img1_axis[..., ::-1])
    axs[1].set_title("Frame 1: Relative Pose")
    axs[2].imshow(canvas[..., ::-1])
    axs[2].set_title("Object Mask Matches (Green: Inliers, Red: Outliers)")
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True, help="Path to scene dir containing rgb/, masks/, depth/, cam_K.txt")
    parser.add_argument("--mode", default="match_dense", choices=["match", "match_dense", "match_lg", "match_3rd_party"],
                        help="RDD mode to use for matching")
    args = parser.parse_args()
    main(args)
