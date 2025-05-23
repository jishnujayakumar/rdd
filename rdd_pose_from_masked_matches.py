import argparse  # For parsing command-line arguments
import cv2  # OpenCV for image processing and computer vision tasks
from matplotlib import pyplot as plt  # For 2D visualization
from pathlib import Path  # For handling file paths in a cross-platform way
from time import time  # For measuring execution time
from RDD.RDD import build  # Custom module to build the RDD model for feature matching
from RDD.RDD_helper import RDD_helper  # Helper class for RDD model operations
from scipy.io import loadmat  # For loading MATLAB .mat files
from transforms3d.quaternions import quat2mat  # For converting quaternions to rotation matrices
import numpy as np  # For numerical computations and array operations


def draw_matches(ref_points, dst_points, img0, img1, inliers=None):
    """
    Draw matching keypoints between two images.

    Args:
        ref_points (np.ndarray): Keypoints in the first image (x, y coordinates).
        dst_points (np.ndarray): Corresponding keypoints in the second image.
        img0 (np.ndarray): First RGB image (reference).
        img1 (np.ndarray): Second RGB image (destination).
        inliers (np.ndarray, optional): Mask indicating inlier matches (1 for inlier, 0 for outlier).

    Returns:
        np.ndarray: Image with drawn matches (green lines for inliers, red dots for outliers).
    """
    # Convert points to OpenCV KeyPoint objects (required for cv2.drawMatches)
    keypoints0 = [cv2.KeyPoint(p[0], p[1], 1) for p in ref_points]
    keypoints1 = [cv2.KeyPoint(p[0], p[1], 1) for p in dst_points]
    # Create a list of matches (each point in ref_points corresponds to the same index in dst_points)
    matches = [cv2.DMatch(i, i, 0) for i in range(len(ref_points))]
    # Convert inliers to a flat array for masking (None if no inliers provided)
    matches_mask = inliers.ravel() if inliers is not None else None
    # Draw matches: green lines for inliers, red dots for keypoints (flags=2 avoids drawing unmatched points)
    return cv2.drawMatches(img0, keypoints0, img1, keypoints1, matches, None,
                           matchesMask=matches_mask, matchColor=(0, 255, 0), singlePointColor=(0, 0, 255), flags=2)


def apply_mask(img, mask):
    """
    Apply a binary mask to an RGB image to isolate the object.

    Args:
        img (np.ndarray): Input RGB image.
        mask (np.ndarray): Binary mask (non-zero where the object is).

    Returns:
        np.ndarray: Masked image (non-object pixels set to black).
    """
    # Convert mask to binary (1 where mask > 0, 0 elsewhere), scale to 255 for OpenCV
    binary_mask = (mask > 0).astype(np.uint8) * 255
    # Apply mask to keep only object pixels in the image
    return cv2.bitwise_and(img, img, mask=binary_mask)


def load_intrinsics(path):
    """
    Load camera intrinsic parameters from a text file.

    Args:
        path (str): Path to the text file containing 9 values (3x3 matrix).

    Returns:
        np.ndarray: 3x3 camera intrinsic matrix (K).
    """
    # Read the file and convert space-separated values to floats
    with open(path, 'r') as f:
        values = list(map(float, f.read().strip().split()))
    # Reshape into 3x3 matrix (fx, 0, cx; 0, fy, cy; 0, 0, 1)
    return np.array(values).reshape(3, 3)


def get_3d_centroid_from_mask_and_depth(mask, depth, K):
    """
    Compute the 3D centroid of an object using its mask and depth image.

    Args:
        mask (np.ndarray): Binary mask of the object (non-zero where object exists).
        depth (np.ndarray): Depth image (in millimeters).
        K (np.ndarray): 3x3 camera intrinsic matrix.

    Returns:
        np.ndarray: 3D centroid coordinates (x, y, z) in meters, in camera coordinate system.
    """
    # Find pixel coordinates (x, y) where the mask is non-zero
    ys, xs = np.where(mask > 0)
    # Check if the mask is empty
    if len(xs) == 0:
        raise ValueError("Mask is empty, can't compute 3D centroid.")
    # Extract depth values at masked pixels
    depths = depth[ys, xs].astype(np.float32)
    # Filter valid depth values (non-zero)
    valid = depths > 0
    if np.sum(valid) == 0:
        raise ValueError("All depth values in mask are zero.")
    # Compute median depth (in meters) to reduce noise
    z = np.median(depths[valid]) / 1000.0
    # Compute mean pixel coordinates within the mask
    cx_img = np.mean(xs[valid])
    cy_img = np.mean(ys[valid])
    # Extract camera intrinsics (focal lengths and principal point)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    # Convert pixel coordinates to 3D camera coordinates (x, y, z)
    x = (cx_img - cx) * z / fx
    y = (cy_img - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)


def draw_object_axis(img, K, R, t, axis_length=0.1, label_axes=True, prefix=""):
    """
    Draw 3D coordinate axes projected onto a 2D image.

    Args:
        img (np.ndarray): Input RGB image to draw on.
        K (np.ndarray): 3x3 camera intrinsic matrix.
        R (np.ndarray): 3x3 rotation matrix of the object.
        t (np.ndarray): 3x1 translation vector of the object (centroid).
        axis_length (float): Length of the axes in meters (default: 0.1).
        label_axes (bool): Whether to label the axes (X, Y, Z).
        prefix (str): Prefix for axis labels (e.g., "Ref" or "Rel").

    Returns:
        np.ndarray: Image with drawn axes (X: red, Y: green, Z: blue).
    """
    # Ensure translation is a flat array
    t = np.array(t).flatten().astype(np.float32)
    # Define origin and axis endpoints in 3D (in meters)
    origin = np.array([[0, 0, 0]], dtype=np.float32)
    axis = np.array([
        [axis_length, 0, 0],  # X-axis
        [0, axis_length, 0],  # Y-axis
        [0, 0, axis_length]   # Z-axis
    ], dtype=np.float32)
    # Convert rotation matrix to Rodrigues vector for OpenCV
    rvec, _ = cv2.Rodrigues(R)
    # Project 3D points (origin and axis ends) to 2D image plane
    pts_2d, _ = cv2.projectPoints(np.vstack((origin, axis)), rvec, t, K, None)
    # Convert projected points to integer tuples for drawing
    origin_2d = tuple(pts_2d[0].ravel().astype(int))
    x_2d = tuple(pts_2d[1].ravel().astype(int))
    y_2d = tuple(pts_2d[2].ravel().astype(int))
    z_2d = tuple(pts_2d[3].ravel().astype(int))
    # Get image dimensions
    h, w = img.shape[:2]
    # Warn if origin is outside image bounds
    if not (0 <= origin_2d[0] < w and 0 <= origin_2d[1] < h):
        print(f"Warning: Origin {origin_2d} is outside image bounds ({w}, {h})")
    # Draw axes: X (red), Y (green), Z (blue)
    cv2.line(img, origin_2d, x_2d, (0, 0, 255), 2)
    cv2.line(img, origin_2d, y_2d, (0, 255, 0), 2)
    cv2.line(img, origin_2d, z_2d, (255, 0, 0), 2)
    # Add labels if requested
    if label_axes:
        cv2.putText(img, f'{prefix}X', x_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(img, f'{prefix}Y', y_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(img, f'{prefix}Z', z_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    return img


def ros_qt_to_rt(rot, trans):
    """
    Convert ROS quaternion and translation to a 4x4 transformation matrix.

    Args:
        rot (np.ndarray): Quaternion (x, y, z, w) from ROS.
        trans (np.ndarray): Translation vector (x, y, z) in meters.

    Returns:
        np.ndarray: 4x4 transformation matrix (rotation and translation).
    """
    # Reorder quaternion from ROS (x, y, z, w) to (w, x, y, z)
    qt = np.zeros((4,), dtype=np.float32)
    qt[0] = rot[3]  # w
    qt[1] = rot[0]  # x
    qt[2] = rot[1]  # y
    qt[3] = rot[2]  # z
    # Initialize 4x4 identity matrix
    obj_T = np.eye(4)
    # Convert quaternion to 3x3 rotation matrix
    obj_T[:3, :3] = quat2mat(qt)
    # Set translation component
    obj_T[:3, 3] = trans
    return obj_T


def main(args):
    """
    Main function to estimate relative pose between two frames and visualize results.

    Args:
        args: Command-line arguments with scene_dir and mode.
    """
    # Convert scene directory to Path object for easy file handling
    scene_dir = Path(args.scene_dir)
    # Define subdirectories for data
    mat_dir = scene_dir / "mat"      # MATLAB files with poses
    rgb_dir = scene_dir / "rgb"      # RGB images
    mask_dir = scene_dir / "masks"   # Object masks
    depth_dir = scene_dir / "depth"  # Depth images
    K_path = scene_dir / "cam_K.txt" # Camera intrinsics file

    # Define frame pairs to process (e.g., images 000000 and 000001)
    names = [
        ('000000', '000001'),
    ]

    # Loop over each frame pair
    for _name0, _name1 in names:
        # Construct file names for images
        name0 = f"{_name0}.png"
        name1 = f"{_name1}.png"
        # Load RGB images
        img0 = cv2.imread(str(rgb_dir / name0.replace('.png', '.jpg')))
        img1 = cv2.imread(str(rgb_dir / name1.replace('.png', '.jpg')))
        # Load masks (grayscale, unchanged to preserve values)
        mask0 = cv2.imread(str(mask_dir / name0), cv2.IMREAD_UNCHANGED)
        mask1 = cv2.imread(str(mask_dir / name1), cv2.IMREAD_UNCHANGED)
        # Load depth images (unchanged to preserve depth values)
        depth0 = cv2.imread(str(depth_dir / name0), cv2.IMREAD_UNCHANGED)
        depth1 = cv2.imread(str(depth_dir / name1), cv2.IMREAD_UNCHANGED)
        
        """
        # Load MATLAB files with object poses and camera transformations
        mat1 = loadmat(str(mat_dir / f"{_name0}.mat"))
        mat2 = loadmat(str(mat_dir / f"{_name1}.mat"))
        # Extract object poses (translation and quaternion)
        T_ca = list(mat1['obj_pose'])[0]  # Object pose in frame 0
        T_cb = list(mat2['obj_pose'])[0]  # Object pose in frame 1
        # Extract camera transformations
        RT_cam1 = mat1['RT_camera']  # Camera pose for frame 0
        RT_cam2 = mat2['RT_camera']  # Camera pose for frame 1
        """
        
        # Ensure all data loaded correctly
        assert img0 is not None and img1 is not None, "Images not loaded"
        assert mask0 is not None and mask1 is not None, "Masks not loaded"
        assert depth0 is not None and depth1 is not None, "Depth images not loaded"

        # Convert masks to binary (1 for object, 0 for background)
        mask0_bin = (mask0 > 0).astype(np.uint8)
        mask1_bin = (mask1 > 0).astype(np.uint8)
        # Apply masks to RGB images to isolate the object
        img0_masked = apply_mask(img0, mask0_bin)
        img1_masked = apply_mask(img1, mask1_bin)
        # Load camera intrinsic matrix
        K = load_intrinsics(K_path)

        # Initialize RDD model for feature matching
        RDD_model = build(weights='./weights/RDD-v2.pth')
        RDD_model.eval()  # Set model to evaluation mode
        RDD_wrap = RDD_helper(RDD_model)  # Wrap model for easier use

        # Measure time for feature matching
        start = time()
        # Find dense matches between masked images (resized to 1024 for efficiency)
        mkpts_0, mkpts_1, conf = RDD_wrap.match_dense(img0_masked, img1_masked, resize=1024)
        print(f"Found {len(mkpts_0)} matches in {time() - start:.2f} seconds")

        # Convert match points to float32 for OpenCV
        pts0 = mkpts_0.astype(np.float32)
        pts1 = mkpts_1.astype(np.float32)

        # Estimate essential matrix using RANSAC
        E, mask = cv2.findEssentialMat(pts0, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        # Recover relative pose (rotation and translation) from essential matrix
        _, R, t, inliers = cv2.recoverPose(E, pts0, pts1, K, mask=mask)

        # Initialize 4x4 transformation matrix for relative pose
        RT = np.eye(4, dtype=np.float32)
        RT[:3, :3] = R  # Set rotation
        # Compute object centroids in 3D for both frames
        t_obj_0 = get_3d_centroid_from_mask_and_depth(mask0_bin, depth0, K).reshape(1, 3)
        t_obj_1 = get_3d_centroid_from_mask_and_depth(mask1_bin, depth1, K).reshape(1, 3)

        # Compute relative translation using centroids
        t_metric = t_obj_1.flatten() - (R @ t_obj_0.flatten())
        
        # t_metric = t.flatten()  # Use translation from recoverPose


        RT[:3, 3] = t_metric.flatten()  # Set translation
        # Invert transformation matrix for saving
        # RT = np.linalg.pinv(RT)

        # Save relative pose to file
        np.savetxt(scene_dir / f"relative_pose_rt_{_name0}-{_name1}.txt", RT, fmt="%.6f")

        # Print debugging information
        print("Relative Camera Rotation R:\n", R)
        print("Relative Camera Translation t:\n", t_metric)
        print("Inlier count:", np.count_nonzero(inliers))
        print("Centroid 0:", t_obj_0)
        print("Centroid 1:", t_obj_1)

        """
        # Compute ground-truth relative pose for comparison
        T_ca = np.linalg.pinv(RT_cam1) @ ros_qt_to_rt(T_ca[3:], T_ca[:3])
        T_cb = np.linalg.pinv(RT_cam2) @ ros_qt_to_rt(T_cb[3:], T_cb[:3])
        T_ac = np.linalg.pinv(T_ca)
        T_ab = T_ac @ T_cb
        print("sum", sum(RT - T_ab))  # Compare estimated and ground-truth poses
        """
        
        # Draw matches between masked images
        canvas = draw_matches(mkpts_0, mkpts_1, img0_masked, img1_masked, inliers)

        # Draw 2D pose axes on images
        axis_length = 0.2
        img0_axis = draw_object_axis(img0.copy(), K, np.eye(3), t_obj_0, axis_length=axis_length, label_axes=True, prefix="Ref")
        img1_axis = draw_object_axis(img1.copy(), K, R.T, t_obj_1, axis_length=axis_length, label_axes=True, prefix="Rel")

        # Create 2D visualization with three subplots
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        # Show frame 0 with pose axes
        axs[0].imshow(img0_axis[..., ::-1])  # Convert BGR to RGB for Matplotlib
        axs[0].set_title("Frame 0: Reference Pose")
        # Show frame 1 with pose axes
        axs[1].imshow(img1_axis[..., ::-1])
        axs[1].set_title("Frame 1: Relative Pose")
        # Show matches (green lines for inliers, red dots for keypoints)
        axs[2].imshow(canvas[..., ::-1])
        axs[2].set_title("Object Mask Matches (Green: Inliers, Red: Outliers)")
        # Remove axes for cleaner visualization
        for ax in axs:
            ax.axis('off')
        plt.tight_layout()
        plt.show()

        # Attempt 3D visualization with Open3D
        try:
            import open3d as o3d

            def create_point_cloud(rgb, depth, K, mask=None):
                """
                Create a colored point cloud from RGB and depth images.

                Args:
                    rgb (np.ndarray): RGB image.
                    depth (np.ndarray): Depth image (in millimeters).
                    K (np.ndarray): 3x3 camera intrinsic matrix.
                    mask (np.ndarray, optional): Binary mask to filter points.

                Returns:
                    o3d.geometry.PointCloud: Point cloud with colors.
                """
                rgb_o3d = o3d.geometry.Image(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
                depth_o3d = o3d.geometry.Image(depth.astype(np.float32))
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    rgb_o3d, depth_o3d, depth_scale=1000.0,
                    depth_trunc=10.0, convert_rgb_to_intensity=False
                )
                h, w = rgb.shape[:2]
                intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    width=w, height=h, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
                )
                pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic)
                if mask is not None:
                    mask_flat = mask.flatten() > 0
                    points = np.asarray(pcd.points)
                    colors = np.asarray(pcd.colors)
                    pcd.points = o3d.utility.Vector3dVector(points[mask_flat])
                    pcd.colors = o3d.utility.Vector3dVector(colors[mask_flat])
                return pcd

            def create_custom_coordinate_frame(t, R, axis_length=0.2):
                """
                Create a custom 3D coordinate frame similar to draw_object_axis logic.

                Args:
                    t (np.ndarray): 3x1 translation vector (centroid).
                    R (np.ndarray): 3x3 rotation matrix.
                    axis_length (float): Length of the axes in meters.

                Returns:
                    o3d.geometry.TriangleMesh: Coordinate frame with X (red), Y (green), Z (blue) axes.
                """
                # Define origin and axis endpoints in 3D (in meters)
                origin = np.array([0, 0, 0], dtype=np.float32)
                axis = np.array([
                    [axis_length, 0, 0],  # X-axis
                    [0, axis_length, 0],  # Y-axis
                    [0, 0, axis_length]   # Z-axis
                ], dtype=np.float32)
                # Apply rotation and translation
                points = origin + t  # Origin at centroid
                axis_points = (R @ axis.T).T + t  # Rotate and translate axes
                # Create line set for axes
                lines = [[0, 1], [0, 2], [0, 3]]  # Lines from origin to X, Y, Z endpoints
                colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # X: red, Y: green, Z: blue
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector(np.vstack([points, axis_points]))
                line_set.lines = o3d.utility.Vector2iVector(lines)
                line_set.colors = o3d.utility.Vector3dVector(colors)
                return line_set

            # Create point clouds for both frames
            pcd0 = create_point_cloud(img0, depth0, K, mask=mask0_bin)
            pcd1 = create_point_cloud(img1, depth1, K, mask=mask1_bin)

            # Define pose for frame 0: Centered at object centroid with no rotation
            obj_pose_0 = np.eye(4)
            obj_pose_0[:3, 3] = t_obj_0.flatten()

            # Define pose for frame 1: Apply relative pose from frame 0
            obj_pose_1 = np.eye(4)
            obj_pose_1[:3, :3] = R
            obj_pose_1[:3, 3] = t_obj_0.flatten() + t_metric.flatten()

            # Transform point clouds to align with poses
            pcd0.transform(obj_pose_0)
            pcd1.transform(obj_pose_1)

            # Create coordinate frames for transformed visualization
            axis_length = 0.2
            mesh_obj_0 = create_custom_coordinate_frame(t_obj_0.flatten(), np.eye(3), axis_length)
            mesh_obj_1 = create_custom_coordinate_frame(t_obj_0.flatten() + t_metric.flatten(), R, axis_length)

            # Create spheres to mark centroids for transformed visualization
            sphere0 = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere0.paint_uniform_color([1, 0, 0])  # Red for frame 0
            sphere1.paint_uniform_color([0, 1, 0])  # Green for frame 1
            sphere0.translate(t_obj_0.flatten())
            sphere1.translate(t_obj_0.flatten() + t_metric.flatten())

            # Downsample point clouds for faster visualization
            pcd0 = pcd0.voxel_down_sample(voxel_size=0.005)
            pcd1 = pcd1.voxel_down_sample(voxel_size=0.005)

            # Visualize transformed point clouds and pose axes with title
            print("Visualizing transformed point clouds and pose axes...")
            vis_transformed = o3d.visualization.Visualizer()
            vis_transformed.create_window(window_name="Transformed Point Clouds with Pose Axes")
            vis_transformed.add_geometry(pcd0)
            vis_transformed.add_geometry(pcd1)
            vis_transformed.add_geometry(mesh_obj_0)
            vis_transformed.add_geometry(mesh_obj_1)
            vis_transformed.add_geometry(sphere0)
            vis_transformed.add_geometry(sphere1)
            vis_transformed.run()
            vis_transformed.destroy_window()

            # Debug: Visualize raw point clouds with identity pose for first and computed pose for second
            pcd0_raw = create_point_cloud(img0, depth0, K, mask=mask0_bin)
            pcd1_raw = create_point_cloud(img1, depth1, K, mask=mask1_bin)

            # Define identity pose for frame 0 (raw)
            raw_pose_0 = np.eye(4)  # No rotation, no translation
            raw_t_0 = np.array([0, 0, 0], dtype=np.float32)  # Origin

            # Define computed pose for frame 1 (raw)
            raw_pose_1 = np.eye(4)
            raw_pose_1[:3, :3] = R
            raw_pose_1[:3, 3] = t_metric.flatten()

            # Transform raw point clouds
            pcd0_raw.transform(raw_pose_0)  # Identity transform (no change)
            pcd1_raw.transform(raw_pose_1)  # Apply computed relative pose

            # Create custom coordinate frames for raw visualization
            mesh_raw_0 = create_custom_coordinate_frame(raw_t_0, np.eye(3), axis_length)
            mesh_raw_1 = create_custom_coordinate_frame(t_metric.flatten(), R, axis_length)

            # Create spheres for centroids in raw visualization
            sphere_raw_0 = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere_raw_1 = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sphere_raw_0.paint_uniform_color([1, 0, 0])  # Red for frame 0
            sphere_raw_1.paint_uniform_color([0, 1, 0])  # Green for frame 1
            sphere_raw_0.translate(raw_t_0)
            sphere_raw_1.translate(t_metric.flatten())

            # Downsample raw point clouds
            pcd0_raw = pcd0_raw.voxel_down_sample(voxel_size=0.005)
            pcd1_raw = pcd1_raw.voxel_down_sample(voxel_size=0.005)

            # Visualize raw point clouds with identity and computed poses
            print("Visualizing raw point clouds with identity pose (frame 0) and computed pose (frame 1)...")
            vis_raw = o3d.visualization.Visualizer()
            vis_raw.create_window(window_name="Raw Point Clouds with Identity and Computed Pose Axes")
            vis_raw.add_geometry(pcd0_raw)
            vis_raw.add_geometry(pcd1_raw)
            vis_raw.add_geometry(mesh_raw_0)
            vis_raw.add_geometry(mesh_raw_1)
            vis_raw.add_geometry(sphere_raw_0)
            vis_raw.add_geometry(sphere_raw_1)
            vis_raw.run()
            vis_raw.destroy_window()

        except ImportError:
            print("[Warning] Open3D is not installed. Skipping 3D visualization.")


if __name__ == "__main__":
    # Set up command-line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", required=True, help="Path to scene dir containing rgb/, masks/, depth/, cam_K.txt")
    parser.add_argument("--mode", default="match_dense", choices=["match", "match_dense", "match_lg", "match_3rd_party"],
                        help="RDD mode to use for matching")
    args = parser.parse_args()
    # Run the main function with parsed arguments
    main(args)
