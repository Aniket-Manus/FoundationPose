from estimater import *
from datareader import *
import argparse
import time

if __name__=='__main__':
    # Parse command-line arguments so you can swap object mesh, dataset path,
    # and refinement settings without editing code each time.
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))
    # 3D model (OBJ) of the object to be tracked.
    parser.add_argument('--mesh_file', type=str, default='/home/aniket/FoundationPose/data/my_custom_object/box_part_meters.obj')
    # Folder containing rgb/, depth/, masks/, and camera intrinsics.
    parser.add_argument('--test_scene_dir', type=str, default='/home/aniket/FoundationPose/data/my_custom_object/test_sequence')
    # Iterations for first-frame registration and later-frame tracking updates.
    parser.add_argument('--est_refine_iter', type=int, default=10)
    parser.add_argument('--track_refine_iter', type=int, default=10)
    # Set default debug to 0 for pure speed. Use 1 only when you want the GUI window display.
    parser.add_argument('--debug', type=int, default=1)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()

    # Reproducibility + readable logs.
    set_logging_format()
    set_seed(0)

    # Load mesh geometry used by FoundationPose for rendering and scoring.
    mesh = trimesh.load(args.mesh_file)
    debug = args.debug
    debug_dir = args.debug_dir
    
    # Pre-create the directory structure ONCE outside the loop execution paths
    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    if debug >= 2:
        os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)

    # Compute a canonical object transform and object-space bounding box.
    # This is used only for visualization overlays (box/axis drawing).
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

    # Build the two learned modules:
    # - ScorePredictor: evaluates pose hypotheses
    # - PoseRefinePredictor: iteratively improves pose
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    # CUDA rasterizer context used for differentiable rendering operations.
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
    logging.info("estimator initialization done")

    # Dataset reader gives per-frame RGB, depth, mask, and camera intrinsics K.
    # K is the 3x3 camera matrix mapping 3D camera points to image pixels.
    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

    # Fast in-memory list to accumulate poses without triggering disk writes
    tracked_poses = []
    
    print("\n>>> Launching high-speed tracking pipeline... please wait...")
    start_time = time.time()

    # Main loop over frames in the sequence.
    for i in range(len(reader.color_files)):
        color = reader.get_color(i)
        depth = reader.get_depth(i)
        
        if i == 0:
            # Frame 0 needs an object mask to initialize the first pose.
            # register(...) returns a 4x4 transform: object pose in camera frame.
            mask = reader.get_mask(0).astype(bool)
            pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
        else:
            # Subsequent frames track from previous state, so no new mask needed.
            pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

        # Store calculations in RAM immediately
        tracked_poses.append((reader.id_strs[i], pose.copy()))

        # Handle UI drawing ONLY if debug mode is active
        if debug >= 1:
            # Convert pose for centered box visualization, then render overlays.
            center_pose = pose @ np.linalg.inv(to_origin)
            vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
            vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
            
            # OpenCV expects BGR for display; vis is RGB, so reverse channels.
            cv2.imshow('FoundationPose Real-time Monitor', vis[...,::-1])
            cv2.waitKey(1)
            
            if debug >= 2:
                # Optional: save each visualization frame to disk.
                imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

    end_time = time.time()
    total_fps = len(reader.color_files) / (end_time - start_time)
    print(f"\n>>> Tracking Finished! Processing speed: {total_fps:.2f} FPS")

    # --- Post-Processing Disk Dump Phase ---
    # Save all estimated poses as text files (one 4x4 matrix per frame).
    # Format: debug/ob_in_cam/<frame_id>.txt
    print(">>> Streaming computed transforms from memory to storage...")
    for id_str, saved_pose in tracked_poses:
        np.savetxt(f'{debug_dir}/ob_in_cam/{id_str}.txt', saved_pose.reshape(4,4))
    print(">>> Storage write complete. Done!")