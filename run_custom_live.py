"""
run_custom_live.py  –  FoundationPose on a live Orbbec Gemini 335L stream.

Workflow
--------
1.  Camera pipeline starts (same settings as record_sequence.py).
2.  A warm-up window lets the auto-exposure/laser settle.
3.  The first stable frame is grabbed and displayed so you can draw a tight
    bounding-box mask around the target object with your mouse.
4.  FoundationPose registers the initial pose from that mask.
5.  All subsequent frames are tracked in real-time; an overlay is shown in an
    OpenCV window.

Controls (tracking window)
--------------------------
  [R]         – re-initialise: pauses tracking, reopens the mask-draw UI on the
                current frame so you can pick the object again.
  [S]         – save the current pose to  debug/ob_in_cam/<timestamp>.txt
  [ESC] / [Q] – quit.
"""

from estimater import *   # noqa: F401,F403  (imports numpy, cv2, trimesh, etc.)
import argparse
import time
import glob

try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode
except ModuleNotFoundError:
    from pyorbbecsdk2 import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode


# ---------------------------------------------------------------------------
# Camera helpers (lifted from record_sequence.py)
# ---------------------------------------------------------------------------

def _find_property_id(device, prop_name):
    for i in range(device.get_support_property_count()):
        p = device.get_supported_property(i)
        if getattr(p, "name", "") == prop_name:
            return p.id
    return None


def _enable_laser_max_power(pipe):
    device = pipe.get_device()
    for name, val, typ in [
        ("OB_PROP_LASER_CONTROL_INT",       None,  "int"),
        ("OB_PROP_LASER_ALWAYS_ON_BOOL",    True,  "bool"),
        ("OB_PROP_LASER_POWER_LEVEL_CONTROL_INT", None, "int"),
    ]:
        pid = _find_property_id(device, name)
        if pid is None:
            continue
        try:
            if typ == "bool":
                device.set_bool_property(pid, val)
            else:
                rng = device.get_int_property_range(pid)
                target = 1 if (val == 1 and rng.min <= 1 <= rng.max) else int(rng.max)
                device.set_int_property(pid, target)
        except Exception as e:
            print(f"Warning [{name}]: {e}")


def _create_pipeline_with_retry(max_wait_sec=12, retry_interval_sec=1.0):
    import subprocess, time as _t
    start = _t.time()
    last_err = None
    while (_t.time() - start) < max_wait_sec:
        try:
            return Pipeline()
        except RuntimeError as e:
            last_err = e
            print("Camera not detected yet. Retrying…")
            _t.sleep(retry_interval_sec)
    raise RuntimeError(str(last_err) if last_err else "No device found")


def start_camera():
    """Open the Orbbec pipeline and return (pipe, K, disparity_value)."""
    pipe = _create_pipeline_with_retry()
    cfg = Config()

    profile_list = pipe.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    color_profile = profile_list.get_video_stream_profile(640, 400, OBFormat.RGB, 30)
    cfg.enable_stream(color_profile)

    depth_profile_list = pipe.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
    depth_profile = depth_profile_list.get_video_stream_profile(640, 400, OBFormat.Y16, 30)
    cfg.enable_stream(depth_profile)

    cfg.set_align_mode(OBAlignMode.HW_MODE)
    pipe.start(cfg)

    # Disparity range → close-range (~0.25 m minimum)
    disparity_value = 256
    try:
        device = pipe.get_device()
        pid = _find_property_id(device, "OB_PROP_DISPARITY_SEARCH_RANGE_INT")
        if pid is not None:
            device.set_int_property(pid, 256)
            disparity_value = device.get_int_property(pid)
            print(f"✓ Disparity search range set to {disparity_value}")
    except Exception as e:
        print(f"⚠ Disparity range: {e}")

    _enable_laser_max_power(pipe)

    # Read camera intrinsics matrix K
    cam_param = pipe.get_camera_param()
    intr = cam_param.rgb_intrinsic
    K = np.array([
        [intr.fx, 0,       intr.cx],
        [0,       intr.fy, intr.cy],
        [0,       0,       1      ],
    ], dtype=np.float64)
    print(f"Camera K:\n{K}")

    return pipe, K, disparity_value


# ---------------------------------------------------------------------------
# Frame grabbing helpers
# ---------------------------------------------------------------------------

def grab_frame(pipe):
    """Return (color_rgb_uint8, depth_float32_meters) or (None, None)."""
    frames = pipe.wait_for_frames(100)
    if not frames:
        return None, None
    color_frame = frames.get_color_frame()
    depth_frame = frames.get_depth_frame()
    if not color_frame or not depth_frame:
        return None, None

    # Color → RGB uint8
    h, w = color_frame.get_height(), color_frame.get_width()
    color_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
    color_rgb = color_data.reshape((h, w, 3)).copy()  # already RGB from SDK

    # Depth → float32 metres
    dh, dw = depth_frame.get_height(), depth_frame.get_width()
    depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16)
    depth_mm = depth_data.reshape((dh, dw)).astype(np.float32)
    depth_m = depth_mm / 1000.0
    depth_m[(depth_m < 0.001)] = 0.0   # zero out invalid pixels

    return color_rgb, depth_m


# ---------------------------------------------------------------------------
# Interactive mask creation
# ---------------------------------------------------------------------------

def draw_mask_interactive(color_rgb):
    """
    Show the user a window to draw a bounding-box mask over the object.

    Returns a boolean numpy array (H x W) with True inside the selected box,
    or None if the user cancelled.

    Instructions are printed in the terminal AND overlaid on the image so the
    user knows what to do without reading the source code.
    """
    display_bgr = color_rgb[..., ::-1].copy()   # RGB → BGR for OpenCV (clean, no overlaid text)
    font = cv2.FONT_HERSHEY_SIMPLEX

    print("\n=======================================================")
    print(" MASK CREATION  (first frame)")
    print(" -> A window will open with the first live frame.")
    print(" -> Press any key to begin drawing.")
    print(" -> Click and drag a box tightly around your object.")
    print(" -> Press [SPACE] or [ENTER] to confirm the selection.")
    print(" -> Press [ESC] to cancel.")
    print("=======================================================\n")

    # Use ASCII-only window name – non-ASCII characters can crash OpenCV on Linux.
    win = "MASK CREATION"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 600)

    # --- Step 1: show the frame and wait for the user to be ready -----------
    # Overwrite the hint text for the "ready" screen.
    ready_img = display_bgr.copy()
    ready_lines = [
        "First frame captured.",
        "Press ANY KEY to start drawing the mask.",
    ]
    for idx, line in enumerate(ready_lines):
        y_pos = 40 + idx * 36
        cv2.putText(ready_img, line, (10, y_pos), font, 0.9, (0, 0, 0),   4, cv2.LINE_AA)
        cv2.putText(ready_img, line, (10, y_pos), font, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow(win, ready_img)
    cv2.waitKey(0)   # Block here until any key – window is stable and focused.

    # --- Step 2: selectROI now that the window is settled and has focus -----
    roi = cv2.selectROI(win, display_bgr, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(win)
    cv2.waitKey(1)   # Flush any pending GUI events after destroying the window.

    x, y, w, h = roi
    if w <= 0 or h <= 0:
        print("Mask selection cancelled.")
        return None

    mask = np.zeros(color_rgb.shape[:2], dtype=bool)
    mask[y:y + h, x:x + w] = True
    print(f"Mask created: x={x}, y={y}, w={w}, h={h}")
    return mask


# ---------------------------------------------------------------------------
# Warm-up: discard the first N frames so auto-exposure settles
# ---------------------------------------------------------------------------

def warmup_camera(pipe, n_frames=30, display=True):
    """Drain N frames from the pipeline; optionally show a live preview."""
    win = "Camera Warm-up"
    if display:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, 960, 400)

    print(f"Warming up camera ({n_frames} frames)...", end="", flush=True)
    grabbed = 0
    while grabbed < n_frames:
        color, depth = grab_frame(pipe)
        if color is None:
            continue
        grabbed += 1
        if display:
            preview_depth = cv2.normalize(depth, None, 0, 255,
                                          cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            preview_depth_color = cv2.applyColorMap(preview_depth, cv2.COLORMAP_JET)
            side_by_side = np.hstack((color[..., ::-1], preview_depth_color))
            pct = int(grabbed / n_frames * 100)
            cv2.putText(side_by_side, f"Warming up... {pct}%",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(win, side_by_side)
            cv2.waitKey(1)

    if display:
        cv2.destroyWindow(win)
    print(" done.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run FoundationPose on a live Orbbec Gemini 335L stream."
    )
    code_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument(
        "--mesh_file", type=str,
        default="/home/aniket/FoundationPose/data/my_custom_object/Mpro_CasingLeft_V007_blender.obj",
        help="Path to the object's 3-D mesh (.obj).",
    )
    parser.add_argument("--est_refine_iter",   type=int, default=15,
                        help="Pose-registration iterations for the first frame.")
    parser.add_argument("--track_refine_iter", type=int, default=5,
                        help="Tracking refinement iterations per frame.")
    parser.add_argument("--debug",             type=int, default=1,
                        help="0=headless, 1=live overlay window.")
    parser.add_argument("--debug_dir",         type=str, default=f"{code_dir}/debug")
    parser.add_argument("--warmup_frames",     type=int, default=30,
                        help="Frames to discard so auto-exposure settles (default 30).")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    set_logging_format()
    set_seed(0)

    os.makedirs(f"{args.debug_dir}/ob_in_cam", exist_ok=True)

    # ------------------------------------------------------------------
    # Load mesh
    # ------------------------------------------------------------------
    print(f"Loading mesh: {args.mesh_file}")
    mesh = trimesh.load(args.mesh_file)
    to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
    bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

    # ------------------------------------------------------------------
    # Build FoundationPose estimator
    # ------------------------------------------------------------------
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices,
        model_normals=mesh.vertex_normals,
        mesh=mesh,
        scorer=scorer,
        refiner=refiner,
        debug_dir=args.debug_dir,
        debug=args.debug,
        glctx=glctx,
    )
    logging.info("Estimator initialised.")

    # ------------------------------------------------------------------
    # Open camera
    # ------------------------------------------------------------------
    print("\nOpening Orbbec camera…")
    pipe, K, _ = start_camera()

    # ------------------------------------------------------------------
    # Warm-up
    # ------------------------------------------------------------------
    warmup_camera(pipe, n_frames=args.warmup_frames, display=(args.debug >= 1))

    # ------------------------------------------------------------------
    # Grab first frame and create mask interactively
    # ------------------------------------------------------------------
    color0, depth0 = None, None
    while color0 is None:
        color0, depth0 = grab_frame(pipe)

    mask = None
    while mask is None:
        mask = draw_mask_interactive(color0)
        if mask is None:
            ans = input("No mask drawn. Try again? [y/n]: ").strip().lower()
            if ans != "y":
                print("Aborted – no mask provided.")
                pipe.stop()
                return

    # ------------------------------------------------------------------
    # Register first pose
    # ------------------------------------------------------------------
    print("\nRegistering initial pose… (this may take a few seconds)")
    pose = est.register(
        K=K,
        rgb=color0,
        depth=depth0,
        ob_mask=mask,
        iteration=args.est_refine_iter,
    )
    print("Initial pose registered.")

    # ------------------------------------------------------------------
    # Live tracking loop
    # ------------------------------------------------------------------
    win_track = "FoundationPose Live Tracking"
    if args.debug >= 1:
        cv2.namedWindow(win_track, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_track, 960, 600)

    frame_idx = 0
    fps_times = []
    print("\n=======================================================")
    print(" Live tracking active.")
    print(f" [R] re-initialise  |  [S] save current pose  |  [Q/ESC] quit")
    print("=======================================================\n")

    try:
        while True:
            color, depth = grab_frame(pipe)
            if color is None:
                continue

            t0 = time.time()

            if frame_idx == 0:
                # Use the pose already registered from the mask step
                current_pose = pose
            else:
                current_pose = est.track_one(
                    rgb=color,
                    depth=depth,
                    K=K,
                    iteration=args.track_refine_iter,
                )

            elapsed = time.time() - t0
            fps_times.append(elapsed)
            if len(fps_times) > 30:
                fps_times.pop(0)
            live_fps = 1.0 / (sum(fps_times) / len(fps_times)) if fps_times else 0.0

            # ---- visualisation ----
            if args.debug >= 1:
                center_pose = current_pose @ np.linalg.inv(to_origin)
                vis = draw_posed_3d_box(K, img=color, ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(
                    color, ob_in_cam=center_pose,
                    scale=0.1, K=K,
                    thickness=3, transparency=0, is_input_rgb=True,
                )
                vis_bgr = vis[..., ::-1].copy()
                cv2.putText(vis_bgr, f"FPS: {live_fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(vis_bgr, f"Frame: {frame_idx}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                cv2.putText(vis_bgr, "[R] reinit  [S] save  [Q] quit",
                            (10, vis_bgr.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
                cv2.imshow(win_track, vis_bgr)

            # ---- keyboard control ----
            key = cv2.waitKey(1) & 0xFF

            if key in (ord('q'), ord('Q'), 27):   # ESC
                print("Quit key pressed. Exiting.")
                break

            elif key in (ord('s'), ord('S')):
                ts = time.strftime("%Y%m%d_%H%M%S")
                save_path = f"{args.debug_dir}/ob_in_cam/live_{ts}.txt"
                np.savetxt(save_path, current_pose.reshape(4, 4))
                print(f"Pose saved → {save_path}")

            elif key in (ord('r'), ord('R')):
                # Re-initialisation: draw a new mask on the current frame
                print("\nRe-initialisation requested…")
                new_mask = draw_mask_interactive(color)
                if new_mask is not None:
                    print("Re-registering pose…")
                    pose = est.register(
                        K=K,
                        rgb=color,
                        depth=depth,
                        ob_mask=new_mask,
                        iteration=args.est_refine_iter,
                    )
                    current_pose = pose
                    frame_idx = 0   # reset so next iteration goes through track_one
                    print("Re-registration done.")
                else:
                    print("Re-initialisation cancelled; continuing with last pose.")

            frame_idx += 1

    finally:
        pipe.stop()
        cv2.destroyAllWindows()
        print("Camera stopped. Bye!")


if __name__ == "__main__":
    main()
