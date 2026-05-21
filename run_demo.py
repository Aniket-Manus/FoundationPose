# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from estimater import *
from datareader import *
import argparse


if __name__=='__main__':
  parser = argparse.ArgumentParser()
  code_dir = os.path.dirname(os.path.realpath(__file__))
  #parser.add_argument('--mesh_file', type=str, default=f'{code_dir}/demo_data/mustard0/mesh/textured_simple.obj')
  #parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
  # Input object mesh and captured RGB-D sequence directory.
  parser.add_argument('--mesh_file', type=str, default=f'/home/aniket/FoundationPose/data/my_custom_object/box_part_meters.obj')
  parser.add_argument('--test_scene_dir', type=str, default=f'/home/aniket/FoundationPose/data/my_custom_object/test_sequence')
  # Iterations for initial registration and per-frame tracking refinement.
  parser.add_argument('--est_refine_iter', type=int, default=10)
  parser.add_argument('--track_refine_iter', type=int, default=3)
  # Debug level controls visualization and extra artifact dumps.
  parser.add_argument('--debug', type=int, default=1)
  parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  # Load mesh and prepare a clean debug output folder.
  mesh = trimesh.load(args.mesh_file)

  debug = args.debug
  debug_dir = args.debug_dir
  os.system(f'rm -rf {debug_dir}/* && mkdir -p {debug_dir}/track_vis {debug_dir}/ob_in_cam')

  to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
  # Axis-aligned box in object frame used for 2D overlay visualization.
  bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

  # Create FoundationPose components (scoring + refinement + rasterization context).
  scorer = ScorePredictor()
  refiner = PoseRefinePredictor()
  glctx = dr.RasterizeCudaContext()
  est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, debug_dir=debug_dir, debug=debug, glctx=glctx)
  logging.info("estimator initialization done")

  # Reader provides RGB, depth, mask, and camera intrinsics from the scene folder.
  reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

  # Frame 0: initialize pose with object mask; later frames: track from previous pose.
  for i in range(len(reader.color_files)):
    logging.info(f'i:{i}')
    color = reader.get_color(i)
    depth = reader.get_depth(i)
    if i==0:
      mask = reader.get_mask(0).astype(bool)
      pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)

      if debug>=3:
        # Optional heavy debug exports: transformed mesh and full scene point cloud.
        m = mesh.copy()
        m.apply_transform(pose)
        m.export(f'{debug_dir}/model_tf.obj')
        xyz_map = depth2xyzmap(depth, reader.K)
        valid = depth>=0.001
        pcd = toOpen3dCloud(xyz_map[valid], color[valid])
        o3d.io.write_point_cloud(f'{debug_dir}/scene_complete.ply', pcd)
    else:
      pose = est.track_one(rgb=color, depth=depth, K=reader.K, iteration=args.track_refine_iter)

    os.makedirs(f'{debug_dir}/ob_in_cam', exist_ok=True)
    # Save per-frame 4x4 object pose in camera coordinates.
    np.savetxt(f'{debug_dir}/ob_in_cam/{reader.id_strs[i]}.txt', pose.reshape(4,4))

    if debug>=1:
      # Visual overlay: projected 3D box + coordinate axes on the RGB frame.
      center_pose = pose@np.linalg.inv(to_origin)
      vis = draw_posed_3d_box(reader.K, img=color, ob_in_cam=center_pose, bbox=bbox)
      vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=3, transparency=0, is_input_rgb=True)
      cv2.imshow('1', vis[...,::-1])
      cv2.waitKey(1)


    if debug>=2:
      # Save visualization frames for offline inspection.
      os.makedirs(f'{debug_dir}/track_vis', exist_ok=True)
      imageio.imwrite(f'{debug_dir}/track_vis/{reader.id_strs[i]}.png', vis)

