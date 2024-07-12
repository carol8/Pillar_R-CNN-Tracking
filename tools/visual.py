import threading

from det3d.core.bbox.box_np_ops import center_to_corner_box3d
import open3d as o3d
import argparse
import pickle
import numpy as np
from pynput import keyboard


def label2color(label):
    colors = [[204 / 255, 0, 0], [52 / 255, 101 / 255, 164 / 255],
              [245 / 255, 121 / 255, 0], [115 / 255, 210 / 255, 22 / 255]]

    return colors[label]


def corners_to_lines(qs, color=None):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    if color is None:
        color = [204 / 255, 0, 0]
    idx = [(1, 0), (5, 4), (2, 3), (6, 7), (1, 2), (5, 6), (0, 3), (4, 7), (1, 5), (0, 4), (2, 6), (3, 7)]
    cl = [color for i in range(12)]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(qs),
        lines=o3d.utility.Vector2iVector(idx),
    )
    line_set.colors = o3d.utility.Vector3dVector(cl)

    return line_set


def plot_boxes(boxes, score_thresh):
    visuals = []
    num_det = boxes['scores'].shape[0]
    for i in range(num_det):
        score = boxes['scores'][i]
        if score < score_thresh:
            continue

        box = boxes['boxes'][i:i + 1]
        label = boxes['classes'][i]
        corner = center_to_corner_box3d(box[:, :3], box[:, 3:6], box[:, -1])[0].tolist()
        color = label2color(label)
        visuals.append(corners_to_lines(corner, color))
    return visuals


def translate_boxes_to_open3d_instance(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    center = gt_boxes[0:3]
    # lwh = gt_boxes[[3, 4, 5]]
    lwh = gt_boxes[[4, 3, 5]]  # wlh -> lwh
    # axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    axis_angles = np.array([0, 0, - gt_boxes[6] - np.pi / 2])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set, box3d


box_colormap = [
    (0, 1, 0),
    (0, 1, 1),
    (1, 1, 0),
]


def draw_box(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        vis.add_geometry(line_set)
    return vis


def draw_scenes(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None,
                draw_origin=False):
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=800, width=1280)

    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.zeros(3)

    # draw origin
    if draw_origin:
        axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(points[:, :3])

    vis.add_geometry(pts)
    if point_colors is None:
        print(f"Before: {np.ones((points.shape[0], 3))}")
        pts.colors = o3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
        print(f"After: {np.ones((points.shape[0], 3))}")
    else:
        print(f"Before: {np.ones((point_colors, 3))}")
        pts.colors = o3d.utility.Vector3dVector(np.ones((point_colors, 3)))
        print(f"After: {np.ones((point_colors, 3))}")

    if gt_boxes is not None:
        vis = draw_box(vis, gt_boxes, (0, 0, 1))

    if ref_boxes is not None:
        vis = draw_box(vis, ref_boxes, (0, 1, 0), ref_labels, ref_scores)

    vis.run()
    vis.destroy_window()


rotation_angles = [0, 0, 0]
translation_values = [0, 0, 0]


def keyboard_press_callback(key):
    global rotation_angles
    global translation_values
    try:
        print('alphanumeric key {0} pressed'.format(key.char))
        if key.char == 'w':
            rotation_angles[0] -= np.pi / 180
        if key.char == 's':
            rotation_angles[0] += np.pi / 180
        if key.char == 'e':
            rotation_angles[1] += np.pi / 180
        if key.char == 'q':
            rotation_angles[1] -= np.pi / 180
        if key.char == 'a':
            rotation_angles[2] += np.pi / 180
        if key.char == 'd':
            rotation_angles[2] -= np.pi / 180
        if key.char == 'i':
            translation_values[0] += 1
        if key.char == 'k':
            translation_values[0] -= 1
        if key.char == 'j':
            translation_values[1] += 1
        if key.char == 'l':
            translation_values[1] -= 1
        if key.char == 'h':
            translation_values[2] += 1
        if key.char == 'n':
            translation_values[2] -= 1
    except AttributeError:
        print('special key {0} pressed'.format(key))


def draw_box_video(vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None, rotation_matrix=None, translation_values=None):
    if rotation_matrix is None:
        rotation_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    if translation_values is None:
        translation_values = [0, 0, 0]
    for i in range(gt_boxes.shape[0]):
        line_set, box3d = translate_boxes_to_open3d_instance_video(gt_boxes[i])
        if ref_labels is None:
            line_set.paint_uniform_color(color)
        else:
            line_set.paint_uniform_color(box_colormap[ref_labels[i]])

        line_set.rotate(rotation_matrix, center=(0, 0, 0))
        line_set.translate(translation_values)
        vis.add_geometry(line_set)
    return vis


def translate_boxes_to_open3d_instance_video(gt_boxes):
    """
             4-------- 6
           /|         /|
          5 -------- 3 .
          | |        | |
          . 7 -------- 1
          |/         |/
          2 -------- 0
    """
    # print(gt_boxes)
    center = gt_boxes[0:3]
    # lwh = gt_boxes[[3, 4, 5]]
    lwh = gt_boxes[[4, 3, 5]]  # wlh -> lwh
    # axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
    axis_angles = np.array([0, 0, 0 - gt_boxes[6] - np.pi / 2])
    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
    box3d = o3d.geometry.OrientedBoundingBox(center, rot, lwh)

    line_set = o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

    # import ipdb; ipdb.set_trace(context=20)
    lines = np.asarray(line_set.lines)
    lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

    line_set.lines = o3d.utility.Vector2iVector(lines)

    return line_set, box3d


vis_video = None
initialisation = True


def draw_scenes_video(points, gt_boxes=None, ref_boxes=None, ref_labels=None, ref_scores=None, point_colors=None,
                      draw_origin=False, reset_window=False):
    global vis_video
    global initialisation
    global rotation_angles

    if initialisation:
        initialisation = False
        keyboard_listener = keyboard.Listener(
            on_press=keyboard_press_callback,
        )
        keyboard_listener.start()

    if reset_window or vis_video is None:
        if vis_video is not None:
            vis_video.destroy_window()
        vis_video = o3d.visualization.Visualizer()
        vis_video.create_window(height=800, width=1280)

        vis_video.get_render_option().point_size = 1.0
        vis_video.get_render_option().background_color = np.zeros(3)
        # vis_video.get_view_control().set_zoom(2)

        # draw origin
        if draw_origin:
            axis_pcd = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
            vis_video.add_geometry(axis_pcd)
    else:
        vis_video.clear_geometries()

    rot = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_angles)

    pts = o3d.geometry.PointCloud()
    pts.points = o3d.utility.Vector3dVector(points[:, :3])

    pts.rotate(rot)
    pts.translate(translation_values)
    vis_video.add_geometry(pts)
    if point_colors is None:
        pts.colors = o3d.utility.Vector3dVector(np.ones((points.shape[0], 3)))
    else:
        pts.colors = o3d.utility.Vector3dVector(np.ones((point_colors, 3)))

    # if gt_boxes is not None:
    #     vis_video = draw_box(vis_video, gt_boxes, (0, 0, 1))
    #
    if ref_boxes is not None:
        vis_video = draw_box_video(vis_video, ref_boxes, (0, 1, 0), ref_labels, ref_scores, rotation_matrix=rot, translation_values=translation_values)

    vis_video.poll_events()
    vis_video.update_renderer()
    # vis_video.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CenterPoint")
    parser.add_argument('--path', help='path to visualization file', type=str)
    parser.add_argument('--thresh', help='visualization threshold', type=float, default=0.3)
    args = parser.parse_args()

    with open(args.path, 'rb') as f:
        data_dicts = pickle.load(f)

    for data in data_dicts:
        points = data['points']
        detections = data['detections']

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])

        visual = [pcd]
        num_dets = detections['scores'].shape[0]
        visual += plot_boxes(detections, args.thresh)

        o3d.visualization.draw_geometries(visual)
