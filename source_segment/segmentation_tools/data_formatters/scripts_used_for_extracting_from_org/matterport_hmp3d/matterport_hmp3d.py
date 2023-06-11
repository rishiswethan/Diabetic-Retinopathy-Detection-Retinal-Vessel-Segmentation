
import math
import os
import random
import shutil
import sys

import git
import imageio
import magnum as mn
import numpy as np

from matplotlib import pyplot as plt
from habitat.utils.visualizations import maps
from habitat_sim.utils.common import d3_40_colors_rgb

# function to display the topdown map
from PIL import Image
import cv2

import habitat_sim
from habitat_sim.utils import common as utils
from habitat_sim.utils import viz_utils as vut


#####################################################
# Sample paths

dir_path = "/home/rishi/programming/AI/experiments/datasets_extractor/"
data_path = os.path.join(dir_path, "data")
output_directory = "output/"
output_path = os.path.join(dir_path, output_directory)
if not os.path.exists(output_path):
    os.mkdir(output_path)

hmp3d_glb_path_v2 = "/home/rishi/programming/AI/experiments/datasets_extractor/data/fresh_mattterport_example/data/scene_datasets/hm3d/minival/00800-TEEsavR23oF/TEEsavR23oF.basis.glb"
# hmp3d_glb_path_v2 = "/home/rishi/programming/AI/experiments/datasets_extractor/data/fresh_mattterport_example/data/scene_datasets/hm3d/minival/00808-y9hTuugGdiq/y9hTuugGdiq.basis.glb"
hmp3d_scene_dataset_path_v2 = "/home/rishi/programming/AI/experiments/datasets_extractor/data/fresh_mattterport_example/data/scene_datasets/hm3d/minival/hm3d_annotated_minival_basis.scene_dataset_config.json"

org_mp_glb_path = "/home/rishi/programming/AI/experiments/datasets_extractor/data/matterport_org_habitat/mp3d_habitat/mp3d/1LXtFkjw3qL/1LXtFkjw3qL.glb"
org_mp_scene_config = "/home/rishi/programming/AI/experiments/datasets_extractor/data/matterport_org_habitat/mp3d_habitat/mp3d.scene_dataset_config.json"

glb_path = hmp3d_glb_path_v2
scene_dataset_path = hmp3d_scene_dataset_path_v2
navmesh_path = glb_path.replace(".glb", ".navmesh")

scene_file_def = glb_path
scene_dataset_json_def = scene_dataset_path


############################################################################################################
rgb_sensor = True
depth_sensor = True
semantic_sensor = True

turn_angle = 45.0

meters_per_pixel = .25
TOPDOWN_MAP_BORDER_PIXELS = 2  # in pixels
NEARBY_POINTS_REJECTION_RADIUS = 4  # in pixels
height = 1  # in meters

DEF_SETTINGS = {
    "width": 1024,  # Spatial resolution of the observations
    "height": 1024,
    "scene": scene_file_def,  # Scene path
    "scene_dataset": scene_dataset_json_def,  # the scene dataset configuration files
    "default_agent": 0,
    "sensor_height": height,  # Height of sensors in meters
    "color_sensor": rgb_sensor,  # RGB sensor
    "depth_sensor": depth_sensor,  # Depth sensor
    "semantic_sensor": semantic_sensor,  # Semantic sensor
    "seed": 1,  # used in the random navigation
    "enable_physics": False,  # kinematics only
}

LABEL_LISTS = {
    'floor': [
        'floor',
        # 'rug'
    ],
    'wall': ['wall'],
    'ceiling': ['ceiling'],
}
SUB_NAME_TO_MAIN_NAME = {
    'floor': 'floor',
    'rug': 'floor',
    'wall': 'wall',
    'ceiling': 'ceiling',
}
LABEL_MASK_OUTPUT_NUMBER = {
    'floor': 1,
    'wall': 2,
    'ceiling': 3,
}

############################################################################################################
def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.scene_dataset_config_file = settings["scene_dataset"]
    sim_cfg.enable_physics = settings["enable_physics"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(semantic_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=turn_angle)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=turn_angle)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    rgb_img = Image.fromarray(rgb_obs)

    arr = [rgb_img, semantic_obs]
    titles = ["rgb", "semantic"]

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show()


def get_all_masks(scene, label_lists=LABEL_LISTS, verbose=False):
    if verbose:
        print(f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects")
        print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

    label_lists["unknown"] = ["unknown"]

    label_name_to_seg_indices = {}
    for label in label_lists:
        label_name_to_seg_indices[label] = []

    for obj in scene.objects:
        obj_name = obj.category.name()

        for label in label_lists:
            obj_flag = False

            for sub_label in label_lists[label]:
                if obj_name == sub_label:
                    obj_flag = True
                    break

            if obj_flag:
                if verbose:
                    print(
                        f"Object id:{obj.id}, category:{obj.category.name()},"
                        f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                    )
                pixel_index = int(str(obj.id).split("_")[-1])
                label_name_to_seg_indices[label].append(pixel_index)

    return label_name_to_seg_indices

def equilidian_distance(p1, p2):
    return np.sqrt(np.sum(np.square(p1 - p2)))

def in_list(list_of_lists, item):
    for list_ in list_of_lists:
        if set(item) == set(list_):
            return True
    return False


def display_map(topdown_map, key_points=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # plot points on map. The must be in the format of x and y coordinates, so invert x and y if you get the index directly from an image array
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker=".", markersize=4, alpha=0.8)
    plt.show()


def init_scene(scene_file, scene_dataset_file=scene_dataset_json_def, sim_settings=DEF_SETTINGS):
    sim_settings["scene"] = scene_file
    sim_settings["scene_dataset"] = scene_dataset_file

    cfg = make_cfg(sim_settings)
    sim = habitat_sim.Simulator(cfg)

    print(get_all_masks(sim.semantic_scene, verbose=True))

    agent = sim.initialize_agent(sim_settings["default_agent"])
    agent_state = habitat_sim.AgentState()
    # agent_state.position = np.array([-0.6, 0.0, 0.0])  # world space
    agent.set_state(agent_state)

    return sim, agent


def get_topdown_map(
        sim,
        scene_name,
        height=height,
        meters_per_pixel=meters_per_pixel,
        display=True
):
    print("The NavMesh bounds are: " + str(sim.pathfinder.get_bounds()))
    # get bounding box minumum elevation for automatic height

    if not sim.pathfinder.is_loaded:
        print("Pathfinder not initialized, aborting.")
    else:
        # This map is a 2D boolean array
        sim_topdown_map = sim.pathfinder.get_topdown_view(meters_per_pixel, height)

        if display:
            hablab_topdown_map = maps.get_topdown_map(
                sim.pathfinder, height, meters_per_pixel=meters_per_pixel
            )
            recolor_map = np.array(
                [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
            )
            hablab_topdown_map = recolor_map[hablab_topdown_map]
            print("Displaying the raw map from get_topdown_view:")
            # display_map(sim_topdown_map)
            print("Displaying the map from the Habitat-Lab maps module:")
            display_map(hablab_topdown_map)


def get_unique_heights(sim):
    pathfinder_seed = 1
    sim.pathfinder.seed(pathfinder_seed)

    # get 50 random navigable points and get all the unique heights
    random_nav_points = []
    for i in range(50):
        nav_point = sim.pathfinder.get_random_navigable_point()

        random_nav_points.append(nav_point)
        print("Random navigable point : " + str(nav_point))
        print("Is point navigable? " + str(sim.pathfinder.is_navigable(nav_point)))

    # get heights alone from the random nav points
    random_nav_points_heights = [point[1] for point in random_nav_points]
    random_nav_points_heights = np.unique(random_nav_points_heights)
    print("Random nav points unique heights: " + str(random_nav_points_heights))

    # keep only one instance of each integer value while keeping the first number's decimals
    unique_heights = []
    unique_heights_prefix = []
    for height in random_nav_points_heights:
        if int(height) not in unique_heights_prefix:
            unique_heights.append(height)
            unique_heights_prefix.append(int(height * 10))

    print("Filtered unique heights: " + str(unique_heights))

    return unique_heights


def recursively_get_nearby_points(
        point, search_pixel_radius=2, recursion_depth=0, nearby_points=[], nearby_points_str_list=[], initial_point=None
):
    if initial_point is None:
        initial_point = point.copy()

    nearby_points_ = []
    # get all the points within a radius of the point
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i == 0 and j == 0:
                continue

            if {(point[0] + i), (point[1] + j)} == set(initial_point):
                continue

            nearby_points_str = str(point[0] + i) + "_" + str(point[1] + j)
            if nearby_points_str not in nearby_points_str_list:
                nearby_points_str_list.append(nearby_points_str)
                nearby_points_.append([point[0] + i, point[1] + j])
                nearby_points.append([point[0] + i, point[1] + j])

    # recursively call this function on the nearby points
    if recursion_depth < search_pixel_radius - 1:
        # print("Recursion depth: " + str(recursion_depth), "Nearby points: ", len(nearby_points))
        for point_ in nearby_points_:
            recursively_get_nearby_points(
                point_, search_pixel_radius, recursion_depth + 1, nearby_points, nearby_points_str_list, initial_point
            )

    if recursion_depth == 0:
        return nearby_points


def make_border_of_mask(
        mask_arr,
        mask_number=1,
        border_mask_number=1,
        border_size=2,
):
    mask_arr = mask_arr.copy()

    # remove all other masks
    mask_arr[mask_arr != mask_number] = 0
    mask_arr[mask_arr == mask_number] = border_mask_number

    # Perform an erosion operation to shrink the object slightly
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (border_size, border_size))
    eroded_mask = cv2.erode(mask_arr, kernel)

    # Subtract the eroded image from the original image
    border_mask = cv2.absdiff(mask_arr, eroded_mask)

    return border_mask


def get_all_images(
        sim,
        agent,
        scene_name,
        meters_per_pixel=meters_per_pixel,
        turn_angle=turn_angle,
        save_folder=output_path,
        display=True
):
    print("NavMesh area = " + str(sim.pathfinder.navigable_area))
    print("Bounds = " + str(sim.pathfinder.get_bounds()))

    random_nav_points_heights = get_unique_heights(sim)

    # get the topdown map for these heights
    height_points_list = {}

    # folders are only reset in the subfolder which represents a single scene.
    # This is to avoid deleting the images from the previous iterations
    if not os.path.exists(os.path.join(save_folder + "images", scene_name)):
        os.makedirs(os.path.join(save_folder + "images", scene_name))
    else:
        shutil.rmtree(os.path.join(save_folder + "images", scene_name))
        os.makedirs(os.path.join(save_folder + "images", scene_name))

    if not os.path.exists(os.path.join(save_folder + "masks", scene_name)):
        os.makedirs(os.path.join(save_folder + "masks", scene_name))
    else:
        shutil.rmtree(os.path.join(save_folder + "masks", scene_name))
        os.makedirs(os.path.join(save_folder + "masks", scene_name))

    for i, height in enumerate(random_nav_points_heights):
        print("Height: " + str(height) + " (" + str(i + 1) + "/" + str(len(random_nav_points_heights)) + ")")

        height_points_list[height] = []
        sim_topdown_view = sim.pathfinder.get_topdown_view(meters_per_pixel, height)
        hablab_topdown_map = maps.get_topdown_map(
            sim.pathfinder, height, draw_border=True, meters_per_pixel=meters_per_pixel
        )
        print("Displaying the topdown map for height: " + str(height), sim_topdown_view.shape)
        print("Points", hablab_topdown_map, hablab_topdown_map.shape)
        # display_map(sim_topdown_view, key_points=random_nav_points)

        # get the points which are on the border of the island and increase the size of the border so that it is avoided
        island_border_points = np.argwhere(hablab_topdown_map == 2)
        print("island border points", len(island_border_points))
        hablab_topdown_map[hablab_topdown_map == 2] = 1
        print("hablab_topdown_map", np.unique(hablab_topdown_map))

        # add neighbouring points to the neighbouring points list to bad points
        hablab_topdown_map_b = make_border_of_mask(hablab_topdown_map, mask_number=1, border_mask_number=2, border_size=TOPDOWN_MAP_BORDER_PIXELS).copy()
        print("hablab_topdown_map_b", np.unique(hablab_topdown_map_b))

        bad_points = list(np.argwhere(hablab_topdown_map_b == 2))
        print("Nearby neighbour points", len(bad_points), bad_points[:10])

        points = np.argwhere(hablab_topdown_map == 1)
        print("pre-filtered points", len(points))

        # filter pixels that are too close to each other. Minimum distance is 2 pixels
        good_points = []
        for point in points:
            # Can be added to the good points list if it is not in the bad points list.
            # Surrounding points of this point will be added to the bad points list
            if in_list(bad_points, point):
                continue
            else:
                good_points.append([point[1], point[0]])

            nearby_points = recursively_get_nearby_points(point, search_pixel_radius=5)

            # remove the neighbours that are too close to the point
            for nearby_point in nearby_points:
                bad_points.append(nearby_point)

        points = good_points.copy()

        grid_dimensions = (hablab_topdown_map.shape[0], hablab_topdown_map.shape[1])
        # get real world coordinates of the points
        real_points = [
                maps.from_grid(
                    point[1],
                    point[0],
                    grid_dimensions,
                    sim=sim,
                    pathfinder=sim.pathfinder,
                )
                for point in points
            ]
        # real_points = points.copy()
        print("Filtered points", real_points)

        print(f"Points for height: {height}", points, len(points))
        if display:
            display_map(hablab_topdown_map, key_points=points)

        # add height to the points
        for i, point in enumerate(real_points):
            real_points[i] = [point[0], height, point[1]]
        real_points = np.array(real_points)

        display_path_agent_renders = True
        if display_path_agent_renders:
            print("Rendering observations at path points: num_points = " + str(len(real_points)))
            agent_state = habitat_sim.AgentState()
            for ix, point in enumerate(real_points):
                if ix < len(real_points) - 1:
                    point = np.array(point)
                    print("Point: " + str(point))

                    agent_state.position = point
                    tangent = real_points[ix + 1] - point
                    #
                    tangent_orientation_matrix = mn.Matrix4.look_at(
                        point, point + tangent, np.array([0.0, 1.0, 0.0])
                    )
                    tangent_orientation_q = mn.Quaternion.from_matrix(
                        tangent_orientation_matrix.rotation()
                    )
                    agent_state.rotation = utils.quat_from_magnum(tangent_orientation_q)
                    agent.set_state(agent_state)

                    sub_folder_1 = scene_name + os.sep
                    sub_folder_2 = f"height{str(height)[:5]}_pt{ix}{os.sep}"
                    if not os.path.exists(os.path.join(save_folder + "images", sub_folder_1, sub_folder_2)):
                        os.makedirs(os.path.join(save_folder + "images", sub_folder_1, sub_folder_2))

                    if not os.path.exists(os.path.join(save_folder + "masks", sub_folder_1, sub_folder_2)):
                        os.makedirs(os.path.join(save_folder + "masks", sub_folder_1, sub_folder_2))

                    num_turns = round(360.0 / turn_angle)
                    for i in range(num_turns):
                        observations = sim.step("turn_right")

                        rgb = observations["color_sensor"][..., :3]
                        semantic = observations["semantic_sensor"]

                        print(str(ix) + "/" + str(len(real_points)) + "-" +str(i * turn_angle) + "deg", "rgb", rgb.shape, "semantic", semantic.shape)

                        # get semantic indices and labels
                        object_ids = get_all_masks(sim.semantic_scene)
                        mask = np.zeros_like(semantic, dtype=np.uint8)
                        for label in LABEL_MASK_OUTPUT_NUMBER:
                            for seg_index in object_ids[label]:
                                mask[semantic == seg_index] = LABEL_MASK_OUTPUT_NUMBER[label]

                        # reject if too many unknowns pixels. This is to reject image when they are out of the scene
                        num_unknown = np.count_nonzero(semantic == 0)

                        unknown_reject_threshold = 0.15
                        mask_pixels = mask.shape[0] * mask.shape[1]
                        if num_unknown / mask_pixels > unknown_reject_threshold:
                            print(f"Too many unknown pixels: {num_unknown / mask_pixels * 100:.2f}%", "num_unknown", num_unknown, "mask_pixels", mask_pixels, "rejecting...")
                            continue
                        else:
                            print(f"Unknown pixels: {num_unknown / mask_pixels * 100:.2f}%", "num_unknown", num_unknown, "mask_pixels", mask_pixels)

                        # clean mask
                        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((10, 10), np.uint8))

                        file = f"{scene_name}_height{str(height)[:5]}_pt{ix}_angle{i * turn_angle}"

                        # rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(save_folder + "images", sub_folder_1, sub_folder_2, file + ".jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                        cv2.imwrite(os.path.join(save_folder + "masks", sub_folder_1, sub_folder_2, file + ".png"), mask)

                        if display:
                            display_sample(rgb_obs=rgb, semantic_obs=semantic)


# function only deletes the contents of the folder of each scene, and not the folder itself. Delete all folders manually if needed in train or val folder
def get_all_scenes(
        dataset_main_path="/media/rishi/34D61BBAD61B7AF6/matterport/hm3d/scene_datasets/hm3d/",
        save_folder="/media/rishi/887C886A7C88553A/matterport_habitat_extracted/"
):
    def _get_scene_data(scene_file, config_file, save_folder):
        sim, agent = init_scene(scene_file=scene_file, scene_dataset_file=config_file)
        get_all_images(sim,
                       agent,
                       scene_name=scene_file.split(os.sep)[-2],
                       save_folder=save_folder,
                       display=False)
        sim.close()
        agent.close()

    train_folder = os.path.join(dataset_main_path, "train")
    val_folder = os.path.join(dataset_main_path, "val")

    train_config_file = os.path.join(train_folder, "hm3d_annotated_train_basis.scene_dataset_config.json")
    val_config_file = os.path.join(val_folder, "hm3d_annotated_val_basis.scene_dataset_config.json")

    for i, folder in enumerate(os.listdir(train_folder)):
        os.system('cls' if os.name == 'nt' else 'clear')

        print("Train: " + str(i + 1) + "/" + str(len(os.listdir(train_folder))))
        folder_path = os.path.join(train_folder, folder)
        if os.path.isdir(folder_path):
            alphabets = folder.split("-")[-1]
            file_name = f"{alphabets}.basis.glb"

            semantic_file_path = os.path.join(folder_path, f"{alphabets}.semantic.glb")
            if not os.path.exists(semantic_file_path):
                print("Semantic file does not exist for", folder_path)
                continue

            scene_file = os.path.join(folder_path, file_name)
            _get_scene_data(scene_file, train_config_file, save_folder + "train" + os.sep)

    for i, folder in enumerate(os.listdir(val_folder)):
        os.system('cls' if os.name == 'nt' else 'clear')

        print("Val: " + str(i + 1) + "/" + str(len(os.listdir(val_folder))))
        folder_path = os.path.join(val_folder, folder)
        if os.path.isdir(folder_path):
            alphabets = folder.split("-")[-1]
            file_name = f"{alphabets}.basis.glb"

            semantic_file_path = os.path.join(folder_path, f"{alphabets}.semantic.glb")
            if not os.path.exists(semantic_file_path):
                print("Semantic file does not exist for", folder_path)
                continue

            scene_file = os.path.join(folder_path, file_name)
            _get_scene_data(scene_file, val_config_file, save_folder + "val" + os.sep)


if __name__ == "__main__":
    # sim, agent = init_scene(scene_file="/home/rishi/programming/AI/experiments/datasets_extractor/data/Replica/apartment_0/habitat/mesh_semantic.ply",
    #                         scene_dataset_file="/home/rishi/programming/AI/experiments/datasets_extractor/data/Replica/apartment_0/habitat/info_semantic.json")
    # get_all_images(sim,
    #                agent,
    #                scene_name=scene_file_def.split(os.sep)[-2],
    #                save_folder="/media/rishi/887C886A7C88553A/datasets_extracted/replica/media/rishi/887C886A7C88553A/datasets_extracted/replica",
    #                display=True)
    # sim.close()
    # agent.close()

    get_all_scenes()
