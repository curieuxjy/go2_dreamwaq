"""
Go2 Inspection code based on Joint Monkey
------------------------------------------------------------------------------------
- Animates degree-of-freedom ranges for a given asset.
- Demonstrates usage of DOF properties and states.
- Demonstrates line drawing utilities to visualize DOF frames (origin and axis).
------------------------------------------------------------------------------------
- egocentric camera
- lidar ray visualization
- terrain
"""

import math
import numpy as np
import random
from isaacgym import gymapi, gymutil
from isaacgym.terrain_utils import *
import matplotlib.pyplot as plt
from isaacgym import gymtorch
import time
import os
from PIL import Image as im

# Resources
RESOURCE_ROOT = "../dreamwaq/legged_gym/resources"
# Simulation Constants
DT = 1.0 / 60.0

# Environment Bounds
ENV_LOWER = gymapi.Vec3(-2.0, -2.0, 0.0)
ENV_UPPER = gymapi.Vec3(2.0, 2.0, 4.0)

# Robot Starting Position
ROBOT_POS = gymapi.Vec3(0.0, 0.0, 0.42)

# Default Joint Angles
DEFAULT_JOINT_ANGLES = {
    "FL_hip_joint": 0.1,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -1.5,
    "FR_hip_joint": -0.1,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint": -1.5,
    "RL_hip_joint": 0.1,
    "RL_thigh_joint": 1.0,
    "RL_calf_joint": -1.5,
    "RR_hip_joint": -0.1,
    "RR_thigh_joint": 1.0,
    "RR_calf_joint": -1.5,
}
DEFAULTS = list(DEFAULT_JOINT_ANGLES.values())

# Viewer Configuration
VIEWER_POS = gymapi.Vec3(1.0, 1.0, 1.0)
TARGET_POS = gymapi.Vec3(0, 0, 0)

# 지형 변수
TERRAIN_WIDTH = 600  # unit
TERRAIN_LENGTH = 600  # unit
TERRAIN_TRANS_X = -2  # [m]
TERRAIN_TRANS_Y = -2  # [m]
TERRAIN_TRANS_Z = 1  # [m]
STEP_WIDTH = 0.30  # [m]
STEP_HEIGHT = -0.17  # [m]

# smaller unit
HORIZONTAL_SCALE = 0.01  # [m] -> cm 단위로 생각
VERTICAL_SCALE = 0.01  # [m] -> UNIT

# joint animation states
ANIM_SEEK_LOWER = 1
ANIM_SEEK_UPPER = 2
ANIM_SEEK_DEFAULT = 3
ANIM_FINISHED = 4

# marks for camera
AXES_GEOM = gymutil.AxesGeometry(0.1)
# a wireframe sphere
sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
sphere_pose = gymapi.Transform(r=sphere_rot)
SPHERE_GEOM = gymutil.WireframeSphereGeometry(
    0.02, 12, 12, sphere_pose, color=(1, 1, 0)
)


def create_egocentric_camera(env, actor_handle):
    camera_sensor_pose = gymapi.Vec3(0.0, 0.0, 0.0)
    CAMERA_SENSOR_INDEX = 22
    poses = gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_POS)["pose"]

    # Get pose for the handles
    d435_handle_pose = gymapi.Transform.from_buffer(poses[CAMERA_SENSOR_INDEX])
    # print("pose: ", d435_handle_pose)

    # Offset d435 transforms to compute d435 locations
    d435_point = d435_handle_pose.transform_point(camera_sensor_pose)
    # print("d435 point: ", d435_point) == transition

    # Create transform from d435 location and d435 rotation
    egocentric_cam_transform = gymapi.Transform(d435_point, d435_handle_pose.r)

    cam_box_handle = gym.get_actor_rigid_body_handle(
        env, actor_handle, CAMERA_SENSOR_INDEX
    )
    cam_box_state = gym.get_actor_rigid_body_states(
        env, actor_handle, gymapi.STATE_POS
    )[-1]

    camera_props = gymapi.CameraProperties()
    # camera_props.horizontal_fov = 1.047197551 # 60 degree
    camera_props.width = 1280
    camera_props.height = 720
    # camera_props.use_collision_geometry = True
    egocentric_cam_handle = gym.create_camera_sensor(env, camera_props)
    # env_cam = gym.create_camera_sensor(env, camera_props)
    # camera_offset = gymapi.Transform() # actor 기준
    # x = 0.45
    # y = 0
    # z = 0.57
    camera_offset = gymapi.Vec3(
        0, 0, 0
    )  # egocentric_cam_transform.p # gymapi.Vec3(x, y, z)
    camera_rotation = gymapi.Quat(
        0, 0, 0.7071068, 0.7071068
    )  # egocentric_cam_transform.r # [ x: -90, y: 90, z: 0 ] # gymapi.Quat.from_euler_zyx(0, -1.57, 0) # [ x: 0, y: -89.9557653, z: 0 ]
    print(camera_offset, camera_rotation)
    # name = 'd435'
    # Depth Field of View (FOV):
    # 87° × 58° (1.51844 × 1.01229)
    # Depth output resolution:
    # Up to 1280 × 720
    # domain_randomization.py // graphics.py
    gym.attach_camera_to_body(
        egocentric_cam_handle,
        env,
        cam_box_handle,
        gymapi.Transform(camera_offset, camera_rotation),
        gymapi.FOLLOW_TRANSFORM,
    )
    return egocentric_cam_handle, egocentric_cam_transform


def create_ball(sim, env, fixed=False):
    ball_urdf = "ball.urdf"
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = fixed
    ball_asset = gym.load_asset(sim, ball_urdf, asset_options)
    ball_pose = gymapi.Transform()
    ball_pose.p = gymapi.Vec3(0.3, 0, 0.1)
    ball_pose.r = gymapi.Quat(0, 0, 0, 1)
    asset_name = "ball"
    # create_actor -> group: int = - 1, filter: int = - 1, segmentationId: int = 0
    ball_handle = gym.create_actor(env, ball_asset, ball_pose, asset_name, 0, 0)
    return ball_handle


def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)


def print_asset_info(asset, name):
    print("======== Asset info %s: ========" % (name))
    num_bodies = gym.get_asset_rigid_body_count(asset)
    num_joints = gym.get_asset_joint_count(asset)
    num_dofs = gym.get_asset_dof_count(asset)
    print("Got %d bodies, %d joints, and %d DOFs" % (num_bodies, num_joints, num_dofs))

    # Iterate through bodies
    print("Bodies:")
    for i in range(num_bodies):
        name = gym.get_asset_rigid_body_name(asset, i)
        print(" %2d: '%s'" % (i, name))
    return


def new_sub_terrain(vertical_scale, horizontal_scale):
    return SubTerrain(
        width=TERRAIN_WIDTH,
        length=TERRAIN_LENGTH,
        vertical_scale=vertical_scale,
        horizontal_scale=horizontal_scale,
    )


def create_plane(sim):
    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    # print("plane_params", plane_params)
    gym.add_ground(sim, plane_params)


def create_stairs(horizontal_scale, vertical_scale, step_width, step_height):
    stair = stairs_terrain(
        new_sub_terrain(vertical_scale, horizontal_scale),
        step_width=step_width,
        step_height=step_height,
    )

    # print(stair.terrain_name)
    # print(stair.vertical_scale)
    # print(stair.horizontal_scale)
    # print(stair.width)
    # print(stair.length)
    # print(stair.height_field_raw.shape)

    heightfield = stair.height_field_raw

    vertices, triangles = convert_heightfield_to_trimesh(
        heightfield,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        slope_threshold=0,
    )
    tm_params = gymapi.TriangleMeshParams()
    tm_params.nb_vertices = vertices.shape[0]
    tm_params.nb_triangles = triangles.shape[0]
    tm_params.transform.p.x = TERRAIN_TRANS_X
    tm_params.transform.p.y = TERRAIN_TRANS_Y
    tm_params.transform.p.z = TERRAIN_TRANS_Z
    gym.add_triangle_mesh(sim, vertices.flatten(), triangles.flatten(), tm_params)
    # hf_params = gymapi.HeightFieldParams()
    # hf_params.transform.p.x = 0
    # hf_params.transform.p.y = 0
    # gym.add_heightfield(sim, heightfield, hf_params)
    return


def create_viewer(sim):
    # create viewer
    viewer_properties = gymapi.CameraProperties()
    viewer_properties.use_collision_geometry = True
    viewer = gym.create_viewer(sim, viewer_properties)
    if viewer is None:
        print("*** Failed to create viewer")
        quit()
    # position the camera
    # viewer_pos = gymapi.Vec3(VIEW_POS_X, VIEW_POS_Y, VIEW_POS_Z)  # (2, -1.5, 3) # (17.2, 2.0, 16)
    # cam_target = gymapi.Vec3(TARGET_POS_X, TARGET_POS_Y, TARGET_POS_Z)  # (5, -2.5, 13)
    gym.viewer_camera_look_at(viewer, None, VIEWER_POS, TARGET_POS)
    return viewer


def create_robot_actor(sim, env, fixed=True, dof_print=False):
    # asset_root = "dreamwaq/legged_gym/resources/"
    asset_file = asset_descriptors[0].file_name
    asset_options = gymapi.AssetOptions()
    asset_options.fix_base_link = fixed
    asset_options.flip_visual_attachments = asset_descriptors[0].flip_visual_attachments
    # asset_options.use_mesh_materials = True  # False
    # asset_options.use_physx_armature = True
    # asset_options.override_com = True
    # asset_options.override_inertia = True
    # asset_options.vhacd_enabled = False

    # ASSET
    print("Loading asset '%s' from '%s'" % (asset_file, RESOURCE_ROOT))
    robot_asset = gym.load_asset(sim, RESOURCE_ROOT, asset_file, asset_options)
    print_asset_info(robot_asset, "robot")

    # DOF PROPERTIES
    # get array of DOF names
    dof_names = gym.get_asset_dof_names(robot_asset)
    print("DOF names: %s" % dof_names)

    # get array of DOF properties
    dof_props = gym.get_asset_dof_properties(robot_asset)
    # print("DOF properties: %s" % dof_props)
    num_dofs = gym.get_asset_dof_count(robot_asset)
    # create an array of DOF states that will be used to update the actors
    dof_states = np.zeros(num_dofs, dtype=gymapi.DofState.dtype)

    # get list of DOF types
    dof_types = [gym.get_asset_dof_type(robot_asset, i) for i in range(num_dofs)]

    # get the position slice of the DOF state array
    dof_positions = dof_states["pos"]

    # get the limit-related slices of the DOF properties array
    stiffnesses = dof_props["stiffness"]
    dampings = dof_props["damping"]
    armatures = dof_props["armature"]  # 전기자
    has_limits = dof_props["hasLimits"]
    lower_limits = dof_props["lower"]
    upper_limits = dof_props["upper"]

    # initialize default positions, limits, and speeds
    # (make sure they are in reasonable ranges)
    defaults = DEFAULTS  # DEFAULTS
    speeds = np.zeros(num_dofs)

    for i in range(num_dofs):
        if has_limits[i]:
            if dof_types[i] == gymapi.DOF_ROTATION:
                lower_limits[i] = clamp(lower_limits[i], -math.pi, math.pi)
                upper_limits[i] = clamp(upper_limits[i], -math.pi, math.pi)
        else:
            # set reasonable animation limits for unlimited joints
            if dof_types[i] == gymapi.DOF_ROTATION:
                # unlimited revolute joint
                lower_limits[i] = -math.pi
                upper_limits[i] = math.pi
            elif dof_types[i] == gymapi.DOF_TRANSLATION:
                # unlimited prismatic joint
                lower_limits[i] = -1.0
                upper_limits[i] = 1.0
        # set DOF position to default
        dof_positions[i] = defaults[i]
        # set speed depending on DOF type and range of motion
        if dof_types[i] == gymapi.DOF_ROTATION:
            # speed_scale = 1.0
            speeds[i] = 1.0 * clamp(
                2 * (upper_limits[i] - lower_limits[i]), 0.25 * math.pi, 3.0 * math.pi
            )
        else:
            speeds[i] = 1.0 * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.1, 7.0)

    if dof_print:
        for i in range(num_dofs):
            print("DOF %d" % i)
            print("  Name:     '%s'" % dof_names[i])
            print("  Type:     %s" % gym.get_dof_type_string(dof_types[i]))
            print("  Stiffness:  %r" % stiffnesses[i])
            print("  Damping:  %r" % dampings[i])
            print("  Armature:  %r" % armatures[i])
            print("  Limited?  %r" % has_limits[i])
            print("  Default:  %r" % defaults[i])
            print("  Speed:    %r" % speeds[i])
            if has_limits[i]:
                print("    Lower   %f" % lower_limits[i])
                print("    Upper   %f" % upper_limits[i])

    # ACTOR
    pose = gymapi.Transform()
    pose.p = ROBOT_POS
    # random roll deg -40
    # from_euler_zyx(x-roll, y, z)
    # random_rad = random.uniform(-1.5, 1.5) # 90deg == 1.57rad
    # random_rad = 0.6
    pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)  # (random_rad, 0, 0)

    # pose.r = gymapi.Quat(0, 0, 0, 1) #-0.7071068, 0.7071068)
    # necessary when loading an asset that is defined using z-up convention
    # into a simulation that uses y-up convention.
    robot_actor = gym.create_actor(env, robot_asset, pose, "actor", 0, 0)
    gym.set_actor_dof_states(env, robot_actor, dof_states, gymapi.STATE_ALL)

    # Egocentric camera
    egocentric_cam_handle, egocentric_cam_transform = create_egocentric_camera(
        env, robot_actor
    )

    return (
        robot_actor,
        dof_props,
        dof_states,
        dof_positions,
        speeds,
        egocentric_cam_handle,
        egocentric_cam_transform,
    )


def print_any_state(sim):
    root_state = gym.acquire_actor_root_state_tensor(sim)
    dof_state_tensor = gym.acquire_dof_state_tensor(sim)
    net_contact_forces = gym.acquire_net_contact_force_tensor(sim)
    rigid_body_states = gym.acquire_rigid_body_state_tensor(sim)
    # get_으로 시작하는 함수는 acquire_로 시작하는 함수들의 이전 버젼

    gym.refresh_dof_state_tensor(sim)
    gym.refresh_actor_root_state_tensor(sim)
    gym.refresh_net_contact_force_tensor(sim)
    gym.refresh_rigid_body_state_tensor(sim)

    root_states = gymtorch.wrap_tensor(root_state)
    dof_state = gymtorch.wrap_tensor(dof_state_tensor)
    net_contact_forces = gymtorch.wrap_tensor(net_contact_forces)
    rigid_body_states = gymtorch.wrap_tensor(
        rigid_body_states
    )  # shape (num_rigid_bodies, 13)
    # print(rigid_body_states.shape) # torch.Size([25, 13])

    dof_pos = dof_state.view(12, 2)[..., 0]
    dof_vel = dof_state.view(12, 2)[..., 1]
    # position([0:3]),
    # rotation([3:7]),
    # linear velocity([7:10]),
    # and angular velocity([10:13])
    base_pos = root_states[0][0:3]
    base_quat = root_states[0][3:7]

    ###############################################################
    print("base pose: ", base_pos)
    print("base quat: ", base_quat)
    print("dof pos: ", dof_pos)
    # order: (same to viewer ordering)
    # "LB_Scap_Joint", "LB_Hip_Joint", "LB_Knee_Joint",
    # "LF_Scap_Joint", "LF_Hip_Joint", "LF_Knee_Joint",
    # "RB_Scap_Joint", "RB_Hip_Joint", "RB_Knee_Joint",
    # "RF_Scap_Joint", "RF_Hip_Joint", "RF_Knee_Joint",

    # print(">>>when {} z base position,  Z: ".format(ROBOT_POS_Z), base_pos[:,2])
    # knee 3,9,15,21 foot 4,10,16,22
    # print(">>> RF foot ", net_contact_forces[22])
    # print(">>> LB ", net_contact_forces[3])
    # print(">>> LF ", net_contact_forces[8])
    # print(">>> RB ", net_contact_forces[13])
    # print(">>> RF ", net_contact_forces[18])


def create_sim():
    # initialize gym
    gym = gymapi.acquire_gym()

    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.dt = DT
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    args = gymutil.parse_arguments(description="Go2 test environment")
    if args.physics_engine == gymapi.SIM_FLEX:
        pass
    elif args.physics_engine == gymapi.SIM_PHYSX:
        # sim_params.physx.contact_collection = gymapi.CC_LAST_SUBSTEP  # Collect contacts for last substep only (value = 1)
        # sim_params.physx.contact_offset = 0.0
        sim_params.physx.solver_type = 1  # better but expensive
        # 0 : PGS (Iterative sequential impulse solver
        # 1 : TGS (Non-linear iterative solver, more robust but slightly more expensive
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.num_threads = args.num_threads
        sim_params.physx.use_gpu = args.use_gpu

    sim_params.use_gpu_pipeline = False
    if args.use_gpu_pipeline:
        print("WARNING: Forcing CPU pipeline.")

    sim = gym.create_sim(
        args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params
    )
    if sim is None:
        print("*** Failed to create sim")
        quit()
    return gym, sim


# simple asset descriptor for selecting from a list
class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments


if __name__ == "__main__":

    asset_descriptors = [AssetDesc("robots/go2/urdf/go2.urdf", True)]

    # 1. 시뮬레이션 객체 불러오기
    gym, sim = create_sim()

    # 2. 지형 불러오기
    # TERRAIN
    # horizontal_scale = HORIZONTAL_SCALE
    # vertical_scale = VERTICAL_SCALE
    # step_width = STEP_WIDTH
    # step_height = STEP_HEIGHT
    create_plane(sim)
    # create_stairs(horizontal_scale, vertical_scale, step_width, step_height)
    # create_pyramid_stairs(horizontal_scale, vertical_scale, step_width, step_height)

    # 3. 창을 띄우는 뷰어 만들기
    # VIEWER
    viewer = create_viewer(sim)

    # 4. 환경 만들기(여러 환경을 만들때 한개의 sim 안에 env들 여러개를 만들 수 있음)
    num_per_row = 1
    env = gym.create_env(sim, ENV_LOWER, ENV_UPPER, num_per_row)

    # 5. actor 만들기 (asset을 불러오고 난 후 env에 할당)
    ball = create_ball(sim, env)
    (
        robot_actor,
        dof_props,
        dof_states,
        dof_positions,
        speeds,
        egocentric_cam_handle,
        egocentric_cam_transform,
    ) = create_robot_actor(sim, env, dof_print=False, fixed=True)

    # 6. 애니메이션을 위한 준비
    lower_limits = dof_props["lower"]
    upper_limits = dof_props["upper"]
    # initialize animation state
    anim_state = ANIM_SEEK_LOWER
    current_dof = 0
    gym.set_actor_dof_states(env, robot_actor, dof_states, gymapi.STATE_POS)

    # 6. 카메라 센서정보를 얻기 위한 준비
    # if not os.path.exists("egocentric_cam"):
    #     os.mkdir("egocentric_cam")
    frame_count = 0
    time = 0
    while not gym.query_viewer_has_closed(viewer):
        # 7. 시뮬레이션 시작
        gym.simulate(sim)
        time += 1
        gym.fetch_results(sim, True)

        # ---------------------ANIMATION-----------------------
        speed = speeds[current_dof]

        # 9. animate the dofs
        if anim_state == ANIM_SEEK_LOWER:
            dof_positions[current_dof] -= speed * DT
            if dof_positions[current_dof] <= lower_limits[current_dof]:
                dof_positions[current_dof] = lower_limits[current_dof]
                anim_state = ANIM_SEEK_UPPER
        elif anim_state == ANIM_SEEK_UPPER:
            dof_positions[current_dof] += speed * DT
            if dof_positions[current_dof] >= upper_limits[current_dof]:
                dof_positions[current_dof] = upper_limits[current_dof]
                anim_state = ANIM_SEEK_DEFAULT
        if anim_state == ANIM_SEEK_DEFAULT:
            dof_positions[current_dof] -= speed * DT
            if (
                dof_positions[current_dof] <= DEFAULTS[current_dof]
            ):  # DEFAULTS[current_dof]:
                dof_positions[current_dof] = DEFAULTS[
                    current_dof
                ]  # DEFAULTS[current_dof]
                anim_state = ANIM_FINISHED
        elif anim_state == ANIM_FINISHED:
            dof_positions[current_dof] = DEFAULTS[current_dof]  # DEFAULTS[current_dof]
            current_dof = (current_dof + 1) % 12  # num_dofs
            anim_state = ANIM_SEEK_LOWER

        gym.clear_lines(viewer)
        gym.set_actor_dof_states(env, robot_actor, dof_states, gymapi.STATE_POS)

        # 10. dof 시각화 get the DOF frame (origin and axis)
        dof_handle = gym.get_actor_dof_handle(env, robot_actor, current_dof)
        frame = gym.get_dof_frame(env, dof_handle)
        # draw a line from DOF origin along the DOF axis
        p1 = frame.origin
        p2 = frame.origin + frame.axis * 0.5
        color = gymapi.Vec3(1.0, 0.0, 0.0)
        gymutil.draw_line(p1, p2, color, gym, viewer, env)

        # 11. 카메라 위치 표시
        # gymutil.draw_lines(AXES_GEOM, gym, viewer, env, egocentric_cam_transform)
        # gymutil.draw_lines(SPHERE_GEOM, gym, viewer, env, egocentric_cam_transform)

        # 12. 시뮬레이션 스텝 실행
        gym.step_graphics(sim)

        print_any_state(sim)

        if frame_count // 100 == 0:
            print_any_state(sim)
            frame_count = 0

        # 13. 카메라 센서 실행
        gym.render_all_camera_sensors(sim)
        # gym.start_access_image_tensors(sim)
        # test_image = gym.get_camera_image(sim, env, camera_sensor, gymapi.IMAGE_DEPTH)
        # print(np.shape(test_image))
        # plt.matshow(test_image)
        # plt.show()

        # 14. 일정 타임 스텝에 이미지 저장
        if np.mod(frame_count, 30) == 0 and frame_count < 200:
            print("captured!")
            # The gym utility to write images to disk is recommended only for RGB images.
            rgb_filename = "egocentric_cam/rgb_frame%d.png" % (frame_count)
            gym.write_camera_image_to_file(
                sim, env, d435_handle, gymapi.IMAGE_COLOR, rgb_filename
            )
            # gym.write_camera_image_to_file(sim, env, env_cam, gymapi.IMAGE_COLOR, rgb_filename)

            # Retrieve image data directly. Use this for Depth, Segmentation, and Optical Flow images
            # Here we retrieve a depth image, normalize it to be visible in an
            # output image and then write it to disk using Pillow
            depth_image = gym.get_camera_image(
                sim, env, d435_handle, gymapi.IMAGE_DEPTH
            )
            # depth_image = gym.get_camera_image(sim, env, env_cam, gymapi.IMAGE_DEPTH)

            # -inf implies no depth value, set it to zero. output will be black.
            depth_image[depth_image == -np.inf] = 0
            # clamp depth image to 10 meters to make output image human friendly
            depth_image[depth_image < -10] = -10

            # flip the direction so near-objects are light and far objects are dark
            normalized_depth = -255.0 * (depth_image / np.min(depth_image + 1e-4))

            # Convert to a pillow image and write it to disk
            normalized_depth_image = im.fromarray(
                normalized_depth.astype(np.uint8), mode="L"
            )
            normalized_depth_image.save(
                "egocentric_cam/depth_frame%d.jpg" % (frame_count)
            )

        # 15. 뷰어 업데이트
        gym.draw_viewer(viewer, sim, True)
        # cam_trans = gym.get_viewer_camera_transform(viewer, env)
        # print(cam_trans.p) # viewer position

        # 16. Wait for dt to elapse in real time.
        # This synchronizes the physics simulation with the rendering rate.
        gym.sync_frame_time(sim)
        frame_count += 1

        # if time%60==0:
        #     print("time(sec): ",time//60)

    print("Done")

    gym.destroy_viewer(viewer)
    gym.destroy_sim(sim)
