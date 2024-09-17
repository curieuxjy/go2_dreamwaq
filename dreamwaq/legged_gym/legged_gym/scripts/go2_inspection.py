"""
Aidin Inspection code based on Joint Monkey
------------
- Animates degree-of-freedom ranges for a given asset.
- Demonstrates usage of DOF properties and states.
- Demonstrates line drawing utilities to visualize DOF frames (origin and axis).
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

DT = 1.0 / 60.0
ENV_LOWER_X = -2.0
ENV_LOWER_Y = -2.0
ENV_LOWER_Z = 0.0
ENV_UPPER_X = 2.0
ENV_UPPER_Y = 2.0
ENV_UPPER_Z = 4.0
ROBOT_POS_X = 0.0
ROBOT_POS_Y = 0.0
ROBOT_POS_Z = 0.5 + 0.2 # STAND
# ROBOT_POS_Z = 0.22 # SIT
DEFAULTS = np.array([0.0, 0.4, 0.15] * 4) # STAND
# DEFAULTS = np.array([0.0, -0.12, 1.05] * 4) # SIT

# 뷰어 변수
VIEW_POS_X = -1
VIEW_POS_Y = -4.5
VIEW_POS_Z = 1.5
TARGET_POS_X = 0
TARGET_POS_Y = 0
TARGET_POS_Z = 1

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

# camera
D435 = gymapi.Vec3(0.0, 0.0, 0.0)
D435_INDEX = 22
# marks for camera
AXES_GEOM = gymutil.AxesGeometry(0.1)
# a wireframe sphere
sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
sphere_pose = gymapi.Transform(r=sphere_rot)
SPHERE_GEOM = gymutil.WireframeSphereGeometry(0.02, 12, 12, sphere_pose, color=(1, 1, 0))

def clamp(x, min_value, max_value):
    return max(min(x, max_value), min_value)

def print_asset_info(asset, name):
    """
    from asset_info.py
    """
    print("======== Asset info %s: ========" % (name))
    num_bodies = gym.get_asset_rigid_body_count(asset)
    num_joints = gym.get_asset_joint_count(asset)
    num_dofs = gym.get_asset_dof_count(asset)
    print("Got %d bodies, %d joints, and %d DOFs" %
          (num_bodies, num_joints, num_dofs))

    # Iterate through bodies
    print("Bodies:")
    for i in range(num_bodies):
        name = gym.get_asset_rigid_body_name(asset, i)
        print(" %2d: '%s'" % (i, name))
    return


def new_sub_terrain(vertical_scale, horizontal_scale):
    return SubTerrain(width=TERRAIN_WIDTH,
                      length=TERRAIN_LENGTH,
                      vertical_scale=vertical_scale,
                      horizontal_scale=horizontal_scale)


def create_plane(sim):
    # add ground plane
    plane_params = gymapi.PlaneParams()
    plane_params.normal = gymapi.Vec3(0, 0, 1)
    # print("plane_params", plane_params)
    gym.add_ground(sim, plane_params)


def create_stairs(horizontal_scale, vertical_scale, step_width, step_height):
    stair = stairs_terrain(new_sub_terrain(vertical_scale, horizontal_scale),
                           step_width=step_width,
                           step_height=step_height)

    # print(stair.terrain_name)
    # print(stair.vertical_scale)
    # print(stair.horizontal_scale)
    # print(stair.width)
    # print(stair.length)
    # print(stair.height_field_raw.shape)

    heightfield = stair.height_field_raw

    vertices, triangles = convert_heightfield_to_trimesh(heightfield,
                                                         horizontal_scale=horizontal_scale,
                                                         vertical_scale=vertical_scale,
                                                         slope_threshold=0)
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
    viewer_pos = gymapi.Vec3(VIEW_POS_X, VIEW_POS_Y, VIEW_POS_Z)  # (2, -1.5, 3) # (17.2, 2.0, 16)
    cam_target = gymapi.Vec3(TARGET_POS_X, TARGET_POS_Y, TARGET_POS_Z)  # (5, -2.5, 13)
    gym.viewer_camera_look_at(viewer, None, viewer_pos, cam_target)
    return viewer


def create_actor(sim, env, fixed=True, dof_print=False):
    asset_root = "../legged_gym/resources/robots"
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
    print("Loading asset '%s' from '%s'" % (asset_file, asset_root))
    robot_asset = gym.load_asset(sim, asset_root, asset_file, asset_options)
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
    dof_positions = dof_states['pos']

    # get the limit-related slices of the DOF properties array
    stiffnesses = dof_props['stiffness']
    dampings = dof_props['damping']
    armatures = dof_props['armature']  # 전기자
    has_limits = dof_props['hasLimits']
    lower_limits = dof_props['lower']
    upper_limits = dof_props['upper']

    # initialize default positions, limits, and speeds
    # (make sure they are in reasonable ranges)
    defaults = DEFAULTS # DEFAULTS
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
            speeds[i] = 1.0 * clamp(2 * (upper_limits[i] - lower_limits[i]), 0.25 * math.pi, 3.0 * math.pi)
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
    # AIDIN ACTOR
    pose = gymapi.Transform()
    pose.p = gymapi.Vec3(ROBOT_POS_X, ROBOT_POS_Y, ROBOT_POS_Z)
    # random roll deg -40
    # from_euler_zyx(x-roll, y, z)
    # random_rad = random.uniform(-1.5, 1.5) # 90deg == 1.57rad
    # random_rad = 0.6
    pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)# (random_rad, 0, 0)

    # pose.r = gymapi.Quat(0, 0, 0, 1) #-0.7071068, 0.7071068)  # aidin이 y축 정렬이라 x축 정렬로 바꿔줌
    # necessary when loading an asset that is defined using z-up convention
    # into a simulation that uses y-up convention.
    aidin_actor = gym.create_actor(env, robot_asset, pose, "actor", 0, 0)
    gym.set_actor_dof_states(env, aidin_actor, dof_states, gymapi.STATE_ALL)
    return robot_asset, aidin_actor, dof_props, dof_states, dof_positions, speeds # , d435_handle, d435_transform

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
    rigid_body_states = gymtorch.wrap_tensor(rigid_body_states) # shape (num_rigid_bodies, 13)
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

    # print("--- RF {} --- RB {} --- LF {} --- LB {} ---".format(rigid_body_states[22][2],
    #                                                            rigid_body_states[16][2],
    #                                                            rigid_body_states[10][2],
    #                                                            rigid_body_states[4][2]))

def create_sim():
    # initialize gym
    gym = gymapi.acquire_gym()

    # configure sim
    sim_params = gymapi.SimParams()
    sim_params.dt = DT
    sim_params.up_axis = gymapi.UP_AXIS_Z
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

    args = gymutil.parse_arguments(description="Aidin test environment")
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

    sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
    if sim is None:
        print("*** Failed to create sim")
        quit()
    return gym, sim


# simple asset descriptor for selecting from a list
class AssetDesc:
    def __init__(self, file_name, flip_visual_attachments=False):
        self.file_name = file_name
        self.flip_visual_attachments = flip_visual_attachments


asset_descriptors = [
    AssetDesc("go2/urdf/anymal_c.urdf", True)
]

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
env_lower = gymapi.Vec3(-2.0, -2.0, 0)
env_upper = gymapi.Vec3(2.0, 2.0, 4.0)
env = gym.create_env(sim, env_lower, env_upper, num_per_row)

# 5. actor 만들기 (asset을 불러오고 난 후 env에 할당)
# create_ball(sim, env)
robot_asset, aidin_actor, dof_props, dof_states, dof_positions, speeds = create_actor(sim, env, dof_print=False, fixed=True)


# 6. 애니메이션을 위한 준비
lower_limits = dof_props['lower']
upper_limits = dof_props['upper']
# initialize animation state
anim_state = ANIM_SEEK_LOWER
current_dof = 0
gym.set_actor_dof_states(env, aidin_actor, dof_states, gymapi.STATE_POS)
# 6. 카메라 센서정보를 얻기 위한 준비
if not os.path.exists("aidin"):
    os.mkdir("aidin")
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
        if dof_positions[current_dof] <= DEFAULTS[current_dof]:# DEFAULTS[current_dof]:
            dof_positions[current_dof] = DEFAULTS[current_dof]# DEFAULTS[current_dof]
            anim_state = ANIM_FINISHED
    elif anim_state == ANIM_FINISHED:
        dof_positions[current_dof] = DEFAULTS[current_dof] # DEFAULTS[current_dof]
        current_dof = (current_dof + 1) % 12  # num_dofs
        anim_state = ANIM_SEEK_LOWER

    gym.clear_lines(viewer)
    gym.set_actor_dof_states(env, aidin_actor, dof_states, gymapi.STATE_POS)
    # --------------------------------------------

    # 10. dof 시각화 get the DOF frame (origin and axis)
    dof_handle = gym.get_actor_dof_handle(env, aidin_actor, current_dof)
    frame = gym.get_dof_frame(env, dof_handle)
    # draw a line from DOF origin along the DOF axis
    p1 = frame.origin
    p2 = frame.origin + frame.axis * 0.7
    color = gymapi.Vec3(1.0, 0.0, 0.0)
    gymutil.draw_line(p1, p2, color, gym, viewer, env)

    # 11. 카메라 위치 표시
    # gymutil.draw_lines(AXES_GEOM, gym, viewer, env, d435_transform)
    # gymutil.draw_lines(SPHERE_GEOM, gym, viewer, env, d435_transform)

    # 12. 시뮬레이션 스텝 실행
    gym.step_graphics(sim)

    print_any_state(sim)

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