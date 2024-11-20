import mujoco,sys
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import os

sys.path.append('../rrt_pick_stack/helper/')
sys.path.append('../rrt_pick_stack/mujoco_utils/')
from mujoco_utils.mujoco_parser import MuJoCoParserClass
from mujoco_utils import mujoco_parser
from helper import slider, transformation, utility

from rrt import RapidlyExploringRandomTreesStarClass
import rrt
ARM_nJnt = 7
xml_path = '../rrt_pick_stack/env/pick_stack.xml'
env = MuJoCoParserClass(name='pick_stack',rel_xml_path=xml_path,verbose=False)
print ("Done.")



#objects have prefix box instead of obj_


# Reset
np.random.seed(seed=0)
env.reset()


# Initialize UR5e
joint_names = ['panda0_joint1','panda0_joint2','panda0_joint3',
               'panda0_joint4','panda0_joint5','panda0_joint6', 'panda0_joint7']

q0 = np.array([ 0, 0, 0, -1.7359,  0,0.2277, 0])
#p0 = env.get_p_body(body_name='ur_base')+np.array([0.5,0.0,0.05])
R0 = transformation.rpy_deg2r([180,0,-90])
#fix R0
#print(R0)
env.forward(q=q0,joint_names=joint_names) # update UR5e pose

# Set cylinder object poses
obj_names = env.get_body_names(prefix='box')
n_obj = len(obj_names)

state = env.get_state()
xyz_click = np.array([0.7, 0.7, 0])
body_name_clicked,p_body_clicked = env.get_body_name_closest(
            xyz_click,body_names=obj_names)

# Restore state and solve IKs
env.set_state(**state,step=True)
q_grasp,_,_ = mujoco_parser.solve_ik(
    env                = env,
    joint_names_for_ik = joint_names,
    body_name_trgt     = 'panda0_leftfinger',
    q_init             = q0,
    p_trgt             = p_body_clicked + np.array([0,0,0.25]),
    R_trgt             = R0,
)
# Interpolate and smooth joint positions
times,traj_interp,traj_smt,times_anchor = utility.interpolate_and_smooth_nd(
    anchors   = np.vstack([q0,q_grasp]),
    HZ        = env.HZ,
    vel_limit = np.deg2rad(30),
)
L = len(times)

# Open the gripper
qpos = env.get_qpos_joints(joint_names=joint_names)
for tick in range(100):
    env.step( # dynamic update
        ctrl        = np.append(q0,[0.4, 0.4]),
        joint_names = joint_names+['panda0_finger_joint1', 'panda0_finger_joint2']
    )

# Set collision configurations 
robot_body_names = env.get_body_names(prefix='panda0')
obj_body_names   = env.get_body_names(prefix='box')
#env_body_names   = ['front_object_table','side_object_table']
env_body_names = []
# (optional) exclude 'body_name_clicked' from 'obj_body_names'
obj_body_names.remove(body_name_clicked)

# Loop
env.init_viewer(
    title='Checking collision while moving to the grasping pose',
    transparent=False,distance=3.0)
tick = 0
while env.is_viewer_alive():
    # Update
    time = times[tick]
    qpos = traj_smt[tick,:]
    env.forward(q=qpos,joint_names=joint_names)

    # Check collsision 
    is_feasible = rrt.is_qpos_feasible(
        env,qpos,joint_names,
        robot_body_names,obj_body_names,env_body_names)
    
    # Render
    if (tick%5)==0 or tick==(L-1):
        env.plot_text(p=np.array([0,0,1]),
                      label='[%d/%d] time:[%.2f]sec'%(tick,L,time))
        env.plot_body_T(body_name='panda0_leftfinger',axis_len=0.1)
        env.plot_body_T(body_name=body_name_clicked,axis_len=0.1)
        if not is_feasible:
            env.plot_sphere(
                p=env.get_p_body(body_name='panda0_leftfinger'),r=0.1,rgba=(1,0,0,0.5))
        env.render()
    # Proceed
    if tick < (L-1): tick = tick + 1
    # if tick == (L-1): tick = 0
    
print ("Done.")