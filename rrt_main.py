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
#q0 = np.zeros(len(joint_names))
#p0 = env.get_p_body(body_name='ur_base')+np.array([0.5,0.0,0.05])
R0 = transformation.rpy_deg2r([180,0,90])
#fix R0
#print(R0)
env.forward(q=q0,joint_names=joint_names) # update UR5e pose

# Set cylinder object poses
obj_names = env.get_body_names(prefix='')
n_obj = len(obj_names)

state = env.get_state()
xyz_click = np.array([-0.35, 1.3, 0.2])
body_name_clicked,p_body_clicked = env.get_body_name_closest(
            xyz_click,body_names=obj_names)
print(str(p_body_clicked) + " p_body_clicked")
print(str(body_name_clicked))

# Restore state and solve IKs
#env.set_state(**state,step=True)
cylinder_height = 0.2
q_grasp,_,_ = mujoco_parser.solve_ik(
    env                = env,
    joint_names_for_ik = joint_names,
    body_name_trgt     = 'panda0_leftfinger',
    q_init             = q0,
    p_trgt             = p_body_clicked + np.array([0,0,0.15+cylinder_height]),
    R_trgt             = R0,
)

print(q_grasp)
#start comment here
# Interpolate and smooth joint positions
times,traj_interp,traj_smt,times_anchor = utility.interpolate_and_smooth_nd(
    anchors   = np.vstack([q0,q_grasp]),
    HZ        = env.HZ,
    vel_limit = np.deg2rad(30), verbose=True
)

#print(traj_interp)
L = len(times)

# Open the gripper
'''
for tick in range(100):
    env.step( # dynamic update
        ctrl        = np.append(q0,[0.04, 0.04]),
        joint_names = joint_names+['panda0_finger_joint1', 'panda0_finger_joint2']
    )
'''

# Set collision configurations 
robot_body_names = env.get_body_names(prefix='panda0')
obj_body_names   = env.get_body_names(prefix='obj_')
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
    qpos = traj_interp[tick,:]
    env.forward(q=qpos,joint_names=joint_names)

    # Check collsision 
    is_feasible = rrt.is_qpos_feasible(
        env,qpos,joint_names,
        robot_body_names,obj_body_names,env_body_names)
    
    # Render
    if (tick%5)==0 or tick==(L-3):
        env.plot_text(p=np.array([0,0,1]),
                      label='[%d/%d] time:[%.2f]sec'%(tick,L,time))
        env.plot_body_T(body_name='panda0_leftfinger',axis_len=0.1)
        env.plot_body_T(body_name=body_name_clicked,axis_len=0.1)
        if not is_feasible:
            env.plot_sphere(
                p=env.get_p_body(body_name='panda0_leftfinger'),r=0.1,rgba=(1,0,0,0.5))
        env.render()
    # Proceed
    if tick < (L-3): tick = tick + 3
    # if tick == (L-1): tick = 0
    
print ("Done.") 
#end comment here

robot_body_names = env.get_body_names(prefix='panda0')
obj_body_names   = env.get_body_names(prefix='obj_')

env_body_names = []
# (optional)
obj_body_names.remove(body_name_clicked)
is_point_feasible = partial(
    rrt.is_qpos_feasible,
    env              = env,
    joint_names      = joint_names,
    robot_body_names = robot_body_names,
    obj_body_names   = obj_body_names,
    env_body_names   = env_body_names,
) # function of 'qpos'
is_point_to_point_connectable = partial(
    rrt.is_qpos_connectable,
    env              = env,
    joint_names      = joint_names,
    robot_body_names = robot_body_names,
    obj_body_names   = obj_body_names,
    env_body_names   = env_body_names,
    deg_th           = 5.0,
) # function of 'qpos1' and 'qpos2'
qpos = env.get_qpos_joints(joint_names=joint_names)

joint_limits = np.array([
    [-2.8973,  2.8973],
    [-1.7628,  1.7628],
    [-2.8973,  2.8973],
    [-3.0718, -0.4   ],
    [-2.8973,  2.8973],
    [-1.6573,  2.1127],
    [-2.8973,  2.8973]
])


point_min = joint_limits[:, 0]  
point_max = joint_limits[:, 1]
#point_min = -np.pi*np.ones(len(joint_names))
#point_max = +np.pi*np.ones(len(joint_names))
rrt = RapidlyExploringRandomTreesStarClass(
    name      ='RRT-Star-UR',
    point_min = point_min,
    point_max = point_max,
    goal_select_rate = 0.01,
    steer_len_max    = np.deg2rad(10),
    search_radius    = np.deg2rad(15), # 10, 30, 50
    norm_ord         = 2, # 2,np.inf,
    n_node_max       = 1500,
    TERMINATE_WHEN_GOAL_REACHED = False, SPEED_UP = True,
)

point_root,point_goal = q0,q_grasp
rrt.init_rrt_star(point_root=point_root,point_goal=point_goal,seed=0)
while True:
    # Randomly sample a point
    while True:
        if np.random.rand() <= rrt.goal_select_rate: 
            point_sample = rrt.point_goal
        else:
            point_sample = rrt.sample_point() # random sampling
        if is_point_feasible(qpos=point_sample): break

    # Get the nearest node ('node_nearest') to 'point_sample' from the tree
    node_nearest = rrt.get_node_nearest(point_sample)
    point_nearest = rrt.get_node_point(node_nearest)

    # Steering towards 'point_sample' to get 'point_new'
    point_new,cost_new = rrt.steer(node_nearest,point_sample)
    if point_new is None: continue # if the steering point is feasible

    if is_point_feasible(qpos=point_new) and \
        is_point_to_point_connectable(qpos1=point_nearest,qpos2=point_new):

        # Assign 'node_min' and initialize 'cost_min' with 'cost_new'
        node_min = node_nearest.copy()
        cost_min = cost_new

        # Select a set of nodes near 'point_new' => 'nodes_near'
        nodes_near = rrt.get_nodes_near(point_new)
    
        # For all 'node_near' find 'node_min'
        for node_near in nodes_near:
            point_near,cost_near = rrt.get_node_point_and_cost(node_near)
            if is_point_to_point_connectable(qpos1=point_near,qpos2=point_new):
                cost_prime = cost_near + rrt.get_dist(point_near,point_new)
                if cost_prime < cost_min:
                    cost_min = cost_near + rrt.get_dist(point_near,point_new)
                    node_min = node_near
        
        # Add 'node_new' and connect it with 'node_min'
        node_new = rrt.add_node(point=point_new,cost=cost_min,node_parent=node_min)

        # New node information for rewiring
        point_new,cost_new = rrt.get_node_point_and_cost(node_new)

        # Rewire
        for node_near in nodes_near:
            if node_near == 0: continue
            if rrt.get_node_parent(node_near) == node_new: continue
            point_near,cost_near = rrt.get_node_point_and_cost(node_near)
            cost_check = cost_new+rrt.get_dist(point_near,point_new)
            if (cost_check < cost_near) and \
                is_point_to_point_connectable(qpos1=point_near,qpos2=point_new):
                rrt.replace_node_parent(node=node_near,node_parent_new=node_new)

        # Re-update cost of all nodes
        if rrt.SPEED_UP: node_source = node_min
        else: node_source = 0
        rrt.update_nodes_cost(node_source=node_source,VERBOSE=False)

    # Print
    n_node = rrt.get_n_node()
    if (n_node % 1000 == 0) or (n_node == (rrt.n_node_max)):
        cost_goal = rrt.get_cost_goal() # cost to goal
        print ("n_node:[%d/%d], cost_goal:[%.5f]"%
               (n_node,rrt.n_node_max,cost_goal))
    
    # Terminate condition (if applicable)
    if n_node >= rrt.n_node_max: break # max node
    if (rrt.get_dist_to_goal() < 1e-6) and rrt.TERMINATE_WHEN_GOAL_REACHED: break
    
print ("Done.")

# ANIMATE RESULTS:
    # Get joint indices
# from 'start' to the point closest to the 'goal'
node_check = rrt.get_node_nearest(rrt.point_goal)
node_list = [node_check]
while node_check:
    node_parent = rrt.get_node_parent(node_check)
    node_list.append(node_parent)
    node_check = node_parent
node_list.reverse()
print ("node_list:%s"%(node_list))

# Get joint trajectories
q_anchors = np.zeros((len(node_list),len(joint_names)))
for idx,node in enumerate(node_list):
    qpos = rrt.get_node_point(node)
    q_anchors[idx,:] = qpos

times_interp,q_interp,_,_ = utility.interpolate_and_smooth_nd(
    anchors   = q_anchors,
    HZ        = env.HZ,
    acc_limit = np.deg2rad(10),
)
L = len(times_interp)
print ("len(node_list):[%d], L:[%d]"%(len(node_list),L))

# Animate
env.reset()
env.set_state(**state,step=True)
env.init_viewer()
tick = 0
while env.is_viewer_alive():
    # Update
    env.forward(q=q_interp[tick,:],joint_names=joint_names)
    # Render
    if tick%20 == 0 or tick == (L-1):
        env.plot_text(p=np.array([0,0,1]),label='tick:[%d/%d]'%(tick,L))
        env.render()
    # Increase tick
    if tick < (L-1): tick = tick + 1

print ("Done.")
#rrt.plot_tree()


 
#plot red nodes of selected path and every 10th node or something else (to reduce time)
#then modularize and implement block stacking.