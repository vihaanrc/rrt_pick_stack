import mujoco,sys
import matplotlib.pyplot as plt
from functools import partial
import os

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

sys.path.append('../rrt_pick_stack/helper/')
sys.path.append('../rrt_pick_stack/mujoco_utils/')
from mujoco_utils.mujoco_parser import MuJoCoParserClass
from mujoco_utils import mujoco_parser
from helper import slider, transformation, utility

from rrt import RapidlyExploringRandomTreesStarClass
import rrt

def visualize_rrt_with_objects(RRT, object_params=None, buffer_size=0.1):
    """
    Visualize the RRT tree, highlighting the path area, start/goal points, and objects.
    THIS IS IN QPOS, so shows AREAS WHERE ROBOT HAS TO AVOID, EMPHASIZES HOW QPOS CHANGES ARE SLIGHT
    Parameters:
        rrt: RapidlyExploringRandomTreesStarClass
            The RRT object containing the tree.
        path_nodes: list
            A list of nodes in the path to goal (optional, used for highlighting the path).
        object_params: dict
            Parameters for the object to visualize (e.g., 'type': 'cylinder', 'center': [x, y, z], 'radius': r, 'height': h).
        buffer_size: float
            The size of the buffer area around the path nodes to visualize.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Extract node points from the RRT
    node_check = RRT.get_node_nearest(RRT.point_goal)
    path_nodes = [node_check]
    while node_check:
        node_parent = RRT.get_node_parent(node_check)
        path_nodes.append(node_parent)
        node_check = node_parent
    path_nodes.reverse()

    node_points = []
    for node in RRT.get_nodes():
        node_points.append(RRT.get_node_point(node))

    node_points = np.array(node_points)

    # Plot all nodes in the tree
    ax.scatter(node_points[:, 0], node_points[:, 1], node_points[:, 2], c='b', marker='o', label='Nodes')

    # Highlight path nodes in red (optional)
    if path_nodes:
        path_points = [RRT.get_node_point(node) for node in path_nodes]
        path_points = np.array(path_points)
        ax.scatter(path_points[:, 0], path_points[:, 1], path_points[:, 2], c='r', marker='o', label='Path')

        # Create shaded buffer around the path nodes
        for point in path_points:
            ax.scatter(point[0], point[1], point[2], c='r', marker='o', s=100)
            ax.scatter(point[0], point[1], point[2], c='r', marker='o', alpha=0.3, s=500)

    # Highlight start point in green (node 0) and goal point in red (goal node)
    start_point = RRT.get_node_point(0)  # Start node is assumed to be node 0
    goal_node = RRT.get_node_goal()
    goal_point = RRT.get_node_point(goal_node)

    ax.scatter(start_point[0], start_point[1], start_point[2], c='g', marker='o', s=100, label='Start')
    ax.scatter(goal_point[0], goal_point[1], goal_point[2], c='r', marker='o', s=100, label='Goal')

    # Visualize object (e.g., cylinder)
    if object_params and object_params.get('type') == 'cylinder':
        center = object_params.get('center', [0, 0, 0])
        radius = object_params.get('radius', 0.5)
        height = object_params.get('height', 1)
        
        # Generate cylinder points
        z = np.linspace(center[2] - height / 2, center[2] + height / 2, 100)
        theta = np.linspace(0, 2 * np.pi, 100)
        z, theta = np.meshgrid(z, theta)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)

        # Plot the cylinder using Poly3DCollection
        verts = [list(zip(x[i], y[i], z[i])) for i in range(len(z))]
        cylinder = Poly3DCollection(verts, color='gray', alpha=0.5)
        ax.add_collection3d(cylinder)

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('RRT with Objects Visualization')

    plt.legend()
    plt.show()
def visualize_rrt(RRT, interval=10):
    """
    Visualizes the RRT in 3D, highlighting the path nodes in red.
    
    Args:
        rrt (RapidlyExploringRandomTreesStarClass): The RRT* object.
        interval (int): Interval for visualizing nodes (default is 10).
    """
    # Initialize the 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    
    # Extract all nodes
    all_nodes = list(RRT.get_nodes())
    all_points = np.array([RRT.get_node_point(node) for node in all_nodes])
    start_point = RRT.get_node_point(0)
    # Plot every 'interval' node
    for i, node in enumerate(all_nodes):
        if i % interval == 0:
            point = all_points[i]
            ax.scatter(point[0], point[1], point[2], c='blue', s=5, label="Nodes" if i == 0 else "")
    
    # Highlight the nodes in the path to the goal in red
    node_check = RRT.get_node_nearest(RRT.point_goal)
    path_nodes = [node_check]
   
    while node_check:
        node_parent = RRT.get_node_parent(node_check)
        path_nodes.append(node_parent)
        node_check = node_parent
    path_nodes.reverse()
    
    path_points = np.array([RRT.get_node_point(node) for node in path_nodes])
    ax.scatter(path_points[:, 0], path_points[:, 1], path_points[:, 2], c='red', s=20, label="Path Nodes")
    ax.scatter(start_point[0], start_point[1], start_point[2], c='purple', marker='o', s=100, label='Start')
    # Plot edges
    for edge in RRT.get_edges():
        point_start = RRT.get_node_point(edge[0])
        point_end = RRT.get_node_point(edge[1])
        ax.plot([point_start[0], point_end[0]],
                [point_start[1], point_end[1]],
                [point_start[2], point_end[2]],
                c='gray', alpha=0.5)
    
    # Goal node
    goal = RRT.point_goal
    ax.scatter(goal[0], goal[1], goal[2], c='green', s=50, label="Goal")

    # Plot settings
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3D Visualization of RRT*")
    ax.legend()
    
    plt.show()


def visualize_collisions(env, q0, joint_names, robot_body_names, obj_body_names, env_body_names, body_name_clicked, times, traj_smt):
#print(traj_interp)
    #env.set_state(**state,step=True)
    L = len(times)

    # Open the gripper

    for tick in range(100):
        env.step( # dynamic update
            ctrl        = np.append(q0,[0.04, 0.04]),
            joint_names = joint_names+['panda0_finger_joint1', 'panda0_finger_joint2']
        )


    # Set collision configurations 
    

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

def find_rrt_path(RRT, point_root, point_goal, env, joint_names, robot_body_names, obj_body_names, env_body_names):
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
    
    


    print(str(point_goal) + 'point goal')
    RRT.init_rrt_star(point_root=point_root,point_goal=point_goal,seed=0)
    while True:
        # Randomly sample a point
        while True:
            if np.random.rand() <= RRT.goal_select_rate: 
                point_sample = RRT.point_goal
            else:
                point_sample = RRT.sample_point() # random sampling
            if is_point_feasible(qpos=point_sample): break

        # Get the nearest node ('node_nearest') to 'point_sample' from the tree
        node_nearest = RRT.get_node_nearest(point_sample)
        point_nearest = RRT.get_node_point(node_nearest)

        # Steering towards 'point_sample' to get 'point_new'
        point_new,cost_new = RRT.steer(node_nearest,point_sample)
        if point_new is None: continue # if the steering point is feasible

        if is_point_feasible(qpos=point_new) and \
            is_point_to_point_connectable(qpos1=point_nearest,qpos2=point_new):

            # Assign 'node_min' and initialize 'cost_min' with 'cost_new'
            node_min = node_nearest.copy()
            cost_min = cost_new

            # Select a set of nodes near 'point_new' => 'nodes_near'
            nodes_near = RRT.get_nodes_near(point_new)
        
            # For all 'node_near' find 'node_min'
            for node_near in nodes_near:
                point_near,cost_near = RRT.get_node_point_and_cost(node_near)
                if is_point_to_point_connectable(qpos1=point_near,qpos2=point_new):
                    cost_prime = cost_near + RRT.get_dist(point_near,point_new)
                    if cost_prime < cost_min:
                        cost_min = cost_near + RRT.get_dist(point_near,point_new)
                        node_min = node_near
            
            # Add 'node_new' and connect it with 'node_min'
            node_new = RRT.add_node(point=point_new,cost=cost_min,node_parent=node_min)

            # New node information for rewiring
            point_new,cost_new = RRT.get_node_point_and_cost(node_new)

            # Rewire
            for node_near in nodes_near:
                if node_near == 0: continue
                if RRT.get_node_parent(node_near) == node_new: continue
                point_near,cost_near = RRT.get_node_point_and_cost(node_near)
                cost_check = cost_new+RRT.get_dist(point_near,point_new)
                if (cost_check < cost_near) and \
                    is_point_to_point_connectable(qpos1=point_near,qpos2=point_new):
                    RRT.replace_node_parent(node=node_near,node_parent_new=node_new)

            # Re-update cost of all nodes
            if RRT.SPEED_UP: node_source = node_min
            else: node_source = 0
            RRT.update_nodes_cost(node_source=node_source,VERBOSE=False)

        # Print
        n_node = RRT.get_n_node()
        if (n_node % 1000 == 0) or (n_node == (RRT.n_node_max)):
            cost_goal = RRT.get_cost_goal() # cost to goal
            print ("n_node:[%d/%d], cost_goal:[%.5f]"%
                (n_node,RRT.n_node_max,cost_goal))
        
        # Terminate condition (if applicable)
        if n_node >= RRT.n_node_max: break # max node
        if (RRT.get_dist_to_goal() < 1e-6) and RRT.TERMINATE_WHEN_GOAL_REACHED: break
        
    print ("Done.")

def render_rrt(env, RRT, joint_names, state):
# ANIMATE RESULTS:
    # Get joint indices
# from 'start' to the point closest to the 'goal'
    node_check = RRT.get_node_nearest(RRT.point_goal)
    node_list = [node_check]
    while node_check:
        node_parent = RRT.get_node_parent(node_check)
        node_list.append(node_parent)
        node_check = node_parent
    node_list.reverse()
    print ("node_list:%s"%(node_list))

    # Get joint trajectories
    q_anchors = np.zeros((len(node_list),len(joint_names)))
    for idx,node in enumerate(node_list):
        qpos = RRT.get_node_point(node)
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
        if tick%20 == 0 or tick == (L-2):
            env.plot_text(p=np.array([0,0,1]),label='tick:[%d/%d]'%(tick,L))
            env.render()
        # Increase tick
        if tick < (L-2): tick = tick + 2

    print ("Done.")

def main():
    ARM_nJnt = 7
    xml_path = '../rrt_pick_stack/env/pick_stack.xml'
    env = MuJoCoParserClass(name='pick_stack',rel_xml_path=xml_path,verbose=False)
    print ("Done.")
    np.random.seed(seed=0)
    env.reset()


    # Initialize Franka-Panda
    joint_names = ['panda0_joint1','panda0_joint2','panda0_joint3',
                'panda0_joint4','panda0_joint5','panda0_joint6', 'panda0_joint7']

    q0 = np.array([ 0, 0, 0, -1.7359,  0,0.2277, 0])
    R0 = transformation.rpy_deg2r([180,0,90])
    env.forward(q=q0,joint_names=joint_names) # update UR5e pose

   
    obj_names = env.get_body_names(prefix='obj_')
    #n_obj = len(obj_names)

    state = env.get_state()
    xyz_click = np.array([-0.35, 1.3, 0.2])
    body_name_clicked,p_body_clicked = env.get_body_name_closest(
                xyz_click,body_names=obj_names)
    print(str(p_body_clicked) + " p_body_clicked")
    print(str(body_name_clicked))

    # Restore state and solve IKs
    
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
    robot_body_names = env.get_body_names(prefix='panda0')
    obj_body_names   = env.get_body_names(prefix='obj_')
    #env_body_names   = ['front_object_table','side_object_table']
    env_body_names = []
    # (optional) exclude 'body_name_clicked' from 'obj_body_names'
    obj_body_names.remove(body_name_clicked)
    
    visualize_collisions(env=env, q0=q0, joint_names=joint_names, robot_body_names=robot_body_names, obj_body_names=obj_body_names, env_body_names=env_body_names, body_name_clicked=body_name_clicked, times=times, traj_smt=traj_smt)
    state = env.get_state()
    joint_limits = np.array([
    [-2.8973,  2.8973],
    [-1.7628,  1.7628],
    [-2.8973,  2.8973],
    [-3.0718, -0.4   ],
    [-2.8973,  2.8973],
    [-1.6573,  2.1127],
    [-2.8973,  2.8973]
    ])
    point_root,point_goal = q0,q_grasp
    point_min = joint_limits[:, 0]  
    point_max = joint_limits[:, 1]
    #point_min = -np.pi*np.ones(len(joint_names))
    #point_max = +np.pi*np.ones(len(joint_names))
    RRT = RapidlyExploringRandomTreesStarClass(
        name      ='RRT-Star-UR',
        point_min = point_min,
        point_max = point_max,
        goal_select_rate = 0.05,
        steer_len_max    = np.deg2rad(10),
        search_radius    = np.deg2rad(15), # 10, 30, 50
        norm_ord         = 2, # 2,np.inf,
        n_node_max       = 3000,
        TERMINATE_WHEN_GOAL_REACHED = False, SPEED_UP = True,
    )

    find_rrt_path(RRT=RRT, point_root=point_root, point_goal=point_goal, env=env, joint_names=joint_names, robot_body_names=robot_body_names, obj_body_names=obj_body_names, env_body_names=env_body_names)
    render_rrt(env=env, RRT=RRT, joint_names=joint_names, state = state)
    visualize_rrt(RRT, interval=10)
    object_params = {
        'type': 'cylinder',
        'center': [-0.1, 1.35, 0.2],
        'radius': 0.065,
        'height': 0.4
    }

    # Get the path to goal and visualize the tree with the object

    visualize_rrt_with_objects(RRT=RRT, object_params=object_params, buffer_size=0.1)

#rrt.plot_tree()


if __name__ == "__main__":
    main()

