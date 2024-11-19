import mujoco,sys
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import os

sys.path.append('../rrt_pick_stack/helper/')
sys.path.append('../rrt_pick_stack/mujoco_utils/')
from mujoco_utils.mujoco_parser import MuJoCoParserClass

from helper import slider, transformation, utility

from rrt import RapidlyExploringRandomTreesStarClass


xml_path = '../rrt_pick_stack/env/pick_stack.xml'
env = MuJoCoParserClass(name='pick_stack',rel_xml_path=xml_path,verbose=True)
print ("Done.")

#objects have prefix box instead of obj_