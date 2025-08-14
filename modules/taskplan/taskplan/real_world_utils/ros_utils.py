import rospy
from nav_msgs.srv import GetPlan
from geometry_msgs.msg import PoseStamped
import tf
import numpy as np
from nav_msgs.msg import OccupancyGrid
from robotics_utils.ros.transform_manager import TransformManager
from spot_skills.srv import NameService


def create_pose_stamped(location):
    pose = PoseStamped()
    pose.header.frame_id = "map"
    pose.pose.position.x = float(location[0])
    pose.pose.position.y = float(location[1])

    quaternion = tf.transformations.quaternion_from_euler(0, 0, location[2])
    pose.pose.orientation.x = quaternion[0]
    pose.pose.orientation.y = quaternion[1]
    pose.pose.orientation.z = quaternion[2]
    pose.pose.orientation.w = quaternion[3]
    return pose


def compute_path(start, goal):
    rospy.wait_for_service('/move_base/make_plan')

    start_pose = create_pose_stamped(start)
    goal_pose = create_pose_stamped(goal)

    try:
        make_plan = rospy.ServiceProxy('/move_base/make_plan', GetPlan)
        response = make_plan(start=start_pose, goal=goal_pose)
        return response.plan
    except rospy.ServiceException as e:
        rospy.logerr("Failed to call /make_plan service: %s", e)
        return None
    

def get_occupancy_grid():
    print("Waiting to get occupancy grid...")
    try:
        msg = rospy.wait_for_message("/rtabmap/grid_map", OccupancyGrid)
    except rospy.ROSException as e:
        rospy.logerr(f"Timeout while waiting for /rtabmap/grid_map: {e}")
        return None
    width = msg.info.width
    height = msg.info.height
    data = np.array(msg.data).reshape((height, width))
    occupancy_grid = np.where(data == 100, 1, 0)

    return occupancy_grid


def get_robot_pose():
    print("Waiting to get robot pose...")
    pose = TransformManager.lookup_transform('body', 'map', when=rospy.Time.now() + rospy.Duration(3))
    return (pose.position.x, pose.position.y, pose.yaw_rad)


def move_spot(container_name):
    rospy.wait_for_service('/spot/navigation/to_waypoint')
    try:
        move_to_waypoint = rospy.ServiceProxy('/spot/navigation/to_waypoint', NameService)
        response = move_to_waypoint(name=container_name)
        return response.success
    except rospy.ServiceException as e:
        rospy.logerr("Failed to call /spot/navigation/to_waypoint: %s", e)
        return False
