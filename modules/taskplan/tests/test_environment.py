import matplotlib.pyplot as plt
from taskplan.environments.breakfast import Breakfast


def test_environment():
    thor_data = Breakfast()

    grid = thor_data.occupancy_grid
    init_robot_pose = thor_data.get_robot_pose()
    x, y = init_robot_pose

    # Get the whole graph from data
    whole_graph = thor_data.get_graph()

    # # Initialize the PartialMap with whole graph
    # partial_map = taskplan.core.PartialMap(whole_graph, grid, distinct=True)
    # partial_map.set_room_info(init_robot_pose, thor_data.rooms)

    plt.clf()
    plt.figure(figsize=(12, 5))
    plt.subplot(131)
    img = grid
    plt.plot(x, y, color='red', marker='.', markersize=5)

    plt.imshow(img.T)

    plt.subplot(132)
    plt.imshow(thor_data.get_top_down_frame().T)
    plt.subplot(133)
    graph = whole_graph['graph_image']
    plt.imshow(graph)
    plt.savefig('/data/test_logs/breakfast-grid-scene.png', dpi=1000)
    assert grid.size > 0
