from importlib.resources import path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import procthor
from pathlib import Path
import numpy as np

I_STEP_SIZE = 0.1


def get_interpolated_paths(robot_all_path, i_step_size=I_STEP_SIZE):
    # Build interpolated path
    distance = 0
    robot_distances = [distance]
    for pose_t, pose_dt in zip(robot_all_path[:-1], robot_all_path[1:]):
        distance += np.sqrt((pose_dt[0] - pose_t[0])**2 + (pose_dt[1] - pose_t[1])**2)
        robot_distances.append(distance)

    robot_xs = [p[0] for p in robot_all_path]
    robot_ys = [p[1] for p in robot_all_path]

    # Interpolate
    i_distance = np.arange(0, distance, i_step_size)
    i_robot_xs = np.interp(i_distance, robot_distances, robot_xs)
    i_robot_ys = np.interp(i_distance, robot_distances, robot_ys)

    return i_robot_xs, i_robot_ys, i_distance


def plot_interpolated_path(ax, robot_all_path, cmap='Blues', zorder=1):
    robot_path = [(x, y) for x, y in zip(robot_all_path[0], robot_all_path[1])]
    i_robot_xs, i_robot_ys, i_distance = get_interpolated_paths(robot_path)
    assert len(i_robot_xs) == len(i_robot_ys) == len(i_distance)
    scatter1 = ax.scatter(i_robot_xs, i_robot_ys,
                          c=i_distance / np.max(i_distance) * 0.5 + 0.5,
                          alpha=1.0,
                          s=1.0,
                          vmin=0, vmax=1,
                          cmap=cmap,
                          zorder=zorder)
    scatter1.set_rasterized(True)


def plot_grid_with_robot_trajectory_gradient(ax, grid, robot_all_poses, trajectory, graph, cmap='Blues'):
    plotting_grid = procthor.plotting.make_plotting_grid(grid.T)
    ax.imshow(plotting_grid)
    plt.plot(trajectory[0], trajectory[1], color='w', linewidth=3, zorder=1)
    plot_interpolated_path(ax, trajectory, cmap=cmap, zorder=2)
    ax.text(robot_all_poses[0].x, robot_all_poses[0].y, '0 - ROBOT', color='brown', size=4)
    for i, pose in enumerate(robot_all_poses[1:]):
        idx = graph.get_node_idx_by_position([pose.x, pose.y])
        if idx is not None:
            name = graph.get_node_name_by_idx(idx)
            ax.text(pose.x, pose.y, f'{i+1} - {name}', color='brown', size=4)
        else:
            print(f'Plotting warning: No node found in graph for pose [{pose.x:.2f}, {pose.y:.2f}]')


def plot_grid_with_robot_trajectory(ax, grid, robot_all_poses, trajectory, graph):
    plotting_grid = procthor.plotting.make_plotting_grid(grid.T)
    ax.imshow(plotting_grid)
    ax.plot(trajectory[0], trajectory[1])
    ax.text(robot_all_poses[0].x, robot_all_poses[0].y, '0 - ROBOT', color='brown', size=4)
    for i, pose in enumerate(robot_all_poses[1:]):
        idx = graph.get_node_idx_by_position([pose.x, pose.y])
        if idx is not None:
            name = graph.get_node_name_by_idx(idx)
            ax.text(pose.x, pose.y, f'{i+1} - {name}', color='brown', size=4)
        else:
            print(f'Plotting warning: No node found in graph for pose [{pose.x:.2f}, {pose.y:.2f}]')


def save_navigation_video(trajectory, thor_interface, video_file_path, fig_title):
    video_file_path = Path(video_file_path)
    video_file_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure()
    writer = animation.FFMpegWriter(12)
    writer.setup(fig, video_file_path, 500)
    for step, grid_coord in enumerate(list(zip(trajectory[0], trajectory[1]))[::5]):
        position = thor_interface.g2p_map[grid_coord]
        thor_interface.controller.step(action="Teleport", position=position, horizon=30)
        plt.clf()
        top_down_image = thor_interface.get_top_down_image(orthographic=False)
        plt.imshow(top_down_image)
        plt.axis('off')
        plt.title(f'{fig_title} [Step: {step}]', fontsize='10')
        writer.grab_frame()
    writer.finish()


def plot_plan_progression(ax, plan):
    textstr = ''
    for p in plan:
        textstr += str(p) + '\n'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # Place a text box in upper left in axes coords
    ax.text(0, 1, textstr, transform=ax.transAxes, fontsize=5,
            verticalalignment='top', bbox=props)
    ax.box(False)
    # Hide x and y ticks
    ax.xticks([])
    ax.yticks([])

    # Add labels and title
    ax.title.set_text('Plan progression')
    ax.title.set_fontsize(6)
