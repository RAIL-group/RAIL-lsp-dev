import numpy as np
import matplotlib.pyplot as plt
import procthor
import matplotlib.animation as animation

colors = {0: 'red', 1: 'blue', 2: 'green', 3: 'orange', 4: 'purple'}

def make_robot_video(known_grid, start, top_down_image, robot_team, video_file_name='/data/video.mp4'):
    fig = plt.figure(figsize=(10, 10), dpi=600)
    plt.tight_layout()
    writer = animation.FFMpegWriter(15)
    writer.setup(fig, video_file_name, 500)
    max_path_len = max(len(robot.all_paths[0]) for robot in robot_team)
    # ax1 = plt.subplot(121)
    # plt.imshow(top_down_image)
    # don't show x and y axis
    plt.axis('off')
    ax2 = plt.gca()
    # ax2 = plt.subplot(122)
    for i in range(max_path_len):
        ax2.clear()
        plotting_grid = procthor.plotting.make_plotting_grid(np.transpose(known_grid))
        plt.imshow(plotting_grid)
        plt.scatter(start.x, start.y, marker='o', color='r')
        plt.text(start.x, start.y, 'start', fontsize=8)
        robot_poses = [robot.all_poses for robot in robot_team]
        for j, robot in enumerate(robot_team):
            path = np.array(robot.all_paths)
            x = path[0][:i+10]
            y = path[1][:i+10]
            current_robot_pose = x[-1], y[-1]
            # if np.all(robot_poses[j][0] == current_robot_pose):
            #     print("ROBOT POSE MATCHES, REACHED")
            plt.plot(x, y, color=colors[j])
            plt.scatter(current_robot_pose[0], current_robot_pose[1], marker='o', color=colors[j])
            plt.text(current_robot_pose[0], current_robot_pose[1], f'R{j+1}', fontsize=8, color=colors[j])
        writer.grab_frame()
    writer.finish()

def plot_and_save_graph_on_grid(known_grid, known_graph, start_pose, image_filename):
    fig = plt.figure(figsize=(10, 10), dpi=600)
    procthor.plotting.plot_graph_on_grid(known_grid, known_graph)
    plt.scatter(start_pose.x, start_pose.y, marker='o', color='r')
    plt.text(start_pose.x, start_pose.y, 'start', fontsize=16)
    plt.tight_layout()
    plt.savefig(image_filename)

def plot_and_save_result(args, known_grid, known_graph, start_pose, robot_team, cost=None, image_filename=None):
    fig = plt.figure(figsize=(10, 10), dpi=300)
    procthor.plotting.plot_graph_on_grid(known_grid, known_graph)
    plt.scatter(start_pose.x, start_pose.y, marker='o', color='r')
    plt.text(start_pose.x, start_pose.y, 'start', fontsize=16)
    for i, robot in enumerate(robot_team):
        path = np.array(robot.all_paths)
        plt.plot(path[0], path[1], color=colors[i])
        x = [pose.x for pose in robot.all_poses]
        y = [pose.y for pose in robot.all_poses]
        plt.scatter(x, y, marker='x', alpha=0.5, color=colors[i])
    if cost is not None:
        title = f'Seed: {args.seed} | Planner: {args.planner} | Cost: {cost:.2f}'
    else:
        title = f'Seed: {args.seed} | Planner: {args.planner}'
    plt.title(title)
    if image_filename is None:
        image_filename = f'{args.save_dir}/mtask_eval_planner_{args.planner}_seed_{args.seed}_result.png'
    plt.savefig(image_filename)


def plot_robot_trajectory_on_grid(known_grid, known_graph, start_pose, robot_team):
    plotting_grid = procthor.plotting.make_plotting_grid(np.transpose(known_grid))
    plt.imshow(plotting_grid)
    plt.scatter(start_pose.x, start_pose.y, marker='o', color='r')
    plt.text(start_pose.x, start_pose.y, 'start', fontsize=16)
    for i, robot in enumerate(robot_team):
        path = np.array(robot.all_paths)
        plt.plot(path[0], path[1], color=colors[i])
        x = [pose.x for pose in robot.all_poses]
        y = [pose.y for pose in robot.all_poses]
        plt.scatter(x, y, marker='x', alpha=0.5, color=colors[i])
