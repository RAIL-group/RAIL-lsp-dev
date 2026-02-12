import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from sctp.param import RobotType
from scipy.stats import gaussian_kde
from matplotlib.patches import FancyArrowPatch

LINE_WIDTH = 2.5

def plot_plan_exec(graph, plt, name="Graph", gpaths=[], dpaths=[], graph_plot=None, start_coords=None, \
                   goal_coords=None, seed=None, cost=0.0, ttime=None, stime=None, verbose=False):
    """Plot graph using matplotlib."""
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    if graph_plot is not None:        
        for i, start in enumerate(start_coords):
            if i >= len(gpaths):
                break
            ax[0].scatter(start[0], start[1], marker='o', color='r')
            ax[0].text(start[0]-4.0, start[1], f'S{i}',color='blue', fontsize=8)
        for i, goal in enumerate(goal_coords):
            if i >= len(gpaths):
                break
            ax[0].scatter(goal[0], goal[1], marker='x', color='r')
            ax[0].text(goal[0]+1.5, goal[1],f'G{i}',color='r', fontsize=8)
        
        box= plot_sctpgraph(graph_plot, ax[0], verbose=verbose, initG=False)
        ax[0].set_aspect('equal', adjustable='box')
        ax[0].set_xlim(box[0][0]-1.5, box[1][0]+1.2)
        ax[0].set_ylim(box[0][1]-0.5, box[1][1]+1.0)
        ax[0].set_title(f'Seed: {seed} | Initial Graph')
    
    for i, start in enumerate(start_coords):
        if i >= len(gpaths):
            break
        ax[1].scatter(start[0], start[1], marker='o', color='r')
        ax[1].text(start[0]-4.0, start[1],f'S{i}',color='blue', fontsize=8)
    for i, goal in enumerate(goal_coords):
        if i >= len(gpaths):
            break
        ax[1].scatter(goal[0], goal[1], marker='x', color='r')
        ax[1].text(goal[0]+1.5, goal[1], f'G{i}',color='r', fontsize=8)
        
    box = plot_sctpgraph(graph, ax[1])
    if len(gpaths[0][0]) > 1:        
        colors = [['purple', 'pink'], ['yellow', 'olive'], ['cyan', 'magenta']]
        for i, path in enumerate(gpaths):
            ax[1].scatter(path[0],path[1], marker='s', s=4.5)
            plot_path_fromPoints(ax=ax[1], xy=path, colors=colors[i])

    if dpaths != [] and len(dpaths[0][0]) >1:
        colors = ['navy', 'blue', 'green']
        for i, path in enumerate(dpaths):
            ax[1].scatter(path[0], path[1], marker='P', s=4.5, alpha=1.0)            
            plot_pathArrow(points=list(zip(path[0], path[1])), ax=ax[1], color=colors[i])
    
    ax[1].set_aspect('equal', adjustable='box')
    ax[1].set_xlim(box[0][0]-1.2, box[1][0]+1.2)
    ax[1].set_ylim(box[0][1]-0.5, box[1][1]+1.0)
    if ttime is None and stime is None:
        ax[1].set_title(f'Seed: {seed} | Planner: {name} | Cost: {cost:.2f}')
    else:
        ax[1].set_title(f'S: {seed} | P: {name} | C: {cost:.2f}m | TT: {ttime:.2f}s | ST: {stime:.2f}s')
    
def plot_policy(graph, name="Policy", actions=[], uav_num=0, ugv_num=1, 
               startID=None, goalID=None, seed=None, verbose=False):
    fig, ax = plt.subplots()
    count = 0
    for node in graph.vertices+graph.pois:
        if startID is not None:
            for ii, start in enumerate(startID):
                if node.id == start:
                    count += 1
                    ax.text(node.coord[0]-1.0, node.coord[1], f"S{ii}", color='blue', fontsize=8)
        if goalID is not None:
            for ii, goal in enumerate(goalID):
                if node.id == goal:
                    count += 1
                    ax.text(node.coord[0] + 0.3, node.coord[1], f"G{ii}", color='red', fontsize=8) 

    box = plot_sctpgraph(graph, ax, verbose=verbose)
    g_cost = 0.0
    # x_drone = []
    if actions != []:
        d_colors = ['yellow', 'blue']
        g_colors = ['orange', 'green']
        g_cost, _ = plot_path_fromActions(ax, graph=graph, actions=actions, dcolors=d_colors, gcolors=g_colors,
                                          uav_num=uav_num, ugv_num=ugv_num)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(box[0][0]-1.0, box[1][0]+1.0)
    ax.set_ylim(box[0][1]-0.5, box[1][1]+0.5)

    plt.title(name+f' | seed = {seed} | cost = {0.0:.2f}')
    if uav_num == 0:
        planner = 'ctp'
    else:
        planner = name
    plt.savefig(f'/data/sctp/sctp_eval_policy_{planner}_seed_{seed}.png')
    plt.show()

def plot_firstAction(graph, action, name="First Action", 
               startID=None, goalID=None, seed=None, verbose=False):
    fig, ax = plt.subplots()
    count = 0
    for node in graph.vertices+graph.pois:
        if startID is not None:
            if node.id == startID:
                count += 1
                ax.text(node.coord[0]-1.0, node.coord[1], "Start", color='blue', fontsize=8)
        if goalID is not None:
            if node.id == goalID:
                count += 1
                ax.text(node.coord[0] + 0.3, node.coord[1], "Goal", color='red', fontsize=8) 

    box = plot_sctpgraph(graph, ax, verbose=verbose)
    g_cost = 0.0
    x_drone = []
    # if actions != []:
    d_colors = ['yellow', 'purple']
    g_colors = ['orange', 'green']
    g_cost, x_drone = plot_path_fromActions(ax, graph=graph, actions=[action], dcolors=d_colors, gcolors=g_colors)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(box[0][0]-1.0, box[1][0]+1.0)
    ax.set_ylim(box[0][1]-0.5, box[1][1]+0.5)

    plt.title(name+f' | seed = {seed} | cost = {g_cost:.2f}')
    if x_drone == []:
        planner = 'base'
    else:
        planner = 'sctp'
    plt.savefig(f'/data/sctp/sctp_eval_policy_{planner}_seed_{seed}.png')
    plt.show()

def plot_path_fromPoints(ax, xy, colors, ugv=False):
    x = xy[0]
    y = xy[1]
    points = list(zip(x, y))
    plot_arrows_withColor(ax, points, colors)


def plot_arrows_withColor(ax, points, color_pair=['orange', 'green']):
    from matplotlib.patches import FancyArrowPatch
    
    # Create colormap
    cmap = LinearSegmentedColormap.from_list('custom', color_pair)
    
    linewidth = LINE_WIDTH    
    for i in range(len(points) - 1):
        if points[i] == points[i + 1]:
            continue
        
        # Get start and end points
        start = points[i]
        end = points[i + 1]
        
        # Draw arrow with matplotlib's FancyArrowPatch
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle='-|>,head_width=0.2,head_length=0.3',
            color=cmap(i / (len(points) - 1)),
            linewidth=linewidth,
            alpha=0.95,
            mutation_scale=10
        )
        ax.add_patch(arrow)


def plot_path_fromActions(ax, graph, actions, dcolors, gcolors, uav_num=0, ugv_num=1):
    g_costs = [0.0 ]*ugv_num
    d_dists = [0.0 ]*uav_num
    rev = 0.2
    x_ugvs = [[] for _ in range(ugv_num)]
    y_ugvs = [[] for _ in range(ugv_num)]
    x_drones = [[] for _ in range(uav_num)]
    y_drones = [[] for _ in range(uav_num)]
    last_robot_action = None
    last_drone_action = None
    for a in actions:
        if a.rtype == RobotType.Ground:
            if x_ugvs[a.robotID] != []:
                g_costs[a.robotID] += np.linalg.norm(np.array([x_ugvs[a.robotID][-1],y_ugvs[a.robotID][-1]]) - np.array(a.start_pose))
            x_ugvs[a.robotID].append(a.start_pose[0])
            y_ugvs[a.robotID].append(a.start_pose[1])
            last_robot_action = a
        elif a.rtype == RobotType.Drone:
            if x_drones[a.robotID] != []:
                d_dists[a.robotID] += np.linalg.norm(np.array([x_drones[a.robotID][-1],y_drones[a.robotID][-1]]) - np.array(a.start_pose))
            x_drones[a.robotID].append(a.start_pose[0])
            y_drones[a.robotID].append(a.start_pose[1])
            last_drone_action = a
    last_vertex = [vertex for vertex in graph.vertices+graph.pois if vertex.id == last_robot_action.target][0]
    for i in range(ugv_num):
        plot_arrows_withColor(ax, [x_ugvs[i], y_ugvs[i]], color_pair=['orange', 'green'])
    if uav_num != 0:
        last_vertex = [vertex for vertex in graph.vertices+graph.pois if vertex.id == last_drone_action.target][0]
        for i in range(uav_num):
            plot_pathArrow([x_drones[i], y_drones[i]], ax, color='white')
    return g_costs, uav_num



def plot_sctpgraph(graph, plt, textsize=7, verbose=False, initG=False):
    x_max = max(enumerate(graph.vertices), key=lambda v: v[1].coord[0])[1].coord[0]
    x_min = min(enumerate(graph.vertices), key=lambda v: v[1].coord[0])[1].coord[0]
    y_max = max(enumerate(graph.vertices), key=lambda v: v[1].coord[1])[1].coord[1]
    y_min = min(enumerate(graph.vertices), key=lambda v: v[1].coord[1])[1].coord[1]
    
    plt.set_ylim(y_min-5.0, y_max+5.0)
    plt.set_xlim(x_min-5.0, x_max+5.0)
    # Plot edges
    count = 0
    for edge in graph.edges:
        x_values = [edge.v1.coord[0], edge.v2.coord[0]]
        y_values = [edge.v1.coord[1], edge.v2.coord[1]]        
        count += 1
        plt.plot(x_values, y_values, 'b-', linewidth=1.0, alpha=1.0)
        # Display block probability
        if verbose:
            mid_x = (edge.v1.coord[0] + edge.v2.coord[0]) / 2
            mid_y = (edge.v1.coord[1] + edge.v2.coord[1]) / 2
            costs = f"{edge.cost:.2f}"
            plt.text(mid_x, mid_y+0.25, costs, color='red', fontsize=textsize)

    # Plot nodes
    for node in graph.vertices:
        plt.scatter(node.coord[0], node.coord[1], color='green', s=25)
        if verbose:
            plt.text(node.coord[0], node.coord[1] + 0.2, f"V{node.id}", color='blue', fontsize=textsize)

    for poi in graph.pois:
        plt.scatter(poi.coord[0], poi.coord[1], color='red', s=25)
        if poi.block_status == 1:
            plt.scatter(poi.coord[0], poi.coord[1], color='black', s=8)
        else:
            plt.scatter(poi.coord[0], poi.coord[1], color='white', s=8)
        if verbose or initG:
            plt.text(poi.coord[0]-0.3, poi.coord[1] + 0.25, f"P{poi.id}"+f"/{poi.block_prob:.2f}", color='blue', fontsize=textsize)
        else:
            plt.text(poi.coord[0]-0.3, poi.coord[1] + 0.25, f"{poi.block_prob:.2f}", color='blue', fontsize=textsize)
    return [[x_min, y_min], [x_max, y_max]]
        

def make_scatter_plot_with_box(data_x, data_y, max_val=None, xlabel='Baseline', ylabel='SCTP'):
    if max_val is None:
        max_val = 1.1 * max(max(data_x), max(data_y))
    fig = plt.figure(figsize=(5, 5), dpi=300)
    gs = fig.add_gridspec(nrows=8, ncols=8)
    f_ax_mid = fig.add_subplot(gs[:, :])
    f_ax_bot = fig.add_subplot(gs[-1, :])
    f_ax_lft = fig.add_subplot(gs[:, 0])

    make_scatter_plot(f_ax_mid, data_x, data_y, max_val)

    f_ax_lft.boxplot(list(data_y), vert=True, showmeans=True)
    f_ax_lft.set_ylim([0, max_val])
    f_ax_lft.set_axis_off()

    f_ax_bot.boxplot(list(data_x), vert=False, showmeans=True)
    f_ax_bot.set_xlim([0, max_val])
    f_ax_bot.set_axis_off()

    f_ax_mid.set_xlabel(xlabel)
    f_ax_mid.set_ylabel(ylabel)

    baseline_cost = np.average(data_x)
    learned_cost = np.average(data_y)
    improv = (baseline_cost - learned_cost) / baseline_cost * 100

    f_ax_mid.set_title(f"{xlabel}: {baseline_cost:.1f}, {ylabel}: {learned_cost:.1f}, Improv: {improv:.1f} %")

    return f_ax_mid

def make_scatter_plot(ax, cost_x, cost_y, max_val):
    y_axis = cost_y
    x_axis = cost_x
    # Calculate the point density
    xy = np.vstack([x_axis, y_axis])
    z = gaussian_kde(xy)(xy)
    colors = plt.get_cmap("Blues")((z - z.min()) / (z.max() - z.min()) * 0.75 + 0.50)

    ax.scatter(x_axis, y_axis, c=colors)
    # Draw a center line
    ax.plot([0, max_val], [0, max_val], color='black', linestyle='--', linewidth=0.5, alpha=0.2)

    # Set X-axis and Y-axis up to same range; Using ax.axis('square') gives scaling issues for box plot
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)




from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.colors as mcolors

def get_arrowHollow(start, end,
                         width_func,
                         n_points=100,
                         edgecolor='tab:blue',
                         facecolor='white',
                         alpha=0.9,
                         edgewidth=3.0):
    """
    Draw an arrow from start to end with continuously varying width.
    
    Parameters:
        width_func: function(t) where t in [0,1] → returns width at that point
        n_points:   resolution along the path (more = smoother)
    """
    start = np.array(start, dtype=float)
    end   = np.array(end, dtype=float)
    vec = end - start
    length = np.linalg.norm(vec)
    if length == 0:
        return None
    
    # Parameter t from 0 to 1 along the segment
    t = np.linspace(0, 1, n_points)
    points = start + np.outer(t, vec)          # centerline points
    widths = np.array([width_func(ti) for ti in t])
    # factor = 1.0 # testing graph
    factor = 9.0 # bridges graph
    
    # Unit tangent and perpendicular
    tangents = factor*vec / length
    
    perp = np.array([-tangents[1], tangents[0]])

    # Left and right boundaries
    left  = points + perp * widths[:, np.newaxis] / 2
    right = points - perp * widths[:, np.newaxis] / 2
    

    # Build closed path: forward on left → tip → back on right → close
    verts = np.concatenate([
        left,                   # go forward along left side
        [end],                  # sharp tip
        right[::-1],            # go back along right side
    ])

    codes = [Path.MOVETO] + [Path.LINETO]*(len(left)-1) + \
            [Path.LINETO, Path.LINETO] + \
            [Path.LINETO]*(len(right)-2) + \
            [Path.CLOSEPOLY]

    path = Path(verts, codes)
    patch = PathPatch(path,
                      facecolor=facecolor,
                      edgecolor=edgecolor,
                      linewidth=edgewidth,
                      capstyle='round',
                      joinstyle='round',
                      alpha=alpha,
                      zorder=2)
    return patch
  

def plot_pathArrow(points, ax, color='white'):
    colors = ['navy', 'blue', 'cyan', 'lime', 'green']
    if color == 'navy':
        colors = ['navy', 'cyan']
    elif color == 'blue':
        colors = ['blue', 'lime']
    elif color == 'green':
        colors = ['green', 'yellow']
    col = mcolors.LinearSegmentedColormap.from_list(color, colors)
    cmap = col
    
    linewidth = LINE_WIDTH
    
    for i in range(len(points) - 1):
        if points[i] == points[i + 1]:
            continue
        
        # Get start and end points
        start = points[i]
        end = points[i + 1]
        
        # Calculate arrow properties
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        # Draw arrow with matplotlib's FancyArrowPatch
        arrow = FancyArrowPatch(
            start,
            end,
            arrowstyle='->,head_width=0.2,head_length=0.3',
            color=cmap(i / (len(points) - 1)),
            linewidth=linewidth,
            alpha=0.95,
            mutation_scale=10
        )
        ax.add_patch(arrow)
