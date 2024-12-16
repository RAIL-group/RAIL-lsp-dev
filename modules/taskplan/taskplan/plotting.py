import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


def gjs_scatter_plot(ax, cost_x, cost_y, max_val, fail_val):
    common_seeds = set.intersection(set(cost_x.keys()), set(cost_y.keys()))
    y_axis = [cost_y[seed] for seed in common_seeds]
    x_axis = [cost_x[seed] for seed in common_seeds]
    # Calculate the point density
    xy = np.vstack([x_axis, y_axis])
    z = gaussian_kde(xy)(xy)
    colors = plt.get_cmap("Blues")((z - z.min()) / (z.max() - z.min()) * 0.75 + 0.50)

    ax.scatter(x_axis, y_axis, c=colors)

    # x axis and y axis should be the same
    # max_val = max(max(x_axis), max(y_axis))
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    # draw a center line
    ax.plot([0, max_val], [0, max_val], color='black', linestyle='--', linewidth=0.5, alpha=0.2)
    # Change tick text size for both x and y axes
    ax.tick_params(axis='x', labelsize=8)  # Set x-axis tick size
    ax.tick_params(axis='y', labelsize=8)  # Set y-axis tick size

    # Plot the failed seeds
    y_fail_seed = set.difference(set(cost_x.keys()), set(cost_y.keys()))
    y_fail_cost = [cost_x[seed] for seed in y_fail_seed]
    x_fail_fill_cost = [fail_val for _ in y_fail_seed]
    if y_fail_cost:
        print(y_fail_cost)
        ax.plot([fail_val, fail_val], [min(y_fail_cost) - 30, max(y_fail_cost) + 30], color='black', linestyle='--',
                linewidth=0.5, alpha=0.2)
        ax.scatter(x_fail_fill_cost, y_fail_cost, color='red', marker='x')

    x_fail_seed = set.difference(set(cost_y.keys()), set(cost_x.keys()))
    x_fail_cost = [cost_y[seed] for seed in x_fail_seed]
    y_fail_fill_cost = [fail_val for _ in x_fail_seed]
    if x_fail_cost:
        print(x_fail_cost)
        ax.plot([min(x_fail_cost) - 30, max(x_fail_cost) + 30], [fail_val, fail_val], color='black', linestyle='--',
                linewidth=0.5, alpha=0.2)
        ax.scatter(x_fail_cost, y_fail_fill_cost, color='red', marker='x')


def make_scatter_with_box(data_x, data_y, max_val=None):
    if max_val is None:
        max_val = 1.1 * max(max(data_x.values()), max(data_y.values()))
    fail_val = max_val - 20
    fig8 = plt.figure(constrained_layout=False, figsize=(4, 4))
    gs1 = fig8.add_gridspec(nrows=8, ncols=8, left=0.1, right=0.9, wspace=0.00, hspace=0)
    f8_ax_mid = fig8.add_subplot(gs1[:, :])
    f8_ax_bot = fig8.add_subplot(gs1[-1, :])
    f8_ax_lft = fig8.add_subplot(gs1[:, 0])

    gjs_scatter_plot(f8_ax_mid, data_x, data_y, max_val, fail_val)
    # f8_ax_mid.set_xlabel(bot_name)
    # f8_ax_mid.set_ylabel(lft_name)
    # f8_ax_mid.set_title(title)

    f8_ax_lft.boxplot(list(data_y.values()), vert=True, showmeans=True)
    f8_ax_lft.set_ylim([0, max_val])
    f8_ax_lft.set_axis_off()

    # Bottom
    f8_ax_bot.boxplot(list(data_x.values()), vert=False, showmeans=True)
    f8_ax_bot.set_xlim([0, max_val])
    f8_ax_bot.set_axis_off()

    return f8_ax_mid


def plot_plan(plan):
    textstr = ''
    for p in plan:
        textstr += str(p) + '\n'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    # Place a text box in upper left in axes coords
    plt.text(0, 1, textstr, transform=plt.gca().transAxes, fontsize=5,
             verticalalignment='top', bbox=props)
    plt.box(False)
    # Hide x and y ticks
    plt.xticks([])
    plt.yticks([])

    # Add labels and title
    plt.title('Plan progression', fontsize=6)
