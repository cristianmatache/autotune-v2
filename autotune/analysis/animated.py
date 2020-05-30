from benchmarks.opt_function_simulation_problem import OptFunctionSimulationProblem
from core import ShapeFamily, RoundRobinShapeFamilyScheduler

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


MAX_EPOCH = 20

fig, ax = plt.subplots()
fig.set_tight_layout(True)

# Query the figure's on-screen size and DPI. Note that when saving the figure to
# a file, we need to provide a DPI for that separately.
print('fig size: {0} DPI, size in inches {1}'.format(
    fig.get_dpi(), fig.get_size_inches()))

# Plot a scatter that persists (isn't redrawn) and the initial line.
# x = np.arange(0, 20, 0.1)
# ax.scatter(x, x + np.random.normal(0, 3.0, len(x)))
# line, = ax.plot(x, x - 5, 'r-', linewidth=2)

families_of_shapes_general = (
    # ShapeFamily(None, 1.5, 10, 15, False),  # with aggressive start
    # ShapeFamily(None, 0.72, 5, 0.15),
    ShapeFamily(None, 4.5, 3, 15),
)

ax.set_ylim([-200, 100])
ax.set_xlim([0, MAX_EPOCH])

problem = OptFunctionSimulationProblem('wave')
scheduler = RoundRobinShapeFamilyScheduler(shape_families=families_of_shapes_general,
                                           max_resources=MAX_EPOCH, init_noise=5)
# Update the line and the axes (with a new xlabel). Return a tuple of
# "artists" that have to be redrawn for this frame.
evaluator = problem.get_evaluator(*scheduler.get_family(), should_plot=True)


def update(i):
    label = 'timestep {0}'.format(i)
    print(label)
    evaluator.evaluate(i % MAX_EPOCH)
    # line.set_ydata(x - 5 + i)
    # ax.set_xlabel(label)
    return ax


if __name__ == '__main__':
    save = True
    # FuncAnimation will call the 'update' function for each frame; here
    # animating over 10 frames, with an interval of 200ms between frames.
    anim = FuncAnimation(fig, update, frames=np.arange(0, 2 * MAX_EPOCH), interval=0.1)
    if save:
        anim.save('line.gif', dpi=80, writer='imagemagick')
    else:
        # plt.show() will just loop the animation forever.
        plt.show()


# if __name__ == "__main__":
#     # function colors: 1 blue 2 green 3 orange 4 red 5 purple 6 brown 7 pink 8 grey
#     opt_func_name = 'wave'
#     problem = OptFunctionSimulationProblem(opt_func_name)
#     families_of_shapes_egg = (
#         ShapeFamily(None, 1.5, 10, 15, False, 0, 200),  # with aggressive start
#         ShapeFamily(None, 0.5, 7, 10, False, 0, 200),  # with average aggressiveness at start and at the beginning
#         ShapeFamily(None, 0.2, 4, 7, True, 0, 200),  # non aggressive start, aggressive end
#     )
#     families_of_shapes_general = (
#         ShapeFamily(None, 1.5, 10, 15, False),  # with aggressive start
#         ShapeFamily(None, 0.5, 7, 10, False),  # with average aggressiveness at start and at the beginning
#         ShapeFamily(None, 0.2, 4, 7, True),  # non aggressive start, aggressive end
#     )
#     families_of_shapes = {
#         'egg': families_of_shapes_egg,
#         'wave': families_of_shapes_general,
#     }.get(opt_func_name, families_of_shapes_general)
#     problem.plot_surface(n_simulations=1000, max_resources=81, n_resources=81,
#                          shape_families=families_of_shapes,
#                          init_noise=0.1)
