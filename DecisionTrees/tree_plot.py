import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
import copy
import numpy as np
import cross_val as cv
import tree as dt
import evaluation as ev
import config as cfg


def init_plot():
    fig, ax = plt.subplots(figsize=(20, 20))
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    ax.axis("off")
    return ax


def add_rectangle(x, y, text, ax):
    color_map = {1: "b", 2: "r", 3: "g", 4: "y"}
    if "room" in text:
        clr = color_map[int(text[-1])]
    else:
        clr = "black"
    rectangle = mpatch.Rectangle((x, y), width=20, height=0.5, color="w")
    ax.add_artist(rectangle)
    cx = x + rectangle.get_width() / 2.0
    cy = y + rectangle.get_height() / 2.0

    ax.annotate(
        text,
        (cx, cy),
        color=clr,
        weight="bold",
        fontsize=rectangle.get_width() / 2,
        ha="center",
        va="center",
    )

    if x <= ax.get_xlim()[0]:
        ax.set_xlim((x - 1, ax.get_xlim()[1]))
    if x + rectangle.get_width() >= ax.get_xlim()[1]:
        ax.set_xlim((ax.get_xlim()[0], x + rectangle.get_width() + 1))
    if y < ax.get_ylim()[0]:
        ax.set_ylim((y - 1, ax.get_ylim()[1]))
    if y + rectangle.get_height() >= ax.get_ylim()[1]:
        ax.set_ylim((ax.get_ylim()[0], y + rectangle.get_height() + 1))

    return (x, y, rectangle.get_width(), rectangle.get_height())


def connect_rectangles(Rectangle_1, Rectangle_2, LeftorRight, ax):
    x1 = Rectangle_1[0] + Rectangle_1[2] / 2
    y1 = Rectangle_1[1]

    x2 = Rectangle_2[0] + Rectangle_2[2] / 2
    y2 = Rectangle_2[1] + Rectangle_2[3]

    if LeftorRight == "left":
        text = "yes"
        p = -1
    else:
        text = "no"
        p = 1
    coordsA = "data"
    coordsB = "data"
    con = mpatch.ConnectionPatch(
        xyA=(x1, y1), xyB=(x2, y2), coordsA=coordsA, coordsB=coordsB, arrowstyle="-"
    )
    ax.add_artist(con)
    ax.annotate(
        text,
        ((x1 + x2) / 2 + p, (y1 + y2) / 2),
        color="b",
        weight="bold",
        fontsize=10,
        ha="center",
        va="center",
    )


def find_input(branch, l_r):
    if ("left" in branch) | ("right" in branch):
        res = {"sig": int(branch[l_r]["info_label"]), "dB": int(branch[l_r]["dB"])}
    else:
        res = {"label": int(branch[l_r])}
    return res



def recurs_plot(x, y, branch, alpha, beta, ax, srec=None):
    if isinstance(branch["left"], float) & isinstance(branch["right"], float):
        for side in ["left", "right"]:
            if side == "left":
                x_rec = x - abs(y) * alpha - alpha
            else:
                x_rec = x + abs(y) * alpha + alpha
            textstr = f"room {int(branch[side])}"
            srec1 = add_rectangle(x_rec, y - beta, textstr, ax)
            connect_rectangles(srec, srec1, side, ax)
        return

    textstr = f'signal {branch["wifi_signal"]}>{branch["dB"]} dB'
    if srec is None:
        srec = add_rectangle(x, y, textstr, ax)
    alpha = 0.5 * alpha
    beta = beta

    for side in ["left", "right"]:
        if side == "left":
            rec_x1 = x - abs(y) * alpha - alpha
            rec_x2 = x - abs(y) * alpha - 2 * alpha - 5
        else:
            rec_x1 = x + abs(y) * alpha + alpha
            rec_x2 = x + abs(y) * alpha + 2 * alpha + 5
        rec_y = y - beta  # same for both
        if isinstance(branch[side], float):
            textstr = "room " + str(int(branch[side]))
            srec1 = add_rectangle(rec_x1, rec_y, textstr, ax)
            connect_rectangles(srec, srec1, side, ax)
        else:
            words = find_input(branch, side)
            textstr = "signal {sig}>{dB} dB".format(**words)
            srec1 = add_rectangle(rec_x2, y - beta, textstr, ax)
            connect_rectangles(srec, srec1, side, ax)
            recurs_plot(rec_x2, rec_y, branch[side], alpha, beta, ax, srec1)

    return


def plot_tree(tree, plot_name):
    # Unpruned tree
    tr = copy.deepcopy(tree)
    ax = init_plot()
    recurs_plot(1, 5, tr, 15, 2, ax)
    plt.savefig(plot_name + ".eps", bbox_inches='tight')
    return

