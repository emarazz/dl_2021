# the use of these external libraries are only used for plotting purposes
import sys

import numpy as np
import matplotlib.pyplot as plt

# define family colors for every plot
COLORS = [("darkred", "red", "lightcoral"),
          ("darkgreen", "green", "lightgreen"),
          ("darkblue", "blue", "cornflowerblue"),
          ("darkslateblue", "darkorchid", "mediumpurple")]

def process_file(filename):
    f = open(filename, "r")

    # extract the information contained in the first line of the file
    first_line = f.readline().split()
    number_of_epochs = int(first_line[0])
    number_of_rounds = int(first_line[1])

    # init matrix to store values
    losses = np.empty((number_of_rounds, number_of_epochs))
    for i in range(number_of_rounds):
        new_line = f.readline().split()
        assert(len(new_line) == number_of_epochs)
        losses[i] = np.array(new_line)

    f.close()

    # compute average, standard deviation, minimum and maxium
    l_avg, l_std = losses.mean(axis=0), losses.std(axis=0)
    l_min, l_max = losses.min(axis=0), losses.max(axis=0)

    return l_avg, l_std, l_min, l_max, number_of_rounds

def get_name(filename):
    # the filename without the extension stands for the name of the plot
    return filename.split('.')[0]

def plot_values(values, name, colors, need_decoration):
    # parse values
    l_avg, l_std, l_min, l_max, number_of_rounds = values

    # define x axis range according to the number of epochs starting from 1
    x_axis = np.arange(1, 1+len(l_avg), 1)

    if need_decoration:
        # set x axis limits from 1 to number of epochs
        axes = plt.gca()
        axes.set_xlim(0, 1+len(l_avg))

        # set axes intervals
        plt.xticks(x_axis)
        plt.yticks(np.arange(0, 101, 10))

        # set axes labels
        plt.xlabel("Epochs [iteration]")
        plt.ylabel("Error rate [%]")

        # add horizontal lines
        plt.hlines(np.arange(0, 101, 10), 1, len(l_avg), colors="gray", linestyles="dotted")

    # plot min/max
    plt.errorbar(x_axis, l_avg, [l_avg-l_min, l_max-l_avg], color=colors[2], elinewidth=1, capsize=3, linestyle='None')
    # plot standard deviation
    plt.errorbar(x_axis, l_avg, l_std, color=colors[0], elinewidth=3, linestyle='None')
    # plot average
    plt.plot(x_axis, l_avg, '.-', label=name, color=colors[1], lw=1)

def main(filenames):
    
    # check we have enough colors for all the plot
    if len(filenames) > len(COLORS):
        print("Error: you need to define more colors")
        return
    
    # plot with the corresponding values, name and colors
    values_dict = {}
    for i in range(len(filenames)):
        filename = filenames[i]
        values = process_file(filename)
        name = get_name(filename)
        plot_values(values, name, COLORS[i], i==0)

    # add title and legends
    plt.title("Comparison of architectures error rate during testing ({} runs)".format(values[4]))
    plt.legend()

    # save and display plot
    plt.savefig("_".join([get_name(f) for f in filenames])+".png", dpi=300, bbox_inches="tight")
    plt.show()

# call main with the files passed as argument
main(sys.argv[1:])
