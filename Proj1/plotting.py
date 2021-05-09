# the use of these external libraries are only used for plotting purposes
import sys

import numpy as np
import matplotlib.pyplot as plt

# define family colors for every plot
COLORS = [("black", "grey", "lightgrey"),
          ("darkred", "red", "lightcoral"),
          ("darkblue", "blue", "cornflowerblue")]

def compute_averages(matrix):
    return (matrix.mean(axis=0), matrix.std(axis=0), matrix.min(axis=0), matrix.max(axis=0))

def process_scores(filename):
    f = open(filename, "r")

    # extract the information contained in the first line of the file
    first_line = f.readline().split()
    number_of_epochs = int(first_line[0])
    number_of_rounds = int(first_line[1])

    # init matrices to store values
    losses = np.empty((number_of_rounds, number_of_epochs))
    train = np.empty((number_of_rounds, number_of_epochs))
    test = np.empty((number_of_rounds, number_of_epochs))
    for i in range(number_of_rounds):
        new_line = f.readline().split()
        assert(len(new_line) == number_of_epochs)
        losses[i] = np.array(new_line)

        new_line = f.readline().split()
        assert(len(new_line) == number_of_epochs)
        train[i] = np.array(new_line)

        new_line = f.readline().split()
        assert(len(new_line) == number_of_epochs)
        test[i] = np.array(new_line)

    f.close()

    losses /= np.max(losses) 
    losses *= 100

    train *= 100
    test *= 100

    return number_of_rounds, number_of_epochs, compute_averages(losses), compute_averages(train), compute_averages(test)

def plot_values(scores):
    # parse values
    _, number_of_epochs, losses, train, test = scores

    l_avg, l_std, l_min, l_max = losses
    tr_avg, tr_std, tr_min, tr_max = train
    te_avg, te_std, te_min, te_max = test

    # define x axis range according to the number of epochs starting from 1
    x_axis = np.arange(1, 1+number_of_epochs, 1)

    # set x axis limits from 1 to number of epochs
    axes = plt.gca()
    axes.set_xlim(0, 1+number_of_epochs)

    # set axes intervals
    plt.xticks(x_axis)
    plt.yticks(np.arange(0, 66, 5))

    # set axes labels
    plt.xlabel("Epochs [iteration]")
    plt.ylabel("Error rate and normalized loss [%]")

    # add horizontal lines
    plt.hlines(np.arange(0, 66, 5), 1, number_of_epochs, colors="gray", linestyles="dotted")

    # plot losses, train and test scores

    plt.plot(x_axis, l_avg, '.-', label="Loss", color="black", lw=1)
    
    plt.plot(x_axis, tr_avg, '.-', label="Train error", color="red", lw=1)

    # plot min/max
    plt.errorbar(x_axis, te_avg, [te_avg-te_min, te_max-te_avg], color="cornflowerblue", elinewidth=1, capsize=3, linestyle='None')
    # plot standard deviation
    plt.errorbar(x_axis, te_avg, te_std, color="darkblue", elinewidth=3, linestyle='None')
    # plot average
    plt.plot(x_axis, te_avg, '.-', label="Test error", color="blue", lw=1)

def main(nameParametersScores):
    
    # check we have enough colors for all the plot
    if len(nameParametersScores) != 3:
        print("You only need the name and two filenames, one for parameters and one for scores")
        return

    name = nameParametersScores[0]
    parameters = nameParametersScores[1]
    scores = nameParametersScores[2]
    
    # plot with the corresponding values, name and colors
    scores = process_scores(scores)
    plot_values(scores)

    # add title and legends
    plt.title("Performance of {} architecture averaged over {} runs".format(name, scores[0]))
    plt.legend()

    # save and display plot
    plt.savefig(name+".png", dpi=300, bbox_inches="tight")
    plt.show()

# call main with the files passed as argument
main(sys.argv[1:])
