# helper function that save a dict of figures into png files in the specified directory

import os
import matplotlib.pyplot as plt

def save_figures(figures, directory):
    # check if the directory exists
    if not os.path.exists(directory):
        # if not, create the directory
        os.makedirs(directory)
    
    # figures is a dict of figures {step: figure}
    # loop through the items and key value pairs
    for step, figure in figures.items():
        # save the figure to png file
        figure.savefig(f"{directory}/{step}.png")
        # close the figure
        plt.close(figure)