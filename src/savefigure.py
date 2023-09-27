# helper function that save a dict of figures into png files in the specified directory

import os
import matplotlib.pyplot as plt
import imageio

def save_figures(figures, directory):
    # check if the directory exists
    if not os.path.exists(directory):
        # if not, create the directory
        os.makedirs(directory)
    
    # figures is a dict of figures {step: figure}
    # loop through the items and key value pairs
    for step, figure in figures.items():
        # save the figure to png file
        plt.imsave(f"{directory}/{step}.png", figure)
        plt.close()
        print(step, "saved")

def save_gif(directory, gif_name):
    # get the list of file names
    filenames = os.listdir(directory)
    # exclude the non-png files
    filenames = [filename for filename in filenames if filename.endswith('.png')]
    # sort the file names
    filenames = sorted(filenames)
    # create the list of file paths
    filenames = [directory + '/' + filename for filename in filenames]
    # create the list of images
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    # save the gif
    imageio.mimsave(gif_name, images, duration=0.2)
