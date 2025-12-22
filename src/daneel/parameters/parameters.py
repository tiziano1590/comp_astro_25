import os
import yaml
import numpy as np
from matplotlib import pyplot as plt


class Parameters:
    """
    The parameters class, crucial to read configuration files and run a complex code

    Keyword arguments:
    input file -- path to the .yaml configuration file where daneel will extract all the important parameters
    Return: a Python dictionary with all the parameters contained in the input file
    """

    def __init__(self, input_file):
        if os.path.exists(input_file) and os.path.isfile(input_file):
            with open(input_file) as in_f:
                self.params = yaml.load(in_f, Loader=yaml.FullLoader)
        else:
            raise FileNotFoundError()

        for par in list(self.params.keys()):
            if self.params[par] == "None":
                self.params[par] = None

    def get(self, param):
        return self.params[param]

# plotting multiple planets in same plot
def plot_transits(transits_list,output_file=None):
    """
    transists_list is list of TransitModel objects from trasit_model.py
    output_file : optional: if None, will use planets name(s) to create output filename
    """
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlabel("Time from central transit (days)")
    ax.set_ylabel("Relative flux")

    for transit in transits_list:
        if not hasattr(transit, 'flux'):
            transit.compute_light_curve()
        plt.plot(transit.t, transit.flux,label=f"{transit.params.name}")
 
    ax.legend()

    if output_file is None:
        if len(transits_list) == 1:
            output_file = f"{transits_list[0].params.name}_lc.png"
        else:
            names = "_".join([transit.params.name for transit in transits_list])
            output_file = f"{names}_lc.png"
            
    plt.savefig(output_file)
    plt.show()
    print(f"Light curve saved to {output_file}")
    
    fig.savefig(output_file)