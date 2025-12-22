## Task G: definition of the daneel.transit method

## mahdis' note: try to biuld a command line function named transit in the daneel.transit module
## that takes the path to a YAML file as input and produces the same plot as above.

#%%
import numpy as np
import batman
import matplotlib.pyplot as plt
import yaml

# yaml : used for storing data in a human readable format. 
# here our inputs (including limb darkenin table and pplanet parameters) are stored in a yaml file.
#%%
## mahdis' note: the function transit takes the path to a YAML file as input and produces the light curve plot as output.
## defined this function named transit.
def transit(param_dict):

    #csv_path = param_dict["csv_path"]
    #limb_dark_data = np.genfromtxt(csv_path, delimiter=' ', skip_header=1)
    #u1 = np.mean(limb_dark_data[:, 8])
    #u2 = np.mean(limb_dark_data[:, 10])

    # Define transit parameters
    params = batman.TransitParams()
    data = param_dict["transit_params"]

    params.t0   = data.get("t0", 0.0)       # Time of central transit (days)
    params.per  = data["per"]               # Orbital period (days)
    params.rp   = data["rp"]                # Planet radius (in units of stellar radii)
    params.a    = data["a"]                 # Semi-major axis (in units of stellar radii)
    params.inc  = data["inc"]               # Orbital inclination (degrees)
    params.ecc  = data["ecc"]               # Eccentricity
    params.w    = data["w"]                 # Longitude of periastron (degrees)
    params.limb_dark = data["limb_dark"]    # limb-darkening model
    params.u    = [data["u1"], data["u2"]]                 # Limb-darkening coefficients
    params.limb_dark = "quadratic"          # Limb-darkening model
    
    # Time array around transit
    t = np.linspace(-0.075, 0.075, 1000)    # Time from central transit (days)

    # Initialize transit model and compute light curve
    m = batman.TransitModel(params, t)
    flux = m.light_curve(params)

    # Plot light curve
    plt.plot(t, flux, label="Transit Light Curve")
    plt.xlabel("Time from central transit (days)")
    plt.ylabel("Relative flux")
    plt.legend()
    plt.title("Transit Simulation")
    plt.savefig(f"transit_plot.png")
    plt.show()