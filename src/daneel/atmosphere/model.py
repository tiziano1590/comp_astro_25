#%%
## run task A with class base atmosphere

import yaml
from base import BaseAtmosphere

def run_taskA():
    #Load parameters
    with open('wasp121b_atmosphere.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    #Create atmosphere model
    atm = BaseAtmosphere(params_dict=params)
    atm.setup_environment()
    atm.setup_profile(random_abundances=True)
    atm.build_models()
    
    # Save outputs
    planet_name = atm.planet_name
    atm.save_spectrum(f"{planet_name}_assignment3_taskA_spectrum.dat")
    atm.save_parameters(f"{planet_name}_assignment3_taskA_parameters.txt")
    atm.plot_tm_spectrum(save=True, filename=f"{planet_name}_assignment3_taskA_spectrum.png")
    
    print(f"\n Task A complete for {planet_name}!")
if __name__ == "__main__":
    run_taskA()