from taurex.data.spectrum.observed import ObservedSpectrum
import numpy as np
from taurex.binning import SimpleBinner

obs = ObservedSpectrum('/Users/tiziano/Library/CloudStorage/GoogleDrive-tiziano.zingales@unipd.it/My Drive/Computational Astrophysics/notebooks/atmosphere/quickstart.dat')
#Make a logarithmic grid or a linear in the wavelength
wngrid = np.sort(10000/np.linspace(1,5,50))
bn = SimpleBinner(wngrid=wngrid)

bin_obs= bn.bindown(obs.wavenumberGrid, obs.spectrum)
errorbars = bn.bindown(obs.wavenumberGrid, obs.errorBar)


plt.figure()
plt.errorbar(1e4/wngrid, bin_obs[1], yerr=errorbars[1], label='Obs')
plt.legend()
plt.show()