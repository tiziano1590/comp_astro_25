mpirun -np 32 taurex -i ./parfile.par -R -C -o ./retrieval.h5 -S ./retrieval.dat --plot
taurex-plot -i ./retrieval.h5 --all -o ./