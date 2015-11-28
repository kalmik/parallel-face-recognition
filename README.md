# parallel-face-recogniton
A parallel implementation of face recogniton based on opencv eigenfaces algorithm

how to run

install opencv libraries and run the command below.

```
g++ `pkg-config --cflags --libs opencv` mpi_eigenfaces.cpp -o faces `pkg-config --libs opencv` -lmpi -g
```

execute

```
mpiexec -n 4 ./faces csv.ext out database/s4/2.pgm
```
