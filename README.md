# parallel-face-recogniton
A parallel implementation of face recogniton based on opencv eigenfaces algorithm

####How to run

install opencv libraries and run the command below.

```
g++ mpi_eigenfaces.cpp -o faces `pkg-config --libs opencv` -lmpi -g
```

####Options
```
-DDISPLAY display result image
-DSHOW_ONLY_TIME print only Wall time to runs test suits
```

####Execute

```
mpiexec -n <ncores> ./faces <scv> <test image>
```

####Test
Test suit.
```
sh test.sh <ngroups> <ntests per core>
```

###Results
