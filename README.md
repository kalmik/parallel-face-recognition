# parallel-face-recognition
A parallel implementation of face recogniton based on opencv eigenfaces algorithm

####How to run

install opencv libraries and run the command below.

```
g++ mpi_eigenfaces.cpp -o faces `pkg-config --libs opencv` -lmpi -g
```

####Options
```
-DDISPLAY display result image
-DSHOW_ONLY_TIME print only Wall time to run test suites
```

####Execute

```
mpiexec -n <ncores> ./faces <scv> <test image>
```

####Test
Test suite.
```
sh test.sh <ngroups> <ntests per core>
```

###Results
![Speed UP](https://raw.githubusercontent.com/kalmik/parallel-face-recogniton/master/benchmarks/speedup.png)

![Efficiency](https://raw.githubusercontent.com/kalmik/parallel-face-recogniton/master/benchmarks/efficiency.png)
