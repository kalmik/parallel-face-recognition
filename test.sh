#!/bin/bash

file=csv$1.ext
output=$1\groups.dat

g++ mpi_eigenfaces.cpp -o faces `pkg-config --libs opencv` -lmpi -g -DSHOW_ONLY_TIME

echo > $output

echo "input csv file $file"
echo "output dat $output"

for j in 1 2 4 8;
do
    cmd="mpiexec -n $j ./faces $file database/s1/1.pgm"
    echo "c$j = [" >> $output
    for i in `seq 1 $2`;
    do
        aux=$($cmd)
        echo $aux >> $output
    done
    echo "];" >> $output
    echo "med$j = median(c$j);" >> $output
done

