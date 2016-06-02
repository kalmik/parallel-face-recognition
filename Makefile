all: main

mpi: mpi_eigenfaces.cpp
		g++ mpi_eigenfaces.cpp -o faces `pkg-config --libs opencv` -lmpi -g -DDISPLAY

crowd: detectface.cpp
		g++ detectface.cpp -o recog `pkg-config --libs opencv` -g

main: pdi_3.cpp
		g++ pdi_3.cpp -o pdi `pkg-config --libs opencv` -lmpi -g