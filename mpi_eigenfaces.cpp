/*
 * Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
 * Released to public domain under terms of the BSD Simplified license.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *   * Neither the name of the organization nor the names of its contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 *   See <http://www.opensource.org/licenses/bsd-license>
 */

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <mpi/mpi.h>

 using namespace cv;
 using namespace std;

 #define MAX_IMAGE 200000

 int my_rank, p;
 MPI_Comm comm;

static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {

    std::ifstream file(filename.c_str(), ifstream::in);
    if (!file) {
        string error_message = "No valid input file was given, please check the given filename.";
        CV_Error(CV_StsBadArg, error_message);
    }

    string line, path, classlabel;
    int position;
    Mat A, B;
    char outbuf[MAX_IMAGE];
    unsigned char *data;
    int rows, cols, label, i;
    while (getline(file, line)) {
        stringstream liness(line);
        getline(liness, path, separator);
        getline(liness, classlabel);
        if(!path.empty() && !classlabel.empty()) {
            position = 0;
            label = atoi(classlabel.c_str());
            if(my_rank == 0){
                A = imread(path, 0);
                B = Mat(A.rows, A.cols, 0, A.data);
                if(label%p == 0){
                    imwrite(format("c0/testemean%i.png", i++), B);
                    images.push_back(A);
                    labels.push_back(label);
                    continue;
                }
                //packing rows, cols, data

                MPI_Pack(&A.rows, 1, MPI_INT, &outbuf, MAX_IMAGE, &position, comm);
                MPI_Pack(&A.cols, 1, MPI_INT, &outbuf, MAX_IMAGE, &position, comm);
                MPI_Pack(A.data, A.rows*A.cols, MPI_CHAR, &outbuf, MAX_IMAGE, &position, comm);

                MPI_Send(&outbuf, MAX_IMAGE, MPI_PACKED, label%p, 0, comm);

            } else {
                if(label%p != my_rank) continue; //not recive

                MPI_Recv(&outbuf, MAX_IMAGE, MPI_PACKED, 0, 0, comm, MPI_STATUS_IGNORE);

                MPI_Unpack(&outbuf, MAX_IMAGE, &position, &rows, 1, MPI_INT, comm);
                MPI_Unpack(&outbuf, MAX_IMAGE, &position, &cols, 1, MPI_INT, comm);

                data = (unsigned char*)malloc(rows*cols*sizeof(unsigned char));
                MPI_Unpack(&outbuf, MAX_IMAGE, &position, data, rows*cols, MPI_CHAR, comm);
                
                A = Mat(rows, cols, 0, data);
                
                images.push_back(A);
                labels.push_back(label);
                imwrite(format("c1/testemean%i.png", i++), A);

                //free(data);
            }
        }
    }
}

int main(int argc, char *argv[]) {
    // Check for valid command line arguments, print usage
    // if no arguments were given.
    if (argc < 2) {
        cout << "usage: " << argv[0] << " <csv.ext> <input image>" << endl;
        exit(1);
    }
    
    int local_n;

    MPI_Init(&argc, &argv);

    comm = MPI_COMM_WORLD;

    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &my_rank);

    // Get the path to your CSV.
    string fn_csv = string(argv[1]);

    // These vectors hold the images and corresponding labels.
    vector<Mat> images;
    vector<int> labels;

    // Read in the data. This can fail if no valid
    // input filename is given.
    try {
        read_csv(fn_csv, images, labels);
    } catch (cv::Exception& e) {
        cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
        // nothing more we can do
        exit(1);
    }

    // Quit if there are not enough images for this demo.
    if(images.size() <= 1) {
        string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
        CV_Error(CV_StsError, error_message);
    }

    int height = images[0].rows;

    Mat testSample;
    int position, rows, cols, i, j;
    double start, finish, elapsed, local_elapsed;

    char inputbuf[MAX_IMAGE];
    unsigned char *inputdata;
    int predictedLabel;

    int predictImage;
    double local_confidence = 0.0;
    double confidence = 0.0;
    
    position = 0;
    MPI_Barrier(comm);

    start = MPI_Wtime();

    //reading desired image

    if(my_rank == 0) {

        //packing desired image
        testSample = imread(argv[2], 0);
        MPI_Pack(&testSample.rows, 1, MPI_INT, &inputbuf, MAX_IMAGE, &position, comm);
        MPI_Pack(&testSample.cols, 1, MPI_INT, &inputbuf, MAX_IMAGE, &position, comm);
        MPI_Pack(testSample.data, testSample.rows*testSample.cols, MPI_CHAR, &inputbuf, MAX_IMAGE, &position, comm);
        
        //broad cast
        MPI_Bcast(&inputbuf, MAX_IMAGE, MPI_PACKED, 0, comm );

    } else {
        MPI_Bcast(&inputbuf, MAX_IMAGE, MPI_PACKED, 0, comm );

        MPI_Unpack(&inputbuf, MAX_IMAGE, &position, &rows, 1, MPI_INT, comm);
        MPI_Unpack(&inputbuf, MAX_IMAGE, &position, &cols, 1, MPI_INT, comm);

        inputdata = (unsigned char*)malloc(rows*cols*sizeof(unsigned char));
        MPI_Unpack(&inputbuf, MAX_IMAGE, &position, inputdata, rows*cols, MPI_CHAR, comm);
        
        testSample = Mat(rows, cols, 0, inputdata);

    }

    
    Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
    model->train(images, labels);

    predictedLabel = model->predict(testSample);

    model->predict(testSample, predictedLabel, local_confidence);

    //Getting the minimum confidence value for all cores.
    MPI_Allreduce(&local_confidence, &confidence, 1, MPI_DOUBLE, MPI_MIN, comm);

    if(local_confidence == confidence && my_rank == 0){ //core 0 has found face
        predictImage = predictedLabel;
    }

    if(local_confidence == confidence && my_rank != 0) {
        MPI_Send(&predictedLabel, 1, MPI_INT, 0, 0, comm);
    }
    else if(local_confidence != confidence && my_rank == 0){
        MPI_Recv(&predictImage, 1, MPI_INT, MPI_ANY_SOURCE, 0, comm, MPI_STATUS_IGNORE);
        
    }

    finish = MPI_Wtime();

    local_elapsed = finish - start;

    MPI_Reduce(&local_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    free(inputdata);

    //display results
    if(my_rank == 0) {

        #ifndef SHOW_ONLY_TIME
        cout << "Cores " << p << endl;
        string result_message = format("Predicted class = %d  / confidence %f", predictImage, confidence);
        cout << result_message << endl << "Elapsed time " << elapsed << endl;
        #else
        cout << elapsed << endl;
        #endif

        #ifdef DISPLAY
        IplImage* n[10];
        imshow("Desired", testSample);

        IplImage* dst=cvCreateImage(cvSize(5*images[0].cols,2*images[0].rows),IPL_DEPTH_8U,3);
        for(i = 0; i < 5; i++){
            for(j = 0; j < 2; j++){
                n[(i + 4*j)] = cvLoadImage(format("./database/s%i/%i.pgm", predictImage, (i + 4*j)+1).c_str());
                cvSetImageROI(dst, cvRect(i*n[(i + 4*j)]->width, j*n[(i + 4*j)]->height,n[(i + 4*j)]->width,n[(i + 4*j)]->height) );
                cvCopy(n[(i + 4*j)],dst,NULL);
                cvResetImageROI(dst);
            }
        }

        cvShowImage( "DataBase", dst );
        
        waitKey(0);
        #endif
    }

    MPI_Finalize();

    return 0;
}
