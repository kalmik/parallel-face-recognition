#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay( Mat frame );

/** Global variables */
String face_cascade_name = "cascades/haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;

/** @function main */
// int main( void )
// {
//     Mat frame;

//     //-- 1. Load the cascades
//     if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading face cascade\n"); return -1; };
//     //if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading eyes cascade\n"); return -1; };

//     frame = imread("crowd.jpg");

//     detectAndDisplay( frame );

//     waitKey();

//     return 0;
// }

/** @function detectAndDisplay */
void detectAndDisplay( Mat frame )
{
    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
    equalizeHist( frame_gray, frame_gray );

    //-- Detect faces
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CASCADE_SCALE_IMAGE );

    for ( size_t i = 0; i < faces.size(); i++ )
    {
        Point center( faces[i].x + faces[i].width/2, faces[i].y + faces[i].height/2 );
        rectangle( frame, faces[i], Scalar( 0, 255, 0), 2);

        Mat faceROI = frame_gray( faces[i] );

    }
    //-- Show what you got
    imshow( window_name, frame );
}
