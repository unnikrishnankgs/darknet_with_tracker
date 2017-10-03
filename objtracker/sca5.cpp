 /** =========================================================================== 
 * @file sca5.c
 * 
 * @brief   The 5 step scene change analysis algorithm
 * 
 * ============================================================================
 *
 * Copyright © Lab 276 (Professor Kaikai Liu), 2017-18
 * 
 * ============================================================================
 *
 * @author : Unnikrishnan
 *
 * ============================================================================
 */

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cv.h>
#include <highgui.h>
using namespace std;
#include <iostream>
#include "opencv2/opencv.hpp"
#include <opencv2/flann.hpp>
#include "sca5.h"
//#include <windows.h>
using namespace std;
using namespace cv;

#define printf(...)

RNG rng(12345);

#define STEP_234
#define STEP_5
#define DRAW_GRIDS

#define SKIP_COUNT 10 
//#define STEP_5_PRINT_MATRIX_MATH

#ifndef STEP_234
#if SKIP_COUNT == 0
#define DUMP_OP_PATH "/step1/continuous/"
#elif SKIP_COUNT == 10
#define DUMP_OP_PATH "/step1/skip10/"
#elif SKIP_COUNT == 20
#define DUMP_OP_PATH "/step1/skip20/"
#elif SKIP_COUNT == 29
#define DUMP_OP_PATH "/step1/skip29/"
#endif
#endif

#if defined(STEP_234) && !defined(STEP_5) 
#if SKIP_COUNT == 0
#define DUMP_OP_PATH "/step234/continuous/"
#elif SKIP_COUNT == 10
#define DUMP_OP_PATH "/step234/skip10/"
#elif SKIP_COUNT == 20
#define DUMP_OP_PATH "/step234/skip20/"
#elif SKIP_COUNT == 29
#define DUMP_OP_PATH "/step234/skip29/"
#endif
#endif

#if defined(STEP_5) 
#if SKIP_COUNT == 0
#define DUMP_OP_PATH "/step5/continuous/"
#elif SKIP_COUNT == 10
#define DUMP_OP_PATH "/step5/skip10/"
#elif SKIP_COUNT == 20
#define DUMP_OP_PATH "/step5/skip20/"
#elif SKIP_COUNT == 29
#define DUMP_OP_PATH "/step5/skip29/"
#endif
#endif


//#define DUMP_OP_IMAGE

bool findIfThereIsARectInVicinity(unsigned char const * pData, int i, int j, Mat& b, const int maxEmptyPixels, int& startI, int& startJ, 
    int& endI, int& endJ, Scalar& color, bool& iDone, bool& jDone)
{
    bool ret = false;
    int cR = 0; /**< count of empty pixels to right */
    int cD = 0;
    int i1, j1;

    iDone = jDone = false;

    if(!pData || i > b.rows || j > b.rows)
        return false;

    startI = startJ = 0;
    endI = i1 = i;
    endJ = j1 = j;
    /** right-> */
    cR = 0;
    while(cR <= maxEmptyPixels && j1 < b.cols)
    {
        i1 = i;
        unsigned char b1 = pData[i1 * b.step + j1];
        unsigned char g1 = pData[i1 * b.step + j1 + 1];
        unsigned char r1 = pData[i1 * b.step + j1 + 2];
        printf("R bgr={%d,%d,%d}\n", b1, g1, r1);
        Scalar color1 = Scalar(b1, g1, r1);

        if(color != color1)
        {
            cR++;
        }
        else
        {
            if(startJ == 0)
            {
                startJ = j1;
                startI = i1;
            }
            else
            {
                endJ = j1; 
            }
            /** down */
            cD = 0;
            while(cD <= maxEmptyPixels && i1 < b.rows)
            {
                unsigned char b1 = pData[i1 * b.step + j1];
                unsigned char g1 = pData[i1 * b.step + j1 + 1];
                unsigned char r1 = pData[i1 * b.step + j1 + 2];
                printf("D bgr={%d,%d,%d}\n", b1, g1, r1);
                Scalar color1 = Scalar(b1, g1, r1);
        
                if(color != color1)
                    cD++;
                else
                {
                    if(startJ == 0)
                    {
                        startJ = j1;
                        startI = i1;
                    }
                    else
                    {
                        endI = i1; 
                    }
                }
                i1++;
            }
        
        }
        j1+=3;
    }

    if(j1 == b.cols)
        jDone = true;

    
    if(endJ != j && endI != i)
        return true;

    return false;

}

Point centerOfRect(Rect& a)
{
    int x = a.x + a.width / 2;
    int y = a.y + a.height / 2;
    return Point(x, y);
}

#if 0
int main(int argc, char* argv[])
{
    Mat nave, nave1, src, frame, b;
    int first = 1;
    //String path="/Users/gotham/personal/vid_samples/PrincessAnneEB\@Lynnhaven-2014-06-04-07h10m20s-udp___239_211_20_101_3000.ts";
    String path;
    if(argc > 1)
        path=String(argv[1]);
    else
        path="/Users/gotham/work/aic/PPT/1080p_WALSH_ST_036_480p.mp4";
    VideoCapture cap(path);
    //if(cap.empty())
    {
    //cout<<"error in frame get";
    }
    namedWindow("cam_feed");
    namedWindow("sca5_sjsu");
    int c = 0;
    /** Abs diff between two frames - the level of motion per pixel
     * We use Canny edge detector on the difference frame to visualize per-object movement between two frames
     * We use border following technique discussed in \cite{suzuki85} to obtain the contours of detected edges
     * Then, we represent the moving objects by collecting the contours using Douglas-Peucker approximation algorithm
     * as a rectangle
     * Such rectangles are a direct representation of motion information
     * We are trying to deduce a quantitative motion density in the 2-D pixel space using rectangels.
     * This helps us in effectively understanding which-objects moved how-much.
      
     * */
    int img_pairs_processed = 0;
    for(;;) {
        char file_name[1024];
        cap >> src;
        for(int i = 0; i < SKIP_COUNT; i++)
        {
            Mat skip;
            printf("skipping\n");
            cap >> skip;
        }
        cv::Rect roi(0, 0, cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
        nave=src(roi);
        //sleep(100);
        cap>>frame;
        cv::Rect roi1(0, 0, cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT));
        //cv::Rect roi1(222, 120, 250, 323);
        nave1=frame(roi1);
        int c=cap.get(CV_CAP_PROP_FPS);
        cout<<c;
        absdiff(nave,nave1,b);
        //    if(first)
        //        add(nave,nave1,b);
        //    else
        //        add(b,res,b);
        
        //cout <<absdiff<<"............";
        cout<<"B ...",b;
        #ifdef STEP_234
        /** step 2, 3, 4 */
        int na;
        unsigned char* pData = b.data;
        printf("type=%d %d\n", b.type(), CV_8UC3);
        Mat canny_output;
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        int thresh = 100;
        int max_thresh = 255;
        
        Canny(b, canny_output, thresh, thresh * 2, 3);

        findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        vector<vector<Point> > contours_poly( contours.size() );
        vector<Rect> boundRect( contours.size() );

        for( int i = 0; i< contours.size(); i++ )
        {
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
            boundRect[i] = boundingRect( Mat(contours_poly[i]) );
            rectangle(b, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
        }
        #endif

        #ifdef STEP_5
        int nH = b.size().height;
        int nW = b.size().width;
        int nStepSize = nW / 100;
        int nWG = nW / nStepSize + 1;
        int nHG = nH / nStepSize + 1;
        printf("nStepSize=%d %dX%d nWG=%d nHG=%d\n", nStepSize, nW, nH, nWG, nHG);
        /** logical grids */
        /** We have nW/nStepSize X nH/nStepSize grids;
         * */
        int** ppGridM = (int**)calloc(1, nHG * sizeof(int*));
        for(int i = 0; i < nHG; i++)
            ppGridM[i] = (int*)calloc(1, nWG * sizeof(int));

        for(int i = 0; i < boundRect.size(); i++)
        {
            Point p[4];
            p[0] = Point(boundRect[i].x, boundRect[i].y); //TL
            p[1] = Point(boundRect[i].br().x, boundRect[i].br().y); //BR
            p[2] = Point(boundRect[i].x + boundRect[i].width, boundRect[i].y); //TR
            p[3] = Point(boundRect[i].x, boundRect[i].y + boundRect[i].height); //BR

            printf("(%d, %d) (%d, %d) (%d, %d) (%d, %d)\n", 
                boundRect[i].x, boundRect[i].y,
                boundRect[i].br().x, boundRect[i].br().y,
                boundRect[i].x + boundRect[i].width, boundRect[i].y,
                boundRect[i].x, boundRect[i].y + boundRect[i].height
                );
            for(int j = 0; j < 4; j++)
            {
                printf("[%d, %d]\n", p[j].x / nStepSize, p[j].y / nStepSize);
                /** grid's row is calculated from its vertical coord
                 * column num is calculated from its horiz coord */
                ppGridM[(p[j].y / nStepSize)][(p[j].x / nStepSize)]++;
            }
            /** also from TL -> TR, for every point, populate corresponding grid 
             * and grids vertically below till bottom
             * */
            for(int x1 = ((p[0].x / nStepSize) * nStepSize) + nStepSize; x1 < p[2].x; x1 += nStepSize) //shall not include TL and TR points itself
            {
                printf("sub x\n");
                ppGridM[(p[0].y / nStepSize)][(x1 / nStepSize)]++;
                /** vertically down till bottom */
                for(int y1 = ((p[0].y / nStepSize) * nStepSize) + nStepSize; y1 < p[3].y; y1 += nStepSize)
                {
                    printf("sub y\n");
                    ppGridM[y1/nStepSize][x1/nStepSize]++;
                }
            }
        }

        printf("draw grids\n");

        #ifdef DRAW_GRIDS
        /** draw grids */
        /** verticals */
        for(int i = 0; i < nW; i += nStepSize)
            line(b, Point(i, 0), Point(i, nH), Scalar(0, 255, 255));
        /** horizontals */
        for(int i = 0; i < nH; i += nStepSize)
            line(b, Point(0, i), Point(nW, i), Scalar(0, 255, 255));
        #endif

        /** draw grid cross if theres movement in grid */
        for(int i = 0; i < nHG; i++)
        {
            for(int j = 0; j < nWG; j++)
            {
                if(ppGridM[i][j])
                {
                    //line(b, Point(j*nStepSize, i*nStepSize), Point(j*nStepSize + nStepSize, i*nStepSize + nStepSize), Scalar(0, 255, 255));
                    char text[10];
                    snprintf(text, 10, "%2d", ppGridM[i][j]);
                    //circle(b, Point(j*nStepSize, i*nStepSize), 3, Scalar(0,255,0), -1, 8);
                    putText(b, text, Point(j*nStepSize, i*nStepSize + nStepSize/2), FONT_HERSHEY_PLAIN, 0.5, Scalar(255,255,255));  
                }
            }
        }
        #endif

        #if 0
        vector<Rect> clusteredRects;
        vector<int> labels;
        int euc_dist = 50;
        int n_labels = partition(boundRect, labels, [euc_dist](const Rect& lhs, const Rect& rhs){
                int euc_dist2 = euc_dist * euc_dist;
                Point a = centerOfRect(lhs);
                Point b = centerOfRect(rhs);
                if((b.x - a.x)*(b.x - a.x) + (b.y - a.y)*(b.y - a.y) < euc_dist2)
                    return true;
                else
                    return false;
            });
        int idxToRem = -1;
        for(int i = 0; i < boundRect.size(); i++)
        {
            /**  */
        }
        #endif

        #if 1
        /** cluster the rectangles:
         * 1) divide the Mat into small grids;
         * say 3 X 3 
         * 2) */

        
        
        #if 0
        vector<Rect> clusteredRects;
        for(int i = 0; i < b.rows; i++)
        {
            int endIMax = i+1;
            for(int j = 0; j < b.cols; j+=3)
            {
                unsigned char b1 = pData[i * b.step + j];
                unsigned char g1 = pData[i * b.step + j + 1];
                unsigned char r1 = pData[i * b.step + j + 2];
                /** collect consecutive grids with our color */
                {
                    Point tl, tr, bl, br;
                    int startI = 0, startJ=0, endI=0, endJ=0;
                    bool iDone = false, jDone = false;
                    printf("going in for j = %d\n", j);
                    while(findIfThereIsARectInVicinity(pData, i, j, b, MAX_EMPTY_GRIDS * GRID_WIDTH, startI, startJ, endI, endJ, color, iDone, jDone))
                    {
                        /** add rect! into list */
                        //clusteredRects.push_back(Rect(startJ, startI, (endJ - startJ), endI - startI));
                        //clusteredRects.push_back(Rect(startI, startJ/3, (endJ - startJ)/3, (endI - startI)));
                        clusteredRects.push_back(Rect(startJ/3, startI, (endI - startI), (endJ - startJ)/3));
                        j = endJ;
                        if(endI > endIMax || endIMax == 0)
                            endIMax = endI;
                    }
                    if(jDone)
                        j = b.cols;
                }
            }
            i = endIMax;
        }

        color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        for( int i = 0; i< clusteredRects.size(); i++ )
        {
            rectangle(b, clusteredRects[i].tl(), clusteredRects[i].br(), color, 2, 8, 0);
        }
        #endif
        #endif
        imshow("cam_feed", src);
        imshow("sca5_sjsu", b);
        if(cv::waitKey(30) >= 0) break;

        img_pairs_processed++;
        #ifdef DUMP_OP_IMAGE
        vector<int> compression_params;
        compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);
        snprintf(file_name, 1024, "mkdir -p output/" DUMP_OP_PATH);
        system(file_name);
        snprintf(file_name, 1024, "output/" DUMP_OP_PATH "output-%04d.png", img_pairs_processed);
        bool ret = imwrite(file_name, b, compression_params);
        printf("ret=%d\n", ret);
        if(img_pairs_processed == 2)
            break;
        #endif

        #ifdef STEP_5_PRINT_MATRIX_MATH
        printf("sheet:\n");
        for(int i = 0; i < nHG; i++)
        {
            for(int j = 0; j < nWG; j++)
            {
                printf("%02d ", ppGridM[i][j]);
            }
            printf("\n");
        }
        #endif


    }
   return 0;
}
#endif

int sca5(Mat src, Mat frame, tGrid* pGrid)
{
    Mat nave = src;
    Mat nave1 = frame;
    Mat b;
    int first = 1;
    namedWindow("cam_feed");
    namedWindow("sca5_sjsu");
    /** Abs diff between two frames - the level of motion per pixel
     * We use Canny edge detector on the difference frame to visualize per-object movement between two frames
     * We use border following technique discussed in \cite{suzuki85} to obtain the contours of detected edges
     * Then, we represent the moving objects by collecting the contours using Douglas-Peucker approximation algorithm
     * as a rectangle
     * Such rectangles are a direct representation of motion information
     * We are trying to deduce a quantitative motion density in the 2-D pixel space using rectangels.
     * This helps us in effectively understanding which-objects moved how-much.
      
     * */
    int img_pairs_processed = 0;
    {
        char file_name[1024];
        absdiff(nave,nave1,b);
        //    if(first)
        //        add(nave,nave1,b);
        //    else
        //        add(b,res,b);
        
        //cout <<absdiff<<"............";
        #ifdef STEP_234
        /** step 2, 3, 4 */
        int na;
        unsigned char* pData = b.data;
        printf("type=%d %d\n", b.type(), CV_8UC3);
        Mat canny_output;
        vector<vector<Point> > contours;
        vector<Vec4i> hierarchy;
        int thresh = 100;
        int max_thresh = 255;
        
        Canny(b, canny_output, thresh, thresh * 2, 3);

        findContours(canny_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
        vector<vector<Point> > contours_poly( contours.size() );
        vector<Rect> boundRect( contours.size() );

        for( int i = 0; i< contours.size(); i++ )
        {
            Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
            approxPolyDP( Mat(contours[i]), contours_poly[i], 3, true );
            boundRect[i] = boundingRect( Mat(contours_poly[i]) );
            rectangle(b, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
        }
        #endif

        #ifdef STEP_5
        int const nH = b.size().height;
        int const nW = b.size().width;
        int const nStepSize = nW / GRID_WIDTH;
        int nWG = nW / nStepSize + 1;
        int nHG = nH / nStepSize + 1;
        printf("nStepSize=%d %dX%d nWG=%d nHG=%d\n", nStepSize, nW, nH, nWG, nHG);
        /** logical grids */
        /** We have nW/nStepSize X nH/nStepSize grids;
         * */
        int** ppGridM = (int**)calloc(1, nHG * sizeof(int*));
        for(int i = 0; i < nHG; i++)
            ppGridM[i] = (int*)calloc(1, nWG * sizeof(int));

        for(int i = 0; i < boundRect.size(); i++)
        {
            Point p[4];
            p[0] = Point(boundRect[i].x, boundRect[i].y); //TL
            p[1] = Point(boundRect[i].br().x, boundRect[i].br().y); //BR
            p[2] = Point(boundRect[i].x + boundRect[i].width, boundRect[i].y); //TR
            p[3] = Point(boundRect[i].x, boundRect[i].y + boundRect[i].height); //BR

            printf("(%d, %d) (%d, %d) (%d, %d) (%d, %d)\n", 
                boundRect[i].x, boundRect[i].y,
                boundRect[i].br().x, boundRect[i].br().y,
                boundRect[i].x + boundRect[i].width, boundRect[i].y,
                boundRect[i].x, boundRect[i].y + boundRect[i].height
                );
            for(int j = 0; j < 4; j++)
            {
                printf("[%d, %d]\n", p[j].x / nStepSize, p[j].y / nStepSize);
                /** grid's row is calculated from its vertical coord
                 * column num is calculated from its horiz coord */
                ppGridM[(p[j].y / nStepSize)][(p[j].x / nStepSize)]++;
            }
            /** also from TL -> TR, for every point, populate corresponding grid 
             * and grids vertically below till bottom
             * */
            for(int x1 = ((p[0].x / nStepSize) * nStepSize) + nStepSize; x1 < p[2].x; x1 += nStepSize) //shall not include TL and TR points itself
            {
                printf("sub x\n");
                ppGridM[(p[0].y / nStepSize)][(x1 / nStepSize)]++;
                /** vertically down till bottom */
                for(int y1 = ((p[0].y / nStepSize) * nStepSize) + nStepSize; y1 < p[3].y; y1 += nStepSize)
                {
                    printf("sub y\n");
                    ppGridM[y1/nStepSize][x1/nStepSize]++;
                }
            }
        }

        printf("draw grids\n");

        #ifdef DRAW_GRIDS
        /** draw grids */
        /** verticals */
        for(int i = 0; i < nW; i += nStepSize)
            line(b, Point(i, 0), Point(i, nH), Scalar(0, 255, 255));
        /** horizontals */
        for(int i = 0; i < nH; i += nStepSize)
            line(b, Point(0, i), Point(nW, i), Scalar(0, 255, 255));
        #endif

        /** draw grid cross if theres movement in grid */
        for(int i = 0; i < nHG; i++)
        {
            for(int j = 0; j < nWG; j++)
            {
                if(ppGridM[i][j])
                {
                    //line(b, Point(j*nStepSize, i*nStepSize), Point(j*nStepSize + nStepSize, i*nStepSize + nStepSize), Scalar(0, 255, 255));
                    char text[10];
                    snprintf(text, 10, "%2d", ppGridM[i][j]);
                    //circle(b, Point(j*nStepSize, i*nStepSize), 3, Scalar(0,255,0), -1, 8);
                    putText(b, text, Point(j*nStepSize, i*nStepSize + nStepSize/2), FONT_HERSHEY_PLAIN, 0.5, Scalar(255,255,255));  
                }
            }
        }
        #endif

        #if 0
        vector<Rect> clusteredRects;
        vector<int> labels;
        int euc_dist = 50;
        int n_labels = partition(boundRect, labels, [euc_dist](const Rect& lhs, const Rect& rhs){
                int euc_dist2 = euc_dist * euc_dist;
                Point a = centerOfRect(lhs);
                Point b = centerOfRect(rhs);
                if((b.x - a.x)*(b.x - a.x) + (b.y - a.y)*(b.y - a.y) < euc_dist2)
                    return true;
                else
                    return false;
            });
        int idxToRem = -1;
        for(int i = 0; i < boundRect.size(); i++)
        {
            /**  */
        }
        #endif

        #if 1
        /** cluster the rectangles:
         * 1) divide the Mat into small grids;
         * say 3 X 3 
         * 2) */

        
        
        #if 0
        vector<Rect> clusteredRects;
        for(int i = 0; i < b.rows; i++)
        {
            int endIMax = i+1;
            for(int j = 0; j < b.cols; j+=3)
            {
                unsigned char b1 = pData[i * b.step + j];
                unsigned char g1 = pData[i * b.step + j + 1];
                unsigned char r1 = pData[i * b.step + j + 2];
                /** collect consecutive grids with our color */
                {
                    Point tl, tr, bl, br;
                    int startI = 0, startJ=0, endI=0, endJ=0;
                    bool iDone = false, jDone = false;
                    printf("going in for j = %d\n", j);
                    while(findIfThereIsARectInVicinity(pData, i, j, b, MAX_EMPTY_GRIDS * GRID_WIDTH, startI, startJ, endI, endJ, color, iDone, jDone))
                    {
                        /** add rect! into list */
                        //clusteredRects.push_back(Rect(startJ, startI, (endJ - startJ), endI - startI));
                        //clusteredRects.push_back(Rect(startI, startJ/3, (endJ - startJ)/3, (endI - startI)));
                        clusteredRects.push_back(Rect(startJ/3, startI, (endI - startI), (endJ - startJ)/3));
                        j = endJ;
                        if(endI > endIMax || endIMax == 0)
                            endIMax = endI;
                    }
                    if(jDone)
                        j = b.cols;
                }
            }
            i = endIMax;
        }

        color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
        for( int i = 0; i< clusteredRects.size(); i++ )
        {
            rectangle(b, clusteredRects[i].tl(), clusteredRects[i].br(), color, 2, 8, 0);
        }
        #endif
        #endif
        imshow("cam_feed", src);
        imshow("sca5_sjsu", b);

        img_pairs_processed++;
        #ifdef DUMP_OP_IMAGE
        vector<int> compression_params;
        compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);
        snprintf(file_name, 1024, "mkdir -p output/" DUMP_OP_PATH);
        system(file_name);
        snprintf(file_name, 1024, "output/" DUMP_OP_PATH "output-%04d.png", img_pairs_processed);
        bool ret = imwrite(file_name, b, compression_params);
        printf("ret=%d\n", ret);
        #endif

        #ifdef STEP_5_PRINT_MATRIX_MATH
        printf("sheet:\n");
        for(int i = 0; i < nHG; i++)
        {
            for(int j = 0; j < nWG; j++)
            {
                printf("%02d ", ppGridM[i][j]);
            }
            printf("\n");
        }
        #endif

        if(pGrid)
        {
            pGrid->nHG = nHG;
            pGrid->nWG = nWG;
            pGrid->ppGridM = ppGridM;
            pGrid->nStepSize = nStepSize;
        }

    }

    return 1;
}
