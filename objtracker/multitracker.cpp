/*----------------------------------------------
 * Usage:
 * example_tracking_multitracker <video_name> [algorithm]
 *
 * example:
 * example_tracking_multitracker Bolt/img/%04d.jpg
 * example_tracking_multitracker faceocc2.webm KCF
 *
 * Note: after the OpenCV libary is installed,
 * please re-compile this code with "HAVE_OPENCV" parameter activated
 * to enable the high precission of fps computation
 *--------------------------------------------------*/

/* after the OpenCV libary is installed
 * please uncomment the the line below and re-compile this code
 * to enable high precission of fps computation
 */
//#define HAVE_OPENCV

#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
#include <ctime>

#include "multitracker.h"
#define DEBUG
#define VERBOSE
#include "debug.h"

#ifdef HAVE_OPENCV
#include <opencv2/flann.hpp>
#endif

#define RESET   "\033[0m"
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */

//#define USE_MULTI_TRACKER
//#define DISPLAY_RESULTS
#define MAX_TRACK_ITERATIONS 1

#define TRACKING_ALGO "MEDIAN_FLOW"
//#define TRACKING_ALGO "KCF"

inline cv::Ptr<cv::Tracker> createTrackerByName(cv::String name)
{
    cv::Ptr<cv::Tracker> tracker;

    if (name == "KCF")
        tracker = cv::TrackerKCF::create();
    else if (name == "TLD")
        tracker = cv::TrackerTLD::create();
    else if (name == "BOOSTING")
        tracker = cv::TrackerBoosting::create();
    else if (name == "MEDIAN_FLOW")
        tracker = cv::TrackerMedianFlow::create();
    else if (name == "MIL")
        tracker = cv::TrackerMIL::create();
    else if (name == "GOTURN")
        tracker = cv::TrackerGOTURN::create();
    else
        CV_Error(cv::Error::StsBadArg, "Invalid tracking algorithm name\n");

    return tracker;
}

using namespace std;
using namespace cv;

#ifdef TEST_CODE
int main( int argc, char** argv ){
  // show help
  if(argc<2){
    cout<<
      " Usage: example_tracking_multitracker <video_name> [algorithm]\n"
      " examples:\n"
      " example_tracking_multitracker Bolt/img/%04d.jpg\n"
      " example_tracking_multitracker faceocc2.webm MEDIANFLOW\n"
      " \n"
      " Note: after the OpenCV libary is installed,\n"
      " please re-compile with the HAVE_OPENCV parameter activated\n"
      " to enable the high precission of fps computation.\n"
      << endl;
    return 0;
  }

  // timer
#ifdef HAVE_OPENCV
  cvflann::StartStopTimer timer;
#else
  clock_t timer;
#endif

  // for showing the speed
  double fps;
  String text;
  char buffer [50];

  // set the default tracking algorithm
  String trackingAlg = "KCF";

  // set the tracking algorithm from parameter
  if(argc>2)
    trackingAlg = argv[2];

  // create the tracker
  MultiTracker trackers;

  // container of the tracked objects
  vector<Rect> ROIs;
  vector<Rect2d> objects;

  // set input video
  String video = argv[1];
  VideoCapture cap(video);

  Mat frame;

  // get bounding box
  cap >> frame;
  selectROIs("tracker",frame,ROIs);
  printf("selecting ROI done size=%d\n", ROIs.size());

  //quit when the tracked object(s) is not provided
  if(ROIs.size()<1)
    return 0;

  std::vector<Ptr<Tracker> > algorithms;
      algorithms.push_back(createTrackerByName(trackingAlg));
  for (size_t i = 0; i < ROIs.size(); i++)
  {
      objects.push_back(ROIs[i]);
  }

  // initialize the tracker
  trackers.add(algorithms,frame,objects);

  // do the tracking
  printf(GREEN "Start the tracking process, press ESC to quit.\n" RESET);
  for ( ;; ){
    // get frame from the video
    cap >> frame;

    // stop the program if no more images
    if(frame.rows==0 || frame.cols==0)
      break;

    // start the timer
#ifdef HAVE_OPENCV
    timer.start();
#else
    timer=clock();
#endif

    //update the tracking result
    //trackers.update(frame);

    // calculate the processing speed
#ifdef HAVE_OPENCV
    timer.stop();
    fps=1.0/timer.value;
    timer.reset();
#else
    timer=clock();
    trackers.update(frame);
    timer=clock()-timer;
    fps=(double)CLOCKS_PER_SEC/(double)timer;
#endif

    // draw the tracked object
    for(unsigned i=0;i<trackers.getObjects().size();i++)
      rectangle( frame, trackers.getObjects()[i], Scalar( 255, 0, 0 ), 2, 1 );

    // draw the processing speed
    sprintf (buffer, "speed: %.0f fps", fps);
    text = buffer;
    putText(frame, text, Point(20,20), FONT_HERSHEY_PLAIN, 1, Scalar(255,255,255));

#ifdef DISPLAY_RESULTS
    // show image with the tracked object
    imshow("tracker",frame);

    //quit on ESC button
    if(waitKey(1)==27)break;
#endif /**< DISPLAY_RESULTS */
  }

}
#endif /**< TEST_CODE */

/** synchronous function
 * caller is responsible for image data passed
 */
static Mat image_to_mat(tFrameInfo* pF)
{
    IplImage* ipl;

    ipl = cvCreateImage(cvSize(pF->im.w,pF->im.h), IPL_DEPTH_8U, pF->im.c);
    ipl->widthStep = pF->widthStep;
    LOGV("width=%d\n", ipl->widthStep);
    ipl->imageData = (char*)pF->im.data;
    return cvarrToMat(ipl);
}
int track_bb_in_frame(tAnnInfo* apBoundingBoxesIn, tFrameInfo* pFBase, tFrameInfo* pFTarg, tAnnInfo** appBoundingBoxesOut)
{
    int ret = 0;
    Mat imgBaseM;
    Mat imgTargM;
    tAnnInfo* pBB;
    
    LOGV("DEBUGME\n");
    if(!apBoundingBoxesIn || !appBoundingBoxesOut || !pFBase || !pFTarg)
    {
        return ret;
    }

    LOGV("DEBUGME w=%d h=%d pFBase->im.c=%d\n", pFBase->im.w, pFBase->im.h, pFBase->im.c);
    imgBaseM = image_to_mat(pFBase);
    imgTargM = image_to_mat(pFTarg);

#if 0
    imshow("base", imgBaseM);
    imshow("targ", imgTargM);
    waitKey(1);
#endif
    /** use base as reference mat to find the bounding boxes on */
  // timer
#ifdef HAVE_OPENCV
  cvflann::StartStopTimer timer;
#else
  clock_t timer;
#endif

    // for showing the speed
    double fps;
    String text;
    char buffer [50];

    // set the default tracking algorithm
    String trackingAlg = TRACKING_ALGO;

    // create the tracker
#ifdef USE_MULTI_TRACKER
    MultiTracker trackers;
#else
#endif

    std::vector<Ptr<Tracker> > algorithms;

    LOGV("DEBUGME\n");
#ifdef USE_MULTI_TRACKER
    // container of the to be tracked objects
    vector<Rect2d> objects;
    pBB = apBoundingBoxesIn;
    while(pBB)
    {
        algorithms.push_back(createTrackerByName(trackingAlg));
        LOGV("adding %d %d %d %d\n", pBB->x, pBB->y, pBB->w, pBB->h);
        objects.push_back(Rect2d(pBB->x, pBB->y, pBB->w, pBB->h));
        pBB = pBB->pNext;
    }
#endif
    LOGV("DEBUGME\n");


    LOGV("DEBUGME\n");
    // initialize the tracker
#ifdef USE_MULTI_TRACKER
    trackers.add(algorithms,imgBaseM,objects);
#else
#endif

    LOGV("DEBUGME\n");

    // do the tracking
    {
        // start the timer
  #ifdef HAVE_OPENCV
        timer.start();
  #else
        timer=clock();
  #endif
  
        //update the tracking result
        //trackers.update(frame);
  
        // calculate the processing speed
  #ifdef HAVE_OPENCV
        timer.stop();
        fps=1.0/timer.value;
        timer.reset();
  #else
        timer=clock();
        #ifdef USE_MULTI_TRACKER
        trackers.update(imgTargM);
        #else
        #endif
        timer=clock()-timer;
        fps=(double)CLOCKS_PER_SEC/(double)timer;
  #endif
  
#ifdef USE_MULTI_TRACKER
        LOGV("DEBUGME %d\n", (int)trackers.getObjects().size());
        // draw the tracked object
        for(unsigned i=0;i<trackers.getObjects().size();i++)
          rectangle(imgTargM, trackers.getObjects()[i], Scalar( 255, 0, 0 ), 2, 1 );
#endif

#ifndef USE_MULTI_TRACKER
        bool ret;
        pBB = apBoundingBoxesIn;
        tAnnInfo* pOutBBs = NULL;
        tAnnInfo* pBBTmp;
        while(pBB)
        {
            Ptr<Tracker> tracker = createTrackerByName(trackingAlg);
            if(!tracker)
                continue;
            ret = tracker->init(imgBaseM, Rect2d(pBB->x, pBB->y, pBB->w, pBB->h));
            LOGV("initialized with %d %d %d %d ret=%d\n", pBB->x, pBB->y, pBB->w, pBB->h, ret);
            if(ret)
            {
                Rect2d object;
                //update with target
                for(int i = 0; i < MAX_TRACK_ITERATIONS; i++)
                {
                    ret = tracker->update(imgTargM, object);
                    if(ret)
                    {
                        LOGV("update: (%f, %f) (%f, %f)\n", object.x, object.y, object.width, object.height);
                    }
                }
                if(ret)
                {
                    LOGD("found\n");
                    #ifdef DISPLAY_RESULTS
                    rectangle(imgTargM, object, Scalar( 255, 0, 0 ), 2, 1 );
                    #endif
                    pBBTmp = (tAnnInfo*)malloc(sizeof(tAnnInfo));
                    memcpy(pBBTmp, pBB, sizeof(tAnnInfo));
                    pBBTmp->pcClassName = (char*)malloc(strlen(pBB->pcClassName) + 1);
                    strcpy(pBBTmp->pcClassName, pBB->pcClassName);
                    pBBTmp->x = (int)(object.x);
                    pBBTmp->y = (int)(object.y);
                    pBBTmp->w = (int)(object.width);
                    pBBTmp->h = (int)(object.height);
                    pBBTmp->fCurrentFrameTimeStamp = pFTarg->fCurrentFrameTimeStamp;
                    pBBTmp->pNext = pOutBBs;
                    pOutBBs = pBBTmp;
                    LOGV("stored %d %d %d %d ret=%d\n", pBBTmp->x, pBBTmp->y, pBBTmp->w, pBBTmp->h, ret);
                }
                else
                {
                    LOGD("unable to track this object in tracker\n");
                }
            }
            //objects.push_back(Rect2d(pBB->x, pBB->y, pBB->w, pBB->h));
            pBB = pBB->pNext;
            //delete tracker;
        }
        *appBoundingBoxesOut = pOutBBs;
#endif
#if 0
        if(trackers.getObjects().size())
        {
            tAnnInfo* pOutBBs = NULL;
            tAnnInfo* pInBB = apBoundingBoxesIn;
            tAnnInfo* pBB = NULL;
            for(unsigned i=0;i<trackers.getObjects().size();i++)
            {
                pBB = (tAnnInfo*)malloc(sizeof(tAnnInfo));
                memcpy(pBB, p);
            }
        }
#endif
  
#ifdef DISPLAY_RESULTS
        // draw the processing speed
        sprintf (buffer, "speed: %.0f fps", fps);
        text = buffer;
        putText(imgTargM, text, Point(20,20), FONT_HERSHEY_PLAIN, 1, Scalar(255,255,255));
  
        // show image with the tracked object
        imshow("tracker",imgTargM);
  
        //quit on ESC button
        waitKey(1);
#endif
    }


    cleanup:

    return ret;
}

int tracker_display_frame(tAnnInfo* apBoundingBoxesIn, tFrameInfo* pFBase)
{
    int ret = 0;
    Mat imgBaseM;
    tAnnInfo* pBB;
    
    LOGV("DEBUGME %p %p\n", apBoundingBoxesIn, pFBase);
    if(!apBoundingBoxesIn || !pFBase)
    {
        return ret;
    }

    LOGV("DEBUGME w=%d h=%d pFBase->im.c=%d\n", pFBase->im.w, pFBase->im.h, pFBase->im.c);
    imgBaseM = image_to_mat(pFBase);

    pBB = apBoundingBoxesIn;
    while(pBB)
    {
        rectangle(imgBaseM, Rect(pBB->x, pBB->y, pBB->w, pBB->h), Scalar( 255, 0, 0 ), 2, 1);
        pBB = pBB->pNext;
    }
#if 1
    imshow("base", imgBaseM);
    waitKey(1);
#endif

}
