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
#include "opencv2/video/tracking.hpp"

#include "multitracker.h"
//#define DEBUG
#define VERBOSE
#include "debug.h"

#ifdef HAVE_OPENCV
#include <opencv2/flann.hpp>
#endif

#define RESET   "\033[0m"
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */

//#define USE_MULTI_TRACKER
#define DISPLAY_RESULTS
#define MAX_TRACK_ITERATIONS 1

#define TRACKING_ALGO "MEDIAN_FLOW"
//#define TRACKING_ALGO "MIL"
#define USE_CV_TRACKING
#define OPTICAL_FLOW
#define OPTICAL_FLOW_APPROXIMATION

#define ABS_DIFF(a, b) ((a) > (b)) ? ((a)-(b)) : ((b)-(a))
#define GOOD_IOU_THRESHOLD (0.4)
#define MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW 5

#define CLASS_AGNOSTIC_BB_TRACKING
#define ASSIGN_BBID_ONLY_ONCE

extern "C"
{

typedef struct CandidateBB tCandidateBB;

struct CandidateBB
{
    tAnnInfo* pBBD;
    double fIoU;
    tCandidateBB* pNext;
};

typedef struct
{
    tAnnInfo trackerBB;
    tAnnInfo opticalFlowBB;
    int bInDetectionList;
    tCandidateBB* pCandidateBBs; /**< a list of detected BBs that could be a match; copied from orig; pointers inside BBs will be hanging */
}tTrackerBBInfo;

void assess_iou_trackerBBs_detectedBBs(tTrackerBBInfo* pTrackerBBs,
                const int nTrackerInSlots,
                tAnnInfo* pDetectedBBs);
}

tAnnInfo* get_apt_candidateBB(tTrackerBBInfo* pTrackerBBs, const int nTrackerInSlots, const int i);

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

void display_results(Mat& imgTargM, tAnnInfo* pFinal);

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

int track_bb_in_frame(tAnnInfo* apBoundingBoxesIn, tFrameInfo* pFBase, tFrameInfo* pFTarg, tAnnInfo** appBoundingBoxesInOut)
{
    int ret = 0;
    Mat imgBaseM;
    Mat imgTargM;
    tAnnInfo* pBB;
    tTrackerBBInfo* pTrackerBBs = NULL;
    int nInBBs = 0;
    int idxIn = 0;
    
    LOGV("DEBUGME\n");
    if(!apBoundingBoxesIn || !appBoundingBoxesInOut || !pFBase || !pFTarg)
    {
        return ret;
    }

    pBB = apBoundingBoxesIn;
    while(pBB)
    {
        nInBBs++;
        pBB = pBB->pNext;
    }
    pTrackerBBs = (tTrackerBBInfo*)calloc(nInBBs, sizeof(tTrackerBBInfo));

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
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
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
        /** optical flow */
        
#ifdef OPTICAL_FLOW
        pBB = apBoundingBoxesIn;
        idxIn = 0;
        tAnnInfo* pOpticalFlowOutBBs = NULL;
        while(pBB)
        {
            tAnnInfo* pBBTmp;
            vector<uchar> status;
            vector<float> err;
            vector<Point2f> points[2];
            points[0].push_back(Point2f((float)(pBB->x) + ((float)pBB->w)/2, (float)(pBB->y) + (float)(pBB->h)/2));
            Size subPixWinSize((float)pBB->w/2,(float)pBB->h/2);
            Mat gray;
            cvtColor(imgBaseM, gray, COLOR_BGR2GRAY);
            //cornerSubPix(gray, points[0], subPixWinSize, Size(-1,-1), termcrit);
            //Size winSize(1920,1080);
            Size winSize(100,100);
            calcOpticalFlowPyrLK(imgBaseM, imgTargM, points[0], points[1], status, err, winSize,
                                 3, termcrit, 0, 0.001);
            size_t i, k;
            LOGV("number of output points=%ld\n", points[1].size());
            for( i = k = 0; i < points[1].size(); i++ )
            {
                //if( norm(point - points[1][i]) <= 5 )

                if( !status[i] )
                    continue;

                points[1][k++] = points[1][i];
                circle( imgTargM, points[1][i], 3, Scalar(0,255,0), -1, 8);
            }
            points[1].resize(k);
            pBBTmp = &pTrackerBBs[idxIn].opticalFlowBB;
            memcpy(pBBTmp, pBB, sizeof(tAnnInfo));
            pBBTmp->x = (int)(points[1][0].x);
            pBBTmp->y = (int)(points[1][0].y);
            pBBTmp->pcClassName = (char*)malloc(strlen(pBB->pcClassName) + 1);
            strcpy(pBBTmp->pcClassName, pBB->pcClassName);
            pBBTmp->fCurrentFrameTimeStamp = pFTarg->fCurrentFrameTimeStamp;
            pBBTmp->pNext = pOpticalFlowOutBBs;
            pOpticalFlowOutBBs = pBBTmp;
            idxIn++;
            pBB = pBB->pNext;
        }
#endif


#ifdef USE_CV_TRACKING
        bool ret;
        pBB = apBoundingBoxesIn;
        tAnnInfo* pTrackerOutBBs = NULL;
        idxIn = 0;
        while(pBB)
        {
            tAnnInfo* pBBTmp;
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
                    pBBTmp = &pTrackerBBs[idxIn].trackerBB;
                    memcpy(pBBTmp, pBB, sizeof(tAnnInfo));
                    pBBTmp->x = (int)(object.x);
                    pBBTmp->y = (int)(object.y);
                    pBBTmp->w = (int)(object.width);
                    pBBTmp->h = (int)(object.height);
                    pBBTmp->pcClassName = (char*)malloc(strlen(pBB->pcClassName) + 1);
                    strcpy(pBBTmp->pcClassName, pBB->pcClassName);
                    pBBTmp->fCurrentFrameTimeStamp = pFTarg->fCurrentFrameTimeStamp;
                    pBBTmp->pNext = pTrackerOutBBs;
                    pTrackerOutBBs = pBBTmp;
                    pTrackerBBs[idxIn].trackerBB = *pBBTmp;
                    LOGV("stored %d %d %d %d ret=%d\n", pBBTmp->x, pBBTmp->y, pBBTmp->w, pBBTmp->h, ret);
                }
                else
                {
                    LOGD("unable to track this object in tracker\n");
                }
            }
            //objects.push_back(Rect2d(pBB->x, pBB->y, pBB->w, pBB->h));
            pBB = pBB->pNext;
            idxIn++;
            //delete tracker;
        }
#endif

        /** process BB's tracked in the detection list */
            /** make the final output list of BBs from the lists generated by
             * tracker and optical flow algos
             */
            /** When both came up with results,
             * take the result which has best IoU match with pBB(parent) 
             */
#if 0
        /** now do a IoU assessment btw the 2 BBs and come up with the list of BBs */
        assess_iou_trackedBBs_detectedBBs(pTrackerOutBBs,
                    *appBoundingBoxesInOut
                    );
#endif
        assess_iou_trackerBBs_detectedBBs(pTrackerBBs,
                    nInBBs,
                    *appBoundingBoxesInOut
                    );
        //*appBoundingBoxesInOut = pTrackerOutBBs;
#endif
#if 0
        if(trackers.getObjects().size())
        {
            tAnnInfo* pTrackerOutBBs = NULL;
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
  
        display_results(imgTargM, *appBoundingBoxesInOut);
        // show image with the tracked object
        imshow("tracker",imgTargM);
  
        //quit on ESC button
        waitKey(1);
#endif
    }



    cleanup:

    if(pTrackerBBs)
        free(pTrackerBBs);
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

/** our IoU analysis functions */
double find_iou(tAnnInfo* pBB1, tAnnInfo* pBB2)
{
    /* The intersection: (w, h) 
     * = ( ((w1 + w2) - (max(x1+w,x2+w) - min(x1,x2))), 
     *          ((h1 + h2) - (max(y1,y2) - min(y1,y2))) )
     * union area = a1 + a1 - a_intersection
     * */ 
    double nAInter, nA1, nA2, nAUnion, wI, hI;
    double iou = 0.0;
    nA1 = (double)(pBB1->w * pBB1->h);
    nA2 = (double)(pBB2->w * pBB2->h);

    wI = (double)((pBB1->w + pBB2->w) - (MAX(pBB1->x + pBB1->w, pBB2->x + pBB2->w) - MIN(pBB1->x, pBB2->x)));
    hI = (double)((pBB1->h + pBB2->h) - (MAX(pBB1->y + pBB1->h, pBB2->y + pBB2->h) - MIN(pBB1->y, pBB2->y)));
    nAInter = wI * hI;

    nAUnion = nA1 + nA2 - nAInter;

    iou = nAUnion ? nAInter / nAUnion : 0;

    if(iou >= 0.0 && iou <= 1.0)
        return iou;

    return 0.0;
}

double return_best_iou(double bb1, double bb2)
{
    /** best IoU is 1.0 */
    double bb1_distance_from_best_iou = ABS_DIFF(1.0, bb1);
    double bb2_distance_from_best_iou = ABS_DIFF(1.0, bb2);

    if(bb1_distance_from_best_iou <= bb2_distance_from_best_iou)
        return bb1;
   
    return bb2; 
}

tAnnInfo* return_BB_with_best_iou(tAnnInfo* pBB1, tAnnInfo* pBB2)
{
    if(!pBB1 || !pBB2)
        return NULL;

    if(!pBB1->pcClassName)
        return pBB2;

    if(!pBB2->pcClassName)
        return pBB1;

    if(return_best_iou(pBB1->fIoU, pBB2->fIoU) == pBB1->fIoU)
        return pBB1;

    return pBB2;
}

#if 0
void assess_iou_trackedBBs_detectedBBs(tAnnInfo* pTrackedBBs,
                tAnnInfo* pDetectedBBs)
{
    tAnnInfo* pBBT;
    tAnnInfo* pBBD;
    tAnnInfo* pBBDWithMaxIoU;
    if(!pTrackedBBs || !pDetectedBBs)
        return;

    pBBT = pTrackedBBs;
    pBBD = pDetectedBBs;

    while(pBBD)
    {
        pBBD->fIoU = 0;
        pBBD->bBBIDAssigned = 0;
        pBBD = pBBD->pNext;
    }

    /** for all tracked BBs, find the corresponding BB in the detection result by
     * matching the IoU between tracked BBs and detected BBs 
     */

    /** find matches for objects in pTrackedBBs in pDetectedBBs */
    
    pBBT = pTrackedBBs;
    while(pBBT)
    {
        LOGV("finding match for [%s] (%d, %d)\n", pBBT->pcClassName, pBBT->x, pBBT->y);
        pDetectedBBs->fIoU = 0.0;
        pBBD = pBBDWithMaxIoU = pDetectedBBs;
        while(pBBD)
        {
            if(pBBD->bBBIDAssigned)
            {
                LOGV("IoU already assigned\n");
            }
            /** check & accept max IoU if the classnames match */
            pBBD->fIoU = find_iou(pBBT, pBBD);
            LOGV("IoU=%f; currentMax=%f [%s] (%d, %d):(%d,%d)\n", pBBD->fIoU, pBBDWithMaxIoU->fIoU, pBBD->pcClassName, pBBD->x, pBBD->y, pBBT->x, pBBT->y);
            if(return_BB_with_best_iou(pBBDWithMaxIoU, pBBD) == pBBD
                && (strcmp(pBBT->pcClassName, pBBD->pcClassName) == 0))
            {
                if(pBBD->fIoU > GOOD_IOU_THRESHOLD)
                {
                    LOGV("max IoU changed\n");
                    pBBDWithMaxIoU = pBBD;
                    /** this is the tracked BB in pDetectedBBs,
                     * so copy the BBID into pDetectedBB candidate */
                    pBBDWithMaxIoU->nBBId = pBBT->nBBId;
                    pBBDWithMaxIoU->bBBIDAssigned = 1;
                    LOGV("changed detected BBID to %d\n", pBBDWithMaxIoU->nBBId);
                }
            }
            pBBD = pBBD->pNext;
        }
        pBBT = pBBT->pNext;
    }

    return;
}
#endif

void assess_iou_trackerBBs_detectedBBs(tTrackerBBInfo* pTrackerBBs,
                const int nTrackerInSlots,
                tAnnInfo* pDetectedBBs)
{
    if(!pTrackerBBs || !pDetectedBBs)
        return;

    tAnnInfo* ppBBT[2];
    tAnnInfo* pBBT = NULL;
    tAnnInfo* pBBD = pDetectedBBs;
    tAnnInfo* pOutBBs = NULL;
    tAnnInfo* pBBDPrev = NULL;
    while(pBBD)
    {
        pBBD->fIoU = 0;
        pBBD->bBBIDAssigned = 0;
        pBBD = pBBD->pNext;
    }

    /** for all tracked BBs, find the corresponding BB in the detection result by
     * matching the IoU between tracked BBs and detected BBs 
     */

    /** find matches for objects btw pTrackerBBs in pDetectedBBs */

    for(int i = 0; i < nTrackerInSlots; i++)
    {
        ppBBT[0] = &pTrackerBBs[i].trackerBB;
        ppBBT[1] = &pTrackerBBs[i].opticalFlowBB;
        pBBD = pDetectedBBs;
        tAnnInfo BBDWithMaxIoU = *pBBD; /**< only fIoU value matters in this */
        tCandidateBB* pBBDCandidate = NULL;
        while(pBBD)
        {
            pBBD->fIoU = 0;
            #ifdef ASSIGN_BBID_ONLY_ONCE
            LOGV("pBBD %p %d\n", pBBD, pBBD->bBBIDAssigned);
            if(pBBD->bBBIDAssigned == 1)
            {
                LOGV("IoU already assigned\n");
                pBBD = pBBD->pNext;
                continue;
            }
            #endif
            /** check & accept max IoU if the classnames match */
            if(!ppBBT[0]->pcClassName && !ppBBT[1]->pcClassName)
            {
                LOGV("both trackers could'nt find this input box\n");
                pBBD = pBBD->pNext;
                continue;
            }
            for(int k = 0; k < 2; k++)
            {
                if(ppBBT[k]->pcClassName)
                {
                    ppBBT[k]->fIoU = find_iou(ppBBT[k], pBBD);
#ifdef OPTICAL_FLOW_APPROXIMATION
                    if(k == 1)
                        if((ppBBT[k]->x < pBBD->x + pBBD->w + MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW) 
                           && (ppBBT[k]->y  < pBBD->y + pBBD->h + MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW) 
                           && (ppBBT[k]->x > (pBBD->x > MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW ? pBBD->x - MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW : 0)) 
                           && (ppBBT[k]->y > (pBBD->y > MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW ? pBBD->y - MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW : 0))
                          )
                            ppBBT[k]->fIoU = 1.0; /**< optical flow alone can have this way */
#endif
                }
            }
            LOGV("IoU results %f %f\n", ppBBT[0]->fIoU, ppBBT[1]->fIoU);
            pBBT = return_BB_with_best_iou(ppBBT[0], ppBBT[1]);
            if(pBBT->fIoU == 0)
            {
                LOGV("both trackers couldn't find this BB\n");
                pBBD = pBBD->pNext;
                continue;
            }
            LOGV("finding match for [%s] (%d, %d)\n", pBBT->pcClassName, pBBT->x, pBBT->y);
            /** select the best tracked BB as the final track result */
            LOGV("IoU=%f; currentMax=%f [%s] (%d, %d):(%d,%d)\n", pBBT->fIoU, BBDWithMaxIoU.fIoU, pBBD->pcClassName, pBBD->x, pBBD->y, pBBT->x, pBBT->y);
            LOGV("%p %p\n", pBBT, return_BB_with_best_iou(&BBDWithMaxIoU, pBBT));
            if(return_BB_with_best_iou(&BBDWithMaxIoU, pBBT) == pBBT /** best tracker IoU of the IoUs with detector BBs, save as current best */
                #ifndef CLASS_AGNOSTIC_BB_TRACKING
                && (strcmp(pBBT->pcClassName, pBBD->pcClassName) == 0)
                #endif /**< CLASS_AGNOSTIC_BB_TRACKING */
              )
            {
                pTrackerBBs[i].bInDetectionList = 1;
                if(pBBT->fIoU > GOOD_IOU_THRESHOLD
                  )
                {
                    LOGV("DEBUGME\n");
                    /** is this the second time we are finding a match for the same pBBT in pBBD list?
                     * if so, before assigning the  */
                    pBBDCandidate = (tCandidateBB*)calloc(1, sizeof(tCandidateBB));
                    pBBDCandidate->pBBD = pBBD;
                    pBBD->fIoU = pBBDCandidate->fIoU = pBBT->fIoU; /**< this value we will be using later */
                    tCandidateBB* pBBDCPrev = NULL;
                    tCandidateBB* pBBDC = pTrackerBBs[i].pCandidateBBs;
                    while(pBBDC)
                    {
                        if(return_best_iou(pBBDCandidate->fIoU, pBBDC->fIoU) == pBBDCandidate->fIoU);
                        {
                            break;
                        }
                        pBBDCPrev = pBBDC;
                        pBBDC->pNext;
                    }
                    pBBDCandidate->pNext = pBBDC;
                    if(pBBDCPrev)
                        pBBDCPrev->pNext = pBBDCandidate;
                    else
                        pTrackerBBs[i].pCandidateBBs = pBBDCandidate; /**< thus the pBBD with best IoU will be first in list */
                    BBDWithMaxIoU = *pBBD;
                    LOGV("DEBUGME\n");
                }
            }
            pBBDPrev = pBBD;
            pBBD = pBBD->pNext;
        }

#if 0
        if(pBBDCandidate)
        {
            pBBDCandidate->nBBId = pTrackerBBs[i].opticalFlowBB.nBBId;
            pBBDCandidate->bBBIDAssigned = 1;
            LOGV("max IoU changed\n");
            /** this is the tracked BB in pDetectedBBs,
             * so copy the BBID into pDetectedBB candidate */
            LOGV("changed detected BBID to %d\n", pBBD->nBBId);
        }
#endif
#if 0
        if(pTrackerBBs[i].bInDetectionList == 0)
        {
            tAnnInfo* pBBTmp;
            tAnnInfo* pBBTmp1 = NULL;
            if(pTrackerBBs[i].opticalFlowBB.pcClassName)
                pBBTmp1 = &pTrackerBBs[i].opticalFlowBB;
            else if(pTrackerBBs[i].trackerBB.pcClassName)
                pBBTmp1 = &pTrackerBBs[i].trackerBB;
            if(pBBTmp1)
            {
                pBBTmp = (tAnnInfo*)malloc(sizeof(tAnnInfo));
                memcpy(pBBTmp, pBBTmp1, sizeof(tAnnInfo));
                pBBTmp->pcClassName = (char*)malloc(strlen(pBBTmp1->pcClassName) + 1);
                strcpy(pBBTmp->pcClassName, pBBTmp1->pcClassName);
            }
            pBBTmp->pNext = pOutBBs;
            pOutBBs = pBBTmp;
        }
#endif
    }

    /** now evaluate candidates and assign them in the best possible way */
    for(int i = 0; i < nTrackerInSlots; i++)
    {
        /** for each tracked object, check if the candidates conflict with
         * any other tracked objects' candidate list
         * If it conflict and the peer's IoU is better than current IoU,
         * chuck it and move forward in the list performing same operation
         */
        LOGV("DEBUGME\n");
        pBBD = get_apt_candidateBB(pTrackerBBs, nTrackerInSlots, i);
        LOGV("candidate=%p\n", pBBD);
        if(pBBD)
        {
            pBBD->nBBId = pTrackerBBs[i].opticalFlowBB.nBBId;
            LOGV("new BBId=%d\n", pTrackerBBs[i].opticalFlowBB.nBBId);
        }

        
    }
    

#if 0
    if(pBBDPrev)
        pBBDPrev->pNext = pOutBBs;
#endif

    return;
}

void display_results(Mat& imgTargM, tAnnInfo* pFinal)
{
    tAnnInfo* pBB = pFinal;

    while(pBB)
    {
        
        rectangle(imgTargM, Rect2d((double)pBB->x, (double)pBB->y, (double)pBB->w, (double)pBB->h), Scalar( 255, 0, 0 ), 2, 1 );
        char disp[50] = {0};
        LOGV("disp: [%d(%d, %d):%s]\n", pBB->nBBId, pBB->x, pBB->y, pBB->pcClassName);
        snprintf(disp, 50-1, "[%d(%d, %d):%s]", pBB->nBBId, pBB->x, pBB->y, pBB->pcClassName);
        putText(imgTargM, disp, Point(pBB->x,pBB->y), FONT_HERSHEY_PLAIN, 1, Scalar(255,255,255));
        pBB = pBB->pNext;
    }
}

int check_for_conflict(tTrackerBBInfo* pTrackerBB, tCandidateBB* pCandidateBB)
{
    tCandidateBB* pBBDCandidate = pTrackerBB->pCandidateBBs;
    while(pBBDCandidate)
    {
        if(pBBDCandidate->pBBD == pCandidateBB->pBBD)
        {
            LOGV("candidate match %f, %f\n", pBBDCandidate->fIoU, pCandidateBB->fIoU);
            if(return_best_iou(pBBDCandidate->fIoU, pCandidateBB->fIoU) == pBBDCandidate->fIoU)
                return 1; /**< conflict detected */
        }
        pBBDCandidate = pBBDCandidate->pNext;
    }
    return 0;
}

tAnnInfo* get_apt_candidateBB(tTrackerBBInfo* pTrackerBBs, const int nTrackerInSlots, const int i)
{
    tCandidateBB* pBBDCandidate = pTrackerBBs[i].pCandidateBBs;
    while(pBBDCandidate)
    {
        int isConflicting = 0;
        for(int k = 0; k < nTrackerInSlots; k++)
        {
            if(k == i)
                continue;
            isConflicting = check_for_conflict(&pTrackerBBs[k], pBBDCandidate);
            LOGV("k=%d, pBBDCandidate=%p conflicting?=%d\n", k, pBBDCandidate, isConflicting);
            if(isConflicting)
                break;
        }
        if(!isConflicting)
        {
            return pBBDCandidate->pBBD;
        }
        pBBDCandidate = pBBDCandidate->pNext;
    }
    LOGV("DEBUGME\n");
    return NULL;
}
