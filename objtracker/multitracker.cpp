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

#include "sca5.h"

//#define DUMP_OP_IMAGE
//#define DUMP_OP_PATH "/hash30/"

#include "multitracker.h"
//#define DEBUG
#define VERBOSE
#include "debug.h"
#include "semaphore.h"

#ifdef HAVE_OPENCV
#include <opencv2/flann.hpp>
#endif

#define RESET   "\033[0m"
#define RED     "\033[31m"      /* Red */
#define GREEN   "\033[32m"      /* Green */

#define DISPLAY_RESULTS
//#define DISPLAY_LANES
#define MAX_TRACK_ITERATIONS 1

//#define TRACKING_ALGO "MEDIAN_FLOW"
#define TRACKING_ALGO "BOOSTING"//"MIL"
//#define TRACKING_ALGO "MIL"
#define USE_CV_TRACKING
#define OPTICAL_FLOW
/** Force OFF the Optical flow resuults with the foll. macro */
#define OPT_FLOW_WINSIZE_W 50
#define OPT_FLOW_WINSIZE_H 50

#define OPT_FLOW_WINSIZE_W_MAX 150
#define OPT_FLOW_WINSIZE_H_MAX 150

#define ABS_DIFF(a, b) (((a) > (b)) ? ((a)-(b)) : ((b)-(a)))
#define GOOD_IOU_THRESHOLD (0.5)
#define MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW 5

#define CLASS_AGNOSTIC_BB_TRACKING
#define ASSIGN_BBID_ONLY_ONCE

#define MULTI_ALGORITHMIC_APPROACH


#ifdef MULTI_ALGORITHMIC_APPROACH

//#define FORCE_OFF_OPTICAL_FLOW_RESULTS
#define OPTICAL_FLOW_APPROXIMATION

#endif

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
    tAnnInfo* pBBT;
    tAnnInfo* pBBTOrig;
}tTrackerBBInfo;

void assess_iou_trackerBBs_detectedBBs(tTrackerBBInfo* pTrackerBBs,
                const int nTrackerInSlots,
                tAnnInfo** ppDetectedBBs,
                tLanesInfo* pLanesInfo);
tAnnInfo* make_copy_of_current_set(tAnnInfo* pDetectedBBs);
/** 
 * @fn int check_if_KalmanAffirmTracker(tAnnInfo* pBBIn, tAnnInfo* apCopyDetectedBBs, tAnnInfo* pDetectedBBs)
 * @param pBBIn [IN] Current Tracked Bounding box
 * @param apCopyDetectedBBs [IN] Dected objects' Bounding boxes from the Frame Previous
 * @param pDetectedBBs [IN] Dected objects' Bounding boxes from the Frame Current
 * @return 1: if Kalman results confirm that the tracking is confidently proper
 *            In this case, even if Object Detection failed, Kalman helps us affirm that 
 *            tracker results are proper and helps us continue to track the object
 *            In fact Kalman results can override our tracker result (shall confirm by trail and error)
 * @brief     Thus, this helps us continue tracking an object even if object detection/tracking fails 
 *            (probably when the object went under a 100% occlusion)
 *            This occlusion could be -- in traffic -- a truck moving by our target object, say a car
 *            and thus blocking the Camera view on that car!
 * ; else return 0
 */
//int check_if_KalmanAffirmTracker(tAnnInfo* pBBIn, tAnnInfo* apCopyDetectedBBs, tAnnInfo* pDetectedBBs);
int check_if_new_BB_acceptable(tAnnInfo* pBBIn, tAnnInfo* apCopyDetectedBBs, tAnnInfo* pDetectedBBs);
int isWithinBB(tAnnInfo* pBBP, tAnnInfo* pBBD);
int isWithinBB_UseCenter(tAnnInfo* pBBP, tAnnInfo* pBBD);
void collect_analysis(tAnnInfo* pCurrFrameBBs, tAnnInfo* pPrevFrameBBs, tLanesInfo* pLanesInfo);
tLane* laneWithThisBB(tLanesInfo* pLanesInfo, tAnnInfo* pBBNode);

}

static tAnnInfo* pCopyDetectedBBs;

tAnnInfo* get_apt_candidateBB(tTrackerBBInfo* pTrackerBBs, const int nTrackerInSlots, const int i);

#include <opencv2/tracking/tracker.hpp>

inline cv::Ptr<cv::Tracker> createTrackerByName(cv::String name)
{
    cv::Ptr<cv::Tracker> tracker;
#if 0
    cv::TrackerKCF::Params params = {
      0.5,
      sigma=
      0.2,
      lambda=
        0.01,
      interp_factor=
        0.075,
      output_sigma_factor=
        1.0/16.0,
      resize=
        true,
      max_patch_size=
        80*80,
      split_coeff=
        true,
      wrap_kernel=
        false,
      desc_npca = 
        GRAY,
      desc_pca = 
        CN,

      //feature compression
      compress_feature=
        true,
      compressed_size=
        2,
      pca_learning_rate=
        0.15,
  }
#endif


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
static Mat image_to_mat(tFrameInfo* pF, bool copy)
{
    IplImage* ipl;
    Mat mat;

    ipl = cvCreateImage(cvSize(pF->im.w,pF->im.h), IPL_DEPTH_8U, pF->im.c);
    ipl->widthStep = pF->widthStep;
    LOGV("width=%d\n", ipl->widthStep);
    ipl->imageData = (char*)pF->im.data;
    mat = cvarrToMat(ipl, copy);
    cvReleaseImage(&ipl);
    return mat;
}

void trackOpenCVAlgo(tAnnInfo* apBoundingBoxesIn, Mat imgBaseM, 
    tFrameInfo* pFTarg, Mat imgTargM,
    tTrackerBBInfo* pTrackerBBs,
    String trackingAlg, int idxIn, tAnnInfo* pBB,
    tAnnInfo** ppOutBBs)
{
#if 0
    int idxIn = 0;
    tAnnInfo* pBB;
        pBB = apBoundingBoxesIn;
#endif
#ifdef USE_CV_TRACKING
        bool ret;
        tAnnInfo* pTrackerOutBBs = *ppOutBBs;
        //while(pBB)
        {
            tAnnInfo* pBBTmp;

#if 0
            if(pBB->nMotionLevel > 40)
            {
                trackingAlg = "BOOSTING";
            }
            else
            {
                    pBBTmp = &pTrackerBBs[idxIn].trackerBB;
                    memcpy(pBBTmp, pBB, sizeof(tAnnInfo));
                    pBBTmp->pcClassName = (char*)malloc(strlen(pBB->pcClassName) + 1);
                    strcpy(pBBTmp->pcClassName, pBB->pcClassName);
                    pBBTmp->fCurrentFrameTimeStamp = pFTarg->fCurrentFrameTimeStamp;
                    pBBTmp->pNext = pTrackerOutBBs;
                    pBBTmp->pTracker = (void*)NULL;
                    pTrackerOutBBs = pBBTmp;
                    pTrackerBBs[idxIn].trackerBB = *pBBTmp;

                    pTrackerBBs[idxIn].pBBTOrig = pBB;
                    pBB = pBB->pNext;
                    idxIn++;
                    LOGV("no movement in this object\n");
                    continue;
            }
#endif


            Ptr<Tracker>* pTracker = NULL;
            if(!pBB->pTracker)
            {
                Ptr<Tracker> tracker = createTrackerByName(trackingAlg);
                pTracker = new Ptr<Tracker>(tracker);
                LOGV("DEBUGME\n");
                ret = (*pTracker)->init(imgBaseM, Rect2d(pBB->x, pBB->y, pBB->w, pBB->h));
                LOGV("initialized with %d %d %d %d ret=%d\n", pBB->x, pBB->y, pBB->w, pBB->h, ret);
            }
            else
            {
                LOGV("obj being tracked\n");
                pTracker = (Ptr<Tracker>*)(pBB->pTracker);
                ret = 1;
            }
            if(ret)
            {
                Rect2d object;
                //update with target
                for(int i = 0; i < MAX_TRACK_ITERATIONS; i++)
                {
                    ret = (*pTracker)->update(imgTargM, object);
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
                    pBBTmp->pTracker = (void*)pTracker;
                    *ppOutBBs = pTrackerOutBBs = pBBTmp;
                    pTrackerBBs[idxIn].trackerBB = *pBBTmp;
                    LOGV("stored %d %d %d %d ret=%d\n", pBBTmp->x, pBBTmp->y, pBBTmp->w, pBBTmp->h, ret);
#ifdef MULTI_ALGORITHMIC_APPROACH
                    pTrackerBBs[idxIn].opticalFlowBB = *pBBTmp;
#endif
                }
                else
                {
                    LOGD("unable to track this object in tracker\n");
                }
            }
            pTrackerBBs[idxIn].pBBTOrig = pBB;
#if 0
            pBB = pBB->pNext;
            idxIn++;
#endif
        }
#endif

}

void trackOpticalFlow(tAnnInfo* apBoundingBoxesIn, Mat imgBaseM, 
    tFrameInfo* pFTarg, Mat imgTargM,
    tTrackerBBInfo* pTrackerBBs, int idxIn, tAnnInfo* pBB,
    tAnnInfo** ppOutBBs, tGrid* gridMotion)
{
#if 0
    int idxIn = 0;
    tAnnInfo* pBB;
        pBB = apBoundingBoxesIn;
#endif
#ifdef OPTICAL_FLOW
        tAnnInfo* pOpticalFlowOutBBs = *ppOutBBs;
        TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,20,0.03);
        //while(pBB)
        {
            tAnnInfo* pBBTmp;
            vector<uchar> status;
            vector<float> err;
            vector<Point2f> points[2];
            points[0].push_back(Point2f((float)(pBB->x) + ((float)pBB->w)/2, (float)(pBB->y) + (float)(pBB->h)/2));


            if(pBB->nMotionLevel > 0)
            {
            Size subPixWinSize((float)pBB->w/2,(float)pBB->h/2);
            Mat gray;
            cvtColor(imgBaseM, gray, COLOR_BGR2GRAY);
            //cornerSubPix(gray, points[0], subPixWinSize, Size(-1,-1), termcrit);
            //Size winSize(1920,1080);
            int w = 0, h = 0;
            getWindowSize(gridMotion, points[0][0].x, points[0][0].y, w, h);
            LOGV("w=%d h=%d\n", w, h);
#if 0
            if(w > OPT_FLOW_WINSIZE_W && h > OPT_FLOW_WINSIZE_H
              && w < OPT_FLOW_WINSIZE_W_MAX && h < OPT_FLOW_WINSIZE_H_MAX
            )
            {

            }
            else
#endif
            {
                w = OPT_FLOW_WINSIZE_W;
                h = OPT_FLOW_WINSIZE_H;
            }
            Size winSize(w, h);
#if 0
            std::vector<Mat> pyramid;
            buildOpticalFlowPyramid(imgBaseM, pyramid, winSize, 2, true);
#endif
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
            }
            else
            {
                status.push_back(1);
                points[1].push_back(Point2f(points[0][0]));
            }
            if( status[0] )
            {
            pBBTmp = &pTrackerBBs[idxIn].trackerBB;
            memcpy(pBBTmp, pBB, sizeof(tAnnInfo));
#if 1
            pBBTmp->x = (int)(points[1][0].x - (((float)pBB->w)/2));
            pBBTmp->y = (int)(points[1][0].y - (((float)pBB->h)/2));
#else
            pBBTmp->x = (int)(points[1][0].x - ((pBB->w * 1.0)/2));
            pBBTmp->y = (int)(points[1][0].y - ((pBB->h * 1.0)/2));
#endif
            pBBTmp->pcClassName = (char*)malloc(strlen(pBB->pcClassName) + 1);
            strcpy(pBBTmp->pcClassName, pBB->pcClassName);
            pBBTmp->fCurrentFrameTimeStamp = pFTarg->fCurrentFrameTimeStamp;
            pBBTmp->pNext = pOpticalFlowOutBBs;
            *ppOutBBs = pOpticalFlowOutBBs = pBBTmp;
#ifdef MULTI_ALGORITHMIC_APPROACH
            pTrackerBBs[idxIn].opticalFlowBB = *pBBTmp;
#endif
            }
            pTrackerBBs[idxIn].pBBTOrig = pBB;
            idxIn++;
            pBB = pBB->pNext;
        }
#endif
}

int track_bb_in_frame(tAnnInfo* apBoundingBoxesIn, tFrameInfo* pFBase, tFrameInfo* pFTarg, tAnnInfo** appBoundingBoxesInOut, tLanesInfo* pLanesInfo)
{
    int ret = 0;
    Mat imgBaseM;
    Mat imgTargM;
    Mat imgBaseMC;
    Mat imgTargMC;
    tAnnInfo* pBB;
    tTrackerBBInfo* pTrackerBBs = NULL;
    int nInBBs = 0;
    tGrid gridMotion = {0};
    
    LOGV("DEBUGME\n");
    if(!apBoundingBoxesIn || !appBoundingBoxesInOut || !pFBase || !pFTarg)
    {
        return ret;
    }

    LOGV("DEBUGME w=%d h=%d pFBase->im.c=%d\n", pFBase->im.w, pFBase->im.h, pFBase->im.c);
    imgBaseMC = image_to_mat(pFBase, false);
    imgTargMC = image_to_mat(pFTarg, false);

    imgBaseM = imgBaseMC.clone();
    imgTargM = imgTargMC.clone();

    sca5(imgBaseMC, imgTargMC, &gridMotion);

    pBB = apBoundingBoxesIn;
    while(pBB)
    {
        nInBBs++;
        double nMotionLevel = getMotionLevel(&gridMotion, pBB->x, pBB->y, pBB->w, pBB->h);
        LOGV("nMotionLevel = %f\n", nMotionLevel);
        pBB->nMotionLevel = nMotionLevel;
        pBB = pBB->pNext;
    }
    pBB = *appBoundingBoxesInOut;
    while(pBB)
    {
        double nMotionLevel = getMotionLevel(&gridMotion, pBB->x, pBB->y, pBB->w, pBB->h);
        LOGV("nMotionLevel = %f\n", nMotionLevel);
        pBB->nMotionLevel = nMotionLevel;
        pBB = pBB->pNext;
    }

    pTrackerBBs = (tTrackerBBInfo*)calloc(nInBBs, sizeof(tTrackerBBInfo));


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
    char buffer [500];

    // set the default tracking algorithm
    String trackingAlg = TRACKING_ALGO;

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
        timer=clock()-timer;
        fps=(double)CLOCKS_PER_SEC/(double)timer;
  #endif

  
#if 1
#if 1
        tAnnInfo* pOutBBs = NULL;
        pBB = apBoundingBoxesIn;
        int idxIn = 0;
        while(pBB)
        {
            if(
#ifdef MULTI_ALGORITHMIC_APPROACH
                pBB->nMotionLevel < 150.0
#else
                1
#endif
              )
            {
                {
                    trackOpticalFlow(apBoundingBoxesIn, imgBaseM, pFTarg, imgTargM, pTrackerBBs, idxIn++, pBB, &pOutBBs, &gridMotion);
                }
                pLanesInfo->gnOptF++;
            }
            else
            {
                LOGV("lot of motion %d motion=%f\n", pBB->nBBId, pBB->nMotionLevel);
                trackOpenCVAlgo(apBoundingBoxesIn, imgTargM, pFTarg, imgTargM, pTrackerBBs, trackingAlg, idxIn++, pBB, &pOutBBs);
            }
            pBB = pBB->pNext;
            pLanesInfo->gnTot++;
        }
#endif
        /** optical flow */
        

        /** other trackers */

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
                    appBoundingBoxesInOut,
                    pLanesInfo);
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

#ifdef DISPLAY_LANES
        if(pLanesInfo)
        {
                LOGV("DEBUGME\n");
            tLane* pL = pLanesInfo->pLanes;
                LOGV("DEBUGME\n");
            while(pL)
            {
                LOGV("DEBUGME\n");
                memset(buffer, 0 , sizeof(buffer));
                LOGV("DEBUGME %p\n", pL);
                LOGV("%d %f %lld\n", pL->nLaneId, pL->fAvgStayDuration, pL->nTotalVehiclesSoFar);
                snprintf(buffer, 499, "lane:%d; (%f,%lld)", pL->nLaneId, pL->fAvgStayDuration, pL->nTotalVehiclesSoFar);
                LOGV("DEBUGME\n");
                text = buffer;
                putText(imgTargM, buffer, Point(pL->pVs->x, pL->pVs->y), FONT_HERSHEY_PLAIN, 1, Scalar(0, 0, 0));
                
                tVertex* p1=pL->pVs;
                tVertex* p2=p1->pNext;

                for(int i=1 ; i < pL->nVs ; i++)
                {
                         line(imgTargM,Point(p1->x, p1->y),Point(p2->x, p2->y),Scalar(0,0,0));
                         p1 = p1->pNext;
                         p2=p1->pNext;
              
                } 
                p2 = pL->pVs;
                line(imgTargM,Point(p1->x, p1->y),Point(p2->x, p2->y),Scalar(0,0,0));


                LOGV("DEBUGME\n");
                pL = pL->pNext;
            }
                LOGV("DEBUGME\n");
        }
#endif /**< DISPLAY_LANES */
  
        #ifdef DUMP_OP_IMAGE
        display_results(imgBaseM, apBoundingBoxesIn);
        display_results(imgTargM, pTrackerOutBBs);
        vector<int> compression_params;
        compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);
        char file_name[1024];
        snprintf(file_name, 1024, "mkdir -p output/" DUMP_OP_PATH);
        system(file_name);
        static int img_pairs_processed = 0;
        snprintf(file_name, 1024, "output/" DUMP_OP_PATH "output-b%04d.png", img_pairs_processed);
        ret = imwrite(file_name, imgBaseM, compression_params);
        snprintf(file_name, 1024, "output/" DUMP_OP_PATH "output-t%04d.png", img_pairs_processed);
        ret = imwrite(file_name, imgTargM, compression_params);
        printf("ret=%d\n", ret);
        if(++img_pairs_processed == 2)
            exit(0);
        #endif
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

    freeGrid(&gridMotion);

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
    imgBaseM = image_to_mat(pFBase, false);

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
    return 1;
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

    LOGV("%f %f\n", bb1_distance_from_best_iou, bb2_distance_from_best_iou);
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
                tAnnInfo** ppDetectedBBs,
                tLanesInfo* pLanesInfo)
{
    tAnnInfo* pTrackedBBs = NULL;
    if(!pTrackerBBs || !ppDetectedBBs)
        return;

    tAnnInfo* pDetectedBBs = *ppDetectedBBs;

    tAnnInfo* ppBBT[2];
    tAnnInfo* pBBT = NULL;
    tAnnInfo* pBBD = pDetectedBBs;
    tAnnInfo* pOutBBs = NULL;
    tAnnInfo* pBBDPrev = NULL;
    while(pBBD)
    {
        if(pBBD->nMotionLevel ==  0.0)
        {
            LOGV("pBBD->nMotionLevel = %f\n", pBBD->nMotionLevel);
        }
        pBBD->fIoU = 0;
        pBBD->bBBIDAssigned = 0;
        pBBD = pBBD->pNext;
    }

#ifdef MULTI_ALGORITHMIC_APPROACH
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
                    {
#ifdef FORCE_OFF_OPTICAL_FLOW_RESULTS
                        ppBBT[k]->fIoU = 0;
#endif
#if 0
                        AnnInfo tmpT = *ppBBT[k];
                        tmpT.x -= (tmpT.w/2);
                        tmpT.y -= (tmpT.h/2);
                        ppBBT[k]->fIoU = find_iou(&tmpT, pBBD);
                        /** if this BB did not move much from previous, use median flow */
                        tAnnInfo* pBBTmp = getBBById(pCopyDetectedBBs, ppBBT[k]->nBBId);
                        double disp = 0;
                        if(pBBTmp && ((disp = displacement_btw_BBs(&tmpT, pBBTmp)) == 0.0)) /**< use median flow */
                        {
                            continue;
                        }
                        LOGV("displacement (%p) is %f %d\n", pBBTmp, disp,  pBBTmp ? MAX(pBBTmp->w, pBBTmp->h) : 0);
#endif
#if 1
                        if(
#if 0
                           (ppBBT[k]->x < pBBD->x + pBBD->w + MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW) 
                           && (ppBBT[k]->y  < pBBD->y + pBBD->h + MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW) 
                           && (ppBBT[k]->x > (pBBD->x > MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW ? pBBD->x - MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW : 0)) 
                           && (ppBBT[k]->y > (pBBD->y > MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW ? pBBD->y - MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW : 0))
#endif
                          isWithinBB_UseCenter(ppBBT[k], pBBD))
#else
                        if((ppBBT[k]->x < pBBD->x + ((pBBD->w * 3.0) / 4) + MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW) 
                           && (ppBBT[k]->y  < pBBD->y + ((pBBD->h * 3.0) / 4)  + MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW) 
                           && (ppBBT[k]->x > (pBBD->x > MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW ? (pBBD->x + ((pBBD->w * 1.0) / 4)) - MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW : 0)) 
                           && (ppBBT[k]->y > (pBBD->y > MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW ? (pBBD->y + ((pBBD->h * 1.0) / 4) ) - MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW : 0))
                          )
#endif
                        {
                            ppBBT[k]->fIoU = 1.0; /**< optical flow alone can have this way */
                        }
                    }
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
                        if(return_best_iou(pBBDCandidate->fIoU, pBBDC->fIoU) == pBBDCandidate->fIoU)
                        {
                            break;
                        }
                        pBBDCPrev = pBBDC;
                        pBBDC = pBBDC->pNext;
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
            pTrackerBBs[i].bInDetectionList = 1;
            //pBBD->nBBId = pTrackerBBs[i].opticalFlowBB.nBBId;
            pBBD->B = pTrackerBBs[i].trackerBB.B;
            pBBD->G = pTrackerBBs[i].trackerBB.G;
            pBBD->R = pTrackerBBs[i].trackerBB.R;
            pBBD->nBBId = pTrackerBBs[i].trackerBB.nBBId;
            pBBD->nMotionLevel = pTrackerBBs[i].trackerBB.nMotionLevel;
            pBBD->pTracker = pTrackerBBs[i].trackerBB.pTracker;
            pBBD->bBBIDAssigned = 1;
            LOGV("new BBId=%d\n", pTrackerBBs[i].trackerBB.nBBId);
#ifdef SEE_TRACKING_ONLY
            pTrackerBBs[i].pBBT = copyBB(pBBD);
#endif
            pBBD->fDirection = pBBD->x - pTrackerBBs[i].pBBTOrig->x > 0 ? 'L'  : 'R';
        }


#ifdef SEE_TRACKING_ONLY
        if(pTrackerBBs[i].bInDetectionList == 1)
        {
            pTrackerBBs[i].pBBT->pNext = pTrackedBBs;
            pTrackedBBs = pTrackerBBs[i].pBBT;
        }
#endif
        
    }
#else

#if 0
    pBBD = pDetectedBBs;
    while(pBBD)
    {
        tAnnInfo* pBBTBestIoU = &pTrackerBBs[0].trackerBB;
        for(int i = 0; i < nTrackerInSlots; i++)
        {
            if(!pTrackerBBs[i].bInDetectionList)
            {
                /** if not precise, the best approximation of
                 * vehicle flux is what traffic engineers seek
                 * [speaker {AS} @ NVIDIA AI City Challenge] */
                if(isWithinBB(&pTrackerBBs[i].trackerBB, pBBD))
#if 0
                pTrackerBBs[i].trackerBB.fIoU = find_iou(&pTrackerBBs[i].trackerBB, pBBD);
                LOGV("%d:%d: IoU=%f\n", pBBD->nBBId, pTrackerBBs[i].trackerBB.nBBId, pTrackerBBs[i].trackerBB.fIoU);
                if(pTrackerBBs[i].trackerBB.fIoU > GOOD_IOU_THRESHOLD)
#endif
                {
                    LOGV("found in final parse\n");
                    pBBD->nBBId = pTrackerBBs[i].trackerBB.nBBId;
                    pBBD->B = pTrackerBBs[i].trackerBB.B;
                    pBBD->G = pTrackerBBs[i].trackerBB.G;
                    pBBD->R = pTrackerBBs[i].trackerBB.R;
                    pBBD->nMotionLevel = pTrackerBBs[i].trackerBB.nMotionLevel;
                    pBBD->pTracker = pTrackerBBs[i].trackerBB.pTracker;
                    pTrackerBBs[i].bInDetectionList = pBBD->bBBIDAssigned = 1;
                    /** */
                    break;
                }
            }
        }
        pBBD = pBBD->pNext;
    }
#endif
#if 1
    pBBD = pDetectedBBs;
    while(pBBD)
    {
        tAnnInfo* pBBTBestIoU = NULL;
        LOGV("finding tracked op for BB-ID(new)=%d \n", pBBD->nBBId);
        int nBestIoUIdx = -1;
        for(int i = 0; i < nTrackerInSlots; i++)
        {
            if(!pTrackerBBs[i].bInDetectionList)
            if(isWithinBB_UseCenter(&pTrackerBBs[i].trackerBB, pBBD))
            {
                pTrackerBBs[i].trackerBB.fIoU = find_iou(&pTrackerBBs[i].trackerBB, pBBD);
                /** if not precise, the best approximation of
                 * vehicle flux is what traffic engineers seek
                 * [speaker {AS} @ NVIDIA AI City Challenge] */
                pTrackerBBs[i].trackerBB.fIoU = find_iou(&pTrackerBBs[i].trackerBB, pBBD);
                if(!pBBTBestIoU || (return_BB_with_best_iou(pBBTBestIoU, &pTrackerBBs[i].trackerBB) != pBBTBestIoU))
                {
                    pBBTBestIoU = &pTrackerBBs[i].trackerBB;
                    nBestIoUIdx = i;
                }
                LOGV("%d:%d: IoU=%f\n", pBBD->nBBId, pTrackerBBs[i].trackerBB.nBBId, pTrackerBBs[i].trackerBB.fIoU);
            }
        }
                if(pBBTBestIoU)
                {
                    LOGV("BestIoU in th elist = %f nBBId=%d \n", pBBTBestIoU->fIoU, pBBTBestIoU->nBBId);
                }
                if(pBBTBestIoU && (return_best_iou(pBBTBestIoU->fIoU, 0.3) != 0.3))
                {
                    LOGV("found in final parse\n");
                    pBBD->nBBId = pBBTBestIoU->nBBId;
                    pBBD->B = pBBTBestIoU->B;
                    pBBD->G = pBBTBestIoU->G;
                    pBBD->R = pBBTBestIoU->R;
                    pBBD->nMotionLevel = pBBTBestIoU->nMotionLevel;
                    pBBD->pTracker = pBBTBestIoU->pTracker;
                    pTrackerBBs[nBestIoUIdx].bInDetectionList = pBBD->bBBIDAssigned = 1;
                    /** */
                }
        pBBD = pBBD->pNext;
    }
#endif

#endif

/** keep tracking an object when detection fail on it;
 * but tracking is successfull
 * NOTE:
 * We shouldnt track it if any of the detected bounding boxes
 * overlap this one
 */
#if 1
    for(int i = 0; i < nTrackerInSlots; i++)
    {
        if(pTrackerBBs[i].bInDetectionList == 0)
        {
            LOGV("not detected!!!!\n");
            tAnnInfo* pBBTmp1 = NULL;
            if(0)//pTrackerBBs[i].opticalFlowBB.pcClassName)
            {
                pBBTmp1 = &pTrackerBBs[i].opticalFlowBB;
                pBBTmp1->x -= pBBTmp1->w/2;
                pBBTmp1->y -= pBBTmp1->h/2;
            }
#if 1
            else if(pTrackerBBs[i].trackerBB.pcClassName)
            {
                pBBTmp1 = &pTrackerBBs[i].trackerBB;
            }
#endif
            if(pBBTmp1)
            {
                if(check_if_new_BB_acceptable(pBBTmp1, pCopyDetectedBBs, pDetectedBBs))
                {
                tAnnInfo* pBBTmp;

                
                pBBTmp = (tAnnInfo*)malloc(sizeof(tAnnInfo));
                memcpy(pBBTmp, pBBTmp1, sizeof(tAnnInfo));
                pBBTmp->pcClassName = (char*)malloc(strlen(pBBTmp1->pcClassName) + 1);
                strcpy(pBBTmp->pcClassName, pBBTmp1->pcClassName);
#ifndef SEE_TRACKING_ONLY
                pBBTmp->pNext = pDetectedBBs;
                pDetectedBBs = pBBTmp;
#else
                pTrackerBBs[i].pBBT = pBBTmp;
#endif
                pTrackerBBs[i].bInDetectionList = 1;
                pBBD = pBBTmp;
                pBBD->fDirection = pBBD->x - pTrackerBBs[i].pBBTOrig->x > 0 ? 'L'  : 'R';
                }
            }
        }
     }
#endif

    {
        for(int i = 0; i < nTrackerInSlots; i++)
        {
            if(pTrackerBBs[i].bInDetectionList == 0)
            {
                LOGD("BBID not assigned %p\n", pTrackerBBs[i].trackerBB.pTracker);
                Ptr<Tracker>* pTracker = (Ptr<Tracker>*)pTrackerBBs[i].trackerBB.pTracker;
                
                if(pTracker)
                {
                    LOGD("deleted ...........\n\n\n\n\n\n\n\n\n\n");
                    delete pTracker;
                    pTrackerBBs[i].trackerBB.pTracker = NULL;
                }
            }
        }
    }

#if 0
    {
        for(int i = 0; i < nTrackerInSlots; i++)
        {
            pBBT = copyBB(&pTrackerBBs[i].trackerBB);
            pBBT->pNext = pTrackedBBs;
            pTrackedBBs = pBBT;
        }
    }
#endif


#ifdef SEE_TRACKING_ONLY
    free_BBs(pDetectedBBs);
    *ppDetectedBBs = pTrackedBBs;
#else
    *ppDetectedBBs = pDetectedBBs;
#endif

    collect_analysis(*ppDetectedBBs, pCopyDetectedBBs, pLanesInfo);

    free_BBs(pCopyDetectedBBs);
    pCopyDetectedBBs = make_copy_of_current_set(*ppDetectedBBs);


    

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
#if 0
        LOGV("nBBId=%d\n", pBB->nBBId);
        if(pBB->nBBId != 3360)//129, 633-->8.mp4
        {
            pBB = pBB->pNext;
            continue;
        }
#endif
        
        rectangle(imgTargM, Rect2d((double)pBB->x, (double)pBB->y, (double)pBB->w, (double)pBB->h), Scalar( pBB->B, pBB->G, pBB->R ), 2, 1 );
        char disp[50] = {0};
        LOGV("disp: [%d(%f)(%d, %d):%s %d]\n", pBB->nBBId, pBB->nMotionLevel, pBB->x, pBB->y, pBB->pcClassName,pBB->nLaneId);
        snprintf(disp, 50-1, "[%d(%f)(%d, %d):%s %d]", pBB->nBBId, pBB->nMotionLevel, pBB->x, pBB->y, pBB->pcClassName,pBB->nLaneId);
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
        for(int k = i+1; k < nTrackerInSlots; k++)
        {
            #if 0
            if(k <= i)
                continue;
            #endif
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

tAnnInfo* make_copy_of_current_set(tAnnInfo* pDetectedBBs)
{
    tAnnInfo* pBBD = pDetectedBBs;
    tAnnInfo* pCDetectedBBs = NULL;

    tAnnInfo* pBBTmp;

    while(pBBD)
    {
        pBBTmp = copyBB(pBBD);
        pBBTmp->pNext = pCDetectedBBs;
        pCDetectedBBs = pBBTmp;
        pBBD = pBBD->pNext;
    }

    return pCDetectedBBs;
}

// This function calculates the angle of the line from A to B with respect to the positive X-axis in degrees
int angle(Point2f A, Point2f B) {
	int val = (B.y-A.y)/(B.x-A.x); // calculate slope between the two points
	val = val - pow(val,3)/3 + pow(val,5)/5; // find arc tan of the slope using taylor series approximation
	val = ((int)(val*180/3.14)) % 360; // Convert the angle in radians to degrees
	if(B.x < A.x) val+=180;
	if(val < 0) val = 360 + val;
	return val;
}

int getDirection(tAnnInfo* pBBFrom, tAnnInfo* pBBTo)
{
        Point2f A(pBBFrom->x, pBBFrom->y);    
        Point2f B(pBBTo->x, pBBTo->y);
        return angle(A, B);
}

int check_if_new_BB_acceptable(tAnnInfo* pBBIn, tAnnInfo* apCopyDetectedBBs, tAnnInfo* pDetectedBBs)
{
    tAnnInfo* pBBD = apCopyDetectedBBs;
    tAnnInfo BBD = *pBBIn;
    tAnnInfo BBD_TL = *pBBIn;

    tAnnInfo BBD_TR = *pBBIn;
    BBD_TR.x = BBD_TR.x + BBD_TR.w;

    tAnnInfo BBD_BL = *pBBIn;
    BBD_BL.y = BBD_BL.y + BBD_BL.h;

    tAnnInfo BBD_BR = *pBBIn;
    BBD_BR.x += BBD_BR.w;
    BBD_BR.y += BBD_BR.h;

    BBD.x = BBD.x + BBD.w/2;
    BBD.y = BBD.y + BBD.h/2;
    double disp;


    bool bAcceptable = false;

#if 1
    while(pBBD)
    {
        if(pBBD->nBBId == BBD.nBBId)
        {
            disp = displacement_btw_BBs(pBBD, &BBD);
            LOGV("disp is %d %f %s %d %f\n", pBBD->nBBId, disp, BBD.pcClassName, pBBD->nBBId, pBBD->nMotionLevel);
            BBD.fDirection = BBD.x - pBBD->x > 0 ? 'L'  : 'R';
            if(1
                //disp >= 1.0 
                //&& BBD.fDirection == pBBD->fDirection
                )
            {
                bAcceptable = true;
                break;
            }
        }
        pBBD = pBBD->pNext;
    }
#endif


#if 1
    /** check if BBD is overlapped by any detected BBs and if they're unmapped, map them */
    pBBD = pDetectedBBs;
    while(pBBD && bAcceptable)
    {
        double c = 0;
        
#if 0
        if(isWithinBB(&BBD, pBBD) /** center */)
            c++;
        if(isWithinBB(&BBD_TL, pBBD))
            c++;
        if(isWithinBB(&BBD_TR, pBBD))
            c++;
        if(isWithinBB(&BBD_BL, pBBD))
            c++;
        if(isWithinBB(&BBD_BR, pBBD))
            c++;

        if(c > 1)
#else
        c = find_iou(&BBD_TL, pBBD);

        LOGV("c=%f pBBD->bBBIDAssigned=%d\n", c, pBBD->bBBIDAssigned);

        if((c = return_best_iou(c, 0.5)) != 0.5)
#endif
        {
            LOGV("c=%f\n", c);
            bAcceptable = false;
            pBBD->bBBIDAssigned = 1;
            pBBD->nBBId = BBD_TL.nBBId;
            pBBIn->bBBIDAssigned = 1;
            break;
        }
        pBBD = pBBD->pNext;
    }
#endif

    bAcceptable = false;

    LOGV("bAcceptable=%d\n", bAcceptable);

    return bAcceptable;
}

class Lanes
{
     public:
     tLanesInfo* pLanesInfo;
     vector<Point> vertices;
     Mat imgBaseM;
     Mat ROI;
     sem_t sem;
     bool finished;
     int count;

     public:
     Lanes(Mat& img, Mat& roi, tLanesInfo* apLanesInfo)
     {
        imgBaseM = img;
        ROI = roi;
        sem_init(&sem, 0, 0);
        finished = 0;
        if(apLanesInfo)
        {
            pLanesInfo = apLanesInfo;
            count = pLanesInfo->nLanes;
        }
        else
        {
            pLanesInfo = (tLanesInfo*)calloc(1, sizeof(tLanesInfo));
            count=0;
        }
     }
};

void mouseHandler(int event,int x,int y,int flags,void* userdata)
{
   Lanes* poly = (Lanes*)userdata;
#if 1
   if(event==EVENT_RBUTTONDOWN){
      cout << "Right mouse button clicked at (" << x << ", " << y << ")" << endl;
      if(poly->vertices.size()<2){
         cout << "You need a minimum of three points!" << endl;
         return;
      }
      // Close polygon
      line(poly->imgBaseM,poly->vertices[poly->vertices.size()-1],poly->vertices[0],Scalar(0,0,0));

      tLane* polygon = (tLane*)calloc(1, sizeof(tLane));
      for(int i = 0; i < poly->vertices.size(); i++)
      {
          tVertex* pV = (tVertex*)calloc(1, sizeof(tVertex));
          pV->x = poly->vertices[i].x;
          pV->y = poly->vertices[i].y;
          pV->pNext = polygon->pVs;
          polygon->pVs = pV;
          polygon->nVs++;
      }
      polygon->nLaneId = ++poly->count;
      polygon->pNext = poly->pLanesInfo->pLanes;
      poly->pLanesInfo->pLanes = polygon;
      poly->pLanesInfo->nLanes++;
      poly->vertices.clear();

      // Mask is black with white where our ROI is
      //Mat mask= Mat::zeros(poly->imgBaseM.rows,poly->imgBaseM.cols,CV_8UC1);
      //vector<vector<Point>> pts{poly->vertices};
      LOGV("DEBUGME\n");
      //fillPoly(mask,poly->vertices,Scalar(255,255,255));
      LOGV("DEBUGME\n");
      //poly->imgBaseM.copyTo(poly->ROI,mask);
      LOGV("DEBUGME\n");
      //finished=true;

      return;
   }
   if(event==EVENT_LBUTTONDOWN){
      cout << "Left mouse button clicked at (" << x << ", " << y << ")" << endl;
      if(poly->vertices.size()==0){
         // First click - just draw point
         poly->imgBaseM.at<Vec3b>(x,y)=Vec3b(255,0,0);
      } else {
         // Second, or later click, draw line to previous vertex
         line(poly->imgBaseM,Point(x,y),poly->vertices[poly->vertices.size()-1],Scalar(0,0,0));
      }
      poly->vertices.push_back(Point(x,y));
      return;
   }

   if(event==EVENT_MBUTTONDOWN)
   {
     LOGV("LButton double clicked\n");
     poly->finished = true;
     return;
   }
#endif
}


tLanesInfo* getLaneInfo(tFrameInfo* pFrame, tLanesInfo* pLanesInfo)
{
    Mat imgBaseM = image_to_mat(pFrame, true);
    Mat roi;
    Lanes polygons(imgBaseM, roi, pLanesInfo);
    tLanesInfo* pPI = NULL;
    const char* winName = "draw lanes";
    cv::imshow(winName, imgBaseM);
    cv::setMouseCallback(winName, mouseHandler, &polygons);
    while(!polygons.finished)
    {
       cv::imshow(winName, imgBaseM);
       waitKey(50);     
    }
    
    pPI = polygons.pLanesInfo;
    //cvReleaseImage();
    destroyWindow(winName);
    return pPI;
}

// Define Infinite (Using INT_MAX caused overflow problems)
#define INF 10000

// Given three colinear points p, q, r, the function checks if
// point q lies on line segment 'pr'
bool onSegment(tVertex p, tVertex q, tVertex r)
{
    if (q.x <= max(p.x, r.x) && q.x >= min(p.x, r.x) &&
            q.y <= max(p.y, r.y) && q.y >= min(p.y, r.y))
        return true;
    return false;
}
 
// To find orientation of ordered triplet (p, q, r).
// The function returns following values
// 0 --> p, q and r are colinear
// 1 --> Clockwise
// 2 --> Counterclockwise
int orientation(tVertex p, tVertex q, tVertex r)
{
    int val = (q.y - p.y) * (r.x - q.x) -
              (q.x - p.x) * (r.y - q.y);
 
    if (val == 0) return 0;  // colinear
    return (val > 0)? 1: 2; // clock or counterclock wise
}
 
// The function that returns true if line segment 'p1q1'
// and 'p2q2' intersect.
bool doIntersect(tVertex p1, tVertex q1, tVertex p2, tVertex q2)
{
    // Find the four orientations needed for general and
    // special cases
    int o1 = orientation(p1, q1, p2);
    int o2 = orientation(p1, q1, q2);
    int o3 = orientation(p2, q2, p1);
    int o4 = orientation(p2, q2, q1);
 
    // General case
    if (o1 != o2 && o3 != o4)
        return true;
 
    // Special Cases
    // p1, q1 and p2 are colinear and p2 lies on segment p1q1
    if (o1 == 0 && onSegment(p1, p2, q1)) return true;
 
    // p1, q1 and p2 are colinear and q2 lies on segment p1q1
    if (o2 == 0 && onSegment(p1, q2, q1)) return true;
 
    // p2, q2 and p1 are colinear and p1 lies on segment p2q2
    if (o3 == 0 && onSegment(p2, p1, q2)) return true;
 
     // p2, q2 and q1 are colinear and q1 lies on segment p2q2
    if (o4 == 0 && onSegment(p2, q1, q2)) return true;
 
    return false; // Doesn't fall in any of the above cases
}

// Returns true if the point p lies inside the polygon[] with n vertices
bool isInside(tLane* pLane, tVertex p)
{
    // There must be at least 3 vertices in polygon[]
    if (!pLane || !pLane->pVs || (pLane->nVs < 3))  return false;

    int n = pLane->nVs;
    tVertex* pP = pLane->pVs;
 
    // Create a point for line segment from p to infinite
    tVertex extreme = {INF, p.y};
 
    // Count intersections of the above line with sides of polygon
    int count = 0;
    do
    {
        tVertex nextV = pP->pNext ? *(pP->pNext) : *(pLane->pVs);
        // Check if the line segment from 'p' to 'extreme' intersects
        // with the line segment from 'polygon[i]' to 'polygon[next]'
        if (doIntersect(*pP, nextV, p, extreme))
        {
            // If the point 'p' is colinear with line segment 'i-next',
            // then check if it lies on segment. If it lies, return true,
            // otherwise false
            if (orientation(*pP, p, nextV) == 0)
               return onSegment(*pP, p, nextV);
 
            count++;
        }
        pP = pP->pNext;
    } while (pP);
 
    // Return true if count is odd, false otherwise
    return count&1;  // Same as (count%2 == 1)
}

int isWithinLane(tLane* pLane, tVertex* pVertex)
{
    if(!pLane || !pVertex)
        return 0;
    return isInside(pLane, *pVertex);
}

int isWithinBB_UseCenter(tAnnInfo* pBBP, tAnnInfo* pBBD)
{
        int x = pBBP->x + ((float)pBBP->w)/2;
        int y = pBBP->y + ((float)pBBP->h)/2;
        if((x < pBBD->x + pBBD->w + MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW) 
                           && (y  < pBBD->y + pBBD->h + MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW) 
                           && (x > (pBBD->x > MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW ? pBBD->x - MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW : 0)) 
                           && (y > (pBBD->y > MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW ? pBBD->y - MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW : 0))
                          )
                return 1;
        return 0;
}

int isWithinBB(tAnnInfo* pBBP, tAnnInfo* pBBD)
{
        int x = pBBP->x;// + ((float)pBBP->w)/2;
        int y = pBBP->y;// + ((float)pBBP->h)/2;
        if((x < pBBD->x + pBBD->w + MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW) 
                           && (y  < pBBD->y + pBBD->h + MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW) 
                           && (x > (pBBD->x > MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW ? pBBD->x - MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW : 0)) 
                           && (y > (pBBD->y > MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW ? pBBD->y - MAX_BB_SIDE_LEN_TOLERANCE_OPT_FLOW : 0))
                          )
                return 1;
        return 0;
}

typedef struct
{
    int from;
    int to;
}tViolation;

tViolation gViolations[] = {{10, 3}, {2, 11}};

int isViolation(int from, int to)
{
    for(int i = 0; i < sizeof(gViolations); i++)
    {
        if(from == gViolations[i].from
           && to == gViolations[i].to)
            return 1;
    }
    return 0;
}


void collect_analysis(tAnnInfo* pCurrFrameBBs, tAnnInfo* pPrevFrameBBs, tLanesInfo* pLanesInfo)
{
    tAnnInfo* pBBNode = NULL;
    tAnnInfo* pCurrBB = NULL;
    if(!pCurrFrameBBs || !pPrevFrameBBs || !pLanesInfo)
        return;

    pBBNode = pPrevFrameBBs;
    while(pBBNode)
    {

        if((pCurrBB = getBBById(pCurrFrameBBs, pBBNode->nBBId)))
        {
            /** an object is tracked in the curr frame;
             * avg-wait-time
             */
            /** assign prev BB's basic detail here - if its not already populated */
            tLane* pLPrev = NULL;
            pLPrev = laneWithThisBB(pLanesInfo, pBBNode);
            if(pBBNode->nLaneId == INVALID_LANE_ID)
            {
                pBBNode->fStartTS = pBBNode->fCurrentFrameTimeStamp;
                pBBNode->nLaneId = pLPrev ? pLPrev->nLaneId : INVALID_LANE_ID;
            }
            tLane* pLCurr = laneWithThisBB(pLanesInfo, pCurrBB);
            {
                pCurrBB->nLaneId = pLCurr ? pLCurr->nLaneId : INVALID_LANE_ID;
                pCurrBB->nLaneHistory = pBBNode->nLaneHistory;
                if(pBBNode->nLaneId != pCurrBB->nLaneId)
                {
                    pCurrBB->nLaneHistory = pBBNode->nLaneId;
                    if(pCurrBB->nLaneId != INVALID_LANE_ID
                       && pBBNode->nLaneHistory != INVALID_LANE_ID)
                    {
                        LOGV("we have route flux from lane %d to %d %d\n", pBBNode->nLaneHistory, pCurrBB->nLaneId, pCurrBB->nClassId);
                        LOGV("deref ppRouteTrafficInfo[]=%p\n", pLanesInfo->ppRouteTrafficInfo[pBBNode->nLaneHistory]);
                        LOGV("deref ppRouteTrafficInfo[][]=%p\n", &pLanesInfo->ppRouteTrafficInfo[pBBNode->nLaneHistory][pCurrBB->nLaneId]);
                        LOGV("deref ppRouteTrafficInfo[][]=%p\n", pLanesInfo->ppRouteTrafficInfo[pBBNode->nLaneHistory][pCurrBB->nLaneId].pnVehicleCount);
                        LOGV("deref ppRouteTrafficInfo[][]=%lld\n", pLanesInfo->ppRouteTrafficInfo[pBBNode->nLaneHistory][pCurrBB->nLaneId].pnVehicleCount[pCurrBB->nClassId]);
                        (pLanesInfo->ppRouteTrafficInfo[pBBNode->nLaneHistory][pCurrBB->nLaneId].pnVehicleCount[pCurrBB->nClassId])++;
                        LOGV("DEBUGME\n");
                        if(isViolation(pBBNode->nLaneHistory, pCurrBB->nLaneId))
                        {
                            LOGV("violation @ %f; BBID=%d type=%s\n", pCurrBB->fCurrentFrameTimeStamp, pCurrBB->nBBId, pCurrBB->pcClassName);
                        }
                    }
                    if(pLPrev)
                        pLPrev->pnVehicleCount[pCurrBB->nClassId]++; 
                    /** object now exited the lane */
                    double fDurationOfStayInThisLane = pCurrBB->fCurrentFrameTimeStamp - pCurrBB->fStartTS;
                    if(pLPrev)
                    {
                        pLPrev->fTotalStayDuration += fDurationOfStayInThisLane;
                        pCurrBB->fStartTS = pCurrBB->fCurrentFrameTimeStamp;
                        pLPrev->nTotalVehiclesSoFar = 0;
                        for(int i = 0; i < pLPrev->nTypes; i++)
                            pLPrev->nTotalVehiclesSoFar += pLPrev->pnVehicleCount[i];
                        pLPrev->fAvgStayDuration = pLPrev->nTotalVehiclesSoFar ? pLPrev->fTotalStayDuration / (pLPrev->nTotalVehiclesSoFar) : 0;
                        LOGV("lane %d; fAvgStayDuration=%f nTotalVehiclesSoFar=%lld\n", pLPrev->nLaneId, pLPrev->fAvgStayDuration, pLPrev->nTotalVehiclesSoFar);
                    }
                }
                else
                {
                    pCurrBB->fStartTS = pBBNode->fStartTS;
                }
            }
        }
        else
        {
            /** the object went out of scene */
            /** counting vehicle types: */
            /** check which lane the vehicle belonged to, and increment corresponding count */
            tLane* pL;
            if(pBBNode->fStartTS /**< count this only if we tracked same obj over atleast 2 frames */
                && (pL = laneWithThisBB(pLanesInfo, pBBNode)))
            {
            }
            else
            {
                LOGV("we couldn't identify lane; \n");
            }
        }
        pBBNode = pBBNode->pNext;
    }

    return;
}

tLane* laneWithThisBB(tLanesInfo* pLanesInfo, tAnnInfo* pBBNode)
{
    tLane* pL;

    if(!pLanesInfo || !pBBNode)
        return NULL;

    pL = pLanesInfo->pLanes;
    while(pL)
    {
        tVertex nTmp = {0};
        nTmp.x = pBBNode->x;
        nTmp.y = pBBNode->y;
        nTmp.x += (pBBNode->w/2);
        nTmp.y += (int)((pBBNode->h * 3.0) / 4);
        if(isWithinLane(pL, &nTmp))
            return pL;
        pL = pL->pNext;
    }

    return NULL;
}
