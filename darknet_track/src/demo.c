#include "network.h"
#include "detection_layer.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "demo.h"
#include <sys/time.h>
#include "darknet_exp.h"
#include "multitracker.h"
#include "cJSON.h"
//#include <opencv2/opencv.hpp>

#define DEMO 1

/** { ISVA */
//#define ENABLE_VIDEO_FILE_READ_AT_TAR_FPS
//#define DISPLAY_RESULS

/** shall track BBs in the same frame */
//#define TEST_TRACKING

/** IMPURE_CNN is when darknet does not use object tracking and interpolation
 * to find bounding boxes on temporally adjacent frames in a video or 
 * realtime camera feed
 * This way, we could collect data with processed information using a camera deployed with an edge device 
 * like TX2
 * TX2 had only 256 GPU cores and was unable to give realtime performance with models like YOLO 
 * NOTE: works only when the input is a video or live camera feed
 */
#define IMPURE_CNN
#define GOOD_IOU_THRESHOLD (0.5)

//#define DEBUG
#define VERBOSE
#include "debug.h"

/** } ISVA */


#ifdef OPENCV

#define MAX_FRAMES_TO_HASH 2

#define ABS_DIFF(a, b) ((a) > (b)) ? ((a)-(b)) : ((b)-(a))

typedef struct Frame tFrame;
typedef struct
{
    char **demo_names;
    image **demo_alphabet;
    int demo_classes;

    network net;
    CvCapture * cap;
    float fps;
    double fStartTime;
    double fEndTime;
    float demo_thresh;
    float demo_hier;
    int running;

    int demo_delay;
    int demo_frame;
    int demo_detections;
    int demo_index;
    int demo_done;
    double demo_time;


    double nTargetFps;
    int nCurFrameCount;
    double nFps;
    int nSkipFramesCnt;
    int bProcessThisFrame;
    tDetectorModel* pDetectorModel;
    tFrame* pFrames;
    tFrame* pFramesHash[MAX_FRAMES_TO_HASH+1]; /**< we do 0-14 and 14-30 if MAX_FRAMES_TO_HASH=30, so a +1 */
    int nBBCount;
    double countFrame;
    double totalFramesInVid;
    tLanesInfo* pLanesInfo;
}tDetector;

struct Frame
{
    image buff;
    double buff_ts;
    image buff_letter;
    tDetector* pDetector;
    IplImage  * ipl;
    float **probs;
    box *boxes;
    int prod_lwn;
    tAnnInfo* pBBs;
    tFrameInfo frameInfoWithCpy;
    tFrame* pNext;
};

static void test_detector_on_img(tDetector* pDetector, char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen);
void *display_in_thread(void *ptr);
void free_frame(tDetector* pDetector, tFrame* apFrame);
void dump_lane_info(tLanesInfo* pLanesInfo);

void init_globals(tDetector* pDetector)
{
    pDetector->demo_names = NULL;
    pDetector->demo_alphabet = NULL;
    pDetector->demo_classes =  0;
    
    pDetector->cap = 0;
    pDetector->fps = 0;
    pDetector->demo_thresh = 0;
    pDetector->demo_hier = .5;
    pDetector->running = 0;
    
    pDetector->demo_delay = 0;
    pDetector->demo_frame = 3;
    pDetector->demo_detections = 0;
    pDetector->demo_done = 0;
    pDetector->demo_time = 0;
    
    
    pDetector->nTargetFps = 1;
    pDetector->nCurFrameCount = 0;
    pDetector->nFps = 0;
    pDetector->nSkipFramesCnt = 0;
    pDetector->bProcessThisFrame = 0;
    pDetector->pDetectorModel = NULL;
}

double get_wall_time()
{
    struct timeval time;
    if (gettimeofday(&time,NULL)){
        return 0;
    }
    return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

void evaluate_detections(tFrame* pFrame, image im, int num, float thresh, box *boxes, float **probs, char **names, image **alphabet, int classes)
{
    int i;
    tAnnInfo annInfo = {0};
    tDetector* pDetector = pFrame->pDetector;

    if(!pDetector || !pDetector->pDetectorModel)
        return;

    for(i = 0; i < num; ++i){
        int class_ = max_index(probs[i], classes);
        float prob = probs[i][class_];
        if(prob > thresh){

            int width = im.h * .006;

            if(0){
                width = pow(prob, 1./2.)*10+1;
                alphabet = 0;
            }

            //LOGD("%d %s: %.0f%%\n", i, names[class_], prob*100);
            LOGD("%s: %.0f%%\n", names[class_], prob*100);
            int offset = class_*123457 % classes;
            float red = get_color(2,offset,classes);
            float green = get_color(1,offset,classes);
            float blue = get_color(0,offset,classes);
            float rgb[3];

            //width = prob*20+2;

            rgb[0] = red;
            rgb[1] = green;
            rgb[2] = blue;
            box b = boxes[i];

            int w, h;
            w = cvGetCaptureProperty(pDetector->cap, CV_CAP_PROP_FRAME_WIDTH);
            h = cvGetCaptureProperty(pDetector->cap, CV_CAP_PROP_FRAME_HEIGHT);
            int left  = (b.x-b.w/2.)*w;
            int right = (b.x+b.w/2.)*w;
            int top   = (b.y-b.h/2.)*h;
            int bot   = (b.y+b.h/2.)*h;

            if(left < 0) left = 0;
            if(right > im.w-1) right = im.w-1;
            if(top < 0) top = 0;
            if(bot > im.h-1) bot = im.h-1;

#ifdef DISPLAY_RESULS
            draw_box_width(im, left, top, right, bot, width, red, green, blue);
            if (alphabet) {
                image label = get_label(alphabet, names[class_], (im.h*.03)/10);
                draw_label(im, top + width, left, label, rgb);
                free_image(label);
            }
#endif
            LOGV("box x:%f y:%f w:%f h:%f; l:%d r:%d t:%d b:%d\n", b.x, b.y, b.w, b.h, left, right, top, bot);
            annInfo.x = (int)(left);
            annInfo.y = (int)(top);
            annInfo.w = (int)(right - left);
            annInfo.h = (int)(bot - top);
            annInfo.pcClassName = (char*)calloc(1, strlen(names[class_]) + 1);
            annInfo.nClassId = class_;
            annInfo.nBBId = pDetector->nBBCount++; /**< the unique object ID assigned to the BB initially by detector; used in IMPURE_CNN mode */
            strcpy(annInfo.pcClassName, names[class_]);
            if(pDetector->pDetectorModel->isVideo)
                annInfo.fCurrentFrameTimeStamp = pFrame->buff_ts;
            else
                annInfo.fCurrentFrameTimeStamp = pDetector->pDetectorModel->nFrameId * 1000;
            annInfo.nVideoId = pDetector->pDetectorModel->nVideoId;
            annInfo.prob = prob;
            LOGD("hello..\n");
            LOGV("annInfo x=%d y=%d w=%d h=%d pcClassName=%s fCurrentFrameTimeStamp=%f\n",
                annInfo.x, annInfo.y, annInfo.w, annInfo.h, annInfo.pcClassName,
                annInfo.fCurrentFrameTimeStamp);
            #ifndef IMPURE_CNN
            if(pDetector->pDetectorModel && pDetector->pDetectorModel->pfnRaiseAnnCb)
                pDetector->pDetectorModel->pfnRaiseAnnCb(annInfo);
            #endif
            /** add BB to the linked list */
            tAnnInfo* pBB = (tAnnInfo*)calloc(1, sizeof(tAnnInfo));
            *pBB = annInfo;
            pBB->pNext = pFrame->pBBs;
            pFrame->pBBs = pBB;
        }
    }
}

void *detect_in_thread(void *ptr)
{
    tFrame* pFrame = (tFrame*)ptr;
    tDetector* pDetector = pFrame->pDetector;
    LOGD("DEBUGME\n");
    pDetector->running = 1;
    float nms = .4;

    layer l = pDetector->net.layers[pDetector->net.n-1];
    float *X = pFrame->buff_letter.data;
    LOGD("DEBUGME\n");
    float *prediction = network_predict(pDetector->net, X);
    LOGD("DEBUGME\n");

#if 0
    memcpy(pDetector->predictions[pDetector->demo_index], prediction, l.outputs*sizeof(float));
    mean_arrays(pDetector->predictions, pDetector->demo_frame, l.outputs, pDetector->avg);
    l.output = pDetector->last_avg2;
    if(pDetector->demo_delay == 0) l.output = pDetector->avg;
#endif
    l.output = prediction;
    if(l.type == DETECTION){
        LOGD("DETECTION!\n\n\n\n");
        get_detection_boxes(l, 1, 1, pDetector->demo_thresh, pFrame->probs, pFrame->boxes, 0);
    } else if (l.type == REGION){
        LOGD("REGION! buf[0].w=%d h=%d net.w=%d h=%d\n\n\n\n",
            pFrame->buff.w,
            pFrame->buff.h,
            pDetector->net.w,
            pDetector->net.h
            );
        get_region_boxes(l, pFrame->buff.w, pFrame->buff.h, pDetector->net.w, pDetector->net.h, pDetector->demo_thresh, pFrame->probs, pFrame->boxes, 0, 0, pDetector->demo_hier, 1);
    } else {
        error("Last layer must produce detections\n");
    }
    if (nms > 0) do_nms_obj(pFrame->boxes, pFrame->probs, l.w*l.h*l.n, l.classes, nms);

    //LOGD("\033[2J");
    //LOGD("\033[1;1H");
    //LOGD("\nFPS:%.1f\n",pDetector->fps);
    LOGD("Objects:\n\n");
    image display = pFrame->buff;
    LOGD("Draw detections pDetector->demo_detections=%d demo_classes=%d demo_thresh=%f\n", pDetector->demo_detections, pDetector->demo_classes, pDetector->demo_thresh);
    #ifdef DISPLAY_RESULS
    draw_detections(display, pDetector->demo_detections, pDetector->demo_thresh, pFrame->boxes, pFrame->probs, pDetector->demo_names, pDetector->demo_alphabet, pDetector->demo_classes);
    #endif
    LOGV("frame BBs=%p\n", pFrame->pBBs);
    evaluate_detections(pFrame, display, pDetector->demo_detections, pDetector->demo_thresh, pFrame->boxes, pFrame->probs, pDetector->demo_names, pDetector->demo_alphabet, pDetector->demo_classes);
    LOGV("frame BBs=%p\n", pFrame->pBBs);
    pDetector->demo_index = (pDetector->demo_index + 1)%pDetector->demo_frame;
    LOGD("demo_index=%d; demo_frame=%d\n", pDetector->demo_index, pDetector->demo_frame);
    pDetector->running = 0;
    #ifdef DISPLAY_RESULS
    display_in_thread(pFrame);
    #endif
    LOGV("cpy w=%d h=%d c=%d\n", pFrame->frameInfoWithCpy.im.w, pFrame->frameInfoWithCpy.im.h, pFrame->frameInfoWithCpy.im.c);
    #ifdef TEST_TRACKING
    tAnnInfo* pOutBBs = NULL;
    track_bb_in_frame(pFrame->pBBs, &pFrame->frameInfoWithCpy, &pFrame->frameInfoWithCpy, &pOutBBs);
    free_BBs(pOutBBs);
    #endif
    return 0;
}


/** 
 * NOTE: not thread safe
 */
tFrame* get_frame_from_cap(tDetector* pDetector, tFrame* apReuseFrame, double seekPos, int bSeek, int bSeekBackAfterRead)
{
    int status = -1;
    tFrame* pFrame = apReuseFrame ? apReuseFrame : NULL;
    double buff_ts = 0.0;
        
    if(apReuseFrame)
    {
        buff_ts = cvGetCaptureProperty(pDetector->cap, CV_CAP_PROP_POS_MSEC);
        status = fill_image_from_stream_sj(pDetector->cap, pFrame->buff, &pFrame->frameInfoWithCpy);
        letterbox_image_into(pFrame->buff, pDetector->net.w, pDetector->net.h, pFrame->buff_letter);
        LOGD("status = %d\n", status);
    }
    else
    {
        int j;
        layer l = pDetector->net.layers[pDetector->net.n-1];
        tFrameInfo frameInfoWithCpy;
        double curPos = cvGetCaptureProperty(pDetector->cap, CV_CAP_PROP_POS_FRAMES);
        LOGV("curPos now = %f\n", curPos);
        if(bSeek)
        {
            cvSetCaptureProperty(pDetector->cap, CV_CAP_PROP_POS_FRAMES, seekPos);
            while(cvGetCaptureProperty(pDetector->cap, CV_CAP_PROP_POS_FRAMES) <= seekPos)
            {    
                if(!cvGrabFrame(pDetector->cap))
                {
                   status = 0;
                   #if 0
                   image buff1 = get_image_from_stream_sj(pDetector->cap, &frameInfoWithCpy);
                   LOGV("last frame?=%p\n", buff1.data);
                   free_image(buff1);
                   #endif
                   break;
                }
            }
            LOGV("now = %f; seek for %f\n", cvGetCaptureProperty(pDetector->cap, CV_CAP_PROP_POS_FRAMES), seekPos);
            buff_ts = cvGetCaptureProperty(pDetector->cap, CV_CAP_PROP_POS_MSEC) - (1.0 / cvGetCaptureProperty(pDetector->cap, CV_CAP_PROP_FPS));
        }
        else
        {
            buff_ts = cvGetCaptureProperty(pDetector->cap, CV_CAP_PROP_POS_MSEC);
        }
        image buff = get_image_from_stream_sj(pDetector->cap, &frameInfoWithCpy);
        LOGV("now = %f\n", cvGetCaptureProperty(pDetector->cap, CV_CAP_PROP_POS_FRAMES));
        if(bSeekBackAfterRead)
        {
            cvSetCaptureProperty(pDetector->cap, CV_CAP_PROP_POS_FRAMES, curPos+1);
        }
        LOGD("DEBUGME\n");
        if(!buff.data)
        {
            status = 0;
            LOGV("could not read!\n");
            goto cleanup;
        }
      
        pFrame = (tFrame*)calloc(1, sizeof(tFrame));
        pFrame->frameInfoWithCpy = frameInfoWithCpy;
        pFrame->buff = buff;
        LOGD("DEBUGME\n");
        pFrame->buff_letter = letterbox_image(pFrame->buff, pDetector->net.w, pDetector->net.h);
        pFrame->ipl = cvCreateImage(cvSize(pFrame->buff.w,pFrame->buff.h), IPL_DEPTH_8U, pFrame->buff.c);
        pFrame->pNext = pDetector->pFrames;
        pDetector->pFrames = pFrame;
        pFrame->prod_lwn = l.w*l.h*l.n;
        //for(j = 0; j < pDetector->demo_frame; ++j) pDetector->predictions[j] = (float *) calloc(l.outputs, sizeof(float));

        pFrame->boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
        pFrame->probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
        for(j = 0; j < l.w*l.h*l.n; ++j) pFrame->probs[j] = (float *)calloc(l.classes+1, sizeof(float));

        pFrame->pDetector = pDetector;
    }

    if(pFrame)
    {
        pFrame->frameInfoWithCpy.fCurrentFrameTimeStamp = pFrame->buff_ts = buff_ts;
    }
    cleanup:

    if(status == 0) pDetector->demo_done = 1;

    return pFrame;
}

/** 
 * NOTE: not thread safe
 */
void free_frame(tDetector* pDetector, tFrame* apFrame)
{
    tFrame* pFrame = pDetector->pFrames;
    tFrame* pFramePrev = NULL;
    if(!apFrame)
        return;
    while(pFrame)
    {
        if(pFrame == apFrame)
        {
            if(pFramePrev)
                pFramePrev->pNext = pFrame->pNext;
            else
                pDetector->pFrames = pFrame->pNext;
            pFrame = NULL;
            break;
        }
        pFramePrev = pFrame;
        pFrame = pFrame->pNext;
    }
    free_image(apFrame->buff);
    if(apFrame->boxes)
        free(apFrame->boxes);
    if(apFrame->probs)
    {
        for(int i = 0;i < apFrame->prod_lwn; i++)
            free(apFrame->probs[i]);
        free(apFrame->probs);
    }
    free_image(apFrame->buff_letter);
    free_BBs(apFrame->pBBs);
    free_image(apFrame->frameInfoWithCpy.im);
    cvReleaseImage(&apFrame->ipl);
    free(apFrame);
}

void *fetch_in_thread(void *ptr)
{
    tFrame* pFrame = (tFrame*)ptr;
    tDetector* pDetector = pFrame->pDetector;
    LOGD("DEBUGME %p\n", pDetector);

    pFrame = get_frame_from_cap(pDetector, pFrame, 0, 0, 0);

    return 0;
}

void *display_in_thread(void *ptr)
{
    tFrame* pFrame = (tFrame*)ptr;
    tDetector* pDetector = pFrame->pDetector;
    LOGD("DEBUGME %p\n", pDetector);
    show_image_cv(pFrame->buff, "Demo", pFrame->ipl);
    int c = cvWaitKey(1);
    if (c != -1) c = c%256;
    if (c == 10){
        if(pDetector->demo_delay == 0) pDetector->demo_delay = 60;
        else if(pDetector->demo_delay == 5) pDetector->demo_delay = 0;
        else if(pDetector->demo_delay == 60) pDetector->demo_delay = 5;
        else pDetector->demo_delay = 0;
    } else if (c == 27) {
        pDetector->demo_done = 1;
        return 0;
    } else if (c == 82) {
        pDetector->demo_thresh += .02;
    } else if (c == 84) {
        pDetector->demo_thresh -= .02;
        if(pDetector->demo_thresh <= .02) pDetector->demo_thresh = .02;
    } else if (c == 83) {
        pDetector->demo_hier += .02;
    } else if (c == 81) {
        pDetector->demo_hier -= .02;
        if(pDetector->demo_hier <= .0) pDetector->demo_hier = .0;
    }
    return 0;
}

void *display_loop(void *ptr)
{
    tDetector* pDetector = (tDetector*)ptr;
    while(1){
        display_in_thread(pDetector);
    }
}

void *detect_loop(void *ptr)
{
    while(1){
        detect_in_thread(0);
    }
}

int initOnce =0;

void fire_bb_callbacks_for_frame(tDetector* pDetector, tFrame* pFrame)
{
    tAnnInfo* pBB;

    if(!pDetector || !pDetector->pDetectorModel || !pFrame 
        || !pFrame->pBBs || !pDetector->pDetectorModel->pfnRaiseAnnCb)
        return;

    LOGV("DEBUGME %p\n", pFrame->pBBs);
    pBB = pFrame->pBBs;    
    while(pBB)
    {
        LOGV("obj (%d, %d) (%d, %d) [%s] %f\n", pBB->x, pBB->y, pBB->w, pBB->h,
                pBB->pcClassName, pBB->fCurrentFrameTimeStamp);
        pDetector->pDetectorModel->pfnRaiseAnnCb(*pBB);
        pBB = pBB->pNext;
    }
    LOGV("DEBUGME\n");
    return;
}

tAnnInfo* if_bb_in_bbs(tAnnInfo* pBB, tAnnInfo* pBBs)
{
    tAnnInfo* pB = pBBs;
    if(!pBB || !pBBs)
        return NULL;

    while(pB)
    {
        if(pB->nBBId == pBB->nBBId)
        {
            LOGV("BB ID match\n");
            return pB;
        }
        pB = pB->pNext;
    }

    return NULL;
}

typedef struct
{
    int xMin;
    int yMin;
    int xMax;
    int yMax;
}tBBBounds;

tBBBounds get_bbbounds_from_BB(tAnnInfo* pBB)
{
    tBBBounds bbb;
    /** assuming BB(x,y) to be the top-left corner of the BB
     * and w,h respectively the width and height */
    bbb.xMin = pBB->x;
    bbb.xMax = pBB->x + pBB->w;
    bbb.yMin = pBB->y;
    bbb.yMax = pBB->y + pBB->h;

    LOGV("box: %p (x=%d y=%d w=%d h=%d)\n", pBB, pBB->x, pBB->y, pBB->w, pBB->h);

    return bbb;
}

void get_BB_from_bbbounds(tAnnInfo* pBB, tBBBounds* pBBB)
{
    pBB->x = pBBB->xMin;
    pBB->y = pBBB->yMin;
    pBB->w = pBBB->xMax - pBBB->xMin;
    pBB->h = pBBB->yMax - pBBB->yMin;
}


tBBBounds interpolateBoundingBox(tBBBounds bbb1, tBBBounds bbb2, double fFrac)
{
    tBBBounds bbbI = {0}; /**< BBB interpolated */

    double ifFrac = ((double)(1.0)) - fFrac;

    /** new point = (1 - frac) * x0 + frac * x1 */
    bbbI.xMin = (ifFrac * bbb1.xMin) + (fFrac * bbb2.xMin);
    bbbI.yMin = (ifFrac * bbb1.yMin) + (fFrac * bbb2.yMin);
    bbbI.xMax = (ifFrac * bbb1.xMax) + (fFrac * bbb2.xMax);
    bbbI.yMax = (ifFrac * bbb1.yMax) + (fFrac * bbb2.yMax);

    return bbbI;
}

int interpolate_bbs_btw_frames(tDetector* pDetector, tFrame** ppFramesHashBase, const int nF, const int nL)//, const int nTarget)
{
    int missedInLast = -1;
    tAnnInfo* pBB2;
    tAnnInfo* pBB1;
    tFrame* pFrame1;
    tFrame* pFrameL;

    LOGV("DEBUGME\n");
    if(!pDetector || !ppFramesHashBase 
        || (nL - nF - 1) == 0/**< nothing to interpolate */)
        return missedInLast;

    LOGV("DEBUGME\n");
    pFrame1 = ppFramesHashBase[nF];
    pFrameL = ppFramesHashBase[nL];
    pBB2 = pFrameL->pBBs;
    while(pBB2)
    {
        LOGV("DEBUGME\n");
        if((pBB1 = if_bb_in_bbs(pBB2, pFrame1->pBBs)))
        {
            LOGV("DEBUGME\n");
            /** find D=displacement between the objects;
             * E = divide D for the number of frames in between nF and nL 
             * now advance the BB[F](x,y) by E to replicate BB[i](x,y) until BB[L](x,y)
             * TODO: (w,h) shall slowly increase or decrease over the interpolation
             */
            tBBBounds bbb1, bbb2;
            bbb2 = get_bbbounds_from_BB(pBB2);
            bbb1 = get_bbbounds_from_BB(pBB1);
            /** D = sqrt(sq(x2-x1) + sq(y2-y1));
             * line is from TL corner to TL corner */
            double D = displacement_btw_BBs(pBB1, pBB2);
            #if 1
            for(int i = nF + 1; i < nL; i++)
            #else
            int i = nTarget;
            #endif
            {
                LOGV("DEBUGME\n");
                {
                    tAnnInfo* pBB1i; /**< between 1 and 2 */
                    pBB1i = (tAnnInfo*)malloc(sizeof(tAnnInfo));
                    tBBBounds bbb1i;
                    double dispToNewPoint = (D / (nL - nF)) * (i); /**< divide D equally between the frames and mult by i; i shall never be 0 cause nF is arr index */ 
                    double fFrac = D ? dispToNewPoint / D : 0;
                    LOGV("fFrac=%f D=%f dispToNewPoint=%f\n", fFrac, D, dispToNewPoint);
                    bbb1i = interpolateBoundingBox(bbb1, bbb2, fFrac);
                    LOGV("bbb1i (%d, %d) -> (%d, %d)\n", bbb1i.xMin, bbb1i.yMin, bbb1i.xMax, bbb1i.yMax);
                    memcpy(pBB1i, pBB1, sizeof(tAnnInfo));
                    pBB1i->pcClassName = (char*)malloc(strlen(pBB1->pcClassName) + 1);
                    strcpy(pBB1i->pcClassName, pBB1->pcClassName);
                    get_BB_from_bbbounds(pBB1i, &bbb1i);
                    pBB1i->fCurrentFrameTimeStamp = pBB1->fCurrentFrameTimeStamp + (((pBB2->fCurrentFrameTimeStamp - pBB1->fCurrentFrameTimeStamp) / (nL - nF)) * i);
                    pBB1i->pNext = ppFramesHashBase[i]->pBBs;
                    ppFramesHashBase[i]->pBBs = pBB1i;
                }
            }
        }
        pBB2 = pBB2->pNext;
    }

    return missedInLast;
}

void detect_object_for_frame(tDetector* pDetector, tFrame* pFrame, int count)
{
    pthread_t detect_thread;
#if 0
    pthread_t fetch_thread;
#endif

    if(!pDetector || !pFrame)
        return;

            if(pthread_create(&detect_thread, 0, detect_in_thread, pFrame)) error("Thread creation failed" );
#if 0
            if(!prefix)
#endif
            {
                if(count % (pDetector->demo_delay+1) == 0){
                    pDetector->fps = 1./(get_wall_time() - pDetector->demo_time);
                    pDetector->demo_time = get_wall_time();
                    #if 0
                    float *swap = pDetector->last_avg;
                    pDetector->last_avg  = pDetector->last_avg2;
                    pDetector->last_avg2 = swap;
                    memcpy(pDetector->last_avg, pDetector->avg, l.outputs*sizeof(float));
                    #endif
                }
            }
#if 0
            else{
                char name[256];
                //LOGD(name, "%s_%08d", prefix, count);
                save_image(pFrame->buff, name);
            }
#endif
            #if 0
            pthread_join(fetch_thread, 0);
            #endif
            pthread_join(detect_thread, 0);

}

char folder_name[100];

void demo2(void* apDetector, char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    tDetector* pDetector = (tDetector*)apDetector;
    double prevDumpTime = get_wall_time();
    LOGD("DEBUGME\n");

    pDetector->demo_delay = delay;
    pDetector->demo_frame = avg_frames;
    image **alphabet = NULL;
#ifdef DISPLAY_RESULS
    alphabet = load_alphabet();
#endif
    pDetector->demo_names = names;
    pDetector->demo_alphabet = alphabet;
    pDetector->demo_classes = classes;
    pDetector->demo_thresh = thresh;
    pDetector->demo_hier = hier;
    LOGD("Demo\n");
    LOGD("classes=%d delay=%d avg_frames=%d hier=%f w=%d h=%d frames=%d fullscreen=%d\n", classes, delay, avg_frames, hier, w, h, frames, fullscreen);
    if(!initOnce)
    {
    #if 0
    pDetector->predictions = (float**)calloc(pDetector->demo_frame, sizeof(float*));
    #endif
    pDetector->net = parse_network_cfg(cfgfile);
    if(weightfile){
        load_weights(&pDetector->net, weightfile);
    }
    set_batch_network(&pDetector->net, 1);
    initOnce = 1;
    }

    srand(2222222);


    if(filename){
        LOGD("video file: %s\n", filename);
        pDetector->cap = cvCaptureFromFile(filename);
        strcpy(folder_name, strstr(filename, ".mp4") - 10);
        LOGV("folder name is %s\n", folder_name);
        LOGD("DEBUGME %p\n", pDetector->cap);
    }else{
        pDetector->cap = cvCaptureFromCAM(cam_index);

        if(w){
            cvSetCaptureProperty(pDetector->cap, CV_CAP_PROP_FRAME_WIDTH, w);
        }
        if(h){
            cvSetCaptureProperty(pDetector->cap, CV_CAP_PROP_FRAME_HEIGHT, h);
        }
        if(frames){
            cvSetCaptureProperty(pDetector->cap, CV_CAP_PROP_FPS, frames);
        }
    }

    if(!pDetector->cap)
    {
        //error("Couldn't connect to webcam.\n");
        LOGE("ERROR; file could not be read / dev could not be opened\n");
        return;
    }

    layer l = pDetector->net.layers[pDetector->net.n-1];
    pDetector->demo_detections = l.n*l.w*l.h;

    LOGD("DEBUGME\n");
    #if 0
    pDetector->avg = (float *) calloc(l.outputs, sizeof(float));
    pDetector->last_avg  = (float *) calloc(l.outputs, sizeof(float));
    pDetector->last_avg2 = (float *) calloc(l.outputs, sizeof(float));
    #endif

#if 0
    pDetector->buff[1] = copy_image(pFrame->buff);
    pDetector->buff[2] = copy_image(pFrame->buff);
    pDetector->buff_letter[1] = letterbox_image(pFrame->buff, pDetector->net.w, pDetector->net.h);
    pDetector->buff_letter[2] = letterbox_image(pFrame->buff, pDetector->net.w, pDetector->net.h);
#endif

    LOGD("DEBUGME\n");
    int count = 0;
    #ifdef DISPLAY_RESULS
    if(!prefix){
        cvNamedWindow("Demo", CV_WINDOW_NORMAL); 
        if(fullscreen){
            cvSetWindowProperty("Demo", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
        } else {
            cvMoveWindow("Demo", 0, 0);
            cvResizeWindow("Demo", 1352, 1013);
        }
    }
    #endif

    pDetector->demo_time = get_wall_time();

    pDetector->nFps = (int)cvGetCaptureProperty(pDetector->cap, CV_CAP_PROP_FPS);
    pDetector->totalFramesInVid = cvGetCaptureProperty(pDetector->cap, CV_CAP_PROP_FRAME_COUNT);
    
    pDetector->nSkipFramesCnt = (int)(pDetector->nFps / pDetector->nTargetFps);

    //cvSetCaptureProperty(pDetector->cap, CV_CAP_PROP_FPS, (double)pDetector->nTargetFps);

    LOGD("DEBUGME %d\n", pDetector->demo_done);
    while(!pDetector->demo_done){
        LOGD("pDetector->demo_done=%d count=%d prefix=%s pDetector->nSkipFramesCnt=%d\n", pDetector->demo_done, count, prefix, pDetector->nSkipFramesCnt);
        LOGD("cap prop; w=%f h=%f frame_count=%f FPS=%f POS_MS=%f pos_count=%f\n", 
                cvGetCaptureProperty(pDetector->cap, CV_CAP_PROP_FRAME_WIDTH),
                cvGetCaptureProperty(pDetector->cap, CV_CAP_PROP_FRAME_HEIGHT),
                cvGetCaptureProperty(pDetector->cap, CV_CAP_PROP_FRAME_COUNT),
                cvGetCaptureProperty(pDetector->cap, CV_CAP_PROP_FPS),
                cvGetCaptureProperty(pDetector->cap, CV_CAP_PROP_POS_MSEC),
                cvGetCaptureProperty(pDetector->cap, CV_CAP_PROP_POS_FRAMES));
#ifdef ENABLE_VIDEO_FILE_READ_AT_TAR_FPS
        pDetector->bProcessThisFrame = (pDetector->nCurFrameCount && !(pDetector->nCurFrameCount % pDetector->nSkipFramesCnt));
        if(pDetector->bProcessThisFrame)
#else
        if(1)
#endif /**< ENABLE_VIDEO_FILE_READ_AT_TAR_FPS */
        {
            #if 0
            if(pthread_create(&fetch_thread, 0, fetch_in_thread, pFrame)) error("Thread creation failed");
            #endif
#ifndef IMPURE_CNN
            tFrame* pFrame = NULL;
            LOGD("DEBUGME\n");
            pFrame = get_frame_from_cap(pDetector, NULL, 0, 0, 0);
            LOGD("DEBUGME\n");
            if(!pFrame)
            {
                LOGE("file read failed\n");
                return;
            }
            detect_object_for_frame(pDetector, pFrame, count);
#else
            /** 1. In a loop, read MAX_FRAMES_TO_HASH frames from file/cam into the 'hash'
             *  2. run darknet (CNN) on the 1st frame
             *  3. run track_bb_in_frame() on the (MAX_FRAMES_TO_HASH - 1)th frame
             *  4. find BBs for all frames in between using interpolation
             *  5. fire pfnRaiseAnnCb() callback for each BB in the frames from 0 to (MAX_FRAMES_TO_HASH - 1)
             *  6. mess_cleanup
             */
            
            pDetector->fStartTime = get_wall_time();
            int nL, nFIdxToReadInto = 0; /**< nL is the count of total frames available in the hash for this go! */
            LOGV("second? %p\n", pDetector->pFramesHash[0]);
            if(pDetector->pFramesHash[0])
            {
                nFIdxToReadInto = 1;
                nL = MAX_FRAMES_TO_HASH; /**< hash in  */
            }
            else
            {
                nL = MAX_FRAMES_TO_HASH;
            }
            for(int i = nFIdxToReadInto; i < nL; i++)
            {
                /** except for nFIdxToReadInto and nL-1, all frames are read only
                 * to do a smooth seek to the (nL-1)th frame */
                tFrame* pFramePrev = (i > (nFIdxToReadInto+1)) ? pDetector->pFramesHash[i-1] : NULL; /**< frame to flush until nL */
                if(i >= (nFIdxToReadInto+1) && i < (nL-1))
                {
                    LOGV("seeking %d\n", i);
                    if(cvGrabFrame(pDetector->cap))
                        pDetector->pFramesHash[i] = (tFrame*)calloc(1, sizeof(tFrame));
                    else
                        pDetector->pFramesHash[i] = NULL;
                }
                else
                {
                    LOGV("reading %d\n", i);
                    pDetector->pFramesHash[i] = get_frame_from_cap(pDetector, NULL, pDetector->countFrame, 0, 0);
                }
                if(!pDetector->pFramesHash[i])
                {
                    if(i == nFIdxToReadInto)
                    {
                        LOGV("could not read very first img in hash gap; done reading the stream\n");
                        /** cleanup any saved hash */
                        if(pDetector->pFramesHash[0])
                            free_frame(pDetector, pDetector->pFramesHash[0]);
                        dump_lane_info(pDetector->pLanesInfo);
                        return;
                    }
                    nL = i - 1;
                    LOGV("prob with read\n");
                    break;
                }
                if(i == 0)
                {
                    FILE* pFile = fopen("lanes.json", "r");
                    if(pFile && !pDetector->pLanesInfo)
                    {
                        cJSON* pJSONPolys = NULL;
                        {
                            char* lane_info = calloc(1, 10240);
                            int j = 0;
                            while(fread(&lane_info[j], sizeof(char), 1, pFile))
                            {
                                j++;
                            }
                            LOGV("lane_info is [%s]\n", lane_info);
                            pJSONPolys = cJSON_Parse(lane_info);
                            free(lane_info);
                        }
                        int nJSONPolys = 0;
                        if(pJSONPolys && (nJSONPolys = cJSON_GetArraySize(pJSONPolys)))
                        {
                            LOGV("DEBUGME\n");
                            pDetector->pLanesInfo = (tLanesInfo*)calloc(1, sizeof(tLanesInfo));
                            LOGV("DEBUGME\n");
                            for(int j = 0; j < nJSONPolys; j++)
                            {
                                LOGV("DEBUGME\n");
                                cJSON* pJSONPoly = cJSON_GetArrayItem(pJSONPolys, j);
                                int nJSONVs = 0;
                                if(pJSONPoly)
                                {
                                    LOGV("DEBUGME\n");
                                    cJSON* pJSONLId = cJSON_GetObjectItem(pJSONPoly, "laneid");
                                    cJSON* pJSONLRoute = cJSON_GetObjectItem(pJSONPoly, "route");
                                    cJSON* pJSONVArr = cJSON_GetObjectItem(pJSONPoly, "vertices");
                                    nJSONVs = cJSON_GetArraySize(pJSONVArr);
                                    tLane* pP = (tLane*)calloc(1, sizeof(tLane));
                                    pP->nLaneId = pJSONLId->valueint;
                                    LOGV("DEBUGME\n");
                                    pP->pcRoute = (char*)calloc(1, strlen(pJSONLRoute->valuestring) + 1);
                                    strcpy(pP->pcRoute, pJSONLRoute->valuestring);
                                    LOGV("DEBUGME\n");
                                    for(int k = 0; k < nJSONVs; k++)
                                    {
                                        LOGV("DEBUGME\n");
                                        tVertex* pV = (tVertex*)calloc(1, sizeof(tVertex));
                                        cJSON* pJVertex = cJSON_GetArrayItem(pJSONVArr, k);
                                        cJSON* pJX = cJSON_GetObjectItem(pJVertex, "x");
                                        cJSON* pJY = cJSON_GetObjectItem(pJVertex, "y");
                                        LOGV("point is (%d, %d)\n", pJX->valueint, pJY->valueint);
#if 0
                                        pV->x = pJX->valueint;
                                        pV->y = pJY->valueint;
#else
                                        pV->x = (int)(((pJX->valueint * 1.0) /1920.0) * 854.0);
                                        pV->y = (int)(((pJY->valueint * 1.0) /1080.0) * 480.0);
#endif
                                        pP->nVs++;
                                        pV->pNext = pP->pVs;
                                        pP->pVs = pV;
                                    LOGV("DEBUGME\n");
                                    }
                                    LOGV("DEBUGME\n");
                                    pP->pNext = pDetector->pLanesInfo->pLanes;
                                    pDetector->pLanesInfo->pLanes = pP;
                                    pDetector->pLanesInfo->nLanes++;
                                }
                            }
                                    LOGV("DEBUGME\n");
                        }
                        cJSON_free(pJSONPolys);
                    }
                    if(pFile)
                        fclose(pFile);
#if 0
                    pDetector->pLanesInfo = getLaneInfo(&pDetector->pFramesHash[0]->frameInfoWithCpy, pDetector->pLanesInfo);
#endif
                    if(!pDetector->pLanesInfo) 
                    {
                        /** ask user to provide the lane info */
                        pDetector->pLanesInfo = getLaneInfo(&pDetector->pFramesHash[0]->frameInfoWithCpy, NULL);
                        LOGV("polygons info=%p\n", pDetector->pLanesInfo);
                        cJSON* pJSONPolys = cJSON_CreateArray();
                        if(pDetector->pLanesInfo)
                        {
                            tLane* pP = pDetector->pLanesInfo->pLanes;
                            while(pP)
                            {
                                cJSON* pJId = cJSON_CreateNumber(pP->nLaneId);
                                cJSON* pJRoute = cJSON_CreateString("default-route");
                                cJSON* pJLaneInfo = cJSON_CreateObject();
                                cJSON_AddItemToObject(pJLaneInfo, "laneid", pJId);
                                cJSON_AddItemToObject(pJLaneInfo, "route", pJRoute);

                                cJSON* pJSONPoly = cJSON_CreateArray();
                                /** polygon: list of vertices in order */
                                tVertex* pV = pP->pVs;
                                while(pV)
                                {
                                    cJSON* pJX = cJSON_CreateNumber(pV->x);
                                    cJSON* pJY = cJSON_CreateNumber(pV->y);
                                    cJSON* pJVertex = cJSON_CreateObject();
                                    cJSON_AddItemToObject(pJVertex, "x", pJX);
                                    cJSON_AddItemToObject(pJVertex, "y", pJY);
                                    cJSON_AddItemToArray(pJSONPoly, pJVertex);
                                    pV = pV->pNext;
                                }
                                cJSON_AddItemToObject(pJLaneInfo, "vertices", pJSONPoly);
                                cJSON_AddItemToArray(pJSONPolys, pJLaneInfo);
                                pP = pP->pNext;
                            }
                        }
                        LOGV("polygon info: [%s]\n", cJSON_Print(pJSONPolys));
                        LOGV("do the one time detect\n");
                        FILE* pFile = fopen("lanes.json", "w");
                        fprintf(pFile, "%s", cJSON_Print(pJSONPolys));
                        fclose(pFile);
                        cJSON_free(pJSONPolys);
                    }
                    /** populate pLanesInfo with class detail */
                    tLane* pLane = pDetector->pLanesInfo->pLanes;
                    while(pLane)
                    {
                        LOGV("number of demo_classes=%d\n", pDetector->demo_classes);
                        pLane->pnVehicleCount = (long long*)calloc(1, sizeof(long long) * (pDetector->demo_classes+1));
                        pLane->nTypes = pDetector->demo_classes;
                        pLane = pLane->pNext;
                    }
                    LOGV("number of lanes=%d %d\n", pDetector->pLanesInfo->nLanes, pDetector->demo_classes);
                    pDetector->pLanesInfo->ppRouteTrafficInfo = (tRouteTrafficInfo**)calloc(pDetector->pLanesInfo->nLanes+1, sizeof(tRouteTrafficInfo*));
                    for(int j = 0; j < pDetector->pLanesInfo->nLanes+1; j++)
                    {
                        pDetector->pLanesInfo->ppRouteTrafficInfo[j] = (tRouteTrafficInfo*)calloc(pDetector->pLanesInfo->nLanes+1, sizeof(tRouteTrafficInfo));
                        for(int k = 0; k < pDetector->pLanesInfo->nLanes+1; k++)
                        {
                            pDetector->pLanesInfo->ppRouteTrafficInfo[j][k].pnVehicleCount = (long long*)calloc(1, sizeof(long long) * (pDetector->demo_classes+1));
                            pDetector->pLanesInfo->ppRouteTrafficInfo[j][k].nTypes = pDetector->demo_classes;
                        }
                    }
                    pDetector->pLanesInfo->names = pDetector->demo_names;
                    pDetector->pLanesInfo->nTypes = pDetector->demo_classes;
                    display_lanes_info(pDetector->pLanesInfo);
                    detect_object_for_frame(pDetector, pDetector->pFramesHash[0], count);
                }
                pDetector->countFrame++;
                /** delete prev frame; we dont need it; this exercise is alt to seek */
                if(pFramePrev)
                {
                    free_frame(pDetector, pFramePrev);
                    pDetector->pFramesHash[i-1] = calloc(1, sizeof(tFrame));;
                }
            }
            pDetector->fEndTime = get_wall_time();
            LOGV("[except for 1st print] seek took %fms; means read is @ %ffps\n", 
                (pDetector->fEndTime - pDetector->fStartTime) * 1000.0,
                (nL * 1.0) / (pDetector->fEndTime - pDetector->fStartTime));
            
            /** use the BBs at pDetector->pFramesHash[0]->pBBs and fetch the tracked replicas of them in
             * pDetector->pFramesHash[MAX_FRAMES_TO_HASH-1] */
            if(pDetector->pFramesHash[0] && pDetector->pFramesHash[nL-1])
            {
                detect_object_for_frame(pDetector, pDetector->pFramesHash[nL-1], count);
                track_bb_in_frame(pDetector->pFramesHash[0]->pBBs, 
                    &pDetector->pFramesHash[0]->frameInfoWithCpy, 
                    &pDetector->pFramesHash[nL-1]->frameInfoWithCpy,
                    &pDetector->pFramesHash[nL-1]->pBBs,
                    pDetector->pLanesInfo);
                /** interpolate all the BBs for frames in between 0 and (nL-1) */
                interpolate_bbs_btw_frames(pDetector, pDetector->pFramesHash, 0, nL-1);
                LOGV("BBs tracked=%p\n", pDetector->pFramesHash[nL-1]->pBBs);
    
            
            }
            else
            {
                LOGV("NOT RIGHT\n");
            }
            pDetector->fEndTime = get_wall_time();
            LOGV("detection, tracking, and interpolation took %fms; means it is @ %ffps\n", 
                (pDetector->fEndTime - pDetector->fStartTime) * 1000.0,
                (nL * 1.0) / (pDetector->fEndTime - pDetector->fStartTime));
            #if 1
            LOGV("i=%d to %d\n", nFIdxToReadInto, nL);
            for(int i = nFIdxToReadInto; i < nL; i++)
            {
                LOGV("DEBUGME %p\n", pDetector->pFramesHash);
                if(pDetector->pFramesHash[i])
                {
                    LOGV("firing CB for frame %d BBs=%p\n", i, pDetector->pFramesHash[i]->pBBs);
                    fire_bb_callbacks_for_frame(pDetector, pDetector->pFramesHash[i]);
                    if((i == (nL-1)) && !pDetector->demo_done)
                    {
                        LOGV("save the %dth hash in 0th for next iteration %p %p\n", i, pDetector->pFramesHash[i], pDetector->pFramesHash[i]->pBBs);
                        free_frame(pDetector, pDetector->pFramesHash[0]);
                        pDetector->pFramesHash[0] = pDetector->pFramesHash[i];
                        pDetector->pFramesHash[i] = NULL;
                        break;
                    }
                    LOGV("free_frame\n");
                    free_frame(pDetector, pDetector->pFramesHash[i]);
                    pDetector->pFramesHash[i] = NULL;
                }
            }
            pDetector->fEndTime = get_wall_time();
            LOGV("detection, tracking, interpolation, and fire_cb took %fms; means it is @ %ffps\n", 
                (pDetector->fEndTime - pDetector->fStartTime) * 1000.0,
                (nL * 1.0) / (pDetector->fEndTime - pDetector->fStartTime));
            #endif
#endif
            ++count;
            /** done detecting one frame */
        }
        else
        {
            cvGrabFrame(pDetector->cap);
        }
        LOGD("DEBUGME\n");
        pDetector->nCurFrameCount++;


        /** dump lane info as and when needed */
        if((get_wall_time() - prevDumpTime >= (1.0 * 60 * 30))
            || pDetector->demo_done)
        {
            prevDumpTime = get_wall_time();
            dump_lane_info(pDetector->pLanesInfo);
        }
    }

    
    LOGD("DEBUGME\n");
    //cvReleaseCapture(pDetector->cap);
    LOGD("DEBUGME\n");
    pDetector->cap = NULL;
}


void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg_frames, float hier, int w, int h, int frames, int fullscreen)
{
    tDetectorModel* pDetectorModel = calloc(1, sizeof(tDetectorModel));
    pDetectorModel->pcCfg = cfgfile;
    pDetectorModel->pcWeights = weightfile;
    pDetectorModel->pcFileName = (char*)filename;
    pDetectorModel->pcNames = (char*)names;
    pDetectorModel->isVideo = 1; //true by default
    //return demo2(pDetector, cfgfile, weightfile, thresh, cam_index, filename, names, classes, delay, prefix, avg_frames, hier, w, h, frames, fullscreen);
    run_detector_model(pDetectorModel);
}
#else
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int delay, char *prefix, int avg, float hier, int w, int h, int frames, int fullscreen)
{
    fLOGD(stderr, "Demo needs OpenCV for webcam images.\n");
}
#endif

int run_detector_model(tDetectorModel* apDetectorModel)
{
    tDetector* pDetector = calloc(1, sizeof(tDetector));
    init_globals(pDetector);
#if 0
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .24);
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    int avg = find_int_arg(argc, argv, "-avg", 3);
    if(argc < 4){
        fLOGD(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        LOGD("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");
    int fullscreen = find_arg(argc, argv, "-fullscreen");
    int width = find_int_arg(argc, argv, "-w", 0);
    int height = find_int_arg(argc, argv, "-h", 0);
    int fps = find_int_arg(argc, argv, "-fps", 0);

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    char *filename = (argc > 6) ? argv[6]: 0;
    if(0==strcmp(argv[2], "test")) test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, outfile, fullscreen);
    else if(0==strcmp(argv[2], "train")) train_detector(datacfg, cfg, weights, gpus, ngpus, clear);
    else if(0==strcmp(argv[2], "valid")) validate_detector(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "valid2")) validate_detector_flip(datacfg, cfg, weights, outfile);
    else if(0==strcmp(argv[2], "recall")) validate_detector_recall(cfg, weights);
    else if(0==strcmp(argv[2], "demo")) {
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
        demo(cfg, weights, thresh, cam_index, filename, names, classes, frame_skip, prefix, avg, hier_thresh, width, height, fps, fullscreen);
    }
#endif
    {
        pDetector->pDetectorModel = apDetectorModel;
        LOGD("isVideo=%d\n", apDetectorModel->isVideo);
        if(1)//apDetectorModel->isVideo)
        {
            LOGD("in %p\n", apDetectorModel);
            LOGD("demo start %s\n", apDetectorModel->pcDataCfg);
            list *options = read_data_cfg(apDetectorModel->pcDataCfg ? apDetectorModel->pcDataCfg : "cfg/aic.data");
            LOGD("h1\n");
            char *name_list = option_find_str(options, "names", apDetectorModel->pcNames ? apDetectorModel->pcNames : "data/names.list");
            LOGD("name_list=%s\n", name_list);
            char **names = get_labels(name_list);
            LOGD("h1\n");
            int classes = option_find_int(options, "classes", 20);
            LOGD("h1\n");
            LOGD("Detector Model cb is %p\n", pDetector->pDetectorModel->pfnRaiseAnnCb);
            demo2(pDetector, apDetectorModel->pcCfg, apDetectorModel->pcWeights, 0.24/**< apDetectorModel->fThresh */, 
                0/**< cam_index */, apDetectorModel->pcFileName, names, classes, 0 /**< frame_skip */, 
                NULL, 1, pDetector->demo_hier, 0 /**< w */, 0 /**< h */, 0 /**< 0 */, 0 /**< fullscreen */
                );
        }
        else
        {
            test_detector_on_img(pDetector, apDetectorModel->pcDataCfg, apDetectorModel->pcCfg, apDetectorModel->pcWeights, apDetectorModel->pcFileName, apDetectorModel->fThresh, pDetector->demo_hier, NULL, 0);
        }

       
    }

    return 0;
}

static void test_detector_on_img(tDetector* pDetector, char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
    LOGD("DEBUGME\n");
    list *options = read_data_cfg(datacfg);
    LOGD("DEBUGME\n");
    char *name_list = option_find_str(options, "names", "data/names.list");
    LOGD("DEBUGME\n");
    char **names = get_labels(name_list);
    LOGD("DEBUGME\n");

    image **alphabet = NULL;
#ifdef DISPLAY_RESULS
    alphabet = load_alphabet();
#endif
    LOGD("DEBUGME\n");
    network net = parse_network_cfg(cfgfile);
    LOGD("DEBUGME\n");
    if(weightfile){
        load_weights(&net, weightfile);
    }
    set_batch_network(&net, 1);
    srand(2222222);
    clock_t time;
    char buff[2560];
    char *input = buff;
    int j;
    float nms=.4;
    LOGD("DEBUGME\n");
    while(1){
    LOGD("DEBUGME [%s]\n", filename);
        if(filename){
            strncpy(input, filename, 2560);
            LOGD("input [%s]\n", input);
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        LOGD("DEBUGME\n");
        image im = load_image_color(input,0,0);
        LOGD("DEBUGME im.data=%p\n", im.data);
        image sized = letterbox_image(im, net.w, net.h);
        LOGD("DEBUGME\n");
        //image sized = resize_image(im, net.w, net.h);
        //image sized2 = resize_max(im, net.w);
        //image sized = crop_image(sized2, -((net.w - sized2.w)/2), -((net.h - sized2.h)/2), net.w, net.h);
        //resize_network(&net, sized.w, sized.h);
        layer l = net.layers[net.n-1];

        box *boxes = (box*)calloc(l.w*l.h*l.n, sizeof(box));
        float **probs = (float**)calloc(l.w*l.h*l.n, sizeof(float *));
        for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float*)calloc(l.classes + 1, sizeof(float *));

        float *X = sized.data;
        time=clock();
        network_predict(net, X);
        printf("%s: Predicted in %f seconds.\n", input, sec(clock()-time));
        get_region_boxes(l, im.w, im.h, net.w, net.h, thresh, probs, boxes, 0, 0, hier_thresh, 1);
        if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        //else if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);
        //evaluate_detections(pDetector, im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
#ifdef DISPLAY_RESULS
        draw_detections(im, l.w*l.h*l.n, thresh, boxes, probs, names, alphabet, l.classes);
        if(outfile){
            save_image(im, outfile);
        }
        else{
            save_image(im, "predictions");
#ifdef OPENCV
            cvNamedWindow("predictions", CV_WINDOW_NORMAL); 
            if(fullscreen){
                cvSetWindowProperty("predictions", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);
            }
            show_image(im, "predictions");
            cvWaitKey(0);
            cvDestroyAllWindows();
#endif
        }
#endif

        free_image(im);
        free_image(sized);
        free(boxes);
        free_ptrs((void **)probs, l.w*l.h*l.n);
        if (filename) break;
    }
}

void dump_lane_info(tLanesInfo* pLanesInfo)
{
    char filename[500] = {0};
    char cmd[500] = {0};
    snprintf(cmd, 500, "mkdir -p data/team1_darknet/%s", folder_name);
    system(cmd);
    snprintf(filename, 500, "data/team1_darknet/%s/traffic_pattern_%f", folder_name, get_wall_time());
    FILE* fp = fopen(filename, "w");
    if(pLanesInfo && fp)
    {   
        tLane* pL = pLanesInfo->pLanes;
        cJSON* pJSONLaneArr = cJSON_CreateArray();
        LOGV("DEBUGME\n");
        while(pL)
        {   
            /** polygon: list of vertices in order */
            LOGV("Lane %p ID: %d; route:[%s] vertices:\n", pL, pL->nLaneId, pL->pcRoute);
            cJSON* pJSONLaneInfo = cJSON_CreateObject();
            cJSON_AddNumberToObject(pJSONLaneInfo, "laneid", pL->nLaneId); 
            cJSON_AddStringToObject(pJSONLaneInfo, "route", pL->pcRoute);
        LOGV("DEBUGME\n");
            for(int k = 1; k < pLanesInfo->nTypes; k++)
            {
        LOGV("DEBUGME %s\n", pLanesInfo->names[k]);
                cJSON_AddNumberToObject(pJSONLaneInfo, pLanesInfo->names[k], pL->pnVehicleCount[k]);
            }
        LOGV("DEBUGME\n");
            cJSON_AddNumberToObject(pJSONLaneInfo, "avg-stay-time", pL->fAvgStayDuration);
            cJSON_AddNumberToObject(pJSONLaneInfo, "total-vehicles", pL->nTotalVehiclesSoFar);
            cJSON_AddItemToArray(pJSONLaneArr, pJSONLaneInfo);
        LOGV("DEBUGME\n");
            pL = pL->pNext;
        }   
        LOGV("DEBUGME\n");
        cJSON* pJSONMat = cJSON_CreateObject();
        for(int i = 1; i < pLanesInfo->nLanes+1; i++)
        {
        LOGV("DEBUGME\n");
            cJSON* pJSONFrom = cJSON_CreateObject();
            char pcFrom[50] = {0};
            snprintf(pcFrom, 50, "from_%d", i);
        LOGV("DEBUGME\n");
            for(int j = 1; j < pLanesInfo->nLanes+1; j++)
            {
                cJSON* pJSONFromTo = cJSON_CreateObject();
                char pcFromTo[1000] = {0};
                snprintf(pcFromTo, 1000, "from_%d_to_%d_%s_to_%s", i, j, getLaneById(pLanesInfo, i)->pcRoute, getLaneById(pLanesInfo, j)->pcRoute);
        LOGV("DEBUGME\n");
                for(int k = 1; k < pLanesInfo->nTypes; k++)
                {
        LOGV("DEBUGME\n");
                    cJSON_AddNumberToObject(pJSONFromTo, pLanesInfo->names[k], pLanesInfo->ppRouteTrafficInfo[i][j].pnVehicleCount[k]);
                }
        LOGV("DEBUGME\n");
                cJSON_AddItemToObject(pJSONFrom, pcFromTo, pJSONFromTo);
            }
        LOGV("DEBUGME\n");
            cJSON_AddItemToObject(pJSONMat, pcFrom, pJSONFrom);
        }
        LOGV("DEBUGME\n");
        cJSON* pJSONFinalPattern = cJSON_CreateObject();
        cJSON_AddItemToObject(pJSONFinalPattern, "lanes_info", pJSONLaneArr);
        cJSON_AddItemToObject(pJSONFinalPattern, "traffic_flux", pJSONMat);
        LOGV("final pattern:[%s]\n", cJSON_Print(pJSONFinalPattern));
        fwrite(cJSON_Print(pJSONFinalPattern), 1, strlen(cJSON_Print(pJSONFinalPattern)), fp);
        cJSON_free(pJSONFinalPattern);
        fclose(fp);
    }   
}
