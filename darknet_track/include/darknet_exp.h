#ifndef _DARKNET_EXP_H_
#define _DARKNET_EXP_H_

typedef struct AnnInfo tAnnInfo;

struct AnnInfo
{
     int x;
     int y;
     int w;
     int h;
     char* pcClassName;
     double fCurrentFrameTimeStamp;
     int nVideoId;
     double prob;
     int nBBId;
     double fIoU; /**< used for processing IoU in darknet framework */
     char bBBIDAssigned;
     tAnnInfo* pNext;
};

typedef struct
{
     image im;
     int widthStep; /**< do not free pointers in this mem; they're garbage; other attributes will have proper value */
     double fCurrentFrameTimeStamp;
}tFrameInfo;

typedef int (*tfnRaiseAnnCb)(tAnnInfo apAnnInfo);
typedef struct
{
    char* pcCfg; /**< yolo.cfg */
    char* pcWeights; /**< yolo.weights */
    char* pcFileName; /**< .mp4 file */
    char* pcDataCfg; /**< say, coco.data */
    double fTargetFps; /**< 1 fps */
    double fThresh; /**< .24 */
    tfnRaiseAnnCb pfnRaiseAnnCb;
    int nVideoId;
    int isVideo;
    int nFrameId;
    char* pcNames;
}tDetectorModel;

int run_detector_model(tDetectorModel* apDetectorModel);

inline void free_BBs(tAnnInfo* pBBs)
{
    tAnnInfo* pBB = pBBs;
    tAnnInfo* pBBToDelete = NULL;
    while((pBBToDelete = pBB))
    {
        pBB = pBB->pNext;
        if(pBBToDelete->pcClassName)
            free(pBBToDelete->pcClassName);
        free(pBBToDelete);
    }
}


#endif
