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
     int fDirection;
     tAnnInfo* pNext;
};

typedef struct
{
     image im;
     int widthStep; /**< do not free pointers in this mem; they're garbage; other attributes will have proper value */
     double fCurrentFrameTimeStamp;
}tFrameInfo;

#ifdef __cplusplus
extern "C" {
#endif

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

inline tAnnInfo* copyBB(tAnnInfo* pBB)
{
    tAnnInfo* pBBN = (tAnnInfo*)malloc(sizeof(tAnnInfo));

    memcpy(pBBN, pBB, sizeof(tAnnInfo));
    pBBN->pcClassName = (char*)malloc(strlen(pBB->pcClassName) + 1);
    strcpy(pBBN->pcClassName, pBB->pcClassName);
    pBBN->pNext = NULL;

    return pBBN;
}

inline void freeBB(tAnnInfo* pBB)
{
    pBB->pNext = NULL;
    free_BBs(pBB);
}

inline double displacement_btw_BBs(tAnnInfo* pBB1, tAnnInfo* pBB2)
{
            double sqX = (double)(pBB2->x - pBB1->x);
            sqX = sqX * sqX;
            double sqY = (double)(pBB2->y - pBB1->y);
            sqY = sqY * sqY;
            double D = sqrt(sqX + sqY);
            return D;
}

inline tAnnInfo* getBBById(tAnnInfo* pBBs, int nBBId)
{
    tAnnInfo* pBB = pBBs;
    while(pBB)
    {
        if(pBB->nBBId == nBBId)
            return pBB;
        pBB = pBB->pNext;
    }

    return NULL;
}

#ifdef __cplusplus
}
#endif
#endif
