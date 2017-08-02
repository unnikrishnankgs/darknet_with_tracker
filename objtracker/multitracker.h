#ifndef __MULTITRACKER_H__
#define __MULTITRACKER_H__
 
#include "darknet.h"
#include "darknet_exp.h"

#ifdef __cplusplus
extern "C" {
#endif

int track_bb_in_frame(tAnnInfo* apBoundingBoxesIn, tFrameInfo* pFBase, tFrameInfo* pFTarg, tAnnInfo** appBoundingBoxesOut);
int tracker_display_frame(tAnnInfo* apBoundingBoxesIn, tFrameInfo* pFBase);

/**
 * This function take 2 BBs list
 * say BB1 and BB2
 * The function shall do an IoU map between the lists
 * and update the BBID in BB2 from BB1 where theres a match  
 */
void assess_iou_trackedBBs_detectedBBs(tAnnInfo* pTrackedBBs,
                tAnnInfo* pDetectedBBs);

#define MAX_POLYGONS 10
typedef struct Vertexx tVertex;
struct Vertexx
{
    int x;
    int y;
    tVertex* pNext;
};

typedef struct PolygonX tPolygon;
struct PolygonX
{
   tVertex* pVs;
   int nVs;
   int nLaneId;
   double fAvgWaitingTime;
   char* pcLaneName;
   tPolygon* pNext;
};

typedef struct
{
   tPolygon* pPolygons;
   int nPolygons;
}tLanesInfo;
tLanesInfo* getLaneInfo(tFrameInfo* pFrame);

inline void display_lanes_info(tLanesInfo* pLanesInfo)
{
    if(pLanesInfo)
    {
        tPolygon* pP = pLanesInfo->pPolygons;
        while(pP)
        {
            /** polygon: list of vertices in order */
            tVertex* pV = pP->pVs;
            printf("Lane ID: %d; vertices:\n", pP->nLaneId);
            while(pV)
            {
                printf("(%d, %d)\n", pV->x, pV->y);
                pV = pV->pNext;
            }
            pP = pP->pNext;
        }
    }
}

int isWithinPolygon(tPolygon* pLane, tVertex* pPoint);


#ifdef __cplusplus
}
#endif

#endif /**< __MULTITRACKER_H__ */
