#ifndef __MULTITRACKER_H__
#define __MULTITRACKER_H__
 
#include "darknet.h"

#include "darknet_exp.h"

#ifdef __cplusplus
extern "C" {
#endif

#define INVALID_LANE_ID (0)

typedef struct Vertexx tVertex;
typedef struct LaneX tLane;
struct Vertexx
{
    int x;
    int y;
    tVertex* pNext;
};

struct LaneX
{
   tVertex* pVs;
   int nVs;
   int nLaneId;
   double fAvgWaitingTime;
   char* pcRoute;
   long long* pnVehicleCount; /**< array of integers - each element would be the count of one type of vehicle: dereferenced as pnVehicleCount[pBB->nClassId] */
   int nTypes;
   double fTotalStayDuration;
   double fAvgStayDuration;
   long long nTotalVehiclesSoFar; /**< updated only when a vehicle move out of the lane */
   tLane* pNext;
};

typedef struct
{
   long long* pnVehicleCount; /**< array of integers - each element would be the count of one type of vehicle: dereferenced as pnVehicleCount[pBB->nClassId] */
   int nTypes;
}tRouteTrafficInfo;

typedef struct
{
   tLane* pLanes;
   int nLanes;
   tRouteTrafficInfo** ppRouteTrafficInfo; /**< 2D array; nLanes X nLanes */
   char** names;
   int nTypes;
}tLanesInfo;

inline tLane* getLaneById(tLanesInfo* pLanesInfo, int nLaneId)
{
    if(!pLanesInfo)
        return NULL;

    tLane* pL = pLanesInfo->pLanes;
    while(pL)
    {
        if(pL->nLaneId == nLaneId)
            return pL;
        pL = pL->pNext;
    }

    return NULL;
}

int track_bb_in_frame(tAnnInfo* apBoundingBoxesIn, tFrameInfo* pFBase, tFrameInfo* pFTarg, tAnnInfo** appBoundingBoxesOut, tLanesInfo* pLanesInfo);
int tracker_display_frame(tAnnInfo* apBoundingBoxesIn, tFrameInfo* pFBase);

/**
 * This function take 2 BBs list
 * say BB1 and BB2
 * The function shall do an IoU map between the lists
 * and update the BBID in BB2 from BB1 where theres a match  
 */
void assess_iou_trackedBBs_detectedBBs(tAnnInfo* pTrackedBBs,
                tAnnInfo* pDetectedBBs);



tLanesInfo* getLaneInfo(tFrameInfo* pFrame, tLanesInfo* pLanesInfo);

inline void display_lanes_info(tLanesInfo* pLanesInfo)
{
    if(pLanesInfo)
    {
        tLane* pP = pLanesInfo->pLanes;
        while(pP)
        {
            /** polygon: list of vertices in order */
            tVertex* pV = pP->pVs;
            printf("Lane %p ID: %d; route:[%s] vertices:\n", pP, pP->nLaneId, pP->pcRoute);
            while(pV)
            {
                printf("(%d, %d)\n", pV->x, pV->y);
                pV = pV->pNext;
            }
            pP = pP->pNext;
        }
    }
}

int isWithinLane(tLane* pLane, tVertex* pPoint);


#ifdef __cplusplus
}
#endif

#endif /**< __MULTITRACKER_H__ */
