#include <opencv2/opencv.hpp>
#define VERBOSE
#include "debug.h"
#define MAX_EMPTY_GRIDS (20)
#define GRID_WIDTH (100)

using namespace cv;
typedef struct
{
    int nHG;
    int nWG;
    int** ppGridM;
    int nStepSize;
}tGrid;
int sca5(Mat src, Mat frame, tGrid* pGrid);

static inline void freeGrid(tGrid* pGrid)
{
    if(pGrid && pGrid->ppGridM)
    {
        for(int i = 0; i < pGrid->nHG; i++)
            free(pGrid->ppGridM[i]);
        free(pGrid->ppGridM);
    }
}

static inline double getMotionLevel(tGrid* pGrid, int x, int y, int w, int h)
{
    double nMotionLevel = 0;

    if(!pGrid || !pGrid->ppGridM || x < 0 || y < 0 || w < 0 || h < 0)
        return nMotionLevel;

    int iStart = y/pGrid->nStepSize;
    int jStart = x/pGrid->nStepSize;
    
    int nPatchGridsW = w/pGrid->nStepSize + 1;
    int nPatchGridsH = h/pGrid->nStepSize + 1;
    int nTotalGrids  = nPatchGridsW * nPatchGridsH;

    int iStop = iStart + (nPatchGridsH);
    int jStop = jStart + (nPatchGridsW);

    LOGV("%d X %d TL:(%d, %d) startG=(%d, %d)[%dX%d] [%d X %d]\n", pGrid->nHG, pGrid->nWG, y, x, iStart, jStart, h, w, iStop, jStop);
        for(int i = iStart; (i < iStop) && (i < pGrid->nHG); i++)
        {
            for(int j = jStart; (j < jStop) && (j < pGrid->nWG); j++)
            {
                nMotionLevel += pGrid->ppGridM[i][j];
            }
        }
        LOGV("\n");

    nMotionLevel = ((nMotionLevel / (nTotalGrids)) * 100.0);
    if(nMotionLevel == 0)
    {
        LOGV("zero nMotionLevel\n");
    }
    return nMotionLevel;
}

static inline void getWindowSize(tGrid* pGrid, int const x, int const y, int& w, int& h)
{
    if(!pGrid || !pGrid->ppGridM)
        return;

    int iStart = y/pGrid->nStepSize;
    int jStart = x/pGrid->nStepSize;

    int iStop = iStart + (h/pGrid->nStepSize + 1);
    int jStop = jStart + (w/pGrid->nStepSize + 1);

    int nMaxW = 0;
    int nMaxH = 0;

    int xTmp = x - 1, yTmp = y;
    
    while(getMotionLevel(pGrid, xTmp--, yTmp, (nMaxW += pGrid->nStepSize), 1))
    {
        
    }

    xTmp = x; yTmp = y - 1;
    while(getMotionLevel(pGrid, xTmp, yTmp--, 1, (nMaxH += pGrid->nStepSize)))
    {
        
    }

    w = nMaxW;
    h = nMaxH;
}
