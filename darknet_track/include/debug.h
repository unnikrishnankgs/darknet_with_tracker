#ifndef _SRC_DEBUG_H_
#define _SRC_DEBUG_H_
#include <stdio.h>
#ifdef DEBUG
#define LOGD(fmt , ...) \
    printf("[DEBUG %s %s %d] " fmt, __FILE__, __FUNCTION__, __LINE__, ##__VA_ARGS__)

#else
#define LOGD(...)
#endif

// #define VERBOSE in your c/cpp file before #include<debug_e.h> to enable debug logs
#ifdef VERBOSE
#define LOGV(fmt , ...) \
    printf("[INFO %s %s %d] " fmt, __FILE__, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#else
#define LOGV(...)
#endif

//error shall be ON always
#define LOGE(fmt , ...) \
    printf("[ERROR %s %s %d] " fmt, __FILE__, __FUNCTION__, __LINE__, ##__VA_ARGS__)
#endif /**< _SRC_DEBUG_H_ */
