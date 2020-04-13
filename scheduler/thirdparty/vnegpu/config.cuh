#ifndef _CONFIG_CUH
#define _CONFIG_CUH

/*! \file
 *  \brief The configuration of some macros and options.
 */

//LIB for XML.
#define XML_LIB_INC "libs/pugixml/src/pugixml.hpp"

//In case there is not a lib for XML
//#define NO_XML_LIB

//Limit for node and edge variables on graph.
#define GRAPH_MAX_VARIABLES 50

//CUDA Default Block Size in threads
#define CUDA_BLOCK_SIZE 512

//CUDA Default Block Size in threads that used shared memory
#define CUDA_BLOCK_SIZE_SHARED 16

//Limit of algorithms iterations
#define ITERATION_LIMIT 1000

//Default Rank max error
#define RANK_MAX_ERROR 0.00001

//Requere use of -lnvToolsExt on compilation
#ifdef USE_NVTX
#include "nvToolsExt.h"

const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define DEBUG_PUSH_RANGE(name,cid) { \
		int color_id = cid; \
		color_id = color_id%num_colors; \
		nvtxEventAttributes_t eventAttrib = {0}; \
		eventAttrib.version = NVTX_VERSION; \
		eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
		eventAttrib.colorType = NVTX_COLOR_ARGB; \
		eventAttrib.color = colors[color_id]; \
		eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
		eventAttrib.message.ascii = name; \
		nvtxRangePushEx(&eventAttrib); \
}
#define DEBUG_POP_RANGE() nvtxRangePop();
#else
#define DEBUG_PUSH_RANGE(name,cid)
#define DEBUG_POP_RANGE()
#endif

#ifndef __func__
#define __func__ __FUNCTION__
#endif

#pragma warning(disable:4503)

#define CUDA_CHECK() GetCudaErrorMacro(__LINE__,__FILE__,__func__)

void inline GetCudaErrorMacro(int line, const char* file, const char* function){
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		printf("Cuda Error on %s:%d:%s Error: %s\n", file, line, function, cudaGetErrorString(err));
		exit(0);
	}
}


#endif
