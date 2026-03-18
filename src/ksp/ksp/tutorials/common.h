#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <sys/time.h>

//#include "utils.h"
#include "omp.h"

#ifndef MAT_VAL_TYPE
#define MAT_VAL_TYPE double
#endif

#ifndef MAT_VAL_LOW_TYPE
#define MAT_VAL_LOW_TYPE float
#endif

#ifndef BENCH_REPEAT
#define BENCH_REPEAT 1000
#endif

#ifndef WARMUP_NUM
#define WARMUP_NUM 200
#endif


#ifndef MAT_PTR_TYPE
#define MAT_PTR_TYPE int
#endif

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef WARP_PER_BLOCK
#define WARP_PER_BLOCK 2
#endif

#ifndef BLOCK_SIZE
#define BLOCK_SIZE  16
#endif

#ifndef NTHREADS_MAX
#define NTHREADS_MAX 1
#endif

#ifndef COO_NNZ_TH
#define COO_NNZ_TH 12
#endif

#ifndef PREFETCH_SMEM_TH
#define PREFETCH_SMEM_TH 4
#endif

#ifndef num_f
#define num_f 240
#endif

#ifndef num_b
#define num_b 15
#endif

#ifndef FORMAT_CONVERSION
#define FORMAT_CONVERSION 0
#endif
