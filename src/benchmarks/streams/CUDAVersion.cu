/*
  STREAM benchmark implementation in CUDA.

    COPY:       a(i) = b(i)
    SCALE:      a(i) = q*b(i)
    SUM:        a(i) = b(i) + c(i)
    TRIAD:      a(i) = b(i) + q*c(i)

  It measures the memory system on the device.
  The implementation is in double precision with a single option.

  Code based on the code developed by John D. McCalpin
  http://www.cs.virginia.edu/stream/FTP/Code/stream.c

  Written by: Massimiliano Fatica, NVIDIA Corporation
  Modified by: Douglas Enright (dpephd-nvidia@yahoo.com), 1 December 2010
  Extensive Revisions, 4 December 2010
  Modified for PETSc by: Matthew G. Knepley 14 Aug 2011

  User interface motivated by bandwidthTest NVIDIA SDK example.
*/
static char help[] = "Double-Precision STREAM Benchmark implementation in CUDA\n Performs Copy, Scale, Add, and Triad double-precision kernels\n\n";

#include <petscconf.h>
#include <petscsys.h>
#include <petsctime.h>

#define N        10000000
#define NTIMES   10

# ifndef MIN
# define MIN(x,y) ((x)<(y) ? (x) : (y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y) ? (x) : (y))
# endif

const float  flt_eps = 1.192092896e-07f;
const double dbl_eps = 2.2204460492503131e-16;

__global__ void set_array(float *a,  float value, size_t len)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < len) {
    a[idx] = value;
    idx   += blockDim.x * gridDim.x;
  }
}

__global__ void set_array_double(double *a,  double value, size_t len)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < len) {
    a[idx] = value;
    idx   += blockDim.x * gridDim.x;
  }
}

__global__ void STREAM_Copy(float *a, float *b, size_t len)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < len) {
    b[idx] = a[idx];
    idx   += blockDim.x * gridDim.x;
  }
}

__global__ void STREAM_Copy_double(double *a, double *b, size_t len)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < len) {
    b[idx] = a[idx];
    idx   += blockDim.x * gridDim.x;
  }
}

__global__ void STREAM_Copy_Optimized(float *a, float *b, size_t len)
{
  /*
   * Ensure size of thread index space is as large as or greater than
   * vector index space else return.
   */
  if (blockDim.x * gridDim.x < len) return;
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) b[idx] = a[idx];
}

__global__ void STREAM_Copy_Optimized_double(double *a, double *b, size_t len)
{
  /*
   * Ensure size of thread index space is as large as or greater than
   * vector index space else return.
   */
  if (blockDim.x * gridDim.x < len) return;
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) b[idx] = a[idx];
}

__global__ void STREAM_Scale(float *a, float *b, float scale,  size_t len)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < len) {
    b[idx] = scale* a[idx];
    idx   += blockDim.x * gridDim.x;
  }
}

__global__ void STREAM_Scale_double(double *a, double *b, double scale,  size_t len)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < len) {
    b[idx] = scale* a[idx];
    idx   += blockDim.x * gridDim.x;
  }
}

__global__ void STREAM_Scale_Optimized(float *a, float *b, float scale,  size_t len)
{
  /*
   * Ensure size of thread index space is as large as or greater than
   * vector index space else return.
   */
  if (blockDim.x * gridDim.x < len) return;
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) b[idx] = scale* a[idx];
}

__global__ void STREAM_Scale_Optimized_double(double *a, double *b, double scale,  size_t len)
{
  /*
   * Ensure size of thread index space is as large as or greater than
   * vector index space else return.
   */
  if (blockDim.x * gridDim.x < len) return;
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) b[idx] = scale* a[idx];
}

__global__ void STREAM_Add(float *a, float *b, float *c,  size_t len)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < len) {
    c[idx] = a[idx]+b[idx];
    idx   += blockDim.x * gridDim.x;
  }
}

__global__ void STREAM_Add_double(double *a, double *b, double *c,  size_t len)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < len) {
    c[idx] = a[idx]+b[idx];
    idx   += blockDim.x * gridDim.x;
  }
}

__global__ void STREAM_Add_Optimized(float *a, float *b, float *c,  size_t len)
{
  /*
   * Ensure size of thread index space is as large as or greater than
   * vector index space else return.
   */
  if (blockDim.x * gridDim.x < len) return;
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) c[idx] = a[idx]+b[idx];
}

__global__ void STREAM_Add_Optimized_double(double *a, double *b, double *c,  size_t len)
{
  /*
   * Ensure size of thread index space is as large as or greater than
   * vector index space else return.
   */
  if (blockDim.x * gridDim.x < len) return;
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) c[idx] = a[idx]+b[idx];
}

__global__ void STREAM_Triad(float *a, float *b, float *c, float scalar, size_t len)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < len) {
    c[idx] = a[idx]+scalar*b[idx];
    idx   += blockDim.x * gridDim.x;
  }
}

__global__ void STREAM_Triad_double(double *a, double *b, double *c, double scalar, size_t len)
{
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  while (idx < len) {
    c[idx] = a[idx]+scalar*b[idx];
    idx   += blockDim.x * gridDim.x;
  }
}

__global__ void STREAM_Triad_Optimized(float *a, float *b, float *c, float scalar, size_t len)
{
  /*
   * Ensure size of thread index space is as large as or greater than
   * vector index space else return.
   */
  if (blockDim.x * gridDim.x < len) return;
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) c[idx] = a[idx]+scalar*b[idx];
}

__global__ void STREAM_Triad_Optimized_double(double *a, double *b, double *c, double scalar, size_t len)
{
  /*
   * Ensure size of thread index space is as large as or greater than
   * vector index space else return.
   */
  if (blockDim.x * gridDim.x < len) return;
  size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len) c[idx] = a[idx]+scalar*b[idx];
}

/* Host side verification routines */
bool STREAM_Copy_verify(float *a, float *b, size_t len)
{
  size_t idx;
  bool   bDifferent = false;

  for (idx = 0; idx < len && !bDifferent; idx++) {
    float expectedResult     = a[idx];
    float diffResultExpected = (b[idx] - expectedResult);
    float relErrorULPS       = (fabsf(diffResultExpected)/fabsf(expectedResult))/flt_eps;
    /* element-wise relative error determination */
    bDifferent = (relErrorULPS > 2.f);
  }

  return bDifferent;
}

bool STREAM_Copy_verify_double(double *a, double *b, size_t len)
{
  size_t idx;
  bool   bDifferent = false;

  for (idx = 0; idx < len && !bDifferent; idx++) {
    double expectedResult     = a[idx];
    double diffResultExpected = (b[idx] - expectedResult);
    double relErrorULPS       = (fabsf(diffResultExpected)/fabsf(expectedResult))/dbl_eps;
    /* element-wise relative error determination */
    bDifferent = (relErrorULPS > 2.);
  }

  return bDifferent;
}

bool STREAM_Scale_verify(float *a, float *b, float scale, size_t len)
{
  size_t idx;
  bool   bDifferent = false;

  for (idx = 0; idx < len && !bDifferent; idx++) {
    float expectedResult     = scale*a[idx];
    float diffResultExpected = (b[idx] - expectedResult);
    float relErrorULPS       = (fabsf(diffResultExpected)/fabsf(expectedResult))/flt_eps;
    /* element-wise relative error determination */
    bDifferent = (relErrorULPS > 2.f);
  }

  return bDifferent;
}

bool STREAM_Scale_verify_double(double *a, double *b, double scale, size_t len)
{
  size_t idx;
  bool   bDifferent = false;

  for (idx = 0; idx < len && !bDifferent; idx++) {
    double expectedResult     = scale*a[idx];
    double diffResultExpected = (b[idx] - expectedResult);
    double relErrorULPS       = (fabsf(diffResultExpected)/fabsf(expectedResult))/flt_eps;
    /* element-wise relative error determination */
    bDifferent = (relErrorULPS > 2.);
  }

  return bDifferent;
}

bool STREAM_Add_verify(float *a, float *b, float *c, size_t len)
{
  size_t idx;
  bool   bDifferent = false;

  for (idx = 0; idx < len && !bDifferent; idx++) {
    float expectedResult     = a[idx] + b[idx];
    float diffResultExpected = (c[idx] - expectedResult);
    float relErrorULPS       = (fabsf(diffResultExpected)/fabsf(expectedResult))/flt_eps;
    /* element-wise relative error determination */
    bDifferent = (relErrorULPS > 2.f);
  }

  return bDifferent;
}

bool STREAM_Add_verify_double(double *a, double *b, double *c, size_t len)
{
  size_t idx;
  bool   bDifferent = false;

  for (idx = 0; idx < len && !bDifferent; idx++) {
    double expectedResult     = a[idx] + b[idx];
    double diffResultExpected = (c[idx] - expectedResult);
    double relErrorULPS       = (fabsf(diffResultExpected)/fabsf(expectedResult))/flt_eps;
    /* element-wise relative error determination */
    bDifferent = (relErrorULPS > 2.);
  }

  return bDifferent;
}

bool STREAM_Triad_verify(float *a, float *b, float *c, float scalar, size_t len)
{
  size_t idx;
  bool   bDifferent = false;

  for (idx = 0; idx < len && !bDifferent; idx++) {
    float expectedResult     = a[idx] + scalar*b[idx];
    float diffResultExpected = (c[idx] - expectedResult);
    float relErrorULPS       = (fabsf(diffResultExpected)/fabsf(expectedResult))/flt_eps;
    /* element-wise relative error determination */
    bDifferent = (relErrorULPS > 3.f);
  }

  return bDifferent;
}

bool STREAM_Triad_verify_double(double *a, double *b, double *c, double scalar, size_t len)
{
  size_t idx;
  bool   bDifferent = false;

  for (idx = 0; idx < len && !bDifferent; idx++) {
    double expectedResult     = a[idx] + scalar*b[idx];
    double diffResultExpected = (c[idx] - expectedResult);
    double relErrorULPS       = (fabsf(diffResultExpected)/fabsf(expectedResult))/flt_eps;
    /* element-wise relative error determination */
    bDifferent = (relErrorULPS > 3.);
  }

  return bDifferent;
}

/* forward declarations */
PetscErrorCode setupStream(PetscInt device, PetscBool runDouble, PetscBool cpuTiming);
PetscErrorCode runStream(const PetscInt iNumThreadsPerBlock, PetscBool bDontUseGPUTiming);
PetscErrorCode runStreamDouble(const PetscInt iNumThreadsPerBlock, PetscBool bDontUseGPUTiming);
PetscErrorCode printResultsReadable(float times[][NTIMES], size_t);

int main(int argc, char *argv[])
{
  PetscInt       device    = 0;
  PetscBool      runDouble = PETSC_TRUE;
  const PetscBool cpuTiming = PETSC_TRUE; // must be true
  PetscErrorCode ierr;

  ierr = cudaSetDeviceFlags(cudaDeviceBlockingSync);CHKERRQ(ierr);

  ierr = PetscInitialize(&argc, &argv, 0, help);if (ierr) return ierr;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "STREAM Benchmark Options", "STREAM");CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-device", "Specify the CUDA device to be used", "STREAM", device, &device, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-double",    "Also run double precision tests",   "STREAM", runDouble, &runDouble, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = setupStream(device, runDouble, cpuTiming);
  if (ierr) {
    ierr = PetscPrintf(PETSC_COMM_SELF, "\n[streamBenchmark] - results:\t%s\n\n", (ierr == 0) ? "PASSES" : "FAILED");CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  return ierr;
}

///////////////////////////////////////////////////////////////////////////////
//Run the appropriate tests
///////////////////////////////////////////////////////////////////////////////
PetscErrorCode setupStream(PetscInt deviceNum, PetscBool runDouble, PetscBool cpuTiming)
{
  PetscInt       iNumThreadsPerBlock = 128;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Check device
  {
    int deviceCount;

    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
      ierr = PetscPrintf(PETSC_COMM_SELF, "!!!!!No devices found!!!!!\n");CHKERRQ(ierr);
      return -1000;
    }

    if (deviceNum >= deviceCount || deviceNum < 0) {
      ierr      = PetscPrintf(PETSC_COMM_SELF, "\n!!!!!Invalid GPU number %d given hence default gpu %d will be used !!!!!\n", deviceNum, 0);CHKERRQ(ierr);
      deviceNum = 0;
    }
  }

  cudaSetDevice(deviceNum);
  // ierr = PetscPrintf(PETSC_COMM_SELF, "Running on...\n\n");CHKERRQ(ierr);
  cudaDeviceProp deviceProp;
  if (cudaGetDeviceProperties(&deviceProp, deviceNum) != cudaSuccess) {
    ierr = PetscPrintf(PETSC_COMM_SELF, " Unable to determine device %d properties, exiting\n");CHKERRQ(ierr);
    return -1;
  }

  if (runDouble && deviceProp.major == 1 && deviceProp.minor < 3) {
    ierr = PetscPrintf(PETSC_COMM_SELF, " Unable to run double-precision STREAM benchmark on a compute capability GPU less than 1.3\n");CHKERRQ(ierr);
    return -1;
  }
  if (deviceProp.major == 2 && deviceProp.minor == 1) iNumThreadsPerBlock = 192; /* GF104 architecture / 48 CUDA Cores per MP */
  else iNumThreadsPerBlock = 128; /* GF100 architecture / 32 CUDA Cores per MP */

  if (runDouble) {
    ierr = runStreamDouble(iNumThreadsPerBlock, cpuTiming);CHKERRQ(ierr);
  } else {
    ierr = runStream(iNumThreadsPerBlock, cpuTiming);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

///////////////////////////////////////////////////////////////////////////
// runStream
///////////////////////////////////////////////////////////////////////////
PetscErrorCode runStream(const PetscInt iNumThreadsPerBlock, PetscBool bDontUseGPUTiming)
{
  float          *d_a, *d_b, *d_c;
  int            k;
  float          times[8][NTIMES];
  float          scalar;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Allocate memory on device */
  ierr = cudaMalloc((void**)&d_a, sizeof(float)*N);CHKERRQ(ierr);
  ierr = cudaMalloc((void**)&d_b, sizeof(float)*N);CHKERRQ(ierr);
  ierr = cudaMalloc((void**)&d_c, sizeof(float)*N);CHKERRQ(ierr);

  /* Compute execution configuration */

  dim3 dimBlock(iNumThreadsPerBlock); /* (iNumThreadsPerBlock,1,1) */
  dim3 dimGrid(N/dimBlock.x); /* (N/dimBlock.x,1,1) */
  if (N % dimBlock.x != 0) dimGrid.x+=1;

  /* Initialize memory on the device */
  set_array<<<dimGrid,dimBlock>>>(d_a, 2.f, N);
  set_array<<<dimGrid,dimBlock>>>(d_b, .5f, N);
  set_array<<<dimGrid,dimBlock>>>(d_c, .5f, N);

  /* --- MAIN LOOP --- repeat test cases NTIMES times --- */
  PetscLogDouble cpuTimer = 0.0;

  scalar=3.0f;
  for (k = 0; k < NTIMES; ++k) {
    PetscTimeSubtract(&cpuTimer);
    STREAM_Copy<<<dimGrid,dimBlock>>>(d_a, d_c, N);
    cudaStreamSynchronize(NULL);
    ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
    PetscTimeAdd(&cpuTimer);
    if (bDontUseGPUTiming) times[0][k] = cpuTimer*1.e3; // millisec

    cpuTimer = 0.0;
    PetscTimeSubtract(&cpuTimer);
    STREAM_Copy_Optimized<<<dimGrid,dimBlock>>>(d_a, d_c, N);
    cudaStreamSynchronize(NULL);
    ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
    //get the total elapsed time in ms
    PetscTimeAdd(&cpuTimer);
    if (bDontUseGPUTiming) times[1][k] = cpuTimer*1.e3;

    cpuTimer = 0.0;
    PetscTimeSubtract(&cpuTimer);
    STREAM_Scale<<<dimGrid,dimBlock>>>(d_b, d_c, scalar,  N);
    cudaStreamSynchronize(NULL);
    ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
    //get the total elapsed time in ms
    PetscTimeAdd(&cpuTimer);
    if (bDontUseGPUTiming) times[2][k] = cpuTimer*1.e3;

    cpuTimer = 0.0;
    PetscTimeSubtract(&cpuTimer);
    STREAM_Scale_Optimized<<<dimGrid,dimBlock>>>(d_b, d_c, scalar,  N);
    cudaStreamSynchronize(NULL);
    ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
    //get the total elapsed time in ms
    PetscTimeAdd(&cpuTimer);
    if (bDontUseGPUTiming) times[3][k] = cpuTimer*1.e3;

    cpuTimer = 0.0;
    PetscTimeSubtract(&cpuTimer);
    // ierr = cudaEventRecord(start, 0);CHKERRQ(ierr);
    STREAM_Add<<<dimGrid,dimBlock>>>(d_a, d_b, d_c,  N);
    cudaStreamSynchronize(NULL);
    ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);    // ierr = cudaEventRecord(stop, 0);CHKERRQ(ierr);
    // ierr = cudaEventSynchronize(stop);CHKERRQ(ierr);
    //get the total elapsed time in ms
    PetscTimeAdd(&cpuTimer);
    if (bDontUseGPUTiming) times[4][k] = cpuTimer*1.e3;
    else {
      // ierr = cudaEventElapsedTime(&times[4][k], start, stop);CHKERRQ(ierr);
    }

    cpuTimer = 0.0;
    PetscTimeSubtract(&cpuTimer);
    STREAM_Add_Optimized<<<dimGrid,dimBlock>>>(d_a, d_b, d_c,  N);
    cudaStreamSynchronize(NULL);
    ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
    //get the total elapsed time in ms
    PetscTimeAdd(&cpuTimer);
    if (bDontUseGPUTiming) times[5][k] = cpuTimer*1.e3;

    cpuTimer = 0.0;
    PetscTimeSubtract(&cpuTimer);
    STREAM_Triad<<<dimGrid,dimBlock>>>(d_b, d_c, d_a, scalar,  N);
    cudaStreamSynchronize(NULL);
    ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
    //get the total elapsed time in ms
    PetscTimeAdd(&cpuTimer);
    if (bDontUseGPUTiming) times[6][k] = cpuTimer*1.e3;

    cpuTimer = 0.0;
    PetscTimeSubtract(&cpuTimer);
    STREAM_Triad_Optimized<<<dimGrid,dimBlock>>>(d_b, d_c, d_a, scalar,  N);
    cudaStreamSynchronize(NULL);
    ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
    //get the total elapsed time in ms
    PetscTimeAdd(&cpuTimer);
    if (bDontUseGPUTiming) times[7][k] = cpuTimer*1.e3;
  }

  if (1) { /* verify kernels */
  float *h_a, *h_b, *h_c;
  bool  errorSTREAMkernel = true;

  if ((h_a = (float*)calloc(N, sizeof(float))) == (float*)NULL) {
    printf("Unable to allocate array h_a, exiting ...\n");
    exit(1);
  }
  if ((h_b = (float*)calloc(N, sizeof(float))) == (float*)NULL) {
    printf("Unable to allocate array h_b, exiting ...\n");
    exit(1);
  }

  if ((h_c = (float*)calloc(N, sizeof(float))) == (float*)NULL) {
    printf("Unalbe to allocate array h_c, exiting ...\n");
    exit(1);
  }

  /*
   * perform kernel, copy device memory into host memory and verify each
   * device kernel output
   */

  /* Initialize memory on the device */
  set_array<<<dimGrid,dimBlock>>>(d_a, 2.f, N);
  set_array<<<dimGrid,dimBlock>>>(d_b, .5f, N);
  set_array<<<dimGrid,dimBlock>>>(d_c, .5f, N);

  STREAM_Copy<<<dimGrid,dimBlock>>>(d_a, d_c, N);
  ierr              = cudaMemcpy(h_a, d_a, sizeof(float) * N, cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  ierr              = cudaMemcpy(h_c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  errorSTREAMkernel = STREAM_Copy_verify(h_a, h_c, N);
  if (errorSTREAMkernel) {
    ierr = PetscPrintf(PETSC_COMM_SELF, " device STREAM_Copy:\t\tError detected in device STREAM_Copy, exiting\n");CHKERRQ(ierr);
    exit(-2000);
  }

  /* Initialize memory on the device */
  set_array<<<dimGrid,dimBlock>>>(d_a, 2.f, N);
  set_array<<<dimGrid,dimBlock>>>(d_b, .5f, N);
  set_array<<<dimGrid,dimBlock>>>(d_c, .5f, N);

  STREAM_Copy_Optimized<<<dimGrid,dimBlock>>>(d_a, d_c, N);
  ierr              = cudaMemcpy(h_a, d_a, sizeof(float) * N, cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  ierr              = cudaMemcpy(h_c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  errorSTREAMkernel = STREAM_Copy_verify(h_a, h_c, N);
  if (errorSTREAMkernel) {
    ierr = PetscPrintf(PETSC_COMM_SELF, " device STREAM_Copy_Optimized:\tError detected in device STREAM_Copy_Optimized, exiting\n");CHKERRQ(ierr);
    exit(-3000);
  }

  /* Initialize memory on the device */
  set_array<<<dimGrid,dimBlock>>>(d_a, 2.f, N);
  set_array<<<dimGrid,dimBlock>>>(d_b, .5f, N);
  set_array<<<dimGrid,dimBlock>>>(d_c, .5f, N);

  STREAM_Scale<<<dimGrid,dimBlock>>>(d_b, d_c, scalar, N);
  ierr              = cudaMemcpy(h_b, d_b, sizeof(float) * N, cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  ierr              = cudaMemcpy(h_c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  errorSTREAMkernel = STREAM_Scale_verify(h_b, h_c, scalar, N);
  if (errorSTREAMkernel) {
    ierr = PetscPrintf(PETSC_COMM_SELF, " device STREAM_Scale:\t\tError detected in device STREAM_Scale, exiting\n");CHKERRQ(ierr);
    exit(-4000);
  }

  /* Initialize memory on the device */
  set_array<<<dimGrid,dimBlock>>>(d_a, 2.f, N);
  set_array<<<dimGrid,dimBlock>>>(d_b, .5f, N);
  set_array<<<dimGrid,dimBlock>>>(d_c, .5f, N);

  STREAM_Add<<<dimGrid,dimBlock>>>(d_a, d_b, d_c, N);
  ierr              = cudaMemcpy(h_a, d_a, sizeof(float) * N, cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  ierr              = cudaMemcpy(h_b, d_b, sizeof(float) * N, cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  ierr              = cudaMemcpy(h_c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  errorSTREAMkernel = STREAM_Add_verify(h_a, h_b, h_c, N);
  if (errorSTREAMkernel) {
    ierr = PetscPrintf(PETSC_COMM_SELF, " device STREAM_Add:\t\tError detected in device STREAM_Add, exiting\n");CHKERRQ(ierr);
    exit(-5000);
  }

  /* Initialize memory on the device */
  set_array<<<dimGrid,dimBlock>>>(d_a, 2.f, N);
  set_array<<<dimGrid,dimBlock>>>(d_b, .5f, N);
  set_array<<<dimGrid,dimBlock>>>(d_c, .5f, N);

  STREAM_Triad<<<dimGrid,dimBlock>>>(d_b, d_c, d_a, scalar, N);
  ierr              = cudaMemcpy(h_a, d_a, sizeof(float) * N, cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  ierr              = cudaMemcpy(h_b, d_b, sizeof(float) * N, cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  ierr              = cudaMemcpy(h_c, d_c, sizeof(float) * N, cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  errorSTREAMkernel = STREAM_Triad_verify(h_b, h_c, h_a, scalar, N);
  if (errorSTREAMkernel) {
    ierr = PetscPrintf(PETSC_COMM_SELF, " device STREAM_Triad:\t\tError detected in device STREAM_Triad, exiting\n");CHKERRQ(ierr);
    exit(-6000);
  }

  free(h_a);
  free(h_b);
  free(h_c);
  }
  /* continue from here */
  printResultsReadable(times, sizeof(float));

  /* Free memory on device */
  ierr = cudaFree(d_a);CHKERRQ(ierr);
  ierr = cudaFree(d_b);CHKERRQ(ierr);
  ierr = cudaFree(d_c);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode runStreamDouble(const PetscInt iNumThreadsPerBlock, PetscBool bDontUseGPUTiming)
{
  double         *d_a, *d_b, *d_c;
  int            k;
  float          times[8][NTIMES];
  double         scalar;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Allocate memory on device */
  ierr = cudaMalloc((void**)&d_a, sizeof(double)*N);CHKERRQ(ierr);
  ierr = cudaMalloc((void**)&d_b, sizeof(double)*N);CHKERRQ(ierr);
  ierr = cudaMalloc((void**)&d_c, sizeof(double)*N);CHKERRQ(ierr);

  /* Compute execution configuration */

  dim3 dimBlock(iNumThreadsPerBlock); /* (iNumThreadsPerBlock,1,1) */
  dim3 dimGrid(N/dimBlock.x); /* (N/dimBlock.x,1,1) */
  if (N % dimBlock.x != 0) dimGrid.x+=1;

  /* Initialize memory on the device */
  set_array_double<<<dimGrid,dimBlock>>>(d_a, 2., N);
  set_array_double<<<dimGrid,dimBlock>>>(d_b, .5, N);
  set_array_double<<<dimGrid,dimBlock>>>(d_c, .5, N);

  /* --- MAIN LOOP --- repeat test cases NTIMES times --- */
  PetscLogDouble cpuTimer = 0.0;

  scalar=3.0;
  for (k = 0; k < NTIMES; ++k) {
    PetscTimeSubtract(&cpuTimer);
    STREAM_Copy_double<<<dimGrid,dimBlock>>>(d_a, d_c, N);
    cudaStreamSynchronize(NULL);
    ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
    //get the total elapsed time in ms
    if (bDontUseGPUTiming) {
      PetscTimeAdd(&cpuTimer);
      times[0][k] = cpuTimer*1.e3;
    }

    cpuTimer = 0.0;
    PetscTimeSubtract(&cpuTimer);
    STREAM_Copy_Optimized_double<<<dimGrid,dimBlock>>>(d_a, d_c, N);
    cudaStreamSynchronize(NULL);
    ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
    //get the total elapsed time in ms
    if (bDontUseGPUTiming) {
      PetscTimeAdd(&cpuTimer);
      times[1][k] = cpuTimer*1.e3;
    }

    cpuTimer = 0.0;
    PetscTimeSubtract(&cpuTimer);
    STREAM_Scale_double<<<dimGrid,dimBlock>>>(d_b, d_c, scalar,  N);
    cudaStreamSynchronize(NULL);
    ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
    //get the total elapsed time in ms
    PetscTimeAdd(&cpuTimer);
    if (bDontUseGPUTiming) times[2][k] = cpuTimer*1.e3;

    cpuTimer = 0.0;
    PetscTimeSubtract(&cpuTimer);
    STREAM_Scale_Optimized_double<<<dimGrid,dimBlock>>>(d_b, d_c, scalar,  N);
    cudaStreamSynchronize(NULL);
    ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
    //get the total elapsed time in ms
    PetscTimeAdd(&cpuTimer);
    if (bDontUseGPUTiming) times[3][k] = cpuTimer*1.e3;

    cpuTimer = 0.0;
    PetscTimeSubtract(&cpuTimer);
    STREAM_Add_double<<<dimGrid,dimBlock>>>(d_a, d_b, d_c,  N);
    cudaStreamSynchronize(NULL);
    ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
    //get the total elapsed time in ms
    PetscTimeAdd(&cpuTimer);
    if (bDontUseGPUTiming) times[4][k] = cpuTimer*1.e3;

    cpuTimer = 0.0;
    PetscTimeSubtract(&cpuTimer);
    STREAM_Add_Optimized_double<<<dimGrid,dimBlock>>>(d_a, d_b, d_c,  N);
    cudaStreamSynchronize(NULL);
    ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
    //get the total elapsed time in ms
    PetscTimeAdd(&cpuTimer);
    if (bDontUseGPUTiming) times[5][k] = cpuTimer*1.e3;

    cpuTimer = 0.0;
    PetscTimeSubtract(&cpuTimer);
    STREAM_Triad_double<<<dimGrid,dimBlock>>>(d_b, d_c, d_a, scalar,  N);
    cudaStreamSynchronize(NULL);
    ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
    //get the total elapsed time in ms
    PetscTimeAdd(&cpuTimer);
    if (bDontUseGPUTiming) times[6][k] = cpuTimer*1.e3;

    cpuTimer = 0.0;
    PetscTimeSubtract(&cpuTimer);
    STREAM_Triad_Optimized_double<<<dimGrid,dimBlock>>>(d_b, d_c, d_a, scalar,  N);
    cudaStreamSynchronize(NULL);
    ierr = MPI_Barrier(MPI_COMM_WORLD);CHKERRQ(ierr);
    //get the total elapsed time in ms
    PetscTimeAdd(&cpuTimer);
    if (bDontUseGPUTiming) times[7][k] = cpuTimer*1.e3;
  }

  if (1) { /* verify kernels */
  double *h_a, *h_b, *h_c;
  bool   errorSTREAMkernel = true;

  if ((h_a = (double*)calloc(N, sizeof(double))) == (double*)NULL) {
    printf("Unable to allocate array h_a, exiting ...\n");
    exit(1);
  }
  if ((h_b = (double*)calloc(N, sizeof(double))) == (double*)NULL) {
    printf("Unable to allocate array h_b, exiting ...\n");
    exit(1);
  }

  if ((h_c = (double*)calloc(N, sizeof(double))) == (double*)NULL) {
    printf("Unalbe to allocate array h_c, exiting ...\n");
    exit(1);
  }

  /*
   * perform kernel, copy device memory into host memory and verify each
   * device kernel output
   */

  /* Initialize memory on the device */
  set_array_double<<<dimGrid,dimBlock>>>(d_a, 2., N);
  set_array_double<<<dimGrid,dimBlock>>>(d_b, .5, N);
  set_array_double<<<dimGrid,dimBlock>>>(d_c, .5, N);

  STREAM_Copy_double<<<dimGrid,dimBlock>>>(d_a, d_c, N);
  ierr              = cudaMemcpy(h_a, d_a, sizeof(double) * N, cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  ierr              = cudaMemcpy(h_c, d_c, sizeof(double) * N, cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  errorSTREAMkernel = STREAM_Copy_verify_double(h_a, h_c, N);
  if (errorSTREAMkernel) {
    ierr = PetscPrintf(PETSC_COMM_SELF, " device STREAM_Copy:\t\tError detected in device STREAM_Copy, exiting\n");CHKERRQ(ierr);
    exit(-2000);
  }

  /* Initialize memory on the device */
  set_array_double<<<dimGrid,dimBlock>>>(d_a, 2., N);
  set_array_double<<<dimGrid,dimBlock>>>(d_b, .5, N);
  set_array_double<<<dimGrid,dimBlock>>>(d_c, .5, N);

  STREAM_Copy_Optimized_double<<<dimGrid,dimBlock>>>(d_a, d_c, N);
  ierr              = cudaMemcpy(h_a, d_a, sizeof(double) * N, cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  ierr              = cudaMemcpy(h_c, d_c, sizeof(double) * N, cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  errorSTREAMkernel = STREAM_Copy_verify_double(h_a, h_c, N);
  if (errorSTREAMkernel) {
    ierr = PetscPrintf(PETSC_COMM_SELF, " device STREAM_Copy_Optimized:\tError detected in device STREAM_Copy_Optimized, exiting\n");CHKERRQ(ierr);
    exit(-3000);
  }

  /* Initialize memory on the device */
  set_array_double<<<dimGrid,dimBlock>>>(d_b, .5, N);
  set_array_double<<<dimGrid,dimBlock>>>(d_c, .5, N);

  STREAM_Scale_double<<<dimGrid,dimBlock>>>(d_b, d_c, scalar, N);
  ierr              = cudaMemcpy(h_b, d_b, sizeof(double) * N, cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  ierr              = cudaMemcpy(h_c, d_c, sizeof(double) * N, cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  errorSTREAMkernel = STREAM_Scale_verify_double(h_b, h_c, scalar, N);
  if (errorSTREAMkernel) {
    ierr = PetscPrintf(PETSC_COMM_SELF, " device STREAM_Scale:\t\tError detected in device STREAM_Scale, exiting\n");CHKERRQ(ierr);
    exit(-4000);
  }

  /* Initialize memory on the device */
  set_array_double<<<dimGrid,dimBlock>>>(d_a, 2., N);
  set_array_double<<<dimGrid,dimBlock>>>(d_b, .5, N);
  set_array_double<<<dimGrid,dimBlock>>>(d_c, .5, N);

  STREAM_Add_double<<<dimGrid,dimBlock>>>(d_a, d_b, d_c, N);
  ierr              = cudaMemcpy(h_a, d_a, sizeof(double) * N, cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  ierr              = cudaMemcpy(h_b, d_b, sizeof(double) * N, cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  ierr              = cudaMemcpy(h_c, d_c, sizeof(double) * N, cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  errorSTREAMkernel = STREAM_Add_verify_double(h_a, h_b, h_c, N);
  if (errorSTREAMkernel) {
    ierr = PetscPrintf(PETSC_COMM_SELF, " device STREAM_Add:\t\tError detected in device STREAM_Add, exiting\n");CHKERRQ(ierr);
    exit(-5000);
  }

  /* Initialize memory on the device */
  set_array_double<<<dimGrid,dimBlock>>>(d_a, 2., N);
  set_array_double<<<dimGrid,dimBlock>>>(d_b, .5, N);
  set_array_double<<<dimGrid,dimBlock>>>(d_c, .5, N);

  STREAM_Triad_double<<<dimGrid,dimBlock>>>(d_b, d_c, d_a, scalar, N);
  ierr              = cudaMemcpy(h_a, d_a, sizeof(double) * N, cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  ierr              = cudaMemcpy(h_b, d_b, sizeof(double) * N, cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  ierr              = cudaMemcpy(h_c, d_c, sizeof(double) * N, cudaMemcpyDeviceToHost);CHKERRQ(ierr);
  errorSTREAMkernel = STREAM_Triad_verify_double(h_b, h_c, h_a, scalar, N);
  if (errorSTREAMkernel) {
    ierr = PetscPrintf(PETSC_COMM_SELF, " device STREAM_Triad:\t\tError detected in device STREAM_Triad, exiting\n");CHKERRQ(ierr);
    exit(-6000);
  }

  free(h_a);
  free(h_b);
  free(h_c);
  }
  /* continue from here */
  printResultsReadable(times,sizeof(double));

  /* Free memory on device */
  ierr = cudaFree(d_a);CHKERRQ(ierr);
  ierr = cudaFree(d_b);CHKERRQ(ierr);
  ierr = cudaFree(d_c);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

///////////////////////////////////////////////////////////////////////////
//Print Results to Screen and File
///////////////////////////////////////////////////////////////////////////
PetscErrorCode printResultsReadable(float times[][NTIMES], const size_t bsize)
{
  PetscErrorCode ierr;
  PetscInt       j, k;
  float          avgtime[8]          = {0., 0., 0., 0., 0., 0., 0., 0.};
  float          maxtime[8]          = {0., 0., 0., 0., 0., 0., 0., 0.};
  float          mintime[8]          = {1e30,1e30,1e30,1e30,1e30,1e30,1e30,1e30};
  // char           *label[8]           = {"Copy:      ", "Copy Opt.: ", "Scale:     ", "Scale Opt: ", "Add:       ", "Add Opt:   ", "Triad:     ", "Triad Opt: "};
  const float    bytes_per_kernel[8] = {
    2. * bsize * N,
    2. * bsize * N,
    2. * bsize * N,
    2. * bsize * N,
    3. * bsize * N,
    3. * bsize * N,
    3. * bsize * N,
    3. * bsize * N
  };
  double         rate,irate;
  int            rank,size;
  PetscFunctionBegin;
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(MPI_COMM_WORLD,&size);CHKERRQ(ierr);
  /* --- SUMMARY --- */
  for (k = 0; k < NTIMES; ++k) {
    for (j = 0; j < 8; ++j) {
      avgtime[j] = avgtime[j] + (1.e-03f * times[j][k]); // millisec --> sec
      mintime[j] = MIN(mintime[j], (1.e-03f * times[j][k]));
      maxtime[j] = MAX(maxtime[j], (1.e-03f * times[j][k]));
    }
  }
  for (j = 0; j < 8; ++j) {
    avgtime[j] = avgtime[j]/(float)(NTIMES-1);
  }
  j = 7;
  irate = 1.0E-06 * bytes_per_kernel[j]/mintime[j];
  ierr = MPI_Reduce(&irate,&rate,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  if (!rank) {
    FILE *fd;
    if (size == 1) {
      printf("%d %11.4f   Rate (MB/s)\n",size, rate);
      fd = fopen("flops","w");
      fprintf(fd,"%g\n",rate);
      fclose(fd);
    } else {
      double prate;
      fd = fopen("flops","r");
      fscanf(fd,"%lg",&prate);
      fclose(fd);
      printf("%d %11.4f   Rate (MB/s) %g \n", size, rate, rate/prate);
    }
  }

  PetscFunctionReturn(0);
}
