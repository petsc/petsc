static const char help[] = "STREAM benchmark specialized for SSE2\n\\n";

/* Note: this file has been modified significantly from its original version */
#include <emmintrin.h>
#include <petsctime.h>
#include <petscsys.h>
#if defined(HAVE_NUMA)
#include <numa.h>
#endif
#include <float.h>

#if !defined(SSE2)
#  define SSE2 1
#endif
#if !defined(__SSE2__)
#  error SSE2 instruction set is not enabled, try adding -march=native to CFLAGS or disable by adding -DSSE2=0
#endif
#if !defined(PREFETCH_NTA) /* Use software prefetch and set non-temporal policy so that lines evicted from L1D will not subsequently reside in L2 or L3. */
#  define PREFETCH_NTA 1
#endif
#if !defined(STATIC_ALLOC) /* Statically allocate the vectors. Most platforms do not find physical pages when memory is allocated, therefore the faulting strategy still affects performance. */
#  define STATIC_ALLOC 0
#endif
#if !defined(FAULT_TOGETHER) /* Faults all three vectors together which usually interleaves DRAM pages in physical memory. */
#  define FAULT_TOGETHER 0
#endif
#if !defined(USE_MEMCPY) /* Literally call memcpy(3) for the COPY benchmark. Some compilers detect the unoptimized loop as memcpy and call this anyway. */
#  define USE_MEMCPY 0
#endif

/*
 * Program: Stream
 * Programmer: Joe R. Zagar
 * Revision: 4.0-BETA, October 24, 1995
 * Original code developed by John D. McCalpin
 *
 * This program measures memory transfer rates in MB/s for simple
 * computational kernels coded in C.  These numbers reveal the quality
 * of code generation for simple uncacheable kernels as well as showing
 * the cost of floating-point operations relative to memory accesses.
 *
 * INSTRUCTIONS:
 *
 *       1) Stream requires a good bit of memory to run.  Adjust the
 *          value of 'N' (below) to give a 'timing calibration' of
 *          at least 20 clock-ticks.  This will provide rate estimates
 *          that should be good to about 5% precision.
 */

# define N      4000000
# define NTIMES     100
# define OFFSET       0

# define HLINE "-------------------------------------------------------------\n"

# if !defined(MIN)
# define MIN(x,y) ((x)<(y) ? (x) : (y))
# endif
# if !defined(MAX)
# define MAX(x,y) ((x)>(y) ? (x) : (y))
# endif

#if STATIC_ALLOC
double a[N+OFFSET],b[N+OFFSET],c[N+OFFSET];
#endif

static int checktick(void);
static double Second(void);

int main(int argc,char *argv[])
{
  const char     *label[4] = {"Copy", "Scale","Add", "Triad"};
  const double   bytes[4]  = {2 * sizeof(double) * N,
                            2 * sizeof(double) * N,
                            3 * sizeof(double) * N,
                            3 * sizeof(double) * N};
  double         rmstime[4] = {0},maxtime[4] = {0},mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};
  int            quantum;
  int            BytesPerWord,j,k,size;
  PetscInt       node = -1;
  double         scalar, t, times[4][NTIMES];
  PetscErrorCode ierr;
#if !STATIC_ALLOC
  double         *PETSC_RESTRICT a,*PETSC_RESTRICT b,*PETSC_RESTRICT c;
#endif

  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-node",&node,NULL);CHKERRQ(ierr);
  /* --- SETUP --- determine precision and check timing --- */

  PetscPrintf(PETSC_COMM_WORLD,HLINE);
  BytesPerWord = sizeof(double);
  PetscPrintf(PETSC_COMM_WORLD,"This system uses %d bytes per DOUBLE PRECISION word.\n",BytesPerWord);

  PetscPrintf(PETSC_COMM_WORLD,HLINE);
  PetscPrintf(PETSC_COMM_WORLD,"Array size = %d, Offset = %d\n", N, OFFSET);
  PetscPrintf(PETSC_COMM_WORLD,"Total memory required = %.1f MB per process.\n",(3 * N * BytesPerWord) / 1048576.0);
  PetscPrintf(PETSC_COMM_WORLD,"Each test is run %d times, but only\n", NTIMES);
  PetscPrintf(PETSC_COMM_WORLD,"the *best* time for each is used.\n");

  /* Get initial value for system clock. */

#if !STATIC_ALLOC
  if (node == -1) {
    posix_memalign((void**)&a,64,N*sizeof(double));
    posix_memalign((void**)&b,64,N*sizeof(double));
    posix_memalign((void**)&c,64,N*sizeof(double));
  } else if (node == -2) {
    a = malloc(N*sizeof(double));
    b = malloc(N*sizeof(double));
    c = malloc(N*sizeof(double));
#if defined(HAVE_NUMA)
  } else {
    a = numa_alloc_onnode(N*sizeof(double),node);
    b = numa_alloc_onnode(N*sizeof(double),node);
    c = numa_alloc_onnode(N*sizeof(double),node);
#endif
  }
#endif
#if FAULT_TOGETHER
  for (j=0; j<N; j++) {
    a[j] = 1.0;
    b[j] = 2.0;
    c[j] = 0.0;
  }
#else
  for (j=0; j<N; j++) a[j] = 1.0;
  for (j=0; j<N; j++) b[j] = 2.0;
  for (j=0; j<N; j++) c[j] = 0.0;
#endif

  PetscPrintf(PETSC_COMM_WORLD,HLINE);

  if  ((quantum = checktick()) >= 1) PetscPrintf(PETSC_COMM_WORLD,"Your clock granularity/precision appears to be %d microseconds.\n", quantum);
  else PetscPrintf(PETSC_COMM_WORLD,"Your clock granularity appears to be less than one microsecond.\n");

  t = Second();
  for (j = 0; j < N; j++) a[j] = 2.0E0 * a[j];
  t = 1.0E6 * (Second() - t);

  PetscPrintf(PETSC_COMM_WORLD,"Each test below will take on the order of %d microseconds.\n", (int) t);
  PetscPrintf(PETSC_COMM_WORLD,"   (= %d clock ticks)\n", (int) (t/quantum));
  PetscPrintf(PETSC_COMM_WORLD,"Increase the size of the arrays if this shows that\n");
  PetscPrintf(PETSC_COMM_WORLD,"you are not getting at least 20 clock ticks per test.\n");

  PetscPrintf(PETSC_COMM_WORLD,HLINE);

  PetscPrintf(PETSC_COMM_WORLD,"WARNING -- The above is only a rough guideline.\n");
  PetscPrintf(PETSC_COMM_WORLD,"For best results, please be sure you know the\n");
  PetscPrintf(PETSC_COMM_WORLD,"precision of your system timer.\n");
  PetscPrintf(PETSC_COMM_WORLD,HLINE);

  /* --- MAIN LOOP --- repeat test cases NTIMES times --- */

  scalar = 3.0;
  for (k=0; k<NTIMES; k++) {
    MPI_Barrier(PETSC_COMM_WORLD);
    /* ### COPY: c <- a ### */
    times[0][k] = Second();
    MPI_Barrier(PETSC_COMM_WORLD);
#if USE_MEMCPY
    memcpy(c,a,N*sizeof(double));
#elif SSE2
    for (j=0; j<N; j+=8) {
      _mm_stream_pd(c+j+0,_mm_load_pd(a+j+0));
      _mm_stream_pd(c+j+2,_mm_load_pd(a+j+2));
      _mm_stream_pd(c+j+4,_mm_load_pd(a+j+4));
      _mm_stream_pd(c+j+6,_mm_load_pd(a+j+6));
#  if PREFETCH_NTA
      _mm_prefetch(a+j+64,_MM_HINT_NTA);
#  endif
    }
#else
    for (j=0; j<N; j++) c[j] = a[j];
#endif
    MPI_Barrier(PETSC_COMM_WORLD);
    times[0][k] = Second() - times[0][k];

    /* ### SCALE: b <- scalar * c ### */
    times[1][k] = Second();
    MPI_Barrier(PETSC_COMM_WORLD);
#if SSE2
    {
      __m128d scalar2 = _mm_set1_pd(scalar);
      for (j=0; j<N; j+=8) {
        _mm_stream_pd(b+j+0,_mm_mul_pd(scalar2,_mm_load_pd(c+j+0)));
        _mm_stream_pd(b+j+2,_mm_mul_pd(scalar2,_mm_load_pd(c+j+2)));
        _mm_stream_pd(b+j+4,_mm_mul_pd(scalar2,_mm_load_pd(c+j+4)));
        _mm_stream_pd(b+j+6,_mm_mul_pd(scalar2,_mm_load_pd(c+j+6)));
#  if PREFETCH_NTA
        _mm_prefetch(c+j+64,_MM_HINT_NTA);
#  endif
      }
    }
#else
    for (j=0; j<N; j++) b[j] = scalar*c[j];
#endif
    MPI_Barrier(PETSC_COMM_WORLD);
    times[1][k] = Second() - times[1][k];

    /* ### ADD: c <- a + b ### */
    times[2][k] = Second();
    MPI_Barrier(PETSC_COMM_WORLD);
#if SSE2
    {
      for (j=0; j<N; j+=8) {
        _mm_stream_pd(c+j+0,_mm_add_pd(_mm_load_pd(a+j+0),_mm_load_pd(b+j+0)));
        _mm_stream_pd(c+j+2,_mm_add_pd(_mm_load_pd(a+j+2),_mm_load_pd(b+j+2)));
        _mm_stream_pd(c+j+4,_mm_add_pd(_mm_load_pd(a+j+4),_mm_load_pd(b+j+4)));
        _mm_stream_pd(c+j+6,_mm_add_pd(_mm_load_pd(a+j+6),_mm_load_pd(b+j+6)));
#  if PREFETCH_NTA
        _mm_prefetch(a+j+64,_MM_HINT_NTA);
        _mm_prefetch(b+j+64,_MM_HINT_NTA);
#  endif
      }
    }
#else
    for (j=0; j<N; j++) c[j] = a[j]+b[j];
#endif
    MPI_Barrier(PETSC_COMM_WORLD);
    times[2][k] = Second() - times[2][k];

    /* ### TRIAD: a <- b + scalar * c ### */
    times[3][k] = Second();
    MPI_Barrier(PETSC_COMM_WORLD);
#if SSE2
    {
      __m128d scalar2 = _mm_set1_pd(scalar);
      for (j=0; j<N; j+=8) {
        _mm_stream_pd(a+j+0,_mm_add_pd(_mm_load_pd(b+j+0),_mm_mul_pd(scalar2,_mm_load_pd(c+j+0))));
        _mm_stream_pd(a+j+2,_mm_add_pd(_mm_load_pd(b+j+2),_mm_mul_pd(scalar2,_mm_load_pd(c+j+2))));
        _mm_stream_pd(a+j+4,_mm_add_pd(_mm_load_pd(b+j+4),_mm_mul_pd(scalar2,_mm_load_pd(c+j+4))));
        _mm_stream_pd(a+j+6,_mm_add_pd(_mm_load_pd(b+j+6),_mm_mul_pd(scalar2,_mm_load_pd(c+j+6))));
#  if PREFETCH_NTA
        _mm_prefetch(b+j+64,_MM_HINT_NTA);
        _mm_prefetch(c+j+64,_MM_HINT_NTA);
#  endif
      }
    }
#else
    for (j=0; j<N; j++) a[j] = b[j]+scalar*c[j];
#endif
    MPI_Barrier(PETSC_COMM_WORLD);
    times[3][k] = Second() - times[3][k];
  }

  /* --- SUMMARY --- */

  for (k=0; k<NTIMES; k++)
    for (j=0; j<4; j++) {
      rmstime[j] = rmstime[j] + (times[j][k] * times[j][k]);
      mintime[j] = MIN(mintime[j], times[j][k]);
      maxtime[j] = MAX(maxtime[j], times[j][k]);
    }

  PetscPrintf(PETSC_COMM_WORLD,"%8s:  %11s  %11s  %11s  %11s  %11s\n","Function","Rate (MB/s)","Total (MB/s)","RMS time","Min time","Max time");
  for (j=0; j<4; j++) {
    rmstime[j] = sqrt(rmstime[j]/(double)NTIMES);
    PetscPrintf(PETSC_COMM_WORLD,"%8s: %11.4f  %11.4f  %11.4f  %11.4f  %11.4f\n", label[j], 1.0e-06*bytes[j]/mintime[j], size*1.0e-06*bytes[j]/mintime[j], rmstime[j], mintime[j], maxtime[j]);
  }
  ierr = PetscFinalize();
  return ierr;
}

static double Second()
{
  double t;
  PetscTime(&t);
  return t;
}

#define M 20
static int checktick(void)
{
  int    i, minDelta, Delta;
  double t1, t2, timesfound[M];

  /*  Collect a sequence of M unique time values from the system. */

  for (i = 0; i < M; i++) {
    t1 = Second();
    while ((t2 = Second()) - t1 < 1.0E-6) {
    }
    timesfound[i] = t1 = t2;
  }

  /*
   * Determine the minimum difference between these M values.
   * This result will be our estimate (in microseconds) for the
   * clock granularity.
   */

  minDelta = 1000000;
  for (i = 1; i < M; i++) {
    Delta    = (int)(1.0E6 * (timesfound[i]-timesfound[i-1]));
    minDelta = MIN(minDelta, MAX(Delta,0));
  }

  return(minDelta);
}
