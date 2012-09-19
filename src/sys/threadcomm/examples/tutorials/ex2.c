static const char help[] = "STREAM benchmark for pthread implemenentations\n\n";

/*-----------------------------------------------------------------------*/
/* Program: Stream                                                       */
/* Revision: $Id: stream.c,v 5.9 2009/04/11 16:35:00 mccalpin Exp mccalpin $ */
/* Original code developed by John D. McCalpin                           */
/* Programmers: John D. McCalpin                                         */
/*              Joe R. Zagar                                             */
/*                                                                       */
/* This program measures memory transfer rates in MB/s for simple        */
/* computational kernels coded in C.                                     */
/*-----------------------------------------------------------------------*/
/* Copyright 1991-2005: John D. McCalpin                                 */
/*-----------------------------------------------------------------------*/
/* License:                                                              */
/*  1. You are free to use this program and/or to redistribute           */
/*     this program.                                                     */
/*  2. You are free to modify this program for your own use,             */
/*     including commercial use, subject to the publication              */
/*     restrictions in item 3.                                           */
/*  3. You are free to publish results obtained from running this        */
/*     program, or from works that you derive from this program,         */
/*     with the following limitations:                                   */
/*     3a. In order to be referred to as "STREAM benchmark results",     */
/*         published results must be in conformance to the STREAM        */
/*         Run Rules, (briefly reviewed below) published at              */
/*         http://www.cs.virginia.edu/stream/ref.html                    */
/*         and incorporated herein by reference.                         */
/*         As the copyright holder, John McCalpin retains the            */
/*         right to determine conformity with the Run Rules.             */
/*     3b. Results based on modified source code or on runs not in       */
/*         accordance with the STREAM Run Rules must be clearly          */
/*         labelled whenever they are published.  Examples of            */
/*         proper labelling include:                                     */
/*         "tuned STREAM benchmark results"                              */
/*         "based on a variant of the STREAM benchmark code"             */
/*         Other comparable, clear and reasonable labelling is           */
/*         acceptable.                                                   */
/*     3c. Submission of results to the STREAM benchmark web site        */
/*         is encouraged, but not required.                              */
/*  4. Use of this program or creation of derived works based on this    */
/*     program constitutes acceptance of these licensing restrictions.   */
/*  5. Absolutely no warranty is expressed or implied.                   */
/*-----------------------------------------------------------------------*/
# include <stdio.h>
# include <math.h>
# include <limits.h>
# include <float.h>
# include <sys/time.h>
# include <petscconf.h>
# include <petscsys.h>
# include <petscthreadcomm.h>
/* INSTRUCTIONS:
 *
 *      1) Stream requires a good bit of memory to run.  Adjust the
 *          value of 'N' (below) to give a 'timing calibration' of
 *          at least 20 clock-ticks.  This will provide rate estimates
 *          that should be good to about 5% precision.
 */

#ifndef N
#   define N  2000000
#endif
#ifndef NTIMES
#   define NTIMES       50
#endif
#ifndef OFFSET
#   define OFFSET       0
#endif

/*
 *      3) Compile the code with full optimization.  Many compilers
 *         generate unreasonably bad code before the optimizer tightens
 *         things up.  If the results are unreasonably good, on the
 *         other hand, the optimizer might be too smart for me!
 *
 *         Try compiling with:
 *               cc -O stream_omp.c -o stream_omp
 *
 *         This is known to work on Cray, SGI, IBM, and Sun machines.
 *
 *
 *      4) Mail the results to mccalpin@cs.virginia.edu
 *         Be sure to include:
 *              a) computer hardware model number and software revision
 *              b) the compiler flags
 *              c) all of the output from the test case.
 * Thanks!
 *
 */

# define HLINE "-------------------------------------------------------------\n"

# ifndef MIN
# define MIN(x,y) ((x)<(y)?(x):(y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y)?(x):(y))
# endif

#ifndef STATIC_ALLOC
#  define STATIC_ALLOC 1
#endif

#if STATIC_ALLOC
static double   a[N+OFFSET],
                b[N+OFFSET],
                c[N+OFFSET];
#endif

static double   avgtime[4] = {0}, maxtime[4] = {0},
                mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};

static const char     *label[4] = {"Copy:      ", "Scale:     ","Add:       ", "Triad:     "};

static double   bytes[4] = {
    2 * sizeof(double) * N,
    2 * sizeof(double) * N,
    3 * sizeof(double) * N,
    3 * sizeof(double) * N
    };

double mysecond();
void checkSTREAMresults();

#ifndef WITH_PTHREADS
#  define WITH_PTHREADS 1
#endif

#if !STATIC_ALLOC
  double *a, *b, *c;
#endif

#if WITH_PTHREADS
PetscInt nworkThreads;
PetscInt *trstarts;

PetscErrorCode tuned_STREAM_2A_Kernel(PetscInt myrank)
{
  int         i;

  for (i=trstarts[myrank];i<trstarts[myrank+1];i++)
    a[i] = 2.0E0*a[i];

  return(0);
}

PetscErrorCode tuned_STREAM_Initialize_Kernel(PetscInt myrank) {
  int        i;

  for (i=trstarts[myrank];i<trstarts[myrank+1];i++) {
    a[i] = 1.0;
    b[i] = 2.0;
    c[i] = 0.0;
  }
  return(0);
}

PetscErrorCode tuned_STREAM_Copy_Kernel(PetscInt myrank)
{
  int         j;

  for (j=trstarts[myrank]; j<trstarts[myrank+1]; j++)
    c[j] = a[j];
  return(0);
}

PetscErrorCode tuned_STREAM_Scale_Kernel(PetscInt myrank,double *scalarp) {
  double      scalar = *scalarp;
  int         j;

  for (j=trstarts[myrank]; j<trstarts[myrank+1]; j++)
    b[j] = scalar*c[j];
  return(0);
}

PetscErrorCode tuned_STREAM_Add_Kernel(PetscInt myrank) {
  int j;

  for (j=trstarts[myrank]; j<trstarts[myrank+1]; j++)
    c[j] = a[j]+b[j];

  return(0);
}

PetscErrorCode tuned_STREAM_Triad_Kernel(PetscInt myrank,double *scalarp) {
  double      scalar = *scalarp;
  int         j;

  for (j=trstarts[myrank]; j<trstarts[myrank+1]; j++)
    a[j] = b[j]+scalar*c[j];

  return(0);
}
#endif

int main(int argc,char *argv[])
{
  PetscErrorCode ierr;
  int            quantum, checktick();
  int            BytesPerWord;
  int            j, k;
  double         scalar=3.0, t, times[4][NTIMES];

  PetscInitialize(&argc,&argv,0,help);
  /* --- SETUP --- determine precision and check timing --- */

  /*printf(HLINE);
    printf("STREAM version $Revision: 5.9 $\n");
    printf(HLINE); */
  BytesPerWord = sizeof(double);
  printf("This system uses %d bytes per DOUBLE PRECISION word.\n",BytesPerWord);
	
  printf(HLINE);
#ifdef NO_LONG_LONG
  printf("Array size = %d, Offset = %d\n" , N, OFFSET);
#else
  printf("Array size = %llu, Offset = %d\n", (unsigned long long) N, OFFSET);
#endif

  printf("Total memory required = %.1f MB.\n",(3.0 * BytesPerWord) * ( (double) N / 1048576.0));
  printf("Each test is run %d times, but only\n", NTIMES);
  printf("the *best* time for each is used.\n");
	
  printf(HLINE);

#if !STATIC_ALLOC
  a = malloc((N+OFFSET)*sizeof(double));
  b = malloc((N+OFFSET)*sizeof(double));
  c = malloc((N+OFFSET)*sizeof(double));
#endif

#if WITH_PTHREADS
  ierr = PetscThreadCommGetNThreads(PETSC_COMM_WORLD,&nworkThreads);CHKERRQ(ierr);
  ierr = PetscMalloc((nworkThreads+1)*sizeof(PetscInt),&trstarts);CHKERRQ(ierr);
  PetscInt  Q,R,nloc;
  PetscBool S;
  Q = (N+OFFSET)/nworkThreads;
  R = (N+OFFSET) - Q*nworkThreads;
  trstarts[0] = 0;
  for (j=0;j < nworkThreads;j++) {
    S = (PetscBool)(j < R);
    nloc = S?Q+1:Q;
    trstarts[j+1] = trstarts[j]+nloc;
  }
  ierr = PetscThreadCommRunKernel(PETSC_COMM_WORLD,(PetscThreadKernel)tuned_STREAM_Initialize_Kernel,1,&scalar);CHKERRQ(ierr);
  ierr = PetscThreadCommBarrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
# else
    for (j=0; j<N; j++) {
      a[j] = 1.0;
      b[j] = 2.0;
      c[j] = 0.0;
    }
#endif

    /*printf(HLINE);*/

    /* Get initial value for system clock. */
    if  ( (quantum = checktick()) >= 1)
      ;
      /*      printf("Your clock granularity/precision appears to be %d microseconds.\n", quantum); */
    else {
      /*   printf("Your clock granularity appears to be less than one microsecond.\n"); */
      quantum = 1;
    }

    t = mysecond();

#if WITH_PTHREADS
    ierr = PetscThreadCommRunKernel(PETSC_COMM_WORLD,(PetscThreadKernel)tuned_STREAM_2A_Kernel,0);CHKERRQ(ierr);
    ierr = PetscThreadCommBarrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
#else
    for (j = 0; j < N; j++)
      a[j] = 2.0E0 * a[j];
#endif
    t = 1.0E6 * (mysecond() - t);

    /*    printf("Each test below will take on the order of %d microseconds.\n", (int) t  );
    printf("   (= %d clock ticks)\n", (int) (t/quantum) );
    printf("Increase the size of the arrays if this shows that\n");
    printf("you are not getting at least 20 clock ticks per test.\n");

    printf(HLINE);
    */
    /*  --- MAIN LOOP --- repeat test cases NTIMES times --- */

    for (k=0; k<NTIMES; k++) {
      times[0][k] = mysecond();
#if WITH_PTHREADS
      ierr = PetscThreadCommRunKernel(PETSC_COMM_WORLD,(PetscThreadKernel)tuned_STREAM_Copy_Kernel,0);CHKERRQ(ierr);
      ierr = PetscThreadCommBarrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
#else
      for (j=0; j<N; j++)
	c[j] = a[j];
#endif
      times[0][k] = mysecond() - times[0][k];

      times[1][k] = mysecond();
#if WITH_PTHREADS
      ierr = PetscThreadCommRunKernel(PETSC_COMM_WORLD,(PetscThreadKernel)tuned_STREAM_Scale_Kernel,1,&scalar);CHKERRQ(ierr);
      ierr = PetscThreadCommBarrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
#else
      for (j=0; j<N; j++)
	b[j] = scalar*c[j];
#endif
      times[1][k] = mysecond() - times[1][k];

      times[2][k] = mysecond();
#if WITH_PTHREADS
      ierr = PetscThreadCommRunKernel(PETSC_COMM_WORLD,(PetscThreadKernel)tuned_STREAM_Add_Kernel,0);CHKERRQ(ierr);
      ierr = PetscThreadCommBarrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
#else
      for (j=0; j<N; j++)
	c[j] = a[j]+b[j];
#endif
      times[2][k] = mysecond() - times[2][k];

      times[3][k] = mysecond();
#if WITH_PTHREADS
      ierr = PetscThreadCommRunKernel(PETSC_COMM_WORLD,(PetscThreadKernel)tuned_STREAM_Triad_Kernel,1,&scalar);CHKERRQ(ierr);
      ierr = PetscThreadCommBarrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
#else
      for (j=0; j<N; j++)
	a[j] = b[j]+scalar*c[j];
#endif
      times[3][k] = mysecond() - times[3][k];
    }

    /*  --- SUMMARY --- */

    for (k=1; k<NTIMES; k++) { /* note -- skip first iteration */
      for (j=0; j<4; j++) {
	avgtime[j] = avgtime[j] + times[j][k];
	mintime[j] = MIN(mintime[j], times[j][k]);
	maxtime[j] = MAX(maxtime[j], times[j][k]);
      }
    }

    printf("Function      Rate (MB/s) \n");
    for (j=0; j<4; j++) {
      avgtime[j] = avgtime[j]/(double)(NTIMES-1);

      printf("%s%11.4f  \n", label[j], 1.0E-06 * bytes[j]/mintime[j]);
    }
    /* printf(HLINE);*/
#if WITH_PTHREADS
    ierr = PetscThreadCommBarrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
#endif
    /* --- Check Results --- */
    checkSTREAMresults();
    /*    printf(HLINE);*/
    PetscFinalize();
    return 0;
}

# define	M	20

int checktick() {
  int		i, minDelta, Delta;
  double	t1, t2, timesfound[M];

  /*  Collect a sequence of M unique time values from the system. */

  for (i = 0; i < M; i++) {
    t1 = mysecond();
    while( ((t2=mysecond()) - t1) < 1.0E-6 )
      ;
    timesfound[i] = t1 = t2;
  }

  /*
   * Determine the minimum difference between these M values.
   * This result will be our estimate (in microseconds) for the
   * clock granularity.
   */

    minDelta = 1000000;
    for (i = 1; i < M; i++) {
      Delta = (int)( 1.0E6 * (timesfound[i]-timesfound[i-1]));
      minDelta = MIN(minDelta, MAX(Delta,0));
    }

    return(minDelta);
}

/* A gettimeofday routine to give access to the wall
   clock timer on most UNIX-like systems.  */

#include <sys/time.h>

double mysecond() {
  struct timeval tp;
  struct timezone tzp;
  int i=0;

  i = gettimeofday(&tp,&tzp);
  return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}

void checkSTREAMresults () {
  double aj,bj,cj,scalar;
  double asum,bsum,csum;
  double epsilon;
  int	j,k;

  /* reproduce initialization */
  aj = 1.0;
  bj = 2.0;
  cj = 0.0;
  /* a[] is modified during timing check */
  aj = 2.0E0 * aj;
  /* now execute timing loop */
  scalar = 3.0;
  for (k=0; k<NTIMES; k++) {
    cj = aj;
    bj = scalar*cj;
    cj = aj+bj;
    aj = bj+scalar*cj;
  }
  aj = aj * (double) (N);
  bj = bj * (double) (N);
  cj = cj * (double) (N);

  asum = 0.0;
  bsum = 0.0;
  csum = 0.0;
  for (j=0; j<N; j++) {
    asum += a[j];
    bsum += b[j];
    csum += c[j];
  }
#ifdef VERBOSE
  printf ("Results Comparison: \n");
  printf ("        Expected  : %f %f %f \n",aj,bj,cj);
  printf ("        Observed  : %f %f %f \n",asum,bsum,csum);
#endif

#ifndef abs
#define abs(a) ((a) >= 0 ? (a) : -(a))
#endif
  epsilon = 1.e-8;

  if (abs(aj-asum)/asum > epsilon) {
    printf ("Failed Validation on array a[]\n");
    printf ("        Expected  : %f \n",aj);
    printf ("        Observed  : %f \n",asum);
  }
  if (abs(bj-bsum)/bsum > epsilon) {
    printf ("Failed Validation on array b[]\n");
    printf ("        Expected  : %f \n",bj);
    printf ("        Observed  : %f \n",bsum);
  }
  if (abs(cj-csum)/csum > epsilon) {
    printf ("Failed Validation on array c[]\n");
    printf ("        Expected  : %f \n",cj);
    printf ("        Observed  : %f \n",csum);
  }
  else {
    ;/*	printf ("Solution Validates\n"); */
  }
}

