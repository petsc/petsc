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
#include <stdlib.h>

/* INSTRUCTIONS:
 *
 *      1) Stream requires a good bit of memory to run.  Adjust the
 *          value of 'N' (below) to give a 'timing calibration' of
 *          at least 20 clock-ticks.  This will provide rate estimates
 *          that should be good to about 5% precision.
 */

#if !defined(N)
#   define N    2000000
#endif
#if !defined(NTIMES)
#   define NTIMES       50
#endif
#if !defined(OFFSET)
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

# if !defined(MIN)
# define MIN(x,y) ((x)<(y) ? (x) : (y))
# endif
# if !defined(MAX)
# define MAX(x,y) ((x)>(y) ? (x) : (y))
# endif

static double a[N+OFFSET],
              b[N+OFFSET],
              c[N+OFFSET];

static double avgtime[4] = {0}, maxtime[4] = {0},
              mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};


static double bytes[1] = {
  3 * sizeof(double) * N
};

extern double mysecond();
extern int omp_get_num_threads();
int main()
{
  int          quantum, checktick();
  register int j, k;
  double       scalar, t, times[4][NTIMES],rate;
  int          size;
  char         *env;
  FILE         *fd;

  env = getenv("OMP_NUM_THREADS");
  sscanf(env,"%d",&size);
  /* --- SETUP --- determine precision and check timing --- */

  /*printf(HLINE);
  printf("STREAM version $Revision: 5.9 $\n");
   printf(HLINE); */
  /*    printf("This system uses %d bytes per DOUBLE PRECISION word.\n",
   BytesPerWord);

   printf(HLINE);
#if defined(NO_LONG_LONG)
  printf("Array size = %d, Offset = %d\n" , N, OFFSET);
#else
  printf("Array size = %llu, Offset = %d\n", (unsigned long long) N, OFFSET);
#endif

  printf("Total memory required = %.1f MB.\n",
      (3.0 * BytesPerWord) * ((double) N / 1048576.0));
  printf("Each test is run %d times, but only\n", NTIMES);
  printf("the *best* time for each is used.\n");

   printf(HLINE); */



  /* Get initial value for system clock. */
#pragma omp parallel for
  for (j=0; j<N; j++) {
    a[j] = 1.0;
    b[j] = 2.0;
    c[j] = 0.0;
  }

  /*printf(HLINE);*/

  if  ((quantum = checktick()) >= 1) ; /*  printf("Your clock granularity/precision appears to be "
        "%d microseconds.\n", quantum);*/
  else {
    ;  /*  printf("Your clock granularity appears to be "
        "less than one microsecond.\n");*/
    quantum = 1;
  }

  t = mysecond();
#pragma omp parallel for
  for (j = 0; j < N; j++) a[j] = 2.0E0 * a[j];
  t = 1.0E6 * (mysecond() - t);

  /*printf("Each test below will take on the order"
      " of %d microseconds.\n", (int) t);
  printf("   (= %d clock ticks)\n", (int) (t/quantum));
  printf("Increase the size of the arrays if this shows that\n");
  printf("you are not getting at least 20 clock ticks per test.\n");

   printf(HLINE);*/

  /*  --- MAIN LOOP --- repeat test cases NTIMES times --- */

  scalar = 3.0;
  for (k=0; k<NTIMES; k++)
  {
    times[0][k] = mysecond();
#pragma omp parallel for
    for (j=0; j<N; j++) a[j] = b[j]+scalar*c[j];
    times[0][k] = mysecond() - times[0][k];
  }

  /*  --- SUMMARY --- */

  for (k=1; k<NTIMES; k++) {  /* note -- skip first iteration */
    for (j=0; j<1; j++)
    {
      avgtime[j] = avgtime[j] + times[j][k];
      mintime[j] = MIN(mintime[j], times[j][k]);
      maxtime[j] = MAX(maxtime[j], times[j][k]);
    }
  }

  rate = 1.0E-06 * bytes[0]/mintime[0];

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
    printf("%d %11.4f   Rate (MB/s) %g \n", size, rate,rate/prate);
  }

  return 0;
}

# define        M        20

int checktick()
{
  int    i, minDelta, Delta;
  double t1, t2, timesfound[M];

/*  Collect a sequence of M unique time values from the system. */

  for (i = 0; i < M; i++) {
    t1 = mysecond();
    while (((t2=mysecond()) - t1) < 1.0E-6) ;
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



/* A gettimeofday routine to give access to the wall
   clock timer on most UNIX-like systems.  */

#include <sys/time.h>

double mysecond()
{
  struct timeval  tp;
  struct timezone tzp;

  (void) gettimeofday(&tp,&tzp);
  return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}

