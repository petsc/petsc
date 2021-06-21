
#include <sys/time.h>
/* int gettimeofday(struct timeval *tp, struct timezone *tzp); */

double second()
{
/* struct timeval { long tv_sec;
                    long tv_usec; };

struct timezone { int tz_minuteswest;
                  int tz_dsttime; }; */

  struct timeval  tp;
  struct timezone tzp;
  int             i;

  i = gettimeofday(&tp,&tzp);
  return ((double) tp.tv_sec + (double) tp.tv_usec * 1.e-6);
}
# include <stdio.h>
# include <math.h>
# include <limits.h>
# include <float.h>
# include <sys/time.h>

/*
  Program: Stream
  Programmer: Joe R. Zagar
  Revision: 4.0-BETA, October 24, 1995
  Original code developed by John D. McCalpin

  This program measures memory transfer rates in MB/s for simple
  computational kernels coded in C.  These numbers reveal the quality
  of code generation for simple uncacheable kernels as well as showing
  the cost of floating-point operations relative to memory accesses.

  INSTRUCTIONS:

        1) Stream requires a good bit of memory to run.  Adjust the
           value of 'N' (below) to give a 'timing calibration' of
           at least 20 clock-ticks.  This will provide rate estimates
           that should be good to about 5% precision.
 */

# define N      200000
# define NTIMES     50
# define OFFSET      0

/*
       3) Compile the code with full optimization.  Many compilers
          generate unreasonably bad code before the optimizer tightens
          things up.  If the results are unreasonably good, on the
          other hand, the optimizer might be too smart for me!

          Try compiling with:
                cc -O stream_d.c second.c -o stream_d -lm

          This is known to work on Cray, SGI, IBM, and Sun machines.

       4) Mail the results to mccalpin@cs.virginia.edu
          Be sure to include:
                 a) computer hardware model number and software revision
                 b) the compiler flags
                 c) all of the output from the test case.
  Thanks!

*/

# define HLINE "-------------------------------------------------------------\n"

# ifndef MIN
# define MIN(x,y) ((x)<(y) ? (x) : (y))
# endif
# ifndef MAX
# define MAX(x,y) ((x)>(y) ? (x) : (y))
# endif

static double a[N+OFFSET],
              b[N+OFFSET],
              c[N+OFFSET];
/*double *a,*b,*c;*/

static double mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};

static const char *label[4] = {"Copy:      ", "Scale:     ", "Add:       ", "Triad:     "};

static double bytes[4] = {
  2 * sizeof(double) * N,
  2 * sizeof(double) * N,
  3 * sizeof(double) * N,
  3 * sizeof(double) * N
};

extern double second();

int main(int argc,char **args)
{
  int           checktick(void);
  register int j, k;
  double       scalar, t, times[4][NTIMES],irate[4];

  /* --- SETUP --- determine precision and check timing --- */

  for (j=0; j<N; j++) {
    a[j] = 1.0;
    b[j] = 2.0;
    c[j] = 0.0;
  }

  t = second();
  for (j = 0; j < N; j++) a[j] = 2.0E0 * a[j];
  t = 1.0E6 * (second() - t);

  /*   --- MAIN LOOP --- repeat test cases NTIMES times --- */

  scalar = 3.0;
  for (k=0; k<NTIMES; k++)
  {

    times[0][k] = second();
/* should all these barriers be pulled outside of the time call? */

    for (j=0; j<N; j++) c[j] = a[j];
    times[0][k] = second() - times[0][k];

    times[1][k] = second();

    for (j=0; j<N; j++) b[j] = scalar*c[j];
    times[1][k] = second() - times[1][k];

    times[2][k] = second();
    for (j=0; j<N; j++) c[j] = a[j]+b[j];
    times[2][k] = second() - times[2][k];

    times[3][k] = second();
    for (j=0; j<N; j++) a[j] = b[j]+scalar*c[j];
    times[3][k] = second() - times[3][k];
  }

  /*   --- SUMMARY --- */

  for (k=0; k<NTIMES; k++)
    for (j=0; j<4; j++) mintime[j] = MIN(mintime[j], times[j][k]);

  for (j=0; j<4; j++) irate[j] = 1.0E-06 * bytes[j]/mintime[j];

  printf("Function      Rate (MB/s) \n");
  for (j=0; j<4; j++) printf("%s%11.4f\n", label[j],irate[j]);
  return 0;
}

# define        M        20

int checktick(void)
{
  int    i, minDelta, Delta;
  double t1, t2, timesfound[M];

/*  Collect a sequence of M unique time values from the system. */

  for (i = 0; i < M; i++) {
    t1 = second();
    while (((t2=second()) - t1) < 1.0E-6) ;
    timesfound[i] = t1 = t2;
  }

/*
  Determine the minimum difference between these M values.
  This result will be our estimate (in microseconds) for the
  clock granularity.
 */

  minDelta = 1000000;
  for (i = 1; i < M; i++) {
    Delta    = (int)(1.0E6 * (timesfound[i]-timesfound[i-1]));
    minDelta = MIN(minDelta, MAX(Delta,0));
  }

  return(minDelta);
}

