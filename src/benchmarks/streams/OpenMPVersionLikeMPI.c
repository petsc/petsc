/*
  A simplification of the Stream benchmark for OpenMP
  The array for each thread is a large distance from the array for the other threads
  Original code developed by John D. McCalpin
*/
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <sys/time.h>
#include <stdlib.h>
#include <omp.h>
#include <petscsys.h>

#define NTIMESINNER 1
#define N 2*4*20000000
//#define N 1200000
//#define N 120000
#define NTIMES       50
#define OFFSET       0

# if !defined(MIN)
# define MIN(x,y) ((x)<(y) ? (x) : (y))
# endif
# if !defined(MAX)
# define MAX(x,y) ((x)>(y) ? (x) : (y))
# endif

static double a[64][10000000],b[64][10000000],c[64][10000000];
static double  mintime = FLT_MAX;
static double bytes =  3 * sizeof(double) * N;

int main()
{
  const static double scalar = 3.0;
#pragma omp threadprivate(scalar)
  double       times[NTIMES],rate;
  int          size;
  static int   n;
#pragma omp threadprivate(n)
  char         *env;
  FILE         *fd;

  env = getenv("OMP_NUM_THREADS");
  if (!env) env = (char *) "1";
  sscanf(env,"%d",&size);

#pragma omp parallel for schedule(static)
  for (int j=0; j<size; j++) {
    n = (N / size + ((N % size) > omp_get_thread_num()));
    for (int i=0; i<n; i++){
      a[j][i] = 1.0;
      b[j][i] = 2.0;
      c[j][i] = 3.0;
    }
  }

  /*  --- MAIN LOOP --- repeat test cases NTIMES times --- */
  for (int k=0; k<NTIMES; k++)
  {
    times[k] = MPI_Wtime();
    // https://www.openmp.org/wp-content/uploads/OpenMP-API-Specification-5-2.pdf
// #pragma omp parallel for  (same performance as below)
// #pragma omp parallel for simd schedule(static)  (same performance as below)
#pragma omp parallel for schedule(static)
    for (int j=0; j<size; j++) {
      n = (N / size + ((N % size) > omp_get_thread_num()));
      double *aa = a[j];  // these don't change the timings
      const double *bb = b[j];
      const double *cc = c[j];
      for (int l=0; l<NTIMESINNER; l++) {
        for (register int i=0; i<n; i++) aa[i] = bb[i]+scalar*cc[i];
          if (size == 65) printf("never printed %g\n",a[0][11]);
      }
    }
    times[k] = MPI_Wtime() - times[k];
  }
  for (int k=1; k<NTIMES; k++) {  /* note -- skip first iteration */
      mintime = MIN(mintime, times[k]);
  }
  rate = 1.0E-06 * bytes*NTIMESINNER/mintime;

  if (size == 1) {
    printf("%d %11.4f   Rate (MB/s) 1\n",size, rate);
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
