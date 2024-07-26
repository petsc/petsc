/*
  A simplification of the Stream benchmark for OpenMP
  Original code developed by John D. McCalpin
*/
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <float.h>
#include <sys/time.h>
#include <stdlib.h>
#include <petscsys.h>

//#define N 2*4*20000000
#define N 80000000
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

static double a[N+OFFSET],b[N+OFFSET],c[N+OFFSET];
static double  mintime = FLT_MAX;
static double bytes =  3 * sizeof(double) * N;

int main(int argc,char **argv)
{
  MPI_Init(&argc,&argv);
  const static double scalar = 3.0;
#pragma omp threadprivate(scalar)
  double       times[NTIMES],rate;
  int          size;
  char         *env;
  FILE         *fd;

  env = getenv("OMP_NUM_THREADS");
  if (!env) env = (char *) "1";
  sscanf(env,"%d",&size);

#pragma omp parallel for schedule(static)
  for (int j=0; j<N; j++) {
    a[j] = 1.0;
    b[j] = 2.0;
    c[j] = 3.0;
  }

  /*  --- MAIN LOOP --- repeat test cases NTIMES times --- */
  for (int k=0; k<NTIMES; k++)
  {
    times[k] = MPI_Wtime();
    // https://www.openmp.org/wp-content/uploads/OpenMP-API-Specification-5-2.pdf
    // #pragma omp parallel for  (same performance as below)
    // #pragma omp parallel for simd schedule(static)  (same performance as below)
#pragma omp parallel for schedule(static)
    for (register int j=0; j<N; j++) a[j] = b[j]+scalar*c[j];
    times[k] = MPI_Wtime() - times[k];
  }
  for (int k=1; k<NTIMES; k++) {  /* note -- skip first iteration */
      mintime = MIN(mintime, times[k]);
  }

  if (size == 65) printf("Never printed %g\n",a[11]);
  rate = 1.0E-06 * bytes/mintime;

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
  MPI_Finalize();
  return 0;
}
