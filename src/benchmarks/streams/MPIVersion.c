
# include <stdio.h>
# include <math.h>
# include <limits.h>
# include <float.h>
#include <petscsys.h>

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

# define N      2000000
# define NTIMES 50
# define OFFSET 0

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

static double bytes[4] = {
  2 * sizeof(double) * N,
  2 * sizeof(double) * N,
  3 * sizeof(double) * N,
  3 * sizeof(double) * N
};

int main(int argc,char **args)
{
  int            quantum, checktick(void);
  register int   j, k;
  double         scalar, t, times[4][NTIMES],irate[4],rate[4];
  int            rank,size,resultlen;
  char           hostname[MPI_MAX_PROCESSOR_NAME];
  MPI_Status     status;
  int            ierr;
  FILE           *fd;

  ierr = PetscInitialize(&argc,&args,NULL,NULL);if (ierr) return ierr;
  ierr = MPI_Comm_rank(MPI_COMM_WORLD,&rank);if (ierr) return ierr;
  ierr = MPI_Comm_size(MPI_COMM_WORLD,&size);if (ierr) return ierr;

  for (j=0; j<MPI_MAX_PROCESSOR_NAME; j++) {
    hostname[j] = 0;
  }
  ierr = MPI_Get_processor_name(hostname,&resultlen);if (ierr) return ierr;
  if (!rank) {
    for (j=1; j<size; j++) {
      ierr = MPI_Recv(hostname,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,j,0,MPI_COMM_WORLD,&status);if (ierr) return ierr;
    }
 } else {
   ierr = MPI_Send(hostname,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,0,0,MPI_COMM_WORLD);if (ierr) return ierr;
 }
 ierr = MPI_Barrier(MPI_COMM_WORLD);

  /* --- SETUP --- determine precision and check timing --- */

  if (!rank) {
    /*printf(HLINE);
    printf("Array size = %d, Offset = %d\n" , N, OFFSET);
    printf("Total memory required = %.1f MB.\n", (3 * N * BytesPerWord) / 1048576.0);
    printf("Each test is run %d times, but only\n", NTIMES);
    printf("the *best* time for each is used.\n");
    printf(HLINE); */
  }

  /* Get initial value for system clock. */

  /*  a = malloc(N*sizeof(double));
  b = malloc(N*sizeof(double));
  c = malloc(N*sizeof(double));*/
  for (j=0; j<N; j++) {
    a[j] = 1.0;
    b[j] = 2.0;
    c[j] = 0.0;
  }

  if (!rank) {
    if  ((quantum = checktick()) >= 1) ; /* printf("Your clock granularity/precision appears to be %d microseconds.\n", quantum); */
    else ; /* printf("Your clock granularity appears to be less than one microsecond.\n");*/
  }

  t = MPI_Wtime();
  for (j = 0; j < N; j++) a[j] = 2.0E0 * a[j];
  t = 1.0E6 * (MPI_Wtime() - t);

  if (!rank) {
    /*  printf("Each test below will take on the order of %d microseconds.\n", (int) t);
    printf("   (= %d clock ticks)\n", (int) (t/quantum));
    printf("Increase the size of the arrays if this shows that\n");
    printf("you are not getting at least 20 clock ticks per test.\n");
    printf(HLINE);*/
  }

  /*   --- MAIN LOOP --- repeat test cases NTIMES times --- */

  scalar = 3.0;
  for (k=0; k<NTIMES; k++)
  {
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    times[0][k] = MPI_Wtime();
    /* should all these barriers be pulled outside of the time call? */
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    for (j=0; j<N; j++) c[j] = a[j];
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    times[0][k] = MPI_Wtime() - times[0][k];

    times[1][k] = MPI_Wtime();
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    for (j=0; j<N; j++) b[j] = scalar*c[j];
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    times[1][k] = MPI_Wtime() - times[1][k];

    times[2][k] = MPI_Wtime();
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    for (j=0; j<N; j++) c[j] = a[j]+b[j];
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    times[2][k] = MPI_Wtime() - times[2][k];

    times[3][k] = MPI_Wtime();
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    for (j=0; j<N; j++) a[j] = b[j]+scalar*c[j];
    ierr = MPI_Barrier(MPI_COMM_WORLD);
    times[3][k] = MPI_Wtime() - times[3][k];
  }

  /*   --- SUMMARY --- */

  for (k=0; k<NTIMES; k++)
    for (j=0; j<4; j++) mintime[j] = MIN(mintime[j], times[j][k]);

  for (j=0; j<4; j++) irate[j] = 1.0E-06 * bytes[j]/mintime[j];
  ierr = MPI_Reduce(irate,rate,4,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  if (ierr) printf("Error calling MPI\n");

  if (!rank) {
    if (size == 1) {
      printf("%d %11.4f   Rate (MB/s)\n",size, rate[3]);
      fd = fopen("flops","w");
      fprintf(fd,"%g\n",rate[3]);
      fclose(fd);
    } else {
      double prate;
      fd = fopen("flops","r");
      fscanf(fd,"%lg",&prate);
      fclose(fd);
      printf("%d %11.4f   Rate (MB/s) %g \n", size, rate[3],rate[3]/prate);
    }
  }
  PetscFinalize();
  return 0;
}

# define        M        20

int checktick(void)
{
  int    i, minDelta, Delta;
  double t1, t2, timesfound[M];

/*  Collect a sequence of M unique time values from the system. */

  for (i = 0; i < M; i++) {
    t1 = MPI_Wtime();
    while (((t2=MPI_Wtime()) - t1) < 1.0E-6) ;
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

