#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <float.h>
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

#define N      2000000
#define M      20
#define NTIMES 50
#define OFFSET 0

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

#define HLINE "-------------------------------------------------------------\n"

static double a[N+OFFSET],b[N+OFFSET],c[N+OFFSET];

static double mintime[4] = {FLT_MAX,FLT_MAX,FLT_MAX,FLT_MAX};

static int checktick(void)
{
  int    minDelta = 1000000;
  double timesfound[M];

  /* Collect a sequence of M unique time values from the system. */

  for (int i = 0; i < M; ++i) {
    const double t1 = MPI_Wtime();

    while (((timesfound[i] = MPI_Wtime()) - t1) < 1.0E-6) ;
  }

  /*
    Determine the minimum difference between these M values.
    This result will be our estimate (in microseconds) for the
    clock granularity.
  */

  for (int i = 1; i < M; ++i) {
    int Delta = (int)(1.0E6*(timesfound[i]-timesfound[i-1]));

    minDelta  = PetscMin(minDelta,PetscMax(Delta,0));
  }
  return minDelta;
}

static double bytes[4] = {
  2 * sizeof(double) * N,
  2 * sizeof(double) * N,
  3 * sizeof(double) * N,
  3 * sizeof(double) * N
};

int main(int argc,char **args)
{
  const double scalar = 3.0;
  double       t,times[4][NTIMES],irate[4],rate[4];
  PetscMPIInt  rank,size,resultlen;
  char         hostname[MPI_MAX_PROCESSOR_NAME] = {0};

  PetscCall(PetscInitialize(&argc,&args,NULL,NULL));
  PetscCallMPI(MPI_Comm_rank(MPI_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(MPI_COMM_WORLD,&size));

  PetscCallMPI(MPI_Get_processor_name(hostname,&resultlen));(void)resultlen;
  if (rank) PetscCallMPI(MPI_Send(hostname,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,0,0,MPI_COMM_WORLD));
  else {
    for (int j = 1; j < size; ++j) {
      PetscCallMPI(MPI_Recv(hostname,MPI_MAX_PROCESSOR_NAME,MPI_CHAR,j,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE));
    }
  }
  PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));

  /* --- SETUP --- determine precision and check timing --- */

  if (rank == 0) {
    /*
      printf(HLINE);
      printf("Array size = %d, Offset = %d\n" , N, OFFSET);
      printf("Total memory required = %.1f MB.\n", (3 * N * BytesPerWord) / 1048576.0);
      printf("Each test is run %d times, but only\n", NTIMES);
      printf("the *best* time for each is used.\n");
      printf(HLINE);
    */
  }

  /* Get initial value for system clock. */
  for (int j = 0; j < N; ++j) {
    a[j] = 1.0;
    b[j] = 2.0;
    c[j] = 0.0;
  }

  if (rank == 0) {
    int quantum;
    if  ((quantum = checktick()) >= 1) { } /* printf("Your clock granularity/precision appears to be %d microseconds.\n", quantum); */
    else { } /* printf("Your clock granularity appears to be less than one microsecond.\n");*/
  }

  t = MPI_Wtime();
  for (int j = 0; j < N; ++j) a[j] *= 2.0;
  t = 1.0E6 * (MPI_Wtime() - t);

  if (rank == 0) {
    /*
      printf("Each test below will take on the order of %d microseconds.\n", (int) t);
      printf("   (= %d clock ticks)\n", (int) (t/quantum));
      printf("Increase the size of the arrays if this shows that\n");
      printf("you are not getting at least 20 clock ticks per test.\n");
      printf(HLINE);
    */
  }

  /*   --- MAIN LOOP --- repeat test cases NTIMES times --- */

  for (int k = 0; k < NTIMES; ++k) {
    PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
    times[0][k] = MPI_Wtime();
    /* should all these barriers be pulled outside of the time call? */
    PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
    PetscCall(PetscArraycpy(c,a,N));
    PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
    times[0][k] = MPI_Wtime() - times[0][k];

    times[1][k] = MPI_Wtime();
    PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
    for (int j = 0; j < N; ++j) b[j] = scalar*c[j];
    PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
    times[1][k] = MPI_Wtime() - times[1][k];

    times[2][k] = MPI_Wtime();
    PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
    for (int j = 0; j < N; ++j) c[j] = a[j]+b[j];
    PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
    times[2][k] = MPI_Wtime() - times[2][k];

    times[3][k] = MPI_Wtime();
    PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
    for (int j = 0; j < N; ++j) a[j] = b[j]+scalar*c[j];
    PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
    times[3][k] = MPI_Wtime() - times[3][k];
  }

  /*   --- SUMMARY --- */

  for (int k = 0; k < NTIMES; ++k) {
    for (int j = 0; j < 4; ++j) mintime[j] = PetscMin(mintime[j],times[j][k]);
  }

  for (int j = 0; j < 4; ++j) irate[j] = 1.0E-06 * bytes[j]/mintime[j];
  PetscCallMPI(MPI_Reduce(irate,rate,4,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD));

  if (rank == 0) {
    FILE *fd;

    if (size != 1) {
      double prate;

      fd = fopen("flops","r");
      fscanf(fd,"%lg",&prate);
      fclose(fd);
      printf("%d %11.4f   Rate (MB/s) %g \n",size,rate[3],rate[3]/prate);
    } else {
      fd = fopen("flops","w");
      fprintf(fd,"%g\n",rate[3]);
      fclose(fd);
      printf("%d %11.4f   Rate (MB/s)\n",size,rate[3]);
    }
  }
  PetscCall(PetscFinalize());
  return 0;
}
