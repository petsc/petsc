#include <omp.h>
#include <petscsys.h>

/*
   See the comments in ex69f.F90
*/
int main(int argc, char **args)
{
  double wtime_start, wtime_end, mpiwtime_start, mpiwtime_end;
  double x[100];
  int    i, maxthreads;

  PetscCall(PetscInitialize(&argc, &args, NULL, NULL));
  wtime_start    = omp_get_wtime();
  mpiwtime_start = MPI_Wtime();
#pragma omp parallel for schedule(static)
  for (i = 0; i < 100; i++) x[i] = exp(3.0 * i);
  wtime_end    = omp_get_wtime();
  mpiwtime_end = MPI_Wtime();
  printf("Wall clock time from MPI_Wtime()     %g\n", wtime_end - wtime_start);
  printf("Wall clock time from omp_get_wtime() %g\n", mpiwtime_end - mpiwtime_start);
  printf("Value of x(22) %g\n", x[22]);
  maxthreads = omp_get_max_threads();
  printf("Number of threads set %d\n", maxthreads);
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: openmp

   test:
     filter: grep -v "Number of threads"

TEST*/
