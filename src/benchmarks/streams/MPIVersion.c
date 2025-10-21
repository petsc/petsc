/*
  An adaption of the Stream benchmark for MPI
  Original code developed by John D. McCalpin
*/
#include <petscsys.h>

#define NTIMESINNER 1
#define N           80000000 // 3*sizeof(double)*N > aggregated last level cache size on a compute node
#define NTIMES      50
#define OFFSET      0

static double a[N + OFFSET], b[N + OFFSET], c[N + OFFSET];
static double mintime = 1e9;
static double bytes   = 3 * sizeof(double) * N;

int main(int argc, char **args)
{
  const double scalar = 3.0;
  double       times[NTIMES], rate;
  PetscMPIInt  rank, size;
  PetscInt     n = PETSC_DECIDE, NN;

  PetscCall(PetscInitialize(&argc, &args, NULL, NULL));
  PetscCallMPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(MPI_COMM_WORLD, &size));

  NN = N;
  PetscCall(PetscSplitOwnership(MPI_COMM_WORLD, &n, &NN));
  for (PetscInt j = 0; j < n; ++j) {
    a[j] = 1.0;
    b[j] = 2.0;
    c[j] = 3.0;
  }

  /*   --- MAIN LOOP --- repeat test cases NTIMES times --- */
  for (PetscInt k = 0; k < NTIMES; ++k) {
    PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
    // Do not include barrier in the timed region
    times[k] = MPI_Wtime();
    for (PetscInt l = 0; l < NTIMESINNER; l++) {
      for (PetscInt j = 0; j < n; j++) a[j] = b[j] + scalar * c[j];
      if (size == 2000) PetscCall(PetscPrintf(PETSC_COMM_SELF, "never printed %g\n", a[11])); // to prevent the compiler from optimizing the loop out
    }
    //   PetscCallMPI(MPI_Barrier(MPI_COMM_WORLD));
    times[k] = MPI_Wtime() - times[k];
  }
  // use maximum time over all MPI processes
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, times, NTIMES, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
  for (PetscInt k = 1; k < NTIMES; ++k) { /* note -- skip first iteration */
    mintime = PetscMin(mintime, times[k]);
  }
  rate = 1.0E-06 * bytes * NTIMESINNER / mintime;

  if (rank == 0) {
    FILE *fd;

    if (size != 1) {
      double prate;

      PetscCall(PetscFOpen(PETSC_COMM_SELF, "flops", "r", &fd));
      PetscCheck(fscanf(fd, "%lg", &prate) == 1, PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Unable to read file");
      PetscCall(PetscFClose(PETSC_COMM_SELF, fd));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "%3d %11.1f   Rate (MB/s) %6.1f\n", size, rate, rate / prate));
    } else {
      PetscCall(PetscFOpen(PETSC_COMM_SELF, "flops", "w", &fd));
      PetscCall(PetscFPrintf(PETSC_COMM_SELF, fd, "%g\n", rate));
      PetscCall(PetscFClose(PETSC_COMM_SELF, fd));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "%3d %11.1f   Rate (MB/s) %6.1f\n", size, rate, 1.0));
    }
  }
  PetscCall(PetscFinalize());
  return 0;
}
