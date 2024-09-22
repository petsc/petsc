static char help[] = "Benchmark dense matrix LU factorization (BLAS/LAPACK)\n\n";

#include <petscbm.h>
#include <petscmat.h>

int main(int argc, char **argv)
{
  PetscBench bm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(MatInitializePackage());
  PetscCall(PetscBenchCreate(PETSC_COMM_SELF, &bm));
  PetscCall(PetscBenchSetType(bm, PETSCBMHPL));
  PetscCall(PetscBenchSetFromOptions(bm));
  PetscCall(PetscBenchSetUp(bm));
  PetscCall(PetscBenchRun(bm));
  PetscCall(PetscBenchView(bm, PETSC_VIEWER_STDOUT_SELF));
  PetscCall(PetscBenchSetSize(bm, 5000));
  PetscCall(PetscBenchRun(bm));
  PetscCall(PetscBenchView(bm, PETSC_VIEWER_STDOUT_SELF));

  PetscCall(PetscBenchDestroy(&bm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
     requires: hpl

TEST*/
