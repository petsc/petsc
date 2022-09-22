static char help[] = "This program illustrates the use of PETSc-fftw interface for sequential real DFT\n";
#include <petscmat.h>
#include <fftw3-mpi.h>
int main(int argc, char **args)
{
  PetscMPIInt rank, size;
  PetscInt    N0 = 10, N1 = 10, N2 = 10, N3 = 10, N4 = 10, N = N0 * N1 * N2 * N3 * N4;
  PetscRandom rdm;
  PetscReal   enorm;
  Vec         x, y, z, input, output;
  Mat         A;
  PetscInt    DIM, dim[5], vsize;
  PetscReal   fac;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "This example requires real numbers");
#endif
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uni-processor example only");
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rdm));
  PetscCall(PetscRandomSetFromOptions(rdm));
  PetscCall(VecCreate(PETSC_COMM_SELF, &input));
  PetscCall(VecSetSizes(input, N, N));
  PetscCall(VecSetFromOptions(input));
  PetscCall(VecSetRandom(input, rdm));
  PetscCall(VecDuplicate(input, &output));

  DIM    = 5;
  dim[0] = N0;
  dim[1] = N1;
  dim[2] = N2;
  dim[3] = N3;
  dim[4] = N4;
  PetscCall(MatCreateFFT(PETSC_COMM_SELF, DIM, dim, MATFFTW, &A));
  PetscCall(MatCreateVecs(A, &x, &y));
  PetscCall(MatCreateVecs(A, &z, NULL));

  PetscCall(VecGetSize(x, &vsize));
  printf("The vector size  of input from the main routine is %d\n", vsize);

  PetscCall(VecGetSize(z, &vsize));
  printf("The vector size of output from the main routine is %d\n", vsize);

  PetscCall(InputTransformFFT(A, input, x));

  PetscCall(MatMult(A, x, y));
  PetscCall(MatMultTranspose(A, y, z));

  PetscCall(OutputTransformFFT(A, z, output));
  fac = 1.0 / (PetscReal)N;
  PetscCall(VecScale(output, fac));
  /*
  PetscCall(VecAssemblyBegin(input));
  PetscCall(VecAssemblyEnd(input));
  PetscCall(VecAssemblyBegin(output));
  PetscCall(VecAssemblyEnd(output));

  PetscCall(VecView(input,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecView(output,PETSC_VIEWER_STDOUT_WORLD));
*/
  PetscCall(VecAXPY(output, -1.0, input));
  PetscCall(VecNorm(output, NORM_1, &enorm));
  /*  if (enorm > 1.e-14) { */
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Error norm of |x - z| %e\n", enorm));
  /*      } */

  PetscCall(VecDestroy(&output));
  PetscCall(VecDestroy(&input));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&z));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscRandomDestroy(&rdm));
  PetscCall(PetscFinalize());
  return 0;
}
