static char help[] = "This program illustrates the use of PETSc-fftw interface for real DFT\n";
#include <petscmat.h>
#include <fftw3-mpi.h>

extern PetscErrorCode InputTransformFFT(Mat, Vec, Vec);
extern PetscErrorCode OutputTransformFFT(Mat, Vec, Vec);
int                   main(int argc, char **args)
{
  PetscMPIInt rank, size;
  PetscInt    N0 = 3, N1 = 3, N2 = 3, N = N0 * N1 * N2;
  PetscRandom rdm;
  PetscScalar a;
  PetscReal   enorm;
  Vec         x, y, z, input, output;
  PetscBool   view = PETSC_FALSE, use_interface = PETSC_TRUE;
  Mat         A;
  PetscInt    DIM, dim[3], vsize;
  PetscReal   fac;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCheck(!PetscDefined(USE_COMPLEX), PETSC_COMM_WORLD, PETSC_ERR_SUP, "This example requires real numbers");

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rdm));
  PetscCall(PetscRandomSetFromOptions(rdm));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &input));
  PetscCall(VecSetSizes(input, PETSC_DECIDE, N));
  PetscCall(VecSetFromOptions(input));
  PetscCall(VecSetRandom(input, rdm));
  PetscCall(VecDuplicate(input, &output));
  /*  PetscCall(VecGetSize(input,&vsize)); */
  /*  printf("Size of the input Vector is %d\n",vsize); */

  DIM    = 3;
  dim[0] = N0;
  dim[1] = N1;
  dim[2] = N2;

  PetscCall(MatCreateFFT(PETSC_COMM_WORLD, DIM, dim, MATFFTW, &A));
  PetscCall(MatCreateVecs(A, &x, &y));
  PetscCall(MatCreateVecs(A, &z, NULL));
  PetscCall(VecGetSize(y, &vsize));
  printf("The vector size from the main routine is %d\n", vsize);

  PetscCall(InputTransformFFT(A, input, x));
  PetscCall(MatMult(A, x, y));
  PetscCall(MatMultTranspose(A, y, z));
  PetscCall(OutputTransformFFT(A, z, output));

  fac = 1.0 / (PetscReal)N;
  PetscCall(VecScale(output, fac));

  PetscCall(VecAssemblyBegin(input));
  PetscCall(VecAssemblyEnd(input));
  PetscCall(VecAssemblyBegin(output));
  PetscCall(VecAssemblyEnd(output));

  PetscCall(VecView(input, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecView(output, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecAXPY(output, -1.0, input));
  PetscCall(VecNorm(output, NORM_1, &enorm));
  /*  if (enorm > 1.e-14) { */
  if (rank == 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Error norm of |x - z| %e\n", enorm));
  /*      } */

  /* PetscCall(MatCreateVecs(A,&z,NULL)); */
  /*  printf("Vector size from ex148 %d\n",vsize); */
  /*  PetscCall(PetscObjectSetName((PetscObject) x, "Real space vector")); */
  /*      PetscCall(PetscObjectSetName((PetscObject) y, "Frequency space vector")); */
  /*      PetscCall(PetscObjectSetName((PetscObject) z, "Reconstructed vector")); */

  PetscCall(PetscFinalize());
  return 0;
}
