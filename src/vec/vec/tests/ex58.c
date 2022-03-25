
static char help[] = "Test VecCreate{Seq|MPI}CUDAWithArrays.\n\n";

#include "petsc.h"

int main(int argc,char **argv)
{
  Vec         x,y;
  PetscMPIInt size;
  PetscInt    n        = 5;
  PetscScalar xHost[5] = {0.,1.,2.,3.,4.};

  PetscCall(PetscInitialize(&argc, &argv, (char*)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  if (size == 1) PetscCall(VecCreateSeqCUDAWithArrays(PETSC_COMM_WORLD,1,n,xHost,NULL,&x));
  else PetscCall(VecCreateMPICUDAWithArrays(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,xHost,NULL,&x));

  /* print x should be equivalent too xHost */
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecSet(x,42.0));
  /* print x should be all 42 */
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  if (size == 1) PetscCall(VecCreateSeqWithArray(PETSC_COMM_WORLD,1,n,xHost,&y));
  else PetscCall(VecCreateMPIWithArray(PETSC_COMM_WORLD,1,n,PETSC_DECIDE,xHost,&y));

  /* print y should be all 42 */
  PetscCall(VecView(y, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&x));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: cuda

   test:
      nsize: 1
      suffix: 1

   test:
      nsize: 2
      suffix: 2

TEST*/
