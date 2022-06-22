static const char help[] = "Test VecScatterCopy() on an SF with duplicated leaves \n\n";

#include <petscvec.h>
#include <petscsf.h>

/*
  Contributed-by: "Hammond, Glenn E" <glenn.hammond@pnnl.gov>
*/
int main(int argc,char* argv[])
{
  PetscMPIInt size;
  PetscInt    n;
  PetscInt    *indices;
  Vec         vec;
  Vec         vec2;
  IS          is;
  IS          is2;
  VecScatter  scatter;
  VecScatter  scatter2;

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This example requires 1 process");

  n = 4;
  PetscCall(PetscMalloc1(n,&indices));
  indices[0] = 0;
  indices[1] = 1;
  indices[2] = 2;
  indices[3] = 3;
  PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,n,indices,PETSC_COPY_VALUES,&is));
  PetscCall(PetscFree(indices));
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD,n,n,&vec));

  n = 4;
  PetscCall(PetscMalloc1(n,&indices));
  indices[0] = 0;
  indices[1] = 0;
  indices[2] = 1;
  indices[3] = 1;
  PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,n,indices,PETSC_COPY_VALUES,&is2));
  PetscCall(PetscFree(indices));
  PetscCall(VecCreateMPI(PETSC_COMM_WORLD,n/2,n/2,&vec2));

  PetscCall(VecScatterCreate(vec,is,vec2,is2,&scatter));
  PetscCall(ISDestroy(&is));
  PetscCall(ISDestroy(&is2));

  PetscCall(VecScatterCopy(scatter,&scatter2));

  PetscCall(VecDestroy(&vec));
  PetscCall(VecDestroy(&vec2));
  PetscCall(VecScatterDestroy(&scatter));
  PetscCall(VecScatterDestroy(&scatter2));
  PetscCall(PetscFinalize());
}

/*TEST
  test:
    nsize: 1
    output_file: output/empty.out
TEST*/
