
static char help[]= "Scatters from a parallel vector to a sequential vector.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscInt       n = 5,i,idx2[3] = {0,2,3},idx1[3] = {0,1,2};
  PetscMPIInt    size,rank;
  PetscScalar    value;
  Vec            x,y;
  IS             is1,is2;
  VecScatter     ctx = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* create two vectors */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetSizes(x,PETSC_DECIDE,size*n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,n,&y));

  /* create two index sets */
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,3,idx1,PETSC_COPY_VALUES,&is1));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF,3,idx2,PETSC_COPY_VALUES,&is2));

  /* fill local part of parallel vector */
  for (i=n*rank; i<n*(rank+1); i++) {
    value = (PetscScalar) i;
    PetscCall(VecSetValues(x,1,&i,&value,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));

  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecSet(y,-1.0));

  PetscCall(VecScatterCreate(x,is1,y,is2,&ctx));
  PetscCall(VecScatterBegin(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterDestroy(&ctx));

  if (rank == 0) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"scattered vector\n"));
    PetscCall(VecView(y,PETSC_VIEWER_STDOUT_SELF));
  }
  PetscCall(ISDestroy(&is1));
  PetscCall(ISDestroy(&is2));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2

TEST*/
