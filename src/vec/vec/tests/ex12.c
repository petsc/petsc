
static char help[] = "Scatters from a sequential vector to a parallel vector.\n\
This does case when we are merely selecting the local part of the\n\
parallel vector.\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscMPIInt    size,rank;
  PetscInt       n = 5,i;
  PetscScalar    value;
  Vec            x,y;
  IS             is1,is2;
  VecScatter     ctx = 0;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* create two vectors */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetSizes(x,PETSC_DECIDE,size*n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,n,&y));

  /* create two index sets */
  PetscCall(ISCreateStride(PETSC_COMM_SELF,n,n*rank,1,&is1));
  PetscCall(ISCreateStride(PETSC_COMM_SELF,n,0,1,&is2));

  /* each processor inserts the entire vector */
  /* this is redundant but tests assembly */
  for (i=0; i<n; i++) {
    value = (PetscScalar) (i + 10*rank);
    PetscCall(VecSetValues(y,1,&i,&value,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(y));
  PetscCall(VecAssemblyEnd(y));

  PetscCall(VecScatterCreate(y,is2,x,is1,&ctx));
  PetscCall(VecScatterBegin(ctx,y,x,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx,y,x,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterDestroy(&ctx));

  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(ISDestroy(&is1));
  PetscCall(ISDestroy(&is2));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2

TEST*/
