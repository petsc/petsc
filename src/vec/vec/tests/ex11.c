
static char help[] = "Scatters from a parallel vector to a sequential vector.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscMPIInt    size,rank;
  PetscInt       i,N;
  PetscScalar    value;
  Vec            x,y;
  IS             is1,is2;
  VecScatter     ctx = 0;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* create two vectors */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,rank+1,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecGetSize(x,&N));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,N-rank,&y));

  /* create two index sets */
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,N-rank,rank,1,&is1));
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,N-rank,0,1,&is2));

  /* fill parallel vector: note this is not efficient way*/
  for (i=0; i<N; i++) {
    value = (PetscScalar) i;
    CHKERRQ(VecSetValues(x,1,&i,&value,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));
  CHKERRQ(VecSet(y,-1.0));

  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(VecScatterCreate(x,is1,y,is2,&ctx));
  CHKERRQ(VecScatterBegin(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterDestroy(&ctx));

  if (rank == 0) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"----\n"));
    CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_SELF));
  }

  CHKERRQ(ISDestroy(&is1));
  CHKERRQ(ISDestroy(&is2));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2

   test:
      suffix: bts
      nsize: 2
      args: -vec_assembly_legacy
      output_file: output/ex11_1.out

TEST*/
