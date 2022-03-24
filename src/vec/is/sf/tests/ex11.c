
static char help[]= "Scatters between parallel vectors. \n\
uses block index sets\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscInt       bs=1,n=5,N,i,low;
  PetscInt       ix0[3] = {5,7,9},iy0[3] = {1,2,4},ix1[3] = {2,3,1},iy1[3] = {0,3,9};
  PetscMPIInt    size,rank;
  PetscScalar    *array;
  Vec            x,x1,y;
  IS             isx,isy;
  VecScatter     ctx;
  VecScatterType type;
  PetscBool      flg;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCheckFalse(size <2,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Must run more than one processor");

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL));
  n    = bs*n;

  /* Create vector x over shared memory */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,n,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(x));

  CHKERRQ(VecGetOwnershipRange(x,&low,NULL));
  CHKERRQ(VecGetArray(x,&array));
  for (i=0; i<n; i++) {
    array[i] = (PetscScalar)(i + low);
  }
  CHKERRQ(VecRestoreArray(x,&array));
  /* CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD)); */

  /* Test some vector functions */
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));

  CHKERRQ(VecGetSize(x,&N));
  CHKERRQ(VecGetLocalSize(x,&n));

  CHKERRQ(VecDuplicate(x,&x1));
  CHKERRQ(VecCopy(x,x1));
  CHKERRQ(VecEqual(x,x1,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)x),PETSC_ERR_ARG_WRONG,"x1 != x");

  CHKERRQ(VecScale(x1,2.0));
  CHKERRQ(VecSet(x1,10.0));
  /* CHKERRQ(VecView(x1,PETSC_VIEWER_STDOUT_WORLD)); */

  /* Create vector y over shared memory */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&y));
  CHKERRQ(VecSetSizes(y,n,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(y));
  CHKERRQ(VecGetArray(y,&array));
  for (i=0; i<n; i++) {
    array[i] = -(PetscScalar) (i + 100*rank);
  }
  CHKERRQ(VecRestoreArray(y,&array));
  CHKERRQ(VecAssemblyBegin(y));
  CHKERRQ(VecAssemblyEnd(y));
  /* CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_WORLD)); */

  /* Create two index sets */
  if (rank == 0) {
    CHKERRQ(ISCreateBlock(PETSC_COMM_SELF,bs,3,ix0,PETSC_COPY_VALUES,&isx));
    CHKERRQ(ISCreateBlock(PETSC_COMM_SELF,bs,3,iy0,PETSC_COPY_VALUES,&isy));
  } else {
    CHKERRQ(ISCreateBlock(PETSC_COMM_SELF,bs,3,ix1,PETSC_COPY_VALUES,&isx));
    CHKERRQ(ISCreateBlock(PETSC_COMM_SELF,bs,3,iy1,PETSC_COPY_VALUES,&isy));
  }

  if (rank == 10) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"\n[%d] isx:\n",rank));
    CHKERRQ(ISView(isx,PETSC_VIEWER_STDOUT_SELF));
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"\n[%d] isy:\n",rank));
    CHKERRQ(ISView(isy,PETSC_VIEWER_STDOUT_SELF));
  }

  /* Create Vector scatter */
  CHKERRQ(VecScatterCreate(x,isx,y,isy,&ctx));
  CHKERRQ(VecScatterSetFromOptions(ctx));
  CHKERRQ(VecScatterGetType(ctx,&type));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"scatter type %s\n",type));

  /* Test forward vecscatter */
  CHKERRQ(VecSet(y,0.0));
  CHKERRQ(VecScatterBegin(ctx,x,y,ADD_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(ctx,x,y,ADD_VALUES,SCATTER_FORWARD));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nSCATTER_FORWARD y:\n"));
  CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_WORLD));

  /* Test reverse vecscatter */
  CHKERRQ(VecSet(x,0.0));
  CHKERRQ(VecSet(y,0.0));
  CHKERRQ(VecGetOwnershipRange(y,&low,NULL));
  CHKERRQ(VecGetArray(y,&array));
  for (i=0; i<n; i++) {
    array[i] = (PetscScalar)(i+ low);
  }
  CHKERRQ(VecRestoreArray(y,&array));
  CHKERRQ(VecScatterBegin(ctx,y,x,ADD_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(ctx,y,x,ADD_VALUES,SCATTER_REVERSE));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nSCATTER_REVERSE x:\n"));
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  /* Free objects */
  CHKERRQ(VecScatterDestroy(&ctx));
  CHKERRQ(ISDestroy(&isx));
  CHKERRQ(ISDestroy(&isy));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&x1));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2

TEST*/
