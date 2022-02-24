
static char help[]= "Scatters from a parallel vector to a sequential vector. \n\
uses block index sets\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       bs=1,n=5,i,low;
  PetscInt       ix0[3] = {5,7,9},iy0[3] = {1,2,4},ix1[3] = {2,3,4},iy1[3] = {0,1,3};
  PetscMPIInt    size,rank;
  PetscScalar    *array;
  Vec            x,y;
  IS             isx,isy;
  VecScatter     ctx;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
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

  /* Create a sequential vector y */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&y));
  CHKERRQ(VecSet(y,0.0));

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
  }

  CHKERRQ(VecScatterCreate(x,isx,y,isy,&ctx));
  CHKERRQ(VecScatterSetFromOptions(ctx));

  /* Test forward vecscatter */
  CHKERRQ(VecScatterBegin(ctx,x,y,ADD_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(ctx,x,y,ADD_VALUES,SCATTER_FORWARD));
  if (rank == 0) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"[%d] y:\n",rank));
    CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_SELF));
  }

  /* Test reverse vecscatter */
  CHKERRQ(VecScale(y,-1.0));
  if (rank) {
    CHKERRQ(VecScale(y,1.0/(size - 1)));
  }

  CHKERRQ(VecScatterBegin(ctx,y,x,ADD_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(ctx,y,x,ADD_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  /* Free spaces */
  CHKERRQ(VecScatterDestroy(&ctx));
  CHKERRQ(ISDestroy(&isx));
  CHKERRQ(ISDestroy(&isy));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 3

TEST*/
