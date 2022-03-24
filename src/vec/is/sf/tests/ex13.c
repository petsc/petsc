
static char help[]= "Scatters from a sequential vector to a parallel vector. \n\
uses block index sets\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscInt       bs=1,n=5,i,low;
  PetscInt       ix0[3] = {5,7,9},iy0[3] = {1,2,4},ix1[3] = {2,3,4},iy1[3] = {0,1,3};
  PetscMPIInt    size,rank;
  PetscScalar    *array;
  Vec            x,y;
  IS             isx,isy;
  VecScatter     ctx;
  PetscViewer    sviewer;

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

  /* Create a sequential vector y */
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&y));
  CHKERRQ(VecGetArray(y,&array));
  for (i=0; i<n; i++) {
    array[i] = (PetscScalar)(i + 100*rank);
  }
  CHKERRQ(VecRestoreArray(y,&array));

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

  CHKERRQ(VecScatterCreate(y,isy,x,isx,&ctx));
  CHKERRQ(VecScatterSetFromOptions(ctx));

  /* Test forward vecscatter */
  CHKERRQ(VecSet(x,0.0));
  CHKERRQ(VecScatterBegin(ctx,y,x,ADD_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(ctx,y,x,ADD_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  /* Test reverse vecscatter */
  CHKERRQ(VecScale(x,-1.0));
  CHKERRQ(VecScatterBegin(ctx,x,y,ADD_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(ctx,x,y,ADD_VALUES,SCATTER_REVERSE));
  CHKERRQ(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  if (rank == 1) {
    CHKERRQ(VecView(y,sviewer));
  }
  CHKERRQ(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));

  /* Free spaces */
  CHKERRQ(VecScatterDestroy(&ctx));
  CHKERRQ(ISDestroy(&isx));
  CHKERRQ(ISDestroy(&isy));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 3

TEST*/
