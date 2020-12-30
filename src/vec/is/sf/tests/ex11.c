
static char help[]= "Scatters between parallel vectors. \n\
uses block index sets\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       bs=1,n=5,N,i,low;
  PetscInt       ix0[3] = {5,7,9},iy0[3] = {1,2,4},ix1[3] = {2,3,1},iy1[3] = {0,3,9};
  PetscMPIInt    size,rank;
  PetscScalar    *array;
  Vec            x,x1,y;
  IS             isx,isy;
  VecScatter     ctx;
  VecScatterType type;
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  if (size <2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Must run more than one processor");

  ierr = PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL);CHKERRQ(ierr);
  n    = bs*n;

  /* Create vector x over shared memory */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  ierr = VecGetOwnershipRange(x,&low,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(x,&array);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    array[i] = (PetscScalar)(i + low);
  }
  ierr = VecRestoreArray(x,&array);CHKERRQ(ierr);
  /* ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

  /* Test some vector functions */
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  ierr = VecGetSize(x,&N);CHKERRQ(ierr);
  ierr = VecGetLocalSize(x,&n);CHKERRQ(ierr);

  ierr = VecDuplicate(x,&x1);CHKERRQ(ierr);
  ierr = VecCopy(x,x1);CHKERRQ(ierr);
  ierr = VecEqual(x,x1,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)x),PETSC_ERR_ARG_WRONG,"x1 != x");

  ierr = VecScale(x1,2.0);CHKERRQ(ierr);
  ierr = VecSet(x1,10.0);CHKERRQ(ierr);
  /* ierr = VecView(x1,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

  /* Create vector y over shared memory */
  ierr = VecCreate(PETSC_COMM_WORLD,&y);CHKERRQ(ierr);
  ierr = VecSetSizes(y,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y);CHKERRQ(ierr);
  ierr = VecGetArray(y,&array);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    array[i] = -(PetscScalar) (i + 100*rank);
  }
  ierr = VecRestoreArray(y,&array);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(y);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(y);CHKERRQ(ierr);
  /* ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */

  /* Create two index sets */
  if (!rank) {
    ierr = ISCreateBlock(PETSC_COMM_SELF,bs,3,ix0,PETSC_COPY_VALUES,&isx);CHKERRQ(ierr);
    ierr = ISCreateBlock(PETSC_COMM_SELF,bs,3,iy0,PETSC_COPY_VALUES,&isy);CHKERRQ(ierr);
  } else {
    ierr = ISCreateBlock(PETSC_COMM_SELF,bs,3,ix1,PETSC_COPY_VALUES,&isx);CHKERRQ(ierr);
    ierr = ISCreateBlock(PETSC_COMM_SELF,bs,3,iy1,PETSC_COPY_VALUES,&isy);CHKERRQ(ierr);
  }

  if (rank == 10) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n[%d] isx:\n",rank);CHKERRQ(ierr);
    ierr = ISView(isx,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"\n[%d] isy:\n",rank);CHKERRQ(ierr);
    ierr = ISView(isy,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  }

  /* Create Vector scatter */
  ierr = VecScatterCreate(x,isx,y,isy,&ctx);CHKERRQ(ierr);
  ierr = VecScatterSetFromOptions(ctx);CHKERRQ(ierr);
  ierr = VecScatterGetType(ctx,&type);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"scatter type %s\n",type);CHKERRQ(ierr);

  /* Test forward vecscatter */
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,x,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,x,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSCATTER_FORWARD y:\n");CHKERRQ(ierr);
  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Test reverse vecscatter */
  ierr = VecSet(x,0.0);CHKERRQ(ierr);
  ierr = VecSet(y,0.0);CHKERRQ(ierr);
  ierr = VecGetOwnershipRange(y,&low,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(y,&array);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    array[i] = (PetscScalar)(i+ low);
  }
  ierr = VecRestoreArray(y,&array);CHKERRQ(ierr);
  ierr = VecScatterBegin(ctx,y,x,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(ctx,y,x,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSCATTER_REVERSE x:\n");CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Free objects */
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);
  ierr = ISDestroy(&isx);CHKERRQ(ierr);
  ierr = ISDestroy(&isy);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&x1);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2

TEST*/
