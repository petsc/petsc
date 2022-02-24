
static char help[]= "Scatters from a parallel vector to a sequential vector.\n\
uses block index sets\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       bs = 1,n = 5,ix0[3] = {5,7,9},ix1[3] = {2,3,4},i,iy0[3] = {1,2,4},iy1[3] = {0,1,3};
  PetscMPIInt    size,rank;
  PetscScalar    value;
  Vec            x,y;
  IS             isx,isy;
  VecScatter     ctx = 0,newctx;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCheckFalse(size != 2,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Must run with 2 processors");

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-bs",&bs,NULL));
  n    = bs*n;

  /* create two vectors */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,PETSC_DECIDE,size*n));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&y));

  /* create two index sets */
  if (rank == 0) {
    CHKERRQ(ISCreateBlock(PETSC_COMM_SELF,bs,3,ix0,PETSC_COPY_VALUES,&isx));
    CHKERRQ(ISCreateBlock(PETSC_COMM_SELF,bs,3,iy0,PETSC_COPY_VALUES,&isy));
  } else {
    CHKERRQ(ISCreateBlock(PETSC_COMM_SELF,bs,3,ix1,PETSC_COPY_VALUES,&isx));
    CHKERRQ(ISCreateBlock(PETSC_COMM_SELF,bs,3,iy1,PETSC_COPY_VALUES,&isy));
  }

  /* fill local part of parallel vector */
  for (i=n*rank; i<n*(rank+1); i++) {
    value = (PetscScalar) i;
    CHKERRQ(VecSetValues(x,1,&i,&value,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));

  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  /* fill local part of parallel vector */
  for (i=0; i<n; i++) {
    value = -(PetscScalar) (i + 100*rank);
    CHKERRQ(VecSetValues(y,1,&i,&value,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(y));
  CHKERRQ(VecAssemblyEnd(y));

  CHKERRQ(VecScatterCreate(x,isx,y,isy,&ctx));
  CHKERRQ(VecScatterCopy(ctx,&newctx));
  CHKERRQ(VecScatterDestroy(&ctx));

  CHKERRQ(VecScatterBegin(newctx,y,x,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(newctx,y,x,INSERT_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterDestroy(&newctx));

  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(ISDestroy(&isx));
  CHKERRQ(ISDestroy(&isy));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2

TEST*/
