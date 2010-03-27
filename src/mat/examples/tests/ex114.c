
static char help[] = "Tests MatGetRowMax(), MatGetRowMin(), MatGetRowMaxAbs()\n";

#include "petscmat.h"

#define M 5
#define N 6

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A;
  Vec            min,max,maxabs;
  PetscInt       m,n;
  PetscInt       imin[M],imax[M],imaxabs[M],indices[N],row;
  PetscScalar    values[N];
  PetscErrorCode ierr;
  const MatType  type;
  PetscMPIInt    size;
  PetscTruth     doTest=PETSC_TRUE;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  row  = 0; 
  indices[0] = 0; indices[1] = 1; indices[2] = 2; indices[3] = 3; indices[4] = 4; indices[5] = 5;
  values[0] = -1.0; values[1] = 0.0; values[2] = 1.0; values[3] = 3.0; values[4] = 4.0; values[5] = -5.0;
  ierr = MatSetValues(A,1,&row,6,indices,values,INSERT_VALUES);CHKERRQ(ierr);
  row = 1;
  ierr = MatSetValues(A,1,&row,3,indices,values,INSERT_VALUES);CHKERRQ(ierr);
  row = 4;
  ierr = MatSetValues(A,1,&row,1,indices+4,values+4,INSERT_VALUES);CHKERRQ(ierr);
  row = 4;
  ierr = MatSetValues(A,1,&row,2,indices+4,values+4,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatGetLocalSize(A, &m,&n);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&min);CHKERRQ(ierr);
  ierr = VecSetSizes(min,m,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(min);CHKERRQ(ierr);
  ierr = VecDuplicate(min,&max);CHKERRQ(ierr);
  ierr = VecDuplicate(min,&maxabs);CHKERRQ(ierr);

  /* Test MatGetRowMin, MatGetRowMax and MatGetRowMaxAbs */
  if (size == 1) {
    ierr = MatGetRowMin(A,min,imin);CHKERRQ(ierr);
    ierr = MatGetRowMax(A,max,imax);CHKERRQ(ierr);
    ierr = MatGetRowMaxAbs(A,maxabs,imaxabs);CHKERRQ(ierr);

    ierr = PetscPrintf(PETSC_COMM_WORLD,"Row Minimums\n");CHKERRQ(ierr);
    ierr = VecView(min,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscIntView(5,imin,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Row Maximums\n");CHKERRQ(ierr);
    ierr = VecView(max,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscIntView(5,imax,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Row Maximum Absolute Values\n");CHKERRQ(ierr);
    ierr = VecView(maxabs,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscIntView(5,imaxabs,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  } else {
    ierr = MatGetType(A,&type);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\nMatrix type: %s\n",type);
    /* AIJ */
    ierr = PetscTypeCompare((PetscObject)A,MATMPIAIJ,&doTest);CHKERRQ(ierr);
    if (doTest){
      ierr = MatGetRowMaxAbs(A,maxabs,PETSC_NULL);CHKERRQ(ierr);
      ierr = MatGetRowMaxAbs(A,maxabs,imaxabs);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Row Maximum Absolute Values:\n");CHKERRQ(ierr);
      ierr = VecView(maxabs,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
    /* BAIJ */
    ierr = PetscTypeCompare((PetscObject)A,MATMPIBAIJ,&doTest);CHKERRQ(ierr);
    if (doTest){
      ierr = MatGetRowMaxAbs(A,maxabs,PETSC_NULL);CHKERRQ(ierr);
      ierr = MatGetRowMaxAbs(A,maxabs,imaxabs);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Row Maximum Absolute Values:\n");CHKERRQ(ierr);
      ierr = VecView(maxabs,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }
  }

  if (size == 1) {
    ierr = MatConvert(A,MATDENSE,MAT_REUSE_MATRIX,&A);CHKERRQ(ierr);

    ierr = MatGetRowMin(A,min,imin);CHKERRQ(ierr);
    ierr = MatGetRowMax(A,max,imax);CHKERRQ(ierr);
    ierr = MatGetRowMaxAbs(A,maxabs,imaxabs);CHKERRQ(ierr);

    ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Row Minimums\n");CHKERRQ(ierr);
    ierr = VecView(min,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscIntView(5,imin,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Row Maximums\n");CHKERRQ(ierr);
    ierr = VecView(max,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscIntView(5,imax,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Row Maximum Absolute Values\n");CHKERRQ(ierr);
    ierr = VecView(maxabs,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscIntView(5,imaxabs,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  ierr = VecDestroy(min);CHKERRQ(ierr);
  ierr = VecDestroy(max);CHKERRQ(ierr);
  ierr = VecDestroy(maxabs);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr); 
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

