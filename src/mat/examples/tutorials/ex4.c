/*
 *
 *  Created on: Sep 25, 2017
 *      Author: Fande Kong
 */

static char help[] = "Illustrate the use of MatResetPreallocation.\n";

#include "petscmat.h"

int main(int argc,char **argv)
{
  Mat             A;
  MPI_Comm        comm;
  PetscInt        n=5,m=5,*dnnz,*onnz,i,rstart,rend,M,N;
  PetscErrorCode  ierr;

  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  comm = MPI_COMM_WORLD;
  ierr = PetscMalloc2(m,&dnnz,m,&onnz);CHKERRQ(ierr);
  for (i=0; i<m; i++) {
    dnnz[i] = 1;
    onnz[i] = 1;
  }
  ierr = MatCreateAIJ(comm,m,n,PETSC_DETERMINE,PETSC_DETERMINE,PETSC_DECIDE,dnnz,PETSC_DECIDE,onnz,&A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = PetscFree2(dnnz,onnz);CHKERRQ(ierr);

  /* This assembly shrinks memory because we do not insert enough number of values */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* MatResetPreallocation restores the memory required by users */
  ierr = MatResetPreallocation(A);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(A,&rstart,&rend);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    ierr = MatSetValue(A,i,i,2.0,INSERT_VALUES);CHKERRQ(ierr);
    if (rend<N) {
      ierr = MatSetValue(A,i,rend,1.0,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1

   test:
      suffix: 2
      nsize: 2

TEST*/
