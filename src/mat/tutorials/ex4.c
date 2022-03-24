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

  CHKERRQ(PetscInitialize(&argc,&argv,0,help));
  comm = MPI_COMM_WORLD;
  CHKERRQ(PetscMalloc2(m,&dnnz,m,&onnz));
  for (i=0; i<m; i++) {
    dnnz[i] = 1;
    onnz[i] = 1;
  }
  CHKERRQ(MatCreateAIJ(comm,m,n,PETSC_DETERMINE,PETSC_DETERMINE,PETSC_DECIDE,dnnz,PETSC_DECIDE,onnz,&A));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(PetscFree2(dnnz,onnz));

  /* This assembly shrinks memory because we do not insert enough number of values */
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  /* MatResetPreallocation restores the memory required by users */
  CHKERRQ(MatResetPreallocation(A));
  CHKERRQ(MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_TRUE));
  CHKERRQ(MatGetOwnershipRange(A,&rstart,&rend));
  CHKERRQ(MatGetSize(A,&M,&N));
  for (i=rstart; i<rend; i++) {
    CHKERRQ(MatSetValue(A,i,i,2.0,INSERT_VALUES));
    if (rend<N) {
      CHKERRQ(MatSetValue(A,i,rend,1.0,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1

   test:
      suffix: 2
      nsize: 2

TEST*/
