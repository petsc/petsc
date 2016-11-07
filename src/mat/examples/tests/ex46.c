
static char help[] = "Tests MatHYPRE\n";

#include <petscmat.h>
#include <petsc/private/matimpl.h>
/* there's no getter API to extract the ParCSR yet */
#include <../src/mat/impls/hypre/mhypre.h>
#include <_hypre_IJ_mv.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A,B,C,D;
  Vec            x,x2,y,y2;
  PetscReal      err;
  PetscInt       i,j,N = 6, M = 6;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
  ierr = MatSetType(A,MATAIJ);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,3,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,3,NULL,3,NULL);CHKERRQ(ierr);
  for (i=0; i<M; i++) {
    for (j=i-2; j<i+1; j++) {
      if (j >= N) {
        ierr = MatSetValue(A,i,N-1,(1.*j*N+i)/3.,ADD_VALUES);CHKERRQ(ierr);
      } else if (i > j) {
        ierr = MatSetValue(A,i,j,(1.*j*N+i)/2.,ADD_VALUES);CHKERRQ(ierr);
      } else {
        ierr = MatSetValue(A,i,j,-1.+(1.*j*N+i)/4.,ADD_VALUES);CHKERRQ(ierr);
      }
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  /* check conversion routines */
  ierr = MatConvert(A,MATHYPRE,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr);
  ierr = MatConvert(A,MATHYPRE,MAT_REUSE_MATRIX,&B);CHKERRQ(ierr);
  ierr = MatConvert(B,MATIS,MAT_INITIAL_MATRIX,&D);CHKERRQ(ierr);
  ierr = MatConvert(B,MATIS,MAT_REUSE_MATRIX,&D);CHKERRQ(ierr);
  ierr = MatConvert(B,MATAIJ,MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);
  ierr = MatConvert(B,MATAIJ,MAT_REUSE_MATRIX,&C);CHKERRQ(ierr);
  ierr = MatAXPY(C,-1.,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(C,NORM_INFINITY,&err);CHKERRQ(ierr);
  if (PetscAbsReal(err) > PETSC_SMALL) SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error Mat AIJ %g",err);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatISGetMPIXAIJ(D,MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);
  ierr = MatAXPY(C,-1.,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(C,NORM_INFINITY,&err);CHKERRQ(ierr);
  if (PetscAbsReal(err) > PETSC_SMALL) SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error Mat IS %g",err);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&D);CHKERRQ(ierr);

  /* check MatHYPRECreateFromParCSR */
  {
    Mat_HYPRE *hB = (Mat_HYPRE*)(B->data);
    hypre_ParCSRMatrix *parcsr = (hypre_ParCSRMatrix*)hypre_IJMatrixObject(hB->ij);
    ierr = MatHYPRECreateFromParCSR(parcsr,PETSC_USE_POINTER,&C);CHKERRQ(ierr);
  }

  /* check matmult */
  ierr = MatCreateVecs(A,&x,&y);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,&x2,&y2);CHKERRQ(ierr);
  ierr = VecSet(x,1.);CHKERRQ(ierr);
  ierr = MatMult(A,x,y);CHKERRQ(ierr);
  ierr = MatMult(B,x,y2);CHKERRQ(ierr);
  ierr = VecAXPY(y,-1.,y2);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&err);CHKERRQ(ierr);
  if (PetscAbsReal(err) > PETSC_SMALL) SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatMult B %g",err);
  ierr = MatMult(C,x,y);CHKERRQ(ierr);
  ierr = VecAXPY(y,-1.,y2);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&err);CHKERRQ(ierr);
  if (PetscAbsReal(err) > PETSC_SMALL) SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatMult C %g",err);
  ierr = VecSet(y,1.);CHKERRQ(ierr);
  ierr = MatMultTranspose(A,y,x);CHKERRQ(ierr);
  ierr = MatMultTranspose(B,y,x2);CHKERRQ(ierr);
  ierr = VecAXPY(x,-1.,x2);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&err);CHKERRQ(ierr);
  if (PetscAbsReal(err) > PETSC_SMALL) SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatMultTranspose C %g",err);
  ierr = MatMultTranspose(C,y,x);CHKERRQ(ierr);
  ierr = VecAXPY(x,-1.,x2);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&err);CHKERRQ(ierr);
  if (PetscAbsReal(err) > PETSC_SMALL) SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Error MatMultTranspose C %g",err);
  ierr = MatDestroy(&C);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&x2);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&y2);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}
