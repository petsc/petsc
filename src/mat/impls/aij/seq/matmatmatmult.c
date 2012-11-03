
/*
  Defines matrix-matrix product routines for pairs of SeqAIJ matrices
          D = A * B * C
*/

#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <../src/mat/utils/freespace.h>
#include <../src/mat/utils/petscheap.h>
#include <petscbt.h>
#include <../src/mat/impls/dense/seq/dense.h> /*I "petscmat.h" I*/

#undef __FUNCT__
#define __FUNCT__ "MatMatMatMult_SeqAIJ_SeqAIJ_SeqAIJ"
PetscErrorCode MatMatMatMult_SeqAIJ_SeqAIJ_SeqAIJ(Mat A,Mat B,Mat C,MatReuse scall,PetscReal fill,Mat *D)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  printf("MatMatMatMult_SeqAIJ_SeqAIJ_SeqAIJ...\n");
  /*
  if (scall == MAT_INITIAL_MATRIX){
    ierr = PetscLogEventBegin(MAT_MatMultSymbolic,A,B,0,0);CHKERRQ(ierr);
    ierr = MatMatMultSymbolic_SeqAIJ_SeqAIJ(A,B,fill,C);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(MAT_MatMultSymbolic,A,B,0,0);CHKERRQ(ierr);
  }
  ierr = PetscLogEventBegin(MAT_MatMultNumeric,A,B,0,0);CHKERRQ(ierr);
  ierr = (*(*C)->ops->matmultnumeric)(A,B,*C);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(MAT_MatMultNumeric,A,B,0,0);CHKERRQ(ierr);
   */
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMatMultSymbolic_SeqAIJ_SeqAIJ_SeqAIJ"
PetscErrorCode MatMatMatMultSymbolic_SeqAIJ_SeqAIJ_SeqAIJ(Mat A,Mat B,Mat C,PetscReal fill,Mat *D)
{
  PetscErrorCode     ierr;
  Mat_SeqAIJ         *a=(Mat_SeqAIJ*)A->data,*b=(Mat_SeqAIJ*)B->data,*c;
  PetscInt           *ai=a->i,*bi=b->i,*ci,*cj;
  PetscInt           am=A->rmap->N,bn=B->cmap->N,bm=B->rmap->N;
  PetscReal          afill;
  PetscInt           i,j,anzi,brow,bnzj,cnzi,*bj,*aj,nlnk_max,*lnk,ndouble=0;
  PetscBT            lnkbt;
  PetscFreeSpaceList free_space=PETSC_NULL,current_space=PETSC_NULL;

  PetscFunctionBegin;
  printf("MatMatMatMultSymbolic_SeqAIJ_SeqAIJ_SeqAIJ...\n");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqAIJ"
PetscErrorCode MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqAIJ(Mat A,Mat B,Mat C,Mat D)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  printf("MatMatMatMultNumeric_SeqAIJ_SeqAIJ_SeqAIJ...\n");
  PetscFunctionReturn(0);
}
