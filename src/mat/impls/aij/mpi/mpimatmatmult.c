/*
  Defines matrix-matrix product routines for pairs of MPIAIJ matrices
          C = A * B
*/
#include "src/mat/impls/aij/seq/aij.h" /*I "petscmat.h" I*/
#include "src/mat/utils/freespace.h"
#include "src/mat/impls/aij/mpi/mpiaij.h"
#include "petscbt.h"

#undef __FUNCT__
#define __FUNCT__ "MatMatMult_MPIAIJ_MPIAIJ"
PetscErrorCode MatMatMult_MPIAIJ_MPIAIJ(Mat A,Mat B,MatReuse scall,PetscReal fill, Mat *C) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX){ 
    ierr = MatMatMultSymbolic_MPIAIJ_MPIAIJ(A,B,fill,C);CHKERRQ(ierr);/* numeric product is computed as well */
  } else if (scall == MAT_REUSE_MATRIX){
    ierr = MatMatMultNumeric_MPIAIJ_MPIAIJ(A,B,*C);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_ARG_WRONG,"Invalid MatReuse %d",(int)scall);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscObjectContainerDestroy_Mat_MatMatMultMPI"
PetscErrorCode PetscObjectContainerDestroy_Mat_MatMatMultMPI(void *ptr)
{
  PetscErrorCode       ierr;
  Mat_MatMatMultMPI    *mult=(Mat_MatMatMultMPI*)ptr;

  PetscFunctionBegin;
  if (mult->startsj){ierr = PetscFree(mult->startsj);CHKERRQ(ierr);}
  if (mult->bufa){ierr = PetscFree(mult->bufa);CHKERRQ(ierr);}
  if (mult->isrowa){ierr = ISDestroy(mult->isrowa);CHKERRQ(ierr);}
  if (mult->isrowb){ierr = ISDestroy(mult->isrowb);CHKERRQ(ierr);}
  if (mult->iscolb){ierr = ISDestroy(mult->iscolb);CHKERRQ(ierr);}
  if (mult->C_seq){ierr = MatDestroy(mult->C_seq);CHKERRQ(ierr);} 
  if (mult->A_loc){ierr = MatDestroy(mult->A_loc);CHKERRQ(ierr); }
  if (mult->B_seq){ierr = MatDestroy(mult->B_seq);CHKERRQ(ierr);}
  if (mult->B_loc){ierr = MatDestroy(mult->B_loc);CHKERRQ(ierr);}
  if (mult->B_oth){ierr = MatDestroy(mult->B_oth);CHKERRQ(ierr);}
  if (mult->abi){ierr = PetscFree(mult->abi);CHKERRQ(ierr);}
  if (mult->abj){ierr = PetscFree(mult->abj);CHKERRQ(ierr);}
  ierr = PetscFree(mult);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

EXTERN PetscErrorCode MatDestroy_AIJ(Mat);
#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MPIAIJ_MatMatMult"
PetscErrorCode MatDestroy_MPIAIJ_MatMatMult(Mat A)
{
  PetscErrorCode       ierr;
  PetscObjectContainer container;
  Mat_MatMatMultMPI    *mult=PETSC_NULL;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)A,"Mat_MatMatMultMPI",(PetscObject *)&container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscObjectContainerGetPointer(container,(void **)&mult);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_ARG_NULL,"Container does not exit");
  }
  ierr = PetscObjectCompose((PetscObject)A,"Mat_MatMatMultMPI",0);CHKERRQ(ierr);
  ierr = (*mult->MatDestroy)(A);CHKERRQ(ierr);
  ierr = PetscObjectContainerDestroy(container);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMatMultSymbolic_MPIAIJ_MPIAIJ"
PetscErrorCode MatMatMultSymbolic_MPIAIJ_MPIAIJ(Mat A,Mat B,PetscReal fill,Mat *C)
{
  Mat_MPIAIJ           *a=(Mat_MPIAIJ*)A->data,*b=(Mat_MPIAIJ*)B->data;
  PetscErrorCode       ierr;
  PetscInt             start,end;
  Mat_MatMatMultMPI    *mult;
  PetscObjectContainer container;
 
  PetscFunctionBegin;
  if (a->cstart != b->rstart || a->cend != b->rend){
    SETERRQ4(PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, (%D, %D) != (%D,%D)",a->cstart,a->cend,b->rstart,b->rend);
  }
  ierr = PetscNew(Mat_MatMatMultMPI,&mult);CHKERRQ(ierr);

  /* create a seq matrix B_seq = submatrix of B by taking rows of B that equal to nonzero col of A */
  ierr = MatGetBrowsOfAcols(A,B,MAT_INITIAL_MATRIX,&mult->isrowb,&mult->iscolb,&mult->brstart,&mult->B_seq);CHKERRQ(ierr);

  /*  create a seq matrix A_seq = submatrix of A by taking all local rows of A */
  start = a->rstart; end = a->rend;
  ierr = ISCreateStride(PETSC_COMM_SELF,end-start,start,1,&mult->isrowa);CHKERRQ(ierr); 
  ierr = MatGetLocalMatCondensed(A,MAT_INITIAL_MATRIX,&mult->isrowa,&mult->isrowb,&mult->A_loc);CHKERRQ(ierr); 

  /* compute C_seq = A_seq * B_seq */
  ierr = MatMatMult_SeqAIJ_SeqAIJ(mult->A_loc,mult->B_seq,MAT_INITIAL_MATRIX,fill,&mult->C_seq);CHKERRQ(ierr);

  /* create mpi matrix C by concatinating C_seq */
  ierr = PetscObjectReference((PetscObject)mult->C_seq);CHKERRQ(ierr); /* prevent C_seq being destroyed by MatMerge() */
  ierr = MatMerge(A->comm,mult->C_seq,B->n,MAT_INITIAL_MATRIX,C);CHKERRQ(ierr); 
 
  /* attach the supporting struct to C for reuse of symbolic C */
  ierr = PetscObjectContainerCreate(PETSC_COMM_SELF,&container);CHKERRQ(ierr);
  ierr = PetscObjectContainerSetPointer(container,mult);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)(*C),"Mat_MatMatMultMPI",(PetscObject)container);CHKERRQ(ierr);
  ierr = PetscObjectContainerSetUserDestroy(container,PetscObjectContainerDestroy_Mat_MatMatMultMPI);CHKERRQ(ierr);
  mult->MatDestroy = (*C)->ops->destroy;

  (*C)->ops->destroy  = MatDestroy_MPIAIJ_MatMatMult; 
  PetscFunctionReturn(0);
}

/* This routine is called ONLY in the case of reusing previously computed symbolic C */
#undef __FUNCT__  
#define __FUNCT__ "MatMatMultNumeric_MPIAIJ_MPIAIJ"
PetscErrorCode MatMatMultNumeric_MPIAIJ_MPIAIJ(Mat A,Mat B,Mat C)
{
  PetscErrorCode       ierr;
  Mat                  *seq;
  Mat_MatMatMultMPI    *mult; 
  PetscObjectContainer container;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)C,"Mat_MatMatMultMPI",(PetscObject *)&container);CHKERRQ(ierr);
  if (container) {
    ierr  = PetscObjectContainerGetPointer(container,(void **)&mult);CHKERRQ(ierr); 
  }

  seq = &mult->B_seq;
  ierr = MatGetSubMatrices(B,1,&mult->isrowb,&mult->iscolb,MAT_REUSE_MATRIX,&seq);CHKERRQ(ierr);
  mult->B_seq = *seq;
  
  seq = &mult->A_loc;
  ierr = MatGetSubMatrices(A,1,&mult->isrowa,&mult->isrowb,MAT_REUSE_MATRIX,&seq);CHKERRQ(ierr);
  mult->A_loc = *seq;

  ierr = MatMatMult_SeqAIJ_SeqAIJ(mult->A_loc,mult->B_seq,MAT_REUSE_MATRIX,0.0,&mult->C_seq);CHKERRQ(ierr);

  ierr = PetscObjectReference((PetscObject)mult->C_seq);CHKERRQ(ierr); 
  ierr = MatMerge(A->comm,mult->C_seq,B->n,MAT_REUSE_MATRIX,&C);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}
