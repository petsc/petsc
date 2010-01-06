#define PETSCMAT_DLL

/*
  Defines matrix-matrix product routines for pairs of MPIAIJ matrices
          C = A * B
*/
#include "../src/mat/impls/aij/seq/aij.h" /*I "petscmat.h" I*/
#include "../src/mat/utils/freespace.h"
#include "../src/mat/impls/aij/mpi/mpiaij.h"
#include "petscbt.h"
#include "../src/mat/impls/dense/mpi/mpidense.h"

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
#define __FUNCT__ "PetscContainerDestroy_Mat_MatMatMultMPI"
PetscErrorCode PetscContainerDestroy_Mat_MatMatMultMPI(void *ptr)
{
  PetscErrorCode       ierr;
  Mat_MatMatMultMPI    *mult=(Mat_MatMatMultMPI*)ptr;

  PetscFunctionBegin;
  ierr = PetscFree2(mult->startsj,mult->startsj_r);CHKERRQ(ierr);
  ierr = PetscFree(mult->bufa);CHKERRQ(ierr);
  if (mult->isrowa){ierr = ISDestroy(mult->isrowa);CHKERRQ(ierr);}
  if (mult->isrowb){ierr = ISDestroy(mult->isrowb);CHKERRQ(ierr);}
  if (mult->iscolb){ierr = ISDestroy(mult->iscolb);CHKERRQ(ierr);}
  if (mult->C_seq){ierr = MatDestroy(mult->C_seq);CHKERRQ(ierr);} 
  if (mult->A_loc){ierr = MatDestroy(mult->A_loc);CHKERRQ(ierr); }
  if (mult->B_seq){ierr = MatDestroy(mult->B_seq);CHKERRQ(ierr);}
  if (mult->B_loc){ierr = MatDestroy(mult->B_loc);CHKERRQ(ierr);}
  if (mult->B_oth){ierr = MatDestroy(mult->B_oth);CHKERRQ(ierr);}
  ierr = PetscFree(mult->abi);CHKERRQ(ierr);
  ierr = PetscFree(mult->abj);CHKERRQ(ierr);
  ierr = PetscFree(mult);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

EXTERN PetscErrorCode MatDestroy_AIJ(Mat);
#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_MPIAIJ_MatMatMult"
PetscErrorCode MatDestroy_MPIAIJ_MatMatMult(Mat A)
{
  PetscErrorCode     ierr;
  PetscContainer     container;
  Mat_MatMatMultMPI  *mult=PETSC_NULL;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)A,"Mat_MatMatMultMPI",(PetscObject *)&container);CHKERRQ(ierr);
  if (container) {
    ierr = PetscContainerGetPointer(container,(void **)&mult);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_PLIB,"Container does not exit");
  }
  A->ops->destroy = mult->MatDestroy;
  ierr = PetscObjectCompose((PetscObject)A,"Mat_MatMatMultMPI",0);CHKERRQ(ierr);
  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  ierr = PetscContainerDestroy(container);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_MPIAIJ_MatMatMult"
PetscErrorCode MatDuplicate_MPIAIJ_MatMatMult(Mat A, MatDuplicateOption op, Mat *M) 
{
  PetscErrorCode     ierr;
  Mat_MatMatMultMPI  *mult; 
  PetscContainer     container;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)A,"Mat_MatMatMultMPI",(PetscObject *)&container);CHKERRQ(ierr);
  if (container) {
    ierr  = PetscContainerGetPointer(container,(void **)&mult);CHKERRQ(ierr); 
  } else {
    SETERRQ(PETSC_ERR_PLIB,"Container does not exit");
  }
  /* Note: the container is not duplicated, because it requires deep copying of
     several large data sets (see PetscContainerDestroy_Mat_MatMatMultMPI()).
     These data sets are only used for repeated calling of MatMatMultNumeric(). 
     *M is unlikely being used in this way. Thus we create *M with pure mpiaij format */
  ierr = (*mult->MatDuplicate)(A,op,M);CHKERRQ(ierr);
  (*M)->ops->destroy   = mult->MatDestroy;   /* = MatDestroy_MPIAIJ, *M doesn't duplicate A's container! */
  (*M)->ops->duplicate = mult->MatDuplicate; /* = MatDuplicate_MPIAIJ */
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMatMultSymbolic_MPIAIJ_MPIAIJ"
PetscErrorCode MatMatMultSymbolic_MPIAIJ_MPIAIJ(Mat A,Mat B,PetscReal fill,Mat *C)
{
  PetscErrorCode     ierr;
  PetscInt           start,end;
  Mat_MatMatMultMPI  *mult;
  PetscContainer     container;
 
  PetscFunctionBegin;
  if (A->cmap->rstart != B->rmap->rstart || A->cmap->rend != B->rmap->rend){
    SETERRQ4(PETSC_ERR_ARG_SIZ,"Matrix local dimensions are incompatible, (%D, %D) != (%D,%D)",A->cmap->rstart,A->cmap->rend,B->rmap->rstart,B->rmap->rend);
  }
  ierr = PetscNew(Mat_MatMatMultMPI,&mult);CHKERRQ(ierr);

  /* create a seq matrix B_seq = submatrix of B by taking rows of B that equal to nonzero col of A */
  ierr = MatGetBrowsOfAcols(A,B,MAT_INITIAL_MATRIX,&mult->isrowb,&mult->iscolb,&mult->brstart,&mult->B_seq);CHKERRQ(ierr);

  /*  create a seq matrix A_seq = submatrix of A by taking all local rows of A */
  start = A->rmap->rstart; end = A->rmap->rend;
  ierr = ISCreateStride(PETSC_COMM_SELF,end-start,start,1,&mult->isrowa);CHKERRQ(ierr); 
  ierr = MatGetLocalMatCondensed(A,MAT_INITIAL_MATRIX,&mult->isrowa,&mult->isrowb,&mult->A_loc);CHKERRQ(ierr); 

  /* compute C_seq = A_seq * B_seq */
  ierr = MatMatMult_SeqAIJ_SeqAIJ(mult->A_loc,mult->B_seq,MAT_INITIAL_MATRIX,fill,&mult->C_seq);CHKERRQ(ierr);

  /* create mpi matrix C by concatinating C_seq */
  ierr = PetscObjectReference((PetscObject)mult->C_seq);CHKERRQ(ierr); /* prevent C_seq being destroyed by MatMerge() */
  ierr = MatMerge(((PetscObject)A)->comm,mult->C_seq,B->cmap->n,MAT_INITIAL_MATRIX,C);CHKERRQ(ierr); 
 
  /* attach the supporting struct to C for reuse of symbolic C */
  ierr = PetscContainerCreate(PETSC_COMM_SELF,&container);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(container,mult);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)(*C),"Mat_MatMatMultMPI",(PetscObject)container);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(container,PetscContainerDestroy_Mat_MatMatMultMPI);CHKERRQ(ierr);
  mult->MatDestroy   = (*C)->ops->destroy;
  mult->MatDuplicate = (*C)->ops->duplicate;

  (*C)->ops->destroy   = MatDestroy_MPIAIJ_MatMatMult; 
  (*C)->ops->duplicate = MatDuplicate_MPIAIJ_MatMatMult;
  PetscFunctionReturn(0);
}

/* This routine is called ONLY in the case of reusing previously computed symbolic C */
#undef __FUNCT__  
#define __FUNCT__ "MatMatMultNumeric_MPIAIJ_MPIAIJ"
PetscErrorCode MatMatMultNumeric_MPIAIJ_MPIAIJ(Mat A,Mat B,Mat C)
{
  PetscErrorCode     ierr;
  Mat                *seq;
  Mat_MatMatMultMPI  *mult; 
  PetscContainer     container;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)C,"Mat_MatMatMultMPI",(PetscObject *)&container);CHKERRQ(ierr);
  if (container) {
    ierr  = PetscContainerGetPointer(container,(void **)&mult);CHKERRQ(ierr); 
  } else {
    SETERRQ(PETSC_ERR_PLIB,"Container does not exit");
  }

  seq = &mult->B_seq;
  ierr = MatGetSubMatrices(B,1,&mult->isrowb,&mult->iscolb,MAT_REUSE_MATRIX,&seq);CHKERRQ(ierr);
  mult->B_seq = *seq;
  
  seq = &mult->A_loc;
  ierr = MatGetSubMatrices(A,1,&mult->isrowa,&mult->isrowb,MAT_REUSE_MATRIX,&seq);CHKERRQ(ierr);
  mult->A_loc = *seq;

  ierr = MatMatMult_SeqAIJ_SeqAIJ(mult->A_loc,mult->B_seq,MAT_REUSE_MATRIX,0.0,&mult->C_seq);CHKERRQ(ierr);

  ierr = PetscObjectReference((PetscObject)mult->C_seq);CHKERRQ(ierr); 
  ierr = MatMerge(((PetscObject)A)->comm,mult->C_seq,B->cmap->n,MAT_REUSE_MATRIX,&C);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMult_MPIAIJ_MPIDense"
PetscErrorCode MatMatMult_MPIAIJ_MPIDense(Mat A,Mat B,MatReuse scall,PetscReal fill,Mat *C) 
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (scall == MAT_INITIAL_MATRIX){
    ierr = MatMatMultSymbolic_MPIAIJ_MPIDense(A,B,fill,C);CHKERRQ(ierr);
  }  
  ierr = MatMatMultNumeric_MPIAIJ_MPIDense(A,B,*C);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

typedef struct {
  Mat         workB;
  PetscScalar *rvalues,*svalues;
  MPI_Request *rwaits,*swaits;
} MPIAIJ_MPIDense;

PetscErrorCode MPIAIJ_MPIDenseDestroy(void *ctx)
{
  MPIAIJ_MPIDense *contents = (MPIAIJ_MPIDense*) ctx;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (contents->workB) {ierr = MatDestroy(contents->workB);CHKERRQ(ierr);}
  ierr = PetscFree4(contents->rvalues,contents->svalues,contents->rwaits,contents->swaits);CHKERRQ(ierr);
  ierr = PetscFree(contents);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMatMultSymbolic_MPIAIJ_MPIDense"
PetscErrorCode MatMatMultSymbolic_MPIAIJ_MPIDense(Mat A,Mat B,PetscReal fill,Mat *C) 
{
  PetscErrorCode         ierr;
  Mat_MPIAIJ             *aij = (Mat_MPIAIJ*) A->data;
  PetscInt               nz = aij->B->cmap->n;
  PetscContainer         cont;
  MPIAIJ_MPIDense        *contents;
  VecScatter             ctx = aij->Mvctx; 
  VecScatter_MPI_General *from = (VecScatter_MPI_General*) ctx->fromdata;
  VecScatter_MPI_General *to   = ( VecScatter_MPI_General*) ctx->todata;
  PetscInt               m=A->rmap->n,n=B->cmap->n;

  PetscFunctionBegin; 
  ierr = MatCreate(((PetscObject)B)->comm,C);CHKERRQ(ierr);
  ierr = MatSetSizes(*C,m,n,A->rmap->N,B->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(*C,MATMPIDENSE);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscContainerCreate(((PetscObject)A)->comm,&cont);CHKERRQ(ierr);
  ierr = PetscNew(MPIAIJ_MPIDense,&contents);CHKERRQ(ierr);
  ierr = PetscContainerSetPointer(cont,contents);CHKERRQ(ierr);
  ierr = PetscContainerSetUserDestroy(cont,MPIAIJ_MPIDenseDestroy);CHKERRQ(ierr);

  /* Create work matrix used to store off processor rows of B needed for local product */
  ierr = MatCreateSeqDense(PETSC_COMM_SELF,nz,B->cmap->N,PETSC_NULL,&contents->workB);CHKERRQ(ierr);

  /* Create work arrays needed */
  ierr = PetscMalloc4(B->cmap->N*from->starts[from->n],PetscScalar,&contents->rvalues,
                      B->cmap->N*to->starts[to->n],PetscScalar,&contents->svalues,
                      from->n,MPI_Request,&contents->rwaits,
                      to->n,MPI_Request,&contents->swaits);CHKERRQ(ierr);

  ierr = PetscObjectCompose((PetscObject)(*C),"workB",(PetscObject)cont);CHKERRQ(ierr);
  ierr = PetscContainerDestroy(cont);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Performs an efficient scatter on the rows of B needed by this process; this is
    a modification of the VecScatterBegin_() routines.
*/
PetscErrorCode MatMPIDenseScatter(Mat A,Mat B,Mat C,Mat *outworkB)
{
  Mat_MPIAIJ             *aij = (Mat_MPIAIJ*)A->data;
  PetscErrorCode         ierr;
  PetscScalar            *b,*w,*svalues,*rvalues;
  VecScatter             ctx = aij->Mvctx; 
  VecScatter_MPI_General *from = (VecScatter_MPI_General*) ctx->fromdata;
  VecScatter_MPI_General *to   = ( VecScatter_MPI_General*) ctx->todata;
  PetscInt               i,j,k;
  PetscInt               *sindices,*sstarts,*rindices,*rstarts;
  PetscMPIInt            *sprocs,*rprocs,nrecvs;
  MPI_Request            *swaits,*rwaits;
  MPI_Comm               comm = ((PetscObject)A)->comm;
  PetscMPIInt            tag = ((PetscObject)ctx)->tag,ncols = B->cmap->N, nrows = aij->B->cmap->n,imdex,nrowsB = B->rmap->n;
  MPI_Status             status;
  MPIAIJ_MPIDense        *contents;
  PetscContainer         cont;
  Mat                    workB;

  PetscFunctionBegin;
  ierr = PetscObjectQuery((PetscObject)C,"workB",(PetscObject*)&cont);CHKERRQ(ierr);
  ierr = PetscContainerGetPointer(cont,(void**)&contents);CHKERRQ(ierr);

  workB = *outworkB = contents->workB;
  if (nrows != workB->rmap->n) SETERRQ2(PETSC_ERR_PLIB,"Number of rows of workB %D not equal to columns of aij->B %D",nrows,workB->cmap->n);
  sindices  = to->indices;
  sstarts   = to->starts;
  sprocs    = to->procs;
  swaits    = contents->swaits;
  svalues   = contents->svalues;

  rindices  = from->indices;
  rstarts   = from->starts;
  rprocs    = from->procs;
  rwaits    = contents->rwaits;
  rvalues   = contents->rvalues;

  ierr = MatGetArray(B,&b);CHKERRQ(ierr);
  ierr = MatGetArray(workB,&w);CHKERRQ(ierr);

  for (i=0; i<from->n; i++) {
    ierr = MPI_Irecv(rvalues+ncols*rstarts[i],ncols*(rstarts[i+1]-rstarts[i]),MPIU_SCALAR,rprocs[i],tag,comm,rwaits+i);CHKERRQ(ierr);
  } 

  for (i=0; i<to->n; i++) {
    /* pack a message at a time */
    CHKMEMQ;
    for (j=0; j<sstarts[i+1]-sstarts[i]; j++){
      for (k=0; k<ncols; k++) {
        svalues[ncols*(sstarts[i] + j) + k] = b[sindices[sstarts[i]+j] + nrowsB*k];
      }
    }
    CHKMEMQ;
    ierr = MPI_Isend(svalues+ncols*sstarts[i],ncols*(sstarts[i+1]-sstarts[i]),MPIU_SCALAR,sprocs[i],tag,comm,swaits+i);CHKERRQ(ierr);
  }

  nrecvs = from->n;
  while (nrecvs) {
    ierr = MPI_Waitany(from->n,rwaits,&imdex,&status);CHKERRQ(ierr);
    nrecvs--;
    /* unpack a message at a time */
    CHKMEMQ;
    for (j=0; j<rstarts[imdex+1]-rstarts[imdex]; j++){
      for (k=0; k<ncols; k++) {
        w[rindices[rstarts[imdex]+j] + nrows*k] = rvalues[ncols*(rstarts[imdex] + j) + k];
      }
    }
    CHKMEMQ;
  }
  if (to->n) {ierr = MPI_Waitall(to->n,swaits,to->sstatus);CHKERRQ(ierr)}

  ierr = MatRestoreArray(B,&b);CHKERRQ(ierr);
  ierr = MatRestoreArray(workB,&w);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(workB,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(workB,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
extern PetscErrorCode MatMatMultNumericAdd_SeqAIJ_SeqDense(Mat,Mat,Mat);

#undef __FUNCT__  
#define __FUNCT__ "MatMatMultNumeric_MPIAIJ_MPIDense"
PetscErrorCode MatMatMultNumeric_MPIAIJ_MPIDense(Mat A,Mat B,Mat C)
{
  PetscErrorCode       ierr;
  Mat_MPIAIJ           *aij = (Mat_MPIAIJ*)A->data;
  Mat_MPIDense         *bdense = (Mat_MPIDense*)B->data;
  Mat_MPIDense         *cdense = (Mat_MPIDense*)C->data;
  Mat                  workB;

  PetscFunctionBegin;

  /* diagonal block of A times all local rows of B*/
  ierr = MatMatMultNumeric_SeqAIJ_SeqDense(aij->A,bdense->A,cdense->A);CHKERRQ(ierr);

  /* get off processor parts of B needed to complete the product */
  ierr = MatMPIDenseScatter(A,B,C,&workB);CHKERRQ(ierr);

  /* off-diagonal block of A times nonlocal rows of B */
  ierr = MatMatMultNumericAdd_SeqAIJ_SeqDense(aij->B,workB,cdense->A);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

