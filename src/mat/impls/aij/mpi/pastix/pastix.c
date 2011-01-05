#define PETSCMAT_DLL

/* 
    Provides an interface to the PaStiX sparse solver
*/
#include "../src/mat/impls/aij/seq/aij.h"
#include "../src/mat/impls/aij/mpi/mpiaij.h"
#include "../src/mat/impls/sbaij/seq/sbaij.h"
#include "../src/mat/impls/sbaij/mpi/mpisbaij.h"

#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if defined(PETSC_HAVE_STRING_H)
#include <string.h> 
#endif

EXTERN_C_BEGIN 
#include "pastix.h"
EXTERN_C_END  

typedef struct Mat_Pastix_ {
  pastix_data_t *pastix_data;              /* Pastix data storage structure                        */
  MatStructure   matstruc;                 				                           
  PetscInt       n;                        /* Number of columns in the matrix                      */
  PetscInt       *colptr;                  /* Index of first element of each column in row and val */
  PetscInt       *row;                     /* Row of each element of the matrix                    */
  PetscScalar    *val;                     /* Value of each element of the matrix                  */
  PetscInt       *perm;                    /* Permutation tabular                                  */
  PetscInt       *invp;                    /* Reverse permutation tabular                          */
  PetscScalar    *rhs;                     /* Rhight-hand-side member                              */
  PetscInt       rhsnbr;                   /* Rhight-hand-side number (must be 1)                  */
  PetscInt       iparm[64];                /* Integer parameters                                   */
  double         dparm[64];                /* Floating point parameters                            */
  MPI_Comm       pastix_comm;              /* PaStiX MPI communicator                              */
  PetscMPIInt    commRank;                 /* MPI rank                                             */
  PetscMPIInt    commSize;                 /* MPI communicator size                                */
  PetscTruth     CleanUpPastix;            /* Boolean indicating if we call PaStiX clean step      */
  VecScatter     scat_rhs;
  VecScatter     scat_sol;
  Vec            b_seq;
  PetscTruth     isAIJ;
  PetscErrorCode (*MatDestroy)(Mat);
} Mat_Pastix;

EXTERN PetscErrorCode MatDuplicate_Pastix(Mat,MatDuplicateOption,Mat*);

#undef __FUNCT__  
#define __FUNCT__ "MatConvertToCSC"
/* 
   convert Petsc seqaij matrix to CSC: colptr[n], row[nz], val[nz] 

  input: 
    A       - matrix in seqaij or mpisbaij (bs=1) format
    valOnly - FALSE: spaces are allocated and values are set for the CSC 
              TRUE:  Only fill values
  output:     
    n       - Size of the matrix
    colptr  - Index of first element of each column in row and val
    row     - Row of each element of the matrix                   
    values  - Value of each element of the matrix                 
 */
PetscErrorCode MatConvertToCSC(Mat A,PetscTruth valOnly,PetscInt *n,PetscInt **colptr,PetscInt **row,PetscScalar **values) 
{
  Mat_SeqAIJ     *aa      = (Mat_SeqAIJ*)A->data;
  PetscInt       *rowptr  = aa->i;
  PetscInt       *col     = aa->j;
  PetscScalar    *rvalues = aa->a;
  PetscInt        m       = A->rmap->N;
  PetscInt        nnz;
  PetscInt        i,j, k;
  PetscInt        base = 1;
  PetscInt        idx;
  PetscErrorCode  ierr;
  PetscInt        colidx;
  PetscInt       *colcount; 
  PetscTruth      isSBAIJ;
  PetscTruth      isSeqSBAIJ;
  PetscTruth      isMpiSBAIJ;
  PetscTruth      isSym;

  PetscFunctionBegin;
  ierr = MatIsSymmetric(A,0.0,&isSym);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)A,MATSBAIJ,&isSBAIJ);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)A,MATSEQSBAIJ,&isSeqSBAIJ);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)A,MATMPISBAIJ,&isMpiSBAIJ);CHKERRQ(ierr);

  *n = A->cmap->N;

  /* PaStiX only needs triangular matrix if matrix is symmetric 
   */
  if (isSym && !(isSBAIJ || isSeqSBAIJ || isMpiSBAIJ)) {
    nnz = (aa->nz - *n)/2 + *n; 
  }
  else {
    nnz     = aa->nz;
  }

  if (!valOnly){
    ierr = PetscMalloc(((*n)+1) *sizeof(PetscInt)   ,colptr );CHKERRQ(ierr);
    ierr = PetscMalloc( nnz     *sizeof(PetscInt)   ,row);CHKERRQ(ierr);
    ierr = PetscMalloc( nnz     *sizeof(PetscScalar),values);CHKERRQ(ierr);

    if (isSBAIJ || isSeqSBAIJ || isMpiSBAIJ) {
	ierr = PetscMemcpy (*colptr, rowptr, ((*n)+1)*sizeof(PetscInt));CHKERRQ(ierr);
	for (i = 0; i < *n+1; i++)
	  (*colptr)[i] += base;
	ierr = PetscMemcpy (*row, col, (nnz)*sizeof(PetscInt));CHKERRQ(ierr);
	for (i = 0; i < nnz; i++)
	  (*row)[i] += base;
	ierr = PetscMemcpy (*values, rvalues, (nnz)*sizeof(PetscScalar));CHKERRQ(ierr);
    } else {
      ierr = PetscMalloc((*n)*sizeof(PetscInt)   ,&colcount);CHKERRQ(ierr); 

      for (i = 0; i < m; i++) colcount[i] = 0;
      /* Fill-in colptr */
      for (i = 0; i < m; i++) {
	for (j = rowptr[i]; j < rowptr[i+1]; j++) { 	
	  if (!isSym || col[j] <= i)  colcount[col[j]]++;
        }
      }
      
      (*colptr)[0] = base;
      for (j = 0; j < *n; j++) {
	(*colptr)[j+1] = (*colptr)[j] + colcount[j];
	/* in next loop we fill starting from (*colptr)[colidx] - base */
	colcount[j] = -base;
      }
      
      /* Fill-in rows and values */
      for (i = 0; i < m; i++) {
	for (j = rowptr[i]; j < rowptr[i+1]; j++) {
	  if (!isSym || col[j] <= i) {
	    colidx = col[j];
	    idx    = (*colptr)[colidx] + colcount[colidx];
	    (*row)[idx]    = i + base;
	    (*values)[idx] = rvalues[j];
	    colcount[colidx]++;
	  }
	}
      }
      ierr = PetscFree(colcount);CHKERRQ(ierr);
    }
  } else {
    /* Fill-in only values */
    for (i = 0; i < m; i++) {
      for (j = rowptr[i]; j < rowptr[i+1]; j++) {
	colidx = col[j];
	if ((isSBAIJ || isSeqSBAIJ || isMpiSBAIJ) ||!isSym || col[j] <= i)
	  {
	    /* look for the value to fill */
	    for (k = (*colptr)[colidx] - base; k < (*colptr)[colidx + 1] - base; k++) {
	      if (((*row)[k]-base) == i) {
		(*values)[k] = rvalues[j];
		break;
	      }
	    }
	    /* data structure of sparse matrix has changed */
	    if (k == (*colptr)[colidx + 1] - base) SETERRQ1(PETSC_ERR_PLIB,"overflow on k %D",k);
	  }
      }
    }
  }  
  {
    PetscScalar *tmpvalues;
    PetscInt    *tmprows,*tmpcolptr;
    tmpvalues = (PetscScalar*)malloc(nnz*sizeof(PetscScalar)); if (!tmpvalues) SETERRQ(PETSC_ERR_MEM,"Unable to allocate memory");
    tmprows   = (PetscInt*)malloc(nnz*sizeof(PetscInt));if (!tmprows) SETERRQ(PETSC_ERR_MEM,"Unable to allocate memory");
    tmpcolptr = (PetscInt*)malloc((*n+1)*sizeof(PetscInt));if (!tmpcolptr) SETERRQ(PETSC_ERR_MEM,"Unable to allocate memory");

    if (sizeof(PetscScalar) != sizeof(pastix_float_t)) {
      SETERRQ2(PETSC_ERR_SUP,"sizeof(PetscScalar) %d != sizeof(pastix_float_t) %d",sizeof(PetscScalar),sizeof(pastix_float_t));
    }
    if (sizeof(PetscInt) != sizeof(pastix_int_t)) {
      SETERRQ2(PETSC_ERR_SUP,"sizeof(PetscInt) %d != sizeof(pastix_int_t) %d",sizeof(PetscInt),sizeof(pastix_int_t));
    }

    ierr = PetscMemcpy(tmpcolptr,*colptr,(*n+1)*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(tmprows,*row,nnz*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(tmpvalues,*values,nnz*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = PetscFree(*row);CHKERRQ(ierr);
    ierr = PetscFree(*values);CHKERRQ(ierr);

    pastix_checkMatrix(MPI_COMM_WORLD,API_VERBOSE_NO,((isSym != 0) ? API_SYM_YES : API_SYM_NO),API_YES,*n,&tmpcolptr,&tmprows,&tmpvalues,NULL,1);
    
    ierr = PetscMemcpy(*colptr,tmpcolptr,(*n+1)*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMalloc(((*colptr)[*n]-1)*sizeof(PetscInt),row);CHKERRQ(ierr);
    ierr = PetscMemcpy(*row,tmprows,((*colptr)[*n]-1)*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMalloc(((*colptr)[*n]-1)*sizeof(PetscScalar),values);CHKERRQ(ierr);
    ierr = PetscMemcpy(*values,tmpvalues,((*colptr)[*n]-1)*sizeof(PetscScalar));CHKERRQ(ierr);
    free(tmpvalues);
    free(tmprows);
    free(tmpcolptr);
  }
  PetscFunctionReturn(0);
}



#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_Pastix"
/*
  Call clean step of PaStiX if lu->CleanUpPastix == true.
  Free the CSC matrix.
 */
PetscErrorCode MatDestroy_Pastix(Mat A)
{
  Mat_Pastix      *lu=(Mat_Pastix*)A->spptr; 
  PetscErrorCode   ierr;
  PetscMPIInt      size=lu->commSize;

  PetscFunctionBegin;
  if (lu->CleanUpPastix) {    
    /* Terminate instance, deallocate memories */
    if (size > 1){
      ierr = VecScatterDestroy(lu->scat_rhs);CHKERRQ(ierr);
      ierr = VecDestroy(lu->b_seq);CHKERRQ(ierr);
      ierr = VecScatterDestroy(lu->scat_sol);CHKERRQ(ierr);
    }
    
    lu->iparm[IPARM_START_TASK]=API_TASK_CLEAN; 
    lu->iparm[IPARM_END_TASK]=API_TASK_CLEAN; 

    pastix((pastix_data_t **)&(lu->pastix_data),
	                      lu->pastix_comm,
	   (pastix_int_t)     lu->n,
	   (pastix_int_t*)    lu->colptr,
	   (pastix_int_t*)    lu->row,
	   (pastix_float_t*)  lu->val,
	   (pastix_int_t*)    lu->perm,
	   (pastix_int_t*)    lu->invp,
	   (pastix_float_t*)  lu->rhs,
	   (pastix_int_t)     lu->rhsnbr,
	   (pastix_int_t*)    lu->iparm,
	                      lu->dparm);

    ierr = PetscFree(lu->colptr);CHKERRQ(ierr);
    ierr = PetscFree(lu->row);  CHKERRQ(ierr);    
    ierr = PetscFree(lu->val);  CHKERRQ(ierr);
    ierr = PetscFree(lu->perm); CHKERRQ(ierr);
    ierr = PetscFree(lu->invp); CHKERRQ(ierr); 
    ierr = MPI_Comm_free(&(lu->pastix_comm));CHKERRQ(ierr);
  }
  ierr = (lu->MatDestroy)(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_PaStiX"
/*
  Gather right-hand-side.
  Call for Solve step.
  Scatter solution.
 */
PetscErrorCode MatSolve_PaStiX(Mat A,Vec b,Vec x) 
{
  Mat_Pastix     *lu=(Mat_Pastix*)A->spptr; 
  PetscScalar    *array;
  Vec             x_seq;
  PetscErrorCode  ierr;

  PetscFunctionBegin; 
  lu->rhsnbr = 1;
  x_seq = lu->b_seq;
  if (lu->commSize > 1){
    /* PaStiX only supports centralized rhs. Scatter b into a seqential rhs vector */
    ierr = VecScatterBegin(lu->scat_rhs,b,x_seq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(lu->scat_rhs,b,x_seq,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecGetArray(x_seq,&array);CHKERRQ(ierr);
  } else {  /* size == 1 */
    ierr = VecCopy(b,x);CHKERRQ(ierr);
    ierr = VecGetArray(x,&array);CHKERRQ(ierr);
  }
  lu->rhs = array;
  if (lu->commSize == 1){
    ierr = VecRestoreArray(x,&array);CHKERRQ(ierr);
  } else {
    ierr = VecRestoreArray(x_seq,&array);CHKERRQ(ierr); 
  }

  /* solve phase */
  /*-------------*/
  lu->iparm[IPARM_START_TASK] = API_TASK_SOLVE; 
  lu->iparm[IPARM_END_TASK]   = API_TASK_REFINE;
  lu->iparm[IPARM_RHS_MAKING] = API_RHS_B;
  
  pastix((pastix_data_t **)&(lu->pastix_data), 
	 (MPI_Comm)         lu->pastix_comm,
	 (pastix_int_t)     lu->n, 
	 (pastix_int_t*)    lu->colptr, 
	 (pastix_int_t*)    lu->row, 
	 (pastix_float_t*)  lu->val, 
	 (pastix_int_t*)    lu->perm, 
	 (pastix_int_t*)    lu->invp, 
	 (pastix_float_t*)  lu->rhs, 
	 (pastix_int_t)     lu->rhsnbr, 
	 (pastix_int_t*)    lu->iparm, 
	 (double*)          lu->dparm);
  
  if (lu->iparm[IPARM_ERROR_NUMBER] < 0) {   
    SETERRQ1(PETSC_ERR_LIB,"Error reported by PaStiX in solve phase: lu->iparm[IPARM_ERROR_NUMBER] = %d\n",lu->iparm[IPARM_ERROR_NUMBER] );
  }

  if (lu->commSize == 1){
    ierr = VecRestoreArray(x,&(lu->rhs));CHKERRQ(ierr);
  } else {
    ierr = VecRestoreArray(x_seq,&(lu->rhs));CHKERRQ(ierr); 
  }

  if (lu->commSize > 1) { /* convert PaStiX centralized solution to petsc mpi x */
    ierr = VecScatterBegin(lu->scat_sol,x_seq,x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(lu->scat_sol,x_seq,x,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
  Numeric factorisation using PaStiX solver.

 */
#undef __FUNCT__   
#define __FUNCT__ "MatFactorNumeric_PASTIX"
PetscErrorCode MatFactorNumeric_PaStiX(Mat F,Mat A,const MatFactorInfo *info) 
{
  Mat_Pastix    *lu =(Mat_Pastix*)(F)->spptr; 
  Mat           *tseq;
  PetscErrorCode ierr = 0;
  PetscInt       icntl;
  PetscInt       M=A->rmap->N;
  PetscTruth     valOnly,flg, isSym;
  Mat            F_diag; 
  IS             is_iden;
  Vec            b;
  IS             isrow;
  PetscTruth     isSeqAIJ,isSeqSBAIJ,isMPIAIJ;

  PetscFunctionBegin; 	
  
  ierr = PetscTypeCompare((PetscObject)A,MATSEQAIJ,&isSeqAIJ);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)A,MATMPIAIJ,&isMPIAIJ);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)A,MATSEQSBAIJ,&isSeqSBAIJ);CHKERRQ(ierr);
  if (lu->matstruc == DIFFERENT_NONZERO_PATTERN){ 
    (F)->ops->solve   = MatSolve_PaStiX;

    /* Initialize a PASTIX instance */
    ierr = MPI_Comm_dup(((PetscObject)A)->comm,&(lu->pastix_comm));CHKERRQ(ierr);
    ierr = MPI_Comm_rank(lu->pastix_comm, &lu->commRank);         CHKERRQ(ierr);
    ierr = MPI_Comm_size(lu->pastix_comm, &lu->commSize);         CHKERRQ(ierr);

    /* Set pastix options */
    lu->iparm[IPARM_MODIFY_PARAMETER] = API_NO;
    lu->iparm[IPARM_START_TASK]       = API_TASK_INIT;
    lu->iparm[IPARM_END_TASK]         = API_TASK_INIT;
    lu->rhsnbr = 1;

    /* Call to set default pastix options */
    pastix((pastix_data_t **)&(lu->pastix_data), 
	   (MPI_Comm)         lu->pastix_comm,
	   (pastix_int_t)     lu->n, 
	   (pastix_int_t*)    lu->colptr, 
	   (pastix_int_t*)    lu->row, 
	   (pastix_float_t*)  lu->val, 
	   (pastix_int_t*)    lu->perm, 
	   (pastix_int_t*)    lu->invp, 
	   (pastix_float_t*)  lu->rhs, 
	   (pastix_int_t)     lu->rhsnbr, 
	   (pastix_int_t*)    lu->iparm, 
	   (double*)          lu->dparm);

    ierr = PetscOptionsBegin(((PetscObject)A)->comm,((PetscObject)A)->prefix,"PaStiX Options","Mat");CHKERRQ(ierr);  

    icntl=-1;
    lu->iparm[IPARM_VERBOSE] = API_VERBOSE_NOT;
    ierr = PetscOptionsInt("-mat_pastix_verbose","iparm[IPARM_VERBOSE] : level of printing (0 to 2)","None",lu->iparm[IPARM_VERBOSE],&icntl,&flg);CHKERRQ(ierr);
    if ((flg && icntl > 0) || PetscLogPrintInfo) {
      lu->iparm[IPARM_VERBOSE] =  icntl; 
    } 
    icntl=-1;
    ierr = PetscOptionsInt("-mat_pastix_threadnbr","iparm[IPARM_THREAD_NBR] : Number of thread by MPI node","None",lu->iparm[IPARM_THREAD_NBR],&icntl,PETSC_NULL);CHKERRQ(ierr);
    if ((flg && icntl > 0)) {
      lu->iparm[IPARM_THREAD_NBR] = icntl; 
    }
    PetscOptionsEnd();
    valOnly = PETSC_FALSE; 
  }  else {
    if (isSeqAIJ || isMPIAIJ)  {
      ierr = PetscFree(lu->colptr);CHKERRQ(ierr);
      ierr = PetscFree(lu->row);CHKERRQ(ierr);
      ierr = PetscFree(lu->val);CHKERRQ(ierr);
      valOnly = PETSC_FALSE;
    } else valOnly = PETSC_TRUE; 
  }

  lu->iparm[IPARM_MATRIX_VERIFICATION] = API_YES;

  /* convert mpi A to seq mat A */
  ierr = ISCreateStride(PETSC_COMM_SELF,M,0,1,&isrow);CHKERRQ(ierr);  
  ierr = MatGetSubMatrices(A,1,&isrow,&isrow,MAT_INITIAL_MATRIX,&tseq);CHKERRQ(ierr);
  ierr = ISDestroy(isrow);CHKERRQ(ierr);

  ierr = MatConvertToCSC(*tseq,valOnly, &lu->n, &lu->colptr, &lu->row, &lu->val);CHKERRQ(ierr);
  ierr = MatIsSymmetric(*tseq,0.0,&isSym);CHKERRQ(ierr);
  ierr = MatDestroyMatrices(1,&tseq);CHKERRQ(ierr);

  if (!lu->perm) {
    ierr = PetscMalloc((lu->n)*sizeof(PetscInt)   ,&(lu->perm));CHKERRQ(ierr);
    ierr = PetscMalloc((lu->n)*sizeof(PetscInt)   ,&(lu->invp));CHKERRQ(ierr);
  }

  if (isSym) {
    /* On symmetric matrix, LLT */
    lu->iparm[IPARM_SYM] = API_SYM_YES;
    lu->iparm[IPARM_FACTORIZATION] = API_FACT_LDLT;
  } else {
    /* On unsymmetric matrix, LU */
    lu->iparm[IPARM_SYM] = API_SYM_NO;
    lu->iparm[IPARM_FACTORIZATION] = API_FACT_LU; 
  }
    
  /*----------------*/  
  if (lu->matstruc == DIFFERENT_NONZERO_PATTERN){ 
    if (!(isSeqAIJ || isSeqSBAIJ)) {
      /* PaStiX only supports centralized rhs. Create scatter scat_rhs for repeated use in MatSolve() */
	ierr = VecCreateSeq(PETSC_COMM_SELF,A->cmap->N,&lu->b_seq);CHKERRQ(ierr);
	ierr = ISCreateStride(PETSC_COMM_SELF,A->cmap->N,0,1,&is_iden);CHKERRQ(ierr);
	ierr = VecCreate(((PetscObject)A)->comm,&b);CHKERRQ(ierr);
	ierr = VecSetSizes(b,A->rmap->n,PETSC_DECIDE);CHKERRQ(ierr);
	ierr = VecSetFromOptions(b);CHKERRQ(ierr);
	
	ierr = VecScatterCreate(b,is_iden,lu->b_seq,is_iden,&lu->scat_rhs);CHKERRQ(ierr);
	ierr = VecScatterCreate(lu->b_seq,is_iden,b,is_iden,&lu->scat_sol);CHKERRQ(ierr);
	ierr = ISDestroy(is_iden);CHKERRQ(ierr);
	ierr = VecDestroy(b);CHKERRQ(ierr);    
    }    
    lu->iparm[IPARM_START_TASK] = API_TASK_ORDERING;
    lu->iparm[IPARM_END_TASK]   = API_TASK_NUMFACT;

    pastix((pastix_data_t **)&(lu->pastix_data), 
	   (MPI_Comm)         lu->pastix_comm,
	   (pastix_int_t)     lu->n, 
	   (pastix_int_t*)    lu->colptr, 
	   (pastix_int_t*)    lu->row, 
	   (pastix_float_t*)  lu->val, 
	   (pastix_int_t*)    lu->perm, 
	   (pastix_int_t*)    lu->invp, 
	   (pastix_float_t*)  lu->rhs, 
	   (pastix_int_t)     lu->rhsnbr, 
	   (pastix_int_t*)    lu->iparm, 
	   (double*)          lu->dparm);
    if (lu->iparm[IPARM_ERROR_NUMBER] < 0) { 
      SETERRQ1(PETSC_ERR_LIB,"Error reported by PaStiX in analysis phase: iparm(IPARM_ERROR_NUMBER)=%d\n",lu->iparm[IPARM_ERROR_NUMBER]); 
    }
  } else {
    lu->iparm[IPARM_START_TASK] = API_TASK_NUMFACT;
    lu->iparm[IPARM_END_TASK]   = API_TASK_NUMFACT;
    pastix((pastix_data_t **)&(lu->pastix_data), 
	   (MPI_Comm)         lu->pastix_comm,
	   (pastix_int_t)     lu->n, 
	   (pastix_int_t*)    lu->colptr, 
	   (pastix_int_t*)    lu->row, 
	   (pastix_float_t*)  lu->val, 
	   (pastix_int_t*)    lu->perm, 
	   (pastix_int_t*)    lu->invp, 
	   (pastix_float_t*)  lu->rhs, 
	   (pastix_int_t)     lu->rhsnbr, 
	   (pastix_int_t*)    lu->iparm, 
	   (double*)          lu->dparm);

    if (lu->iparm[IPARM_ERROR_NUMBER] < 0) { 
      SETERRQ1(PETSC_ERR_LIB,"Error reported by PaStiX in analysis phase: iparm(IPARM_ERROR_NUMBER)=%d\n",lu->iparm[IPARM_ERROR_NUMBER]);
    }
  }

  if (lu->commSize > 1){
    if ((F)->factor == MAT_FACTOR_LU){
      F_diag = ((Mat_MPIAIJ *)(F)->data)->A;
    } else {
      F_diag = ((Mat_MPISBAIJ *)(F)->data)->A;
    }
    F_diag->assembled = PETSC_TRUE;
  }
  (F)->assembled     = PETSC_TRUE;
  lu->matstruc       = SAME_NONZERO_PATTERN;
  lu->CleanUpPastix  = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* Note the Petsc r and c permutations are ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_AIJPASTIX"
PetscErrorCode MatLUFactorSymbolic_AIJPASTIX(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  Mat_Pastix      *lu = (Mat_Pastix*)F->spptr;   

  PetscFunctionBegin;
  lu->iparm[IPARM_FACTORIZATION] = API_FACT_LU;
  lu->iparm[IPARM_SYM]           = API_SYM_YES;
  lu->matstruc                   = DIFFERENT_NONZERO_PATTERN;
  F->ops->lufactornumeric        = MatFactorNumeric_PaStiX;
  PetscFunctionReturn(0); 
}


/* Note the Petsc r permutation is ignored */
#undef __FUNCT__  
#define __FUNCT__ "MatCholeskyFactorSymbolic_SBAIJPASTIX"
PetscErrorCode MatCholeskyFactorSymbolic_SBAIJPASTIX(Mat F,Mat A,IS r,const MatFactorInfo *info) 
{
  Mat_Pastix      *lu = (Mat_Pastix*)(F)->spptr;   

  PetscFunctionBegin;
  lu->iparm[IPARM_FACTORIZATION]  = API_FACT_LLT;
  lu->iparm[IPARM_SYM]            = API_SYM_NO;
  lu->matstruc                    = DIFFERENT_NONZERO_PATTERN;
  (F)->ops->choleskyfactornumeric = MatFactorNumeric_PaStiX;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_PaStiX"
PetscErrorCode MatView_PaStiX(Mat A,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscTruth        iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO){
      Mat_Pastix      *lu=(Mat_Pastix*)A->spptr;

      ierr = PetscViewerASCIIPrintf(viewer,"PaStiX run parameters:\n");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Matrix type :                      %s \n",((lu->iparm[IPARM_SYM] == API_SYM_YES)?"Symmetric":"Unsymmetric"));CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Level of printing (0,1,2):         %d \n",lu->iparm[IPARM_VERBOSE]);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  Number of refinements iterations : %d \n",lu->iparm[IPARM_NBITER]);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF,"  Error :                        %g \n",lu->dparm[DPARM_RELATIVE_ERROR]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}


/*MC
     MAT_SOLVER_PASTIX  - A solver package providing direct solvers (LU) for distributed
  and sequential matrices via the external package PaStiX.

  Use config/configure.py --download-pastix to have PETSc installed with PaStiX

  Options Database Keys:
+ -mat_pastix_verbose   <0,1,2>   - print level
- -mat_pastix_threadnbr <integer> - Set the thread number by MPI task.

  Level: beginner

.seealso: PCFactorSetMatSolverPackage(), MatSolverPackage

M*/


#undef __FUNCT__  
#define __FUNCT__ "MatGetInfo_PaStiX"
PetscErrorCode MatGetInfo_PaStiX(Mat A,MatInfoType flag,MatInfo *info)
{
    Mat_Pastix  *lu =(Mat_Pastix*)A->spptr;

    PetscFunctionBegin;
    info->block_size        = 1.0;
    info->nz_allocated      = lu->iparm[IPARM_NNZEROS];
    info->nz_used           = lu->iparm[IPARM_NNZEROS];
    info->nz_unneeded       = 0.0;
    info->assemblies        = 0.0;
    info->mallocs           = 0.0;
    info->memory            = 0.0;
    info->fill_ratio_given  = 0;
    info->fill_ratio_needed = 0;
    info->factor_mallocs    = 0;
    PetscFunctionReturn(0);
}

EXTERN_C_BEGIN 
#undef __FUNCT__  
#define __FUNCT__ "MatFactorGetSolverPackage_pastix"
PetscErrorCode MatFactorGetSolverPackage_pastix(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MAT_SOLVER_PASTIX;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN 
/*
    The seq and mpi versions of this function are the same 
*/
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_seqaij_pastix"
PetscErrorCode MatGetFactor_seqaij_pastix(Mat A,MatFactorType ftype,Mat *F) 
{
  Mat            B;
  PetscErrorCode ierr;
  Mat_Pastix    *pastix;

  PetscFunctionBegin;
  if (ftype != MAT_FACTOR_LU) {
    SETERRQ(PETSC_ERR_SUP,"Cannot use PETSc AIJ matrices with PaStiX Cholesky, use SBAIJ matrix");
  }
  /* Create the factorization matrix */
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,PETSC_NULL);CHKERRQ(ierr);

  B->ops->lufactorsymbolic = MatLUFactorSymbolic_AIJPASTIX;
  B->ops->view             = MatView_PaStiX;
  B->ops->getinfo          = MatGetInfo_PaStiX;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatFactorGetSolverPackage_C","MatFactorGetSolverPackage_pastix", MatFactorGetSolverPackage_pastix);CHKERRQ(ierr);
  B->factor                = MAT_FACTOR_LU;  

  ierr = PetscNewLog(B,Mat_Pastix,&pastix);CHKERRQ(ierr);
  pastix->CleanUpPastix             = PETSC_FALSE;
  pastix->isAIJ                     = PETSC_TRUE;
  pastix->scat_rhs                  = PETSC_NULL;
  pastix->scat_sol                  = PETSC_NULL;
  pastix->MatDestroy                = B->ops->destroy;
  B->ops->destroy                   = MatDestroy_Pastix;
  B->spptr                          = (void*)pastix;

  *F = B;
  PetscFunctionReturn(0); 
}
EXTERN_C_END


EXTERN_C_BEGIN 
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_mpiaij_pastix"
PetscErrorCode MatGetFactor_mpiaij_pastix(Mat A,MatFactorType ftype,Mat *F) 
{
  Mat            B;
  PetscErrorCode ierr;
  Mat_Pastix    *pastix;

  PetscFunctionBegin;
  if (ftype != MAT_FACTOR_LU) SETERRQ(PETSC_ERR_SUP,"Cannot use PETSc AIJ matrices with PaStiX Cholesky, use SBAIJ matrix");
  /* Create the factorization matrix */
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(B,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);

  B->ops->lufactorsymbolic = MatLUFactorSymbolic_AIJPASTIX;
  B->ops->view             = MatView_PaStiX;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatFactorGetSolverPackage_C","MatFactorGetSolverPackage_pastix",MatFactorGetSolverPackage_pastix);CHKERRQ(ierr);
  B->factor                = MAT_FACTOR_LU;  

  ierr = PetscNewLog(B,Mat_Pastix,&pastix);CHKERRQ(ierr);
  pastix->CleanUpPastix             = PETSC_FALSE;
  pastix->isAIJ                     = PETSC_TRUE;
  pastix->scat_rhs                  = PETSC_NULL;
  pastix->scat_sol                  = PETSC_NULL;
  pastix->MatDestroy                = B->ops->destroy;
  B->ops->destroy                  = MatDestroy_Pastix;
  B->spptr                         = (void*)pastix;

  *F = B;
  PetscFunctionReturn(0); 
}
EXTERN_C_END

EXTERN_C_BEGIN 
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_seqsbaij_pastix"
PetscErrorCode MatGetFactor_seqsbaij_pastix(Mat A,MatFactorType ftype,Mat *F) 
{
  Mat            B;
  PetscErrorCode ierr;
  Mat_Pastix    *pastix;

  PetscFunctionBegin;
  if (ftype != MAT_FACTOR_CHOLESKY) {
    SETERRQ(PETSC_ERR_SUP,"Cannot use PETSc SBAIJ matrices with PaStiX LU, use AIJ matrix");
  }
  /* Create the factorization matrix */ 
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(B,1,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(B,1,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);

  B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SBAIJPASTIX;
  B->ops->view                   = MatView_PaStiX;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatFactorGetSolverPackage_C","MatFactorGetSolverPackage_pastix",MatFactorGetSolverPackage_pastix);CHKERRQ(ierr);

  B->factor                      = MAT_FACTOR_CHOLESKY;

  ierr = PetscNewLog(B,Mat_Pastix,&pastix);CHKERRQ(ierr);
  pastix->CleanUpPastix             = PETSC_FALSE;
  pastix->isAIJ                     = PETSC_TRUE;
  pastix->scat_rhs                  = PETSC_NULL;
  pastix->scat_sol                  = PETSC_NULL;
  pastix->MatDestroy                = B->ops->destroy;
  B->ops->destroy                  = MatDestroy_Pastix;
  B->spptr                         = (void*)pastix;

  *F = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN 
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_mpisbaij_pastix"
PetscErrorCode MatGetFactor_mpisbaij_pastix(Mat A,MatFactorType ftype,Mat *F) 
{
  Mat            B;
  PetscErrorCode ierr;
  Mat_Pastix    *pastix;
 
  PetscFunctionBegin;
  if (ftype != MAT_FACTOR_CHOLESKY) SETERRQ(PETSC_ERR_SUP,"Cannot use PETSc SBAIJ matrices with PaStiX LU, use AIJ matrix");

  /* Create the factorization matrix */ 
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSeqSBAIJSetPreallocation(B,1,0,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatMPISBAIJSetPreallocation(B,1,0,PETSC_NULL,0,PETSC_NULL);CHKERRQ(ierr);

  B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SBAIJPASTIX;
  B->ops->view                   = MatView_PaStiX;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatFactorGetSolverPackage_C","MatFactorGetSolverPackage_pastix",MatFactorGetSolverPackage_pastix);CHKERRQ(ierr);
  B->factor                      = MAT_FACTOR_CHOLESKY;

  ierr = PetscNewLog(B,Mat_Pastix,&pastix);CHKERRQ(ierr);
  pastix->CleanUpPastix             = PETSC_FALSE;
  pastix->isAIJ                     = PETSC_TRUE;
  pastix->scat_rhs                  = PETSC_NULL;
  pastix->scat_sol                  = PETSC_NULL;
  pastix->MatDestroy                = B->ops->destroy;
  B->ops->destroy                   = MatDestroy_Pastix;
  B->spptr                          = (void*)pastix;

  *F = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END
