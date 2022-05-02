/*
 Provides an interface to the PaStiX sparse solver
 */
#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#include <../src/mat/impls/sbaij/mpi/mpisbaij.h>

#if defined(PETSC_USE_COMPLEX)
#define _H_COMPLEX
#endif

EXTERN_C_BEGIN
#include <pastix.h>
EXTERN_C_END

#if defined(PETSC_USE_COMPLEX)
#if defined(PETSC_USE_REAL_SINGLE)
#define PASTIX_CALL c_pastix
#else
#define PASTIX_CALL z_pastix
#endif

#else /* PETSC_USE_COMPLEX */

#if defined(PETSC_USE_REAL_SINGLE)
#define PASTIX_CALL s_pastix
#else
#define PASTIX_CALL d_pastix
#endif

#endif /* PETSC_USE_COMPLEX */

typedef PetscScalar PastixScalar;

typedef struct Mat_Pastix_ {
  pastix_data_t *pastix_data;    /* Pastix data storage structure                        */
  MatStructure  matstruc;
  PetscInt      n;               /* Number of columns in the matrix                      */
  PetscInt      *colptr;         /* Index of first element of each column in row and val */
  PetscInt      *row;            /* Row of each element of the matrix                    */
  PetscScalar   *val;            /* Value of each element of the matrix                  */
  PetscInt      *perm;           /* Permutation tabular                                  */
  PetscInt      *invp;           /* Reverse permutation tabular                          */
  PetscScalar   *rhs;            /* Rhight-hand-side member                              */
  PetscInt      rhsnbr;          /* Rhight-hand-side number (must be 1)                  */
  PetscInt      iparm[IPARM_SIZE];       /* Integer parameters                                   */
  double        dparm[DPARM_SIZE];       /* Floating point parameters                            */
  MPI_Comm      pastix_comm;     /* PaStiX MPI communicator                              */
  PetscMPIInt   commRank;        /* MPI rank                                             */
  PetscMPIInt   commSize;        /* MPI communicator size                                */
  PetscBool     CleanUpPastix;   /* Boolean indicating if we call PaStiX clean step      */
  VecScatter    scat_rhs;
  VecScatter    scat_sol;
  Vec           b_seq;
} Mat_Pastix;

extern PetscErrorCode MatDuplicate_Pastix(Mat,MatDuplicateOption,Mat*);

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
PetscErrorCode MatConvertToCSC(Mat A,PetscBool valOnly,PetscInt *n,PetscInt **colptr,PetscInt **row,PetscScalar **values)
{
  Mat_SeqAIJ  *aa      = (Mat_SeqAIJ*)A->data;
  PetscInt    *rowptr  = aa->i;
  PetscInt    *col     = aa->j;
  PetscScalar *rvalues = aa->a;
  PetscInt     m       = A->rmap->N;
  PetscInt     nnz;
  PetscInt     i,j, k;
  PetscInt     base    = 1;
  PetscInt     idx;
  PetscInt     colidx;
  PetscInt    *colcount;
  PetscBool    isSBAIJ;
  PetscBool    isSeqSBAIJ;
  PetscBool    isMpiSBAIJ;
  PetscBool    isSym;

  PetscFunctionBegin;
  PetscCall(MatIsSymmetric(A,0.0,&isSym));
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATSBAIJ,&isSBAIJ));
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATSEQSBAIJ,&isSeqSBAIJ));
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATMPISBAIJ,&isMpiSBAIJ));

  *n = A->cmap->N;

  /* PaStiX only needs triangular matrix if matrix is symmetric
   */
  if (isSym && !(isSBAIJ || isSeqSBAIJ || isMpiSBAIJ)) nnz = (aa->nz - *n)/2 + *n;
  else nnz = aa->nz;

  if (!valOnly) {
    PetscCall(PetscMalloc1((*n)+1,colptr));
    PetscCall(PetscMalloc1(nnz,row));
    PetscCall(PetscMalloc1(nnz,values));

    if (isSBAIJ || isSeqSBAIJ || isMpiSBAIJ) {
      PetscCall(PetscArraycpy (*colptr, rowptr, (*n)+1));
      for (i = 0; i < *n+1; i++) (*colptr)[i] += base;
      PetscCall(PetscArraycpy (*row, col, nnz));
      for (i = 0; i < nnz; i++) (*row)[i] += base;
      PetscCall(PetscArraycpy (*values, rvalues, nnz));
    } else {
      PetscCall(PetscMalloc1(*n,&colcount));

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
            colidx         = col[j];
            idx            = (*colptr)[colidx] + colcount[colidx];
            (*row)[idx]    = i + base;
            (*values)[idx] = rvalues[j];
            colcount[colidx]++;
          }
        }
      }
      PetscCall(PetscFree(colcount));
    }
  } else {
    /* Fill-in only values */
    for (i = 0; i < m; i++) {
      for (j = rowptr[i]; j < rowptr[i+1]; j++) {
        colidx = col[j];
        if ((isSBAIJ || isSeqSBAIJ || isMpiSBAIJ) ||!isSym || col[j] <= i) {
          /* look for the value to fill */
          for (k = (*colptr)[colidx] - base; k < (*colptr)[colidx + 1] - base; k++) {
            if (((*row)[k]-base) == i) {
              (*values)[k] = rvalues[j];
              break;
            }
          }
          /* data structure of sparse matrix has changed */
          PetscCheck(k != (*colptr)[colidx + 1] - base,PETSC_COMM_SELF,PETSC_ERR_PLIB,"overflow on k %" PetscInt_FMT,k);
        }
      }
    }
  }
  PetscFunctionReturn(0);
}

/*
  Call clean step of PaStiX if lu->CleanUpPastix == true.
  Free the CSC matrix.
 */
PetscErrorCode MatDestroy_Pastix(Mat A)
{
  Mat_Pastix *lu = (Mat_Pastix*)A->data;

  PetscFunctionBegin;
  if (lu->CleanUpPastix) {
    /* Terminate instance, deallocate memories */
    PetscCall(VecScatterDestroy(&lu->scat_rhs));
    PetscCall(VecDestroy(&lu->b_seq));
    PetscCall(VecScatterDestroy(&lu->scat_sol));

    lu->iparm[IPARM_START_TASK]=API_TASK_CLEAN;
    lu->iparm[IPARM_END_TASK]  =API_TASK_CLEAN;

    PASTIX_CALL(&(lu->pastix_data),
                lu->pastix_comm,
                lu->n,
                lu->colptr,
                lu->row,
                (PastixScalar*)lu->val,
                lu->perm,
                lu->invp,
                (PastixScalar*)lu->rhs,
                lu->rhsnbr,
                lu->iparm,
                lu->dparm);
    PetscCheck(lu->iparm[IPARM_ERROR_NUMBER] == 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by PaStiX in destroy: iparm(IPARM_ERROR_NUMBER)=%" PetscInt_FMT,lu->iparm[IPARM_ERROR_NUMBER]);
    PetscCall(PetscFree(lu->colptr));
    PetscCall(PetscFree(lu->row));
    PetscCall(PetscFree(lu->val));
    PetscCall(PetscFree(lu->perm));
    PetscCall(PetscFree(lu->invp));
    PetscCallMPI(MPI_Comm_free(&(lu->pastix_comm)));
  }
  PetscCall(PetscFree(A->data));
  PetscFunctionReturn(0);
}

/*
  Gather right-hand-side.
  Call for Solve step.
  Scatter solution.
 */
PetscErrorCode MatSolve_PaStiX(Mat A,Vec b,Vec x)
{
  Mat_Pastix  *lu = (Mat_Pastix*)A->data;
  PetscScalar *array;
  Vec          x_seq;

  PetscFunctionBegin;
  lu->rhsnbr = 1;
  x_seq      = lu->b_seq;
  if (lu->commSize > 1) {
    /* PaStiX only supports centralized rhs. Scatter b into a sequential rhs vector */
    PetscCall(VecScatterBegin(lu->scat_rhs,b,x_seq,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(lu->scat_rhs,b,x_seq,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecGetArray(x_seq,&array));
  } else {  /* size == 1 */
    PetscCall(VecCopy(b,x));
    PetscCall(VecGetArray(x,&array));
  }
  lu->rhs = array;
  if (lu->commSize == 1) {
    PetscCall(VecRestoreArray(x,&array));
  } else {
    PetscCall(VecRestoreArray(x_seq,&array));
  }

  /* solve phase */
  /*-------------*/
  lu->iparm[IPARM_START_TASK] = API_TASK_SOLVE;
  lu->iparm[IPARM_END_TASK]   = API_TASK_REFINE;
  lu->iparm[IPARM_RHS_MAKING] = API_RHS_B;

  PASTIX_CALL(&(lu->pastix_data),
              lu->pastix_comm,
              lu->n,
              lu->colptr,
              lu->row,
              (PastixScalar*)lu->val,
              lu->perm,
              lu->invp,
              (PastixScalar*)lu->rhs,
              lu->rhsnbr,
              lu->iparm,
              lu->dparm);
  PetscCheck(lu->iparm[IPARM_ERROR_NUMBER] == 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by PaStiX in solve phase: lu->iparm[IPARM_ERROR_NUMBER] = %" PetscInt_FMT,lu->iparm[IPARM_ERROR_NUMBER]);

  if (lu->commSize == 1) {
    PetscCall(VecRestoreArray(x,&(lu->rhs)));
  } else {
    PetscCall(VecRestoreArray(x_seq,&(lu->rhs)));
  }

  if (lu->commSize > 1) { /* convert PaStiX centralized solution to petsc mpi x */
    PetscCall(VecScatterBegin(lu->scat_sol,x_seq,x,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(lu->scat_sol,x_seq,x,INSERT_VALUES,SCATTER_FORWARD));
  }
  PetscFunctionReturn(0);
}

/*
  Numeric factorisation using PaStiX solver.

 */
PetscErrorCode MatFactorNumeric_PaStiX(Mat F,Mat A,const MatFactorInfo *info)
{
  Mat_Pastix     *lu =(Mat_Pastix*)(F)->data;
  Mat            *tseq;
  PetscInt       icntl;
  PetscInt       M=A->rmap->N;
  PetscBool      valOnly,flg, isSym;
  IS             is_iden;
  Vec            b;
  IS             isrow;
  PetscBool      isSeqAIJ,isSeqSBAIJ,isMPIAIJ;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&isSeqAIJ));
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATMPIAIJ,&isMPIAIJ));
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATSEQSBAIJ,&isSeqSBAIJ));
  if (lu->matstruc == DIFFERENT_NONZERO_PATTERN) {
    (F)->ops->solve = MatSolve_PaStiX;

    /* Initialize a PASTIX instance */
    PetscCallMPI(MPI_Comm_dup(PetscObjectComm((PetscObject)A),&(lu->pastix_comm)));
    PetscCallMPI(MPI_Comm_rank(lu->pastix_comm, &lu->commRank));
    PetscCallMPI(MPI_Comm_size(lu->pastix_comm, &lu->commSize));

    /* Set pastix options */
    lu->iparm[IPARM_MODIFY_PARAMETER] = API_NO;
    lu->iparm[IPARM_START_TASK]       = API_TASK_INIT;
    lu->iparm[IPARM_END_TASK]         = API_TASK_INIT;

    lu->rhsnbr = 1;

    /* Call to set default pastix options */
    PASTIX_CALL(&(lu->pastix_data),
                lu->pastix_comm,
                lu->n,
                lu->colptr,
                lu->row,
                (PastixScalar*)lu->val,
                lu->perm,
                lu->invp,
                (PastixScalar*)lu->rhs,
                lu->rhsnbr,
                lu->iparm,
                lu->dparm);
    PetscCheck(lu->iparm[IPARM_ERROR_NUMBER] == 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by PaStiX in MatFactorNumeric: iparm(IPARM_ERROR_NUMBER)=%" PetscInt_FMT,lu->iparm[IPARM_ERROR_NUMBER]);

    PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"PaStiX Options","Mat");
    icntl = -1;
    lu->iparm[IPARM_VERBOSE] = API_VERBOSE_NOT;
    PetscCall(PetscOptionsInt("-mat_pastix_verbose","iparm[IPARM_VERBOSE] : level of printing (0 to 2)","None",lu->iparm[IPARM_VERBOSE],&icntl,&flg));
    if ((flg && icntl >= 0) || PetscLogPrintInfo) {
      lu->iparm[IPARM_VERBOSE] =  icntl;
    }
    icntl=-1;
    PetscCall(PetscOptionsInt("-mat_pastix_threadnbr","iparm[IPARM_THREAD_NBR] : Number of thread by MPI node","None",lu->iparm[IPARM_THREAD_NBR],&icntl,&flg));
    if ((flg && icntl > 0)) {
      lu->iparm[IPARM_THREAD_NBR] = icntl;
    }
    PetscOptionsEnd();
    valOnly = PETSC_FALSE;
  } else {
    if (isSeqAIJ || isMPIAIJ) {
      PetscCall(PetscFree(lu->colptr));
      PetscCall(PetscFree(lu->row));
      PetscCall(PetscFree(lu->val));
      valOnly = PETSC_FALSE;
    } else valOnly = PETSC_TRUE;
  }

  lu->iparm[IPARM_MATRIX_VERIFICATION] = API_YES;

  /* convert mpi A to seq mat A */
  PetscCall(ISCreateStride(PETSC_COMM_SELF,M,0,1,&isrow));
  PetscCall(MatCreateSubMatrices(A,1,&isrow,&isrow,MAT_INITIAL_MATRIX,&tseq));
  PetscCall(ISDestroy(&isrow));

  PetscCall(MatConvertToCSC(*tseq,valOnly, &lu->n, &lu->colptr, &lu->row, &lu->val));
  PetscCall(MatIsSymmetric(*tseq,0.0,&isSym));
  PetscCall(MatDestroyMatrices(1,&tseq));

  if (!lu->perm) {
    PetscCall(PetscMalloc1(lu->n,&(lu->perm)));
    PetscCall(PetscMalloc1(lu->n,&(lu->invp)));
  }

  if (isSym) {
    /* On symmetric matrix, LLT */
    lu->iparm[IPARM_SYM]           = API_SYM_YES;
    lu->iparm[IPARM_FACTORIZATION] = API_FACT_LDLT;
  } else {
    /* On unsymmetric matrix, LU */
    lu->iparm[IPARM_SYM]           = API_SYM_NO;
    lu->iparm[IPARM_FACTORIZATION] = API_FACT_LU;
  }

  /*----------------*/
  if (lu->matstruc == DIFFERENT_NONZERO_PATTERN) {
    if (!(isSeqAIJ || isSeqSBAIJ) && !lu->b_seq) {
      /* PaStiX only supports centralized rhs. Create scatter scat_rhs for repeated use in MatSolve() */
      PetscCall(VecCreateSeq(PETSC_COMM_SELF,A->cmap->N,&lu->b_seq));
      PetscCall(ISCreateStride(PETSC_COMM_SELF,A->cmap->N,0,1,&is_iden));
      PetscCall(MatCreateVecs(A,NULL,&b));
      PetscCall(VecScatterCreate(b,is_iden,lu->b_seq,is_iden,&lu->scat_rhs));
      PetscCall(VecScatterCreate(lu->b_seq,is_iden,b,is_iden,&lu->scat_sol));
      PetscCall(ISDestroy(&is_iden));
      PetscCall(VecDestroy(&b));
    }
    lu->iparm[IPARM_START_TASK] = API_TASK_ORDERING;
    lu->iparm[IPARM_END_TASK]   = API_TASK_NUMFACT;

    PASTIX_CALL(&(lu->pastix_data),
                lu->pastix_comm,
                lu->n,
                lu->colptr,
                lu->row,
                (PastixScalar*)lu->val,
                lu->perm,
                lu->invp,
                (PastixScalar*)lu->rhs,
                lu->rhsnbr,
                lu->iparm,
                lu->dparm);
    PetscCheck(lu->iparm[IPARM_ERROR_NUMBER] == 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by PaStiX in analysis phase: iparm(IPARM_ERROR_NUMBER)=%" PetscInt_FMT,lu->iparm[IPARM_ERROR_NUMBER]);
  } else {
    lu->iparm[IPARM_START_TASK] = API_TASK_NUMFACT;
    lu->iparm[IPARM_END_TASK]   = API_TASK_NUMFACT;
    PASTIX_CALL(&(lu->pastix_data),
                lu->pastix_comm,
                lu->n,
                lu->colptr,
                lu->row,
                (PastixScalar*)lu->val,
                lu->perm,
                lu->invp,
                (PastixScalar*)lu->rhs,
                lu->rhsnbr,
                lu->iparm,
                lu->dparm);
    PetscCheck(lu->iparm[IPARM_ERROR_NUMBER] == 0,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error reported by PaStiX in analysis phase: iparm(IPARM_ERROR_NUMBER)=%" PetscInt_FMT,lu->iparm[IPARM_ERROR_NUMBER]);
  }

  (F)->assembled    = PETSC_TRUE;
  lu->matstruc      = SAME_NONZERO_PATTERN;
  lu->CleanUpPastix = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/* Note the Petsc r and c permutations are ignored */
PetscErrorCode MatLUFactorSymbolic_AIJPASTIX(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  Mat_Pastix *lu = (Mat_Pastix*)F->data;

  PetscFunctionBegin;
  lu->iparm[IPARM_FACTORIZATION] = API_FACT_LU;
  lu->iparm[IPARM_SYM]           = API_SYM_YES;
  lu->matstruc                   = DIFFERENT_NONZERO_PATTERN;
  F->ops->lufactornumeric        = MatFactorNumeric_PaStiX;
  PetscFunctionReturn(0);
}

PetscErrorCode MatCholeskyFactorSymbolic_SBAIJPASTIX(Mat F,Mat A,IS r,const MatFactorInfo *info)
{
  Mat_Pastix *lu = (Mat_Pastix*)(F)->data;

  PetscFunctionBegin;
  lu->iparm[IPARM_FACTORIZATION]  = API_FACT_LLT;
  lu->iparm[IPARM_SYM]            = API_SYM_NO;
  lu->matstruc                    = DIFFERENT_NONZERO_PATTERN;
  (F)->ops->choleskyfactornumeric = MatFactorNumeric_PaStiX;
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_PaStiX(Mat A,PetscViewer viewer)
{
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscViewerGetFormat(viewer,&format));
    if (format == PETSC_VIEWER_ASCII_INFO) {
      Mat_Pastix *lu=(Mat_Pastix*)A->data;

      PetscCall(PetscViewerASCIIPrintf(viewer,"PaStiX run parameters:\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Matrix type :                      %s \n",((lu->iparm[IPARM_SYM] == API_SYM_YES) ? "Symmetric" : "Unsymmetric")));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Level of printing (0,1,2):         %" PetscInt_FMT " \n",lu->iparm[IPARM_VERBOSE]));
      PetscCall(PetscViewerASCIIPrintf(viewer,"  Number of refinements iterations : %" PetscInt_FMT " \n",lu->iparm[IPARM_NBITER]));
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"  Error :                        %g \n",lu->dparm[DPARM_RELATIVE_ERROR]));
    }
  }
  PetscFunctionReturn(0);
}

/*MC
     MATSOLVERPASTIX  - A solver package providing direct solvers (LU) for distributed
  and sequential matrices via the external package PaStiX.

  Use ./configure --download-pastix --download-ptscotch  to have PETSc installed with PasTiX

  Use -pc_type lu -pc_factor_mat_solver_type pastix to use this direct solver

  Options Database Keys:
+ -mat_pastix_verbose   <0,1,2>   - print level
- -mat_pastix_threadnbr <integer> - Set the thread number by MPI task.

  Notes:
    This only works for matrices with symmetric nonzero structure, if you pass it a matrix with
   nonsymmetric structure PasTiX and hence PETSc return with an error.

  Level: beginner

.seealso: `PCFactorSetMatSolverType()`, `MatSolverType`

M*/

PetscErrorCode MatGetInfo_PaStiX(Mat A,MatInfoType flag,MatInfo *info)
{
  Mat_Pastix *lu =(Mat_Pastix*)A->data;

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

static PetscErrorCode MatFactorGetSolverType_pastix(Mat A,MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERPASTIX;
  PetscFunctionReturn(0);
}

/*
    The seq and mpi versions of this function are the same
*/
static PetscErrorCode MatGetFactor_seqaij_pastix(Mat A,MatFactorType ftype,Mat *F)
{
  Mat         B;
  Mat_Pastix *pastix;

  PetscFunctionBegin;
  PetscCheck(ftype == MAT_FACTOR_LU,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot use PETSc AIJ matrices with PaStiX Cholesky, use SBAIJ matrix");
  /* Create the factorization matrix */
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&B));
  PetscCall(MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N));
  PetscCall(PetscStrallocpy("pastix",&((PetscObject)B)->type_name));
  PetscCall(MatSetUp(B));

  B->trivialsymbolic       = PETSC_TRUE;
  B->ops->lufactorsymbolic = MatLUFactorSymbolic_AIJPASTIX;
  B->ops->view             = MatView_PaStiX;
  B->ops->getinfo          = MatGetInfo_PaStiX;

  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_pastix));

  B->factortype = MAT_FACTOR_LU;

  /* set solvertype */
  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERPASTIX,&B->solvertype));

  PetscCall(PetscNewLog(B,&pastix));

  pastix->CleanUpPastix = PETSC_FALSE;
  pastix->scat_rhs      = NULL;
  pastix->scat_sol      = NULL;
  B->ops->getinfo       = MatGetInfo_External;
  B->ops->destroy       = MatDestroy_Pastix;
  B->data               = (void*)pastix;

  *F = B;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetFactor_mpiaij_pastix(Mat A,MatFactorType ftype,Mat *F)
{
  Mat         B;
  Mat_Pastix *pastix;

  PetscFunctionBegin;
  PetscCheck(ftype == MAT_FACTOR_LU,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot use PETSc AIJ matrices with PaStiX Cholesky, use SBAIJ matrix");
  /* Create the factorization matrix */
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&B));
  PetscCall(MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N));
  PetscCall(PetscStrallocpy("pastix",&((PetscObject)B)->type_name));
  PetscCall(MatSetUp(B));

  B->trivialsymbolic       = PETSC_TRUE;
  B->ops->lufactorsymbolic = MatLUFactorSymbolic_AIJPASTIX;
  B->ops->view             = MatView_PaStiX;
  B->ops->getinfo          = MatGetInfo_PaStiX;
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_pastix));

  B->factortype = MAT_FACTOR_LU;

  /* set solvertype */
  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERPASTIX,&B->solvertype));

  PetscCall(PetscNewLog(B,&pastix));

  pastix->CleanUpPastix = PETSC_FALSE;
  pastix->scat_rhs      = NULL;
  pastix->scat_sol      = NULL;
  B->ops->getinfo       = MatGetInfo_External;
  B->ops->destroy       = MatDestroy_Pastix;
  B->data               = (void*)pastix;

  *F = B;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetFactor_seqsbaij_pastix(Mat A,MatFactorType ftype,Mat *F)
{
  Mat         B;
  Mat_Pastix *pastix;

  PetscFunctionBegin;
  PetscCheck(ftype == MAT_FACTOR_CHOLESKY,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot use PETSc SBAIJ matrices with PaStiX LU, use AIJ matrix");
  /* Create the factorization matrix */
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&B));
  PetscCall(MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N));
  PetscCall(PetscStrallocpy("pastix",&((PetscObject)B)->type_name));
  PetscCall(MatSetUp(B));

  B->trivialsymbolic             = PETSC_TRUE;
  B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SBAIJPASTIX;
  B->ops->view                   = MatView_PaStiX;
  B->ops->getinfo                = MatGetInfo_PaStiX;
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_pastix));

  B->factortype = MAT_FACTOR_CHOLESKY;

  /* set solvertype */
  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERPASTIX,&B->solvertype));

  PetscCall(PetscNewLog(B,&pastix));

  pastix->CleanUpPastix = PETSC_FALSE;
  pastix->scat_rhs      = NULL;
  pastix->scat_sol      = NULL;
  B->ops->getinfo       = MatGetInfo_External;
  B->ops->destroy       = MatDestroy_Pastix;
  B->data               = (void*)pastix;
  *F = B;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetFactor_mpisbaij_pastix(Mat A,MatFactorType ftype,Mat *F)
{
  Mat         B;
  Mat_Pastix *pastix;

  PetscFunctionBegin;
  PetscCheck(ftype == MAT_FACTOR_CHOLESKY,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot use PETSc SBAIJ matrices with PaStiX LU, use AIJ matrix");

  /* Create the factorization matrix */
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A),&B));
  PetscCall(MatSetSizes(B,A->rmap->n,A->cmap->n,A->rmap->N,A->cmap->N));
  PetscCall(PetscStrallocpy("pastix",&((PetscObject)B)->type_name));
  PetscCall(MatSetUp(B));

  B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_SBAIJPASTIX;
  B->ops->view                   = MatView_PaStiX;
  B->ops->getinfo                = MatGetInfo_PaStiX;
  B->ops->destroy                = MatDestroy_Pastix;
  PetscCall(PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_pastix));

  B->factortype = MAT_FACTOR_CHOLESKY;

  /* set solvertype */
  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERPASTIX,&B->solvertype));

  PetscCall(PetscNewLog(B,&pastix));

  pastix->CleanUpPastix = PETSC_FALSE;
  pastix->scat_rhs      = NULL;
  pastix->scat_sol      = NULL;
  B->data               = (void*)pastix;

  *F = B;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_Pastix(void)
{
  PetscFunctionBegin;
  PetscCall(MatSolverTypeRegister(MATSOLVERPASTIX,MATMPIAIJ,        MAT_FACTOR_LU,MatGetFactor_mpiaij_pastix));
  PetscCall(MatSolverTypeRegister(MATSOLVERPASTIX,MATSEQAIJ,        MAT_FACTOR_LU,MatGetFactor_seqaij_pastix));
  PetscCall(MatSolverTypeRegister(MATSOLVERPASTIX,MATMPISBAIJ,      MAT_FACTOR_CHOLESKY,MatGetFactor_mpisbaij_pastix));
  PetscCall(MatSolverTypeRegister(MATSOLVERPASTIX,MATSEQSBAIJ,      MAT_FACTOR_CHOLESKY,MatGetFactor_seqsbaij_pastix));
  PetscFunctionReturn(0);
}
