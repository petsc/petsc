/*$Id: superlu.c,v 1.10 2001/08/15 15:56:50 bsmith Exp $*/

/* 
        Provides an interface to the SuperLU sparse solver
          Modified for SuperLU 2.0 by Matthew Knepley
*/

#include "src/mat/impls/aij/seq/aij.h"

EXTERN_C_BEGIN
#if defined(PETSC_USE_COMPLEX)
#include "zsp_defs.h"
#else
#include "dsp_defs.h"
#endif  
#include "util.h"
EXTERN_C_END

typedef struct {
  SuperMatrix  A,B,AC,L,U;
  int          *perm_r,*perm_c,ispec,relax,panel_size;
  double       pivot_threshold;
  NCformat     *store;
  MatStructure flg;
  PetscTruth   SuperluMatOdering;

  /* A few function pointers for inheritance */
  int (*MatView)(Mat,PetscViewer);
  int (*MatAssemblyEnd)(Mat,MatAssemblyType);
  int (*MatDestroy)(Mat);

  /* Flag to clean up (non-global) SuperLU objects during Destroy */
  PetscTruth CleanUpSuperLU;
} Mat_SeqAIJ_SuperLU;


extern int MatDestroy_SeqAIJ(Mat);

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_SeqAIJ_SuperLU"
int MatDestroy_SeqAIJ_SuperLU(Mat A)
{
  Mat_SeqAIJ_SuperLU *lu = (Mat_SeqAIJ_SuperLU*)A->spptr;
  
  int                ierr,(*destroy)(Mat);

  PetscFunctionBegin;
  /* It looks like this is decreasing the reference count a second time during MatDestroy?! */
  if (--A->refct > 0)PetscFunctionReturn(0);
  /* We have to free the global data or SuperLU crashes (sucky design)*/
  /* Since we don't know if more solves on other matrices may be done
     we cannot free the yucky SuperLU global data
    StatFree(); 
  */

  /* Free the SuperLU datastructures */
  if (lu->CleanUpSuperLU) {
    Destroy_CompCol_Permuted(&lu->AC);
    Destroy_SuperNode_Matrix(&lu->L);
    Destroy_CompCol_Matrix(&lu->U);
    ierr = PetscFree(lu->B.Store);CHKERRQ(ierr);
    ierr = PetscFree(lu->A.Store);CHKERRQ(ierr);
    ierr = PetscFree(lu->perm_r);CHKERRQ(ierr);
    ierr = PetscFree(lu->perm_c);CHKERRQ(ierr);
  }

  destroy = lu->MatDestroy;
  ierr = PetscFree(lu);CHKERRQ(ierr);
  ierr = (*destroy)(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_SeqAIJ_Spooles"
int MatView_SeqAIJ_SuperLU(Mat A,PetscViewer viewer)
{
  int                   ierr;
  PetscTruth            isascii;
  PetscViewerFormat     format;
  Mat_SeqAIJ_SuperLU   *lu=(Mat_SeqAIJ_SuperLU*)(A->spptr);

  PetscFunctionBegin;
  ierr = (*lu->MatView)(A,viewer);CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
      ierr = MatSeqAIJFactorInfo_SuperLU(A,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_SeqAIJ_SuperLU"
int MatAssemblyEnd_SeqAIJ_SuperLU(Mat A,MatAssemblyType mode) {
  int                ierr;
  Mat_SeqAIJ_SuperLU *lu=(Mat_SeqAIJ_SuperLU*)(A->spptr);

  PetscFunctionBegin;
  ierr = (*lu->MatAssemblyEnd)(A,mode);CHKERRQ(ierr);
  ierr = MatUseSuperLU_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include "src/mat/impls/dense/seq/dense.h"
#undef __FUNCT__  
#define __FUNCT__ "MatCreateNull_SeqAIJ_SuperLU"
int MatCreateNull_SeqAIJ_SuperLU(Mat A,Mat *nullMat)
{
  Mat_SeqAIJ_SuperLU  *lu = (Mat_SeqAIJ_SuperLU*)A->spptr;
  int                 numRows = A->m,numCols = A->n;
  SCformat            *Lstore;
  int                 numNullCols,size;
#if defined(PETSC_USE_COMPLEX)
  doublecomplex       *nullVals,*workVals;
#else
  PetscScalar         *nullVals,*workVals;
#endif
  int                 row,newRow,col,newCol,block,b,ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE);
  PetscValidPointer(nullMat);
  if (!A->factor) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Unfactored matrix");
  numNullCols = numCols - numRows;
  if (numNullCols < 0) SETERRQ(PETSC_ERR_ARG_WRONG,"Function only applies to underdetermined problems");
  /* Create the null matrix */
  ierr = MatCreateSeqDense(A->comm,numRows,numNullCols,PETSC_NULL,nullMat);CHKERRQ(ierr);
  if (numNullCols == 0) {
    ierr = MatAssemblyBegin(*nullMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*nullMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
#if defined(PETSC_USE_COMPLEX)
  nullVals = (doublecomplex*)((Mat_SeqDense*)(*nullMat)->data)->v;
#else
  nullVals = ((Mat_SeqDense*)(*nullMat)->data)->v;
#endif
  /* Copy in the columns */
  Lstore = (SCformat*)lu->L.Store;
  for(block = 0; block <= Lstore->nsuper; block++) {
    newRow = Lstore->sup_to_col[block];
    size   = Lstore->sup_to_col[block+1] - Lstore->sup_to_col[block];
    for(col = Lstore->rowind_colptr[newRow]; col < Lstore->rowind_colptr[newRow+1]; col++) {
      newCol = Lstore->rowind[col];
      if (newCol >= numRows) {
        for(b = 0; b < size; b++)
#if defined(PETSC_USE_COMPLEX)
          nullVals[(newCol-numRows)*numRows+newRow+b] = ((doublecomplex*)Lstore->nzval)[Lstore->nzval_colptr[newRow+b]+col];
#else
          nullVals[(newCol-numRows)*numRows+newRow+b] = ((double*)Lstore->nzval)[Lstore->nzval_colptr[newRow+b]+col];
#endif
      }
    }
  }
  /* Permute rhs to form P^T_c B */
  ierr = PetscMalloc(numRows*sizeof(double),&workVals);CHKERRQ(ierr);
  for(b = 0; b < numNullCols; b++) {
    for(row = 0; row < numRows; row++) workVals[lu->perm_c[row]] = nullVals[b*numRows+row];
    for(row = 0; row < numRows; row++) nullVals[b*numRows+row]   = workVals[row];
  }
  /* Backward solve the upper triangle A x = b */
  for(b = 0; b < numNullCols; b++) {
#if defined(PETSC_USE_COMPLEX)
    sp_ztrsv("L","T","U",&lu->L,&lu->U,&nullVals[b*numRows],&ierr);
#else
    sp_dtrsv("L","T","U",&lu->L,&lu->U,&nullVals[b*numRows],&ierr);
#endif
    if (ierr < 0)
      SETERRQ1(PETSC_ERR_ARG_WRONG,"The argument %d was invalid",-ierr);
  }
  ierr = PetscFree(workVals);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*nullMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*nullMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SeqAIJ_SuperLU"
int MatSolve_SeqAIJ_SuperLU(Mat A,Vec b,Vec x)
{
  Mat_SeqAIJ_SuperLU *lu = (Mat_SeqAIJ_SuperLU*)A->spptr;
  PetscScalar        *array;
  int                m,ierr;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(b,&m);CHKERRQ(ierr);
  ierr = VecCopy(b,x);CHKERRQ(ierr);
  ierr = VecGetArray(x,&array);CHKERRQ(ierr);
  /* Create the Rhs */
  lu->B.Stype        = SLU_DN;
  lu->B.Mtype        = SLU_GE;
  lu->B.nrow         = m;
  lu->B.ncol         = 1;
  ((DNformat*)lu->B.Store)->lda   = m;
  ((DNformat*)lu->B.Store)->nzval = array;
#if defined(PETSC_USE_COMPLEX)
  lu->B.Dtype        = SLU_Z;
  zgstrs("T",&lu->L,&lu->U,lu->perm_r,lu->perm_c,&lu->B,&ierr);
#else
  lu->B.Dtype        = SLU_D;
  dgstrs("T",&lu->L,&lu->U,lu->perm_r,lu->perm_c,&lu->B,&ierr);
#endif
  if (ierr < 0) SETERRQ1(PETSC_ERR_ARG_WRONG,"The diagonal element of row %d was invalid",-ierr);
  ierr = VecRestoreArray(x,&array);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static int StatInitCalled = 0;

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_SeqAIJ_SuperLU"
int MatLUFactorNumeric_SeqAIJ_SuperLU(Mat A,Mat *F)
{
  Mat_SeqAIJ         *aa = (Mat_SeqAIJ*)(A)->data;
  Mat_SeqAIJ_SuperLU *lu = (Mat_SeqAIJ_SuperLU*)(*F)->spptr;
  int                *etree,i,ierr;
  PetscTruth         flag;

  PetscFunctionBegin;
  /* Create the SuperMatrix for A^T:
       Since SuperLU only likes column-oriented matrices,we pass it the transpose,
       and then solve A^T X = B in MatSolve().
  */
  
  if ( lu->flg == DIFFERENT_NONZERO_PATTERN){ /* first numerical factorization */
    lu->A.Stype   = SLU_NC;
#if defined(PETSC_USE_COMPLEX)
    lu->A.Dtype   = SLU_Z;
#else
    lu->A.Dtype   = SLU_D;
#endif
    lu->A.Mtype   = SLU_GE;
    lu->A.nrow    = A->n;
    lu->A.ncol    = A->m;
  
    ierr = PetscMalloc(sizeof(NCformat),&lu->store);CHKERRQ(ierr); 
    ierr = PetscMalloc(sizeof(DNformat),&lu->B.Store);CHKERRQ(ierr);
  }
  lu->store->nnz    = aa->nz;
  lu->store->colptr = aa->i;
  lu->store->rowind = aa->j;
  lu->store->nzval  = aa->a; 
  lu->A.Store       = lu->store; 
  
  /* Set SuperLU options */
  lu->relax      = sp_ienv(2);
  lu->panel_size = sp_ienv(1);
  /* We have to initialize global data or SuperLU crashes (sucky design) */
  if (!StatInitCalled) {
    StatInit(lu->panel_size,lu->relax);
  }
  StatInitCalled++;

  ierr = PetscOptionsBegin(A->comm,A->prefix,"SuperLU Options","Mat");CHKERRQ(ierr);
  /* use SuperLU mat ordeing */
  ierr = PetscOptionsInt("-mat_superlu_ordering","SuperLU ordering type (one of 0, 1, 2, 3)\n   0: natural ordering;\n   1: MMD applied to A'*A;\n   2: MMD applied to A'+A;\n   3: COLAMD, approximate minimum degree column ordering","None",lu->ispec,&lu->ispec,&flag);CHKERRQ(ierr);
  if (flag) {
    get_perm_c(lu->ispec, &lu->A, lu->perm_c);
    lu->SuperluMatOdering = PETSC_TRUE; 
  }
  PetscOptionsEnd();

  /* Create the elimination tree */
  ierr = PetscMalloc(A->n*sizeof(int),&etree);CHKERRQ(ierr);
  sp_preorder("N",&lu->A,lu->perm_c,etree,&lu->AC);
  /* Factor the matrix */
#if defined(PETSC_USE_COMPLEX)
  zgstrf("N",&lu->AC,lu->pivot_threshold,0.0,lu->relax,lu->panel_size,etree,PETSC_NULL,0,lu->perm_r,lu->perm_c,&lu->L,&lu->U,&ierr);
#else
  dgstrf("N",&lu->AC,lu->pivot_threshold,0.0,lu->relax,lu->panel_size,etree,PETSC_NULL,0,lu->perm_r,lu->perm_c,&lu->L,&lu->U,&ierr);
#endif
  if (ierr < 0) {
    SETERRQ1(PETSC_ERR_ARG_WRONG,"The diagonal element of row %d was invalid",-ierr);
  } else if (ierr > 0) {
    if (ierr <= A->m) {
      SETERRQ1(PETSC_ERR_ARG_WRONG,"The diagonal element %d of U is exactly zero",ierr);
    } else {
      SETERRQ1(PETSC_ERR_ARG_WRONG,"Memory allocation failure after %d bytes were allocated",ierr-A->m);
    }
  }

  /* Cleanup */
  ierr = PetscFree(etree);CHKERRQ(ierr);

  lu->flg = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

/*
   Note the r permutation is ignored
*/
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_SeqAIJ_SuperLU"
int MatLUFactorSymbolic_SeqAIJ_SuperLU(Mat A,IS r,IS c,MatFactorInfo *info,Mat *F)
{
  Mat                 B;
  Mat_SeqAIJ_SuperLU  *lu;
  int                 ierr,*ca;

  PetscFunctionBegin;
  
  ierr            = MatCreateSeqAIJ(A->comm,A->m,A->n,0,PETSC_NULL,F);CHKERRQ(ierr);
  B               = *F;
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJ_SuperLU;
  B->ops->solve           = MatSolve_SeqAIJ_SuperLU;
  B->ops->destroy         = MatDestroy_SeqAIJ_SuperLU;
  B->factor               = FACTOR_LU;
  (*F)->assembled         = PETSC_TRUE;  /* required by -sles_view */
  
  ierr            = PetscNew(Mat_SeqAIJ_SuperLU,&lu);CHKERRQ(ierr);
  B->spptr        = (void*)lu;
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatCreateNull","MatCreateNull_SeqAIJ_SuperLU",
                                    (void(*)(void))MatCreateNull_SeqAIJ_SuperLU);CHKERRQ(ierr);

  /* Allocate the work arrays required by SuperLU (notice sizes are for the transpose) */
  ierr = PetscMalloc(A->n*sizeof(int),&lu->perm_r);CHKERRQ(ierr);
  ierr = PetscMalloc(A->m*sizeof(int),&lu->perm_c);CHKERRQ(ierr);

  /* use PETSc mat ordering */  
  ierr = ISGetIndices(c,&ca);CHKERRQ(ierr);
  ierr = PetscMemcpy(lu->perm_c,ca,A->m*sizeof(int));CHKERRQ(ierr);
  ierr = ISRestoreIndices(c,&ca);CHKERRQ(ierr);
  lu->SuperluMatOdering = PETSC_FALSE;

  lu->pivot_threshold = info->dtcol; 
  PetscLogObjectMemory(B,(A->m+A->n)*sizeof(int)+sizeof(Mat_SeqAIJ_SuperLU));

  lu->flg            = DIFFERENT_NONZERO_PATTERN;
  lu->CleanUpSuperLU = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatUseSuperLU_SeqAIJ"
int MatUseSuperLU_SeqAIJ(Mat A)
{
  PetscTruth flg;
  int        ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_COOKIE);  
  ierr = PetscTypeCompare((PetscObject)A,MATSUPERLU,&flg);
  if (!flg) PetscFunctionReturn(0);

  A->ops->lufactorsymbolic = MatLUFactorSymbolic_SeqAIJ_SuperLU;

  PetscFunctionReturn(0);
}

/* used by -sles_view */
#undef __FUNCT__  
#define __FUNCT__ "MatSeqAIJFactorInfo_SuperLU"
int MatSeqAIJFactorInfo_SuperLU(Mat A,PetscViewer viewer)
{
  Mat_SeqAIJ_SuperLU      *lu= (Mat_SeqAIJ_SuperLU*)A->spptr;
  int                     ierr;
  PetscFunctionBegin;
  /* check if matrix is SuperLU type */
  if (A->ops->solve != MatSolve_SeqAIJ_SuperLU) PetscFunctionReturn(0);

  ierr = PetscViewerASCIIPrintf(viewer,"SuperLU run parameters:\n");CHKERRQ(ierr);
  if(lu->SuperluMatOdering) ierr = PetscViewerASCIIPrintf(viewer,"  SuperLU mat ordering: %d\n",lu->ispec);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_SeqAIJ_SuperLU"
int MatCreate_SeqAIJ_SuperLU(Mat A) {
  int                ierr;
  Mat_SeqAIJ_SuperLU *lu;

  PetscFunctionBegin;
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatUseSuperLU_SeqAIJ(A);CHKERRQ(ierr);

  ierr                = PetscNew(Mat_SeqAIJ_SuperLU,&lu);CHKERRQ(ierr);
  lu->MatView         = A->ops->view;
  lu->MatAssemblyEnd  = A->ops->assemblyend;
  lu->MatDestroy      = A->ops->destroy;
  lu->CleanUpSuperLU  = PETSC_FALSE;
  A->spptr            = (void*)lu;
  A->ops->view        = MatView_SeqAIJ_SuperLU;
  A->ops->assemblyend = MatAssemblyEnd_SeqAIJ_SuperLU;
  A->ops->destroy     = MatDestroy_SeqAIJ_SuperLU;
  PetscFunctionReturn(0);
}
EXTERN_C_END
