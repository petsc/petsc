/*$Id: superlu.c,v 1.10 2001/08/15 15:56:50 bsmith Exp $*/

/* 
        Provides an interface to the SuperLU 3.0 sparse solver
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
  SuperMatrix       A,L,U,B,X;
  superlu_options_t options;
  int               *perm_c; /* column permutation vector */
  int               *perm_r; /* row permutations from partial pivoting */
  int               *etree;
  double            *R, *C;
  char              equed[1];
  int               lwork;
  void              *work;
  double            rpg, rcond;
  mem_usage_t       mem_usage;
  MatStructure      flg;
  /*
  SuperMatrix  A,B,AC,L,U;
  int          *perm_r,*perm_c,ispec,relax,panel_size;
  double       pivot_threshold;
  NCformat     *store;
  MatStructure flg;
  PetscTruth   SuperluMatOdering;
  */

  /* A few function pointers for inheritance */
  int (*MatDuplicate)(Mat,MatDuplicateOption,Mat*);
  int (*MatView)(Mat,PetscViewer);
  int (*MatAssemblyEnd)(Mat,MatAssemblyType);
  int (*MatLUFactorSymbolic)(Mat,IS,IS,MatFactorInfo*,Mat*);
  int (*MatDestroy)(Mat);

  /* Flag to clean up (non-global) SuperLU objects during Destroy */
  PetscTruth CleanUpSuperLU;
} Mat_SuperLU;


EXTERN int MatFactorInfo_SuperLU(Mat,PetscViewer);
EXTERN int MatLUFactorSymbolic_SuperLU(Mat,IS,IS,MatFactorInfo*,Mat*);

EXTERN_C_BEGIN
EXTERN int MatConvert_SuperLU_SeqAIJ(Mat,MatType,Mat*);
EXTERN int MatConvert_SeqAIJ_SuperLU(Mat,MatType,Mat*);
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "MatDestroy_SuperLU"
int MatDestroy_SuperLU(Mat A)
{
  int         ierr;
  Mat_SuperLU *lu = (Mat_SuperLU*)A->spptr;

  PetscFunctionBegin;
  if (lu->CleanUpSuperLU) { /* Free the SuperLU datastructures */
    /* Destroy_CompCol_Matrix(&lu->A);  */  /* hangs inside memory.c! */
    Destroy_SuperMatrix_Store(&lu->A); 
    Destroy_SuperMatrix_Store(&lu->B);
    Destroy_SuperMatrix_Store(&lu->X); 

    ierr = PetscFree(lu->etree);CHKERRQ(ierr);
    ierr = PetscFree(lu->perm_r);CHKERRQ(ierr);
    ierr = PetscFree(lu->perm_c);CHKERRQ(ierr);
    ierr = PetscFree(lu->R);CHKERRQ(ierr);
    ierr = PetscFree(lu->C);CHKERRQ(ierr);
    if ( lu->lwork >= 0 ) {
      Destroy_SuperNode_Matrix(&lu->L);
      Destroy_CompCol_Matrix(&lu->U);
    }
  }
  ierr = MatConvert_SuperLU_SeqAIJ(A,MATSEQAIJ,&A);CHKERRQ(ierr);
  ierr = (*A->ops->destroy)(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_SuperLU"
int MatView_SuperLU(Mat A,PetscViewer viewer)
{
  int               ierr;
  PetscTruth        isascii;
  PetscViewerFormat format;
  Mat_SuperLU       *lu=(Mat_SuperLU*)(A->spptr);

  PetscFunctionBegin;
  ierr = (*lu->MatView)(A,viewer);CHKERRQ(ierr);

  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_FACTOR_INFO) {
      ierr = MatFactorInfo_SuperLU(A,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatAssemblyEnd_SuperLU"
int MatAssemblyEnd_SuperLU(Mat A,MatAssemblyType mode) {
  int         ierr;
  Mat_SuperLU *lu=(Mat_SuperLU*)(A->spptr);

  PetscFunctionBegin;
  ierr = (*lu->MatAssemblyEnd)(A,mode);CHKERRQ(ierr);

  lu->MatLUFactorSymbolic  = A->ops->lufactorsymbolic;
  A->ops->lufactorsymbolic = MatLUFactorSymbolic_SuperLU;
  PetscFunctionReturn(0);
}

/* This function was written for SuperLU 2.0 by Matthew Knepley. Not tested for SuperLU 3.0! */
#ifdef SuperLU2
#include "src/mat/impls/dense/seq/dense.h"
#undef __FUNCT__  
#define __FUNCT__ "MatCreateNull_SuperLU"
int MatCreateNull_SuperLU(Mat A,Mat *nullMat)
{
  Mat_SuperLU   *lu = (Mat_SuperLU*)A->spptr;
  int           numRows = A->m,numCols = A->n;
  SCformat      *Lstore;
  int           numNullCols,size;
  SuperLUStat_t stat;
#if defined(PETSC_USE_COMPLEX)
  doublecomplex *nullVals,*workVals;
#else
  PetscScalar   *nullVals,*workVals;
#endif
  int           row,newRow,col,newCol,block,b,ierr;

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
    sp_ztrsv("L","T","U",&lu->L,&lu->U,&nullVals[b*numRows],&stat,&ierr);
#else
    sp_dtrsv("L","T","U",&lu->L,&lu->U,&nullVals[b*numRows],&stat,&ierr);
#endif
    if (ierr < 0)
      SETERRQ1(PETSC_ERR_ARG_WRONG,"The argument %d was invalid",-ierr);
  }
  ierr = PetscFree(workVals);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*nullMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*nullMat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SuperLU"
int MatSolve_SuperLU(Mat A,Vec b,Vec x)
{
  Mat_SuperLU   *lu = (Mat_SuperLU*)A->spptr;
  PetscScalar   *barray,*xarray;
  int           m,n,ierr,lwork,info,i;
  SuperLUStat_t stat;
  double        ferr,berr,*rhsb,*rhsx;

  PetscFunctionBegin;
  /* rhs vector */
  lu->B.ncol = 1;   /* Set the number of right-hand side */
  ierr = VecGetArray(b,&barray);CHKERRQ(ierr);
  ((DNformat*)lu->B.Store)->nzval = barray;

  /* solution vector */
  ierr = VecGetArray(x,&xarray);CHKERRQ(ierr);
  ((DNformat*)lu->X.Store)->nzval = xarray;

  /* Initialize the statistics variables. */
  StatInit(&stat);

  lu->options.Fact  = FACTORED; /* Indicate the factored form of A is supplied. */
  lu->options.Trans = TRANS;
  dgssvx(&lu->options, &lu->A, lu->perm_c, lu->perm_r, lu->etree, lu->equed, lu->R, lu->C,
           &lu->L, &lu->U, lu->work, lu->lwork, &lu->B, &lu->X, &lu->rpg, &lu->rcond, &ferr, &berr,
           &lu->mem_usage, &stat, &info);
   
  ierr = VecRestoreArray(b,&barray);CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&xarray);CHKERRQ(ierr);

  if ( info == 0 || info == lu->A.ncol+1 ) {
    if ( lu->options.IterRefine ) {
      printf("Iterative Refinement:\n");
      printf("%8s%8s%16s%16s\n", "rhs", "Steps", "FERR", "BERR");
      for (i = 0; i < 1; ++i)
        printf("%8d%8d%16e%16e\n", i+1, stat.RefineSteps, ferr, berr);
    }
    fflush(stdout);
  } else if ( info > 0 && lu->lwork == -1 ) {
    printf("** Estimated memory: %d bytes\n", info - n);
  }

  if ( lu->options.PrintStat ) StatPrint(&stat);
  StatFree(&stat);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_SuperLU"
int MatLUFactorNumeric_SuperLU(Mat A,Mat *F)
{
  Mat_SeqAIJ    *aa = (Mat_SeqAIJ*)(A)->data;
  Mat_SuperLU   *lu = (Mat_SuperLU*)(*F)->spptr;
  int           ierr,info;
  PetscTruth    flag;
  SuperLUStat_t stat;
  double        ferr, berr; 
  
  PetscFunctionBegin;
  if (lu->flg == SAME_NONZERO_PATTERN){ /* successing numerical factorization */
    lu->options.Fact = SamePattern;
    /* Ref: ~SuperLU_3.0/EXAMPLE/dlinsolx2.c */
    Destroy_SuperMatrix_Store(&lu->A); 
    if ( lu->lwork >= 0 ) { 
      Destroy_SuperNode_Matrix(&lu->L);
      Destroy_CompCol_Matrix(&lu->U);
      lu->options.Fact = SamePattern;
    }
  }

  /* Create the SuperMatrix for lu->A=A^T:
       Since SuperLU likes column-oriented matrices,we pass it the transpose,
       and then solve A^T X = B in MatSolve(). */
#if defined(PETSC_USE_COMPLEX)
  zCreate_CompCol_Matrix(&lu->A,A->n,A->m,aa->nz,aa->a,aa->j,aa->i,
                           SLU_NC,SLU_Z,SLU_GE);
#else
  dCreate_CompCol_Matrix(&lu->A,A->n,A->m,aa->nz,aa->a,aa->j,aa->i,
                           SLU_NC,SLU_D,SLU_GE);
#endif
  
  /* Initialize the statistics variables. */
  StatInit(&stat);

  /* Numerical factorization */
  lu->lwork = 0;   /* allocate space internally by system malloc */
  lu->B.ncol = 0;  /* Indicate not to solve the system */
  dgssvx(&lu->options, &lu->A, lu->perm_c, lu->perm_r, lu->etree, lu->equed, lu->R, lu->C,
           &lu->L, &lu->U, lu->work, lu->lwork, &lu->B, &lu->X, &lu->rpg, &lu->rcond, &ferr, &berr,
           &lu->mem_usage, &stat, &info);
  
  if ( info == 0 || info == lu->A.ncol+1 ) {
    if ( lu->options.PivotGrowth ) printf("Recip. pivot growth = %e\n", lu->rpg);
    if ( lu->options.ConditionNumber )
      printf("Recip. condition number = %e\n", lu->rcond);
        /*
          NCformat       *Ustore;
          SCformat       *Lstore;
        Lstore = (SCformat *) lu->L.Store;
        Ustore = (NCformat *) lu->U.Store;
	printf("No of nonzeros in factor L = %d\n", Lstore->nnz);
    	printf("No of nonzeros in factor U = %d\n", Ustore->nnz);
    	printf("No of nonzeros in L+U = %d\n", Lstore->nnz + Ustore->nnz - n);
	printf("L\\U MB %.3f\ttotal MB needed %.3f\texpansions %d\n",
	       mem_usage.for_lu/1e6, mem_usage.total_needed/1e6,
	       mem_usage.expansions);
        */
    fflush(stdout);

  } else if ( info > 0 && lu->lwork == -1 ) {
    printf("** Estimated memory: %d bytes\n", info - lu->A.ncol);
  }

  if ( lu->options.PrintStat ) StatPrint(&stat);
  StatFree(&stat);

  lu->flg = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(0);

#ifdef OLD
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
#endif /* OLD */

}

/*
   Note the r permutation is ignored
*/
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_SuperLU"
int MatLUFactorSymbolic_SuperLU(Mat A,IS r,IS c,MatFactorInfo *info,Mat *F)
{
  Mat          B;
  Mat_SuperLU  *lu;
  int          ierr,m=A->m,n=A->n,indx;  
  PetscTruth   flg;
  char         *colperm[]={"NATURAL","MMD_ATA","MMD_AT_PLUS_A","COLAMD"}; /* MY_PERMC - not supported by petsc interface yet */
  char         *iterrefine[]={"NOREFINE", "SINGLE", "DOUBLE", "EXTRA"};
  char         *rowperm[]={"NOROWPERM", "LargeDiag"}; /* MY_PERMC - not supported by petsc interface yet */

  PetscFunctionBegin;
  
  ierr = MatCreate(A->comm,A->m,A->n,PETSC_DETERMINE,PETSC_DETERMINE,&B);CHKERRQ(ierr);
  ierr = MatSetType(B,MATSUPERLU);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,PETSC_NULL);CHKERRQ(ierr);

  B->ops->lufactornumeric = MatLUFactorNumeric_SuperLU;
  B->ops->solve           = MatSolve_SuperLU;
  B->factor               = FACTOR_LU;
  B->assembled            = PETSC_TRUE;  /* required by -sles_view */
  
  lu = (Mat_SuperLU*)(B->spptr);

  /* Set SuperLU options */
    /* the default values for options argument:
	options.Fact = DOFACT;
        options.Equil = YES;
    	options.ColPerm = COLAMD;
	options.DiagPivotThresh = 1.0;
    	options.Trans = NOTRANS;
    	options.IterRefine = NOREFINE;
    	options.SymmetricMode = NO;
    	options.PivotGrowth = NO;
    	options.ConditionNumber = NO;
    	options.PrintStat = YES;
    */
  set_default_options(&lu->options);
  lu->options.Equil = NO;  /* equilibration causes error in solve */
  lu->options.PrintStat = NO;

  ierr = PetscOptionsBegin(A->comm,A->prefix,"SuperLU Options","Mat");CHKERRQ(ierr);
  /* 
  ierr = PetscOptionsLogical("-mat_superlu_Equil","Equil","None",PETSC_FALSE,&flg,0);CHKERRQ(ierr);
  if (flg) lu->options.Equil = YES; -- do not work!!!
  */
  ierr = PetscOptionsEList("-mat_superlu_ColPerm","ColPerm","None",colperm,4,colperm[3],&indx,&flg);CHKERRQ(ierr);
  if (flg) {lu->options.ColPerm = (colperm_t)indx;}
  ierr = PetscOptionsEList("-mat_superlu_IterRefine","IterRefine","None",iterrefine,4,iterrefine[0],&indx,&flg);CHKERRQ(ierr);
  if (flg) { lu->options.IterRefine = (IterRefine_t)indx;}
  ierr = PetscOptionsLogical("-mat_superlu_SymmetricMode","SymmetricMode","None",PETSC_FALSE,&flg,0);CHKERRQ(ierr);
  if (flg) lu->options.SymmetricMode = YES; 
  ierr = PetscOptionsReal("-mat_superlu_DiagPivotThresh","DiagPivotThresh","None",lu->options.DiagPivotThresh,&lu->options.DiagPivotThresh,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsLogical("-mat_superlu_PivotGrowth","PivotGrowth","None",PETSC_FALSE,&flg,0);CHKERRQ(ierr);
  if (flg) lu->options.PivotGrowth = YES;
  ierr = PetscOptionsLogical("-mat_superlu_ConditionNumber","ConditionNumber","None",PETSC_FALSE,&flg,0);CHKERRQ(ierr);
  if (flg) lu->options.ConditionNumber = YES;
  ierr = PetscOptionsEList("-mat_superlu_RowPerm","RowPerm","None",rowperm,2,rowperm[0],&indx,&flg);CHKERRQ(ierr);
  if (flg) {lu->options.RowPerm = (rowperm_t)indx;}
  ierr = PetscOptionsLogical("-mat_superlu_ReplaceTinyPivot","ReplaceTinyPivot","None",PETSC_FALSE,&flg,0);CHKERRQ(ierr);
  if (flg) lu->options.ReplaceTinyPivot = YES; 
  ierr = PetscOptionsLogical("-mat_superlu_PrintStat","PrintStat","None",PETSC_FALSE,&flg,0);CHKERRQ(ierr);
  if (flg) lu->options.PrintStat = YES; 
  PetscOptionsEnd();

#ifdef SUPERLU2
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatCreateNull","MatCreateNull_SuperLU",
                                    (void(*)(void))MatCreateNull_SuperLU);CHKERRQ(ierr);
#endif

  /* Allocate spaces (notice sizes are for the transpose) */
  ierr = PetscMalloc(m*sizeof(int),&lu->etree);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(int),&lu->perm_r);CHKERRQ(ierr);
  ierr = PetscMalloc(m*sizeof(int),&lu->perm_c);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(int),&lu->R);CHKERRQ(ierr);
  ierr = PetscMalloc(m*sizeof(int),&lu->C);CHKERRQ(ierr);
 
  /* create rhs and solution x without allocate space for .Store */
  dCreate_Dense_Matrix(&lu->B, m, 1, PETSC_NULL, m, SLU_DN, SLU_D, SLU_GE);
  dCreate_Dense_Matrix(&lu->X, m, 1, PETSC_NULL, m, SLU_DN, SLU_D, SLU_GE);

  lu->flg            = DIFFERENT_NONZERO_PATTERN;
  lu->CleanUpSuperLU = PETSC_TRUE;

  *F = B;
  PetscLogObjectMemory(B,(A->m+A->n)*sizeof(int)+sizeof(Mat_SuperLU));
  PetscFunctionReturn(0);
}

/* used by -sles_view */
#undef __FUNCT__  
#define __FUNCT__ "MatFactorInfo_SuperLU"
int MatFactorInfo_SuperLU(Mat A,PetscViewer viewer)
{
  Mat_SuperLU       *lu= (Mat_SuperLU*)A->spptr;
  int               ierr;
  superlu_options_t options;

  PetscFunctionBegin;
  /* check if matrix is superlu_dist type */
  if (A->ops->solve != MatSolve_SuperLU) PetscFunctionReturn(0);

  options = lu->options;
  ierr = PetscViewerASCIIPrintf(viewer,"SuperLU run parameters:\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Equil: %s\n",(options.Equil != NO) ? "YES": "NO");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ColPerm: %d\n",options.ColPerm);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  IterRefine: %d\n",options.IterRefine);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  SymmetricMode: %s\n",(options.SymmetricMode != NO) ? "YES": "NO");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  DiagPivotThresh: %g\n",options.DiagPivotThresh);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  PivotGrowth: %s\n",(options.PivotGrowth != NO) ? "YES": "NO");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ConditionNumber: %s\n",(options.ConditionNumber != NO) ? "YES": "NO");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  RowPerm: %d\n",options.RowPerm);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ReplaceTinyPivot: %s\n",(options.ReplaceTinyPivot != NO) ? "YES": "NO");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  PrintStat: %s\n",(options.PrintStat != NO) ? "YES": "NO");CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_SuperLU"
int MatDuplicate_SuperLU(Mat A, MatDuplicateOption op, Mat *M) {
  int         ierr;
  Mat_SuperLU *lu=(Mat_SuperLU *)A->spptr;

  PetscFunctionBegin;
  ierr = (*lu->MatDuplicate)(A,op,M);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_SuperLU(*M,MATSUPERLU,M);CHKERRQ(ierr);
  ierr = PetscMemcpy((*M)->spptr,lu,sizeof(Mat_SuperLU));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_SuperLU_SeqAIJ"
int MatConvert_SuperLU_SeqAIJ(Mat A,MatType type,Mat *newmat) {
  /* This routine is only called to convert an unfactored PETSc-SuperLU matrix */
  /* to its base PETSc type, so we will ignore 'MatType type'. */
  int                  ierr;
  Mat                  B=*newmat;
  Mat_SuperLU   *lu=(Mat_SuperLU *)A->spptr;

  PetscFunctionBegin;
  if (B != A) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }
  /* Reset the original function pointers */
  B->ops->duplicate        = lu->MatDuplicate;
  B->ops->view             = lu->MatView;
  B->ops->assemblyend      = lu->MatAssemblyEnd;
  B->ops->lufactorsymbolic = lu->MatLUFactorSymbolic;
  B->ops->destroy          = lu->MatDestroy;
  /* lu is only a function pointer stash unless we've factored the matrix, which we haven't! */
  ierr = PetscFree(lu);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSEQAIJ);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatConvert_SeqAIJ_SuperLU"
int MatConvert_SeqAIJ_SuperLU(Mat A,MatType type,Mat *newmat) {
  /* This routine is only called to convert to MATSUPERLU */
  /* from MATSEQAIJ, so we will ignore 'MatType type'. */
  int         ierr;
  Mat         B=*newmat;
  Mat_SuperLU *lu;

  PetscFunctionBegin;
  if (B != A) {
    ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  }

  ierr = PetscNew(Mat_SuperLU,&lu);CHKERRQ(ierr);
  lu->MatDuplicate         = A->ops->duplicate;
  lu->MatView              = A->ops->view;
  lu->MatAssemblyEnd       = A->ops->assemblyend;
  lu->MatLUFactorSymbolic  = A->ops->lufactorsymbolic;
  lu->MatDestroy           = A->ops->destroy;
  lu->CleanUpSuperLU       = PETSC_FALSE;

  B->spptr                 = (void*)lu;
  B->ops->duplicate        = MatDuplicate_SuperLU;
  B->ops->view             = MatView_SuperLU;
  B->ops->assemblyend      = MatAssemblyEnd_SuperLU;
  B->ops->lufactorsymbolic = MatLUFactorSymbolic_SuperLU;
  B->ops->destroy          = MatDestroy_SuperLU;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_seqaij_superlu_C",
                                           "MatConvert_SeqAIJ_SuperLU",MatConvert_SeqAIJ_SuperLU);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatConvert_superlu_seqaij_C",
                                           "MatConvert_SuperLU_SeqAIJ",MatConvert_SuperLU_SeqAIJ);CHKERRQ(ierr);
  PetscLogInfo(0,"Using SuperLU for SeqAIJ LU factorization and solves.");
  ierr = PetscObjectChangeTypeName((PetscObject)B,MATSUPERLU);CHKERRQ(ierr);
  *newmat = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*MC
  MATSUPERLU - MATSUPERLU = "superlu" - A matrix type providing direct solvers (LU) for sequential matrices 
  via the external package SuperLU.

  If SuperLU is installed (see the manual for
  instructions on how to declare the existence of external packages),
  a matrix type can be constructed which invokes SuperLU solvers.
  After calling MatCreate(...,A), simply call MatSetType(A,MATSUPERLU).
  This matrix type is only supported for double precision real.

  This matrix inherits from MATSEQAIJ.  As a result, MatSeqAIJSetPreallocation is 
  supported for this matrix type.  One can also call MatConvert for an inplace conversion to or from 
  the MATSEQAIJ type without data copy.

  Options Database Keys:
+ -mat_type superlu - sets the matrix type to "superlu" during a call to MatSetFromOptions()
- -mat_superlu_ordering <0,1,2,3> - 0: natural ordering, 
                                    1: MMD applied to A'*A, 
                                    2: MMD applied to A'+A, 
                                    3: COLAMD, approximate minimum degree column ordering

   Level: beginner

.seealso: PCLU
M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "MatCreate_SuperLU"
int MatCreate_SuperLU(Mat A) {
  int ierr;

  PetscFunctionBegin;
  /* Change type name before calling MatSetType to force proper construction of SeqAIJ and SUPERLU types */
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATSUPERLU);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatConvert_SeqAIJ_SuperLU(A,MATSUPERLU,&A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
