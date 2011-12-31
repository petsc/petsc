
/*  -------------------------------------------------------------------- 

     This file implements a subclass of the SeqAIJ matrix class that uses
     the SuperLU sparse solver. You can use this as a starting point for 
     implementing your own subclass of a PETSc matrix class.

     This demonstrates a way to make an implementation inheritence of a PETSc
     matrix type. This means constructing a new matrix type (SuperLU) by changing some
     of the methods of the previous type (SeqAIJ), adding additional data, and possibly
     additional method. (See any book on object oriented programming).
*/

/*
     Defines the data structure for the base matrix type (SeqAIJ)
*/
#include <../src/mat/impls/aij/seq/aij.h>    /*I "petscmat.h" I*/

/*
     SuperLU include files
*/
EXTERN_C_BEGIN
#if defined(PETSC_USE_COMPLEX)
#include <slu_zdefs.h>
#else
#include <slu_ddefs.h>
#endif  
#include <slu_util.h>
EXTERN_C_END

/*
     This is the data we are "ADDING" to the SeqAIJ matrix type to get the SuperLU matrix type
*/
typedef struct {
  SuperMatrix       A,L,U,B,X;
  superlu_options_t options;
  PetscInt          *perm_c; /* column permutation vector */
  PetscInt          *perm_r; /* row permutations from partial pivoting */
  PetscInt          *etree;
  PetscReal         *R, *C;
  char              equed[1];
  PetscInt          lwork;
  void              *work;
  PetscReal         rpg, rcond;
  mem_usage_t       mem_usage;
  MatStructure      flg;
  SuperLUStat_t     stat;
  Mat               A_dup;
  PetscScalar       *rhs_dup;

  /* Flag to clean up (non-global) SuperLU objects during Destroy */
  PetscBool  CleanUpSuperLU;
} Mat_SuperLU;

extern PetscErrorCode MatFactorInfo_SuperLU(Mat,PetscViewer);
extern PetscErrorCode MatLUFactorNumeric_SuperLU(Mat,Mat,const MatFactorInfo *);
extern PetscErrorCode MatDestroy_SuperLU(Mat);
extern PetscErrorCode MatView_SuperLU(Mat,PetscViewer);
extern PetscErrorCode MatAssemblyEnd_SuperLU(Mat,MatAssemblyType);
extern PetscErrorCode MatSolve_SuperLU(Mat,Vec,Vec);
extern PetscErrorCode MatMatSolve_SuperLU(Mat,Mat,Mat);
extern PetscErrorCode MatSolveTranspose_SuperLU(Mat,Vec,Vec);
extern PetscErrorCode MatLUFactorSymbolic_SuperLU(Mat,Mat,IS,IS,const MatFactorInfo*);
extern PetscErrorCode MatDuplicate_SuperLU(Mat, MatDuplicateOption, Mat *);

/*
    Utility function
*/
#undef __FUNCT__  
#define __FUNCT__ "MatFactorInfo_SuperLU"
PetscErrorCode MatFactorInfo_SuperLU(Mat A,PetscViewer viewer)
{
  Mat_SuperLU       *lu= (Mat_SuperLU*)A->spptr;
  PetscErrorCode    ierr;
  superlu_options_t options;

  PetscFunctionBegin;
  /* check if matrix is superlu_dist type */
  if (A->ops->solve != MatSolve_SuperLU) PetscFunctionReturn(0);

  options = lu->options;
  ierr = PetscViewerASCIIPrintf(viewer,"SuperLU run parameters:\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  Equil: %s\n",(options.Equil != NO) ? "YES": "NO");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ColPerm: %D\n",options.ColPerm);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  IterRefine: %D\n",options.IterRefine);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  SymmetricMode: %s\n",(options.SymmetricMode != NO) ? "YES": "NO");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  DiagPivotThresh: %g\n",options.DiagPivotThresh);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  PivotGrowth: %s\n",(options.PivotGrowth != NO) ? "YES": "NO");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ConditionNumber: %s\n",(options.ConditionNumber != NO) ? "YES": "NO");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  RowPerm: %D\n",options.RowPerm);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  ReplaceTinyPivot: %s\n",(options.ReplaceTinyPivot != NO) ? "YES": "NO");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  PrintStat: %s\n",(options.PrintStat != NO) ? "YES": "NO");CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer,"  lwork: %D\n",lu->lwork);CHKERRQ(ierr);
  if (A->factortype == MAT_FACTOR_ILU){ 
    ierr = PetscViewerASCIIPrintf(viewer,"  ILU_DropTol: %g\n",options.ILU_DropTol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  ILU_FillTol: %g\n",options.ILU_FillTol);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  ILU_FillFactor: %g\n",options.ILU_FillFactor);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  ILU_DropRule: %D\n",options.ILU_DropRule);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  ILU_Norm: %D\n",options.ILU_Norm);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  ILU_MILU: %D\n",options.ILU_MILU);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
    These are the methods provided to REPLACE the corresponding methods of the 
   SeqAIJ matrix class. Other methods of SeqAIJ are not replaced
*/
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorNumeric_SuperLU"
PetscErrorCode MatLUFactorNumeric_SuperLU(Mat F,Mat A,const MatFactorInfo *info)
{
  Mat_SuperLU    *lu = (Mat_SuperLU*)F->spptr;
  Mat_SeqAIJ     *aa;
  PetscErrorCode ierr;
  PetscInt       sinfo;
  PetscReal      ferr, berr; 
  NCformat       *Ustore;
  SCformat       *Lstore;
  
  PetscFunctionBegin;
  if (lu->flg == SAME_NONZERO_PATTERN){ /* successing numerical factorization */
    lu->options.Fact = SamePattern;
    /* Ref: ~SuperLU_3.0/EXAMPLE/dlinsolx2.c */
    Destroy_SuperMatrix_Store(&lu->A); 
    if (lu->options.Equil){
      ierr = MatCopy_SeqAIJ(A,lu->A_dup,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    }
    if ( lu->lwork >= 0 ) { 
      Destroy_SuperNode_Matrix(&lu->L);
      Destroy_CompCol_Matrix(&lu->U);
      lu->options.Fact = SamePattern;
    }
  }

  /* Create the SuperMatrix for lu->A=A^T:
       Since SuperLU likes column-oriented matrices,we pass it the transpose,
       and then solve A^T X = B in MatSolve(). */
  if (lu->options.Equil){
    aa = (Mat_SeqAIJ*)(lu->A_dup)->data;
  } else {
    aa = (Mat_SeqAIJ*)(A)->data;
  }
#if defined(PETSC_USE_COMPLEX)
  zCreate_CompCol_Matrix(&lu->A,A->cmap->n,A->rmap->n,aa->nz,(doublecomplex*)aa->a,aa->j,aa->i,
                           SLU_NC,SLU_Z,SLU_GE);
#else
  dCreate_CompCol_Matrix(&lu->A,A->cmap->n,A->rmap->n,aa->nz,aa->a,aa->j,aa->i,
                           SLU_NC,SLU_D,SLU_GE);
#endif

  /* Numerical factorization */
  lu->B.ncol = 0;  /* Indicate not to solve the system */
  if (F->factortype == MAT_FACTOR_LU){
#if defined(PETSC_USE_COMPLEX)
    zgssvx(&lu->options, &lu->A, lu->perm_c, lu->perm_r, lu->etree, lu->equed, lu->R, lu->C,
           &lu->L, &lu->U, lu->work, lu->lwork, &lu->B, &lu->X, &lu->rpg, &lu->rcond, &ferr, &berr,
           &lu->mem_usage, &lu->stat, &sinfo);
#else
    dgssvx(&lu->options, &lu->A, lu->perm_c, lu->perm_r, lu->etree, lu->equed, lu->R, lu->C,
           &lu->L, &lu->U, lu->work, lu->lwork, &lu->B, &lu->X, &lu->rpg, &lu->rcond, &ferr, &berr,
           &lu->mem_usage, &lu->stat, &sinfo);
#endif
  } else if (F->factortype == MAT_FACTOR_ILU){
    /* Compute the incomplete factorization, condition number and pivot growth */
#if defined(PETSC_USE_COMPLEX)
    zgsisx(&lu->options, &lu->A, lu->perm_c, lu->perm_r,lu->etree, lu->equed, lu->R, lu->C, 
           &lu->L, &lu->U, lu->work, lu->lwork, &lu->B, &lu->X, &lu->rpg, &lu->rcond,
           &lu->mem_usage, &lu->stat, &sinfo);
#else
    dgsisx(&lu->options, &lu->A, lu->perm_c, lu->perm_r, lu->etree, lu->equed, lu->R, lu->C, 
          &lu->L, &lu->U, lu->work, lu->lwork, &lu->B, &lu->X, &lu->rpg, &lu->rcond, 
          &lu->mem_usage, &lu->stat, &sinfo);
#endif
  } else {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported");
  }
  if ( !sinfo || sinfo == lu->A.ncol+1 ) {
    if ( lu->options.PivotGrowth ) 
      ierr = PetscPrintf(PETSC_COMM_SELF,"  Recip. pivot growth = %e\n", lu->rpg);
    if ( lu->options.ConditionNumber )
      ierr = PetscPrintf(PETSC_COMM_SELF,"  Recip. condition number = %e\n", lu->rcond);
  } else if ( sinfo > 0 ){
    if ( lu->lwork == -1 ) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"  ** Estimated memory: %D bytes\n", sinfo - lu->A.ncol);
    } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_MAT_LU_ZRPVT,"Zero pivot in row %D",sinfo);
  } else { /* sinfo < 0 */
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB, "info = %D, the %D-th argument in gssvx() had an illegal value", sinfo,-sinfo); 
  }

  if ( lu->options.PrintStat ) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"MatLUFactorNumeric_SuperLU():\n");
    StatPrint(&lu->stat);
    Lstore = (SCformat *) lu->L.Store;
    Ustore = (NCformat *) lu->U.Store;
    ierr = PetscPrintf(PETSC_COMM_SELF,"  No of nonzeros in factor L = %D\n", Lstore->nnz);
    ierr = PetscPrintf(PETSC_COMM_SELF,"  No of nonzeros in factor U = %D\n", Ustore->nnz);
    ierr = PetscPrintf(PETSC_COMM_SELF,"  No of nonzeros in L+U = %D\n", Lstore->nnz + Ustore->nnz - lu->A.ncol);
    ierr = PetscPrintf(PETSC_COMM_SELF,"  L\\U MB %.3f\ttotal MB needed %.3f\n",
	       lu->mem_usage.for_lu/1e6, lu->mem_usage.total_needed/1e6);
  }

  lu->flg = SAME_NONZERO_PATTERN;
  F->ops->solve          = MatSolve_SuperLU;
  F->ops->solvetranspose = MatSolveTranspose_SuperLU;
  F->ops->matsolve       = MatMatSolve_SuperLU;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_SuperLU"
PetscErrorCode MatDestroy_SuperLU(Mat A)
{
  PetscErrorCode ierr;
  Mat_SuperLU    *lu=(Mat_SuperLU*)A->spptr;

  PetscFunctionBegin;
  if (lu && lu->CleanUpSuperLU) { /* Free the SuperLU datastructures */
    Destroy_SuperMatrix_Store(&lu->A);
    Destroy_SuperMatrix_Store(&lu->B);
    Destroy_SuperMatrix_Store(&lu->X);
    StatFree(&lu->stat);
    if (lu->lwork >= 0) {
      Destroy_SuperNode_Matrix(&lu->L);
      Destroy_CompCol_Matrix(&lu->U);
    }
  }
  if (lu) {
    ierr = PetscFree(lu->etree);CHKERRQ(ierr);
    ierr = PetscFree(lu->perm_r);CHKERRQ(ierr);
    ierr = PetscFree(lu->perm_c);CHKERRQ(ierr);
    ierr = PetscFree(lu->R);CHKERRQ(ierr);
    ierr = PetscFree(lu->C);CHKERRQ(ierr);
    ierr = PetscFree(lu->rhs_dup);CHKERRQ(ierr);
    ierr = MatDestroy(&lu->A_dup);CHKERRQ(ierr);
  }
  ierr = PetscFree(A->spptr);CHKERRQ(ierr);

  /* clear composed functions */
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatFactorGetSolverPackage_C","",PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)A,"MatSuperluSetILUDropTol_C","",PETSC_NULL);CHKERRQ(ierr);

  ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatView_SuperLU"
PetscErrorCode MatView_SuperLU(Mat A,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      ierr = MatFactorInfo_SuperLU(A,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SuperLU_Private"
PetscErrorCode MatSolve_SuperLU_Private(Mat A,Vec b,Vec x)
{
  Mat_SuperLU    *lu = (Mat_SuperLU*)A->spptr;
  PetscScalar    *barray,*xarray;
  PetscErrorCode ierr;
  PetscInt       info,i,n=x->map->n;
  PetscReal      ferr,berr; 
 
  PetscFunctionBegin;
  if ( lu->lwork == -1 ) {
    PetscFunctionReturn(0);
  }

  lu->B.ncol = 1;   /* Set the number of right-hand side */
  if (lu->options.Equil && !lu->rhs_dup){
    /* superlu overwrites b when Equil is used, thus create rhs_dup to keep user's b unchanged */
    ierr = PetscMalloc(n*sizeof(PetscScalar),&lu->rhs_dup);CHKERRQ(ierr); 
  }
  if (lu->options.Equil){
    /* Copy b into rsh_dup */
    ierr = VecGetArray(b,&barray);CHKERRQ(ierr);
    ierr = PetscMemcpy(lu->rhs_dup,barray,n*sizeof(PetscScalar));CHKERRQ(ierr);
    ierr = VecRestoreArray(b,&barray);CHKERRQ(ierr);
    barray = lu->rhs_dup;
  } else {
    ierr = VecGetArray(b,&barray);CHKERRQ(ierr);
  }
  ierr = VecGetArray(x,&xarray);CHKERRQ(ierr);

#if defined(PETSC_USE_COMPLEX)
  ((DNformat*)lu->B.Store)->nzval = (doublecomplex*)barray;
  ((DNformat*)lu->X.Store)->nzval = (doublecomplex*)xarray;
#else
  ((DNformat*)lu->B.Store)->nzval = barray;
  ((DNformat*)lu->X.Store)->nzval = xarray;
#endif

  lu->options.Fact = FACTORED; /* Indicate the factored form of A is supplied. */
  if (A->factortype == MAT_FACTOR_LU){
#if defined(PETSC_USE_COMPLEX)
    zgssvx(&lu->options, &lu->A, lu->perm_c, lu->perm_r, lu->etree, lu->equed, lu->R, lu->C,
           &lu->L, &lu->U, lu->work, lu->lwork, &lu->B, &lu->X, &lu->rpg, &lu->rcond, &ferr, &berr,
           &lu->mem_usage, &lu->stat, &info);
#else
    dgssvx(&lu->options, &lu->A, lu->perm_c, lu->perm_r, lu->etree, lu->equed, lu->R, lu->C,
           &lu->L, &lu->U, lu->work, lu->lwork, &lu->B, &lu->X, &lu->rpg, &lu->rcond, &ferr, &berr,
           &lu->mem_usage, &lu->stat, &info);
#endif
  } else if (A->factortype == MAT_FACTOR_ILU){ 
#if defined(PETSC_USE_COMPLEX)
    zgsisx(&lu->options, &lu->A, lu->perm_c, lu->perm_r, lu->etree, lu->equed, lu->R, lu->C,
           &lu->L, &lu->U, lu->work, lu->lwork, &lu->B, &lu->X, &lu->rpg, &lu->rcond, 
           &lu->mem_usage, &lu->stat, &info);
#else
    dgsisx(&lu->options, &lu->A, lu->perm_c, lu->perm_r, lu->etree, lu->equed, lu->R, lu->C,
           &lu->L, &lu->U, lu->work, lu->lwork, &lu->B, &lu->X, &lu->rpg, &lu->rcond, 
           &lu->mem_usage, &lu->stat, &info);
#endif
  } else {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported");
  }
  if (!lu->options.Equil){
    ierr = VecRestoreArray(b,&barray);CHKERRQ(ierr);
  }
  ierr = VecRestoreArray(x,&xarray);CHKERRQ(ierr);

  if ( !info || info == lu->A.ncol+1 ) {
    if ( lu->options.IterRefine ) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"Iterative Refinement:\n");
      ierr = PetscPrintf(PETSC_COMM_SELF,"  %8s%8s%16s%16s\n", "rhs", "Steps", "FERR", "BERR");
      for (i = 0; i < 1; ++i)
        ierr = PetscPrintf(PETSC_COMM_SELF,"  %8d%8d%16e%16e\n", i+1, lu->stat.RefineSteps, ferr, berr);
    }
  } else if ( info > 0 ){
    if ( lu->lwork == -1 ) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"  ** Estimated memory: %D bytes\n", info - lu->A.ncol);
    } else {
      ierr = PetscPrintf(PETSC_COMM_SELF,"  Warning: gssvx() returns info %D\n",info);
    }
  } else if (info < 0){
    SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB, "info = %D, the %D-th argument in gssvx() had an illegal value", info,-info);
  }

  if ( lu->options.PrintStat ) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"MatSolve__SuperLU():\n");
    StatPrint(&lu->stat);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolve_SuperLU"
PetscErrorCode MatSolve_SuperLU(Mat A,Vec b,Vec x)
{
  Mat_SuperLU    *lu = (Mat_SuperLU*)A->spptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  lu->options.Trans = TRANS;
  ierr = MatSolve_SuperLU_Private(A,b,x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatSolveTranspose_SuperLU"
PetscErrorCode MatSolveTranspose_SuperLU(Mat A,Vec b,Vec x)
{
  Mat_SuperLU    *lu = (Mat_SuperLU*)A->spptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  lu->options.Trans = NOTRANS;
  ierr = MatSolve_SuperLU_Private(A,b,x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "MatMatSolve_SuperLU"
PetscErrorCode MatMatSolve_SuperLU(Mat A,Mat B,Mat X)
{
  Mat_SuperLU    *lu = (Mat_SuperLU*)A->spptr;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTypeCompareAny((PetscObject)B,&flg,MATSEQDENSE,MATMPIDENSE,PETSC_NULL);CHKERRQ(ierr);
  if (!flg) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONG,"Matrix B must be MATDENSE matrix");
  ierr = PetscTypeCompareAny((PetscObject)X,&flg,MATSEQDENSE,MATMPIDENSE,PETSC_NULL);CHKERRQ(ierr);
  if (!flg) SETERRQ(((PetscObject)A)->comm,PETSC_ERR_ARG_WRONG,"Matrix X must be MATDENSE matrix");  lu->options.Trans = TRANS;
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatMatSolve_SuperLU() is not implemented yet");
  PetscFunctionReturn(0);
}

/*
   Note the r permutation is ignored
*/
#undef __FUNCT__  
#define __FUNCT__ "MatLUFactorSymbolic_SuperLU"
PetscErrorCode MatLUFactorSymbolic_SuperLU(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  Mat_SuperLU    *lu = (Mat_SuperLU*)(F->spptr);
 
  PetscFunctionBegin;
  lu->flg                 = DIFFERENT_NONZERO_PATTERN;
  lu->CleanUpSuperLU      = PETSC_TRUE;
  F->ops->lufactornumeric = MatLUFactorNumeric_SuperLU;
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatSuperluSetILUDropTol_SuperLU"
PetscErrorCode MatSuperluSetILUDropTol_SuperLU(Mat F,PetscReal dtol)
{
  Mat_SuperLU *lu= (Mat_SuperLU*)F->spptr;

  PetscFunctionBegin;
  lu->options.ILU_DropTol = dtol;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__   
#define __FUNCT__ "MatSuperluSetILUDropTol"
/*@
  MatSuperluSetILUDropTol - Set SuperLU ILU drop tolerance
   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-SuperLU interface
-  dtol - drop tolerance

  Options Database:
.   -mat_superlu_ilu_droptol <dtol>

   Level: beginner

   References: SuperLU Users' Guide 

.seealso: MatGetFactor()
@*/
PetscErrorCode MatSuperluSetILUDropTol(Mat F,PetscReal dtol)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(F,dtol,2);
  ierr = PetscTryMethod(F,"MatSuperluSetILUDropTol_C",(Mat,PetscReal),(F,dtol));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN 
#undef __FUNCT__  
#define __FUNCT__ "MatFactorGetSolverPackage_seqaij_superlu"
PetscErrorCode MatFactorGetSolverPackage_seqaij_superlu(Mat A,const MatSolverPackage *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERSUPERLU;
  PetscFunctionReturn(0);
}
EXTERN_C_END
  

/*MC
  MATSOLVERSUPERLU = "superlu" - A solver package providing solvers LU and ILU for sequential matrices 
  via the external package SuperLU.

  Use ./configure --download-superlu to have PETSc installed with SuperLU

  Options Database Keys:
+ -mat_superlu_equil <FALSE>            - Equil (None)
. -mat_superlu_colperm <COLAMD>         - (choose one of) NATURAL MMD_ATA MMD_AT_PLUS_A COLAMD
. -mat_superlu_iterrefine <NOREFINE>    - (choose one of) NOREFINE SINGLE DOUBLE EXTRA
. -mat_superlu_symmetricmode: <FALSE>   - SymmetricMode (None)
. -mat_superlu_diagpivotthresh <1>      - DiagPivotThresh (None)
. -mat_superlu_pivotgrowth <FALSE>      - PivotGrowth (None)
. -mat_superlu_conditionnumber <FALSE>  - ConditionNumber (None)
. -mat_superlu_rowperm <NOROWPERM>      - (choose one of) NOROWPERM LargeDiag
. -mat_superlu_replacetinypivot <FALSE> - ReplaceTinyPivot (None)
. -mat_superlu_printstat <FALSE>        - PrintStat (None)
. -mat_superlu_lwork <0>                - size of work array in bytes used by factorization (None)
. -mat_superlu_ilu_droptol <0>          - ILU_DropTol (None)
. -mat_superlu_ilu_filltol <0>          - ILU_FillTol (None)
. -mat_superlu_ilu_fillfactor <0>       - ILU_FillFactor (None)
. -mat_superlu_ilu_droprull <0>         - ILU_DropRule (None)
. -mat_superlu_ilu_norm <0>             - ILU_Norm (None)
- -mat_superlu_ilu_milu <0>             - ILU_MILU (None)

   Notes: Do not confuse this with MATSOLVERSUPERLU_DIST which is for parallel sparse solves

   Level: beginner

.seealso: PCLU, PCILU, MATSOLVERSUPERLU_DIST, MATSOLVERMUMPS, MATSOLVERSPOOLES, PCFactorSetMatSolverPackage(), MatSolverPackage
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "MatGetFactor_seqaij_superlu"
PetscErrorCode MatGetFactor_seqaij_superlu(Mat A,MatFactorType ftype,Mat *F)
{
  Mat            B;
  Mat_SuperLU    *lu;
  PetscErrorCode ierr;
  PetscInt       indx,m=A->rmap->n,n=A->cmap->n;  
  PetscBool      flg;
  const char     *colperm[]={"NATURAL","MMD_ATA","MMD_AT_PLUS_A","COLAMD"}; /* MY_PERMC - not supported by the petsc interface yet */
  const char     *iterrefine[]={"NOREFINE", "SINGLE", "DOUBLE", "EXTRA"};
  const char     *rowperm[]={"NOROWPERM", "LargeDiag"}; /* MY_PERMC - not supported by the petsc interface yet */

  PetscFunctionBegin;
  ierr = MatCreate(((PetscObject)A)->comm,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,PETSC_NULL);CHKERRQ(ierr);

  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU){
    B->ops->lufactorsymbolic  = MatLUFactorSymbolic_SuperLU;
    B->ops->ilufactorsymbolic = MatLUFactorSymbolic_SuperLU; 
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported");

  B->ops->destroy          = MatDestroy_SuperLU;
  B->ops->view             = MatView_SuperLU;
  B->factortype            = ftype; 
  B->assembled             = PETSC_TRUE;  /* required by -ksp_view */
  B->preallocated          = PETSC_TRUE;
  
  ierr = PetscNewLog(B,Mat_SuperLU,&lu);CHKERRQ(ierr);
  
  if (ftype == MAT_FACTOR_LU){
    set_default_options(&lu->options);
    /* Comments from SuperLU_4.0/SRC/dgssvx.c:
      "Whether or not the system will be equilibrated depends on the
       scaling of the matrix A, but if equilibration is used, A is
       overwritten by diag(R)*A*diag(C) and B by diag(R)*B
       (if options->Trans=NOTRANS) or diag(C)*B (if options->Trans = TRANS or CONJ)."
     We set 'options.Equil = NO' as default because additional space is needed for it.
    */
    lu->options.Equil = NO;
  } else if (ftype == MAT_FACTOR_ILU){
    /* Set the default input options of ilu: */
    ilu_set_default_options(&lu->options);
  }
  lu->options.PrintStat = NO;
  
  /* Initialize the statistics variables. */
  StatInit(&lu->stat);
  lu->lwork = 0;   /* allocate space internally by system malloc */

  ierr = PetscOptionsBegin(((PetscObject)A)->comm,((PetscObject)A)->prefix,"SuperLU Options","Mat");CHKERRQ(ierr);
    ierr = PetscOptionsBool("-mat_superlu_equil","Equil","None",(PetscBool)lu->options.Equil,(PetscBool*)&lu->options.Equil,0);CHKERRQ(ierr);
    ierr = PetscOptionsEList("-mat_superlu_colperm","ColPerm","None",colperm,4,colperm[3],&indx,&flg);CHKERRQ(ierr);
    if (flg) {lu->options.ColPerm = (colperm_t)indx;}
    ierr = PetscOptionsEList("-mat_superlu_iterrefine","IterRefine","None",iterrefine,4,iterrefine[0],&indx,&flg);CHKERRQ(ierr);
    if (flg) { lu->options.IterRefine = (IterRefine_t)indx;}
    ierr = PetscOptionsBool("-mat_superlu_symmetricmode","SymmetricMode","None",(PetscBool)lu->options.SymmetricMode,&flg,0);CHKERRQ(ierr);
    if (flg) lu->options.SymmetricMode = YES; 
    ierr = PetscOptionsReal("-mat_superlu_diagpivotthresh","DiagPivotThresh","None",lu->options.DiagPivotThresh,&lu->options.DiagPivotThresh,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-mat_superlu_pivotgrowth","PivotGrowth","None",(PetscBool)lu->options.PivotGrowth,&flg,0);CHKERRQ(ierr);
    if (flg) lu->options.PivotGrowth = YES;
    ierr = PetscOptionsBool("-mat_superlu_conditionnumber","ConditionNumber","None",(PetscBool)lu->options.ConditionNumber,&flg,0);CHKERRQ(ierr);
    if (flg) lu->options.ConditionNumber = YES;
    ierr = PetscOptionsEList("-mat_superlu_rowperm","rowperm","None",rowperm,2,rowperm[lu->options.RowPerm],&indx,&flg);CHKERRQ(ierr);
    if (flg) {lu->options.RowPerm = (rowperm_t)indx;}
    ierr = PetscOptionsBool("-mat_superlu_replacetinypivot","ReplaceTinyPivot","None",(PetscBool)lu->options.ReplaceTinyPivot,&flg,0);CHKERRQ(ierr);
    if (flg) lu->options.ReplaceTinyPivot = YES; 
    ierr = PetscOptionsBool("-mat_superlu_printstat","PrintStat","None",(PetscBool)lu->options.PrintStat,&flg,0);CHKERRQ(ierr);
    if (flg) lu->options.PrintStat = YES; 
    ierr = PetscOptionsInt("-mat_superlu_lwork","size of work array in bytes used by factorization","None",lu->lwork,&lu->lwork,PETSC_NULL);CHKERRQ(ierr); 
    if (lu->lwork > 0 ){
      ierr = PetscMalloc(lu->lwork,&lu->work);CHKERRQ(ierr); 
    } else if (lu->lwork != 0 && lu->lwork != -1){
      ierr = PetscPrintf(PETSC_COMM_SELF,"   Warning: lwork %D is not supported by SUPERLU. The default lwork=0 is used.\n",lu->lwork);
      lu->lwork = 0;
    }
    /* ilu options */
    ierr = PetscOptionsReal("-mat_superlu_ilu_droptol","ILU_DropTol","None",lu->options.ILU_DropTol,&lu->options.ILU_DropTol,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-mat_superlu_ilu_filltol","ILU_FillTol","None",lu->options.ILU_FillTol,&lu->options.ILU_FillTol,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-mat_superlu_ilu_fillfactor","ILU_FillFactor","None",lu->options.ILU_FillFactor,&lu->options.ILU_FillFactor,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_superlu_ilu_droprull","ILU_DropRule","None",lu->options.ILU_DropRule,&lu->options.ILU_DropRule,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-mat_superlu_ilu_norm","ILU_Norm","None",lu->options.ILU_Norm,&indx,&flg);CHKERRQ(ierr);
    if (flg){
      lu->options.ILU_Norm = (norm_t)indx;
    }
    ierr = PetscOptionsInt("-mat_superlu_ilu_milu","ILU_MILU","None",lu->options.ILU_MILU,&indx,&flg);CHKERRQ(ierr);
    if (flg){
      lu->options.ILU_MILU = (milu_t)indx;
    }
  PetscOptionsEnd();
  if (lu->options.Equil == YES) {
    /* superlu overwrites input matrix and rhs when Equil is used, thus create A_dup to keep user's A unchanged */
    ierr = MatDuplicate_SeqAIJ(A,MAT_COPY_VALUES,&lu->A_dup);CHKERRQ(ierr); 
  }

  /* Allocate spaces (notice sizes are for the transpose) */
  ierr = PetscMalloc(m*sizeof(PetscInt),&lu->etree);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscInt),&lu->perm_r);CHKERRQ(ierr);
  ierr = PetscMalloc(m*sizeof(PetscInt),&lu->perm_c);CHKERRQ(ierr);
  ierr = PetscMalloc(n*sizeof(PetscScalar),&lu->R);CHKERRQ(ierr);
  ierr = PetscMalloc(m*sizeof(PetscScalar),&lu->C);CHKERRQ(ierr);
 
  /* create rhs and solution x without allocate space for .Store */
#if defined(PETSC_USE_COMPLEX)
  zCreate_Dense_Matrix(&lu->B, m, 1, PETSC_NULL, m, SLU_DN, SLU_Z, SLU_GE);
  zCreate_Dense_Matrix(&lu->X, m, 1, PETSC_NULL, m, SLU_DN, SLU_Z, SLU_GE);
#else
  dCreate_Dense_Matrix(&lu->B, m, 1, PETSC_NULL, m, SLU_DN, SLU_D, SLU_GE);
  dCreate_Dense_Matrix(&lu->X, m, 1, PETSC_NULL, m, SLU_DN, SLU_D, SLU_GE);
#endif

#ifdef SUPERLU2
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatCreateNull","MatCreateNull_SuperLU",(void(*)(void))MatCreateNull_SuperLU);CHKERRQ(ierr);
#endif
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatFactorGetSolverPackage_C","MatFactorGetSolverPackage_seqaij_superlu",MatFactorGetSolverPackage_seqaij_superlu);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)B,"MatSuperluSetILUDropTol_C","MatSuperluSetILUDropTol_SuperLU",MatSuperluSetILUDropTol_SuperLU);CHKERRQ(ierr);
  B->spptr = lu;
  *F = B;
  PetscFunctionReturn(0);
}
EXTERN_C_END

