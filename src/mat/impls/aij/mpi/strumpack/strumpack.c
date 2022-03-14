#include <../src/mat/impls/aij/seq/aij.h>            /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <StrumpackSparseSolver.h>

static PetscErrorCode MatGetDiagonal_STRUMPACK(Mat A,Vec v)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Mat type: STRUMPACK factor");
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_STRUMPACK(Mat A)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)A->spptr;
  PetscErrorCode         ierr;
  PetscBool              flg;

  PetscFunctionBegin;
  /* Deallocate STRUMPACK storage */
  PetscStackCall("STRUMPACK_destroy",STRUMPACK_destroy(S));
  ierr = PetscFree(A->spptr);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatDestroy_SeqAIJ(A);CHKERRQ(ierr);
  } else {
    ierr = MatDestroy_MPIAIJ(A);CHKERRQ(ierr);
  }

  /* clear composed functions */
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverType_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSTRUMPACKSetReordering_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSTRUMPACKSetColPerm_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSTRUMPACKSetHSSRelTol_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSTRUMPACKSetHSSAbsTol_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSTRUMPACKSetHSSMaxRank_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSTRUMPACKSetHSSLeafSize_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSTRUMPACKSetHSSMinSepSize_C",NULL);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode MatSTRUMPACKSetReordering_STRUMPACK(Mat F,MatSTRUMPACKReordering reordering)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)F->spptr;

  PetscFunctionBegin;
  PetscStackCall("STRUMPACK_reordering_method",STRUMPACK_set_reordering_method(*S,(STRUMPACK_REORDERING_STRATEGY)reordering));
  PetscFunctionReturn(0);
}

/*@
  MatSTRUMPACKSetReordering - Set STRUMPACK fill-reducing reordering

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-STRUMPACK interface
-  reordering - the code to be used to find the fill-reducing reordering
      Possible values: NATURAL=0 METIS=1 PARMETIS=2 SCOTCH=3 PTSCOTCH=4 RCM=5

  Options Database:
.   -mat_strumpack_reordering <METIS>  - Sparsity reducing matrix reordering (choose one of) NATURAL METIS PARMETIS SCOTCH PTSCOTCH RCM (None)

   Level: beginner

   References:
.  * - STRUMPACK manual

.seealso: MatGetFactor()
@*/
PetscErrorCode MatSTRUMPACKSetReordering(Mat F,MatSTRUMPACKReordering reordering)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  PetscValidLogicalCollectiveEnum(F,reordering,2);
  ierr = PetscTryMethod(F,"MatSTRUMPACKSetReordering_C",(Mat,MatSTRUMPACKReordering),(F,reordering));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSTRUMPACKSetColPerm_STRUMPACK(Mat F,PetscBool cperm)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)F->spptr;

  PetscFunctionBegin;
  PetscStackCall("STRUMPACK_set_mc64job",STRUMPACK_set_mc64job(*S,cperm ? 5 : 0));
  PetscFunctionReturn(0);
}

/*@
  MatSTRUMPACKSetColPerm - Set whether STRUMPACK should try to permute the columns of the matrix in order to get a nonzero diagonal

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-STRUMPACK interface
-  cperm - PETSC_TRUE to permute (internally) the columns of the matrix

  Options Database:
.   -mat_strumpack_colperm <cperm> - true to use the permutation

   Level: beginner

   References:
.  * - STRUMPACK manual

.seealso: MatGetFactor()
@*/
PetscErrorCode MatSTRUMPACKSetColPerm(Mat F,PetscBool cperm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  PetscValidLogicalCollectiveBool(F,cperm,2);
  ierr = PetscTryMethod(F,"MatSTRUMPACKSetColPerm_C",(Mat,PetscBool),(F,cperm));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSTRUMPACKSetHSSRelTol_STRUMPACK(Mat F,PetscReal rtol)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)F->spptr;

  PetscFunctionBegin;
  PetscStackCall("STRUMPACK_set_HSS_rel_tol", STRUMPACK_set_HSS_rel_tol(*S,rtol));
  PetscFunctionReturn(0);
}

/*@
  MatSTRUMPACKSetHSSRelTol - Set STRUMPACK relative tolerance for HSS compression

  Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-STRUMPACK interface
-  rtol - relative compression tolerance

  Options Database:
.   -mat_strumpack_hss_rel_tol <1e-2>         - Relative compression tolerance (None)

   Level: beginner

   References:
.  * - STRUMPACK manual

.seealso: MatGetFactor()
@*/
PetscErrorCode MatSTRUMPACKSetHSSRelTol(Mat F,PetscReal rtol)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  PetscValidLogicalCollectiveReal(F,rtol,2);
  ierr = PetscTryMethod(F,"MatSTRUMPACKSetHSSRelTol_C",(Mat,PetscReal),(F,rtol));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSTRUMPACKSetHSSAbsTol_STRUMPACK(Mat F,PetscReal atol)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)F->spptr;

  PetscFunctionBegin;
  PetscStackCall("STRUMPACK_set_HSS_abs_tol", STRUMPACK_set_HSS_abs_tol(*S,atol));
  PetscFunctionReturn(0);
}

/*@
  MatSTRUMPACKSetHSSAbsTol - Set STRUMPACK absolute tolerance for HSS compression

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-STRUMPACK interface
-  atol - absolute compression tolerance

  Options Database:
.   -mat_strumpack_hss_abs_tol <1e-8>         - Absolute compression tolerance (None)

   Level: beginner

   References:
.  * - STRUMPACK manual

.seealso: MatGetFactor()
@*/
PetscErrorCode MatSTRUMPACKSetHSSAbsTol(Mat F,PetscReal atol)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  PetscValidLogicalCollectiveReal(F,atol,2);
  ierr = PetscTryMethod(F,"MatSTRUMPACKSetHSSAbsTol_C",(Mat,PetscReal),(F,atol));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSTRUMPACKSetHSSMaxRank_STRUMPACK(Mat F,PetscInt hssmaxrank)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)F->spptr;

  PetscFunctionBegin;
  PetscStackCall("STRUMPACK_set_HSS_max_rank", STRUMPACK_set_HSS_max_rank(*S,hssmaxrank));
  PetscFunctionReturn(0);
}

/*@
  MatSTRUMPACKSetHSSMaxRank - Set STRUMPACK maximum HSS rank

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-STRUMPACK interface
-  hssmaxrank - maximum rank used in low-rank approximation

  Options Database:
.   -mat_strumpack_max_rank    - Maximum rank in HSS compression, when using pctype ilu (None)

   Level: beginner

   References:
.  * - STRUMPACK manual

.seealso: MatGetFactor()
@*/
PetscErrorCode MatSTRUMPACKSetHSSMaxRank(Mat F,PetscInt hssmaxrank)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(F,hssmaxrank,2);
  ierr = PetscTryMethod(F,"MatSTRUMPACKSetHSSMaxRank_C",(Mat,PetscInt),(F,hssmaxrank));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSTRUMPACKSetHSSLeafSize_STRUMPACK(Mat F,PetscInt leaf_size)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)F->spptr;

  PetscFunctionBegin;
  PetscStackCall("STRUMPACK_set_HSS_leaf_size", STRUMPACK_set_HSS_leaf_size(*S,leaf_size));
  PetscFunctionReturn(0);
}

/*@
  MatSTRUMPACKSetHSSLeafSize - Set STRUMPACK HSS leaf size

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-STRUMPACK interface
-  leaf_size - Size of diagonal blocks in HSS approximation

  Options Database:
.   -mat_strumpack_leaf_size    - Size of diagonal blocks in HSS approximation, when using pctype ilu (None)

   Level: beginner

   References:
.  * - STRUMPACK manual

.seealso: MatGetFactor()
@*/
PetscErrorCode MatSTRUMPACKSetHSSLeafSize(Mat F,PetscInt leaf_size)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(F,leaf_size,2);
  ierr = PetscTryMethod(F,"MatSTRUMPACKSetHSSLeafSize_C",(Mat,PetscInt),(F,leaf_size));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSTRUMPACKSetHSSMinSepSize_STRUMPACK(Mat F,PetscInt hssminsize)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)F->spptr;

  PetscFunctionBegin;
  PetscStackCall("STRUMPACK_set_HSS_min_sep_size", STRUMPACK_set_HSS_min_sep_size(*S,hssminsize));
  PetscFunctionReturn(0);
}

/*@
  MatSTRUMPACKSetHSSMinSepSize - Set STRUMPACK minimum separator size for low-rank approximation

   Logically Collective on Mat

   Input Parameters:
+  F - the factored matrix obtained by calling MatGetFactor() from PETSc-STRUMPACK interface
-  hssminsize - minimum dense matrix size for low-rank approximation

  Options Database:
.   -mat_strumpack_hss_min_sep_size <hssminsize> - set the minimum separator size

   Level: beginner

   References:
.  * - STRUMPACK manual

.seealso: MatGetFactor()
@*/
PetscErrorCode MatSTRUMPACKSetHSSMinSepSize(Mat F,PetscInt hssminsize)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(F,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(F,hssminsize,2);
  ierr = PetscTryMethod(F,"MatSTRUMPACKSetHSSMinSepSize_C",(Mat,PetscInt),(F,hssminsize));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSolve_STRUMPACK(Mat A,Vec b_mpi,Vec x)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)A->spptr;
  STRUMPACK_RETURN_CODE  sp_err;
  PetscErrorCode         ierr;
  const PetscScalar      *bptr;
  PetscScalar            *xptr;

  PetscFunctionBegin;
  ierr = VecGetArray(x,&xptr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(b_mpi,&bptr);CHKERRQ(ierr);

  PetscStackCall("STRUMPACK_solve",sp_err = STRUMPACK_solve(*S,(PetscScalar*)bptr,xptr,0));
  switch (sp_err) {
  case STRUMPACK_SUCCESS: break;
  case STRUMPACK_MATRIX_NOT_SET:   { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: matrix was not set"); break; }
  case STRUMPACK_REORDERING_ERROR: { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: matrix reordering failed"); break; }
  default:                           SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: solve failed");
  }
  ierr = VecRestoreArray(x,&xptr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(b_mpi,&bptr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMatSolve_STRUMPACK(Mat A,Mat B_mpi,Mat X)
{
  PetscErrorCode   ierr;
  PetscBool        flg;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompareAny((PetscObject)B_mpi,&flg,MATSEQDENSE,MATMPIDENSE,NULL);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix B must be MATDENSE matrix");
  ierr = PetscObjectTypeCompareAny((PetscObject)X,&flg,MATSEQDENSE,MATMPIDENSE,NULL);CHKERRQ(ierr);
  PetscCheckFalse(!flg,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONG,"Matrix X must be MATDENSE matrix");
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatMatSolve_STRUMPACK() is not implemented yet");
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_Info_STRUMPACK(Mat A,PetscViewer viewer)
{
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* check if matrix is strumpack type */
  if (A->ops->solve != MatSolve_STRUMPACK) PetscFunctionReturn(0);
  ierr = PetscViewerASCIIPrintf(viewer,"STRUMPACK sparse solver!\n");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_STRUMPACK(Mat A,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerGetFormat(viewer,&format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO) {
      ierr = MatView_Info_STRUMPACK(A,viewer);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorNumeric_STRUMPACK(Mat F,Mat A,const MatFactorInfo *info)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver*)F->spptr;
  STRUMPACK_RETURN_CODE  sp_err;
  Mat_SeqAIJ             *A_d,*A_o;
  Mat_MPIAIJ             *mat;
  PetscErrorCode         ierr;
  PetscInt               M=A->rmap->N,m=A->rmap->n;
  PetscBool              flg;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATMPIAIJ,&flg);CHKERRQ(ierr);
  if (flg) { /* A is MATMPIAIJ */
    mat = (Mat_MPIAIJ*)A->data;
    A_d = (Mat_SeqAIJ*)(mat->A)->data;
    A_o = (Mat_SeqAIJ*)(mat->B)->data;
    PetscStackCall("STRUMPACK_set_MPIAIJ_matrix",STRUMPACK_set_MPIAIJ_matrix(*S,&m,A_d->i,A_d->j,A_d->a,A_o->i,A_o->j,A_o->a,mat->garray));
  } else { /* A is MATSEQAIJ */
    A_d = (Mat_SeqAIJ*)A->data;
    PetscStackCall("STRUMPACK_set_csr_matrix",STRUMPACK_set_csr_matrix(*S,&M,A_d->i,A_d->j,A_d->a,0));
  }

  /* Reorder and Factor the matrix. */
  /* TODO figure out how to avoid reorder if the matrix values changed, but the pattern remains the same. */
  PetscStackCall("STRUMPACK_reorder",sp_err = STRUMPACK_reorder(*S));
  PetscStackCall("STRUMPACK_factor",sp_err = STRUMPACK_factor(*S));
  switch (sp_err) {
  case STRUMPACK_SUCCESS: break;
  case STRUMPACK_MATRIX_NOT_SET:   { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: matrix was not set"); break; }
  case STRUMPACK_REORDERING_ERROR: { SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: matrix reordering failed"); break; }
  default:                           SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"STRUMPACK error: factorization failed");
  }
  F->assembled    = PETSC_TRUE;
  F->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatLUFactorSymbolic_STRUMPACK(Mat F,Mat A,IS r,IS c,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  F->ops->lufactornumeric = MatLUFactorNumeric_STRUMPACK;
  F->ops->solve           = MatSolve_STRUMPACK;
  F->ops->matsolve        = MatMatSolve_STRUMPACK;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatFactorGetSolverType_aij_strumpack(Mat A,MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERSTRUMPACK;
  PetscFunctionReturn(0);
}

/*MC
  MATSOLVERSSTRUMPACK = "strumpack" - A solver package providing a direct sparse solver (PCLU)
  and a preconditioner (PCILU) using low-rank compression via the external package STRUMPACK.

  Consult the STRUMPACK-sparse manual for more info.

  Use
     ./configure --download-strumpack
  to have PETSc installed with STRUMPACK

  Use
    -pc_type lu -pc_factor_mat_solver_type strumpack
  to use this as an exact (direct) solver, use
    -pc_type ilu -pc_factor_mat_solver_type strumpack
  to enable low-rank compression (i.e, use as a preconditioner).

  Works with AIJ matrices

  Options Database Keys:
+ -mat_strumpack_verbose
. -mat_strumpack_hss_rel_tol <1e-2>         - Relative compression tolerance (None)
. -mat_strumpack_hss_abs_tol <1e-8>         - Absolute compression tolerance (None)
. -mat_strumpack_colperm <TRUE>             - Permute matrix to make diagonal nonzeros (None)
. -mat_strumpack_hss_min_sep_size <256>     - Minimum size of separator for HSS compression (None)
. -mat_strumpack_max_rank                   - Maximum rank in HSS compression, when using pctype ilu (None)
. -mat_strumpack_leaf_size                  - Size of diagonal blocks in HSS approximation, when using pctype ilu (None)
. -mat_strumpack_reordering <METIS>         - Sparsity reducing matrix reordering (choose one of) NATURAL METIS PARMETIS SCOTCH PTSCOTCH RCM (None)
- -mat_strumpack_iterative_solver <DIRECT>  - Select iterative solver from STRUMPACK (choose one of) AUTO DIRECT REFINE PREC_GMRES GMRES PREC_BICGSTAB BICGSTAB (None)

 Level: beginner

.seealso: PCLU, PCILU, MATSOLVERSUPERLU_DIST, MATSOLVERMUMPS, PCFactorSetMatSolverType(), MatSolverType
M*/
static PetscErrorCode MatGetFactor_aij_strumpack(Mat A,MatFactorType ftype,Mat *F)
{
  Mat                           B;
  PetscErrorCode                ierr;
  PetscInt                      M=A->rmap->N,N=A->cmap->N;
  PetscBool                     verb,flg,set;
  PetscReal                     ctol;
  PetscInt                      hssminsize,max_rank,leaf_size;
  STRUMPACK_SparseSolver        *S;
  STRUMPACK_INTERFACE           iface;
  STRUMPACK_REORDERING_STRATEGY ndcurrent,ndvalue;
  STRUMPACK_KRYLOV_SOLVER       itcurrent,itsolver;
  const STRUMPACK_PRECISION     table[2][2][2] =
    {{{STRUMPACK_FLOATCOMPLEX_64, STRUMPACK_DOUBLECOMPLEX_64},
      {STRUMPACK_FLOAT_64,        STRUMPACK_DOUBLE_64}},
     {{STRUMPACK_FLOATCOMPLEX,    STRUMPACK_DOUBLECOMPLEX},
      {STRUMPACK_FLOAT,           STRUMPACK_DOUBLE}}};
  const STRUMPACK_PRECISION     prec = table[(sizeof(PetscInt)==8)?0:1][(PETSC_SCALAR==PETSC_COMPLEX)?0:1][(PETSC_REAL==PETSC_FLOAT)?0:1];
  const char *const             STRUMPACKNDTypes[] = {"NATURAL","METIS","PARMETIS","SCOTCH","PTSCOTCH","RCM","STRUMPACKNDTypes","",0};
  const char *const             SolverTypes[] = {"AUTO","NONE","REFINE","PREC_GMRES","GMRES","PREC_BICGSTAB","BICGSTAB","SolverTypes","",0};

  PetscFunctionBegin;
  /* Create the factorization matrix */
  ierr = MatCreate(PetscObjectComm((PetscObject)A),&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,A->rmap->n,A->cmap->n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(B,((PetscObject)A)->type_name);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(B,0,NULL);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(B,0,NULL,0,NULL);CHKERRQ(ierr);
  B->trivialsymbolic = PETSC_TRUE;
  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU) {
    B->ops->lufactorsymbolic  = MatLUFactorSymbolic_STRUMPACK;
    B->ops->ilufactorsymbolic = MatLUFactorSymbolic_STRUMPACK;
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported");
  B->ops->view        = MatView_STRUMPACK;
  B->ops->destroy     = MatDestroy_STRUMPACK;
  B->ops->getdiagonal = MatGetDiagonal_STRUMPACK;
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatFactorGetSolverType_C",MatFactorGetSolverType_aij_strumpack);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSTRUMPACKSetReordering_C",MatSTRUMPACKSetReordering_STRUMPACK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSTRUMPACKSetColPerm_C",MatSTRUMPACKSetColPerm_STRUMPACK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSTRUMPACKSetHSSRelTol_C",MatSTRUMPACKSetHSSRelTol_STRUMPACK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSTRUMPACKSetHSSAbsTol_C",MatSTRUMPACKSetHSSAbsTol_STRUMPACK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSTRUMPACKSetHSSMaxRank_C",MatSTRUMPACKSetHSSMaxRank_STRUMPACK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSTRUMPACKSetHSSLeafSize_C",MatSTRUMPACKSetHSSLeafSize_STRUMPACK);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)B,"MatSTRUMPACKSetHSSMinSepSize_C",MatSTRUMPACKSetHSSMinSepSize_STRUMPACK);CHKERRQ(ierr);
  B->factortype = ftype;
  ierr     = PetscNewLog(B,&S);CHKERRQ(ierr);
  B->spptr = S;

  ierr = PetscObjectTypeCompare((PetscObject)A,MATSEQAIJ,&flg);CHKERRQ(ierr);
  iface = flg ? STRUMPACK_MT : STRUMPACK_MPI_DIST;

  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject)A),((PetscObject)A)->prefix,"STRUMPACK Options","Mat");CHKERRQ(ierr);

  verb = PetscLogPrintInfo ? PETSC_TRUE : PETSC_FALSE;
  ierr = PetscOptionsBool("-mat_strumpack_verbose","Print STRUMPACK information","None",verb,&verb,NULL);CHKERRQ(ierr);

  PetscStackCall("STRUMPACK_init",STRUMPACK_init(S,PetscObjectComm((PetscObject)A),prec,iface,0,NULL,verb));

  PetscStackCall("STRUMPACK_HSS_rel_tol",ctol = (PetscReal)STRUMPACK_HSS_rel_tol(*S));
  ierr = PetscOptionsReal("-mat_strumpack_hss_rel_tol","Relative compression tolerance","None",ctol,&ctol,&set);CHKERRQ(ierr);
  if (set) PetscStackCall("STRUMPACK_set_HSS_rel_tol",STRUMPACK_set_HSS_rel_tol(*S,(double)ctol));

  PetscStackCall("STRUMPACK_HSS_abs_tol",ctol = (PetscReal)STRUMPACK_HSS_abs_tol(*S));
  ierr = PetscOptionsReal("-mat_strumpack_hss_abs_tol","Absolute compression tolerance","None",ctol,&ctol,&set);CHKERRQ(ierr);
  if (set) PetscStackCall("STRUMPACK_set_HSS_abs_tol",STRUMPACK_set_HSS_abs_tol(*S,(double)ctol));

  PetscStackCall("STRUMPACK_mc64job",flg = (STRUMPACK_mc64job(*S) == 0) ? PETSC_FALSE : PETSC_TRUE);
  ierr = PetscOptionsBool("-mat_strumpack_colperm","Find a col perm to get nonzero diagonal","None",flg,&flg,&set);CHKERRQ(ierr);
  if (set) PetscStackCall("STRUMPACK_set_mc64job",STRUMPACK_set_mc64job(*S,flg ? 5 : 0));

  PetscStackCall("STRUMPACK_HSS_min_sep_size",hssminsize = (PetscInt)STRUMPACK_HSS_min_sep_size(*S));
  ierr = PetscOptionsInt("-mat_strumpack_hss_min_sep_size","Minimum size of separator for HSS compression","None",hssminsize,&hssminsize,&set);CHKERRQ(ierr);
  if (set) PetscStackCall("STRUMPACK_set_HSS_min_sep_size",STRUMPACK_set_HSS_min_sep_size(*S,(int)hssminsize));

  PetscStackCall("STRUMPACK_HSS_max_rank",max_rank = (PetscInt)STRUMPACK_HSS_max_rank(*S));
  ierr = PetscOptionsInt("-mat_strumpack_max_rank","Maximum rank in HSS approximation","None",max_rank,&max_rank,&set);CHKERRQ(ierr);
  if (set) PetscStackCall("STRUMPACK_set_HSS_max_rank",STRUMPACK_set_HSS_max_rank(*S,(int)max_rank));

  PetscStackCall("STRUMPACK_HSS_leaf_size",leaf_size = (PetscInt)STRUMPACK_HSS_leaf_size(*S));
  ierr = PetscOptionsInt("-mat_strumpack_leaf_size","Size of diagonal blocks in HSS approximation","None",leaf_size,&leaf_size,&set);CHKERRQ(ierr);
  if (set) PetscStackCall("STRUMPACK_set_HSS_leaf_size",STRUMPACK_set_HSS_leaf_size(*S,(int)leaf_size));

  PetscStackCall("STRUMPACK_reordering_method",ndcurrent = STRUMPACK_reordering_method(*S));
  ierr = PetscOptionsEnum("-mat_strumpack_reordering","Sparsity reducing matrix reordering","None",STRUMPACKNDTypes,(PetscEnum)ndcurrent,(PetscEnum*)&ndvalue,&set);CHKERRQ(ierr);
  if (set) PetscStackCall("STRUMPACK_set_reordering_method",STRUMPACK_set_reordering_method(*S,ndvalue));

  if (ftype == MAT_FACTOR_ILU) {
    /* When enabling HSS compression, the STRUMPACK solver becomes an incomplete              */
    /* (or approximate) LU factorization.                                                     */
    PetscStackCall("STRUMPACK_enable_HSS",STRUMPACK_enable_HSS(*S));
  }

  /* Disable the outer iterative solver from STRUMPACK.                                       */
  /* When STRUMPACK is used as a direct solver, it will by default do iterative refinement.   */
  /* When STRUMPACK is used as an approximate factorization preconditioner (by enabling       */
  /* low-rank compression), it will use it's own preconditioned GMRES. Here we can disable    */
  /* the outer iterative solver, as PETSc uses STRUMPACK from within a KSP.                   */
  PetscStackCall("STRUMPACK_set_Krylov_solver", STRUMPACK_set_Krylov_solver(*S, STRUMPACK_DIRECT));

  PetscStackCall("STRUMPACK_Krylov_solver",itcurrent = STRUMPACK_Krylov_solver(*S));
  ierr = PetscOptionsEnum("-mat_strumpack_iterative_solver","Select iterative solver from STRUMPACK","None",SolverTypes,(PetscEnum)itcurrent,(PetscEnum*)&itsolver,&set);CHKERRQ(ierr);
  if (set) PetscStackCall("STRUMPACK_set_Krylov_solver",STRUMPACK_set_Krylov_solver(*S,itsolver));

  PetscOptionsEnd();

  *F = B;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_STRUMPACK(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatSolverTypeRegister(MATSOLVERSTRUMPACK,MATMPIAIJ,MAT_FACTOR_LU,MatGetFactor_aij_strumpack);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERSTRUMPACK,MATSEQAIJ,MAT_FACTOR_LU,MatGetFactor_aij_strumpack);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERSTRUMPACK,MATMPIAIJ,MAT_FACTOR_ILU,MatGetFactor_aij_strumpack);CHKERRQ(ierr);
  ierr = MatSolverTypeRegister(MATSOLVERSTRUMPACK,MATSEQAIJ,MAT_FACTOR_ILU,MatGetFactor_aij_strumpack);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
