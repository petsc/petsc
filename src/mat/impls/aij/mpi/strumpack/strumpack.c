#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <StrumpackSparseSolver.h>

static PetscErrorCode MatGetDiagonal_STRUMPACK(Mat A, Vec v)
{
  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Mat type: STRUMPACK factor");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_STRUMPACK(Mat A)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)A->data;

  PetscFunctionBegin;
  /* Deallocate STRUMPACK storage */
  PetscStackCallExternalVoid("STRUMPACK_destroy", STRUMPACK_destroy(S));
  PetscCall(PetscFree(A->data));

  /* clear composed functions */
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatFactorGetSolverType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKSetReordering_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKSetColPerm_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKSetHSSRelTol_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKSetHSSAbsTol_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKSetHSSMaxRank_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKSetHSSLeafSize_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKSetHSSMinSepSize_C", NULL));

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSTRUMPACKSetReordering_STRUMPACK(Mat F, MatSTRUMPACKReordering reordering)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_reordering_method", STRUMPACK_set_reordering_method(*S, (STRUMPACK_REORDERING_STRATEGY)reordering));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSTRUMPACKSetReordering - Set STRUMPACK fill-reducing reordering

   Input Parameters:
+  F - the factored matrix obtained by calling `MatGetFactor()`
-  reordering - the code to be used to find the fill-reducing reordering, see `MatSTRUMPACKReordering`

  Options Database Key:
.   -mat_strumpack_reordering <METIS>  - Sparsity reducing matrix reordering, see `MatSTRUMPACKReordering`

   Level: beginner

   References:
.  * - STRUMPACK manual

.seealso: [](chapter_matrices), `Mat`, `MatSTRUMPACKReordering`, `MatGetFactor()`, `MatSTRUMPACKSetColPerm()`, `MatSTRUMPACKSetHSSRelTol()`,
          `MatSTRUMPACKSetHSSAbsTol()`, `MatSTRUMPACKSetHSSMaxRank()`, `MatSTRUMPACKSetHSSLeafSize()`, `MatSTRUMPACKSetHSSMinSepSize()`
@*/
PetscErrorCode MatSTRUMPACKSetReordering(Mat F, MatSTRUMPACKReordering reordering)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(F, reordering, 2);
  PetscTryMethod(F, "MatSTRUMPACKSetReordering_C", (Mat, MatSTRUMPACKReordering), (F, reordering));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSTRUMPACKSetColPerm_STRUMPACK(Mat F, PetscBool cperm)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_set_mc64job", STRUMPACK_set_mc64job(*S, cperm ? 5 : 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSTRUMPACKSetColPerm - Set whether STRUMPACK should try to permute the columns of the matrix in order to get a nonzero diagonal

   Logically Collective

   Input Parameters:
+  F - the factored matrix obtained by calling `MatGetFactor()`
-  cperm - `PETSC_TRUE` to permute (internally) the columns of the matrix

  Options Database Key:
.   -mat_strumpack_colperm <cperm> - true to use the permutation

   Level: beginner

   References:
.  * - STRUMPACK manual

.seealso: [](chapter_matrices), `MatSTRUMPACKSetReordering()`, `Mat`, `MatGetFactor()`, `MatSTRUMPACKSetHSSRelTol()`, `MatSTRUMPACKSetHSSAbsTol()`,
          `MatSTRUMPACKSetHSSMaxRank()`, `MatSTRUMPACKSetHSSLeafSize()`, `MatSTRUMPACKSetHSSMinSepSize()`
@*/
PetscErrorCode MatSTRUMPACKSetColPerm(Mat F, PetscBool cperm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveBool(F, cperm, 2);
  PetscTryMethod(F, "MatSTRUMPACKSetColPerm_C", (Mat, PetscBool), (F, cperm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSTRUMPACKSetHSSRelTol_STRUMPACK(Mat F, PetscReal rtol)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_set_HSS_rel_tol", STRUMPACK_set_HSS_rel_tol(*S, rtol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSTRUMPACKSetHSSRelTol - Set STRUMPACK relative tolerance for HSS compression

  Logically Collective

   Input Parameters:
+  F - the factored matrix obtained by calling `MatGetFactor()`
-  rtol - relative compression tolerance

  Options Database Key:
.   -mat_strumpack_hss_rel_tol <1e-2>         - Relative compression tolerance (None)

   Level: beginner

   References:
.  * - STRUMPACK manual

.seealso: [](chapter_matrices), `Mat`, `MatGetFactor()`, `MatSTRUMPACKSetReordering()`, `MatSTRUMPACKSetColPerm()`, `MatSTRUMPACKSetHSSAbsTol()`,
          `MatSTRUMPACKSetHSSMaxRank()`, `MatSTRUMPACKSetHSSLeafSize()`, `MatSTRUMPACKSetHSSMinSepSize()`
@*/
PetscErrorCode MatSTRUMPACKSetHSSRelTol(Mat F, PetscReal rtol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveReal(F, rtol, 2);
  PetscTryMethod(F, "MatSTRUMPACKSetHSSRelTol_C", (Mat, PetscReal), (F, rtol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSTRUMPACKSetHSSAbsTol_STRUMPACK(Mat F, PetscReal atol)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_set_HSS_abs_tol", STRUMPACK_set_HSS_abs_tol(*S, atol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSTRUMPACKSetHSSAbsTol - Set STRUMPACK absolute tolerance for HSS compression

   Logically Collective

   Input Parameters:
+  F - the factored matrix obtained by calling `MatGetFactor()`
-  atol - absolute compression tolerance

  Options Database Key:
.   -mat_strumpack_hss_abs_tol <1e-8>         - Absolute compression tolerance (None)

   Level: beginner

   References:
.  * - STRUMPACK manual

.seealso: [](chapter_matrices), `Mat`, `MatGetFactor()`, `MatSTRUMPACKSetReordering()`, `MatSTRUMPACKSetColPerm()`, `MatSTRUMPACKSetHSSRelTol()`,
          `MatSTRUMPACKSetHSSMaxRank()`, `MatSTRUMPACKSetHSSLeafSize()`, `MatSTRUMPACKSetHSSMinSepSize()`
@*/
PetscErrorCode MatSTRUMPACKSetHSSAbsTol(Mat F, PetscReal atol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveReal(F, atol, 2);
  PetscTryMethod(F, "MatSTRUMPACKSetHSSAbsTol_C", (Mat, PetscReal), (F, atol));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSTRUMPACKSetHSSMaxRank_STRUMPACK(Mat F, PetscInt hssmaxrank)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_set_HSS_max_rank", STRUMPACK_set_HSS_max_rank(*S, hssmaxrank));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSTRUMPACKSetHSSMaxRank - Set STRUMPACK maximum HSS rank

   Logically Collective

   Input Parameters:
+  F - the factored matrix obtained by calling `MatGetFactor()`
-  hssmaxrank - maximum rank used in low-rank approximation

  Options Database Key:
.   -mat_strumpack_max_rank - Maximum rank in HSS compression, when using pctype ilu (None)

   Level: beginner

   References:
.  * - STRUMPACK manual

.seealso: [](chapter_matrices), `Mat`, `MatGetFactor()`, `MatSTRUMPACKSetReordering()`, `MatSTRUMPACKSetColPerm()`, `MatSTRUMPACKSetHSSRelTol()`,
          `MatSTRUMPACKSetHSSAbsTol()`, `MatSTRUMPACKSetHSSLeafSize()`, `MatSTRUMPACKSetHSSMinSepSize()`
@*/
PetscErrorCode MatSTRUMPACKSetHSSMaxRank(Mat F, PetscInt hssmaxrank)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveInt(F, hssmaxrank, 2);
  PetscTryMethod(F, "MatSTRUMPACKSetHSSMaxRank_C", (Mat, PetscInt), (F, hssmaxrank));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSTRUMPACKSetHSSLeafSize_STRUMPACK(Mat F, PetscInt leaf_size)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_set_HSS_leaf_size", STRUMPACK_set_HSS_leaf_size(*S, leaf_size));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSTRUMPACKSetHSSLeafSize - Set STRUMPACK HSS leaf size

   Logically Collective

   Input Parameters:
+  F - the factored matrix obtained by calling `MatGetFactor()`
-  leaf_size - Size of diagonal blocks in HSS approximation

  Options Database Key:
.   -mat_strumpack_leaf_size - Size of diagonal blocks in HSS approximation, when using pctype ilu (None)

   Level: beginner

   References:
.  * - STRUMPACK manual

.seealso: [](chapter_matrices), `Mat`, `MatGetFactor()`, `MatSTRUMPACKSetReordering()`, `MatSTRUMPACKSetColPerm()`, `MatSTRUMPACKSetHSSRelTol()`,
          `MatSTRUMPACKSetHSSAbsTol()`, `MatSTRUMPACKSetHSSMaxRank()`, `MatSTRUMPACKSetHSSMinSepSize()`
@*/
PetscErrorCode MatSTRUMPACKSetHSSLeafSize(Mat F, PetscInt leaf_size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveInt(F, leaf_size, 2);
  PetscTryMethod(F, "MatSTRUMPACKSetHSSLeafSize_C", (Mat, PetscInt), (F, leaf_size));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSTRUMPACKSetHSSMinSepSize_STRUMPACK(Mat F, PetscInt hssminsize)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_set_HSS_min_sep_size", STRUMPACK_set_HSS_min_sep_size(*S, hssminsize));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSTRUMPACKSetHSSMinSepSize - Set STRUMPACK minimum separator size for low-rank approximation

   Logically Collective

   Input Parameters:
+  F - the factored matrix obtained by calling `MatGetFactor()`
-  hssminsize - minimum dense matrix size for low-rank approximation

  Options Database Key:
.   -mat_strumpack_hss_min_sep_size <hssminsize> - set the minimum separator size

   Level: beginner

   References:
.  * - STRUMPACK manual

.seealso: [](chapter_matrices), `Mat`, `MatGetFactor()`, `MatSTRUMPACKSetReordering()`, `MatSTRUMPACKSetColPerm()`, `MatSTRUMPACKSetHSSRelTol()`,
          `MatSTRUMPACKSetHSSAbsTol()`, `MatSTRUMPACKSetHSSMaxRank()`, `MatSTRUMPACKSetHSSLeafSize()`
@*/
PetscErrorCode MatSTRUMPACKSetHSSMinSepSize(Mat F, PetscInt hssminsize)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveInt(F, hssminsize, 2);
  PetscTryMethod(F, "MatSTRUMPACKSetHSSMinSepSize_C", (Mat, PetscInt), (F, hssminsize));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_STRUMPACK(Mat A, Vec b_mpi, Vec x)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)A->data;
  STRUMPACK_RETURN_CODE   sp_err;
  const PetscScalar      *bptr;
  PetscScalar            *xptr;

  PetscFunctionBegin;
  PetscCall(VecGetArray(x, &xptr));
  PetscCall(VecGetArrayRead(b_mpi, &bptr));

  PetscStackCallExternalVoid("STRUMPACK_solve", sp_err = STRUMPACK_solve(*S, (PetscScalar *)bptr, xptr, 0));
  switch (sp_err) {
  case STRUMPACK_SUCCESS:
    break;
  case STRUMPACK_MATRIX_NOT_SET: {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "STRUMPACK error: matrix was not set");
    break;
  }
  case STRUMPACK_REORDERING_ERROR: {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "STRUMPACK error: matrix reordering failed");
    break;
  }
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "STRUMPACK error: solve failed");
  }
  PetscCall(VecRestoreArray(x, &xptr));
  PetscCall(VecRestoreArrayRead(b_mpi, &bptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMatSolve_STRUMPACK(Mat A, Mat B_mpi, Mat X)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)B_mpi, &flg, MATSEQDENSE, MATMPIDENSE, NULL));
  PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Matrix B must be MATDENSE matrix");
  PetscCall(PetscObjectTypeCompareAny((PetscObject)X, &flg, MATSEQDENSE, MATMPIDENSE, NULL));
  PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Matrix X must be MATDENSE matrix");
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "MatMatSolve_STRUMPACK() is not implemented yet");
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatView_Info_STRUMPACK(Mat A, PetscViewer viewer)
{
  PetscFunctionBegin;
  /* check if matrix is strumpack type */
  if (A->ops->solve != MatSolve_STRUMPACK) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscViewerASCIIPrintf(viewer, "STRUMPACK sparse solver!\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatView_STRUMPACK(Mat A, PetscViewer viewer)
{
  PetscBool         iascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_INFO) PetscCall(MatView_Info_STRUMPACK(A, viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLUFactorNumeric_STRUMPACK(Mat F, Mat A, const MatFactorInfo *info)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;
  STRUMPACK_RETURN_CODE   sp_err;
  Mat_SeqAIJ             *A_d, *A_o;
  Mat_MPIAIJ             *mat;
  PetscInt                M = A->rmap->N, m = A->rmap->n;
  PetscBool               flg;
  const PetscScalar      *A_d_a, *A_o_a;

  PetscFunctionBegin;
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A, MATMPIAIJ, &flg));
  if (flg) { /* A might be MATMPIAIJCUSPARSE etc */
    mat = (Mat_MPIAIJ *)A->data;
    PetscCall(MatSeqAIJGetArrayRead(mat->A, &A_d_a)); /* Make sure mat is sync'ed to host */
    PetscCall(MatSeqAIJGetArrayRead(mat->B, &A_o_a));
    A_d = (Mat_SeqAIJ *)(mat->A)->data;
    A_o = (Mat_SeqAIJ *)(mat->B)->data;
    PetscStackCallExternalVoid("STRUMPACK_set_MPIAIJ_matrix", STRUMPACK_set_MPIAIJ_matrix(*S, &m, A_d->i, A_d->j, A_d_a, A_o->i, A_o->j, A_o_a, mat->garray));
    PetscCall(MatSeqAIJRestoreArrayRead(mat->A, &A_d_a));
    PetscCall(MatSeqAIJRestoreArrayRead(mat->B, &A_o_a));
  } else { /* A might be MATSEQAIJCUSPARSE etc */
    PetscCall(MatSeqAIJGetArrayRead(A, &A_d_a));
    A_d = (Mat_SeqAIJ *)A->data;
    PetscStackCallExternalVoid("STRUMPACK_set_csr_matrix", STRUMPACK_set_csr_matrix(*S, &M, A_d->i, A_d->j, A_d_a, 0));
    PetscCall(MatSeqAIJRestoreArrayRead(A, &A_d_a));
  }

  /* Reorder and Factor the matrix. */
  /* TODO figure out how to avoid reorder if the matrix values changed, but the pattern remains the same. */
  PetscStackCallExternalVoid("STRUMPACK_reorder", sp_err = STRUMPACK_reorder(*S));
  PetscStackCallExternalVoid("STRUMPACK_factor", sp_err = STRUMPACK_factor(*S));
  switch (sp_err) {
  case STRUMPACK_SUCCESS:
    break;
  case STRUMPACK_MATRIX_NOT_SET: {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "STRUMPACK error: matrix was not set");
    break;
  }
  case STRUMPACK_REORDERING_ERROR: {
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "STRUMPACK error: matrix reordering failed");
    break;
  }
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "STRUMPACK error: factorization failed");
  }
  F->assembled    = PETSC_TRUE;
  F->preallocated = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLUFactorSymbolic_STRUMPACK(Mat F, Mat A, IS r, IS c, const MatFactorInfo *info)
{
  STRUMPACK_SparseSolver       *S = (STRUMPACK_SparseSolver *)F->data;
  PetscBool                     flg, set;
  PetscReal                     ctol;
  PetscInt                      hssminsize, max_rank, leaf_size;
  STRUMPACK_REORDERING_STRATEGY ndcurrent, ndvalue;
  STRUMPACK_KRYLOV_SOLVER       itcurrent, itsolver;
  const char *const             STRUMPACKNDTypes[] = {"NATURAL", "METIS", "PARMETIS", "SCOTCH", "PTSCOTCH", "RCM", "STRUMPACKNDTypes", "", 0};
  const char *const             SolverTypes[]      = {"AUTO", "NONE", "REFINE", "PREC_GMRES", "GMRES", "PREC_BICGSTAB", "BICGSTAB", "SolverTypes", "", 0};

  PetscFunctionBegin;
  /* Set options to F */
  PetscOptionsBegin(PetscObjectComm((PetscObject)F), ((PetscObject)F)->prefix, "STRUMPACK Options", "Mat");
  PetscStackCallExternalVoid("STRUMPACK_HSS_rel_tol", ctol = (PetscReal)STRUMPACK_HSS_rel_tol(*S));
  PetscCall(PetscOptionsReal("-mat_strumpack_hss_rel_tol", "Relative compression tolerance", "None", ctol, &ctol, &set));
  if (set) PetscStackCallExternalVoid("STRUMPACK_set_HSS_rel_tol", STRUMPACK_set_HSS_rel_tol(*S, (double)ctol));

  PetscStackCallExternalVoid("STRUMPACK_HSS_abs_tol", ctol = (PetscReal)STRUMPACK_HSS_abs_tol(*S));
  PetscCall(PetscOptionsReal("-mat_strumpack_hss_abs_tol", "Absolute compression tolerance", "None", ctol, &ctol, &set));
  if (set) PetscStackCallExternalVoid("STRUMPACK_set_HSS_abs_tol", STRUMPACK_set_HSS_abs_tol(*S, (double)ctol));

  PetscStackCallExternalVoid("STRUMPACK_mc64job", flg = (STRUMPACK_mc64job(*S) == 0) ? PETSC_FALSE : PETSC_TRUE);
  PetscCall(PetscOptionsBool("-mat_strumpack_colperm", "Find a col perm to get nonzero diagonal", "None", flg, &flg, &set));
  if (set) PetscStackCallExternalVoid("STRUMPACK_set_mc64job", STRUMPACK_set_mc64job(*S, flg ? 5 : 0));

  PetscStackCallExternalVoid("STRUMPACK_HSS_min_sep_size", hssminsize = (PetscInt)STRUMPACK_HSS_min_sep_size(*S));
  PetscCall(PetscOptionsInt("-mat_strumpack_hss_min_sep_size", "Minimum size of separator for HSS compression", "None", hssminsize, &hssminsize, &set));
  if (set) PetscStackCallExternalVoid("STRUMPACK_set_HSS_min_sep_size", STRUMPACK_set_HSS_min_sep_size(*S, (int)hssminsize));

  PetscStackCallExternalVoid("STRUMPACK_HSS_max_rank", max_rank = (PetscInt)STRUMPACK_HSS_max_rank(*S));
  PetscCall(PetscOptionsInt("-mat_strumpack_max_rank", "Maximum rank in HSS approximation", "None", max_rank, &max_rank, &set));
  if (set) PetscStackCallExternalVoid("STRUMPACK_set_HSS_max_rank", STRUMPACK_set_HSS_max_rank(*S, (int)max_rank));

  PetscStackCallExternalVoid("STRUMPACK_HSS_leaf_size", leaf_size = (PetscInt)STRUMPACK_HSS_leaf_size(*S));
  PetscCall(PetscOptionsInt("-mat_strumpack_leaf_size", "Size of diagonal blocks in HSS approximation", "None", leaf_size, &leaf_size, &set));
  if (set) PetscStackCallExternalVoid("STRUMPACK_set_HSS_leaf_size", STRUMPACK_set_HSS_leaf_size(*S, (int)leaf_size));

  PetscStackCallExternalVoid("STRUMPACK_reordering_method", ndcurrent = STRUMPACK_reordering_method(*S));
  PetscCall(PetscOptionsEnum("-mat_strumpack_reordering", "Sparsity reducing matrix reordering", "None", STRUMPACKNDTypes, (PetscEnum)ndcurrent, (PetscEnum *)&ndvalue, &set));
  if (set) PetscStackCallExternalVoid("STRUMPACK_set_reordering_method", STRUMPACK_set_reordering_method(*S, ndvalue));

  /* Disable the outer iterative solver from STRUMPACK.                                       */
  /* When STRUMPACK is used as a direct solver, it will by default do iterative refinement.   */
  /* When STRUMPACK is used as an approximate factorization preconditioner (by enabling       */
  /* low-rank compression), it will use it's own preconditioned GMRES. Here we can disable    */
  /* the outer iterative solver, as PETSc uses STRUMPACK from within a KSP.                   */
  PetscStackCallExternalVoid("STRUMPACK_set_Krylov_solver", STRUMPACK_set_Krylov_solver(*S, STRUMPACK_DIRECT));

  PetscStackCallExternalVoid("STRUMPACK_Krylov_solver", itcurrent = STRUMPACK_Krylov_solver(*S));
  PetscCall(PetscOptionsEnum("-mat_strumpack_iterative_solver", "Select iterative solver from STRUMPACK", "None", SolverTypes, (PetscEnum)itcurrent, (PetscEnum *)&itsolver, &set));
  if (set) PetscStackCallExternalVoid("STRUMPACK_set_Krylov_solver", STRUMPACK_set_Krylov_solver(*S, itsolver));
  PetscOptionsEnd();

  F->ops->lufactornumeric = MatLUFactorNumeric_STRUMPACK;
  F->ops->solve           = MatSolve_STRUMPACK;
  F->ops->matsolve        = MatMatSolve_STRUMPACK;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFactorGetSolverType_aij_strumpack(Mat A, MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERSTRUMPACK;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  MATSOLVERSSTRUMPACK = "strumpack" - A solver package providing a direct sparse solver (PCLU)
  and a preconditioner (PCILU) using low-rank compression via the external package STRUMPACK.

  Consult the STRUMPACK-sparse manual for more info.

  Use ` ./configure --download-strumpack` to have PETSc installed with STRUMPACK

  Use `-pc_type lu` `-pc_factor_mat_solver_type strumpack` to use this as an exact (direct) solver.

  Use `-pc_type ilu` `-pc_factor_mat_solver_type strumpack` to enable low-rank compression (i.e, use as a preconditioner).

  Works with `MATAIJ` matrices

  Options Database Keys:
+ -mat_strumpack_verbose                    - verbose info
. -mat_strumpack_hss_rel_tol <1e-2>         - Relative compression tolerance (None)
. -mat_strumpack_hss_abs_tol <1e-8>         - Absolute compression tolerance (None)
. -mat_strumpack_colperm <TRUE>             - Permute matrix to make diagonal nonzeros (None)
. -mat_strumpack_hss_min_sep_size <256>     - Minimum size of separator for HSS compression (None)
. -mat_strumpack_max_rank                   - Maximum rank in HSS compression, when using pctype ilu (None)
. -mat_strumpack_leaf_size                  - Size of diagonal blocks in HSS approximation, when using pctype ilu (None)
. -mat_strumpack_reordering <METIS>         - Sparsity reducing matrix reordering see `MatSTRUMPACKReordering`
- -mat_strumpack_iterative_solver <DIRECT>  - Select iterative solver from STRUMPACK (choose one of) `AUTO`, `DIRECT`, `REFINE`, `PREC_GMRES`,
                                              `GMRES`, `PREC_BICGSTAB`, `BICGSTAB`

 Level: beginner

.seealso: [](chapter_matrices), `Mat`, `PCLU`, `PCILU`, `MATSOLVERSUPERLU_DIST`, `MATSOLVERMUMPS`, `PCFactorSetMatSolverType()`, `MatSolverType`,
          `MatGetFactor()`, `MatSTRUMPACKSetReordering()`, `MatSTRUMPACKSetColPerm()`, `MatSTRUMPACKSetHSSRelTol()`,
          `MatSTRUMPACKSetHSSAbsTol()`, `MatSTRUMPACKSetHSSMaxRank()`, `MatSTRUMPACKSetHSSLeafSize()`, `MatSTRUMPACKSetHSSMinSepSize()`
M*/
static PetscErrorCode MatGetFactor_aij_strumpack(Mat A, MatFactorType ftype, Mat *F)
{
  Mat                       B;
  PetscInt                  M = A->rmap->N, N = A->cmap->N;
  PetscBool                 verb, flg;
  STRUMPACK_SparseSolver   *S;
  STRUMPACK_INTERFACE       iface;
  const STRUMPACK_PRECISION table[2][2][2] = {
    {{STRUMPACK_FLOATCOMPLEX_64, STRUMPACK_DOUBLECOMPLEX_64}, {STRUMPACK_FLOAT_64, STRUMPACK_DOUBLE_64}},
    {{STRUMPACK_FLOATCOMPLEX, STRUMPACK_DOUBLECOMPLEX},       {STRUMPACK_FLOAT, STRUMPACK_DOUBLE}      }
  };
  const STRUMPACK_PRECISION prec = table[(sizeof(PetscInt) == 8) ? 0 : 1][(PETSC_SCALAR == PETSC_COMPLEX) ? 0 : 1][(PETSC_REAL == PETSC_FLOAT) ? 0 : 1];

  PetscFunctionBegin;
  /* Create the factorization matrix */
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
  PetscCall(MatSetSizes(B, A->rmap->n, A->cmap->n, M, N));
  PetscCall(PetscStrallocpy("strumpack", &((PetscObject)B)->type_name));
  PetscCall(MatSetUp(B));
  B->trivialsymbolic = PETSC_TRUE;
  if (ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU) {
    B->ops->lufactorsymbolic  = MatLUFactorSymbolic_STRUMPACK;
    B->ops->ilufactorsymbolic = MatLUFactorSymbolic_STRUMPACK;
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Factor type not supported");
  B->ops->getinfo     = MatGetInfo_External;
  B->ops->view        = MatView_STRUMPACK;
  B->ops->destroy     = MatDestroy_STRUMPACK;
  B->ops->getdiagonal = MatGetDiagonal_STRUMPACK;
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatFactorGetSolverType_C", MatFactorGetSolverType_aij_strumpack));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKSetReordering_C", MatSTRUMPACKSetReordering_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKSetColPerm_C", MatSTRUMPACKSetColPerm_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKSetHSSRelTol_C", MatSTRUMPACKSetHSSRelTol_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKSetHSSAbsTol_C", MatSTRUMPACKSetHSSAbsTol_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKSetHSSMaxRank_C", MatSTRUMPACKSetHSSMaxRank_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKSetHSSLeafSize_C", MatSTRUMPACKSetHSSLeafSize_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKSetHSSMinSepSize_C", MatSTRUMPACKSetHSSMinSepSize_STRUMPACK));
  B->factortype = ftype;
  PetscCall(PetscNew(&S));
  B->data = S;

  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A, MATSEQAIJ, &flg)); /* A might be MATSEQAIJCUSPARSE */
  iface = flg ? STRUMPACK_MT : STRUMPACK_MPI_DIST;

  PetscOptionsBegin(PetscObjectComm((PetscObject)B), ((PetscObject)B)->prefix, "STRUMPACK Options", "Mat");
  verb = PetscLogPrintInfo ? PETSC_TRUE : PETSC_FALSE;
  PetscCall(PetscOptionsBool("-mat_strumpack_verbose", "Print STRUMPACK information", "None", verb, &verb, NULL));

  PetscStackCallExternalVoid("STRUMPACK_init", STRUMPACK_init(S, PetscObjectComm((PetscObject)A), prec, iface, 0, NULL, verb));

  if (ftype == MAT_FACTOR_ILU) {
    /* When enabling HSS compression, the STRUMPACK solver becomes an incomplete              */
    /* (or approximate) LU factorization.                                                     */
    PetscStackCallExternalVoid("STRUMPACK_enable_HSS", STRUMPACK_enable_HSS(*S));
  }
  PetscOptionsEnd();

  /* set solvertype */
  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERSTRUMPACK, &B->solvertype));

  *F = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_STRUMPACK(void)
{
  PetscFunctionBegin;
  PetscCall(MatSolverTypeRegister(MATSOLVERSTRUMPACK, MATMPIAIJ, MAT_FACTOR_LU, MatGetFactor_aij_strumpack));
  PetscCall(MatSolverTypeRegister(MATSOLVERSTRUMPACK, MATSEQAIJ, MAT_FACTOR_LU, MatGetFactor_aij_strumpack));
  PetscCall(MatSolverTypeRegister(MATSOLVERSTRUMPACK, MATMPIAIJ, MAT_FACTOR_ILU, MatGetFactor_aij_strumpack));
  PetscCall(MatSolverTypeRegister(MATSOLVERSTRUMPACK, MATSEQAIJ, MAT_FACTOR_ILU, MatGetFactor_aij_strumpack));
  PetscFunctionReturn(PETSC_SUCCESS);
}
