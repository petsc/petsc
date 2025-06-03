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
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKGetReordering_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKSetColPerm_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKGetColPerm_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKSetGeometricNxyz_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKSetGeometricComponents_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKSetGeometricWidth_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKSetGPU_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKGetGPU_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKSetCompression_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKGetCompression_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKSetCompRelTol_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKGetCompRelTol_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKSetCompAbsTol_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKGetCompAbsTol_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKSetCompMaxRank_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKGetCompMaxRank_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKSetCompLeafSize_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKGetCompLeafSize_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKSetCompMinSepSize_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKGetCompMinSepSize_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKSetCompLossyPrecision_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKGetCompLossyPrecision_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKSetCompButterflyLevels_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSTRUMPACKGetCompButterflyLevels_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSTRUMPACKSetReordering_STRUMPACK(Mat F, MatSTRUMPACKReordering reordering)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_set_reordering_method", STRUMPACK_set_reordering_method(*S, (STRUMPACK_REORDERING_STRATEGY)reordering));
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode MatSTRUMPACKGetReordering_STRUMPACK(Mat F, MatSTRUMPACKReordering *reordering)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_reordering_method", *reordering = (MatSTRUMPACKReordering)STRUMPACK_reordering_method(*S));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSTRUMPACKSetReordering - Set STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master> fill-reducing reordering

  Logically Collective

  Input Parameters:
+ F          - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface
- reordering - the code to be used to find the fill-reducing reordering

  Options Database Key:
. -mat_strumpack_reordering <METIS> - Sparsity reducing matrix reordering, see `MatSTRUMPACKReordering`

  Level: intermediate

.seealso: `MATSOLVERSTRUMPACK`, [](ch_matrices), `Mat`, `MatSTRUMPACKReordering`, `MatGetFactor()`, `MatSTRUMPACKSetColPerm()`, `MatSTRUMPACKGetReordering()`
@*/
PetscErrorCode MatSTRUMPACKSetReordering(Mat F, MatSTRUMPACKReordering reordering)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(F, reordering, 2);
  PetscTryMethod(F, "MatSTRUMPACKSetReordering_C", (Mat, MatSTRUMPACKReordering), (F, reordering));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  MatSTRUMPACKGetReordering - Get STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master> fill-reducing reordering

  Logically Collective

  Input Parameters:
. F - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface

  Output Parameter:
. reordering - the code to be used to find the fill-reducing reordering

  Level: intermediate

.seealso: `MATSOLVERSTRUMPACK`, [](ch_matrices), `Mat`, `MatSTRUMPACKReordering`, `MatGetFactor()`, `MatSTRUMPACKSetReordering()`, `MatSTRUMPACKSetColPerm()`
@*/
PetscErrorCode MatSTRUMPACKGetReordering(Mat F, MatSTRUMPACKReordering *reordering)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscTryMethod(F, "MatSTRUMPACKGetReordering_C", (Mat, MatSTRUMPACKReordering *), (F, reordering));
  PetscValidLogicalCollectiveEnum(F, *reordering, 2);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSTRUMPACKSetColPerm_STRUMPACK(Mat F, PetscBool cperm)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_set_matching", STRUMPACK_set_matching(*S, cperm ? STRUMPACK_MATCHING_MAX_DIAGONAL_PRODUCT_SCALING : STRUMPACK_MATCHING_NONE));
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode MatSTRUMPACKGetColPerm_STRUMPACK(Mat F, PetscBool *cperm)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_matching", *cperm = (PetscBool)(STRUMPACK_matching(*S) != STRUMPACK_MATCHING_NONE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSTRUMPACKSetColPerm - Set whether STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master>
  should try to permute the columns of the matrix in order to get a nonzero diagonal

  Logically Collective

  Input Parameters:
+ F     - the factored matrix obtained by calling `MatGetFactor()`
- cperm - `PETSC_TRUE` to permute (internally) the columns of the matrix

  Options Database Key:
. -mat_strumpack_colperm <cperm> - true to use the permutation

  Level: intermediate

.seealso: `MATSOLVERSTRUMPACK`, [](ch_matrices), `MatSTRUMPACKSetReordering()`, `Mat`, `MatGetFactor()`, `MatSTRUMPACKGetColPerm()`
@*/
PetscErrorCode MatSTRUMPACKSetColPerm(Mat F, PetscBool cperm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveBool(F, cperm, 2);
  PetscTryMethod(F, "MatSTRUMPACKSetColPerm_C", (Mat, PetscBool), (F, cperm));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  MatSTRUMPACKGetColPerm - Get whether STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master>
  will try to permute the columns of the matrix in order to get a nonzero diagonal

  Logically Collective

  Input Parameters:
. F - the factored matrix obtained by calling `MatGetFactor()`

  Output Parameter:
. cperm - Indicates whether STRUMPACK will permute columns

  Level: intermediate

.seealso: `MATSOLVERSTRUMPACK`, [](ch_matrices), `MatSTRUMPACKSetReordering()`, `Mat`, `MatGetFactor()`, `MatSTRUMPACKSetColPerm()`
@*/
PetscErrorCode MatSTRUMPACKGetColPerm(Mat F, PetscBool *cperm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscTryMethod(F, "MatSTRUMPACKGetColPerm_C", (Mat, PetscBool *), (F, cperm));
  PetscValidLogicalCollectiveBool(F, *cperm, 2);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSTRUMPACKSetGPU_STRUMPACK(Mat F, PetscBool gpu)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  if (gpu) {
#if !(defined(STRUMPACK_USE_CUDA) || defined(STRUMPACK_USE_HIP) || defined(STRUMPACK_USE_SYCL))
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Warning: strumpack was not configured with GPU support\n"));
#endif
    PetscStackCallExternalVoid("STRUMPACK_enable_gpu", STRUMPACK_enable_gpu(*S));
  } else PetscStackCallExternalVoid("STRUMPACK_disable_gpu", STRUMPACK_disable_gpu(*S));
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode MatSTRUMPACKGetGPU_STRUMPACK(Mat F, PetscBool *gpu)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_use_gpu", *gpu = (PetscBool)STRUMPACK_use_gpu(*S));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSTRUMPACKSetGPU - Set whether STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master>
  should enable GPU acceleration (not supported for all compression types)

  Logically Collective

  Input Parameters:
+ F   - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface
- gpu - whether or not to use GPU acceleration

  Options Database Key:
. -mat_strumpack_gpu <gpu> - true to use gpu offload

  Level: intermediate

.seealso: `MATSOLVERSTRUMPACK`, `MatGetFactor()`, `MatSTRUMPACKGetGPU()`
@*/
PetscErrorCode MatSTRUMPACKSetGPU(Mat F, PetscBool gpu)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveBool(F, gpu, 2);
  PetscTryMethod(F, "MatSTRUMPACKSetGPU_C", (Mat, PetscBool), (F, gpu));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  MatSTRUMPACKGetGPU - Get whether STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master>
  will try to use GPU acceleration (not supported for all compression types)

  Logically Collective

  Input Parameters:
. F - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface

  Output Parameter:
. gpu - whether or not STRUMPACK will try to use GPU acceleration

  Level: intermediate

.seealso: `MATSOLVERSTRUMPACK`, `MatGetFactor()`, `MatSTRUMPACKSetGPU()`
@*/
PetscErrorCode MatSTRUMPACKGetGPU(Mat F, PetscBool *gpu)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscTryMethod(F, "MatSTRUMPACKGetGPU_C", (Mat, PetscBool *), (F, gpu));
  PetscValidLogicalCollectiveBool(F, *gpu, 2);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSTRUMPACKSetCompression_STRUMPACK(Mat F, MatSTRUMPACKCompressionType comp)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
#if !defined(STRUMPACK_USE_BPACK)
  PetscCheck(comp != MAT_STRUMPACK_COMPRESSION_TYPE_HODLR && comp != MAT_STRUMPACK_COMPRESSION_TYPE_BLR_HODLR && comp != MAT_STRUMPACK_COMPRESSION_TYPE_ZFP_BLR_HODLR, PetscObjectComm((PetscObject)F), PETSC_ERR_SUP, "Compression scheme requires ButterflyPACK, please reconfigure with --download-butterflypack");
#endif
#if !defined(STRUMPACK_USE_ZFP)
  PetscCheck(comp != MAT_STRUMPACK_COMPRESSION_TYPE_ZFP_BLR_HODLR && comp != MAT_STRUMPACK_COMPRESSION_TYPE_LOSSLESS && comp != MAT_STRUMPACK_COMPRESSION_TYPE_LOSSY, PetscObjectComm((PetscObject)F), PETSC_ERR_SUP, "Compression scheme requires ZFP, please reconfigure with --download-zfp");
#endif
  PetscStackCallExternalVoid("STRUMPACK_set_compression", STRUMPACK_set_compression(*S, (STRUMPACK_COMPRESSION_TYPE)comp));
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode MatSTRUMPACKGetCompression_STRUMPACK(Mat F, MatSTRUMPACKCompressionType *comp)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_compression", *comp = (MatSTRUMPACKCompressionType)STRUMPACK_compression(*S));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSTRUMPACKSetCompression - Set STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master> compression type

  Input Parameters:
+ F    - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface
- comp - Type of compression to be used in the approximate sparse factorization

  Options Database Key:
. -mat_strumpack_compression <NONE> - Type of rank-structured compression in sparse LU factors (choose one of) NONE HSS BLR HODLR BLR_HODLR ZFP_BLR_HODLR LOSSLESS LOSSY

  Level: intermediate

  Note:
  Default for `comp` is `MAT_STRUMPACK_COMPRESSION_TYPE_NONE` for `-pc_type lu` and `MAT_STRUMPACK_COMPRESSION_TYPE_BLR`
  for `-pc_type ilu`

.seealso: `MATSOLVERSTRUMPACK`, `MatGetFactor()`, `MatSTRUMPACKCompressionType`, `MatSTRUMPACKGetCompression()`
@*/
PetscErrorCode MatSTRUMPACKSetCompression(Mat F, MatSTRUMPACKCompressionType comp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(F, comp, 2);
  PetscTryMethod(F, "MatSTRUMPACKSetCompression_C", (Mat, MatSTRUMPACKCompressionType), (F, comp));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  MatSTRUMPACKGetCompression - Get STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master> compression type

  Input Parameters:
. F - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface

  Output Parameter:
. comp - Type of compression to be used in the approximate sparse factorization

  Level: intermediate

  Note:
  Default is `MAT_STRUMPACK_COMPRESSION_TYPE_NONE` for `-pc_type lu` and `MAT_STRUMPACK_COMPRESSION_TYPE_BLR` for `-pc_type ilu`

.seealso: `MATSOLVERSTRUMPACK`, `MatGetFactor()`, `MatSTRUMPACKCompressionType`, `MatSTRUMPACKSetCompression()`
@*/
PetscErrorCode MatSTRUMPACKGetCompression(Mat F, MatSTRUMPACKCompressionType *comp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscTryMethod(F, "MatSTRUMPACKGetCompression_C", (Mat, MatSTRUMPACKCompressionType *), (F, comp));
  PetscValidLogicalCollectiveEnum(F, *comp, 2);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSTRUMPACKSetCompRelTol_STRUMPACK(Mat F, PetscReal rtol)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_set_compression_rel_tol", STRUMPACK_set_compression_rel_tol(*S, rtol));
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode MatSTRUMPACKGetCompRelTol_STRUMPACK(Mat F, PetscReal *rtol)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_compression_rel_tol", *rtol = (PetscReal)STRUMPACK_compression_rel_tol(*S));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSTRUMPACKSetCompRelTol - Set STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master> relative tolerance for compression

  Logically Collective

  Input Parameters:
+ F    - the factored matrix obtained by calling `MatGetFactor()`
- rtol - relative compression tolerance

  Options Database Key:
. -mat_strumpack_compression_rel_tol <1e-4> - Relative compression tolerance, when using `-pctype ilu`

  Level: intermediate

.seealso: `MATSOLVERSTRUMPACK`, [](ch_matrices), `Mat`, `MatGetFactor()`, `MatSTRUMPACKGetCompRelTol()`, `MatSTRUMPACKSetReordering()`, `MatSTRUMPACKSetColPerm()`
@*/
PetscErrorCode MatSTRUMPACKSetCompRelTol(Mat F, PetscReal rtol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveReal(F, rtol, 2);
  PetscTryMethod(F, "MatSTRUMPACKSetCompRelTol_C", (Mat, PetscReal), (F, rtol));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  MatSTRUMPACKGetCompRelTol - Get STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master> relative tolerance for compression

  Logically Collective

  Input Parameters:
. F - the factored matrix obtained by calling `MatGetFactor()`

  Output Parameter:
. rtol - relative compression tolerance

  Level: intermediate

.seealso: `MATSOLVERSTRUMPACK`, [](ch_matrices), `Mat`, `MatGetFactor()`, `MatSTRUMPACKSetCompRelTol()`, `MatSTRUMPACKSetReordering()`, `MatSTRUMPACKSetColPerm()`
@*/
PetscErrorCode MatSTRUMPACKGetCompRelTol(Mat F, PetscReal *rtol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscTryMethod(F, "MatSTRUMPACKGetCompRelTol_C", (Mat, PetscReal *), (F, rtol));
  PetscValidLogicalCollectiveReal(F, *rtol, 2);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSTRUMPACKSetCompAbsTol_STRUMPACK(Mat F, PetscReal atol)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_set_compression_abs_tol", STRUMPACK_set_compression_abs_tol(*S, atol));
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode MatSTRUMPACKGetCompAbsTol_STRUMPACK(Mat F, PetscReal *atol)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_compression_abs_tol", *atol = (PetscReal)STRUMPACK_compression_abs_tol(*S));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSTRUMPACKSetCompAbsTol - Set STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master> absolute tolerance for compression

  Logically Collective

  Input Parameters:
+ F    - the factored matrix obtained by calling `MatGetFactor()`
- atol - absolute compression tolerance

  Options Database Key:
. -mat_strumpack_compression_abs_tol <1e-10> - Absolute compression tolerance, when using `-pctype ilu`

  Level: intermediate

.seealso: `MATSOLVERSTRUMPACK`, [](ch_matrices), `Mat`, `MatGetFactor()`, `MatSTRUMPACKGetCompAbsTol()`, `MatSTRUMPACKSetReordering()`, `MatSTRUMPACKSetColPerm()`
@*/
PetscErrorCode MatSTRUMPACKSetCompAbsTol(Mat F, PetscReal atol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveReal(F, atol, 2);
  PetscTryMethod(F, "MatSTRUMPACKSetCompAbsTol_C", (Mat, PetscReal), (F, atol));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  MatSTRUMPACKGetCompAbsTol - Get STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master> absolute tolerance for compression

  Logically Collective

  Input Parameters:
. F - the factored matrix obtained by calling `MatGetFactor()`

  Output Parameter:
. atol - absolute compression tolerance

  Level: intermediate

.seealso: `MATSOLVERSTRUMPACK`, [](ch_matrices), `Mat`, `MatGetFactor()`, `MatSTRUMPACKSetCompAbsTol()`, `MatSTRUMPACKSetReordering()`, `MatSTRUMPACKSetColPerm()`
@*/
PetscErrorCode MatSTRUMPACKGetCompAbsTol(Mat F, PetscReal *atol)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscTryMethod(F, "MatSTRUMPACKGetCompAbsTol_C", (Mat, PetscReal *), (F, atol));
  PetscValidLogicalCollectiveReal(F, *atol, 2);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSTRUMPACKSetCompLeafSize_STRUMPACK(Mat F, PetscInt leaf_size)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_set_compression_leaf_size", STRUMPACK_set_compression_leaf_size(*S, leaf_size));
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode MatSTRUMPACKGetCompLeafSize_STRUMPACK(Mat F, PetscInt *leaf_size)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_compression_leaf_size", *leaf_size = (PetscInt)STRUMPACK_compression_leaf_size(*S));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSTRUMPACKSetCompLeafSize - Set STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master> leaf size for HSS, BLR, HODLR...

  Logically Collective

  Input Parameters:
+ F         - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface
- leaf_size - Size of diagonal blocks in rank-structured approximation

  Options Database Key:
. -mat_strumpack_compression_leaf_size - Size of diagonal blocks in rank-structured approximation, when using `-pctype ilu`

  Level: intermediate

.seealso: `MATSOLVERSTRUMPACK`, [](ch_matrices), `Mat`, `MatGetFactor()`, `MatSTRUMPACKGetCompLeafSize()`, `MatSTRUMPACKSetReordering()`, `MatSTRUMPACKSetColPerm()`
@*/
PetscErrorCode MatSTRUMPACKSetCompLeafSize(Mat F, PetscInt leaf_size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveInt(F, leaf_size, 2);
  PetscTryMethod(F, "MatSTRUMPACKSetCompLeafSize_C", (Mat, PetscInt), (F, leaf_size));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  MatSTRUMPACKGetCompLeafSize - Get STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master> leaf size for HSS, BLR, HODLR...

  Logically Collective

  Input Parameters:
. F - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface

  Output Parameter:
. leaf_size - Size of diagonal blocks in rank-structured approximation

  Level: intermediate

.seealso: `MATSOLVERSTRUMPACK`, [](ch_matrices), `Mat`, `MatGetFactor()`, `MatSTRUMPACKSetCompLeafSize()`, `MatSTRUMPACKSetReordering()`, `MatSTRUMPACKSetColPerm()`
@*/
PetscErrorCode MatSTRUMPACKGetCompLeafSize(Mat F, PetscInt *leaf_size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscTryMethod(F, "MatSTRUMPACKGetCompLeafSize_C", (Mat, PetscInt *), (F, leaf_size));
  PetscValidLogicalCollectiveInt(F, *leaf_size, 2);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSTRUMPACKSetGeometricNxyz_STRUMPACK(Mat F, PetscInt nx, PetscInt ny, PetscInt nz)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  if (nx < 1) {
    PetscCheck(nx == PETSC_DECIDE || nx == PETSC_DEFAULT, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_OUTOFRANGE, "nx < 1");
    nx = 1;
  }
  PetscStackCallExternalVoid("STRUMPACK_set_nx", STRUMPACK_set_nx(*S, nx));
  if (ny < 1) {
    PetscCheck(ny == PETSC_DECIDE || ny == PETSC_DEFAULT, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_OUTOFRANGE, "ny < 1");
    ny = 1;
  }
  PetscStackCallExternalVoid("STRUMPACK_set_ny", STRUMPACK_set_ny(*S, ny));
  if (nz < 1) {
    PetscCheck(nz == PETSC_DECIDE || nz == PETSC_DEFAULT, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_OUTOFRANGE, "nz < 1");
    nz = 1;
  }
  PetscStackCallExternalVoid("STRUMPACK_set_nz", STRUMPACK_set_nz(*S, nz));
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode MatSTRUMPACKSetGeometricComponents_STRUMPACK(Mat F, PetscInt nc)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_set_components", STRUMPACK_set_components(*S, nc));
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode MatSTRUMPACKSetGeometricWidth_STRUMPACK(Mat F, PetscInt w)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_set_separator_width", STRUMPACK_set_separator_width(*S, w));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSTRUMPACKSetGeometricNxyz - Set STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master> mesh x, y and z dimensions, for use with GEOMETRIC ordering.

  Logically Collective

  Input Parameters:
+ F  - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface
. nx - x dimension of the mesh
. ny - y dimension of the mesh
- nz - z dimension of the mesh

  Level: intermediate

  Note:
  If the mesh is two (or one) dimensional one can use 1, `PETSC_DECIDE` or `PETSC_DEFAULT`
  for the missing z (and y) dimensions.

.seealso: `MATSOLVERSTRUMPACK`, `MatGetFactor()`
@*/
PetscErrorCode MatSTRUMPACKSetGeometricNxyz(Mat F, PetscInt nx, PetscInt ny, PetscInt nz)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveInt(F, nx, 2);
  PetscValidLogicalCollectiveInt(F, ny, 3);
  PetscValidLogicalCollectiveInt(F, nz, 4);
  PetscTryMethod(F, "MatSTRUMPACKSetGeometricNxyz_C", (Mat, PetscInt, PetscInt, PetscInt), (F, nx, ny, nz));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  MatSTRUMPACKSetGeometricComponents - Set STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master>
  number of degrees of freedom per mesh point, for use with GEOMETRIC ordering.

  Logically Collective

  Input Parameters:
+ F  - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface
- nc - Number of components/dof's per grid point

  Options Database Key:
. -mat_strumpack_geometric_components <1> - Number of components per mesh point, for geometric nested dissection ordering

  Level: intermediate

.seealso: `MATSOLVERSTRUMPACK`, `MatGetFactor()`
@*/
PetscErrorCode MatSTRUMPACKSetGeometricComponents(Mat F, PetscInt nc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveInt(F, nc, 2);
  PetscTryMethod(F, "MatSTRUMPACKSetGeometricComponents_C", (Mat, PetscInt), (F, nc));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  MatSTRUMPACKSetGeometricWidth - Set STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master> width of the separator, for use with GEOMETRIC ordering.

  Logically Collective

  Input Parameters:
+ F - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface
- w - width of the separator

  Options Database Key:
. -mat_strumpack_geometric_width <1> - Width of the separator of the mesh, for geometric nested dissection ordering

  Level: intermediate

.seealso: `MATSOLVERSTRUMPACK`, `MatGetFactor()`
@*/
PetscErrorCode MatSTRUMPACKSetGeometricWidth(Mat F, PetscInt w)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveInt(F, w, 2);
  PetscTryMethod(F, "MatSTRUMPACKSetGeometricWidth_C", (Mat, PetscInt), (F, w));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSTRUMPACKSetCompMinSepSize_STRUMPACK(Mat F, PetscInt min_sep_size)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_set_compression_min_sep_size", STRUMPACK_set_compression_min_sep_size(*S, min_sep_size));
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode MatSTRUMPACKGetCompMinSepSize_STRUMPACK(Mat F, PetscInt *min_sep_size)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_compression_min_sep_size", *min_sep_size = (PetscInt)STRUMPACK_compression_min_sep_size(*S));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSTRUMPACKSetCompMinSepSize - Set STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master> minimum separator size for low-rank approximation

  Logically Collective

  Input Parameters:
+ F            - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface
- min_sep_size - minimum dense matrix size for low-rank approximation

  Options Database Key:
. -mat_strumpack_compression_min_sep_size <min_sep_size> - Minimum size of dense sub-block for low-rank compression

  Level: intermediate

.seealso: `MATSOLVERSTRUMPACK`, `MatGetFactor()`, `MatSTRUMPACKGetCompMinSepSize()`
@*/
PetscErrorCode MatSTRUMPACKSetCompMinSepSize(Mat F, PetscInt min_sep_size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveInt(F, min_sep_size, 2);
  PetscTryMethod(F, "MatSTRUMPACKSetCompMinSepSize_C", (Mat, PetscInt), (F, min_sep_size));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  MatSTRUMPACKGetCompMinSepSize - Get STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master> minimum separator size for low-rank approximation

  Logically Collective

  Input Parameters:
. F - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface

  Output Parameter:
. min_sep_size - minimum dense matrix size for low-rank approximation

  Level: intermediate

.seealso: `MATSOLVERSTRUMPACK`, `MatGetFactor()`, `MatSTRUMPACKSetCompMinSepSize()`
@*/
PetscErrorCode MatSTRUMPACKGetCompMinSepSize(Mat F, PetscInt *min_sep_size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscTryMethod(F, "MatSTRUMPACKGetCompMinSepSize_C", (Mat, PetscInt *), (F, min_sep_size));
  PetscValidLogicalCollectiveInt(F, *min_sep_size, 2);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSTRUMPACKSetCompLossyPrecision_STRUMPACK(Mat F, PetscInt lossy_prec)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_set_compression_lossy_precision", STRUMPACK_set_compression_lossy_precision(*S, lossy_prec));
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode MatSTRUMPACKGetCompLossyPrecision_STRUMPACK(Mat F, PetscInt *lossy_prec)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_compression_lossy_precision", *lossy_prec = (PetscInt)STRUMPACK_compression_lossy_precision(*S));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSTRUMPACKSetCompLossyPrecision - Set STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master> precision for lossy compression (requires ZFP support)

  Logically Collective

  Input Parameters:
+ F          - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface
- lossy_prec - Number of bitplanes to use in lossy compression

  Options Database Key:
. -mat_strumpack_compression_lossy_precision <lossy_prec> - Precision when using lossy compression [1-64], when using `-pctype ilu -mat_strumpack_compression MAT_STRUMPACK_COMPRESSION_TYPE_LOSSY`

  Level: intermediate

.seealso: `MATSOLVERSTRUMPACK`, `MatGetFactor()`, `MatSTRUMPACKGetCompLossyPrecision()`
@*/
PetscErrorCode MatSTRUMPACKSetCompLossyPrecision(Mat F, PetscInt lossy_prec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveInt(F, lossy_prec, 2);
  PetscTryMethod(F, "MatSTRUMPACKSetCompLossyPrecision_C", (Mat, PetscInt), (F, lossy_prec));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  MatSTRUMPACKGetCompLossyPrecision - Get STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master> precision for lossy compression (requires ZFP support)

  Logically Collective

  Input Parameters:
. F - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface

  Output Parameter:
. lossy_prec - Number of bitplanes to use in lossy compression

  Level: intermediate

.seealso: `MATSOLVERSTRUMPACK`, `MatGetFactor()`, `MatSTRUMPACKSetCompLossyPrecision()`
@*/
PetscErrorCode MatSTRUMPACKGetCompLossyPrecision(Mat F, PetscInt *lossy_prec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscTryMethod(F, "MatSTRUMPACKGetCompLossyPrecision_C", (Mat, PetscInt *), (F, lossy_prec));
  PetscValidLogicalCollectiveInt(F, *lossy_prec, 2);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSTRUMPACKSetCompButterflyLevels_STRUMPACK(Mat F, PetscInt bfly_lvls)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_set_compression_butterfly_levels", STRUMPACK_set_compression_butterfly_levels(*S, bfly_lvls));
  PetscFunctionReturn(PETSC_SUCCESS);
}
static PetscErrorCode MatSTRUMPACKGetCompButterflyLevels_STRUMPACK(Mat F, PetscInt *bfly_lvls)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;

  PetscFunctionBegin;
  PetscStackCallExternalVoid("STRUMPACK_compression_butterfly_levels", *bfly_lvls = (PetscInt)STRUMPACK_compression_butterfly_levels(*S));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSTRUMPACKSetCompButterflyLevels - Set STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master>
  number of butterfly levels in HODLR compression (requires ButterflyPACK support)

  Logically Collective

  Input Parameters:
+ F         - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface
- bfly_lvls - Number of levels of butterfly compression in HODLR compression

  Options Database Key:
. -mat_strumpack_compression_butterfly_levels <bfly_lvls> - Number of levels in the hierarchically off-diagonal matrix for which to use butterfly,
                                                            when using `-pctype ilu`, (BLR_)HODLR compression

  Level: intermediate

.seealso: `MATSOLVERSTRUMPACK`, `MatGetFactor()`, `MatSTRUMPACKGetCompButterflyLevels()`
@*/
PetscErrorCode MatSTRUMPACKSetCompButterflyLevels(Mat F, PetscInt bfly_lvls)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveInt(F, bfly_lvls, 2);
  PetscTryMethod(F, "MatSTRUMPACKSetButterflyLevels_C", (Mat, PetscInt), (F, bfly_lvls));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*@
  MatSTRUMPACKGetCompButterflyLevels - Get STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master>
  number of butterfly levels in HODLR compression (requires ButterflyPACK support)

  Logically Collective

  Input Parameters:
. F - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface

  Output Parameter:
. bfly_lvls - Number of levels of butterfly compression in HODLR compression

  Level: intermediate

.seealso: `MATSOLVERSTRUMPACK`, `MatGetFactor()`, `MatSTRUMPACKSetCompButterflyLevels()`
@*/
PetscErrorCode MatSTRUMPACKGetCompButterflyLevels(Mat F, PetscInt *bfly_lvls)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscTryMethod(F, "MatSTRUMPACKGetButterflyLevels_C", (Mat, PetscInt *), (F, bfly_lvls));
  PetscValidLogicalCollectiveInt(F, *bfly_lvls, 2);
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
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)A->data;
  STRUMPACK_RETURN_CODE   sp_err;
  PetscBool               flg;
  PetscInt                m = A->rmap->n, nrhs;
  const PetscScalar      *bptr;
  PetscScalar            *xptr;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)B_mpi, &flg, MATSEQDENSE, MATMPIDENSE, NULL));
  PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Matrix B must be MATDENSE matrix");
  PetscCall(PetscObjectTypeCompareAny((PetscObject)X, &flg, MATSEQDENSE, MATMPIDENSE, NULL));
  PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Matrix X must be MATDENSE matrix");

  PetscCall(MatGetSize(B_mpi, NULL, &nrhs));
  PetscCall(MatDenseGetArray(X, &xptr));
  PetscCall(MatDenseGetArrayRead(B_mpi, &bptr));

  PetscStackCallExternalVoid("STRUMPACK_matsolve", sp_err = STRUMPACK_matsolve(*S, nrhs, bptr, m, xptr, m, 0));
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
  PetscCall(MatDenseRestoreArrayRead(B_mpi, &bptr));
  PetscCall(MatDenseRestoreArray(X, &xptr));
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
  PetscBool         isascii;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_INFO) PetscCall(MatView_Info_STRUMPACK(A, viewer));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLUFactorNumeric_STRUMPACK(Mat F, Mat A, const MatFactorInfo *info)
{
  STRUMPACK_SparseSolver *S = (STRUMPACK_SparseSolver *)F->data;
  STRUMPACK_RETURN_CODE   sp_err;
  Mat                     Aloc;
  const PetscScalar      *av;
  const PetscInt         *ai = NULL, *aj = NULL;
  PetscInt                M = A->rmap->N, m = A->rmap->n, dummy;
  PetscBool               ismpiaij, isseqaij, flg;

  PetscFunctionBegin;
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A, MATSEQAIJ, &isseqaij));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A, MATMPIAIJ, &ismpiaij));
  if (ismpiaij) {
    PetscCall(MatMPIAIJGetLocalMat(A, MAT_INITIAL_MATRIX, &Aloc));
  } else if (isseqaij) {
    PetscCall(PetscObjectReference((PetscObject)A));
    Aloc = A;
  } else SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Not for type %s", ((PetscObject)A)->type_name);

  PetscCall(MatGetRowIJ(Aloc, 0, PETSC_FALSE, PETSC_FALSE, &dummy, &ai, &aj, &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_SUP, "GetRowIJ failed");
  PetscCall(MatSeqAIJGetArrayRead(Aloc, &av));

  if (ismpiaij) {
    const PetscInt *dist = NULL;
    PetscCall(MatGetOwnershipRanges(A, &dist));
    PetscStackCallExternalVoid("STRUMPACK_set_distributed_csr_matrix", STRUMPACK_set_distributed_csr_matrix(*S, &m, ai, aj, av, dist, 0));
  } else if (isseqaij) {
    PetscStackCallExternalVoid("STRUMPACK_set_csr_matrix", STRUMPACK_set_csr_matrix(*S, &M, ai, aj, av, 0));
  }

  PetscCall(MatRestoreRowIJ(Aloc, 0, PETSC_FALSE, PETSC_FALSE, &dummy, &ai, &aj, &flg));
  PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_SUP, "RestoreRowIJ failed");
  PetscCall(MatSeqAIJRestoreArrayRead(Aloc, &av));
  PetscCall(MatDestroy(&Aloc));

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
  PetscFunctionBegin;
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
  MATSOLVERSTRUMPACK = "strumpack" - A solver package providing a direct sparse solver (PCLU)
  and a preconditioner (PCILU) using low-rank compression via the external package STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master>.

  Use `./configure --download-strumpack --download-metis` to have PETSc installed with STRUMPACK.

  For full functionality, add `--download-slate --download-magma --download-parmetis --download-ptscotch --download-zfp --download-butterflypack`.
  SLATE provides GPU support in the multi-GPU setting, providing ScaLAPACK functionality but with GPU acceleration.
  MAGMA can optionally be used for on node GPU support instead cuBLAS/cuSOLVER, and performs slightly better.
  ParMETIS and PTScotch can be used for parallel fill-reducing ordering.
  ZFP is used for floating point compression of the sparse factors (LOSSY or LOSSLESS compression).
  ButterflyPACK is used for HODLR (Hierarchically Off-Diagonal Low Rank) and HODBF (Hierarchically Off-Diagonal Butterfly) compression of the sparse factors.

  Options Database Keys:
+ -mat_strumpack_verbose                      - Enable verbose output
. -mat_strumpack_compression                  - Type of rank-structured compression in sparse LU factors (choose one of) NONE HSS BLR HODLR BLR_HODLR ZFP_BLR_HODLR LOSSLESS LOSSY
. -mat_strumpack_compression_rel_tol          - Relative compression tolerance, when using `-pctype ilu`
. -mat_strumpack_compression_abs_tol          - Absolute compression tolerance, when using `-pctype ilu`
. -mat_strumpack_compression_min_sep_size     - Minimum size of separator for rank-structured compression, when using `-pctype ilu`
. -mat_strumpack_compression_leaf_size        - Size of diagonal blocks in rank-structured approximation, when using `-pctype ilu`
. -mat_strumpack_compression_lossy_precision  - Precision when using lossy compression [1-64], when using `-pctype ilu`, compression LOSSY (requires ZFP support)
. -mat_strumpack_compression_butterfly_levels - Number of levels in the hierarchically off-diagonal matrix for which to use butterfly, when using `-pctype ilu`, (BLR_)HODLR compression (requires ButterflyPACK support)
. -mat_strumpack_gpu                          - Enable GPU acceleration in numerical factorization (not supported for all compression types)
. -mat_strumpack_colperm <TRUE>               - Permute matrix to make diagonal nonzeros
. -mat_strumpack_reordering <METIS>           - Sparsity reducing matrix reordering (choose one of) NATURAL METIS PARMETIS SCOTCH PTSCOTCH RCM GEOMETRIC AMD MMD AND MLF SPECTRAL
. -mat_strumpack_geometric_xyz <1,1,1>        - Mesh x,y,z dimensions, for use with GEOMETRIC ordering
. -mat_strumpack_geometric_components <1>     - Number of components per mesh point, for geometric nested dissection ordering
. -mat_strumpack_geometric_width <1>          - Width of the separator of the mesh, for geometric nested dissection ordering
- -mat_strumpack_metis_nodeNDP                - Use METIS_NodeNDP instead of METIS_NodeND, for a more balanced tree

 Level: beginner

 Notes:
 Recommended use is 1 MPI process per GPU.

 Use `-pc_type lu` `-pc_factor_mat_solver_type strumpack` to use this as an exact (direct) solver.

 Use `-pc_type ilu` `-pc_factor_mat_solver_type strumpack` to enable low-rank compression (i.e, use as a preconditioner), by default using block low rank (BLR).

 Works with `MATAIJ` matrices

 HODLR, BLR_HODBF and ZFP_BLR_HODLR compression require STRUMPACK to be configured with ButterflyPACK support (`--download-butterflypack`).

 LOSSY, LOSSLESS and ZFP_BLR_HODLR compression require STRUMPACK to be configured with ZFP support (`--download-zfp`).

.seealso: `MATSOLVERSTRUMPACK`, [](ch_matrices), `Mat`, `PCLU`, `PCILU`, `MATSOLVERSUPERLU_DIST`, `MATSOLVERMUMPS`, `PCFactorSetMatSolverType()`, `MatSolverType`,
          `MatGetFactor()`, `MatSTRUMPACKSetReordering()`, `MatSTRUMPACKReordering`, `MatSTRUMPACKCompressionType`, `MatSTRUMPACKSetColPerm()`.
M*/
static PetscErrorCode MatGetFactor_aij_strumpack(Mat A, MatFactorType ftype, Mat *F)
{
  Mat       B;
  PetscInt  M = A->rmap->N, N = A->cmap->N;
  PetscBool verb, flg, set;
  PetscReal ctol;
  PetscInt  min_sep_size, leaf_size, nxyz[3], nrdims, nc, w;
#if defined(STRUMPACK_USE_ZFP)
  PetscInt lossy_prec;
#endif
#if defined(STRUMPACK_USE_BPACK)
  PetscInt bfly_lvls;
#endif
#if defined(STRUMPACK_USE_SLATE_SCALAPACK)
  PetscMPIInt mpithreads;
#endif
  STRUMPACK_SparseSolver       *S;
  STRUMPACK_INTERFACE           iface;
  STRUMPACK_REORDERING_STRATEGY ndcurrent, ndvalue;
  STRUMPACK_COMPRESSION_TYPE    compcurrent, compvalue;
  const STRUMPACK_PRECISION     table[2][2][2] = {
    {{STRUMPACK_FLOATCOMPLEX_64, STRUMPACK_DOUBLECOMPLEX_64}, {STRUMPACK_FLOAT_64, STRUMPACK_DOUBLE_64}},
    {{STRUMPACK_FLOATCOMPLEX, STRUMPACK_DOUBLECOMPLEX},       {STRUMPACK_FLOAT, STRUMPACK_DOUBLE}      }
  };
  const STRUMPACK_PRECISION prec               = table[(sizeof(PetscInt) == 8) ? 0 : 1][(PETSC_SCALAR == PETSC_COMPLEX) ? 0 : 1][(PETSC_REAL == PETSC_FLOAT) ? 0 : 1];
  const char *const         STRUMPACKNDTypes[] = {"NATURAL", "METIS", "PARMETIS", "SCOTCH", "PTSCOTCH", "RCM", "GEOMETRIC", "AMD", "MMD", "AND", "MLF", "SPECTRAL", "STRUMPACKNDTypes", "", 0};
  const char *const         CompTypes[]        = {"NONE", "HSS", "BLR", "HODLR", "BLR_HODLR", "ZFP_BLR_HODLR", "LOSSLESS", "LOSSY", "CompTypes", "", 0};

  PetscFunctionBegin;
#if defined(STRUMPACK_USE_SLATE_SCALAPACK)
  PetscCallMPI(MPI_Query_thread(&mpithreads));
  PetscCheck(mpithreads == MPI_THREAD_MULTIPLE, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP_SYS, "SLATE requires MPI_THREAD_MULTIPLE");
#endif
  /* Create the factorization matrix */
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
  PetscCall(MatSetSizes(B, A->rmap->n, A->cmap->n, M, N));
  PetscCall(PetscStrallocpy("strumpack", &((PetscObject)B)->type_name));
  PetscCall(MatSetUp(B));
  PetscCall(MatSeqAIJSetPreallocation(B, 0, NULL));
  PetscCall(MatMPIAIJSetPreallocation(B, 0, NULL, 0, NULL));
  B->trivialsymbolic = PETSC_TRUE;
  PetscCheck(ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_ILU, PETSC_COMM_SELF, PETSC_ERR_SUP, "Factor type not supported");
  B->ops->lufactorsymbolic  = MatLUFactorSymbolic_STRUMPACK;
  B->ops->ilufactorsymbolic = MatLUFactorSymbolic_STRUMPACK;
  B->ops->getinfo           = MatGetInfo_External;
  B->ops->view              = MatView_STRUMPACK;
  B->ops->destroy           = MatDestroy_STRUMPACK;
  B->ops->getdiagonal       = MatGetDiagonal_STRUMPACK;
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatFactorGetSolverType_C", MatFactorGetSolverType_aij_strumpack));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKSetReordering_C", MatSTRUMPACKSetReordering_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKGetReordering_C", MatSTRUMPACKGetReordering_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKSetColPerm_C", MatSTRUMPACKSetColPerm_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKGetColPerm_C", MatSTRUMPACKGetColPerm_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKSetGeometricNxyz_C", MatSTRUMPACKSetGeometricNxyz_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKSetGeometricComponents_C", MatSTRUMPACKSetGeometricComponents_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKSetGeometricWidth_C", MatSTRUMPACKSetGeometricWidth_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKSetGPU_C", MatSTRUMPACKSetGPU_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKGetGPU_C", MatSTRUMPACKGetGPU_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKSetCompression_C", MatSTRUMPACKSetCompression_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKGetCompression_C", MatSTRUMPACKGetCompression_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKSetCompRelTol_C", MatSTRUMPACKSetCompRelTol_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKGetCompRelTol_C", MatSTRUMPACKGetCompRelTol_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKSetCompAbsTol_C", MatSTRUMPACKSetCompAbsTol_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKGetCompAbsTol_C", MatSTRUMPACKGetCompAbsTol_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKSetCompLeafSize_C", MatSTRUMPACKSetCompLeafSize_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKGetCompLeafSize_C", MatSTRUMPACKGetCompLeafSize_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKSetCompMinSepSize_C", MatSTRUMPACKSetCompMinSepSize_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKGetCompMinSepSize_C", MatSTRUMPACKGetCompMinSepSize_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKSetCompLossyPrecision_C", MatSTRUMPACKSetCompLossyPrecision_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKGetCompLossyPrecision_C", MatSTRUMPACKGetCompLossyPrecision_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKSetCompButterflyLevels_C", MatSTRUMPACKSetCompButterflyLevels_STRUMPACK));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSTRUMPACKGetCompButterflyLevels_C", MatSTRUMPACKGetCompButterflyLevels_STRUMPACK));
  B->factortype = ftype;

  /* set solvertype */
  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERSTRUMPACK, &B->solvertype));

  PetscCall(PetscNew(&S));
  B->data = S;

  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A, MATSEQAIJ, &flg)); /* A might be MATSEQAIJCUSPARSE */
  iface = flg ? STRUMPACK_MT : STRUMPACK_MPI_DIST;

  PetscOptionsBegin(PetscObjectComm((PetscObject)B), ((PetscObject)B)->prefix, "STRUMPACK Options", "Mat");
  verb = PetscLogPrintInfo ? PETSC_TRUE : PETSC_FALSE;
  PetscCall(PetscOptionsBool("-mat_strumpack_verbose", "Print STRUMPACK information", "None", verb, &verb, NULL));

  PetscStackCallExternalVoid("STRUMPACK_init", STRUMPACK_init(S, PetscObjectComm((PetscObject)A), prec, iface, 0, NULL, verb));

  /* By default, no compression is done. Compression is enabled when the user enables it with        */
  /*  -mat_strumpack_compression with anything else than NONE, or when selecting ilu                 */
  /* preconditioning, in which case we default to STRUMPACK_BLR compression.                         */
  /* When compression is enabled, the STRUMPACK solver becomes an incomplete                         */
  /* (or approximate) LU factorization.                                                              */
  PetscStackCallExternalVoid("STRUMPACK_compression", compcurrent = STRUMPACK_compression(*S));
  PetscCall(PetscOptionsEnum("-mat_strumpack_compression", "Rank-structured compression type", "None", CompTypes, (PetscEnum)compcurrent, (PetscEnum *)&compvalue, &set));
  if (set) {
    PetscCall(MatSTRUMPACKSetCompression(B, (MatSTRUMPACKCompressionType)compvalue));
  } else {
    if (ftype == MAT_FACTOR_ILU) PetscStackCallExternalVoid("STRUMPACK_set_compression", STRUMPACK_set_compression(*S, STRUMPACK_BLR));
  }

  PetscStackCallExternalVoid("STRUMPACK_compression_rel_tol", ctol = (PetscReal)STRUMPACK_compression_rel_tol(*S));
  PetscCall(PetscOptionsReal("-mat_strumpack_compression_rel_tol", "Relative compression tolerance", "None", ctol, &ctol, &set));
  if (set) PetscStackCallExternalVoid("STRUMPACK_set_compression_rel_tol", STRUMPACK_set_compression_rel_tol(*S, (double)ctol));

  PetscStackCallExternalVoid("STRUMPACK_compression_abs_tol", ctol = (PetscReal)STRUMPACK_compression_abs_tol(*S));
  PetscCall(PetscOptionsReal("-mat_strumpack_compression_abs_tol", "Absolute compression tolerance", "None", ctol, &ctol, &set));
  if (set) PetscStackCallExternalVoid("STRUMPACK_set_compression_abs_tol", STRUMPACK_set_compression_abs_tol(*S, (double)ctol));

  PetscStackCallExternalVoid("STRUMPACK_compression_min_sep_size", min_sep_size = (PetscInt)STRUMPACK_compression_min_sep_size(*S));
  PetscCall(PetscOptionsInt("-mat_strumpack_compression_min_sep_size", "Minimum size of separator for compression", "None", min_sep_size, &min_sep_size, &set));
  if (set) PetscStackCallExternalVoid("STRUMPACK_set_compression_min_sep_size", STRUMPACK_set_compression_min_sep_size(*S, (int)min_sep_size));

  PetscStackCallExternalVoid("STRUMPACK_compression_leaf_size", leaf_size = (PetscInt)STRUMPACK_compression_leaf_size(*S));
  PetscCall(PetscOptionsInt("-mat_strumpack_compression_leaf_size", "Size of diagonal blocks in rank-structured approximation", "None", leaf_size, &leaf_size, &set));
  if (set) PetscStackCallExternalVoid("STRUMPACK_set_compression_leaf_size", STRUMPACK_set_compression_leaf_size(*S, (int)leaf_size));

#if defined(STRUMPACK_USE_ZFP)
  PetscStackCallExternalVoid("STRUMPACK_compression_lossy_precision", lossy_prec = (PetscInt)STRUMPACK_compression_lossy_precision(*S));
  PetscCall(PetscOptionsInt("-mat_strumpack_compression_lossy_precision", "Number of bitplanes to use in lossy compression", "None", lossy_prec, &lossy_prec, &set));
  if (set) PetscStackCallExternalVoid("STRUMPACK_set_compression_lossy_precision", STRUMPACK_set_compression_lossy_precision(*S, (int)lossy_prec));
#endif

#if defined(STRUMPACK_USE_BPACK)
  PetscStackCallExternalVoid("STRUMPACK_compression_butterfly_levels", bfly_lvls = (PetscInt)STRUMPACK_compression_butterfly_levels(*S));
  PetscCall(PetscOptionsInt("-mat_strumpack_compression_butterfly_levels", "Number of levels in the HODLR matrix for which to use butterfly compression", "None", bfly_lvls, &bfly_lvls, &set));
  if (set) PetscStackCallExternalVoid("STRUMPACK_set_compression_butterfly_levels", STRUMPACK_set_compression_butterfly_levels(*S, (int)bfly_lvls));
#endif

#if defined(STRUMPACK_USE_CUDA) || defined(STRUMPACK_USE_HIP) || defined(STRUMPACK_USE_SYCL)
  PetscStackCallExternalVoid("STRUMPACK_use_gpu", flg = (STRUMPACK_use_gpu(*S) == 0) ? PETSC_FALSE : PETSC_TRUE);
  PetscCall(PetscOptionsBool("-mat_strumpack_gpu", "Enable GPU acceleration (not supported for all compression types)", "None", flg, &flg, &set));
  if (set) MatSTRUMPACKSetGPU(B, flg);
#endif

  PetscStackCallExternalVoid("STRUMPACK_matching", flg = (STRUMPACK_matching(*S) == STRUMPACK_MATCHING_NONE) ? PETSC_FALSE : PETSC_TRUE);
  PetscCall(PetscOptionsBool("-mat_strumpack_colperm", "Find a col perm to get nonzero diagonal", "None", flg, &flg, &set));
  if (set) PetscStackCallExternalVoid("STRUMPACK_set_matching", STRUMPACK_set_matching(*S, flg ? STRUMPACK_MATCHING_MAX_DIAGONAL_PRODUCT_SCALING : STRUMPACK_MATCHING_NONE));

  PetscStackCallExternalVoid("STRUMPACK_reordering_method", ndcurrent = STRUMPACK_reordering_method(*S));
  PetscCall(PetscOptionsEnum("-mat_strumpack_reordering", "Sparsity reducing matrix reordering", "None", STRUMPACKNDTypes, (PetscEnum)ndcurrent, (PetscEnum *)&ndvalue, &set));
  if (set) PetscStackCallExternalVoid("STRUMPACK_set_reordering_method", STRUMPACK_set_reordering_method(*S, ndvalue));

  /* geometric ordering, for a regular 1D/2D/3D mesh in the natural ordering, */
  /* with nc DOF's per gridpoint, and possibly a wider stencil                */
  nrdims  = 3;
  nxyz[0] = nxyz[1] = nxyz[2] = 1;
  PetscCall(PetscOptionsIntArray("-mat_strumpack_geometric_xyz", "Mesh sizes nx,ny,nz (Use 1 for default)", "", nxyz, &nrdims, &set));
  if (set) {
    PetscCheck(nrdims == 1 || nrdims == 2 || nrdims == 3, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_OUTOFRANGE, "'-mat_strumpack_geometric_xyz' requires 1, 2, or 3 values.");
    PetscCall(MatSTRUMPACKSetGeometricNxyz(B, (int)nxyz[0], (int)nxyz[1], (int)nxyz[2]));
  }
  PetscCall(PetscOptionsInt("-mat_strumpack_geometric_components", "Number of components per mesh point, for geometric nested dissection ordering", "None", 1, &nc, &set));
  if (set) PetscStackCallExternalVoid("STRUMPACK_set_components", STRUMPACK_set_components(*S, (int)nc));
  PetscCall(PetscOptionsInt("-mat_strumpack_geometric_width", "Width of the separator (for instance a 1D 3-point wide stencil needs a 1 point wide separator, a 1D 5-point stencil needs a 2 point wide separator), for geometric nested dissection ordering", "None", 1, &w, &set));
  if (set) PetscStackCallExternalVoid("STRUMPACK_set_separator_width", STRUMPACK_set_separator_width(*S, (int)w));

  PetscStackCallExternalVoid("STRUMPACK_use_METIS_NodeNDP", flg = (STRUMPACK_use_METIS_NodeNDP(*S) == 0) ? PETSC_FALSE : PETSC_TRUE);
  PetscCall(PetscOptionsBool("-mat_strumpack_metis_nodeNDP", "Use METIS_NodeNDP instead of METIS_NodeND, for a more balanced tree", "None", flg, &flg, &set));
  if (set) {
    if (flg) {
      PetscStackCallExternalVoid("STRUMPACK_enable_METIS_NodeNDP", STRUMPACK_enable_METIS_NodeNDP(*S));
    } else {
      PetscStackCallExternalVoid("STRUMPACK_disable_METIS_NodeNDP", STRUMPACK_disable_METIS_NodeNDP(*S));
    }
  }

  /* Disable the outer iterative solver from STRUMPACK.                                       */
  /* When STRUMPACK is used as a direct solver, it will by default do iterative refinement.   */
  /* When STRUMPACK is used as an approximate factorization preconditioner (by enabling       */
  /* low-rank compression), it will use it's own preconditioned GMRES. Here we can disable    */
  /* the outer iterative solver, as PETSc uses STRUMPACK from within a KSP.                   */
  PetscStackCallExternalVoid("STRUMPACK_set_Krylov_solver", STRUMPACK_set_Krylov_solver(*S, STRUMPACK_DIRECT));

  PetscOptionsEnd();

  *F = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatSolverTypeRegister_STRUMPACK(void)
{
  PetscFunctionBegin;
  PetscCall(MatSolverTypeRegister(MATSOLVERSTRUMPACK, MATMPIAIJ, MAT_FACTOR_LU, MatGetFactor_aij_strumpack));
  PetscCall(MatSolverTypeRegister(MATSOLVERSTRUMPACK, MATSEQAIJ, MAT_FACTOR_LU, MatGetFactor_aij_strumpack));
  PetscCall(MatSolverTypeRegister(MATSOLVERSTRUMPACK, MATMPIAIJ, MAT_FACTOR_ILU, MatGetFactor_aij_strumpack));
  PetscCall(MatSolverTypeRegister(MATSOLVERSTRUMPACK, MATSEQAIJ, MAT_FACTOR_ILU, MatGetFactor_aij_strumpack));
  PetscFunctionReturn(PETSC_SUCCESS);
}
