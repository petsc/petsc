#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petscmat.h>

/*@
  MatSTRUMPACKSetReordering - Set STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master> fill-reducing reordering

  Logically Collective

  Input Parameters:
+ F          - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface
- reordering - the code to be used to find the fill-reducing reordering

  Options Database Key:
. -mat_strumpack_reordering (natural|metis|parmetis|scotch|ptscotch|rcm|geometric|amd|mmd|and|mlf|spectral) - Sparsity reducing matrix reordering, see `MatSTRUMPACKReordering`

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

/*@
  MatSTRUMPACKSetColPerm - Set whether STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master>
  should try to permute the columns of the matrix in order to get a nonzero diagonal

  Logically Collective

  Input Parameters:
+ F     - the factored matrix obtained by calling `MatGetFactor()`
- cperm - `PETSC_TRUE` to permute (internally) the columns of the matrix

  Options Database Key:
. -mat_strumpack_colperm (true|false) - true to use the permutation

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

/*@
  MatSTRUMPACKSetGPU - Set whether STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master>
  should enable GPU acceleration (not supported for all compression types)

  Logically Collective

  Input Parameters:
+ F   - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface
- gpu - whether or not to use GPU acceleration

  Options Database Key:
. -mat_strumpack_gpu (true|false) - true to use GPU offload

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

/*@
  MatSTRUMPACKSetCompression - Set STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master> compression type

  Input Parameters:
+ F    - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface
- comp - Type of compression to be used in the approximate sparse factorization

  Options Database Key:
. -mat_strumpack_compression (none|hss|blr|hodlr|blr_hodlr|zfp_blr_hodlr|lossless|lossy) - Type of rank-structured compression in sparse LU factors

  Level: intermediate

  Note:
  Default for compression is `MAT_STRUMPACK_COMPRESSION_TYPE_NONE` for `-pc_type lu` and `MAT_STRUMPACK_COMPRESSION_TYPE_BLR`
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

/*@
  MatSTRUMPACKSetCompRelTol - Set STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master> relative tolerance for compression

  Logically Collective

  Input Parameters:
+ F    - the factored matrix obtained by calling `MatGetFactor()`
- rtol - relative compression tolerance

  Options Database Key:
. -mat_strumpack_compression_rel_tol rel_tol - Relative compression tolerance, when using `-pctype ilu`

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

/*@
  MatSTRUMPACKSetCompAbsTol - Set STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master> absolute tolerance for compression

  Logically Collective

  Input Parameters:
+ F    - the factored matrix obtained by calling `MatGetFactor()`
- atol - absolute compression tolerance

  Options Database Key:
. -mat_strumpack_compression_abs_tol abs_tol - Absolute compression tolerance, when using `-pctype ilu`

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

/*@
  MatSTRUMPACKSetCompLeafSize - Set STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master> leaf size for HSS, BLR, HODLR...

  Logically Collective

  Input Parameters:
+ F         - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface
- leaf_size - Size of diagonal blocks in rank-structured approximation

  Options Database Key:
. -mat_strumpack_compression_leaf_size size - Size of diagonal blocks in rank-structured approximation, when using `-pctype ilu`

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

/*@
  MatSTRUMPACKSetGeometricNxyz - Set STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master> mesh x, y and z dimensions, for use with
  the `MAT_STRUMPACK_GEOMETRIC` value of `MatSTRUMPACKReordering`

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

.seealso: `MATSOLVERSTRUMPACK`, `MatGetFactor()`, `MAT_STRUMPACK_GEOMETRIC`, `MatSTRUMPACKReordering`
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
  number of degrees of freedom per mesh point, for use with the `MAT_STRUMPACK_GEOMETRIC` value of `MatSTRUMPACKReordering`

  Logically Collective

  Input Parameters:
+ F  - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface
- nc - Number of components/dof's per grid point

  Options Database Key:
. -mat_strumpack_geometric_components dof - Number of components per mesh point, for geometric nested dissection ordering

  Level: intermediate

.seealso: `MATSOLVERSTRUMPACK`, `MatGetFactor()`, `MAT_STRUMPACK_GEOMETRIC`, `MatSTRUMPACKReordering`
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
  MatSTRUMPACKSetGeometricWidth - Set STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master> width of the separator, for use with
  the `MAT_STRUMPACK_GEOMETRIC` value of `MatSTRUMPACKReordering`

  Logically Collective

  Input Parameters:
+ F - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface
- w - width of the separator

  Options Database Key:
. -mat_strumpack_geometric_width width - Width of the separator of the mesh, for geometric nested dissection ordering

  Level: intermediate

.seealso: `MATSOLVERSTRUMPACK`, `MatGetFactor()`, `MAT_STRUMPACK_GEOMETRIC`, `MatSTRUMPACKReordering`
@*/
PetscErrorCode MatSTRUMPACKSetGeometricWidth(Mat F, PetscInt w)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveInt(F, w, 2);
  PetscTryMethod(F, "MatSTRUMPACKSetGeometricWidth_C", (Mat, PetscInt), (F, w));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSTRUMPACKSetCompMinSepSize - Set STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master> minimum separator size for low-rank approximation

  Logically Collective

  Input Parameters:
+ F            - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface
- min_sep_size - minimum dense matrix size for low-rank approximation

  Options Database Key:
. -mat_strumpack_compression_min_sep_size min_sep_size - Minimum size of dense sub-block for low-rank compression

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

/*@
  MatSTRUMPACKSetCompLossyPrecision - Set STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master> precision for lossy compression,
  that is `MAT_STRUMPACK_COMPRESSION_TYPE_LOSSY` (requires ZFP support)

  Logically Collective

  Input Parameters:
+ F          - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface
- lossy_prec - Number of bitplanes to use in lossy compression

  Options Database Key:
. -mat_strumpack_compression_lossy_precision lossy_prec - Precision when using lossy compression [1-64], when using `-pctype ilu -mat_strumpack_compression lossy`

  Level: intermediate

.seealso: `MATSOLVERSTRUMPACK`, `MatGetFactor()`, `MatSTRUMPACKGetCompLossyPrecision()`, `MAT_STRUMPACK_COMPRESSION_TYPE_LOSSY`
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
  MatSTRUMPACKGetCompLossyPrecision - Get STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master> precision for lossy compression,
  that is `MAT_STRUMPACK_COMPRESSION_TYPE_LOSSY`

  Logically Collective

  Input Parameters:
. F - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface

  Output Parameter:
. lossy_prec - Number of bitplanes to use in lossy compression

  Level: intermediate

.seealso: `MATSOLVERSTRUMPACK`, `MatGetFactor()`, `MatSTRUMPACKSetCompLossyPrecision()`, `MAT_STRUMPACK_COMPRESSION_TYPE_LOSSY`
@*/
PetscErrorCode MatSTRUMPACKGetCompLossyPrecision(Mat F, PetscInt *lossy_prec)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscTryMethod(F, "MatSTRUMPACKGetCompLossyPrecision_C", (Mat, PetscInt *), (F, lossy_prec));
  PetscValidLogicalCollectiveInt(F, *lossy_prec, 2);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatSTRUMPACKSetCompButterflyLevels - Set STRUMPACK <https://portal.nersc.gov/project/sparse/strumpack/master>
  number of butterfly levels in `MAT_STRUMPACK_COMPRESSION_TYPE_HODLR`, `MAT_STRUMPACK_COMPRESSION_TYPE_BLR_HODLR`, or
  `MAT_STRUMPACK_COMPRESSION_TYPE_ZFP_BLR_HODLR` compression (requires ButterflyPACK support)

  Logically Collective

  Input Parameters:
+ F         - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface
- bfly_lvls - Number of levels of butterfly compression in HODLR compression

  Options Database Key:
. -mat_strumpack_compression_butterfly_levels bfly_lvls - Number of levels in the hierarchically off-diagonal matrix for which to use butterfly

  Level: intermediate

.seealso: `MATSOLVERSTRUMPACK`, `MatGetFactor()`, `MatSTRUMPACKGetCompButterflyLevels()`, `MAT_STRUMPACK_COMPRESSION_TYPE_HODLR`,
          `MAT_STRUMPACK_COMPRESSION_TYPE_BLR_HODLR`, `MAT_STRUMPACK_COMPRESSION_TYPE_ZFP_BLR_HODLR`
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
  number of butterfly levels in `MAT_STRUMPACK_COMPRESSION_TYPE_HODLR`, `MAT_STRUMPACK_COMPRESSION_TYPE_BLR_HODLR`, or
  `MAT_STRUMPACK_COMPRESSION_TYPE_ZFP_BLR_HODLR` compression (requires ButterflyPACK support)

  Logically Collective

  Input Parameters:
. F - the factored matrix obtained by calling `MatGetFactor()` from PETSc-STRUMPACK interface

  Output Parameter:
. bfly_lvls - Number of levels of butterfly compression in compression

  Level: intermediate

.seealso: `MATSOLVERSTRUMPACK`, `MatGetFactor()`, `MatSTRUMPACKSetCompButterflyLevels()`, `MAT_STRUMPACK_COMPRESSION_TYPE_HODLR`,
          `MAT_STRUMPACK_COMPRESSION_TYPE_BLR_HODLR`, `MAT_STRUMPACK_COMPRESSION_TYPE_ZFP_BLR_HODLR`
@*/
PetscErrorCode MatSTRUMPACKGetCompButterflyLevels(Mat F, PetscInt *bfly_lvls)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(F, MAT_CLASSID, 1);
  PetscTryMethod(F, "MatSTRUMPACKGetButterflyLevels_C", (Mat, PetscInt *), (F, bfly_lvls));
  PetscValidLogicalCollectiveInt(F, *bfly_lvls, 2);
  PetscFunctionReturn(PETSC_SUCCESS);
}
