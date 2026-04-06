#include <petsc/private/petscimpl.h>
#include <petscmathtool.h>

/*@C
  MatHtoolGetHierarchicalMat - Retrieves the opaque pointer to a Htool virtual matrix stored in a `MATHTOOL`.

  Not Collective; No Fortran Support

  Input Parameter:
. A - hierarchical matrix

  Output Parameter:
. distributed_operator - opaque pointer to a Htool virtual matrix

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`
@*/
PETSC_EXTERN PetscErrorCode MatHtoolGetHierarchicalMat(Mat A, void *distributed_operator)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(distributed_operator, 2);
  PetscUseMethod(A, "MatHtoolGetHierarchicalMat_C", (Mat, void *), (A, distributed_operator));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatHtoolSetKernel - Sets the kernel and context used for the assembly of a `MATHTOOL`.

  Collective; No Fortran Support

  Input Parameters:
+ A         - hierarchical matrix
. kernel    - computational kernel (or `NULL`)
- kernelctx - kernel context (if kernel is `NULL`, the pointer must be of type `htool::VirtualGenerator<PetscScalar> *`)

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`, `MatCreateHtoolFromKernel()`
@*/
PetscErrorCode MatHtoolSetKernel(Mat A, MatHtoolKernelFn *kernel, void *kernelctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  if (!kernelctx) PetscValidFunction(kernel, 2);
  if (!kernel) PetscAssertPointer(kernelctx, 3);
  PetscTryMethod(A, "MatHtoolSetKernel_C", (Mat, MatHtoolKernelFn *, void *), (A, kernel, kernelctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatHtoolGetPermutationSource - Gets the permutation associated to the source cluster for a `MATHTOOL` matrix.

  Not Collective

  Input Parameter:
. A - hierarchical matrix

  Output Parameter:
. is - permutation

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`, `MatHtoolGetPermutationTarget()`, `MatHtoolUsePermutation()`
@*/
PetscErrorCode MatHtoolGetPermutationSource(Mat A, IS *is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(is, 2);
  PetscUseMethod(A, "MatHtoolGetPermutationSource_C", (Mat, IS *), (A, is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatHtoolGetPermutationTarget - Gets the permutation associated to the target cluster for a `MATHTOOL` matrix.

  Not Collective

  Input Parameter:
. A - hierarchical matrix

  Output Parameter:
. is - permutation

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`, `MatHtoolGetPermutationSource()`, `MatHtoolUsePermutation()`
@*/
PetscErrorCode MatHtoolGetPermutationTarget(Mat A, IS *is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(is, 2);
  PetscUseMethod(A, "MatHtoolGetPermutationTarget_C", (Mat, IS *), (A, is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatHtoolUsePermutation - Sets whether a `MATHTOOL` matrix should permute input (resp. output) vectors following its internal source (resp. target) permutation.

  Logically Collective

  Input Parameters:
+ A   - hierarchical matrix
- use - Boolean value

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`, `MatHtoolGetPermutationSource()`, `MatHtoolGetPermutationTarget()`
@*/
PetscErrorCode MatHtoolUsePermutation(Mat A, PetscBool use)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveBool(A, use, 2);
  PetscTryMethod(A, "MatHtoolUsePermutation_C", (Mat, PetscBool), (A, use));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatHtoolUseRecompression - Sets whether a `MATHTOOL` matrix should use recompression.

  Logically Collective

  Input Parameters:
+ A   - hierarchical matrix
- use - Boolean value

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`
@*/
PetscErrorCode MatHtoolUseRecompression(Mat A, PetscBool use)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveBool(A, use, 2);
  PetscTryMethod(A, "MatHtoolUseRecompression_C", (Mat, PetscBool), (A, use));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatCreateHtoolFromKernel - Creates a `MATHTOOL` from a user-supplied kernel.

  Collective; No Fortran Support

  Input Parameters:
+ comm          - MPI communicator
. m             - number of local rows (or `PETSC_DECIDE` to have calculated if `M` is given)
. n             - number of local columns (or `PETSC_DECIDE` to have calculated if `N` is given)
. M             - number of global rows (or `PETSC_DETERMINE` to have calculated if `m` is given)
. N             - number of global columns (or `PETSC_DETERMINE` to have calculated if `n` is given)
. spacedim      - dimension of the space coordinates
. coords_target - coordinates of the target
. coords_source - coordinates of the source
. kernel        - computational kernel (or `NULL`)
- kernelctx     - kernel context (if kernel is `NULL`, the pointer must be of type htool::VirtualGenerator<PetscScalar>*)

  Output Parameter:
. B - matrix

  Options Database Keys:
+ -mat_htool_max_cluster_leaf_size <`PetscInt`>                                                - maximal leaf size in cluster tree
. -mat_htool_epsilon <`PetscReal`>                                                             - relative error in Frobenius norm when approximating a block
. -mat_htool_eta <`PetscReal`>                                                                 - admissibility condition tolerance
. -mat_htool_min_target_depth <`PetscInt`>                                                     - minimal cluster tree depth associated with the rows
. -mat_htool_min_source_depth <`PetscInt`>                                                     - minimal cluster tree depth associated with the columns
. -mat_htool_block_tree_consistency <`PetscBool`>                                              - block tree consistency
. -mat_htool_recompression <`PetscBool`>                                                       - use recompression
. -mat_htool_compressor <sympartialACA, fullACA, SVD>                                          - type of compression
- -mat_htool_clustering <PCARegular, PCAGeometric, BounbingBox1Regular, BoundingBox1Geometric> - type of clustering

  Level: intermediate

.seealso: [](ch_matrices), `Mat`, `MatCreate()`, `MATHTOOL`, `PCSetCoordinates()`, `MatHtoolSetKernel()`, `MatHtoolCompressorType`, `MatHtoolClusteringType`, `MATH2OPUS`, `MatCreateH2OpusFromKernel()`
@*/
PetscErrorCode MatCreateHtoolFromKernel(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt spacedim, const PetscReal coords_target[], const PetscReal coords_source[], MatHtoolKernelFn *kernel, void *kernelctx, Mat *B)
{
  Mat A;

  PetscFunctionBegin;
  PetscCall(MatCreate(comm, &A));
  PetscValidLogicalCollectiveInt(A, spacedim, 6);
  PetscAssertPointer(coords_target, 7);
  PetscAssertPointer(coords_source, 8);
  if (!kernelctx) PetscValidFunction(kernel, 9);
  if (!kernel) PetscAssertPointer(kernelctx, 10);
  PetscCall(MatSetSizes(A, m, n, M, N));
  PetscCall(MatSetType(A, MATHTOOL));
  PetscCall(MatSetUp(A));
  PetscUseMethod(A, "MatHtoolCreateFromKernel_C", (Mat, PetscInt, const PetscReal[], const PetscReal[], MatHtoolKernelFn *, void *), (A, spacedim, coords_target, coords_source, kernel, kernelctx));
  *B = A;
  PetscFunctionReturn(PETSC_SUCCESS);
}
