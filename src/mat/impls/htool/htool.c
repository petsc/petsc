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

/*@
  MatHtoolGetEpsilon - Gets the relative error tolerance in Frobenius norm used by a `MATHTOOL` matrix.

  Not Collective

  Input Parameter:
. A - hierarchical matrix

  Output Parameter:
. epsilon - relative error tolerance

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`, `MatHtoolSetEpsilon()`
@*/
PetscErrorCode MatHtoolGetEpsilon(Mat A, PetscReal *epsilon)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(epsilon, 2);
  PetscUseMethod(A, "MatHtoolGetEpsilon_C", (Mat, PetscReal *), (A, epsilon));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatHtoolSetEpsilon - Sets the relative error tolerance in Frobenius norm used by a `MATHTOOL` matrix.

  Logically Collective

  Input Parameters:
+ A       - hierarchical matrix
- epsilon - relative error tolerance

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`, `MatHtoolGetEpsilon()`
@*/
PetscErrorCode MatHtoolSetEpsilon(Mat A, PetscReal epsilon)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveReal(A, epsilon, 2);
  PetscTryMethod(A, "MatHtoolSetEpsilon_C", (Mat, PetscReal), (A, epsilon));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatHtoolGetEta - Gets the admissibility condition parameter used by a `MATHTOOL` matrix.

  Not Collective

  Input Parameter:
. A - hierarchical matrix

  Output Parameter:
. eta - admissibility condition parameter

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`, `MatHtoolSetEta()`
@*/
PetscErrorCode MatHtoolGetEta(Mat A, PetscReal *eta)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(eta, 2);
  PetscUseMethod(A, "MatHtoolGetEta_C", (Mat, PetscReal *), (A, eta));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatHtoolSetEta - Sets the admissibility condition parameter used by a `MATHTOOL` matrix.

  Logically Collective

  Input Parameters:
+ A   - hierarchical matrix
- eta - admissibility condition parameter

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`, `MatHtoolGetEta()`
@*/
PetscErrorCode MatHtoolSetEta(Mat A, PetscReal eta)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveReal(A, eta, 2);
  PetscTryMethod(A, "MatHtoolSetEta_C", (Mat, PetscReal), (A, eta));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatHtoolGetMaxClusterLeafSize - Gets the maximum size of a leaf in the cluster tree used by a `MATHTOOL` matrix.

  Not Collective

  Input Parameter:
. A - hierarchical matrix

  Output Parameter:
. size - maximum leaf size

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`, `MatHtoolSetMaxClusterLeafSize()`
@*/
PetscErrorCode MatHtoolGetMaxClusterLeafSize(Mat A, PetscInt *size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(size, 2);
  PetscUseMethod(A, "MatHtoolGetMaxClusterLeafSize_C", (Mat, PetscInt *), (A, size));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatHtoolSetMaxClusterLeafSize - Sets the maximum size of a leaf in the cluster tree used by a `MATHTOOL` matrix.

  Logically Collective

  Input Parameters:
+ A    - hierarchical matrix
- size - maximum leaf size

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`, `MatHtoolGetMaxClusterLeafSize()`
@*/
PetscErrorCode MatHtoolSetMaxClusterLeafSize(Mat A, PetscInt size)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveInt(A, size, 2);
  PetscTryMethod(A, "MatHtoolSetMaxClusterLeafSize_C", (Mat, PetscInt), (A, size));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatHtoolGetMinTargetDepth - Gets the minimum depth of the target cluster tree used by a `MATHTOOL` matrix.

  Not Collective

  Input Parameter:
. A - hierarchical matrix

  Output Parameter:
. depth - minimum depth of the target cluster tree

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`, `MatHtoolSetMinTargetDepth()`, `MatHtoolGetMinSourceDepth()`
@*/
PetscErrorCode MatHtoolGetMinTargetDepth(Mat A, PetscInt *depth)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(depth, 2);
  PetscUseMethod(A, "MatHtoolGetMinTargetDepth_C", (Mat, PetscInt *), (A, depth));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatHtoolSetMinTargetDepth - Sets the minimum depth of the target cluster tree used by a `MATHTOOL` matrix.

  Logically Collective

  Input Parameters:
+ A     - hierarchical matrix
- depth - minimum depth of the target cluster tree

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`, `MatHtoolGetMinTargetDepth()`, `MatHtoolSetMinSourceDepth()`
@*/
PetscErrorCode MatHtoolSetMinTargetDepth(Mat A, PetscInt depth)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveInt(A, depth, 2);
  PetscTryMethod(A, "MatHtoolSetMinTargetDepth_C", (Mat, PetscInt), (A, depth));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatHtoolGetMinSourceDepth - Gets the minimum depth of the source cluster tree used by a `MATHTOOL` matrix.

  Not Collective

  Input Parameter:
. A - hierarchical matrix

  Output Parameter:
. depth - minimum depth of the source cluster tree

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`, `MatHtoolSetMinSourceDepth()`, `MatHtoolGetMinTargetDepth()`
@*/
PetscErrorCode MatHtoolGetMinSourceDepth(Mat A, PetscInt *depth)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(depth, 2);
  PetscUseMethod(A, "MatHtoolGetMinSourceDepth_C", (Mat, PetscInt *), (A, depth));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatHtoolSetMinSourceDepth - Sets the minimum depth of the source cluster tree used by a `MATHTOOL` matrix.

  Logically Collective

  Input Parameters:
+ A     - hierarchical matrix
- depth - minimum depth of the source cluster tree

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`, `MatHtoolGetMinSourceDepth()`, `MatHtoolSetMinTargetDepth()`
@*/
PetscErrorCode MatHtoolSetMinSourceDepth(Mat A, PetscInt depth)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveInt(A, depth, 2);
  PetscTryMethod(A, "MatHtoolSetMinSourceDepth_C", (Mat, PetscInt), (A, depth));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatHtoolGetBlockTreeConsistency - Gets whether a `MATHTOOL` matrix enforces block tree consistency.

  Not Collective

  Input Parameter:
. A - hierarchical matrix

  Output Parameter:
. block_tree_consistency - whether block tree consistency is enforced

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`, `MatHtoolSetBlockTreeConsistency()`
@*/
PetscErrorCode MatHtoolGetBlockTreeConsistency(Mat A, PetscBool *block_tree_consistency)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(block_tree_consistency, 2);
  PetscUseMethod(A, "MatHtoolGetBlockTreeConsistency_C", (Mat, PetscBool *), (A, block_tree_consistency));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatHtoolSetBlockTreeConsistency - Sets whether a `MATHTOOL` matrix should enforce block tree consistency.

  Logically Collective

  Input Parameters:
+ A                      - hierarchical matrix
- block_tree_consistency - whether to enforce block tree consistency

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`, `MatHtoolGetBlockTreeConsistency()`
@*/
PetscErrorCode MatHtoolSetBlockTreeConsistency(Mat A, PetscBool block_tree_consistency)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveBool(A, block_tree_consistency, 2);
  PetscTryMethod(A, "MatHtoolSetBlockTreeConsistency_C", (Mat, PetscBool), (A, block_tree_consistency));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatHtoolGetCompressorType - Gets the type of compressor used by a `MATHTOOL` matrix.

  Not Collective

  Input Parameter:
. A - hierarchical matrix

  Output Parameter:
. compressor - type of compressor

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`, `MatHtoolCompressorType`, `MatHtoolSetCompressorType()`
@*/
PetscErrorCode MatHtoolGetCompressorType(Mat A, MatHtoolCompressorType *compressor)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(compressor, 2);
  PetscUseMethod(A, "MatHtoolGetCompressorType_C", (Mat, MatHtoolCompressorType *), (A, compressor));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatHtoolSetCompressorType - Sets the type of compressor used by a `MATHTOOL` matrix.

  Logically Collective

  Input Parameters:
+ A          - hierarchical matrix
- compressor - type of compressor

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`, `MatHtoolCompressorType`, `MatHtoolGetCompressorType()`
@*/
PetscErrorCode MatHtoolSetCompressorType(Mat A, MatHtoolCompressorType compressor)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(A, compressor, 2);
  PetscTryMethod(A, "MatHtoolSetCompressorType_C", (Mat, MatHtoolCompressorType), (A, compressor));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatHtoolGetClusteringType - Gets the type of clustering used by a `MATHTOOL` matrix.

  Not Collective

  Input Parameter:
. A - hierarchical matrix

  Output Parameter:
. clustering - type of clustering

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`, `MatHtoolClusteringType`, `MatHtoolSetClusteringType()`
@*/
PetscErrorCode MatHtoolGetClusteringType(Mat A, MatHtoolClusteringType *clustering)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(clustering, 2);
  PetscUseMethod(A, "MatHtoolGetClusteringType_C", (Mat, MatHtoolClusteringType *), (A, clustering));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatHtoolSetClusteringType - Sets the type of clustering used by a `MATHTOOL` matrix.

  Logically Collective

  Input Parameters:
+ A          - hierarchical matrix
- clustering - type of clustering

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`, `MatHtoolClusteringType`, `MatHtoolGetClusteringType()`
@*/
PetscErrorCode MatHtoolSetClusteringType(Mat A, MatHtoolClusteringType clustering)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveEnum(A, clustering, 2);
  PetscTryMethod(A, "MatHtoolSetClusteringType_C", (Mat, MatHtoolClusteringType), (A, clustering));
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

.seealso: [](ch_matrices), `Mat`, `MatCreate()`, `MATHTOOL`, `PCSetCoordinates()`, `MatHtoolSetKernel()`, `MatHtoolCompressorType`, `MatHtoolClusteringType`, `MatHtoolGetEpsilon()`, `MatHtoolSetEpsilon()`, `MatHtoolGetEta()`, `MatHtoolSetEta()`, `MatHtoolGetMaxClusterLeafSize()`, `MatHtoolSetMaxClusterLeafSize()`, `MatHtoolGetMinTargetDepth()`, `MatHtoolSetMinTargetDepth()`, `MatHtoolGetMinSourceDepth()`, `MatHtoolSetMinSourceDepth()`, `MatHtoolGetBlockTreeConsistency()`, `MatHtoolSetBlockTreeConsistency()`, `MatHtoolGetCompressorType()`, `MatHtoolSetCompressorType()`, `MatHtoolGetClusteringType()`, `MatHtoolSetClusteringType()`, `MATH2OPUS`, `MatCreateH2OpusFromKernel()`
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
