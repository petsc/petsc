#pragma once

#include <petsc/private/matimpl.h>
#include <petscmathtool.h>

PETSC_PRAGMA_DIAGNOSTIC_IGNORED_BEGIN("-Wsign-compare")
#include <htool/clustering/tree_builder/tree_builder.hpp>
#include <htool/hmatrix/lrmat/SVD.hpp>
#include <htool/hmatrix/lrmat/fullACA.hpp>
#include <htool/hmatrix/lrmat/recompressed_low_rank_generator.hpp>
#include <htool/hmatrix/hmatrix_distributed_output.hpp>
#include <htool/hmatrix/linalg/factorization.hpp>
#include <htool/distributed_operator/utility.hpp>
#include <htool/distributed_operator/linalg/add_distributed_operator_vector_product_local_to_local.hpp>
#include <htool/distributed_operator/linalg/add_distributed_operator_matrix_product_local_to_local.hpp>
PETSC_PRAGMA_DIAGNOSTIC_IGNORED_END()

class WrapperHtool : public htool::VirtualGenerator<PetscScalar> {
  MatHtoolKernelFn *&kernel;
  PetscInt           sdim;
  void              *ctx;

public:
  WrapperHtool(PetscInt dim, MatHtoolKernelFn *&g, void *kernelctx) : VirtualGenerator<PetscScalar>(), kernel(g), sdim(dim), ctx(kernelctx) { }
  void copy_submatrix(PetscInt M, PetscInt N, const PetscInt *rows, const PetscInt *cols, PetscScalar *ptr) const
  {
#if !PetscDefined(HAVE_OPENMP)
    PetscFunctionBegin;
#endif
    PetscCallAbort(PETSC_COMM_SELF, kernel(sdim, M, N, rows, cols, ptr, ctx));
#if !PetscDefined(HAVE_OPENMP)
    PetscFunctionReturnVoid();
#endif
  }
};

struct Mat_Htool {
  PetscInt                                                         dim;
  PetscReal                                                       *gcoords_target;
  PetscReal                                                       *gcoords_source;
  PetscInt                                                         max_cluster_leaf_size;
  PetscReal                                                        epsilon;
  PetscReal                                                        eta;
  PetscInt                                                         depth[2];
  PetscBool                                                        block_tree_consistency;
  PetscBool                                                        permutation;
  PetscBool                                                        recompression;
  MatHtoolCompressorType                                           compressor;
  MatHtoolClusteringType                                           clustering;
  MatHtoolKernelFn                                                *kernel;
  void                                                            *kernelctx;
  WrapperHtool                                                    *wrapper;
  std::unique_ptr<htool::Cluster<PetscReal>>                       target_cluster;
  std::unique_ptr<htool::Cluster<PetscReal>>                       source_cluster;
  std::unique_ptr<htool::DefaultApproximationBuilder<PetscScalar>> distributed_operator_holder;
};

struct MatHtoolKernelTranspose {
  Mat               A;
  MatHtoolKernelFn *kernel;
  void             *kernelctx;
};
