#pragma once

#include <petsc/private/matimpl.h>
#include <petscmathtool.h>

PETSC_PRAGMA_DIAGNOSTIC_IGNORED_BEGIN("-Wsign-compare")
#include <htool/clustering/clustering.hpp>
#include <htool/hmatrix/lrmat/SVD.hpp>
#include <htool/hmatrix/lrmat/fullACA.hpp>
#include <htool/hmatrix/hmatrix_distributed_output.hpp>
#include <htool/hmatrix/linalg/factorization.hpp>
#include <htool/distributed_operator/utility.hpp>
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
  PetscInt                                                            dim;
  PetscReal                                                          *gcoords_target;
  PetscReal                                                          *gcoords_source;
  PetscScalar                                                        *work_target;
  PetscScalar                                                        *work_source;
  PetscInt                                                            min_cluster_size;
  PetscReal                                                           epsilon;
  PetscReal                                                           eta;
  PetscInt                                                            depth[2];
  PetscBool                                                           block_tree_consistency;
  MatHtoolCompressorType                                              compressor;
  MatHtoolClusteringType                                              clustering;
  MatHtoolKernelFn                                                   *kernel;
  void                                                               *kernelctx;
  WrapperHtool                                                       *wrapper;
  std::unique_ptr<htool::Cluster<PetscReal>>                          target_cluster;
  std::unique_ptr<htool::Cluster<PetscReal>>                          source_cluster;
  std::unique_ptr<htool::DistributedOperatorFromHMatrix<PetscScalar>> distributed_operator_holder;
};

struct MatHtoolKernelTranspose {
  Mat               A;
  MatHtoolKernelFn *kernel;
  void             *kernelctx;
};
