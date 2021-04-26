#include <petsc/private/matimpl.h>
#include <htool/misc/petsc.hpp>

class WrapperHtool : public htool::IMatrix<PetscScalar> {
  PetscInt        dim;
  MatHtoolKernel& kernel;
  void*           ctx;

  public:
  WrapperHtool(PetscInt M,PetscInt N,PetscInt sdim,MatHtoolKernel& g,void* kernelctx) : IMatrix(M,N), dim(sdim), kernel(g), ctx(kernelctx) { }
  PetscScalar get_coef(const PetscInt& i,const PetscInt& j) const { return kernel(dim,i,j,ctx); }
};

struct Mat_Htool {
  PetscInt               dim;
  PetscReal              *gcoords_target;
  PetscReal              *gcoords_source;
  PetscScalar            *work_target;
  PetscScalar            *work_source;
  PetscScalar            s;
  PetscInt               bs[2];
  PetscReal              epsilon;
  PetscReal              eta;
  PetscInt               depth[2];
  MatHtoolCompressorType compressor;
  MatHtoolKernel         kernel;
  void*                  kernelctx;
  WrapperHtool           *wrapper;
  htool::HMatrixVirtual<PetscScalar> *hmatrix;
};

struct MatHtoolKernelTranspose {
  Mat            A;
  MatHtoolKernel kernel;
  void*          kernelctx;
};
