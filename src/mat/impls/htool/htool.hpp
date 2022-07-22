#include <petsc/private/matimpl.h>
#include <htool/misc/petsc.hpp>

class WrapperHtool : public htool::VirtualGenerator<PetscScalar> {
  PetscInt        dim;
  MatHtoolKernel& kernel;
  void*           ctx;

  public:
  WrapperHtool(PetscInt M,PetscInt N,PetscInt sdim,MatHtoolKernel& g,void* kernelctx) : VirtualGenerator(M,N), dim(sdim), kernel(g), ctx(kernelctx) { }
  void copy_submatrix(PetscInt M, PetscInt N, const PetscInt *rows, const PetscInt *cols, PetscScalar *ptr) const {
#if !PetscDefined(HAVE_OPENMP)
    PetscFunctionBegin;
#endif
    PetscCallAbort(PETSC_COMM_SELF,kernel(dim,M,N,rows,cols,ptr,ctx));
#if !PetscDefined(HAVE_OPENMP)
    PetscFunctionReturnVoid();
#endif
  }
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
  MatHtoolClusteringType clustering;
  MatHtoolKernel         kernel;
  void*                  kernelctx;
  WrapperHtool           *wrapper;
  htool::VirtualHMatrix<PetscScalar> *hmatrix;
};

struct MatHtoolKernelTranspose {
  Mat            A;
  MatHtoolKernel kernel;
  void*          kernelctx;
};
