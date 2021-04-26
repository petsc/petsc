#if !defined(PETSCMATHTOOL_H)
#define PETSCMATHTOOL_H

#include <petscmat.h>

namespace htool {
  template<class> class HMatrixVirtual; /* forward definition of a single needed Htool class */
}

PETSC_EXTERN PetscErrorCode MatHtoolGetHierarchicalMat(Mat,const htool::HMatrixVirtual<PetscScalar>**);

#endif /* PETSCMATHTOOL_H */
