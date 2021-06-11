#if !defined(PETSCMATHTOOL_H)
#define PETSCMATHTOOL_H

#include <petscmat.h>

namespace htool {
  template<class> class VirtualHMatrix; /* forward definition of a single needed Htool class */
}

PETSC_EXTERN PetscErrorCode MatHtoolGetHierarchicalMat(Mat,const htool::VirtualHMatrix<PetscScalar>**);

#endif /* PETSCMATHTOOL_H */
