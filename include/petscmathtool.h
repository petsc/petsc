#ifndef PETSCMATHTOOL_H
#define PETSCMATHTOOL_H

#include <petscmat.h>

namespace htool
{
template <class>
class VirtualHMatrix; /* forward definition of a single needed Htool class */
} // namespace htool

PETSC_EXTERN PetscErrorCode MatHtoolGetHierarchicalMat(Mat, const htool::VirtualHMatrix<PetscScalar> **);

#endif /* PETSCMATHTOOL_H */
