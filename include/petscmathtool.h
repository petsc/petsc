#pragma once

#include <petscmat.h>

/* MANSEC = Mat */

namespace htool
{
template <class>
class DistributedOperator; /* forward definition of a single needed Htool class */
} // namespace htool

PETSC_EXTERN PetscErrorCode MatHtoolGetHierarchicalMat(Mat, const htool::DistributedOperator<PetscScalar> **);
