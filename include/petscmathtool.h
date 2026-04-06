#pragma once

#include <petscmat.h>

/* MANSEC = Mat */

PETSC_EXTERN PetscErrorCode MatHtoolGetHierarchicalMat(Mat, void *);
