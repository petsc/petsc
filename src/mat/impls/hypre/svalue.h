#if !defined(SVALUE_H)
#define SVALUE_H
#include <petscpkg_version.h>
#include <petsc/private/petschypre.h>
#include <petscmathypre.h>
#include <petsc/private/matimpl.h>
#include <petsc/private/deviceimpl.h>
#include <../src/mat/impls/hypre/mhypre.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/vec/vec/impls/hypre/vhyp.h>

PETSC_INTERN PetscErrorCode setdiagonal(Mat mat, PetscInt *diag);
PETSC_INTERN PetscErrorCode setCOOValue(Mat mat, const PetscScalar v[], InsertMode imode);
PETSC_INTERN PetscErrorCode MatSetPreallocationCOO_SeqAIJ_hypre(Mat mat, PetscCount coo_n, const PetscInt coo_i[], const PetscInt coo_j[]);

#endif /* SVALUE_H */
