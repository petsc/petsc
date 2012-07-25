#ifndef _matelemimpl_h
#define _matelemimpl_h

#include <elemental.hpp>
#include <petsc-private/matimpl.h>

#if defined (PETSC_USE_COMPLEX)
typedef elem::Complex<PetscReal> PetscElemScalar;
#else
typedef PetscScalar PetscElemScalar;
#endif

typedef struct {
  PetscInt commsize;
  PetscInt m[2];       /* Number of entries in a local block of the row (column) space */
  PetscInt mr[2];      /* First incomplete/ragged rank of (row) column space.
                          We expose a blocked ordering to the user because that is what all other PETSc infrastructure uses.
                          With the blocked ordering when the number of processes do not evenly divide the vector size, 
                          we still need to be able to convert from PETSc/blocked ordering to VC/VR ordering. */
  elem::Grid *grid;
  elem::DistMatrix<PetscElemScalar> *emat;
  elem::Matrix<PetscElemScalar> *esubmat; /* Used for adding off-proc matrix entries */
  elem::AxpyInterface<PetscElemScalar> *interface;
  elem::DistMatrix<PetscInt,elem::VC,elem::STAR> *pivot; /* pivot vector representing the pivot matrix P in PA = LU */
} Mat_Elemental;

typedef struct {
  elem::Grid *grid;
  PetscInt   grid_refct;
} Mat_Elemental_Grid;

PETSC_STATIC_INLINE void P2RO(Mat A,PetscInt rc,PetscInt p,PetscInt *rank,PetscInt *offset) {
  Mat_Elemental *a = (Mat_Elemental*)A->data;
  PetscInt critical = a->m[rc]*a->mr[rc];
  if (p < critical) {
    *rank = p / a->m[rc];
    *offset = p % a->m[rc];
  } else {
    *rank = a->mr[rc] + (p - critical) / (a->m[rc] - 1);
    *offset = (p - critical) % (a->m[rc] - 1);
  }
}
PETSC_STATIC_INLINE void RO2P(Mat A,PetscInt rc,PetscInt rank,PetscInt offset,PetscInt *p) {
  Mat_Elemental *a = (Mat_Elemental*)A->data;
  if (rank < a->mr[rc]) {
    *p = rank*a->m[rc] + offset;
  } else {
    *p = a->mr[rc]*a->m[rc] + (rank - a->mr[rc])*(a->m[rc]-1) + offset;
  }
}

PETSC_STATIC_INLINE void E2RO(Mat A,PetscInt rc,PetscInt p,PetscInt *rank,PetscInt *offset) {
  Mat_Elemental *a = (Mat_Elemental*)A->data;
  *rank = p % a->commsize;
  *offset = p / a->commsize;
}
PETSC_STATIC_INLINE void RO2E(Mat A,PetscInt rc,PetscInt rank,PetscInt offset,PetscInt *e) {
  Mat_Elemental *a = (Mat_Elemental*)A->data;
  *e = offset * a->commsize + rank;
}

#endif
