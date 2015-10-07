#include <petscsys.h>
#if defined(PETSC_HAVE_P4EST)
#include <p4est_to_p8est.h>
#endif

static const PetscInt PetscFaceToP4estFace[6] = {4, 5, 2, 3, 1, 0};

static const PetscInt PetscOrntToP4estOrnt[6][8] =
{
  {0, 4, 7, 3, 1, 5, 6, 2},
  {1, 5, 6, 2, 0, 4, 7, 3},
  {1, 5, 6, 2, 0, 4, 7, 3},
  {4, 7, 3, 0, 2, 1, 5, 6},
  {0, 4, 7, 3, 1, 5, 6, 2},
  {1, 5, 6, 2, 0, 4, 7, 3}
};

static PetscInt PetscOrientToP4estOrient(PetscInt p4estFace, PetscInt petscOrient)
{
  return PetscOrntToP4estOrnt[p4estFace][4+petscOrient];
}

#define _append_pforest(a) a ## _p8est
#include "pforest.c"
