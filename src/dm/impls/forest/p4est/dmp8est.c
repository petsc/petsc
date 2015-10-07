#include <petscsys.h>
#if defined(PETSC_HAVE_P4EST)
#include <p4est_to_p8est.h>
#endif

static const PetscInt PetscFaceToP4estFace[6] = {4, 5, 2, 3, 1, 0};
static const PetscInt P4estFaceToPetscOrnt[6] = {-4, 0, 0, -1, 4, 0};

#define _append_pforest(a) a ## _p8est
#include "pforest.c"
