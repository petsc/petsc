#include <petscsys.h>

static const PetscInt PetscFaceToP4estFace[4] = {2, 1, 3, 0};
static const PetscInt P4estFaceToPetscOrnt[4] = {-2, 0, 0, -2};

#define _append_pforest(a) a ## _p4est
#include "pforest.c"
