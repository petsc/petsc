#include <petscsys.h>

static const PetscInt PetscFaceToP4estFace[4] = {2, 1, 3, 0};
static const PetscInt P4estFaceToPetscFace[4] = {3, 1, 0, 2};
static const PetscInt P4estFaceToPetscOrnt[4] = {-2, 0, 0, -2};
static const PetscInt PetscVertToP4estVert[4] = {0, 1, 3, 2};
static const PetscInt P4estVertToPetscVert[4] = {0, 1, 3, 2};

#define DMPFOREST DMP4EST

#define _append_pforest(a)  a ## _p4est
#define _infix_pforest(a,b) a ## _p4est ## b
#include "pforest.c"
