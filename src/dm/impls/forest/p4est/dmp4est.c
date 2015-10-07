#include <petscsys.h>

static const PetscInt PetscFaceToP4estFace[4] = {2, 1, 3, 0};
static const PetscInt PetscOrntToP4estOrnt[4][4] = {{0, 1, 1, 0},{1, 0, 0, 0},{1, 0, 0, 0},{0, 1, 1, 0}};

static PetscInt PetscOrientToP4estOrient(PetscInt p4estFace, PetscInt petscOrient)
{
  return PetscOrntToP4estOrnt[p4estFace][2+petscOrient];
}

#define _append_pforest(a) a ## _p4est
#include "pforest.c"
