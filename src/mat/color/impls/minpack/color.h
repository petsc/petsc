
#ifndef _MINPACK_COLOR_H
#define _MINPACK_COLOR_H
#include <petscmat.h>

/*
     Prototypes for Minpack coloring routines
*/
extern PetscErrorCode MINPACKdegr(PetscInt *, const PetscInt *, const PetscInt *, const PetscInt *, const PetscInt *, PetscInt *, PetscInt *);
extern PetscErrorCode MINPACKdsm(PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *);
extern PetscErrorCode MINPACKido(PetscInt *, PetscInt *, const PetscInt *, const PetscInt *, const PetscInt *, const PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *);
extern PetscErrorCode MINPACKnumsrt(PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *);
extern PetscErrorCode MINPACKseq(PetscInt *, const PetscInt *, const PetscInt *, const PetscInt *, const PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *);
extern PetscErrorCode MINPACKsetr(PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *);
extern PetscErrorCode MINPACKslo(PetscInt *, const PetscInt *, const PetscInt *, const PetscInt *, const PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *);

#endif
