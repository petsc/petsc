
#if !defined(_MINPACK_COLOR_H)
#define _MINPACK_COLOR_H
#include "petscmat.h"

/*
     Prototypes for Minpack coloring routines 
*/
EXTERN PetscErrorCode MINPACKdegr(PetscInt*,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *);
EXTERN PetscErrorCode MINPACKdsm(PetscInt*,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *);
EXTERN PetscErrorCode MINPACKido(PetscInt*,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *);
EXTERN PetscErrorCode MINPACKnumsrt(PetscInt*,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *);
EXTERN PetscErrorCode MINPACKseq(PetscInt*,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *);
EXTERN PetscErrorCode MINPACKsetr(PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
EXTERN PetscErrorCode MINPACKslo(PetscInt*,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *);

#endif
