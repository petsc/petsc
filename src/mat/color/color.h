
#if !defined(_MINPACK_COLOR_H)
#define _MINPACK_COLOR_H
#include <petscmat.h>

/*
     Prototypes for Minpack coloring routines 
*/
extern PetscErrorCode MINPACKdegr(PetscInt*,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *);
extern PetscErrorCode MINPACKdsm(PetscInt*,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *);
extern PetscErrorCode MINPACKido(PetscInt*,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *);
extern PetscErrorCode MINPACKnumsrt(PetscInt*,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *);
extern PetscErrorCode MINPACKseq(PetscInt*,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *);
extern PetscErrorCode MINPACKsetr(PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
extern PetscErrorCode MINPACKslo(PetscInt*,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *,PetscInt *);

#endif
