
#if !defined(_MINPACK_COLOR_H)
#define _MINPACK_COLOR_H

/*
     Prototypes for Minpack coloring routines 
*/
EXTERN PetscErrorCode MINPACKdegr(int*,int *,int *,int *,int *,int *,int *);
EXTERN PetscErrorCode MINPACKdsm(int*,int *,int *,int *,int *,int *,int *,int *,int *,int *,int *,int *,int *);
EXTERN PetscErrorCode MINPACKido(int*,int *,int *,int *,int *,int *,int *,int *,int *,int *,int *,int *,int *);
EXTERN PetscErrorCode MINPACKnumsrt(int*,int *,int *,int *,int *,int *,int *);
EXTERN PetscErrorCode MINPACKseq(int*,int *,int *,int *,int *,int *,int *,int *,int *);
EXTERN PetscErrorCode MINPACKsetr(int*,int*,int*,int*,int*,int*,int*);
EXTERN PetscErrorCode MINPACKslo(int*,int *,int *,int *,int *,int *,int *,int *,int *,int *,int *,int *);

#endif
