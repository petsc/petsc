#include <petscmat.h>
/*
   Defines the interface to the SparsePack routines, translated into C.
*/
extern PetscErrorCode SPARSEPACKgen1wd(const PetscInt*,const PetscInt*,const PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
extern PetscErrorCode SPARSEPACKgennd(const PetscInt*,const PetscInt*,const PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);
extern PetscErrorCode SPARSEPACKgenrcm(const PetscInt*,const PetscInt*,const PetscInt*,PetscInt*,PetscInt*,PetscInt*);
extern PetscErrorCode SPARSEPACKgenqmd(const PetscInt*,const PetscInt*,const PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*,PetscInt*);


