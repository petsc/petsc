#if !defined(_SFGATHERV_H)
#define _SFGATHERV_H

#include <../src/vec/is/sf/impls/basic/allgatherv/sfallgatherv.h>

PETSC_INTERN PetscErrorCode PetscSFFetchAndOpBegin_Gatherv(PetscSF,MPI_Datatype,PetscMemType,void*,PetscMemType,const void*,void*,MPI_Op);
#endif
