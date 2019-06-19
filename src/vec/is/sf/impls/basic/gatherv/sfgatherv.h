#if !defined(_SFGATHERV_H)
#define _SFGATHERV_H

#include <../src/vec/is/sf/impls/basic/allgatherv/sfallgatherv.h>

typedef PetscSFPack_Allgatherv PetscSFPack_Gatherv;
#define PetscSFPackGet_Gatherv PetscSFPackGet_Allgatherv

/* Reuse the type. The difference is some fields (displs, recvcounts) are only significant
   on rank 0 in Gatherv. On other ranks they are harmless NULL.
 */
typedef PetscSF_Allgatherv PetscSF_Gatherv;

PETSC_INTERN PetscErrorCode PetscSFFetchAndOpBegin_Gatherv(PetscSF,MPI_Datatype,void*,const void*,void*,MPI_Op);
#endif
