#if !defined(_SFALLGATHERV_H)
#define _SFALLGATHERV_H

#include <petsc/private/sfimpl.h> /*I "petscsf.h" I*/
#include <../src/vec/is/sf/impls/basic/sfpack.h>
#include <../src/vec/is/sf/impls/basic/sfbasic.h>

typedef struct {
  SFBASICHEADER;
  PetscMPIInt   *displs,*recvcounts;
} PetscSF_Allgatherv;

PETSC_INTERN PetscErrorCode PetscSFSetUp_Allgather(PetscSF);
PETSC_INTERN PetscErrorCode PetscSFSetUp_Allgatherv(PetscSF);
PETSC_INTERN PetscErrorCode PetscSFReset_Allgatherv(PetscSF);
PETSC_INTERN PetscErrorCode PetscSFDestroy_Allgatherv(PetscSF);
PETSC_INTERN PetscErrorCode PetscSFFetchAndOpBegin_Allgatherv(PetscSF sf,MPI_Datatype,PetscMemType,void*,PetscMemType,const void*,void*,MPI_Op);
PETSC_INTERN PetscErrorCode PetscSFFetchAndOpEnd_Allgatherv(PetscSF,MPI_Datatype,void*,const void*,void*,MPI_Op);
PETSC_INTERN PetscErrorCode PetscSFGetRootRanks_Allgatherv(PetscSF,PetscInt*,const PetscMPIInt**,const PetscInt**,const PetscInt**,const PetscInt**);
PETSC_INTERN PetscErrorCode PetscSFGetLeafRanks_Allgatherv(PetscSF,PetscInt*,const PetscMPIInt**,const PetscInt**,const PetscInt**);
PETSC_INTERN PetscErrorCode PetscSFCreateLocalSF_Allgatherv(PetscSF,PetscSF*);
PETSC_INTERN PetscErrorCode PetscSFGetGraph_Allgatherv(PetscSF,PetscInt*,PetscInt*,const PetscInt**,const PetscSFNode**);
#endif
