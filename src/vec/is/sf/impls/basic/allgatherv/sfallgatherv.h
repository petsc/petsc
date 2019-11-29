#if !defined(_SFALLGATHERV_H)
#define _SFALLGATHERV_H

#include <petsc/private/sfimpl.h> /*I "petscsf.h" I*/
#include <../src/vec/is/sf/impls/basic/sfpack.h>
#include <../src/vec/is/sf/impls/basic/sfbasic.h>

typedef struct {
  SFBASICHEADER;
  PetscMPIInt   *displs,*recvcounts;
} PetscSF_Allgatherv;

PETSC_INTERN PetscErrorCode PetscSFSetUp_Allgatherv(PetscSF);
PETSC_INTERN PetscErrorCode PetscSFPackGet_Allgatherv(PetscSF,MPI_Datatype,PetscMemType,const void*,PetscMemType,const void*,PetscSFPack*);
PETSC_INTERN PetscErrorCode PetscSFReset_Allgatherv(PetscSF);
PETSC_INTERN PetscErrorCode PetscSFDestroy_Allgatherv(PetscSF);
PETSC_INTERN PetscErrorCode PetscSFBcastAndOpEnd_Allgatherv(PetscSF,MPI_Datatype,PetscMemType,const void*,PetscMemType,void*,MPI_Op);
PETSC_INTERN PetscErrorCode PetscSFReduceEnd_Allgatherv(PetscSF,MPI_Datatype,PetscMemType,const void*,PetscMemType,void*,MPI_Op);
PETSC_INTERN PetscErrorCode PetscSFFetchAndOpBegin_Allgatherv(PetscSF sf,MPI_Datatype,PetscMemType,void*,PetscMemType,const void*,void*,MPI_Op);
PETSC_INTERN PetscErrorCode PetscSFFetchAndOpEnd_Allgatherv(PetscSF,MPI_Datatype,PetscMemType,void*,PetscMemType,const void*,void*,MPI_Op);
PETSC_INTERN PetscErrorCode PetscSFGetRootRanks_Allgatherv(PetscSF,PetscInt*,const PetscMPIInt**,const PetscInt**,const PetscInt**,const PetscInt**);
PETSC_INTERN PetscErrorCode PetscSFGetLeafRanks_Allgatherv(PetscSF,PetscInt*,const PetscMPIInt**,const PetscInt**,const PetscInt**);
PETSC_INTERN PetscErrorCode PetscSFCreateLocalSF_Allgatherv(PetscSF,PetscSF*);
PETSC_INTERN PetscErrorCode PetscSFGetGraph_Allgatherv(PetscSF,PetscInt*,PetscInt*,const PetscInt**,const PetscSFNode**);

PETSC_INTERN PetscErrorCode PetscSFBcastPrepareMPIBuffers_Allgatherv(PetscSF,PetscSFPack,MPI_Op,PetscMemType*,const void**,PetscMemType*,void**);
PETSC_INTERN PetscErrorCode PetscSFReducePrepareMPIBuffers_Allgatherv(PetscSF,PetscSFPack,MPI_Op,PetscMemType*,void**,PetscMemType*,const void**);
#endif
