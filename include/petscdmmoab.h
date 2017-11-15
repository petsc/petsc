#if !defined(__PETSCMOAB_H)
#define __PETSCMOAB_H

#include <petscvec.h>
#include <petscmat.h>
#include <petscdm.h>

#include <moab/Core.hpp>
#include <moab/ParallelComm.hpp>

/* The MBERR macro is used to save typing. It checks a MOAB error code
 * (rval) and calls SETERRQ if not MB_SUCCESS. A message (msg) can
 * also be passed in. */
#define MBERR(msg,rval) do{if(rval != moab::MB_SUCCESS) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_LIB,"MOAB ERROR (%i): %s",rval,msg);} while(0)
#define MBERRNM(rval) do{if(rval != moab::MB_SUCCESS) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"MOAB ERROR (%i)",rval);} while(0)

PETSC_EXTERN PetscErrorCode DMMoabCreate(MPI_Comm comm, DM *moab);
PETSC_EXTERN PetscErrorCode DMMoabCreateMoab(MPI_Comm comm, moab::Interface *mbiface, moab::ParallelComm *pcomm, moab::Tag ltog_tag, moab::Range *range, DM *moab);
PETSC_EXTERN PetscErrorCode DMMoabSetParallelComm(DM dm,moab::ParallelComm *pcomm);
PETSC_EXTERN PetscErrorCode DMMoabGetParallelComm(DM dm,moab::ParallelComm **pcomm);
PETSC_EXTERN PetscErrorCode DMMoabSetInterface(DM dm,moab::Interface *iface);
PETSC_EXTERN PetscErrorCode DMMoabGetInterface(DM dm,moab::Interface **mbint);
PETSC_EXTERN PetscErrorCode DMMoabSetRange(DM dm,moab::Range range);
PETSC_EXTERN PetscErrorCode DMMoabGetRange(DM dm,moab::Range *range);
PETSC_EXTERN PetscErrorCode DMMoabSetLocalToGlobalTag(DM dm,moab::Tag ltogtag);
PETSC_EXTERN PetscErrorCode DMMoabGetLocalToGlobalTag(DM dm,moab::Tag *ltog_tag);

PETSC_EXTERN PetscErrorCode DMMoabCreateVector(DM dm, moab::Tag tag,PetscInt tag_size,moab::Range range,PetscBool serial, PetscBool destroy_tag,Vec *X);
PETSC_EXTERN PetscErrorCode DMMoabGetVecTag(Vec vec,moab::Tag *tag);
PETSC_EXTERN PetscErrorCode DMMoabGetVecRange(Vec vec,moab::Range *range);
#endif
