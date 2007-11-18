

#if !defined(_DMIMPL_H)
#define _DMIMPL_H

#include "petscda.h"

/*  
    Operations shared by all DM implementations, including DA, VecPack and Mesh
*/
#define DMOPS(type)	\
  PetscErrorCode (*view)(type,PetscViewer); \
  PetscErrorCode (*createglobalvector)(type,Vec*);\
  PetscErrorCode (*createlocalvector)(type,Vec*);\
\
  PetscErrorCode (*getcoloring)(type,ISColoringType,ISColoring*);\
  PetscErrorCode (*getmatrix)(type, MatType,Mat*);\
  PetscErrorCode (*getinterpolation)(type,type,Mat*,Vec*);\
  PetscErrorCode (*getaggregates)(type,type,Mat*);\
  PetscErrorCode (*getinjection)(type,type,VecScatter*);\
\
  PetscErrorCode (*refine)(type,MPI_Comm,type*);\
  PetscErrorCode (*coarsen)(type,MPI_Comm,type*);\
  PetscErrorCode (*refinehierarchy)(type,PetscInt,type**);\
  PetscErrorCode (*coarsenhierarchy)(type,PetscInt,type**);\
\
  PetscErrorCode (*forminitialguess)(type,PetscErrorCode (*)(void),Vec,void*);\
  PetscErrorCode (*formfunction)(type,PetscErrorCode (*)(void),Vec,Vec);\
\
  PetscErrorCode (*globaltolocalbegin)(type,Vec,InsertMode,Vec);		\
  PetscErrorCode (*globaltolocalend)(type,Vec,InsertMode,Vec); \
  PetscErrorCode (*localtoglobal)(type,Vec,InsertMode,Vec); 


typedef struct _DMOps *DMOps;
struct _DMOps {
  DMOPS(DM)
};

struct _p_DM {
  PETSCHEADER(struct _DMOps);
};



#endif
