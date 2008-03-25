

#if !defined(_DMIMPL_H)
#define _DMIMPL_H

#include "petscda.h"

/*  
    Operations shared by all DM implementations
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
  PetscErrorCode (*localtoglobal)(type,Vec,InsertMode,Vec); \
\
  PetscErrorCode (*getelements)(DM,PetscInt*,const PetscInt*[]);   \
  PetscErrorCode (*restoreelements)(DM,PetscInt*,const PetscInt*[]);


typedef struct _DMOps *DMOps;
struct _DMOps {
  DMOPS(DM)
};

#define DM_MAX_WORK_VECTORS 10 /* work vectors available to users  via DMGetGlobalVector(), DMGetLocalVector() */

#define DMHEADER \
  Vec   localin[DM_MAX_WORK_VECTORS],localout[DM_MAX_WORK_VECTORS];   \
  Vec   globalin[DM_MAX_WORK_VECTORS],globalout[DM_MAX_WORK_VECTORS]; 


struct _p_DM {
  PETSCHEADER(struct _DMOps);
  DMHEADER
};



#endif
