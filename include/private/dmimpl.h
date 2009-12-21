

#if !defined(_DMIMPL_H)
#define _DMIMPL_H

#include "petscda.h"

/*  
    Operations shared by all DM implementations
*/
#define DMOPS(type)	\
  PetscErrorCode (*view)(type,PetscViewer); \
  PetscErrorCode (*setfromoptions)(type); \
  PetscErrorCode (*createglobalvector)(type,Vec*);\
  PetscErrorCode (*createlocalvector)(type,Vec*);\
\
  PetscErrorCode (*getcoloring)(type,ISColoringType,const MatType,ISColoring*);	\
  PetscErrorCode (*getmatrix)(type, const MatType,Mat*);\
  PetscErrorCode (*getinterpolation)(type,type,Mat*,Vec*);\
  PetscErrorCode (*getaggregates)(type,type,Mat*);\
  PetscErrorCode (*getinjection)(type,type,VecScatter*);\
\
  PetscErrorCode (*refine)(type,MPI_Comm,type*);\
  PetscErrorCode (*coarsen)(type,MPI_Comm,type*);\
  PetscErrorCode (*refinehierarchy)(type,PetscInt,type*);\
  PetscErrorCode (*coarsenhierarchy)(type,PetscInt,type*);\
\
  PetscErrorCode (*forminitialguess)(type,PetscErrorCode (*)(void),Vec,void*);\
  PetscErrorCode (*formfunction)(type,PetscErrorCode (*)(void),Vec,Vec);\
\
  PetscErrorCode (*globaltolocalbegin)(type,Vec,InsertMode,Vec);		\
  PetscErrorCode (*globaltolocalend)(type,Vec,InsertMode,Vec); \
  PetscErrorCode (*localtoglobal)(type,Vec,InsertMode,Vec); \
\
  PetscErrorCode (*getelements)(type,PetscInt*,const PetscInt*[]);   \
  PetscErrorCode (*restoreelements)(type,PetscInt*,const PetscInt*[]); \
\
  PetscErrorCode (*destroy)(type);


typedef struct _DMOps *DMOps;
struct _DMOps {
  DMOPS(DM)
};

#define DM_MAX_WORK_VECTORS 100 /* work vectors available to users  via DMGetGlobalVector(), DMGetLocalVector() */

#define DMHEADER \
  Vec   localin[DM_MAX_WORK_VECTORS],localout[DM_MAX_WORK_VECTORS];   \
  Vec   globalin[DM_MAX_WORK_VECTORS],globalout[DM_MAX_WORK_VECTORS]; 


struct _p_DM {
  PETSCHEADER(struct _DMOps);
  DMHEADER
};

/*

          Composite Vectors 

      Single global representation
      Individual global representations
      Single local representation
      Individual local representations

      Subsets of individual as a single????? Do we handle this by having DMComposite inside composite??????

       DA da_u, da_v, da_p

       DMComposite dm_velocities

       DMComposite dm

       DACreate(,&da_u);
       DACreate(,&da_v);
       DMCompositeCreate(,&dm_velocities);
       DMCompositeAddDM(dm_velocities,(DM)du);
       DMCompositeAddDM(dm_velocities,(DM)dv);

       DACreate(,&da_p);
       DMCompositeCreate(,&dm_velocities);
       DMCompositeAddDM(dm,(DM)dm_velocities);     
       DMCompositeAddDM(dm,(DM)dm_p);     


    Access parts of composite vectors (DMComposite only)
    ---------------------------------
      DMCompositeGetAccess  - access the global vector as subvectors and array (for redundant arrays)
      ADD for local vector - 

    Element access
    --------------
      From global vectors 
         -DAVecGetArray   - for DA
         -VecGetArray - for Sliced
         ADD for DMComposite???  maybe

      From individual vector
          -DAVecGetArray - for DA
          -VecGetArray -for sliced  
         ADD for DMComposite??? maybe

      From single local vector
          ADD         * single local vector as arrays?

   Communication 
   -------------
      DMGlobalToLocal - global vector to single local vector

      DMCompositeScatter/Gather - direct to individual local vectors and arrays   CHANGE name closer to GlobalToLocal?

   Obtaining vectors
   ----------------- 
      DMCreateGlobal/Local 
      DMGetGlobal/Local 
      DMCompositeGetLocalVectors   - gives individual local work vectors and arrays
         

?????   individual global vectors   ????

*/

#endif
