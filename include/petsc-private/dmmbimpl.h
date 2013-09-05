#if !defined(_MOABIMPL_H)
#define _MOABIMPL_H

#include <petscmat.h>       /*I      "petscmat.h"          I*/
#include <petscdmmoab.h> /*I      "petscdmplex.h"    I*/
#include "petsc-private/dmimpl.h"

/* PETSC_EXTERN PetscLogEvent DMMOAB_Read, DMMOAB_Write; */

/* This is an integer map, in addition it is also a container class
   Design points:
     - Low storage is the most important design point
     - We want flexible insertion and deletion
     - We can live with O(log) query, but we need O(1) iteration over strata
*/
typedef struct {
  moab::Interface    *mbiface;
  moab::ParallelComm *pcomm;
  moab::Range        *tag_range; /* entities to which this tag applies */
  moab::Tag           tag;
  PetscInt            tag_size;
  PetscBool           new_tag;
  PetscBool           is_global_vec;
} Vec_MOAB;


typedef struct {
  PetscInt                bs;                     /* Number of degrees of freedom on each entity, aka tag size in moab */
  PetscInt                dim;
  PetscInt                n,nloc,nghost;           /* Number of global, local only and shared degrees of freedom for current partition */
  PetscInt                nele,neleloc;           /* Number of global, local only and shared degrees of freedom for current partition */
  PetscBool               icreatedinstance;       /* true if DM created moab instance internally, will destroy instance in DMDestroy */
  PetscInt                *gsindices;
  PetscInt                *gidmap,*lidmap,*lmap,*lgmap;

  moab::ParallelComm      *pcomm;
  moab::Interface         *mbiface;
  moab::Tag               ltog_tag;               /* moab supports "global id" tags, which are usually local to global numbering */
  ISLocalToGlobalMapping  ltog_map;
  VecScatter              ltog_sendrecv;
  moab::Range             *vlocal, *vowned, *vghost;
  moab::Range             *elocal, *eghost;
  moab::EntityHandle      fileset;

  PetscBool               *isbndyvtx,*isbndyfaces,*isbndyelems;

  PetscInt                nfields;
  const char**            fields;
} DM_Moab;


PETSC_EXTERN PetscErrorCode DMCreateGlobalVector_Moab(DM,Vec *);
PETSC_EXTERN PetscErrorCode DMCreateLocalVector_Moab(DM,Vec *);
PETSC_EXTERN PetscErrorCode DMCreateMatrix_Moab(DM dm, MatType mtype,Mat *J);
PETSC_EXTERN PetscErrorCode DMGlobalToLocalBegin_Moab(DM,Vec,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMGlobalToLocalEnd_Moab(DM,Vec,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMLocalToGlobalBegin_Moab(DM,Vec,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMLocalToGlobalEnd_Moab(DM,Vec,InsertMode,Vec);


#endif /* _MOABIMPL_H */

