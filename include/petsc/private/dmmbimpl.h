#if !defined(_DMMBIMPL_H)
#define _DMMBIMPL_H

#include <petscdmmoab.h>    /*I      "petscdmmoab.h"    I*/
#include "petsc/private/dmimpl.h"

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
  PetscBool           is_native_vec;
  Vec                 local;
} Vec_MOAB;


typedef struct {
  PetscInt                dim;                            /* Current topological dimension handled by DMMoab */
  PetscInt                n,nloc,nghost;                  /* Number of global, local only and shared vertices for current partition */
  PetscInt                nele,neleloc,neleghost;         /* Number of global, local only and shared elements for current partition */
  PetscInt                bs;                             /* Block size that controls the strided vs interlaced configuration in discrete systems -
                                                             This affects the layout and hence the degree-of-freedom of the different fields (components) */
  PetscInt                *gsindices;                     /* Global ID for all local+ghosted vertices */
  PetscInt                *gidmap,*lidmap,*llmap,*lgmap;  /* Global ID indices, Local ID indices, field-based local map, field-based global map */
  PetscInt                vstart,vend;                    /* Global start and end index for distributed Vec */

  /* MOAB objects cached internally in DMMoab */
  moab::Interface         *mbiface;                       /* MOAB Interface/Core reference */
  moab::ParallelComm      *pcomm;                         /* MOAB ParallelComm reference */
  moab::Tag               ltog_tag;                       /* MOAB supports "global id" tags */
  moab::Tag               material_tag;                   /* MOAB supports "material_set" tags */
  moab::Range             *vowned, *vghost, *vlocal;      /* Vertex entities: strictly owned, strictly ghosted, owned+ghosted */
  moab::Range             *elocal, *eghost;               /* Topological dimensional entities: strictly owned, strictly ghosted */
  moab::Range             *bndyvtx,*bndyfaces,*bndyelems; /* Boundary entities: skin vertices, skin faces and elements on the outer skin */
  moab::EntityHandle      fileset;                        /* The Global set to which all local entities belong */

  PetscInt               *dfill, *ofill;

  /* store the mapping information */
  ISLocalToGlobalMapping  ltog_map;
  VecScatter              ltog_sendrecv;

  /* store options to customize DMMoab */
  PetscInt                rw_dbglevel;
  PetscBool               partition_by_rank;
  char                    extra_read_options[PETSC_MAX_PATH_LEN];
  char                    extra_write_options[PETSC_MAX_PATH_LEN];
  MoabReadMode            read_mode;
  MoabWriteMode           write_mode;

  PetscInt                numFields;
  const char              **fieldNames;
  PetscBool               icreatedinstance;               /* true if DM created moab instance internally, will destroy instance in DMDestroy */
} DM_Moab;


PETSC_EXTERN PetscErrorCode DMCreateGlobalVector_Moab(DM,Vec *);
PETSC_EXTERN PetscErrorCode DMCreateLocalVector_Moab(DM,Vec *);
PETSC_EXTERN PetscErrorCode DMCreateMatrix_Moab(DM dm,Mat *J);
PETSC_EXTERN PetscErrorCode DMGlobalToLocalBegin_Moab(DM,Vec,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMGlobalToLocalEnd_Moab(DM,Vec,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMLocalToGlobalBegin_Moab(DM,Vec,InsertMode,Vec);
PETSC_EXTERN PetscErrorCode DMLocalToGlobalEnd_Moab(DM,Vec,InsertMode,Vec);

#endif /* _DMMBIMPL_H */

