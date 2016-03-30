
#if !defined(_SWARMIMPL_H)
#define _SWARMIMPL_H

#include <petscvec.h> /*I "petscvec.h" I*/
#include <petscmat.h>       /*I      "petscmat.h"          I*/
#include <petscdmswarm.h> /*I      "petscdmswarm.h"    I*/
#include <petsc/private/dmimpl.h>

typedef struct _p_DataField* DataField;
typedef struct _p_DataBucket* DataBucket;

typedef struct {
  DataBucket db;
  
  PetscBool field_registration_initialized;
  PetscBool field_registration_finalized;
  //DM *geometry_dm;
  //DMSwarmProjectMethod *swarm_project;/* swarm, geometry, result */

  //PetscInt overlap;
  //PetscErrorCode (*update_overlap)(void);

  char      vec_field_name[PETSC_MAX_PATH_LEN];
  PetscBool vec_field_set;
  PetscInt  vec_field_bs,vec_field_nlocal;

  PetscBool          issetup;
  DMSwarmType        swarm_type;
  DMSwarmMigrateType migrate_type;
  DMSwarmCollectType collect_type;
  
  DM        dmcell;

  PetscBool migrate_error_on_missing_point;
  
  PetscBool collect_view_active;
  PetscInt  collect_view_reset_nlocal;
} DM_Swarm;

#endif /* _SWARMIMPL_H */
