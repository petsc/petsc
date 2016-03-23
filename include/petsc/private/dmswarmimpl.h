
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
  
  PetscBool field_registration_finalized;
  //DM *geometry_dm;
  //DMSwarmProjectMethod *swarm_project;/* swarm, geometry, result */

  //PetscInt overlap;
  //PetscErrorCode (*update_overlap)(void);
} DM_Swarm;

#endif /* _SWARMIMPL_H */
