#pragma once

#include <petscvec.h>     /*I "petscvec.h" I*/
#include <petscmat.h>     /*I      "petscmat.h"          I*/
#include <petscdmswarm.h> /*I      "petscdmswarm.h"    I*/
#include <petsc/private/dmimpl.h>

PETSC_EXTERN PetscBool  SwarmProjcite;
PETSC_EXTERN const char SwarmProjCitation[];

PETSC_EXTERN PetscLogEvent DMSWARM_Migrate;
PETSC_EXTERN PetscLogEvent DMSWARM_SetSizes;
PETSC_EXTERN PetscLogEvent DMSWARM_AddPoints;
PETSC_EXTERN PetscLogEvent DMSWARM_RemovePoints;
PETSC_EXTERN PetscLogEvent DMSWARM_Sort;
PETSC_EXTERN PetscLogEvent DMSWARM_DataExchangerTopologySetup;
PETSC_EXTERN PetscLogEvent DMSWARM_DataExchangerBegin;
PETSC_EXTERN PetscLogEvent DMSWARM_DataExchangerEnd;
PETSC_EXTERN PetscLogEvent DMSWARM_DataExchangerSendCount;
PETSC_EXTERN PetscLogEvent DMSWARM_DataExchangerPack;

typedef struct _p_CellDMInfo *CellDMInfo;
struct _p_CellDMInfo {
  DM         dm;         // The cell DM
  PetscInt   Nf;         // The number of DM fields
  char     **dmFields;   // Swarm fields defining this DM
  char      *coordField; // Swarm field for coordinates on this DM
  CellDMInfo next;       // Next struct down in the stack
};

/*
 Error checking to ensure the swarm type is correct and that a cell DM has been set
*/
#define DMSWARMPICVALID(obj) \
  do { \
    DM_Swarm *_swarm = (DM_Swarm *)(obj)->data; \
    PetscCheck(_swarm->swarm_type == DMSWARM_PIC, PetscObjectComm((PetscObject)(dm)), PETSC_ERR_SUP, "Valid only for DMSwarm-PIC. You must call DMSwarmSetType(dm,DMSWARM_PIC)"); \
    PetscCheck(_swarm->cellinfo && _swarm->cellinfo[0].dm, PetscObjectComm((PetscObject)(dm)), PETSC_ERR_SUP, "Valid only for DMSwarmPIC if the cell DM is set. You must call DMSwarmSetCellDM() or DMSwarmPushCellDM()"); \
  } while (0)

typedef struct {
  DMSwarmDataBucket db;
  PetscInt          refct;
  PetscBool         field_registration_initialized;
  PetscBool         field_registration_finalized;
  /* DMSwarmProjectMethod *swarm_project;*/ /* swarm, geometry, result */

  /* PetscInt overlap; */
  /* PetscErrorCode (*update_overlap)(void); */

  const char **vec_field_names;
  PetscInt     vec_field_num;
  PetscInt     vec_field_bs, vec_field_nlocal;
  const char  *coord_name;

  PetscBool          issetup;
  DMSwarmType        swarm_type;
  DMSwarmMigrateType migrate_type;
  DMSwarmCollectType collect_type;

  CellDMInfo cellinfo; // The stack of cell DMs

  PetscBool migrate_error_on_missing_point;

  PetscBool   collect_view_active;
  PetscInt    collect_view_reset_nlocal;
  DMSwarmSort sort_context;

  /* Support for PIC */
  PetscInt Ns; /* The number of particle species */

  PetscSimplePointFn *coordFunc; /* Function to set particle coordinates */
  PetscSimplePointFn *velFunc;   /* Function to set particle velocities */

  /* Debugging */
  PetscInt printCoords;
  PetscInt printWeights;
} DM_Swarm;

typedef struct {
  PetscInt point_index;
  PetscInt cell_index;
} SwarmPoint;

struct _p_DMSwarmSort {
  PetscBool   isvalid;
  PetscInt    ncells, npoints;
  PetscInt   *pcell_offsets;
  SwarmPoint *list;
};

PETSC_INTERN PetscErrorCode DMSwarmMigrate_Push_Basic(DM, PetscBool);
PETSC_INTERN PetscErrorCode DMSwarmMigrate_CellDMScatter(DM, PetscBool);
PETSC_INTERN PetscErrorCode DMSwarmMigrate_CellDMExact(DM, PetscBool);

PETSC_EXTERN PetscErrorCode DMSwarmReplace_Internal(DM, DM *);
