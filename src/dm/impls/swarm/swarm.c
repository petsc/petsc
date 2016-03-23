
#define PETSCDM_DLL
#include <petsc/private/dmswarmimpl.h>    /*I   "petscdmswarm.h"   I*/

//typedef PetscErrorCode (*swarm_project)(DM,DM,Vec) DMSwarmProjectMethod; /* swarm, geometry, result */

//typedef enum { PROJECT_DMDA_AQ1=0, PROJECT_DMDA_P0 } DMSwarmDMDAProjectionType;

#if 0
PetscErrorCode DMSwarmRegisterPetscDatatypeField(DM dm,const char fieldname[],PetscInt blocksize,PetscDataType type)
{

  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmRegisterUserDatatypeField(DM dm,const char fieldname[],size_t bytes)
{
  
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmGetField(DM dm,const char fieldname[],PetscInt *blocksize,PetscDataType *type,void **data)
{
  
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmDefineFieldVector(DM dm,const char *fieldnames[])
{
  /* Check all fields are of type PETSC_REAL or PETSC_SCALAR */
  /* Compute summed block size */
  /* Set guard */
  PetscFunctionReturn(0);
}

PetscErrorCode DMSwarmGetGlobalVectorFromFields(DM dm,Vec *vec)
{
  PetscFunctionReturn(0);
}
PetscErrorCode DMSwarmRestoreGlobalVectorFromFields(DM dm,Vec *vec)
{
  PetscFunctionReturn(0);
}

/* requires DMSwarmDefineFieldVector has been called */
PetscErrorCode DMCreateGlobalVector_Swarm()
{

  PetscFunctionReturn(0);
}

/* requires DMSwarmDefineFieldVector has been called */
PetscErrorCode DMLocalGlobalVector_Swarm()
{
  
  PetscFunctionReturn();
}

/* Defines what the local space will be */
PetscErrorCode DMSwarmSetOverlap()
{
  
  PetscFunctionReturn();
}


/* coordinates */
/*
DMGetCoordinateDM returns self
DMGetCoordinates and DMGetCoordinatesLocal return same thing
Local view could be used to define overlapping information
*/

#endif

#undef __FUNCT__
#define __FUNCT__ "DMDestroy_Swarm"
PetscErrorCode DMDestroy_Swarm(DM dm)
{
  DM_Swarm *swarm = (DM_Swarm*)dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(swarm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreate_Swarm"
PETSC_EXTERN PetscErrorCode DMCreate_Swarm(DM dm)
{
  DM_Swarm      *swarm;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr     = PetscNewLog(dm,&swarm);CHKERRQ(ierr);
  
  dm->dim  = 0;
  dm->data = swarm;
  
  dm->ops->view                            = NULL;
  dm->ops->load                            = NULL;
  dm->ops->setfromoptions                  = NULL;
  dm->ops->clone                           = NULL;
  dm->ops->setup                           = NULL;
  dm->ops->createdefaultsection            = NULL;
  dm->ops->createdefaultconstraints        = NULL;
  dm->ops->createglobalvector              = NULL;
  dm->ops->createlocalvector               = NULL;
  dm->ops->getlocaltoglobalmapping         = NULL;
  dm->ops->createfieldis                   = NULL;
  dm->ops->createcoordinatedm              = NULL;
  dm->ops->getcoloring                     = NULL;
  dm->ops->creatematrix                    = NULL;
  dm->ops->createinterpolation             = NULL;
  dm->ops->getaggregates                   = NULL;
  dm->ops->getinjection                    = NULL;
  dm->ops->refine                          = NULL;
  dm->ops->coarsen                         = NULL;
  dm->ops->refinehierarchy                 = NULL;
  dm->ops->coarsenhierarchy                = NULL;
  dm->ops->globaltolocalbegin              = NULL;
  dm->ops->globaltolocalend                = NULL;
  dm->ops->localtoglobalbegin              = NULL;
  dm->ops->localtoglobalend                = NULL;
  dm->ops->destroy                         = DMDestroy_Swarm;
  dm->ops->createsubdm                     = NULL;
  dm->ops->getdimpoints                    = NULL;
  dm->ops->locatepoints                    = NULL;
  
  PetscFunctionReturn(0);
}