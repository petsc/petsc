#define PETSCDM_DLL
#include <petsc/private/dmnetworkimpl.h>    /*I   "petscdmnetwork.h"   I*/
#include <petscdmda.h>

#undef __FUNCT__
#define __FUNCT__ "DMSetFromOptions_Network"
PetscErrorCode  DMSetFromOptions_Network(PetscOptionItems *PetscOptionsObject,DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscOptionsHead(PetscOptionsObject,"DMNetwork Options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* External function declarations here */
extern PetscErrorCode DMCreateMatrix_Network(DM, Mat*);
extern PetscErrorCode DMDestroy_Network(DM);
extern PetscErrorCode DMView_Network(DM, PetscViewer);
extern PetscErrorCode DMGlobalToLocalBegin_Network(DM, Vec, InsertMode, Vec);
extern PetscErrorCode DMGlobalToLocalEnd_Network(DM, Vec, InsertMode, Vec);
extern PetscErrorCode DMLocalToGlobalBegin_Network(DM, Vec, InsertMode, Vec);
extern PetscErrorCode DMLocalToGlobalEnd_Network(DM, Vec, InsertMode, Vec);
extern PetscErrorCode DMSetUp_Network(DM);
extern PetscErrorCode DMClone_Network(DM, DM*);


#undef __FUNCT__
#define __FUNCT__ "DMCreateGlobalVector_Network"
static PetscErrorCode DMCreateGlobalVector_Network(DM dm,Vec *vec)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*) dm->data;

  PetscFunctionBegin;
  ierr = DMCreateGlobalVector(network->plex,vec);CHKERRQ(ierr);
  ierr = VecSetDM(*vec,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateLocalVector_Network"
static PetscErrorCode DMCreateLocalVector_Network(DM dm,Vec *vec)
{
  PetscErrorCode ierr;
  DM_Network     *network = (DM_Network*) dm->data;

  PetscFunctionBegin;
  ierr = DMCreateLocalVector(network->plex,vec);CHKERRQ(ierr);
  ierr = VecSetDM(*vec,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMInitialize_Network"
PetscErrorCode DMInitialize_Network(DM dm)
{

  PetscFunctionBegin;

  dm->ops->view                            = NULL;
  dm->ops->setfromoptions                  = DMSetFromOptions_Network;
  dm->ops->clone                           = DMClone_Network;
  dm->ops->setup                           = DMSetUp_Network;
  dm->ops->createglobalvector              = DMCreateGlobalVector_Network;
  dm->ops->createlocalvector               = DMCreateLocalVector_Network;
  dm->ops->getlocaltoglobalmapping         = NULL;
  dm->ops->createfieldis                   = NULL;
  dm->ops->createcoordinatedm              = NULL;
  dm->ops->getcoloring                     = 0;
  dm->ops->creatematrix                    = DMCreateMatrix_Network;
  dm->ops->createinterpolation             = 0;
  dm->ops->getaggregates                   = 0;
  dm->ops->getinjection                    = 0;
  dm->ops->refine                          = 0;
  dm->ops->coarsen                         = 0;
  dm->ops->refinehierarchy                 = 0;
  dm->ops->coarsenhierarchy                = 0;
  dm->ops->globaltolocalbegin              = DMGlobalToLocalBegin_Network;
  dm->ops->globaltolocalend                = DMGlobalToLocalEnd_Network;
  dm->ops->localtoglobalbegin              = DMLocalToGlobalBegin_Network;
  dm->ops->localtoglobalend                = DMLocalToGlobalEnd_Network;
  dm->ops->destroy                         = DMDestroy_Network;
  dm->ops->createsubdm                     = NULL;
  dm->ops->locatepoints                    = NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMClone_Network"
PetscErrorCode DMClone_Network(DM dm, DM *newdm)
{
  DM_Network     *network = (DM_Network *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  network->refct++;
  (*newdm)->data = network;
  ierr = PetscObjectChangeTypeName((PetscObject) *newdm, DMNETWORK);CHKERRQ(ierr);
  ierr = DMInitialize_Network(*newdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
  DMNETWORK = "network" - A DM object that encapsulates an unstructured network. The implementation is based on the DM object
                          DMPlex that manages unstructured grids. Distributed networks use a non-overlapping partitioning of
                          the edges. In the local representation, Vecs contain all unknowns in the interior and shared boundary.
                          This is specified by a PetscSection object. Ownership in the global representation is determined by
                          ownership of the underlying DMPlex points. This is specified by another PetscSection object.

  Level: intermediate

.seealso: DMType, DMNetworkCreate(), DMCreate(), DMSetType()
M*/

#undef __FUNCT__
#define __FUNCT__ "DMCreate_Network"
PETSC_EXTERN PetscErrorCode DMCreate_Network(DM dm)
{
  DM_Network     *network;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr     = PetscNewLog(dm,&network);CHKERRQ(ierr);
  dm->data = network;

  network->refct          = 1;
  network->NNodes         = -1;
  network->NEdges         = -1;
  network->nNodes         = -1;
  network->nEdges         = -1;

  ierr = DMInitialize_Network(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMNetworkCreate"
/*@
  DMNetworkCreate - Creates a DMNetwork object, which encapsulates an unstructured network.

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the DMNetwork object

  Output Parameter:
. network  - The DMNetwork object

  Level: beginner

.keywords: DMNetwork, create
@*/
PetscErrorCode DMNetworkCreate(MPI_Comm comm, DM *network)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(network,2);
  ierr = DMCreate(comm, network);CHKERRQ(ierr);
  ierr = DMSetType(*network, DMNETWORK);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
