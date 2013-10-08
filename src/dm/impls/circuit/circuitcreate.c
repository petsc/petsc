#define PETSCDM_DLL
#include <petsc-private/dmcircuitimpl.h>    /*I   "petscdmcircuit.h"   I*/
#include <petscdmda.h>

#undef __FUNCT__
#define __FUNCT__ "DMSetFromOptions_Circuit"
PetscErrorCode  DMSetFromOptions_Circuit(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscOptionsHead("DMCircuit Options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* External function declarations here */
extern PetscErrorCode DMCreateMatrix_Circuit(DM, Mat*);
extern PetscErrorCode DMDestroy_Circuit(DM);
extern PetscErrorCode DMView_Circuit(DM, PetscViewer);
extern PetscErrorCode DMGlobalToLocalBegin_Circuit(DM, Vec, InsertMode, Vec);
extern PetscErrorCode DMGlobalToLocalEnd_Circuit(DM, Vec, InsertMode, Vec);
extern PetscErrorCode DMLocalToGlobalBegin_Circuit(DM, Vec, InsertMode, Vec);
extern PetscErrorCode DMLocalToGlobalEnd_Circuit(DM, Vec, InsertMode, Vec);
extern PetscErrorCode DMSetUp_Circuit(DM);


#undef __FUNCT__
#define __FUNCT__ "DMCreateGlobalVector_Circuit"
static PetscErrorCode DMCreateGlobalVector_Circuit(DM dm,Vec *vec)
{
  PetscErrorCode ierr;
  DM_Circuit     *circuit = (DM_Circuit*) dm->data;

  PetscFunctionBegin;
  ierr = DMCreateGlobalVector(circuit->plex,vec);CHKERRQ(ierr);
  ierr = VecSetDM(*vec,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreateLocalVector_Circuit"
static PetscErrorCode DMCreateLocalVector_Circuit(DM dm,Vec *vec)
{
  PetscErrorCode ierr;
  DM_Circuit     *circuit = (DM_Circuit*) dm->data;

  PetscFunctionBegin;
  ierr = DMCreateLocalVector(circuit->plex,vec);CHKERRQ(ierr);
  ierr = VecSetDM(*vec,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMInitialize_Circuit"
PetscErrorCode DMInitialize_Circuit(DM dm)
{

  PetscFunctionBegin;

  dm->ops->view                            = NULL;
  dm->ops->setfromoptions                  = DMSetFromOptions_Circuit;
  dm->ops->clone                           = NULL;
  dm->ops->setup                           = DMSetUp_Circuit;
  dm->ops->createglobalvector              = DMCreateGlobalVector_Circuit;
  dm->ops->createlocalvector               = DMCreateLocalVector_Circuit;
  dm->ops->getlocaltoglobalmapping         = NULL;
  dm->ops->getlocaltoglobalmappingblock    = NULL;
  dm->ops->createfieldis                   = NULL;
  dm->ops->createcoordinatedm              = NULL;
  dm->ops->getcoloring                     = 0;
  dm->ops->creatematrix                    = DMCreateMatrix_Circuit;
  dm->ops->createinterpolation             = 0;
  dm->ops->getaggregates                   = 0;
  dm->ops->getinjection                    = 0;
  dm->ops->refine                          = 0;
  dm->ops->coarsen                         = 0;
  dm->ops->refinehierarchy                 = 0;
  dm->ops->coarsenhierarchy                = 0;
  dm->ops->globaltolocalbegin              = DMGlobalToLocalBegin_Circuit;
  dm->ops->globaltolocalend                = DMGlobalToLocalEnd_Circuit;
  dm->ops->localtoglobalbegin              = DMLocalToGlobalBegin_Circuit;
  dm->ops->localtoglobalend                = DMLocalToGlobalEnd_Circuit;
  dm->ops->destroy                         = DMDestroy_Circuit;
  dm->ops->createsubdm                     = NULL;
  dm->ops->locatepoints                    = NULL;
  PetscFunctionReturn(0);
}

/*MC
  DMCIRCUIT = "circuit" - A DM object that encapsulates an unstructured circuit. The implementation is based on the DM object
                          DMPlex that manages unstructured grids. Distributed circuits use a non-overlapping partitioning of
                          the edges. In the local representation, Vecs contain all unknowns in the interior and shared boundary.
                          This is specified by a PetscSection object. Ownership in the global representation is determined by
                          ownership of the underlying DMPlex points. This is specified by another PetscSection object.

  Level: intermediate

.seealso: DMType, DMCircuitCreate(), DMCreate(), DMSetType()
M*/

#undef __FUNCT__
#define __FUNCT__ "DMCreate_Circuit"
PETSC_EXTERN PetscErrorCode DMCreate_Circuit(DM dm)
{
  DM_Circuit     *circuit;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr     = PetscNewLog(dm, DM_Circuit, &circuit);CHKERRQ(ierr);
  dm->data = circuit;

  circuit->refct          = 1;
  circuit->NNodes         = -1;
  circuit->NEdges         = -1;
  circuit->nNodes         = -1;
  circuit->nEdges         = -1;

  ierr = DMInitialize_Circuit(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCircuitCreate"
/*@
  DMCircuitCreate - Creates a DMCircuit object, which encapsulates an unstructured circuit.

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the DMCircuit object

  Output Parameter:
. circuit  - The DMCircuit object

  Level: beginner

.keywords: DMCircuit, create
@*/
PetscErrorCode DMCircuitCreate(MPI_Comm comm, DM *circuit)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(circuit,2);
  ierr = DMCreate(comm, circuit);CHKERRQ(ierr);
  ierr = DMSetType(*circuit, DMCIRCUIT);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
