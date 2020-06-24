#include <petsc/private/dmforestimpl.h> /*I "petscdmforest.h" I*/
#include <petsc/private/dmimpl.h>       /*I "petscdm.h" I*/
#include <petsc/private/dmlabelimpl.h>  /*I "petscdmlabel.h" I*/
#include <petscsf.h>

PetscBool DMForestPackageInitialized = PETSC_FALSE;

typedef struct _DMForestTypeLink*DMForestTypeLink;

struct _DMForestTypeLink
{
  char             *name;
  DMForestTypeLink next;
};

DMForestTypeLink DMForestTypeList;

static PetscErrorCode DMForestPackageFinalize(void)
{
  DMForestTypeLink oldLink, link = DMForestTypeList;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  while (link) {
    oldLink = link;
    ierr    = PetscFree(oldLink->name);CHKERRQ(ierr);
    link    = oldLink->next;
    ierr    = PetscFree(oldLink);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMForestPackageInitialize(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (DMForestPackageInitialized) PetscFunctionReturn(0);
  DMForestPackageInitialized = PETSC_TRUE;

  ierr = DMForestRegisterType(DMFOREST);CHKERRQ(ierr);
  ierr = PetscRegisterFinalize(DMForestPackageFinalize);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMForestRegisterType - Registers a DMType as a subtype of DMFOREST (so that DMIsForest() will be correct)

  Not Collective

  Input parameter:
. name - the name of the type

  Level: advanced

.seealso: DMFOREST, DMIsForest()
@*/
PetscErrorCode DMForestRegisterType(DMType name)
{
  DMForestTypeLink link;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr             = DMForestPackageInitialize();CHKERRQ(ierr);
  ierr             = PetscNew(&link);CHKERRQ(ierr);
  ierr             = PetscStrallocpy(name,&link->name);CHKERRQ(ierr);
  link->next       = DMForestTypeList;
  DMForestTypeList = link;
  PetscFunctionReturn(0);
}

/*@
  DMIsForest - Check whether a DM uses the DMFOREST interface for hierarchically-refined meshes

  Not Collective

  Input parameter:
. dm - the DM object

  Output parameter:
. isForest - whether dm is a subtype of DMFOREST

  Level: intermediate

.seealso: DMFOREST, DMForestRegisterType()
@*/
PetscErrorCode DMIsForest(DM dm, PetscBool *isForest)
{
  DMForestTypeLink link = DMForestTypeList;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  while (link) {
    PetscBool sameType;
    ierr = PetscObjectTypeCompare((PetscObject)dm,link->name,&sameType);CHKERRQ(ierr);
    if (sameType) {
      *isForest = PETSC_TRUE;
      PetscFunctionReturn(0);
    }
    link = link->next;
  }
  *isForest = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/*@
  DMForestTemplate - Create a new DM that will be adapted from a source DM.  The new DM reproduces the configuration
  of the source, but is not yet setup, so that the user can then define only the ways that the new DM should differ
  (by, e.g., refinement or repartitioning).  The source DM is also set as the adaptivity source DM of the new DM (see
  DMForestSetAdaptivityForest()).

  Collective on dm

  Input Parameters:
+ dm - the source DM object
- comm - the communicator for the new DM (this communicator is currently ignored, but is present so that DMForestTemplate() can be used within DMCoarsen())

  Output Parameter:
. tdm - the new DM object

  Level: intermediate

.seealso: DMForestSetAdaptivityForest()
@*/
PetscErrorCode DMForestTemplate(DM dm, MPI_Comm comm, DM *tdm)
{
  DM_Forest                  *forest = (DM_Forest*) dm->data;
  DMType                     type;
  DM                         base;
  DMForestTopology           topology;
  MatType                    mtype;
  PetscInt                   dim, overlap, ref, factor;
  DMForestAdaptivityStrategy strat;
  void                       *ctx;
  PetscErrorCode             (*map)(DM, PetscInt, PetscInt, const PetscReal[], PetscReal[], void*);
  void                       *mapCtx;
  PetscErrorCode             ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMCreate(PetscObjectComm((PetscObject)dm),tdm);CHKERRQ(ierr);
  ierr = DMGetType(dm,&type);CHKERRQ(ierr);
  ierr = DMSetType(*tdm,type);CHKERRQ(ierr);
  ierr = DMForestGetBaseDM(dm,&base);CHKERRQ(ierr);
  ierr = DMForestSetBaseDM(*tdm,base);CHKERRQ(ierr);
  ierr = DMForestGetTopology(dm,&topology);CHKERRQ(ierr);
  ierr = DMForestSetTopology(*tdm,topology);CHKERRQ(ierr);
  ierr = DMForestGetAdjacencyDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMForestSetAdjacencyDimension(*tdm,dim);CHKERRQ(ierr);
  ierr = DMForestGetPartitionOverlap(dm,&overlap);CHKERRQ(ierr);
  ierr = DMForestSetPartitionOverlap(*tdm,overlap);CHKERRQ(ierr);
  ierr = DMForestGetMinimumRefinement(dm,&ref);CHKERRQ(ierr);
  ierr = DMForestSetMinimumRefinement(*tdm,ref);CHKERRQ(ierr);
  ierr = DMForestGetMaximumRefinement(dm,&ref);CHKERRQ(ierr);
  ierr = DMForestSetMaximumRefinement(*tdm,ref);CHKERRQ(ierr);
  ierr = DMForestGetAdaptivityStrategy(dm,&strat);CHKERRQ(ierr);
  ierr = DMForestSetAdaptivityStrategy(*tdm,strat);CHKERRQ(ierr);
  ierr = DMForestGetGradeFactor(dm,&factor);CHKERRQ(ierr);
  ierr = DMForestSetGradeFactor(*tdm,factor);CHKERRQ(ierr);
  ierr = DMForestGetBaseCoordinateMapping(dm,&map,&mapCtx);CHKERRQ(ierr);
  ierr = DMForestSetBaseCoordinateMapping(*tdm,map,mapCtx);CHKERRQ(ierr);
  if (forest->ftemplate) {
    ierr = (*forest->ftemplate)(dm, *tdm);CHKERRQ(ierr);
  }
  ierr = DMForestSetAdaptivityForest(*tdm,dm);CHKERRQ(ierr);
  ierr = DMCopyDisc(dm,*tdm);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dm,&ctx);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(*tdm,&ctx);CHKERRQ(ierr);
  {
    PetscBool            isper;
    const PetscReal      *maxCell, *L;
    const DMBoundaryType *bd;

    ierr = DMGetPeriodicity(dm,&isper,&maxCell,&L,&bd);CHKERRQ(ierr);
    ierr = DMSetPeriodicity(*tdm,isper,maxCell,L,bd);CHKERRQ(ierr);
  }
  ierr = DMCopyBoundary(dm,*tdm);CHKERRQ(ierr);
  ierr = DMGetMatType(dm,&mtype);CHKERRQ(ierr);
  ierr = DMSetMatType(*tdm,mtype);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMInitialize_Forest(DM dm);

PETSC_EXTERN PetscErrorCode DMClone_Forest(DM dm, DM *newdm)
{
  DM_Forest      *forest = (DM_Forest*) dm->data;
  const char     *type;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  forest->refct++;
  (*newdm)->data = forest;
  ierr           = PetscObjectGetType((PetscObject) dm, &type);CHKERRQ(ierr);
  ierr           = PetscObjectChangeTypeName((PetscObject) *newdm, type);CHKERRQ(ierr);
  ierr           = DMInitialize_Forest(*newdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMDestroy_Forest(DM dm)
{
  DM_Forest      *forest = (DM_Forest*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (--forest->refct > 0) PetscFunctionReturn(0);
  if (forest->destroy) {ierr = (*forest->destroy)(dm);CHKERRQ(ierr);}
  ierr = PetscSFDestroy(&forest->cellSF);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&forest->preCoarseToFine);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&forest->coarseToPreFine);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&forest->adaptLabel);CHKERRQ(ierr);
  ierr = PetscFree(forest->adaptStrategy);CHKERRQ(ierr);
  ierr = DMDestroy(&forest->base);CHKERRQ(ierr);
  ierr = DMDestroy(&forest->adapt);CHKERRQ(ierr);
  ierr = PetscFree(forest->topology);CHKERRQ(ierr);
  ierr = PetscFree(forest);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMForestSetTopology - Set the topology of a DMForest during the pre-setup phase.  The topology is a string (e.g.
  "cube", "shell") and can be interpreted by subtypes of DMFOREST) to construct the base DM of a forest during
  DMSetUp().

  Logically collective on dm

  Input parameters:
+ dm - the forest
- topology - the topology of the forest

  Level: intermediate

.seealso: DMForestGetTopology(), DMForestSetBaseDM()
@*/
PetscErrorCode DMForestSetTopology(DM dm, DMForestTopology topology)
{
  DM_Forest      *forest = (DM_Forest*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the topology after setup");
  ierr = PetscFree(forest->topology);CHKERRQ(ierr);
  ierr = PetscStrallocpy((const char*)topology,(char**) &forest->topology);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMForestGetTopology - Get a string describing the topology of a DMForest.

  Not collective

  Input parameter:
. dm - the forest

  Output parameter:
. topology - the topology of the forest (e.g., 'cube', 'shell')

  Level: intermediate

.seealso: DMForestSetTopology()
@*/
PetscErrorCode DMForestGetTopology(DM dm, DMForestTopology *topology)
{
  DM_Forest *forest = (DM_Forest*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(topology,2);
  *topology = forest->topology;
  PetscFunctionReturn(0);
}

/*@
  DMForestSetBaseDM - During the pre-setup phase, set the DM that defines the base mesh of a DMForest forest.  The
  forest will be hierarchically refined from the base, and all refinements/coarsenings of the forest will share its
  base.  In general, two forest must share a base to be comparable, to do things like construct interpolators.

  Logically collective on dm

  Input Parameters:
+ dm - the forest
- base - the base DM of the forest

  Notes:
    Currently the base DM must be a DMPLEX

  Level: intermediate

.seealso: DMForestGetBaseDM()
@*/
PetscErrorCode DMForestSetBaseDM(DM dm, DM base)
{
  DM_Forest      *forest = (DM_Forest*) dm->data;
  PetscInt       dim, dimEmbed;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the base after setup");
  ierr         = PetscObjectReference((PetscObject)base);CHKERRQ(ierr);
  ierr         = DMDestroy(&forest->base);CHKERRQ(ierr);
  forest->base = base;
  if (base) {
    PetscBool        isper;
    const PetscReal *maxCell, *L;
    const DMBoundaryType *bd;

    PetscValidHeaderSpecific(base, DM_CLASSID, 2);
    ierr = DMGetDimension(base,&dim);CHKERRQ(ierr);
    ierr = DMSetDimension(dm,dim);CHKERRQ(ierr);
    ierr = DMGetCoordinateDim(base,&dimEmbed);CHKERRQ(ierr);
    ierr = DMSetCoordinateDim(dm,dimEmbed);CHKERRQ(ierr);
    ierr = DMGetPeriodicity(base,&isper,&maxCell,&L,&bd);CHKERRQ(ierr);
    ierr = DMSetPeriodicity(dm,isper,maxCell,L,bd);CHKERRQ(ierr);
  } else {
    ierr = DMSetPeriodicity(dm,PETSC_FALSE,NULL,NULL,NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  DMForestGetBaseDM - Get the base DM of a DMForest forest.  The forest will be hierarchically refined from the base,
  and all refinements/coarsenings of the forest will share its base.  In general, two forest must share a base to be
  comparable, to do things like construct interpolators.

  Not collective

  Input Parameter:
. dm - the forest

  Output Parameter:
. base - the base DM of the forest

  Notes:
    After DMSetUp(), the base DM will be redundantly distributed across MPI processes

  Level: intermediate

.seealso: DMForestSetBaseDM()
@*/
PetscErrorCode DMForestGetBaseDM(DM dm, DM *base)
{
  DM_Forest *forest = (DM_Forest*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(base, 2);
  *base = forest->base;
  PetscFunctionReturn(0);
}

PetscErrorCode DMForestSetBaseCoordinateMapping(DM dm, PetscErrorCode (*func)(DM,PetscInt,PetscInt,const PetscReal [],PetscReal [],void*),void *ctx)
{
  DM_Forest *forest = (DM_Forest*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  forest->mapcoordinates    = func;
  forest->mapcoordinatesctx = ctx;
  PetscFunctionReturn(0);
}

PetscErrorCode DMForestGetBaseCoordinateMapping(DM dm, PetscErrorCode (**func) (DM,PetscInt,PetscInt,const PetscReal [],PetscReal [],void*),void *ctx)
{
  DM_Forest *forest = (DM_Forest*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (func) *func = forest->mapcoordinates;
  if (ctx) *((void**) ctx) = forest->mapcoordinatesctx;
  PetscFunctionReturn(0);
}

/*@
  DMForestSetAdaptivityForest - During the pre-setup phase, set the forest from which the current forest will be
  adapted (e.g., the current forest will be refined/coarsened/repartitioned from it) im DMSetUp().  Usually not needed
  by users directly: DMForestTemplate() constructs a new forest to be adapted from an old forest and calls this
  routine.

  Note that this can be called after setup with adapt = NULL, which will clear all internal data related to the
  adaptivity forest from dm.  This way, repeatedly adapting does not leave stale DM objects in memory.

  Logically collective on dm

  Input Parameter:
+ dm - the new forest, which will be constructed from adapt
- adapt - the old forest

  Level: intermediate

.seealso: DMForestGetAdaptivityForest(), DMForestSetAdaptivityPurpose()
@*/
PetscErrorCode DMForestSetAdaptivityForest(DM dm,DM adapt)
{
  DM_Forest      *forest, *adaptForest, *oldAdaptForest;
  DM             oldAdapt;
  PetscBool      isForest;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (adapt) PetscValidHeaderSpecific(adapt, DM_CLASSID, 2);
  ierr = DMIsForest(dm, &isForest);CHKERRQ(ierr);
  if (!isForest) PetscFunctionReturn(0);
  if (adapt != NULL && dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the adaptation forest after setup");
  forest         = (DM_Forest*) dm->data;
  ierr           = DMForestGetAdaptivityForest(dm,&oldAdapt);CHKERRQ(ierr);
  adaptForest    = (DM_Forest*) (adapt ? adapt->data : NULL);
  oldAdaptForest = (DM_Forest*) (oldAdapt ? oldAdapt->data : NULL);
  if (adaptForest != oldAdaptForest) {
    ierr = PetscSFDestroy(&forest->preCoarseToFine);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&forest->coarseToPreFine);CHKERRQ(ierr);
    if (forest->clearadaptivityforest) {ierr = (*forest->clearadaptivityforest)(dm);CHKERRQ(ierr);}
  }
  switch (forest->adaptPurpose) {
  case DM_ADAPT_DETERMINE:
    ierr          = PetscObjectReference((PetscObject)adapt);CHKERRQ(ierr);
    ierr          = DMDestroy(&(forest->adapt));CHKERRQ(ierr);
    forest->adapt = adapt;
    break;
  case DM_ADAPT_REFINE:
    ierr = DMSetCoarseDM(dm,adapt);CHKERRQ(ierr);
    break;
  case DM_ADAPT_COARSEN:
  case DM_ADAPT_COARSEN_LAST:
    ierr = DMSetFineDM(dm,adapt);CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"invalid adaptivity purpose");
  }
  PetscFunctionReturn(0);
}

/*@
  DMForestGetAdaptivityForest - Get the forest from which the current forest is adapted.

  Not collective

  Input Parameter:
. dm - the forest

  Output Parameter:
. adapt - the forest from which dm is/was adapted

  Level: intermediate

.seealso: DMForestSetAdaptivityForest(), DMForestSetAdaptivityPurpose()
@*/
PetscErrorCode DMForestGetAdaptivityForest(DM dm, DM *adapt)
{
  DM_Forest      *forest;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  forest = (DM_Forest*) dm->data;
  switch (forest->adaptPurpose) {
  case DM_ADAPT_DETERMINE:
    *adapt = forest->adapt;
    break;
  case DM_ADAPT_REFINE:
    ierr = DMGetCoarseDM(dm,adapt);CHKERRQ(ierr);
    break;
  case DM_ADAPT_COARSEN:
  case DM_ADAPT_COARSEN_LAST:
    ierr = DMGetFineDM(dm,adapt);CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"invalid adaptivity purpose");
  }
  PetscFunctionReturn(0);
}

/*@
  DMForestSetAdaptivityPurpose - During the pre-setup phase, set whether the current DM is being adapted from its
  source (set with DMForestSetAdaptivityForest()) for the purpose of refinement (DM_ADAPT_REFINE), coarsening
  (DM_ADAPT_COARSEN), or undefined (DM_ADAPT_DETERMINE).  This only matters for the purposes of reference counting:
  during DMDestroy(), cyclic references can be found between DMs only if the cyclic reference is due to a fine/coarse
  relationship (see DMSetFineDM()/DMSetCoarseDM()).  If the purpose is not refinement or coarsening, and the user does
  not maintain a reference to the post-adaptation forest (i.e., the one created by DMForestTemplate()), then this can
  cause a memory leak.  This method is used by subtypes of DMForest when automatically constructing mesh hierarchies.

  Logically collective on dm

  Input Parameters:
+ dm - the forest
- purpose - the adaptivity purpose

  Level: advanced

.seealso: DMForestTemplate(), DMForestSetAdaptivityForest(), DMForestGetAdaptivityForest(), DMAdaptFlag
@*/
PetscErrorCode DMForestSetAdaptivityPurpose(DM dm, DMAdaptFlag purpose)
{
  DM_Forest      *forest;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  forest = (DM_Forest*) dm->data;
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the adaptation forest after setup");
  if (purpose != forest->adaptPurpose) {
    DM adapt;

    ierr = DMForestGetAdaptivityForest(dm,&adapt);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)adapt);CHKERRQ(ierr);
    ierr = DMForestSetAdaptivityForest(dm,NULL);CHKERRQ(ierr);

    forest->adaptPurpose = purpose;

    ierr = DMForestSetAdaptivityForest(dm,adapt);CHKERRQ(ierr);
    ierr = DMDestroy(&adapt);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  DMForestGetAdaptivityPurpose - Get whether the current DM is being adapted from its source (set with
  DMForestSetAdaptivityForest()) for the purpose of refinement (DM_ADAPT_REFINE), coarsening (DM_ADAPT_COARSEN),
  coarsening only the last level (DM_ADAPT_COARSEN_LAST) or undefined (DM_ADAPT_DETERMINE).
  This only matters for the purposes of reference counting: during DMDestroy(), cyclic
  references can be found between DMs only if the cyclic reference is due to a fine/coarse relationship (see
  DMSetFineDM()/DMSetCoarseDM()).  If the purpose is not refinement or coarsening, and the user does not maintain a
  reference to the post-adaptation forest (i.e., the one created by DMForestTemplate()), then this can cause a memory
  leak.  This method is used by subtypes of DMForest when automatically constructing mesh hierarchies.

  Not collective

  Input Parameter:
. dm - the forest

  Output Parameter:
. purpose - the adaptivity purpose

  Level: advanced

.seealso: DMForestTemplate(), DMForestSetAdaptivityForest(), DMForestGetAdaptivityForest(), DMAdaptFlag
@*/
PetscErrorCode DMForestGetAdaptivityPurpose(DM dm, DMAdaptFlag *purpose)
{
  DM_Forest *forest;

  PetscFunctionBegin;
  forest   = (DM_Forest*) dm->data;
  *purpose = forest->adaptPurpose;
  PetscFunctionReturn(0);
}

/*@
  DMForestSetAdjacencyDimension - During the pre-setup phase, set the dimension of interface points that determine
  cell adjacency (for the purposes of partitioning and overlap).

  Logically collective on dm

  Input Parameters:
+ dm - the forest
- adjDim - default 0 (i.e., vertices determine adjacency)

  Level: intermediate

.seealso: DMForestGetAdjacencyDimension(), DMForestSetAdjacencyCodimension(), DMForestSetPartitionOverlap()
@*/
PetscErrorCode DMForestSetAdjacencyDimension(DM dm, PetscInt adjDim)
{
  PetscInt       dim;
  DM_Forest      *forest = (DM_Forest*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the adjacency dimension after setup");
  if (adjDim < 0) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"adjacency dim cannot be < 0: %d", adjDim);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if (adjDim > dim) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"adjacency dim cannot be > %d: %d", dim, adjDim);
  forest->adjDim = adjDim;
  PetscFunctionReturn(0);
}

/*@
  DMForestSetAdjacencyCodimension - Like DMForestSetAdjacencyDimension(), but specified as a co-dimension (so that,
  e.g., adjacency based on facets can be specified by codimension 1 in all cases)

  Logically collective on dm

  Input Parameters:
+ dm - the forest
- adjCodim - default isthe dimension of the forest (see DMGetDimension()), since this is the codimension of vertices

  Level: intermediate

.seealso: DMForestGetAdjacencyCodimension(), DMForestSetAdjacencyDimension()
@*/
PetscErrorCode DMForestSetAdjacencyCodimension(DM dm, PetscInt adjCodim)
{
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMForestSetAdjacencyDimension(dm,dim-adjCodim);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMForestGetAdjacencyDimension - Get the dimension of interface points that determine cell adjacency (for the
  purposes of partitioning and overlap).

  Not collective

  Input Parameter:
. dm - the forest

  Output Parameter:
. adjDim - default 0 (i.e., vertices determine adjacency)

  Level: intermediate

.seealso: DMForestSetAdjacencyDimension(), DMForestGetAdjacencyCodimension(), DMForestSetPartitionOverlap()
@*/
PetscErrorCode DMForestGetAdjacencyDimension(DM dm, PetscInt *adjDim)
{
  DM_Forest *forest = (DM_Forest*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(adjDim,2);
  *adjDim = forest->adjDim;
  PetscFunctionReturn(0);
}

/*@
  DMForestGetAdjacencyCodimension - Like DMForestGetAdjacencyDimension(), but specified as a co-dimension (so that,
  e.g., adjacency based on facets can be specified by codimension 1 in all cases)

  Not collective

  Input Parameter:
. dm - the forest

  Output Parameter:
. adjCodim - default isthe dimension of the forest (see DMGetDimension()), since this is the codimension of vertices

  Level: intermediate

.seealso: DMForestSetAdjacencyCodimension(), DMForestGetAdjacencyDimension()
@*/
PetscErrorCode DMForestGetAdjacencyCodimension(DM dm, PetscInt *adjCodim)
{
  DM_Forest      *forest = (DM_Forest*) dm->data;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(adjCodim,2);
  ierr      = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  *adjCodim = dim - forest->adjDim;
  PetscFunctionReturn(0);
}

/*@
  DMForestSetPartitionOverlap - During the pre-setup phase, set the amount of cell-overlap present in parallel
  partitions of a forest, with values > 0 indicating subdomains that are expanded by that many iterations of adding
  adjacent cells

  Logically collective on dm

  Input Parameters:
+ dm - the forest
- overlap - default 0

  Level: intermediate

.seealso: DMForestGetPartitionOverlap(), DMForestSetAdjacencyDimension(), DMForestSetAdjacencyCodimension()
@*/
PetscErrorCode DMForestSetPartitionOverlap(DM dm, PetscInt overlap)
{
  DM_Forest *forest = (DM_Forest*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the overlap after setup");
  if (overlap < 0) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"overlap cannot be < 0: %d", overlap);
  forest->overlap = overlap;
  PetscFunctionReturn(0);
}

/*@
  DMForestGetPartitionOverlap - Get the amount of cell-overlap present in parallel partitions of a forest, with values
  > 0 indicating subdomains that are expanded by that many iterations of adding adjacent cells

  Not collective

  Input Parameter:
. dm - the forest

  Output Parameter:
. overlap - default 0

  Level: intermediate

.seealso: DMForestGetPartitionOverlap(), DMForestSetAdjacencyDimension(), DMForestSetAdjacencyCodimension()
@*/
PetscErrorCode DMForestGetPartitionOverlap(DM dm, PetscInt *overlap)
{
  DM_Forest *forest = (DM_Forest*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(overlap,2);
  *overlap = forest->overlap;
  PetscFunctionReturn(0);
}

/*@
  DMForestSetMinimumRefinement - During the pre-setup phase, set the minimum level of refinement (relative to the base
  DM, see DMForestGetBaseDM()) allowed in the forest.  If the forest is being created by coarsening a previous forest
  (see DMForestGetAdaptivityForest()) this limits the amount of coarsening.

  Logically collective on dm

  Input Parameters:
+ dm - the forest
- minRefinement - default PETSC_DEFAULT (interpreted by the subtype of DMForest)

  Level: intermediate

.seealso: DMForestGetMinimumRefinement(), DMForestSetMaximumRefinement(), DMForestSetInitialRefinement(), DMForestGetBaseDM(), DMForestGetAdaptivityForest()
@*/
PetscErrorCode DMForestSetMinimumRefinement(DM dm, PetscInt minRefinement)
{
  DM_Forest *forest = (DM_Forest*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the minimum refinement after setup");
  forest->minRefinement = minRefinement;
  PetscFunctionReturn(0);
}

/*@
  DMForestGetMinimumRefinement - Get the minimum level of refinement (relative to the base DM, see
  DMForestGetBaseDM()) allowed in the forest.  If the forest is being created by coarsening a previous forest (see
  DMForestGetAdaptivityForest()), this limits the amount of coarsening.

  Not collective

  Input Parameter:
. dm - the forest

  Output Parameter:
. minRefinement - default PETSC_DEFAULT (interpreted by the subtype of DMForest)

  Level: intermediate

.seealso: DMForestSetMinimumRefinement(), DMForestGetMaximumRefinement(), DMForestGetInitialRefinement(), DMForestGetBaseDM(), DMForestGetAdaptivityForest()
@*/
PetscErrorCode DMForestGetMinimumRefinement(DM dm, PetscInt *minRefinement)
{
  DM_Forest *forest = (DM_Forest*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(minRefinement,2);
  *minRefinement = forest->minRefinement;
  PetscFunctionReturn(0);
}

/*@
  DMForestSetInitialRefinement - During the pre-setup phase, set the initial level of refinement (relative to the base
  DM, see DMForestGetBaseDM()) allowed in the forest.

  Logically collective on dm

  Input Parameters:
+ dm - the forest
- initefinement - default PETSC_DEFAULT (interpreted by the subtype of DMForest)

  Level: intermediate

.seealso: DMForestSetMinimumRefinement(), DMForestSetMaximumRefinement(), DMForestGetBaseDM()
@*/
PetscErrorCode DMForestSetInitialRefinement(DM dm, PetscInt initRefinement)
{
  DM_Forest *forest = (DM_Forest*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the initial refinement after setup");
  forest->initRefinement = initRefinement;
  PetscFunctionReturn(0);
}

/*@
  DMForestGetInitialRefinement - Get the initial level of refinement (relative to the base DM, see
  DMForestGetBaseDM()) allowed in the forest.

  Not collective

  Input Parameter:
. dm - the forest

  Output Paramater:
. initRefinement - default PETSC_DEFAULT (interpreted by the subtype of DMForest)

  Level: intermediate

.seealso: DMForestSetMinimumRefinement(), DMForestSetMaximumRefinement(), DMForestGetBaseDM()
@*/
PetscErrorCode DMForestGetInitialRefinement(DM dm, PetscInt *initRefinement)
{
  DM_Forest *forest = (DM_Forest*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(initRefinement,2);
  *initRefinement = forest->initRefinement;
  PetscFunctionReturn(0);
}

/*@
  DMForestSetMaximumRefinement - During the pre-setup phase, set the maximum level of refinement (relative to the base
  DM, see DMForestGetBaseDM()) allowed in the forest.  If the forest is being created by refining a previous forest
  (see DMForestGetAdaptivityForest()), this limits the amount of refinement.

  Logically collective on dm

  Input Parameters:
+ dm - the forest
- maxRefinement - default PETSC_DEFAULT (interpreted by the subtype of DMForest)

  Level: intermediate

.seealso: DMForestGetMinimumRefinement(), DMForestSetMaximumRefinement(), DMForestSetInitialRefinement(), DMForestGetBaseDM(), DMForestGetAdaptivityDM()
@*/
PetscErrorCode DMForestSetMaximumRefinement(DM dm, PetscInt maxRefinement)
{
  DM_Forest *forest = (DM_Forest*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the maximum refinement after setup");
  forest->maxRefinement = maxRefinement;
  PetscFunctionReturn(0);
}

/*@
  DMForestGetMaximumRefinement - Get the maximum level of refinement (relative to the base DM, see
  DMForestGetBaseDM()) allowed in the forest.  If the forest is being created by refining a previous forest (see
  DMForestGetAdaptivityForest()), this limits the amount of refinement.

  Not collective

  Input Parameter:
. dm - the forest

  Output Parameter:
. maxRefinement - default PETSC_DEFAULT (interpreted by the subtype of DMForest)

  Level: intermediate

.seealso: DMForestSetMaximumRefinement(), DMForestGetMinimumRefinement(), DMForestGetInitialRefinement(), DMForestGetBaseDM(), DMForestGetAdaptivityForest()
@*/
PetscErrorCode DMForestGetMaximumRefinement(DM dm, PetscInt *maxRefinement)
{
  DM_Forest *forest = (DM_Forest*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(maxRefinement,2);
  *maxRefinement = forest->maxRefinement;
  PetscFunctionReturn(0);
}

/*@C
  DMForestSetAdaptivityStrategy - During the pre-setup phase, set the strategy for combining adaptivity labels from multiple processes.
  Subtypes of DMForest may define their own strategies.  Two default strategies are DMFORESTADAPTALL, which indicates that all processes must agree
  for a refinement/coarsening flag to be valid, and DMFORESTADAPTANY, which indicates that only one process needs to
  specify refinement/coarsening.

  Logically collective on dm

  Input Parameters:
+ dm - the forest
- adaptStrategy - default DMFORESTADAPTALL

  Level: advanced

.seealso: DMForestGetAdaptivityStrategy()
@*/
PetscErrorCode DMForestSetAdaptivityStrategy(DM dm, DMForestAdaptivityStrategy adaptStrategy)
{
  DM_Forest      *forest = (DM_Forest*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscFree(forest->adaptStrategy);CHKERRQ(ierr);
  ierr = PetscStrallocpy((const char*) adaptStrategy,(char**)&forest->adaptStrategy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMForestSetAdaptivityStrategy - Get the strategy for combining adaptivity labels from multiple processes.  Subtypes
  of DMForest may define their own strategies.  Two default strategies are DMFORESTADAPTALL, which indicates that all
  processes must agree for a refinement/coarsening flag to be valid, and DMFORESTADAPTANY, which indicates that only
  one process needs to specify refinement/coarsening.

  Not collective

  Input Parameter:
. dm - the forest

  Output Parameter:
. adaptStrategy - the adaptivity strategy (default DMFORESTADAPTALL)

  Level: advanced

.seealso: DMForestSetAdaptivityStrategy()
@*/
PetscErrorCode DMForestGetAdaptivityStrategy(DM dm, DMForestAdaptivityStrategy *adaptStrategy)
{
  DM_Forest *forest = (DM_Forest*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(adaptStrategy,2);
  *adaptStrategy = forest->adaptStrategy;
  PetscFunctionReturn(0);
}

/*@
  DMForestGetAdaptivitySuccess - Return whether the requested adaptation (refinement, coarsening, repartitioning,
  etc.) was successful.  PETSC_FALSE indicates that the post-adaptation forest is the same as the pre-adpatation
  forest.  A requested adaptation may have been unsuccessful if, for example, the requested refinement would have
  exceeded the maximum refinement level.

  Collective on dm

  Input Parameter:

. dm - the post-adaptation forest

  Output Parameter:

. success - PETSC_TRUE if the post-adaptation forest is different from the pre-adaptation forest.

  Level: intermediate

.see
@*/
PetscErrorCode DMForestGetAdaptivitySuccess(DM dm, PetscBool *success)
{
  DM_Forest      *forest;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (!dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"DMSetUp() has not been called yet.");
  forest = (DM_Forest *) dm->data;
  ierr = (forest->getadaptivitysuccess)(dm,success);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMForestSetComputeAdaptivitySF - During the pre-setup phase, set whether transfer PetscSFs should be computed
  relating the cells of the pre-adaptation forest to the post-adaptiation forest.  After DMSetUp() is called, these transfer PetscSFs can be accessed with DMForestGetAdaptivitySF().

  Logically collective on dm

  Input Parameters:
+ dm - the post-adaptation forest
- computeSF - default PETSC_TRUE

  Level: advanced

.seealso: DMForestGetComputeAdaptivitySF(), DMForestGetAdaptivitySF()
@*/
PetscErrorCode DMForestSetComputeAdaptivitySF(DM dm, PetscBool computeSF)
{
  DM_Forest *forest;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot compute adaptivity PetscSFs after setup is called");
  forest                 = (DM_Forest*) dm->data;
  forest->computeAdaptSF = computeSF;
  PetscFunctionReturn(0);
}

PetscErrorCode DMForestTransferVec(DM dmIn, Vec vecIn, DM dmOut, Vec vecOut, PetscBool useBCs, PetscReal time)
{
  DM_Forest      *forest;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dmIn   ,DM_CLASSID  ,1);
  PetscValidHeaderSpecific(vecIn  ,VEC_CLASSID ,2);
  PetscValidHeaderSpecific(dmOut  ,DM_CLASSID  ,3);
  PetscValidHeaderSpecific(vecOut ,VEC_CLASSID ,4);
  forest = (DM_Forest *) dmIn->data;
  if (!forest->transfervec) SETERRQ(PetscObjectComm((PetscObject)dmIn),PETSC_ERR_SUP,"DMForestTransferVec() not implemented");
  ierr = (forest->transfervec)(dmIn,vecIn,dmOut,vecOut,useBCs,time);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMForestTransferVecFromBase(DM dm, Vec vecIn, Vec vecOut)
{
  DM_Forest      *forest;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm   ,DM_CLASSID  ,1);
  PetscValidHeaderSpecific(vecIn  ,VEC_CLASSID ,2);
  PetscValidHeaderSpecific(vecOut ,VEC_CLASSID ,3);
  forest = (DM_Forest *) dm->data;
  if (!forest->transfervecfrombase) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"DMForestTransferVecFromBase() not implemented");
  ierr = (forest->transfervecfrombase)(dm,vecIn,vecOut);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  DMForestGetComputeAdaptivitySF - Get whether transfer PetscSFs should be computed relating the cells of the
  pre-adaptation forest to the post-adaptiation forest.  After DMSetUp() is called, these transfer PetscSFs can be
  accessed with DMForestGetAdaptivitySF().

  Not collective

  Input Parameter:
. dm - the post-adaptation forest

  Output Parameter:
. computeSF - default PETSC_TRUE

  Level: advanced

.seealso: DMForestSetComputeAdaptivitySF(), DMForestGetAdaptivitySF()
@*/
PetscErrorCode DMForestGetComputeAdaptivitySF(DM dm, PetscBool *computeSF)
{
  DM_Forest *forest;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  forest     = (DM_Forest*) dm->data;
  *computeSF = forest->computeAdaptSF;
  PetscFunctionReturn(0);
}

/*@
  DMForestGetAdaptivitySF - Get PetscSFs that relate the pre-adaptation forest to the post-adaptation forest.
  Adaptation can be any combination of refinement, coarsening, repartition, and change of overlap, so there may be
  some cells of the pre-adaptation that are parents of post-adaptation cells, and vice versa.  Therefore there are two
  PetscSFs: one that relates pre-adaptation coarse cells to post-adaptation fine cells, and one that relates
  pre-adaptation fine cells to post-adaptation coarse cells.

  Not collective

  Input Parameter:
  dm - the post-adaptation forest

  Output Parameter:
  preCoarseToFine - pre-adaptation coarse cells to post-adaptation fine cells: BCast goes from pre- to post-
  coarseToPreFine - post-adaptation coarse cells to pre-adaptation fine cells: BCast goes from post- to pre-

  Level: advanced

.seealso: DMForestGetComputeAdaptivitySF(), DMForestSetComputeAdaptivitySF()
@*/
PetscErrorCode DMForestGetAdaptivitySF(DM dm, PetscSF *preCoarseToFine, PetscSF *coarseToPreFine)
{
  DM_Forest      *forest;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr   = DMSetUp(dm);CHKERRQ(ierr);
  forest = (DM_Forest*) dm->data;
  if (preCoarseToFine) *preCoarseToFine = forest->preCoarseToFine;
  if (coarseToPreFine) *coarseToPreFine = forest->coarseToPreFine;
  PetscFunctionReturn(0);
}

/*@
  DMForestSetGradeFactor - During the pre-setup phase, set the desired amount of grading in the mesh, e.g. give 2 to
  indicate that the diameter of neighboring cells should differ by at most a factor of 2.  Subtypes of DMForest may
  only support one particular choice of grading factor.

  Logically collective on dm

  Input Parameters:
+ dm - the forest
- grade - the grading factor

  Level: advanced

.seealso: DMForestGetGradeFactor()
@*/
PetscErrorCode DMForestSetGradeFactor(DM dm, PetscInt grade)
{
  DM_Forest *forest = (DM_Forest*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the grade factor after setup");
  forest->gradeFactor = grade;
  PetscFunctionReturn(0);
}

/*@
  DMForestGetGradeFactor - Get the desired amount of grading in the mesh, e.g. give 2 to indicate that the diameter of
  neighboring cells should differ by at most a factor of 2.  Subtypes of DMForest may only support one particular
  choice of grading factor.

  Not collective

  Input Parameter:
. dm - the forest

  Output Parameter:
. grade - the grading factor

  Level: advanced

.seealso: DMForestSetGradeFactor()
@*/
PetscErrorCode DMForestGetGradeFactor(DM dm, PetscInt *grade)
{
  DM_Forest *forest = (DM_Forest*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(grade,2);
  *grade = forest->gradeFactor;
  PetscFunctionReturn(0);
}

/*@
  DMForestSetCellWeightFactor - During the pre-setup phase, set the factor by which the level of refinement changes
  the cell weight (see DMForestSetCellWeights()) when calculating partitions.  The final weight of a cell will be
  (cellWeight) * (weightFactor^refinementLevel).  A factor of 1 indicates that the weight of a cell does not depend on
  its level; a factor of 2, for example, might be appropriate for sub-cycling time-stepping methods, when the
  computation associated with a cell is multiplied by a factor of 2 for each additional level of refinement.

  Logically collective on dm

  Input Parameters:
+ dm - the forest
- weightsFactors - default 1.

  Level: advanced

.seealso: DMForestGetCellWeightFactor(), DMForestSetCellWeights()
@*/
PetscErrorCode DMForestSetCellWeightFactor(DM dm, PetscReal weightsFactor)
{
  DM_Forest *forest = (DM_Forest*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the weights factor after setup");
  forest->weightsFactor = weightsFactor;
  PetscFunctionReturn(0);
}

/*@
  DMForestGetCellWeightFactor - Get the factor by which the level of refinement changes the cell weight (see
  DMForestSetCellWeights()) when calculating partitions.  The final weight of a cell will be (cellWeight) *
  (weightFactor^refinementLevel).  A factor of 1 indicates that the weight of a cell does not depend on its level; a
  factor of 2, for example, might be appropriate for sub-cycling time-stepping methods, when the computation
  associated with a cell is multiplied by a factor of 2 for each additional level of refinement.

  Not collective

  Input Parameter:
. dm - the forest

  Output Parameter:
. weightsFactors - default 1.

  Level: advanced

.seealso: DMForestSetCellWeightFactor(), DMForestSetCellWeights()
@*/
PetscErrorCode DMForestGetCellWeightFactor(DM dm, PetscReal *weightsFactor)
{
  DM_Forest *forest = (DM_Forest*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidRealPointer(weightsFactor,2);
  *weightsFactor = forest->weightsFactor;
  PetscFunctionReturn(0);
}

/*@
  DMForestGetCellChart - After the setup phase, get the local half-open interval of the chart of cells on this process

  Not collective

  Input Parameter:
. dm - the forest

  Output Parameters:
+ cStart - the first cell on this process
- cEnd - one after the final cell on this process

  Level: intermediate

.seealso: DMForestGetCellSF()
@*/
PetscErrorCode DMForestGetCellChart(DM dm, PetscInt *cStart, PetscInt *cEnd)
{
  DM_Forest      *forest = (DM_Forest*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(cStart,2);
  PetscValidIntPointer(cEnd,2);
  if (((forest->cStart == PETSC_DETERMINE) || (forest->cEnd == PETSC_DETERMINE)) && forest->createcellchart) {
    ierr = forest->createcellchart(dm,&forest->cStart,&forest->cEnd);CHKERRQ(ierr);
  }
  *cStart =  forest->cStart;
  *cEnd   =  forest->cEnd;
  PetscFunctionReturn(0);
}

/*@
  DMForestGetCellSF - After the setup phase, get the PetscSF for overlapping cells between processes

  Not collective

  Input Parameter:
. dm - the forest

  Output Parameter:
. cellSF - the PetscSF

  Level: intermediate

.seealso: DMForestGetCellChart()
@*/
PetscErrorCode DMForestGetCellSF(DM dm, PetscSF *cellSF)
{
  DM_Forest      *forest = (DM_Forest*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(cellSF,2);
  if ((!forest->cellSF) && forest->createcellsf) {
    ierr = forest->createcellsf(dm,&forest->cellSF);CHKERRQ(ierr);
  }
  *cellSF = forest->cellSF;
  PetscFunctionReturn(0);
}

/*@C
  DMForestSetAdaptivityLabel - During the pre-setup phase, set the label of the pre-adaptation forest (see
  DMForestGetAdaptivityForest()) that holds the adaptation flags (refinement, coarsening, or some combination).  The
  interpretation of the label values is up to the subtype of DMForest, but DM_ADAPT_DETERMINE, DM_ADAPT_KEEP,
  DM_ADAPT_REFINE, and DM_ADAPT_COARSEN have been reserved as choices that should be accepted by all subtypes.

  Logically collective on dm

  Input Parameters:
- dm - the forest
+ adaptLabel - the label in the pre-adaptation forest

  Level: intermediate

.seealso DMForestGetAdaptivityLabel()
@*/
PetscErrorCode DMForestSetAdaptivityLabel(DM dm, DMLabel adaptLabel)
{
  DM_Forest      *forest = (DM_Forest*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (adaptLabel) PetscValidHeaderSpecific(adaptLabel,DMLABEL_CLASSID,2);
  ierr = PetscObjectReference((PetscObject)adaptLabel);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&forest->adaptLabel);CHKERRQ(ierr);
  forest->adaptLabel = adaptLabel;
  PetscFunctionReturn(0);
}

/*@C
  DMForestGetAdaptivityLabel - Get the label of the pre-adaptation forest (see DMForestGetAdaptivityForest()) that
  holds the adaptation flags (refinement, coarsening, or some combination).  The interpretation of the label values is
  up to the subtype of DMForest, but DM_ADAPT_DETERMINE, DM_ADAPT_KEEP, DM_ADAPT_REFINE, and DM_ADAPT_COARSEN have
  been reserved as choices that should be accepted by all subtypes.

  Not collective

  Input Parameter:
. dm - the forest

  Output Parameter:
. adaptLabel - the name of the label in the pre-adaptation forest

  Level: intermediate

.seealso DMForestSetAdaptivityLabel()
@*/
PetscErrorCode DMForestGetAdaptivityLabel(DM dm, DMLabel *adaptLabel)
{
  DM_Forest *forest = (DM_Forest*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  *adaptLabel = forest->adaptLabel;
  PetscFunctionReturn(0);
}

/*@
  DMForestSetCellWeights - Set the weights assigned to each of the cells (see DMForestGetCellChart()) of the current
  process: weights are used to determine parallel partitioning.  Partitions will be created so that each process's
  ratio of weight to capacity (see DMForestSetWeightCapacity()) is roughly equal. If NULL, each cell receives a weight
  of 1.

  Logically collective on dm

  Input Parameters:
+ dm - the forest
. weights - the array of weights for all cells, or NULL to indicate each cell has weight 1.
- copyMode - how weights should reference weights

  Level: advanced

.seealso: DMForestGetCellWeights(), DMForestSetWeightCapacity()
@*/
PetscErrorCode DMForestSetCellWeights(DM dm, PetscReal weights[], PetscCopyMode copyMode)
{
  DM_Forest      *forest = (DM_Forest*) dm->data;
  PetscInt       cStart, cEnd;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMForestGetCellChart(dm,&cStart,&cEnd);CHKERRQ(ierr);
  if (cEnd < cStart) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"cell chart [%d,%d) is not valid",cStart,cEnd);
  if (copyMode == PETSC_COPY_VALUES) {
    if (forest->cellWeightsCopyMode != PETSC_OWN_POINTER || forest->cellWeights == weights) {
      ierr = PetscMalloc1(cEnd-cStart,&forest->cellWeights);CHKERRQ(ierr);
    }
    ierr                        = PetscArraycpy(forest->cellWeights,weights,cEnd-cStart);CHKERRQ(ierr);
    forest->cellWeightsCopyMode = PETSC_OWN_POINTER;
    PetscFunctionReturn(0);
  }
  if (forest->cellWeightsCopyMode == PETSC_OWN_POINTER) {
    ierr = PetscFree(forest->cellWeights);CHKERRQ(ierr);
  }
  forest->cellWeights         = weights;
  forest->cellWeightsCopyMode = copyMode;
  PetscFunctionReturn(0);
}

/*@
  DMForestGetCellWeights - Get the weights assigned to each of the cells (see DMForestGetCellChart()) of the current
  process: weights are used to determine parallel partitioning.  Partitions will be created so that each process's
  ratio of weight to capacity (see DMForestSetWeightCapacity()) is roughly equal. If NULL, each cell receives a weight
  of 1.

  Not collective

  Input Parameter:
. dm - the forest

  Output Parameter:
. weights - the array of weights for all cells, or NULL to indicate each cell has weight 1.

  Level: advanced

.seealso: DMForestSetCellWeights(), DMForestSetWeightCapacity()
@*/
PetscErrorCode DMForestGetCellWeights(DM dm, PetscReal **weights)
{
  DM_Forest *forest = (DM_Forest*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(weights,2);
  *weights = forest->cellWeights;
  PetscFunctionReturn(0);
}

/*@
  DMForestSetWeightCapacity - During the pre-setup phase, set the capacity of the current process when repartitioning
  a pre-adaptation forest (see DMForestGetAdaptivityForest()).  After partitioning, the ratio of the weight of each
  process's cells to the process's capacity will be roughly equal for all processes.  A capacity of 0 indicates that
  the current process should not have any cells after repartitioning.

  Logically Collective on dm

  Input parameters:
+ dm - the forest
- capacity - this process's capacity

  Level: advanced

.seealso DMForestGetWeightCapacity(), DMForestSetCellWeights(), DMForestSetCellWeightFactor()
@*/
PetscErrorCode DMForestSetWeightCapacity(DM dm, PetscReal capacity)
{
  DM_Forest *forest = (DM_Forest*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the weight capacity after setup");
  if (capacity < 0.) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Cannot have negative weight capacity; %f",capacity);
  forest->weightCapacity = capacity;
  PetscFunctionReturn(0);
}

/*@
  DMForestGetWeightCapacity - Set the capacity of the current process when repartitioning a pre-adaptation forest (see
  DMForestGetAdaptivityForest()).  After partitioning, the ratio of the weight of each process's cells to the
  process's capacity will be roughly equal for all processes.  A capacity of 0 indicates that the current process
  should not have any cells after repartitioning.

  Not collective

  Input parameter:
. dm - the forest

  Output parameter:
. capacity - this process's capacity

  Level: advanced

.seealso DMForestSetWeightCapacity(), DMForestSetCellWeights(), DMForestSetCellWeightFactor()
@*/
PetscErrorCode DMForestGetWeightCapacity(DM dm, PetscReal *capacity)
{
  DM_Forest *forest = (DM_Forest*) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidRealPointer(capacity,2);
  *capacity = forest->weightCapacity;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMSetFromOptions_Forest(PetscOptionItems *PetscOptionsObject,DM dm)
{
  PetscBool                  flg, flg1, flg2, flg3, flg4;
  DMForestTopology           oldTopo;
  char                       stringBuffer[256];
  PetscViewer                viewer;
  PetscViewerFormat          format;
  PetscInt                   adjDim, adjCodim, overlap, minRefinement, initRefinement, maxRefinement, grade;
  PetscReal                  weightsFactor;
  DMForestAdaptivityStrategy adaptStrategy;
  PetscErrorCode             ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr                         = DMForestGetTopology(dm, &oldTopo);CHKERRQ(ierr);
  ierr                         = PetscOptionsHead(PetscOptionsObject,"DMForest Options");CHKERRQ(ierr);
  ierr                         = PetscOptionsString("-dm_forest_topology","the topology of the forest's base mesh","DMForestSetTopology",oldTopo,stringBuffer,sizeof(stringBuffer),&flg1);CHKERRQ(ierr);
  ierr                         = PetscOptionsViewer("-dm_forest_base_dm","load the base DM from a viewer specification","DMForestSetBaseDM",&viewer,&format,&flg2);CHKERRQ(ierr);
  ierr                         = PetscOptionsViewer("-dm_forest_coarse_forest","load the coarse forest from a viewer specification","DMForestSetCoarseForest",&viewer,&format,&flg3);CHKERRQ(ierr);
  ierr                         = PetscOptionsViewer("-dm_forest_fine_forest","load the fine forest from a viewer specification","DMForestSetFineForest",&viewer,&format,&flg4);CHKERRQ(ierr);
  if ((PetscInt) flg1 + (PetscInt) flg2 + (PetscInt) flg3 + (PetscInt) flg4 > 1) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_INCOMP,"Specify only one of -dm_forest_{topology,base_dm,coarse_forest,fine_forest}");
  if (flg1) {
    ierr = DMForestSetTopology(dm,(DMForestTopology)stringBuffer);CHKERRQ(ierr);
    ierr = DMForestSetBaseDM(dm,NULL);CHKERRQ(ierr);
    ierr = DMForestSetAdaptivityForest(dm,NULL);CHKERRQ(ierr);
  }
  if (flg2) {
    DM base;

    ierr = DMCreate(PetscObjectComm((PetscObject)dm),&base);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = DMLoad(base,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = DMForestSetBaseDM(dm,base);CHKERRQ(ierr);
    ierr = DMDestroy(&base);CHKERRQ(ierr);
    ierr = DMForestSetTopology(dm,NULL);CHKERRQ(ierr);
    ierr = DMForestSetAdaptivityForest(dm,NULL);CHKERRQ(ierr);
  }
  if (flg3) {
    DM coarse;

    ierr = DMCreate(PetscObjectComm((PetscObject)dm),&coarse);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = DMLoad(coarse,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = DMForestSetAdaptivityForest(dm,coarse);CHKERRQ(ierr);
    ierr = DMDestroy(&coarse);CHKERRQ(ierr);
    ierr = DMForestSetTopology(dm,NULL);CHKERRQ(ierr);
    ierr = DMForestSetBaseDM(dm,NULL);CHKERRQ(ierr);
  }
  if (flg4) {
    DM fine;

    ierr = DMCreate(PetscObjectComm((PetscObject)dm),&fine);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = DMLoad(fine,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = DMForestSetAdaptivityForest(dm,fine);CHKERRQ(ierr);
    ierr = DMDestroy(&fine);CHKERRQ(ierr);
    ierr = DMForestSetTopology(dm,NULL);CHKERRQ(ierr);
    ierr = DMForestSetBaseDM(dm,NULL);CHKERRQ(ierr);
  }
  ierr = DMForestGetAdjacencyDimension(dm,&adjDim);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-dm_forest_adjacency_dimension","set the dimension of points that define adjacency in the forest","DMForestSetAdjacencyDimension",adjDim,&adjDim,&flg,0);CHKERRQ(ierr);
  if (flg) {
    ierr = DMForestSetAdjacencyDimension(dm,adjDim);CHKERRQ(ierr);
  } else {
    ierr = DMForestGetAdjacencyCodimension(dm,&adjCodim);CHKERRQ(ierr);
    ierr = PetscOptionsBoundedInt("-dm_forest_adjacency_codimension","set the codimension of points that define adjacency in the forest","DMForestSetAdjacencyCodimension",adjCodim,&adjCodim,&flg,1);CHKERRQ(ierr);
    if (flg) {
      ierr = DMForestSetAdjacencyCodimension(dm,adjCodim);CHKERRQ(ierr);
    }
  }
  ierr = DMForestGetPartitionOverlap(dm,&overlap);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-dm_forest_partition_overlap","set the degree of partition overlap","DMForestSetPartitionOverlap",overlap,&overlap,&flg,0);CHKERRQ(ierr);
  if (flg) {
    ierr = DMForestSetPartitionOverlap(dm,overlap);CHKERRQ(ierr);
  }
#if 0
  ierr = PetscOptionsBoundedInt("-dm_refine","equivalent to -dm_forest_set_minimum_refinement and -dm_forest_set_initial_refinement with the same value",NULL,minRefinement,&minRefinement,&flg,0);CHKERRQ(ierr);
  if (flg) {
    ierr = DMForestSetMinimumRefinement(dm,minRefinement);CHKERRQ(ierr);
    ierr = DMForestSetInitialRefinement(dm,minRefinement);CHKERRQ(ierr);
  }
  ierr = PetscOptionsBoundedInt("-dm_refine_hierarchy","equivalent to -dm_forest_set_minimum_refinement 0 and -dm_forest_set_initial_refinement",NULL,initRefinement,&initRefinement,&flg,0);CHKERRQ(ierr);
  if (flg) {
    ierr = DMForestSetMinimumRefinement(dm,0);CHKERRQ(ierr);
    ierr = DMForestSetInitialRefinement(dm,initRefinement);CHKERRQ(ierr);
  }
#endif
  ierr = DMForestGetMinimumRefinement(dm,&minRefinement);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-dm_forest_minimum_refinement","set the minimum level of refinement in the forest","DMForestSetMinimumRefinement",minRefinement,&minRefinement,&flg,0);CHKERRQ(ierr);
  if (flg) {
    ierr = DMForestSetMinimumRefinement(dm,minRefinement);CHKERRQ(ierr);
  }
  ierr = DMForestGetInitialRefinement(dm,&initRefinement);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-dm_forest_initial_refinement","set the initial level of refinement in the forest","DMForestSetInitialRefinement",initRefinement,&initRefinement,&flg,0);CHKERRQ(ierr);
  if (flg) {
    ierr = DMForestSetInitialRefinement(dm,initRefinement);CHKERRQ(ierr);
  }
  ierr = DMForestGetMaximumRefinement(dm,&maxRefinement);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-dm_forest_maximum_refinement","set the maximum level of refinement in the forest","DMForestSetMaximumRefinement",maxRefinement,&maxRefinement,&flg,0);CHKERRQ(ierr);
  if (flg) {
    ierr = DMForestSetMaximumRefinement(dm,maxRefinement);CHKERRQ(ierr);
  }
  ierr = DMForestGetAdaptivityStrategy(dm,&adaptStrategy);CHKERRQ(ierr);
  ierr = PetscOptionsString("-dm_forest_adaptivity_strategy","the forest's adaptivity-flag resolution strategy","DMForestSetAdaptivityStrategy",adaptStrategy,stringBuffer,sizeof(stringBuffer),&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMForestSetAdaptivityStrategy(dm,(DMForestAdaptivityStrategy)stringBuffer);CHKERRQ(ierr);
  }
  ierr = DMForestGetGradeFactor(dm,&grade);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-dm_forest_grade_factor","grade factor between neighboring cells","DMForestSetGradeFactor",grade,&grade,&flg,0);CHKERRQ(ierr);
  if (flg) {
    ierr = DMForestSetGradeFactor(dm,grade);CHKERRQ(ierr);
  }
  ierr = DMForestGetCellWeightFactor(dm,&weightsFactor);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dm_forest_cell_weight_factor","multiplying weight factor for cell refinement","DMForestSetCellWeightFactor",weightsFactor,&weightsFactor,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMForestSetCellWeightFactor(dm,weightsFactor);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCreateSubDM_Forest(DM dm, PetscInt numFields, const PetscInt fields[], IS *is, DM *subdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (subdm) {ierr = DMClone(dm, subdm);CHKERRQ(ierr);}
  ierr = DMCreateSectionSubDM(dm, numFields, fields, is, subdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMRefine_Forest(DM dm, MPI_Comm comm, DM *dmRefined)
{
  DMLabel        refine;
  DM             fineDM;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetFineDM(dm,&fineDM);CHKERRQ(ierr);
  if (fineDM) {
    ierr       = PetscObjectReference((PetscObject)fineDM);CHKERRQ(ierr);
    *dmRefined = fineDM;
    PetscFunctionReturn(0);
  }
  ierr = DMForestTemplate(dm,comm,dmRefined);CHKERRQ(ierr);
  ierr = DMGetLabel(dm,"refine",&refine);CHKERRQ(ierr);
  if (!refine) {
    ierr = DMLabelCreate(PETSC_COMM_SELF, "refine",&refine);CHKERRQ(ierr);
    ierr = DMLabelSetDefaultValue(refine,DM_ADAPT_REFINE);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectReference((PetscObject) refine);CHKERRQ(ierr);
  }
  ierr = DMForestSetAdaptivityLabel(*dmRefined,refine);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&refine);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMCoarsen_Forest(DM dm, MPI_Comm comm, DM *dmCoarsened)
{
  DMLabel        coarsen;
  DM             coarseDM;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  {
    PetscMPIInt mpiComparison;
    MPI_Comm    dmcomm = PetscObjectComm((PetscObject)dm);

    ierr = MPI_Comm_compare(comm, dmcomm, &mpiComparison);CHKERRQ(ierr);
    if (mpiComparison != MPI_IDENT && mpiComparison != MPI_CONGRUENT) SETERRQ(dmcomm,PETSC_ERR_SUP,"No support for different communicators yet");
  }
  ierr = DMGetCoarseDM(dm,&coarseDM);CHKERRQ(ierr);
  if (coarseDM) {
    ierr         = PetscObjectReference((PetscObject)coarseDM);CHKERRQ(ierr);
    *dmCoarsened = coarseDM;
    PetscFunctionReturn(0);
  }
  ierr = DMForestTemplate(dm,comm,dmCoarsened);CHKERRQ(ierr);
  ierr = DMForestSetAdaptivityPurpose(*dmCoarsened,DM_ADAPT_COARSEN);CHKERRQ(ierr);
  ierr = DMGetLabel(dm,"coarsen",&coarsen);CHKERRQ(ierr);
  if (!coarsen) {
    ierr = DMLabelCreate(PETSC_COMM_SELF, "coarsen",&coarsen);CHKERRQ(ierr);
    ierr = DMLabelSetDefaultValue(coarsen,DM_ADAPT_COARSEN);CHKERRQ(ierr);
  } else {
    ierr = PetscObjectReference((PetscObject) coarsen);CHKERRQ(ierr);
  }
  ierr = DMForestSetAdaptivityLabel(*dmCoarsened,coarsen);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&coarsen);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMAdaptLabel_Forest(DM dm, DMLabel label, DM *adaptedDM)
{
  PetscBool      success;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMForestTemplate(dm,PetscObjectComm((PetscObject)dm),adaptedDM);CHKERRQ(ierr);
  ierr = DMForestSetAdaptivityLabel(*adaptedDM,label);CHKERRQ(ierr);
  ierr = DMSetUp(*adaptedDM);CHKERRQ(ierr);
  ierr = DMForestGetAdaptivitySuccess(*adaptedDM,&success);CHKERRQ(ierr);
  if (!success) {
    ierr = DMDestroy(adaptedDM);CHKERRQ(ierr);
    *adaptedDM = NULL;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMInitialize_Forest(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemzero(dm->ops,sizeof(*(dm->ops)));CHKERRQ(ierr);

  dm->ops->clone          = DMClone_Forest;
  dm->ops->setfromoptions = DMSetFromOptions_Forest;
  dm->ops->destroy        = DMDestroy_Forest;
  dm->ops->createsubdm    = DMCreateSubDM_Forest;
  dm->ops->refine         = DMRefine_Forest;
  dm->ops->coarsen        = DMCoarsen_Forest;
  dm->ops->adaptlabel     = DMAdaptLabel_Forest;
  PetscFunctionReturn(0);
}

/*MC

     DMFOREST = "forest" - A DM object that encapsulates a hierarchically refined mesh.  Forests usually have a base DM
  (see DMForestGetBaseDM()), from which it is refined.  The refinement and partitioning of forests is considered
  immutable after DMSetUp() is called.  To adapt a mesh, one should call DMForestTemplate() to create a new mesh that
  will default to being identical to it, specify how that mesh should differ, and then calling DMSetUp() on the new
  mesh.

  To specify that a mesh should be refined or coarsened from the previous mesh, a label should be defined on the
  previous mesh whose values indicate which cells should be refined (DM_ADAPT_REFINE) or coarsened (DM_ADAPT_COARSEN)
  and how (subtypes are free to allow additional values for things like anisotropic refinement).  The label should be
  given to the *new* mesh with DMForestSetAdaptivityLabel().

  Level: advanced

.seealso: DMType, DMCreate(), DMSetType(), DMForestGetBaseDM(), DMForestSetBaseDM(), DMForestTemplate(), DMForestSetAdaptivityLabel()
M*/

PETSC_EXTERN PetscErrorCode DMCreate_Forest(DM dm)
{
  DM_Forest      *forest;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr                         = PetscNewLog(dm,&forest);CHKERRQ(ierr);
  dm->dim                      = 0;
  dm->data                     = forest;
  forest->refct                = 1;
  forest->data                 = NULL;
  forest->topology             = NULL;
  forest->adapt                = NULL;
  forest->base                 = NULL;
  forest->adaptPurpose         = DM_ADAPT_DETERMINE;
  forest->adjDim               = PETSC_DEFAULT;
  forest->overlap              = PETSC_DEFAULT;
  forest->minRefinement        = PETSC_DEFAULT;
  forest->maxRefinement        = PETSC_DEFAULT;
  forest->initRefinement       = PETSC_DEFAULT;
  forest->cStart               = PETSC_DETERMINE;
  forest->cEnd                 = PETSC_DETERMINE;
  forest->cellSF               = NULL;
  forest->adaptLabel           = NULL;
  forest->gradeFactor          = 2;
  forest->cellWeights          = NULL;
  forest->cellWeightsCopyMode  = PETSC_USE_POINTER;
  forest->weightsFactor        = 1.;
  forest->weightCapacity       = 1.;
  ierr                         = DMForestSetAdaptivityStrategy(dm,DMFORESTADAPTALL);CHKERRQ(ierr);
  ierr                         = DMInitialize_Forest(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
