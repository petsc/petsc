#include <petsc/private/dmforestimpl.h> /*I petscdmforest.h I*/
#include <petsc/private/dmimpl.h>       /*I petscdm.h */
#include <petscsf.h>

PetscBool DMForestPackageInitialized = PETSC_FALSE;

typedef struct _DMForestTypeLink *DMForestTypeLink;

struct _DMForestTypeLink
{
  char *name;
  DMForestTypeLink next;
};

DMForestTypeLink DMForestTypeList;

#undef __FUNCT__
#define __FUNCT__ "DMForestPackageFinalize"
static PetscErrorCode DMForestPackageFinalize(void)
{
  DMForestTypeLink oldLink, link = DMForestTypeList;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  while (link) {
    oldLink = link;
    ierr = PetscFree(oldLink->name);
    link = oldLink->next;
    ierr = PetscFree(oldLink);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestPackageInitialize"
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

#undef __FUNCT__
#define __FUNCT__ "DMForestRegisterType"
PetscErrorCode DMForestRegisterType(DMType name)
{
  DMForestTypeLink link;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMForestPackageInitialize();CHKERRQ(ierr);
  ierr = PetscNew(&link);CHKERRQ(ierr);
  ierr = PetscStrallocpy(name,&link->name);CHKERRQ(ierr);
  link->next = DMForestTypeList;
  DMForestTypeList = link;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMIsForest"
PetscErrorCode DMIsForest(DM dm, PetscBool *isForest)
{
  DMForestTypeLink link = DMForestTypeList;
  PetscErrorCode ierr;

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

#undef __FUNCT__
#define __FUNCT__ "DMForestTemplate"
PETSC_EXTERN PetscErrorCode DMForestTemplate(DM dm, DM tdm)
{
  DM_Forest        *forest = (DM_Forest *) dm->data;
  DM               base;
  DMForestTopology topology;
  PetscInt         dim, overlap, ref, factor;
  DMForestAdaptivityStrategy strat;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(tdm, DM_CLASSID, 2);
  ierr = DMForestGetBaseDM(dm,&base);CHKERRQ(ierr);
  ierr = DMForestSetBaseDM(tdm,base);CHKERRQ(ierr);
  ierr = DMForestGetTopology(dm,&topology);CHKERRQ(ierr);
  ierr = DMForestSetTopology(tdm,topology);CHKERRQ(ierr);
  ierr = DMForestGetAdjacencyDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMForestSetAdjacencyDimension(tdm,dim);CHKERRQ(ierr);
  ierr = DMForestGetPartitionOverlap(dm,&overlap);CHKERRQ(ierr);
  ierr = DMForestSetPartitionOverlap(tdm,overlap);CHKERRQ(ierr);
  ierr = DMForestGetMinimumRefinement(dm,&ref);CHKERRQ(ierr);
  ierr = DMForestSetMinimumRefinement(tdm,ref);CHKERRQ(ierr);
  ierr = DMForestGetMaximumRefinement(dm,&ref);CHKERRQ(ierr);
  ierr = DMForestSetMaximumRefinement(tdm,ref);CHKERRQ(ierr);
  ierr = DMForestGetAdaptivityStrategy(dm,&strat);CHKERRQ(ierr);
  ierr = DMForestSetAdaptivityStrategy(tdm,strat);CHKERRQ(ierr);
  ierr = DMForestGetGradeFactor(dm,&factor);CHKERRQ(ierr);
  ierr = DMForestSetGradeFactor(tdm,factor);CHKERRQ(ierr);
  if (forest->ftemplate) {
    ierr = (forest->ftemplate) (dm, tdm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMInitialize_Forest(DM dm);

#undef __FUNCT__
#define __FUNCT__ "DMClone_Forest"
PETSC_EXTERN PetscErrorCode DMClone_Forest(DM dm, DM *newdm)
{
  DM_Forest        *forest = (DM_Forest *) dm->data;
  const char       *type;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  forest->refct++;
  (*newdm)->data = forest;
  ierr = PetscObjectGetType((PetscObject) dm, &type);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) *newdm, type);CHKERRQ(ierr);
  ierr = DMInitialize_Forest(*newdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMDestroy_Forest"
static PetscErrorCode DMDestroy_Forest(DM dm)
{
  DM_Forest     *forest = (DM_Forest*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (--forest->refct > 0) PetscFunctionReturn(0);
  if (forest->destroy) {ierr = forest->destroy(dm);CHKERRQ(ierr);}
  ierr = PetscSFDestroy(&forest->cellSF);CHKERRQ(ierr);
  if (forest->adaptCopyMode == PETSC_OWN_POINTER) {
    ierr = PetscFree(forest->adaptMarkers);CHKERRQ(ierr);
  }
  if (forest->cellWeightsCopyMode == PETSC_OWN_POINTER) {
    ierr = PetscFree(forest->cellWeights);CHKERRQ(ierr);
  }
  ierr = PetscFree(forest->adaptStrategy);CHKERRQ(ierr);
  ierr = DMDestroy(&forest->base);CHKERRQ(ierr);
  ierr = PetscFree(forest->topology);CHKERRQ(ierr);
  ierr = PetscFree(forest);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestSetTopology"
PetscErrorCode DMForestSetTopology(DM dm, DMForestTopology topology)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the topology after setup");
  ierr = PetscFree(forest->topology);CHKERRQ(ierr);
  ierr = PetscStrallocpy((const char *)topology,(char **) &forest->topology);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestGetTopology"
PetscErrorCode DMForestGetTopology(DM dm, DMForestTopology *topology)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(topology,2);
  *topology = forest->topology;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestSetBaseDM"
PetscErrorCode DMForestSetBaseDM(DM dm, DM base)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;
  PetscInt       dim, dimEmbed;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the base after setup");
  ierr = PetscObjectReference((PetscObject)base);CHKERRQ(ierr);
  ierr = DMDestroy(&forest->base);CHKERRQ(ierr);
  forest->base = base;
  if (base) {
    PetscValidHeaderSpecific(base, DM_CLASSID, 2);
    ierr = DMGetDimension(base,&dim);CHKERRQ(ierr);
    ierr = DMSetDimension(dm,dim);CHKERRQ(ierr);
    ierr = DMGetCoordinateDim(base,&dimEmbed);CHKERRQ(ierr);
    ierr = DMSetCoordinateDim(dm,dimEmbed);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestGetBaseDM"
PetscErrorCode DMForestGetBaseDM(DM dm, DM *base)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(base, 2);
  *base = forest->base;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestSetCoarseForest"
PetscErrorCode DMForestSetCoarseForest(DM dm,DM coarse)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the coarse forest after setup");
  ierr = DMSetCoarseDM(dm,coarse);CHKERRQ(ierr);
  if (coarse) {
    PetscValidHeaderSpecific(coarse, DM_CLASSID, 2);
    ierr = DMForestTemplate(coarse,dm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestGetCoarseForest"
PetscErrorCode DMForestGetCoarseForest(DM dm, DM *coarse)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetCoarseDM(dm,coarse);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestSetFineForest"
PetscErrorCode DMForestSetFineForest(DM dm,DM fine)
{
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the fine forest after setup");
  ierr = DMSetFineDM(dm,fine);CHKERRQ(ierr);
  if (fine) {
    PetscValidHeaderSpecific(fine, DM_CLASSID, 2);
    ierr = DMForestTemplate(fine,dm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestGetFineForest"
PetscErrorCode DMForestGetFineForest(DM dm, DM *fine)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetFineDM(dm,fine);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestSetAdjacencyDimension"
PetscErrorCode DMForestSetAdjacencyDimension(DM dm, PetscInt adjDim)
{
  PetscInt        dim;
  DM_Forest      *forest = (DM_Forest *) dm->data;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the adjacency dimension after setup");
  if (adjDim < 0) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"adjacency dim cannot be < 0: %d", adjDim);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if (adjDim > dim) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"adjacency dim cannot be > %d: %d", dim, adjDim);
  forest->adjDim = adjDim;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestSetAdjacencyCodimension"
PetscErrorCode DMForestSetAdjacencyCodimension(DM dm, PetscInt adjCodim)
{
  PetscInt        dim;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  ierr = DMForestSetAdjacencyDimension(dm,dim-adjCodim);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestGetAdjacencyDimension"
PetscErrorCode DMForestGetAdjacencyDimension(DM dm, PetscInt *adjDim)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(adjDim,2);
  *adjDim = forest->adjDim;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestGetAdjacencyCodimension"
PetscErrorCode DMForestGetAdjacencyCodimension(DM dm, PetscInt *adjCodim)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(adjCodim,2);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  *adjCodim = dim - forest->adjDim;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestSetPartitionOverlap"
PetscErrorCode DMForestSetPartitionOverlap(DM dm, PetscInt overlap)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the overlap after setup");
  if (overlap < 0) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"overlap cannot be < 0: %d", overlap);
  forest->overlap = overlap;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestGetPartitionOverlap"
PetscErrorCode DMForestGetPartitionOverlap(DM dm, PetscInt *overlap)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(overlap,2);
  *overlap = forest->overlap;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestSetMinimumRefinement"
PetscErrorCode DMForestSetMinimumRefinement(DM dm, PetscInt minRefinement)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the minimum refinement after setup");
  forest->minRefinement = minRefinement;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestGetMinimumRefinement"
PetscErrorCode DMForestGetMinimumRefinement(DM dm, PetscInt *minRefinement)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(minRefinement,2);
  *minRefinement = forest->minRefinement;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestSetInitialRefinement"
PetscErrorCode DMForestSetInitialRefinement(DM dm, PetscInt initRefinement)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the initial refinement after setup");
  forest->initRefinement = initRefinement;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestGetInitialRefinement"
PetscErrorCode DMForestGetInitialRefinement(DM dm, PetscInt *initRefinement)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(initRefinement,2);
  *initRefinement = forest->initRefinement;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestSetMaximumRefinement"
PetscErrorCode DMForestSetMaximumRefinement(DM dm, PetscInt maxRefinement)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the maximum refinement after setup");
  forest->maxRefinement = maxRefinement;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestGetMaximumRefinement"
PetscErrorCode DMForestGetMaximumRefinement(DM dm, PetscInt *maxRefinement)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(maxRefinement,2);
  *maxRefinement = forest->maxRefinement;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestSetAdaptivityStrategy"
PetscErrorCode DMForestSetAdaptivityStrategy(DM dm, DMForestAdaptivityStrategy adaptStrategy)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr = PetscFree(forest->adaptStrategy);CHKERRQ(ierr);
  ierr = PetscStrallocpy((const char *) adaptStrategy,(char **)&forest->adaptStrategy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestGetAdaptivityStrategy"
PetscErrorCode DMForestGetAdaptivityStrategy(DM dm, DMForestAdaptivityStrategy *adaptStrategy)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(adaptStrategy,2);
  *adaptStrategy = forest->adaptStrategy;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestSetGradeFactor"
PetscErrorCode DMForestSetGradeFactor(DM dm, PetscInt grade)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the grade factor after setup");
  forest->gradeFactor = grade;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestGetGradeFactor"
PetscErrorCode DMForestGetGradeFactor(DM dm, PetscInt *grade)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(grade,2);
  *grade = forest->gradeFactor;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestSetCellWeightFactor"
PetscErrorCode DMForestSetCellWeightFactor(DM dm, PetscReal weightsFactor)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the weights factor after setup");
  forest->weightsFactor = weightsFactor;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestGetCellWeightFactor"
PetscErrorCode DMForestGetCellWeightFactor(DM dm, PetscReal *weightsFactor)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidRealPointer(weightsFactor,2);
  *weightsFactor = forest->weightsFactor;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestGetCellChart"
PetscErrorCode DMForestGetCellChart(DM dm, PetscInt *cStart, PetscInt *cEnd)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;
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

#undef __FUNCT__
#define __FUNCT__ "DMForestGetCellSF"
PetscErrorCode DMForestGetCellSF(DM dm, PetscSF *cellSF)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;
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

#undef __FUNCT__
#define __FUNCT__ "DMForestSetAdaptivityMarkers"
PetscErrorCode DMForestSetAdaptivityMarkers(DM dm, PetscInt markers[], PetscCopyMode copyMode)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;
  PetscInt       cStart, cEnd;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMForestGetCellChart(dm,&cStart,&cEnd);CHKERRQ(ierr);
  if (cEnd < cStart) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"cell chart [%d,%d) is not valid",cStart,cEnd);
  if (copyMode == PETSC_COPY_VALUES) {
    if (forest->adaptCopyMode != PETSC_OWN_POINTER || forest->adaptMarkers == markers) {
      ierr = PetscMalloc1(cEnd-cStart,&forest->adaptMarkers);CHKERRQ(ierr);
    }
    ierr = PetscMemcpy(forest->adaptMarkers,markers,(cEnd-cStart)*sizeof(*markers));CHKERRQ(ierr);
    forest->adaptCopyMode = PETSC_OWN_POINTER;
    PetscFunctionReturn(0);
  }
  if (forest->adaptCopyMode == PETSC_OWN_POINTER) {
    ierr = PetscFree(forest->adaptMarkers);CHKERRQ(ierr);
  }
  forest->adaptMarkers  = markers;
  forest->adaptCopyMode = copyMode;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestGetAdaptivityMarkers"
PetscErrorCode DMForestGetAdaptivityMarkers(DM dm, PetscInt **markers)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(markers,2);
  *markers = forest->adaptMarkers;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestSetCellWeights"
PetscErrorCode DMForestSetCellWeights(DM dm, PetscReal weights[], PetscCopyMode copyMode)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;
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
    ierr = PetscMemcpy(forest->cellWeights,weights,(cEnd-cStart)*sizeof(*weights));CHKERRQ(ierr);
    forest->cellWeightsCopyMode = PETSC_OWN_POINTER;
    PetscFunctionReturn(0);
  }
  if (forest->cellWeightsCopyMode == PETSC_OWN_POINTER) {
    ierr = PetscFree(forest->cellWeights);CHKERRQ(ierr);
  }
  forest->cellWeights  = weights;
  forest->cellWeightsCopyMode = copyMode;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestGetCellWeights"
PetscErrorCode DMForestGetCellWeights(DM dm, PetscReal **weights)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(weights,2);
  *weights = forest->cellWeights;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestSetWeightCapacity"
PetscErrorCode DMForestSetWeightCapacity(DM dm, PetscReal capacity)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the weight capacity after setup");
  if (capacity < 0.) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"Cannot have negative weight capacity; %f",capacity);
  forest->weightCapacity = capacity;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestGetWeightCapacity"
PetscErrorCode DMForestGetWeightCapacity(DM dm, PetscReal *capacity)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidRealPointer(capacity,2);
  *capacity = forest->weightCapacity;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMSetFromOptions_Forest"
PETSC_EXTERN PetscErrorCode DMSetFromOptions_Forest(PetscOptionItems *PetscOptionsObject,DM dm)
{
  DM_Forest                  *forest = (DM_Forest *) dm->data;
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
  forest->setFromOptions = PETSC_TRUE;
  ierr = DMForestGetTopology(dm, &oldTopo);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"DMForest Options");CHKERRQ(ierr);
  ierr = PetscOptionsString("-dm_forest_topology","the topology of the forest's base mesh","DMForestSetTopology",oldTopo,stringBuffer,256,&flg1);CHKERRQ(ierr);
  ierr = PetscOptionsViewer("-dm_forest_base_dm","load the base DM from a viewer specification","DMForestSetBaseDM",&viewer,&format,&flg2);CHKERRQ(ierr);
  ierr = PetscOptionsViewer("-dm_forest_coarse_forest","load the coarse forest from a viewer specification","DMForestSetCoarseForest",&viewer,&format,&flg3);CHKERRQ(ierr);
  ierr = PetscOptionsViewer("-dm_forest_fine_forest","load the fine forest from a viewer specification","DMForestSetFineForest",&viewer,&format,&flg4);CHKERRQ(ierr);
  if ((PetscInt) flg1 + (PetscInt) flg2 + (PetscInt) flg3 + (PetscInt) flg4 > 1) {
    SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_INCOMP,"Specify only one of -dm_forest_{topology,base_dm,coarse_forest,fine_forest}");
  }
  if (flg1) {
    ierr = DMForestSetTopology(dm,(DMForestTopology)stringBuffer);CHKERRQ(ierr);
    ierr = DMForestSetBaseDM(dm,NULL);CHKERRQ(ierr);
    ierr = DMForestSetCoarseForest(dm,NULL);CHKERRQ(ierr);
    ierr = DMForestSetFineForest(dm,NULL);CHKERRQ(ierr);
  }
  if (flg2) {
    DM         base;

    ierr = DMCreate(PetscObjectComm((PetscObject)dm),&base);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = DMLoad(base,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = DMForestSetBaseDM(dm,base);CHKERRQ(ierr);
    ierr = DMDestroy(&base);CHKERRQ(ierr);
    ierr = DMForestSetTopology(dm,NULL);CHKERRQ(ierr);
    ierr = DMForestSetCoarseForest(dm,NULL);CHKERRQ(ierr);
    ierr = DMForestSetFineForest(dm,NULL);CHKERRQ(ierr);
  }
  if (flg3) {
    DM         coarse;

    ierr = DMCreate(PetscObjectComm((PetscObject)dm),&coarse);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = DMLoad(coarse,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = DMForestSetCoarseForest(dm,coarse);CHKERRQ(ierr);
    ierr = DMDestroy(&coarse);CHKERRQ(ierr);
    ierr = DMForestSetTopology(dm,NULL);CHKERRQ(ierr);
    ierr = DMForestSetBaseDM(dm,NULL);CHKERRQ(ierr);
    ierr = DMForestSetFineForest(dm,NULL);CHKERRQ(ierr);
  }
  if (flg4) {
    DM         fine;

    ierr = DMCreate(PetscObjectComm((PetscObject)dm),&fine);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = DMLoad(fine,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = DMForestSetFineForest(dm,fine);CHKERRQ(ierr);
    ierr = DMDestroy(&fine);CHKERRQ(ierr);
    ierr = DMForestSetTopology(dm,NULL);CHKERRQ(ierr);
    ierr = DMForestSetBaseDM(dm,NULL);CHKERRQ(ierr);
    ierr = DMForestSetCoarseForest(dm,NULL);CHKERRQ(ierr);
  }
  ierr = DMForestGetAdjacencyDimension(dm,&adjDim);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dm_forest_adjacency_dimension","set the dimension of points that define adjacency in the forest","DMForestSetAdjacencyDimension",adjDim,&adjDim,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMForestSetAdjacencyDimension(dm,adjDim);CHKERRQ(ierr);
  }
  else {
    ierr = DMForestGetAdjacencyCodimension(dm,&adjCodim);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-dm_forest_adjacency_codimension","set the codimension of points that define adjacency in the forest","DMForestSetAdjacencyCodimension",adjCodim,&adjCodim,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = DMForestSetAdjacencyCodimension(dm,adjCodim);CHKERRQ(ierr);
    }
  }
  ierr = DMForestGetPartitionOverlap(dm,&overlap);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dm_forest_partition_overlap","set the degree of partition overlap","DMForestSetPartitionOverlap",overlap,&overlap,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMForestSetPartitionOverlap(dm,overlap);CHKERRQ(ierr);
  }
#if 0
  ierr = PetscOptionsInt("-dm_refine","equivalent to -dm_forest_set_minimum_refinement and -dm_forest_set_initial_refinement with the same value",NULL,minRefinement,&minRefinement,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMForestSetMinimumRefinement(dm,minRefinement);CHKERRQ(ierr);
    ierr = DMForestSetInitialRefinement(dm,minRefinement);CHKERRQ(ierr);
  }
  ierr = PetscOptionsInt("-dm_refine_hierarchy","equivalent to -dm_forest_set_minimum_refinement 0 and -dm_forest_set_initial_refinement",NULL,initRefinement,&initRefinement,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMForestSetMinimumRefinement(dm,0);CHKERRQ(ierr);
    ierr = DMForestSetInitialRefinement(dm,initRefinement);CHKERRQ(ierr);
  }
#endif
  ierr = DMForestGetMinimumRefinement(dm,&minRefinement);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dm_forest_minimum_refinement","set the minimum level of refinement in the forest","DMForestSetMinimumRefinement",minRefinement,&minRefinement,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMForestSetMinimumRefinement(dm,minRefinement);CHKERRQ(ierr);
  }
  ierr = DMForestGetInitialRefinement(dm,&initRefinement);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dm_forest_initial_refinement","set the initial level of refinement in the forest","DMForestSetInitialRefinement",initRefinement,&initRefinement,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMForestSetInitialRefinement(dm,initRefinement);CHKERRQ(ierr);
  }
  ierr = DMForestGetMaximumRefinement(dm,&maxRefinement);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dm_forest_maximum_refinement","set the maximum level of refinement in the forest","DMForestSetMaximumRefinement",maxRefinement,&maxRefinement,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMForestSetMaximumRefinement(dm,maxRefinement);CHKERRQ(ierr);
  }
  ierr = DMForestGetAdaptivityStrategy(dm,&adaptStrategy);CHKERRQ(ierr);
  ierr = PetscOptionsString("-dm_forest_adaptivity_strategy","the forest's adaptivity-flag resolution strategy","DMForestSetAdaptivityStrategy",adaptStrategy,stringBuffer,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMForestSetAdaptivityStrategy(dm,(DMForestAdaptivityStrategy)stringBuffer);CHKERRQ(ierr);
  }
  ierr = DMForestGetGradeFactor(dm,&grade);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dm_forest_grade_factor","grade factor between neighboring cells","DMForestSetGradeFactor",grade,&grade,&flg);CHKERRQ(ierr);
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

#undef __FUNCT__
#define __FUNCT__ "DMCreateSubDM_Forest"
PetscErrorCode DMCreateSubDM_Forest(DM dm, PetscInt numFields, PetscInt fields[], IS *is, DM *subdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (subdm) {ierr = DMClone(dm, subdm);CHKERRQ(ierr);}
  ierr = DMCreateSubDM_Section_Private(dm, numFields, fields, is, subdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMInitialize_Forest"
static PetscErrorCode DMInitialize_Forest(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemzero(dm->ops,sizeof(*(dm->ops)));CHKERRQ(ierr);

  dm->ops->clone          = DMClone_Forest;
  dm->ops->setfromoptions = DMSetFromOptions_Forest;
  dm->ops->destroy        = DMDestroy_Forest;
  dm->ops->createsubdm    = DMCreateSubDM_Forest;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMCreate_Forest"
PETSC_EXTERN PetscErrorCode DMCreate_Forest(DM dm)
{
  DM_Forest      *forest;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  ierr                        = PetscNewLog(dm,&forest);CHKERRQ(ierr);
  dm->dim                     = 0;
  dm->data                    = forest;
  forest->refct               = 1;
  forest->data                = NULL;
  forest->setFromOptions      = PETSC_FALSE;
  forest->topology            = NULL;
  forest->base                = NULL;
  forest->adjDim              = PETSC_DEFAULT;
  forest->overlap             = PETSC_DEFAULT;
  forest->minRefinement       = PETSC_DEFAULT;
  forest->maxRefinement       = PETSC_DEFAULT;
  forest->initRefinement      = PETSC_DEFAULT;
  forest->cStart              = PETSC_DETERMINE;
  forest->cEnd                = PETSC_DETERMINE;
  forest->cellSF              = 0;
  forest->adaptMarkers        = NULL;
  forest->adaptCopyMode       = PETSC_USE_POINTER;
  forest->gradeFactor         = 2;
  forest->cellWeights         = NULL;
  forest->cellWeightsCopyMode = PETSC_USE_POINTER;
  forest->weightsFactor       = 1.;
  forest->weightCapacity      = 1.;
  ierr = DMForestSetAdaptivityStrategy(dm,DMFORESTADAPTALL);CHKERRQ(ierr);
  ierr = DMInitialize_Forest(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

