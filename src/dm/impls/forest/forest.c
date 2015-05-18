#include <petsc/private/dmforestimpl.h> /*I petscdmforest.h I*/
#include <petsc/private/dmimpl.h>       /*I petscdm.h */
#include <petscsf.h>

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
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the base after setup");
  ierr = PetscObjectReference((PetscObject)base);CHKERRQ(ierr);
  ierr = DMDestroy(&forest->base);CHKERRQ(ierr);
  forest->base = base;
  ierr = DMGetDimension(base,&dim);CHKERRQ(ierr);
  ierr = DMSetDimension(dm,dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(base,&dimEmbed);CHKERRQ(ierr);
  ierr = DMSetCoordinateDim(dm,dimEmbed);CHKERRQ(ierr);
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
  DM_Forest        *forest       = (DM_Forest *) dm->data;
  DM               base;
  DMForestTopology topology;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the coarse forest after setup");
  if (!coarse->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot set a coarse forest that is not set up");
  ierr = PetscObjectReference((PetscObject)coarse);CHKERRQ(ierr);
  ierr = DMDestroy(&forest->coarse);CHKERRQ(ierr);
  ierr = DMForestGetBaseDM(coarse,&base);CHKERRQ(ierr);
  ierr = DMForestSetBaseDM(dm,base);CHKERRQ(ierr);
  ierr = DMForestGetTopology(coarse,&topology);CHKERRQ(ierr);
  ierr = DMForestSetTopology(dm,topology);CHKERRQ(ierr);
  forest->coarse = coarse;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestGetCoarseForest"
PetscErrorCode DMForestGetCoarseForest(DM dm, DM *coarse)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(coarse, 2);
  *coarse = forest->coarse;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestSetFineForest"
PetscErrorCode DMForestSetFineForest(DM dm,DM fine)
{
  DM_Forest        *forest = (DM_Forest *) dm->data;
  DM               base;
  DMForestTopology topology;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  if (dm->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the fine forest after setup");
  if (!fine->setupcalled) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot set a fine forest that is not set up");
  ierr = PetscObjectReference((PetscObject)fine);CHKERRQ(ierr);
  ierr = DMDestroy(&forest->fine);CHKERRQ(ierr);
  ierr = DMForestGetBaseDM(fine,&base);CHKERRQ(ierr);
  ierr = DMForestSetBaseDM(dm,base);CHKERRQ(ierr);
  ierr = DMForestGetTopology(fine,&topology);CHKERRQ(ierr);
  ierr = DMForestSetTopology(dm,topology);CHKERRQ(ierr);
  forest->fine = fine;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestGetFineForest"
PetscErrorCode DMForestGetFineForest(DM dm, DM *fine)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(fine, 2);
  *fine = forest->fine;
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
  ierr = PetscStrallocpy((const char *)adaptStrategy,(char **)adaptStrategy);CHKERRQ(ierr);
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
PETSC_EXTERN PetscErrorCode DMSetFromOptions_Forest(PetscOptions *PetscOptionsObject,DM dm)
{
  DM_Forest                  *forest = (DM_Forest *) dm->data;
  PetscBool                  flg;
  DMForestTopology           oldTopo;
  char                       stringBuffer[256];
  PetscViewer                viewer;
  PetscViewerFormat          format;
  PetscInt                   adjDim, adjCodim, overlap, minRefinement, maxRefinement, grade;
  PetscReal                  weightsFactor;
  DMForestAdaptivityStrategy adaptStrategy;
  PetscErrorCode             ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  forest->setFromOptions = PETSC_TRUE;
  ierr = PetscOptionsHead(PetscOptionsObject,"DMForest Options");CHKERRQ(ierr);
  ierr = DMForestGetTopology(dm, &oldTopo);CHKERRQ(ierr);
  ierr = PetscOptionsString("-dm_forest_topology","the topology of the forest's base mesh","DMForestSetTopology",oldTopo,stringBuffer,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMForestSetTopology(dm,(DMForestTopology)stringBuffer);CHKERRQ(ierr);
  }
  ierr = PetscOptionsViewer("-dm_forest_base_dm","load the base DM from a viewer specification","DMForestSetBaseDM",&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg) {
    DM         base;

    ierr = DMCreate(PetscObjectComm((PetscObject)dm),&base);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = DMLoad(base,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = DMForestSetBaseDM(dm,base);CHKERRQ(ierr);
    ierr = DMDestroy(&base);CHKERRQ(ierr);
  }
  ierr = PetscOptionsViewer("-dm_forest_coarse_forest","load the coarse forest from a viewer specification","DMForestSetCoarseForest",&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg) {
    DM         coarse;

    ierr = DMCreate(PetscObjectComm((PetscObject)dm),&coarse);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = DMLoad(coarse,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = DMForestSetCoarseForest(dm,coarse);CHKERRQ(ierr);
    ierr = DMDestroy(&coarse);CHKERRQ(ierr);
  }
  ierr = PetscOptionsViewer("-dm_forest_fine_forest","load the fine forest from a viewer specification","DMForestSetFineForest",&viewer,&format,&flg);CHKERRQ(ierr);
  if (flg) {
    DM         fine;

    ierr = DMCreate(PetscObjectComm((PetscObject)dm),&fine);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = DMLoad(fine,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = DMForestSetFineForest(dm,fine);CHKERRQ(ierr);
    ierr = DMDestroy(&fine);CHKERRQ(ierr);
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
  ierr = DMForestGetMinimumRefinement(dm,&minRefinement);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dm_forest_minimum_refinement","set the minimum level of refinement in the forest","DMForestSetMinimumRefinement",minRefinement,&minRefinement,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMForestSetMinimumRefinement(dm,minRefinement);CHKERRQ(ierr);
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
#define __FUNCT__ "DMInitialize_Forest"
static PetscErrorCode DMInitialize_Forest(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemzero(dm->ops,sizeof(*(dm->ops)));CHKERRQ(ierr);

  dm->ops->clone          = DMClone_Forest;
  dm->ops->setfromoptions = DMSetFromOptions_Forest;
  dm->ops->destroy        = DMDestroy_Forest;
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
  forest->coarse              = NULL;
  forest->fine                = NULL;
  forest->adjDim              = PETSC_DEFAULT;
  forest->overlap             = PETSC_DEFAULT;
  forest->minRefinement       = PETSC_DEFAULT;
  forest->maxRefinement       = PETSC_DEFAULT;
  forest->cStart              = PETSC_DETERMINE;
  forest->cEnd                = PETSC_DETERMINE;
  forest->cellSF              = 0;
  forest->adaptMarkers        = NULL;
  forest->adaptCopyMode       = PETSC_USE_POINTER;
  forest->adaptStrategy       = DMFORESTADAPTALL;
  forest->gradeFactor         = 2;
  forest->cellWeights         = NULL;
  forest->cellWeightsCopyMode = PETSC_USE_POINTER;
  forest->weightsFactor       = 1.;
  forest->weightCapacity      = 1.;
  ierr = DMInitialize_Forest(dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

