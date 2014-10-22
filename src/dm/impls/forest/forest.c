#include <petsc-private/dmforestimpl.h>
#include <petsc-private/dmimpl.h>
#include <petscsf.h>                     /*I "petscsf.h" */

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
PetscErrorCode DMDestroy_Forest(DM dm)
{
  DM_Forest     *forest = (DM_Forest*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (--forest->refct > 0) PetscFunctionReturn(0);
  ierr = PetscSFDestroy(&forest->cellSF);CHKERRQ(ierr);
  if (forest->adaptCopyMode == PETSC_OWN_POINTER) {
    ierr = PetscFree(forest->adaptMarkers);CHKERRQ(ierr);
  }
  if (forest->cellWeightsCopyMode == PETSC_OWN_POINTER) {
    ierr = PetscFree(forest->cellWeights);CHKERRQ(ierr);
  }
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
  if (forest->setup) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the topology after setup");
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
  if (forest->setup) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the base after setup");
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
  DM_Forest        *forest = (DM_Forest *) dm->data;
  DM               base;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  if (forest->setup) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the coarse forest after setup");
  ierr = PetscObjectReference((PetscObject)coarse);CHKERRQ(ierr);
  ierr = DMDestroy(&forest->coarse);CHKERRQ(ierr);
  ierr = DMForestGetBaseDM(coarse,&base);CHKERRQ(ierr);
  ierr = DMForestSetBaseDM(dm,base);CHKERRQ(ierr);
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
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(dm, DM_CLASSID, 2);
  if (forest->setup) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the fine forest after setup");
  ierr = PetscObjectReference((PetscObject)fine);CHKERRQ(ierr);
  ierr = DMDestroy(&forest->fine);CHKERRQ(ierr);
  ierr = DMForestGetBaseDM(fine,&base);CHKERRQ(ierr);
  ierr = DMForestSetBaseDM(dm,base);CHKERRQ(ierr);
  forest->fine = fine;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestGetFineForest_Forest"
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
  if (forest->setup) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the adjacency dimension after setup");
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
#define __FUNCT__ "DMForestSetParititionOverlap"
PetscErrorCode DMForestSetPartitionOverlap(DM dm, PetscInt overlap)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (forest->setup) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the overlap after setup");
  if (overlap < 0) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_OUTOFRANGE,"overlap cannot be < 0: %d", overlap);
  forest->overlap = overlap;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestGetPartitionOverlap"
PetscErrorCode DMForestGetPartitionOverlap (DM dm, PetscInt *overlap)
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
PetscErrorCode DMForestSetMinimumRefinement (DM dm, PetscInt minRefinement)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (forest->setup) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the minimum refinement after setup");
  forest->minRefinement = minRefinement;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestGetMinimumRefinement"
PetscErrorCode DMForestGetMinimumRefinement (DM dm, PetscInt *minRefinement)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(minRefinement,2);
  *minRefinement = forest->minRefinement;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestSetMinimumRefinement"
PetscErrorCode DMForestSetMinimumRefinement (DM dm, PetscInt minRefinement)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (forest->setup) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot change the minimum refinement after setup");
  forest->minRefinement = minRefinement;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMForestGetMinimumRefinement"
PetscErrorCode DMForestGetMinimumRefinement (DM dm, PetscInt *minRefinement)
{
  DM_Forest      *forest = (DM_Forest *) dm->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidIntPointer(minRefinement,2);
  *minRefinement = forest->minRefinement;
  PetscFunctionReturn(0);
}
#undef __FUNCT__
#define __FUNCT__ "DMSetFromOptions_Forest"
PetscErrorCode DMSetFromFromOptions_Forest(DM dm)
{
  DM_Forest        *forest = (DM_Forest *) dm->data;
  PetscBool        flg;
  DMForestTopology oldTopo;
  char             topology[256];
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscInt         adjDim, adjCodim, overlap, minRefinement;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  forest->setFromOptions = PETSC_TRUE;
  ierr = PetscOptionsHead("DMForest Options");CHKERRQ(ierr);
  ierr = DMForestGetTopology(dm, &oldTopo);CHKERRQ(ierr);
  ierr = PetscOptionsString("-dm_forest_topology","the topology of the forest's base mesh","DMForestSetTopology",oldTopo,topology,256,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMForestSetTopology(dm,(DMForestTopology)topology);CHKERRQ(ierr);
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
  ierr = PetscOptionsTail();CHKERRQ(ierr);
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
  forest->setup               = 0;
  forest->setFromOptions      = PETSC_FALSE;
  forest->topology            = NULL;
  forest->base                = NULL;
  forest->coarse              = NULL;
  forest->fine                = NULL;
  forest->adjDim              = PETSC_DEFAULT;
  forest->overlap             = PETSC_DEFAULT;
  forest->minRefinement       = PETSC_DEFAULT;
  forest->maxRefinement       = PETSC_DEFAULT;
  forest->cStart              = 0;
  forest->cEnd                = 0;
  forest->cellSF              = 0;
  forest->adaptMarkers        = NULL;
  forest->adaptCopyMode       = PETSC_USE_POINTER;
  forest->adaptStrategy       = DMFORESTADAPTALL;
  forest->gradeFactor         = 2;
  forest->cellWeights         = NULL;
  forest->cellWeightsCopyMode = PETSC_USE_POINTER;
  forest->weightsFactor       = 1.;
  forest->weightCapacity      = 1.;
  PetscFunctionReturn(0);
}

