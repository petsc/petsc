#include <petsc/private/dmimpl.h>
#include <petsc/private/dmforestimpl.h>
#include <petsc/private/dmpleximpl.h>
#include <petsc/private/dmlabelimpl.h>
#include <petsc/private/viewerimpl.h>
#include <../src/sys/classes/viewer/impls/vtk/vtkvimpl.h>
#include "petsc_p4est_package.h"

#if defined(PETSC_HAVE_P4EST)

/* we need two levels of macros to stringify the results of macro expansion */
#define _pforest_string(a) _pforest_string_internal(a)
#define _pforest_string_internal(a) #a

#if !defined(P4_TO_P8)
#include <p4est.h>
#include <p4est_extended.h>
#include <p4est_geometry.h>
#include <p4est_ghost.h>
#include <p4est_lnodes.h>
#include <p4est_vtk.h>
#include <p4est_plex.h>
#include <p4est_bits.h>
#include <p4est_algorithms.h>
#else
#include <p8est.h>
#include <p8est_extended.h>
#include <p8est_geometry.h>
#include <p8est_ghost.h>
#include <p8est_lnodes.h>
#include <p8est_vtk.h>
#include <p8est_plex.h>
#include <p8est_bits.h>
#include <p8est_algorithms.h>
#endif

#define DMInitialize_pforest                  _append_pforest(DMInitialize)
#define DMCreate_pforest                      _append_pforest(DMCreate)
#define DMForestDestroy_pforest               _append_pforest(DMForestDestroy)
#define DMForestTemplate_pforest              _append_pforest(DMForestTemplate)
#define DMSetUp_pforest                       _append_pforest(DMSetUp)
#define DMView_pforest                        _append_pforest(DMView)
#define DMView_VTK_pforest                    _append_pforest(DMView_VTK)
#define DM_Forest_pforest                     _append_pforest(DM_Forest)
#define DMFTopology_pforest                   _append_pforest(DMFTopology)
#define DMFTopologyDestroy_pforest            _append_pforest(DMFTopologyDestroy)
#define DMFTopologyCreate_pforest             _append_pforest(DMFTopologyCreate)
#define DMFTopologyCreateBrick_pforest        _append_pforest(DMFTopologyCreateBrick)
#define DMConvert_plex_pforest                _append_pforest(DMConvert_plex)
#define DMPlexCreateConnectivity_pforest      _append_pforest(DMPlexCreateConnectivity)
#define DMConvert_pforest_plex                _infix_pforest(DMConvert,_plex)
#define DMCreateCoordinateDM_pforest          _append_pforest(DMCreateCoordinateDM)
#define DMCreateGlobalVector_pforest          _append_pforest(DMCreateGlobalVector)
#define DMCreateLocalVector_pforest           _append_pforest(DMCreateLocalVector)
#define DMCreateMatrix_pforest                _append_pforest(DMCreateMatrix)
#define DMProjectFunctionLocal_pforest        _append_pforest(DMProjectFunctionLocal)
#define DMProjectFunctionLabelLocal_pforest   _append_pforest(DMProjectFunctionLabelLocal)
#define DMCreateDefaultSection_pforest        _append_pforest(DMCreateDefaultSection)
#define DMCreateDefaultConstraints_pforest    _append_pforest(DMCreateDefaultConstraints)
#define DMComputeL2Diff_pforest               _append_pforest(DMComputeL2Diff)

static PetscErrorCode DMConvert_pforest_plex(DM,DMType,DM*);

typedef struct {
  PetscInt             refct;
  p4est_connectivity_t *conn;
  p4est_geometry_t     *geom;
  PetscInt             *tree_face_to_uniq; /* p4est does not explicitly enumerate facets, but we must to keep track of labels */
} DMFTopology_pforest;

typedef struct {
  DMFTopology_pforest *topo;
  p4est_t             *forest;
  p4est_ghost_t       *ghost;
  p4est_lnodes_t      *lnodes;
  PetscBool            partition_for_coarsening;
  PetscBool            coarsen_hierarchy;
  PetscBool            labelsFinalized;
  PetscInt             cLocalStart;
  PetscInt             cLocalEnd;
  DM                   plex;
} DM_Forest_pforest;

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMFTopologyDestroy_pforest)
static PetscErrorCode DMFTopologyDestroy_pforest(DMFTopology_pforest **topo)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!(*topo)) {
    PetscFunctionReturn(0);
  }
  if (--((*topo)->refct) > 0) {
    *topo = NULL;
    PetscFunctionReturn(0);
  }
  PetscStackCallP4est(p4est_geometry_destroy,((*topo)->geom));
  PetscStackCallP4est(p4est_connectivity_destroy,((*topo)->conn));
  ierr = PetscFree((*topo)->tree_face_to_uniq);CHKERRQ(ierr);
  ierr = PetscFree(*topo);CHKERRQ(ierr);
  *topo = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode PforestConnectivityEnumerateFacets(p4est_connectivity_t*,PetscInt **);

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMFTopologyCreateBrick_pforest)
static PetscErrorCode DMFTopologyCreateBrick_pforest(DM dm,PetscInt N[], PetscInt P[], DMFTopology_pforest **topo, PetscBool useMorton)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!useMorton) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Lexicographic ordering not implemented yet");
  ierr = PetscNewLog(dm,topo);CHKERRQ(ierr);

  (*topo)->refct = 1;
#if !defined(P4_TO_P8)
  PetscStackCallP4estReturn((*topo)->conn,p4est_connectivity_new_brick,((int) N[0], (int) N[1], (P[0] == DM_BOUNDARY_NONE) ? 0 : 1, (P[1] == DM_BOUNDARY_NONE) ? 0 : 1));
#else
  PetscStackCallP4estReturn((*topo)->conn,p8est_connectivity_new_brick,((int) N[0], (int) N[1], (int) N[2], (P[0] == DM_BOUNDARY_NONE) ? 0 : 1, (P[1] == DM_BOUNDARY_NONE) ? 0 : 1, (P[2] == DM_BOUNDARY_NONE) ? 0 : 1));
#endif
  PetscStackCallP4estReturn((*topo)->geom,p4est_geometry_new_connectivity,((*topo)->conn));
  ierr = PforestConnectivityEnumerateFacets((*topo)->conn,&(*topo)->tree_face_to_uniq);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMFTopologyCreate_pforest)
static PetscErrorCode DMFTopologyCreate_pforest(DM dm, DMForestTopology topologyName, DMFTopology_pforest **topo)
{
  DM_Forest  *forest = (DM_Forest *) dm->data;
  const char *name   = (const char *) topologyName;
  const char *prefix;
  PetscBool  isBrick, isShell, isSphere;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidCharPointer(name,2);
  PetscValidPointer(topo,3);
  ierr = PetscStrcmp(name,"brick",&isBrick);CHKERRQ(ierr);
  ierr = PetscStrcmp(name,"shell",&isShell);CHKERRQ(ierr);
  ierr = PetscStrcmp(name,"sphere",&isSphere);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)dm,&prefix);CHKERRQ(ierr);
  if (isBrick) {
    PetscBool  flgN, flgP, flgM, useMorton = PETSC_TRUE;
    PetscInt   N[P4EST_DIM] = {2}, P[P4EST_DIM] = {0}, nretN = P4EST_DIM, nretP = P4EST_DIM, i;

    if (forest->setFromOptions) {
      ierr = PetscOptionsGetIntArray(((PetscObject)dm)->options,prefix,"-dm_p4est_brick_size",N,&nretN,&flgN);CHKERRQ(ierr);
      ierr = PetscOptionsGetIntArray(((PetscObject)dm)->options,prefix,"-dm_p4est_brick_periodicity",P,&nretP,&flgP);CHKERRQ(ierr);
      ierr = PetscOptionsGetBool(((PetscObject)dm)->options,prefix,"-dm_p4est_brick_use_morton_curve",&useMorton,&flgM);CHKERRQ(ierr);
      if (flgN && nretN != P4EST_DIM) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_SIZ,"Need to give %d sizes in -dm_p4est_brick_size, gave %d",P4EST_DIM,nretN);
      if (flgP && nretP != P4EST_DIM) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_SIZ,"Need to give %d periodicities in -dm_p4est_brick_size, gave %d",P4EST_DIM,nretP);
    }
    for (i = 0; i < P4EST_DIM; i++) {
      P[i] = (P[i] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE);
    }
    ierr = DMFTopologyCreateBrick_pforest(dm,N,P,topo,useMorton);CHKERRQ(ierr);
  }
  else {
    ierr = PetscNewLog(dm,topo);CHKERRQ(ierr);

    (*topo)->refct = 1;
    PetscStackCallP4estReturn((*topo)->conn,p4est_connectivity_new_byname,(name));
#if !defined(P4_TO_P8)
    PetscStackCallP4estReturn((*topo)->geom,p4est_geometry_new_connectivity,((*topo)->conn));
#else
    if (isShell) {
      PetscReal R2 = 1., R1 = .55;

      if (forest->setFromOptions) {
        ierr = PetscOptionsGetReal(((PetscObject)dm)->options,prefix,"-dm_p4est_shell_outer_radius",&R2,NULL);CHKERRQ(ierr);
        ierr = PetscOptionsGetReal(((PetscObject)dm)->options,prefix,"-dm_p4est_shell_inner_radius",&R1,NULL);CHKERRQ(ierr);
      }
      PetscStackCallP4estReturn((*topo)->geom,p8est_geometry_new_shell,((*topo)->conn,R2,R1));
    }
    else if (isSphere) {
      PetscReal R2 = 1., R1 = 0.191728, R0 = 0.039856;

      if (forest->setFromOptions) {
        ierr = PetscOptionsGetReal(((PetscObject)dm)->options,prefix,"-dm_p4est_sphere_outer_radius",&R2,NULL);CHKERRQ(ierr);
        ierr = PetscOptionsGetReal(((PetscObject)dm)->options,prefix,"-dm_p4est_sphere_inner_radius",&R1,NULL);CHKERRQ(ierr);
        ierr = PetscOptionsGetReal(((PetscObject)dm)->options,prefix,"-dm_p4est_sphere_core_radius",&R0,NULL);CHKERRQ(ierr);
      }
      PetscStackCallP4estReturn((*topo)->geom,p8est_geometry_new_sphere,((*topo)->conn,R2,R1,R0));
    }
    else {
      PetscStackCallP4estReturn((*topo)->geom,p4est_geometry_new_connectivity,((*topo)->conn));
    }
#endif
    ierr = PforestConnectivityEnumerateFacets((*topo)->conn,&(*topo)->tree_face_to_uniq);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMForestDestroy_pforest)
static PetscErrorCode DMForestDestroy_pforest(DM dm)
{
  DM_Forest         *forest  = (DM_Forest *) dm->data;
  DM_Forest_pforest *pforest = (DM_Forest_pforest *) forest->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (pforest->lnodes) PetscStackCallP4est(p4est_lnodes_destroy,(pforest->lnodes));
  pforest->lnodes = NULL;
  if (pforest->ghost)  PetscStackCallP4est(p4est_ghost_destroy,(pforest->ghost));
  pforest->ghost = NULL;
  if (pforest->forest) PetscStackCallP4est(p4est_destroy,(pforest->forest));
  pforest->forest = NULL;
  ierr = DMFTopologyDestroy_pforest(&pforest->topo);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)dm,_pforest_string(DMConvert_plex_pforest) "_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)dm,_pforest_string(DMConvert_pforest_plex) "_C",NULL);CHKERRQ(ierr);
  ierr = DMDestroy(&pforest->plex);CHKERRQ(ierr);
  ierr = PetscFree(forest->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMForestTemplate_pforest)
static PetscErrorCode DMForestTemplate_pforest(DM dm, DM tdm)
{
  DM_Forest_pforest *pforest  = (DM_Forest_pforest *) ((DM_Forest *) dm->data)->data;
  DM_Forest_pforest *tpforest = (DM_Forest_pforest *) ((DM_Forest *) tdm->data)->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (pforest->topo) pforest->topo->refct++;
  ierr = DMFTopologyDestroy_pforest(&(tpforest->topo));CHKERRQ(ierr);
  tpforest->topo = pforest->topo;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCreateConnectivity_pforest(DM,p4est_connectivity_t**,PetscInt**);

static int pforest_coarsen_uniform (p4est_t * p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quadrants[])
{
  PetscInt minLevel = *((PetscInt *) p4est->user_pointer);

  if ((PetscInt) quadrants[0]->level > minLevel) {
    return 1;
  }
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMSetUp_pforest)
static PetscErrorCode DMSetUp_pforest(DM dm)
{
  DM_Forest         *forest  = (DM_Forest *) dm->data;
  DM_Forest_pforest *pforest = (DM_Forest_pforest *) forest->data;
  DM                base, coarse, fine, adaptFrom;
  DMForestTopology  topoName;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* sanity check */
  ierr = DMForestGetCoarseForest(dm,&coarse);CHKERRQ(ierr);
  ierr = DMForestGetFineForest(dm,&fine);CHKERRQ(ierr);
  ierr = DMForestGetBaseDM(dm,&base);CHKERRQ(ierr);
  ierr = DMForestGetTopology(dm,&topoName);CHKERRQ(ierr);
  if (coarse && fine)                   SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot adapt from both a coarse and a fine forest");
  adaptFrom = coarse ? coarse : fine;
  if (!adaptFrom && !base && !topoName) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"A forest needs a topology, a base DM, or a DM to adapt from");

  /* === Step 1: DMFTopology === */
  if (adaptFrom) { /* reference already created topology */
    PetscBool         ispforest;
    DM_Forest         *aforest  = (DM_Forest *) adaptFrom->data;
    DM_Forest_pforest *apforest = (DM_Forest_pforest *) aforest->data;

    ierr = PetscObjectTypeCompare((PetscObject)adaptFrom,DMPFOREST,&ispforest);CHKERRQ(ierr);
    if (!ispforest) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_NOTSAMETYPE,"Trying to adapt from %s, which is not %s\n",((PetscObject)adaptFrom)->type_name,DMPFOREST);
    if (!apforest->topo) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"The pre-adaptation forest must have a topology");
    ierr = DMSetUp(adaptFrom);CHKERRQ(ierr);
    ierr = DMForestTemplate(adaptFrom,dm);CHKERRQ(ierr);
    ierr = DMForestGetBaseDM(dm,&base);CHKERRQ(ierr);
    ierr = DMForestGetTopology(dm,&topoName);CHKERRQ(ierr);
  }
  else if (base) { /* construct a connectivity from base */
    PetscBool isPlex, isDA;

    ierr = PetscObjectGetName((PetscObject)base,&topoName);CHKERRQ(ierr);
    ierr = DMForestSetTopology(dm,topoName);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)base,DMPLEX,&isPlex);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)base,DMDA,&isDA);CHKERRQ(ierr);
    if (isPlex) {
      MPI_Comm             comm = PetscObjectComm((PetscObject)dm);
      PetscInt             depth;
      PetscMPIInt          size;
      p4est_connectivity_t *conn = NULL;
      DMFTopology_pforest  *topo;
      PetscInt             *tree_face_to_uniq;
      PetscErrorCode       ierr;

      ierr = DMPlexGetDepth(base,&depth);CHKERRQ(ierr);
      if (depth == 1) {
        DM connDM;

        ierr = DMPlexInterpolate(base,&connDM);CHKERRQ(ierr);
        base = connDM;
        ierr = DMForestSetBaseDM(dm,base);CHKERRQ(ierr);
        ierr = DMDestroy(&connDM);CHKERRQ(ierr);
      }
      else if (depth != P4EST_DIM) {
        SETERRQ2(comm,PETSC_ERR_ARG_WRONG,"Base plex is neither interpolated nor uninterpolated? depth %d, expected 2 or %d\n",depth,P4EST_DIM + 1);
      }
      ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
      if (size > 1) {
        DM dmRedundant;

        ierr = DMPlexGetRedundantDM(base,&dmRedundant);CHKERRQ(ierr);
        if (!dmRedundant) SETERRQ(comm,PETSC_ERR_PLIB,"Could not create redundant DM\n");
        base = dmRedundant;
        ierr = DMForestSetBaseDM(dm,base);CHKERRQ(ierr);
        ierr = DMDestroy(&dmRedundant);CHKERRQ(ierr);
      }
      ierr = DMPlexCreateConnectivity_pforest(base,&conn,&tree_face_to_uniq);CHKERRQ(ierr);
      ierr = PetscNewLog(dm,&topo);CHKERRQ(ierr);
      topo->refct             = 1;
      topo->conn              = conn;
      PetscStackCallP4estReturn(topo->geom,p4est_geometry_new_connectivity,(conn));
      topo->tree_face_to_uniq = tree_face_to_uniq;
      pforest->topo           = topo;
    }
    else if (isDA) {
      SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Not implemented yet");
#if 0
      PetscInt N[3], P[3];

      /* get the sizes, periodicities */
      /* ... */
                                                                  /* don't use Morton order */
      ierr = DMFTopologyCreateBrick_pforest(dm,N,P,&pforest->topo,PETSC_FALSE);CHKERRQ(ierr);
#endif
    }
    {
      PetscInt numLabels, l;

      ierr = DMGetNumLabels(base,&numLabels);CHKERRQ(ierr);
      for (l = 0; l < numLabels; l++) {
        PetscBool  isDepth;
        DMLabel    label, labelNew;
        PetscInt   defVal;
        const char *name;

        ierr = DMGetLabelName(base, l, &name);CHKERRQ(ierr);
        ierr = DMGetLabelByNum(base, l, &label);CHKERRQ(ierr);
        ierr = PetscStrcmp(name,"depth",&isDepth);CHKERRQ(ierr);
        if (isDepth) continue;
        ierr = DMCreateLabel(dm,name);CHKERRQ(ierr);
        ierr = DMGetLabel(dm,name,&labelNew);CHKERRQ(ierr);
        ierr = DMLabelGetDefaultValue(label,&defVal);CHKERRQ(ierr);
        ierr = DMLabelSetDefaultValue(labelNew,defVal);CHKERRQ(ierr);
      }
    }
  }
  else { /* construct from topology name */
    DMFTopology_pforest *topo;

    ierr = DMFTopologyCreate_pforest(dm,topoName,&topo);CHKERRQ(ierr);
    pforest->topo = topo;
    /* TODO: construct base? */
  }

  /* === Step 2: get the leaves of the forest === */
  if (adaptFrom) { /* start with the old forest */
    DM_Forest         *aforest  = (DM_Forest *) adaptFrom->data;
    DM_Forest_pforest *apforest = (DM_Forest_pforest *) aforest->data;

    PetscStackCallP4estReturn(pforest->forest,p4est_copy,(apforest->forest, 0)); /* 0 indicates no data copying */
    /* apply the refinement/coarsening by flags, plus minimum/maximum refinement */
    if (fine) { /* coarsen */
      /* TODO: Non uniform coarsening case?  Recursive?  Flags. */
      PetscInt minLevel;

      ierr = DMForestGetMinimumRefinement(dm,&minLevel);CHKERRQ(ierr);
      pforest->forest->user_pointer = (void *) &minLevel;
      PetscStackCallP4est(p4est_coarsen,(pforest->forest,0,pforest_coarsen_uniform,NULL));
      pforest->forest->user_pointer = (void *) dm;
    }
    else { /* refine */
      SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Not implemented yet");
    }
    PetscStackCallP4est(p4est_reset_data,(pforest->forest,0,NULL,(void *)dm)); /* this dm is the user context for the new forest */
    {
      PetscInt numLabels, l;

      ierr = DMGetNumLabels(adaptFrom,&numLabels);CHKERRQ(ierr);
      for (l = 0; l < numLabels; l++) {
        PetscBool  isDepth;
        DMLabel    label, labelNew;
        PetscInt   defVal;
        const char *name;

        ierr = DMGetLabelName(adaptFrom, l, &name);CHKERRQ(ierr);
        ierr = DMGetLabelByNum(adaptFrom, l, &label);CHKERRQ(ierr);
        ierr = PetscStrcmp(name,"depth",&isDepth);CHKERRQ(ierr);
        if (isDepth) continue;
        ierr = DMCreateLabel(dm,name);CHKERRQ(ierr);
        ierr = DMGetLabel(dm,name,&labelNew);CHKERRQ(ierr);
        ierr = DMLabelGetDefaultValue(label,&defVal);CHKERRQ(ierr);
        ierr = DMLabelSetDefaultValue(labelNew,defVal);CHKERRQ(ierr);
      }
    }
  }
  else { /* initial */
    PetscInt initLevel, minLevel;

    ierr = DMForestGetInitialRefinement(dm,&initLevel);CHKERRQ(ierr);
    ierr = DMForestGetMinimumRefinement(dm,&minLevel);CHKERRQ(ierr);
    PetscStackCallP4estReturn(pforest->forest,p4est_new_ext,(PetscObjectComm((PetscObject)dm),pforest->topo->conn,
                                                             0,           /* minimum number of quadrants per processor */
                                                             initLevel,   /* level of refinement */
                                                             1,           /* uniform refinement */
                                                             0,           /* we don't allocate any per quadrant data */
                                                             NULL,        /* there is no special quadrant initialization */
                                                             (void *)dm)); /* this dm is the user context */
    if (initLevel > minLevel) {
      pforest->coarsen_hierarchy = PETSC_TRUE;
    }
  }
  if (pforest->coarsen_hierarchy) {
    PetscInt initLevel, minLevel;

    ierr = DMForestGetInitialRefinement(dm,&initLevel);CHKERRQ(ierr);
    ierr = DMForestGetMinimumRefinement(dm,&minLevel);CHKERRQ(ierr);
    if (initLevel > minLevel) {
      DM_Forest_pforest *coarse_pforest;
      DMType type;
      DM     coarseDM;

      ierr = DMCreate(PetscObjectComm((PetscObject)dm),&coarseDM);CHKERRQ(ierr);
      ierr = DMGetType(dm,&type);CHKERRQ(ierr);
      ierr = DMSetType(coarseDM,type);CHKERRQ(ierr);
      ierr = DMForestSetFineForest(coarseDM,dm);CHKERRQ(ierr);
      ierr = DMSetCoarseDM(dm,coarseDM);CHKERRQ(ierr);
      if (forest->setFromOptions) {
        ierr = DMSetFromOptions(coarseDM);CHKERRQ(ierr);
      }
      ierr = DMForestSetInitialRefinement(coarseDM,initLevel - 1);CHKERRQ(ierr);
      coarse_pforest = (DM_Forest_pforest *) ((DM_Forest *) coarseDM->data)->data;
      coarse_pforest->coarsen_hierarchy = PETSC_TRUE;
      ierr = DMDestroy(&coarseDM);CHKERRQ(ierr);
    }
  }
  if (pforest->partition_for_coarsening || forest->cellWeights || forest->weightCapacity != 1. || forest->weightsFactor != 1.) {
    if (!forest->cellWeights && forest->weightCapacity == 1. && forest->weightsFactor == 1.) {
      PetscStackCallP4est(p4est_partition,(pforest->forest,(int)pforest->partition_for_coarsening,NULL));
    }
    else {
      /* TODO: handle non-uniform partition cases */
      SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Not implemented yet");
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMView_VTK_pforest)
static PetscErrorCode DMView_VTK_pforest(PetscObject odm, PetscViewer viewer)
{
  DM                dm       = (DM) odm;
  DM_Forest         *forest  = (DM_Forest *) dm->data;
  DM_Forest_pforest *pforest = (DM_Forest_pforest *) forest->data;
  PetscBool         isvtk;
  PetscReal         vtkScale = 1. - PETSC_MACHINE_EPSILON;
  PetscViewer_VTK   *vtk = (PetscViewer_VTK*)viewer->data;
  const char        *name;
  char              *filenameStrip = NULL;
  PetscBool         hasExt;
  size_t            len;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK, &isvtk);CHKERRQ(ierr);
  if (!isvtk) SETERRQ1(PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_INCOMP, "Cannot use viewer type %s", ((PetscObject)viewer)->type_name);
  switch (viewer->format) {
  case PETSC_VIEWER_VTK_VTU:
    if (!pforest->forest) SETERRQ (PetscObjectComm(odm),PETSC_ERR_ARG_WRONG,"DM has not been setup with a valid forest");
    name = vtk->filename;
    ierr = PetscStrlen(name,&len);CHKERRQ(ierr);
    ierr = PetscStrcasecmp(name+len-4,".vtu",&hasExt);CHKERRQ(ierr);
    if (hasExt) {
      ierr = PetscStrallocpy(name,&filenameStrip);CHKERRQ(ierr);
      filenameStrip[len-4]='\0';
      name = filenameStrip;
    }
    PetscStackCallP4est(p4est_vtk_write_all,(pforest->forest,pforest->topo->geom,(double)vtkScale,
                                             1, /* write tree */
                                             1, /* write level */
                                             1, /* write rank */
                                             0, /* do not wrap rank */
                                             0, /* no scalar fields */
                                             0, /* no vector fields */
                                             name));
    ierr = PetscFree(filenameStrip);CHKERRQ(ierr);
    break;
  default: SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "No support for format '%s'", PetscViewerFormats[viewer->format]);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMView_pforest)
static PetscErrorCode DMView_pforest(DM dm, PetscViewer viewer)
{
  PetscBool      isvtk;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK,   &isvtk);CHKERRQ(ierr);
  if (isvtk) {
    ierr = DMView_VTK_pforest((PetscObject) dm,viewer);CHKERRQ(ierr);
  }
  else {
    SETERRQ(PetscObjectComm((PetscObject) dm),PETSC_ERR_SUP,"Non-vtk viewers not implemented yet");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PforestConnectivityEnumerateFacets"
static PetscErrorCode PforestConnectivityEnumerateFacets(p4est_connectivity_t *conn, PetscInt **tree_face_to_uniq)
{
  PetscInt       *ttf, f, t, g, count;
  PetscInt       numFacets;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  numFacets = conn->num_trees * P4EST_FACES;
  ierr = PetscMalloc1(numFacets,&ttf);CHKERRQ(ierr);
  for (f = 0; f < numFacets; f++) ttf[f] = -1;
  for (g = 0, count = 0, t = 0; t < conn->num_trees; t++) {
    for (f = 0; f < P4EST_FACES; f++, g++) {
      if (ttf[g] == -1) {
        PetscInt ng;

        ttf[g] = count++;
        ng = conn->tree_to_tree[g] * P4EST_FACES + (conn->tree_to_face[g] % P4EST_FACES);
        ttf[ng] = ttf[g];
      }
    }
  }
  *tree_face_to_uniq = ttf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMPlexCreateConnectivity_pforest)
static PetscErrorCode DMPlexCreateConnectivity_pforest(DM dm, p4est_connectivity_t **connOut, PetscInt **tree_face_to_uniq)
{
  p4est_topidx_t       numTrees, numVerts, numCorns, numCtt;
  PetscSection         ctt;
#if defined(P4_TO_P8)
  p4est_topidx_t       numEdges, numEtt;
  PetscSection         ett;
  PetscInt             eStart, eEnd, e, ettSize;
  PetscInt             vertOff = 1 + P4EST_FACES + P8EST_EDGES;
  PetscInt             edgeOff = 1 + P4EST_FACES;
#else
  PetscInt             vertOff = 1 + P4EST_FACES;
#endif
  p4est_connectivity_t *conn;
  PetscInt             cStart, cEnd, c, vStart, vEnd, v, fStart, fEnd, f;
  PetscInt             *star = NULL, *closure = NULL, closureSize, starSize, cttSize;
  PetscInt             *ttf;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  /* 1: count objects, allocate */
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = P4estTopidxCast(cEnd-cStart,&numTrees);CHKERRQ(ierr);
  numVerts = P4EST_CHILDREN * numTrees;
  ierr = DMPlexGetDepthStratum(dm,0,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = P4estTopidxCast(vEnd-vStart,&numCorns);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PETSC_COMM_SELF,&ctt);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(ctt,vStart,vEnd);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; v++) {
    PetscInt s;

    ierr = DMPlexGetTransitiveClosure(dm,v,PETSC_FALSE,&starSize,&star);CHKERRQ(ierr);
    for (s = 0; s < starSize; s++) {
      PetscInt p = star[2*s];

      if (p >= cStart && p < cEnd) {
        /* we want to count every time cell p references v, so we see how many times it comes up in the closure.  This
         * only protects against periodicity problems */
        ierr = DMPlexGetTransitiveClosure(dm,p,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
        for (c = 0; c < P4EST_CHILDREN; c++) {
          PetscInt cellVert = closure[2 * (c + vertOff)];

          if (cellVert < vStart || cellVert >= vEnd) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Non-standard closure: vertices\n");
          if (cellVert == v) {
            ierr = PetscSectionAddDof(ctt,v,1);CHKERRQ(ierr);
          }
        }
        ierr = DMPlexRestoreTransitiveClosure(dm,p,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm,v,PETSC_FALSE,&starSize,&star);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(ctt);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(ctt,&cttSize);CHKERRQ(ierr);
  ierr = P4estTopidxCast(cttSize,&numCtt);CHKERRQ(ierr);
#if defined(P4_TO_P8)
  ierr = DMPlexGetDepthStratum(dm,1,&eStart,&eEnd);CHKERRQ(ierr);
  ierr = P4estTopidxCast(eEnd-eStart,&numEdges);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PETSC_COMM_SELF,&ett);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(ett,eStart,eEnd);CHKERRQ(ierr);
  for (e = eStart; e < eEnd; e++) {
    PetscInt s;

    ierr = DMPlexGetTransitiveClosure(dm,e,PETSC_FALSE,&starSize,&star);CHKERRQ(ierr);
    for (s = 0; s < starSize; s++) {
      PetscInt p = star[2*c];

      if (p >= cStart && p < cEnd) {
        /* we want to count every time cell p references e, so we see how many times it comes up in the closure.  This
         * only protects against periodicity problems */
        ierr = DMPlexGetTransitiveClosure(dm,p,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
        for (c = 0; c < P8EST_EDGES; c++) {
          PetscInt cellEdge = closure[2 * (c + edgeOff)];

          if (cellEdge < eStart || cellEdge >= eEnd) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Non-standard closure: edges\n");
          if (cellEdge == e) {
            ierr = PetscSectionAddDof(ett,e,1);CHKERRQ(ierr);
          }
        }
        ierr = DMPlexRestoreTransitiveClosure(dm,p,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm,e,PETSC_FALSE,&starSize,&star);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(ett);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(ett,&ettSize);CHKERRQ(ierr);
  ierr = P4estTopidxCast(ettSize,&numEtt);CHKERRQ(ierr);

  /* This routine allocates space for the arrays, which we fill below */
  PetscStackCallP4estReturn(conn,p8est_connectivity_new,(numVerts,numTrees,numEdges,numEtt,numCorns,numCtt));
#else
  PetscStackCallP4estReturn(conn,p4est_connectivity_new,(numVerts,numTrees,numCorns,numCtt));
#endif

  /* 2: visit every face, determine neighboring cells(trees) */
  ierr = DMPlexGetHeightStratum(dm,1,&fStart,&fEnd);CHKERRQ(ierr);
  ierr = PetscMalloc1((cEnd-cStart) * P4EST_FACES,&ttf);CHKERRQ(ierr);
  for (f = fStart; f < fEnd; f++) {
    PetscInt numSupp, s;
    PetscInt myFace[2] = {-1, -1};
    PetscInt myOrnt[2] = {PETSC_MIN_INT, PETSC_MIN_INT};
    const PetscInt *supp;

    ierr = DMPlexGetSupportSize(dm, f, &numSupp);CHKERRQ(ierr);
    if (numSupp != 1 && numSupp != 2) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"point %d has facet with %d sides: must be 1 or 2 (boundary or conformal)\n",f,numSupp);
    ierr = DMPlexGetSupport(dm, f, &supp);CHKERRQ(ierr);

    for (s = 0; s < numSupp; s++) {
      PetscInt p = supp[s], i;
      PetscInt numCone;
      const PetscInt *cone;
      const PetscInt *ornt;
      PetscInt orient = PETSC_MIN_INT;

      ierr = DMPlexGetConeSize(dm, p, &numCone);CHKERRQ(ierr);
      if (numCone != P4EST_FACES) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"cell %d has %d facets, expect %d\n",p,numCone,P4EST_FACES);
      ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, p, &ornt);CHKERRQ(ierr);
      for (i = 0; i < P4EST_FACES; i++) {
        if (cone[i] == f) {
          orient = ornt[i];
          break;
        }
      }
      if (i >= P4EST_FACES) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"cell %d faced %d mismatch\n",p,f);
      ttf[P4EST_FACES * (p - cStart) + PetscFaceToP4estFace[i]] = f - fStart;
      if (numSupp == 1) {
        /* boundary faces indicated by self reference */
        conn->tree_to_tree[P4EST_FACES * (p - cStart) + PetscFaceToP4estFace[i]] = p - cStart;
        conn->tree_to_face[P4EST_FACES * (p - cStart) + PetscFaceToP4estFace[i]] = (int8_t) PetscFaceToP4estFace[i];
      }
      else {
        conn->tree_to_tree[P4EST_FACES * (p - cStart) + PetscFaceToP4estFace[i]] = supp[1 - s] - cStart;
        myFace[s] = PetscFaceToP4estFace[i];
        /* get the orientation of cell p in p4est-type closure to facet f, by composing the p4est-closure to
         * petsc-closure permutation and the petsc-closure to facet orientation */
        myOrnt[s] = DihedralCompose((P4EST_CHILDREN/2),P4estFaceToPetscOrnt[myFace[s]],orient);
      }
    }
    if (numSupp == 2) {
      for (s = 0; s < numSupp; s++) {
        PetscInt p = supp[s];
        PetscInt orntAtoB;
        PetscInt p4estOrient;

        /* composing the forward permutation with the other cell's inverse permutation gives the self-to-neighbor
         * permutation of this cell-facet's cone */
        orntAtoB = DihedralCompose((P4EST_CHILDREN/2),myOrnt[s],DihedralInvert((P4EST_CHILDREN/2),myOrnt[1-s]));

        /* convert cone-description permutation (i.e., edges around facet) to cap-description permutation (i.e.,
         * vertices around facet) */
#if !defined(P4_TO_P8)
        p4estOrient = orntAtoB < 0 ? -(orntAtoB + 1) : orntAtoB;
#else
        {
          PetscInt firstVert = orntAtoB < 0 ? ((-orntAtoB) % (P4EST_CHILDREN/2)): orntAtoB;
          PetscInt p4estFirstVert = firstVert < 2 ? firstVert : (firstVert ^ 1);

                                                                                           /* swap bits */
          p4estOrient = ((myFace[s] <= myFace[1 - s]) || (orntAtoB < 0)) ? p4estFirstVert : ((p4estFirstVert >> 1) | ((p4estFirstVert & 1) << 1));
        }
#endif
        /* encode neighbor face and orientation in tree_to_face per p4est_connectivity standard (see
         * p4est_connectivity.h, p8est_connectivity.h) */
        conn->tree_to_face[P4EST_FACES * (p - cStart) + myFace[s]] = (int8_t) myFace[1 - s] + p4estOrient * P4EST_FACES;
      }
    }
  }

#if defined(P4_TO_P8)
  /* 3: visit every edge */
  conn->ett_offset[0] = 0;
  for (e = eStart; e < eEnd; e++) {
    PetscInt off, s;

    ierr = PetscSectionGetOffset(ett,e,&off);CHKERRQ(ierr);
    conn->ett_offset[e - eStart] = (p4est_topidx_t) off;
    ierr = DMPlexGetTransitiveClosure(dm,e,PETSC_FALSE,&starSize,&star);CHKERRQ(ierr);
    for (s = 0; s < starSize; s++) {
      PetscInt p = star[2 * s];

      if (p >= cStart && p < cEnd) {
        ierr = DMPlexGetTransitiveClosure(dm,p,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
        if (closureSize != P4EST_INSUL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Non-standard closure\n");
        for (c = 0; c < P8EST_EDGES; c++) {
          PetscInt cellEdge = closure[2 * (c + edgeOff)];
          PetscInt cellOrnt = closure[2 * (c + edgeOff) + 1];

          if (cellEdge == e) {
            PetscInt p4estEdge = PetscEdgeToP4estEdge[c];
            PetscInt totalOrient;

            /* compose p4est-closure to petsc-closure permutation and petsc-closure to edge orientation */
            totalOrient = DihedralCompose(2,P4estEdgeToPetscOrnt[p4estEdge],cellOrnt);
            /* p4est orientations are positive: -2 => 1, -1 => 0 */
            totalOrient = (totalOrient < 0) ? -(totalOrient + 1) : totalOrient;
            conn->edge_to_tree[off] = (p4est_locidx_t) (p - cStart);
            /* encode cell-edge and orientation in edge_to_edge per p8est_connectivity standart (see
             * p8est_connectivity.h) */
            conn->edge_to_edge[off++] = (int8_t) p4estEdge + P8EST_EDGES * totalOrient;
            conn->tree_to_edge[P8EST_EDGES * (p - cStart) + p4estEdge] = e - eStart;
          }
        }
        ierr = DMPlexRestoreTransitiveClosure(dm,p,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm,e,PETSC_FALSE,&starSize,&star);CHKERRQ(ierr);
  }
  ierr = PetscSectionDestroy(&ett);CHKERRQ(ierr);
#endif

  /* 4: visit every vertex */
  conn->ctt_offset[0] = 0;
  for (v = vStart; v < vEnd; v++) {
    PetscInt off, s;

    ierr = PetscSectionGetOffset(ctt,v,&off);CHKERRQ(ierr);
    conn->ctt_offset[v - vStart] = (p4est_topidx_t) off;
    ierr = DMPlexGetTransitiveClosure(dm,v,PETSC_FALSE,&starSize,&star);CHKERRQ(ierr);
    for (s = 0; s < starSize; s++) {
      PetscInt p = star[2 * s];

      if (p >= cStart && p < cEnd) {
        ierr = DMPlexGetTransitiveClosure(dm,p,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
        if (closureSize != P4EST_INSUL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Non-standard closure\n");
        for (c = 0; c < P4EST_CHILDREN; c++) {
          PetscInt cellVert = closure[2 * (c + vertOff)];

          if (cellVert == v) {
            PetscInt p4estVert = PetscVertToP4estVert[c];

            conn->corner_to_tree[off] = (p4est_locidx_t) (p - cStart);
            conn->corner_to_corner[off++] = (int8_t) p4estVert;
            conn->tree_to_corner[P4EST_CHILDREN * (p - cStart) + p4estVert] = v - vStart;
          }
        }
        ierr = DMPlexRestoreTransitiveClosure(dm,p,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(dm,v,PETSC_FALSE,&starSize,&star);CHKERRQ(ierr);
  }
  ierr = PetscSectionDestroy(&ctt);CHKERRQ(ierr);

  /* 5: Compute the coordinates */
  {
    PetscInt     coordDim;
    Vec          coordVec;
    PetscSection coordSec;

    ierr = DMGetCoordinateDim(dm,&coordDim);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm,&coordVec);CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(dm,&coordSec);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; c++) {
      PetscInt    dof;
      PetscScalar *cellCoords = NULL;

      ierr = DMPlexVecGetClosure(dm, coordSec, coordVec, c, &dof, &cellCoords);CHKERRQ(ierr);
      if (dof != P4EST_CHILDREN * coordDim) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Need coordinates at the corners\n");
      for (v = 0; v < P4EST_CHILDREN; v++) {
        PetscInt i, lim = PetscMin(3, coordDim);
        PetscInt p4estVert = PetscVertToP4estVert[v];

        conn->tree_to_vertex[P4EST_CHILDREN * (c - cStart) + v] = P4EST_CHILDREN * (c - cStart) + v;
        /* p4est vertices are always embedded in R^3 */
        for (i = 0; i < 3; i++) {
          conn->vertices[3 * (P4EST_CHILDREN * (c - cStart) + p4estVert) + i] = 0.;
        }
        for (i = 0; i < lim; i++) {
          conn->vertices[3 * (P4EST_CHILDREN * (c - cStart) + p4estVert) + i] = PetscRealPart(cellCoords[v * coordDim + i]);
        }
      }
      ierr = DMPlexVecRestoreClosure(dm, coordSec, coordVec, c, &dof, &cellCoords);CHKERRQ(ierr);
    }
  }

#if P4EST_ENABLE_DEBUG
  if (!p4est_connectivity_is_valid(conn)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Plex to p4est conversion failed\n");
#endif

  *connOut = conn;

  *tree_face_to_uniq = ttf;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMConvert_plex_pforest)
static PetscErrorCode DMConvert_plex_pforest(DM dm, DMType newtype, DM *pforest)
{
  MPI_Comm       comm;
  PetscBool      isPlex;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  comm = PetscObjectComm((PetscObject)dm);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMPLEX,&isPlex);CHKERRQ(ierr);
  if (!isPlex) SETERRQ2(comm,PETSC_ERR_ARG_WRONG,"Expected DM type %s, got %s\n",DMPLEX,((PetscObject)dm)->type_name);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if (dim != P4EST_DIM) SETERRQ2(comm,PETSC_ERR_ARG_WRONG,"Expected DM dimension %d, got %d\n",P4EST_DIM,dim);
  ierr = DMCreate(comm,pforest);CHKERRQ(ierr);
  ierr = DMSetType(*pforest,DMPFOREST);CHKERRQ(ierr);
  ierr = DMForestSetBaseDM(*pforest,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "locidx_to_PetscInt"
static PetscErrorCode locidx_to_PetscInt (sc_array_t * array)
{
  sc_array_t         *newarray;
  size_t              zz, count = array->elem_count;

  PetscFunctionBegin;
  if (array->elem_size != sizeof (p4est_locidx_t)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong locidx size");

  if (sizeof (p4est_locidx_t) == sizeof (PetscInt)) {
    PetscFunctionReturn(0);
  }

  newarray = sc_array_new_size (sizeof (PetscInt), array->elem_count);
  for (zz = 0; zz < count; zz++) {
    p4est_locidx_t      il = *((p4est_locidx_t *) sc_array_index (array, zz));
    PetscInt           *ip = (PetscInt *) sc_array_index (newarray, zz);

    *ip = (PetscInt) il;
  }

  sc_array_reset (array);
  sc_array_init_size (array, sizeof (PetscInt), count);
  sc_array_copy (array, newarray);
  sc_array_destroy (newarray);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "coords_double_to_PetscScalar"
static PetscErrorCode coords_double_to_PetscScalar (sc_array_t * array, PetscInt dim)
{
  sc_array_t         *newarray;
  size_t              zz, count = array->elem_count;

  PetscFunctionBegin;
  if (array->elem_size != 3 * sizeof (double)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong coordinate size");

  if (sizeof (double) == sizeof (PetscScalar) && dim == 3) {
    PetscFunctionReturn(0);
  }

  newarray = sc_array_new_size (dim * sizeof (PetscScalar), array->elem_count);
  for (zz = 0; zz < count; zz++) {
    int                 i;
    double             *id = (double *) sc_array_index (array, zz);
    PetscScalar        *ip = (PetscScalar *) sc_array_index (newarray, zz);

    for (i = 0; i < dim; i++) {
      ip[i] = 0.;
    }
    for (i = 0; i < PetscMin(dim,3); i++) {
      ip[i] = (PetscScalar) id[i];
    }
  }

  sc_array_reset (array);
  sc_array_init_size (array, dim * sizeof (PetscScalar), count);
  sc_array_copy (array, newarray);
  sc_array_destroy (newarray);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "locidx_pair_to_PetscSFNode"
static PetscErrorCode locidx_pair_to_PetscSFNode (sc_array_t * array)
{
  sc_array_t         *newarray;
  size_t              zz, count = array->elem_count;

  PetscFunctionBegin;
  if (array->elem_size != 2 * sizeof (p4est_locidx_t)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong locidx size");

  newarray = sc_array_new_size (sizeof (PetscSFNode), array->elem_count);
  for (zz = 0; zz < count; zz++) {
    p4est_locidx_t     *il = (p4est_locidx_t *) sc_array_index (array, zz);
    PetscSFNode        *ip = (PetscSFNode *) sc_array_index (newarray, zz);

    ip->rank = (PetscInt) il[0];
    ip->index = (PetscInt) il[1];
  }

  sc_array_reset (array);
  sc_array_init_size (array, sizeof (PetscSFNode), count);
  sc_array_copy (array, newarray);
  sc_array_destroy (newarray);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMShareDiscretization"
static PetscErrorCode DMShareDiscretization(DM dmA, DM dmB)
{
  PetscDS        ds;
  void           *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetApplicationContext(dmA,&ctx);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dmB,ctx);CHKERRQ(ierr);
  ierr = DMGetDS(dmA,&ds);CHKERRQ(ierr);
  ierr = DMSetDS(dmB,ds);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)dmA->defaultSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&(dmB->defaultSection));CHKERRQ(ierr);
  dmB->defaultSection = dmA->defaultSection;
  ierr = PetscObjectReference((PetscObject)dmA->defaultGlobalSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&(dmB->defaultGlobalSection));CHKERRQ(ierr);
  dmB->defaultGlobalSection = dmA->defaultGlobalSection;
  ierr = PetscObjectReference((PetscObject)dmA->defaultSF);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&dmB->defaultSF);CHKERRQ(ierr);
  dmB->defaultSF = dmA->defaultSF;
  dmA->boundary->refct++;
  ierr = DMBoundaryDestroy(&(dmB->boundary));CHKERRQ(ierr);
  dmB->boundary = dmA->boundary;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPforestComputeOverlappingRanks"
/* range of processes whose B sections overlap this ranks A section */
static PetscErrorCode DMPforestComputeOverlappingRanks(PetscMPIInt size, PetscMPIInt rank, p4est_t *p4estA, p4est_t *p4estB, PetscInt *startB, PetscInt *endB)
{
  p4est_quadrant_t * myCoarseStart = &(p4estA->global_first_position[rank]);
  p4est_quadrant_t * myCoarseEnd   = &(p4estA->global_first_position[rank+1]);
  p4est_quadrant_t * globalFirstB  = p4estB->global_first_position;

  PetscFunctionBegin;
  *startB = -1;
  *endB = -1;
  if (p4estA->local_num_quadrants) {
    PetscInt lo, hi, guess;
    /* binary search to find interval containing myCoarseStart */
    lo    = 0;
    hi    = size;
    guess = rank;
    while (1) {
      int startCompMy, myCompEnd;

      PetscStackCallP4estReturn(startCompMy,p4est_quadrant_compare_piggy,(&globalFirstB[guess],myCoarseStart));
      PetscStackCallP4estReturn(myCompEnd,p4est_quadrant_compare_piggy,(myCoarseStart,&globalFirstB[guess+1]));
      if (startCompMy <= 0 && myCompEnd < 0) {
        *startB = guess;
        break;
      }
      else if (startCompMy > 0) { /* guess is to high */
        hi = guess;
      }
      else { /* guess is to low */
        lo = guess + 1;
      }
      guess = lo + (hi - lo) / 2;
    }
    /* reset bounds, but not guess */
    lo = 0;
    hi = size;
    while (1) {
      int startCompMy, myCompEnd;

      PetscStackCallP4estReturn(startCompMy,p4est_quadrant_compare_piggy,(&globalFirstB[guess],myCoarseEnd));
      PetscStackCallP4estReturn(myCompEnd,p4est_quadrant_compare_piggy,(myCoarseEnd,&globalFirstB[guess+1]));
      if (startCompMy < 0 && myCompEnd <= 0) { /* notice that the comparison operators are different from above */
        *endB = guess + 1;
        break;
      }
      else if (startCompMy >= 0) { /* guess is to high */
        hi = guess;
      }
      else { /* guess is to low */
        lo = guess + 1;
      }
      guess = lo + (hi - lo) / 2;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPforestGetCellCoveringSF"
/* Get an SF that broadcasts a coarse-cell covering of the local fine cells */
static PetscErrorCode DMPforestGetCellCoveringSF(MPI_Comm comm,p4est_t *p4estC, p4est_t *p4estF, PetscInt cStart, PetscInt cEnd, PetscInt cLocalStart, PetscSF *coveringSF)
{
  PetscInt       startF, endF, startC, endC, p, nLeaves;
  PetscSFNode    *leaves;
  PetscSF        sf;
  PetscInt       *recv, *send;
  PetscMPIInt    tag;
  MPI_Request    *recvReqs, *sendReqs;
  PetscSection   section;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPforestComputeOverlappingRanks(p4estC->mpisize,p4estC->mpirank,p4estF,p4estC,&startC,&endC);CHKERRQ(ierr);
  ierr = PetscMalloc2(2*(endC-startC),&recv,endC-startC,&recvReqs);CHKERRQ(ierr);
  ierr = PetscCommGetNewTag(comm,&tag);CHKERRQ(ierr);
  for (p = startC; p < endC; p++) {
    recvReqs[p-startC] = MPI_REQUEST_NULL; /* just in case we don't initiate a receive */
    if (p4estC->global_first_quadrant[p] == p4estC->global_first_quadrant[p+1]) { /* empty coarse partition */
      recv[2*(p-startC)] = 0;
      recv[2*(p-startC)+1] = 0;
      continue;
    }

    ierr = MPI_Irecv(&recv[2*(p-startC)],2,MPIU_INT,p,tag,comm,&recvReqs[p-startC]);CHKERRQ(ierr);
  }
  ierr = DMPforestComputeOverlappingRanks(p4estC->mpisize,p4estC->mpirank,p4estC,p4estF,&startF,&endF);CHKERRQ(ierr);
  ierr = PetscMalloc2(2*(endF-startF),&send,endF-startF,&sendReqs);CHKERRQ(ierr);
  ierr = PetscSFCreate(p4estC->mpicomm,&sf);CHKERRQ(ierr);
  /* count the quadrants rank will send to each of [startF,endF) */
  for (p = startF; p < endF; p++) {
    p4est_quadrant_t *myFineStart = &p4estF->global_first_position[p];
    p4est_quadrant_t *myFineEnd   = &p4estF->global_first_position[p+1];
    PetscInt tStart = (PetscInt) myFineStart->p.which_tree;
    PetscInt tEnd   = (PetscInt) myFineEnd->p.which_tree;
    PetscInt firstCell = -1, lastCell = -1;
    p4est_tree_t *treeStart = &(((p4est_tree_t *) p4estC->trees->array)[tStart]);
    p4est_tree_t *treeEnd   = (size_t) tEnd < p4estC->trees->elem_count ? &(((p4est_tree_t *) p4estC->trees->array)[tEnd]) : NULL;
    ssize_t overlapIndex;

    sendReqs[p-startF] = MPI_REQUEST_NULL; /* just in case we don't initiate a send */
    if (p4estF->global_first_quadrant[p] == p4estF->global_first_quadrant[p+1]) {
      continue;
    }

    /* locate myFineStart in (or before) a cell */
    if (treeStart->quadrants.elem_count) {
      PetscStackCallP4estReturn(overlapIndex,sc_array_bsearch,(&(treeStart->quadrants),myFineStart,p4est_quadrant_disjoint));
      if (overlapIndex < 0) {
        firstCell = 0;
      }
      else {
        firstCell = treeStart->quadrants_offset + overlapIndex;
      }
    }
    else {
      firstCell = 0;
    }
    if (treeEnd && treeEnd->quadrants.elem_count) {
      PetscStackCallP4estReturn(overlapIndex,sc_array_bsearch,(&(treeEnd->quadrants),myFineEnd,p4est_quadrant_disjoint));
      if (overlapIndex < 0) { /* all of this local section is overlapped */
        lastCell = p4estC->local_num_quadrants;
      }
      else {
        p4est_quadrant_t *container = &(((p4est_quadrant_t *) treeEnd->quadrants.array)[overlapIndex]);
        p4est_quadrant_t first_desc;
        int equal;

        PetscStackCallP4est(p4est_quadrant_first_descendant,(container,&first_desc,P4EST_QMAXLEVEL));
        PetscStackCallP4estReturn(equal,p4est_quadrant_is_equal,(myFineEnd,&first_desc));
        if (equal) {
          lastCell = treeEnd->quadrants_offset + overlapIndex;
        }
        else {
          lastCell = treeEnd->quadrants_offset + overlapIndex + 1;
        }
      }
    }
    else {
      lastCell = p4estC->local_num_quadrants;
    }
    send[2*(p-startF)] = firstCell + cLocalStart;
    send[2*(p-startF)+1] = lastCell - firstCell;
    ierr = MPI_Isend(&send[2*(p-startF)],2,MPIU_INT,p,tag,comm,&sendReqs[p-startF]);CHKERRQ(ierr);
  }
  ierr = MPI_Waitall((PetscMPIInt)(endC-startC),recvReqs,NULL);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PETSC_COMM_SELF,&section);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(section,startC,endC);CHKERRQ(ierr);
  for (p = startC; p < endC; p++) {
    PetscInt numCells = recv[2*(p-startC)+1];
    ierr = PetscSectionSetDof(section,p,numCells);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(section);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(section,&nLeaves);CHKERRQ(ierr);
  ierr = PetscMalloc1(nLeaves,&leaves);CHKERRQ(ierr);
  for (p = startC; p < endC; p++) {
    PetscInt firstCell = recv[2*(p-startC)];
    PetscInt numCells = recv[2*(p-startC)+1];
    PetscInt off, i;

    ierr = PetscSectionGetOffset(section,p,&off);CHKERRQ(ierr);
    for (i = 0; i < numCells; i++) {
      leaves[off+i].rank = p;
      leaves[off+i].index = firstCell + i;
    }
  }
  ierr = PetscSFCreate(comm,&sf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(sf,cEnd-cStart,nLeaves,NULL,PETSC_OWN_POINTER,leaves,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  ierr = MPI_Waitall((PetscMPIInt)(endF-startF),sendReqs,NULL);CHKERRQ(ierr);
  ierr = PetscFree2(send,sendReqs);CHKERRQ(ierr);
  ierr = PetscFree2(recv,recvReqs);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPforestGetPlex(DM,DM*);

#undef __FUNCT__
#define __FUNCT__ "DMPforestGetCellSFNodes"
/* closure points for locally-owned cells */
static PetscErrorCode DMPforestGetCellSFNodes(DM dm, PetscInt numClosureIndices, const PetscInt closureIndices[], PetscInt *numClosurePoints, PetscSFNode **closurePoints,PetscBool redirect)
{
  PetscInt          cStart, cEnd;
  PetscInt          count, c;
  PetscMPIInt       rank;
  PetscInt          closureSize = -1;
  PetscInt          *closure = NULL;
  PetscSF           pointSF;
  PetscInt          nleaves, nroots;
  const PetscInt    *ilocal;
  const PetscSFNode *iremote;
  DM                plex;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMPforestGetPlex(dm,&plex);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(plex,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMGetPointSF(dm,&pointSF);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(pointSF,&nroots,&nleaves,&ilocal,&iremote);CHKERRQ(ierr);
  *numClosurePoints = numClosureIndices * (cEnd - cStart);
  ierr = PetscMalloc1(*numClosurePoints,closurePoints);
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRQ(ierr);
  for (c = cStart, count = 0; c < cEnd; c++) {
    PetscInt i;
    ierr = DMPlexGetTransitiveClosure(plex,c,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);

    for (i = 0; i < numClosureIndices; i++, count++) {
      PetscInt j = closureIndices[i];
      PetscInt p = closure[2 * j];
      PetscInt loc = -1;

      ierr = PetscFindInt(p,nleaves,ilocal,&loc);CHKERRQ(ierr);
      if (redirect && loc >= 0) {
        (*closurePoints)[count].rank  = iremote[loc].rank;
        (*closurePoints)[count].index = iremote[loc].index;
      }
      else {
        (*closurePoints)[count].rank  = rank;
        (*closurePoints)[count].index = p;
      }
    }
    ierr = DMPlexRestoreTransitiveClosure(plex,c,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static void DMPforestMaxSFNode(void *a, void *b, PetscMPIInt *len, MPI_Datatype *type)
{
  PetscMPIInt i;

  for (i = 0; i < *len; i++) {
    PetscSFNode *A = (PetscSFNode *)a;
    PetscSFNode *B = (PetscSFNode *)b;

    if (B->rank < 0) {
      *B = *A;
    }
  }
}

#undef __FUNCT__
#define __FUNCT__ "DMPforestGetTransferSF_Internal"
/* children are sf leaves of parents */
static PetscErrorCode DMPforestGetTransferSF_Internal(DM coarse, DM fine, const PetscInt dofPerDim[], PetscSF *sf, PetscBool transferIdent)
{
  MPI_Comm          comm;
  PetscMPIInt       rank, size;
  DM_Forest_pforest *pforestC, *pforestF;
  p4est_t           *p4estC, *p4estF;
  PetscInt          numClosureIndices, *closureIndices;
  PetscInt          numClosurePointsC, numClosurePointsF;
  PetscSFNode       *closurePointsC, *closurePointsF;
  p4est_quadrant_t  *coverQuads = NULL;
  p4est_quadrant_t  **treeQuads;
  PetscInt          *treeQuadCounts;
  MPI_Datatype      nodeType;
  MPI_Datatype      nodeClosureType;
  MPI_Op            sfNodeReduce;
  p4est_topidx_t    fltF, lltF, t;
  DM                plexC;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  pforestC = (DM_Forest_pforest *) ((DM_Forest *) coarse->data)->data;
  pforestF = (DM_Forest_pforest *) ((DM_Forest *) fine->data)->data;
  p4estC   = pforestC->forest;
  p4estF   = pforestF->forest;
  if (pforestC->topo != pforestF->topo) SETERRQ(PetscObjectComm((PetscObject)coarse),PETSC_ERR_ARG_INCOMP,"DM's must have the same base DM");
  comm = PetscObjectComm((PetscObject)coarse);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);

  /* count the number of closure points that have dofs and create a list */
  numClosureIndices = 0;
  if (dofPerDim[P4EST_DIM]     > 0) numClosureIndices += 1;
  if (dofPerDim[P4EST_DIM - 1] > 0) numClosureIndices += P4EST_FACES;
#if defined(P4_TO_P8)
  if (dofPerDim[P4EST_DIM - 2] > 0) numClosureIndices += P8EST_EDGES;
#endif
  if (dofPerDim[0]             > 0) numClosureIndices += P4EST_CHILDREN;
  ierr = PetscMalloc1(numClosureIndices,&closureIndices);CHKERRQ(ierr);
  {
    PetscInt count = 0, offset = 0;
    if (dofPerDim[P4EST_DIM] > 0) {
      closureIndices[count++] = offset;
    }
    offset++;
    if (dofPerDim[P4EST_DIM - 1] > 0) {
      PetscInt i;

      for (i = 0; i < P4EST_FACES; i++) {
        closureIndices[count + i] = offset + i;
      }
      count += P4EST_FACES;
    }
    offset += P4EST_FACES;
#if defined(P4_TO_P8)
    if (dofPerDim[P4EST_DIM - 2] > 0) {
      PetscInt i;

      for (i = 0; i < P8EST_EDGES; i++) {
        closureIndices[count + i] = offset + i;
      }
      count += P8EST_EDGES;
    }
    offset += P8EST_EDGES;
#endif
    if (dofPerDim[0] > 0) {
      PetscInt i;

      for (i = 0; i < P4EST_CHILDREN; i++) {
        closureIndices[count + i] = offset + i;
      }
    }
  }
  /* create the datatype */
  ierr = MPI_Type_contiguous(2,MPIU_INT,&nodeType);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&nodeType);CHKERRQ(ierr);
  ierr = MPI_Op_create(DMPforestMaxSFNode,PETSC_FALSE,&sfNodeReduce);CHKERRQ(ierr);
  ierr = MPI_Type_contiguous(numClosureIndices*2,MPIU_INT,&nodeClosureType);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&nodeClosureType);CHKERRQ(ierr);
  /* everything has to go through cells: for each cell, create a list of the sfnodes in its closure */
  /* get lists of closure point SF nodes for every cell */
  ierr = DMPforestGetCellSFNodes(coarse,numClosureIndices,closureIndices,&numClosurePointsC,&closurePointsC,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPforestGetCellSFNodes(fine  ,numClosureIndices,closureIndices,&numClosurePointsF,&closurePointsF,PETSC_FALSE);CHKERRQ(ierr);
  /* create pointers for tree lists */
  fltF = p4estF->first_local_tree;
  lltF = p4estF->last_local_tree;
  ierr = PetscCalloc2(lltF + 1  - fltF, &treeQuads, lltF + 1 - fltF, &treeQuadCounts);CHKERRQ(ierr);
  ierr = DMPforestGetPlex(coarse,&plexC);CHKERRQ(ierr);
  /* if the partitions don't match, ship the coarse to cover the fine */
  if (size > 1) {
    PetscInt p;
    for (p = 0; p < size; p++) {
      int equal;

      PetscStackCallP4estReturn(equal,p4est_quadrant_is_equal_piggy,(&p4estC->global_first_position[p],&p4estF->global_first_position[p]));
      if (!equal) {
        break;
      }
    }
    if (p < size) { /* non-matching distribution: send the coarse to cover the fine */
      PetscInt cStartC, cEndC;
      PetscSF coveringSF;
      PetscInt nleaves;
      PetscInt count;
      PetscSFNode *newClosurePointsC;
      p4est_quadrant_t *coverQuadsSend;
      p4est_topidx_t fltC = p4estC->first_local_tree;
      p4est_topidx_t lltC = p4estC->last_local_tree;
      p4est_topidx_t t;
      PetscMPIInt blockSizes[5] = {P4EST_DIM,2,1,1,1};
      MPI_Aint    blockOffsets[5] = {offsetof(p4est_quadrant_t,x),
                                    offsetof(p4est_quadrant_t,level),
                                    offsetof(p4est_quadrant_t,pad16),
                                    offsetof(p4est_quadrant_t,p),
                                    sizeof (p4est_quadrant_t)};
      MPI_Datatype blockTypes[5] = {MPI_INT32_T,MPI_INT8_T,MPI_INT16_T,MPI_INT32_T,MPI_UB};
      MPI_Datatype quadType;
      ierr = DMPlexGetHeightStratum(plexC,0,&cStartC,&cEndC);CHKERRQ(ierr);
      ierr = DMPforestGetCellCoveringSF(comm,p4estC,p4estF,cStartC,cEndC,pforestC->cLocalStart,&coveringSF);CHKERRQ(ierr);
      ierr = PetscSFGetGraph(coveringSF,NULL,&nleaves,NULL,NULL);CHKERRQ(ierr);
      ierr = PetscMalloc1(numClosureIndices*nleaves,&newClosurePointsC);CHKERRQ(ierr);
      ierr = PetscMalloc1(nleaves,&coverQuads);CHKERRQ(ierr);
      ierr = PetscMalloc1(cEndC-cStartC,&coverQuadsSend);CHKERRQ(ierr);
      count = 0;
      for (t = fltC; t < lltC; t++) { /* unfortunately, we need to pack a send array, since quads are not stored packed in p4est */
        p4est_tree_t *tree = &(((p4est_tree_t *) p4estC->trees->array)[t]);
        PetscInt q;

        ierr = PetscMemcpy(&coverQuadsSend[count],tree->quadrants.array,tree->quadrants.elem_count * sizeof(p4est_quadrant_t));CHKERRQ(ierr);
        for (q = 0; q < tree->quadrants.elem_count; q++) {
          coverQuadsSend[count+q].p.which_tree = t;
        }
        count += tree->quadrants.elem_count;
      }
      ierr = MPI_Type_create_struct(5,blockSizes,blockOffsets,blockTypes,&quadType);CHKERRQ(ierr);
      ierr = MPI_Type_commit(&quadType);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(coveringSF,nodeClosureType,closurePointsC,newClosurePointsC);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(coveringSF,quadType,coverQuadsSend,coverQuads);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(coveringSF,nodeClosureType,closurePointsC,newClosurePointsC);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(coveringSF,quadType,coverQuadsSend,coverQuads);CHKERRQ(ierr);
      ierr = MPI_Type_free(&quadType);CHKERRQ(ierr);
      ierr = PetscFree(coverQuadsSend);CHKERRQ(ierr);
      ierr = PetscFree(closurePointsC);CHKERRQ(ierr);
      ierr = PetscSFDestroy(&coveringSF);CHKERRQ(ierr);
      closurePointsC = newClosurePointsC;

      /* assign tree quads based on locations in coverQuads */
      {
        PetscInt q;
        for (q = 0; q < nleaves; q++) {
          p4est_locidx_t t = coverQuads[q].p.which_tree;
          if (!treeQuadCounts[t-fltF]++) {
            treeQuads[t-fltF] = &coverQuads[q];
          }
        }
      }
    }
  }
  if (!coverQuads) { /* matching partitions: assign tree quads based on locations in p4est native arrays */
    for (t = fltF; t <= lltF; t++) {
      p4est_tree_t *tree = &(((p4est_tree_t *) p4estC->trees->array)[t]);

      treeQuadCounts[t - fltF] = tree->quadrants.elem_count;
      treeQuads[t - fltF] = (p4est_quadrant_t *) tree->quadrants.array;
    }
  }

  {
    PetscInt pStartF, pEndF, p;
    PetscInt cLocalStartF;
    PetscSF  pointSF;
    PetscSFNode *roots;
    DM       plexF;

    ierr = DMPforestGetPlex(fine,&plexF);CHKERRQ(ierr);
    ierr = DMPlexGetChart(plexF,&pStartF,&pEndF);CHKERRQ(ierr);
    ierr = PetscMalloc1(pEndF-pStartF,&roots);CHKERRQ(ierr);
    ierr = DMGetPointSF(fine,&pointSF);CHKERRQ(ierr);
    for (p = pStartF; p < pEndF; p++) {
      roots[p-pStartF].rank  = -1;
      roots[p-pStartF].index = -1;
    }
    cLocalStartF = pforestF->cLocalStart;
    for (t = fltF; t < lltF; t++) {
      p4est_tree_t *tree = &(((p4est_tree_t *) p4estF->trees->array)[t]);
      PetscInt numCoarseQuads = treeQuadCounts[t - fltF];
      PetscInt numFineQuads = tree->quadrants.elem_count;
      p4est_quadrant_t *coarseQuads = treeQuads[t - fltF];
      p4est_quadrant_t *fineQuads = (p4est_quadrant_t *) tree->quadrants.array;
      PetscInt i, coarseCount = 0;
      PetscInt offset = cLocalStartF + tree->quadrants_offset;
      sc_array_t coarseQuadsArray;

      PetscStackCallP4est(sc_array_init_data,(&coarseQuadsArray,coarseQuads,sizeof(p4est_quadrant_t),(size_t) numCoarseQuads));
      for (i = 0; i < numFineQuads; i++) {
        PetscInt c = i + offset;
        p4est_quadrant_t *quad = &fineQuads[i];
        p4est_quadrant_t *quadCoarse;
        ssize_t overlap = -1;

        while (overlap < 0 && coarseCount < numCoarseQuads) {
          quadCoarse = &coarseQuads[coarseCount];
          PetscStackCallP4estReturn(overlap,p4est_quadrant_disjoint,(quadCoarse,quad));
          if (overlap < 0) {
            coarseCount++;
          }
        }
        if (overlap != 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"did not find overlapping coarse quad");
        if (quadCoarse->level > quad->level || (quadCoarse->level == quad->level && !transferIdent)) { /* the "coarse" mesh is finer than the fine mesh at the point: continue */
          continue;
        }
        if (quadCoarse->level == quad->level) { /* same quad present in coarse and fine mesh */
          PetscInt j;
          for (j = 0; j < numClosureIndices; j++) {
            PetscInt p = closurePointsF[numClosureIndices * c + j].index;

            roots[p-pStartF] = closurePointsC[numClosureIndices * coarseCount + j];
          }
        }
        else {
          p4est_qcoord_t coarseBound[2][P4EST_DIM] = {{quadCoarse->x,quadCoarse->y,
#if defined(P4_TO_P8)
                                                   quadCoarse->z
#endif
          },{0}};
          p4est_qcoord_t fineBound[2][P4EST_DIM] = {{quad->x,quad->y,
#if defined(P4_TO_P8)
                                                 quad->z
#endif
          },{0}};
          PetscInt j;
          for (j = 0; j < P4EST_DIM; j++) { /* get the coordinates of cell boundaries in each direction */
            coarseBound[1][j] = coarseBound[0][j] + P4EST_QUADRANT_LEN(quadCoarse->level);
            fineBound[1][j]   = fineBound[0][j]   + P4EST_QUADRANT_LEN(quad->level);
          }
          for (j = 0; j < numClosureIndices; j++) {
            PetscInt k = closureIndices[j];
            PetscInt l, m;

            if (k == 0) { /* volume: ancestor is volume */
              l = 0;
            }
            else if (k < 1 + P4EST_FACES) { /* facet */
              PetscInt face = PetscFaceToP4estFace[k - 1];
              PetscInt direction = face / 2;
              PetscInt coarseFace = -1;

              if (coarseBound[face % 2][direction] == fineBound[face % 2][direction]) {
                coarseFace = face;
                l = 1 + P4estFaceToPetscFace[coarseFace];
              }
              else {
                l = 0;
              }
            }
#if defined(P4_TO_P8)
            else if (k < 1 + P4EST_FACES + P8EST_EDGES) {
              PetscInt edge = PetscEdgeToP4estEdge[k - (1 + P4EST_FACES)];
              PetscInt direction = edge / 4;
              PetscInt mod = edge % 4;
              PetscInt coarseEdge = -1, coarseFace = -1;
              PetscInt minDir = PetscMin((direction + 1) % 3,(direction + 2) % 3);
              PetscInt maxDir = PetscMax((direction + 1) % 3,(direction + 2) % 3);
              PetscBool dirTest[2];

              dirTest[0] = (coarseBound[mod % 2][minDir] == fineBound[mod % 2][minDir]);
              dirTest[1] = (coarseBound[mod / 2][maxDir] == fineBound[mod / 2][maxDir]);

              if (dirTest[0] && dirTest[1]) { /* fine edge falls on coarse edge */
                coarseEdge = edge;
                l = 1 + P4EST_FACES + P4estEdgeToPetscEdge[coarseEdge];
              }
              else if (dirTest[0]) { /* fine edge falls on a coarse face in the minDir direction */
                coarseFace = 2 * minDir + (mod % 2);
                l = 1 + P4estFaceToPetscFace[coarseFace];
              }
              else if (dirTest[1]) { /* fine edge falls on a coarse face in the maxDir direction */
                coarseFace = 2 * maxDir + (mod / 2);
                l = 1 + P4estFaceToPetscFace[coarseFace];
              }
              else {
                l = 0;
              }
            }
#endif
            else {
              PetscInt vertex = PetscVertToP4estVert[P4EST_CHILDREN - (P4EST_INSUL - k)];
              PetscBool dirTest[P4EST_DIM];
              PetscInt m;
              PetscInt numMatch = 0;
              PetscInt coarseVertex = -1, coarseFace = -1;
#if defined(P4_TO_P8)
              PetscInt coarseEdge = -1;
#endif

              for (m = 0; m < P4EST_DIM; m++) {
                dirTest[m] = (coarseBound[(vertex >> m) & 1][m] == fineBound[(vertex >> m) & 1][m]);
                if (dirTest[m]) {
                  numMatch++;
                }
              }
              if (numMatch == P4EST_DIM) { /* vertex on vertex */
                coarseVertex = vertex;
                l = P4EST_INSUL - (P4EST_CHILDREN - P4estVertToPetscVert[coarseVertex]);
              }
              else if (numMatch == 1) { /* vertex on face */
                for (m = 0; m < P4EST_DIM; m++) {
                  if (dirTest[m]) {
                    coarseFace = 2 * m + ((vertex >> m) & 1);
                    break;
                  }
                }
                l = 1 + P4estFaceToPetscFace[coarseFace];
              }
#if defined(P4_TO_P8)
              else if (numMatch == 2) { /* vertex on edge */
                for (m = 0; m < P4EST_DIM; m++) {
                  if (!dirTest[m]) {
                    PetscInt otherDir1 = (m + 1) % 3;
                    PetscInt otherDir2 = (m + 2) % 3;
                    PetscInt minDir = PetscMin(otherDir1,otherDir2);
                    PetscInt maxDir = PetscMax(otherDir1,otherDir2);

                    coarseEdge = m * 4 + maxDir * 2 * ((vertex >> maxDir) & 1) + minDir * ((vertex >> minDir) & 1);
                    break;
                  }
                }
                l = 1 + P4EST_FACES + P4estEdgeToPetscEdge[coarseEdge];
              }
#endif
              else { /* volume */
                l = 0;
              }
            }
            for (m = 0; m < numClosureIndices; m++) {
              if (closureIndices[m] == l) {
                PetscInt p = closurePointsF[numClosureIndices * c + j].index;

                roots[p-pStartF] = closurePointsC[numClosureIndices * coarseCount + m];
                break;
              }
            }
          }
        }
      }
    }

    /* now every cell has labeled the points in its closure, so we first make sure everyone agrees by reducing to roots, and the broadcast the agreements */
    ierr = PetscSFReduceBegin(pointSF,nodeType,roots,roots,sfNodeReduce);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(pointSF,nodeType,roots,roots,sfNodeReduce);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(pointSF,nodeType,roots,roots);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(pointSF,nodeType,roots,roots);CHKERRQ(ierr);

    {
      PetscInt pStartC, pEndC;
      PetscInt numRoots;
      PetscInt numLeaves;
      PetscInt *leaves, *remoteOffsets;
      PetscInt d;
      PetscSFNode *iremote;
      PetscSF  pointTransferSF;
      PetscSection leafSection, rootSection;
      /* count leaves */

      ierr = DMPlexGetChart(plexC,&pStartC,&pEndC);CHKERRQ(ierr);
      numRoots = pEndC - pStartC;

      numLeaves = 0;
      for (p = pStartF; p < pEndF; p++) {
        if (roots[p-pStartF].index >= 0) {
          numLeaves++;
        }
      }
      ierr = PetscMalloc1(numLeaves,&leaves);CHKERRQ(ierr);
      ierr = PetscMalloc1(numLeaves,&iremote);CHKERRQ(ierr);
      numLeaves = 0;
      for (p = pStartF; p < pEndF; p++) {
        if (roots[p-pStartF].index >= 0) {
          leaves[numLeaves] = p-pStartF;
          iremote[numLeaves] = roots[p-pStartF];
          numLeaves++;
        }
      }
      ierr = PetscFree(roots);CHKERRQ(ierr);
      ierr = PetscSFCreate(comm,&pointTransferSF);CHKERRQ(ierr);
      ierr = PetscSFSetGraph(pointTransferSF,numRoots,numLeaves,leaves,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
      ierr = PetscSectionCreate(PETSC_COMM_SELF,&leafSection);CHKERRQ(ierr);
      ierr = PetscSectionCreate(PETSC_COMM_SELF,&rootSection);CHKERRQ(ierr);
      ierr = PetscSectionSetChart(rootSection,pStartC,pEndC);CHKERRQ(ierr);
      for (d = 0; d <= P4EST_DIM; d++) {
        PetscInt startC, endC, startF, endF, e;

        ierr = DMPlexGetDepthStratum(plexC,d,&startC,&endC);CHKERRQ(ierr);
        ierr = DMPlexGetDepthStratum(plexF,d,&startF,&endF);CHKERRQ(ierr);
        for (e = startC; e < endC; e++) {
          ierr = PetscSectionSetDof(rootSection,e,dofPerDim[d]);CHKERRQ(ierr);
        }
      }
      ierr = PetscSectionSetUp(rootSection);CHKERRQ(ierr);
      ierr = PetscSFDistributeSection(pointTransferSF,rootSection,&remoteOffsets,leafSection);CHKERRQ(ierr);
      ierr = PetscSFCreateSectionSF(pointTransferSF,rootSection,remoteOffsets,leafSection,sf);CHKERRQ(ierr);
      ierr = PetscSectionDestroy(&leafSection);CHKERRQ(ierr);
      ierr = PetscSectionDestroy(&rootSection);CHKERRQ(ierr);
      ierr = PetscSFDestroy(&pointTransferSF);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree2(treeQuads,treeQuadCounts);CHKERRQ(ierr);
  ierr = PetscFree(coverQuads);CHKERRQ(ierr);
  ierr = PetscFree(closurePointsC);CHKERRQ(ierr);
  ierr = PetscFree(closurePointsF);CHKERRQ(ierr);
  ierr = PetscFree(closureIndices);CHKERRQ(ierr);
  ierr = MPI_Type_free(&nodeClosureType);CHKERRQ(ierr);
  ierr = MPI_Op_free(&sfNodeReduce);CHKERRQ(ierr);
  ierr = MPI_Type_free(&nodeType);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPforestGetTransferSF"
static PetscErrorCode DMPforestGetTransferSF(DM dmA, DM dmB, const PetscInt dofPerDim[], PetscSF *sfAtoB, PetscSF *sfBtoA)
{
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (sfAtoB) {
    ierr = DMPforestGetTransferSF_Internal(dmA,dmB,dofPerDim,sfAtoB,PETSC_TRUE);CHKERRQ(ierr);
  }
  if (sfBtoA) {
    ierr = DMPforestGetTransferSF_Internal(dmB,dmA,dofPerDim,sfBtoA,(sfAtoB == NULL));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPforestGetLabelsFinalized"
static PetscErrorCode DMPforestGetLabelsFinalized(DM dm, PetscBool *finalized)
{
  DM_Forest_pforest *pforest = (DM_Forest_pforest *) ((DM_Forest *)dm->data)->data;

  PetscFunctionBegin;
  *finalized = pforest->labelsFinalized;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPforestLabelsInitialize"
static PetscErrorCode DMPforestLabelsInitialize(DM dm, DM plex)
{
  DM_Forest         *forest  = (DM_Forest *) dm->data;
  DM_Forest_pforest *pforest = (DM_Forest_pforest *) forest->data;
  PetscInt          cLocalStart, cLocalEnd, cStart, cEnd, fStart, fEnd, eStart, eEnd, vStart, vEnd;
  PetscInt          cStartBase, cEndBase, fStartBase, fEndBase, vStartBase, vEndBase, eStartBase, eEndBase;
  PetscInt          pStart, pEnd, p;
  DM                base;
  PetscInt          *star = NULL, starSize;
  DMLabelLink       next = dm->labels->next;
  PetscInt          guess = 0;
  p4est_topidx_t    num_trees = pforest->topo->conn->num_trees;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMForestGetBaseDM(dm,&base);CHKERRQ(ierr);
  cLocalStart = pforest->cLocalStart;
  cLocalEnd   = pforest->cLocalEnd;
  ierr = DMPlexGetHeightStratum(base,0,&cStartBase,&cEndBase);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(base,1,&fStartBase,&fEndBase);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(base,1,&eStartBase,&eEndBase);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(base,0,&vStartBase,&vEndBase);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(plex,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(plex,1,&fStart,&fEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(plex,1,&eStart,&eEnd);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(plex,0,&vStart,&vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetChart(plex,&pStart,&pEnd);CHKERRQ(ierr);
  /* go through the mesh: use star to find a quadrant that borders a point.  Use the closure to determine the
   * orientation of the quadrant relative to that point.  Use that to relate the point to the numbering in the base
   * mesh, and extract a label value (since the base mesh is redundantly distributed, can be found locally). */
  while (next) {
    DMLabel baseLabel;
    DMLabel label = next->label;
    PetscBool isDepth;

    ierr = PetscStrcmp(label->name,"depth",&isDepth);CHKERRQ(ierr);
    if (isDepth) {
      next = next->next;
      continue;
    }
    ierr = DMGetLabel(base,label->name,&baseLabel);CHKERRQ(ierr);
    ierr = DMLabelCreateIndex(baseLabel,pStart,pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; p++) {
      PetscInt s, c = -1, l;
      PetscInt *closure = NULL, closureSize;
      p4est_quadrant_t * ghosts = (p4est_quadrant_t *) pforest->ghost->ghosts.array;
      p4est_tree_t *trees = (p4est_tree_t *) pforest->forest->trees->array;
      p4est_quadrant_t * q;
      PetscInt t, val;

      ierr = DMPlexGetTransitiveClosure(plex,p,PETSC_FALSE,&starSize,&star);CHKERRQ(ierr);
      for (s = 0; s < starSize; s++) {
        PetscInt point = star[2*s];

        if (cStart <= point && point < cEnd) {
          ierr = DMPlexGetTransitiveClosure(plex,point,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
          for (l = 0; l < closureSize; l++) {
            if (closure[2 * l] == p) {
              c = point;
              break;
            }
          }
          ierr = DMPlexRestoreTransitiveClosure(plex,point,PETSC_TRUE,NULL,&closure);CHKERRQ(ierr);
          if (l < closureSize) {
            break;
          }
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(plex,p,PETSC_FALSE,NULL,&star);CHKERRQ(ierr);
      if (s == starSize) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Failed to find cell with point %d in its closure",p);

      if (c < cLocalStart) {
        /* get from the beginning of the ghost layer */
        q = &(ghosts[c]);
        t = (PetscInt) q->p.which_tree;
      }
      else if (c < cLocalEnd) {
        PetscInt lo = 0, hi = num_trees;
        /* get from local quadrants: have to find the right tree */

        c -= cLocalStart;

        do {
          p4est_tree_t *tree;

          if (guess < lo || guess >= num_trees || lo >= hi) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"failed binary search");
          tree = &trees[guess];
          if (c < tree->quadrants_offset) {
            hi = guess;
          }
          else if (c < tree->quadrants_offset + (PetscInt) tree->quadrants.elem_count) {
            q = &((p4est_quadrant_t *)tree->quadrants.array)[c - (PetscInt) tree->quadrants_offset];
            t = guess;
            break;
          }
          else {
            lo = guess + 1;
          }
          guess = lo + (hi - lo) / 2;
        } while (1);
      }
      else {
        /* get from the end of the ghost layer */
        c -= (cLocalEnd - cLocalEnd);

        q = &(ghosts[c]);
        t = (PetscInt) q->p.which_tree;
      }

      if (l == 0) { /* cell */
        ierr = DMLabelGetValue(baseLabel,t+cStartBase,&val);CHKERRQ(ierr);
        ierr = DMLabelSetValue(label,p,val);CHKERRQ(ierr);
      }
      else if (l >= 1 && l < 1 + P4EST_FACES) { /* facet */
        p4est_quadrant_t nq;
        int              isInside;

        l = PetscFaceToP4estFace[l - 1];
        PetscStackCallP4est(p4est_quadrant_face_neighbor,(q,l,&nq));
        PetscStackCallP4estReturn(isInside,p4est_quadrant_is_inside_root,(&nq));
        if (isInside) {
          /* this facet is in the interior of a tree, so it inherits the label of the tree */
          ierr = DMLabelGetValue(baseLabel,t+cStartBase,&val);CHKERRQ(ierr);
          ierr = DMLabelSetValue(label,p,val);CHKERRQ(ierr);
        }
        else {
          PetscInt f = pforest->topo->tree_face_to_uniq[P4EST_FACES * t + l];

          ierr = DMLabelGetValue(baseLabel,f+fStartBase,&val);CHKERRQ(ierr);
          ierr = DMLabelSetValue(label,p,val);CHKERRQ(ierr);
        }
      }
#if defined(P4_TO_P8)
      else if (l >= 1 + P4EST_FACES && l < 1 + P4EST_FACES + P8EST_EDGES) { /* edge */
        p4est_quadrant_t nq;
        int              isInside;

        l = PetscEdgeToP4estEdge[l - (1 + P4EST_FACES)];
        PetscStackCallP4est(p8est_quadrant_edge_neighbor,(q,l,&nq));
        PetscStackCallP4estReturn(isInside,p4est_quadrant_is_inside_root,(&nq));
        if (isInside) {
          /* this edge is in the interior of a tree, so it inherits the label of the tree */
          ierr = DMLabelGetValue(baseLabel,t+cStartBase,&val);CHKERRQ(ierr);
          ierr = DMLabelSetValue(label,p,val);CHKERRQ(ierr);
        }
        else {
          int isOutsideFace;

          PetscStackCallP4estReturn(isOutsideFace,p4est_quadrant_is_outside_face,(&nq));
          if (isOutsideFace) {
            PetscInt f;

            if (nq.x < 0) {
              f = 0;
            }
            else if (nq.x >= P4EST_ROOT_LEN) {
              f = 1;
            }
            else if (nq.y < 0) {
              f = 2;
            }
            else if (nq.y >= P4EST_ROOT_LEN) {
              f = 3;
            }
            else if (nq.z < 0) {
              f = 4;
            }
            else {
              f = 5;
            }
            f = pforest->topo->tree_face_to_uniq[P4EST_FACES * t + f];
            ierr = DMLabelGetValue(baseLabel,f+fStartBase,&val);CHKERRQ(ierr);
            ierr = DMLabelSetValue(label,p,val);CHKERRQ(ierr);
          }
          else { /* the quadrant edge corresponds to the tree edge */
            PetscInt e = pforest->topo->conn->tree_to_edge[P8EST_EDGES * t + l];

            ierr = DMLabelGetValue(baseLabel,e+eStartBase,&val);CHKERRQ(ierr);
            ierr = DMLabelSetValue(label,p,val);CHKERRQ(ierr);
          }
        }
      }
#endif
      else { /* vertex */
        p4est_quadrant_t nq;
        int              isInside;

#if defined(P4_TO_P8)
        l = PetscVertToP4estVert[l - (1 + P4EST_FACES + P8EST_CHILDREN)];
#else
        l = PetscVertToP4estVert[l - (1 + P4EST_FACES)];
#endif
        PetscStackCallP4est(p4est_quadrant_corner_neighbor,(q,l,&nq));
        PetscStackCallP4estReturn(isInside,p4est_quadrant_is_inside_root,(&nq));
        if (isInside) {
          ierr = DMLabelGetValue(baseLabel,t+cStartBase,&val);CHKERRQ(ierr);
          ierr = DMLabelSetValue(label,p,val);CHKERRQ(ierr);
        }
        else {
          int isOutside;

          PetscStackCallP4estReturn(isOutside,p4est_quadrant_is_outside_face,(&nq));
          if (isOutside) {
            PetscInt f = -1;

            if (nq.x < 0) {
              f = 0;
            }
            else if (nq.x >= P4EST_ROOT_LEN) {
              f = 1;
            }
            else if (nq.y < 0) {
              f = 2;
            }
            else if (nq.y >= P4EST_ROOT_LEN) {
              f = 3;
            }
#if defined(P4_TO_P8)
            else if (nq.z < 0) {
              f = 4;
            }
            else {
              f = 5;
            }
#endif
            f = pforest->topo->tree_face_to_uniq[P4EST_FACES * t + f];
            ierr = DMLabelGetValue(baseLabel,f+fStartBase,&val);CHKERRQ(ierr);
            ierr = DMLabelSetValue(label,p,val);CHKERRQ(ierr);
            continue;
          }
#if defined(P4_TO_P8)
          PetscStackCallP4estReturn(isOutside,p8est_quadrant_is_outside_edge,(&nq));
          if (isOutside) {
            /* outside edge */
            PetscInt e = -1;

            if (nq.x >= 0 && nq.x < P4EST_ROOT_LEN) {
              if (nq.z < 0) {
                if (nq.y < 0) {
                  e = 0;
                }
                else {
                  e = 1;
                }
              }
              else {
                if (nq.y < 0) {
                  e = 2;
                }
                else {
                  e = 3;
                }
              }
            }
            else if (nq.y >= 0 && nq.y < P4EST_ROOT_LEN) {
              if (nq.z < 0) {
                if (nq.x < 0) {
                  e = 4;
                }
                else {
                  e = 5;
                }
              }
              else {
                if (nq.x < 0) {
                  e = 6;
                }
                else {
                  e = 7;
                }
              }
            }
            else {
              if (nq.y < 0) {
                if (nq.x < 0) {
                  e = 8;
                }
                else {
                  e = 9;
                }
              }
              else {
                if (nq.x < 0) {
                  e = 10;
                }
                else {
                  e = 11;
                }
              }
            }

            e = pforest->topo->conn->tree_to_edge[P8EST_EDGES * t + e];
            ierr = DMLabelGetValue(baseLabel,e+eStartBase,&val);CHKERRQ(ierr);
            ierr = DMLabelSetValue(label,p,val);CHKERRQ(ierr);
            continue;
          }
#endif
          else {
            /* outside vertex: same corner as quadrant corner */
            PetscInt v = pforest->topo->conn->tree_to_corner[P4EST_CHILDREN * t + l];

            ierr = DMLabelGetValue(baseLabel,v+vStartBase,&val);CHKERRQ(ierr);
            ierr = DMLabelSetValue(label,p,val);CHKERRQ(ierr);
          }
        }
      }
    }
    next = next->next;
  }
  pforest->labelsFinalized = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPforestLabelsFinalize"
static PetscErrorCode DMPforestLabelsFinalize(DM dm, DM plex)
{
  DM_Forest_pforest *pforest = (DM_Forest_pforest *) ((DM_Forest *) dm->data)->data;
  DM                adapt;
  PetscBool         labelsFinalized = PETSC_TRUE;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (pforest->labelsFinalized) PetscFunctionReturn(0);
  ierr = DMGetCoarseDM(dm,&adapt);CHKERRQ(ierr);
  if (adapt) {
    ierr = DMPforestGetLabelsFinalized(adapt,&labelsFinalized);CHKERRQ(ierr);
  }
  if (!adapt || !labelsFinalized) {
    ierr = DMGetFineDM(dm,&adapt);CHKERRQ(ierr);
    if (adapt) {
      ierr = DMPforestGetLabelsFinalized(adapt,&labelsFinalized);CHKERRQ(ierr);
    }
  }
  if (adapt && labelsFinalized) {
    PetscInt    dofPerDim[4]={1, 1, 1, 1};
    PetscSF     transferForward, transferBackward;
    PetscInt    pStart, pEnd, pStartA, pEndA;
    PetscInt    *values, *adaptValues;
    DMLabelLink next = adapt->labels->next;
    DM          adaptPlex;

    ierr = DMPforestGetPlex(adapt,&adaptPlex);CHKERRQ(ierr);
    ierr = DMPforestGetTransferSF(adapt,dm,dofPerDim,&transferForward,&transferBackward);CHKERRQ(ierr);
    ierr = DMPlexGetChart(plex,&pStart,&pEnd);CHKERRQ(ierr);
    ierr = DMPlexGetChart(adaptPlex,&pStartA,&pEndA);CHKERRQ(ierr);
    ierr = PetscMalloc2(pEnd-pStart,&values,pEndA-pStartA,&adaptValues);CHKERRQ(ierr);
    while (next) {
      DMLabel adaptLabel = next->label;
      const char *name = adaptLabel->name;
      PetscBool isDepth;
      PetscInt  p;

      ierr = PetscStrcmp(name,"depth",&isDepth);CHKERRQ(ierr);
      if (!isDepth) {
        DMLabel label;

        /* label was created earlier */
        ierr = DMGetLabel(dm,name,&label);CHKERRQ(ierr);

        for (p = pStartA; p < pEndA; p++) {
          ierr = DMLabelGetValue(adaptLabel,p,&adaptValues[p]);CHKERRQ(ierr);
        }

        if (transferForward) {
          ierr = PetscSFBcastBegin(transferForward,MPIU_INT,adaptValues,values);CHKERRQ(ierr);
        }
        if (transferBackward) {
          ierr = PetscSFReduceBegin(transferBackward,MPIU_INT,adaptValues,values,MPIU_MAX);CHKERRQ(ierr);
        }
        if (transferForward) {
          ierr = PetscSFBcastEnd(transferForward,MPIU_INT,adaptValues,values);CHKERRQ(ierr);
        }
        if (transferBackward) {
          ierr = PetscSFReduceEnd(transferBackward,MPIU_INT,adaptValues,values,MPIU_MAX);CHKERRQ(ierr);
        }

        for (p = pStart; p < pEnd; p++) {
          ierr = DMLabelSetValue(label,p,values[p]);CHKERRQ(ierr);
        }

      }
      next = next->next;
    }
    ierr = PetscFree2(values,adaptValues);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&transferForward);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&transferBackward);CHKERRQ(ierr);
    pforest->labelsFinalized = PETSC_TRUE;
  }
  else {
    /* Initialize labels from the base dm */
    ierr = DMPforestLabelsInitialize(dm,plex);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMConvert_pforest_plex)
static PetscErrorCode DMConvert_pforest_plex(DM dm, DMType newtype, DM *plex)
{
  DM_Forest      *forest;
  DM_Forest_pforest *pforest;
  DM             refTree, newPlex;
  PetscInt       adjDim, adjCodim, coordDim;
  MPI_Comm       comm;
  PetscBool      isPforest;
  PetscInt       dim;
  PetscInt       overlap;
  p4est_connect_type_t ctype;
  p4est_locidx_t first_local_quad = -1;
  sc_array_t     *points_per_dim, *cone_sizes, *cones, *cone_orientations, *coords, *children, *parents, *childids, *leaves, *remotes;
  PetscSection   parentSection;
  PetscSF        pointSF;
  size_t         zz, count;
  PetscInt       pStart, pEnd;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  comm = PetscObjectComm((PetscObject)dm);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMPFOREST,&isPforest);CHKERRQ(ierr);
  if (!isPforest) SETERRQ2(comm,PETSC_ERR_ARG_WRONG,"Expected DM type %s, got %s\n",DMPFOREST,((PetscObject)dm)->type_name);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if (dim != P4EST_DIM) SETERRQ2(comm,PETSC_ERR_ARG_WRONG,"Expected DM dimension %d, got %d\n",P4EST_DIM,dim);
  forest = (DM_Forest *) dm->data;
  pforest = (DM_Forest_pforest *) forest->data;
  if (!pforest->plex) {
    ierr = DMCreate(comm,&newPlex);CHKERRQ(ierr);
    ierr = DMSetType(newPlex,DMPLEX);CHKERRQ(ierr);
    ierr = PetscFree(newPlex->labels);CHKERRQ(ierr); /* share labels */
    dm->labels->refct++;
    newPlex->labels = dm->labels;
    ierr = DMForestGetAdjacencyDimension(dm,&adjDim);CHKERRQ(ierr);
    ierr = DMForestGetAdjacencyCodimension(dm,&adjCodim);CHKERRQ(ierr);
    ierr = DMGetCoordinateDim(dm,&coordDim);CHKERRQ(ierr);
    if (adjDim == 0) {
      ctype = P4EST_CONNECT_FULL;
    }
    else if (adjCodim == 1) {
      ctype = P4EST_CONNECT_FACE;
    }
#if defined(P4_TO_P8)
    else if (adjDim == 1) {
      ctype = P8EST_CONNECT_EDGE;
    }
#endif
    else {
      SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONG,"Invalid adjacency dimension %d",adjDim);
    }
    ierr = DMForestGetPartitionOverlap(dm,&overlap);CHKERRQ(ierr);

    points_per_dim    = sc_array_new(sizeof (p4est_locidx_t));
    cone_sizes        = sc_array_new(sizeof (p4est_locidx_t));
    cones             = sc_array_new(sizeof (p4est_locidx_t));
    cone_orientations = sc_array_new(sizeof (p4est_locidx_t));
    coords            = sc_array_new(3 * sizeof (double));
    children          = sc_array_new(sizeof (p4est_locidx_t));
    parents           = sc_array_new(sizeof (p4est_locidx_t));
    childids          = sc_array_new(sizeof (p4est_locidx_t));
    leaves            = sc_array_new(sizeof (p4est_locidx_t));
    remotes           = sc_array_new(2 * sizeof (p4est_locidx_t));

    PetscStackCallP4est(p4est_get_plex_data_ext,(pforest->forest,&pforest->ghost,&pforest->lnodes,ctype,(int)overlap,&first_local_quad,points_per_dim,cone_sizes,cones,cone_orientations,coords,children,parents,childids,leaves,remotes));

    pforest->cLocalStart = (PetscInt) first_local_quad;
    pforest->cLocalEnd   = pforest->cLocalStart + (PetscInt) pforest->forest->local_num_quadrants;
    ierr = locidx_to_PetscInt(points_per_dim);CHKERRQ(ierr);
    ierr = locidx_to_PetscInt(cone_sizes);CHKERRQ(ierr);
    ierr = locidx_to_PetscInt(cones);CHKERRQ(ierr);
    ierr = locidx_to_PetscInt(cone_orientations);CHKERRQ(ierr);
    ierr = coords_double_to_PetscScalar(coords, coordDim);CHKERRQ(ierr);
    ierr = locidx_to_PetscInt(children);CHKERRQ(ierr);
    ierr = locidx_to_PetscInt(parents);CHKERRQ(ierr);
    ierr = locidx_to_PetscInt(childids);CHKERRQ(ierr);
    ierr = locidx_to_PetscInt(leaves);CHKERRQ(ierr);
    ierr = locidx_pair_to_PetscSFNode(remotes);CHKERRQ(ierr);

    ierr = DMSetDimension(newPlex,P4EST_DIM);CHKERRQ(ierr);
    ierr = DMSetCoordinateDim(newPlex,coordDim);CHKERRQ(ierr);
    ierr = DMPlexCreateFromDAG(newPlex,P4EST_DIM,(PetscInt *)points_per_dim->array,(PetscInt *)cone_sizes->array,(PetscInt *)cones->array,(PetscInt *)cone_orientations->array,(PetscScalar *)coords->array);CHKERRQ(ierr);
    ierr = PetscSFCreate(comm,&pointSF);CHKERRQ(ierr);
    ierr = DMPlexCreateDefaultReferenceTree(comm,P4EST_DIM,PETSC_FALSE,&refTree);CHKERRQ(ierr);
    ierr = DMPlexSetReferenceTree(newPlex,refTree);CHKERRQ(ierr);
    ierr = DMDestroy(&refTree);CHKERRQ(ierr);
    ierr = PetscSectionCreate(comm,&parentSection);CHKERRQ(ierr);
    ierr = DMPlexGetChart(newPlex,&pStart,&pEnd);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(parentSection,pStart,pEnd);CHKERRQ(ierr);
    count = children->elem_count;
    for(zz = 0;zz < count;zz++) {
      PetscInt            child = *((PetscInt *) sc_array_index(children,zz));

      ierr = PetscSectionSetDof(parentSection,child,1);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(parentSection);CHKERRQ(ierr);
    ierr = DMPlexSetTree(newPlex,parentSection,(PetscInt *)parents->array,(PetscInt *)childids->array);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&parentSection);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(pointSF,pEnd - pStart,(PetscInt)leaves->elem_count,(PetscInt *)leaves->array,PETSC_COPY_VALUES,(PetscSFNode *)remotes->array,PETSC_COPY_VALUES);CHKERRQ(ierr);
    ierr = DMSetPointSF(newPlex,pointSF);CHKERRQ(ierr);
    ierr = DMSetPointSF(dm,pointSF);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&pointSF);CHKERRQ(ierr);
    pforest->plex = newPlex;

    sc_array_destroy (points_per_dim);
    sc_array_destroy (cone_sizes);
    sc_array_destroy (cones);
    sc_array_destroy (cone_orientations);
    sc_array_destroy (coords);
    sc_array_destroy (children);
    sc_array_destroy (parents);
    sc_array_destroy (childids);
    sc_array_destroy (leaves);
    sc_array_destroy (remotes);

    /* copy labels */
    ierr = DMPforestLabelsFinalize(dm,newPlex);CHKERRQ(ierr);

    if (forest->setFromOptions) {
      ierr = PetscObjectOptionsBegin((PetscObject)newPlex);CHKERRQ(ierr);
      ierr = DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject,newPlex);CHKERRQ(ierr);
      ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject) newPlex);CHKERRQ(ierr);
      ierr = PetscOptionsEnd();CHKERRQ(ierr);
    }
  }
  newPlex = pforest->plex;
  if (plex) {
    DM      coordDM;

    ierr = DMClone(newPlex,plex);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(newPlex,&coordDM);CHKERRQ(ierr);
    ierr = DMSetCoordinateDM(*plex,coordDM);CHKERRQ(ierr);

    ierr = DMShareDiscretization(dm,*plex);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMSetFromOptions_pforest)
static PetscErrorCode DMSetFromOptions_pforest(PetscOptionItems *PetscOptionsObject,DM dm)
{
  DM_Forest_pforest *pforest = (DM_Forest_pforest *) ((DM_Forest *) dm->data)->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMSetFromOptions_Forest(PetscOptionsObject,dm);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"DM" P4EST_STRING " options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_p4est_partition_for_coarsening","partition forest to allow for coarsening","DMP4estSetPartitionForCoarsening",pforest->partition_for_coarsening,&(pforest->partition_for_coarsening),NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if !defined(P4_TO_P8)
#define DMPforestGetPartitionForCoarsening DMP4estGetPartitionForCoarsening
#define DMPforestSetPartitionForCoarsening DMP4estSetPartitionForCoarsening
#else
#define DMPforestGetPartitionForCoarsening DMP8estGetPartitionForCoarsening
#define DMPforestSetPartitionForCoarsening DMP8estSetPartitionForCoarsening
#endif

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMPforestGetPartitionForCoarsening)
PETSC_EXTERN PetscErrorCode DMPforestGetPartitionForCoarsening(DM dm, PetscBool *flg)
{
  DM_Forest_pforest *pforest;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  pforest = (DM_Forest_pforest *) ((DM_Forest *) dm->data)->data;
  *flg = pforest->partition_for_coarsening;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMPforestSetPartitionForCoarsening)
PETSC_EXTERN PetscErrorCode DMPforestSetPartitionForCoarsening(DM dm, PetscBool flg)
{
  DM_Forest_pforest *pforest;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  pforest = (DM_Forest_pforest *) ((DM_Forest *) dm->data)->data;
  pforest->partition_for_coarsening = flg;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "DMPforestGetPlex"
static PetscErrorCode DMPforestGetPlex(DM dm,DM *plex)
{
  DM_Forest_pforest *pforest;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  pforest = (DM_Forest_pforest *) ((DM_Forest *) dm->data)->data;
  if (!pforest->plex) {
    ierr = DMConvert_pforest_plex(dm,DMPLEX,NULL);CHKERRQ(ierr);
  }
  ierr = DMShareDiscretization(dm,pforest->plex);CHKERRQ(ierr);
  *plex = pforest->plex;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMCreateCoordinateDM_pforest)
static PetscErrorCode DMCreateCoordinateDM_pforest(DM dm,DM *cdm)
{
  DM                 plex;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = DMPforestGetPlex(dm,&plex);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(plex,cdm);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)*cdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMCreateGlobalVector_pforest)
static PetscErrorCode DMCreateGlobalVector_pforest(DM dm,Vec *vec)
{
  DM                plex;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMPforestGetPlex(dm,&plex);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(plex,vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if 0
#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMCreateLocalVector_pforest)
static PetscErrorCode DMCreateLocalVector_pforest(DM dm,Vec *vec)
{
  DM                plex;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMPforestGetPlex(dm,&plex);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(plex,vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMCreateMatrix_pforest)
static PetscErrorCode DMCreateMatrix_pforest(DM dm,Mat *mat)
{
  DM                plex;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMPforestGetPlex(dm,&plex);CHKERRQ(ierr);
  ierr = DMCreateMatrix(plex,mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMProjectFunctionLocal_pforest)
static PetscErrorCode DMProjectFunctionLocal_pforest(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, InsertMode mode, Vec localX)
{
  DM                plex;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMPforestGetPlex(dm,&plex);CHKERRQ(ierr);
  ierr = DMProjectFunctionLocal(plex,time,funcs,ctxs,mode,localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMProjectFunctionLabelLocal_pforest)
static PetscErrorCode DMProjectFunctionLabelLocal_pforest(DM dm, PetscReal time, DMLabel label, PetscInt numIds, const PetscInt ids[], PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, InsertMode mode, Vec localX)
{
  DM                plex;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMPforestGetPlex(dm,&plex);CHKERRQ(ierr);
  ierr = DMProjectFunctionLabelLocal(plex,time,label,numIds,ids,funcs,ctxs,mode,localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMComputeL2Diff_pforest)
PetscErrorCode DMComputeL2Diff_pforest(DM dm, PetscReal time, PetscErrorCode (**funcs)(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *), void **ctxs, Vec X, PetscReal *diff)
{
  DM                plex;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMPforestGetPlex(dm,&plex);CHKERRQ(ierr);
  ierr = DMComputeL2Diff(plex,time,funcs,ctxs,X,diff);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMCreateDefaultSection_pforest)
static PetscErrorCode DMCreateDefaultSection_pforest(DM dm)
{
  DM                plex;
  PetscSection      section;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMPforestGetPlex(dm,&plex);CHKERRQ(ierr);
  ierr = DMGetDefaultSection(plex,&section);CHKERRQ(ierr);
  ierr = DMSetDefaultSection(dm,section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMCreateDefaultConstraints_pforest)
static PetscErrorCode DMCreateDefaultConstraints_pforest(DM dm)
{
  DM                plex;
  Mat               mat;
  PetscSection      section;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMPforestGetPlex(dm,&plex);CHKERRQ(ierr);
  ierr = DMGetDefaultConstraints(plex,&section,&mat);CHKERRQ(ierr);
  ierr = DMSetDefaultConstraints(dm,section,mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMInitialize_pforest)
static PetscErrorCode DMInitialize_pforest(DM dm)
{
  PetscFunctionBegin;
  dm->ops->setup                     = DMSetUp_pforest;
  dm->ops->view                      = DMView_pforest;
  dm->ops->setfromoptions            = DMSetFromOptions_pforest;
  dm->ops->createcoordinatedm        = DMCreateCoordinateDM_pforest;
  dm->ops->createglobalvector        = DMCreateGlobalVector_pforest;
  dm->ops->createlocalvector         = DMCreateLocalVector_Section_Private;
  dm->ops->creatematrix              = DMCreateMatrix_pforest;
  dm->ops->projectfunctionlocal      = DMProjectFunctionLocal_pforest;
  dm->ops->projectfunctionlabellocal = DMProjectFunctionLabelLocal_pforest;
  dm->ops->createdefaultsection      = DMCreateDefaultSection_pforest;
  dm->ops->createdefaultconstraints  = DMCreateDefaultConstraints_pforest;
  dm->ops->computel2diff             = DMComputeL2Diff_pforest;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMCreate_pforest)
PETSC_EXTERN PetscErrorCode DMCreate_pforest(DM dm)
{
  DM_Forest         *forest;
  DM_Forest_pforest *pforest;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscP4estInitialize();CHKERRQ(ierr);
  ierr = DMCreate_Forest(dm);CHKERRQ(ierr);
  ierr = DMInitialize_pforest(dm);CHKERRQ(ierr);
  ierr = DMSetDimension(dm,P4EST_DIM);CHKERRQ(ierr);

  /* set forest defaults */
  ierr = DMForestSetTopology(dm,"unit");CHKERRQ(ierr);
  ierr = DMForestSetMinimumRefinement(dm,0);CHKERRQ(ierr);
  ierr = DMForestSetInitialRefinement(dm,0);CHKERRQ(ierr);
  ierr = DMForestSetMaximumRefinement(dm,P4EST_QMAXLEVEL);CHKERRQ(ierr);
  ierr = DMForestSetGradeFactor(dm,2);CHKERRQ(ierr);
  ierr = DMForestSetAdjacencyDimension(dm,0);CHKERRQ(ierr);
  ierr = DMForestSetPartitionOverlap(dm,0);CHKERRQ(ierr);

  /* create p4est data */
  ierr = PetscNewLog(dm,&pforest);CHKERRQ(ierr);

  forest                            = (DM_Forest *) dm->data;
  forest->data                      = pforest;
  forest->destroy                   = DMForestDestroy_pforest;
  forest->ftemplate                 = DMForestTemplate_pforest;
  pforest->topo                     = NULL;
  pforest->forest                   = NULL;
  pforest->ghost                    = NULL;
  pforest->lnodes                   = NULL;
  pforest->partition_for_coarsening = PETSC_TRUE;
  pforest->coarsen_hierarchy        = PETSC_FALSE;
  pforest->cLocalStart              = -1;
  pforest->cLocalEnd                = -1;
  pforest->labelsFinalized          = PETSC_FALSE;

  ierr = PetscObjectComposeFunction((PetscObject)dm,_pforest_string(DMConvert_plex_pforest) "_C",DMConvert_plex_pforest);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)dm,_pforest_string(DMConvert_pforest_plex) "_C",DMConvert_pforest_plex);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#endif /* defined(PETSC_HAVE_P4EST) */
