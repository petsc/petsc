#include <petsc-private/dmforestimpl.h>

#if defined(PETSC_HAVE_P4EST)
#define _pforest_string(a) #a

#if !defined(P4_TO_P8)
#include <p4est.h>
#include <p4est_extended.h>
#include <p4est_geometry.h>
#include <p4est_ghost.h>
#include <p4est_lnodes.h>
#else
#include <p8est.h>
#include <p8est_extended.h>
#include <p8est_geometry.h>
#include <p8est_ghost.h>
#include <p8est_lnodes.h>
#endif

#define DMInitialize_pforest            _append_pforest(DMInitialize)
#define DMCreate_pforest                _append_pforest(DMCreate)
#define DMDestroy_pforest               _append_pforest(DMDestroy)
#define DMSetUp_pforest                 _append_pforest(DMSetUp)
#define DM_Forest_pforest               _append_pforest(DM_Forest)
#define DMFTopology_pforest             _append_pforest(DMFTopology)
#define DMFTopologyDestroy_pforest      _append_pforest(DMFTopologyDestroy)
#define DMFTopologyCreate_pforest       _append_pforest(DMFTopologyCreate)
#define DMFTopologyCreateBrick_pforest  _append_pforest(DMFTopologyCreateBrick)

typedef struct {
  PetscInt             refct;
  p4est_connectivity_t *conn;
  p4est_geometry_t     *geom;
} DMFTopology_pforest;

typedef struct {
  DMFTopology_pforest *topo;
  p4est_t             *forest;
  p4est_ghost_t       *ghost;
  p4est_lnodes_t      *lnodes;
} DM_Forest_pforest;

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMFTopologyDestroy_pforest)
static PetscErrorCode DMFTopologyDestroy_pforest(DMFTopology_pforest **topo)
{
  PetscFunctionBegin;
  if (!(*topo)) {
    PetscFunctionReturn(0);
  }
  if (--((*topo)->refct) > 0) {
    *topo = NULL;
    PetscFunctionReturn(0);
  }
  p4est_geometry_destroy((*topo)->geom);
  p4est_connectivity_destroy((*topo)->conn);
  *topo = NULL;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMFTopologyCreateBrick_pforest)
static PetscErrorCode DMFTopologyCreateBrick_pforest(DM dm,PetscInt N[], PetscInt P[], DMFTopology_pforest **topo, PetscBool useMorton)
{

  PetscFunctionBegin;
  if (!useMorton) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Lexicographic ordering not implemented yet");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMFTopologyCreate_pforest)
static PetscErrorCode DMFTopologyCreate_pforest(DM dm, DMForestTopology topologyName, DMFTopology_pforest **topo)
{
  DM_Forest  *forest = (DM_Forest *) dm->data;
  const char *name   = (const char *) topologyName;
  PetscBool  isBrick, isSphere, isShell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidCharPointer(name,2);
  PetscValidPointer(topo,3);
  ierr = PetscNewLog(dm,topo);CHKERRQ(ierr);

  (*topo)->refct = 1;
  ierr = PetscStrcmp(name,"brick",&isBrick);CHKERRQ(ierr);
  if (isBrick && forest->setFromOptions) {
    const char *prefix;
    PetscBool  flgN, flgP, flgM, useMorton = PETSC_TRUE;
    PetscInt   N[P4EST_DIM] = {2}, P[P4EST_DIM] = {0}, nretN = P4EST_DIM, nretP = P4EST_DIM, i;

    ierr = PetscObjectGetOptionsPrefix((PetscObject)dm,&prefix);CHKERRQ(ierr);
    ierr = PetscOptionsGetIntArray(prefix,"-brick_size",N,&nretN,&flgN);CHKERRQ(ierr);
    ierr = PetscOptionsGetIntArray(prefix,"-brick_periodicity",P,&nretP,&flgP);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(prefix,"-brick_use_morton_curve",&useMorton,&flgM);CHKERRQ(ierr);
    if (flgN && nretN != P4EST_DIM) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_SIZ,"Need to give %d sizes in -brick_size, gave %d",P4EST_DIM,nretN);
    if (flgP && nretP != P4EST_DIM) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_SIZ,"Need to give %d periodicities in -brick_size, gave %d",P4EST_DIM,nretP);
    for (i = 0; i < P4EST_DIM; i++) {
      P[i] = (P[i] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE);
    }
    ierr = DMFTopologyCreateBrick_pforest(dm,N,P,topo,PETSC_FALSE);CHKERRQ(ierr);
  }
  else {
    (*topo)->conn  = p4est_connectivity_new_byname(name);
    (*topo)->geom  = p4est_geometry_new_connectivity((*topo)->conn);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMDestroy_pforest)
static PetscErrorCode DMForestDestroy_pforest(DM dm)
{
  DM_Forest         *forest  = (DM_Forest *) dm->data;
  DM_Forest_pforest *pforest = (DM_Forest_pforest *) forest->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (pforest->lnodes) p4est_lnodes_destroy(pforest->lnodes);
  pforest->lnodes = NULL;
  if (pforest->ghost)  p4est_ghost_destroy(pforest->ghost);
  pforest->ghost = NULL;
  if (pforest->forest) p4est_destroy(pforest->forest);
  pforest->forest = NULL;
  ierr = DMFTopologyDestroy_pforest(&pforest->topo);CHKERRQ(ierr);
  ierr = PetscFree(forest->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMSetUp_pforest)
static PetscErrorCode DMSetUp_pforest(DM dm)
{
  DM_Forest         *forest  = (DM_Forest *) dm->data;
  DM_Forest_pforest *pforest = (DM_Forest_pforest *) forest->data;
  DM                base, coarse, fine, adaptFrom;
  DMForestTopology  topology;

  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* sanity check */
  ierr = DMForestGetCoarseForest(dm,&coarse);CHKERRQ(ierr);
  ierr = DMForestGetFineForest(dm,&fine);CHKERRQ(ierr);
  ierr = DMForestGetBaseDM(dm,&base);CHKERRQ(ierr);
  ierr = DMForestGetTopology(dm,&topology);CHKERRQ(ierr);
  if (coarse && fine)            SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot adapt from both a coarse and a fine forest");
  adaptFrom = coarse ? coarse : fine;
  if (adaptFrom && !base) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Did not get base DM from either coarse or fine");
  if (adaptFrom && !topology) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Did not get topology from either coarse or fine");
  if (!base && !topology)        SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"A forest needs either a topology or a base DM");

  /* we need a DMFTopology */
  if (adaptFrom) { /* reference already created */
    DM_Forest         *aforest  = (DM_Forest *) adaptFrom->data;
    DM_Forest_pforest *apforest = (DM_Forest_pforest *) aforest->data;

    if (!apforest->topo) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"The pre-adaptation forest must have a topology");
    apforest->topo->refct++;
    pforest->topo = apforest->topo;
  }
  else if (!base && topology) { /* use a name constructor */
    DMFTopology_pforest *topo;

    ierr = DMFTopologyCreate_pforest(dm,topology,&topo);CHKERRQ(ierr);
    pforest->topo = topo;
    /* TODO: construct base? */
  }
  else if (base && !topology) { /* construct */
    PetscBool isPlex, isDA;
    const char *name;

    SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Converting a DM to a" P4EST_STRING "_connectivity_t needs to be implemented");
    ierr = PetscObjectGetName((PetscObject)base,&name);CHKERRQ(ierr);
    ierr = DMForestSetTopology(dm,(DMForestTopology)name);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)base,DMPLEX,&isPlex);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompare((PetscObject)base,DMDA,&isDA);CHKERRQ(ierr);
    if (isPlex) {
#if 0
      DM redundantBase;

      DMPlexGetRedundantDM(base,&redundantBase);CHKERRQ(ierr);
      if (redundantBase) {
        DMForestSetBaseDM(dm,redundantBase);CHKERRQ(ierr);
        base = redundantBase;
        ierr = DMDestroy(&redundantBase);CHKERRQ(ierr);
      }
      ierr = DMPlexConvert_DMFTopology_pforest(base,&pforest->topo);CHKERRQ(ierr);
#endif
    }
    else if (isDA) {
#if 0
      PetscInt N[3], P[3];

      /* get the sizes, periodicities */
      /* ... */
                                                                  /* don't use Morton order */
      ierr = DMFTopologyCreateBrick_pforest(dm,N,P,&pforest->topo,PETSC_FALSE);CHKERRQ(ierr);
#endif
    }
  }
  else {
  }
  if (!topology) {
    /* try to figure out the topology */
  }
  if (!topology) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Could not determine forest topology");
  ierr = DMForestSetTopology(dm,topology);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ _pforest_string(DMInitialize_pforest)
static PetscErrorCode DMInitialize_pforest(DM dm)
{
  PetscFunctionBegin;
  dm->ops->setup = DMSetUp_pforest;
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
  ierr = DMCreate_Forest(dm);CHKERRQ(ierr);
  ierr = DMInitialize_pforest(dm);CHKERRQ(ierr);
  ierr = DMSetDimension(dm,P4EST_DIM);CHKERRQ(ierr);
  ierr = DMForestSetTopology(dm,"unit");CHKERRQ(ierr);
  ierr = PetscNewLog(dm,&pforest);CHKERRQ(ierr);

  forest          = (DM_Forest *) dm->data;
  forest->data    = pforest;
  forest->destroy = DMForestDestroy_pforest;
  pforest->topo   = NULL;
  pforest->forest = NULL;
  pforest->ghost  = NULL;
  pforest->lnodes = NULL;
  PetscFunctionReturn(0);
}

#endif /* defined(PETSC_HAVE_P4EST) */
