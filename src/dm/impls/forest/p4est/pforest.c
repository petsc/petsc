#include <petsc-private/dmforestimpl.h>
#include <petsc-private/viewerimpl.h>
#include <../src/sys/classes/viewer/impls/vtk/vtkvimpl.h>

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
#else
#include <p8est.h>
#include <p8est_extended.h>
#include <p8est_geometry.h>
#include <p8est_ghost.h>
#include <p8est_lnodes.h>
#include <p8est_vtk.h>
#endif

#define DMInitialize_pforest            _append_pforest(DMInitialize)
#define DMCreate_pforest                _append_pforest(DMCreate)
#define DMForestDestroy_pforest         _append_pforest(DMForestDestroy)
#define DMSetUp_pforest                 _append_pforest(DMSetUp)
#define DMView_pforest                  _append_pforest(DMView)
#define DMView_VTK_pforest              _append_pforest(DMView_VTK)
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
  PetscErrorCode ierr;

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
  ierr = PetscFree(*topo);CHKERRQ(ierr);
  *topo = NULL;
  PetscFunctionReturn(0);
}

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
  (*topo)->conn  = p4est_connectivity_new_brick((int) N[0], (int) N[1], (P[0] == DM_BOUNDARY_NONE) ? 0 : 1, (P[1] == DM_BOUNDARY_NONE) ? 0 : 1);
#else
  (*topo)->conn  = p8est_connectivity_new_brick((int) N[0], (int) N[1], (int) N[2], (P[0] == DM_BOUNDARY_NONE) ? 0 : 1, (P[1] == DM_BOUNDARY_NONE) ? 0 : 1, (P[2] == DM_BOUNDARY_NONE) ? 0 : 1);
#endif
  (*topo)->geom  = p4est_geometry_new_connectivity((*topo)->conn);
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
      ierr = PetscOptionsGetIntArray(prefix,"-dm_p4est_brick_size",N,&nretN,&flgN);CHKERRQ(ierr);
      ierr = PetscOptionsGetIntArray(prefix,"-dm_p4est_brick_periodicity",P,&nretP,&flgP);CHKERRQ(ierr);
      ierr = PetscOptionsGetBool(prefix,"-dm_p4est_brick_use_morton_curve",&useMorton,&flgM);CHKERRQ(ierr);
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
    (*topo)->conn  = p4est_connectivity_new_byname(name);
#if !defined(P4_TO_P8)
    (*topo)->geom  = p4est_geometry_new_connectivity((*topo)->conn);
#else
    if (isShell) {
      PetscReal R2 = 1., R1 = .55;

      if (forest->setFromOptions) {
        ierr = PetscOptionsGetReal(prefix,"-dm_p4est_shell_outer_radius",&R2,NULL);CHKERRQ(ierr);
        ierr = PetscOptionsGetReal(prefix,"-dm_p4est_shell_inner_radius",&R1,NULL);CHKERRQ(ierr);
      }
      (*topo)->geom = p8est_geometry_new_shell((*topo)->conn,R2,R1);
    }
    else if (isSphere) {
      PetscReal R2 = 1., R1 = 0.191728, R0 = 0.039856;

      if (forest->setFromOptions) {
        ierr = PetscOptionsGetReal(prefix,"-dm_p4est_sphere_outer_radius",&R2,NULL);CHKERRQ(ierr);
        ierr = PetscOptionsGetReal(prefix,"-dm_p4est_sphere_inner_radius",&R1,NULL);CHKERRQ(ierr);
        ierr = PetscOptionsGetReal(prefix,"-dm_p4est_sphere_core_radius",&R0,NULL);CHKERRQ(ierr);
      }
      (*topo)->geom = p8est_geometry_new_sphere((*topo)->conn,R2,R1,R0);
    }
    else {
      (*topo)->geom  = p4est_geometry_new_connectivity((*topo)->conn);
    }
#endif
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
  DMForestTopology  topoName;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  /* sanity check */
  ierr = DMForestGetCoarseForest(dm,&coarse);CHKERRQ(ierr);
  ierr = DMForestGetFineForest(dm,&fine);CHKERRQ(ierr);
  ierr = DMForestGetBaseDM(dm,&base);CHKERRQ(ierr);
  ierr = DMForestGetTopology(dm,&topoName);CHKERRQ(ierr);
  if (coarse && fine)            SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"Cannot adapt from both a coarse and a fine forest");
  adaptFrom = coarse ? coarse : fine;
  if (!base && !topoName)        SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"A forest needs either a topology or a base DM");

  /* === Step 1: DMFTopology === */
  if (adaptFrom) { /* reference already created topology */
    DM_Forest         *aforest  = (DM_Forest *) adaptFrom->data;
    DM_Forest_pforest *apforest = (DM_Forest_pforest *) aforest->data;

    if (!apforest->topo) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"The pre-adaptation forest must have a topology");
    apforest->topo->refct++;
    pforest->topo = apforest->topo;
  }
  else if (topoName) { /* use a name constructor */
    DMFTopology_pforest *topo;

    ierr = DMFTopologyCreate_pforest(dm,topoName,&topo);CHKERRQ(ierr);
    pforest->topo = topo;
    /* TODO: construct base? */
  }
  else { /* construct from base */
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

  /* === Step 2: get the leaves of the forest === */
  if (adaptFrom) { /* start with the old forest */
    DM_Forest         *aforest  = (DM_Forest *) adaptFrom->data;
    DM_Forest_pforest *apforest = (DM_Forest_pforest *) aforest->data;

    pforest->forest = p4est_copy(apforest->forest, 0); /* 0 indicates no data copying */
    /* apply the refinement/coarsening by flags, plus minimum/maximum refinement */
    /* ... */
    p4est_reset_data(pforest->forest,0,NULL,(void *)dm); /* this dm is the user context for the new forest */
  }
  else {
    PetscInt minLevel;

    ierr = DMForestGetMinimumRefinement(dm,&minLevel);CHKERRQ(ierr);
    pforest->forest = p4est_new_ext(PetscObjectComm((PetscObject)dm),pforest->topo->conn,
                                    0,           /* minimum number of quadrants per processor */
                                    minLevel,    /* minimum level of refinement */
                                    1,           /* uniform refinement */
                                    0,           /* we don't allocate any per quadrant data */
                                    NULL,        /* there is no special quadrant initialization */
                                    (void *)dm); /* this dm is the user context */
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
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK, &isvtk);CHKERRQ(ierr);
  if (!isvtk) SETERRQ1(PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_INCOMP, "Cannot use viewer type %s", ((PetscObject)viewer)->type_name);
  switch (viewer->format) {
  case PETSC_VIEWER_VTK_VTU:
    if (!dm->setupcalled || !pforest->forest) SETERRQ (PetscObjectComm(odm),PETSC_ERR_ARG_WRONG,"DM has not been setup with a valid forest");
    name = vtk->filename;
    ierr = PetscStrlen(name,&len);CHKERRQ(ierr);
    ierr = PetscStrcasecmp(name+len-4,".vtu",&hasExt);CHKERRQ(ierr);
    if (hasExt) {
      ierr = PetscStrallocpy(name,&filenameStrip);CHKERRQ(ierr);
      filenameStrip[len-4]='\0';
      name = filenameStrip;
    }
    p4est_vtk_write_all(pforest->forest,pforest->topo->geom,(double)vtkScale,
                        1, /* write tree */
                        1, /* write level */
                        1, /* write rank */
                        0, /* do not wrap rank */
                        0, /* no scalar fields */
                        0, /* no vector fields */
                        name);
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
#define __FUNCT__ _pforest_string(DMInitialize_pforest)
static PetscErrorCode DMInitialize_pforest(DM dm)
{
  PetscFunctionBegin;
  dm->ops->setup = DMSetUp_pforest;
  dm->ops->view  = DMView_pforest;
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

  /* TODO: interface libsc signals and logging with petsc.  For now, just silencing all libsc/p4est  */
  sc_set_log_defaults(NULL,NULL,SC_LP_SILENT);

  /* set forest defaults */
  ierr = DMForestSetTopology(dm,"unit");CHKERRQ(ierr);
  ierr = DMForestSetMinimumRefinement(dm,0);CHKERRQ(ierr);
  ierr = DMForestSetMaximumRefinement(dm,P4EST_QMAXLEVEL);CHKERRQ(ierr);
  ierr = DMForestSetGradeFactor(dm,2);CHKERRQ(ierr);
  ierr = DMForestSetAdjacencyDimension(dm,0);CHKERRQ(ierr);
  ierr = DMForestSetPartitionOverlap(dm,0);CHKERRQ(ierr);

  /* create p4est data */
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
