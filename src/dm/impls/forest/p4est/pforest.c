#include <petscds.h>
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

typedef enum {PATTERN_HASH,PATTERN_FRACTAL,PATTERN_CORNER,PATTERN_CENTER,PATTERN_COUNT} DMRefinePattern;
static const char *DMRefinePatternName[PATTERN_COUNT] = {"hash","fractal","corner","center"};

typedef struct _DMRefinePatternCtx
{
  PetscInt       corner;
  PetscBool      fractal[P4EST_CHILDREN];
  PetscReal      hashLikelihood;
  PetscInt       maxLevel;
  p4est_refine_t refine_fn;
}
DMRefinePatternCtx;

static int DMRefinePattern_Corner(p4est_t * p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quadrant)
{
  p4est_quadrant_t   root, rootcorner;
  DMRefinePatternCtx *ctx;

  ctx = (DMRefinePatternCtx*) p4est->user_pointer;
  if (quadrant->level >= ctx->maxLevel) return 0;

  root.x = root.y = 0;
#if defined(P4_TO_P8)
  root.z = 0;
#endif
  root.level = 0;
  p4est_quadrant_corner_descendant(&root,&rootcorner,ctx->corner,quadrant->level);
  if (p4est_quadrant_is_equal(quadrant,&rootcorner)) return 1;
  return 0;
}

static int DMRefinePattern_Center(p4est_t * p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quadrant)
{
  int                cid;
  p4est_quadrant_t   ancestor, ancestorcorner;
  DMRefinePatternCtx *ctx;

  ctx = (DMRefinePatternCtx*) p4est->user_pointer;
  if (quadrant->level >= ctx->maxLevel) return 0;
  if (quadrant->level <= 1) return 1;

  p4est_quadrant_ancestor(quadrant,1,&ancestor);
  cid = p4est_quadrant_child_id(&ancestor);
  p4est_quadrant_corner_descendant(&ancestor,&ancestorcorner,P4EST_CHILDREN - 1 - cid,quadrant->level);
  if (p4est_quadrant_is_equal(quadrant,&ancestorcorner)) return 1;
  return 0;
}

static int DMRefinePattern_Fractal(p4est_t * p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quadrant)
{
  int                cid;
  DMRefinePatternCtx *ctx;

  ctx = (DMRefinePatternCtx*) p4est->user_pointer;
  if (quadrant->level >= ctx->maxLevel) return 0;
  if (!quadrant->level) return 1;
  cid = p4est_quadrant_child_id(quadrant);
  if (ctx->fractal[cid ^ ((int) (quadrant->level % P4EST_CHILDREN))]) return 1;
  return 0;
}

/* simplified from MurmurHash3 by Austin Appleby */
#define DMPROT32(x, y) ((x << y) | (x >> (32 - y)))
static uint32_t DMPforestHash(const uint32_t *blocks, uint32_t nblocks)
{
  uint32_t c1   = 0xcc9e2d51;
  uint32_t c2   = 0x1b873593;
  uint32_t r1   = 15;
  uint32_t r2   = 13;
  uint32_t m    = 5;
  uint32_t n    = 0xe6546b64;
  uint32_t hash = 0;
  int      len  = nblocks * 4;
  uint32_t i;

  for (i = 0; i < nblocks; i++) {
    uint32_t k;

    k  = blocks[i];
    k *= c1;
    k  = DMPROT32(k, r1);
    k *= c2;

    hash ^= k;
    hash  = DMPROT32(hash, r2) * m + n;
  }

  hash ^= len;
  hash ^= (hash >> 16);
  hash *= 0x85ebca6b;
  hash ^= (hash >> 13);
  hash *= 0xc2b2ae35;
  hash ^= (hash >> 16);

  return hash;
}

#if defined(UINT32_MAX)
#define DMP4EST_HASH_MAX UINT32_MAX
#else
#define DMP4EST_HASH_MAX ((uint32_t) 0xffffffff)
#endif

static int DMRefinePattern_Hash(p4est_t * p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quadrant)
{
  uint32_t           data[5];
  uint32_t           result;
  DMRefinePatternCtx *ctx;

  ctx = (DMRefinePatternCtx*) p4est->user_pointer;
  if (quadrant->level >= ctx->maxLevel) return 0;
  data[0] = ((uint32_t) quadrant->level) << 24;
  data[1] = (uint32_t) which_tree;
  data[2] = (uint32_t) quadrant->x;
  data[3] = (uint32_t) quadrant->y;
#if defined(P4_TO_P8)
  data[4] = (uint32_t) quadrant->z;
#endif

  result = DMPforestHash(data,2+P4EST_DIM);
  if (((double) result / (double) DMP4EST_HASH_MAX) < ctx->hashLikelihood) return 1;
  return 0;
}

#define DMConvert_pforest_plex _infix_pforest(DMConvert,_plex)
static PetscErrorCode DMConvert_pforest_plex(DM,DMType,DM*);

#define DMFTopology_pforest _append_pforest(DMFTopology)
typedef struct {
  PetscInt             refct;
  p4est_connectivity_t *conn;
  p4est_geometry_t     *geom;
  PetscInt             *tree_face_to_uniq; /* p4est does not explicitly enumerate facets, but we must to keep track of labels */
} DMFTopology_pforest;

#define DM_Forest_pforest _append_pforest(DM_Forest)
typedef struct {
  DMFTopology_pforest *topo;
  p4est_t             *forest;
  p4est_ghost_t       *ghost;
  p4est_lnodes_t      *lnodes;
  PetscBool           partition_for_coarsening;
  PetscBool           coarsen_hierarchy;
  PetscBool           labelsFinalized;
  PetscBool           adaptivitySuccess;
  PetscInt            cLocalStart;
  PetscInt            cLocalEnd;
  DM                  plex;
  char                *ghostName;
  PetscSF             pointAdaptToSelfSF;
  PetscSF             pointSelfToAdaptSF;
  PetscInt            *pointAdaptToSelfCids;
  PetscInt            *pointSelfToAdaptCids;
} DM_Forest_pforest;

#define DM_Forest_geometry_pforest _append_pforest(DM_Forest_geometry)
typedef struct {
  DM base;
  PetscErrorCode   (*map)(DM, PetscInt, PetscInt, const PetscReal[], PetscReal[], void*);
  void             *mapCtx;
  PetscInt         coordDim;
  p4est_geometry_t *inner;
}
DM_Forest_geometry_pforest;

#define GeometryMapping_pforest _append_pforest(GeometryMapping)
static void GeometryMapping_pforest(p4est_geometry_t *geom, p4est_topidx_t which_tree, const double abc[3], double xyz[3])
{
  DM_Forest_geometry_pforest *geom_pforest = (DM_Forest_geometry_pforest*)geom->user;
  PetscReal                  PetscABC[3]   = {0.};
  PetscReal                  PetscXYZ[3]   = {0.};
  PetscInt                   i, d = PetscMin(3,geom_pforest->coordDim);
  double                     ABC[3];
  PetscErrorCode             ierr;

  (geom_pforest->inner->X)(geom_pforest->inner,which_tree,abc,ABC);

  for (i = 0; i < d; i++) PetscABC[i] = ABC[i];
  ierr = (geom_pforest->map)(geom_pforest->base,(PetscInt) which_tree,geom_pforest->coordDim,PetscABC,PetscXYZ,geom_pforest->mapCtx);PETSC_P4EST_ASSERT(!ierr);
  for (i = 0; i < d; i++) xyz[i] = PetscXYZ[i];
}

#define GeometryDestroy_pforest _append_pforest(GeometryDestroy)
static void GeometryDestroy_pforest(p4est_geometry_t *geom)
{
  DM_Forest_geometry_pforest *geom_pforest = (DM_Forest_geometry_pforest*)geom->user;
  PetscErrorCode             ierr;

  p4est_geometry_destroy(geom_pforest->inner);
  ierr = PetscFree(geom->user);PETSC_P4EST_ASSERT(!ierr);
  ierr = PetscFree(geom);PETSC_P4EST_ASSERT(!ierr);
}

#define DMFTopologyDestroy_pforest _append_pforest(DMFTopologyDestroy)
static PetscErrorCode DMFTopologyDestroy_pforest(DMFTopology_pforest **topo)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!(*topo)) PetscFunctionReturn(0);
  if (--((*topo)->refct) > 0) {
    *topo = NULL;
    PetscFunctionReturn(0);
  }
  if ((*topo)->geom) PetscStackCallP4est(p4est_geometry_destroy,((*topo)->geom));
  PetscStackCallP4est(p4est_connectivity_destroy,((*topo)->conn));
  ierr  = PetscFree((*topo)->tree_face_to_uniq);CHKERRQ(ierr);
  ierr  = PetscFree(*topo);CHKERRQ(ierr);
  *topo = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode PforestConnectivityEnumerateFacets(p4est_connectivity_t*,PetscInt**);

#define DMFTopologyCreateBrick_pforest _append_pforest(DMFTopologyCreateBrick)
static PetscErrorCode DMFTopologyCreateBrick_pforest(DM dm,PetscInt N[], PetscInt P[], PetscReal B[],DMFTopology_pforest **topo, PetscBool useMorton)
{
  double         *vertices;
  PetscInt       i, numVerts;
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
  numVerts = (*topo)->conn->num_vertices;
  vertices = (*topo)->conn->vertices;
  for (i = 0; i < 3 * numVerts; i++) {
    PetscInt j = i % 3;

    vertices[i] = B[2 * j] + (vertices[i]/N[j]) * (B[2 * j + 1] - B[2 * j]);
  }
  (*topo)->geom = NULL;
  ierr          = PforestConnectivityEnumerateFacets((*topo)->conn,&(*topo)->tree_face_to_uniq);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define DMFTopologyCreate_pforest _append_pforest(DMFTopologyCreate)
static PetscErrorCode DMFTopologyCreate_pforest(DM dm, DMForestTopology topologyName, DMFTopology_pforest **topo)
{
  const char     *name = (const char*) topologyName;
  const char     *prefix;
  PetscBool      isBrick, isShell, isSphere, isMoebius;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidCharPointer(name,2);
  PetscValidPointer(topo,3);
  ierr = PetscStrcmp(name,"brick",&isBrick);CHKERRQ(ierr);
  ierr = PetscStrcmp(name,"shell",&isShell);CHKERRQ(ierr);
  ierr = PetscStrcmp(name,"sphere",&isSphere);CHKERRQ(ierr);
  ierr = PetscStrcmp(name,"moebius",&isMoebius);CHKERRQ(ierr);
  ierr = PetscObjectGetOptionsPrefix((PetscObject)dm,&prefix);CHKERRQ(ierr);
  if (isBrick) {
    PetscBool flgN, flgP, flgM, flgB, useMorton = PETSC_TRUE, periodic = PETSC_FALSE;
    PetscInt  N[3] = {2,2,2}, P[3] = {0,0,0}, nretN = P4EST_DIM, nretP = P4EST_DIM, nretB = 2 * P4EST_DIM, i;
    PetscReal B[6] = {0.0,1.0,0.0,1.0,0.0,1.0};

    if (dm->setfromoptionscalled) {
      ierr = PetscOptionsGetIntArray(((PetscObject)dm)->options,prefix,"-dm_p4est_brick_size",N,&nretN,&flgN);CHKERRQ(ierr);
      ierr = PetscOptionsGetIntArray(((PetscObject)dm)->options,prefix,"-dm_p4est_brick_periodicity",P,&nretP,&flgP);CHKERRQ(ierr);
      ierr = PetscOptionsGetRealArray(((PetscObject)dm)->options,prefix,"-dm_p4est_brick_bounds",B,&nretB,&flgB);CHKERRQ(ierr);
      ierr = PetscOptionsGetBool(((PetscObject)dm)->options,prefix,"-dm_p4est_brick_use_morton_curve",&useMorton,&flgM);CHKERRQ(ierr);
      if (flgN && nretN != P4EST_DIM) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_SIZ,"Need to give %d sizes in -dm_p4est_brick_size, gave %d",P4EST_DIM,nretN);
      if (flgP && nretP != P4EST_DIM) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_SIZ,"Need to give %d periodicities in -dm_p4est_brick_periodicity, gave %d",P4EST_DIM,nretP);
      if (flgB && nretB != 2 * P4EST_DIM) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_SIZ,"Need to give %d bounds in -dm_p4est_brick_bounds, gave %d",P4EST_DIM,nretP);
    }
    for (i = 0; i < P4EST_DIM; i++) {
      P[i]  = (P[i] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE);
      periodic = (PetscBool)(P[i] || periodic);
      if (!flgB) B[2 * i + 1] = N[i];
    }
    ierr = DMFTopologyCreateBrick_pforest(dm,N,P,B,topo,useMorton);CHKERRQ(ierr);
    /* the maxCell trick is not robust enough, localize on all cells if periodic */
    ierr = DMSetPeriodicity(dm,periodic,NULL,NULL,NULL);CHKERRQ(ierr);
  } else {
    ierr = PetscNewLog(dm,topo);CHKERRQ(ierr);

    (*topo)->refct = 1;
    PetscStackCallP4estReturn((*topo)->conn,p4est_connectivity_new_byname,(name));
    (*topo)->geom = NULL;
    if (isMoebius) {
      ierr = DMSetCoordinateDim(dm,3);CHKERRQ(ierr);
    }
#if defined(P4_TO_P8)
    if (isShell) {
      PetscReal R2 = 1., R1 = .55;

      if (dm->setfromoptionscalled) {
        ierr = PetscOptionsGetReal(((PetscObject)dm)->options,prefix,"-dm_p4est_shell_outer_radius",&R2,NULL);CHKERRQ(ierr);
        ierr = PetscOptionsGetReal(((PetscObject)dm)->options,prefix,"-dm_p4est_shell_inner_radius",&R1,NULL);CHKERRQ(ierr);
      }
      PetscStackCallP4estReturn((*topo)->geom,p8est_geometry_new_shell,((*topo)->conn,R2,R1));
    } else if (isSphere) {
      PetscReal R2 = 1., R1 = 0.191728, R0 = 0.039856;

      if (dm->setfromoptionscalled) {
        ierr = PetscOptionsGetReal(((PetscObject)dm)->options,prefix,"-dm_p4est_sphere_outer_radius",&R2,NULL);CHKERRQ(ierr);
        ierr = PetscOptionsGetReal(((PetscObject)dm)->options,prefix,"-dm_p4est_sphere_inner_radius",&R1,NULL);CHKERRQ(ierr);
        ierr = PetscOptionsGetReal(((PetscObject)dm)->options,prefix,"-dm_p4est_sphere_core_radius",&R0,NULL);CHKERRQ(ierr);
      }
      PetscStackCallP4estReturn((*topo)->geom,p8est_geometry_new_sphere,((*topo)->conn,R2,R1,R0));
    }
#endif
    ierr = PforestConnectivityEnumerateFacets((*topo)->conn,&(*topo)->tree_face_to_uniq);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#define DMConvert_plex_pforest _append_pforest(DMConvert_plex)
static PetscErrorCode DMConvert_plex_pforest(DM dm, DMType newtype, DM *pforest)
{
  MPI_Comm       comm;
  PetscBool      isPlex;
  PetscInt       dim;
  void           *ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;

  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  comm = PetscObjectComm((PetscObject)dm);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMPLEX,&isPlex);CHKERRQ(ierr);
  if (!isPlex) SETERRQ2(comm,PETSC_ERR_ARG_WRONG,"Expected DM type %s, got %s",DMPLEX,((PetscObject)dm)->type_name);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if (dim != P4EST_DIM) SETERRQ2(comm,PETSC_ERR_ARG_WRONG,"Expected DM dimension %d, got %d",P4EST_DIM,dim);
  ierr = DMCreate(comm,pforest);CHKERRQ(ierr);
  ierr = DMSetType(*pforest,DMPFOREST);CHKERRQ(ierr);
  ierr = DMForestSetBaseDM(*pforest,dm);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(dm,&ctx);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(*pforest,ctx);CHKERRQ(ierr);
  ierr = DMCopyDisc(dm,*pforest);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define DMForestDestroy_pforest _append_pforest(DMForestDestroy)
static PetscErrorCode DMForestDestroy_pforest(DM dm)
{
  DM_Forest         *forest  = (DM_Forest*) dm->data;
  DM_Forest_pforest *pforest = (DM_Forest_pforest*) forest->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (pforest->lnodes) PetscStackCallP4est(p4est_lnodes_destroy,(pforest->lnodes));
  pforest->lnodes = NULL;
  if (pforest->ghost) PetscStackCallP4est(p4est_ghost_destroy,(pforest->ghost));
  pforest->ghost = NULL;
  if (pforest->forest) PetscStackCallP4est(p4est_destroy,(pforest->forest));
  pforest->forest = NULL;
  ierr            = DMFTopologyDestroy_pforest(&pforest->topo);CHKERRQ(ierr);
  ierr            = PetscObjectComposeFunction((PetscObject)dm,_pforest_string(DMConvert_plex_pforest) "_C",NULL);CHKERRQ(ierr);
  ierr            = PetscObjectComposeFunction((PetscObject)dm,_pforest_string(DMConvert_pforest_plex) "_C",NULL);CHKERRQ(ierr);
  ierr            = PetscFree(pforest->ghostName);CHKERRQ(ierr);
  ierr            = DMDestroy(&pforest->plex);CHKERRQ(ierr);
  ierr            = PetscSFDestroy(&pforest->pointAdaptToSelfSF);CHKERRQ(ierr);
  ierr            = PetscSFDestroy(&pforest->pointSelfToAdaptSF);CHKERRQ(ierr);
  ierr            = PetscFree(pforest->pointAdaptToSelfCids);CHKERRQ(ierr);
  ierr            = PetscFree(pforest->pointSelfToAdaptCids);CHKERRQ(ierr);
  ierr            = PetscFree(forest->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define DMForestTemplate_pforest _append_pforest(DMForestTemplate)
static PetscErrorCode DMForestTemplate_pforest(DM dm, DM tdm)
{
  DM_Forest_pforest *pforest  = (DM_Forest_pforest*) ((DM_Forest*) dm->data)->data;
  DM_Forest_pforest *tpforest = (DM_Forest_pforest*) ((DM_Forest*) tdm->data)->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (pforest->topo) pforest->topo->refct++;
  ierr           = DMFTopologyDestroy_pforest(&(tpforest->topo));CHKERRQ(ierr);
  tpforest->topo = pforest->topo;
  PetscFunctionReturn(0);
}

#define DMPlexCreateConnectivity_pforest _append_pforest(DMPlexCreateConnectivity)
static PetscErrorCode DMPlexCreateConnectivity_pforest(DM,p4est_connectivity_t**,PetscInt**);

typedef struct _PforestAdaptCtx
{
  PetscInt  maxLevel;
  PetscInt  minLevel;
  PetscInt  currLevel;
  PetscBool anyChange;
}
PforestAdaptCtx;

static int pforest_coarsen_currlevel(p4est_t * p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quadrants[])
{
  PforestAdaptCtx *ctx      = (PforestAdaptCtx*) p4est->user_pointer;
  PetscInt        minLevel  = ctx->minLevel;
  PetscInt        currLevel = ctx->currLevel;

  if (quadrants[0]->level <= minLevel) return 0;
  return (int) ((PetscInt) quadrants[0]->level == currLevel);
}

static int pforest_coarsen_uniform(p4est_t * p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quadrants[])
{
  PforestAdaptCtx *ctx     = (PforestAdaptCtx*) p4est->user_pointer;
  PetscInt        minLevel = ctx->minLevel;

  return (int) ((PetscInt) quadrants[0]->level > minLevel);
}

static int pforest_coarsen_flag_any(p4est_t * p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quadrants[])
{
  PetscInt        i;
  PetscBool       any      = PETSC_FALSE;
  PforestAdaptCtx *ctx     = (PforestAdaptCtx*) p4est->user_pointer;
  PetscInt        minLevel = ctx->minLevel;

  if (quadrants[0]->level <= minLevel) return 0;
  for (i = 0; i < P4EST_CHILDREN; i++) {
    if (quadrants[i]->p.user_int == DM_ADAPT_KEEP) {
      any = PETSC_FALSE;
      break;
    }
    if (quadrants[i]->p.user_int == DM_ADAPT_COARSEN) {
      any = PETSC_TRUE;
      break;
    }
  }
  return any ? 1 : 0;
}

static int pforest_coarsen_flag_all(p4est_t * p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quadrants[])
{
  PetscInt        i;
  PetscBool       all      = PETSC_TRUE;
  PforestAdaptCtx *ctx     = (PforestAdaptCtx*) p4est->user_pointer;
  PetscInt        minLevel = ctx->minLevel;

  if (quadrants[0]->level <= minLevel) return 0;
  for (i = 0; i < P4EST_CHILDREN; i++) {
    if (quadrants[i]->p.user_int != DM_ADAPT_COARSEN) {
      all = PETSC_FALSE;
      break;
    }
  }
  return all ? 1 : 0;
}

static void pforest_init_determine(p4est_t *p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quadrant)
{
  quadrant->p.user_int = DM_ADAPT_DETERMINE;
}

static int pforest_refine_uniform(p4est_t * p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quadrant)
{
  PforestAdaptCtx *ctx     = (PforestAdaptCtx*) p4est->user_pointer;
  PetscInt        maxLevel = ctx->maxLevel;

  return ((PetscInt) quadrant->level < maxLevel);
}

static int pforest_refine_flag(p4est_t * p4est, p4est_topidx_t which_tree, p4est_quadrant_t *quadrant)
{
  PforestAdaptCtx *ctx     = (PforestAdaptCtx*) p4est->user_pointer;
  PetscInt        maxLevel = ctx->maxLevel;

  if ((PetscInt) quadrant->level >= maxLevel) return 0;

  return (quadrant->p.user_int == DM_ADAPT_REFINE);
}

static PetscErrorCode DMPforestComputeLocalCellTransferSF_loop(p4est_t *p4estFrom, PetscInt FromOffset, p4est_t *p4estTo, PetscInt ToOffset, p4est_topidx_t flt, p4est_topidx_t llt, PetscInt *toFineLeavesCount, PetscInt *toLeaves, PetscSFNode *fromRoots, PetscInt *fromFineLeavesCount, PetscInt *fromLeaves, PetscSFNode *toRoots)
{
  PetscMPIInt    rank = p4estFrom->mpirank;
  p4est_topidx_t t;
  PetscInt       toFineLeaves = 0, fromFineLeaves = 0;

  PetscFunctionBegin;
  for (t = flt; t <= llt; t++) { /* count roots and leaves */
    p4est_tree_t     *treeFrom  = &(((p4est_tree_t*) p4estFrom->trees->array)[t]);
    p4est_tree_t     *treeTo    = &(((p4est_tree_t*) p4estTo->trees->array)[t]);
    p4est_quadrant_t *firstFrom = &treeFrom->first_desc;
    p4est_quadrant_t *firstTo   = &treeTo->first_desc;
    PetscInt         numFrom    = (PetscInt) treeFrom->quadrants.elem_count;
    PetscInt         numTo      = (PetscInt) treeTo->quadrants.elem_count;
    p4est_quadrant_t *quadsFrom = (p4est_quadrant_t*) treeFrom->quadrants.array;
    p4est_quadrant_t *quadsTo   = (p4est_quadrant_t*) treeTo->quadrants.array;
    PetscInt         currentFrom, currentTo;
    PetscInt         treeOffsetFrom = (PetscInt) treeFrom->quadrants_offset;
    PetscInt         treeOffsetTo   = (PetscInt) treeTo->quadrants_offset;
    int              comp;

    PetscStackCallP4estReturn(comp,p4est_quadrant_is_equal,(firstFrom,firstTo));
    if (!comp) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"non-matching partitions");

    for (currentFrom = 0, currentTo = 0; currentFrom < numFrom && currentTo < numTo; ) {
      p4est_quadrant_t *quadFrom = &quadsFrom[currentFrom];
      p4est_quadrant_t *quadTo   = &quadsTo[currentTo];

      if (quadFrom->level == quadTo->level) {
        if (toLeaves) {
          toLeaves[toFineLeaves]        = currentTo + treeOffsetTo + ToOffset;
          fromRoots[toFineLeaves].rank  = rank;
          fromRoots[toFineLeaves].index = currentFrom + treeOffsetFrom + FromOffset;
        }
        toFineLeaves++;
        currentFrom++;
        currentTo++;
      } else {
        int fromIsAncestor;

        PetscStackCallP4estReturn(fromIsAncestor,p4est_quadrant_is_ancestor,(quadFrom,quadTo));
        if (fromIsAncestor) {
          p4est_quadrant_t lastDesc;

          if (toLeaves) {
            toLeaves[toFineLeaves]        = currentTo + treeOffsetTo + ToOffset;
            fromRoots[toFineLeaves].rank  = rank;
            fromRoots[toFineLeaves].index = currentFrom + treeOffsetFrom + FromOffset;
          }
          toFineLeaves++;
          currentTo++;
          PetscStackCallP4est(p4est_quadrant_last_descendant,(quadFrom,&lastDesc,quadTo->level));
          PetscStackCallP4estReturn(comp,p4est_quadrant_is_equal,(quadTo,&lastDesc));
          if (comp) currentFrom++;
        } else {
          p4est_quadrant_t lastDesc;

          if (fromLeaves) {
            fromLeaves[fromFineLeaves]    = currentFrom + treeOffsetFrom + FromOffset;
            toRoots[fromFineLeaves].rank  = rank;
            toRoots[fromFineLeaves].index = currentTo + treeOffsetTo + ToOffset;
          }
          fromFineLeaves++;
          currentFrom++;
          PetscStackCallP4est(p4est_quadrant_last_descendant,(quadTo,&lastDesc,quadFrom->level));
          PetscStackCallP4estReturn(comp,p4est_quadrant_is_equal,(quadFrom,&lastDesc));
          if (comp) currentTo++;
        }
      }
    }
  }
  *toFineLeavesCount   = toFineLeaves;
  *fromFineLeavesCount = fromFineLeaves;
  PetscFunctionReturn(0);
}

/* Compute the maximum level across all the trees */
static PetscErrorCode DMPforestGetRefinementLevel(DM dm, PetscInt *lev)
{
  p4est_topidx_t    t, flt, llt;
  DM_Forest         *forest  = (DM_Forest*) dm->data;
  DM_Forest_pforest *pforest = (DM_Forest_pforest*) forest->data;
  PetscInt          maxlevelloc = 0;
  p4est_t           *p4est;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!pforest) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Missing DM_Forest_pforest");
  if (!pforest->forest) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Missing p4est_t");
  p4est = pforest->forest;
  flt   = p4est->first_local_tree;
  llt   = p4est->last_local_tree;
  for (t = flt; t <= llt; t++) {
    p4est_tree_t *tree  = &(((p4est_tree_t*) p4est->trees->array)[t]);
    maxlevelloc = PetscMax((PetscInt)tree->maxlevel,maxlevelloc);
  }
  ierr = MPIU_Allreduce(&maxlevelloc,lev,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Puts identity in coarseToFine */
/* assumes a matching partition */
static PetscErrorCode DMPforestComputeLocalCellTransferSF(MPI_Comm comm, p4est_t *p4estFrom, PetscInt FromOffset, p4est_t *p4estTo, PetscInt ToOffset, PetscSF *fromCoarseToFine, PetscSF *toCoarseFromFine)
{
  p4est_topidx_t flt, llt;
  PetscSF        fromCoarse, toCoarse;
  PetscInt       numRootsFrom, numRootsTo, numLeavesFrom, numLeavesTo;
  PetscInt       *fromLeaves = NULL, *toLeaves = NULL;
  PetscSFNode    *fromRoots  = NULL, *toRoots = NULL;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  flt  = p4estFrom->first_local_tree;
  llt  = p4estFrom->last_local_tree;
  ierr = PetscSFCreate(comm,&fromCoarse);CHKERRQ(ierr);
  if (toCoarseFromFine) {
    ierr = PetscSFCreate(comm,&toCoarse);CHKERRQ(ierr);
  }
  numRootsFrom = p4estFrom->local_num_quadrants + FromOffset;
  numRootsTo   = p4estTo->local_num_quadrants + ToOffset;
  ierr         = DMPforestComputeLocalCellTransferSF_loop(p4estFrom,FromOffset,p4estTo,ToOffset,flt,llt,&numLeavesTo,NULL,NULL,&numLeavesFrom,NULL,NULL);CHKERRQ(ierr);
  ierr         = PetscMalloc1(numLeavesTo,&toLeaves);CHKERRQ(ierr);
  ierr         = PetscMalloc1(numLeavesTo,&fromRoots);CHKERRQ(ierr);
  if (toCoarseFromFine) {
    ierr = PetscMalloc1(numLeavesFrom,&fromLeaves);CHKERRQ(ierr);
    ierr = PetscMalloc1(numLeavesFrom,&fromRoots);CHKERRQ(ierr);
  }
  ierr = DMPforestComputeLocalCellTransferSF_loop(p4estFrom,FromOffset,p4estTo,ToOffset,flt,llt,&numLeavesTo,toLeaves,fromRoots,&numLeavesFrom,fromLeaves,toRoots);CHKERRQ(ierr);
  if (!ToOffset && (numLeavesTo == numRootsTo)) { /* compress */
    ierr = PetscFree(toLeaves);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(fromCoarse,numRootsFrom,numLeavesTo,NULL,PETSC_OWN_POINTER,fromRoots,PETSC_OWN_POINTER);CHKERRQ(ierr);
  } else { /* generic */
    ierr = PetscSFSetGraph(fromCoarse,numRootsFrom,numLeavesTo,toLeaves,PETSC_OWN_POINTER,fromRoots,PETSC_OWN_POINTER);CHKERRQ(ierr);
  }
  *fromCoarseToFine = fromCoarse;
  if (toCoarseFromFine) {
    ierr              = PetscSFSetGraph(toCoarse,numRootsTo,numLeavesFrom,fromLeaves,PETSC_OWN_POINTER,toRoots,PETSC_OWN_POINTER);CHKERRQ(ierr);
    *toCoarseFromFine = toCoarse;
  }
  PetscFunctionReturn(0);
}

/* range of processes whose B sections overlap this ranks A section */
static PetscErrorCode DMPforestComputeOverlappingRanks(PetscMPIInt size, PetscMPIInt rank, p4est_t *p4estA, p4est_t *p4estB, PetscInt *startB, PetscInt *endB)
{
  p4est_quadrant_t * myCoarseStart = &(p4estA->global_first_position[rank]);
  p4est_quadrant_t * myCoarseEnd   = &(p4estA->global_first_position[rank+1]);
  p4est_quadrant_t * globalFirstB  = p4estB->global_first_position;

  PetscFunctionBegin;
  *startB = -1;
  *endB   = -1;
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
      } else if (startCompMy > 0) {  /* guess is to high */
        hi = guess;
      } else { /* guess is to low */
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
      } else if (startCompMy >= 0) { /* guess is to high */
        hi = guess;
      } else { /* guess is to low */
        lo = guess + 1;
      }
      guess = lo + (hi - lo) / 2;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPforestGetPlex(DM,DM*);

#define DMSetUp_pforest _append_pforest(DMSetUp)
static PetscErrorCode DMSetUp_pforest(DM dm)
{
  DM_Forest         *forest  = (DM_Forest*) dm->data;
  DM_Forest_pforest *pforest = (DM_Forest_pforest*) forest->data;
  DM                base, adaptFrom;
  DMForestTopology  topoName;
  PetscSF           preCoarseToFine = NULL, coarseToPreFine = NULL;
  PforestAdaptCtx   ctx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ctx.minLevel  = PETSC_MAX_INT;
  ctx.maxLevel  = 0;
  ctx.currLevel = 0;
  ctx.anyChange = PETSC_FALSE;
  /* sanity check */
  ierr = DMForestGetAdaptivityForest(dm,&adaptFrom);CHKERRQ(ierr);
  ierr = DMForestGetBaseDM(dm,&base);CHKERRQ(ierr);
  ierr = DMForestGetTopology(dm,&topoName);CHKERRQ(ierr);
  if (!adaptFrom && !base && !topoName) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"A forest needs a topology, a base DM, or a DM to adapt from");

  /* === Step 1: DMFTopology === */
  if (adaptFrom) { /* reference already created topology */
    PetscBool         ispforest;
    DM_Forest         *aforest  = (DM_Forest*) adaptFrom->data;
    DM_Forest_pforest *apforest = (DM_Forest_pforest*) aforest->data;

    ierr = PetscObjectTypeCompare((PetscObject)adaptFrom,DMPFOREST,&ispforest);CHKERRQ(ierr);
    if (!ispforest) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_NOTSAMETYPE,"Trying to adapt from %s, which is not %s",((PetscObject)adaptFrom)->type_name,DMPFOREST);
    if (!apforest->topo) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"The pre-adaptation forest must have a topology");
    ierr = DMSetUp(adaptFrom);CHKERRQ(ierr);
    ierr = DMForestGetBaseDM(dm,&base);CHKERRQ(ierr);
    ierr = DMForestGetTopology(dm,&topoName);CHKERRQ(ierr);
  } else if (base) { /* construct a connectivity from base */
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
      PetscInt             *tree_face_to_uniq = NULL;
      PetscErrorCode       ierr;

      ierr = DMPlexGetDepth(base,&depth);CHKERRQ(ierr);
      if (depth == 1) {
        DM connDM;

        ierr = DMPlexInterpolate(base,&connDM);CHKERRQ(ierr);
        base = connDM;
        ierr = DMForestSetBaseDM(dm,base);CHKERRQ(ierr);
        ierr = DMDestroy(&connDM);CHKERRQ(ierr);
      } else if (depth != P4EST_DIM) SETERRQ2(comm,PETSC_ERR_ARG_WRONG,"Base plex is neither interpolated nor uninterpolated? depth %D, expected 2 or %d",depth,P4EST_DIM + 1);
      ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
      if (size > 1) {
        DM      dmRedundant;
        PetscSF sf;

        ierr = DMPlexGetRedundantDM(base,&sf,&dmRedundant);CHKERRQ(ierr);
        if (!dmRedundant) SETERRQ(comm,PETSC_ERR_PLIB,"Could not create redundant DM");
        ierr = PetscObjectCompose((PetscObject)dmRedundant,"_base_migration_sf",(PetscObject)sf);CHKERRQ(ierr);
        ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
        base = dmRedundant;
        ierr = DMForestSetBaseDM(dm,base);CHKERRQ(ierr);
        ierr = DMDestroy(&dmRedundant);CHKERRQ(ierr);
      }
      ierr        = DMViewFromOptions(base,NULL,"-dm_p4est_base_view");CHKERRQ(ierr);
      ierr        = DMPlexCreateConnectivity_pforest(base,&conn,&tree_face_to_uniq);CHKERRQ(ierr);
      ierr        = PetscNewLog(dm,&topo);CHKERRQ(ierr);
      topo->refct = 1;
      topo->conn  = conn;
      topo->geom  = NULL;
      {
        PetscErrorCode (*map)(DM,PetscInt,PetscInt,const PetscReal[],PetscReal[],void*);
        void           *mapCtx;

        ierr = DMForestGetBaseCoordinateMapping(dm,&map,&mapCtx);CHKERRQ(ierr);
        if (map) {
          DM_Forest_geometry_pforest *geom_pforest;
          p4est_geometry_t           *geom;

          ierr                 = PetscNew(&geom_pforest);CHKERRQ(ierr);
          ierr                 = DMGetCoordinateDim(dm,&geom_pforest->coordDim);CHKERRQ(ierr);
          geom_pforest->map    = map;
          geom_pforest->mapCtx = mapCtx;
          PetscStackCallP4estReturn(geom_pforest->inner,p4est_geometry_new_connectivity,(conn));
          ierr          = PetscNew(&geom);CHKERRQ(ierr);
          geom->name    = topoName;
          geom->user    = geom_pforest;
          geom->X       = GeometryMapping_pforest;
          geom->destroy = GeometryDestroy_pforest;
          topo->geom    = geom;
        }
      }
      topo->tree_face_to_uniq = tree_face_to_uniq;
      pforest->topo           = topo;
    } else if (isDA) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Not implemented yet");
#if 0
      PetscInt N[3], P[3];

      /* get the sizes, periodicities */
      /* ... */
                                                                  /* don't use Morton order */
      ierr = DMFTopologyCreateBrick_pforest(dm,N,P,&pforest->topo,PETSC_FALSE);CHKERRQ(ierr);
#endif
    {
      PetscInt numLabels, l;

      ierr = DMGetNumLabels(base,&numLabels);CHKERRQ(ierr);
      for (l = 0; l < numLabels; l++) {
        PetscBool  isDepth, isGhost, isVTK, isDim;
        DMLabel    label, labelNew;
        PetscInt   defVal;
        const char *name;

        ierr = DMGetLabelName(base, l, &name);CHKERRQ(ierr);
        ierr = DMGetLabelByNum(base, l, &label);CHKERRQ(ierr);
        ierr = PetscStrcmp(name,"depth",&isDepth);CHKERRQ(ierr);
        if (isDepth) continue;
        ierr = PetscStrcmp(name,"dim",&isDim);CHKERRQ(ierr);
        if (isDim) continue;
        ierr = PetscStrcmp(name,"ghost",&isGhost);CHKERRQ(ierr);
        if (isGhost) continue;
        ierr = PetscStrcmp(name,"vtk",&isVTK);CHKERRQ(ierr);
        if (isVTK) continue;
        ierr = DMCreateLabel(dm,name);CHKERRQ(ierr);
        ierr = DMGetLabel(dm,name,&labelNew);CHKERRQ(ierr);
        ierr = DMLabelGetDefaultValue(label,&defVal);CHKERRQ(ierr);
        ierr = DMLabelSetDefaultValue(labelNew,defVal);CHKERRQ(ierr);
      }
      /* map dm points (internal plex) to base
         we currently create the subpoint_map for the entire hierarchy, starting from the finest forest
         and propagating back to the coarsest
         This is not an optimal approach, since we need the map only on the coarsest level
         during DMForestTransferVecFromBase */
      ierr = DMForestGetMinimumRefinement(dm,&l);CHKERRQ(ierr);
      if (!l) {
        ierr = DMCreateLabel(dm,"_forest_base_subpoint_map");CHKERRQ(ierr);
      }
    }
  } else { /* construct from topology name */
    DMFTopology_pforest *topo;

    ierr          = DMFTopologyCreate_pforest(dm,topoName,&topo);CHKERRQ(ierr);
    pforest->topo = topo;
    /* TODO: construct base? */
  }

  /* === Step 2: get the leaves of the forest === */
  if (adaptFrom) { /* start with the old forest */
    DMLabel           adaptLabel;
    PetscInt          defaultValue;
    PetscInt          numValues, numValuesGlobal, cLocalStart, count;
    DM_Forest         *aforest  = (DM_Forest*) adaptFrom->data;
    DM_Forest_pforest *apforest = (DM_Forest_pforest*) aforest->data;
    PetscBool         computeAdaptSF;
    p4est_topidx_t    flt, llt, t;

    flt         = apforest->forest->first_local_tree;
    llt         = apforest->forest->last_local_tree;
    cLocalStart = apforest->cLocalStart;
    ierr = DMForestGetComputeAdaptivitySF(dm,&computeAdaptSF);CHKERRQ(ierr);
    PetscStackCallP4estReturn(pforest->forest,p4est_copy,(apforest->forest, 0)); /* 0 indicates no data copying */
    ierr = DMForestGetAdaptivityLabel(dm,&adaptLabel);CHKERRQ(ierr);
    if (adaptLabel) {
      /* apply the refinement/coarsening by flags, plus minimum/maximum refinement */
      ierr = DMLabelGetNumValues(adaptLabel,&numValues);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&numValues,&numValuesGlobal,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)adaptFrom));CHKERRQ(ierr);
      ierr = DMLabelGetDefaultValue(adaptLabel,&defaultValue);CHKERRQ(ierr);
      if (!numValuesGlobal && defaultValue == DM_ADAPT_COARSEN_LAST) { /* uniform coarsen of the last level only (equivalent to DM_ADAPT_COARSEN for conforming grids)  */
        ierr                          = DMForestGetMinimumRefinement(dm,&ctx.minLevel);CHKERRQ(ierr);
        ierr                          = DMPforestGetRefinementLevel(dm,&ctx.currLevel);CHKERRQ(ierr);
        pforest->forest->user_pointer = (void*) &ctx;
        PetscStackCallP4est(p4est_coarsen,(pforest->forest,0,pforest_coarsen_currlevel,NULL));
        pforest->forest->user_pointer = (void*) dm;
        PetscStackCallP4est(p4est_balance,(pforest->forest,P4EST_CONNECT_FULL,NULL));
        /* we will have to change the offset after we compute the overlap */
        if (computeAdaptSF) {
          ierr = DMPforestComputeLocalCellTransferSF(PetscObjectComm((PetscObject)dm),pforest->forest,0,apforest->forest,apforest->cLocalStart,&coarseToPreFine,NULL);CHKERRQ(ierr);
        }
      } else if (!numValuesGlobal && defaultValue == DM_ADAPT_COARSEN) { /* uniform coarsen */
        ierr                          = DMForestGetMinimumRefinement(dm,&ctx.minLevel);CHKERRQ(ierr);
        pforest->forest->user_pointer = (void*) &ctx;
        PetscStackCallP4est(p4est_coarsen,(pforest->forest,0,pforest_coarsen_uniform,NULL));
        pforest->forest->user_pointer = (void*) dm;
        PetscStackCallP4est(p4est_balance,(pforest->forest,P4EST_CONNECT_FULL,NULL));
        /* we will have to change the offset after we compute the overlap */
        if (computeAdaptSF) {
          ierr = DMPforestComputeLocalCellTransferSF(PetscObjectComm((PetscObject)dm),pforest->forest,0,apforest->forest,apforest->cLocalStart,&coarseToPreFine,NULL);CHKERRQ(ierr);
        }
      } else if (!numValuesGlobal && defaultValue == DM_ADAPT_REFINE) { /* uniform refine */
        ierr                          = DMForestGetMaximumRefinement(dm,&ctx.maxLevel);CHKERRQ(ierr);
        pforest->forest->user_pointer = (void*) &ctx;
        PetscStackCallP4est(p4est_refine,(pforest->forest,0,pforest_refine_uniform,NULL));
        pforest->forest->user_pointer = (void*) dm;
        PetscStackCallP4est(p4est_balance,(pforest->forest,P4EST_CONNECT_FULL,NULL));
        /* we will have to change the offset after we compute the overlap */
        if (computeAdaptSF) {
          ierr = DMPforestComputeLocalCellTransferSF(PetscObjectComm((PetscObject)dm),apforest->forest,apforest->cLocalStart,pforest->forest,0,&preCoarseToFine,NULL);CHKERRQ(ierr);
        }
      } else if (numValuesGlobal) {
        p4est_t                    *p4est = pforest->forest;
        PetscInt                   *cellFlags;
        DMForestAdaptivityStrategy strategy;
        PetscSF                    cellSF;
        PetscInt                   c, cStart, cEnd;
        PetscBool                  adaptAny;

        ierr = DMForestGetMaximumRefinement(dm,&ctx.maxLevel);CHKERRQ(ierr);
        ierr = DMForestGetMinimumRefinement(dm,&ctx.minLevel);CHKERRQ(ierr);
        ierr = DMForestGetAdaptivityStrategy(dm,&strategy);CHKERRQ(ierr);
        ierr = PetscStrncmp(strategy,"any",3,&adaptAny);CHKERRQ(ierr);
        ierr = DMForestGetCellChart(adaptFrom,&cStart,&cEnd);CHKERRQ(ierr);
        ierr = DMForestGetCellSF(adaptFrom,&cellSF);CHKERRQ(ierr);
        ierr = PetscMalloc1(cEnd-cStart,&cellFlags);CHKERRQ(ierr);
        for (c = cStart; c < cEnd; c++) {ierr = DMLabelGetValue(adaptLabel,c,&cellFlags[c-cStart]);CHKERRQ(ierr);}
        if (cellSF) {
          if (adaptAny) {
            ierr = PetscSFReduceBegin(cellSF,MPIU_INT,cellFlags,cellFlags,MPI_MAX);CHKERRQ(ierr);
            ierr = PetscSFReduceEnd(cellSF,MPIU_INT,cellFlags,cellFlags,MPI_MAX);CHKERRQ(ierr);
          } else {
            ierr = PetscSFReduceBegin(cellSF,MPIU_INT,cellFlags,cellFlags,MPI_MIN);CHKERRQ(ierr);
            ierr = PetscSFReduceEnd(cellSF,MPIU_INT,cellFlags,cellFlags,MPI_MIN);CHKERRQ(ierr);
          }
        }
        for (t = flt, count = cLocalStart; t <= llt; t++) {
          p4est_tree_t       *tree    = &(((p4est_tree_t*) p4est->trees->array)[t]);
          PetscInt           numQuads = (PetscInt) tree->quadrants.elem_count, i;
          p4est_quadrant_t   *quads   = (p4est_quadrant_t *) tree->quadrants.array;

          for (i = 0; i < numQuads; i++) {
            p4est_quadrant_t *q = &quads[i];
            q->p.user_int = cellFlags[count++];
          }
        }
        ierr = PetscFree(cellFlags);CHKERRQ(ierr);

        pforest->forest->user_pointer = (void*) &ctx;
        if (adaptAny) {
          PetscStackCallP4est(p4est_coarsen,(pforest->forest,0,pforest_coarsen_flag_any,pforest_init_determine));
        } else {
          PetscStackCallP4est(p4est_coarsen,(pforest->forest,0,pforest_coarsen_flag_all,pforest_init_determine));
        }
        PetscStackCallP4est(p4est_refine,(pforest->forest,0,pforest_refine_flag,NULL));
        pforest->forest->user_pointer = (void*) dm;
        PetscStackCallP4est(p4est_balance,(pforest->forest,P4EST_CONNECT_FULL,NULL));
        if (computeAdaptSF) {
          ierr = DMPforestComputeLocalCellTransferSF(PetscObjectComm((PetscObject)dm),apforest->forest,apforest->cLocalStart,pforest->forest,0,&preCoarseToFine,&coarseToPreFine);CHKERRQ(ierr);
        }
      }
      for (t = flt, count = cLocalStart; t <= llt; t++) {
        p4est_tree_t       *atree    = &(((p4est_tree_t*) apforest->forest->trees->array)[t]);
        p4est_tree_t       *tree     = &(((p4est_tree_t*) pforest->forest->trees->array)[t]);
        PetscInt           anumQuads = (PetscInt) atree->quadrants.elem_count, i;
        PetscInt           numQuads  = (PetscInt) tree->quadrants.elem_count;
        p4est_quadrant_t   *aquads   = (p4est_quadrant_t *) atree->quadrants.array;
        p4est_quadrant_t   *quads    = (p4est_quadrant_t *) tree->quadrants.array;

        if (anumQuads != numQuads) {
          ctx.anyChange = PETSC_TRUE;
        } else {
          for (i = 0; i < numQuads; i++) {
            p4est_quadrant_t *aq = &aquads[i];
            p4est_quadrant_t *q  = &quads[i];

            if (aq->level != q->level) {
              ctx.anyChange = PETSC_TRUE;
              break;
            }
          }
        }
        if (ctx.anyChange) {
          break;
        }
      }
    }
    {
      PetscInt numLabels, l;

      ierr = DMGetNumLabels(adaptFrom,&numLabels);CHKERRQ(ierr);
      for (l = 0; l < numLabels; l++) {
        PetscBool  isDepth, isGhost, isVTK;
        DMLabel    label, labelNew;
        PetscInt   defVal;
        const char *name;

        ierr = DMGetLabelName(adaptFrom, l, &name);CHKERRQ(ierr);
        ierr = DMGetLabelByNum(adaptFrom, l, &label);CHKERRQ(ierr);
        ierr = PetscStrcmp(name,"depth",&isDepth);CHKERRQ(ierr);
        if (isDepth) continue;
        ierr = PetscStrcmp(name,"ghost",&isGhost);CHKERRQ(ierr);
        if (isGhost) continue;
        ierr = PetscStrcmp(name,"vtk",&isVTK);CHKERRQ(ierr);
        if (isVTK) continue;
        ierr = DMCreateLabel(dm,name);CHKERRQ(ierr);
        ierr = DMGetLabel(dm,name,&labelNew);CHKERRQ(ierr);
        ierr = DMLabelGetDefaultValue(label,&defVal);CHKERRQ(ierr);
        ierr = DMLabelSetDefaultValue(labelNew,defVal);CHKERRQ(ierr);
      }
    }
  } else { /* initial */
    PetscInt initLevel, minLevel;

    ierr = DMForestGetInitialRefinement(dm,&initLevel);CHKERRQ(ierr);
    ierr = DMForestGetMinimumRefinement(dm,&minLevel);CHKERRQ(ierr);
    PetscStackCallP4estReturn(pforest->forest,p4est_new_ext,(PetscObjectComm((PetscObject)dm),pforest->topo->conn,
                                                             0,           /* minimum number of quadrants per processor */
                                                             initLevel,   /* level of refinement */
                                                             1,           /* uniform refinement */
                                                             0,           /* we don't allocate any per quadrant data */
                                                             NULL,        /* there is no special quadrant initialization */
                                                             (void*)dm)); /* this dm is the user context */

    if (initLevel > minLevel) pforest->coarsen_hierarchy = PETSC_TRUE;
    if (dm->setfromoptionscalled) {
      PetscBool  flgPattern, flgFractal;
      PetscInt   corner = 0;
      PetscInt   corners[P4EST_CHILDREN], ncorner = P4EST_CHILDREN;
      PetscReal  likelihood = 1./ P4EST_DIM;
      PetscInt   pattern;
      const char *prefix;

      ierr = PetscObjectGetOptionsPrefix((PetscObject)dm,&prefix);CHKERRQ(ierr);
      ierr = PetscOptionsGetEList(((PetscObject)dm)->options,prefix,"-dm_p4est_refine_pattern",DMRefinePatternName,PATTERN_COUNT,&pattern,&flgPattern);CHKERRQ(ierr);
      ierr = PetscOptionsGetInt(((PetscObject)dm)->options,prefix,"-dm_p4est_refine_corner",&corner,NULL);CHKERRQ(ierr);
      ierr = PetscOptionsGetIntArray(((PetscObject)dm)->options,prefix,"-dm_p4est_refine_fractal_corners",corners,&ncorner,&flgFractal);CHKERRQ(ierr);
      ierr = PetscOptionsGetReal(((PetscObject)dm)->options,prefix,"-dm_p4est_refine_hash_likelihood",&likelihood,NULL);CHKERRQ(ierr);

      if (flgPattern) {
        DMRefinePatternCtx *ctx;
        PetscInt           maxLevel;

        ierr          = DMForestGetMaximumRefinement(dm,&maxLevel);CHKERRQ(ierr);
        ierr          = PetscNewLog(dm,&ctx);CHKERRQ(ierr);
        ctx->maxLevel = PetscMin(maxLevel,P4EST_QMAXLEVEL);
        if (initLevel + ctx->maxLevel > minLevel) pforest->coarsen_hierarchy = PETSC_TRUE;
        switch (pattern) {
        case PATTERN_HASH:
          ctx->refine_fn      = DMRefinePattern_Hash;
          ctx->hashLikelihood = likelihood;
          break;
        case PATTERN_CORNER:
          ctx->corner    = corner;
          ctx->refine_fn = DMRefinePattern_Corner;
          break;
        case PATTERN_CENTER:
          ctx->refine_fn = DMRefinePattern_Center;
          break;
        case PATTERN_FRACTAL:
          if (flgFractal) {
            PetscInt i;

            for (i = 0; i < ncorner; i++) ctx->fractal[corners[i]] = PETSC_TRUE;
          } else {
#if !defined(P4_TO_P8)
            ctx->fractal[0] = ctx->fractal[1] = ctx->fractal[2] = PETSC_TRUE;
#else
            ctx->fractal[0] = ctx->fractal[3] = ctx->fractal[5] = ctx->fractal[6] = PETSC_TRUE;
#endif
          }
          ctx->refine_fn = DMRefinePattern_Fractal;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"Not a valid refinement pattern");
        }

        pforest->forest->user_pointer = (void*) ctx;
        PetscStackCallP4est(p4est_refine,(pforest->forest,1,ctx->refine_fn,NULL));
        PetscStackCallP4est(p4est_balance,(pforest->forest,P4EST_CONNECT_FULL,NULL));
        ierr                          = PetscFree(ctx);CHKERRQ(ierr);
        pforest->forest->user_pointer = (void*) dm;
      }
    }
  }
  if (pforest->coarsen_hierarchy) {
    PetscInt initLevel, currLevel, minLevel;

    ierr = DMPforestGetRefinementLevel(dm,&currLevel);CHKERRQ(ierr);
    ierr = DMForestGetInitialRefinement(dm,&initLevel);CHKERRQ(ierr);
    ierr = DMForestGetMinimumRefinement(dm,&minLevel);CHKERRQ(ierr);
    if (currLevel > minLevel) {
      DM_Forest_pforest *coarse_pforest;
      DMLabel           coarsen;
      DM                coarseDM;

      ierr = DMForestTemplate(dm,MPI_COMM_NULL,&coarseDM);CHKERRQ(ierr);
      ierr = DMForestSetAdaptivityPurpose(coarseDM,DM_ADAPT_COARSEN);CHKERRQ(ierr);
      ierr = DMLabelCreate(PETSC_COMM_SELF, "coarsen",&coarsen);CHKERRQ(ierr);
      ierr = DMLabelSetDefaultValue(coarsen,DM_ADAPT_COARSEN);CHKERRQ(ierr);
      ierr = DMForestSetAdaptivityLabel(coarseDM,coarsen);CHKERRQ(ierr);
      ierr = DMLabelDestroy(&coarsen);CHKERRQ(ierr);
      ierr = DMSetCoarseDM(dm,coarseDM);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)coarseDM);CHKERRQ(ierr);
      initLevel = currLevel == initLevel ? initLevel - 1 : initLevel;
      ierr                              = DMForestSetInitialRefinement(coarseDM,initLevel);CHKERRQ(ierr);
      ierr                              = DMForestSetMinimumRefinement(coarseDM,minLevel);CHKERRQ(ierr);
      coarse_pforest                    = (DM_Forest_pforest*) ((DM_Forest*) coarseDM->data)->data;
      coarse_pforest->coarsen_hierarchy = PETSC_TRUE;
    }
  }

  { /* repartitioning and overlap */
    PetscMPIInt size, rank;

    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)dm),&size);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRQ(ierr);
    if ((size > 1) && (pforest->partition_for_coarsening || forest->cellWeights || forest->weightCapacity != 1. || forest->weightsFactor != 1.)) {
      PetscBool      copyForest   = PETSC_FALSE;
      p4est_t        *forest_copy = NULL;
      p4est_gloidx_t shipped      = 0;

      if (preCoarseToFine || coarseToPreFine) copyForest = PETSC_TRUE;
      if (copyForest) PetscStackCallP4estReturn(forest_copy,p4est_copy,(pforest->forest,0));

      if (!forest->cellWeights && forest->weightCapacity == 1. && forest->weightsFactor == 1.) {
        PetscStackCallP4estReturn(shipped,p4est_partition_ext,(pforest->forest,(int)pforest->partition_for_coarsening,NULL));
      } else SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Non-uniform partition cases not implemented yet");
      if (shipped) ctx.anyChange = PETSC_TRUE;
      if (forest_copy) {
        if (preCoarseToFine || coarseToPreFine) {
          PetscSF        repartSF; /* repartSF has roots in the old partition */
          PetscInt       pStart = -1, pEnd = -1, p;
          PetscInt       numRoots, numLeaves;
          PetscSFNode    *repartRoots;
          p4est_gloidx_t postStart  = pforest->forest->global_first_quadrant[rank];
          p4est_gloidx_t postEnd    = pforest->forest->global_first_quadrant[rank+1];
          p4est_gloidx_t partOffset = postStart;

          numRoots  = (PetscInt) (forest_copy->global_first_quadrant[rank + 1] - forest_copy->global_first_quadrant[rank]);
          numLeaves = (PetscInt) (postEnd - postStart);
          ierr      = DMPforestComputeOverlappingRanks(size,rank,pforest->forest,forest_copy,&pStart,&pEnd);CHKERRQ(ierr);
          ierr      = PetscMalloc1((PetscInt) pforest->forest->local_num_quadrants,&repartRoots);CHKERRQ(ierr);
          for (p = pStart; p < pEnd; p++) {
            p4est_gloidx_t preStart = forest_copy->global_first_quadrant[p];
            p4est_gloidx_t preEnd   = forest_copy->global_first_quadrant[p+1];
            PetscInt       q;

            if (preEnd == preStart) continue;
            if (preStart > postStart) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Bad partition overlap computation");
            preEnd = preEnd > postEnd ? postEnd : preEnd;
            for (q = partOffset; q < preEnd; q++) {
              repartRoots[q - postStart].rank  = p;
              repartRoots[q - postStart].index = partOffset - preStart;
            }
            partOffset = preEnd;
          }
          ierr = PetscSFCreate(PetscObjectComm((PetscObject)dm),&repartSF);CHKERRQ(ierr);
          ierr = PetscSFSetGraph(repartSF,numRoots,numLeaves,NULL,PETSC_OWN_POINTER,repartRoots,PETSC_OWN_POINTER);CHKERRQ(ierr);
          ierr = PetscSFSetUp(repartSF);CHKERRQ(ierr);
          if (preCoarseToFine) {
            PetscSF        repartSFembed, preCoarseToFineNew;
            PetscInt       nleaves;
            const PetscInt *leaves;

            ierr = PetscSFSetUp(preCoarseToFine);CHKERRQ(ierr);
            ierr = PetscSFGetGraph(preCoarseToFine,NULL,&nleaves,&leaves,NULL);CHKERRQ(ierr);
            if (leaves) {
              ierr = PetscSFCreateEmbeddedSF(repartSF,nleaves,leaves,&repartSFembed);CHKERRQ(ierr);
            } else {
              repartSFembed = repartSF;
              ierr          = PetscObjectReference((PetscObject)repartSFembed);CHKERRQ(ierr);
            }
            ierr            = PetscSFCompose(preCoarseToFine,repartSFembed,&preCoarseToFineNew);CHKERRQ(ierr);
            ierr            = PetscSFDestroy(&preCoarseToFine);CHKERRQ(ierr);
            ierr            = PetscSFDestroy(&repartSFembed);CHKERRQ(ierr);
            preCoarseToFine = preCoarseToFineNew;
          }
          if (coarseToPreFine) {
            PetscSF repartSFinv, coarseToPreFineNew;

            ierr            = PetscSFCreateInverseSF(repartSF,&repartSFinv);CHKERRQ(ierr);
            ierr            = PetscSFCompose(repartSFinv,coarseToPreFine,&coarseToPreFineNew);CHKERRQ(ierr);
            ierr            = PetscSFDestroy(&coarseToPreFine);CHKERRQ(ierr);
            ierr            = PetscSFDestroy(&repartSFinv);CHKERRQ(ierr);
            coarseToPreFine = coarseToPreFineNew;
          }
          ierr = PetscSFDestroy(&repartSF);CHKERRQ(ierr);
        }
        PetscStackCallP4est(p4est_destroy,(forest_copy));
      }
    }
    if (size > 1) {
      PetscInt overlap;

      ierr = DMForestGetPartitionOverlap(dm,&overlap);CHKERRQ(ierr);

      if (adaptFrom) {
        PetscInt aoverlap;

        ierr = DMForestGetPartitionOverlap(adaptFrom,&aoverlap);CHKERRQ(ierr);
        if (aoverlap != overlap) {
          ctx.anyChange = PETSC_TRUE;
        }
      }

      if (overlap > 0) {
        PetscInt i, cLocalStart;
        PetscInt cEnd;
        PetscSF  preCellSF = NULL, cellSF = NULL;

        PetscStackCallP4estReturn(pforest->ghost,p4est_ghost_new,(pforest->forest,P4EST_CONNECT_FULL));
        PetscStackCallP4estReturn(pforest->lnodes,p4est_lnodes_new,(pforest->forest,pforest->ghost,-P4EST_DIM));
        PetscStackCallP4est(p4est_ghost_support_lnodes,(pforest->forest,pforest->lnodes,pforest->ghost));
        for (i = 1; i < overlap; i++) PetscStackCallP4est(p4est_ghost_expand_by_lnodes,(pforest->forest,pforest->lnodes,pforest->ghost));

        cLocalStart = pforest->cLocalStart = pforest->ghost->proc_offsets[rank];
        cEnd        = pforest->forest->local_num_quadrants + pforest->ghost->proc_offsets[size];

        /* shift sfs by cLocalStart, expand by cell SFs */
        if (preCoarseToFine || coarseToPreFine) {
          if (adaptFrom) {ierr = DMForestGetCellSF(adaptFrom,&preCellSF);CHKERRQ(ierr);}
          dm->setupcalled = PETSC_TRUE;
          ierr = DMForestGetCellSF(dm,&cellSF);CHKERRQ(ierr);
        }
        if (preCoarseToFine) {
          PetscSF           preCoarseToFineNew;
          PetscInt          nleaves, nroots, *leavesNew, i, nleavesNew;
          const PetscInt    *leaves;
          const PetscSFNode *remotes;
          PetscSFNode       *remotesAll;

          ierr = PetscSFSetUp(preCoarseToFine);CHKERRQ(ierr);
          ierr = PetscSFGetGraph(preCoarseToFine,&nroots,&nleaves,&leaves,&remotes);CHKERRQ(ierr);
          ierr = PetscMalloc1(cEnd,&remotesAll);CHKERRQ(ierr);
          for (i = 0; i < cEnd; i++) {
            remotesAll[i].rank  = -1;
            remotesAll[i].index = -1;
          }
          for (i = 0; i < nleaves; i++) remotesAll[(leaves ? leaves[i] : i) + cLocalStart] = remotes[i];
          ierr       = PetscSFSetUp(cellSF);CHKERRQ(ierr);
          ierr       = PetscSFBcastBegin(cellSF,MPIU_2INT,remotesAll,remotesAll);CHKERRQ(ierr);
          ierr       = PetscSFBcastEnd(cellSF,MPIU_2INT,remotesAll,remotesAll);CHKERRQ(ierr);
          nleavesNew = 0;
          for (i = 0; i < nleaves; i++) {
            if (remotesAll[i].rank >= 0) nleavesNew++;
          }
          ierr       = PetscMalloc1(nleavesNew,&leavesNew);CHKERRQ(ierr);
          nleavesNew = 0;
          for (i = 0; i < nleaves; i++) {
            if (remotesAll[i].rank >= 0) {
              leavesNew[nleavesNew] = i;
              if (i > nleavesNew) remotesAll[nleavesNew] = remotesAll[i];
              nleavesNew++;
            }
          }
          ierr = PetscSFCreate(PetscObjectComm((PetscObject)dm),&preCoarseToFineNew);CHKERRQ(ierr);
          if (nleavesNew < cEnd) {
            ierr = PetscSFSetGraph(preCoarseToFineNew,nroots,nleavesNew,leavesNew,PETSC_OWN_POINTER,remotesAll,PETSC_COPY_VALUES);CHKERRQ(ierr);
          } else { /* all cells are leaves */
            ierr = PetscFree(leavesNew);CHKERRQ(ierr);
            ierr = PetscSFSetGraph(preCoarseToFineNew,nroots,nleavesNew,NULL,PETSC_OWN_POINTER,remotesAll,PETSC_COPY_VALUES);CHKERRQ(ierr);
          }
          ierr            = PetscFree(remotesAll);CHKERRQ(ierr);
          ierr            = PetscSFDestroy(&preCoarseToFine);CHKERRQ(ierr);
          preCoarseToFine = preCoarseToFineNew;
          preCoarseToFine = preCoarseToFineNew;
        }
        if (coarseToPreFine) {
          PetscSF           coarseToPreFineNew;
          PetscInt          nleaves, nroots, i, nleavesCellSF, nleavesExpanded, *leavesNew;
          const PetscInt    *leaves;
          const PetscSFNode *remotes;
          PetscSFNode       *remotesNew, *remotesNewRoot, *remotesExpanded;

          ierr = PetscSFSetUp(coarseToPreFine);CHKERRQ(ierr);
          ierr = PetscSFGetGraph(coarseToPreFine,&nroots,&nleaves,&leaves,&remotes);CHKERRQ(ierr);
          ierr = PetscSFGetGraph(preCellSF,NULL,&nleavesCellSF,NULL,NULL);CHKERRQ(ierr);
          ierr = PetscMalloc1(nroots,&remotesNewRoot);CHKERRQ(ierr);
          ierr = PetscMalloc1(nleaves,&remotesNew);CHKERRQ(ierr);
          for (i = 0; i < nroots; i++) {
            remotesNewRoot[i].rank  = rank;
            remotesNewRoot[i].index = i + cLocalStart;
          }
          ierr = PetscSFBcastBegin(coarseToPreFine,MPIU_2INT,remotesNewRoot,remotesNew);CHKERRQ(ierr);
          ierr = PetscSFBcastEnd(coarseToPreFine,MPIU_2INT,remotesNewRoot,remotesNew);CHKERRQ(ierr);
          ierr = PetscFree(remotesNewRoot);CHKERRQ(ierr);
          ierr = PetscMalloc1(nleavesCellSF,&remotesExpanded);CHKERRQ(ierr);
          for (i = 0; i < nleavesCellSF; i++) {
            remotesExpanded[i].rank  = -1;
            remotesExpanded[i].index = -1;
          }
          for (i = 0; i < nleaves; i++) remotesExpanded[leaves ? leaves[i] : i] = remotesNew[i];
          ierr = PetscFree(remotesNew);CHKERRQ(ierr);
          ierr = PetscSFBcastBegin(preCellSF,MPIU_2INT,remotesExpanded,remotesExpanded);CHKERRQ(ierr);
          ierr = PetscSFBcastEnd(preCellSF,MPIU_2INT,remotesExpanded,remotesExpanded);CHKERRQ(ierr);

          nleavesExpanded = 0;
          for (i = 0; i < nleavesCellSF; i++) {
            if (remotesExpanded[i].rank >= 0) nleavesExpanded++;
          }
          ierr            = PetscMalloc1(nleavesExpanded,&leavesNew);CHKERRQ(ierr);
          nleavesExpanded = 0;
          for (i = 0; i < nleavesCellSF; i++) {
            if (remotesExpanded[i].rank >= 0) {
              leavesNew[nleavesExpanded] = i;
              if (i > nleavesExpanded) remotesExpanded[nleavesExpanded] = remotes[i];
              nleavesExpanded++;
            }
          }
          ierr = PetscSFCreate(PetscObjectComm((PetscObject)dm),&coarseToPreFineNew);CHKERRQ(ierr);
          if (nleavesExpanded < nleavesCellSF) {
            ierr = PetscSFSetGraph(coarseToPreFineNew,cEnd,nleavesExpanded,leavesNew,PETSC_OWN_POINTER,remotesExpanded,PETSC_COPY_VALUES);CHKERRQ(ierr);
          } else {
            ierr = PetscFree(leavesNew);CHKERRQ(ierr);
            ierr = PetscSFSetGraph(coarseToPreFineNew,cEnd,nleavesExpanded,NULL,PETSC_OWN_POINTER,remotesExpanded,PETSC_COPY_VALUES);CHKERRQ(ierr);
          }
          ierr            = PetscFree(remotesExpanded);CHKERRQ(ierr);
          ierr            = PetscSFDestroy(&coarseToPreFine);CHKERRQ(ierr);
          coarseToPreFine = coarseToPreFineNew;
        }
      }
    }
  }
  forest->preCoarseToFine = preCoarseToFine;
  forest->coarseToPreFine = coarseToPreFine;
  dm->setupcalled         = PETSC_TRUE;
  ierr = MPI_Allreduce(&ctx.anyChange,&(pforest->adaptivitySuccess),1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
  ierr = DMPforestGetPlex(dm,NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define DMForestGetAdaptivitySuccess_pforest _append_pforest(DMForestGetAdaptivitySuccess)
static PetscErrorCode DMForestGetAdaptivitySuccess_pforest(DM dm, PetscBool *success)
{
  DM_Forest         *forest;
  DM_Forest_pforest *pforest;

  PetscFunctionBegin;
  forest   = (DM_Forest *) dm->data;
  pforest  = (DM_Forest_pforest *) forest->data;
  *success = pforest->adaptivitySuccess;
  PetscFunctionReturn(0);
}

#define DMView_ASCII_pforest _append_pforest(DMView_ASCII)
static PetscErrorCode DMView_ASCII_pforest(PetscObject odm, PetscViewer viewer)
{
  DM             dm = (DM) odm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  switch (viewer->format) {
  case PETSC_VIEWER_DEFAULT:
  case PETSC_VIEWER_ASCII_INFO:
  {
    PetscInt   dim;
    const char *name;

    ierr = PetscObjectGetName((PetscObject) dm, &name);CHKERRQ(ierr);
    ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
    if (name) {ierr = PetscViewerASCIIPrintf(viewer, "Forest %s in %D dimensions:\n", name, dim);CHKERRQ(ierr);}
    else      {ierr = PetscViewerASCIIPrintf(viewer, "Forest in %D dimensions:\n", dim);CHKERRQ(ierr);}
  }
  case PETSC_VIEWER_ASCII_INFO_DETAIL:
  case PETSC_VIEWER_LOAD_BALANCE:
  {
    DM plex;

    ierr = DMPforestGetPlex(dm, &plex);CHKERRQ(ierr);
    ierr = DMView(plex, viewer);CHKERRQ(ierr);
  }
  break;
  default: SETERRQ1(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "No support for format '%s'", PetscViewerFormats[viewer->format]);
  }
  PetscFunctionReturn(0);
}

#define DMView_VTK_pforest _append_pforest(DMView_VTK)
static PetscErrorCode DMView_VTK_pforest(PetscObject odm, PetscViewer viewer)
{
  DM                dm       = (DM) odm;
  DM_Forest         *forest  = (DM_Forest*) dm->data;
  DM_Forest_pforest *pforest = (DM_Forest_pforest*) forest->data;
  PetscBool         isvtk;
  PetscReal         vtkScale = 1. - PETSC_MACHINE_EPSILON;
  PetscViewer_VTK   *vtk     = (PetscViewer_VTK*)viewer->data;
  const char        *name;
  char              *filenameStrip = NULL;
  PetscBool         hasExt;
  size_t            len;
  p4est_geometry_t  *geom;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  geom = pforest->topo->geom;
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK, &isvtk);CHKERRQ(ierr);
  if (!isvtk) SETERRQ1(PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_INCOMP, "Cannot use viewer type %s", ((PetscObject)viewer)->type_name);
  switch (viewer->format) {
  case PETSC_VIEWER_VTK_VTU:
    if (!pforest->forest) SETERRQ (PetscObjectComm(odm),PETSC_ERR_ARG_WRONG,"DM has not been setup with a valid forest");
    name = vtk->filename;
    ierr = PetscStrlen(name,&len);CHKERRQ(ierr);
    ierr = PetscStrcasecmp(name+len-4,".vtu",&hasExt);CHKERRQ(ierr);
    if (hasExt) {
      ierr                = PetscStrallocpy(name,&filenameStrip);CHKERRQ(ierr);
      filenameStrip[len-4]='\0';
      name                = filenameStrip;
    }
    if (!pforest->topo->geom) PetscStackCallP4estReturn(geom,p4est_geometry_new_connectivity,(pforest->topo->conn));
    {
      p4est_vtk_context_t *pvtk;
      int                 footerr;

      PetscStackCallP4estReturn(pvtk,p4est_vtk_context_new,(pforest->forest,name));
      PetscStackCallP4est(p4est_vtk_context_set_geom,(pvtk,geom));
      PetscStackCallP4est(p4est_vtk_context_set_scale,(pvtk,(double)vtkScale));
      PetscStackCallP4estReturn(pvtk,p4est_vtk_write_header,(pvtk));
      if (!pvtk) SETERRQ(PetscObjectComm((PetscObject)odm),PETSC_ERR_LIB,P4EST_STRING "_vtk_write_header() failed");
      PetscStackCallP4estReturn(pvtk,p4est_vtk_write_cell_dataf,(pvtk,
                                                                 1, /* write tree */
                                                                 1, /* write level */
                                                                 1, /* write rank */
                                                                 0, /* do not wrap rank */
                                                                 0, /* no scalar fields */
                                                                 0, /* no vector fields */
                                                                 pvtk));
      if (!pvtk) SETERRQ(PetscObjectComm((PetscObject)odm),PETSC_ERR_LIB,P4EST_STRING "_vtk_write_cell_dataf() failed");
      PetscStackCallP4estReturn(footerr,p4est_vtk_write_footer,(pvtk));
      if (footerr) SETERRQ(PetscObjectComm((PetscObject)odm),PETSC_ERR_LIB,P4EST_STRING "_vtk_write_footer() failed");
    }
    if (!pforest->topo->geom) PetscStackCallP4est(p4est_geometry_destroy,(geom));
    ierr = PetscFree(filenameStrip);CHKERRQ(ierr);
    break;
  default: SETERRQ1(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "No support for format '%s'", PetscViewerFormats[viewer->format]);
  }
  PetscFunctionReturn(0);
}

#define DMView_HDF5_pforest _append_pforest(DMView_HDF5)
static PetscErrorCode DMView_HDF5_pforest(DM dm, PetscViewer viewer)
{
  DM             plex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = DMPforestGetPlex(dm, &plex);CHKERRQ(ierr);
  ierr = DMView(plex, viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define DMView_GLVis_pforest _append_pforest(DMView_GLVis)
static PetscErrorCode DMView_GLVis_pforest(DM dm, PetscViewer viewer)
{
  DM             plex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMSetUp(dm);CHKERRQ(ierr);
  ierr = DMPforestGetPlex(dm, &plex);CHKERRQ(ierr);
  ierr = DMView(plex, viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define DMView_pforest _append_pforest(DMView)
static PetscErrorCode DMView_pforest(DM dm, PetscViewer viewer)
{
  PetscBool      isascii, isvtk, ishdf5, isglvis;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK,   &isvtk);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,  &ishdf5);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERGLVIS, &isglvis);CHKERRQ(ierr);
  if (isascii) {
    ierr = DMView_ASCII_pforest((PetscObject) dm,viewer);CHKERRQ(ierr);
  } else if (isvtk) {
    ierr = DMView_VTK_pforest((PetscObject) dm,viewer);CHKERRQ(ierr);
  } else if (ishdf5) {
    ierr = DMView_HDF5_pforest(dm, viewer);CHKERRQ(ierr);
  } else if (isglvis) {
    ierr = DMView_GLVis_pforest(dm, viewer);CHKERRQ(ierr);
  } else SETERRQ(PetscObjectComm((PetscObject) dm),PETSC_ERR_SUP,"Viewer not supported (not VTK, HDF5, or GLVis)");
  PetscFunctionReturn(0);
}

static PetscErrorCode PforestConnectivityEnumerateFacets(p4est_connectivity_t *conn, PetscInt **tree_face_to_uniq)
{
  PetscInt       *ttf, f, t, g, count;
  PetscInt       numFacets;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  numFacets = conn->num_trees * P4EST_FACES;
  ierr      = PetscMalloc1(numFacets,&ttf);CHKERRQ(ierr);
  for (f = 0; f < numFacets; f++) ttf[f] = -1;
  for (g = 0, count = 0, t = 0; t < conn->num_trees; t++) {
    for (f = 0; f < P4EST_FACES; f++, g++) {
      if (ttf[g] == -1) {
        PetscInt ng;

        ttf[g]  = count++;
        ng      = conn->tree_to_tree[g] * P4EST_FACES + (conn->tree_to_face[g] % P4EST_FACES);
        ttf[ng] = ttf[g];
      }
    }
  }
  *tree_face_to_uniq = ttf;
  PetscFunctionReturn(0);
}

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
  PetscInt             cStart, cEnd, cEndInterior, c, vStart, vEnd, vEndInterior, v, fStart, fEnd, fEndInterior, f, eEndInterior;
  PetscInt             *star = NULL, *closure = NULL, closureSize, starSize, cttSize;
  PetscInt             *ttf;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  /* 1: count objects, allocate */
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(dm, &cEndInterior, &fEndInterior, &eEndInterior, &vEndInterior);CHKERRQ(ierr);
  cEnd = cEndInterior < 0 ? cEnd : cEndInterior;
  ierr     = P4estTopidxCast(cEnd-cStart,&numTrees);CHKERRQ(ierr);
  numVerts = P4EST_CHILDREN * numTrees;
  ierr     = DMPlexGetDepthStratum(dm,0,&vStart,&vEnd);CHKERRQ(ierr);
  vEnd     = vEndInterior < 0 ? vEnd : vEndInterior;
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
        if (closureSize != P4EST_INSUL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Cell with wrong closure size");
        for (c = 0; c < P4EST_CHILDREN; c++) {
          PetscInt cellVert = closure[2 * (c + vertOff)];

          if (cellVert < vStart || cellVert >= vEnd) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Non-standard closure: vertices");
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
  eEnd = eEndInterior < 0 ? eEnd : eEndInterior;
  ierr = P4estTopidxCast(eEnd-eStart,&numEdges);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PETSC_COMM_SELF,&ett);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(ett,eStart,eEnd);CHKERRQ(ierr);
  for (e = eStart; e < eEnd; e++) {
    PetscInt s;

    ierr = DMPlexGetTransitiveClosure(dm,e,PETSC_FALSE,&starSize,&star);CHKERRQ(ierr);
    for (s = 0; s < starSize; s++) {
      PetscInt p = star[2*s];

      if (p >= cStart && p < cEnd) {
        /* we want to count every time cell p references e, so we see how many times it comes up in the closure.  This
         * only protects against periodicity problems */
        ierr = DMPlexGetTransitiveClosure(dm,p,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
        if (closureSize != P4EST_INSUL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Cell with wrong closure size");
        for (c = 0; c < P8EST_EDGES; c++) {
          PetscInt cellEdge = closure[2 * (c + edgeOff)];

          if (cellEdge < eStart || cellEdge >= eEnd) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Non-standard closure: edges");
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
  fEnd = fEndInterior < 0 ? fEnd : fEndInterior;
  ierr = PetscMalloc1((cEnd-cStart) * P4EST_FACES,&ttf);CHKERRQ(ierr);
  for (f = fStart; f < fEnd; f++) {
    PetscInt       numSupp, s;
    PetscInt       myFace[2] = {-1, -1};
    PetscInt       myOrnt[2] = {PETSC_MIN_INT, PETSC_MIN_INT};
    const PetscInt *supp;

    ierr = DMPlexGetSupportSize(dm, f, &numSupp);CHKERRQ(ierr);
    if (numSupp != 1 && numSupp != 2) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"point %D has facet with %D sides: must be 1 or 2 (boundary or conformal)",f,numSupp);
    ierr = DMPlexGetSupport(dm, f, &supp);CHKERRQ(ierr);

    for (s = 0; s < numSupp; s++) {
      PetscInt p = supp[s];

      if (p >= cEnd) {
        numSupp--;
        if (s) supp = &supp[1 - s];
        break;
      }
    }
    for (s = 0; s < numSupp; s++) {
      PetscInt       p = supp[s], i;
      PetscInt       numCone;
      const PetscInt *cone;
      const PetscInt *ornt;
      PetscInt       orient = PETSC_MIN_INT;

      ierr = DMPlexGetConeSize(dm, p, &numCone);CHKERRQ(ierr);
      if (numCone != P4EST_FACES) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"cell %D has %D facets, expect %d",p,numCone,P4EST_FACES);
      ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
      ierr = DMPlexGetConeOrientation(dm, p, &ornt);CHKERRQ(ierr);
      for (i = 0; i < P4EST_FACES; i++) {
        if (cone[i] == f) {
          orient = ornt[i];
          break;
        }
      }
      if (i >= P4EST_FACES) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"cell %D faced %D mismatch",p,f);
      ttf[P4EST_FACES * (p - cStart) + PetscFaceToP4estFace[i]] = f - fStart;
      if (numSupp == 1) {
        /* boundary faces indicated by self reference */
        conn->tree_to_tree[P4EST_FACES * (p - cStart) + PetscFaceToP4estFace[i]] = p - cStart;
        conn->tree_to_face[P4EST_FACES * (p - cStart) + PetscFaceToP4estFace[i]] = (int8_t) PetscFaceToP4estFace[i];
      } else {
        const PetscInt N = P4EST_CHILDREN / 2;

        conn->tree_to_tree[P4EST_FACES * (p - cStart) + PetscFaceToP4estFace[i]] = supp[1 - s] - cStart;
        myFace[s] = PetscFaceToP4estFace[i];
        /* get the orientation of cell p in p4est-type closure to facet f, by composing the p4est-closure to
         * petsc-closure permutation and the petsc-closure to facet orientation */
        myOrnt[s] = DihedralCompose(N,orient,P4estFaceToPetscOrnt[myFace[s]]);
      }
    }
    if (numSupp == 2) {
      for (s = 0; s < numSupp; s++) {
        PetscInt       p = supp[s];
        PetscInt       orntAtoB;
        PetscInt       p4estOrient;
        const PetscInt N = P4EST_CHILDREN / 2;

        /* composing the forward permutation with the other cell's inverse permutation gives the self-to-neighbor
         * permutation of this cell-facet's cone */
        orntAtoB = DihedralCompose(N,DihedralInvert(N,myOrnt[1-s]),myOrnt[s]);

        /* convert cone-description permutation (i.e., edges around facet) to cap-description permutation (i.e.,
         * vertices around facet) */
#if !defined(P4_TO_P8)
        p4estOrient = orntAtoB < 0 ? -(orntAtoB + 1) : orntAtoB;
#else
        {
          PetscInt firstVert      = orntAtoB < 0 ? ((-orntAtoB) % N) : orntAtoB;
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

    ierr                         = PetscSectionGetOffset(ett,e,&off);CHKERRQ(ierr);
    conn->ett_offset[e - eStart] = (p4est_topidx_t) off;
    ierr                         = DMPlexGetTransitiveClosure(dm,e,PETSC_FALSE,&starSize,&star);CHKERRQ(ierr);
    for (s = 0; s < starSize; s++) {
      PetscInt p = star[2 * s];

      if (p >= cStart && p < cEnd) {
        ierr = DMPlexGetTransitiveClosure(dm,p,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
        if (closureSize != P4EST_INSUL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Non-standard closure");
        for (c = 0; c < P8EST_EDGES; c++) {
          PetscInt cellEdge = closure[2 * (c + edgeOff)];
          PetscInt cellOrnt = closure[2 * (c + edgeOff) + 1];

          if (cellEdge == e) {
            PetscInt p4estEdge = PetscEdgeToP4estEdge[c];
            PetscInt totalOrient;

            /* compose p4est-closure to petsc-closure permutation and petsc-closure to edge orientation */
            totalOrient = DihedralCompose(2,cellOrnt,P4estEdgeToPetscOrnt[p4estEdge]);
            /* p4est orientations are positive: -2 => 1, -1 => 0 */
            totalOrient             = (totalOrient < 0) ? -(totalOrient + 1) : totalOrient;
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

    ierr                         = PetscSectionGetOffset(ctt,v,&off);CHKERRQ(ierr);
    conn->ctt_offset[v - vStart] = (p4est_topidx_t) off;
    ierr                         = DMPlexGetTransitiveClosure(dm,v,PETSC_FALSE,&starSize,&star);CHKERRQ(ierr);
    for (s = 0; s < starSize; s++) {
      PetscInt p = star[2 * s];

      if (p >= cStart && p < cEnd) {
        ierr = DMPlexGetTransitiveClosure(dm,p,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
        if (closureSize != P4EST_INSUL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Non-standard closure");
        for (c = 0; c < P4EST_CHILDREN; c++) {
          PetscInt cellVert = closure[2 * (c + vertOff)];

          if (cellVert == v) {
            PetscInt p4estVert = PetscVertToP4estVert[c];

            conn->corner_to_tree[off]     = (p4est_locidx_t) (p - cStart);
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
    PetscBool    localized;

    ierr = DMGetCoordinateDim(dm, &coordDim);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordVec);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocalizedLocal(dm, &localized);CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(dm, &coordSec);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; c++) {
      PetscInt    dof;
      PetscScalar *cellCoords = NULL;

      ierr = DMPlexVecGetClosure(dm, coordSec, coordVec, c, &dof, &cellCoords);CHKERRQ(ierr);
      if (!localized && dof != P4EST_CHILDREN * coordDim) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Need coordinates at the corners: (dof) %D != %D * %D (sdim)", dof, P4EST_CHILDREN, coordDim);
      for (v = 0; v < P4EST_CHILDREN; v++) {
        PetscInt i, lim = PetscMin(3, coordDim);
        PetscInt p4estVert = PetscVertToP4estVert[v];

        conn->tree_to_vertex[P4EST_CHILDREN * (c - cStart) + v] = P4EST_CHILDREN * (c - cStart) + v;
        /* p4est vertices are always embedded in R^3 */
        for (i = 0; i < 3; i++)   conn->vertices[3 * (P4EST_CHILDREN * (c - cStart) + p4estVert) + i] = 0.;
        for (i = 0; i < lim; i++) conn->vertices[3 * (P4EST_CHILDREN * (c - cStart) + p4estVert) + i] = PetscRealPart(cellCoords[v * coordDim + i]);
      }
      ierr = DMPlexVecRestoreClosure(dm, coordSec, coordVec, c, &dof, &cellCoords);CHKERRQ(ierr);
    }
  }

#if defined(P4EST_ENABLE_DEBUG)
  if (!p4est_connectivity_is_valid(conn)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Plex to p4est conversion failed");
#endif

  *connOut = conn;

  *tree_face_to_uniq = ttf;

  PetscFunctionReturn(0);
}

static PetscErrorCode locidx_to_PetscInt(sc_array_t * array)
{
  sc_array_t *newarray;
  size_t     zz, count = array->elem_count;

  PetscFunctionBegin;
  if (array->elem_size != sizeof(p4est_locidx_t)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong locidx size");

  if (sizeof(p4est_locidx_t) == sizeof(PetscInt)) PetscFunctionReturn(0);

  newarray = sc_array_new_size (sizeof(PetscInt), array->elem_count);
  for (zz = 0; zz < count; zz++) {
    p4est_locidx_t il  = *((p4est_locidx_t*) sc_array_index (array, zz));
    PetscInt       *ip = (PetscInt*) sc_array_index (newarray, zz);

    *ip = (PetscInt) il;
  }

  sc_array_reset (array);
  sc_array_init_size (array, sizeof(PetscInt), count);
  sc_array_copy (array, newarray);
  sc_array_destroy (newarray);
  PetscFunctionReturn(0);
}

static PetscErrorCode coords_double_to_PetscScalar(sc_array_t * array, PetscInt dim)
{
  sc_array_t *newarray;
  size_t     zz, count = array->elem_count;

  PetscFunctionBegin;
  if (array->elem_size != 3 * sizeof(double)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong coordinate size");
#if !defined(PETSC_USE_COMPLEX)
  if (sizeof(double) == sizeof(PetscScalar) && dim == 3) PetscFunctionReturn(0);
#endif

  newarray = sc_array_new_size (dim * sizeof(PetscScalar), array->elem_count);
  for (zz = 0; zz < count; zz++) {
    int         i;
    double      *id = (double*) sc_array_index (array, zz);
    PetscScalar *ip = (PetscScalar*) sc_array_index (newarray, zz);

    for (i = 0; i < dim; i++) ip[i] = 0.;
    for (i = 0; i < PetscMin(dim,3); i++) ip[i] = (PetscScalar) id[i];
  }

  sc_array_reset (array);
  sc_array_init_size (array, dim * sizeof(PetscScalar), count);
  sc_array_copy (array, newarray);
  sc_array_destroy (newarray);
  PetscFunctionReturn(0);
}

static PetscErrorCode locidx_pair_to_PetscSFNode(sc_array_t * array)
{
  sc_array_t *newarray;
  size_t     zz, count = array->elem_count;

  PetscFunctionBegin;
  if (array->elem_size != 2 * sizeof(p4est_locidx_t)) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong locidx size");

  newarray = sc_array_new_size (sizeof(PetscSFNode), array->elem_count);
  for (zz = 0; zz < count; zz++) {
    p4est_locidx_t *il = (p4est_locidx_t*) sc_array_index (array, zz);
    PetscSFNode    *ip = (PetscSFNode*) sc_array_index (newarray, zz);

    ip->rank  = (PetscInt) il[0];
    ip->index = (PetscInt) il[1];
  }

  sc_array_reset (array);
  sc_array_init_size (array, sizeof(PetscSFNode), count);
  sc_array_copy (array, newarray);
  sc_array_destroy (newarray);
  PetscFunctionReturn(0);
}

static PetscErrorCode P4estToPlex_Local(p4est_t *p4est, DM * plex)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  {
    sc_array_t     *points_per_dim    = sc_array_new(sizeof(p4est_locidx_t));
    sc_array_t     *cone_sizes        = sc_array_new(sizeof(p4est_locidx_t));
    sc_array_t     *cones             = sc_array_new(sizeof(p4est_locidx_t));
    sc_array_t     *cone_orientations = sc_array_new(sizeof(p4est_locidx_t));
    sc_array_t     *coords            = sc_array_new(3 * sizeof(double));
    sc_array_t     *children          = sc_array_new(sizeof(p4est_locidx_t));
    sc_array_t     *parents           = sc_array_new(sizeof(p4est_locidx_t));
    sc_array_t     *childids          = sc_array_new(sizeof(p4est_locidx_t));
    sc_array_t     *leaves            = sc_array_new(sizeof(p4est_locidx_t));
    sc_array_t     *remotes           = sc_array_new(2 * sizeof(p4est_locidx_t));
    p4est_locidx_t first_local_quad;

    PetscStackCallP4est(p4est_get_plex_data,(p4est,P4EST_CONNECT_FULL,0,&first_local_quad,points_per_dim,cone_sizes,cones,cone_orientations,coords,children,parents,childids,leaves,remotes));

    ierr = locidx_to_PetscInt(points_per_dim);CHKERRQ(ierr);
    ierr = locidx_to_PetscInt(cone_sizes);CHKERRQ(ierr);
    ierr = locidx_to_PetscInt(cones);CHKERRQ(ierr);
    ierr = locidx_to_PetscInt(cone_orientations);CHKERRQ(ierr);
    ierr = coords_double_to_PetscScalar(coords, P4EST_DIM);CHKERRQ(ierr);

    ierr = DMPlexCreate(PETSC_COMM_SELF,plex);CHKERRQ(ierr);
    ierr = DMSetDimension(*plex,P4EST_DIM);CHKERRQ(ierr);
    ierr = DMPlexCreateFromDAG(*plex,P4EST_DIM,(PetscInt*)points_per_dim->array,(PetscInt*)cone_sizes->array,(PetscInt*)cones->array,(PetscInt*)cone_orientations->array,(PetscScalar*)coords->array);CHKERRQ(ierr);
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
  }
  PetscFunctionReturn(0);
}

#define DMReferenceTreeGetChildSymmetry_pforest _append_pforest(DMReferenceTreeGetChildSymmetry)
static PetscErrorCode DMReferenceTreeGetChildSymmetry_pforest(DM dm, PetscInt parent, PetscInt parentOrientA, PetscInt childOrientA, PetscInt childA, PetscInt parentOrientB, PetscInt *childOrientB,PetscInt *childB)
{
  PetscInt       coneSize, dStart, dEnd, vStart, vEnd, dim, ABswap, oAvert, oBvert, ABswapVert;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (parentOrientA == parentOrientB) {
    if (childOrientB) *childOrientB = childOrientA;
    if (childB) *childB = childA;
    PetscFunctionReturn(0);
  }
  ierr = DMPlexGetDepthStratum(dm,0,&vStart,&vEnd);CHKERRQ(ierr);
  if (childA >= vStart && childA < vEnd) { /* vertices (always in the middle) are invarient under rotation */
    if (childOrientB) *childOrientB = 0;
    if (childB) *childB = childA;
    PetscFunctionReturn(0);
  }
  for (dim = 0; dim < 3; dim++) {
    ierr = DMPlexGetDepthStratum(dm,dim,&dStart,&dEnd);CHKERRQ(ierr);
    if (parent >= dStart && parent <= dEnd) break;
  }
  if (dim > 2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot perform child symmetry for %d-cells",dim);
  if (!dim) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"A vertex has no children");
  if (childA < dStart || childA >= dEnd) { /* a 1-cell in a 2-cell */
    /* this is a lower-dimensional child: bootstrap */
    PetscInt       size, i, sA = -1, sB, sOrientB, sConeSize;
    const PetscInt *supp, *coneA, *coneB, *oA, *oB;

    ierr = DMPlexGetSupportSize(dm,childA,&size);CHKERRQ(ierr);
    ierr = DMPlexGetSupport(dm,childA,&supp);CHKERRQ(ierr);

    /* find a point sA in supp(childA) that has the same parent */
    for (i = 0; i < size; i++) {
      PetscInt sParent;

      sA = supp[i];
      if (sA == parent) continue;
      ierr = DMPlexGetTreeParent(dm,sA,&sParent,NULL);CHKERRQ(ierr);
      if (sParent == parent) break;
    }
    if (i == size) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"could not find support in children");
    /* find out which point sB is in an equivalent position to sA under
     * parentOrientB */
    ierr = DMReferenceTreeGetChildSymmetry_pforest(dm,parent,parentOrientA,0,sA,parentOrientB,&sOrientB,&sB);CHKERRQ(ierr);
    ierr = DMPlexGetConeSize(dm,sA,&sConeSize);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm,sA,&coneA);CHKERRQ(ierr);
    ierr = DMPlexGetCone(dm,sB,&coneB);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dm,sA,&oA);CHKERRQ(ierr);
    ierr = DMPlexGetConeOrientation(dm,sB,&oB);CHKERRQ(ierr);
    /* step through the cone of sA in natural order */
    for (i = 0; i < sConeSize; i++) {
      if (coneA[i] == childA) {
        /* if childA is at position i in coneA,
         * then we want the point that is at sOrientB*i in coneB */
        PetscInt j = (sOrientB >= 0) ? ((sOrientB + i) % sConeSize) : ((sConeSize -(sOrientB+1) - i) % sConeSize);
        if (childB) *childB = coneB[j];
        if (childOrientB) {
          PetscInt oBtrue;

          ierr = DMPlexGetConeSize(dm,childA,&coneSize);CHKERRQ(ierr);
          /* compose sOrientB and oB[j] */
          if (coneSize != 0 && coneSize != 2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Expected a vertex or an edge");
          /* we may have to flip an edge */
          oBtrue        = coneSize ? ((sOrientB >= 0) ? oB[j] : -(oB[j] + 2)) : 0;
          ABswap        = DihedralSwap(coneSize,oA[i],oBtrue);
          *childOrientB = DihedralCompose(coneSize,childOrientA,ABswap);
        }
        break;
      }
    }
    if (i == sConeSize) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"support cone mismatch");
    PetscFunctionReturn(0);
  }
  /* get the cone size and symmetry swap */
  ierr   = DMPlexGetConeSize(dm,parent,&coneSize);CHKERRQ(ierr);
  ABswap = DihedralSwap(coneSize, parentOrientA, parentOrientB);
  if (dim == 2) {
    /* orientations refer to cones: we want them to refer to vertices:
     * if it's a rotation, they are the same, but if the order is reversed, a
     * permutation that puts side i first does *not* put vertex i first */
    oAvert     = (parentOrientA >= 0) ? parentOrientA : -((-parentOrientA % coneSize) + 1);
    oBvert     = (parentOrientB >= 0) ? parentOrientB : -((-parentOrientB % coneSize) + 1);
    ABswapVert = DihedralSwap(coneSize, oAvert, oBvert);
  } else {
    oAvert     = parentOrientA;
    oBvert     = parentOrientB;
    ABswapVert = ABswap;
  }
  if (childB) {
    /* assume that each child corresponds to a vertex, in the same order */
    PetscInt       p, posA = -1, numChildren, i;
    const PetscInt *children;

    /* count which position the child is in */
    ierr = DMPlexGetTreeChildren(dm,parent,&numChildren,&children);CHKERRQ(ierr);
    for (i = 0; i < numChildren; i++) {
      p = children[i];
      if (p == childA) {
        if (dim == 1) {
          posA = i;
        } else { /* 2D Morton to rotation */
          posA = (i & 2) ? (i ^ 1) : i;
        }
        break;
      }
    }
    if (posA >= coneSize) {
      SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Could not find childA in children of parent");
    } else {
      /* figure out position B by applying ABswapVert */
      PetscInt posB, childIdB;

      posB = (ABswapVert >= 0) ? ((ABswapVert + posA) % coneSize) : ((coneSize -(ABswapVert + 1) - posA) % coneSize);
      if (dim == 1) {
        childIdB = posB;
      } else { /* 2D rotation to Morton */
        childIdB = (posB & 2) ? (posB ^ 1) : posB;
      }
      if (childB) *childB = children[childIdB];
    }
  }
  if (childOrientB) *childOrientB = DihedralCompose(coneSize,childOrientA,ABswap);
  PetscFunctionReturn(0);
}

#define DMCreateReferenceTree_pforest _append_pforest(DMCreateReferenceTree)
static PetscErrorCode DMCreateReferenceTree_pforest(MPI_Comm comm, DM *dm)
{
  p4est_connectivity_t *refcube;
  p4est_t              *root, *refined;
  DM                   dmRoot, dmRefined;
  DM_Plex              *mesh;
  PetscMPIInt          rank;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  PetscStackCallP4estReturn(refcube,p4est_connectivity_new_byname,("unit"));
  { /* [-1,1]^d geometry */
    PetscInt i, j;

    for(i = 0; i < P4EST_CHILDREN; i++) {
      for (j = 0; j < 3; j++) {
        refcube->vertices[3 * i + j] *= 2.;
        refcube->vertices[3 * i + j] -= 1.;
      }
    }
  }
  PetscStackCallP4estReturn(root,p4est_new,(PETSC_COMM_SELF,refcube,0,NULL,NULL));
  PetscStackCallP4estReturn(refined,p4est_new_ext,(PETSC_COMM_SELF,refcube,0,1,1,0,NULL,NULL));
  ierr = P4estToPlex_Local(root,&dmRoot);CHKERRQ(ierr);
  ierr = P4estToPlex_Local(refined,&dmRefined);CHKERRQ(ierr);
  {
#if !defined(P4_TO_P8)
    PetscInt nPoints  = 25;
    PetscInt perm[25] = {0, 1, 2, 3,
                          4, 12, 8, 14,
                              6, 9, 15,
                          5, 13,    10,
                              7,    11,
                         16, 22, 20, 24,
                             17,     21,
                                 18, 23,
                                     19};
    PetscInt ident[25] = {0, 0, 0, 0,
                          1, 1, 2, 2, 3, 3, 4, 4, 0, 0, 0, 0,
                          5, 6, 7, 8, 1, 2, 3, 4, 0};
#else
    PetscInt nPoints   = 125;
    PetscInt perm[125] = {0, 1, 2, 3, 4, 5, 6, 7,
                           8, 32, 16, 36, 24, 40,
                              12, 17, 37, 25, 41,
                           9, 33,     20, 26, 42,
                              13,     21, 27, 43,
                          10, 34, 18, 38,     28,
                              14, 19, 39,     29,
                          11, 35,     22,     30,
                              15,     23,     31,
                          44, 84, 76, 92, 52, 86, 68, 94, 60, 78, 70, 96,
                          45, 85, 77, 93,     54,     72,     62,     74,
                              46,     80, 53, 87, 69, 95,         64, 82,
                              47,     81,     55,     73,             66,
                                  48, 88,         56, 90, 61, 79, 71, 97,
                                  49, 89,             58,     63,     75,
                                      50,         57, 91,         65, 83,
                                      51,             59,             67,
                           98, 106, 110, 122, 114, 120, 118, 124,
                                99,      111,      115,      119,
                                    100, 107,           116, 121,
                                         101,                117,
                                              102, 108, 112, 123,
                                                   103,      113,
                                                        104, 109,
                                                             105};
    PetscInt ident[125] = {0, 0, 0, 0, 0, 0, 0, 0,
                           1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6,
                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 18, 18,
                           1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6,
                           0, 0, 0, 0, 0, 0,
                           19, 20, 21, 22, 23, 24, 25, 26,
                           7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                           1, 2, 3, 4, 5, 6,
                           0};

#endif
    IS permIS;
    DM dmPerm;

    ierr = ISCreateGeneral(PETSC_COMM_SELF,nPoints,perm,PETSC_USE_POINTER,&permIS);CHKERRQ(ierr);
    ierr = DMPlexPermute(dmRefined,permIS,&dmPerm);CHKERRQ(ierr);
    if (dmPerm) {
      ierr      = DMDestroy(&dmRefined);CHKERRQ(ierr);
      dmRefined = dmPerm;
    }
    ierr = ISDestroy(&permIS);CHKERRQ(ierr);
    {
      PetscInt p;
      ierr = DMCreateLabel(dmRoot,"identity");CHKERRQ(ierr);
      ierr = DMCreateLabel(dmRefined,"identity");CHKERRQ(ierr);
      for (p = 0; p < P4EST_INSUL; p++) {
        ierr = DMSetLabelValue(dmRoot,"identity",p,p);CHKERRQ(ierr);
      }
      for (p = 0; p < nPoints; p++) {
        ierr = DMSetLabelValue(dmRefined,"identity",p,ident[p]);CHKERRQ(ierr);
      }
    }
  }
  ierr                   = DMPlexCreateReferenceTree_Union(dmRoot,dmRefined,"identity",dm);CHKERRQ(ierr);
  mesh                   = (DM_Plex*) (*dm)->data;
  mesh->getchildsymmetry = DMReferenceTreeGetChildSymmetry_pforest;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = DMViewFromOptions(dmRoot,   NULL,"-dm_p4est_ref_root_view");CHKERRQ(ierr);
    ierr = DMViewFromOptions(dmRefined,NULL,"-dm_p4est_ref_refined_view");CHKERRQ(ierr);
    ierr = DMViewFromOptions(dmRefined,NULL,"-dm_p4est_ref_tree_view");CHKERRQ(ierr);
  }
  ierr                   = DMDestroy(&dmRefined);CHKERRQ(ierr);
  ierr                   = DMDestroy(&dmRoot);CHKERRQ(ierr);
  PetscStackCallP4est(p4est_destroy,(refined));
  PetscStackCallP4est(p4est_destroy,(root));
  PetscStackCallP4est(p4est_connectivity_destroy,(refcube));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMShareDiscretization(DM dmA, DM dmB)
{
  PetscDS        ds, dsB;
  PetscBool      newDS;
  void           *ctx;
  PetscInt       num;
  PetscReal      val;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr  = DMGetApplicationContext(dmA,&ctx);CHKERRQ(ierr);
  ierr  = DMSetApplicationContext(dmB,ctx);CHKERRQ(ierr);
  ierr  = DMGetDS(dmA,&ds);CHKERRQ(ierr);
  ierr  = DMGetDS(dmB,&dsB);CHKERRQ(ierr);
  newDS = (PetscBool) (ds != dsB);
  ierr  = DMCopyDisc(dmA,dmB);CHKERRQ(ierr);
  ierr  = DMGetOutputSequenceNumber(dmA,&num,&val);CHKERRQ(ierr);
  ierr  = DMSetOutputSequenceNumber(dmB,num,val);CHKERRQ(ierr);
  if (newDS) {
    ierr = DMClearGlobalVectors(dmB);CHKERRQ(ierr);
    ierr = DMClearLocalVectors(dmB);CHKERRQ(ierr);
    ierr = PetscObjectReference((PetscObject)dmA->localSection);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&(dmB->localSection));CHKERRQ(ierr);
    dmB->localSection = dmA->localSection;
    ierr = PetscObjectReference((PetscObject)dmA->defaultConstraintSection);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&(dmB->defaultConstraintSection));CHKERRQ(ierr);
    dmB->defaultConstraintSection = dmA->defaultConstraintSection;
    ierr = PetscObjectReference((PetscObject)dmA->defaultConstraintMat);CHKERRQ(ierr);
    ierr = MatDestroy(&(dmB->defaultConstraintMat));CHKERRQ(ierr);
    dmB->defaultConstraintMat = dmA->defaultConstraintMat;
    ierr = PetscObjectReference((PetscObject)dmA->globalSection);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&(dmB->globalSection));CHKERRQ(ierr);
    dmB->globalSection = dmA->globalSection;
    ierr = PetscObjectReference((PetscObject)dmA->sectionSF);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&dmB->sectionSF);CHKERRQ(ierr);
    dmB->sectionSF = dmA->sectionSF;
    if (dmA->map) {ierr = PetscLayoutReference(dmA->map,&dmB->map);CHKERRQ(ierr);}
  }
  PetscFunctionReturn(0);
}

/* Get an SF that broadcasts a coarse-cell covering of the local fine cells */
static PetscErrorCode DMPforestGetCellCoveringSF(MPI_Comm comm,p4est_t *p4estC, p4est_t *p4estF, PetscInt cStart, PetscInt cEnd, PetscSF *coveringSF)
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
      recv[2*(p-startC)]   = 0;
      recv[2*(p-startC)+1] = 0;
      continue;
    }

    ierr = MPI_Irecv(&recv[2*(p-startC)],2,MPIU_INT,p,tag,comm,&recvReqs[p-startC]);CHKERRQ(ierr);
  }
  ierr = DMPforestComputeOverlappingRanks(p4estC->mpisize,p4estC->mpirank,p4estC,p4estF,&startF,&endF);CHKERRQ(ierr);
  ierr = PetscMalloc2(2*(endF-startF),&send,endF-startF,&sendReqs);CHKERRQ(ierr);
  /* count the quadrants rank will send to each of [startF,endF) */
  for (p = startF; p < endF; p++) {
    p4est_quadrant_t *myFineStart = &p4estF->global_first_position[p];
    p4est_quadrant_t *myFineEnd   = &p4estF->global_first_position[p+1];
    PetscInt         tStart       = (PetscInt) myFineStart->p.which_tree;
    PetscInt         tEnd         = (PetscInt) myFineEnd->p.which_tree;
    PetscInt         firstCell    = -1, lastCell = -1;
    p4est_tree_t     *treeStart   = &(((p4est_tree_t*) p4estC->trees->array)[tStart]);
    p4est_tree_t     *treeEnd     = (size_t) tEnd < p4estC->trees->elem_count ? &(((p4est_tree_t*) p4estC->trees->array)[tEnd]) : NULL;
    ssize_t          overlapIndex;

    sendReqs[p-startF] = MPI_REQUEST_NULL; /* just in case we don't initiate a send */
    if (p4estF->global_first_quadrant[p] == p4estF->global_first_quadrant[p+1]) continue;

    /* locate myFineStart in (or before) a cell */
    if (treeStart->quadrants.elem_count) {
      PetscStackCallP4estReturn(overlapIndex,sc_array_bsearch,(&(treeStart->quadrants),myFineStart,p4est_quadrant_disjoint));
      if (overlapIndex < 0) {
        firstCell = 0;
      } else {
        firstCell = treeStart->quadrants_offset + overlapIndex;
      }
    } else {
      firstCell = 0;
    }
    if (treeEnd && treeEnd->quadrants.elem_count) {
      PetscStackCallP4estReturn(overlapIndex,sc_array_bsearch,(&(treeEnd->quadrants),myFineEnd,p4est_quadrant_disjoint));
      if (overlapIndex < 0) { /* all of this local section is overlapped */
        lastCell = p4estC->local_num_quadrants;
      } else {
        p4est_quadrant_t *container = &(((p4est_quadrant_t*) treeEnd->quadrants.array)[overlapIndex]);
        p4est_quadrant_t first_desc;
        int              equal;

        PetscStackCallP4est(p4est_quadrant_first_descendant,(container,&first_desc,P4EST_QMAXLEVEL));
        PetscStackCallP4estReturn(equal,p4est_quadrant_is_equal,(myFineEnd,&first_desc));
        if (equal) {
          lastCell = treeEnd->quadrants_offset + overlapIndex;
        } else {
          lastCell = treeEnd->quadrants_offset + overlapIndex + 1;
        }
      }
    } else {
      lastCell = p4estC->local_num_quadrants;
    }
    send[2*(p-startF)]   = firstCell;
    send[2*(p-startF)+1] = lastCell - firstCell;
    ierr                 = MPI_Isend(&send[2*(p-startF)],2,MPIU_INT,p,tag,comm,&sendReqs[p-startF]);CHKERRQ(ierr);
  }
  ierr = MPI_Waitall((PetscMPIInt)(endC-startC),recvReqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
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
    PetscInt numCells  = recv[2*(p-startC)+1];
    PetscInt off, i;

    ierr = PetscSectionGetOffset(section,p,&off);CHKERRQ(ierr);
    for (i = 0; i < numCells; i++) {
      leaves[off+i].rank  = p;
      leaves[off+i].index = firstCell + i;
    }
  }
  ierr        = PetscSFCreate(comm,&sf);CHKERRQ(ierr);
  ierr        = PetscSFSetGraph(sf,cEnd-cStart,nLeaves,NULL,PETSC_OWN_POINTER,leaves,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr        = PetscSectionDestroy(&section);CHKERRQ(ierr);
  ierr        = MPI_Waitall((PetscMPIInt)(endF-startF),sendReqs,MPI_STATUSES_IGNORE);CHKERRQ(ierr);
  ierr        = PetscFree2(send,sendReqs);CHKERRQ(ierr);
  ierr        = PetscFree2(recv,recvReqs);CHKERRQ(ierr);
  *coveringSF = sf;
  PetscFunctionReturn(0);
}

/* closure points for locally-owned cells */
static PetscErrorCode DMPforestGetCellSFNodes(DM dm, PetscInt numClosureIndices, PetscInt *numClosurePoints, PetscSFNode **closurePoints,PetscBool redirect)
{
  PetscInt          cStart, cEnd;
  PetscInt          count, c;
  PetscMPIInt       rank;
  PetscInt          closureSize = -1;
  PetscInt          *closure    = NULL;
  PetscSF           pointSF;
  PetscInt          nleaves, nroots;
  const PetscInt    *ilocal;
  const PetscSFNode *iremote;
  DM                plex;
  DM_Forest         *forest;
  DM_Forest_pforest *pforest;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  forest            = (DM_Forest *) dm->data;
  pforest           = (DM_Forest_pforest *) forest->data;
  cStart            = pforest->cLocalStart;
  cEnd              = pforest->cLocalEnd;
  ierr              = DMPforestGetPlex(dm,&plex);CHKERRQ(ierr);
  ierr              = DMGetPointSF(dm,&pointSF);CHKERRQ(ierr);
  ierr              = PetscSFGetGraph(pointSF,&nroots,&nleaves,&ilocal,&iremote);CHKERRQ(ierr);
  nleaves           = PetscMax(0,nleaves);
  nroots            = PetscMax(0,nroots);
  *numClosurePoints = numClosureIndices * (cEnd - cStart);
  ierr              = PetscMalloc1(*numClosurePoints,closurePoints);
  ierr              = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRQ(ierr);
  for (c = cStart, count = 0; c < cEnd; c++) {
    PetscInt i;
    ierr = DMPlexGetTransitiveClosure(plex,c,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);

    for (i = 0; i < numClosureIndices; i++, count++) {
      PetscInt p   = closure[2 * i];
      PetscInt loc = -1;

      ierr = PetscFindInt(p,nleaves,ilocal,&loc);CHKERRQ(ierr);
      if (redirect && loc >= 0) {
        (*closurePoints)[count].rank  = iremote[loc].rank;
        (*closurePoints)[count].index = iremote[loc].index;
      } else {
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
    PetscSFNode *A = (PetscSFNode*)a;
    PetscSFNode *B = (PetscSFNode*)b;

    if (B->rank < 0) *B = *A;
  }
}

static PetscErrorCode DMPforestGetTransferSF_Point(DM coarse, DM fine, PetscSF *sf, PetscBool transferIdent, PetscInt *childIds[])
{
  MPI_Comm          comm;
  PetscMPIInt       rank, size;
  DM_Forest_pforest *pforestC, *pforestF;
  p4est_t           *p4estC, *p4estF;
  PetscInt          numClosureIndices;
  PetscInt          numClosurePointsC, numClosurePointsF;
  PetscSFNode       *closurePointsC, *closurePointsF;
  p4est_quadrant_t  *coverQuads = NULL;
  p4est_quadrant_t  **treeQuads;
  PetscInt          *treeQuadCounts;
  MPI_Datatype      nodeType;
  MPI_Datatype      nodeClosureType;
  MPI_Op            sfNodeReduce;
  p4est_topidx_t    fltF, lltF, t;
  DM                plexC, plexF;
  PetscInt          pStartF, pEndF, pStartC, pEndC;
  PetscBool         saveInCoarse = PETSC_FALSE;
  PetscBool         saveInFine   = PETSC_FALSE;
  PetscBool         formCids     = (childIds != NULL) ? PETSC_TRUE : PETSC_FALSE;
  PetscInt          *cids        = NULL;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  pforestC = (DM_Forest_pforest*) ((DM_Forest*) coarse->data)->data;
  pforestF = (DM_Forest_pforest*) ((DM_Forest*) fine->data)->data;
  p4estC   = pforestC->forest;
  p4estF   = pforestF->forest;
  if (pforestC->topo != pforestF->topo) SETERRQ(PetscObjectComm((PetscObject)coarse),PETSC_ERR_ARG_INCOMP,"DM's must have the same base DM");
  comm = PetscObjectComm((PetscObject)coarse);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = DMPforestGetPlex(fine,&plexF);CHKERRQ(ierr);
  ierr = DMPlexGetChart(plexF,&pStartF,&pEndF);CHKERRQ(ierr);
  ierr = DMPforestGetPlex(coarse,&plexC);CHKERRQ(ierr);
  ierr = DMPlexGetChart(plexC,&pStartC,&pEndC);CHKERRQ(ierr);
  { /* check if the results have been cached */
    DM adaptCoarse, adaptFine;

    ierr = DMForestGetAdaptivityForest(coarse,&adaptCoarse);CHKERRQ(ierr);
    ierr = DMForestGetAdaptivityForest(fine,&adaptFine);CHKERRQ(ierr);
    if (adaptCoarse && adaptCoarse->data == fine->data) { /* coarse is adapted from fine */
      if (pforestC->pointSelfToAdaptSF) {
        ierr = PetscObjectReference((PetscObject)(pforestC->pointSelfToAdaptSF));CHKERRQ(ierr);
        *sf  = pforestC->pointSelfToAdaptSF;
        if (childIds) {
          ierr      = PetscMalloc1(pEndF-pStartF,&cids);CHKERRQ(ierr);
          ierr      = PetscArraycpy(cids,pforestC->pointSelfToAdaptCids,pEndF-pStartF);CHKERRQ(ierr);
          *childIds = cids;
        }
        PetscFunctionReturn(0);
      } else {
        saveInCoarse = PETSC_TRUE;
        formCids     = PETSC_TRUE;
      }
    } else if (adaptFine && adaptFine->data == coarse->data) { /* fine is adapted from coarse */
      if (pforestF->pointAdaptToSelfSF) {
        ierr = PetscObjectReference((PetscObject)(pforestF->pointAdaptToSelfSF));CHKERRQ(ierr);
        *sf  = pforestF->pointAdaptToSelfSF;
        if (childIds) {
          ierr      = PetscMalloc1(pEndF-pStartF,&cids);CHKERRQ(ierr);
          ierr      = PetscArraycpy(cids,pforestF->pointAdaptToSelfCids,pEndF-pStartF);CHKERRQ(ierr);
          *childIds = cids;
        }
        PetscFunctionReturn(0);
      } else {
        saveInFine = PETSC_TRUE;
        formCids   = PETSC_TRUE;
      }
    }
  }

  /* count the number of closure points that have dofs and create a list */
  numClosureIndices = P4EST_INSUL;
  /* create the datatype */
  ierr = MPI_Type_contiguous(2,MPIU_INT,&nodeType);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&nodeType);CHKERRQ(ierr);
  ierr = MPI_Op_create(DMPforestMaxSFNode,PETSC_FALSE,&sfNodeReduce);CHKERRQ(ierr);
  ierr = MPI_Type_contiguous(numClosureIndices*2,MPIU_INT,&nodeClosureType);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&nodeClosureType);CHKERRQ(ierr);
  /* everything has to go through cells: for each cell, create a list of the sfnodes in its closure */
  /* get lists of closure point SF nodes for every cell */
  ierr = DMPforestGetCellSFNodes(coarse,numClosureIndices,&numClosurePointsC,&closurePointsC,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMPforestGetCellSFNodes(fine  ,numClosureIndices,&numClosurePointsF,&closurePointsF,PETSC_FALSE);CHKERRQ(ierr);
  /* create pointers for tree lists */
  fltF = p4estF->first_local_tree;
  lltF = p4estF->last_local_tree;
  ierr = PetscCalloc2(lltF + 1  - fltF, &treeQuads, lltF + 1 - fltF, &treeQuadCounts);CHKERRQ(ierr);
  /* if the partitions don't match, ship the coarse to cover the fine */
  if (size > 1) {
    PetscInt p;

    for (p = 0; p < size; p++) {
      int equal;

      PetscStackCallP4estReturn(equal,p4est_quadrant_is_equal_piggy,(&p4estC->global_first_position[p],&p4estF->global_first_position[p]));
      if (!equal) break;
    }
    if (p < size) { /* non-matching distribution: send the coarse to cover the fine */
      PetscInt         cStartC, cEndC, cEndCInterior;
      PetscSF          coveringSF;
      PetscInt         nleaves;
      PetscInt         count;
      PetscSFNode      *newClosurePointsC;
      p4est_quadrant_t *coverQuadsSend;
      p4est_topidx_t   fltC = p4estC->first_local_tree;
      p4est_topidx_t   lltC = p4estC->last_local_tree;
      p4est_topidx_t   t;
      PetscMPIInt      blockSizes[4]   = {P4EST_DIM,2,1,1};
      MPI_Aint         blockOffsets[4] = {offsetof(p4est_quadrant_t,x),
                                          offsetof(p4est_quadrant_t,level),
                                          offsetof(p4est_quadrant_t,pad16),
                                          offsetof(p4est_quadrant_t,p)};
      MPI_Datatype     blockTypes[4] = {MPI_INT32_T,MPI_INT8_T,MPI_INT16_T,MPI_INT32_T/* p.which_tree */};
      MPI_Datatype     quadStruct,quadType;

      ierr  = DMPlexGetHeightStratum(plexC,0,&cStartC,&cEndC);CHKERRQ(ierr);
      ierr  = DMPlexGetHybridBounds(plexC,&cEndCInterior,NULL,NULL,NULL);CHKERRQ(ierr);
      cEndC = (cEndCInterior < 0) ? cEndC : cEndCInterior;
      ierr  = DMPforestGetCellCoveringSF(comm,p4estC,p4estF,pforestC->cLocalStart,pforestC->cLocalEnd,&coveringSF);CHKERRQ(ierr);
      ierr  = PetscSFGetGraph(coveringSF,NULL,&nleaves,NULL,NULL);CHKERRQ(ierr);
      ierr  = PetscMalloc1(numClosureIndices*nleaves,&newClosurePointsC);CHKERRQ(ierr);
      ierr  = PetscMalloc1(nleaves,&coverQuads);CHKERRQ(ierr);
      ierr  = PetscMalloc1(cEndC-cStartC,&coverQuadsSend);CHKERRQ(ierr);
      count = 0;
      for (t = fltC; t <= lltC; t++) { /* unfortunately, we need to pack a send array, since quads are not stored packed in p4est */
        p4est_tree_t *tree = &(((p4est_tree_t*) p4estC->trees->array)[t]);
        PetscInt     q;

        ierr = PetscMemcpy(&coverQuadsSend[count],tree->quadrants.array,tree->quadrants.elem_count * sizeof(p4est_quadrant_t));CHKERRQ(ierr);
        for (q = 0; (size_t) q < tree->quadrants.elem_count; q++) coverQuadsSend[count+q].p.which_tree = t;
        count += tree->quadrants.elem_count;
      }
      /* p is of a union type p4est_quadrant_data, but only the p.which_tree field is active at this time. So, we
         have a simple blockTypes[] to use. Note that quadStruct does not count potential padding in array of
         p4est_quadrant_t. We have to call MPI_Type_create_resized() to change upper-bound of quadStruct.
       */
      ierr           = MPI_Type_create_struct(4,blockSizes,blockOffsets,blockTypes,&quadStruct);CHKERRQ(ierr);
      ierr           = MPI_Type_create_resized(quadStruct,0,sizeof(p4est_quadrant_t),&quadType);CHKERRQ(ierr);
      ierr           = MPI_Type_commit(&quadType);CHKERRQ(ierr);
      ierr           = PetscSFBcastBegin(coveringSF,nodeClosureType,closurePointsC,newClosurePointsC);CHKERRQ(ierr);
      ierr           = PetscSFBcastBegin(coveringSF,quadType,coverQuadsSend,coverQuads);CHKERRQ(ierr);
      ierr           = PetscSFBcastEnd(coveringSF,nodeClosureType,closurePointsC,newClosurePointsC);CHKERRQ(ierr);
      ierr           = PetscSFBcastEnd(coveringSF,quadType,coverQuadsSend,coverQuads);CHKERRQ(ierr);
      ierr           = MPI_Type_free(&quadStruct);CHKERRQ(ierr);
      ierr           = MPI_Type_free(&quadType);CHKERRQ(ierr);
      ierr           = PetscFree(coverQuadsSend);CHKERRQ(ierr);
      ierr           = PetscFree(closurePointsC);CHKERRQ(ierr);
      ierr           = PetscSFDestroy(&coveringSF);CHKERRQ(ierr);
      closurePointsC = newClosurePointsC;

      /* assign tree quads based on locations in coverQuads */
      {
        PetscInt q;
        for (q = 0; q < nleaves; q++) {
          p4est_locidx_t t = coverQuads[q].p.which_tree;
          if (!treeQuadCounts[t-fltF]++) treeQuads[t-fltF] = &coverQuads[q];
        }
      }
    }
  }
  if (!coverQuads) { /* matching partitions: assign tree quads based on locations in p4est native arrays */
    for (t = fltF; t <= lltF; t++) {
      p4est_tree_t *tree = &(((p4est_tree_t*) p4estC->trees->array)[t]);

      treeQuadCounts[t - fltF] = tree->quadrants.elem_count;
      treeQuads[t - fltF]      = (p4est_quadrant_t*) tree->quadrants.array;
    }
  }

  {
    PetscInt    p;
    PetscInt    cLocalStartF;
    PetscSF     pointSF;
    PetscSFNode *roots;
    PetscInt    *rootType;
    DM          refTree = NULL;
    DMLabel     canonical;
    PetscInt    *childClosures[P4EST_CHILDREN] = {NULL};
    PetscInt    *rootClosure                   = NULL;
    PetscInt    coarseOffset;
    PetscInt    numCoarseQuads;

    ierr = PetscMalloc1(pEndF-pStartF,&roots);CHKERRQ(ierr);
    ierr = PetscMalloc1(pEndF-pStartF,&rootType);CHKERRQ(ierr);
    ierr = DMGetPointSF(fine,&pointSF);CHKERRQ(ierr);
    for (p = pStartF; p < pEndF; p++) {
      roots[p-pStartF].rank  = -1;
      roots[p-pStartF].index = -1;
      rootType[p-pStartF]    = -1;
    }
    if (formCids) {
      PetscInt child;

      ierr = PetscMalloc1(pEndF-pStartF,&cids);CHKERRQ(ierr);
      for (p = pStartF; p < pEndF; p++) cids[p - pStartF] = -2;
      ierr = DMPlexGetReferenceTree(plexF,&refTree);CHKERRQ(ierr);
      ierr = DMPlexGetTransitiveClosure(refTree,0,PETSC_TRUE,NULL,&rootClosure);CHKERRQ(ierr);
      for (child = 0; child < P4EST_CHILDREN; child++) { /* get the closures of the child cells in the reference tree */
        ierr = DMPlexGetTransitiveClosure(refTree,child+1,PETSC_TRUE,NULL,&childClosures[child]);CHKERRQ(ierr);
      }
      ierr = DMGetLabel(refTree,"canonical",&canonical);CHKERRQ(ierr);
    }
    cLocalStartF = pforestF->cLocalStart;
    for (t = fltF, coarseOffset = 0, numCoarseQuads = 0; t <= lltF; t++, coarseOffset += numCoarseQuads) {
      p4est_tree_t     *tree        = &(((p4est_tree_t*) p4estF->trees->array)[t]);
      PetscInt         numFineQuads = tree->quadrants.elem_count;
      p4est_quadrant_t *coarseQuads = treeQuads[t - fltF];
      p4est_quadrant_t *fineQuads   = (p4est_quadrant_t*) tree->quadrants.array;
      PetscInt         i, coarseCount = 0;
      PetscInt         offset = tree->quadrants_offset;
      sc_array_t       coarseQuadsArray;

      numCoarseQuads = treeQuadCounts[t - fltF];
      PetscStackCallP4est(sc_array_init_data,(&coarseQuadsArray,coarseQuads,sizeof(p4est_quadrant_t),(size_t) numCoarseQuads));
      for (i = 0; i < numFineQuads; i++) {
        PetscInt         c     = i + offset;
        p4est_quadrant_t *quad = &fineQuads[i];
        p4est_quadrant_t *quadCoarse = NULL;
        ssize_t          disjoint = -1;

        while (disjoint < 0 && coarseCount < numCoarseQuads) {
          quadCoarse = &coarseQuads[coarseCount];
          PetscStackCallP4estReturn(disjoint,p4est_quadrant_disjoint,(quadCoarse,quad));
          if (disjoint < 0) coarseCount++;
        }
        if (disjoint != 0) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"did not find overlapping coarse quad");
        if (quadCoarse->level > quad->level || (quadCoarse->level == quad->level && !transferIdent)) { /* the "coarse" mesh is finer than the fine mesh at the point: continue */
          if (transferIdent) { /* find corners */
            PetscInt j = 0;

            do {
              if (j < P4EST_CHILDREN) {
                p4est_quadrant_t cornerQuad;
                int              equal;

                PetscStackCallP4est(p4est_quadrant_corner_descendant,(quad,&cornerQuad,j,quadCoarse->level));
                PetscStackCallP4estReturn(equal,p4est_quadrant_is_equal,(&cornerQuad,quadCoarse));
                if (equal) {
                  PetscInt    petscJ = P4estVertToPetscVert[j];
                  PetscInt    p      = closurePointsF[numClosureIndices * c + (P4EST_INSUL - P4EST_CHILDREN) + petscJ].index;
                  PetscSFNode q      = closurePointsC[numClosureIndices * (coarseCount + coarseOffset) + (P4EST_INSUL - P4EST_CHILDREN) + petscJ];

                  roots[p-pStartF]    = q;
                  rootType[p-pStartF] = PETSC_MAX_INT;
                  cids[p-pStartF]     = -1;
                  j++;
                }
              }
              coarseCount++;
              disjoint = 1;
              if (coarseCount < numCoarseQuads) {
                quadCoarse = &coarseQuads[coarseCount];
                PetscStackCallP4estReturn(disjoint,p4est_quadrant_disjoint,(quadCoarse,quad));
              }
            } while (!disjoint);
          }
          continue;
        }
        if (quadCoarse->level == quad->level) { /* same quad present in coarse and fine mesh */
          PetscInt j;
          for (j = 0; j < numClosureIndices; j++) {
            PetscInt p = closurePointsF[numClosureIndices * c + j].index;

            roots[p-pStartF]    = closurePointsC[numClosureIndices * (coarseCount + coarseOffset) + j];
            rootType[p-pStartF] = PETSC_MAX_INT; /* unconditionally accept */
            cids[p-pStartF]     = -1;
          }
        } else {
          PetscInt levelDiff = quad->level - quadCoarse->level;
          PetscInt proposedCids[P4EST_INSUL] = {0};

          if (formCids) {
            PetscInt cl;
            PetscInt *pointClosure = NULL;
            int      cid;

            if (levelDiff > 1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_USER,"Recursive child ids not implemented");
            PetscStackCallP4estReturn(cid,p4est_quadrant_child_id,(quad));
            ierr = DMPlexGetTransitiveClosure(plexF,c + cLocalStartF,PETSC_TRUE,NULL,&pointClosure);CHKERRQ(ierr);
            for (cl = 0; cl < P4EST_INSUL; cl++) {
              PetscInt p      = pointClosure[2 * cl];
              PetscInt point  = childClosures[cid][2 * cl];
              PetscInt ornt   = childClosures[cid][2 * cl + 1];
              PetscInt newcid = -1;

              if (rootType[p-pStartF] == PETSC_MAX_INT) continue;
              if (!cl) {
                newcid = cid + 1;
              } else {
                PetscInt rcl, parent, parentOrnt = 0;

                ierr = DMPlexGetTreeParent(refTree,point,&parent,NULL);CHKERRQ(ierr);
                if (parent == point) {
                  newcid = -1;
                } else if (!parent) { /* in the root */
                  newcid = point;
                } else {
                  for (rcl = 1; rcl < P4EST_INSUL; rcl++) {
                    if (rootClosure[2 * rcl] == parent) {
                      parentOrnt = rootClosure[2 * rcl + 1];
                      break;
                    }
                  }
                  if (rcl >= P4EST_INSUL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Couldn't find parent in root closure");
                  ierr = DMPlexReferenceTreeGetChildSymmetry(refTree,parent,parentOrnt,ornt,point,pointClosure[2 * rcl + 1],NULL,&newcid);CHKERRQ(ierr);
                }
              }
              if (newcid >= 0) {

                if (canonical) {
                  ierr = DMLabelGetValue(canonical,newcid,&newcid);CHKERRQ(ierr);
                }
                proposedCids[cl] = newcid;
              }
            }
            ierr = DMPlexRestoreTransitiveClosure(plexF,c + cLocalStartF,PETSC_TRUE,NULL,&pointClosure);CHKERRQ(ierr);
          }
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
          PetscInt       j;
          for (j = 0; j < P4EST_DIM; j++) { /* get the coordinates of cell boundaries in each direction */
            coarseBound[1][j] = coarseBound[0][j] + P4EST_QUADRANT_LEN(quadCoarse->level);
            fineBound[1][j]   = fineBound[0][j]   + P4EST_QUADRANT_LEN(quad->level);
          }
          for (j = 0; j < numClosureIndices; j++) {
            PetscInt    l, p;
            PetscSFNode q;

            p = closurePointsF[numClosureIndices * c + j].index;
            if (rootType[p-pStartF] == PETSC_MAX_INT) continue;
            if (j == 0) { /* volume: ancestor is volume */
              l = 0;
            } else if (j < 1 + P4EST_FACES) { /* facet */
              PetscInt face = PetscFaceToP4estFace[j - 1];
              PetscInt direction = face / 2;
              PetscInt coarseFace = -1;

              if (coarseBound[face % 2][direction] == fineBound[face % 2][direction]) {
                coarseFace = face;
                l = 1 + P4estFaceToPetscFace[coarseFace];
              } else {
                l = 0;
              }
#if defined(P4_TO_P8)
            } else if (j < 1 + P4EST_FACES + P8EST_EDGES) {
              PetscInt  edge       = PetscEdgeToP4estEdge[j - (1 + P4EST_FACES)];
              PetscInt  direction  = edge / 4;
              PetscInt  mod        = edge % 4;
              PetscInt  coarseEdge = -1, coarseFace = -1;
              PetscInt  minDir     = PetscMin((direction + 1) % 3,(direction + 2) % 3);
              PetscInt  maxDir     = PetscMax((direction + 1) % 3,(direction + 2) % 3);
              PetscBool dirTest[2];

              dirTest[0] = (PetscBool) (coarseBound[mod % 2][minDir] == fineBound[mod % 2][minDir]);
              dirTest[1] = (PetscBool) (coarseBound[mod / 2][maxDir] == fineBound[mod / 2][maxDir]);

              if (dirTest[0] && dirTest[1]) { /* fine edge falls on coarse edge */
                coarseEdge = edge;
                l          = 1 + P4EST_FACES + P4estEdgeToPetscEdge[coarseEdge];
              } else if (dirTest[0]) { /* fine edge falls on a coarse face in the minDir direction */
                coarseFace = 2 * minDir + (mod % 2);
                l = 1 + P4estFaceToPetscFace[coarseFace];
              } else if (dirTest[1]) { /* fine edge falls on a coarse face in the maxDir direction */
                coarseFace = 2 * maxDir + (mod / 2);
                l = 1 + P4estFaceToPetscFace[coarseFace];
              } else {
                l = 0;
              }
#endif
            } else {
              PetscInt  vertex = PetscVertToP4estVert[P4EST_CHILDREN - (P4EST_INSUL - j)];
              PetscBool dirTest[P4EST_DIM];
              PetscInt  m;
              PetscInt  numMatch     = 0;
              PetscInt  coarseVertex = -1, coarseFace = -1;
#if defined(P4_TO_P8)
              PetscInt coarseEdge = -1;
#endif

              for (m = 0; m < P4EST_DIM; m++) {
                dirTest[m] = (PetscBool) (coarseBound[(vertex >> m) & 1][m] == fineBound[(vertex >> m) & 1][m]);
                if (dirTest[m]) numMatch++;
              }
              if (numMatch == P4EST_DIM) { /* vertex on vertex */
                coarseVertex = vertex;
                l            = P4EST_INSUL - (P4EST_CHILDREN - P4estVertToPetscVert[coarseVertex]);
              } else if (numMatch == 1) { /* vertex on face */
                for (m = 0; m < P4EST_DIM; m++) {
                  if (dirTest[m]) {
                    coarseFace = 2 * m + ((vertex >> m) & 1);
                    break;
                  }
                }
                l = 1 + P4estFaceToPetscFace[coarseFace];
#if defined(P4_TO_P8)
              } else if (numMatch == 2) { /* vertex on edge */
                for (m = 0; m < P4EST_DIM; m++) {
                  if (!dirTest[m]) {
                    PetscInt otherDir1 = (m + 1) % 3;
                    PetscInt otherDir2 = (m + 2) % 3;
                    PetscInt minDir    = PetscMin(otherDir1,otherDir2);
                    PetscInt maxDir    = PetscMax(otherDir1,otherDir2);

                    coarseEdge = m * 4 + 2 * ((vertex >> maxDir) & 1) + ((vertex >> minDir) & 1);
                    break;
                  }
                }
                l = 1 + P4EST_FACES + P4estEdgeToPetscEdge[coarseEdge];
#endif
              } else { /* volume */
                l = 0;
              }
            }
            q = closurePointsC[numClosureIndices * (coarseCount + coarseOffset) + l];
            if (l > rootType[p-pStartF]) {
              if (l >= P4EST_INSUL - P4EST_CHILDREN) { /* vertex on vertex: unconditional acceptance */
                if (transferIdent) {
                  roots[p-pStartF] = q;
                  rootType[p-pStartF] = PETSC_MAX_INT;
                  if (formCids) cids[p-pStartF] = -1;
                }
              } else {
                PetscInt k, thisp = p, limit;

                roots[p-pStartF] = q;
                rootType[p-pStartF] = l;
                if (formCids) cids[p - pStartF] = proposedCids[j];
                limit = transferIdent ? levelDiff : (levelDiff - 1);
                for (k = 0; k < limit; k++) {
                  PetscInt parent;

                  ierr = DMPlexGetTreeParent(plexF,thisp,&parent,NULL);CHKERRQ(ierr);
                  if (parent == thisp) break;

                  roots[parent-pStartF] = q;
                  rootType[parent-pStartF] = PETSC_MAX_INT;
                  if (formCids) cids[parent-pStartF] = -1;
                  thisp = parent;
                }
              }
            }
          }
        }
      }
    }

    /* now every cell has labeled the points in its closure, so we first make sure everyone agrees by reducing to roots, and the broadcast the agreements */
    if (size > 1) {
      PetscInt *rootTypeCopy, p;

      ierr = PetscMalloc1(pEndF-pStartF,&rootTypeCopy);CHKERRQ(ierr);
      ierr = PetscArraycpy(rootTypeCopy,rootType,pEndF-pStartF);CHKERRQ(ierr);
      ierr = PetscSFReduceBegin(pointSF,MPIU_INT,rootTypeCopy,rootTypeCopy,MPIU_MAX);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(pointSF,MPIU_INT,rootTypeCopy,rootTypeCopy,MPIU_MAX);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(pointSF,MPIU_INT,rootTypeCopy,rootTypeCopy);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(pointSF,MPIU_INT,rootTypeCopy,rootTypeCopy);CHKERRQ(ierr);
      for (p = pStartF; p < pEndF; p++) {
        if (rootTypeCopy[p-pStartF] > rootType[p-pStartF]) { /* another process found a root of higher type (e.g. vertex instead of edge), which we want to accept, so nullify this */
          roots[p-pStartF].rank  = -1;
          roots[p-pStartF].index = -1;
        }
        if (formCids && rootTypeCopy[p-pStartF] == PETSC_MAX_INT) {
          cids[p-pStartF] = -1; /* we have found an antecedent that is the same: no child id */
        }
      }
      ierr = PetscFree(rootTypeCopy);CHKERRQ(ierr);
      ierr = PetscSFReduceBegin(pointSF,nodeType,roots,roots,sfNodeReduce);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(pointSF,nodeType,roots,roots,sfNodeReduce);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(pointSF,nodeType,roots,roots);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(pointSF,nodeType,roots,roots);CHKERRQ(ierr);
    }
    ierr = PetscFree(rootType);CHKERRQ(ierr);

    {
      PetscInt    numRoots;
      PetscInt    numLeaves;
      PetscInt    *leaves;
      PetscSFNode *iremote;
      /* count leaves */

      numRoots = pEndC - pStartC;

      numLeaves = 0;
      for (p = pStartF; p < pEndF; p++) {
        if (roots[p-pStartF].index >= 0) numLeaves++;
      }
      ierr      = PetscMalloc1(numLeaves,&leaves);CHKERRQ(ierr);
      ierr      = PetscMalloc1(numLeaves,&iremote);CHKERRQ(ierr);
      numLeaves = 0;
      for (p = pStartF; p < pEndF; p++) {
        if (roots[p-pStartF].index >= 0) {
          leaves[numLeaves]  = p-pStartF;
          iremote[numLeaves] = roots[p-pStartF];
          numLeaves++;
        }
      }
      ierr = PetscFree(roots);CHKERRQ(ierr);
      ierr = PetscSFCreate(comm,sf);CHKERRQ(ierr);
      if (numLeaves == (pEndF-pStartF)) {
        ierr = PetscFree(leaves);CHKERRQ(ierr);
        ierr = PetscSFSetGraph(*sf,numRoots,numLeaves,NULL,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
      } else {
        ierr = PetscSFSetGraph(*sf,numRoots,numLeaves,leaves,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
      }
    }
    if (formCids) {
      PetscSF  pointSF;
      PetscInt child;

      ierr = DMPlexGetReferenceTree(plexF,&refTree);CHKERRQ(ierr);
      ierr = DMGetPointSF(plexF,&pointSF);CHKERRQ(ierr);
      ierr = PetscSFReduceBegin(pointSF,MPIU_INT,cids,cids,MPIU_MAX);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(pointSF,MPIU_INT,cids,cids,MPIU_MAX);CHKERRQ(ierr);
      if (childIds) *childIds = cids;
      for (child = 0; child < P4EST_CHILDREN; child++) {
        ierr = DMPlexRestoreTransitiveClosure(refTree,child+1,PETSC_TRUE,NULL,&childClosures[child]);CHKERRQ(ierr);
      }
      ierr = DMPlexRestoreTransitiveClosure(refTree,0,PETSC_TRUE,NULL,&rootClosure);CHKERRQ(ierr);
    }
  }
  if (saveInCoarse) { /* cache results */
    ierr = PetscObjectReference((PetscObject)*sf);CHKERRQ(ierr);
    pforestC->pointSelfToAdaptSF = *sf;
    if (!childIds) {
      pforestC->pointSelfToAdaptCids = cids;
    } else {
      ierr = PetscMalloc1(pEndF-pStartF,&pforestC->pointSelfToAdaptCids);CHKERRQ(ierr);
      ierr = PetscArraycpy(pforestC->pointSelfToAdaptCids,cids,pEndF-pStartF);CHKERRQ(ierr);
    }
  } else if (saveInFine) {
    ierr = PetscObjectReference((PetscObject)*sf);CHKERRQ(ierr);
    pforestF->pointAdaptToSelfSF = *sf;
    if (!childIds) {
      pforestF->pointAdaptToSelfCids = cids;
    } else {
      ierr = PetscMalloc1(pEndF-pStartF,&pforestF->pointAdaptToSelfCids);CHKERRQ(ierr);
      ierr = PetscArraycpy(pforestF->pointAdaptToSelfCids,cids,pEndF-pStartF);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree2(treeQuads,treeQuadCounts);CHKERRQ(ierr);
  ierr = PetscFree(coverQuads);CHKERRQ(ierr);
  ierr = PetscFree(closurePointsC);CHKERRQ(ierr);
  ierr = PetscFree(closurePointsF);CHKERRQ(ierr);
  ierr = MPI_Type_free(&nodeClosureType);CHKERRQ(ierr);
  ierr = MPI_Op_free(&sfNodeReduce);CHKERRQ(ierr);
  ierr = MPI_Type_free(&nodeType);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* children are sf leaves of parents */
static PetscErrorCode DMPforestGetTransferSF_Internal(DM coarse, DM fine, const PetscInt dofPerDim[], PetscSF *sf, PetscBool transferIdent, PetscInt *childIds[])
{
  MPI_Comm          comm;
  PetscMPIInt       rank, size;
  DM_Forest_pforest *pforestC, *pforestF;
  PetscInt          numClosureIndices;
  DM                plexC, plexF;
  PetscInt          pStartC, pEndC, pStartF, pEndF;
  PetscSF           pointTransferSF;
  PetscBool         allOnes = PETSC_TRUE;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  pforestC = (DM_Forest_pforest*) ((DM_Forest*) coarse->data)->data;
  pforestF = (DM_Forest_pforest*) ((DM_Forest*) fine->data)->data;
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
  {
    PetscInt i;
    for (i = 0; i <= P4EST_DIM; i++) {
      if (dofPerDim[i] != 1) {
        allOnes = PETSC_FALSE;
        break;
      }
    }
  }
  ierr = DMPforestGetTransferSF_Point(coarse,fine,&pointTransferSF,transferIdent,childIds);CHKERRQ(ierr);
  if (allOnes) {
    *sf = pointTransferSF;
    PetscFunctionReturn(0);
  }

  ierr = DMPforestGetPlex(fine,&plexF);CHKERRQ(ierr);
  ierr = DMPlexGetChart(plexF,&pStartF,&pEndF);CHKERRQ(ierr);
  ierr = DMPforestGetPlex(coarse,&plexC);CHKERRQ(ierr);
  ierr = DMPlexGetChart(plexC,&pStartC,&pEndC);CHKERRQ(ierr);
  {
    PetscInt          numRoots;
    PetscInt          numLeaves;
    const PetscInt    *leaves;
    const PetscSFNode *iremote;
    PetscInt          d;
    PetscSection      leafSection, rootSection;
    PetscInt          endInterior[4];
    /* count leaves */

    ierr = PetscSFGetGraph(pointTransferSF,&numRoots,&numLeaves,&leaves,&iremote);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PETSC_COMM_SELF,&rootSection);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PETSC_COMM_SELF,&leafSection);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(rootSection,pStartC,pEndC);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(leafSection,pStartF,pEndF);CHKERRQ(ierr);

    ierr = DMPlexGetHybridBounds(plexC,&endInterior[P4EST_DIM],&endInterior[P4EST_DIM - 1],&endInterior[1],&endInterior[0]);CHKERRQ(ierr);
    for (d = 0; d <= P4EST_DIM; d++) {
      PetscInt startC, endC, e;

      ierr = DMPlexGetDepthStratum(plexC,d,&startC,&endC);CHKERRQ(ierr);
      endC = endInterior[d] < 0 ? endC : endInterior[d];
      for (e = startC; e < endC; e++) {
        ierr = PetscSectionSetDof(rootSection,e,dofPerDim[d]);CHKERRQ(ierr);
      }
    }

    ierr = DMPlexGetHybridBounds(plexF,&endInterior[P4EST_DIM],&endInterior[P4EST_DIM - 1],&endInterior[1],&endInterior[0]);CHKERRQ(ierr);
    for (d = 0; d <= P4EST_DIM; d++) {
      PetscInt startF, endF, e;

      ierr = DMPlexGetDepthStratum(plexF,d,&startF,&endF);CHKERRQ(ierr);
      endF = endInterior[d] < 0 ? endF : endInterior[d];
      for (e = startF; e < endF; e++) {
        ierr = PetscSectionSetDof(leafSection,e,dofPerDim[d]);CHKERRQ(ierr);
      }
    }

    ierr = PetscSectionSetUp(rootSection);CHKERRQ(ierr);
    ierr = PetscSectionSetUp(leafSection);CHKERRQ(ierr);
    {
      PetscInt    nroots, nleaves;
      PetscInt    *mine, i, p;
      PetscInt    *offsets, *offsetsRoot;
      PetscSFNode *remote;

      ierr = PetscMalloc1(pEndF-pStartF,&offsets);CHKERRQ(ierr);
      ierr = PetscMalloc1(pEndC-pStartC,&offsetsRoot);CHKERRQ(ierr);
      for (p = pStartC; p < pEndC; p++) {
        ierr = PetscSectionGetOffset(rootSection,p,&offsetsRoot[p-pStartC]);CHKERRQ(ierr);
      }
      ierr    = PetscSFBcastBegin(pointTransferSF,MPIU_INT,offsetsRoot,offsets);CHKERRQ(ierr);
      ierr    = PetscSFBcastEnd(pointTransferSF,MPIU_INT,offsetsRoot,offsets);CHKERRQ(ierr);
      ierr    = PetscSectionGetStorageSize(rootSection,&nroots);CHKERRQ(ierr);
      nleaves = 0;
      for (i = 0; i < numLeaves; i++) {
        PetscInt leaf = leaves ? leaves[i] : i;
        PetscInt dof;

        ierr     = PetscSectionGetDof(leafSection,leaf,&dof);CHKERRQ(ierr);
        nleaves += dof;
      }
      ierr    = PetscMalloc1(nleaves,&mine);CHKERRQ(ierr);
      ierr    = PetscMalloc1(nleaves,&remote);CHKERRQ(ierr);
      nleaves = 0;
      for (i = 0; i < numLeaves; i++) {
        PetscInt leaf = leaves ? leaves[i] : i;
        PetscInt dof;
        PetscInt off, j;

        ierr = PetscSectionGetDof(leafSection,leaf,&dof);CHKERRQ(ierr);
        ierr = PetscSectionGetOffset(leafSection,leaf,&off);CHKERRQ(ierr);
        for (j = 0; j < dof; j++) {
          remote[nleaves].rank  = iremote[i].rank;
          remote[nleaves].index = offsets[leaf] + j;
          mine[nleaves++]       = off + j;
        }
      }
      ierr = PetscFree(offsetsRoot);CHKERRQ(ierr);
      ierr = PetscFree(offsets);CHKERRQ(ierr);
      ierr = PetscSFCreate(comm,sf);CHKERRQ(ierr);
      ierr = PetscSFSetGraph(*sf,nroots,nleaves,mine,PETSC_OWN_POINTER,remote,PETSC_OWN_POINTER);CHKERRQ(ierr);
    }
    ierr = PetscSectionDestroy(&leafSection);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&rootSection);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&pointTransferSF);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPforestGetTransferSF(DM dmA, DM dmB, const PetscInt dofPerDim[], PetscSF *sfAtoB, PetscSF *sfBtoA)
{
  DM             adaptA, adaptB;
  DMAdaptFlag    purpose;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMForestGetAdaptivityForest(dmA,&adaptA);CHKERRQ(ierr);
  ierr = DMForestGetAdaptivityForest(dmB,&adaptB);CHKERRQ(ierr);
  /* it is more efficient when the coarser mesh is the first argument: reorder if we know one is coarser than the other */
  if (adaptA && adaptA->data == dmB->data) { /* dmA was adapted from dmB */
    ierr = DMForestGetAdaptivityPurpose(dmA,&purpose);CHKERRQ(ierr);
    if (purpose == DM_ADAPT_REFINE) {
      ierr = DMPforestGetTransferSF(dmB, dmA, dofPerDim, sfBtoA, sfAtoB);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  } else if (adaptB && adaptB->data == dmA->data) { /* dmB was adapted from dmA */
    ierr = DMForestGetAdaptivityPurpose(dmB,&purpose);CHKERRQ(ierr);
    if (purpose == DM_ADAPT_COARSEN) {
      ierr = DMPforestGetTransferSF(dmB, dmA, dofPerDim, sfBtoA, sfAtoB);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    }
  }
  if (sfAtoB) {
    ierr = DMPforestGetTransferSF_Internal(dmA,dmB,dofPerDim,sfAtoB,PETSC_TRUE,NULL);CHKERRQ(ierr);
  }
  if (sfBtoA) {
    ierr = DMPforestGetTransferSF_Internal(dmB,dmA,dofPerDim,sfBtoA,(PetscBool) (sfAtoB == NULL),NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPforestLabelsInitialize(DM dm, DM plex)
{
  DM_Forest         *forest  = (DM_Forest*) dm->data;
  DM_Forest_pforest *pforest = (DM_Forest_pforest*) forest->data;
  PetscInt          cLocalStart, cLocalEnd, cStart, cEnd, fStart, fEnd, eStart, eEnd, vStart, vEnd;
  PetscInt          cStartBase, cEndBase, fStartBase, fEndBase, vStartBase, vEndBase, eStartBase, eEndBase;
  PetscInt          cEndBaseInterior, fEndBaseInterior, vEndBaseInterior, eEndBaseInterior;
  PetscInt          cEndInterior, fEndInterior, vEndInterior, eEndInterior;
  PetscInt          pStart, pEnd, pStartBase, pEndBase, p;
  DM                base;
  PetscInt          *star     = NULL, starSize;
  DMLabelLink       next      = dm->labels->next;
  PetscInt          guess     = 0;
  p4est_topidx_t    num_trees = pforest->topo->conn->num_trees;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  pforest->labelsFinalized = PETSC_TRUE;
  cLocalStart              = pforest->cLocalStart;
  cLocalEnd                = pforest->cLocalEnd;
  ierr                     = DMForestGetBaseDM(dm,&base);CHKERRQ(ierr);
  if (!base) {
    if (pforest->ghostName) { /* insert a label to make the boundaries, with stratum values denoting which face of the element touches the boundary */
      p4est_connectivity_t *conn  = pforest->topo->conn;
      p4est_t              *p4est = pforest->forest;
      p4est_tree_t         *trees = (p4est_tree_t*) p4est->trees->array;
      p4est_topidx_t       t, flt = p4est->first_local_tree;
      p4est_topidx_t       llt = pforest->forest->last_local_tree;
      DMLabel              ghostLabel;
      PetscInt             c;

      ierr = DMCreateLabel(plex,pforest->ghostName);CHKERRQ(ierr);
      ierr = DMGetLabel(plex,pforest->ghostName,&ghostLabel);CHKERRQ(ierr);
      for (c = cLocalStart, t = flt; t <= llt; t++) {
        p4est_tree_t     *tree    = &trees[t];
        p4est_quadrant_t *quads   = (p4est_quadrant_t*) tree->quadrants.array;
        PetscInt         numQuads = (PetscInt) tree->quadrants.elem_count;
        PetscInt         q;

        for (q = 0; q < numQuads; q++, c++) {
          p4est_quadrant_t *quad = &quads[q];
          PetscInt         f;

          for (f = 0; f < P4EST_FACES; f++) {
            p4est_quadrant_t neigh;
            int              isOutside;

            PetscStackCallP4est(p4est_quadrant_face_neighbor,(quad,f,&neigh));
            PetscStackCallP4estReturn(isOutside,p4est_quadrant_is_outside_face,(&neigh));
            if (isOutside) {
              p4est_topidx_t nt;
              PetscInt       nf;

              nt = conn->tree_to_tree[t * P4EST_FACES + f];
              nf = (PetscInt) conn->tree_to_face[t * P4EST_FACES + f];
              nf = nf % P4EST_FACES;
              if (nt == t && nf == f) {
                PetscInt       plexF = P4estFaceToPetscFace[f];
                const PetscInt *cone;

                ierr = DMPlexGetCone(plex,c,&cone);CHKERRQ(ierr);
                ierr = DMLabelSetValue(ghostLabel,cone[plexF],plexF+1);CHKERRQ(ierr);
              }
            }
          }
        }
      }
    }
    PetscFunctionReturn(0);
  }
  ierr     = DMPlexGetHybridBounds(base,&cEndBaseInterior,&fEndBaseInterior,&eEndBaseInterior,&vEndBaseInterior);CHKERRQ(ierr);
  ierr     = DMPlexGetHeightStratum(base,0,&cStartBase,&cEndBase);CHKERRQ(ierr);
  cEndBase = cEndBaseInterior < 0 ? cEndBase : cEndBaseInterior;
  ierr     = DMPlexGetHeightStratum(base,1,&fStartBase,&fEndBase);CHKERRQ(ierr);
  fEndBase = fEndBaseInterior < 0 ? fEndBase : fEndBaseInterior;
  ierr     = DMPlexGetDepthStratum(base,1,&eStartBase,&eEndBase);CHKERRQ(ierr);
  eEndBase = eEndBaseInterior < 0 ? eEndBase : eEndBaseInterior;
  ierr     = DMPlexGetDepthStratum(base,0,&vStartBase,&vEndBase);CHKERRQ(ierr);
  vEndBase = vEndBaseInterior < 0 ? vEndBase : vEndBaseInterior;

  ierr = DMPlexGetHybridBounds(plex,&cEndInterior,&fEndInterior,&eEndInterior,&vEndInterior);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(plex,0,&cStart,&cEnd);CHKERRQ(ierr);
  cEnd = cEndInterior < 0 ? cEnd : cEndInterior;
  ierr = DMPlexGetHeightStratum(plex,1,&fStart,&fEnd);CHKERRQ(ierr);
  fEnd = fEndInterior < 0 ? fEnd : fEndInterior;
  ierr = DMPlexGetDepthStratum(plex,1,&eStart,&eEnd);CHKERRQ(ierr);
  eEnd = eEndInterior < 0 ? eEnd : eEndInterior;
  ierr = DMPlexGetDepthStratum(plex,0,&vStart,&vEnd);CHKERRQ(ierr);
  vEnd = vEndInterior < 0 ? vEnd : vEndInterior;

  ierr = DMPlexGetChart(plex,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = DMPlexGetChart(base,&pStartBase,&pEndBase);CHKERRQ(ierr);
  /* go through the mesh: use star to find a quadrant that borders a point.  Use the closure to determine the
   * orientation of the quadrant relative to that point.  Use that to relate the point to the numbering in the base
   * mesh, and extract a label value (since the base mesh is redundantly distributed, can be found locally). */
  while (next) {
    DMLabel   baseLabel;
    DMLabel   label = next->label;
    PetscBool isDepth, isGhost, isVTK, isSpmap;
    const char *name;

    ierr = PetscObjectGetName((PetscObject) label, &name);CHKERRQ(ierr);
    ierr = PetscStrcmp(name,"depth",&isDepth);CHKERRQ(ierr);
    if (isDepth) {
      next = next->next;
      continue;
    }
    ierr = PetscStrcmp(name,"ghost",&isGhost);CHKERRQ(ierr);
    if (isGhost) {
      next = next->next;
      continue;
    }
    ierr = PetscStrcmp(name,"vtk",&isVTK);CHKERRQ(ierr);
    if (isVTK) {
      next = next->next;
      continue;
    }
    ierr = PetscStrcmp(name,"_forest_base_subpoint_map",&isSpmap);CHKERRQ(ierr);
    if (!isSpmap) {
      ierr = DMGetLabel(base,name,&baseLabel);CHKERRQ(ierr);
      if (!baseLabel) {
        next = next->next;
        continue;
      }
      ierr = DMLabelCreateIndex(baseLabel,pStartBase,pEndBase);CHKERRQ(ierr);
    } else baseLabel = NULL;

    for (p = pStart; p < pEnd; p++) {
      PetscInt         s, c = -1, l;
      PetscInt         *closure = NULL, closureSize;
      p4est_quadrant_t * ghosts = (p4est_quadrant_t*) pforest->ghost->ghosts.array;
      p4est_tree_t     *trees   = (p4est_tree_t*) pforest->forest->trees->array;
      p4est_quadrant_t * q;
      PetscInt         t, val;
      PetscBool        zerosupportpoint = PETSC_FALSE;

      ierr = DMPlexGetTransitiveClosure(plex,p,PETSC_FALSE,&starSize,&star);CHKERRQ(ierr);
      for (s = 0; s < starSize; s++) {
        PetscInt point = star[2*s];

        if (cStart <= point && point < cEnd) {
          ierr = DMPlexGetTransitiveClosure(plex,point,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
          for (l = 0; l < closureSize; l++) {
            PetscInt qParent = closure[2 * l], q, pp = p, pParent = p;
            do { /* check parents of q */
              q = qParent;
              if (q == p) {
                c = point;
                break;
              }
              ierr = DMPlexGetTreeParent(plex,q,&qParent,NULL);CHKERRQ(ierr);
            } while (qParent != q);
            if (c != -1) break;
            ierr = DMPlexGetTreeParent(plex,pp,&pParent,NULL);CHKERRQ(ierr);
            q = closure[2 * l];
            while (pParent != pp) { /* check parents of p */
              pp = pParent;
              if (pp == q) {
                c = point;
                break;
              }
              ierr = DMPlexGetTreeParent(plex,pp,&pParent,NULL);CHKERRQ(ierr);
            }
            if (c != -1) break;
          }
          ierr = DMPlexRestoreTransitiveClosure(plex,point,PETSC_TRUE,NULL,&closure);CHKERRQ(ierr);
          if (l < closureSize) break;
        } else {
          PetscInt supportSize;

          ierr = DMPlexGetSupportSize(plex,point,&supportSize);CHKERRQ(ierr);
          zerosupportpoint = (PetscBool) (zerosupportpoint || !supportSize);
        }
      }
      if (c < 0) {
        const char* prefix;
        PetscBool   print = PETSC_FALSE;

        ierr = PetscObjectGetOptionsPrefix((PetscObject)dm,&prefix);CHKERRQ(ierr);
        ierr = PetscOptionsGetBool(((PetscObject)dm)->options,prefix,"-dm_forest_print_label_error",&print,NULL);CHKERRQ(ierr);
        if (print) {
          PetscInt i;

          ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] Failed to find cell with point %D in its closure for label %s (starSize %D)\n",PetscGlobalRank,p,baseLabel ? ((PetscObject)baseLabel)->name : "_forest_base_subpoint_map",starSize);
          for (i = 0; i < starSize; i++) { ierr = PetscPrintf(PETSC_COMM_SELF,"  star[%D] = %D,%D\n",i,star[2*i],star[2*i+1]);CHKERRQ(ierr); }
        }
        ierr = DMPlexRestoreTransitiveClosure(plex,p,PETSC_FALSE,NULL,&star);CHKERRQ(ierr);
        if (zerosupportpoint) continue;
        else SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Failed to find cell with point %D in its closure for label %s. Rerun with -dm_forest_print_label_error for more information",p,baseLabel ? ((PetscObject) baseLabel)->name : "_forest_base_subpoint_map");
      }
      ierr = DMPlexRestoreTransitiveClosure(plex,p,PETSC_FALSE,NULL,&star);CHKERRQ(ierr);

      if (c < cLocalStart) {
        /* get from the beginning of the ghost layer */
        q = &(ghosts[c]);
        t = (PetscInt) q->p.which_tree;
      } else if (c < cLocalEnd) {
        PetscInt lo = 0, hi = num_trees;
        /* get from local quadrants: have to find the right tree */

        c -= cLocalStart;

        do {
          p4est_tree_t *tree;

          if (guess < lo || guess >= num_trees || lo >= hi) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"failed binary search");
          tree = &trees[guess];
          if (c < tree->quadrants_offset) {
            hi = guess;
          } else if (c < tree->quadrants_offset + (PetscInt) tree->quadrants.elem_count) {
            q = &((p4est_quadrant_t *)tree->quadrants.array)[c - (PetscInt) tree->quadrants_offset];
            t = guess;
            break;
          } else {
            lo = guess + 1;
          }
          guess = lo + (hi - lo) / 2;
        } while (1);
      } else {
        /* get from the end of the ghost layer */
        c -= (cLocalEnd - cLocalStart);

        q = &(ghosts[c]);
        t = (PetscInt) q->p.which_tree;
      }

      if (l == 0) { /* cell */
        if (baseLabel) {
          ierr = DMLabelGetValue(baseLabel,t+cStartBase,&val);CHKERRQ(ierr);
        } else {
          val  = t+cStartBase;
        }
        ierr = DMLabelSetValue(label,p,val);CHKERRQ(ierr);
      } else if (l >= 1 && l < 1 + P4EST_FACES) { /* facet */
        p4est_quadrant_t nq;
        int              isInside;

        l = PetscFaceToP4estFace[l - 1];
        PetscStackCallP4est(p4est_quadrant_face_neighbor,(q,l,&nq));
        PetscStackCallP4estReturn(isInside,p4est_quadrant_is_inside_root,(&nq));
        if (isInside) {
          /* this facet is in the interior of a tree, so it inherits the label of the tree */
          if (baseLabel) {
            ierr = DMLabelGetValue(baseLabel,t+cStartBase,&val);CHKERRQ(ierr);
          } else {
            val  = t+cStartBase;
          }
          ierr = DMLabelSetValue(label,p,val);CHKERRQ(ierr);
        } else {
          PetscInt f = pforest->topo->tree_face_to_uniq[P4EST_FACES * t + l];

          if (baseLabel) {
            ierr = DMLabelGetValue(baseLabel,f+fStartBase,&val);CHKERRQ(ierr);
          } else {
            val  = f+fStartBase;
          }
          ierr = DMLabelSetValue(label,p,val);CHKERRQ(ierr);
        }
#if defined(P4_TO_P8)
      } else if (l >= 1 + P4EST_FACES && l < 1 + P4EST_FACES + P8EST_EDGES) { /* edge */
        p4est_quadrant_t nq;
        int              isInside;

        l = PetscEdgeToP4estEdge[l - (1 + P4EST_FACES)];
        PetscStackCallP4est(p8est_quadrant_edge_neighbor,(q,l,&nq));
        PetscStackCallP4estReturn(isInside,p4est_quadrant_is_inside_root,(&nq));
        if (isInside) {
          /* this edge is in the interior of a tree, so it inherits the label of the tree */
          if (baseLabel) {
            ierr = DMLabelGetValue(baseLabel,t+cStartBase,&val);CHKERRQ(ierr);
          } else {
            val  = t+cStartBase;
          }
          ierr = DMLabelSetValue(label,p,val);CHKERRQ(ierr);
        } else {
          int isOutsideFace;

          PetscStackCallP4estReturn(isOutsideFace,p4est_quadrant_is_outside_face,(&nq));
          if (isOutsideFace) {
            PetscInt f;

            if (nq.x < 0) {
              f = 0;
            } else if (nq.x >= P4EST_ROOT_LEN) {
              f = 1;
            } else if (nq.y < 0) {
              f = 2;
            } else if (nq.y >= P4EST_ROOT_LEN) {
              f = 3;
            } else if (nq.z < 0) {
              f = 4;
            } else {
              f = 5;
            }
            f    = pforest->topo->tree_face_to_uniq[P4EST_FACES * t + f];
            if (baseLabel) {
              ierr = DMLabelGetValue(baseLabel,f+fStartBase,&val);CHKERRQ(ierr);
            } else {
              val  = f+fStartBase;
            }
            ierr = DMLabelSetValue(label,p,val);CHKERRQ(ierr);
          } else { /* the quadrant edge corresponds to the tree edge */
            PetscInt e = pforest->topo->conn->tree_to_edge[P8EST_EDGES * t + l];

            if (baseLabel) {
              ierr = DMLabelGetValue(baseLabel,e+eStartBase,&val);CHKERRQ(ierr);
            } else {
              val  = e+eStartBase;
            }
            ierr = DMLabelSetValue(label,p,val);CHKERRQ(ierr);
          }
        }
#endif
      } else { /* vertex */
        p4est_quadrant_t nq;
        int              isInside;

#if defined(P4_TO_P8)
        l = PetscVertToP4estVert[l - (1 + P4EST_FACES + P8EST_EDGES)];
#else
        l = PetscVertToP4estVert[l - (1 + P4EST_FACES)];
#endif
        PetscStackCallP4est(p4est_quadrant_corner_neighbor,(q,l,&nq));
        PetscStackCallP4estReturn(isInside,p4est_quadrant_is_inside_root,(&nq));
        if (isInside) {
          if (baseLabel) {
            ierr = DMLabelGetValue(baseLabel,t+cStartBase,&val);CHKERRQ(ierr);
          } else {
            val  = t+cStartBase;
          }
          ierr = DMLabelSetValue(label,p,val);CHKERRQ(ierr);
        } else {
          int isOutside;

          PetscStackCallP4estReturn(isOutside,p4est_quadrant_is_outside_face,(&nq));
          if (isOutside) {
            PetscInt f = -1;

            if (nq.x < 0) {
              f = 0;
            } else if (nq.x >= P4EST_ROOT_LEN) {
              f = 1;
            } else if (nq.y < 0) {
              f = 2;
            } else if (nq.y >= P4EST_ROOT_LEN) {
              f = 3;
#if defined(P4_TO_P8)
            } else if (nq.z < 0) {
              f = 4;
            } else {
              f = 5;
#endif
            }
            f    = pforest->topo->tree_face_to_uniq[P4EST_FACES * t + f];
            if (baseLabel) {
              ierr = DMLabelGetValue(baseLabel,f+fStartBase,&val);CHKERRQ(ierr);
            } else {
              val  = f+fStartBase;
            }
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
                } else {
                  e = 1;
                }
              } else {
                if (nq.y < 0) {
                  e = 2;
                } else {
                  e = 3;
                }
              }
            } else if (nq.y >= 0 && nq.y < P4EST_ROOT_LEN) {
              if (nq.z < 0) {
                if (nq.x < 0) {
                  e = 4;
                } else {
                  e = 5;
                }
              } else {
                if (nq.x < 0) {
                  e = 6;
                } else {
                  e = 7;
                }
              }
            } else {
              if (nq.y < 0) {
                if (nq.x < 0) {
                  e = 8;
                } else {
                  e = 9;
                }
              } else {
                if (nq.x < 0) {
                  e = 10;
                } else {
                  e = 11;
                }
              }
            }

            e    = pforest->topo->conn->tree_to_edge[P8EST_EDGES * t + e];
            if (baseLabel) {
              ierr = DMLabelGetValue(baseLabel,e+eStartBase,&val);CHKERRQ(ierr);
            } else {
              val  = e+eStartBase;
            }
            ierr = DMLabelSetValue(label,p,val);CHKERRQ(ierr);
            continue;
          }
#endif
          {
            /* outside vertex: same corner as quadrant corner */
            PetscInt v = pforest->topo->conn->tree_to_corner[P4EST_CHILDREN * t + l];

            if (baseLabel) {
              ierr = DMLabelGetValue(baseLabel,v+vStartBase,&val);CHKERRQ(ierr);
            } else {
              val  = v+vStartBase;
            }
            ierr = DMLabelSetValue(label,p,val);CHKERRQ(ierr);
          }
        }
      }
    }
    next = next->next;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPforestLabelsFinalize(DM dm, DM plex)
{
  DM_Forest_pforest *pforest = (DM_Forest_pforest*) ((DM_Forest*) dm->data)->data;
  DM                adapt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (pforest->labelsFinalized) PetscFunctionReturn(0);
  pforest->labelsFinalized = PETSC_TRUE;
  ierr                     = DMForestGetAdaptivityForest(dm,&adapt);CHKERRQ(ierr);
  if (!adapt) {
    /* Initialize labels from the base dm */
    ierr = DMPforestLabelsInitialize(dm,plex);CHKERRQ(ierr);
  } else {
    PetscInt    dofPerDim[4]={1, 1, 1, 1};
    PetscSF     transferForward, transferBackward, pointSF;
    PetscInt    pStart, pEnd, pStartA, pEndA;
    PetscInt    *values, *adaptValues;
    DMLabelLink next = adapt->labels->next;
    DMLabel     adaptLabel;
    DM          adaptPlex;

    ierr = DMForestGetAdaptivityLabel(dm,&adaptLabel);CHKERRQ(ierr);
    ierr = DMPforestGetPlex(adapt,&adaptPlex);CHKERRQ(ierr);
    ierr = DMPforestGetTransferSF(adapt,dm,dofPerDim,&transferForward,&transferBackward);CHKERRQ(ierr);
    ierr = DMPlexGetChart(plex,&pStart,&pEnd);CHKERRQ(ierr);
    ierr = DMPlexGetChart(adaptPlex,&pStartA,&pEndA);CHKERRQ(ierr);
    ierr = PetscMalloc2(pEnd-pStart,&values,pEndA-pStartA,&adaptValues);CHKERRQ(ierr);
    ierr = DMGetPointSF(plex,&pointSF);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
    {
      PetscInt p;
      for (p = pStartA; p < pEndA; p++) adaptValues[p-pStartA] = -1;
      for (p = pStart; p < pEnd; p++)   values[p-pStart]       = -2;
      if (transferForward) {
        ierr = PetscSFBcastBegin(transferForward,MPIU_INT,adaptValues,values);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(transferForward,MPIU_INT,adaptValues,values);CHKERRQ(ierr);
      }
      if (transferBackward) {
        ierr = PetscSFReduceBegin(transferBackward,MPIU_INT,adaptValues,values,MPIU_MAX);CHKERRQ(ierr);
        ierr = PetscSFReduceEnd(transferBackward,MPIU_INT,adaptValues,values,MPIU_MAX);CHKERRQ(ierr);
      }
      for (p = pStart; p < pEnd; p++) {
        PetscInt q = p, parent;

        ierr = DMPlexGetTreeParent(plex,q,&parent,NULL);CHKERRQ(ierr);
        while (parent != q) {
          if (values[parent] == -2) values[parent] = values[q];
          q    = parent;
          ierr = DMPlexGetTreeParent(plex,q,&parent,NULL);CHKERRQ(ierr);
        }
      }
      ierr = PetscSFReduceBegin(pointSF,MPIU_INT,values,values,MPIU_MAX);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(pointSF,MPIU_INT,values,values,MPIU_MAX);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(pointSF,MPIU_INT,values,values);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(pointSF,MPIU_INT,values,values);CHKERRQ(ierr);
      for (p = pStart; p < pEnd; p++) {
        if (values[p-pStart] == -2) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"uncovered point %D",p);
      }
    }
#endif
    while (next) {
      DMLabel    nextLabel = next->label;
      const char *name;
      PetscBool  isDepth, isGhost, isVTK;
      DMLabel    label;
      PetscInt   p;

      ierr = PetscObjectGetName((PetscObject) nextLabel, &name);CHKERRQ(ierr);
      ierr = PetscStrcmp(name,"depth",&isDepth);CHKERRQ(ierr);
      if (isDepth) {
        next = next->next;
        continue;
      }
      ierr = PetscStrcmp(name,"ghost",&isGhost);CHKERRQ(ierr);
      if (isGhost) {
        next = next->next;
        continue;
      }
      ierr = PetscStrcmp(name,"vtk",&isVTK);CHKERRQ(ierr);
      if (isVTK) {
        next = next->next;
        continue;
      }
      if (nextLabel == adaptLabel) {
        next = next->next;
        continue;
      }
      /* label was created earlier */
      ierr = DMGetLabel(dm,name,&label);CHKERRQ(ierr);
      for (p = pStartA; p < pEndA; p++) {
        ierr = DMLabelGetValue(nextLabel,p,&adaptValues[p]);CHKERRQ(ierr);
      }
      for (p = pStart; p < pEnd; p++) values[p] = PETSC_MIN_INT;

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
        PetscInt q = p, parent;

        ierr = DMPlexGetTreeParent(plex,q,&parent,NULL);CHKERRQ(ierr);
        while (parent != q) {
          if (values[parent] == PETSC_MIN_INT) values[parent] = values[q];
          q    = parent;
          ierr = DMPlexGetTreeParent(plex,q,&parent,NULL);CHKERRQ(ierr);
        }
      }
      ierr = PetscSFReduceBegin(pointSF,MPIU_INT,values,values,MPIU_MAX);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(pointSF,MPIU_INT,values,values,MPIU_MAX);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(pointSF,MPIU_INT,values,values);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(pointSF,MPIU_INT,values,values);CHKERRQ(ierr);

      for (p = pStart; p < pEnd; p++) {
        ierr = DMLabelSetValue(label,p,values[p]);CHKERRQ(ierr);
      }
      next = next->next;
    }
    ierr                     = PetscFree2(values,adaptValues);CHKERRQ(ierr);
    ierr                     = PetscSFDestroy(&transferForward);CHKERRQ(ierr);
    ierr                     = PetscSFDestroy(&transferBackward);CHKERRQ(ierr);
    pforest->labelsFinalized = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPforestMapCoordinates_Cell(DM plex, p4est_geometry_t *geom, PetscInt cell, p4est_quadrant_t *q, p4est_topidx_t t, p4est_connectivity_t * conn, PetscScalar *coords)
{
  PetscInt       closureSize, c, coordStart, coordEnd, coordDim;
  PetscInt       *closure = NULL;
  PetscSection   coordSec;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr          = DMGetCoordinateSection(plex,&coordSec);CHKERRQ(ierr);
  ierr          = PetscSectionGetChart(coordSec,&coordStart,&coordEnd);CHKERRQ(ierr);
  ierr          = DMGetCoordinateDim(plex,&coordDim);CHKERRQ(ierr);
  ierr          = DMPlexGetTransitiveClosure(plex,cell,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
  for (c = 0; c < closureSize; c++) {
    PetscInt point = closure[2 * c];

    if (point >= coordStart && point < coordEnd) {
      PetscInt dof, off;
      PetscInt nCoords, i;
      ierr = PetscSectionGetDof(coordSec,point,&dof);CHKERRQ(ierr);
      if (dof % coordDim) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Did not understand coordinate layout");
      nCoords = dof / coordDim;
      ierr    = PetscSectionGetOffset(coordSec,point,&off);CHKERRQ(ierr);
      for (i = 0; i < nCoords; i++) {
        PetscScalar *coord              = &coords[off + i * coordDim];
        double      coordP4est[3]       = {0.};
        double      coordP4estMapped[3] = {0.};
        PetscInt    j;
        PetscReal   treeCoords[P4EST_CHILDREN][3] = {{0.}};
        PetscReal   eta[3]                        = {0.};
        PetscInt    numRounds                     = 10;
        PetscReal   coordGuess[3]                 = {0.};

        eta[0] = (PetscReal) q->x / (PetscReal) P4EST_ROOT_LEN;
        eta[1] = (PetscReal) q->y / (PetscReal) P4EST_ROOT_LEN;
#if defined(P4_TO_P8)
        eta[2] = (PetscReal) q->z / (PetscReal) P4EST_ROOT_LEN;
#endif

        for (j = 0; j < P4EST_CHILDREN; j++) {
          PetscInt k;

          for (k = 0; k < 3; k++) treeCoords[j][k] = conn->vertices[3 * conn->tree_to_vertex[P4EST_CHILDREN * t + j] + k];
        }

        for (j = 0; j < P4EST_CHILDREN; j++) {
          PetscInt  k;
          PetscReal prod = 1.;

          for (k = 0; k < P4EST_DIM; k++) prod *= (j & (1 << k)) ? eta[k] : (1. - eta[k]);
          for (k = 0; k < 3; k++) coordGuess[k] += prod * treeCoords[j][k];
        }

        for (j = 0; j < numRounds; j++) {
          PetscInt dir;

          for (dir = 0; dir < P4EST_DIM; dir++) {
            PetscInt  k;
            PetscReal diff[3];
            PetscReal dXdeta[3] = {0.};
            PetscReal rhs, scale, update;

            for (k = 0; k < 3; k++) diff[k] = coordP4est[k] - coordGuess[k];
            for (k = 0; k < P4EST_CHILDREN; k++) {
              PetscInt  l;
              PetscReal prod = 1.;

              for (l = 0; l < P4EST_DIM; l++) {
                if (l == dir) {
                  prod *= (k & (1 << l)) ?  1. : -1.;
                } else {
                  prod *= (k & (1 << l)) ? eta[l] : (1. - eta[l]);
                }
              }
              for (l = 0; l < 3; l++) dXdeta[l] += prod * treeCoords[k][l];
            }
            rhs   = 0.;
            scale = 0;
            for (k = 0; k < 3; k++) {
              rhs   += diff[k] * dXdeta[k];
              scale += dXdeta[k] * dXdeta[k];
            }
            update    = rhs / scale;
            eta[dir] += update;
            eta[dir]  = PetscMin(eta[dir],1.);
            eta[dir]  = PetscMax(eta[dir],0.);

            coordGuess[0] = coordGuess[1] = coordGuess[2] = 0.;
            for (k = 0; k < P4EST_CHILDREN; k++) {
              PetscInt  l;
              PetscReal prod = 1.;

              for (l = 0; l < P4EST_DIM; l++) prod *= (k & (1 << l)) ? eta[l] : (1. - eta[l]);
              for (l = 0; l < 3; l++) coordGuess[l] += prod * treeCoords[k][l];
            }
          }
        }
        for (j = 0; j < 3; j++) coordP4est[j] = (double) eta[j];

        if (geom) {
          (geom->X)(geom,t,coordP4est,coordP4estMapped);
          for (j = 0; j < coordDim; j++) coord[j] = (PetscScalar) coordP4estMapped[j];
        } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not coded");
      }
    }
  }
  ierr = DMPlexRestoreTransitiveClosure(plex,cell,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPforestMapCoordinates(DM dm, DM plex)
{
  DM_Forest         *forest;
  DM_Forest_pforest *pforest;
  p4est_geometry_t  *geom;
  PetscInt          cLocalStart, cLocalEnd;
  Vec               coordLocalVec;
  PetscScalar       *coords;
  p4est_topidx_t    flt, llt, t;
  p4est_tree_t      *trees;
  PetscErrorCode    (*map)(DM,PetscInt, PetscInt, const PetscReal [], PetscReal [], void*);
  void              *mapCtx;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  forest  = (DM_Forest*) dm->data;
  pforest = (DM_Forest_pforest*) forest->data;
  geom    = pforest->topo->geom;
  ierr    = DMForestGetBaseCoordinateMapping(dm,&map,&mapCtx);CHKERRQ(ierr);
  if (!geom && !map) PetscFunctionReturn(0);
  ierr        = DMGetCoordinatesLocal(plex,&coordLocalVec);CHKERRQ(ierr);
  ierr        = VecGetArray(coordLocalVec,&coords);CHKERRQ(ierr);
  cLocalStart = pforest->cLocalStart;
  cLocalEnd   = pforest->cLocalEnd;
  flt         = pforest->forest->first_local_tree;
  llt         = pforest->forest->last_local_tree;
  trees       = (p4est_tree_t*) pforest->forest->trees->array;
  if (map) { /* apply the map directly to the existing coordinates */
    PetscSection coordSec;
    PetscInt     coordStart, coordEnd, p, coordDim, p4estCoordDim, cStart, cEnd, cEndInterior;
    DM           base;

    ierr          = DMPlexGetHeightStratum(plex,0,&cStart,&cEnd);CHKERRQ(ierr);
    ierr          = DMPlexGetHybridBounds(plex,&cEndInterior,NULL,NULL,NULL);CHKERRQ(ierr);
    cEnd          = cEndInterior < 0 ? cEnd : cEndInterior;
    ierr          = DMForestGetBaseDM(dm,&base);CHKERRQ(ierr);
    ierr          = DMGetCoordinateSection(plex,&coordSec);CHKERRQ(ierr);
    ierr          = PetscSectionGetChart(coordSec,&coordStart,&coordEnd);CHKERRQ(ierr);
    ierr          = DMGetCoordinateDim(plex,&coordDim);CHKERRQ(ierr);
    p4estCoordDim = PetscMin(coordDim,3);
    for (p = coordStart; p < coordEnd; p++) {
      PetscInt *star = NULL, starSize;
      PetscInt dof, off, cell = -1, coarsePoint = -1;
      PetscInt nCoords, i;
      ierr = PetscSectionGetDof(coordSec,p,&dof);CHKERRQ(ierr);
      if (dof % coordDim) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Did not understand coordinate layout");
      nCoords = dof / coordDim;
      ierr    = PetscSectionGetOffset(coordSec,p,&off);CHKERRQ(ierr);
      ierr    = DMPlexGetTransitiveClosure(plex,p,PETSC_FALSE,&starSize,&star);CHKERRQ(ierr);
      for (i = 0; i < starSize; i++) {
        PetscInt point = star[2 * i];

        if (cStart <= point && point < cEnd) {
          cell = point;
          break;
        }
      }
      ierr = DMPlexRestoreTransitiveClosure(plex,p,PETSC_FALSE,&starSize,&star);CHKERRQ(ierr);
      if (cell >= 0) {
        if (cell < cLocalStart) {
          p4est_quadrant_t *ghosts = (p4est_quadrant_t*) pforest->ghost->ghosts.array;

          coarsePoint = ghosts[cell].p.which_tree;
        } else if (cell < cLocalEnd) {
          cell -= cLocalStart;
          for (t = flt; t <= llt; t++) {
            p4est_tree_t *tree = &(trees[t]);

            if (cell >= tree->quadrants_offset && (size_t) cell < tree->quadrants_offset + tree->quadrants.elem_count) {
              coarsePoint = t;
              break;
            }
          }
        } else {
          p4est_quadrant_t *ghosts = (p4est_quadrant_t*) pforest->ghost->ghosts.array;

          coarsePoint = ghosts[cell - cLocalEnd].p.which_tree;
        }
      }
      for (i = 0; i < nCoords; i++) {
        PetscScalar *coord              = &coords[off + i * coordDim];
        PetscReal   coordP4est[3]       = {0.};
        PetscReal   coordP4estMapped[3] = {0.};
        PetscInt    j;

        for (j = 0; j < p4estCoordDim; j++) coordP4est[j] = PetscRealPart(coord[j]);
        ierr = (map)(base,coarsePoint,p4estCoordDim,coordP4est,coordP4estMapped,mapCtx);CHKERRQ(ierr);
        for (j = 0; j < p4estCoordDim; j++) coord[j] = (PetscScalar) coordP4estMapped[j];
      }
    }
  } else { /* we have to transform coordinates back to the unit cube (where geom is defined), and then apply geom */
    PetscInt cStart, cEnd, cEndInterior;

    ierr = DMPlexGetHeightStratum(plex,0,&cStart,&cEnd);CHKERRQ(ierr);
    ierr = DMPlexGetHybridBounds(plex,&cEndInterior,NULL,NULL,NULL);CHKERRQ(ierr);
    cEnd = cEndInterior < 0 ? cEnd : cEndInterior;
    if (cLocalStart > 0) {
      p4est_quadrant_t *ghosts = (p4est_quadrant_t*) pforest->ghost->ghosts.array;
      PetscInt         count;

      for (count = 0; count < cLocalStart; count++) {
        p4est_quadrant_t *quad = &ghosts[count];
        p4est_topidx_t   t     = quad->p.which_tree;

        ierr = DMPforestMapCoordinates_Cell(plex,geom,count,quad,t,pforest->topo->conn,coords);CHKERRQ(ierr);
      }
    }
    for (t = flt; t <= llt; t++) {
      p4est_tree_t     *tree    = &(trees[t]);
      PetscInt         offset   = cLocalStart + tree->quadrants_offset, i;
      PetscInt         numQuads = (PetscInt) tree->quadrants.elem_count;
      p4est_quadrant_t *quads   = (p4est_quadrant_t*) tree->quadrants.array;

      for (i = 0; i < numQuads; i++) {
        PetscInt count = i + offset;

        ierr = DMPforestMapCoordinates_Cell(plex,geom,count,&quads[i],t,pforest->topo->conn,coords);CHKERRQ(ierr);
      }
    }
    if (cLocalEnd - cLocalStart < cEnd - cStart) {
      p4est_quadrant_t *ghosts   = (p4est_quadrant_t*) pforest->ghost->ghosts.array;
      PetscInt         numGhosts = (PetscInt) pforest->ghost->ghosts.elem_count;
      PetscInt         count;

      for (count = 0; count < numGhosts - cLocalStart; count++) {
        p4est_quadrant_t *quad = &ghosts[count + cLocalStart];
        p4est_topidx_t   t     = quad->p.which_tree;

        ierr = DMPforestMapCoordinates_Cell(plex,geom,count + cLocalEnd,quad,t,pforest->topo->conn,coords);CHKERRQ(ierr);
      }
    }
  }
  ierr = VecRestoreArray(coordLocalVec,&coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPforestLocalizeCoordinates(DM dm, DM plex)
{
  DM_Forest         *forest;
  DM_Forest_pforest *pforest;
  DM                base;
  Vec               coordinates, cVec;
  PetscSection      oldSection, baseSection = NULL, newSection;
  const PetscScalar *coords;
  PetscScalar       *coords2;
  PetscInt          cLocalStart, cLocalEnd, coarsePoint;
  PetscInt          cDim, newStart, newEnd, dof, cdof = -1;
  PetscInt          v, vStart, vEnd, cp, cStart, cEnd, cEndInterior, *coarsePoints;
  PetscInt          *localize, overlap;
  p4est_topidx_t    flt, llt, t;
  p4est_tree_t      *trees;
  PetscBool         isper, baseLocalized = PETSC_FALSE;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMGetPeriodicity(dm,&isper,NULL,NULL,NULL);CHKERRQ(ierr);
  if (!isper) PetscFunctionReturn(0);
  /* we localize on all cells if we don't have a base DM or the base DM coordinates have not been localized */
  ierr = DMGetCoordinateDim(dm, &cDim);CHKERRQ(ierr);
  cdof = P4EST_CHILDREN*cDim;
  ierr = DMForestGetBaseDM(dm,&base);CHKERRQ(ierr);
  if (base) {
    ierr = DMGetCoordinatesLocalized(base,&baseLocalized);CHKERRQ(ierr);
  }
  if (!baseLocalized) base = NULL;
  ierr = DMPlexGetChart(plex, &newStart, &newEnd);CHKERRQ(ierr);

  ierr = DMForestGetPartitionOverlap(dm,&overlap);CHKERRQ(ierr);
  ierr = PetscCalloc1(overlap ? newEnd - newStart : 0,&localize);CHKERRQ(ierr);

  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) dm), &newSection);CHKERRQ(ierr);
  ierr = PetscSectionSetNumFields(newSection, 1);CHKERRQ(ierr);
  ierr = PetscSectionSetFieldComponents(newSection, 0, cDim);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(newSection, newStart, newEnd);CHKERRQ(ierr);

  ierr = DMGetCoordinateSection(plex, &oldSection);CHKERRQ(ierr);
  if (base) { ierr = DMGetCoordinateSection(base, &baseSection);CHKERRQ(ierr); }
  ierr = DMPlexGetDepthStratum(plex,0,&vStart,&vEnd);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    ierr = PetscSectionGetDof(oldSection, v, &dof);CHKERRQ(ierr);
    ierr = PetscSectionSetDof(newSection, v, dof);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldDof(newSection, v, 0, dof);CHKERRQ(ierr);
    if (overlap) localize[v] = dof;
  }

  forest      = (DM_Forest*) dm->data;
  pforest     = (DM_Forest_pforest*) forest->data;
  cLocalStart = pforest->cLocalStart;
  cLocalEnd   = pforest->cLocalEnd;
  flt         = pforest->forest->first_local_tree;
  llt         = pforest->forest->last_local_tree;
  trees       = (p4est_tree_t*) pforest->forest->trees->array;

  cp = 0;
  ierr = DMPlexGetHeightStratum(plex,0,&cStart,&cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetHybridBounds(plex, &cEndInterior, NULL, NULL, NULL);CHKERRQ(ierr);
  cEnd = cEndInterior < 0 ? cEnd : cEndInterior;
  ierr = PetscMalloc1(cEnd-cStart,&coarsePoints);CHKERRQ(ierr);
  if (cLocalStart > 0) {
    p4est_quadrant_t *ghosts = (p4est_quadrant_t*) pforest->ghost->ghosts.array;
    PetscInt         count;

    for (count = 0; count < cLocalStart; count++) {
      p4est_quadrant_t *quad = &ghosts[count];
      coarsePoint = quad->p.which_tree;

      if (baseSection) { ierr = PetscSectionGetFieldDof(baseSection, coarsePoint, 0, &cdof);CHKERRQ(ierr); }
      ierr = PetscSectionSetDof(newSection, count, cdof);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(newSection, count, 0, cdof);CHKERRQ(ierr);
      coarsePoints[cp++] = cdof ? coarsePoint : -1;
      if (overlap) localize[count] = cdof;
    }
  }
  for (t = flt; t <= llt; t++) {
    p4est_tree_t *tree    = &(trees[t]);
    PetscInt     offset   = cLocalStart + tree->quadrants_offset;
    PetscInt     numQuads = (PetscInt) tree->quadrants.elem_count;
    PetscInt     i;

    if (!numQuads) continue;
    coarsePoint = t;
    if (baseSection) { ierr = PetscSectionGetFieldDof(baseSection, coarsePoint, 0, &cdof);CHKERRQ(ierr); }
    for (i = 0; i < numQuads; i++) {
      PetscInt newCell = i + offset;

      ierr = PetscSectionSetDof(newSection, newCell, cdof);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(newSection, newCell, 0, cdof);CHKERRQ(ierr);
      coarsePoints[cp++] = cdof ? coarsePoint : -1;
      if (overlap) localize[newCell] = cdof;
    }
  }
  if (cLocalEnd - cLocalStart < cEnd - cStart) {
    p4est_quadrant_t *ghosts   = (p4est_quadrant_t*) pforest->ghost->ghosts.array;
    PetscInt         numGhosts = (PetscInt) pforest->ghost->ghosts.elem_count;
    PetscInt         count;

    for (count = 0; count < numGhosts - cLocalStart; count++) {
      p4est_quadrant_t *quad = &ghosts[count + cLocalStart];
      coarsePoint = quad->p.which_tree;
      PetscInt newCell = count + cLocalEnd;

      if (baseSection) { ierr = PetscSectionGetFieldDof(baseSection, coarsePoint, 0, &cdof);CHKERRQ(ierr); }
      ierr = PetscSectionSetDof(newSection, newCell, cdof);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(newSection, newCell, 0, cdof);CHKERRQ(ierr);
      coarsePoints[cp++] = cdof ? coarsePoint : -1;
      if (overlap) localize[newCell] = cdof;
    }
  }
  if (cp != cEnd - cStart) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected number of fine cells %D != %D",cp,cEnd-cStart);

  if (base) { /* we need to localize on all the cells in the star of the coarse cell vertices */
    PetscInt *closure = NULL, closureSize;
    PetscInt p, i, c, vStartBase, vEndBase, cStartBase, cEndBase;

    ierr = DMPlexGetHeightStratum(base,0,&cStartBase,&cEndBase);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(base,0,&vStartBase,&vEndBase);CHKERRQ(ierr);
    for (p = cStart; p < cEnd; p++) {
      coarsePoint = coarsePoints[p-cStart];
      if (coarsePoint < 0) continue;
      if (baseSection) { ierr = PetscSectionGetFieldDof(baseSection, coarsePoint, 0, &cdof);CHKERRQ(ierr); }
      ierr = DMPlexGetTransitiveClosure(base,coarsePoint,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
      for (c = 0; c < closureSize; c++) {
        PetscInt *star = NULL, starSize;
        PetscInt j, v = closure[2 * c];

        if (v < vStartBase || v > vEndBase) continue;
        ierr = DMPlexGetTransitiveClosure(base,v,PETSC_FALSE,&starSize,&star);CHKERRQ(ierr);
        for (j = 0; j < starSize; j++) {
          PetscInt cell = star[2 * j];

          if (cStartBase <= cell && cell < cEndBase) {
            p4est_tree_t *tree;
            PetscInt     offset,numQuads;

            if (cell < flt || cell > llt) continue;
            tree     = &(trees[cell]);
            offset   = cLocalStart + tree->quadrants_offset;
            numQuads = (PetscInt) tree->quadrants.elem_count;
            for (i = 0; i < numQuads; i++) {
              PetscInt newCell = i + offset;

              ierr = PetscSectionSetDof(newSection, newCell, cdof);CHKERRQ(ierr);
              ierr = PetscSectionSetFieldDof(newSection, newCell, 0, cdof);CHKERRQ(ierr);
              if (overlap) localize[newCell] = cdof;
            }
          }
        }
        ierr = DMPlexRestoreTransitiveClosure(base,v,PETSC_FALSE,&starSize,&star);CHKERRQ(ierr);
      }
      ierr = DMPlexRestoreTransitiveClosure(base,coarsePoint,PETSC_TRUE,&closureSize,&closure);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(coarsePoints);CHKERRQ(ierr);

  /* final consensus with overlap */
  if (overlap) {
    PetscSF  sf;
    PetscInt *localizeGlobal;

    ierr = DMGetPointSF(plex,&sf);CHKERRQ(ierr);
    ierr = PetscMalloc1(newEnd-newStart,&localizeGlobal);CHKERRQ(ierr);
    for (v = newStart; v < newEnd; v++) localizeGlobal[v - newStart] = localize[v - newStart];
    ierr = PetscSFBcastBegin(sf,MPIU_INT,localize,localizeGlobal);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf,MPIU_INT,localize,localizeGlobal);CHKERRQ(ierr);
    for (v = newStart; v < newEnd; v++) {
      ierr = PetscSectionSetDof(newSection, v, localizeGlobal[v-newStart]);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(newSection, v, 0, localizeGlobal[v-newStart]);CHKERRQ(ierr);
    }
    ierr = PetscFree(localizeGlobal);CHKERRQ(ierr);
  }
  ierr = PetscFree(localize);CHKERRQ(ierr);
  ierr = PetscSectionSetUp(newSection);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)oldSection);CHKERRQ(ierr);
  ierr = DMSetCoordinateSection(plex, cDim, newSection);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(newSection, &v);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF, &cVec);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)cVec,"coordinates");CHKERRQ(ierr);
  ierr = VecSetBlockSize(cVec, cDim);CHKERRQ(ierr);
  ierr = VecSetSizes(cVec, v, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSetType(cVec, VECSTANDARD);CHKERRQ(ierr);
  ierr = VecSet(cVec, PETSC_MIN_REAL);CHKERRQ(ierr);

  /* Copy over vertex coordinates */
  ierr = DMGetCoordinatesLocal(plex, &coordinates);CHKERRQ(ierr);
  if (!coordinates) SETERRQ(PetscObjectComm((PetscObject)plex),PETSC_ERR_SUP,"Missing local coordinates vector");
  ierr = VecGetArray(cVec, &coords2);CHKERRQ(ierr);
  ierr = VecGetArrayRead(coordinates, &coords);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {
    PetscInt d, off,off2;

    ierr = PetscSectionGetDof(oldSection, v, &dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(oldSection, v, &off);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(newSection, v, &off2);CHKERRQ(ierr);
    for (d = 0; d < dof; ++d) coords2[off2+d] = coords[off+d];
  }
  ierr = VecRestoreArrayRead(coordinates, &coords);CHKERRQ(ierr);

  /* Localize coordinates on cells if needed */
  for (t = flt; t <= llt; t++) {
    p4est_tree_t     *tree    = &(trees[t]);
    const double     *v       = pforest->topo->conn->vertices;
    p4est_quadrant_t *quads   = (p4est_quadrant_t*) tree->quadrants.array;
    PetscInt         offset   = cLocalStart + tree->quadrants_offset;
    PetscInt         numQuads = (PetscInt) tree->quadrants.elem_count;
    p4est_topidx_t   vt[8]    = {0,0,0,0,0,0,0,0};
    PetscInt         i,k;

    if (!numQuads) continue;
    for (k = 0; k < P4EST_CHILDREN; ++k) {
      vt[k] = pforest->topo->conn->tree_to_vertex[t * P4EST_CHILDREN + k];
    }

    for (i = 0; i < numQuads; i++) {
      p4est_quadrant_t  *quad = &quads[i];
      const PetscReal   intsize = 1.0 / P4EST_ROOT_LEN;
      PetscReal         h2;
      PetscScalar       xyz[3];
#ifdef P4_TO_P8
      PetscInt          zi;
#endif
      PetscInt          yi,xi;
      PetscInt          off2;
      PetscInt          newCell = i + offset;

      ierr = PetscSectionGetFieldDof(newSection, newCell, 0, &cdof);CHKERRQ(ierr);
      if (!cdof) continue;

      h2   = .5 * intsize * P4EST_QUADRANT_LEN (quad->level);
      k    = 0;
      ierr = PetscSectionGetOffset(newSection, newCell, &off2);CHKERRQ(ierr);
#ifdef P4_TO_P8
      for (zi = 0; zi < 2; ++zi) {
        const PetscReal eta_z = intsize * quad->z + h2 * (1. + (zi * 2 - 1));
#else
      {
        const PetscReal eta_z = 0.0;
#endif
        for (yi = 0; yi < 2; ++yi) {
          const PetscReal eta_y = intsize * quad->y + h2 * (1. + (yi * 2 - 1));
          for (xi = 0; xi < 2; ++xi) {
            const PetscReal eta_x = intsize * quad->x + h2 * (1. + (xi * 2 - 1));
            PetscInt    j;

            for (j = 0; j < 3; ++j) {
              xyz[j] = ((1. - eta_z) * ((1. - eta_y) * ((1. - eta_x) * v[3 * vt[0] + j] +
                                                              eta_x  * v[3 * vt[1] + j]) +
                                              eta_y  * ((1. - eta_x) * v[3 * vt[2] + j] +
                                                              eta_x  * v[3 * vt[3] + j]))
                        +     eta_z  * ((1. - eta_y) * ((1. - eta_x) * v[3 * vt[4] + j] +
                                                              eta_x  * v[3 * vt[5] + j]) +
                                              eta_y  * ((1. - eta_x) * v[3 * vt[6] + j] +
                                                              eta_x  * v[3 * vt[7] + j])));
            }
            for (j = 0; j < cDim; ++j) coords2[off2 + cDim*P4estVertToPetscVert[k] + j] = xyz[j];
            ++k;
          }
        }
      }
    }
  }
  ierr = VecRestoreArray(cVec, &coords2);CHKERRQ(ierr);
  ierr = DMSetCoordinatesLocal(plex, cVec);CHKERRQ(ierr);
  ierr = VecDestroy(&cVec);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&newSection);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&oldSection);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define DMForestClearAdaptivityForest_pforest _append_pforest(DMForestClearAdaptivityForest)
static PetscErrorCode DMForestClearAdaptivityForest_pforest(DM dm)
{
  DM_Forest         *forest;
  DM_Forest_pforest *pforest;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  forest  = (DM_Forest*) dm->data;
  pforest = (DM_Forest_pforest *) forest->data;
  ierr = PetscSFDestroy(&(pforest->pointAdaptToSelfSF));CHKERRQ(ierr);
  ierr = PetscSFDestroy(&(pforest->pointSelfToAdaptSF));CHKERRQ(ierr);
  ierr = PetscFree(pforest->pointAdaptToSelfCids);CHKERRQ(ierr);
  ierr = PetscFree(pforest->pointSelfToAdaptCids);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMConvert_pforest_plex(DM dm, DMType newtype, DM *plex)
{
  DM_Forest            *forest;
  DM_Forest_pforest    *pforest;
  DM                   refTree, newPlex, base;
  PetscInt             adjDim, adjCodim, coordDim;
  MPI_Comm             comm;
  PetscBool            isPforest;
  PetscInt             dim;
  PetscInt             overlap;
  p4est_connect_type_t ctype;
  p4est_locidx_t       first_local_quad = -1;
  sc_array_t           *points_per_dim, *cone_sizes, *cones, *cone_orientations, *coords, *children, *parents, *childids, *leaves, *remotes;
  PetscSection         parentSection;
  PetscSF              pointSF;
  size_t               zz, count;
  PetscInt             pStart, pEnd;
  DMLabel              ghostLabelBase = NULL;
  PetscErrorCode       ierr;

  PetscFunctionBegin;

  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  comm = PetscObjectComm((PetscObject)dm);
  ierr = PetscObjectTypeCompare((PetscObject)dm,DMPFOREST,&isPforest);CHKERRQ(ierr);
  if (!isPforest) SETERRQ2(comm,PETSC_ERR_ARG_WRONG,"Expected DM type %s, got %s",DMPFOREST,((PetscObject)dm)->type_name);
  ierr = DMGetDimension(dm,&dim);CHKERRQ(ierr);
  if (dim != P4EST_DIM) SETERRQ2(comm,PETSC_ERR_ARG_WRONG,"Expected DM dimension %d, got %d",P4EST_DIM,dim);
  forest  = (DM_Forest*) dm->data;
  pforest = (DM_Forest_pforest*) forest->data;
  ierr    = DMForestGetBaseDM(dm,&base);CHKERRQ(ierr);
  if (base) {
    ierr = DMGetLabel(base,"ghost",&ghostLabelBase);CHKERRQ(ierr);
  }
  if (!pforest->plex) {
    PetscMPIInt size;

    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
    ierr = DMCreate(comm,&newPlex);CHKERRQ(ierr);
    ierr = DMSetType(newPlex,DMPLEX);CHKERRQ(ierr);
    ierr = DMSetMatType(newPlex,dm->mattype);CHKERRQ(ierr);
    ierr = PetscFree(newPlex->labels);CHKERRQ(ierr); /* share labels */
    dm->labels->refct++;
    newPlex->labels = dm->labels;
    ierr            = DMForestGetAdjacencyDimension(dm,&adjDim);CHKERRQ(ierr);
    ierr            = DMForestGetAdjacencyCodimension(dm,&adjCodim);CHKERRQ(ierr);
    ierr            = DMGetCoordinateDim(dm,&coordDim);CHKERRQ(ierr);
    if (adjDim == 0) {
      ctype = P4EST_CONNECT_FULL;
    } else if (adjCodim == 1) {
      ctype = P4EST_CONNECT_FACE;
#if defined(P4_TO_P8)
    } else if (adjDim == 1) {
      ctype = P8EST_CONNECT_EDGE;
#endif
    } else {
      SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONG,"Invalid adjacency dimension %d",adjDim);
    }
    if (ctype != P4EST_CONNECT_FULL) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONG,"Adjacency dimension %D / codimension %D not supported yet",adjDim,adjCodim);
    ierr = DMForestGetPartitionOverlap(dm,&overlap);CHKERRQ(ierr);

    points_per_dim    = sc_array_new(sizeof(p4est_locidx_t));
    cone_sizes        = sc_array_new(sizeof(p4est_locidx_t));
    cones             = sc_array_new(sizeof(p4est_locidx_t));
    cone_orientations = sc_array_new(sizeof(p4est_locidx_t));
    coords            = sc_array_new(3 * sizeof(double));
    children          = sc_array_new(sizeof(p4est_locidx_t));
    parents           = sc_array_new(sizeof(p4est_locidx_t));
    childids          = sc_array_new(sizeof(p4est_locidx_t));
    leaves            = sc_array_new(sizeof(p4est_locidx_t));
    remotes           = sc_array_new(2 * sizeof(p4est_locidx_t));

    PetscStackCallP4est(p4est_get_plex_data_ext,(pforest->forest,&pforest->ghost,&pforest->lnodes,ctype,(int)((size > 1) ? overlap : 0),&first_local_quad,points_per_dim,cone_sizes,cones,cone_orientations,coords,children,parents,childids,leaves,remotes,1));

    pforest->cLocalStart = (PetscInt) first_local_quad;
    pforest->cLocalEnd   = pforest->cLocalStart + (PetscInt) pforest->forest->local_num_quadrants;
    ierr                 = locidx_to_PetscInt(points_per_dim);CHKERRQ(ierr);
    ierr                 = locidx_to_PetscInt(cone_sizes);CHKERRQ(ierr);
    ierr                 = locidx_to_PetscInt(cones);CHKERRQ(ierr);
    ierr                 = locidx_to_PetscInt(cone_orientations);CHKERRQ(ierr);
    ierr                 = coords_double_to_PetscScalar(coords, coordDim);CHKERRQ(ierr);
    ierr                 = locidx_to_PetscInt(children);CHKERRQ(ierr);
    ierr                 = locidx_to_PetscInt(parents);CHKERRQ(ierr);
    ierr                 = locidx_to_PetscInt(childids);CHKERRQ(ierr);
    ierr                 = locidx_to_PetscInt(leaves);CHKERRQ(ierr);
    ierr                 = locidx_pair_to_PetscSFNode(remotes);CHKERRQ(ierr);

    ierr  = DMSetDimension(newPlex,P4EST_DIM);CHKERRQ(ierr);
    ierr  = DMSetCoordinateDim(newPlex,coordDim);CHKERRQ(ierr);
    ierr  = DMPlexSetMaxProjectionHeight(newPlex,P4EST_DIM - 1);CHKERRQ(ierr);
    ierr  = DMPlexCreateFromDAG(newPlex,P4EST_DIM,(PetscInt*)points_per_dim->array,(PetscInt*)cone_sizes->array,(PetscInt*)cones->array,(PetscInt*)cone_orientations->array,(PetscScalar*)coords->array);CHKERRQ(ierr);
    ierr  = DMCreateReferenceTree_pforest(comm,&refTree);CHKERRQ(ierr);
    ierr  = DMPlexSetReferenceTree(newPlex,refTree);CHKERRQ(ierr);
    ierr  = PetscSectionCreate(comm,&parentSection);CHKERRQ(ierr);
    ierr  = DMPlexGetChart(newPlex,&pStart,&pEnd);CHKERRQ(ierr);
    ierr  = PetscSectionSetChart(parentSection,pStart,pEnd);CHKERRQ(ierr);
    count = children->elem_count;
    for (zz = 0; zz < count; zz++) {
      PetscInt child = *((PetscInt*) sc_array_index(children,zz));

      ierr = PetscSectionSetDof(parentSection,child,1);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(parentSection);CHKERRQ(ierr);
    ierr = DMPlexSetTree(newPlex,parentSection,(PetscInt*)parents->array,(PetscInt*)childids->array);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&parentSection);CHKERRQ(ierr);
    ierr = PetscSFCreate(comm,&pointSF);CHKERRQ(ierr);
    ierr = PetscSFSetGraph(pointSF,pEnd - pStart,(PetscInt)leaves->elem_count,(PetscInt*)leaves->array,PETSC_COPY_VALUES,(PetscSFNode*)remotes->array,PETSC_COPY_VALUES);CHKERRQ(ierr);
    ierr = DMSetPointSF(newPlex,pointSF);CHKERRQ(ierr);
    ierr = DMSetPointSF(dm,pointSF);CHKERRQ(ierr);
    {
      DM coordDM;

      ierr = DMGetCoordinateDM(newPlex,&coordDM);CHKERRQ(ierr);
      ierr = DMSetPointSF(coordDM,pointSF);CHKERRQ(ierr);
    }
    ierr = PetscSFDestroy(&pointSF);CHKERRQ(ierr);
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

    {
      PetscBool             isper;
      const PetscReal      *maxCell, *L;
      const DMBoundaryType *bd;

      ierr = DMGetPeriodicity(dm,&isper,&maxCell,&L,&bd);CHKERRQ(ierr);
      ierr = DMSetPeriodicity(newPlex,isper,maxCell,L,bd);CHKERRQ(ierr);
      ierr = DMPforestLocalizeCoordinates(dm,newPlex);CHKERRQ(ierr);
    }

    if (overlap > 0) { /* the p4est routine can't set all of the coordinates in its routine if there is overlap */
      Vec               coordsGlobal, coordsLocal;
      const PetscScalar *globalArray;
      PetscScalar       *localArray;
      PetscSF           coordSF;
      DM                coordDM;

      ierr = DMGetCoordinateDM(newPlex,&coordDM);CHKERRQ(ierr);
      ierr = DMGetSectionSF(coordDM,&coordSF);CHKERRQ(ierr);
      ierr = DMGetCoordinates(newPlex, &coordsGlobal);CHKERRQ(ierr);
      ierr = DMGetCoordinatesLocal(newPlex, &coordsLocal);CHKERRQ(ierr);
      ierr = VecGetArrayRead(coordsGlobal, &globalArray);CHKERRQ(ierr);
      ierr = VecGetArray(coordsLocal, &localArray);CHKERRQ(ierr);
      ierr = PetscSFBcastBegin(coordSF,MPIU_SCALAR,globalArray,localArray);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(coordSF,MPIU_SCALAR,globalArray,localArray);CHKERRQ(ierr);
      ierr = VecRestoreArray(coordsLocal, &localArray);CHKERRQ(ierr);
      ierr = VecRestoreArrayRead(coordsGlobal, &globalArray);CHKERRQ(ierr);
      ierr = DMSetCoordinatesLocal(newPlex, coordsLocal);CHKERRQ(ierr);
    }
    ierr = DMPforestMapCoordinates(dm,newPlex);CHKERRQ(ierr);

    pforest->plex = newPlex;

    /* copy labels */
    ierr = DMPforestLabelsFinalize(dm,newPlex);CHKERRQ(ierr);

    if (ghostLabelBase || pforest->ghostName) { /* we have to do this after copying labels because the labels drive the construction of ghost cells */
      PetscInt numAdded;
      DM       newPlexGhosted;
      void     *ctx;

      ierr = DMPlexConstructGhostCells(newPlex,pforest->ghostName,&numAdded,&newPlexGhosted);CHKERRQ(ierr);
      ierr = DMGetApplicationContext(newPlex,&ctx);CHKERRQ(ierr);
      ierr = DMSetApplicationContext(newPlexGhosted,ctx);CHKERRQ(ierr);
      /* we want the sf for the ghost dm to be the one for the p4est dm as well */
      ierr    = DMGetPointSF(newPlexGhosted,&pointSF);CHKERRQ(ierr);
      ierr    = DMSetPointSF(dm,pointSF);CHKERRQ(ierr);
      ierr    = DMDestroy(&newPlex);CHKERRQ(ierr);
      ierr    = DMPlexSetReferenceTree(newPlexGhosted,refTree);CHKERRQ(ierr);
      ierr    = DMForestClearAdaptivityForest_pforest(dm);CHKERRQ(ierr);
      newPlex = newPlexGhosted;

      /* share the labels back */
      ierr = DMDestroyLabelLinkList_Internal(dm);CHKERRQ(ierr);
      newPlex->labels->refct++;
      dm->labels = newPlex->labels;

      pforest->plex = newPlex;
    }
    ierr  = DMDestroy(&refTree);CHKERRQ(ierr);
    if (dm->setfromoptionscalled) {
      ierr = PetscObjectOptionsBegin((PetscObject)newPlex);CHKERRQ(ierr);
      ierr = DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject,newPlex);CHKERRQ(ierr);
      ierr = PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject) newPlex);CHKERRQ(ierr);
      ierr = PetscOptionsEnd();CHKERRQ(ierr);
    }
    ierr = DMViewFromOptions(newPlex,NULL,"-dm_p4est_plex_view");CHKERRQ(ierr);
    {
      PetscSection coordsSec;
      Vec          coords;
      PetscInt     cDim;

      ierr = DMGetCoordinateDim(newPlex,&cDim);CHKERRQ(ierr);
      ierr = DMGetCoordinateSection(newPlex,&coordsSec);CHKERRQ(ierr);
      ierr = DMSetCoordinateSection(dm,cDim,coordsSec);CHKERRQ(ierr);
      ierr = DMGetCoordinatesLocal(newPlex,&coords);CHKERRQ(ierr);
      ierr = DMSetCoordinatesLocal(dm,coords);CHKERRQ(ierr);
    }
  }
  newPlex = pforest->plex;
  if (plex) {
    DM coordDM;

    ierr = DMClone(newPlex,plex);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(newPlex,&coordDM);CHKERRQ(ierr);
    ierr = DMSetCoordinateDM(*plex,coordDM);CHKERRQ(ierr);
    ierr = DMShareDiscretization(dm,*plex);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSetFromOptions_pforest(PetscOptionItems *PetscOptionsObject,DM dm)
{
  DM_Forest_pforest *pforest = (DM_Forest_pforest*) ((DM_Forest*) dm->data)->data;
  char              stringBuffer[256];
  PetscBool         flg;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMSetFromOptions_Forest(PetscOptionsObject,dm);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"DM" P4EST_STRING " options");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-dm_p4est_partition_for_coarsening","partition forest to allow for coarsening","DMP4estSetPartitionForCoarsening",pforest->partition_for_coarsening,&(pforest->partition_for_coarsening),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-dm_p4est_ghost_label_name","the name of the ghost label when converting from a DMPlex",NULL,NULL,stringBuffer,256,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if (flg) {
    ierr = PetscFree(pforest->ghostName);CHKERRQ(ierr);
    ierr = PetscStrallocpy(stringBuffer,&pforest->ghostName);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#if !defined(P4_TO_P8)
#define DMPforestGetPartitionForCoarsening DMP4estGetPartitionForCoarsening
#define DMPforestSetPartitionForCoarsening DMP4estSetPartitionForCoarsening
#else
#define DMPforestGetPartitionForCoarsening DMP8estGetPartitionForCoarsening
#define DMPforestSetPartitionForCoarsening DMP8estSetPartitionForCoarsening
#endif

PETSC_EXTERN PetscErrorCode DMPforestGetPartitionForCoarsening(DM dm, PetscBool *flg)
{
  DM_Forest_pforest *pforest;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  pforest = (DM_Forest_pforest*) ((DM_Forest*) dm->data)->data;
  *flg    = pforest->partition_for_coarsening;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMPforestSetPartitionForCoarsening(DM dm, PetscBool flg)
{
  DM_Forest_pforest *pforest;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  pforest                           = (DM_Forest_pforest*) ((DM_Forest*) dm->data)->data;
  pforest->partition_for_coarsening = flg;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPforestGetPlex(DM dm,DM *plex)
{
  DM_Forest_pforest *pforest;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (plex) *plex = NULL;
  ierr    = DMSetUp(dm);CHKERRQ(ierr);
  pforest = (DM_Forest_pforest*) ((DM_Forest*) dm->data)->data;
  if (!pforest->plex) {
    ierr = DMConvert_pforest_plex(dm,DMPLEX,NULL);CHKERRQ(ierr);
  }
  ierr = DMShareDiscretization(dm,pforest->plex);CHKERRQ(ierr);
  if (plex) *plex = pforest->plex;
  PetscFunctionReturn(0);
}

#define DMCreateInterpolation_pforest _append_pforest(DMCreateInterpolation)
static PetscErrorCode DMCreateInterpolation_pforest(DM dmCoarse, DM dmFine, Mat *interpolation, Vec *scaling)
{
  PetscSection   gsc, gsf;
  PetscInt       m, n;
  DM             cdm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetGlobalSection(dmFine, &gsf);CHKERRQ(ierr);
  ierr = PetscSectionGetConstrainedStorageSize(gsf, &m);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dmCoarse, &gsc);CHKERRQ(ierr);
  ierr = PetscSectionGetConstrainedStorageSize(gsc, &n);CHKERRQ(ierr);

  ierr = MatCreate(PetscObjectComm((PetscObject) dmFine), interpolation);CHKERRQ(ierr);
  ierr = MatSetSizes(*interpolation, m, n, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(*interpolation, MATAIJ);CHKERRQ(ierr);

  ierr = DMGetCoarseDM(dmFine, &cdm);CHKERRQ(ierr);
  if (cdm != dmCoarse) SETERRQ(PetscObjectComm((PetscObject)dmFine),PETSC_ERR_SUP,"Only interpolation from coarse DM for now");

  {
    DM       plexF, plexC;
    PetscSF  sf;
    PetscInt *cids;
    PetscInt dofPerDim[4] = {1,1,1,1};

    ierr = DMPforestGetPlex(dmCoarse,&plexC);CHKERRQ(ierr);
    ierr = DMPforestGetPlex(dmFine,&plexF);CHKERRQ(ierr);
    ierr = DMPforestGetTransferSF_Internal(dmCoarse, dmFine, dofPerDim, &sf, PETSC_TRUE, &cids);CHKERRQ(ierr);
    ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
    ierr = DMPlexComputeInterpolatorTree(plexC, plexF, sf, cids, *interpolation);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
    ierr = PetscFree(cids);
  }
  ierr = MatViewFromOptions(*interpolation, NULL, "-interp_mat_view");CHKERRQ(ierr);
  /* Use naive scaling */
  ierr = DMCreateInterpolationScale(dmCoarse, dmFine, *interpolation, scaling);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define DMCreateInjection_pforest _append_pforest(DMCreateInjection)
static PetscErrorCode DMCreateInjection_pforest(DM dmCoarse, DM dmFine, Mat *injection)
{
  PetscSection   gsc, gsf;
  PetscInt       m, n;
  DM             cdm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetGlobalSection(dmFine, &gsf);CHKERRQ(ierr);
  ierr = PetscSectionGetConstrainedStorageSize(gsf, &n);CHKERRQ(ierr);
  ierr = DMGetGlobalSection(dmCoarse, &gsc);CHKERRQ(ierr);
  ierr = PetscSectionGetConstrainedStorageSize(gsc, &m);CHKERRQ(ierr);

  ierr = MatCreate(PetscObjectComm((PetscObject) dmFine), injection);CHKERRQ(ierr);
  ierr = MatSetSizes(*injection, m, n, PETSC_DETERMINE, PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetType(*injection, MATAIJ);CHKERRQ(ierr);

  ierr = DMGetCoarseDM(dmFine, &cdm);CHKERRQ(ierr);
  if (cdm != dmCoarse) SETERRQ(PetscObjectComm((PetscObject)dmFine),PETSC_ERR_SUP,"Only injection to coarse DM for now");

  {
    DM       plexF, plexC;
    PetscSF  sf;
    PetscInt *cids;
    PetscInt dofPerDim[4] = {1,1,1,1};

    ierr = DMPforestGetPlex(dmCoarse,&plexC);CHKERRQ(ierr);
    ierr = DMPforestGetPlex(dmFine,&plexF);CHKERRQ(ierr);
    ierr = DMPforestGetTransferSF_Internal(dmCoarse, dmFine, dofPerDim, &sf, PETSC_TRUE, &cids);CHKERRQ(ierr);
    ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
    ierr = DMPlexComputeInjectorTree(plexC, plexF, sf, cids, *injection);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
    ierr = PetscFree(cids);
  }
  ierr = MatViewFromOptions(*injection, NULL, "-inject_mat_view");CHKERRQ(ierr);
  /* Use naive scaling */
  PetscFunctionReturn(0);
}

static void transfer_func_0(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                            const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                            const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                            PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar uexact[])
{
  PetscInt f = 0; /* I would like to have f = (PetscInt)(*ctx) */
  PetscInt oa, ou;
  for (ou = 0, oa = aOff[f]; oa < aOff[f+1]; ou++, oa++) uexact[ou] = a[oa];
}

static void transfer_func_1(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                            const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                            const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                            PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar uexact[])
{
  PetscInt f = 1;
  PetscInt oa, ou;
  for (ou = 0, oa = aOff[f]; oa < aOff[f+1]; ou++, oa++) uexact[ou] = a[oa];
}

static void transfer_func_2(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                            const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                            const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                            PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar uexact[])
{
  PetscInt f = 2;
  PetscInt oa, ou;
  for (ou = 0, oa = aOff[f]; oa < aOff[f+1]; ou++, oa++) uexact[ou] = a[oa];
}

#define DMForestTransferVecFromBase_pforest _append_pforest(DMForestTransferVecFromBase)
static PetscErrorCode DMForestTransferVecFromBase_pforest(DM dm, Vec vecIn, Vec vecOut)
{
  DM             dmIn, dmVecIn, base, basec, plex, dmAux, coarseDM;
  DM             *hierarchy;
  PetscSF        sfRed = NULL;
  PetscDS        ds;
  Vec            dummy, vecInLocal, vecOutLocal;
  DMLabel        subpointMap;
  PetscInt       minLevel, mh, n_hi, i;
  PetscBool      hiforest, *hierarchy_forest;
  PetscErrorCode ierr;
  void           (*funcs[3]) (PetscInt dim, PetscInt Nf, PetscInt NfAux,
                              const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                              const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                              PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar uexact[]) = {transfer_func_0,transfer_func_1,transfer_func_2};

  PetscFunctionBegin;
  ierr = VecGetDM(vecIn,&dmVecIn);CHKERRQ(ierr);
  ierr = DMGetDS(dmVecIn,&ds);CHKERRQ(ierr);
  if (!ds) SETERRQ(PetscObjectComm((PetscObject)dmVecIn),PETSC_ERR_SUP,"Cannot transfer without a PetscDS object");CHKERRQ(ierr);
  { /* we cannot stick user contexts into function callbacks for DMProjectFieldLocal! */
    PetscSection section;
    PetscInt     Nf;

    ierr = DMGetLocalSection(dmVecIn,&section);CHKERRQ(ierr);
    ierr = PetscSectionGetNumFields(section,&Nf);CHKERRQ(ierr);
    if (Nf > 3) SETERRQ1(PetscObjectComm((PetscObject)dmVecIn),PETSC_ERR_SUP,"Number of fields %D are currently not supported! Send an email at petsc-dev@mcs.anl.gov",Nf);CHKERRQ(ierr);
  }
  ierr = DMForestGetMinimumRefinement(dm,&minLevel);CHKERRQ(ierr);
  if (minLevel) SETERRQ1(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Cannot transfer with minimum refinement set to %D. Rerun with DMForestSetMinimumRefinement(dm,0)",minLevel);CHKERRQ(ierr);
  ierr = DMForestGetBaseDM(dm,&base);CHKERRQ(ierr);
  if (!base) SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Missing base DM");CHKERRQ(ierr);

  ierr = VecSet(vecOut,0.0);CHKERRQ(ierr);
  if (dmVecIn == base) { /* sequential runs */
    ierr = PetscObjectReference((PetscObject)vecIn);CHKERRQ(ierr);
  } else {
    PetscSection secIn, secInRed;
    Vec          vecInRed, vecInLocal;

    ierr = PetscObjectQuery((PetscObject)base,"_base_migration_sf",(PetscObject*)&sfRed);CHKERRQ(ierr);
    if (!sfRed) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Not the DM set with DMForestSetBaseDM()");
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject)dmVecIn),&secInRed);CHKERRQ(ierr);
    ierr = VecCreate(PETSC_COMM_SELF,&vecInRed);CHKERRQ(ierr);
    ierr = DMGetLocalSection(dmVecIn,&secIn);CHKERRQ(ierr);
    ierr = DMGetLocalVector(dmVecIn,&vecInLocal);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(dmVecIn,vecIn,INSERT_VALUES,vecInLocal);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(dmVecIn,vecIn,INSERT_VALUES,vecInLocal);CHKERRQ(ierr);
    ierr = DMPlexDistributeField(dmVecIn,sfRed,secIn,vecInLocal,secInRed,vecInRed);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dmVecIn,&vecInLocal);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&secInRed);CHKERRQ(ierr);
    vecIn = vecInRed;
  }

  /* we first search through the AdaptivityForest hierarchy
     once we found the first disconnected forest, we upsweep the DM hierarchy */
  hiforest = PETSC_TRUE;

  /* upsweep to the coarsest DM */
  n_hi = 0;
  coarseDM = dm;
  do {
    PetscBool isforest;

    dmIn = coarseDM;
    /* need to call DMSetUp to have the hierarchy recursively setup */
    ierr = DMSetUp(dmIn);CHKERRQ(ierr);
    ierr = DMIsForest(dmIn,&isforest);CHKERRQ(ierr);
    if (!isforest) SETERRQ1(PetscObjectComm((PetscObject)dmIn),PETSC_ERR_SUP,"Cannot currently transfer through a mixed hierarchy! Found DM type %s",((PetscObject)dmIn)->type_name);
    coarseDM = NULL;
    if (hiforest) {
      ierr = DMForestGetAdaptivityForest(dmIn,&coarseDM);CHKERRQ(ierr);
    }
    if (!coarseDM) { /* DMForest hierarchy ended, we keep upsweeping through the DM hierarchy */
      hiforest = PETSC_FALSE;
      ierr = DMGetCoarseDM(dmIn,&coarseDM);CHKERRQ(ierr);
    }
    n_hi++;
  } while (coarseDM);

  ierr = PetscMalloc2(n_hi,&hierarchy,n_hi,&hierarchy_forest);CHKERRQ(ierr);

  i = 0;
  hiforest = PETSC_TRUE;
  coarseDM = dm;
  do {
    dmIn = coarseDM;
    coarseDM = NULL;
    if (hiforest) {
      ierr = DMForestGetAdaptivityForest(dmIn,&coarseDM);CHKERRQ(ierr);
    }
    if (!coarseDM) { /* DMForest hierarchy ended, we keep upsweeping through the DM hierarchy */
      hiforest = PETSC_FALSE;
      ierr = DMGetCoarseDM(dmIn,&coarseDM);CHKERRQ(ierr);
    }
    i++;
    hierarchy[n_hi - i] = dmIn;
  } while (coarseDM);

  /* project base vector on the coarsest forest (minimum refinement = 0) */
  ierr = DMPforestGetPlex(dmIn,&plex);CHKERRQ(ierr);

  /* Check this plex is compatible with the base */
  {
    IS       gnum[2];
    PetscInt ncells[2],gncells[2];

    ierr = DMPlexGetCellNumbering(base,&gnum[0]);CHKERRQ(ierr);
    ierr = DMPlexGetCellNumbering(plex,&gnum[1]);CHKERRQ(ierr);
    ierr = ISGetMinMax(gnum[0],NULL,&ncells[0]);CHKERRQ(ierr);
    ierr = ISGetMinMax(gnum[1],NULL,&ncells[1]);CHKERRQ(ierr);
    ierr = MPIU_Allreduce(ncells,gncells,2,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)dm));CHKERRQ(ierr);
    if (gncells[0] != gncells[1]) SETERRQ2(PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Invalid number of base cells! Expected %D, found %D",gncells[0]+1,gncells[1]+1);
  }
  ierr = PetscObjectQuery((PetscObject)plex,"dmAux",(PetscObject*)&dmAux);CHKERRQ(ierr);
  if (dmAux) SETERRQ(PetscObjectComm((PetscObject)dmAux),PETSC_ERR_SUP,"Cannot currently transfer from base when a dmAux is present");

  ierr = DMGetLabel(dmIn,"_forest_base_subpoint_map",&subpointMap);CHKERRQ(ierr);
  if (!subpointMap) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing _forest_base_subpoint_map label");CHKERRQ(ierr);

  ierr = DMPlexGetMaxProjectionHeight(base,&mh);CHKERRQ(ierr);
  ierr = DMPlexSetMaxProjectionHeight(plex,mh);CHKERRQ(ierr);

  ierr = DMClone(base,&basec);CHKERRQ(ierr);
  ierr = DMCopyDisc(dmVecIn,basec);CHKERRQ(ierr);
  if (sfRed) {
    ierr = PetscObjectReference((PetscObject)vecIn);CHKERRQ(ierr);
    vecInLocal = vecIn;
  } else {
    ierr = DMCreateLocalVector(basec,&vecInLocal);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(basec,vecIn,INSERT_VALUES,vecInLocal);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(basec,vecIn,INSERT_VALUES,vecInLocal);CHKERRQ(ierr);
  }
  ierr = PetscObjectCompose((PetscObject)plex,"dmAux",(PetscObject)basec);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)plex,"A",(PetscObject)vecInLocal);CHKERRQ(ierr);
  ierr = VecDestroy(&vecInLocal);CHKERRQ(ierr);
  ierr = VecDestroy(&vecIn);CHKERRQ(ierr);
  ierr = DMPlexSetSubpointMap(basec,subpointMap);CHKERRQ(ierr);
  ierr = DMViewFromOptions(basec,NULL,"-dm_basec_view");CHKERRQ(ierr);
  ierr = DMDestroy(&basec);CHKERRQ(ierr);

  ierr = DMGetLocalVector(dmIn,&dummy);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dmIn,&vecOutLocal);CHKERRQ(ierr);
  ierr = DMProjectFieldLocal(dmIn,0.0,dummy,funcs,INSERT_ALL_VALUES,vecOutLocal);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dmIn,&dummy);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)plex,"A",NULL);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)plex,"dmAux",NULL);CHKERRQ(ierr);

  /* output */
  if (n_hi > 1) { /* downsweep the stored hierarchy */
    Vec vecOut1, vecOut2;
    DM  fineDM;

    ierr = DMGetGlobalVector(dmIn,&vecOut1);CHKERRQ(ierr);
    ierr = DMLocalToGlobal(dmIn,vecOutLocal,INSERT_VALUES,vecOut1);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dmIn,&vecOutLocal);CHKERRQ(ierr);
    for (i = 1; i < n_hi-1; i++) {
      fineDM  = hierarchy[i];
      ierr    = DMGetGlobalVector(fineDM,&vecOut2);CHKERRQ(ierr);
      ierr    = DMForestTransferVec(dmIn,vecOut1,fineDM,vecOut2,PETSC_TRUE,0.0);CHKERRQ(ierr);
      ierr    = DMRestoreGlobalVector(dmIn,&vecOut1);CHKERRQ(ierr);
      vecOut1 = vecOut2;
      dmIn    = fineDM;
    }
    ierr = DMForestTransferVec(dmIn,vecOut1,dm,vecOut,PETSC_TRUE,0.0);CHKERRQ(ierr);
    ierr = DMRestoreGlobalVector(dmIn,&vecOut1);CHKERRQ(ierr);
  } else {
    ierr = DMLocalToGlobal(dmIn,vecOutLocal,INSERT_VALUES,vecOut);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(dmIn,&vecOutLocal);CHKERRQ(ierr);
  }
  ierr = PetscFree2(hierarchy,hierarchy_forest);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define DMForestTransferVec_pforest _append_pforest(DMForestTransferVec)
static PetscErrorCode DMForestTransferVec_pforest(DM dmIn, Vec vecIn, DM dmOut, Vec vecOut, PetscBool useBCs, PetscReal time)
{
  DM             adaptIn, adaptOut, plexIn, plexOut;
  DM_Forest      *forestIn, *forestOut, *forestAdaptIn, *forestAdaptOut;
  PetscInt       dofPerDim[] = {1, 1, 1, 1};
  PetscSF        inSF = NULL, outSF = NULL;
  PetscInt       *inCids = NULL, *outCids = NULL;
  DMAdaptFlag    purposeIn, purposeOut;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  forestOut = (DM_Forest *) dmOut->data;
  forestIn  = (DM_Forest *) dmIn->data;

  ierr = DMForestGetAdaptivityForest(dmOut,&adaptOut);CHKERRQ(ierr);
  ierr = DMForestGetAdaptivityPurpose(dmOut,&purposeOut);CHKERRQ(ierr);
  forestAdaptOut = adaptOut ? (DM_Forest *) adaptOut->data : NULL;

  ierr = DMForestGetAdaptivityForest(dmIn,&adaptIn);CHKERRQ(ierr);
  ierr = DMForestGetAdaptivityPurpose(dmIn,&purposeIn);CHKERRQ(ierr);
  forestAdaptIn  = adaptIn ? (DM_Forest *) adaptIn->data : NULL;

  if (forestAdaptOut == forestIn) {
    switch (purposeOut) {
    case DM_ADAPT_REFINE:
      ierr = DMPforestGetTransferSF_Internal(dmIn,dmOut,dofPerDim,&inSF,PETSC_TRUE,&inCids);CHKERRQ(ierr);
      ierr = PetscSFSetUp(inSF);CHKERRQ(ierr);
      break;
    case DM_ADAPT_COARSEN:
    case DM_ADAPT_COARSEN_LAST:
      ierr = DMPforestGetTransferSF_Internal(dmOut,dmIn,dofPerDim,&outSF,PETSC_TRUE,&outCids);CHKERRQ(ierr);
      ierr = PetscSFSetUp(outSF);CHKERRQ(ierr);
      break;
    default:
      ierr = DMPforestGetTransferSF_Internal(dmIn,dmOut,dofPerDim,&inSF,PETSC_TRUE,&inCids);CHKERRQ(ierr);
      ierr = DMPforestGetTransferSF_Internal(dmOut,dmIn,dofPerDim,&outSF,PETSC_FALSE,&outCids);CHKERRQ(ierr);
      ierr = PetscSFSetUp(inSF);CHKERRQ(ierr);
      ierr = PetscSFSetUp(outSF);CHKERRQ(ierr);
    }
  } else if (forestAdaptIn == forestOut) {
    switch (purposeIn) {
    case DM_ADAPT_REFINE:
      ierr = DMPforestGetTransferSF_Internal(dmOut,dmIn,dofPerDim,&outSF,PETSC_TRUE,&inCids);CHKERRQ(ierr);
      ierr = PetscSFSetUp(outSF);CHKERRQ(ierr);
      break;
    case DM_ADAPT_COARSEN:
    case DM_ADAPT_COARSEN_LAST:
      ierr = DMPforestGetTransferSF_Internal(dmIn,dmOut,dofPerDim,&inSF,PETSC_TRUE,&inCids);CHKERRQ(ierr);
      ierr = PetscSFSetUp(inSF);CHKERRQ(ierr);
      break;
    default:
      ierr = DMPforestGetTransferSF_Internal(dmIn,dmOut,dofPerDim,&inSF,PETSC_TRUE,&inCids);CHKERRQ(ierr);
      ierr = DMPforestGetTransferSF_Internal(dmOut,dmIn,dofPerDim,&outSF,PETSC_FALSE,&outCids);CHKERRQ(ierr);
      ierr = PetscSFSetUp(inSF);CHKERRQ(ierr);
      ierr = PetscSFSetUp(outSF);CHKERRQ(ierr);
    }
  } else SETERRQ(PetscObjectComm((PetscObject)dmIn),PETSC_ERR_SUP,"Only support transfer from pre-adaptivity to post-adaptivity right now");
  ierr = DMPforestGetPlex(dmIn,&plexIn);CHKERRQ(ierr);
  ierr = DMPforestGetPlex(dmOut,&plexOut);CHKERRQ(ierr);

  ierr = DMPlexTransferVecTree(plexIn,vecIn,plexOut,vecOut,inSF,outSF,inCids,outCids,useBCs,time);CHKERRQ(ierr);
  ierr = PetscFree(inCids);CHKERRQ(ierr);
  ierr = PetscFree(outCids);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&inSF);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&outSF);CHKERRQ(ierr);
  ierr = PetscFree(inCids);CHKERRQ(ierr);
  ierr = PetscFree(outCids);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define DMCreateCoordinateDM_pforest _append_pforest(DMCreateCoordinateDM)
static PetscErrorCode DMCreateCoordinateDM_pforest(DM dm,DM *cdm)
{
  DM             plex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMPforestGetPlex(dm,&plex);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(plex,cdm);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)*cdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define VecViewLocal_pforest _append_pforest(VecViewLocal)
static PetscErrorCode VecViewLocal_pforest(Vec vec,PetscViewer viewer)
{
  DM             dm, plex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(vec,&dm);CHKERRQ(ierr);
  ierr = DMPforestGetPlex(dm,&plex);CHKERRQ(ierr);
  ierr = VecSetDM(vec,plex);CHKERRQ(ierr);
  ierr = VecView_Plex_Local(vec,viewer);CHKERRQ(ierr);
  ierr = VecSetDM(vec,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define VecView_pforest _append_pforest(VecView)
static PetscErrorCode VecView_pforest(Vec vec,PetscViewer viewer)
{
  DM             dm, plex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(vec,&dm);CHKERRQ(ierr);
  ierr = DMPforestGetPlex(dm,&plex);CHKERRQ(ierr);
  ierr = VecSetDM(vec,plex);CHKERRQ(ierr);
  ierr = VecView_Plex(vec,viewer);CHKERRQ(ierr);
  ierr = VecSetDM(vec,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define VecView_pforest_Native _infix_pforest(VecView,_Native)
static PetscErrorCode VecView_pforest_Native(Vec vec,PetscViewer viewer)
{
  DM             dm, plex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(vec,&dm);CHKERRQ(ierr);
  ierr = DMPforestGetPlex(dm,&plex);CHKERRQ(ierr);
  ierr = VecSetDM(vec,plex);CHKERRQ(ierr);
  ierr = VecView_Plex_Native(vec,viewer);CHKERRQ(ierr);
  ierr = VecSetDM(vec,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define VecLoad_pforest _append_pforest(VecLoad)
static PetscErrorCode VecLoad_pforest(Vec vec,PetscViewer viewer)
{
  DM             dm, plex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(vec,&dm);CHKERRQ(ierr);
  ierr = DMPforestGetPlex(dm,&plex);CHKERRQ(ierr);
  ierr = VecSetDM(vec,plex);CHKERRQ(ierr);
  ierr = VecLoad_Plex(vec,viewer);CHKERRQ(ierr);
  ierr = VecSetDM(vec,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define VecLoad_pforest_Native _infix_pforest(VecLoad,_Native)
static PetscErrorCode VecLoad_pforest_Native(Vec vec,PetscViewer viewer)
{
  DM             dm, plex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetDM(vec,&dm);CHKERRQ(ierr);
  ierr = DMPforestGetPlex(dm,&plex);CHKERRQ(ierr);
  ierr = VecSetDM(vec,plex);CHKERRQ(ierr);
  ierr = VecLoad_Plex_Native(vec,viewer);CHKERRQ(ierr);
  ierr = VecSetDM(vec,dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define DMCreateGlobalVector_pforest _append_pforest(DMCreateGlobalVector)
static PetscErrorCode DMCreateGlobalVector_pforest(DM dm,Vec *vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreateGlobalVector_Section_Private(dm,vec);CHKERRQ(ierr);
  /* ierr = VecSetOperation(*vec, VECOP_DUPLICATE, (void(*)(void)) VecDuplicate_MPI_DM);CHKERRQ(ierr); */
  ierr = VecSetOperation(*vec, VECOP_VIEW, (void (*)(void))VecView_pforest);CHKERRQ(ierr);
  ierr = VecSetOperation(*vec, VECOP_VIEWNATIVE, (void (*)(void))VecView_pforest_Native);CHKERRQ(ierr);
  ierr = VecSetOperation(*vec, VECOP_LOAD, (void (*)(void))VecLoad_pforest);CHKERRQ(ierr);
  ierr = VecSetOperation(*vec, VECOP_LOADNATIVE, (void (*)(void))VecLoad_pforest_Native);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define DMCreateLocalVector_pforest _append_pforest(DMCreateLocalVector)
static PetscErrorCode DMCreateLocalVector_pforest(DM dm,Vec *vec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreateLocalVector_Section_Private(dm,vec);CHKERRQ(ierr);
  ierr = VecSetOperation(*vec, VECOP_VIEW, (void (*)(void))VecViewLocal_pforest);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define DMCreateMatrix_pforest _append_pforest(DMCreateMatrix)
static PetscErrorCode DMCreateMatrix_pforest(DM dm,Mat *mat)
{
  DM             plex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMPforestGetPlex(dm,&plex);CHKERRQ(ierr);
  ierr = DMCreateMatrix(plex,mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define DMProjectFunctionLocal_pforest _append_pforest(DMProjectFunctionLocal)
static PetscErrorCode DMProjectFunctionLocal_pforest(DM dm, PetscReal time, PetscErrorCode (**funcs) (PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void*), void **ctxs, InsertMode mode, Vec localX)
{
  DM             plex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMPforestGetPlex(dm,&plex);CHKERRQ(ierr);
  ierr = DMProjectFunctionLocal(plex,time,funcs,ctxs,mode,localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define DMProjectFunctionLabelLocal_pforest _append_pforest(DMProjectFunctionLabelLocal)
static PetscErrorCode DMProjectFunctionLabelLocal_pforest(DM dm, PetscReal time, DMLabel label, PetscInt numIds, const PetscInt ids[], PetscInt Ncc, const PetscInt comps[], PetscErrorCode (**funcs) (PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void*), void **ctxs, InsertMode mode, Vec localX)
{
  DM             plex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMPforestGetPlex(dm,&plex);CHKERRQ(ierr);
  ierr = DMProjectFunctionLabelLocal(plex,time,label,numIds,ids,Ncc,comps,funcs,ctxs,mode,localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define DMProjectFieldLocal_pforest _append_pforest(DMProjectFieldLocal)
PetscErrorCode DMProjectFieldLocal_pforest(DM dm, PetscReal time, Vec localU,void (**funcs) (PetscInt, PetscInt, PetscInt,
                                                                             const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                             const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                             PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),InsertMode mode, Vec localX)
{
  DM             plex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMPforestGetPlex(dm,&plex);CHKERRQ(ierr);
  ierr = DMProjectFieldLocal(plex,time,localU,funcs,mode,localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define DMComputeL2Diff_pforest _append_pforest(DMComputeL2Diff)
PetscErrorCode DMComputeL2Diff_pforest(DM dm, PetscReal time, PetscErrorCode (**funcs) (PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void*), void **ctxs, Vec X, PetscReal *diff)
{
  DM             plex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMPforestGetPlex(dm,&plex);CHKERRQ(ierr);
  ierr = DMComputeL2Diff(plex,time,funcs,ctxs,X,diff);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define DMComputeL2FieldDiff_pforest _append_pforest(DMComputeL2FieldDiff)
PetscErrorCode DMComputeL2FieldDiff_pforest(DM dm, PetscReal time, PetscErrorCode (**funcs) (PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void*), void **ctxs, Vec X, PetscReal diff[])
{
  DM             plex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMPforestGetPlex(dm,&plex);CHKERRQ(ierr);
  ierr = DMComputeL2FieldDiff(plex,time,funcs,ctxs,X,diff);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define DMCreatelocalsection_pforest _append_pforest(DMCreatelocalsection)
static PetscErrorCode DMCreatelocalsection_pforest(DM dm)
{
  DM             plex;
  PetscSection   section;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMPforestGetPlex(dm,&plex);CHKERRQ(ierr);
  ierr = DMGetLocalSection(plex,&section);CHKERRQ(ierr);
  ierr = DMSetLocalSection(dm,section);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define DMCreateDefaultConstraints_pforest _append_pforest(DMCreateDefaultConstraints)
static PetscErrorCode DMCreateDefaultConstraints_pforest(DM dm)
{
  DM             plex;
  Mat            mat;
  PetscSection   section;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMPforestGetPlex(dm,&plex);CHKERRQ(ierr);
  ierr = DMGetDefaultConstraints(plex,&section,&mat);CHKERRQ(ierr);
  ierr = DMSetDefaultConstraints(dm,section,mat);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define DMGetDimPoints_pforest _append_pforest(DMGetDimPoints)
static PetscErrorCode DMGetDimPoints_pforest(DM dm, PetscInt dim, PetscInt *cStart, PetscInt *cEnd)
{
  DM             plex;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  ierr = DMPforestGetPlex(dm,&plex);CHKERRQ(ierr);
  ierr = DMGetDimPoints(plex,dim,cStart,cEnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Need to forward declare */
#define DMInitialize_pforest _append_pforest(DMInitialize)
static PetscErrorCode DMInitialize_pforest(DM dm);

#define DMClone_pforest _append_pforest(DMClone)
static PetscErrorCode DMClone_pforest(DM dm, DM *newdm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMClone_Forest(dm,newdm);CHKERRQ(ierr);
  ierr = DMInitialize_pforest(*newdm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define DMForestCreateCellChart_pforest _append_pforest(DMForestCreateCellChart)
static PetscErrorCode DMForestCreateCellChart_pforest(DM dm, PetscInt *cStart, PetscInt *cEnd)
{
  DM_Forest         *forest;
  DM_Forest_pforest *pforest;
  PetscInt          overlap;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr    = DMSetUp(dm);CHKERRQ(ierr);
  forest  = (DM_Forest*) dm->data;
  pforest = (DM_Forest_pforest*) forest->data;
  *cStart = 0;
  ierr    = DMForestGetPartitionOverlap(dm,&overlap);CHKERRQ(ierr);
  if (overlap && pforest->ghost) {
    *cEnd = pforest->forest->local_num_quadrants + pforest->ghost->proc_offsets[pforest->forest->mpisize];
  } else {
    *cEnd = pforest->forest->local_num_quadrants;
  }
  PetscFunctionReturn(0);
}

#define DMForestCreateCellSF_pforest _append_pforest(DMForestCreateCellSF)
static PetscErrorCode DMForestCreateCellSF_pforest(DM dm, PetscSF *cellSF)
{
  DM_Forest         *forest;
  DM_Forest_pforest *pforest;
  PetscMPIInt       rank;
  PetscInt          overlap;
  PetscInt          cStart, cEnd, cLocalStart, cLocalEnd;
  PetscInt          nRoots, nLeaves, *mine = NULL;
  PetscSFNode       *remote = NULL;
  PetscSF           sf;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr        = DMForestGetCellChart(dm,&cStart,&cEnd);CHKERRQ(ierr);
  forest      = (DM_Forest*)         dm->data;
  pforest     = (DM_Forest_pforest*) forest->data;
  nRoots      = cEnd - cStart;
  cLocalStart = pforest->cLocalStart;
  cLocalEnd   = pforest->cLocalEnd;
  nLeaves     = 0;
  ierr        = DMForestGetPartitionOverlap(dm,&overlap);CHKERRQ(ierr);
  ierr        = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRQ(ierr);
  if (overlap && pforest->ghost) {
    PetscSFNode      *mirror;
    p4est_quadrant_t *mirror_array;
    PetscInt         nMirror, nGhostPre, nSelf, q;
    void             **mirrorPtrs;

    nMirror      = (PetscInt) pforest->ghost->mirrors.elem_count;
    nSelf        = cLocalEnd - cLocalStart;
    nLeaves      = nRoots - nSelf;
    nGhostPre    = (PetscInt) pforest->ghost->proc_offsets[rank];
    ierr         = PetscMalloc1(nLeaves,&mine);CHKERRQ(ierr);
    ierr         = PetscMalloc1(nLeaves,&remote);CHKERRQ(ierr);
    ierr         = PetscMalloc2(nMirror,&mirror,nMirror,&mirrorPtrs);CHKERRQ(ierr);
    mirror_array = (p4est_quadrant_t*) pforest->ghost->mirrors.array;
    for (q = 0; q < nMirror; q++) {
      p4est_quadrant_t *mir = &(mirror_array[q]);

      mirror[q].rank  = rank;
      mirror[q].index = (PetscInt) mir->p.piggy3.local_num + cLocalStart;
      mirrorPtrs[q]   = (void*) &(mirror[q]);
    }
    PetscStackCallP4est(p4est_ghost_exchange_custom,(pforest->forest,pforest->ghost,sizeof(PetscSFNode),mirrorPtrs,remote));
    ierr = PetscFree2(mirror,mirrorPtrs);CHKERRQ(ierr);
    for (q = 0; q < nGhostPre; q++) mine[q] = q;
    for (; q < nLeaves; q++) mine[q] = (q - nGhostPre) + cLocalEnd;
  }
  ierr    = PetscSFCreate(PetscObjectComm((PetscObject)dm),&sf);CHKERRQ(ierr);
  ierr    = PetscSFSetGraph(sf,nRoots,nLeaves,mine,PETSC_OWN_POINTER,remote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  *cellSF = sf;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMInitialize_pforest(DM dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  dm->ops->setup                     = DMSetUp_pforest;
  dm->ops->view                      = DMView_pforest;
  dm->ops->clone                     = DMClone_pforest;
  dm->ops->createinterpolation       = DMCreateInterpolation_pforest;
  dm->ops->createinjection           = DMCreateInjection_pforest;
  dm->ops->setfromoptions            = DMSetFromOptions_pforest;
  dm->ops->createcoordinatedm        = DMCreateCoordinateDM_pforest;
  dm->ops->createglobalvector        = DMCreateGlobalVector_pforest;
  dm->ops->createlocalvector         = DMCreateLocalVector_pforest;
  dm->ops->creatematrix              = DMCreateMatrix_pforest;
  dm->ops->projectfunctionlocal      = DMProjectFunctionLocal_pforest;
  dm->ops->projectfunctionlabellocal = DMProjectFunctionLabelLocal_pforest;
  dm->ops->projectfieldlocal         = DMProjectFieldLocal_pforest;
  dm->ops->createlocalsection      = DMCreatelocalsection_pforest;
  dm->ops->createdefaultconstraints  = DMCreateDefaultConstraints_pforest;
  dm->ops->computel2diff             = DMComputeL2Diff_pforest;
  dm->ops->computel2fielddiff        = DMComputeL2FieldDiff_pforest;
  dm->ops->getdimpoints              = DMGetDimPoints_pforest;

  ierr = PetscObjectComposeFunction((PetscObject)dm,_pforest_string(DMConvert_plex_pforest) "_C",DMConvert_plex_pforest);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)dm,_pforest_string(DMConvert_pforest_plex) "_C",DMConvert_pforest_plex);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define DMCreate_pforest _append_pforest(DMCreate)
PETSC_EXTERN PetscErrorCode DMCreate_pforest(DM dm)
{
  DM_Forest         *forest;
  DM_Forest_pforest *pforest;
  PetscErrorCode    ierr;

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

  forest                            = (DM_Forest*) dm->data;
  forest->data                      = pforest;
  forest->destroy                   = DMForestDestroy_pforest;
  forest->ftemplate                 = DMForestTemplate_pforest;
  forest->transfervec               = DMForestTransferVec_pforest;
  forest->transfervecfrombase       = DMForestTransferVecFromBase_pforest;
  forest->createcellchart           = DMForestCreateCellChart_pforest;
  forest->createcellsf              = DMForestCreateCellSF_pforest;
  forest->clearadaptivityforest     = DMForestClearAdaptivityForest_pforest;
  forest->getadaptivitysuccess      = DMForestGetAdaptivitySuccess_pforest;
  pforest->topo                     = NULL;
  pforest->forest                   = NULL;
  pforest->ghost                    = NULL;
  pforest->lnodes                   = NULL;
  pforest->partition_for_coarsening = PETSC_TRUE;
  pforest->coarsen_hierarchy        = PETSC_FALSE;
  pforest->cLocalStart              = -1;
  pforest->cLocalEnd                = -1;
  pforest->labelsFinalized          = PETSC_FALSE;
  pforest->ghostName                = NULL;
  PetscFunctionReturn(0);
}

#endif /* defined(PETSC_HAVE_P4EST) */
