#include <petscds.h>
#include <petsc/private/dmimpl.h>
#include <petsc/private/dmforestimpl.h>
#include <petsc/private/dmpleximpl.h>
#include <petsc/private/dmlabelimpl.h>
#include <petsc/private/viewerimpl.h>
#include <../src/sys/classes/viewer/impls/vtk/vtkvimpl.h>
#include "petsc_p4est_package.h"

#if defined(PETSC_HAVE_P4EST)

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
  PetscFunctionBegin;
  if (!(*topo)) PetscFunctionReturn(0);
  if (--((*topo)->refct) > 0) {
    *topo = NULL;
    PetscFunctionReturn(0);
  }
  if ((*topo)->geom) PetscStackCallP4est(p4est_geometry_destroy,((*topo)->geom));
  PetscStackCallP4est(p4est_connectivity_destroy,((*topo)->conn));
  PetscCall(PetscFree((*topo)->tree_face_to_uniq));
  PetscCall(PetscFree(*topo));
  *topo = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode PforestConnectivityEnumerateFacets(p4est_connectivity_t*,PetscInt**);

#define DMFTopologyCreateBrick_pforest _append_pforest(DMFTopologyCreateBrick)
static PetscErrorCode DMFTopologyCreateBrick_pforest(DM dm,PetscInt N[], PetscInt P[], PetscReal B[],DMFTopology_pforest **topo, PetscBool useMorton)
{
  double         *vertices;
  PetscInt       i, numVerts;

  PetscFunctionBegin;
  PetscCheck(useMorton,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Lexicographic ordering not implemented yet");
  PetscCall(PetscNewLog(dm,topo));

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
  PetscCall(PforestConnectivityEnumerateFacets((*topo)->conn,&(*topo)->tree_face_to_uniq));
  PetscFunctionReturn(0);
}

#define DMFTopologyCreate_pforest _append_pforest(DMFTopologyCreate)
static PetscErrorCode DMFTopologyCreate_pforest(DM dm, DMForestTopology topologyName, DMFTopology_pforest **topo)
{
  const char     *name = (const char*) topologyName;
  const char     *prefix;
  PetscBool      isBrick, isShell, isSphere, isMoebius;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidCharPointer(name,2);
  PetscValidPointer(topo,3);
  PetscCall(PetscStrcmp(name,"brick",&isBrick));
  PetscCall(PetscStrcmp(name,"shell",&isShell));
  PetscCall(PetscStrcmp(name,"sphere",&isSphere));
  PetscCall(PetscStrcmp(name,"moebius",&isMoebius));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)dm,&prefix));
  if (isBrick) {
    PetscBool flgN, flgP, flgM, flgB, useMorton = PETSC_TRUE, periodic = PETSC_FALSE;
    PetscInt  N[3] = {2,2,2}, P[3] = {0,0,0}, nretN = P4EST_DIM, nretP = P4EST_DIM, nretB = 2 * P4EST_DIM, i;
    PetscReal B[6] = {0.0,1.0,0.0,1.0,0.0,1.0};

    if (dm->setfromoptionscalled) {
      PetscCall(PetscOptionsGetIntArray(((PetscObject)dm)->options,prefix,"-dm_p4est_brick_size",N,&nretN,&flgN));
      PetscCall(PetscOptionsGetIntArray(((PetscObject)dm)->options,prefix,"-dm_p4est_brick_periodicity",P,&nretP,&flgP));
      PetscCall(PetscOptionsGetRealArray(((PetscObject)dm)->options,prefix,"-dm_p4est_brick_bounds",B,&nretB,&flgB));
      PetscCall(PetscOptionsGetBool(((PetscObject)dm)->options,prefix,"-dm_p4est_brick_use_morton_curve",&useMorton,&flgM));
      PetscCheck(!flgN || nretN == P4EST_DIM,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_SIZ,"Need to give %d sizes in -dm_p4est_brick_size, gave %" PetscInt_FMT,P4EST_DIM,nretN);
      PetscCheck(!flgP || nretP == P4EST_DIM,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_SIZ,"Need to give %d periodicities in -dm_p4est_brick_periodicity, gave %" PetscInt_FMT,P4EST_DIM,nretP);
      PetscCheck(!flgB || nretB == 2 * P4EST_DIM,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_SIZ,"Need to give %d bounds in -dm_p4est_brick_bounds, gave %" PetscInt_FMT,P4EST_DIM,nretP);
    }
    for (i = 0; i < P4EST_DIM; i++) {
      P[i]  = (P[i] ? DM_BOUNDARY_PERIODIC : DM_BOUNDARY_NONE);
      periodic = (PetscBool)(P[i] || periodic);
      if (!flgB) B[2 * i + 1] = N[i];
    }
    PetscCall(DMFTopologyCreateBrick_pforest(dm,N,P,B,topo,useMorton));
    /* the maxCell trick is not robust enough, localize on all cells if periodic */
    PetscCall(DMSetPeriodicity(dm,periodic,NULL,NULL,NULL));
  } else {
    PetscCall(PetscNewLog(dm,topo));

    (*topo)->refct = 1;
    PetscStackCallP4estReturn((*topo)->conn,p4est_connectivity_new_byname,(name));
    (*topo)->geom = NULL;
    if (isMoebius) PetscCall(DMSetCoordinateDim(dm,3));
#if defined(P4_TO_P8)
    if (isShell) {
      PetscReal R2 = 1., R1 = .55;

      if (dm->setfromoptionscalled) {
        PetscCall(PetscOptionsGetReal(((PetscObject)dm)->options,prefix,"-dm_p4est_shell_outer_radius",&R2,NULL));
        PetscCall(PetscOptionsGetReal(((PetscObject)dm)->options,prefix,"-dm_p4est_shell_inner_radius",&R1,NULL));
      }
      PetscStackCallP4estReturn((*topo)->geom,p8est_geometry_new_shell,((*topo)->conn,R2,R1));
    } else if (isSphere) {
      PetscReal R2 = 1., R1 = 0.191728, R0 = 0.039856;

      if (dm->setfromoptionscalled) {
        PetscCall(PetscOptionsGetReal(((PetscObject)dm)->options,prefix,"-dm_p4est_sphere_outer_radius",&R2,NULL));
        PetscCall(PetscOptionsGetReal(((PetscObject)dm)->options,prefix,"-dm_p4est_sphere_inner_radius",&R1,NULL));
        PetscCall(PetscOptionsGetReal(((PetscObject)dm)->options,prefix,"-dm_p4est_sphere_core_radius",&R0,NULL));
      }
      PetscStackCallP4estReturn((*topo)->geom,p8est_geometry_new_sphere,((*topo)->conn,R2,R1,R0));
    }
#endif
    PetscCall(PforestConnectivityEnumerateFacets((*topo)->conn,&(*topo)->tree_face_to_uniq));
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

  PetscFunctionBegin;

  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  comm = PetscObjectComm((PetscObject)dm);
  PetscCall(PetscObjectTypeCompare((PetscObject)dm,DMPLEX,&isPlex));
  PetscCheck(isPlex,comm,PETSC_ERR_ARG_WRONG,"Expected DM type %s, got %s",DMPLEX,((PetscObject)dm)->type_name);
  PetscCall(DMGetDimension(dm,&dim));
  PetscCheck(dim == P4EST_DIM,comm,PETSC_ERR_ARG_WRONG,"Expected DM dimension %d, got %" PetscInt_FMT,P4EST_DIM,dim);
  PetscCall(DMCreate(comm,pforest));
  PetscCall(DMSetType(*pforest,DMPFOREST));
  PetscCall(DMForestSetBaseDM(*pforest,dm));
  PetscCall(DMGetApplicationContext(dm,&ctx));
  PetscCall(DMSetApplicationContext(*pforest,ctx));
  PetscCall(DMCopyDisc(dm,*pforest));
  PetscFunctionReturn(0);
}

#define DMForestDestroy_pforest _append_pforest(DMForestDestroy)
static PetscErrorCode DMForestDestroy_pforest(DM dm)
{
  DM_Forest         *forest  = (DM_Forest*) dm->data;
  DM_Forest_pforest *pforest = (DM_Forest_pforest*) forest->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (pforest->lnodes) PetscStackCallP4est(p4est_lnodes_destroy,(pforest->lnodes));
  pforest->lnodes = NULL;
  if (pforest->ghost) PetscStackCallP4est(p4est_ghost_destroy,(pforest->ghost));
  pforest->ghost = NULL;
  if (pforest->forest) PetscStackCallP4est(p4est_destroy,(pforest->forest));
  pforest->forest = NULL;
  PetscCall(DMFTopologyDestroy_pforest(&pforest->topo));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm,PetscStringize(DMConvert_plex_pforest) "_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm,PetscStringize(DMConvert_pforest_plex) "_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm,"DMSetUpGLVisViewer_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm,"DMCreateNeumannOverlap_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm,"DMPlexGetOverlap_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm,"MatComputeNeumannOverlap_C",NULL));
  PetscCall(PetscFree(pforest->ghostName));
  PetscCall(DMDestroy(&pforest->plex));
  PetscCall(PetscSFDestroy(&pforest->pointAdaptToSelfSF));
  PetscCall(PetscSFDestroy(&pforest->pointSelfToAdaptSF));
  PetscCall(PetscFree(pforest->pointAdaptToSelfCids));
  PetscCall(PetscFree(pforest->pointSelfToAdaptCids));
  PetscCall(PetscFree(forest->data));
  PetscFunctionReturn(0);
}

#define DMForestTemplate_pforest _append_pforest(DMForestTemplate)
static PetscErrorCode DMForestTemplate_pforest(DM dm, DM tdm)
{
  DM_Forest_pforest *pforest  = (DM_Forest_pforest*) ((DM_Forest*) dm->data)->data;
  DM_Forest_pforest *tpforest = (DM_Forest_pforest*) ((DM_Forest*) tdm->data)->data;

  PetscFunctionBegin;
  if (pforest->topo) pforest->topo->refct++;
  PetscCall(DMFTopologyDestroy_pforest(&(tpforest->topo)));
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
    PetscCheck(comp,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"non-matching partitions");

    for (currentFrom = 0, currentTo = 0; currentFrom < numFrom && currentTo < numTo;) {
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

  PetscFunctionBegin;
  PetscCheck(pforest,PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Missing DM_Forest_pforest");
  PetscCheck(pforest->forest,PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Missing p4est_t");
  p4est = pforest->forest;
  flt   = p4est->first_local_tree;
  llt   = p4est->last_local_tree;
  for (t = flt; t <= llt; t++) {
    p4est_tree_t *tree  = &(((p4est_tree_t*) p4est->trees->array)[t]);
    maxlevelloc = PetscMax((PetscInt)tree->maxlevel,maxlevelloc);
  }
  PetscCall(MPIU_Allreduce(&maxlevelloc,lev,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)dm)));
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

  PetscFunctionBegin;
  flt  = p4estFrom->first_local_tree;
  llt  = p4estFrom->last_local_tree;
  PetscCall(PetscSFCreate(comm,&fromCoarse));
  if (toCoarseFromFine) {
    PetscCall(PetscSFCreate(comm,&toCoarse));
  }
  numRootsFrom = p4estFrom->local_num_quadrants + FromOffset;
  numRootsTo   = p4estTo->local_num_quadrants + ToOffset;
  PetscCall(DMPforestComputeLocalCellTransferSF_loop(p4estFrom,FromOffset,p4estTo,ToOffset,flt,llt,&numLeavesTo,NULL,NULL,&numLeavesFrom,NULL,NULL));
  PetscCall(PetscMalloc1(numLeavesTo,&toLeaves));
  PetscCall(PetscMalloc1(numLeavesTo,&fromRoots));
  if (toCoarseFromFine) {
    PetscCall(PetscMalloc1(numLeavesFrom,&fromLeaves));
    PetscCall(PetscMalloc1(numLeavesFrom,&fromRoots));
  }
  PetscCall(DMPforestComputeLocalCellTransferSF_loop(p4estFrom,FromOffset,p4estTo,ToOffset,flt,llt,&numLeavesTo,toLeaves,fromRoots,&numLeavesFrom,fromLeaves,toRoots));
  if (!ToOffset && (numLeavesTo == numRootsTo)) { /* compress */
    PetscCall(PetscFree(toLeaves));
    PetscCall(PetscSFSetGraph(fromCoarse,numRootsFrom,numLeavesTo,NULL,PETSC_OWN_POINTER,fromRoots,PETSC_OWN_POINTER));
  } else PetscCall(PetscSFSetGraph(fromCoarse,numRootsFrom,numLeavesTo,toLeaves,PETSC_OWN_POINTER,fromRoots,PETSC_OWN_POINTER));
  *fromCoarseToFine = fromCoarse;
  if (toCoarseFromFine) {
    PetscCall(PetscSFSetGraph(toCoarse,numRootsTo,numLeavesFrom,fromLeaves,PETSC_OWN_POINTER,toRoots,PETSC_OWN_POINTER));
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

  PetscFunctionBegin;
  ctx.minLevel  = PETSC_MAX_INT;
  ctx.maxLevel  = 0;
  ctx.currLevel = 0;
  ctx.anyChange = PETSC_FALSE;
  /* sanity check */
  PetscCall(DMForestGetAdaptivityForest(dm,&adaptFrom));
  PetscCall(DMForestGetBaseDM(dm,&base));
  PetscCall(DMForestGetTopology(dm,&topoName));
  PetscCheck(adaptFrom || base || topoName,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"A forest needs a topology, a base DM, or a DM to adapt from");

  /* === Step 1: DMFTopology === */
  if (adaptFrom) { /* reference already created topology */
    PetscBool         ispforest;
    DM_Forest         *aforest  = (DM_Forest*) adaptFrom->data;
    DM_Forest_pforest *apforest = (DM_Forest_pforest*) aforest->data;

    PetscCall(PetscObjectTypeCompare((PetscObject)adaptFrom,DMPFOREST,&ispforest));
    PetscCheck(ispforest,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_NOTSAMETYPE,"Trying to adapt from %s, which is not %s",((PetscObject)adaptFrom)->type_name,DMPFOREST);
    PetscCheck(apforest->topo,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONGSTATE,"The pre-adaptation forest must have a topology");
    PetscCall(DMSetUp(adaptFrom));
    PetscCall(DMForestGetBaseDM(dm,&base));
    PetscCall(DMForestGetTopology(dm,&topoName));
  } else if (base) { /* construct a connectivity from base */
    PetscBool isPlex, isDA;

    PetscCall(PetscObjectGetName((PetscObject)base,&topoName));
    PetscCall(DMForestSetTopology(dm,topoName));
    PetscCall(PetscObjectTypeCompare((PetscObject)base,DMPLEX,&isPlex));
    PetscCall(PetscObjectTypeCompare((PetscObject)base,DMDA,&isDA));
    if (isPlex) {
      MPI_Comm             comm = PetscObjectComm((PetscObject)dm);
      PetscInt             depth;
      PetscMPIInt          size;
      p4est_connectivity_t *conn = NULL;
      DMFTopology_pforest  *topo;
      PetscInt             *tree_face_to_uniq = NULL;

      PetscCall(DMPlexGetDepth(base,&depth));
      if (depth == 1) {
        DM connDM;

        PetscCall(DMPlexInterpolate(base,&connDM));
        base = connDM;
        PetscCall(DMForestSetBaseDM(dm,base));
        PetscCall(DMDestroy(&connDM));
      } else PetscCheck(depth == P4EST_DIM,comm,PETSC_ERR_ARG_WRONG,"Base plex is neither interpolated nor uninterpolated? depth %" PetscInt_FMT ", expected 2 or %d",depth,P4EST_DIM + 1);
      PetscCallMPI(MPI_Comm_size(comm,&size));
      if (size > 1) {
        DM      dmRedundant;
        PetscSF sf;

        PetscCall(DMPlexGetRedundantDM(base,&sf,&dmRedundant));
        PetscCheck(dmRedundant,comm,PETSC_ERR_PLIB,"Could not create redundant DM");
        PetscCall(PetscObjectCompose((PetscObject)dmRedundant,"_base_migration_sf",(PetscObject)sf));
        PetscCall(PetscSFDestroy(&sf));
        base = dmRedundant;
        PetscCall(DMForestSetBaseDM(dm,base));
        PetscCall(DMDestroy(&dmRedundant));
      }
      PetscCall(DMViewFromOptions(base,NULL,"-dm_p4est_base_view"));
      PetscCall(DMPlexCreateConnectivity_pforest(base,&conn,&tree_face_to_uniq));
      PetscCall(PetscNewLog(dm,&topo));
      topo->refct = 1;
      topo->conn  = conn;
      topo->geom  = NULL;
      {
        PetscErrorCode (*map)(DM,PetscInt,PetscInt,const PetscReal[],PetscReal[],void*);
        void           *mapCtx;

        PetscCall(DMForestGetBaseCoordinateMapping(dm,&map,&mapCtx));
        if (map) {
          DM_Forest_geometry_pforest *geom_pforest;
          p4est_geometry_t           *geom;

          PetscCall(PetscNew(&geom_pforest));
          PetscCall(DMGetCoordinateDim(dm,&geom_pforest->coordDim));
          geom_pforest->map    = map;
          geom_pforest->mapCtx = mapCtx;
          PetscStackCallP4estReturn(geom_pforest->inner,p4est_geometry_new_connectivity,(conn));
          PetscCall(PetscNew(&geom));
          geom->name    = topoName;
          geom->user    = geom_pforest;
          geom->X       = GeometryMapping_pforest;
          geom->destroy = GeometryDestroy_pforest;
          topo->geom    = geom;
        }
      }
      topo->tree_face_to_uniq = tree_face_to_uniq;
      pforest->topo           = topo;
    } else PetscCheck(!isDA,PetscObjectComm((PetscObject)dm),PETSC_ERR_PLIB,"Not implemented yet");
#if 0
      PetscInt N[3], P[3];

      /* get the sizes, periodicities */
      /* ... */
                                                                  /* don't use Morton order */
      PetscCall(DMFTopologyCreateBrick_pforest(dm,N,P,&pforest->topo,PETSC_FALSE));
#endif
    {
      PetscInt numLabels, l;

      PetscCall(DMGetNumLabels(base,&numLabels));
      for (l = 0; l < numLabels; l++) {
        PetscBool  isDepth, isGhost, isVTK, isDim, isCellType;
        DMLabel    label, labelNew;
        PetscInt   defVal;
        const char *name;

        PetscCall(DMGetLabelName(base, l, &name));
        PetscCall(DMGetLabelByNum(base, l, &label));
        PetscCall(PetscStrcmp(name,"depth",&isDepth));
        if (isDepth) continue;
        PetscCall(PetscStrcmp(name,"dim",&isDim));
        if (isDim) continue;
        PetscCall(PetscStrcmp(name,"celltype",&isCellType));
        if (isCellType) continue;
        PetscCall(PetscStrcmp(name,"ghost",&isGhost));
        if (isGhost) continue;
        PetscCall(PetscStrcmp(name,"vtk",&isVTK));
        if (isVTK) continue;
        PetscCall(DMCreateLabel(dm,name));
        PetscCall(DMGetLabel(dm,name,&labelNew));
        PetscCall(DMLabelGetDefaultValue(label,&defVal));
        PetscCall(DMLabelSetDefaultValue(labelNew,defVal));
      }
      /* map dm points (internal plex) to base
         we currently create the subpoint_map for the entire hierarchy, starting from the finest forest
         and propagating back to the coarsest
         This is not an optimal approach, since we need the map only on the coarsest level
         during DMForestTransferVecFromBase */
      PetscCall(DMForestGetMinimumRefinement(dm,&l));
      if (!l) {
        PetscCall(DMCreateLabel(dm,"_forest_base_subpoint_map"));
      }
    }
  } else { /* construct from topology name */
    DMFTopology_pforest *topo;

    PetscCall(DMFTopologyCreate_pforest(dm,topoName,&topo));
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
    PetscCall(DMForestGetComputeAdaptivitySF(dm,&computeAdaptSF));
    PetscStackCallP4estReturn(pforest->forest,p4est_copy,(apforest->forest, 0)); /* 0 indicates no data copying */
    PetscCall(DMForestGetAdaptivityLabel(dm,&adaptLabel));
    if (adaptLabel) {
      /* apply the refinement/coarsening by flags, plus minimum/maximum refinement */
      PetscCall(DMLabelGetNumValues(adaptLabel,&numValues));
      PetscCallMPI(MPI_Allreduce(&numValues,&numValuesGlobal,1,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)adaptFrom)));
      PetscCall(DMLabelGetDefaultValue(adaptLabel,&defaultValue));
      if (!numValuesGlobal && defaultValue == DM_ADAPT_COARSEN_LAST) { /* uniform coarsen of the last level only (equivalent to DM_ADAPT_COARSEN for conforming grids)  */
        PetscCall(DMForestGetMinimumRefinement(dm,&ctx.minLevel));
        PetscCall(DMPforestGetRefinementLevel(dm,&ctx.currLevel));
        pforest->forest->user_pointer = (void*) &ctx;
        PetscStackCallP4est(p4est_coarsen,(pforest->forest,0,pforest_coarsen_currlevel,NULL));
        pforest->forest->user_pointer = (void*) dm;
        PetscStackCallP4est(p4est_balance,(pforest->forest,P4EST_CONNECT_FULL,NULL));
        /* we will have to change the offset after we compute the overlap */
        if (computeAdaptSF) {
          PetscCall(DMPforestComputeLocalCellTransferSF(PetscObjectComm((PetscObject)dm),pforest->forest,0,apforest->forest,apforest->cLocalStart,&coarseToPreFine,NULL));
        }
      } else if (!numValuesGlobal && defaultValue == DM_ADAPT_COARSEN) { /* uniform coarsen */
        PetscCall(DMForestGetMinimumRefinement(dm,&ctx.minLevel));
        pforest->forest->user_pointer = (void*) &ctx;
        PetscStackCallP4est(p4est_coarsen,(pforest->forest,0,pforest_coarsen_uniform,NULL));
        pforest->forest->user_pointer = (void*) dm;
        PetscStackCallP4est(p4est_balance,(pforest->forest,P4EST_CONNECT_FULL,NULL));
        /* we will have to change the offset after we compute the overlap */
        if (computeAdaptSF) {
          PetscCall(DMPforestComputeLocalCellTransferSF(PetscObjectComm((PetscObject)dm),pforest->forest,0,apforest->forest,apforest->cLocalStart,&coarseToPreFine,NULL));
        }
      } else if (!numValuesGlobal && defaultValue == DM_ADAPT_REFINE) { /* uniform refine */
        PetscCall(DMForestGetMaximumRefinement(dm,&ctx.maxLevel));
        pforest->forest->user_pointer = (void*) &ctx;
        PetscStackCallP4est(p4est_refine,(pforest->forest,0,pforest_refine_uniform,NULL));
        pforest->forest->user_pointer = (void*) dm;
        PetscStackCallP4est(p4est_balance,(pforest->forest,P4EST_CONNECT_FULL,NULL));
        /* we will have to change the offset after we compute the overlap */
        if (computeAdaptSF) {
          PetscCall(DMPforestComputeLocalCellTransferSF(PetscObjectComm((PetscObject)dm),apforest->forest,apforest->cLocalStart,pforest->forest,0,&preCoarseToFine,NULL));
        }
      } else if (numValuesGlobal) {
        p4est_t                    *p4est = pforest->forest;
        PetscInt                   *cellFlags;
        DMForestAdaptivityStrategy strategy;
        PetscSF                    cellSF;
        PetscInt                   c, cStart, cEnd;
        PetscBool                  adaptAny;

        PetscCall(DMForestGetMaximumRefinement(dm,&ctx.maxLevel));
        PetscCall(DMForestGetMinimumRefinement(dm,&ctx.minLevel));
        PetscCall(DMForestGetAdaptivityStrategy(dm,&strategy));
        PetscCall(PetscStrncmp(strategy,"any",3,&adaptAny));
        PetscCall(DMForestGetCellChart(adaptFrom,&cStart,&cEnd));
        PetscCall(DMForestGetCellSF(adaptFrom,&cellSF));
        PetscCall(PetscMalloc1(cEnd-cStart,&cellFlags));
        for (c = cStart; c < cEnd; c++) PetscCall(DMLabelGetValue(adaptLabel,c,&cellFlags[c-cStart]));
        if (cellSF) {
          if (adaptAny) {
            PetscCall(PetscSFReduceBegin(cellSF,MPIU_INT,cellFlags,cellFlags,MPI_MAX));
            PetscCall(PetscSFReduceEnd(cellSF,MPIU_INT,cellFlags,cellFlags,MPI_MAX));
          } else {
            PetscCall(PetscSFReduceBegin(cellSF,MPIU_INT,cellFlags,cellFlags,MPI_MIN));
            PetscCall(PetscSFReduceEnd(cellSF,MPIU_INT,cellFlags,cellFlags,MPI_MIN));
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
        PetscCall(PetscFree(cellFlags));

        pforest->forest->user_pointer = (void*) &ctx;
        if (adaptAny) PetscStackCallP4est(p4est_coarsen,(pforest->forest,0,pforest_coarsen_flag_any,pforest_init_determine));
        else PetscStackCallP4est(p4est_coarsen,(pforest->forest,0,pforest_coarsen_flag_all,pforest_init_determine));
        PetscStackCallP4est(p4est_refine,(pforest->forest,0,pforest_refine_flag,NULL));
        pforest->forest->user_pointer = (void*) dm;
        PetscStackCallP4est(p4est_balance,(pforest->forest,P4EST_CONNECT_FULL,NULL));
        if (computeAdaptSF) {
          PetscCall(DMPforestComputeLocalCellTransferSF(PetscObjectComm((PetscObject)dm),apforest->forest,apforest->cLocalStart,pforest->forest,0,&preCoarseToFine,&coarseToPreFine));
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

      PetscCall(DMGetNumLabels(adaptFrom,&numLabels));
      for (l = 0; l < numLabels; l++) {
        PetscBool  isDepth, isCellType, isGhost, isVTK;
        DMLabel    label, labelNew;
        PetscInt   defVal;
        const char *name;

        PetscCall(DMGetLabelName(adaptFrom, l, &name));
        PetscCall(DMGetLabelByNum(adaptFrom, l, &label));
        PetscCall(PetscStrcmp(name,"depth",&isDepth));
        if (isDepth) continue;
        PetscCall(PetscStrcmp(name,"celltype",&isCellType));
        if (isCellType) continue;
        PetscCall(PetscStrcmp(name,"ghost",&isGhost));
        if (isGhost) continue;
        PetscCall(PetscStrcmp(name,"vtk",&isVTK));
        if (isVTK) continue;
        PetscCall(DMCreateLabel(dm,name));
        PetscCall(DMGetLabel(dm,name,&labelNew));
        PetscCall(DMLabelGetDefaultValue(label,&defVal));
        PetscCall(DMLabelSetDefaultValue(labelNew,defVal));
      }
    }
  } else { /* initial */
    PetscInt initLevel, minLevel;

    PetscCall(DMForestGetInitialRefinement(dm,&initLevel));
    PetscCall(DMForestGetMinimumRefinement(dm,&minLevel));
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

      PetscCall(PetscObjectGetOptionsPrefix((PetscObject)dm,&prefix));
      PetscCall(PetscOptionsGetEList(((PetscObject)dm)->options,prefix,"-dm_p4est_refine_pattern",DMRefinePatternName,PATTERN_COUNT,&pattern,&flgPattern));
      PetscCall(PetscOptionsGetInt(((PetscObject)dm)->options,prefix,"-dm_p4est_refine_corner",&corner,NULL));
      PetscCall(PetscOptionsGetIntArray(((PetscObject)dm)->options,prefix,"-dm_p4est_refine_fractal_corners",corners,&ncorner,&flgFractal));
      PetscCall(PetscOptionsGetReal(((PetscObject)dm)->options,prefix,"-dm_p4est_refine_hash_likelihood",&likelihood,NULL));

      if (flgPattern) {
        DMRefinePatternCtx *ctx;
        PetscInt           maxLevel;

        PetscCall(DMForestGetMaximumRefinement(dm,&maxLevel));
        PetscCall(PetscNewLog(dm,&ctx));
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
        PetscCall(PetscFree(ctx));
        pforest->forest->user_pointer = (void*) dm;
      }
    }
  }
  if (pforest->coarsen_hierarchy) {
    PetscInt initLevel, currLevel, minLevel;

    PetscCall(DMPforestGetRefinementLevel(dm,&currLevel));
    PetscCall(DMForestGetInitialRefinement(dm,&initLevel));
    PetscCall(DMForestGetMinimumRefinement(dm,&minLevel));
    if (currLevel > minLevel) {
      DM_Forest_pforest *coarse_pforest;
      DMLabel           coarsen;
      DM                coarseDM;

      PetscCall(DMForestTemplate(dm,MPI_COMM_NULL,&coarseDM));
      PetscCall(DMForestSetAdaptivityPurpose(coarseDM,DM_ADAPT_COARSEN));
      PetscCall(DMLabelCreate(PETSC_COMM_SELF, "coarsen",&coarsen));
      PetscCall(DMLabelSetDefaultValue(coarsen,DM_ADAPT_COARSEN));
      PetscCall(DMForestSetAdaptivityLabel(coarseDM,coarsen));
      PetscCall(DMLabelDestroy(&coarsen));
      PetscCall(DMSetCoarseDM(dm,coarseDM));
      PetscCall(PetscObjectDereference((PetscObject)coarseDM));
      initLevel = currLevel == initLevel ? initLevel - 1 : initLevel;
      PetscCall(DMForestSetInitialRefinement(coarseDM,initLevel));
      PetscCall(DMForestSetMinimumRefinement(coarseDM,minLevel));
      coarse_pforest                    = (DM_Forest_pforest*) ((DM_Forest*) coarseDM->data)->data;
      coarse_pforest->coarsen_hierarchy = PETSC_TRUE;
    }
  }

  { /* repartitioning and overlap */
    PetscMPIInt size, rank;

    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm),&size));
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));
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
          PetscCall(DMPforestComputeOverlappingRanks(size,rank,pforest->forest,forest_copy,&pStart,&pEnd));
          PetscCall(PetscMalloc1((PetscInt) pforest->forest->local_num_quadrants,&repartRoots));
          for (p = pStart; p < pEnd; p++) {
            p4est_gloidx_t preStart = forest_copy->global_first_quadrant[p];
            p4est_gloidx_t preEnd   = forest_copy->global_first_quadrant[p+1];
            PetscInt       q;

            if (preEnd == preStart) continue;
            PetscCheck(preStart <= postStart,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Bad partition overlap computation");
            preEnd = preEnd > postEnd ? postEnd : preEnd;
            for (q = partOffset; q < preEnd; q++) {
              repartRoots[q - postStart].rank  = p;
              repartRoots[q - postStart].index = partOffset - preStart;
            }
            partOffset = preEnd;
          }
          PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)dm),&repartSF));
          PetscCall(PetscSFSetGraph(repartSF,numRoots,numLeaves,NULL,PETSC_OWN_POINTER,repartRoots,PETSC_OWN_POINTER));
          PetscCall(PetscSFSetUp(repartSF));
          if (preCoarseToFine) {
            PetscSF        repartSFembed, preCoarseToFineNew;
            PetscInt       nleaves;
            const PetscInt *leaves;

            PetscCall(PetscSFSetUp(preCoarseToFine));
            PetscCall(PetscSFGetGraph(preCoarseToFine,NULL,&nleaves,&leaves,NULL));
            if (leaves) {
              PetscCall(PetscSFCreateEmbeddedRootSF(repartSF,nleaves,leaves,&repartSFembed));
            } else {
              repartSFembed = repartSF;
              PetscCall(PetscObjectReference((PetscObject)repartSFembed));
            }
            PetscCall(PetscSFCompose(preCoarseToFine,repartSFembed,&preCoarseToFineNew));
            PetscCall(PetscSFDestroy(&preCoarseToFine));
            PetscCall(PetscSFDestroy(&repartSFembed));
            preCoarseToFine = preCoarseToFineNew;
          }
          if (coarseToPreFine) {
            PetscSF repartSFinv, coarseToPreFineNew;

            PetscCall(PetscSFCreateInverseSF(repartSF,&repartSFinv));
            PetscCall(PetscSFCompose(repartSFinv,coarseToPreFine,&coarseToPreFineNew));
            PetscCall(PetscSFDestroy(&coarseToPreFine));
            PetscCall(PetscSFDestroy(&repartSFinv));
            coarseToPreFine = coarseToPreFineNew;
          }
          PetscCall(PetscSFDestroy(&repartSF));
        }
        PetscStackCallP4est(p4est_destroy,(forest_copy));
      }
    }
    if (size > 1) {
      PetscInt overlap;

      PetscCall(DMForestGetPartitionOverlap(dm,&overlap));

      if (adaptFrom) {
        PetscInt aoverlap;

        PetscCall(DMForestGetPartitionOverlap(adaptFrom,&aoverlap));
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
          if (adaptFrom) PetscCall(DMForestGetCellSF(adaptFrom,&preCellSF));
          dm->setupcalled = PETSC_TRUE;
          PetscCall(DMForestGetCellSF(dm,&cellSF));
        }
        if (preCoarseToFine) {
          PetscSF           preCoarseToFineNew;
          PetscInt          nleaves, nroots, *leavesNew, i, nleavesNew;
          const PetscInt    *leaves;
          const PetscSFNode *remotes;
          PetscSFNode       *remotesAll;

          PetscCall(PetscSFSetUp(preCoarseToFine));
          PetscCall(PetscSFGetGraph(preCoarseToFine,&nroots,&nleaves,&leaves,&remotes));
          PetscCall(PetscMalloc1(cEnd,&remotesAll));
          for (i = 0; i < cEnd; i++) {
            remotesAll[i].rank  = -1;
            remotesAll[i].index = -1;
          }
          for (i = 0; i < nleaves; i++) remotesAll[(leaves ? leaves[i] : i) + cLocalStart] = remotes[i];
          PetscCall(PetscSFSetUp(cellSF));
          PetscCall(PetscSFBcastBegin(cellSF,MPIU_2INT,remotesAll,remotesAll,MPI_REPLACE));
          PetscCall(PetscSFBcastEnd(cellSF,MPIU_2INT,remotesAll,remotesAll,MPI_REPLACE));
          nleavesNew = 0;
          for (i = 0; i < nleaves; i++) {
            if (remotesAll[i].rank >= 0) nleavesNew++;
          }
          PetscCall(PetscMalloc1(nleavesNew,&leavesNew));
          nleavesNew = 0;
          for (i = 0; i < nleaves; i++) {
            if (remotesAll[i].rank >= 0) {
              leavesNew[nleavesNew] = i;
              if (i > nleavesNew) remotesAll[nleavesNew] = remotesAll[i];
              nleavesNew++;
            }
          }
          PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)dm),&preCoarseToFineNew));
          if (nleavesNew < cEnd) {
            PetscCall(PetscSFSetGraph(preCoarseToFineNew,nroots,nleavesNew,leavesNew,PETSC_OWN_POINTER,remotesAll,PETSC_COPY_VALUES));
          } else { /* all cells are leaves */
            PetscCall(PetscFree(leavesNew));
            PetscCall(PetscSFSetGraph(preCoarseToFineNew,nroots,nleavesNew,NULL,PETSC_OWN_POINTER,remotesAll,PETSC_COPY_VALUES));
          }
          PetscCall(PetscFree(remotesAll));
          PetscCall(PetscSFDestroy(&preCoarseToFine));
          preCoarseToFine = preCoarseToFineNew;
          preCoarseToFine = preCoarseToFineNew;
        }
        if (coarseToPreFine) {
          PetscSF           coarseToPreFineNew;
          PetscInt          nleaves, nroots, i, nleavesCellSF, nleavesExpanded, *leavesNew;
          const PetscInt    *leaves;
          const PetscSFNode *remotes;
          PetscSFNode       *remotesNew, *remotesNewRoot, *remotesExpanded;

          PetscCall(PetscSFSetUp(coarseToPreFine));
          PetscCall(PetscSFGetGraph(coarseToPreFine,&nroots,&nleaves,&leaves,&remotes));
          PetscCall(PetscSFGetGraph(preCellSF,NULL,&nleavesCellSF,NULL,NULL));
          PetscCall(PetscMalloc1(nroots,&remotesNewRoot));
          PetscCall(PetscMalloc1(nleaves,&remotesNew));
          for (i = 0; i < nroots; i++) {
            remotesNewRoot[i].rank  = rank;
            remotesNewRoot[i].index = i + cLocalStart;
          }
          PetscCall(PetscSFBcastBegin(coarseToPreFine,MPIU_2INT,remotesNewRoot,remotesNew,MPI_REPLACE));
          PetscCall(PetscSFBcastEnd(coarseToPreFine,MPIU_2INT,remotesNewRoot,remotesNew,MPI_REPLACE));
          PetscCall(PetscFree(remotesNewRoot));
          PetscCall(PetscMalloc1(nleavesCellSF,&remotesExpanded));
          for (i = 0; i < nleavesCellSF; i++) {
            remotesExpanded[i].rank  = -1;
            remotesExpanded[i].index = -1;
          }
          for (i = 0; i < nleaves; i++) remotesExpanded[leaves ? leaves[i] : i] = remotesNew[i];
          PetscCall(PetscFree(remotesNew));
          PetscCall(PetscSFBcastBegin(preCellSF,MPIU_2INT,remotesExpanded,remotesExpanded,MPI_REPLACE));
          PetscCall(PetscSFBcastEnd(preCellSF,MPIU_2INT,remotesExpanded,remotesExpanded,MPI_REPLACE));

          nleavesExpanded = 0;
          for (i = 0; i < nleavesCellSF; i++) {
            if (remotesExpanded[i].rank >= 0) nleavesExpanded++;
          }
          PetscCall(PetscMalloc1(nleavesExpanded,&leavesNew));
          nleavesExpanded = 0;
          for (i = 0; i < nleavesCellSF; i++) {
            if (remotesExpanded[i].rank >= 0) {
              leavesNew[nleavesExpanded] = i;
              if (i > nleavesExpanded) remotesExpanded[nleavesExpanded] = remotes[i];
              nleavesExpanded++;
            }
          }
          PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)dm),&coarseToPreFineNew));
          if (nleavesExpanded < nleavesCellSF) {
            PetscCall(PetscSFSetGraph(coarseToPreFineNew,cEnd,nleavesExpanded,leavesNew,PETSC_OWN_POINTER,remotesExpanded,PETSC_COPY_VALUES));
          } else {
            PetscCall(PetscFree(leavesNew));
            PetscCall(PetscSFSetGraph(coarseToPreFineNew,cEnd,nleavesExpanded,NULL,PETSC_OWN_POINTER,remotesExpanded,PETSC_COPY_VALUES));
          }
          PetscCall(PetscFree(remotesExpanded));
          PetscCall(PetscSFDestroy(&coarseToPreFine));
          coarseToPreFine = coarseToPreFineNew;
        }
      }
    }
  }
  forest->preCoarseToFine = preCoarseToFine;
  forest->coarseToPreFine = coarseToPreFine;
  dm->setupcalled         = PETSC_TRUE;
  PetscCallMPI(MPI_Allreduce(&ctx.anyChange,&(pforest->adaptivitySuccess),1,MPIU_BOOL,MPI_LOR,PetscObjectComm((PetscObject)dm)));
  PetscCall(DMPforestGetPlex(dm,NULL));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(DMSetUp(dm));
  switch (viewer->format) {
  case PETSC_VIEWER_DEFAULT:
  case PETSC_VIEWER_ASCII_INFO:
  {
    PetscInt   dim;
    const char *name;

    PetscCall(PetscObjectGetName((PetscObject) dm, &name));
    PetscCall(DMGetDimension(dm, &dim));
    if (name) PetscCall(PetscViewerASCIIPrintf(viewer, "Forest %s in %" PetscInt_FMT " dimensions:\n", name, dim));
    else      PetscCall(PetscViewerASCIIPrintf(viewer, "Forest in %" PetscInt_FMT " dimensions:\n", dim));
  }
  case PETSC_VIEWER_ASCII_INFO_DETAIL:
  case PETSC_VIEWER_LOAD_BALANCE:
  {
    DM plex;

    PetscCall(DMPforestGetPlex(dm, &plex));
    PetscCall(DMView(plex, viewer));
  }
  break;
  default: SETERRQ(PetscObjectComm((PetscObject) dm), PETSC_ERR_SUP, "No support for format '%s'", PetscViewerFormats[viewer->format]);
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(DMSetUp(dm));
  geom = pforest->topo->geom;
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK, &isvtk));
  PetscCheck(isvtk,PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_INCOMP, "Cannot use viewer type %s", ((PetscObject)viewer)->type_name);
  switch (viewer->format) {
  case PETSC_VIEWER_VTK_VTU:
    PetscCheck(pforest->forest,PetscObjectComm(odm),PETSC_ERR_ARG_WRONG,"DM has not been setup with a valid forest");
    name = vtk->filename;
    PetscCall(PetscStrlen(name,&len));
    PetscCall(PetscStrcasecmp(name+len-4,".vtu",&hasExt));
    if (hasExt) {
      PetscCall(PetscStrallocpy(name,&filenameStrip));
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
      PetscCheck(pvtk,PetscObjectComm((PetscObject)odm),PETSC_ERR_LIB,P4EST_STRING "_vtk_write_header() failed");
      PetscStackCallP4estReturn(pvtk,p4est_vtk_write_cell_dataf,(pvtk,
                                                                 1, /* write tree */
                                                                 1, /* write level */
                                                                 1, /* write rank */
                                                                 0, /* do not wrap rank */
                                                                 0, /* no scalar fields */
                                                                 0, /* no vector fields */
                                                                 pvtk));
      PetscCheck(pvtk,PetscObjectComm((PetscObject)odm),PETSC_ERR_LIB,P4EST_STRING "_vtk_write_cell_dataf() failed");
      PetscStackCallP4estReturn(footerr,p4est_vtk_write_footer,(pvtk));
      PetscCheck(!footerr,PetscObjectComm((PetscObject)odm),PETSC_ERR_LIB,P4EST_STRING "_vtk_write_footer() failed");
    }
    if (!pforest->topo->geom) PetscStackCallP4est(p4est_geometry_destroy,(geom));
    PetscCall(PetscFree(filenameStrip));
    break;
  default: SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "No support for format '%s'", PetscViewerFormats[viewer->format]);
  }
  PetscFunctionReturn(0);
}

#define DMView_HDF5_pforest _append_pforest(DMView_HDF5)
static PetscErrorCode DMView_HDF5_pforest(DM dm, PetscViewer viewer)
{
  DM             plex;

  PetscFunctionBegin;
  PetscCall(DMSetUp(dm));
  PetscCall(DMPforestGetPlex(dm, &plex));
  PetscCall(DMView(plex, viewer));
  PetscFunctionReturn(0);
}

#define DMView_GLVis_pforest _append_pforest(DMView_GLVis)
static PetscErrorCode DMView_GLVis_pforest(DM dm, PetscViewer viewer)
{
  DM             plex;

  PetscFunctionBegin;
  PetscCall(DMSetUp(dm));
  PetscCall(DMPforestGetPlex(dm, &plex));
  PetscCall(DMView(plex, viewer));
  PetscFunctionReturn(0);
}

#define DMView_pforest _append_pforest(DMView)
static PetscErrorCode DMView_pforest(DM dm, PetscViewer viewer)
{
  PetscBool      isascii, isvtk, ishdf5, isglvis;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii));
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERVTK,   &isvtk));
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERHDF5,  &ishdf5));
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERGLVIS, &isglvis));
  if (isascii) {
    PetscCall(DMView_ASCII_pforest((PetscObject) dm,viewer));
  } else if (isvtk) {
    PetscCall(DMView_VTK_pforest((PetscObject) dm,viewer));
  } else if (ishdf5) {
    PetscCall(DMView_HDF5_pforest(dm, viewer));
  } else if (isglvis) {
    PetscCall(DMView_GLVis_pforest(dm, viewer));
  } else SETERRQ(PetscObjectComm((PetscObject) dm),PETSC_ERR_SUP,"Viewer not supported (not VTK, HDF5, or GLVis)");
  PetscFunctionReturn(0);
}

static PetscErrorCode PforestConnectivityEnumerateFacets(p4est_connectivity_t *conn, PetscInt **tree_face_to_uniq)
{
  PetscInt       *ttf, f, t, g, count;
  PetscInt       numFacets;

  PetscFunctionBegin;
  numFacets = conn->num_trees * P4EST_FACES;
  PetscCall(PetscMalloc1(numFacets,&ttf));
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
  PetscInt             cStart, cEnd, c, vStart, vEnd, v, fStart, fEnd, f;
  PetscInt             *star = NULL, *closure = NULL, closureSize, starSize, cttSize;
  PetscInt             *ttf;

  PetscFunctionBegin;
  /* 1: count objects, allocate */
  PetscCall(DMPlexGetSimplexOrBoxCells(dm,0,&cStart,&cEnd));
  PetscCall(P4estTopidxCast(cEnd-cStart,&numTrees));
  numVerts = P4EST_CHILDREN * numTrees;
  PetscCall(DMPlexGetDepthStratum(dm,0,&vStart,&vEnd));
  PetscCall(P4estTopidxCast(vEnd-vStart,&numCorns));
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF,&ctt));
  PetscCall(PetscSectionSetChart(ctt,vStart,vEnd));
  for (v = vStart; v < vEnd; v++) {
    PetscInt s;

    PetscCall(DMPlexGetTransitiveClosure(dm,v,PETSC_FALSE,&starSize,&star));
    for (s = 0; s < starSize; s++) {
      PetscInt p = star[2*s];

      if (p >= cStart && p < cEnd) {
        /* we want to count every time cell p references v, so we see how many times it comes up in the closure.  This
         * only protects against periodicity problems */
        PetscCall(DMPlexGetTransitiveClosure(dm,p,PETSC_TRUE,&closureSize,&closure));
        PetscCheck(closureSize == P4EST_INSUL,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Cell %" PetscInt_FMT " with wrong closure size %" PetscInt_FMT " != %d", p, closureSize, P4EST_INSUL);
        for (c = 0; c < P4EST_CHILDREN; c++) {
          PetscInt cellVert = closure[2 * (c + vertOff)];

          PetscCheck(cellVert >= vStart && cellVert < vEnd,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Non-standard closure: vertices");
          if (cellVert == v) {
            PetscCall(PetscSectionAddDof(ctt,v,1));
          }
        }
        PetscCall(DMPlexRestoreTransitiveClosure(dm,p,PETSC_TRUE,&closureSize,&closure));
      }
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm,v,PETSC_FALSE,&starSize,&star));
  }
  PetscCall(PetscSectionSetUp(ctt));
  PetscCall(PetscSectionGetStorageSize(ctt,&cttSize));
  PetscCall(P4estTopidxCast(cttSize,&numCtt));
#if defined(P4_TO_P8)
  PetscCall(DMPlexGetSimplexOrBoxCells(dm,P4EST_DIM-1,&eStart,&eEnd));
  PetscCall(P4estTopidxCast(eEnd-eStart,&numEdges));
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF,&ett));
  PetscCall(PetscSectionSetChart(ett,eStart,eEnd));
  for (e = eStart; e < eEnd; e++) {
    PetscInt s;

    PetscCall(DMPlexGetTransitiveClosure(dm,e,PETSC_FALSE,&starSize,&star));
    for (s = 0; s < starSize; s++) {
      PetscInt p = star[2*s];

      if (p >= cStart && p < cEnd) {
        /* we want to count every time cell p references e, so we see how many times it comes up in the closure.  This
         * only protects against periodicity problems */
        PetscCall(DMPlexGetTransitiveClosure(dm,p,PETSC_TRUE,&closureSize,&closure));
        PetscCheck(closureSize == P4EST_INSUL,PETSC_COMM_SELF,PETSC_ERR_ARG_INCOMP,"Cell with wrong closure size");
        for (c = 0; c < P8EST_EDGES; c++) {
          PetscInt cellEdge = closure[2 * (c + edgeOff)];

          PetscCheck(cellEdge >= eStart && cellEdge < eEnd,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Non-standard closure: edges");
          if (cellEdge == e) {
            PetscCall(PetscSectionAddDof(ett,e,1));
          }
        }
        PetscCall(DMPlexRestoreTransitiveClosure(dm,p,PETSC_TRUE,&closureSize,&closure));
      }
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm,e,PETSC_FALSE,&starSize,&star));
  }
  PetscCall(PetscSectionSetUp(ett));
  PetscCall(PetscSectionGetStorageSize(ett,&ettSize));
  PetscCall(P4estTopidxCast(ettSize,&numEtt));

  /* This routine allocates space for the arrays, which we fill below */
  PetscStackCallP4estReturn(conn,p8est_connectivity_new,(numVerts,numTrees,numEdges,numEtt,numCorns,numCtt));
#else
  PetscStackCallP4estReturn(conn,p4est_connectivity_new,(numVerts,numTrees,numCorns,numCtt));
#endif

  /* 2: visit every face, determine neighboring cells(trees) */
  PetscCall(DMPlexGetSimplexOrBoxCells(dm,1,&fStart,&fEnd));
  PetscCall(PetscMalloc1((cEnd-cStart) * P4EST_FACES,&ttf));
  for (f = fStart; f < fEnd; f++) {
    PetscInt       numSupp, s;
    PetscInt       myFace[2] = {-1, -1};
    PetscInt       myOrnt[2] = {PETSC_MIN_INT, PETSC_MIN_INT};
    const PetscInt *supp;

    PetscCall(DMPlexGetSupportSize(dm, f, &numSupp));
    PetscCheck(numSupp == 1 || numSupp == 2,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"point %" PetscInt_FMT " has facet with %" PetscInt_FMT " sides: must be 1 or 2 (boundary or conformal)",f,numSupp);
    PetscCall(DMPlexGetSupport(dm, f, &supp));

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
      DMPolytopeType ct;
      const PetscInt *cone;
      const PetscInt *ornt;
      PetscInt       orient = PETSC_MIN_INT;

      PetscCall(DMPlexGetConeSize(dm, p, &numCone));
      PetscCheck(numCone == P4EST_FACES,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"cell %" PetscInt_FMT " has %" PetscInt_FMT " facets, expect %d",p,numCone,P4EST_FACES);
      PetscCall(DMPlexGetCone(dm, p, &cone));
      PetscCall(DMPlexGetCellType(dm, cone[0], &ct));
      PetscCall(DMPlexGetConeOrientation(dm, p, &ornt));
      for (i = 0; i < P4EST_FACES; i++) {
        if (cone[i] == f) {
          orient = DMPolytopeConvertNewOrientation_Internal(ct, ornt[i]);
          break;
        }
      }
      PetscCheck(i < P4EST_FACES,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"cell %" PetscInt_FMT " faced %" PetscInt_FMT " mismatch",p,f);
      if (p < cStart || p >= cEnd) {
        DMPolytopeType ct;
        PetscCall(DMPlexGetCellType(dm, p, &ct));
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"cell %" PetscInt_FMT " (%s) should be in [%" PetscInt_FMT ", %" PetscInt_FMT ")",p,DMPolytopeTypes[ct],cStart,cEnd);
      }
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
        myOrnt[s] = DihedralCompose(N,orient,DMPolytopeConvertNewOrientation_Internal(ct, P4estFaceToPetscOrnt[myFace[s]]));
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

    PetscCall(PetscSectionGetOffset(ett,e,&off));
    conn->ett_offset[e - eStart] = (p4est_topidx_t) off;
    PetscCall(DMPlexGetTransitiveClosure(dm,e,PETSC_FALSE,&starSize,&star));
    for (s = 0; s < starSize; s++) {
      PetscInt p = star[2 * s];

      if (p >= cStart && p < cEnd) {
        PetscCall(DMPlexGetTransitiveClosure(dm,p,PETSC_TRUE,&closureSize,&closure));
        PetscCheck(closureSize == P4EST_INSUL,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Non-standard closure");
        for (c = 0; c < P8EST_EDGES; c++) {
          PetscInt cellEdge = closure[2 * (c + edgeOff)];
          PetscInt cellOrnt = closure[2 * (c + edgeOff) + 1];
          DMPolytopeType ct;

          PetscCall(DMPlexGetCellType(dm, cellEdge, &ct));
          cellOrnt = DMPolytopeConvertNewOrientation_Internal(ct, cellOrnt);
          if (cellEdge == e) {
            PetscInt p4estEdge = PetscEdgeToP4estEdge[c];
            PetscInt totalOrient;

            /* compose p4est-closure to petsc-closure permutation and petsc-closure to edge orientation */
            totalOrient = DihedralCompose(2,cellOrnt,DMPolytopeConvertNewOrientation_Internal(DM_POLYTOPE_SEGMENT, P4estEdgeToPetscOrnt[p4estEdge]));
            /* p4est orientations are positive: -2 => 1, -1 => 0 */
            totalOrient             = (totalOrient < 0) ? -(totalOrient + 1) : totalOrient;
            conn->edge_to_tree[off] = (p4est_locidx_t) (p - cStart);
            /* encode cell-edge and orientation in edge_to_edge per p8est_connectivity standart (see
             * p8est_connectivity.h) */
            conn->edge_to_edge[off++] = (int8_t) p4estEdge + P8EST_EDGES * totalOrient;
            conn->tree_to_edge[P8EST_EDGES * (p - cStart) + p4estEdge] = e - eStart;
          }
        }
        PetscCall(DMPlexRestoreTransitiveClosure(dm,p,PETSC_TRUE,&closureSize,&closure));
      }
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm,e,PETSC_FALSE,&starSize,&star));
  }
  PetscCall(PetscSectionDestroy(&ett));
#endif

  /* 4: visit every vertex */
  conn->ctt_offset[0] = 0;
  for (v = vStart; v < vEnd; v++) {
    PetscInt off, s;

    PetscCall(PetscSectionGetOffset(ctt,v,&off));
    conn->ctt_offset[v - vStart] = (p4est_topidx_t) off;
    PetscCall(DMPlexGetTransitiveClosure(dm,v,PETSC_FALSE,&starSize,&star));
    for (s = 0; s < starSize; s++) {
      PetscInt p = star[2 * s];

      if (p >= cStart && p < cEnd) {
        PetscCall(DMPlexGetTransitiveClosure(dm,p,PETSC_TRUE,&closureSize,&closure));
        PetscCheck(closureSize == P4EST_INSUL,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Non-standard closure");
        for (c = 0; c < P4EST_CHILDREN; c++) {
          PetscInt cellVert = closure[2 * (c + vertOff)];

          if (cellVert == v) {
            PetscInt p4estVert = PetscVertToP4estVert[c];

            conn->corner_to_tree[off]     = (p4est_locidx_t) (p - cStart);
            conn->corner_to_corner[off++] = (int8_t) p4estVert;
            conn->tree_to_corner[P4EST_CHILDREN * (p - cStart) + p4estVert] = v - vStart;
          }
        }
        PetscCall(DMPlexRestoreTransitiveClosure(dm,p,PETSC_TRUE,&closureSize,&closure));
      }
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm,v,PETSC_FALSE,&starSize,&star));
  }
  PetscCall(PetscSectionDestroy(&ctt));

  /* 5: Compute the coordinates */
  {
    PetscInt     coordDim;
    Vec          coordVec;
    PetscSection coordSec;
    PetscBool    localized;

    PetscCall(DMGetCoordinateDim(dm, &coordDim));
    PetscCall(DMGetCoordinatesLocal(dm, &coordVec));
    PetscCall(DMGetCoordinatesLocalizedLocal(dm, &localized));
    PetscCall(DMGetCoordinateSection(dm, &coordSec));
    for (c = cStart; c < cEnd; c++) {
      PetscInt    dof;
      PetscScalar *cellCoords = NULL;

      PetscCall(DMPlexVecGetClosure(dm, coordSec, coordVec, c, &dof, &cellCoords));
      PetscCheck(localized || dof == P4EST_CHILDREN * coordDim,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Need coordinates at the corners: (dof) %" PetscInt_FMT " != %d * %" PetscInt_FMT " (sdim)", dof, P4EST_CHILDREN, coordDim);
      for (v = 0; v < P4EST_CHILDREN; v++) {
        PetscInt i, lim = PetscMin(3, coordDim);
        PetscInt p4estVert = PetscVertToP4estVert[v];

        conn->tree_to_vertex[P4EST_CHILDREN * (c - cStart) + v] = P4EST_CHILDREN * (c - cStart) + v;
        /* p4est vertices are always embedded in R^3 */
        for (i = 0; i < 3; i++)   conn->vertices[3 * (P4EST_CHILDREN * (c - cStart) + p4estVert) + i] = 0.;
        for (i = 0; i < lim; i++) conn->vertices[3 * (P4EST_CHILDREN * (c - cStart) + p4estVert) + i] = PetscRealPart(cellCoords[v * coordDim + i]);
      }
      PetscCall(DMPlexVecRestoreClosure(dm, coordSec, coordVec, c, &dof, &cellCoords));
    }
  }

#if defined(P4EST_ENABLE_DEBUG)
  PetscCheck(p4est_connectivity_is_valid(conn),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Plex to p4est conversion failed");
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
  PetscCheck(array->elem_size == sizeof(p4est_locidx_t),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong locidx size");

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
  PetscCheck(array->elem_size == 3 * sizeof(double),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong coordinate size");
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
  PetscCheck(array->elem_size == 2 * sizeof(p4est_locidx_t),PETSC_COMM_SELF,PETSC_ERR_PLIB,"Wrong locidx size");

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

    PetscCall(locidx_to_PetscInt(points_per_dim));
    PetscCall(locidx_to_PetscInt(cone_sizes));
    PetscCall(locidx_to_PetscInt(cones));
    PetscCall(locidx_to_PetscInt(cone_orientations));
    PetscCall(coords_double_to_PetscScalar(coords, P4EST_DIM));

    PetscCall(DMPlexCreate(PETSC_COMM_SELF,plex));
    PetscCall(DMSetDimension(*plex,P4EST_DIM));
    PetscCall(DMPlexCreateFromDAG(*plex,P4EST_DIM,(PetscInt*)points_per_dim->array,(PetscInt*)cone_sizes->array,(PetscInt*)cones->array,(PetscInt*)cone_orientations->array,(PetscScalar*)coords->array));
    PetscCall(DMPlexConvertOldOrientations_Internal(*plex));
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

  PetscFunctionBegin;
  if (parentOrientA == parentOrientB) {
    if (childOrientB) *childOrientB = childOrientA;
    if (childB) *childB = childA;
    PetscFunctionReturn(0);
  }
  PetscCall(DMPlexGetDepthStratum(dm,0,&vStart,&vEnd));
  if (childA >= vStart && childA < vEnd) { /* vertices (always in the middle) are invariant under rotation */
    if (childOrientB) *childOrientB = 0;
    if (childB) *childB = childA;
    PetscFunctionReturn(0);
  }
  for (dim = 0; dim < 3; dim++) {
    PetscCall(DMPlexGetDepthStratum(dm,dim,&dStart,&dEnd));
    if (parent >= dStart && parent <= dEnd) break;
  }
  PetscCheck(dim <= 2,PETSC_COMM_SELF,PETSC_ERR_SUP,"Cannot perform child symmetry for %" PetscInt_FMT "-cells",dim);
  PetscCheck(dim,PETSC_COMM_SELF,PETSC_ERR_PLIB,"A vertex has no children");
  if (childA < dStart || childA >= dEnd) { /* a 1-cell in a 2-cell */
    /* this is a lower-dimensional child: bootstrap */
    PetscInt       size, i, sA = -1, sB, sOrientB, sConeSize;
    const PetscInt *supp, *coneA, *coneB, *oA, *oB;

    PetscCall(DMPlexGetSupportSize(dm,childA,&size));
    PetscCall(DMPlexGetSupport(dm,childA,&supp));

    /* find a point sA in supp(childA) that has the same parent */
    for (i = 0; i < size; i++) {
      PetscInt sParent;

      sA = supp[i];
      if (sA == parent) continue;
      PetscCall(DMPlexGetTreeParent(dm,sA,&sParent,NULL));
      if (sParent == parent) break;
    }
    PetscCheck(i != size,PETSC_COMM_SELF,PETSC_ERR_PLIB,"could not find support in children");
    /* find out which point sB is in an equivalent position to sA under
     * parentOrientB */
    PetscCall(DMReferenceTreeGetChildSymmetry_pforest(dm,parent,parentOrientA,0,sA,parentOrientB,&sOrientB,&sB));
    PetscCall(DMPlexGetConeSize(dm,sA,&sConeSize));
    PetscCall(DMPlexGetCone(dm,sA,&coneA));
    PetscCall(DMPlexGetCone(dm,sB,&coneB));
    PetscCall(DMPlexGetConeOrientation(dm,sA,&oA));
    PetscCall(DMPlexGetConeOrientation(dm,sB,&oB));
    /* step through the cone of sA in natural order */
    for (i = 0; i < sConeSize; i++) {
      if (coneA[i] == childA) {
        /* if childA is at position i in coneA,
         * then we want the point that is at sOrientB*i in coneB */
        PetscInt j = (sOrientB >= 0) ? ((sOrientB + i) % sConeSize) : ((sConeSize -(sOrientB+1) - i) % sConeSize);
        if (childB) *childB = coneB[j];
        if (childOrientB) {
          DMPolytopeType ct;
          PetscInt       oBtrue;

          PetscCall(DMPlexGetConeSize(dm,childA,&coneSize));
          /* compose sOrientB and oB[j] */
          PetscCheck(coneSize == 0 || coneSize == 2,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Expected a vertex or an edge");
          ct = coneSize ? DM_POLYTOPE_SEGMENT : DM_POLYTOPE_POINT;
          /* we may have to flip an edge */
          oBtrue        = (sOrientB >= 0) ? oB[j] : DMPolytopeTypeComposeOrientationInv(ct, -1, oB[j]);
          oBtrue        = DMPolytopeConvertNewOrientation_Internal(ct, oBtrue);
          ABswap        = DihedralSwap(coneSize,DMPolytopeConvertNewOrientation_Internal(ct, oA[i]),oBtrue);
          *childOrientB = DihedralCompose(coneSize,childOrientA,ABswap);
        }
        break;
      }
    }
    PetscCheck(i != sConeSize,PETSC_COMM_SELF,PETSC_ERR_PLIB,"support cone mismatch");
    PetscFunctionReturn(0);
  }
  /* get the cone size and symmetry swap */
  PetscCall(DMPlexGetConeSize(dm,parent,&coneSize));
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
    PetscCall(DMPlexGetTreeChildren(dm,parent,&numChildren,&children));
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

  PetscFunctionBegin;
  PetscStackCallP4estReturn(refcube,p4est_connectivity_new_byname,("unit"));
  { /* [-1,1]^d geometry */
    PetscInt i, j;

    for (i = 0; i < P4EST_CHILDREN; i++) {
      for (j = 0; j < 3; j++) {
        refcube->vertices[3 * i + j] *= 2.;
        refcube->vertices[3 * i + j] -= 1.;
      }
    }
  }
  PetscStackCallP4estReturn(root,p4est_new,(PETSC_COMM_SELF,refcube,0,NULL,NULL));
  PetscStackCallP4estReturn(refined,p4est_new_ext,(PETSC_COMM_SELF,refcube,0,1,1,0,NULL,NULL));
  PetscCall(P4estToPlex_Local(root,&dmRoot));
  PetscCall(P4estToPlex_Local(refined,&dmRefined));
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

    PetscCall(ISCreateGeneral(PETSC_COMM_SELF,nPoints,perm,PETSC_USE_POINTER,&permIS));
    PetscCall(DMPlexPermute(dmRefined,permIS,&dmPerm));
    if (dmPerm) {
      PetscCall(DMDestroy(&dmRefined));
      dmRefined = dmPerm;
    }
    PetscCall(ISDestroy(&permIS));
    {
      PetscInt p;
      PetscCall(DMCreateLabel(dmRoot,"identity"));
      PetscCall(DMCreateLabel(dmRefined,"identity"));
      for (p = 0; p < P4EST_INSUL; p++) {
        PetscCall(DMSetLabelValue(dmRoot,"identity",p,p));
      }
      for (p = 0; p < nPoints; p++) {
        PetscCall(DMSetLabelValue(dmRefined,"identity",p,ident[p]));
      }
    }
  }
  PetscCall(DMPlexCreateReferenceTree_Union(dmRoot,dmRefined,"identity",dm));
  mesh                   = (DM_Plex*) (*dm)->data;
  mesh->getchildsymmetry = DMReferenceTreeGetChildSymmetry_pforest;
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  if (rank == 0) {
    PetscCall(DMViewFromOptions(dmRoot,   NULL,"-dm_p4est_ref_root_view"));
    PetscCall(DMViewFromOptions(dmRefined,NULL,"-dm_p4est_ref_refined_view"));
    PetscCall(DMViewFromOptions(dmRefined,NULL,"-dm_p4est_ref_tree_view"));
  }
  PetscCall(DMDestroy(&dmRefined));
  PetscCall(DMDestroy(&dmRoot));
  PetscStackCallP4est(p4est_destroy,(refined));
  PetscStackCallP4est(p4est_destroy,(root));
  PetscStackCallP4est(p4est_connectivity_destroy,(refcube));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMShareDiscretization(DM dmA, DM dmB)
{
  void          *ctx;
  PetscInt       num;
  PetscReal      val;

  PetscFunctionBegin;
  PetscCall(DMGetApplicationContext(dmA,&ctx));
  PetscCall(DMSetApplicationContext(dmB,ctx));
  PetscCall(DMCopyDisc(dmA,dmB));
  PetscCall(DMGetOutputSequenceNumber(dmA,&num,&val));
  PetscCall(DMSetOutputSequenceNumber(dmB,num,val));
  if (dmB->localSection != dmA->localSection || dmB->globalSection != dmA->globalSection) {
    PetscCall(DMClearLocalVectors(dmB));
    PetscCall(PetscObjectReference((PetscObject)dmA->localSection));
    PetscCall(PetscSectionDestroy(&(dmB->localSection)));
    dmB->localSection = dmA->localSection;
    PetscCall(DMClearGlobalVectors(dmB));
    PetscCall(PetscObjectReference((PetscObject)dmA->globalSection));
    PetscCall(PetscSectionDestroy(&(dmB->globalSection)));
    dmB->globalSection = dmA->globalSection;
    PetscCall(PetscObjectReference((PetscObject)dmA->defaultConstraint.section));
    PetscCall(PetscSectionDestroy(&(dmB->defaultConstraint.section)));
    dmB->defaultConstraint.section = dmA->defaultConstraint.section;
    PetscCall(PetscObjectReference((PetscObject)dmA->defaultConstraint.mat));
    PetscCall(MatDestroy(&(dmB->defaultConstraint.mat)));
    dmB->defaultConstraint.mat = dmA->defaultConstraint.mat;
    if (dmA->map) PetscCall(PetscLayoutReference(dmA->map, &dmB->map));
  }
  if (dmB->sectionSF != dmA->sectionSF) {
    PetscCall(PetscObjectReference((PetscObject)dmA->sectionSF));
    PetscCall(PetscSFDestroy(&dmB->sectionSF));
    dmB->sectionSF = dmA->sectionSF;
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

  PetscFunctionBegin;
  PetscCall(DMPforestComputeOverlappingRanks(p4estC->mpisize,p4estC->mpirank,p4estF,p4estC,&startC,&endC));
  PetscCall(PetscMalloc2(2*(endC-startC),&recv,endC-startC,&recvReqs));
  PetscCall(PetscCommGetNewTag(comm,&tag));
  for (p = startC; p < endC; p++) {
    recvReqs[p-startC] = MPI_REQUEST_NULL; /* just in case we don't initiate a receive */
    if (p4estC->global_first_quadrant[p] == p4estC->global_first_quadrant[p+1]) { /* empty coarse partition */
      recv[2*(p-startC)]   = 0;
      recv[2*(p-startC)+1] = 0;
      continue;
    }

    PetscCallMPI(MPI_Irecv(&recv[2*(p-startC)],2,MPIU_INT,p,tag,comm,&recvReqs[p-startC]));
  }
  PetscCall(DMPforestComputeOverlappingRanks(p4estC->mpisize,p4estC->mpirank,p4estC,p4estF,&startF,&endF));
  PetscCall(PetscMalloc2(2*(endF-startF),&send,endF-startF,&sendReqs));
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
    PetscCallMPI(MPI_Isend(&send[2*(p-startF)],2,MPIU_INT,p,tag,comm,&sendReqs[p-startF]));
  }
  PetscCallMPI(MPI_Waitall((PetscMPIInt)(endC-startC),recvReqs,MPI_STATUSES_IGNORE));
  PetscCall(PetscSectionCreate(PETSC_COMM_SELF,&section));
  PetscCall(PetscSectionSetChart(section,startC,endC));
  for (p = startC; p < endC; p++) {
    PetscInt numCells = recv[2*(p-startC)+1];
    PetscCall(PetscSectionSetDof(section,p,numCells));
  }
  PetscCall(PetscSectionSetUp(section));
  PetscCall(PetscSectionGetStorageSize(section,&nLeaves));
  PetscCall(PetscMalloc1(nLeaves,&leaves));
  for (p = startC; p < endC; p++) {
    PetscInt firstCell = recv[2*(p-startC)];
    PetscInt numCells  = recv[2*(p-startC)+1];
    PetscInt off, i;

    PetscCall(PetscSectionGetOffset(section,p,&off));
    for (i = 0; i < numCells; i++) {
      leaves[off+i].rank  = p;
      leaves[off+i].index = firstCell + i;
    }
  }
  PetscCall(PetscSFCreate(comm,&sf));
  PetscCall(PetscSFSetGraph(sf,cEnd-cStart,nLeaves,NULL,PETSC_OWN_POINTER,leaves,PETSC_OWN_POINTER));
  PetscCall(PetscSectionDestroy(&section));
  PetscCallMPI(MPI_Waitall((PetscMPIInt)(endF-startF),sendReqs,MPI_STATUSES_IGNORE));
  PetscCall(PetscFree2(send,sendReqs));
  PetscCall(PetscFree2(recv,recvReqs));
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

  PetscFunctionBegin;
  forest            = (DM_Forest *) dm->data;
  pforest           = (DM_Forest_pforest *) forest->data;
  cStart            = pforest->cLocalStart;
  cEnd              = pforest->cLocalEnd;
  PetscCall(DMPforestGetPlex(dm,&plex));
  PetscCall(DMGetPointSF(dm,&pointSF));
  PetscCall(PetscSFGetGraph(pointSF,&nroots,&nleaves,&ilocal,&iremote));
  nleaves           = PetscMax(0,nleaves);
  nroots            = PetscMax(0,nroots);
  *numClosurePoints = numClosureIndices * (cEnd - cStart);
  PetscCall(PetscMalloc1(*numClosurePoints,closurePoints));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));
  for (c = cStart, count = 0; c < cEnd; c++) {
    PetscInt i;
    PetscCall(DMPlexGetTransitiveClosure(plex,c,PETSC_TRUE,&closureSize,&closure));

    for (i = 0; i < numClosureIndices; i++, count++) {
      PetscInt p   = closure[2 * i];
      PetscInt loc = -1;

      PetscCall(PetscFindInt(p,nleaves,ilocal,&loc));
      if (redirect && loc >= 0) {
        (*closurePoints)[count].rank  = iremote[loc].rank;
        (*closurePoints)[count].index = iremote[loc].index;
      } else {
        (*closurePoints)[count].rank  = rank;
        (*closurePoints)[count].index = p;
      }
    }
    PetscCall(DMPlexRestoreTransitiveClosure(plex,c,PETSC_TRUE,&closureSize,&closure));
  }
  PetscFunctionReturn(0);
}

static void MPIAPI DMPforestMaxSFNode(void *a, void *b, PetscMPIInt *len, MPI_Datatype *type)
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

  PetscFunctionBegin;
  pforestC = (DM_Forest_pforest*) ((DM_Forest*) coarse->data)->data;
  pforestF = (DM_Forest_pforest*) ((DM_Forest*) fine->data)->data;
  p4estC   = pforestC->forest;
  p4estF   = pforestF->forest;
  PetscCheck(pforestC->topo == pforestF->topo,PetscObjectComm((PetscObject)coarse),PETSC_ERR_ARG_INCOMP,"DM's must have the same base DM");
  comm = PetscObjectComm((PetscObject)coarse);
  PetscCallMPI(MPI_Comm_rank(comm,&rank));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  PetscCall(DMPforestGetPlex(fine,&plexF));
  PetscCall(DMPlexGetChart(plexF,&pStartF,&pEndF));
  PetscCall(DMPforestGetPlex(coarse,&plexC));
  PetscCall(DMPlexGetChart(plexC,&pStartC,&pEndC));
  { /* check if the results have been cached */
    DM adaptCoarse, adaptFine;

    PetscCall(DMForestGetAdaptivityForest(coarse,&adaptCoarse));
    PetscCall(DMForestGetAdaptivityForest(fine,&adaptFine));
    if (adaptCoarse && adaptCoarse->data == fine->data) { /* coarse is adapted from fine */
      if (pforestC->pointSelfToAdaptSF) {
        PetscCall(PetscObjectReference((PetscObject)(pforestC->pointSelfToAdaptSF)));
        *sf  = pforestC->pointSelfToAdaptSF;
        if (childIds) {
          PetscCall(PetscMalloc1(pEndF-pStartF,&cids));
          PetscCall(PetscArraycpy(cids,pforestC->pointSelfToAdaptCids,pEndF-pStartF));
          *childIds = cids;
        }
        PetscFunctionReturn(0);
      } else {
        saveInCoarse = PETSC_TRUE;
        formCids     = PETSC_TRUE;
      }
    } else if (adaptFine && adaptFine->data == coarse->data) { /* fine is adapted from coarse */
      if (pforestF->pointAdaptToSelfSF) {
        PetscCall(PetscObjectReference((PetscObject)(pforestF->pointAdaptToSelfSF)));
        *sf  = pforestF->pointAdaptToSelfSF;
        if (childIds) {
          PetscCall(PetscMalloc1(pEndF-pStartF,&cids));
          PetscCall(PetscArraycpy(cids,pforestF->pointAdaptToSelfCids,pEndF-pStartF));
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
  PetscCallMPI(MPI_Type_contiguous(2,MPIU_INT,&nodeType));
  PetscCallMPI(MPI_Type_commit(&nodeType));
  PetscCallMPI(MPI_Op_create(DMPforestMaxSFNode,PETSC_FALSE,&sfNodeReduce));
  PetscCallMPI(MPI_Type_contiguous(numClosureIndices*2,MPIU_INT,&nodeClosureType));
  PetscCallMPI(MPI_Type_commit(&nodeClosureType));
  /* everything has to go through cells: for each cell, create a list of the sfnodes in its closure */
  /* get lists of closure point SF nodes for every cell */
  PetscCall(DMPforestGetCellSFNodes(coarse,numClosureIndices,&numClosurePointsC,&closurePointsC,PETSC_TRUE));
  PetscCall(DMPforestGetCellSFNodes(fine  ,numClosureIndices,&numClosurePointsF,&closurePointsF,PETSC_FALSE));
  /* create pointers for tree lists */
  fltF = p4estF->first_local_tree;
  lltF = p4estF->last_local_tree;
  PetscCall(PetscCalloc2(lltF + 1  - fltF, &treeQuads, lltF + 1 - fltF, &treeQuadCounts));
  /* if the partitions don't match, ship the coarse to cover the fine */
  if (size > 1) {
    PetscInt p;

    for (p = 0; p < size; p++) {
      int equal;

      PetscStackCallP4estReturn(equal,p4est_quadrant_is_equal_piggy,(&p4estC->global_first_position[p],&p4estF->global_first_position[p]));
      if (!equal) break;
    }
    if (p < size) { /* non-matching distribution: send the coarse to cover the fine */
      PetscInt         cStartC, cEndC;
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

      PetscCall(DMPlexGetSimplexOrBoxCells(plexC,0,&cStartC,&cEndC));
      PetscCall(DMPforestGetCellCoveringSF(comm,p4estC,p4estF,pforestC->cLocalStart,pforestC->cLocalEnd,&coveringSF));
      PetscCall(PetscSFGetGraph(coveringSF,NULL,&nleaves,NULL,NULL));
      PetscCall(PetscMalloc1(numClosureIndices*nleaves,&newClosurePointsC));
      PetscCall(PetscMalloc1(nleaves,&coverQuads));
      PetscCall(PetscMalloc1(cEndC-cStartC,&coverQuadsSend));
      count = 0;
      for (t = fltC; t <= lltC; t++) { /* unfortunately, we need to pack a send array, since quads are not stored packed in p4est */
        p4est_tree_t *tree = &(((p4est_tree_t*) p4estC->trees->array)[t]);
        PetscInt     q;

        PetscCall(PetscMemcpy(&coverQuadsSend[count],tree->quadrants.array,tree->quadrants.elem_count * sizeof(p4est_quadrant_t)));
        for (q = 0; (size_t) q < tree->quadrants.elem_count; q++) coverQuadsSend[count+q].p.which_tree = t;
        count += tree->quadrants.elem_count;
      }
      /* p is of a union type p4est_quadrant_data, but only the p.which_tree field is active at this time. So, we
         have a simple blockTypes[] to use. Note that quadStruct does not count potential padding in array of
         p4est_quadrant_t. We have to call MPI_Type_create_resized() to change upper-bound of quadStruct.
       */
      PetscCallMPI(MPI_Type_create_struct(4,blockSizes,blockOffsets,blockTypes,&quadStruct));
      PetscCallMPI(MPI_Type_create_resized(quadStruct,0,sizeof(p4est_quadrant_t),&quadType));
      PetscCallMPI(MPI_Type_commit(&quadType));
      PetscCall(PetscSFBcastBegin(coveringSF,nodeClosureType,closurePointsC,newClosurePointsC,MPI_REPLACE));
      PetscCall(PetscSFBcastBegin(coveringSF,quadType,coverQuadsSend,coverQuads,MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(coveringSF,nodeClosureType,closurePointsC,newClosurePointsC,MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(coveringSF,quadType,coverQuadsSend,coverQuads,MPI_REPLACE));
      PetscCallMPI(MPI_Type_free(&quadStruct));
      PetscCallMPI(MPI_Type_free(&quadType));
      PetscCall(PetscFree(coverQuadsSend));
      PetscCall(PetscFree(closurePointsC));
      PetscCall(PetscSFDestroy(&coveringSF));
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

    PetscCall(PetscMalloc1(pEndF-pStartF,&roots));
    PetscCall(PetscMalloc1(pEndF-pStartF,&rootType));
    PetscCall(DMGetPointSF(fine,&pointSF));
    for (p = pStartF; p < pEndF; p++) {
      roots[p-pStartF].rank  = -1;
      roots[p-pStartF].index = -1;
      rootType[p-pStartF]    = -1;
    }
    if (formCids) {
      PetscInt child;

      PetscCall(PetscMalloc1(pEndF-pStartF,&cids));
      for (p = pStartF; p < pEndF; p++) cids[p - pStartF] = -2;
      PetscCall(DMPlexGetReferenceTree(plexF,&refTree));
      PetscCall(DMPlexGetTransitiveClosure(refTree,0,PETSC_TRUE,NULL,&rootClosure));
      for (child = 0; child < P4EST_CHILDREN; child++) { /* get the closures of the child cells in the reference tree */
        PetscCall(DMPlexGetTransitiveClosure(refTree,child+1,PETSC_TRUE,NULL,&childClosures[child]));
      }
      PetscCall(DMGetLabel(refTree,"canonical",&canonical));
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
        PetscCheck(disjoint == 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"did not find overlapping coarse quad");
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

            PetscCheck(levelDiff <= 1,PETSC_COMM_SELF,PETSC_ERR_USER,"Recursive child ids not implemented");
            PetscStackCallP4estReturn(cid,p4est_quadrant_child_id,(quad));
            PetscCall(DMPlexGetTransitiveClosure(plexF,c + cLocalStartF,PETSC_TRUE,NULL,&pointClosure));
            for (cl = 0; cl < P4EST_INSUL; cl++) {
              PetscInt p      = pointClosure[2 * cl];
              PetscInt point  = childClosures[cid][2 * cl];
              PetscInt ornt   = childClosures[cid][2 * cl + 1];
              PetscInt newcid = -1;
              DMPolytopeType ct;

              if (rootType[p-pStartF] == PETSC_MAX_INT) continue;
              PetscCall(DMPlexGetCellType(refTree, point, &ct));
              ornt = DMPolytopeConvertNewOrientation_Internal(ct, ornt);
              if (!cl) {
                newcid = cid + 1;
              } else {
                PetscInt rcl, parent, parentOrnt = 0;

                PetscCall(DMPlexGetTreeParent(refTree,point,&parent,NULL));
                if (parent == point) {
                  newcid = -1;
                } else if (!parent) { /* in the root */
                  newcid = point;
                } else {
                  DMPolytopeType rct = DM_POLYTOPE_UNKNOWN;

                  for (rcl = 1; rcl < P4EST_INSUL; rcl++) {
                    if (rootClosure[2 * rcl] == parent) {
                      PetscCall(DMPlexGetCellType(refTree, parent, &rct));
                      parentOrnt = DMPolytopeConvertNewOrientation_Internal(rct, rootClosure[2 * rcl + 1]);
                      break;
                    }
                  }
                  PetscCheck(rcl < P4EST_INSUL,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Couldn't find parent in root closure");
                  PetscCall(DMPlexReferenceTreeGetChildSymmetry(refTree,parent,parentOrnt,ornt,point,DMPolytopeConvertNewOrientation_Internal(rct, pointClosure[2 * rcl + 1]),NULL,&newcid));
                }
              }
              if (newcid >= 0) {

                if (canonical) {
                  PetscCall(DMLabelGetValue(canonical,newcid,&newcid));
                }
                proposedCids[cl] = newcid;
              }
            }
            PetscCall(DMPlexRestoreTransitiveClosure(plexF,c + cLocalStartF,PETSC_TRUE,NULL,&pointClosure));
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

                  PetscCall(DMPlexGetTreeParent(plexF,thisp,&parent,NULL));
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

      PetscCall(PetscMalloc1(pEndF-pStartF,&rootTypeCopy));
      PetscCall(PetscArraycpy(rootTypeCopy,rootType,pEndF-pStartF));
      PetscCall(PetscSFReduceBegin(pointSF,MPIU_INT,rootTypeCopy,rootTypeCopy,MPIU_MAX));
      PetscCall(PetscSFReduceEnd(pointSF,MPIU_INT,rootTypeCopy,rootTypeCopy,MPIU_MAX));
      PetscCall(PetscSFBcastBegin(pointSF,MPIU_INT,rootTypeCopy,rootTypeCopy,MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(pointSF,MPIU_INT,rootTypeCopy,rootTypeCopy,MPI_REPLACE));
      for (p = pStartF; p < pEndF; p++) {
        if (rootTypeCopy[p-pStartF] > rootType[p-pStartF]) { /* another process found a root of higher type (e.g. vertex instead of edge), which we want to accept, so nullify this */
          roots[p-pStartF].rank  = -1;
          roots[p-pStartF].index = -1;
        }
        if (formCids && rootTypeCopy[p-pStartF] == PETSC_MAX_INT) {
          cids[p-pStartF] = -1; /* we have found an antecedent that is the same: no child id */
        }
      }
      PetscCall(PetscFree(rootTypeCopy));
      PetscCall(PetscSFReduceBegin(pointSF,nodeType,roots,roots,sfNodeReduce));
      PetscCall(PetscSFReduceEnd(pointSF,nodeType,roots,roots,sfNodeReduce));
      PetscCall(PetscSFBcastBegin(pointSF,nodeType,roots,roots,MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(pointSF,nodeType,roots,roots,MPI_REPLACE));
    }
    PetscCall(PetscFree(rootType));

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
      PetscCall(PetscMalloc1(numLeaves,&leaves));
      PetscCall(PetscMalloc1(numLeaves,&iremote));
      numLeaves = 0;
      for (p = pStartF; p < pEndF; p++) {
        if (roots[p-pStartF].index >= 0) {
          leaves[numLeaves]  = p-pStartF;
          iremote[numLeaves] = roots[p-pStartF];
          numLeaves++;
        }
      }
      PetscCall(PetscFree(roots));
      PetscCall(PetscSFCreate(comm,sf));
      if (numLeaves == (pEndF-pStartF)) {
        PetscCall(PetscFree(leaves));
        PetscCall(PetscSFSetGraph(*sf,numRoots,numLeaves,NULL,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER));
      } else {
        PetscCall(PetscSFSetGraph(*sf,numRoots,numLeaves,leaves,PETSC_OWN_POINTER,iremote,PETSC_OWN_POINTER));
      }
    }
    if (formCids) {
      PetscSF  pointSF;
      PetscInt child;

      PetscCall(DMPlexGetReferenceTree(plexF,&refTree));
      PetscCall(DMGetPointSF(plexF,&pointSF));
      PetscCall(PetscSFReduceBegin(pointSF,MPIU_INT,cids,cids,MPIU_MAX));
      PetscCall(PetscSFReduceEnd(pointSF,MPIU_INT,cids,cids,MPIU_MAX));
      if (childIds) *childIds = cids;
      for (child = 0; child < P4EST_CHILDREN; child++) {
        PetscCall(DMPlexRestoreTransitiveClosure(refTree,child+1,PETSC_TRUE,NULL,&childClosures[child]));
      }
      PetscCall(DMPlexRestoreTransitiveClosure(refTree,0,PETSC_TRUE,NULL,&rootClosure));
    }
  }
  if (saveInCoarse) { /* cache results */
    PetscCall(PetscObjectReference((PetscObject)*sf));
    pforestC->pointSelfToAdaptSF = *sf;
    if (!childIds) {
      pforestC->pointSelfToAdaptCids = cids;
    } else {
      PetscCall(PetscMalloc1(pEndF-pStartF,&pforestC->pointSelfToAdaptCids));
      PetscCall(PetscArraycpy(pforestC->pointSelfToAdaptCids,cids,pEndF-pStartF));
    }
  } else if (saveInFine) {
    PetscCall(PetscObjectReference((PetscObject)*sf));
    pforestF->pointAdaptToSelfSF = *sf;
    if (!childIds) {
      pforestF->pointAdaptToSelfCids = cids;
    } else {
      PetscCall(PetscMalloc1(pEndF-pStartF,&pforestF->pointAdaptToSelfCids));
      PetscCall(PetscArraycpy(pforestF->pointAdaptToSelfCids,cids,pEndF-pStartF));
    }
  }
  PetscCall(PetscFree2(treeQuads,treeQuadCounts));
  PetscCall(PetscFree(coverQuads));
  PetscCall(PetscFree(closurePointsC));
  PetscCall(PetscFree(closurePointsF));
  PetscCallMPI(MPI_Type_free(&nodeClosureType));
  PetscCallMPI(MPI_Op_free(&sfNodeReduce));
  PetscCallMPI(MPI_Type_free(&nodeType));
  PetscFunctionReturn(0);
}

/* children are sf leaves of parents */
static PetscErrorCode DMPforestGetTransferSF_Internal(DM coarse, DM fine, const PetscInt dofPerDim[], PetscSF *sf, PetscBool transferIdent, PetscInt *childIds[])
{
  MPI_Comm          comm;
  PetscMPIInt       rank;
  DM_Forest_pforest *pforestC, *pforestF;
  DM                plexC, plexF;
  PetscInt          pStartC, pEndC, pStartF, pEndF;
  PetscSF           pointTransferSF;
  PetscBool         allOnes = PETSC_TRUE;

  PetscFunctionBegin;
  pforestC = (DM_Forest_pforest*) ((DM_Forest*) coarse->data)->data;
  pforestF = (DM_Forest_pforest*) ((DM_Forest*) fine->data)->data;
  PetscCheck(pforestC->topo == pforestF->topo,PetscObjectComm((PetscObject)coarse),PETSC_ERR_ARG_INCOMP,"DM's must have the same base DM");
  comm = PetscObjectComm((PetscObject)coarse);
  PetscCallMPI(MPI_Comm_rank(comm,&rank));

  {
    PetscInt i;
    for (i = 0; i <= P4EST_DIM; i++) {
      if (dofPerDim[i] != 1) {
        allOnes = PETSC_FALSE;
        break;
      }
    }
  }
  PetscCall(DMPforestGetTransferSF_Point(coarse,fine,&pointTransferSF,transferIdent,childIds));
  if (allOnes) {
    *sf = pointTransferSF;
    PetscFunctionReturn(0);
  }

  PetscCall(DMPforestGetPlex(fine,&plexF));
  PetscCall(DMPlexGetChart(plexF,&pStartF,&pEndF));
  PetscCall(DMPforestGetPlex(coarse,&plexC));
  PetscCall(DMPlexGetChart(plexC,&pStartC,&pEndC));
  {
    PetscInt          numRoots;
    PetscInt          numLeaves;
    const PetscInt    *leaves;
    const PetscSFNode *iremote;
    PetscInt          d;
    PetscSection      leafSection, rootSection;
    /* count leaves */

    PetscCall(PetscSFGetGraph(pointTransferSF,&numRoots,&numLeaves,&leaves,&iremote));
    PetscCall(PetscSectionCreate(PETSC_COMM_SELF,&rootSection));
    PetscCall(PetscSectionCreate(PETSC_COMM_SELF,&leafSection));
    PetscCall(PetscSectionSetChart(rootSection,pStartC,pEndC));
    PetscCall(PetscSectionSetChart(leafSection,pStartF,pEndF));

    for (d = 0; d <= P4EST_DIM; d++) {
      PetscInt startC, endC, e;

      PetscCall(DMPlexGetSimplexOrBoxCells(plexC,P4EST_DIM-d,&startC,&endC));
      for (e = startC; e < endC; e++) {
        PetscCall(PetscSectionSetDof(rootSection,e,dofPerDim[d]));
      }
    }

    for (d = 0; d <= P4EST_DIM; d++) {
      PetscInt startF, endF, e;

      PetscCall(DMPlexGetSimplexOrBoxCells(plexF,P4EST_DIM-d,&startF,&endF));
      for (e = startF; e < endF; e++) {
        PetscCall(PetscSectionSetDof(leafSection,e,dofPerDim[d]));
      }
    }

    PetscCall(PetscSectionSetUp(rootSection));
    PetscCall(PetscSectionSetUp(leafSection));
    {
      PetscInt    nroots, nleaves;
      PetscInt    *mine, i, p;
      PetscInt    *offsets, *offsetsRoot;
      PetscSFNode *remote;

      PetscCall(PetscMalloc1(pEndF-pStartF,&offsets));
      PetscCall(PetscMalloc1(pEndC-pStartC,&offsetsRoot));
      for (p = pStartC; p < pEndC; p++) {
        PetscCall(PetscSectionGetOffset(rootSection,p,&offsetsRoot[p-pStartC]));
      }
      PetscCall(PetscSFBcastBegin(pointTransferSF,MPIU_INT,offsetsRoot,offsets,MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(pointTransferSF,MPIU_INT,offsetsRoot,offsets,MPI_REPLACE));
      PetscCall(PetscSectionGetStorageSize(rootSection,&nroots));
      nleaves = 0;
      for (i = 0; i < numLeaves; i++) {
        PetscInt leaf = leaves ? leaves[i] : i;
        PetscInt dof;

        PetscCall(PetscSectionGetDof(leafSection,leaf,&dof));
        nleaves += dof;
      }
      PetscCall(PetscMalloc1(nleaves,&mine));
      PetscCall(PetscMalloc1(nleaves,&remote));
      nleaves = 0;
      for (i = 0; i < numLeaves; i++) {
        PetscInt leaf = leaves ? leaves[i] : i;
        PetscInt dof;
        PetscInt off, j;

        PetscCall(PetscSectionGetDof(leafSection,leaf,&dof));
        PetscCall(PetscSectionGetOffset(leafSection,leaf,&off));
        for (j = 0; j < dof; j++) {
          remote[nleaves].rank  = iremote[i].rank;
          remote[nleaves].index = offsets[leaf] + j;
          mine[nleaves++]       = off + j;
        }
      }
      PetscCall(PetscFree(offsetsRoot));
      PetscCall(PetscFree(offsets));
      PetscCall(PetscSFCreate(comm,sf));
      PetscCall(PetscSFSetGraph(*sf,nroots,nleaves,mine,PETSC_OWN_POINTER,remote,PETSC_OWN_POINTER));
    }
    PetscCall(PetscSectionDestroy(&leafSection));
    PetscCall(PetscSectionDestroy(&rootSection));
    PetscCall(PetscSFDestroy(&pointTransferSF));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPforestGetTransferSF(DM dmA, DM dmB, const PetscInt dofPerDim[], PetscSF *sfAtoB, PetscSF *sfBtoA)
{
  DM             adaptA, adaptB;
  DMAdaptFlag    purpose;

  PetscFunctionBegin;
  PetscCall(DMForestGetAdaptivityForest(dmA,&adaptA));
  PetscCall(DMForestGetAdaptivityForest(dmB,&adaptB));
  /* it is more efficient when the coarser mesh is the first argument: reorder if we know one is coarser than the other */
  if (adaptA && adaptA->data == dmB->data) { /* dmA was adapted from dmB */
    PetscCall(DMForestGetAdaptivityPurpose(dmA,&purpose));
    if (purpose == DM_ADAPT_REFINE) {
      PetscCall(DMPforestGetTransferSF(dmB, dmA, dofPerDim, sfBtoA, sfAtoB));
      PetscFunctionReturn(0);
    }
  } else if (adaptB && adaptB->data == dmA->data) { /* dmB was adapted from dmA */
    PetscCall(DMForestGetAdaptivityPurpose(dmB,&purpose));
    if (purpose == DM_ADAPT_COARSEN) {
      PetscCall(DMPforestGetTransferSF(dmB, dmA, dofPerDim, sfBtoA, sfAtoB));
      PetscFunctionReturn(0);
    }
  }
  if (sfAtoB) PetscCall(DMPforestGetTransferSF_Internal(dmA,dmB,dofPerDim,sfAtoB,PETSC_TRUE,NULL));
  if (sfBtoA) PetscCall(DMPforestGetTransferSF_Internal(dmB,dmA,dofPerDim,sfBtoA,(PetscBool) (sfAtoB == NULL),NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPforestLabelsInitialize(DM dm, DM plex)
{
  DM_Forest         *forest  = (DM_Forest*) dm->data;
  DM_Forest_pforest *pforest = (DM_Forest_pforest*) forest->data;
  PetscInt          cLocalStart, cLocalEnd, cStart, cEnd, fStart, fEnd, eStart, eEnd, vStart, vEnd;
  PetscInt          cStartBase, cEndBase, fStartBase, fEndBase, vStartBase, vEndBase, eStartBase, eEndBase;
  PetscInt          pStart, pEnd, pStartBase, pEndBase, p;
  DM                base;
  PetscInt          *star     = NULL, starSize;
  DMLabelLink       next      = dm->labels;
  PetscInt          guess     = 0;
  p4est_topidx_t    num_trees = pforest->topo->conn->num_trees;

  PetscFunctionBegin;
  pforest->labelsFinalized = PETSC_TRUE;
  cLocalStart              = pforest->cLocalStart;
  cLocalEnd                = pforest->cLocalEnd;
  PetscCall(DMForestGetBaseDM(dm,&base));
  if (!base) {
    if (pforest->ghostName) { /* insert a label to make the boundaries, with stratum values denoting which face of the element touches the boundary */
      p4est_connectivity_t *conn  = pforest->topo->conn;
      p4est_t              *p4est = pforest->forest;
      p4est_tree_t         *trees = (p4est_tree_t*) p4est->trees->array;
      p4est_topidx_t       t, flt = p4est->first_local_tree;
      p4est_topidx_t       llt = pforest->forest->last_local_tree;
      DMLabel              ghostLabel;
      PetscInt             c;

      PetscCall(DMCreateLabel(plex,pforest->ghostName));
      PetscCall(DMGetLabel(plex,pforest->ghostName,&ghostLabel));
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

                PetscCall(DMPlexGetCone(plex,c,&cone));
                PetscCall(DMLabelSetValue(ghostLabel,cone[plexF],plexF+1));
              }
            }
          }
        }
      }
    }
    PetscFunctionReturn(0);
  }
  PetscCall(DMPlexGetSimplexOrBoxCells(base,0,&cStartBase,&cEndBase));
  PetscCall(DMPlexGetSimplexOrBoxCells(base,1,&fStartBase,&fEndBase));
  PetscCall(DMPlexGetSimplexOrBoxCells(base,P4EST_DIM-1,&eStartBase,&eEndBase));
  PetscCall(DMPlexGetDepthStratum(base,0,&vStartBase,&vEndBase));

  PetscCall(DMPlexGetSimplexOrBoxCells(plex,0,&cStart,&cEnd));
  PetscCall(DMPlexGetSimplexOrBoxCells(plex,1,&fStart,&fEnd));
  PetscCall(DMPlexGetSimplexOrBoxCells(plex,P4EST_DIM-1,&eStart,&eEnd));
  PetscCall(DMPlexGetDepthStratum(plex,0,&vStart,&vEnd));

  PetscCall(DMPlexGetChart(plex,&pStart,&pEnd));
  PetscCall(DMPlexGetChart(base,&pStartBase,&pEndBase));
  /* go through the mesh: use star to find a quadrant that borders a point.  Use the closure to determine the
   * orientation of the quadrant relative to that point.  Use that to relate the point to the numbering in the base
   * mesh, and extract a label value (since the base mesh is redundantly distributed, can be found locally). */
  while (next) {
    DMLabel   baseLabel;
    DMLabel   label = next->label;
    PetscBool isDepth, isCellType, isGhost, isVTK, isSpmap;
    const char *name;

    PetscCall(PetscObjectGetName((PetscObject) label, &name));
    PetscCall(PetscStrcmp(name,"depth",&isDepth));
    if (isDepth) {
      next = next->next;
      continue;
    }
    PetscCall(PetscStrcmp(name,"celltype",&isCellType));
    if (isCellType) {
      next = next->next;
      continue;
    }
    PetscCall(PetscStrcmp(name,"ghost",&isGhost));
    if (isGhost) {
      next = next->next;
      continue;
    }
    PetscCall(PetscStrcmp(name,"vtk",&isVTK));
    if (isVTK) {
      next = next->next;
      continue;
    }
    PetscCall(PetscStrcmp(name,"_forest_base_subpoint_map",&isSpmap));
    if (!isSpmap) {
      PetscCall(DMGetLabel(base,name,&baseLabel));
      if (!baseLabel) {
        next = next->next;
        continue;
      }
      PetscCall(DMLabelCreateIndex(baseLabel,pStartBase,pEndBase));
    } else baseLabel = NULL;

    for (p = pStart; p < pEnd; p++) {
      PetscInt         s, c = -1, l;
      PetscInt         *closure = NULL, closureSize;
      p4est_quadrant_t * ghosts = (p4est_quadrant_t*) pforest->ghost->ghosts.array;
      p4est_tree_t     *trees   = (p4est_tree_t*) pforest->forest->trees->array;
      p4est_quadrant_t * q;
      PetscInt         t, val;
      PetscBool        zerosupportpoint = PETSC_FALSE;

      PetscCall(DMPlexGetTransitiveClosure(plex,p,PETSC_FALSE,&starSize,&star));
      for (s = 0; s < starSize; s++) {
        PetscInt point = star[2*s];

        if (cStart <= point && point < cEnd) {
          PetscCall(DMPlexGetTransitiveClosure(plex,point,PETSC_TRUE,&closureSize,&closure));
          for (l = 0; l < closureSize; l++) {
            PetscInt qParent = closure[2 * l], q, pp = p, pParent = p;
            do { /* check parents of q */
              q = qParent;
              if (q == p) {
                c = point;
                break;
              }
              PetscCall(DMPlexGetTreeParent(plex,q,&qParent,NULL));
            } while (qParent != q);
            if (c != -1) break;
            PetscCall(DMPlexGetTreeParent(plex,pp,&pParent,NULL));
            q = closure[2 * l];
            while (pParent != pp) { /* check parents of p */
              pp = pParent;
              if (pp == q) {
                c = point;
                break;
              }
              PetscCall(DMPlexGetTreeParent(plex,pp,&pParent,NULL));
            }
            if (c != -1) break;
          }
          PetscCall(DMPlexRestoreTransitiveClosure(plex,point,PETSC_TRUE,NULL,&closure));
          if (l < closureSize) break;
        } else {
          PetscInt supportSize;

          PetscCall(DMPlexGetSupportSize(plex,point,&supportSize));
          zerosupportpoint = (PetscBool) (zerosupportpoint || !supportSize);
        }
      }
      if (c < 0) {
        const char* prefix;
        PetscBool   print = PETSC_FALSE;

        PetscCall(PetscObjectGetOptionsPrefix((PetscObject)dm,&prefix));
        PetscCall(PetscOptionsGetBool(((PetscObject)dm)->options,prefix,"-dm_forest_print_label_error",&print,NULL));
        if (print) {
          PetscInt i;

          PetscCall(PetscPrintf(PETSC_COMM_SELF,"[%d] Failed to find cell with point %" PetscInt_FMT " in its closure for label %s (starSize %" PetscInt_FMT ")\n",PetscGlobalRank,p,baseLabel ? ((PetscObject)baseLabel)->name : "_forest_base_subpoint_map",starSize));
          for (i = 0; i < starSize; i++) PetscCall(PetscPrintf(PETSC_COMM_SELF,"  star[%" PetscInt_FMT "] = %" PetscInt_FMT ",%" PetscInt_FMT "\n",i,star[2*i],star[2*i+1]));
        }
        PetscCall(DMPlexRestoreTransitiveClosure(plex,p,PETSC_FALSE,NULL,&star));
        if (zerosupportpoint) continue;
        else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Failed to find cell with point %" PetscInt_FMT " in its closure for label %s. Rerun with -dm_forest_print_label_error for more information",p,baseLabel ? ((PetscObject) baseLabel)->name : "_forest_base_subpoint_map");
      }
      PetscCall(DMPlexRestoreTransitiveClosure(plex,p,PETSC_FALSE,NULL,&star));

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

          PetscCheck(guess >= lo && guess < num_trees && lo < hi,PETSC_COMM_SELF,PETSC_ERR_PLIB,"failed binary search");
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
          PetscCall(DMLabelGetValue(baseLabel,t+cStartBase,&val));
        } else {
          val  = t+cStartBase;
        }
        PetscCall(DMLabelSetValue(label,p,val));
      } else if (l >= 1 && l < 1 + P4EST_FACES) { /* facet */
        p4est_quadrant_t nq;
        int              isInside;

        l = PetscFaceToP4estFace[l - 1];
        PetscStackCallP4est(p4est_quadrant_face_neighbor,(q,l,&nq));
        PetscStackCallP4estReturn(isInside,p4est_quadrant_is_inside_root,(&nq));
        if (isInside) {
          /* this facet is in the interior of a tree, so it inherits the label of the tree */
          if (baseLabel) {
            PetscCall(DMLabelGetValue(baseLabel,t+cStartBase,&val));
          } else {
            val  = t+cStartBase;
          }
          PetscCall(DMLabelSetValue(label,p,val));
        } else {
          PetscInt f = pforest->topo->tree_face_to_uniq[P4EST_FACES * t + l];

          if (baseLabel) {
            PetscCall(DMLabelGetValue(baseLabel,f+fStartBase,&val));
          } else {
            val  = f+fStartBase;
          }
          PetscCall(DMLabelSetValue(label,p,val));
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
            PetscCall(DMLabelGetValue(baseLabel,t+cStartBase,&val));
          } else {
            val  = t+cStartBase;
          }
          PetscCall(DMLabelSetValue(label,p,val));
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
              PetscCall(DMLabelGetValue(baseLabel,f+fStartBase,&val));
            } else {
              val  = f+fStartBase;
            }
            PetscCall(DMLabelSetValue(label,p,val));
          } else { /* the quadrant edge corresponds to the tree edge */
            PetscInt e = pforest->topo->conn->tree_to_edge[P8EST_EDGES * t + l];

            if (baseLabel) {
              PetscCall(DMLabelGetValue(baseLabel,e+eStartBase,&val));
            } else {
              val  = e+eStartBase;
            }
            PetscCall(DMLabelSetValue(label,p,val));
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
            PetscCall(DMLabelGetValue(baseLabel,t+cStartBase,&val));
          } else {
            val  = t+cStartBase;
          }
          PetscCall(DMLabelSetValue(label,p,val));
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
              PetscCall(DMLabelGetValue(baseLabel,f+fStartBase,&val));
            } else {
              val  = f+fStartBase;
            }
            PetscCall(DMLabelSetValue(label,p,val));
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
              PetscCall(DMLabelGetValue(baseLabel,e+eStartBase,&val));
            } else {
              val  = e+eStartBase;
            }
            PetscCall(DMLabelSetValue(label,p,val));
            continue;
          }
#endif
          {
            /* outside vertex: same corner as quadrant corner */
            PetscInt v = pforest->topo->conn->tree_to_corner[P4EST_CHILDREN * t + l];

            if (baseLabel) {
              PetscCall(DMLabelGetValue(baseLabel,v+vStartBase,&val));
            } else {
              val  = v+vStartBase;
            }
            PetscCall(DMLabelSetValue(label,p,val));
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

  PetscFunctionBegin;
  if (pforest->labelsFinalized) PetscFunctionReturn(0);
  pforest->labelsFinalized = PETSC_TRUE;
  PetscCall(DMForestGetAdaptivityForest(dm,&adapt));
  if (!adapt) {
    /* Initialize labels from the base dm */
    PetscCall(DMPforestLabelsInitialize(dm,plex));
  } else {
    PetscInt    dofPerDim[4]={1, 1, 1, 1};
    PetscSF     transferForward, transferBackward, pointSF;
    PetscInt    pStart, pEnd, pStartA, pEndA;
    PetscInt    *values, *adaptValues;
    DMLabelLink next = adapt->labels;
    DMLabel     adaptLabel;
    DM          adaptPlex;

    PetscCall(DMForestGetAdaptivityLabel(dm,&adaptLabel));
    PetscCall(DMPforestGetPlex(adapt,&adaptPlex));
    PetscCall(DMPforestGetTransferSF(adapt,dm,dofPerDim,&transferForward,&transferBackward));
    PetscCall(DMPlexGetChart(plex,&pStart,&pEnd));
    PetscCall(DMPlexGetChart(adaptPlex,&pStartA,&pEndA));
    PetscCall(PetscMalloc2(pEnd-pStart,&values,pEndA-pStartA,&adaptValues));
    PetscCall(DMGetPointSF(plex,&pointSF));
    if (PetscDefined(USE_DEBUG)) {
      PetscInt p;
      for (p = pStartA; p < pEndA; p++) adaptValues[p-pStartA] = -1;
      for (p = pStart; p < pEnd; p++)   values[p-pStart]       = -2;
      if (transferForward) {
        PetscCall(PetscSFBcastBegin(transferForward,MPIU_INT,adaptValues,values,MPI_REPLACE));
        PetscCall(PetscSFBcastEnd(transferForward,MPIU_INT,adaptValues,values,MPI_REPLACE));
      }
      if (transferBackward) {
        PetscCall(PetscSFReduceBegin(transferBackward,MPIU_INT,adaptValues,values,MPIU_MAX));
        PetscCall(PetscSFReduceEnd(transferBackward,MPIU_INT,adaptValues,values,MPIU_MAX));
      }
      for (p = pStart; p < pEnd; p++) {
        PetscInt q = p, parent;

        PetscCall(DMPlexGetTreeParent(plex,q,&parent,NULL));
        while (parent != q) {
          if (values[parent] == -2) values[parent] = values[q];
          q    = parent;
          PetscCall(DMPlexGetTreeParent(plex,q,&parent,NULL));
        }
      }
      PetscCall(PetscSFReduceBegin(pointSF,MPIU_INT,values,values,MPIU_MAX));
      PetscCall(PetscSFReduceEnd(pointSF,MPIU_INT,values,values,MPIU_MAX));
      PetscCall(PetscSFBcastBegin(pointSF,MPIU_INT,values,values,MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(pointSF,MPIU_INT,values,values,MPI_REPLACE));
      for (p = pStart; p < pEnd; p++) {
        PetscCheck(values[p-pStart] != -2,PETSC_COMM_SELF,PETSC_ERR_PLIB,"uncovered point %" PetscInt_FMT,p);
      }
    }
    while (next) {
      DMLabel    nextLabel = next->label;
      const char *name;
      PetscBool  isDepth, isCellType, isGhost, isVTK;
      DMLabel    label;
      PetscInt   p;

      PetscCall(PetscObjectGetName((PetscObject) nextLabel, &name));
      PetscCall(PetscStrcmp(name,"depth",&isDepth));
      if (isDepth) {
        next = next->next;
        continue;
      }
      PetscCall(PetscStrcmp(name,"celltype",&isCellType));
      if (isCellType) {
        next = next->next;
        continue;
      }
      PetscCall(PetscStrcmp(name,"ghost",&isGhost));
      if (isGhost) {
        next = next->next;
        continue;
      }
      PetscCall(PetscStrcmp(name,"vtk",&isVTK));
      if (isVTK) {
        next = next->next;
        continue;
      }
      if (nextLabel == adaptLabel) {
        next = next->next;
        continue;
      }
      /* label was created earlier */
      PetscCall(DMGetLabel(dm,name,&label));
      for (p = pStartA; p < pEndA; p++) {
        PetscCall(DMLabelGetValue(nextLabel,p,&adaptValues[p]));
      }
      for (p = pStart; p < pEnd; p++) values[p] = PETSC_MIN_INT;

      if (transferForward) PetscCall(PetscSFBcastBegin(transferForward,MPIU_INT,adaptValues,values,MPI_REPLACE));
      if (transferBackward) PetscCall(PetscSFReduceBegin(transferBackward,MPIU_INT,adaptValues,values,MPIU_MAX));
      if (transferForward) PetscCall(PetscSFBcastEnd(transferForward,MPIU_INT,adaptValues,values,MPI_REPLACE));
      if (transferBackward) PetscCall(PetscSFReduceEnd(transferBackward,MPIU_INT,adaptValues,values,MPIU_MAX));
      for (p = pStart; p < pEnd; p++) {
        PetscInt q = p, parent;

        PetscCall(DMPlexGetTreeParent(plex,q,&parent,NULL));
        while (parent != q) {
          if (values[parent] == PETSC_MIN_INT) values[parent] = values[q];
          q    = parent;
          PetscCall(DMPlexGetTreeParent(plex,q,&parent,NULL));
        }
      }
      PetscCall(PetscSFReduceBegin(pointSF,MPIU_INT,values,values,MPIU_MAX));
      PetscCall(PetscSFReduceEnd(pointSF,MPIU_INT,values,values,MPIU_MAX));
      PetscCall(PetscSFBcastBegin(pointSF,MPIU_INT,values,values,MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(pointSF,MPIU_INT,values,values,MPI_REPLACE));

      for (p = pStart; p < pEnd; p++) {
        PetscCall(DMLabelSetValue(label,p,values[p]));
      }
      next = next->next;
    }
    PetscCall(PetscFree2(values,adaptValues));
    PetscCall(PetscSFDestroy(&transferForward));
    PetscCall(PetscSFDestroy(&transferBackward));
    pforest->labelsFinalized = PETSC_TRUE;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPforestMapCoordinates_Cell(DM plex, p4est_geometry_t *geom, PetscInt cell, p4est_quadrant_t *q, p4est_topidx_t t, p4est_connectivity_t * conn, PetscScalar *coords)
{
  PetscInt       closureSize, c, coordStart, coordEnd, coordDim;
  PetscInt       *closure = NULL;
  PetscSection   coordSec;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateSection(plex,&coordSec));
  PetscCall(PetscSectionGetChart(coordSec,&coordStart,&coordEnd));
  PetscCall(DMGetCoordinateDim(plex,&coordDim));
  PetscCall(DMPlexGetTransitiveClosure(plex,cell,PETSC_TRUE,&closureSize,&closure));
  for (c = 0; c < closureSize; c++) {
    PetscInt point = closure[2 * c];

    if (point >= coordStart && point < coordEnd) {
      PetscInt dof, off;
      PetscInt nCoords, i;
      PetscCall(PetscSectionGetDof(coordSec,point,&dof));
      PetscCheck(dof % coordDim == 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Did not understand coordinate layout");
      nCoords = dof / coordDim;
      PetscCall(PetscSectionGetOffset(coordSec,point,&off));
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
  PetscCall(DMPlexRestoreTransitiveClosure(plex,cell,PETSC_TRUE,&closureSize,&closure));
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

  PetscFunctionBegin;
  forest  = (DM_Forest*) dm->data;
  pforest = (DM_Forest_pforest*) forest->data;
  geom    = pforest->topo->geom;
  PetscCall(DMForestGetBaseCoordinateMapping(dm,&map,&mapCtx));
  if (!geom && !map) PetscFunctionReturn(0);
  PetscCall(DMGetCoordinatesLocal(plex,&coordLocalVec));
  PetscCall(VecGetArray(coordLocalVec,&coords));
  cLocalStart = pforest->cLocalStart;
  cLocalEnd   = pforest->cLocalEnd;
  flt         = pforest->forest->first_local_tree;
  llt         = pforest->forest->last_local_tree;
  trees       = (p4est_tree_t*) pforest->forest->trees->array;
  if (map) { /* apply the map directly to the existing coordinates */
    PetscSection coordSec;
    PetscInt     coordStart, coordEnd, p, coordDim, p4estCoordDim, cStart, cEnd, cEndInterior;
    DM           base;

    PetscCall(DMPlexGetHeightStratum(plex,0,&cStart,&cEnd));
    PetscCall(DMPlexGetGhostCellStratum(plex,&cEndInterior,NULL));
    cEnd          = cEndInterior < 0 ? cEnd : cEndInterior;
    PetscCall(DMForestGetBaseDM(dm,&base));
    PetscCall(DMGetCoordinateSection(plex,&coordSec));
    PetscCall(PetscSectionGetChart(coordSec,&coordStart,&coordEnd));
    PetscCall(DMGetCoordinateDim(plex,&coordDim));
    p4estCoordDim = PetscMin(coordDim,3);
    for (p = coordStart; p < coordEnd; p++) {
      PetscInt *star = NULL, starSize;
      PetscInt dof, off, cell = -1, coarsePoint = -1;
      PetscInt nCoords, i;
      PetscCall(PetscSectionGetDof(coordSec,p,&dof));
      PetscCheck(dof % coordDim == 0,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Did not understand coordinate layout");
      nCoords = dof / coordDim;
      PetscCall(PetscSectionGetOffset(coordSec,p,&off));
      PetscCall(DMPlexGetTransitiveClosure(plex,p,PETSC_FALSE,&starSize,&star));
      for (i = 0; i < starSize; i++) {
        PetscInt point = star[2 * i];

        if (cStart <= point && point < cEnd) {
          cell = point;
          break;
        }
      }
      PetscCall(DMPlexRestoreTransitiveClosure(plex,p,PETSC_FALSE,&starSize,&star));
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
        PetscCall((map)(base,coarsePoint,p4estCoordDim,coordP4est,coordP4estMapped,mapCtx));
        for (j = 0; j < p4estCoordDim; j++) coord[j] = (PetscScalar) coordP4estMapped[j];
      }
    }
  } else { /* we have to transform coordinates back to the unit cube (where geom is defined), and then apply geom */
    PetscInt cStart, cEnd, cEndInterior;

    PetscCall(DMPlexGetHeightStratum(plex,0,&cStart,&cEnd));
    PetscCall(DMPlexGetGhostCellStratum(plex,&cEndInterior,NULL));
    cEnd = cEndInterior < 0 ? cEnd : cEndInterior;
    if (cLocalStart > 0) {
      p4est_quadrant_t *ghosts = (p4est_quadrant_t*) pforest->ghost->ghosts.array;
      PetscInt         count;

      for (count = 0; count < cLocalStart; count++) {
        p4est_quadrant_t *quad = &ghosts[count];
        p4est_topidx_t   t     = quad->p.which_tree;

        PetscCall(DMPforestMapCoordinates_Cell(plex,geom,count,quad,t,pforest->topo->conn,coords));
      }
    }
    for (t = flt; t <= llt; t++) {
      p4est_tree_t     *tree    = &(trees[t]);
      PetscInt         offset   = cLocalStart + tree->quadrants_offset, i;
      PetscInt         numQuads = (PetscInt) tree->quadrants.elem_count;
      p4est_quadrant_t *quads   = (p4est_quadrant_t*) tree->quadrants.array;

      for (i = 0; i < numQuads; i++) {
        PetscInt count = i + offset;

        PetscCall(DMPforestMapCoordinates_Cell(plex,geom,count,&quads[i],t,pforest->topo->conn,coords));
      }
    }
    if (cLocalEnd - cLocalStart < cEnd - cStart) {
      p4est_quadrant_t *ghosts   = (p4est_quadrant_t*) pforest->ghost->ghosts.array;
      PetscInt         numGhosts = (PetscInt) pforest->ghost->ghosts.elem_count;
      PetscInt         count;

      for (count = 0; count < numGhosts - cLocalStart; count++) {
        p4est_quadrant_t *quad = &ghosts[count + cLocalStart];
        p4est_topidx_t   t     = quad->p.which_tree;

        PetscCall(DMPforestMapCoordinates_Cell(plex,geom,count + cLocalEnd,quad,t,pforest->topo->conn,coords));
      }
    }
  }
  PetscCall(VecRestoreArray(coordLocalVec,&coords));
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

  PetscFunctionBegin;
  PetscCall(DMGetPeriodicity(dm,&isper,NULL,NULL,NULL));
  if (!isper) PetscFunctionReturn(0);
  /* we localize on all cells if we don't have a base DM or the base DM coordinates have not been localized */
  PetscCall(DMGetCoordinateDim(dm, &cDim));
  cdof = P4EST_CHILDREN*cDim;
  PetscCall(DMForestGetBaseDM(dm,&base));
  if (base) {
    PetscCall(DMGetCoordinatesLocalized(base,&baseLocalized));
  }
  if (!baseLocalized) base = NULL;
  PetscCall(DMPlexGetChart(plex, &newStart, &newEnd));

  PetscCall(DMForestGetPartitionOverlap(dm,&overlap));
  PetscCall(PetscCalloc1(overlap ? newEnd - newStart : 0,&localize));

  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject) dm), &newSection));
  PetscCall(PetscSectionSetNumFields(newSection, 1));
  PetscCall(PetscSectionSetFieldComponents(newSection, 0, cDim));
  PetscCall(PetscSectionSetChart(newSection, newStart, newEnd));

  PetscCall(DMGetCoordinateSection(plex, &oldSection));
  if (base) PetscCall(DMGetCoordinateSection(base, &baseSection));
  PetscCall(DMPlexGetDepthStratum(plex,0,&vStart,&vEnd));
  for (v = vStart; v < vEnd; ++v) {
    PetscCall(PetscSectionGetDof(oldSection, v, &dof));
    PetscCall(PetscSectionSetDof(newSection, v, dof));
    PetscCall(PetscSectionSetFieldDof(newSection, v, 0, dof));
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
  PetscCall(DMPlexGetHeightStratum(plex,0,&cStart,&cEnd));
  PetscCall(DMPlexGetGhostCellStratum(plex,&cEndInterior,NULL));
  cEnd = cEndInterior < 0 ? cEnd : cEndInterior;
  PetscCall(PetscMalloc1(cEnd-cStart,&coarsePoints));
  if (cLocalStart > 0) {
    p4est_quadrant_t *ghosts = (p4est_quadrant_t*) pforest->ghost->ghosts.array;
    PetscInt         count;

    for (count = 0; count < cLocalStart; count++) {
      p4est_quadrant_t *quad = &ghosts[count];
      coarsePoint = quad->p.which_tree;

      if (baseSection) PetscCall(PetscSectionGetFieldDof(baseSection, coarsePoint, 0, &cdof));
      PetscCall(PetscSectionSetDof(newSection, count, cdof));
      PetscCall(PetscSectionSetFieldDof(newSection, count, 0, cdof));
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
    if (baseSection) PetscCall(PetscSectionGetFieldDof(baseSection, coarsePoint, 0, &cdof));
    for (i = 0; i < numQuads; i++) {
      PetscInt newCell = i + offset;

      PetscCall(PetscSectionSetDof(newSection, newCell, cdof));
      PetscCall(PetscSectionSetFieldDof(newSection, newCell, 0, cdof));
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

      if (baseSection) PetscCall(PetscSectionGetFieldDof(baseSection, coarsePoint, 0, &cdof));
      PetscCall(PetscSectionSetDof(newSection, newCell, cdof));
      PetscCall(PetscSectionSetFieldDof(newSection, newCell, 0, cdof));
      coarsePoints[cp++] = cdof ? coarsePoint : -1;
      if (overlap) localize[newCell] = cdof;
    }
  }
  PetscCheck(cp == cEnd - cStart,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected number of fine cells %" PetscInt_FMT " != %" PetscInt_FMT,cp,cEnd-cStart);

  if (base) { /* we need to localize on all the cells in the star of the coarse cell vertices */
    PetscInt *closure = NULL, closureSize;
    PetscInt p, i, c, vStartBase, vEndBase, cStartBase, cEndBase;

    PetscCall(DMPlexGetHeightStratum(base,0,&cStartBase,&cEndBase));
    PetscCall(DMPlexGetDepthStratum(base,0,&vStartBase,&vEndBase));
    for (p = cStart; p < cEnd; p++) {
      coarsePoint = coarsePoints[p-cStart];
      if (coarsePoint < 0) continue;
      if (baseSection) PetscCall(PetscSectionGetFieldDof(baseSection, coarsePoint, 0, &cdof));
      PetscCall(DMPlexGetTransitiveClosure(base,coarsePoint,PETSC_TRUE,&closureSize,&closure));
      for (c = 0; c < closureSize; c++) {
        PetscInt *star = NULL, starSize;
        PetscInt j, v = closure[2 * c];

        if (v < vStartBase || v > vEndBase) continue;
        PetscCall(DMPlexGetTransitiveClosure(base,v,PETSC_FALSE,&starSize,&star));
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

              PetscCall(PetscSectionSetDof(newSection, newCell, cdof));
              PetscCall(PetscSectionSetFieldDof(newSection, newCell, 0, cdof));
              if (overlap) localize[newCell] = cdof;
            }
          }
        }
        PetscCall(DMPlexRestoreTransitiveClosure(base,v,PETSC_FALSE,&starSize,&star));
      }
      PetscCall(DMPlexRestoreTransitiveClosure(base,coarsePoint,PETSC_TRUE,&closureSize,&closure));
    }
  }
  PetscCall(PetscFree(coarsePoints));

  /* final consensus with overlap */
  if (overlap) {
    PetscSF  sf;
    PetscInt *localizeGlobal;

    PetscCall(DMGetPointSF(plex,&sf));
    PetscCall(PetscMalloc1(newEnd-newStart,&localizeGlobal));
    for (v = newStart; v < newEnd; v++) localizeGlobal[v - newStart] = localize[v - newStart];
    PetscCall(PetscSFBcastBegin(sf,MPIU_INT,localize,localizeGlobal,MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf,MPIU_INT,localize,localizeGlobal,MPI_REPLACE));
    for (v = newStart; v < newEnd; v++) {
      PetscCall(PetscSectionSetDof(newSection, v, localizeGlobal[v-newStart]));
      PetscCall(PetscSectionSetFieldDof(newSection, v, 0, localizeGlobal[v-newStart]));
    }
    PetscCall(PetscFree(localizeGlobal));
  }
  PetscCall(PetscFree(localize));
  PetscCall(PetscSectionSetUp(newSection));
  PetscCall(PetscObjectReference((PetscObject)oldSection));
  PetscCall(DMSetCoordinateSection(plex, cDim, newSection));
  PetscCall(PetscSectionGetStorageSize(newSection, &v));
  PetscCall(VecCreate(PETSC_COMM_SELF, &cVec));
  PetscCall(PetscObjectSetName((PetscObject)cVec,"coordinates"));
  PetscCall(VecSetBlockSize(cVec, cDim));
  PetscCall(VecSetSizes(cVec, v, PETSC_DETERMINE));
  PetscCall(VecSetType(cVec, VECSTANDARD));
  PetscCall(VecSet(cVec, PETSC_MIN_REAL));

  /* Copy over vertex coordinates */
  PetscCall(DMGetCoordinatesLocal(plex, &coordinates));
  PetscCheck(coordinates,PetscObjectComm((PetscObject)plex),PETSC_ERR_SUP,"Missing local coordinates vector");
  PetscCall(VecGetArray(cVec, &coords2));
  PetscCall(VecGetArrayRead(coordinates, &coords));
  for (v = vStart; v < vEnd; ++v) {
    PetscInt d, off,off2;

    PetscCall(PetscSectionGetDof(oldSection, v, &dof));
    PetscCall(PetscSectionGetOffset(oldSection, v, &off));
    PetscCall(PetscSectionGetOffset(newSection, v, &off2));
    for (d = 0; d < dof; ++d) coords2[off2+d] = coords[off+d];
  }
  PetscCall(VecRestoreArrayRead(coordinates, &coords));

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

      PetscCall(PetscSectionGetFieldDof(newSection, newCell, 0, &cdof));
      if (!cdof) continue;

      h2   = .5 * intsize * P4EST_QUADRANT_LEN (quad->level);
      k    = 0;
      PetscCall(PetscSectionGetOffset(newSection, newCell, &off2));
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
  PetscCall(VecRestoreArray(cVec, &coords2));
  PetscCall(DMSetCoordinatesLocal(plex, cVec));
  PetscCall(VecDestroy(&cVec));
  PetscCall(PetscSectionDestroy(&newSection));
  PetscCall(PetscSectionDestroy(&oldSection));
  PetscFunctionReturn(0);
}

#define DMForestClearAdaptivityForest_pforest _append_pforest(DMForestClearAdaptivityForest)
static PetscErrorCode DMForestClearAdaptivityForest_pforest(DM dm)
{
  DM_Forest         *forest;
  DM_Forest_pforest *pforest;

  PetscFunctionBegin;
  forest  = (DM_Forest*) dm->data;
  pforest = (DM_Forest_pforest *) forest->data;
  PetscCall(PetscSFDestroy(&(pforest->pointAdaptToSelfSF)));
  PetscCall(PetscSFDestroy(&(pforest->pointSelfToAdaptSF)));
  PetscCall(PetscFree(pforest->pointAdaptToSelfCids));
  PetscCall(PetscFree(pforest->pointSelfToAdaptCids));
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

  PetscFunctionBegin;

  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  comm = PetscObjectComm((PetscObject)dm);
  PetscCall(PetscObjectTypeCompare((PetscObject)dm,DMPFOREST,&isPforest));
  PetscCheck(isPforest,comm,PETSC_ERR_ARG_WRONG,"Expected DM type %s, got %s",DMPFOREST,((PetscObject)dm)->type_name);
  PetscCall(DMGetDimension(dm,&dim));
  PetscCheck(dim == P4EST_DIM,comm,PETSC_ERR_ARG_WRONG,"Expected DM dimension %d, got %" PetscInt_FMT,P4EST_DIM,dim);
  forest  = (DM_Forest*) dm->data;
  pforest = (DM_Forest_pforest*) forest->data;
  PetscCall(DMForestGetBaseDM(dm,&base));
  if (base) {
    PetscCall(DMGetLabel(base,"ghost",&ghostLabelBase));
  }
  if (!pforest->plex) {
    PetscMPIInt size;

    PetscCallMPI(MPI_Comm_size(comm,&size));
    PetscCall(DMCreate(comm,&newPlex));
    PetscCall(DMSetType(newPlex,DMPLEX));
    PetscCall(DMSetMatType(newPlex,dm->mattype));
    /* share labels */
    PetscCall(DMCopyLabels(dm, newPlex, PETSC_OWN_POINTER, PETSC_TRUE, DM_COPY_LABELS_FAIL));
    PetscCall(DMForestGetAdjacencyDimension(dm,&adjDim));
    PetscCall(DMForestGetAdjacencyCodimension(dm,&adjCodim));
    PetscCall(DMGetCoordinateDim(dm,&coordDim));
    if (adjDim == 0) {
      ctype = P4EST_CONNECT_FULL;
    } else if (adjCodim == 1) {
      ctype = P4EST_CONNECT_FACE;
#if defined(P4_TO_P8)
    } else if (adjDim == 1) {
      ctype = P8EST_CONNECT_EDGE;
#endif
    } else {
      SETERRQ(PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONG,"Invalid adjacency dimension %" PetscInt_FMT,adjDim);
    }
    PetscCheck(ctype == P4EST_CONNECT_FULL,PetscObjectComm((PetscObject)dm),PETSC_ERR_ARG_WRONG,"Adjacency dimension %" PetscInt_FMT " / codimension %" PetscInt_FMT " not supported yet",adjDim,adjCodim);
    PetscCall(DMForestGetPartitionOverlap(dm,&overlap));
    PetscCall(DMPlexSetOverlap_Plex(newPlex,NULL,overlap));

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
    PetscCall(locidx_to_PetscInt(points_per_dim));
    PetscCall(locidx_to_PetscInt(cone_sizes));
    PetscCall(locidx_to_PetscInt(cones));
    PetscCall(locidx_to_PetscInt(cone_orientations));
    PetscCall(coords_double_to_PetscScalar(coords, coordDim));
    PetscCall(locidx_to_PetscInt(children));
    PetscCall(locidx_to_PetscInt(parents));
    PetscCall(locidx_to_PetscInt(childids));
    PetscCall(locidx_to_PetscInt(leaves));
    PetscCall(locidx_pair_to_PetscSFNode(remotes));

    PetscCall(DMSetDimension(newPlex,P4EST_DIM));
    PetscCall(DMSetCoordinateDim(newPlex,coordDim));
    PetscCall(DMPlexSetMaxProjectionHeight(newPlex,P4EST_DIM - 1));
    PetscCall(DMPlexCreateFromDAG(newPlex,P4EST_DIM,(PetscInt*)points_per_dim->array,(PetscInt*)cone_sizes->array,(PetscInt*)cones->array,(PetscInt*)cone_orientations->array,(PetscScalar*)coords->array));
    PetscCall(DMPlexConvertOldOrientations_Internal(newPlex));
    PetscCall(DMCreateReferenceTree_pforest(comm,&refTree));
    PetscCall(DMPlexSetReferenceTree(newPlex,refTree));
    PetscCall(PetscSectionCreate(comm,&parentSection));
    PetscCall(DMPlexGetChart(newPlex,&pStart,&pEnd));
    PetscCall(PetscSectionSetChart(parentSection,pStart,pEnd));
    count = children->elem_count;
    for (zz = 0; zz < count; zz++) {
      PetscInt child = *((PetscInt*) sc_array_index(children,zz));

      PetscCall(PetscSectionSetDof(parentSection,child,1));
    }
    PetscCall(PetscSectionSetUp(parentSection));
    PetscCall(DMPlexSetTree(newPlex,parentSection,(PetscInt*)parents->array,(PetscInt*)childids->array));
    PetscCall(PetscSectionDestroy(&parentSection));
    PetscCall(PetscSFCreate(comm,&pointSF));
    /*
       These arrays defining the sf are from the p4est library, but the code there shows the leaves being populated in increasing order.
       https://gitlab.com/petsc/petsc/merge_requests/2248#note_240186391
    */
    PetscCall(PetscSFSetGraph(pointSF,pEnd - pStart,(PetscInt)leaves->elem_count,(PetscInt*)leaves->array,PETSC_COPY_VALUES,(PetscSFNode*)remotes->array,PETSC_COPY_VALUES));
    PetscCall(DMSetPointSF(newPlex,pointSF));
    PetscCall(DMSetPointSF(dm,pointSF));
    {
      DM coordDM;

      PetscCall(DMGetCoordinateDM(newPlex,&coordDM));
      PetscCall(DMSetPointSF(coordDM,pointSF));
    }
    PetscCall(PetscSFDestroy(&pointSF));
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

      PetscCall(DMGetPeriodicity(dm,&isper,&maxCell,&L,&bd));
      PetscCall(DMSetPeriodicity(newPlex,isper,maxCell,L,bd));
      PetscCall(DMPforestLocalizeCoordinates(dm,newPlex));
    }

    if (overlap > 0) { /* the p4est routine can't set all of the coordinates in its routine if there is overlap */
      Vec               coordsGlobal, coordsLocal;
      const PetscScalar *globalArray;
      PetscScalar       *localArray;
      PetscSF           coordSF;
      DM                coordDM;

      PetscCall(DMGetCoordinateDM(newPlex,&coordDM));
      PetscCall(DMGetSectionSF(coordDM,&coordSF));
      PetscCall(DMGetCoordinates(newPlex, &coordsGlobal));
      PetscCall(DMGetCoordinatesLocal(newPlex, &coordsLocal));
      PetscCall(VecGetArrayRead(coordsGlobal, &globalArray));
      PetscCall(VecGetArray(coordsLocal, &localArray));
      PetscCall(PetscSFBcastBegin(coordSF,MPIU_SCALAR,globalArray,localArray,MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(coordSF,MPIU_SCALAR,globalArray,localArray,MPI_REPLACE));
      PetscCall(VecRestoreArray(coordsLocal, &localArray));
      PetscCall(VecRestoreArrayRead(coordsGlobal, &globalArray));
      PetscCall(DMSetCoordinatesLocal(newPlex, coordsLocal));
    }
    PetscCall(DMPforestMapCoordinates(dm,newPlex));

    pforest->plex = newPlex;

    /* copy labels */
    PetscCall(DMPforestLabelsFinalize(dm,newPlex));

    if (ghostLabelBase || pforest->ghostName) { /* we have to do this after copying labels because the labels drive the construction of ghost cells */
      PetscInt numAdded;
      DM       newPlexGhosted;
      void     *ctx;

      PetscCall(DMPlexConstructGhostCells(newPlex,pforest->ghostName,&numAdded,&newPlexGhosted));
      PetscCall(DMGetApplicationContext(newPlex,&ctx));
      PetscCall(DMSetApplicationContext(newPlexGhosted,ctx));
      /* we want the sf for the ghost dm to be the one for the p4est dm as well */
      PetscCall(DMGetPointSF(newPlexGhosted,&pointSF));
      PetscCall(DMSetPointSF(dm,pointSF));
      PetscCall(DMDestroy(&newPlex));
      PetscCall(DMPlexSetReferenceTree(newPlexGhosted,refTree));
      PetscCall(DMForestClearAdaptivityForest_pforest(dm));
      newPlex = newPlexGhosted;

      /* share the labels back */
      PetscCall(DMDestroyLabelLinkList_Internal(dm));
      PetscCall(DMCopyLabels(newPlex, dm, PETSC_OWN_POINTER, PETSC_TRUE, DM_COPY_LABELS_FAIL));
      pforest->plex = newPlex;
    }
    PetscCall(DMDestroy(&refTree));
    if (dm->setfromoptionscalled) {

      PetscObjectOptionsBegin((PetscObject)newPlex);
      PetscCall(DMSetFromOptions_NonRefinement_Plex(PetscOptionsObject,newPlex));
      PetscCall(PetscObjectProcessOptionsHandlers(PetscOptionsObject,(PetscObject)newPlex));
      PetscOptionsEnd();
    }
    PetscCall(DMViewFromOptions(newPlex,NULL,"-dm_p4est_plex_view"));
    {
      PetscSection coordsSec;
      Vec          coords;
      PetscInt     cDim;

      PetscCall(DMGetCoordinateDim(newPlex,&cDim));
      PetscCall(DMGetCoordinateSection(newPlex,&coordsSec));
      PetscCall(DMSetCoordinateSection(dm,cDim,coordsSec));
      PetscCall(DMGetCoordinatesLocal(newPlex,&coords));
      PetscCall(DMSetCoordinatesLocal(dm,coords));
    }
  }
  newPlex = pforest->plex;
  if (plex) {
    DM coordDM;

    PetscCall(DMClone(newPlex,plex));
    PetscCall(DMGetCoordinateDM(newPlex,&coordDM));
    PetscCall(DMSetCoordinateDM(*plex,coordDM));
    PetscCall(DMShareDiscretization(dm,*plex));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMSetFromOptions_pforest(PetscOptionItems *PetscOptionsObject,DM dm)
{
  DM_Forest_pforest *pforest = (DM_Forest_pforest*) ((DM_Forest*) dm->data)->data;
  char              stringBuffer[256];
  PetscBool         flg;

  PetscFunctionBegin;
  PetscCall(DMSetFromOptions_Forest(PetscOptionsObject,dm));
  PetscOptionsHeadBegin(PetscOptionsObject,"DM" P4EST_STRING " options");
  PetscCall(PetscOptionsBool("-dm_p4est_partition_for_coarsening","partition forest to allow for coarsening","DMP4estSetPartitionForCoarsening",pforest->partition_for_coarsening,&(pforest->partition_for_coarsening),NULL));
  PetscCall(PetscOptionsString("-dm_p4est_ghost_label_name","the name of the ghost label when converting from a DMPlex",NULL,NULL,stringBuffer,sizeof(stringBuffer),&flg));
  PetscOptionsHeadEnd();
  if (flg) {
    PetscCall(PetscFree(pforest->ghostName));
    PetscCall(PetscStrallocpy(stringBuffer,&pforest->ghostName));
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

  PetscFunctionBegin;
  if (plex) *plex = NULL;
  PetscCall(DMSetUp(dm));
  pforest = (DM_Forest_pforest*) ((DM_Forest*) dm->data)->data;
  if (!pforest->plex) {
    PetscCall(DMConvert_pforest_plex(dm,DMPLEX,NULL));
  }
  PetscCall(DMShareDiscretization(dm,pforest->plex));
  if (plex) *plex = pforest->plex;
  PetscFunctionReturn(0);
}

#define DMCreateInterpolation_pforest _append_pforest(DMCreateInterpolation)
static PetscErrorCode DMCreateInterpolation_pforest(DM dmCoarse, DM dmFine, Mat *interpolation, Vec *scaling)
{
  PetscSection   gsc, gsf;
  PetscInt       m, n;
  DM             cdm;

  PetscFunctionBegin;
  PetscCall(DMGetGlobalSection(dmFine, &gsf));
  PetscCall(PetscSectionGetConstrainedStorageSize(gsf, &m));
  PetscCall(DMGetGlobalSection(dmCoarse, &gsc));
  PetscCall(PetscSectionGetConstrainedStorageSize(gsc, &n));

  PetscCall(MatCreate(PetscObjectComm((PetscObject) dmFine), interpolation));
  PetscCall(MatSetSizes(*interpolation, m, n, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(*interpolation, MATAIJ));

  PetscCall(DMGetCoarseDM(dmFine, &cdm));
  PetscCheck(cdm == dmCoarse,PetscObjectComm((PetscObject)dmFine),PETSC_ERR_SUP,"Only interpolation from coarse DM for now");

  {
    DM       plexF, plexC;
    PetscSF  sf;
    PetscInt *cids;
    PetscInt dofPerDim[4] = {1,1,1,1};

    PetscCall(DMPforestGetPlex(dmCoarse,&plexC));
    PetscCall(DMPforestGetPlex(dmFine,&plexF));
    PetscCall(DMPforestGetTransferSF_Internal(dmCoarse, dmFine, dofPerDim, &sf, PETSC_TRUE, &cids));
    PetscCall(PetscSFSetUp(sf));
    PetscCall(DMPlexComputeInterpolatorTree(plexC, plexF, sf, cids, *interpolation));
    PetscCall(PetscSFDestroy(&sf));
    PetscCall(PetscFree(cids));
  }
  PetscCall(MatViewFromOptions(*interpolation, NULL, "-interp_mat_view"));
  /* Use naive scaling */
  PetscCall(DMCreateInterpolationScale(dmCoarse, dmFine, *interpolation, scaling));
  PetscFunctionReturn(0);
}

#define DMCreateInjection_pforest _append_pforest(DMCreateInjection)
static PetscErrorCode DMCreateInjection_pforest(DM dmCoarse, DM dmFine, Mat *injection)
{
  PetscSection   gsc, gsf;
  PetscInt       m, n;
  DM             cdm;

  PetscFunctionBegin;
  PetscCall(DMGetGlobalSection(dmFine, &gsf));
  PetscCall(PetscSectionGetConstrainedStorageSize(gsf, &n));
  PetscCall(DMGetGlobalSection(dmCoarse, &gsc));
  PetscCall(PetscSectionGetConstrainedStorageSize(gsc, &m));

  PetscCall(MatCreate(PetscObjectComm((PetscObject) dmFine), injection));
  PetscCall(MatSetSizes(*injection, m, n, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(*injection, MATAIJ));

  PetscCall(DMGetCoarseDM(dmFine, &cdm));
  PetscCheck(cdm == dmCoarse,PetscObjectComm((PetscObject)dmFine),PETSC_ERR_SUP,"Only injection to coarse DM for now");

  {
    DM       plexF, plexC;
    PetscSF  sf;
    PetscInt *cids;
    PetscInt dofPerDim[4] = {1,1,1,1};

    PetscCall(DMPforestGetPlex(dmCoarse,&plexC));
    PetscCall(DMPforestGetPlex(dmFine,&plexF));
    PetscCall(DMPforestGetTransferSF_Internal(dmCoarse, dmFine, dofPerDim, &sf, PETSC_TRUE, &cids));
    PetscCall(PetscSFSetUp(sf));
    PetscCall(DMPlexComputeInjectorTree(plexC, plexF, sf, cids, *injection));
    PetscCall(PetscSFDestroy(&sf));
    PetscCall(PetscFree(cids));
  }
  PetscCall(MatViewFromOptions(*injection, NULL, "-inject_mat_view"));
  /* Use naive scaling */
  PetscFunctionReturn(0);
}

#define DMForestTransferVecFromBase_pforest _append_pforest(DMForestTransferVecFromBase)
static PetscErrorCode DMForestTransferVecFromBase_pforest(DM dm, Vec vecIn, Vec vecOut)
{
  DM             dmIn, dmVecIn, base, basec, plex, coarseDM;
  DM             *hierarchy;
  PetscSF        sfRed = NULL;
  PetscDS        ds;
  Vec            vecInLocal, vecOutLocal;
  DMLabel        subpointMap;
  PetscInt       minLevel, mh, n_hi, i;
  PetscBool      hiforest, *hierarchy_forest;

  PetscFunctionBegin;
  PetscCall(VecGetDM(vecIn,&dmVecIn));
  PetscCall(DMGetDS(dmVecIn,&ds));
  PetscCheck(ds,PetscObjectComm((PetscObject)dmVecIn),PETSC_ERR_SUP,"Cannot transfer without a PetscDS object");
  { /* we cannot stick user contexts into function callbacks for DMProjectFieldLocal! */
    PetscSection section;
    PetscInt     Nf;

    PetscCall(DMGetLocalSection(dmVecIn,&section));
    PetscCall(PetscSectionGetNumFields(section,&Nf));
    PetscCheck(Nf <= 3,PetscObjectComm((PetscObject)dmVecIn),PETSC_ERR_SUP,"Number of fields %" PetscInt_FMT " are currently not supported! Send an email at petsc-dev@mcs.anl.gov",Nf);
  }
  PetscCall(DMForestGetMinimumRefinement(dm,&minLevel));
  PetscCheck(!minLevel,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Cannot transfer with minimum refinement set to %" PetscInt_FMT ". Rerun with DMForestSetMinimumRefinement(dm,0)",minLevel);
  PetscCall(DMForestGetBaseDM(dm,&base));
  PetscCheck(base,PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Missing base DM");

  PetscCall(VecSet(vecOut,0.0));
  if (dmVecIn == base) { /* sequential runs */
    PetscCall(PetscObjectReference((PetscObject)vecIn));
  } else {
    PetscSection secIn, secInRed;
    Vec          vecInRed, vecInLocal;

    PetscCall(PetscObjectQuery((PetscObject)base,"_base_migration_sf",(PetscObject*)&sfRed));
    PetscCheck(sfRed,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not the DM set with DMForestSetBaseDM()");
    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dmVecIn),&secInRed));
    PetscCall(VecCreate(PETSC_COMM_SELF,&vecInRed));
    PetscCall(DMGetLocalSection(dmVecIn,&secIn));
    PetscCall(DMGetLocalVector(dmVecIn,&vecInLocal));
    PetscCall(DMGlobalToLocalBegin(dmVecIn,vecIn,INSERT_VALUES,vecInLocal));
    PetscCall(DMGlobalToLocalEnd(dmVecIn,vecIn,INSERT_VALUES,vecInLocal));
    PetscCall(DMPlexDistributeField(dmVecIn,sfRed,secIn,vecInLocal,secInRed,vecInRed));
    PetscCall(DMRestoreLocalVector(dmVecIn,&vecInLocal));
    PetscCall(PetscSectionDestroy(&secInRed));
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
    PetscCall(DMSetUp(dmIn));
    PetscCall(DMIsForest(dmIn,&isforest));
    PetscCheck(isforest,PetscObjectComm((PetscObject)dmIn),PETSC_ERR_SUP,"Cannot currently transfer through a mixed hierarchy! Found DM type %s",((PetscObject)dmIn)->type_name);
    coarseDM = NULL;
    if (hiforest) {
      PetscCall(DMForestGetAdaptivityForest(dmIn,&coarseDM));
    }
    if (!coarseDM) { /* DMForest hierarchy ended, we keep upsweeping through the DM hierarchy */
      hiforest = PETSC_FALSE;
      PetscCall(DMGetCoarseDM(dmIn,&coarseDM));
    }
    n_hi++;
  } while (coarseDM);

  PetscCall(PetscMalloc2(n_hi,&hierarchy,n_hi,&hierarchy_forest));

  i = 0;
  hiforest = PETSC_TRUE;
  coarseDM = dm;
  do {
    dmIn = coarseDM;
    coarseDM = NULL;
    if (hiforest) {
      PetscCall(DMForestGetAdaptivityForest(dmIn,&coarseDM));
    }
    if (!coarseDM) { /* DMForest hierarchy ended, we keep upsweeping through the DM hierarchy */
      hiforest = PETSC_FALSE;
      PetscCall(DMGetCoarseDM(dmIn,&coarseDM));
    }
    i++;
    hierarchy[n_hi - i] = dmIn;
  } while (coarseDM);

  /* project base vector on the coarsest forest (minimum refinement = 0) */
  PetscCall(DMPforestGetPlex(dmIn,&plex));

  /* Check this plex is compatible with the base */
  {
    IS       gnum[2];
    PetscInt ncells[2],gncells[2];

    PetscCall(DMPlexGetCellNumbering(base,&gnum[0]));
    PetscCall(DMPlexGetCellNumbering(plex,&gnum[1]));
    PetscCall(ISGetMinMax(gnum[0],NULL,&ncells[0]));
    PetscCall(ISGetMinMax(gnum[1],NULL,&ncells[1]));
    PetscCall(MPIU_Allreduce(ncells,gncells,2,MPIU_INT,MPI_MAX,PetscObjectComm((PetscObject)dm)));
    PetscCheck(gncells[0] == gncells[1],PetscObjectComm((PetscObject)dm),PETSC_ERR_SUP,"Invalid number of base cells! Expected %" PetscInt_FMT ", found %" PetscInt_FMT,gncells[0]+1,gncells[1]+1);
  }

  PetscCall(DMGetLabel(dmIn,"_forest_base_subpoint_map",&subpointMap));
  PetscCheck(subpointMap,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing _forest_base_subpoint_map label");

  PetscCall(DMPlexGetMaxProjectionHeight(base,&mh));
  PetscCall(DMPlexSetMaxProjectionHeight(plex,mh));

  PetscCall(DMClone(base,&basec));
  PetscCall(DMCopyDisc(dmVecIn,basec));
  if (sfRed) {
    PetscCall(PetscObjectReference((PetscObject)vecIn));
    vecInLocal = vecIn;
  } else {
    PetscCall(DMCreateLocalVector(basec,&vecInLocal));
    PetscCall(DMGlobalToLocalBegin(basec,vecIn,INSERT_VALUES,vecInLocal));
    PetscCall(DMGlobalToLocalEnd(basec,vecIn,INSERT_VALUES,vecInLocal));
  }

  PetscCall(DMGetLocalVector(dmIn,&vecOutLocal));
  { /* get degrees of freedom ordered onto dmIn */
    PetscSF            basetocoarse;
    PetscInt           bStart, bEnd, nroots;
    PetscInt           iStart, iEnd, nleaves, leaf;
    PetscMPIInt        rank;
    PetscSFNode       *remotes;
    PetscSection       secIn, secOut;
    PetscInt          *remoteOffsets;
    PetscSF            transferSF;
    const PetscScalar *inArray;
    PetscScalar       *outArray;

    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)basec), &rank));
    PetscCall(DMPlexGetChart(basec, &bStart, &bEnd));
    nroots = PetscMax(bEnd - bStart, 0);
    PetscCall(DMPlexGetChart(plex, &iStart, &iEnd));
    nleaves = PetscMax(iEnd - iStart, 0);

    PetscCall(PetscMalloc1(nleaves, &remotes));
    for (leaf = iStart; leaf < iEnd; leaf++) {
      PetscInt index;

      remotes[leaf - iStart].rank = rank;
      PetscCall(DMLabelGetValue(subpointMap, leaf, &index));
      remotes[leaf - iStart].index = index;
    }

    PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)basec), &basetocoarse));
    PetscCall(PetscSFSetGraph(basetocoarse, nroots, nleaves, NULL, PETSC_OWN_POINTER, remotes, PETSC_OWN_POINTER));
    PetscCall(PetscSFSetUp(basetocoarse));
    PetscCall(DMGetLocalSection(basec,&secIn));
    PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject)dmIn),&secOut));
    PetscCall(PetscSFDistributeSection(basetocoarse, secIn, &remoteOffsets, secOut));
    PetscCall(PetscSFCreateSectionSF(basetocoarse, secIn, remoteOffsets, secOut, &transferSF));
    PetscCall(PetscFree(remoteOffsets));
    PetscCall(VecGetArrayWrite(vecOutLocal, &outArray));
    PetscCall(VecGetArrayRead(vecInLocal, &inArray));
    PetscCall(PetscSFBcastBegin(transferSF, MPIU_SCALAR, inArray, outArray,MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(transferSF, MPIU_SCALAR, inArray, outArray,MPI_REPLACE));
    PetscCall(VecRestoreArrayRead(vecInLocal, &inArray));
    PetscCall(VecRestoreArrayWrite(vecOutLocal, &outArray));
    PetscCall(PetscSFDestroy(&transferSF));
    PetscCall(PetscSectionDestroy(&secOut));
    PetscCall(PetscSFDestroy(&basetocoarse));
  }
  PetscCall(VecDestroy(&vecInLocal));
  PetscCall(DMDestroy(&basec));
  PetscCall(VecDestroy(&vecIn));

  /* output */
  if (n_hi > 1) { /* downsweep the stored hierarchy */
    Vec vecOut1, vecOut2;
    DM  fineDM;

    PetscCall(DMGetGlobalVector(dmIn,&vecOut1));
    PetscCall(DMLocalToGlobal(dmIn,vecOutLocal,INSERT_VALUES,vecOut1));
    PetscCall(DMRestoreLocalVector(dmIn,&vecOutLocal));
    for (i = 1; i < n_hi-1; i++) {
      fineDM  = hierarchy[i];
      PetscCall(DMGetGlobalVector(fineDM,&vecOut2));
      PetscCall(DMForestTransferVec(dmIn,vecOut1,fineDM,vecOut2,PETSC_TRUE,0.0));
      PetscCall(DMRestoreGlobalVector(dmIn,&vecOut1));
      vecOut1 = vecOut2;
      dmIn    = fineDM;
    }
    PetscCall(DMForestTransferVec(dmIn,vecOut1,dm,vecOut,PETSC_TRUE,0.0));
    PetscCall(DMRestoreGlobalVector(dmIn,&vecOut1));
  } else {
    PetscCall(DMLocalToGlobal(dmIn,vecOutLocal,INSERT_VALUES,vecOut));
    PetscCall(DMRestoreLocalVector(dmIn,&vecOutLocal));
  }
  PetscCall(PetscFree2(hierarchy,hierarchy_forest));
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

  PetscFunctionBegin;
  forestOut = (DM_Forest *) dmOut->data;
  forestIn  = (DM_Forest *) dmIn->data;

  PetscCall(DMForestGetAdaptivityForest(dmOut,&adaptOut));
  PetscCall(DMForestGetAdaptivityPurpose(dmOut,&purposeOut));
  forestAdaptOut = adaptOut ? (DM_Forest *) adaptOut->data : NULL;

  PetscCall(DMForestGetAdaptivityForest(dmIn,&adaptIn));
  PetscCall(DMForestGetAdaptivityPurpose(dmIn,&purposeIn));
  forestAdaptIn  = adaptIn ? (DM_Forest *) adaptIn->data : NULL;

  if (forestAdaptOut == forestIn) {
    switch (purposeOut) {
    case DM_ADAPT_REFINE:
      PetscCall(DMPforestGetTransferSF_Internal(dmIn,dmOut,dofPerDim,&inSF,PETSC_TRUE,&inCids));
      PetscCall(PetscSFSetUp(inSF));
      break;
    case DM_ADAPT_COARSEN:
    case DM_ADAPT_COARSEN_LAST:
      PetscCall(DMPforestGetTransferSF_Internal(dmOut,dmIn,dofPerDim,&outSF,PETSC_TRUE,&outCids));
      PetscCall(PetscSFSetUp(outSF));
      break;
    default:
      PetscCall(DMPforestGetTransferSF_Internal(dmIn,dmOut,dofPerDim,&inSF,PETSC_TRUE,&inCids));
      PetscCall(DMPforestGetTransferSF_Internal(dmOut,dmIn,dofPerDim,&outSF,PETSC_FALSE,&outCids));
      PetscCall(PetscSFSetUp(inSF));
      PetscCall(PetscSFSetUp(outSF));
    }
  } else if (forestAdaptIn == forestOut) {
    switch (purposeIn) {
    case DM_ADAPT_REFINE:
      PetscCall(DMPforestGetTransferSF_Internal(dmOut,dmIn,dofPerDim,&outSF,PETSC_TRUE,&inCids));
      PetscCall(PetscSFSetUp(outSF));
      break;
    case DM_ADAPT_COARSEN:
    case DM_ADAPT_COARSEN_LAST:
      PetscCall(DMPforestGetTransferSF_Internal(dmIn,dmOut,dofPerDim,&inSF,PETSC_TRUE,&inCids));
      PetscCall(PetscSFSetUp(inSF));
      break;
    default:
      PetscCall(DMPforestGetTransferSF_Internal(dmIn,dmOut,dofPerDim,&inSF,PETSC_TRUE,&inCids));
      PetscCall(DMPforestGetTransferSF_Internal(dmOut,dmIn,dofPerDim,&outSF,PETSC_FALSE,&outCids));
      PetscCall(PetscSFSetUp(inSF));
      PetscCall(PetscSFSetUp(outSF));
    }
  } else SETERRQ(PetscObjectComm((PetscObject)dmIn),PETSC_ERR_SUP,"Only support transfer from pre-adaptivity to post-adaptivity right now");
  PetscCall(DMPforestGetPlex(dmIn,&plexIn));
  PetscCall(DMPforestGetPlex(dmOut,&plexOut));

  PetscCall(DMPlexTransferVecTree(plexIn,vecIn,plexOut,vecOut,inSF,outSF,inCids,outCids,useBCs,time));
  PetscCall(PetscFree(inCids));
  PetscCall(PetscFree(outCids));
  PetscCall(PetscSFDestroy(&inSF));
  PetscCall(PetscSFDestroy(&outSF));
  PetscCall(PetscFree(inCids));
  PetscCall(PetscFree(outCids));
  PetscFunctionReturn(0);
}

#define DMCreateCoordinateDM_pforest _append_pforest(DMCreateCoordinateDM)
static PetscErrorCode DMCreateCoordinateDM_pforest(DM dm,DM *cdm)
{
  DM             plex;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMPforestGetPlex(dm,&plex));
  PetscCall(DMGetCoordinateDM(plex,cdm));
  PetscCall(PetscObjectReference((PetscObject)*cdm));
  PetscFunctionReturn(0);
}

#define VecViewLocal_pforest _append_pforest(VecViewLocal)
static PetscErrorCode VecViewLocal_pforest(Vec vec,PetscViewer viewer)
{
  DM             dm, plex;

  PetscFunctionBegin;
  PetscCall(VecGetDM(vec,&dm));
  PetscCall(DMPforestGetPlex(dm,&plex));
  PetscCall(VecSetDM(vec,plex));
  PetscCall(VecView_Plex_Local(vec,viewer));
  PetscCall(VecSetDM(vec,dm));
  PetscFunctionReturn(0);
}

#define VecView_pforest _append_pforest(VecView)
static PetscErrorCode VecView_pforest(Vec vec,PetscViewer viewer)
{
  DM             dm, plex;

  PetscFunctionBegin;
  PetscCall(VecGetDM(vec,&dm));
  PetscCall(DMPforestGetPlex(dm,&plex));
  PetscCall(VecSetDM(vec,plex));
  PetscCall(VecView_Plex(vec,viewer));
  PetscCall(VecSetDM(vec,dm));
  PetscFunctionReturn(0);
}

#define VecView_pforest_Native _infix_pforest(VecView,_Native)
static PetscErrorCode VecView_pforest_Native(Vec vec,PetscViewer viewer)
{
  DM             dm, plex;

  PetscFunctionBegin;
  PetscCall(VecGetDM(vec,&dm));
  PetscCall(DMPforestGetPlex(dm,&plex));
  PetscCall(VecSetDM(vec,plex));
  PetscCall(VecView_Plex_Native(vec,viewer));
  PetscCall(VecSetDM(vec,dm));
  PetscFunctionReturn(0);
}

#define VecLoad_pforest _append_pforest(VecLoad)
static PetscErrorCode VecLoad_pforest(Vec vec,PetscViewer viewer)
{
  DM             dm, plex;

  PetscFunctionBegin;
  PetscCall(VecGetDM(vec,&dm));
  PetscCall(DMPforestGetPlex(dm,&plex));
  PetscCall(VecSetDM(vec,plex));
  PetscCall(VecLoad_Plex(vec,viewer));
  PetscCall(VecSetDM(vec,dm));
  PetscFunctionReturn(0);
}

#define VecLoad_pforest_Native _infix_pforest(VecLoad,_Native)
static PetscErrorCode VecLoad_pforest_Native(Vec vec,PetscViewer viewer)
{
  DM             dm, plex;

  PetscFunctionBegin;
  PetscCall(VecGetDM(vec,&dm));
  PetscCall(DMPforestGetPlex(dm,&plex));
  PetscCall(VecSetDM(vec,plex));
  PetscCall(VecLoad_Plex_Native(vec,viewer));
  PetscCall(VecSetDM(vec,dm));
  PetscFunctionReturn(0);
}

#define DMCreateGlobalVector_pforest _append_pforest(DMCreateGlobalVector)
static PetscErrorCode DMCreateGlobalVector_pforest(DM dm,Vec *vec)
{
  PetscFunctionBegin;
  PetscCall(DMCreateGlobalVector_Section_Private(dm,vec));
  /* PetscCall(VecSetOperation(*vec, VECOP_DUPLICATE, (void(*)(void)) VecDuplicate_MPI_DM)); */
  PetscCall(VecSetOperation(*vec, VECOP_VIEW, (void (*)(void))VecView_pforest));
  PetscCall(VecSetOperation(*vec, VECOP_VIEWNATIVE, (void (*)(void))VecView_pforest_Native));
  PetscCall(VecSetOperation(*vec, VECOP_LOAD, (void (*)(void))VecLoad_pforest));
  PetscCall(VecSetOperation(*vec, VECOP_LOADNATIVE, (void (*)(void))VecLoad_pforest_Native));
  PetscFunctionReturn(0);
}

#define DMCreateLocalVector_pforest _append_pforest(DMCreateLocalVector)
static PetscErrorCode DMCreateLocalVector_pforest(DM dm,Vec *vec)
{
  PetscFunctionBegin;
  PetscCall(DMCreateLocalVector_Section_Private(dm,vec));
  PetscCall(VecSetOperation(*vec, VECOP_VIEW, (void (*)(void))VecViewLocal_pforest));
  PetscFunctionReturn(0);
}

#define DMCreateMatrix_pforest _append_pforest(DMCreateMatrix)
static PetscErrorCode DMCreateMatrix_pforest(DM dm,Mat *mat)
{
  DM             plex;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMPforestGetPlex(dm,&plex));
  if (plex->prealloc_only != dm->prealloc_only) plex->prealloc_only = dm->prealloc_only;  /* maybe this should go into forest->plex */
  PetscCall(DMCreateMatrix(plex,mat));
  PetscCall(MatSetDM(*mat,dm));
  PetscFunctionReturn(0);
}

#define DMProjectFunctionLocal_pforest _append_pforest(DMProjectFunctionLocal)
static PetscErrorCode DMProjectFunctionLocal_pforest(DM dm, PetscReal time, PetscErrorCode (**funcs) (PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void*), void **ctxs, InsertMode mode, Vec localX)
{
  DM             plex;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMPforestGetPlex(dm,&plex));
  PetscCall(DMProjectFunctionLocal(plex,time,funcs,ctxs,mode,localX));
  PetscFunctionReturn(0);
}

#define DMProjectFunctionLabelLocal_pforest _append_pforest(DMProjectFunctionLabelLocal)
static PetscErrorCode DMProjectFunctionLabelLocal_pforest(DM dm, PetscReal time, DMLabel label, PetscInt numIds, const PetscInt ids[], PetscInt Ncc, const PetscInt comps[], PetscErrorCode (**funcs) (PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void*), void **ctxs, InsertMode mode, Vec localX)
{
  DM             plex;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMPforestGetPlex(dm,&plex));
  PetscCall(DMProjectFunctionLabelLocal(plex,time,label,numIds,ids,Ncc,comps,funcs,ctxs,mode,localX));
  PetscFunctionReturn(0);
}

#define DMProjectFieldLocal_pforest _append_pforest(DMProjectFieldLocal)
PetscErrorCode DMProjectFieldLocal_pforest(DM dm, PetscReal time, Vec localU,void (**funcs) (PetscInt, PetscInt, PetscInt,
                                                                             const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                             const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                                                                             PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]),InsertMode mode, Vec localX)
{
  DM             plex;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMPforestGetPlex(dm,&plex));
  PetscCall(DMProjectFieldLocal(plex,time,localU,funcs,mode,localX));
  PetscFunctionReturn(0);
}

#define DMComputeL2Diff_pforest _append_pforest(DMComputeL2Diff)
PetscErrorCode DMComputeL2Diff_pforest(DM dm, PetscReal time, PetscErrorCode (**funcs) (PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void*), void **ctxs, Vec X, PetscReal *diff)
{
  DM             plex;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMPforestGetPlex(dm,&plex));
  PetscCall(DMComputeL2Diff(plex,time,funcs,ctxs,X,diff));
  PetscFunctionReturn(0);
}

#define DMComputeL2FieldDiff_pforest _append_pforest(DMComputeL2FieldDiff)
PetscErrorCode DMComputeL2FieldDiff_pforest(DM dm, PetscReal time, PetscErrorCode (**funcs) (PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void*), void **ctxs, Vec X, PetscReal diff[])
{
  DM             plex;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMPforestGetPlex(dm,&plex));
  PetscCall(DMComputeL2FieldDiff(plex,time,funcs,ctxs,X,diff));
  PetscFunctionReturn(0);
}

#define DMCreatelocalsection_pforest _append_pforest(DMCreatelocalsection)
static PetscErrorCode DMCreatelocalsection_pforest(DM dm)
{
  DM             plex;
  PetscSection   section;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMPforestGetPlex(dm,&plex));
  PetscCall(DMGetLocalSection(plex,&section));
  PetscCall(DMSetLocalSection(dm,section));
  PetscFunctionReturn(0);
}

#define DMCreateDefaultConstraints_pforest _append_pforest(DMCreateDefaultConstraints)
static PetscErrorCode DMCreateDefaultConstraints_pforest(DM dm)
{
  DM             plex;
  Mat            mat;
  Vec            bias;
  PetscSection   section;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMPforestGetPlex(dm,&plex));
  PetscCall(DMGetDefaultConstraints(plex,&section,&mat,&bias));
  PetscCall(DMSetDefaultConstraints(dm,section,mat,bias));
  PetscFunctionReturn(0);
}

#define DMGetDimPoints_pforest _append_pforest(DMGetDimPoints)
static PetscErrorCode DMGetDimPoints_pforest(DM dm, PetscInt dim, PetscInt *cStart, PetscInt *cEnd)
{
  DM             plex;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscCall(DMPforestGetPlex(dm,&plex));
  PetscCall(DMGetDimPoints(plex,dim,cStart,cEnd));
  PetscFunctionReturn(0);
}

/* Need to forward declare */
#define DMInitialize_pforest _append_pforest(DMInitialize)
static PetscErrorCode DMInitialize_pforest(DM dm);

#define DMClone_pforest _append_pforest(DMClone)
static PetscErrorCode DMClone_pforest(DM dm, DM *newdm)
{
  PetscFunctionBegin;
  PetscCall(DMClone_Forest(dm,newdm));
  PetscCall(DMInitialize_pforest(*newdm));
  PetscFunctionReturn(0);
}

#define DMForestCreateCellChart_pforest _append_pforest(DMForestCreateCellChart)
static PetscErrorCode DMForestCreateCellChart_pforest(DM dm, PetscInt *cStart, PetscInt *cEnd)
{
  DM_Forest         *forest;
  DM_Forest_pforest *pforest;
  PetscInt          overlap;

  PetscFunctionBegin;
  PetscCall(DMSetUp(dm));
  forest  = (DM_Forest*) dm->data;
  pforest = (DM_Forest_pforest*) forest->data;
  *cStart = 0;
  PetscCall(DMForestGetPartitionOverlap(dm,&overlap));
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

  PetscFunctionBegin;
  PetscCall(DMForestGetCellChart(dm,&cStart,&cEnd));
  forest      = (DM_Forest*)         dm->data;
  pforest     = (DM_Forest_pforest*) forest->data;
  nRoots      = cEnd - cStart;
  cLocalStart = pforest->cLocalStart;
  cLocalEnd   = pforest->cLocalEnd;
  nLeaves     = 0;
  PetscCall(DMForestGetPartitionOverlap(dm,&overlap));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank));
  if (overlap && pforest->ghost) {
    PetscSFNode      *mirror;
    p4est_quadrant_t *mirror_array;
    PetscInt         nMirror, nGhostPre, nSelf, q;
    void             **mirrorPtrs;

    nMirror      = (PetscInt) pforest->ghost->mirrors.elem_count;
    nSelf        = cLocalEnd - cLocalStart;
    nLeaves      = nRoots - nSelf;
    nGhostPre    = (PetscInt) pforest->ghost->proc_offsets[rank];
    PetscCall(PetscMalloc1(nLeaves,&mine));
    PetscCall(PetscMalloc1(nLeaves,&remote));
    PetscCall(PetscMalloc2(nMirror,&mirror,nMirror,&mirrorPtrs));
    mirror_array = (p4est_quadrant_t*) pforest->ghost->mirrors.array;
    for (q = 0; q < nMirror; q++) {
      p4est_quadrant_t *mir = &(mirror_array[q]);

      mirror[q].rank  = rank;
      mirror[q].index = (PetscInt) mir->p.piggy3.local_num + cLocalStart;
      mirrorPtrs[q]   = (void*) &(mirror[q]);
    }
    PetscStackCallP4est(p4est_ghost_exchange_custom,(pforest->forest,pforest->ghost,sizeof(PetscSFNode),mirrorPtrs,remote));
    PetscCall(PetscFree2(mirror,mirrorPtrs));
    for (q = 0; q < nGhostPre; q++) mine[q] = q;
    for (; q < nLeaves; q++) mine[q] = (q - nGhostPre) + cLocalEnd;
  }
  PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)dm),&sf));
  PetscCall(PetscSFSetGraph(sf,nRoots,nLeaves,mine,PETSC_OWN_POINTER,remote,PETSC_OWN_POINTER));
  *cellSF = sf;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMCreateNeumannOverlap_pforest(DM dm, IS* ovl, Mat *J, PetscErrorCode (**setup)(Mat, PetscReal, Vec, Vec, PetscReal, IS, void*), void **setup_ctx)
{
  DM             plex;

  PetscFunctionBegin;
  PetscCall(DMPforestGetPlex(dm,&plex));
  PetscCall(DMCreateNeumannOverlap_Plex(plex,ovl,J,setup,setup_ctx));
  if (!*setup) {
    PetscCall(PetscObjectQueryFunction((PetscObject)dm, "MatComputeNeumannOverlap_C", setup));
    if (*setup) {
      PetscCall(PetscObjectCompose((PetscObject)*ovl, "_DM_Original_HPDDM", (PetscObject)dm));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMInitialize_pforest(DM dm)
{
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
  dm->ops->createlocalsection        = DMCreatelocalsection_pforest;
  dm->ops->createdefaultconstraints  = DMCreateDefaultConstraints_pforest;
  dm->ops->computel2diff             = DMComputeL2Diff_pforest;
  dm->ops->computel2fielddiff        = DMComputeL2FieldDiff_pforest;
  dm->ops->getdimpoints              = DMGetDimPoints_pforest;

  PetscCall(PetscObjectComposeFunction((PetscObject)dm,PetscStringize(DMConvert_plex_pforest) "_C",DMConvert_plex_pforest));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm,PetscStringize(DMConvert_pforest_plex) "_C",DMConvert_pforest_plex));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm,"DMCreateNeumannOverlap_C",DMCreateNeumannOverlap_pforest));
  PetscCall(PetscObjectComposeFunction((PetscObject)dm,"DMPlexGetOverlap_C",DMForestGetPartitionOverlap));
  PetscFunctionReturn(0);
}

#define DMCreate_pforest _append_pforest(DMCreate)
PETSC_EXTERN PetscErrorCode DMCreate_pforest(DM dm)
{
  DM_Forest         *forest;
  DM_Forest_pforest *pforest;

  PetscFunctionBegin;
  PetscCall(PetscP4estInitialize());
  PetscCall(DMCreate_Forest(dm));
  PetscCall(DMInitialize_pforest(dm));
  PetscCall(DMSetDimension(dm,P4EST_DIM));

  /* set forest defaults */
  PetscCall(DMForestSetTopology(dm,"unit"));
  PetscCall(DMForestSetMinimumRefinement(dm,0));
  PetscCall(DMForestSetInitialRefinement(dm,0));
  PetscCall(DMForestSetMaximumRefinement(dm,P4EST_QMAXLEVEL));
  PetscCall(DMForestSetGradeFactor(dm,2));
  PetscCall(DMForestSetAdjacencyDimension(dm,0));
  PetscCall(DMForestSetPartitionOverlap(dm,0));

  /* create p4est data */
  PetscCall(PetscNewLog(dm,&pforest));

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
