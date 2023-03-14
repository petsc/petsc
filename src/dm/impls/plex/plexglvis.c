#include <petsc/private/glvisviewerimpl.h>
#include <petsc/private/petscimpl.h>
#include <petsc/private/dmpleximpl.h>
#include <petscbt.h>
#include <petscdmplex.h>
#include <petscsf.h>
#include <petscds.h>

typedef struct {
  PetscInt    nf;
  VecScatter *scctx;
} GLVisViewerCtx;

static PetscErrorCode DestroyGLVisViewerCtx_Private(void *vctx)
{
  GLVisViewerCtx *ctx = (GLVisViewerCtx *)vctx;
  PetscInt        i;

  PetscFunctionBegin;
  for (i = 0; i < ctx->nf; i++) PetscCall(VecScatterDestroy(&ctx->scctx[i]));
  PetscCall(PetscFree(ctx->scctx));
  PetscCall(PetscFree(vctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexSampleGLVisFields_Private(PetscObject oX, PetscInt nf, PetscObject oXfield[], void *vctx)
{
  GLVisViewerCtx *ctx = (GLVisViewerCtx *)vctx;
  PetscInt        f;

  PetscFunctionBegin;
  for (f = 0; f < nf; f++) {
    PetscCall(VecScatterBegin(ctx->scctx[f], (Vec)oX, (Vec)oXfield[f], INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(ctx->scctx[f], (Vec)oX, (Vec)oXfield[f], INSERT_VALUES, SCATTER_FORWARD));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* for FEM, it works for H1 fields only and extracts dofs at cell vertices, discarding any other dof */
PetscErrorCode DMSetUpGLVisViewer_Plex(PetscObject odm, PetscViewer viewer)
{
  DM              dm = (DM)odm;
  Vec             xlocal, xfield, *Ufield;
  PetscDS         ds;
  IS              globalNum, isfield;
  PetscBT         vown;
  char          **fieldname = NULL, **fec_type = NULL;
  const PetscInt *gNum;
  PetscInt       *nlocal, *bs, *idxs, *dims;
  PetscInt        f, maxfields, nfields, c, totc, totdofs, Nv, cum, i;
  PetscInt        dim, cStart, cEnd, vStart, vEnd;
  GLVisViewerCtx *ctx;
  PetscSection    s;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(PetscObjectQuery((PetscObject)dm, "_glvis_plex_gnum", (PetscObject *)&globalNum));
  if (!globalNum) {
    PetscCall(DMPlexCreateCellNumbering_Internal(dm, PETSC_TRUE, &globalNum));
    PetscCall(PetscObjectCompose((PetscObject)dm, "_glvis_plex_gnum", (PetscObject)globalNum));
    PetscCall(PetscObjectDereference((PetscObject)globalNum));
  }
  PetscCall(ISGetIndices(globalNum, &gNum));
  PetscCall(PetscBTCreate(vEnd - vStart, &vown));
  for (c = cStart, totc = 0; c < cEnd; c++) {
    if (gNum[c - cStart] >= 0) {
      PetscInt i, numPoints, *points = NULL;

      totc++;
      PetscCall(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &numPoints, &points));
      for (i = 0; i < numPoints * 2; i += 2) {
        if ((points[i] >= vStart) && (points[i] < vEnd)) PetscCall(PetscBTSet(vown, points[i] - vStart));
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &numPoints, &points));
    }
  }
  for (f = 0, Nv = 0; f < vEnd - vStart; f++)
    if (PetscLikely(PetscBTLookup(vown, f))) Nv++;

  PetscCall(DMCreateLocalVector(dm, &xlocal));
  PetscCall(VecGetLocalSize(xlocal, &totdofs));
  PetscCall(DMGetLocalSection(dm, &s));
  PetscCall(PetscSectionGetNumFields(s, &nfields));
  for (f = 0, maxfields = 0; f < nfields; f++) {
    PetscInt bs;

    PetscCall(PetscSectionGetFieldComponents(s, f, &bs));
    maxfields += bs;
  }
  PetscCall(PetscCalloc7(maxfields, &fieldname, maxfields, &nlocal, maxfields, &bs, maxfields, &dims, maxfields, &fec_type, totdofs, &idxs, maxfields, &Ufield));
  PetscCall(PetscNew(&ctx));
  PetscCall(PetscCalloc1(maxfields, &ctx->scctx));
  PetscCall(DMGetDS(dm, &ds));
  if (ds) {
    for (f = 0; f < nfields; f++) {
      const char *fname;
      char        name[256];
      PetscObject disc;
      size_t      len;

      PetscCall(PetscSectionGetFieldName(s, f, &fname));
      PetscCall(PetscStrlen(fname, &len));
      if (len) {
        PetscCall(PetscStrncpy(name, fname, sizeof(name)));
      } else {
        PetscCall(PetscSNPrintf(name, 256, "Field%" PetscInt_FMT, f));
      }
      PetscCall(PetscDSGetDiscretization(ds, f, &disc));
      if (disc) {
        PetscClassId id;
        PetscInt     Nc;
        char         fec[64];

        PetscCall(PetscObjectGetClassId(disc, &id));
        if (id == PETSCFE_CLASSID) {
          PetscFE            fem = (PetscFE)disc;
          PetscDualSpace     sp;
          PetscDualSpaceType spname;
          PetscInt           order;
          PetscBool          islag, continuous, H1 = PETSC_TRUE;

          PetscCall(PetscFEGetNumComponents(fem, &Nc));
          PetscCall(PetscFEGetDualSpace(fem, &sp));
          PetscCall(PetscDualSpaceGetType(sp, &spname));
          PetscCall(PetscStrcmp(spname, PETSCDUALSPACELAGRANGE, &islag));
          PetscCheck(islag, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unsupported dual space");
          PetscCall(PetscDualSpaceLagrangeGetContinuity(sp, &continuous));
          PetscCall(PetscDualSpaceGetOrder(sp, &order));
          if (continuous && order > 0) { /* no support for high-order viz, still have to figure out the numbering */
            PetscCall(PetscSNPrintf(fec, 64, "FiniteElementCollection: H1_%" PetscInt_FMT "D_P1", dim));
          } else {
            PetscCheck(continuous || !order, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Discontinuous space visualization currently unsupported for order %" PetscInt_FMT, order);
            H1 = PETSC_FALSE;
            PetscCall(PetscSNPrintf(fec, 64, "FiniteElementCollection: L2_%" PetscInt_FMT "D_P%" PetscInt_FMT, dim, order));
          }
          PetscCall(PetscStrallocpy(name, &fieldname[ctx->nf]));
          bs[ctx->nf]   = Nc;
          dims[ctx->nf] = dim;
          if (H1) {
            nlocal[ctx->nf] = Nc * Nv;
            PetscCall(PetscStrallocpy(fec, &fec_type[ctx->nf]));
            PetscCall(VecCreateSeq(PETSC_COMM_SELF, Nv * Nc, &xfield));
            for (i = 0, cum = 0; i < vEnd - vStart; i++) {
              PetscInt j, off;

              if (PetscUnlikely(!PetscBTLookup(vown, i))) continue;
              PetscCall(PetscSectionGetFieldOffset(s, i + vStart, f, &off));
              for (j = 0; j < Nc; j++) idxs[cum++] = off + j;
            }
            PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)xlocal), Nv * Nc, idxs, PETSC_USE_POINTER, &isfield));
          } else {
            nlocal[ctx->nf] = Nc * totc;
            PetscCall(PetscStrallocpy(fec, &fec_type[ctx->nf]));
            PetscCall(VecCreateSeq(PETSC_COMM_SELF, Nc * totc, &xfield));
            for (i = 0, cum = 0; i < cEnd - cStart; i++) {
              PetscInt j, off;

              if (PetscUnlikely(gNum[i] < 0)) continue;
              PetscCall(PetscSectionGetFieldOffset(s, i + cStart, f, &off));
              for (j = 0; j < Nc; j++) idxs[cum++] = off + j;
            }
            PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)xlocal), totc * Nc, idxs, PETSC_USE_POINTER, &isfield));
          }
          PetscCall(VecScatterCreate(xlocal, isfield, xfield, NULL, &ctx->scctx[ctx->nf]));
          PetscCall(VecDestroy(&xfield));
          PetscCall(ISDestroy(&isfield));
          ctx->nf++;
        } else if (id == PETSCFV_CLASSID) {
          PetscInt c;

          PetscCall(PetscFVGetNumComponents((PetscFV)disc, &Nc));
          PetscCall(PetscSNPrintf(fec, 64, "FiniteElementCollection: L2_%" PetscInt_FMT "D_P0", dim));
          for (c = 0; c < Nc; c++) {
            char comp[256];
            PetscCall(PetscSNPrintf(comp, 256, "%s-Comp%" PetscInt_FMT, name, c));
            PetscCall(PetscStrallocpy(comp, &fieldname[ctx->nf]));
            bs[ctx->nf]     = 1; /* Does PetscFV support components with different block size? */
            nlocal[ctx->nf] = totc;
            dims[ctx->nf]   = dim;
            PetscCall(PetscStrallocpy(fec, &fec_type[ctx->nf]));
            PetscCall(VecCreateSeq(PETSC_COMM_SELF, totc, &xfield));
            for (i = 0, cum = 0; i < cEnd - cStart; i++) {
              PetscInt off;

              if (PetscUnlikely(gNum[i]) < 0) continue;
              PetscCall(PetscSectionGetFieldOffset(s, i + cStart, f, &off));
              idxs[cum++] = off + c;
            }
            PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)xlocal), totc, idxs, PETSC_USE_POINTER, &isfield));
            PetscCall(VecScatterCreate(xlocal, isfield, xfield, NULL, &ctx->scctx[ctx->nf]));
            PetscCall(VecDestroy(&xfield));
            PetscCall(ISDestroy(&isfield));
            ctx->nf++;
          }
        } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "Unknown discretization type for field %" PetscInt_FMT, f);
      } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Missing discretization for field %" PetscInt_FMT, f);
    }
  } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Needs a DS attached to the DM");
  PetscCall(PetscBTDestroy(&vown));
  PetscCall(VecDestroy(&xlocal));
  PetscCall(ISRestoreIndices(globalNum, &gNum));

  /* create work vectors */
  for (f = 0; f < ctx->nf; f++) {
    PetscCall(VecCreateMPI(PetscObjectComm((PetscObject)dm), nlocal[f], PETSC_DECIDE, &Ufield[f]));
    PetscCall(PetscObjectSetName((PetscObject)Ufield[f], fieldname[f]));
    PetscCall(VecSetBlockSize(Ufield[f], bs[f]));
    PetscCall(VecSetDM(Ufield[f], dm));
  }

  /* customize the viewer */
  PetscCall(PetscViewerGLVisSetFields(viewer, ctx->nf, (const char **)fec_type, dims, DMPlexSampleGLVisFields_Private, (PetscObject *)Ufield, ctx, DestroyGLVisViewerCtx_Private));
  for (f = 0; f < ctx->nf; f++) {
    PetscCall(PetscFree(fieldname[f]));
    PetscCall(PetscFree(fec_type[f]));
    PetscCall(VecDestroy(&Ufield[f]));
  }
  PetscCall(PetscFree7(fieldname, nlocal, bs, dims, fec_type, idxs, Ufield));
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef enum {
  MFEM_POINT = 0,
  MFEM_SEGMENT,
  MFEM_TRIANGLE,
  MFEM_SQUARE,
  MFEM_TETRAHEDRON,
  MFEM_CUBE,
  MFEM_PRISM,
  MFEM_UNDEF
} MFEM_cid;

MFEM_cid mfem_table_cid[4][7] = {
  {MFEM_POINT, MFEM_UNDEF, MFEM_UNDEF,   MFEM_UNDEF,    MFEM_UNDEF,       MFEM_UNDEF, MFEM_UNDEF},
  {MFEM_POINT, MFEM_UNDEF, MFEM_SEGMENT, MFEM_UNDEF,    MFEM_UNDEF,       MFEM_UNDEF, MFEM_UNDEF},
  {MFEM_POINT, MFEM_UNDEF, MFEM_SEGMENT, MFEM_TRIANGLE, MFEM_SQUARE,      MFEM_UNDEF, MFEM_UNDEF},
  {MFEM_POINT, MFEM_UNDEF, MFEM_SEGMENT, MFEM_UNDEF,    MFEM_TETRAHEDRON, MFEM_PRISM, MFEM_CUBE }
};

MFEM_cid mfem_table_cid_unint[4][9] = {
  {MFEM_POINT, MFEM_UNDEF, MFEM_UNDEF,   MFEM_UNDEF,    MFEM_UNDEF,       MFEM_UNDEF, MFEM_PRISM, MFEM_UNDEF, MFEM_UNDEF},
  {MFEM_POINT, MFEM_UNDEF, MFEM_SEGMENT, MFEM_UNDEF,    MFEM_UNDEF,       MFEM_UNDEF, MFEM_PRISM, MFEM_UNDEF, MFEM_UNDEF},
  {MFEM_POINT, MFEM_UNDEF, MFEM_SEGMENT, MFEM_TRIANGLE, MFEM_SQUARE,      MFEM_UNDEF, MFEM_PRISM, MFEM_UNDEF, MFEM_UNDEF},
  {MFEM_POINT, MFEM_UNDEF, MFEM_SEGMENT, MFEM_UNDEF,    MFEM_TETRAHEDRON, MFEM_UNDEF, MFEM_PRISM, MFEM_UNDEF, MFEM_CUBE }
};

static PetscErrorCode DMPlexGetPointMFEMCellID_Internal(DM dm, DMLabel label, PetscInt minl, PetscInt p, PetscInt *mid, PetscInt *cid)
{
  DMLabel  dlabel;
  PetscInt depth, csize, pdepth, dim;

  PetscFunctionBegin;
  PetscCall(DMPlexGetDepthLabel(dm, &dlabel));
  PetscCall(DMLabelGetValue(dlabel, p, &pdepth));
  PetscCall(DMPlexGetConeSize(dm, p, &csize));
  PetscCall(DMPlexGetDepth(dm, &depth));
  PetscCall(DMGetDimension(dm, &dim));
  if (label) {
    PetscCall(DMLabelGetValue(label, p, mid));
    *mid = *mid - minl + 1; /* MFEM does not like negative markers */
  } else *mid = 1;
  if (depth >= 0 && dim != depth) { /* not interpolated, it assumes cell-vertex mesh */
    PetscCheck(dim >= 0 && dim <= 3, PETSC_COMM_SELF, PETSC_ERR_SUP, "Dimension %" PetscInt_FMT, dim);
    PetscCheck(csize <= 8, PETSC_COMM_SELF, PETSC_ERR_SUP, "Found cone size %" PetscInt_FMT " for point %" PetscInt_FMT, csize, p);
    PetscCheck(depth == 1, PETSC_COMM_SELF, PETSC_ERR_SUP, "Found depth %" PetscInt_FMT " for point %" PetscInt_FMT ". You should interpolate the mesh first", depth, p);
    *cid = mfem_table_cid_unint[dim][csize];
  } else {
    PetscCheck(csize <= 6, PETSC_COMM_SELF, PETSC_ERR_SUP, "Cone size %" PetscInt_FMT " for point %" PetscInt_FMT, csize, p);
    PetscCheck(pdepth >= 0 && pdepth <= 3, PETSC_COMM_SELF, PETSC_ERR_SUP, "Depth %" PetscInt_FMT " for point %" PetscInt_FMT, csize, p);
    *cid = mfem_table_cid[pdepth][csize];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexGetPointMFEMVertexIDs_Internal(DM dm, PetscInt p, PetscSection csec, PetscInt *nv, PetscInt vids[])
{
  PetscInt dim, sdim, dof = 0, off = 0, i, q, vStart, vEnd, numPoints, *points = NULL;

  PetscFunctionBegin;
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(DMGetDimension(dm, &dim));
  sdim = dim;
  if (csec) {
    PetscInt sStart, sEnd;

    PetscCall(DMGetCoordinateDim(dm, &sdim));
    PetscCall(PetscSectionGetChart(csec, &sStart, &sEnd));
    PetscCall(PetscSectionGetOffset(csec, vStart, &off));
    off = off / sdim;
    if (p >= sStart && p < sEnd) PetscCall(PetscSectionGetDof(csec, p, &dof));
  }
  if (!dof) {
    PetscCall(DMPlexGetTransitiveClosure(dm, p, PETSC_TRUE, &numPoints, &points));
    for (i = 0, q = 0; i < numPoints * 2; i += 2)
      if ((points[i] >= vStart) && (points[i] < vEnd)) vids[q++] = points[i] - vStart + off;
    PetscCall(DMPlexRestoreTransitiveClosure(dm, p, PETSC_TRUE, &numPoints, &points));
  } else {
    PetscCall(PetscSectionGetOffset(csec, p, &off));
    PetscCall(PetscSectionGetDof(csec, p, &dof));
    for (q = 0; q < dof / sdim; q++) vids[q] = off / sdim + q;
  }
  *nv = q;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GLVisCreateFE(PetscFE femIn, char name[32], PetscFE *fem, IS *perm)
{
  DM              K;
  PetscSpace      P;
  PetscDualSpace  Q;
  PetscQuadrature q, fq;
  PetscInt        dim, deg, dof;
  DMPolytopeType  ptype;
  PetscBool       isSimplex, isTensor;
  PetscBool       continuity = PETSC_FALSE;
  PetscDTNodeType nodeType   = PETSCDTNODES_GAUSSJACOBI;
  PetscBool       endpoint   = PETSC_TRUE;
  MPI_Comm        comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)femIn, &comm));
  PetscCall(PetscFEGetBasisSpace(femIn, &P));
  PetscCall(PetscFEGetDualSpace(femIn, &Q));
  PetscCall(PetscDualSpaceGetDM(Q, &K));
  PetscCall(DMGetDimension(K, &dim));
  PetscCall(PetscSpaceGetDegree(P, &deg, NULL));
  PetscCall(PetscSpaceGetNumComponents(P, &dof));
  PetscCall(DMPlexGetCellType(K, 0, &ptype));
  switch (ptype) {
  case DM_POLYTOPE_QUADRILATERAL:
  case DM_POLYTOPE_HEXAHEDRON:
    isSimplex = PETSC_FALSE;
    break;
  default:
    isSimplex = PETSC_TRUE;
    break;
  }
  isTensor = isSimplex ? PETSC_FALSE : PETSC_TRUE;
  if (isSimplex) deg = PetscMin(deg, 3); /* Permutation not coded for degree higher than 3 */
  /* Create space */
  PetscCall(PetscSpaceCreate(comm, &P));
  PetscCall(PetscSpaceSetType(P, PETSCSPACEPOLYNOMIAL));
  PetscCall(PetscSpacePolynomialSetTensor(P, isTensor));
  PetscCall(PetscSpaceSetNumComponents(P, dof));
  PetscCall(PetscSpaceSetNumVariables(P, dim));
  PetscCall(PetscSpaceSetDegree(P, deg, PETSC_DETERMINE));
  PetscCall(PetscSpaceSetUp(P));
  /* Create dual space */
  PetscCall(PetscDualSpaceCreate(comm, &Q));
  PetscCall(PetscDualSpaceSetType(Q, PETSCDUALSPACELAGRANGE));
  PetscCall(PetscDualSpaceLagrangeSetTensor(Q, isTensor));
  PetscCall(PetscDualSpaceLagrangeSetContinuity(Q, continuity));
  PetscCall(PetscDualSpaceLagrangeSetNodeType(Q, nodeType, endpoint, 0));
  PetscCall(PetscDualSpaceSetNumComponents(Q, dof));
  PetscCall(PetscDualSpaceSetOrder(Q, deg));
  PetscCall(DMPlexCreateReferenceCell(PETSC_COMM_SELF, DMPolytopeTypeSimpleShape(dim, isSimplex), &K));
  PetscCall(PetscDualSpaceSetDM(Q, K));
  PetscCall(DMDestroy(&K));
  PetscCall(PetscDualSpaceSetUp(Q));
  /* Create quadrature */
  if (isSimplex) {
    PetscCall(PetscDTStroudConicalQuadrature(dim, 1, deg + 1, -1, +1, &q));
    PetscCall(PetscDTStroudConicalQuadrature(dim - 1, 1, deg + 1, -1, +1, &fq));
  } else {
    PetscCall(PetscDTGaussTensorQuadrature(dim, 1, deg + 1, -1, +1, &q));
    PetscCall(PetscDTGaussTensorQuadrature(dim - 1, 1, deg + 1, -1, +1, &fq));
  }
  /* Create finite element */
  PetscCall(PetscFECreate(comm, fem));
  PetscCall(PetscSNPrintf(name, 32, "L2_T1_%" PetscInt_FMT "D_P%" PetscInt_FMT, dim, deg));
  PetscCall(PetscObjectSetName((PetscObject)*fem, name));
  PetscCall(PetscFESetType(*fem, PETSCFEBASIC));
  PetscCall(PetscFESetNumComponents(*fem, dof));
  PetscCall(PetscFESetBasisSpace(*fem, P));
  PetscCall(PetscFESetDualSpace(*fem, Q));
  PetscCall(PetscFESetQuadrature(*fem, q));
  PetscCall(PetscFESetFaceQuadrature(*fem, fq));
  PetscCall(PetscFESetUp(*fem));

  /* Both MFEM and PETSc are lexicographic, but PLEX stores the swapped cone */
  *perm = NULL;
  if (isSimplex && dim == 3) {
    PetscInt celldofs, *pidx;

    PetscCall(PetscDualSpaceGetDimension(Q, &celldofs));
    celldofs /= dof;
    PetscCall(PetscMalloc1(celldofs, &pidx));
    switch (celldofs) {
    case 4:
      pidx[0] = 2;
      pidx[1] = 0;
      pidx[2] = 1;
      pidx[3] = 3;
      break;
    case 10:
      pidx[0] = 5;
      pidx[1] = 3;
      pidx[2] = 0;
      pidx[3] = 4;
      pidx[4] = 1;
      pidx[5] = 2;
      pidx[6] = 8;
      pidx[7] = 6;
      pidx[8] = 7;
      pidx[9] = 9;
      break;
    case 20:
      pidx[0]  = 9;
      pidx[1]  = 7;
      pidx[2]  = 4;
      pidx[3]  = 0;
      pidx[4]  = 8;
      pidx[5]  = 5;
      pidx[6]  = 1;
      pidx[7]  = 6;
      pidx[8]  = 2;
      pidx[9]  = 3;
      pidx[10] = 15;
      pidx[11] = 13;
      pidx[12] = 10;
      pidx[13] = 14;
      pidx[14] = 11;
      pidx[15] = 12;
      pidx[16] = 18;
      pidx[17] = 16;
      pidx[18] = 17;
      pidx[19] = 19;
      break;
    default:
      SETERRQ(comm, PETSC_ERR_SUP, "Unhandled degree,dof pair %" PetscInt_FMT ",%" PetscInt_FMT, deg, celldofs);
      break;
    }
    PetscCall(ISCreateBlock(PETSC_COMM_SELF, dof, celldofs, pidx, PETSC_OWN_POINTER, perm));
  }

  /* Cleanup */
  PetscCall(PetscSpaceDestroy(&P));
  PetscCall(PetscDualSpaceDestroy(&Q));
  PetscCall(PetscQuadratureDestroy(&q));
  PetscCall(PetscQuadratureDestroy(&fq));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   ASCII visualization/dump: full support for simplices and tensor product cells. It supports AMR
   Higher order meshes are also supported
*/
static PetscErrorCode DMPlexView_GLVis_ASCII(DM dm, PetscViewer viewer)
{
  DMLabel            label;
  PetscSection       coordSection, coordSectionCell, parentSection, hoSection = NULL;
  Vec                coordinates, coordinatesCell, hovec;
  const PetscScalar *array;
  PetscInt           bf, p, sdim, dim, depth, novl, minl;
  PetscInt           cStart, cEnd, vStart, vEnd, nvert;
  PetscMPIInt        size;
  PetscBool          localized, isascii;
  PetscBool          enable_mfem, enable_boundary, enable_ncmesh, view_ovl = PETSC_FALSE;
  PetscBT            pown, vown;
  PetscContainer     glvis_container;
  PetscBool          cellvertex = PETSC_FALSE, enabled = PETSC_TRUE;
  PetscBool          enable_emark, enable_bmark;
  const char        *fmt;
  char               emark[64] = "", bmark[64] = "";

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  PetscCheck(isascii, PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Viewer must be of type VIEWERASCII");
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)viewer), &size));
  PetscCheck(size <= 1, PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Use single sequential viewers for parallel visualization");
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetDepth(dm, &depth));

  /* get container: determines if a process visualizes is portion of the data or not */
  PetscCall(PetscObjectQuery((PetscObject)viewer, "_glvis_info_container", (PetscObject *)&glvis_container));
  PetscCheck(glvis_container, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Missing GLVis container");
  {
    PetscViewerGLVisInfo glvis_info;
    PetscCall(PetscContainerGetPointer(glvis_container, (void **)&glvis_info));
    enabled = glvis_info->enabled;
    fmt     = glvis_info->fmt;
  }

  /* Users can attach a coordinate vector to the DM in case they have a higher-order mesh */
  PetscCall(PetscObjectQuery((PetscObject)dm, "_glvis_mesh_coords", (PetscObject *)&hovec));
  PetscCall(PetscObjectReference((PetscObject)hovec));
  if (!hovec) {
    DM           cdm;
    PetscFE      disc;
    PetscClassId classid;

    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMGetField(cdm, 0, NULL, (PetscObject *)&disc));
    PetscCall(PetscObjectGetClassId((PetscObject)disc, &classid));
    if (classid == PETSCFE_CLASSID) {
      DM      hocdm;
      PetscFE hodisc;
      Vec     vec;
      Mat     mat;
      char    name[32], fec_type[64];
      IS      perm = NULL;

      PetscCall(GLVisCreateFE(disc, name, &hodisc, &perm));
      PetscCall(DMClone(cdm, &hocdm));
      PetscCall(DMSetField(hocdm, 0, NULL, (PetscObject)hodisc));
      PetscCall(PetscFEDestroy(&hodisc));
      PetscCall(DMCreateDS(hocdm));

      PetscCall(DMGetCoordinates(dm, &vec));
      PetscCall(DMCreateGlobalVector(hocdm, &hovec));
      PetscCall(DMCreateInterpolation(cdm, hocdm, &mat, NULL));
      PetscCall(MatInterpolate(mat, vec, hovec));
      PetscCall(MatDestroy(&mat));
      PetscCall(DMGetLocalSection(hocdm, &hoSection));
      PetscCall(PetscSectionSetClosurePermutation(hoSection, (PetscObject)hocdm, depth, perm));
      PetscCall(ISDestroy(&perm));
      PetscCall(DMDestroy(&hocdm));
      PetscCall(PetscSNPrintf(fec_type, sizeof(fec_type), "FiniteElementCollection: %s", name));
      PetscCall(PetscObjectSetName((PetscObject)hovec, fec_type));
    }
  }

  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMPlexGetGhostCellStratum(dm, &p, NULL));
  if (p >= 0) cEnd = p;
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(DMGetCoordinatesLocalized(dm, &localized));
  PetscCall(DMGetCoordinateSection(dm, &coordSection));
  PetscCall(DMGetCoordinateDim(dm, &sdim));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCheck(coordinates || hovec, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Missing local coordinates vector");

  /*
     a couple of sections of the mesh specification are disabled
       - boundary: the boundary is not needed for proper mesh visualization unless we want to visualize boundary attributes or we have high-order coordinates in 3D (topologically)
       - vertex_parents: used for non-conforming meshes only when we want to use MFEM as a discretization package
                         and be able to derefine the mesh (MFEM does not currently have to ability to read ncmeshes in parallel)
  */
  enable_boundary = PETSC_FALSE;
  enable_ncmesh   = PETSC_FALSE;
  enable_mfem     = PETSC_FALSE;
  enable_emark    = PETSC_FALSE;
  enable_bmark    = PETSC_FALSE;
  /* I'm tired of problems with negative values in the markers, disable them */
  PetscOptionsBegin(PetscObjectComm((PetscObject)dm), ((PetscObject)dm)->prefix, "GLVis PetscViewer DMPlex Options", "PetscViewer");
  PetscCall(PetscOptionsBool("-viewer_glvis_dm_plex_enable_boundary", "Enable boundary section in mesh representation", NULL, enable_boundary, &enable_boundary, NULL));
  PetscCall(PetscOptionsBool("-viewer_glvis_dm_plex_enable_ncmesh", "Enable vertex_parents section in mesh representation (allows derefinement)", NULL, enable_ncmesh, &enable_ncmesh, NULL));
  PetscCall(PetscOptionsBool("-viewer_glvis_dm_plex_enable_mfem", "Dump a mesh that can be used with MFEM's FiniteElementSpaces", NULL, enable_mfem, &enable_mfem, NULL));
  PetscCall(PetscOptionsBool("-viewer_glvis_dm_plex_overlap", "Include overlap region in local meshes", NULL, view_ovl, &view_ovl, NULL));
  PetscCall(PetscOptionsString("-viewer_glvis_dm_plex_emarker", "String for the material id label", NULL, emark, emark, sizeof(emark), &enable_emark));
  PetscCall(PetscOptionsString("-viewer_glvis_dm_plex_bmarker", "String for the boundary id label", NULL, bmark, bmark, sizeof(bmark), &enable_bmark));
  PetscOptionsEnd();
  if (enable_bmark) enable_boundary = PETSC_TRUE;

  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  PetscCheck(!enable_ncmesh || size == 1, PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Not supported in parallel");
  PetscCheck(!enable_boundary || depth < 0 || dim == depth, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG,
             "Mesh must be interpolated. "
             "Alternatively, run with -viewer_glvis_dm_plex_enable_boundary 0");
  PetscCheck(!enable_ncmesh || depth < 0 || dim == depth, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG,
             "Mesh must be interpolated. "
             "Alternatively, run with -viewer_glvis_dm_plex_enable_ncmesh 0");
  if (depth >= 0 && dim != depth) { /* not interpolated, it assumes cell-vertex mesh */
    PetscCheck(depth == 1, PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported depth %" PetscInt_FMT ". You should interpolate the mesh first", depth);
    cellvertex = PETSC_TRUE;
  }

  /* Identify possible cells in the overlap */
  novl = 0;
  pown = NULL;
  if (size > 1) {
    IS              globalNum = NULL;
    const PetscInt *gNum;
    PetscBool       ovl = PETSC_FALSE;

    PetscCall(PetscObjectQuery((PetscObject)dm, "_glvis_plex_gnum", (PetscObject *)&globalNum));
    if (!globalNum) {
      if (view_ovl) {
        PetscCall(ISCreateStride(PetscObjectComm((PetscObject)dm), cEnd - cStart, 0, 1, &globalNum));
      } else {
        PetscCall(DMPlexCreateCellNumbering_Internal(dm, PETSC_TRUE, &globalNum));
      }
      PetscCall(PetscObjectCompose((PetscObject)dm, "_glvis_plex_gnum", (PetscObject)globalNum));
      PetscCall(PetscObjectDereference((PetscObject)globalNum));
    }
    PetscCall(ISGetIndices(globalNum, &gNum));
    for (p = cStart; p < cEnd; p++) {
      if (gNum[p - cStart] < 0) {
        ovl = PETSC_TRUE;
        novl++;
      }
    }
    if (ovl) {
      /* it may happen that pown get not destroyed, if the user closes the window while this function is running.
         TODO: garbage collector? attach pown to dm?  */
      PetscCall(PetscBTCreate(cEnd - cStart, &pown));
      for (p = cStart; p < cEnd; p++) {
        if (gNum[p - cStart] < 0) continue;
        else PetscCall(PetscBTSet(pown, p - cStart));
      }
    }
    PetscCall(ISRestoreIndices(globalNum, &gNum));
  }

  /* vertex_parents (Non-conforming meshes) */
  parentSection = NULL;
  if (enable_ncmesh) {
    PetscCall(DMPlexGetTree(dm, &parentSection, NULL, NULL, NULL, NULL));
    enable_ncmesh = (PetscBool)(enable_ncmesh && parentSection);
  }
  /* return if this process is disabled */
  if (!enabled) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "MFEM mesh %s\n", enable_ncmesh ? "v1.1" : "v1.0"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "\ndimension\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT "\n", dim));
    PetscCall(PetscViewerASCIIPrintf(viewer, "\nelements\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "0\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "\nboundary\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "0\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "\nvertices\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "0\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT "\n", sdim));
    PetscCall(PetscBTDestroy(&pown));
    PetscCall(VecDestroy(&hovec));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if (enable_mfem) {
    if (localized && !hovec) { /* we need to generate a vector of L2 coordinates, as this is how MFEM handles periodic meshes */
      PetscInt     vpc = 0;
      char         fec[64];
      PetscInt     vids[8] = {0, 1, 2, 3, 4, 5, 6, 7};
      PetscInt     hexv[8] = {0, 1, 3, 2, 4, 5, 7, 6}, tetv[4] = {0, 1, 2, 3};
      PetscInt     quadv[8] = {0, 1, 3, 2}, triv[3] = {0, 1, 2};
      PetscInt    *dof = NULL;
      PetscScalar *array, *ptr;

      PetscCall(PetscSNPrintf(fec, sizeof(fec), "FiniteElementCollection: L2_T1_%" PetscInt_FMT "D_P1", dim));
      if (cEnd - cStart) {
        PetscInt fpc;

        PetscCall(DMPlexGetConeSize(dm, cStart, &fpc));
        switch (dim) {
        case 1:
          vpc = 2;
          dof = hexv;
          break;
        case 2:
          switch (fpc) {
          case 3:
            vpc = 3;
            dof = triv;
            break;
          case 4:
            vpc = 4;
            dof = quadv;
            break;
          default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unhandled case: faces per cell %" PetscInt_FMT, fpc);
          }
          break;
        case 3:
          switch (fpc) {
          case 4: /* TODO: still need to understand L2 ordering for tets */
            vpc = 4;
            dof = tetv;
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unhandled tethraedral case");
          case 6:
            PetscCheck(!cellvertex, PETSC_COMM_SELF, PETSC_ERR_SUP, "Unhandled case: vertices per cell %" PetscInt_FMT, fpc);
            vpc = 8;
            dof = hexv;
            break;
          case 8:
            PetscCheck(cellvertex, PETSC_COMM_SELF, PETSC_ERR_SUP, "Unhandled case: faces per cell %" PetscInt_FMT, fpc);
            vpc = 8;
            dof = hexv;
            break;
          default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unhandled case: faces per cell %" PetscInt_FMT, fpc);
          }
          break;
        default:
          SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Unhandled dim");
        }
        PetscCall(DMPlexReorderCell(dm, cStart, vids));
      }
      PetscCheck(dof, PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "Missing dofs");
      PetscCall(VecCreateSeq(PETSC_COMM_SELF, (cEnd - cStart - novl) * vpc * sdim, &hovec));
      PetscCall(PetscObjectSetName((PetscObject)hovec, fec));
      PetscCall(VecGetArray(hovec, &array));
      ptr = array;
      for (p = cStart; p < cEnd; p++) {
        PetscInt     csize, v, d;
        PetscScalar *vals = NULL;

        if (PetscUnlikely(pown && !PetscBTLookup(pown, p - cStart))) continue;
        PetscCall(DMPlexVecGetClosure(dm, coordSection, coordinates, p, &csize, &vals));
        PetscCheck(csize == vpc * sdim || csize == vpc * sdim * 2, PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported closure size %" PetscInt_FMT " (vpc %" PetscInt_FMT ", sdim %" PetscInt_FMT ")", csize, vpc, sdim);
        for (v = 0; v < vpc; v++) {
          for (d = 0; d < sdim; d++) ptr[sdim * dof[v] + d] = vals[sdim * vids[v] + d];
        }
        ptr += vpc * sdim;
        PetscCall(DMPlexVecRestoreClosure(dm, coordSection, coordinates, p, &csize, &vals));
      }
      PetscCall(VecRestoreArray(hovec, &array));
    }
  }
  /* if we have high-order coordinates in 3D, we need to specify the boundary */
  if (hovec && dim == 3) enable_boundary = PETSC_TRUE;

  /* header */
  PetscCall(PetscViewerASCIIPrintf(viewer, "MFEM mesh %s\n", enable_ncmesh ? "v1.1" : "v1.0"));

  /* topological dimension */
  PetscCall(PetscViewerASCIIPrintf(viewer, "\ndimension\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT "\n", dim));

  /* elements */
  minl  = 1;
  label = NULL;
  if (enable_emark) {
    PetscInt lminl = PETSC_MAX_INT;

    PetscCall(DMGetLabel(dm, emark, &label));
    if (label) {
      IS       vals;
      PetscInt ldef;

      PetscCall(DMLabelGetDefaultValue(label, &ldef));
      PetscCall(DMLabelGetValueIS(label, &vals));
      PetscCall(ISGetMinMax(vals, &lminl, NULL));
      PetscCall(ISDestroy(&vals));
      lminl = PetscMin(ldef, lminl);
    }
    PetscCall(MPIU_Allreduce(&lminl, &minl, 1, MPIU_INT, MPI_MIN, PetscObjectComm((PetscObject)dm)));
    if (minl == PETSC_MAX_INT) minl = 1;
  }
  PetscCall(PetscViewerASCIIPrintf(viewer, "\nelements\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT "\n", cEnd - cStart - novl));
  for (p = cStart; p < cEnd; p++) {
    PetscInt vids[8];
    PetscInt i, nv = 0, cid = -1, mid = 1;

    if (PetscUnlikely(pown && !PetscBTLookup(pown, p - cStart))) continue;
    PetscCall(DMPlexGetPointMFEMCellID_Internal(dm, label, minl, p, &mid, &cid));
    PetscCall(DMPlexGetPointMFEMVertexIDs_Internal(dm, p, (localized && !hovec) ? coordSection : NULL, &nv, vids));
    PetscCall(DMPlexReorderCell(dm, p, vids));
    PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT " %" PetscInt_FMT, mid, cid));
    for (i = 0; i < nv; i++) PetscCall(PetscViewerASCIIPrintf(viewer, " %" PetscInt_FMT, vids[i]));
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
  }

  /* boundary */
  PetscCall(PetscViewerASCIIPrintf(viewer, "\nboundary\n"));
  if (!enable_boundary) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "0\n"));
  } else {
    DMLabel  perLabel;
    PetscBT  bfaces;
    PetscInt fStart, fEnd, *fcells;

    PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
    PetscCall(PetscBTCreate(fEnd - fStart, &bfaces));
    PetscCall(DMPlexGetMaxSizes(dm, NULL, &p));
    PetscCall(PetscMalloc1(p, &fcells));
    PetscCall(DMGetLabel(dm, "glvis_periodic_cut", &perLabel));
    if (!perLabel && localized) { /* this periodic cut can be moved up to DMPlex setup */
      PetscCall(DMCreateLabel(dm, "glvis_periodic_cut"));
      PetscCall(DMGetLabel(dm, "glvis_periodic_cut", &perLabel));
      PetscCall(DMLabelSetDefaultValue(perLabel, 1));
      PetscCall(DMGetCellCoordinateSection(dm, &coordSectionCell));
      PetscCall(DMGetCellCoordinatesLocal(dm, &coordinatesCell));
      for (p = cStart; p < cEnd; p++) {
        DMPolytopeType cellType;
        PetscInt       dof;

        PetscCall(DMPlexGetCellType(dm, p, &cellType));
        PetscCall(PetscSectionGetDof(coordSectionCell, p, &dof));
        if (dof) {
          PetscInt     uvpc, v, csize, csizeCell, cellClosureSize, *cellClosure = NULL, *vidxs = NULL;
          PetscScalar *vals = NULL, *valsCell = NULL;

          uvpc = DMPolytopeTypeGetNumVertices(cellType);
          PetscCheck(dof % sdim == 0, PETSC_COMM_SELF, PETSC_ERR_USER, "Incompatible number of cell dofs %" PetscInt_FMT " and space dimension %" PetscInt_FMT, dof, sdim);
          PetscCall(DMPlexVecGetClosure(dm, coordSection, coordinates, p, &csize, &vals));
          PetscCall(DMPlexVecGetClosure(dm, coordSectionCell, coordinatesCell, p, &csizeCell, &valsCell));
          PetscCheck(csize == csizeCell, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Cell %" PetscInt_FMT " has invalid localized coordinates", p);
          PetscCall(DMPlexGetTransitiveClosure(dm, p, PETSC_TRUE, &cellClosureSize, &cellClosure));
          for (v = 0; v < cellClosureSize; v++)
            if (cellClosure[2 * v] >= vStart && cellClosure[2 * v] < vEnd) {
              vidxs = cellClosure + 2 * v;
              break;
            }
          PetscCheck(vidxs, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing vertices");
          for (v = 0; v < uvpc; v++) {
            PetscInt s;

            for (s = 0; s < sdim; s++) {
              if (PetscAbsScalar(vals[v * sdim + s] - valsCell[v * sdim + s]) > PETSC_MACHINE_EPSILON) PetscCall(DMLabelSetValue(perLabel, vidxs[2 * v], 2));
            }
          }
          PetscCall(DMPlexRestoreTransitiveClosure(dm, p, PETSC_TRUE, &cellClosureSize, &cellClosure));
          PetscCall(DMPlexVecRestoreClosure(dm, coordSection, coordinates, p, &csize, &vals));
          PetscCall(DMPlexVecRestoreClosure(dm, coordSectionCell, coordinatesCell, p, &csizeCell, &valsCell));
        }
      }
      if (dim > 1) {
        PetscInt eEnd, eStart;

        PetscCall(DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd));
        for (p = eStart; p < eEnd; p++) {
          const PetscInt *cone;
          PetscInt        coneSize, i;
          PetscBool       ispe = PETSC_TRUE;

          PetscCall(DMPlexGetCone(dm, p, &cone));
          PetscCall(DMPlexGetConeSize(dm, p, &coneSize));
          for (i = 0; i < coneSize; i++) {
            PetscInt v;

            PetscCall(DMLabelGetValue(perLabel, cone[i], &v));
            ispe = (PetscBool)(ispe && (v == 2));
          }
          if (ispe && coneSize) {
            PetscInt        ch, numChildren;
            const PetscInt *children;

            PetscCall(DMLabelSetValue(perLabel, p, 2));
            PetscCall(DMPlexGetTreeChildren(dm, p, &numChildren, &children));
            for (ch = 0; ch < numChildren; ch++) PetscCall(DMLabelSetValue(perLabel, children[ch], 2));
          }
        }
        if (dim > 2) {
          for (p = fStart; p < fEnd; p++) {
            const PetscInt *cone;
            PetscInt        coneSize, i;
            PetscBool       ispe = PETSC_TRUE;

            PetscCall(DMPlexGetCone(dm, p, &cone));
            PetscCall(DMPlexGetConeSize(dm, p, &coneSize));
            for (i = 0; i < coneSize; i++) {
              PetscInt v;

              PetscCall(DMLabelGetValue(perLabel, cone[i], &v));
              ispe = (PetscBool)(ispe && (v == 2));
            }
            if (ispe && coneSize) {
              PetscInt        ch, numChildren;
              const PetscInt *children;

              PetscCall(DMLabelSetValue(perLabel, p, 2));
              PetscCall(DMPlexGetTreeChildren(dm, p, &numChildren, &children));
              for (ch = 0; ch < numChildren; ch++) PetscCall(DMLabelSetValue(perLabel, children[ch], 2));
            }
          }
        }
      }
    }
    for (p = fStart; p < fEnd; p++) {
      const PetscInt *support;
      PetscInt        supportSize;
      PetscBool       isbf = PETSC_FALSE;

      PetscCall(DMPlexGetSupportSize(dm, p, &supportSize));
      if (pown) {
        PetscBool has_owned = PETSC_FALSE, has_ghost = PETSC_FALSE;
        PetscInt  i;

        PetscCall(DMPlexGetSupport(dm, p, &support));
        for (i = 0; i < supportSize; i++) {
          if (PetscLikely(PetscBTLookup(pown, support[i] - cStart))) has_owned = PETSC_TRUE;
          else has_ghost = PETSC_TRUE;
        }
        isbf = (PetscBool)((supportSize == 1 && has_owned) || (supportSize > 1 && has_owned && has_ghost));
      } else {
        isbf = (PetscBool)(supportSize == 1);
      }
      if (!isbf && perLabel) {
        const PetscInt *cone;
        PetscInt        coneSize, i;

        PetscCall(DMPlexGetCone(dm, p, &cone));
        PetscCall(DMPlexGetConeSize(dm, p, &coneSize));
        isbf = PETSC_TRUE;
        for (i = 0; i < coneSize; i++) {
          PetscInt v, d;

          PetscCall(DMLabelGetValue(perLabel, cone[i], &v));
          PetscCall(DMLabelGetDefaultValue(perLabel, &d));
          isbf = (PetscBool)(isbf && v != d);
        }
      }
      if (isbf) PetscCall(PetscBTSet(bfaces, p - fStart));
    }
    /* count boundary faces */
    for (p = fStart, bf = 0; p < fEnd; p++) {
      if (PetscUnlikely(PetscBTLookup(bfaces, p - fStart))) {
        const PetscInt *support;
        PetscInt        supportSize, c;

        PetscCall(DMPlexGetSupportSize(dm, p, &supportSize));
        PetscCall(DMPlexGetSupport(dm, p, &support));
        for (c = 0; c < supportSize; c++) {
          const PetscInt *cone;
          PetscInt        cell, cl, coneSize;

          cell = support[c];
          if (pown && PetscUnlikely(!PetscBTLookup(pown, cell - cStart))) continue;
          PetscCall(DMPlexGetCone(dm, cell, &cone));
          PetscCall(DMPlexGetConeSize(dm, cell, &coneSize));
          for (cl = 0; cl < coneSize; cl++) {
            if (cone[cl] == p) {
              bf += 1;
              break;
            }
          }
        }
      }
    }
    minl  = 1;
    label = NULL;
    if (enable_bmark) {
      PetscInt lminl = PETSC_MAX_INT;

      PetscCall(DMGetLabel(dm, bmark, &label));
      if (label) {
        IS       vals;
        PetscInt ldef;

        PetscCall(DMLabelGetDefaultValue(label, &ldef));
        PetscCall(DMLabelGetValueIS(label, &vals));
        PetscCall(ISGetMinMax(vals, &lminl, NULL));
        PetscCall(ISDestroy(&vals));
        lminl = PetscMin(ldef, lminl);
      }
      PetscCall(MPIU_Allreduce(&lminl, &minl, 1, MPIU_INT, MPI_MIN, PetscObjectComm((PetscObject)dm)));
      if (minl == PETSC_MAX_INT) minl = 1;
    }
    PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT "\n", bf));
    for (p = fStart; p < fEnd; p++) {
      if (PetscUnlikely(PetscBTLookup(bfaces, p - fStart))) {
        const PetscInt *support;
        PetscInt        supportSize, c, nc = 0;

        PetscCall(DMPlexGetSupportSize(dm, p, &supportSize));
        PetscCall(DMPlexGetSupport(dm, p, &support));
        if (pown) {
          for (c = 0; c < supportSize; c++) {
            if (PetscLikely(PetscBTLookup(pown, support[c] - cStart))) fcells[nc++] = support[c];
          }
        } else
          for (c = 0; c < supportSize; c++) fcells[nc++] = support[c];
        for (c = 0; c < nc; c++) {
          const DMPolytopeType *faceTypes;
          DMPolytopeType        cellType;
          const PetscInt       *faceSizes, *cone;
          PetscInt              vids[8], *faces, st, i, coneSize, cell, cl, nv, cid = -1, mid = -1;

          cell = fcells[c];
          PetscCall(DMPlexGetCone(dm, cell, &cone));
          PetscCall(DMPlexGetConeSize(dm, cell, &coneSize));
          for (cl = 0; cl < coneSize; cl++)
            if (cone[cl] == p) break;
          if (cl == coneSize) continue;

          /* face material id and type */
          PetscCall(DMPlexGetPointMFEMCellID_Internal(dm, label, minl, p, &mid, &cid));
          PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT " %" PetscInt_FMT, mid, cid));
          /* vertex ids */
          PetscCall(DMPlexGetCellType(dm, cell, &cellType));
          PetscCall(DMPlexGetPointMFEMVertexIDs_Internal(dm, cell, (localized && !hovec) ? coordSection : NULL, &nv, vids));
          PetscCall(DMPlexGetRawFaces_Internal(dm, cellType, vids, NULL, &faceTypes, &faceSizes, (const PetscInt **)&faces));
          st = 0;
          for (i = 0; i < cl; i++) st += faceSizes[i];
          PetscCall(DMPlexInvertCell(faceTypes[cl], faces + st));
          for (i = 0; i < faceSizes[cl]; i++) PetscCall(PetscViewerASCIIPrintf(viewer, " %" PetscInt_FMT, faces[st + i]));
          PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
          PetscCall(DMPlexRestoreRawFaces_Internal(dm, cellType, vids, NULL, &faceTypes, &faceSizes, (const PetscInt **)&faces));
          bf -= 1;
        }
      }
    }
    PetscCheck(!bf, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Remaining boundary faces %" PetscInt_FMT, bf);
    PetscCall(PetscBTDestroy(&bfaces));
    PetscCall(PetscFree(fcells));
  }

  /* mark owned vertices */
  vown = NULL;
  if (pown) {
    PetscCall(PetscBTCreate(vEnd - vStart, &vown));
    for (p = cStart; p < cEnd; p++) {
      PetscInt i, closureSize, *closure = NULL;

      if (PetscUnlikely(!PetscBTLookup(pown, p - cStart))) continue;
      PetscCall(DMPlexGetTransitiveClosure(dm, p, PETSC_TRUE, &closureSize, &closure));
      for (i = 0; i < closureSize; i++) {
        const PetscInt pp = closure[2 * i];

        if (pp >= vStart && pp < vEnd) PetscCall(PetscBTSet(vown, pp - vStart));
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, p, PETSC_TRUE, &closureSize, &closure));
    }
  }

  if (parentSection) {
    PetscInt vp, gvp;

    for (vp = 0, p = vStart; p < vEnd; p++) {
      DMLabel  dlabel;
      PetscInt parent, depth;

      if (PetscUnlikely(vown && !PetscBTLookup(vown, p - vStart))) continue;
      PetscCall(DMPlexGetDepthLabel(dm, &dlabel));
      PetscCall(DMLabelGetValue(dlabel, p, &depth));
      PetscCall(DMPlexGetTreeParent(dm, p, &parent, NULL));
      if (parent != p) vp++;
    }
    PetscCall(MPIU_Allreduce(&vp, &gvp, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)dm)));
    if (gvp) {
      PetscInt   maxsupp;
      PetscBool *skip = NULL;

      PetscCall(PetscViewerASCIIPrintf(viewer, "\nvertex_parents\n"));
      PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT "\n", vp));
      PetscCall(DMPlexGetMaxSizes(dm, NULL, &maxsupp));
      PetscCall(PetscMalloc1(maxsupp, &skip));
      for (p = vStart; p < vEnd; p++) {
        DMLabel  dlabel;
        PetscInt parent;

        if (PetscUnlikely(vown && !PetscBTLookup(vown, p - vStart))) continue;
        PetscCall(DMPlexGetDepthLabel(dm, &dlabel));
        PetscCall(DMPlexGetTreeParent(dm, p, &parent, NULL));
        if (parent != p) {
          PetscInt        vids[8] = {-1, -1, -1, -1, -1, -1, -1, -1}; /* silent overzealous clang static analyzer */
          PetscInt        i, nv, ssize, n, numChildren, depth = -1;
          const PetscInt *children;

          PetscCall(DMPlexGetConeSize(dm, parent, &ssize));
          switch (ssize) {
          case 2: /* edge */
            nv = 0;
            PetscCall(DMPlexGetPointMFEMVertexIDs_Internal(dm, parent, localized ? coordSection : NULL, &nv, vids));
            PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT, p - vStart));
            for (i = 0; i < nv; i++) PetscCall(PetscViewerASCIIPrintf(viewer, " %" PetscInt_FMT, vids[i]));
            PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
            vp--;
            break;
          case 4: /* face */
            PetscCall(DMPlexGetTreeChildren(dm, parent, &numChildren, &children));
            for (n = 0; n < numChildren; n++) {
              PetscCall(DMLabelGetValue(dlabel, children[n], &depth));
              if (!depth) {
                const PetscInt *hvsupp, *hesupp, *cone;
                PetscInt        hvsuppSize, hesuppSize, coneSize;
                PetscInt        hv = children[n], he = -1, f;

                PetscCall(PetscArrayzero(skip, maxsupp));
                PetscCall(DMPlexGetSupportSize(dm, hv, &hvsuppSize));
                PetscCall(DMPlexGetSupport(dm, hv, &hvsupp));
                for (i = 0; i < hvsuppSize; i++) {
                  PetscInt ep;
                  PetscCall(DMPlexGetTreeParent(dm, hvsupp[i], &ep, NULL));
                  if (ep != hvsupp[i]) {
                    he = hvsupp[i];
                  } else {
                    skip[i] = PETSC_TRUE;
                  }
                }
                PetscCheck(he != -1, PETSC_COMM_SELF, PETSC_ERR_SUP, "Vertex %" PetscInt_FMT " support size %" PetscInt_FMT ": hanging edge not found", hv, hvsuppSize);
                PetscCall(DMPlexGetCone(dm, he, &cone));
                vids[0] = (cone[0] == hv) ? cone[1] : cone[0];
                PetscCall(DMPlexGetSupportSize(dm, he, &hesuppSize));
                PetscCall(DMPlexGetSupport(dm, he, &hesupp));
                for (f = 0; f < hesuppSize; f++) {
                  PetscInt j;

                  PetscCall(DMPlexGetCone(dm, hesupp[f], &cone));
                  PetscCall(DMPlexGetConeSize(dm, hesupp[f], &coneSize));
                  for (j = 0; j < coneSize; j++) {
                    PetscInt k;
                    for (k = 0; k < hvsuppSize; k++) {
                      if (hvsupp[k] == cone[j]) {
                        skip[k] = PETSC_TRUE;
                        break;
                      }
                    }
                  }
                }
                for (i = 0; i < hvsuppSize; i++) {
                  if (!skip[i]) {
                    PetscCall(DMPlexGetCone(dm, hvsupp[i], &cone));
                    vids[1] = (cone[0] == hv) ? cone[1] : cone[0];
                  }
                }
                PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT, hv - vStart));
                for (i = 0; i < 2; i++) PetscCall(PetscViewerASCIIPrintf(viewer, " %" PetscInt_FMT, vids[i] - vStart));
                PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
                vp--;
              }
            }
            break;
          default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Don't know how to deal with support size %" PetscInt_FMT, ssize);
          }
        }
      }
      PetscCall(PetscFree(skip));
    }
    PetscCheck(!vp, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected %" PetscInt_FMT " hanging vertices", vp);
  }
  PetscCall(PetscBTDestroy(&vown));

  /* vertices */
  if (hovec) { /* higher-order meshes */
    const char *fec;
    PetscInt    i, n, s;
    PetscCall(PetscViewerASCIIPrintf(viewer, "\nvertices\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT "\n", vEnd - vStart));
    PetscCall(PetscViewerASCIIPrintf(viewer, "nodes\n"));
    PetscCall(PetscObjectGetName((PetscObject)hovec, &fec));
    PetscCall(PetscViewerASCIIPrintf(viewer, "FiniteElementSpace\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "%s\n", fec));
    PetscCall(PetscViewerASCIIPrintf(viewer, "VDim: %" PetscInt_FMT "\n", sdim));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Ordering: 1\n\n")); /*Ordering::byVDIM*/
    if (hoSection) {
      DM cdm;

      PetscCall(VecGetDM(hovec, &cdm));
      for (p = cStart; p < cEnd; p++) {
        PetscScalar *vals = NULL;
        PetscInt     csize;

        if (PetscUnlikely(pown && !PetscBTLookup(pown, p - cStart))) continue;
        PetscCall(DMPlexVecGetClosure(cdm, hoSection, hovec, p, &csize, &vals));
        PetscCheck(csize % sdim == 0, PETSC_COMM_SELF, PETSC_ERR_USER, "Size of closure %" PetscInt_FMT " incompatible with space dimension %" PetscInt_FMT, csize, sdim);
        for (i = 0; i < csize / sdim; i++) {
          for (s = 0; s < sdim; s++) PetscCall(PetscViewerASCIIPrintf(viewer, fmt, (double)PetscRealPart(vals[i * sdim + s])));
          PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
        }
        PetscCall(DMPlexVecRestoreClosure(cdm, hoSection, hovec, p, &csize, &vals));
      }
    } else {
      PetscCall(VecGetArrayRead(hovec, &array));
      PetscCall(VecGetLocalSize(hovec, &n));
      PetscCheck(n % sdim == 0, PETSC_COMM_SELF, PETSC_ERR_USER, "Size of local coordinate vector %" PetscInt_FMT " incompatible with space dimension %" PetscInt_FMT, n, sdim);
      for (i = 0; i < n / sdim; i++) {
        for (s = 0; s < sdim; s++) PetscCall(PetscViewerASCIIPrintf(viewer, fmt, (double)PetscRealPart(array[i * sdim + s])));
        PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
      }
      PetscCall(VecRestoreArrayRead(hovec, &array));
    }
  } else {
    PetscCall(VecGetLocalSize(coordinates, &nvert));
    PetscCall(PetscViewerASCIIPrintf(viewer, "\nvertices\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT "\n", nvert / sdim));
    PetscCall(PetscViewerASCIIPrintf(viewer, "%" PetscInt_FMT "\n", sdim));
    PetscCall(VecGetArrayRead(coordinates, &array));
    for (p = 0; p < nvert / sdim; p++) {
      PetscInt s;
      for (s = 0; s < sdim; s++) {
        PetscReal v = PetscRealPart(array[p * sdim + s]);

        PetscCall(PetscViewerASCIIPrintf(viewer, fmt, PetscIsInfOrNanReal(v) ? 0.0 : (double)v));
      }
      PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
    }
    PetscCall(VecRestoreArrayRead(coordinates, &array));
  }
  PetscCall(PetscBTDestroy(&pown));
  PetscCall(VecDestroy(&hovec));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexView_GLVis(DM dm, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscCall(DMView_GLVis(dm, viewer, DMPlexView_GLVis_ASCII));
  PetscFunctionReturn(PETSC_SUCCESS);
}
