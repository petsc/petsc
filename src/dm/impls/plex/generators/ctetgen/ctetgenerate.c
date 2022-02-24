#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

#ifdef PETSC_HAVE_EGADS
#include <egads.h>
#endif

#include <ctetgen.h>

/* This is to fix the tetrahedron orientation from TetGen */
static PetscErrorCode DMPlexInvertCells_CTetgen(PetscInt numCells, PetscInt numCorners, PetscInt cells[])
{
  PetscInt bound = numCells*numCorners, coff;

  PetscFunctionBegin;
#define SWAP(a,b) do { PetscInt tmp = (a); (a) = (b); (b) = tmp; } while (0)
  for (coff = 0; coff < bound; coff += numCorners) SWAP(cells[coff],cells[coff+1]);
#undef SWAP
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMPlexGenerate_CTetgen(DM boundary, PetscBool interpolate, DM *dm)
{
  MPI_Comm               comm;
  const PetscInt         dim = 3;
  PLC                   *in, *out;
  DMUniversalLabel       universal;
  PetscInt               vStart, vEnd, v, eStart, eEnd, e, fStart, fEnd, f, verbose = 0;
  DMPlexInterpolatedFlag isInterpolated;
  PetscMPIInt            rank;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsGetInt(NULL,((PetscObject) boundary)->prefix, "-ctetgen_verbose", &verbose, NULL));
  CHKERRQ(PetscObjectGetComm((PetscObject)boundary,&comm));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(DMPlexIsInterpolatedCollective(boundary, &isInterpolated));
  CHKERRQ(DMUniversalLabelCreate(boundary, &universal));

  CHKERRQ(PLCCreate(&in));
  CHKERRQ(PLCCreate(&out));

  CHKERRQ(DMPlexGetDepthStratum(boundary, 0, &vStart, &vEnd));
  in->numberofpoints = vEnd - vStart;
  if (in->numberofpoints > 0) {
    PetscSection       coordSection;
    Vec                coordinates;
    const PetscScalar *array;

    CHKERRQ(PetscMalloc1(in->numberofpoints*dim, &in->pointlist));
    CHKERRQ(PetscMalloc1(in->numberofpoints,     &in->pointmarkerlist));
    CHKERRQ(DMGetCoordinatesLocal(boundary, &coordinates));
    CHKERRQ(DMGetCoordinateSection(boundary, &coordSection));
    CHKERRQ(VecGetArrayRead(coordinates, &array));
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d, m;

      CHKERRQ(PetscSectionGetOffset(coordSection, v, &off));
      for (d = 0; d < dim; ++d) in->pointlist[idx*dim + d] = PetscRealPart(array[off+d]);
      CHKERRQ(DMLabelGetValue(universal->label, v, &m));
      in->pointmarkerlist[idx] = (int) m;
    }
    CHKERRQ(VecRestoreArrayRead(coordinates, &array));
  }

  CHKERRQ(DMPlexGetHeightStratum(boundary, 1, &eStart, &eEnd));
  in->numberofedges = eEnd - eStart;
  if (isInterpolated == DMPLEX_INTERPOLATED_FULL && in->numberofedges > 0) {
    CHKERRQ(PetscMalloc1(in->numberofedges*2, &in->edgelist));
    CHKERRQ(PetscMalloc1(in->numberofedges,   &in->edgemarkerlist));
    for (e = eStart; e < eEnd; ++e) {
      const PetscInt  idx = e - eStart;
      const PetscInt *cone;
      PetscInt        coneSize, val;

      CHKERRQ(DMPlexGetConeSize(boundary, e, &coneSize));
      CHKERRQ(DMPlexGetCone(boundary, e, &cone));
      in->edgelist[idx*2]     = cone[0] - vStart;
      in->edgelist[idx*2 + 1] = cone[1] - vStart;

      CHKERRQ(DMLabelGetValue(universal->label, e, &val));
      in->edgemarkerlist[idx] = (int) val;
    }
  }

  CHKERRQ(DMPlexGetHeightStratum(boundary, 0, &fStart, &fEnd));
  in->numberoffacets = fEnd - fStart;
  if (in->numberoffacets > 0) {
    CHKERRQ(PetscMalloc1(in->numberoffacets, &in->facetlist));
    CHKERRQ(PetscMalloc1(in->numberoffacets, &in->facetmarkerlist));
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt idx    = f - fStart;
      PetscInt      *points = NULL, numPoints, p, numVertices = 0, v, m = -1;
      polygon       *poly;

      in->facetlist[idx].numberofpolygons = 1;
      CHKERRQ(PetscMalloc1(in->facetlist[idx].numberofpolygons, &in->facetlist[idx].polygonlist));
      in->facetlist[idx].numberofholes    = 0;
      in->facetlist[idx].holelist         = NULL;

      CHKERRQ(DMPlexGetTransitiveClosure(boundary, f, PETSC_TRUE, &numPoints, &points));
      for (p = 0; p < numPoints*2; p += 2) {
        const PetscInt point = points[p];
        if ((point >= vStart) && (point < vEnd)) points[numVertices++] = point;
      }

      poly                   = in->facetlist[idx].polygonlist;
      poly->numberofvertices = numVertices;
      CHKERRQ(PetscMalloc1(poly->numberofvertices, &poly->vertexlist));
      for (v = 0; v < numVertices; ++v) {
        const PetscInt vIdx = points[v] - vStart;
        poly->vertexlist[v] = vIdx;
      }
      CHKERRQ(DMLabelGetValue(universal->label, f, &m));
      in->facetmarkerlist[idx] = (int) m;
      CHKERRQ(DMPlexRestoreTransitiveClosure(boundary, f, PETSC_TRUE, &numPoints, &points));
    }
  }
  if (rank == 0) {
    TetGenOpts t;

    CHKERRQ(TetGenOptsInitialize(&t));
    t.in        = boundary; /* Should go away */
    t.plc       = 1;
    t.quality   = 1;
    t.edgesout  = 1;
    t.zeroindex = 1;
    t.quiet     = 1;
    t.verbose   = verbose;
#if 0
#ifdef PETSC_HAVE_EGADS
    /* Need to add in more TetGen code */
    t.nobisect  = 1; /* Add Y to preserve Surface Mesh for EGADS */
#endif
#endif

    CHKERRQ(TetGenCheckOpts(&t));
    CHKERRQ(TetGenTetrahedralize(&t, in, out));
  }
  {
    const PetscInt numCorners  = 4;
    const PetscInt numCells    = out->numberoftetrahedra;
    const PetscInt numVertices = out->numberofpoints;
    PetscReal      *meshCoords = NULL;
    PetscInt       *cells      = NULL;

    if (sizeof (PetscReal) == sizeof (out->pointlist[0])) {
      meshCoords = (PetscReal *) out->pointlist;
    } else {
      PetscInt i;

      CHKERRQ(PetscMalloc1(dim * numVertices, &meshCoords));
      for (i = 0; i < dim * numVertices; ++i) meshCoords[i] = (PetscReal) out->pointlist[i];
    }
    if (sizeof (PetscInt) == sizeof (out->tetrahedronlist[0])) {
      cells = (PetscInt *) out->tetrahedronlist;
    } else {
      PetscInt i;

      CHKERRQ(PetscMalloc1(numCells * numCorners, &cells));
      for (i = 0; i < numCells * numCorners; i++) cells[i] = (PetscInt) out->tetrahedronlist[i];
    }

    CHKERRQ(DMPlexInvertCells_CTetgen(numCells, numCorners, cells));
    CHKERRQ(DMPlexCreateFromCellListPetsc(comm, dim, numCells, numVertices, numCorners, interpolate, cells, dim, meshCoords, dm));
    if (sizeof (PetscReal) != sizeof (out->pointlist[0])) {
      CHKERRQ(PetscFree(meshCoords));
    }
    if (sizeof (PetscInt) != sizeof (out->tetrahedronlist[0])) {
      CHKERRQ(PetscFree(cells));
    }

    /* Set labels */
    CHKERRQ(DMUniversalLabelCreateLabels(universal, PETSC_TRUE, *dm));
    for (v = 0; v < numVertices; ++v) {
      if (out->pointmarkerlist[v]) {
        CHKERRQ(DMUniversalLabelSetLabelValue(universal, *dm, PETSC_TRUE, v+numCells, out->pointmarkerlist[v]));
      }
    }
    if (interpolate) {
      PetscInt e;

      for (e = 0; e < out->numberofedges; e++) {
        if (out->edgemarkerlist[e]) {
          const PetscInt  vertices[2] = {out->edgelist[e*2+0]+numCells, out->edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          CHKERRQ(DMPlexGetJoin(*dm, 2, vertices, &numEdges, &edges));
          PetscCheck(numEdges == 1,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          CHKERRQ(DMUniversalLabelSetLabelValue(universal, *dm, PETSC_TRUE, edges[0], out->edgemarkerlist[e]));
          CHKERRQ(DMPlexRestoreJoin(*dm, 2, vertices, &numEdges, &edges));
        }
      }
      for (f = 0; f < out->numberoftrifaces; f++) {
        if (out->trifacemarkerlist[f]) {
          const PetscInt  vertices[3] = {out->trifacelist[f*3+0]+numCells, out->trifacelist[f*3+1]+numCells, out->trifacelist[f*3+2]+numCells};
          const PetscInt *faces;
          PetscInt        numFaces;

          CHKERRQ(DMPlexGetFullJoin(*dm, 3, vertices, &numFaces, &faces));
          PetscCheck(numFaces == 1,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Three vertices must cover only one face, not %D", numFaces);
          CHKERRQ(DMUniversalLabelSetLabelValue(universal, *dm, PETSC_TRUE, faces[0], out->trifacemarkerlist[f]));
          CHKERRQ(DMPlexRestoreJoin(*dm, 3, vertices, &numFaces, &faces));
        }
      }
    }

#ifdef PETSC_HAVE_EGADS
    {
      DMLabel        bodyLabel;
      PetscContainer modelObj;
      PetscInt       cStart, cEnd, c, eStart, eEnd, fStart, fEnd;
      ego           *bodies;
      ego            model, geom;
      int            Nb, oclass, mtype, *senses;

      /* Get Attached EGADS Model from Original DMPlex */
      CHKERRQ(PetscObjectQuery((PetscObject) boundary, "EGADS Model", (PetscObject *) &modelObj));
      if (modelObj) {
        CHKERRQ(PetscContainerGetPointer(modelObj, (void **) &model));
        CHKERRQ(EG_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));
        /* Transfer EGADS Model to Volumetric Mesh */
        CHKERRQ(PetscObjectCompose((PetscObject) *dm, "EGADS Model", (PetscObject) modelObj));

        /* Set Cell Labels */
        CHKERRQ(DMGetLabel(*dm, "EGADS Body ID", &bodyLabel));
        CHKERRQ(DMPlexGetHeightStratum(*dm, 0, &cStart, &cEnd));
        CHKERRQ(DMPlexGetHeightStratum(*dm, 1, &fStart, &fEnd));
        CHKERRQ(DMPlexGetDepthStratum(*dm, 1, &eStart, &eEnd));

        for (c = cStart; c < cEnd; ++c) {
          PetscReal centroid[3] = {0., 0., 0.};
          PetscInt  b;

          /* Deterimine what body the cell's centroid is located in */
          if (!interpolate) {
            PetscSection   coordSection;
            Vec            coordinates;
            PetscScalar   *coords = NULL;
            PetscInt       coordSize, s, d;

            CHKERRQ(DMGetCoordinatesLocal(*dm, &coordinates));
            CHKERRQ(DMGetCoordinateSection(*dm, &coordSection));
            CHKERRQ(DMPlexVecGetClosure(*dm, coordSection, coordinates, c, &coordSize, &coords));
            for (s = 0; s < coordSize; ++s) for (d = 0; d < dim; ++d) centroid[d] += coords[s*dim+d];
            CHKERRQ(DMPlexVecRestoreClosure(*dm, coordSection, coordinates, c, &coordSize, &coords));
          } else {
            CHKERRQ(DMPlexComputeCellGeometryFVM(*dm, c, NULL, centroid, NULL));
          }
          for (b = 0; b < Nb; ++b) {
            if (EG_inTopology(bodies[b], centroid) == EGADS_SUCCESS) break;
          }
          if (b < Nb) {
            PetscInt   cval = b, eVal, fVal;
            PetscInt *closure = NULL, Ncl, cl;

            CHKERRQ(DMLabelSetValue(bodyLabel, c, cval));
            CHKERRQ(DMPlexGetTransitiveClosure(*dm, c, PETSC_TRUE, &Ncl, &closure));
            for (cl = 0; cl < Ncl; ++cl) {
              const PetscInt p = closure[cl*2];

              if (p >= eStart && p < eEnd) {
                CHKERRQ(DMLabelGetValue(bodyLabel, p, &eVal));
                if (eVal < 0) CHKERRQ(DMLabelSetValue(bodyLabel, p, cval));
              }
              if (p >= fStart && p < fEnd) {
                CHKERRQ(DMLabelGetValue(bodyLabel, p, &fVal));
                if (fVal < 0) CHKERRQ(DMLabelSetValue(bodyLabel, p, cval));
              }
            }
            CHKERRQ(DMPlexRestoreTransitiveClosure(*dm, c, PETSC_TRUE, &Ncl, &closure));
          }
        }
      }
    }
#endif
    CHKERRQ(DMPlexSetRefinementUniform(*dm, PETSC_FALSE));
  }

  CHKERRQ(DMUniversalLabelDestroy(&universal));
  CHKERRQ(PLCDestroy(&in));
  CHKERRQ(PLCDestroy(&out));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMPlexRefine_CTetgen(DM dm, PetscReal *maxVolumes, DM *dmRefined)
{
  MPI_Comm               comm;
  const PetscInt         dim = 3;
  PLC                   *in, *out;
  DMUniversalLabel       universal;
  PetscInt               vStart, vEnd, v, eStart, eEnd, e, fStart, fEnd, f, cStart, cEnd, c, verbose = 0;
  DMPlexInterpolatedFlag isInterpolated;
  PetscMPIInt            rank;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsGetInt(NULL,((PetscObject) dm)->prefix, "-ctetgen_verbose", &verbose, NULL));
  CHKERRQ(PetscObjectGetComm((PetscObject)dm,&comm));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(DMPlexIsInterpolatedCollective(dm, &isInterpolated));
  CHKERRQ(DMUniversalLabelCreate(dm, &universal));

  CHKERRQ(PLCCreate(&in));
  CHKERRQ(PLCCreate(&out));

  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  in->numberofpoints = vEnd - vStart;
  if (in->numberofpoints > 0) {
    PetscSection coordSection;
    Vec          coordinates;
    PetscScalar *array;

    CHKERRQ(PetscMalloc1(in->numberofpoints*dim, &in->pointlist));
    CHKERRQ(PetscMalloc1(in->numberofpoints, &in->pointmarkerlist));
    CHKERRQ(DMGetCoordinatesLocal(dm, &coordinates));
    CHKERRQ(DMGetCoordinateSection(dm, &coordSection));
    CHKERRQ(VecGetArray(coordinates, &array));
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d, m;

      CHKERRQ(PetscSectionGetOffset(coordSection, v, &off));
      for (d = 0; d < dim; ++d) in->pointlist[idx*dim + d] = PetscRealPart(array[off+d]);
      CHKERRQ(DMLabelGetValue(universal->label, v, &m));
      in->pointmarkerlist[idx] = (int) m;
    }
    CHKERRQ(VecRestoreArray(coordinates, &array));
  }

  CHKERRQ(DMPlexGetDepthStratum(dm, 1, &eStart, &eEnd));
  in->numberofedges = eEnd - eStart;
  if (isInterpolated == DMPLEX_INTERPOLATED_FULL && in->numberofedges > 0) {
    CHKERRQ(PetscMalloc1(in->numberofedges * 2, &in->edgelist));
    CHKERRQ(PetscMalloc1(in->numberofedges,     &in->edgemarkerlist));
    for (e = eStart; e < eEnd; ++e) {
      const PetscInt  idx = e - eStart;
      const PetscInt *cone;
      PetscInt        coneSize, val;

      CHKERRQ(DMPlexGetConeSize(dm, e, &coneSize));
      CHKERRQ(DMPlexGetCone(dm, e, &cone));
      in->edgelist[idx*2]     = cone[0] - vStart;
      in->edgelist[idx*2 + 1] = cone[1] - vStart;

      CHKERRQ(DMLabelGetValue(universal->label, e, &val));
      in->edgemarkerlist[idx] = (int) val;
    }
  }

  CHKERRQ(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  in->numberoftrifaces = 0;
  for (f = fStart; f < fEnd; ++f) {
    PetscInt supportSize;

    CHKERRQ(DMPlexGetSupportSize(dm, f, &supportSize));
    if (supportSize == 1) ++in->numberoftrifaces;
  }
  if (isInterpolated == DMPLEX_INTERPOLATED_FULL && in->numberoftrifaces > 0) {
    PetscInt tf = 0;

    CHKERRQ(PetscMalloc1(in->numberoftrifaces*3, &in->trifacelist));
    CHKERRQ(PetscMalloc1(in->numberoftrifaces, &in->trifacemarkerlist));
    for (f = fStart; f < fEnd; ++f) {
      PetscInt *points = NULL;
      PetscInt supportSize, numPoints, p, Nv = 0, val;

      CHKERRQ(DMPlexGetSupportSize(dm, f, &supportSize));
      if (supportSize != 1) continue;
      CHKERRQ(DMPlexGetTransitiveClosure(dm, f, PETSC_TRUE, &numPoints, &points));
      for (p = 0; p < numPoints*2; p += 2) {
        const PetscInt point = points[p];
        if ((point >= vStart) && (point < vEnd)) in->trifacelist[tf*3 + Nv++] = point - vStart;
      }
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm, f, PETSC_TRUE, &numPoints, &points));
      PetscCheck(Nv == 3,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Face %D has %D vertices, not 3", f, Nv);
      CHKERRQ(DMLabelGetValue(universal->label, f, &val));
      in->trifacemarkerlist[tf] = (int) val;
      ++tf;
    }
  }

  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  in->numberofcorners       = 4;
  in->numberoftetrahedra    = cEnd - cStart;
  in->tetrahedronvolumelist = maxVolumes;
  if (in->numberoftetrahedra > 0) {
    CHKERRQ(PetscMalloc1(in->numberoftetrahedra*in->numberofcorners, &in->tetrahedronlist));
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt idx     = c - cStart;
      PetscInt      *closure = NULL;
      PetscInt       closureSize;

      CHKERRQ(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
      PetscCheck((closureSize == 5) || (closureSize == 15),comm, PETSC_ERR_ARG_WRONG, "Mesh has cell which is not a tetrahedron, %" PetscInt_FMT " vertices in closure", closureSize);
      for (v = 0; v < 4; ++v) in->tetrahedronlist[idx*in->numberofcorners + v] = closure[(v+closureSize-4)*2] - vStart;
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
    }
  }

  if (rank == 0) {
    TetGenOpts t;

    CHKERRQ(TetGenOptsInitialize(&t));

    t.in        = dm; /* Should go away */
    t.refine    = 1;
    t.varvolume = 1;
    t.quality   = 1;
    t.edgesout  = 1;
    t.zeroindex = 1;
    t.quiet     = 1;
    t.verbose   = verbose; /* Change this */

    CHKERRQ(TetGenCheckOpts(&t));
    CHKERRQ(TetGenTetrahedralize(&t, in, out));
  }

  in->tetrahedronvolumelist = NULL;
  {
    const PetscInt numCorners  = 4;
    const PetscInt numCells    = out->numberoftetrahedra;
    const PetscInt numVertices = out->numberofpoints;
    PetscReal      *meshCoords = NULL;
    PetscInt       *cells      = NULL;
    PetscBool      interpolate = isInterpolated == DMPLEX_INTERPOLATED_FULL ? PETSC_TRUE : PETSC_FALSE;

    if (sizeof (PetscReal) == sizeof (out->pointlist[0])) {
      meshCoords = (PetscReal *) out->pointlist;
    } else {
      PetscInt i;

      CHKERRQ(PetscMalloc1(dim * numVertices, &meshCoords));
      for (i = 0; i < dim * numVertices; ++i) meshCoords[i] = (PetscReal) out->pointlist[i];
    }
    if (sizeof (PetscInt) == sizeof (out->tetrahedronlist[0])) {
      cells = (PetscInt *) out->tetrahedronlist;
    } else {
      PetscInt i;

      CHKERRQ(PetscMalloc1(numCells * numCorners, &cells));
      for (i = 0; i < numCells * numCorners; ++i) cells[i] = (PetscInt) out->tetrahedronlist[i];
    }

    CHKERRQ(DMPlexInvertCells_CTetgen(numCells, numCorners, cells));
    CHKERRQ(DMPlexCreateFromCellListPetsc(comm, dim, numCells, numVertices, numCorners, interpolate, cells, dim, meshCoords, dmRefined));
    if (sizeof (PetscReal) != sizeof (out->pointlist[0])) CHKERRQ(PetscFree(meshCoords));
    if (sizeof (PetscInt) != sizeof (out->tetrahedronlist[0])) CHKERRQ(PetscFree(cells));

    /* Set labels */
    CHKERRQ(DMUniversalLabelCreateLabels(universal, PETSC_TRUE, *dmRefined));
    for (v = 0; v < numVertices; ++v) {
      if (out->pointmarkerlist[v]) {
        CHKERRQ(DMUniversalLabelSetLabelValue(universal, *dmRefined, PETSC_TRUE, v+numCells, out->pointmarkerlist[v]));
      }
    }
    if (interpolate) {
      PetscInt e, f;

      for (e = 0; e < out->numberofedges; e++) {
        if (out->edgemarkerlist[e]) {
          const PetscInt  vertices[2] = {out->edgelist[e*2+0]+numCells, out->edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          CHKERRQ(DMPlexGetJoin(*dmRefined, 2, vertices, &numEdges, &edges));
          PetscCheck(numEdges == 1,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          CHKERRQ(DMUniversalLabelSetLabelValue(universal, *dmRefined, PETSC_TRUE, edges[0], out->edgemarkerlist[e]));
          CHKERRQ(DMPlexRestoreJoin(*dmRefined, 2, vertices, &numEdges, &edges));
        }
      }
      for (f = 0; f < out->numberoftrifaces; f++) {
        if (out->trifacemarkerlist[f]) {
          const PetscInt  vertices[3] = {out->trifacelist[f*3+0]+numCells, out->trifacelist[f*3+1]+numCells, out->trifacelist[f*3+2]+numCells};
          const PetscInt *faces;
          PetscInt        numFaces;

          CHKERRQ(DMPlexGetFullJoin(*dmRefined, 3, vertices, &numFaces, &faces));
          PetscCheck(numFaces == 1,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Three vertices must cover only one face, not %D", numFaces);
          CHKERRQ(DMUniversalLabelSetLabelValue(universal, *dmRefined, PETSC_TRUE, faces[0], out->trifacemarkerlist[f]));
          CHKERRQ(DMPlexRestoreJoin(*dmRefined, 3, vertices, &numFaces, &faces));
        }
      }
    }

#ifdef PETSC_HAVE_EGADS
    {
      DMLabel        bodyLabel;
      PetscContainer modelObj;
      PetscInt       cStart, cEnd, c, eStart, eEnd, fStart, fEnd;
      ego           *bodies;
      ego            model, geom;
      int            Nb, oclass, mtype, *senses;

      /* Get Attached EGADS Model from Original DMPlex */
      CHKERRQ(PetscObjectQuery((PetscObject) dm, "EGADS Model", (PetscObject *) &modelObj));
      if (modelObj) {
        CHKERRQ(PetscContainerGetPointer(modelObj, (void **) &model));
        CHKERRQ(EG_getTopology(model, &geom, &oclass, &mtype, NULL, &Nb, &bodies, &senses));
        /* Transfer EGADS Model to Volumetric Mesh */
        CHKERRQ(PetscObjectCompose((PetscObject) *dmRefined, "EGADS Model", (PetscObject) modelObj));

        /* Set Cell Labels */
        CHKERRQ(DMGetLabel(*dmRefined, "EGADS Body ID", &bodyLabel));
        CHKERRQ(DMPlexGetHeightStratum(*dmRefined, 0, &cStart, &cEnd));
        CHKERRQ(DMPlexGetHeightStratum(*dmRefined, 1, &fStart, &fEnd));
        CHKERRQ(DMPlexGetDepthStratum(*dmRefined, 1, &eStart, &eEnd));

        for (c = cStart; c < cEnd; ++c) {
          PetscReal centroid[3] = {0., 0., 0.};
          PetscInt  b;

          /* Deterimine what body the cell's centroid is located in */
          if (!interpolate) {
            PetscSection   coordSection;
            Vec            coordinates;
            PetscScalar   *coords = NULL;
            PetscInt       coordSize, s, d;

            CHKERRQ(DMGetCoordinatesLocal(*dmRefined, &coordinates));
            CHKERRQ(DMGetCoordinateSection(*dmRefined, &coordSection));
            CHKERRQ(DMPlexVecGetClosure(*dmRefined, coordSection, coordinates, c, &coordSize, &coords));
            for (s = 0; s < coordSize; ++s) for (d = 0; d < dim; ++d) centroid[d] += coords[s*dim+d];
            CHKERRQ(DMPlexVecRestoreClosure(*dmRefined, coordSection, coordinates, c, &coordSize, &coords));
          } else {
            CHKERRQ(DMPlexComputeCellGeometryFVM(*dmRefined, c, NULL, centroid, NULL));
          }
          for (b = 0; b < Nb; ++b) {
            if (EG_inTopology(bodies[b], centroid) == EGADS_SUCCESS) break;
          }
          if (b < Nb) {
            PetscInt   cval = b, eVal, fVal;
            PetscInt *closure = NULL, Ncl, cl;

            CHKERRQ(DMLabelSetValue(bodyLabel, c, cval));
            CHKERRQ(DMPlexGetTransitiveClosure(*dmRefined, c, PETSC_TRUE, &Ncl, &closure));
            for (cl = 0; cl < Ncl; cl += 2) {
              const PetscInt p = closure[cl];

              if (p >= eStart && p < eEnd) {
                CHKERRQ(DMLabelGetValue(bodyLabel, p, &eVal));
                if (eVal < 0) CHKERRQ(DMLabelSetValue(bodyLabel, p, cval));
              }
              if (p >= fStart && p < fEnd) {
                CHKERRQ(DMLabelGetValue(bodyLabel, p, &fVal));
                if (fVal < 0) CHKERRQ(DMLabelSetValue(bodyLabel, p, cval));
              }
            }
            CHKERRQ(DMPlexRestoreTransitiveClosure(*dmRefined, c, PETSC_TRUE, &Ncl, &closure));
          }
        }
      }
    }
#endif
    CHKERRQ(DMPlexSetRefinementUniform(*dmRefined, PETSC_FALSE));
  }
  CHKERRQ(DMUniversalLabelDestroy(&universal));
  CHKERRQ(PLCDestroy(&in));
  CHKERRQ(PLCDestroy(&out));
  PetscFunctionReturn(0);
}
