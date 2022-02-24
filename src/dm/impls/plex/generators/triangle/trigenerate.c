#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

#if !defined(ANSI_DECLARATORS)
#define ANSI_DECLARATORS
#endif
#include <triangle.h>

static PetscErrorCode InitInput_Triangle(struct triangulateio *inputCtx)
{
  PetscFunctionBegin;
  inputCtx->numberofpoints             = 0;
  inputCtx->numberofpointattributes    = 0;
  inputCtx->pointlist                  = NULL;
  inputCtx->pointattributelist         = NULL;
  inputCtx->pointmarkerlist            = NULL;
  inputCtx->numberofsegments           = 0;
  inputCtx->segmentlist                = NULL;
  inputCtx->segmentmarkerlist          = NULL;
  inputCtx->numberoftriangleattributes = 0;
  inputCtx->trianglelist               = NULL;
  inputCtx->numberofholes              = 0;
  inputCtx->holelist                   = NULL;
  inputCtx->numberofregions            = 0;
  inputCtx->regionlist                 = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode InitOutput_Triangle(struct triangulateio *outputCtx)
{
  PetscFunctionBegin;
  outputCtx->numberofpoints        = 0;
  outputCtx->pointlist             = NULL;
  outputCtx->pointattributelist    = NULL;
  outputCtx->pointmarkerlist       = NULL;
  outputCtx->numberoftriangles     = 0;
  outputCtx->trianglelist          = NULL;
  outputCtx->triangleattributelist = NULL;
  outputCtx->neighborlist          = NULL;
  outputCtx->segmentlist           = NULL;
  outputCtx->segmentmarkerlist     = NULL;
  outputCtx->numberofedges         = 0;
  outputCtx->edgelist              = NULL;
  outputCtx->edgemarkerlist        = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode FiniOutput_Triangle(struct triangulateio *outputCtx)
{
  PetscFunctionBegin;
  free(outputCtx->pointlist);
  free(outputCtx->pointmarkerlist);
  free(outputCtx->segmentlist);
  free(outputCtx->segmentmarkerlist);
  free(outputCtx->edgelist);
  free(outputCtx->edgemarkerlist);
  free(outputCtx->trianglelist);
  free(outputCtx->neighborlist);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMPlexGenerate_Triangle(DM boundary, PetscBool interpolate, DM *dm)
{
  MPI_Comm             comm;
  DM_Plex             *mesh             = (DM_Plex *) boundary->data;
  PetscInt             dim              = 2;
  const PetscBool      createConvexHull = PETSC_FALSE;
  const PetscBool      constrained      = PETSC_FALSE;
  const char          *labelName        = "marker";
  const char          *labelName2       = "Face Sets";
  struct triangulateio in;
  struct triangulateio out;
  DMLabel              label, label2;
  PetscInt             vStart, vEnd, v, eStart, eEnd, e;
  PetscMPIInt          rank;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)boundary,&comm));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(InitInput_Triangle(&in));
  CHKERRQ(InitOutput_Triangle(&out));
  CHKERRQ(DMPlexGetDepthStratum(boundary, 0, &vStart, &vEnd));
  CHKERRQ(DMGetLabel(boundary, labelName,  &label));
  CHKERRQ(DMGetLabel(boundary, labelName2, &label2));

  in.numberofpoints = vEnd - vStart;
  if (in.numberofpoints > 0) {
    PetscSection coordSection;
    Vec          coordinates;
    PetscScalar *array;

    CHKERRQ(PetscMalloc1(in.numberofpoints*dim, &in.pointlist));
    CHKERRQ(PetscMalloc1(in.numberofpoints, &in.pointmarkerlist));
    CHKERRQ(DMGetCoordinatesLocal(boundary, &coordinates));
    CHKERRQ(DMGetCoordinateSection(boundary, &coordSection));
    CHKERRQ(VecGetArray(coordinates, &array));
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       val, off, d;

      CHKERRQ(PetscSectionGetOffset(coordSection, v, &off));
      for (d = 0; d < dim; ++d) {
        in.pointlist[idx*dim + d] = PetscRealPart(array[off+d]);
      }
      if (label) {
        CHKERRQ(DMLabelGetValue(label, v, &val));
        in.pointmarkerlist[idx] = val;
      }
    }
    CHKERRQ(VecRestoreArray(coordinates, &array));
  }
  CHKERRQ(DMPlexGetHeightStratum(boundary, 0, &eStart, &eEnd));
  in.numberofsegments = eEnd - eStart;
  if (in.numberofsegments > 0) {
    CHKERRQ(PetscMalloc1(in.numberofsegments*2, &in.segmentlist));
    CHKERRQ(PetscMalloc1(in.numberofsegments, &in.segmentmarkerlist));
    for (e = eStart; e < eEnd; ++e) {
      const PetscInt  idx = e - eStart;
      const PetscInt *cone;
      PetscInt        val;

      CHKERRQ(DMPlexGetCone(boundary, e, &cone));

      in.segmentlist[idx*2+0] = cone[0] - vStart;
      in.segmentlist[idx*2+1] = cone[1] - vStart;

      if (label) {
        CHKERRQ(DMLabelGetValue(label, e, &val));
        in.segmentmarkerlist[idx] = val;
      }
    }
  }
#if 0 /* Do not currently support holes */
  PetscReal *holeCoords;
  PetscInt   h, d;

  CHKERRQ(DMPlexGetHoles(boundary, &in.numberofholes, &holeCords));
  if (in.numberofholes > 0) {
    CHKERRQ(PetscMalloc1(in.numberofholes*dim, &in.holelist));
    for (h = 0; h < in.numberofholes; ++h) {
      for (d = 0; d < dim; ++d) {
        in.holelist[h*dim+d] = holeCoords[h*dim+d];
      }
    }
  }
#endif
  if (rank == 0) {
    char args[32];

    /* Take away 'Q' for verbose output */
    CHKERRQ(PetscStrcpy(args, "pqezQ"));
    if (createConvexHull)   CHKERRQ(PetscStrcat(args, "c"));
    if (constrained)        CHKERRQ(PetscStrcpy(args, "zepDQ"));
    if (mesh->triangleOpts) {triangulate(mesh->triangleOpts, &in, &out, NULL);}
    else                    {triangulate(args, &in, &out, NULL);}
  }
  CHKERRQ(PetscFree(in.pointlist));
  CHKERRQ(PetscFree(in.pointmarkerlist));
  CHKERRQ(PetscFree(in.segmentlist));
  CHKERRQ(PetscFree(in.segmentmarkerlist));
  CHKERRQ(PetscFree(in.holelist));

  {
    DMLabel          glabel      = NULL;
    DMLabel          glabel2     = NULL;
    const PetscInt   numCorners  = 3;
    const PetscInt   numCells    = out.numberoftriangles;
    const PetscInt   numVertices = out.numberofpoints;
    PetscInt         *cells;
    PetscReal        *meshCoords;

    if (sizeof (PetscReal) == sizeof (out.pointlist[0])) {
      meshCoords = (PetscReal *) out.pointlist;
    } else {
      PetscInt i;

      CHKERRQ(PetscMalloc1(dim * numVertices,&meshCoords));
      for (i = 0; i < dim * numVertices; i++) {
        meshCoords[i] = (PetscReal) out.pointlist[i];
      }
    }
    if (sizeof (PetscInt) == sizeof (out.trianglelist[0])) {
      cells = (PetscInt *) out.trianglelist;
    } else {
      PetscInt i;

      CHKERRQ(PetscMalloc1(numCells * numCorners, &cells));
      for (i = 0; i < numCells * numCorners; i++) {
        cells[i] = (PetscInt) out.trianglelist[i];
      }
    }
    CHKERRQ(DMPlexCreateFromCellListPetsc(comm, dim, numCells, numVertices, numCorners, interpolate, cells, dim, meshCoords, dm));
    if (sizeof (PetscReal) != sizeof (out.pointlist[0])) {
      CHKERRQ(PetscFree(meshCoords));
    }
    if (sizeof (PetscInt) != sizeof (out.trianglelist[0])) {
      CHKERRQ(PetscFree(cells));
    }
    if (label)  {
      CHKERRQ(DMCreateLabel(*dm, labelName));
      CHKERRQ(DMGetLabel(*dm, labelName, &glabel));
    }
    if (label2) {
      CHKERRQ(DMCreateLabel(*dm, labelName2));
      CHKERRQ(DMGetLabel(*dm, labelName2, &glabel2));
    }
    /* Set labels */
    for (v = 0; v < numVertices; ++v) {
      if (out.pointmarkerlist[v]) {
        if (glabel) CHKERRQ(DMLabelSetValue(glabel, v+numCells, out.pointmarkerlist[v]));
      }
    }
    if (interpolate) {
      for (e = 0; e < out.numberofedges; e++) {
        if (out.edgemarkerlist[e]) {
          const PetscInt  vertices[2] = {out.edgelist[e*2+0]+numCells, out.edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          CHKERRQ(DMPlexGetJoin(*dm, 2, vertices, &numEdges, &edges));
          PetscCheckFalse(numEdges != 1,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          if (glabel)  CHKERRQ(DMLabelSetValue(glabel,  edges[0], out.edgemarkerlist[e]));
          if (glabel2) CHKERRQ(DMLabelSetValue(glabel2, edges[0], out.edgemarkerlist[e]));
          CHKERRQ(DMPlexRestoreJoin(*dm, 2, vertices, &numEdges, &edges));
        }
      }
    }
    CHKERRQ(DMPlexSetRefinementUniform(*dm, PETSC_FALSE));
  }
#if 0 /* Do not currently support holes */
  CHKERRQ(DMPlexCopyHoles(*dm, boundary));
#endif
  CHKERRQ(FiniOutput_Triangle(&out));
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMPlexRefine_Triangle(DM dm, PetscReal *inmaxVolumes, DM *dmRefined)
{
  MPI_Comm             comm;
  PetscInt             dim       = 2;
  const char          *labelName = "marker";
  struct triangulateio in;
  struct triangulateio out;
  DMLabel              label;
  PetscInt             vStart, vEnd, v, gcStart, cStart, cEnd, c, depth, depthGlobal;
  PetscMPIInt          rank;
  double               *maxVolumes;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetComm((PetscObject)dm,&comm));
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(InitInput_Triangle(&in));
  CHKERRQ(InitOutput_Triangle(&out));
  CHKERRQ(DMPlexGetDepth(dm, &depth));
  CHKERRMPI(MPIU_Allreduce(&depth, &depthGlobal, 1, MPIU_INT, MPI_MAX, comm));
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  CHKERRQ(DMGetLabel(dm, labelName, &label));

  in.numberofpoints = vEnd - vStart;
  if (in.numberofpoints > 0) {
    PetscSection coordSection;
    Vec          coordinates;
    PetscScalar *array;

    CHKERRQ(PetscMalloc1(in.numberofpoints*dim, &in.pointlist));
    CHKERRQ(PetscMalloc1(in.numberofpoints, &in.pointmarkerlist));
    CHKERRQ(DMGetCoordinatesLocal(dm, &coordinates));
    CHKERRQ(DMGetCoordinateSection(dm, &coordSection));
    CHKERRQ(VecGetArray(coordinates, &array));
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d, val;

      CHKERRQ(PetscSectionGetOffset(coordSection, v, &off));
      for (d = 0; d < dim; ++d) {
        in.pointlist[idx*dim + d] = PetscRealPart(array[off+d]);
      }
      if (label) {
        CHKERRQ(DMLabelGetValue(label, v, &val));
        in.pointmarkerlist[idx] = val;
      }
    }
    CHKERRQ(VecRestoreArray(coordinates, &array));
  }
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  CHKERRQ(DMPlexGetGhostCellStratum(dm, &gcStart, NULL));
  if (gcStart >= 0) cEnd = gcStart;

  in.numberofcorners   = 3;
  in.numberoftriangles = cEnd - cStart;

#if !defined(PETSC_USE_REAL_DOUBLE)
  CHKERRQ(PetscMalloc1(cEnd - cStart,&maxVolumes));
  for (c = 0; c < cEnd-cStart; ++c) maxVolumes[c] = (double)inmaxVolumes[c];
#else
  maxVolumes = inmaxVolumes;
#endif

  in.trianglearealist  = (double*) maxVolumes;
  if (in.numberoftriangles > 0) {
    CHKERRQ(PetscMalloc1(in.numberoftriangles*in.numberofcorners, &in.trianglelist));
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt idx      = c - cStart;
      PetscInt      *closure = NULL;
      PetscInt       closureSize;

      CHKERRQ(DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
      PetscCheckFalse((closureSize != 4) && (closureSize != 7),comm, PETSC_ERR_ARG_WRONG, "Mesh has cell which is not a triangle, %D vertices in closure", closureSize);
      for (v = 0; v < 3; ++v) {
        in.trianglelist[idx*in.numberofcorners + v] = closure[(v+closureSize-3)*2] - vStart;
      }
      CHKERRQ(DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure));
    }
  }
  /* TODO: Segment markers are missing on input */
#if 0 /* Do not currently support holes */
  PetscReal *holeCoords;
  PetscInt   h, d;

  CHKERRQ(DMPlexGetHoles(boundary, &in.numberofholes, &holeCords));
  if (in.numberofholes > 0) {
    CHKERRQ(PetscMalloc1(in.numberofholes*dim, &in.holelist));
    for (h = 0; h < in.numberofholes; ++h) {
      for (d = 0; d < dim; ++d) {
        in.holelist[h*dim+d] = holeCoords[h*dim+d];
      }
    }
  }
#endif
  if (rank == 0) {
    char args[32];

    /* Take away 'Q' for verbose output */
    CHKERRQ(PetscStrcpy(args, "pqezQra"));
    triangulate(args, &in, &out, NULL);
  }
  CHKERRQ(PetscFree(in.pointlist));
  CHKERRQ(PetscFree(in.pointmarkerlist));
  CHKERRQ(PetscFree(in.segmentlist));
  CHKERRQ(PetscFree(in.segmentmarkerlist));
  CHKERRQ(PetscFree(in.trianglelist));

  {
    DMLabel          rlabel      = NULL;
    const PetscInt   numCorners  = 3;
    const PetscInt   numCells    = out.numberoftriangles;
    const PetscInt   numVertices = out.numberofpoints;
    PetscInt         *cells;
    PetscReal        *meshCoords;
    PetscBool        interpolate = depthGlobal > 1 ? PETSC_TRUE : PETSC_FALSE;

    if (sizeof (PetscReal) == sizeof (out.pointlist[0])) {
      meshCoords = (PetscReal *) out.pointlist;
    } else {
      PetscInt i;

      CHKERRQ(PetscMalloc1(dim * numVertices,&meshCoords));
      for (i = 0; i < dim * numVertices; i++) {
        meshCoords[i] = (PetscReal) out.pointlist[i];
      }
    }
    if (sizeof (PetscInt) == sizeof (out.trianglelist[0])) {
      cells = (PetscInt *) out.trianglelist;
    } else {
      PetscInt i;

      CHKERRQ(PetscMalloc1(numCells * numCorners, &cells));
      for (i = 0; i < numCells * numCorners; i++) {
        cells[i] = (PetscInt) out.trianglelist[i];
      }
    }

    CHKERRQ(DMPlexCreateFromCellListPetsc(comm, dim, numCells, numVertices, numCorners, interpolate, cells, dim, meshCoords, dmRefined));
    if (label) {
      CHKERRQ(DMCreateLabel(*dmRefined, labelName));
      CHKERRQ(DMGetLabel(*dmRefined, labelName, &rlabel));
    }
    if (sizeof (PetscReal) != sizeof (out.pointlist[0])) {
      CHKERRQ(PetscFree(meshCoords));
    }
    if (sizeof (PetscInt) != sizeof (out.trianglelist[0])) {
      CHKERRQ(PetscFree(cells));
    }
    /* Set labels */
    for (v = 0; v < numVertices; ++v) {
      if (out.pointmarkerlist[v]) {
        if (rlabel) CHKERRQ(DMLabelSetValue(rlabel, v+numCells, out.pointmarkerlist[v]));
      }
    }
    if (interpolate) {
      PetscInt e;

      for (e = 0; e < out.numberofedges; e++) {
        if (out.edgemarkerlist[e]) {
          const PetscInt  vertices[2] = {out.edgelist[e*2+0]+numCells, out.edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          CHKERRQ(DMPlexGetJoin(*dmRefined, 2, vertices, &numEdges, &edges));
          PetscCheckFalse(numEdges != 1,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          if (rlabel) CHKERRQ(DMLabelSetValue(rlabel, edges[0], out.edgemarkerlist[e]));
          CHKERRQ(DMPlexRestoreJoin(*dmRefined, 2, vertices, &numEdges, &edges));
        }
      }
    }
    CHKERRQ(DMPlexSetRefinementUniform(*dmRefined, PETSC_FALSE));
  }
#if 0 /* Do not currently support holes */
  CHKERRQ(DMPlexCopyHoles(*dm, boundary));
#endif
  CHKERRQ(FiniOutput_Triangle(&out));
#if !defined(PETSC_USE_REAL_DOUBLE)
  CHKERRQ(PetscFree(maxVolumes));
#endif
  PetscFunctionReturn(0);
}
