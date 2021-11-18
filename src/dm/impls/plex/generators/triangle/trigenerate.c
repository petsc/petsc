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
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)boundary,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = InitInput_Triangle(&in);CHKERRQ(ierr);
  ierr = InitOutput_Triangle(&out);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(boundary, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMGetLabel(boundary, labelName,  &label);CHKERRQ(ierr);
  ierr = DMGetLabel(boundary, labelName2, &label2);CHKERRQ(ierr);

  in.numberofpoints = vEnd - vStart;
  if (in.numberofpoints > 0) {
    PetscSection coordSection;
    Vec          coordinates;
    PetscScalar *array;

    ierr = PetscMalloc1(in.numberofpoints*dim, &in.pointlist);CHKERRQ(ierr);
    ierr = PetscMalloc1(in.numberofpoints, &in.pointmarkerlist);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(boundary, &coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(boundary, &coordSection);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &array);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       val, off, d;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        in.pointlist[idx*dim + d] = PetscRealPart(array[off+d]);
      }
      if (label) {
        ierr = DMLabelGetValue(label, v, &val);CHKERRQ(ierr);
        in.pointmarkerlist[idx] = val;
      }
    }
    ierr = VecRestoreArray(coordinates, &array);CHKERRQ(ierr);
  }
  ierr  = DMPlexGetHeightStratum(boundary, 0, &eStart, &eEnd);CHKERRQ(ierr);
  in.numberofsegments = eEnd - eStart;
  if (in.numberofsegments > 0) {
    ierr = PetscMalloc1(in.numberofsegments*2, &in.segmentlist);CHKERRQ(ierr);
    ierr = PetscMalloc1(in.numberofsegments, &in.segmentmarkerlist);CHKERRQ(ierr);
    for (e = eStart; e < eEnd; ++e) {
      const PetscInt  idx = e - eStart;
      const PetscInt *cone;
      PetscInt        val;

      ierr = DMPlexGetCone(boundary, e, &cone);CHKERRQ(ierr);

      in.segmentlist[idx*2+0] = cone[0] - vStart;
      in.segmentlist[idx*2+1] = cone[1] - vStart;

      if (label) {
        ierr = DMLabelGetValue(label, e, &val);CHKERRQ(ierr);
        in.segmentmarkerlist[idx] = val;
      }
    }
  }
#if 0 /* Do not currently support holes */
  PetscReal *holeCoords;
  PetscInt   h, d;

  ierr = DMPlexGetHoles(boundary, &in.numberofholes, &holeCords);CHKERRQ(ierr);
  if (in.numberofholes > 0) {
    ierr = PetscMalloc1(in.numberofholes*dim, &in.holelist);CHKERRQ(ierr);
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
    ierr = PetscStrcpy(args, "pqezQ");CHKERRQ(ierr);
    if (createConvexHull)   {ierr = PetscStrcat(args, "c");CHKERRQ(ierr);}
    if (constrained)        {ierr = PetscStrcpy(args, "zepDQ");CHKERRQ(ierr);}
    if (mesh->triangleOpts) {triangulate(mesh->triangleOpts, &in, &out, NULL);}
    else                    {triangulate(args, &in, &out, NULL);}
  }
  ierr = PetscFree(in.pointlist);CHKERRQ(ierr);
  ierr = PetscFree(in.pointmarkerlist);CHKERRQ(ierr);
  ierr = PetscFree(in.segmentlist);CHKERRQ(ierr);
  ierr = PetscFree(in.segmentmarkerlist);CHKERRQ(ierr);
  ierr = PetscFree(in.holelist);CHKERRQ(ierr);

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

      ierr = PetscMalloc1(dim * numVertices,&meshCoords);CHKERRQ(ierr);
      for (i = 0; i < dim * numVertices; i++) {
        meshCoords[i] = (PetscReal) out.pointlist[i];
      }
    }
    if (sizeof (PetscInt) == sizeof (out.trianglelist[0])) {
      cells = (PetscInt *) out.trianglelist;
    } else {
      PetscInt i;

      ierr = PetscMalloc1(numCells * numCorners, &cells);CHKERRQ(ierr);
      for (i = 0; i < numCells * numCorners; i++) {
        cells[i] = (PetscInt) out.trianglelist[i];
      }
    }
    ierr = DMPlexCreateFromCellListPetsc(comm, dim, numCells, numVertices, numCorners, interpolate, cells, dim, meshCoords, dm);CHKERRQ(ierr);
    if (sizeof (PetscReal) != sizeof (out.pointlist[0])) {
      ierr = PetscFree(meshCoords);CHKERRQ(ierr);
    }
    if (sizeof (PetscInt) != sizeof (out.trianglelist[0])) {
      ierr = PetscFree(cells);CHKERRQ(ierr);
    }
    if (label)  {
      ierr = DMCreateLabel(*dm, labelName);CHKERRQ(ierr);
      ierr = DMGetLabel(*dm, labelName, &glabel);CHKERRQ(ierr);
    }
    if (label2) {
      ierr = DMCreateLabel(*dm, labelName2);CHKERRQ(ierr);
      ierr = DMGetLabel(*dm, labelName2, &glabel2);CHKERRQ(ierr);
    }
    /* Set labels */
    for (v = 0; v < numVertices; ++v) {
      if (out.pointmarkerlist[v]) {
        if (glabel) {ierr = DMLabelSetValue(glabel, v+numCells, out.pointmarkerlist[v]);CHKERRQ(ierr);}
      }
    }
    if (interpolate) {
      for (e = 0; e < out.numberofedges; e++) {
        if (out.edgemarkerlist[e]) {
          const PetscInt  vertices[2] = {out.edgelist[e*2+0]+numCells, out.edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          ierr = DMPlexGetJoin(*dm, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          if (glabel)  {ierr = DMLabelSetValue(glabel,  edges[0], out.edgemarkerlist[e]);CHKERRQ(ierr);}
          if (glabel2) {ierr = DMLabelSetValue(glabel2, edges[0], out.edgemarkerlist[e]);CHKERRQ(ierr);}
          ierr = DMPlexRestoreJoin(*dm, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
        }
      }
    }
    ierr = DMPlexSetRefinementUniform(*dm, PETSC_FALSE);CHKERRQ(ierr);
  }
#if 0 /* Do not currently support holes */
  ierr = DMPlexCopyHoles(*dm, boundary);CHKERRQ(ierr);
#endif
  ierr = FiniOutput_Triangle(&out);CHKERRQ(ierr);
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
  PetscErrorCode       ierr;
  double               *maxVolumes;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = InitInput_Triangle(&in);CHKERRQ(ierr);
  ierr = InitOutput_Triangle(&out);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&depth, &depthGlobal, 1, MPIU_INT, MPI_MAX, comm);CHKERRMPI(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, labelName, &label);CHKERRQ(ierr);

  in.numberofpoints = vEnd - vStart;
  if (in.numberofpoints > 0) {
    PetscSection coordSection;
    Vec          coordinates;
    PetscScalar *array;

    ierr = PetscMalloc1(in.numberofpoints*dim, &in.pointlist);CHKERRQ(ierr);
    ierr = PetscMalloc1(in.numberofpoints, &in.pointmarkerlist);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &array);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d, val;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        in.pointlist[idx*dim + d] = PetscRealPart(array[off+d]);
      }
      if (label) {
        ierr = DMLabelGetValue(label, v, &val);CHKERRQ(ierr);
        in.pointmarkerlist[idx] = val;
      }
    }
    ierr = VecRestoreArray(coordinates, &array);CHKERRQ(ierr);
  }
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetGhostCellStratum(dm, &gcStart, NULL);CHKERRQ(ierr);
  if (gcStart >= 0) cEnd = gcStart;

  in.numberofcorners   = 3;
  in.numberoftriangles = cEnd - cStart;

#if !defined(PETSC_USE_REAL_DOUBLE)
  ierr = PetscMalloc1(cEnd - cStart,&maxVolumes);CHKERRQ(ierr);
  for (c = 0; c < cEnd-cStart; ++c) maxVolumes[c] = (double)inmaxVolumes[c];
#else
  maxVolumes = inmaxVolumes;
#endif

  in.trianglearealist  = (double*) maxVolumes;
  if (in.numberoftriangles > 0) {
    ierr = PetscMalloc1(in.numberoftriangles*in.numberofcorners, &in.trianglelist);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt idx      = c - cStart;
      PetscInt      *closure = NULL;
      PetscInt       closureSize;

      ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      if ((closureSize != 4) && (closureSize != 7)) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Mesh has cell which is not a triangle, %D vertices in closure", closureSize);
      for (v = 0; v < 3; ++v) {
        in.trianglelist[idx*in.numberofcorners + v] = closure[(v+closureSize-3)*2] - vStart;
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
  }
  /* TODO: Segment markers are missing on input */
#if 0 /* Do not currently support holes */
  PetscReal *holeCoords;
  PetscInt   h, d;

  ierr = DMPlexGetHoles(boundary, &in.numberofholes, &holeCords);CHKERRQ(ierr);
  if (in.numberofholes > 0) {
    ierr = PetscMalloc1(in.numberofholes*dim, &in.holelist);CHKERRQ(ierr);
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
    ierr = PetscStrcpy(args, "pqezQra");CHKERRQ(ierr);
    triangulate(args, &in, &out, NULL);
  }
  ierr = PetscFree(in.pointlist);CHKERRQ(ierr);
  ierr = PetscFree(in.pointmarkerlist);CHKERRQ(ierr);
  ierr = PetscFree(in.segmentlist);CHKERRQ(ierr);
  ierr = PetscFree(in.segmentmarkerlist);CHKERRQ(ierr);
  ierr = PetscFree(in.trianglelist);CHKERRQ(ierr);

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

      ierr = PetscMalloc1(dim * numVertices,&meshCoords);CHKERRQ(ierr);
      for (i = 0; i < dim * numVertices; i++) {
        meshCoords[i] = (PetscReal) out.pointlist[i];
      }
    }
    if (sizeof (PetscInt) == sizeof (out.trianglelist[0])) {
      cells = (PetscInt *) out.trianglelist;
    } else {
      PetscInt i;

      ierr = PetscMalloc1(numCells * numCorners, &cells);CHKERRQ(ierr);
      for (i = 0; i < numCells * numCorners; i++) {
        cells[i] = (PetscInt) out.trianglelist[i];
      }
    }

    ierr = DMPlexCreateFromCellListPetsc(comm, dim, numCells, numVertices, numCorners, interpolate, cells, dim, meshCoords, dmRefined);CHKERRQ(ierr);
    if (label) {
      ierr = DMCreateLabel(*dmRefined, labelName);CHKERRQ(ierr);
      ierr = DMGetLabel(*dmRefined, labelName, &rlabel);CHKERRQ(ierr);
    }
    if (sizeof (PetscReal) != sizeof (out.pointlist[0])) {
      ierr = PetscFree(meshCoords);CHKERRQ(ierr);
    }
    if (sizeof (PetscInt) != sizeof (out.trianglelist[0])) {
      ierr = PetscFree(cells);CHKERRQ(ierr);
    }
    /* Set labels */
    for (v = 0; v < numVertices; ++v) {
      if (out.pointmarkerlist[v]) {
        if (rlabel) {ierr = DMLabelSetValue(rlabel, v+numCells, out.pointmarkerlist[v]);CHKERRQ(ierr);}
      }
    }
    if (interpolate) {
      PetscInt e;

      for (e = 0; e < out.numberofedges; e++) {
        if (out.edgemarkerlist[e]) {
          const PetscInt  vertices[2] = {out.edgelist[e*2+0]+numCells, out.edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          ierr = DMPlexGetJoin(*dmRefined, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          if (rlabel) {ierr = DMLabelSetValue(rlabel, edges[0], out.edgemarkerlist[e]);CHKERRQ(ierr);}
          ierr = DMPlexRestoreJoin(*dmRefined, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
        }
      }
    }
    ierr = DMPlexSetRefinementUniform(*dmRefined, PETSC_FALSE);CHKERRQ(ierr);
  }
#if 0 /* Do not currently support holes */
  ierr = DMPlexCopyHoles(*dm, boundary);CHKERRQ(ierr);
#endif
  ierr = FiniOutput_Triangle(&out);CHKERRQ(ierr);
#if !defined(PETSC_USE_REAL_DOUBLE)
  ierr = PetscFree(maxVolumes);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
