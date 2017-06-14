#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

PetscErrorCode DMPlexInvertCell_Internal(PetscInt dim, PetscInt numCorners, PetscInt cone[])
{
  int tmpc;

  PetscFunctionBegin;
  if (dim != 3) PetscFunctionReturn(0);
  switch (numCorners) {
  case 4:
    tmpc    = cone[0];
    cone[0] = cone[1];
    cone[1] = tmpc;
    break;
  case 8:
    tmpc    = cone[1];
    cone[1] = cone[3];
    cone[3] = tmpc;
    break;
  default: break;
  }
  PetscFunctionReturn(0);
}

/*@C
  DMPlexInvertCell - This flips tetrahedron and hexahedron orientation since Plex stores them internally with outward normals. Other cells are left untouched.

  Input Parameters:
+ numCorners - The number of vertices in a cell
- cone - The incoming cone

  Output Parameter:
. cone - The inverted cone (in-place)

  Level: developer

.seealso: DMPlexGenerate()
@*/
PetscErrorCode DMPlexInvertCell(PetscInt dim, PetscInt numCorners, int cone[])
{
  int tmpc;

  PetscFunctionBegin;
  if (dim != 3) PetscFunctionReturn(0);
  switch (numCorners) {
  case 4:
    tmpc    = cone[0];
    cone[0] = cone[1];
    cone[1] = tmpc;
    break;
  case 8:
    tmpc    = cone[1];
    cone[1] = cone[3];
    cone[3] = tmpc;
    break;
  default: break;
  }
  PetscFunctionReturn(0);
}

/* This is to fix the tetrahedron orientation from TetGen */
PETSC_UNUSED static PetscErrorCode DMPlexInvertCells_Internal(PetscInt dim, PetscInt numCells, PetscInt numCorners, int cells[])
{
  PetscInt       bound = numCells*numCorners, coff;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  for (coff = 0; coff < bound; coff += numCorners) {
    ierr = DMPlexInvertCell(dim, numCorners, &cells[coff]);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  DMPlexTriangleSetOptions - Set the options used for the Triangle mesh generator

  Not Collective

  Inputs Parameters:
+ dm - The DMPlex object
- opts - The command line options

  Level: developer

.keywords: mesh, points
.seealso: DMPlexTetgenSetOptions(), DMPlexGenerate()
@*/
PetscErrorCode DMPlexTriangleSetOptions(DM dm, const char *opts)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(opts, 2);
  ierr = PetscFree(mesh->triangleOpts);CHKERRQ(ierr);
  ierr = PetscStrallocpy(opts, &mesh->triangleOpts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMPlexTetgenSetOptions - Set the options used for the Tetgen mesh generator

  Not Collective

  Inputs Parameters:
+ dm - The DMPlex object
- opts - The command line options

  Level: developer

.keywords: mesh, points
.seealso: DMPlexTriangleSetOptions(), DMPlexGenerate()
@*/
PetscErrorCode DMPlexTetgenSetOptions(DM dm, const char *opts)
{
  DM_Plex       *mesh = (DM_Plex*) dm->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscValidPointer(opts, 2);
  ierr = PetscFree(mesh->tetgenOpts);CHKERRQ(ierr);
  ierr = PetscStrallocpy(opts, &mesh->tetgenOpts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_TRIANGLE)
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

PetscErrorCode DMPlexGenerate_Triangle(DM boundary, PetscBool interpolate, DM *dm)
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
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
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
      PetscInt       off, d;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        in.pointlist[idx*dim + d] = PetscRealPart(array[off+d]);
      }
      if (label) {ierr = DMLabelGetValue(label, v, &in.pointmarkerlist[idx]);CHKERRQ(ierr);}
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

      ierr = DMPlexGetCone(boundary, e, &cone);CHKERRQ(ierr);

      in.segmentlist[idx*2+0] = cone[0] - vStart;
      in.segmentlist[idx*2+1] = cone[1] - vStart;

      if (label) {ierr = DMLabelGetValue(label, e, &in.segmentmarkerlist[idx]);CHKERRQ(ierr);}
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
  if (!rank) {
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
    DMLabel        glabel      = NULL;
    DMLabel        glabel2     = NULL;
    const PetscInt numCorners  = 3;
    const PetscInt numCells    = out.numberoftriangles;
    const PetscInt numVertices = out.numberofpoints;
    const int     *cells      = out.trianglelist;
    const double  *meshCoords = out.pointlist;

    ierr = DMPlexCreateFromCellList(comm, dim, numCells, numVertices, numCorners, interpolate, cells, dim, meshCoords, dm);CHKERRQ(ierr);
    if (label)  {ierr = DMCreateLabel(*dm, labelName);  ierr = DMGetLabel(*dm, labelName,  &glabel);}
    if (label2) {ierr = DMCreateLabel(*dm, labelName2); ierr = DMGetLabel(*dm, labelName2, &glabel2);}
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

PetscErrorCode DMPlexRefine_Triangle(DM dm, double *maxVolumes, DM *dmRefined)
{
  MPI_Comm             comm;
  PetscInt             dim       = 2;
  const char          *labelName = "marker";
  struct triangulateio in;
  struct triangulateio out;
  DMLabel              label;
  PetscInt             vStart, vEnd, v, cStart, cEnd, c, depth, depthGlobal;
  PetscMPIInt          rank;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = InitInput_Triangle(&in);CHKERRQ(ierr);
  ierr = InitOutput_Triangle(&out);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&depth, &depthGlobal, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
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
      PetscInt       off, d;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        in.pointlist[idx*dim + d] = PetscRealPart(array[off+d]);
      }
      if (label) {ierr = DMLabelGetValue(label, v, &in.pointmarkerlist[idx]);CHKERRQ(ierr);}
    }
    ierr = VecRestoreArray(coordinates, &array);CHKERRQ(ierr);
  }
  ierr  = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);

  in.numberofcorners   = 3;
  in.numberoftriangles = cEnd - cStart;

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
  if (!rank) {
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
    DMLabel        rlabel      = NULL;
    const PetscInt numCorners  = 3;
    const PetscInt numCells    = out.numberoftriangles;
    const PetscInt numVertices = out.numberofpoints;
    const int     *cells      = out.trianglelist;
    const double  *meshCoords = out.pointlist;
    PetscBool      interpolate = depthGlobal > 1 ? PETSC_TRUE : PETSC_FALSE;

    ierr = DMPlexCreateFromCellList(comm, dim, numCells, numVertices, numCorners, interpolate, cells, dim, meshCoords, dmRefined);CHKERRQ(ierr);
    if (label) {ierr = DMCreateLabel(*dmRefined, labelName); ierr = DMGetLabel(*dmRefined, labelName, &rlabel);}
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
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_TRIANGLE */

#if defined(PETSC_HAVE_TETGEN)
#include <tetgen.h>
PetscErrorCode DMPlexGenerate_Tetgen(DM boundary, PetscBool interpolate, DM *dm)
{
  MPI_Comm       comm;
  DM_Plex       *mesh      = (DM_Plex *) boundary->data;
  const PetscInt dim       = 3;
  const char    *labelName = "marker";
  ::tetgenio     in;
  ::tetgenio     out;
  DMLabel        label;
  PetscInt       vStart, vEnd, v, fStart, fEnd, f;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)boundary,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(boundary, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMGetLabel(boundary, labelName, &label);CHKERRQ(ierr);

  in.numberofpoints = vEnd - vStart;
  if (in.numberofpoints > 0) {
    PetscSection coordSection;
    Vec          coordinates;
    PetscScalar *array;

    in.pointlist       = new double[in.numberofpoints*dim];
    in.pointmarkerlist = new int[in.numberofpoints];

    ierr = DMGetCoordinatesLocal(boundary, &coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(boundary, &coordSection);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &array);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) in.pointlist[idx*dim + d] = PetscRealPart(array[off+d]);
      if (label) {
        PetscInt val;

        ierr = DMLabelGetValue(label, v, &val);CHKERRQ(ierr);
        in.pointmarkerlist[idx] = (int) val;
      }
    }
    ierr = VecRestoreArray(coordinates, &array);CHKERRQ(ierr);
  }
  ierr  = DMPlexGetHeightStratum(boundary, 0, &fStart, &fEnd);CHKERRQ(ierr);

  in.numberoffacets = fEnd - fStart;
  if (in.numberoffacets > 0) {
    in.facetlist       = new tetgenio::facet[in.numberoffacets];
    in.facetmarkerlist = new int[in.numberoffacets];
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt idx     = f - fStart;
      PetscInt      *points = NULL, numPoints, p, numVertices = 0, v;

      in.facetlist[idx].numberofpolygons = 1;
      in.facetlist[idx].polygonlist      = new tetgenio::polygon[in.facetlist[idx].numberofpolygons];
      in.facetlist[idx].numberofholes    = 0;
      in.facetlist[idx].holelist         = NULL;

      ierr = DMPlexGetTransitiveClosure(boundary, f, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
      for (p = 0; p < numPoints*2; p += 2) {
        const PetscInt point = points[p];
        if ((point >= vStart) && (point < vEnd)) points[numVertices++] = point;
      }

      tetgenio::polygon *poly = in.facetlist[idx].polygonlist;
      poly->numberofvertices = numVertices;
      poly->vertexlist       = new int[poly->numberofvertices];
      for (v = 0; v < numVertices; ++v) {
        const PetscInt vIdx = points[v] - vStart;
        poly->vertexlist[v] = vIdx;
      }
      if (label) {
        PetscInt val;

        ierr = DMLabelGetValue(label, f, &val);CHKERRQ(ierr);
        in.facetmarkerlist[idx] = (int) val;
      }
      ierr = DMPlexRestoreTransitiveClosure(boundary, f, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
    }
  }
  if (!rank) {
    char args[32];

    /* Take away 'Q' for verbose output */
    ierr = PetscStrcpy(args, "pqezQ");CHKERRQ(ierr);
    if (mesh->tetgenOpts) {::tetrahedralize(mesh->tetgenOpts, &in, &out);}
    else                  {::tetrahedralize(args, &in, &out);}
  }
  {
    DMLabel        glabel      = NULL;
    const PetscInt numCorners  = 4;
    const PetscInt numCells    = out.numberoftetrahedra;
    const PetscInt numVertices = out.numberofpoints;
    const double   *meshCoords = out.pointlist;
    int            *cells      = out.tetrahedronlist;

    ierr = DMPlexInvertCells_Internal(dim, numCells, numCorners, cells);CHKERRQ(ierr);
    ierr = DMPlexCreateFromCellList(comm, dim, numCells, numVertices, numCorners, interpolate, cells, dim, meshCoords, dm);CHKERRQ(ierr);
    if (label) {ierr = DMCreateLabel(*dm, labelName); ierr = DMGetLabel(*dm, labelName, &glabel);}
    /* Set labels */
    for (v = 0; v < numVertices; ++v) {
      if (out.pointmarkerlist[v]) {
        if (glabel) {ierr = DMLabelSetValue(glabel, v+numCells, out.pointmarkerlist[v]);CHKERRQ(ierr);}
      }
    }
    if (interpolate) {
#if 0
      PetscInt e;

      /* This check is never actually executed for ctetgen (which never returns edgemarkers) and seems to be broken for
       * tetgen */
      for (e = 0; e < out.numberofedges; e++) {
        if (out.edgemarkerlist[e]) {
          const PetscInt  vertices[2] = {out.edgelist[e*2+0]+numCells, out.edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          ierr = DMPlexGetJoin(*dm, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          if (glabel) {ierr = DMLabelSetValue(glabel, edges[0], out.edgemarkerlist[e]);CHKERRQ(ierr);}
          ierr = DMPlexRestoreJoin(*dm, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
        }
      }
#endif
      for (f = 0; f < out.numberoftrifaces; f++) {
        if (out.trifacemarkerlist[f]) {
          const PetscInt  vertices[3] = {out.trifacelist[f*3+0]+numCells, out.trifacelist[f*3+1]+numCells, out.trifacelist[f*3+2]+numCells};
          const PetscInt *faces;
          PetscInt        numFaces;

          ierr = DMPlexGetFullJoin(*dm, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
          if (numFaces != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Three vertices must cover only one face, not %D", numFaces);
          if (glabel) {ierr = DMLabelSetValue(glabel, faces[0], out.trifacemarkerlist[f]);CHKERRQ(ierr);}
          ierr = DMPlexRestoreJoin(*dm, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
        }
      }
    }
    ierr = DMPlexSetRefinementUniform(*dm, PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexRefine_Tetgen(DM dm, double *maxVolumes, DM *dmRefined)
{
  MPI_Comm       comm;
  const PetscInt dim       = 3;
  const char    *labelName = "marker";
  ::tetgenio     in;
  ::tetgenio     out;
  DMLabel        label;
  PetscInt       vStart, vEnd, v, cStart, cEnd, c, depth, depthGlobal;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&depth, &depthGlobal, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, labelName, &label);CHKERRQ(ierr);

  in.numberofpoints = vEnd - vStart;
  if (in.numberofpoints > 0) {
    PetscSection coordSection;
    Vec          coordinates;
    PetscScalar *array;

    in.pointlist       = new double[in.numberofpoints*dim];
    in.pointmarkerlist = new int[in.numberofpoints];

    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &array);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) in.pointlist[idx*dim + d] = PetscRealPart(array[off+d]);
      if (label) {
        PetscInt val;
        
        ierr = DMLabelGetValue(label, v, &val);CHKERRQ(ierr);
        in.pointmarkerlist[idx] = (int) val;
      }
    }
    ierr = VecRestoreArray(coordinates, &array);CHKERRQ(ierr);
  }
  ierr  = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);

  in.numberofcorners       = 4;
  in.numberoftetrahedra    = cEnd - cStart;
  in.tetrahedronvolumelist = (double*) maxVolumes;
  if (in.numberoftetrahedra > 0) {
    in.tetrahedronlist = new int[in.numberoftetrahedra*in.numberofcorners];
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt idx      = c - cStart;
      PetscInt      *closure = NULL;
      PetscInt       closureSize;

      ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      if ((closureSize != 5) && (closureSize != 15)) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Mesh has cell which is not a tetrahedron, %D vertices in closure", closureSize);
      for (v = 0; v < 4; ++v) {
        in.tetrahedronlist[idx*in.numberofcorners + v] = closure[(v+closureSize-4)*2] - vStart;
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
  }
  /* TODO: Put in boundary faces with markers */
  if (!rank) {
    char args[32];

#if 1
    /* Take away 'Q' for verbose output */
    ierr = PetscStrcpy(args, "qezQra");CHKERRQ(ierr);
#else
    ierr = PetscStrcpy(args, "qezraVVVV");CHKERRQ(ierr);
#endif
    ::tetrahedralize(args, &in, &out);
  }
  in.tetrahedronvolumelist = NULL;

  {
    DMLabel        rlabel      = NULL;
    const PetscInt numCorners  = 4;
    const PetscInt numCells    = out.numberoftetrahedra;
    const PetscInt numVertices = out.numberofpoints;
    const double   *meshCoords = out.pointlist;
    int            *cells      = out.tetrahedronlist;

    PetscBool      interpolate = depthGlobal > 1 ? PETSC_TRUE : PETSC_FALSE;

    ierr = DMPlexInvertCells_Internal(dim, numCells, numCorners, cells);CHKERRQ(ierr);
    ierr = DMPlexCreateFromCellList(comm, dim, numCells, numVertices, numCorners, interpolate, cells, dim, meshCoords, dmRefined);CHKERRQ(ierr);
    if (label) {ierr = DMCreateLabel(*dmRefined, labelName); ierr = DMGetLabel(*dmRefined, labelName, &rlabel);}
    /* Set labels */
    for (v = 0; v < numVertices; ++v) {
      if (out.pointmarkerlist[v]) {
        if (rlabel) {ierr = DMLabelSetValue(rlabel, v+numCells, out.pointmarkerlist[v]);CHKERRQ(ierr);}
      }
    }
    if (interpolate) {
      PetscInt f;
#if 0
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
#endif
      for (f = 0; f < out.numberoftrifaces; f++) {
        if (out.trifacemarkerlist[f]) {
          const PetscInt  vertices[3] = {out.trifacelist[f*3+0]+numCells, out.trifacelist[f*3+1]+numCells, out.trifacelist[f*3+2]+numCells};
          const PetscInt *faces;
          PetscInt        numFaces;

          ierr = DMPlexGetFullJoin(*dmRefined, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
          if (numFaces != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Three vertices must cover only one face, not %D", numFaces);
          if (rlabel) {ierr = DMLabelSetValue(rlabel, faces[0], out.trifacemarkerlist[f]);CHKERRQ(ierr);}
          ierr = DMPlexRestoreJoin(*dmRefined, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
        }
      }
    }
    ierr = DMPlexSetRefinementUniform(*dmRefined, PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_TETGEN */

#if defined(PETSC_HAVE_CTETGEN)
#include <ctetgen.h>

PetscErrorCode DMPlexGenerate_CTetgen(DM boundary, PetscBool interpolate, DM *dm)
{
  MPI_Comm       comm;
  const PetscInt dim       = 3;
  const char    *labelName = "marker";
  PLC           *in, *out;
  DMLabel        label;
  PetscInt       verbose = 0, vStart, vEnd, v, fStart, fEnd, f;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)boundary,&comm);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,((PetscObject) boundary)->prefix, "-ctetgen_verbose", &verbose, NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(boundary, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMGetLabel(boundary, labelName, &label);CHKERRQ(ierr);
  ierr = PLCCreate(&in);CHKERRQ(ierr);
  ierr = PLCCreate(&out);CHKERRQ(ierr);

  in->numberofpoints = vEnd - vStart;
  if (in->numberofpoints > 0) {
    PetscSection coordSection;
    Vec          coordinates;
    PetscScalar *array;

    ierr = PetscMalloc1(in->numberofpoints*dim, &in->pointlist);CHKERRQ(ierr);
    ierr = PetscMalloc1(in->numberofpoints,       &in->pointmarkerlist);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(boundary, &coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(boundary, &coordSection);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &array);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d, m = -1;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        in->pointlist[idx*dim + d] = PetscRealPart(array[off+d]);
      }
      if (label) {ierr = DMLabelGetValue(label, v, &m);CHKERRQ(ierr);}

      in->pointmarkerlist[idx] = (int) m;
    }
    ierr = VecRestoreArray(coordinates, &array);CHKERRQ(ierr);
  }
  ierr  = DMPlexGetHeightStratum(boundary, 0, &fStart, &fEnd);CHKERRQ(ierr);

  in->numberoffacets = fEnd - fStart;
  if (in->numberoffacets > 0) {
    ierr = PetscMalloc1(in->numberoffacets, &in->facetlist);CHKERRQ(ierr);
    ierr = PetscMalloc1(in->numberoffacets,   &in->facetmarkerlist);CHKERRQ(ierr);
    for (f = fStart; f < fEnd; ++f) {
      const PetscInt idx     = f - fStart;
      PetscInt      *points = NULL, numPoints, p, numVertices = 0, v, m = -1;
      polygon       *poly;

      in->facetlist[idx].numberofpolygons = 1;

      ierr = PetscMalloc1(in->facetlist[idx].numberofpolygons, &in->facetlist[idx].polygonlist);CHKERRQ(ierr);

      in->facetlist[idx].numberofholes    = 0;
      in->facetlist[idx].holelist         = NULL;

      ierr = DMPlexGetTransitiveClosure(boundary, f, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
      for (p = 0; p < numPoints*2; p += 2) {
        const PetscInt point = points[p];
        if ((point >= vStart) && (point < vEnd)) points[numVertices++] = point;
      }

      poly                   = in->facetlist[idx].polygonlist;
      poly->numberofvertices = numVertices;
      ierr                   = PetscMalloc1(poly->numberofvertices, &poly->vertexlist);CHKERRQ(ierr);
      for (v = 0; v < numVertices; ++v) {
        const PetscInt vIdx = points[v] - vStart;
        poly->vertexlist[v] = vIdx;
      }
      if (label) {ierr = DMLabelGetValue(label, f, &m);CHKERRQ(ierr);}
      in->facetmarkerlist[idx] = (int) m;
      ierr = DMPlexRestoreTransitiveClosure(boundary, f, PETSC_TRUE, &numPoints, &points);CHKERRQ(ierr);
    }
  }
  if (!rank) {
    TetGenOpts t;

    ierr        = TetGenOptsInitialize(&t);CHKERRQ(ierr);
    t.in        = boundary; /* Should go away */
    t.plc       = 1;
    t.quality   = 1;
    t.edgesout  = 1;
    t.zeroindex = 1;
    t.quiet     = 1;
    t.verbose   = verbose;
    ierr        = TetGenCheckOpts(&t);CHKERRQ(ierr);
    ierr        = TetGenTetrahedralize(&t, in, out);CHKERRQ(ierr);
  }
  {
    DMLabel        glabel      = NULL;
    const PetscInt numCorners  = 4;
    const PetscInt numCells    = out->numberoftetrahedra;
    const PetscInt numVertices = out->numberofpoints;
    double         *meshCoords;
    int            *cells      = out->tetrahedronlist;

    if (sizeof (PetscReal) == sizeof (double)) {
      meshCoords = (double *) out->pointlist;
    }
    else {
      PetscInt i;

      ierr = PetscMalloc1(3 * numVertices,&meshCoords);CHKERRQ(ierr);
      for (i = 0; i < 3 * numVertices; i++) {
        meshCoords[i] = (PetscReal) out->pointlist[i];
      }
    }

    ierr = DMPlexInvertCells_Internal(dim, numCells, numCorners, cells);CHKERRQ(ierr);
    ierr = DMPlexCreateFromCellList(comm, dim, numCells, numVertices, numCorners, interpolate, cells, dim, meshCoords, dm);CHKERRQ(ierr);
    if (sizeof (PetscReal) != sizeof (double)) {
      ierr = PetscFree(meshCoords);CHKERRQ(ierr);
    }
    if (label) {ierr = DMCreateLabel(*dm, labelName); ierr = DMGetLabel(*dm, labelName, &glabel);}
    /* Set labels */
    for (v = 0; v < numVertices; ++v) {
      if (out->pointmarkerlist[v]) {
        if (glabel) {ierr = DMLabelSetValue(glabel, v+numCells, out->pointmarkerlist[v]);CHKERRQ(ierr);}
      }
    }
    if (interpolate) {
      PetscInt e;

      for (e = 0; e < out->numberofedges; e++) {
        if (out->edgemarkerlist[e]) {
          const PetscInt  vertices[2] = {out->edgelist[e*2+0]+numCells, out->edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          ierr = DMPlexGetJoin(*dm, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          if (glabel) {ierr = DMLabelSetValue(glabel, edges[0], out->edgemarkerlist[e]);CHKERRQ(ierr);}
          ierr = DMPlexRestoreJoin(*dm, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
        }
      }
      for (f = 0; f < out->numberoftrifaces; f++) {
        if (out->trifacemarkerlist[f]) {
          const PetscInt  vertices[3] = {out->trifacelist[f*3+0]+numCells, out->trifacelist[f*3+1]+numCells, out->trifacelist[f*3+2]+numCells};
          const PetscInt *faces;
          PetscInt        numFaces;

          ierr = DMPlexGetFullJoin(*dm, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
          if (numFaces != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Three vertices must cover only one face, not %D", numFaces);
          if (glabel) {ierr = DMLabelSetValue(glabel, faces[0], out->trifacemarkerlist[f]);CHKERRQ(ierr);}
          ierr = DMPlexRestoreJoin(*dm, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
        }
      }
    }
    ierr = DMPlexSetRefinementUniform(*dm, PETSC_FALSE);CHKERRQ(ierr);
  }

  ierr = PLCDestroy(&in);CHKERRQ(ierr);
  ierr = PLCDestroy(&out);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexRefine_CTetgen(DM dm, PetscReal *maxVolumes, DM *dmRefined)
{
  MPI_Comm       comm;
  const PetscInt dim       = 3;
  const char    *labelName = "marker";
  PLC           *in, *out;
  DMLabel        label;
  PetscInt       verbose = 0, vStart, vEnd, v, cStart, cEnd, c, depth, depthGlobal;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm,&comm);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,((PetscObject) dm)->prefix, "-ctetgen_verbose", &verbose, NULL);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMPlexGetDepth(dm, &depth);CHKERRQ(ierr);
  ierr = MPIU_Allreduce(&depth, &depthGlobal, 1, MPIU_INT, MPI_MAX, comm);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, labelName, &label);CHKERRQ(ierr);
  ierr = PLCCreate(&in);CHKERRQ(ierr);
  ierr = PLCCreate(&out);CHKERRQ(ierr);

  in->numberofpoints = vEnd - vStart;
  if (in->numberofpoints > 0) {
    PetscSection coordSection;
    Vec          coordinates;
    PetscScalar *array;

    ierr = PetscMalloc1(in->numberofpoints*dim, &in->pointlist);CHKERRQ(ierr);
    ierr = PetscMalloc1(in->numberofpoints,       &in->pointmarkerlist);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
    ierr = DMGetCoordinateSection(dm, &coordSection);CHKERRQ(ierr);
    ierr = VecGetArray(coordinates, &array);CHKERRQ(ierr);
    for (v = vStart; v < vEnd; ++v) {
      const PetscInt idx = v - vStart;
      PetscInt       off, d, m = -1;

      ierr = PetscSectionGetOffset(coordSection, v, &off);CHKERRQ(ierr);
      for (d = 0; d < dim; ++d) {
        in->pointlist[idx*dim + d] = PetscRealPart(array[off+d]);
      }
      if (label) {ierr = DMLabelGetValue(label, v, &m);CHKERRQ(ierr);}

      in->pointmarkerlist[idx] = (int) m;
    }
    ierr = VecRestoreArray(coordinates, &array);CHKERRQ(ierr);
  }
  ierr  = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);

  in->numberofcorners       = 4;
  in->numberoftetrahedra    = cEnd - cStart;
  in->tetrahedronvolumelist = maxVolumes;
  if (in->numberoftetrahedra > 0) {
    ierr = PetscMalloc1(in->numberoftetrahedra*in->numberofcorners, &in->tetrahedronlist);CHKERRQ(ierr);
    for (c = cStart; c < cEnd; ++c) {
      const PetscInt idx      = c - cStart;
      PetscInt      *closure = NULL;
      PetscInt       closureSize;

      ierr = DMPlexGetTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
      if ((closureSize != 5) && (closureSize != 15)) SETERRQ1(comm, PETSC_ERR_ARG_WRONG, "Mesh has cell which is not a tetrahedron, %D vertices in closure", closureSize);
      for (v = 0; v < 4; ++v) {
        in->tetrahedronlist[idx*in->numberofcorners + v] = closure[(v+closureSize-4)*2] - vStart;
      }
      ierr = DMPlexRestoreTransitiveClosure(dm, c, PETSC_TRUE, &closureSize, &closure);CHKERRQ(ierr);
    }
  }
  if (!rank) {
    TetGenOpts t;

    ierr = TetGenOptsInitialize(&t);CHKERRQ(ierr);

    t.in        = dm; /* Should go away */
    t.refine    = 1;
    t.varvolume = 1;
    t.quality   = 1;
    t.edgesout  = 1;
    t.zeroindex = 1;
    t.quiet     = 1;
    t.verbose   = verbose; /* Change this */

    ierr = TetGenCheckOpts(&t);CHKERRQ(ierr);
    ierr = TetGenTetrahedralize(&t, in, out);CHKERRQ(ierr);
  }
  in->tetrahedronvolumelist = NULL;
  {
    DMLabel        rlabel      = NULL;
    const PetscInt numCorners  = 4;
    const PetscInt numCells    = out->numberoftetrahedra;
    const PetscInt numVertices = out->numberofpoints;
    double         *meshCoords;
    int            *cells      = out->tetrahedronlist;
    PetscBool      interpolate = depthGlobal > 1 ? PETSC_TRUE : PETSC_FALSE;

    if (sizeof (PetscReal) == sizeof (double)) {
      meshCoords = (double *) out->pointlist;
    }
    else {
      PetscInt i;

      ierr = PetscMalloc1(3 * numVertices,&meshCoords);CHKERRQ(ierr);
      for (i = 0; i < 3 * numVertices; i++) {
        meshCoords[i] = (PetscReal) out->pointlist[i];
      }
    }

    ierr = DMPlexInvertCells_Internal(dim, numCells, numCorners, cells);CHKERRQ(ierr);
    ierr = DMPlexCreateFromCellList(comm, dim, numCells, numVertices, numCorners, interpolate, cells, dim, meshCoords, dmRefined);CHKERRQ(ierr);
    if (sizeof (PetscReal) != sizeof (double)) {
      ierr = PetscFree(meshCoords);CHKERRQ(ierr);
    }
    if (label) {ierr = DMCreateLabel(*dmRefined, labelName); ierr = DMGetLabel(*dmRefined, labelName, &rlabel);}
    /* Set labels */
    for (v = 0; v < numVertices; ++v) {
      if (out->pointmarkerlist[v]) {
        if (rlabel) {ierr = DMLabelSetValue(rlabel, v+numCells, out->pointmarkerlist[v]);CHKERRQ(ierr);}
      }
    }
    if (interpolate) {
      PetscInt e, f;

      for (e = 0; e < out->numberofedges; e++) {
        if (out->edgemarkerlist[e]) {
          const PetscInt  vertices[2] = {out->edgelist[e*2+0]+numCells, out->edgelist[e*2+1]+numCells};
          const PetscInt *edges;
          PetscInt        numEdges;

          ierr = DMPlexGetJoin(*dmRefined, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
          if (numEdges != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Two vertices must cover only one edge, not %D", numEdges);
          if (rlabel) {ierr = DMLabelSetValue(rlabel, edges[0], out->edgemarkerlist[e]);CHKERRQ(ierr);}
          ierr = DMPlexRestoreJoin(*dmRefined, 2, vertices, &numEdges, &edges);CHKERRQ(ierr);
        }
      }
      for (f = 0; f < out->numberoftrifaces; f++) {
        if (out->trifacemarkerlist[f]) {
          const PetscInt  vertices[3] = {out->trifacelist[f*3+0]+numCells, out->trifacelist[f*3+1]+numCells, out->trifacelist[f*3+2]+numCells};
          const PetscInt *faces;
          PetscInt        numFaces;

          ierr = DMPlexGetFullJoin(*dmRefined, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
          if (numFaces != 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Three vertices must cover only one face, not %D", numFaces);
          if (rlabel) {ierr = DMLabelSetValue(rlabel, faces[0], out->trifacemarkerlist[f]);CHKERRQ(ierr);}
          ierr = DMPlexRestoreJoin(*dmRefined, 3, vertices, &numFaces, &faces);CHKERRQ(ierr);
        }
      }
    }
    ierr = DMPlexSetRefinementUniform(*dmRefined, PETSC_FALSE);CHKERRQ(ierr);
  }
  ierr = PLCDestroy(&in);CHKERRQ(ierr);
  ierr = PLCDestroy(&out);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif /* PETSC_HAVE_CTETGEN */

/*@C
  DMPlexGenerate - Generates a mesh.

  Not Collective

  Input Parameters:
+ boundary - The DMPlex boundary object
. name - The mesh generation package name
- interpolate - Flag to create intermediate mesh elements

  Output Parameter:
. mesh - The DMPlex object

  Level: intermediate

.keywords: mesh, elements
.seealso: DMPlexCreate(), DMRefine()
@*/
PetscErrorCode DMPlexGenerate(DM boundary, const char name[], PetscBool interpolate, DM *mesh)
{
  PetscInt       dim;
  char           genname[1024];
  PetscBool      isTriangle = PETSC_FALSE, isTetgen = PETSC_FALSE, isCTetgen = PETSC_FALSE, flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(boundary, DM_CLASSID, 1);
  PetscValidLogicalCollectiveBool(boundary, interpolate, 2);
  ierr = DMGetDimension(boundary, &dim);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(((PetscObject) boundary)->options,((PetscObject) boundary)->prefix, "-dm_plex_generator", genname, 1024, &flg);CHKERRQ(ierr);
  if (flg) name = genname;
  if (name) {
    ierr = PetscStrcmp(name, "triangle", &isTriangle);CHKERRQ(ierr);
    ierr = PetscStrcmp(name, "tetgen",   &isTetgen);CHKERRQ(ierr);
    ierr = PetscStrcmp(name, "ctetgen",  &isCTetgen);CHKERRQ(ierr);
  }
  switch (dim) {
  case 1:
    if (!name || isTriangle) {
#if defined(PETSC_HAVE_TRIANGLE)
      ierr = DMPlexGenerate_Triangle(boundary, interpolate, mesh);CHKERRQ(ierr);
#else
      SETERRQ(PetscObjectComm((PetscObject)boundary), PETSC_ERR_SUP, "Mesh generation needs external package support.\nPlease reconfigure with --download-triangle.");
#endif
    } else SETERRQ1(PetscObjectComm((PetscObject)boundary), PETSC_ERR_SUP, "Unknown 2D mesh generation package %s", name);
    break;
  case 2:
    if (!name) {
#if defined(PETSC_HAVE_CTETGEN)
      ierr = DMPlexGenerate_CTetgen(boundary, interpolate, mesh);CHKERRQ(ierr);
#elif defined(PETSC_HAVE_TETGEN)
      ierr = DMPlexGenerate_Tetgen(boundary, interpolate, mesh);CHKERRQ(ierr);
#else
      SETERRQ(PetscObjectComm((PetscObject)boundary), PETSC_ERR_SUP, "External package CTetgen or Tetgen needed.\nPlease reconfigure with '--download-ctetgen' or '--with-clanguage=cxx --download-tetgen'.");
#endif
    } else if (isCTetgen) {
#if defined(PETSC_HAVE_CTETGEN)
      ierr = DMPlexGenerate_CTetgen(boundary, interpolate, mesh);CHKERRQ(ierr);
#else
      SETERRQ(PetscObjectComm((PetscObject)boundary), PETSC_ERR_SUP, "CTetgen needs external package support.\nPlease reconfigure with --download-ctetgen.");
#endif
    } else if (isTetgen) {
#if defined(PETSC_HAVE_TETGEN)
      ierr = DMPlexGenerate_Tetgen(boundary, interpolate, mesh);CHKERRQ(ierr);
#else
      SETERRQ(PetscObjectComm((PetscObject)boundary), PETSC_ERR_SUP, "Tetgen needs external package support.\nPlease reconfigure with --download-tetgen.");
#endif
    } else SETERRQ1(PetscObjectComm((PetscObject)boundary), PETSC_ERR_SUP, "Unknown 3D mesh generation package %s", name);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)boundary), PETSC_ERR_SUP, "Mesh generation for a dimension %d boundary is not supported.", dim);
  }
  PetscFunctionReturn(0);
}
