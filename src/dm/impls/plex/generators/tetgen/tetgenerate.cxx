#include <petsc/private/dmpleximpl.h>   /*I      "petscdmplex.h"   I*/

#if defined(PETSC_HAVE_TETGEN_TETLIBRARY_NEEDED)
#define TETLIBRARY
#endif
#include <tetgen.h>

/* This is to fix the tetrahedron orientation from TetGen */
static PetscErrorCode DMPlexInvertCells_Tetgen(PetscInt numCells, PetscInt numCorners, PetscInt cells[])
{
  PetscInt bound = numCells*numCorners, coff;

  PetscFunctionBegin;
#define SWAP(a,b) do { PetscInt tmp = (a); (a) = (b); (b) = tmp; } while (0)
  for (coff = 0; coff < bound; coff += numCorners) SWAP(cells[coff],cells[coff+1]);
#undef SWAP
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMPlexGenerate_Tetgen(DM boundary, PetscBool interpolate, DM *dm)
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
    DMLabel          glabel      = NULL;
    const PetscInt   numCorners  = 4;
    const PetscInt   numCells    = out.numberoftetrahedra;
    const PetscInt   numVertices = out.numberofpoints;
    PetscReal        *meshCoords = NULL;
    PetscInt         *cells      = NULL;

    if (sizeof (PetscReal) == sizeof (out.pointlist[0])) {
      meshCoords = (PetscReal *) out.pointlist;
    } else {
      PetscInt i;

      meshCoords = new PetscReal[dim * numVertices];
      for (i = 0; i < dim * numVertices; i++) {
        meshCoords[i] = (PetscReal) out.pointlist[i];
      }
    }
    if (sizeof (PetscInt) == sizeof (out.tetrahedronlist[0])) {
      cells = (PetscInt *) out.tetrahedronlist;
    } else {
      PetscInt i;

      cells = new PetscInt[numCells * numCorners];
      for (i = 0; i < numCells * numCorners; i++) {
        cells[i] = (PetscInt) out.tetrahedronlist[i];
      }
    }

    ierr = DMPlexInvertCells_Tetgen(numCells, numCorners, cells);CHKERRQ(ierr);
    ierr = DMPlexCreateFromCellListPetsc(comm, dim, numCells, numVertices, numCorners, interpolate, cells, dim, meshCoords, dm);CHKERRQ(ierr);
    if (label) {ierr = DMCreateLabel(*dm, labelName);CHKERRQ(ierr); ierr = DMGetLabel(*dm, labelName, &glabel);CHKERRQ(ierr);}
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

PETSC_EXTERN PetscErrorCode DMPlexRefine_Tetgen(DM dm, double *maxVolumes, DM *dmRefined)
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
    DMLabel          rlabel      = NULL;
    const PetscInt   numCorners  = 4;
    const PetscInt   numCells    = out.numberoftetrahedra;
    const PetscInt   numVertices = out.numberofpoints;
    PetscReal        *meshCoords = NULL;
    PetscInt         *cells      = NULL;
    PetscBool        interpolate = depthGlobal > 1 ? PETSC_TRUE : PETSC_FALSE;

    if (sizeof (PetscReal) == sizeof (out.pointlist[0])) {
      meshCoords = (PetscReal *) out.pointlist;
    } else {
      PetscInt i;

      meshCoords = new PetscReal[dim * numVertices];
      for (i = 0; i < dim * numVertices; i++) {
        meshCoords[i] = (PetscReal) out.pointlist[i];
      }
    }
    if (sizeof (PetscInt) == sizeof (out.tetrahedronlist[0])) {
      cells = (PetscInt *) out.tetrahedronlist;
    } else {
      PetscInt i;

      cells = new PetscInt[numCells * numCorners];
      for (i = 0; i < numCells * numCorners; i++) {
        cells[i] = (PetscInt) out.tetrahedronlist[i];
      }
    }

    ierr = DMPlexInvertCells_Tetgen(numCells, numCorners, cells);CHKERRQ(ierr);
    ierr = DMPlexCreateFromCellListPetsc(comm, dim, numCells, numVertices, numCorners, interpolate, cells, dim, meshCoords, dmRefined);CHKERRQ(ierr);
    if (label) {
      ierr = DMCreateLabel(*dmRefined, labelName);CHKERRQ(ierr);
      ierr = DMGetLabel(*dmRefined, labelName, &rlabel);CHKERRQ(ierr);
    }
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
