#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h>    /*I   "petscdmplex.h"   I*/

#undef I /* Very old CGNS stupidly uses I as a variable, which fails when using complex. Curse you idiot package managers */
#if defined(PETSC_HAVE_CGNS)
#include <cgnslib.h>
#include <cgns_io.h>
#endif
#if !defined(CGNS_ENUMT)
#define CGNS_ENUMT(a) a
#endif
#if !defined(CGNS_ENUMV)
#define CGNS_ENUMV(a) a
#endif

#define PetscCallCGNS(ierr) \
do { \
  int _cgns_ier = (ierr); \
  PetscCheck(!_cgns_ier,PETSC_COMM_SELF,PETSC_ERR_LIB,"CGNS error %d %s",_cgns_ier,cg_get_error()); \
} while (0)

/*@C
  DMPlexCreateCGNS - Create a DMPlex mesh from a CGNS file.

  Collective

  Input Parameters:
+ comm  - The MPI communicator
. filename - The name of the CGNS file
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The DM object representing the mesh

  Note: http://www.grc.nasa.gov/WWW/cgns/CGNS_docs_current/index.html

  Level: beginner

.seealso: DMPlexCreate(), DMPlexCreateCGNS(), DMPlexCreateExodus()
@*/
PetscErrorCode DMPlexCreateCGNSFromFile(MPI_Comm comm, const char filename[], PetscBool interpolate, DM *dm)
{
  PetscMPIInt    rank;
#if defined(PETSC_HAVE_CGNS)
  int cgid = -1;
#endif

  PetscFunctionBegin;
  PetscValidCharPointer(filename, 2);
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
#if defined(PETSC_HAVE_CGNS)
  if (rank == 0) {
    PetscCallCGNS(cg_open(filename, CG_MODE_READ, &cgid));
    PetscCheckFalse(cgid <= 0,PETSC_COMM_SELF, PETSC_ERR_LIB, "cg_open(\"%s\",...) did not return a valid file ID", filename);
  }
  PetscCall(DMPlexCreateCGNS(comm, cgid, interpolate, dm));
  if (rank == 0) PetscCallCGNS(cg_close(cgid));
  PetscFunctionReturn(0);
#else
  SETERRQ(comm, PETSC_ERR_SUP, "Loading meshes requires CGNS support. Reconfigure using --with-cgns-dir");
#endif
}

/*@
  DMPlexCreateCGNS - Create a DMPlex mesh from a CGNS file ID.

  Collective

  Input Parameters:
+ comm  - The MPI communicator
. cgid - The CG id associated with a file and obtained using cg_open
- interpolate - Create faces and edges in the mesh

  Output Parameter:
. dm  - The DM object representing the mesh

  Note: http://www.grc.nasa.gov/WWW/cgns/CGNS_docs_current/index.html

  Level: beginner

.seealso: DMPlexCreate(), DMPlexCreateExodus()
@*/
PetscErrorCode DMPlexCreateCGNS(MPI_Comm comm, PetscInt cgid, PetscBool interpolate, DM *dm)
{
#if defined(PETSC_HAVE_CGNS)
  PetscMPIInt    num_proc, rank;
  DM             cdm;
  DMLabel        label;
  PetscSection   coordSection;
  Vec            coordinates;
  PetscScalar   *coords;
  PetscInt      *cellStart, *vertStart, v;
  PetscInt       labelIdRange[2], labelId;
  /* Read from file */
  char basename[CGIO_MAX_NAME_LENGTH+1];
  char buffer[CGIO_MAX_NAME_LENGTH+1];
  int  dim    = 0, physDim = 0, coordDim =0, numVertices = 0, numCells = 0;
  int  nzones = 0;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_CGNS)
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &num_proc));
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));

  /* Open CGNS II file and read basic information on rank 0, then broadcast to all processors */
  if (rank == 0) {
    int nbases, z;

    PetscCallCGNS(cg_nbases(cgid, &nbases));
    PetscCheckFalse(nbases > 1,PETSC_COMM_SELF,PETSC_ERR_LIB,"CGNS file must have a single base, not %d",nbases);
    PetscCallCGNS(cg_base_read(cgid, 1, basename, &dim, &physDim));
    PetscCallCGNS(cg_nzones(cgid, 1, &nzones));
    PetscCall(PetscCalloc2(nzones+1, &cellStart, nzones+1, &vertStart));
    for (z = 1; z <= nzones; ++z) {
      cgsize_t sizes[3]; /* Number of vertices, number of cells, number of boundary vertices */

      PetscCallCGNS(cg_zone_read(cgid, 1, z, buffer, sizes));
      numVertices += sizes[0];
      numCells    += sizes[1];
      cellStart[z] += sizes[1] + cellStart[z-1];
      vertStart[z] += sizes[0] + vertStart[z-1];
    }
    for (z = 1; z <= nzones; ++z) {
      vertStart[z] += numCells;
    }
    coordDim = dim;
  }
  PetscCallMPI(MPI_Bcast(basename, CGIO_MAX_NAME_LENGTH+1, MPI_CHAR, 0, comm));
  PetscCallMPI(MPI_Bcast(&dim, 1, MPI_INT, 0, comm));
  PetscCallMPI(MPI_Bcast(&coordDim, 1, MPI_INT, 0, comm));
  PetscCallMPI(MPI_Bcast(&nzones, 1, MPI_INT, 0, comm));

  PetscCall(PetscObjectSetName((PetscObject) *dm, basename));
  PetscCall(DMSetDimension(*dm, dim));
  PetscCall(DMCreateLabel(*dm, "celltype"));
  PetscCall(DMPlexSetChart(*dm, 0, numCells+numVertices));

  /* Read zone information */
  if (rank == 0) {
    int z, c, c_loc;

    /* Read the cell set connectivity table and build mesh topology
       CGNS standard requires that cells in a zone be numbered sequentially and be pairwise disjoint. */
    /* First set sizes */
    for (z = 1, c = 0; z <= nzones; ++z) {
      CGNS_ENUMT(ZoneType_t)    zonetype;
      int                       nsections;
      CGNS_ENUMT(ElementType_t) cellType;
      cgsize_t                  start, end;
      int                       nbndry, parentFlag;
      PetscInt                  numCorners;
      DMPolytopeType            ctype;

      PetscCallCGNS(cg_zone_type(cgid, 1, z, &zonetype));
      PetscCheckFalse(zonetype == CGNS_ENUMV(Structured),PETSC_COMM_SELF,PETSC_ERR_LIB,"Can only handle Unstructured zones for CGNS");
      PetscCallCGNS(cg_nsections(cgid, 1, z, &nsections));
      PetscCheckFalse(nsections > 1,PETSC_COMM_SELF,PETSC_ERR_LIB,"CGNS file must have a single section, not %d",nsections);
      PetscCallCGNS(cg_section_read(cgid, 1, z, 1, buffer, &cellType, &start, &end, &nbndry, &parentFlag));
      /* This alone is reason enough to bludgeon every single CGNDS developer, this must be what they describe as the "idiocy of crowds" */
      if (cellType == CGNS_ENUMV(MIXED)) {
        cgsize_t elementDataSize, *elements;
        PetscInt off;

        PetscCallCGNS(cg_ElementDataSize(cgid, 1, z, 1, &elementDataSize));
        PetscCall(PetscMalloc1(elementDataSize, &elements));
        PetscCallCGNS(cg_poly_elements_read(cgid, 1, z, 1, elements, NULL, NULL));
        for (c_loc = start, off = 0; c_loc <= end; ++c_loc, ++c) {
          switch (elements[off]) {
          case CGNS_ENUMV(BAR_2):   numCorners = 2; ctype = DM_POLYTOPE_SEGMENT;       break;
          case CGNS_ENUMV(TRI_3):   numCorners = 3; ctype = DM_POLYTOPE_TRIANGLE;      break;
          case CGNS_ENUMV(QUAD_4):  numCorners = 4; ctype = DM_POLYTOPE_QUADRILATERAL; break;
          case CGNS_ENUMV(TETRA_4): numCorners = 4; ctype = DM_POLYTOPE_TETRAHEDRON;   break;
          case CGNS_ENUMV(PENTA_6): numCorners = 6; ctype = DM_POLYTOPE_TRI_PRISM;     break;
          case CGNS_ENUMV(HEXA_8):  numCorners = 8; ctype = DM_POLYTOPE_HEXAHEDRON;    break;
          default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid cell type %d", (int) elements[off]);
          }
          PetscCall(DMPlexSetConeSize(*dm, c, numCorners));
          PetscCall(DMPlexSetCellType(*dm, c, ctype));
          off += numCorners+1;
        }
        PetscCall(PetscFree(elements));
      } else {
        switch (cellType) {
        case CGNS_ENUMV(BAR_2):   numCorners = 2; ctype = DM_POLYTOPE_SEGMENT;       break;
        case CGNS_ENUMV(TRI_3):   numCorners = 3; ctype = DM_POLYTOPE_TRIANGLE;      break;
        case CGNS_ENUMV(QUAD_4):  numCorners = 4; ctype = DM_POLYTOPE_QUADRILATERAL; break;
        case CGNS_ENUMV(TETRA_4): numCorners = 4; ctype = DM_POLYTOPE_TETRAHEDRON;   break;
        case CGNS_ENUMV(PENTA_6): numCorners = 6; ctype = DM_POLYTOPE_TRI_PRISM;     break;
        case CGNS_ENUMV(HEXA_8):  numCorners = 8; ctype = DM_POLYTOPE_HEXAHEDRON;    break;
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid cell type %d", (int) cellType);
        }
        for (c_loc = start; c_loc <= end; ++c_loc, ++c) {
          PetscCall(DMPlexSetConeSize(*dm, c, numCorners));
          PetscCall(DMPlexSetCellType(*dm, c, ctype));
        }
      }
    }
    for (v = numCells; v < numCells+numVertices; ++v) {
      PetscCall(DMPlexSetCellType(*dm, v, DM_POLYTOPE_POINT));
    }
  }

  PetscCall(DMSetUp(*dm));

  PetscCall(DMCreateLabel(*dm, "zone"));
  if (rank == 0) {
    int z, c, c_loc, v_loc;

    PetscCall(DMGetLabel(*dm, "zone", &label));
    for (z = 1, c = 0; z <= nzones; ++z) {
      CGNS_ENUMT(ElementType_t)   cellType;
      cgsize_t                    elementDataSize, *elements, start, end;
      int                          nbndry, parentFlag;
      PetscInt                    *cone, numc, numCorners, maxCorners = 27;

      PetscCallCGNS(cg_section_read(cgid, 1, z, 1, buffer, &cellType, &start, &end, &nbndry, &parentFlag));
      numc = end - start;
      /* This alone is reason enough to bludgeon every single CGNDS developer, this must be what they describe as the "idiocy of crowds" */
      PetscCallCGNS(cg_ElementDataSize(cgid, 1, z, 1, &elementDataSize));
      PetscCall(PetscMalloc2(elementDataSize,&elements,maxCorners,&cone));
      PetscCallCGNS(cg_poly_elements_read(cgid, 1, z, 1, elements, NULL, NULL));
      if (cellType == CGNS_ENUMV(MIXED)) {
        /* CGNS uses Fortran-based indexing, DMPlex uses C-style and numbers cell first then vertices. */
        for (c_loc = 0, v = 0; c_loc <= numc; ++c_loc, ++c) {
          switch (elements[v]) {
          case CGNS_ENUMV(BAR_2):   numCorners = 2; break;
          case CGNS_ENUMV(TRI_3):   numCorners = 3; break;
          case CGNS_ENUMV(QUAD_4):  numCorners = 4; break;
          case CGNS_ENUMV(TETRA_4): numCorners = 4; break;
          case CGNS_ENUMV(PENTA_6): numCorners = 6; break;
          case CGNS_ENUMV(HEXA_8):  numCorners = 8; break;
          default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid cell type %d", (int) elements[v]);
          }
          ++v;
          for (v_loc = 0; v_loc < numCorners; ++v_loc, ++v) {
            cone[v_loc] = elements[v]+numCells-1;
          }
          PetscCall(DMPlexReorderCell(*dm, c, cone));
          PetscCall(DMPlexSetCone(*dm, c, cone));
          PetscCall(DMLabelSetValue(label, c, z));
        }
      } else {
        switch (cellType) {
        case CGNS_ENUMV(BAR_2):   numCorners = 2; break;
        case CGNS_ENUMV(TRI_3):   numCorners = 3; break;
        case CGNS_ENUMV(QUAD_4):  numCorners = 4; break;
        case CGNS_ENUMV(TETRA_4): numCorners = 4; break;
        case CGNS_ENUMV(PENTA_6): numCorners = 6; break;
        case CGNS_ENUMV(HEXA_8):  numCorners = 8; break;
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid cell type %d", (int) cellType);
        }
        /* CGNS uses Fortran-based indexing, DMPlex uses C-style and numbers cell first then vertices. */
        for (c_loc = 0, v = 0; c_loc <= numc; ++c_loc, ++c) {
          for (v_loc = 0; v_loc < numCorners; ++v_loc, ++v) {
            cone[v_loc] = elements[v]+numCells-1;
          }
          PetscCall(DMPlexReorderCell(*dm, c, cone));
          PetscCall(DMPlexSetCone(*dm, c, cone));
          PetscCall(DMLabelSetValue(label, c, z));
        }
      }
      PetscCall(PetscFree2(elements,cone));
    }
  }

  PetscCall(DMPlexSymmetrize(*dm));
  PetscCall(DMPlexStratify(*dm));
  if (interpolate) {
    DM idm;

    PetscCall(DMPlexInterpolate(*dm, &idm));
    PetscCall(DMDestroy(dm));
    *dm  = idm;
  }

  /* Read coordinates */
  PetscCall(DMSetCoordinateDim(*dm, coordDim));
  PetscCall(DMGetCoordinateDM(*dm, &cdm));
  PetscCall(DMGetLocalSection(cdm, &coordSection));
  PetscCall(PetscSectionSetNumFields(coordSection, 1));
  PetscCall(PetscSectionSetFieldComponents(coordSection, 0, coordDim));
  PetscCall(PetscSectionSetChart(coordSection, numCells, numCells + numVertices));
  for (v = numCells; v < numCells+numVertices; ++v) {
    PetscCall(PetscSectionSetDof(coordSection, v, dim));
    PetscCall(PetscSectionSetFieldDof(coordSection, v, 0, coordDim));
  }
  PetscCall(PetscSectionSetUp(coordSection));

  PetscCall(DMCreateLocalVector(cdm, &coordinates));
  PetscCall(VecGetArray(coordinates, &coords));
  if (rank == 0) {
    PetscInt off = 0;
    float   *x[3];
    int      z, d;

    PetscCall(PetscMalloc3(numVertices,&x[0],numVertices,&x[1],numVertices,&x[2]));
    for (z = 1; z <= nzones; ++z) {
      CGNS_ENUMT(DataType_t) datatype;
      cgsize_t               sizes[3]; /* Number of vertices, number of cells, number of boundary vertices */
      cgsize_t               range_min[3] = {1, 1, 1};
      cgsize_t               range_max[3] = {1, 1, 1};
      int                    ngrids, ncoords;

      PetscCallCGNS(cg_zone_read(cgid, 1, z, buffer, sizes));
      range_max[0] = sizes[0];
      PetscCallCGNS(cg_ngrids(cgid, 1, z, &ngrids));
      PetscCheckFalse(ngrids > 1,PETSC_COMM_SELF,PETSC_ERR_LIB,"CGNS file must have a single grid, not %d",ngrids);
      PetscCallCGNS(cg_ncoords(cgid, 1, z, &ncoords));
      PetscCheckFalse(ncoords != coordDim,PETSC_COMM_SELF,PETSC_ERR_LIB,"CGNS file must have a coordinate array for each dimension, not %d",ncoords);
      for (d = 0; d < coordDim; ++d) {
        PetscCallCGNS(cg_coord_info(cgid, 1, z, 1+d, &datatype, buffer));
        PetscCallCGNS(cg_coord_read(cgid, 1, z, buffer, CGNS_ENUMV(RealSingle), range_min, range_max, x[d]));
      }
      if (coordDim >= 1) {
        for (v = 0; v < sizes[0]; ++v) coords[(v+off)*coordDim+0] = x[0][v];
      }
      if (coordDim >= 2) {
        for (v = 0; v < sizes[0]; ++v) coords[(v+off)*coordDim+1] = x[1][v];
      }
      if (coordDim >= 3) {
        for (v = 0; v < sizes[0]; ++v) coords[(v+off)*coordDim+2] = x[2][v];
      }
      off += sizes[0];
    }
    PetscCall(PetscFree3(x[0],x[1],x[2]));
  }
  PetscCall(VecRestoreArray(coordinates, &coords));

  PetscCall(PetscObjectSetName((PetscObject) coordinates, "coordinates"));
  PetscCall(VecSetBlockSize(coordinates, coordDim));
  PetscCall(DMSetCoordinatesLocal(*dm, coordinates));
  PetscCall(VecDestroy(&coordinates));

  /* Read boundary conditions */
  PetscCall(DMGetNumLabels(*dm, &labelIdRange[0]));
  if (rank == 0) {
    CGNS_ENUMT(BCType_t)        bctype;
    CGNS_ENUMT(DataType_t)      datatype;
    CGNS_ENUMT(PointSetType_t)  pointtype;
    cgsize_t                    *points;
    PetscReal                   *normals;
    int                         normal[3];
    char                        *bcname = buffer;
    cgsize_t                    npoints, nnormals;
    int                         z, nbc, bc, c, ndatasets;

    for (z = 1; z <= nzones; ++z) {
      PetscCallCGNS(cg_nbocos(cgid, 1, z, &nbc));
      for (bc = 1; bc <= nbc; ++bc) {
        PetscCallCGNS(cg_boco_info(cgid, 1, z, bc, bcname, &bctype, &pointtype, &npoints, normal, &nnormals, &datatype, &ndatasets));
        PetscCall(DMCreateLabel(*dm, bcname));
        PetscCall(DMGetLabel(*dm, bcname, &label));
        PetscCall(PetscMalloc2(npoints, &points, nnormals, &normals));
        PetscCallCGNS(cg_boco_read(cgid, 1, z, bc, points, (void *) normals));
        if (pointtype == CGNS_ENUMV(ElementRange)) {
          /* Range of cells: assuming half-open interval since the documentation sucks */
          for (c = points[0]; c < points[1]; ++c) {
            PetscCall(DMLabelSetValue(label, c - cellStart[z-1], 1));
          }
        } else if (pointtype == CGNS_ENUMV(ElementList)) {
          /* List of cells */
          for (c = 0; c < npoints; ++c) {
            PetscCall(DMLabelSetValue(label, points[c] - cellStart[z-1], 1));
          }
        } else if (pointtype == CGNS_ENUMV(PointRange)) {
          CGNS_ENUMT(GridLocation_t) gridloc;

          /* List of points: Oh please, someone get the CGNS developers away from a computer. This is unconscionable. */
          PetscCallCGNS(cg_goto(cgid, 1, "Zone_t", z, "BC_t", bc, "end"));
          PetscCallCGNS(cg_gridlocation_read(&gridloc));
          /* Range of points: assuming half-open interval since the documentation sucks */
          for (c = points[0]; c < points[1]; ++c) {
            if (gridloc == CGNS_ENUMV(Vertex)) PetscCall(DMLabelSetValue(label, c - vertStart[z-1], 1));
            else                               PetscCall(DMLabelSetValue(label, c - cellStart[z-1], 1));
          }
        } else if (pointtype == CGNS_ENUMV(PointList)) {
          CGNS_ENUMT(GridLocation_t) gridloc;

          /* List of points: Oh please, someone get the CGNS developers away from a computer. This is unconscionable. */
          PetscCallCGNS(cg_goto(cgid, 1, "Zone_t", z, "BC_t", bc, "end"));
          PetscCallCGNS(cg_gridlocation_read(&gridloc));
          for (c = 0; c < npoints; ++c) {
            if (gridloc == CGNS_ENUMV(Vertex)) PetscCall(DMLabelSetValue(label, points[c] - vertStart[z-1], 1));
            else                               PetscCall(DMLabelSetValue(label, points[c] - cellStart[z-1], 1));
          }
        } else SETERRQ(comm, PETSC_ERR_SUP, "Unsupported point set type %d", (int) pointtype);
        PetscCall(PetscFree2(points, normals));
      }
    }
    PetscCall(PetscFree2(cellStart, vertStart));
  }
  PetscCall(DMGetNumLabels(*dm, &labelIdRange[1]));
  PetscCallMPI(MPI_Bcast(labelIdRange, 2, MPIU_INT, 0, comm));

  /* Create BC labels at all processes */
  for (labelId = labelIdRange[0]; labelId < labelIdRange[1]; ++labelId) {
    char *labelName = buffer;
    size_t len = sizeof(buffer);
    const char *locName;

    if (rank == 0) {
      PetscCall(DMGetLabelByNum(*dm, labelId, &label));
      PetscCall(PetscObjectGetName((PetscObject)label, &locName));
      PetscCall(PetscStrncpy(labelName, locName, len));
    }
    PetscCallMPI(MPI_Bcast(labelName, (PetscMPIInt)len, MPIU_INT, 0, comm));
    PetscCallMPI(DMCreateLabel(*dm, labelName));
  }
  PetscFunctionReturn(0);
#else
  SETERRQ(comm, PETSC_ERR_SUP, "This method requires CGNS support. Reconfigure using --with-cgns-dir");
#endif
}
