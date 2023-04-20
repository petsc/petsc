#define PETSCDM_DLL
#include <petsc/private/dmpleximpl.h> /*I   "petscdmplex.h"   I*/
#include <petsc/private/viewercgnsimpl.h>

#include <pcgnslib.h>
#include <cgns_io.h>

#if !defined(CGNS_ENUMT)
  #define CGNS_ENUMT(a) a
#endif
#if !defined(CGNS_ENUMV)
  #define CGNS_ENUMV(a) a
#endif

PetscErrorCode DMPlexCreateCGNSFromFile_Internal(MPI_Comm comm, const char filename[], PetscBool interpolate, DM *dm)
{
  PetscMPIInt rank;
  int         cgid = -1;

  PetscFunctionBegin;
  PetscValidCharPointer(filename, 2);
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (rank == 0) {
    PetscCallCGNS(cg_open(filename, CG_MODE_READ, &cgid));
    PetscCheck(cgid > 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "cg_open(\"%s\",...) did not return a valid file ID", filename);
  }
  PetscCall(DMPlexCreateCGNS(comm, cgid, interpolate, dm));
  if (rank == 0) PetscCallCGNS(cg_close(cgid));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexCreateCGNS_Internal(MPI_Comm comm, PetscInt cgid, PetscBool interpolate, DM *dm)
{
  PetscMPIInt  num_proc, rank;
  DM           cdm;
  DMLabel      label;
  PetscSection coordSection;
  Vec          coordinates;
  PetscScalar *coords;
  PetscInt    *cellStart, *vertStart, v;
  PetscInt     labelIdRange[2], labelId;
  /* Read from file */
  char basename[CGIO_MAX_NAME_LENGTH + 1];
  char buffer[CGIO_MAX_NAME_LENGTH + 1];
  int  dim = 0, physDim = 0, coordDim = 0, numVertices = 0, numCells = 0;
  int  nzones = 0;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &num_proc));
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));

  /* Open CGNS II file and read basic information on rank 0, then broadcast to all processors */
  if (rank == 0) {
    int nbases, z;

    PetscCallCGNS(cg_nbases(cgid, &nbases));
    PetscCheck(nbases <= 1, PETSC_COMM_SELF, PETSC_ERR_LIB, "CGNS file must have a single base, not %d", nbases);
    PetscCallCGNS(cg_base_read(cgid, 1, basename, &dim, &physDim));
    PetscCallCGNS(cg_nzones(cgid, 1, &nzones));
    PetscCall(PetscCalloc2(nzones + 1, &cellStart, nzones + 1, &vertStart));
    for (z = 1; z <= nzones; ++z) {
      cgsize_t sizes[3]; /* Number of vertices, number of cells, number of boundary vertices */

      PetscCallCGNS(cg_zone_read(cgid, 1, z, buffer, sizes));
      numVertices += sizes[0];
      numCells += sizes[1];
      cellStart[z] += sizes[1] + cellStart[z - 1];
      vertStart[z] += sizes[0] + vertStart[z - 1];
    }
    for (z = 1; z <= nzones; ++z) vertStart[z] += numCells;
    coordDim = dim;
  }
  PetscCallMPI(MPI_Bcast(basename, CGIO_MAX_NAME_LENGTH + 1, MPI_CHAR, 0, comm));
  PetscCallMPI(MPI_Bcast(&dim, 1, MPI_INT, 0, comm));
  PetscCallMPI(MPI_Bcast(&coordDim, 1, MPI_INT, 0, comm));
  PetscCallMPI(MPI_Bcast(&nzones, 1, MPI_INT, 0, comm));

  PetscCall(PetscObjectSetName((PetscObject)*dm, basename));
  PetscCall(DMSetDimension(*dm, dim));
  PetscCall(DMCreateLabel(*dm, "celltype"));
  PetscCall(DMPlexSetChart(*dm, 0, numCells + numVertices));

  /* Read zone information */
  if (rank == 0) {
    int z, c, c_loc;

    /* Read the cell set connectivity table and build mesh topology
       CGNS standard requires that cells in a zone be numbered sequentially and be pairwise disjoint. */
    /* First set sizes */
    for (z = 1, c = 0; z <= nzones; ++z) {
      CGNS_ENUMT(ZoneType_t) zonetype;
      int nsections;
      CGNS_ENUMT(ElementType_t) cellType;
      cgsize_t       start, end;
      int            nbndry, parentFlag;
      PetscInt       numCorners;
      DMPolytopeType ctype;

      PetscCallCGNS(cg_zone_type(cgid, 1, z, &zonetype));
      PetscCheck(zonetype != CGNS_ENUMV(Structured), PETSC_COMM_SELF, PETSC_ERR_LIB, "Can only handle Unstructured zones for CGNS");
      PetscCallCGNS(cg_nsections(cgid, 1, z, &nsections));
      PetscCheck(nsections <= 1, PETSC_COMM_SELF, PETSC_ERR_LIB, "CGNS file must have a single section, not %d", nsections);
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
          case CGNS_ENUMV(BAR_2):
            numCorners = 2;
            ctype      = DM_POLYTOPE_SEGMENT;
            break;
          case CGNS_ENUMV(TRI_3):
            numCorners = 3;
            ctype      = DM_POLYTOPE_TRIANGLE;
            break;
          case CGNS_ENUMV(QUAD_4):
            numCorners = 4;
            ctype      = DM_POLYTOPE_QUADRILATERAL;
            break;
          case CGNS_ENUMV(TETRA_4):
            numCorners = 4;
            ctype      = DM_POLYTOPE_TETRAHEDRON;
            break;
          case CGNS_ENUMV(PENTA_6):
            numCorners = 6;
            ctype      = DM_POLYTOPE_TRI_PRISM;
            break;
          case CGNS_ENUMV(HEXA_8):
            numCorners = 8;
            ctype      = DM_POLYTOPE_HEXAHEDRON;
            break;
          default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid cell type %d", (int)elements[off]);
          }
          PetscCall(DMPlexSetConeSize(*dm, c, numCorners));
          PetscCall(DMPlexSetCellType(*dm, c, ctype));
          off += numCorners + 1;
        }
        PetscCall(PetscFree(elements));
      } else {
        switch (cellType) {
        case CGNS_ENUMV(BAR_2):
          numCorners = 2;
          ctype      = DM_POLYTOPE_SEGMENT;
          break;
        case CGNS_ENUMV(TRI_3):
          numCorners = 3;
          ctype      = DM_POLYTOPE_TRIANGLE;
          break;
        case CGNS_ENUMV(QUAD_4):
          numCorners = 4;
          ctype      = DM_POLYTOPE_QUADRILATERAL;
          break;
        case CGNS_ENUMV(TETRA_4):
          numCorners = 4;
          ctype      = DM_POLYTOPE_TETRAHEDRON;
          break;
        case CGNS_ENUMV(PENTA_6):
          numCorners = 6;
          ctype      = DM_POLYTOPE_TRI_PRISM;
          break;
        case CGNS_ENUMV(HEXA_8):
          numCorners = 8;
          ctype      = DM_POLYTOPE_HEXAHEDRON;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid cell type %d", (int)cellType);
        }
        for (c_loc = start; c_loc <= end; ++c_loc, ++c) {
          PetscCall(DMPlexSetConeSize(*dm, c, numCorners));
          PetscCall(DMPlexSetCellType(*dm, c, ctype));
        }
      }
    }
    for (v = numCells; v < numCells + numVertices; ++v) PetscCall(DMPlexSetCellType(*dm, v, DM_POLYTOPE_POINT));
  }

  PetscCall(DMSetUp(*dm));

  PetscCall(DMCreateLabel(*dm, "zone"));
  if (rank == 0) {
    int z, c, c_loc, v_loc;

    PetscCall(DMGetLabel(*dm, "zone", &label));
    for (z = 1, c = 0; z <= nzones; ++z) {
      CGNS_ENUMT(ElementType_t) cellType;
      cgsize_t  elementDataSize, *elements, start, end;
      int       nbndry, parentFlag;
      PetscInt *cone, numc, numCorners, maxCorners = 27;

      PetscCallCGNS(cg_section_read(cgid, 1, z, 1, buffer, &cellType, &start, &end, &nbndry, &parentFlag));
      numc = end - start;
      /* This alone is reason enough to bludgeon every single CGNDS developer, this must be what they describe as the "idiocy of crowds" */
      PetscCallCGNS(cg_ElementDataSize(cgid, 1, z, 1, &elementDataSize));
      PetscCall(PetscMalloc2(elementDataSize, &elements, maxCorners, &cone));
      PetscCallCGNS(cg_poly_elements_read(cgid, 1, z, 1, elements, NULL, NULL));
      if (cellType == CGNS_ENUMV(MIXED)) {
        /* CGNS uses Fortran-based indexing, DMPlex uses C-style and numbers cell first then vertices. */
        for (c_loc = 0, v = 0; c_loc <= numc; ++c_loc, ++c) {
          switch (elements[v]) {
          case CGNS_ENUMV(BAR_2):
            numCorners = 2;
            break;
          case CGNS_ENUMV(TRI_3):
            numCorners = 3;
            break;
          case CGNS_ENUMV(QUAD_4):
            numCorners = 4;
            break;
          case CGNS_ENUMV(TETRA_4):
            numCorners = 4;
            break;
          case CGNS_ENUMV(PENTA_6):
            numCorners = 6;
            break;
          case CGNS_ENUMV(HEXA_8):
            numCorners = 8;
            break;
          default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid cell type %d", (int)elements[v]);
          }
          ++v;
          for (v_loc = 0; v_loc < numCorners; ++v_loc, ++v) cone[v_loc] = elements[v] + numCells - 1;
          PetscCall(DMPlexReorderCell(*dm, c, cone));
          PetscCall(DMPlexSetCone(*dm, c, cone));
          PetscCall(DMLabelSetValue(label, c, z));
        }
      } else {
        switch (cellType) {
        case CGNS_ENUMV(BAR_2):
          numCorners = 2;
          break;
        case CGNS_ENUMV(TRI_3):
          numCorners = 3;
          break;
        case CGNS_ENUMV(QUAD_4):
          numCorners = 4;
          break;
        case CGNS_ENUMV(TETRA_4):
          numCorners = 4;
          break;
        case CGNS_ENUMV(PENTA_6):
          numCorners = 6;
          break;
        case CGNS_ENUMV(HEXA_8):
          numCorners = 8;
          break;
        default:
          SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid cell type %d", (int)cellType);
        }
        /* CGNS uses Fortran-based indexing, DMPlex uses C-style and numbers cell first then vertices. */
        for (c_loc = 0, v = 0; c_loc <= numc; ++c_loc, ++c) {
          for (v_loc = 0; v_loc < numCorners; ++v_loc, ++v) cone[v_loc] = elements[v] + numCells - 1;
          PetscCall(DMPlexReorderCell(*dm, c, cone));
          PetscCall(DMPlexSetCone(*dm, c, cone));
          PetscCall(DMLabelSetValue(label, c, z));
        }
      }
      PetscCall(PetscFree2(elements, cone));
    }
  }

  PetscCall(DMPlexSymmetrize(*dm));
  PetscCall(DMPlexStratify(*dm));
  if (interpolate) {
    DM idm;

    PetscCall(DMPlexInterpolate(*dm, &idm));
    PetscCall(DMDestroy(dm));
    *dm = idm;
  }

  /* Read coordinates */
  PetscCall(DMSetCoordinateDim(*dm, coordDim));
  PetscCall(DMGetCoordinateDM(*dm, &cdm));
  PetscCall(DMGetLocalSection(cdm, &coordSection));
  PetscCall(PetscSectionSetNumFields(coordSection, 1));
  PetscCall(PetscSectionSetFieldComponents(coordSection, 0, coordDim));
  PetscCall(PetscSectionSetChart(coordSection, numCells, numCells + numVertices));
  for (v = numCells; v < numCells + numVertices; ++v) {
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

    PetscCall(PetscMalloc3(numVertices, &x[0], numVertices, &x[1], numVertices, &x[2]));
    for (z = 1; z <= nzones; ++z) {
      CGNS_ENUMT(DataType_t) datatype;
      cgsize_t sizes[3]; /* Number of vertices, number of cells, number of boundary vertices */
      cgsize_t range_min[3] = {1, 1, 1};
      cgsize_t range_max[3] = {1, 1, 1};
      int      ngrids, ncoords;

      PetscCallCGNS(cg_zone_read(cgid, 1, z, buffer, sizes));
      range_max[0] = sizes[0];
      PetscCallCGNS(cg_ngrids(cgid, 1, z, &ngrids));
      PetscCheck(ngrids <= 1, PETSC_COMM_SELF, PETSC_ERR_LIB, "CGNS file must have a single grid, not %d", ngrids);
      PetscCallCGNS(cg_ncoords(cgid, 1, z, &ncoords));
      PetscCheck(ncoords == coordDim, PETSC_COMM_SELF, PETSC_ERR_LIB, "CGNS file must have a coordinate array for each dimension, not %d", ncoords);
      for (d = 0; d < coordDim; ++d) {
        PetscCallCGNS(cg_coord_info(cgid, 1, z, 1 + d, &datatype, buffer));
        PetscCallCGNS(cg_coord_read(cgid, 1, z, buffer, CGNS_ENUMV(RealSingle), range_min, range_max, x[d]));
      }
      if (coordDim >= 1) {
        for (v = 0; v < sizes[0]; ++v) coords[(v + off) * coordDim + 0] = x[0][v];
      }
      if (coordDim >= 2) {
        for (v = 0; v < sizes[0]; ++v) coords[(v + off) * coordDim + 1] = x[1][v];
      }
      if (coordDim >= 3) {
        for (v = 0; v < sizes[0]; ++v) coords[(v + off) * coordDim + 2] = x[2][v];
      }
      off += sizes[0];
    }
    PetscCall(PetscFree3(x[0], x[1], x[2]));
  }
  PetscCall(VecRestoreArray(coordinates, &coords));

  PetscCall(PetscObjectSetName((PetscObject)coordinates, "coordinates"));
  PetscCall(VecSetBlockSize(coordinates, coordDim));
  PetscCall(DMSetCoordinatesLocal(*dm, coordinates));
  PetscCall(VecDestroy(&coordinates));

  /* Read boundary conditions */
  PetscCall(DMGetNumLabels(*dm, &labelIdRange[0]));
  if (rank == 0) {
    CGNS_ENUMT(BCType_t) bctype;
    CGNS_ENUMT(DataType_t) datatype;
    CGNS_ENUMT(PointSetType_t) pointtype;
    cgsize_t  *points;
    PetscReal *normals;
    int        normal[3];
    char      *bcname = buffer;
    cgsize_t   npoints, nnormals;
    int        z, nbc, bc, c, ndatasets;

    for (z = 1; z <= nzones; ++z) {
      PetscCallCGNS(cg_nbocos(cgid, 1, z, &nbc));
      for (bc = 1; bc <= nbc; ++bc) {
        PetscCallCGNS(cg_boco_info(cgid, 1, z, bc, bcname, &bctype, &pointtype, &npoints, normal, &nnormals, &datatype, &ndatasets));
        PetscCall(DMCreateLabel(*dm, bcname));
        PetscCall(DMGetLabel(*dm, bcname, &label));
        PetscCall(PetscMalloc2(npoints, &points, nnormals, &normals));
        PetscCallCGNS(cg_boco_read(cgid, 1, z, bc, points, (void *)normals));
        if (pointtype == CGNS_ENUMV(ElementRange)) {
          /* Range of cells: assuming half-open interval since the documentation sucks */
          for (c = points[0]; c < points[1]; ++c) PetscCall(DMLabelSetValue(label, c - cellStart[z - 1], 1));
        } else if (pointtype == CGNS_ENUMV(ElementList)) {
          /* List of cells */
          for (c = 0; c < npoints; ++c) PetscCall(DMLabelSetValue(label, points[c] - cellStart[z - 1], 1));
        } else if (pointtype == CGNS_ENUMV(PointRange)) {
          CGNS_ENUMT(GridLocation_t) gridloc;

          /* List of points: Oh please, someone get the CGNS developers away from a computer. This is unconscionable. */
          PetscCallCGNS(cg_goto(cgid, 1, "Zone_t", z, "BC_t", bc, "end"));
          PetscCallCGNS(cg_gridlocation_read(&gridloc));
          /* Range of points: assuming half-open interval since the documentation sucks */
          for (c = points[0]; c < points[1]; ++c) {
            if (gridloc == CGNS_ENUMV(Vertex)) PetscCall(DMLabelSetValue(label, c - vertStart[z - 1], 1));
            else PetscCall(DMLabelSetValue(label, c - cellStart[z - 1], 1));
          }
        } else if (pointtype == CGNS_ENUMV(PointList)) {
          CGNS_ENUMT(GridLocation_t) gridloc;

          /* List of points: Oh please, someone get the CGNS developers away from a computer. This is unconscionable. */
          PetscCallCGNS(cg_goto(cgid, 1, "Zone_t", z, "BC_t", bc, "end"));
          PetscCallCGNS(cg_gridlocation_read(&gridloc));
          for (c = 0; c < npoints; ++c) {
            if (gridloc == CGNS_ENUMV(Vertex)) PetscCall(DMLabelSetValue(label, points[c] - vertStart[z - 1], 1));
            else PetscCall(DMLabelSetValue(label, points[c] - cellStart[z - 1], 1));
          }
        } else SETERRQ(comm, PETSC_ERR_SUP, "Unsupported point set type %d", (int)pointtype);
        PetscCall(PetscFree2(points, normals));
      }
    }
    PetscCall(PetscFree2(cellStart, vertStart));
  }
  PetscCall(DMGetNumLabels(*dm, &labelIdRange[1]));
  PetscCallMPI(MPI_Bcast(labelIdRange, 2, MPIU_INT, 0, comm));

  /* Create BC labels at all processes */
  for (labelId = labelIdRange[0]; labelId < labelIdRange[1]; ++labelId) {
    char       *labelName = buffer;
    size_t      len       = sizeof(buffer);
    const char *locName;

    if (rank == 0) {
      PetscCall(DMGetLabelByNum(*dm, labelId, &label));
      PetscCall(PetscObjectGetName((PetscObject)label, &locName));
      PetscCall(PetscStrncpy(labelName, locName, len));
    }
    PetscCallMPI(MPI_Bcast(labelName, (PetscMPIInt)len, MPIU_INT, 0, comm));
    PetscCallMPI(DMCreateLabel(*dm, labelName));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Permute plex closure ordering to CGNS
static PetscErrorCode DMPlexCGNSGetPermutation_Internal(DMPolytopeType cell_type, PetscInt closure_size, CGNS_ENUMT(ElementType_t) * element_type, const int **perm)
{
  // https://cgns.github.io/CGNS_docs_current/sids/conv.html#unst_example
  static const int bar_2[2]   = {0, 1};
  static const int bar_3[3]   = {1, 2, 0};
  static const int bar_4[4]   = {2, 3, 0, 1};
  static const int bar_5[5]   = {3, 4, 0, 1, 2};
  static const int tri_3[3]   = {0, 1, 2};
  static const int tri_6[6]   = {3, 4, 5, 0, 1, 2};
  static const int tri_10[10] = {7, 8, 9, 1, 2, 3, 4, 5, 6, 0};
  static const int quad_4[4]  = {0, 1, 2, 3};
  static const int quad_9[9]  = {
    5, 6, 7, 8, // vertices
    1, 2, 3, 4, // edges
    0,          // center
  };
  static const int quad_16[] = {
    12, 13, 14, 15,               // vertices
    4,  5,  6,  7,  8, 9, 10, 11, // edges
    0,  1,  3,  2,                // centers
  };
  static const int quad_25[] = {
    21, 22, 23, 24,                                 // vertices
    9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, // edges
    0,  1,  2,  5,  8,  7,  6,  3,  4,              // centers
  };
  static const int tetra_4[4]   = {0, 2, 1, 3};
  static const int tetra_10[10] = {6, 8, 7, 9, 2, 1, 0, 3, 5, 4};
  static const int tetra_20[20] = {
    16, 18, 17, 19,         // vertices
    9,  8,  7,  6,  5,  4,  // bottom edges
    10, 11, 14, 15, 13, 12, // side edges
    0,  2,  3,  1,          // faces
  };
  static const int hexa_8[8]   = {0, 3, 2, 1, 4, 5, 6, 7};
  static const int hexa_27[27] = {
    19, 22, 21, 20, 23, 24, 25, 26, // vertices
    10, 9,  8,  7,                  // bottom edges
    16, 15, 18, 17,                 // mid edges
    11, 12, 13, 14,                 // top edges
    1,  3,  5,  4,  6,  2,          // faces
    0,                              // center
  };
  static const int hexa_64[64] = {
    // debug with $PETSC_ARCH/tests/dm/impls/plex/tests/ex49 -dm_plex_simplex 0 -dm_plex_dim 3 -dm_plex_box_faces 1,1,1 -dm_coord_petscspace_degree 3
    56, 59, 58, 57, 60, 61, 62, 63, // vertices
    39, 38, 37, 36, 35, 34, 33, 32, // bottom edges
    51, 50, 48, 49, 52, 53, 55, 54, // mid edges; Paraview needs edge 21-22 swapped with 23-24
    40, 41, 42, 43, 44, 45, 46, 47, // top edges
    8,  10, 11, 9,                  // z-minus face
    16, 17, 19, 18,                 // y-minus face
    24, 25, 27, 26,                 // x-plus face
    20, 21, 23, 22,                 // y-plus face
    30, 28, 29, 31,                 // x-minus face
    12, 13, 15, 14,                 // z-plus face
    0,  1,  3,  2,  4,  5,  7,  6,  // center
  };

  PetscFunctionBegin;
  *element_type = CGNS_ENUMV(ElementTypeNull);
  *perm         = NULL;
  switch (cell_type) {
  case DM_POLYTOPE_SEGMENT:
    switch (closure_size) {
    case 2:
      *element_type = CGNS_ENUMV(BAR_2);
      *perm         = bar_2;
    case 3:
      *element_type = CGNS_ENUMV(BAR_3);
      *perm         = bar_3;
    case 4:
      *element_type = CGNS_ENUMV(BAR_4);
      *perm         = bar_4;
      break;
    case 5:
      *element_type = CGNS_ENUMV(BAR_5);
      *perm         = bar_5;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cell type %s with closure size %" PetscInt_FMT, DMPolytopeTypes[cell_type], closure_size);
    }
    break;
  case DM_POLYTOPE_TRIANGLE:
    switch (closure_size) {
    case 3:
      *element_type = CGNS_ENUMV(TRI_3);
      *perm         = tri_3;
      break;
    case 6:
      *element_type = CGNS_ENUMV(TRI_6);
      *perm         = tri_6;
      break;
    case 10:
      *element_type = CGNS_ENUMV(TRI_10);
      *perm         = tri_10;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cell type %s with closure size %" PetscInt_FMT, DMPolytopeTypes[cell_type], closure_size);
    }
    break;
  case DM_POLYTOPE_QUADRILATERAL:
    switch (closure_size) {
    case 4:
      *element_type = CGNS_ENUMV(QUAD_4);
      *perm         = quad_4;
      break;
    case 9:
      *element_type = CGNS_ENUMV(QUAD_9);
      *perm         = quad_9;
      break;
    case 16:
      *element_type = CGNS_ENUMV(QUAD_16);
      *perm         = quad_16;
      break;
    case 25:
      *element_type = CGNS_ENUMV(QUAD_25);
      *perm         = quad_25;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cell type %s with closure size %" PetscInt_FMT, DMPolytopeTypes[cell_type], closure_size);
    }
    break;
  case DM_POLYTOPE_TETRAHEDRON:
    switch (closure_size) {
    case 4:
      *element_type = CGNS_ENUMV(TETRA_4);
      *perm         = tetra_4;
      break;
    case 10:
      *element_type = CGNS_ENUMV(TETRA_10);
      *perm         = tetra_10;
      break;
    case 20:
      *element_type = CGNS_ENUMV(TETRA_20);
      *perm         = tetra_20;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cell type %s with closure size %" PetscInt_FMT, DMPolytopeTypes[cell_type], closure_size);
    }
    break;
  case DM_POLYTOPE_HEXAHEDRON:
    switch (closure_size) {
    case 8:
      *element_type = CGNS_ENUMV(HEXA_8);
      *perm         = hexa_8;
      break;
    case 27:
      *element_type = CGNS_ENUMV(HEXA_27);
      *perm         = hexa_27;
      break;
    case 64:
      *element_type = CGNS_ENUMV(HEXA_64);
      *perm         = hexa_64;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cell type %s with closure size %" PetscInt_FMT, DMPolytopeTypes[cell_type], closure_size);
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cell type %s with closure size %" PetscInt_FMT, DMPolytopeTypes[cell_type], closure_size);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// node_l2g must be freed
static PetscErrorCode DMPlexCreateNodeNumbering(DM dm, PetscInt *num_local_nodes, PetscInt *num_global_nodes, PetscInt *nStart, PetscInt *nEnd, const PetscInt **node_l2g)
{
  PetscSection    local_section;
  PetscSF         point_sf;
  PetscInt        pStart, pEnd, spStart, spEnd, *points, nleaves, ncomp, *nodes;
  PetscMPIInt     comm_size;
  const PetscInt *ilocal, field = 0;

  PetscFunctionBegin;
  *num_local_nodes  = -1;
  *num_global_nodes = -1;
  *nStart           = -1;
  *nEnd             = -1;
  *node_l2g         = NULL;

  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &comm_size));
  PetscCall(DMGetLocalSection(dm, &local_section));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(PetscSectionGetChart(local_section, &spStart, &spEnd));
  PetscCall(DMGetPointSF(dm, &point_sf));
  if (comm_size == 1) nleaves = 0;
  else PetscCall(PetscSFGetGraph(point_sf, NULL, &nleaves, &ilocal, NULL));
  PetscCall(PetscSectionGetFieldComponents(local_section, field, &ncomp));

  PetscInt local_node = 0, owned_node = 0, owned_start = 0;
  for (PetscInt p = spStart, leaf = 0; p < spEnd; p++) {
    PetscInt dof;
    PetscCall(PetscSectionGetFieldDof(local_section, p, field, &dof));
    PetscAssert(dof % ncomp == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Field dof %" PetscInt_FMT " must be divisible by components %" PetscInt_FMT, dof, ncomp);
    local_node += dof / ncomp;
    if (leaf < nleaves && p == ilocal[leaf]) { // skip points owned by a different process
      leaf++;
    } else {
      owned_node += dof / ncomp;
    }
  }
  PetscCallMPI(MPI_Exscan(&owned_node, &owned_start, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)dm)));
  PetscCall(PetscMalloc1(pEnd - pStart, &points));
  owned_node = 0;
  for (PetscInt p = spStart, leaf = 0; p < spEnd; p++) {
    if (leaf < nleaves && p == ilocal[leaf]) { // skip points owned by a different process
      points[p - pStart] = -1;
      leaf++;
      continue;
    }
    PetscInt dof, offset;
    PetscCall(PetscSectionGetFieldDof(local_section, p, field, &dof));
    PetscCall(PetscSectionGetFieldOffset(local_section, p, field, &offset));
    PetscAssert(offset % ncomp == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Field offset %" PetscInt_FMT " must be divisible by components %" PetscInt_FMT, offset, ncomp);
    points[p - pStart] = owned_start + owned_node;
    owned_node += dof / ncomp;
  }
  if (comm_size > 1) {
    PetscCall(PetscSFBcastBegin(point_sf, MPIU_INT, points, points, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(point_sf, MPIU_INT, points, points, MPI_REPLACE));
  }

  // Set up global indices for each local node
  PetscCall(PetscMalloc1(local_node, &nodes));
  for (PetscInt p = spStart; p < spEnd; p++) {
    PetscInt dof, offset;
    PetscCall(PetscSectionGetFieldDof(local_section, p, field, &dof));
    PetscCall(PetscSectionGetFieldOffset(local_section, p, field, &offset));
    for (PetscInt n = 0; n < dof / ncomp; n++) nodes[offset / ncomp + n] = points[p - pStart] + n;
  }
  PetscCall(PetscFree(points));
  *num_local_nodes = local_node;
  *nStart          = owned_start;
  *nEnd            = owned_start + owned_node;
  PetscCall(MPIU_Allreduce(&owned_node, num_global_nodes, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)dm)));
  *node_l2g = nodes;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMView_PlexCGNS(DM dm, PetscViewer viewer)
{
  PetscViewer_CGNS *cgv = (PetscViewer_CGNS *)viewer->data;
  PetscInt          topo_dim, coord_dim, num_global_elems;
  PetscInt          cStart, cEnd, num_local_nodes, num_global_nodes, nStart, nEnd;
  const PetscInt   *node_l2g;
  Vec               coord;
  DM                colloc_dm, cdm;
  PetscMPIInt       size;
  const char       *dm_name;
  int               base, zone;
  cgsize_t          isize[3];

  PetscFunctionBegin;
  if (!cgv->file_num) {
    PetscInt time_step;
    PetscCall(DMGetOutputSequenceNumber(dm, &time_step, NULL));
    PetscCall(PetscViewerCGNSFileOpen_Internal(viewer, time_step));
  }
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  PetscCall(DMGetDimension(dm, &topo_dim));
  PetscCall(DMGetCoordinateDim(dm, &coord_dim));
  PetscCall(PetscObjectGetName((PetscObject)dm, &dm_name));
  PetscCallCGNS(cg_base_write(cgv->file_num, dm_name, topo_dim, coord_dim, &base));
  PetscCallCGNS(cg_goto(cgv->file_num, base, NULL));
  PetscCallCGNS(cg_dataclass_write(CGNS_ENUMV(NormalizedByDimensional)));

  {
    PetscFE        fe, fe_coord;
    PetscDualSpace dual_space, dual_space_coord;
    PetscInt       num_fields, field_order, field_order_coord;
    PetscBool      is_simplex;
    PetscCall(DMGetNumFields(dm, &num_fields));
    if (num_fields > 0) PetscCall(DMGetField(dm, 0, NULL, (PetscObject *)&fe));
    else fe = NULL;
    if (fe) {
      PetscCall(PetscFEGetDualSpace(fe, &dual_space));
      PetscCall(PetscDualSpaceGetOrder(dual_space, &field_order));
    } else field_order = 1;
    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMGetField(cdm, 0, NULL, (PetscObject *)&fe_coord));
    if (fe_coord) {
      PetscCall(PetscFEGetDualSpace(fe_coord, &dual_space_coord));
      PetscCall(PetscDualSpaceGetOrder(dual_space_coord, &field_order_coord));
    } else field_order_coord = 1;
    if (field_order != field_order_coord) {
      PetscInt quadrature_order = field_order;
      PetscCall(DMClone(dm, &colloc_dm));
      { // Inform the new colloc_dm that it is a coordinate DM so isoperiodic affine corrections can be applied
        PetscSF face_sf;
        PetscCall(DMPlexGetIsoperiodicFaceSF(dm, &face_sf));
        PetscCall(DMPlexSetIsoperiodicFaceSF(colloc_dm, face_sf));
        if (face_sf) colloc_dm->periodic.setup = DMPeriodicCoordinateSetUp_Internal;
      }
      PetscCall(DMPlexIsSimplex(dm, &is_simplex));
      PetscCall(PetscFECreateLagrange(PetscObjectComm((PetscObject)dm), topo_dim, coord_dim, is_simplex, field_order, quadrature_order, &fe));
      PetscCall(DMProjectCoordinates(colloc_dm, fe));
      PetscCall(PetscFEDestroy(&fe));
    } else {
      PetscCall(PetscObjectReference((PetscObject)dm));
      colloc_dm = dm;
    }
  }
  PetscCall(DMGetCoordinateDM(colloc_dm, &cdm));
  PetscCall(DMPlexCreateNodeNumbering(cdm, &num_local_nodes, &num_global_nodes, &nStart, &nEnd, &node_l2g));
  PetscCall(DMGetCoordinatesLocal(colloc_dm, &coord));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  num_global_elems = cEnd - cStart;
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE, &num_global_elems, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)dm)));
  isize[0] = num_global_nodes;
  isize[1] = num_global_elems;
  isize[2] = 0;
  PetscCallCGNS(cg_zone_write(cgv->file_num, base, "Zone", isize, CGNS_ENUMV(Unstructured), &zone));

  {
    const PetscScalar *X;
    PetscScalar       *x;
    int                coord_ids[3];

    PetscCall(VecGetArrayRead(coord, &X));
    for (PetscInt d = 0; d < coord_dim; d++) {
      const double exponents[] = {0, 1, 0, 0, 0};
      char         coord_name[64];
      PetscCall(PetscSNPrintf(coord_name, sizeof coord_name, "Coordinate%c", 'X' + (int)d));
      PetscCallCGNS(cgp_coord_write(cgv->file_num, base, zone, CGNS_ENUMV(RealDouble), coord_name, &coord_ids[d]));
      PetscCallCGNS(cg_goto(cgv->file_num, base, "Zone_t", zone, "GridCoordinates", 0, coord_name, 0, NULL));
      PetscCallCGNS(cg_exponents_write(CGNS_ENUMV(RealDouble), exponents));
    }

    DMPolytopeType cell_type;
    int            section;
    cgsize_t       e_owned, e_global, e_start, *conn = NULL;
    const int     *perm;
    CGNS_ENUMT(ElementType_t) element_type = CGNS_ENUMV(ElementTypeNull);
    {
      PetscCall(PetscMalloc1(nEnd - nStart, &x));
      for (PetscInt d = 0; d < coord_dim; d++) {
        for (PetscInt n = 0; n < num_local_nodes; n++) {
          PetscInt gn = node_l2g[n];
          if (gn < nStart || nEnd <= gn) continue;
          x[gn - nStart] = X[n * coord_dim + d];
        }
        // CGNS nodes use 1-based indexing
        cgsize_t start = nStart + 1, end = nEnd;
        PetscCallCGNS(cgp_coord_write_data(cgv->file_num, base, zone, coord_ids[d], &start, &end, x));
      }
      PetscCall(PetscFree(x));
      PetscCall(VecRestoreArrayRead(coord, &X));
    }

    PetscCall(DMPlexGetCellType(dm, cStart, &cell_type));
    for (PetscInt i = cStart, c = 0; i < cEnd; i++) {
      PetscInt closure_dof, *closure_indices, elem_size;
      PetscCall(DMPlexGetClosureIndices(cdm, cdm->localSection, cdm->localSection, i, PETSC_FALSE, &closure_dof, &closure_indices, NULL, NULL));
      elem_size = closure_dof / coord_dim;
      if (!conn) PetscCall(PetscMalloc1((cEnd - cStart) * elem_size, &conn));
      PetscCall(DMPlexCGNSGetPermutation_Internal(cell_type, closure_dof / coord_dim, &element_type, &perm));
      for (PetscInt j = 0; j < elem_size; j++) conn[c++] = node_l2g[closure_indices[perm[j] * coord_dim] / coord_dim] + 1;
      PetscCall(DMPlexRestoreClosureIndices(cdm, cdm->localSection, cdm->localSection, i, PETSC_FALSE, &closure_dof, &closure_indices, NULL, NULL));
    }
    e_owned = cEnd - cStart;
    PetscCall(MPIU_Allreduce(&e_owned, &e_global, 1, MPIU_INT64, MPI_SUM, PetscObjectComm((PetscObject)dm)));
    PetscCheck(e_global == num_global_elems, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected number of elements %" PetscInt64_FMT "vs %" PetscInt_FMT, e_global, num_global_elems);
    e_start = 0;
    PetscCallMPI(MPI_Exscan(&e_owned, &e_start, 1, MPIU_INT64, MPI_SUM, PetscObjectComm((PetscObject)dm)));
    PetscCallCGNS(cgp_section_write(cgv->file_num, base, zone, "Elem", element_type, 1, e_global, 0, &section));
    PetscCallCGNS(cgp_elements_write_data(cgv->file_num, base, zone, section, e_start + 1, e_start + e_owned, conn));
    PetscCall(PetscFree(conn));

    cgv->base            = base;
    cgv->zone            = zone;
    cgv->node_l2g        = node_l2g;
    cgv->num_local_nodes = num_local_nodes;
    cgv->nStart          = nStart;
    cgv->nEnd            = nEnd;
    if (1) {
      PetscMPIInt rank;
      int        *efield;
      int         sol, field;
      DMLabel     label;
      PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
      PetscCall(PetscMalloc1(e_owned, &efield));
      for (PetscInt i = 0; i < e_owned; i++) efield[i] = rank;
      PetscCallCGNS(cg_sol_write(cgv->file_num, base, zone, "CellInfo", CGNS_ENUMV(CellCenter), &sol));
      PetscCallCGNS(cgp_field_write(cgv->file_num, base, zone, sol, CGNS_ENUMV(Integer), "Rank", &field));
      cgsize_t start = e_start + 1, end = e_start + e_owned;
      PetscCallCGNS(cgp_field_write_data(cgv->file_num, base, zone, sol, field, &start, &end, efield));
      PetscCall(DMGetLabel(dm, "Cell Sets", &label));
      if (label) {
        for (PetscInt c = cStart; c < cEnd; c++) {
          PetscInt value;
          PetscCall(DMLabelGetValue(label, c, &value));
          efield[c - cStart] = value;
        }
        PetscCallCGNS(cgp_field_write(cgv->file_num, base, zone, sol, CGNS_ENUMV(Integer), "CellSet", &field));
        PetscCallCGNS(cgp_field_write_data(cgv->file_num, base, zone, sol, field, &start, &end, efield));
      }
      PetscCall(PetscFree(efield));
    }
  }
  PetscCall(DMDestroy(&colloc_dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PetscCGNSDataType(PetscDataType pd, CGNS_ENUMT(DataType_t) * cd)
{
  PetscFunctionBegin;
  switch (pd) {
  case PETSC_FLOAT:
    *cd = CGNS_ENUMV(RealSingle);
    break;
  case PETSC_DOUBLE:
    *cd = CGNS_ENUMV(RealDouble);
    break;
  case PETSC_COMPLEX:
    *cd = CGNS_ENUMV(ComplexDouble);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Data type %s", PetscDataTypes[pd]);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecView_Plex_Local_CGNS(Vec V, PetscViewer viewer)
{
  PetscViewer_CGNS  *cgv = (PetscViewer_CGNS *)viewer->data;
  DM                 dm;
  PetscSection       section;
  PetscInt           ncomp, time_step;
  PetscReal          time, *time_slot;
  size_t            *step_slot;
  const PetscInt     field = 0;
  const PetscScalar *v;
  char               solution_name[PETSC_MAX_PATH_LEN];
  int                sol;

  PetscFunctionBegin;
  PetscCall(VecGetDM(V, &dm));
  if (!cgv->node_l2g) PetscCall(DMView(dm, viewer));
  if (!cgv->nodal_field) PetscCall(PetscMalloc1(cgv->nEnd - cgv->nStart, &cgv->nodal_field));
  if (!cgv->output_times) PetscCall(PetscSegBufferCreate(sizeof(PetscReal), 20, &cgv->output_times));
  if (!cgv->output_steps) PetscCall(PetscSegBufferCreate(sizeof(size_t), 20, &cgv->output_steps));

  PetscCall(DMGetOutputSequenceNumber(dm, &time_step, &time));
  if (time_step < 0) {
    time_step = 0;
    time      = 0.;
  }
  PetscCall(PetscSegBufferGet(cgv->output_times, 1, &time_slot));
  *time_slot = time;
  PetscCall(PetscSegBufferGet(cgv->output_steps, 1, &step_slot));
  *step_slot = time_step;
  PetscCall(PetscSNPrintf(solution_name, sizeof solution_name, "FlowSolution%" PetscInt_FMT, time_step));
  PetscCallCGNS(cg_sol_write(cgv->file_num, cgv->base, cgv->zone, solution_name, CGNS_ENUMV(Vertex), &sol));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscSectionGetFieldComponents(section, field, &ncomp));
  PetscCall(VecGetArrayRead(V, &v));
  for (PetscInt comp = 0; comp < ncomp; comp++) {
    int         cgfield;
    const char *comp_name;
    CGNS_ENUMT(DataType_t) datatype;
    PetscCall(PetscSectionGetComponentName(section, field, comp, &comp_name));
    PetscCall(PetscCGNSDataType(PETSC_SCALAR, &datatype));
    PetscCallCGNS(cgp_field_write(cgv->file_num, cgv->base, cgv->zone, sol, datatype, comp_name, &cgfield));
    for (PetscInt n = 0; n < cgv->num_local_nodes; n++) {
      PetscInt gn = cgv->node_l2g[n];
      if (gn < cgv->nStart || cgv->nEnd <= gn) continue;
      cgv->nodal_field[gn - cgv->nStart] = v[n * ncomp + comp];
    }
    // CGNS nodes use 1-based indexing
    cgsize_t start = cgv->nStart + 1, end = cgv->nEnd;
    PetscCallCGNS(cgp_field_write_data(cgv->file_num, cgv->base, cgv->zone, sol, cgfield, &start, &end, cgv->nodal_field));
  }
  PetscCall(VecRestoreArrayRead(V, &v));
  PetscCall(PetscViewerCGNSCheckBatch_Internal(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}
