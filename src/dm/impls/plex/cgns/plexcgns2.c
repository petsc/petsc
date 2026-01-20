#include <petsc/private/dmpleximpl.h> /*I   "petscdmplex.h"   I*/
#include <petsc/private/sfimpl.h>     /*I "petscsf.h" I*/
#include <petsc/private/viewercgnsimpl.h>
#include <petsc/private/hashseti.h>

#include <pcgnslib.h>
#include <cgns_io.h>

#if !defined(CGNS_ENUMT)
  #define CGNS_ENUMT(a) a
#endif
#if !defined(CGNS_ENUMV)
  #define CGNS_ENUMV(a) a
#endif
// Permute plex closure ordering to CGNS
static PetscErrorCode DMPlexCGNSGetPermutation_Internal(DMPolytopeType cell_type, PetscInt closure_size, CGNS_ENUMT(ElementType_t) * element_type, const int **perm)
{
  CGNS_ENUMT(ElementType_t) element_type_tmp;

  // https://cgns.github.io/CGNS_docs_current/sids/conv.html#unstructgrid
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
  element_type_tmp = CGNS_ENUMV(ElementTypeNull);
  *perm            = NULL;
  switch (cell_type) {
  case DM_POLYTOPE_SEGMENT:
    switch (closure_size) {
    case 2:
      element_type_tmp = CGNS_ENUMV(BAR_2);
      *perm            = bar_2;
      break;
    case 3:
      element_type_tmp = CGNS_ENUMV(BAR_3);
      *perm            = bar_3;
      break;
    case 4:
      element_type_tmp = CGNS_ENUMV(BAR_4);
      *perm            = bar_4;
      break;
    case 5:
      element_type_tmp = CGNS_ENUMV(BAR_5);
      *perm            = bar_5;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cell type %s with closure size %" PetscInt_FMT, DMPolytopeTypes[cell_type], closure_size);
    }
    break;
  case DM_POLYTOPE_TRIANGLE:
    switch (closure_size) {
    case 3:
      element_type_tmp = CGNS_ENUMV(TRI_3);
      *perm            = tri_3;
      break;
    case 6:
      element_type_tmp = CGNS_ENUMV(TRI_6);
      *perm            = tri_6;
      break;
    case 10:
      element_type_tmp = CGNS_ENUMV(TRI_10);
      *perm            = tri_10;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cell type %s with closure size %" PetscInt_FMT, DMPolytopeTypes[cell_type], closure_size);
    }
    break;
  case DM_POLYTOPE_QUADRILATERAL:
    switch (closure_size) {
    case 4:
      element_type_tmp = CGNS_ENUMV(QUAD_4);
      *perm            = quad_4;
      break;
    case 9:
      element_type_tmp = CGNS_ENUMV(QUAD_9);
      *perm            = quad_9;
      break;
    case 16:
      element_type_tmp = CGNS_ENUMV(QUAD_16);
      *perm            = quad_16;
      break;
    case 25:
      element_type_tmp = CGNS_ENUMV(QUAD_25);
      *perm            = quad_25;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cell type %s with closure size %" PetscInt_FMT, DMPolytopeTypes[cell_type], closure_size);
    }
    break;
  case DM_POLYTOPE_TETRAHEDRON:
    switch (closure_size) {
    case 4:
      element_type_tmp = CGNS_ENUMV(TETRA_4);
      *perm            = tetra_4;
      break;
    case 10:
      element_type_tmp = CGNS_ENUMV(TETRA_10);
      *perm            = tetra_10;
      break;
    case 20:
      element_type_tmp = CGNS_ENUMV(TETRA_20);
      *perm            = tetra_20;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cell type %s with closure size %" PetscInt_FMT, DMPolytopeTypes[cell_type], closure_size);
    }
    break;
  case DM_POLYTOPE_HEXAHEDRON:
    switch (closure_size) {
    case 8:
      element_type_tmp = CGNS_ENUMV(HEXA_8);
      *perm            = hexa_8;
      break;
    case 27:
      element_type_tmp = CGNS_ENUMV(HEXA_27);
      *perm            = hexa_27;
      break;
    case 64:
      element_type_tmp = CGNS_ENUMV(HEXA_64);
      *perm            = hexa_64;
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cell type %s with closure size %" PetscInt_FMT, DMPolytopeTypes[cell_type], closure_size);
    }
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cell type %s with closure size %" PetscInt_FMT, DMPolytopeTypes[cell_type], closure_size);
  }
  if (element_type) *element_type = element_type_tmp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Input Parameters:
+ cellType  - The CGNS-defined element type

  Output Parameters:
+ dmcelltype  - The equivalent DMPolytopeType for the cellType
. numCorners - Number of corners of the polytope
- dim - The topological dimension of the polytope

CGNS elements defined in: https://cgns.github.io/CGNS_docs_current/sids/conv.html#unstructgrid
*/
static inline PetscErrorCode CGNSElementTypeGetTopologyInfo(CGNS_ENUMT(ElementType_t) cellType, DMPolytopeType *dmcelltype, PetscInt *numCorners, PetscInt *dim)
{
  DMPolytopeType _dmcelltype;

  PetscFunctionBegin;
  switch (cellType) {
  case CGNS_ENUMV(BAR_2):
  case CGNS_ENUMV(BAR_3):
  case CGNS_ENUMV(BAR_4):
  case CGNS_ENUMV(BAR_5):
    _dmcelltype = DM_POLYTOPE_SEGMENT;
    break;
  case CGNS_ENUMV(TRI_3):
  case CGNS_ENUMV(TRI_6):
  case CGNS_ENUMV(TRI_9):
  case CGNS_ENUMV(TRI_10):
  case CGNS_ENUMV(TRI_12):
  case CGNS_ENUMV(TRI_15):
    _dmcelltype = DM_POLYTOPE_TRIANGLE;
    break;
  case CGNS_ENUMV(QUAD_4):
  case CGNS_ENUMV(QUAD_8):
  case CGNS_ENUMV(QUAD_9):
  case CGNS_ENUMV(QUAD_12):
  case CGNS_ENUMV(QUAD_16):
  case CGNS_ENUMV(QUAD_P4_16):
  case CGNS_ENUMV(QUAD_25):
    _dmcelltype = DM_POLYTOPE_QUADRILATERAL;
    break;
  case CGNS_ENUMV(TETRA_4):
  case CGNS_ENUMV(TETRA_10):
  case CGNS_ENUMV(TETRA_16):
  case CGNS_ENUMV(TETRA_20):
  case CGNS_ENUMV(TETRA_22):
  case CGNS_ENUMV(TETRA_34):
  case CGNS_ENUMV(TETRA_35):
    _dmcelltype = DM_POLYTOPE_TETRAHEDRON;
    break;
  case CGNS_ENUMV(PYRA_5):
  case CGNS_ENUMV(PYRA_13):
  case CGNS_ENUMV(PYRA_14):
  case CGNS_ENUMV(PYRA_21):
  case CGNS_ENUMV(PYRA_29):
  case CGNS_ENUMV(PYRA_P4_29):
  case CGNS_ENUMV(PYRA_30):
  case CGNS_ENUMV(PYRA_50):
  case CGNS_ENUMV(PYRA_55):
    _dmcelltype = DM_POLYTOPE_PYRAMID;
    break;
  case CGNS_ENUMV(PENTA_6):
  case CGNS_ENUMV(PENTA_15):
  case CGNS_ENUMV(PENTA_18):
  case CGNS_ENUMV(PENTA_24):
  case CGNS_ENUMV(PENTA_33):
  case CGNS_ENUMV(PENTA_38):
  case CGNS_ENUMV(PENTA_40):
  case CGNS_ENUMV(PENTA_66):
  case CGNS_ENUMV(PENTA_75):
    _dmcelltype = DM_POLYTOPE_TRI_PRISM;
    break;
  case CGNS_ENUMV(HEXA_8):
  case CGNS_ENUMV(HEXA_20):
  case CGNS_ENUMV(HEXA_27):
  case CGNS_ENUMV(HEXA_32):
  case CGNS_ENUMV(HEXA_44):
  case CGNS_ENUMV(HEXA_56):
  case CGNS_ENUMV(HEXA_64):
  case CGNS_ENUMV(HEXA_98):
  case CGNS_ENUMV(HEXA_125):
    _dmcelltype = DM_POLYTOPE_HEXAHEDRON;
    break;
  case CGNS_ENUMV(MIXED):
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid CGNS ElementType_t: MIXED");
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid CGNS ElementType_t: %d", (int)cellType);
  }

  if (dmcelltype) *dmcelltype = _dmcelltype;
  if (numCorners) *numCorners = DMPolytopeTypeGetNumVertices(_dmcelltype);
  if (dim) *dim = DMPolytopeTypeGetDim(_dmcelltype);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Input Parameters:
+ cellType  - The CGNS-defined cell type

  Output Parameters:
+ numClosure - Number of nodes that define the function space on the cell
- pOrder - The polynomial order of the cell

CGNS elements defined in: https://cgns.github.io/CGNS_docs_current/sids/conv.html#unstructgrid

Note: we only support "full" elements, ie. not seredipity elements
*/
static inline PetscErrorCode CGNSElementTypeGetDiscretizationInfo(CGNS_ENUMT(ElementType_t) cellType, PetscInt *numClosure, PetscInt *pOrder)
{
  PetscInt _numClosure, _pOrder;

  PetscFunctionBegin;
  switch (cellType) {
  case CGNS_ENUMV(BAR_2):
    _numClosure = 2;
    _pOrder     = 1;
    break;
  case CGNS_ENUMV(BAR_3):
    _numClosure = 3;
    _pOrder     = 2;
    break;
  case CGNS_ENUMV(BAR_4):
    _numClosure = 4;
    _pOrder     = 3;
    break;
  case CGNS_ENUMV(BAR_5):
    _numClosure = 5;
    _pOrder     = 4;
    break;
  case CGNS_ENUMV(TRI_3):
    _numClosure = 3;
    _pOrder     = 1;
    break;
  case CGNS_ENUMV(TRI_6):
    _numClosure = 6;
    _pOrder     = 2;
    break;
  case CGNS_ENUMV(TRI_10):
    _numClosure = 10;
    _pOrder     = 3;
    break;
  case CGNS_ENUMV(TRI_15):
    _numClosure = 15;
    _pOrder     = 4;
    break;
  case CGNS_ENUMV(QUAD_4):
    _numClosure = 4;
    _pOrder     = 1;
    break;
  case CGNS_ENUMV(QUAD_9):
    _numClosure = 9;
    _pOrder     = 2;
    break;
  case CGNS_ENUMV(QUAD_16):
    _numClosure = 16;
    _pOrder     = 3;
    break;
  case CGNS_ENUMV(QUAD_25):
    _numClosure = 25;
    _pOrder     = 4;
    break;
  case CGNS_ENUMV(TETRA_4):
    _numClosure = 4;
    _pOrder     = 1;
    break;
  case CGNS_ENUMV(TETRA_10):
    _numClosure = 10;
    _pOrder     = 2;
    break;
  case CGNS_ENUMV(TETRA_20):
    _numClosure = 20;
    _pOrder     = 3;
    break;
  case CGNS_ENUMV(TETRA_35):
    _numClosure = 35;
    _pOrder     = 4;
    break;
  case CGNS_ENUMV(PYRA_5):
    _numClosure = 5;
    _pOrder     = 1;
    break;
  case CGNS_ENUMV(PYRA_14):
    _numClosure = 14;
    _pOrder     = 2;
    break;
  case CGNS_ENUMV(PYRA_30):
    _numClosure = 30;
    _pOrder     = 3;
    break;
  case CGNS_ENUMV(PYRA_55):
    _numClosure = 55;
    _pOrder     = 4;
    break;
  case CGNS_ENUMV(PENTA_6):
    _numClosure = 6;
    _pOrder     = 1;
    break;
  case CGNS_ENUMV(PENTA_18):
    _numClosure = 18;
    _pOrder     = 2;
    break;
  case CGNS_ENUMV(PENTA_40):
    _numClosure = 40;
    _pOrder     = 3;
    break;
  case CGNS_ENUMV(PENTA_75):
    _numClosure = 75;
    _pOrder     = 4;
    break;
  case CGNS_ENUMV(HEXA_8):
    _numClosure = 8;
    _pOrder     = 1;
    break;
  case CGNS_ENUMV(HEXA_27):
    _numClosure = 27;
    _pOrder     = 2;
    break;
  case CGNS_ENUMV(HEXA_64):
    _numClosure = 64;
    _pOrder     = 3;
    break;
  case CGNS_ENUMV(HEXA_125):
    _numClosure = 125;
    _pOrder     = 4;
    break;
  case CGNS_ENUMV(MIXED):
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid CGNS ElementType_t: MIXED");
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported or Invalid cell type %d", (int)cellType);
  }
  if (numClosure) *numClosure = _numClosure;
  if (pOrder) *pOrder = _pOrder;
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

PetscErrorCode DMPlexCreateCGNSFromFile_Internal(MPI_Comm comm, const char filename[], PetscBool interpolate, DM *dm)
{
  int       cgid                = -1;
  PetscBool use_parallel_viewer = PETSC_FALSE;

  PetscFunctionBegin;
  PetscAssertPointer(filename, 2);
  PetscCall(PetscViewerCGNSRegisterLogEvents_Internal());
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-dm_plex_cgns_parallel", &use_parallel_viewer, NULL));

  if (use_parallel_viewer) {
    PetscCallCGNS(cgp_mpi_comm(comm));
    PetscCallCGNSOpen(cgp_open(filename, CG_MODE_READ, &cgid), 0, 0);
    PetscCheck(cgid > 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "cgp_open(\"%s\",...) did not return a valid file ID", filename);
    PetscCall(DMPlexCreateCGNS(comm, cgid, interpolate, dm));
    PetscCallCGNSClose(cgp_close(cgid), 0, 0);
  } else {
    PetscCallCGNSOpen(cg_open(filename, CG_MODE_READ, &cgid), 0, 0);
    PetscCheck(cgid > 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "cg_open(\"%s\",...) did not return a valid file ID", filename);
    PetscCall(DMPlexCreateCGNS(comm, cgid, interpolate, dm));
    PetscCallCGNSClose(cg_close(cgid), 0, 0);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMPlexCreateCGNS_Internal_Serial(MPI_Comm comm, PetscInt cgid, PetscBool interpolate, DM *dm)
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
  char      basename[CGIO_MAX_NAME_LENGTH + 1];
  char      buffer[CGIO_MAX_NAME_LENGTH + 1];
  int       dim = 0, physDim = 0, coordDim = 0, numVertices = 0, numCells = 0;
  int       nzones = 0;
  const int B      = 1; // Only support single base

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &num_proc));
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));

  /* Open CGNS II file and read basic information on rank 0, then broadcast to all processors */
  if (rank == 0) {
    int nbases, z;

    PetscCallCGNSRead(cg_nbases(cgid, &nbases), *dm, 0);
    PetscCheck(nbases <= 1, PETSC_COMM_SELF, PETSC_ERR_LIB, "CGNS file must have a single base, not %d", nbases);
    PetscCallCGNSRead(cg_base_read(cgid, B, basename, &dim, &physDim), *dm, 0);
    PetscCallCGNSRead(cg_nzones(cgid, B, &nzones), *dm, 0);
    PetscCall(PetscCalloc2(nzones + 1, &cellStart, nzones + 1, &vertStart));
    for (z = 1; z <= nzones; ++z) {
      cgsize_t sizes[3]; /* Number of vertices, number of cells, number of boundary vertices */

      PetscCallCGNSRead(cg_zone_read(cgid, B, z, buffer, sizes), *dm, 0);
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
      PetscInt       numCorners, pOrder;
      DMPolytopeType ctype;
      const int      S = 1; // Only support single section

      PetscCallCGNSRead(cg_zone_type(cgid, B, z, &zonetype), *dm, 0);
      PetscCheck(zonetype != CGNS_ENUMV(Structured), PETSC_COMM_SELF, PETSC_ERR_LIB, "Can only handle Unstructured zones for CGNS");
      PetscCallCGNSRead(cg_nsections(cgid, B, z, &nsections), *dm, 0);
      PetscCheck(nsections <= 1, PETSC_COMM_SELF, PETSC_ERR_LIB, "CGNS file must have a single section, not %d", nsections);
      PetscCallCGNSRead(cg_section_read(cgid, B, z, S, buffer, &cellType, &start, &end, &nbndry, &parentFlag), *dm, 0);
      if (cellType == CGNS_ENUMV(MIXED)) {
        cgsize_t elementDataSize, *elements;
        PetscInt off;

        PetscCallCGNSRead(cg_ElementDataSize(cgid, B, z, S, &elementDataSize), *dm, 0);
        PetscCall(PetscMalloc1(elementDataSize, &elements));
        PetscCallCGNSReadData(cg_poly_elements_read(cgid, B, z, S, elements, NULL, NULL), *dm, 0);
        for (c_loc = start, off = 0; c_loc <= end; ++c_loc, ++c) {
          PetscCall(CGNSElementTypeGetTopologyInfo((CGNS_ENUMT(ElementType_t))elements[off], &ctype, &numCorners, NULL));
          PetscCall(CGNSElementTypeGetDiscretizationInfo((CGNS_ENUMT(ElementType_t))elements[off], NULL, &pOrder));
          PetscCheck(pOrder == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Serial CGNS reader only supports first order elements, not %" PetscInt_FMT " order", pOrder);
          PetscCall(DMPlexSetConeSize(*dm, c, numCorners));
          PetscCall(DMPlexSetCellType(*dm, c, ctype));
          off += numCorners + 1;
        }
        PetscCall(PetscFree(elements));
      } else {
        PetscCall(CGNSElementTypeGetTopologyInfo(cellType, &ctype, &numCorners, NULL));
        PetscCall(CGNSElementTypeGetDiscretizationInfo(cellType, NULL, &pOrder));
        PetscCheck(pOrder == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Serial CGNS reader only supports first order elements, not %" PetscInt_FMT " order", pOrder);
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
      PetscInt *cone, numc, numCorners, maxCorners = 27, pOrder;
      const int S = 1; // Only support single section

      PetscCallCGNSRead(cg_section_read(cgid, B, z, S, buffer, &cellType, &start, &end, &nbndry, &parentFlag), *dm, 0);
      numc = end - start;
      PetscCallCGNSRead(cg_ElementDataSize(cgid, B, z, S, &elementDataSize), *dm, 0);
      PetscCall(PetscMalloc2(elementDataSize, &elements, maxCorners, &cone));
      PetscCallCGNSReadData(cg_poly_elements_read(cgid, B, z, S, elements, NULL, NULL), *dm, 0);
      if (cellType == CGNS_ENUMV(MIXED)) {
        /* CGNS uses Fortran-based indexing, DMPlex uses C-style and numbers cell first then vertices. */
        for (c_loc = 0, v = 0; c_loc <= numc; ++c_loc, ++c) {
          PetscCall(CGNSElementTypeGetTopologyInfo((CGNS_ENUMT(ElementType_t))elements[v], NULL, &numCorners, NULL));
          PetscCall(CGNSElementTypeGetDiscretizationInfo((CGNS_ENUMT(ElementType_t))elements[v], NULL, &pOrder));
          PetscCheck(pOrder == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Serial CGNS reader only supports first order elements, not %" PetscInt_FMT " order", pOrder);
          ++v;
          for (v_loc = 0; v_loc < numCorners; ++v_loc, ++v) cone[v_loc] = elements[v] + numCells - 1;
          PetscCall(DMPlexReorderCell(*dm, c, cone));
          PetscCall(DMPlexSetCone(*dm, c, cone));
          PetscCall(DMLabelSetValue(label, c, z));
        }
      } else {
        PetscCall(CGNSElementTypeGetTopologyInfo(cellType, NULL, &numCorners, NULL));
        PetscCall(CGNSElementTypeGetDiscretizationInfo(cellType, NULL, &pOrder));
        PetscCheck(pOrder == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Serial CGNS reader only supports first order elements, not %" PetscInt_FMT " order", pOrder);
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
  if (interpolate) PetscCall(DMPlexInterpolateInPlace_Internal(*dm));

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

      PetscCallCGNSRead(cg_zone_read(cgid, B, z, buffer, sizes), *dm, 0);
      range_max[0] = sizes[0];
      PetscCallCGNSRead(cg_ngrids(cgid, B, z, &ngrids), *dm, 0);
      PetscCheck(ngrids <= 1, PETSC_COMM_SELF, PETSC_ERR_LIB, "CGNS file must have a single grid, not %d", ngrids);
      PetscCallCGNSRead(cg_ncoords(cgid, B, z, &ncoords), *dm, 0);
      PetscCheck(ncoords == coordDim, PETSC_COMM_SELF, PETSC_ERR_LIB, "CGNS file must have a coordinate array for each dimension, not %d", ncoords);
      for (d = 0; d < coordDim; ++d) {
        PetscCallCGNSRead(cg_coord_info(cgid, B, z, 1 + d, &datatype, buffer), *dm, 0);
        PetscCallCGNSReadData(cg_coord_read(cgid, B, z, buffer, CGNS_ENUMV(RealSingle), range_min, range_max, x[d]), *dm, 0);
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
      PetscCallCGNSRead(cg_nbocos(cgid, B, z, &nbc), *dm, 0);
      for (bc = 1; bc <= nbc; ++bc) {
        PetscCallCGNSRead(cg_boco_info(cgid, B, z, bc, bcname, &bctype, &pointtype, &npoints, normal, &nnormals, &datatype, &ndatasets), *dm, 0);
        PetscCall(DMCreateLabel(*dm, bcname));
        PetscCall(DMGetLabel(*dm, bcname, &label));
        PetscCall(PetscMalloc2(npoints, &points, nnormals, &normals));
        PetscCallCGNSReadData(cg_boco_read(cgid, B, z, bc, points, (void *)normals), *dm, 0);
        if (pointtype == CGNS_ENUMV(ElementRange)) {
          // Range of cells: assuming half-open interval
          for (c = points[0]; c < points[1]; ++c) PetscCall(DMLabelSetValue(label, c - cellStart[z - 1], 1));
        } else if (pointtype == CGNS_ENUMV(ElementList)) {
          // List of cells
          for (c = 0; c < npoints; ++c) PetscCall(DMLabelSetValue(label, points[c] - cellStart[z - 1], 1));
        } else if (pointtype == CGNS_ENUMV(PointRange)) {
          CGNS_ENUMT(GridLocation_t) gridloc;

          // List of points:
          PetscCallCGNS(cg_goto(cgid, 1, "Zone_t", z, "BC_t", bc, "end"));
          PetscCallCGNSRead(cg_gridlocation_read(&gridloc), *dm, 0);
          // Range of points: assuming half-open interval
          for (c = points[0]; c < points[1]; ++c) {
            if (gridloc == CGNS_ENUMV(Vertex)) PetscCall(DMLabelSetValue(label, c - vertStart[z - 1], 1));
            else PetscCall(DMLabelSetValue(label, c - cellStart[z - 1], 1));
          }
        } else if (pointtype == CGNS_ENUMV(PointList)) {
          CGNS_ENUMT(GridLocation_t) gridloc;

          // List of points:
          PetscCallCGNS(cg_goto(cgid, 1, "Zone_t", z, "BC_t", bc, "end"));
          PetscCallCGNSRead(cg_gridlocation_read(&gridloc), *dm, 0);
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

typedef struct {
  cgsize_t start, end;
} CGRange;

// Creates a PetscLayout from the given sizes, but adjusts the ranges by the offset. So the first rank ranges will be [offset, offset + local_size) rather than [0, local_size)
static PetscErrorCode PetscLayoutCreateFromSizesAndOffset(MPI_Comm comm, PetscInt n, PetscInt N, PetscInt bs, PetscInt offset, PetscLayout *map)
{
  PetscLayout     init;
  const PetscInt *ranges;
  PetscInt       *new_ranges;
  PetscMPIInt     num_ranks;

  PetscFunctionBegin;
  PetscCall(PetscLayoutCreateFromSizes(comm, n, N, bs, &init));
  PetscCall(PetscLayoutGetRanges(init, &ranges));
  PetscCallMPI(MPI_Comm_size(comm, &num_ranks));
  PetscCall(PetscMalloc1(num_ranks + 1, &new_ranges));
  PetscCall(PetscArraycpy(new_ranges, ranges, num_ranks + 1));
  for (PetscInt r = 0; r < num_ranks + 1; r++) new_ranges[r] += offset;
  PetscCall(PetscLayoutCreateFromRanges(comm, new_ranges, PETSC_OWN_POINTER, bs, map));
  PetscCall(PetscLayoutDestroy(&init));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Creates connectivity array of CGNS elements' corners in Plex ordering, with a `PetscSection` to describe the data layout

  @param[in]  dm           `DM`
  @param[in]  cgid         CGNS file ID
  @param[in]  base         CGNS base ID
  @param[in]  zone         CGNS zone ID
  @param[in]  num_sections Number of sections to put in connectivity
  @param[in]  section_ids  CGNS section IDs to obtain connectivity from
  @param[out] section      Section describing the connectivity for each element
  @param[out] cellTypes    Array specifying the CGNS ElementType_t for each element
  @param[out] cell_ids     CGNS IDs of the cells
  @param[out] layouts      Array of `PetscLayout` that describes the distributed ownership of the cells in each `section_ids`
  @param[out] connectivity Array of the cell connectivity, described by `section`. The vertices are the local Plex IDs

  @description

  Each point in `section` corresponds to the `cellTypes` and `cell_ids` arrays.
  The dof and offset of `section` maps into the `connectivity` array.

  The `layouts` array is intended to be used with `PetscLayoutFindOwnerIndex_CGNSSectionLayouts()`
**/
static PetscErrorCode DMPlexCGNS_CreateCornersConnectivitySection(DM dm, PetscInt cgid, int base, int zone, PetscInt num_sections, const int section_ids[], PetscSection *section, CGNS_ENUMT(ElementType_t) * cellTypes[], PetscInt *cell_ids[], PetscLayout *layouts[], PetscInt *connectivity[])
{
  MPI_Comm     comm = PetscObjectComm((PetscObject)dm);
  PetscSection section_;
  char         buffer[CGIO_MAX_NAME_LENGTH + 1];
  CGNS_ENUMT(ElementType_t) * sectionCellTypes, *cellTypes_;
  CGRange       *ranges;
  PetscInt       nlocal_cells = 0, global_cell_dim = -1;
  PetscSegBuffer conn_sb;
  PetscLayout   *layouts_;

  PetscFunctionBegin;
  PetscCall(PetscMalloc2(num_sections, &ranges, num_sections, &sectionCellTypes));
  PetscCall(PetscMalloc1(num_sections, &layouts_));
  for (PetscInt s = 0; s < num_sections; s++) {
    int      nbndry, parentFlag;
    PetscInt local_size;

    PetscCallCGNSRead(cg_section_read(cgid, base, zone, section_ids[s], buffer, &sectionCellTypes[s], &ranges[s].start, &ranges[s].end, &nbndry, &parentFlag), dm, 0);
    PetscCheck(sectionCellTypes[s] != CGNS_ENUMV(NGON_n) && sectionCellTypes[s] != CGNS_ENUMV(NFACE_n), comm, PETSC_ERR_SUP, "CGNS reader does not support elements of type NGON_n or NFACE_n");
    PetscInt num_section_cells = ranges[s].end - ranges[s].start + 1;
    PetscCall(PetscLayoutCreateFromSizesAndOffset(comm, PETSC_DECIDE, num_section_cells, 1, ranges[s].start, &layouts_[s]));
    PetscCall(PetscLayoutGetLocalSize(layouts_[s], &local_size));
    nlocal_cells += local_size;
  }
  PetscCall(PetscSectionCreate(comm, &section_));
  PetscCall(PetscSectionSetChart(section_, 0, nlocal_cells));

  PetscCall(PetscMalloc1(nlocal_cells, cell_ids));
  PetscCall(PetscMalloc1(nlocal_cells, &cellTypes_));
  PetscCall(PetscSegBufferCreate(sizeof(PetscInt), nlocal_cells * 2, &conn_sb));
  for (PetscInt s = 0, c = 0; s < num_sections; s++) {
    PetscInt mystart, myend, myowned;

    PetscCall(PetscLayoutGetRange(layouts_[s], &mystart, &myend));
    PetscCall(PetscLayoutGetLocalSize(layouts_[s], &myowned));
    if (sectionCellTypes[s] == CGNS_ENUMV(MIXED)) {
      cgsize_t *offsets, *conn_cg;

      PetscCall(PetscMalloc1(myowned + 1, &offsets)); // The last element in the array is the total size of the connectivity for the given [start,end] range
      PetscCallCGNSRead(cgp_poly_elements_read_data_offsets(cgid, base, zone, section_ids[s], mystart, myend - 1, offsets), dm, 0);
      PetscCall(PetscMalloc1(offsets[myowned + 1], &conn_cg));
      PetscCallCGNSRead(cgp_poly_elements_read_data_elements(cgid, base, zone, section_ids[s], mystart, myend - 1, offsets, conn_cg), dm, 0);
      for (PetscInt i = 0; i < myowned; i++) {
        DMPolytopeType dm_cell_type = DM_POLYTOPE_UNKNOWN;
        PetscInt       numCorners, cell_dim, *conn_sb_seg;
        const int     *perm;

        (*cell_ids)[c] = mystart + i;

        cellTypes_[c] = (CGNS_ENUMT(ElementType_t))conn_cg[offsets[i]];
        PetscCall(CGNSElementTypeGetTopologyInfo(cellTypes_[c], &dm_cell_type, &numCorners, &cell_dim));
        if (global_cell_dim == -1) global_cell_dim = cell_dim;
        else
          PetscCheck(cell_dim == global_cell_dim, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Can only combine cells of the same dimension. Global cell dimension detected as %" PetscInt_FMT ", but CGNS element %" PetscInt_FMT " is dimension %" PetscInt_FMT, global_cell_dim, (*cell_ids)[c], cell_dim);
        PetscCall(PetscSegBufferGetInts(conn_sb, numCorners, &conn_sb_seg));

        PetscCall(DMPlexCGNSGetPermutation_Internal(dm_cell_type, numCorners, NULL, &perm));
        for (PetscInt v = 0; v < numCorners; ++v) conn_sb_seg[perm[v]] = conn_cg[offsets[i] + 1 + v];
        PetscCall(PetscSectionSetDof(section_, c, numCorners));
        c++;
      }
      PetscCall(PetscFree(offsets));
      PetscCall(PetscFree(conn_cg));
    } else {
      PetscInt       numCorners, cell_dim;
      PetscInt      *conn_sb_seg;
      DMPolytopeType dm_cell_type = DM_POLYTOPE_UNKNOWN;
      int            npe; // nodes per element
      const int     *perm;
      cgsize_t      *conn_cg;

      PetscCall(CGNSElementTypeGetTopologyInfo(sectionCellTypes[s], &dm_cell_type, &numCorners, &cell_dim));
      if (global_cell_dim == -1) global_cell_dim = cell_dim;
      else
        PetscCheck(cell_dim == global_cell_dim, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Can only combine cells of the same dimension. Global cell dimension detected as %" PetscInt_FMT ", but CGNS element %" PetscInt_FMT " is dimension %" PetscInt_FMT, global_cell_dim, (*cell_ids)[c], cell_dim);

      PetscCallCGNSRead(cg_npe(sectionCellTypes[s], &npe), dm, 0);
      PetscCall(PetscMalloc1(myowned * npe, &conn_cg));
      PetscCallCGNSRead(cgp_elements_read_data(cgid, base, zone, section_ids[s], mystart, myend - 1, conn_cg), dm, 0);
      PetscCall(DMPlexCGNSGetPermutation_Internal(dm_cell_type, numCorners, NULL, &perm));
      PetscCall(PetscSegBufferGetInts(conn_sb, numCorners * myowned, &conn_sb_seg));
      for (PetscInt i = 0; i < myowned; i++) {
        (*cell_ids)[c] = mystart + i;
        cellTypes_[c]  = sectionCellTypes[s];
        for (PetscInt v = 0; v < numCorners; ++v) conn_sb_seg[i * numCorners + perm[v]] = conn_cg[i * npe + v];
        PetscCall(PetscSectionSetDof(section_, c, numCorners));
        c++;
      }
      PetscCall(PetscFree(conn_cg));
    }
  }

  PetscCall(PetscSectionSetUp(section_));
  *section = section_;
  PetscCall(PetscSegBufferExtractAlloc(conn_sb, connectivity));
  PetscInt connSize;
  PetscCall(PetscSectionGetStorageSize(section_, &connSize));
  for (PetscInt i = 0; i < connSize; i++) (*connectivity)[i] -= 1; // vertices should be 0-based indexing for consistency with DMPlexBuildFromCellListParallel()
  *layouts = layouts_;
  if (cellTypes) *cellTypes = cellTypes_;
  else PetscCall(PetscFree(cellTypes_));

  PetscCall(PetscSegBufferDestroy(&conn_sb));
  PetscCall(PetscFree2(ranges, sectionCellTypes));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline PetscErrorCode PetscFindIntUnsorted(PetscInt key, PetscInt size, const PetscInt array[], PetscInt *loc)
{
  PetscFunctionBegin;
  *loc = -1;
  for (PetscInt i = 0; i < size; i++) {
    if (array[i] == key) {
      *loc = i;
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Same as `PetscLayoutFindOwnerIndex()`, but does not fail if owner not in PetscLayout
static PetscErrorCode PetscLayoutFindOwnerIndex_Internal(PetscLayout map, PetscInt idx, PetscMPIInt *owner, PetscInt *lidx, PetscBool *found_owner)
{
  PetscMPIInt lo = 0, hi, t;

  PetscFunctionBegin;
  PetscAssert((map->n >= 0) && (map->N >= 0) && (map->range), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "PetscLayoutSetUp() must be called first");
  if (owner) *owner = -1;
  if (lidx) *lidx = -1;
  if (idx < map->range[0] && idx >= map->range[map->size + 1]) {
    *found_owner = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  hi = map->size;
  while (hi - lo > 1) {
    t = lo + (hi - lo) / 2;
    if (idx < map->range[t]) hi = t;
    else lo = t;
  }
  if (owner) *owner = lo;
  if (lidx) *lidx = idx - map->range[lo];
  if (found_owner) *found_owner = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// This function assumes that there is an array that `maps` describes the layout too. Locally, the range of each map is concatenated onto each other.
// So [maps[0].start, ..., maps[0].end - 1, maps[1].start, ..., maps[1].end - 1, ...]
// The returned index is the index into this array
static PetscErrorCode PetscLayoutFindOwnerIndex_CGNSSectionLayouts(PetscLayout maps[], PetscInt nmaps, PetscInt idx, PetscMPIInt *owner, PetscInt *lidx, PetscInt *mapidx)
{
  PetscFunctionBegin;
  for (PetscInt m = 0; m < nmaps; m++) {
    PetscBool found_owner = PETSC_FALSE;
    PetscCall(PetscLayoutFindOwnerIndex_Internal(maps[m], idx, owner, lidx, &found_owner));
    if (found_owner) {
      // Now loop back through the previous maps to get the local offset for the containing index
      for (PetscInt mm = m - 1; mm >= 0; mm--) {
        PetscInt size = maps[mm]->range[*owner + 1] - maps[mm]->range[*owner];
        *lidx += size;
      }
      if (mapidx) *mapidx = m;
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }
  SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "CGNS id %" PetscInt_FMT " not found in layouts", idx);
}

/*
  @brief Matching root and leaf indices across processes, allowing multiple roots per leaf

  Collective

  Input Parameters:
+ layout           - `PetscLayout` defining the global index space and the MPI rank that brokers each index
. numRootIndices   - size of `rootIndices`
. rootIndices      - array of global indices of which this process requests ownership
. rootLocalIndices - root local index permutation (`NULL` if no permutation)
. rootLocalOffset  - offset to be added to `rootLocalIndices`
. numLeafIndices   - size of `leafIndices`
. leafIndices      - array of global indices with which this process requires data associated
. leafLocalIndices - leaf local index permutation (`NULL` if no permutation)
- leafLocalOffset  - offset to be added to `leafLocalIndices`

  Output Parameters:
+ matchSection - The `PetscSection` describing the layout of `matches` with respect to the `leafIndices` (the points of the section indexes into `leafIndices`)
- matches      - Array of `PetscSFNode` denoting the location (rank and index) of the root matching the leaf.

  Example 1:
.vb
  rank             : 0            1            2
  rootIndices      : [1 0 2]      [3]          [3]
  rootLocalOffset  : 100          200          300
  layout           : [0 1]        [2]          [3]
  leafIndices      : [0]          [2]          [0 3]
  leafLocalOffset  : 400          500          600

would build the following result

  rank 0: [0] 400 <- [0] (0,101)
  ------------------------------
  rank 1: [0] 500 <- [0] (0,102)
  ------------------------------
  rank 2: [0] 600 <- [0] (0,101)
          [1] 601 <- [1] (1,200)
          [1] 601 <- [2] (2,300)
           |   |      |     |
           |   |      |     + `matches`, the rank and index of the respective match with the leaf index
           |   |      + index into `matches` array
           |   + the leaves for the respective root
           + The point in `matchSection` (indexes into `leafIndices`)

  For rank 2, the `matchSection` would be:

  [0]: (1, 0)
  [1]: (2, 1)
   |    |  |
   |    |  + offset
   |    + ndof
   + point
.ve

  Notes:
  This function is identical to `PetscSFCreateByMatchingIndices()` except it includes *all* matching indices instead of designating a single rank as the "owner".
  Attempting to create an SF with all matching indices would create an invalid SF, thus we give an array of `matches` and `matchSection` to describe the layout
  Compare the examples in this document to those in `PetscSFCreateByMatchingIndices()`.

.seealso: [](sec_petscsf), `PetscSF`, `PetscSFCreate()`, `PetscSFCreateByMatchingIndices()`
*/
static PetscErrorCode PetscSFFindMatchingIndices(PetscLayout layout, PetscInt numRootIndices, const PetscInt rootIndices[], const PetscInt rootLocalIndices[], PetscInt rootLocalOffset, PetscInt numLeafIndices, const PetscInt leafIndices[], const PetscInt leafLocalIndices[], PetscInt leafLocalOffset, PetscSection *matchSection, PetscSFNode *matches[])
{
  MPI_Comm     comm = layout->comm;
  PetscMPIInt  rank;
  PetscSF      sf1;
  PetscSection sectionBuffer, matchSection_;
  PetscInt     numMatches;
  PetscSFNode *roots, *buffer, *matches_;
  PetscInt     N, n, pStart, pEnd;
  PetscBool    areIndicesSame;

  PetscFunctionBegin;
  if (rootIndices) PetscAssertPointer(rootIndices, 3);
  if (rootLocalIndices) PetscAssertPointer(rootLocalIndices, 4);
  if (leafIndices) PetscAssertPointer(leafIndices, 7);
  if (leafLocalIndices) PetscAssertPointer(leafLocalIndices, 8);
  PetscCheck(numRootIndices >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "numRootIndices (%" PetscInt_FMT ") must be non-negative", numRootIndices);
  PetscCheck(numLeafIndices >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "numLeafIndices (%" PetscInt_FMT ") must be non-negative", numLeafIndices);
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscLayoutSetUp(layout));
  PetscCall(PetscLayoutGetSize(layout, &N));
  PetscCall(PetscLayoutGetLocalSize(layout, &n));
  areIndicesSame = (PetscBool)(leafIndices == rootIndices);
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &areIndicesSame, 1, MPI_C_BOOL, MPI_LAND, comm));
  PetscCheck(!areIndicesSame || numLeafIndices == numRootIndices, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "leafIndices == rootIndices, but numLeafIndices (%" PetscInt_FMT ") != numRootIndices(%" PetscInt_FMT ")", numLeafIndices, numRootIndices);
  if (PetscDefined(USE_DEBUG)) {
    PetscInt N1 = PETSC_INT_MIN;
    for (PetscInt i = 0; i < numRootIndices; i++)
      if (rootIndices[i] > N1) N1 = rootIndices[i];
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &N1, 1, MPIU_INT, MPI_MAX, comm));
    PetscCheck(N1 < N, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Max. root index (%" PetscInt_FMT ") out of layout range [0,%" PetscInt_FMT ")", N1, N);
    if (!areIndicesSame) {
      N1 = PETSC_INT_MIN;
      for (PetscInt i = 0; i < numLeafIndices; i++)
        if (leafIndices[i] > N1) N1 = leafIndices[i];
      PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &N1, 1, MPIU_INT, MPI_MAX, comm));
      PetscCheck(N1 < N, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Max. leaf index (%" PetscInt_FMT ") out of layout range [0,%" PetscInt_FMT ")", N1, N);
    }
  }

  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  { /* Reduce: roots -> buffer */
    // Data in buffer described by section_buffer. The chart of `section_buffer` maps onto the local portion of `layout`, with dofs denoting how many matches there are.
    PetscInt        bufsize;
    const PetscInt *root_degree;

    PetscCall(PetscSFCreate(comm, &sf1));
    PetscCall(PetscSFSetFromOptions(sf1));
    PetscCall(PetscSFSetGraphLayout(sf1, layout, numRootIndices, NULL, PETSC_OWN_POINTER, rootIndices));

    PetscCall(PetscSFComputeDegreeBegin(sf1, &root_degree));
    PetscCall(PetscSFComputeDegreeEnd(sf1, &root_degree));
    PetscCall(PetscSectionCreate(comm, &sectionBuffer));
    PetscCall(PetscSectionSetChart(sectionBuffer, 0, n));
    PetscCall(PetscMalloc1(numRootIndices, &roots));
    for (PetscInt i = 0; i < numRootIndices; ++i) {
      roots[i].rank  = rank;
      roots[i].index = rootLocalOffset + (rootLocalIndices ? rootLocalIndices[i] : i);
    }
    for (PetscInt i = 0; i < n; i++) PetscCall(PetscSectionSetDof(sectionBuffer, i, root_degree[i]));
    PetscCall(PetscSectionSetUp(sectionBuffer));
    PetscCall(PetscSectionGetStorageSize(sectionBuffer, &bufsize));
    PetscCall(PetscMalloc1(bufsize, &buffer));
    for (PetscInt i = 0; i < bufsize; ++i) {
      buffer[i].index = -1;
      buffer[i].rank  = -1;
    }
    PetscCall(PetscSFGatherBegin(sf1, MPIU_SF_NODE, roots, buffer));
    PetscCall(PetscSFGatherEnd(sf1, MPIU_SF_NODE, roots, buffer));
    PetscCall(PetscFree(roots));
  }

  // Distribute data in buffers to the leaf locations. The chart of `sectionMatches` maps to `leafIndices`, with dofs denoting how many matches there are for each leaf.
  if (!areIndicesSame) PetscCall(PetscSFSetGraphLayout(sf1, layout, numLeafIndices, NULL, PETSC_OWN_POINTER, leafIndices));
  PetscCall(PetscSectionCreate(comm, &matchSection_));
  PetscCall(PetscSectionMigrateData(sf1, MPIU_SF_NODE, sectionBuffer, buffer, matchSection_, (void **)&matches_, NULL));
  PetscCall(PetscSectionGetChart(matchSection_, &pStart, &pEnd));
  PetscCheck(pEnd - pStart == numLeafIndices, comm, PETSC_ERR_PLIB, "Section of matches has different chart size (%" PetscInt_FMT ")  than number of leaf indices %" PetscInt_FMT ". Section chart is [%" PetscInt_FMT ", %" PetscInt_FMT ")", pEnd - pStart, numLeafIndices, pStart, pEnd);
  PetscCall(PetscSectionGetStorageSize(matchSection_, &numMatches));
  for (PetscInt p = pStart; p < pEnd; p++) {
    PetscInt ndofs;
    PetscCall(PetscSectionGetDof(matchSection_, p, &ndofs));
    PetscCheck(ndofs > 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No match found for index %" PetscInt_FMT, leafIndices[p]);
  }

  *matchSection = matchSection_;
  *matches      = matches_;

  PetscCall(PetscFree(buffer));
  PetscCall(PetscSectionDestroy(&sectionBuffer));
  PetscCall(PetscSFDestroy(&sf1));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/**
  @brief Match CGNS faces to their Plex equivalents

  @param[in]  dm                 DM that holds the Plex to match against
  @param[in]  nuniq_verts        Number of unique CGNS vertices on this rank
  @param[in]  uniq_verts         Unique CGNS vertices on this rank
  @param[in]  plex_vertex_offset Index offset to match `uniq_vertices` to their respective Plex vertices
  @param[in]  NVertices          Number of vertices for Layout for `PetscSFFindMatchingIndices()`
  @param[in]  connSection        PetscSection describing the CGNS face connectivity
  @param[in]  face_ids           Array of the CGNS face IDs
  @param[in]  conn               Array of the CGNS face connectivity
  @param[out] cg2plexSF          PetscSF describing the mapping from owned CGNS faces to remote `plexFaces`
  @param[out] plexFaces          Matching Plex face IDs

  @description

   `cg2plexSF` is a mapping from the owned CGNS faces to the rank whose local Plex has that face.
   `plexFaces` holds the actual mesh point in the local Plex that corresponds to the owned CGNS face (which is the root)

         cg2plexSF
   __________|__________
   |                   |

   [F0_11] -----> [P0_0]  [38]
   [F0_12] --                        Rank 0
   ~~~~~~~~~ \ ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
              --> [P1_0]  [59]       Rank 1
   [F1_13] -----> [P1_1]  [98]
      |              |     |
      |              |     + plexFaces, maps the leaves of cg2plexSF to local Plex face mesh points
      |              + Leaves of cg2plexSF. P(rank)_(root_index)
      + Roots of cg2plexSF, F(rank)_(CGNS face ID)

   Note that, unlike a pointSF, the leaves of `cg2plexSF` do not map onto chart of the local Plex, but just onto an array.
   The plexFaces array is then what maps the leaves to the actual local Plex mesh points.

  `plex_vertex_offset` is used to map the CGNS vertices in `uniq_vertices` to their respective Plex vertices.
   From `DMPlexBuildFromCellListParallel()`, the mapping of CGNS vertices to Plex vertices is uniq_vert[i] -> i + plex_vertex_offset, where the right side is the Plex point ID.
   So with `plex_vertex_offset = 5`,
   uniq_vertices:  [19, 52, 1, 89]
   plex_point_ids: [5,   6, 7, 8]
**/
static PetscErrorCode DMPlexCGNS_MatchCGNSFacesToPlexFaces(DM dm, PetscInt nuniq_verts, const PetscInt uniq_verts[], PetscInt plex_vertex_offset, PetscInt NVertices, PetscSection connSection, const PetscInt face_ids[], const PetscInt conn[], PetscSF *cg2plexSF, PetscInt *plexFaces[])
{
  MPI_Comm    comm = PetscObjectComm((PetscObject)dm);
  PetscMPIInt myrank, nranks;
  PetscInt    fownedStart, fownedEnd, fownedSize;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &myrank));
  PetscCallMPI(MPI_Comm_size(comm, &nranks));

  { // -- Create cg2plexSF
    PetscInt     nuniq_face_verts, *uniq_face_verts;
    PetscSection fvert2mvertSection;
    PetscSFNode *fvert2mvert = NULL;

    { // -- Create fvert2mvert, which map CGNS vertices in the owned-face connectivity to the CGNS vertices in the global mesh
      PetscLayout layout;

      PetscCall(PetscLayoutCreateFromSizes(comm, PETSC_DECIDE, NVertices, 1, &layout));
      { // Count locally unique vertices in the face connectivity
        PetscHSetI vhash;
        PetscInt   off = 0, conn_size;

        PetscCall(PetscHSetICreate(&vhash));
        PetscCall(PetscSectionGetStorageSize(connSection, &conn_size));
        for (PetscInt v = 0; v < conn_size; ++v) PetscCall(PetscHSetIAdd(vhash, conn[v]));
        PetscCall(PetscHSetIGetSize(vhash, &nuniq_face_verts));
        PetscCall(PetscMalloc1(nuniq_face_verts, &uniq_face_verts));
        PetscCall(PetscHSetIGetElems(vhash, &off, uniq_face_verts));
        PetscCall(PetscHSetIDestroy(&vhash));
        PetscCheck(off == nuniq_face_verts, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid number of local vertices %" PetscInt_FMT " should be %" PetscInt_FMT, off, nuniq_face_verts);
      }
      PetscCall(PetscSortInt(nuniq_face_verts, uniq_face_verts));
      PetscCall(PetscSFFindMatchingIndices(layout, nuniq_verts, uniq_verts, NULL, 0, nuniq_face_verts, uniq_face_verts, NULL, 0, &fvert2mvertSection, &fvert2mvert));

      PetscCall(PetscLayoutDestroy(&layout));
    }

    PetscSFNode *plexFaceRemotes, *ownedFaceRemotes;
    PetscCount   nPlexFaceRemotes;
    PetscInt    *local_rank_count;

    PetscCall(PetscSectionGetChart(connSection, &fownedStart, &fownedEnd));
    fownedSize = fownedEnd - fownedStart;
    PetscCall(PetscCalloc1(nranks, &local_rank_count));

    { // Find the rank(s) whose local Plex has the owned CGNS face
      // We determine ownership by determining which ranks contain all the vertices in a face's connectivity
      PetscInt       maxRanksPerVert;
      PetscInt      *face_ranks;
      PetscSegBuffer plexFaceRemotes_SB, ownedFaceRemotes_SB;

      PetscCall(PetscSegBufferCreate(sizeof(PetscSFNode), fownedSize, &plexFaceRemotes_SB));
      PetscCall(PetscSegBufferCreate(sizeof(PetscSFNode), fownedSize, &ownedFaceRemotes_SB));
      PetscCall(PetscSectionGetMaxDof(fvert2mvertSection, &maxRanksPerVert));
      PetscCall(PetscMalloc1(maxRanksPerVert, &face_ranks));
      for (PetscInt f = fownedStart, f_i = 0; f < fownedEnd; f++, f_i++) {
        PetscInt fndof, foffset, lndof, loffset, idx, nface_ranks = 0;

        PetscCall(PetscSectionGetDof(connSection, f, &fndof));
        PetscCall(PetscSectionGetOffset(connSection, f, &foffset));
        PetscCall(PetscFindInt(conn[foffset + 0], nuniq_face_verts, uniq_face_verts, &idx));
        PetscCall(PetscSectionGetDof(fvert2mvertSection, idx, &lndof));
        PetscCall(PetscSectionGetOffset(fvert2mvertSection, idx, &loffset));
        // Loop over ranks of the first vertex in the face connectivity
        for (PetscInt l = 0; l < lndof; l++) {
          PetscInt rank = fvert2mvert[loffset + l].rank;
          // Loop over vertices of face (except for the first) to see if those vertices have the same candidate rank
          for (PetscInt v = 1; v < fndof; v++) {
            PetscInt  ldndof, ldoffset, idx;
            PetscBool vert_has_rank = PETSC_FALSE;

            PetscCall(PetscFindInt(conn[foffset + v], nuniq_face_verts, uniq_face_verts, &idx));
            PetscCall(PetscSectionGetDof(fvert2mvertSection, idx, &ldndof));
            PetscCall(PetscSectionGetOffset(fvert2mvertSection, idx, &ldoffset));
            // Loop over ranks of the vth vertex to see if it has the candidate rank
            for (PetscInt ld = 0; ld < ldndof; ld++) vert_has_rank = (fvert2mvert[ldoffset + ld].rank == rank) || vert_has_rank;
            if (vert_has_rank) continue; // This vertex has the candidate rank, proceed to the next vertex
            else goto next_candidate_rank;
          }
          face_ranks[nface_ranks++] = rank;
        next_candidate_rank:
          continue;
        }
        PetscCheck(nface_ranks > 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Rank containing CGNS face %" PetscInt_FMT " could not be found", face_ids[f_i]);

        // Once we have found the rank(s) whose Plex has the vertices of CGNS face `f`, we can begin to build the information for the SF.
        // We want the roots to be the CGNS faces and the leaves to be the corresponding Plex faces, but we have the opposite; for the owned face on myrank, we know the rank that has the corresponding Plex face.
        // To get the inverse, we assign each SF edge with a tuple of PetscSFNodes; one in `plexFaceRemotes` and the other in `ownedFaceRemotes`.
        // `plexFaceRemotes` has the rank and index of the Plex face.
        // `ownedFaceRemotes` has the rank and index of the owned CGNS face.
        // Note that `ownedFaceRemotes` is all on my rank (e.g. rank == myrank).
        //
        // Then, to build the `cg2plexSF`, we communicate the `ownedFaceRemotes` to the `plexFaceRemotes` (via SFReduce).
        // Those `ownedFaceRemotes` then act as the leaves to the roots on this process.
        //
        // Conceptually, this is the same as calling `PetscSFCreateInverseSF` on an SF with `iremotes = plexFaceRemotes` and the `ilocal = ownedFaceRemotes[:].index`.
        // However, we cannot use this much simpler way because when there are multiple matching Plex faces, `PetscSFCreateInverseSF()` will be invalid due to `ownedFaceRemotes[:].index` having repeated values (only root vertices of the SF graph may have degree > 1)
        PetscSFNode *plexFaceRemotes_buffer, *ownedFaceRemotes_buffer;
        PetscCall(PetscSegBufferGet(plexFaceRemotes_SB, nface_ranks, &plexFaceRemotes_buffer));
        PetscCall(PetscSegBufferGet(ownedFaceRemotes_SB, nface_ranks, &ownedFaceRemotes_buffer));
        for (PetscInt n = 0; n < nface_ranks; n++) {
          plexFaceRemotes_buffer[n].rank = face_ranks[n];
          local_rank_count[face_ranks[n]]++;
          ownedFaceRemotes_buffer[n] = (PetscSFNode){.rank = myrank, .index = f_i};
        }
      }
      PetscCall(PetscFree(face_ranks));

      PetscCall(PetscSegBufferGetSize(plexFaceRemotes_SB, &nPlexFaceRemotes));
      PetscCall(PetscSegBufferExtractAlloc(plexFaceRemotes_SB, &plexFaceRemotes));
      PetscCall(PetscSegBufferDestroy(&plexFaceRemotes_SB));
      PetscCall(PetscSegBufferExtractAlloc(ownedFaceRemotes_SB, &ownedFaceRemotes));
      PetscCall(PetscSegBufferDestroy(&ownedFaceRemotes_SB));

      // To get the index for plexFaceRemotes, we partition the leaves on each rank (e.g. the array that will hold the local Plex face mesh points) by each rank that has the CGNS owned rank.
      // For r in [0,numranks), local_rank_count[r] holds the number plexFaces that myrank holds.
      // This determines how large a partition the leaves on rank r need to create for myrank.
      // To get the offset into the leaves, we use Exscan to get rank_start.
      // For r in [0, numranks), rank_start[r] holds the offset into rank r's leaves that myrank will index into.

      // Below is an example:
      //
      // myrank:             | 0           1           2
      // local_rank_count:   | [3, 2, 0]   [1, 0, 2]   [2, 2, 1]
      // myrank_total_count: | 6           4           3
      // rank_start:         | [0, 0, 0]   [3, 2, 0]   [4, 2, 2]
      //                     |
      // plexFaceRemotes: 0  | (0, 0)      (2, 0)      (2, 2)        <-- (rank, index) tuples (e.g. PetscSFNode)
      //                  1  | (1, 0)      (2, 1)      (0, 4)
      //                  2  | (0, 1)      (0, 3)      (1, 2)
      //                  3  | (0, 2)                  (0, 5)
      //                  4  | (1, 1)                  (1, 3)
      //
      // leaves:          0  | (0, 0)      (0, 1)      (1, 0)    (rank and index into plexFaceRemotes)
      //                  1  | (0, 2)      (0, 4)      (1, 1)
      //                  2  | (0, 3)      (2, 2)      (2, 0)
      //                  3  | (1, 2)      (2, 4)
      //                  4  | (2, 1)
      //                  5  | (2, 3)
      //
      // Note how at the leaves, the ranks are contiguous and in order
      PetscInt myrank_total_count;
      {
        PetscInt *rank_start, *rank_offset;

        PetscCall(PetscCalloc2(nranks, &rank_start, nranks, &rank_offset));
        PetscCallMPI(MPIU_Allreduce(local_rank_count, rank_start, nranks, MPIU_INT, MPI_SUM, comm));
        myrank_total_count = rank_start[myrank];
        PetscCall(PetscArrayzero(rank_start, nranks));
        PetscCallMPI(MPI_Exscan(local_rank_count, rank_start, nranks, MPIU_INT, MPI_SUM, comm));

        for (PetscInt r = 0; r < nPlexFaceRemotes; r++) {
          PetscInt rank            = plexFaceRemotes[r].rank;
          plexFaceRemotes[r].index = rank_start[rank] + rank_offset[rank];
          rank_offset[rank]++;
        }
        PetscCall(PetscFree2(rank_start, rank_offset));
      }

      { // Communicate the leaves to roots and build cg2plexSF
        PetscSF      plexRemotes2ownedRemotesSF;
        PetscSFNode *iremote_cg2plexSF;

        PetscCall(PetscSFCreate(comm, &plexRemotes2ownedRemotesSF));
        PetscCall(PetscSFSetGraph(plexRemotes2ownedRemotesSF, myrank_total_count, nPlexFaceRemotes, NULL, PETSC_COPY_VALUES, plexFaceRemotes, PETSC_OWN_POINTER));
        PetscCall(PetscMalloc1(myrank_total_count, &iremote_cg2plexSF));
        PetscCall(PetscSFViewFromOptions(plexRemotes2ownedRemotesSF, NULL, "-plex2ownedremotes_sf_view"));
        for (PetscInt i = 0; i < myrank_total_count; i++) iremote_cg2plexSF[i] = (PetscSFNode){.rank = -1, .index = -1};
        PetscCall(PetscSFReduceBegin(plexRemotes2ownedRemotesSF, MPIU_SF_NODE, ownedFaceRemotes, iremote_cg2plexSF, MPI_REPLACE));
        PetscCall(PetscSFReduceEnd(plexRemotes2ownedRemotesSF, MPIU_SF_NODE, ownedFaceRemotes, iremote_cg2plexSF, MPI_REPLACE));
        PetscCall(PetscSFDestroy(&plexRemotes2ownedRemotesSF));
        for (PetscInt i = 0; i < myrank_total_count; i++) PetscCheck(iremote_cg2plexSF[i].rank >= 0 && iremote_cg2plexSF[i].index != -1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Owned face SFNode was not reduced properly");

        PetscCall(PetscSFCreate(comm, cg2plexSF));
        PetscCall(PetscSFSetGraph(*cg2plexSF, fownedSize, myrank_total_count, NULL, PETSC_COPY_VALUES, iremote_cg2plexSF, PETSC_OWN_POINTER));

        PetscCall(PetscFree(ownedFaceRemotes));
      }

      PetscCall(PetscFree(local_rank_count));
    }

    PetscCall(PetscSectionDestroy(&fvert2mvertSection));
    PetscCall(PetscFree(uniq_face_verts));
    PetscCall(PetscFree(fvert2mvert));
  }

  { // -- Find plexFaces
    // Distribute owned-CGNS-face connectivity to the ranks which have corresponding Plex faces, and then find the corresponding Plex faces
    PetscSection connDistSection;
    PetscInt    *connDist;

    // Distribute the face connectivity to the rank that has that face
    PetscCall(PetscSectionCreate(comm, &connDistSection));
    PetscCall(PetscSectionMigrateData(*cg2plexSF, MPIU_INT, connSection, conn, connDistSection, (void **)&connDist, NULL));

    { // Translate CGNS vertex numbering to local Plex numbering
      PetscInt *dmplex_verts, *uniq_verts_sorted;
      PetscInt  connDistSize;

      PetscCall(PetscMalloc2(nuniq_verts, &dmplex_verts, nuniq_verts, &uniq_verts_sorted));
      PetscCall(PetscArraycpy(uniq_verts_sorted, uniq_verts, nuniq_verts));
      // uniq_verts are one-to-one with the DMPlex vertices with an offset, see DMPlexBuildFromCellListParallel()
      for (PetscInt v = 0; v < nuniq_verts; v++) dmplex_verts[v] = v + plex_vertex_offset;
      PetscCall(PetscSortIntWithArray(nuniq_verts, uniq_verts_sorted, dmplex_verts));

      PetscCall(PetscSectionGetStorageSize(connDistSection, &connDistSize));
      for (PetscInt v = 0; v < connDistSize; v++) {
        PetscInt idx;
        PetscCall(PetscFindInt(connDist[v], nuniq_verts, uniq_verts_sorted, &idx));
        PetscCheck(idx >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not find CGNS vertex id (from face connectivity) in local plex");
        connDist[v] = dmplex_verts[idx];
      }
      PetscCall(PetscFree2(dmplex_verts, uniq_verts_sorted));
    }

    // Debugging info
    PetscBool view_connectivity = PETSC_FALSE;
    PetscCall(PetscOptionsGetBool(NULL, NULL, "-dm_plex_cgns_view_face_connectivity", &view_connectivity, NULL));
    if (view_connectivity) {
      PetscSection conesSection;
      PetscInt    *cones;

      PetscCall(PetscPrintf(comm, "Distributed CGNS Face Connectivity (in Plex vertex numbering):\n"));
      PetscCall(PetscSectionArrayView(connDistSection, connDist, PETSC_INT, NULL));
      PetscCall(DMPlexGetCones(dm, &cones));
      PetscCall(DMPlexGetConeSection(dm, &conesSection));
      PetscCall(PetscPrintf(comm, "Plex Cones:\n"));
      PetscCall(PetscSectionArrayView(conesSection, cones, PETSC_INT, NULL));
    }

    // For every face in connDistSection, find the transitive support of a vertex in that face connectivity.
    // Loop through the faces of the transitive support and find the matching face
    PetscBT  plex_face_found;
    PetscInt fplexStart, fplexEnd, vplexStart, vplexEnd;
    PetscInt fdistStart, fdistEnd, numfdist;
    PetscCall(DMPlexGetHeightStratum(dm, 1, &fplexStart, &fplexEnd));
    PetscCall(DMPlexGetDepthStratum(dm, 0, &vplexStart, &vplexEnd));
    PetscCall(PetscSectionGetChart(connDistSection, &fdistStart, &fdistEnd));
    numfdist = fdistEnd - fdistStart;
    PetscCall(PetscMalloc1(numfdist, plexFaces));
    PetscCall(PetscBTCreate(numfdist, &plex_face_found));
    for (PetscInt i = 0; i < numfdist; i++) (*plexFaces)[i] = -1;

    for (PetscInt f = fdistStart, f_i = 0; f < fdistEnd; f++, f_i++) {
      PetscInt  ndof, offset, support_size;
      PetscInt *support = NULL;

      PetscCall(PetscSectionGetDof(connDistSection, f, &ndof));
      PetscCall(PetscSectionGetOffset(connDistSection, f, &offset));

      // Loop through transitive support of a vertex in the CGNS face connectivity
      PetscCall(DMPlexGetTransitiveClosure(dm, connDist[offset + 0], PETSC_FALSE, &support_size, &support));
      for (PetscInt s = 0; s < support_size; s++) {
        PetscInt face_point = support[s * 2]; // closure stores points and orientations, [p_0, o_0, p_1, o_1, ...]
        PetscInt trans_cone_size, *trans_cone = NULL;

        if (face_point < fplexStart || face_point >= fplexEnd) continue; // Skip non-face points
        // See if face_point has the same vertices
        PetscCall(DMPlexGetTransitiveClosure(dm, face_point, PETSC_TRUE, &trans_cone_size, &trans_cone));
        for (PetscInt c = 0; c < trans_cone_size; c++) {
          PetscInt vertex_point = trans_cone[c * 2], conn_has_vertex;
          if (vertex_point < vplexStart || vertex_point >= vplexEnd) continue; // Skip non-vertex points
          PetscCall(PetscFindIntUnsorted(vertex_point, ndof, &connDist[offset], &conn_has_vertex));
          if (conn_has_vertex < 0) goto check_next_face;
        }
        (*plexFaces)[f_i] = face_point;
        PetscCall(DMPlexRestoreTransitiveClosure(dm, face_point, PETSC_TRUE, &trans_cone_size, &trans_cone));
        break;
      check_next_face:
        PetscCall(DMPlexRestoreTransitiveClosure(dm, face_point, PETSC_TRUE, &trans_cone_size, &trans_cone));
        continue;
      }
      PetscCall(DMPlexRestoreTransitiveClosure(dm, connDist[offset + 0], PETSC_FALSE, &support_size, &support));
      if ((*plexFaces)[f_i] != -1) PetscCall(PetscBTSet(plex_face_found, f_i));
    }

    // Some distributed CGNS faces did not find a matching plex face
    // This can happen if a partition has all the faces surrounding a distributed CGNS face, but does not have the face itself (it's parent element is owned by a different partition).
    // Thus, the partition has the vertices associated with the CGNS face, but doesn't actually have the face itself.
    // For example, take the following quad mesh, where the numbers represent the owning rank and CGNS face ID.
    //
    //    2     3     4     <-- face ID
    //  ----- ----- -----
    // |  0  |  1  |  0  |  <-- rank
    //  ----- ----- -----
    //    5     6     7     <-- face ID
    //
    // In this case, rank 0 will have all the vertices of face 3 and 6 in it's Plex, but does not actually have either face.
    //
    // To address this, we remove the leaves associated with these missing faces from cg2plexSF and then verify that all owned faces did find a matching plex face (e.g. root degree > 1)
    PetscCount num_plex_faces_found = PetscBTCountSet(plex_face_found, numfdist);
    PetscBool  some_faces_not_found = num_plex_faces_found < numfdist;
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &some_faces_not_found, 1, MPI_C_BOOL, MPI_LOR, comm));
    if (some_faces_not_found) {
      PetscSFNode    *iremote_cg2plex_new;
      const PetscInt *root_degree;
      PetscInt        num_roots, *plexFacesNew;

      PetscCall(PetscMalloc1(num_plex_faces_found, &iremote_cg2plex_new));
      PetscCall(PetscCalloc1(num_plex_faces_found, &plexFacesNew));
      { // Get SFNodes with matching faces
        const PetscSFNode *iremote_cg2plex_old;
        PetscInt           num_leaves_old, n = 0;
        PetscCall(PetscSFGetGraph(*cg2plexSF, &num_roots, &num_leaves_old, NULL, &iremote_cg2plex_old));
        PetscAssert(num_roots == fownedSize, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent roots and owned faces.");
        PetscAssert(num_leaves_old == numfdist, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inconsistent leaves and distributed faces.");
        for (PetscInt o = 0; o < num_leaves_old; o++) {
          if (PetscBTLookupSet(plex_face_found, o)) {
            iremote_cg2plex_new[n] = iremote_cg2plex_old[o];
            plexFacesNew[n]        = (*plexFaces)[o];
            n++;
          }
        }
        PetscAssert(n == num_plex_faces_found, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Found %" PetscCount_FMT " matching plex faces, but only set %" PetscInt_FMT " SFNodes", num_plex_faces_found, n);
      }
      PetscCall(PetscSFSetGraph(*cg2plexSF, num_roots, num_plex_faces_found, NULL, PETSC_COPY_VALUES, iremote_cg2plex_new, PETSC_OWN_POINTER));

      // Verify that all CGNS faces have a matching Plex face on any rank
      PetscCall(PetscSFComputeDegreeBegin(*cg2plexSF, &root_degree));
      PetscCall(PetscSFComputeDegreeEnd(*cg2plexSF, &root_degree));
      for (PetscInt r = 0; r < num_roots; r++) {
        PetscCheck(root_degree[r] > 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not find plex face for the CGNS face %" PetscInt_FMT, face_ids[r]);
        PetscCheck(root_degree[r] == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Found more than one plex face for the CGNS face %" PetscInt_FMT ". Face may be internal rather than at mesh domain boundary", face_ids[r]);
      }

      if (PetscDefined(USE_DEBUG)) {
        for (PetscInt i = 0; i < num_plex_faces_found; i++)
          PetscCheck(plexFacesNew[i] >= fplexStart && plexFacesNew[i] < fplexEnd, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Plex face ID %" PetscInt_FMT "outside of face stratum [%" PetscInt_FMT ", %" PetscInt_FMT ")", plexFacesNew[i], fplexStart, fplexEnd);
      }

      PetscCall(PetscFree(*plexFaces));
      *plexFaces = plexFacesNew;
    }

    PetscCall(PetscBTDestroy(&plex_face_found));
    PetscCall(PetscSectionDestroy(&connDistSection));
    PetscCall(PetscFree(connDist));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Copied from PetscOptionsStringToInt
static inline PetscErrorCode PetscStrtoInt(const char name[], PetscInt *a)
{
  size_t len;
  char  *endptr;
  long   strtolval;

  PetscFunctionBegin;
  PetscCall(PetscStrlen(name, &len));
  PetscCheck(len, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "character string of length zero has no numerical value");

  strtolval = strtol(name, &endptr, 10);
  PetscCheck((size_t)(endptr - name) == len, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Input string \"%s\" has no integer value (do not include . in it)", name);

#if defined(PETSC_USE_64BIT_INDICES) && defined(PETSC_HAVE_ATOLL)
  (void)strtolval;
  *a = atoll(name);
#elif defined(PETSC_USE_64BIT_INDICES) && defined(PETSC_HAVE___INT64)
  (void)strtolval;
  *a = _atoi64(name);
#else
  *a = (PetscInt)strtolval;
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

typedef struct {
  char     name[CGIO_MAX_NAME_LENGTH + 1];
  int      normal[3], ndatasets;
  cgsize_t npoints, nnormals;
  CGNS_ENUMT(BCType_t) bctype;
  CGNS_ENUMT(DataType_t) normal_datatype;
  CGNS_ENUMT(PointSetType_t) pointtype;
} CGBCInfo;

PetscErrorCode DMPlexCreateCGNS_Internal_Parallel(MPI_Comm comm, PetscInt cgid, PetscBool interpolate, DM *dm)
{
  PetscMPIInt num_proc, rank;
  /* Read from file */
  char     basename[CGIO_MAX_NAME_LENGTH + 1];
  char     buffer[CGIO_MAX_NAME_LENGTH + 1];
  int      dim = 0, physDim = 0, coordDim = 0;
  PetscInt NVertices = 0, NCells = 0;
  int      nzones = 0, nbases;
  int      zone   = 1; // Only supports single zone files
  int      base   = 1; // Only supports single base

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &num_proc));
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));

  PetscCallCGNSRead(cg_nbases(cgid, &nbases), *dm, 0);
  PetscCheck(nbases <= 1, PETSC_COMM_SELF, PETSC_ERR_LIB, "CGNS file must have a single base, not %d", nbases);
  //  From the CGNS web page                 cell_dim  phys_dim (embedding space in PETSc) CGNS defines as length of spatial vectors/components)
  PetscCallCGNSRead(cg_base_read(cgid, base, basename, &dim, &physDim), *dm, 0);
  PetscCallCGNSRead(cg_nzones(cgid, base, &nzones), *dm, 0);
  PetscCheck(nzones == 1, PETSC_COMM_SELF, PETSC_ERR_LIB, "Parallel reader limited to one zone, not %d", nzones);
  {
    cgsize_t sizes[3]; /* Number of vertices, number of cells, number of boundary vertices */

    PetscCallCGNSRead(cg_zone_read(cgid, base, zone, buffer, sizes), *dm, 0);
    NVertices = sizes[0];
    NCells    = sizes[1];
  }

  PetscCall(PetscObjectSetName((PetscObject)*dm, basename));
  PetscCall(DMSetDimension(*dm, dim));
  coordDim = dim;

  // This is going to be a headache for mixed-topology and multiple sections. We may have to restore reading the data twice (once before  the SetChart
  // call to get this right but continuing for now with single section, single topology, one zone.
  // establish element ranges for my rank
  PetscInt    mystarte, myende, mystartv, myendv, myownede, myownedv;
  PetscLayout elem_map, vtx_map;
  PetscCall(PetscLayoutCreateFromSizes(comm, PETSC_DECIDE, NCells, 1, &elem_map));
  PetscCall(PetscLayoutCreateFromSizes(comm, PETSC_DECIDE, NVertices, 1, &vtx_map));
  PetscCall(PetscLayoutGetRange(elem_map, &mystarte, &myende));
  PetscCall(PetscLayoutGetLocalSize(elem_map, &myownede));
  PetscCall(PetscLayoutGetRange(vtx_map, &mystartv, &myendv));
  PetscCall(PetscLayoutGetLocalSize(vtx_map, &myownedv));

  // -- Build Plex in parallel
  DMPolytopeType dm_cell_type = DM_POLYTOPE_UNKNOWN;
  PetscInt       pOrder = 1, numClosure = -1;
  cgsize_t      *elements = NULL;
  int           *face_section_ids, *cell_section_ids, num_face_sections = 0, num_cell_sections = 0;
  PetscInt      *uniq_verts, nuniq_verts;
  {
    int        nsections;
    PetscInt  *elementsQ1, numCorners = -1;
    const int *perm;
    cgsize_t   start, end; // Throwaway

    cg_nsections(cgid, base, zone, &nsections);
    PetscCall(PetscMalloc2(nsections, &face_section_ids, nsections, &cell_section_ids));
    // Read element connectivity
    for (int index_sect = 1; index_sect <= nsections; index_sect++) {
      int      nbndry, parentFlag;
      PetscInt cell_dim;
      CGNS_ENUMT(ElementType_t) cellType;

      PetscCallCGNSRead(cg_section_read(cgid, base, zone, index_sect, buffer, &cellType, &start, &end, &nbndry, &parentFlag), *dm, 0);

      PetscCall(CGNSElementTypeGetTopologyInfo(cellType, &dm_cell_type, &numCorners, &cell_dim));
      // Skip over element that are not max dimension (ie. boundary elements)
      if (cell_dim == dim) cell_section_ids[num_cell_sections++] = index_sect;
      else if (cell_dim == dim - 1) face_section_ids[num_face_sections++] = index_sect;
    }
    PetscCheck(num_cell_sections == 1, comm, PETSC_ERR_SUP, "CGNS Reader does not support more than 1 full-dimension cell section");

    {
      int index_sect = cell_section_ids[0], nbndry, parentFlag;
      CGNS_ENUMT(ElementType_t) cellType;

      PetscCallCGNSRead(cg_section_read(cgid, base, zone, index_sect, buffer, &cellType, &start, &end, &nbndry, &parentFlag), *dm, 0);
      PetscCall(CGNSElementTypeGetDiscretizationInfo(cellType, &numClosure, &pOrder));
      PetscCall(PetscMalloc1(myownede * numClosure, &elements));
      PetscCallCGNSReadData(cgp_elements_read_data(cgid, base, zone, index_sect, mystarte + 1, myende, elements), *dm, 0);
      for (PetscInt v = 0; v < myownede * numClosure; ++v) elements[v] -= 1; // 0 based

      // Create corners-only connectivity
      PetscCall(CGNSElementTypeGetTopologyInfo(cellType, &dm_cell_type, &numCorners, NULL));
      PetscCall(PetscMalloc1(myownede * numCorners, &elementsQ1));
      PetscCall(DMPlexCGNSGetPermutation_Internal(dm_cell_type, numCorners, NULL, &perm));
      for (PetscInt e = 0; e < myownede; ++e) {
        for (PetscInt v = 0; v < numCorners; ++v) elementsQ1[e * numCorners + perm[v]] = elements[e * numClosure + v];
      }
    }

    // Build cell-vertex Plex
    PetscCall(DMPlexBuildFromCellListParallel(*dm, myownede, myownedv, NVertices, numCorners, elementsQ1, NULL, &uniq_verts));
    PetscCall(DMViewFromOptions(*dm, NULL, "-corner_dm_view"));
    {
      PetscInt pStart, pEnd;
      PetscCall(DMPlexGetChart(*dm, &pStart, &pEnd));
      nuniq_verts = (pEnd - pStart) - myownede;
    }
    PetscCall(PetscFree(elementsQ1));
  }

  if (interpolate) PetscCall(DMPlexInterpolateInPlace_Internal(*dm));

  // -- Create SF for naive nodal-data read to elements
  PetscSF plex_to_cgns_sf;
  {
    PetscInt     nleaves, num_comp;
    PetscInt    *leaf, num_leaves = 0;
    PetscInt     cStart, cEnd;
    const int   *perm;
    PetscSF      cgns_to_local_sf;
    PetscSection local_section;
    PetscFE      fe;

    // sfNatural requires PetscSection to handle DMDistribute, so we use PetscFE to define the section
    // Use number of components = 1 to work with just the nodes themselves
    PetscCall(PetscFECreateLagrangeByCell(PETSC_COMM_SELF, dim, 1, dm_cell_type, pOrder, PETSC_DETERMINE, &fe));
    PetscCall(PetscObjectSetName((PetscObject)fe, "FE for sfNatural"));
    PetscCall(DMAddField(*dm, NULL, (PetscObject)fe));
    PetscCall(DMCreateDS(*dm));
    PetscCall(PetscFEDestroy(&fe));

    PetscCall(DMGetLocalSection(*dm, &local_section));
    PetscCall(PetscSectionViewFromOptions(local_section, NULL, "-fe_natural_section_view"));
    PetscCall(PetscSectionGetFieldComponents(local_section, 0, &num_comp));
    PetscCall(PetscSectionGetStorageSize(local_section, &nleaves));
    nleaves /= num_comp;
    PetscCall(PetscMalloc1(nleaves, &leaf));
    for (PetscInt i = 0; i < nleaves; i++) leaf[i] = -1;

    // Get the permutation from CGNS closure numbering to PLEX closure numbering
    PetscCall(DMPlexCGNSGetPermutation_Internal(dm_cell_type, numClosure, NULL, &perm));
    PetscCall(DMPlexGetHeightStratum(*dm, 0, &cStart, &cEnd));
    for (PetscInt cell = cStart; cell < cEnd; ++cell) {
      PetscInt num_closure_dof, *closure_idx = NULL;

      PetscCall(DMPlexGetClosureIndices(*dm, local_section, local_section, cell, PETSC_FALSE, &num_closure_dof, &closure_idx, NULL, NULL));
      PetscAssert(numClosure * num_comp == num_closure_dof, comm, PETSC_ERR_PLIB, "Closure dof size does not match polytope");
      for (PetscInt i = 0; i < numClosure; i++) {
        PetscInt li = closure_idx[perm[i] * num_comp] / num_comp;
        if (li < 0) continue;

        PetscInt cgns_idx = elements[cell * numClosure + i];
        if (leaf[li] == -1) {
          leaf[li] = cgns_idx;
          num_leaves++;
        } else PetscAssert(leaf[li] == cgns_idx, PETSC_COMM_SELF, PETSC_ERR_PLIB, "leaf does not match previously set");
      }
      PetscCall(DMPlexRestoreClosureIndices(*dm, local_section, local_section, cell, PETSC_FALSE, &num_closure_dof, &closure_idx, NULL, NULL));
    }
    PetscAssert(num_leaves == nleaves, PETSC_COMM_SELF, PETSC_ERR_PLIB, "leaf count in closure does not match nleaves");
    PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)*dm), &cgns_to_local_sf));
    PetscCall(PetscSFSetGraphLayout(cgns_to_local_sf, vtx_map, nleaves, NULL, PETSC_USE_POINTER, leaf));
    PetscCall(PetscObjectSetName((PetscObject)cgns_to_local_sf, "CGNS to Plex SF"));
    PetscCall(PetscSFViewFromOptions(cgns_to_local_sf, NULL, "-CGNStoPlex_sf_view"));
    PetscCall(PetscFree(leaf));
    PetscCall(PetscFree(elements));

    { // Convert cgns_to_local to global_to_cgns
      PetscSF sectionsf, cgns_to_global_sf;

      PetscCall(DMGetSectionSF(*dm, &sectionsf));
      PetscCall(PetscSFComposeInverse(cgns_to_local_sf, sectionsf, &cgns_to_global_sf));
      PetscCall(PetscSFDestroy(&cgns_to_local_sf));
      PetscCall(PetscSFCreateInverseSF(cgns_to_global_sf, &plex_to_cgns_sf));
      PetscCall(PetscObjectSetName((PetscObject)plex_to_cgns_sf, "Global Plex to CGNS"));
      PetscCall(PetscSFDestroy(&cgns_to_global_sf));
    }
  }

  { // -- Set coordinates for DM
    PetscScalar *coords;
    float       *x[3];
    double      *xd[3];
    PetscBool    read_with_double;
    PetscFE      cfe;

    // Setup coordinate space first. Use pOrder here for isoparametric; revisit with CPEX-0045 High Order.
    PetscCall(PetscFECreateLagrangeByCell(PETSC_COMM_SELF, dim, coordDim, dm_cell_type, pOrder, PETSC_DETERMINE, &cfe));
    PetscCall(DMSetCoordinateDisc(*dm, cfe, PETSC_FALSE, PETSC_FALSE));
    PetscCall(PetscFEDestroy(&cfe));

    { // Determine if coords are written in single or double precision
      CGNS_ENUMT(DataType_t) datatype;

      PetscCallCGNSRead(cg_coord_info(cgid, base, zone, 1, &datatype, buffer), *dm, 0);
      read_with_double = datatype == CGNS_ENUMV(RealDouble) ? PETSC_TRUE : PETSC_FALSE;
    }

    // Read coords from file and set into component-major ordering
    if (read_with_double) PetscCall(PetscMalloc3(myownedv, &xd[0], myownedv, &xd[1], myownedv, &xd[2]));
    else PetscCall(PetscMalloc3(myownedv, &x[0], myownedv, &x[1], myownedv, &x[2]));
    PetscCall(PetscMalloc1(myownedv * coordDim, &coords));
    {
      cgsize_t sizes[3]; /* Number of vertices, number of cells, number of boundary vertices */
      cgsize_t range_min[3] = {mystartv + 1, 1, 1};
      cgsize_t range_max[3] = {myendv, 1, 1};
      int      ngrids, ncoords;

      PetscCallCGNSRead(cg_zone_read(cgid, base, zone, buffer, sizes), *dm, 0);
      PetscCallCGNSRead(cg_ngrids(cgid, base, zone, &ngrids), *dm, 0);
      PetscCheck(ngrids <= 1, PETSC_COMM_SELF, PETSC_ERR_LIB, "CGNS file must have a single grid, not %d", ngrids);
      PetscCallCGNSRead(cg_ncoords(cgid, base, zone, &ncoords), *dm, 0);
      PetscCheck(ncoords == coordDim, PETSC_COMM_SELF, PETSC_ERR_LIB, "CGNS file must have a coordinate array for each dimension, not %d", ncoords);
      if (read_with_double) {
        for (int d = 0; d < coordDim; ++d) PetscCallCGNSReadData(cgp_coord_read_data(cgid, base, zone, (d + 1), range_min, range_max, xd[d]), *dm, 0);
        if (coordDim >= 1) {
          for (PetscInt v = 0; v < myownedv; ++v) coords[v * coordDim + 0] = xd[0][v];
        }
        if (coordDim >= 2) {
          for (PetscInt v = 0; v < myownedv; ++v) coords[v * coordDim + 1] = xd[1][v];
        }
        if (coordDim >= 3) {
          for (PetscInt v = 0; v < myownedv; ++v) coords[v * coordDim + 2] = xd[2][v];
        }
      } else {
        for (int d = 0; d < coordDim; ++d) PetscCallCGNSReadData(cgp_coord_read_data(cgid, 1, zone, (d + 1), range_min, range_max, x[d]), *dm, 0);
        if (coordDim >= 1) {
          for (PetscInt v = 0; v < myownedv; ++v) coords[v * coordDim + 0] = x[0][v];
        }
        if (coordDim >= 2) {
          for (PetscInt v = 0; v < myownedv; ++v) coords[v * coordDim + 1] = x[1][v];
        }
        if (coordDim >= 3) {
          for (PetscInt v = 0; v < myownedv; ++v) coords[v * coordDim + 2] = x[2][v];
        }
      }
    }
    if (read_with_double) PetscCall(PetscFree3(xd[0], xd[1], xd[2]));
    else PetscCall(PetscFree3(x[0], x[1], x[2]));

    { // Reduce CGNS-ordered coordinate nodes to Plex ordering, and set DM's coordinates
      Vec          coord_global;
      MPI_Datatype unit;
      PetscScalar *coord_global_array;
      DM           cdm;

      PetscCall(DMGetCoordinateDM(*dm, &cdm));
      PetscCall(DMCreateGlobalVector(cdm, &coord_global));
      PetscCall(VecGetArrayWrite(coord_global, &coord_global_array));
      PetscCallMPI(MPI_Type_contiguous(coordDim, MPIU_SCALAR, &unit));
      PetscCallMPI(MPI_Type_commit(&unit));
      PetscCall(PetscSFReduceBegin(plex_to_cgns_sf, unit, coords, coord_global_array, MPI_REPLACE));
      PetscCall(PetscSFReduceEnd(plex_to_cgns_sf, unit, coords, coord_global_array, MPI_REPLACE));
      PetscCall(VecRestoreArrayWrite(coord_global, &coord_global_array));
      PetscCallMPI(MPI_Type_free(&unit));
      PetscCall(DMSetCoordinates(*dm, coord_global));
      PetscCall(VecDestroy(&coord_global));
    }
    PetscCall(PetscFree(coords));
  }

  PetscCall(DMViewFromOptions(*dm, NULL, "-corner_interpolated_dm_view"));

  int nbocos;
  PetscCallCGNSRead(cg_nbocos(cgid, base, zone, &nbocos), *dm, 0);
  // In order to extract boundary condition (boco) information into DMLabels, each rank holds:
  // - The local Plex
  // - Naively read CGNS face connectivity
  // - Naively read list of CGNS faces for each boco
  //
  // First, we need to build a mapping from the CGNS faces to the (probably off-rank) Plex face.
  // The CGNS faces that each rank owns is known globally via cgnsLayouts.
  // The cg2plexSF maps these CGNS face IDs to their (probably off-rank) Plex face.
  // The plexFaces array maps the (contiguous) leaves of cg2plexSF to the local Plex face point.
  //
  // Next, we read the list of CGNS faces for each boco and find the location of that face's owner using cgnsLayouts.
  // Then, we can communicate the label value to the local Plex which corresponds to the CGNS face.
  if (interpolate && num_face_sections != 0 && nbocos != 0) {
    PetscSection connSection;
    PetscInt     nCgFaces, nPlexFaces;
    PetscInt    *face_ids, *conn, *plexFaces;
    PetscSF      cg2plexSF;
    PetscLayout *cgnsLayouts;

    PetscCall(DMPlexCGNS_CreateCornersConnectivitySection(*dm, cgid, base, zone, num_face_sections, face_section_ids, &connSection, NULL, &face_ids, &cgnsLayouts, &conn));
    {
      PetscBool view_connectivity = PETSC_FALSE;
      PetscCall(PetscOptionsGetBool(NULL, NULL, "-dm_plex_cgns_view_face_connectivity", &view_connectivity, NULL));
      if (view_connectivity) PetscCall(PetscSectionArrayView(connSection, conn, PETSC_INT, NULL));
    }
    PetscCall(DMPlexCGNS_MatchCGNSFacesToPlexFaces(*dm, nuniq_verts, uniq_verts, myownede, NVertices, connSection, face_ids, conn, &cg2plexSF, &plexFaces));
    PetscCall(PetscSFGetGraph(cg2plexSF, NULL, &nPlexFaces, NULL, NULL));
    {
      PetscInt start, end;
      PetscCall(PetscSectionGetChart(connSection, &start, &end));
      nCgFaces = end - start;
    }

    PetscInt *plexFaceValues, *cgFaceValues;
    PetscCall(PetscMalloc2(nPlexFaces, &plexFaceValues, nCgFaces, &cgFaceValues));
    for (PetscInt BC = 1; BC <= nbocos; BC++) {
      cgsize_t *points;
      CGBCInfo  bcinfo;
      PetscBool is_faceset  = PETSC_FALSE;
      PetscInt  label_value = 1;

      PetscCallCGNSRead(cg_boco_info(cgid, base, zone, BC, bcinfo.name, &bcinfo.bctype, &bcinfo.pointtype, &bcinfo.npoints, bcinfo.normal, &bcinfo.nnormals, &bcinfo.normal_datatype, &bcinfo.ndatasets), *dm, 0);

      PetscCall(PetscStrbeginswith(bcinfo.name, "FaceSet", &is_faceset));
      if (is_faceset) {
        size_t faceset_len;
        PetscCall(PetscStrlen("FaceSet", &faceset_len));
        PetscCall(PetscStrtoInt(bcinfo.name + faceset_len, &label_value));
      }
      const char *label_name = is_faceset ? "Face Sets" : bcinfo.name;

      if (bcinfo.npoints < 1) continue;

      PetscLayout bc_layout;
      PetscInt    bcStart, bcEnd, bcSize;
      PetscCall(PetscLayoutCreateFromSizes(comm, PETSC_DECIDE, bcinfo.npoints, 1, &bc_layout));
      PetscCall(PetscLayoutGetRange(bc_layout, &bcStart, &bcEnd));
      PetscCall(PetscLayoutGetLocalSize(bc_layout, &bcSize));
      PetscCall(PetscLayoutDestroy(&bc_layout));
      PetscCall(DMGetWorkArray(*dm, bcSize, MPIU_CGSIZE, &points));

      const char *labels[] = {"Zone_t", "ZoneBC_t", "BC_t", "PointList"};
      PetscCallCGNSRead(cg_golist(cgid, base, 4, (char **)labels, (int[]){zone, 1, BC, 0}), *dm, 0);
      PetscCallCGNSReadData(cgp_ptlist_read_data(cgid, bcStart + 1, bcEnd, points), *dm, 0);

      PetscInt    *label_values;
      PetscSFNode *remotes;
      PetscCall(PetscMalloc2(bcSize, &remotes, bcSize, &label_values));
      for (PetscInt p = 0; p < bcSize; p++) {
        PetscMPIInt bcrank;
        PetscInt    bcidx;

        PetscCall(PetscLayoutFindOwnerIndex_CGNSSectionLayouts(cgnsLayouts, num_face_sections, points[p], &bcrank, &bcidx, NULL));
        remotes[p].rank  = bcrank;
        remotes[p].index = bcidx;
        label_values[p]  = label_value;
      }
      PetscCall(DMRestoreWorkArray(*dm, bcSize, MPIU_CGSIZE, &points));

      { // Communicate the BC values to their Plex-face owners
        PetscSF cg2bcSF;
        DMLabel label;

        for (PetscInt i = 0; i < nCgFaces; i++) cgFaceValues[i] = -1;
        for (PetscInt i = 0; i < nPlexFaces; i++) plexFaceValues[i] = -1;

        PetscCall(PetscSFCreate(comm, &cg2bcSF));
        PetscCall(PetscSFSetGraph(cg2bcSF, nCgFaces, bcSize, NULL, PETSC_COPY_VALUES, remotes, PETSC_USE_POINTER));

        PetscCall(PetscSFReduceBegin(cg2bcSF, MPIU_INT, label_values, cgFaceValues, MPI_REPLACE));
        PetscCall(PetscSFReduceEnd(cg2bcSF, MPIU_INT, label_values, cgFaceValues, MPI_REPLACE));
        PetscCall(PetscSFBcastBegin(cg2plexSF, MPIU_INT, cgFaceValues, plexFaceValues, MPI_REPLACE));
        PetscCall(PetscSFBcastEnd(cg2plexSF, MPIU_INT, cgFaceValues, plexFaceValues, MPI_REPLACE));
        PetscCall(PetscSFDestroy(&cg2bcSF));
        PetscCall(PetscFree2(remotes, label_values));

        // Set the label values for the communicated faces
        PetscCall(DMGetLabel(*dm, label_name, &label));
        if (label == NULL) {
          PetscCall(DMCreateLabel(*dm, label_name));
          PetscCall(DMGetLabel(*dm, label_name, &label));
        }
        for (PetscInt i = 0; i < nPlexFaces; i++) {
          if (plexFaceValues[i] == -1) continue;
          PetscCall(DMLabelSetValue(label, plexFaces[i], plexFaceValues[i]));
        }
      }
    }
    PetscCall(PetscFree2(plexFaceValues, cgFaceValues));
    PetscCall(PetscFree(plexFaces));
    PetscCall(PetscSFDestroy(&cg2plexSF));
    PetscCall(PetscFree(conn));
    for (PetscInt s = 0; s < num_face_sections; s++) PetscCall(PetscLayoutDestroy(&cgnsLayouts[s]));
    PetscCall(PetscSectionDestroy(&connSection));
    PetscCall(PetscFree(cgnsLayouts));
    PetscCall(PetscFree(face_ids));
  }
  PetscCall(PetscFree(uniq_verts));
  PetscCall(PetscFree2(face_section_ids, cell_section_ids));

  // -- Set sfNatural for solution vectors in CGNS file
  // NOTE: We set sfNatural to be the map between the original CGNS ordering of nodes and the Plex ordering of nodes.
  PetscCall(PetscSFViewFromOptions(plex_to_cgns_sf, NULL, "-sfNatural_init_view"));
  PetscCall(DMSetNaturalSF(*dm, plex_to_cgns_sf));
  PetscCall(DMSetUseNatural(*dm, PETSC_TRUE));
  PetscCall(PetscSFDestroy(&plex_to_cgns_sf));

  PetscCall(PetscLayoutDestroy(&elem_map));
  PetscCall(PetscLayoutDestroy(&vtx_map));
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
  PetscCallMPI(MPIU_Allreduce(&owned_node, num_global_nodes, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)dm)));
  *node_l2g = nodes;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMView_PlexCGNS(DM dm, PetscViewer viewer)
{
  MPI_Comm          comm = PetscObjectComm((PetscObject)dm);
  PetscViewer_CGNS *cgv  = (PetscViewer_CGNS *)viewer->data;
  PetscInt          fvGhostStart;
  PetscInt          topo_dim, coord_dim, num_global_elems;
  PetscInt          cStart, cEnd, num_local_nodes, num_global_nodes, nStart, nEnd, fStart, fEnd;
  const PetscInt   *node_l2g;
  Vec               coord;
  DM                colloc_dm, cdm;
  PetscMPIInt       size;
  const char       *dm_name;
  int               base, zone;
  cgsize_t          isize[3], elem_offset = 0;

  PetscFunctionBegin;
  if (!cgv->file_num) {
    PetscInt time_step;
    PetscCall(DMGetOutputSequenceNumber(dm, &time_step, NULL));
    PetscCall(PetscViewerCGNSFileOpen_Internal(viewer, time_step));
  }
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(DMGetDimension(dm, &topo_dim));
  PetscCall(DMGetCoordinateDim(dm, &coord_dim));
  PetscCall(PetscObjectGetName((PetscObject)dm, &dm_name));
  PetscCallCGNSWrite(cg_base_write(cgv->file_num, dm_name, topo_dim, coord_dim, &base), dm, viewer);
  PetscCallCGNS(cg_goto(cgv->file_num, base, NULL));
  PetscCallCGNSWrite(cg_dataclass_write(CGNS_ENUMV(NormalizedByDimensional)), dm, viewer);

  {
    PetscFE        fe, fe_coord;
    PetscClassId   ds_id;
    PetscDualSpace dual_space, dual_space_coord;
    PetscInt       num_fields, field_order = -1, field_order_coord;
    PetscBool      is_simplex;
    PetscCall(DMGetNumFields(dm, &num_fields));
    if (num_fields > 0) {
      PetscCall(DMGetField(dm, 0, NULL, (PetscObject *)&fe));
      PetscCall(PetscObjectGetClassId((PetscObject)fe, &ds_id));
      if (ds_id != PETSCFE_CLASSID) {
        fe = NULL;
        if (ds_id == PETSCFV_CLASSID) field_order = -1; // use whatever is present for coords; field will be CellCenter
        else field_order = 1;                           // assume vertex-based linear elements
      }
    } else fe = NULL;
    if (fe) {
      PetscCall(PetscFEGetDualSpace(fe, &dual_space));
      PetscCall(PetscDualSpaceGetOrder(dual_space, &field_order));
    }
    PetscCall(DMGetCoordinateDM(dm, &cdm));
    PetscCall(DMGetField(cdm, 0, NULL, (PetscObject *)&fe_coord));
    {
      PetscClassId id;
      PetscCall(PetscObjectGetClassId((PetscObject)fe_coord, &id));
      if (id != PETSCFE_CLASSID) fe_coord = NULL;
    }
    if (fe_coord) {
      PetscCall(PetscFEGetDualSpace(fe_coord, &dual_space_coord));
      PetscCall(PetscDualSpaceGetOrder(dual_space_coord, &field_order_coord));
    } else field_order_coord = 1;
    if (field_order > 0 && field_order != field_order_coord) {
      PetscInt quadrature_order = field_order;
      PetscCall(DMClone(dm, &colloc_dm));
      { // Inform the new colloc_dm that it is a coordinate DM so isoperiodic affine corrections can be applied
        const PetscSF *face_sfs;
        PetscInt       num_face_sfs;
        PetscCall(DMPlexGetIsoperiodicFaceSF(dm, &num_face_sfs, &face_sfs));
        PetscCall(DMPlexSetIsoperiodicFaceSF(colloc_dm, num_face_sfs, (PetscSF *)face_sfs));
        if (face_sfs) colloc_dm->periodic.setup = DMPeriodicCoordinateSetUp_Internal;
      }
      PetscCall(DMPlexIsSimplex(dm, &is_simplex));
      PetscCall(PetscFECreateLagrange(comm, topo_dim, coord_dim, is_simplex, field_order, quadrature_order, &fe));
      PetscCall(DMSetCoordinateDisc(colloc_dm, fe, PETSC_FALSE, PETSC_TRUE));
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
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  PetscCall(DMPlexGetCellTypeStratum(dm, DM_POLYTOPE_FV_GHOST, &fvGhostStart, NULL));
  if (fvGhostStart >= 0) cEnd = fvGhostStart;
  num_global_elems = cEnd - cStart;
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &num_global_elems, 1, MPIU_INT, MPI_SUM, comm));
  isize[0] = num_global_nodes;
  isize[1] = num_global_elems;
  isize[2] = 0;
  PetscCallCGNSWrite(cg_zone_write(cgv->file_num, base, "Zone", isize, CGNS_ENUMV(Unstructured), &zone), dm, viewer);

  cgsize_t e_owned, e_global, e_start;
  {
    const PetscScalar *X;
    PetscScalar       *x;
    int                coord_ids[3];

    PetscCall(VecGetArrayRead(coord, &X));
    for (PetscInt d = 0; d < coord_dim; d++) {
      const double exponents[] = {0, 1, 0, 0, 0};
      char         coord_name[64];
      PetscCall(PetscSNPrintf(coord_name, sizeof coord_name, "Coordinate%c", 'X' + (int)d));
      PetscCallCGNSWrite(cgp_coord_write(cgv->file_num, base, zone, CGNS_ENUMV(RealDouble), coord_name, &coord_ids[d]), dm, viewer);
      PetscCallCGNS(cg_goto(cgv->file_num, base, "Zone_t", zone, "GridCoordinates", 0, coord_name, 0, NULL));
      PetscCallCGNSWrite(cg_exponents_write(CGNS_ENUMV(RealDouble), exponents), dm, viewer);
    }

    int        section;
    cgsize_t  *conn = NULL;
    const int *perm;
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
        PetscCallCGNSWriteData(cgp_coord_write_data(cgv->file_num, base, zone, coord_ids[d], &start, &end, x), dm, viewer);
      }
      PetscCall(PetscFree(x));
      PetscCall(VecRestoreArrayRead(coord, &X));
    }

    e_owned = cEnd - cStart;
    if (e_owned > 0) {
      DMPolytopeType cell_type;

      PetscCall(DMPlexGetCellType(dm, cStart, &cell_type));
      for (PetscInt i = cStart, c = 0; i < cEnd; i++) {
        PetscInt closure_dof, *closure_indices, elem_size;

        PetscCall(DMPlexGetClosureIndices(cdm, cdm->localSection, cdm->localSection, i, PETSC_FALSE, &closure_dof, &closure_indices, NULL, NULL));
        elem_size = closure_dof / coord_dim;
        if (!conn) PetscCall(PetscMalloc1(e_owned * elem_size, &conn));
        PetscCall(DMPlexCGNSGetPermutation_Internal(cell_type, closure_dof / coord_dim, &element_type, &perm));
        for (PetscInt j = 0; j < elem_size; j++) conn[c++] = node_l2g[closure_indices[perm[j] * coord_dim] / coord_dim] + 1;
        PetscCall(DMPlexRestoreClosureIndices(cdm, cdm->localSection, cdm->localSection, i, PETSC_FALSE, &closure_dof, &closure_indices, NULL, NULL));
      }
    }

    { // Get global element_type (for ranks that do not have owned elements)
      PetscInt local_element_type, global_element_type;

      local_element_type = e_owned > 0 ? (PetscInt)element_type : -1;
      PetscCallMPI(MPIU_Allreduce(&local_element_type, &global_element_type, 1, MPIU_INT, MPI_MAX, comm));
      if (local_element_type != -1)
        PetscCheck(local_element_type == global_element_type, PETSC_COMM_SELF, PETSC_ERR_SUP, "Ranks with different element types not supported. Local element type is %s, but global is %s", cg_ElementTypeName(local_element_type), cg_ElementTypeName(global_element_type));
      element_type = (CGNS_ENUMT(ElementType_t))global_element_type;
    }
    PetscCallMPI(MPIU_Allreduce(&e_owned, &e_global, 1, MPIU_CGSIZE, MPI_SUM, comm));
    PetscCheck(e_global == num_global_elems, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected number of elements %" PRIdCGSIZE " vs %" PetscInt_FMT, e_global, num_global_elems);
    e_start = 0;
    PetscCallMPI(MPI_Exscan(&e_owned, &e_start, 1, MPIU_CGSIZE, MPI_SUM, comm));
    e_start += elem_offset;
    PetscCallCGNSWrite(cgp_section_write(cgv->file_num, base, zone, "Elem", element_type, 1, e_global, 0, &section), dm, viewer);
    PetscCallCGNSWriteData(cgp_elements_write_data(cgv->file_num, base, zone, section, e_start + 1, e_start + e_owned, conn), dm, viewer);
    elem_offset = e_global;
    PetscCall(PetscFree(conn));

    cgv->base            = base;
    cgv->zone            = zone;
    cgv->node_l2g        = node_l2g;
    cgv->num_local_nodes = num_local_nodes;
    cgv->nStart          = nStart;
    cgv->nEnd            = nEnd;
    cgv->eStart          = e_start;
    cgv->eEnd            = e_start + e_owned;
    if (1) {
      PetscMPIInt rank;
      int        *efield;
      int         sol, field;
      DMLabel     label;
      PetscCallMPI(MPI_Comm_rank(comm, &rank));
      PetscCall(PetscMalloc1(e_owned, &efield));
      for (PetscInt i = 0; i < e_owned; i++) efield[i] = rank;
      PetscCallCGNSWrite(cg_sol_write(cgv->file_num, base, zone, "CellInfo", CGNS_ENUMV(CellCenter), &sol), dm, viewer);
      PetscCallCGNSWrite(cgp_field_write(cgv->file_num, base, zone, sol, CGNS_ENUMV(Integer), "Rank", &field), dm, viewer);
      cgsize_t start = e_start + 1, end = e_start + e_owned;
      PetscCallCGNSWriteData(cgp_field_write_data(cgv->file_num, base, zone, sol, field, &start, &end, efield), dm, viewer);
      PetscCall(DMGetLabel(dm, "Cell Sets", &label));
      if (label) {
        for (PetscInt c = cStart; c < cEnd; c++) {
          PetscInt value;
          PetscCall(DMLabelGetValue(label, c, &value));
          efield[c - cStart] = value;
        }
        PetscCallCGNSWrite(cgp_field_write(cgv->file_num, base, zone, sol, CGNS_ENUMV(Integer), "CellSet", &field), dm, viewer);
        PetscCallCGNSWriteData(cgp_field_write_data(cgv->file_num, base, zone, sol, field, &start, &end, efield), dm, viewer);
      }
      PetscCall(PetscFree(efield));
    }
  }

  DMLabel  fsLabel;
  PetscInt num_fs_global;
  IS       fsValuesGlobalIS;
  PetscCall(DMGetLabel(dm, "Face Sets", &fsLabel));
  PetscCall(DMLabelGetValueISGlobal(comm, fsLabel, PETSC_TRUE, &fsValuesGlobalIS));
  PetscCall(ISGetSize(fsValuesGlobalIS, &num_fs_global));

  if (num_fs_global > 0) {
    CGNS_ENUMT(ElementType_t) element_type = CGNS_ENUMV(ElementTypeNull);
    const PetscInt *fsValuesLocal;
    IS              stratumIS, fsFacesAll;
    int             section;
    const int      *perm;
    cgsize_t        f_owned = 0, f_global, f_start;
    cgsize_t       *parents, *conn = NULL;
    PetscInt        fStart, fEnd;

    PetscInt num_fs_local;
    IS       fsValuesLocalIS;

    if (fsLabel) {
      PetscCall(DMLabelGetNonEmptyStratumValuesIS(fsLabel, &fsValuesLocalIS));
      PetscCall(ISGetSize(fsValuesLocalIS, &num_fs_local));
      PetscCall(ISGetIndices(fsValuesLocalIS, &fsValuesLocal));
    } else num_fs_local = 0;

    PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
    { // Get single IS without duplicates of the local face IDs in the FaceSets
      IS *fsPoints = NULL;

      PetscCall(PetscMalloc1(num_fs_local, &fsPoints));
      for (PetscInt fs = 0; fs < num_fs_local; ++fs) PetscCall(DMLabelGetStratumIS(fsLabel, fsValuesLocal[fs], &fsPoints[fs]));
      PetscCall(ISConcatenate(PETSC_COMM_SELF, num_fs_local, fsPoints, &fsFacesAll));
      PetscCall(ISSortRemoveDups(fsFacesAll));
      PetscCall(ISGeneralFilter(fsFacesAll, fStart, fEnd)); // Remove non-face mesh points from the IS
      {
        PetscInt f_owned_int;
        PetscCall(ISGetSize(fsFacesAll, &f_owned_int));
        f_owned = f_owned_int;
      }
      for (PetscInt fs = 0; fs < num_fs_local; ++fs) PetscCall(ISDestroy(&fsPoints[fs]));
      PetscCall(PetscFree(fsPoints));
    }
    PetscCall(ISRestoreIndices(fsValuesLocalIS, &fsValuesLocal));
    PetscCall(ISDestroy(&fsValuesLocalIS));

    {
      const PetscInt *faces;
      DMPolytopeType  cell_type, cell_type_f;
      PetscInt        closure_dof = -1, closure_dof_f;

      PetscCall(ISGetIndices(fsFacesAll, &faces));
      if (f_owned) PetscCall(DMPlexGetCellType(dm, faces[0], &cell_type));
      PetscCall(PetscCalloc1(f_owned * 2, &parents));
      for (PetscInt f = 0, c = 0; f < f_owned; f++) {
        PetscInt      *closure_indices, elem_size;
        const PetscInt face = faces[f];

        PetscCall(DMPlexGetCellType(dm, face, &cell_type_f));
        PetscCheck(cell_type_f == cell_type, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Only mono-topology face sets are supported currently. Face %" PetscInt_FMT " is %s, which is different than the previous type %s", face, DMPolytopeTypes[cell_type_f], DMPolytopeTypes[cell_type]);

        // Get connectivity of the face
        PetscCall(DMPlexGetClosureIndices(cdm, cdm->localSection, cdm->localSection, face, PETSC_FALSE, &closure_dof_f, &closure_indices, NULL, NULL));
        elem_size = closure_dof_f / coord_dim;
        if (!conn) {
          PetscCall(PetscMalloc1(f_owned * elem_size, &conn));
          closure_dof = closure_dof_f;
        }
        PetscCheck(closure_dof_f == closure_dof, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Closure of face %" PetscInt_FMT " has %" PetscInt_FMT " dofs instead of the previously written %" PetscInt_FMT " dofs. Only mono-topology face sets are supported currently", face, closure_dof_f, closure_dof);
        PetscCall(DMPlexCGNSGetPermutation_Internal(cell_type, elem_size, &element_type, &perm));
        for (PetscInt j = 0; j < elem_size; j++) conn[c++] = node_l2g[closure_indices[perm[j] * coord_dim] / coord_dim] + 1;
        PetscCall(DMPlexRestoreClosureIndices(cdm, cdm->localSection, cdm->localSection, face, PETSC_FALSE, &closure_dof_f, &closure_indices, NULL, NULL));
      }
      PetscCall(ISRestoreIndices(fsFacesAll, &faces));
    }

    {   // Write connectivity for face sets
      { // Get global element type
        PetscInt local_element_type, global_element_type;

        local_element_type = f_owned > 0 ? (PetscInt)element_type : -1;
        PetscCallMPI(MPIU_Allreduce(&local_element_type, &global_element_type, 1, MPIU_INT, MPI_MAX, comm));
        if (local_element_type != -1)
          PetscCheck(local_element_type == global_element_type, PETSC_COMM_SELF, PETSC_ERR_SUP, "Ranks with different element types not supported. Local element type is %s, but global is %s", cg_ElementTypeName(local_element_type), cg_ElementTypeName(global_element_type));
        element_type = (CGNS_ENUMT(ElementType_t))global_element_type;
      }
      PetscCallMPI(MPIU_Allreduce(&f_owned, &f_global, 1, MPIU_CGSIZE, MPI_SUM, comm));
      f_start = 0;
      PetscCallMPI(MPI_Exscan(&f_owned, &f_start, 1, MPIU_CGSIZE, MPI_SUM, comm));
      f_start += elem_offset;
      PetscCallCGNSWrite(cgp_section_write(cgv->file_num, base, zone, "Faces", element_type, elem_offset + 1, elem_offset + f_global, 0, &section), dm, viewer);
      PetscCallCGNSWriteData(cgp_elements_write_data(cgv->file_num, base, zone, section, f_start + 1, f_start + f_owned, conn), dm, viewer);

      PetscCall(PetscFree(conn));
      PetscCall(PetscFree(parents));
    }

    const PetscInt *fsValuesGlobal = NULL;
    PetscCall(ISGetIndices(fsValuesGlobalIS, &fsValuesGlobal));
    for (PetscInt fs = 0; fs < num_fs_global; ++fs) {
      int            BC;
      const PetscInt fsID    = fsValuesGlobal[fs];
      PetscInt      *fs_pnts = NULL;
      char           bc_name[33];
      cgsize_t       fs_start, fs_owned, fs_global;
      cgsize_t      *fs_pnts_cg;

      PetscCall(DMLabelGetStratumIS(fsLabel, fsID, &stratumIS));
      if (stratumIS) { // Get list of only face points
        PetscSegBuffer  fs_pntsSB;
        PetscCount      fs_owned_count;
        PetscInt        nstratumPnts;
        const PetscInt *stratumPnts;

        PetscCall(PetscSegBufferCreate(sizeof(PetscInt), 16, &fs_pntsSB));
        PetscCall(ISGetIndices(stratumIS, &stratumPnts));
        PetscCall(ISGetSize(stratumIS, &nstratumPnts));
        for (PetscInt i = 0; i < nstratumPnts; i++) {
          PetscInt *fs_pnts_buffer, stratumPnt = stratumPnts[i];
          if (stratumPnt < fStart || stratumPnt >= fEnd) continue; // Skip non-face points
          PetscCall(PetscSegBufferGetInts(fs_pntsSB, 1, &fs_pnts_buffer));
          *fs_pnts_buffer = stratumPnt;
        }
        PetscCall(PetscSegBufferGetSize(fs_pntsSB, &fs_owned_count));
        fs_owned = fs_owned_count;
        PetscCall(PetscSegBufferExtractAlloc(fs_pntsSB, &fs_pnts));

        PetscCall(PetscSegBufferDestroy(&fs_pntsSB));
        PetscCall(ISRestoreIndices(stratumIS, &stratumPnts));
        PetscCall(ISDestroy(&stratumIS));
      } else fs_owned = 0;

      PetscCallMPI(MPIU_Allreduce(&fs_owned, &fs_global, 1, MPIU_CGSIZE, MPI_SUM, comm));
      fs_start = 0;
      PetscCallMPI(MPI_Exscan(&fs_owned, &fs_start, 1, MPIU_CGSIZE, MPI_SUM, comm));
      PetscCheck(fs_start + fs_owned <= fs_global, PETSC_COMM_SELF, PETSC_ERR_PLIB, "End range of point set (%" PRIdCGSIZE ") greater than global point set size (%" PRIdCGSIZE ")", fs_start + fs_owned, fs_global);

      PetscCall(PetscSNPrintf(bc_name, sizeof bc_name, "FaceSet%" PetscInt_FMT, fsID));
      PetscCallCGNSWrite(cg_boco_write(cgv->file_num, base, zone, bc_name, CGNS_ENUMV(BCTypeNull), CGNS_ENUMV(PointList), fs_global, NULL, &BC), dm, viewer);

      PetscCall(PetscMalloc1(fs_owned, &fs_pnts_cg));
      for (PetscInt i = 0; i < fs_owned; i++) {
        PetscInt is_idx;

        PetscCall(ISLocate(fsFacesAll, fs_pnts[i], &is_idx));
        PetscCheck(is_idx >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Could not find face %" PetscInt_FMT " in list of all local face points", fs_pnts[i]);
        fs_pnts_cg[i] = is_idx + f_start + 1;
      }

      const char *labels[] = {"Zone_t", "ZoneBC_t", "BC_t", "PointList"};
      PetscCallCGNSWrite(cg_golist(cgv->file_num, base, 4, (char **)labels, (int[]){zone, 1, BC, 0}), dm, 0);
      PetscCallCGNSWriteData(cgp_ptlist_write_data(cgv->file_num, fs_start + 1, fs_start + fs_owned, fs_pnts_cg), dm, viewer);

      CGNS_ENUMT(GridLocation_t) grid_loc = CGNS_ENUMV(GridLocationNull);
      if (topo_dim == 3) grid_loc = CGNS_ENUMV(FaceCenter);
      else if (topo_dim == 2) grid_loc = CGNS_ENUMV(EdgeCenter);
      else if (topo_dim == 1) grid_loc = CGNS_ENUMV(Vertex);
      PetscCallCGNSWriteData(cg_boco_gridlocation_write(cgv->file_num, base, zone, BC, grid_loc), dm, viewer);

      PetscCall(PetscFree(fs_pnts_cg));
      PetscCall(PetscFree(fs_pnts));
    }
    PetscCall(ISDestroy(&fsFacesAll));
    PetscCall(ISRestoreIndices(fsValuesGlobalIS, &fsValuesGlobal));
    elem_offset += f_global;
  }
  PetscCall(ISDestroy(&fsValuesGlobalIS));

  PetscCall(DMDestroy(&colloc_dm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecView_Plex_Local_CGNS(Vec V, PetscViewer viewer)
{
  PetscViewer_CGNS  *cgv = (PetscViewer_CGNS *)viewer->data;
  DM                 dm;
  PetscSection       section;
  PetscInt           time_step, num_fields, pStart, pEnd, fvGhostStart;
  PetscReal          time, *time_slot;
  size_t            *step_slot;
  const PetscScalar *v;
  char               solution_name[PETSC_MAX_PATH_LEN];
  int                sol;

  PetscFunctionBegin;
  PetscCall(VecGetDM(V, &dm));
  PetscCall(DMGetLocalSection(dm, &section));
  PetscCall(PetscSectionGetChart(section, &pStart, &pEnd));
  PetscCall(DMPlexGetCellTypeStratum(dm, DM_POLYTOPE_FV_GHOST, &fvGhostStart, NULL));
  if (fvGhostStart >= 0) pEnd = fvGhostStart;

  if (!cgv->node_l2g) PetscCall(DMView(dm, viewer));
  if (!cgv->grid_loc) { // Determine if writing to cell-centers or to nodes
    PetscInt cStart, cEnd;
    PetscInt local_grid_loc, global_grid_loc;

    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    if (fvGhostStart >= 0) cEnd = fvGhostStart;
    if (cgv->num_local_nodes == 0) local_grid_loc = -1;
    else if (cStart == pStart && cEnd == pEnd) local_grid_loc = CGNS_ENUMV(CellCenter);
    else local_grid_loc = CGNS_ENUMV(Vertex);

    PetscCallMPI(MPIU_Allreduce(&local_grid_loc, &global_grid_loc, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)viewer)));
    if (local_grid_loc != -1)
      PetscCheck(local_grid_loc == global_grid_loc, PETSC_COMM_SELF, PETSC_ERR_SUP, "Ranks with different grid locations not supported. Local has %" PetscInt_FMT ", allreduce returned %" PetscInt_FMT, local_grid_loc, global_grid_loc);
    PetscCheck((global_grid_loc == CGNS_ENUMV(CellCenter)) || (global_grid_loc == CGNS_ENUMV(Vertex)), PetscObjectComm((PetscObject)viewer), PETSC_ERR_SUP, "Grid location should only be CellCenter (%d) or Vertex(%d), but have %" PetscInt_FMT, CGNS_ENUMV(CellCenter), CGNS_ENUMV(Vertex), global_grid_loc);
    cgv->grid_loc = (CGNS_ENUMT(GridLocation_t))global_grid_loc;
  }
  if (!cgv->nodal_field) {
    switch (cgv->grid_loc) {
    case CGNS_ENUMV(Vertex): {
      PetscCall(PetscMalloc1(cgv->nEnd - cgv->nStart, &cgv->nodal_field));
    } break;
    case CGNS_ENUMV(CellCenter): {
      PetscCall(PetscMalloc1(cgv->eEnd - cgv->eStart, &cgv->nodal_field));
    } break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Can only write for Vertex and CellCenter grid locations");
    }
  }
  if (!cgv->output_times) PetscCall(PetscSegBufferCreate(sizeof(PetscReal), 20, &cgv->output_times));
  if (!cgv->output_steps) PetscCall(PetscSegBufferCreate(sizeof(size_t), 20, &cgv->output_steps));

  PetscCall(DMGetOutputSequenceNumber(dm, &time_step, &time));
  if (time_step < 0) {
    time_step = 0;
    time      = 0.;
  }
  // Avoid "Duplicate child name found" error by not writing an already-written solution.
  // This usually occurs when a solution is written and then diverges on the very next timestep.
  if (time_step == cgv->previous_output_step) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscSegBufferGet(cgv->output_times, 1, &time_slot));
  *time_slot = time;
  PetscCall(PetscSegBufferGet(cgv->output_steps, 1, &step_slot));
  *step_slot = cgv->previous_output_step = time_step;
  PetscCall(PetscSNPrintf(solution_name, sizeof solution_name, "FlowSolution%" PetscInt_FMT, time_step));
  PetscCallCGNSWrite(cg_sol_write(cgv->file_num, cgv->base, cgv->zone, solution_name, cgv->grid_loc, &sol), V, viewer);
  PetscCall(VecGetArrayRead(V, &v));
  PetscCall(PetscSectionGetNumFields(section, &num_fields));
  for (PetscInt field = 0; field < num_fields; field++) {
    PetscInt    ncomp;
    const char *field_name;
    PetscCall(PetscSectionGetFieldName(section, field, &field_name));
    PetscCall(PetscSectionGetFieldComponents(section, field, &ncomp));
    for (PetscInt comp = 0; comp < ncomp; comp++) {
      int         cgfield;
      const char *comp_name;
      char        cgns_field_name[32]; // CGNS max field name is 32
      CGNS_ENUMT(DataType_t) datatype;
      PetscCall(PetscSectionGetComponentName(section, field, comp, &comp_name));
      if (ncomp == 1 && comp_name[0] == '0' && comp_name[1] == '\0' && field_name[0] != '\0') PetscCall(PetscStrncpy(cgns_field_name, field_name, sizeof cgns_field_name));
      else if (field_name[0] == '\0') PetscCall(PetscStrncpy(cgns_field_name, comp_name, sizeof cgns_field_name));
      else PetscCall(PetscSNPrintf(cgns_field_name, sizeof cgns_field_name, "%s.%s", field_name, comp_name));
      PetscCall(PetscCGNSDataType(PETSC_SCALAR, &datatype));
      PetscCallCGNSWrite(cgp_field_write(cgv->file_num, cgv->base, cgv->zone, sol, datatype, cgns_field_name, &cgfield), V, viewer);
      for (PetscInt p = pStart, n = 0; p < pEnd; p++) {
        PetscInt off, dof;
        PetscCall(PetscSectionGetFieldDof(section, p, field, &dof));
        if (dof == 0) continue;
        PetscCall(PetscSectionGetFieldOffset(section, p, field, &off));
        for (PetscInt c = comp; c < dof; c += ncomp, n++) {
          switch (cgv->grid_loc) {
          case CGNS_ENUMV(Vertex): {
            PetscInt gn = cgv->node_l2g[n];
            if (gn < cgv->nStart || cgv->nEnd <= gn) continue;
            cgv->nodal_field[gn - cgv->nStart] = v[off + c];
          } break;
          case CGNS_ENUMV(CellCenter): {
            cgv->nodal_field[n] = v[off + c];
          } break;
          default:
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Can only pack for Vertex and CellCenter grid locations");
          }
        }
      }
      // CGNS nodes use 1-based indexing
      cgsize_t start, end;
      switch (cgv->grid_loc) {
      case CGNS_ENUMV(Vertex): {
        start = cgv->nStart + 1;
        end   = cgv->nEnd;
      } break;
      case CGNS_ENUMV(CellCenter): {
        start = cgv->eStart + 1;
        end   = cgv->eEnd;
      } break;
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Can only write for Vertex and CellCenter grid locations");
      }
      PetscCallCGNSWriteData(cgp_field_write_data(cgv->file_num, cgv->base, cgv->zone, sol, cgfield, &start, &end, cgv->nodal_field), V, viewer);
    }
  }
  PetscCall(VecRestoreArrayRead(V, &v));
  PetscCall(PetscViewerCGNSCheckBatch_Internal(viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecLoad_Plex_CGNS_Internal(Vec V, PetscViewer viewer)
{
  MPI_Comm          comm;
  char              buffer[CGIO_MAX_NAME_LENGTH + 1];
  PetscViewer_CGNS *cgv                 = (PetscViewer_CGNS *)viewer->data;
  int               cgid                = cgv->file_num;
  PetscBool         use_parallel_viewer = PETSC_FALSE;
  int               z                   = 1; // Only support one zone
  int               B                   = 1; // Only support one base
  int               numComp;
  PetscInt          V_numComps, mystartv, myendv, myownedv;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)V, &comm));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-dm_plex_cgns_parallel", &use_parallel_viewer, NULL));
  PetscCheck(use_parallel_viewer, comm, PETSC_ERR_USER_INPUT, "Cannot use VecLoad with CGNS file in serial reader; use -dm_plex_cgns_parallel to enable parallel reader");

  { // Get CGNS node ownership information
    int         nbases, nzones;
    PetscInt    NVertices;
    PetscLayout vtx_map;
    cgsize_t    sizes[3]; /* Number of vertices, number of cells, number of boundary vertices */

    PetscCallCGNSRead(cg_nbases(cgid, &nbases), V, viewer);
    PetscCheck(nbases <= 1, PETSC_COMM_SELF, PETSC_ERR_LIB, "CGNS file must have a single base, not %d", nbases);
    PetscCallCGNSRead(cg_nzones(cgid, B, &nzones), V, viewer);
    PetscCheck(nzones == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "limited to one zone %d", (int)nzones);

    PetscCallCGNSRead(cg_zone_read(cgid, B, z, buffer, sizes), V, viewer);
    NVertices = sizes[0];

    PetscCall(PetscLayoutCreateFromSizes(comm, PETSC_DECIDE, NVertices, 1, &vtx_map));
    PetscCall(PetscLayoutGetRange(vtx_map, &mystartv, &myendv));
    PetscCall(PetscLayoutGetLocalSize(vtx_map, &myownedv));
    PetscCall(PetscLayoutDestroy(&vtx_map));
  }

  { // -- Read data from file into Vec
    PetscScalar *fields = NULL;
    PetscSF      sfNatural;

    { // Check compatibility between sfNatural and the data source and sink
      DM       dm;
      PetscInt nleaves, nroots, V_local_size;

      PetscCall(VecGetDM(V, &dm));
      PetscCall(DMGetNaturalSF(dm, &sfNatural));
      PetscCheck(sfNatural, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "DM of Vec must have sfNatural");
      PetscCall(PetscSFGetGraph(sfNatural, &nroots, &nleaves, NULL, NULL));
      PetscCall(VecGetLocalSize(V, &V_local_size));
      PetscCheck(nleaves == myownedv, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Number of locally owned vertices (% " PetscInt_FMT ") must match number of leaves in sfNatural (% " PetscInt_FMT ")", myownedv, nleaves);
      if (nroots == 0) {
        PetscCheck(V_local_size == nroots, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Local Vec size (% " PetscInt_FMT ") must be zero if number of roots in sfNatural is zero", V_local_size);
        V_numComps = 0;
      } else {
        PetscCheck(V_local_size % nroots == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Local Vec size (% " PetscInt_FMT ") not evenly divisible by number of roots in sfNatural (% " PetscInt_FMT ")", V_local_size, nroots);
        V_numComps = V_local_size / nroots;
      }
    }

    { // Read data into component-major ordering
      int isol, numSols;
      CGNS_ENUMT(DataType_t) datatype;
      double *fields_CGNS;

      PetscCallCGNSRead(cg_nsols(cgid, B, z, &numSols), V, viewer);
      PetscCall(PetscViewerCGNSGetSolutionFileIndex_Internal(viewer, &isol));
      PetscCallCGNSRead(cg_nfields(cgid, B, z, isol, &numComp), V, viewer);
      PetscCheck(V_numComps == numComp || V_numComps == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Vec sized for  % " PetscInt_FMT " components per node, but file has %d components per node", V_numComps, numComp);

      cgsize_t range_min[3] = {mystartv + 1, 1, 1};
      cgsize_t range_max[3] = {myendv, 1, 1};
      PetscCall(PetscMalloc1(myownedv * numComp, &fields_CGNS));
      PetscCall(PetscMalloc1(myownedv * numComp, &fields));
      for (int d = 0; d < numComp; ++d) {
        PetscCallCGNSRead(cg_field_info(cgid, B, z, isol, (d + 1), &datatype, buffer), V, viewer);
        PetscCheck(datatype == CGNS_ENUMV(RealDouble), PETSC_COMM_SELF, PETSC_ERR_ARG_NOTSAMETYPE, "Field %s in file is not of type double", buffer);
        PetscCallCGNSReadData(cgp_field_read_data(cgid, B, z, isol, (d + 1), range_min, range_max, &fields_CGNS[d * myownedv]), V, viewer);
      }
      for (int d = 0; d < numComp; ++d) {
        for (PetscInt v = 0; v < myownedv; ++v) fields[v * numComp + d] = fields_CGNS[d * myownedv + v];
      }
      PetscCall(PetscFree(fields_CGNS));
    }

    { // Reduce fields into Vec array
      PetscScalar *V_array;
      MPI_Datatype fieldtype;

      PetscCall(VecGetArrayWrite(V, &V_array));
      PetscCallMPI(MPI_Type_contiguous(numComp, MPIU_SCALAR, &fieldtype));
      PetscCallMPI(MPI_Type_commit(&fieldtype));
      PetscCall(PetscSFReduceBegin(sfNatural, fieldtype, fields, V_array, MPI_REPLACE));
      PetscCall(PetscSFReduceEnd(sfNatural, fieldtype, fields, V_array, MPI_REPLACE));
      PetscCallMPI(MPI_Type_free(&fieldtype));
      PetscCall(VecRestoreArrayWrite(V, &V_array));
    }
    PetscCall(PetscFree(fields));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
