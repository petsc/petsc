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
    case 3:
      element_type_tmp = CGNS_ENUMV(BAR_3);
      *perm            = bar_3;
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

  PetscFunctionBeginUser;
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

  PetscFunctionBeginUser;
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

PetscErrorCode DMPlexCreateCGNS_Internal_Parallel(MPI_Comm comm, PetscInt cgid, PetscBool interpolate, DM *dm)
{
  PetscMPIInt num_proc, rank;
  /* Read from file */
  char     basename[CGIO_MAX_NAME_LENGTH + 1];
  char     buffer[CGIO_MAX_NAME_LENGTH + 1];
  int      dim = 0, physDim = 0, coordDim = 0;
  PetscInt NVertices = 0, NCells = 0;
  int      nzones = 0, nbases;
  int      z      = 1; // Only supports single zone files
  int      B      = 1; // Only supports single base

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &num_proc));
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));

  PetscCallCGNSRead(cg_nbases(cgid, &nbases), *dm, 0);
  PetscCheck(nbases <= 1, PETSC_COMM_SELF, PETSC_ERR_LIB, "CGNS file must have a single base, not %d", nbases);
  //  From the CGNS web page                 cell_dim  phys_dim (embedding space in PETSc) CGNS defines as length of spatial vectors/components)
  PetscCallCGNSRead(cg_base_read(cgid, B, basename, &dim, &physDim), *dm, 0);
  PetscCallCGNSRead(cg_nzones(cgid, B, &nzones), *dm, 0);
  PetscCheck(nzones == 1, PETSC_COMM_SELF, PETSC_ERR_LIB, "Parallel reader limited to one zone, not %d", nzones);
  {
    cgsize_t sizes[3]; /* Number of vertices, number of cells, number of boundary vertices */

    PetscCallCGNSRead(cg_zone_read(cgid, B, z, buffer, sizes), *dm, 0);
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
  cgsize_t      *elements;
  {
    int        nsections;
    PetscInt  *elementsQ1, numCorners = -1;
    const int *perm;
    cgsize_t   start, end; // Throwaway

    cg_nsections(cgid, B, z, &nsections);
    // Read element connectivity
    for (int index_sect = 1; index_sect <= nsections; index_sect++) {
      int      nbndry, parentFlag;
      PetscInt cell_dim;
      CGNS_ENUMT(ElementType_t) cellType;

      PetscCallCGNSRead(cg_section_read(cgid, B, z, index_sect, buffer, &cellType, &start, &end, &nbndry, &parentFlag), *dm, 0);

      PetscCall(CGNSElementTypeGetTopologyInfo(cellType, &dm_cell_type, &numCorners, &cell_dim));
      // Skip over element that are not max dimension (ie. boundary elements)
      if (cell_dim != dim) continue;
      PetscCall(CGNSElementTypeGetDiscretizationInfo(cellType, &numClosure, &pOrder));
      PetscCall(PetscMalloc1(myownede * numClosure, &elements));
      PetscCallCGNSReadData(cgp_elements_read_data(cgid, B, z, index_sect, mystarte + 1, myende, elements), *dm, 0);
      for (PetscInt v = 0; v < myownede * numClosure; ++v) elements[v] -= 1; // 0 based
      break;
    }

    // Create corners-only connectivity
    PetscCall(PetscMalloc1(myownede * numCorners, &elementsQ1));
    PetscCall(DMPlexCGNSGetPermutation_Internal(dm_cell_type, numCorners, NULL, &perm));
    for (PetscInt e = 0; e < myownede; ++e) {
      for (PetscInt v = 0; v < numCorners; ++v) elementsQ1[e * numCorners + perm[v]] = elements[e * numClosure + v];
    }

    // Build cell-vertex Plex
    PetscCall(DMPlexBuildFromCellListParallel(*dm, myownede, myownedv, NVertices, numCorners, elementsQ1, NULL, NULL));
    PetscCall(DMViewFromOptions(*dm, NULL, "-corner_dm_view"));
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

      PetscCallCGNSRead(cg_coord_info(cgid, B, z, 1, &datatype, buffer), *dm, 0);
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

      PetscCallCGNSRead(cg_zone_read(cgid, B, z, buffer, sizes), *dm, 0);
      PetscCallCGNSRead(cg_ngrids(cgid, B, z, &ngrids), *dm, 0);
      PetscCheck(ngrids <= 1, PETSC_COMM_SELF, PETSC_ERR_LIB, "CGNS file must have a single grid, not %d", ngrids);
      PetscCallCGNSRead(cg_ncoords(cgid, B, z, &ncoords), *dm, 0);
      PetscCheck(ncoords == coordDim, PETSC_COMM_SELF, PETSC_ERR_LIB, "CGNS file must have a coordinate array for each dimension, not %d", ncoords);
      if (read_with_double) {
        for (int d = 0; d < coordDim; ++d) PetscCallCGNSReadData(cgp_coord_read_data(cgid, B, z, (d + 1), range_min, range_max, xd[d]), *dm, 0);
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
        for (int d = 0; d < coordDim; ++d) PetscCallCGNSReadData(cgp_coord_read_data(cgid, 1, z, (d + 1), range_min, range_max, x[d]), *dm, 0);
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
  PetscViewer_CGNS *cgv = (PetscViewer_CGNS *)viewer->data;
  PetscInt          fvGhostStart;
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
      PetscCall(PetscFECreateLagrange(PetscObjectComm((PetscObject)dm), topo_dim, coord_dim, is_simplex, field_order, quadrature_order, &fe));
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
  PetscCall(DMPlexGetCellTypeStratum(dm, DM_POLYTOPE_FV_GHOST, &fvGhostStart, NULL));
  if (fvGhostStart >= 0) cEnd = fvGhostStart;
  num_global_elems = cEnd - cStart;
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &num_global_elems, 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject)dm)));
  isize[0] = num_global_nodes;
  isize[1] = num_global_elems;
  isize[2] = 0;
  PetscCallCGNSWrite(cg_zone_write(cgv->file_num, base, "Zone", isize, CGNS_ENUMV(Unstructured), &zone), dm, viewer);

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
    cgsize_t   e_owned, e_global, e_start, *conn = NULL;
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

      local_element_type = e_owned > 0 ? element_type : -1;
      PetscCallMPI(MPIU_Allreduce(&local_element_type, &global_element_type, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)viewer)));
      if (local_element_type != -1) PetscCheck(local_element_type == global_element_type, PETSC_COMM_SELF, PETSC_ERR_SUP, "Ranks with different element types not supported");
      element_type = (CGNS_ENUMT(ElementType_t))global_element_type;
    }
    PetscCallMPI(MPIU_Allreduce(&e_owned, &e_global, 1, MPIU_CGSIZE, MPI_SUM, PetscObjectComm((PetscObject)dm)));
    PetscCheck(e_global == num_global_elems, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected number of elements %" PRIdCGSIZE " vs %" PetscInt_FMT, e_global, num_global_elems);
    e_start = 0;
    PetscCallMPI(MPI_Exscan(&e_owned, &e_start, 1, MPIU_CGSIZE, MPI_SUM, PetscObjectComm((PetscObject)dm)));
    PetscCallCGNSWrite(cgp_section_write(cgv->file_num, base, zone, "Elem", element_type, 1, e_global, 0, &section), dm, viewer);
    PetscCallCGNSWriteData(cgp_elements_write_data(cgv->file_num, base, zone, section, e_start + 1, e_start + e_owned, conn), dm, viewer);
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
      PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
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

  PetscFunctionBeginUser;
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
      PetscCheck(V_local_size % nroots == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Local Vec size (% " PetscInt_FMT ") not evenly divisible by number of roots in sfNatural (% " PetscInt_FMT ")", V_local_size, nroots);
      V_numComps = V_local_size / nroots;
    }

    { // Read data into component-major ordering
      int isol, numSols;
      CGNS_ENUMT(DataType_t) datatype;
      double *fields_CGNS;

      PetscCallCGNSRead(cg_nsols(cgid, B, z, &numSols), V, viewer);
      PetscCall(PetscViewerCGNSGetSolutionFileIndex_Internal(viewer, &isol));
      PetscCallCGNSRead(cg_nfields(cgid, B, z, isol, &numComp), V, viewer);
      PetscCheck(V_numComps == numComp, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Vec sized for  % " PetscInt_FMT " components per node, but file has %d components per node", V_numComps, numComp);

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
