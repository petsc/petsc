static char help[] = "Tests for parallel mesh loading and parallel topological interpolation\n\n";

#include <petsc/private/dmpleximpl.h>
/* List of test meshes

Network
-------
Test 0 (2 ranks):

network=0:
---------
  cell 0   cell 1   cell 2          nCells-1       (edge)
0 ------ 1 ------ 2 ------ 3 -- -- v --  -- nCells (vertex)

  vertex distribution:
    rank 0: 0 1
    rank 1: 2 3 ... nCells
  cell(edge) distribution:
    rank 0: 0 1
    rank 1: 2 ... nCells-1

network=1:
---------
               v2
                ^
                |
               cell 2
                |
 v0 --cell 0--> v3--cell 1--> v1

  vertex distribution:
    rank 0: 0 1 3
    rank 1: 2
  cell(edge) distribution:
    rank 0: 0 1
    rank 1: 2

  example:
    mpiexec -n 2 ./ex18 -distribute 1 -dim 1 -orig_dm_view -dist_dm_view -dist_dm_view -petscpartitioner_type parmetis -ncells 50

Triangle
--------
Test 0 (2 ranks):
Two triangles sharing a face

        2
      / | \
     /  |  \
    /   |   \
   0  0 | 1  3
    \   |   /
     \  |  /
      \ | /
        1

  vertex distribution:
    rank 0: 0 1
    rank 1: 2 3
  cell distribution:
    rank 0: 0
    rank 1: 1

Test 1 (3 ranks):
Four triangles partitioned across 3 ranks

   0 _______ 3
   | \     / |
   |  \ 1 /  |
   |   \ /   |
   | 0  2  2 |
   |   / \   |
   |  / 3 \  |
   | /     \ |
   1 ------- 4

  vertex distribution:
    rank 0: 0 1
    rank 1: 2 3
    rank 2: 4
  cell distribution:
    rank 0: 0
    rank 1: 1
    rank 2: 2 3

Test 2 (3 ranks):
Four triangles partitioned across 3 ranks

   1 _______ 3
   | \     / |
   |  \ 1 /  |
   |   \ /   |
   | 0  0  2 |
   |   / \   |
   |  / 3 \  |
   | /     \ |
   2 ------- 4

  vertex distribution:
    rank 0: 0 1
    rank 1: 2 3
    rank 2: 4
  cell distribution:
    rank 0: 0
    rank 1: 1
    rank 2: 2 3

Tetrahedron
-----------
Test 0:
Two tets sharing a face

 cell   3 _______    cell
 0    / | \      \   1
     /  |  \      \
    /   |   \      \
   0----|----4-----2
    \   |   /      /
     \  |  /      /
      \ | /      /
        1-------
   y
   | x
   |/
   *----z

  vertex distribution:
    rank 0: 0 1
    rank 1: 2 3 4
  cell distribution:
    rank 0: 0
    rank 1: 1

Quadrilateral
-------------
Test 0 (2 ranks):
Two quads sharing a face

   3-------2-------5
   |       |       |
   |   0   |   1   |
   |       |       |
   0-------1-------4

  vertex distribution:
    rank 0: 0 1 2
    rank 1: 3 4 5
  cell distribution:
    rank 0: 0
    rank 1: 1

TODO Test 1:
A quad and a triangle sharing a face

   5-------4
   |       | \
   |   0   |  \
   |       | 1 \
   2-------3----6

Hexahedron
----------
Test 0 (2 ranks):
Two hexes sharing a face

cell   7-------------6-------------11 cell
0     /|            /|            /|     1
     / |   F1      / |   F7      / |
    /  |          /  |          /  |
   4-------------5-------------10  |
   |   |     F4  |   |     F10 |   |
   |   |         |   |         |   |
   |F5 |         |F3 |         |F9 |
   |   |  F2     |   |   F8    |   |
   |   3---------|---2---------|---9
   |  /          |  /          |  /
   | /   F0      | /    F6     | /
   |/            |/            |/
   0-------------1-------------8

  vertex distribution:
    rank 0: 0 1 2 3 4 5
    rank 1: 6 7 8 9 10 11
  cell distribution:
    rank 0: 0
    rank 1: 1

*/

typedef enum {NONE, CREATE, AFTER_CREATE, AFTER_DISTRIBUTE} InterpType;

typedef struct {
  PetscInt   debug;                        /* The debugging level */
  PetscInt   testNum;                      /* Indicates the mesh to create */
  PetscInt   dim;                          /* The topological mesh dimension */
  PetscBool  cellSimplex;                  /* Use simplices or hexes */
  PetscBool  distribute;                   /* Distribute the mesh */
  InterpType interpolate;                  /* Interpolate the mesh before or after DMPlexDistribute() */
  PetscBool  useGenerator;                 /* Construct mesh with a mesh generator */
  PetscBool  testOrientIF;                 /* Test for different original interface orientations */
  PetscBool  testHeavy;                    /* Run the heavy PointSF test */
  PetscBool  customView;                   /* Show results of DMPlexIsInterpolated() etc. */
  PetscInt   ornt[2];                      /* Orientation of interface on rank 0 and rank 1 */
  PetscInt   faces[3];                     /* Number of faces per dimension for generator */
  PetscReal  coords[128];
  PetscReal  coordsTol;
  PetscInt   ncoords;
  PetscInt   pointsToExpand[128];
  PetscInt   nPointsToExpand;
  PetscBool  testExpandPointsEmpty;
  char       filename[PETSC_MAX_PATH_LEN]; /* Import mesh from file */
} AppCtx;

struct _n_PortableBoundary {
  Vec coordinates;
  PetscInt depth;
  PetscSection *sections;
};
typedef struct _n_PortableBoundary * PortableBoundary;

#if defined(PETSC_USE_LOG)
static PetscLogStage  stage[3];
#endif

static PetscErrorCode DMPlexCheckPointSFHeavy(DM, PortableBoundary);
static PetscErrorCode DMPlexSetOrientInterface_Private(DM,PetscBool);
static PetscErrorCode DMPlexGetExpandedBoundary_Private(DM, PortableBoundary *);
static PetscErrorCode DMPlexExpandedConesToFaces_Private(DM, IS, PetscSection, IS *);

static PetscErrorCode PortableBoundaryDestroy(PortableBoundary *bnd)
{
  PetscInt       d;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*bnd) PetscFunctionReturn(0);
  ierr = VecDestroy(&(*bnd)->coordinates);CHKERRQ(ierr);
  for (d=0; d < (*bnd)->depth; d++) {
    ierr = PetscSectionDestroy(&(*bnd)->sections[d]);CHKERRQ(ierr);
  }
  ierr = PetscFree((*bnd)->sections);CHKERRQ(ierr);
  ierr = PetscFree(*bnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char    *interpTypes[4]  = {"none", "create", "after_create", "after_distribute"};
  PetscInt       interp=NONE, dim;
  PetscBool      flg1, flg2;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->debug        = 0;
  options->testNum      = 0;
  options->dim          = 2;
  options->cellSimplex  = PETSC_TRUE;
  options->distribute   = PETSC_FALSE;
  options->interpolate  = NONE;
  options->useGenerator = PETSC_FALSE;
  options->testOrientIF = PETSC_FALSE;
  options->testHeavy    = PETSC_TRUE;
  options->customView   = PETSC_FALSE;
  options->testExpandPointsEmpty = PETSC_FALSE;
  options->ornt[0]      = 0;
  options->ornt[1]      = 0;
  options->faces[0]     = 2;
  options->faces[1]     = 2;
  options->faces[2]     = 2;
  options->filename[0]  = '\0';
  options->coordsTol    = PETSC_DEFAULT;

  ierr = PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-debug", "The debugging level", "ex18.c", options->debug, &options->debug, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-testnum", "The mesh to create", "ex18.c", options->testNum, &options->testNum, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cell_simplex", "Generate simplices if true, otherwise hexes", "ex18.c", options->cellSimplex, &options->cellSimplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-distribute", "Distribute the mesh", "ex18.c", options->distribute, &options->distribute, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-interpolate", "Type of mesh interpolation (none, create, after_create, after_distribute)", "ex18.c", interpTypes, 4, interpTypes[options->interpolate], &interp, NULL);CHKERRQ(ierr);
  options->interpolate = (InterpType) interp;
  if (!options->distribute && options->interpolate == AFTER_DISTRIBUTE) SETERRQ(comm, PETSC_ERR_SUP, "-interpolate after_distribute  needs  -distribute 1");
  ierr = PetscOptionsBool("-use_generator", "Use a mesh generator to build the mesh", "ex18.c", options->useGenerator, &options->useGenerator, NULL);CHKERRQ(ierr);
  options->ncoords = 128;
  ierr = PetscOptionsRealArray("-view_vertices_from_coords", "Print DAG points corresponding to vertices with given coordinates", "ex18.c", options->coords, &options->ncoords, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-view_vertices_from_coords_tol", "Tolerance for -view_vertices_from_coords", "ex18.c", options->coordsTol, &options->coordsTol, NULL);CHKERRQ(ierr);
  options->nPointsToExpand = 128;
  ierr = PetscOptionsIntArray("-test_expand_points", "Expand given array of DAG point using DMPlexGetConeRecursive() and print results", "ex18.c", options->pointsToExpand, &options->nPointsToExpand, NULL);CHKERRQ(ierr);
  if (options->nPointsToExpand) {
    ierr = PetscOptionsBool("-test_expand_points_empty", "For -test_expand_points, rank 0 will have empty input array", "ex18.c", options->testExpandPointsEmpty, &options->testExpandPointsEmpty, NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsBool("-test_heavy", "Run the heavy PointSF test", "ex18.c", options->testHeavy, &options->testHeavy, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-custom_view", "Custom DMPlex view", "ex18.c", options->customView, &options->customView, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim", "The topological mesh dimension", "ex18.c", options->dim, &options->dim, &flg1,1,3);CHKERRQ(ierr);
  dim = 3;
  ierr = PetscOptionsIntArray("-faces", "Number of faces per dimension", "ex18.c", options->faces, &dim, &flg2);CHKERRQ(ierr);
  if (flg2) {
    if (flg1 && dim != options->dim) SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE, "specified -dim %D is not equal to length %D of -faces (note that -dim can be omitted)", options->dim, dim);
    options->dim = dim;
  }
  ierr = PetscOptionsString("-filename", "The mesh file", "ex18.c", options->filename, options->filename, sizeof(options->filename), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-rotate_interface_0", "Rotation (relative orientation) of interface on rank 0; implies -interpolate create -distribute 0", "ex18.c", options->ornt[0], &options->ornt[0], &options->testOrientIF,0);CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-rotate_interface_1", "Rotation (relative orientation) of interface on rank 1; implies -interpolate create -distribute 0", "ex18.c", options->ornt[1], &options->ornt[1], &flg2,0);CHKERRQ(ierr);
  if (flg2 != options->testOrientIF) SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "neither or both -rotate_interface_0 -rotate_interface_1 must be set");
  if (options->testOrientIF) {
    PetscInt i;
    for (i=0; i<2; i++) {
      if (options->ornt[i] >= 10) options->ornt[i] = -(options->ornt[i]-10);  /* 11 12 13 become -1 -2 -3 */
    }
    options->filename[0]  = 0;
    options->useGenerator = PETSC_FALSE;
    options->dim          = 3;
    options->cellSimplex  = PETSC_TRUE;
    options->interpolate  = CREATE;
    options->distribute   = PETSC_FALSE;
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh_1D(MPI_Comm comm, PetscBool interpolate, AppCtx *user, DM *dm)
{
  PetscInt       testNum = user->testNum;
  PetscMPIInt    rank,size;
  PetscErrorCode ierr;
  PetscInt       numCorners=2,i;
  PetscInt       numCells,numVertices,network;
  PetscInt       *cells;
  PetscReal      *coords;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  if (size > 2) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for <=2 processes",testNum);

  numCells = 3;
  ierr = PetscOptionsGetInt(NULL, NULL, "-ncells", &numCells, NULL);CHKERRQ(ierr);
  if (numCells < 3) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test ncells must >=3",numCells);

  if (size == 1) {
    PetscReal *dcoords;
    numVertices = numCells + 1;
    ierr = PetscMalloc2(2*numCells,&cells,2*numVertices,&dcoords);CHKERRQ(ierr);
    for (i=0; i<numCells; i++) {
      cells[2*i] = i; cells[2*i+1] = i + 1;
      dcoords[2*i] = i; dcoords[2*i+1] = i + 1;
    }

    ierr = DMPlexCreateFromCellListPetsc(comm, user->dim, numCells, numVertices, numCorners, PETSC_FALSE, cells, user->dim, dcoords, dm);CHKERRQ(ierr);
    ierr = PetscFree2(cells,coords);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  network = 0;
  ierr = PetscOptionsGetInt(NULL, NULL, "-network_case", &network, NULL);CHKERRQ(ierr);
  if (network == 0) {
    switch (rank) {
    case 0:
    {
      numCells    = 2;
      numVertices = numCells;
      ierr = PetscMalloc2(2*numCells,&cells,2*numCells,&coords);CHKERRQ(ierr);
      cells[0] = 0; cells[1] = 1;
      cells[2] = 1; cells[3] = 2;
      coords[0] = 0.; coords[1] = 1.;
      coords[2] = 1.; coords[3] = 2.;
    }
    break;
    case 1:
    {
      numCells    -= 2;
      numVertices = numCells + 1;
      ierr = PetscMalloc2(2*numCells,&cells,2*numCells,&coords);CHKERRQ(ierr);
      for (i=0; i<numCells; i++) {
        cells[2*i] = 2+i; cells[2*i+1] = 2 + i + 1;
        coords[2*i] = 2+i; coords[2*i+1] = 2 + i + 1;
      }
    }
    break;
    default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
  } else { /* network_case = 1 */
    /* ----------------------- */
    switch (rank) {
    case 0:
    {
      numCells    = 2;
      numVertices = 3;
      ierr = PetscMalloc2(2*numCells,&cells,2*numCells,&coords);CHKERRQ(ierr);
      cells[0] = 0; cells[1] = 3;
      cells[2] = 3; cells[3] = 1;
    }
    break;
    case 1:
    {
      numCells    = 1;
      numVertices = 1;
      ierr = PetscMalloc2(2*numCells,&cells,2*numCells,&coords);CHKERRQ(ierr);
      cells[0] = 3; cells[1] = 2;
    }
    break;
    default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
  }
  ierr = DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, PETSC_FALSE, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
  ierr = PetscFree2(cells,coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateSimplex_2D(MPI_Comm comm, PetscBool interpolate, AppCtx *user, DM *dm)
{
  PetscInt       testNum = user->testNum, p;
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  switch (testNum) {
  case 0:
    if (size != 2) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 2 processes", testNum);
    switch (rank) {
      case 0:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const PetscInt cells[3]  = {0, 1, 2};
        PetscReal      coords[4] = {-0.5, 0.5, 0.0, 0.0};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        ierr = DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const PetscInt cells[3]  = {1, 3, 2};
        PetscReal      coords[4] = {0.0, 1.0, 0.5, 0.5};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        ierr = DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
    break;
  case 1:
    if (size != 3) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 3 processes", testNum);
    switch (rank) {
      case 0:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const PetscInt cells[3]  = {0, 1, 2};
        PetscReal      coords[4] = {0.0, 1.0, 0.0, 0.0};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        ierr = DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const PetscInt cells[3]  = {0, 2, 3};
        PetscReal      coords[4] = {0.5, 0.5, 1.0, 1.0};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        ierr = DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 2:
      {
        const PetscInt numCells  = 2, numVertices = 1, numCorners = 3;
        const PetscInt cells[6]  = {2, 4, 3, 2, 1, 4};
        PetscReal      coords[2] = {1.0, 0.0};
        PetscInt       markerPoints[10] = {2, 1, 3, 1, 4, 1, 5, 1, 6, 1};

        ierr = DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
    break;
  case 2:
    if (size != 3) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 3 processes", testNum);
    switch (rank) {
      case 0:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const PetscInt cells[3]  = {1, 2, 0};
        PetscReal      coords[4] = {0.5, 0.5, 0.0, 1.0};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        ierr = DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const PetscInt cells[3]  = {1, 0, 3};
        PetscReal      coords[4] = {0.0, 0.0, 1.0, 1.0};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        ierr = DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 2:
      {
        const PetscInt numCells  = 2, numVertices = 1, numCorners = 3;
        const PetscInt cells[6]  = {0, 4, 3, 0, 2, 4};
        PetscReal      coords[2] = {1.0, 0.0};
        PetscInt       markerPoints[10] = {2, 1, 3, 1, 4, 1, 5, 1, 6, 1};

        ierr = DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
    break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %D", testNum);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateSimplex_3D(MPI_Comm comm, PetscBool interpolate, AppCtx *user, DM *dm)
{
  PetscInt       testNum = user->testNum, p;
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  switch (testNum) {
  case 0:
    if (size != 2) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 2 processes", testNum);
    switch (rank) {
      case 0:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 4;
        const PetscInt cells[4]  = {0, 2, 1, 3};
        PetscReal      coords[6] = {0.0, 0.0, -0.5,  0.0, -0.5, 0.0};
        PetscInt       markerPoints[8] = {1, 1, 2, 1, 3, 1, 4, 1};

        ierr = DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 3, numCorners = 4;
        const PetscInt cells[4]  = {1, 2, 4, 3};
        PetscReal      coords[9] = {1.0, 0.0, 0.0,  0.0, 0.5, 0.0,  0.0, 0.0, 0.5};
        PetscInt       markerPoints[8] = {1, 1, 2, 1, 3, 1, 4, 1};

        ierr = DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
    break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %D", testNum);
  }
  if (user->testOrientIF) {
    PetscInt ifp[] = {8, 6};

    ierr = PetscObjectSetName((PetscObject) *dm, "Mesh before orientation");CHKERRQ(ierr);
    ierr = DMViewFromOptions(*dm, NULL, "-before_orientation_dm_view");CHKERRQ(ierr);
    /* rotate interface face ifp[rank] by given orientation ornt[rank] */
    ierr = DMPlexOrientPoint(*dm, ifp[rank], user->ornt[rank]);CHKERRQ(ierr);
    ierr = DMViewFromOptions(*dm, NULL, "-before_orientation_dm_view");CHKERRQ(ierr);
    ierr = DMPlexCheckFaces(*dm, 0);CHKERRQ(ierr);
    ierr = DMPlexOrientInterface_Internal(*dm);CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "Orientation test PASSED\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateQuad_2D(MPI_Comm comm, PetscBool interpolate, AppCtx *user, DM *dm)
{
  PetscInt       testNum = user->testNum, p;
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  switch (testNum) {
  case 0:
    if (size != 2) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 2 processes", testNum);
    switch (rank) {
      case 0:
      {
        const PetscInt numCells  = 1, numVertices = 3, numCorners = 4;
        const PetscInt cells[4]  = {0, 1, 2, 3};
        PetscReal      coords[6] = {-0.5, 0.0, 0.0, 0.0, 0.0, 1.0};
        PetscInt       markerPoints[4*2] = {1, 1, 2, 1, 3, 1, 4, 1};

        ierr = DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 3, numCorners = 4;
        const PetscInt cells[4]  = {1, 4, 5, 2};
        PetscReal      coords[6] = {-0.5, 1.0, 0.5, 0.0, 0.5, 1.0};
        PetscInt       markerPoints[4*2] = {1, 1, 2, 1, 3, 1, 4, 1};

        ierr = DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
    break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %D", testNum);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateHex_3D(MPI_Comm comm, PetscBool interpolate, AppCtx *user, DM *dm)
{
  PetscInt       testNum = user->testNum, p;
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  switch (testNum) {
  case 0:
    if (size != 2) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 2 processes", testNum);
    switch (rank) {
    case 0:
    {
      const PetscInt numCells  = 1, numVertices = 6, numCorners = 8;
      const PetscInt cells[8]  = {0, 3, 2, 1, 4, 5, 6, 7};
      PetscReal      coords[6*3] = {-0.5,0.0,0.0, 0.0,0.0,0.0, 0.0,1.0,0.0, -0.5,1.0,0.0, -0.5,0.0,1.0, 0.0,0.0,1.0};
      PetscInt       markerPoints[8*2] = {2,1,3,1,4,1,5,1,6,1,7,1,8,1,9,1};

      ierr = DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
      for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
    }
    break;
    case 1:
    {
      const PetscInt numCells  = 1, numVertices = 6, numCorners = 8;
      const PetscInt cells[8]  = {1, 2, 9, 8, 5, 10, 11, 6};
      PetscReal      coords[6*3] = {0.0,1.0,1.0, -0.5,1.0,1.0, 0.5,0.0,0.0, 0.5,1.0,0.0, 0.5,0.0,1.0,  0.5,1.0,1.0};
      PetscInt       markerPoints[8*2] = {2,1,3,1,4,1,5,1,6,1,7,1,8,1,9,1};

      ierr = DMPlexCreateFromCellListParallelPetsc(comm, user->dim, numCells, numVertices, PETSC_DECIDE, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
      for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
    }
    break;
    default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
  break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %D", testNum);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CustomView(DM dm, PetscViewer v)
{
  DMPlexInterpolatedFlag interpolated;
  PetscBool              distributed;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = DMPlexIsDistributed(dm, &distributed);CHKERRQ(ierr);
  ierr = DMPlexIsInterpolatedCollective(dm, &interpolated);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(v, "DMPlexIsDistributed: %s\n", PetscBools[distributed]);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(v, "DMPlexIsInterpolatedCollective: %s\n", DMPlexInterpolatedFlags[interpolated]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMeshFromFile(MPI_Comm comm, AppCtx *user, DM *dm, DM *serialDM)
{
  const char    *filename       = user->filename;
  PetscBool      testHeavy      = user->testHeavy;
  PetscBool      interpCreate   = user->interpolate == CREATE ? PETSC_TRUE : PETSC_FALSE;
  PetscBool      distributed    = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *serialDM = NULL;
  if (testHeavy && interpCreate) {ierr = DMPlexSetOrientInterface_Private(NULL, PETSC_FALSE);CHKERRQ(ierr);}
  ierr = PetscLogStagePush(stage[0]);CHKERRQ(ierr);
  ierr = DMPlexCreateFromFile(comm, filename, "ex18_plex", interpCreate, dm);CHKERRQ(ierr); /* with DMPlexOrientInterface_Internal() call skipped so that PointSF issues are left to DMPlexCheckPointSFHeavy() */
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  if (testHeavy && interpCreate) {ierr = DMPlexSetOrientInterface_Private(NULL, PETSC_TRUE);CHKERRQ(ierr);}
  ierr = DMPlexIsDistributed(*dm, &distributed);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "DMPlexCreateFromFile produced %s mesh.\n", distributed ? "distributed" : "serial");CHKERRQ(ierr);
  if (testHeavy && distributed) {
    ierr = PetscOptionsSetValue(NULL, "-dm_plex_hdf5_force_sequential", NULL);CHKERRQ(ierr);
    ierr = DMPlexCreateFromFile(comm, filename, "ex18_plex", interpCreate, serialDM);CHKERRQ(ierr);
    ierr = DMPlexIsDistributed(*serialDM, &distributed);CHKERRQ(ierr);
    if (distributed) SETERRQ(comm, PETSC_ERR_PLIB, "unable to create a serial DM from file");
  }
  ierr = DMGetDimension(*dm, &user->dim);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscPartitioner part;
  PortableBoundary boundary     = NULL;
  DM             serialDM       = NULL;
  PetscBool      cellSimplex    = user->cellSimplex;
  PetscBool      useGenerator   = user->useGenerator;
  PetscBool      interpCreate   = user->interpolate == CREATE ? PETSC_TRUE : PETSC_FALSE;
  PetscBool      interpSerial   = user->interpolate == AFTER_CREATE ? PETSC_TRUE : PETSC_FALSE;
  PetscBool      interpParallel = user->interpolate == AFTER_DISTRIBUTE ? PETSC_TRUE : PETSC_FALSE;
  PetscBool      testHeavy      = user->testHeavy;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  if (user->filename[0]) {
    ierr = CreateMeshFromFile(comm, user, dm, &serialDM);CHKERRQ(ierr);
  } else if (useGenerator) {
    ierr = PetscLogStagePush(stage[0]);CHKERRQ(ierr);
    ierr = DMPlexCreateBoxMesh(comm, user->dim, cellSimplex, user->faces, NULL, NULL, NULL, interpCreate, dm);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
  } else {
    ierr = PetscLogStagePush(stage[0]);CHKERRQ(ierr);
    switch (user->dim) {
    case 1:
      ierr = CreateMesh_1D(comm, interpCreate, user, dm);CHKERRQ(ierr);
      break;
    case 2:
      if (cellSimplex) {
        ierr = CreateSimplex_2D(comm, interpCreate, user, dm);CHKERRQ(ierr);
      } else {
        ierr = CreateQuad_2D(comm, interpCreate, user, dm);CHKERRQ(ierr);
      }
      break;
    case 3:
      if (cellSimplex) {
        ierr = CreateSimplex_3D(comm, interpCreate, user, dm);CHKERRQ(ierr);
      } else {
        ierr = CreateHex_3D(comm, interpCreate, user, dm);CHKERRQ(ierr);
      }
      break;
    default:
      SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Cannot make meshes for dimension %D", user->dim);
    }
    ierr = PetscLogStagePop();CHKERRQ(ierr);
  }
  if (user->ncoords % user->dim) SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE, "length of coordinates array %D must be divisable by spatial dimension %D", user->ncoords, user->dim);
  ierr = PetscObjectSetName((PetscObject) *dm, "Original Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-orig_dm_view");CHKERRQ(ierr);

  if (interpSerial) {
    DM idm;

    if (testHeavy) {ierr = DMPlexSetOrientInterface_Private(*dm, PETSC_FALSE);CHKERRQ(ierr);}
    ierr = PetscLogStagePush(stage[2]);CHKERRQ(ierr);
    ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr); /* with DMPlexOrientInterface_Internal() call skipped so that PointSF issues are left to DMPlexCheckPointSFHeavy() */
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    if (testHeavy) {ierr = DMPlexSetOrientInterface_Private(*dm, PETSC_TRUE);CHKERRQ(ierr);}
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm = idm;
    ierr = PetscObjectSetName((PetscObject) *dm, "Interpolated Mesh");CHKERRQ(ierr);
    ierr = DMViewFromOptions(*dm, NULL, "-intp_dm_view");CHKERRQ(ierr);
  }

  /* Set partitioner options */
  ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
  if (part) {
    ierr = PetscPartitionerSetType(part, PETSCPARTITIONERSIMPLE);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
  }

  if (user->customView) {ierr = CustomView(*dm, PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);}
  if (testHeavy) {
    PetscBool distributed;

    ierr = DMPlexIsDistributed(*dm, &distributed);CHKERRQ(ierr);
    if (!serialDM && !distributed) {
      serialDM = *dm;
      ierr = PetscObjectReference((PetscObject)*dm);CHKERRQ(ierr);
    }
    if (serialDM) {
      ierr = DMPlexGetExpandedBoundary_Private(serialDM, &boundary);CHKERRQ(ierr);
    }
    if (boundary) {
      /* check DM which has been created in parallel and already interpolated */
      ierr = DMPlexCheckPointSFHeavy(*dm, boundary);CHKERRQ(ierr);
    }
    /* Orient interface because it could be deliberately skipped above. It is idempotent. */
    ierr = DMPlexOrientInterface_Internal(*dm);CHKERRQ(ierr);
  }
  if (user->distribute) {
    DM               pdm = NULL;

    /* Redistribute mesh over processes using that partitioner */
    ierr = PetscLogStagePush(stage[1]);CHKERRQ(ierr);
    ierr = DMPlexDistribute(*dm, 0, NULL, &pdm);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
    if (pdm) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = pdm;
      ierr = PetscObjectSetName((PetscObject) *dm, "Redistributed Mesh");CHKERRQ(ierr);
      ierr = DMViewFromOptions(*dm, NULL, "-dist_dm_view");CHKERRQ(ierr);
    }

    if (interpParallel) {
      DM idm;

      if (testHeavy) {ierr = DMPlexSetOrientInterface_Private(*dm, PETSC_FALSE);CHKERRQ(ierr);}
      ierr = PetscLogStagePush(stage[2]);CHKERRQ(ierr);
      ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr); /* with DMPlexOrientInterface_Internal() call skipped so that PointSF issues are left to DMPlexCheckPointSFHeavy() */
      ierr = PetscLogStagePop();CHKERRQ(ierr);
      if (testHeavy) {ierr = DMPlexSetOrientInterface_Private(*dm, PETSC_TRUE);CHKERRQ(ierr);}
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm = idm;
      ierr = PetscObjectSetName((PetscObject) *dm, "Interpolated Redistributed Mesh");CHKERRQ(ierr);
      ierr = DMViewFromOptions(*dm, NULL, "-intp_dm_view");CHKERRQ(ierr);
    }
  }
  if (testHeavy) {
    if (boundary) {
      ierr = DMPlexCheckPointSFHeavy(*dm, boundary);CHKERRQ(ierr);
    }
    /* Orient interface because it could be deliberately skipped above. It is idempotent. */
    ierr = DMPlexOrientInterface_Internal(*dm);CHKERRQ(ierr);
  }

  ierr = PetscObjectSetName((PetscObject) *dm, "Parallel Mesh");CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);

  if (user->customView) {ierr = CustomView(*dm, PETSC_VIEWER_STDOUT_(comm));CHKERRQ(ierr);}
  ierr = DMDestroy(&serialDM);CHKERRQ(ierr);
  ierr = PortableBoundaryDestroy(&boundary);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode coord2str(char buf[], size_t len, PetscInt dim, PetscReal *coords, PetscReal tol)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (dim > 3) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "dim must be less than or equal 3");
  if (tol >= 1e-3) {
    switch (dim) {
      case 1: ierr = PetscSNPrintf(buf,len,"(%12.3f)",(double)coords[0]);CHKERRQ(ierr);
      case 2: ierr = PetscSNPrintf(buf,len,"(%12.3f, %12.3f)",(double)coords[0],(double)coords[1]);CHKERRQ(ierr);
      case 3: ierr = PetscSNPrintf(buf,len,"(%12.3f, %12.3f, %12.3f)",(double)coords[0],(double)coords[1],(double)coords[2]);CHKERRQ(ierr);
    }
  } else {
    switch (dim) {
      case 1: ierr = PetscSNPrintf(buf,len,"(%12.6f)",(double)coords[0]);CHKERRQ(ierr);
      case 2: ierr = PetscSNPrintf(buf,len,"(%12.6f, %12.6f)",(double)coords[0],(double)coords[1]);CHKERRQ(ierr);
      case 3: ierr = PetscSNPrintf(buf,len,"(%12.6f, %12.6f, %12.6f)",(double)coords[0],(double)coords[1],(double)coords[2]);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode ViewVerticesFromCoords(DM dm, PetscInt npoints, PetscReal coords[], PetscReal tol, PetscViewer viewer)
{
  PetscInt       dim, i;
  PetscInt       *points;
  char           coordstr[128];
  MPI_Comm       comm;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushSynchronized(viewer);CHKERRQ(ierr);
  ierr = PetscMalloc1(npoints, &points);CHKERRQ(ierr);
  ierr = DMPlexFindVertices(dm, npoints, coords, tol, points);CHKERRQ(ierr);
  for (i=0; i < npoints; i++) {
    ierr = coord2str(coordstr, sizeof(coordstr), dim, &coords[i*dim], tol);CHKERRQ(ierr);
    if (rank == 0 && i) {ierr = PetscViewerASCIISynchronizedPrintf(viewer, "-----\n");CHKERRQ(ierr);}
    ierr = PetscViewerASCIISynchronizedPrintf(viewer, "[%d] %s --> points[%D] = %D\n", rank, coordstr, i, points[i]);CHKERRQ(ierr);
    ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPopSynchronized(viewer);CHKERRQ(ierr);
  ierr = PetscFree(points);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestExpandPoints(DM dm, AppCtx *user)
{
  IS                is;
  PetscSection      *sects;
  IS                *iss;
  PetscInt          d,depth;
  PetscMPIInt       rank;
  PetscErrorCode    ierr;
  PetscViewer       viewer=PETSC_VIEWER_STDOUT_WORLD, sviewer;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm),&rank);CHKERRMPI(ierr);
  if (user->testExpandPointsEmpty && rank == 0) {
    ierr = ISCreateGeneral(PETSC_COMM_SELF, 0, NULL, PETSC_USE_POINTER, &is);CHKERRQ(ierr);
  } else {
    ierr = ISCreateGeneral(PETSC_COMM_SELF, user->nPointsToExpand, user->pointsToExpand, PETSC_USE_POINTER, &is);CHKERRQ(ierr);
  }
  ierr = DMPlexGetConeRecursive(dm, is, &depth, &iss, &sects);CHKERRQ(ierr);
  ierr = PetscViewerGetSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(sviewer, "[%d] ==========================\n",rank);CHKERRQ(ierr);
  for (d=depth-1; d>=0; d--) {
    IS          checkIS;
    PetscBool   flg;

    ierr = PetscViewerASCIIPrintf(sviewer, "depth %D ---------------\n",d);CHKERRQ(ierr);
    ierr = PetscSectionView(sects[d], sviewer);CHKERRQ(ierr);
    ierr = ISView(iss[d], sviewer);CHKERRQ(ierr);
    /* check reverse operation */
    if (d < depth-1) {
      ierr = DMPlexExpandedConesToFaces_Private(dm, iss[d], sects[d], &checkIS);CHKERRQ(ierr);
      ierr = ISEqualUnsorted(checkIS, iss[d+1], &flg);CHKERRQ(ierr);
      if (!flg) SETERRQ(PetscObjectComm((PetscObject) checkIS), PETSC_ERR_PLIB, "DMPlexExpandedConesToFaces_Private produced wrong IS");
      ierr = ISDestroy(&checkIS);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerRestoreSubViewer(viewer,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
  ierr = PetscViewerFlush(viewer);CHKERRQ(ierr);
  ierr = DMPlexRestoreConeRecursive(dm, is, &depth, &iss, &sects);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexExpandedConesToFaces_Private(DM dm, IS is, PetscSection section, IS *newis)
{
  PetscInt          n,n1,ncone,numCoveredPoints,o,p,q,start,end;
  const PetscInt    *coveredPoints;
  const PetscInt    *arr, *cone;
  PetscInt          *newarr;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = ISGetLocalSize(is, &n);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(section, &n1);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(section, &start, &end);CHKERRQ(ierr);
  if (n != n1) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_PLIB, "IS size = %D != %D = section storage size\n", n, n1);
  ierr = ISGetIndices(is, &arr);CHKERRQ(ierr);
  ierr = PetscMalloc1(end-start, &newarr);CHKERRQ(ierr);
  for (q=start; q<end; q++) {
    ierr = PetscSectionGetDof(section, q, &ncone);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(section, q, &o);CHKERRQ(ierr);
    cone = &arr[o];
    if (ncone == 1) {
      numCoveredPoints = 1;
      p = cone[0];
    } else {
      PetscInt i;
      p = PETSC_MAX_INT;
      for (i=0; i<ncone; i++) if (cone[i] < 0) {p = -1; break;}
      if (p >= 0) {
        ierr = DMPlexGetJoin(dm, ncone, cone, &numCoveredPoints, &coveredPoints);CHKERRQ(ierr);
        if (numCoveredPoints > 1) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "more than one covered points for section point %D",q);
        if (numCoveredPoints) p = coveredPoints[0];
        else                  p = -2;
        ierr = DMPlexRestoreJoin(dm, ncone, cone, &numCoveredPoints, &coveredPoints);CHKERRQ(ierr);
      }
    }
    newarr[q-start] = p;
  }
  ierr = ISRestoreIndices(is, &arr);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, end-start, newarr, PETSC_OWN_POINTER, newis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexExpandedVerticesToFaces_Private(DM dm, IS boundary_expanded_is, PetscInt depth, PetscSection sections[], IS *boundary_is)
{
  PetscInt          d;
  IS                is,newis;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  is = boundary_expanded_is;
  ierr = PetscObjectReference((PetscObject)is);CHKERRQ(ierr);
  for (d = 0; d < depth-1; ++d) {
    ierr = DMPlexExpandedConesToFaces_Private(dm, is, sections[d], &newis);CHKERRQ(ierr);
    ierr = ISDestroy(&is);CHKERRQ(ierr);
    is = newis;
  }
  *boundary_is = is;
  PetscFunctionReturn(0);
}

#define CHKERRQI(incall,ierr) if (ierr) {incall = PETSC_FALSE; }

static PetscErrorCode DMLabelViewFromOptionsOnComm_Private(DMLabel label, const char optionname[], MPI_Comm comm)
{
  PetscErrorCode    ierr;
  PetscViewer       viewer;
  PetscBool         flg;
  static PetscBool  incall = PETSC_FALSE;
  PetscViewerFormat format;

  PetscFunctionBegin;
  if (incall) PetscFunctionReturn(0);
  incall = PETSC_TRUE;
  ierr   = PetscOptionsGetViewer(comm,((PetscObject)label)->options,((PetscObject)label)->prefix,optionname,&viewer,&format,&flg);CHKERRQI(incall,ierr);
  if (flg) {
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQI(incall,ierr);
    ierr = DMLabelView(label, viewer);CHKERRQI(incall,ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQI(incall,ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQI(incall,ierr);
  }
  incall = PETSC_FALSE;
  PetscFunctionReturn(0);
}

/* TODO: this is hotfixing DMLabelGetStratumIS() - it should be fixed systematically instead */
PETSC_STATIC_INLINE PetscErrorCode DMLabelGetStratumISOnComm_Private(DMLabel label, PetscInt value, MPI_Comm comm, IS *is)
{
  IS                tmpis;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = DMLabelGetStratumIS(label, value, &tmpis);CHKERRQ(ierr);
  if (!tmpis) {ierr = ISCreateGeneral(PETSC_COMM_SELF, 0, NULL, PETSC_USE_POINTER, &tmpis);CHKERRQ(ierr);}
  ierr = ISOnComm(tmpis, comm, PETSC_COPY_VALUES, is);CHKERRQ(ierr);
  ierr = ISDestroy(&tmpis);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* currently only for simple PetscSection without fields or constraints */
static PetscErrorCode PetscSectionReplicate_Private(MPI_Comm comm, PetscMPIInt rootrank, PetscSection sec0, PetscSection *secout)
{
  PetscSection      sec;
  PetscInt          chart[2], p;
  PetscInt          *dofarr;
  PetscMPIInt       rank;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  if (rank == rootrank) {
    ierr = PetscSectionGetChart(sec0, &chart[0], &chart[1]);CHKERRQ(ierr);
  }
  ierr = MPI_Bcast(chart, 2, MPIU_INT, rootrank, comm);CHKERRMPI(ierr);
  ierr = PetscMalloc1(chart[1]-chart[0], &dofarr);CHKERRQ(ierr);
  if (rank == rootrank) {
    for (p = chart[0]; p < chart[1]; p++) {
      ierr = PetscSectionGetDof(sec0, p, &dofarr[p-chart[0]]);CHKERRQ(ierr);
    }
  }
  ierr = MPI_Bcast(dofarr, chart[1]-chart[0], MPIU_INT, rootrank, comm);CHKERRMPI(ierr);
  ierr = PetscSectionCreate(comm, &sec);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(sec, chart[0], chart[1]);CHKERRQ(ierr);
  for (p = chart[0]; p < chart[1]; p++) {
    ierr = PetscSectionSetDof(sec, p, dofarr[p-chart[0]]);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(sec);CHKERRQ(ierr);
  ierr = PetscFree(dofarr);CHKERRQ(ierr);
  *secout = sec;
  PetscFunctionReturn(0);
}

static PetscErrorCode VecToPetscReal_Private(Vec vec, PetscReal *rvals[])
{
  PetscInt          n;
  const PetscScalar *svals;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(vec, &n);CHKERRQ(ierr);
  ierr = VecGetArrayRead(vec, &svals);CHKERRQ(ierr);
  ierr = PetscMalloc1(n, rvals);CHKERRQ(ierr);
#if defined(PETSC_USE_COMPLEX)
  {
    PetscInt i;
    for (i=0; i<n; i++) (*rvals)[i] = PetscRealPart(svals[i]);
  }
#else
  ierr = PetscMemcpy(*rvals, svals, n*sizeof(PetscReal));CHKERRQ(ierr);
#endif
  ierr = VecRestoreArrayRead(vec, &svals);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexExpandedVerticesCoordinatesToFaces_Private(DM ipdm, PortableBoundary bnd, IS *face_is)
{
  PetscInt            dim, ncoords, npoints;
  PetscReal           *rcoords;
  PetscInt            *points;
  IS                  faces_expanded_is;
  PetscErrorCode      ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(ipdm, &dim);CHKERRQ(ierr);
  ierr = VecGetLocalSize(bnd->coordinates, &ncoords);CHKERRQ(ierr);
  ierr = VecToPetscReal_Private(bnd->coordinates, &rcoords);CHKERRQ(ierr);
  npoints = ncoords / dim;
  ierr = PetscMalloc1(npoints, &points);CHKERRQ(ierr);
  ierr = DMPlexFindVertices(ipdm, npoints, rcoords, 0.0, points);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PETSC_COMM_SELF, npoints, points, PETSC_OWN_POINTER, &faces_expanded_is);CHKERRQ(ierr);
  ierr = DMPlexExpandedVerticesToFaces_Private(ipdm, faces_expanded_is, bnd->depth, bnd->sections, face_is);CHKERRQ(ierr);
  ierr = PetscFree(rcoords);CHKERRQ(ierr);
  ierr = ISDestroy(&faces_expanded_is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* hack disabling DMPlexOrientInterface() call in DMPlexInterpolate() via -dm_plex_interpolate_orient_interfaces option */
static PetscErrorCode DMPlexSetOrientInterface_Private(DM dm, PetscBool enable)
{
  PetscOptions      options = NULL;
  const char        *prefix = NULL;
  const char        opt[] = "-dm_plex_interpolate_orient_interfaces";
  char              prefix_opt[512];
  PetscBool         flg, set;
  static PetscBool  wasSetTrue = PETSC_FALSE;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (dm) {
    ierr = PetscObjectGetOptionsPrefix((PetscObject)dm, &prefix);CHKERRQ(ierr);
    options = ((PetscObject)dm)->options;
  }
  ierr = PetscStrcpy(prefix_opt, "-");CHKERRQ(ierr);
  ierr = PetscStrlcat(prefix_opt, prefix, sizeof(prefix_opt));CHKERRQ(ierr);
  ierr = PetscStrlcat(prefix_opt, &opt[1], sizeof(prefix_opt));CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(options, prefix, opt, &flg, &set);CHKERRQ(ierr);
  if (!enable) {
    if (set && flg) wasSetTrue = PETSC_TRUE;
    ierr = PetscOptionsSetValue(options, prefix_opt, "0");CHKERRQ(ierr);
  } else if (set && !flg) {
    if (wasSetTrue) {
      ierr = PetscOptionsSetValue(options, prefix_opt, "1");CHKERRQ(ierr);
    } else {
      /* default is PETSC_TRUE */
      ierr = PetscOptionsClearValue(options, prefix_opt);CHKERRQ(ierr);
    }
    wasSetTrue = PETSC_FALSE;
  }
  if (PetscDefined(USE_DEBUG)) {
    ierr = PetscOptionsGetBool(options, prefix, opt, &flg, &set);CHKERRQ(ierr);
    if (PetscUnlikely(set && flg != enable)) SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_PLIB, "PetscOptionsSetValue did not have the desired effect");
  }
  PetscFunctionReturn(0);
}

/* get coordinate description of the whole-domain boundary */
static PetscErrorCode DMPlexGetExpandedBoundary_Private(DM dm, PortableBoundary *boundary)
{
  PortableBoundary       bnd0, bnd;
  MPI_Comm               comm;
  DM                     idm;
  DMLabel                label;
  PetscInt               d;
  const char             boundaryName[] = "DMPlexDistributeInterpolateMarkInterface_boundary";
  IS                     boundary_is;
  IS                     *boundary_expanded_iss;
  PetscMPIInt            rootrank = 0;
  PetscMPIInt            rank, size;
  PetscInt               value = 1;
  DMPlexInterpolatedFlag intp;
  PetscBool              flg;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&bnd);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr = DMPlexIsDistributed(dm, &flg);CHKERRQ(ierr);
  if (flg) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "serial DM (all points on one rank) needed");

  /* interpolate serial DM if not yet interpolated */
  ierr = DMPlexIsInterpolatedCollective(dm, &intp);CHKERRQ(ierr);
  if (intp == DMPLEX_INTERPOLATED_FULL) {
    idm = dm;
    ierr = PetscObjectReference((PetscObject)dm);CHKERRQ(ierr);
  } else {
    ierr = DMPlexInterpolate(dm, &idm);CHKERRQ(ierr);
    ierr = DMViewFromOptions(idm, NULL, "-idm_view");CHKERRQ(ierr);
  }

  /* mark whole-domain boundary of the serial DM */
  ierr = DMLabelCreate(PETSC_COMM_SELF, boundaryName, &label);CHKERRQ(ierr);
  ierr = DMAddLabel(idm, label);CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(idm, value, label);CHKERRQ(ierr);
  ierr = DMLabelViewFromOptionsOnComm_Private(label, "-idm_boundary_view", comm);CHKERRQ(ierr);
  ierr = DMLabelGetStratumIS(label, value, &boundary_is);CHKERRQ(ierr);

  /* translate to coordinates */
  ierr = PetscNew(&bnd0);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocalSetUp(idm);CHKERRQ(ierr);
  if (rank == rootrank) {
    ierr = DMPlexGetConeRecursive(idm, boundary_is, &bnd0->depth, &boundary_expanded_iss, &bnd0->sections);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocalTuple(dm, boundary_expanded_iss[0], NULL, &bnd0->coordinates);CHKERRQ(ierr);
    /* self-check */
    {
      IS is0;
      ierr = DMPlexExpandedVerticesCoordinatesToFaces_Private(idm, bnd0, &is0);CHKERRQ(ierr);
      ierr = ISEqual(is0, boundary_is, &flg);CHKERRQ(ierr);
      if (!flg) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "DMPlexExpandedVerticesCoordinatesToFaces_Private produced a wrong IS");
      ierr = ISDestroy(&is0);CHKERRQ(ierr);
    }
  } else {
    ierr = VecCreateSeq(PETSC_COMM_SELF, 0, &bnd0->coordinates);CHKERRQ(ierr);
  }

  {
    Vec         tmp;
    VecScatter  sc;
    IS          xis;
    PetscInt    n;

    /* just convert seq vectors to mpi vector */
    ierr = VecGetLocalSize(bnd0->coordinates, &n);CHKERRQ(ierr);
    ierr = MPI_Bcast(&n, 1, MPIU_INT, rootrank, comm);CHKERRMPI(ierr);
    if (rank == rootrank) {
      ierr = VecCreateMPI(comm, n, n, &tmp);CHKERRQ(ierr);
    } else {
      ierr = VecCreateMPI(comm, 0, n, &tmp);CHKERRQ(ierr);
    }
    ierr = VecCopy(bnd0->coordinates, tmp);CHKERRQ(ierr);
    ierr = VecDestroy(&bnd0->coordinates);CHKERRQ(ierr);
    bnd0->coordinates = tmp;

    /* replicate coordinates from root rank to all ranks */
    ierr = VecCreateMPI(comm, n, n*size, &bnd->coordinates);CHKERRQ(ierr);
    ierr = ISCreateStride(comm, n, 0, 1, &xis);CHKERRQ(ierr);
    ierr = VecScatterCreate(bnd0->coordinates, xis, bnd->coordinates, NULL, &sc);CHKERRQ(ierr);
    ierr = VecScatterBegin(sc, bnd0->coordinates, bnd->coordinates, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(  sc, bnd0->coordinates, bnd->coordinates, INSERT_VALUES, SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&sc);CHKERRQ(ierr);
    ierr = ISDestroy(&xis);CHKERRQ(ierr);
  }
  bnd->depth = bnd0->depth;
  ierr = MPI_Bcast(&bnd->depth, 1, MPIU_INT, rootrank, comm);CHKERRMPI(ierr);
  ierr = PetscMalloc1(bnd->depth, &bnd->sections);CHKERRQ(ierr);
  for (d=0; d<bnd->depth; d++) {
    ierr = PetscSectionReplicate_Private(comm, rootrank, (rank == rootrank) ? bnd0->sections[d] : NULL, &bnd->sections[d]);CHKERRQ(ierr);
  }

  if (rank == rootrank) {
    ierr = DMPlexRestoreConeRecursive(idm, boundary_is, &bnd0->depth, &boundary_expanded_iss, &bnd0->sections);CHKERRQ(ierr);
  }
  ierr = PortableBoundaryDestroy(&bnd0);CHKERRQ(ierr);
  ierr = DMRemoveLabelBySelf(idm, &label, PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&label);CHKERRQ(ierr);
  ierr = ISDestroy(&boundary_is);CHKERRQ(ierr);
  ierr = DMDestroy(&idm);CHKERRQ(ierr);
  *boundary = bnd;
  PetscFunctionReturn(0);
}

/* get faces of inter-partition interface */
static PetscErrorCode DMPlexGetInterfaceFaces_Private(DM ipdm, IS boundary_faces_is, IS *interface_faces_is)
{
  MPI_Comm               comm;
  DMLabel                label;
  IS                     part_boundary_faces_is;
  const char             partBoundaryName[] = "DMPlexDistributeInterpolateMarkInterface_partBoundary";
  PetscInt               value = 1;
  DMPlexInterpolatedFlag intp;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ipdm, &comm);CHKERRQ(ierr);
  ierr = DMPlexIsInterpolatedCollective(ipdm, &intp);CHKERRQ(ierr);
  if (intp != DMPLEX_INTERPOLATED_FULL) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "only for fully interpolated DMPlex");

  /* get ipdm partition boundary (partBoundary) */
  ierr = DMLabelCreate(PETSC_COMM_SELF, partBoundaryName, &label);CHKERRQ(ierr);
  ierr = DMAddLabel(ipdm, label);CHKERRQ(ierr);
  ierr = DMPlexMarkBoundaryFaces(ipdm, value, label);CHKERRQ(ierr);
  ierr = DMLabelViewFromOptionsOnComm_Private(label, "-ipdm_part_boundary_view", comm);CHKERRQ(ierr);
  ierr = DMLabelGetStratumISOnComm_Private(label, value, comm, &part_boundary_faces_is);CHKERRQ(ierr);
  ierr = DMRemoveLabelBySelf(ipdm, &label, PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&label);CHKERRQ(ierr);

  /* remove ipdm whole-domain boundary (boundary_faces_is) from ipdm partition boundary (part_boundary_faces_is), resulting just in inter-partition interface */
  ierr = ISDifference(part_boundary_faces_is,boundary_faces_is,interface_faces_is);CHKERRQ(ierr);
  ierr = ISDestroy(&part_boundary_faces_is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* compute inter-partition interface including edges and vertices */
static PetscErrorCode DMPlexComputeCompleteInterface_Private(DM ipdm, IS interface_faces_is, IS *interface_is)
{
  DMLabel                label;
  PetscInt               value = 1;
  const char             interfaceName[] = "DMPlexDistributeInterpolateMarkInterface_interface";
  DMPlexInterpolatedFlag intp;
  MPI_Comm               comm;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ipdm, &comm);CHKERRQ(ierr);
  ierr = DMPlexIsInterpolatedCollective(ipdm, &intp);CHKERRQ(ierr);
  if (intp != DMPLEX_INTERPOLATED_FULL) SETERRQ(comm, PETSC_ERR_ARG_WRONG, "only for fully interpolated DMPlex");

  ierr = DMLabelCreate(PETSC_COMM_SELF, interfaceName, &label);CHKERRQ(ierr);
  ierr = DMAddLabel(ipdm, label);CHKERRQ(ierr);
  ierr = DMLabelSetStratumIS(label, value, interface_faces_is);CHKERRQ(ierr);
  ierr = DMLabelViewFromOptionsOnComm_Private(label, "-interface_faces_view", comm);CHKERRQ(ierr);
  ierr = DMPlexLabelComplete(ipdm, label);CHKERRQ(ierr);
  ierr = DMLabelViewFromOptionsOnComm_Private(label, "-interface_view", comm);CHKERRQ(ierr);
  ierr = DMLabelGetStratumISOnComm_Private(label, value, comm, interface_is);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)*interface_is, "interface_is");CHKERRQ(ierr);
  ierr = ISViewFromOptions(*interface_is, NULL, "-interface_is_view");CHKERRQ(ierr);
  ierr = DMRemoveLabelBySelf(ipdm, &label, PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMLabelDestroy(&label);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PointSFGetOutwardInterfacePoints(PetscSF sf, IS *is)
{
  PetscInt        n;
  const PetscInt  *arr;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscSFGetGraph(sf, NULL, &n, &arr, NULL);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)sf), n, arr, PETSC_USE_POINTER, is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PointSFGetInwardInterfacePoints(PetscSF sf, IS *is)
{
  PetscInt        n;
  const PetscInt  *rootdegree;
  PetscInt        *arr;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscSFSetUp(sf);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeBegin(sf, &rootdegree);CHKERRQ(ierr);
  ierr = PetscSFComputeDegreeEnd(sf, &rootdegree);CHKERRQ(ierr);
  ierr = PetscSFComputeMultiRootOriginalNumbering(sf, rootdegree, &n, &arr);CHKERRQ(ierr);
  ierr = ISCreateGeneral(PetscObjectComm((PetscObject)sf), n, arr, PETSC_OWN_POINTER, is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PointSFGetInterfacePoints_Private(PetscSF pointSF, IS *is)
{
  IS pointSF_out_is, pointSF_in_is;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PointSFGetOutwardInterfacePoints(pointSF, &pointSF_out_is);CHKERRQ(ierr);
  ierr = PointSFGetInwardInterfacePoints(pointSF, &pointSF_in_is);CHKERRQ(ierr);
  ierr = ISExpand(pointSF_out_is, pointSF_in_is, is);CHKERRQ(ierr);
  ierr = ISDestroy(&pointSF_out_is);CHKERRQ(ierr);
  ierr = ISDestroy(&pointSF_in_is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define CHKERRMY(ierr) do {if (PetscUnlikely(ierr)) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "PointSF is wrong. Unable to show details!");} while (0)

static PetscErrorCode ViewPointsWithType_Internal(DM dm, IS pointsIS, PetscViewer v)
{
  DMLabel         label;
  PetscSection    coordsSection;
  Vec             coordsVec;
  PetscScalar     *coordsScalar;
  PetscInt        coneSize, depth, dim, i, p, npoints;
  const PetscInt  *points;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMGetCoordinateSection(dm, &coordsSection);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordsVec);CHKERRQ(ierr);
  ierr = VecGetArray(coordsVec, &coordsScalar);CHKERRQ(ierr);
  ierr = ISGetLocalSize(pointsIS, &npoints);CHKERRQ(ierr);
  ierr = ISGetIndices(pointsIS, &points);CHKERRQ(ierr);
  ierr = DMPlexGetDepthLabel(dm, &label);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPushTab(v);CHKERRQ(ierr);
  for (i=0; i<npoints; i++) {
    p = points[i];
    ierr = DMLabelGetValue(label, p, &depth);CHKERRQ(ierr);
    if (!depth) {
      PetscInt        c, n, o;
      PetscReal       coords[3];
      char            coordstr[128];

      ierr = PetscSectionGetDof(coordsSection, p, &n);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(coordsSection, p, &o);CHKERRQ(ierr);
      for (c=0; c<n; c++) coords[c] = PetscRealPart(coordsScalar[o+c]);
      ierr = coord2str(coordstr, sizeof(coordstr), n, coords, 1.0);CHKERRQ(ierr);
      ierr = PetscViewerASCIISynchronizedPrintf(v, "vertex %D w/ coordinates %s\n", p, coordstr);CHKERRQ(ierr);
    } else {
      char            entityType[16];

      switch (depth) {
        case 1: ierr = PetscStrcpy(entityType, "edge");CHKERRQ(ierr); break;
        case 2: ierr = PetscStrcpy(entityType, "face");CHKERRQ(ierr); break;
        case 3: ierr = PetscStrcpy(entityType, "cell");CHKERRQ(ierr); break;
        default: SETERRQ(PetscObjectComm((PetscObject)v), PETSC_ERR_SUP, "Only for depth <= 3");
      }
      if (depth == dim && dim < 3) {
        ierr = PetscStrlcat(entityType, " (cell)", sizeof(entityType));CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIISynchronizedPrintf(v, "%s %D\n", entityType, p);CHKERRQ(ierr);
    }
    ierr = DMPlexGetConeSize(dm, p, &coneSize);CHKERRQ(ierr);
    if (coneSize) {
      const PetscInt *cone;
      IS             coneIS;

      ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PETSC_COMM_SELF, coneSize, cone, PETSC_USE_POINTER, &coneIS);CHKERRQ(ierr);
      ierr = ViewPointsWithType_Internal(dm, coneIS, v);CHKERRQ(ierr);
      ierr = ISDestroy(&coneIS);CHKERRQ(ierr);
    }
  }
  ierr = PetscViewerASCIIPopTab(v);CHKERRQ(ierr);
  ierr = VecRestoreArray(coordsVec, &coordsScalar);CHKERRQ(ierr);
  ierr = ISRestoreIndices(pointsIS, &points);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode ViewPointsWithType(DM dm, IS points, PetscViewer v)
{
  PetscBool       flg;
  PetscInt        npoints;
  PetscMPIInt     rank;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)v, PETSCVIEWERASCII, &flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PetscObjectComm((PetscObject)v), PETSC_ERR_SUP, "Only for ASCII viewer");
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)v), &rank);CHKERRMPI(ierr);
  ierr = PetscViewerASCIIPushSynchronized(v);CHKERRQ(ierr);
  ierr = ISGetLocalSize(points, &npoints);CHKERRQ(ierr);
  if (npoints) {
    ierr = PetscViewerASCIISynchronizedPrintf(v, "[%d] --------\n", rank);CHKERRQ(ierr);
    ierr = ViewPointsWithType_Internal(dm, points, v);CHKERRQ(ierr);
  }
  ierr = PetscViewerFlush(v);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopSynchronized(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexComparePointSFWithInterface_Private(DM ipdm, IS interface_is)
{
  PetscSF         pointsf;
  IS              pointsf_is;
  PetscBool       flg;
  MPI_Comm        comm;
  PetscMPIInt     size;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)ipdm, &comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRMPI(ierr);
  ierr = DMGetPointSF(ipdm, &pointsf);CHKERRQ(ierr);
  if (pointsf) {
    PetscInt nroots;
    ierr = PetscSFGetGraph(pointsf, &nroots, NULL, NULL, NULL);CHKERRQ(ierr);
    if (nroots < 0) pointsf = NULL; /* uninitialized SF */
  }
  if (!pointsf) {
    PetscInt N=0;
    if (interface_is) {ierr = ISGetSize(interface_is, &N);CHKERRQ(ierr);}
    if (N) SETERRQ(comm, PETSC_ERR_PLIB, "interface_is should be NULL or empty for PointSF being NULL");
    PetscFunctionReturn(0);
  }

  /* get PointSF points as IS pointsf_is */
  ierr = PointSFGetInterfacePoints_Private(pointsf, &pointsf_is);CHKERRQ(ierr);

  /* compare pointsf_is with interface_is */
  ierr = ISEqual(interface_is, pointsf_is, &flg);CHKERRQ(ierr);
  ierr = MPI_Allreduce(MPI_IN_PLACE,&flg,1,MPIU_BOOL,MPI_LAND,comm);CHKERRMPI(ierr);
  if (!flg) {
    IS pointsf_extra_is, pointsf_missing_is;
    PetscViewer errv = PETSC_VIEWER_STDERR_(comm);
    ierr = ISDifference(interface_is, pointsf_is, &pointsf_missing_is);CHKERRMY(ierr);
    ierr = ISDifference(pointsf_is, interface_is, &pointsf_extra_is);CHKERRMY(ierr);
    ierr = PetscViewerASCIIPrintf(errv, "Points missing in PointSF:\n");CHKERRMY(ierr);
    ierr = ViewPointsWithType(ipdm, pointsf_missing_is, errv);CHKERRMY(ierr);
    ierr = PetscViewerASCIIPrintf(errv, "Extra points in PointSF:\n");CHKERRMY(ierr);
    ierr = ViewPointsWithType(ipdm, pointsf_extra_is, errv);CHKERRMY(ierr);
    ierr = ISDestroy(&pointsf_extra_is);CHKERRMY(ierr);
    ierr = ISDestroy(&pointsf_missing_is);CHKERRMY(ierr);
    SETERRQ(comm, PETSC_ERR_PLIB, "PointSF is wrong! See details above.");
  }
  ierr = ISDestroy(&pointsf_is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* remove faces & edges from label, leave just vertices */
static PetscErrorCode DMPlexISFilterVertices_Private(DM dm, IS points)
{
  PetscInt        vStart, vEnd;
  MPI_Comm        comm;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = ISGeneralFilter(points, vStart, vEnd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
  DMPlexCheckPointSFHeavy - Thoroughly test that the PointSF after parallel DMPlexInterpolate() includes exactly all interface points.

  Collective

  Input Parameters:
. dm - The DMPlex object

  Notes:
  The input DMPlex must be serial (one partition has all points, the other partitions have no points).
  This is a heavy test which involves DMPlexInterpolate() if the input DM is not interpolated yet, and depends on having a representation of the whole-domain boundary (PortableBoundary), which can be obtained only by DMPlexGetExpandedBoundary_Private() (which involves DMPlexInterpolate() of a sequential DM).
  This is mainly intended for debugging/testing purposes.

  Algorithm:
  1. boundary faces of the serial version of the whole mesh are found using DMPlexMarkBoundaryFaces()
  2. boundary faces are translated into vertices using DMPlexGetConeRecursive() and these are translated into coordinates - this description (aka PortableBoundary) is completely independent of partitioning and point numbering
  3. the mesh is distributed or loaded in parallel
  4. boundary faces of the distributed mesh are reconstructed from PortableBoundary using DMPlexFindVertices()
  5. partition boundary faces of the parallel mesh are found using DMPlexMarkBoundaryFaces()
  6. partition interfaces are computed as set difference of partition boundary faces minus the reconstructed boundary
  7. check that interface covered by PointSF (union of inward and outward points) is equal to the partition interface for each rank, otherwise print the difference and throw an error

  Level: developer

.seealso: DMGetPointSF(), DMPlexCheckSymmetry(), DMPlexCheckSkeleton(), DMPlexCheckFaces()
*/
static PetscErrorCode DMPlexCheckPointSFHeavy(DM dm, PortableBoundary bnd)
{
  DM                     ipdm=NULL;
  IS                     boundary_faces_is, interface_faces_is, interface_is;
  DMPlexInterpolatedFlag intp;
  MPI_Comm               comm;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)dm, &comm);CHKERRQ(ierr);

  ierr = DMPlexIsInterpolatedCollective(dm, &intp);CHKERRQ(ierr);
  if (intp == DMPLEX_INTERPOLATED_FULL) {
    ipdm = dm;
  } else {
    /* create temporary interpolated DM if input DM is not interpolated */
    ierr = DMPlexSetOrientInterface_Private(dm, PETSC_FALSE);CHKERRQ(ierr);
    ierr = DMPlexInterpolate(dm, &ipdm);CHKERRQ(ierr); /* with DMPlexOrientInterface_Internal() call skipped so that PointSF issues are left to DMPlexComparePointSFWithInterface_Private() below */
    ierr = DMPlexSetOrientInterface_Private(dm, PETSC_TRUE);CHKERRQ(ierr);
  }
  ierr = DMViewFromOptions(ipdm, NULL, "-ipdm_view");CHKERRQ(ierr);

  /* recover ipdm whole-domain boundary faces from the expanded vertices coordinates */
  ierr = DMPlexExpandedVerticesCoordinatesToFaces_Private(ipdm, bnd, &boundary_faces_is);CHKERRQ(ierr);
  /* get inter-partition interface faces (interface_faces_is)*/
  ierr = DMPlexGetInterfaceFaces_Private(ipdm, boundary_faces_is, &interface_faces_is);CHKERRQ(ierr);
  /* compute inter-partition interface including edges and vertices (interface_is) */
  ierr = DMPlexComputeCompleteInterface_Private(ipdm, interface_faces_is, &interface_is);CHKERRQ(ierr);
  /* destroy immediate ISs */
  ierr = ISDestroy(&boundary_faces_is);CHKERRQ(ierr);
  ierr = ISDestroy(&interface_faces_is);CHKERRQ(ierr);

  /* for uninterpolated case, keep just vertices in interface */
  if (!intp) {
    ierr = DMPlexISFilterVertices_Private(ipdm, interface_is);CHKERRQ(ierr);
    ierr = DMDestroy(&ipdm);CHKERRQ(ierr);
  }

  /* compare PointSF with the boundary reconstructed from coordinates */
  ierr = DMPlexComparePointSFWithInterface_Private(dm, interface_is);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "DMPlexCheckPointSFHeavy PASSED\n");CHKERRQ(ierr);
  ierr = ISDestroy(&interface_is);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = PetscLogStageRegister("create",&stage[0]);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("distribute",&stage[1]);CHKERRQ(ierr);
  ierr = PetscLogStageRegister("interpolate",&stage[2]);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  if (user.nPointsToExpand) {
    ierr = TestExpandPoints(dm, &user);CHKERRQ(ierr);
  }
  if (user.ncoords) {
    ierr = ViewVerticesFromCoords(dm, user.ncoords/user.dim, user.coords, user.coordsTol, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  testset:
    nsize: 2
    args: -dm_view ascii::ascii_info_detail
    args: -dm_plex_check_all
    test:
      suffix: 1_tri_dist0
      args: -distribute 0 -interpolate {{none create}separate output}
    test:
      suffix: 1_tri_dist1
      args: -distribute 1 -interpolate {{none create after_distribute}separate output}
    test:
      suffix: 1_quad_dist0
      args: -cell_simplex 0 -distribute 0 -interpolate {{none create}separate output}
    test:
      suffix: 1_quad_dist1
      args: -cell_simplex 0 -distribute 1 -interpolate {{none create after_distribute}separate output}
    test:
      suffix: 1_1d_dist1
      args: -dim 1 -distribute 1

  testset:
    nsize: 3
    args: -testnum 1 -interpolate create
    args: -dm_plex_check_all
    test:
      suffix: 2
      args: -dm_view ascii::ascii_info_detail
    test:
      suffix: 2a
      args: -dm_plex_check_cones_conform_on_interfaces_verbose
    test:
      suffix: 2b
      args: -test_expand_points 0,1,2,5,6
    test:
      suffix: 2c
      args: -test_expand_points 0,1,2,5,6 -test_expand_points_empty

  testset:
    # the same as 1% for 3D
    nsize: 2
    args: -dim 3 -dm_view ascii::ascii_info_detail
    args: -dm_plex_check_all
    test:
      suffix: 4_tet_dist0
      args: -distribute 0 -interpolate {{none create}separate output}
    test:
      suffix: 4_tet_dist1
      args: -distribute 1 -interpolate {{none create after_distribute}separate output}
    test:
      suffix: 4_hex_dist0
      args: -cell_simplex 0 -distribute 0 -interpolate {{none create}separate output}
    test:
      suffix: 4_hex_dist1
      args: -cell_simplex 0 -distribute 1 -interpolate {{none create after_distribute}separate output}

  test:
    # the same as 4_tet_dist0 but test different initial orientations
    suffix: 4_tet_test_orient
    nsize: 2
    args: -dim 3 -distribute 0
    args: -dm_plex_check_all
    args: -rotate_interface_0 {{0 1 2 11 12 13}}
    args: -rotate_interface_1 {{0 1 2 11 12 13}}

  testset:
    requires: exodusii
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/TwoQuads.exo
    args: -dm_view ascii::ascii_info_detail
    args: -dm_plex_check_all
    args: -custom_view
    test:
      suffix: 5_seq
      nsize: 1
      args: -distribute 0 -interpolate {{none create}separate output}
    test:
      # Detail viewing in a non-distributed mesh is broken because the DMLabelView() is collective, but the label is not shared
      suffix: 5_dist0
      nsize: 2
      args: -distribute 0 -interpolate {{none create}separate output} -dm_view
    test:
      suffix: 5_dist1
      nsize: 2
      args: -distribute 1 -interpolate {{none create after_distribute}separate output}

  testset:
    nsize: {{1 2 4}}
    args: -use_generator
    args: -dm_plex_check_all
    args: -distribute -interpolate none
    test:
      suffix: 6_tri
      requires: triangle
      args: -faces {{2,2  1,3  7,4}} -cell_simplex 1 -dm_plex_generator triangle
    test:
      suffix: 6_quad
      args: -faces {{2,2  1,3  7,4}} -cell_simplex 0
    test:
      suffix: 6_tet
      requires: ctetgen
      args: -faces {{2,2,2  1,3,5  3,4,7}} -cell_simplex 1 -dm_plex_generator ctetgen
    test:
      suffix: 6_hex
      args: -faces {{2,2,2  1,3,5  3,4,7}} -cell_simplex 0
  testset:
    nsize: {{1 2 4}}
    args: -use_generator
    args: -dm_plex_check_all
    args: -distribute -interpolate create
    test:
      suffix: 6_int_tri
      requires: triangle
      args: -faces {{2,2  1,3  7,4}} -cell_simplex 1 -dm_plex_generator triangle
    test:
      suffix: 6_int_quad
      args: -faces {{2,2  1,3  7,4}} -cell_simplex 0
    test:
      suffix: 6_int_tet
      requires: ctetgen
      args: -faces {{2,2,2  1,3,5  3,4,7}} -cell_simplex 1 -dm_plex_generator ctetgen
    test:
      suffix: 6_int_hex
      args: -faces {{2,2,2  1,3,5  3,4,7}} -cell_simplex 0
  testset:
    nsize: {{2 4}}
    args: -use_generator
    args: -dm_plex_check_all
    args: -distribute -interpolate after_distribute
    test:
      suffix: 6_parint_tri
      requires: triangle
      args: -faces {{2,2  1,3  7,4}} -cell_simplex 1 -dm_plex_generator triangle
    test:
      suffix: 6_parint_quad
      args: -faces {{2,2  1,3  7,4}} -cell_simplex 0
    test:
      suffix: 6_parint_tet
      requires: ctetgen
      args: -faces {{2,2,2  1,3,5  3,4,7}} -cell_simplex 1 -dm_plex_generator ctetgen
    test:
      suffix: 6_parint_hex
      args: -faces {{2,2,2  1,3,5  3,4,7}} -cell_simplex 0

  testset: # 7 EXODUS
    requires: exodusii
    args: -dm_plex_check_all
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo
    args: -distribute
    test: # seq load, simple partitioner
      suffix: 7_exo
      nsize: {{1 2 4 5}}
      args: -interpolate none
    test: # seq load, seq interpolation, simple partitioner
      suffix: 7_exo_int_simple
      nsize: {{1 2 4 5}}
      args: -interpolate create
    test: # seq load, seq interpolation, metis partitioner
      suffix: 7_exo_int_metis
      requires: parmetis
      nsize: {{2 4 5}}
      args: -interpolate create
      args: -petscpartitioner_type parmetis
    test: # seq load, simple partitioner, par interpolation
      suffix: 7_exo_simple_int
      nsize: {{2 4 5}}
      args: -interpolate after_distribute
    test: # seq load, metis partitioner, par interpolation
      suffix: 7_exo_metis_int
      requires: parmetis
      nsize: {{2 4 5}}
      args: -interpolate after_distribute
      args: -petscpartitioner_type parmetis

  testset: # 7 HDF5 SEQUANTIAL LOAD
    requires: hdf5 !complex
    args: -dm_plex_check_all
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.h5 -dm_plex_create_from_hdf5_xdmf
    args: -dm_plex_hdf5_force_sequential
    args: -distribute
    test: # seq load, simple partitioner
      suffix: 7_seq_hdf5_simple
      nsize: {{1 2 4 5}}
      args: -interpolate none
    test: # seq load, seq interpolation, simple partitioner
      suffix: 7_seq_hdf5_int_simple
      nsize: {{1 2 4 5}}
      args: -interpolate after_create
    test: # seq load, seq interpolation, metis partitioner
      nsize: {{2 4 5}}
      suffix: 7_seq_hdf5_int_metis
      requires: parmetis
      args: -interpolate after_create
      args: -petscpartitioner_type parmetis
    test: # seq load, simple partitioner, par interpolation
      suffix: 7_seq_hdf5_simple_int
      nsize: {{2 4 5}}
      args: -interpolate after_distribute
    test: # seq load, metis partitioner, par interpolation
      nsize: {{2 4 5}}
      suffix: 7_seq_hdf5_metis_int
      requires: parmetis
      args: -interpolate after_distribute
      args: -petscpartitioner_type parmetis

  testset: # 7 HDF5 PARALLEL LOAD
    requires: hdf5 !complex
    nsize: {{2 4 5}}
    args: -dm_plex_check_all
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.h5 -dm_plex_create_from_hdf5_xdmf
    test: # par load
      suffix: 7_par_hdf5
      args: -interpolate none
    test: # par load, par interpolation
      suffix: 7_par_hdf5_int
      args: -interpolate after_create
    test: # par load, parmetis repartitioner
      TODO: Parallel partitioning of uninterpolated meshes not supported
      suffix: 7_par_hdf5_parmetis
      requires: parmetis
      args: -distribute -petscpartitioner_type parmetis
      args: -interpolate none
    test: # par load, par interpolation, parmetis repartitioner
      suffix: 7_par_hdf5_int_parmetis
      requires: parmetis
      args: -distribute -petscpartitioner_type parmetis
      args: -interpolate after_create
    test: # par load, parmetis partitioner, par interpolation
      TODO: Parallel partitioning of uninterpolated meshes not supported
      suffix: 7_par_hdf5_parmetis_int
      requires: parmetis
      args: -distribute -petscpartitioner_type parmetis
      args: -interpolate after_distribute

    test:
      suffix: 7_hdf5_hierarch
      requires: hdf5 ptscotch !complex
      nsize: {{2 3 4}separate output}
      args: -distribute
      args: -interpolate after_create
      args: -petscpartitioner_type matpartitioning -petscpartitioner_view ::ascii_info
      args: -mat_partitioning_type hierarch -mat_partitioning_hierarchical_nfineparts 2
      args: -mat_partitioning_hierarchical_coarseparttype ptscotch -mat_partitioning_hierarchical_fineparttype ptscotch

  test:
    suffix: 8
    requires: hdf5 !complex
    nsize: 4
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.h5 -dm_plex_create_from_hdf5_xdmf
    args: -distribute 0 -interpolate after_create
    args: -view_vertices_from_coords 0.,1.,0.,-0.5,1.,0.,0.583,-0.644,0.,-2.,-2.,-2. -view_vertices_from_coords_tol 1e-3
    args: -dm_plex_check_all
    args: -custom_view

  testset: # 9 HDF5 SEQUANTIAL LOAD
    requires: hdf5 !complex datafilespath
    args: -dm_plex_check_all
    args: -filename ${DATAFILESPATH}/meshes/cube-hexahedra-refined.h5 -dm_plex_create_from_hdf5_xdmf -dm_plex_hdf5_topology_path /cells -dm_plex_hdf5_geometry_path /coordinates
    args: -dm_plex_hdf5_force_sequential
    args: -distribute
    test: # seq load, simple partitioner
      suffix: 9_seq_hdf5_simple
      nsize: {{1 2 4 5}}
      args: -interpolate none
    test: # seq load, seq interpolation, simple partitioner
      suffix: 9_seq_hdf5_int_simple
      nsize: {{1 2 4 5}}
      args: -interpolate after_create
    test: # seq load, seq interpolation, metis partitioner
      nsize: {{2 4 5}}
      suffix: 9_seq_hdf5_int_metis
      requires: parmetis
      args: -interpolate after_create
      args: -petscpartitioner_type parmetis
    test: # seq load, simple partitioner, par interpolation
      suffix: 9_seq_hdf5_simple_int
      nsize: {{2 4 5}}
      args: -interpolate after_distribute
    test: # seq load, simple partitioner, par interpolation
      # This is like 9_seq_hdf5_simple_int but testing error output of DMPlexCheckPointSFHeavy().
      # Once 9_seq_hdf5_simple_int gets fixed, this one gets broken.
      # We can then provide an intentionally broken mesh instead.
      TODO: This test is broken because PointSF is fixed.
      suffix: 9_seq_hdf5_simple_int_err
      nsize: 4
      args: -interpolate after_distribute
      filter: sed -e "/PETSC ERROR/,$$d"
    test: # seq load, metis partitioner, par interpolation
      nsize: {{2 4 5}}
      suffix: 9_seq_hdf5_metis_int
      requires: parmetis
      args: -interpolate after_distribute
      args: -petscpartitioner_type parmetis

  testset: # 9 HDF5 PARALLEL LOAD
    requires: hdf5 !complex datafilespath
    nsize: {{2 4 5}}
    args: -dm_plex_check_all
    args: -filename ${DATAFILESPATH}/meshes/cube-hexahedra-refined.h5 -dm_plex_create_from_hdf5_xdmf -dm_plex_hdf5_topology_path /cells -dm_plex_hdf5_geometry_path /coordinates
    test: # par load
      suffix: 9_par_hdf5
      args: -interpolate none
    test: # par load, par interpolation
      suffix: 9_par_hdf5_int
      args: -interpolate after_create
    test: # par load, parmetis repartitioner
      TODO: Parallel partitioning of uninterpolated meshes not supported
      suffix: 9_par_hdf5_parmetis
      requires: parmetis
      args: -distribute -petscpartitioner_type parmetis
      args: -interpolate none
    test: # par load, par interpolation, parmetis repartitioner
      suffix: 9_par_hdf5_int_parmetis
      requires: parmetis
      args: -distribute -petscpartitioner_type parmetis
      args: -interpolate after_create
    test: # par load, parmetis partitioner, par interpolation
      TODO: Parallel partitioning of uninterpolated meshes not supported
      suffix: 9_par_hdf5_parmetis_int
      requires: parmetis
      args: -distribute -petscpartitioner_type parmetis
      args: -interpolate after_distribute

  testset: # 10 HDF5 PARALLEL LOAD
    requires: hdf5 !complex datafilespath
    nsize: {{2 4 7}}
    args: -dm_plex_check_all
    args: -filename ${DATAFILESPATH}/meshes/cube-hexahedra-refined2.h5 -dm_plex_create_from_hdf5_xdmf -dm_plex_hdf5_topology_path /topo -dm_plex_hdf5_geometry_path /geom
    test: # par load, par interpolation
      suffix: 10_par_hdf5_int
      args: -interpolate after_create
TEST*/
