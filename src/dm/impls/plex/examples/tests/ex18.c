static char help[] = "Tests for parallel mesh loading\n\n";

#include <petsc/private/dmpleximpl.h>

/* List of test meshes

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

typedef enum {NONE, SERIAL, PARALLEL} InterpType;

typedef struct {
  PetscInt   debug;                        /* The debugging level */
  PetscInt   testNum;                      /* Indicates the mesh to create */
  PetscInt   dim;                          /* The topological mesh dimension */
  PetscBool  cellSimplex;                  /* Use simplices or hexes */
  PetscBool  distribute;                   /* Distribute the mesh */
  InterpType interpolate;                  /* Interpolate the mesh before or after DMPlexDistribute() */
  PetscBool  useGenerator;                 /* Construct mesh with a mesh generator */
  PetscBool  testOrientIF;                 /* Test for different original interface orientations */
  PetscInt   ornt[2];                      /* Orientation of interface on rank 0 and rank 1 */
  PetscInt   faces[3];                     /* Number of faces per dimension for generator */
  char       filename[PETSC_MAX_PATH_LEN]; /* Import mesh from file */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  const char    *interpTypes[3]  = {"none", "serial", "parallel"};
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
  options->ornt[0]      = 0;
  options->ornt[1]      = 0;
  options->faces[0]     = 2;
  options->faces[1]     = 2;
  options->faces[2]     = 2;
  options->filename[0]  = '\0';

  ierr = PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-debug", "The debugging level", "ex18.c", options->debug, &options->debug, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-testnum", "The mesh to create", "ex18.c", options->testNum, &options->testNum, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex18.c", options->dim, &options->dim, &flg1);CHKERRQ(ierr);
  if (options->dim < 1 || options->dim > 3) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "dimension set to %d, must be between 1 and 3", options->dim);
  ierr = PetscOptionsBool("-cell_simplex", "Use simplices if true, otherwise hexes", "ex18.c", options->cellSimplex, &options->cellSimplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-distribute", "Distribute the mesh", "ex18.c", options->distribute, &options->distribute, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEList("-interpolate", "Type of mesh interpolation, e.g. none, serial, parallel", "ex18.c", interpTypes, 3, interpTypes[options->interpolate], &interp, NULL);CHKERRQ(ierr);
  options->interpolate = (InterpType) interp;
  if (!options->distribute && options->interpolate == PARALLEL) SETERRQ(comm, PETSC_ERR_SUP, "-interpolate parallel  needs  -distribute 1");
  ierr = PetscOptionsBool("-use_generator", "Use a mesh generator to build the mesh", "ex18.c", options->useGenerator, &options->useGenerator, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-rotate_interface_0", "Rotation (relative orientation) of interface on rank 0; implies -interpolate serial -distribute 0", "ex18.c", options->ornt[0], &options->ornt[0], &options->testOrientIF);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-rotate_interface_1", "Rotation (relative orientation) of interface on rank 1; implies -interpolate serial -distribute 0", "ex18.c", options->ornt[1], &options->ornt[1], &flg2);CHKERRQ(ierr);
  if (flg2 != options->testOrientIF) SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "neither or both -rotate_interface_0 -rotate_interface_1 must be set");
  if (options->testOrientIF) {
    PetscInt i;
    for (i=0; i<2; i++) {
      if (options->ornt[i] >= 10) options->ornt[i] = -(options->ornt[i]-10);  /* 11 12 13 become -1 -2 -3 */
    }
    options->interpolate = SERIAL;
    options->distribute = PETSC_FALSE;
  }
  dim = 3;
  ierr = PetscOptionsIntArray("-faces", "Number of faces per dimension", "ex18.c", options->faces, &dim, &flg2);CHKERRQ(ierr);
  if (flg2) {
    if (flg1 && dim != options->dim) SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE, "specified -dim %D is not equal to length %D of -faces (note that -dim can be omitted)", options->dim, dim);
    options->dim = dim;
  }
  ierr = PetscOptionsString("-filename", "The mesh file", "ex18.c", options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateSimplex_2D(MPI_Comm comm, PetscBool interpolate, AppCtx *user, DM *dm)
{
  PetscInt       testNum = user->testNum, p;
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  switch (testNum) {
  case 0:
    if (size != 2) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 2 processes", testNum);
    switch (rank) {
      case 0:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const int      cells[3]  = {0, 1, 2};
        PetscReal      coords[4] = {-0.5, 0.5, 0.0, 0.0};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const int      cells[3]  = {1, 3, 2};
        PetscReal      coords[4] = {0.0, 1.0, 0.5, 0.5};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
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
        const int      cells[3]  = {0, 1, 2};
        PetscReal      coords[4] = {0.0, 1.0, 0.0, 0.0};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 3;
        const int      cells[3]  = {0, 2, 3};
        PetscReal      coords[4] = {0.5, 0.5, 1.0, 1.0};
        PetscInt       markerPoints[6] = {1, 1, 2, 1, 3, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 3; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 2:
      {
        const PetscInt numCells  = 2, numVertices = 1, numCorners = 3;
        const int      cells[6]  = {2, 4, 3, 2, 1, 4};
        PetscReal      coords[2] = {1.0, 0.0};
        PetscInt       markerPoints[10] = {2, 1, 3, 1, 4, 1, 5, 1, 6, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
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

/* TODO this dirty hotfix can be removed once point SF is fixed systematically */
static PetscErrorCode CreateSimplex_3D_HotfixSF(DM idm)
{
  PetscSF sf=NULL;
  PetscInt nroots,nleaves;
  const PetscInt *ilocal=NULL;
  const PetscSFNode *iremote=NULL;
  PetscInt idx=3;
  PetscMPIInt rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)idm),&rank);CHKERRQ(ierr);
  ierr = DMGetPointSF(idm, &sf);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf, &nroots, &nleaves, &ilocal, &iremote);CHKERRQ(ierr);
  if (!rank) {
    PetscInt *ilocal1=NULL;
    PetscSFNode *iremote1=NULL;

    /* insert 8 <- (1,6) as leave idx */
    ierr = PetscMalloc1(nleaves+1, &ilocal1);CHKERRQ(ierr);
    ierr = PetscMalloc1(nleaves+1, &iremote1);CHKERRQ(ierr);
    ierr = PetscMemcpy(ilocal1, ilocal, idx*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(iremote1, iremote, idx*sizeof(PetscSFNode));CHKERRQ(ierr);
    ilocal1[idx] = 8;
    iremote1[idx].rank = 1;
    iremote1[idx].index = 6;
    ierr = PetscMemcpy(ilocal1+idx+1, ilocal+idx, (nleaves-idx)*sizeof(PetscInt));CHKERRQ(ierr);
    ierr = PetscMemcpy(iremote1+idx+1, iremote+idx, (nleaves-idx)*sizeof(PetscSFNode));CHKERRQ(ierr);
    ierr = PetscSFSetGraph(sf, nroots, nleaves+1, ilocal1, PETSC_OWN_POINTER, iremote1, PETSC_OWN_POINTER);CHKERRQ(ierr);
  } else {
    ierr = PetscSFSetGraph(sf, nroots, nleaves, ilocal, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateSimplex_3D(MPI_Comm comm, PetscBool interpolate, AppCtx *user, DM *dm)
{
  PetscInt       testNum = user->testNum, p;
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  if (user->interpolate) {
    /* TODO this dirty hotfix can be removed once point SF is fixed systematically */
    ierr = PetscOptionsSetValue(NULL, "-dm_plex_interpolate_orient_interfaces", "0");CHKERRQ(ierr);  /* disable the fix in DMPlexInterpolate, will be done later */
  }
  switch (testNum) {
  case 0:
    if (size != 2) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 2 processes", testNum);
    switch (rank) {
      case 0:
      {
        const PetscInt numCells  = 1, numVertices = 2, numCorners = 4;
        const int      cells[4]  = {0, 2, 1, 3};
        PetscReal      coords[6] = {0.0, 0.0, -0.5,  0.0, -0.5, 0.0};
        PetscInt       markerPoints[8] = {1, 1, 2, 1, 3, 1, 4, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 3, numCorners = 4;
        const int      cells[4]  = {1, 2, 4, 3};
        PetscReal      coords[9] = {1.0, 0.0, 0.0,  0.0, 0.5, 0.0,  0.0, 0.0, 0.5};
        PetscInt       markerPoints[8] = {1, 1, 2, 1, 3, 1, 4, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      default: SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh for rank %d", rank);
    }
    break;
  default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No test mesh %D", testNum);
  }
  if (user->interpolate == SERIAL) {
    /* TODO this dirty hotfix can be removed once point SF is fixed systematically */
    ierr = CreateSimplex_3D_HotfixSF(*dm);CHKERRQ(ierr);
  }
  if (user->testOrientIF) {
    PetscInt start;
    PetscBool reverse;
    PetscInt ifp[] = {8, 6};

    ierr = PetscObjectSetName((PetscObject) *dm, "Mesh before orientation");CHKERRQ(ierr);
    ierr = DMViewFromOptions(*dm, NULL, "-before_orientation_dm_view");CHKERRQ(ierr);
    /* rotate interface face ifp[rank] by given orientation ornt[rank] */
    ierr = DMPlexFixFaceOrientations_Translate_Private(user->ornt[rank], &start, &reverse);CHKERRQ(ierr);
    ierr = DMPlexOrientCell_Internal(*dm, ifp[rank], start, reverse);CHKERRQ(ierr);
    ierr = DMPlexCheckFaces(*dm, user->cellSimplex, 0);CHKERRQ(ierr);
  }
  if (user->interpolate == SERIAL) {
    ierr = DMPlexOrientInterface(*dm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateQuad_2D(MPI_Comm comm, PetscBool interpolate, AppCtx *user, DM *dm)
{
  PetscInt       testNum = user->testNum, p;
  PetscMPIInt    rank, size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  switch (testNum) {
  case 0:
    if (size != 2) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 2 processes", testNum);
    switch (rank) {
      case 0:
      {
        const PetscInt numCells  = 1, numVertices = 3, numCorners = 4;
        const int      cells[4]  = {0, 1, 2, 3};
        PetscReal      coords[6] = {-0.5, 0.0, 0.0, 0.0, 0.0, 1.0};
        PetscInt       markerPoints[4*2] = {1, 1, 2, 1, 3, 1, 4, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
        for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
      }
      break;
      case 1:
      {
        const PetscInt numCells  = 1, numVertices = 3, numCorners = 4;
        const int      cells[4]  = {1, 4, 5, 2};
        PetscReal      coords[6] = {-0.5, 1.0, 0.5, 0.0, 0.5, 1.0};
        PetscInt       markerPoints[4*2] = {1, 1, 2, 1, 3, 1, 4, 1};

        ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
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
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm, &size);CHKERRQ(ierr);
  switch (testNum) {
  case 0:
    if (size != 2) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Test mesh %d only for 2 processes", testNum);
    switch (rank) {
    case 0:
    {
      const PetscInt numCells  = 1, numVertices = 6, numCorners = 8;
      const int      cells[8]  = {0, 3, 2, 1, 4, 5, 6, 7};
      PetscReal      coords[6*3] = {-0.5,0.0,0.0, 0.0,0.0,0.0, 0.0,1.0,0.0, -0.5,1.0,0.0, -0.5,0.0,1.0, 0.0,0.0,1.0};
      PetscInt       markerPoints[8*2] = {2,1,3,1,4,1,5,1,6,1,7,1,8,1,9,1};

      ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
      for (p = 0; p < 4; ++p) {ierr = DMSetLabelValue(*dm, "marker", markerPoints[p*2], markerPoints[p*2+1]);CHKERRQ(ierr);}
    }
    break;
    case 1:
    {
      const PetscInt numCells  = 1, numVertices = 6, numCorners = 8;
      const int      cells[8]  = {1, 2, 9, 8, 5, 10, 11, 6};
      PetscReal      coords[6*3] = {0.0,1.0,1.0, -0.5,1.0,1.0, 0.5,0.0,0.0, 0.5,1.0,0.0, 0.5,0.0,1.0,  0.5,1.0,1.0};
      PetscInt       markerPoints[8*2] = {2,1,3,1,4,1,5,1,6,1,7,1,8,1,9,1};

      ierr = DMPlexCreateFromCellListParallel(comm, user->dim, numCells, numVertices, numCorners, interpolate, cells, user->dim, coords, NULL, dm);CHKERRQ(ierr);
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

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim            = user->dim;
  PetscBool      cellSimplex    = user->cellSimplex;
  PetscBool      useGenerator   = user->useGenerator;
  PetscBool      interpSerial   = user->interpolate == SERIAL ? PETSC_TRUE : PETSC_FALSE;
  PetscBool      interpParallel = user->interpolate == PARALLEL ? PETSC_TRUE : PETSC_FALSE;
  PetscBool      hotfix         = PETSC_FALSE;
  const char    *filename       = user->filename;
  size_t         len;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (len) {
    ierr = DMPlexCreateFromFile(comm, filename, interpSerial, dm);CHKERRQ(ierr);
    ierr = DMGetDimension(*dm, &dim);CHKERRQ(ierr);
  } else if (useGenerator) {
    ierr = DMPlexCreateBoxMesh(comm, dim, cellSimplex, user->faces, NULL, NULL, NULL, interpSerial, dm);CHKERRQ(ierr);
  } else {
    switch (dim) {
    case 2:
      if (cellSimplex) {
        ierr = CreateSimplex_2D(comm, interpSerial, user, dm);CHKERRQ(ierr);
      } else {
        ierr = CreateQuad_2D(comm, interpSerial, user, dm);CHKERRQ(ierr);
      }
      break;
    case 3:
      if (cellSimplex) {
        ierr = CreateSimplex_3D(comm, interpSerial, user, dm);CHKERRQ(ierr);
        hotfix = PETSC_TRUE;
      } else {
        ierr = CreateHex_3D(comm, interpSerial, user, dm);CHKERRQ(ierr);
      }
      break;
    default:
      SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "Cannot make meshes for dimension %D", dim);
    }
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Original Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-orig_dm_view");CHKERRQ(ierr);

  if (user->distribute) {
    DM               pdm = NULL;
    PetscPartitioner part;

    /* Set partitioner options */
    ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetType(part, PETSCPARTITIONERSIMPLE);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);

    /* Redistribute mesh over processes using that partitioner */
    ierr = DMPlexDistribute(*dm, 0, NULL, &pdm);CHKERRQ(ierr);
    if (pdm) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = pdm;
      ierr = PetscObjectSetName((PetscObject) *dm, "Redistributed Mesh");CHKERRQ(ierr);
      ierr = DMViewFromOptions(*dm, NULL, "-dist_dm_view");CHKERRQ(ierr);
    }

    if (interpParallel) {
      DM idm;

      ierr = DMPlexInterpolate(*dm, &idm);CHKERRQ(ierr);
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm = idm;
      if (hotfix) {
        /* TODO this dirty hotfix can be removed once point SF is fixed systematically */
        /* CreateSimplex_3D didn't do the hotfix, do it now */
        ierr = CreateSimplex_3D_HotfixSF(*dm);CHKERRQ(ierr);
        ierr = DMPlexOrientInterface(*dm);CHKERRQ(ierr);
      }
      ierr = PetscObjectSetName((PetscObject) *dm, "Interpolated Redistributed Mesh");CHKERRQ(ierr);
      ierr = DMViewFromOptions(*dm, NULL, "-intp_dm_view");CHKERRQ(ierr);
    }
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Parallel Mesh");CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  if (user.interpolate != NONE) {
    ierr = DMPlexCheckPointSF(dm);CHKERRQ(ierr);
    ierr = DMPlexCheckFaces(dm, user.cellSimplex, 0);CHKERRQ(ierr);
  }
  ierr = DMPlexCheckConesConformOnInterfaces(dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  testset:
    nsize: 2
    args: -dm_view ascii::ascii_info_detail -dm_plex_check_symmetry -dm_plex_check_skeleton unknown -dm_plex_check_geometry
    test:
      suffix: 1_tri_dist0
      args: -distribute 0 -interpolate {{none serial}separate output}
    test:
      # Add back in 'none' and 'parallel'
      suffix: 1_tri_dist1
      args: -distribute 1 -interpolate {{serial}separate output}
    test:
      suffix: 1_quad_dist0
      args: -cell_simplex 0 -distribute 0 -interpolate {{none serial}separate output}
    test:
      # Add back in 'none' and 'parallel'
      suffix: 1_quad_dist1
      args: -cell_simplex 0 -distribute 1 -interpolate {{serial}separate output}

  test:
    suffix: 2
    nsize: 3
    args: -testnum 1 -interpolate serial -dm_view ascii::ascii_info_detail -dm_plex_check_symmetry -dm_plex_check_skeleton unknown -dm_plex_check_geometry

  testset:
    # the same as 1% for 3D
    nsize: 2
    args: -dim 3 -dm_view ascii::ascii_info_detail
    test:
      suffix: 4_tet_dist0
      args: -distribute 0 -interpolate {{none serial}separate output}
    test:
      # Add back in 'none' and 'parallel'
      suffix: 4_tet_dist1
      args: -distribute 1 -interpolate {{serial}separate output}
    test:
      TODO: fails due to wrong SF
      suffix: 4_hex_dist0
      args: -cell_simplex 0 -distribute 0 -interpolate {{none serial}separate output}
    test:
      TODO: fails due to wrong SF
      suffix: 4_hex_dist1
      args: -cell_simplex 0 -distribute 1 -interpolate {{none serial parallel}separate output}

  test:
    # the same as 4_tet_dist0 but test different initial orientations
    suffix: 4_tet_test_orient
    nsize: 2
    args: -dim 3 -distribute 0 -dm_plex_check_symmetry -dm_plex_check_skeleton unknown -dm_plex_check_geometry
    args: -rotate_interface_0 {{0 1 2 11 12 13}}
    args: -rotate_interface_1 {{0 1 2 11 12 13}}

  testset:
    requires: exodusii
    nsize: 2
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/TwoQuads.exo
    args: -cell_simplex 0 -dm_view ascii::ascii_info_detail -dm_plex_check_symmetry -dm_plex_check_skeleton tensor -dm_plex_check_geometry
    test:
      suffix: 5_dist0
      args: -distribute 0 -interpolate {{none serial}separate output}
    test:
      suffix: 5_dist1
      args: -distribute 1 -interpolate {{none serial parallel}separate output}

  testset:
    nsize: {{1 2 4}}
    args: -use_generator -dm_plex_check_symmetry -dm_plex_check_geometry
    args: -distribute -interpolate {{none serial parallel}}
    test:
      suffix: 6_tri
      requires: triangle
      args: -faces {{2,2  1,3  7,4}} -cell_simplex 1 -dm_plex_generator triangle -dm_plex_check_skeleton simplex
    test:
      suffix: 6_quad
      args: -faces {{2,2  1,3  7,4}} -cell_simplex 0 -dm_plex_check_skeleton tensor
    test:
      suffix: 6_tet
      requires: ctetgen
      args: -faces {{2,2,2  1,3,5  3,4,7}} -cell_simplex 1 -dm_plex_generator ctetgen -dm_plex_check_skeleton simplex
    test:
      TODO: fails due to wrong SF
      suffix: 6_hex
      args: -faces {{2,2,2  1,3,5  3,4,7}} -cell_simplex 0 -dm_plex_check_skeleton tensor

  testset:
    nsize: {{1 2 4 5}}
    args: -cell_simplex 0 -distribute -interpolate {{none serial parallel}} -dm_plex_check_symmetry -dm_plex_check_skeleton unknown -dm_plex_check_geometry
    test:
      TODO: fails due to wrong SF
      suffix: 7_exo
      requires: exodusii
      args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo
    test:
      TODO: fails due to wrong SF
      suffix: 7_hdf5
      requires: hdf5
      args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.h5 -dm_plex_create_from_hdf5_xdmf
TEST*/
