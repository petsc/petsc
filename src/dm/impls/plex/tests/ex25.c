static char help[] = "Test DMCreateLocalVector_Plex, DMPlexGetCellFields and DMPlexRestoreCellFields work properly for 0 fields/cells/DS dimension\n\n";
static char FILENAME[] = "ex25.c";

#include <petscdmplex.h>
#include <petscds.h>
#include <petscsnes.h>


typedef struct {
  PetscInt  test;
  PetscInt  dim;                          /* The topological mesh dimension */
  PetscInt  faces[3];                     /* Number of faces per dimension */
  PetscBool simplex;                      /* Use simplices or hexes */
  PetscBool interpolate;                  /* Interpolate mesh */
  char      filename[PETSC_MAX_PATH_LEN]; /* Import mesh from file */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->test         = 0;
  options->dim          = 3;
  options->simplex      = PETSC_TRUE;
  options->interpolate  = PETSC_FALSE;
  options->filename[0]  = '\0';
  ierr = PetscOptionsBegin(comm, "", "Zero-sized DMPlexGetCellFields Test Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBoundedInt("-test", "Test to run", FILENAME, options->test, &options->test, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim", "The topological mesh dimension", FILENAME, options->dim, &options->dim, NULL,1,3);CHKERRQ(ierr);
  if (options->dim > 3) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "dimension set to %d, must be <= 3", options->dim);
  ierr = PetscOptionsBool("-simplex", "Use simplices if true, otherwise hexes", FILENAME, options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Interpolate the mesh", FILENAME, options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", FILENAME, options->filename, options->filename, sizeof(options->filename), NULL);CHKERRQ(ierr);
  options->faces[0] = 1; options->faces[1] = 1; options->faces[2] = 1;
  dim = options->dim;
  ierr = PetscOptionsIntArray("-faces", "Number of faces per dimension", FILENAME, options->faces, &dim, NULL);CHKERRQ(ierr);
  if (dim) options->dim = dim;
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *options, DM *dm)
{
  PetscInt       dim          = options->dim;
  PetscInt      *faces        = options->faces;
  PetscBool      simplex      = options->simplex;
  PetscBool      interpolate  = options->interpolate;
  const char    *filename     = options->filename;
  size_t         len;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (len) {
    ierr = DMPlexCreateFromFile(comm, filename, interpolate, dm);CHKERRQ(ierr);
    ierr = DMGetDimension(*dm, &options->dim);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateBoxMesh(comm, dim, simplex, faces, NULL, NULL, NULL, interpolate, dm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* no discretization is given so DMGetNumFields yields 0 */
static PetscErrorCode test0(DM dm, AppCtx *options)
{
  Vec            locX;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetLocalVector(dm, &locX);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &locX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* no discretization is given so DMGetNumFields and PetscDSGetTotalDimension yield 0 */
static PetscErrorCode test1(DM dm, AppCtx *options)
{
  IS             cells;
  Vec            locX, locX_t, locA;
  PetscScalar    *u, *u_t, *a;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = ISCreateStride(PETSC_COMM_SELF, 0, 0, 1, &cells);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &locX);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &locX_t);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &locA);CHKERRQ(ierr);
  ierr = DMPlexGetCellFields(    dm, cells, locX, locX_t, locA, &u, &u_t, &a);CHKERRQ(ierr);
  ierr = DMPlexRestoreCellFields(dm, cells, locX, locX_t, locA, &u, &u_t, &a);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &locX);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &locX_t);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &locA);CHKERRQ(ierr);
  ierr = ISDestroy(&cells);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* no discretization is given so DMGetNumFields and PetscDSGetTotalDimension yield 0 */
static PetscErrorCode test2(DM dm, AppCtx *options)
{
  IS             cells;
  Vec            locX, locX_t, locA;
  PetscScalar    *u, *u_t, *a;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF, rank ? 0 : 1, 0, 1, &cells);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &locX);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &locX_t);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &locA);CHKERRQ(ierr);
  ierr = DMPlexGetCellFields(    dm, cells, locX, locX_t, locA, &u, &u_t, &a);CHKERRQ(ierr);
  ierr = DMPlexRestoreCellFields(dm, cells, locX, locX_t, locA, &u, &u_t, &a);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &locX);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &locX_t);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &locA);CHKERRQ(ierr);
  ierr = ISDestroy(&cells);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode test3(DM dm, AppCtx *options)
{
  PetscDS        ds;
  PetscFE        fe;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), options->dim, 1, options->simplex, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(ds, 0, (PetscObject)fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = test1(dm, options);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode test4(DM dm, AppCtx *options)
{
  PetscFE        fe;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), options->dim, 1, options->simplex, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = DMSetField(dm, 0, NULL, (PetscObject)fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = test2(dm, options);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode test5(DM dm, AppCtx *options)
{
  IS             cells;
  Vec            locX, locX_t, locA;
  PetscScalar    *u, *u_t, *a;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  locX_t = NULL;
  locA = NULL;
  ierr = ISCreateStride(PETSC_COMM_SELF, 0, 0, 1, &cells);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &locX);CHKERRQ(ierr);
  ierr = DMPlexGetCellFields(    dm, cells, locX, locX_t, locA, &u, &u_t, &a);CHKERRQ(ierr);
  ierr = DMPlexRestoreCellFields(dm, cells, locX, locX_t, locA, &u, &u_t, &a);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &locX);CHKERRQ(ierr);
  ierr = ISDestroy(&cells);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode test6(DM dm, AppCtx *options)
{
  IS             cells;
  Vec            locX, locX_t, locA;
  PetscScalar    *u, *u_t, *a;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank);CHKERRQ(ierr);
  locX_t = NULL;
  locA = NULL;
  ierr = ISCreateStride(PETSC_COMM_SELF, rank ? 0 : 1, 0, 1, &cells);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm, &locX);CHKERRQ(ierr);
  ierr = DMPlexGetCellFields(    dm, cells, locX, locX_t, locA, &u, &u_t, &a);CHKERRQ(ierr);
  ierr = DMPlexRestoreCellFields(dm, cells, locX, locX_t, locA, &u, &u_t, &a);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm, &locX);CHKERRQ(ierr);
  ierr = ISDestroy(&cells);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode test7(DM dm, AppCtx *options)
{
  PetscFE        fe;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), options->dim, 1, options->simplex, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = DMSetField(dm, 0, NULL, (PetscObject)fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = test5(dm, options);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode test8(DM dm, AppCtx *options)
{
  PetscFE        fe;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), options->dim, 1, options->simplex, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = DMSetField(dm, 0, NULL, (PetscObject)fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = test6(dm, options);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  MPI_Comm       comm;
  DM             dm;
  AppCtx         options;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &options);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &options, &dm);CHKERRQ(ierr);

  switch (options.test) {
    case 0: ierr = test0(dm, &options);CHKERRQ(ierr); break;
    case 1: ierr = test1(dm, &options);CHKERRQ(ierr); break;
    case 2: ierr = test2(dm, &options);CHKERRQ(ierr); break;
    case 3: ierr = test3(dm, &options);CHKERRQ(ierr); break;
    case 4: ierr = test4(dm, &options);CHKERRQ(ierr); break;
    case 5: ierr = test5(dm, &options);CHKERRQ(ierr); break;
    case 6: ierr = test6(dm, &options);CHKERRQ(ierr); break;
    case 7: ierr = test7(dm, &options);CHKERRQ(ierr); break;
    case 8: ierr = test8(dm, &options);CHKERRQ(ierr); break;
    default: SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "No such test: %D", options.test);
  }

  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    requires: ctetgen
    args: -test 0
  test:
    suffix: 1
    requires: ctetgen
    args: -test 1
  test:
    suffix: 2
    requires: ctetgen
    args: -test 2
  test:
    suffix: 3
    requires: ctetgen
    args: -test 3
  test:
    suffix: 4
    requires: ctetgen
    args: -test 4
  test:
    suffix: 5
    requires: ctetgen
    args: -test 5
  test:
    suffix: 6
    requires: ctetgen
    args: -test 6
  test:
    suffix: 7
    requires: ctetgen
    args: -test 7
  test:
    suffix: 8
    requires: ctetgen
    args: -test 8
  test:
    suffix: 9
    requires: ctetgen
    nsize: 2
    args: -test 1
  test:
    suffix: 10
    requires: ctetgen
    nsize: 2
    args: -test 2
  test:
    suffix: 11
    requires: ctetgen
    nsize: 2
    args: -test 3
  test:
    suffix: 12
    requires: ctetgen
    nsize: 2
    args: -test 4

TEST*/

