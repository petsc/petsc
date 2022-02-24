static char help[] = "Test DMCreateLocalVector_Plex, DMPlexGetCellFields and DMPlexRestoreCellFields work properly for 0 fields/cells/DS dimension\n\n";
static char FILENAME[] = "ex25.c";

#include <petscdmplex.h>
#include <petscds.h>
#include <petscsnes.h>

typedef struct {
  PetscInt test;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->test = 0;
  ierr = PetscOptionsBegin(comm, "", "Zero-sized DMPlexGetCellFields Test Options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsBoundedInt("-test", "Test to run", FILENAME, options->test, &options->test, NULL,0));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *options, DM *dm)
{
  PetscFunctionBegin;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

/* no discretization is given so DMGetNumFields yields 0 */
static PetscErrorCode test0(DM dm, AppCtx *options)
{
  Vec            locX;

  PetscFunctionBegin;
  CHKERRQ(DMGetLocalVector(dm, &locX));
  CHKERRQ(DMRestoreLocalVector(dm, &locX));
  PetscFunctionReturn(0);
}

/* no discretization is given so DMGetNumFields and PetscDSGetTotalDimension yield 0 */
static PetscErrorCode test1(DM dm, AppCtx *options)
{
  IS             cells;
  Vec            locX, locX_t, locA;
  PetscScalar    *u, *u_t, *a;

  PetscFunctionBegin;
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF, 0, 0, 1, &cells));
  CHKERRQ(DMGetLocalVector(dm, &locX));
  CHKERRQ(DMGetLocalVector(dm, &locX_t));
  CHKERRQ(DMGetLocalVector(dm, &locA));
  CHKERRQ(DMPlexGetCellFields(    dm, cells, locX, locX_t, locA, &u, &u_t, &a));
  CHKERRQ(DMPlexRestoreCellFields(dm, cells, locX, locX_t, locA, &u, &u_t, &a));
  CHKERRQ(DMRestoreLocalVector(dm, &locX));
  CHKERRQ(DMRestoreLocalVector(dm, &locX_t));
  CHKERRQ(DMRestoreLocalVector(dm, &locA));
  CHKERRQ(ISDestroy(&cells));
  PetscFunctionReturn(0);
}

/* no discretization is given so DMGetNumFields and PetscDSGetTotalDimension yield 0 */
static PetscErrorCode test2(DM dm, AppCtx *options)
{
  IS             cells;
  Vec            locX, locX_t, locA;
  PetscScalar    *u, *u_t, *a;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF, rank ? 0 : 1, 0, 1, &cells));
  CHKERRQ(DMGetLocalVector(dm, &locX));
  CHKERRQ(DMGetLocalVector(dm, &locX_t));
  CHKERRQ(DMGetLocalVector(dm, &locA));
  CHKERRQ(DMPlexGetCellFields(    dm, cells, locX, locX_t, locA, &u, &u_t, &a));
  CHKERRQ(DMPlexRestoreCellFields(dm, cells, locX, locX_t, locA, &u, &u_t, &a));
  CHKERRQ(DMRestoreLocalVector(dm, &locX));
  CHKERRQ(DMRestoreLocalVector(dm, &locX_t));
  CHKERRQ(DMRestoreLocalVector(dm, &locA));
  CHKERRQ(ISDestroy(&cells));
  PetscFunctionReturn(0);
}

static PetscErrorCode test3(DM dm, AppCtx *options)
{
  PetscDS        ds;
  PetscFE        fe;
  PetscInt       dim;
  PetscBool      simplex;

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexIsSimplex(dm, &simplex));
  CHKERRQ(DMGetDS(dm, &ds));
  CHKERRQ(PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, simplex, NULL, -1, &fe));
  CHKERRQ(PetscDSSetDiscretization(ds, 0, (PetscObject)fe));
  CHKERRQ(PetscFEDestroy(&fe));
  CHKERRQ(test1(dm, options));
  PetscFunctionReturn(0);
}

static PetscErrorCode test4(DM dm, AppCtx *options)
{
  PetscFE        fe;
  PetscInt       dim;
  PetscBool      simplex;

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexIsSimplex(dm, &simplex));
  CHKERRQ(PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, simplex, NULL, -1, &fe));
  CHKERRQ(DMSetField(dm, 0, NULL, (PetscObject)fe));
  CHKERRQ(PetscFEDestroy(&fe));
  CHKERRQ(DMCreateDS(dm));
  CHKERRQ(test2(dm, options));
  PetscFunctionReturn(0);
}

static PetscErrorCode test5(DM dm, AppCtx *options)
{
  IS             cells;
  Vec            locX, locX_t, locA;
  PetscScalar    *u, *u_t, *a;

  PetscFunctionBegin;
  locX_t = NULL;
  locA = NULL;
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF, 0, 0, 1, &cells));
  CHKERRQ(DMGetLocalVector(dm, &locX));
  CHKERRQ(DMPlexGetCellFields(    dm, cells, locX, locX_t, locA, &u, &u_t, &a));
  CHKERRQ(DMPlexRestoreCellFields(dm, cells, locX, locX_t, locA, &u, &u_t, &a));
  CHKERRQ(DMRestoreLocalVector(dm, &locX));
  CHKERRQ(ISDestroy(&cells));
  PetscFunctionReturn(0);
}

static PetscErrorCode test6(DM dm, AppCtx *options)
{
  IS             cells;
  Vec            locX, locX_t, locA;
  PetscScalar    *u, *u_t, *a;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  CHKERRMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  locX_t = NULL;
  locA = NULL;
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF, rank ? 0 : 1, 0, 1, &cells));
  CHKERRQ(DMGetLocalVector(dm, &locX));
  CHKERRQ(DMPlexGetCellFields(    dm, cells, locX, locX_t, locA, &u, &u_t, &a));
  CHKERRQ(DMPlexRestoreCellFields(dm, cells, locX, locX_t, locA, &u, &u_t, &a));
  CHKERRQ(DMRestoreLocalVector(dm, &locX));
  CHKERRQ(ISDestroy(&cells));
  PetscFunctionReturn(0);
}

static PetscErrorCode test7(DM dm, AppCtx *options)
{
  PetscFE        fe;
  PetscInt       dim;
  PetscBool      simplex;

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexIsSimplex(dm, &simplex));
  CHKERRQ(PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, simplex, NULL, -1, &fe));
  CHKERRQ(DMSetField(dm, 0, NULL, (PetscObject)fe));
  CHKERRQ(PetscFEDestroy(&fe));
  CHKERRQ(DMCreateDS(dm));
  CHKERRQ(test5(dm, options));
  PetscFunctionReturn(0);
}

static PetscErrorCode test8(DM dm, AppCtx *options)
{
  PetscFE        fe;
  PetscInt       dim;
  PetscBool      simplex;

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexIsSimplex(dm, &simplex));
  CHKERRQ(PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, simplex, NULL, -1, &fe));
  CHKERRQ(DMSetField(dm, 0, NULL, (PetscObject)fe));
  CHKERRQ(PetscFEDestroy(&fe));
  CHKERRQ(DMCreateDS(dm));
  CHKERRQ(test6(dm, options));
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
  CHKERRQ(ProcessOptions(comm, &options));
  CHKERRQ(CreateMesh(comm, &options, &dm));

  switch (options.test) {
    case 0: CHKERRQ(test0(dm, &options)); break;
    case 1: CHKERRQ(test1(dm, &options)); break;
    case 2: CHKERRQ(test2(dm, &options)); break;
    case 3: CHKERRQ(test3(dm, &options)); break;
    case 4: CHKERRQ(test4(dm, &options)); break;
    case 5: CHKERRQ(test5(dm, &options)); break;
    case 6: CHKERRQ(test6(dm, &options)); break;
    case 7: CHKERRQ(test7(dm, &options)); break;
    case 8: CHKERRQ(test8(dm, &options)); break;
    default: SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "No such test: %D", options.test);
  }

  CHKERRQ(DMDestroy(&dm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  testset:
    args: -dm_plex_dim 3 -dm_plex_interpolate 0

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
