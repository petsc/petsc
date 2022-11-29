static char help[]     = "Test DMCreateLocalVector_Plex, DMPlexGetCellFields and DMPlexRestoreCellFields work properly for 0 fields/cells/DS dimension\n\n";
static char FILENAME[] = "ex25.c";

#include <petscdmplex.h>
#include <petscds.h>
#include <petscsnes.h>

typedef struct {
  PetscInt test;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBegin;
  options->test = 0;
  PetscOptionsBegin(comm, "", "Zero-sized DMPlexGetCellFields Test Options", "DMPLEX");
  PetscCall(PetscOptionsBoundedInt("-test", "Test to run", FILENAME, options->test, &options->test, NULL, 0));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *options, DM *dm)
{
  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

/* no discretization is given so DMGetNumFields yields 0 */
static PetscErrorCode test0(DM dm, AppCtx *options)
{
  Vec locX;

  PetscFunctionBegin;
  PetscCall(DMGetLocalVector(dm, &locX));
  PetscCall(DMRestoreLocalVector(dm, &locX));
  PetscFunctionReturn(0);
}

/* no discretization is given so DMGetNumFields and PetscDSGetTotalDimension yield 0 */
static PetscErrorCode test1(DM dm, AppCtx *options)
{
  IS           cells;
  Vec          locX, locX_t, locA;
  PetscScalar *u, *u_t, *a;

  PetscFunctionBegin;
  PetscCall(ISCreateStride(PETSC_COMM_SELF, 0, 0, 1, &cells));
  PetscCall(DMGetLocalVector(dm, &locX));
  PetscCall(DMGetLocalVector(dm, &locX_t));
  PetscCall(DMGetLocalVector(dm, &locA));
  PetscCall(DMPlexGetCellFields(dm, cells, locX, locX_t, locA, &u, &u_t, &a));
  PetscCall(DMPlexRestoreCellFields(dm, cells, locX, locX_t, locA, &u, &u_t, &a));
  PetscCall(DMRestoreLocalVector(dm, &locX));
  PetscCall(DMRestoreLocalVector(dm, &locX_t));
  PetscCall(DMRestoreLocalVector(dm, &locA));
  PetscCall(ISDestroy(&cells));
  PetscFunctionReturn(0);
}

/* no discretization is given so DMGetNumFields and PetscDSGetTotalDimension yield 0 */
static PetscErrorCode test2(DM dm, AppCtx *options)
{
  IS           cells;
  Vec          locX, locX_t, locA;
  PetscScalar *u, *u_t, *a;
  PetscMPIInt  rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, rank ? 0 : 1, 0, 1, &cells));
  PetscCall(DMGetLocalVector(dm, &locX));
  PetscCall(DMGetLocalVector(dm, &locX_t));
  PetscCall(DMGetLocalVector(dm, &locA));
  PetscCall(DMPlexGetCellFields(dm, cells, locX, locX_t, locA, &u, &u_t, &a));
  PetscCall(DMPlexRestoreCellFields(dm, cells, locX, locX_t, locA, &u, &u_t, &a));
  PetscCall(DMRestoreLocalVector(dm, &locX));
  PetscCall(DMRestoreLocalVector(dm, &locX_t));
  PetscCall(DMRestoreLocalVector(dm, &locA));
  PetscCall(ISDestroy(&cells));
  PetscFunctionReturn(0);
}

static PetscErrorCode test3(DM dm, AppCtx *options)
{
  PetscDS   ds;
  PetscFE   fe;
  PetscInt  dim;
  PetscBool simplex;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject)dm), dim, 1, simplex, NULL, -1, &fe));
  PetscCall(PetscDSSetDiscretization(ds, 0, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(test1(dm, options));
  PetscFunctionReturn(0);
}

static PetscErrorCode test4(DM dm, AppCtx *options)
{
  PetscFE   fe;
  PetscInt  dim;
  PetscBool simplex;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject)dm), dim, 1, simplex, NULL, -1, &fe));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(test2(dm, options));
  PetscFunctionReturn(0);
}

static PetscErrorCode test5(DM dm, AppCtx *options)
{
  IS           cells;
  Vec          locX, locX_t, locA;
  PetscScalar *u, *u_t, *a;

  PetscFunctionBegin;
  locX_t = NULL;
  locA   = NULL;
  PetscCall(ISCreateStride(PETSC_COMM_SELF, 0, 0, 1, &cells));
  PetscCall(DMGetLocalVector(dm, &locX));
  PetscCall(DMPlexGetCellFields(dm, cells, locX, locX_t, locA, &u, &u_t, &a));
  PetscCall(DMPlexRestoreCellFields(dm, cells, locX, locX_t, locA, &u, &u_t, &a));
  PetscCall(DMRestoreLocalVector(dm, &locX));
  PetscCall(ISDestroy(&cells));
  PetscFunctionReturn(0);
}

static PetscErrorCode test6(DM dm, AppCtx *options)
{
  IS           cells;
  Vec          locX, locX_t, locA;
  PetscScalar *u, *u_t, *a;
  PetscMPIInt  rank;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)dm), &rank));
  locX_t = NULL;
  locA   = NULL;
  PetscCall(ISCreateStride(PETSC_COMM_SELF, rank ? 0 : 1, 0, 1, &cells));
  PetscCall(DMGetLocalVector(dm, &locX));
  PetscCall(DMPlexGetCellFields(dm, cells, locX, locX_t, locA, &u, &u_t, &a));
  PetscCall(DMPlexRestoreCellFields(dm, cells, locX, locX_t, locA, &u, &u_t, &a));
  PetscCall(DMRestoreLocalVector(dm, &locX));
  PetscCall(ISDestroy(&cells));
  PetscFunctionReturn(0);
}

static PetscErrorCode test7(DM dm, AppCtx *options)
{
  PetscFE   fe;
  PetscInt  dim;
  PetscBool simplex;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject)dm), dim, 1, simplex, NULL, -1, &fe));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(test5(dm, options));
  PetscFunctionReturn(0);
}

static PetscErrorCode test8(DM dm, AppCtx *options)
{
  PetscFE   fe;
  PetscInt  dim;
  PetscBool simplex;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject)dm), dim, 1, simplex, NULL, -1, &fe));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(test6(dm, options));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  MPI_Comm comm;
  DM       dm;
  AppCtx   options;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(ProcessOptions(comm, &options));
  PetscCall(CreateMesh(comm, &options, &dm));

  switch (options.test) {
  case 0:
    PetscCall(test0(dm, &options));
    break;
  case 1:
    PetscCall(test1(dm, &options));
    break;
  case 2:
    PetscCall(test2(dm, &options));
    break;
  case 3:
    PetscCall(test3(dm, &options));
    break;
  case 4:
    PetscCall(test4(dm, &options));
    break;
  case 5:
    PetscCall(test5(dm, &options));
    break;
  case 6:
    PetscCall(test6(dm, &options));
    break;
  case 7:
    PetscCall(test7(dm, &options));
    break;
  case 8:
    PetscCall(test8(dm, &options));
    break;
  default:
    SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "No such test: %" PetscInt_FMT, options.test);
  }

  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
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
