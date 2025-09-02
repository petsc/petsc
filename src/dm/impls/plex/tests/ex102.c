static char help[] = "Test degenerate near null space";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscbag.h>
#include <petscconvest.h>

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupBoundaries(DM dm)
{
  DMLabel  label;
  PetscInt id;
  PetscInt dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinateDim(dm, &dim));
  PetscCall(DMGetLabel(dm, "Face Sets", &label));
  PetscCheck(label, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_INCOMP, "Must have face sets label");

  if (dim == 2) {
    PetscInt cmp, cmps_y[] = {0, 1};

    cmp = 0;
    id  = 4;
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "left", label, 1, &id, 0, 1, &cmp, NULL, NULL, NULL, NULL));
    cmp = 0;
    id  = 2;
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "right", label, 1, &id, 0, 1, &cmp, NULL, NULL, NULL, NULL));
    id = 1;
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "bottom", label, 1, &id, 0, 2, cmps_y, NULL, NULL, NULL, NULL));
  } else if (dim == 3) {
    PetscInt cmps_xy[] = {0, 1};
    PetscInt cmps_z[]  = {0, 1, 2};

    id = 6;
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "left", label, 1, &id, 0, 2, cmps_xy, NULL, NULL, NULL, NULL));
    id = 5;
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "right", label, 1, &id, 0, 2, cmps_xy, NULL, NULL, NULL, NULL));
    id = 2;
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "bottom", label, 1, &id, 0, 3, cmps_z, NULL, NULL, NULL, NULL));
    id = 3;
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "front", label, 1, &id, 0, 2, cmps_xy, NULL, NULL, NULL, NULL));
    id = 4;
    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "back", label, 1, &id, 0, 2, cmps_xy, NULL, NULL, NULL, NULL));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode SetupFE(DM dm, const char name[])
{
  PetscFE        fe;
  char           prefix[PETSC_MAX_PATH_LEN];
  DMPolytopeType ct;
  PetscInt       dim, cStart;

  PetscFunctionBegin;
  /* Create finite element */
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  PetscCall(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name));
  PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, dim, ct, name ? prefix : NULL, -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, name));
  /* Set discretization and boundary conditions for each mesh */
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(SetupBoundaries(dm));
  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MakeNullSpaceRigidBody(DM dm)
{
  Mat          A;
  MatNullSpace null_space;

  PetscFunctionBegin;
  /* Create null space and set onto matrix */
  PetscCall(DMCreateMatrix(dm, &A));
  PetscCall(DMPlexCreateRigidBody(dm, 0, &null_space));
  PetscCall(MatSetNearNullSpace(A, null_space));
  PetscCall(MatNullSpaceView(null_space, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatNullSpaceDestroy(&null_space));
  PetscCall(MatDestroy(&A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM dm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
  PetscCall(SetupFE(dm, "displacement"));
  PetscCall(MakeNullSpaceRigidBody(dm));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 3d_q1
    args: -dm_plex_box_faces 1,1,2 -dm_plex_simplex 0 -dm_plex_dim 3 -displacement_petscspace_degree 1

  test:
    suffix: 3d_q2
    args: -dm_plex_box_faces 1,1,2 -dm_plex_simplex 0 -dm_plex_dim 3 -displacement_petscspace_degree 2

  test:
    suffix: 2d_q1
    args: -dm_plex_box_faces 1,2 -dm_plex_simplex 0 -dm_plex_dim 2 -displacement_petscspace_degree 1

  test:
    suffix: 2d_q2
    args: -dm_plex_box_faces 1,2 -dm_plex_simplex 0 -dm_plex_dim 2 -displacement_petscspace_degree 2

TEST*/
