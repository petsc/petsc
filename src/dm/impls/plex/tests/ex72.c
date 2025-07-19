static char help[] = "Tests for geometry models\n\n";

#include <petscdmplex.h>
#include <petscds.h>

typedef struct {
  PetscReal exactVol; // The exact volume of the shape
  PetscReal volTol;   // The relative tolerance for checking the volume
} AppCtx;

static void identity(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 1.0;
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  options->exactVol = 4. * PETSC_PI / 3.;
  options->volTol   = PETSC_SMALL;

  PetscFunctionBeginUser;
  PetscOptionsBegin(comm, "", "Geometry Model Test Options", "DMPLEX");
  PetscCall(PetscOptionsReal("-exact_vol", "Exact volume of the shape", __FILE__, options->exactVol, &options->exactVol, NULL));
  PetscCall(PetscOptionsReal("-vol_tol", "Relative tolerance for checking the volume", __FILE__, options->volTol, &options->volTol, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateDiscretization(DM dm)
{
  PetscFE        fe;
  PetscDS        ds;
  DMPolytopeType ct;
  PetscInt       dim, cStart;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, NULL, -1, &fe));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetObjective(ds, 0, identity));
  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM          dm;
  Vec         u;
  PetscScalar volume;
  AppCtx      user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
  PetscCall(CreateDiscretization(dm));
  PetscCall(DMGetGlobalVector(dm, &u));
  PetscCall(VecSet(u, 0.));
  PetscCall(DMPlexComputeIntegralFEM(dm, u, &volume, NULL));
  PetscCall(DMRestoreGlobalVector(dm, &u));
  PetscCheck(PetscAbsScalar((volume - user.exactVol) / user.exactVol) < user.volTol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Invalid volume %g != %g", (double)PetscRealPart(volume), (double)user.exactVol);
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # -dm_refine 6 -vol_tol 6e-4 works
  test:
    suffix: ball_0
    requires: ctetgen
    args: -dm_plex_dim 3 -dm_plex_shape ball -dm_refine 3 -dm_geom_model ball -vol_tol 5e-2
    output_file: output/empty.out

  # -dm_refine 4 -vol_tol 2e-3 works
  test:
    suffix: cylinder_0
    args: -dm_plex_dim 3 -dm_plex_shape cylinder -dm_plex_cylinder_num_refine 1 \
          -dm_refine 1 -dm_geom_model cylinder -exact_vol 3.141592653589 -vol_tol 1e-1
    output_file: output/empty.out

TEST*/
