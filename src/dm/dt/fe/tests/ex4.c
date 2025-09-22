static const char help[] = "Tests DMCreateMassMatrix and DMCreateMassMatrixLumped";

#include <petscdmplex.h>
#include <petscfe.h>
#include <petscds.h>

static void volume(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar obj[])
{
  obj[0] = 1.0;
}

static PetscErrorCode SetupDiscretization(DM dm)
{
  PetscFE   fe0, fe1, fe2;
  DM        plex;
  PetscInt  dim;
  PetscBool simplex;
  PetscDS   ds;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMConvert(dm, DMPLEX, &plex));
  PetscCall(DMPlexIsSimplex(plex, &simplex));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &simplex, 1, MPI_C_BOOL, MPI_LOR, PetscObjectComm((PetscObject)dm)));
  PetscCall(DMDestroy(&plex));

  /* Create finite element */
  PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject)dm), dim, 1, simplex, "f0_", -1, &fe0));
  PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject)dm), dim, 2, simplex, "f1_", -1, &fe1));
  PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject)dm), dim, dim, simplex, "f2_", -1, &fe2));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe0));
  PetscCall(DMSetField(dm, 1, NULL, (PetscObject)fe1));
  PetscCall(DMSetField(dm, 2, NULL, (PetscObject)fe2));
  PetscCall(DMCreateDS(dm));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetObjective(ds, 0, volume));
  PetscCall(PetscDSSetObjective(ds, 1, volume));
  PetscCall(PetscDSSetObjective(ds, 2, volume));
  PetscCall(PetscFEDestroy(&fe0));
  PetscCall(PetscFEDestroy(&fe1));
  PetscCall(PetscFEDestroy(&fe2));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define CheckVals(a, b, rtol, atol, msg) \
  do { \
    if (!PetscIsCloseAtTolScalar(a, b, rtol, atol)) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s: %g (%g - %g)\n", msg, (double)PetscAbsScalar(a - b), (double)PetscAbsScalar(a), (double)PetscAbsScalar(b))); \
  } while (0)

int main(int argc, char **argv)
{
  DM          dm;
  Mat         M;
  Vec         llM, lM, ones, work;
  PetscReal   rtol = PETSC_SMALL, atol = 0.0;
  PetscScalar vals[6];
  PetscInt    dim;
  PetscBool   amr = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-amr", &amr, NULL));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMGetDimension(dm, &dim));
  if (amr) {
    DM dmConv;

    PetscCall(DMConvert(dm, dim == 2 ? DMP4EST : DMP8EST, &dmConv));
    PetscCall(DMDestroy(&dm));
    dm = dmConv;
    PetscCall(DMSetFromOptions(dm));
    PetscCall(DMSetUp(dm));
  }
  PetscCall(DMViewFromOptions(dm, NULL, "-mesh_view"));
  PetscCall(SetupDiscretization(dm));
  PetscCall(DMGetGlobalVector(dm, &ones));
  PetscCall(DMGetGlobalVector(dm, &work));
  PetscCall(VecSet(ones, 1.0));
  PetscCall(DMCreateMassMatrixLumped(dm, &llM, &lM));
  PetscCall(VecViewFromOptions(llM, NULL, "-local_lumped_mass_view"));
  PetscCall(VecViewFromOptions(lM, NULL, "-lumped_mass_view"));
  PetscCall(DMCreateMassMatrix(dm, NULL, &M));
  PetscCall(MatViewFromOptions(M, NULL, "-mass_view"));
  PetscCall(MatMult(M, ones, work));
  PetscCall(VecViewFromOptions(work, NULL, "-mass_rowsum_view"));
  PetscCall(VecSum(work, &vals[3]));
  PetscCall(VecSet(work, 0.0));
  PetscCall(DMLocalToGlobal(dm, llM, ADD_VALUES, work));
  PetscCall(VecSum(work, &vals[4]));
  PetscCall(VecSum(lM, &vals[5]));
  PetscCall(DMPlexComputeIntegralFEM(dm, ones, vals, NULL));
  CheckVals(vals[0], vals[1], rtol, atol, "Error volume");
  CheckVals((3 + dim) * vals[0], vals[3], rtol, atol, "Error mass");
  CheckVals((3 + dim) * vals[0], vals[4], rtol, atol, "Error local lumped mass");
  CheckVals((3 + dim) * vals[0], vals[5], rtol, atol, "Error lumped mass");
  PetscCall(MatDestroy(&M));
  PetscCall(VecDestroy(&lM));
  PetscCall(VecDestroy(&llM));
  PetscCall(DMRestoreGlobalVector(dm, &work));
  PetscCall(DMRestoreGlobalVector(dm, &ones));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    output_file: output/empty.out
    nsize: {{1 2}}
    args: -dm_plex_box_faces 3,3,3 -dm_plex_dim {{2 3}} -dm_plex_simplex 0 -f0_petscspace_degree {{0 1}} -f1_petscspace_degree {{0 1}} -f2_petscspace_degree {{0 1}} -petscpartitioner_type simple

  test:
    TODO: broken
    suffix: 0_amr
    output_file: output/empty.out
    requires: p4est
    nsize: {{1 2}}
    args: -dm_plex_dim {{2 3}} -dm_plex_simplex 0 -f0_petscspace_degree {{0 1}} -f1_petscspace_degree {{0 1}} -f2_petscspace_degree {{0 1}} -amr -dm_p4est_refine_pattern hash -dm_forest_initial_refinement 0 -dm_forest_maximum_refinement 2

  test:
    suffix: 1
    output_file: output/empty.out
    requires: triangle
    nsize: {{1 2}}
    args: -dm_plex_box_faces 3,3 -dm_plex_dim 2 -dm_plex_simplex 1 -f0_petscspace_degree {{0 1}} -f1_petscspace_degree {{0 1}} -f2_petscspace_degree {{0 1}} -petscpartitioner_type simple

  test:
    suffix: 2
    output_file: output/empty.out
    requires: ctetgen
    nsize: {{1 2}}
    args: -dm_plex_box_faces 3,3,3 -dm_plex_dim 3 -dm_plex_simplex 1 -f0_petscspace_degree {{0 1}} -f1_petscspace_degree {{0 1}} -f2_petscspace_degree {{0 1}} -petscpartitioner_type simple

TEST*/
