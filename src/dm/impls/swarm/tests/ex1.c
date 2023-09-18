static const char help[] = "Test initialization and migration with swarm.\n";

#include <petscdmplex.h>
#include <petscdmswarm.h>

typedef struct {
  PetscReal L[3]; /* Dimensions of the mesh bounding box */
  PetscBool setClosurePermutation;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  PetscOptionsBegin(comm, "", "Swarm configuration options", "DMSWARM");
  options->setClosurePermutation = PETSC_FALSE;
  PetscCall(PetscOptionsBool("-set_closure_permutation", "Set the closure permutation to tensor ordering", NULL, options->setClosurePermutation, &options->setClosurePermutation, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscReal low[3], high[3];
  PetscInt  cdim, d;

  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  if (user->setClosurePermutation) {
    DM cdm;

    // -- Set tensor permutation
    PetscCall(DMGetCoordinateDM(*dm, &cdm));
    PetscCall(DMPlexSetClosurePermutationTensor(*dm, PETSC_DETERMINE, NULL));
    PetscCall(DMPlexSetClosurePermutationTensor(cdm, PETSC_DETERMINE, NULL));
  }
  PetscCall(DMGetCoordinateDim(*dm, &cdim));
  PetscCall(DMGetBoundingBox(*dm, low, high));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "dim: %" PetscInt_FMT "\n", cdim));
  for (d = 0; d < cdim; ++d) user->L[d] = high[d] - low[d];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  This function initializes all particles on rank 0.
  They are sent to other ranks to test migration across non nearest neighbors
*/
static PetscErrorCode CreateSwarm(DM dm, DM *sw, AppCtx *user)
{
  PetscInt    particleInitSize = 10;
  PetscReal  *coords, upper[3], lower[3];
  PetscInt   *cellid, Np, dim;
  PetscMPIInt rank, size;
  MPI_Comm    comm;

  PetscFunctionBegin;
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(DMGetBoundingBox(dm, lower, upper));
  PetscCall(DMCreate(PETSC_COMM_WORLD, sw));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));
  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*sw, dm));
  PetscCall(DMSwarmFinalizeFieldRegister(*sw));
  PetscCall(DMSwarmSetLocalSizes(*sw, rank == 0 ? particleInitSize : 0, 0));
  PetscCall(DMSetFromOptions(*sw));
  PetscCall(DMSwarmGetField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid));
  PetscCall(DMSwarmGetLocalSize(*sw, &Np));
  for (PetscInt p = 0; p < Np; ++p) {
    for (PetscInt d = 0; d < dim; ++d) { coords[p * dim + d] = 0.5 * (upper[d] - lower[d]); }
    coords[p * dim + 1] = (upper[1] - lower[1]) / particleInitSize * p + lower[1];
    cellid[p]           = 0;
  }
  PetscCall(DMSwarmRestoreField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid));
  PetscCall(DMViewFromOptions(*sw, NULL, "-sw_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Configure the swarm on rank 0 with all particles
  located there, then migrate where they need to be
*/
static PetscErrorCode CheckMigrate(DM sw)
{
  Vec       preMigrate, postMigrate, tmp;
  PetscInt  preSize, postSize;
  PetscReal prenorm, postnorm;

  PetscFunctionBeginUser;
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, DMSwarmPICField_coor, &tmp));
  PetscCall(VecDuplicate(tmp, &preMigrate));
  PetscCall(VecCopy(tmp, preMigrate));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, DMSwarmPICField_coor, &tmp));
  PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, DMSwarmPICField_coor, &tmp));
  PetscCall(VecDuplicate(tmp, &postMigrate));
  PetscCall(VecCopy(tmp, postMigrate));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, DMSwarmPICField_coor, &tmp));
  PetscCall(VecGetSize(preMigrate, &preSize));
  PetscCall(VecGetSize(postMigrate, &postSize));
  PetscCheck(preSize == postSize, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Particles either lost or duplicated. Pre migrate global size %" PetscInt_FMT " != Post migrate size %" PetscInt_FMT "", preSize, postSize);
  PetscCall(VecNorm(preMigrate, NORM_2, &prenorm));
  PetscCall(VecNorm(postMigrate, NORM_2, &postnorm));
  PetscCheck(PetscAbsReal(prenorm - postnorm) < 100. * PETSC_MACHINE_EPSILON, PETSC_COMM_SELF, PETSC_ERR_COR, "Particle coordinates corrupted in migrate with abs(norm(pre) - norm(post)) = %.16g", PetscAbsReal(prenorm - postnorm));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Migrate check passes.\n"));
  PetscCall(VecDestroy(&preMigrate));
  PetscCall(VecDestroy(&postMigrate));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Checks added points persist on migrate
*/
static PetscErrorCode CheckPointInsertion(DM sw)
{
  PetscInt    Np_pre, Np_post;
  PetscMPIInt rank, size;
  MPI_Comm    comm;

  PetscFunctionBeginUser;
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscPrintf(comm, "Basic point insertion check...\n"));
  PetscCall(DMSwarmGetSize(sw, &Np_pre));
  if (rank == 0) PetscCall(DMSwarmAddPoint(sw));
  PetscCall(DMSwarmGetSize(sw, &Np_post));
  PetscCheck(Np_post == (Np_pre + 1), PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Particle insertion failed. Global size pre insertion: %" PetscInt_FMT " global size post insertion: %" PetscInt_FMT, Np_pre, Np_post);
  PetscCall(CheckMigrate(sw));
  PetscCall(PetscPrintf(comm, "Basic point insertion check passes.\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Checks tie breaking works properly when a particle
  is located at a shared boundary. The higher rank should
  recieve the particle while the lower rank deletes it.

  TODO: Currently only works for 2 procs.
*/
static PetscErrorCode CheckPointInsertion_Boundary(DM sw)
{
  PetscInt    Np_loc_pre, Np_loc_post, dim;
  PetscMPIInt rank, size;
  PetscReal   lbox_low[3], lbox_high[3], gbox_low[3], gbox_high[3];
  MPI_Comm    comm;
  DM          cdm;

  PetscFunctionBeginUser;
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(PetscPrintf(comm, "Rank boundary point insertion check...\n"));
  PetscCall(DMSwarmGetCellDM(sw, &cdm));
  PetscCall(DMGetDimension(cdm, &dim));
  PetscCall(DMGetBoundingBox(cdm, gbox_low, gbox_high));
  if (rank == 0) {
    PetscReal *coords;
    PetscInt   adjacentdim = 0, Np;

    PetscCall(DMGetLocalBoundingBox(cdm, lbox_low, lbox_high));
    // find a face that belongs to the neighbor.
    for (PetscInt d = 0; d < dim; ++d) {
      if ((gbox_high[d] - lbox_high[d]) != 0.) adjacentdim = d;
    }
    PetscCall(DMSwarmAddPoint(sw));
    PetscCall(DMSwarmGetLocalSize(sw, &Np));
    PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
    for (PetscInt d = 0; d < dim; ++d) coords[(Np - 1) * dim + d] = 0.5 * (lbox_high[d] - lbox_low[d]);
    coords[(Np - 1) * dim + adjacentdim] = lbox_high[adjacentdim];
    PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  }
  PetscCall(DMSwarmGetLocalSize(sw, &Np_loc_pre));
  PetscCall(CheckMigrate(sw));
  PetscCall(DMSwarmGetLocalSize(sw, &Np_loc_post));
  if (rank == 0) PetscCheck(Np_loc_pre == (Np_loc_post + 1), PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Migration tie breaking failed on rank %d. Particle on boundary not sent.", rank);
  if (rank == 1) PetscCheck(Np_loc_pre == (Np_loc_post - 1), PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Migration tie breaking failed on rank %d. Particle on boundary not recieved.", rank);
  PetscCall(PetscPrintf(comm, "Rank boundary point insertion check passes.\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM       dm, sw;
  MPI_Comm comm;
  AppCtx   user;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(ProcessOptions(comm, &user));
  PetscCall(CreateMesh(comm, &dm, &user));
  PetscCall(CreateSwarm(dm, &sw, &user));
  PetscCall(CheckMigrate(sw));
  PetscCall(CheckPointInsertion(sw));
  PetscCall(CheckPointInsertion_Boundary(sw));
  PetscCall(DMDestroy(&sw));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # Swarm does not handle complex or quad
  build:
    requires: !complex double
  # swarm_migrate_hash and swarm_migrate_scan test swarm migration against point location types
  # with a distributed mesh where ranks overlap by 1. Points in the shared boundary should
  # be sent to the process which has the highest rank that has that portion of the domain.
  test:
    suffix: swarm_migrate_hash
    nsize: 2
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10,10\
          -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10. -dm_plex_box_bd none,none,none\
          -dm_plex_hash_location true
    filter: grep -v marker | grep -v atomic | grep -v usage
  test:
    suffix: swarm_migrate_hash_tensor_permutation
    nsize: 2
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10,10\
          -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10. -dm_plex_box_bd none,none,none\
          -dm_plex_hash_location true -set_closure_permutation
    filter: grep -v marker | grep -v atomic | grep -v usage
  test:
    suffix: swarm_migrate_scan
    nsize: 2
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10,10\
          -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10. -dm_plex_box_bd none,none,none\
          -dm_plex_hash_location false
    filter: grep -v marker | grep -v atomic | grep -v usage
  test:
    suffix: swarm_migrate_scan_tensor_permutation
    nsize: 2
    args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_distribute_overlap 1 -dm_plex_box_faces 10,10,10\
          -dm_plex_box_lower 0.,0.,0. -dm_plex_box_upper 1.,1.,10. -dm_plex_box_bd none,none,none\
          -dm_plex_hash_location false -set_closure_permutation
    filter: grep -v marker | grep -v atomic | grep -v usage
TEST*/
