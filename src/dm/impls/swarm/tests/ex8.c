static char help[] = "Tests for KS test\n\n";

#include <petscdmswarm.h>
#include <petscdmplex.h>
#include <petsc/private/petscfeimpl.h>

/*
  View the KS test using

    -ks_monitor draw -draw_size 500,500 -draw_pause 3

  Set total number to n0 / Mp = 3e14 / 1e12 =  300 using -dm_swarm_num_particles 300

*/

#define BOLTZMANN_K 1.380649e-23 /* J/K */

typedef struct {
  PetscReal mass[2]; /* Electron, Sr+ Mass [kg] */
  PetscReal T[2];    /* Electron, Ion Temperature [K] */
  PetscReal v0[2];   /* Species mean velocity in 1D */
} AppCtx;

PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->mass[0] = 9.10938356e-31; /* Electron Mass [kg] */
  options->mass[1] = 87.62 * 1.66054e-27; /* Sr+ Mass [kg] */
  options->T[0]    = 1.; /* Electron Temperature [K] */
  options->T[1]    = 25.; /* Sr+ Temperature [K] */
  options->v0[0]   = PetscSqrtReal(BOLTZMANN_K * options->T[0] / options->mass[0]); /* electron mean velocity in 1D */
  options->v0[1]   = PetscSqrtReal(BOLTZMANN_K * options->T[1] / options->mass[1]); /* ion mean velocity in 1D */

  ierr = PetscOptionsBegin(comm, "", "KS Test Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBeginUser;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateSwarm(DM dm, AppCtx *user, DM *sw)
{
  PetscInt       dim;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMCreate(PetscObjectComm((PetscObject) dm), sw));
  CHKERRQ(DMSetType(*sw, DMSWARM));
  CHKERRQ(DMSetDimension(*sw, dim));
  CHKERRQ(DMSwarmSetType(*sw, DMSWARM_PIC));
  CHKERRQ(DMSwarmSetCellDM(*sw, dm));
  CHKERRQ(DMSwarmRegisterPetscDatatypeField(*sw, "w_q", 1, PETSC_SCALAR));
  CHKERRQ(DMSwarmRegisterPetscDatatypeField(*sw, "velocity", dim, PETSC_REAL));
  CHKERRQ(DMSwarmRegisterPetscDatatypeField(*sw, "species", 1, PETSC_INT));
  CHKERRQ(DMSwarmFinalizeFieldRegister(*sw));
  CHKERRQ(DMSwarmComputeLocalSizeFromOptions(*sw));
  CHKERRQ(DMSwarmInitializeCoordinates(*sw));
  CHKERRQ(DMSwarmInitializeVelocitiesFromOptions(*sw, user->v0));
  CHKERRQ(DMSetFromOptions(*sw));
  CHKERRQ(PetscObjectSetName((PetscObject) *sw, "Particles"));
  CHKERRQ(DMViewFromOptions(*sw, NULL, "-swarm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestDistribution(DM sw, PetscReal confidenceLevel, AppCtx *user)
{
  Vec            locv;
  PetscProbFunc  cdf;
  PetscReal      alpha;
  PetscInt       dim;
  MPI_Comm       comm;

  PetscFunctionBeginUser;
  CHKERRQ(PetscObjectGetComm((PetscObject) sw, &comm));
  CHKERRQ(DMGetDimension(sw, &dim));
  switch (dim) {
    case 1: cdf = PetscCDFMaxwellBoltzmann1D;break;
    case 2: cdf = PetscCDFMaxwellBoltzmann2D;break;
    case 3: cdf = PetscCDFMaxwellBoltzmann3D;break;
    default: SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Dimension %D not supported", dim);
  }
  CHKERRQ(DMSwarmCreateLocalVectorFromField(sw, "velocity", &locv));
  CHKERRQ(PetscProbComputeKSStatistic(locv, cdf, &alpha));
  CHKERRQ(DMSwarmDestroyLocalVectorFromField(sw, "velocity", &locv));
  if (alpha < confidenceLevel) CHKERRQ(PetscPrintf(comm, "The KS test accepts the null hypothesis at level %.2g\n", (double) confidenceLevel));
  else                         CHKERRQ(PetscPrintf(comm, "The KS test rejects the null hypothesis at level %.2g (%.2g)\n", (double) confidenceLevel, (double) alpha));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm, sw;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &user));
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &dm));
  CHKERRQ(CreateSwarm(dm, &user, &sw));
  CHKERRQ(TestDistribution(sw, 0.05, &user));
  CHKERRQ(DMDestroy(&sw));
  CHKERRQ(DMDestroy(&dm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    requires: ks !complex
    args: -dm_plex_dim 1 -dm_plex_box_lower -1 -dm_plex_box_upper 1 -dm_swarm_num_particles 300 -dm_swarm_coordinate_density {{constant gaussian}}

TEST*/
