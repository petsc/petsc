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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateSwarm(DM dm, AppCtx *user, DM *sw)
{
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), sw);CHKERRQ(ierr);
  ierr = DMSetType(*sw, DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(*sw, dim);CHKERRQ(ierr);
  ierr = DMSwarmSetType(*sw, DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(*sw, dm);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(*sw, "w_q", 1, PETSC_SCALAR);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(*sw, "velocity", dim, PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(*sw, "species", 1, PETSC_INT);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(*sw);CHKERRQ(ierr);
  ierr = DMSwarmComputeLocalSizeFromOptions(*sw);CHKERRQ(ierr);
  ierr = DMSwarmInitializeCoordinates(*sw);CHKERRQ(ierr);
  ierr = DMSwarmInitializeVelocitiesFromOptions(*sw, user->v0);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*sw);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *sw, "Particles");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*sw, NULL, "-swarm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestDistribution(DM sw, PetscReal confidenceLevel, AppCtx *user)
{
  Vec            locv;
  PetscProbFunc  cdf;
  PetscReal      alpha;
  PetscInt       dim;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject) sw, &comm);CHKERRQ(ierr);
  ierr = DMGetDimension(sw, &dim);CHKERRQ(ierr);
  switch (dim) {
    case 1: cdf = PetscCDFMaxwellBoltzmann1D;break;
    case 2: cdf = PetscCDFMaxwellBoltzmann2D;break;
    case 3: cdf = PetscCDFMaxwellBoltzmann3D;break;
    default: SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Dimension %D not supported", dim);
  }
  ierr = DMSwarmCreateLocalVectorFromField(sw, "velocity", &locv);CHKERRQ(ierr);
  ierr = PetscProbComputeKSStatistic(locv, cdf, &alpha);CHKERRQ(ierr);
  ierr = DMSwarmDestroyLocalVectorFromField(sw, "velocity", &locv);CHKERRQ(ierr);
  if (alpha < confidenceLevel) {ierr = PetscPrintf(comm, "The KS test accepts the null hypothesis at level %.2g\n", (double) confidenceLevel);CHKERRQ(ierr);}
  else                         {ierr = PetscPrintf(comm, "The KS test rejects the null hypothesis at level %.2g (%.2g)\n", (double) confidenceLevel, (double) alpha);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm, sw;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = CreateSwarm(dm, &user, &sw);CHKERRQ(ierr);
  ierr = TestDistribution(sw, 0.05, &user);CHKERRQ(ierr);
  ierr = DMDestroy(&sw);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    requires: ks !complex
    args: -dm_plex_dim 1 -dm_plex_box_lower -1 -dm_plex_box_upper 1 -dm_swarm_num_particles 300 -dm_swarm_coordinate_density {{constant gaussian}}

TEST*/
