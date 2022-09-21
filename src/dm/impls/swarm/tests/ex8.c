static char help[] = "Tests for particle initialization using the KS test\n\n";

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
  PetscFunctionBegin;
  options->mass[0] = 9.10938356e-31;                                                /* Electron Mass [kg] */
  options->mass[1] = 87.62 * 1.66054e-27;                                           /* Sr+ Mass [kg] */
  options->T[0]    = 1.;                                                            /* Electron Temperature [K] */
  options->T[1]    = 25.;                                                           /* Sr+ Temperature [K] */
  options->v0[0]   = PetscSqrtReal(BOLTZMANN_K * options->T[0] / options->mass[0]); /* electron mean velocity in 1D */
  options->v0[1]   = PetscSqrtReal(BOLTZMANN_K * options->T[1] / options->mass[1]); /* ion mean velocity in 1D */

  PetscOptionsBegin(comm, "", "KS Test Options", "DMPLEX");
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateSwarm(DM dm, AppCtx *user, DM *sw)
{
  PetscInt dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject)dm), sw));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));
  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*sw, dm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "w_q", 1, PETSC_SCALAR));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "velocity", dim, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "species", 1, PETSC_INT));
  PetscCall(DMSwarmFinalizeFieldRegister(*sw));
  PetscCall(DMSwarmComputeLocalSizeFromOptions(*sw));
  PetscCall(DMSwarmInitializeCoordinates(*sw));
  PetscCall(DMSwarmInitializeVelocitiesFromOptions(*sw, user->v0));
  PetscCall(DMSetFromOptions(*sw));
  PetscCall(PetscObjectSetName((PetscObject)*sw, "Particles"));
  PetscCall(DMViewFromOptions(*sw, NULL, "-swarm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestDistribution(DM sw, PetscReal confidenceLevel, AppCtx *user)
{
  Vec           locv;
  PetscProbFunc cdf;
  PetscReal     alpha;
  PetscInt      dim;
  MPI_Comm      comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)sw, &comm));
  PetscCall(DMGetDimension(sw, &dim));
  switch (dim) {
  case 1:
    cdf = PetscCDFMaxwellBoltzmann1D;
    break;
  case 2:
    cdf = PetscCDFMaxwellBoltzmann2D;
    break;
  case 3:
    cdf = PetscCDFMaxwellBoltzmann3D;
    break;
  default:
    SETERRQ(comm, PETSC_ERR_ARG_OUTOFRANGE, "Dimension %" PetscInt_FMT " not supported", dim);
  }
  PetscCall(DMSwarmCreateLocalVectorFromField(sw, "velocity", &locv));
  PetscCall(PetscProbComputeKSStatistic(locv, cdf, &alpha));
  PetscCall(DMSwarmDestroyLocalVectorFromField(sw, "velocity", &locv));
  if (alpha < confidenceLevel) PetscCall(PetscPrintf(comm, "The KS test accepts the null hypothesis at level %.2g\n", (double)confidenceLevel));
  else PetscCall(PetscPrintf(comm, "The KS test rejects the null hypothesis at level %.2g (%.2g)\n", (double)confidenceLevel, (double)alpha));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM     dm, sw;
  AppCtx user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
  PetscCall(CreateSwarm(dm, &user, &sw));
  PetscCall(TestDistribution(sw, 0.05, &user));
  PetscCall(DMDestroy(&sw));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    requires: ks !complex
    args: -dm_plex_dim 1 -dm_plex_box_lower -1 -dm_plex_box_upper 1 -dm_swarm_num_particles 375 -dm_swarm_coordinate_density {{constant gaussian}}

TEST*/
