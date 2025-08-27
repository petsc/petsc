static char help[] = "Test periodic DMDA for DMSwarm point location.\n";

#include <petscdmda.h>
#include <petscdmswarm.h>

typedef struct {
  PetscInt dim; // Mesh dimension
  PetscInt Np;  // Number of particles along each dimension
} UserContext;

static PetscErrorCode ProcessOptions(UserContext *options)
{
  PetscFunctionBeginUser;
  options->dim = 3;
  options->Np  = -1;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &options->dim, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-np", &options->Np, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(DM *da, UserContext *user)
{
  PetscReal gmin[3] = {0, 0., 0.}, gmax[3] = {0, 0., 0.};

  PetscFunctionBeginUser;
  PetscCall(DMDACreate(PETSC_COMM_WORLD, da));
  PetscCall(DMSetDimension(*da, user->dim));
  PetscCall(DMDASetSizes(*da, 7, 7, 7));
  PetscCall(DMDASetBoundaryType(*da, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED));
  PetscCall(DMDASetDof(*da, 2));
  PetscCall(DMDASetStencilType(*da, DMDA_STENCIL_BOX));
  PetscCall(DMDASetStencilWidth(*da, 1));
  PetscCall(DMDASetElementType(*da, DMDA_ELEMENT_Q1));
  PetscCall(DMSetFromOptions(*da));
  PetscCall(DMSetUp(*da));
  PetscCall(DMDASetUniformCoordinates(*da, 0., 1., 0., 1., 0., 1.));
  PetscCall(DMSetApplicationContext(*da, user));
  PetscCall(DMView(*da, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMGetBoundingBox(*da, gmin, gmax));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "min: (%g, %g, %g) max: (%g, %g, %g)\n", gmin[0], gmin[1], gmin[2], gmax[0], gmax[1], gmax[2]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSwarm(DM mesh, DM *swarm, UserContext *user)
{
  MPI_Comm    comm;
  PetscMPIInt size;
  PetscInt    dim;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)mesh, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(DMCreate(comm, swarm));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*swarm, "pic_"));
  PetscCall(DMSetType(*swarm, DMSWARM));
  PetscCall(PetscObjectSetName((PetscObject)*swarm, "ions"));
  PetscCall(DMGetDimension(mesh, &dim));
  PetscCall(DMSetDimension(*swarm, dim));
  PetscCall(DMSwarmSetType(*swarm, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*swarm, mesh));
  PetscCall(DMSwarmInitializeFieldRegister(*swarm));
  PetscCall(DMSwarmFinalizeFieldRegister(*swarm));
  PetscCall(DMSwarmSetLocalSizes(*swarm, user->Np / size, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitializeParticles(DM sw, UserContext *user)
{
  DM        da;
  PetscReal gmin[3], gmax[3];
  PetscInt  ndir[3];

  PetscFunctionBeginUser;
  PetscCall(DMSwarmGetCellDM(sw, &da));
  PetscCall(DMGetBoundingBox(da, gmin, gmax));
  ndir[0] = user->Np;
  ndir[1] = user->Np;
  ndir[2] = user->Np;
  PetscCall(DMSwarmSetPointsUniformCoordinates(sw, gmin, gmax, ndir, INSERT_VALUES));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  DM          dm, sw;
  UserContext user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(ProcessOptions(&user));
  PetscCall(CreateMesh(&dm, &user));
  PetscCall(CreateSwarm(dm, &sw, &user));

  PetscCall(InitializeParticles(sw, &user));
  PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));
  PetscCall(DMViewFromOptions(sw, NULL, "-sw_view"));

  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&sw));
  PetscCall(PetscFinalize());
  return 0;
}

// ./dmswarm-coords -nx 8 -ny 8 -np 4 -pic_sw_view -dim 2 -periodic 0
// ./dmswarm-coords -nx 8 -ny 8 -np 4 -pic_sw_view -dim 2 -periodic 1
// ./dmswarm-coords -nx 8 -ny 8 -nz 8 -np 4 -pic_sw_view -dim 3 -periodic 0
// ./dmswarm-coords -nx 8 -ny 8 -nz 8 -np 4 -pic_sw_view -dim 3 -periodic 1

/*TEST

  build:
    requires: double !complex

  testset:
    suffix: 2d
    args: -dim 2 -da_grid_x 8 -da_grid_y 8 -np 4 -da_bd_all {{none periodic}} -pic_sw_view

    test:
      suffix: p1

    test:
      suffix: p2
      nsize: 2

    test:
      suffix: p4
      nsize: 4

  testset:
    suffix: 3d
    args: -dim 3 -da_grid_x 8 -da_grid_y 8 -da_grid_z 8 -np 4 -da_bd_all {{none periodic}} -pic_sw_view

    test:
      suffix: p1

    test:
      suffix: p2
      nsize: 2

    test:
      suffix: p4
      nsize: 4

TEST*/
