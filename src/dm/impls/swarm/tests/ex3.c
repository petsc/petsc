static char help[] = "Example usage of extracting single cells with their associated fields from a swarm and putting it in a new swarm object\n";

#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscts.h>

typedef struct {
  PetscInt particlesPerCell; /* The number of partices per cell */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->particlesPerCell = 1;

  PetscOptionsBegin(comm, "", "CellSwarm Options", "DMSWARM");
  PetscCall(PetscOptionsInt("-particles_per_cell", "Number of particles per cell", "ex3.c", options->particlesPerCell, &options->particlesPerCell, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateParticles(DM dm, DM *sw, AppCtx *user)
{
  PetscInt *cellid;
  PetscInt  dim, cStart, cEnd, c, Np = user->particlesPerCell, p;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject)dm), sw));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));
  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*sw, dm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "kinematics", 2, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(*sw));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMSwarmSetLocalSizes(*sw, (cEnd - cStart) * Np, 0));
  PetscCall(DMSetFromOptions(*sw));
  PetscCall(DMSwarmGetField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid));
  for (c = cStart; c < cEnd; ++c) {
    for (p = 0; p < Np; ++p) {
      const PetscInt n = c * Np + p;

      cellid[n] = c;
    }
  }
  PetscCall(DMSwarmRestoreField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **)&cellid));
  PetscCall(PetscObjectSetName((PetscObject)*sw, "Particles"));
  PetscCall(DMViewFromOptions(*sw, NULL, "-sw_view"));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM       dm, sw, cellsw; /* Mesh and particle managers */
  MPI_Comm comm;
  AppCtx   user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(ProcessOptions(comm, &user));
  PetscCall(CreateMesh(comm, &dm, &user));
  PetscCall(CreateParticles(dm, &sw, &user));
  PetscCall(DMSetApplicationContext(sw, &user));
  PetscCall(DMCreate(comm, &cellsw));
  PetscCall(PetscObjectSetName((PetscObject)cellsw, "SubParticles"));
  PetscCall(DMSwarmGetCellSwarm(sw, 1, cellsw));
  PetscCall(DMViewFromOptions(cellsw, NULL, "-subswarm_view"));
  PetscCall(DMSwarmRestoreCellSwarm(sw, 1, cellsw));
  PetscCall(DMDestroy(&sw));
  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&cellsw));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
   build:
     requires: triangle !single !complex
   test:
     suffix: 1
     args: -particles_per_cell 2 -dm_plex_box_faces 2,1 -dm_view -sw_view -subswarm_view
TEST*/
