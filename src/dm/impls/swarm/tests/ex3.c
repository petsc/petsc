static char help[] = "Example usage of extracting single cells with their associated fields from a swarm and putting it in a new swarm object\n";

#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscts.h>

typedef struct {
  PetscInt particlesPerCell; /* The number of partices per cell */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->particlesPerCell = 1;

  ierr = PetscOptionsBegin(comm, "", "CellSwarm Options", "DMSWARM");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsInt("-particles_per_cell", "Number of particles per cell", "ex3.c", options->particlesPerCell, &options->particlesPerCell, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscFunctionBeginUser;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateParticles(DM dm, DM *sw, AppCtx *user)
{
  PetscInt      *cellid;
  PetscInt       dim, cStart, cEnd, c, Np = user->particlesPerCell, p;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMCreate(PetscObjectComm((PetscObject) dm), sw));
  CHKERRQ(DMSetType(*sw, DMSWARM));
  CHKERRQ(DMSetDimension(*sw, dim));
  CHKERRQ(DMSwarmSetType(*sw, DMSWARM_PIC));
  CHKERRQ(DMSwarmSetCellDM(*sw, dm));
  CHKERRQ(DMSwarmRegisterPetscDatatypeField(*sw, "kinematics", 2, PETSC_REAL));
  CHKERRQ(DMSwarmFinalizeFieldRegister(*sw));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  CHKERRQ(DMSwarmSetLocalSizes(*sw, (cEnd - cStart) * Np, 0));
  CHKERRQ(DMSetFromOptions(*sw));
  CHKERRQ(DMSwarmGetField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid));
  for (c = cStart; c < cEnd; ++c) {
    for (p = 0; p < Np; ++p) {
      const PetscInt n = c*Np + p;

      cellid[n] = c;
    }
  }
  CHKERRQ(DMSwarmRestoreField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid));
  CHKERRQ(PetscObjectSetName((PetscObject) *sw, "Particles"));
  CHKERRQ(DMViewFromOptions(*sw, NULL, "-sw_view"));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  DM             dm, sw, cellsw; /* Mesh and particle managers */
  MPI_Comm       comm;
  AppCtx         user;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  CHKERRQ(ProcessOptions(comm, &user));
  CHKERRQ(CreateMesh(comm, &dm, &user));
  CHKERRQ(CreateParticles(dm, &sw, &user));
  CHKERRQ(DMSetApplicationContext(sw, &user));
  CHKERRQ(DMCreate(comm, &cellsw));
  CHKERRQ(DMSwarmGetCellSwarm(sw, 1, cellsw));
  CHKERRQ(DMViewFromOptions(cellsw, NULL, "-subswarm_view"));
  CHKERRQ(DMSwarmRestoreCellSwarm(sw, 1, cellsw));
  CHKERRQ(DMDestroy(&sw));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(DMDestroy(&cellsw));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST
   build:
     requires: triangle !single !complex
   test:
     suffix: 1
     args: -particles_per_cell 2 -dm_plex_box_faces 2,1 -dm_view -sw_view -subswarm_view
TEST*/
