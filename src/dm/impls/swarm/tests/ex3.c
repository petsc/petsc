static char help[] = "Example usage of extracting single cells with their associated fields from a swarm and putting it in a new swarm object\n";

#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscts.h>

typedef struct {
  PetscInt  dim;                          /* The topological mesh dimension */
  PetscBool simplex;                      /* Flag for simplices or tensor cells */
  char      filename[PETSC_MAX_PATH_LEN]; /* Name of the mesh filename if any */
  PetscInt  particlesPerCell;             /* The number of partices per cell */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim              = 2;
  options->simplex          = PETSC_TRUE;
  options->particlesPerCell = 1;
  ierr = PetscStrcpy(options->filename, "");CHKERRQ(ierr);
  ierr = PetscOptionsBegin(comm, "", "CellSwarm Options", "DMSWARM");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex3.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "The flag for simplices or tensor cells", "ex3.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-mesh", "Name of the mesh filename if any", "ex3.c", options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-particles_per_cell", "Number of particles per cell", "ex3.c", options->particlesPerCell, &options->particlesPerCell, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscStrcmp(user->filename, "", &flg);CHKERRQ(ierr);
  if (flg) {
    ierr = DMPlexCreateBoxMesh(comm, user->dim, user->simplex, NULL, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateFromFile(comm, user->filename, PETSC_TRUE, dm);CHKERRQ(ierr);
    ierr = DMGetDimension(*dm, &user->dim);CHKERRQ(ierr);
  }
  {
    DM distributedMesh = NULL;

    ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
  }
  ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr); /* needed for periodic */
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateParticles(DM dm, DM *sw, AppCtx *user)
{
  PetscInt      *cellid;
  PetscInt       dim, cStart, cEnd, c, Np = user->particlesPerCell, p;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), sw);CHKERRQ(ierr);
  ierr = DMSetType(*sw, DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(*sw, dim);CHKERRQ(ierr);
  ierr = DMSwarmSetType(*sw, DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(*sw, dm);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(*sw, "kinematics", 2, PETSC_REAL);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(*sw);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMSwarmSetLocalSizes(*sw, (cEnd - cStart) * Np, 0);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*sw);CHKERRQ(ierr);
  ierr = DMSwarmGetField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    for (p = 0; p < Np; ++p) {
      const PetscInt n = c*Np + p;

      cellid[n] = c;
    }
  }
  ierr = DMSwarmRestoreField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *sw, "Particles");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*sw, NULL, "-sw_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  DM             dm, sw, cellsw; /* Mesh and particle managers */
  MPI_Comm       comm;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &dm, &user);CHKERRQ(ierr);
  ierr = CreateParticles(dm, &sw, &user);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(sw, &user);CHKERRQ(ierr);
  ierr = DMCreate(comm, &cellsw);CHKERRQ(ierr);
  ierr = DMSwarmGetCellSwarm(sw, 1, cellsw);CHKERRQ(ierr);
  ierr = DMViewFromOptions(cellsw, NULL, "-subswarm_view");CHKERRQ(ierr);
  ierr = DMSwarmRestoreCellSwarm(sw, 1, cellsw);CHKERRQ(ierr);
  ierr = DMDestroy(&sw);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&cellsw);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
   build:
     requires: triangle !single !complex
   test:
     suffix: 1
     args: -particles_per_cell 2 -dm_plex_box_faces 2,1 -dm_view -sw_view -subswarm_view
TEST*/
