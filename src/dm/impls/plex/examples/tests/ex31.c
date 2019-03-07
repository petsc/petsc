static char help[] = "Test that shared points on interface of partitions can be rebalanced.\n\n";
static char FILENAME[] = "ex31.c";

#include <petscdmplex.h>
#include <petscviewerhdf5.h>

typedef struct {
  PetscInt  dim;                          /* The topological mesh dimension */
  PetscInt  faces[3];                     /* Number of faces per dimension */
  PetscBool simplex;                      /* Use simplices or hexes */
  PetscBool interpolate;                  /* Interpolate mesh */
  PetscBool parallel;                     /* Use ParMetis or Metis */
  PetscBool useInitialGuess;              /* Only active when in parallel, uses RefineKway of ParMetis */
  PetscInt  entityDepth;                  /* depth of the entities to rebalance ( 0 => vertices ) */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt dim;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->dim             = 3;
  options->simplex         = PETSC_FALSE;
  options->interpolate     = PETSC_FALSE;
  options->entityDepth     = 0;
  options->parallel        = PETSC_FALSE;
  options->useInitialGuess = PETSC_FALSE;
  options->entityDepth     = 0;
  ierr = PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-entity_depth", "Depth of the entities to rebalance ( 0 => vertices )", FILENAME, options->entityDepth, &options->entityDepth, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", FILENAME, options->dim, &options->dim, NULL);CHKERRQ(ierr);
  if (options->dim > 3) SETERRQ1(comm, PETSC_ERR_ARG_OUTOFRANGE, "dimension set to %d, must be <= 3", options->dim);
  ierr = PetscOptionsBool("-simplex", "Use simplices if true, otherwise hexes", FILENAME, options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Interpolate the mesh", FILENAME, options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-parallel", "Use ParMetis instead of Metis", FILENAME, options->parallel, &options->parallel, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-use_initial_guess", "Use RefineKway function of ParMetis", FILENAME, options->useInitialGuess, &options->useInitialGuess, NULL);CHKERRQ(ierr);
  options->faces[0] = 1; options->faces[1] = 1; options->faces[2] = 1;
  dim = options->dim;
  ierr = PetscOptionsIntArray("-faces", "Number of faces per dimension", FILENAME, options->faces, &dim, NULL);CHKERRQ(ierr);
  if (dim) options->dim = dim;
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}


static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscInt       dim          = user->dim;
  PetscInt      *faces        = user->faces;
  PetscBool      simplex      = user->simplex;
  PetscBool      interpolate  = user->interpolate;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
  ierr = DMPlexCreateBoxMesh(comm, dim, simplex, faces, NULL, NULL, NULL, interpolate, dm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  MPI_Comm       comm;
  DM             dm, dmdist;
  PetscPartitioner part;
  AppCtx         user;
  IS             is=NULL;
  PetscSection   s=NULL;
  PetscErrorCode ierr;
  PetscMPIInt    size;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &user, &dm);CHKERRQ(ierr);

  /* partition dm using PETSCPARTITIONERPARMETIS */
  ierr = DMPlexGetPartitioner(dm, &part);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)part,"p_");CHKERRQ(ierr);
  ierr = PetscPartitionerSetType(part, PETSCPARTITIONERPARMETIS);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &s);CHKERRQ(ierr);
  ierr = PetscPartitionerPartition(part, dm, s, &is);CHKERRQ(ierr);

  ierr = DMPlexDistribute(dm, 0, NULL, &dmdist);CHKERRQ(ierr);
  if (dmdist) {
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    dm = dmdist;
  }

  /* cleanup */
  ierr = PetscSectionDestroy(&s);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);

  ierr = DMPlexRebalanceSharedPoints(dm, user.entityDepth, user.useInitialGuess, user.parallel);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    # rebalance a mesh
    suffix: 0
    nsize: {{1 2 3 4}separate output}
    requires: parmetis
    args: -faces {{2,3,4  5,4,3  7,11,5}separate output} -partitioning parmetis -interpolate -dm_rebalance_partition_view -entity_depth {{0 1}separate output} -parallel {{FALSE TRUE}separate output} -use_initial_guess {{FALSE TRUE}separate output}

TEST*/

