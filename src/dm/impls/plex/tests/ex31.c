static char help[] = "Test that shared points on interface of partitions can be rebalanced.\n\n";
static char FILENAME[] = "ex31.c";

#include <petscdmplex.h>
#include <petscviewerhdf5.h>
#include <petscsf.h>

typedef struct {
  PetscBool parallel;        /* Use ParMetis or Metis */
  PetscBool useInitialGuess; /* Only active when in parallel, uses RefineKway of ParMetis */
  PetscInt  entityDepth;     /* depth of the entities to rebalance ( 0 => vertices) */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->parallel        = PETSC_FALSE;
  options->useInitialGuess = PETSC_FALSE;
  options->entityDepth     = 0;

  ierr = PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsBoundedInt("-entity_depth", "Depth of the entities to rebalance (0 => vertices)", FILENAME, options->entityDepth, &options->entityDepth, NULL,0));
  CHKERRQ(PetscOptionsBool("-parallel", "Use ParMetis instead of Metis", FILENAME, options->parallel, &options->parallel, NULL));
  CHKERRQ(PetscOptionsBool("-use_initial_guess", "Use RefineKway function of ParMetis", FILENAME, options->useInitialGuess, &options->useInitialGuess, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBegin;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  MPI_Comm       comm;
  DM             dm, dmdist;
  PetscPartitioner part;
  AppCtx         user;
  IS             is=NULL;
  PetscSection   s=NULL, gsection=NULL;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscSF        sf;
  PetscInt       pStart, pEnd, p, minBefore, maxBefore, minAfter, maxAfter, gSizeBefore, gSizeAfter;
  PetscBool      success;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  CHKERRMPI(MPI_Comm_size(comm, &size));
  CHKERRQ(ProcessOptions(comm, &user));
  CHKERRQ(CreateMesh(comm, &user, &dm));

  /* partition dm using PETSCPARTITIONERPARMETIS */
  CHKERRQ(DMPlexGetPartitioner(dm, &part));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject)part,"p_"));
  CHKERRQ(PetscPartitionerSetType(part, PETSCPARTITIONERPARMETIS));
  CHKERRQ(PetscPartitionerSetFromOptions(part));
  CHKERRQ(PetscSectionCreate(comm, &s));
  CHKERRQ(PetscPartitionerDMPlexPartition(part, dm, NULL, s, &is));

  CHKERRQ(DMPlexDistribute(dm, 0, NULL, &dmdist));
  if (dmdist) {
    CHKERRQ(DMDestroy(&dm));
    dm = dmdist;
  }

  /* cleanup */
  CHKERRQ(PetscSectionDestroy(&s));
  CHKERRQ(ISDestroy(&is));

  /* We make a PetscSection with a DOF on every mesh entity of depth
   * user.entityDepth, then make a global section and look at its storage size.
   * We do the same thing after the rebalancing and then assert that the size
   * remains the same. We also make sure that the balance has improved at least
   * a little bit compared to the initial decomposition. */

  if (size>1) {
    CHKERRQ(PetscSectionCreate(comm, &s));
    CHKERRQ(PetscSectionSetNumFields(s, 1));
    CHKERRQ(PetscSectionSetFieldComponents(s, 0, 1));
    CHKERRQ(DMPlexGetDepthStratum(dm, user.entityDepth, &pStart, &pEnd));
    CHKERRQ(PetscSectionSetChart(s, pStart, pEnd));
    for (p = pStart; p < pEnd; ++p) {
      CHKERRQ(PetscSectionSetDof(s, p, 1));
      CHKERRQ(PetscSectionSetFieldDof(s, p, 0, 1));
    }
    CHKERRQ(PetscSectionSetUp(s));
    CHKERRQ(DMGetPointSF(dm, &sf));
    CHKERRQ(PetscSectionCreateGlobalSection(s, sf, PETSC_FALSE, PETSC_FALSE, &gsection));
    CHKERRQ(PetscSectionGetStorageSize(gsection, &gSizeBefore));
    minBefore = gSizeBefore;
    maxBefore = gSizeBefore;
    CHKERRMPI(MPI_Allreduce(MPI_IN_PLACE, &gSizeBefore, 1, MPIU_INT, MPI_SUM, comm));
    CHKERRMPI(MPI_Allreduce(MPI_IN_PLACE, &minBefore, 1, MPIU_INT, MPI_MIN, comm));
    CHKERRMPI(MPI_Allreduce(MPI_IN_PLACE, &maxBefore, 1, MPIU_INT, MPI_MAX, comm));
    CHKERRQ(PetscSectionDestroy(&gsection));
  }

  CHKERRQ(DMPlexRebalanceSharedPoints(dm, user.entityDepth, user.useInitialGuess, user.parallel, &success));

  if (size>1) {
    CHKERRQ(PetscSectionCreateGlobalSection(s, sf, PETSC_FALSE, PETSC_FALSE, &gsection));
    CHKERRQ(PetscSectionGetStorageSize(gsection, &gSizeAfter));
    minAfter = gSizeAfter;
    maxAfter = gSizeAfter;
    CHKERRMPI(MPI_Allreduce(MPI_IN_PLACE, &gSizeAfter, 1, MPIU_INT, MPI_SUM, comm));
    CHKERRMPI(MPI_Allreduce(MPI_IN_PLACE, &minAfter, 1, MPIU_INT, MPI_MIN, comm));
    CHKERRMPI(MPI_Allreduce(MPI_IN_PLACE, &maxAfter, 1, MPIU_INT, MPI_MAX, comm));
    PetscCheckFalse(gSizeAfter != gSizeBefore,comm, PETSC_ERR_PLIB, "Global section has not the same size before and after.");
    PetscCheckFalse(!(minAfter >= minBefore && maxAfter <= maxBefore && (minAfter > minBefore || maxAfter < maxBefore)),comm, PETSC_ERR_PLIB, "DMPlexRebalanceSharedPoints did not improve mesh point balance.");
    CHKERRQ(PetscSectionDestroy(&gsection));
    CHKERRQ(PetscSectionDestroy(&s));
  }

  CHKERRQ(DMDestroy(&dm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  testset:
    args: -dm_plex_dim 3 -dm_plex_simplex 0

    test:
      # rebalance a mesh
      suffix: 0
      nsize: {{2 3 4}}
      requires: parmetis
      args: -dm_plex_box_faces {{2,3,4  5,4,3  7,11,5}} -entity_depth {{0 1}} -parallel {{FALSE TRUE}} -use_initial_guess FALSE

    test:
      # rebalance a mesh but use the initial guess (uses a random algorithm and gives different results on different machines, so just check that it runs).
      suffix: 1
      nsize: {{2 3 4}}
      requires: parmetis
      args: -dm_plex_box_faces {{2,3,4  5,4,3  7,11,5}} -entity_depth {{0 1}} -parallel TRUE -use_initial_guess TRUE

    test:
      # no-op in serial
      suffix: 2
      nsize: {{1}}
      requires: parmetis
      args: -dm_plex_box_faces 2,3,4 -entity_depth 0 -parallel FALSE -use_initial_guess FALSE

TEST*/
