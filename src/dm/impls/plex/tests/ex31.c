static char help[]     = "Test that shared points on interface of partitions can be rebalanced.\n\n";
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
  PetscFunctionBegin;
  options->parallel        = PETSC_FALSE;
  options->useInitialGuess = PETSC_FALSE;
  options->entityDepth     = 0;

  PetscOptionsBegin(comm, "", "Meshing Interpolation Test Options", "DMPLEX");
  PetscCall(PetscOptionsBoundedInt("-entity_depth", "Depth of the entities to rebalance (0 => vertices)", FILENAME, options->entityDepth, &options->entityDepth, NULL, 0));
  PetscCall(PetscOptionsBool("-parallel", "Use ParMetis instead of Metis", FILENAME, options->parallel, &options->parallel, NULL));
  PetscCall(PetscOptionsBool("-use_initial_guess", "Use RefineKway function of ParMetis", FILENAME, options->useInitialGuess, &options->useInitialGuess, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  MPI_Comm         comm;
  DM               dm, dmdist;
  PetscPartitioner part;
  AppCtx           user;
  IS               is = NULL;
  PetscSection     s = NULL, gsection = NULL;
  PetscMPIInt      size;
  PetscSF          sf;
  PetscInt         pStart, pEnd, p, minBefore, maxBefore, minAfter, maxAfter, gSizeBefore, gSizeAfter;
  PetscBool        success;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCall(ProcessOptions(comm, &user));
  PetscCall(CreateMesh(comm, &user, &dm));

  /* partition dm using PETSCPARTITIONERPARMETIS */
  PetscCall(DMPlexGetPartitioner(dm, &part));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)part, "p_"));
  PetscCall(PetscPartitionerSetType(part, PETSCPARTITIONERPARMETIS));
  PetscCall(PetscPartitionerSetFromOptions(part));
  PetscCall(PetscSectionCreate(comm, &s));
  PetscCall(PetscPartitionerDMPlexPartition(part, dm, NULL, s, &is));

  PetscCall(DMPlexDistribute(dm, 0, NULL, &dmdist));
  if (dmdist) {
    PetscCall(DMDestroy(&dm));
    dm = dmdist;
  }

  /* cleanup */
  PetscCall(PetscSectionDestroy(&s));
  PetscCall(ISDestroy(&is));

  /* We make a PetscSection with a DOF on every mesh entity of depth
   * user.entityDepth, then make a global section and look at its storage size.
   * We do the same thing after the rebalancing and then assert that the size
   * remains the same. We also make sure that the balance has improved at least
   * a little bit compared to the initial decomposition. */

  if (size > 1) {
    PetscCall(PetscSectionCreate(comm, &s));
    PetscCall(PetscSectionSetNumFields(s, 1));
    PetscCall(PetscSectionSetFieldComponents(s, 0, 1));
    PetscCall(DMPlexGetDepthStratum(dm, user.entityDepth, &pStart, &pEnd));
    PetscCall(PetscSectionSetChart(s, pStart, pEnd));
    for (p = pStart; p < pEnd; ++p) {
      PetscCall(PetscSectionSetDof(s, p, 1));
      PetscCall(PetscSectionSetFieldDof(s, p, 0, 1));
    }
    PetscCall(PetscSectionSetUp(s));
    PetscCall(DMGetPointSF(dm, &sf));
    PetscCall(PetscSectionCreateGlobalSection(s, sf, PETSC_FALSE, PETSC_FALSE, &gsection));
    PetscCall(PetscSectionGetStorageSize(gsection, &gSizeBefore));
    minBefore = gSizeBefore;
    maxBefore = gSizeBefore;
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &gSizeBefore, 1, MPIU_INT, MPI_SUM, comm));
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &minBefore, 1, MPIU_INT, MPI_MIN, comm));
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &maxBefore, 1, MPIU_INT, MPI_MAX, comm));
    PetscCall(PetscSectionDestroy(&gsection));
  }

  PetscCall(DMPlexRebalanceSharedPoints(dm, user.entityDepth, user.useInitialGuess, user.parallel, &success));

  if (size > 1) {
    PetscCall(PetscSectionCreateGlobalSection(s, sf, PETSC_FALSE, PETSC_FALSE, &gsection));
    PetscCall(PetscSectionGetStorageSize(gsection, &gSizeAfter));
    minAfter = gSizeAfter;
    maxAfter = gSizeAfter;
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &gSizeAfter, 1, MPIU_INT, MPI_SUM, comm));
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &minAfter, 1, MPIU_INT, MPI_MIN, comm));
    PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE, &maxAfter, 1, MPIU_INT, MPI_MAX, comm));
    PetscCheck(gSizeAfter == gSizeBefore, comm, PETSC_ERR_PLIB, "Global section has not the same size before and after.");
    PetscCheck(minAfter >= minBefore && maxAfter <= maxBefore && (minAfter > minBefore || maxAfter < maxBefore), comm, PETSC_ERR_PLIB, "DMPlexRebalanceSharedPoints did not improve mesh point balance.");
    PetscCall(PetscSectionDestroy(&gsection));
    PetscCall(PetscSectionDestroy(&s));
  }

  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
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
