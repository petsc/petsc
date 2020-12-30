static char help[] = "Test that shared points on interface of partitions can be rebalanced.\n\n";
static char FILENAME[] = "ex31.c";

#include <petscdmplex.h>
#include <petscviewerhdf5.h>
#include "petscsf.h"


typedef struct {
  PetscInt  dim;                          /* The topological mesh dimension */ PetscInt  faces[3];                     /* Number of faces per dimension */
  PetscBool simplex;                      /* Use simplices or hexes */
  PetscBool interpolate;                  /* Interpolate mesh */
  PetscBool parallel;                     /* Use ParMetis or Metis */
  PetscBool useInitialGuess;              /* Only active when in parallel, uses RefineKway of ParMetis */
  PetscInt  entityDepth;                  /* depth of the entities to rebalance ( 0 => vertices) */
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
  ierr = PetscOptionsBoundedInt("-entity_depth", "Depth of the entities to rebalance (0 => vertices)", FILENAME, options->entityDepth, &options->entityDepth, NULL,0);CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim", "The topological mesh dimension", FILENAME, options->dim, &options->dim, NULL,1,3);CHKERRQ(ierr);
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
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
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
  PetscSection   s=NULL, gsection=NULL;
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscSF        sf;
  PetscInt       pStart, pEnd, p, minBefore, maxBefore, minAfter, maxAfter, gSizeBefore, gSizeAfter;
  PetscBool      success;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &user, &dm);CHKERRQ(ierr);

  /* partition dm using PETSCPARTITIONERPARMETIS */
  ierr = DMPlexGetPartitioner(dm, &part);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)part,"p_");CHKERRQ(ierr);
  ierr = PetscPartitionerSetType(part, PETSCPARTITIONERPARMETIS);CHKERRQ(ierr);
  ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &s);CHKERRQ(ierr);
  ierr = PetscPartitionerDMPlexPartition(part, dm, NULL, s, &is);CHKERRQ(ierr);

  ierr = DMPlexDistribute(dm, 0, NULL, &dmdist);CHKERRQ(ierr);
  if (dmdist) {
    ierr = DMDestroy(&dm);CHKERRQ(ierr);
    dm = dmdist;
  }

  /* cleanup */
  ierr = PetscSectionDestroy(&s);CHKERRQ(ierr);
  ierr = ISDestroy(&is);CHKERRQ(ierr);

  /* We make a PetscSection with a DOF on every mesh entity of depth
   * user.entityDepth, then make a global section and look at its storage size.
   * We do the same thing after the rebalancing and then assert that the size
   * remains the same. We also make sure that the balance has improved at least
   * a little bit compared to the initial decomposition. */

  if (size>1) {
    ierr = PetscSectionCreate(comm, &s);CHKERRQ(ierr);
    ierr = PetscSectionSetNumFields(s, 1);CHKERRQ(ierr);
    ierr = PetscSectionSetFieldComponents(s, 0, 1);CHKERRQ(ierr);
    ierr = DMPlexGetDepthStratum(dm, user.entityDepth, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = PetscSectionSetChart(s, pStart, pEnd);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {
      ierr = PetscSectionSetDof(s, p, 1);CHKERRQ(ierr);
      ierr = PetscSectionSetFieldDof(s, p, 0, 1);CHKERRQ(ierr);
    }
    ierr = PetscSectionSetUp(s);CHKERRQ(ierr);
    ierr = DMGetPointSF(dm, &sf);CHKERRQ(ierr);
    ierr = PetscSectionCreateGlobalSection(s, sf, PETSC_FALSE, PETSC_FALSE, &gsection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(gsection, &gSizeBefore);CHKERRQ(ierr);
    minBefore = gSizeBefore;
    maxBefore = gSizeBefore;
    ierr = MPI_Allreduce(MPI_IN_PLACE, &gSizeBefore, 1, MPIU_INT, MPI_SUM, comm);CHKERRMPI(ierr);
    ierr = MPI_Allreduce(MPI_IN_PLACE, &minBefore, 1, MPIU_INT, MPI_MIN, comm);CHKERRMPI(ierr);
    ierr = MPI_Allreduce(MPI_IN_PLACE, &maxBefore, 1, MPIU_INT, MPI_MAX, comm);CHKERRMPI(ierr);
    ierr = PetscSectionDestroy(&gsection);CHKERRQ(ierr);
  }

  ierr = DMPlexRebalanceSharedPoints(dm, user.entityDepth, user.useInitialGuess, user.parallel, &success);CHKERRQ(ierr);

  if (size>1) {
    ierr = PetscSectionCreateGlobalSection(s, sf, PETSC_FALSE, PETSC_FALSE, &gsection);CHKERRQ(ierr);
    ierr = PetscSectionGetStorageSize(gsection, &gSizeAfter);CHKERRQ(ierr);
    minAfter = gSizeAfter;
    maxAfter = gSizeAfter;
    ierr = MPI_Allreduce(MPI_IN_PLACE, &gSizeAfter, 1, MPIU_INT, MPI_SUM, comm);CHKERRMPI(ierr);
    ierr = MPI_Allreduce(MPI_IN_PLACE, &minAfter, 1, MPIU_INT, MPI_MIN, comm);CHKERRMPI(ierr);
    ierr = MPI_Allreduce(MPI_IN_PLACE, &maxAfter, 1, MPIU_INT, MPI_MAX, comm);CHKERRMPI(ierr);
    if (gSizeAfter != gSizeBefore) SETERRQ(comm, PETSC_ERR_PLIB, "Global section has not the same size before and after.");
    if (!(minAfter >= minBefore && maxAfter <= maxBefore && (minAfter > minBefore || maxAfter < maxBefore))) SETERRQ(comm, PETSC_ERR_PLIB, "DMPlexRebalanceSharedPoints did not improve mesh point balance.");
    ierr = PetscSectionDestroy(&gsection);CHKERRQ(ierr);
    ierr = PetscSectionDestroy(&s);CHKERRQ(ierr);
  }

  ierr = DMDestroy(&dm);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    # rebalance a mesh
    suffix: 0
    nsize: {{2 3 4}}
    requires: parmetis
    args: -faces {{2,3,4  5,4,3  7,11,5}} -interpolate -entity_depth {{0 1}} -parallel {{FALSE TRUE}} -use_initial_guess FALSE

  test:
    # rebalance a mesh but use the initial guess (uses a random algorithm and gives different results on different machines, so just check that it runs).
    suffix: 1
    nsize: {{2 3 4}}
    requires: parmetis
    args: -faces {{2,3,4  5,4,3  7,11,5}} -interpolate -entity_depth {{0 1}} -parallel TRUE -use_initial_guess TRUE

  test:
    # no-op in serial
    suffix: 2
    nsize: {{1}}
    requires: parmetis
    args: -faces 2,3,4 -interpolate -entity_depth 0 -parallel FALSE -use_initial_guess FALSE

TEST*/
