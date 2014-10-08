static char help[] = "Partition a mesh in parallel, perhaps with overlap\n\n";

#include <petscdmplex.h>

typedef struct {
  /* Domain and mesh definition */
  PetscInt  dim;                          /* The topological mesh dimension */
  PetscBool cellSimplex;                  /* Use simplices or hexes */
  char      filename[PETSC_MAX_PATH_LEN]; /* Import mesh from file */
  PetscInt  overlap;                      /* The cell overlap to use during partitioning */
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "ProcessOptions"
PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  options->dim         = 2;
  options->cellSimplex = PETSC_TRUE;
  options->filename[0] = '\0';
  options->overlap     = 0;

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex12.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cell_simplex", "Use simplices if true, otherwise hexes", "ex12.c", options->cellSimplex, &options->cellSimplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", "ex12.c", options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-overlap", "The cell overlap for partitioning", "ex12.c", options->overlap, &options->overlap, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
};

#undef __FUNCT__
#define __FUNCT__ "CreateMesh"
PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  DM             distMesh    = NULL;
  PetscInt       dim         = user->dim;
  PetscBool      cellSimplex = user->cellSimplex;
  const char    *filename    = user->filename;
  PetscInt       overlap     = user->overlap >= 0 ? user->overlap : 0;
  size_t         len;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrlen(filename, &len);CHKERRQ(ierr);
  if (len) {
    const char *extGmsh = ".msh";
    PetscBool   isGmsh;

    ierr = PetscStrncmp(&filename[PetscMax(0,len-4)], extGmsh, 4, &isGmsh);CHKERRQ(ierr);
    if (isGmsh) {
      PetscViewer viewer;

      ierr = PetscViewerCreate(comm, &viewer);CHKERRQ(ierr);
      ierr = PetscViewerSetType(viewer, PETSCVIEWERASCII);CHKERRQ(ierr);
      ierr = PetscViewerFileSetMode(viewer, FILE_MODE_READ);CHKERRQ(ierr);
      ierr = PetscViewerFileSetName(viewer, filename);CHKERRQ(ierr);
      ierr = DMPlexCreateGmsh(comm, viewer, PETSC_TRUE, dm);CHKERRQ(ierr);
      ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    } else {
      ierr = DMPlexCreateCGNSFromFile(comm, filename, PETSC_TRUE, dm);CHKERRQ(ierr);
    }
  } else if (cellSimplex) {
    ierr = DMPlexCreateBoxMesh(comm, dim, PETSC_TRUE, dm);CHKERRQ(ierr);
  } else {
    const PetscInt cells[3] = {2, 2, 2};

    ierr = DMPlexCreateHexBoxMesh(comm, dim, cells, PETSC_FALSE, PETSC_FALSE, PETSC_FALSE, dm);CHKERRQ(ierr);
  }
  ierr = DMPlexDistribute(*dm, NULL, overlap, NULL, &distMesh);CHKERRQ(ierr);
  if (distMesh) {
    ierr = DMDestroy(dm);CHKERRQ(ierr);
    *dm  = distMesh;
  }
  ierr = PetscObjectSetName((PetscObject) *dm, cellSimplex ? "Simplicial Mesh" : "Tensor Product Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#include <petscsf.h>

#undef __FUNCT__
#define __FUNCT__ "ParallelOverlap"
PetscErrorCode ParallelOverlap(DM dm, AppCtx *user)
{
  MPI_Comm           comm;
  PetscSF            sfPoint;
  PetscSection       rootSection, leafSection;
  IS                 rootrank, leafrank;
  PetscSF            overlapSF, migrationSF;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (user->overlap >= 0) PetscFunctionReturn(0);
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Calculating parallel overlap partition\n");CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedAllow(PETSC_VIEWER_STDOUT_WORLD, PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMGetPointSF(dm, &sfPoint);CHKERRQ(ierr);
  /* Make SF two-sided: Get owner information for shared points */
  ierr = PetscSectionCreate(comm, &rootSection);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) rootSection, "Root Section");CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm, &leafSection);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) leafSection, "Leaf Section");CHKERRQ(ierr);
  ierr = DMPlexDistributeOwnership(dm, rootSection, &rootrank, leafSection, &leafrank);CHKERRQ(ierr);
  {
    const PetscInt *rootdegree;
    PetscInt        pStart, pEnd, p;
    PetscMPIInt     rank;

    ierr = MPI_Comm_rank(comm, &rank);CHKERRQ(ierr);
    ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeBegin(sfPoint, &rootdegree);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeEnd(sfPoint, &rootdegree);CHKERRQ(ierr);
    ierr = PetscSynchronizedPrintf(comm, "Rank %d:", rank);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; ++p) {ierr = PetscSynchronizedPrintf(comm, " %d (%d)", p, rootdegree[p-pStart]);CHKERRQ(ierr);}
    ierr = PetscSynchronizedPrintf(comm, "\n");CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(comm, NULL);CHKERRQ(ierr);
    ierr = PetscSectionView(rootSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISView(rootrank, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = PetscSectionView(leafSection, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = ISView(leafrank, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  /* Collect cl(st(overlap)), use a DMLabel to separate the points by rank */
  ierr = DMPlexCreateOverlap(dm, rootSection, rootrank, leafSection, leafrank, &overlapSF);CHKERRQ(ierr);
  {
    ierr = PetscPrintf(comm, "Overlap SF\n");CHKERRQ(ierr);
    ierr = PetscSFView(overlapSF, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&rootrank);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&rootSection);CHKERRQ(ierr);
  ierr = ISDestroy(&leafrank);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&leafSection);CHKERRQ(ierr);
  /* Build migration SF that re-maps local points and adds remote ones */
  ierr = DMPlexCreateOverlapMigrationSF(dm, overlapSF, &migrationSF);CHKERRQ(ierr);
  {
    ierr = PetscPrintf(comm, "Overlap Migration SF\n");CHKERRQ(ierr);
    ierr = PetscSFView(migrationSF, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
#if 0

  \item Get closure of star and record depths

  \item Send points+cone sizes+depths to remote meshes

  \item Renumber points locally (looking up points in overlap to translate)

  \item Create new local mesh with room for new points at correct depths and for cones

  \item Send cones to remote meshes

  \item Fill in cones and symmetrize mesh

  \item recreate labels

  \item recreate SF
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user; /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = ParallelOverlap(dm, &user);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
