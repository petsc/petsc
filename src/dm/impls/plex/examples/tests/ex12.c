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
  PetscInt       overlap     = user->overlap;
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
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
