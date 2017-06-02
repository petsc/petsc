static char help[] = "Read in a mesh and test whether it is valid\n\n";

#include <petscdmplex.h>
#if defined(PETSC_HAVE_CGNS)
#undef I /* Very old CGNS stupidly uses I as a variable, which fails when using complex. Curse you idiot package managers */
#include <cgnslib.h>
#endif
#if defined(PETSC_HAVE_EXODUSII)
#include <exodusII.h>
#endif

typedef struct {
  PetscBool interpolate;                  /* Generate intermediate mesh elements */
  char      filename[PETSC_MAX_PATH_LEN]; /* Mesh filename */
  PetscInt  dim;
  PetscErrorCode (**bcFuncs)(PetscInt dim, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
} AppCtx;

static PetscErrorCode zero(PetscInt dim, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  PetscInt i;
  for (i = 0; i < dim; ++i) u[i] = 0.0;
  return 0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->interpolate = PETSC_TRUE;
  options->filename[0] = '\0';
  options->dim         = 2;
  options->bcFuncs     = NULL;

  ierr = PetscOptionsBegin(comm, "", "Meshing Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-interpolate", "Generate intermediate mesh elements", "ex2.c", options->interpolate, &options->interpolate, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-filename", "The mesh file", "ex2.c", options->filename, options->filename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The dimension of problem used for non-file mesh", "ex2.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  size_t         len;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscStrlen(user->filename, &len);CHKERRQ(ierr);
  if (!len) {
    DMLabel  label;
    PetscInt id = 1;

    ierr = DMPlexCreateBoxMesh(comm, user->dim, 2, user->interpolate, dm);CHKERRQ(ierr);
    /* Mark boundary and set BC */
    ierr = DMCreateLabel(*dm, "boundary");CHKERRQ(ierr);
    ierr = DMGetLabel(*dm, "boundary", &label);CHKERRQ(ierr);
    ierr = DMPlexMarkBoundaryFaces(*dm, label);CHKERRQ(ierr);
    ierr = DMPlexLabelComplete(*dm, label);CHKERRQ(ierr);
    ierr = PetscMalloc1(1, &user->bcFuncs);CHKERRQ(ierr);
    user->bcFuncs[0] = zero;
    ierr = DMAddBoundary(*dm, DM_BC_ESSENTIAL, "wall", "boundary", 0, 0, NULL, (void (*)(void)) user->bcFuncs[0], 1, &id, user);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateFromFile(comm, user->filename, user->interpolate, dm);CHKERRQ(ierr);
  }
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckMeshTopology(DM dm)
{
  PetscInt       dim, coneSize, cStart;
  PetscBool      isSimplex;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dm, cStart, &coneSize);CHKERRQ(ierr);
  isSimplex = coneSize == dim+1 ? PETSC_TRUE : PETSC_FALSE;
  ierr = DMPlexCheckSymmetry(dm);CHKERRQ(ierr);
  ierr = DMPlexCheckSkeleton(dm, isSimplex, 0);CHKERRQ(ierr);
  ierr = DMPlexCheckFaces(dm, isSimplex, 0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckMeshGeometry(DM dm)
{
  PetscInt       dim, coneSize, cStart, cEnd, c;
  PetscReal     *v0, *J, *invJ, detJ;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMPlexGetConeSize(dm, cStart, &coneSize);CHKERRQ(ierr);
  ierr = PetscMalloc3(dim,&v0,dim*dim,&J,dim*dim,&invJ);CHKERRQ(ierr);
  for (c = cStart; c < cEnd; ++c) {
    ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
    if (detJ <= 0.0) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Invalid determinant %g for cell %D", (double)detJ, c);
  }
  ierr = PetscFree3(v0,J,invJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = CheckMeshTopology(dm);CHKERRQ(ierr);
  ierr = CheckMeshGeometry(dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFree(user.bcFuncs);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  # CGNS meshes 0-1
  test:
    suffix: 0
    requires: cgns
    TODO: broken
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/tut21.cgns -interpolate 1
  test:
    suffix: 1
    requires: cgns
    TODO: broken
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/grid_c.cgns -interpolate 1
  # Gmsh meshes 2-4
  test:
    suffix: 2
    requires: double
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/doublet-tet.msh -interpolate 1
  test:
    suffix: 3
    requires: double
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/square.msh -interpolate 1
  test:
    suffix: 4
    requires: double
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/square_bin.msh -interpolate 1
  # Exodus meshes 5-9
  test:
    suffix: 5
    requires: exodusii
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside-quad.exo -interpolate 1
  test:
    suffix: 6
    requires: exodusii
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside-quad-15.exo -interpolate 1
  test:
    suffix: 7
    requires: exodusii
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/squaremotor-30.exo -interpolate 1
  test:
    suffix: 8
    requires: exodusii
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo -interpolate 1
  test:
    suffix: 9
    requires: exodusii
    args: -filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/simpleblock-100.exo -interpolate 1

TEST*/
