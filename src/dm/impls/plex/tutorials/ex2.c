static char help[] = "Read in a mesh and test whether it is valid\n\n";

#include <petscdmplex.h>
#if defined(PETSC_HAVE_CGNS)
#undef I /* Very old CGNS stupidly uses I as a variable, which fails when using complex. Curse you idiot package managers */
#include <cgnslib.h>
#endif
#if defined(PETSC_HAVE_EXODUSII)
#include <exodusII.h>
#endif

static PetscErrorCode zero(PetscInt dim, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt i;
  for (i = 0; i < Nc; ++i) u[i] = 0.0;
  return 0;
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  DMLabel        label;
  const PetscInt id = 1;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = CreateMesh(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = DMSetNumFields(dm, 1);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "boundary", &label);CHKERRQ(ierr);
  ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) zero, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  testset:
    args: -dm_plex_boundary_label boundary -dm_plex_check_all
    # CGNS meshes 0-1
    test:
      suffix: 0
      requires: cgns
      args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/tut21.cgns
    test:
      suffix: 1
      requires: cgns
      TODO: broken
      args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/grid_c.cgns
    # Gmsh meshes 2-4
    test:
      suffix: 2
      requires: double
      args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/doublet-tet.msh
    test:
      suffix: 3
      requires: double
      args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/square.msh
    test:
      suffix: 4
      requires: double
      args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/square_bin.msh
    # Exodus meshes 5-9
    test:
      suffix: 5
      requires: exodusii
      args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside-quad.exo
    test:
      suffix: 6
      requires: exodusii
      args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/sevenside-quad-15.exo
    test:
      suffix: 7
      requires: exodusii
      args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/squaremotor-30.exo
    test:
      suffix: 8
      requires: exodusii
      args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/blockcylinder-50.exo
    test:
      suffix: 9
      requires: exodusii
     args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/simpleblock-100.exo

TEST*/
