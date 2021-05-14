static char help[] = "Define a simple field over the mesh\n\n";

#include <petscdmplex.h>

int main(int argc, char **argv)
{
  DM             dm;
  Vec            u;
  PetscSection   section;
  PetscViewer    viewer;
  PetscInt       dim, numFields, numBC, i;
  PetscInt       numComp[3];
  PetscInt       numDof[12];
  PetscInt       bcField[1];
  IS             bcPointIS[1];
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  /* Create a mesh */
  ierr = DMCreate(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  /* Create a scalar field u, a vector field v, and a surface vector field w */
  numFields  = 3;
  numComp[0] = 1;
  numComp[1] = dim;
  numComp[2] = dim-1;
  for (i = 0; i < numFields*(dim+1); ++i) numDof[i] = 0;
  /* Let u be defined on vertices */
  numDof[0*(dim+1)+0]     = 1;
  /* Let v be defined on cells */
  numDof[1*(dim+1)+dim]   = dim;
  /* Let w be defined on faces */
  numDof[2*(dim+1)+dim-1] = dim-1;
  /* Setup boundary conditions */
  numBC = 1;
  /* Prescribe a Dirichlet condition on u on the boundary
       Label "marker" is made by the mesh creation routine */
  bcField[0] = 0;
  ierr = DMGetStratumIS(dm, "marker", 1, &bcPointIS[0]);CHKERRQ(ierr);
  /* Create a PetscSection with this data layout */
  ierr = DMSetNumFields(dm, numFields);CHKERRQ(ierr);
  ierr = DMPlexCreateSection(dm, NULL, numComp, numDof, numBC, bcField, NULL, bcPointIS, NULL, &section);CHKERRQ(ierr);
  ierr = ISDestroy(&bcPointIS[0]);CHKERRQ(ierr);
  /* Name the Field variables */
  ierr = PetscSectionSetFieldName(section, 0, "u");CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(section, 1, "v");CHKERRQ(ierr);
  ierr = PetscSectionSetFieldName(section, 2, "w");CHKERRQ(ierr);
  ierr = PetscSectionView(section, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /* Tell the DM to use this data layout */
  ierr = DMSetLocalSection(dm, section);CHKERRQ(ierr);
  /* Create a Vec with this layout and view it */
  ierr = DMGetGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = PetscViewerCreate(PETSC_COMM_WORLD, &viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(viewer, PETSCVIEWERVTK);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(viewer, "sol.vtu");CHKERRQ(ierr);
  ierr = VecView(u, viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &u);CHKERRQ(ierr);
  /* Cleanup */
  ierr = PetscSectionDestroy(&section);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    requires: triangle
    args: -info :~sys,mat
  test:
    suffix: 1
    requires: ctetgen
    args: -dm_plex_dim 3 -info :~sys,mat

TEST*/
