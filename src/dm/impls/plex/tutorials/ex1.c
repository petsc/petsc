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

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL,help));
  /* Create a mesh */
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(DMGetDimension(dm, &dim));
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
  PetscCall(DMGetStratumIS(dm, "marker", 1, &bcPointIS[0]));
  /* Create a PetscSection with this data layout */
  PetscCall(DMSetNumFields(dm, numFields));
  PetscCall(DMPlexCreateSection(dm, NULL, numComp, numDof, numBC, bcField, NULL, bcPointIS, NULL, &section));
  PetscCall(ISDestroy(&bcPointIS[0]));
  /* Name the Field variables */
  PetscCall(PetscSectionSetFieldName(section, 0, "u"));
  PetscCall(PetscSectionSetFieldName(section, 1, "v"));
  PetscCall(PetscSectionSetFieldName(section, 2, "w"));
  PetscCall(PetscSectionView(section, PETSC_VIEWER_STDOUT_WORLD));
  /* Tell the DM to use this data layout */
  PetscCall(DMSetLocalSection(dm, section));
  /* Create a Vec with this layout and view it */
  PetscCall(DMGetGlobalVector(dm, &u));
  PetscCall(PetscViewerCreate(PETSC_COMM_WORLD, &viewer));
  PetscCall(PetscViewerSetType(viewer, PETSCVIEWERVTK));
  PetscCall(PetscViewerFileSetName(viewer, "sol.vtu"));
  PetscCall(VecView(u, viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(DMRestoreGlobalVector(dm, &u));
  /* Cleanup */
  PetscCall(PetscSectionDestroy(&section));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
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
