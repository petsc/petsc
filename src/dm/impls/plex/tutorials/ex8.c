static char help[] = "Element closure restrictions in tensor/lexicographic/spectral-element ordering using DMPlex\n\n";

#include <petscdmplex.h>

static PetscErrorCode ViewOffsets(DM dm, Vec X)
{
  PetscInt num_elem, elem_size, num_comp, num_dof;
  PetscInt *elem_restr_offsets;
  const PetscScalar *x = NULL;
  const char *name;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectGetName((PetscObject)dm, &name));
  CHKERRQ(DMPlexGetLocalOffsets(dm, NULL, 0, 0, 0, &num_elem, &elem_size, &num_comp, &num_dof, &elem_restr_offsets));
  ierr = PetscPrintf(PETSC_COMM_SELF,"DM %s offsets: num_elem %" PetscInt_FMT ", size %" PetscInt_FMT
                     ", comp %" PetscInt_FMT ", dof %" PetscInt_FMT "\n",
                     name, num_elem, elem_size, num_comp, num_dof);CHKERRQ(ierr);
  if (X) CHKERRQ(VecGetArrayRead(X, &x));
  for (PetscInt c=0; c<num_elem; c++) {
    CHKERRQ(PetscIntView(elem_size, &elem_restr_offsets[c*elem_size], PETSC_VIEWER_STDOUT_SELF));
    if (x) {
      for (PetscInt i=0; i<elem_size; i++) {
        CHKERRQ(PetscScalarView(num_comp, &x[elem_restr_offsets[c*elem_size+i]], PETSC_VIEWER_STDERR_SELF));
      }
    }
  }
  if (X) CHKERRQ(VecRestoreArrayRead(X, &x));
  CHKERRQ(PetscFree(elem_restr_offsets));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  PetscSection   section;
  PetscFE        fe;
  PetscInt       dim,c,cStart,cEnd;
  PetscBool      view_coord = PETSC_FALSE, tensor = PETSC_TRUE;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Tensor closure restrictions", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsBool("-closure_tensor", "Apply DMPlexSetClosurePermutationTensor", "ex8.c", tensor, &tensor, NULL));
  CHKERRQ(PetscOptionsBool("-view_coord", "View coordinates of element closures", "ex8.c", view_coord, &view_coord, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  CHKERRQ(DMCreate(PETSC_COMM_WORLD, &dm));
  CHKERRQ(DMSetType(dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));
  CHKERRQ(DMGetDimension(dm, &dim));

  CHKERRQ(PetscFECreateDefault(PETSC_COMM_SELF,dim,1,PETSC_FALSE,NULL,PETSC_DETERMINE,&fe));
  CHKERRQ(DMAddField(dm,NULL,(PetscObject)fe));
  CHKERRQ(DMCreateDS(dm));
  if (tensor) CHKERRQ(DMPlexSetClosurePermutationTensor(dm,PETSC_DETERMINE,NULL));
  CHKERRQ(DMGetLocalSection(dm,&section));
  CHKERRQ(DMPlexGetHeightStratum(dm,0,&cStart,&cEnd));
  for (c=cStart; c<cEnd; c++) {
    PetscInt numindices,*indices;
    CHKERRQ(DMPlexGetClosureIndices(dm,section,section,c,PETSC_TRUE,&numindices,&indices,NULL,NULL));
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Element #%" PetscInt_FMT "\n",c-cStart));
    CHKERRQ(PetscIntView(numindices,indices,PETSC_VIEWER_STDOUT_SELF));
    CHKERRQ(DMPlexRestoreClosureIndices(dm,section,section,c,PETSC_TRUE,&numindices,&indices,NULL,NULL));
  }
  if (view_coord) {
    DM cdm;
    Vec X;
    PetscInt cdim;
    CHKERRQ(DMGetCoordinateDM(dm,&cdm));
    CHKERRQ(PetscObjectSetName((PetscObject)cdm, "coords"));
    if (tensor) CHKERRQ(DMPlexSetClosurePermutationTensor(cdm,PETSC_DETERMINE,NULL));
    CHKERRQ(DMGetCoordinatesLocal(dm, &X));
    CHKERRQ(DMGetCoordinateDim(dm, &cdim));
    for (c=cStart; c<cEnd; c++) {
      PetscScalar *x = NULL;
      PetscInt ndof;
      CHKERRQ(DMPlexVecGetClosure(cdm, NULL, X, c, &ndof, &x));
      PetscCheck(ndof % cdim == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "ndof not divisible by cdim");
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Element #%" PetscInt_FMT " coordinates\n",c-cStart));
      for (PetscInt i=0; i<ndof; i+= cdim) {
        CHKERRQ(PetscScalarView(cdim, &x[i], PETSC_VIEWER_STDOUT_SELF));
      }
      CHKERRQ(DMPlexVecRestoreClosure(cdm, NULL, X, c, &ndof, &x));
    }
    CHKERRQ(ViewOffsets(dm, NULL));
    CHKERRQ(ViewOffsets(cdm, X));
  }
  CHKERRQ(PetscFEDestroy(&fe));
  CHKERRQ(DMDestroy(&dm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 1d_q2
    args: -dm_plex_dim 1 -petscspace_degree 2 -dm_plex_simplex 0 -dm_plex_box_faces 2
  test:
    suffix: 2d_q1
    args: -dm_plex_dim 2 -petscspace_degree 1 -dm_plex_simplex 0 -dm_plex_box_faces 2,2
  test:
    suffix: 2d_q2
    args: -dm_plex_dim 2 -petscspace_degree 2 -dm_plex_simplex 0 -dm_plex_box_faces 2,2
  test:
    suffix: 2d_q3
    args: -dm_plex_dim 2 -petscspace_degree 3 -dm_plex_simplex 0 -dm_plex_box_faces 1,1
  test:
    suffix: 3d_q1
    args: -dm_plex_dim 3 -petscspace_degree 1 -dm_plex_simplex 0 -dm_plex_box_faces 1,1,1
  test:
    suffix: 1d_q1_periodic
    requires: !complex
    args: -dm_plex_dim 1 -petscspace_degree 1 -dm_plex_simplex 0 -dm_plex_box_faces 3 -dm_plex_box_bd periodic -dm_view -view_coord
  test:
    suffix: 2d_q1_periodic
    requires: !complex
    args: -dm_plex_dim 2 -petscspace_degree 1 -dm_plex_simplex 0 -dm_plex_box_faces 3,2 -dm_plex_box_bd periodic,none -dm_view -view_coord
  test:
    suffix: 3d_q1_periodic
    requires: !complex
    args: -dm_plex_dim 3 -petscspace_degree 1 -dm_plex_simplex 0 -dm_plex_box_faces 3,2,1 -dm_plex_box_bd periodic,none,none -dm_view -view_coord
  test:
    suffix: 3d_q2_periodic  # not actually periodic because only 2 cells
    args: -dm_plex_dim 3 -petscspace_degree 2 -dm_plex_simplex 0 -dm_plex_box_faces 2,2,2 -dm_plex_box_bd periodic,none,periodic -dm_view

TEST*/
