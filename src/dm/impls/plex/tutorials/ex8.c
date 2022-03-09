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
  ierr = PetscObjectGetName((PetscObject)dm, &name);CHKERRQ(ierr);
  ierr = DMPlexGetLocalOffsets(dm, NULL, 0, 0, 0, &num_elem, &elem_size, &num_comp, &num_dof, &elem_restr_offsets);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"DM %s offsets: num_elem %" PetscInt_FMT ", size %" PetscInt_FMT
                     ", comp %" PetscInt_FMT ", dof %" PetscInt_FMT "\n",
                     name, num_elem, elem_size, num_comp, num_dof);CHKERRQ(ierr);
  if (X) {ierr = VecGetArrayRead(X, &x);CHKERRQ(ierr);}
  for (PetscInt c=0; c<num_elem; c++) {
    ierr = PetscIntView(elem_size, &elem_restr_offsets[c*elem_size], PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    if (x) {
      for (PetscInt i=0; i<elem_size; i++) {
        ierr = PetscScalarView(num_comp, &x[elem_restr_offsets[c*elem_size+i]], PETSC_VIEWER_STDERR_SELF);CHKERRQ(ierr);
      }
    }
  }
  if (X) {ierr = VecRestoreArrayRead(X, &x);CHKERRQ(ierr);}
  ierr = PetscFree(elem_restr_offsets);CHKERRQ(ierr);
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
  ierr = PetscOptionsBool("-closure_tensor", "Apply DMPlexSetClosurePermutationTensor", "ex8.c", tensor, &tensor, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-view_coord", "View coordinates of element closures", "ex8.c", view_coord, &view_coord, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = DMCreate(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = DMSetType(dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);

  ierr = PetscFECreateDefault(PETSC_COMM_SELF,dim,1,PETSC_FALSE,NULL,PETSC_DETERMINE,&fe);CHKERRQ(ierr);
  ierr = DMAddField(dm,NULL,(PetscObject)fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  if (tensor) {ierr = DMPlexSetClosurePermutationTensor(dm,PETSC_DETERMINE,NULL);CHKERRQ(ierr);}
  ierr = DMGetLocalSection(dm,&section);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  for (c=cStart; c<cEnd; c++) {
    PetscInt numindices,*indices;
    ierr = DMPlexGetClosureIndices(dm,section,section,c,PETSC_TRUE,&numindices,&indices,NULL,NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"Element #%" PetscInt_FMT "\n",c-cStart);CHKERRQ(ierr);
    ierr = PetscIntView(numindices,indices,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    ierr = DMPlexRestoreClosureIndices(dm,section,section,c,PETSC_TRUE,&numindices,&indices,NULL,NULL);CHKERRQ(ierr);
  }
  if (view_coord) {
    DM cdm;
    Vec X;
    PetscInt cdim;
    ierr = DMGetCoordinateDM(dm,&cdm);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject)cdm, "coords");CHKERRQ(ierr);
    if (tensor) {ierr = DMPlexSetClosurePermutationTensor(cdm,PETSC_DETERMINE,NULL);CHKERRQ(ierr);}
    ierr = DMGetCoordinatesLocal(dm, &X);CHKERRQ(ierr);
    ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
    for (c=cStart; c<cEnd; c++) {
      PetscScalar *x = NULL;
      PetscInt ndof;
      ierr = DMPlexVecGetClosure(cdm, NULL, X, c, &ndof, &x);CHKERRQ(ierr);
      PetscCheck(ndof % cdim == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "ndof not divisible by cdim");
      ierr = PetscPrintf(PETSC_COMM_SELF,"Element #%" PetscInt_FMT " coordinates\n",c-cStart);CHKERRQ(ierr);
      for (PetscInt i=0; i<ndof; i+= cdim) {
        ierr = PetscScalarView(cdim, &x[i], PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
      }
      ierr = DMPlexVecRestoreClosure(cdm, NULL, X, c, &ndof, &x);CHKERRQ(ierr);
    }
    ierr = ViewOffsets(dm, NULL);CHKERRQ(ierr);
    ierr = ViewOffsets(cdm, X);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
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
    args: -dm_plex_dim 1 -petscspace_degree 1 -dm_plex_simplex 0 -dm_plex_box_faces 3 -dm_plex_box_bd periodic -dm_view -view_coord
  test:
    suffix: 2d_q1_periodic
    args: -dm_plex_dim 2 -petscspace_degree 1 -dm_plex_simplex 0 -dm_plex_box_faces 3,2 -dm_plex_box_bd periodic,none -dm_view -view_coord
  test:
    suffix: 3d_q2_periodic
    args: -dm_plex_dim 3 -petscspace_degree 2 -dm_plex_simplex 0 -dm_plex_box_faces 2,2,2 -dm_plex_box_bd periodic,none,periodic -dm_view

TEST*/
