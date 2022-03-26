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
  PetscCall(PetscObjectGetName((PetscObject)dm, &name));
  PetscCall(DMPlexGetLocalOffsets(dm, NULL, 0, 0, 0, &num_elem, &elem_size, &num_comp, &num_dof, &elem_restr_offsets));
  ierr = PetscPrintf(PETSC_COMM_SELF,"DM %s offsets: num_elem %" PetscInt_FMT ", size %" PetscInt_FMT
                     ", comp %" PetscInt_FMT ", dof %" PetscInt_FMT "\n",
                     name, num_elem, elem_size, num_comp, num_dof);PetscCall(ierr);
  if (X) PetscCall(VecGetArrayRead(X, &x));
  for (PetscInt c=0; c<num_elem; c++) {
    PetscCall(PetscIntView(elem_size, &elem_restr_offsets[c*elem_size], PETSC_VIEWER_STDOUT_SELF));
    if (x) {
      for (PetscInt i=0; i<elem_size; i++) {
        PetscCall(PetscScalarView(num_comp, &x[elem_restr_offsets[c*elem_size+i]], PETSC_VIEWER_STDERR_SELF));
      }
    }
  }
  if (X) PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(PetscFree(elem_restr_offsets));
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

  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Tensor closure restrictions", "DMPLEX");PetscCall(ierr);
  PetscCall(PetscOptionsBool("-closure_tensor", "Apply DMPlexSetClosurePermutationTensor", "ex8.c", tensor, &tensor, NULL));
  PetscCall(PetscOptionsBool("-view_coord", "View coordinates of element closures", "ex8.c", view_coord, &view_coord, NULL));
  ierr = PetscOptionsEnd();PetscCall(ierr);

  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(DMGetDimension(dm, &dim));

  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF,dim,1,PETSC_FALSE,NULL,PETSC_DETERMINE,&fe));
  PetscCall(DMAddField(dm,NULL,(PetscObject)fe));
  PetscCall(DMCreateDS(dm));
  if (tensor) PetscCall(DMPlexSetClosurePermutationTensor(dm,PETSC_DETERMINE,NULL));
  PetscCall(DMGetLocalSection(dm,&section));
  PetscCall(DMPlexGetHeightStratum(dm,0,&cStart,&cEnd));
  for (c=cStart; c<cEnd; c++) {
    PetscInt numindices,*indices;
    PetscCall(DMPlexGetClosureIndices(dm,section,section,c,PETSC_TRUE,&numindices,&indices,NULL,NULL));
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Element #%" PetscInt_FMT "\n",c-cStart));
    PetscCall(PetscIntView(numindices,indices,PETSC_VIEWER_STDOUT_SELF));
    PetscCall(DMPlexRestoreClosureIndices(dm,section,section,c,PETSC_TRUE,&numindices,&indices,NULL,NULL));
  }
  if (view_coord) {
    DM cdm;
    Vec X;
    PetscInt cdim;
    PetscCall(DMGetCoordinateDM(dm,&cdm));
    PetscCall(PetscObjectSetName((PetscObject)cdm, "coords"));
    if (tensor) PetscCall(DMPlexSetClosurePermutationTensor(cdm,PETSC_DETERMINE,NULL));
    PetscCall(DMGetCoordinatesLocal(dm, &X));
    PetscCall(DMGetCoordinateDim(dm, &cdim));
    for (c=cStart; c<cEnd; c++) {
      PetscScalar *x = NULL;
      PetscInt ndof;
      PetscCall(DMPlexVecGetClosure(cdm, NULL, X, c, &ndof, &x));
      PetscCheck(ndof % cdim == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "ndof not divisible by cdim");
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"Element #%" PetscInt_FMT " coordinates\n",c-cStart));
      for (PetscInt i=0; i<ndof; i+= cdim) {
        PetscCall(PetscScalarView(cdim, &x[i], PETSC_VIEWER_STDOUT_SELF));
      }
      PetscCall(DMPlexVecRestoreClosure(cdm, NULL, X, c, &ndof, &x));
    }
    PetscCall(ViewOffsets(dm, NULL));
    PetscCall(ViewOffsets(cdm, X));
  }
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
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
