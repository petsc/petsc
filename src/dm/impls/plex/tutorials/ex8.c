static char help[] = "Element closure restrictions in tensor/lexicographic/spectral-element ordering using DMPlex\n\n";

#include <petscdmplex.h>

int main(int argc, char **argv)
{
  DM             dm;
  PetscSection   section;
  PetscFE        fe;
  PetscInt       dim,c,cStart,cEnd;
  PetscBool      view_coord = PETSC_FALSE;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Tensor closure restrictions", "DMPLEX");CHKERRQ(ierr);
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
  ierr = DMPlexSetClosurePermutationTensor(dm,PETSC_DETERMINE,NULL);CHKERRQ(ierr);
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
    ierr = DMPlexSetClosurePermutationTensor(cdm,PETSC_DETERMINE,NULL);CHKERRQ(ierr);
    ierr = DMGetCoordinatesLocal(dm, &X);CHKERRQ(ierr);
    ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
    for (c=cStart; c<cEnd; c++) {
      PetscScalar *x;
      PetscInt ndof;
      ierr = DMPlexVecGetClosure(cdm, NULL, X, c, &ndof, &x);CHKERRQ(ierr);
      PetscCheck(ndof % cdim == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "ndof not divisible by cdim");
      ierr = PetscPrintf(PETSC_COMM_SELF,"Element #%" PetscInt_FMT " coordinates\n",c-cStart);CHKERRQ(ierr);
      for (PetscInt i=0; i<ndof; i+= cdim) {
        ierr = PetscScalarView(cdim, &x[i], PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
      }
      ierr = DMPlexVecRestoreClosure(cdm, NULL, X, c, &ndof, &x);CHKERRQ(ierr);
    }
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
    args: -dm_plex_dim 1 -petscspace_degree 1 -dm_plex_simplex 0 -dm_plex_box_faces 2 -dm_plex_box_bd periodic -dm_view
  test:
    suffix: 2d_q1_periodic
    args: -dm_plex_dim 2 -petscspace_degree 1 -dm_plex_simplex 0 -dm_plex_box_faces 2,2 -dm_plex_box_bd periodic,none -dm_view
  test:
    suffix: 3d_q2_periodic
    args: -dm_plex_dim 3 -petscspace_degree 2 -dm_plex_simplex 0 -dm_plex_box_faces 2,2,2 -dm_plex_box_bd periodic,none,periodic -dm_view

TEST*/
