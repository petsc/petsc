static char help[] = "Element closure restrictions in tensor/lexicographic/spectral-element ordering using DMPlex\n\n";

#include <petscdmplex.h>

int main(int argc, char **argv)
{
  DM             dm;
  PetscSection   section;
  PetscFE        fe;
  PetscInt       cells[3] = {2, 2, 2},dim = 2,c,cStart,cEnd,tmp;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Spectral/tensor element restrictions",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsRangeInt("-dim","Topological dimension",NULL,dim,&dim,NULL,1,3);CHKERRQ(ierr);
  tmp = dim;
  ierr = PetscOptionsIntArray("-cells","Number of cells per dimension",NULL,cells,&tmp,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD,dim,PETSC_FALSE,cells,NULL,NULL,NULL,PETSC_TRUE,&dm);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PETSC_COMM_SELF,dim,1,PETSC_FALSE,NULL,PETSC_DETERMINE,&fe);CHKERRQ(ierr);
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);
  ierr = DMAddField(dm,NULL,(PetscObject)fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = DMPlexSetClosurePermutationTensor(dm,PETSC_DETERMINE,NULL);CHKERRQ(ierr);
  ierr = DMGetLocalSection(dm,&section);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm,0,&cStart,&cEnd);CHKERRQ(ierr);
  for (c=cStart; c<cEnd; c++) {
    PetscInt numindices,*indices;
    ierr = DMPlexGetClosureIndices(dm,section,section,c,&numindices,&indices,NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_SELF,"Element #%D\n",c-cStart);CHKERRQ(ierr);
    ierr = PetscIntView(numindices,indices,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    ierr = DMPlexRestoreClosureIndices(dm,section,section,c,&numindices,&indices,NULL);CHKERRQ(ierr);
  }

  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 1d_q2
    args: -dim 1 -petscspace_degree 2
  test:
    suffix: 2d_q1
    args: -dim 2 -petscspace_degree 1
  test:
    suffix: 2d_q2
    args: -dim 2 -petscspace_degree 2
  test:
    suffix: 2d_q3
    args: -dim 2 -petscspace_degree 3 -cells 1,1
  test:
    suffix: 3d_q1
    args: -dim 3 -petscspace_degree 1 -cells 1,1,1

TEST*/
