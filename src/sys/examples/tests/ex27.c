
static char help[] = "Tests PetscMergeIntArray\n";

#include <petscsys.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  const PetscInt first[] = {0,2,3,5,8}, second[] = {1,3,4,8,10,11};
  PetscInt       *result,n;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscMergeIntArray(5,first,6,second,&n,&result);CHKERRQ(ierr);
  ierr = PetscIntView(n,result,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

