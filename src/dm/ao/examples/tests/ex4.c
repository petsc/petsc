
static char help[] = "Test AO with on IS with 0 entries - contributed by Ethan Coon <ecoon@lanl.gov>, Apr 2011.\n\n";

#include <petscsys.h>
#include <petscao.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  AO             ao;
  PetscInt       *localvert=PETSC_NULL, nlocal, rank;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscMalloc(4*sizeof(PetscInt),&localvert);CHKERRQ(ierr);

  if (!rank) {
    nlocal = 4;
    localvert[0] = 0;
    localvert[1] = 1;
    localvert[2] = 2;
    localvert[3] = 3;
  } else {
    nlocal = 0;
  }

  /* Test AOCreateBasic() */
  ierr = AOCreateBasic(PETSC_COMM_WORLD, nlocal, localvert, PETSC_NULL, &ao);CHKERRQ(ierr);
  ierr = AODestroy(&ao);CHKERRQ(ierr);

  /* Test AOCreateMemoryScalable() */
  ierr = AOCreateMemoryScalable(PETSC_COMM_WORLD, nlocal, localvert, PETSC_NULL, &ao);CHKERRQ(ierr);
  ierr = AODestroy(&ao);CHKERRQ(ierr);

  ierr = PetscFree(localvert);CHKERRQ(ierr);
  ierr=PetscFinalize();
  PetscFunctionReturn(0);
}
