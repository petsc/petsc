
static char help[] = "Test AO with on IS with 0 entries - contributed by Ethan Coon <ecoon@lanl.gov>, Apr 2011.\n\n";

#include <petscsys.h>
#include <petscao.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  AO             ao;
  PetscInt       *localvert=NULL, nlocal;
  PetscMPIInt    rank;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  ierr = PetscMalloc1(4,&localvert);CHKERRQ(ierr);

  if (rank == 0) {
    nlocal       = 4;
    localvert[0] = 0;
    localvert[1] = 1;
    localvert[2] = 2;
    localvert[3] = 3;
  } else {
    nlocal = 0;
  }

  /* Test AOCreateBasic() */
  ierr = AOCreateBasic(PETSC_COMM_WORLD, nlocal, localvert, NULL, &ao);CHKERRQ(ierr);
  ierr = AODestroy(&ao);CHKERRQ(ierr);

  /* Test AOCreateMemoryScalable() */
  ierr = AOCreateMemoryScalable(PETSC_COMM_WORLD, nlocal, localvert, NULL, &ao);CHKERRQ(ierr);
  ierr = AODestroy(&ao);CHKERRQ(ierr);

  ierr = PetscFree(localvert);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

   test:
      suffix: 2
      nsize: 2
      output_file: output/ex4_1.out

TEST*/
