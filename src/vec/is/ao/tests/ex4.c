
static char help[] = "Test AO with on IS with 0 entries - contributed by Ethan Coon <ecoon@lanl.gov>, Apr 2011.\n\n";

#include <petscsys.h>
#include <petscao.h>

int main(int argc,char **argv)
{
  AO             ao;
  PetscInt       *localvert=NULL, nlocal;
  PetscMPIInt    rank;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(PetscMalloc1(4,&localvert));

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
  CHKERRQ(AOCreateBasic(PETSC_COMM_WORLD, nlocal, localvert, NULL, &ao));
  CHKERRQ(AODestroy(&ao));

  /* Test AOCreateMemoryScalable() */
  CHKERRQ(AOCreateMemoryScalable(PETSC_COMM_WORLD, nlocal, localvert, NULL, &ao));
  CHKERRQ(AODestroy(&ao));

  CHKERRQ(PetscFree(localvert));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

   test:
      suffix: 2
      nsize: 2
      output_file: output/ex4_1.out

TEST*/
