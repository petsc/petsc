
static char help[] = "Tests ISToGeneral().\n\n";

#include <petscis.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       step = 2;
  IS             is;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  ierr = PetscOptionsGetInt(NULL,NULL,"-step",&step,NULL);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,10,0,step,&is);CHKERRQ(ierr);

  ierr = ISToGeneral(is);CHKERRQ(ierr);

  ierr = ISDestroy(&is);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
     output_file: output/ex1_1.out

TEST*/
