
static char help[] = "Tests deletion of mixed case options";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsSetValue(NULL,"-abc",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL,"-FOO",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsClearValue(NULL,"-FOO");CHKERRQ(ierr);
  ierr = PetscOptionsView(NULL,NULL);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      args: -skip_petscrc -options_left 0
      filter: egrep -v \(malloc_test\|saws_port_auto_select\|vecscatter_mpi1\)
TEST*/
