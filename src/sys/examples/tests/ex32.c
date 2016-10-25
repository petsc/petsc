
static char help[] = "Tests deletion of mixed case options";

#include <petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;

  PetscInitialize(&argc,&argv,NULL,help);
  ierr = PetscOptionsSetValue(NULL,"-abc",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsSetValue(NULL,"-FOO",NULL);CHKERRQ(ierr);
  ierr = PetscOptionsClearValue(NULL,"-FOO");CHKERRQ(ierr);
  ierr = PetscOptionsView(NULL,NULL);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
