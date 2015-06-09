
static char help[] = "Tests PetscAtan2Real\n";

#include <petscsys.h>
#include <petscviewer.h>
#include <petscmath.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{

  PetscReal a;
  PetscErrorCode ierr;

  PetscInitialize(&argc,&argv,(char*)0,help);
  a = PetscAtan2Real(1.0,1.0);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"PetscAtan2Real(1.0,1.0) = %g\n",(double)a);CHKERRQ(ierr);
  a = PetscAtan2Real(1.0,0.0);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"PetscAtan2Real(1.0,0.0) = %g\n",(double)a);CHKERRQ(ierr);
  a = PetscAtan2Real(0.0,1.0);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"PetscAtan2Real(0.0,1.0) = %g\n",(double)a);CHKERRQ(ierr);
  a = PetscAtan2Real(0.0,0.0);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"PetscAtan2Real(0.0,0.0) = %g\n",(double)a);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
