
static char help[] = "Tests PetscAtan2Real\n";

#include <petscsys.h>
#include <petscviewer.h>
#include <petscmath.h>

int main(int argc,char **argv)
{

  PetscReal a;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  a = PetscAtan2Real(1.0,1.0);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"PetscAtan2Real(1.0,1.0) = %g\n",(double)a);CHKERRQ(ierr);
  a = PetscAtan2Real(1.0,0.0);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"PetscAtan2Real(1.0,0.0) = %g\n",(double)a);CHKERRQ(ierr);
  a = PetscAtan2Real(0.0,1.0);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"PetscAtan2Real(0.0,1.0) = %g\n",(double)a);CHKERRQ(ierr);
  a = PetscAtan2Real(0.0,0.0);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"PetscAtan2Real(0.0,0.0) = %g\n",(double)a);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
