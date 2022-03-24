
static char help[] = "Tests PetscAtan2Real\n";

#include <petscsys.h>
#include <petscviewer.h>
#include <petscmath.h>

int main(int argc,char **argv)
{

  PetscReal a;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  a = PetscAtan2Real(1.0,1.0);
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"PetscAtan2Real(1.0,1.0) = %g\n",(double)a));
  a = PetscAtan2Real(1.0,0.0);
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"PetscAtan2Real(1.0,0.0) = %g\n",(double)a));
  a = PetscAtan2Real(0.0,1.0);
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"PetscAtan2Real(0.0,1.0) = %g\n",(double)a));
  a = PetscAtan2Real(0.0,0.0);
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"PetscAtan2Real(0.0,0.0) = %g\n",(double)a));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
