
static char help[] = "Tests string options with spaces";

#include <petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscBool      ts_view = PETSC_FALSE;
  PetscInt       ts_max_steps = 0, snes_max_steps = 0;
  PetscReal      ts_final_time = 0.;

  PetscInitialize(&argc,&argv,PETSC_NULL,help);
  ierr = PetscOptionsGetBool(0,"-ts_view",&ts_view,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(0,"-ts_final_time",&ts_final_time,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(0,"-ts_max_steps",&ts_max_steps,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(0,"-snes_max_steps",&snes_max_steps,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"-ts_view = %s\n-ts_final_time = %f\n-ts_max_steps = %i\n-snes_max_steps = %i\n",ts_view?"true":"false",ts_final_time,ts_max_steps,snes_max_steps);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
