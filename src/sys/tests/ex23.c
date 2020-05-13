
static char help[] = "Tests string options with spaces";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscBool      ts_view       = PETSC_FALSE;
  PetscInt       ts_max_steps  = 0, snes_max_it = 0;
  PetscReal      ts_max_time = 0.;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsGetBool(NULL,0,"-ts_view",&ts_view,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,0,"-ts_max_time",&ts_max_time,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,0,"-ts_max_steps",&ts_max_steps,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,0,"-snes_max_it",&snes_max_it,NULL);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"-ts_view = %s\n-ts_max_time = %f\n-ts_max_steps = %i\n-snes_max_it = %i\n",ts_view ? "true" : "false",ts_max_time,ts_max_steps,snes_max_it);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      requires: yaml
      args: -options_file_yaml ex23options
      localrunfiles: ex23options

   test:
      suffix: string
      requires: yaml
      args: -options_string_yaml "
        ts:
          view: true
          max:
            steps: 10
            time: 1.4
        snes:
          max_it: 5"

TEST*/
