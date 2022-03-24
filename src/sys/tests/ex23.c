
static char help[] = "Tests string options with spaces";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscBool      ts_view       = PETSC_FALSE;
  PetscInt       ts_max_steps  = 0, snes_max_it = 0;
  PetscReal      ts_max_time = 0.;
  PetscBool      foo_view       = PETSC_FALSE;
  PetscInt       foo_max_steps  = 0, bar_max_it = 0;
  PetscReal      foo_max_time = 0.;

  CHKERRQ(PetscInitialize(&argc,&argv,NULL,help));
  CHKERRQ(PetscOptionsGetBool(NULL,0,"-ts_view",&ts_view,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,0,"-ts_max_time",&ts_max_time,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,0,"-ts_max_steps",&ts_max_steps,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,0,"-foo_view",&foo_view,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,0,"-foo_max_time",&foo_max_time,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,0,"-foo_max_steps",&foo_max_steps,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,0,"-snes_max_it",&snes_max_it,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,0,"-bar_max_it",&bar_max_it,NULL));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"-ts_view = %s\n-ts_max_time = %f\n-ts_max_steps = %" PetscInt_FMT "\n-snes_max_it = %" PetscInt_FMT "\n",ts_view ? "true" : "false",(double)ts_max_time,ts_max_steps,snes_max_it));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -options_file_yaml ex23options
      localrunfiles: ex23options

   test:
      suffix: string
      args: -options_string_yaml "
        foo: &foo
          view: true
          max: &foomax
            steps: 3
            time: 1.4
        bar: &bar
          max_it: 5
        ts:
          <<: *foo
          max:
            <<: *foomax
            steps: 10
        snes: *bar"

TEST*/
