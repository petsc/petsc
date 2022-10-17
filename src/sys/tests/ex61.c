static const char help[] = "Tests env: directive in test harness language.\n\n";

#include <petscsys.h>

int main(int argc, char *argv[])
{
  PetscBool env_set;
  char      env_vars[PETSC_MAX_PATH_LEN];
  int       num_env;
  char    **env_vars_arr;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscArrayzero(env_vars, PETSC_MAX_PATH_LEN));

  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Test Options", NULL);
  PetscCall(PetscOptionsString("-env_vars_def", "Environment variables set", NULL, env_vars, env_vars, sizeof(env_vars), &env_set));
  PetscOptionsEnd();

  PetscCall(PetscStrToArray(env_vars, ' ', &num_env, &env_vars_arr));
  for (int i = 0; i < num_env; ++i) {
    const char *current_var = env_vars_arr[i];
    PetscBool   set, equal;
    size_t      name_len;
    char        env[PETSC_MAX_PATH_LEN];
    char       *name, *value;

    // given FOO=bar we want to extract
    // name = FOO
    // value = bar
    PetscCall(PetscStrchr(current_var, '=', &value));
    PetscCheck(value, PETSC_COMM_SELF, PETSC_ERR_PLIB, "= not found in %s", current_var);
    PetscCheck(value >= current_var, PETSC_COMM_SELF, PETSC_ERR_PLIB, "= not found in %s", current_var);
    // value points to '=' so increment it first
    ++value;

    name_len = (size_t)(value - current_var);
    PetscCall(PetscMalloc1(name_len, &name));
    PetscCall(PetscStrncpy(name, env_vars_arr[i], name_len));

    PetscCall(PetscArrayzero(env, PETSC_MAX_PATH_LEN));
    PetscCall(PetscOptionsGetenv(PETSC_COMM_WORLD, name, env, sizeof(env), &set));
    PetscCheck(set, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Test harness failed to set %s", name);
    PetscCall(PetscStrcmp(value, env, &equal));
    PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Test harness failed to set %s to the right value. Expected '%s', have '%s'", name, value, env);
    PetscCall(PetscFree(name));
  }
  PetscCall(PetscStrToArrayDestroy(num_env, env_vars_arr));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    output_file: ./output/empty.out
    args: -env_vars_def 'FOO=1 BAR=0 BAZ= BOP=1'
    suffix: env_set
    test:
      env: FOO=1 BAR=0 BAZ= BOP=${FOO}
      suffix: all_one_line
    test:
      env: FOO=1
      env: BAR=0
      env: BAZ=
      env: BOP=${FOO}
      suffix: all_seperate_lines

  test:
    output_file: ./output/empty.out
    args: -env_vars_def 'FOO=hello'
    env: FOO='hello'
    suffix: env_set_quoted

  test:
    output_file: ./output/empty.out
    suffix: env_not_set

TEST*/
