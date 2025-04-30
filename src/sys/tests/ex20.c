static const char help[] = "Tests PetscOptionsPrefix{Push,Pop} and PetscOptionsDeprecated\n\n";

#include <petscsys.h>

int main(int argc, char *argv[])
{
  PetscInt  opts[6] = {0};
  PetscBool hascl = PETSC_FALSE, hasstr = PETSC_FALSE;
  char      deprecated_prefix[PETSC_MAX_OPTION_NAME] = {0};
  PetscBool oldopt = PETSC_FALSE, newopt = PETSC_FALSE, useprefix;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, 0, help));
  PetscCall(PetscOptionsSetValue(NULL, "-zero", "0"));
  PetscCall(PetscOptionsPrefixPush(NULL, "a_"));
  PetscCall(PetscOptionsSetValue(NULL, "-one", "1"));
  PetscCall(PetscOptionsPrefixPush(NULL, "bb_"));
  PetscCall(PetscOptionsSetValue(NULL, "-two", "2"));
  PetscCall(PetscOptionsPrefixPop(NULL));
  PetscCall(PetscOptionsSetValue(NULL, "-three", "3"));
  PetscCall(PetscOptionsPrefixPush(NULL, "cc_"));
  PetscCall(PetscOptionsPrefixPush(NULL, "ddd_"));
  PetscCall(PetscOptionsSetValue(NULL, "-four", "4"));
  PetscCall(PetscOptionsPrefixPop(NULL));
  PetscCall(PetscOptionsPrefixPop(NULL));
  PetscCall(PetscOptionsPrefixPop(NULL));
  PetscCall(PetscOptionsSetValue(NULL, "-five", "5"));

  PetscCall(PetscOptionsGetInt(NULL, 0, "-zero", &opts[0], 0));
  PetscCall(PetscOptionsGetInt(NULL, 0, "-a_one", &opts[1], 0));
  PetscCall(PetscOptionsGetInt(NULL, 0, "-a_bb_two", &opts[2], 0));
  PetscCall(PetscOptionsGetInt(NULL, 0, "-a_three", &opts[3], 0));
  PetscCall(PetscOptionsGetInt(NULL, 0, "-a_cc_ddd_four", &opts[4], 0));
  PetscCall(PetscOptionsGetInt(NULL, 0, "-five", &opts[5], 0));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "opts = {%" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "}\n", opts[0], opts[1], opts[2], opts[3], opts[4], opts[5]));

  PetscCall(PetscOptionsGetBool(NULL, 0, "-cl", &hascl, 0));
  if (hascl) {
    PetscCall(PetscMemzero(opts, sizeof(opts)));
    PetscCall(PetscOptionsGetInt(NULL, 0, "-cl_zero", &opts[0], 0));
    PetscCall(PetscOptionsGetInt(NULL, 0, "-cl_a_one", &opts[1], 0));
    PetscCall(PetscOptionsGetInt(NULL, 0, "-cl_a_bb_two", &opts[2], 0));
    PetscCall(PetscOptionsGetInt(NULL, 0, "-cl_a_three", &opts[3], 0));
    PetscCall(PetscOptionsGetInt(NULL, 0, "-cl_a_cc_ddd_four", &opts[4], 0));
    PetscCall(PetscOptionsGetInt(NULL, 0, "-cl_five", &opts[5], 0));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "cl_opts = {%" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "}\n", opts[0], opts[1], opts[2], opts[3], opts[4], opts[5]));
  }

  PetscCall(PetscOptionsGetBool(NULL, 0, "-str", &hasstr, 0));
  if (hasstr) {
    PetscCall(
      PetscOptionsInsertString(NULL, "-prefix_push str_ -zero 100 -prefix_push a_ -one 101 -prefix_push bb_ -two 102 -prefix_pop -three 103 -prefix_push cc_ -prefix_push ddd_ -four 104 -prefix_pop -prefix_pop -prefix_pop -five 105 -prefix_pop"));
    PetscCall(PetscMemzero(opts, sizeof(opts)));
    PetscCall(PetscOptionsGetInt(NULL, 0, "-str_zero", &opts[0], 0));
    PetscCall(PetscOptionsGetInt(NULL, 0, "-str_a_one", &opts[1], 0));
    PetscCall(PetscOptionsGetInt(NULL, 0, "-str_a_bb_two", &opts[2], 0));
    PetscCall(PetscOptionsGetInt(NULL, 0, "-str_a_three", &opts[3], 0));
    PetscCall(PetscOptionsGetInt(NULL, 0, "-str_a_cc_ddd_four", &opts[4], 0));
    PetscCall(PetscOptionsGetInt(NULL, 0, "-str_five", &opts[5], 0));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "str_opts = {%" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT " %" PetscInt_FMT "}\n", opts[0], opts[1], opts[2], opts[3], opts[4], opts[5]));
  }

  PetscCall(PetscOptionsGetString(NULL, 0, "-deprecated_prefix", deprecated_prefix, sizeof(deprecated_prefix), &useprefix));
  PetscOptionsBegin(PETSC_COMM_WORLD, useprefix ? deprecated_prefix : NULL, "test deprecated options", NULL);
  PetscCall(PetscOptionsBool("-old_option", NULL, NULL, oldopt, &oldopt, NULL));
  PetscCall(PetscOptionsDeprecated("-old_option", "-new_option", "0.0", NULL));
  PetscCall(PetscOptionsBool("-new_option", NULL, NULL, newopt, &newopt, NULL));
  PetscOptionsEnd();
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "opts = {old %d new %d}\n", oldopt, newopt));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -old_option 1 -new_option 0

   test:
      suffix: 2
      args: -cl -prefix_push cl_ -zero 10 -prefix_push a_ -one 11 -prefix_push bb_ -two 12 -prefix_pop -three 13 -prefix_push cc_ -prefix_push ddd_ -four 14 -prefix_pop -prefix_pop -prefix_pop -five 15 -prefix_pop

   test:
      suffix: 3
      args: -str -deprecated_prefix zz_ -zz_old_option 0 -zz_new_option 1

   test:
      suffix: 4
      args: -deprecated_prefix yy_ -yy_old_option 1

TEST*/
