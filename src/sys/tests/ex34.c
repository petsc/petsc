static char help[] = "Tests IsInf/IsNan routines.\n";

#include <petscsys.h>

PETSC_INTERN PetscReal zero;
PetscReal              zero = 0;
PETSC_INTERN PetscReal zero2;
PetscReal              zero2 = 0;

#define CALL(call) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%-32s -> %s\n", #call, (call) ? "True" : "False"))

int main(int argc, char **argv)
{
  PetscReal neg_zero = PetscRealConstant(-0.0);
  PetscReal pos_zero = PetscRealConstant(+0.0);
  PetscReal neg_one  = PetscRealConstant(-1.0);
  PetscReal pos_one  = PetscRealConstant(+1.0);
  PetscReal neg_inf  = neg_one / zero;          /* -inf */
  PetscReal pos_inf  = pos_one / zero;          /* +inf */
  PetscReal x_nan    = zero2 / zero; /*  NaN */ /* some compilers may optimize out zero/zero and set x_nan = 1! */

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  CALL(PetscIsInfReal(neg_zero));
  CALL(PetscIsInfReal(pos_zero));
  CALL(PetscIsInfReal(neg_one));
  CALL(PetscIsInfReal(pos_one));
  CALL(PetscIsInfReal(neg_inf));
  CALL(PetscIsInfReal(pos_inf));
  CALL(PetscIsInfReal(x_nan));

  CALL(PetscIsNanReal(neg_zero));
  CALL(PetscIsNanReal(pos_zero));
  CALL(PetscIsNanReal(neg_one));
  CALL(PetscIsNanReal(pos_one));
  CALL(PetscIsNanReal(neg_inf));
  CALL(PetscIsNanReal(pos_inf));
  CALL(PetscIsNanReal(x_nan));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      output_file: output/ex34.out

TEST*/
