static char help[] = "Tests PetscIsCloseAtTol() routine.\n";

#include <petscsys.h>

PETSC_INTERN PetscReal zero;
PetscReal zero = 0;

#define CALL(call) do { \
    PetscErrorCode _ierr;                                               \
    _ierr = PetscPrintf(PETSC_COMM_WORLD,"%s -> %s\n",#call,(call)?"True":"False");CHKERRQ(_ierr); \
  } while(0);

int main(int argc, char **argv) {

  PetscReal eps      = PETSC_MACHINE_EPSILON;
  PetscReal neg_zero = PetscRealConstant(-0.0);
  PetscReal pos_zero = PetscRealConstant(+0.0);
  PetscReal neg_one  = PetscRealConstant(-1.0);
  PetscReal pos_one  = PetscRealConstant(+1.0);
  PetscReal neg_inf  = neg_one/zero; /* -inf */
  PetscReal pos_inf  = pos_one/zero; /* +inf */
  PetscReal x_nan    = zero/zero;    /*  NaN */

  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;

  CALL(PetscIsCloseAtTol(pos_zero,neg_zero,0,0));
  CALL(PetscIsCloseAtTol(pos_one,pos_one,0,0));
  CALL(PetscIsCloseAtTol(pos_one,neg_one,0,0));
  CALL(PetscIsCloseAtTol(pos_one,neg_one,0,2));
  CALL(PetscIsCloseAtTol(pos_one,neg_one,2,0));

  CALL(PetscIsCloseAtTol(pos_one+eps,pos_one,0,0));
  CALL(PetscIsCloseAtTol(pos_one-eps,pos_one,0,0));
  CALL(PetscIsCloseAtTol(pos_one+eps,pos_one,0,0));
  CALL(PetscIsCloseAtTol(pos_one-eps,pos_one,0,0));

  CALL(PetscIsCloseAtTol(pos_one+eps,pos_one,0,eps));
  CALL(PetscIsCloseAtTol(pos_one-eps,pos_one,0,eps));
  CALL(PetscIsCloseAtTol(pos_one+eps,pos_one,eps,0));
  CALL(PetscIsCloseAtTol(pos_one-eps,pos_one,eps,0));

  CALL(PetscIsCloseAtTol(pos_one+2*eps,pos_one,eps,0));
  CALL(PetscIsCloseAtTol(pos_one-2*eps,pos_one,eps,0));
  CALL(PetscIsCloseAtTol(pos_one+2*eps,pos_one,0,eps));
  CALL(PetscIsCloseAtTol(pos_one-2*eps,pos_one,0,eps));

  CALL(PetscIsCloseAtTol(neg_inf,neg_zero,2,2));
  CALL(PetscIsCloseAtTol(neg_inf,pos_zero,2,2));
  CALL(PetscIsCloseAtTol(neg_inf,neg_one,2,2));
  CALL(PetscIsCloseAtTol(neg_inf,pos_one,2,2));
  CALL(PetscIsCloseAtTol(neg_inf,neg_inf,2,2));
  CALL(PetscIsCloseAtTol(neg_inf,pos_inf,2,2));
  CALL(PetscIsCloseAtTol(neg_inf,x_nan,2,2));

  CALL(PetscIsCloseAtTol(pos_inf,neg_zero,2,2));
  CALL(PetscIsCloseAtTol(pos_inf,pos_zero,2,2));
  CALL(PetscIsCloseAtTol(pos_inf,neg_one,2,2));
  CALL(PetscIsCloseAtTol(pos_inf,pos_one,2,2));
  CALL(PetscIsCloseAtTol(pos_inf,neg_inf,2,2));
  CALL(PetscIsCloseAtTol(pos_inf,pos_inf,2,2));
  CALL(PetscIsCloseAtTol(pos_inf,x_nan,2,2));

  CALL(PetscIsCloseAtTol(x_nan,neg_zero,2,2));
  CALL(PetscIsCloseAtTol(x_nan,pos_zero,2,2));
  CALL(PetscIsCloseAtTol(x_nan,neg_one,2,2));
  CALL(PetscIsCloseAtTol(x_nan,pos_one,2,2));
  CALL(PetscIsCloseAtTol(x_nan,neg_inf,2,2));
  CALL(PetscIsCloseAtTol(x_nan,pos_inf,2,2));
  CALL(PetscIsCloseAtTol(x_nan,x_nan,2,2));

  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      output_file: output/ex39.out

TEST*/
