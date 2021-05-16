static char help[] = "Tests wrapping of math.h functions for real, complex, and scalar types \n";
#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Real tests:\n");CHKERRQ(ierr);
  {
    PetscReal a,b,c;
    a = PetscRealConstant(0.5);
    c = PetscRealConstant(2.0);

    b = PetscSqrtReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"sqrt(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscCbrtReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"cbrt(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);

    b = PetscHypotReal(a,c);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"hypot(%f,%f) = %f\n",(double)a,(double)c,(double)b);CHKERRQ(ierr);
    b = PetscAtan2Real(a,c);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"atan2(%f,%f) = %f\n",(double)a,(double)c,(double)b);CHKERRQ(ierr);

    b = PetscPowReal(a,c);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"pow(%f,%f) = %f\n",(double)a,(double)c,(double)b);CHKERRQ(ierr);
    b = PetscExpReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"exp(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscLogReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"log(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscLog10Real(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"log10(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscLog2Real(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"log2(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);

    b = PetscSinReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"sin(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscCosReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"cos(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscTanReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"tan(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);

    b = PetscAsinReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"asin(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscAcosReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"acos(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscAtanReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"atan(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);

    b = PetscSinhReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"sinh(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscCoshReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"cosh(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscTanhReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"tanh(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);

    b = PetscAsinhReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"asinh(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscAcoshReal(c);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"acosh(%f) = %f\n",(double)c,(double)b);CHKERRQ(ierr);
    b = PetscAtanhReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"atanh(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);

    b = PetscCeilReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"ceil(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscFloorReal(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"floor(%f) = %f\n",(double)a,(double)b);CHKERRQ(ierr);
    b = PetscFmodReal(a,c);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"fmod(%f,%f) = %f\n",(double)a,(double)c,(double)b);CHKERRQ(ierr);
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Scalar tests:\n");CHKERRQ(ierr);
  {
    PetscScalar a,b,c;
    a = PetscRealConstant(0.5);
    c = PetscRealConstant(2.0);

    b = PetscAbsScalar(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"abs(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b));CHKERRQ(ierr);
    b = PetscArgScalar(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"arg(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b));CHKERRQ(ierr);
    b = PetscConj(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"conj(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b));CHKERRQ(ierr);

    b = PetscSqrtScalar(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"sqrt(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b));CHKERRQ(ierr);

    b = PetscPowScalar(a,c);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"pow(%f,%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(c),(double)PetscRealPart(b));CHKERRQ(ierr);
    b = PetscExpScalar(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"exp(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b));CHKERRQ(ierr);
    b = PetscLogScalar(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"log(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b));CHKERRQ(ierr);

    b = PetscSinScalar(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"sin(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b));CHKERRQ(ierr);
    b = PetscCosScalar(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"cos(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b));CHKERRQ(ierr);
    b = PetscTanScalar(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"tan(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b));CHKERRQ(ierr);

    b = PetscAsinScalar(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"asin(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b));CHKERRQ(ierr);
    b = PetscAcosScalar(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"acos(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b));CHKERRQ(ierr);
    b = PetscAtanScalar(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"atan(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b));CHKERRQ(ierr);

    b = PetscSinhScalar(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"sinh(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b));CHKERRQ(ierr);
    b = PetscCoshScalar(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"cosh(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b));CHKERRQ(ierr);
    b = PetscTanhScalar(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"tanh(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b));CHKERRQ(ierr);

    b = PetscAsinhScalar(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"asinh(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b));CHKERRQ(ierr);
    b = PetscAcoshScalar(c);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"acosh(%f) = %f\n",(double)PetscRealPart(c),(double)PetscRealPart(b));CHKERRQ(ierr);
    b = PetscAtanhScalar(a);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"atanh(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b));CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
