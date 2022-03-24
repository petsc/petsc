static char help[] = "Tests wrapping of math.h functions for real, complex, and scalar types \n";
#include <petscsys.h>

int main(int argc,char **argv)
{

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Real tests:\n"));
  {
    PetscReal a,b,c;
    a = PetscRealConstant(0.5);
    c = PetscRealConstant(2.0);

    b = PetscSqrtReal(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"sqrt(%f) = %f\n",(double)a,(double)b));
    b = PetscCbrtReal(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"cbrt(%f) = %f\n",(double)a,(double)b));

    b = PetscHypotReal(a,c);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"hypot(%f,%f) = %f\n",(double)a,(double)c,(double)b));
    b = PetscAtan2Real(a,c);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"atan2(%f,%f) = %f\n",(double)a,(double)c,(double)b));

    b = PetscPowReal(a,c);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"pow(%f,%f) = %f\n",(double)a,(double)c,(double)b));
    b = PetscExpReal(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"exp(%f) = %f\n",(double)a,(double)b));
    b = PetscLogReal(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"log(%f) = %f\n",(double)a,(double)b));
    b = PetscLog10Real(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"log10(%f) = %f\n",(double)a,(double)b));
    b = PetscLog2Real(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"log2(%f) = %f\n",(double)a,(double)b));

    b = PetscSinReal(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"sin(%f) = %f\n",(double)a,(double)b));
    b = PetscCosReal(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"cos(%f) = %f\n",(double)a,(double)b));
    b = PetscTanReal(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"tan(%f) = %f\n",(double)a,(double)b));

    b = PetscAsinReal(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"asin(%f) = %f\n",(double)a,(double)b));
    b = PetscAcosReal(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"acos(%f) = %f\n",(double)a,(double)b));
    b = PetscAtanReal(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"atan(%f) = %f\n",(double)a,(double)b));

    b = PetscSinhReal(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"sinh(%f) = %f\n",(double)a,(double)b));
    b = PetscCoshReal(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"cosh(%f) = %f\n",(double)a,(double)b));
    b = PetscTanhReal(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"tanh(%f) = %f\n",(double)a,(double)b));

    b = PetscAsinhReal(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"asinh(%f) = %f\n",(double)a,(double)b));
    b = PetscAcoshReal(c);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"acosh(%f) = %f\n",(double)c,(double)b));
    b = PetscAtanhReal(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"atanh(%f) = %f\n",(double)a,(double)b));

    b = PetscCeilReal(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"ceil(%f) = %f\n",(double)a,(double)b));
    b = PetscFloorReal(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"floor(%f) = %f\n",(double)a,(double)b));
    b = PetscFmodReal(a,c);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"fmod(%f,%f) = %f\n",(double)a,(double)c,(double)b));
  }
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Scalar tests:\n"));
  {
    PetscScalar a,b,c;
    a = PetscRealConstant(0.5);
    c = PetscRealConstant(2.0);

    b = PetscAbsScalar(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"abs(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b)));
    b = PetscArgScalar(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"arg(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b)));
    b = PetscConj(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"conj(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b)));

    b = PetscSqrtScalar(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"sqrt(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b)));

    b = PetscPowScalar(a,c);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"pow(%f,%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(c),(double)PetscRealPart(b)));
    b = PetscExpScalar(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"exp(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b)));
    b = PetscLogScalar(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"log(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b)));

    b = PetscSinScalar(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"sin(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b)));
    b = PetscCosScalar(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"cos(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b)));
    b = PetscTanScalar(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"tan(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b)));

    b = PetscAsinScalar(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"asin(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b)));
    b = PetscAcosScalar(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"acos(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b)));
    b = PetscAtanScalar(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"atan(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b)));

    b = PetscSinhScalar(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"sinh(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b)));
    b = PetscCoshScalar(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"cosh(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b)));
    b = PetscTanhScalar(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"tanh(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b)));

    b = PetscAsinhScalar(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"asinh(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b)));
    b = PetscAcoshScalar(c);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"acosh(%f) = %f\n",(double)PetscRealPart(c),(double)PetscRealPart(b)));
    b = PetscAtanhScalar(a);
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"atanh(%f) = %f\n",(double)PetscRealPart(a),(double)PetscRealPart(b)));
  }
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
