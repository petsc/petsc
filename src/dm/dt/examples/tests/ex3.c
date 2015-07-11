static char help[] = "Tests quadrature.\n\n";

#include <petscdt.h>

#undef __FUNCT__
#define __FUNCT__ "func1"
static void func1(PetscReal x, PetscReal *val)
{
  *val = x*log(1+x);
}

#undef __FUNCT__
#define __FUNCT__ "func2"
static void func2(PetscReal x, PetscReal *val)
{
  *val = x*x*PetscAtanReal(x);
}

#undef __FUNCT__
#define __FUNCT__ "func3"
static void func3(PetscReal x, PetscReal *val)
{
  *val = PetscExpReal(x)*PetscCosReal(x);
}

#undef __FUNCT__
#define __FUNCT__ "func4"
static void func4(PetscReal x, PetscReal *val)
{
  *val = PetscAtanReal(PetscSqrtReal(2+x*x))/((1+x*x)*PetscSqrtReal(2+x*x));
}

#undef __FUNCT__
#define __FUNCT__ "func5"
static void func5(PetscReal x, PetscReal *val)
{
  *val = PetscSqrtReal(x)*PetscLogReal(x);
}

#undef __FUNCT__
#define __FUNCT__ "func6"
static void func6(PetscReal x, PetscReal *val)
{
  *val = PetscSqrtReal(1-x*x);
}

#undef __FUNCT__
#define __FUNCT__ "func7"
static void func7(PetscReal x, PetscReal *val)
{
  *val = PetscSqrtReal(x)/PetscSqrtReal(1-x*x);
}

#undef __FUNCT__
#define __FUNCT__ "func8"
static void func8(PetscReal x, PetscReal *val)
{
  *val = PetscLogReal(x)*PetscLogReal(x);
}

#undef __FUNCT__
#define __FUNCT__ "func9"
static void func9(PetscReal x, PetscReal *val)
{
  *val = PetscLogReal(PetscCosReal(x));
}

#undef __FUNCT__
#define __FUNCT__ "func10"
static void func10(PetscReal x, PetscReal *val)
{
  *val = PetscSqrtReal(PetscTanReal(x));
}
#undef __FUNCT__
#define __FUNCT__ "func11"
static void func11(PetscReal x, PetscReal *val)
{
  *val = 1/(1-2*x+2*x*x);
}
#undef __FUNCT__
#define __FUNCT__ "func12"
static void func12(PetscReal x, PetscReal *val)
{
  *val = PetscExpReal(1-1/x)/PetscSqrtReal(x*x*x-x*x*x*x);
}
#undef __FUNCT__


#define __FUNCT__ "main"
int main(int argc, char **argv)
{

  
  const PetscInt  digits      = 12;
  const PetscReal analytic[12] = 
    {
      0.250000000000000,
      0.210657251225806988108092302182988001695680805674,
      1.905238690482675827736517833351916563195085437332,
      0.514041895890070761397629739576882871630921844127,
      -.444444444444444444444444444444444444444444444444,
      0.785398163397448309615660845819875721049292349843,
      0,
      2.000000000000000000000000000000000000000000000000,
      -1.08879304515180106525034444911880697366929185018,
      2.221441469079183123507940495030346849307310844687,
      1.570796326794896619231321691639751442098584699687,
      1.772453850905516027298167483341145182797549456122,
    };
  const PetscReal epsilon     = 1.0e-14;
  PetscReal       integral[12];
  PetscErrorCode  ierr;


  ierr = PetscInitialize(&argc, &argv, PETSC_NULL, help);CHKERRQ(ierr);

  //integrate each function
  ierr = PetscDTTanhSinhIntegrate(func1, 0.0, 1, digits, &integral[0]);CHKERRQ(ierr);
  ierr = PetscDTTanhSinhIntegrate(func2, 0.0, 1, digits, &integral[1]);CHKERRQ(ierr);
  ierr = PetscDTTanhSinhIntegrate(func3, 0.0, PETSC_PI/2., digits, &integral[2]);CHKERRQ(ierr);
  ierr = PetscDTTanhSinhIntegrate(func4, 0.0, 1, digits, &integral[3]);CHKERRQ(ierr);
  ierr = PetscDTTanhSinhIntegrate(func5, 0.0, 1, digits, &integral[4]);CHKERRQ(ierr);
  ierr = PetscDTTanhSinhIntegrate(func6, 0.0, 1, digits, &integral[5]);CHKERRQ(ierr);
  ierr = PetscDTTanhSinhIntegrate(func7, 0.0, 1, digits, &integral[6]);CHKERRQ(ierr);
  ierr = PetscDTTanhSinhIntegrate(func8, 0.0, 1, digits, &integral[7]);CHKERRQ(ierr);
  ierr = PetscDTTanhSinhIntegrate(func9, 0.0, PETSC_PI/2., digits, &integral[8]);CHKERRQ(ierr);
  ierr = PetscDTTanhSinhIntegrate(func10, 0.0, PETSC_PI/2., digits, &integral[9]);CHKERRQ(ierr);
  ierr = PetscDTTanhSinhIntegrate(func11, 0.0, 1, digits, &integral[10]);CHKERRQ(ierr);
  ierr = PetscDTTanhSinhIntegrate(func12, 0.0, 1, digits, &integral[11]);CHKERRQ(ierr);
  

  for(int i=0; i<12; ++i) {
    
    if (PetscAbsReal(integral[i] - analytic[i]) < epsilon) {ierr = PetscPrintf(PETSC_COMM_SELF, "The integral of func%d is correct\n", i);CHKERRQ(ierr);}
    else                                                {ierr = PetscPrintf(PETSC_COMM_SELF, "The integral of func%d is wrong: %15.15f (%15.15f)\n", i, integral[i], PetscAbsReal(integral[i] - analytic[i]));CHKERRQ(ierr);}
    
  }
  
  PetscFinalize();
  
  return 0;
}
