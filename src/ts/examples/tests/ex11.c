static char help[] = "Simple wrapper object to solve DAE of the form:\n\
                             \\dot{U} = f(U,V)\n\
                             F(U,V) = 0\n\n";

#include <petscts.h>

#undef __FUNCT__
#define __FUNCT__ "f"
/*
   Simple example:   f(U,V) = U + V

*/
PetscErrorCode f(PetscReal t,Vec U,Vec V,Vec F,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecWAXPY(F,1.0,U,V);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "F"
/*
   Simple example: F(U,V) = U - V

*/
PetscErrorCode F(PetscReal t,Vec U,Vec V,Vec F,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecWAXPY(F,-1.0,V,U);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  TS             ts;
  Vec            U,V,Usolution;

  PetscInitialize(&argc,&argv,(char*)0,help);

  ierr = VecCreateMPI(PETSC_COMM_WORLD,1,PETSC_DETERMINE,&U);CHKERRQ(ierr);
  ierr = VecCreateMPI(PETSC_COMM_WORLD,1,PETSC_DETERMINE,&V);CHKERRQ(ierr);

  ierr = VecDuplicate(U,&Usolution);CHKERRQ(ierr);
  ierr = VecSet(Usolution,1.0);CHKERRQ(ierr);
  ierr = VecSet(V,1.0);CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSDAESIMPLERED);CHKERRQ(ierr);
  ierr = TSDAESimpleSetRHSFunction(ts,Usolution,f,NULL);CHKERRQ(ierr);
  ierr = TSDAESimpleSetIFunction(ts,V,F,NULL);CHKERRQ(ierr);

  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSolve(ts,Usolution);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = VecDestroy(&Usolution);CHKERRQ(ierr);
  ierr = VecDestroy(&V);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}



