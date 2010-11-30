
#include "petscsnes.h"

typedef struct {
  Vec         q,zz,lb,ub;
  Mat         M;
} AppCtx;

/* 
   User-defined routines
*/
extern PetscErrorCode FormJacobian1(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern PetscErrorCode FormFunction1(SNES,Vec,Vec,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  SNES           snes;         /* nonlinear solver context */
  Vec            r;          /* solution, residual vectors */
  PetscErrorCode ierr;
  AppCtx         user;         /* user-defined work context */
  PetscViewer    viewer;

  PetscInitialize(&argc,&argv,(char *)0,0);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,"videfinition",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&user.M);CHKERRQ(ierr); ierr = MatLoad(user.M,viewer);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF,&user.q);CHKERRQ(ierr); ierr = VecLoad(user.q,viewer);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF,&user.lb);CHKERRQ(ierr); ierr = VecLoad(user.lb,viewer);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF,&user.ub);CHKERRQ(ierr);ierr = VecLoad(user.ub,viewer);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_SELF,&user.zz);CHKERRQ(ierr);ierr = VecLoad(user.zz,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  ierr = VecDuplicate(user.q,&r);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  ierr = SNESSetFunction(snes,r,FormFunction1,&user);CHKERRQ(ierr);

  ierr = SNESSetJacobian(snes,user.M,user.M,FormJacobian1,PETSC_NULL);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = SNESVISetVariableBounds(snes,user.lb,user.ub);CHKERRQ(ierr);
  ierr = SNESSolve(snes,PETSC_NULL,user.zz);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormFunction1"
/* 
   FormFunction1 - Evaluates nonlinear function, F(x).

   Input Parameters:
.  snes - the SNES context
.  x    - input vector
.  ctx  - optional user-defined context

   Output Parameter:
.  f - function vector
 */
PetscErrorCode FormFunction1(SNES snes,Vec x,Vec f,void *ctx)
{
  PetscErrorCode ierr;
  AppCtx         *user = (AppCtx*)ctx;

  ierr = MatMult(user->M,x,f);CHKERRQ(ierr);
  ierr = VecAXPY(f,1.0,user->q);CHKERRQ(ierr);
 
  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormJacobian1"
/*
   FormJacobian1 - Evaluates Jacobian matrix.

   Input Parameters:
.  snes - the SNES context
.  x - input vector
.  dummy - optional user-defined context (not used here)

   Output Parameters:
.  jac - Jacobian matrix
.  B - optionally different preconditioning matrix
.  flag - flag indicating matrix structure
*/
PetscErrorCode FormJacobian1(SNES snes,Vec x,Mat *jac,Mat *B,MatStructure *flag,void *dummy)
{
  PetscErrorCode ierr;

  return 0;
}


