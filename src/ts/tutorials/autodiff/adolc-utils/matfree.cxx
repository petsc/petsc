#include <petscdmda.h>
#include <petscts.h>
#include <adolc/interfaces.h>
#include "contexts.cxx"

/*
   REQUIRES configuration of PETSc with option --download-adolc.

   For documentation on ADOL-C, see
     $PETSC_ARCH/externalpackages/ADOL-C-2.6.0/ADOL-C/doc/adolc-manual.pdf
*/

/*
  ADOL-C implementation for Jacobian vector product, using the forward mode of AD.
  Intended to overload MatMult in matrix-free methods where implicit timestepping
  has been used.

  For an implicit Jacobian we may use the rule that
     G = M*xdot + f(x)    ==>     dG/dx = a*M + df/dx,
  where a = d(xdot)/dx is a constant. Evaluated at x0 and acting upon a vector x1:
     (dG/dx)(x0) * x1 = (a*df/d(xdot)(x0) + df/dx)(x0) * x1.

  Input parameters:
  A_shell - Jacobian matrix of MatShell type
  X       - vector to be multiplied by A_shell

  Output parameters:
  Y       - product of A_shell and X
*/
PetscErrorCode PetscAdolcIJacobianVectorProduct(Mat A_shell,Vec X,Vec Y)
{
  AdolcMatCtx        *mctx;
  PetscInt          m,n,i,j,k = 0,d;
  const PetscScalar *x0;
  PetscScalar       *action,*x1;
  Vec               localX1;
  DM                da;
  DMDALocalInfo     info;

  PetscFunctionBegin;

  /* Get matrix-free context info */
  CHKERRQ(MatShellGetContext(A_shell,&mctx));
  m = mctx->m;
  n = mctx->n;

  /* Get local input vectors and extract data, x0 and x1*/
  CHKERRQ(TSGetDM(mctx->ts,&da));
  CHKERRQ(DMDAGetLocalInfo(da,&info));
  CHKERRQ(DMGetLocalVector(da,&localX1));
  CHKERRQ(DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX1));
  CHKERRQ(DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX1));

  CHKERRQ(VecGetArrayRead(mctx->localX0,&x0));
  CHKERRQ(VecGetArray(localX1,&x1));

  /* dF/dx part */
  CHKERRQ(PetscMalloc1(m,&action));
  CHKERRQ(PetscLogEventBegin(mctx->event1,0,0,0,0));
  fos_forward(mctx->tag1,m,n,0,x0,x1,NULL,action);
  for (j=info.gys; j<info.gys+info.gym; j++) {
    for (i=info.gxs; i<info.gxs+info.gxm; i++) {
      for (d=0; d<2; d++) {
        if ((i >= info.xs) && (i < info.xs+info.xm) && (j >= info.ys) && (j < info.ys+info.ym)) {
          CHKERRQ(VecSetValuesLocal(Y,1,&k,&action[k],INSERT_VALUES));
        }
        k++;
      }
    }
  }
  CHKERRQ(PetscLogEventEnd(mctx->event1,0,0,0,0));
  k = 0;
  CHKERRQ(VecAssemblyBegin(Y)); /* Note: Need to assemble between separate calls */
  CHKERRQ(VecAssemblyEnd(Y));   /*       to INSERT_VALUES and ADD_VALUES         */

  /* a * dF/d(xdot) part */
  CHKERRQ(PetscLogEventBegin(mctx->event2,0,0,0,0));
  fos_forward(mctx->tag2,m,n,0,x0,x1,NULL,action);
  for (j=info.gys; j<info.gys+info.gym; j++) {
    for (i=info.gxs; i<info.gxs+info.gxm; i++) {
      for (d=0; d<2; d++) {
        if ((i >= info.xs) && (i < info.xs+info.xm) && (j >= info.ys) && (j < info.ys+info.ym)) {
          action[k] *= mctx->shift;
          CHKERRQ(VecSetValuesLocal(Y,1,&k,&action[k],ADD_VALUES));
        }
        k++;
      }
    }
  }
  CHKERRQ(PetscLogEventEnd(mctx->event2,0,0,0,0));
  CHKERRQ(VecAssemblyBegin(Y));
  CHKERRQ(VecAssemblyEnd(Y));
  CHKERRQ(PetscFree(action));

  /* Restore local vector */
  CHKERRQ(VecRestoreArray(localX1,&x1));
  CHKERRQ(VecRestoreArrayRead(mctx->localX0,&x0));
  CHKERRQ(DMRestoreLocalVector(da,&localX1));
  PetscFunctionReturn(0);
}

/*
  ADOL-C implementation for Jacobian vector product, using the forward mode of AD.
  Intended to overload MatMult in matrix-free methods where implicit timestepping
  has been applied to a problem of the form
                             du/dt = F(u).

  Input parameters:
  A_shell - Jacobian matrix of MatShell type
  X       - vector to be multiplied by A_shell

  Output parameters:
  Y       - product of A_shell and X
*/
PetscErrorCode PetscAdolcIJacobianVectorProductIDMass(Mat A_shell,Vec X,Vec Y)
{
  AdolcMatCtx       *mctx;
  PetscInt          m,n,i,j,k = 0,d;
  const PetscScalar *x0;
  PetscScalar       *action,*x1;
  Vec               localX1;
  DM                da;
  DMDALocalInfo     info;

  PetscFunctionBegin;

  /* Get matrix-free context info */
  CHKERRQ(MatShellGetContext(A_shell,&mctx));
  m = mctx->m;
  n = mctx->n;

  /* Get local input vectors and extract data, x0 and x1*/
  CHKERRQ(TSGetDM(mctx->ts,&da));
  CHKERRQ(DMDAGetLocalInfo(da,&info));
  CHKERRQ(DMGetLocalVector(da,&localX1));
  CHKERRQ(DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX1));
  CHKERRQ(DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX1));

  CHKERRQ(VecGetArrayRead(mctx->localX0,&x0));
  CHKERRQ(VecGetArray(localX1,&x1));

  /* dF/dx part */
  CHKERRQ(PetscMalloc1(m,&action));
  CHKERRQ(PetscLogEventBegin(mctx->event1,0,0,0,0));
  fos_forward(mctx->tag1,m,n,0,x0,x1,NULL,action);
  for (j=info.gys; j<info.gys+info.gym; j++) {
    for (i=info.gxs; i<info.gxs+info.gxm; i++) {
      for (d=0; d<2; d++) {
        if ((i >= info.xs) && (i < info.xs+info.xm) && (j >= info.ys) && (j < info.ys+info.ym)) {
          CHKERRQ(VecSetValuesLocal(Y,1,&k,&action[k],INSERT_VALUES));
        }
        k++;
      }
    }
  }
  CHKERRQ(PetscLogEventEnd(mctx->event1,0,0,0,0));
  k = 0;
  CHKERRQ(VecAssemblyBegin(Y)); /* Note: Need to assemble between separate calls */
  CHKERRQ(VecAssemblyEnd(Y));   /*       to INSERT_VALUES and ADD_VALUES         */
  CHKERRQ(PetscFree(action));

  /* Restore local vector */
  CHKERRQ(VecRestoreArray(localX1,&x1));
  CHKERRQ(VecRestoreArrayRead(mctx->localX0,&x0));
  CHKERRQ(DMRestoreLocalVector(da,&localX1));

  /* a * dF/d(xdot) part */
  CHKERRQ(PetscLogEventBegin(mctx->event2,0,0,0,0));
  CHKERRQ(VecAXPY(Y,mctx->shift,X));
  CHKERRQ(PetscLogEventEnd(mctx->event2,0,0,0,0));
  PetscFunctionReturn(0);
}

/*
  ADOL-C implementation for Jacobian transpose vector product, using the reverse mode of AD.
  Intended to overload MatMultTranspose in matrix-free methods where implicit timestepping
  has been used.

  Input parameters:
  A_shell - Jacobian matrix of MatShell type
  Y       - vector to be multiplied by A_shell transpose

  Output parameters:
  X       - product of A_shell transpose and X
*/
PetscErrorCode PetscAdolcIJacobianTransposeVectorProduct(Mat A_shell,Vec Y,Vec X)
{
  AdolcMatCtx       *mctx;
  PetscInt          m,n,i,j,k = 0,d;
  const PetscScalar *x;
  PetscScalar       *action,*y;
  Vec               localY;
  DM                da;
  DMDALocalInfo     info;

  PetscFunctionBegin;

  /* Get matrix-free context info */
  CHKERRQ(MatShellGetContext(A_shell,&mctx));
  m = mctx->m;
  n = mctx->n;

  /* Get local input vectors and extract data, x0 and x1*/
  CHKERRQ(TSGetDM(mctx->ts,&da));
  CHKERRQ(DMDAGetLocalInfo(da,&info));
  CHKERRQ(DMGetLocalVector(da,&localY));
  CHKERRQ(DMGlobalToLocalBegin(da,Y,INSERT_VALUES,localY));
  CHKERRQ(DMGlobalToLocalEnd(da,Y,INSERT_VALUES,localY));
  CHKERRQ(VecGetArrayRead(mctx->localX0,&x));
  CHKERRQ(VecGetArray(localY,&y));

  /* dF/dx part */
  CHKERRQ(PetscMalloc1(n,&action));
  CHKERRQ(PetscLogEventBegin(mctx->event3,0,0,0,0));
  if (!mctx->flg)
    zos_forward(mctx->tag1,m,n,1,x,NULL);
  fos_reverse(mctx->tag1,m,n,y,action);
  for (j=info.gys; j<info.gys+info.gym; j++) {
    for (i=info.gxs; i<info.gxs+info.gxm; i++) {
      for (d=0; d<2; d++) {
        if ((i >= info.xs) && (i < info.xs+info.xm) && (j >= info.ys) && (j < info.ys+info.ym)) {
          CHKERRQ(VecSetValuesLocal(X,1,&k,&action[k],INSERT_VALUES));
        }
        k++;
      }
    }
  }
  CHKERRQ(PetscLogEventEnd(mctx->event3,0,0,0,0));
  k = 0;
  CHKERRQ(VecAssemblyBegin(X)); /* Note: Need to assemble between separate calls */
  CHKERRQ(VecAssemblyEnd(X));   /*       to INSERT_VALUES and ADD_VALUES         */

  /* a * dF/d(xdot) part */
  CHKERRQ(PetscLogEventBegin(mctx->event4,0,0,0,0));
  if (!mctx->flg) {
    zos_forward(mctx->tag2,m,n,1,x,NULL);
    mctx->flg = PETSC_TRUE;
  }
  fos_reverse(mctx->tag2,m,n,y,action);
  for (j=info.gys; j<info.gys+info.gym; j++) {
    for (i=info.gxs; i<info.gxs+info.gxm; i++) {
      for (d=0; d<2; d++) {
        if ((i >= info.xs) && (i < info.xs+info.xm) && (j >= info.ys) && (j < info.ys+info.ym)) {
          action[k] *= mctx->shift;
          CHKERRQ(VecSetValuesLocal(X,1,&k,&action[k],ADD_VALUES));
        }
        k++;
      }
    }
  }
  CHKERRQ(PetscLogEventEnd(mctx->event4,0,0,0,0));
  CHKERRQ(VecAssemblyBegin(X));
  CHKERRQ(VecAssemblyEnd(X));
  CHKERRQ(PetscFree(action));

  /* Restore local vector */
  CHKERRQ(VecRestoreArray(localY,&y));
  CHKERRQ(VecRestoreArrayRead(mctx->localX0,&x));
  CHKERRQ(DMRestoreLocalVector(da,&localY));
  PetscFunctionReturn(0);
}

/*
  ADOL-C implementation for Jacobian transpose vector product, using the reverse mode of AD.
  Intended to overload MatMultTranspose in matrix-free methods where implicit timestepping
  has been applied to a problem of the form
                          du/dt = F(u).

  Input parameters:
  A_shell - Jacobian matrix of MatShell type
  Y       - vector to be multiplied by A_shell transpose

  Output parameters:
  X       - product of A_shell transpose and X
*/
PetscErrorCode PetscAdolcIJacobianTransposeVectorProductIDMass(Mat A_shell,Vec Y,Vec X)
{
  AdolcMatCtx       *mctx;
  PetscInt          m,n,i,j,k = 0,d;
  const PetscScalar *x;
  PetscScalar       *action,*y;
  Vec               localY;
  DM                da;
  DMDALocalInfo     info;

  PetscFunctionBegin;

  /* Get matrix-free context info */
  CHKERRQ(MatShellGetContext(A_shell,&mctx));
  m = mctx->m;
  n = mctx->n;

  /* Get local input vectors and extract data, x0 and x1*/
  CHKERRQ(TSGetDM(mctx->ts,&da));
  CHKERRQ(DMDAGetLocalInfo(da,&info));
  CHKERRQ(DMGetLocalVector(da,&localY));
  CHKERRQ(DMGlobalToLocalBegin(da,Y,INSERT_VALUES,localY));
  CHKERRQ(DMGlobalToLocalEnd(da,Y,INSERT_VALUES,localY));
  CHKERRQ(VecGetArrayRead(mctx->localX0,&x));
  CHKERRQ(VecGetArray(localY,&y));

  /* dF/dx part */
  CHKERRQ(PetscMalloc1(n,&action));
  CHKERRQ(PetscLogEventBegin(mctx->event3,0,0,0,0));
  if (!mctx->flg) zos_forward(mctx->tag1,m,n,1,x,NULL);
  fos_reverse(mctx->tag1,m,n,y,action);
  for (j=info.gys; j<info.gys+info.gym; j++) {
    for (i=info.gxs; i<info.gxs+info.gxm; i++) {
      for (d=0; d<2; d++) {
        if ((i >= info.xs) && (i < info.xs+info.xm) && (j >= info.ys) && (j < info.ys+info.ym)) {
          CHKERRQ(VecSetValuesLocal(X,1,&k,&action[k],INSERT_VALUES));
        }
        k++;
      }
    }
  }
  CHKERRQ(PetscLogEventEnd(mctx->event3,0,0,0,0));
  k = 0;
  CHKERRQ(VecAssemblyBegin(X)); /* Note: Need to assemble between separate calls */
  CHKERRQ(VecAssemblyEnd(X));   /*       to INSERT_VALUES and ADD_VALUES         */
  CHKERRQ(PetscFree(action));

  /* Restore local vector */
  CHKERRQ(VecRestoreArray(localY,&y));
  CHKERRQ(VecRestoreArrayRead(mctx->localX0,&x));
  CHKERRQ(DMRestoreLocalVector(da,&localY));

  /* a * dF/d(xdot) part */
  CHKERRQ(PetscLogEventBegin(mctx->event4,0,0,0,0));
  CHKERRQ(VecAXPY(X,mctx->shift,Y));
  CHKERRQ(PetscLogEventEnd(mctx->event4,0,0,0,0));
  PetscFunctionReturn(0);
}
