/*T
   Concepts: KSP^solving a system of linear equations
   Concepts: KSP^Laplacian, 2d
   Concepts: KSP^semi-implicit
   Processors: n
T*/

/*
This is intended to be a prototypical example of the semi-implicit algorithm for
a PDE. We have three phases:

  1) An explicit predictor step

     u^{k+1/3} = P(u^k)

  2) An implicit corrector step

     \Delta u^{k+2/3} = F(u^{k+1/3})

  3) An explicit update step

     u^{k+1} = C(u^{k+2/3})

We will solve on the unit square with Dirichlet boundary conditions

   u = f(x,y) for x = 0, x = 1, y = 0, y = 1

Although we are using a DA, and thus have a structured mesh, we will discretize
the problem with finite elements, splitting each cell of the DA into two
triangles.

This uses multigrid to solve the linear system
*/

static char help[] = "Solves 2D inhomogeneous Laplacian using multigrid.\n\n";

#include "petscda.h"
#include "petscksp.h"
#include "petscmg.h"

extern PetscErrorCode ComputeInitialGuess(DMMG,Vec);
extern PetscErrorCode ComputePredictor(DMMG,Vec,Vec);
extern PetscErrorCode ComputeJacobian(DMMG,Mat);
extern PetscErrorCode ComputeRHS(DMMG,Vec);
extern PetscErrorCode ComputeCorrector(DMMG,Vec,Vec);
extern PetscErrorCode dummy(SNES,Vec,void *);

typedef struct {
  PetscScalar phi;
} UserContext;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DMMG           *dmmg;
  DA             da;
  UserContext    user;
  PetscErrorCode ierr;
  PetscInt       l;

  PetscInitialize(&argc,&argv,(char *)0,help);

  ierr = DMMGCreate(PETSC_COMM_WORLD,3,PETSC_NULL,&dmmg);CHKERRQ(ierr);
  ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,3,3,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,&da);CHKERRQ(ierr);  
  ierr = DMMGSetDM(dmmg,(DM)da);
  ierr = DADestroy(da);CHKERRQ(ierr);
  for (l = 0; l < DMMGGetLevels(dmmg); l++) {
    ierr = DMMGSetUser(dmmg,l,&user);CHKERRQ(ierr);
  }

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for PCICE", "DMMG");
    user.phi = 0.5;
    ierr = PetscOptionsScalar("-phi", "The time weighting parameter", "ex31.c", user.phi, &user.phi, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = ComputeInitialGuess(dmmg[0], DMMGGetr(dmmg));
  ierr = ComputePredictor(dmmg[0], DMMGGetr(dmmg), DMMGGetx(dmmg));

  ierr = DMMGSetKSP(dmmg,ComputeRHS,ComputeJacobian);CHKERRQ(ierr);
  ierr = DMMGSetInitialGuess(dmmg, dummy);CHKERRQ(ierr);
  ierr = DMMGSolve(dmmg);CHKERRQ(ierr);

  ierr = ComputeCorrector(dmmg[0], DMMGGetx(dmmg), DMMGGetr(dmmg));

  ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);

  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "ComputeInitialGuess"
PetscErrorCode ComputeInitialGuess(DMMG dmmg, Vec u)
{
  PetscScalar    one = 1.0;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecSet(&one, u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputePredictor"
PetscErrorCode ComputePredictor(DMMG dmmg, Vec uOld, Vec u)
{
  DA             da   = (DA)dmmg->dm;
  PetscScalar    zero = 0.0;
  Vec            uOldLocal, uLocal;
  PetscScalar  **pOld;
  PetscScalar  **p;
  PetscInt       i,j,mx,my,xm,ym,xs,ys;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = VecSet(&zero, u);CHKERRQ(ierr);
  ierr = DAGetLocalVector(da, &uOldLocal);CHKERRQ(ierr);
  ierr = DAGetLocalVector(da, &uLocal);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da, uOld, INSERT_VALUES, uOldLocal);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da, uOld, INSERT_VALUES, uOldLocal);CHKERRQ(ierr);
  ierr = DAVecGetArray(da, uOldLocal, (void *) &pOld);CHKERRQ(ierr);
  ierr = DAVecGetArray(da, uLocal,    (void *) &p);CHKERRQ(ierr);
  ierr = DAGetInfo(da, 0, &mx, &my, 0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  for(j = ys; j < ys+ym; j++) {
    for(i = xs; i < xs+xm; i++) {
      p[j][i] = pOld[j][i];
    }
  }
  ierr = DAVecRestoreArray(da, uOldLocal, (void *) &pOld);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da, uLocal,    (void *) &p);CHKERRQ(ierr);
  ierr = DALocalToGlobal(da, uLocal, ADD_VALUES, u);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(da, &uOldLocal);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(da, &uLocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeRHS"
/*
  We integrate over each cell

  (i, j+1)----(i+1, j+1)
      | \         |
      |  \        |
      |   \       |
      |    \      |
      |     \     |
      |      \    |
      |       \   |
  (i,   j)----(i+1, j)
*/
PetscErrorCode ComputeRHS(DMMG dmmg, Vec b)
{
  DA             da   = (DA)dmmg->dm;
  UserContext   *user = (UserContext *) dmmg->user;
  PetscScalar    phi  = user->phi;
  PetscScalar  **array;
  PetscReal      hx, hy;
  PetscInt       i,j,mx,my,xm,ym,xs,ys;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DAGetInfo(da, 0, &mx, &my, 0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  hx   = 1.0 / (PetscReal)(mx-1);
  hy   = 1.0 / (PetscReal)(my-1);
  ierr = DAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DAVecGetArray(da, b, &array);CHKERRQ(ierr);
  for (j=ys; j<ys+ym; j++){
    for(i=xs; i<xs+xm; i++){
      array[j][i] = phi;
    }
  }
  ierr = DAVecRestoreArray(da, b, &array);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeJacobian"
/*
  We integrate over each cell

  (i, j+1)----(i+1, j+1)
      | \         |
      |  \        |
      |   \       |
      |    \      |
      |     \     |
      |      \    |
      |       \   |
  (i,   j)----(i+1, j)

However, the element stiffnes matrix for the identity in linear elements is

  1  /2 1 1\
  -  |1 2 1|
  12 \1 1 2/

no matter what the shape of the triangle. The Laplacian stiffness matrix is

  1  /         (x_2 - x_1)^2 + (y_2 - y_1)^2           -(x_2 - x_0)(x_2 - x_1) - (y_2 - y_1)(y_2 - y_0)  (x_1 - x_0)(x_2 - x_1) + (y_1 - y_0)(y_2 - y_1)\
  -  |-(x_2 - x_0)(x_2 - x_1) - (y_2 - y_1)(y_2 - y_0)           (x_2 - x_0)^2 + (y_2 - y_0)^2          -(x_1 - x_0)(x_2 - x_0) - (y_1 - y_0)(y_2 - y_0)|
  A  \ (x_1 - x_0)(x_2 - x_1) + (y_1 - y_0)(y_2 - y_1) -(x_1 - x_0)(x_2 - x_0) - (y_1 - y_0)(y_2 - y_0)           (x_1 - x_0)^2 + (y_1 - y_0)^2         /

where A is the area of the triangle, and (x_i, y_i) is its i'th vertex.
*/
PetscErrorCode ComputeJacobian(DMMG dmmg, Mat jac)
{
  DA             da   = (DA) dmmg->dm;
  UserContext   *user = (UserContext *) dmmg->user;
  PetscScalar    phi  = user->phi;
  PetscScalar    identity[9] = {0.16666666667, 0.08333333333, 0.08333333333,
                                0.08333333333, 0.16666666667, 0.08333333333,
                                0.08333333333, 0.08333333333, 0.16666666667};
  PetscScalar    values[9];
  PetscInt       rows[3], cols[3];
  PetscReal      hx, hy, hx2, hy2, area;
  PetscInt       i,j,mx,my,xm,ym,xs,ys;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DAGetInfo(da, 0, &mx, &my, 0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  hx   = 1.0 / (PetscReal)(mx-1);
  hy   = 1.0 / (PetscReal)(my-1);
  area = 0.5*hx*hy;
  hx2  = hx*hx/area;
  hy2  = hy*hy/area;
  for(j = ys; j < ys+ym-1; j++) {
    for(i = xs; i < xs+xm-1; i++) {
      rows[0] = i;   cols[0] = j;
      rows[1] = i+1; cols[1] = j;
      rows[2] = i;   cols[2] = j+1;
      values[0] = hx2 + hy2; values[1] = -hy2; values[2] = -hx2;
      values[3] = -hy2;      values[4] = hy2;  values[5] = 0.0;
      values[6] = -hx2;      values[7] = 0.0;  values[8] = hx2;
      ierr = MatSetValues(jac,3,rows,3,cols,values,ADD_VALUES);CHKERRQ(ierr);
      rows[0] = i+1; cols[0] = j+1;
      rows[1] = i;   cols[1] = j+1;
      rows[2] = i+1; cols[2] = j;
      ierr = MatSetValues(jac,3,rows,3,cols,values,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ComputeCorrector"
PetscErrorCode ComputeCorrector(DMMG dmmg, Vec uOld, Vec u)
{
  DA             da   = (DA)dmmg->dm;
  PetscScalar    zero = 0.0;
  Vec            uOldLocal, uLocal;
  PetscScalar  **cOld;
  PetscScalar  **c;
  PetscInt       i,j,mx,my,xm,ym,xs,ys;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = VecSet(&zero, u);CHKERRQ(ierr);
  ierr = DAGetLocalVector(da, &uOldLocal);CHKERRQ(ierr);
  ierr = DAGetLocalVector(da, &uLocal);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da, uOld, INSERT_VALUES, uOldLocal);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da, uOld, INSERT_VALUES, uOldLocal);CHKERRQ(ierr);
  ierr = DAVecGetArray(da, uOldLocal, (void *) &cOld);CHKERRQ(ierr);
  ierr = DAVecGetArray(da, uLocal,    (void *) &c);CHKERRQ(ierr);
  ierr = DAGetInfo(da, 0, &mx, &my, 0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  for(j = ys; j < ys+ym; j++) {
    for(i = xs; i < xs+xm; i++) {
      c[j][i] = cOld[j][i];
    }
  }
  ierr = DAVecRestoreArray(da, uOldLocal, (void *) &cOld);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da, uLocal,    (void *) &c);CHKERRQ(ierr);
  ierr = DALocalToGlobal(da, uLocal, ADD_VALUES, u);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(da, &uOldLocal);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(da, &uLocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "dummy"
PetscErrorCode dummy(SNES snes, Vec x, void *ctx) {
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}
