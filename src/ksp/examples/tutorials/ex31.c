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

typedef struct {
  PetscScalar rho;
  PetscScalar rho_u;
  PetscScalar rho_v;
  PetscScalar rho_e;
} Fields;

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
  ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,3,3,PETSC_DECIDE,PETSC_DECIDE,4,1,0,0,&da);CHKERRQ(ierr);  
  ierr = DMMGSetDM(dmmg,(DM)da);
  ierr = DADestroy(da);CHKERRQ(ierr);
  for (l = 0; l < DMMGGetLevels(dmmg); l++) {
    ierr = DMMGSetUser(dmmg,l,&user);CHKERRQ(ierr);
  }

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, "", "Options for PCICE", "DMMG");
    user.phi = 0.5;
    ierr = PetscOptionsScalar("-phi", "The time weighting parameter", "ex31.c", user.phi, &user.phi, PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  ierr = ComputeInitialGuess(DMMGGetDMMG(dmmg), DMMGGetr(dmmg));
  ierr = ComputePredictor(DMMGGetDMMG(dmmg), DMMGGetr(dmmg), DMMGGetx(dmmg));

  ierr = DMMGSetKSP(dmmg,ComputeRHS,ComputeJacobian);CHKERRQ(ierr);
  ierr = DMMGSetInitialGuess(dmmg, DMMGInitialGuessCurrent);CHKERRQ(ierr);
  ierr = DMMGSolve(dmmg);CHKERRQ(ierr);

  ierr = ComputeCorrector(DMMGGetDMMG(dmmg), DMMGGetx(dmmg), DMMGGetr(dmmg));

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
  Fields        *pOld;
  Fields        *p;
  PetscInt       i,ne;
  const PetscInt *e;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = VecSet(&zero, u);CHKERRQ(ierr);
  ierr = DAGetLocalVector(da, &uOldLocal);CHKERRQ(ierr);
  ierr = DAGetLocalVector(da, &uLocal);CHKERRQ(ierr);
  ierr = VecSet(&zero, uLocal);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da, uOld, INSERT_VALUES, uOldLocal);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da, uOld, INSERT_VALUES, uOldLocal);CHKERRQ(ierr);
  ierr = VecGetArray(uOldLocal, (PetscScalar **) &pOld);CHKERRQ(ierr);
  ierr = VecGetArray(uLocal,    (PetscScalar **) &p);CHKERRQ(ierr);
  ierr = DAGetElements(da,&ne,&e);CHKERRQ(ierr);
  /* Source terms are all zero right now */
  for(i = 0; i < ne; i++) {
    /* Rich now is using element averages for all explicit values fed back into the finite element integrals. I think
       we should maintain them as unassembled sums of element functions. */
    /* Determine time-weighted values of \rho^{n+\phi} and (\rho\vu)^{n+\phi} */
    /* this is nonsense, but copy each nodal value */
    p[e[3*i]]   = pOld[e[3*i]];
    p[e[3*i+1]] = pOld[e[3*i+1]];
    p[e[3*i+2]] = pOld[e[3*i+2]];
  }
  /* Solve equation (9) for \delta(\rho\vu) and (\rho\vu)^* */
  /* Solve equation (13) for \delta\rho and \rho^* */
  /* Solve equation (15) for \delta(\rho e_t) and (\rho e_t)^* */
  /* Apply artifical dissipation */
  /* Determine the smoothed explicit pressure, \tilde P and temperature \tilde T using the equation of state */
  ierr = DARestoreElements(da,&ne,&e);CHKERRQ(ierr);

  ierr = VecRestoreArray(uOldLocal, (PetscScalar **) &pOld);CHKERRQ(ierr);
  ierr = VecRestoreArray(uLocal,    (PetscScalar **) &p);CHKERRQ(ierr);
  ierr = DALocalToGlobalBegin(da, uLocal, u);CHKERRQ(ierr);
  ierr = DALocalToGlobalEnd(da, uLocal, u);CHKERRQ(ierr);
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
  UserContext    *user = (UserContext *) dmmg->user;
  PetscScalar    phi  = user->phi;
  Fields        *array;
  PetscInt       ne,i;
  const PetscInt *e;
  PetscErrorCode ierr;
  Vec            blocal;

  PetscFunctionBegin;
  /* access a local vector with room for the ghost points */
  ierr = DAGetLocalVector(da,&blocal);CHKERRQ(ierr);
  ierr = VecGetArray(blocal, (PetscScalar **) &array);CHKERRQ(ierr);

  /* access the list of elements on this processor and loop over them */
  ierr = DAGetElements(da,&ne,&e);CHKERRQ(ierr);
  for (i=0; i<ne; i++) {

    /* this is nonsense, but set each nodal value to phi (will actually do integration over element */
    array[e[3*i]].rho   = phi;
    array[e[3*i+1]].rho = phi;
    array[e[3*i+2]].rho = phi;
  }
  ierr = VecRestoreArray(blocal, (PetscScalar **) &array);CHKERRQ(ierr);
  ierr = DARestoreElements(da,&ne,&e);CHKERRQ(ierr);

  /* add our partial sums over all processors into b */
  ierr = DALocalToGlobalBegin(da,blocal,b);CHKERRQ(ierr);
  ierr = DALocalToGlobalEnd(da,blocal,b);CHKERRQ(ierr);
  ierr = DARestoreLocalVector(da,&blocal);CHKERRQ(ierr);
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
  PetscScalar    values[3][3];
  PetscInt       idx[3];
  PetscReal      hx, hy, hx2, hy2, area;
  PetscInt       i,mx,my,xm,ym,xs,ys;
  PetscInt       ne;
  const PetscInt *e;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DAGetInfo(da, 0, &mx, &my, 0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  ierr = DAGetCorners(da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  hx   = 1.0 / (PetscReal)(mx-1);
  hy   = 1.0 / (PetscReal)(my-1);
  area = 0.5*hx*hy;
  hx2  = hx*hx/area;
  hy2  = hy*hy/area;

  /* initially all elements have identical geometry so all element stiffness are identical */
  values[0][0] = hx2 + hy2; values[0][1] = -hy2; values[0][2] = -hx2;
  values[1][0] = -hy2;      values[1][1] = hy2;  values[1][2] = 0.0;
  values[2][0] = -hx2;      values[2][1] = 0.0;  values[2][2] = hx2;

  ierr = DAGetElements(da,&ne,&e);CHKERRQ(ierr);
  for (i=0; i<ne; i++) {
    idx[0] = e[3*i];
    idx[1] = e[3*i+1];
    idx[2] = e[3*i+2];
    ierr = MatSetValuesLocal(jac,3,idx,3,idx,(PetscScalar*)values,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = DARestoreElements(da,&ne,&e);CHKERRQ(ierr);
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
  PetscScalar    *cOld;
  PetscScalar    *c;
  PetscInt       i,ne;
  const PetscInt *e;
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ierr = VecSet(&zero, u);CHKERRQ(ierr);
  ierr = DAGetLocalVector(da, &uOldLocal);CHKERRQ(ierr);
  ierr = DAGetLocalVector(da, &uLocal);CHKERRQ(ierr);
  ierr = VecSet(&zero, uLocal);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da, uOld, INSERT_VALUES, uOldLocal);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da, uOld, INSERT_VALUES, uOldLocal);CHKERRQ(ierr);
  ierr = VecGetArray(uOldLocal, (PetscScalar **) &cOld);CHKERRQ(ierr);
  ierr = VecGetArray(uLocal,    (PetscScalar **) &c);CHKERRQ(ierr);

  /* access the list of elements on this processor and loop over them */
  ierr = DAGetElements(da,&ne,&e);CHKERRQ(ierr);
  for (i=0; i<ne; i++) {

    /* this is nonsense, but copy each nodal value*/
    c[e[3*i]]   = cOld[e[3*i]];
    c[e[3*i+1]] = cOld[e[3*i+1]];
    c[e[3*i+2]] = cOld[e[3*i+2]];
  }
  ierr = DARestoreElements(da,&ne,&e);CHKERRQ(ierr);
  ierr = VecRestoreArray(uOldLocal, (PetscScalar **) &cOld);CHKERRQ(ierr);
  ierr = VecRestoreArray(uLocal,    (PetscScalar **) &c);CHKERRQ(ierr);
  ierr = DALocalToGlobalBegin(da, uLocal, u);CHKERRQ(ierr);
  ierr = DALocalToGlobalEnd(da, uLocal, u);CHKERRQ(ierr);
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
