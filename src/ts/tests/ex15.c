static char help[] ="Test conservation properties for 2-variable system\n\n";

/*F
We consider a linear reaction system with two concentrations

\begin{align}
  \frac{\partial c_0}{\partial t} &= -c_0 \\
  \frac{\partial c_1}{\partial t} &= c_0,
\end{align}

wherethe sum $c_0 + c_1$ is conserved, as can be seen by adding the two equations.

We now consider a different set of variables, defined implicitly by $c(u) = e^u$.  This type of transformation is
sometimes used to ensure positivity, and related transformations are sometimes used to develop a well-conditioned
formulation in limits such as zero Mach number.  In this instance, the relation is explicitly invertible, but that is
not always the case.  We can rewrite the differential equation in terms of non-conservative variables u,

\begin{align}
  \frac{\partial c_0}{\partial u_0} \frac{\partial u_0}{\partial t} &= -c_0(u_0) \\
  \frac{\partial c_1}{\partial u_1} \frac{\partial u_1}{\partial t} &= c_0(u_0).
\end{align}

We'll consider this three ways, each using an IFunction

1. CONSERVATIVE: standard integration in conservative variables: F(C, Cdot) = 0
2. NONCONSERVATIVE: chain rule formulation entirely in primitive variables: F(U, Udot) = 0
3. TRANSIENTVAR: Provide function C(U) and solve F(U, Cdot) = 0, where the time integrators handles the transformation

We will see that 1 and 3 are conservative (up to machine precision/solver tolerance, independent of temporal
discretization error) while 2 is not conservative (i.e., scales with temporal discretization error).

F*/

#include <petscts.h>

typedef enum {VAR_CONSERVATIVE, VAR_NONCONSERVATIVE, VAR_TRANSIENTVAR} VarMode;
static const char *const VarModes[] = {"CONSERVATIVE", "NONCONSERVATIVE", "TRANSIENTVAR", "VarMode", "VAR_", NULL};

static PetscErrorCode IFunction_Conservative(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,void *ctx)
{
  const PetscScalar *u,*udot;
  PetscScalar       *f;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  f[0] = udot[0] + u[0];
  f[1] = udot[1] - u[0];

  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IFunction_Nonconservative(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,void *ctx)
{
  const PetscScalar *u,*udot;
  PetscScalar       *f;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  f[0] = PetscExpScalar(u[0])*udot[0] + PetscExpScalar(u[0]);
  f[1] = PetscExpScalar(u[1])*udot[1] - PetscExpScalar(u[0]);

  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IFunction_TransientVar(TS ts,PetscReal t,Vec U,Vec Cdot,Vec F,void *ctx)
{
  const PetscScalar *u,*cdot;
  PetscScalar       *f;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Cdot,&cdot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  f[0] = cdot[0] + PetscExpScalar(u[0]);
  f[1] = cdot[1] - PetscExpScalar(u[0]);

  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Cdot,&cdot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TransientVar(TS ts,Vec U,Vec C,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(U,C);CHKERRQ(ierr);
  ierr = VecExp(C);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  TS             ts;
  DM             dm;
  Vec            U;
  VarMode        var = VAR_CONSERVATIVE;
  PetscScalar    sum;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"TS conservation example","");CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-var","Variable formulation",NULL,VarModes,(PetscEnum)var,(PetscEnum*)&var,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBDF);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,2,&U);CHKERRQ(ierr);
  ierr = VecSetValue(U,0,2.,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecSetValue(U,1,1.,INSERT_VALUES);CHKERRQ(ierr);
  switch (var) {
  case VAR_CONSERVATIVE:
    ierr = DMTSSetIFunction(dm,IFunction_Conservative,NULL);CHKERRQ(ierr);
    break;
  case VAR_NONCONSERVATIVE:
    ierr = VecLog(U);CHKERRQ(ierr);
    ierr = DMTSSetIFunction(dm,IFunction_Nonconservative,NULL);CHKERRQ(ierr);
    break;
  case VAR_TRANSIENTVAR:
    ierr = VecLog(U);CHKERRQ(ierr);
    ierr = DMTSSetIFunction(dm,IFunction_TransientVar,NULL);CHKERRQ(ierr);
    ierr = DMTSSetTransientVariable(dm,TransientVar,NULL);CHKERRQ(ierr);
  }
  ierr = TSSetMaxTime(ts,1.);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSSolve(ts,U);CHKERRQ(ierr);
  switch (var) {
  case VAR_CONSERVATIVE:
    break;
  case VAR_NONCONSERVATIVE:
  case VAR_TRANSIENTVAR:
    ierr = VecExp(U);CHKERRQ(ierr);
    break;
  }
  ierr = VecView(U,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecSum(U,&sum);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Conservation error %g\n", PetscRealPart(sum - 3.));CHKERRQ(ierr);

  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: conservative
    args: -snes_fd -var conservative
  test:
    suffix: nonconservative
    args: -snes_fd -var nonconservative
  test:
    suffix: transientvar
    args: -snes_fd -var transientvar

TEST*/
