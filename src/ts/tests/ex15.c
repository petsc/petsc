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

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(Udot,&udot));
  CHKERRQ(VecGetArray(F,&f));

  f[0] = udot[0] + u[0];
  f[1] = udot[1] - u[0];

  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(Udot,&udot));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode IFunction_Nonconservative(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,void *ctx)
{
  const PetscScalar *u,*udot;
  PetscScalar       *f;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(Udot,&udot));
  CHKERRQ(VecGetArray(F,&f));

  f[0] = PetscExpScalar(u[0])*udot[0] + PetscExpScalar(u[0]);
  f[1] = PetscExpScalar(u[1])*udot[1] - PetscExpScalar(u[0]);

  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(Udot,&udot));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode IFunction_TransientVar(TS ts,PetscReal t,Vec U,Vec Cdot,Vec F,void *ctx)
{
  const PetscScalar *u,*cdot;
  PetscScalar       *f;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(Cdot,&cdot));
  CHKERRQ(VecGetArray(F,&f));

  f[0] = cdot[0] + PetscExpScalar(u[0]);
  f[1] = cdot[1] - PetscExpScalar(u[0]);

  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(Cdot,&cdot));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode TransientVar(TS ts,Vec U,Vec C,void *ctx)
{
  PetscFunctionBegin;
  CHKERRQ(VecCopy(U,C));
  CHKERRQ(VecExp(C));
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
  CHKERRQ(PetscOptionsEnum("-var","Variable formulation",NULL,VarModes,(PetscEnum)var,(PetscEnum*)&var,NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetType(ts,TSBDF));
  CHKERRQ(TSGetDM(ts,&dm));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,2,&U));
  CHKERRQ(VecSetValue(U,0,2.,INSERT_VALUES));
  CHKERRQ(VecSetValue(U,1,1.,INSERT_VALUES));
  switch (var) {
  case VAR_CONSERVATIVE:
    CHKERRQ(DMTSSetIFunction(dm,IFunction_Conservative,NULL));
    break;
  case VAR_NONCONSERVATIVE:
    CHKERRQ(VecLog(U));
    CHKERRQ(DMTSSetIFunction(dm,IFunction_Nonconservative,NULL));
    break;
  case VAR_TRANSIENTVAR:
    CHKERRQ(VecLog(U));
    CHKERRQ(DMTSSetIFunction(dm,IFunction_TransientVar,NULL));
    CHKERRQ(DMTSSetTransientVariable(dm,TransientVar,NULL));
  }
  CHKERRQ(TSSetMaxTime(ts,1.));
  CHKERRQ(TSSetFromOptions(ts));

  CHKERRQ(TSSolve(ts,U));
  switch (var) {
  case VAR_CONSERVATIVE:
    break;
  case VAR_NONCONSERVATIVE:
  case VAR_TRANSIENTVAR:
    CHKERRQ(VecExp(U));
    break;
  }
  CHKERRQ(VecView(U,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecSum(U,&sum));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Conservation error %g\n", PetscRealPart(sum - 3.)));

  CHKERRQ(VecDestroy(&U));
  CHKERRQ(TSDestroy(&ts));
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
