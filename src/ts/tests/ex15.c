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

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArrayRead(Udot,&udot));
  PetscCall(VecGetArray(F,&f));

  f[0] = udot[0] + u[0];
  f[1] = udot[1] - u[0];

  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(VecRestoreArrayRead(Udot,&udot));
  PetscCall(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode IFunction_Nonconservative(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,void *ctx)
{
  const PetscScalar *u,*udot;
  PetscScalar       *f;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArrayRead(Udot,&udot));
  PetscCall(VecGetArray(F,&f));

  f[0] = PetscExpScalar(u[0])*udot[0] + PetscExpScalar(u[0]);
  f[1] = PetscExpScalar(u[1])*udot[1] - PetscExpScalar(u[0]);

  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(VecRestoreArrayRead(Udot,&udot));
  PetscCall(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode IFunction_TransientVar(TS ts,PetscReal t,Vec U,Vec Cdot,Vec F,void *ctx)
{
  const PetscScalar *u,*cdot;
  PetscScalar       *f;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArrayRead(Cdot,&cdot));
  PetscCall(VecGetArray(F,&f));

  f[0] = cdot[0] + PetscExpScalar(u[0]);
  f[1] = cdot[1] - PetscExpScalar(u[0]);

  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(VecRestoreArrayRead(Cdot,&cdot));
  PetscCall(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode TransientVar(TS ts,Vec U,Vec C,void *ctx)
{
  PetscFunctionBeginUser;
  PetscCall(VecCopy(U,C));
  PetscCall(VecExp(C));
  PetscFunctionReturn(0);
}

int main(int argc, char *argv[])
{
  TS             ts;
  DM             dm;
  Vec            U;
  VarMode        var = VAR_CONSERVATIVE;
  PetscScalar    sum;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"TS conservation example","");
  PetscCall(PetscOptionsEnum("-var","Variable formulation",NULL,VarModes,(PetscEnum)var,(PetscEnum*)&var,NULL));
  PetscOptionsEnd();

  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetType(ts,TSBDF));
  PetscCall(TSGetDM(ts,&dm));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,2,&U));
  PetscCall(VecSetValue(U,0,2.,INSERT_VALUES));
  PetscCall(VecSetValue(U,1,1.,INSERT_VALUES));
  switch (var) {
  case VAR_CONSERVATIVE:
    PetscCall(DMTSSetIFunction(dm,IFunction_Conservative,NULL));
    break;
  case VAR_NONCONSERVATIVE:
    PetscCall(VecLog(U));
    PetscCall(DMTSSetIFunction(dm,IFunction_Nonconservative,NULL));
    break;
  case VAR_TRANSIENTVAR:
    PetscCall(VecLog(U));
    PetscCall(DMTSSetIFunction(dm,IFunction_TransientVar,NULL));
    PetscCall(DMTSSetTransientVariable(dm,TransientVar,NULL));
  }
  PetscCall(TSSetMaxTime(ts,1.));
  PetscCall(TSSetFromOptions(ts));

  PetscCall(TSSolve(ts,U));
  switch (var) {
  case VAR_CONSERVATIVE:
    break;
  case VAR_NONCONSERVATIVE:
  case VAR_TRANSIENTVAR:
    PetscCall(VecExp(U));
    break;
  }
  PetscCall(VecView(U,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecSum(U,&sum));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Conservation error %g\n", (double)PetscRealPart(sum - 3.)));

  PetscCall(VecDestroy(&U));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFinalize());
  return 0;
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
