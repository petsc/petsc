static char help[] = "Solves a ODE-constrained optimization problem -- finding the optimal initial conditions for the van der Pol equation.\n";

/*
  Concepts: TS^time-dependent nonlinear problems
  Concepts: TS^van der Pol equation DAE equivalent
  Concepts: TS^Optimization using adjoint sensitivity analysis
  Processors: 1
*/
/*
  Notes:
  This code demonstrates how to solve an ODE-constrained optimization problem with TAO, TSAdjoint and TS.
  The nonlinear problem is written in an ODE equivalent form.
  The objective is to minimize the difference between observation and model prediction by finding optimal values for initial conditions.
  The gradient is computed with the discrete adjoint of an implicit method or an explicit method, see ex20adj.c for details.
*/

#include <petsctao.h>
#include <petscts.h>

typedef struct _n_User *User;
struct _n_User {
  TS        ts;
  PetscReal mu;
  PetscReal next_output;

  /* Sensitivity analysis support */
  PetscInt  steps;
  PetscReal ftime;
  Mat       A;                       /* Jacobian matrix for ODE */
  Mat       Jacp;                    /* JacobianP matrix for ODE*/
  Mat       H;                       /* Hessian matrix for optimization */
  Vec       U,Lambda[1],Mup[1];      /* first-order adjoint variables */
  Vec       Lambda2[2];              /* second-order adjoint variables */
  Vec       Ihp1[1];                 /* working space for Hessian evaluations */
  Vec       Dir;                     /* direction vector */
  PetscReal ob[2];                   /* observation used by the cost function */
  PetscBool implicitform;            /* implicit ODE? */
};
PetscErrorCode Adjoint2(Vec,PetscScalar[],User);

/* ----------------------- Explicit form of the ODE  -------------------- */

static PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec U,Vec F,void *ctx)
{
  User              user = (User)ctx;
  PetscScalar       *f;
  const PetscScalar *u;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArray(F,&f));
  f[0] = u[1];
  f[1] = user->mu*((1.-u[0]*u[0])*u[1]-u[0]);
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec U,Mat A,Mat B,void *ctx)
{
  User              user = (User)ctx;
  PetscReal         mu   = user->mu;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *u;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U,&u));
  J[0][0] = 0;
  J[1][0] = -mu*(2.0*u[1]*u[0]+1.);
  J[0][1] = 1.0;
  J[1][1] = mu*(1.0-u[0]*u[0]);
  PetscCall(MatSetValues(A,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  if (A != B) {
    PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  }
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSHessianProductUU(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV,void *ctx)
{
  const PetscScalar *vl,*vr,*u;
  PetscScalar       *vhv;
  PetscScalar       dJdU[2][2][2]={{{0}}};
  PetscInt          i,j,k;
  User              user = (User)ctx;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArrayRead(Vl[0],&vl));
  PetscCall(VecGetArrayRead(Vr,&vr));
  PetscCall(VecGetArray(VHV[0],&vhv));

  dJdU[1][0][0] = -2.*user->mu*u[1];
  dJdU[1][1][0] = -2.*user->mu*u[0];
  dJdU[1][0][1] = -2.*user->mu*u[0];
  for (j=0;j<2;j++) {
    vhv[j] = 0;
    for (k=0;k<2;k++)
      for (i=0;i<2;i++)
        vhv[j] += vl[i]*dJdU[i][j][k]*vr[k];
  }
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(VecRestoreArrayRead(Vl[0],&vl));
  PetscCall(VecRestoreArrayRead(Vr,&vr));
  PetscCall(VecRestoreArray(VHV[0],&vhv));
  PetscFunctionReturn(0);
}

/* ----------------------- Implicit form of the ODE  -------------------- */

static PetscErrorCode IFunction(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,void *ctx)
{
  User              user = (User)ctx;
  const PetscScalar *u,*udot;
  PetscScalar       *f;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArrayRead(Udot,&udot));
  PetscCall(VecGetArray(F,&f));
  f[0] = udot[0] - u[1];
  f[1] = udot[1] - user->mu*((1.0-u[0]*u[0])*u[1] - u[0]) ;
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(VecRestoreArrayRead(Udot,&udot));
  PetscCall(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode IJacobian(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat A,Mat B,void *ctx)
{
  User              user = (User)ctx;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *u;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U,&u));
  J[0][0] = a;     J[0][1] =  -1.0;
  J[1][0] = user->mu*(1.0 + 2.0*u[0]*u[1]);   J[1][1] = a - user->mu*(1.0-u[0]*u[0]);
  PetscCall(MatSetValues(B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES));
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  if (A != B) {
    PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

/* Monitor timesteps and use interpolation to output at integer multiples of 0.1 */
static PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec U,void *ctx)
{
  const PetscScalar *u;
  PetscReal         tfinal, dt;
  User              user = (User)ctx;
  Vec               interpolatedU;

  PetscFunctionBeginUser;
  PetscCall(TSGetTimeStep(ts,&dt));
  PetscCall(TSGetMaxTime(ts,&tfinal));

  while (user->next_output <= t && user->next_output <= tfinal) {
    PetscCall(VecDuplicate(U,&interpolatedU));
    PetscCall(TSInterpolate(ts,user->next_output,interpolatedU));
    PetscCall(VecGetArrayRead(interpolatedU,&u));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"[%g] %D TS %g (dt = %g) X %g %g\n",
                        (double)user->next_output,step,(double)t,(double)dt,(double)PetscRealPart(u[0]),
                        (double)PetscRealPart(u[1])));
    PetscCall(VecRestoreArrayRead(interpolatedU,&u));
    PetscCall(VecDestroy(&interpolatedU));
    user->next_output += 0.1;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode IHessianProductUU(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV,void *ctx)
{
  const PetscScalar *vl,*vr,*u;
  PetscScalar       *vhv;
  PetscScalar       dJdU[2][2][2]={{{0}}};
  PetscInt          i,j,k;
  User              user = (User)ctx;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(U,&u));
  PetscCall(VecGetArrayRead(Vl[0],&vl));
  PetscCall(VecGetArrayRead(Vr,&vr));
  PetscCall(VecGetArray(VHV[0],&vhv));
  dJdU[1][0][0] = 2.*user->mu*u[1];
  dJdU[1][1][0] = 2.*user->mu*u[0];
  dJdU[1][0][1] = 2.*user->mu*u[0];
  for (j=0;j<2;j++) {
    vhv[j] = 0;
    for (k=0;k<2;k++)
      for (i=0;i<2;i++)
        vhv[j] += vl[i]*dJdU[i][j][k]*vr[k];
  }
  PetscCall(VecRestoreArrayRead(U,&u));
  PetscCall(VecRestoreArrayRead(Vl[0],&vl));
  PetscCall(VecRestoreArrayRead(Vr,&vr));
  PetscCall(VecRestoreArray(VHV[0],&vhv));
  PetscFunctionReturn(0);
}

/* ------------------ User-defined routines for TAO -------------------------- */

static PetscErrorCode FormFunctionGradient(Tao tao,Vec IC,PetscReal *f,Vec G,void *ctx)
{
  User              user_ptr = (User)ctx;
  TS                ts = user_ptr->ts;
  const PetscScalar *x_ptr;
  PetscScalar       *y_ptr;

  PetscFunctionBeginUser;
  PetscCall(VecCopy(IC,user_ptr->U)); /* set up the initial condition */

  PetscCall(TSSetTime(ts,0.0));
  PetscCall(TSSetStepNumber(ts,0));
  PetscCall(TSSetTimeStep(ts,0.001)); /* can be overwritten by command line options */
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSolve(ts,user_ptr->U));

  PetscCall(VecGetArrayRead(user_ptr->U,&x_ptr));
  PetscCall(VecGetArray(user_ptr->Lambda[0],&y_ptr));
  *f       = (x_ptr[0]-user_ptr->ob[0])*(x_ptr[0]-user_ptr->ob[0])+(x_ptr[1]-user_ptr->ob[1])*(x_ptr[1]-user_ptr->ob[1]);
  y_ptr[0] = 2.*(x_ptr[0]-user_ptr->ob[0]);
  y_ptr[1] = 2.*(x_ptr[1]-user_ptr->ob[1]);
  PetscCall(VecRestoreArray(user_ptr->Lambda[0],&y_ptr));
  PetscCall(VecRestoreArrayRead(user_ptr->U,&x_ptr));

  PetscCall(TSSetCostGradients(ts,1,user_ptr->Lambda,NULL));
  PetscCall(TSAdjointSolve(ts));
  PetscCall(VecCopy(user_ptr->Lambda[0],G));
  PetscFunctionReturn(0);
}

static PetscErrorCode FormHessian(Tao tao,Vec U,Mat H,Mat Hpre,void *ctx)
{
  User           user_ptr = (User)ctx;
  PetscScalar    harr[2];
  PetscScalar    *x_ptr;
  const PetscInt rows[2] = {0,1};
  PetscInt       col;

  PetscFunctionBeginUser;
  PetscCall(VecCopy(U,user_ptr->U));
  PetscCall(VecGetArray(user_ptr->Dir,&x_ptr));
  x_ptr[0] = 1.;
  x_ptr[1] = 0.;
  PetscCall(VecRestoreArray(user_ptr->Dir,&x_ptr));
  PetscCall(Adjoint2(user_ptr->U,harr,user_ptr));
  col      = 0;
  PetscCall(MatSetValues(H,2,rows,1,&col,harr,INSERT_VALUES));

  PetscCall(VecCopy(U,user_ptr->U));
  PetscCall(VecGetArray(user_ptr->Dir,&x_ptr));
  x_ptr[0] = 0.;
  x_ptr[1] = 1.;
  PetscCall(VecRestoreArray(user_ptr->Dir,&x_ptr));
  PetscCall(Adjoint2(user_ptr->U,harr,user_ptr));
  col      = 1;
  PetscCall(MatSetValues(H,2,rows,1,&col,harr,INSERT_VALUES));

  PetscCall(MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY));
  if (H != Hpre) {
    PetscCall(MatAssemblyBegin(Hpre,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Hpre,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatrixFreeHessian(Tao tao,Vec U,Mat H,Mat Hpre,void *ctx)
{
  User           user_ptr = (User)ctx;

  PetscFunctionBeginUser;
  PetscCall(VecCopy(U,user_ptr->U));
  PetscFunctionReturn(0);
}

/* ------------ Routines calculating second-order derivatives -------------- */

/*
  Compute the Hessian-vector product for the cost function using Second-order adjoint
*/
PetscErrorCode Adjoint2(Vec U,PetscScalar arr[],User ctx)
{
  TS             ts = ctx->ts;
  PetscScalar    *x_ptr,*y_ptr;
  Mat            tlmsen;

  PetscFunctionBeginUser;
  PetscCall(TSAdjointReset(ts));

  PetscCall(TSSetTime(ts,0.0));
  PetscCall(TSSetStepNumber(ts,0));
  PetscCall(TSSetTimeStep(ts,0.001));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSetCostHessianProducts(ts,1,ctx->Lambda2,NULL,ctx->Dir));
  PetscCall(TSAdjointSetForward(ts,NULL));
  PetscCall(TSSolve(ts,U));

  /* Set terminal conditions for first- and second-order adjonts */
  PetscCall(VecGetArray(U,&x_ptr));
  PetscCall(VecGetArray(ctx->Lambda[0],&y_ptr));
  y_ptr[0] = 2.*(x_ptr[0]-ctx->ob[0]);
  y_ptr[1] = 2.*(x_ptr[1]-ctx->ob[1]);
  PetscCall(VecRestoreArray(ctx->Lambda[0],&y_ptr));
  PetscCall(VecRestoreArray(U,&x_ptr));
  PetscCall(TSForwardGetSensitivities(ts,NULL,&tlmsen));
  PetscCall(MatDenseGetColumn(tlmsen,0,&x_ptr));
  PetscCall(VecGetArray(ctx->Lambda2[0],&y_ptr));
  y_ptr[0] = 2.*x_ptr[0];
  y_ptr[1] = 2.*x_ptr[1];
  PetscCall(VecRestoreArray(ctx->Lambda2[0],&y_ptr));
  PetscCall(MatDenseRestoreColumn(tlmsen,&x_ptr));

  PetscCall(TSSetCostGradients(ts,1,ctx->Lambda,NULL));
  if (ctx->implicitform) {
    PetscCall(TSSetIHessianProduct(ts,ctx->Ihp1,IHessianProductUU,NULL,NULL,NULL,NULL,NULL,NULL,ctx));
  } else {
    PetscCall(TSSetRHSHessianProduct(ts,ctx->Ihp1,RHSHessianProductUU,NULL,NULL,NULL,NULL,NULL,NULL,ctx));
  }
  PetscCall(TSAdjointSolve(ts));

  PetscCall(VecGetArray(ctx->Lambda2[0],&x_ptr));
  arr[0] = x_ptr[0];
  arr[1] = x_ptr[1];
  PetscCall(VecRestoreArray(ctx->Lambda2[0],&x_ptr));

  PetscCall(TSAdjointReset(ts));
  PetscCall(TSAdjointResetForward(ts));
  PetscFunctionReturn(0);
}

PetscErrorCode FiniteDiff(Vec U,PetscScalar arr[],User ctx)
{
  Vec               Up,G,Gp;
  const PetscScalar eps = PetscRealConstant(1e-7);
  PetscScalar       *u;
  Tao               tao = NULL;
  PetscReal         f;

  PetscFunctionBeginUser;
  PetscCall(VecDuplicate(U,&Up));
  PetscCall(VecDuplicate(U,&G));
  PetscCall(VecDuplicate(U,&Gp));

  PetscCall(FormFunctionGradient(tao,U,&f,G,ctx));

  PetscCall(VecCopy(U,Up));
  PetscCall(VecGetArray(Up,&u));
  u[0] += eps;
  PetscCall(VecRestoreArray(Up,&u));
  PetscCall(FormFunctionGradient(tao,Up,&f,Gp,ctx));
  PetscCall(VecAXPY(Gp,-1,G));
  PetscCall(VecScale(Gp,1./eps));
  PetscCall(VecGetArray(Gp,&u));
  arr[0] = u[0];
  arr[1] = u[1];
  PetscCall(VecRestoreArray(Gp,&u));

  PetscCall(VecCopy(U,Up));
  PetscCall(VecGetArray(Up,&u));
  u[1] += eps;
  PetscCall(VecRestoreArray(Up,&u));
  PetscCall(FormFunctionGradient(tao,Up,&f,Gp,ctx));
  PetscCall(VecAXPY(Gp,-1,G));
  PetscCall(VecScale(Gp,1./eps));
  PetscCall(VecGetArray(Gp,&u));
  arr[2] = u[0];
  arr[3] = u[1];
  PetscCall(VecRestoreArray(Gp,&u));

  PetscCall(VecDestroy(&G));
  PetscCall(VecDestroy(&Gp));
  PetscCall(VecDestroy(&Up));
  PetscFunctionReturn(0);
}

static PetscErrorCode HessianProductMat(Mat mat,Vec svec,Vec y)
{
  User           user_ptr;
  PetscScalar    *y_ptr;

  PetscFunctionBeginUser;
  PetscCall(MatShellGetContext(mat,&user_ptr));
  PetscCall(VecCopy(svec,user_ptr->Dir));
  PetscCall(VecGetArray(y,&y_ptr));
  PetscCall(Adjoint2(user_ptr->U,y_ptr,user_ptr));
  PetscCall(VecRestoreArray(y,&y_ptr));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscBool      monitor = PETSC_FALSE,mf = PETSC_TRUE;
  PetscInt       mode = 0;
  PetscMPIInt    size;
  struct _n_User user;
  Vec            x; /* working vector for TAO */
  PetscScalar    *x_ptr,arr[4];
  PetscScalar    ic1 = 2.2,ic2 = -0.7; /* initial guess for TAO */
  Tao            tao;
  KSP            ksp;
  PC             pc;

  /* Initialize program */
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /* Set runtime options */
  user.next_output  = 0.0;
  user.mu           = 1.0e3;
  user.steps        = 0;
  user.ftime        = 0.5;
  user.implicitform = PETSC_TRUE;

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-mu",&user.mu,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-mode",&mode,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-ic1",&ic1,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-ic2",&ic2,NULL));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-my_tao_mf",&mf,NULL)); /* matrix-free hessian for optimization */
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-implicitform",&user.implicitform,NULL));

  /* Create necessary matrix and vectors, solve same ODE on every process */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&user.A));
  PetscCall(MatSetSizes(user.A,PETSC_DECIDE,PETSC_DECIDE,2,2));
  PetscCall(MatSetFromOptions(user.A));
  PetscCall(MatSetUp(user.A));
  PetscCall(MatCreateVecs(user.A,&user.U,NULL));
  PetscCall(MatCreateVecs(user.A,&user.Dir,NULL));
  PetscCall(MatCreateVecs(user.A,&user.Lambda[0],NULL));
  PetscCall(MatCreateVecs(user.A,&user.Lambda2[0],NULL));
  PetscCall(MatCreateVecs(user.A,&user.Ihp1[0],NULL));

  /* Create timestepping solver context */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&user.ts));
  PetscCall(TSSetEquationType(user.ts,TS_EQ_ODE_EXPLICIT)); /* less Jacobian evaluations when adjoint BEuler is used, otherwise no effect */
  if (user.implicitform) {
    PetscCall(TSSetIFunction(user.ts,NULL,IFunction,&user));
    PetscCall(TSSetIJacobian(user.ts,user.A,user.A,IJacobian,&user));
    PetscCall(TSSetType(user.ts,TSCN));
  } else {
    PetscCall(TSSetRHSFunction(user.ts,NULL,RHSFunction,&user));
    PetscCall(TSSetRHSJacobian(user.ts,user.A,user.A,RHSJacobian,&user));
    PetscCall(TSSetType(user.ts,TSRK));
  }
  PetscCall(TSSetMaxTime(user.ts,user.ftime));
  PetscCall(TSSetExactFinalTime(user.ts,TS_EXACTFINALTIME_MATCHSTEP));

  if (monitor) {
    PetscCall(TSMonitorSet(user.ts,Monitor,&user,NULL));
  }

  /* Set ODE initial conditions */
  PetscCall(VecGetArray(user.U,&x_ptr));
  x_ptr[0] = 2.0;
  x_ptr[1] = -2.0/3.0 + 10.0/(81.0*user.mu) - 292.0/(2187.0*user.mu*user.mu);
  PetscCall(VecRestoreArray(user.U,&x_ptr));

  /* Set runtime options */
  PetscCall(TSSetFromOptions(user.ts));

  /* Obtain the observation */
  PetscCall(TSSolve(user.ts,user.U));
  PetscCall(VecGetArray(user.U,&x_ptr));
  user.ob[0] = x_ptr[0];
  user.ob[1] = x_ptr[1];
  PetscCall(VecRestoreArray(user.U,&x_ptr));

  PetscCall(VecDuplicate(user.U,&x));
  PetscCall(VecGetArray(x,&x_ptr));
  x_ptr[0] = ic1;
  x_ptr[1] = ic2;
  PetscCall(VecRestoreArray(x,&x_ptr));

  /* Save trajectory of solution so that TSAdjointSolve() may be used */
  PetscCall(TSSetSaveTrajectory(user.ts));

  /* Compare finite difference and second-order adjoint. */
  switch (mode) {
    case 2 :
      PetscCall(FiniteDiff(x,arr,&user));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\n Finite difference approximation of the Hessian\n"));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%g %g\n%g %g\n",(double)arr[0],(double)arr[1],(double)arr[2],(double)arr[3]));
      break;
    case 1 : /* Compute the Hessian column by column */
      PetscCall(VecCopy(x,user.U));
      PetscCall(VecGetArray(user.Dir,&x_ptr));
      x_ptr[0] = 1.;
      x_ptr[1] = 0.;
      PetscCall(VecRestoreArray(user.Dir,&x_ptr));
      PetscCall(Adjoint2(user.U,arr,&user));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nFirst column of the Hessian\n"));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%g\n%g\n",(double)arr[0],(double)arr[1]));
      PetscCall(VecCopy(x,user.U));
      PetscCall(VecGetArray(user.Dir,&x_ptr));
      x_ptr[0] = 0.;
      x_ptr[1] = 1.;
      PetscCall(VecRestoreArray(user.Dir,&x_ptr));
      PetscCall(Adjoint2(user.U,arr,&user));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nSecond column of the Hessian\n"));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%g\n%g\n",(double)arr[0],(double)arr[1]));
      break;
    case 0 :
      /* Create optimization context and set up */
      PetscCall(TaoCreate(PETSC_COMM_WORLD,&tao));
      PetscCall(TaoSetType(tao,TAOBLMVM));
      PetscCall(TaoSetObjectiveAndGradient(tao,NULL,FormFunctionGradient,(void *)&user));

      if (mf) {
        PetscCall(MatCreateShell(PETSC_COMM_SELF,2,2,2,2,(void*)&user,&user.H));
        PetscCall(MatShellSetOperation(user.H,MATOP_MULT,(void(*)(void))HessianProductMat));
        PetscCall(MatSetOption(user.H,MAT_SYMMETRIC,PETSC_TRUE));
        PetscCall(TaoSetHessian(tao,user.H,user.H,MatrixFreeHessian,(void *)&user));
      } else { /* Create Hessian matrix */
        PetscCall(MatCreate(PETSC_COMM_WORLD,&user.H));
        PetscCall(MatSetSizes(user.H,PETSC_DECIDE,PETSC_DECIDE,2,2));
        PetscCall(MatSetUp(user.H));
        PetscCall(TaoSetHessian(tao,user.H,user.H,FormHessian,(void *)&user));
      }

      /* Not use any preconditioner */
      PetscCall(TaoGetKSP(tao,&ksp));
      if (ksp) {
        PetscCall(KSPGetPC(ksp,&pc));
        PetscCall(PCSetType(pc,PCNONE));
      }

      /* Set initial solution guess */
      PetscCall(TaoSetSolution(tao,x));
      PetscCall(TaoSetFromOptions(tao));
      PetscCall(TaoSolve(tao));
      PetscCall(TaoDestroy(&tao));
      PetscCall(MatDestroy(&user.H));
      break;
    default:
      break;
  }

  /* Free work space.  All PETSc objects should be destroyed when they are no longer needed. */
  PetscCall(MatDestroy(&user.A));
  PetscCall(VecDestroy(&user.U));
  PetscCall(VecDestroy(&user.Lambda[0]));
  PetscCall(VecDestroy(&user.Lambda2[0]));
  PetscCall(VecDestroy(&user.Ihp1[0]));
  PetscCall(VecDestroy(&user.Dir));
  PetscCall(TSDestroy(&user.ts));
  PetscCall(VecDestroy(&x));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
    build:
      requires: !complex !single

    test:
      args:  -ts_type cn -viewer_binary_skip_info -tao_monitor -tao_view -mu 1000 -ts_dt 0.03125
      output_file: output/ex20opt_ic_1.out

    test:
      suffix: 2
      args:  -ts_type beuler -viewer_binary_skip_info -tao_monitor -tao_view -mu 100 -ts_dt 0.01 -tao_type bntr -tao_bnk_pc_type none
      output_file: output/ex20opt_ic_2.out

    test:
      suffix: 3
      args:  -ts_type cn -viewer_binary_skip_info -tao_monitor -tao_view -mu 100 -ts_dt 0.01 -tao_type bntr -tao_bnk_pc_type none
      output_file: output/ex20opt_ic_3.out

    test:
      suffix: 4
      args: -implicitform 0 -ts_dt 0.01 -ts_max_steps 2 -ts_rhs_jacobian_test_mult_transpose -mat_shell_test_mult_transpose_view -ts_rhs_jacobian_test_mult -mat_shell_test_mult_view -mode 1 -my_tao_mf
TEST*/
