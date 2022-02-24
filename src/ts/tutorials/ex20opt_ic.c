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
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArray(F,&f));
  f[0] = u[1];
  f[1] = user->mu*((1.-u[0]*u[0])*u[1]-u[0]);
  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArray(F,&f));
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
  CHKERRQ(VecGetArrayRead(U,&u));
  J[0][0] = 0;
  J[1][0] = -mu*(2.0*u[1]*u[0]+1.);
  J[0][1] = 1.0;
  J[1][1] = mu*(1.0-u[0]*u[0]);
  CHKERRQ(MatSetValues(A,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  if (A != B) {
    CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  }
  CHKERRQ(VecRestoreArrayRead(U,&u));
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
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(Vl[0],&vl));
  CHKERRQ(VecGetArrayRead(Vr,&vr));
  CHKERRQ(VecGetArray(VHV[0],&vhv));

  dJdU[1][0][0] = -2.*user->mu*u[1];
  dJdU[1][1][0] = -2.*user->mu*u[0];
  dJdU[1][0][1] = -2.*user->mu*u[0];
  for (j=0;j<2;j++) {
    vhv[j] = 0;
    for (k=0;k<2;k++)
      for (i=0;i<2;i++)
        vhv[j] += vl[i]*dJdU[i][j][k]*vr[k];
  }
  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(Vl[0],&vl));
  CHKERRQ(VecRestoreArrayRead(Vr,&vr));
  CHKERRQ(VecRestoreArray(VHV[0],&vhv));
  PetscFunctionReturn(0);
}

/* ----------------------- Implicit form of the ODE  -------------------- */

static PetscErrorCode IFunction(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,void *ctx)
{
  User              user = (User)ctx;
  const PetscScalar *u,*udot;
  PetscScalar       *f;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(Udot,&udot));
  CHKERRQ(VecGetArray(F,&f));
  f[0] = udot[0] - u[1];
  f[1] = udot[1] - user->mu*((1.0-u[0]*u[0])*u[1] - u[0]) ;
  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(Udot,&udot));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode IJacobian(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat A,Mat B,void *ctx)
{
  User              user = (User)ctx;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *u;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(U,&u));
  J[0][0] = a;     J[0][1] =  -1.0;
  J[1][0] = user->mu*(1.0 + 2.0*u[0]*u[1]);   J[1][1] = a - user->mu*(1.0-u[0]*u[0]);
  CHKERRQ(MatSetValues(B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES));
  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  if (A != B) {
    CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

/* Monitor timesteps and use interpolation to output at integer multiples of 0.1 */
static PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec U,void *ctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *u;
  PetscReal         tfinal, dt;
  User              user = (User)ctx;
  Vec               interpolatedU;

  PetscFunctionBeginUser;
  CHKERRQ(TSGetTimeStep(ts,&dt));
  CHKERRQ(TSGetMaxTime(ts,&tfinal));

  while (user->next_output <= t && user->next_output <= tfinal) {
    CHKERRQ(VecDuplicate(U,&interpolatedU));
    CHKERRQ(TSInterpolate(ts,user->next_output,interpolatedU));
    CHKERRQ(VecGetArrayRead(interpolatedU,&u));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"[%g] %D TS %g (dt = %g) X %g %g\n",
                       (double)user->next_output,step,(double)t,(double)dt,(double)PetscRealPart(u[0]),
                       (double)PetscRealPart(u[1]));CHKERRQ(ierr);
    CHKERRQ(VecRestoreArrayRead(interpolatedU,&u));
    CHKERRQ(VecDestroy(&interpolatedU));
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
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(Vl[0],&vl));
  CHKERRQ(VecGetArrayRead(Vr,&vr));
  CHKERRQ(VecGetArray(VHV[0],&vhv));
  dJdU[1][0][0] = 2.*user->mu*u[1];
  dJdU[1][1][0] = 2.*user->mu*u[0];
  dJdU[1][0][1] = 2.*user->mu*u[0];
  for (j=0;j<2;j++) {
    vhv[j] = 0;
    for (k=0;k<2;k++)
      for (i=0;i<2;i++)
        vhv[j] += vl[i]*dJdU[i][j][k]*vr[k];
  }
  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(Vl[0],&vl));
  CHKERRQ(VecRestoreArrayRead(Vr,&vr));
  CHKERRQ(VecRestoreArray(VHV[0],&vhv));
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
  CHKERRQ(VecCopy(IC,user_ptr->U)); /* set up the initial condition */

  CHKERRQ(TSSetTime(ts,0.0));
  CHKERRQ(TSSetStepNumber(ts,0));
  CHKERRQ(TSSetTimeStep(ts,0.001)); /* can be overwritten by command line options */
  CHKERRQ(TSSetFromOptions(ts));
  CHKERRQ(TSSolve(ts,user_ptr->U));

  CHKERRQ(VecGetArrayRead(user_ptr->U,&x_ptr));
  CHKERRQ(VecGetArray(user_ptr->Lambda[0],&y_ptr));
  *f       = (x_ptr[0]-user_ptr->ob[0])*(x_ptr[0]-user_ptr->ob[0])+(x_ptr[1]-user_ptr->ob[1])*(x_ptr[1]-user_ptr->ob[1]);
  y_ptr[0] = 2.*(x_ptr[0]-user_ptr->ob[0]);
  y_ptr[1] = 2.*(x_ptr[1]-user_ptr->ob[1]);
  CHKERRQ(VecRestoreArray(user_ptr->Lambda[0],&y_ptr));
  CHKERRQ(VecRestoreArrayRead(user_ptr->U,&x_ptr));

  CHKERRQ(TSSetCostGradients(ts,1,user_ptr->Lambda,NULL));
  CHKERRQ(TSAdjointSolve(ts));
  CHKERRQ(VecCopy(user_ptr->Lambda[0],G));
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
  CHKERRQ(VecCopy(U,user_ptr->U));
  CHKERRQ(VecGetArray(user_ptr->Dir,&x_ptr));
  x_ptr[0] = 1.;
  x_ptr[1] = 0.;
  CHKERRQ(VecRestoreArray(user_ptr->Dir,&x_ptr));
  CHKERRQ(Adjoint2(user_ptr->U,harr,user_ptr));
  col      = 0;
  CHKERRQ(MatSetValues(H,2,rows,1,&col,harr,INSERT_VALUES));

  CHKERRQ(VecCopy(U,user_ptr->U));
  CHKERRQ(VecGetArray(user_ptr->Dir,&x_ptr));
  x_ptr[0] = 0.;
  x_ptr[1] = 1.;
  CHKERRQ(VecRestoreArray(user_ptr->Dir,&x_ptr));
  CHKERRQ(Adjoint2(user_ptr->U,harr,user_ptr));
  col      = 1;
  CHKERRQ(MatSetValues(H,2,rows,1,&col,harr,INSERT_VALUES));

  CHKERRQ(MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY));
  if (H != Hpre) {
    CHKERRQ(MatAssemblyBegin(Hpre,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(Hpre,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatrixFreeHessian(Tao tao,Vec U,Mat H,Mat Hpre,void *ctx)
{
  User           user_ptr = (User)ctx;

  PetscFunctionBeginUser;
  CHKERRQ(VecCopy(U,user_ptr->U));
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
  CHKERRQ(TSAdjointReset(ts));

  CHKERRQ(TSSetTime(ts,0.0));
  CHKERRQ(TSSetStepNumber(ts,0));
  CHKERRQ(TSSetTimeStep(ts,0.001));
  CHKERRQ(TSSetFromOptions(ts));
  CHKERRQ(TSSetCostHessianProducts(ts,1,ctx->Lambda2,NULL,ctx->Dir));
  CHKERRQ(TSAdjointSetForward(ts,NULL));
  CHKERRQ(TSSolve(ts,U));

  /* Set terminal conditions for first- and second-order adjonts */
  CHKERRQ(VecGetArray(U,&x_ptr));
  CHKERRQ(VecGetArray(ctx->Lambda[0],&y_ptr));
  y_ptr[0] = 2.*(x_ptr[0]-ctx->ob[0]);
  y_ptr[1] = 2.*(x_ptr[1]-ctx->ob[1]);
  CHKERRQ(VecRestoreArray(ctx->Lambda[0],&y_ptr));
  CHKERRQ(VecRestoreArray(U,&x_ptr));
  CHKERRQ(TSForwardGetSensitivities(ts,NULL,&tlmsen));
  CHKERRQ(MatDenseGetColumn(tlmsen,0,&x_ptr));
  CHKERRQ(VecGetArray(ctx->Lambda2[0],&y_ptr));
  y_ptr[0] = 2.*x_ptr[0];
  y_ptr[1] = 2.*x_ptr[1];
  CHKERRQ(VecRestoreArray(ctx->Lambda2[0],&y_ptr));
  CHKERRQ(MatDenseRestoreColumn(tlmsen,&x_ptr));

  CHKERRQ(TSSetCostGradients(ts,1,ctx->Lambda,NULL));
  if (ctx->implicitform) {
    CHKERRQ(TSSetIHessianProduct(ts,ctx->Ihp1,IHessianProductUU,NULL,NULL,NULL,NULL,NULL,NULL,ctx));
  } else {
    CHKERRQ(TSSetRHSHessianProduct(ts,ctx->Ihp1,RHSHessianProductUU,NULL,NULL,NULL,NULL,NULL,NULL,ctx));
  }
  CHKERRQ(TSAdjointSolve(ts));

  CHKERRQ(VecGetArray(ctx->Lambda2[0],&x_ptr));
  arr[0] = x_ptr[0];
  arr[1] = x_ptr[1];
  CHKERRQ(VecRestoreArray(ctx->Lambda2[0],&x_ptr));

  CHKERRQ(TSAdjointReset(ts));
  CHKERRQ(TSAdjointResetForward(ts));
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
  CHKERRQ(VecDuplicate(U,&Up));
  CHKERRQ(VecDuplicate(U,&G));
  CHKERRQ(VecDuplicate(U,&Gp));

  CHKERRQ(FormFunctionGradient(tao,U,&f,G,ctx));

  CHKERRQ(VecCopy(U,Up));
  CHKERRQ(VecGetArray(Up,&u));
  u[0] += eps;
  CHKERRQ(VecRestoreArray(Up,&u));
  CHKERRQ(FormFunctionGradient(tao,Up,&f,Gp,ctx));
  CHKERRQ(VecAXPY(Gp,-1,G));
  CHKERRQ(VecScale(Gp,1./eps));
  CHKERRQ(VecGetArray(Gp,&u));
  arr[0] = u[0];
  arr[1] = u[1];
  CHKERRQ(VecRestoreArray(Gp,&u));

  CHKERRQ(VecCopy(U,Up));
  CHKERRQ(VecGetArray(Up,&u));
  u[1] += eps;
  CHKERRQ(VecRestoreArray(Up,&u));
  CHKERRQ(FormFunctionGradient(tao,Up,&f,Gp,ctx));
  CHKERRQ(VecAXPY(Gp,-1,G));
  CHKERRQ(VecScale(Gp,1./eps));
  CHKERRQ(VecGetArray(Gp,&u));
  arr[2] = u[0];
  arr[3] = u[1];
  CHKERRQ(VecRestoreArray(Gp,&u));

  CHKERRQ(VecDestroy(&G));
  CHKERRQ(VecDestroy(&Gp));
  CHKERRQ(VecDestroy(&Up));
  PetscFunctionReturn(0);
}

static PetscErrorCode HessianProductMat(Mat mat,Vec svec,Vec y)
{
  User           user_ptr;
  PetscScalar    *y_ptr;

  PetscFunctionBeginUser;
  CHKERRQ(MatShellGetContext(mat,&user_ptr));
  CHKERRQ(VecCopy(svec,user_ptr->Dir));
  CHKERRQ(VecGetArray(y,&y_ptr));
  CHKERRQ(Adjoint2(user_ptr->U,y_ptr,user_ptr));
  CHKERRQ(VecRestoreArray(y,&y_ptr));
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
  PetscErrorCode ierr;

  /* Initialize program */
  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /* Set runtime options */
  user.next_output  = 0.0;
  user.mu           = 1.0e3;
  user.steps        = 0;
  user.ftime        = 0.5;
  user.implicitform = PETSC_TRUE;

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-mu",&user.mu,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mode",&mode,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-ic1",&ic1,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-ic2",&ic2,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-my_tao_mf",&mf,NULL)); /* matrix-free hessian for optimization */
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-implicitform",&user.implicitform,NULL));

  /* Create necessary matrix and vectors, solve same ODE on every process */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user.A));
  CHKERRQ(MatSetSizes(user.A,PETSC_DECIDE,PETSC_DECIDE,2,2));
  CHKERRQ(MatSetFromOptions(user.A));
  CHKERRQ(MatSetUp(user.A));
  CHKERRQ(MatCreateVecs(user.A,&user.U,NULL));
  CHKERRQ(MatCreateVecs(user.A,&user.Dir,NULL));
  CHKERRQ(MatCreateVecs(user.A,&user.Lambda[0],NULL));
  CHKERRQ(MatCreateVecs(user.A,&user.Lambda2[0],NULL));
  CHKERRQ(MatCreateVecs(user.A,&user.Ihp1[0],NULL));

  /* Create timestepping solver context */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&user.ts));
  CHKERRQ(TSSetEquationType(user.ts,TS_EQ_ODE_EXPLICIT)); /* less Jacobian evaluations when adjoint BEuler is used, otherwise no effect */
  if (user.implicitform) {
    CHKERRQ(TSSetIFunction(user.ts,NULL,IFunction,&user));
    CHKERRQ(TSSetIJacobian(user.ts,user.A,user.A,IJacobian,&user));
    CHKERRQ(TSSetType(user.ts,TSCN));
  } else {
    CHKERRQ(TSSetRHSFunction(user.ts,NULL,RHSFunction,&user));
    CHKERRQ(TSSetRHSJacobian(user.ts,user.A,user.A,RHSJacobian,&user));
    CHKERRQ(TSSetType(user.ts,TSRK));
  }
  CHKERRQ(TSSetMaxTime(user.ts,user.ftime));
  CHKERRQ(TSSetExactFinalTime(user.ts,TS_EXACTFINALTIME_MATCHSTEP));

  if (monitor) {
    CHKERRQ(TSMonitorSet(user.ts,Monitor,&user,NULL));
  }

  /* Set ODE initial conditions */
  CHKERRQ(VecGetArray(user.U,&x_ptr));
  x_ptr[0] = 2.0;
  x_ptr[1] = -2.0/3.0 + 10.0/(81.0*user.mu) - 292.0/(2187.0*user.mu*user.mu);
  CHKERRQ(VecRestoreArray(user.U,&x_ptr));

  /* Set runtime options */
  CHKERRQ(TSSetFromOptions(user.ts));

  /* Obtain the observation */
  CHKERRQ(TSSolve(user.ts,user.U));
  CHKERRQ(VecGetArray(user.U,&x_ptr));
  user.ob[0] = x_ptr[0];
  user.ob[1] = x_ptr[1];
  CHKERRQ(VecRestoreArray(user.U,&x_ptr));

  CHKERRQ(VecDuplicate(user.U,&x));
  CHKERRQ(VecGetArray(x,&x_ptr));
  x_ptr[0] = ic1;
  x_ptr[1] = ic2;
  CHKERRQ(VecRestoreArray(x,&x_ptr));

  /* Save trajectory of solution so that TSAdjointSolve() may be used */
  CHKERRQ(TSSetSaveTrajectory(user.ts));

  /* Compare finite difference and second-order adjoint. */
  switch (mode) {
    case 2 :
      CHKERRQ(FiniteDiff(x,arr,&user));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\n Finite difference approximation of the Hessian\n"));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%g %g\n%g %g\n",(double)arr[0],(double)arr[1],(double)arr[2],(double)arr[3]));
      break;
    case 1 : /* Compute the Hessian column by column */
      CHKERRQ(VecCopy(x,user.U));
      CHKERRQ(VecGetArray(user.Dir,&x_ptr));
      x_ptr[0] = 1.;
      x_ptr[1] = 0.;
      CHKERRQ(VecRestoreArray(user.Dir,&x_ptr));
      CHKERRQ(Adjoint2(user.U,arr,&user));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nFirst column of the Hessian\n"));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%g\n%g\n",(double)arr[0],(double)arr[1]));
      CHKERRQ(VecCopy(x,user.U));
      CHKERRQ(VecGetArray(user.Dir,&x_ptr));
      x_ptr[0] = 0.;
      x_ptr[1] = 1.;
      CHKERRQ(VecRestoreArray(user.Dir,&x_ptr));
      CHKERRQ(Adjoint2(user.U,arr,&user));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nSecond column of the Hessian\n"));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%g\n%g\n",(double)arr[0],(double)arr[1]));
      break;
    case 0 :
      /* Create optimization context and set up */
      CHKERRQ(TaoCreate(PETSC_COMM_WORLD,&tao));
      CHKERRQ(TaoSetType(tao,TAOBLMVM));
      CHKERRQ(TaoSetObjectiveAndGradient(tao,NULL,FormFunctionGradient,(void *)&user));

      if (mf) {
        CHKERRQ(MatCreateShell(PETSC_COMM_SELF,2,2,2,2,(void*)&user,&user.H));
        CHKERRQ(MatShellSetOperation(user.H,MATOP_MULT,(void(*)(void))HessianProductMat));
        CHKERRQ(MatSetOption(user.H,MAT_SYMMETRIC,PETSC_TRUE));
        CHKERRQ(TaoSetHessian(tao,user.H,user.H,MatrixFreeHessian,(void *)&user));
      } else { /* Create Hessian matrix */
        CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user.H));
        CHKERRQ(MatSetSizes(user.H,PETSC_DECIDE,PETSC_DECIDE,2,2));
        CHKERRQ(MatSetUp(user.H));
        CHKERRQ(TaoSetHessian(tao,user.H,user.H,FormHessian,(void *)&user));
      }

      /* Not use any preconditioner */
      CHKERRQ(TaoGetKSP(tao,&ksp));
      if (ksp) {
        CHKERRQ(KSPGetPC(ksp,&pc));
        CHKERRQ(PCSetType(pc,PCNONE));
      }

      /* Set initial solution guess */
      CHKERRQ(TaoSetSolution(tao,x));
      CHKERRQ(TaoSetFromOptions(tao));
      CHKERRQ(TaoSolve(tao));
      CHKERRQ(TaoDestroy(&tao));
      CHKERRQ(MatDestroy(&user.H));
      break;
    default:
      break;
  }

  /* Free work space.  All PETSc objects should be destroyed when they are no longer needed. */
  CHKERRQ(MatDestroy(&user.A));
  CHKERRQ(VecDestroy(&user.U));
  CHKERRQ(VecDestroy(&user.Lambda[0]));
  CHKERRQ(VecDestroy(&user.Lambda2[0]));
  CHKERRQ(VecDestroy(&user.Ihp1[0]));
  CHKERRQ(VecDestroy(&user.Dir));
  CHKERRQ(TSDestroy(&user.ts));
  CHKERRQ(VecDestroy(&x));
  ierr = PetscFinalize();
  return(ierr);
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
