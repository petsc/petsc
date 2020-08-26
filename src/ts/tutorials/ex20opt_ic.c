static char help[] = "Solves a ODE-constrained optimization problem -- finding the optimal initial conditions for the van der Pol equation.\n";

/**
  Concepts: TS^time-dependent nonlinear problems
  Concepts: TS^van der Pol equation DAE equivalent
  Concepts: TS^Optimization using adjoint sensitivity analysis
  Processors: 1
*/
/**
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
  PetscErrorCode    ierr;
  User              user = (User)ctx;
  PetscScalar       *f;
  const PetscScalar *u;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = u[1];
  f[1] = user->mu*((1.-u[0]*u[0])*u[1]-u[0]);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSJacobian(TS ts,PetscReal t,Vec U,Mat A,Mat B,void *ctx)
{
  PetscErrorCode    ierr;
  User              user = (User)ctx;
  PetscReal         mu   = user->mu;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *u;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  J[0][0] = 0;
  J[1][0] = -mu*(2.0*u[1]*u[0]+1.);
  J[0][1] = 1.0;
  J[1][1] = mu*(1.0-u[0]*u[0]);
  ierr    = MatSetValues(A,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSHessianProductUU(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV,void *ctx)
{
  const PetscScalar *vl,*vr,*u;
  PetscScalar       *vhv;
  PetscScalar       dJdU[2][2][2]={{{0}}};
  PetscInt          i,j,k;
  User              user = (User)ctx;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Vl[0],&vl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Vr,&vr);CHKERRQ(ierr);
  ierr = VecGetArray(VHV[0],&vhv);CHKERRQ(ierr);

  dJdU[1][0][0] = -2.*user->mu*u[1];
  dJdU[1][1][0] = -2.*user->mu*u[0];
  dJdU[1][0][1] = -2.*user->mu*u[0];
  for (j=0;j<2;j++) {
    vhv[j] = 0;
    for (k=0;k<2;k++)
      for (i=0;i<2;i++)
        vhv[j] += vl[i]*dJdU[i][j][k]*vr[k];
  }
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Vl[0],&vl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Vr,&vr);CHKERRQ(ierr);
  ierr = VecRestoreArray(VHV[0],&vhv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------- Implicit form of the ODE  -------------------- */

static PetscErrorCode IFunction(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,void *ctx)
{
  PetscErrorCode    ierr;
  User              user = (User)ctx;
  const PetscScalar *u,*udot;
  PetscScalar       *f;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = udot[0] - u[1];
  f[1] = udot[1] - user->mu*((1.0-u[0]*u[0])*u[1] - u[0]) ;
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IJacobian(TS ts,PetscReal t,Vec U,Vec Udot,PetscReal a,Mat A,Mat B,void *ctx)
{
  PetscErrorCode    ierr;
  User              user = (User)ctx;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *u;

  PetscFunctionBeginUser;
  ierr    = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  J[0][0] = a;     J[0][1] =  -1.0;
  J[1][0] = user->mu*(1.0 + 2.0*u[0]*u[1]);   J[1][1] = a - user->mu*(1.0-u[0]*u[0]);
  ierr    = MatSetValues(B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr    = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr    = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr    = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr  = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr  = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
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
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  ierr = TSGetMaxTime(ts,&tfinal);CHKERRQ(ierr);

  while (user->next_output <= t && user->next_output <= tfinal) {
    ierr = VecDuplicate(U,&interpolatedU);CHKERRQ(ierr);
    ierr = TSInterpolate(ts,user->next_output,interpolatedU);CHKERRQ(ierr);
    ierr = VecGetArrayRead(interpolatedU,&u);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"[%g] %D TS %g (dt = %g) X %g %g\n",
                       (double)user->next_output,step,(double)t,(double)dt,(double)PetscRealPart(u[0]),
                       (double)PetscRealPart(u[1]));CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(interpolatedU,&u);CHKERRQ(ierr);
    ierr = VecDestroy(&interpolatedU);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr          = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr          = VecGetArrayRead(Vl[0],&vl);CHKERRQ(ierr);
  ierr          = VecGetArrayRead(Vr,&vr);CHKERRQ(ierr);
  ierr          = VecGetArray(VHV[0],&vhv);CHKERRQ(ierr);
  dJdU[1][0][0] = 2.*user->mu*u[1];
  dJdU[1][1][0] = 2.*user->mu*u[0];
  dJdU[1][0][1] = 2.*user->mu*u[0];
  for (j=0;j<2;j++) {
    vhv[j] = 0;
    for (k=0;k<2;k++)
      for (i=0;i<2;i++)
        vhv[j] += vl[i]*dJdU[i][j][k]*vr[k];
  }
  ierr          = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr          = VecRestoreArrayRead(Vl[0],&vl);CHKERRQ(ierr);
  ierr          = VecRestoreArrayRead(Vr,&vr);CHKERRQ(ierr);
  ierr          = VecRestoreArray(VHV[0],&vhv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------------ User-defined routines for TAO -------------------------- */

static PetscErrorCode FormFunctionGradient(Tao tao,Vec IC,PetscReal *f,Vec G,void *ctx)
{
  User              user_ptr = (User)ctx;
  TS                ts = user_ptr->ts;
  const PetscScalar *x_ptr;
  PetscScalar       *y_ptr;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecCopy(IC,user_ptr->U);CHKERRQ(ierr); /* set up the initial condition */

  ierr = TSSetTime(ts,0.0);CHKERRQ(ierr);
  ierr = TSSetStepNumber(ts,0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.001);CHKERRQ(ierr); /* can be overwritten by command line options */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSolve(ts,user_ptr->U);CHKERRQ(ierr);

  ierr     = VecGetArrayRead(user_ptr->U,&x_ptr);CHKERRQ(ierr);
  ierr     = VecGetArray(user_ptr->Lambda[0],&y_ptr);CHKERRQ(ierr);
  *f       = (x_ptr[0]-user_ptr->ob[0])*(x_ptr[0]-user_ptr->ob[0])+(x_ptr[1]-user_ptr->ob[1])*(x_ptr[1]-user_ptr->ob[1]);
  y_ptr[0] = 2.*(x_ptr[0]-user_ptr->ob[0]);
  y_ptr[1] = 2.*(x_ptr[1]-user_ptr->ob[1]);
  ierr     = VecRestoreArray(user_ptr->Lambda[0],&y_ptr);CHKERRQ(ierr);
  ierr     = VecRestoreArrayRead(user_ptr->U,&x_ptr);CHKERRQ(ierr);

  ierr = TSSetCostGradients(ts,1,user_ptr->Lambda,NULL);CHKERRQ(ierr);
  ierr = TSAdjointSolve(ts);CHKERRQ(ierr);
  ierr = VecCopy(user_ptr->Lambda[0],G);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormHessian(Tao tao,Vec U,Mat H,Mat Hpre,void *ctx)
{
  User           user_ptr = (User)ctx;
  PetscScalar    harr[2];
  PetscScalar    *x_ptr;
  const PetscInt rows[2] = {0,1};
  PetscInt       col;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr     = VecCopy(U,user_ptr->U);CHKERRQ(ierr);
  ierr     = VecGetArray(user_ptr->Dir,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 1.;
  x_ptr[1] = 0.;
  ierr     = VecRestoreArray(user_ptr->Dir,&x_ptr);CHKERRQ(ierr);
  ierr     = Adjoint2(user_ptr->U,harr,user_ptr);CHKERRQ(ierr);
  col      = 0;
  ierr     = MatSetValues(H,2,rows,1,&col,harr,INSERT_VALUES);CHKERRQ(ierr);

  ierr     = VecCopy(U,user_ptr->U);CHKERRQ(ierr);
  ierr     = VecGetArray(user_ptr->Dir,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 0.;
  x_ptr[1] = 1.;
  ierr     = VecRestoreArray(user_ptr->Dir,&x_ptr);CHKERRQ(ierr);
  ierr     = Adjoint2(user_ptr->U,harr,user_ptr);CHKERRQ(ierr);
  col      = 1;
  ierr     = MatSetValues(H,2,rows,1,&col,harr,INSERT_VALUES);CHKERRQ(ierr);

  ierr   = MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr   = MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (H != Hpre) {
    ierr = MatAssemblyBegin(Hpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Hpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatrixFreeHessian(Tao tao,Vec U,Mat H,Mat Hpre,void *ctx)
{
  User           user_ptr = (User)ctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = VecCopy(U,user_ptr->U);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ------------ Routines calculating second-order derivatives -------------- */

/**
  Compute the Hessian-vector product for the cost function using Second-order adjoint
*/
PetscErrorCode Adjoint2(Vec U,PetscScalar arr[],User ctx)
{
  TS             ts = ctx->ts;
  PetscScalar    *x_ptr,*y_ptr;
  Mat            tlmsen;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSAdjointReset(ts);CHKERRQ(ierr);

  ierr = TSSetTime(ts,0.0);CHKERRQ(ierr);
  ierr = TSSetStepNumber(ts,0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,0.001);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetCostHessianProducts(ts,1,ctx->Lambda2,NULL,ctx->Dir);CHKERRQ(ierr);
  ierr = TSAdjointSetForward(ts,NULL);CHKERRQ(ierr);
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  /* Set terminal conditions for first- and second-order adjonts */
  ierr = VecGetArray(U,&x_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->Lambda[0],&y_ptr);CHKERRQ(ierr);
  y_ptr[0] = 2.*(x_ptr[0]-ctx->ob[0]);
  y_ptr[1] = 2.*(x_ptr[1]-ctx->ob[1]);
  ierr = VecRestoreArray(ctx->Lambda[0],&y_ptr);CHKERRQ(ierr);
  ierr = VecRestoreArray(U,&x_ptr);CHKERRQ(ierr);
  ierr = TSForwardGetSensitivities(ts,NULL,&tlmsen);CHKERRQ(ierr);
  ierr = MatDenseGetColumn(tlmsen,0,&x_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->Lambda2[0],&y_ptr);CHKERRQ(ierr);
  y_ptr[0] = 2.*x_ptr[0];
  y_ptr[1] = 2.*x_ptr[1];
  ierr = VecRestoreArray(ctx->Lambda2[0],&y_ptr);CHKERRQ(ierr);
  ierr = MatDenseRestoreColumn(tlmsen,&x_ptr);CHKERRQ(ierr);

  ierr = TSSetCostGradients(ts,1,ctx->Lambda,NULL);CHKERRQ(ierr);
  if (ctx->implicitform) {
    ierr = TSSetIHessianProduct(ts,ctx->Ihp1,IHessianProductUU,NULL,NULL,NULL,NULL,NULL,NULL,ctx);CHKERRQ(ierr);
  } else {
    ierr = TSSetRHSHessianProduct(ts,ctx->Ihp1,RHSHessianProductUU,NULL,NULL,NULL,NULL,NULL,NULL,ctx);CHKERRQ(ierr);
  }
  ierr = TSAdjointSolve(ts);CHKERRQ(ierr);

  ierr   = VecGetArray(ctx->Lambda2[0],&x_ptr);CHKERRQ(ierr);
  arr[0] = x_ptr[0];
  arr[1] = x_ptr[1];
  ierr   = VecRestoreArray(ctx->Lambda2[0],&x_ptr);CHKERRQ(ierr);

  ierr = TSAdjointReset(ts);CHKERRQ(ierr);
  ierr = TSAdjointResetForward(ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FiniteDiff(Vec U,PetscScalar arr[],User ctx)
{
  Vec               Up,G,Gp;
  const PetscScalar eps = PetscRealConstant(1e-7);
  PetscScalar       *u;
  Tao               tao = NULL;
  PetscReal         f;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecDuplicate(U,&Up);CHKERRQ(ierr);
  ierr = VecDuplicate(U,&G);CHKERRQ(ierr);
  ierr = VecDuplicate(U,&Gp);CHKERRQ(ierr);

  ierr = FormFunctionGradient(tao,U,&f,G,ctx);CHKERRQ(ierr);

  ierr = VecCopy(U,Up);CHKERRQ(ierr);
  ierr = VecGetArray(Up,&u);CHKERRQ(ierr);
  u[0] += eps;
  ierr = VecRestoreArray(Up,&u);CHKERRQ(ierr);
  ierr = FormFunctionGradient(tao,Up,&f,Gp,ctx);CHKERRQ(ierr);
  ierr = VecAXPY(Gp,-1,G);CHKERRQ(ierr);
  ierr = VecScale(Gp,1./eps);CHKERRQ(ierr);
  ierr = VecGetArray(Gp,&u);CHKERRQ(ierr);
  arr[0] = u[0];
  arr[1] = u[1];
  ierr  = VecRestoreArray(Gp,&u);CHKERRQ(ierr);

  ierr = VecCopy(U,Up);CHKERRQ(ierr);
  ierr = VecGetArray(Up,&u);CHKERRQ(ierr);
  u[1] += eps;
  ierr = VecRestoreArray(Up,&u);CHKERRQ(ierr);
  ierr = FormFunctionGradient(tao,Up,&f,Gp,ctx);CHKERRQ(ierr);
  ierr = VecAXPY(Gp,-1,G);CHKERRQ(ierr);
  ierr = VecScale(Gp,1./eps);CHKERRQ(ierr);
  ierr = VecGetArray(Gp,&u);CHKERRQ(ierr);
  arr[2] = u[0];
  arr[3] = u[1];
  ierr  = VecRestoreArray(Gp,&u);CHKERRQ(ierr);

  ierr = VecDestroy(&G);CHKERRQ(ierr);
  ierr = VecDestroy(&Gp);CHKERRQ(ierr);
  ierr = VecDestroy(&Up);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode HessianProductMat(Mat mat,Vec svec,Vec y)
{
  User           user_ptr;
  PetscScalar    *y_ptr;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(mat,(void*)&user_ptr);CHKERRQ(ierr);
  ierr = VecCopy(svec,user_ptr->Dir);CHKERRQ(ierr);
  ierr = VecGetArray(y,&y_ptr);CHKERRQ(ierr);
  ierr = Adjoint2(user_ptr->U,y_ptr,user_ptr);CHKERRQ(ierr);
  ierr = VecRestoreArray(y,&y_ptr);CHKERRQ(ierr);
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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /* Set runtime options */
  user.next_output  = 0.0;
  user.mu           = 1.0e3;
  user.steps        = 0;
  user.ftime        = 0.5;
  user.implicitform = PETSC_TRUE;

  ierr = PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-mu",&user.mu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-mode",&mode,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-ic1",&ic1,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-ic2",&ic2,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-my_tao_mf",&mf,NULL);CHKERRQ(ierr); /* matrix-free hessian for optimization */
  ierr = PetscOptionsGetBool(NULL,NULL,"-implicitform",&user.implicitform,NULL);CHKERRQ(ierr);

  /* Create necessary matrix and vectors, solve same ODE on every process */
  ierr = MatCreate(PETSC_COMM_WORLD,&user.A);CHKERRQ(ierr);
  ierr = MatSetSizes(user.A,PETSC_DECIDE,PETSC_DECIDE,2,2);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user.A);CHKERRQ(ierr);
  ierr = MatSetUp(user.A);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.A,&user.U,NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.A,&user.Dir,NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.A,&user.Lambda[0],NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.A,&user.Lambda2[0],NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.A,&user.Ihp1[0],NULL);CHKERRQ(ierr);

  /* Create timestepping solver context */
  ierr = TSCreate(PETSC_COMM_WORLD,&user.ts);CHKERRQ(ierr);
  ierr = TSSetEquationType(user.ts,TS_EQ_ODE_EXPLICIT);CHKERRQ(ierr); /* less Jacobian evaluations when adjoint BEuler is used, otherwise no effect */
  if (user.implicitform) {
    ierr = TSSetIFunction(user.ts,NULL,IFunction,&user);CHKERRQ(ierr);
    ierr = TSSetIJacobian(user.ts,user.A,user.A,IJacobian,&user);CHKERRQ(ierr);
    ierr = TSSetType(user.ts,TSCN);CHKERRQ(ierr);
  } else {
    ierr = TSSetRHSFunction(user.ts,NULL,RHSFunction,&user);CHKERRQ(ierr);
    ierr = TSSetRHSJacobian(user.ts,user.A,user.A,RHSJacobian,&user);CHKERRQ(ierr);
    ierr = TSSetType(user.ts,TSRK);CHKERRQ(ierr);
  }
  ierr = TSSetMaxTime(user.ts,user.ftime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(user.ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

  if (monitor) {
    ierr = TSMonitorSet(user.ts,Monitor,&user,NULL);CHKERRQ(ierr);
  }

  /* Set ODE initial conditions */
  ierr     = VecGetArray(user.U,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 2.0;
  x_ptr[1] = -2.0/3.0 + 10.0/(81.0*user.mu) - 292.0/(2187.0*user.mu*user.mu);
  ierr     = VecRestoreArray(user.U,&x_ptr);CHKERRQ(ierr);

  /* Set runtime options */
  ierr = TSSetFromOptions(user.ts);CHKERRQ(ierr);

  /* Obtain the observation */
  ierr       = TSSolve(user.ts,user.U);CHKERRQ(ierr);
  ierr       = VecGetArray(user.U,&x_ptr);CHKERRQ(ierr);
  user.ob[0] = x_ptr[0];
  user.ob[1] = x_ptr[1];
  ierr       = VecRestoreArray(user.U,&x_ptr);CHKERRQ(ierr);

  ierr     = VecDuplicate(user.U,&x);CHKERRQ(ierr);
  ierr     = VecGetArray(x,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = ic1;
  x_ptr[1] = ic2;
  ierr     = VecRestoreArray(x,&x_ptr);CHKERRQ(ierr);

  /* Save trajectory of solution so that TSAdjointSolve() may be used */
  ierr = TSSetSaveTrajectory(user.ts);CHKERRQ(ierr);

  /* Compare finite difference and second-order adjoint. */
  switch (mode) {
    case 2 :
      ierr = FiniteDiff(x,arr,&user);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"\n Finite difference approximation of the Hessian\n");CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"%g %g\n%g %g\n",(double)arr[0],(double)arr[1],(double)arr[2],(double)arr[3]);CHKERRQ(ierr);
      break;
    case 1 : /* Compute the Hessian column by column */
      ierr     = VecCopy(x,user.U);CHKERRQ(ierr);
      ierr     = VecGetArray(user.Dir,&x_ptr);CHKERRQ(ierr);
      x_ptr[0] = 1.;
      x_ptr[1] = 0.;
      ierr     = VecRestoreArray(user.Dir,&x_ptr);CHKERRQ(ierr);
      ierr     = Adjoint2(user.U,arr,&user);CHKERRQ(ierr);
      ierr     = PetscPrintf(PETSC_COMM_WORLD,"\nFirst column of the Hessian\n");CHKERRQ(ierr);
      ierr     = PetscPrintf(PETSC_COMM_WORLD,"%g\n%g\n",(double)arr[0],(double)arr[1]);CHKERRQ(ierr);
      ierr     = VecCopy(x,user.U);CHKERRQ(ierr);
      ierr     = VecGetArray(user.Dir,&x_ptr);CHKERRQ(ierr);
      x_ptr[0] = 0.;
      x_ptr[1] = 1.;
      ierr     = VecRestoreArray(user.Dir,&x_ptr);CHKERRQ(ierr);
      ierr     = Adjoint2(user.U,arr,&user);CHKERRQ(ierr);
      ierr     = PetscPrintf(PETSC_COMM_WORLD,"\nSecond column of the Hessian\n");CHKERRQ(ierr);
      ierr     = PetscPrintf(PETSC_COMM_WORLD,"%g\n%g\n",(double)arr[0],(double)arr[1]);CHKERRQ(ierr);
      break;
    case 0 :
      /* Create optimization context and set up */
      ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
      ierr = TaoSetType(tao,TAOBLMVM);CHKERRQ(ierr);
      ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,(void *)&user);CHKERRQ(ierr);

      if (mf) {
        ierr = MatCreateShell(PETSC_COMM_SELF,2,2,2,2,(void*)&user,&user.H);CHKERRQ(ierr);
        ierr = MatShellSetOperation(user.H,MATOP_MULT,(void(*)(void))HessianProductMat);CHKERRQ(ierr);
        ierr = MatSetOption(user.H,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
        ierr = TaoSetHessianRoutine(tao,user.H,user.H,MatrixFreeHessian,(void *)&user);CHKERRQ(ierr);
      } else { /* Create Hessian matrix */
        ierr = MatCreate(PETSC_COMM_WORLD,&user.H);CHKERRQ(ierr);
        ierr = MatSetSizes(user.H,PETSC_DECIDE,PETSC_DECIDE,2,2);CHKERRQ(ierr);
        ierr = MatSetUp(user.H);CHKERRQ(ierr);
        ierr = TaoSetHessianRoutine(tao,user.H,user.H,FormHessian,(void *)&user);CHKERRQ(ierr);
      }

      /* Not use any preconditioner */
      ierr   = TaoGetKSP(tao,&ksp);CHKERRQ(ierr);
      if (ksp) {
        ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
        ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
      }

      /* Set initial solution guess */
      ierr = TaoSetInitialVector(tao,x);CHKERRQ(ierr);
      ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
      ierr = TaoSolve(tao);CHKERRQ(ierr);
      ierr = TaoDestroy(&tao);CHKERRQ(ierr);
      ierr = MatDestroy(&user.H);CHKERRQ(ierr);
      break;
    default:
      break;
  }

  /* Free work space.  All PETSc objects should be destroyed when they are no longer needed. */
  ierr = MatDestroy(&user.A);CHKERRQ(ierr);
  ierr = VecDestroy(&user.U);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Lambda[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Lambda2[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Ihp1[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Dir);CHKERRQ(ierr);
  ierr = TSDestroy(&user.ts);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
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
