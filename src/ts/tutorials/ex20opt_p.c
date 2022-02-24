
static char help[] = "Solves the van der Pol equation.\n\
Input parameters include:\n";

/*
   Concepts: TS^time-dependent nonlinear problems
   Concepts: TS^van der Pol equation DAE equivalent
   Concepts: TS^Optimization using adjoint sensitivity analysis
   Processors: 1
*/
/* ------------------------------------------------------------------------

  Notes:
  This code demonstrates how to solve a DAE-constrained optimization problem with TAO, TSAdjoint and TS.
  The nonlinear problem is written in a DAE equivalent form.
  The objective is to minimize the difference between observation and model prediction by finding an optimal value for parameter \mu.
  The gradient is computed with the discrete adjoint of an implicit theta method, see ex20adj.c for details.
  ------------------------------------------------------------------------- */
#include <petsctao.h>
#include <petscts.h>

typedef struct _n_User *User;
struct _n_User {
  TS        ts;
  PetscReal mu;
  PetscReal next_output;

  /* Sensitivity analysis support */
  PetscReal ftime;
  Mat       A;                       /* Jacobian matrix */
  Mat       Jacp;                    /* JacobianP matrix */
  Mat       H;                       /* Hessian matrix for optimization */
  Vec       U,Lambda[1],Mup[1];      /* adjoint variables */
  Vec       Lambda2[1],Mup2[1];      /* second-order adjoint variables */
  Vec       Ihp1[1];                 /* working space for Hessian evaluations */
  Vec       Ihp2[1];                 /* working space for Hessian evaluations */
  Vec       Ihp3[1];                 /* working space for Hessian evaluations */
  Vec       Ihp4[1];                 /* working space for Hessian evaluations */
  Vec       Dir;                     /* direction vector */
  PetscReal ob[2];                   /* observation used by the cost function */
  PetscBool implicitform;            /* implicit ODE? */
};

PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal*,Vec,void*);
PetscErrorCode FormHessian(Tao,Vec,Mat,Mat,void*);
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
  if (B && A != B) {
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
  for (j=0; j<2; j++) {
    vhv[j] = 0;
    for (k=0; k<2; k++)
      for (i=0; i<2; i++)
        vhv[j] += vl[i]*dJdU[i][j][k]*vr[k];
  }

  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(Vl[0],&vl));
  CHKERRQ(VecRestoreArrayRead(Vr,&vr));
  CHKERRQ(VecRestoreArray(VHV[0],&vhv));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSHessianProductUP(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV,void *ctx)
{
  const PetscScalar *vl,*vr,*u;
  PetscScalar       *vhv;
  PetscScalar       dJdP[2][2][1]={{{0}}};
  PetscInt          i,j,k;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(Vl[0],&vl));
  CHKERRQ(VecGetArrayRead(Vr,&vr));
  CHKERRQ(VecGetArray(VHV[0],&vhv));

  dJdP[1][0][0] = -(1.+2.*u[0]*u[1]);
  dJdP[1][1][0] = 1.-u[0]*u[0];
  for (j=0; j<2; j++) {
    vhv[j] = 0;
    for (k=0; k<1; k++)
      for (i=0; i<2; i++)
        vhv[j] += vl[i]*dJdP[i][j][k]*vr[k];
  }

  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(Vl[0],&vl));
  CHKERRQ(VecRestoreArrayRead(Vr,&vr));
  CHKERRQ(VecRestoreArray(VHV[0],&vhv));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSHessianProductPU(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV,void *ctx)
{
  const PetscScalar *vl,*vr,*u;
  PetscScalar       *vhv;
  PetscScalar       dJdU[2][1][2]={{{0}}};
  PetscInt          i,j,k;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(Vl[0],&vl));
  CHKERRQ(VecGetArrayRead(Vr,&vr));
  CHKERRQ(VecGetArray(VHV[0],&vhv));

  dJdU[1][0][0] = -1.-2.*u[1]*u[0];
  dJdU[1][0][1] = 1.-u[0]*u[0];
  for (j=0; j<1; j++) {
    vhv[j] = 0;
    for (k=0; k<2; k++)
      for (i=0; i<2; i++)
        vhv[j] += vl[i]*dJdU[i][j][k]*vr[k];
  }

  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(Vl[0],&vl));
  CHKERRQ(VecRestoreArrayRead(Vr,&vr));
  CHKERRQ(VecRestoreArray(VHV[0],&vhv));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSHessianProductPP(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV,void *ctx)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(0);
}

/* ----------------------- Implicit form of the ODE  -------------------- */

static PetscErrorCode IFunction(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,void *ctx)
{
  User              user = (User)ctx;
  PetscScalar       *f;
  const PetscScalar *u,*udot;

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
  User              user     = (User)ctx;
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

  CHKERRQ(VecRestoreArrayRead(U,&u));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSJacobianP(TS ts,PetscReal t,Vec U,Mat A,void *ctx)
{
  PetscInt          row[] = {0,1},col[]={0};
  PetscScalar       J[2][1];
  const PetscScalar *u;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(U,&u));

  J[0][0] = 0;
  J[1][0] = (1.-u[0]*u[0])*u[1]-u[0];
  CHKERRQ(MatSetValues(A,2,row,1,col,&J[0][0],INSERT_VALUES));
  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  CHKERRQ(VecRestoreArrayRead(U,&u));
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
  for (j=0; j<2; j++) {
    vhv[j] = 0;
    for (k=0; k<2; k++)
      for (i=0; i<2; i++)
        vhv[j] += vl[i]*dJdU[i][j][k]*vr[k];
  }

  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(Vl[0],&vl));
  CHKERRQ(VecRestoreArrayRead(Vr,&vr));
  CHKERRQ(VecRestoreArray(VHV[0],&vhv));
  PetscFunctionReturn(0);
}

static PetscErrorCode IHessianProductUP(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV,void *ctx)
{
  const PetscScalar *vl,*vr,*u;
  PetscScalar       *vhv;
  PetscScalar       dJdP[2][2][1]={{{0}}};
  PetscInt          i,j,k;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(Vl[0],&vl));
  CHKERRQ(VecGetArrayRead(Vr,&vr));
  CHKERRQ(VecGetArray(VHV[0],&vhv));

  dJdP[1][0][0] = 1.+2.*u[0]*u[1];
  dJdP[1][1][0] = u[0]*u[0]-1.;
  for (j=0; j<2; j++) {
    vhv[j] = 0;
    for (k=0; k<1; k++)
      for (i=0; i<2; i++)
        vhv[j] += vl[i]*dJdP[i][j][k]*vr[k];
  }

  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(Vl[0],&vl));
  CHKERRQ(VecRestoreArrayRead(Vr,&vr));
  CHKERRQ(VecRestoreArray(VHV[0],&vhv));
  PetscFunctionReturn(0);
}

static PetscErrorCode IHessianProductPU(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV,void *ctx)
{
  const PetscScalar *vl,*vr,*u;
  PetscScalar       *vhv;
  PetscScalar       dJdU[2][1][2]={{{0}}};
  PetscInt          i,j,k;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(Vl[0],&vl));
  CHKERRQ(VecGetArrayRead(Vr,&vr));
  CHKERRQ(VecGetArray(VHV[0],&vhv));

  dJdU[1][0][0] = 1.+2.*u[1]*u[0];
  dJdU[1][0][1] = u[0]*u[0]-1.;
  for (j=0; j<1; j++) {
    vhv[j] = 0;
    for (k=0; k<2; k++)
      for (i=0; i<2; i++)
        vhv[j] += vl[i]*dJdU[i][j][k]*vr[k];
  }

  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(Vl[0],&vl));
  CHKERRQ(VecRestoreArrayRead(Vr,&vr));
  CHKERRQ(VecRestoreArray(VHV[0],&vhv));
  PetscFunctionReturn(0);
}

static PetscErrorCode IHessianProductPP(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV,void *ctx)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(0);
}

/* Monitor timesteps and use interpolation to output at integer multiples of 0.1 */
static PetscErrorCode Monitor(TS ts,PetscInt step,PetscReal t,Vec X,void *ctx)
{
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscReal         tfinal, dt;
  User              user = (User)ctx;
  Vec               interpolatedX;

  PetscFunctionBeginUser;
  CHKERRQ(TSGetTimeStep(ts,&dt));
  CHKERRQ(TSGetMaxTime(ts,&tfinal));

  while (user->next_output <= t && user->next_output <= tfinal) {
    CHKERRQ(VecDuplicate(X,&interpolatedX));
    CHKERRQ(TSInterpolate(ts,user->next_output,interpolatedX));
    CHKERRQ(VecGetArrayRead(interpolatedX,&x));
    ierr = PetscPrintf(PETSC_COMM_WORLD,"[%g] %D TS %g (dt = %g) X %g %g\n",
                       (double)user->next_output,step,(double)t,(double)dt,(double)PetscRealPart(x[0]),
                       (double)PetscRealPart(x[1]));CHKERRQ(ierr);
    CHKERRQ(VecRestoreArrayRead(interpolatedX,&x));
    CHKERRQ(VecDestroy(&interpolatedX));
    user->next_output += PetscRealConstant(0.1);
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Vec                P;
  PetscBool          monitor = PETSC_FALSE;
  PetscScalar        *x_ptr;
  const PetscScalar  *y_ptr;
  PetscMPIInt        size;
  struct _n_User     user;
  PetscErrorCode     ierr;
  Tao                tao;
  KSP                ksp;
  PC                 pc;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,NULL,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /* Create TAO solver and set desired solution method */
  CHKERRQ(TaoCreate(PETSC_COMM_WORLD,&tao));
  CHKERRQ(TaoSetType(tao,TAOBQNLS));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.next_output  = 0.0;
  user.mu           = PetscRealConstant(1.0e3);
  user.ftime        = PetscRealConstant(0.5);
  user.implicitform = PETSC_TRUE;

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL));
  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-mu",&user.mu,NULL));
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-implicitform",&user.implicitform,NULL));

  /* Create necessary matrix and vectors, solve same ODE on every process */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user.A));
  CHKERRQ(MatSetSizes(user.A,PETSC_DECIDE,PETSC_DECIDE,2,2));
  CHKERRQ(MatSetFromOptions(user.A));
  CHKERRQ(MatSetUp(user.A));
  CHKERRQ(MatCreateVecs(user.A,&user.U,NULL));
  CHKERRQ(MatCreateVecs(user.A,&user.Lambda[0],NULL));
  CHKERRQ(MatCreateVecs(user.A,&user.Lambda2[0],NULL));
  CHKERRQ(MatCreateVecs(user.A,&user.Ihp1[0],NULL));
  CHKERRQ(MatCreateVecs(user.A,&user.Ihp2[0],NULL));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user.Jacp));
  CHKERRQ(MatSetSizes(user.Jacp,PETSC_DECIDE,PETSC_DECIDE,2,1));
  CHKERRQ(MatSetFromOptions(user.Jacp));
  CHKERRQ(MatSetUp(user.Jacp));
  CHKERRQ(MatCreateVecs(user.Jacp,&user.Dir,NULL));
  CHKERRQ(MatCreateVecs(user.Jacp,&user.Mup[0],NULL));
  CHKERRQ(MatCreateVecs(user.Jacp,&user.Mup2[0],NULL));
  CHKERRQ(MatCreateVecs(user.Jacp,&user.Ihp3[0],NULL));
  CHKERRQ(MatCreateVecs(user.Jacp,&user.Ihp4[0],NULL));

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
  CHKERRQ(TSSetRHSJacobianP(user.ts,user.Jacp,RHSJacobianP,&user));
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
  CHKERRQ(TSSetTimeStep(user.ts,PetscRealConstant(0.001)));

  /* Set runtime options */
  CHKERRQ(TSSetFromOptions(user.ts));

  CHKERRQ(TSSolve(user.ts,user.U));
  CHKERRQ(VecGetArrayRead(user.U,&y_ptr));
  user.ob[0] = y_ptr[0];
  user.ob[1] = y_ptr[1];
  CHKERRQ(VecRestoreArrayRead(user.U,&y_ptr));

  /* Save trajectory of solution so that TSAdjointSolve() may be used.
     Skip checkpointing for the first TSSolve since no adjoint run follows it.
   */
  CHKERRQ(TSSetSaveTrajectory(user.ts));

  /* Optimization starts */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&user.H));
  CHKERRQ(MatSetSizes(user.H,PETSC_DECIDE,PETSC_DECIDE,1,1));
  CHKERRQ(MatSetUp(user.H)); /* Hessian should be symmetric. Do we need to do MatSetOption(user.H,MAT_SYMMETRIC,PETSC_TRUE) ? */

  /* Set initial solution guess */
  CHKERRQ(MatCreateVecs(user.Jacp,&P,NULL));
  CHKERRQ(VecGetArray(P,&x_ptr));
  x_ptr[0] = PetscRealConstant(1.2);
  CHKERRQ(VecRestoreArray(P,&x_ptr));
  CHKERRQ(TaoSetSolution(tao,P));

  /* Set routine for function and gradient evaluation */
  CHKERRQ(TaoSetObjectiveAndGradient(tao,NULL,FormFunctionGradient,(void *)&user));
  CHKERRQ(TaoSetHessian(tao,user.H,user.H,FormHessian,(void *)&user));

  /* Check for any TAO command line options */
  CHKERRQ(TaoGetKSP(tao,&ksp));
  if (ksp) {
    CHKERRQ(KSPGetPC(ksp,&pc));
    CHKERRQ(PCSetType(pc,PCNONE));
  }
  CHKERRQ(TaoSetFromOptions(tao));

  CHKERRQ(TaoSolve(tao));

  CHKERRQ(VecView(P,PETSC_VIEWER_STDOUT_WORLD));
  /* Free TAO data structures */
  CHKERRQ(TaoDestroy(&tao));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatDestroy(&user.H));
  CHKERRQ(MatDestroy(&user.A));
  CHKERRQ(VecDestroy(&user.U));
  CHKERRQ(MatDestroy(&user.Jacp));
  CHKERRQ(VecDestroy(&user.Lambda[0]));
  CHKERRQ(VecDestroy(&user.Mup[0]));
  CHKERRQ(VecDestroy(&user.Lambda2[0]));
  CHKERRQ(VecDestroy(&user.Mup2[0]));
  CHKERRQ(VecDestroy(&user.Ihp1[0]));
  CHKERRQ(VecDestroy(&user.Ihp2[0]));
  CHKERRQ(VecDestroy(&user.Ihp3[0]));
  CHKERRQ(VecDestroy(&user.Ihp4[0]));
  CHKERRQ(VecDestroy(&user.Dir));
  CHKERRQ(TSDestroy(&user.ts));
  CHKERRQ(VecDestroy(&P));
  ierr = PetscFinalize();
  return ierr;
}

/* ------------------------------------------------------------------ */
/*
   FormFunctionGradient - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   X   - the input vector
   ptr - optional user-defined context, as set by TaoSetObjectiveAndGradient()

   Output Parameters:
   f   - the newly evaluated function
   G   - the newly evaluated gradient
*/
PetscErrorCode FormFunctionGradient(Tao tao,Vec P,PetscReal *f,Vec G,void *ctx)
{
  User              user_ptr = (User)ctx;
  TS                ts = user_ptr->ts;
  PetscScalar       *x_ptr,*g;
  const PetscScalar *y_ptr;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(P,&y_ptr));
  user_ptr->mu = y_ptr[0];
  CHKERRQ(VecRestoreArrayRead(P,&y_ptr));

  CHKERRQ(TSSetTime(ts,0.0));
  CHKERRQ(TSSetStepNumber(ts,0));
  CHKERRQ(TSSetTimeStep(ts,PetscRealConstant(0.001))); /* can be overwritten by command line options */
  CHKERRQ(TSSetFromOptions(ts));
  CHKERRQ(VecGetArray(user_ptr->U,&x_ptr));
  x_ptr[0] = 2.0;
  x_ptr[1] = -2.0/3.0 + 10.0/(81.0*user_ptr->mu) - 292.0/(2187.0*user_ptr->mu*user_ptr->mu);
  CHKERRQ(VecRestoreArray(user_ptr->U,&x_ptr));

  CHKERRQ(TSSolve(ts,user_ptr->U));

  CHKERRQ(VecGetArrayRead(user_ptr->U,&y_ptr));
  *f   = (y_ptr[0]-user_ptr->ob[0])*(y_ptr[0]-user_ptr->ob[0])+(y_ptr[1]-user_ptr->ob[1])*(y_ptr[1]-user_ptr->ob[1]);

  /*   Reset initial conditions for the adjoint integration */
  CHKERRQ(VecGetArray(user_ptr->Lambda[0],&x_ptr));
  x_ptr[0] = 2.*(y_ptr[0]-user_ptr->ob[0]);
  x_ptr[1] = 2.*(y_ptr[1]-user_ptr->ob[1]);
  CHKERRQ(VecRestoreArrayRead(user_ptr->U,&y_ptr));
  CHKERRQ(VecRestoreArray(user_ptr->Lambda[0],&x_ptr));

  CHKERRQ(VecGetArray(user_ptr->Mup[0],&x_ptr));
  x_ptr[0] = 0.0;
  CHKERRQ(VecRestoreArray(user_ptr->Mup[0],&x_ptr));
  CHKERRQ(TSSetCostGradients(ts,1,user_ptr->Lambda,user_ptr->Mup));

  CHKERRQ(TSAdjointSolve(ts));

  CHKERRQ(VecGetArray(user_ptr->Mup[0],&x_ptr));
  CHKERRQ(VecGetArrayRead(user_ptr->Lambda[0],&y_ptr));
  CHKERRQ(VecGetArray(G,&g));
  g[0] = y_ptr[1]*(-10.0/(81.0*user_ptr->mu*user_ptr->mu)+2.0*292.0/(2187.0*user_ptr->mu*user_ptr->mu*user_ptr->mu))+x_ptr[0];
  CHKERRQ(VecRestoreArray(user_ptr->Mup[0],&x_ptr));
  CHKERRQ(VecRestoreArrayRead(user_ptr->Lambda[0],&y_ptr));
  CHKERRQ(VecRestoreArray(G,&g));
  PetscFunctionReturn(0);
}

PetscErrorCode FormHessian(Tao tao,Vec P,Mat H,Mat Hpre,void *ctx)
{
  User           user_ptr = (User)ctx;
  PetscScalar    harr[1];
  const PetscInt rows[1] = {0};
  PetscInt       col = 0;

  PetscFunctionBeginUser;
  CHKERRQ(Adjoint2(P,harr,user_ptr));
  CHKERRQ(MatSetValues(H,1,rows,1,&col,harr,INSERT_VALUES));

  CHKERRQ(MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY));
  if (H != Hpre) {
    CHKERRQ(MatAssemblyBegin(Hpre,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(Hpre,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode Adjoint2(Vec P,PetscScalar arr[],User ctx)
{
  TS                ts = ctx->ts;
  const PetscScalar *z_ptr;
  PetscScalar       *x_ptr,*y_ptr,dzdp,dzdp2;
  Mat               tlmsen;

  PetscFunctionBeginUser;
  /* Reset TSAdjoint so that AdjointSetUp will be called again */
  CHKERRQ(TSAdjointReset(ts));

  /* The directional vector should be 1 since it is one-dimensional */
  CHKERRQ(VecGetArray(ctx->Dir,&x_ptr));
  x_ptr[0] = 1.;
  CHKERRQ(VecRestoreArray(ctx->Dir,&x_ptr));

  CHKERRQ(VecGetArrayRead(P,&z_ptr));
  ctx->mu = z_ptr[0];
  CHKERRQ(VecRestoreArrayRead(P,&z_ptr));

  dzdp  = -10.0/(81.0*ctx->mu*ctx->mu) + 2.0*292.0/(2187.0*ctx->mu*ctx->mu*ctx->mu);
  dzdp2 = 2.*10.0/(81.0*ctx->mu*ctx->mu*ctx->mu) - 3.0*2.0*292.0/(2187.0*ctx->mu*ctx->mu*ctx->mu*ctx->mu);

  CHKERRQ(TSSetTime(ts,0.0));
  CHKERRQ(TSSetStepNumber(ts,0));
  CHKERRQ(TSSetTimeStep(ts,PetscRealConstant(0.001))); /* can be overwritten by command line options */
  CHKERRQ(TSSetFromOptions(ts));
  CHKERRQ(TSSetCostHessianProducts(ts,1,ctx->Lambda2,ctx->Mup2,ctx->Dir));

  CHKERRQ(MatZeroEntries(ctx->Jacp));
  CHKERRQ(MatSetValue(ctx->Jacp,1,0,dzdp,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(ctx->Jacp,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(ctx->Jacp,MAT_FINAL_ASSEMBLY));

  CHKERRQ(TSAdjointSetForward(ts,ctx->Jacp));
  CHKERRQ(VecGetArray(ctx->U,&y_ptr));
  y_ptr[0] = 2.0;
  y_ptr[1] = -2.0/3.0 + 10.0/(81.0*ctx->mu) - 292.0/(2187.0*ctx->mu*ctx->mu);
  CHKERRQ(VecRestoreArray(ctx->U,&y_ptr));
  CHKERRQ(TSSolve(ts,ctx->U));

  /* Set terminal conditions for first- and second-order adjonts */
  CHKERRQ(VecGetArrayRead(ctx->U,&z_ptr));
  CHKERRQ(VecGetArray(ctx->Lambda[0],&y_ptr));
  y_ptr[0] = 2.*(z_ptr[0]-ctx->ob[0]);
  y_ptr[1] = 2.*(z_ptr[1]-ctx->ob[1]);
  CHKERRQ(VecRestoreArray(ctx->Lambda[0],&y_ptr));
  CHKERRQ(VecRestoreArrayRead(ctx->U,&z_ptr));
  CHKERRQ(VecGetArray(ctx->Mup[0],&y_ptr));
  y_ptr[0] = 0.0;
  CHKERRQ(VecRestoreArray(ctx->Mup[0],&y_ptr));
  CHKERRQ(TSForwardGetSensitivities(ts,NULL,&tlmsen));
  CHKERRQ(MatDenseGetColumn(tlmsen,0,&x_ptr));
  CHKERRQ(VecGetArray(ctx->Lambda2[0],&y_ptr));
  y_ptr[0] = 2.*x_ptr[0];
  y_ptr[1] = 2.*x_ptr[1];
  CHKERRQ(VecRestoreArray(ctx->Lambda2[0],&y_ptr));
  CHKERRQ(VecGetArray(ctx->Mup2[0],&y_ptr));
  y_ptr[0] = 0.0;
  CHKERRQ(VecRestoreArray(ctx->Mup2[0],&y_ptr));
  CHKERRQ(MatDenseRestoreColumn(tlmsen,&x_ptr));
  CHKERRQ(TSSetCostGradients(ts,1,ctx->Lambda,ctx->Mup));
  if (ctx->implicitform) {
    CHKERRQ(TSSetIHessianProduct(ts,ctx->Ihp1,IHessianProductUU,ctx->Ihp2,IHessianProductUP,ctx->Ihp3,IHessianProductPU,ctx->Ihp4,IHessianProductPP,ctx));
  } else {
    CHKERRQ(TSSetRHSHessianProduct(ts,ctx->Ihp1,RHSHessianProductUU,ctx->Ihp2,RHSHessianProductUP,ctx->Ihp3,RHSHessianProductPU,ctx->Ihp4,RHSHessianProductPP,ctx));
  }
  CHKERRQ(TSAdjointSolve(ts));

  CHKERRQ(VecGetArray(ctx->Lambda[0],&x_ptr));
  CHKERRQ(VecGetArray(ctx->Lambda2[0],&y_ptr));
  CHKERRQ(VecGetArrayRead(ctx->Mup2[0],&z_ptr));

  arr[0] = x_ptr[1]*dzdp2 + y_ptr[1]*dzdp2 + z_ptr[0];

  CHKERRQ(VecRestoreArray(ctx->Lambda2[0],&x_ptr));
  CHKERRQ(VecRestoreArray(ctx->Lambda2[0],&y_ptr));
  CHKERRQ(VecRestoreArrayRead(ctx->Mup2[0],&z_ptr));

  /* Disable second-order adjoint mode */
  CHKERRQ(TSAdjointReset(ts));
  CHKERRQ(TSAdjointResetForward(ts));
  PetscFunctionReturn(0);
}

/*TEST
    build:
      requires: !complex !single
    test:
      args:  -implicitform 0 -ts_type rk -ts_adapt_type none -mu 10 -ts_dt 0.1 -viewer_binary_skip_info -tao_monitor -tao_view
      output_file: output/ex20opt_p_1.out

    test:
      suffix: 2
      args:  -implicitform 0 -ts_type rk -ts_adapt_type none -mu 10 -ts_dt 0.01 -viewer_binary_skip_info -tao_monitor -tao_type bntr -tao_bnk_pc_type none
      output_file: output/ex20opt_p_2.out

    test:
      suffix: 3
      args:  -ts_type cn -ts_adapt_type none -mu 100 -ts_dt 0.01 -viewer_binary_skip_info -tao_monitor -tao_view
      output_file: output/ex20opt_p_3.out

    test:
      suffix: 4
      args:  -ts_type cn -ts_adapt_type none -mu 100 -ts_dt 0.01 -viewer_binary_skip_info -tao_monitor -tao_type bntr -tao_bnk_pc_type none
      output_file: output/ex20opt_p_4.out

TEST*/
