
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
  if (B && A != B) {
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
  for (j=0; j<2; j++) {
    vhv[j] = 0;
    for (k=0; k<2; k++)
      for (i=0; i<2; i++)
        vhv[j] += vl[i]*dJdU[i][j][k]*vr[k];
  }

  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Vl[0],&vl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Vr,&vr);CHKERRQ(ierr);
  ierr = VecRestoreArray(VHV[0],&vhv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSHessianProductUP(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV,void *ctx)
{
  const PetscScalar *vl,*vr,*u;
  PetscScalar       *vhv;
  PetscScalar       dJdP[2][2][1]={{{0}}};
  PetscInt          i,j,k;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Vl[0],&vl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Vr,&vr);CHKERRQ(ierr);
  ierr = VecGetArray(VHV[0],&vhv);CHKERRQ(ierr);

  dJdP[1][0][0] = -(1.+2.*u[0]*u[1]);
  dJdP[1][1][0] = 1.-u[0]*u[0];
  for (j=0; j<2; j++) {
    vhv[j] = 0;
    for (k=0; k<1; k++)
      for (i=0; i<2; i++)
        vhv[j] += vl[i]*dJdP[i][j][k]*vr[k];
  }

  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Vl[0],&vl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Vr,&vr);CHKERRQ(ierr);
  ierr = VecRestoreArray(VHV[0],&vhv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSHessianProductPU(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV,void *ctx)
{
  const PetscScalar *vl,*vr,*u;
  PetscScalar       *vhv;
  PetscScalar       dJdU[2][1][2]={{{0}}};
  PetscInt          i,j,k;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Vl[0],&vl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Vr,&vr);CHKERRQ(ierr);
  ierr = VecGetArray(VHV[0],&vhv);CHKERRQ(ierr);

  dJdU[1][0][0] = -1.-2.*u[1]*u[0];
  dJdU[1][0][1] = 1.-u[0]*u[0];
  for (j=0; j<1; j++) {
    vhv[j] = 0;
    for (k=0; k<2; k++)
      for (i=0; i<2; i++)
        vhv[j] += vl[i]*dJdU[i][j][k]*vr[k];
  }

  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Vl[0],&vl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Vr,&vr);CHKERRQ(ierr);
  ierr = VecRestoreArray(VHV[0],&vhv);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
  User              user = (User)ctx;
  PetscScalar       *f;
  const PetscScalar *u,*udot;

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
  User              user     = (User)ctx;
  PetscInt          rowcol[] = {0,1};
  PetscScalar       J[2][2];
  const PetscScalar *u;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);

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

  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSJacobianP(TS ts,PetscReal t,Vec U,Mat A,void *ctx)
{
  PetscErrorCode    ierr;
  PetscInt          row[] = {0,1},col[]={0};
  PetscScalar       J[2][1];
  const PetscScalar *u;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);

  J[0][0] = 0;
  J[1][0] = (1.-u[0]*u[0])*u[1]-u[0];
  ierr    = MatSetValues(A,2,row,1,col,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr    = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr    = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr    = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
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
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Vl[0],&vl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Vr,&vr);CHKERRQ(ierr);
  ierr = VecGetArray(VHV[0],&vhv);CHKERRQ(ierr);

  dJdU[1][0][0] = 2.*user->mu*u[1];
  dJdU[1][1][0] = 2.*user->mu*u[0];
  dJdU[1][0][1] = 2.*user->mu*u[0];
  for (j=0; j<2; j++) {
    vhv[j] = 0;
    for (k=0; k<2; k++)
      for (i=0; i<2; i++)
        vhv[j] += vl[i]*dJdU[i][j][k]*vr[k];
  }

  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Vl[0],&vl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Vr,&vr);CHKERRQ(ierr);
  ierr = VecRestoreArray(VHV[0],&vhv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IHessianProductUP(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV,void *ctx)
{
  const PetscScalar *vl,*vr,*u;
  PetscScalar       *vhv;
  PetscScalar       dJdP[2][2][1]={{{0}}};
  PetscInt          i,j,k;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Vl[0],&vl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Vr,&vr);CHKERRQ(ierr);
  ierr = VecGetArray(VHV[0],&vhv);CHKERRQ(ierr);

  dJdP[1][0][0] = 1.+2.*u[0]*u[1];
  dJdP[1][1][0] = u[0]*u[0]-1.;
  for (j=0; j<2; j++) {
    vhv[j] = 0;
    for (k=0; k<1; k++)
      for (i=0; i<2; i++)
        vhv[j] += vl[i]*dJdP[i][j][k]*vr[k];
  }

  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Vl[0],&vl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Vr,&vr);CHKERRQ(ierr);
  ierr = VecRestoreArray(VHV[0],&vhv);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode IHessianProductPU(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV,void *ctx)
{
  const PetscScalar *vl,*vr,*u;
  PetscScalar       *vhv;
  PetscScalar       dJdU[2][1][2]={{{0}}};
  PetscInt          i,j,k;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Vl[0],&vl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Vr,&vr);CHKERRQ(ierr);
  ierr = VecGetArray(VHV[0],&vhv);CHKERRQ(ierr);

  dJdU[1][0][0] = 1.+2.*u[1]*u[0];
  dJdU[1][0][1] = u[0]*u[0]-1.;
  for (j=0; j<1; j++) {
    vhv[j] = 0;
    for (k=0; k<2; k++)
      for (i=0; i<2; i++)
        vhv[j] += vl[i]*dJdU[i][j][k]*vr[k];
  }

  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Vl[0],&vl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Vr,&vr);CHKERRQ(ierr);
  ierr = VecRestoreArray(VHV[0],&vhv);CHKERRQ(ierr);
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
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  ierr = TSGetMaxTime(ts,&tfinal);CHKERRQ(ierr);

  while (user->next_output <= t && user->next_output <= tfinal) {
    ierr = VecDuplicate(X,&interpolatedX);CHKERRQ(ierr);
    ierr = TSInterpolate(ts,user->next_output,interpolatedX);CHKERRQ(ierr);
    ierr = VecGetArrayRead(interpolatedX,&x);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"[%g] %D TS %g (dt = %g) X %g %g\n",
                       (double)user->next_output,step,(double)t,(double)dt,(double)PetscRealPart(x[0]),
                       (double)PetscRealPart(x[1]));CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(interpolatedX,&x);CHKERRQ(ierr);
    ierr = VecDestroy(&interpolatedX);CHKERRQ(ierr);
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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /* Create TAO solver and set desired solution method */
  ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOBQNLS);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.next_output  = 0.0;
  user.mu           = PetscRealConstant(1.0e3);
  user.ftime        = PetscRealConstant(0.5);
  user.implicitform = PETSC_TRUE;

  ierr = PetscOptionsGetBool(NULL,NULL,"-monitor",&monitor,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-mu",&user.mu,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-implicitform",&user.implicitform,NULL);CHKERRQ(ierr);

  /* Create necessary matrix and vectors, solve same ODE on every process */
  ierr = MatCreate(PETSC_COMM_WORLD,&user.A);CHKERRQ(ierr);
  ierr = MatSetSizes(user.A,PETSC_DECIDE,PETSC_DECIDE,2,2);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user.A);CHKERRQ(ierr);
  ierr = MatSetUp(user.A);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.A,&user.U,NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.A,&user.Lambda[0],NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.A,&user.Lambda2[0],NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.A,&user.Ihp1[0],NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.A,&user.Ihp2[0],NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&user.Jacp);CHKERRQ(ierr);
  ierr = MatSetSizes(user.Jacp,PETSC_DECIDE,PETSC_DECIDE,2,1);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user.Jacp);CHKERRQ(ierr);
  ierr = MatSetUp(user.Jacp);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.Jacp,&user.Dir,NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.Jacp,&user.Mup[0],NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.Jacp,&user.Mup2[0],NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.Jacp,&user.Ihp3[0],NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(user.Jacp,&user.Ihp4[0],NULL);CHKERRQ(ierr);

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
  ierr = TSSetRHSJacobianP(user.ts,user.Jacp,RHSJacobianP,&user);CHKERRQ(ierr);
  ierr = TSSetMaxTime(user.ts,user.ftime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(user.ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

  if (monitor) {
    ierr = TSMonitorSet(user.ts,Monitor,&user,NULL);CHKERRQ(ierr);
  }

  /* Set ODE initial conditions */
  ierr = VecGetArray(user.U,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 2.0;
  x_ptr[1] = -2.0/3.0 + 10.0/(81.0*user.mu) - 292.0/(2187.0*user.mu*user.mu);
  ierr = VecRestoreArray(user.U,&x_ptr);CHKERRQ(ierr);
  ierr = TSSetTimeStep(user.ts,PetscRealConstant(0.001));CHKERRQ(ierr);

  /* Set runtime options */
  ierr = TSSetFromOptions(user.ts);CHKERRQ(ierr);

  ierr       = TSSolve(user.ts,user.U);CHKERRQ(ierr);
  ierr       = VecGetArrayRead(user.U,&y_ptr);CHKERRQ(ierr);
  user.ob[0] = y_ptr[0];
  user.ob[1] = y_ptr[1];
  ierr       = VecRestoreArrayRead(user.U,&y_ptr);CHKERRQ(ierr);

  /* Save trajectory of solution so that TSAdjointSolve() may be used.
     Skip checkpointing for the first TSSolve since no adjoint run follows it.
   */
  ierr = TSSetSaveTrajectory(user.ts);CHKERRQ(ierr);

  /* Optimization starts */
  ierr = MatCreate(PETSC_COMM_WORLD,&user.H);CHKERRQ(ierr);
  ierr = MatSetSizes(user.H,PETSC_DECIDE,PETSC_DECIDE,1,1);CHKERRQ(ierr);
  ierr = MatSetUp(user.H);CHKERRQ(ierr); /* Hessian should be symmetric. Do we need to do MatSetOption(user.H,MAT_SYMMETRIC,PETSC_TRUE) ? */

  /* Set initial solution guess */
  ierr = MatCreateVecs(user.Jacp,&P,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(P,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = PetscRealConstant(1.2);
  ierr = VecRestoreArray(P,&x_ptr);CHKERRQ(ierr);
  ierr = TaoSetSolution(tao,P);CHKERRQ(ierr);

  /* Set routine for function and gradient evaluation */
  ierr = TaoSetObjectiveAndGradient(tao,NULL,FormFunctionGradient,(void *)&user);CHKERRQ(ierr);
  ierr = TaoSetHessian(tao,user.H,user.H,FormHessian,(void *)&user);CHKERRQ(ierr);

  /* Check for any TAO command line options */
  ierr = TaoGetKSP(tao,&ksp);CHKERRQ(ierr);
  if (ksp) {
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  }
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);

  ierr = TaoSolve(tao);CHKERRQ(ierr);

  ierr = VecView(P,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /* Free TAO data structures */
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&user.H);CHKERRQ(ierr);
  ierr = MatDestroy(&user.A);CHKERRQ(ierr);
  ierr = VecDestroy(&user.U);CHKERRQ(ierr);
  ierr = MatDestroy(&user.Jacp);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Lambda[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Mup[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Lambda2[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Mup2[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Ihp1[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Ihp2[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Ihp3[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Ihp4[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&user.Dir);CHKERRQ(ierr);
  ierr = TSDestroy(&user.ts);CHKERRQ(ierr);
  ierr = VecDestroy(&P);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(P,&y_ptr);CHKERRQ(ierr);
  user_ptr->mu = y_ptr[0];
  ierr = VecRestoreArrayRead(P,&y_ptr);CHKERRQ(ierr);

  ierr = TSSetTime(ts,0.0);CHKERRQ(ierr);
  ierr = TSSetStepNumber(ts,0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,PetscRealConstant(0.001));CHKERRQ(ierr); /* can be overwritten by command line options */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = VecGetArray(user_ptr->U,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 2.0;
  x_ptr[1] = -2.0/3.0 + 10.0/(81.0*user_ptr->mu) - 292.0/(2187.0*user_ptr->mu*user_ptr->mu);
  ierr = VecRestoreArray(user_ptr->U,&x_ptr);CHKERRQ(ierr);

  ierr = TSSolve(ts,user_ptr->U);CHKERRQ(ierr);

  ierr = VecGetArrayRead(user_ptr->U,&y_ptr);CHKERRQ(ierr);
  *f   = (y_ptr[0]-user_ptr->ob[0])*(y_ptr[0]-user_ptr->ob[0])+(y_ptr[1]-user_ptr->ob[1])*(y_ptr[1]-user_ptr->ob[1]);

  /*   Reset initial conditions for the adjoint integration */
  ierr = VecGetArray(user_ptr->Lambda[0],&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 2.*(y_ptr[0]-user_ptr->ob[0]);
  x_ptr[1] = 2.*(y_ptr[1]-user_ptr->ob[1]);
  ierr = VecRestoreArrayRead(user_ptr->U,&y_ptr);CHKERRQ(ierr);
  ierr = VecRestoreArray(user_ptr->Lambda[0],&x_ptr);CHKERRQ(ierr);

  ierr = VecGetArray(user_ptr->Mup[0],&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 0.0;
  ierr = VecRestoreArray(user_ptr->Mup[0],&x_ptr);CHKERRQ(ierr);
  ierr = TSSetCostGradients(ts,1,user_ptr->Lambda,user_ptr->Mup);CHKERRQ(ierr);

  ierr = TSAdjointSolve(ts);CHKERRQ(ierr);

  ierr = VecGetArray(user_ptr->Mup[0],&x_ptr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(user_ptr->Lambda[0],&y_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(G,&g);CHKERRQ(ierr);
  g[0] = y_ptr[1]*(-10.0/(81.0*user_ptr->mu*user_ptr->mu)+2.0*292.0/(2187.0*user_ptr->mu*user_ptr->mu*user_ptr->mu))+x_ptr[0];
  ierr = VecRestoreArray(user_ptr->Mup[0],&x_ptr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(user_ptr->Lambda[0],&y_ptr);CHKERRQ(ierr);
  ierr = VecRestoreArray(G,&g);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormHessian(Tao tao,Vec P,Mat H,Mat Hpre,void *ctx)
{
  User           user_ptr = (User)ctx;
  PetscScalar    harr[1];
  const PetscInt rows[1] = {0};
  PetscInt       col = 0;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = Adjoint2(P,harr,user_ptr);CHKERRQ(ierr);
  ierr = MatSetValues(H,1,rows,1,&col,harr,INSERT_VALUES);CHKERRQ(ierr);

  ierr     = MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr     = MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (H != Hpre) {
    ierr   = MatAssemblyBegin(Hpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr   = MatAssemblyEnd(Hpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode Adjoint2(Vec P,PetscScalar arr[],User ctx)
{
  TS                ts = ctx->ts;
  const PetscScalar *z_ptr;
  PetscScalar       *x_ptr,*y_ptr,dzdp,dzdp2;
  Mat               tlmsen;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  /* Reset TSAdjoint so that AdjointSetUp will be called again */
  ierr = TSAdjointReset(ts);CHKERRQ(ierr);

  /* The directional vector should be 1 since it is one-dimensional */
  ierr     = VecGetArray(ctx->Dir,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 1.;
  ierr     = VecRestoreArray(ctx->Dir,&x_ptr);CHKERRQ(ierr);

  ierr = VecGetArrayRead(P,&z_ptr);CHKERRQ(ierr);
  ctx->mu = z_ptr[0];
  ierr = VecRestoreArrayRead(P,&z_ptr);CHKERRQ(ierr);

  dzdp  = -10.0/(81.0*ctx->mu*ctx->mu) + 2.0*292.0/(2187.0*ctx->mu*ctx->mu*ctx->mu);
  dzdp2 = 2.*10.0/(81.0*ctx->mu*ctx->mu*ctx->mu) - 3.0*2.0*292.0/(2187.0*ctx->mu*ctx->mu*ctx->mu*ctx->mu);

  ierr = TSSetTime(ts,0.0);CHKERRQ(ierr);
  ierr = TSSetStepNumber(ts,0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,PetscRealConstant(0.001));CHKERRQ(ierr); /* can be overwritten by command line options */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetCostHessianProducts(ts,1,ctx->Lambda2,ctx->Mup2,ctx->Dir);CHKERRQ(ierr);

  ierr = MatZeroEntries(ctx->Jacp);CHKERRQ(ierr);
  ierr = MatSetValue(ctx->Jacp,1,0,dzdp,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(ctx->Jacp,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(ctx->Jacp,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = TSAdjointSetForward(ts,ctx->Jacp);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->U,&y_ptr);CHKERRQ(ierr);
  y_ptr[0] = 2.0;
  y_ptr[1] = -2.0/3.0 + 10.0/(81.0*ctx->mu) - 292.0/(2187.0*ctx->mu*ctx->mu);
  ierr = VecRestoreArray(ctx->U,&y_ptr);CHKERRQ(ierr);
  ierr = TSSolve(ts,ctx->U);CHKERRQ(ierr);

  /* Set terminal conditions for first- and second-order adjonts */
  ierr = VecGetArrayRead(ctx->U,&z_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->Lambda[0],&y_ptr);CHKERRQ(ierr);
  y_ptr[0] = 2.*(z_ptr[0]-ctx->ob[0]);
  y_ptr[1] = 2.*(z_ptr[1]-ctx->ob[1]);
  ierr = VecRestoreArray(ctx->Lambda[0],&y_ptr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(ctx->U,&z_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->Mup[0],&y_ptr);CHKERRQ(ierr);
  y_ptr[0] = 0.0;
  ierr = VecRestoreArray(ctx->Mup[0],&y_ptr);CHKERRQ(ierr);
  ierr = TSForwardGetSensitivities(ts,NULL,&tlmsen);CHKERRQ(ierr);
  ierr = MatDenseGetColumn(tlmsen,0,&x_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->Lambda2[0],&y_ptr);CHKERRQ(ierr);
  y_ptr[0] = 2.*x_ptr[0];
  y_ptr[1] = 2.*x_ptr[1];
  ierr = VecRestoreArray(ctx->Lambda2[0],&y_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->Mup2[0],&y_ptr);CHKERRQ(ierr);
  y_ptr[0] = 0.0;
  ierr = VecRestoreArray(ctx->Mup2[0],&y_ptr);CHKERRQ(ierr);
  ierr = MatDenseRestoreColumn(tlmsen,&x_ptr);CHKERRQ(ierr);
  ierr = TSSetCostGradients(ts,1,ctx->Lambda,ctx->Mup);CHKERRQ(ierr);
  if (ctx->implicitform) {
    ierr = TSSetIHessianProduct(ts,ctx->Ihp1,IHessianProductUU,ctx->Ihp2,IHessianProductUP,ctx->Ihp3,IHessianProductPU,ctx->Ihp4,IHessianProductPP,ctx);CHKERRQ(ierr);
  } else {
    ierr = TSSetRHSHessianProduct(ts,ctx->Ihp1,RHSHessianProductUU,ctx->Ihp2,RHSHessianProductUP,ctx->Ihp3,RHSHessianProductPU,ctx->Ihp4,RHSHessianProductPP,ctx);CHKERRQ(ierr);
  }
  ierr = TSAdjointSolve(ts);CHKERRQ(ierr);

  ierr = VecGetArray(ctx->Lambda[0],&x_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(ctx->Lambda2[0],&y_ptr);CHKERRQ(ierr);
  ierr = VecGetArrayRead(ctx->Mup2[0],&z_ptr);CHKERRQ(ierr);

  arr[0] = x_ptr[1]*dzdp2 + y_ptr[1]*dzdp2 + z_ptr[0];

  ierr = VecRestoreArray(ctx->Lambda2[0],&x_ptr);CHKERRQ(ierr);
  ierr = VecRestoreArray(ctx->Lambda2[0],&y_ptr);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(ctx->Mup2[0],&z_ptr);CHKERRQ(ierr);

  /* Disable second-order adjoint mode */
  ierr = TSAdjointReset(ts);CHKERRQ(ierr);
  ierr = TSAdjointResetForward(ts);CHKERRQ(ierr);
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
