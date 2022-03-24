#include <petsctao.h>
#include <petscts.h>

typedef struct _n_aircraft *Aircraft;
struct _n_aircraft {
  TS        ts,quadts;
  Vec       V,W;    /* control variables V and W */
  PetscInt  nsteps; /* number of time steps */
  PetscReal ftime;
  Mat       A,H;
  Mat       Jacp,DRDU,DRDP;
  Vec       U,Lambda[1],Mup[1],Lambda2[1],Mup2[1],Dir;
  Vec       rhshp1[1],rhshp2[1],rhshp3[1],rhshp4[1],inthp1[1],inthp2[1],inthp3[1],inthp4[1];
  PetscReal lv,lw;
  PetscBool mf,eh;
};

PetscErrorCode FormObjFunctionGradient(Tao,Vec,PetscReal *,Vec,void *);
PetscErrorCode FormObjHessian(Tao,Vec,Mat,Mat,void *);
PetscErrorCode ComputeObjHessianWithSOA(Vec,PetscScalar[],Aircraft);
PetscErrorCode MatrixFreeObjHessian(Tao,Vec,Mat,Mat,void *);
PetscErrorCode MyMatMult(Mat,Vec,Vec);

static PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec U,Vec F,void *ctx)
{
  Aircraft          actx = (Aircraft)ctx;
  const PetscScalar *u,*v,*w;
  PetscScalar       *f;
  PetscInt          step;

  PetscFunctionBeginUser;
  CHKERRQ(TSGetStepNumber(ts,&step));
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(actx->V,&v));
  CHKERRQ(VecGetArrayRead(actx->W,&w));
  CHKERRQ(VecGetArray(F,&f));
  f[0] = v[step]*PetscCosReal(w[step]);
  f[1] = v[step]*PetscSinReal(w[step]);
  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(actx->V,&v));
  CHKERRQ(VecRestoreArrayRead(actx->W,&w));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSJacobianP(TS ts,PetscReal t,Vec U,Mat A,void *ctx)
{
  Aircraft          actx = (Aircraft)ctx;
  const PetscScalar *u,*v,*w;
  PetscInt          step,rows[2] = {0,1},rowcol[2];
  PetscScalar       Jp[2][2];

  PetscFunctionBeginUser;
  CHKERRQ(MatZeroEntries(A));
  CHKERRQ(TSGetStepNumber(ts,&step));
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(actx->V,&v));
  CHKERRQ(VecGetArrayRead(actx->W,&w));

  Jp[0][0] = PetscCosReal(w[step]);
  Jp[0][1] = -v[step]*PetscSinReal(w[step]);
  Jp[1][0] = PetscSinReal(w[step]);
  Jp[1][1] = v[step]*PetscCosReal(w[step]);

  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(actx->V,&v));
  CHKERRQ(VecRestoreArrayRead(actx->W,&w));

  rowcol[0] = 2*step;
  rowcol[1] = 2*step+1;
  CHKERRQ(MatSetValues(A,2,rows,2,rowcol,&Jp[0][0],INSERT_VALUES));

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSHessianProductUU(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV,void *ctx)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSHessianProductUP(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV,void *ctx)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSHessianProductPU(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV,void *ctx)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSHessianProductPP(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV,void *ctx)
{
  Aircraft          actx = (Aircraft)ctx;
  const PetscScalar *v,*w,*vl,*vr,*u;
  PetscScalar       *vhv;
  PetscScalar       dJpdP[2][2][2]={{{0}}};
  PetscInt          step,i,j,k;

  PetscFunctionBeginUser;
  CHKERRQ(TSGetStepNumber(ts,&step));
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(actx->V,&v));
  CHKERRQ(VecGetArrayRead(actx->W,&w));
  CHKERRQ(VecGetArrayRead(Vl[0],&vl));
  CHKERRQ(VecGetArrayRead(Vr,&vr));
  CHKERRQ(VecSet(VHV[0],0.0));
  CHKERRQ(VecGetArray(VHV[0],&vhv));

  dJpdP[0][0][1] = -PetscSinReal(w[step]);
  dJpdP[0][1][0] = -PetscSinReal(w[step]);
  dJpdP[0][1][1] = -v[step]*PetscCosReal(w[step]);
  dJpdP[1][0][1] = PetscCosReal(w[step]);
  dJpdP[1][1][0] = PetscCosReal(w[step]);
  dJpdP[1][1][1] = -v[step]*PetscSinReal(w[step]);

  for (j=0; j<2; j++) {
    vhv[2*step+j] = 0;
    for (k=0; k<2; k++)
      for (i=0; i<2; i++)
        vhv[2*step+j] += vl[i]*dJpdP[i][j][k]*vr[2*step+k];
  }
  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(Vl[0],&vl));
  CHKERRQ(VecRestoreArrayRead(Vr,&vr));
  CHKERRQ(VecRestoreArray(VHV[0],&vhv));
  PetscFunctionReturn(0);
}

/* Vl in NULL,updates to VHV must be added */
static PetscErrorCode IntegrandHessianProductUU(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV,void *ctx)
{
  Aircraft          actx = (Aircraft)ctx;
  const PetscScalar *v,*w,*vr,*u;
  PetscScalar       *vhv;
  PetscScalar       dRudU[2][2]={{0}};
  PetscInt          step,j,k;

  PetscFunctionBeginUser;
  CHKERRQ(TSGetStepNumber(ts,&step));
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(actx->V,&v));
  CHKERRQ(VecGetArrayRead(actx->W,&w));
  CHKERRQ(VecGetArrayRead(Vr,&vr));
  CHKERRQ(VecGetArray(VHV[0],&vhv));

  dRudU[0][0] = 2.0;
  dRudU[1][1] = 2.0;

  for (j=0; j<2; j++) {
    vhv[j] = 0;
    for (k=0; k<2; k++)
        vhv[j] += dRudU[j][k]*vr[k];
  }
  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(Vr,&vr));
  CHKERRQ(VecRestoreArray(VHV[0],&vhv));
  PetscFunctionReturn(0);
}

static PetscErrorCode IntegrandHessianProductUP(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV,void *ctx)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(0);
}

static PetscErrorCode IntegrandHessianProductPU(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV,void *ctx)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(0);
}

static PetscErrorCode IntegrandHessianProductPP(TS ts,PetscReal t,Vec U,Vec *Vl,Vec Vr,Vec *VHV,void *ctx)
{
  PetscFunctionBeginUser;
  PetscFunctionReturn(0);
}

static PetscErrorCode CostIntegrand(TS ts,PetscReal t,Vec U,Vec R,void *ctx)
{
  Aircraft          actx = (Aircraft)ctx;
  PetscScalar       *r;
  PetscReal         dx,dy;
  const PetscScalar *u;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArray(R,&r));
  dx   = u[0] - actx->lv*t*PetscCosReal(actx->lw);
  dy   = u[1] - actx->lv*t*PetscSinReal(actx->lw);
  r[0] = dx*dx+dy*dy;
  CHKERRQ(VecRestoreArray(R,&r));
  CHKERRQ(VecRestoreArrayRead(U,&u));
  PetscFunctionReturn(0);
}

static PetscErrorCode DRDUJacobianTranspose(TS ts,PetscReal t,Vec U,Mat DRDU,Mat B,void *ctx)
{
  Aircraft          actx = (Aircraft)ctx;
  PetscScalar       drdu[2][1];
  const PetscScalar *u;
  PetscReal         dx,dy;
  PetscInt          row[] = {0,1},col[] = {0};

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(U,&u));
  dx      = u[0] - actx->lv*t*PetscCosReal(actx->lw);
  dy      = u[1] - actx->lv*t*PetscSinReal(actx->lw);
  drdu[0][0] = 2.*dx;
  drdu[1][0] = 2.*dy;
  CHKERRQ(MatSetValues(DRDU,2,row,1,col,&drdu[0][0],INSERT_VALUES));
  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(MatAssemblyBegin(DRDU,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(DRDU,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

static PetscErrorCode DRDPJacobianTranspose(TS ts,PetscReal t,Vec U,Mat DRDP,void *ctx)
{
  PetscFunctionBegin;
  CHKERRQ(MatZeroEntries(DRDP));
  CHKERRQ(MatAssemblyBegin(DRDP,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(DRDP,MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  Vec                P,PL,PU;
  struct _n_aircraft aircraft;
  PetscMPIInt        size;
  Tao                tao;
  KSP                ksp;
  PC                 pc;
  PetscScalar        *u,*p;
  PetscInt           i;

  /* Initialize program */
  CHKERRQ(PetscInitialize(&argc,&argv,NULL,NULL));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /* Parameter settings */
  aircraft.ftime = 1.;   /* time interval in hour */
  aircraft.nsteps = 10; /* number of steps */
  aircraft.lv = 2.0; /* leader speed in kmph */
  aircraft.lw = PETSC_PI/4.; /* leader heading angle */

  CHKERRQ(PetscOptionsGetReal(NULL,NULL,"-ftime",&aircraft.ftime,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nsteps",&aircraft.nsteps,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-matrixfree",&aircraft.mf));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-exacthessian",&aircraft.eh));

  /* Create TAO solver and set desired solution method */
  CHKERRQ(TaoCreate(PETSC_COMM_WORLD,&tao));
  CHKERRQ(TaoSetType(tao,TAOBQNLS));

  /* Create necessary matrix and vectors, solve same ODE on every process */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&aircraft.A));
  CHKERRQ(MatSetSizes(aircraft.A,PETSC_DECIDE,PETSC_DECIDE,2,2));
  CHKERRQ(MatSetFromOptions(aircraft.A));
  CHKERRQ(MatSetUp(aircraft.A));
  CHKERRQ(MatAssemblyBegin(aircraft.A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(aircraft.A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatShift(aircraft.A,1));
  CHKERRQ(MatShift(aircraft.A,-1));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&aircraft.Jacp));
  CHKERRQ(MatSetSizes(aircraft.Jacp,PETSC_DECIDE,PETSC_DECIDE,2,2*aircraft.nsteps));
  CHKERRQ(MatSetFromOptions(aircraft.Jacp));
  CHKERRQ(MatSetUp(aircraft.Jacp));
  CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,2*aircraft.nsteps,1,NULL,&aircraft.DRDP));
  CHKERRQ(MatSetUp(aircraft.DRDP));
  CHKERRQ(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,&aircraft.DRDU));
  CHKERRQ(MatSetUp(aircraft.DRDU));

  /* Create timestepping solver context */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&aircraft.ts));
  CHKERRQ(TSSetType(aircraft.ts,TSRK));
  CHKERRQ(TSSetRHSFunction(aircraft.ts,NULL,RHSFunction,&aircraft));
  CHKERRQ(TSSetRHSJacobian(aircraft.ts,aircraft.A,aircraft.A,TSComputeRHSJacobianConstant,&aircraft));
  CHKERRQ(TSSetRHSJacobianP(aircraft.ts,aircraft.Jacp,RHSJacobianP,&aircraft));
  CHKERRQ(TSSetExactFinalTime(aircraft.ts,TS_EXACTFINALTIME_MATCHSTEP));
  CHKERRQ(TSSetEquationType(aircraft.ts,TS_EQ_ODE_EXPLICIT)); /* less Jacobian evaluations when adjoint BEuler is used, otherwise no effect */

  /* Set initial conditions */
  CHKERRQ(MatCreateVecs(aircraft.A,&aircraft.U,NULL));
  CHKERRQ(TSSetSolution(aircraft.ts,aircraft.U));
  CHKERRQ(VecGetArray(aircraft.U,&u));
  u[0] = 1.5;
  u[1] = 0;
  CHKERRQ(VecRestoreArray(aircraft.U,&u));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&aircraft.V));
  CHKERRQ(VecSetSizes(aircraft.V,PETSC_DECIDE,aircraft.nsteps));
  CHKERRQ(VecSetUp(aircraft.V));
  CHKERRQ(VecDuplicate(aircraft.V,&aircraft.W));
  CHKERRQ(VecSet(aircraft.V,1.));
  CHKERRQ(VecSet(aircraft.W,PETSC_PI/4.));

  /* Save trajectory of solution so that TSAdjointSolve() may be used */
  CHKERRQ(TSSetSaveTrajectory(aircraft.ts));

  /* Set sensitivity context */
  CHKERRQ(TSCreateQuadratureTS(aircraft.ts,PETSC_FALSE,&aircraft.quadts));
  CHKERRQ(TSSetRHSFunction(aircraft.quadts,NULL,(TSRHSFunction)CostIntegrand,&aircraft));
  CHKERRQ(TSSetRHSJacobian(aircraft.quadts,aircraft.DRDU,aircraft.DRDU,(TSRHSJacobian)DRDUJacobianTranspose,&aircraft));
  CHKERRQ(TSSetRHSJacobianP(aircraft.quadts,aircraft.DRDP,(TSRHSJacobianP)DRDPJacobianTranspose,&aircraft));
  CHKERRQ(MatCreateVecs(aircraft.A,&aircraft.Lambda[0],NULL));
  CHKERRQ(MatCreateVecs(aircraft.Jacp,&aircraft.Mup[0],NULL));
  if (aircraft.eh) {
    CHKERRQ(MatCreateVecs(aircraft.A,&aircraft.rhshp1[0],NULL));
    CHKERRQ(MatCreateVecs(aircraft.A,&aircraft.rhshp2[0],NULL));
    CHKERRQ(MatCreateVecs(aircraft.Jacp,&aircraft.rhshp3[0],NULL));
    CHKERRQ(MatCreateVecs(aircraft.Jacp,&aircraft.rhshp4[0],NULL));
    CHKERRQ(MatCreateVecs(aircraft.DRDU,&aircraft.inthp1[0],NULL));
    CHKERRQ(MatCreateVecs(aircraft.DRDU,&aircraft.inthp2[0],NULL));
    CHKERRQ(MatCreateVecs(aircraft.DRDP,&aircraft.inthp3[0],NULL));
    CHKERRQ(MatCreateVecs(aircraft.DRDP,&aircraft.inthp4[0],NULL));
    CHKERRQ(MatCreateVecs(aircraft.Jacp,&aircraft.Dir,NULL));
    CHKERRQ(TSSetRHSHessianProduct(aircraft.ts,aircraft.rhshp1,RHSHessianProductUU,aircraft.rhshp2,RHSHessianProductUP,aircraft.rhshp3,RHSHessianProductPU,aircraft.rhshp4,RHSHessianProductPP,&aircraft));
    CHKERRQ(TSSetRHSHessianProduct(aircraft.quadts,aircraft.inthp1,IntegrandHessianProductUU,aircraft.inthp2,IntegrandHessianProductUP,aircraft.inthp3,IntegrandHessianProductPU,aircraft.inthp4,IntegrandHessianProductPP,&aircraft));
    CHKERRQ(MatCreateVecs(aircraft.A,&aircraft.Lambda2[0],NULL));
    CHKERRQ(MatCreateVecs(aircraft.Jacp,&aircraft.Mup2[0],NULL));
  }
  CHKERRQ(TSSetFromOptions(aircraft.ts));
  CHKERRQ(TSSetMaxTime(aircraft.ts,aircraft.ftime));
  CHKERRQ(TSSetTimeStep(aircraft.ts,aircraft.ftime/aircraft.nsteps));

  /* Set initial solution guess */
  CHKERRQ(MatCreateVecs(aircraft.Jacp,&P,NULL));
  CHKERRQ(VecGetArray(P,&p));
  for (i=0; i<aircraft.nsteps; i++) {
    p[2*i] = 2.0;
    p[2*i+1] = PETSC_PI/2.0;
  }
  CHKERRQ(VecRestoreArray(P,&p));
  CHKERRQ(VecDuplicate(P,&PU));
  CHKERRQ(VecDuplicate(P,&PL));
  CHKERRQ(VecGetArray(PU,&p));
  for (i=0; i<aircraft.nsteps; i++) {
    p[2*i] = 2.0;
    p[2*i+1] = PETSC_PI;
  }
  CHKERRQ(VecRestoreArray(PU,&p));
  CHKERRQ(VecGetArray(PL,&p));
  for (i=0; i<aircraft.nsteps; i++) {
    p[2*i] = 0.0;
    p[2*i+1] = -PETSC_PI;
  }
  CHKERRQ(VecRestoreArray(PL,&p));

  CHKERRQ(TaoSetSolution(tao,P));
  CHKERRQ(TaoSetVariableBounds(tao,PL,PU));
  /* Set routine for function and gradient evaluation */
  CHKERRQ(TaoSetObjectiveAndGradient(tao,NULL,FormObjFunctionGradient,(void *)&aircraft));

  if (aircraft.eh) {
    if (aircraft.mf) {
      CHKERRQ(MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,2*aircraft.nsteps,2*aircraft.nsteps,(void*)&aircraft,&aircraft.H));
      CHKERRQ(MatShellSetOperation(aircraft.H,MATOP_MULT,(void(*)(void))MyMatMult));
      CHKERRQ(MatSetOption(aircraft.H,MAT_SYMMETRIC,PETSC_TRUE));
      CHKERRQ(TaoSetHessian(tao,aircraft.H,aircraft.H,MatrixFreeObjHessian,(void*)&aircraft));
    } else {
      CHKERRQ(MatCreateDense(MPI_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,2*aircraft.nsteps,2*aircraft.nsteps,NULL,&(aircraft.H)));
      CHKERRQ(MatSetOption(aircraft.H,MAT_SYMMETRIC,PETSC_TRUE));
      CHKERRQ(TaoSetHessian(tao,aircraft.H,aircraft.H,FormObjHessian,(void *)&aircraft));
    }
  }

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
  CHKERRQ(TSDestroy(&aircraft.ts));
  CHKERRQ(MatDestroy(&aircraft.A));
  CHKERRQ(VecDestroy(&aircraft.U));
  CHKERRQ(VecDestroy(&aircraft.V));
  CHKERRQ(VecDestroy(&aircraft.W));
  CHKERRQ(VecDestroy(&P));
  CHKERRQ(VecDestroy(&PU));
  CHKERRQ(VecDestroy(&PL));
  CHKERRQ(MatDestroy(&aircraft.Jacp));
  CHKERRQ(MatDestroy(&aircraft.DRDU));
  CHKERRQ(MatDestroy(&aircraft.DRDP));
  CHKERRQ(VecDestroy(&aircraft.Lambda[0]));
  CHKERRQ(VecDestroy(&aircraft.Mup[0]));
  CHKERRQ(VecDestroy(&P));
  if (aircraft.eh) {
    CHKERRQ(VecDestroy(&aircraft.Lambda2[0]));
    CHKERRQ(VecDestroy(&aircraft.Mup2[0]));
    CHKERRQ(VecDestroy(&aircraft.Dir));
    CHKERRQ(VecDestroy(&aircraft.rhshp1[0]));
    CHKERRQ(VecDestroy(&aircraft.rhshp2[0]));
    CHKERRQ(VecDestroy(&aircraft.rhshp3[0]));
    CHKERRQ(VecDestroy(&aircraft.rhshp4[0]));
    CHKERRQ(VecDestroy(&aircraft.inthp1[0]));
    CHKERRQ(VecDestroy(&aircraft.inthp2[0]));
    CHKERRQ(VecDestroy(&aircraft.inthp3[0]));
    CHKERRQ(VecDestroy(&aircraft.inthp4[0]));
    CHKERRQ(MatDestroy(&aircraft.H));
  }
  CHKERRQ(PetscFinalize());
  return 0;
}

/*
   FormObjFunctionGradient - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   P   - the input vector
   ctx - optional aircraft-defined context, as set by TaoSetObjectiveAndGradient()

   Output Parameters:
   f   - the newly evaluated function
   G   - the newly evaluated gradient
*/
PetscErrorCode FormObjFunctionGradient(Tao tao,Vec P,PetscReal *f,Vec G,void *ctx)
{
  Aircraft          actx = (Aircraft)ctx;
  TS                ts = actx->ts;
  Vec               Q;
  const PetscScalar *p,*q;
  PetscScalar       *u,*v,*w;
  PetscInt          i;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(P,&p));
  CHKERRQ(VecGetArray(actx->V,&v));
  CHKERRQ(VecGetArray(actx->W,&w));
  for (i=0; i<actx->nsteps; i++) {
    v[i] = p[2*i];
    w[i] = p[2*i+1];
  }
  CHKERRQ(VecRestoreArrayRead(P,&p));
  CHKERRQ(VecRestoreArray(actx->V,&v));
  CHKERRQ(VecRestoreArray(actx->W,&w));

  CHKERRQ(TSSetTime(ts,0.0));
  CHKERRQ(TSSetStepNumber(ts,0));
  CHKERRQ(TSSetFromOptions(ts));
  CHKERRQ(TSSetTimeStep(ts,actx->ftime/actx->nsteps));

  /* reinitialize system state */
  CHKERRQ(VecGetArray(actx->U,&u));
  u[0] = 2.0;
  u[1] = 0;
  CHKERRQ(VecRestoreArray(actx->U,&u));

  /* reinitialize the integral value */
  CHKERRQ(TSGetCostIntegral(ts,&Q));
  CHKERRQ(VecSet(Q,0.0));

  CHKERRQ(TSSolve(ts,actx->U));

  /* Reset initial conditions for the adjoint integration */
  CHKERRQ(VecSet(actx->Lambda[0],0.0));
  CHKERRQ(VecSet(actx->Mup[0],0.0));
  CHKERRQ(TSSetCostGradients(ts,1,actx->Lambda,actx->Mup));

  CHKERRQ(TSAdjointSolve(ts));
  CHKERRQ(VecCopy(actx->Mup[0],G));
  CHKERRQ(TSGetCostIntegral(ts,&Q));
  CHKERRQ(VecGetArrayRead(Q,&q));
  *f   = q[0];
  CHKERRQ(VecRestoreArrayRead(Q,&q));
  PetscFunctionReturn(0);
}

PetscErrorCode FormObjHessian(Tao tao,Vec P,Mat H,Mat Hpre,void *ctx)
{
  Aircraft          actx = (Aircraft)ctx;
  const PetscScalar *p;
  PetscScalar       *harr,*v,*w,one = 1.0;
  PetscInt          ind[1];
  PetscInt          *cols,i;
  Vec               Dir;

  PetscFunctionBeginUser;
  /* set up control parameters */
  CHKERRQ(VecGetArrayRead(P,&p));
  CHKERRQ(VecGetArray(actx->V,&v));
  CHKERRQ(VecGetArray(actx->W,&w));
  for (i=0; i<actx->nsteps; i++) {
    v[i] = p[2*i];
    w[i] = p[2*i+1];
  }
  CHKERRQ(VecRestoreArrayRead(P,&p));
  CHKERRQ(VecRestoreArray(actx->V,&v));
  CHKERRQ(VecRestoreArray(actx->W,&w));

  CHKERRQ(PetscMalloc1(2*actx->nsteps,&harr));
  CHKERRQ(PetscMalloc1(2*actx->nsteps,&cols));
  for (i=0; i<2*actx->nsteps; i++) cols[i] = i;
  CHKERRQ(VecDuplicate(P,&Dir));
  for (i=0; i<2*actx->nsteps; i++) {
    ind[0] = i;
    CHKERRQ(VecSet(Dir,0.0));
    CHKERRQ(VecSetValues(Dir,1,ind,&one,INSERT_VALUES));
    CHKERRQ(VecAssemblyBegin(Dir));
    CHKERRQ(VecAssemblyEnd(Dir));
    CHKERRQ(ComputeObjHessianWithSOA(Dir,harr,actx));
    CHKERRQ(MatSetValues(H,1,ind,2*actx->nsteps,cols,harr,INSERT_VALUES));
    CHKERRQ(MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY));
    if (H != Hpre) {
      CHKERRQ(MatAssemblyBegin(Hpre,MAT_FINAL_ASSEMBLY));
      CHKERRQ(MatAssemblyEnd(Hpre,MAT_FINAL_ASSEMBLY));
    }
  }
  CHKERRQ(PetscFree(cols));
  CHKERRQ(PetscFree(harr));
  CHKERRQ(VecDestroy(&Dir));
  PetscFunctionReturn(0);
}

PetscErrorCode MatrixFreeObjHessian(Tao tao, Vec P, Mat H, Mat Hpre, void *ctx)
{
  Aircraft          actx = (Aircraft)ctx;
  PetscScalar       *v,*w;
  const PetscScalar *p;
  PetscInt          i;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(P,&p));
  CHKERRQ(VecGetArray(actx->V,&v));
  CHKERRQ(VecGetArray(actx->W,&w));
  for (i=0; i<actx->nsteps; i++) {
    v[i] = p[2*i];
    w[i] = p[2*i+1];
  }
  CHKERRQ(VecRestoreArrayRead(P,&p));
  CHKERRQ(VecRestoreArray(actx->V,&v));
  CHKERRQ(VecRestoreArray(actx->W,&w));
  PetscFunctionReturn(0);
}

PetscErrorCode MyMatMult(Mat H_shell, Vec X, Vec Y)
{
  PetscScalar    *y;
  void           *ptr;

  PetscFunctionBegin;
  CHKERRQ(MatShellGetContext(H_shell,&ptr));
  CHKERRQ(VecGetArray(Y,&y));
  CHKERRQ(ComputeObjHessianWithSOA(X,y,(Aircraft)ptr));
  CHKERRQ(VecRestoreArray(Y,&y));
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeObjHessianWithSOA(Vec Dir,PetscScalar arr[],Aircraft actx)
{
  TS                ts = actx->ts;
  const PetscScalar *z_ptr;
  PetscScalar       *u;
  Vec               Q;
  PetscInt          i;

  PetscFunctionBeginUser;
  /* Reset TSAdjoint so that AdjointSetUp will be called again */
  CHKERRQ(TSAdjointReset(ts));

  CHKERRQ(TSSetTime(ts,0.0));
  CHKERRQ(TSSetStepNumber(ts,0));
  CHKERRQ(TSSetFromOptions(ts));
  CHKERRQ(TSSetTimeStep(ts,actx->ftime/actx->nsteps));
  CHKERRQ(TSSetCostHessianProducts(actx->ts,1,actx->Lambda2,actx->Mup2,Dir));

  /* reinitialize system state */
  CHKERRQ(VecGetArray(actx->U,&u));
  u[0] = 2.0;
  u[1] = 0;
  CHKERRQ(VecRestoreArray(actx->U,&u));

  /* reinitialize the integral value */
  CHKERRQ(TSGetCostIntegral(ts,&Q));
  CHKERRQ(VecSet(Q,0.0));

  /* initialize tlm variable */
  CHKERRQ(MatZeroEntries(actx->Jacp));
  CHKERRQ(MatAssemblyBegin(actx->Jacp,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(actx->Jacp,MAT_FINAL_ASSEMBLY));
  CHKERRQ(TSAdjointSetForward(ts,actx->Jacp));

  CHKERRQ(TSSolve(ts,actx->U));

  /* Set terminal conditions for first- and second-order adjonts */
  CHKERRQ(VecSet(actx->Lambda[0],0.0));
  CHKERRQ(VecSet(actx->Mup[0],0.0));
  CHKERRQ(VecSet(actx->Lambda2[0],0.0));
  CHKERRQ(VecSet(actx->Mup2[0],0.0));
  CHKERRQ(TSSetCostGradients(ts,1,actx->Lambda,actx->Mup));

  CHKERRQ(TSGetCostIntegral(ts,&Q));

  /* Reset initial conditions for the adjoint integration */
  CHKERRQ(TSAdjointSolve(ts));

  /* initial condition does not depend on p, so that lambda is not needed to assemble G */
  CHKERRQ(VecGetArrayRead(actx->Mup2[0],&z_ptr));
  for (i=0; i<2*actx->nsteps; i++) arr[i] = z_ptr[i];
  CHKERRQ(VecRestoreArrayRead(actx->Mup2[0],&z_ptr));

  /* Disable second-order adjoint mode */
  CHKERRQ(TSAdjointReset(ts));
  CHKERRQ(TSAdjointResetForward(ts));
  PetscFunctionReturn(0);
}

/*TEST
    build:
      requires: !complex !single

    test:
      args:  -ts_adapt_type none -ts_type rk -ts_rk_type 3 -viewer_binary_skip_info -tao_monitor -tao_gatol 1e-7

    test:
      suffix: 2
      args:  -ts_adapt_type none -ts_type rk -ts_rk_type 3 -viewer_binary_skip_info -tao_monitor -tao_view -tao_type bntr -tao_bnk_pc_type none -exacthessian

    test:
      suffix: 3
      args:  -ts_adapt_type none -ts_type rk -ts_rk_type 3 -viewer_binary_skip_info -tao_monitor -tao_view -tao_type bntr -tao_bnk_pc_type none -exacthessian -matrixfree
TEST*/
