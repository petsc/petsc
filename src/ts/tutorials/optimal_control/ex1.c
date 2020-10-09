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
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = TSGetStepNumber(ts,&step);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(actx->V,&v);CHKERRQ(ierr);
  ierr = VecGetArrayRead(actx->W,&w);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = v[step]*PetscCosReal(w[step]);
  f[1] = v[step]*PetscSinReal(w[step]);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(actx->V,&v);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(actx->W,&w);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RHSJacobianP(TS ts,PetscReal t,Vec U,Mat A,void *ctx)
{
  Aircraft          actx = (Aircraft)ctx;
  const PetscScalar *u,*v,*w;
  PetscInt          step,rows[2] = {0,1},rowcol[2];
  PetscScalar       Jp[2][2];
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = MatZeroEntries(A);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&step);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(actx->V,&v);CHKERRQ(ierr);
  ierr = VecGetArrayRead(actx->W,&w);CHKERRQ(ierr);

  Jp[0][0] = PetscCosReal(w[step]);
  Jp[0][1] = -v[step]*PetscSinReal(w[step]);
  Jp[1][0] = PetscSinReal(w[step]);
  Jp[1][1] = v[step]*PetscCosReal(w[step]);

  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(actx->V,&v);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(actx->W,&w);CHKERRQ(ierr);

  rowcol[0] = 2*step;
  rowcol[1] = 2*step+1;
  ierr      = MatSetValues(A,2,rows,2,rowcol,&Jp[0][0],INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = TSGetStepNumber(ts,&step);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(actx->V,&v);CHKERRQ(ierr);
  ierr = VecGetArrayRead(actx->W,&w);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Vl[0],&vl);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Vr,&vr);CHKERRQ(ierr);
  ierr = VecSet(VHV[0],0.0);CHKERRQ(ierr);
  ierr = VecGetArray(VHV[0],&vhv);CHKERRQ(ierr);

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
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Vl[0],&vl);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Vr,&vr);CHKERRQ(ierr);
  ierr = VecRestoreArray(VHV[0],&vhv);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = TSGetStepNumber(ts,&step);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(actx->V,&v);CHKERRQ(ierr);
  ierr = VecGetArrayRead(actx->W,&w);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Vr,&vr);CHKERRQ(ierr);
  ierr = VecGetArray(VHV[0],&vhv);CHKERRQ(ierr);

  dRudU[0][0] = 2.0;
  dRudU[1][1] = 2.0;

  for (j=0; j<2; j++) {
    vhv[j] = 0;
    for (k=0; k<2; k++)
        vhv[j] += dRudU[j][k]*vr[k];
  }
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Vr,&vr);CHKERRQ(ierr);
  ierr = VecRestoreArray(VHV[0],&vhv);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArray(R,&r);CHKERRQ(ierr);
  dx   = u[0] - actx->lv*t*PetscCosReal(actx->lw);
  dy   = u[1] - actx->lv*t*PetscSinReal(actx->lw);
  r[0] = dx*dx+dy*dy;
  ierr = VecRestoreArray(R,&r);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DRDUJacobianTranspose(TS ts,PetscReal t,Vec U,Mat DRDU,Mat B,void *ctx)
{
  Aircraft          actx = (Aircraft)ctx;
  PetscScalar       drdu[2][1];
  const PetscScalar *u;
  PetscReal         dx,dy;
  PetscInt          row[] = {0,1},col[] = {0};
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr    = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  dx      = u[0] - actx->lv*t*PetscCosReal(actx->lw);
  dy      = u[1] - actx->lv*t*PetscSinReal(actx->lw);
  drdu[0][0] = 2.*dx;
  drdu[1][0] = 2.*dy;
  ierr    = MatSetValues(DRDU,2,row,1,col,&drdu[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr    = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr    = MatAssemblyBegin(DRDU,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr    = MatAssemblyEnd(DRDU,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DRDPJacobianTranspose(TS ts,PetscReal t,Vec U,Mat DRDP,void *ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatZeroEntries(DRDP);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(DRDP,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(DRDP,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
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
  PetscErrorCode     ierr;

  /* Initialize program */
  ierr = PetscInitialize(&argc,&argv,NULL,NULL);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /* Parameter settings */
  aircraft.ftime = 1.;   /* time interval in hour */
  aircraft.nsteps = 10; /* number of steps */
  aircraft.lv = 2.0; /* leader speed in kmph */
  aircraft.lw = PETSC_PI/4.; /* leader heading angle */

  ierr = PetscOptionsGetReal(NULL,NULL,"-ftime",&aircraft.ftime,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-nsteps",&aircraft.nsteps,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-matrixfree",&aircraft.mf);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-exacthessian",&aircraft.eh);CHKERRQ(ierr);

  /* Create TAO solver and set desired solution method */
  ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOBQNLS);CHKERRQ(ierr);

  /* Create necessary matrix and vectors, solve same ODE on every process */
  ierr = MatCreate(PETSC_COMM_WORLD,&aircraft.A);CHKERRQ(ierr);
  ierr = MatSetSizes(aircraft.A,PETSC_DECIDE,PETSC_DECIDE,2,2);CHKERRQ(ierr);
  ierr = MatSetFromOptions(aircraft.A);CHKERRQ(ierr);
  ierr = MatSetUp(aircraft.A);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(aircraft.A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(aircraft.A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatShift(aircraft.A,1);CHKERRQ(ierr);
  ierr = MatShift(aircraft.A,-1);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&aircraft.Jacp);CHKERRQ(ierr);
  ierr = MatSetSizes(aircraft.Jacp,PETSC_DECIDE,PETSC_DECIDE,2,2*aircraft.nsteps);CHKERRQ(ierr);
  ierr = MatSetFromOptions(aircraft.Jacp);CHKERRQ(ierr);
  ierr = MatSetUp(aircraft.Jacp);CHKERRQ(ierr);
  ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,2*aircraft.nsteps,1,NULL,&aircraft.DRDP);CHKERRQ(ierr);
  ierr = MatSetUp(aircraft.DRDP);CHKERRQ(ierr);
  ierr = MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,2,1,NULL,&aircraft.DRDU);CHKERRQ(ierr);
  ierr = MatSetUp(aircraft.DRDU);CHKERRQ(ierr);

  /* Create timestepping solver context */
  ierr = TSCreate(PETSC_COMM_WORLD,&aircraft.ts);CHKERRQ(ierr);
  ierr = TSSetType(aircraft.ts,TSRK);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(aircraft.ts,NULL,RHSFunction,&aircraft);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(aircraft.ts,aircraft.A,aircraft.A,TSComputeRHSJacobianConstant,&aircraft);CHKERRQ(ierr);
  ierr = TSSetRHSJacobianP(aircraft.ts,aircraft.Jacp,RHSJacobianP,&aircraft);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(aircraft.ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetEquationType(aircraft.ts,TS_EQ_ODE_EXPLICIT);CHKERRQ(ierr); /* less Jacobian evaluations when adjoint BEuler is used, otherwise no effect */

  /* Set initial conditions */
  ierr = MatCreateVecs(aircraft.A,&aircraft.U,NULL);CHKERRQ(ierr);
  ierr = TSSetSolution(aircraft.ts,aircraft.U);CHKERRQ(ierr);
  ierr = VecGetArray(aircraft.U,&u);CHKERRQ(ierr);
  u[0] = 1.5;
  u[1] = 0;
  ierr = VecRestoreArray(aircraft.U,&u);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&aircraft.V);CHKERRQ(ierr);
  ierr = VecSetSizes(aircraft.V,PETSC_DECIDE,aircraft.nsteps);CHKERRQ(ierr);
  ierr = VecSetUp(aircraft.V);CHKERRQ(ierr);
  ierr = VecDuplicate(aircraft.V,&aircraft.W);CHKERRQ(ierr);
  ierr = VecSet(aircraft.V,1.);CHKERRQ(ierr);
  ierr = VecSet(aircraft.W,PETSC_PI/4.);CHKERRQ(ierr);

  /* Save trajectory of solution so that TSAdjointSolve() may be used */
  ierr = TSSetSaveTrajectory(aircraft.ts);CHKERRQ(ierr);

  /* Set sensitivity context */
  ierr = TSCreateQuadratureTS(aircraft.ts,PETSC_FALSE,&aircraft.quadts);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(aircraft.quadts,NULL,(TSRHSFunction)CostIntegrand,&aircraft);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(aircraft.quadts,aircraft.DRDU,aircraft.DRDU,(TSRHSJacobian)DRDUJacobianTranspose,&aircraft);CHKERRQ(ierr);
  ierr = TSSetRHSJacobianP(aircraft.quadts,aircraft.DRDP,(TSRHSJacobianP)DRDPJacobianTranspose,&aircraft);CHKERRQ(ierr);
  ierr = MatCreateVecs(aircraft.A,&aircraft.Lambda[0],NULL);CHKERRQ(ierr);
  ierr = MatCreateVecs(aircraft.Jacp,&aircraft.Mup[0],NULL);CHKERRQ(ierr);
  if (aircraft.eh) {
    ierr = MatCreateVecs(aircraft.A,&aircraft.rhshp1[0],NULL);CHKERRQ(ierr);
    ierr = MatCreateVecs(aircraft.A,&aircraft.rhshp2[0],NULL);CHKERRQ(ierr);
    ierr = MatCreateVecs(aircraft.Jacp,&aircraft.rhshp3[0],NULL);CHKERRQ(ierr);
    ierr = MatCreateVecs(aircraft.Jacp,&aircraft.rhshp4[0],NULL);CHKERRQ(ierr);
    ierr = MatCreateVecs(aircraft.DRDU,&aircraft.inthp1[0],NULL);CHKERRQ(ierr);
    ierr = MatCreateVecs(aircraft.DRDU,&aircraft.inthp2[0],NULL);CHKERRQ(ierr);
    ierr = MatCreateVecs(aircraft.DRDP,&aircraft.inthp3[0],NULL);CHKERRQ(ierr);
    ierr = MatCreateVecs(aircraft.DRDP,&aircraft.inthp4[0],NULL);CHKERRQ(ierr);
    ierr = MatCreateVecs(aircraft.Jacp,&aircraft.Dir,NULL);CHKERRQ(ierr);
    ierr = TSSetRHSHessianProduct(aircraft.ts,aircraft.rhshp1,RHSHessianProductUU,aircraft.rhshp2,RHSHessianProductUP,aircraft.rhshp3,RHSHessianProductPU,aircraft.rhshp4,RHSHessianProductPP,&aircraft);CHKERRQ(ierr);
    ierr = TSSetRHSHessianProduct(aircraft.quadts,aircraft.inthp1,IntegrandHessianProductUU,aircraft.inthp2,IntegrandHessianProductUP,aircraft.inthp3,IntegrandHessianProductPU,aircraft.inthp4,IntegrandHessianProductPP,&aircraft);CHKERRQ(ierr);
    ierr = MatCreateVecs(aircraft.A,&aircraft.Lambda2[0],NULL);CHKERRQ(ierr);
    ierr = MatCreateVecs(aircraft.Jacp,&aircraft.Mup2[0],NULL);CHKERRQ(ierr);
  }
  ierr = TSSetFromOptions(aircraft.ts);CHKERRQ(ierr);
  ierr = TSSetMaxTime(aircraft.ts,aircraft.ftime);CHKERRQ(ierr);
  ierr = TSSetTimeStep(aircraft.ts,aircraft.ftime/aircraft.nsteps);CHKERRQ(ierr);

  /* Set initial solution guess */
  ierr = MatCreateVecs(aircraft.Jacp,&P,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(P,&p);CHKERRQ(ierr);
  for (i=0; i<aircraft.nsteps; i++) {
    p[2*i] = 2.0;
    p[2*i+1] = PETSC_PI/2.0;
  }
  ierr = VecRestoreArray(P,&p);CHKERRQ(ierr);
  ierr = VecDuplicate(P,&PU);CHKERRQ(ierr);
  ierr = VecDuplicate(P,&PL);CHKERRQ(ierr);
  ierr = VecGetArray(PU,&p);CHKERRQ(ierr);
  for (i=0; i<aircraft.nsteps; i++) {
    p[2*i] = 2.0;
    p[2*i+1] = PETSC_PI;
  }
  ierr = VecRestoreArray(PU,&p);CHKERRQ(ierr);
  ierr = VecGetArray(PL,&p);CHKERRQ(ierr);
  for (i=0; i<aircraft.nsteps; i++) {
    p[2*i] = 0.0;
    p[2*i+1] = -PETSC_PI;
  }
  ierr = VecRestoreArray(PL,&p);CHKERRQ(ierr);

  ierr = TaoSetInitialVector(tao,P);CHKERRQ(ierr);
  ierr = TaoSetVariableBounds(tao,PL,PU);CHKERRQ(ierr);
  /* Set routine for function and gradient evaluation */
  ierr = TaoSetObjectiveAndGradientRoutine(tao,FormObjFunctionGradient,(void *)&aircraft);CHKERRQ(ierr);

  if (aircraft.eh) {
    if (aircraft.mf) {
      ierr = MatCreateShell(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,2*aircraft.nsteps,2*aircraft.nsteps,(void*)&aircraft,&aircraft.H);CHKERRQ(ierr);
      ierr = MatShellSetOperation(aircraft.H,MATOP_MULT,(void(*)(void))MyMatMult);CHKERRQ(ierr);
      ierr = MatSetOption(aircraft.H,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
      ierr = TaoSetHessianRoutine(tao,aircraft.H,aircraft.H,MatrixFreeObjHessian,(void*)&aircraft);CHKERRQ(ierr);
    } else {
      ierr = MatCreateDense(MPI_COMM_WORLD,PETSC_DETERMINE,PETSC_DETERMINE,2*aircraft.nsteps,2*aircraft.nsteps,NULL,&(aircraft.H));CHKERRQ(ierr);
      ierr = MatSetOption(aircraft.H,MAT_SYMMETRIC,PETSC_TRUE);CHKERRQ(ierr);
      ierr = TaoSetHessianRoutine(tao,aircraft.H,aircraft.H,FormObjHessian,(void *)&aircraft);CHKERRQ(ierr);
    }
  }

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
  ierr = TSDestroy(&aircraft.ts);CHKERRQ(ierr);
  ierr = MatDestroy(&aircraft.A);CHKERRQ(ierr);
  ierr = VecDestroy(&aircraft.U);CHKERRQ(ierr);
  ierr = VecDestroy(&aircraft.V);CHKERRQ(ierr);
  ierr = VecDestroy(&aircraft.W);CHKERRQ(ierr);
  ierr = VecDestroy(&P);CHKERRQ(ierr);
  ierr = VecDestroy(&PU);CHKERRQ(ierr);
  ierr = VecDestroy(&PL);CHKERRQ(ierr);
  ierr = MatDestroy(&aircraft.Jacp);CHKERRQ(ierr);
  ierr = MatDestroy(&aircraft.DRDU);CHKERRQ(ierr);
  ierr = MatDestroy(&aircraft.DRDP);CHKERRQ(ierr);
  ierr = VecDestroy(&aircraft.Lambda[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&aircraft.Mup[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&P);CHKERRQ(ierr);
  if (aircraft.eh) {
    ierr = VecDestroy(&aircraft.Lambda2[0]);CHKERRQ(ierr);
    ierr = VecDestroy(&aircraft.Mup2[0]);CHKERRQ(ierr);
    ierr = VecDestroy(&aircraft.Dir);CHKERRQ(ierr);
    ierr = VecDestroy(&aircraft.rhshp1[0]);CHKERRQ(ierr);
    ierr = VecDestroy(&aircraft.rhshp2[0]);CHKERRQ(ierr);
    ierr = VecDestroy(&aircraft.rhshp3[0]);CHKERRQ(ierr);
    ierr = VecDestroy(&aircraft.rhshp4[0]);CHKERRQ(ierr);
    ierr = VecDestroy(&aircraft.inthp1[0]);CHKERRQ(ierr);
    ierr = VecDestroy(&aircraft.inthp2[0]);CHKERRQ(ierr);
    ierr = VecDestroy(&aircraft.inthp3[0]);CHKERRQ(ierr);
    ierr = VecDestroy(&aircraft.inthp4[0]);CHKERRQ(ierr);
    ierr = MatDestroy(&aircraft.H);CHKERRQ(ierr);
  }
  ierr = PetscFinalize();
  return ierr;
}

/*
   FormObjFunctionGradient - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   P   - the input vector
   ctx - optional aircraft-defined context, as set by TaoSetObjectiveAndGradientRoutine()

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
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(P,&p);CHKERRQ(ierr);
  ierr = VecGetArray(actx->V,&v);CHKERRQ(ierr);
  ierr = VecGetArray(actx->W,&w);CHKERRQ(ierr);
  for (i=0; i<actx->nsteps; i++) {
    v[i] = p[2*i];
    w[i] = p[2*i+1];
  }
  ierr = VecRestoreArrayRead(P,&p);CHKERRQ(ierr);
  ierr = VecRestoreArray(actx->V,&v);CHKERRQ(ierr);
  ierr = VecRestoreArray(actx->W,&w);CHKERRQ(ierr);

  ierr = TSSetTime(ts,0.0);CHKERRQ(ierr);
  ierr = TSSetStepNumber(ts,0);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,actx->ftime/actx->nsteps);CHKERRQ(ierr);

  /* reinitialize system state */
  ierr = VecGetArray(actx->U,&u);CHKERRQ(ierr);
  u[0] = 2.0;
  u[1] = 0;
  ierr = VecRestoreArray(actx->U,&u);CHKERRQ(ierr);

  /* reinitialize the integral value */
  ierr = TSGetCostIntegral(ts,&Q);CHKERRQ(ierr);
  ierr = VecSet(Q,0.0);CHKERRQ(ierr);

  ierr = TSSolve(ts,actx->U);CHKERRQ(ierr);

  /* Reset initial conditions for the adjoint integration */
  ierr = VecSet(actx->Lambda[0],0.0);CHKERRQ(ierr);
  ierr = VecSet(actx->Mup[0],0.0);CHKERRQ(ierr);
  ierr = TSSetCostGradients(ts,1,actx->Lambda,actx->Mup);CHKERRQ(ierr);

  ierr = TSAdjointSolve(ts);CHKERRQ(ierr);
  ierr = VecCopy(actx->Mup[0],G);CHKERRQ(ierr);
  ierr = TSGetCostIntegral(ts,&Q);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Q,&q);CHKERRQ(ierr);
  *f   = q[0];
  ierr = VecRestoreArrayRead(Q,&q);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  /* set up control parameters */
  ierr = VecGetArrayRead(P,&p);CHKERRQ(ierr);
  ierr = VecGetArray(actx->V,&v);CHKERRQ(ierr);
  ierr = VecGetArray(actx->W,&w);CHKERRQ(ierr);
  for (i=0; i<actx->nsteps; i++) {
    v[i] = p[2*i];
    w[i] = p[2*i+1];
  }
  ierr = VecRestoreArrayRead(P,&p);CHKERRQ(ierr);
  ierr = VecRestoreArray(actx->V,&v);CHKERRQ(ierr);
  ierr = VecRestoreArray(actx->W,&w);CHKERRQ(ierr);

  ierr = PetscMalloc1(2*actx->nsteps,&harr);CHKERRQ(ierr);
  ierr = PetscMalloc1(2*actx->nsteps,&cols);CHKERRQ(ierr);
  for (i=0; i<2*actx->nsteps; i++) cols[i] = i;
  ierr = VecDuplicate(P,&Dir);CHKERRQ(ierr);
  for (i=0; i<2*actx->nsteps; i++) {
    ind[0] = i;
    ierr   = VecSet(Dir,0.0);CHKERRQ(ierr);
    ierr   = VecSetValues(Dir,1,ind,&one,INSERT_VALUES);CHKERRQ(ierr);
    ierr   = VecAssemblyBegin(Dir);CHKERRQ(ierr);
    ierr   = VecAssemblyEnd(Dir);CHKERRQ(ierr);
    ierr   = ComputeObjHessianWithSOA(Dir,harr,actx);CHKERRQ(ierr);
    ierr   = MatSetValues(H,1,ind,2*actx->nsteps,cols,harr,INSERT_VALUES);CHKERRQ(ierr);
    ierr   = MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr   = MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    if (H != Hpre) {
      ierr   = MatAssemblyBegin(Hpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr   = MatAssemblyEnd(Hpre,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    }
  }
  ierr = PetscFree(cols);CHKERRQ(ierr);
  ierr = PetscFree(harr);CHKERRQ(ierr);
  ierr = VecDestroy(&Dir);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatrixFreeObjHessian(Tao tao, Vec P, Mat H, Mat Hpre, void *ctx)
{
  Aircraft          actx = (Aircraft)ctx;
  PetscScalar       *v,*w;
  const PetscScalar *p;
  PetscInt          i;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = VecGetArrayRead(P,&p);CHKERRQ(ierr);
  ierr = VecGetArray(actx->V,&v);CHKERRQ(ierr);
  ierr = VecGetArray(actx->W,&w);CHKERRQ(ierr);
  for (i=0; i<actx->nsteps; i++) {
    v[i] = p[2*i];
    w[i] = p[2*i+1];
  }
  ierr = VecRestoreArrayRead(P,&p);CHKERRQ(ierr);
  ierr = VecRestoreArray(actx->V,&v);CHKERRQ(ierr);
  ierr = VecRestoreArray(actx->W,&w);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MyMatMult(Mat H_shell, Vec X, Vec Y)
{
  PetscScalar    *y;
  void           *ptr;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatShellGetContext(H_shell,&ptr);CHKERRQ(ierr);
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  ierr = ComputeObjHessianWithSOA(X,y,(Aircraft)ptr);CHKERRQ(ierr);
  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeObjHessianWithSOA(Vec Dir,PetscScalar arr[],Aircraft actx)
{
  TS                ts = actx->ts;
  const PetscScalar *z_ptr;
  PetscScalar       *u;
  Vec               Q;
  PetscInt          i;
  PetscErrorCode    ierr;

  PetscFunctionBeginUser;
  /* Reset TSAdjoint so that AdjointSetUp will be called again */
  ierr = TSAdjointReset(ts);CHKERRQ(ierr);

  ierr = TSSetTime(ts,0.0);CHKERRQ(ierr);
  ierr = TSSetStepNumber(ts,0);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,actx->ftime/actx->nsteps);CHKERRQ(ierr);
  ierr = TSSetCostHessianProducts(actx->ts,1,actx->Lambda2,actx->Mup2,Dir);CHKERRQ(ierr);

  /* reinitialize system state */
  ierr = VecGetArray(actx->U,&u);CHKERRQ(ierr);
  u[0] = 2.0;
  u[1] = 0;
  ierr = VecRestoreArray(actx->U,&u);CHKERRQ(ierr);

  /* reinitialize the integral value */
  ierr = TSGetCostIntegral(ts,&Q);CHKERRQ(ierr);
  ierr = VecSet(Q,0.0);CHKERRQ(ierr);

  /* initialize tlm variable */
  ierr = MatZeroEntries(actx->Jacp);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(actx->Jacp,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(actx->Jacp,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = TSAdjointSetForward(ts,actx->Jacp);CHKERRQ(ierr);

  ierr = TSSolve(ts,actx->U);CHKERRQ(ierr);

  /* Set terminal conditions for first- and second-order adjonts */
  ierr = VecSet(actx->Lambda[0],0.0);CHKERRQ(ierr);
  ierr = VecSet(actx->Mup[0],0.0);CHKERRQ(ierr);
  ierr = VecSet(actx->Lambda2[0],0.0);CHKERRQ(ierr);
  ierr = VecSet(actx->Mup2[0],0.0);CHKERRQ(ierr);
  ierr = TSSetCostGradients(ts,1,actx->Lambda,actx->Mup);CHKERRQ(ierr);

  ierr = TSGetCostIntegral(ts,&Q);CHKERRQ(ierr);

  /* Reset initial conditions for the adjoint integration */
  ierr = TSAdjointSolve(ts);CHKERRQ(ierr);

  /* initial condition does not depend on p, so that lambda is not needed to assemble G */
  ierr = VecGetArrayRead(actx->Mup2[0],&z_ptr);CHKERRQ(ierr);
  for (i=0; i<2*actx->nsteps; i++) arr[i] = z_ptr[i];
  ierr = VecRestoreArrayRead(actx->Mup2[0],&z_ptr);CHKERRQ(ierr);

  /* Disable second-order adjoint mode */
  ierr = TSAdjointReset(ts);CHKERRQ(ierr);
  ierr = TSAdjointResetForward(ts);CHKERRQ(ierr);
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


