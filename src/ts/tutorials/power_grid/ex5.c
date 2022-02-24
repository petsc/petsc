
static char help[] = "Basic equation for an induction generator driven by a wind turbine.\n";

/*F
\begin{eqnarray}
          T_w\frac{dv_w}{dt} & = & v_w - v_we \\
          2(H_t+H_m)\frac{ds}{dt} & = & P_w - P_e
\end{eqnarray}
F*/
/*
 - Pw is the power extracted from the wind turbine given by
           Pw = 0.5*\rho*cp*Ar*vw^3

 - The wind speed time series is modeled using a Weibull distribution and then
   passed through a low pass filter (with time constant T_w).
 - v_we is the wind speed data calculated using Weibull distribution while v_w is
   the output of the filter.
 - P_e is assumed as constant electrical torque

 - This example does not work with adaptive time stepping!

Reference:
Power System Modeling and Scripting - F. Milano
*/
/*T

T*/

#include <petscts.h>

#define freq 50
#define ws (2*PETSC_PI*freq)
#define MVAbase 100

typedef struct {
  /* Parameters for wind speed model */
  PetscInt  nsamples; /* Number of wind samples */
  PetscReal cw;   /* Scale factor for Weibull distribution */
  PetscReal kw;   /* Shape factor for Weibull distribution */
  Vec       wind_data; /* Vector to hold wind speeds */
  Vec       t_wind; /* Vector to hold wind speed times */
  PetscReal Tw;     /* Filter time constant */

  /* Wind turbine parameters */
  PetscScalar Rt; /* Rotor radius */
  PetscScalar Ar; /* Area swept by rotor (pi*R*R) */
  PetscReal   nGB; /* Gear box ratio */
  PetscReal   Ht;  /* Turbine inertia constant */
  PetscReal   rho; /* Atmospheric pressure */

  /* Induction generator parameters */
  PetscInt    np; /* Number of poles */
  PetscReal   Xm; /* Magnetizing reactance */
  PetscReal   Xs; /* Stator Reactance */
  PetscReal   Xr; /* Rotor reactance */
  PetscReal   Rs; /* Stator resistance */
  PetscReal   Rr; /* Rotor resistance */
  PetscReal   Hm; /* Motor inertia constant */
  PetscReal   Xp; /* Xs + Xm*Xr/(Xm + Xr) */
  PetscScalar Te; /* Electrical Torque */

  Mat      Sol;   /* Solution matrix */
  PetscInt stepnum;   /* Column number of solution matrix */
} AppCtx;

/* Initial values computed by Power flow and initialization */
PetscScalar s = -0.00011577790353;
/*Pw = 0.011064344110238; %Te*wm */
PetscScalar       vwa  = 22.317142184449754;
PetscReal         tmax = 20.0;

/* Saves the solution at each time to a matrix */
PetscErrorCode SaveSolution(TS ts)
{
  AppCtx            *user;
  Vec               X;
  PetscScalar       *mat;
  const PetscScalar *x;
  PetscInt          idx;
  PetscReal         t;

  PetscFunctionBegin;
  CHKERRQ(TSGetApplicationContext(ts,&user));
  CHKERRQ(TSGetTime(ts,&t));
  CHKERRQ(TSGetSolution(ts,&X));
  idx      =  3*user->stepnum;
  CHKERRQ(MatDenseGetArray(user->Sol,&mat));
  CHKERRQ(VecGetArrayRead(X,&x));
  mat[idx] = t;
  CHKERRQ(PetscArraycpy(mat+idx+1,x,2));
  CHKERRQ(MatDenseRestoreArray(user->Sol,&mat));
  CHKERRQ(VecRestoreArrayRead(X,&x));
  user->stepnum++;
  PetscFunctionReturn(0);
}

/* Computes the wind speed using Weibull distribution */
PetscErrorCode WindSpeeds(AppCtx *user)
{
  PetscErrorCode ierr;
  PetscScalar    *x,*t,avg_dev,sum;
  PetscInt       i;

  PetscFunctionBegin;
  user->cw       = 5;
  user->kw       = 2; /* Rayleigh distribution */
  user->nsamples = 2000;
  user->Tw       = 0.2;
  ierr           = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Wind Speed Options","");CHKERRQ(ierr);
  {
    CHKERRQ(PetscOptionsReal("-cw","","",user->cw,&user->cw,NULL));
    CHKERRQ(PetscOptionsReal("-kw","","",user->kw,&user->kw,NULL));
    CHKERRQ(PetscOptionsInt("-nsamples","","",user->nsamples,&user->nsamples,NULL));
    CHKERRQ(PetscOptionsReal("-Tw","","",user->Tw,&user->Tw,NULL));
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&user->wind_data));
  CHKERRQ(VecSetSizes(user->wind_data,PETSC_DECIDE,user->nsamples));
  CHKERRQ(VecSetFromOptions(user->wind_data));
  CHKERRQ(VecDuplicate(user->wind_data,&user->t_wind));

  CHKERRQ(VecGetArray(user->t_wind,&t));
  for (i=0; i < user->nsamples; i++) t[i] = (i+1)*tmax/user->nsamples;
  CHKERRQ(VecRestoreArray(user->t_wind,&t));

  /* Wind speed deviation = (-log(rand)/cw)^(1/kw) */
  CHKERRQ(VecSetRandom(user->wind_data,NULL));
  CHKERRQ(VecLog(user->wind_data));
  CHKERRQ(VecScale(user->wind_data,-1/user->cw));
  CHKERRQ(VecGetArray(user->wind_data,&x));
  for (i=0;i < user->nsamples;i++) x[i] = PetscPowScalar(x[i],(1/user->kw));
  CHKERRQ(VecRestoreArray(user->wind_data,&x));
  CHKERRQ(VecSum(user->wind_data,&sum));
  avg_dev = sum/user->nsamples;
  /* Wind speed (t) = (1 + wind speed deviation(t) - avg_dev)*average wind speed */
  CHKERRQ(VecShift(user->wind_data,(1-avg_dev)));
  CHKERRQ(VecScale(user->wind_data,vwa));
  PetscFunctionReturn(0);
}

/* Sets the parameters for wind turbine */
PetscErrorCode SetWindTurbineParams(AppCtx *user)
{
  PetscFunctionBegin;
  user->Rt  = 35;
  user->Ar  = PETSC_PI*user->Rt*user->Rt;
  user->nGB = 1.0/89.0;
  user->rho = 1.225;
  user->Ht  = 1.5;
  PetscFunctionReturn(0);
}

/* Sets the parameters for induction generator */
PetscErrorCode SetInductionGeneratorParams(AppCtx *user)
{
  PetscFunctionBegin;
  user->np = 4;
  user->Xm = 3.0;
  user->Xs = 0.1;
  user->Xr = 0.08;
  user->Rs = 0.01;
  user->Rr = 0.01;
  user->Xp = user->Xs + user->Xm*user->Xr/(user->Xm + user->Xr);
  user->Hm = 1.0;
  user->Te = 0.011063063063251968;
  PetscFunctionReturn(0);
}

/* Computes the power extracted from wind */
PetscErrorCode GetWindPower(PetscScalar wm,PetscScalar vw,PetscScalar *Pw,AppCtx *user)
{
  PetscScalar temp,lambda,lambda_i,cp;

  PetscFunctionBegin;
  temp     = user->nGB*2*user->Rt*ws/user->np;
  lambda   = temp*wm/vw;
  lambda_i = 1/(1/lambda + 0.002);
  cp       = 0.44*(125/lambda_i - 6.94)*PetscExpScalar(-16.5/lambda_i);
  *Pw      = 0.5*user->rho*cp*user->Ar*vw*vw*vw/(MVAbase*1e6);
  PetscFunctionReturn(0);
}

/*
     Defines the ODE passed to the ODE solver
*/
static PetscErrorCode IFunction(TS ts,PetscReal t,Vec U,Vec Udot,Vec F,AppCtx *user)
{
  PetscScalar       *f,wm,Pw,*wd;
  const PetscScalar *u,*udot;
  PetscInt          stepnum;

  PetscFunctionBegin;
  CHKERRQ(TSGetStepNumber(ts,&stepnum));
  /*  The next three lines allow us to access the entries of the vectors directly */
  CHKERRQ(VecGetArrayRead(U,&u));
  CHKERRQ(VecGetArrayRead(Udot,&udot));
  CHKERRQ(VecGetArray(F,&f));
  CHKERRQ(VecGetArray(user->wind_data,&wd));

  f[0] = user->Tw*udot[0] - wd[stepnum] + u[0];
  wm   = 1-u[1];
  CHKERRQ(GetWindPower(wm,u[0],&Pw,user));
  f[1] = 2.0*(user->Ht+user->Hm)*udot[1] - Pw/wm + user->Te;

  CHKERRQ(VecRestoreArray(user->wind_data,&wd));
  CHKERRQ(VecRestoreArrayRead(U,&u));
  CHKERRQ(VecRestoreArrayRead(Udot,&udot));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  TS                ts;            /* ODE integrator */
  Vec               U;             /* solution will be stored here */
  Mat               A;             /* Jacobian matrix */
  PetscErrorCode    ierr;
  PetscMPIInt       size;
  PetscInt          n = 2,idx;
  AppCtx            user;
  PetscScalar       *u;
  SNES              snes;
  PetscScalar       *mat;
  const PetscScalar *x,*rmat;
  Mat               B;
  PetscScalar       *amat;
  PetscViewer       viewer;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Only for sequential runs");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,n,n,PETSC_DETERMINE,PETSC_DETERMINE));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));

  CHKERRQ(MatCreateVecs(A,&U,NULL));

  /* Create wind speed data using Weibull distribution */
  CHKERRQ(WindSpeeds(&user));
  /* Set parameters for wind turbine and induction generator */
  CHKERRQ(SetWindTurbineParams(&user));
  CHKERRQ(SetInductionGeneratorParams(&user));

  CHKERRQ(VecGetArray(U,&u));
  u[0] = vwa;
  u[1] = s;
  CHKERRQ(VecRestoreArray(U,&u));

  /* Create matrix to save solutions at each time step */
  user.stepnum = 0;

  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,3,2010,NULL,&user.Sol));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetProblemType(ts,TS_NONLINEAR));
  CHKERRQ(TSSetType(ts,TSBEULER));
  CHKERRQ(TSSetIFunction(ts,NULL,(TSIFunction) IFunction,&user));

  CHKERRQ(TSGetSNES(ts,&snes));
  CHKERRQ(SNESSetJacobian(snes,A,A,SNESComputeJacobianDefault,NULL));
  /*  CHKERRQ(TSSetIJacobian(ts,A,A,(TSIJacobian)IJacobian,&user)); */
  CHKERRQ(TSSetApplicationContext(ts,&user));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetSolution(ts,U));

  /* Save initial solution */
  idx=3*user.stepnum;

  CHKERRQ(MatDenseGetArray(user.Sol,&mat));
  CHKERRQ(VecGetArrayRead(U,&x));

  mat[idx] = 0.0;

  CHKERRQ(PetscArraycpy(mat+idx+1,x,2));
  CHKERRQ(MatDenseRestoreArray(user.Sol,&mat));
  CHKERRQ(VecRestoreArrayRead(U,&x));
  user.stepnum++;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetMaxTime(ts,20.0));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP));
  CHKERRQ(TSSetTimeStep(ts,.01));
  CHKERRQ(TSSetFromOptions(ts));
  CHKERRQ(TSSetPostStep(ts,SaveSolution));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSolve(ts,U));

  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,3,user.stepnum,NULL,&B));
  CHKERRQ(MatDenseGetArrayRead(user.Sol,&rmat));
  CHKERRQ(MatDenseGetArray(B,&amat));
  CHKERRQ(PetscArraycpy(amat,rmat,user.stepnum*3));
  CHKERRQ(MatDenseRestoreArray(B,&amat));
  CHKERRQ(MatDenseRestoreArrayRead(user.Sol,&rmat));

  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_SELF,"out.bin",FILE_MODE_WRITE,&viewer));
  CHKERRQ(MatView(B,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(MatDestroy(&user.Sol));
  CHKERRQ(MatDestroy(&B));
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecDestroy(&user.wind_data));
  CHKERRQ(VecDestroy(&user.t_wind));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&U));
  CHKERRQ(TSDestroy(&ts));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: !complex

   test:

TEST*/
