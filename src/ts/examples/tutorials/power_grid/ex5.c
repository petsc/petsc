
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
  PetscErrorCode    ierr;
  AppCtx            *user;
  Vec               X;
  PetscScalar       *mat;
  const PetscScalar *x;
  PetscInt          idx;
  PetscReal         t;

  PetscFunctionBegin;
  ierr     = TSGetApplicationContext(ts,&user);CHKERRQ(ierr);
  ierr     = TSGetTime(ts,&t);CHKERRQ(ierr);
  ierr     = TSGetSolution(ts,&X);CHKERRQ(ierr);
  idx      =  3*user->stepnum;
  ierr     = MatDenseGetArray(user->Sol,&mat);CHKERRQ(ierr);
  ierr     = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  mat[idx] = t;
  ierr     = PetscArraycpy(mat+idx+1,x,2);CHKERRQ(ierr);
  ierr     = MatDenseRestoreArray(user->Sol,&mat);CHKERRQ(ierr);
  ierr     = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
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
    ierr = PetscOptionsReal("-cw","","",user->cw,&user->cw,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-kw","","",user->kw,&user->kw,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsInt("-nsamples","","",user->nsamples,&user->nsamples,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-Tw","","",user->Tw,&user->Tw,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&user->wind_data);CHKERRQ(ierr);
  ierr = VecSetSizes(user->wind_data,PETSC_DECIDE,user->nsamples);CHKERRQ(ierr);
  ierr = VecSetFromOptions(user->wind_data);CHKERRQ(ierr);
  ierr = VecDuplicate(user->wind_data,&user->t_wind);CHKERRQ(ierr);

  ierr = VecGetArray(user->t_wind,&t);CHKERRQ(ierr);
  for (i=0; i < user->nsamples; i++) t[i] = (i+1)*tmax/user->nsamples;
  ierr = VecRestoreArray(user->t_wind,&t);CHKERRQ(ierr);

  /* Wind speed deviation = (-log(rand)/cw)^(1/kw) */
  ierr = VecSetRandom(user->wind_data,NULL);CHKERRQ(ierr);
  ierr = VecLog(user->wind_data);CHKERRQ(ierr);
  ierr = VecScale(user->wind_data,-1/user->cw);CHKERRQ(ierr);
  ierr = VecGetArray(user->wind_data,&x);CHKERRQ(ierr);
  for (i=0;i < user->nsamples;i++) x[i] = PetscPowScalar(x[i],(1/user->kw));
  ierr = VecRestoreArray(user->wind_data,&x);CHKERRQ(ierr);
  ierr = VecSum(user->wind_data,&sum);CHKERRQ(ierr);
  avg_dev = sum/user->nsamples;
  /* Wind speed (t) = (1 + wind speed deviation(t) - avg_dev)*average wind speed */
  ierr = VecShift(user->wind_data,(1-avg_dev));CHKERRQ(ierr);
  ierr = VecScale(user->wind_data,vwa);CHKERRQ(ierr);
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
  PetscErrorCode    ierr;
  PetscScalar       *f,wm,Pw,*wd;
  const PetscScalar *u,*udot;
  PetscInt          stepnum;

  PetscFunctionBegin;
  ierr = TSGetStepNumber(ts,&stepnum);CHKERRQ(ierr);
  /*  The next three lines allow us to access the entries of the vectors directly */
  ierr = VecGetArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecGetArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  ierr = VecGetArray(user->wind_data,&wd);CHKERRQ(ierr);

  f[0] = user->Tw*udot[0] - wd[stepnum] + u[0];
  wm   = 1-u[1];
  ierr = GetWindPower(wm,u[0],&Pw,user);CHKERRQ(ierr);
  f[1] = 2.0*(user->Ht+user->Hm)*udot[1] - Pw/wm + user->Te;

  ierr = VecRestoreArray(user->wind_data,&wd);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&u);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(Udot,&udot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
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
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size > 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only for sequential runs");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,n,n,PETSC_DETERMINE,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);

  ierr = MatCreateVecs(A,&U,NULL);CHKERRQ(ierr);

  /* Create wind speed data using Weibull distribution */
  ierr = WindSpeeds(&user);CHKERRQ(ierr);
  /* Set parameters for wind turbine and induction generator */
  ierr = SetWindTurbineParams(&user);CHKERRQ(ierr);
  ierr = SetInductionGeneratorParams(&user);CHKERRQ(ierr);

  ierr = VecGetArray(U,&u);CHKERRQ(ierr);
  u[0] = vwa;
  u[1] = s;
  ierr = VecRestoreArray(U,&u);CHKERRQ(ierr);

  /* Create matrix to save solutions at each time step */
  user.stepnum = 0;

  ierr = MatCreateSeqDense(PETSC_COMM_SELF,3,2010,NULL,&user.Sol);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(ts,TS_NONLINEAR);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,(TSIFunction) IFunction,&user);CHKERRQ(ierr);

  ierr = TSGetSNES(ts,&snes);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,A,A,SNESComputeJacobianDefault,NULL);CHKERRQ(ierr);
  /*  ierr = TSSetIJacobian(ts,A,A,(TSIJacobian)IJacobian,&user);CHKERRQ(ierr); */
  ierr = TSSetApplicationContext(ts,&user);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetSolution(ts,U);CHKERRQ(ierr);

  /* Save initial solution */
  idx=3*user.stepnum;

  ierr = MatDenseGetArray(user.Sol,&mat);CHKERRQ(ierr);
  ierr = VecGetArrayRead(U,&x);CHKERRQ(ierr);

  mat[idx] = 0.0;

  ierr = PetscArraycpy(mat+idx+1,x,2);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(user.Sol,&mat);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(U,&x);CHKERRQ(ierr);
  user.stepnum++;


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set solver options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetMaxTime(ts,20.0);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,.01);CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSSetPostStep(ts,SaveSolution);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,U);CHKERRQ(ierr);

  ierr = MatCreateSeqDense(PETSC_COMM_SELF,3,user.stepnum,NULL,&B);CHKERRQ(ierr);
  ierr = MatDenseGetArrayRead(user.Sol,&rmat);CHKERRQ(ierr);
  ierr = MatDenseGetArray(B,&amat);CHKERRQ(ierr);
  ierr = PetscArraycpy(amat,rmat,user.stepnum*3);CHKERRQ(ierr);
  ierr = MatDenseRestoreArray(B,&amat);CHKERRQ(ierr);
  ierr = MatDenseRestoreArrayRead(user.Sol,&rmat);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,"out.bin",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(B,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = MatDestroy(&user.Sol);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&user.wind_data);CHKERRQ(ierr);
  ierr = VecDestroy(&user.t_wind);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   build:
      requires: !complex

   test:


TEST*/
