/**********************************************************************
    American Put Options Pricing using the Black-Scholes Equation

   Background (European Options):
     The standard European option is a contract where the holder has the right
     to either buy (call option) or sell (put option) an underlying asset at
     a designated future time and price.

     The classic Black-Scholes model begins with an assumption that the
     price of the underlying asset behaves as a lognormal random walk.
     Using this assumption and a no-arbitrage argument, the following
     linear parabolic partial differential equation for the value of the
     option results:

       dV/dt + 0.5(sigma**2)(S**alpha)(d2V/dS2) + (r - D)S(dV/dS) - rV = 0.

     Here, sigma is the volatility of the underling asset, alpha is a
     measure of elasticity (typically two), D measures the dividend payments
     on the underling asset, and r is the interest rate.

     To completely specify the problem, we need to impose some boundary
     conditions.  These are as follows:

       V(S, T) = max(E - S, 0)
       V(0, t) = E for all 0 <= t <= T
       V(s, t) = 0 for all 0 <= t <= T and s->infinity

     where T is the exercise time time and E the strike price (price paid
     for the contract).

     An explicit formula for the value of an European option can be
     found.  See the references for examples.

   Background (American Options):
     The American option is similar to its European counterpart.  The
     difference is that the holder of the American option can excercise
     their right to buy or sell the asset at any time prior to the
     expiration.  This additional ability introduce a free boundary into
     the Black-Scholes equation which can be modeled as a linear
     complementarity problem.

       0 <= -(dV/dt + 0.5(sigma**2)(S**alpha)(d2V/dS2) + (r - D)S(dV/dS) - rV)
         complements
       V(S,T) >= max(E-S,0)

     where the variables are the same as before and we have the same boundary
     conditions.

     There is not explicit formula for calculating the value of an American
     option.  Therefore, we discretize the above problem and solve the
     resulting linear complementarity problem.

     We will use backward differences for the time variables and central
     differences for the space variables.  Crank-Nicholson averaging will
     also be used in the discretization.  The algorithm used by the code
     solves for V(S,t) for a fixed t and then uses this value in the
     calculation of V(S,t - dt).  The method stops when V(S,0) has been
     found.

   References:
     Huang and Pang, "Options Pricing and Linear Complementarity,"
       Journal of Computational Finance, volume 2, number 3, 1998.
     Wilmott, "Derivatives: The Theory and Practice of Financial Engineering,"
       John Wiley and Sons, New York, 1998.
***************************************************************************/

/*
  Include "petsctao.h" so we can use TAO solvers.
  Include "petscdmda.h" so that we can use distributed meshes (DMs) for managing
  the parallel mesh.
*/

#include <petscdmda.h>
#include <petsctao.h>

static char  help[] =
"This example demonstrates use of the TAO package to\n\
solve a linear complementarity problem for pricing American put options.\n\
The code uses backward differences in time and central differences in\n\
space.  The command line options are:\n\
  -rate <r>, where <r> = interest rate\n\
  -sigma <s>, where <s> = volatility of the underlying\n\
  -alpha <a>, where <a> = elasticity of the underlying\n\
  -delta <d>, where <d> = dividend rate\n\
  -strike <e>, where <e> = strike price\n\
  -expiry <t>, where <t> = the expiration date\n\
  -mt <tg>, where <tg> = number of grid points in time\n\
  -ms <sg>, where <sg> = number of grid points in space\n\
  -es <se>, where <se> = ending point of the space discretization\n\n";

/*T
   Concepts: TAO^Solving a complementarity problem
   Routines: TaoCreate(); TaoDestroy();
   Routines: TaoSetJacobianRoutine(); TaoAppSetConstraintRoutine();
   Routines: TaoSetFromOptions();
   Routines: TaoSolveApplication();
   Routines: TaoSetVariableBoundsRoutine(); TaoSetInitialSolutionVec();
   Processors: 1
T*/

/*
  User-defined application context - contains data needed by the
  application-provided call-back routines, FormFunction(), and FormJacobian().
*/

typedef struct {
  PetscReal *Vt1;                /* Value of the option at time T + dt */
  PetscReal *c;                  /* Constant -- (r - D)S */
  PetscReal *d;                  /* Constant -- -0.5(sigma**2)(S**alpha) */

  PetscReal rate;                /* Interest rate */
  PetscReal sigma, alpha, delta; /* Underlying asset properties */
  PetscReal strike, expiry;      /* Option contract properties */

  PetscReal es;                  /* Finite value used for maximum asset value */
  PetscReal ds, dt;              /* Discretization properties */
  PetscInt  ms, mt;               /* Number of elements */

  DM        dm;
} AppCtx;

/* -------- User-defined Routines --------- */

PetscErrorCode FormConstraints(Tao, Vec, Vec, void *);
PetscErrorCode FormJacobian(Tao, Vec, Mat, Mat, void *);
PetscErrorCode ComputeVariableBounds(Tao, Vec, Vec, void*);

int main(int argc, char **argv)
{
  PetscErrorCode ierr;    /* used to check for functions returning nonzeros */
  Vec            x;             /* solution vector */
  Vec            c;             /* Constraints function vector */
  Mat            J;                  /* jacobian matrix */
  PetscBool      flg;         /* A return variable when checking for user options */
  Tao            tao;          /* Tao solver context */
  AppCtx         user;            /* user-defined work context */
  PetscInt       i, j;
  PetscInt       xs,xm,gxs,gxm;
  PetscReal      sval = 0;
  PetscReal      *x_array;
  Vec            localX;

  /* Initialize PETSc, TAO */
  ierr = PetscInitialize(&argc, &argv, (char *)0, help);if (ierr) return ierr;

  /*
     Initialize the user-defined application context with reasonable
     values for the American option to price
  */
  user.rate = 0.04;
  user.sigma = 0.40;
  user.alpha = 2.00;
  user.delta = 0.01;
  user.strike = 10.0;
  user.expiry = 1.0;
  user.mt = 10;
  user.ms = 150;
  user.es = 100.0;

  /* Read in alternative values for the American option to price */
  ierr = PetscOptionsGetReal(NULL,NULL, "-alpha", &user.alpha, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL, "-delta", &user.delta, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL, "-es", &user.es, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL, "-expiry", &user.expiry, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL, "-ms", &user.ms, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL, "-mt", &user.mt, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL, "-rate", &user.rate, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL, "-sigma", &user.sigma, &flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL, "-strike", &user.strike, &flg);CHKERRQ(ierr);

  /* Check that the options set are allowable (needs to be done) */

  user.ms++;
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,user.ms,1,1,NULL,&user.dm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(user.dm);CHKERRQ(ierr);
  ierr = DMSetUp(user.dm);CHKERRQ(ierr);
  /* Create appropriate vectors and matrices */

  ierr = DMDAGetCorners(user.dm,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(user.dm,&gxs,NULL,NULL,&gxm,NULL,NULL);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(user.dm,&x);CHKERRQ(ierr);
  /*
     Finish filling in the user-defined context with the values for
     dS, dt, and allocating space for the constants
  */
  user.ds = user.es / (user.ms-1);
  user.dt = user.expiry / user.mt;

  ierr = PetscMalloc1(gxm,&(user.Vt1));CHKERRQ(ierr);
  ierr = PetscMalloc1(gxm,&(user.c));CHKERRQ(ierr);
  ierr = PetscMalloc1(gxm,&(user.d));CHKERRQ(ierr);

  /*
     Calculate the values for the constant.  Vt1 begins with the ending
     boundary condition.
  */
  for (i=0; i<gxm; i++) {
    sval = (gxs+i)*user.ds;
    user.Vt1[i] = PetscMax(user.strike - sval, 0);
    user.c[i] = (user.delta - user.rate)*sval;
    user.d[i] = -0.5*user.sigma*user.sigma*PetscPowReal(sval, user.alpha);
  }
  if (gxs+gxm==user.ms) {
    user.Vt1[gxm-1] = 0;
  }
  ierr = VecDuplicate(x, &c);CHKERRQ(ierr);

  /*
     Allocate the matrix used by TAO for the Jacobian.  Each row of
     the Jacobian matrix will have at most three elements.
  */
  ierr = DMCreateMatrix(user.dm,&J);CHKERRQ(ierr);

  /* The TAO code begins here */

  /* Create TAO solver and set desired solution method  */
  ierr = TaoCreate(PETSC_COMM_WORLD, &tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOSSILS);CHKERRQ(ierr);

  /* Set routines for constraints function and Jacobian evaluation */
  ierr = TaoSetConstraintsRoutine(tao, c, FormConstraints, (void *)&user);CHKERRQ(ierr);
  ierr = TaoSetJacobianRoutine(tao, J, J, FormJacobian, (void *)&user);CHKERRQ(ierr);

  /* Set the variable bounds */
  ierr = TaoSetVariableBoundsRoutine(tao,ComputeVariableBounds,(void*)&user);CHKERRQ(ierr);

  /* Set initial solution guess */
  ierr = VecGetArray(x,&x_array);CHKERRQ(ierr);
  for (i=0; i< xm; i++)
    x_array[i] = user.Vt1[i-gxs+xs];
  ierr = VecRestoreArray(x,&x_array);CHKERRQ(ierr);
  /* Set data structure */
  ierr = TaoSetInitialVector(tao, x);CHKERRQ(ierr);

  /* Set routines for function and Jacobian evaluation */
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);

  /* Iteratively solve the linear complementarity problems  */
  for (i = 1; i < user.mt; i++) {

    /* Solve the current version */
    ierr = TaoSolve(tao);CHKERRQ(ierr);

    /* Update Vt1 with the solution */
    ierr = DMGetLocalVector(user.dm,&localX);CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(user.dm,x,INSERT_VALUES,localX);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(user.dm,x,INSERT_VALUES,localX);CHKERRQ(ierr);
    ierr = VecGetArray(localX,&x_array);CHKERRQ(ierr);
    for (j = 0; j < gxm; j++) {
      user.Vt1[j] = x_array[j];
    }
    ierr = VecRestoreArray(x,&x_array);CHKERRQ(ierr);
    ierr = DMRestoreLocalVector(user.dm,&localX);CHKERRQ(ierr);
  }

  /* Free TAO data structures */
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);

  /* Free PETSc data structures */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&c);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  /* Free user-defined workspace */
  ierr = PetscFree(user.Vt1);CHKERRQ(ierr);
  ierr = PetscFree(user.c);CHKERRQ(ierr);
  ierr = PetscFree(user.d);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/* -------------------------------------------------------------------- */
PetscErrorCode ComputeVariableBounds(Tao tao, Vec xl, Vec xu, void*ctx)
{
  AppCtx         *user = (AppCtx *) ctx;
  PetscErrorCode ierr;
  PetscInt       i;
  PetscInt       xs,xm;
  PetscInt       ms = user->ms;
  PetscReal      sval=0.0,*xl_array,ub= PETSC_INFINITY;

  /* Set the variable bounds */
  ierr = VecSet(xu, ub);CHKERRQ(ierr);
  ierr = DMDAGetCorners(user->dm,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  ierr = VecGetArray(xl,&xl_array);CHKERRQ(ierr);
  for (i = 0; i < xm; i++) {
    sval = (xs+i)*user->ds;
    xl_array[i] = PetscMax(user->strike - sval, 0);
  }
  ierr = VecRestoreArray(xl,&xl_array);CHKERRQ(ierr);

  if (xs==0) {
    ierr = VecGetArray(xu,&xl_array);CHKERRQ(ierr);
    xl_array[0] = PetscMax(user->strike, 0);
    ierr = VecRestoreArray(xu,&xl_array);CHKERRQ(ierr);
  }
  if (xs+xm==ms) {
    ierr = VecGetArray(xu,&xl_array);CHKERRQ(ierr);
    xl_array[xm-1] = 0;
    ierr = VecRestoreArray(xu,&xl_array);CHKERRQ(ierr);
  }

  return 0;
}
/* -------------------------------------------------------------------- */

/*
    FormFunction - Evaluates gradient of f.

    Input Parameters:
.   tao  - the Tao context
.   X    - input vector
.   ptr  - optional user-defined context, as set by TaoAppSetConstraintRoutine()

    Output Parameters:
.   F - vector containing the newly evaluated gradient
*/
PetscErrorCode FormConstraints(Tao tao, Vec X, Vec F, void *ptr)
{
  AppCtx         *user = (AppCtx *) ptr;
  PetscReal      *x, *f;
  PetscReal      *Vt1 = user->Vt1, *c = user->c, *d = user->d;
  PetscReal      rate = user->rate;
  PetscReal      dt = user->dt, ds = user->ds;
  PetscInt       ms = user->ms;
  PetscErrorCode ierr;
  PetscInt       i, xs,xm,gxs,gxm;
  Vec            localX,localF;
  PetscReal      zero=0.0;

  ierr = DMGetLocalVector(user->dm,&localX);CHKERRQ(ierr);
  ierr = DMGetLocalVector(user->dm,&localF);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(user->dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(user->dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMDAGetCorners(user->dm,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(user->dm,&gxs,NULL,NULL,&gxm,NULL,NULL);CHKERRQ(ierr);
  ierr = VecSet(F, zero);CHKERRQ(ierr);
  /*
     The problem size is smaller than the discretization because of the
     two fixed elements (V(0,T) = E and V(Send,T) = 0.
  */

  /* Get pointers to the vector data */
  ierr = VecGetArray(localX, &x);CHKERRQ(ierr);
  ierr = VecGetArray(localF, &f);CHKERRQ(ierr);

  /* Left Boundary */
  if (gxs==0) {
    f[0] = x[0]-user->strike;
  } else {
    f[0] = 0;
  }

  /* All points in the interior */
  /*  for (i=gxs+1;i<gxm-1;i++) { */
  for (i=1;i<gxm-1;i++) {
    f[i] = (1.0/dt + rate)*x[i] - Vt1[i]/dt + (c[i]/(4*ds))*(x[i+1] - x[i-1] + Vt1[i+1] - Vt1[i-1]) +
           (d[i]/(2*ds*ds))*(x[i+1] -2*x[i] + x[i-1] + Vt1[i+1] - 2*Vt1[i] + Vt1[i-1]);
  }

  /* Right boundary */
  if (gxs+gxm==ms) {
    f[xm-1]=x[gxm-1];
  } else {
    f[xm-1]=0;
  }

  /* Restore vectors */
  ierr = VecRestoreArray(localX, &x);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF, &f);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(user->dm,localF,INSERT_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(user->dm,localF,INSERT_VALUES,F);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->dm,&localX);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(user->dm,&localF);CHKERRQ(ierr);
  ierr = PetscLogFlops(24.0*(gxm-2));CHKERRQ(ierr);
  /*
  info=VecView(F,PETSC_VIEWER_STDOUT_WORLD);
  */
  return 0;
}

/* ------------------------------------------------------------------- */
/*
   FormJacobian - Evaluates Jacobian matrix.

   Input Parameters:
.  tao  - the Tao context
.  X    - input vector
.  ptr  - optional user-defined context, as set by TaoSetJacobian()

   Output Parameters:
.  J    - Jacobian matrix
*/
PetscErrorCode FormJacobian(Tao tao, Vec X, Mat J, Mat tJPre, void *ptr)
{
  AppCtx         *user = (AppCtx *) ptr;
  PetscReal      *c = user->c, *d = user->d;
  PetscReal      rate = user->rate;
  PetscReal      dt = user->dt, ds = user->ds;
  PetscInt       ms = user->ms;
  PetscReal      val[3];
  PetscErrorCode ierr;
  PetscInt       col[3];
  PetscInt       i;
  PetscInt       gxs,gxm;
  PetscBool      assembled;

  /* Set various matrix options */
  ierr = MatSetOption(J,MAT_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatAssembled(J,&assembled);CHKERRQ(ierr);
  if (assembled) {ierr = MatZeroEntries(J);CHKERRQ(ierr);}

  ierr = DMDAGetGhostCorners(user->dm,&gxs,NULL,NULL,&gxm,NULL,NULL);CHKERRQ(ierr);

  if (gxs==0) {
    i = 0;
    col[0] = 0;
    val[0]=1.0;
    ierr = MatSetValues(J,1,&i,1,col,val,INSERT_VALUES);CHKERRQ(ierr);
  }
  for (i=1; i < gxm-1; i++) {
    col[0] = gxs + i - 1;
    col[1] = gxs + i;
    col[2] = gxs + i + 1;
    val[0] = -c[i]/(4*ds) + d[i]/(2*ds*ds);
    val[1] = 1.0/dt + rate - d[i]/(ds*ds);
    val[2] =  c[i]/(4*ds) + d[i]/(2*ds*ds);
    ierr = MatSetValues(J,1,&col[1],3,col,val,INSERT_VALUES);CHKERRQ(ierr);
  }
  if (gxs+gxm==ms) {
    i = ms-1;
    col[0] = i;
    val[0]=1.0;
    ierr = MatSetValues(J,1,&i,1,col,val,INSERT_VALUES);CHKERRQ(ierr);
  }

  /* Assemble the Jacobian matrix */
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscLogFlops(18.0*(gxm)+5);CHKERRQ(ierr);
  return 0;
}

/*TEST

   build:
      requires: !complex

   test:
      args: -tao_monitor -tao_type ssils -tao_gttol 1.e-5
      requires: !single

   test:
      suffix: 2
      args: -tao_monitor -tao_type ssfls -tao_max_it 10 -tao_gttol 1.e-5
      requires: !single

   test:
      suffix: 3
      args: -tao_monitor -tao_type asils -tao_subset_type subvec -tao_gttol 1.e-5
      requires: !single

   test:
      suffix: 4
      args: -tao_monitor -tao_type asils -tao_subset_type mask -tao_gttol 1.e-5
      requires: !single

   test:
      suffix: 5
      args: -tao_monitor -tao_type asils -tao_subset_type matrixfree -pc_type jacobi -tao_max_it 6 -tao_gttol 1.e-5
      requires: !single

   test:
      suffix: 6
      args: -tao_monitor -tao_type asfls -tao_subset_type subvec -tao_max_it 10 -tao_gttol 1.e-5
      requires: !single

   test:
      suffix: 7
      args: -tao_monitor -tao_type asfls -tao_subset_type mask -tao_max_it 10 -tao_gttol 1.e-5
      requires: !single

TEST*/
