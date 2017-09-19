
static char help[] ="Solves a simple data assimulation problem with the heat equation using TSAdjoint\n\n";

/*

    Not yet tested in parallel

*/
/*
   Concepts: TS^time-dependent linear problems
   Concepts: TS^heat equation
   Concepts: TS^diffusion equation
   Concepts: adjoints
   Processors: n
*/

/* ------------------------------------------------------------------------

   This program uses the one-dimensional heat equation (also called the
   diffusion equation),
       u_t = u_xx,
   on the domain 0 <= x <= 1, with periodic boundary conditions

   to demonstrate solving a data assimulation problem of finding the initial conditions
   to produce a given solution at a fixed time.

   We discretize the right-hand side using the spectral element method

  ------------------------------------------------------------------------- */

/*
   Include "petscts.h" so that we can use TS solvers.  Note that this file
   automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h  - vectors
     petscmat.h  - matrices
     petscis.h     - index sets            petscksp.h  - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h   - preconditioners
     petscksp.h   - linear solvers        petscsnes.h - nonlinear solvers
*/

#include <petsctao.h>
#include <petscts.h>
#include <petscgll.h>
#include <petscdraw.h>
#include <petscdmda.h>

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines.
*/

typedef struct {
  PetscInt    N;             /* grid points per elements*/
  PetscInt    E;              /* number of elements */
  PetscReal   tol_L2,tol_max; /* error norms */
  PetscInt    steps;          /* number of timesteps */
  PetscReal   Tend;           /* endtime */
  PetscReal   mu;             /* viscosity */
  PetscReal   dt;             /* timestep*/
  PetscReal   L;              /* total length of domain */   
  PetscReal   Le; 
  PetscReal   Tadj;
} PetscParam;

typedef struct {
  Vec         obj;               /* desired end state */
  Vec         grid;              /* total grid */   
  Vec         grad;
  Vec         ic;
  Vec         curr_sol;
  PetscReal   *Z;                 /* mesh grid */
  PetscScalar *W;                 /* weights */
} PetscData;

typedef struct {
  Vec         grid;              /* total grid */   
  Vec         mass;              /* mass matrix for total integration */
  Mat         stiff;             // stifness matrix
  PetscGLL    gll;
} PetscSEMOperators;

typedef struct {
  DM                da;                /* distributed array data structure */
  PetscSEMOperators SEMop;
  PetscParam        param;
  PetscData         dat;
  PetscBool         debug;
  TS                ts;
  PetscReal         initial_dt;
} AppCtx;

/*
   User-defined routines
*/
extern PetscErrorCode InitialConditions(Vec,AppCtx*);
extern PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal*,Vec,void*);
extern PetscErrorCode RHSMatrixHeatgllDM(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode RHSAdjointgllDM(TS,PetscReal,Vec,Mat,Mat,void*);
extern PetscErrorCode RHSFunctionHeat(TS,PetscReal,Vec,Vec,void*);
extern PetscErrorCode Objective(PetscReal,Vec,AppCtx*);

int main(int argc,char **argv)
{
  AppCtx         appctx;                 /* user-defined application context */
  Tao            tao;
  Vec            u;                      /* approximate solution vector */
  PetscErrorCode ierr;
  PetscInt       i, xs, xm, ind, j, lenglob;
  PetscReal      x, *wrk_ptr1, *wrk_ptr2;
  MatNullSpace   nsp;

   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program and set problem parameters
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscFunctionBegin;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  /*initialize parameters */
  appctx.param.N    = 10;  /* order of the spectral element */
  appctx.param.E    = 8;  /* number of elements */
  appctx.param.L    = 1.0;  /* length of the domain */
  appctx.param.mu   = 0.001; /* diffusion coefficient */
  appctx.initial_dt = 1e-4;
  appctx.param.steps = PETSC_MAX_INT;
  appctx.param.Tend  = 0.01;

  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&appctx.param.N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-E",&appctx.param.E,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-debug",&appctx.debug);CHKERRQ(ierr);
  appctx.param.Le = appctx.param.L/appctx.param.E;


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = PetscGLLCreate(appctx.param.N,PETSCGLL_VIA_LINEARALGEBRA,&appctx.SEMop.gll);CHKERRQ(ierr);
  
  ierr = PetscMalloc1(appctx.param.N, &appctx.dat.Z);
  ierr = PetscMalloc1(appctx.param.N, &appctx.dat.W);

  for(i=0; i<appctx.param.N; i++) { 
    appctx.dat.Z[i]=(appctx.SEMop.gll.nodes[i]+1.0);
    appctx.dat.W[i]=appctx.SEMop.gll.weights[i]; 
  }


  //lenloc   = appctx.param.E*appctx.param.N; //only if I want to do it totally local for explicit
  lenglob  = appctx.param.E*(appctx.param.N-1);

  /*
     Create distributed array (DMDA) to manage parallel grid and vectors
     and to set up the ghost point communication pattern.  There are E*(Nl-1)+1
     total grid values spread equally among all the processors, except first and last
  */

  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,lenglob,1,1,NULL,&appctx.da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(appctx.da);CHKERRQ(ierr);
  ierr = DMSetUp(appctx.da);CHKERRQ(ierr);
  //ierr = DMDAGetInfo(appctx.da,NULL,&E,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
 
  /*
     Extract global and local vectors from DMDA; we use these to store the
     approximate solution.  Then duplicate these for remaining vectors that
     have the same types.
  */

  ierr = DMCreateGlobalVector(appctx.da,&u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.dat.ic);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.dat.obj);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.dat.grad);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.SEMop.grid);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.SEMop.mass);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&appctx.dat.curr_sol);CHKERRQ(ierr);
 
  ierr = DMDAGetCorners(appctx.da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  ierr = DMDAVecGetArray(appctx.da,appctx.SEMop.grid,&wrk_ptr1);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(appctx.da,appctx.SEMop.mass,&wrk_ptr2);CHKERRQ(ierr);
  
  //Compute function over the locally owned part of the grid
  
    xs=xs/(appctx.param.N-1);
    xm=xm/(appctx.param.N-1);
  
  /* 
     Build total grid and mass over entire mesh (multi-elemental) 
  */ 

  for (i=xs; i<xs+xm; i++) {
    for (j=0; j<appctx.param.N-1; j++) {
      x = (appctx.param.Le/2.0)*(appctx.dat.Z[j])+appctx.param.Le*i; 
      ind=i*(appctx.param.N-1)+j;
      wrk_ptr1[ind]=x;
      wrk_ptr2[ind]=.5*appctx.param.Le*appctx.dat.W[j];
      if (j==0) wrk_ptr2[ind]+=.5*appctx.param.Le*appctx.dat.W[j];
    } 
  }
  ierr = DMDAVecRestoreArray(appctx.da,appctx.SEMop.grid,&wrk_ptr1);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(appctx.da,appctx.SEMop.mass,&wrk_ptr2);CHKERRQ(ierr);


  //Set Objective and Initial conditions for the problem 
  ierr = Objective(appctx.param.Tend+2,appctx.dat.obj,&appctx);CHKERRQ(ierr);
  ierr = InitialConditions(appctx.dat.ic,&appctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Create matrix data structure; set matrix evaluation routine.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMSetMatrixPreallocateOnly(appctx.da, PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMCreateMatrix(appctx.da,&appctx.SEMop.stiff);CHKERRQ(ierr);

  /*
   For linear problems with a time-dependent f(u,t) in the equation
   u_t = f(u,t), the user provides the discretized right-hand-side
   as a time-dependent matrix.
   */
  ierr = RHSMatrixHeatgllDM(appctx.ts,0.0,u,appctx.SEMop.stiff,appctx.SEMop.stiff,&appctx);CHKERRQ(ierr);

  /* attach the null space to the matrix, this probably is not needed but does no harm */
  ierr = MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_TRUE,0,NULL,&nsp);CHKERRQ(ierr);
  ierr = MatSetNullSpace(appctx.SEMop.stiff,nsp);CHKERRQ(ierr);
  ierr = MatNullSpaceTest(nsp,appctx.SEMop.stiff,NULL);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nsp);CHKERRQ(ierr);

  // Create TAO solver and set desired solution method 
  ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOBLMVM);CHKERRQ(ierr);
  ierr = TaoSetInitialVector(tao,appctx.dat.ic);CHKERRQ(ierr);

  // Set routine for function and gradient evaluation 
  ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,(void *)&appctx);CHKERRQ(ierr);

  // Check for any TAO command line options 
  ierr = TaoSetTolerances(tao,1e-8,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);

  ierr = TaoSolve(tao); CHKERRQ(ierr);

  ierr = TaoDestroy(&tao);CHKERRQ(ierr);
  ierr = MatDestroy(&appctx.SEMop.stiff);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.ic);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.obj);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.grad);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.SEMop.grid);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.SEMop.mass);CHKERRQ(ierr);
  ierr = VecDestroy(&appctx.dat.curr_sol);CHKERRQ(ierr);
  ierr = PetscGLLDestroy(&appctx.SEMop.gll);CHKERRQ(ierr);
  ierr = DMDestroy(&appctx.da);CHKERRQ(ierr);
  ierr = PetscFree(appctx.dat.Z);
  ierr = PetscFree(appctx.dat.W);

  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary).
  */
    ierr = PetscFinalize();
    return ierr;
}
/* --------------------------------------------------------------------- */
/*
   InitialConditions - Computes the solution at the initial time.

   Input Parameter:
   u - uninitialized solution vector (global)
   appctx - user-defined application context

   Output Parameter:
   u - vector with solution at initial time (global)
*/
PetscErrorCode InitialConditions(Vec u,AppCtx *appctx)
{
  PetscScalar    *s_localptr, *xg_localptr;
  PetscErrorCode ierr;
  PetscInt       i,lenglob;
  PetscReal      sum;

  /*
     Get pointers to vector data
  */
  ierr = DMDAVecGetArray(appctx->da,u,&s_localptr);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(appctx->da,appctx->SEMop.grid,&xg_localptr);CHKERRQ(ierr);

  lenglob  = appctx->param.E*(appctx->param.N-1);

  /*      for (i=0; i<lenglob; i++) {
        s_localptr[i]= PetscSinScalar(2.0*PETSC_PI*xg_localptr[i]);
   }  */

  for (i=0; i<lenglob; i++) {
    s_localptr[i]=PetscSinScalar(2.0*PETSC_PI*xg_localptr[i])+PetscCosScalar(4.0*PETSC_PI*xg_localptr[i])+3.0*PetscSinScalar(2.0*PETSC_PI*xg_localptr[i])*PetscCosScalar(6.0*PETSC_PI*xg_localptr[i]);
  }

  ierr = DMDAVecRestoreArray(appctx->da,appctx->SEMop.grid,&xg_localptr);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(appctx->da,appctx->dat.ic,&s_localptr);CHKERRQ(ierr);
  /* make sure initial conditions do not contain the constant functions */
  ierr = VecSum(appctx->dat.ic,&sum);CHKERRQ(ierr);
  ierr = VecShift(appctx->dat.ic,-sum/lenglob);CHKERRQ(ierr);
  return 0;
}
/* --------------------------------------------------------------------- */
/*
   Sets the profile at end time

   Input Parameters:
   t - current time
   obj - vector storing the end function
   appctx - user-defined application context

   Output Parameter:
   solution - vector with the newly computed exact solution
*/
PetscErrorCode Objective(PetscReal t,Vec obj,AppCtx *appctx)
{
  PetscScalar    *s_localptr,*xg_localptr,tc = -.001*4*PETSC_PI*PETSC_PI*t;
  PetscErrorCode ierr;
  PetscInt       i, lenglob;
  
  /*
     Simply write the solution directly into the array locations.
     Alternatively, we culd use VecSetValues() or VecSetValuesLocal().
  */
  
  ierr = DMDAVecGetArray(appctx->da,obj,&s_localptr);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(appctx->da,appctx->SEMop.grid,&xg_localptr);CHKERRQ(ierr);

  lenglob  = appctx->param.E*(appctx->param.N-1);
  
    for (i=0; i<lenglob; i++) {
      s_localptr[i]=PetscSinScalar(2.0*PETSC_PI*xg_localptr[i])*PetscExpScalar(tc);
   }
  
    /*for (i=0; i<lenglob; i++) {
      s_localptr[i]=1.0;
     }*/


/*
     Restore vectors
*/
  ierr = DMDAVecRestoreArray(appctx->da,appctx->SEMop.grid,&xg_localptr);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(appctx->da,appctx->dat.obj,&s_localptr);CHKERRQ(ierr);

  return 0;
}


/* --------------------------------------------------------------------- */

/*
   RHSMatrixHeat - User-provided routine to compute the right-hand-side
   matrix for the heat equation.

   Input Parameters:
   ts - the TS context
   t - current time
   global_in - global input vector
   dummy - optional user-defined context, as set by TSetRHSJacobian()

   Output Parameters:
   AA - Jacobian matrix
   BB - optionally different preconditioning matrix
   str - flag indicating matrix structure

*/
PetscErrorCode RHSMatrixHeatgllDM(TS ts,PetscReal t,Vec X,Mat A,Mat BB,void *ctx)
{
  PetscReal      **temp;
  PetscReal      vv;
  AppCtx         *appctx = (AppCtx*)ctx;     /* user-defined application context */
  PetscErrorCode ierr;
  PetscInt       i,xs,xn,l,j;
  PetscInt       *rowsDM;

  /*
   Creates the element stiffness matrix for the given gll
   */
  ierr = PetscGLLElementLaplacianCreate(&appctx->SEMop.gll,&temp);CHKERRQ(ierr);

  // scale by the size of the element
  for (i=0; i<appctx->param.N; i++) {
    vv=-appctx->param.mu*2.0/appctx->param.Le;
    for (j=0; j<appctx->param.N; j++) temp[i][j]=temp[i][j]*vv;
  }

  ierr = MatSetOption(A,MAT_NEW_NONZERO_ALLOCATION_ERR,PETSC_FALSE);CHKERRQ(ierr);
  ierr = DMDAGetCorners(appctx->da,&xs,NULL,NULL,&xn,NULL,NULL);CHKERRQ(ierr);

  xs   = xs/(appctx->param.N-1);
  xn   = xn/(appctx->param.N-1);

  ierr = PetscMalloc1(appctx->param.N,&rowsDM);CHKERRQ(ierr);
  /*
   loop over local elements
   */
  for (j=xs; j<xs+xn; j++) {
    for (l=0; l<appctx->param.N; l++) {
      rowsDM[l] = 1+(j-xs)*(appctx->param.N-1)+l;
    }
    ierr = MatSetValuesLocal(A,appctx->param.N,rowsDM,appctx->param.N,rowsDM,&temp[0][0],ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = PetscFree(rowsDM);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecReciprocal(appctx->SEMop.mass);
  ierr = MatDiagonalScale(A,appctx->SEMop.mass,0);
  ierr = VecReciprocal(appctx->SEMop.mass);

  ierr = PetscGLLElementLaplacianDestroy(&appctx->SEMop.gll,&temp);CHKERRQ(ierr);
  return 0;
}

/* ------------------------------------------------------------------ */
/*
   FormFunctionGradient - Evaluates the function and corresponding gradient.

   Input Parameters:
   tao - the Tao context
   X   - the input vector
   ptr - optional user-defined context, as set by TaoSetObjectiveAndGradientRoutine()

   Output Parameters:
   f   - the newly evaluated function
   G   - the newly evaluated gradient
*/
PetscErrorCode FormFunctionGradient(Tao tao,Vec IC,PetscReal *f,Vec G,void *ctx)
{
  AppCtx           *appctx = (AppCtx*)ctx;     /* user-defined application context */
  PetscErrorCode    ierr;
  Vec               temp, temp2;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Create timestepping solver context
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = TSCreate(PETSC_COMM_WORLD,&appctx->ts);CHKERRQ(ierr);
  ierr = TSSetProblemType(appctx->ts,TS_LINEAR);CHKERRQ(ierr);
  ierr = TSSetType(appctx->ts,TSRK);CHKERRQ(ierr);
  ierr = TSSetDM(appctx->ts,appctx->da);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
   Set time
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetTime(appctx->ts,0.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(appctx->ts,appctx->initial_dt);CHKERRQ(ierr);
  ierr = TSSetMaxSteps(appctx->ts,appctx->param.steps);CHKERRQ(ierr);
  ierr = TSSetMaxTime(appctx->ts,appctx->param.Tend);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(appctx->ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

  ierr = TSSetTolerances(appctx->ts,1e-7,NULL,1e-7,NULL);CHKERRQ(ierr);
  ierr = TSSetFromOptions(appctx->ts);
  /* Need to save initial timestep user may have set with -ts_dt so it can be reset for each new TSSolve() */
  ierr = TSGetTimeStep(appctx->ts,&appctx->initial_dt);CHKERRQ(ierr);

  ierr = TSSetTime(appctx->ts,0.0);CHKERRQ(ierr);
  ierr = TSSetTimeStep(appctx->ts,appctx->initial_dt);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(appctx->ts,NULL,TSComputeRHSFunctionLinear,&appctx);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(appctx->ts,appctx->SEMop.stiff,appctx->SEMop.stiff,TSComputeRHSJacobianConstant,&appctx);CHKERRQ(ierr);
  ierr = VecCopy(IC,appctx->dat.curr_sol);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Save trajectory of solution so that TSAdjointSolve() may be used
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
   ierr = TSSetSaveTrajectory(appctx->ts);CHKERRQ(ierr);

  ierr = TSSolve(appctx->ts,appctx->dat.curr_sol);CHKERRQ(ierr);

  /*
     Compute the L2-norm of the objective function, cost function is f
  */
  ierr = VecDuplicate(appctx->dat.obj,&temp);CHKERRQ(ierr);
  ierr = VecCopy(appctx->dat.obj,temp);CHKERRQ(ierr);
  ierr = VecAXPY(temp,-1.0,appctx->dat.curr_sol);CHKERRQ(ierr);

  ierr = VecDuplicate(temp,&temp2);CHKERRQ(ierr);
  ierr = VecPointwiseMult(temp2,temp,temp);CHKERRQ(ierr);
  ierr = VecDot(temp2,appctx->SEMop.mass,f);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Adjoint model starts here
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
   Initial conditions for the adjoint integration
   */

  ierr = VecScale(temp, -2.0);
  ierr = VecPointwiseMult(temp,temp,appctx->SEMop.mass);CHKERRQ(ierr);
  ierr = VecCopy(temp,appctx->dat.grad);CHKERRQ(ierr);
  ierr = TSSetCostGradients(appctx->ts,1,&appctx->dat.grad,NULL);CHKERRQ(ierr);
  ierr = TSAdjointSolve(appctx->ts);CHKERRQ(ierr);
  ierr = VecCopy(appctx->dat.grad,G);CHKERRQ(ierr);
  ierr = VecDestroy(&temp);CHKERRQ(ierr);
  ierr = VecDestroy(&temp2);CHKERRQ(ierr);
  ierr = TSDestroy(&appctx->ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*TEST

  build:
      requires: !complex

   test:
     requires: !single
     args: -tao_monitor  -ts_adapt_dt_max 3.e-3 

   test:
     suffix: cn
     requires: !single
     args: -tao_monitor -ts_type cn -ts_dt .003 -pc_type lu

TEST*/
