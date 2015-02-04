#define c11 1.0
#define c12 0
#define c21 2.0
#define c22 1.0
static char help[] = "Solves the van der Pol equation.\n\
Input parameters include:\n";

/*
   Concepts: TS^time-dependent nonlinear problems
   Concepts: TS^van der Pol equation DAE equivalent
   Concepts: Optimization using adjoint sensitivity analysis
   Processors: 1
*/
/* ------------------------------------------------------------------------

   This program solves the van der Pol DAE ODE equivalent
       y' = z                 (1)
       z' = mu[(1-y^2)z-y]
   on the domain 0 <= x <= 1, with the boundary conditions
       y(0) = 2, y'(0) = -6.666665432100101e-01,
   and
       mu = 10^6.
   This is a nonlinear equation.

   Notes:
   This code demonstrates the TS solver interface to a variant of
   linear problems, u_t = f(u,t), namely turning (1) into a system of
   first order differential equations,

   [ y' ] = [          z          ]
   [ z' ]   [     mu[(1-y^2)z-y]  ]

   which then we can write as a vector equation

   [ u_1' ] = [      u_2              ]  (2)
   [ u_2' ]   [ mu[(1-u_1^2)u_2-u_1]  ]

   which is now in the desired form of u_t = f(u,t). 

  ------------------------------------------------------------------------- */
#include <petsctao.h>
#include <petscts.h>

typedef struct _n_User *User;
struct _n_User {
  PetscReal mu;
  PetscReal next_output;
 
  /* Sensitivity analysis support */ 
  PetscInt  steps,nstages;
  PetscReal ftime,x_ob[2];
  Mat       A;            /* Jacobian matrix */
  Vec       x,lambda[2];  /* adjoint variables */
  PetscBool reversemode;
};

PetscErrorCode FormFunctionGradient(Tao,Vec,PetscReal*,Vec,void*);

/*
*  User-defined routines
*/
#undef __FUNCT__
#define __FUNCT__ "IFunction"
static PetscErrorCode IFunction(TS ts,PetscReal t,Vec X,Vec Xdot,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  User           user = (User)ctx;
  PetscScalar    *x,*xdot,*f;

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  f[0] = xdot[0] - x[1];
  f[1] = c21*(xdot[0]-x[1]) + xdot[1] - user->mu*((1.0-x[0]*x[0])*x[1] - x[0]) ;
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(Xdot,&xdot);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IJacobian"
static PetscErrorCode IJacobian(TS ts,PetscReal t,Vec X,Vec Xdot,PetscReal a,Mat A,Mat B,void *ctx)
{
  PetscErrorCode ierr;
  User           user     = (User)ctx;
  PetscInt       rowcol[] = {0,1};
  PetscScalar    *x,J[2][2];

  PetscFunctionBeginUser;
  ierr    = VecGetArray(X,&x);CHKERRQ(ierr);

  J[0][0] = a;     J[0][1] =  -1.0;
  J[1][0] = c21*a + user->mu*(1.0 + 2.0*x[0]*x[1]);   J[1][1] = -c21 + a - user->mu*(1.0-x[0]*x[0]);
 
  ierr    = MatSetValues(B,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr    = VecRestoreArray(X,&x);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (A != B) {
    ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Monitor"
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
  ierr = TSGetDuration(ts,NULL,&tfinal);CHKERRQ(ierr);

  while (user->next_output <= t && user->next_output <= tfinal) {
    ierr = VecDuplicate(X,&interpolatedX);CHKERRQ(ierr);
    ierr = TSInterpolate(ts,user->next_output,interpolatedX);CHKERRQ(ierr);
    ierr = VecGetArrayRead(interpolatedX,&x);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"[%.1f] %D TS %.6f (dt = %.6f) X % 12.6e % 12.6e\n",
                       user->next_output,step,t,dt,(double)PetscRealPart(x[0]),
                       (double)PetscRealPart(x[1]));CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(interpolatedX,&x);CHKERRQ(ierr);
    ierr = VecDestroy(&interpolatedX);CHKERRQ(ierr);
    user->next_output += 0.1;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "OutputBIN"
static PetscErrorCode OutputBIN(const char *filename, PetscViewer *viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscViewerCreate(PETSC_COMM_WORLD, viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetType(*viewer, PETSCVIEWERBINARY);CHKERRQ(ierr);
  ierr = PetscViewerFileSetMode(*viewer,FILE_MODE_WRITE);CHKERRQ(ierr);
  ierr = PetscViewerFileSetName(*viewer, filename);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MonitorBIN"
static PetscErrorCode MonitorBIN(TS ts,PetscInt stepnum,PetscReal time,Vec X,void *ctx)
{
  PetscViewer    viewer;
  PetscInt       ns,i;
  Vec            *Y;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscReal      tprev;

  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = TSGetPrevTime(ts,&tprev);CHKERRQ(ierr);
  ierr = PetscSNPrintf(filename,sizeof filename,"ex20-SA-%06d.bin",stepnum);CHKERRQ(ierr);
  ierr = OutputBIN(filename,&viewer);CHKERRQ(ierr);
  ierr = VecView(X,viewer);CHKERRQ(ierr);
  /* ierr = PetscRealView(1,&time,viewer);CHKERRQ(ierr); */
  ierr = PetscViewerBinaryWrite(viewer,&tprev,1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);
  /* ierr = PetscViewerBinaryWrite(viewer,&h ,1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr); */
  ierr = TSGetStages(ts,&ns,&Y);CHKERRQ(ierr);

  for (i=0;i<ns && stepnum>0;i++) {
    ierr = VecView(Y[i],viewer);CHKERRQ(ierr);
  }

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MonitorADJ2"
static PetscErrorCode MonitorADJ2(TS ts,PetscInt step,PetscReal t,Vec X,void *ctx)
{
  PetscReal      ptime;
  Vec            Sol,*Y;
  PetscInt       Nr,i;
  PetscViewer    viewer;
  PetscReal      timepre;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscSNPrintf(filename,sizeof filename,"ex20-SA-%06d.bin",step);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);

  ierr = TSGetSolution(ts,&Sol);CHKERRQ(ierr);
  ierr = VecLoad(Sol,viewer);CHKERRQ(ierr);

  Nr   = 1;
  /* ierr = PetscRealLoad(Nr,&Nr,&timepre,viewer);CHKERRQ(ierr); */
  ierr = PetscViewerBinaryRead(viewer,&timepre,1,PETSC_REAL);CHKERRQ(ierr);

  ierr = TSGetStages(ts,&Nr,&Y);CHKERRQ(ierr);
  for (i=0;i<Nr ;i++) {
    ierr = VecLoad(Y[i],viewer);CHKERRQ(ierr);
  }

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = TSGetTime(ts,&ptime);CHKERRQ(ierr);
  ierr = TSSetTimeStep(ts,-ptime+timepre);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  TS             ts;            /* nonlinear solver */
  Vec            ic;           
  PetscBool      monitor = PETSC_FALSE;
  PetscScalar    *x_ptr;
  PetscMPIInt    size;
  struct _n_User user;
  PetscErrorCode ierr;


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInitialize(&argc,&argv,NULL,help);
  
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");


  Tao            tao;
  TaoConvergedReason reason;
  /* Create TAO solver and set desired solution method */
  ierr = TaoCreate(PETSC_COMM_WORLD,&tao);CHKERRQ(ierr);
  ierr = TaoSetType(tao,TAOCG);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.next_output = 0.0;
  user.mu          = 1.0e6;
  user.steps       = 0;
  user.ftime       = 0.5;
  user.reversemode = PETSC_TRUE;
  ierr = PetscOptionsGetBool(NULL,"-monitor",&monitor,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mu","Stiffness parameter","<1.0e6>",user.mu,&user.mu,NULL);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors, solve same ODE on every process
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&user.A);CHKERRQ(ierr);
  ierr = MatSetSizes(user.A,PETSC_DECIDE,PETSC_DECIDE,2,2);CHKERRQ(ierr);
  ierr = MatSetFromOptions(user.A);CHKERRQ(ierr);
  ierr = MatSetUp(user.A);CHKERRQ(ierr);

  ierr = MatGetVecs(user.A,&user.x,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,IFunction,&user);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,user.A,user.A,IJacobian,&user);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,PETSC_DEFAULT,user.ftime);CHKERRQ(ierr);
  if (monitor) {
    ierr = TSMonitorSet(ts,Monitor,&user,NULL);CHKERRQ(ierr);
  }
  ierr = TSMonitorSet(ts,MonitorBIN,&user,NULL);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecGetArray(user.x,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 2.0;   x_ptr[1] = -0.66666654321;
  ierr = VecRestoreArray(user.x,&x_ptr);CHKERRQ(ierr);
  /* ierr = TSSetInitialTimeStep(ts,0.0,.001);CHKERRQ(ierr); */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSGetStages(ts,&user.nstages,NULL);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,user.x);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&user.ftime);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&user.steps);CHKERRQ(ierr);
  ierr = VecGetArray(user.x,&x_ptr);CHKERRQ(ierr);
  user.x_ob[0] = x_ptr[0];
  user.x_ob[1] = x_ptr[1];
  ierr = VecView(user.x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Create sensitivity variable */
  ierr = MatGetVecs(user.A,&user.lambda[0],NULL);CHKERRQ(ierr);
  ierr = MatGetVecs(user.A,&user.lambda[1],NULL);CHKERRQ(ierr);
  /*
     Optimization starts
  */
  /* Set initial solution guess */
  ierr = MatGetVecs(user.A,&ic,NULL);CHKERRQ(ierr);
  ierr = VecGetArray(ic,&x_ptr);CHKERRQ(ierr);
  x_ptr[0]  = 2.1;
  x_ptr[1]  = -0.7;
  ierr = VecRestoreArray(ic,&x_ptr);CHKERRQ(ierr);

  ierr = TaoSetInitialVector(tao,ic);CHKERRQ(ierr);

  /* Set routine for function and gradient evaluation */
  ierr = TaoSetObjectiveAndGradientRoutine(tao,FormFunctionGradient,(void *)&user);CHKERRQ(ierr);

  /* Set bounds for the optimization */
  /*
  Vec lowerb,upperb;
  ierr = VecDuplicate(ic,&lowerb);CHKERRQ(ierr);
  ierr = VecDuplicate(ic,&upperb);CHKERRQ(ierr);
  ierr = VecGetArray(lowerb,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 1.0; x_ptr[1] = 0.1;
  ierr = VecRestoreArray(lowerb,&x_ptr);CHKERRQ(ierr);
  ierr = VecGetArray(upperb,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 3.0; x_ptr[1] = 0.9;
  ierr = VecRestoreArray(upperb,&x_ptr);CHKERRQ(ierr);
  ierr = TaoSetVariableBounds(tao,lowerb,upperb);
  */
  /* Check for any TAO command line options */
  KSP ksp;
  PC  pc;
  ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);
  ierr = TaoGetKSP(tao,&ksp);CHKERRQ(ierr);
  if (ksp) {
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
  }

  ierr = TaoSetTolerances(tao,1e-10,1e-10,1e-10,PETSC_DEFAULT,PETSC_DEFAULT);

  /* SOLVE THE APPLICATION */
  ierr = TaoSolve(tao); CHKERRQ(ierr);

  /* Get information on termination */
  ierr = TaoGetConvergedReason(tao,&reason);CHKERRQ(ierr);
  if (reason <= 0){
      ierr=PetscPrintf(MPI_COMM_WORLD, "Try another method! \n");CHKERRQ(ierr);
  }

  ierr = VecView(ic,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /* Free TAO data structures */
  ierr = TaoDestroy(&tao);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&user.A);CHKERRQ(ierr);
  ierr = VecDestroy(&user.x);CHKERRQ(ierr);
  ierr = VecDestroy(&user.lambda[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&user.lambda[1]);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  /*ierr = VecDestroy(&lowerb);CHKERRQ(ierr);
  ierr = VecDestroy(&upperb);CHKERRQ(ierr);*/
  ierr = VecDestroy(&ic);CHKERRQ(ierr);
  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}


/* ------------------------------------------------------------------ */
#undef __FUNCT__
#define __FUNCT__ "FormFunctionGradient"
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
  User              user_ptr = (User)ctx;
  TS                ts;
  PetscScalar       *x_ptr,*y_ptr;
  PetscErrorCode    ierr;

  ierr = VecCopy(IC,user_ptr->x);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSBEULER);CHKERRQ(ierr);
  ierr = TSSetIFunction(ts,NULL,IFunction,user_ptr);CHKERRQ(ierr);
  ierr = TSSetIJacobian(ts,user_ptr->A,user_ptr->A,IJacobian,user_ptr);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);

  user_ptr->reversemode = PETSC_FALSE;
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set time
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetTime(ts,0.0);CHKERRQ(ierr);
  /*ierr = TSSetInitialTimeStep(ts,0.0,.001);CHKERRQ(ierr);*/
  ierr = TSSetDuration(ts,2000,0.5);CHKERRQ(ierr);
 
  ierr = TSMonitorSet(ts,MonitorBIN,user_ptr,NULL);CHKERRQ(ierr); 
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);
  ierr = TSGetStages(ts,&user_ptr->nstages,NULL);CHKERRQ(ierr);

  ierr = TSSolve(ts,user_ptr->x);CHKERRQ(ierr);
  ierr = VecGetArray(user_ptr->x,&x_ptr);CHKERRQ(ierr);
  *f   = (x_ptr[0]-user_ptr->x_ob[0])*(x_ptr[0]-user_ptr->x_ob[0])+(x_ptr[1]-user_ptr->x_ob[1])*(x_ptr[1]-user_ptr->x_ob[1]);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Observed value y_ob=[%f; %f], ODE solution y=[%f;%f], Cost function f=%f\n",(double)user_ptr->x_ob[0],(double)user_ptr->x_ob[1],(double)x_ptr[0],(double)x_ptr[1],(double)(*f));CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Adjoint model starts here
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user_ptr->reversemode = PETSC_TRUE;
  /*   Redet initial conditions for the adjoint integration */
  ierr = VecGetArray(user_ptr->lambda[0],&y_ptr);CHKERRQ(ierr);
  y_ptr[0] = 2.*(x_ptr[0]-user_ptr->x_ob[0]);
  y_ptr[1] = 2.*(x_ptr[1]-user_ptr->x_ob[1]);
  /*
  y_ptr[0] = 1.0;
  y_ptr[1] = 0.0;
  */
  ierr = VecRestoreArray(user_ptr->lambda[0],&y_ptr);CHKERRQ(ierr);
  ierr = TSAdjointSetSensitivity(ts,user_ptr->lambda,1);CHKERRQ(ierr);

  /*   Reset start time for the adjoint integration */
  ierr = TSSetTime(ts,user_ptr->ftime);CHKERRQ(ierr);

  /*   Set duration */
  ierr = TSSetDuration(ts,user_ptr->steps,PETSC_DEFAULT);CHKERRQ(ierr);

  /*   Set up monitor */
  ierr = TSMonitorCancel(ts);CHKERRQ(ierr);
  ierr = TSMonitorSet(ts,MonitorADJ2,user_ptr,NULL);CHKERRQ(ierr);

  ierr = TSSolve(ts,user_ptr->x);CHKERRQ(ierr);

  ierr = VecCopy(user_ptr->lambda[0],G);

  /* print statistics */
  /* 
  ierr = TSGetSolveTime(ts,&user_ptr->ftime);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&user_ptr->steps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"mu %g, steps %D, ftime %g\n",(double)user_ptr->mu,user->steps,(double)user->ftime);CHKERRQ(ierr);
  ierr = VecView(G,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(IC,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  */

  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

