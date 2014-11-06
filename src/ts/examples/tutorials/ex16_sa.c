/*  
   First step toward adjoint ERK solver
   Features:
     1. checkpointed data is stored in the user context;
     2. register a new method with coefficients for adjoint;
     3. all computation is done in the user driver without changing petsc code.
     4. tested with a single lambda variable  
*/
static char help[] = "Solves the van der Pol equation.\n\
Input parameters include:\n\
      -mu : stiffness parameter\n\n";

/*
   Concepts: TS^time-dependent nonlinear problems
   Concepts: TS^van der Pol equation
   Processors: 1
*/
/* ------------------------------------------------------------------------

   This program solves the van der Pol equation
       y'' - \mu (1-y^2)*y' + y = 0        (1)
   on the domain 0 <= x <= 1, with the boundary conditions
       y(0) = 2, y'(0) = 0,
   This is a nonlinear equation.

   Notes:
   This code demonstrates the TS solver interface to two variants of
   linear problems, u_t = f(u,t), namely turning (1) into a system of
   first order differential equations,

   [ y' ] = [          z          ]
   [ z' ]   [ \mu (1 - y^2) z - y ]

   which then we can write as a vector equation

   [ u_1' ] = [             u_2           ]  (2)
   [ u_2' ]   [ \mu (1 - u_1^2) u_2 - u_1 ]

   which is now in the desired form of u_t = f(u,t). 
  ------------------------------------------------------------------------- */

#include <petscts.h>
#include <petscmat.h>
typedef struct _n_User *User;
struct _n_User {
  PetscReal mu;
  PetscBool imex;
  PetscReal next_output;
  PetscReal tprev;
  PetscInt  stage,nstages;
  
  Vec       *Y; // stage values
  Vec       X; // solution states 
};

/*
*  User-defined routines
*/
#undef __FUNCT__
#define __FUNCT__ "RHSFunction"
static PetscErrorCode RHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  User           user = (User)ctx;
  PetscScalar    *x,*f;

  PetscFunctionBeginUser;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  //f[0] = (user->imex ? x[1] : 0);
  //f[1] = 0.0;
  f[0] = x[1];
  f[1] = user->mu*(1.-x[0]*x[0])*x[1]-x[0];
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

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
  f[0] = xdot[0] + (user->imex ? 0 : x[1]);
  f[1] = xdot[1] - user->mu*(1. - x[0]*x[0])*x[1] + x[0];
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
  PetscReal      mu       = user->mu;
  PetscInt       rowcol[] = {0,1};
  PetscScalar    *x,J[2][2];

  PetscFunctionBeginUser;
  ierr    = VecGetArray(X,&x);CHKERRQ(ierr);
  J[0][0] = a;                    J[0][1] = (user->imex ? 0 : 1.);
  J[1][0] = 2.*mu*x[0]*x[1]+1.;   J[1][1] = a - mu*(1. - x[0]*x[0]);
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
#define __FUNCT__ "ADJRHSFunction"
static PetscErrorCode ADJRHSFunction(TS tsadj,PetscReal t,Vec X,Vec F,void *ctx)
{
  PetscErrorCode ierr;
  User           user = (User)ctx;
  PetscReal      mu   = user->mu;
  PetscScalar    *x,J[2][2];
  PetscInt       rowcol[] = {0,1};
  Mat            JTRAN;
  
  PetscFunctionBeginUser;
  ierr        = MatCreate(PetscObjectComm((PetscObject)tsadj),&JTRAN);CHKERRQ(ierr);
  ierr        = MatSetSizes(JTRAN, PETSC_DECIDE, PETSC_DECIDE, 2, 2);CHKERRQ(ierr);
  ierr        = MatSetUp(JTRAN);CHKERRQ(ierr);
  ierr        = VecGetArray(user->Y[user->stage],&x);CHKERRQ(ierr);
  J[0][0]     = 0;
  J[0][1]     = -2.*mu*x[1]*x[0]+1;
  J[1][0]     = 1.0;
  J[1][1]     = mu*(1.0-x[0]*x[0]);
  ierr        = MatSetValues(JTRAN,2,rowcol,2,rowcol,&J[0][0],INSERT_VALUES);CHKERRQ(ierr);
  ierr        = MatAssemblyBegin(JTRAN,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr        = MatAssemblyEnd(JTRAN,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr        = MatMult(JTRAN,X,F);CHKERRQ(ierr);
  user->stage = (user->stage+1)%(user->nstages);
  ierr        = MatDestroy(&JTRAN);CHKERRQ(ierr);  
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RegisterMyARK2"
static PetscErrorCode RegisterMyARK2(void)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  {
    const PetscReal
      A[3][3] = {{0,0,0},
                 {0.41421356237309504880,0,0},
                 {0.75,0.25,0}},
      At[3][3] = {{0,0,0},
                  {0.12132034355964257320,0.29289321881345247560,0},
                  {0.20710678118654752440,0.50000000000000000000,0.29289321881345247560}},
      *bembedt = NULL,*bembed = NULL;
    ierr = TSARKIMEXRegister("myark2",2,3,&At[0][0],NULL,NULL,&A[0][0],NULL,NULL,bembedt,bembed,0,NULL,NULL);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RegisterRKADJ"
/* Aij = Aij*Bi/Bj*/
static PetscErrorCode RegisterRKADJ(void)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  {
    const PetscReal
      A[4][4] = {{0,0,0,0},
                 {0,0,0,0},
                 {0,-1.0,0,0},
                 {0,0,-3.0/4.0,0}},
      b[4]    = {0,-4.0/9.0,-1.0/3.0,-2.0/9.0},
      bembed[4] = {7.0/24.0,1.0/4.0,1.0/3.0,1.0/8.0};
    ierr = TSRKRegister("rk3bsadj",3,4,&A[0][0],b,NULL,bembed,3,b);CHKERRQ(ierr);
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
  PetscReal         tfinal, dt, tprev;
  User              user = (User)ctx;
//  Vec               interpolatedX;

  PetscFunctionBeginUser;
  ierr = TSGetTimeStep(ts,&dt);CHKERRQ(ierr);
  ierr = TSGetDuration(ts,NULL,&tfinal);CHKERRQ(ierr);
  ierr = TSGetPrevTime(ts,&tprev);CHKERRQ(ierr);
//  while (user->next_output <= t && user->next_output <= tfinal) {
//    ierr = VecDuplicate(X,&interpolatedX);CHKERRQ(ierr);
//    ierr = TSInterpolate(ts,user->next_output,interpolatedX);CHKERRQ(ierr);
    ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"[%.1f] %D TS %.6f (dt = %.6f) X % 12.6e % 12.6e\n",user->next_output,step,t,dt,(double)PetscRealPart(x[0]),(double)PetscRealPart(x[1]));CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"t %.6f (tprev = %.6f) \n",t,tprev);CHKERRQ(ierr);
//    ierr = VecRestoreArrayRead(interpolatedX,&x);CHKERRQ(ierr);
//    ierr = VecDestroy(&interpolatedX);CHKERRQ(ierr);
    
//    user->next_output += 0.1;
//  }
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
#define __FUNCT__ "TestMonitor"
static PetscErrorCode TestMonitor(const char *filename, Vec X, PetscReal time, PetscInt ns, Vec *Y, PetscInt stepnum)
{
  Vec            odesol,stagesol;
  PetscInt       Nr,i;
  PetscBool      equal;
  PetscReal      timeread;
  PetscViewer    viewer;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&odesol);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&stagesol);CHKERRQ(ierr);
  ierr = VecLoad(odesol,viewer);CHKERRQ(ierr);
  VecEqual(X,odesol,&equal);
  if(!equal) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_FILE_UNEXPECTED,"Error in reading the ODE solution from file");
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"IO test OK for ODE solution\n");CHKERRQ(ierr);
  }

  Nr   = 1;
  //ierr = PetscRealLoad(Nr,&Nr,&timeread,viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&timeread,1,PETSC_REAL);CHKERRQ(ierr);


  for (i=0;i<ns && stepnum >1;i++) {
    ierr = VecLoad(stagesol,viewer);CHKERRQ(ierr);
    VecEqual(Y[i],stagesol,&equal);
    if(!equal) {
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_FILE_UNEXPECTED,"Error in reading the %2d-th stage value from file",i);
    } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"IO test OK for Stage values\n");CHKERRQ(ierr);
    }
  }

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&odesol);CHKERRQ(ierr);
  ierr = VecDestroy(&stagesol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "LoadChkpts"
static PetscErrorCode LoadChkpts(PetscInt stepnum, void *ctx)
{ 
  User           user = (User)ctx;
  PetscInt       Nr,i;
  PetscViewer    viewer;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscSNPrintf(filename,sizeof filename,"ex16-SA-%06d.bin",stepnum);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = VecLoad(user->X,viewer);CHKERRQ(ierr);

  Nr   = 1;
  //ierr = PetscRealLoad(Nr,&Nr,&timeread,viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer,&(user->tprev),1,PETSC_REAL);CHKERRQ(ierr);

  for (i=user->nstages-1;i>=0 ;i--) {
    ierr = VecLoad(user->Y[i],viewer);CHKERRQ(ierr);
  }
  
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
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
  ierr = PetscSNPrintf(filename,sizeof filename,"ex16-SA-%06d.bin",stepnum);CHKERRQ(ierr);
  ierr = OutputBIN(filename,&viewer);CHKERRQ(ierr);
  ierr = VecView(X,viewer);CHKERRQ(ierr);
  //ierr = PetscRealView(1,&time,viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryWrite(viewer,&tprev,1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);
  //ierr = PetscViewerBinaryWrite(viewer,&h ,1,PETSC_REAL,PETSC_FALSE);CHKERRQ(ierr);
  ierr = TSGetStages(ts,&ns,&Y);CHKERRQ(ierr);

  for (i=0;i<ns && stepnum>0;i++) {
    ierr = VecView(Y[i],viewer);CHKERRQ(ierr);
  }

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  //ierr = TestMonitor(filename,X,time,ns,Y,stepnum);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MonitorADJ"
/* Monitor timesteps and use interpolation to output at integer multiples of 0.1 */
static PetscErrorCode MonitorADJ(TS tsadj,PetscInt step,PetscReal t,Vec X,void *ctx)
{
  PetscErrorCode    ierr;
  PetscReal         ptime;
  User              user = (User)ctx;

  PetscFunctionBeginUser;
  ierr = LoadChkpts(step,ctx);
  ierr = TSGetTime(tsadj,&ptime);CHKERRQ(ierr);
  ierr = TSSetTimeStep(tsadj,-ptime+user->tprev);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  TS             ts;            /* nonlinear solver */
  Vec            x;             /* solution, residual vectors */
  Mat            A;             /* Jacobian matrix */
  Vec            lambda;        /* adjoint variable */
  PetscInt       steps;
  PetscReal      ftime   =0.5;
  PetscBool      monitor = PETSC_FALSE;
  PetscScalar    *x_ptr;
  PetscMPIInt    size;
  struct _n_User user;
  PetscErrorCode ierr;
  TSAdapt        adapt;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize program
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscInitialize(&argc,&argv,NULL,help);

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");

  ierr = RegisterMyARK2();CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Set runtime options
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  user.mu          = 1;
  user.imex        = PETSC_FALSE;
  user.next_output = 0.0;
  user.stage   = 0; 


  ierr = PetscOptionsGetReal(NULL,"-mu",&user.mu,NULL);CHKERRQ(ierr);
  //ierr = PetscOptionsGetBool(NULL,"-imex",&user.imex,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,"-monitor",&monitor,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create necessary matrix and vectors, solve same ODE on every process
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,2,2);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatGetVecs(A,&x,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSRK);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,RHSFunction,&user);CHKERRQ(ierr);
  //ierr = TSSetIFunction(ts,NULL,IFunction,&user);CHKERRQ(ierr);
  //ierr = TSSetIJacobian(ts,A,A,IJacobian,&user);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,PETSC_DEFAULT,ftime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP);CHKERRQ(ierr);
  if (monitor) {
    ierr = TSMonitorSet(ts,Monitor,&user,NULL);CHKERRQ(ierr);
  }
  ierr = TSMonitorSet(ts,MonitorBIN,&user,NULL);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecGetArray(x,&x_ptr);CHKERRQ(ierr);

  x_ptr[0] = 2;   x_ptr[1] = 0.66666654321;

  ierr = VecRestoreArray(x,&x_ptr);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,.001);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"mu %g, steps %D, ftime %g\n",(double)user.mu,steps,(double)ftime);CHKERRQ(ierr);

  //ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
  //ierr = TSAdaptSetType(adapt,TSADAPTNONE);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = TSGetStages(ts,&user.nstages,NULL);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,x);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"mu %g, steps %D, ftime %g\n",(double)user.mu,steps,(double)ftime);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = TSReset(ts);CHKERRQ(ierr);
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Adjoint model starts here
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  TS        tsadj;
  // SNES      snesadj;
  DM        dm;
  PetscInt  i;
  PetscReal ptime;
  user.stage   = 0; 

  ierr = VecCreate(PETSC_COMM_WORLD,&user.X);CHKERRQ(ierr);
  ierr = MatGetVecs(A,&lambda,NULL);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(lambda,user.nstages,&user.Y);CHKERRQ(ierr);

  ierr = TSCreate(PetscObjectComm((PetscObject)ts),&tsadj);CHKERRQ(ierr);

  //ierr = TSGetSNES(ts,&snesadj);CHKERRQ(ierr);
  //ierr = TSSetSNES(tsadj,snesadj);CHKERRQ(ierr);

  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = TSSetDM(tsadj,dm);CHKERRQ(ierr);

  //   Set initial conditions for the adjoint integration
  ierr = VecGetArray(lambda,&x_ptr);CHKERRQ(ierr);
  x_ptr[0] = 1.0;   x_ptr[1] = 0.0;
  ierr = VecRestoreArray(lambda,&x_ptr);CHKERRQ(ierr);
  ierr = VecView(lambda,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  //   Reset start time for the adjoint integration
  ierr = TSSetTime(tsadj,ftime);CHKERRQ(ierr);
  //ierr = TSSetDuration(tsadj,PETSC_DEFAULT,ftime);CHKERRQ(ierr);

  //   Register the coefficients for the adjoint integration
  ierr = RegisterRKADJ();CHKERRQ(ierr);
  ierr = TSSetType(tsadj,TSRK);CHKERRQ(ierr);
  ierr = TSRKSetType(tsadj,"rk3bsadj");CHKERRQ(ierr);

  //   Set RHS function and number of steps for the adjoint integration
  ierr = TSSetRHSFunction(tsadj,NULL,ADJRHSFunction,&user);CHKERRQ(ierr);
  ierr = TSSetDuration(tsadj,steps,PETSC_DEFAULT);CHKERRQ(ierr);

  //   Turn off adapt for the adjoint integration (???)
  ierr = TSGetAdapt(tsadj,&adapt);CHKERRQ(ierr);
  ierr = TSAdaptSetType(adapt,TSADAPTNONE);CHKERRQ(ierr);

  //   Set up monitor
  ierr = TSMonitorSet(tsadj,MonitorADJ,&user,NULL);CHKERRQ(ierr);
  
  ierr = TSSetSolution(tsadj,lambda);CHKERRQ(ierr);
  ierr = TSSetUp(tsadj);CHKERRQ(ierr);
  for (i=steps; i>0; i--) {
    ierr = LoadChkpts(i,&user);
    ierr = TSGetTime(tsadj,&ptime);CHKERRQ(ierr);
    ierr = TSGetTimeStepNumber(tsadj,&steps);CHKERRQ(ierr);
    ierr = TSSetTimeStep(tsadj,-ptime+user.tprev);CHKERRQ(ierr);
    //ierr = TSMonitor(tsadj,steps,ptime,lambda);CHKERRQ(ierr);
    ierr = TSStep(tsadj);CHKERRQ(ierr);
  }

  ierr = TSGetSolution(tsadj,&lambda);CHKERRQ(ierr);
  ierr = TSGetSolveTime(tsadj,&ptime);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(tsadj,&steps);CHKERRQ(ierr);
  //ierr = TSMonitor(tsadj,steps,ptime,lambda);CHKERRQ(ierr);
  ierr = TSViewFromOptions(tsadj,NULL,"-ts_view");CHKERRQ(ierr);
  ierr = PetscObjectSAWsBlock((PetscObject)tsadj);CHKERRQ(ierr);

  ierr = VecView(lambda,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
/* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&lambda);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = PetscFinalize();
  PetscFunctionReturn(0);
}
