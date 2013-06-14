static char help[] = "Solves the ordinary differential equations (IVPs) using explicit and implicit time-integration methods.\n";

/*

  Concepts:   TS
  Useful command line parameters:
  -problem <hull1972a1>: choose which problem to solve (see references
                      for complete listing of problems).
  -ts_type <euler>: specify time-integrator
  -ts_adapt_type <basic>: specify time-step adapting (none,basic,advanced)
  -refinement_levels <1>: number of refinement levels for convergence analysis
  -refinement_factor <2.0>: factor to refine time step size by for convergence analysis
  -dt <0.01>: specify time step (initial time step for convergence analysis)

*/

/*
List of cases and their names in the code:-
  From Hull, T.E., Enright, W.H., Fellen, B.M., and Sedgwick, A.E.,
      "Comparing Numerical Methods for Ordinary Differential
       Equations", SIAM J. Numer. Anal., 9(4), 1972, pp. 603 - 635
    A1 -> "hull1972a1" (exact solution available)
    A2 -> "hull1972a2" (exact solution available)
    A3 -> "hull1972a3" (exact solution available)
    A4 -> "hull1972a4" (exact solution available)
    A5 -> "hull1972a5"
    B1 -> "hull1972b1"
    B2 -> "hull1972b2"
    B3 -> "hull1972b3"
    B4 -> "hull1972b4"
    B5 -> "hull1972b5"
    C1 -> "hull1972c1"
    C2 -> "hull1972c2"
    C3 -> "hull1972c3"
    C4 -> "hull1972c4"
*/

#include <petscts.h>

/* Function declarations  */
PetscErrorCode  SolveODE      (char*,PetscReal,PetscReal,PetscInt,PetscReal*,PetscBool*);
PetscInt        GetSize       (char*);
PetscErrorCode  Initialize    (Vec,void*);
PetscErrorCode  RHSFunction   (TS,PetscReal,Vec,Vec,void*);
PetscErrorCode  IFunction     (TS,PetscReal,Vec,Vec,Vec,void*);
PetscErrorCode  IJacobian     (TS,PetscReal,Vec,Vec,PetscReal,Mat*,Mat*,MatStructure*,void*);
PetscErrorCode  ExactSolution (Vec,void*,PetscReal,PetscBool*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  PetscErrorCode  ierr;                       /* Error code                                           */
  char            ptype[256] = "hull1972a1";  /* Problem specification                                */
  PetscInt        n_refine   = 1;             /* Number of refinement levels for convergence analysis */
  PetscReal       refine_fac = 2.0;           /* Refinement factor for dt                             */
  PetscReal       dt_initial = 0.01;          /* Initial default value of dt                          */
  PetscReal       tfinal     = 20.0;          /* Final time for the time-integration                  */
  PetscInt        maxiter    = 100000;        /* Maximum number of time-integration iterations        */
  PetscReal      *error;                      /* Array to store the errors for convergence analysis   */
  PetscInt        nproc;                      /* No of processors                                     */
  PetscBool       flag;                       /* Flag denoting availability of exact solution         */
  PetscInt        r;              

  /* Initialize program */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);

  /* Check if running with only 1 proc */
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&nproc);    CHKERRQ(ierr);
  if (nproc>1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only for sequential runs");

  ierr = PetscOptionsString("-problem","Problem specification","<hull1972a1>",
                            ptype,ptype,sizeof(ptype),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-refinement_levels","Number of refinement levels for convergence analysis",
                         "<1>",n_refine,&n_refine,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-refinement_factor","Refinement factor for dt","<2.0>",
                          refine_fac,&refine_fac,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-dt","Time step size (for convergence analysis, initial time step)",
                          "<0.01>",dt_initial,&dt_initial,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-final_time","Final time for the time-integration","<20.0>",
                          tfinal,&tfinal,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscMalloc(n_refine*sizeof(PetscReal),&error);CHKERRQ(ierr);
  for (r = 0; r < n_refine; r++) {
    PetscReal dt;
    error[r] = 0;
    if (!r) dt = dt_initial;
    else    dt /= refine_fac;
    
    printf("Solving ODE \"%s\" with dt %f, final time %f and system size %d.\n",ptype,dt,tfinal,GetSize(&ptype[0]));
    ierr = SolveODE(&ptype[0],dt,tfinal,maxiter,&error[r],&flag);
    if (flag) {
      /* If exact solution available for the specified ODE */
      if (r > 0) {
        PetscReal conv_rate = (log(error[r]) - log(error[r-1])) / (-log(refine_fac));
        printf("Error = %E,\tConvergence rate = %f\n.",error[r],conv_rate);
      } else {
        printf("Error = %E.\n",error[r]);
      }
    }
  }
  PetscFree(error);CHKERRQ(ierr);

  /* Exit */
  ierr = PetscFinalize(); CHKERRQ(ierr);
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "SolveODE"
/* Solves the specified ODE and computes the error if exact solution is available */
PetscErrorCode SolveODE(char* ptype, PetscReal dt, PetscReal tfinal, PetscInt maxiter, PetscReal *error, PetscBool *exact_flag)
{
  PetscErrorCode  ierr;             /* Error code                             */
  TS              ts;               /* time-integrator                        */
  Vec             Y;                /* Solution vector                        */
  Vec             Yex;              /* Exact solution                         */
  PetscInt        N;                /* Size of the system of equations        */
  TSType          time_scheme;      /* Type of time-integration scheme        */
  PetscBool       impl_flg;         /* Flag whether implicit time-integration */
  Mat             Jac;              /* Jacobian matrix                        */

  PetscFunctionBegin;
  N = GetSize(&ptype[0]);
  if (N < 0) {
    printf("Error: Illegal problem specification.\n");
    return(0);
  }
  ierr = VecCreate(PETSC_COMM_WORLD,&Y);CHKERRQ(ierr);
  ierr = VecSetSizes(Y,N,PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecSetUp(Y);                   CHKERRQ(ierr);
  ierr = VecSet(Y,0);                   CHKERRQ(ierr);

  /* Initialize the problem */
  ierr = Initialize(Y,&ptype[0]);

  /* Create and initialize the time-integrator                            */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);                      CHKERRQ(ierr);
  /* Default time integration options                                     */
  ierr = TSSetType(ts,TSEULER);                               CHKERRQ(ierr);
  ierr = TSSetDuration(ts,maxiter,tfinal);                    CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,dt);                     CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_MATCHSTEP); CHKERRQ(ierr);
  /* Read command line options for time integration                       */
  ierr = TSSetFromOptions(ts);                                CHKERRQ(ierr);
  /* Set solution vector                                                  */
  ierr = TSSetSolution(ts,Y);                                 CHKERRQ(ierr);
  /* Specify left/right-hand side functions                               */
  ierr = TSGetType(ts,&time_scheme);                          CHKERRQ(ierr);
  if ((!strcmp(time_scheme,TSEULER)) || (!strcmp(time_scheme,TSRK)) || (!strcmp(time_scheme,TSSSP))) {
    /* Explicit time-integration -> specify right-hand side function ydot = f(y) */
    impl_flg = PETSC_FALSE;
    ierr = TSSetRHSFunction(ts,PETSC_NULL,RHSFunction,&ptype[0]);CHKERRQ(ierr);
  } else if ((!strcmp(time_scheme,TSBEULER)) || (!strcmp(time_scheme,TSARKIMEX))) {
    /* Implicit time-integration -> specify left-hand side function ydot-f(y) = 0 */
    /* and its Jacobian function                                                 */
    impl_flg = PETSC_TRUE;
    ierr = TSSetIFunction(ts,PETSC_NULL,IFunction,&ptype[0]); CHKERRQ(ierr);
    ierr = MatCreate(PETSC_COMM_WORLD,&Jac);                  CHKERRQ(ierr);
    ierr = MatSetSizes(Jac,PETSC_DECIDE,PETSC_DECIDE,N,N);    CHKERRQ(ierr);
    ierr = MatSetFromOptions(Jac);                            CHKERRQ(ierr);
    ierr = MatSetUp(Jac);                                     CHKERRQ(ierr);
    ierr = TSSetIJacobian(ts,Jac,Jac,IJacobian,&ptype[0]);    CHKERRQ(ierr);
  }

  /* Solve */
  ierr = TSSolve(ts,Y);CHKERRQ(ierr);

  /* Exact solution */
  ierr = VecDuplicate(Y,&Yex); CHKERRQ(ierr);
  ierr = ExactSolution(Yex,&ptype[0],tfinal,exact_flag);

  /* Calculate Error */
  ierr = VecAYPX(Yex,-1.0,Y);       CHKERRQ(ierr);
  ierr = VecNorm(Yex,NORM_2,error); CHKERRQ(ierr);
  *error = sqrt(((*error)*(*error))/N);

  /* Clean up and finalize */
  if (impl_flg) ierr = MatDestroy(&Jac);  CHKERRQ(ierr);
  ierr = TSDestroy(&ts);                  CHKERRQ(ierr);
  ierr = VecDestroy(&Yex);                CHKERRQ(ierr);
  ierr = VecDestroy(&Y);                  CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "GetSize"
/* Returns the size of the system of equations depending on problem specification */
PetscInt GetSize(char *p)
{
  PetscFunctionBegin;
  if      ((!strcmp(p,"hull1972a1")) 
         ||(!strcmp(p,"hull1972a2"))
         ||(!strcmp(p,"hull1972a3"))
         ||(!strcmp(p,"hull1972a4"))
         ||(!strcmp(p,"hull1972a5")) )  PetscFunctionReturn(1);
  else if  (!strcmp(p,"hull1972b1")  )  PetscFunctionReturn(2);
  else if ((!strcmp(p,"hull1972b2"))
         ||(!strcmp(p,"hull1972b3")) 
         ||(!strcmp(p,"hull1972b4")) 
         ||(!strcmp(p,"hull1972b5")) )  PetscFunctionReturn(3);
  else if ((!strcmp(p,"hull1972c1"))
         ||(!strcmp(p,"hull1972c2")) 
         ||(!strcmp(p,"hull1972c3")) )  PetscFunctionReturn(10);
  else if  (!strcmp(p,"hull1972c4")  )  PetscFunctionReturn(51);
  else                                  PetscFunctionReturn(-1);
}

#undef __FUNCT__
#define __FUNCT__ "Initialize"
/* Sets the initial solution for the IVP */
PetscErrorCode Initialize(Vec Y, void* s)
{
  PetscErrorCode ierr;
  char          *p = (char*) s;
  PetscScalar   *y;

  PetscFunctionBegin;
  VecZeroEntries(Y);
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  if     ((!strcmp(p,"hull1972a1"))
        ||(!strcmp(p,"hull1972a2"))
        ||(!strcmp(p,"hull1972a3"))
        ||(!strcmp(p,"hull1972a4"))) {
    y[0] = 1.0;
  } else if (!strcmp(p,"hull1972a5")) {
    y[0] = 4.0;  
  } else if (!strcmp(p,"hull1972b1")) {
    y[0] = 1.0;
    y[1] = 3.0;
  } else if (!strcmp(p,"hull1972b2")) {
    y[0] = 2.0;
    y[1] = 0.0;
    y[2] = 1.0;
  } else if (!strcmp(p,"hull1972b3")) {
    y[0] = 1.0;
    y[1] = 0.0;
    y[2] = 0.0;
  } else if (!strcmp(p,"hull1972b4")) {
    y[0] = 3.0;
    y[1] = 0.0;
    y[2] = 0.0;
  } else if (!strcmp(p,"hull1972b5")) {
    y[0] = 0.0;
    y[1] = 1.0;
    y[2] = 1.0;
  } else if ((!strcmp(p,"hull1972c1"))
           ||(!strcmp(p,"hull1972c2")) 
           ||(!strcmp(p,"hull1972c3")) 
           ||(!strcmp(p,"hull1972c4")) ) {
    y[0] = 1.0;
  }
  PetscInt N = GetSize(s);
  PetscBool flg;
  ierr = PetscOptionsGetRealArray(PETSC_NULL,"-yinit",y,&N,&flg);CHKERRQ(ierr);
  if ((N != GetSize(s)) && flg) { 
    printf("Error: number of initial values %d does not match problem size %d.\n",N,GetSize(s));
  }
  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RHSFunction"
/* Calculates F = f(y) in ydot = f(y) for explicit time-integration */
PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec Y, Vec F, void *s)
{
  PetscErrorCode  ierr;
  char           *p = (char*) s;
  PetscScalar    *y,*f;
  PetscInt        N;

  PetscFunctionBegin;
  ierr = VecGetSize (Y,&N);CHKERRQ(ierr);
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  if (!strcmp(p,"hull1972a1")) {
    f[0] = -y[0]; 
  } else if (!strcmp(p,"hull1972a2")) {
    f[0] = -0.5*y[0]*y[0]*y[0];
  } else if (!strcmp(p,"hull1972a3")) {
    f[0] = y[0]*cos(t);
  } else if (!strcmp(p,"hull1972a4")) {
    f[0] = (0.25*y[0])*(1.0-0.05*y[0]); 
  } else if (!strcmp(p,"hull1972a5")) {
    f[0] = (y[0]-t)/(y[0]+t);
  } else if (!strcmp(p,"hull1972b1")) {
    f[0] = 2.0*(y[0] - y[0]*y[1]);
    f[1] = -(y[1]-y[0]*y[1]);
  } else if (!strcmp(p,"hull1972b2")) {
    f[0] = -y[0] + y[1];
    f[1] = y[0] - 2*y[1] + y[2];
    f[2] = y[1] - y[2];
  } else if (!strcmp(p,"hull1972b3")) {
    f[0] = -y[0];
    f[1] = y[0]-y[1]*y[1];
    f[2] = y[1]*y[1];
  } else if (!strcmp(p,"hull1972b4")) {
    f[0] = -y[1] - y[0]*y[2]/sqrt(y[0]*y[0]+y[1]*y[1]);
    f[1] =  y[0] - y[1]*y[2]/sqrt(y[0]*y[0]+y[1]*y[1]);
    f[2] = y[0]/sqrt(y[0]*y[0]+y[1]*y[1]);
  } else if (!strcmp(p,"hull1972b5")) {
    f[0] = y[1]*y[2];
    f[1] = -y[0]*y[2];
    f[2] = -0.51*y[0]*y[1];
  } else if (!strcmp(p,"hull1972c1")) {
    PetscInt i;
    f[0] = -y[0];
    for (i = 1; i < N-1; i++) {
      f[i] = y[i-1] - y[i];
    }
    f[N-1] = y[N-2];
  } else if (!strcmp(p,"hull1972c2")) {
    PetscInt i;
    f[0] = -y[0];
    for (i = 1; i < N-1; i++) {
      f[i] = (PetscReal)i*y[i-1] - (PetscReal)(i+1)*y[i];
    }
    f[N-1] = (PetscReal)(N-1)*y[N-2];
  } else if ((!strcmp(p,"hull1972c3")) 
           ||(!strcmp(p,"hull1972c4"))) {
    PetscInt i;
    f[0] = -2.0*y[0] + y[1];
    for (i = 1; i < N-1; i++) {
      f[i] = y[i-1] - 2*y[i] + y[i+1];
    }
    f[N-1] = y[N-2] - 2*y[N-1];
  } else {
    /* Invalid problem specifications */
    VecZeroEntries(F);
  }
  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IFunction"
/* Calculates F = ydot - f(y) for implicit time-integration */
PetscErrorCode IFunction(TS ts, PetscReal t, Vec Y, Vec Ydot, Vec F, void *s)
{
  PetscErrorCode  ierr;
  char           *p = (char*) s;
  PetscScalar    *y,*f;
  PetscInt        N;

  PetscFunctionBegin;
  ierr = VecGetSize (Y,&N);CHKERRQ(ierr);
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  if (!strcmp(p,"hull1972a1")) {
    f[0] = -y[0]; 
  } else if (!strcmp(p,"hull1972a2")) {
    f[0] = -0.5*y[0]*y[0]*y[0];
  } else if (!strcmp(p,"hull1972a3")) {
    f[0] = y[0]*cos(t);
  } else if (!strcmp(p,"hull1972a4")) {
    f[0] = (0.25*y[0])*(1.0-0.05*y[0]); 
  } else if (!strcmp(p,"hull1972a5")) {
    f[0] = (y[0]-t)/(y[0]+t);
  } else if (!strcmp(p,"hull1972b1")) {
    f[0] = 2.0*(y[0] - y[0]*y[1]);
    f[1] = -(y[1]-y[0]*y[1]);
  } else if (!strcmp(p,"hull1972b2")) {
    f[0] = -y[0] + y[1];
    f[1] = y[0] - 2*y[1] + y[2];
    f[2] = y[1] - y[2];
  } else if (!strcmp(p,"hull1972b3")) {
    f[0] = -y[0];
    f[1] = y[0]-y[1]*y[1];
    f[2] = y[1]*y[1];
  } else if (!strcmp(p,"hull1972b4")) {
    f[0] = -y[1] - y[0]*y[2]/sqrt(y[0]*y[0]+y[1]*y[1]);
    f[1] =  y[0] - y[1]*y[2]/sqrt(y[0]*y[0]+y[1]*y[1]);
    f[2] = y[0]/sqrt(y[0]*y[0]+y[1]*y[1]);
  } else if (!strcmp(p,"hull1972b5")) {
    f[0] = y[1]*y[2];
    f[1] = -y[0]*y[2];
    f[2] = -0.51*y[0]*y[1];
  } else if (!strcmp(p,"hull1972c1")) {
    PetscInt i;
    f[0] = -y[0];
    for (i = 1; i < N-1; i++) {
      f[i] = y[i-1] - y[i];
    }
    f[N-1] = y[N-2];
  } else if (!strcmp(p,"hull1972c2")) {
    PetscInt i;
    f[0] = -y[0];
    for (i = 1; i < N-1; i++) {
      f[i] = (PetscReal)i*y[i-1] - (PetscReal)(i+1)*y[i];
    }
    f[N-1] = (PetscReal)(N-1)*y[N-2];
  } else if ((!strcmp(p,"hull1972c3")) 
           ||(!strcmp(p,"hull1972c4"))) {
    PetscInt i;
    f[0] = -2.0*y[0] + y[1];
    for (i = 1; i < N-1; i++) {
      f[i] = y[i-1] - 2*y[i] + y[i+1];
    }
    f[N-1] = y[N-2] - 2*y[N-1];
  } else {
    /* Invalid problem specifications */
    VecZeroEntries(F);
  }
  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  /* Left hand side = ydot - f(y) */
  ierr = VecAYPX(F,-1.0,Ydot);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "IJacobian"
/* Calculates the Jacobian A = [a{dF/d(ydot)} - dF/dy] where F = ydot - f(y) for implicit time integration*/
PetscErrorCode IJacobian(TS ts, PetscReal t, Vec Y, Vec Ydot, PetscReal a, Mat *A, Mat *B, MatStructure *flag, void *s)
{
  PetscErrorCode  ierr;
  char           *p = (char*) s;
  PetscScalar    *y;
  PetscInt        N;

  PetscFunctionBegin;
  ierr = VecGetSize (Y,&N);CHKERRQ(ierr);
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  if (!strcmp(p,"hull1972a1")) {
    PetscInt  row   = 0;
    PetscInt  col   = 0;
    PetscReal value = a - 1.0;
    ierr = MatSetValues(*A,1,&row,1,&col,&value,INSERT_VALUES);CHKERRQ(ierr);
  } else if (!strcmp(p,"hull1972a2")) {
    PetscInt  row   = 0;
    PetscInt  col   = 0;
    PetscReal value = a + 0.5*3.0*y[0]*y[0];
    ierr = MatSetValues(*A,1,&row,1,&col,&value,INSERT_VALUES);CHKERRQ(ierr);
  } else if (!strcmp(p,"hull1972a3")) {
    PetscInt  row   = 0;
    PetscInt  col   = 0;
    PetscReal value = a - cos(t);
    ierr = MatSetValues(*A,1,&row,1,&col,&value,INSERT_VALUES);CHKERRQ(ierr);
  } else if (!strcmp(p,"hull1972a4")) {
    PetscInt  row   = 0;
    PetscInt  col   = 0;
    PetscReal value = a - 0.25*(1.0-0.05*y[0]) + (0.25*y[0])*0.05;
    ierr = MatSetValues(*A,1,&row,1,&col,&value,INSERT_VALUES);CHKERRQ(ierr);
  } else if (!strcmp(p,"hull1972a5")) {
    PetscInt  row = 0;
    PetscInt  col = 0;
    PetscReal value = a - 2*t/((t+y[0])*(t+y[0]));
    ierr = MatSetValues(*A,1,&row,1,&col,&value,INSERT_VALUES);CHKERRQ(ierr);
  } else if (!strcmp(p,"hull1972b1")) {
    PetscInt  row[2] = {0,1};
    PetscReal value[2][2];
    value[0][0] = a - 2.0*(1.0-y[1]);    value[0][1] = 2.0*y[0];
    value[1][0] = -y[1];                 value[1][1] = a + 1.0 - y[0];
    ierr = MatSetValues(*A,2,&row[0],2,&row[0],&value[0][0],INSERT_VALUES);CHKERRQ(ierr);
  } else if (!strcmp(p,"hull1972b2")) {
    PetscInt  row[3] = {0,1,2};
    PetscReal value[3][3];
    value[0][0] = a + 1.0;  value[0][1] = -1.0;     value[0][2] = 0;
    value[1][0] = -1.0;     value[1][1] = a + 2.0;  value[1][2] = -1.0;
    value[2][0] = 0;        value[2][1] = -1.0;     value[2][2] = a + 1.0;
    ierr = MatSetValues(*A,3,&row[0],3,&row[0],&value[0][0],INSERT_VALUES);CHKERRQ(ierr);
  } else if (!strcmp(p,"hull1972b3")) {
    PetscInt  row[3] = {0,1,2};
    PetscReal value[3][3];
    value[0][0] = a + 1.0; value[0][1] = 0;             value[0][2] = 0;
    value[1][0] = -1.0;    value[1][1] = a + 2.0*y[1];  value[1][2] = 0;
    value[2][0] = 0;       value[2][1] = -2.0*y[1];     value[2][2] = a;
    ierr = MatSetValues(*A,3,&row[0],3,&row[0],&value[0][0],INSERT_VALUES);CHKERRQ(ierr);
  } else if (!strcmp(p,"hull1972b4")) {
    PetscInt  row[3] = {0,1,2};
    PetscReal value[3][3];
    PetscReal fac  = PetscPowReal(y[0]*y[0]+y[1]*y[1],-1.5);
    PetscReal fac2 = PetscPowReal(y[0]*y[0]+y[1]*y[1],-0.5);
    value[0][0] = a + (y[1]*y[1]*y[2])*fac; 
    value[0][1] = 1.0 - (y[0]*y[1]*y[2])*fac; 
    value[0][2] = y[0]*fac2;
    value[1][0] = -1.0 - y[0]*y[1]*y[2]*fac; 
    value[1][1] = a + y[0]*y[0]*y[2]*fac; 
    value[1][2] = y[1]*fac2;
    value[2][0] = -y[1]*y[1]*fac; 
    value[2][1] = y[0]*y[1]*fac; 
    value[2][2] = a;
    ierr = MatSetValues(*A,3,&row[0],3,&row[0],&value[0][0],INSERT_VALUES);CHKERRQ(ierr);
  } else if (!strcmp(p,"hull1972b5")) {
    PetscInt  row[3] = {0,1,2};
    PetscReal value[3][3];
    value[0][0] = a;          value[0][1] = -y[2];      value[0][2] = -y[1];
    value[1][0] = y[2];       value[1][1] = a;          value[1][2] = y[0];
    value[2][0] = 0.51*y[1];  value[2][1] = 0.51*y[0];  value[2][2] = a;
    ierr = MatSetValues(*A,3,&row[0],3,&row[0],&value[0][0],INSERT_VALUES);CHKERRQ(ierr);
  } else if (!strcmp(p,"hull1972c1")) {
    PetscInt  i;
    PetscInt  col[2];
    PetscReal value[2];
    for (i = 0; i < N; i++) {
      if (i == 0) {
        value[0] = a-1; col[0] = i;
        value[1] =  0;  col[1] = i+1;
      } else if (i == N-1) {
        value[0] = -1;  col[0] = i-1;
        value[1] =  a;  col[1] = i;
      } else { 
        value[0] =  -1; col[0] = i-1;
        value[1] = a+1; col[1] = i;
      }
      ierr = MatSetValues(*A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
  } else if (!strcmp(p,"hull1972c2")) {
    PetscInt  i;
    PetscInt  col[2];
    PetscReal value[2];
    for (i = 0; i < N; i++) {
      if (i == 0) {
        value[0] = a-(PetscReal)(-i-1); col[0] = i;
        value[1] = 0;                   col[1] = i+1;
      } else if (i == N-1) {
        value[0] = -(PetscReal) i;      col[0] = i-1;
        value[1] = a;                   col[1] = i;
      } else { 
        value[0] = -(PetscReal) i;      col[0] = i-1;
        value[1] = a-(PetscReal)(-1-1); col[1] = i;
      }
      ierr = MatSetValues(*A,1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
  } else if ((!strcmp(p,"hull1972c3")) 
          || (!strcmp(p,"hull1972c4"))) {
    PetscInt  i;
    PetscInt  col[3];
    PetscReal value[3];
    for (i = 0; i < N; i++) {
      if (i == 0) {
        value[0] = a+2;  col[0] = i;
        value[1] =  -1;  col[1] = i+1;
        value[2] =  0;   col[2] = i+2;
      } else if (i == N-1) {
        value[0] =  0;   col[0] = i-2;
        value[1] =  -1;  col[1] = i-1;
        value[2] = a+2;  col[2] = i;
      } else { 
        value[0] = -1;   col[0] = i-1;
        value[1] = a+2;  col[1] = i;
        value[2] = -1;   col[2] = i+1;
      }
      ierr = MatSetValues(*A,1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd  (*A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "ExactSolution"
/* Calculates the exact solution to problems that have one */
PetscErrorCode ExactSolution(Vec Y, void* s, PetscReal t, PetscBool *flag)
{
  PetscErrorCode ierr;
  char          *p = (char*) s;
  PetscScalar   *y;

  PetscFunctionBegin;
  if (!strcmp(p,"hull1972a1")) {
    ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
    y[0] = exp(-t);
    *flag = PETSC_TRUE;
    ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  } else if (!strcmp(p,"hull1972a2")) {
    ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
    y[0] = 1.0/sqrt(t+1);
    *flag = PETSC_TRUE;
    ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  } else if (!strcmp(p,"hull1972a3")) {
    ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
    y[0] = exp(sin(t));
    *flag = PETSC_TRUE;
    ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  } else if (!strcmp(p,"hull1972a4")) {
    ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
    y[0] = 20.0/(1+19.0*exp(-t/4.0));
    *flag = PETSC_TRUE;
    ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  } else {
    ierr = VecSet(Y,0);CHKERRQ(ierr);
    *flag = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}

