static char help[] = "Solves the Hull IVPs using explicit and implicit time-integration methods.\n";

/*
  Concepts:   TS
  Reference:  Hull, T.E., Enright, W.H., Fellen, B.M., and Sedgwick, A.E.,
              "Comparing Numerical Methods for Ordinary Differential
               Equations", SIAM J. Numer. Anal., 9(4), 1972, pp. 603 - 635
  Useful command line parameters:
  -hull_problem <a1>: choose which Hull problem to solve (see reference
                      for complete listing of problems).
*/

#include <petscts.h>

/* Function declarations  */
PetscInt        GetSize     (char);
PetscErrorCode  Initialize  (Vec,void*);
PetscErrorCode  RHSFunction (TS,PetscReal,Vec,Vec,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  PetscErrorCode  ierr;     /* Error code                       */
  TS              ts;       /* time-integrator                  */
  Vec             Y;        /* Solution vector                  */
//  Vec             Yex;      /* Exact solution                   */
  char            ptype[3]; /* Problem specification            */
  PetscInt        N;        /* Size of the system of equations  */
  PetscInt        nproc;    /* No of processors                 */

  /* Initialize program and read options */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&nproc);    CHKERRQ(ierr);
  if (nproc>1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Only for sequential runs");
  ierr = PetscOptionsString("-hull_problem","Problem specification","<a1>",
                            ptype,ptype,sizeof(ptype),PETSC_NULL);CHKERRQ(ierr);

  N = GetSize(ptype[0]);
  if (N < 0) {
    printf("Error: Illegal problem specification.\n");
    return(0);
  }
  ierr = VecCreate(PETSC_COMM_WORLD,&Y);CHKERRQ(ierr);
  ierr = VecSetSizes(Y,N,PETSC_DECIDE); CHKERRQ(ierr);
  ierr = VecZeroEntries(Y);CHKERRQ(ierr);

  /* Initialize the problem */
  ierr = Initialize(Y,&ptype[0]);

  /* Create and initialize the time-integrator                             */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);                       CHKERRQ(ierr);
  ierr = TSSetType(ts,TSRK);                                   CHKERRQ(ierr);
  /* Default duration, number of iterations and time-step size             */
  ierr = TSSetDuration(ts,1000,20.0);                          CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,0.02);                    CHKERRQ(ierr);
  /* Read command line options for time integration                        */
  ierr = TSSetFromOptions(ts);                                 CHKERRQ(ierr);
  /* Set solution vector                                                   */
  ierr = TSSetSolution(ts,Y);                                  CHKERRQ(ierr);
  /* Specify right-hand side function                                      */
  ierr = TSSetRHSFunction(ts,PETSC_NULL,RHSFunction,&ptype[0]);CHKERRQ(ierr);

  /* Solve */
  ierr = TSSolve(ts,Y);CHKERRQ(ierr);

  /* Clean up and finalize */
  ierr = TSDestroy(&ts);  CHKERRQ(ierr);
  ierr = VecDestroy(&Y);  CHKERRQ(ierr);
  ierr = PetscFinalize(); CHKERRQ(ierr);
  return(0);
}

#undef __FUNCT__
#define __FUNCT__ "GetSize"
PetscInt GetSize(char x)
{
  PetscFunctionBegin;
  if      (x == 'a')  return(1);
  else if (x == 'b')  return(3);
  else if (x == 'c')  return(10);
  else                return(-1);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "Initialize"
PetscErrorCode Initialize(Vec Y, void* s)
{
  PetscErrorCode ierr;
  char          *p = (char*) s;
  PetscScalar   *y;

  PetscFunctionBegin;
  ierr = VecGetArray(Y,&y);CHKERRQ(ierr);
  if (p[0] == 'a') {
    if (p[1] == '5') {
      y[0] = 4.0;
    } else {
      y[0] = 1.0;
    }
    ierr = PetscOptionsReal("-y0","Initial value of y(t)",
                            "<1.0> (<4.0> for a5)",
                            y[0],&y[0],PETSC_NULL);CHKERRQ(ierr);
  } else if (p[0] == 'b') {
  } else if (p[0] == 'c') {
  }
  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "RHSFunction"
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
  if (p[0] == 'a') {
    /* Problem class A: Single equations. */
    if        (p[1] == '1') {
      f[0] = -y[0];                         /* Problem A1 */
    } else if (p[1] == '2') {
      f[0] = -0.5*y[0]*y[0]*y[0];           /* Problem A2 */
    } else if (p[1] == '3') {
      f[0] = y[0]*cos(t);                   /* Problem A3 */
    } else if (p[1] == '4') {
      f[0] = (0.25*y[0])*(1.0-0.05*y[0]);   /* Problem A4 */
    } else if (p[1] == '5') {
      f[0] = (y[0]-t)/(y[0]+t);             /* Problem A5 */
    } else {
      f[0] = 0.0;                           /* Invalid problem */
    }
  } else if (p[0] == 'b') {
    /* Problem class B: Small systems.    */
  } else if (p[0] == 'c') {
    /* Problem class C: Moderate systems. */
  } else {
  }
  ierr = VecRestoreArray(Y,&y);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
