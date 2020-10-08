/*
   Include "petsctao.h" so that we can use TAO solvers.  Note that this
   file automatically includes libraries such as:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - sysem routines        petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners

*/

#include <petsctao.h>

/*
Description:   These data are the result of a NIST study involving
               ultrasonic calibration.  The response variable is
               ultrasonic response, and the predictor variable is
               metal distance.

Reference:     Chwirut, D., NIST (197?).
               Ultrasonic Reference Block Study.
*/

static char help[]="Finds the nonlinear least-squares solution to the model \n\
            y = exp[-b1*x]/(b2+b3*x)  +  e \n";

/* T
   Concepts: TAO^Solving a system of nonlinear equations, nonlinear least squares
   Routines: TaoCreate();
   Routines: TaoSetType();
   Routines: TaoSetResidualRoutine();
   Routines: TaoSetMonitor();
   Routines: TaoSetInitialVector();
   Routines: TaoSetFromOptions();
   Routines: TaoSolve();
   Routines: TaoDestroy();
   Processors: n
T*/

#define NOBSERVATIONS 214
#define NPARAMETERS 3

#define DIE_TAG 2000
#define IDLE_TAG 1000

/* User-defined application context */
typedef struct {
  /* Working space */
  PetscReal   t[NOBSERVATIONS];   /* array of independent variables of observation */
  PetscReal   y[NOBSERVATIONS];   /* array of dependent variables */
  PetscMPIInt size,rank;
} AppCtx;

/* User provided Routines */
PetscErrorCode InitializeData(AppCtx *user);
PetscErrorCode FormStartingPoint(Vec);
PetscErrorCode EvaluateFunction(Tao, Vec, Vec, void *);
PetscErrorCode TaskWorker(AppCtx *user);
PetscErrorCode StopWorkers(AppCtx *user);
PetscErrorCode RunSimulation(PetscReal *x, PetscInt i, PetscReal*f, AppCtx *user);

/*--------------------------------------------------------------------*/
int main(int argc,char **argv)
{
  PetscErrorCode ierr;           /* used to check for functions returning nonzeros */
  Vec            x, f;               /* solution, function */
  Tao            tao;                /* Tao solver context */
  AppCtx         user;               /* user-defined work context */

   /* Initialize TAO and PETSc */
  ierr = PetscInitialize(&argc,&argv,(char *)0,help);if (ierr) return ierr;
  MPI_Comm_size(MPI_COMM_WORLD,&user.size);
  MPI_Comm_rank(MPI_COMM_WORLD,&user.rank);
  ierr = InitializeData(&user);CHKERRQ(ierr);

  /* Run optimization on rank 0 */
  if (user.rank == 0) {
    /* Allocate vectors */
    ierr = VecCreateSeq(PETSC_COMM_SELF,NPARAMETERS,&x);CHKERRQ(ierr);
    ierr = VecCreateSeq(PETSC_COMM_SELF,NOBSERVATIONS,&f);CHKERRQ(ierr);

    /* TAO code begins here */

    /* Create TAO solver and set desired solution method */
    ierr = TaoCreate(PETSC_COMM_SELF,&tao);CHKERRQ(ierr);
    ierr = TaoSetType(tao,TAOPOUNDERS);CHKERRQ(ierr);

    /* Set the function and Jacobian routines. */
    ierr = FormStartingPoint(x);CHKERRQ(ierr);
    ierr = TaoSetInitialVector(tao,x);CHKERRQ(ierr);
    ierr = TaoSetResidualRoutine(tao,f,EvaluateFunction,(void*)&user);CHKERRQ(ierr);

    /* Check for any TAO command line arguments */
    ierr = TaoSetFromOptions(tao);CHKERRQ(ierr);

    /* Perform the Solve */
    ierr = TaoSolve(tao);CHKERRQ(ierr);

    /* Free TAO data structures */
    ierr = TaoDestroy(&tao);CHKERRQ(ierr);

    /* Free PETSc data structures */
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&f);CHKERRQ(ierr);
    StopWorkers(&user);
  } else {
    TaskWorker(&user);
  }
  ierr = PetscFinalize();
  return ierr;
}

/*--------------------------------------------------------------------*/
PetscErrorCode EvaluateFunction(Tao tao, Vec X, Vec F, void *ptr)
{
  AppCtx         *user = (AppCtx *)ptr;
  PetscInt       i;
  PetscReal      *x,*f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);
  if (user->size == 1) {
    /* Single processor */
    for (i=0;i<NOBSERVATIONS;i++) {
      ierr = RunSimulation(x,i,&f[i],user);CHKERRQ(ierr);
    }
  } else {
    /* Multiprocessor master */
    PetscMPIInt tag;
    PetscInt    finishedtasks,next_task,checkedin;
    PetscReal   f_i=0.0;
    MPI_Status  status;

    next_task=0;
    finishedtasks=0;
    checkedin=0;

    while (finishedtasks < NOBSERVATIONS || checkedin < user->size-1) {
      ierr = MPI_Recv(&f_i,1,MPIU_REAL,MPI_ANY_SOURCE,MPI_ANY_TAG,PETSC_COMM_WORLD,&status);CHKERRMPI(ierr);
      if (status.MPI_TAG == IDLE_TAG) {
        checkedin++;
      } else {

        tag = status.MPI_TAG;
        f[tag] = (PetscReal)f_i;
        finishedtasks++;
      }

      if (next_task<NOBSERVATIONS) {
        ierr = MPI_Send(x,NPARAMETERS,MPIU_REAL,status.MPI_SOURCE,next_task,PETSC_COMM_WORLD);CHKERRMPI(ierr);
        next_task++;

      } else {
        /* Send idle message */
        ierr = MPI_Send(x,NPARAMETERS,MPIU_REAL,status.MPI_SOURCE,IDLE_TAG,PETSC_COMM_WORLD);CHKERRMPI(ierr);
      }
    }
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscLogFlops(6*NOBSERVATIONS);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
PetscErrorCode FormStartingPoint(Vec X)
{
  PetscReal      *x;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  x[0] = 0.15;
  x[1] = 0.008;
  x[2] = 0.010;
  VecRestoreArray(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ---------------------------------------------------------------------- */
PetscErrorCode InitializeData(AppCtx *user)
{
  PetscReal *t=user->t,*y=user->y;
  PetscInt  i=0;

  PetscFunctionBegin;
  y[i] =   92.9000;   t[i++] =  0.5000;
  y[i] =    78.7000;  t[i++] =   0.6250;
  y[i] =    64.2000;  t[i++] =   0.7500;
  y[i] =    64.9000;  t[i++] =   0.8750;
  y[i] =    57.1000;  t[i++] =   1.0000;
  y[i] =    43.3000;  t[i++] =   1.2500;
  y[i] =    31.1000;   t[i++] =  1.7500;
  y[i] =    23.6000;   t[i++] =  2.2500;
  y[i] =    31.0500;   t[i++] =  1.7500;
  y[i] =    23.7750;   t[i++] =  2.2500;
  y[i] =    17.7375;   t[i++] =  2.7500;
  y[i] =    13.8000;   t[i++] =  3.2500;
  y[i] =    11.5875;   t[i++] =  3.7500;
  y[i] =     9.4125;   t[i++] =  4.2500;
  y[i] =     7.7250;   t[i++] =  4.7500;
  y[i] =     7.3500;   t[i++] =  5.2500;
  y[i] =     8.0250;   t[i++] =  5.7500;
  y[i] =    90.6000;   t[i++] =  0.5000;
  y[i] =    76.9000;   t[i++] =  0.6250;
  y[i] =    71.6000;   t[i++] = 0.7500;
  y[i] =    63.6000;   t[i++] =  0.8750;
  y[i] =    54.0000;   t[i++] =  1.0000;
  y[i] =    39.2000;   t[i++] =  1.2500;
  y[i] =    29.3000;   t[i++] = 1.7500;
  y[i] =    21.4000;   t[i++] =  2.2500;
  y[i] =    29.1750;   t[i++] =  1.7500;
  y[i] =    22.1250;   t[i++] =  2.2500;
  y[i] =    17.5125;   t[i++] =  2.7500;
  y[i] =    14.2500;   t[i++] =  3.2500;
  y[i] =     9.4500;   t[i++] =  3.7500;
  y[i] =     9.1500;   t[i++] =  4.2500;
  y[i] =     7.9125;   t[i++] =  4.7500;
  y[i] =     8.4750;   t[i++] =  5.2500;
  y[i] =     6.1125;   t[i++] =  5.7500;
  y[i] =    80.0000;   t[i++] =  0.5000;
  y[i] =    79.0000;   t[i++] =  0.6250;
  y[i] =    63.8000;   t[i++] =  0.7500;
  y[i] =    57.2000;   t[i++] =  0.8750;
  y[i] =    53.2000;   t[i++] =  1.0000;
  y[i] =   42.5000;   t[i++] =  1.2500;
  y[i] =   26.8000;   t[i++] =  1.7500;
  y[i] =    20.4000;   t[i++] =  2.2500;
  y[i] =    26.8500;  t[i++] =   1.7500;
  y[i] =    21.0000;  t[i++] =   2.2500;
  y[i] =    16.4625;  t[i++] =   2.7500;
  y[i] =    12.5250;  t[i++] =   3.2500;
  y[i] =    10.5375;  t[i++] =   3.7500;
  y[i] =     8.5875;  t[i++] =   4.2500;
  y[i] =     7.1250;  t[i++] =   4.7500;
  y[i] =     6.1125;  t[i++] =   5.2500;
  y[i] =     5.9625;  t[i++] =   5.7500;
  y[i] =    74.1000;  t[i++] =   0.5000;
  y[i] =    67.3000;  t[i++] =   0.6250;
  y[i] =    60.8000;  t[i++] =   0.7500;
  y[i] =    55.5000;  t[i++] =   0.8750;
  y[i] =    50.3000;  t[i++] =   1.0000;
  y[i] =    41.0000;  t[i++] =   1.2500;
  y[i] =    29.4000;  t[i++] =   1.7500;
  y[i] =    20.4000;  t[i++] =   2.2500;
  y[i] =    29.3625;  t[i++] =   1.7500;
  y[i] =    21.1500;  t[i++] =   2.2500;
  y[i] =    16.7625;  t[i++] =   2.7500;
  y[i] =    13.2000;  t[i++] =   3.2500;
  y[i] =    10.8750;  t[i++] =   3.7500;
  y[i] =     8.1750;  t[i++] =   4.2500;
  y[i] =     7.3500;  t[i++] =   4.7500;
  y[i] =     5.9625;  t[i++] =  5.2500;
  y[i] =     5.6250;  t[i++] =   5.7500;
  y[i] =    81.5000;  t[i++] =    .5000;
  y[i] =    62.4000;  t[i++] =    .7500;
  y[i] =    32.5000;  t[i++] =   1.5000;
  y[i] =    12.4100;  t[i++] =   3.0000;
  y[i] =    13.1200;  t[i++] =   3.0000;
  y[i] =    15.5600;  t[i++] =   3.0000;
  y[i] =     5.6300;  t[i++] =   6.0000;
  y[i] =    78.0000;   t[i++] =   .5000;
  y[i] =    59.9000;  t[i++] =    .7500;
  y[i] =    33.2000;  t[i++] =   1.5000;
  y[i] =    13.8400;  t[i++] =   3.0000;
  y[i] =    12.7500;  t[i++] =   3.0000;
  y[i] =    14.6200;  t[i++] =   3.0000;
  y[i] =     3.9400;  t[i++] =   6.0000;
  y[i] =    76.8000;  t[i++] =    .5000;
  y[i] =    61.0000;  t[i++] =    .7500;
  y[i] =    32.9000;  t[i++] =   1.5000;
  y[i] =   13.8700;   t[i++] = 3.0000;
  y[i] =    11.8100;  t[i++] =   3.0000;
  y[i] =    13.3100;  t[i++] =   3.0000;
  y[i] =     5.4400;  t[i++] =   6.0000;
  y[i] =    78.0000;  t[i++] =    .5000;
  y[i] =    63.5000;  t[i++] =    .7500;
  y[i] =    33.8000;  t[i++] =   1.5000;
  y[i] =    12.5600;  t[i++] =   3.0000;
  y[i] =     5.6300;  t[i++] =   6.0000;
  y[i] =    12.7500;  t[i++] =   3.0000;
  y[i] =    13.1200;  t[i++] =   3.0000;
  y[i] =     5.4400;  t[i++] =   6.0000;
  y[i] =    76.8000;  t[i++] =    .5000;
  y[i] =    60.0000;  t[i++] =    .7500;
  y[i] =    47.8000;  t[i++] =   1.0000;
  y[i] =    32.0000;  t[i++] =   1.5000;
  y[i] =    22.2000;  t[i++] =   2.0000;
  y[i] =    22.5700;  t[i++] =   2.0000;
  y[i] =    18.8200;  t[i++] =   2.5000;
  y[i] =    13.9500;  t[i++] =   3.0000;
  y[i] =    11.2500;  t[i++] =   4.0000;
  y[i] =     9.0000;  t[i++] =   5.0000;
  y[i] =     6.6700;  t[i++] =   6.0000;
  y[i] =    75.8000;  t[i++] =    .5000;
  y[i] =    62.0000;  t[i++] =    .7500;
  y[i] =    48.8000;  t[i++] =   1.0000;
  y[i] =    35.2000;  t[i++] =   1.5000;
  y[i] =    20.0000;  t[i++] =   2.0000;
  y[i] =    20.3200;  t[i++] =   2.0000;
  y[i] =    19.3100;  t[i++] =   2.5000;
  y[i] =    12.7500;  t[i++] =   3.0000;
  y[i] =    10.4200;  t[i++] =   4.0000;
  y[i] =     7.3100;  t[i++] =   5.0000;
  y[i] =     7.4200;  t[i++] =   6.0000;
  y[i] =    70.5000;  t[i++] =    .5000;
  y[i] =    59.5000;  t[i++] =    .7500;
  y[i] =    48.5000;  t[i++] =   1.0000;
  y[i] =    35.8000;  t[i++] =   1.5000;
  y[i] =    21.0000;  t[i++] =   2.0000;
  y[i] =    21.6700;  t[i++] =   2.0000;
  y[i] =    21.0000;  t[i++] =   2.5000;
  y[i] =    15.6400;  t[i++] =   3.0000;
  y[i] =     8.1700;  t[i++] =   4.0000;
  y[i] =     8.5500;  t[i++] =   5.0000;
  y[i] =    10.1200;  t[i++] =   6.0000;
  y[i] =    78.0000;  t[i++] =    .5000;
  y[i] =    66.0000;  t[i++] =    .6250;
  y[i] =    62.0000;  t[i++] =    .7500;
  y[i] =    58.0000;  t[i++] =    .8750;
  y[i] =    47.7000;  t[i++] =   1.0000;
  y[i] =    37.8000;  t[i++] =   1.2500;
  y[i] =    20.2000;  t[i++] =   2.2500;
  y[i] =    21.0700;  t[i++] =   2.2500;
  y[i] =    13.8700;  t[i++] =   2.7500;
  y[i] =     9.6700;  t[i++] =   3.2500;
  y[i] =     7.7600;  t[i++] =   3.7500;
  y[i] =    5.4400;   t[i++] =  4.2500;
  y[i] =    4.8700;   t[i++] =  4.7500;
  y[i] =     4.0100;  t[i++] =   5.2500;
  y[i] =     3.7500;  t[i++] =   5.7500;
  y[i] =    24.1900;  t[i++] =   3.0000;
  y[i] =    25.7600;  t[i++] =   3.0000;
  y[i] =    18.0700;  t[i++] =   3.0000;
  y[i] =    11.8100;  t[i++] =   3.0000;
  y[i] =    12.0700;  t[i++] =   3.0000;
  y[i] =    16.1200;  t[i++] =   3.0000;
  y[i] =    70.8000;  t[i++] =    .5000;
  y[i] =    54.7000;  t[i++] =    .7500;
  y[i] =    48.0000;  t[i++] =   1.0000;
  y[i] =    39.8000;  t[i++] =   1.5000;
  y[i] =    29.8000;  t[i++] =   2.0000;
  y[i] =    23.7000;  t[i++] =   2.5000;
  y[i] =    29.6200;  t[i++] =   2.0000;
  y[i] =    23.8100;  t[i++] =   2.5000;
  y[i] =    17.7000;  t[i++] =   3.0000;
  y[i] =    11.5500;  t[i++] =   4.0000;
  y[i] =    12.0700;  t[i++] =   5.0000;
  y[i] =     8.7400;  t[i++] =   6.0000;
  y[i] =    80.7000;  t[i++] =    .5000;
  y[i] =    61.3000;  t[i++] =    .7500;
  y[i] =    47.5000;  t[i++] =   1.0000;
   y[i] =   29.0000;  t[i++] =   1.5000;
   y[i] =   24.0000;  t[i++] =   2.0000;
  y[i] =    17.7000;  t[i++] =   2.5000;
  y[i] =    24.5600;  t[i++] =   2.0000;
  y[i] =    18.6700;  t[i++] =   2.5000;
   y[i] =   16.2400;  t[i++] =   3.0000;
  y[i] =     8.7400;  t[i++] =   4.0000;
  y[i] =     7.8700;  t[i++] =   5.0000;
  y[i] =     8.5100;  t[i++] =   6.0000;
  y[i] =    66.7000;  t[i++] =    .5000;
  y[i] =    59.2000;  t[i++] =    .7500;
  y[i] =    40.8000;  t[i++] =   1.0000;
  y[i] =    30.7000;  t[i++] =   1.5000;
  y[i] =    25.7000;  t[i++] =   2.0000;
  y[i] =    16.3000;  t[i++] =   2.5000;
  y[i] =    25.9900;  t[i++] =   2.0000;
  y[i] =    16.9500;  t[i++] =   2.5000;
  y[i] =    13.3500;  t[i++] =   3.0000;
  y[i] =     8.6200;  t[i++] =   4.0000;
  y[i] =     7.2000;  t[i++] =   5.0000;
  y[i] =     6.6400;  t[i++] =   6.0000;
  y[i] =    13.6900;  t[i++] =   3.0000;
  y[i] =    81.0000;  t[i++] =    .5000;
  y[i] =    64.5000;  t[i++] =    .7500;
  y[i] =    35.5000;  t[i++] =   1.5000;
   y[i] =   13.3100;  t[i++] =   3.0000;
  y[i] =     4.8700;  t[i++] =   6.0000;
  y[i] =    12.9400;  t[i++] =   3.0000;
  y[i] =     5.0600;  t[i++] =   6.0000;
  y[i] =    15.1900;  t[i++] =   3.0000;
  y[i] =    14.6200;  t[i++] =   3.0000;
  y[i] =    15.6400;  t[i++] =   3.0000;
  y[i] =    25.5000;  t[i++] =   1.7500;
  y[i] =    25.9500;  t[i++] =   1.7500;
  y[i] =    81.7000;  t[i++] =    .5000;
  y[i] =    61.6000;  t[i++] =    .7500;
  y[i] =    29.8000;  t[i++] =   1.7500;
  y[i] =    29.8100;  t[i++] =   1.7500;
  y[i] =    17.1700;  t[i++] =   2.7500;
  y[i] =    10.3900;  t[i++] =   3.7500;
  y[i] =    28.4000;  t[i++] =   1.7500;
  y[i] =    28.6900;  t[i++] =   1.7500;
  y[i] =    81.3000;  t[i++] =    .5000;
  y[i] =    60.9000;  t[i++] =    .7500;
  y[i] =    16.6500;  t[i++] =   2.7500;
  y[i] =    10.0500;  t[i++] =   3.7500;
  y[i] =    28.9000;  t[i++] =   1.7500;
  y[i] =    28.9500;  t[i++] =   1.7500;
  PetscFunctionReturn(0);
}

PetscErrorCode TaskWorker(AppCtx *user)
{
  PetscReal      x[NPARAMETERS],f = 0.0;
  PetscMPIInt    tag=IDLE_TAG;
  PetscInt       index;
  MPI_Status     status;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Send check-in message to master */

  ierr = MPI_Send(&f,1,MPIU_REAL,0,IDLE_TAG,PETSC_COMM_WORLD);CHKERRMPI(ierr);
  while (tag != DIE_TAG) {
    ierr = MPI_Recv(x,NPARAMETERS,MPIU_REAL,0,MPI_ANY_TAG,PETSC_COMM_WORLD,&status);CHKERRMPI(ierr);
    tag = status.MPI_TAG;
    if (tag == IDLE_TAG) {
      ierr = MPI_Send(&f,1,MPIU_REAL,0,IDLE_TAG,PETSC_COMM_WORLD);CHKERRMPI(ierr);
    } else if (tag != DIE_TAG) {
      index = (PetscInt)tag;
      ierr=RunSimulation(x,index,&f,user);CHKERRQ(ierr);
      ierr=MPI_Send(&f,1,MPIU_REAL,0,tag,PETSC_COMM_WORLD);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode RunSimulation(PetscReal *x, PetscInt i, PetscReal*f, AppCtx *user)
{
  PetscReal *t = user->t;
  PetscReal *y = user->y;
#if defined(PETSC_USE_REAL_SINGLE)
  *f = y[i] - exp(-x[0]*t[i])/(x[1] + x[2]*t[i]); /* expf() for single-precision breaks this example on freebsd, valgrind errors on linux */
#else
  *f = y[i] - PetscExpScalar(-x[0]*t[i])/(x[1] + x[2]*t[i]);
#endif
  return(0);
}

PetscErrorCode StopWorkers(AppCtx *user)
{
  PetscInt       checkedin;
  MPI_Status     status;
  PetscReal      f,x[NPARAMETERS];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  checkedin=0;
  while (checkedin < user->size-1) {
    ierr = MPI_Recv(&f,1,MPIU_REAL,MPI_ANY_SOURCE,MPI_ANY_TAG,PETSC_COMM_WORLD,&status);CHKERRMPI(ierr);
    checkedin++;
    ierr = PetscArrayzero(x,NPARAMETERS);CHKERRQ(ierr);
    ierr = MPI_Send(x,NPARAMETERS,MPIU_REAL,status.MPI_SOURCE,DIE_TAG,PETSC_COMM_WORLD);CHKERRMPI(ierr);
  }
  PetscFunctionReturn(0);
}

/*TEST

   build:
      requires: !complex

   test:
      nsize: 3
      requires: !single
      args: -tao_smonitor -tao_max_it 100 -tao_type pounders -tao_gatol 1.e-5

TEST*/
