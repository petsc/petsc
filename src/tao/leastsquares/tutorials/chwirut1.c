/*
   Include "petsctao.h" so that we can use TAO solvers.  Note that this
   file automatically includes libraries such as:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - system routines        petscmat.h - matrices
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

/*T
   Concepts: TAO^Solving a system of nonlinear equations, nonlinear least squares
   Routines: TaoCreate();
   Routines: TaoSetType();
   Routines: TaoSetSeparableObjectiveRoutine();
   Routines: TaoSetJacobianRoutine();
   Routines: TaoSetSolution();
   Routines: TaoSetFromOptions();
   Routines: TaoSetConvergenceHistory(); TaoGetConvergenceHistory();
   Routines: TaoSolve();
   Routines: TaoView(); TaoDestroy();
   Processors: 1
T*/

#define NOBSERVATIONS 214
#define NPARAMETERS 3

/* User-defined application context */
typedef struct {
  /* Working space */
  PetscReal t[NOBSERVATIONS];   /* array of independent variables of observation */
  PetscReal y[NOBSERVATIONS];   /* array of dependent variables */
  PetscReal j[NOBSERVATIONS][NPARAMETERS]; /* dense jacobian matrix array*/
  PetscInt idm[NOBSERVATIONS];  /* Matrix indices for jacobian */
  PetscInt idn[NPARAMETERS];
} AppCtx;

/* User provided Routines */
PetscErrorCode InitializeData(AppCtx *user);
PetscErrorCode FormStartingPoint(Vec);
PetscErrorCode EvaluateFunction(Tao, Vec, Vec, void *);
PetscErrorCode EvaluateJacobian(Tao, Vec, Mat, Mat, void *);

/*--------------------------------------------------------------------*/
int main(int argc,char **argv)
{
  PetscErrorCode ierr;           /* used to check for functions returning nonzeros */
  Vec            x, f;               /* solution, function */
  Mat            J;                  /* Jacobian matrix */
  Tao            tao;                /* Tao solver context */
  PetscInt       i;               /* iteration information */
  PetscReal      hist[100],resid[100];
  PetscInt       lits[100];
  AppCtx         user;               /* user-defined work context */

  ierr = PetscInitialize(&argc,&argv,(char *)0,help);if (ierr) return ierr;
  /* Allocate vectors */
  CHKERRQ(VecCreateSeq(MPI_COMM_SELF,NPARAMETERS,&x));
  CHKERRQ(VecCreateSeq(MPI_COMM_SELF,NOBSERVATIONS,&f));

  /* Create the Jacobian matrix. */
  CHKERRQ(MatCreateSeqDense(MPI_COMM_SELF,NOBSERVATIONS,NPARAMETERS,NULL,&J));

  for (i=0;i<NOBSERVATIONS;i++) user.idm[i] = i;

  for (i=0;i<NPARAMETERS;i++) user.idn[i] = i;

  /* Create TAO solver and set desired solution method */
  CHKERRQ(TaoCreate(PETSC_COMM_SELF,&tao));
  CHKERRQ(TaoSetType(tao,TAOPOUNDERS));

 /* Set the function and Jacobian routines. */
  CHKERRQ(InitializeData(&user));
  CHKERRQ(FormStartingPoint(x));
  CHKERRQ(TaoSetSolution(tao,x));
  CHKERRQ(TaoSetResidualRoutine(tao,f,EvaluateFunction,(void*)&user));
  CHKERRQ(TaoSetJacobianResidualRoutine(tao, J, J, EvaluateJacobian, (void*)&user));

  /* Check for any TAO command line arguments */
  CHKERRQ(TaoSetFromOptions(tao));

  CHKERRQ(TaoSetConvergenceHistory(tao,hist,resid,0,lits,100,PETSC_TRUE));
  /* Perform the Solve */
  CHKERRQ(TaoSolve(tao));

  /* View the vector; then destroy it.  */
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_SELF));

  /* Free TAO data structures */
  CHKERRQ(TaoDestroy(&tao));

   /* Free PETSc data structures */
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&f));
  CHKERRQ(MatDestroy(&J));

  ierr = PetscFinalize();
  return ierr;
}

/*--------------------------------------------------------------------*/
PetscErrorCode EvaluateFunction(Tao tao, Vec X, Vec F, void *ptr)
{
  AppCtx         *user = (AppCtx *)ptr;
  PetscInt       i;
  const PetscReal *x;
  PetscReal      *y=user->y,*f,*t=user->t;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArray(F,&f));

  for (i=0;i<NOBSERVATIONS;i++) {
    f[i] = y[i] - PetscExpScalar(-x[0]*t[i])/(x[1] + x[2]*t[i]);
  }
  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscLogFlops(6*NOBSERVATIONS);
  PetscFunctionReturn(0);
}

/*------------------------------------------------------------*/
/* J[i][j] = df[i]/dt[j] */
PetscErrorCode EvaluateJacobian(Tao tao, Vec X, Mat J, Mat Jpre, void *ptr)
{
  AppCtx         *user = (AppCtx *)ptr;
  PetscInt       i;
  const PetscReal *x;
  PetscReal      *t=user->t;
  PetscReal      base;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(X,&x));
  for (i=0;i<NOBSERVATIONS;i++) {
    base = PetscExpScalar(-x[0]*t[i])/(x[1] + x[2]*t[i]);

    user->j[i][0] = t[i]*base;
    user->j[i][1] = base/(x[1] + x[2]*t[i]);
    user->j[i][2] = base*t[i]/(x[1] + x[2]*t[i]);
  }

  /* Assemble the matrix */
  CHKERRQ(MatSetValues(J,NOBSERVATIONS,user->idm, NPARAMETERS, user->idn,(PetscReal *)user->j,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY));

  CHKERRQ(VecRestoreArrayRead(X,&x));
  PetscLogFlops(NOBSERVATIONS * 13);
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------ */
PetscErrorCode FormStartingPoint(Vec X)
{
  PetscReal      *x;

  PetscFunctionBegin;
  CHKERRQ(VecGetArray(X,&x));
  x[0] = 0.15;
  x[1] = 0.008;
  x[2] = 0.010;
  CHKERRQ(VecRestoreArray(X,&x));
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

/*TEST

   build:
      requires: !complex !single

   test:
      args: -tao_smonitor -tao_max_it 100 -tao_type pounders -tao_gatol 1.e-5

   test:
      suffix: 2
      args: -tao_smonitor -tao_max_it 100 -tao_type brgn -tao_brgn_regularization_type l2prox -tao_brgn_regularizer_weight 1e-4 -tao_gatol 1.e-5

   test:
      suffix: 3
      args: -tao_smonitor -tao_max_it 100 -tao_type brgn -tao_brgn_regularization_type l1dict -tao_brgn_regularizer_weight 1e-4 -tao_brgn_l1_smooth_epsilon 1e-6 -tao_gatol 1.e-5

   test:
      suffix: 4
      args: -tao_smonitor -tao_max_it 100 -tao_type brgn -tao_brgn_regularization_type lm -tao_gatol 1.e-5 -tao_brgn_subsolver_tao_type bnls

TEST*/
