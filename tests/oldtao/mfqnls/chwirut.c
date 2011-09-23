/* Program usage: rat43 [-help] [all TAO options] */

/* 
   Include "tao.h" so that we can use TAO solvers.  Note that this
   file automatically includes libraries such as:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - sysem routines        petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners

*/

#include "tao.h"
#include <math.h>  /*  For pow(), fabs(), log(), and exp()  */


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
   Concepts: TAO - Solving a system of nonlinear equations, nonlinear ;east squares
   Routines: TaoInitialize(); TaoFinalize(); 
   Routines: TaoCreate(); TaoDestroy();
   Routines: TaoPetscApplicationCreate(); TaoApplicationDestroy();
   Routines: TaoSetPetscFunction(); 
   Routines: TaoSetPetscConstraintsFunction(); TaoSetPetscJacobian(); 
   Routines: TaoSetPetscInitialVector();
   Routines: TaoSetApplication(); TaoSetFromOptions(); TaoSolve(); TaoView(); 
   Processors: 1
T*/

#define NOBSERVATIONS 214
#define NPARAMETERS 3

/* User-defined application context */
typedef struct {
  /* Working space */
  double t[NOBSERVATIONS];   /* array of independent variables of observation */
  double y[NOBSERVATIONS];   /* array of dependent variables */
  double j[NOBSERVATIONS][NPARAMETERS]; /* dense jacobian matrix array*/
  int idm[NOBSERVATIONS];  /* Matrix indices for jacobian */
  int idn[NPARAMETERS];
} AppCtx;

/* User provided Routines */
int InitializeData(AppCtx *user);
int FormStartingPoint(Vec);
int EvaluateConstraints(TAO_SOLVER, Vec, Vec, void *);
int EvaluateJacobian(TAO_SOLVER, Vec, Mat*, void *);


/*--------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  int        info;               /* used to check for functions returning nonzeros */
  Vec        x, r;               /* solution, function */
  Mat        J;                  /* Jacobian matrix */ 
  TAO_SOLVER tao;                /* TAO_SOLVER solver context */
  TAO_APPLICATION taoapp;        /* The PETSc application */
  int        iter,i;               /* iteration information */
  double     ff;
  double     zero = 0.0, one = 1.0;
  AppCtx     user;               /* user-defined work context */

   /* Initialize TAO and PETSc */
  PetscInitialize(&argc,&argv,(char *)0,help);
  TaoInitialize(&argc,&argv,(char *)0,help);


  /* Allocate vectors */
  info = VecCreateSeq(MPI_COMM_SELF,NPARAMETERS,&x); CHKERRQ(info);
  info = VecCreateSeq(MPI_COMM_SELF,NOBSERVATIONS,&r); CHKERRQ(info);

  /* Create the Jacobian matrix. */
  info = MatCreateSeqDense(MPI_COMM_SELF,NOBSERVATIONS,NPARAMETERS,PETSC_NULL,&J);  
  CHKERRQ(info);

  for (i=0;i<NOBSERVATIONS;i++)
    user.idm[i] = i;

  for (i=0;i<NPARAMETERS;i++)
    user.idn[i] = i;

  info = InitializeData(&user); CHKERRQ(info);


  /* TAO code begins here */

  /* Create TAO solver and set desired solution method */
  info = TaoCreate(MPI_COMM_SELF,"tao_nlsq",&tao);CHKERRQ(info);
  info = TaoPetscApplicationCreate(MPI_COMM_SELF,&taoapp); CHKERRQ(info);


  /* Set the function and Jacobian routines. */
  info = TaoSetPetscFunction(taoapp,x,TAO_NULL,TAO_NULL); CHKERRQ(info);
  info = TaoSetPetscJacobian(taoapp, J, EvaluateJacobian, (void*)&user);  CHKERRQ(info);
  info = TaoSetPetscConstraintsFunction(taoapp, r, EvaluateConstraints, (void*)&user); CHKERRQ(info);


  /*  Compute the starting point. */
  info = FormStartingPoint(x); CHKERRQ(info);
  info = TaoSetPetscInitialVector(taoapp,x); CHKERRQ(info);

  /* Now that the PETSc application is set, attach to TAO context */
  info = TaoSetApplication(tao,taoapp); CHKERRQ(info); 

  /* Check for any TAO command line arguments */
  info = TaoSetFromOptions(tao); CHKERRQ(info);

  /* Perform the Solve */
  info = TaoSolve(tao); CHKERRQ(info);

  /* View iteration data */
  info = TaoGetIterationData(tao,&iter,&ff,0,0,0,0); CHKERRQ(info);
  PetscPrintf(PETSC_COMM_SELF,"Solved: Iterations: %D, Residual: %5.3e\n",
	      iter,ff);

  /* Use VecView to print x to screen 
     info = VecView(x,PETSC_VIEWER_STDOUT_SELF); CHKERRQ(info);
  */

  /* Free TAO data structures */
  info = TaoDestroy(tao); CHKERRQ(info);
  info = TaoApplicationDestroy(taoapp); CHKERRQ(info);

   /* Free PETSc data structures */
  info = VecDestroy(&x); CHKERRQ(info);
  info = VecDestroy(&r); CHKERRQ(info);
  info = MatDestroy(&J); CHKERRQ(info);


  /* Finalize TAO */
  TaoFinalize();
  PetscFinalize();
  return 0;     
}




/*--------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "EvaluateConstraints"
int EvaluateConstraints(TAO_SOLVER tao, Vec X, Vec R, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;
  int i;
  double *y=user->y,*x,*r,*t=user->t;
  int info;
  double onedx3;

  TaoFunctionBegin;

  /* Get handles to the Vectors */
  info = VecGetArray(X,&x); CHKERRQ(info);
  info = VecGetArray(R,&r); CHKERRQ(info);



  for (i=0;i<NOBSERVATIONS;i++) {
    r[i] = y[i] - exp(-x[0]*t[i])/(x[1] + x[2]*t[i]);
  }


  /* Return the handles */
  info = VecRestoreArray(X,&x); CHKERRQ(info);
  info = VecRestoreArray(R,&r); CHKERRQ(info);

  PetscLogFlops(6*NOBSERVATIONS);

  TaoFunctionReturn(0);
}

/*------------------------------------------------------------*/
/* J[i][j] = dr[i]/dt[j] */
#undef __FUNCT__
#define __FUNCT__ "EvaluateJacobian"
int EvaluateJacobian(TAO_SOLVER tao, Vec X, Mat *J, void *ptr)
{
  AppCtx *user = (AppCtx *)ptr;
  int i,info;
  double *x,*y=user->y,*t=user->t;
  double base;

  TaoFunctionBegin;


  /* Get handles to the Vectors */
  info = VecGetArray(X,&x); CHKERRQ(info);



  for (i=0;i<NOBSERVATIONS;i++) {
    base = exp(-x[0]*t[i])/(x[1] + x[2]*t[i]);

    user->j[i][0] = t[i]*base;
    user->j[i][1] = base/(x[1] + x[2]*t[i]);
    user->j[i][2] = base*t[i]/(x[1] + x[2]*t[i]);

  }

  /* Assemble the matrix */
  info = MatSetValues(*J,NOBSERVATIONS,user->idm, NPARAMETERS, user->idn,(double *)user->j,
		      INSERT_VALUES); CHKERRQ(info);
  info = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY); CHKERRQ(info);
  info = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY); CHKERRQ(info);
  
  /* Return the handles */
  info = VecRestoreArray(X,&x); CHKERRQ(info);

  PetscLogFlops(NOBSERVATIONS * 13);

  TaoFunctionReturn(0);
}

/* ------------------------------------------------------------ */
#undef __FUNCT__
#define __FUNCT__ "FormStartingPoint"
int FormStartingPoint(Vec X)
{
  double *x;
  int info;
  
  TaoFunctionBegin;

  info = VecGetArray(X,&x); CHKERRQ(info);

  x[0] = 0.15;
  x[1] = 0.008;
  x[2] = 0.010;

  VecRestoreArray(X,&x); CHKERRQ(info);

  TaoFunctionReturn(0);
}


/* ---------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "InitializeData"
int InitializeData(AppCtx *user)
{
  double *t=user->t,*y=user->y;
  int info,i=0;

  TaoFunctionBegin;


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

  
  TaoFunctionReturn(0);
}
