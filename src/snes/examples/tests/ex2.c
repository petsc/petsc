#ifndef lint
static char vcid[] = "$Id: ex2.c,v 1.12 1995/05/03 13:21:59 bsmith Exp bsmith $";
#endif

static char help[] = "Uses Newton method to solve a two variable system\n";


#include "snes.h"

int  FormJacobian(SNES snes,Vec,Mat*,Mat*,int*,void*),
     FormFunction(SNES snes,Vec,Vec,void*),
     FormInitialGuess(SNES snes,Vec,void*),
     Monitor(SNES,int,double,void *);

int main( int argc, char **argv )
{
  SNES         snes;
  SNESMethod   method = SNES_NLS;  /* nonlinear solution method */
  Vec          x,r;
  Mat          J;
  int          ierr, its; 

  PetscInitialize( &argc, &argv, 0,0 );

  if (OptionsHasName(0,0,"-help")) fprintf(stderr,"%s",help);
  ierr = VecCreateSequential(MPI_COMM_SELF,2,&x); CHKERRA(ierr);
  ierr = VecDuplicate(x,&r); CHKERRA(ierr);
  ierr = MatCreateSequentialDense(MPI_COMM_SELF,2,2,&J); CHKERRA(ierr);

  ierr = SNESCreate(MPI_COMM_WORLD,&snes); CHKERRA(ierr);
  ierr = SNESSetMethod(snes,method); CHKERRA(ierr);
  ierr = SNESSetMonitor(snes,Monitor,0);
  ierr = SNESSetFromOptions(snes); CHKERR(ierr);

  /* Set various routines */
  SNESSetSolution( snes, x,FormInitialGuess,0 );
  SNESSetFunction( snes, r,FormFunction,0, 0 );
  SNESSetJacobian( snes, J, J, FormJacobian,0 );	

  SNESSetUp( snes );				       

  /* Execute solution method */
  ierr = SNESSolve( snes,&its );				       
  printf( "number of Newton iterations = %d\n\n", its );

  VecDestroy(x);
  VecDestroy(r);
  MatDestroy(J);
  SNESDestroy( snes );				       
  PetscFinalize();

  return 0;
}
/* ------------------------------------------------ */
/*
    Evaluate Function F(x).
 */

int FormFunction(SNES snes,Vec x,Vec  f,void *dummy )
{
   Scalar *xx, *ff;
   VecGetArray(x,&xx); VecGetArray(f,&ff);
   ff[0] = xx[0]*xx[0] + xx[0]*xx[1] - 3.0;
   ff[1] = xx[0]*xx[1] + xx[1]*xx[1] - 6.0;
   return 0;
}
/* ------------------------------------------------ */
/*
    Form initial approximation.
 */
int FormInitialGuess(SNES snes,Vec x,void *dummy)
{
   Scalar pfive = .50;
   VecSet(&pfive,x);
   return 0;
}
/* ------------------------------------------------ */
/*
   Evaluate Jacobian matrix F'(x).
 */
int FormJacobian(SNES snes,Vec x,Mat *jac,Mat *B, int *flag,void *dummy)
{
  Scalar *xx, A[4];
  int    idx[2] = {0,1};
  VecGetArray(x,&xx);
  A[0] = 2.0*xx[0] + xx[1]; A[1] = xx[0];
  A[2] = xx[1]; A[3] = xx[0] + 2.0*xx[1];
  MatSetValues(*jac,2,idx,2,idx,A,INSERTVALUES);
  *flag = 0;
  return 0;
}

int Monitor(SNES snes,int its,double fnorm,void *dummy)
{
  Vec x;
  fprintf( stdout, "iter = %d, Function norm %g \n",its,fnorm);
  SNESGetSolution(snes,&x);
  VecView(x,STDOUT_VIEWER);
  return 0;
}
