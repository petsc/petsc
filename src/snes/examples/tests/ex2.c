#ifndef lint
static char vcid[] = "$Id: ex1.c,v 1.1 1995/03/20 23:28:15 bsmith Exp bsmith $";
#endif

static char help[] = "Uses Newton method to solve x^2 = 2\n";


#include "snes.h"

int  FormJacobian(Vec,Mat*,void*),
     FormResidual(Vec,Vec,void*),
     FormInitialGuess(Vec,void*),
     Monitor(SNES,int, Vec,Vec,double,void *);

int main( int argc, char **argv )
{
  SNES         snes;
  SLES         sles;
  SNESMETHOD   method = SNES_NLS1;  /* nonlinear solution method */
  Vec          x,r;
  Mat          J;
  int          ierr, its; 

  PetscInitialize( &argc, &argv, 0,0 );

  ierr = VecCreateSequential(1,&x); CHKERRA(ierr);
  ierr = VecCreate(x,&r); CHKERRA(ierr);
  ierr = MatCreateSequentialDense(1,1,&J); CHKERRA(ierr);

  ierr = SNESCreate(&snes); CHKERRA(ierr);
  ierr = SNESSetMethod(snes,method); CHKERRA(ierr);
  ierr = SNESSetMonitor(snes,Monitor,0);
  ierr = SNESSetFromOptions(snes); CHKERR(ierr);

  /* Set various routines */
  SNESSetSolution( snes, x,FormInitialGuess,0 );
  SNESSetResidual( snes, r,FormResidual,0, 0 );
  SNESSetJacobian( snes, J, FormJacobian,0 );	

  SNESGetSLES(snes,&sles);
  SLESSetFromOptions(sles);

  SNESSetUp( snes );				       

  /* Execute solution method */
  ierr = SNESSolve( snes,&its );				       
  printf( "number of Newton iterations = %d\n\n", its );

  SNESDestroy( snes );				       

  return 0;
}
/* ------------------------------------------------ */
/*
    Evaluate residual F(x).
 */

int FormResidual(Vec x,Vec  f,void *dummy )
{
   Scalar *xx, *ff;
   VecGetArray(x,&xx); VecGetArray(f,&ff);
   *ff = (*xx)*(*xx) - 2.0;
   return 0;
}
/* ------------------------------------------------ */
/*
    Form initial approximation.
 */
int FormInitialGuess(Vec x,void *dummy)
{
   Scalar one = 1.0;
   VecSet(&one,x);
   return 0;
}
/* ------------------------------------------------ */
/*
   Evaluate Jacobian matrix F'(x).
 */
int FormJacobian(Vec x,Mat *jac,void *dummy)
{
  Scalar *xx, *jj;
  VecGetArray(x,&xx);
  MatGetArray(*jac,&jj);
  *jj = 2.0*(*xx);
  return 0;
}

int Monitor(SNES snes,int its, Vec x,Vec f,double fnorm,void *dummy)
{
  fprintf( stdout, "iter = %d, residual norm %g \n",its,fnorm);
  VecView(x,STDOUT_VIEWER);
  return 0;
}
