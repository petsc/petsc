
static char help[] = 
"This example uses Newton's method to solve a two-variable system.\n\n";

#include "snes.h"

int  FormJacobian(SNES snes,Vec,Mat*,Mat*,MatStructure*,void*),
     FormFunction(SNES snes,Vec,Vec,void*),
     FormInitialGuess(SNES snes,Vec,void*),
     Monitor(SNES,int,double,void *);

int main( int argc, char **argv )
{
  SNES         snes;               /* SNES context */
  Vec          x,r;                /* solution, residual vectors */
  Mat          J;                  /* Jacobian matrix */
  int          ierr, its;

  PetscInitialize( &argc, &argv, 0,0 );
  if (OptionsHasName(0,"-help")) fprintf(stdout,"%s",help);

  /* Set up data structures */
  ierr = VecCreateSequential(MPI_COMM_SELF,2,&x); CHKERRA(ierr);
  ierr = VecDuplicate(x,&r); CHKERRA(ierr);
  ierr = MatCreateSequentialDense(MPI_COMM_SELF,2,2,&J); CHKERRA(ierr);

  /* Create nonlinear solver */
  ierr = SNESCreate(MPI_COMM_WORLD,&snes); CHKERRA(ierr);

  /* Set various routines */
  ierr = SNESSetSolution(snes,x,FormInitialGuess,0); CHKERRA(ierr);
  ierr = SNESSetFunction(snes,r,FormFunction,0,0); CHKERRA(ierr);
  ierr = SNESSetJacobian(snes,J,J,FormJacobian,0); CHKERRA(ierr);
  ierr = SNESSetMonitor(snes,Monitor,0); CHKERRA(ierr);

  /* Set up nonlinear solver; then execute it */
  ierr = SNESSetFromOptions(snes); CHKERRA(ierr);
  ierr = SNESSetUp(snes); CHKERRA(ierr);
  ierr = SNESSolve(snes,&its); CHKERRA(ierr);
  printf("number of Newton iterations = %d\n\n", its);

  /* Free data structures */
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(r); CHKERRA(ierr);
  ierr = MatDestroy(J); CHKERRA(ierr);
  ierr = SNESDestroy(snes); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}/* --------------------  Evaluate Function F(x) --------------------- */
int FormFunction(SNES snes,Vec x,Vec f,void *dummy )
{
   int    ierr;
   Scalar *xx, *ff;
   ierr = VecGetArray(x,&xx); CHKERRQ(ierr);
   ierr = VecGetArray(f,&ff); CHKERRQ(ierr);
   ff[0] = xx[0]*xx[0] + xx[0]*xx[1] - 3.0;
   ff[1] = xx[0]*xx[1] + xx[1]*xx[1] - 6.0;
   return 0;
}/* --------------------  Form initial approximation ----------------- */
int FormInitialGuess(SNES snes,Vec x,void *dummy)
{
   int    ierr;
   Scalar pfive = .50;
   ierr = VecSet(&pfive,x); CHKERRQ(ierr);
   return 0;
}/* --------------------  Evaluate Jacobian F'(x) -------------------- */
int FormJacobian(SNES snes,Vec x,Mat *jac,Mat *B,MatStructure *flag,
                 void *dummy)
{
  Scalar *xx, A[4];
  int    ierr, idx[2] = {0,1};
  ierr = VecGetArray(x,&xx); CHKERRQ(ierr);
  A[0] = 2.0*xx[0] + xx[1]; A[1] = xx[0];
  A[2] = xx[1]; A[3] = xx[0] + 2.0*xx[1];
  ierr = MatSetValues(*jac,2,idx,2,idx,A,INSERTVALUES); CHKERRQ(ierr);
  *flag = 0;
  return 0;
}/* --------------------  User-defined monitor ----------------------- */
int Monitor(SNES snes,int its,double fnorm,void *dummy)
{
  int ierr;
  Vec x;
  fprintf( stdout, "iter = %d, Function norm %g \n",its,fnorm);
  ierr = SNESGetSolution(snes,&x); CHKERRQ(ierr);
  ierr = VecView(x,STDOUT_VIEWER); CHKERRQ(ierr);
  return 0;
}
