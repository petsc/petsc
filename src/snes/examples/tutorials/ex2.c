#ifndef lint
static char vcid[] = "$Id: ex3.c,v 1.27 1995/08/02 04:19:12 bsmith Exp curfman $";
#endif

static char help[] = 
"This example uses Newton-like methods to solve u`` + u^{2} = f.\n\n";

#include "draw.h"
#include "snes.h"
#include "petsc.h"
#include <math.h>

int  FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*),
     FormFunction(SNES,Vec,Vec,void*),
     FormInitialGuess(SNES,Vec,void*),
     Monitor(SNES,int,double,void *);

typedef struct {
   DrawCtx win1;
} MonitorCtx;

int main( int argc, char **argv )
{
  SNES         snes;               /* SNES context */
  Vec          x,r,F,U;
  Mat          J;                  /* Jacobian matrix */
  int          ierr, its, n = 5,i;
  Scalar       h,xp = 0.0,v;
  MonitorCtx   monP;               /* monitoring context */

  PetscInitialize( &argc, &argv, 0,0 );
  if (OptionsHasName(0,"-help")) fprintf(stdout,"%s",help);
  OptionsGetInt(0,"-n",&n);
  h = 1.0/(n-1);

  /* Set up data structures */
  ierr = DrawOpenX(MPI_COMM_SELF,0,0,0,0,400,400,&monP.win1); CHKERRA(ierr);
  ierr = VecCreateSequential(MPI_COMM_SELF,n,&x); CHKERRA(ierr);
  PetscObjectSetName((PetscObject)x,"Approximate Solution");
  ierr = VecDuplicate(x,&r); CHKERRA(ierr);
  ierr = VecDuplicate(x,&F); CHKERRA(ierr);
  ierr = VecDuplicate(x,&U); CHKERRA(ierr); 
  PetscObjectSetName((PetscObject)U,"Exact Solution");
  ierr = MatCreateSequentialAIJ(MPI_COMM_SELF,n,n,3,0,&J); CHKERRA(ierr);

  /* Store right-hand-side of PDE and exact solution */
  for ( i=0; i<n; i++ ) {
    v = 6.0*xp + pow(xp+1.e-12,6.0); /* +1.e-12 is to prevent 0^6 */
    ierr = VecSetValues(F,1,&i,&v,INSERTVALUES); CHKERRA(ierr);
    v= xp*xp*xp;
    ierr = VecSetValues(U,1,&i,&v,INSERTVALUES); CHKERRA(ierr);
    xp += h;
  }

  /* Create nonlinear solver */  
  ierr = SNESCreate(MPI_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes); 
  CHKERRA(ierr);

  /* Set various routines */
  ierr = SNESSetSolution(snes,x,FormInitialGuess,0); CHKERRA(ierr);
  ierr = SNESSetFunction(snes,r,FormFunction,(void*)F,1); CHKERRA(ierr);
  ierr = SNESSetJacobian(snes,J,J,FormJacobian,0); CHKERRA(ierr);
  ierr = SNESSetMonitor(snes,Monitor,(void*)&monP); CHKERRA(ierr);

  /* Set up nonlinear solver; then execute it */
  ierr = SNESSetFromOptions(snes); CHKERRA(ierr);
  ierr = SNESSetUp(snes); CHKERRA(ierr);
  ierr = SNESSolve(snes,&its); CHKERRA(ierr);
  ierr = SNESView(snes,STDOUT_VIEWER_COMM); CHKERRA(ierr);
  printf( "number of Newton iterations = %d\n\n", its );

  /* Free data structures */
  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(r); CHKERRA(ierr);
  ierr = VecDestroy(U); CHKERRA(ierr);
  ierr = VecDestroy(F); CHKERRA(ierr);
  ierr = MatDestroy(J); CHKERRA(ierr);
  ierr = SNESDestroy(snes); CHKERRA(ierr);
  ierr = DrawDestroy(monP.win1); CHKERRA(ierr);
  PetscFinalize();

  return 0;
}
/* --------------------  Evaluate Function F(x) --------------------- */

int FormFunction(SNES snes,Vec x,Vec f,void *dummy)
{
   Scalar *xx, *ff,*FF,d;
   int    i, ierr, n;
   ierr = VecGetArray(x,&xx); CHKERRQ(ierr);
   ierr = VecGetArray(f,&ff); CHKERRQ(ierr);
   ierr = VecGetArray((Vec) dummy,&FF); CHKERRQ(ierr);
   ierr = VecGetSize(x,&n); CHKERRQ(ierr);
   d = (double) (n - 1); d = d*d;
   ff[0]   = -xx[0];
   for ( i=1; i<n-1; i++ ) {
     ff[i] = -d*(xx[i-1] - 2.0*xx[i] + xx[i+1]) - xx[i]*xx[i] + FF[i];
   }
   ff[n-1] = -xx[n-1] + 1.0;
   return 0;
}
/* --------------------  Form initial approximation ----------------- */

int FormInitialGuess(SNES snes,Vec x,void *dummy)
{
   int    ierr;
   Scalar pfive = .50;
   ierr = VecSet(&pfive,x); CHKERRQ(ierr);
   return 0;
}
/* --------------------  Evaluate Jacobian F'(x) -------------------- */

int FormJacobian(SNES snes,Vec x,Mat *jac,Mat *B,MatStructure*flag,void *dummy)
{
  Scalar *xx, A, d;
  int    i, n, j, ierr;
  ierr = VecGetArray(x,&xx); CHKERRQ(ierr);
  ierr =  VecGetSize(x,&n); CHKERRQ(ierr);
  d = (double)(n - 1); d = d*d;
  i = 0; A = 1.0; 
  ierr = MatSetValues(*jac,1,&i,1,&i,&A,INSERTVALUES); CHKERRQ(ierr);
  for ( i=1; i<n-1; i++ ) {
    A = d; 
    j = i - 1; 
    ierr = MatSetValues(*jac,1,&i,1,&j,&A,INSERTVALUES); CHKERRQ(ierr);
    j = i + 1; 
    ierr = MatSetValues(*jac,1,&i,1,&j,&A,INSERTVALUES); CHKERRQ(ierr);
    A = -2.0*d + 2.0*xx[i];
    j = i + 1; 
    ierr = MatSetValues(*jac,1,&i,1,&i,&A,INSERTVALUES); CHKERRQ(ierr);
  }
  i = n-1; A = 1.0; 
  ierr = MatSetValues(*jac,1,&i,1,&i,&A,INSERTVALUES); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*jac,FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,FINAL_ASSEMBLY); CHKERRQ(ierr);
/*  *flag = MAT_SAME_NONZERO_PATTERN; */
  return 0;
}
/* --------------------  User-defined monitor ----------------------- */

int Monitor(SNES snes,int its,double fnorm,void *dummy)
{
  int        ierr;
  MonitorCtx *monP = (MonitorCtx*) dummy;
  Vec        x;
  fprintf(stdout, "iter = %d, Function norm %g \n",its,fnorm);
  ierr = SNESGetSolution(snes,&x); CHKERRQ(ierr);
  ierr = VecView(x,(Viewer)monP->win1); CHKERRQ(ierr);
  return 0;
}
