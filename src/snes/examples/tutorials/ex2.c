#ifndef lint
static char vcid[] = "$Id: ex3.c,v 1.14 1995/05/02 23:39:45 bsmith Exp bsmith $";
#endif

static char help[] = "Uses Newton method to solve u`` + u^{2} = f\n";

#include "draw.h"
#include "snes.h"
#include "options.h"
#include <math.h>

int  FormJacobian(SNES,Vec,Mat*,Mat*,int*,void*),
     FormFunction(SNES,Vec,Vec,void*),
     FormInitialGuess(SNES,Vec,void*),
     Monitor(SNES,int,double,void *);

typedef struct {
   DrawCtx win1;
} MonitorCtx;

int main( int argc, char **argv )
{
  SNES         snes;
  SLES         sles;
  SNESMethod   method = SNES_NLS;  /* nonlinear solution method */
  Vec          x,r,F,U;
  Mat          J;
  int          ierr, its, n = 5,i;
  double       h,xp = 0.0,v;
  MonitorCtx   monP;

  PetscInitialize( &argc, &argv, 0,0 );
  OptionsGetInt(0,0,"-n",&n);
  h = 1.0/(n-1);

  ierr = DrawOpenX(MPI_COMM_SELF,0,0,0,0,400,400,&monP.win1); CHKERR(ierr);
  ierr = VecCreateSequential(MPI_COMM_SELF,n,&x); CHKERRA(ierr);
  PetscObjectSetName((PetscObject)x,"Approximate Solution");
  ierr = VecDuplicate(x,&r); CHKERRA(ierr);
  ierr = VecDuplicate(x,&F); CHKERRA(ierr);
  ierr = VecDuplicate(x,&U); CHKERRA(ierr); 
  PetscObjectSetName((PetscObject)U,"Exact Solution");
  ierr = MatCreateSequentialAIJ(MPI_COMM_SELF,n,n,3,0,&J); CHKERRA(ierr);

  /* store right hand side to PDE; and exact solution */
  for ( i=0; i<n; i++ ) {
    v = 6.0*xp + pow(xp,6.0);
    VecSetValues(F,1,&i,&v,INSERTVALUES);
    v= xp*xp*xp;
    VecSetValues(U,1,&i,&v,INSERTVALUES);
    xp += h;
  }

  ierr = SNESCreate(MPI_COMM_WORLD,&snes); CHKERRA(ierr);
  ierr = SNESSetMethod(snes,method); CHKERRA(ierr);
  ierr = SNESSetMonitor(snes,Monitor,(void*)&monP);

  /* Set various routines */
  SNESSetSolution( snes, x,FormInitialGuess,0 );
  SNESSetFunction( snes, r,FormFunction,(void*)F, 1 );
  SNESSetJacobian( snes, J,J, FormJacobian,0 );	

  ierr = SNESSetFromOptions(snes); CHKERR(ierr);
  SNESSetUp( snes );				       

  /* Execute solution method */
  ierr = SNESSolve( snes,&its );				       
  printf( "number of Newton iterations = %d\n\n", its );

  VecDestroy(x);
  VecDestroy(r);
  VecDestroy(U);
  VecDestroy(F);
  MatDestroy(J);
  SNESDestroy( snes );	
  DrawDestroy(monP.win1);			       
  PetscFinalize();

  return 0;
}
/* ------------------------------------------------ */
/*
    Evaluate Function F(x).
 */

int FormFunction(SNES snes,Vec x,Vec  f,void *dummy )
{
   Scalar *xx, *ff,*FF,d;
   int    i,n;
   VecGetArray(x,&xx); VecGetArray(f,&ff); VecGetArray((Vec) dummy,&FF);
   VecGetSize(x,&n);
   d = (double) (n - 1); d = d*d;
   ff[0]   = -xx[0];
   for ( i=1; i<n-1; i++ ) {
     ff[i] = -d*(xx[i-1] - 2.0*xx[i] + xx[i+1]) - xx[i]*xx[i] + FF[i];
   }
   ff[n-1] = -xx[n-1] + 1.0;
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
int FormJacobian(SNES snes,Vec x,Mat *jac,Mat *B,int *flag, void *dummy)
{
  Scalar *xx, A,d;
  int    i,n,j,ierr;
  VecGetArray(x,&xx); VecGetSize(x,&n);
  d = (double)(n - 1); d = d*d;
  i = 0; A = 1.0; MatSetValues(*jac,1,&i,1,&i,&A,INSERTVALUES);
  for ( i=1; i<n-1; i++ ) {
    A = d; 
    j = i - 1; MatSetValues(*jac,1,&i,1,&j,&A,INSERTVALUES);
    j = i + 1; MatSetValues(*jac,1,&i,1,&j,&A,INSERTVALUES);
    A = -2.0*d + 2.0*xx[i];
    j = i + 1; MatSetValues(*jac,1,&i,1,&i,&A,INSERTVALUES);
  }
  i = n-1; A = 1.0; MatSetValues(*jac,1,&i,1,&i,&A,INSERTVALUES);
  ierr = MatAssemblyBegin(*jac,FINAL_ASSEMBLY); CHKERR(ierr);
  ierr = MatAssemblyEnd(*jac,FINAL_ASSEMBLY); CHKERR(ierr);
  *flag = MAT_SAME_NONZERO_PATTERN;
  return 0;
}

int Monitor(SNES snes,int its,double fnorm,void *dummy)
{
  MonitorCtx *monP = (MonitorCtx*) dummy;
  Vec        x;
  fprintf( stdout, "iter = %d, Function norm %g \n",its,fnorm);
  SNESGetSolution(snes,&x);
  VecView(x,(Viewer)monP->win1);
  return 0;
}
