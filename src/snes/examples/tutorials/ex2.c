#ifndef lint
static char vcid[] = "$Id: ex3.c,v 1.9 1995/04/27 20:17:12 bsmith Exp bsmith $";
#endif

static char help[] = "Uses Newton method to solve u`` + u^{2} = f\n";

#include "draw.h"
#include "snes.h"
#include "options.h"
#include <math.h>

int  FormJacobian(Vec,Mat*,void*),
     FormResidual(Vec,Vec,void*),
     FormInitialGuess(Vec,void*),
     Monitor(SNES,int,double,void *);

typedef struct {
   DrawCtx win1,win2;
   Vec     U;
} MonitorCtx;

int main( int argc, char **argv )
{
  SNES         snes;
  SLES         sles;
  SNESMETHOD   method = SNES_NLS;  /* nonlinear solution method */
  Vec          x,r,F,U;
  Mat          J;
  int          ierr, its, n = 5,i;
  double       h,xp = 0.0,v;
  MonitorCtx   monP;

  PetscInitialize( &argc, &argv, 0,0 );
  OptionsGetInt(0,0,"-n",&n);
  h = 1.0/(n-1);

  ierr = DrawOpenX(MPI_COMM_SELF,0,0,0,0,400,400,&monP.win1); CHKERR(ierr);
  ierr = DrawOpenX(MPI_COMM_SELF,0,0,400,0,400,400,&monP.win2); CHKERR(ierr);
  ierr = VecCreateSequential(MPI_COMM_SELF,n,&x); CHKERRA(ierr);
  PetscObjectSetName((PetscObject)x,"Approximate Solution");
  ierr = VecCreate(x,&r); CHKERRA(ierr);
  ierr = VecCreate(x,&F); CHKERRA(ierr);
  ierr = VecCreate(x,&U); CHKERRA(ierr); 
  PetscObjectSetName((PetscObject)U,"Exact Solution");
  monP.U = U; 
  ierr = MatCreateSequentialAIJ(MPI_COMM_SELF,n,n,3,0,&J); CHKERRA(ierr);

  /* store right hand side to PDE; and exact solution */
  for ( i=0; i<n; i++ ) {
    v = 6.0*xp + pow(xp,6.0);
    VecSetValues(F,1,&i,&v,InsertValues);
    v= xp*xp*xp;
    VecSetValues(U,1,&i,&v,InsertValues);
    xp += h;
  }

  ierr = SNESCreate(MPI_COMM_WORLD,&snes); CHKERRA(ierr);
  ierr = SNESSetMethod(snes,method); CHKERRA(ierr);
  ierr = SNESSetMonitor(snes,Monitor,(void*)&monP);
  ierr = SNESSetFromOptions(snes); CHKERR(ierr);

  /* Set various routines */
  SNESSetSolution( snes, x,FormInitialGuess,0 );
  SNESSetFunction( snes, r,FormResidual,(void*)F, 1 );
  SNESSetJacobian( snes, J, FormJacobian,0 );	

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
  DrawDestroy(monP.win2);			       
  PetscFinalize();

  return 0;
}
/* ------------------------------------------------ */
/*
    Evaluate residual F(x).
 */

int FormResidual(Vec x,Vec  f,void *dummy )
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
int FormInitialGuess(Vec x,void *dummy)
{
   Scalar pfive = .50;
   VecSet(&pfive,x);
   return 0;
}
/* ------------------------------------------------ */
/*
   Evaluate Jacobian matrix F'(x).
 */
int FormJacobian(Vec x,Mat *jac,void *dummy)
{
  Scalar *xx, A,d;
  int    i,n,j,ierr;
  VecGetArray(x,&xx); VecGetSize(x,&n);
  d = (double)(n - 1); d = d*d;
  i = 0; A = 1.0; MatSetValues(*jac,1,&i,1,&i,&A,InsertValues);
  for ( i=1; i<n-1; i++ ) {
    A = d; 
    j = i - 1; MatSetValues(*jac,1,&i,1,&j,&A,InsertValues);
    j = i + 1; MatSetValues(*jac,1,&i,1,&j,&A,InsertValues);
    A = -2.0*d + 2.0*xx[i];
    j = i + 1; MatSetValues(*jac,1,&i,1,&i,&A,InsertValues);
  }
  i = n-1; A = 1.0; MatSetValues(*jac,1,&i,1,&i,&A,InsertValues);
  ierr = MatBeginAssembly(*jac,FINAL_ASSEMBLY); CHKERR(ierr);
  ierr = MatEndAssembly(*jac,FINAL_ASSEMBLY); CHKERR(ierr);
  return 0;
}

int Monitor(SNES snes,int its,double fnorm,void *dummy)
{
  MonitorCtx *monP = (MonitorCtx*) dummy;
  Vec        x;
  fprintf( stdout, "iter = %d, residual norm %g \n",its,fnorm);
  SNESGetSolution(snes,&x);
  VecView(x,(Viewer)monP->win1);
  VecView(monP->U,(Viewer)monP->win2);
  return 0;
}
