#ifndef lint
static char vcid[] = "$Id: ex3.c,v 1.52 1996/07/08 22:23:15 bsmith Exp curfman $";
#endif

static char help[] = "Uses Newton-like methods to solve u`` + u^{2} = f.\n\n";

#include "draw.h"
#include "snes.h"
#include <math.h>

int  FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*),
     FormFunction(SNES,Vec,Vec,void*),
     FormInitialGuess(SNES,Vec),
     Monitor(SNES,int,double,void *);

typedef struct {
   Viewer viewer;
} MonitorCtx;

int main( int argc, char **argv )
{
  SNES       snes;                   /* SNES context */
  Vec        x, r, F, U;             /* vectors */
  Mat        J;                      /* Jacobian matrix */
  MonitorCtx monP;                   /* monitoring context */
  int        ierr, its, n = 5, i, flg, maxit, maxf;
  Scalar     h, xp = 0.0, v;
  double     atol, rtol, stol;

  PetscInitialize( &argc, &argv,(char *)0,help );
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg); CHKERRA(ierr);
  h = 1.0/(n-1);

  /* Set up data structures */
  ierr = ViewerDrawOpenX(MPI_COMM_SELF,0,0,0,0,400,400,&monP.viewer);CHKERRA(ierr);
  ierr = VecCreateSeq(MPI_COMM_SELF,n,&x); CHKERRA(ierr);
  PetscObjectSetName((PetscObject)x,"Approximate Solution");
  ierr = VecDuplicate(x,&r); CHKERRA(ierr);
  ierr = VecDuplicate(x,&F); CHKERRA(ierr);
  ierr = VecDuplicate(x,&U); CHKERRA(ierr); 
  PetscObjectSetName((PetscObject)U,"Exact Solution");
  ierr = MatCreateSeqAIJ(MPI_COMM_SELF,n,n,3,PETSC_NULL,&J); CHKERRA(ierr);

  /* Store right-hand-side of PDE and exact solution */
  for ( i=0; i<n; i++ ) {
    v = 6.0*xp + pow(xp+1.e-12,6.0); /* +1.e-12 is to prevent 0^6 */
    ierr = VecSetValues(F,1,&i,&v,INSERT_VALUES); CHKERRA(ierr);
    v= xp*xp*xp;
    ierr = VecSetValues(U,1,&i,&v,INSERT_VALUES); CHKERRA(ierr);
    xp += h;
  }

  /* Create nonlinear solver */  
  ierr = SNESCreate(MPI_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes); CHKERRA(ierr);

  /* Set various routines and options */
  ierr = SNESSetFunction(snes,r,FormFunction,(void*)F);  CHKERRA(ierr);
  ierr = SNESSetJacobian(snes,J,J,FormJacobian,0); CHKERRA(ierr);
  ierr = SNESSetMonitor(snes,Monitor,(void*)&monP); CHKERRA(ierr); 
  ierr = SNESSetFromOptions(snes); CHKERRA(ierr);

  /* Print the various parameters used for convergence testing */
  ierr = SNESGetTolerances(snes,&atol,&rtol,&stol,&maxit,&maxf); CHKERRA(ierr);
  PetscPrintf(MPI_COMM_WORLD,"atol=%g, rtol=%g, stol=%g, maxit=%d, maxf=%d\n",
     atol,rtol,stol,maxit,maxf);


  /* Solve nonlinear system */
  ierr = FormInitialGuess(snes,x); CHKERRA(ierr); CHKERRA(ierr);
  ierr = SNESSolve(snes,x,&its); CHKERRA(ierr);
  PetscPrintf(MPI_COMM_SELF,"number of Newton iterations = %d\n\n", its );

  /* Free data structures */
  ierr = VecDestroy(x); CHKERRA(ierr);  ierr = VecDestroy(r); CHKERRA(ierr);
  ierr = VecDestroy(U); CHKERRA(ierr);  ierr = VecDestroy(F); CHKERRA(ierr);
  ierr = MatDestroy(J); CHKERRA(ierr);  ierr = SNESDestroy(snes); CHKERRA(ierr);
  ierr = ViewerDestroy(monP.viewer); CHKERRA(ierr);
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
   ierr = VecGetArray((Vec)dummy,&FF); CHKERRQ(ierr);
   ierr = VecGetSize(x,&n); CHKERRQ(ierr);
   d = (double) (n - 1); d = d*d;
   ff[0]   = xx[0];
   for ( i=1; i<n-1; i++ ) {
     ff[i] = d*(xx[i-1] - 2.0*xx[i] + xx[i+1]) + xx[i]*xx[i] - FF[i];
   }
   ff[n-1] = xx[n-1] - 1.0;
   ierr = VecRestoreArray(x,&xx); CHKERRQ(ierr);
   ierr = VecRestoreArray(f,&ff); CHKERRQ(ierr);
   ierr = VecRestoreArray((Vec)dummy,&FF); CHKERRQ(ierr);
   return 0;
}
/* --------------------  Form initial approximation ----------------- */

int FormInitialGuess(SNES snes,Vec x)
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
  ierr = MatSetValues(*jac,1,&i,1,&i,&A,INSERT_VALUES); CHKERRQ(ierr);
  for ( i=1; i<n-1; i++ ) {
    A = d; 
    j = i - 1; 
    ierr = MatSetValues(*jac,1,&i,1,&j,&A,INSERT_VALUES); CHKERRQ(ierr);
    j = i + 1; 
    ierr = MatSetValues(*jac,1,&i,1,&j,&A,INSERT_VALUES); CHKERRQ(ierr);
    A = -2.0*d + 2.0*xx[i];
    j = i + 1; 
    ierr = MatSetValues(*jac,1,&i,1,&i,&A,INSERT_VALUES); CHKERRQ(ierr);
  }
  i = n-1; A = 1.0; 
  ierr = MatSetValues(*jac,1,&i,1,&i,&A,INSERT_VALUES); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = VecRestoreArray(x,&xx); CHKERRQ(ierr);
/*  *flag = MAT_SAME_NONZERO_PATTERN; */
  return 0;
}
/* --------------------  User-defined monitor ----------------------- */

int Monitor(SNES snes,int its,double fnorm,void *dummy)
{
  int        ierr;
  MonitorCtx *monP = (MonitorCtx*) dummy;
  Vec        x;

  fprintf(stdout, "iter = %d, SNES Function norm %g \n",its,fnorm);
  ierr = SNESGetSolution(snes,&x); CHKERRQ(ierr);
  ierr = VecView(x,monP->viewer); CHKERRQ(ierr);
  return 0;
}
