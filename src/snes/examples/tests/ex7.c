#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex7.c,v 1.41 1999/03/19 21:22:50 bsmith Exp bsmith $";
#endif

static char help[] = "Solves u`` + u^{2} = f with Newton-like methods, using\n\
 matrix-free techniques with user-provided explicit preconditioner matrix.\n\n";

#include "snes.h"

extern int  FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern int  FormFunction(SNES,Vec,Vec,void*);
extern int  FormInitialGuess(SNES,Vec);
extern int  Monitor(SNES,int,double,void *);

typedef struct {
   Viewer viewer;
} MonitorCtx;

typedef struct {
  Mat precond;
} AppCtx;

#undef __FUNC__
#define __FUNC__ "main"
int main( int argc, char **argv )
{
  SNES         snes;                 /* SNES context */
  SNESType     method = SNES_EQ_LS;  /* default nonlinear solution method */
  Vec          x, r, F, U;           /* vectors */
  Mat          J, B;                 /* Jacobian matrix-free, explicit preconditioner */
  MonitorCtx   monP;                 /* monitoring context */
  AppCtx       user;                 /* user-defined work context */
  Scalar       h, xp = 0.0, v;
  int          ierr, its, n = 5, i, flg;

  PetscInitialize( &argc, &argv,(char *)0,help );
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg); CHKERRA(ierr);
  h = 1.0/(n-1);

  /* Set up data structures */
  ierr = ViewerDrawOpen(PETSC_COMM_SELF,0,0,0,0,400,400,&monP.viewer); CHKERRA(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,&x); CHKERRA(ierr);
  PetscObjectSetName((PetscObject)x,"Approximate Solution");
  ierr = VecDuplicate(x,&r); CHKERRA(ierr);
  ierr = VecDuplicate(x,&F); CHKERRA(ierr);
  ierr = VecDuplicate(x,&U); CHKERRA(ierr); 
  PetscObjectSetName((PetscObject)U,"Exact Solution");

  /* create explict matrix preconditioner */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,3,PETSC_NULL,&B); CHKERRA(ierr);
  user.precond = B;

  /* Store right-hand-side of PDE and exact solution */
  for ( i=0; i<n; i++ ) {
    v = 6.0*xp + pow(xp+1.e-12,6.0); /* +1.e-12 is to prevent 0^6 */
    ierr = VecSetValues(F,1,&i,&v,INSERT_VALUES); CHKERRA(ierr);
    v= xp*xp*xp;
    ierr = VecSetValues(U,1,&i,&v,INSERT_VALUES); CHKERRA(ierr);
    xp += h;
  }

  /* Create nonlinear solver */  
  ierr = SNESCreate(PETSC_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes); CHKERRA(ierr);
  ierr = SNESSetType(snes,method); CHKERRA(ierr);

  /* create matrix free matrix for Jacobian */
  ierr = MatCreateSNESMF(snes,x,&J); CHKERRA(ierr);

  /* Set various routines and options */
  ierr = SNESSetFunction(snes,r,FormFunction,(void*)F); CHKERRA(ierr);
  ierr = SNESSetJacobian(snes,J,B,FormJacobian,(void*)&user); CHKERRA(ierr);
  ierr = SNESSetMonitor(snes,Monitor,(void*)&monP,0); CHKERRA(ierr);
  ierr = SNESSetFromOptions(snes); CHKERRA(ierr);

  /* Solve nonlinear system */
  ierr = FormInitialGuess(snes,x); CHKERRA(ierr);
  ierr = SNESSolve(snes,x,&its); CHKERRA(ierr);
  PetscPrintf(PETSC_COMM_SELF,"number of Newton iterations = %d\n\n", its );

  /* Free data structures */
  ierr = VecDestroy(x); CHKERRA(ierr);  ierr = VecDestroy(r); CHKERRA(ierr);
  ierr = VecDestroy(U); CHKERRA(ierr);  ierr = VecDestroy(F); CHKERRA(ierr);
  ierr = MatDestroy(J); CHKERRA(ierr);  ierr = MatDestroy(B); CHKERRA(ierr);
  ierr = SNESDestroy(snes); CHKERRA(ierr);
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
  ierr = VecGetArray((Vec) dummy,&FF); CHKERRQ(ierr);
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

#undef __FUNC__
#define __FUNC__ "FormInitialGuess"
int FormInitialGuess(SNES snes,Vec x)
{
  int    ierr;
  Scalar pfive = .50;
  ierr = VecSet(&pfive,x); CHKERRQ(ierr);
  return 0;
}
#undef __FUNC__
#define __FUNC__ "FormJacobian"
/* --------------------  Evaluate Jacobian F'(x) -------------------- */
/*  Evaluates a matrix that is used to precondition the matrix-free
    jacobian. In this case, the explict preconditioner matrix is 
    also EXACTLY the Jacobian. In general, it would be some lower
    order, simplified apprioximation */

int FormJacobian(SNES snes,Vec x,Mat *jac,Mat *B,MatStructure*flag,void *dummy)
{
  Scalar *xx, A[3], d;
  int    i, n, j[3], ierr, iter;
  AppCtx *user = (AppCtx*) dummy;

  ierr = SNESGetIterationNumber(snes,&iter); CHKERRQ(ierr);

  if (iter%2 ==0) { /* Compute new preconditioner matrix */
    printf("iter=%d, computing new preconditioning matrix\n",iter+1);
    *B = user->precond;
    ierr = VecGetArray(x,&xx); CHKERRQ(ierr);
    ierr = VecGetSize(x,&n); CHKERRQ(ierr);
    d = (double)(n - 1); d = d*d;

    /* do nothing with Jac since it is Matrix-free */
    i = 0; A[0] = 1.0; 
    ierr = MatSetValues(*B,1,&i,1,&i,&A[0],INSERT_VALUES); CHKERRQ(ierr);
    for ( i=1; i<n-1; i++ ) {
      j[0] = i - 1; j[1] = i;                   j[2] = i + 1; 
      A[0] = d;     A[1] = -2.0*d + 2.0*xx[i];  A[2] = d; 
      ierr = MatSetValues(*B,1,&i,3,j,A,INSERT_VALUES); CHKERRQ(ierr);
    }
    i = n-1; A[0] = 1.0; 
    ierr = MatSetValues(*B,1,&i,1,&i,&A[0],INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatAssemblyBegin(*B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*B,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = VecRestoreArray(x,&xx); CHKERRQ(ierr);
    *flag = SAME_NONZERO_PATTERN;
  }  else { /* reuse preconditioner from last iteration */
    printf("iter=%d, using old preconditioning matrix\n",iter+1);
    *flag = SAME_PRECONDITIONER;
  }

  return 0;
}
/* --------------------  User-defined monitor ----------------------- */

#undef __FUNC__
#define __FUNC__ "Monitor"
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
