#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex6.c,v 1.51 1999/01/12 23:17:53 bsmith Exp curfman $";
#endif

static char help[] = "Uses Newton-like methods to solve u`` + u^{2} = f.  Different\n\
matrices are used for the Jacobian and the preconditioner.  The code also\n\
demonstrates the use of matrix-free Newton-Krylov methods in conjunction\n\
with a user-provided preconditioner.  Input arguments are:\n\
   -snes_mf : Use matrix-free Newton methods\n\
   -user_precond : Employ a user-defined preconditioner.  Used only with\n\
                   matrix-free methods in this example.\n\n";

/*T
   Concepts: SNES^Using different matrices for the Jacobian and preconditioner;
   Concepts: SNES^Using matrix-free methods and a user-provided preconditioner;
   Routines: SNESCreate(); SNESSetFunction(); SNESSetJacobian();
   Routines: SNESSolve(); SNESSetFromOptions(); SNESGetSLES();
   Routines: SLESGetPC(); PCSetType(); PCShellSetApply(); PCSetType();
   Routines: SNESSetConvergenceHistory(); SNESGetConvergenceHistory();
   Routines: KSPSetResidualHistory(); KSPGetResidualHistory();
   Processors: 1
T*/

/* 
   Include "snes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petsc.h  - base PETSc routines   vec.h - vectors
     sys.h    - system routines       mat.h - matrices
     is.h     - index sets            ksp.h - Krylov subspace methods
     viewer.h - viewers               pc.h  - preconditioners
     sles.h   - linear solvers
*/
#include "snes.h"

/* 
   User-defined routines
*/
int FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
int FormFunction(SNES,Vec,Vec,void*);
int MatrixFreePreconditioner(void*,Vec,Vec);

int main( int argc, char **argv )
{
  SNES     snes;                 /* SNES context */
  SLES     sles;                 /* SLES context */
  PC       pc;                   /* PC context */
  KSP      ksp;                  /* KSP context */
  Vec      x, r, F;              /* vectors */
  Mat      J, JPrec;             /* Jacobian, preconditioner matrices */
  int      ierr, its, n = 5, i, size, flg;
  int      *sres_hist_its, res_hist_len = 200, sres_hist_len = 10;
  double   h, xp = 0.0, *res_hist, *sres_hist;
  Scalar   v, pfive = .5;

  PetscInitialize( &argc, &argv,(char *)0,help );
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  if (size != 1) SETERRA(1,0,"This is a uniprocessor example only!");
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg); CHKERRA(ierr);
  h = 1.0/(n-1);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SNESCreate(PETSC_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures; set function evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = VecCreate(PETSC_COMM_SELF,PETSC_DECIDE,n,&x); CHKERRA(ierr);
  ierr = VecSetFromOptions(x);CHKERRA(ierr);
  ierr = VecDuplicate(x,&r); CHKERRA(ierr);
  ierr = VecDuplicate(x,&F); CHKERRA(ierr);

  ierr = SNESSetFunction(snes,r,FormFunction,(void*)F); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structures; set Jacobian evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,3,PETSC_NULL,&J); CHKERRA(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,1,PETSC_NULL,&JPrec); CHKERRA(ierr);

  /*
     Note that in this case we create separate matrices for the Jacobian
     and preconditioner matrix.  Both of these are computed in the
     routine FormJacobian()
  */
  ierr = SNESSetJacobian(snes,J,JPrec,FormJacobian,0); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* Set preconditioner for matrix-free method */
  ierr = OptionsHasName(PETSC_NULL,"-snes_mf",&flg); CHKERRA(ierr);
  if (flg) {
    ierr = SNESGetSLES(snes,&sles); CHKERRA(ierr);
    ierr = SLESGetPC(sles,&pc); CHKERRA(ierr);
    ierr = OptionsHasName(PETSC_NULL,"-user_precond",&flg); CHKERRA(ierr);
    if (flg) { /* user-defined precond */
      ierr = PCSetType(pc,PCSHELL); CHKERRA(ierr);
      ierr = PCShellSetApply(pc,MatrixFreePreconditioner,PETSC_NULL);CHKERRA(ierr);
    } else {ierr = PCSetType(pc,PCNONE); CHKERRA(ierr);}
  }

  ierr = SNESSetFromOptions(snes); CHKERRA(ierr);

  /*
     Save all the linear residuals for all the Newton steps; this enables us
     to retain complete convergence history for printing after the conclusion
     of SNESSolve().  Alternatively, one could use the monitoring options
           -snes_monitor -ksp_monitor
     to see this information during the solver's execution; however, such
     output during the run distorts performance evaluation data.  So, the
     following is a good option when monitoring code performance, for example
     when using -log_summary.
  */
  ierr = OptionsHasName(PETSC_NULL,"-rhistory",&flg);CHKERRA(ierr);
  if (flg) {
    ierr     = SNESGetSLES(snes,&sles);CHKERRA(ierr);
    ierr     = SLESGetKSP(sles,&ksp);CHKERRA(ierr);
    res_hist = (double *) PetscMalloc(res_hist_len*sizeof(double));CHKPTRA(res_hist);
    ierr     = KSPSetResidualHistory(ksp,res_hist,res_hist_len,PETSC_FALSE);CHKERRA(ierr);
    sres_hist = (double *) PetscMalloc(sres_hist_len*sizeof(double));CHKPTRA(sres_hist);
    sres_hist_its = (int*) PetscMalloc(sres_hist_len*sizeof(int));CHKPTRA(sres_hist_its);
    ierr     = SNESSetConvergenceHistory(snes,sres_hist,sres_hist_its,sres_hist_len,PETSC_FALSE);CHKERRA(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize application:
     Store right-hand-side of PDE and exact solution
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  xp = 0.0;
  for ( i=0; i<n; i++ ) {
    v = 6.0*xp + pow(xp+1.e-12,6.0); /* +1.e-12 is to prevent 0^6 */
    ierr = VecSetValues(F,1,&i,&v,INSERT_VALUES); CHKERRA(ierr);
    xp += h;
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = VecSet(&pfive,x); CHKERRA(ierr);
  ierr = SNESSolve(snes,x,&its); CHKERRA(ierr);
  PetscPrintf(PETSC_COMM_SELF,"number of Newton iterations = %d\n\n", its );

  ierr = OptionsHasName(PETSC_NULL,"-rhistory",&flg);CHKERRA(ierr);
  if (flg) {
    ierr = KSPGetResidualHistory(ksp,PETSC_NULL,&res_hist_len);CHKERRA(ierr);
    PetscDoubleView(res_hist_len,res_hist,VIEWER_STDOUT_SELF);
    PetscFree(res_hist);
    ierr = SNESGetConvergenceHistory(snes,PETSC_NULL,PETSC_NULL,&sres_hist_len);CHKERRA(ierr);
    PetscDoubleView(sres_hist_len,sres_hist,VIEWER_STDOUT_SELF);
    PetscIntView(sres_hist_len,sres_hist_its,VIEWER_STDOUT_SELF);
    PetscFree(sres_hist);
    PetscFree(sres_hist_its);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = VecDestroy(x); CHKERRA(ierr);     ierr = VecDestroy(r); CHKERRA(ierr);
  ierr = VecDestroy(F); CHKERRA(ierr);     ierr = MatDestroy(J); CHKERRA(ierr);
  ierr = MatDestroy(JPrec); CHKERRA(ierr); ierr = SNESDestroy(snes); CHKERRA(ierr);
  PetscFinalize();

  return 0;
}
/* ------------------------------------------------------------------- */
/* 
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
 */
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
/* ------------------------------------------------------------------- */
/*
   FormJacobian - This routine demonstrates the use of different
   matrices for the Jacobian and preconditioner

   Input Parameters:
.  snes - the SNES context
.  x - input vector
.  ptr - optional user-defined context, as set by SNESSetJacobian()

   Output Parameters:
.  A - Jacobian matrix
.  B - different preconditioning matrix
.  flag - flag indicating matrix structure
*/
int FormJacobian(SNES snes,Vec x,Mat *jac,Mat *prejac,MatStructure *flag,
                 void *dummy)
{
  Scalar *xx, A[3], d;
  int    i, n, j[3], ierr;

  ierr = VecGetArray(x,&xx); CHKERRQ(ierr);
  ierr = VecGetSize(x,&n); CHKERRQ(ierr);
  d = (double)(n - 1); d = d*d;

  /* Form Jacobian.  Also form a different preconditioning matrix that 
     has only the diagonal elements. */
  i = 0; A[0] = 1.0; 
  ierr = MatSetValues(*jac,1,&i,1,&i,&A[0],INSERT_VALUES); CHKERRQ(ierr);
  ierr = MatSetValues(*prejac,1,&i,1,&i,&A[0],INSERT_VALUES); CHKERRQ(ierr);
  for ( i=1; i<n-1; i++ ) {
    j[0] = i - 1; j[1] = i;                   j[2] = i + 1; 
    A[0] = d;     A[1] = -2.0*d + 2.0*xx[i];  A[2] = d; 
    ierr = MatSetValues(*jac,1,&i,3,j,A,INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatSetValues(*prejac,1,&i,1,&i,&A[1],INSERT_VALUES); CHKERRQ(ierr);
  }
  i = n-1; A[0] = 1.0; 
  ierr = MatSetValues(*jac,1,&i,1,&i,&A[0],INSERT_VALUES); CHKERRQ(ierr);
  ierr = MatSetValues(*prejac,1,&i,1,&i,&A[0],INSERT_VALUES); CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*prejac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*prejac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  ierr = VecRestoreArray(x,&xx); CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  return 0;
}
/* ------------------------------------------------------------------- */
/*
   MatrixFreePreconditioner - This routine demonstrates the use of a
   user-provided preconditioner.  This code implements just the null
   preconditioner, which of course is not recommended for general use.

   Input Parameters:
.  ctx - optional user-defined context, as set by PCShellSetApply()
.  x - input vector

   Output Parameter:
.  y - preconditioned vector
*/
int MatrixFreePreconditioner(void *ctx,Vec x,Vec y)
{
  int ierr;
  ierr = VecCopy(x,y); CHKERRQ(ierr);  
  return 0;
}
