#ifndef lint
static char vcid[] = "$Id: ex6.c,v 1.39 1996/07/08 22:23:15 bsmith Exp curfman $";
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
   Routines: SNESSolve(); SNESSetFromOptions();
   Processors: n
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
#include <math.h>

/* 
   User-defined routines
*/
int FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
int FormFunction(SNES,Vec,Vec,void*);
int MatrixFreePreconditioner(void *ctx,Vec x,Vec y);

int main( int argc, char **argv )
{
  SNES     snes;                 /* SNES context */
  SNESType method = SNES_EQ_LS;  /* default nonlinear solution method */
  SLES     sles;                 /* SLES context */
  PC       pc;                   /* PC context */
  Vec      x, r, F;              /* solution, residual, work vector */
  Mat      J, JPrec;             /* Jacobian, preconditioner matrices */
  int      ierr, its, n = 5, i,flg;
  double   h, xp = 0.0;
  Scalar   v, pfive = .5;

  PetscInitialize( &argc, &argv,(char *)0,help );
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg); CHKERRA(ierr);
  h = 1.0/(n-1);

  /* Set up data structures */
  ierr = VecCreateSeq(MPI_COMM_SELF,n,&x); CHKERRA(ierr);
  ierr = VecDuplicate(x,&r); CHKERRA(ierr);
  ierr = VecDuplicate(x,&F); CHKERRA(ierr);
  ierr = MatCreateSeqAIJ(MPI_COMM_SELF,n,n,3,PETSC_NULL,&J); CHKERRA(ierr);
  ierr = MatCreateSeqAIJ(MPI_COMM_SELF,n,n,1,PETSC_NULL,&JPrec); CHKERRA(ierr);

  /* Store right-hand-side of PDE */
  for ( i=0; i<n; i++ ) {
    v = 6.0*xp + pow(xp+1.e-12,6.0); /* +1.e-12 is to prevent 0^6 */
    ierr = VecSetValues(F,1,&i,&v,INSERT_VALUES); CHKERRA(ierr);
    xp += h;
  }

  /* Create nonlinear solver */  
  ierr = SNESCreate(MPI_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes);CHKERRA(ierr);
  ierr = SNESSetType(snes,method); CHKERRA(ierr);

  /* Set various routines */
  ierr = SNESSetFunction(snes,r,FormFunction,(void*)F); CHKERRA(ierr);
  ierr = SNESSetJacobian(snes,J,JPrec,FormJacobian,0); CHKERRA(ierr);

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

  /* Form initial guess */
  ierr = VecSet(&pfive,x); CHKERRQ(ierr);

  /* Set options; then solve nonlinear system */
  ierr = SNESSetFromOptions(snes); CHKERRA(ierr);
  ierr = SNESSolve(snes,x,&its); CHKERRA(ierr);
  PetscPrintf(MPI_COMM_SELF,"number of Newton iterations = %d\n\n", its );

  /* Free data structures */
  ierr = VecDestroy(x); CHKERRA(ierr);     ierr = VecDestroy(r); CHKERRA(ierr);
  ierr = VecDestroy(F); CHKERRA(ierr);     ierr = MatDestroy(J); CHKERRA(ierr);
  ierr = MatDestroy(JPrec); CHKERRA(ierr); ierr = SNESDestroy(snes); CHKERRA(ierr);
  PetscFinalize();

  return 0;
}/* --------------------  Evaluate Function F(x) --------------------- */
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
}/* --------------------  Evaluate Jacobian F'(x) -------------------- */
/* This routine demonstrates the use of different matrices for the Jacobian 
   and preconditioner */
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
}/* --------------------  User-defined preconditioner  -------------------- */
/* This routine demonstrates the use of a user-provided preconditioner and
   is intended as a template for customized versions.  This code implements
   just the null preconditioner, which of course is not recommended for
   general use.
*/
int MatrixFreePreconditioner(void *ctx,Vec x,Vec y)
{
  int ierr;
  ierr = VecCopy(x,y); CHKERRQ(ierr);  
  return 0;
}
