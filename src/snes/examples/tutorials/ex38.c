
static char help[] = "u`` + u^{2} = f. Different matrices for the Jacobian and the preconditioner.\n\
Demonstrates the use of matrix-free Newton-Krylov methods in conjunction\n\
with a user-provided preconditioner.  Input arguments are:\n\
   -snes_mf : Use matrix-free Newton methods\n\
   -user_precond : Employ a user-defined preconditioner.  Used only with\n\
                   matrix-free methods in this example.\n\n";
/*
  Modified from ex6.c by Mike McCourt <mccomic@iit.edu>
   for testing SNESLineSearchSet() 
 */

/*T
   Concepts: SNES^different matrices for the Jacobian and preconditioner;
   Concepts: SNES^matrix-free methods
   Concepts: SNES^user-provided preconditioner;
   Concepts: matrix-free methods
   Concepts: user-provided preconditioner;
   Processors: 1
T*/

/* 
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/
#include <iostream>
using namespace std;
#include <petscsnes.h>

struct AppCtx{int testint;};

/* 
   User-defined routines
*/
PetscErrorCode FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
PetscErrorCode MatrixFreePreconditioner(PC,Vec,Vec);
PetscErrorCode FormLineSearch(SNES,void*,Vec,Vec,Vec,Vec,Vec,PetscReal,PetscReal*,PetscReal*,PetscBool *);

int main(int argc,char **argv)
{
  SNES           snes;                /* SNES context */
  KSP            ksp;                /* KSP context */
  PC             pc;                  /* PC context */
  Vec            x,r,F;               /* vectors */
  Mat            J,JPrec;             /* Jacobian,preconditioner matrices */
  PetscErrorCode ierr;
  PetscInt       it,n = 5,i;
  PetscMPIInt    size;
  PetscInt       *Shistit = 0,Khistl = 200,Shistl = 10;
  PetscReal      h,xp = 0.0,*Khist = 0,*Shist = 0;
  PetscScalar    v,pfive = .5;
  PetscBool      flg;
  AppCtx	 user;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_SELF,1,"This is a uniprocessor example only!");
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  h = 1.0/(n-1);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures; set function evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = VecCreate(PETSC_COMM_SELF,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&F);CHKERRQ(ierr);

  ierr = SNESSetFunction(snes,r,FormFunction,(void*)F);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structures; set Jacobian evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,3,PETSC_NULL,&J);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,1,PETSC_NULL,&JPrec);CHKERRQ(ierr);

  /*
     Note that in this case we create separate matrices for the Jacobian
     and preconditioner matrix.  Both of these are computed in the
     routine FormJacobian()
  */
  ierr = SNESSetJacobian(snes,J,JPrec,FormJacobian,0);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* Set preconditioner for matrix-free method */
  flg  = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-snes_mf",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) {
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PetscOptionsHasName(PETSC_NULL,"-user_precond",&flg);CHKERRQ(ierr);
    if (flg) { /* user-defined precond */
      ierr = PCSetType(pc,PCSHELL);CHKERRQ(ierr);
      ierr = PCShellSetApply(pc,MatrixFreePreconditioner);CHKERRQ(ierr);
    } else {ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);}
  }

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  user.testint = 0;
  ierr = SNESLineSearchSet(snes,FormLineSearch,&user);CHKERRQ(ierr);

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
  ierr = PetscOptionsHasName(PETSC_NULL,"-rhistory",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = SNESGetKSP(snes,&ksp);CHKERRQ(ierr);
    ierr = PetscMalloc(Khistl*sizeof(PetscReal),&Khist);CHKERRQ(ierr);
    ierr = KSPSetResidualHistory(ksp,Khist,Khistl,PETSC_FALSE);CHKERRQ(ierr);
    ierr = PetscMalloc(Shistl*sizeof(PetscReal),&Shist);CHKERRQ(ierr);
    ierr = PetscMalloc(Shistl*sizeof(PetscInt),&Shistit);CHKERRQ(ierr);
    ierr = SNESSetConvergenceHistory(snes,Shist,Shistit,Shistl,PETSC_FALSE);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize application:
     Store right-hand-side of PDE and exact solution
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  xp = 0.0;
  for (i=0; i<n; i++) {
    v = 6.0*xp + pow(xp+1.e-12,6.0); /* +1.e-12 is to prevent 0^6 */
    ierr = VecSetValues(F,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
    xp += h;
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = VecSet(x,pfive);CHKERRQ(ierr);
  ierr = SNESSolve(snes,PETSC_NULL,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&it);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"SNES iterations = %D\n\n",it);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-rhistory",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = KSPGetResidualHistory(ksp,PETSC_NULL,&Khistl);CHKERRQ(ierr);
    ierr = PetscRealView(Khistl,Khist,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    ierr = PetscFree(Khist);CHKERRQ(ierr);CHKERRQ(ierr);
    ierr = SNESGetConvergenceHistory(snes,PETSC_NULL,PETSC_NULL,&Shistl);CHKERRQ(ierr);
    ierr = PetscRealView(Shistl,Shist,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    ierr = PetscIntView(Shistl,Shistit,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    ierr = PetscFree(Shist);CHKERRQ(ierr);
    ierr = PetscFree(Shistit);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = VecDestroy(&x);CHKERRQ(ierr);     ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);     ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = MatDestroy(&JPrec);CHKERRQ(ierr); ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = PetscFinalize();

  return 0;
}

PetscErrorCode FormLineSearch(SNES snes,void* user,Vec X,Vec F,Vec G,Vec Y,Vec W,PetscReal fnorm,
                              PetscReal *ynorm,PetscReal *gnorm,PetscBool  *flag)
{
  PetscErrorCode ierr;
  PetscScalar mone=-1.0;
  AppCtx *myguy = (AppCtx*)user;
  *flag=PETSC_TRUE;

  cout << "Inside FormLineSearch \n user.testint=" << myguy->testint << endl;
  ierr=VecNorm(Y,NORM_2,ynorm);
  ierr=VecWAXPY(W,mone,Y,X); /* W = -Y + X */
  ierr=SNESComputeFunction(snes,W,G);CHKERRQ(ierr);
  ierr=VecNorm(G,NORM_2,gnorm);CHKERRQ(ierr);
  myguy->testint++;
  return ierr;
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
PetscErrorCode FormFunction(SNES snes,Vec x,Vec f,void *dummy)
{
  PetscScalar    *xx,*ff,*FF,d;
  PetscErrorCode ierr;
  PetscInt       i,n;

  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetArray(f,&ff);CHKERRQ(ierr);
  ierr = VecGetArray((Vec)dummy,&FF);CHKERRQ(ierr);
  ierr = VecGetSize(x,&n);CHKERRQ(ierr);
  d = (PetscReal)(n - 1); d = d*d;
  ff[0]   = xx[0];
  for (i=1; i<n-1; i++) {
    ff[i] = d*(xx[i-1] - 2.0*xx[i] + xx[i+1]) + xx[i]*xx[i] - FF[i];
  }
  ff[n-1] = xx[n-1] - 1.0;
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&ff);CHKERRQ(ierr);
  ierr = VecRestoreArray((Vec)dummy,&FF);CHKERRQ(ierr);
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
PetscErrorCode FormJacobian(SNES snes,Vec x,Mat *jac,Mat *prejac,MatStructure *flag,void *dummy)
{
  PetscScalar    *xx,A[3],d;
  PetscInt       i,n,j[3];
  PetscErrorCode ierr;

  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
  ierr = VecGetSize(x,&n);CHKERRQ(ierr);
  d = (PetscReal)(n - 1); d = d*d;

  /* Form Jacobian.  Also form a different preconditioning matrix that 
     has only the diagonal elements. */
  i = 0; A[0] = 1.0; 
  ierr = MatSetValues(*jac,1,&i,1,&i,&A[0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValues(*prejac,1,&i,1,&i,&A[0],INSERT_VALUES);CHKERRQ(ierr);
  for (i=1; i<n-1; i++) {
    j[0] = i - 1; j[1] = i;                   j[2] = i + 1; 
    A[0] = d;     A[1] = -2.0*d + 2.0*xx[i];  A[2] = d; 
    ierr = MatSetValues(*jac,1,&i,3,j,A,INSERT_VALUES);CHKERRQ(ierr);
    ierr = MatSetValues(*prejac,1,&i,1,&i,&A[1],INSERT_VALUES);CHKERRQ(ierr);
  }
  i = n-1; A[0] = 1.0; 
  ierr = MatSetValues(*jac,1,&i,1,&i,&A[0],INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatSetValues(*prejac,1,&i,1,&i,&A[0],INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(*prejac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*prejac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;
  return 0;
}
/* ------------------------------------------------------------------- */
/*
   MatrixFreePreconditioner - This routine demonstrates the use of a
   user-provided preconditioner.  This code implements just the null
   preconditioner, which of course is not recommended for general use.

   Input Parameters:
+  pc - preconditioner object
-  x - input vector

   Output Parameter:
.  y - preconditioned vector
*/
PetscErrorCode MatrixFreePreconditioner(PC pc,Vec x,Vec y)
{
  PetscErrorCode ierr;
  ierr = VecCopy(x,y);CHKERRQ(ierr);  
  return 0;
}
