#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex1.c,v 1.10 1999/03/19 21:23:07 bsmith Exp bsmith $";
#endif

static char help[] = "Uses Newton's method to solve a two-variable system.\n\n";

/*T
   Concepts: SNES^Solving a system of nonlinear equations (basic uniprocessor example);
   Routines: SNESCreate(); SNESSetFunction(); SNESSetJacobian(); SNESGetSLES();
   Routines: SNESSolve(); SNESSetFromOptions(); 
   Routines: SLESGetPC(); SLESGetKSP(); KSPSetTolerances(); PCSetType();
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
extern int FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern int FormFunction(SNES,Vec,Vec,void*);

#undef __FUNC__
#define __FUNC__ "main"
int main( int argc, char **argv )
{
  SNES     snes;         /* nonlinear solver context */
  SLES     sles;         /* linear solver context */
  PC       pc;           /* preconditioner context */
  KSP      ksp;          /* Krylov subspace method context */
  Vec      x, r;         /* solution, residual vectors */
  Mat      J;            /* Jacobian matrix */
  int      ierr, its, size;
  Scalar   pfive = .5;

  PetscInitialize( &argc, &argv,(char *)0,help );
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  if (size != 1) SETERRA(1,0,"This is a uniprocessor example only!");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SNESCreate(PETSC_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix and vector data structures; set corresponding routines
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create vectors for solution and nonlinear function
  */
  ierr = VecCreateSeq(PETSC_COMM_SELF,2,&x); CHKERRA(ierr);
  ierr = VecDuplicate(x,&r); CHKERRA(ierr);

  /*
     Create Jacobian matrix data structure
  */
  ierr = MatCreate(PETSC_COMM_SELF,PETSC_DECIDE,PETSC_DECIDE,2,2,&J); CHKERRA(ierr);

  /* 
     Set function evaluation routine and vector.
  */
  ierr = SNESSetFunction(snes,r,FormFunction,PETSC_NULL); CHKERRA(ierr);

  /* 
     Set Jacobian matrix data structure and Jacobian evaluation routine
  */
  ierr = SNESSetJacobian(snes,J,J,FormJacobian,PETSC_NULL); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Set linear solver defaults for this problem. By extracting the
     SLES, KSP, and PC contexts from the SNES context, we can then
     directly call any SLES, KSP, and PC routines to set various options.
  */
  ierr = SNESGetSLES(snes,&sles); CHKERRA(ierr);
  ierr = SLESGetKSP(sles,&ksp); CHKERRA(ierr);
  ierr = SLESGetPC(sles,&pc); CHKERRA(ierr);
  ierr = PCSetType(pc,PCNONE); CHKERRA(ierr);
  ierr = KSPSetTolerances(ksp,1.e-4,PETSC_DEFAULT,PETSC_DEFAULT,20); CHKERRA(ierr);

  /* 
     Set SNES/SLES/KSP/PC runtime options, e.g.,
         -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc>
     These options will override those specified above as long as
     SNESSetFromOptions() is called _after_ any other customization
     routines.
  */
  ierr = SNESSetFromOptions(snes); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Note: The user should initialize the vector, x, with the initial guess
     for the nonlinear solver prior to calling SNESSolve().  In particular,
     to employ an initial guess of zero, the user should explicitly set
     this vector to zero by calling VecSet().
  */
  ierr = VecSet(&pfive,x); CHKERRA(ierr);
  ierr = SNESSolve(snes,x,&its); CHKERRA(ierr);
  PetscPrintf(PETSC_COMM_SELF,"number of Newton iterations = %d\n\n", its);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = VecDestroy(x); CHKERRA(ierr); ierr = VecDestroy(r); CHKERRA(ierr);
  ierr = MatDestroy(J); CHKERRA(ierr); ierr = SNESDestroy(snes); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "FormFunction"
/* 
   FormFunction - Evaluates nonlinear function, F(x).

   Input Parameters:
.  snes - the SNES context
.  x - input vector
.  dummy - optional user-defined context (not used here)

   Output Parameter:
.  f - function vector
 */
int FormFunction(SNES snes,Vec x,Vec f,void *dummy)
{
  int    ierr;
  Scalar *xx, *ff;

  /*
     Get pointers to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr = VecGetArray(x,&xx); CHKERRQ(ierr);
  ierr = VecGetArray(f,&ff); CHKERRQ(ierr);

  /*
     Compute function
  */
  ff[0] = xx[0]*xx[0] + xx[0]*xx[1] - 3.0;
  ff[1] = xx[0]*xx[1] + xx[1]*xx[1] - 6.0;

  /*
     Restore vectors
  */
  ierr = VecRestoreArray(x,&xx); CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&ff); CHKERRQ(ierr); 

  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "FormJacobian"
/*
   FormJacobian - Evaluates Jacobian matrix.

   Input Parameters:
.  snes - the SNES context
.  x - input vector
.  dummy - optional user-defined context (not used here)

   Output Parameters:
.  jac - Jacobian matrix
.  B - optionally different preconditioning matrix
.  flag - flag indicating matrix structure
*/
int FormJacobian(SNES snes,Vec x,Mat *jac,Mat *B,MatStructure *flag,void *dummy)
{
  Scalar *xx, A[4];
  int    ierr, idx[2] = {0,1};

  /*
     Get pointer to vector data
  */
  ierr = VecGetArray(x,&xx); CHKERRQ(ierr);

  /*
     Compute Jacobian entries and insert into matrix.
      - Since this is such a small problem, we set all entries for
        the matrix at once.
  */
  A[0] = 2.0*xx[0] + xx[1]; A[1] = xx[0];
  A[2] = xx[1]; A[3] = xx[0] + 2.0*xx[1];
  ierr = MatSetValues(*jac,2,idx,2,idx,A,INSERT_VALUES); CHKERRQ(ierr);
  *flag = SAME_NONZERO_PATTERN;

  /*
     Restore vector
  */
  ierr = VecRestoreArray(x,&xx); CHKERRQ(ierr);

  /*
     Assemble matrix
  */
  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  return 0;
}


