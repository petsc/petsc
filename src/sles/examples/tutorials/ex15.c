#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex15.c,v 1.10 1999/05/04 20:35:25 balay Exp balay $";
#endif

static char help[] = "Solves a linear system in parallel with SLES.  Also\n\
illustrates setting a user-defined shell preconditioner and using the\n\
macro __FUNC__ to define routine names for use in error handling.\n\
Input parameters include:\n\
  -user_defined_pc : Activate a user-defined preconditioner\n\n";

/*T
   Concepts: SLES^Solving a system of linear equations (basic parallel example);
   Concepts: PC^Setting a user-defined "shell" preconditioner
   Concepts: Error Handling^Using the macro __FUNC__ to define routine names;
   Routines: SLESCreate(); SLESSetOperators(); SLESSetFromOptions();
   Routines: SLESSolve(); SLESGetKSP(); SLESGetPC(); KSPSetTolerances(); 
   Routines: PCSetType(); PCShellSetApply(); PCShellSetName();
   Processors: n
T*/

/* 
  Include "sles.h" so that we can use SLES solvers.  Note that this file
  automatically includes:
     petsc.h  - base PETSc routines   vec.h - vectors
     sys.h    - system routines       mat.h - matrices
     is.h     - index sets            ksp.h - Krylov subspace methods
     viewer.h - viewers               pc.h  - preconditioners
*/
#include "sles.h"

/* Define context for user-provided preconditioner */
typedef struct {
  Vec diag;
} SampleShellPC;

/* Declare routines for user-provided preconditioner */
extern int SampleShellPCCreate(SampleShellPC**);
extern int SampleShellPCSetUp(SampleShellPC*,Mat,Vec);
extern int SampleShellPCApply(void*,Vec x,Vec y);
extern int SampleShellPCDestroy(SampleShellPC*);

/* 
   User-defined routines.  Note that immediately before each routine below,
   we define the macro __FUNC__ to be a string containing the routine name.
   If defined, this macro is used in the PETSc error handlers to provide a
   complete traceback of routine names.  All PETSc library routines use this
   macro, and users can optionally employ it as well in their application
   codes.  Note that users can get a traceback of PETSc errors regardless of
   whether they define __FUNC__ in application codes; this macro merely
   provides the added traceback detail of the application routine names.
*/

#undef __FUNC__  
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Vec           x, b, u;   /* approx solution, RHS, exact solution */
  Mat           A;         /* linear system matrix */
  SLES          sles;      /* linear solver context */
  PC            pc;        /* preconditioner context */
  KSP           ksp;       /* Krylov subspace method context */
  double        norm;      /* norm of solution error */
  SampleShellPC *shell;    /* user-defined preconditioner context */
  Scalar        v, one = 1.0, none = -1.0;
  int           i, j, I, J, Istart, Iend, ierr, m = 8, n = 7;
  int           user_defined_pc, its, flg;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = OptionsGetInt(PETSC_NULL,"-m",&m,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg);CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* 
     Create parallel matrix, specifying only its global dimensions.
     When using MatCreate(), the matrix format can be specified at
     runtime. Also, the parallel partioning of the matrix is
     determined by PETSc at runtime.
  */
  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n,&A);CHKERRA(ierr);

  /* 
     Currently, all PETSc parallel matrix formats are partitioned by
     contiguous chunks of rows across the processors.  Determine which
     rows of the matrix are locally owned. 
  */
  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRA(ierr);

  /* 
     Set matrix elements for the 2-D, five-point stencil in parallel.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly). 
      - Always specify global rows and columns of matrix entries.
   */
  for ( I=Istart; I<Iend; I++ ) { 
    v = -1.0; i = I/n; j = I - i*n;  
    if ( i>0 )   {J = I - n; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
    if ( i<m-1 ) {J = I + n; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
    if ( j>0 )   {J = I - 1; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
    if ( j<n-1 ) {J = I + 1; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
    v = 4.0; MatSetValues(A,1,&I,1,&I,&v,INSERT_VALUES);
  }

  /* 
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd()
     Computations can be done while messages are in transition
     by placing code between these two statements.
  */
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  /* 
     Create parallel vectors.
      - When using VecCreate() and VecSetFromOptions(), we specify only the vector's global
        dimension; the parallel partitioning is determined at runtime. 
      - Note: We form 1 vector from scratch and then duplicate as needed.
  */
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,m*n,&u);CHKERRA(ierr);
  ierr = VecSetFromOptions(u);CHKERRA(ierr);
  ierr = VecDuplicate(u,&b);CHKERRA(ierr); 
  ierr = VecDuplicate(b,&x);CHKERRA(ierr);

  /* 
     Set exact solution; then compute right-hand-side vector.
  */
  ierr = VecSet(&one,u);CHKERRA(ierr);
  ierr = MatMult(A,u,b);CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Create linear solver context
  */
  ierr = SLESCreate(PETSC_COMM_WORLD,&sles);CHKERRA(ierr);

  /* 
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);

  /* 
     Set linear solver defaults for this problem (optional).
     - By extracting the KSP and PC contexts from the SLES context,
       we can then directly call any KSP and PC routines
       to set various options.
  */
  ierr = SLESGetKSP(sles,&ksp);CHKERRA(ierr);
  ierr = SLESGetPC(sles,&pc);CHKERRA(ierr);
  ierr = KSPSetTolerances(ksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,
         PETSC_DEFAULT);CHKERRA(ierr);

  /*
     Set a user-defined "shell" preconditioner if desired
  */
  ierr = OptionsHasName(PETSC_NULL,"-user_defined_pc",&user_defined_pc);CHKERRA(ierr);
  if (user_defined_pc) {
    /* (Required) Indicate to PETSc that we're using a "shell" preconditioner */
    ierr = PCSetType(pc,PCSHELL);CHKERRA(ierr);

    /* (Optional) Create a context for the user-defined preconditioner; this
       context can be used to contain any application-specific data. */
    ierr = SampleShellPCCreate(&shell);CHKERRA(ierr);

    /* (Required) Set the user-defined routine for applying the preconditioner */
    ierr = PCShellSetApply(pc,SampleShellPCApply,(void*)shell);CHKERRA(ierr);

    /* (Optional) Set a name for the preconditioner, used for PCView() */
    ierr = PCShellSetName(pc,"MyPreconditioner");CHKERRQ(ierr);

    /* (Optional) Do any setup required for the preconditioner */
    ierr = SampleShellPCSetUp(shell,A,x);CHKERRA(ierr);

  } else {
    ierr = PCSetType(pc,PCJACOBI);CHKERRA(ierr);
  }

  /* 
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    SLESSetFromOptions() is called _after_ any other customization
    routines.
  */
  ierr = SLESSetFromOptions(sles);CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SLESSolve(sles,b,x,&its);CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Check the error
  */
  ierr = VecAXPY(&none,u,x);CHKERRA(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRA(ierr);
  if (norm > 1.e-12)
    PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g iterations %d\n",norm,its);
  else 
    PetscPrintf(PETSC_COMM_WORLD,"Norm of error < 1.e-12 Iterations %d\n",its);

  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = SLESDestroy(sles);CHKERRA(ierr);
  ierr = VecDestroy(u);CHKERRA(ierr);  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(b);CHKERRA(ierr);  ierr = MatDestroy(A);CHKERRA(ierr);

  if (user_defined_pc) {
    ierr = SampleShellPCDestroy(shell);CHKERRA(ierr);
  }

  PetscFinalize();
  return 0;

}

/***********************************************************************/
/*          Routines for a user-defined shell preconditioner           */
/***********************************************************************/

#undef __FUNC__  
#define __FUNC__ "SampleShellPCCreate"
/*
   SampleShellPCCreate - This routine creates a user-defined
   preconditioner context.

   Output Parameter:
.  shell - user-defined preconditioner context
*/
int SampleShellPCCreate(SampleShellPC **shell)
{
  SampleShellPC *newctx = PetscNew(SampleShellPC);CHKPTRQ(newctx);
  newctx->diag = 0;
  *shell = newctx;
  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "SampleShellPCSetUp"
/*
   SampleShellPCSetUp - This routine sets up a user-defined
   preconditioner context.  

   Input Parameters:
.  shell - user-defined preconditioner context
.  pmat  - preconditioner matrix
.  x     - vector

   Output Parameter:
.  shell - fully set up user-defined preconditioner context

   Notes:
   In this example, we define the shell preconditioner to be Jacobi's
   method.  Thus, here we create a work vector for storing the reciprocal
   of the diagonal of the preconditioner matrix; this vector is then
   used within the routine SampleShellPCApply().
*/
int SampleShellPCSetUp(SampleShellPC *shell,Mat pmat,Vec x)
{
  Vec diag;
  int ierr;

  ierr = VecDuplicate(x,&diag);CHKERRQ(ierr);
  ierr = MatGetDiagonal(pmat,diag);CHKERRQ(ierr);
  ierr = VecReciprocal(diag);CHKERRQ(ierr);
  shell->diag = diag;

  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "SampleShellPCApply"
/*
   SampleShellPCApply - This routine demonstrates the use of a
   user-provided preconditioner.

   Input Parameters:
.  ctx - optional user-defined context, as set by PCShellSetApply()
.  x - input vector

   Output Parameter:
.  y - preconditioned vector

   Notes:
   Note that the PCSHELL preconditioner passes a void pointer as the
   first input argument.  This can be cast to be the whatever the user
   has set (via PCSetShellApply()) the application-defined context to be.

   This code implements the Jacobi preconditioner, merely as an
   example of working with a PCSHELL.  Note that the Jacobi method
   is already provided within PETSc.
*/
int SampleShellPCApply(void *ctx,Vec x,Vec y)
{
  SampleShellPC *shell = (SampleShellPC *) ctx;
  int           ierr;

  ierr = VecPointwiseMult(x,shell->diag,y);CHKERRQ(ierr);

  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNC__  
#define __FUNC__ "SampleShellPCDestroy"
/*
   SampleShellPCDestroy - This routine destroys a user-defined
   preconditioner context.

   Input Parameter:
.  shell - user-defined preconditioner context
*/
int SampleShellPCDestroy(SampleShellPC *shell)
{
  int ierr;

  ierr = VecDestroy(shell->diag);CHKERRQ(ierr);
  ierr = PetscFree(shell);CHKERRQ(ierr);

  return 0;
}
