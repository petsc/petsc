#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex3func.c,v 1.1 1997/04/08 03:56:46 bsmith Exp bsmith $";
#endif

#include "snes.h"

/* 
   User-defined routines
*/
int FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
int FormFunction(SNES,Vec,Vec,void*);
int FormInitialGuess(Vec);
int Monitor(SNES,int,double,void *);

/*
   User-defined context for monitoring
*/
typedef struct {
   Viewer viewer;
} MonitorCtx;

int Function(Vec F, Vec x)
{
  SNES       snes;                   /* SNES context */
  Vec        r;                      /* vectors */
  Mat        J;                      /* Jacobian matrix */
  MonitorCtx monP;                   /* monitoring context */
  int        ierr, its, n, maxit, maxf;
  Scalar     h;
  double     atol, rtol, stol;

  ierr = VecGetSize(x,&n); CHKERRQ(ierr);

  h = 1.0/(n-1);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver contex, norm;t
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SNESCreate(PETSC_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures; set function evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Duplicate vector as needed.
  */
  ierr = VecDuplicate(F,&r); CHKERRA(ierr);


  /* 
     Set function evaluation routine and vector
  */
  ierr = SNESSetFunction(snes,r,FormFunction,(void*)F);  CHKERRA(ierr);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,n,n,&J); CHKERRA(ierr);

  /* 
     Set Jacobian matrix data structure and default Jacobian evaluation
     routine. User can override with:
     -snes_fd : default finite differencing approximation of Jacobian
     -snes_mf : matrix-free Newton-Krylov method with no preconditioning
                (unless user explicitly sets preconditioner) 
     -snes_mf_operator : form preconditioning matrix as set by the user,
                         but use matrix-free approx for Jacobian-vector
                         products within Newton-Krylov method
  */

  ierr = SNESSetJacobian(snes,J,J,FormJacobian,PETSC_NULL); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Set an optional user-defined monitoring routine
  */

  /*
     Set names for some vectors to facilitate monitoring (optional)
  */
  PetscObjectSetName((PetscObject)x,"Approximate Solution");

  /* 
     Set SNES//KSP/PC runtime options, e.g.,
         -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc>
  */
  ierr = SNESSetFromOptions(snes); CHKERRA(ierr);

  /* 
     Print parameters used for convergence testing (optional) ... just
     to demonstrate this routine; this information is also printed with
     the option -snes_view
  */
  ierr = SNESGetTolerances(snes,&atol,&rtol,&stol,&maxit,&maxf); CHKERRA(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"atol=%g, rtol=%g, stol=%g, maxit=%d, maxf=%d\n",
     atol,rtol,stol,maxit,maxf);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Note: The user should initialize the vector, x, with the initial guess
     for the nonlinear solver prior to calling SNESSolve().  In particular,
     to employ an initial guess of zero, the user should explicitly set
     this vector to zero by calling VecSet().
  */
  ierr = FormInitialGuess(x); CHKERRA(ierr);
  ierr = SNESSolve(snes,x,&its); CHKERRA(ierr);
  PetscPrintf(PETSC_COMM_WORLD,"number of Newton iterations = %d\n\n", its );

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Check solution and clean up
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */


  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(r); CHKERRA(ierr);
  ierr = MatDestroy(J); CHKERRA(ierr);  ierr = SNESDestroy(snes); CHKERRA(ierr);

  return 0;
}
/* ------------------------------------------------------------------- */
/*
   FormInitialGuess - Computes initial guess.

   Input/Output Parameter:
.  x - the solution vector
*/
int FormInitialGuess(Vec x)
{
   int    ierr;
   Scalar pfive = 5.50;
   ierr = VecSet(&pfive,x); CHKERRQ(ierr);
   return 0;
}
/* ------------------------------------------------------------------- */
/* 
   FormFunction - Evaluates nonlinear function, F(x).

   Input Parameters:
.  snes - the SNES context
.  x - input vector
.  ctx - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  f - function vector

   Note:
   The user-defined context can contain any application-specific data
   needed for the function evaluation (such as various parameters, work
   vectors, and grid information).  In this program the context is just
   a vector containing the right-hand-side of the discretized PDE.
 */

int FormFunction(SNES snes,Vec x,Vec f,void *ctx)
{
   Vec    g = (Vec)ctx;
   Scalar *xx, *ff,*gg,d;
   int    i, ierr, n;

  /*
     Get pointers to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
   ierr = VecGetArray(x,&xx); CHKERRQ(ierr);
   ierr = VecGetArray(f,&ff); CHKERRQ(ierr);
   ierr = VecGetArray(g,&gg); CHKERRQ(ierr);

  /*
     Compute function
  */
   ierr = VecGetSize(x,&n); CHKERRQ(ierr);
   d = (double) (n - 1); d = d*d;
   ff[0]   = xx[0];
   for ( i=1; i<n-1; i++ ) {
     /*     ff[i] = d*(xx[i-1] - 2.0*xx[i] + xx[i+1]) + xx[i]*xx[i] - gg[i]; */
     ff[i] = d*(xx[i-1] - 2.0*xx[i] + xx[i+1])  - gg[i];
   }
   ff[n-1] = xx[n-1] - 1.0;

  /*
     Restore vectors
  */
  ierr = VecRestoreArray(x,&xx); CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&ff); CHKERRQ(ierr);
  ierr = VecRestoreArray(g,&gg); CHKERRQ(ierr);
  return 0;
}
/* ------------------------------------------------------------------- */
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

int FormJacobian(SNES snes,Vec x,Mat *jac,Mat *B,MatStructure*flag,void *dummy)
{
  Scalar *xx, A[3], d;
  int    i, n, j[3], ierr;

  /*
     Get pointer to vector data
  */
  ierr = VecGetArray(x,&xx); CHKERRQ(ierr);

  /*
     Compute Jacobian entries and insert into matrix.
      - Note that in this case we set all elements for a particular
        row at once.
  */
  ierr = VecGetSize(x,&n); CHKERRQ(ierr);
  d = (double)(n - 1); d = d*d;

  /*
     Interior grid points
  */
  for ( i=1; i<n-1; i++ ) {
    j[0] = i - 1; j[1] = i; j[2] = i + 1; 
    /*    A[0] = A[2] = d; A[1] = -2.0*d + 2.0*xx[i];  */
    A[0] = A[2] = d; A[1] = -2.0*d;
    ierr = MatSetValues(*jac,1,&i,3,j,A,INSERT_VALUES); CHKERRQ(ierr);
  }

  /*
     Boundary points
  */
  i = 0;   A[0] = 1.0; 
  ierr = MatSetValues(*jac,1,&i,1,&i,A,INSERT_VALUES); CHKERRQ(ierr);
  i = n-1; A[0] = 1.0; 
  ierr = MatSetValues(*jac,1,&i,1,&i,A,INSERT_VALUES); CHKERRQ(ierr);

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
