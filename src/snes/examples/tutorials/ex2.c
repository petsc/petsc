#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex2.c,v 1.68 1999/09/27 21:31:55 bsmith Exp bsmith $";
#endif

static char help[] = "Uses Newton-like methods to solve u'' + u^{2} = f.\n\
This example employs a user-defined monitoring routine.\n\n";

/*T
   Concepts: SNES^Solving a system of nonlinear equations (basic uniprocessor example)
   Concepts: SNES^Setting a user-defined monitoring routine
   Routines: SNESCreate(); SNESSetFunction(); SNESSetJacobian(); SNESSolve();
   Routines: SNESGetTolerances(); SNESSetFromOptions(); SNESSetMonitor();
   Routines: SNESGetSolution(); ViewerDrawOpen(); PetscObjectSetName();
   Processors: 1
T*/

/* 
   Include "draw.h" so that we can use PETSc drawing routines.
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
extern int FormInitialGuess(Vec);
extern int Monitor(SNES,int,double,void *);

/*
   User-defined context for monitoring
*/
typedef struct {
   Viewer viewer;
} MonitorCtx;

#undef __FUNC__
#define __FUNC__ "main"
int main( int argc, char **argv )
{
  SNES       snes;                   /* SNES context */
  Vec        x, r, F, U;             /* vectors */
  Mat        J;                      /* Jacobian matrix */
  MonitorCtx monP;                   /* monitoring context */
  int        ierr, its, n = 5, i, flg, maxit, maxf, size;
  Scalar     h, xp, v, none = -1.0;
  double     atol, rtol, stol, norm;

  PetscInitialize( &argc, &argv,(char *)0,help );
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRA(ierr);
  if (size != 1) SETERRA(1,0,"This is a uniprocessor example only!");
  ierr = OptionsGetInt(PETSC_NULL,"-n",&n,&flg);CHKERRA(ierr);
  h = 1.0/(n-1);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SNESCreate(PETSC_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes);CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures; set function evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Note that we form 1 vector from scratch and then duplicate as needed.
  */
  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,n,&x);CHKERRA(ierr);
  ierr = VecSetFromOptions(x);CHKERRA(ierr);
  ierr = VecDuplicate(x,&r);CHKERRA(ierr);
  ierr = VecDuplicate(x,&F);CHKERRA(ierr);
  ierr = VecDuplicate(x,&U);CHKERRA(ierr); 

  /* 
     Set function evaluation routine and vector
  */
  ierr = SNESSetFunction(snes,r,FormFunction,(void*)F); CHKERRA(ierr);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,&J);CHKERRA(ierr);

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

  ierr = SNESSetJacobian(snes,J,J,FormJacobian,PETSC_NULL);CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Set an optional user-defined monitoring routine
  */
  ierr = ViewerDrawOpen(PETSC_COMM_WORLD,0,0,0,0,400,400,&monP.viewer);CHKERRA(ierr);
  ierr = SNESSetMonitor(snes,Monitor,&monP,0);CHKERRA(ierr); 

  /*
     Set names for some vectors to facilitate monitoring (optional)
  */
  ierr = PetscObjectSetName((PetscObject)x,"Approximate Solution");CHKERRA(ierr);
  ierr = PetscObjectSetName((PetscObject)U,"Exact Solution");CHKERRA(ierr);

  /* 
     Set SNES/SLES/KSP/PC runtime options, e.g.,
         -snes_view -snes_monitor -ksp_type <ksp> -pc_type <pc>
  */
  ierr = SNESSetFromOptions(snes);CHKERRA(ierr);

  /* 
     Print parameters used for convergence testing (optional) ... just
     to demonstrate this routine; this information is also printed with
     the option -snes_view
  */
  ierr = SNESGetTolerances(snes,&atol,&rtol,&stol,&maxit,&maxf);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"atol=%g, rtol=%g, stol=%g, maxit=%d, maxf=%d\n",atol,rtol,stol,maxit,maxf);CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize application:
     Store right-hand-side of PDE and exact solution
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  xp = 0.0;
  for ( i=0; i<n; i++ ) {
    v = 6.0*xp + PetscPowScalar(xp+1.e-12,6.0); /* +1.e-12 is to prevent 0^6 */
    ierr = VecSetValues(F,1,&i,&v,INSERT_VALUES);CHKERRA(ierr);
    v= xp*xp*xp;
    ierr = VecSetValues(U,1,&i,&v,INSERT_VALUES);CHKERRA(ierr);
    xp += h;
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Note: The user should initialize the vector, x, with the initial guess
     for the nonlinear solver prior to calling SNESSolve().  In particular,
     to employ an initial guess of zero, the user should explicitly set
     this vector to zero by calling VecSet().
  */
  ierr = FormInitialGuess(x);CHKERRA(ierr);
  ierr = SNESSolve(snes,x,&its);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"number of Newton iterations = %d\n\n", its );CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Check solution and clean up
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Check the error
  */
  ierr = VecAXPY(&none,U,x);CHKERRA(ierr);
  ierr  = VecNorm(x,NORM_2,&norm);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %A, Iterations %d\n",norm,its);CHKERRA(ierr);


  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(x);CHKERRA(ierr);  ierr = VecDestroy(r);CHKERRA(ierr);
  ierr = VecDestroy(U);CHKERRA(ierr);  ierr = VecDestroy(F);CHKERRA(ierr);
  ierr = MatDestroy(J);CHKERRA(ierr);  ierr = SNESDestroy(snes);CHKERRA(ierr);
  ierr = ViewerDestroy(monP.viewer);CHKERRA(ierr);
  PetscFinalize();

  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "FormInitialGuess"
/*
   FormInitialGuess - Computes initial guess.

   Input/Output Parameter:
.  x - the solution vector
*/
int FormInitialGuess(Vec x)
{
   int    ierr;
   Scalar pfive = .50;
   ierr = VecSet(&pfive,x);CHKERRQ(ierr);
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
   ierr = VecGetArray(x,&xx);CHKERRQ(ierr);
   ierr = VecGetArray(f,&ff);CHKERRQ(ierr);
   ierr = VecGetArray(g,&gg);CHKERRQ(ierr);

  /*
     Compute function
  */
   ierr = VecGetSize(x,&n);CHKERRQ(ierr);
   d = (double) (n - 1); d = d*d;
   ff[0]   = xx[0];
   for ( i=1; i<n-1; i++ ) {
     ff[i] = d*(xx[i-1] - 2.0*xx[i] + xx[i+1]) + xx[i]*xx[i] - gg[i];
   }
   ff[n-1] = xx[n-1] - 1.0;

  /*
     Restore vectors
  */
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);
  ierr = VecRestoreArray(f,&ff);CHKERRQ(ierr);
  ierr = VecRestoreArray(g,&gg);CHKERRQ(ierr);
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

int FormJacobian(SNES snes,Vec x,Mat *jac,Mat *B,MatStructure*flag,void *dummy)
{
  Scalar *xx, A[3], d;
  int    i, n, j[3], ierr;

  /*
     Get pointer to vector data
  */
  ierr = VecGetArray(x,&xx);CHKERRQ(ierr);

  /*
     Compute Jacobian entries and insert into matrix.
      - Note that in this case we set all elements for a particular
        row at once.
  */
  ierr = VecGetSize(x,&n);CHKERRQ(ierr);
  d = (double)(n - 1); d = d*d;

  /*
     Interior grid points
  */
  for ( i=1; i<n-1; i++ ) {
    j[0] = i - 1; j[1] = i; j[2] = i + 1; 
    A[0] = A[2] = d; A[1] = -2.0*d + 2.0*xx[i];
    ierr = MatSetValues(*jac,1,&i,3,j,A,INSERT_VALUES);CHKERRQ(ierr);
  }

  /*
     Boundary points
  */
  i = 0;   A[0] = 1.0; 
  ierr = MatSetValues(*jac,1,&i,1,&i,A,INSERT_VALUES);CHKERRQ(ierr);
  i = n-1; A[0] = 1.0; 
  ierr = MatSetValues(*jac,1,&i,1,&i,A,INSERT_VALUES);CHKERRQ(ierr);

  /*
     Restore vector
  */
  ierr = VecRestoreArray(x,&xx);CHKERRQ(ierr);

  /*
     Assemble matrix
  */
  ierr = MatAssemblyBegin(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "Monitor"
/*
   Monitor - User-defined monitoring routine that views the
   current iterate with an x-window plot.

   Input Parameters:
   snes - the SNES context
   its - iteration number
   norm - 2-norm function value (may be estimated)
   ctx - optional user-defined context for private data for the 
         monitor routine, as set by SNESSetMonitor()

   Note:
   See the manpage for ViewerDrawOpen() for useful runtime options,
   such as -nox to deactivate all x-window output.
 */
int Monitor(SNES snes,int its,double fnorm,void *ctx)
{
  int        ierr;
  MonitorCtx *monP = (MonitorCtx*) ctx;
  Vec        x;

  ierr = PetscPrintf(PETSC_COMM_WORLD,"iter = %d, SNES Function norm %g\n",its,fnorm);CHKERRQ(ierr);
  ierr = SNESGetSolution(snes,&x);CHKERRQ(ierr);
  ierr = VecView(x,monP->viewer);CHKERRQ(ierr);
  return 0;
}
