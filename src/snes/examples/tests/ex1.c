#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex4.c,v 1.67 1999/05/04 20:36:19 balay Exp bsmith $";
#endif

/* Program usage:  ex4 [-help] [all PETSc options] */

static char help[] = "Solves a nonlinear system on 1 processor with SNES. We\n\
solve the Bratu (SFI - solid fuel ignition) problem in a 2D rectangular domain.\n\
This example also illustrates the use of matrix coloring.  Runtime options include:\n\
  -par <parameter>, where <parameter> indicates the problem's nonlinearity\n\
     problem SFI:  <parameter> = Bratu parameter (0 <= par <= 6.81)\n\
  -mx <xg>, where <xg> = number of grid points in the x-direction\n\
  -my <yg>, where <yg> = number of grid points in the y-direction\n\n";

/*T
   Concepts: SNES^Solving a system of nonlinear equations (sequential Bratu example);
   Routines: SNESCreate(); SNESSetFunction(); SNESSetJacobian();
   Routines: SNESSolve(); SNESSetFromOptions(); SNESSetConvergenceHistory();
   Routines: MatGetColoring(); MatFDColoringCreate(); MatFDColoringSetFromOptions();
   Routines: MatFDColoringDestroy(); ISColoringDestroy();SNESDefaultComputeJacobianColor();
   Routines: DrawOpenX(); DrawTensorContour(); DrawDestroy();
   Processors: 1
T*/

/* ------------------------------------------------------------------------

    Solid Fuel Ignition (SFI) problem.  This problem is modeled by
    the partial differential equation
  
            -Laplacian u - lambda*exp(u) = 0,  0 < x,y < 1 ,
  
    with boundary conditions
   
             u = 0  for  x = 0, x = 1, y = 0, y = 1.
  
    A finite difference approximation with the usual 5-point stencil
    is used to discretize the boundary value problem to obtain a nonlinear 
    system of equations.

    The parallel version of this code is snes/examples/tutorials/ex5.c

  ------------------------------------------------------------------------- */

/* 
   Include "draw.h" so that we can use PETSc drawing routines.
   Include "snes.h" so that we can use SNES solvers.  Note that
   this file automatically includes:
     petsc.h  - base PETSc routines   vec.h - vectors
     sys.h    - system routines       mat.h - matrices
     is.h     - index sets            ksp.h - Krylov subspace methods
     viewer.h - viewers               pc.h  - preconditioners
     sles.h   - linear solvers
*/

#include "snes.h"

/* 
   User-defined application context - contains data needed by the 
   application-provided call-back routines, FormJacobian() and
   FormFunction().
*/
typedef struct {
      double      param;        /* test problem parameter */
      int         mx;           /* Discretization in x-direction */
      int         my;           /* Discretization in y-direction */
} AppCtx;

/* 
   User-defined routines
*/
extern int FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern int FormFunction(SNES,Vec,Vec,void*);
extern int FormInitialGuess(AppCtx*,Vec);

#undef __FUNC__
#define __FUNC__ "main"
int main( int argc, char **argv )
{
  SNES           snes;                 /* nonlinear solver context */
  Vec            x, r;                 /* solution, residual vectors */
  Mat            J;                    /* Jacobian matrix */
  AppCtx         user;                 /* user-defined application context */
  Draw           draw;                 /* drawing context */
  int            i, ierr, its, N, flg, matrix_free, size, fd_coloring, hist_its[50]; 
  double         bratu_lambda_max = 6.81, bratu_lambda_min = 0., history[50];
  MatFDColoring  fdcoloring;           
  Scalar         *array;

  PetscInitialize( &argc, &argv,(char *)0,help );
  MPI_Comm_size(PETSC_COMM_WORLD,&size);
  if (size != 1) SETERRA(1,0,"This is a uniprocessor example only!");

  /*
     Initialize problem parameters
  */
  user.mx = 4; user.my = 4; user.param = 6.0;
  ierr = OptionsGetInt(PETSC_NULL,"-mx",&user.mx,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-my",&user.my,&flg);CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-par",&user.param,&flg);CHKERRA(ierr);
  if (user.param >= bratu_lambda_max || user.param <= bratu_lambda_min) {
    SETERRA(1,0,"Lambda is out of range");
  }
  N = user.mx*user.my;
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SNESCreate(PETSC_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes);CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures; set function evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = VecCreate(PETSC_COMM_WORLD,PETSC_DECIDE,N,&x);CHKERRA(ierr);
  ierr = VecSetFromOptions(x);CHKERRA(ierr);
  ierr = VecDuplicate(x,&r);CHKERRA(ierr);

  /* 
     Set function evaluation routine and vector.  Whenever the nonlinear
     solver needs to evaluate the nonlinear function, it will call this
     routine.
      - Note that the final routine argument is the user-defined
        context that provides application-specific data for the
        function evaluation routine.
  */
  ierr = SNESSetFunction(snes,r,FormFunction,(void *)&user);CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create matrix. Here we only approximately preallocate storage space
     for the Jacobian.  See the users manual for a discussion of better 
     techniques for preallocating matrix memory.
  */
  ierr = OptionsHasName(PETSC_NULL,"-snes_mf",&matrix_free);CHKERRA(ierr);
  if (!matrix_free) {
    ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,N,N,5,PETSC_NULL,&J);CHKERRA(ierr);
  }

  /*
     This option will cause the Jacobian to be computed via finite differences
    efficiently using a coloring of the columns of the matrix.
  */
  ierr = OptionsHasName(PETSC_NULL,"-snes_fd_coloring",&fd_coloring);CHKERRA(ierr);
  if (fd_coloring) {
    ISColoring   iscoloring;
    MatStructure str;

    /* 
      This initializes the nonzero structure of the Jacobian. This is artificial
      because clearly if we had a routine to compute the Jacobian we won't need
      to use finite differences.
    */
    ierr = FormJacobian(snes,x,&J,&J,&str,&user);CHKERRA(ierr);

    /*
       Color the matrix, i.e. determine groups of columns that share no common 
      rows. These columns in the Jacobian can all be computed simulataneously.
    */
    ierr = MatGetColoring(J,MATCOLORING_NATURAL,&iscoloring);CHKERRA(ierr);
    /*
       Create the data structure that SNESDefaultComputeJacobianColor() uses
       to compute the actual Jacobians via finite differences.
    */
    ierr = MatFDColoringCreate(J,iscoloring,&fdcoloring);CHKERRA(ierr);
    ierr = MatFDColoringSetFunction(fdcoloring,(int (*)(void))FormFunction,&user);CHKERRA(ierr);
    ierr = MatFDColoringSetFromOptions(fdcoloring);CHKERRA(ierr);
    /*
        Tell SNES to use the routine SNESDefaultComputeJacobianColor()
      to compute Jacobians.
    */
    ierr = SNESSetJacobian(snes,J,J,SNESDefaultComputeJacobianColor,fdcoloring);CHKERRA(ierr);  
    ierr = ISColoringDestroy(iscoloring);CHKERRA(ierr);
  }
  /* 
     Set Jacobian matrix data structure and default Jacobian evaluation
     routine.  Whenever the nonlinear solver needs to compute the
     Jacobian matrix, it will call this routine.
      - Note that the final routine argument is the user-defined
        context that provides application-specific data for the
        Jacobian evaluation routine.
      - The user can override with:
         -snes_fd : default finite differencing approximation of Jacobian
         -snes_mf : matrix-free Newton-Krylov method with no preconditioning
                    (unless user explicitly sets preconditioner) 
         -snes_mf_operator : form preconditioning matrix as set by the user,
                             but use matrix-free approx for Jacobian-vector
                             products within Newton-Krylov method
  */
  else if (!matrix_free) {
    ierr = SNESSetJacobian(snes,J,J,FormJacobian,(void*)&user);CHKERRA(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Set runtime options (e.g., -snes_monitor -snes_rtol <rtol> -ksp_type <type>)
  */
  ierr = SNESSetFromOptions(snes);CHKERRA(ierr);

  /*
     Set array that saves the function norms.  This array is intended
     when the user wants to save the convergence history for later use
     rather than just to view the function norms via -snes_monitor.
  */
  ierr = SNESSetConvergenceHistory(snes,history,hist_its,50,PETSC_TRUE);CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Note: The user should initialize the vector, x, with the initial guess
     for the nonlinear solver prior to calling SNESSolve().  In particular,
     to employ an initial guess of zero, the user should explicitly set
     this vector to zero by calling VecSet().
  */
  ierr = FormInitialGuess(&user,x);CHKERRA(ierr);
  ierr = SNESSolve(snes,x,&its);CHKERRA(ierr); 
  PetscPrintf(PETSC_COMM_WORLD,"Number of Newton iterations = %d\n", its );

  /*
     Draw contour plot of solution
  */
  ierr = DrawOpenX(PETSC_COMM_WORLD,0,"Solution",300,0,300,300,&draw);CHKERRA(ierr);
  ierr = VecGetArray(x,&array);CHKERRQ(ierr);
  ierr = DrawTensorContour(draw,user.mx,user.my,0,0,array);CHKERRA(ierr);
  ierr = VecRestoreArray(x,&array);CHKERRQ(ierr);

  /* 
     Print the convergence history.  This is intended just to demonstrate
     use of the data attained via SNESSetConvergenceHistory().  
  */
  ierr = OptionsHasName(PETSC_NULL,"-print_history",&flg);CHKERRA(ierr);
  if (flg) for (i=0; i<its; i++)
    PetscPrintf(PETSC_COMM_WORLD,"iteration %d: Linear iterations %d Function norm = %g\n",i,hist_its[i],history[i]);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  if (!matrix_free) {
    ierr = MatDestroy(J);CHKERRA(ierr);
  }
  if (fd_coloring) {
    ierr = MatFDColoringDestroy(fdcoloring);CHKERRA(ierr);
  }
  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(r);CHKERRA(ierr);
  ierr = DrawDestroy(draw);CHKERRA(ierr);
  ierr = SNESDestroy(snes);CHKERRA(ierr);
  PetscFinalize();

  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "FormInitialGuess"
/* 
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
 */
int FormInitialGuess(AppCtx *user,Vec X)
{
  int     i, j, row, mx, my, ierr;
  double  lambda, temp1, temp, hx, hy;
  Scalar  *x;

  mx	 = user->mx; 
  my	 = user->my;
  lambda = user->param;

  hx    = 1.0 / (double)(mx-1);
  hy    = 1.0 / (double)(my-1);

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  temp1 = lambda/(lambda + 1.0);
  for (j=0; j<my; j++) {
    temp = (double)(PetscMin(j,my-j-1))*hy;
    for (i=0; i<mx; i++) {
      row = i + j*mx;  
      if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
        x[row] = 0.0; 
        continue;
      }
      x[row] = temp1*sqrt( PetscMin( (double)(PetscMin(i,mx-i-1))*hx,temp) ); 
    }
  }

  /*
     Restore vector
  */
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "FormFunction"
/* 
   FormFunction - Evaluates nonlinear function, F(x).

   Input Parameters:
.  snes - the SNES context
.  X - input vector
.  ptr - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  F - function vector
 */
int FormFunction(SNES snes,Vec X,Vec F,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;
  int     ierr, i, j, row, mx, my;
  double  two = 2.0, one = 1.0, lambda,hx, hy, hxdhy, hydhx;
  Scalar  ut, ub, ul, ur, u, uxx, uyy, sc,*x,*f;

  mx	 = user->mx; 
  my	 = user->my;
  lambda = user->param;
  hx     = one / (double)(mx-1);
  hy     = one / (double)(my-1);
  sc     = hx*hy;
  hxdhy  = hx/hy;
  hydhx  = hy/hx;

  /*
     Get pointers to vector data
  */
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  /*
     Compute function 
  */
  for (j=0; j<my; j++) {
    for (i=0; i<mx; i++) {
      row = i + j*mx;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
        f[row] = x[row];
        continue;
      }
      u = x[row];
      ub = x[row - mx];
      ul = x[row - 1];
      ut = x[row + mx];
      ur = x[row + 1];
      uxx = (-ur + two*u - ul)*hydhx;
      uyy = (-ut + two*u - ub)*hxdhy;
      f[row] = uxx + uyy - sc*lambda*PetscExpScalar(u);
    }
  }

  /*
     Restore vectors
  */
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
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
.  ptr - optional user-defined context, as set by SNESSetJacobian()

   Output Parameters:
.  A - Jacobian matrix
.  B - optionally different preconditioning matrix
.  flag - flag indicating matrix structure
*/
int FormJacobian(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  AppCtx *user = (AppCtx *) ptr;   /* user-defined applicatin context */
  Mat     jac = *J;                /* Jacobian matrix */
  int     i, j, row, mx, my, col[5], ierr;
  Scalar  two = 2.0, one = 1.0, lambda, v[5],sc, *x;
  double  hx, hy, hxdhy, hydhx;

  mx	 = user->mx; 
  my	 = user->my;
  lambda = user->param;
  hx     = 1.0 / (double)(mx-1);
  hy     = 1.0 / (double)(my-1);
  sc     = hx*hy;
  hxdhy  = hx/hy;
  hydhx  = hy/hx;

  /*
     Get pointer to vector data
  */
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);

  /* 
     Compute entries of the Jacobian
  */
  for (j=0; j<my; j++) {
    for (i=0; i<mx; i++) {
      row = i + j*mx;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1 ) {
        ierr = MatSetValues(jac,1,&row,1,&row,&one,INSERT_VALUES);CHKERRQ(ierr);
        continue;
      }
      v[0] = -hxdhy; col[0] = row - mx;
      v[1] = -hydhx; col[1] = row - 1;
      v[2] = two*(hydhx + hxdhy) - sc*lambda*PetscExpScalar(x[row]); col[2] = row;
      v[3] = -hydhx; col[3] = row + 1;
      v[4] = -hxdhy; col[4] = row + mx;
      ierr = MatSetValues(jac,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  /*
     Restore vector
  */
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);

  /* 
     Assemble matrix
  */
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /*
     Set flag to indicate that the Jacobian matrix retains an identical
     nonzero structure throughout all nonlinear iterations (although the
     values of the entries change). Thus, we can save some work in setting
     up the preconditioner (e.g., no need to redo symbolic factorization for
     ILU/ICC preconditioners).
      - If the nonzero structure of the matrix is different during
        successive linear solves, then the flag DIFFERENT_NONZERO_PATTERN
        must be used instead.  If you are unsure whether the matrix
        structure has changed or not, use the flag DIFFERENT_NONZERO_PATTERN.
      - Caution:  If you specify SAME_NONZERO_PATTERN, PETSc
        believes your assertion and does not check the structure
        of the matrix.  If you erroneously claim that the structure
        is the same when it actually is not, the new preconditioner
        will not function correctly.  Thus, use this optimization
        feature with caution!
  */
  *flag = SAME_NONZERO_PATTERN;
  return 0;
}

