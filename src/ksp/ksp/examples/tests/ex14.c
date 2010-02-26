
/* Program usage:  mpiexec -n <procs> ex14 [-help] [all PETSc options] */

static char help[] = "Solves a nonlinear system in parallel with a user-defined Newton method.\n\
Uses KSP to solve the linearized Newton sytems.  This solver\n\
is a very simplistic inexact Newton method.  The intent of this code is to\n\
demonstrate the repeated solution of linear sytems with the same nonzero pattern.\n\
\n\
This is NOT the recommended approach for solving nonlinear problems with PETSc!\n\
We urge users to employ the SNES component for solving nonlinear problems whenever\n\
possible, as it offers many advantages over coding nonlinear solvers independently.\n\
\n\
We solve the  Bratu (SFI - solid fuel ignition) problem in a 2D rectangular\n\
domain, using distributed arrays (DAs) to partition the parallel grid.\n\
The command line options include:\n\
  -par <parameter>, where <parameter> indicates the problem's nonlinearity\n\
     problem SFI:  <parameter> = Bratu parameter (0 <= par <= 6.81)\n\
  -mx <xg>, where <xg> = number of grid points in the x-direction\n\
  -my <yg>, where <yg> = number of grid points in the y-direction\n\
  -Nx <npx>, where <npx> = number of processors in the x-direction\n\
  -Ny <npy>, where <npy> = number of processors in the y-direction\n\n";

/*T
   Concepts: KSP^writing a user-defined nonlinear solver (parallel Bratu example);
   Concepts: DA^using distributed arrays;
   Processors: n
T*/

/* ------------------------------------------------------------------------

    Solid Fuel Ignition (SFI) problem.  This problem is modeled by
    the partial differential equation
  
            -Laplacian u - lambda*exp(u) = 0,  0 < x,y < 1,
  
    with boundary conditions
   
             u = 0  for  x = 0, x = 1, y = 0, y = 1.
  
    A finite difference approximation with the usual 5-point stencil
    is used to discretize the boundary value problem to obtain a nonlinear 
    system of equations.

    The SNES version of this problem is:  snes/examples/tutorials/ex5.c
    We urge users to employ the SNES component for solving nonlinear
    problems whenever possible, as it offers many advantages over coding 
    nonlinear solvers independently.

  ------------------------------------------------------------------------- */

/* 
   Include "petscda.h" so that we can use distributed arrays (DAs).
   Include "petscksp.h" so that we can use KSP solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
*/
#include "petscda.h"
#include "petscksp.h"

/* 
   User-defined application context - contains data needed by the 
   application-provided call-back routines, ComputeJacobian() and
   ComputeFunction().
*/
typedef struct {
   PetscReal   param;          /* test problem parameter */
   PetscInt    mx,my;          /* discretization in x,y directions */
   Vec         localX,localF; /* ghosted local vector */
   DA          da;             /* distributed array data structure */
   PetscInt    rank;           /* processor rank */
} AppCtx;

/* 
   User-defined routines
*/
extern PetscErrorCode ComputeFunction(AppCtx*,Vec,Vec),FormInitialGuess(AppCtx*,Vec);
extern PetscErrorCode ComputeJacobian(AppCtx*,Vec,Mat,MatStructure*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  /* -------------- Data to define application problem ---------------- */
  MPI_Comm       comm;                /* communicator */
  KSP            ksp;                /* linear solver */
  Vec            X,Y,F;             /* solution, update, residual vectors */
  Mat            J;                   /* Jacobian matrix */
  AppCtx         user;                /* user-defined work context */
  PetscInt       Nx,Ny;              /* number of preocessors in x- and y- directions */
  PetscMPIInt    size;                /* number of processors */
  PetscReal      bratu_lambda_max = 6.81,bratu_lambda_min = 0.;
  PetscInt       m,N;
  PetscErrorCode ierr;

  /* --------------- Data to define nonlinear solver -------------- */
  PetscReal      rtol = 1.e-8;        /* relative convergence tolerance */
  PetscReal      xtol = 1.e-8;        /* step convergence tolerance */
  PetscReal      ttol;                /* convergence tolerance */
  PetscReal      fnorm,ynorm,xnorm; /* various vector norms */
  PetscInt       max_nonlin_its = 10; /* maximum number of iterations for nonlinear solver */
  PetscInt       max_functions = 50;  /* maximum number of function evaluations */
  PetscInt       lin_its;             /* number of linear solver iterations for each step */
  PetscInt       i;                   /* nonlinear solve iteration number */
  MatStructure   mat_flag;        /* flag indicating structure of preconditioner matrix */
  PetscTruth     no_output = PETSC_FALSE;           /* flag indicating whether to surpress output */

  PetscInitialize(&argc,&argv,(char *)0,help);
  comm = PETSC_COMM_WORLD;
  ierr = MPI_Comm_rank(comm,&user.rank);CHKERRQ(ierr);
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-no_output",&no_output,PETSC_NULL);CHKERRQ(ierr);

  /*
     Initialize problem parameters
  */
  user.mx = 4; user.my = 4; user.param = 6.0;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mx",&user.mx,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-my",&user.my,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-par",&user.param,PETSC_NULL);CHKERRQ(ierr);
  if (user.param >= bratu_lambda_max || user.param <= bratu_lambda_min) {
    SETERRQ(1,"Lambda is out of range");
  }
  N = user.mx*user.my;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create linear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = KSPCreate(comm,&ksp);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create distributed array (DA) to manage parallel grid and vectors
  */
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  Nx = PETSC_DECIDE; Ny = PETSC_DECIDE;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-Nx",&Nx,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-Ny",&Ny,PETSC_NULL);CHKERRQ(ierr);
  if (Nx*Ny != size && (Nx != PETSC_DECIDE || Ny != PETSC_DECIDE))
    SETERRQ(1,"Incompatible number of processors:  Nx * Ny != size");
  ierr = DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_STAR,user.mx,
                    user.my,Nx,Ny,1,1,PETSC_NULL,PETSC_NULL,&user.da);CHKERRQ(ierr);

  /*
     Extract global and local vectors from DA; then duplicate for remaining
     vectors that are the same types
  */
  ierr = DACreateGlobalVector(user.da,&X);CHKERRQ(ierr);
  ierr = DACreateLocalVector(user.da,&user.localX);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&F);CHKERRQ(ierr);
  ierr = VecDuplicate(X,&Y);CHKERRQ(ierr);
  ierr = VecDuplicate(user.localX,&user.localF);CHKERRQ(ierr);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure for Jacobian
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Note:  For the parallel case, vectors and matrices MUST be partitioned
     accordingly.  When using distributed arrays (DAs) to create vectors,
     the DAs determine the problem partitioning.  We must explicitly
     specify the local matrix dimensions upon its creation for compatibility
     with the vector distribution.  Thus, the generic MatCreate() routine
     is NOT sufficient when working with distributed arrays.

     Note: Here we only approximately preallocate storage space for the
     Jacobian.  See the users manual for a discussion of better techniques
     for preallocating matrix memory.
  */
  if (size == 1) {
    ierr = MatCreateSeqAIJ(comm,N,N,5,PETSC_NULL,&J);CHKERRQ(ierr);
  } else {
    ierr = VecGetLocalSize(X,&m);CHKERRQ(ierr);
    ierr = MatCreateMPIAIJ(comm,m,m,N,N,5,PETSC_NULL,3,PETSC_NULL,&J);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize linear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Set runtime options (e.g.,-ksp_monitor -ksp_rtol <rtol> -ksp_type <type>)
  */
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = FormInitialGuess(&user,X);CHKERRQ(ierr);
  ierr = ComputeFunction(&user,X,F);CHKERRQ(ierr);   /* Compute F(X)    */
  ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);     /* fnorm = || F || */
  ttol = fnorm*rtol;
  if (!no_output) PetscPrintf(comm,"Initial function norm = %G\n",fnorm);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system with a user-defined method
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
      This solver is a very simplistic inexact Newton method, with no
      no damping strategies or bells and whistles. The intent of this code
      is  merely to demonstrate the repeated solution with KSP of linear
      sytems with the same nonzero structure.

      This is NOT the recommended approach for solving nonlinear problems
      with PETSc!  We urge users to employ the SNES component for solving
      nonlinear problems whenever possible with application codes, as it
      offers many advantages over coding nonlinear solvers independently.
   */

  for (i=0; i<max_nonlin_its; i++) {

    /* 
        Compute the Jacobian matrix.  See the comments in this routine for
        important information about setting the flag mat_flag.
     */
    ierr = ComputeJacobian(&user,X,J,&mat_flag);CHKERRQ(ierr);

    /* 
        Solve J Y = F, where J is the Jacobian matrix.
          - First, set the KSP linear operators.  Here the matrix that
            defines the linear system also serves as the preconditioning
            matrix.
          - Then solve the Newton system.
     */
    ierr = KSPSetOperators(ksp,J,J,mat_flag);CHKERRQ(ierr);
    ierr = KSPSolve(ksp,F,Y);CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp,&lin_its);CHKERRQ(ierr);

    /* 
       Compute updated iterate
     */
    ierr = VecNorm(Y,NORM_2,&ynorm);CHKERRQ(ierr);       /* ynorm = || Y || */
    ierr = VecAYPX(Y,-1.0,X);CHKERRQ(ierr);              /* Y <- X - Y      */
    ierr = VecCopy(Y,X);CHKERRQ(ierr);                   /* X <- Y          */
    ierr = VecNorm(X,NORM_2,&xnorm);CHKERRQ(ierr);       /* xnorm = || X || */
    if (!no_output) {
      ierr = PetscPrintf(comm,"   linear solve iterations = %D, xnorm=%G, ynorm=%G\n",lin_its,xnorm,ynorm);CHKERRQ(ierr);
    }

    /* 
       Evaluate new nonlinear function
     */
    ierr = ComputeFunction(&user,X,F);CHKERRQ(ierr);     /* Compute F(X)    */
    ierr = VecNorm(F,NORM_2,&fnorm);CHKERRQ(ierr);       /* fnorm = || F || */
    if (!no_output) {
      ierr = PetscPrintf(comm,"Iteration %D, function norm = %G\n",i+1,fnorm);CHKERRQ(ierr);
    }

    /*
       Test for convergence
     */
    if (fnorm <= ttol) {
      if (!no_output) {
         ierr = PetscPrintf(comm,"Converged due to function norm %G < %G (relative tolerance)\n",fnorm,ttol);CHKERRQ(ierr);
      }
      break;
    }
    if (ynorm < xtol*(xnorm)) {
      if (!no_output) {
         ierr = PetscPrintf(comm,"Converged due to small update length: %G < %G * %G\n",ynorm,xtol,xnorm);CHKERRQ(ierr);
      }
      break;
    }
    if (i > max_functions) {
      if (!no_output) {
        ierr = PetscPrintf(comm,"Exceeded maximum number of function evaluations: %D > %D\n",i,max_functions);CHKERRQ(ierr);
      }
      break;
    }  
  }
  ierr = PetscPrintf(comm,"Number of Newton iterations = %D\n",i+1);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatDestroy(J);CHKERRQ(ierr);           ierr = VecDestroy(Y);CHKERRQ(ierr);
  ierr = VecDestroy(user.localX);CHKERRQ(ierr); ierr = VecDestroy(X);CHKERRQ(ierr);
  ierr = VecDestroy(user.localF);CHKERRQ(ierr); ierr = VecDestroy(F);CHKERRQ(ierr);      
  ierr = KSPDestroy(ksp);CHKERRQ(ierr);  ierr = DADestroy(user.da);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);

  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess"
/* 
   FormInitialGuess - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
 */
PetscErrorCode FormInitialGuess(AppCtx *user,Vec X)
{
  PetscInt     i,j,row,mx,my,ierr,xs,ys,xm,ym,gxm,gym,gxs,gys;
  PetscReal    one = 1.0,lambda,temp1,temp,hx,hy;
  PetscScalar  *x;
  Vec          localX = user->localX;

  mx = user->mx;            my = user->my;            lambda = user->param;
  hx = one/(PetscReal)(mx-1);  hy = one/(PetscReal)(my-1);
  temp1 = lambda/(lambda + one);

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);

  /*
     Get local grid boundaries (for 2-dimensional DA):
       xs, ys   - starting grid indices (no ghost points)
       xm, ym   - widths of local grid (no ghost points)
       gxs, gys - starting grid indices (including ghost points)
       gxm, gym - widths of local grid (including ghost points)
  */
  ierr = DAGetCorners(user->da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(user->da,&gxs,&gys,PETSC_NULL,&gxm,&gym,PETSC_NULL);CHKERRQ(ierr);

  /*
     Compute initial guess over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) {
    temp = (PetscReal)(PetscMin(j,my-j-1))*hy;
    for (i=xs; i<xs+xm; i++) {
      row = i - gxs + (j - gys)*gxm; 
      if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
        x[row] = 0.0; 
        continue;
      }
      x[row] = temp1*sqrt(PetscMin((PetscReal)(PetscMin(i,mx-i-1))*hx,temp)); 
    }
  }

  /*
     Restore vector
  */
  ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);

  /*
     Insert values into global vector
  */
  ierr = DALocalToGlobal(user->da,localX,INSERT_VALUES,X);CHKERRQ(ierr);
  return 0;
} 
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "ComputeFunction"
/* 
   ComputeFunction - Evaluates nonlinear function, F(x).

   Input Parameters:
.  X - input vector
.  user - user-defined application context

   Output Parameter:
.  F - function vector
 */
PetscErrorCode ComputeFunction(AppCtx *user,Vec X,Vec F)
{
  PetscErrorCode ierr;
  PetscInt       i,j,row,mx,my,xs,ys,xm,ym,gxs,gys,gxm,gym;
  PetscReal      two = 2.0,one = 1.0,lambda,hx,hy,hxdhy,hydhx,sc;
  PetscScalar    u,uxx,uyy,*x,*f;
  Vec            localX = user->localX,localF = user->localF; 

  mx = user->mx;            my = user->my;            lambda = user->param;
  hx = one/(PetscReal)(mx-1);  hy = one/(PetscReal)(my-1);
  sc = hx*hy*lambda;        hxdhy = hx/hy;            hydhx = hy/hx;

  /*
     Scatter ghost points to local vector, using the 2-step process
        DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DAGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /*
     Get pointers to vector data
  */
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DAGetCorners(user->da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(user->da,&gxs,&gys,PETSC_NULL,&gxm,&gym,PETSC_NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  for (j=ys; j<ys+ym; j++) {
    row = (j - gys)*gxm + xs - gxs - 1; 
    for (i=xs; i<xs+xm; i++) {
      row++;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
        f[row] = x[row];
        continue;
      }
      u = x[row];
      uxx = (two*u - x[row-1] - x[row+1])*hydhx;
      uyy = (two*u - x[row-gxm] - x[row+gxm])*hxdhy;
      f[row] = uxx + uyy - sc*PetscExpScalar(u);
    }
  }

  /*
     Restore vectors
  */
  ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);

  /*
     Insert values into global vector
  */
  ierr = DALocalToGlobal(user->da,localF,INSERT_VALUES,F);CHKERRQ(ierr);
  ierr = PetscLogFlops(11.0*ym*xm);CHKERRQ(ierr);
  return 0; 
} 
/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "ComputeJacobian"
/*
   ComputeJacobian - Evaluates Jacobian matrix.

   Input Parameters:
.  x - input vector
.  user - user-defined application context

   Output Parameters:
.  jac - Jacobian matrix
.  flag - flag indicating matrix structure

   Notes:
   Due to grid point reordering with DAs, we must always work
   with the local grid points, and then transform them to the new
   global numbering with the "ltog" mapping (via DAGetGlobalIndices()).
   We cannot work directly with the global numbers for the original
   uniprocessor grid!
*/
PetscErrorCode ComputeJacobian(AppCtx *user,Vec X,Mat jac,MatStructure *flag)
{
  PetscErrorCode ierr;
  Vec            localX = user->localX;   /* local vector */
  PetscInt       *ltog;                   /* local-to-global mapping */
  PetscInt       i,j,row,mx,my,col[5];
  PetscInt       nloc,xs,ys,xm,ym,gxs,gys,gxm,gym,grow;
  PetscScalar    two = 2.0,one = 1.0,lambda,v[5],hx,hy,hxdhy,hydhx,sc,*x;

  mx = user->mx;            my = user->my;            lambda = user->param;
  hx = one/(PetscReal)(mx-1);  hy = one/(PetscReal)(my-1);
  sc = hx*hy;               hxdhy = hx/hy;            hydhx = hy/hx;

  /*
     Scatter ghost points to local vector, using the 2-step process
        DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DAGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /*
     Get pointer to vector data
  */
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DAGetCorners(user->da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(user->da,&gxs,&gys,PETSC_NULL,&gxm,&gym,PETSC_NULL);CHKERRQ(ierr);

  /*
     Get the global node numbers for all local nodes, including ghost points
  */
  ierr = DAGetGlobalIndices(user->da,&nloc,&ltog);CHKERRQ(ierr);

  /* 
     Compute entries for the locally owned part of the Jacobian.
      - Currently, all PETSc parallel matrix formats are partitioned by
        contiguous chunks of rows across the processors. The "grow"
        parameter computed below specifies the global row number 
        corresponding to each local grid point.
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly). 
      - Always specify global row and columns of matrix entries.
      - Here, we set all entries for a particular row at once.
  */
  for (j=ys; j<ys+ym; j++) {
    row = (j - gys)*gxm + xs - gxs - 1; 
    for (i=xs; i<xs+xm; i++) {
      row++;
      grow = ltog[row];
      /* boundary points */
      if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
        ierr = MatSetValues(jac,1,&grow,1,&grow,&one,INSERT_VALUES);CHKERRQ(ierr);
        continue;
      }
      /* interior grid points */
      v[0] = -hxdhy; col[0] = ltog[row - gxm];
      v[1] = -hydhx; col[1] = ltog[row - 1];
      v[2] = two*(hydhx + hxdhy) - sc*lambda*PetscExpScalar(x[row]); col[2] = grow;
      v[3] = -hydhx; col[3] = ltog[row + 1];
      v[4] = -hxdhy; col[4] = ltog[row + gxm];
      ierr = MatSetValues(jac,1,&grow,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }

  /* 
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);
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
