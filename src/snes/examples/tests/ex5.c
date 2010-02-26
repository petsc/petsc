
static char help[] = "Solves a nonlinear system in parallel with SNES.\n\
We solve the modified Bratu problem in a 2D rectangular domain,\n\
using distributed arrays (DAs) to partition the parallel grid.\n\
The command line options include:\n\
  -lambda <parameter>, where <parameter> indicates the problem's nonlinearity\n\
  -kappa  <parameter>, where <parameter> indicates the problem's nonlinearity\n\
  -mx <xg>, where <xg> = number of grid points in the x-direction\n\
  -my <yg>, where <yg> = number of grid points in the y-direction\n\
  -Nx <npx>, where <npx> = number of processors in the x-direction\n\
  -Ny <npy>, where <npy> = number of processors in the y-direction\n\n";

/*T
   Concepts: SNES^solving a system of nonlinear equations (parallel Bratu example);
   Concepts: DA^using distributed arrays;
   Processors: n
T*/

/* ------------------------------------------------------------------------

    Modified Solid Fuel Ignition problem.  This problem is modeled by
    the partial differential equation

        -Laplacian u - kappa*\PartDer{u}{x} - lambda*exp(u) = 0,

    where

         0 < x,y < 1,
  
    with boundary conditions
   
             u = 0  for  x = 0, x = 1, y = 0, y = 1.
  
    A finite difference approximation with the usual 5-point stencil
    is used to discretize the boundary value problem to obtain a nonlinear 
    system of equations.

  ------------------------------------------------------------------------- */

/* 
   Include "petscda.h" so that we can use distributed arrays (DAs).
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/
#include "petscda.h"
#include "petscsnes.h"

/* 
   User-defined application context - contains data needed by the 
   application-provided call-back routines, FormJacobian() and
   FormFunction().
*/
typedef struct {
   PetscReal   param;          /* test problem parameter */
   PetscReal   param2;         /* test problem parameter */
   PetscInt    mx,my;          /* discretization in x, y directions */
   Vec         localX,localF; /* ghosted local vector */
   DA          da;             /* distributed array data structure */
   PetscMPIInt rank;           /* processor rank */
} AppCtx;

/* 
   User-defined routines
*/
extern PetscErrorCode FormFunction(SNES,Vec,Vec,void*),FormInitialGuess(AppCtx*,Vec);
extern PetscErrorCode FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  SNES           snes;                /* nonlinear solver */
  Vec            x,r;                /* solution, residual vectors */
  Mat            J;                   /* Jacobian matrix */
  AppCtx         user;                /* user-defined work context */
  PetscInt       its;                 /* iterations for convergence */
  PetscInt       Nx,Ny;              /* number of preocessors in x- and y- directions */
  PetscTruth     matrix_free = PETSC_FALSE;         /* flag - 1 indicates matrix-free version */
  PetscMPIInt    size;                /* number of processors */
  PetscInt       m,N;
  PetscErrorCode ierr;
  PetscReal      bratu_lambda_max = 6.81,bratu_lambda_min = 0.;
  PetscReal      bratu_kappa_max = 10000,bratu_kappa_min = 0.;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&user.rank);CHKERRQ(ierr);

  /*
     Initialize problem parameters
  */
  user.mx = 4; user.my = 4; user.param = 6.0; user.param2 = 0.0;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mx",&user.mx,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-my",&user.my,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-lambda",&user.param,PETSC_NULL);CHKERRQ(ierr);
  if (user.param >= bratu_lambda_max || user.param <= bratu_lambda_min) {
    SETERRQ(1,"Lambda is out of range");
  }
  ierr = PetscOptionsGetReal(PETSC_NULL,"-kappa",&user.param2,PETSC_NULL);CHKERRQ(ierr);
  if (user.param2 >= bratu_kappa_max || user.param2 < bratu_kappa_min) {
    SETERRQ(1,"Kappa is out of range");
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Solving the Bratu problem with lambda=%G, kappa=%G\n",user.param,user.param2);CHKERRQ(ierr);

  N = user.mx*user.my;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures; set function evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create distributed array (DA) to manage parallel grid and vectors
  */
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  Nx = PETSC_DECIDE; Ny = PETSC_DECIDE;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-Nx",&Nx,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-Ny",&Ny,PETSC_NULL);CHKERRQ(ierr);
  if (Nx*Ny != size && (Nx != PETSC_DECIDE || Ny != PETSC_DECIDE))
    SETERRQ(1,"Incompatible number of processors:  Nx * Ny != size");
  ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,user.mx,user.my,Nx,Ny,1,1,PETSC_NULL,PETSC_NULL,&user.da);CHKERRQ(ierr);

  /*
     Visualize the distribution of the array across the processors
  */
  /* ierr =  DAView(user.da,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr); */


  /*
     Extract global and local vectors from DA; then duplicate for remaining
     vectors that are the same types
  */
  ierr = DACreateGlobalVector(user.da,&x);CHKERRQ(ierr);
  ierr = DACreateLocalVector(user.da,&user.localX);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);
  ierr = VecDuplicate(user.localX,&user.localF);CHKERRQ(ierr);

  /* 
     Set function evaluation routine and vector
  */
  ierr = SNESSetFunction(snes,r,FormFunction,(void*)&user);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Set Jacobian matrix data structure and default Jacobian evaluation
     routine. User can override with:
     -snes_fd : default finite differencing approximation of Jacobian
     -snes_mf : matrix-free Newton-Krylov method with no preconditioning
                (unless user explicitly sets preconditioner) 
     -snes_mf_operator : form preconditioning matrix as set by the user,
                         but use matrix-free approx for Jacobian-vector
                         products within Newton-Krylov method

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
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-snes_mf",&matrix_free,PETSC_NULL);CHKERRQ(ierr);
  if (!matrix_free) {
    PetscTruth matrix_free_operator = PETSC_FALSE;
    ierr = PetscOptionsGetTruth(PETSC_NULL,"-snes_mf_operator",&matrix_free_operator,PETSC_NULL);CHKERRQ(ierr);
    if (matrix_free_operator) matrix_free = PETSC_FALSE;
  }
  if (!matrix_free) {
    if (size == 1) {
      ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,N,N,5,PETSC_NULL,&J);CHKERRQ(ierr);
    } else {
      ierr = VecGetLocalSize(x,&m);CHKERRQ(ierr);
      ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD,m,m,N,N,5,PETSC_NULL,3,PETSC_NULL,&J);CHKERRQ(ierr);
    }
    ierr = SNESSetJacobian(snes,J,J,FormJacobian,&user);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Set runtime options (e.g., -snes_monitor -snes_rtol <rtol> -ksp_type <type>)
  */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Evaluate initial guess; then solve nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Note: The user should initialize the vector, x, with the initial guess
     for the nonlinear solver prior to calling SNESSolve().  In particular,
     to employ an initial guess of zero, the user should explicitly set
     this vector to zero by calling VecSet().
  */
  ierr = FormInitialGuess(&user,x);CHKERRQ(ierr);
  ierr = SNESSolve(snes,PETSC_NULL,x);CHKERRQ(ierr); 
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of Newton iterations = %D\n",its);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  if (!matrix_free) {
    ierr = MatDestroy(J);CHKERRQ(ierr);
  }
  ierr = VecDestroy(user.localX);CHKERRQ(ierr); ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(user.localF);CHKERRQ(ierr); ierr = VecDestroy(r);CHKERRQ(ierr);      
  ierr = SNESDestroy(snes);CHKERRQ(ierr);  ierr = DADestroy(user.da);CHKERRQ(ierr);
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
  PetscInt       i,j,row,mx,my,xs,ys,xm,ym,gxm,gym,gxs,gys;
  PetscErrorCode ierr;
  PetscReal      one = 1.0,lambda,temp1,temp,hx,hy,hxdhy,hydhx,sc;
  PetscScalar    *x;
  Vec            localX = user->localX;

  mx = user->mx;            my = user->my;            lambda = user->param;
  hx = one/(PetscReal)(mx-1);  hy = one/(PetscReal)(my-1);
  sc = hx*hy*lambda;        hxdhy = hx/hy;            hydhx = hy/hx;
  temp1 = lambda/(lambda + one);

  /*
     Get a pointer to vector data.
       - For default PETSc vectors,VecGetArray() returns a pointer to
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
#define __FUNCT__ "FormFunction"
/* 
   FormFunction - Evaluates nonlinear function, F(x).

   Input Parameters:
.  snes - the SNES context
.  X - input vector
.  ptr - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  F - function vector
 */
PetscErrorCode FormFunction(SNES snes,Vec X,Vec F,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  PetscErrorCode ierr;
  PetscInt       i,j,row,mx,my,xs,ys,xm,ym,gxs,gys,gxm,gym;
  PetscReal      two = 2.0,one = 1.0,half = 0.5;
  PetscReal      lambda,hx,hy,hxdhy,hydhx,sc;
  PetscScalar    u,ux,uxx,uyy,*x,*f,kappa;
  Vec            localX = user->localX,localF = user->localF; 

  mx = user->mx;            my = user->my;            lambda = user->param;
  hx = one/(PetscReal)(mx-1);  hy = one/(PetscReal)(my-1);
  sc = hx*hy*lambda;        hxdhy = hx/hy;            hydhx = hy/hx;
  kappa = user->param2;

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
      ux  = (x[row+1] - x[row-1])*half*hy;
      uxx = (two*u - x[row-1] - x[row+1])*hydhx;
      uyy = (two*u - x[row-gxm] - x[row+gxm])*hxdhy;
      f[row] = uxx + uyy - kappa*ux - sc*exp(u);
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
#define __FUNCT__ "FormJacobian"
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

   Notes:
   Due to grid point reordering with DAs, we must always work
   with the local grid points, and then transform them to the new
   global numbering with the "ltog" mapping (via DAGetGlobalIndices()).
   We cannot work directly with the global numbers for the original
   uniprocessor grid!
*/
PetscErrorCode FormJacobian(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;  /* user-defined application context */
  Mat            jac = *B;                /* Jacobian matrix */
  Vec            localX = user->localX;   /* local vector */
  PetscErrorCode ierr;
  PetscInt       *ltog;                   /* local-to-global mapping */
  PetscInt       i,j,row,mx,my,col[5];
  PetscInt       nloc,xs,ys,xm,ym,gxs,gys,gxm,gym,grow;
  PetscScalar    two = 2.0,one = 1.0,lambda,v[5],hx,hy,hxdhy,hydhx,sc,*x;

  mx = user->mx;            my = user->my;            lambda = user->param;
  hx = one/(PetscReal)(mx-1);  hy = one/(PetscReal)(my-1);
  sc = hx*hy;               hxdhy = hx/hy;            hydhx = hy/hx;

  /*
     Scatter ghost points to local vector,using the 2-step process
        DAGlobalToLocalBegin(),DAGlobalToLocalEnd().
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
      v[2] = two*(hydhx + hxdhy) - sc*lambda*exp(x[row]); col[2] = grow;
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
  /*
      Tell the matrix we will never add a new nonzero location to the
    matrix. If we do it will generate an error.
  */
  /* ierr = MatSetOption(jac,MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr); */
  return 0;
}

