/*$Id: ex14.c,v 1.5 1999/09/12 17:19:16 bsmith Exp bsmith $*/

/* Program usage:  mpirun -np <procs> ex14 [-help] [all PETSc options] */

static char help[] = "Solves a nonlinear system in parallel with SNES.\n\
We solve the  Bratu (SFI - solid fuel ignition) problem in a 3D rectangular\n\
domain, using distributed arrays (DAs) to partition the parallel grid.\n\
The command line options include:\n\
  -par <parameter>, where <parameter> indicates the problem's nonlinearity\n\
     problem SFI:  <parameter> = Bratu parameter (0 <= par <= 6.81)\n\
  -mx <xg>, where <xg> = number of grid points in the x-direction\n\
  -my <yg>, where <yg> = number of grid points in the y-direction\n\
  -mz <zg>, where <zg> = number of grid points in the z-direction\n\
  -Nx <npx>, where <npx> = number of processors in the x-direction\n\
  -Ny <npy>, where <npy> = number of processors in the y-direction\n\
  -Nz <npy>, where <npz> = number of processors in the z-direction\n\
  -debug : Activate debugging printouts\n\n";

/*
     This example is the 3-dimensional analog of ex5.c.
*/

/*T
   Concepts: SNES^Solving a system of nonlinear equations (parallel Bratu example);
   Concepts: DA^Using distributed arrays;
   Routines: SNESCreate(); SNESSetFunction(); SNESSetJacobian();
   Routines: SNESSolve(); SNESSetFromOptions(); DAView();
   Routines: DACreate3d(); DADestroy(); DACreateGlobalVector(); DACreateLocalVector();
   Routines: DAGetCorners(); DAGetGhostCorners(); DALocalToGlobal();
   Routines: DAGlobalToLocalBegin(); DAGlobalToLocalEnd(); DAGetGlobalIndices();
   Processors: n
T*/

/* ------------------------------------------------------------------------

    Solid Fuel Ignition (SFI) problem.  This problem is modeled by
    the partial differential equation

            -Laplacian u - lambda*exp(u) = 0,  0 < x,y,z < 1 ,
  
    with boundary conditions
   
             u = 0  for  x = 0, x = 1, y = 0, y = 1, z = 0, z = 1.
  
    A finite difference approximation with the usual 5-point stencil
    is used to discretize the boundary value problem to obtain a nonlinear 
    system of equations.

    The uniprocessor version of this code is snes/examples/tutorials/ex4_3d.c

  ------------------------------------------------------------------------- */

/* 
   Include "da.h" so that we can use distributed arrays (DAs).
   Include "snes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petsc.h  - base PETSc routines   vec.h - vectors
     sys.h    - system routines       mat.h - matrices
     is.h     - index sets            ksp.h - Krylov subspace methods
     viewer.h - viewers               pc.h  - preconditioners
     sles.h   - linear solvers
*/
#include "da.h"
#include "snes.h"

/* 
   User-defined application context - contains data needed by the 
   application-provided call-back routines, FormJacobian() and
   FormFunction().
*/
typedef struct {
   double      param;          /* test problem parameter */
   int         mx, my, mz;     /* discretization in x, y, z directions */
   Vec         localX, localF; /* ghosted local vector */
   DA          da;             /* distributed array data structure */
   int         rank;           /* processor rank */
   int         size;           /* number of processors */
   int         debug;          /* debugging flag: 1 means debugging is activated */
} AppCtx;

/* 
   User-defined routines
*/
int FormFunction(SNES,Vec,Vec,void*), FormInitialGuess(AppCtx*,Vec);
int FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
int ApplicationFunction(AppCtx*,Scalar*,Scalar*);
int ApplicationInitialGuess(AppCtx*,Scalar*);

int main( int argc, char **argv )
{
  SNES     snes;                /* nonlinear solver */
  Vec      x, r;                /* solution, residual vectors */
  Mat      J;                   /* Jacobian matrix */
  AppCtx   user;                /* user-defined work context */
  int      its;                 /* iterations for convergence */
  int      Nx, Ny, Nz;          /* number of preocessors in x-, y- and z- directions */
  int      matrix_free;         /* flag - 1 indicates matrix-free version */
  int      m, flg, N, ierr, nloc, *ltog;
  double   bratu_lambda_max = 6.81, bratu_lambda_min = 0.;

  PetscInitialize( &argc, &argv,(char *)0,help );
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&user.rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&user.size);CHKERRA(ierr);

  /*
     Initialize problem parameters
  */
  user.mx = 4; user.my = 4; user.mz = 4; user.param = 6.0;
  ierr = OptionsGetInt(PETSC_NULL,"-mx",&user.mx,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-my",&user.my,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-mz",&user.mz,&flg); CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-par",&user.param,&flg); CHKERRA(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-debug",&user.debug); CHKERRA(ierr);
  if (user.param >= bratu_lambda_max || user.param <= bratu_lambda_min) {
    SETERRA(1,0,"Lambda is out of range");
  }
  N = user.mx*user.my*user.mz;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"mx=%d, my=%d, mz=%d, N=%d, lambda=%g\n",
              user.mx,user.my,user.mz,N,user.param);CHKERRA(ierr);
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SNESCreate(PETSC_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures; set function evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create distributed array (DA) to manage parallel grid and vectors
  */

  Nx = PETSC_DECIDE; Ny = PETSC_DECIDE; Nz = PETSC_DECIDE;
  ierr = OptionsGetInt(PETSC_NULL,"-Nx",&Nx,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-Ny",&Ny,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-Nz",&Nz,&flg); CHKERRA(ierr);
  if (Nx*Ny*Nz != user.size && (Nx != PETSC_DECIDE || Ny != PETSC_DECIDE
                        || Nz != PETSC_DECIDE))
    SETERRA(1,0,"Incompatible number of processors:  Nx*Ny*Nz != user.size");
  ierr = DACreate3d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,user.mx,
                    user.my,user.mz,Nx,Ny,Nz,1,1,PETSC_NULL,PETSC_NULL,
                    PETSC_NULL,&user.da); CHKERRA(ierr);

  /*
     Visualize the distribution of the array across the processors
  */
  /* ierr =  DAView(user.da,VIEWER_DRAWX_WORLD); CHKERRA(ierr); */

  /*
     Extract global and local vectors from DA; then duplicate for remaining
     vectors that are the same types
  */
  ierr = DACreateGlobalVector(user.da,&x); CHKERRA(ierr);
  ierr = DACreateLocalVector(user.da,&user.localX); CHKERRA(ierr);
  ierr = VecDuplicate(x,&r); CHKERRA(ierr);
  ierr = VecDuplicate(user.localX,&user.localF); CHKERRA(ierr);

  /* 
     Set function evaluation routine and vector
  */
  ierr = SNESSetFunction(snes,r,FormFunction,(void*)&user); CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Set Jacobian matrix data structure and default Jacobian evaluation
     routine. 
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
  ierr = OptionsHasName(PETSC_NULL,"-snes_mf",&matrix_free); CHKERRA(ierr);
  if (!matrix_free) {
    if (user.size == 1) {
      ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,N,N,7,PETSC_NULL,&J); CHKERRA(ierr);
    } else {
      ierr = VecGetLocalSize(x,&m); CHKERRA(ierr);
      ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD,m,m,N,N,7,PETSC_NULL,3,PETSC_NULL,&J); CHKERRA(ierr);
    }
    ierr = SNESSetJacobian(snes,J,J,FormJacobian,&user); CHKERRA(ierr);

    /*
       Get the global node numbers for all local nodes, including ghost points.
       Associate this mapping with the matrix for later use in setting matrix
       entries via MatSetValuesLocal().
    */
    ierr = DAGetGlobalIndices(user.da,&nloc,&ltog); CHKERRA(ierr);
    {
      ISLocalToGlobalMapping isltog;
      ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,nloc,ltog,&isltog); CHKERRA(ierr);
      ierr = MatSetLocalToGlobalMapping(J,isltog); CHKERRA(ierr);
      ierr = ISLocalToGlobalMappingDestroy(isltog); CHKERRA(ierr);
    }
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Set runtime options (e.g., -snes_monitor -snes_rtol <rtol> -ksp_type <type>)
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
  ierr = FormInitialGuess(&user,x); CHKERRA(ierr);
  ierr = SNESSolve(snes,x,&its); CHKERRA(ierr); 
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of Newton iterations = %d\n", its );CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  if (!matrix_free) {
    ierr = MatDestroy(J); CHKERRA(ierr);
  }
  ierr = VecDestroy(user.localX); CHKERRA(ierr); ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(user.localF); CHKERRA(ierr); ierr = VecDestroy(r); CHKERRA(ierr);      
  ierr = SNESDestroy(snes); CHKERRA(ierr);  ierr = DADestroy(user.da); CHKERRA(ierr);
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

   Notes:
   This routine serves as a wrapper for the lower-level routine
   "ApplicationInitialGuess", where the actual computations are 
   done locally by treating the local vector data as an
   array over the local mesh.  This routine merely handles ghost
   point scatters and accesses the local vector data via
   VecGetArray() and VecRestoreArray().
 */
int FormInitialGuess(AppCtx *user,Vec X)
{
  Scalar *x;
  Vec    localX = user->localX;
  int    ierr;

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr = VecGetArray(localX,&x); CHKERRQ(ierr);

  /*
     Compute initial guess over the locally owned part of the grid
  */
  ierr =  ApplicationInitialGuess(user,x); CHKERRQ(ierr);

  /*
     Restore vector
  */
  ierr = VecRestoreArray(localX,&x); CHKERRQ(ierr);

  /*
     Insert values into global vector
  */
  ierr = DALocalToGlobal(user->da,localX,INSERT_VALUES,X); CHKERRQ(ierr);
  return 0;
} 
/* ------------------------------------------------------------------- */
/* 
   ApplicationInitialGuess - Computes local initial approximation, called 
   by the higher level routine FormInitialGuess().
 
   Input Parameter:
   x - local vector data
 
   Output Parameters:
   x - local vector data
   ierr - error code 
 
   Notes:
   This routine uses standard Fortran-style computations over a 2-dim array.
*/
int ApplicationInitialGuess(AppCtx *user,Scalar *x)
{
  int     i, j, k, row, mx, my, mz, ierr;
  int     xs, ys, zs, xm, ym, zm, gxm, gym, gzm, gxs, gys, gzs;
  double  one = 1.0, lambda, temp1, hx, hy, h_z, temp_k, temp_jk;

  mx     = user->mx;
  my     = user->my;
  mz     = user->mz;
  lambda = user->param;
  hx     = one/(double)(mx-1);
  hy     = one/(double)(my-1);
  h_z     = one/(double)(mz-1);
  temp1  = lambda/(lambda + one);

  /*
     Get local grid boundaries (for 2-dimensional DA):
       xs, ys, zs    - starting grid indices (no ghost points)
       xm, ym, zm    - widths of local grid (no ghost points)
       gxs, gys, gzs - starting grid indices (including ghost points)
       gxm, gym, gzm - widths of local grid (including ghost points)
  */
  ierr = DAGetCorners(user->da,&xs,&ys,&zs,&xm,&ym,&zm); CHKERRQ(ierr);
  ierr = DAGetGhostCorners(user->da,&gxs,&gys,&gzs,&gxm,&gym,&gzm); CHKERRQ(ierr);

  /*
     Compute initial guess over the locally owned part of the grid
  */
  for (k=zs; k<zs+zm; k++) {
    temp_k = (double)PetscMin(k,mz-k-1)*h_z;
    for (j=ys; j<ys+ym; j++) {
      temp_jk = PetscMin((double)(PetscMin(j,my-j-1))*hy,temp_k);
      for (i=xs; i<xs+xm; i++) {
        row = (i - gxs) + (j - gys)*gxm + (k - gzs)*gxm*gym;
        if (i == 0 || j == 0 || k == 0 ||  i == mx-1 || j == my-1 || k == mz-1 ) {
          x[row] = 0.0;
          continue;
        }
        x[row] = temp1*sqrt( PetscMin((double)(PetscMin(i,mx-i-1))*hx,temp_jk) );
      }
    }
  }

  return 0;
}
/* ------------------------------------------------------------------- */
/* 
   FormFunction - Evaluates nonlinear function, F(x).

   Input Parameters:
.  snes - the SNES context
.  X - input vector
.  ptr - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  F - vector in which the nonlinear function is computed

   Notes:
   This routine serves as a wrapper for the lower-level routine
   "ApplicationFunction", where the actual computations are 
   done locally by treating the local vector data as an
   array over the local mesh.  This routine merely handles ghost
   point scatters and accesses the local vector data via
   VecGetArray() and VecRestoreArray().
 */
int FormFunction(SNES snes,Vec X,Vec F,void *ptr)
{
  AppCtx  *user = (AppCtx *) ptr;
  Scalar  *x, *f;
  int     ierr;
  Vec     localX = user->localX, localF = user->localF; 

  /*
     Scatter ghost points to local vector, using the 2-step process
        DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DAGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX); CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX); CHKERRQ(ierr);

  /*
     Get pointers to vector data
  */
  ierr = VecGetArray(localX,&x); CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f); CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  ierr = ApplicationFunction(user,x,f); CHKERRQ(ierr);

  /*
     Restore vectors
  */
  ierr = VecRestoreArray(localX,&x); CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f); CHKERRQ(ierr);

  /*
     Insert values into global vector
  */
  ierr = DALocalToGlobal(user->da,localF,INSERT_VALUES,F); CHKERRQ(ierr);

  /*
     Print vectors if desired (primarily intended for debugging)
     Note: Since these vectors were obtained from a DA (distributed
           array), VecView() reorders the vectors to the natural 
           ordering that would be used in the uniprocessor case.
  */
  if (user->debug) {
     ierr = PetscPrintf(PETSC_COMM_WORLD,"Vector X:\n");CHKERRQ(ierr);
     ierr = VecView(X,VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
     ierr = PetscPrintf(PETSC_COMM_WORLD,"Vector F(X):\n");CHKERRQ(ierr);
     ierr = VecView(F,VIEWER_STDOUT_WORLD); CHKERRQ(ierr);
  }

  return 0; 
}
/* ------------------------------------------------------------------- */ 
/*
   ApplicationFunction - Computes local nonlinear function, called by
   the higher level routine FormFunction().
 
   Input Parameter:
   x - local vector data
 
   Output Parameters:
   f - local vector data, f(x)
 */
int ApplicationFunction(AppCtx *user,Scalar *x,Scalar *f)
{ 
  int     ierr, i, j, k, row, mx, my, mz;
  int     xs, ys, zs, xm, ym, zm, gxm, gym, gzm, gxs, gys, gzs;
  double  two = 2.0, one = 1.0, lambda, hx, hy, h_z, hxhzdhy, hyhzdhx, hxhydhz;
  Scalar  u_north, u_south, u_east, u_west, u_up, u_down, u;
  Scalar  u_xx, u_yy, u_zz, sc;

  mx	  = user->mx; 
  my	  = user->my;
  mz	  = user->mz;
  lambda  = user->param;
  hx      = one / (double)(mx-1);
  hy      = one / (double)(my-1);
  h_z      = one / (double)(mz-1);
  sc      = hx*hy*h_z;
  hxhzdhy = hx*h_z/hy;
  hyhzdhx = hy*h_z/hx;
  hxhydhz = hx*hy/h_z;

  /*
     Get local grid boundaries
  */
  ierr = DAGetCorners(user->da,&xs,&ys,&zs,&xm,&ym,&zm); CHKERRQ(ierr);
  ierr = DAGetGhostCorners(user->da,&gxs,&gys,&gzs,&gxm,&gym,&gzm); CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
  */
  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      row = (k-gzs)*gxm*gym + (j-gys)*gxm + xs - gxs - 1; 
      for (i=xs; i<xs+xm; i++) {
        row++;
        if (i == 0 || j == 0 || k == 0 || i == mx-1 || j == my-1 || k == mz-1) {
          f[row] = x[row];
          continue;
        }
        u       = x[row];
        u_east  = x[row + 1];
        u_west  = x[row - 1];
        u_north = x[row + gxm];
        u_south = x[row - gxm];
        u_up    = x[row + gxm*gym];
        u_down  = x[row - gxm*gym];
        u_xx    = (-u_east + two*u - u_west)*hyhzdhx;
        u_yy    = (-u_north + two*u - u_south)*hxhzdhy;
        u_zz    = (-u_up + two*u - u_down)*hxhydhz;
        f[row]  = u_xx + u_yy + u_zz - sc*lambda*PetscExpScalar(u);
      }
    }
  }

  return 0;
} 
/* ------------------------------------------------------------------- */
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

   Two methods are available for imposing this transformation
   when setting matrix entries:
     (A) MatSetValuesLocal(), using the local ordering (including
         ghost points!)
         - Use DAGetGlobalIndices() to extract the local-to-global map
         - Associate this map with the matrix by calling
           MatSetLocalToGlobalMapping() once
         - Set matrix entries using the local ordering
           by calling MatSetValuesLocal()
     (B) MatSetValues(), using the global ordering 
         - Use DAGetGlobalIndices() to extract the local-to-global map
         - Then apply this map explicitly yourself
         - Set matrix entries using the global ordering by calling
           MatSetValues()
   Option (A) seems cleaner/easier in many cases, and is the procedure
   used in this example.
*/
int FormJacobian(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  AppCtx  *user = (AppCtx *) ptr;  /* user-defined application context */
  Mat     jac = *J;                /* Jacobian matrix */
  Vec     localX = user->localX;   /* local vector */
  int     ierr, i, j, k, row, mx, my, mz, col[7];
  int     xs, ys, zs, xm, ym, zm, gxm, gym, gzm, gxs, gys, gzs;
  Scalar  two = 2.0, one = 1.0, lambda, v[7], hx, hy, h_z, hxhzdhy, hyhzdhx, hxhydhz, sc, *x;

  mx	  = user->mx; 
  my	  = user->my;
  mz	  = user->mz;
  lambda  = user->param;
  hx      = one / (double)(mx-1);
  hy      = one / (double)(my-1);
  h_z      = one / (double)(mz-1);
  sc      = hx*hy*h_z;
  hxhzdhy = hx*h_z/hy;
  hyhzdhx = hy*h_z/hx;
  hxhydhz = hx*hy/h_z;

  /*
     Scatter ghost points to local vector, using the 2-step process
        DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DAGlobalToLocalBegin(user->da,X,INSERT_VALUES,localX); CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(user->da,X,INSERT_VALUES,localX); CHKERRQ(ierr);

  /*
     Get pointer to vector data
  */
  ierr = VecGetArray(localX,&x); CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DAGetCorners(user->da,&xs,&ys,&zs,&xm,&ym,&zm); CHKERRQ(ierr);
  ierr = DAGetGhostCorners(user->da,&gxs,&gys,&gzs,&gxm,&gym,&gzm); CHKERRQ(ierr);

  /* 
     Compute entries for the locally owned part of the Jacobian.
      - Currently, all PETSc parallel matrix formats are partitioned by
        contiguous chunks of rows across the processors. 
      - Each processor needs to insert only elements that it owns
        locally (but any non-local elements will be sent to the
        appropriate processor during matrix assembly). 
      - Here, we set all entries for a particular row at once.
      - We can set matrix entries either using either
        MatSetValuesLocal() or MatSetValues(), as discussed above.
  */
  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      row = (k-gzs)*gxm*gym + (j-gys)*gxm + xs - gxs - 1; 
      for (i=xs; i<xs+xm; i++) {
        row++;
        /* boundary points */
        if (i == 0 || j == 0 || k==0 || i == mx-1 || j == my-1 || k == mz-1) {
          ierr = MatSetValuesLocal(jac,1,&row,1,&row,&one,INSERT_VALUES); CHKERRQ(ierr);
          continue;
      }
        /* interior grid points */
        v[0] = -hxhydhz; col[0] = row - gxm*gym;
        v[1] = -hxhzdhy; col[1] = row - gxm;
        v[2] = -hyhzdhx; col[2] = row - 1;
        v[3] = two*(hyhzdhx + hxhzdhy + hxhydhz) - sc*lambda*PetscExpScalar(x[row]); col[3] = row;
        v[4] = -hyhzdhx; col[4] = row + 1;
        v[5] = -hxhzdhy; col[5] = row + gxm;
        v[6] = -hxhydhz; col[6] = row + gxm*gym;
        ierr = MatSetValuesLocal(jac,1,&row,7,col,v,INSERT_VALUES); CHKERRQ(ierr);
      }
    }
  }

  /* 
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = VecRestoreArray(localX,&x); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

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
  ierr = MatSetOption(jac,MAT_NEW_NONZERO_LOCATION_ERR); CHKERRQ(ierr);
  return 0;
}

