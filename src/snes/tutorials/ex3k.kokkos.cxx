

static char help[] = "Newton methods to solve u'' + u^{2} = f in parallel. Uses Kokkos\n\\n";

/*
    This is a simplied version of ex3.c except it uses Kokkos to set the initial conditions
*/

/*
   Include "petscdm.h" so that we can use data management objects (DMs)
   Include "petscdmda.h" so that we can use distributed arrays (DMDAs).
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
*/

#include <petscdm.h>
#include <petscdmda.h>
#include <petscsnes.h>

/*
    Include Kokkos files

*/
#include <Kokkos_Core.hpp>
#include <Kokkos_OffsetView.hpp>

/*
   Application-defined routines.
*/
PetscErrorCode FormJacobian(SNES,Vec,Mat,Mat,void*);
PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
PetscErrorCode FormInitialGuess(Vec);

/*
   User-defined application context
*/
typedef struct {
  DM          da;      /* distributed array */
  Vec         F;       /* right-hand-side of PDE */
  PetscReal   h;       /* mesh spacing */
} ApplicationCtx;

int main(int argc,char **argv)
{
  SNES           snes;                 /* SNES context */
  Mat            J;                    /* Jacobian matrix */
  ApplicationCtx ctx;                  /* user-defined context */
  Vec            x,r,U,F;              /* vectors */
  PetscScalar    *FF,*UU,none = -1.0;
  PetscErrorCode ierr;
  PetscInt       its,N = 5,maxit,maxf,xs,xm;
  PetscReal      abstol,rtol,stol,norm;
  PetscBool      viewinitial = PETSC_FALSE;
  PetscBool      view_kokkos_configuration = PETSC_TRUE;


  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr  = PetscOptionsGetInt(NULL,NULL,"-n",&N,NULL);CHKERRQ(ierr);
  ierr  = PetscOptionsGetBool(NULL,NULL,"-view_kokkos_configuration",&view_kokkos_configuration,NULL);CHKERRQ(ierr);
  ctx.h = 1.0/(N-1);
  ierr  = PetscOptionsGetBool(NULL,NULL,"-view_initial",&viewinitial,NULL);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures; set function evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Create distributed array (DMDA) to manage parallel grid and vectors
  */
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,N,1,1,NULL,&ctx.da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(ctx.da);CHKERRQ(ierr);
  ierr = DMSetUp(ctx.da);CHKERRQ(ierr);

  /*
     Extract global and local vectors from DMDA; then duplicate for remaining
     vectors that are the same types
  */
  ierr = DMCreateGlobalVector(ctx.da,&x);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)x,"Approximate Solution");CHKERRQ(ierr);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&F);CHKERRQ(ierr); ctx.F = F;
  ierr = PetscObjectSetName((PetscObject)F,"Forcing function");CHKERRQ(ierr);
  ierr = VecDuplicate(x,&U);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject)U,"Exact Solution");CHKERRQ(ierr);

  /*
     Set function evaluation routine and vector.  Whenever the nonlinear
     solver needs to compute the nonlinear function, it will call this
     routine.
      - Note that the final routine argument is the user-defined
        context that provides application-specific data for the
        function evaluation routine.
  */
  ierr = SNESSetFunction(snes,r,FormFunction,&ctx);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = DMCreateMatrix(ctx.da,&J);CHKERRQ(ierr);

  /*
     Set Jacobian matrix data structure and default Jacobian evaluation
     routine.  Whenever the nonlinear solver needs to compute the
     Jacobian matrix, it will call this routine.
      - Note that the final routine argument is the user-defined
        context that provides application-specific data for the
        Jacobian evaluation routine.
  */
  ierr = SNESSetJacobian(snes,J,J,FormJacobian,&ctx);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = SNESGetTolerances(snes,&abstol,&rtol,&stol,&maxit,&maxf);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"atol=%g, rtol=%g, stol=%g, maxit=%D, maxf=%D\n",(double)abstol,(double)rtol,(double)stol,maxit,maxf);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize application:
     Store forcing function of PDE and exact solution
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Get local grid boundaries (for 1-dimensional DMDA):
       xs, xm - starting grid index, width of local grid (no ghost points)
  */
  ierr = DMDAGetCorners(ctx.da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  /*
     Get pointers to vector data
  */
  ierr = VecGetArrayWrite(F,&FF);CHKERRQ(ierr);
  ierr = VecGetArrayWrite(U,&UU);CHKERRQ(ierr);

  /* -------------------------------------------------------------------------------------------------- */
  /*  ./configure --download-kokkos --download-hwloc [one of --with-openmp --with-pthread --with-cuda] */
  /*  export OMP_PROC_BIND="threads"  export OMP_PROC_BIND="spread" */

  if (!getenv("KOKKOS_NUM_THREADS")) setenv("KOKKOS_NUM_THREADS","4",1);
  if (!view_kokkos_configuration) {
    /* use private routine to turn off warnings about negative number of cores etc */
    Kokkos::Impl::pre_initialize(Kokkos::InitArguments(-1, -1, -1, true));
  }
  Kokkos::initialize( argc, argv );
  if (view_kokkos_configuration) {
    Kokkos::print_configuration(std::cout, true);
  }

  /* introduce a view object; reference like object  */
  Kokkos::Experimental::OffsetView<PetscScalar*> xFF(Kokkos::View<PetscScalar*>(FF,xm),{xs}), xUU(Kokkos::View<PetscScalar*>(UU,xm),{xs});

  PetscReal xpbase = xs*ctx.h;
  Kokkos:: parallel_for(Kokkos::RangePolicy<> (xs,xm), KOKKOS_LAMBDA ( int j ) {
    PetscReal xp = xpbase + j*ctx.h;
    xFF(j) = 6.0*xp + PetscPowScalar(xp+1.e-12,6.0); /* +1.e-12 is to prevent 0^6 */
    xUU(j) = xp*xp*xp;
  });

  Kokkos::finalize();

  /*
     Restore vectors
  */
  ierr = VecRestoreArrayWrite(F,&FF);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(U,&UU);CHKERRQ(ierr);
  if (viewinitial) {
    ierr = VecView(U,NULL);CHKERRQ(ierr);
    ierr = VecView(F,NULL);CHKERRQ(ierr);
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
  ierr = FormInitialGuess(x);CHKERRQ(ierr);
  ierr = SNESSolve(snes,NULL,x);CHKERRQ(ierr);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of SNES iterations = %D\n",its);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Check solution and clean up
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Check the error
  */
  ierr = VecAXPY(x,none,U);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %g Iterations %D\n",(double)norm,its);CHKERRQ(ierr);

  /*
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&U);CHKERRQ(ierr);
  ierr = VecDestroy(&F);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&ctx.da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/* ------------------------------------------------------------------- */
/*
   FormInitialGuess - Computes initial guess.

   Input/Output Parameter:
.  x - the solution vector
*/
PetscErrorCode FormInitialGuess(Vec x)
{
  PetscErrorCode ierr;
  PetscScalar    pfive = .50;

  PetscFunctionBeginUser;
  ierr = VecSet(x,pfive);CHKERRQ(ierr);
  PetscFunctionReturn(0);
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
   The user-defined context can contain any application-specific
   data needed for the function evaluation.
*/
PetscErrorCode FormFunction(SNES snes,Vec x,Vec f,void *ctx)
{
  ApplicationCtx *user = (ApplicationCtx*) ctx;
  DM             da    = user->da;
  PetscScalar    *xx,*ff,*FF,d;
  PetscErrorCode ierr;
  PetscInt       i,M,xs,xm;
  Vec            xlocal;

  PetscFunctionBeginUser;
  ierr = DMGetLocalVector(da,&xlocal);CHKERRQ(ierr);
  /*
     Scatter ghost points to local vector, using the 2-step process
        DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
     By placing code between these two statements, computations can
     be done while messages are in transition.
  */
  ierr = DMGlobalToLocalBegin(da,x,INSERT_VALUES,xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,x,INSERT_VALUES,xlocal);CHKERRQ(ierr);

  /*
     Get pointers to vector data.
       - The vector xlocal includes ghost point; the vectors x and f do
         NOT include ghost points.
       - Using DMDAVecGetArray() allows accessing the values using global ordering
  */
  ierr = DMDAVecGetArray(da,xlocal,&xx);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,f,&ff);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,user->F,&FF);CHKERRQ(ierr);

  /*
     Get local grid boundaries (for 1-dimensional DMDA):
       xs, xm  - starting grid index, width of local grid (no ghost points)
  */
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);

  /*
     Set function values for boundary points; define local interior grid point range:
        xsi - starting interior grid index
        xei - ending interior grid index
  */
  if (xs == 0) { /* left boundary */
    ff[0] = xx[0];
    xs++;xm--;
  }
  if (xs+xm == M) {  /* right boundary */
    ff[xs+xm-1] = xx[xs+xm-1] - 1.0;
    xm--;
  }

  /*
     Compute function over locally owned part of the grid (interior points only)
  */
  d = 1.0/(user->h*user->h);
  for (i=xs; i<xs+xm; i++) ff[i] = d*(xx[i-1] - 2.0*xx[i] + xx[i+1]) + xx[i]*xx[i] - FF[i];

  /*
     Restore vectors
  */
  ierr = DMDAVecRestoreArray(da,xlocal,&xx);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,f,&ff);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,user->F,&FF);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&xlocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
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
PetscErrorCode FormJacobian(SNES snes,Vec x,Mat jac,Mat B,void *ctx)
{
  ApplicationCtx *user = (ApplicationCtx*) ctx;
  PetscScalar    *xx,d,A[3];
  PetscErrorCode ierr;
  PetscInt       i,j[3],M,xs,xm;
  DM             da = user->da;

  PetscFunctionBeginUser;
  /*
     Get pointer to vector data
  */
  ierr = DMDAVecGetArrayRead(da,x,&xx);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  /*
    Get range of locally owned matrix
  */
  ierr = DMDAGetInfo(da,NULL,&M,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);

  /*
     Determine starting and ending local indices for interior grid points.
     Set Jacobian entries for boundary points.
  */

  if (xs == 0) {  /* left boundary */
    i = 0; A[0] = 1.0;

    ierr = MatSetValues(jac,1,&i,1,&i,A,INSERT_VALUES);CHKERRQ(ierr);
    xs++;xm--;
  }
  if (xs+xm == M) { /* right boundary */
    i    = M-1;
    A[0] = 1.0;
    ierr = MatSetValues(jac,1,&i,1,&i,A,INSERT_VALUES);CHKERRQ(ierr);
    xm--;
  }

  /*
     Interior grid points
      - Note that in this case we set all elements for a particular
        row at once.
  */
  d = 1.0/(user->h*user->h);
  for (i=xs; i<xs+xm; i++) {
    j[0] = i - 1; j[1] = i; j[2] = i + 1;
    A[0] = A[2] = d; A[1] = -2.0*d + 2.0*xx[i];
    ierr = MatSetValues(jac,1,&i,3,j,A,INSERT_VALUES);CHKERRQ(ierr);
  }

  /*
     Assemble matrix, using the 2-step process:
       MatAssemblyBegin(), MatAssemblyEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.

     Also, restore vector.
  */

  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayRead(da,x,&xx);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

/*
     The first test works for Kokkos wtih OpenMP and PThreads, the second with CUDA.

     The second test requires -dm_vec_type cuda and -vec_pinned_memory_min 0 because Kokkos
     assumes all memory it is given is allocated with cudaMallocHost() and can be used with
     cudaMemcpy() without indicating it the memory is GPU or CPU. Note this is NOT unified memory.

     The third test requires -dm_mat_type aijcusparse to create the correct vectors for block Jacobi
*/

/*TEST

   build:
     requires: kokkos

   test:
     requires: kokkos double !complex !single !cuda
     args: -view_initial -view_kokkos_configuration false -snes_monitor
     filter: grep -v type:

   test:
     suffix: 2
     requires: kokkos double !complex !single cuda
     args: -dm_vec_type cuda -vec_pinned_memory_min 0 -use_gpu_aware_mpi 0 -view_initial -view_kokkos_configuration false  -snes_monitor
     output_file: output/ex3k_1.out
     filter: grep -v type:

   test:
     suffix: 3
     requires: kokkos double !complex !single defined(PETSC_HAVE_cusparseCreateSolveAnalysisInfo) cuda
     nsize: 2
     args: -dm_vec_type cuda -dm_mat_type aijcusparse -vec_pinned_memory_min 0 -use_gpu_aware_mpi 0 -view_initial -view_kokkos_configuration false  -snes_monitor
     output_file: output/ex3k_1.out
     filter: grep -v type:

TEST*/
