
static char help[] = "2d Bratu problem in shared memory parallel with SNES.\n\
We solve the  Bratu (SFI - solid fuel ignition) problem in a 2D rectangular\n\
domain, uses SHARED MEMORY to evaluate the user function.\n\
The command line options include:\n\
  -par <parameter>, where <parameter> indicates the problem's nonlinearity\n\
     problem SFI:  <parameter> = Bratu parameter (0 <= par <= 6.81)\n\
  -mx <xg>, where <xg> = number of grid points in the x-direction\n\
  -my <yg>, where <yg> = number of grid points in the y-direction\n\
  -use_fortran_function: use Fortran coded function, rather than C\n";

/*
             This code compiles ONLY on SGI systems
            ========================================
*/
/*T
   Concepts: SNES^parallel Bratu example
   Concepts: shared memory
   Processors: n
T*/

/*

     Programming model: Combination of 
        1) MPI message passing for PETSc routines
        2) automatic loop parallism (using shared memory) for user
           provided function.

       While the user function is being evaluated all MPI processes except process
     0 blocks. Process zero spawns nt threads to evaluate the user function. Once 
     the user function is complete, the worker threads are suspended and all the MPI processes
     continue.

     Other useful options:

       -snes_mf : use matrix free operator and no preconditioner
       -snes_mf_operator : use matrix free operator but compute Jacobian via
                           finite differences to form preconditioner

       Environmental variable:

         setenv MPC_NUM_THREADS nt <- set number of threads processor 0 should 
                                      use to evaluate user provided function

       Note: The number of MPI processes (set with the mpiexec option -np) can 
       be set completely independently from the number of threads process 0 
       uses to evaluate the function (though usually one would make them the same).
*/
       
/* ------------------------------------------------------------------------

    Solid Fuel Ignition (SFI) problem.  This problem is modeled by
    the partial differential equation
  
            -Laplacian u - lambda*exp(u) = 0,  0 < x,y < 1,
  
    with boundary conditions
   
             u = 0  for  x = 0, x = 1, y = 0, y = 1.
  
    A finite difference approximation with the usual 5-point stencil
    is used to discretize the boundary value problem to obtain a nonlinear 
    system of equations.

    The uniprocessor version of this code is snes/examples/tutorials/ex4.c
    A parallel distributed memory version is snes/examples/tutorials/ex5.c and ex5f.F

  ------------------------------------------------------------------------- */

/* 
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/
#include <petscsnes.h>

/* 
   User-defined application context - contains data needed by the 
   application-provided call-back routines   FormFunction().
*/
typedef struct {
   PetscReal   param;          /* test problem parameter */
   int         mx,my;          /* discretization in x, y directions */
   int         rank;           /* processor rank */
} AppCtx;

/* 
   User-defined routines
*/
extern int FormFunction(SNES,Vec,Vec,void*),FormInitialGuess(AppCtx*,Vec);
extern int FormFunctionFortran(SNES,Vec,Vec,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
/* 
    The main program is written in C while the user provided function
 is given in both Fortran and C. The main program could also be written 
 in Fortran; the ONE PROBLEM is that VecGetArray() cannot be called from 
 Fortran on the SGI machines; thus the routine FormFunctionFortran() must
 be written in C.
*/
int main(int argc,char **argv)
{
  SNES           snes;                /* nonlinear solver */
  Vec            x,r;                 /* solution, residual vectors */
  AppCtx         user;                /* user-defined work context */
  int            its;                 /* iterations for convergence */
  int            N,ierr,rstart,rend,*colors,i,ii,ri,rj;
  PetscErrorCode (*fnc)(SNES,Vec,Vec,void*);
  PetscReal      bratu_lambda_max = 6.81,bratu_lambda_min = 0.;
  MatFDColoring  fdcoloring;           
  ISColoring     iscoloring;
  Mat            J;
  PetscScalar    zero = 0.0;
  PetscBool      flg;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&user.rank);CHKERRQ(ierr);

  /*
     Initialize problem parameters
  */
  user.mx = 4; user.my = 4; user.param = 6.0;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mx",&user.mx,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-my",&user.my,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-par",&user.param,PETSC_NULL);CHKERRQ(ierr);
  if (user.param >= bratu_lambda_max || user.param <= bratu_lambda_min) SETERRQ(PETSC_COMM_SELF,1,"Lambda is out of range");
  N = user.mx*user.my;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SNESCreate(PETSC_COMM_WORLD,&snes);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create vector data structures; set function evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
      The routine VecCreateShared() creates a parallel vector with each processor
    assigned its own segment, BUT, in addition, the first processor has access to the 
    entire array. This is to allow the users function to be based on loop level
    parallelism rather than MPI.
  */
  ierr = VecCreateShared(PETSC_COMM_WORLD,PETSC_DECIDE,N,&x);
  ierr = VecDuplicate(x,&r);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-use_fortran_function",&flg);CHKERRQ(ierr);
  if (flg) {
    fnc = FormFunctionFortran;
  } else {
    fnc = FormFunction;
  }

  /* 
     Set function evaluation routine and vector
  */
  ierr = SNESSetFunction(snes,r,fnc,&user);CHKERRQ(ierr);

  /*
       Currently when using VecCreateShared() and using loop level parallelism
    to automatically parallelise the user function it makes no sense for the 
    Jacobian to be computed via loop level parallelism, because all the threads
    would be simultaneously calling MatSetValues() causing a bottle-neck.

    Thus this example uses the PETSc Jacobian calculations via finite differencing
    to approximate the Jacobian
  */ 

  /*

  */
  ierr = VecGetOwnershipRange(r,&rstart,&rend);CHKERRQ(ierr);
  ierr = PetscMalloc((rend-rstart)*sizeof(PetscInt),&colors);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    colors[i - rstart] = 3*((i/user.mx) % 3) + (i % 3);
  }
  ierr   = ISColoringCreate(PETSC_COMM_WORLD,3*2+2,rend-rstart,colors,&iscoloring);CHKERRQ(ierr);
  ierr = PetscFree(colors);CHKERRQ(ierr);

  /*
     Create and set the nonzero pattern for the Jacobian: This is not done 
     particularly efficiently. One should process the boundary nodes separately and 
     then use a simple loop for the interior nodes.
       Note that for this code we use the "natural" number of the nodes on the 
     grid (since that is what is good for the user provided function). In the 
     DMDA examples we must use the DMDA numbering where each processor is assigned a
     chunk of data.
  */
  ierr = MatCreateAIJ(PETSC_COMM_WORLD,rend-rstart,rend-rstart,N,N,5,0,0,0,&J);CHKERRQ(ierr);
  for (i=rstart; i<rend; i++) {
    rj = i % user.mx;         /* column in grid */
    ri = i / user.mx;         /* row in grid */
    if (ri != 0) {     /* first row does not have neighbor below */
      ii   = i - user.mx;
      ierr = MatSetValues(J,1,&i,1,&ii,&zero,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (ri != user.my - 1) { /* last row does not have neighbors above */
      ii   = i + user.mx;
      ierr = MatSetValues(J,1,&i,1,&ii,&zero,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (rj != 0) {     /* first column does not have neighbor to left */
      ii   = i - 1;
      ierr = MatSetValues(J,1,&i,1,&ii,&zero,INSERT_VALUES);CHKERRQ(ierr);
    }
    if (rj != user.mx - 1) {     /* last column does not have neighbor to right */
      ii   = i + 1;
      ierr = MatSetValues(J,1,&i,1,&ii,&zero,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatSetValues(J,1,&i,1,&i,&zero,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /*
       Create the data structure that SNESDefaultComputeJacobianColor() uses
       to compute the actual Jacobians via finite differences.
  */
  ierr = MatFDColoringCreate(J,iscoloring,&fdcoloring);CHKERRQ(ierr);
  ierr = MatFDColoringSetFunction(fdcoloring,(PetscErrorCode (*)(void))fnc,&user);CHKERRQ(ierr);
  ierr = MatFDColoringSetFromOptions(fdcoloring);CHKERRQ(ierr);
  /*
        Tell SNES to use the routine SNESDefaultComputeJacobianColor()
      to compute Jacobians.
  */
  ierr = SNESSetJacobian(snes,J,J,SNESDefaultComputeJacobianColor,fdcoloring);CHKERRQ(ierr);  
  ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);


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
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of SNES iterations = %D\n",its);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);      
  ierr = SNESDestroy(&snes);CHKERRQ(ierr); 
  ierr = PetscFinalize();

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
int FormInitialGuess(AppCtx *user,Vec X)
{
  int          i,j,row,mx,my,ierr;
  PetscReal    one = 1.0,lambda,temp1,temp,hx,hy,hxdhy,hydhx,sc;
  PetscScalar  *x;

  /*
      Process 0 has to wait for all other processes to get here 
   before proceeding to write in the shared vector
  */
  ierr = PetscBarrier((PetscObject)X);CHKERRQ(ierr);
  if (user->rank) {
     /*
        All the non-busy processors have to wait here for process 0 to finish
        evaluating the function; otherwise they will start using the vector values
        before they have been computed
     */
     ierr = PetscBarrier((PetscObject)X);CHKERRQ(ierr);
     return 0;
  }

  mx = user->mx;            my = user->my;            lambda = user->param;
  hx = one/(PetscReal)(mx-1);  hy = one/(PetscReal)(my-1);
  sc = hx*hy*lambda;        hxdhy = hx/hy;            hydhx = hy/hx;
  temp1 = lambda/(lambda + one);

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);

  /*
     Compute initial guess over the locally owned part of the grid
  */
#pragma arl(4)
#pragma distinct (*x,*f)
#pragma no side effects (sqrt)
  for (j=0; j<my; j++) {
    temp = (PetscReal)(PetscMin(j,my-j-1))*hy;
    for (i=0; i<mx; i++) {
      row = i + j*mx; 
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
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);

  ierr = PetscBarrier((PetscObject)X);CHKERRQ(ierr);
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
int FormFunction(SNES snes,Vec X,Vec F,void *ptr)
{
  AppCtx       *user = (AppCtx*)ptr;
  int          ierr,i,j,row,mx,my;
  PetscReal    two = 2.0,one = 1.0,lambda,hx,hy,hxdhy,hydhx,sc;
  PetscScalar  u,uxx,uyy,*x,*f;

  /*
      Process 0 has to wait for all other processes to get here 
   before proceeding to write in the shared vector
  */
  ierr = PetscBarrier((PetscObject)X);CHKERRQ(ierr);

  if (user->rank) {
     /*
        All the non-busy processors have to wait here for process 0 to finish
        evaluating the function; otherwise they will start using the vector values
        before they have been computed
     */
     ierr = PetscBarrier((PetscObject)X);CHKERRQ(ierr);
     return 0;
  }

  mx = user->mx;            my = user->my;            lambda = user->param;
  hx = one/(PetscReal)(mx-1);  hy = one/(PetscReal)(my-1);
  sc = hx*hy*lambda;        hxdhy = hx/hy;            hydhx = hy/hx;

  /*
     Get pointers to vector data
  */
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  /*
      The next line tells the SGI compiler that x and f contain no overlapping 
    regions and thus it can use addition optimizations.
  */
#pragma arl(4)
#pragma distinct (*x,*f)
#pragma no side effects (exp)

  /*
     Compute function over the entire  grid
  */
  for (j=0; j<my; j++) {
    for (i=0; i<mx; i++) {
      row = i + j*mx;
      if (i == 0 || j == 0 || i == mx-1 || j == my-1) {
        f[row] = x[row];
        continue;
      } 
      u = x[row];
      uxx = (two*u - x[row-1] - x[row+1])*hydhx;
      uyy = (two*u - x[row-mx] - x[row+mx])*hxdhy;
      f[row] = uxx + uyy - sc*exp(u);
    }
  }

  /*
     Restore vectors
  */
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);

  ierr = PetscLogFlops(11.0*(mx-2)*(my-2))CHKERRQ(ierr);
  ierr = PetscBarrier((PetscObject)X);CHKERRQ(ierr);
  return 0; 
} 

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define applicationfunctionfortran_ APPLICATIONFUNCTIONFORTRAN
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define applicationfunctionfortran_ applicationfunctionfortran
#endif

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormFunctionFortran"
/* 
   FormFunctionFortran - Evaluates nonlinear function, F(x) in Fortran.

*/
int FormFunctionFortran(SNES snes,Vec X,Vec F,void *ptr)
{
  AppCtx  *user = (AppCtx*)ptr;
  int     ierr;
  PetscScalar  *x,*f;

  /*
      Process 0 has to wait for all other processes to get here 
   before proceeding to write in the shared vector
  */
  ierr = PetscBarrier((PetscObject)snes);CHKERRQ(ierr);
  if (!user->rank) {
    ierr = VecGetArray(X,&x);CHKERRQ(ierr);
    ierr = VecGetArray(F,&f);CHKERRQ(ierr);
    applicationfunctionfortran_(&user->param,&user->mx,&user->my,x,f,&ierr);
    ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
    ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
    ierr = PetscLogFlops(11.0*(user->mx-2)*(user->my-2))CHKERRQ(ierr);
  }
  /*
      All the non-busy processors have to wait here for process 0 to finish
      evaluating the function; otherwise they will start using the vector values
      before they have been computed
  */
  ierr = PetscBarrier((PetscObject)snes);CHKERRQ(ierr);
  return 0;
}


