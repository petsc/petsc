/*$Id: ex8.c,v 1.32 2000/06/14 20:42:11 bsmith Exp $*/

static char help[] = "Solves a nonlinear system in parallel with SNES.\n\
  \n\
The 2D driven cavity problem is solved in a velocity-vorticity formulation.\n\
The flow can be driven with the lid or with bouyancy or both:\n\
  -lidvelocity <lid>, where <lid> = dimensionless velocity of lid\n\
  -grashof <gr>, where <gr> = dimensionless temperature gradient\n\
  -prandtl <pr>, where <pr> = dimensionless thermal/momentum diffusity ratio\n\
Mesh parameters are:\n\
  -mx <xg>, where <xg> = number of grid points in the x-direction\n\
  -my <yg>, where <yg> = number of grid points in the y-direction\n\
  -printg : print grid information\n\
Graphics of the contours of (U,V,Omega,T) are available on each grid:\n\
  -contours : draw contour plots of solution\n\
Parallelism can be invoked based on the DA construct:\n\
  -Nx <npx>, where <npx> = number of processors in the x-direction\n\
  -Ny <npy>, where <npy> = number of processors in the y-direction\n\n";

/*T
   Concepts: SNES^Solving a system of nonlinear equations (parallel multicomponent example);
   Concepts: DA^Using distributed arrays;
   Routines: SNESCreate(); SNESSetFunction(); SNESSetJacobian();
   Routines: SNESSolve(); SNESSetFromOptions(); DAView(); DAGetColoring();
   Routines: DACreate2d(); DADestroy(); DACreateGlobalVector(); DACreateLocalVector();
   Routines: DAGetCorners(); DAGetGhostCorners(); DALocalToGlobal(); DASetFieldName();
   Routines: DAGlobalToLocalBegin(); DAGlobalToLocalEnd(); 
   Routines: MatFDColoringCreate(); MatFDColoringSetFunction();
   Routines: MatFDColoringSetFromOptions(); MatFDColoringDestroy(); 
   Routines: ISColoringGetIS(); ISView(); ISColoringDestroy(); 

   Processors: n
T*/

/* ------------------------------------------------------------------------

    We thank David E. Keyes for contributing the driven cavity discretization
    within this example code.

    This example solves the problem for a single grid; mesh sequencing
    is incorporated for the same problem in ex9.c; additional files needed:
        common8and9.c - initial guess and nonlinear function evaluation 
                        routines (used by both ex8.c and ex9.c)
        ex8and9.h     - include file used by ex8.c and ex9.c

    This problem is modeled by the partial differential equation system
  
	- Lap(U) - Grad_y(Omega) = 0
	- Lap(V) + Grad_x(Omega) = 0
	- Lap(Omega) + Div([U*Omega,V*Omega]) - GR*Grad_x(T) = 0
	- Lap(T) + PR*Div([U*T,V*T]) = 0

    in the unit square, which is uniformly discretized in each of x and
    y in this simple encoding.

    No-slip, rigid-wall Dirichlet conditions are used for [U,V].
    Dirichlet conditions are used for Omega, based on the definition of
    vorticity: Omega = - Grad_y(U) + Grad_x(V), where along each
    constant coordinate boundary, the tangential derivative is zero.
    Dirichlet conditions are used for T on the left and right walls,
    and insulation homogeneous Neumann conditions are used for T on
    the top and bottom walls. 

    A finite difference approximation with the usual 5-point stencil 
    is used to discretize the boundary value problem to obtain a 
    nonlinear system of equations.  Upwinding is used for the divergence
    (convective) terms and central for the gradient (source) terms.
    
    The Jacobian can be either
      * formed via finite differencing using coloring (the default), or
      * applied matrix-free via the option -snes_mf 
        (for larger grid problems this variant may not converge 
        without a preconditioner due to ill-conditioning).

  ------------------------------------------------------------------------- */

/* 
   Include "petscda.h" so that we can use distributed arrays (DAs).
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - system routines       petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscsles.h   - linear solvers 
*/
#include "petscsnes.h"
#include "petscda.h"
#include "ex8and9.h"

/* 
   User-defined routines
*/
extern int FormInitialGuess(AppCtx*,Vec);
extern int FormFunction(SNES,Vec,Vec,void*);
extern int InitializeProblem(int,AppCtx*,Vec*);

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  SNES          snes;                /* nonlinear solver */
  Vec           x,r;                /* solution, residual vectors */
  Mat           J;                   /* Jacobian matrix */
  AppCtx        user;                /* user-defined work context */
  int           its;                 /* iterations for convergence */
  MatFDColoring fdcoloring;          /* matrix coloring context */
  int           dim,k = 0,ierr;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&user.rank);CHKERRA(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&user.size);CHKERRA(ierr);
  user.comm = PETSC_COMM_WORLD;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create user context, set problem data, create vector data structures.
    Also, compute the initial guess.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = InitializeProblem(k,&user,&x);CHKERRA(ierr);
  ierr = VecDuplicate(x,&r);CHKERRA(ierr);
  ierr = VecDuplicate(user.localX,&user.localF);CHKERRA(ierr);

 /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = SNESCreate(PETSC_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes);CHKERRA(ierr);
  ierr = VecGetSize(x,&dim);CHKERRA(ierr);
  ierr = PetscPrintf(user.comm,"global size = %d, lid velocity = %g, prandtl # = %g, grashof # = %g\n",
     dim,user.lidvelocity,user.prandtl,user.grashof);CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set function evaluation routine and vector
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    ierr = SNESSetFunction(snes,r,FormFunction,&user);CHKERRA(ierr);

   /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create matrix data structure; set Jacobian evaluation routine
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Set up coloring information needed for sparse finite difference
     approximation of the Jacobian
   */
  {
  ISColoring iscoloring;
  ierr = DAGetColoring(user.da,&iscoloring,&J);CHKERRQ(ierr);

  ierr = MatFDColoringCreate(J,iscoloring,&fdcoloring);CHKERRQ(ierr); 
  ierr = MatFDColoringSetFunction(fdcoloring,(int (*)(void))FormFunction,&user);CHKERRQ(ierr);
  ierr = MatFDColoringSetFromOptions(fdcoloring);CHKERRQ(ierr); 
  ierr = ISColoringDestroy(iscoloring);CHKERRQ(ierr);
  }

  /* 
     Set Jacobian matrix data structure and default Jacobian evaluation
     routine. User can override with:

     -snes_mf : matrix-free Newton-Krylov method with no preconditioning
     -snes_mf_operator : form preconditioning matrix as set by the user,
                         but use matrix-free approx for Jacobian-vector
                         products within Newton-Krylov method

  */

  ierr = SNESSetJacobian(snes,J,J,SNESDefaultComputeJacobianColor,fdcoloring);CHKERRA(ierr);

 /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Customize nonlinear solver; set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /*
     Set runtime options (e.g., -snes_monitor -snes_rtol <rtol> -ksp_type <type>)
  */
  ierr = SNESSetFromOptions(snes);CHKERRA(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve the nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /*
     Note: The user should initialize the vector, x, with the initial guess
     for the nonlinear solver prior to calling SNESSolve().  In particular,
     to employ an initial guess of zero, the user should explicitly set
     this vector to zero by calling VecSet(). [Here we set the initial 
     guess in the routine InitializeProblem().]
  */
  ierr = SNESSolve(snes,x,&its);CHKERRA(ierr); 

  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of Newton iterations = %d\n", its);CHKERRA(ierr);

  /*
     Visualize solution
  */

  if (user.draw_contours) {
    ierr = VecView(x,VIEWER_DRAW_WORLD);CHKERRA(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = MatDestroy(J);CHKERRA(ierr);
  ierr = MatFDColoringDestroy(fdcoloring);CHKERRA(ierr);  
  ierr = VecDestroy(user.localX);CHKERRA(ierr); ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = VecDestroy(user.localF);CHKERRA(ierr); ierr = VecDestroy(r);CHKERRA(ierr);      
  ierr = SNESDestroy(snes);CHKERRA(ierr);  
  ierr = DADestroy(user.da);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "InitializeProblem"
/* 
   InitializeProblem - Initializes the problem.  This routine forms the
   DA and vector data structures, and also computes the starting solution
   guess for the nonlinear solver.

   Input Parameters:
   user - user-defined application context

   Output Parameter:
   xvec - solution vector
 */
int InitializeProblem(int icycle,AppCtx *user,Vec *xvec)
{
  int    Nx,Ny;              /* number of processors in x- and y- directions */
  int    xs,xm,ys,ym,Nlocal,ierr;
  Vec    xv;

  /*
     Initialize problem parameters
  */
  user->mx = 4; user->my = 4; 
  ierr = OptionsGetInt(PETSC_NULL,"-mx",&user->mx,PETSC_NULL);CHKERRQ(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-my",&user->my,PETSC_NULL);CHKERRQ(ierr);
  /*
     Number of components in the unknown vector and auxiliary vector
  */
  user->mc = 4;
  /* 
     Problem parameters (velocity of lid, prandtl, and grashof numbers)
  */
  user->lidvelocity = 1.0/(user->mx*user->my);
  ierr = OptionsGetDouble(PETSC_NULL,"-lidvelocity",&user->lidvelocity,PETSC_NULL);CHKERRQ(ierr);
  user->prandtl = 1.0;
  ierr = OptionsGetDouble(PETSC_NULL,"-prandtl",&user->prandtl,PETSC_NULL);CHKERRQ(ierr);
  user->grashof = 1.0;
  ierr = OptionsGetDouble(PETSC_NULL,"-grashof",&user->grashof,PETSC_NULL);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-printv",&user->print_vecs);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-printg",&user->print_grid);CHKERRQ(ierr);
  ierr = OptionsHasName(PETSC_NULL,"-contours",&user->draw_contours);CHKERRQ(ierr);

  /*
     Create distributed array (DA) to manage parallel grid and vectors
     for principal unknowns (x) and governing residuals (f)
  */
  Nx = PETSC_DECIDE; Ny = PETSC_DECIDE;
  ierr = OptionsGetInt(PETSC_NULL,"-Nx",&Nx,PETSC_NULL);CHKERRQ(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-Ny",&Ny,PETSC_NULL);CHKERRQ(ierr);
  ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,user->mx,
                    user->my,Nx,Ny,user->mc,1,PETSC_NULL,PETSC_NULL,&user->da);CHKERRQ(ierr);
  ierr = DASetFieldName(user->da,0,"x-velocity");CHKERRQ(ierr);
  ierr = DASetFieldName(user->da,1,"y-velocity");CHKERRQ(ierr);
  ierr = DASetFieldName(user->da,2,"Omega");CHKERRQ(ierr);
  ierr = DASetFieldName(user->da,3,"temperature");CHKERRQ(ierr);

  /*
     Create global and local vectors from DA
  */
  ierr = DACreateGlobalVector(user->da,&xv);CHKERRQ(ierr);
  ierr = DACreateLocalVector(user->da,&user->localX);CHKERRQ(ierr);

  /* Print grid info */
  if (user->print_grid) {
    ierr = DAView(user->da,VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    ierr = DAGetCorners(user->da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"global grid: %d X %d with %d components per node ==> global vector dimension %d\n",
      user->mx,user->my,user->mc,user->mc*user->mx*user->my);CHKERRQ(ierr);
    ierr = VecGetLocalSize(xv,&Nlocal);CHKERRQ(ierr);
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"[%d] local grid %d X %d with %d components per node ==> local vector dimension %d\n",
      user->rank,xm,ym,user->mc,Nlocal);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRQ(ierr);
  }  

  /* Compute initial guess */
  FormInitialGuess(user,xv);CHKERRQ(ierr);
  
  *xvec = xv;
  return 0;
}
