/*$Id: ex19.c,v 1.7 2000/09/28 14:45:23 bsmith Exp bsmith $*/

static char help[] = "Solves nonlinear driven cavity with multigrid.\n\
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
  -contours : draw contour plots of solution\n\n";

/*T
   Concepts: SNES^solving a system of nonlinear equations (parallel multicomponent example);
   Concepts: DA^using distributed arrays;
   Concepts: multicomponent
   Processors: n
T*/

/* ------------------------------------------------------------------------

    This code is the same as ex8.c except it uses a multigrid preconditioner

    We thank David E. Keyes for contributing the driven cavity discretization
    within this example code.

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

/* 
   User-defined routines
*/
extern int FormInitialGuess(SNES,Vec,void*);
extern int FormFunction(SNES,Vec,Vec,void*);

typedef struct {
   double     lidvelocity,prandtl,grashof;  /* physical parameters */
   PetscTruth draw_contours;                /* flag - 1 indicates drawing contours */
} AppCtx;

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  DMMG     *dmmg;               /* multilevel grid structure */
  AppCtx   user;                /* user-defined work context */
  int      mx,my,its;
  int      ierr,nlevels = 2;
  MPI_Comm comm;
  SNES     snes;

  PetscInitialize(&argc,&argv,(char *)0,help);
  comm = PETSC_COMM_WORLD;

  mx = 4; 
  my = 4; 
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mx",&mx,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-my",&my,PETSC_NULL);CHKERRA(ierr);

  /* 
     Problem parameters (velocity of lid, prandtl, and grashof numbers)
  */
  user.lidvelocity = 1.0/(mx*my);
  user.prandtl     = 1.0;
  user.grashof     = 1.0;
  ierr = PetscOptionsGetDouble(PETSC_NULL,"-lidvelocity",&user.lidvelocity,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetDouble(PETSC_NULL,"-prandtl",&user.prandtl,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetDouble(PETSC_NULL,"-grashof",&user.grashof,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-contours",&user.draw_contours);CHKERRQ(ierr);

  PreLoadBegin(PETSC_TRUE,"SetUp");
  ierr = DMMGCreate(comm,nlevels,&user,&dmmg);CHKERRQ(ierr);

  /*
     Create distributed array multigrid object (DMMG) to manage parallel grid and vectors
     for principal unknowns (x) and governing residuals (f)
  */

  ierr = DMMGSetDA(dmmg,2,DA_NONPERIODIC,DA_STENCIL_STAR,mx,my,0,4,1);CHKERRQ(ierr);
  ierr = DASetFieldName(DMMGGetDA(dmmg),0,"x-velocity");CHKERRQ(ierr);
  ierr = DASetFieldName(DMMGGetDA(dmmg),1,"y-velocity");CHKERRQ(ierr);
  ierr = DASetFieldName(DMMGGetDA(dmmg),2,"Omega");CHKERRQ(ierr);
  ierr = DASetFieldName(DMMGGetDA(dmmg),3,"temperature");CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Create user context, set problem data, create vector data structures.
    Also, compute the initial guess.
    - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

 /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = DMMGSetSNES(dmmg,FormFunction,0);
  ierr = PetscPrintf(comm,"lid velocity = %g, prandtl # = %g, grashof # = %g\n",
                     user.lidvelocity,user.prandtl,user.grashof);CHKERRA(ierr);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve the nonlinear system
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMMGSetInitialGuess(dmmg,FormInitialGuess);CHKERRQ(ierr);

  PreLoadStage("Solve");
  ierr = DMMGSolve(dmmg);CHKERRA(ierr); 

  snes = DMMGGetSNES(dmmg);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRA(ierr);
  ierr = PetscPrintf(comm,"Number of Newton iterations = %d\n", its);CHKERRA(ierr);

  /*
     Visualize solution
  */

  if (user.draw_contours) {
    ierr = VecView(DMMGGetx(dmmg),PETSC_VIEWER_DRAW_WORLD);CHKERRA(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = DMMGDestroy(dmmg);CHKERRA(ierr);
  PreLoadEnd();

  PetscFinalize();
  return 0;
}

/* ------------------------------------------------------------------- */

/*
   Define macros to allow us to easily access the components of the PDE
   solution and nonlinear residual vectors.
      Note: the "4" below is a hardcoding of "user.mc" 
*/
#define U(i)     4*(i)
#define V(i)     4*(i)+1
#define Omega(i) 4*(i)+2
#define Temp(i)  4*(i)+3

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
int FormInitialGuess(SNES snes,Vec X,void *ptr)
{
  DMMG    dmmg = (DMMG)ptr;
  AppCtx  *user = (AppCtx*)dmmg->user;
  DA      da = (DA)dmmg->dm;
  int     i,j,row,mx,ierr,xs,ys,xm,ym,gxm,gym,gxs,gys;
  double  grashof,dx;
  Scalar  *x;
  Vec     localX;
  
  ierr = DAGetLocalVector((DA)dmmg->dm,&localX);CHKERRQ(ierr);
  grashof = user->grashof;

  ierr = DAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  dx  = 1.0/(mx-1);

  /*
     Get local grid boundaries (for 2-dimensional DA):
       xs, ys   - starting grid indices (no ghost points)
       xm, ym   - widths of local grid (no ghost points)
       gxs, gys - starting grid indices (including ghost points)
       gxm, gym - widths of local grid (including ghost points)
  */
  ierr = DAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,PETSC_NULL,&gxm,&gym,PETSC_NULL);CHKERRQ(ierr);

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);

  /*
     Compute initial guess over the locally owned part of the grid
     Initial condition is motionless fluid and equilibrium temperature
  */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row = i - gxs + (j - gys)*gxm; 
      x[U(row)]     = 0.0;
      x[V(row)]     = 0.0;
      x[Omega(row)] = 0.0;
      x[Temp(row)]  = (grashof>0)*i*dx;
    }
  }

  /*
     Restore vector
  */
  ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);

  /*
     Insert values into global vector
  */
  ierr = DALocalToGlobal(da,localX,INSERT_VALUES,X);CHKERRQ(ierr);
  ierr = DARestoreLocalVector((DA)dmmg->dm,&localX);CHKERRQ(ierr);
  return 0;
} 
/* ------------------------------------------------------------------- */
#undef __FUNC__
#define __FUNC__ "FormFunction"
/* 
   FormFunction - Evaluates the nonlinear function, F(x).

   Input Parameters:
.  snes - the SNES context
.  X - input vector
.  ptr - optional user-defined context, as set by SNESSetFunction()

   Output Parameter:
.  F - function vector

   Notes:
   We process the boundary nodes before handling the interior
   nodes, so that no conditional statements are needed within the
   double loop over the local grid indices. 
 */
int FormFunction(SNES snes,Vec X,Vec F,void *ptr)
{
  DMMG    dmmg = (DMMG)ptr;
  AppCtx  *user = (AppCtx*)dmmg->user;
  int     ierr,i,j,row,mx,my,xs,ys,xm,ym,gxs,gys,gxm,gym;
  int     xints,xinte,yints,yinte;
  double  two = 2.0,one = 1.0,p5 = 0.5,hx,hy,dhx,dhy,hxdhy,hydhx;
  double  grashof,prandtl,lid;
  Scalar  u,uxx,uyy,vx,vy,avx,avy,vxp,vxm,vyp,vym;
  Scalar  *x,*f;
  Vec     localX,localF; 
  DA      da = (DA)dmmg->dm;

  ierr = DAGetLocalVector((DA)dmmg->dm,&localX);CHKERRQ(ierr);
  ierr = DAGetLocalVector((DA)dmmg->dm,&localF);CHKERRQ(ierr);
  ierr = DAGetInfo(da,0,&mx,&my,0,0,0,0,0,0,0,0);CHKERRQ(ierr);

  grashof = user->grashof;  
  prandtl = user->prandtl;
  lid     = user->lidvelocity;

  /* 
     Define mesh intervals ratios for uniform grid.
     [Note: FD formulae below are normalized by multiplying through by
     local volume element to obtain coefficients O(1) in two dimensions.]
  */
  dhx = (double)(mx-1);     dhy = (double)(my-1);
  hx = one/dhx;             hy = one/dhy;
  hxdhy = hx*dhy;           hydhx = hy*dhx;

  /*
     Scatter ghost points to local vector, using the 2-step process
        DAGlobalToLocalBegin(), DAGlobalToLocalEnd().
     By placing code between these two statements, computations can be
     done while messages are in transition.
  */
  ierr = DAGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);

  /*
     Get pointers to vector data
  */
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);

  /*
     Get local grid boundaries
  */
  ierr = DAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(da,&gxs,&gys,PETSC_NULL,&gxm,&gym,PETSC_NULL);CHKERRQ(ierr);

  /*
     Compute function over the locally owned part of the grid
     (physical corner points are set twice to avoid more conditionals).
  */
  xints = xs; xinte = xs+xm; yints = ys; yinte = ys+ym;

  /* Test whether we are on the bottom edge of the global array */
  if (yints == 0) {
    yints = yints + 1;
    /* bottom edge */
    row = xs - gxs - 1; 
    for (i=xs; i<xs+xm; i++) {
      row++;
        f[U(row)]     = x[U(row)];
        f[V(row)]     = x[V(row)];
        f[Omega(row)] = x[Omega(row)] + (x[U(row+gxm)] - x[U(row)])*dhy; 
	f[Temp(row)]  = x[Temp(row)]-x[Temp(row+gxm)];
    }
  }

  /* Test whether we are on the top edge of the global array */
  if (yinte == my) {
    yinte = yinte - 1;
    /* top edge */
    row = (ys + ym - 1 - gys)*gxm + xs - gxs - 1; 
    for (i=xs; i<xs+xm; i++) {
      row++;
        f[U(row)]     = x[U(row)] - lid;
        f[V(row)]     = x[V(row)];
        f[Omega(row)] = x[Omega(row)] + (x[U(row)] - x[U(row-gxm)])*dhy; 
	f[Temp(row)]  = x[Temp(row)]-x[Temp(row-gxm)];
    }
  }

  /* Test whether we are on the left edge of the global array */
  if (xints == 0) {
    xints = xints + 1;
    /* left edge */
    for (j=ys; j<ys+ym; j++) {
      row = (j - gys)*gxm + xs - gxs; 
      f[U(row)]     = x[U(row)];
      f[V(row)]     = x[V(row)];
      f[Omega(row)] = x[Omega(row)] - (x[V(row+1)] - x[V(row)])*dhx; 
      f[Temp(row)]  = x[Temp(row)];
    }
  }

  /* Test whether we are on the right edge of the global array */
  if (xinte == mx) {
    xinte = xinte - 1;
    /* right edge */ 
    for (j=ys; j<ys+ym; j++) {
      row = (j - gys)*gxm + xs + xm - gxs - 1; 
      f[U(row)]     = x[U(row)];
      f[V(row)]     = x[V(row)];
      f[Omega(row)] = x[Omega(row)] - (x[V(row)] - x[V(row-1)])*dhx; 
      f[Temp(row)]  = x[Temp(row)] - (double)(grashof>0);
    }
  }

  /* Compute over the interior points */
  for (j=yints; j<yinte; j++) {
    row = (j - gys)*gxm + xints - gxs - 1; 
    for (i=xints; i<xinte; i++) {
      row++;

	/*
	  convective coefficients for upwinding
        */
	vx = x[U(row)]; avx = PetscAbsScalar(vx);
        vxp = p5*(vx+avx); vxm = p5*(vx-avx);
	vy = x[V(row)]; avy = PetscAbsScalar(vy);
        vyp = p5*(vy+avy); vym = p5*(vy-avy);

	/* U velocity */
        u          = x[U(row)];
        uxx        = (two*u - x[U(row-1)] - x[U(row+1)])*hydhx;
        uyy        = (two*u - x[U(row-gxm)] - x[U(row+gxm)])*hxdhy;
        f[U(row)]  = uxx + uyy - p5*(x[Omega(row+gxm)]-x[Omega(row-gxm)])*hx;

	/* V velocity */
        u          = x[V(row)];
        uxx        = (two*u - x[V(row-1)] - x[V(row+1)])*hydhx;
        uyy        = (two*u - x[V(row-gxm)] - x[V(row+gxm)])*hxdhy;
        f[V(row)]  = uxx + uyy + p5*(x[Omega(row+1)]-x[Omega(row-1)])*hy;

	/* Omega */
        u          = x[Omega(row)];
        uxx        = (two*u - x[Omega(row-1)] - x[Omega(row+1)])*hydhx;
        uyy        = (two*u - x[Omega(row-gxm)] - x[Omega(row+gxm)])*hxdhy;
	f[Omega(row)] = uxx + uyy + 
			(vxp*(u - x[Omega(row-1)]) +
			  vxm*(x[Omega(row+1)] - u)) * hy +
			(vyp*(u - x[Omega(row-gxm)]) +
			  vym*(x[Omega(row+gxm)] - u)) * hx -
			p5 * grashof * (x[Temp(row+1)] - x[Temp(row-1)]) * hy;

        /* Temperature */
        u             = x[Temp(row)];
        uxx           = (two*u - x[Temp(row-1)] - x[Temp(row+1)])*hydhx;
        uyy           = (two*u - x[Temp(row-gxm)] - x[Temp(row+gxm)])*hxdhy;
	f[Temp(row)] =  uxx + uyy  + prandtl * (
			(vxp*(u - x[Temp(row-1)]) +
			  vxm*(x[Temp(row+1)] - u)) * hy +
		        (vyp*(u - x[Temp(row-gxm)]) +
		       	  vym*(x[Temp(row+gxm)] - u)) * hx);
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
  ierr = DALocalToGlobal(da,localF,INSERT_VALUES,F);CHKERRQ(ierr);
  ierr = DARestoreLocalVector((DA)dmmg->dm,&localX);CHKERRQ(ierr);
  ierr = DARestoreLocalVector((DA)dmmg->dm,&localF);CHKERRQ(ierr);

  /*
     Flop count (multiply-adds are counted as 2 operations)
  */
  ierr = PetscLogFlops(84*ym*xm);CHKERRQ(ierr);

  return 0; 
} 
