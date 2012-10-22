
static char help[] = "Transient nonlinear driven cavity in 2d.\n\
  \n\
The 2D driven cavity problem is solved in a velocity-vorticity formulation.\n\
The flow can be driven with the lid or with bouyancy or both:\n\
  -lidvelocity <lid>, where <lid> = dimensionless velocity of lid\n\
  -grashof <gr>, where <gr> = dimensionless temperature gradent\n\
  -prandtl <pr>, where <pr> = dimensionless thermal/momentum diffusity ratio\n\
  -contours : draw contour plots of solution\n\n";
/*
      See src/snes/examples/tutorials/ex50.c for the steady-state version.

      See src/snes/examples/tutorials/ex27.c for a version written specifically
      for pseudo-transient continuation, without using TS.
*/

/*T
   Concepts: TS^solving a system of nonlinear equations (parallel multicomponent example);
   Concepts: DMDA^using distributed arrays;
   Concepts: multicomponent;differential-algebraic equation
   Processors: n
T*/
/* ------------------------------------------------------------------------

    We thank David E. Keyes for contributing the driven cavity discretization
    within this example code.

    This problem is modeled by the partial differential equation system

	- Lap(U) - Grad_y(Omega) = 0
	- Lap(V) + Grad_x(Omega) = 0
	Omega_t - Lap(Omega) + Div([U*Omega,V*Omega]) - GR*Grad_x(T) = 0
	T_t - Lap(T) + PR*Div([U*T,V*T]) = 0

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
   Include "petscdmda.h" so that we can use distributed arrays (DMDAs).
   Include "petscts.h" so that we can use TS solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers         petscsnes.h - nonlinear solvers
*/
#include <petscts.h>
#include <petscdmda.h>

/*
   User-defined routines and data structures
*/
typedef struct {
  PetscScalar u,v,omega,temp;
} Field;

PetscErrorCode FormIFunctionLocal(DMDALocalInfo*,PetscReal,Field**,Field**,Field**,void*);

typedef struct {
  PassiveReal  lidvelocity,prandtl,grashof;  /* physical parameters */
  PetscBool    parabolic;                    /* allow a transient term corresponding roughly to artificial compressibility */
  PetscReal    cfl_initial;                  /* CFL for first time step */
} AppCtx;

PetscErrorCode FormInitialSolution(TS,Vec,AppCtx*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  AppCtx         user;                /* user-defined work context */
  PetscInt       mx,my,steps;
  PetscErrorCode ierr;
  TS             ts;
  DM             da;
  Vec            X;
  PetscReal      ftime;
  TSConvergedReason reason;

  ierr = PetscInitialize(&argc,&argv,(char *)0,help);if (ierr) return(1);

  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);

  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR,-4,-4,PETSC_DECIDE,PETSC_DECIDE,4,1,0,0,&da);CHKERRQ(ierr);
  ierr = TSSetDM(ts,(DM)da);CHKERRQ(ierr);

  ierr = DMDAGetInfo(da,0,&mx,&my,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
		   PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  /*
     Problem parameters (velocity of lid, prandtl, and grashof numbers)
  */
  user.lidvelocity   = 1.0/(mx*my);
  user.prandtl       = 1.0;
  user.grashof       = 1.0;
  user.parabolic     = PETSC_FALSE;
  user.cfl_initial   = 50.;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"Driven cavity/natural convection options","");CHKERRQ(ierr);
  ierr = PetscOptionsReal("-lidvelocity","Lid velocity, related to Reynolds number","",user.lidvelocity,&user.lidvelocity,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-prandtl","Ratio of viscous to thermal diffusivity","",user.prandtl,&user.prandtl,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-grashof","Ratio of bouyant to viscous forces","",user.grashof,&user.grashof,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-parabolic","Relax incompressibility to make the system parabolic instead of differential-algebraic","",user.parabolic,&user.parabolic,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-cfl_initial","Advective CFL for the first time step","",user.cfl_initial,&user.cfl_initial,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = DMDASetFieldName(da,0,"x-velocity");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,1,"y-velocity");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,2,"Omega");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,3,"temperature");CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create user context, set problem data, create vector data structures.
     Also, compute the initial guess.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create time integration context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);
  ierr = DMDATSSetIFunctionLocal(da,INSERT_VALUES,(DMDATSIFunctionLocal)FormIFunctionLocal,&user);CHKERRQ(ierr);
  ierr = TSSetDuration(ts,10000,1e12);CHKERRQ(ierr);
  ierr = TSSetInitialTimeStep(ts,0.0,user.cfl_initial/(user.lidvelocity*mx));CHKERRQ(ierr);
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"%Dx%D grid, lid velocity = %G, prandtl # = %G, grashof # = %G\n",mx,my,user.lidvelocity,user.prandtl,user.grashof);CHKERRQ(ierr);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve the nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  ierr = DMCreateGlobalVector(da,&X);CHKERRQ(ierr);
  ierr = FormInitialSolution(ts,X,&user);CHKERRQ(ierr);

  ierr = TSSolve(ts,X,&ftime);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts,&reason);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s at time %G after %D steps\n",TSConvergedReasons[reason],ftime,steps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}

/* ------------------------------------------------------------------- */


#undef __FUNCT__
#define __FUNCT__ "FormInitialSolution"
/*
   FormInitialSolution - Forms initial approximation.

   Input Parameters:
   user - user-defined application context
   X - vector

   Output Parameter:
   X - vector
 */
PetscErrorCode FormInitialSolution(TS ts,Vec X,AppCtx *user)
{
  DM             da;
  PetscInt       i,j,mx,xs,ys,xm,ym;
  PetscErrorCode ierr;
  PetscReal      grashof,dx;
  Field          **x;

  grashof = user->grashof;
  ierr = TSGetDM(ts,&da);CHKERRQ(ierr);
  ierr = DMDAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  dx  = 1.0/(mx-1);

  /*
     Get local grid boundaries (for 2-dimensional DMDA):
       xs, ys   - starting grid indices (no ghost points)
       xm, ym   - widths of local grid (no ghost points)
  */
  ierr = DMDAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);

  /*
     Compute initial guess over the locally owned part of the grid
     Initial condition is motionless fluid and equilibrium temperature
  */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      x[j][i].u     = 0.0;
      x[j][i].v     = 0.0;
      x[j][i].omega = 0.0;
      x[j][i].temp  = (grashof>0)*i*dx;
    }
  }

  /*
     Restore vector
  */
  ierr = DMDAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormIFunctionLocal"
PetscErrorCode FormIFunctionLocal(DMDALocalInfo *info,PetscReal ptime,Field **x,Field **xdot,Field **f,void *ptr)
 {
  AppCtx         *user = (AppCtx*)ptr;
  PetscErrorCode ierr;
  PetscInt       xints,xinte,yints,yinte,i,j;
  PetscReal      hx,hy,dhx,dhy,hxdhy,hydhx;
  PetscReal      grashof,prandtl,lid;
  PetscScalar    u,udot,uxx,uyy,vx,vy,avx,avy,vxp,vxm,vyp,vym;

  PetscFunctionBegin;
  grashof = user->grashof;
  prandtl = user->prandtl;
  lid     = user->lidvelocity;

  /*
     Define mesh intervals ratios for uniform grid.

     Note: FD formulae below are normalized by multiplying through by
     local volume element (i.e. hx*hy) to obtain coefficients O(1) in two dimensions.


  */
  dhx = (PetscReal)(info->mx-1);  dhy = (PetscReal)(info->my-1);
  hx = 1.0/dhx;                   hy = 1.0/dhy;
  hxdhy = hx*dhy;                 hydhx = hy*dhx;

  xints = info->xs; xinte = info->xs+info->xm; yints = info->ys; yinte = info->ys+info->ym;

  /* Test whether we are on the bottom edge of the global array */
  if (yints == 0) {
    j = 0;
    yints = yints + 1;
    /* bottom edge */
    for (i=info->xs; i<info->xs+info->xm; i++) {
      f[j][i].u     = x[j][i].u;
      f[j][i].v     = x[j][i].v;
      f[j][i].omega = x[j][i].omega + (x[j+1][i].u - x[j][i].u)*dhy;
      f[j][i].temp  = x[j][i].temp-x[j+1][i].temp;
    }
  }

  /* Test whether we are on the top edge of the global array */
  if (yinte == info->my) {
    j = info->my - 1;
    yinte = yinte - 1;
    /* top edge */
    for (i=info->xs; i<info->xs+info->xm; i++) {
        f[j][i].u     = x[j][i].u - lid;
        f[j][i].v     = x[j][i].v;
        f[j][i].omega = x[j][i].omega + (x[j][i].u - x[j-1][i].u)*dhy;
	f[j][i].temp  = x[j][i].temp-x[j-1][i].temp;
    }
  }

  /* Test whether we are on the left edge of the global array */
  if (xints == 0) {
    i = 0;
    xints = xints + 1;
    /* left edge */
    for (j=info->ys; j<info->ys+info->ym; j++) {
      f[j][i].u     = x[j][i].u;
      f[j][i].v     = x[j][i].v;
      f[j][i].omega = x[j][i].omega - (x[j][i+1].v - x[j][i].v)*dhx;
      f[j][i].temp  = x[j][i].temp;
    }
  }

  /* Test whether we are on the right edge of the global array */
  if (xinte == info->mx) {
    i = info->mx - 1;
    xinte = xinte - 1;
    /* right edge */
    for (j=info->ys; j<info->ys+info->ym; j++) {
      f[j][i].u     = x[j][i].u;
      f[j][i].v     = x[j][i].v;
      f[j][i].omega = x[j][i].omega - (x[j][i].v - x[j][i-1].v)*dhx;
      f[j][i].temp  = x[j][i].temp - (PetscReal)(grashof>0);
    }
  }

  /* Compute over the interior points */
  for (j=yints; j<yinte; j++) {
    for (i=xints; i<xinte; i++) {

	/*
	  convective coefficients for upwinding
        */
	vx = x[j][i].u; avx = PetscAbsScalar(vx);
        vxp = .5*(vx+avx); vxm = .5*(vx-avx);
	vy = x[j][i].v; avy = PetscAbsScalar(vy);
        vyp = .5*(vy+avy); vym = .5*(vy-avy);

	/* U velocity */
        u          = x[j][i].u;
        udot       = user->parabolic ? xdot[j][i].u : 0.;
        uxx        = (2.0*u - x[j][i-1].u - x[j][i+1].u)*hydhx;
        uyy        = (2.0*u - x[j-1][i].u - x[j+1][i].u)*hxdhy;
        f[j][i].u  = udot + uxx + uyy - .5*(x[j+1][i].omega-x[j-1][i].omega)*hx;

	/* V velocity */
        u          = x[j][i].v;
        udot       = user->parabolic ? xdot[j][i].v : 0.;
        uxx        = (2.0*u - x[j][i-1].v - x[j][i+1].v)*hydhx;
        uyy        = (2.0*u - x[j-1][i].v - x[j+1][i].v)*hxdhy;
        f[j][i].v  = udot + uxx + uyy + .5*(x[j][i+1].omega-x[j][i-1].omega)*hy;

	/* Omega */
        u          = x[j][i].omega;
        uxx        = (2.0*u - x[j][i-1].omega - x[j][i+1].omega)*hydhx;
        uyy        = (2.0*u - x[j-1][i].omega - x[j+1][i].omega)*hxdhy;
	f[j][i].omega = (xdot[j][i].omega + uxx + uyy
                         + (vxp*(u - x[j][i-1].omega)
                            + vxm*(x[j][i+1].omega - u)) * hy
                         + (vyp*(u - x[j-1][i].omega)
                            + vym*(x[j+1][i].omega - u)) * hx
                         - .5 * grashof * (x[j][i+1].temp - x[j][i-1].temp) * hy);

        /* Temperature */
        u             = x[j][i].temp;
        uxx           = (2.0*u - x[j][i-1].temp - x[j][i+1].temp)*hydhx;
        uyy           = (2.0*u - x[j-1][i].temp - x[j+1][i].temp)*hxdhy;
	f[j][i].temp =  (xdot[j][i].temp + uxx + uyy
                         + prandtl * ((vxp*(u - x[j][i-1].temp)
                                       + vxm*(x[j][i+1].temp - u)) * hy
                                      + (vyp*(u - x[j-1][i].temp)
                                         + vym*(x[j+1][i].temp - u)) * hx));
    }
  }

  /*
     Flop count (multiply-adds are counted as 2 operations)
  */
  ierr = PetscLogFlops(84.0*info->ym*info->xm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
