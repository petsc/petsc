
static char help[] = "Nonlinear driven cavity with multigrid in 2d.\n\
  \n\
The 2D driven cavity problem is solved in a velocity-vorticity formulation.\n\
The flow can be driven with the lid or with bouyancy or both:\n\
  -lidvelocity <lid>, where <lid> = dimensionless velocity of lid\n\
  -grashof <gr>, where <gr> = dimensionless temperature gradent\n\
  -prandtl <pr>, where <pr> = dimensionless thermal/momentum diffusity ratio\n\
  -contours : draw contour plots of solution\n\n";

/*T
   Concepts: SNES^solving a system of nonlinear equations (parallel multicomponent example);
   Concepts: DA^using distributed arrays;
   Concepts: multicomponent
   Processors: n
T*/

/* ------------------------------------------------------------------------

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
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers 
*/
#include "petscsnes.h"
#include "petscda.h"
#include "petscdmmg.h"
#include "../src/snes/impls/ls/ls.h"
/* 
   User-defined routines and data structures
*/
typedef struct {
  PetscScalar u,v,omega;
} Field;

extern PetscErrorCode FormInitialGuess(DMMG,Vec);
extern PetscErrorCode FormFunctionLocal(DALocalInfo*,Field**,Field**,void*);
extern PetscErrorCode FormFunctionLocali(DALocalInfo*,MatStencil*,Field**,PetscScalar*,void*);
extern PetscErrorCode FormFunctionLocali4(DALocalInfo*,MatStencil*,Field**,PetscScalar*,void*);

typedef struct {
  PassiveReal  lidvelocity,prandtl,grashof,re;  /* physical parameters */
   PetscTruth     draw_contours;                /* flag - 1 indicates drawing contours */
} AppCtx;

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DMMG           *dmmg;               /* multilevel grid structure */
  AppCtx         user;                /* user-defined work context */
  PetscInt       mx,my,its;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  SNES           snes;
  DA             da;

  PetscInitialize(&argc,&argv,(char *)0,help);
  comm = PETSC_COMM_WORLD;


  // in order only run once, I block it: PreLoadBegin(PETSC_TRUE,"SetUp");
    ierr = DMMGCreate(comm,2,&user,&dmmg);CHKERRQ(ierr);


    /*
      Create distributed array multigrid object (DMMG) to manage parallel grid and vectors
      for principal unknowns (x) and governing residuals (f)
    */
    ierr = DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_STAR,-4,-4,PETSC_DECIDE,PETSC_DECIDE,4,1,0,0,&da);CHKERRQ(ierr);
    ierr = DMMGSetDM(dmmg,(DM)da);CHKERRQ(ierr);
    ierr = DADestroy(da);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"mx = %d, my= %d\n",
		       mx,my);CHKERRQ(ierr);

    ierr = DAGetInfo(DMMGGetDA(dmmg),0,&mx,&my,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                     PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"mx = %d, my= %d\n",
		       mx,my);CHKERRQ(ierr);
   /* 
     Problem parameters (velocity of lid, prandtl, and grashof numbers)
    */
    user.lidvelocity = 1.0;///(mx*my);
    user.prandtl     = 1.0;
    user.grashof     = 1.0;
    user.re          = 1.0;
    ierr = PetscOptionsGetReal(PETSC_NULL,"-lidvelocity",&user.lidvelocity,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(PETSC_NULL,"-prandtl",&user.prandtl,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(PETSC_NULL,"-grashof",&user.grashof,PETSC_NULL);CHKERRQ(ierr);    
    ierr = PetscOptionsGetReal(PETSC_NULL,"-re",&user.re,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsHasName(PETSC_NULL,"-contours",&user.draw_contours);CHKERRQ(ierr);

    ierr = DASetFieldName(DMMGGetDA(dmmg),0,"x-velocity");CHKERRQ(ierr);
    ierr = DASetFieldName(DMMGGetDA(dmmg),1,"y-velocity");CHKERRQ(ierr);
    ierr = DASetFieldName(DMMGGetDA(dmmg),2,"Omega");CHKERRQ(ierr);
    //  ierr = DASetFieldName(DMMGGetDA(dmmg),3,"temperature");CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create user context, set problem data, create vector data structures.
       Also, compute the initial guess.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create nonlinear solver context

       Process adiC(36): FormFunctionLocal FormFunctionLocali FormFunctionLocali4
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = DMMGSetSNESLocal(dmmg,FormFunctionLocal,0,ad_FormFunctionLocal,admf_FormFunctionLocal);CHKERRQ(ierr);
    ierr = DMMGSetFromOptions(dmmg);CHKERRQ(ierr);
    ierr = DMMGSetSNESLocali(dmmg,FormFunctionLocali,ad_FormFunctionLocali,admf_FormFunctionLocali);CHKERRQ(ierr);
    ierr = DMMGSetSNESLocalib(dmmg,FormFunctionLocali4,ad_FormFunctionLocali4,admf_FormFunctionLocali4);CHKERRQ(ierr);

    ierr = PetscPrintf(comm,"lid velocity = %g, prandtl # = %g, grashof # = %g, re=%g \n",
		       user.lidvelocity,user.prandtl,user.grashof,user.re);CHKERRQ(ierr);


    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Solve the nonlinear system
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = DMMGSetInitialGuess(dmmg,FormInitialGuess);CHKERRQ(ierr);

    //I block it:  PreLoadStage("Solve");
    ierr = DMMGSolve(dmmg);CHKERRQ(ierr); 

    snes = DMMGGetSNES(dmmg);
    ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
    ierr = PetscPrintf(comm,"Number of Newton iterations = %D\n", its);CHKERRQ(ierr);
     
    /*
      Visualize solution
    */

    if (user.draw_contours) {
      ierr = VecView(DMMGGetx(dmmg),PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr); 
      // ierr = VecView(DMMGGetx(dmmg),PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
    //PreLoadEnd();

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
PetscErrorCode FormInitialGuess(DMMG dmmg,Vec X)
{
  AppCtx         *user = (AppCtx*)dmmg->user;
  DA             da = (DA)dmmg->dm;
  PetscInt       i,j,mx,xs,ys,xm,ym;
  PetscErrorCode ierr;
  PetscReal      grashof,dx;
  Field          **x;

  grashof = user->grashof;

  ierr = DAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  dx  = 1.0/(mx-1);

  /*
     Get local grid boundaries (for 2-dimensional DA):
       xs, ys   - starting grid indices (no ghost points)
       xm, ym   - widths of local grid (no ghost points)
  */
  ierr = DAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr = DAVecGetArray(da,X,&x);CHKERRQ(ierr);

  /*
     Compute initial guess over the locally owned part of the grid
     Initial condition is motionless fluid and equilibrium temperature
  */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      x[j][i].u     = 0.0;
      x[j][i].v     = 0.0;
      x[j][i].omega = 0.0;
      //      x[j][i].temp  = (grashof>0)*i*dx;
    }
  }

  /*
     Restore vector
  */
  ierr = DAVecRestoreArray(da,X,&x);CHKERRQ(ierr);
  return 0;
} 
PetscErrorCode FormFunctionLocal(DALocalInfo *info,Field **x,Field **f,void *ptr)
 {
  AppCtx         *user = (AppCtx*)ptr;
  PetscErrorCode ierr;
  PetscInt       xints,xinte,yints,yinte,i,j;
  PetscReal      hx,hy,dhx,dhy,hxdhy,hydhx;
  PetscReal      grashof,prandtl,lid,re;
  PetscScalar    u,uxx,uyy,vx,vy,avx,avy,vxp,vxm,vyp,vym;

  PetscFunctionBegin;
  grashof = user->grashof;  
  prandtl = user->prandtl;
  lid     = user->lidvelocity;
  re      = user->re;
  /* 
     Define mesh intervals ratios for uniform grid.
     [Note: FD formulae below are normalized by multiplying through by
     local volume element to obtain coefficients O(1) in two dimensions.]
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
      //f[j][i].temp  = x[j][i].temp-x[j+1][i].temp;
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
	//f[j][i].temp  = x[j][i].temp-x[j-1][i].temp;
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
      //f[j][i].temp  = x[j][i].temp;
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
      //f[j][i].temp  = x[j][i].temp - (PetscReal)(grashof>0);
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
        uxx        = (2.0*u - x[j][i-1].u - x[j][i+1].u)*hydhx;
        uyy        = (2.0*u - x[j-1][i].u - x[j+1][i].u)*hxdhy;
        f[j][i].u  = uxx + uyy - .5*(x[j+1][i].omega-x[j-1][i].omega)*hx;

	/* V velocity */
        u          = x[j][i].v;
        uxx        = (2.0*u - x[j][i-1].v - x[j][i+1].v)*hydhx;
        uyy        = (2.0*u - x[j-1][i].v - x[j+1][i].v)*hxdhy;
        f[j][i].v  = uxx + uyy + .5*(x[j][i+1].omega-x[j][i-1].omega)*hy;

	/* Omega */
        u          = x[j][i].omega;
        uxx        = (2.0*u - x[j][i-1].omega - x[j][i+1].omega)*hydhx;
        uyy        = (2.0*u - x[j-1][i].omega - x[j+1][i].omega)*hxdhy;
	f[j][i].omega = (uxx + uyy)/re + 
			(vxp*(u - x[j][i-1].omega) +
			  vxm*(x[j][i+1].omega - u)) * hy +
			(vyp*(u - x[j-1][i].omega) +
			 vym*(x[j+1][i].omega - u)) * hx;// -
	//.5 * grashof * (x[j][i+1].temp - x[j][i-1].temp) * hy;

        /* Temperature */
        /*u             = x[j][i].temp;
        uxx           = (2.0*u - x[j][i-1].temp - x[j][i+1].temp)*hydhx;
        uyy           = (2.0*u - x[j-1][i].temp - x[j+1][i].temp)*hxdhy;
	f[j][i].temp =  uxx + uyy  + prandtl * (
			(vxp*(u - x[j][i-1].temp) +
			  vxm*(x[j][i+1].temp - u)) * hy +
		        (vyp*(u - x[j-1][i].temp) +
			vym*(x[j+1][i].temp - u)) * hx);*/
    }
  }

  /*
     Flop count (multiply-adds are counted as 2 operations)
  */
  ierr = PetscLogFlops(84.0*info->ym*info->xm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 

/*
    This is an experimental function and can be safely ignored.
*/
PetscErrorCode FormFunctionLocali(DALocalInfo *info,MatStencil *st,Field **x,PetscScalar *f,void *ptr)
 {
  AppCtx      *user = (AppCtx*)ptr;
  PetscInt    i,j,c;
  PassiveReal hx,hy,dhx,dhy,hxdhy,hydhx;
  PassiveReal grashof,prandtl,lid,re;
  PetscScalar u,uxx,uyy,vx,vy,avx,avy,vxp,vxm,vyp,vym;

  PetscFunctionBegin;
  grashof = user->grashof;  
  prandtl = user->prandtl;
  lid     = user->lidvelocity;
  re      = user->re;
  /* 
     Define mesh intervals ratios for uniform grid.
     [Note: FD formulae below are normalized by multiplying through by
     local volume element to obtain coefficients O(1) in two dimensions.]
  */
  dhx = (PetscReal)(info->mx-1);     dhy = (PetscReal)(info->my-1);
  hx = 1.0/dhx;                   hy = 1.0/dhy;
  hxdhy = hx*dhy;                 hydhx = hy*dhx;

  i = st->i; j = st->j; c = st->c;

  /* Test whether we are on the right edge of the global array */
  if (i == info->mx-1) {
    if (c == 0)      *f = x[j][i].u;
    else if (c == 1) *f = x[j][i].v;
    else if (c == 2) *f = x[j][i].omega - (x[j][i].v - x[j][i-1].v)*dhx; 
    else             ;//*f = x[j][i].temp - (PetscReal)(grashof>0);

  /* Test whether we are on the left edge of the global array */
  } else if (i == 0) {
    if (c == 0)      *f = x[j][i].u;
    else if (c == 1) *f = x[j][i].v;
    else if (c == 2) *f = x[j][i].omega - (x[j][i+1].v - x[j][i].v)*dhx; 
    else             ;//*f = x[j][i].temp;

  /* Test whether we are on the top edge of the global array */
  } else if (j == info->my-1) {
    if (c == 0)      *f = x[j][i].u - lid;
    else if (c == 1) *f = x[j][i].v;
    else if (c == 2) *f = x[j][i].omega + (x[j][i].u - x[j-1][i].u)*dhy; 
    else             ;//*f = x[j][i].temp-x[j-1][i].temp;

  /* Test whether we are on the bottom edge of the global array */
  } else if (j == 0) {
    if (c == 0)      *f = x[j][i].u;
    else if (c == 1) *f = x[j][i].v;
    else if (c == 2) *f = x[j][i].omega + (x[j+1][i].u - x[j][i].u)*dhy; 
    else             ;//*f = x[j][i].temp - x[j+1][i].temp;

  /* Compute over the interior points */
  } else {
    /*
      convective coefficients for upwinding
    */
    vx = x[j][i].u; avx = PetscAbsScalar(vx);
    vxp = .5*(vx+avx); vxm = .5*(vx-avx);
    vy = x[j][i].v; avy = PetscAbsScalar(vy);
    vyp = .5*(vy+avy); vym = .5*(vy-avy);

    /* U velocity */
    if (c == 0) {
      u          = x[j][i].u;
      uxx        = (2.0*u - x[j][i-1].u - x[j][i+1].u)*hydhx;
      uyy        = (2.0*u - x[j-1][i].u - x[j+1][i].u)*hxdhy;
      *f         = uxx + uyy - .5*(x[j+1][i].omega-x[j-1][i].omega)*hx;

    /* V velocity */
    } else if (c == 1) {
      u          = x[j][i].v;
      uxx        = (2.0*u - x[j][i-1].v - x[j][i+1].v)*hydhx;
      uyy        = (2.0*u - x[j-1][i].v - x[j+1][i].v)*hxdhy;
      *f         = uxx + uyy + .5*(x[j][i+1].omega-x[j][i-1].omega)*hy;
    
    /* Omega */
    } else if (c == 2) {
      u          = x[j][i].omega;
      uxx        = (2.0*u - x[j][i-1].omega - x[j][i+1].omega)*hydhx;
      uyy        = (2.0*u - x[j-1][i].omega - x[j+1][i].omega)*hxdhy;
      *f         = (uxx + uyy)/re + 
	(vxp*(u - x[j][i-1].omega) +
	 vxm*(x[j][i+1].omega - u)) * hy +
	(vyp*(u - x[j-1][i].omega) +
	 vym*(x[j+1][i].omega - u)) * hx;// -
      // .5 * grashof * (x[j][i+1].temp - x[j][i-1].temp) * hy;
    
    /* Temperature */
    } else {
      /*u           = x[j][i].temp;
      uxx         = (2.0*u - x[j][i-1].temp - x[j][i+1].temp)*hydhx;
      uyy         = (2.0*u - x[j-1][i].temp - x[j+1][i].temp)*hxdhy;
      *f          =  uxx + uyy  + prandtl * (
					     (vxp*(u - x[j][i-1].temp) +
					      vxm*(x[j][i+1].temp - u)) * hy +
					     (vyp*(u - x[j-1][i].temp) +
					     vym*(x[j+1][i].temp - u)) * hx);*/
    }
  }

  PetscFunctionReturn(0);
} 

/*
    This is an experimental function and can be safely ignored.
*/
PetscErrorCode FormFunctionLocali4(DALocalInfo *info,MatStencil *st,Field **x,PetscScalar *ff,void *ptr)
 {
   Field *f = (Field*)ff;
  AppCtx      *user = (AppCtx*)ptr;
  PetscInt    i,j;
  PassiveReal hx,hy,dhx,dhy,hxdhy,hydhx;
  PassiveReal grashof,prandtl,lid,re;
  PetscScalar u,uxx,uyy,vx,vy,avx,avy,vxp,vxm,vyp,vym;

  PetscFunctionBegin;
  grashof = user->grashof;  
  prandtl = user->prandtl;
  lid     = user->lidvelocity;
  re      = user->re;
  /* 
     Define mesh intervals ratios for uniform grid.
     [Note: FD formulae below are normalized by multiplying through by
     local volume element to obtain coefficients O(1) in two dimensions.]
  */
  dhx = (PetscReal)(info->mx-1);     dhy = (PetscReal)(info->my-1);
  hx = 1.0/dhx;                   hy = 1.0/dhy;
  hxdhy = hx*dhy;                 hydhx = hy*dhx;

  i = st->i; j = st->j; 

  /* Test whether we are on the right edge of the global array */
  if (i == info->mx-1) {
    f->u = x[j][i].u;
    f->v = x[j][i].v;
    f->omega = x[j][i].omega - (x[j][i].v - x[j][i-1].v)*dhx; 
    f->temp = x[j][i].temp - (PetscReal)(grashof>0);

    /* Test whether we are on the left edge of the global array */
  } else if (i == 0) {
    f->u = x[j][i].u;
    f->v = x[j][i].v;
    f->omega = x[j][i].omega - (x[j][i+1].v - x[j][i].v)*dhx; 
    f->temp = x[j][i].temp;
      
    /* Test whether we are on the top edge of the global array */
  } else if (j == info->my-1) {
    f->u = x[j][i].u - lid;
    f->v = x[j][i].v;
    f->omega = x[j][i].omega + (x[j][i].u - x[j-1][i].u)*dhy; 
    f->temp = x[j][i].temp-x[j-1][i].temp;

    /* Test whether we are on the bottom edge of the global array */
  } else if (j == 0) {
    f->u = x[j][i].u;
    f->v = x[j][i].v;
    f->omega = x[j][i].omega + (x[j+1][i].u - x[j][i].u)*dhy; 
    f->temp = x[j][i].temp - x[j+1][i].temp;

    /* Compute over the interior points */
  } else {
    /*
      convective coefficients for upwinding
    */
    vx = x[j][i].u; avx = PetscAbsScalar(vx);
    vxp = .5*(vx+avx); vxm = .5*(vx-avx);
    vy = x[j][i].v; avy = PetscAbsScalar(vy);
    vyp = .5*(vy+avy); vym = .5*(vy-avy);

    /* U velocity */
    u          = x[j][i].u;
    uxx        = (2.0*u - x[j][i-1].u - x[j][i+1].u)*hydhx;
    uyy        = (2.0*u - x[j-1][i].u - x[j+1][i].u)*hxdhy;
    f->u        = uxx + uyy - .5*(x[j+1][i].omega-x[j-1][i].omega)*hx;

    /* V velocity */
    u          = x[j][i].v;
    uxx        = (2.0*u - x[j][i-1].v - x[j][i+1].v)*hydhx;
    uyy        = (2.0*u - x[j-1][i].v - x[j+1][i].v)*hxdhy;
    f->v        = uxx + uyy + .5*(x[j][i+1].omega-x[j][i-1].omega)*hy;
    
    /* Omega */
    u          = x[j][i].omega;
    uxx        = (2.0*u - x[j][i-1].omega - x[j][i+1].omega)*hydhx;
    uyy        = (2.0*u - x[j-1][i].omega - x[j+1][i].omega)*hxdhy;
    f->omega    = uxx + uyy +
                    (vx*(u - x[j][i-1].omega)) * hy +
                    (vy*(u - x[j-1][i].omega)) * hx ;-
                         .5 * grashof * (x[j][i+1].temp - x[j][i-1].temp) * hy;

    u           = x[j][i].temp;
    uxx         = (2.0*u - x[j][i-1].temp - x[j][i+1].temp)*hydhx;
    uyy         = (2.0*u - x[j-1][i].temp - x[j+1][i].temp)*hxdhy;
    f->temp     =  uxx + uyy  + prandtl * (
                                           (vxp*(u - x[j][i-1].temp) +
                                            vxm*(x[j][i+1].temp - u)) * hy +
                                           (vyp*(u - x[j-1][i].temp) +
                                           vym*(x[j+1][i].temp - u)) * hx);  
  }

  PetscFunctionReturn(0);
} 
