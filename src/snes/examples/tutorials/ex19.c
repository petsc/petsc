
static char help[] = "Nonlinear driven cavity with multigrid in 2d.\n\
  \n\
The 2D driven cavity problem is solved in a velocity-vorticity formulation.\n\
The flow can be driven with the lid or with bouyancy or both:\n\
  -lidvelocity <lid>, where <lid> = dimensionless velocity of lid\n\
  -grashof <gr>, where <gr> = dimensionless temperature gradent\n\
  -prandtl <pr>, where <pr> = dimensionless thermal/momentum diffusity ratio\n\
  -contours : draw contour plots of solution\n\n";
/*
      See src/ksp/ksp/examples/tutorials/ex45.c
*/

/*T
   Concepts: SNES^solving a system of nonlinear equations (parallel multicomponent example);
   Concepts: DMDA^using distributed arrays;
   Concepts: multicomponent
   Processors: n
T*/
/*F-----------------------------------------------------------------------

    We thank David E. Keyes for contributing the driven cavity discretization within this example code.

    This problem is modeled by the partial differential equation system

\begin{eqnarray}
        - \triangle U - \nabla_y \Omega & = & 0  \\
        - \triangle V + \nabla_x\Omega & = & 0  \\
        - \triangle \Omega + \nabla \cdot ([U*\Omega,V*\Omega]) - GR* \nabla_x T & = & 0  \\
        - \triangle T + PR* \nabla \cdot ([U*T,V*T]) & = & 0
\end{eqnarray}

    in the unit square, which is uniformly discretized in each of x and y in this simple encoding.

    No-slip, rigid-wall Dirichlet conditions are used for $ [U,V]$.
    Dirichlet conditions are used for Omega, based on the definition of
    vorticity: $ \Omega = - \nabla_y U + \nabla_x V$, where along each
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

  ------------------------------------------------------------------------F*/

/*
   Include "petscdmda.h" so that we can use distributed arrays (DMDAs).
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petscsys.h       - base PETSc routines   petscvec.h - vectors
     petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers
*/
#if defined(PETSC_APPLE_FRAMEWORK)
#import <PETSc/petscsnes.h>
#import <PETSc/petscdmda.h>
#else
#include <petscsnes.h>
#include <petscdmda.h>
#endif

/*
   User-defined routines and data structures
*/
typedef struct {
  PetscScalar u,v,omega,temp;
} Field;

PetscErrorCode FormFunctionLocal(DMDALocalInfo*,Field**,Field**,void*);
PetscErrorCode FormFunction(SNES,Vec,Vec,void*);

typedef struct {
  PassiveReal lidvelocity,prandtl,grashof;  /* physical parameters */
  PetscBool   draw_contours;                /* flag - 1 indicates drawing contours */
} AppCtx;

extern PetscErrorCode FormInitialGuess(AppCtx*,DM,Vec);
extern PetscErrorCode NonlinearGS(SNES,Vec,Vec,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  AppCtx         user;                /* user-defined work context */
  PetscInt       mx,my,its;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  SNES           snes;
  DM             da;
  Vec            x;

  ierr = PetscInitialize(&argc,&argv,(char *)0,help);if (ierr) return(1);
  comm = PETSC_COMM_WORLD;

  ierr = SNESCreate(comm,&snes);CHKERRQ(ierr);

  /*
      Create distributed array object to manage parallel grid and vectors
      for principal unknowns (x) and governing residuals (f)
  */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR,-4,-4,PETSC_DECIDE,PETSC_DECIDE,4,1,0,0,&da);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,(DM)da);CHKERRQ(ierr);
  ierr = SNESSetGS(snes, NonlinearGS, (void *)&user);CHKERRQ(ierr);

  ierr = DMDAGetInfo(da,0,&mx,&my,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                     PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  /*
     Problem parameters (velocity of lid, prandtl, and grashof numbers)
  */
  user.lidvelocity = 1.0/(mx*my);
  user.prandtl     = 1.0;
  user.grashof     = 1.0;
  ierr = PetscOptionsGetReal(PETSC_NULL,"-lidvelocity",&user.lidvelocity,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-prandtl",&user.prandtl,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-grashof",&user.grashof,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-contours",&user.draw_contours);CHKERRQ(ierr);

  ierr = DMDASetFieldName(da,0,"x_velocity");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,1,"y_velocity");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,2,"Omega");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da,3,"temperature");CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create user context, set problem data, create vector data structures.
     Also, compute the initial guess.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create nonlinear solver context

     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMSetApplicationContext(da,&user);CHKERRQ(ierr);
  ierr = DMDASetLocalFunction(da,(DMDALocalFunction1)FormFunctionLocal);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"lid velocity = %G, prandtl # = %G, grashof # = %G\n",user.lidvelocity,user.prandtl,user.grashof);CHKERRQ(ierr);


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve the nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = FormInitialGuess(&user,da,x);CHKERRQ(ierr);

  ierr = SNESSolve(snes,PETSC_NULL,x);CHKERRQ(ierr);

  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"Number of SNES iterations = %D\n", its);CHKERRQ(ierr);

  /*
     Visualize solution
  */
  if (user.draw_contours) {
    ierr = VecView(x,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
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
PetscErrorCode FormInitialGuess(AppCtx *user,DM da,Vec X)
{
  PetscInt       i,j,mx,xs,ys,xm,ym;
  PetscErrorCode ierr;
  PetscReal      grashof,dx;
  Field          **x;

  grashof = user->grashof;

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
#define __FUNCT__ "FormFunctionLocal"
PetscErrorCode FormFunctionLocal(DMDALocalInfo *info,Field **x,Field **f,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  PetscErrorCode ierr;
  PetscInt       xints,xinte,yints,yinte,i,j;
  PetscReal      hx,hy,dhx,dhy,hxdhy,hydhx;
  PetscReal      grashof,prandtl,lid;
  PetscScalar    u,uxx,uyy,vx,vy,avx,avy,vxp,vxm,vyp,vym;

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
      f[j][i].omega = uxx + uyy + (vxp*(u - x[j][i-1].omega) + vxm*(x[j][i+1].omega - u))*hy +
                      (vyp*(u - x[j-1][i].omega) + vym*(x[j+1][i].omega - u))*hx -
                      .5*grashof*(x[j][i+1].temp - x[j][i-1].temp)*hy;

      /* Temperature */
      u             = x[j][i].temp;
      uxx           = (2.0*u - x[j][i-1].temp - x[j][i+1].temp)*hydhx;
      uyy           = (2.0*u - x[j-1][i].temp - x[j+1][i].temp)*hxdhy;
      f[j][i].temp =  uxx + uyy  + prandtl*((vxp*(u - x[j][i-1].temp) + vxm*(x[j][i+1].temp - u))*hy +
                                            (vyp*(u - x[j-1][i].temp) + vym*(x[j+1][i].temp - u))*hx);
    }
  }

  /*
     Flop count (multiply-adds are counted as 2 operations)
  */
  ierr = PetscLogFlops(84.0*info->ym*info->xm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "FormFunction"
PetscErrorCode FormFunction(SNES snes, Vec X, Vec F, void *user)
{
  DMDALocalInfo  info;
  Field          **u,**fu;
  PetscErrorCode ierr;
  Vec            localX;
  DM             da;

  PetscFunctionBegin;
  ierr = SNESGetDM(snes,(DM*)&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);
  /*
     Scatter ghost points to local vector, using the 2-step process
        DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
  */
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,localX,&u);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,F,&fu);CHKERRQ(ierr);
  ierr = FormFunctionLocal(&info,u,fu,user);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,localX,&u);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da,F,&fu);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "NonlinearGS"
PetscErrorCode NonlinearGS(SNES snes, Vec X, Vec B, void *ctx)
{
  DMDALocalInfo  info;
  Field          **x,**b;
  PetscErrorCode ierr;
  Vec            localX, localB;
  DM             da;
  PetscInt       xints,xinte,yints,yinte,i,j,k,l;
  PetscInt       n_pointwise = 50;
  PetscInt       n_sweeps = 3;
  PetscReal      hx,hy,dhx,dhy,hxdhy,hydhx;
  PetscReal      grashof,prandtl,lid;
  PetscScalar    u,uxx,uyy,vx,vy,avx,avy,vxp,vxm,vyp,vym;
  PetscScalar    fu, fv, fomega, ftemp;
  PetscScalar    dfudu;
  PetscScalar    dfvdv;
  PetscScalar    dfodu, dfodv, dfodo;
  PetscScalar    dftdu, dftdv, dftdt;
  PetscScalar    yu, yv, yo, yt;
  PetscScalar    bjiu, bjiv, bjiomega, bjitemp;
  PetscBool      ptconverged;
  PetscScalar    pfnorm, pfnorm0;
  AppCtx         *user = (AppCtx*)ctx;

  PetscFunctionBegin;
  grashof = user->grashof;
  prandtl = user->prandtl;
  lid     = user->lidvelocity;
  ierr = SNESGetDM(snes,(DM*)&da);CHKERRQ(ierr);
  ierr = DMGetLocalVector(da,&localX);CHKERRQ(ierr);
  if (B) {
    ierr = DMGetLocalVector(da,&localB);CHKERRQ(ierr);
  }
  /*
     Scatter ghost points to local vector, using the 2-step process
        DMGlobalToLocalBegin(), DMGlobalToLocalEnd().
  */
  ierr = DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  if (B) {
    ierr = DMGlobalToLocalBegin(da,B,INSERT_VALUES,localB);CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da,B,INSERT_VALUES,localB);CHKERRQ(ierr);
  }
  ierr = DMDAGetLocalInfo(da,&info);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,localX,&x);CHKERRQ(ierr);
  if (B) {
    ierr = DMDAVecGetArray(da,localB,&b);CHKERRQ(ierr);
  }
  /* looks like a combination of the formfunction / formjacobian routines */
  dhx   = (PetscReal)(info.mx-1);dhy   = (PetscReal)(info.my-1);
  hx    = 1.0/dhx;               hy    = 1.0/dhy;
  hxdhy = hx*dhy;                hydhx = hy*dhx;

  xints = info.xs; xinte = info.xs+info.xm; yints = info.ys; yinte = info.ys+info.ym;

  /* Set the boundary conditions on the momentum equations */
  /* Test whether we are on the bottom edge of the global array */
  if (yints == 0) {
    j = 0;
    yints = yints + 1;
    /* bottom edge */
    for (i=info.xs; i<info.xs+info.xm; i++) {

      if (B) {
        bjiu = b[j][i].u;
        bjiv = b[j][i].v;
      } else {
        bjiu = 0.0;
        bjiv = 0.0;
      }
      x[j][i].u     = 0.0 + bjiu;
      x[j][i].v     = 0.0 + bjiv;
    }
  }

  /* Test whether we are on the top edge of the global array */
  if (yinte == info.my) {
    j = info.my - 1;
    yinte = yinte - 1;
    /* top edge */
    for (i=info.xs; i<info.xs+info.xm; i++) {
      if (B) {
        bjiu = b[j][i].u;
        bjiv = b[j][i].v;
      } else {
        bjiu = 0.0;
        bjiv = 0.0;
      }
      x[j][i].u     = lid + bjiu;
      x[j][i].v     = bjiv;
    }
  }

  /* Test whether we are on the left edge of the global array */
  if (xints == 0) {
    i = 0;
    xints = xints + 1;
    /* left edge */
    for (j=info.ys; j<info.ys+info.ym; j++) {
      if (B) {
        bjiu = b[j][i].u;
        bjiv = b[j][i].v;
      } else {
        bjiu = 0.0;
        bjiv = 0.0;
      }
      x[j][i].u     = 0.0 + bjiu;
      x[j][i].v     = 0.0 + bjiv;
    }
  }

  /* Test whether we are on the right edge of the global array */
  if (xinte == info.mx) {
    i = info.mx - 1;
    xinte = xinte - 1;
    /* right edge */
    for (j=info.ys; j<info.ys+info.ym; j++) {
      if (B) {
        bjiu = b[j][i].u;
        bjiv = b[j][i].v;
      } else {
        bjiu = 0.0;
        bjiv = 0.0;
      }
      x[j][i].u     = 0.0 + bjiu;
      x[j][i].v     = 0.0 + bjiv;
    }
  }

  for (k=0; k < n_sweeps; k++) {
    for (j=info.ys; j<info.ys + info.ym; j++) {
      for (i=info.xs; i<info.xs + info.xm; i++) {
        ptconverged = PETSC_FALSE;
        pfnorm0 = 0.0;
        pfnorm = 0.0;
        fu = 0.0;
        fv = 0.0;
        fomega = 0.0;
        ftemp = 0.0;
        for (l = 0; l < n_pointwise && !ptconverged; l++) {
          if (B) {
            bjiu = b[j][i].u;
            bjiv = b[j][i].v;
            bjiomega = b[j][i].omega;
            bjitemp = b[j][i].temp;
          } else {
            bjiu = 0.0;
            bjiv = 0.0;
            bjiomega = 0.0;
            bjitemp = 0.0;
          }

          if (i != 0 && i != info.mx - 1 && j != 0 && j != info.my-1) {
            /* U velocity */
            u          = x[j][i].u;
            uxx        = (2.0*u - x[j][i-1].u - x[j][i+1].u)*hydhx;
            uyy        = (2.0*u - x[j-1][i].u - x[j+1][i].u)*hxdhy;
            fu    = uxx + uyy - .5*(x[j+1][i].omega-x[j-1][i].omega)*hx - bjiu;
            dfudu = 2.0*(hydhx + hxdhy);
            /* V velocity */
            u          = x[j][i].v;
            uxx        = (2.0*u - x[j][i-1].v - x[j][i+1].v)*hydhx;
            uyy        = (2.0*u - x[j-1][i].v - x[j+1][i].v)*hxdhy;
            fv    = uxx + uyy + .5*(x[j][i+1].omega-x[j][i-1].omega)*hy - bjiv;
            dfvdv = 2.0*(hydhx + hxdhy);
            /*
             convective coefficients for upwinding
             */
            vx = x[j][i].u; avx = PetscAbsScalar(vx);
            vxp = .5*(vx+avx); vxm = .5*(vx-avx);
            vy = x[j][i].v; avy = PetscAbsScalar(vy);
            vyp = .5*(vy+avy); vym = .5*(vy-avy);
            /* Omega */
            u          = x[j][i].omega;
            uxx        = (2.0*u - x[j][i-1].omega - x[j][i+1].omega)*hydhx;
            uyy        = (2.0*u - x[j-1][i].omega - x[j+1][i].omega)*hxdhy;
            fomega = uxx + uyy +  (vxp*(u - x[j][i-1].omega) + vxm*(x[j][i+1].omega - u))*hy +
                     (vyp*(u - x[j-1][i].omega) + vym*(x[j+1][i].omega - u))*hx -
                     .5*grashof*(x[j][i+1].temp - x[j][i-1].temp)*hy - bjiomega;
            /* convective coefficient derivatives */
            dfodo = 2.0*(hydhx + hxdhy) + ((vxp - vxm)*hy + (vyp - vym)*hx);
            if (PetscRealPart(vx) > 0.0) {
              dfodu = (u - x[j][i-1].omega)*hy;
            } else {
              dfodu = (x[j][i+1].omega - u)*hy;
            }
            if (PetscRealPart(vy) > 0.0) {
              dfodv = (u - x[j-1][i].omega)*hx;
            } else {
              dfodv = (x[j+1][i].omega - u)*hx;
            }
            /* Temperature */
            u             = x[j][i].temp;
            uxx           = (2.0*u - x[j][i-1].temp - x[j][i+1].temp)*hydhx;
            uyy           = (2.0*u - x[j-1][i].temp - x[j+1][i].temp)*hxdhy;
            ftemp =  uxx + uyy  + prandtl*((vxp*(u - x[j][i-1].temp) + vxm*(x[j][i+1].temp - u))*hy +
                                           (vyp*(u - x[j-1][i].temp) + vym*(x[j+1][i].temp - u))*hx) - bjitemp;
            dftdt = 2.0*(hydhx + hxdhy) + prandtl*((vxp - vxm)*hy + (vyp - vym)*hx);
            if (PetscRealPart(vx) > 0.0) {
              dftdu = prandtl*(u - x[j][i-1].temp)*hy;
            } else {
              dftdu = prandtl*(x[j][i+1].temp - u)*hy;
            }
            if (PetscRealPart(vy) > 0.0) {
              dftdv = prandtl*(u - x[j-1][i].temp)*hx;
            } else {
              dftdv = prandtl*(x[j+1][i].temp - u)*hx;
            }
            /* invert the system:
             [ dfu / du     0        0        0    ][yu] = [fu]
             [     0    dfv / dv     0        0    ][yv]   [fv]
             [ dfo / du dfo / dv dfo / do     0    ][yo]   [fo]
             [ dft / du dft / dv     0    dft / dt ][yt]   [ft]
             by simple back-substitution
           */
            yu = fu / dfudu;
            yv = fv / dfvdv;
            yo = fomega / dfodo;
            yt = ftemp / dftdt;
            yo = (fomega - (dfodu*yu + dfodv*yv)) / dfodo;
            yt = (ftemp - (dftdu*yu + dftdv*yv)) / dftdt;

            x[j][i].u = x[j][i].u - yu;
            x[j][i].v = x[j][i].v - yv;
            x[j][i].temp  = x[j][i].temp - yt;
            x[j][i].omega = x[j][i].omega - yo;
          }
          if (i == 0) {
            fomega = x[j][i].omega - (x[j][i+1].v - x[j][i].v)*dhx - bjiomega;
            ftemp  = x[j][i].temp - bjitemp;
            x[j][i].omega = x[j][i].omega - fomega;
            x[j][i].temp  = x[j][i].temp - ftemp;
          }
          if (i == info.mx - 1) {
            fomega = x[j][i].omega - (x[j][i].v - x[j][i-1].v)*dhx - bjiomega;
            ftemp  = x[j][i].temp - (PetscReal)(grashof>0) - bjitemp;
            x[j][i].omega = x[j][i].omega - fomega;
            x[j][i].temp  = x[j][i].temp - ftemp;
          }
          if (j == 0) {
            fomega = x[j][i].omega + (x[j+1][i].u - x[j][i].u)*dhy - bjiomega;
            ftemp  = x[j][i].temp-x[j+1][i].temp - bjitemp;
            x[j][i].omega = x[j][i].omega - fomega;
            x[j][i].temp  = x[j][i].temp - ftemp;
          }
          if (j == info.my - 1) {
            fomega = x[j][i].omega + (x[j][i].u - x[j-1][i].u)*dhy - bjiomega;
            ftemp  = x[j][i].temp-x[j-1][i].temp - bjitemp;
            x[j][i].omega = x[j][i].omega - fomega;
            x[j][i].temp  = x[j][i].temp - ftemp;
          }
          pfnorm = fu*fu + fv*fv + fomega*fomega + ftemp*ftemp;
          pfnorm = PetscSqrtScalar(pfnorm);
          if (l == 0) pfnorm0 = pfnorm;
          if (1e-15*PetscRealPart(pfnorm0) > PetscRealPart(pfnorm)) ptconverged = PETSC_TRUE;
        }
      }
    }
  }
  ierr = DMDAVecRestoreArray(da,localX,&x);CHKERRQ(ierr);
  if (B) {
    ierr = DMDAVecRestoreArray(da,localB,&b);CHKERRQ(ierr);
  }
  ierr = DMLocalToGlobalBegin(da,localX,INSERT_VALUES,X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,localX,INSERT_VALUES,X);CHKERRQ(ierr);
  ierr = PetscLogFlops(n_sweeps*n_pointwise*(84.0 + 41)*info.ym*info.xm);CHKERRQ(ierr);
  if (B) {
    ierr = DMLocalToGlobalBegin(da,localB,INSERT_VALUES,B);CHKERRQ(ierr);
    ierr = DMLocalToGlobalEnd(da,localB,INSERT_VALUES,B);CHKERRQ(ierr);
  }
  ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
  if (B) {
    ierr = DMRestoreLocalVector(da,&localB);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
