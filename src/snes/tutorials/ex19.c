
static char help[] = "Nonlinear driven cavity with multigrid in 2d.\n \
  \n\
The 2D driven cavity problem is solved in a velocity-vorticity formulation.\n\
The flow can be driven with the lid or with bouyancy or both:\n\
  -lidvelocity &ltlid&gt, where &ltlid&gt = dimensionless velocity of lid\n\
  -grashof &ltgr&gt, where &ltgr&gt = dimensionless temperature gradent\n\
  -prandtl &ltpr&gt, where &ltpr&gt = dimensionless thermal/momentum diffusity ratio\n\
 -contours : draw contour plots of solution\n\n";
/* in HTML, '&lt' = '<' and '&gt' = '>' */

/*
      See src/ksp/ksp/tutorials/ex45.c
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
#include <petscdm.h>
#include <petscdmda.h>
#endif

/*
   User-defined routines and data structures
*/
typedef struct {
  PetscScalar u,v,omega,temp;
} Field;

PetscErrorCode FormFunctionLocal(DMDALocalInfo*,Field**,Field**,void*);

typedef struct {
  PetscReal   lidvelocity,prandtl,grashof;  /* physical parameters */
  PetscBool   draw_contours;                /* flag - 1 indicates drawing contours */
} AppCtx;

extern PetscErrorCode FormInitialGuess(AppCtx*,DM,Vec);
extern PetscErrorCode NonlinearGS(SNES,Vec,Vec,void*);

int main(int argc,char **argv)
{
  AppCtx         user;                /* user-defined work context */
  PetscInt       mx,my,its;
  PetscErrorCode ierr;
  MPI_Comm       comm;
  SNES           snes;
  DM             da;
  Vec            x;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  PetscFunctionBeginUser;
  comm = PETSC_COMM_WORLD;
  ierr = SNESCreate(comm,&snes);CHKERRQ(ierr);

  /*
      Create distributed array object to manage parallel grid and vectors
      for principal unknowns (x) and governing residuals (f)
  */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,4,4,PETSC_DECIDE,PETSC_DECIDE,4,1,0,0,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = SNESSetDM(snes,(DM)da);CHKERRQ(ierr);
  ierr = SNESSetNGS(snes, NonlinearGS, (void*)&user);CHKERRQ(ierr);

  ierr = DMDAGetInfo(da,0,&mx,&my,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);
  /*
     Problem parameters (velocity of lid, prandtl, and grashof numbers)
  */
  user.lidvelocity = 1.0/(mx*my);
  user.prandtl     = 1.0;
  user.grashof     = 1.0;

  ierr = PetscOptionsGetReal(NULL,NULL,"-lidvelocity",&user.lidvelocity,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-prandtl",&user.prandtl,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-grashof",&user.grashof,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-contours",&user.draw_contours);CHKERRQ(ierr);

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
  ierr = DMDASNESSetFunctionLocal(da,INSERT_VALUES,(PetscErrorCode (*)(DMDALocalInfo*,void*,void*,void*))FormFunctionLocal,&user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = PetscPrintf(comm,"lid velocity = %g, prandtl # = %g, grashof # = %g\n",(double)user.lidvelocity,(double)user.prandtl,(double)user.grashof);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve the nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(da,&x);CHKERRQ(ierr);
  ierr = FormInitialGuess(&user,da,x);CHKERRQ(ierr);

  ierr = SNESSolve(snes,NULL,x);CHKERRQ(ierr);

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
  return ierr;
}

/* ------------------------------------------------------------------- */

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

  PetscFunctionBeginUser;
  grashof = user->grashof;

  ierr = DMDAGetInfo(da,0,&mx,0,0,0,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  dx   = 1.0/(mx-1);

  /*
     Get local grid boundaries (for 2-dimensional DMDA):
       xs, ys   - starting grid indices (no ghost points)
       xm, ym   - widths of local grid (no ghost points)
  */
  ierr = DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL);CHKERRQ(ierr);

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */
  ierr = DMDAVecGetArrayWrite(da,X,&x);CHKERRQ(ierr);

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
  ierr = DMDAVecRestoreArrayWrite(da,X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode FormFunctionLocal(DMDALocalInfo *info,Field **x,Field **f,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  PetscErrorCode ierr;
  PetscInt       xints,xinte,yints,yinte,i,j;
  PetscReal      hx,hy,dhx,dhy,hxdhy,hydhx;
  PetscReal      grashof,prandtl,lid;
  PetscScalar    u,uxx,uyy,vx,vy,avx,avy,vxp,vxm,vyp,vym;

  PetscFunctionBeginUser;
  grashof = user->grashof;
  prandtl = user->prandtl;
  lid     = user->lidvelocity;

  /*
     Define mesh intervals ratios for uniform grid.

     Note: FD formulae below are normalized by multiplying through by
     local volume element (i.e. hx*hy) to obtain coefficients O(1) in two dimensions.

  */
  dhx   = (PetscReal)(info->mx-1);  dhy = (PetscReal)(info->my-1);
  hx    = 1.0/dhx;                   hy = 1.0/dhy;
  hxdhy = hx*dhy;                 hydhx = hy*dhx;

  xints = info->xs; xinte = info->xs+info->xm; yints = info->ys; yinte = info->ys+info->ym;

  /* Test whether we are on the bottom edge of the global array */
  if (yints == 0) {
    j     = 0;
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
    j     = info->my - 1;
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
    i     = 0;
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
    i     = info->mx - 1;
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
      vx  = x[j][i].u; avx = PetscAbsScalar(vx);
      vxp = .5*(vx+avx); vxm = .5*(vx-avx);
      vy  = x[j][i].v; avy = PetscAbsScalar(vy);
      vyp = .5*(vy+avy); vym = .5*(vy-avy);

      /* U velocity */
      u         = x[j][i].u;
      uxx       = (2.0*u - x[j][i-1].u - x[j][i+1].u)*hydhx;
      uyy       = (2.0*u - x[j-1][i].u - x[j+1][i].u)*hxdhy;
      f[j][i].u = uxx + uyy - .5*(x[j+1][i].omega-x[j-1][i].omega)*hx;

      /* V velocity */
      u         = x[j][i].v;
      uxx       = (2.0*u - x[j][i-1].v - x[j][i+1].v)*hydhx;
      uyy       = (2.0*u - x[j-1][i].v - x[j+1][i].v)*hxdhy;
      f[j][i].v = uxx + uyy + .5*(x[j][i+1].omega-x[j][i-1].omega)*hy;

      /* Omega */
      u             = x[j][i].omega;
      uxx           = (2.0*u - x[j][i-1].omega - x[j][i+1].omega)*hydhx;
      uyy           = (2.0*u - x[j-1][i].omega - x[j+1][i].omega)*hxdhy;
      f[j][i].omega = uxx + uyy + (vxp*(u - x[j][i-1].omega) + vxm*(x[j][i+1].omega - u))*hy +
                      (vyp*(u - x[j-1][i].omega) + vym*(x[j+1][i].omega - u))*hx -
                      .5*grashof*(x[j][i+1].temp - x[j][i-1].temp)*hy;

      /* Temperature */
      u            = x[j][i].temp;
      uxx          = (2.0*u - x[j][i-1].temp - x[j][i+1].temp)*hydhx;
      uyy          = (2.0*u - x[j-1][i].temp - x[j+1][i].temp)*hxdhy;
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

/*
    Performs sweeps of point block nonlinear Gauss-Seidel on all the local grid points
*/
PetscErrorCode NonlinearGS(SNES snes, Vec X, Vec B, void *ctx)
{
  DMDALocalInfo  info;
  Field          **x,**b;
  PetscErrorCode ierr;
  Vec            localX, localB;
  DM             da;
  PetscInt       xints,xinte,yints,yinte,i,j,k,l;
  PetscInt       max_its,tot_its;
  PetscInt       sweeps;
  PetscReal      rtol,atol,stol;
  PetscReal      hx,hy,dhx,dhy,hxdhy,hydhx;
  PetscReal      grashof,prandtl,lid;
  PetscScalar    u,uxx,uyy,vx,vy,avx,avy,vxp,vxm,vyp,vym;
  PetscScalar    fu, fv, fomega, ftemp;
  PetscScalar    dfudu;
  PetscScalar    dfvdv;
  PetscScalar    dfodu, dfodv, dfodo;
  PetscScalar    dftdu, dftdv, dftdt;
  PetscScalar    yu=0, yv=0, yo=0, yt=0;
  PetscScalar    bjiu, bjiv, bjiomega, bjitemp;
  PetscBool      ptconverged;
  PetscReal      pfnorm,pfnorm0,pynorm,pxnorm;
  AppCtx         *user = (AppCtx*)ctx;

  PetscFunctionBeginUser;
  grashof = user->grashof;
  prandtl = user->prandtl;
  lid     = user->lidvelocity;
  tot_its = 0;
  ierr    = SNESNGSGetTolerances(snes,&rtol,&atol,&stol,&max_its);CHKERRQ(ierr);
  ierr    = SNESNGSGetSweeps(snes,&sweeps);CHKERRQ(ierr);
  ierr    = SNESGetDM(snes,(DM*)&da);CHKERRQ(ierr);
  ierr    = DMGetLocalVector(da,&localX);CHKERRQ(ierr);
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
  ierr = DMDAVecGetArrayWrite(da,localX,&x);CHKERRQ(ierr);
  if (B) {
    ierr = DMDAVecGetArrayRead(da,localB,&b);CHKERRQ(ierr);
  }
  /* looks like a combination of the formfunction / formjacobian routines */
  dhx   = (PetscReal)(info.mx-1);dhy   = (PetscReal)(info.my-1);
  hx    = 1.0/dhx;               hy    = 1.0/dhy;
  hxdhy = hx*dhy;                hydhx = hy*dhx;

  xints = info.xs; xinte = info.xs+info.xm; yints = info.ys; yinte = info.ys+info.ym;

  /* Set the boundary conditions on the momentum equations */
  /* Test whether we are on the bottom edge of the global array */
  if (yints == 0) {
    j     = 0;
    /* bottom edge */
    for (i=info.xs; i<info.xs+info.xm; i++) {

      if (B) {
        bjiu = b[j][i].u;
        bjiv = b[j][i].v;
      } else {
        bjiu = 0.0;
        bjiv = 0.0;
      }
      x[j][i].u = 0.0 + bjiu;
      x[j][i].v = 0.0 + bjiv;
    }
  }

  /* Test whether we are on the top edge of the global array */
  if (yinte == info.my) {
    j     = info.my - 1;
    /* top edge */
    for (i=info.xs; i<info.xs+info.xm; i++) {
      if (B) {
        bjiu = b[j][i].u;
        bjiv = b[j][i].v;
      } else {
        bjiu = 0.0;
        bjiv = 0.0;
      }
      x[j][i].u = lid + bjiu;
      x[j][i].v = bjiv;
    }
  }

  /* Test whether we are on the left edge of the global array */
  if (xints == 0) {
    i     = 0;
    /* left edge */
    for (j=info.ys; j<info.ys+info.ym; j++) {
      if (B) {
        bjiu = b[j][i].u;
        bjiv = b[j][i].v;
      } else {
        bjiu = 0.0;
        bjiv = 0.0;
      }
      x[j][i].u = 0.0 + bjiu;
      x[j][i].v = 0.0 + bjiv;
    }
  }

  /* Test whether we are on the right edge of the global array */
  if (xinte == info.mx) {
    i     = info.mx - 1;
    /* right edge */
    for (j=info.ys; j<info.ys+info.ym; j++) {
      if (B) {
        bjiu = b[j][i].u;
        bjiv = b[j][i].v;
      } else {
        bjiu = 0.0;
        bjiv = 0.0;
      }
      x[j][i].u = 0.0 + bjiu;
      x[j][i].v = 0.0 + bjiv;
    }
  }

  for (k=0; k < sweeps; k++) {
    for (j=info.ys; j<info.ys + info.ym; j++) {
      for (i=info.xs; i<info.xs + info.xm; i++) {
        ptconverged = PETSC_FALSE;
        pfnorm0     = 0.0;
        fu          = 0.0;
        fv          = 0.0;
        fomega      = 0.0;
        ftemp       = 0.0;
        /*  Run Newton's method on a single grid point */
        for (l = 0; l < max_its && !ptconverged; l++) {
          if (B) {
            bjiu     = b[j][i].u;
            bjiv     = b[j][i].v;
            bjiomega = b[j][i].omega;
            bjitemp  = b[j][i].temp;
          } else {
            bjiu     = 0.0;
            bjiv     = 0.0;
            bjiomega = 0.0;
            bjitemp  = 0.0;
          }

          if (i != 0 && i != info.mx - 1 && j != 0 && j != info.my-1) {
            /* U velocity */
            u     = x[j][i].u;
            uxx   = (2.0*u - x[j][i-1].u - x[j][i+1].u)*hydhx;
            uyy   = (2.0*u - x[j-1][i].u - x[j+1][i].u)*hxdhy;
            fu    = uxx + uyy - .5*(x[j+1][i].omega-x[j-1][i].omega)*hx - bjiu;
            dfudu = 2.0*(hydhx + hxdhy);
            /* V velocity */
            u     = x[j][i].v;
            uxx   = (2.0*u - x[j][i-1].v - x[j][i+1].v)*hydhx;
            uyy   = (2.0*u - x[j-1][i].v - x[j+1][i].v)*hxdhy;
            fv    = uxx + uyy + .5*(x[j][i+1].omega-x[j][i-1].omega)*hy - bjiv;
            dfvdv = 2.0*(hydhx + hxdhy);
            /*
             convective coefficients for upwinding
             */
            vx  = x[j][i].u; avx = PetscAbsScalar(vx);
            vxp = .5*(vx+avx); vxm = .5*(vx-avx);
            vy  = x[j][i].v; avy = PetscAbsScalar(vy);
            vyp = .5*(vy+avy); vym = .5*(vy-avy);
            /* Omega */
            u      = x[j][i].omega;
            uxx    = (2.0*u - x[j][i-1].omega - x[j][i+1].omega)*hydhx;
            uyy    = (2.0*u - x[j-1][i].omega - x[j+1][i].omega)*hxdhy;
            fomega = uxx + uyy +  (vxp*(u - x[j][i-1].omega) + vxm*(x[j][i+1].omega - u))*hy +
                     (vyp*(u - x[j-1][i].omega) + vym*(x[j+1][i].omega - u))*hx -
                     .5*grashof*(x[j][i+1].temp - x[j][i-1].temp)*hy - bjiomega;
            /* convective coefficient derivatives */
            dfodo = 2.0*(hydhx + hxdhy) + ((vxp - vxm)*hy + (vyp - vym)*hx);
            if (PetscRealPart(vx) > 0.0) dfodu = (u - x[j][i-1].omega)*hy;
            else dfodu = (x[j][i+1].omega - u)*hy;

            if (PetscRealPart(vy) > 0.0) dfodv = (u - x[j-1][i].omega)*hx;
            else dfodv = (x[j+1][i].omega - u)*hx;

            /* Temperature */
            u     = x[j][i].temp;
            uxx   = (2.0*u - x[j][i-1].temp - x[j][i+1].temp)*hydhx;
            uyy   = (2.0*u - x[j-1][i].temp - x[j+1][i].temp)*hxdhy;
            ftemp =  uxx + uyy  + prandtl*((vxp*(u - x[j][i-1].temp) + vxm*(x[j][i+1].temp - u))*hy + (vyp*(u - x[j-1][i].temp) + vym*(x[j+1][i].temp - u))*hx) - bjitemp;
            dftdt = 2.0*(hydhx + hxdhy) + prandtl*((vxp - vxm)*hy + (vyp - vym)*hx);
            if (PetscRealPart(vx) > 0.0) dftdu = prandtl*(u - x[j][i-1].temp)*hy;
            else dftdu = prandtl*(x[j][i+1].temp - u)*hy;

            if (PetscRealPart(vy) > 0.0) dftdv = prandtl*(u - x[j-1][i].temp)*hx;
            else dftdv = prandtl*(x[j+1][i].temp - u)*hx;

            /* invert the system:
             [ dfu / du     0        0        0    ][yu] = [fu]
             [     0    dfv / dv     0        0    ][yv]   [fv]
             [ dfo / du dfo / dv dfo / do     0    ][yo]   [fo]
             [ dft / du dft / dv     0    dft / dt ][yt]   [ft]
             by simple back-substitution
           */
            yu = fu / dfudu;
            yv = fv / dfvdv;
            yo = (fomega - (dfodu*yu + dfodv*yv)) / dfodo;
            yt = (ftemp - (dftdu*yu + dftdv*yv)) / dftdt;

            x[j][i].u     = x[j][i].u - yu;
            x[j][i].v     = x[j][i].v - yv;
            x[j][i].temp  = x[j][i].temp - yt;
            x[j][i].omega = x[j][i].omega - yo;
          }
          if (i == 0) {
            fomega        = x[j][i].omega - (x[j][i+1].v - x[j][i].v)*dhx - bjiomega;
            ftemp         = x[j][i].temp - bjitemp;
            yo            = fomega;
            yt            = ftemp;
            x[j][i].omega = x[j][i].omega - fomega;
            x[j][i].temp  = x[j][i].temp - ftemp;
          }
          if (i == info.mx - 1) {
            fomega        = x[j][i].omega - (x[j][i].v - x[j][i-1].v)*dhx - bjiomega;
            ftemp         = x[j][i].temp - (PetscReal)(grashof>0) - bjitemp;
            yo            = fomega;
            yt            = ftemp;
            x[j][i].omega = x[j][i].omega - fomega;
            x[j][i].temp  = x[j][i].temp - ftemp;
          }
          if (j == 0) {
            fomega        = x[j][i].omega + (x[j+1][i].u - x[j][i].u)*dhy - bjiomega;
            ftemp         = x[j][i].temp-x[j+1][i].temp - bjitemp;
            yo            = fomega;
            yt            = ftemp;
            x[j][i].omega = x[j][i].omega - fomega;
            x[j][i].temp  = x[j][i].temp - ftemp;
          }
          if (j == info.my - 1) {
            fomega        = x[j][i].omega + (x[j][i].u - x[j-1][i].u)*dhy - bjiomega;
            ftemp         = x[j][i].temp-x[j-1][i].temp - bjitemp;
            yo            = fomega;
            yt            = ftemp;
            x[j][i].omega = x[j][i].omega - fomega;
            x[j][i].temp  = x[j][i].temp - ftemp;
          }
          tot_its++;
          pfnorm = PetscRealPart(fu*fu + fv*fv + fomega*fomega + ftemp*ftemp);
          pfnorm = PetscSqrtReal(pfnorm);
          pynorm = PetscRealPart(yu*yu + yv*yv + yo*yo + yt*yt);
          pynorm = PetscSqrtReal(pynorm);
          pxnorm = PetscRealPart(x[j][i].u*x[j][i].u + x[j][i].v*x[j][i].v + x[j][i].omega*x[j][i].omega + x[j][i].temp*x[j][i].temp);
          pxnorm = PetscSqrtReal(pxnorm);
          if (l == 0) pfnorm0 = pfnorm;
          if (rtol*pfnorm0 >pfnorm || atol > pfnorm || pxnorm*stol > pynorm) ptconverged = PETSC_TRUE;
        }
      }
    }
  }
  ierr = DMDAVecRestoreArrayWrite(da,localX,&x);CHKERRQ(ierr);
  if (B) {
    ierr = DMDAVecRestoreArrayRead(da,localB,&b);CHKERRQ(ierr);
  }
  ierr = DMLocalToGlobalBegin(da,localX,INSERT_VALUES,X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,localX,INSERT_VALUES,X);CHKERRQ(ierr);
  ierr = PetscLogFlops(tot_its*(84.0 + 41.0 + 26.0));CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(da,&localX);CHKERRQ(ierr);
  if (B) {
    ierr = DMRestoreLocalVector(da,&localB);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*TEST

   test:
      nsize: 2
      args: -da_refine 3 -snes_monitor_short -pc_type mg -ksp_type fgmres -pc_mg_type full
      requires: !single

   test:
      suffix: 10
      nsize: 3
      args: -snes_monitor_short -ksp_monitor_short -pc_type fieldsplit -pc_fieldsplit_type symmetric_multiplicative -snes_view -da_refine 1 -ksp_type fgmres
      requires: !single

   test:
      suffix: 11
      nsize: 4
      requires: pastix
      args: -snes_monitor_short -pc_type redundant -dm_mat_type mpiaij -redundant_pc_factor_mat_solver_type pastix -pc_redundant_number 2 -da_refine 4 -ksp_type fgmres

   test:
      suffix: 12
      nsize: 12
      requires: pastix
      args: -snes_monitor_short -pc_type redundant -dm_mat_type mpiaij -redundant_pc_factor_mat_solver_type pastix -pc_redundant_number 5 -da_refine 4 -ksp_type fgmres

   test:
      suffix: 13
      nsize: 3
      args: -snes_monitor_short -ksp_monitor_short -pc_type fieldsplit -pc_fieldsplit_type multiplicative -snes_view -da_refine 1 -ksp_type fgmres -snes_mf_operator
      requires: !single

   test:
      suffix: 14
      nsize: 4
      args: -snes_monitor_short -pc_type mg -dm_mat_type baij -mg_coarse_pc_type bjacobi -da_refine 3 -ksp_type fgmres
      requires: !single

   test:
      suffix: 14_ds
      nsize: 4
      args: -snes_converged_reason -pc_type mg -dm_mat_type baij -mg_coarse_pc_type bjacobi -da_refine 3 -ksp_type fgmres -mat_fd_type ds
      output_file: output/ex19_2.out
      requires: !single

   test:
      suffix: 17
      args: -snes_monitor_short -ksp_pc_side right
      requires: !single

   test:
      suffix: 18
      args: -snes_monitor_ksp draw::draw_lg -ksp_pc_side right
      requires: x !single

   test:
      suffix: 19
      nsize: 2
      args: -da_refine 3 -snes_monitor_short -pc_type mg -ksp_type fgmres -pc_mg_type full -snes_type newtontrdc
      requires: !single

   test:
      suffix: 20
      nsize: 2
      args: -da_refine 3 -snes_monitor_short -pc_type mg -ksp_type fgmres -pc_mg_type full -snes_type newtontrdc -snes_trdc_use_cauchy false
      requires: !single

   test:
      suffix: 2
      nsize: 4
      args: -da_refine 3 -snes_converged_reason -pc_type mg -mat_fd_type ds
      requires: !single

   test:
      suffix: 2_bcols1
      nsize: 4
      args: -da_refine 3 -snes_converged_reason -pc_type mg -mat_fd_type ds -mat_fd_coloring_bcols
      output_file: output/ex19_2.out
      requires: !single

   test:
      suffix: 3
      nsize: 4
      requires: mumps
      args: -da_refine 3 -snes_monitor_short -pc_type redundant -dm_mat_type mpiaij -redundant_ksp_type preonly -redundant_pc_factor_mat_solver_type mumps -pc_redundant_number 2

   test:
      suffix: 4
      nsize: 12
      requires: mumps
      args: -da_refine 3 -snes_monitor_short -pc_type redundant -dm_mat_type mpiaij -redundant_ksp_type preonly -redundant_pc_factor_mat_solver_type mumps -pc_redundant_number 5
      output_file: output/ex19_3.out

   test:
      suffix: 6
      args: -snes_monitor_short -ksp_monitor_short -pc_type fieldsplit -snes_view -ksp_type fgmres -da_refine 1
      requires: !single

   test:
      suffix: 7
      nsize: 3
      args: -snes_monitor_short -ksp_monitor_short -pc_type fieldsplit -snes_view -da_refine 1 -ksp_type fgmres

      requires: !single
   test:
      suffix: 8
      args: -snes_monitor_short -ksp_monitor_short -pc_type fieldsplit -pc_fieldsplit_block_size 2 -pc_fieldsplit_0_fields 0,1 -pc_fieldsplit_1_fields 0,1 -pc_fieldsplit_type multiplicative -snes_view -fieldsplit_pc_type lu -da_refine 1 -ksp_type fgmres
      requires: !single

   test:
      suffix: 9
      nsize: 3
      args: -snes_monitor_short -ksp_monitor_short -pc_type fieldsplit -pc_fieldsplit_type multiplicative -snes_view -da_refine 1 -ksp_type fgmres
      requires: !single

   test:
      suffix: aspin
      nsize: 4
      args: -da_refine 3 -da_overlap 2 -snes_monitor_short -snes_type aspin -grashof 4e4 -lidvelocity 100 -ksp_monitor_short
      requires: !single

   test:
      suffix: bcgsl
      nsize: 2
      args: -ksp_type bcgsl -ksp_monitor_short -da_refine 2 -ksp_bcgsl_ell 3 -snes_view
      requires: !single

   test:
      suffix: bcols1
      nsize: 2
      args: -da_refine 3 -snes_monitor_short -pc_type mg -ksp_type fgmres -pc_mg_type full -mat_fd_coloring_bcols 1
      output_file: output/ex19_1.out
      requires: !single

   test:
      suffix: bjacobi
      nsize: 4
      args: -da_refine 4 -ksp_type fgmres -pc_type bjacobi -pc_bjacobi_blocks 2 -sub_ksp_type gmres -sub_ksp_max_it 2 -sub_pc_type bjacobi -sub_sub_ksp_type preonly -sub_sub_pc_type ilu -snes_monitor_short
      requires: !single

   test:
      suffix: cgne
      args: -da_refine 2 -pc_type lu -ksp_type cgne -ksp_monitor_short -ksp_converged_reason -ksp_view -ksp_norm_type unpreconditioned
      filter: grep -v HERMITIAN
      requires: !single

   test:
      suffix: cgs
      args: -da_refine 1 -ksp_monitor_short -ksp_type cgs
      requires: !single

   test:
      suffix: composite_fieldsplit
      args: -ksp_type fgmres -pc_type composite -pc_composite_type MULTIPLICATIVE -pc_composite_pcs fieldsplit,none -sub_0_pc_fieldsplit_block_size 4 -sub_0_pc_fieldsplit_type additive -sub_0_pc_fieldsplit_0_fields 0,1,2 -sub_0_pc_fieldsplit_1_fields 3 -snes_monitor_short -ksp_monitor_short
      requires: !single

   test:
      suffix: composite_fieldsplit_bjacobi
      args: -ksp_type fgmres -pc_type composite -pc_composite_type MULTIPLICATIVE -pc_composite_pcs fieldsplit,bjacobi -sub_0_pc_fieldsplit_block_size 4 -sub_0_pc_fieldsplit_type additive -sub_0_pc_fieldsplit_0_fields 0,1,2 -sub_0_pc_fieldsplit_1_fields 3 -sub_1_pc_bjacobi_blocks 16 -sub_1_sub_pc_type lu -snes_monitor_short -ksp_monitor_short
      requires: !single

   test:
      suffix: composite_fieldsplit_bjacobi_2
      nsize: 4
      args: -ksp_type fgmres -pc_type composite -pc_composite_type MULTIPLICATIVE -pc_composite_pcs fieldsplit,bjacobi -sub_0_pc_fieldsplit_block_size 4 -sub_0_pc_fieldsplit_type additive -sub_0_pc_fieldsplit_0_fields 0,1,2 -sub_0_pc_fieldsplit_1_fields 3 -sub_1_pc_bjacobi_blocks 16 -sub_1_sub_pc_type lu -snes_monitor_short -ksp_monitor_short
      requires: !single

   test:
      suffix: composite_gs_newton
      nsize: 2
      args: -da_refine 3 -grashof 4e4 -lidvelocity 100 -snes_monitor_short -snes_type composite -snes_composite_type additiveoptimal -snes_composite_sneses ngs,newtonls -sub_0_snes_max_it 20 -sub_1_pc_type mg
      requires: !single

   test:
      suffix: cuda
      requires: cuda !single
      args: -dm_vec_type cuda -dm_mat_type aijcusparse -pc_type none -ksp_type fgmres -snes_monitor_short -snes_rtol 1.e-5

   test:
      suffix: draw
      args: -pc_type fieldsplit -snes_view draw -fieldsplit_x_velocity_pc_type mg -fieldsplit_x_velocity_pc_mg_galerkin pmat -fieldsplit_x_velocity_pc_mg_levels 2 -da_refine 1 -fieldsplit_x_velocity_mg_coarse_pc_type svd
      requires: x !single

   test:
      suffix: drawports
      args: -snes_monitor_solution draw::draw_ports -da_refine 1
      output_file: output/ex19_draw.out
      requires: x !single

   test:
      suffix: fas
      args: -da_refine 4 -snes_monitor_short -snes_type fas -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 3 -fas_levels_snes_ngs_atol 0.0 -fas_levels_snes_ngs_stol 0.0 -grashof 4e4 -snes_fas_smoothup 6 -snes_fas_smoothdown 6 -lidvelocity 100
      requires: !single

   test:
      suffix: fas_full
      args: -da_refine 4 -snes_monitor_short -snes_type fas -snes_fas_type full -snes_fas_full_downsweep -fas_levels_snes_type ngs -fas_levels_snes_ngs_sweeps 3 -fas_levels_snes_ngs_atol 0.0 -fas_levels_snes_ngs_stol 0.0 -grashof 4e4 -snes_fas_smoothup 6 -snes_fas_smoothdown 6 -lidvelocity 100
      requires: !single

   test:
      suffix: fdcoloring_ds
      args: -da_refine 3 -snes_converged_reason -pc_type mg -mat_fd_type ds
      output_file: output/ex19_2.out
      requires: !single

   test:
      suffix: fdcoloring_ds_baij
      args: -da_refine 3 -snes_converged_reason -pc_type mg -mat_fd_type ds -dm_mat_type baij
      output_file: output/ex19_2.out
      requires: !single

   test:
      suffix: fdcoloring_ds_bcols1
      args: -da_refine 3 -snes_converged_reason -pc_type mg -mat_fd_type ds -mat_fd_coloring_bcols 1
      output_file: output/ex19_2.out
      requires: !single

   test:
      suffix: fdcoloring_wp
      args: -da_refine 3 -snes_monitor_short -pc_type mg
      requires: !single

   test:
      suffix: fdcoloring_wp_baij
      args: -da_refine 3 -snes_monitor_short -pc_type mg -dm_mat_type baij
      output_file: output/ex19_fdcoloring_wp.out
      requires: !single

   test:
      suffix: fdcoloring_wp_bcols1
      args: -da_refine 3 -snes_monitor_short -pc_type mg -mat_fd_coloring_bcols 1
      output_file: output/ex19_fdcoloring_wp.out
      requires: !single

   test:
      suffix: fieldsplit_2
      args: -ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_block_size 4 -pc_fieldsplit_type additive -pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3 -snes_monitor_short -ksp_monitor_short
      requires: !single

   test:
      suffix: fieldsplit_3
      args: -ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_block_size 4 -pc_fieldsplit_type additive -pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3 -fieldsplit_0_pc_type lu -fieldsplit_1_pc_type lu -snes_monitor_short -ksp_monitor_short
      requires: !single

   test:
      suffix: fieldsplit_4
      args: -ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_block_size 4 -pc_fieldsplit_type SCHUR -pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3 -fieldsplit_0_pc_type lu -fieldsplit_1_pc_type lu -snes_monitor_short -ksp_monitor_short
      requires: !single

   # HYPRE PtAP broken with complex numbers
   test:
      suffix: fieldsplit_hypre
      nsize: 2
      requires: hypre mumps !complex !defined(PETSC_HAVE_HYPRE_DEVICE)
      args: -pc_type fieldsplit -pc_fieldsplit_block_size 4 -pc_fieldsplit_type SCHUR -pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3 -fieldsplit_0_pc_type lu -fieldsplit_0_pc_factor_mat_solver_type mumps -fieldsplit_1_pc_type hypre -fieldsplit_1_pc_hypre_type boomeramg -snes_monitor_short -ksp_monitor_short

   test:
      suffix: fieldsplit_mumps
      nsize: 2
      requires: mumps
      args: -pc_type fieldsplit -pc_fieldsplit_block_size 4 -pc_fieldsplit_type SCHUR -pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3 -fieldsplit_0_pc_type lu -fieldsplit_1_pc_type lu -snes_monitor_short -ksp_monitor_short -fieldsplit_0_pc_factor_mat_solver_type mumps -fieldsplit_1_pc_factor_mat_solver_type mumps
      output_file: output/ex19_fieldsplit_5.out

   test:
      suffix: greedy_coloring
      nsize: 2
      args: -da_refine 3 -snes_monitor_short -snes_fd_color -snes_fd_color_use_mat -mat_coloring_type greedy -mat_coloring_weight_type lf -mat_coloring_view> ex19_greedy_coloring.tmp 2>&1
      requires: !single

   # HYPRE PtAP broken with complex numbers
   test:
      suffix: hypre
      nsize: 2
      requires: hypre !complex !defined(PETSC_HAVE_HYPRE_DEVICE)
      args: -da_refine 3 -snes_monitor_short -pc_type hypre -ksp_norm_type unpreconditioned

   # ibcgs is broken when using device vectors
   test:
      suffix: ibcgs
      nsize: 2
      args: -ksp_type ibcgs -ksp_monitor_short -da_refine 2 -snes_view
      requires: !complex !single

   test:
      suffix: kaczmarz
      nsize: 2
      args: -pc_type kaczmarz -ksp_monitor_short -snes_monitor_short -snes_view
      requires: !single

   test:
      suffix: klu
      requires: suitesparse
      args: -da_grid_x 20 -da_grid_y 20 -pc_type lu -pc_factor_mat_solver_type klu
      output_file: output/ex19_superlu.out

   test:
      suffix: klu_2
      requires: suitesparse
      args: -da_grid_x 20 -da_grid_y 20 -pc_type lu -pc_factor_mat_solver_type klu -pc_factor_mat_ordering_type nd
      output_file: output/ex19_superlu.out

   test:
      suffix: klu_3
      requires: suitesparse
      args: -da_grid_x 20 -da_grid_y 20 -pc_type lu -pc_factor_mat_solver_type klu -mat_klu_use_btf 0
      output_file: output/ex19_superlu.out

   test:
      suffix: ml
      nsize: 2
      requires: ml
      args: -da_refine 3 -snes_monitor_short -pc_type ml

   test:
      suffix: ngmres_fas
      args: -da_refine 4 -snes_monitor_short -snes_type ngmres -npc_fas_levels_snes_type ngs -npc_fas_levels_snes_ngs_sweeps 3 -npc_fas_levels_snes_ngs_atol 0.0 -npc_fas_levels_snes_ngs_stol 0.0 -npc_snes_type fas -npc_fas_levels_snes_type ngs -npc_snes_max_it 1 -npc_snes_fas_smoothup 6 -npc_snes_fas_smoothdown 6 -lidvelocity 100 -grashof 4e4
      requires: !single

   test:
      suffix: ngmres_fas_gssecant
      args: -da_refine 3 -snes_monitor_short -snes_type ngmres -npc_snes_type fas -npc_fas_levels_snes_type ngs -npc_fas_levels_snes_max_it 6 -npc_fas_levels_snes_ngs_secant -npc_fas_levels_snes_ngs_max_it 1 -npc_fas_coarse_snes_max_it 1 -lidvelocity 100 -grashof 4e4
      requires: !single

   test:
      suffix: ngmres_fas_ms
      nsize: 2
      args: -snes_grid_sequence 2 -lidvelocity 200 -grashof 1e4 -snes_monitor_short -snes_view -snes_converged_reason -snes_type ngmres -npc_snes_type fas -npc_fas_coarse_snes_type newtonls -npc_fas_coarse_ksp_type preonly -npc_snes_max_it 1
      requires: !single

   test:
      suffix: ngmres_nasm
      nsize: 4
      args: -da_refine 4 -da_overlap 2 -snes_monitor_short -snes_type ngmres -snes_max_it 10 -npc_snes_type nasm -npc_snes_nasm_type basic -grashof 4e4 -lidvelocity 100
      requires: !single

   test:
      suffix: ngs
      args: -snes_type ngs -snes_view -snes_monitor -snes_rtol 1e-4
      requires: !single

   test:
      suffix: ngs_fd
      args: -snes_type ngs -snes_ngs_secant -snes_view -snes_monitor -snes_rtol 1e-4
      requires: !single

   test:
      suffix: parms
      nsize: 2
      requires: parms
      args: -pc_type parms -ksp_monitor_short -snes_view

   test:
      suffix: superlu
      requires: superlu
      args: -da_grid_x 20 -da_grid_y 20 -pc_type lu -pc_factor_mat_solver_type superlu

   test:
      suffix: superlu_sell
      requires: superlu
      args: -da_grid_x 20 -da_grid_y 20 -pc_type lu -pc_factor_mat_solver_type superlu -dm_mat_type sell -pc_factor_mat_ordering_type natural
      output_file: output/ex19_superlu.out

   test:
      suffix: superlu_dist
      requires: superlu_dist
      args: -da_grid_x 20 -da_grid_y 20 -pc_type lu -pc_factor_mat_solver_type superlu_dist
      output_file: output/ex19_superlu.out

   test:
      suffix: superlu_dist_2
      nsize: 2
      requires: superlu_dist
      args: -da_grid_x 20 -da_grid_y 20 -pc_type lu -pc_factor_mat_solver_type superlu_dist
      output_file: output/ex19_superlu.out

   test:
      suffix: superlu_equil
      requires: superlu
      args: -da_grid_x 20 -da_grid_y 20 -{snes,ksp}_monitor_short -pc_type lu -pc_factor_mat_solver_type superlu -mat_superlu_equil

   test:
      suffix: superlu_equil_sell
      requires: superlu
      args: -da_grid_x 20 -da_grid_y 20 -{snes,ksp}_monitor_short -pc_type lu -pc_factor_mat_solver_type superlu -mat_superlu_equil -dm_mat_type sell -pc_factor_mat_ordering_type natural
      output_file: output/ex19_superlu_equil.out

   test:
      suffix: tcqmr
      args: -da_refine 1 -ksp_monitor_short -ksp_type tcqmr
      requires: !single

   test:
      suffix: tfqmr
      args: -da_refine 1 -ksp_monitor_short -ksp_type tfqmr
      requires: !single

   test:
      suffix: umfpack
      requires: suitesparse
      args: -da_refine 2 -pc_type lu -pc_factor_mat_solver_type umfpack -snes_view -snes_monitor_short -ksp_monitor_short -pc_factor_mat_ordering_type external

   test:
      suffix: tut_1
      nsize: 4
      requires: !single
      args: -da_refine 5 -snes_monitor -ksp_monitor -snes_view

   test:
      suffix: tut_2
      nsize: 4
      requires: !single
      args: -da_refine 5 -snes_monitor -ksp_monitor -snes_view -pc_type mg

   # HYPRE PtAP broken with complex numbers
   test:
      suffix: tut_3
      nsize: 4
      requires: hypre !single !complex !defined(PETSC_HAVE_HYPRE_DEVICE)
      args: -da_refine 5 -snes_monitor -ksp_monitor -snes_view -pc_type hypre

   test:
      suffix: tut_8
      nsize: 4
      requires: ml !single
      args: -da_refine 5 -snes_monitor -ksp_monitor -snes_view -pc_type ml

   test:
      suffix: tut_4
      nsize: 1
      requires: !single
      args: -da_refine 5 -log_view
      filter: head -n 2
      filter_output: head -n 2

   test:
      suffix: tut_5
      nsize: 1
      requires: !single
      args: -da_refine 5 -log_view -pc_type mg
      filter: head -n 2
      filter_output: head -n 2

   test:
      suffix: tut_6
      nsize: 4
      requires: !single
      args: -da_refine 5 -log_view
      filter: head -n 2
      filter_output: head -n 2

   test:
      suffix: tut_7
      nsize: 4
      requires: !single
      args: -da_refine 5 -log_view -pc_type mg
      filter: head -n 2
      filter_output: head -n 2

   test:
      suffix: cuda_1
      nsize: 1
      requires: cuda
      args: -snes_monitor -dm_mat_type seqaijcusparse -dm_vec_type seqcuda -pc_type gamg -ksp_monitor -mg_levels_ksp_max_it 3

   test:
      suffix: cuda_2
      nsize: 3
      requires: cuda !single
      args: -snes_monitor -dm_mat_type mpiaijcusparse -dm_vec_type mpicuda -pc_type gamg -ksp_monitor  -mg_levels_ksp_max_it 3

   test:
      suffix: cuda_dm_bind_below
      nsize: 2
      requires: cuda
      args: -dm_mat_type aijcusparse -dm_vec_type cuda -da_refine 3 -pc_type mg -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi -log_view -pc_mg_log -dm_bind_below 10000
      filter: awk "/Level/ {print \$24}"

   test:
      suffix: viennacl_dm_bind_below
      nsize: 2
      requires: viennacl
      args: -dm_mat_type aijviennacl -dm_vec_type viennacl -da_refine 3 -pc_type mg -mg_levels_ksp_type chebyshev -mg_levels_pc_type jacobi -log_view -pc_mg_log -dm_bind_below 10000
      filter: awk "/Level/ {print \$24}"

   test:
      suffix: seqbaijmkl
      nsize: 1
      requires: defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE)
      args: -dm_mat_type baij -snes_monitor -ksp_monitor -snes_view

   test:
      suffix: mpibaijmkl
      nsize: 2
      requires:  defined(PETSC_HAVE_MKL_SPARSE_OPTIMIZE)
      args: -dm_mat_type baij -snes_monitor -ksp_monitor -snes_view

   test:
     suffix: cpardiso
     nsize: 4
     requires: mkl_cpardiso
     args: -pc_type lu -pc_factor_mat_solver_type mkl_cpardiso -ksp_monitor

   test:
     suffix: logviewmemory
     requires: defined(PETSC_USE_LOG) !defined(PETSCTEST_VALGRIND)
     args: -log_view -log_view_memory -da_refine 4
     filter: grep MatFDColorSetUp | wc -w | xargs  -I % sh -c "expr % \> 21"

   test:
     suffix: fs
     args: -pc_type fieldsplit -da_refine 3  -all_ksp_monitor -fieldsplit_y_velocity_pc_type lu  -fieldsplit_temperature_pc_type lu -fieldsplit_x_velocity_pc_type lu  -snes_view

   test:
     suffix: asm_matconvert
     args: -mat_type aij -pc_type asm -pc_asm_sub_mat_type dense -snes_view

   test:
      suffix: euclid
      nsize: 2
      requires: hypre !single !complex !defined(PETSC_HAVE_HYPRE_MIXEDINT) !defined(PETSC_HAVE_HYPRE_DEVICE)
      args: -da_refine 2 -ksp_monitor -snes_monitor -snes_view -pc_type hypre -pc_hypre_type euclid

   test:
      suffix: euclid_bj
      nsize: 2
      requires: hypre !single !complex !defined(PETSC_HAVE_HYPRE_MIXEDINT) !defined(PETSC_HAVE_HYPRE_DEVICE)
      args: -da_refine 2 -ksp_monitor -snes_monitor -snes_view -pc_type hypre -pc_hypre_type euclid -pc_hypre_euclid_bj

   test:
      suffix: euclid_droptolerance
      nsize: 1
      requires: hypre !single !complex !defined(PETSC_HAVE_HYPRE_MIXEDINT) !defined(PETSC_HAVE_HYPRE_DEVICE)
      args: -da_refine 2 -ksp_monitor -snes_monitor -snes_view -pc_type hypre -pc_hypre_type euclid -pc_hypre_euclid_droptolerance .1

TEST*/
