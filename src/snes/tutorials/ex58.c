
#include <petscsnes.h>
#include <petscdm.h>
#include <petscdmda.h>

static const char help[] = "Parallel version of the minimum surface area problem in 2D using DMDA.\n\
 It solves a system of nonlinear equations in mixed\n\
complementarity form.This example is based on a\n\
problem from the MINPACK-2 test suite.  Given a rectangular 2-D domain and\n\
boundary values along the edges of the domain, the objective is to find the\n\
surface with the minimal area that satisfies the boundary conditions.\n\
This application solves this problem using complimentarity -- We are actually\n\
solving the system  (grad f)_i >= 0, if x_i == l_i \n\
                    (grad f)_i = 0, if l_i < x_i < u_i \n\
                    (grad f)_i <= 0, if x_i == u_i  \n\
where f is the function to be minimized. \n\
\n\
The command line options are:\n\
  -da_grid_x <nx>, where <nx> = number of grid points in the 1st coordinate direction\n\
  -da_grid_y <ny>, where <ny> = number of grid points in the 2nd coordinate direction\n\
  -start <st>, where <st> =0 for zero vector, and an average of the boundary conditions otherwise\n\
  -lb <value>, lower bound on the variables\n\
  -ub <value>, upper bound on the variables\n\n";

/*
   User-defined application context - contains data needed by the
   application-provided call-back routines, FormJacobian() and
   FormFunction().
*/

/*
     This is a new version of the ../tests/ex8.c code

     Run, for example, with the options ./ex58 -snes_vi_monitor -ksp_monitor -mg_levels_ksp_monitor -pc_type mg -pc_mg_levels 2 -pc_mg_galerkin pmat -ksp_type fgmres

     Or to run with grid sequencing on the nonlinear problem (note that you do not need to provide the number of
         multigrid levels, it will be determined automatically based on the number of refinements done)

      ./ex58 -pc_type mg -ksp_monitor  -snes_view -pc_mg_galerkin pmat -snes_grid_sequence 3
             -mg_levels_ksp_monitor -snes_vi_monitor -mg_levels_pc_type sor -pc_mg_type full

*/

typedef struct {
  PetscScalar *bottom, *top, *left, *right;
  PetscScalar lb,ub;
} AppCtx;

/* -------- User-defined Routines --------- */

extern PetscErrorCode FormBoundaryConditions(SNES,AppCtx**);
extern PetscErrorCode DestroyBoundaryConditions(AppCtx**);
extern PetscErrorCode ComputeInitialGuess(SNES,Vec,void*);
extern PetscErrorCode FormGradient(SNES,Vec,Vec,void*);
extern PetscErrorCode FormJacobian(SNES,Vec,Mat,Mat,void*);
extern PetscErrorCode FormBounds(SNES,Vec,Vec);

int main(int argc, char **argv)
{
  Vec            x,r;               /* solution and residual vectors */
  SNES           snes;              /* nonlinear solver context */
  Mat            J;                 /* Jacobian matrix */
  DM             da;

  CHKERRQ(PetscInitialize(&argc, &argv, (char*)0, help));

  /* Create distributed array to manage the 2d grid */
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,4,4,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));

  /* Extract global vectors from DMDA; */
  CHKERRQ(DMCreateGlobalVector(da,&x));
  CHKERRQ(VecDuplicate(x, &r));

  CHKERRQ(DMSetMatType(da,MATAIJ));
  CHKERRQ(DMCreateMatrix(da,&J));

  /* Create nonlinear solver context */
  CHKERRQ(SNESCreate(PETSC_COMM_WORLD,&snes));
  CHKERRQ(SNESSetDM(snes,da));

  /*  Set function evaluation and Jacobian evaluation  routines */
  CHKERRQ(SNESSetFunction(snes,r,FormGradient,NULL));
  CHKERRQ(SNESSetJacobian(snes,J,J,FormJacobian,NULL));

  CHKERRQ(SNESSetComputeApplicationContext(snes,(PetscErrorCode (*)(SNES,void**))FormBoundaryConditions,(PetscErrorCode (*)(void**))DestroyBoundaryConditions));

  CHKERRQ(SNESSetComputeInitialGuess(snes,ComputeInitialGuess,NULL));

  CHKERRQ(SNESVISetComputeVariableBounds(snes,FormBounds));

  CHKERRQ(SNESSetFromOptions(snes));

  /* Solve the application */
  CHKERRQ(SNESSolve(snes,NULL,x));

  /* Free memory */
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&r));
  CHKERRQ(MatDestroy(&J));
  CHKERRQ(SNESDestroy(&snes));

  /* Free user-created data structures */
  CHKERRQ(DMDestroy(&da));

  CHKERRQ(PetscFinalize());
  return 0;
}

/* -------------------------------------------------------------------- */

/*  FormBounds - sets the upper and lower bounds

    Input Parameters:
.   snes  - the SNES context

    Output Parameters:
.   xl - lower bounds
.   xu - upper bounds
*/
PetscErrorCode FormBounds(SNES snes, Vec xl, Vec xu)
{
  AppCtx         *ctx;

  PetscFunctionBeginUser;
  CHKERRQ(SNESGetApplicationContext(snes,&ctx));
  CHKERRQ(VecSet(xl,ctx->lb));
  CHKERRQ(VecSet(xu,ctx->ub));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------- */

/*  FormGradient - Evaluates gradient of f.

    Input Parameters:
.   snes  - the SNES context
.   X     - input vector
.   ptr   - optional user-defined context, as set by SNESSetFunction()

    Output Parameters:
.   G - vector containing the newly evaluated gradient
*/
PetscErrorCode FormGradient(SNES snes, Vec X, Vec G, void *ptr)
{
  AppCtx      *user;
  PetscInt    i,j;
  PetscInt    mx, my;
  PetscScalar hx,hy, hydhx, hxdhy;
  PetscScalar f1,f2,f3,f4,f5,f6,d1,d2,d3,d4,d5,d6,d7,d8,xc,xl,xr,xt,xb,xlt,xrb;
  PetscScalar df1dxc,df2dxc,df3dxc,df4dxc,df5dxc,df6dxc;
  PetscScalar **g, **x;
  PetscInt    xs,xm,ys,ym;
  Vec         localX;
  DM          da;

  PetscFunctionBeginUser;
  CHKERRQ(SNESGetDM(snes,&da));
  CHKERRQ(SNESGetApplicationContext(snes,&user));
  CHKERRQ(DMDAGetInfo(da,PETSC_IGNORE,&mx,&my,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));
  hx   = 1.0/(mx+1);hy=1.0/(my+1); hydhx=hy/hx; hxdhy=hx/hy;

  CHKERRQ(VecSet(G,0.0));

  /* Get local vector */
  CHKERRQ(DMGetLocalVector(da,&localX));
  /* Get ghost points */
  CHKERRQ(DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX));
  CHKERRQ(DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX));
  /* Get pointer to local vector data */
  CHKERRQ(DMDAVecGetArray(da,localX, &x));
  CHKERRQ(DMDAVecGetArray(da,G, &g));

  CHKERRQ(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));
  /* Compute function over the locally owned part of the mesh */
  for (j=ys; j < ys+ym; j++) {
    for (i=xs; i< xs+xm; i++) {

      xc = x[j][i];
      xlt=xrb=xl=xr=xb=xt=xc;

      if (i==0) { /* left side */
        xl  = user->left[j+1];
        xlt = user->left[j+2];
      } else xl = x[j][i-1];

      if (j==0) { /* bottom side */
        xb  = user->bottom[i+1];
        xrb = user->bottom[i+2];
      } else xb = x[j-1][i];

      if (i+1 == mx) { /* right side */
        xr  = user->right[j+1];
        xrb = user->right[j];
      } else xr = x[j][i+1];

      if (j+1==0+my) { /* top side */
        xt  = user->top[i+1];
        xlt = user->top[i];
      } else xt = x[j+1][i];

      if (i>0 && j+1<my) xlt = x[j+1][i-1]; /* left top side */
      if (j>0 && i+1<mx) xrb = x[j-1][i+1]; /* right bottom */

      d1 = (xc-xl);
      d2 = (xc-xr);
      d3 = (xc-xt);
      d4 = (xc-xb);
      d5 = (xr-xrb);
      d6 = (xrb-xb);
      d7 = (xlt-xl);
      d8 = (xt-xlt);

      df1dxc = d1*hydhx;
      df2dxc = (d1*hydhx + d4*hxdhy);
      df3dxc = d3*hxdhy;
      df4dxc = (d2*hydhx + d3*hxdhy);
      df5dxc = d2*hydhx;
      df6dxc = d4*hxdhy;

      d1 /= hx;
      d2 /= hx;
      d3 /= hy;
      d4 /= hy;
      d5 /= hy;
      d6 /= hx;
      d7 /= hy;
      d8 /= hx;

      f1 = PetscSqrtScalar(1.0 + d1*d1 + d7*d7);
      f2 = PetscSqrtScalar(1.0 + d1*d1 + d4*d4);
      f3 = PetscSqrtScalar(1.0 + d3*d3 + d8*d8);
      f4 = PetscSqrtScalar(1.0 + d3*d3 + d2*d2);
      f5 = PetscSqrtScalar(1.0 + d2*d2 + d5*d5);
      f6 = PetscSqrtScalar(1.0 + d4*d4 + d6*d6);

      df1dxc /= f1;
      df2dxc /= f2;
      df3dxc /= f3;
      df4dxc /= f4;
      df5dxc /= f5;
      df6dxc /= f6;

      g[j][i] = (df1dxc+df2dxc+df3dxc+df4dxc+df5dxc+df6dxc)/2.0;

    }
  }

  /* Restore vectors */
  CHKERRQ(DMDAVecRestoreArray(da,localX, &x));
  CHKERRQ(DMDAVecRestoreArray(da,G, &g));
  CHKERRQ(DMRestoreLocalVector(da,&localX));
  CHKERRQ(PetscLogFlops(67.0*mx*my));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
   FormJacobian - Evaluates Jacobian matrix.

   Input Parameters:
.  snes - SNES context
.  X    - input vector
.  ptr  - optional user-defined context, as set by SNESSetJacobian()

   Output Parameters:
.  tH    - Jacobian matrix

*/
PetscErrorCode FormJacobian(SNES snes, Vec X, Mat H, Mat tHPre, void *ptr)
{
  AppCtx         *user;
  PetscInt       i,j,k;
  PetscInt       mx, my;
  MatStencil     row,col[7];
  PetscScalar    hx, hy, hydhx, hxdhy;
  PetscScalar    f1,f2,f3,f4,f5,f6,d1,d2,d3,d4,d5,d6,d7,d8,xc,xl,xr,xt,xb,xlt,xrb;
  PetscScalar    hl,hr,ht,hb,hc,htl,hbr;
  PetscScalar    **x, v[7];
  PetscBool      assembled;
  PetscInt       xs,xm,ys,ym;
  Vec            localX;
  DM             da;

  PetscFunctionBeginUser;
  CHKERRQ(SNESGetDM(snes,&da));
  CHKERRQ(SNESGetApplicationContext(snes,&user));
  CHKERRQ(DMDAGetInfo(da,PETSC_IGNORE,&mx,&my,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));
  hx   = 1.0/(mx+1); hy=1.0/(my+1); hydhx=hy/hx; hxdhy=hx/hy;

/* Set various matrix options */
  CHKERRQ(MatAssembled(H,&assembled));
  if (assembled) CHKERRQ(MatZeroEntries(H));

  /* Get local vector */
  CHKERRQ(DMGetLocalVector(da,&localX));
  /* Get ghost points */
  CHKERRQ(DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX));
  CHKERRQ(DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX));

  /* Get pointers to vector data */
  CHKERRQ(DMDAVecGetArray(da,localX, &x));

  CHKERRQ(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));
  /* Compute Jacobian over the locally owned part of the mesh */
  for (j=ys; j< ys+ym; j++) {
    for (i=xs; i< xs+xm; i++) {
      xc = x[j][i];
      xlt=xrb=xl=xr=xb=xt=xc;

      /* Left */
      if (i==0) {
        xl  = user->left[j+1];
        xlt = user->left[j+2];
      } else xl = x[j][i-1];

      /* Bottom */
      if (j==0) {
        xb  =user->bottom[i+1];
        xrb = user->bottom[i+2];
      } else xb = x[j-1][i];

      /* Right */
      if (i+1 == mx) {
        xr  =user->right[j+1];
        xrb = user->right[j];
      } else xr = x[j][i+1];

      /* Top */
      if (j+1==my) {
        xt  =user->top[i+1];
        xlt = user->top[i];
      } else xt = x[j+1][i];

      /* Top left */
      if (i>0 && j+1<my) xlt = x[j+1][i-1];

      /* Bottom right */
      if (j>0 && i+1<mx) xrb = x[j-1][i+1];

      d1 = (xc-xl)/hx;
      d2 = (xc-xr)/hx;
      d3 = (xc-xt)/hy;
      d4 = (xc-xb)/hy;
      d5 = (xrb-xr)/hy;
      d6 = (xrb-xb)/hx;
      d7 = (xlt-xl)/hy;
      d8 = (xlt-xt)/hx;

      f1 = PetscSqrtScalar(1.0 + d1*d1 + d7*d7);
      f2 = PetscSqrtScalar(1.0 + d1*d1 + d4*d4);
      f3 = PetscSqrtScalar(1.0 + d3*d3 + d8*d8);
      f4 = PetscSqrtScalar(1.0 + d3*d3 + d2*d2);
      f5 = PetscSqrtScalar(1.0 + d2*d2 + d5*d5);
      f6 = PetscSqrtScalar(1.0 + d4*d4 + d6*d6);

      hl = (-hydhx*(1.0+d7*d7)+d1*d7)/(f1*f1*f1)+
           (-hydhx*(1.0+d4*d4)+d1*d4)/(f2*f2*f2);
      hr = (-hydhx*(1.0+d5*d5)+d2*d5)/(f5*f5*f5)+
           (-hydhx*(1.0+d3*d3)+d2*d3)/(f4*f4*f4);
      ht = (-hxdhy*(1.0+d8*d8)+d3*d8)/(f3*f3*f3)+
           (-hxdhy*(1.0+d2*d2)+d2*d3)/(f4*f4*f4);
      hb = (-hxdhy*(1.0+d6*d6)+d4*d6)/(f6*f6*f6)+
           (-hxdhy*(1.0+d1*d1)+d1*d4)/(f2*f2*f2);

      hbr = -d2*d5/(f5*f5*f5) - d4*d6/(f6*f6*f6);
      htl = -d1*d7/(f1*f1*f1) - d3*d8/(f3*f3*f3);

      hc = hydhx*(1.0+d7*d7)/(f1*f1*f1) + hxdhy*(1.0+d8*d8)/(f3*f3*f3) +
           hydhx*(1.0+d5*d5)/(f5*f5*f5) + hxdhy*(1.0+d6*d6)/(f6*f6*f6) +
           (hxdhy*(1.0+d1*d1)+hydhx*(1.0+d4*d4)-2.0*d1*d4)/(f2*f2*f2) +
           (hxdhy*(1.0+d2*d2)+hydhx*(1.0+d3*d3)-2.0*d2*d3)/(f4*f4*f4);

      hl/=2.0; hr/=2.0; ht/=2.0; hb/=2.0; hbr/=2.0; htl/=2.0;  hc/=2.0;

      k     =0;
      row.i = i;row.j= j;
      /* Bottom */
      if (j>0) {
        v[k]     =hb;
        col[k].i = i; col[k].j=j-1; k++;
      }

      /* Bottom right */
      if (j>0 && i < mx -1) {
        v[k]     =hbr;
        col[k].i = i+1; col[k].j = j-1; k++;
      }

      /* left */
      if (i>0) {
        v[k]     = hl;
        col[k].i = i-1; col[k].j = j; k++;
      }

      /* Centre */
      v[k]= hc; col[k].i= row.i; col[k].j = row.j; k++;

      /* Right */
      if (i < mx-1) {
        v[k]    = hr;
        col[k].i= i+1; col[k].j = j;k++;
      }

      /* Top left */
      if (i>0 && j < my-1) {
        v[k]     = htl;
        col[k].i = i-1;col[k].j = j+1; k++;
      }

      /* Top */
      if (j < my-1) {
        v[k]     = ht;
        col[k].i = i; col[k].j = j+1; k++;
      }

      CHKERRQ(MatSetValuesStencil(H,1,&row,k,col,v,INSERT_VALUES));
    }
  }

  /* Assemble the matrix */
  CHKERRQ(MatAssemblyBegin(H,MAT_FINAL_ASSEMBLY));
  CHKERRQ(DMDAVecRestoreArray(da,localX,&x));
  CHKERRQ(MatAssemblyEnd(H,MAT_FINAL_ASSEMBLY));
  CHKERRQ(DMRestoreLocalVector(da,&localX));

  CHKERRQ(PetscLogFlops(199.0*mx*my));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
   FormBoundaryConditions -  Calculates the boundary conditions for
   the region.

   Input Parameter:
.  user - user-defined application context

   Output Parameter:
.  user - user-defined application context
*/
PetscErrorCode FormBoundaryConditions(SNES snes,AppCtx **ouser)
{
  PetscInt       i,j,k,limit=0,maxits=5;
  PetscInt       mx,my;
  PetscInt       bsize=0, lsize=0, tsize=0, rsize=0;
  PetscScalar    one  =1.0, two=2.0, three=3.0;
  PetscScalar    det,hx,hy,xt=0,yt=0;
  PetscReal      fnorm, tol=1e-10;
  PetscScalar    u1,u2,nf1,nf2,njac11,njac12,njac21,njac22;
  PetscScalar    b=-0.5, t=0.5, l=-0.5, r=0.5;
  PetscScalar    *boundary;
  AppCtx         *user;
  DM             da;

  PetscFunctionBeginUser;
  CHKERRQ(SNESGetDM(snes,&da));
  CHKERRQ(PetscNew(&user));
  *ouser   = user;
  user->lb = .05;
  user->ub = PETSC_INFINITY;
  CHKERRQ(DMDAGetInfo(da,PETSC_IGNORE,&mx,&my,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  /* Check if lower and upper bounds are set */
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL, "-lb", &user->lb, 0));
  CHKERRQ(PetscOptionsGetScalar(NULL,NULL, "-ub", &user->ub, 0));
  bsize=mx+2; lsize=my+2; rsize=my+2; tsize=mx+2;

  CHKERRQ(PetscMalloc1(bsize, &user->bottom));
  CHKERRQ(PetscMalloc1(tsize, &user->top));
  CHKERRQ(PetscMalloc1(lsize, &user->left));
  CHKERRQ(PetscMalloc1(rsize, &user->right));

  hx= (r-l)/(mx+1.0); hy=(t-b)/(my+1.0);

  for (j=0; j<4; j++) {
    if (j==0) {
      yt       = b;
      xt       = l;
      limit    = bsize;
      boundary = user->bottom;
    } else if (j==1) {
      yt       = t;
      xt       = l;
      limit    = tsize;
      boundary = user->top;
    } else if (j==2) {
      yt       = b;
      xt       = l;
      limit    = lsize;
      boundary = user->left;
    } else { /* if  (j==3) */
      yt       = b;
      xt       = r;
      limit    = rsize;
      boundary = user->right;
    }

    for (i=0; i<limit; i++) {
      u1=xt;
      u2=-yt;
      for (k=0; k<maxits; k++) {
        nf1   = u1 + u1*u2*u2 - u1*u1*u1/three-xt;
        nf2   = -u2 - u1*u1*u2 + u2*u2*u2/three-yt;
        fnorm = PetscRealPart(PetscSqrtScalar(nf1*nf1+nf2*nf2));
        if (fnorm <= tol) break;
        njac11=one+u2*u2-u1*u1;
        njac12=two*u1*u2;
        njac21=-two*u1*u2;
        njac22=-one - u1*u1 + u2*u2;
        det   = njac11*njac22-njac21*njac12;
        u1    = u1-(njac22*nf1-njac12*nf2)/det;
        u2    = u2-(njac11*nf2-njac21*nf1)/det;
      }

      boundary[i]=u1*u1-u2*u2;
      if (j==0 || j==1) xt=xt+hx;
      else yt=yt+hy; /* if (j==2 || j==3) */
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DestroyBoundaryConditions(AppCtx **ouser)
{
  AppCtx         *user = *ouser;

  PetscFunctionBeginUser;
  CHKERRQ(PetscFree(user->bottom));
  CHKERRQ(PetscFree(user->top));
  CHKERRQ(PetscFree(user->left));
  CHKERRQ(PetscFree(user->right));
  CHKERRQ(PetscFree(*ouser));
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------- */
/*
   ComputeInitialGuess - Calculates the initial guess

   Input Parameters:
.  user - user-defined application context
.  X - vector for initial guess

   Output Parameters:
.  X - newly computed initial guess
*/
PetscErrorCode ComputeInitialGuess(SNES snes, Vec X,void *dummy)
{
  PetscInt       i,j,mx,my;
  DM             da;
  AppCtx         *user;
  PetscScalar    **x;
  PetscInt       xs,xm,ys,ym;

  PetscFunctionBeginUser;
  CHKERRQ(SNESGetDM(snes,&da));
  CHKERRQ(SNESGetApplicationContext(snes,&user));

  CHKERRQ(DMDAGetCorners(da,&xs,&ys,NULL,&xm,&ym,NULL));
  CHKERRQ(DMDAGetInfo(da,PETSC_IGNORE,&mx,&my,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE));

  /* Get pointers to vector data */
  CHKERRQ(DMDAVecGetArray(da,X,&x));
  /* Perform local computations */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i< xs+xm; i++) {
      x[j][i] = (((j+1.0)*user->bottom[i+1]+(my-j+1.0)*user->top[i+1])/(my+2.0)+((i+1.0)*user->left[j+1]+(mx-i+1.0)*user->right[j+1])/(mx+2.0))/2.0;
    }
  }
  /* Restore vectors */
  CHKERRQ(DMDAVecRestoreArray(da,X,&x));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      args: -snes_type vinewtonrsls -pc_type mg -ksp_monitor_short -pc_mg_galerkin pmat -da_refine 5 -snes_vi_monitor -pc_mg_type full -snes_max_it 100 -snes_converged_reason
      requires: !single

   test:
      suffix: 2
      args: -snes_type vinewtonssls -pc_type mg -ksp_monitor_short -pc_mg_galerkin pmat -da_refine 5 -snes_vi_monitor -pc_mg_type full -snes_max_it 100 -snes_converged_reason
      requires: !single

TEST*/
