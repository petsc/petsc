
static char help[] ="Nonlinear Radiative Transport PDE with multigrid in 2d.\n\
Uses 2-dimensional distributed arrays.\n\
A 2-dim simplified Radiative Transport test problem is used, with analytic Jacobian. \n\
\n\
  Solves the linear systems via multilevel methods \n\
\n\
The command line\n\
options are:\n\
  -tleft <tl>, where <tl> indicates the left Diriclet BC \n\
  -tright <tr>, where <tr> indicates the right Diriclet BC \n\
  -beta <beta>, where <beta> indicates the exponent in T \n\n";

/*

    This example models the partial differential equation

         - Div(alpha* T^beta (GRAD T)) = 0.

    where beta = 2.5 and alpha = 1.0

    BC: T_left = 1.0, T_right = 0.1, dT/dn_top = dTdn_bottom = 0.

    in the unit square, which is uniformly discretized in each of x and
    y in this simple encoding.  The degrees of freedom are cell centered.

    A finite volume approximation with the usual 5-point stencil
    is used to discretize the boundary value problem to obtain a
    nonlinear system of equations.

    This code was contributed by David Keyes

*/

#include <petscsnes.h>
#include <petscdm.h>
#include <petscdmda.h>

/* User-defined application context */

typedef struct {
  PetscReal tleft,tright;    /* Dirichlet boundary conditions */
  PetscReal beta,bm1,coef;   /* nonlinear diffusivity parameterizations */
} AppCtx;

#define POWFLOP 5 /* assume a pow() takes five flops */

extern PetscErrorCode FormInitialGuess(SNES,Vec,void*);
extern PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
extern PetscErrorCode FormJacobian(SNES,Vec,Mat,Mat,void*);

int main(int argc,char **argv)
{
  SNES           snes;
  AppCtx         user;
  PetscInt       its,lits;
  PetscReal      litspit;
  DM             da;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,NULL,help));

  /* set problem parameters */
  user.tleft  = 1.0;
  user.tright = 0.1;
  user.beta   = 2.5;
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-tleft",&user.tleft,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-tright",&user.tright,NULL));
  PetscCall(PetscOptionsGetReal(NULL,NULL,"-beta",&user.beta,NULL));
  user.bm1    = user.beta - 1.0;
  user.coef   = user.beta/2.0;

  /*
      Create the multilevel DM data structure
  */
  PetscCall(SNESCreate(PETSC_COMM_WORLD,&snes));

  /*
      Set the DMDA (grid structure) for the grids.
  */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,5,5,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMSetApplicationContext(da,&user));
  PetscCall(SNESSetDM(snes,(DM)da));

  /*
     Create the nonlinear solver, and tell it the functions to use
  */
  PetscCall(SNESSetFunction(snes,NULL,FormFunction,&user));
  PetscCall(SNESSetJacobian(snes,NULL,NULL,FormJacobian,&user));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(SNESSetComputeInitialGuess(snes,FormInitialGuess,NULL));

  PetscCall(SNESSolve(snes,NULL,NULL));
  PetscCall(SNESGetIterationNumber(snes,&its));
  PetscCall(SNESGetLinearSolveIterations(snes,&lits));
  litspit = ((PetscReal)lits)/((PetscReal)its);
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Number of SNES iterations = %" PetscInt_FMT "\n",its));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Number of Linear iterations = %" PetscInt_FMT "\n",lits));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Average Linear its / SNES = %e\n",(double)litspit));

  PetscCall(DMDestroy(&da));
  PetscCall(SNESDestroy(&snes));
  PetscCall(PetscFinalize());
  return 0;
}
/* --------------------  Form initial approximation ----------------- */
PetscErrorCode FormInitialGuess(SNES snes,Vec X,void *ctx)
{
  AppCtx         *user;
  PetscInt       i,j,xs,ys,xm,ym;
  PetscReal      tleft;
  PetscScalar    **x;
  DM             da;

  PetscFunctionBeginUser;
  PetscCall(SNESGetDM(snes,&da));
  PetscCall(DMGetApplicationContext(da,&user));
  tleft = user->tleft;
  /* Get ghost points */
  PetscCall(DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0));
  PetscCall(DMDAVecGetArray(da,X,&x));

  /* Compute initial guess */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      x[j][i] = tleft;
    }
  }
  PetscCall(DMDAVecRestoreArray(da,X,&x));
  PetscFunctionReturn(0);
}
/* --------------------  Evaluate Function F(x) --------------------- */
PetscErrorCode FormFunction(SNES snes,Vec X,Vec F,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  PetscInt       i,j,mx,my,xs,ys,xm,ym;
  PetscScalar    zero = 0.0,one = 1.0;
  PetscScalar    hx,hy,hxdhy,hydhx;
  PetscScalar    t0,tn,ts,te,tw,an,as,ae,aw,dn,ds,de,dw,fn = 0.0,fs = 0.0,fe =0.0,fw = 0.0;
  PetscScalar    tleft,tright,beta;
  PetscScalar    **x,**f;
  Vec            localX;
  DM             da;

  PetscFunctionBeginUser;
  PetscCall(SNESGetDM(snes,&da));
  PetscCall(DMGetLocalVector(da,&localX));
  PetscCall(DMDAGetInfo(da,NULL,&mx,&my,0,0,0,0,0,0,0,0,0,0));
  hx    = one/(PetscReal)(mx-1);  hy    = one/(PetscReal)(my-1);
  hxdhy = hx/hy;               hydhx = hy/hx;
  tleft = user->tleft;         tright = user->tright;
  beta  = user->beta;

  /* Get ghost points */
  PetscCall(DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX));
  PetscCall(DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX));
  PetscCall(DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0));
  PetscCall(DMDAVecGetArray(da,localX,&x));
  PetscCall(DMDAVecGetArray(da,F,&f));

  /* Evaluate function */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      t0 = x[j][i];

      if (i > 0 && i < mx-1 && j > 0 && j < my-1) {

        /* general interior volume */

        tw = x[j][i-1];
        aw = 0.5*(t0 + tw);
        dw = PetscPowScalar(aw,beta);
        fw = dw*(t0 - tw);

        te = x[j][i+1];
        ae = 0.5*(t0 + te);
        de = PetscPowScalar(ae,beta);
        fe = de*(te - t0);

        ts = x[j-1][i];
        as = 0.5*(t0 + ts);
        ds = PetscPowScalar(as,beta);
        fs = ds*(t0 - ts);

        tn = x[j+1][i];
        an = 0.5*(t0 + tn);
        dn = PetscPowScalar(an,beta);
        fn = dn*(tn - t0);

      } else if (i == 0) {

        /* left-hand boundary */
        tw = tleft;
        aw = 0.5*(t0 + tw);
        dw = PetscPowScalar(aw,beta);
        fw = dw*(t0 - tw);

        te = x[j][i+1];
        ae = 0.5*(t0 + te);
        de = PetscPowScalar(ae,beta);
        fe = de*(te - t0);

        if (j > 0) {
          ts = x[j-1][i];
          as = 0.5*(t0 + ts);
          ds = PetscPowScalar(as,beta);
          fs = ds*(t0 - ts);
        } else fs = zero;

        if (j < my-1) {
          tn = x[j+1][i];
          an = 0.5*(t0 + tn);
          dn = PetscPowScalar(an,beta);
          fn = dn*(tn - t0);
        } else fn = zero;

      } else if (i == mx-1) {

        /* right-hand boundary */
        tw = x[j][i-1];
        aw = 0.5*(t0 + tw);
        dw = PetscPowScalar(aw,beta);
        fw = dw*(t0 - tw);

        te = tright;
        ae = 0.5*(t0 + te);
        de = PetscPowScalar(ae,beta);
        fe = de*(te - t0);

        if (j > 0) {
          ts = x[j-1][i];
          as = 0.5*(t0 + ts);
          ds = PetscPowScalar(as,beta);
          fs = ds*(t0 - ts);
        } else fs = zero;

        if (j < my-1) {
          tn = x[j+1][i];
          an = 0.5*(t0 + tn);
          dn = PetscPowScalar(an,beta);
          fn = dn*(tn - t0);
        } else fn = zero;

      } else if (j == 0) {

        /* bottom boundary,and i <> 0 or mx-1 */
        tw = x[j][i-1];
        aw = 0.5*(t0 + tw);
        dw = PetscPowScalar(aw,beta);
        fw = dw*(t0 - tw);

        te = x[j][i+1];
        ae = 0.5*(t0 + te);
        de = PetscPowScalar(ae,beta);
        fe = de*(te - t0);

        fs = zero;

        tn = x[j+1][i];
        an = 0.5*(t0 + tn);
        dn = PetscPowScalar(an,beta);
        fn = dn*(tn - t0);

      } else if (j == my-1) {

        /* top boundary,and i <> 0 or mx-1 */
        tw = x[j][i-1];
        aw = 0.5*(t0 + tw);
        dw = PetscPowScalar(aw,beta);
        fw = dw*(t0 - tw);

        te = x[j][i+1];
        ae = 0.5*(t0 + te);
        de = PetscPowScalar(ae,beta);
        fe = de*(te - t0);

        ts = x[j-1][i];
        as = 0.5*(t0 + ts);
        ds = PetscPowScalar(as,beta);
        fs = ds*(t0 - ts);

        fn = zero;

      }

      f[j][i] = -hydhx*(fe-fw) - hxdhy*(fn-fs);

    }
  }
  PetscCall(DMDAVecRestoreArray(da,localX,&x));
  PetscCall(DMDAVecRestoreArray(da,F,&f));
  PetscCall(DMRestoreLocalVector(da,&localX));
  PetscCall(PetscLogFlops((22.0 + 4.0*POWFLOP)*ym*xm));
  PetscFunctionReturn(0);
}
/* --------------------  Evaluate Jacobian F(x) --------------------- */
PetscErrorCode FormJacobian(SNES snes,Vec X,Mat jac,Mat B,void *ptr)
{
  AppCtx         *user = (AppCtx*)ptr;
  PetscInt       i,j,mx,my,xs,ys,xm,ym;
  PetscScalar    one = 1.0,hx,hy,hxdhy,hydhx,t0,tn,ts,te,tw;
  PetscScalar    dn,ds,de,dw,an,as,ae,aw,bn,bs,be,bw,gn,gs,ge,gw;
  PetscScalar    tleft,tright,beta,bm1,coef;
  PetscScalar    v[5],**x;
  Vec            localX;
  MatStencil     col[5],row;
  DM             da;

  PetscFunctionBeginUser;
  PetscCall(SNESGetDM(snes,&da));
  PetscCall(DMGetLocalVector(da,&localX));
  PetscCall(DMDAGetInfo(da,NULL,&mx,&my,0,0,0,0,0,0,0,0,0,0));
  hx    = one/(PetscReal)(mx-1);  hy     = one/(PetscReal)(my-1);
  hxdhy = hx/hy;               hydhx  = hy/hx;
  tleft = user->tleft;         tright = user->tright;
  beta  = user->beta;          bm1    = user->bm1;          coef = user->coef;

  /* Get ghost points */
  PetscCall(DMGlobalToLocalBegin(da,X,INSERT_VALUES,localX));
  PetscCall(DMGlobalToLocalEnd(da,X,INSERT_VALUES,localX));
  PetscCall(DMDAGetCorners(da,&xs,&ys,0,&xm,&ym,0));
  PetscCall(DMDAVecGetArray(da,localX,&x));

  /* Evaluate Jacobian of function */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      t0 = x[j][i];

      if (i > 0 && i < mx-1 && j > 0 && j < my-1) {

        /* general interior volume */

        tw = x[j][i-1];
        aw = 0.5*(t0 + tw);
        bw = PetscPowScalar(aw,bm1);
        /* dw = bw * aw */
        dw = PetscPowScalar(aw,beta);
        gw = coef*bw*(t0 - tw);

        te = x[j][i+1];
        ae = 0.5*(t0 + te);
        be = PetscPowScalar(ae,bm1);
        /* de = be * ae; */
        de = PetscPowScalar(ae,beta);
        ge = coef*be*(te - t0);

        ts = x[j-1][i];
        as = 0.5*(t0 + ts);
        bs = PetscPowScalar(as,bm1);
        /* ds = bs * as; */
        ds = PetscPowScalar(as,beta);
        gs = coef*bs*(t0 - ts);

        tn = x[j+1][i];
        an = 0.5*(t0 + tn);
        bn = PetscPowScalar(an,bm1);
        /* dn = bn * an; */
        dn = PetscPowScalar(an,beta);
        gn = coef*bn*(tn - t0);

        v[0] = -hxdhy*(ds - gs);                                       col[0].j = j-1;       col[0].i = i;
        v[1] = -hydhx*(dw - gw);                                       col[1].j = j;         col[1].i = i-1;
        v[2] = hxdhy*(ds + dn + gs - gn) + hydhx*(dw + de + gw - ge);  col[2].j = row.j = j; col[2].i = row.i = i;
        v[3] = -hydhx*(de + ge);                                       col[3].j = j;         col[3].i = i+1;
        v[4] = -hxdhy*(dn + gn);                                       col[4].j = j+1;       col[4].i = i;
        PetscCall(MatSetValuesStencil(B,1,&row,5,col,v,INSERT_VALUES));

      } else if (i == 0) {

        /* left-hand boundary */
        tw = tleft;
        aw = 0.5*(t0 + tw);
        bw = PetscPowScalar(aw,bm1);
        /* dw = bw * aw */
        dw = PetscPowScalar(aw,beta);
        gw = coef*bw*(t0 - tw);

        te = x[j][i + 1];
        ae = 0.5*(t0 + te);
        be = PetscPowScalar(ae,bm1);
        /* de = be * ae; */
        de = PetscPowScalar(ae,beta);
        ge = coef*be*(te - t0);

        /* left-hand bottom boundary */
        if (j == 0) {

          tn = x[j+1][i];
          an = 0.5*(t0 + tn);
          bn = PetscPowScalar(an,bm1);
          /* dn = bn * an; */
          dn = PetscPowScalar(an,beta);
          gn = coef*bn*(tn - t0);

          v[0] = hxdhy*(dn - gn) + hydhx*(dw + de + gw - ge); col[0].j = row.j = j; col[0].i = row.i = i;
          v[1] = -hydhx*(de + ge);                            col[1].j = j;         col[1].i = i+1;
          v[2] = -hxdhy*(dn + gn);                            col[2].j = j+1;       col[2].i = i;
          PetscCall(MatSetValuesStencil(B,1,&row,3,col,v,INSERT_VALUES));

          /* left-hand interior boundary */
        } else if (j < my-1) {

          ts = x[j-1][i];
          as = 0.5*(t0 + ts);
          bs = PetscPowScalar(as,bm1);
          /* ds = bs * as; */
          ds = PetscPowScalar(as,beta);
          gs = coef*bs*(t0 - ts);

          tn = x[j+1][i];
          an = 0.5*(t0 + tn);
          bn = PetscPowScalar(an,bm1);
          /* dn = bn * an; */
          dn = PetscPowScalar(an,beta);
          gn = coef*bn*(tn - t0);

          v[0] = -hxdhy*(ds - gs);                                       col[0].j = j-1;       col[0].i = i;
          v[1] = hxdhy*(ds + dn + gs - gn) + hydhx*(dw + de + gw - ge);  col[1].j = row.j = j; col[1].i = row.i = i;
          v[2] = -hydhx*(de + ge);                                       col[2].j = j;         col[2].i = i+1;
          v[3] = -hxdhy*(dn + gn);                                       col[3].j = j+1;       col[3].i = i;
          PetscCall(MatSetValuesStencil(B,1,&row,4,col,v,INSERT_VALUES));
          /* left-hand top boundary */
        } else {

          ts = x[j-1][i];
          as = 0.5*(t0 + ts);
          bs = PetscPowScalar(as,bm1);
          /* ds = bs * as; */
          ds = PetscPowScalar(as,beta);
          gs = coef*bs*(t0 - ts);

          v[0] = -hxdhy*(ds - gs);                             col[0].j = j-1;       col[0].i = i;
          v[1] = hxdhy*(ds + gs) + hydhx*(dw + de + gw - ge);  col[1].j = row.j = j; col[1].i = row.i = i;
          v[2] = -hydhx*(de + ge);                             col[2].j = j;         col[2].i = i+1;
          PetscCall(MatSetValuesStencil(B,1,&row,3,col,v,INSERT_VALUES));
        }

      } else if (i == mx-1) {

        /* right-hand boundary */
        tw = x[j][i-1];
        aw = 0.5*(t0 + tw);
        bw = PetscPowScalar(aw,bm1);
        /* dw = bw * aw */
        dw = PetscPowScalar(aw,beta);
        gw = coef*bw*(t0 - tw);

        te = tright;
        ae = 0.5*(t0 + te);
        be = PetscPowScalar(ae,bm1);
        /* de = be * ae; */
        de = PetscPowScalar(ae,beta);
        ge = coef*be*(te - t0);

        /* right-hand bottom boundary */
        if (j == 0) {

          tn = x[j+1][i];
          an = 0.5*(t0 + tn);
          bn = PetscPowScalar(an,bm1);
          /* dn = bn * an; */
          dn = PetscPowScalar(an,beta);
          gn = coef*bn*(tn - t0);

          v[0] = -hydhx*(dw - gw);                            col[0].j = j;         col[0].i = i-1;
          v[1] = hxdhy*(dn - gn) + hydhx*(dw + de + gw - ge); col[1].j = row.j = j; col[1].i = row.i = i;
          v[2] = -hxdhy*(dn + gn);                            col[2].j = j+1;       col[2].i = i;
          PetscCall(MatSetValuesStencil(B,1,&row,3,col,v,INSERT_VALUES));

          /* right-hand interior boundary */
        } else if (j < my-1) {

          ts = x[j-1][i];
          as = 0.5*(t0 + ts);
          bs = PetscPowScalar(as,bm1);
          /* ds = bs * as; */
          ds = PetscPowScalar(as,beta);
          gs = coef*bs*(t0 - ts);

          tn = x[j+1][i];
          an = 0.5*(t0 + tn);
          bn = PetscPowScalar(an,bm1);
          /* dn = bn * an; */
          dn = PetscPowScalar(an,beta);
          gn = coef*bn*(tn - t0);

          v[0] = -hxdhy*(ds - gs);                                       col[0].j = j-1;       col[0].i = i;
          v[1] = -hydhx*(dw - gw);                                       col[1].j = j;         col[1].i = i-1;
          v[2] = hxdhy*(ds + dn + gs - gn) + hydhx*(dw + de + gw - ge);  col[2].j = row.j = j; col[2].i = row.i = i;
          v[3] = -hxdhy*(dn + gn);                                       col[3].j = j+1;       col[3].i = i;
          PetscCall(MatSetValuesStencil(B,1,&row,4,col,v,INSERT_VALUES));
        /* right-hand top boundary */
        } else {

          ts = x[j-1][i];
          as = 0.5*(t0 + ts);
          bs = PetscPowScalar(as,bm1);
          /* ds = bs * as; */
          ds = PetscPowScalar(as,beta);
          gs = coef*bs*(t0 - ts);

          v[0] = -hxdhy*(ds - gs);                             col[0].j = j-1;       col[0].i = i;
          v[1] = -hydhx*(dw - gw);                             col[1].j = j;         col[1].i = i-1;
          v[2] = hxdhy*(ds + gs) + hydhx*(dw + de + gw - ge);  col[2].j = row.j = j; col[2].i = row.i = i;
          PetscCall(MatSetValuesStencil(B,1,&row,3,col,v,INSERT_VALUES));
        }

        /* bottom boundary,and i <> 0 or mx-1 */
      } else if (j == 0) {

        tw = x[j][i-1];
        aw = 0.5*(t0 + tw);
        bw = PetscPowScalar(aw,bm1);
        /* dw = bw * aw */
        dw = PetscPowScalar(aw,beta);
        gw = coef*bw*(t0 - tw);

        te = x[j][i+1];
        ae = 0.5*(t0 + te);
        be = PetscPowScalar(ae,bm1);
        /* de = be * ae; */
        de = PetscPowScalar(ae,beta);
        ge = coef*be*(te - t0);

        tn = x[j+1][i];
        an = 0.5*(t0 + tn);
        bn = PetscPowScalar(an,bm1);
        /* dn = bn * an; */
        dn = PetscPowScalar(an,beta);
        gn = coef*bn*(tn - t0);

        v[0] = -hydhx*(dw - gw);                            col[0].j = j;         col[0].i = i-1;
        v[1] = hxdhy*(dn - gn) + hydhx*(dw + de + gw - ge); col[1].j = row.j = j; col[1].i = row.i = i;
        v[2] = -hydhx*(de + ge);                            col[2].j = j;         col[2].i = i+1;
        v[3] = -hxdhy*(dn + gn);                            col[3].j = j+1;       col[3].i = i;
        PetscCall(MatSetValuesStencil(B,1,&row,4,col,v,INSERT_VALUES));

        /* top boundary,and i <> 0 or mx-1 */
      } else if (j == my-1) {

        tw = x[j][i-1];
        aw = 0.5*(t0 + tw);
        bw = PetscPowScalar(aw,bm1);
        /* dw = bw * aw */
        dw = PetscPowScalar(aw,beta);
        gw = coef*bw*(t0 - tw);

        te = x[j][i+1];
        ae = 0.5*(t0 + te);
        be = PetscPowScalar(ae,bm1);
        /* de = be * ae; */
        de = PetscPowScalar(ae,beta);
        ge = coef*be*(te - t0);

        ts = x[j-1][i];
        as = 0.5*(t0 + ts);
        bs = PetscPowScalar(as,bm1);
        /* ds = bs * as; */
        ds = PetscPowScalar(as,beta);
        gs = coef*bs*(t0 - ts);

        v[0] = -hxdhy*(ds - gs);                             col[0].j = j-1;       col[0].i = i;
        v[1] = -hydhx*(dw - gw);                             col[1].j = j;         col[1].i = i-1;
        v[2] = hxdhy*(ds + gs) + hydhx*(dw + de + gw - ge);  col[2].j = row.j = j; col[2].i = row.i = i;
        v[3] = -hydhx*(de + ge);                             col[3].j = j;         col[3].i = i+1;
        PetscCall(MatSetValuesStencil(B,1,&row,4,col,v,INSERT_VALUES));

      }
    }
  }
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(DMDAVecRestoreArray(da,localX,&x));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  PetscCall(DMRestoreLocalVector(da,&localX));
  if (jac != B) {
    PetscCall(MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY));
  }

  PetscCall(PetscLogFlops((41.0 + 8.0*POWFLOP)*xm*ym));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      suffix: 1
      args: -pc_type mg -ksp_type fgmres -da_refine 2 -pc_mg_galerkin pmat -snes_view
      requires: !single

   test:
      suffix: 2
      args: -pc_type mg -ksp_type fgmres -da_refine 2 -pc_mg_galerkin pmat -snes_view -snes_type newtontrdc
      requires: !single

TEST*/
