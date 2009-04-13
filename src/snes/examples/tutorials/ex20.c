
static char help[] ="Nonlinear Radiative Transport PDE with multigrid in 3d.\n\
Uses 3-dimensional distributed arrays.\n\
A 3-dim simplified Radiative Transport test problem is used, with analytic Jacobian. \n\
\n\
  Solves the linear systems via multilevel methods \n\
\n\
The command line\n\
options are:\n\
  -tleft <tl>, where <tl> indicates the left Diriclet BC \n\
  -tright <tr>, where <tr> indicates the right Diriclet BC \n\
  -beta <beta>, where <beta> indicates the exponent in T \n\n";

/*T
   Concepts: SNES^solving a system of nonlinear equations
   Concepts: DA^using distributed arrays
   Concepts: multigrid;
   Processors: n
T*/

/*  
  
    This example models the partial differential equation 
   
         - Div(alpha* T^beta (GRAD T)) = 0.
       
    where beta = 2.5 and alpha = 1.0
 
    BC: T_left = 1.0, T_right = 0.1, dT/dn_top = dTdn_bottom = dT/dn_up = dT/dn_down = 0.
    
    in the unit square, which is uniformly discretized in each of x and 
    y in this simple encoding.  The degrees of freedom are cell centered.
 
    A finite volume approximation with the usual 7-point stencil 
    is used to discretize the boundary value problem to obtain a 
    nonlinear system of equations. 

    This code was contributed by Nickolas Jovanovic based on ex18.c
 
*/

#include "petscsnes.h"
#include "petscda.h"
#include "petscmg.h"
#include "petscdmmg.h"

/* User-defined application context */

typedef struct {
   PetscReal   tleft,tright;  /* Dirichlet boundary conditions */
   PetscReal   beta,bm1,coef; /* nonlinear diffusivity parameterizations */
} AppCtx;

#define POWFLOP 5 /* assume a pow() takes five flops */

extern PetscErrorCode FormInitialGuess(DMMG,Vec);
extern PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
extern PetscErrorCode FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DMMG           *dmmg;
  SNES           snes;                      
  AppCtx         user;
  PetscErrorCode ierr;
  PetscInt       its,lits;
  PetscReal      litspit;
  DA             da;

  PetscInitialize(&argc,&argv,PETSC_NULL,help);

  /* set problem parameters */
  user.tleft  = 1.0; 
  user.tright = 0.1;
  user.beta   = 2.5; 
  ierr = PetscOptionsGetReal(PETSC_NULL,"-tleft",&user.tleft,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-tright",&user.tright,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-beta",&user.beta,PETSC_NULL);CHKERRQ(ierr);
  user.bm1  = user.beta - 1.0;
  user.coef = user.beta/2.0;


  /*
      Create the multilevel DA data structure 
  */
  ierr = DMMGCreate(PETSC_COMM_WORLD,2,&user,&dmmg);CHKERRQ(ierr);

  /*
      Set the DA (grid structure) for the grids.
  */
  ierr = DACreate3d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,-5,-5,-5,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,0,&da);CHKERRQ(ierr);
  ierr = DMMGSetDM(dmmg,(DM)da);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);

  /*
     Create the nonlinear solver, and tell the DMMG structure to use it
  */
  ierr = DMMGSetSNES(dmmg,FormFunction,FormJacobian);CHKERRQ(ierr);
  ierr = DMMGSetFromOptions(dmmg);CHKERRQ(ierr);

  /*
      PreLoadBegin() means that the following section of code is run twice. The first time
     through the flag PreLoading is on this the nonlinear solver is only run for a single step.
     The second time through (the actually timed code) the maximum iterations is set to 10
     Preload of the executable is done to eliminate from the timing the time spent bring the 
     executable into memory from disk (paging in).
  */
  PreLoadBegin(PETSC_TRUE,"Solve");
    ierr = DMMGSetInitialGuess(dmmg,FormInitialGuess);CHKERRQ(ierr);
    ierr = DMMGSolve(dmmg);CHKERRQ(ierr);
  PreLoadEnd();
  snes = DMMGGetSNES(dmmg);
  ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
  ierr = SNESGetLinearSolveIterations(snes,&lits);CHKERRQ(ierr);
  litspit = ((PetscReal)lits)/((PetscReal)its);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of Newton iterations = %D\n",its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of Linear iterations = %D\n",lits);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Average Linear its / Newton = %e\n",litspit);CHKERRQ(ierr);

  ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);

  return 0;
}
/* --------------------  Form initial approximation ----------------- */
#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess"
PetscErrorCode FormInitialGuess(DMMG dmmg,Vec X)
{
  AppCtx         *user = (AppCtx*)dmmg->user;
  PetscInt       i,j,k,xs,ys,xm,ym,zs,zm;
  PetscErrorCode ierr;
  PetscReal      tleft = user->tleft;
  PetscScalar    ***x;

  PetscFunctionBegin;

  /* Get ghost points */
  ierr = DAGetCorners((DA)dmmg->dm,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DAVecGetArray((DA)dmmg->dm,X,&x);CHKERRQ(ierr);

  /* Compute initial guess */
  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        x[k][j][i] = tleft;
      }
    }
  }
  ierr = DAVecRestoreArray((DA)dmmg->dm,X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* --------------------  Evaluate Function F(x) --------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormFunction"
PetscErrorCode FormFunction(SNES snes,Vec X,Vec F,void* ptr)
{
  DMMG           dmmg = (DMMG)ptr;
  AppCtx         *user = (AppCtx*)dmmg->user;
  PetscErrorCode ierr;
  PetscInt       i,j,k,mx,my,mz,xs,ys,zs,xm,ym,zm;
  PetscScalar    zero = 0.0,one = 1.0;
  PetscScalar    hx,hy,hz,hxhydhz,hyhzdhx,hzhxdhy;
  PetscScalar    t0,tn,ts,te,tw,an,as,ae,aw,dn,ds,de,dw,fn = 0.0,fs = 0.0,fe =0.0,fw = 0.0;
  PetscScalar    tleft,tright,beta,td,ad,dd,fd,tu,au,du,fu;
  PetscScalar    ***x,***f;
  Vec            localX;

  PetscFunctionBegin;
  ierr = DAGetLocalVector((DA)dmmg->dm,&localX);CHKERRQ(ierr);
  ierr = DAGetInfo((DA)dmmg->dm,PETSC_NULL,&mx,&my,&mz,0,0,0,0,0,0,0);CHKERRQ(ierr);
  hx    = one/(PetscReal)(mx-1);  hy    = one/(PetscReal)(my-1);  hz = one/(PetscReal)(mz-1);
  hxhydhz = hx*hy/hz;   hyhzdhx = hy*hz/hx;   hzhxdhy = hz*hx/hy;
  tleft = user->tleft;         tright = user->tright;
  beta  = user->beta;
 
  /* Get ghost points */
  ierr = DAGlobalToLocalBegin((DA)dmmg->dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd((DA)dmmg->dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGetCorners((DA)dmmg->dm,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DAVecGetArray((DA)dmmg->dm,localX,&x);CHKERRQ(ierr);
  ierr = DAVecGetArray((DA)dmmg->dm,F,&f);CHKERRQ(ierr);

  /* Evaluate function */
  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        t0 = x[k][j][i];

        if (i > 0 && i < mx-1 && j > 0 && j < my-1 && k > 0 && k < mz-1) {

  	  /* general interior volume */

          tw = x[k][j][i-1];
          aw = 0.5*(t0 + tw);                 
          dw = pow(aw,beta);
          fw = dw*(t0 - tw);

       	  te = x[k][j][i+1];
          ae = 0.5*(t0 + te);
          de = pow(ae,beta);
          fe = de*(te - t0);

	  ts = x[k][j-1][i];
          as = 0.5*(t0 + ts);
          ds = pow(as,beta);
          fs = ds*(t0 - ts);
  
          tn = x[k][j+1][i];
          an = 0.5*(t0 + tn);
          dn = pow(an,beta);
          fn = dn*(tn - t0);

          td = x[k-1][j][i];
          ad = 0.5*(t0 + td);
          dd = pow(ad,beta);
          fd = dd*(t0 - td);

          tu = x[k+1][j][i];
          au = 0.5*(t0 + tu);
          du = pow(au,beta);
          fu = du*(tu - t0);

        } else if (i == 0) {

 	  /* left-hand (west) boundary */
          tw = tleft;   
          aw = 0.5*(t0 + tw);                 
          dw = pow(aw,beta);
          fw = dw*(t0 - tw);

	  te = x[k][j][i+1];
          ae = 0.5*(t0 + te);
          de = pow(ae,beta);
          fe = de*(te - t0);

	  if (j > 0) {
	    ts = x[k][j-1][i];
            as = 0.5*(t0 + ts);
            ds = pow(as,beta);
            fs = ds*(t0 - ts);
	  } else {
 	    fs = zero;
	  }

	  if (j < my-1) { 
            tn = x[k][j+1][i];
            an = 0.5*(t0 + tn);
            dn = pow(an,beta);
	    fn = dn*(tn - t0);
	  } else {
	    fn = zero; 
   	  }

	  if (k > 0) {
            td = x[k-1][j][i];  
            ad = 0.5*(t0 + td);
            dd = pow(ad,beta);
            fd = dd*(t0 - td);
	  } else {
 	    fd = zero;
	  }

	  if (k < mz-1) { 
            tu = x[k+1][j][i];  
            au = 0.5*(t0 + tu);
            du = pow(au,beta);
            fu = du*(tu - t0);
	  } else {
 	    fu = zero;
	  }

        } else if (i == mx-1) {

          /* right-hand (east) boundary */ 
          tw = x[k][j][i-1];   
          aw = 0.5*(t0 + tw);
          dw = pow(aw,beta);
          fw = dw*(t0 - tw);
 
          te = tright;
          ae = 0.5*(t0 + te);
          de = pow(ae,beta);
          fe = de*(te - t0);
 
          if (j > 0) { 
            ts = x[k][j-1][i];
            as = 0.5*(t0 + ts);
            ds = pow(as,beta);
            fs = ds*(t0 - ts);
          } else {
            fs = zero;
          }
 
          if (j < my-1) {
            tn = x[k][j+1][i];
            an = 0.5*(t0 + tn);
            dn = pow(an,beta);
            fn = dn*(tn - t0); 
          } else {   
            fn = zero; 
          }

	  if (k > 0) {
            td = x[k-1][j][i];  
            ad = 0.5*(t0 + td);
            dd = pow(ad,beta);
            fd = dd*(t0 - td);
	  } else {
 	    fd = zero;
	  }

	  if (k < mz-1) { 
            tu = x[k+1][j][i];  
            au = 0.5*(t0 + tu);
            du = pow(au,beta);
            fu = du*(tu - t0);
	  } else {
 	    fu = zero;
	  }

        } else if (j == 0) {

	  /* bottom (south) boundary, and i <> 0 or mx-1 */
          tw = x[k][j][i-1];
          aw = 0.5*(t0 + tw);
          dw = pow(aw,beta);
          fw = dw*(t0 - tw);

          te = x[k][j][i+1];
          ae = 0.5*(t0 + te);
          de = pow(ae,beta);
          fe = de*(te - t0);

          fs = zero;

          tn = x[k][j+1][i];
          an = 0.5*(t0 + tn);
          dn = pow(an,beta);
          fn = dn*(tn - t0);

	  if (k > 0) {
            td = x[k-1][j][i];  
            ad = 0.5*(t0 + td);
            dd = pow(ad,beta);
            fd = dd*(t0 - td);
	  } else {
 	    fd = zero;
	  }

	  if (k < mz-1) { 
            tu = x[k+1][j][i];  
            au = 0.5*(t0 + tu);
            du = pow(au,beta);
            fu = du*(tu - t0);
	  } else {
 	    fu = zero;
	  }

        } else if (j == my-1) {

	  /* top (north) boundary, and i <> 0 or mx-1 */ 
          tw = x[k][j][i-1];
          aw = 0.5*(t0 + tw);
          dw = pow(aw,beta);
          fw = dw*(t0 - tw);

          te = x[k][j][i+1];
          ae = 0.5*(t0 + te);
          de = pow(ae,beta);
          fe = de*(te - t0);

          ts = x[k][j-1][i];
          as = 0.5*(t0 + ts);
          ds = pow(as,beta);
          fs = ds*(t0 - ts);

          fn = zero;

	  if (k > 0) {
            td = x[k-1][j][i];  
            ad = 0.5*(t0 + td);
            dd = pow(ad,beta);
            fd = dd*(t0 - td);
	  } else {
 	    fd = zero;
	  }

	  if (k < mz-1) { 
            tu = x[k+1][j][i];  
            au = 0.5*(t0 + tu);
            du = pow(au,beta);
            fu = du*(tu - t0);
	  } else {
 	    fu = zero;
	  }

        } else if (k == 0) {

	  /* down boundary (interior only) */
          tw = x[k][j][i-1];
          aw = 0.5*(t0 + tw);
          dw = pow(aw,beta);
          fw = dw*(t0 - tw);

          te = x[k][j][i+1];
          ae = 0.5*(t0 + te);
          de = pow(ae,beta);
          fe = de*(te - t0);

          ts = x[k][j-1][i];
          as = 0.5*(t0 + ts);
          ds = pow(as,beta);
          fs = ds*(t0 - ts);

          tn = x[k][j+1][i];
          an = 0.5*(t0 + tn);
          dn = pow(an,beta);
          fn = dn*(tn - t0);

 	  fd = zero;

          tu = x[k+1][j][i];  
          au = 0.5*(t0 + tu);
          du = pow(au,beta);
          fu = du*(tu - t0);
	
        } else if (k == mz-1) {

	  /* up boundary (interior only) */
          tw = x[k][j][i-1];
          aw = 0.5*(t0 + tw);
          dw = pow(aw,beta);
          fw = dw*(t0 - tw);

          te = x[k][j][i+1];
          ae = 0.5*(t0 + te);
          de = pow(ae,beta);
          fe = de*(te - t0);

          ts = x[k][j-1][i];
          as = 0.5*(t0 + ts);
          ds = pow(as,beta);
          fs = ds*(t0 - ts);

          tn = x[k][j+1][i];
          an = 0.5*(t0 + tn);
          dn = pow(an,beta);
          fn = dn*(tn - t0);

          td = x[k-1][j][i];  
          ad = 0.5*(t0 + td);
          dd = pow(ad,beta);
          fd = dd*(t0 - td);

          fu = zero;
	}

        f[k][j][i] = - hyhzdhx*(fe-fw) - hzhxdhy*(fn-fs) - hxhydhz*(fu-fd); 
      }
    }
  }
  ierr = DAVecRestoreArray((DA)dmmg->dm,localX,&x);CHKERRQ(ierr);
  ierr = DAVecRestoreArray((DA)dmmg->dm,F,&f);CHKERRQ(ierr);
  ierr = DARestoreLocalVector((DA)dmmg->dm,&localX);CHKERRQ(ierr);
  ierr = PetscLogFlops((22.0 + 4.0*POWFLOP)*ym*xm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 
/* --------------------  Evaluate Jacobian F(x) --------------------- */
#undef __FUNCT__
#define __FUNCT__ "FormJacobian"
PetscErrorCode FormJacobian(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flg,void *ptr)
{
  DMMG           dmmg = (DMMG)ptr;
  AppCtx         *user = (AppCtx*)dmmg->user;
  PetscErrorCode ierr;
  PetscInt       i,j,k,mx,my,mz,xs,ys,zs,xm,ym,zm;
  PetscScalar    one = 1.0;
  PetscScalar    hx,hy,hz,hxhydhz,hyhzdhx,hzhxdhy;
  PetscScalar    t0,tn,ts,te,tw,an,as,ae,aw,dn,ds,de,dw;
  PetscScalar    tleft,tright,beta,td,ad,dd,tu,au,du,v[7],bm1,coef;
  PetscScalar    ***x,bn,bs,be,bw,bu,bd,gn,gs,ge,gw,gu,gd;
  Vec            localX;
  MatStencil     c[7],row;
  Mat            jac = *B;

  PetscFunctionBegin;
  ierr = DAGetLocalVector((DA)dmmg->dm,&localX);CHKERRQ(ierr);
  ierr = DAGetInfo((DA)dmmg->dm,PETSC_NULL,&mx,&my,&mz,0,0,0,0,0,0,0);CHKERRQ(ierr);
  hx    = one/(PetscReal)(mx-1);  hy    = one/(PetscReal)(my-1);  hz = one/(PetscReal)(mz-1);
  hxhydhz = hx*hy/hz;   hyhzdhx = hy*hz/hx;   hzhxdhy = hz*hx/hy;
  tleft = user->tleft;         tright = user->tright;
  beta  = user->beta;	       bm1    = user->bm1;		coef = user->coef;

  /* Get ghost points */
  ierr = DAGlobalToLocalBegin((DA)dmmg->dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd((DA)dmmg->dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGetCorners((DA)dmmg->dm,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DAVecGetArray((DA)dmmg->dm,localX,&x);CHKERRQ(ierr);

  /* Evaluate Jacobian of function */
  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        t0    = x[k][j][i];
        row.k = k; row.j = j; row.i = i;
        if (i > 0 && i < mx-1 && j > 0 && j < my-1 && k > 0 && k < mz-1) {

          /* general interior volume */

          tw = x[k][j][i-1];
          aw = 0.5*(t0 + tw);                 
          bw = pow(aw,bm1);
	  /* dw = bw * aw */
	  dw = pow(aw,beta); 
	  gw = coef*bw*(t0 - tw);

          te = x[k][j][i+1];
          ae = 0.5*(t0 + te);
          be = pow(ae,bm1);
	  /* de = be * ae; */
	  de = pow(ae,beta);
          ge = coef*be*(te - t0);

          ts = x[k][j-1][i];
          as = 0.5*(t0 + ts);
          bs = pow(as,bm1);
	  /* ds = bs * as; */
	  ds = pow(as,beta);
          gs = coef*bs*(t0 - ts);
  
          tn = x[k][j+1][i];
          an = 0.5*(t0 + tn);
          bn = pow(an,bm1);
	  /* dn = bn * an; */
	  dn = pow(an,beta);
          gn = coef*bn*(tn - t0);

          td = x[k-1][j][i];
          ad = 0.5*(t0 + td);
          bd = pow(ad,bm1);
	  /* dd = bd * ad; */
	  dd = pow(ad,beta);
          gd = coef*bd*(t0 - td);
  
          tu = x[k+1][j][i];
          au = 0.5*(t0 + tu);
          bu = pow(au,bm1);
	  /* du = bu * au; */
	  du = pow(au,beta);
          gu = coef*bu*(tu - t0);

	  c[0].k = k-1; c[0].j = j; c[0].i = i; v[0]   = - hxhydhz*(dd - gd); 
	  c[1].k = k; c[1].j = j-1; c[1].i = i;
          v[1]   = - hzhxdhy*(ds - gs); 
	  c[2].k = k; c[2].j = j; c[2].i = i-1;
          v[2]   = - hyhzdhx*(dw - gw); 
	  c[3].k = k; c[3].j = j; c[3].i = i;
          v[3]   =   hzhxdhy*(ds + dn + gs - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + du + gd - gu);
	  c[4].k = k; c[4].j = j; c[4].i = i+1;
          v[4]   = - hyhzdhx*(de + ge);
	  c[5].k = k; c[5].j = j+1; c[5].i = i;
          v[5]   = - hzhxdhy*(dn + gn);
	  c[6].k = k+1; c[6].j = j; c[6].i = i;
          v[6]   = - hxhydhz*(du + gu);
          ierr   =   MatSetValuesStencil(jac,1,&row,7,c,v,INSERT_VALUES);CHKERRQ(ierr);

        } else if (i == 0) {

          /* left-hand plane boundary */
          tw = tleft;
          aw = 0.5*(t0 + tw);                  
          bw = pow(aw,bm1); 
          /* dw = bw * aw */
          dw = pow(aw,beta); 
          gw = coef*bw*(t0 - tw); 
 
          te = x[k][j][i+1];
          ae = 0.5*(t0 + te); 
          be = pow(ae,bm1); 
          /* de = be * ae; */
          de = pow(ae,beta);
          ge = coef*be*(te - t0);
 
          /* left-hand bottom edge */
          if (j == 0) {

            tn = x[k][j+1][i];
            an = 0.5*(t0 + tn); 
            bn = pow(an,bm1); 
            /* dn = bn * an; */
            dn = pow(an,beta);
            gn = coef*bn*(tn - t0); 
          
            /* left-hand bottom down corner */
            if (k == 0) {

              tu = x[k+1][j][i];
              au = 0.5*(t0 + tu); 
              bu = pow(au,bm1); 
              /* du = bu * au; */
              du = pow(au,beta);
              gu = coef*bu*(tu - t0); 
          
	      c[0].k = k; c[0].j = j; c[0].i = i;
              v[0]   =   hzhxdhy*(dn - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(du - gu); 
	      c[1].k = k; c[1].j = j; c[1].i = i+1;
              v[1]   = - hyhzdhx*(de + ge); 
	      c[2].k = k; c[2].j = j+1; c[2].i = i;
              v[2]   = - hzhxdhy*(dn + gn); 
	      c[3].k = k+1; c[3].j = j; c[3].i = i;
              v[3]   = - hxhydhz*(du + gu);
              ierr   =   MatSetValuesStencil(jac,1,&row,4,c,v,INSERT_VALUES);CHKERRQ(ierr);

            /* left-hand bottom interior edge */
            } else if (k < mz-1) {

              tu = x[k+1][j][i];
              au = 0.5*(t0 + tu); 
              bu = pow(au,bm1); 
              /* du = bu * au; */
              du = pow(au,beta);
              gu = coef*bu*(tu - t0); 
          
              td = x[k-1][j][i];
              ad = 0.5*(t0 + td); 
              bd = pow(ad,bm1); 
              /* dd = bd * ad; */
              dd = pow(ad,beta);
              gd = coef*bd*(td - t0); 
          
	      c[0].k = k-1; c[0].j = j; c[0].i = i;
              v[0]   = - hxhydhz*(dd - gd); 
	      c[1].k = k; c[1].j = j; c[1].i = i;
              v[1]   =   hzhxdhy*(dn - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + du + gd - gu); 
	      c[2].k = k; c[2].j = j; c[2].i = i+1;
              v[2]   = - hyhzdhx*(de + ge); 
	      c[3].k = k; c[3].j = j+1; c[3].i = i;
              v[3]   = - hzhxdhy*(dn + gn); 
	      c[4].k = k+1; c[4].j = j; c[4].i = i;
              v[4]   = - hxhydhz*(du + gu);
              ierr   =   MatSetValuesStencil(jac,1,&row,5,c,v,INSERT_VALUES);CHKERRQ(ierr);

            /* left-hand bottom up corner */
            } else {

              td = x[k-1][j][i];
              ad = 0.5*(t0 + td); 
              bd = pow(ad,bm1); 
              /* dd = bd * ad; */
              dd = pow(ad,beta);
              gd = coef*bd*(td - t0); 
          
	      c[0].k = k-1; c[0].j = j; c[0].i = i;
              v[0]   = - hxhydhz*(dd - gd); 
	      c[1].k = k; c[1].j = j; c[1].i = i;
              v[1]   =   hzhxdhy*(dn - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + gd); 
	      c[2].k = k; c[2].j = j; c[2].i = i+1;
              v[2]   = - hyhzdhx*(de + ge); 
	      c[3].k = k; c[3].j = j+1; c[3].i = i;
              v[3]   = - hzhxdhy*(dn + gn); 
              ierr   =   MatSetValuesStencil(jac,1,&row,4,c,v,INSERT_VALUES);CHKERRQ(ierr);
            }

          /* left-hand top edge */
          } else if (j == my-1) {

            ts = x[k][j-1][i];
            as = 0.5*(t0 + ts); 
            bs = pow(as,bm1); 
            /* ds = bs * as; */
            ds = pow(as,beta);
            gs = coef*bs*(ts - t0); 
          
            /* left-hand top down corner */
            if (k == 0) {

              tu = x[k+1][j][i];
              au = 0.5*(t0 + tu); 
              bu = pow(au,bm1); 
              /* du = bu * au; */
              du = pow(au,beta);
              gu = coef*bu*(tu - t0); 
           
	      c[0].k = k; c[0].j = j-1; c[0].i = i;
              v[0]   = - hzhxdhy*(ds - gs); 
	      c[1].k = k; c[1].j = j; c[1].i = i;
              v[1]   =   hzhxdhy*(ds + gs) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(du - gu); 
	      c[2].k = k; c[2].j = j; c[2].i = i+1;
              v[2]   = - hyhzdhx*(de + ge); 
	      c[3].k = k+1; c[3].j = j; c[3].i = i;
              v[3]   = - hxhydhz*(du + gu);
              ierr   =   MatSetValuesStencil(jac,1,&row,4,c,v,INSERT_VALUES);CHKERRQ(ierr);

            /* left-hand top interior edge */
            } else if (k < mz-1) {

              tu = x[k+1][j][i];
              au = 0.5*(t0 + tu); 
              bu = pow(au,bm1); 
              /* du = bu * au; */
              du = pow(au,beta);
              gu = coef*bu*(tu - t0); 
          
              td = x[k-1][j][i];
              ad = 0.5*(t0 + td); 
              bd = pow(ad,bm1); 
              /* dd = bd * ad; */
              dd = pow(ad,beta);
              gd = coef*bd*(td - t0); 
          
	      c[0].k = k-1; c[0].j = j; c[0].i = i;
              v[0]   = - hxhydhz*(dd - gd); 
	      c[1].k = k; c[1].j = j-1; c[1].i = i;
              v[1]   = - hzhxdhy*(ds - gs); 
	      c[2].k = k; c[2].j = j; c[2].i = i;
              v[2]   =   hzhxdhy*(ds + gs) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + du + gd - gu); 
	      c[3].k = k; c[3].j = j; c[3].i = i+1;
              v[3]   = - hyhzdhx*(de + ge); 
	      c[4].k = k+1; c[4].j = j; c[4].i = i;
              v[4]   = - hxhydhz*(du + gu);
              ierr   =   MatSetValuesStencil(jac,1,&row,5,c,v,INSERT_VALUES);CHKERRQ(ierr);

            /* left-hand top up corner */
            } else {

              td = x[k-1][j][i];
              ad = 0.5*(t0 + td); 
              bd = pow(ad,bm1); 
              /* dd = bd * ad; */
              dd = pow(ad,beta);
              gd = coef*bd*(td - t0); 
          
	      c[0].k = k-1; c[0].j = j; c[0].i = i;
              v[0]   = - hxhydhz*(dd - gd); 
	      c[1].k = k; c[1].j = j-1; c[1].i = i;
              v[1]   = - hzhxdhy*(ds - gs); 
	      c[2].k = k; c[2].j = j; c[2].i = i;
              v[2]   =   hzhxdhy*(ds + gs) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + gd); 
	      c[3].k = k; c[3].j = j; c[3].i = i+1;
              v[3]   = - hyhzdhx*(de + ge); 
              ierr   =   MatSetValuesStencil(jac,1,&row,4,c,v,INSERT_VALUES);CHKERRQ(ierr);
            }

          } else {

            ts = x[k][j-1][i];
            as = 0.5*(t0 + ts);
            bs = pow(as,bm1);
            /* ds = bs * as; */
            ds = pow(as,beta);
            gs = coef*bs*(t0 - ts);
  
            tn = x[k][j+1][i];
            an = 0.5*(t0 + tn);
            bn = pow(an,bm1);
            /* dn = bn * an; */
            dn = pow(an,beta);
            gn = coef*bn*(tn - t0);

            /* left-hand down interior edge */
            if (k == 0) {

              tu = x[k+1][j][i];
              au = 0.5*(t0 + tu);
              bu = pow(au,bm1);
              /* du = bu * au; */
              du = pow(au,beta);
              gu = coef*bu*(tu - t0);

	      c[0].k = k; c[0].j = j-1; c[0].i = i;
              v[0]   = - hzhxdhy*(ds - gs); 
	      c[1].k = k; c[1].j = j; c[1].i = i;
              v[1]   =   hzhxdhy*(ds + dn + gs - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(du - gu);
	      c[2].k = k; c[2].j = j; c[2].i = i+1;
              v[2]   = - hyhzdhx*(de + ge); 
	      c[3].k = k; c[3].j = j+1; c[3].i = i;
              v[3]   = - hzhxdhy*(dn + gn); 
	      c[4].k = k+1; c[4].j = j; c[4].i = i;
              v[4]   = - hxhydhz*(du + gu);
              ierr   =   MatSetValuesStencil(jac,1,&row,5,c,v,INSERT_VALUES);CHKERRQ(ierr);
            }

            /* left-hand up interior edge */
            else if (k == mz-1) {

              td = x[k-1][j][i];
              ad = 0.5*(t0 + td);
              bd = pow(ad,bm1);
              /* dd = bd * ad; */
              dd = pow(ad,beta);
              gd = coef*bd*(t0 - td);
  
	      c[0].k = k-1; c[0].j = j; c[0].i = i;
              v[0]   = - hxhydhz*(dd - gd); 
	      c[1].k = k; c[1].j = j-1; c[1].i = i;
              v[1]   = - hzhxdhy*(ds - gs); 
	      c[2].k = k; c[2].j = j; c[2].i = i;
              v[2]   =   hzhxdhy*(ds + dn + gs - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + gd);
	      c[3].k = k; c[3].j = j; c[3].i = i+1;
              v[3]   = - hyhzdhx*(de + ge); 
	      c[4].k = k; c[4].j = j+1; c[4].i = i;
              v[4]   = - hzhxdhy*(dn + gn); 
              ierr   =   MatSetValuesStencil(jac,1,&row,5,c,v,INSERT_VALUES);CHKERRQ(ierr);
            }

            /* left-hand interior plane */
            else {

              td = x[k-1][j][i];
              ad = 0.5*(t0 + td);
              bd = pow(ad,bm1);
              /* dd = bd * ad; */
              dd = pow(ad,beta);
              gd = coef*bd*(t0 - td);
  
              tu = x[k+1][j][i];
              au = 0.5*(t0 + tu);
              bu = pow(au,bm1);
              /* du = bu * au; */
              du = pow(au,beta);
              gu = coef*bu*(tu - t0);

	      c[0].k = k-1; c[0].j = j; c[0].i = i;
              v[0]   = - hxhydhz*(dd - gd); 
	      c[1].k = k; c[1].j = j-1; c[1].i = i;
              v[1]   = - hzhxdhy*(ds - gs); 
	      c[2].k = k; c[2].j = j; c[2].i = i;
              v[2]   =   hzhxdhy*(ds + dn + gs - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + du + gd - gu);
	      c[3].k = k; c[3].j = j; c[3].i = i+1;
              v[3]   = - hyhzdhx*(de + ge); 
	      c[4].k = k; c[4].j = j+1; c[4].i = i;
              v[4]   = - hzhxdhy*(dn + gn); 
	      c[5].k = k+1; c[5].j = j; c[5].i = i;
              v[5]   = - hxhydhz*(du + gu);
              ierr   =   MatSetValuesStencil(jac,1,&row,6,c,v,INSERT_VALUES);CHKERRQ(ierr);
            }
          }

        } else if (i == mx-1) {

          /* right-hand plane boundary */
          tw = x[k][j][i-1];
          aw = 0.5*(t0 + tw);                  
          bw = pow(aw,bm1); 
          /* dw = bw * aw */
          dw = pow(aw,beta); 
          gw = coef*bw*(t0 - tw); 
 
          te = tright; 
          ae = 0.5*(t0 + te); 
          be = pow(ae,bm1); 
          /* de = be * ae; */
          de = pow(ae,beta);
          ge = coef*be*(te - t0);
 
          /* right-hand bottom edge */
          if (j == 0) {

            tn = x[k][j+1][i];
            an = 0.5*(t0 + tn); 
            bn = pow(an,bm1); 
            /* dn = bn * an; */
            dn = pow(an,beta);
            gn = coef*bn*(tn - t0); 
          
            /* right-hand bottom down corner */
            if (k == 0) {

              tu = x[k+1][j][i];
              au = 0.5*(t0 + tu); 
              bu = pow(au,bm1); 
              /* du = bu * au; */
              du = pow(au,beta);
              gu = coef*bu*(tu - t0); 
          
	      c[0].k = k; c[0].j = j; c[0].i = i-1;
              v[0]   = - hyhzdhx*(dw - gw); 
	      c[1].k = k; c[1].j = j; c[1].i = i;
              v[1]   =   hzhxdhy*(dn - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(du - gu); 
	      c[2].k = k; c[2].j = j+1; c[2].i = i;
              v[2]   = - hzhxdhy*(dn + gn); 
	      c[3].k = k+1; c[3].j = j; c[3].i = i;
              v[3]   = - hxhydhz*(du + gu);
              ierr   =   MatSetValuesStencil(jac,1,&row,4,c,v,INSERT_VALUES);CHKERRQ(ierr);

            /* right-hand bottom interior edge */
            } else if (k < mz-1) {

              tu = x[k+1][j][i];
              au = 0.5*(t0 + tu); 
              bu = pow(au,bm1); 
              /* du = bu * au; */
              du = pow(au,beta);
              gu = coef*bu*(tu - t0); 
          
              td = x[k-1][j][i];
              ad = 0.5*(t0 + td); 
              bd = pow(ad,bm1); 
              /* dd = bd * ad; */
              dd = pow(ad,beta);
              gd = coef*bd*(td - t0); 
          
	      c[0].k = k-1; c[0].j = j; c[0].i = i;
              v[0]   = - hxhydhz*(dd - gd); 
	      c[1].k = k; c[1].j = j; c[1].i = i-1;
              v[1]   = - hyhzdhx*(dw - gw); 
	      c[2].k = k; c[2].j = j; c[2].i = i;
              v[2]   =   hzhxdhy*(dn - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + du + gd - gu); 
	      c[3].k = k; c[3].j = j+1; c[3].i = i;
              v[3]   = - hzhxdhy*(dn + gn); 
	      c[4].k = k+1; c[4].j = j; c[4].i = i;
              v[4]   = - hxhydhz*(du + gu);
              ierr   =   MatSetValuesStencil(jac,1,&row,5,c,v,INSERT_VALUES);CHKERRQ(ierr);

            /* right-hand bottom up corner */
            } else {

              td = x[k-1][j][i];
              ad = 0.5*(t0 + td); 
              bd = pow(ad,bm1); 
              /* dd = bd * ad; */
              dd = pow(ad,beta);
              gd = coef*bd*(td - t0); 
          
	      c[0].k = k-1; c[0].j = j; c[0].i = i;
              v[0]   = - hxhydhz*(dd - gd); 
	      c[1].k = k; c[1].j = j; c[1].i = i-1;
              v[1]   = - hyhzdhx*(dw - gw); 
	      c[2].k = k; c[2].j = j; c[2].i = i;
              v[2]   =   hzhxdhy*(dn - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + gd); 
	      c[3].k = k; c[3].j = j+1; c[3].i = i;
              v[3]   = - hzhxdhy*(dn + gn); 
              ierr   =   MatSetValuesStencil(jac,1,&row,4,c,v,INSERT_VALUES);CHKERRQ(ierr);
            }

          /* right-hand top edge */
          } else if (j == my-1) {

            ts = x[k][j-1][i];
            as = 0.5*(t0 + ts); 
            bs = pow(as,bm1); 
            /* ds = bs * as; */
            ds = pow(as,beta);
            gs = coef*bs*(ts - t0); 
          
            /* right-hand top down corner */
            if (k == 0) {

              tu = x[k+1][j][i];
              au = 0.5*(t0 + tu); 
              bu = pow(au,bm1); 
              /* du = bu * au; */
              du = pow(au,beta);
              gu = coef*bu*(tu - t0); 
           
	      c[0].k = k; c[0].j = j-1; c[0].i = i;
              v[0]   = - hzhxdhy*(ds - gs); 
	      c[1].k = k; c[1].j = j; c[1].i = i-1;
              v[1]   = - hyhzdhx*(dw - gw); 
	      c[2].k = k; c[2].j = j; c[2].i = i;
              v[2]   =   hzhxdhy*(ds + gs) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(du - gu); 
	      c[3].k = k+1; c[3].j = j; c[3].i = i;
              v[3]   = - hxhydhz*(du + gu);
              ierr   =   MatSetValuesStencil(jac,1,&row,4,c,v,INSERT_VALUES);CHKERRQ(ierr);

            /* right-hand top interior edge */
            } else if (k < mz-1) {

              tu = x[k+1][j][i];
              au = 0.5*(t0 + tu); 
              bu = pow(au,bm1); 
              /* du = bu * au; */
              du = pow(au,beta);
              gu = coef*bu*(tu - t0); 
          
              td = x[k-1][j][i];
              ad = 0.5*(t0 + td); 
              bd = pow(ad,bm1); 
              /* dd = bd * ad; */
              dd = pow(ad,beta);
              gd = coef*bd*(td - t0); 
          
	      c[0].k = k-1; c[0].j = j; c[0].i = i;
              v[0]   = - hxhydhz*(dd - gd); 
	      c[1].k = k; c[1].j = j-1; c[1].i = i;
              v[1]   = - hzhxdhy*(ds - gs); 
	      c[2].k = k; c[2].j = j; c[2].i = i-1;
              v[2]   = - hyhzdhx*(dw - gw); 
	      c[3].k = k; c[3].j = j; c[3].i = i;
              v[3]   =   hzhxdhy*(ds + gs) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + du + gd - gu); 
	      c[4].k = k+1; c[4].j = j; c[4].i = i;
              v[4]   = - hxhydhz*(du + gu);
              ierr   =   MatSetValuesStencil(jac,1,&row,5,c,v,INSERT_VALUES);CHKERRQ(ierr);

            /* right-hand top up corner */
            } else {

              td = x[k-1][j][i];
              ad = 0.5*(t0 + td); 
              bd = pow(ad,bm1); 
              /* dd = bd * ad; */
              dd = pow(ad,beta);
              gd = coef*bd*(td - t0); 
          
	      c[0].k = k-1; c[0].j = j; c[0].i = i;
              v[0]   = - hxhydhz*(dd - gd); 
	      c[1].k = k; c[1].j = j-1; c[1].i = i;
              v[1]   = - hzhxdhy*(ds - gs); 
	      c[2].k = k; c[2].j = j; c[2].i = i-1;
              v[2]   = - hyhzdhx*(dw - gw); 
	      c[3].k = k; c[3].j = j; c[3].i = i;
              v[3]   =   hzhxdhy*(ds + gs) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + gd); 
              ierr   =   MatSetValuesStencil(jac,1,&row,4,c,v,INSERT_VALUES);CHKERRQ(ierr);
            }

          } else {

            ts = x[k][j-1][i];
            as = 0.5*(t0 + ts);
            bs = pow(as,bm1);
            /* ds = bs * as; */
            ds = pow(as,beta);
            gs = coef*bs*(t0 - ts);
 
            tn = x[k][j+1][i];
            an = 0.5*(t0 + tn);
            bn = pow(an,bm1);
            /* dn = bn * an; */
            dn = pow(an,beta);
            gn = coef*bn*(tn - t0);

            /* right-hand down interior edge */
            if (k == 0) {

              tu = x[k+1][j][i];
              au = 0.5*(t0 + tu);
              bu = pow(au,bm1);
              /* du = bu * au; */
              du = pow(au,beta);
              gu = coef*bu*(tu - t0);

	      c[0].k = k; c[0].j = j-1; c[0].i = i;
              v[0]   = - hzhxdhy*(ds - gs); 
	      c[1].k = k; c[1].j = j; c[1].i = i-1;
              v[1]   = - hyhzdhx*(dw - gw); 
	      c[2].k = k; c[2].j = j; c[2].i = i;
              v[2]   =   hzhxdhy*(ds + dn + gs - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(du - gu);
	      c[3].k = k; c[3].j = j+1; c[3].i = i;
              v[3]   = - hzhxdhy*(dn + gn); 
	      c[4].k = k+1; c[4].j = j; c[4].i = i;
              v[4]   = - hxhydhz*(du + gu);
              ierr   =   MatSetValuesStencil(jac,1,&row,5,c,v,INSERT_VALUES);CHKERRQ(ierr);
            }

            /* right-hand up interior edge */
            else if (k == mz-1) {

              td = x[k-1][j][i];
              ad = 0.5*(t0 + td);
              bd = pow(ad,bm1);
              /* dd = bd * ad; */
              dd = pow(ad,beta);
              gd = coef*bd*(t0 - td);
  
	      c[0].k = k-1; c[0].j = j; c[0].i = i;
              v[0]   = - hxhydhz*(dd - gd); 
	      c[1].k = k; c[1].j = j-1; c[1].i = i;
              v[1]   = - hzhxdhy*(ds - gs); 
	      c[2].k = k; c[2].j = j; c[2].i = i-1;
              v[2]   = - hyhzdhx*(dw - gw); 
	      c[3].k = k; c[3].j = j; c[3].i = i;
              v[3]   =   hzhxdhy*(ds + dn + gs - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + gd);
	      c[4].k = k; c[4].j = j+1; c[4].i = i;
              v[4]   = - hzhxdhy*(dn + gn); 
              ierr   =   MatSetValuesStencil(jac,1,&row,5,c,v,INSERT_VALUES);CHKERRQ(ierr);
            }

            /* right-hand interior plane */
            else {

              td = x[k-1][j][i];
              ad = 0.5*(t0 + td);
              bd = pow(ad,bm1);
              /* dd = bd * ad; */
              dd = pow(ad,beta);
              gd = coef*bd*(t0 - td);
  
              tu = x[k+1][j][i];
              au = 0.5*(t0 + tu);
              bu = pow(au,bm1);
              /* du = bu * au; */
              du = pow(au,beta);
              gu = coef*bu*(tu - t0);

	      c[0].k = k-1; c[0].j = j; c[0].i = i;
              v[0]   = - hxhydhz*(dd - gd); 
	      c[1].k = k; c[1].j = j-1; c[1].i = i;
              v[1]   = - hzhxdhy*(ds - gs); 
	      c[2].k = k; c[2].j = j; c[2].i = i-1;
              v[2]   = - hyhzdhx*(dw - gw); 
	      c[3].k = k; c[3].j = j; c[3].i = i;
              v[3]   =   hzhxdhy*(ds + dn + gs - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + du + gd - gu);
	      c[4].k = k; c[4].j = j+1; c[4].i = i;
              v[4]   = - hzhxdhy*(dn + gn); 
	      c[5].k = k+1; c[5].j = j; c[5].i = i;
              v[5]   = - hxhydhz*(du + gu);
              ierr   =   MatSetValuesStencil(jac,1,&row,6,c,v,INSERT_VALUES);CHKERRQ(ierr);
            }
          }

        } else if (j == 0) {

          tw = x[k][j][i-1];
          aw = 0.5*(t0 + tw);                 
          bw = pow(aw,bm1);
          /* dw = bw * aw */
          dw = pow(aw,beta); 
          gw = coef*bw*(t0 - tw);

          te = x[k][j][i+1];
          ae = 0.5*(t0 + te);
          be = pow(ae,bm1);
          /* de = be * ae; */
          de = pow(ae,beta);
          ge = coef*be*(te - t0);

          tn = x[k][j+1][i];
          an = 0.5*(t0 + tn);
          bn = pow(an,bm1);
          /* dn = bn * an; */
          dn = pow(an,beta);
          gn = coef*bn*(tn - t0);


          /* bottom down interior edge */
          if (k == 0) {

            tu = x[k+1][j][i];
            au = 0.5*(t0 + tu);
            bu = pow(au,bm1);
            /* du = bu * au; */
            du = pow(au,beta);
            gu = coef*bu*(tu - t0);

	    c[0].k = k; c[0].j = j; c[0].i = i-1;
            v[0]   = - hyhzdhx*(dw - gw); 
	    c[1].k = k; c[1].j = j; c[1].i = i;
            v[1]   =   hzhxdhy*(dn - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(du - gu);
	    c[2].k = k; c[2].j = j; c[2].i = i+1;
            v[2]   = - hyhzdhx*(de + ge); 
	    c[3].k = k; c[3].j = j+1; c[3].i = i;
            v[3]   = - hzhxdhy*(dn + gn); 
	    c[4].k = k+1; c[4].j = j; c[4].i = i;
            v[4]   = - hxhydhz*(du + gu);
            ierr   =   MatSetValuesStencil(jac,1,&row,5,c,v,INSERT_VALUES);CHKERRQ(ierr);
          }

          /* bottom up interior edge */
          else if (k == mz-1) {

            td = x[k-1][j][i];
            ad = 0.5*(t0 + td);
            bd = pow(ad,bm1);
            /* dd = bd * ad; */
            dd = pow(ad,beta);
            gd = coef*bd*(td - t0);

	    c[0].k = k-1; c[0].j = j; c[0].i = i;
            v[0]   = - hxhydhz*(dd - gd);
	    c[1].k = k; c[1].j = j; c[1].i = i-1;
            v[1]   = - hyhzdhx*(dw - gw); 
	    c[2].k = k; c[2].j = j; c[2].i = i;
            v[2]   =   hzhxdhy*(dn - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + gd);
	    c[3].k = k; c[3].j = j; c[3].i = i+1;
            v[3]   = - hyhzdhx*(de + ge); 
	    c[4].k = k; c[4].j = j+1; c[4].i = i;
            v[4]   = - hzhxdhy*(dn + gn); 
            ierr   =   MatSetValuesStencil(jac,1,&row,5,c,v,INSERT_VALUES);CHKERRQ(ierr);
          } 

          /* bottom interior plane */
          else {

            tu = x[k+1][j][i];
            au = 0.5*(t0 + tu);
            bu = pow(au,bm1);
            /* du = bu * au; */
            du = pow(au,beta);
            gu = coef*bu*(tu - t0);

            td = x[k-1][j][i];
            ad = 0.5*(t0 + td);
            bd = pow(ad,bm1);
            /* dd = bd * ad; */
            dd = pow(ad,beta);
            gd = coef*bd*(td - t0);

	    c[0].k = k-1; c[0].j = j; c[0].i = i;
            v[0]   = - hxhydhz*(dd - gd);
	    c[1].k = k; c[1].j = j; c[1].i = i-1;
            v[1]   = - hyhzdhx*(dw - gw); 
	    c[2].k = k; c[2].j = j; c[2].i = i;
            v[2]   =   hzhxdhy*(dn - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + du + gd - gu);
	    c[3].k = k; c[3].j = j; c[3].i = i+1;
            v[3]   = - hyhzdhx*(de + ge); 
	    c[4].k = k; c[4].j = j+1; c[4].i = i;
            v[4]   = - hzhxdhy*(dn + gn); 
	    c[5].k = k+1; c[5].j = j; c[5].i = i;
            v[5]   = - hxhydhz*(du + gu);
            ierr   =   MatSetValuesStencil(jac,1,&row,6,c,v,INSERT_VALUES);CHKERRQ(ierr);
          } 

        } else if (j == my-1) {

          tw = x[k][j][i-1];
          aw = 0.5*(t0 + tw);                 
          bw = pow(aw,bm1);
          /* dw = bw * aw */
          dw = pow(aw,beta); 
          gw = coef*bw*(t0 - tw);

          te = x[k][j][i+1];
          ae = 0.5*(t0 + te);
          be = pow(ae,bm1);
          /* de = be * ae; */
          de = pow(ae,beta);
          ge = coef*be*(te - t0);

          ts = x[k][j-1][i];
          as = 0.5*(t0 + ts);
          bs = pow(as,bm1);
          /* ds = bs * as; */
          ds = pow(as,beta);
          gs = coef*bs*(t0 - ts);
  
          /* top down interior edge */
          if (k == 0) {

            tu = x[k+1][j][i];
            au = 0.5*(t0 + tu);
            bu = pow(au,bm1);
            /* du = bu * au; */
            du = pow(au,beta);
            gu = coef*bu*(tu - t0);

	    c[0].k = k; c[0].j = j-1; c[0].i = i;
            v[0]   = - hzhxdhy*(ds - gs); 
	    c[1].k = k; c[1].j = j; c[1].i = i-1;
            v[1]   = - hyhzdhx*(dw - gw); 
	    c[2].k = k; c[2].j = j; c[2].i = i;
            v[2]   =   hzhxdhy*(ds + gs) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(du - gu);
	    c[3].k = k; c[3].j = j; c[3].i = i+1;
            v[3]   = - hyhzdhx*(de + ge); 
	    c[4].k = k+1; c[4].j = j; c[4].i = i;
            v[4]   = - hxhydhz*(du + gu);
            ierr   =   MatSetValuesStencil(jac,1,&row,5,c,v,INSERT_VALUES);CHKERRQ(ierr);
          }

          /* top up interior edge */
          else if (k == mz-1) {

            td = x[k-1][j][i];
            ad = 0.5*(t0 + td);
            bd = pow(ad,bm1);
            /* dd = bd * ad; */
            dd = pow(ad,beta);
            gd = coef*bd*(td - t0);

	    c[0].k = k-1; c[0].j = j; c[0].i = i;
            v[0]   = - hxhydhz*(dd - gd);
	    c[1].k = k; c[1].j = j-1; c[1].i = i;
            v[1]   = - hzhxdhy*(ds - gs); 
	    c[2].k = k; c[2].j = j; c[2].i = i-1;
            v[2]   = - hyhzdhx*(dw - gw); 
	    c[3].k = k; c[3].j = j; c[3].i = i;
            v[3]   =   hzhxdhy*(ds + gs) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + gd);
	    c[4].k = k; c[4].j = j; c[4].i = i+1;
            v[4]   = - hyhzdhx*(de + ge); 
            ierr   =   MatSetValuesStencil(jac,1,&row,5,c,v,INSERT_VALUES);CHKERRQ(ierr);
          }

          /* top interior plane */
          else {

            tu = x[k+1][j][i];
            au = 0.5*(t0 + tu);
            bu = pow(au,bm1);
            /* du = bu * au; */
            du = pow(au,beta);
            gu = coef*bu*(tu - t0);

            td = x[k-1][j][i];
            ad = 0.5*(t0 + td);
            bd = pow(ad,bm1);
            /* dd = bd * ad; */
            dd = pow(ad,beta);
            gd = coef*bd*(td - t0);

	    c[0].k = k-1; c[0].j = j; c[0].i = i;
            v[0]   = - hxhydhz*(dd - gd);
	    c[1].k = k; c[1].j = j-1; c[1].i = i;
            v[1]   = - hzhxdhy*(ds - gs); 
	    c[2].k = k; c[2].j = j; c[2].i = i-1;
            v[2]   = - hyhzdhx*(dw - gw); 
	    c[3].k = k; c[3].j = j; c[3].i = i;
            v[3]   =   hzhxdhy*(ds + gs) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + du + gd - gu);
	    c[4].k = k; c[4].j = j; c[4].i = i+1;
            v[4]   = - hyhzdhx*(de + ge); 
	    c[5].k = k+1; c[5].j = j; c[5].i = i;
            v[5]   = - hxhydhz*(du + gu);
            ierr   =   MatSetValuesStencil(jac,1,&row,6,c,v,INSERT_VALUES);CHKERRQ(ierr);
          } 

        } else if (k == 0) {

          /* down interior plane */

          tw = x[k][j][i-1];
          aw = 0.5*(t0 + tw);                 
          bw = pow(aw,bm1);
          /* dw = bw * aw */
          dw = pow(aw,beta); 
          gw = coef*bw*(t0 - tw);

          te = x[k][j][i+1];
          ae = 0.5*(t0 + te);
          be = pow(ae,bm1);
          /* de = be * ae; */
          de = pow(ae,beta);
          ge = coef*be*(te - t0);

          ts = x[k][j-1][i];
          as = 0.5*(t0 + ts);
          bs = pow(as,bm1);
          /* ds = bs * as; */
          ds = pow(as,beta);
          gs = coef*bs*(t0 - ts);
  
          tn = x[k][j+1][i];
          an = 0.5*(t0 + tn);
          bn = pow(an,bm1);
          /* dn = bn * an; */
          dn = pow(an,beta);
          gn = coef*bn*(tn - t0);
 
          tu = x[k+1][j][i];
          au = 0.5*(t0 + tu);
          bu = pow(au,bm1);
          /* du = bu * au; */
          du = pow(au,beta);
          gu = coef*bu*(tu - t0);

	  c[0].k = k; c[0].j = j-1; c[0].i = i;
          v[0]   = - hzhxdhy*(ds - gs); 
	  c[1].k = k; c[1].j = j; c[1].i = i-1;
          v[1]   = - hyhzdhx*(dw - gw); 
	  c[2].k = k; c[2].j = j; c[2].i = i;
	  v[2]   =   hzhxdhy*(ds + dn + gs - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(du - gu);
	  c[3].k = k; c[3].j = j; c[3].i = i+1;
          v[3]   = - hyhzdhx*(de + ge); 
	  c[4].k = k; c[4].j = j+1; c[4].i = i;
          v[4]   = - hzhxdhy*(dn + gn); 
	  c[5].k = k+1; c[5].j = j; c[5].i = i;
	  v[5]   = - hxhydhz*(du + gu);
          ierr   =   MatSetValuesStencil(jac,1,&row,6,c,v,INSERT_VALUES);CHKERRQ(ierr);
        } 
	
	else if (k == mz-1) {

          /* up interior plane */

          tw = x[k][j][i-1];
          aw = 0.5*(t0 + tw);                 
          bw = pow(aw,bm1);
	  /* dw = bw * aw */
	  dw = pow(aw,beta); 
	  gw = coef*bw*(t0 - tw);

          te = x[k][j][i+1];
          ae = 0.5*(t0 + te);
          be = pow(ae,bm1);
	  /* de = be * ae; */
	  de = pow(ae,beta);
          ge = coef*be*(te - t0);

          ts = x[k][j-1][i];
          as = 0.5*(t0 + ts);
          bs = pow(as,bm1);
	  /* ds = bs * as; */
	  ds = pow(as,beta);
          gs = coef*bs*(t0 - ts);
  
          tn = x[k][j+1][i];
          an = 0.5*(t0 + tn);
          bn = pow(an,bm1);
	  /* dn = bn * an; */
	  dn = pow(an,beta);
          gn = coef*bn*(tn - t0);
 
          td = x[k-1][j][i];
          ad = 0.5*(t0 + td);
          bd = pow(ad,bm1);
	  /* dd = bd * ad; */
	  dd = pow(ad,beta);
          gd = coef*bd*(t0 - td);
  
	  c[0].k = k-1; c[0].j = j; c[0].i = i;
          v[0]   = - hxhydhz*(dd - gd); 
	  c[1].k = k; c[1].j = j-1; c[1].i = i;
          v[1]   = - hzhxdhy*(ds - gs); 
	  c[2].k = k; c[2].j = j; c[2].i = i-1;
          v[2]   = - hyhzdhx*(dw - gw); 
	  c[3].k = k; c[3].j = j; c[3].i = i;
	  v[3]   =   hzhxdhy*(ds + dn + gs - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + gd);
	  c[4].k = k; c[4].j = j; c[4].i = i+1;
          v[4]   = - hyhzdhx*(de + ge); 
	  c[5].k = k; c[5].j = j+1; c[5].i = i;
          v[5]   = - hzhxdhy*(dn + gn); 
          ierr   =   MatSetValuesStencil(jac,1,&row,6,c,v,INSERT_VALUES);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (jac != *J) {
    ierr = MatAssemblyBegin(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*J,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  ierr = DAVecRestoreArray((DA)dmmg->dm,localX,&x);CHKERRQ(ierr);
  ierr = DARestoreLocalVector((DA)dmmg->dm,&localX);CHKERRQ(ierr);

  ierr = PetscLogFlops((41.0 + 8.0*POWFLOP)*xm*ym);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
