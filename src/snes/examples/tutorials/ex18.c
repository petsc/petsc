
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
 
    BC: T_left = 1.0, T_right = 0.1, dT/dn_top = dTdn_bottom = 0.
    
    in the unit square, which is uniformly discretized in each of x and 
    y in this simple encoding.  The degrees of freedom are cell centered.
 
    A finite volume approximation with the usual 5-point stencil 
    is used to discretize the boundary value problem to obtain a 
    nonlinear system of equations. 

    This code was contributed by David Keyes
 
*/

#include "petscsnes.h"
#include "petscda.h"
#include "petscmg.h"
#include "petscdmmg.h"

/* User-defined application context */

typedef struct {
   PetscReal  tleft,tright;  /* Dirichlet boundary conditions */
   PetscReal  beta,bm1,coef; /* nonlinear diffusivity parameterizations */
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
  ierr = DMMGCreate(PETSC_COMM_WORLD,3,&user,&dmmg);CHKERRQ(ierr);

  /*
      Set the DA (grid structure) for the grids.
  */
  ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,5,5,PETSC_DECIDE,PETSC_DECIDE,1,1,0,0,&da);CHKERRQ(ierr);
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
  PetscInt       i,j,xs,ys,xm,ym;
  PetscErrorCode ierr;
  PetscReal      tleft = user->tleft;
  PetscScalar    **x;

  PetscFunctionBegin;

  /* Get ghost points */
  ierr = DAGetCorners((DA)dmmg->dm,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DAVecGetArray((DA)dmmg->dm,X,&x);CHKERRQ(ierr);

  /* Compute initial guess */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      x[j][i] = tleft;
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
  PetscInt       i,j,mx,my,xs,ys,xm,ym;
  PetscScalar    zero = 0.0,one = 1.0;
  PetscScalar    hx,hy,hxdhy,hydhx;
  PetscScalar    t0,tn,ts,te,tw,an,as,ae,aw,dn,ds,de,dw,fn = 0.0,fs = 0.0,fe =0.0,fw = 0.0;
  PetscScalar    tleft,tright,beta;
  PetscScalar    **x,**f;
  Vec            localX;

  PetscFunctionBegin;
  ierr = DAGetLocalVector((DA)dmmg->dm,&localX);CHKERRQ(ierr);
  ierr = DAGetInfo((DA)dmmg->dm,PETSC_NULL,&mx,&my,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  hx    = one/(PetscReal)(mx-1);  hy    = one/(PetscReal)(my-1);
  hxdhy = hx/hy;               hydhx = hy/hx;
  tleft = user->tleft;         tright = user->tright;
  beta  = user->beta;
 
  /* Get ghost points */
  ierr = DAGlobalToLocalBegin((DA)dmmg->dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd((DA)dmmg->dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGetCorners((DA)dmmg->dm,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DAVecGetArray((DA)dmmg->dm,localX,&x);CHKERRQ(ierr);
  ierr = DAVecGetArray((DA)dmmg->dm,F,&f);CHKERRQ(ierr);

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
	} else {
 	  fs = zero;
	}

	if (j < my-1) { 
          tn = x[j+1][i];
          an = 0.5*(t0 + tn);
          dn = PetscPowScalar(an,beta);
	  fn = dn*(tn - t0);
	} else {
	  fn = zero; 
	}

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
        } else {
          fs = zero;
        }
 
        if (j < my-1) {
          tn = x[j+1][i];
          an = 0.5*(t0 + tn);
          dn = PetscPowScalar(an,beta);
          fn = dn*(tn - t0); 
        } else {   
          fn = zero; 
        }

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

      f[j][i] = - hydhx*(fe-fw) - hxdhy*(fn-fs); 

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
  Mat            jac = *J;
  PetscErrorCode ierr;
  PetscInt       i,j,mx,my,xs,ys,xm,ym;
  PetscScalar    one = 1.0,hx,hy,hxdhy,hydhx,t0,tn,ts,te,tw; 
  PetscScalar    dn,ds,de,dw,an,as,ae,aw,bn,bs,be,bw,gn,gs,ge,gw;
  PetscScalar    tleft,tright,beta,bm1,coef;
  PetscScalar    v[5],**x;
  Vec            localX;
  MatStencil     col[5],row;

  PetscFunctionBegin;
  ierr = DAGetLocalVector((DA)dmmg->dm,&localX);CHKERRQ(ierr);
  *flg = SAME_NONZERO_PATTERN;
  ierr = DAGetInfo((DA)dmmg->dm,PETSC_NULL,&mx,&my,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  hx    = one/(PetscReal)(mx-1);  hy     = one/(PetscReal)(my-1);
  hxdhy = hx/hy;               hydhx  = hy/hx;
  tleft = user->tleft;         tright = user->tright;
  beta  = user->beta;	       bm1    = user->bm1;		coef = user->coef;

  /* Get ghost points */
  ierr = DAGlobalToLocalBegin((DA)dmmg->dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd((DA)dmmg->dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGetCorners((DA)dmmg->dm,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DAVecGetArray((DA)dmmg->dm,localX,&x);CHKERRQ(ierr);

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

        v[0] = - hxdhy*(ds - gs);                                      col[0].j = j-1;       col[0].i = i; 
        v[1] = - hydhx*(dw - gw);                                      col[1].j = j;         col[1].i = i-1; 
        v[2] = hxdhy*(ds + dn + gs - gn) + hydhx*(dw + de + gw - ge);  col[2].j = row.j = j; col[2].i = row.i = i;
        v[3] = - hydhx*(de + ge);                                      col[3].j = j;         col[3].i = i+1;   
        v[4] = - hxdhy*(dn + gn);                                      col[4].j = j+1;       col[4].i = i; 
        ierr = MatSetValuesStencil(jac,1,&row,5,col,v,INSERT_VALUES);CHKERRQ(ierr);

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
          v[1] = - hydhx*(de + ge);                           col[1].j = j;         col[1].i = i+1;
          v[2] = - hxdhy*(dn + gn);                           col[2].j = j+1;       col[2].i = i;
          ierr = MatSetValuesStencil(jac,1,&row,3,col,v,INSERT_VALUES);CHKERRQ(ierr);
 
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
          
          v[0] = - hxdhy*(ds - gs);                                      col[0].j = j-1;       col[0].i = i; 
          v[1] = hxdhy*(ds + dn + gs - gn) + hydhx*(dw + de + gw - ge);  col[1].j = row.j = j; col[1].i = row.i = i;
          v[2] = - hydhx*(de + ge);                                      col[2].j = j;         col[2].i = i+1; 
          v[3] = - hxdhy*(dn + gn);                                      col[3].j = j+1;       col[3].i = i; 
          ierr = MatSetValuesStencil(jac,1,&row,4,col,v,INSERT_VALUES);CHKERRQ(ierr);  
	/* left-hand top boundary */
	} else {

          ts = x[j-1][i];
          as = 0.5*(t0 + ts);
          bs = PetscPowScalar(as,bm1);
	  /* ds = bs * as; */
	  ds = PetscPowScalar(as,beta);
          gs = coef*bs*(t0 - ts);
          
          v[0] = - hxdhy*(ds - gs);                            col[0].j = j-1;       col[0].i = i; 
          v[1] = hxdhy*(ds + gs) + hydhx*(dw + de + gw - ge);  col[1].j = row.j = j; col[1].i = row.i = i;
          v[2] = - hydhx*(de + ge);                            col[2].j = j;         col[2].i = i+1; 
          ierr = MatSetValuesStencil(jac,1,&row,3,col,v,INSERT_VALUES);CHKERRQ(ierr); 
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
          
          v[0] = - hydhx*(dw - gw);                           col[0].j = j;         col[0].i = i-1;  
          v[1] = hxdhy*(dn - gn) + hydhx*(dw + de + gw - ge); col[1].j = row.j = j; col[1].i = row.i = i;
          v[2] = - hxdhy*(dn + gn);                           col[2].j = j+1;       col[2].i = i; 
          ierr = MatSetValuesStencil(jac,1,&row,3,col,v,INSERT_VALUES);CHKERRQ(ierr);
 
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
          
          v[0] = - hxdhy*(ds - gs);                                      col[0].j = j-1;       col[0].i = i; 
          v[1] = - hydhx*(dw - gw);                                      col[1].j = j;         col[1].i = i-1; 
          v[2] = hxdhy*(ds + dn + gs - gn) + hydhx*(dw + de + gw - ge);  col[2].j = row.j = j; col[2].i = row.i = i;
          v[3] = - hxdhy*(dn + gn);                                      col[3].j = j+1;       col[3].i = i; 
          ierr = MatSetValuesStencil(jac,1,&row,4,col,v,INSERT_VALUES);CHKERRQ(ierr);  
	/* right-hand top boundary */
	} else {

          ts = x[j-1][i];
          as = 0.5*(t0 + ts);
          bs = PetscPowScalar(as,bm1);
	  /* ds = bs * as; */
	  ds = PetscPowScalar(as,beta);
          gs = coef*bs*(t0 - ts);
          
          v[0] = - hxdhy*(ds - gs);                            col[0].j = j-1;       col[0].i = i; 
          v[1] = - hydhx*(dw - gw);                            col[1].j = j;         col[1].i = i-1; 
          v[2] = hxdhy*(ds + gs) + hydhx*(dw + de + gw - ge);  col[2].j = row.j = j; col[2].i = row.i = i;
          ierr = MatSetValuesStencil(jac,1,&row,3,col,v,INSERT_VALUES);CHKERRQ(ierr); 
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
 
        v[0] = - hydhx*(dw - gw);                           col[0].j = j;         col[0].i = i-1; 
        v[1] = hxdhy*(dn - gn) + hydhx*(dw + de + gw - ge); col[1].j = row.j = j; col[1].i = row.i = i;
        v[2] = - hydhx*(de + ge);                           col[2].j = j;         col[2].i = i+1; 
        v[3] = - hxdhy*(dn + gn);                           col[3].j = j+1;       col[3].i = i; 
        ierr = MatSetValuesStencil(jac,1,&row,4,col,v,INSERT_VALUES);CHKERRQ(ierr);
 
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

        v[0] = - hxdhy*(ds - gs);                            col[0].j = j-1;       col[0].i = i; 
        v[1] = - hydhx*(dw - gw);                            col[1].j = j;         col[1].i = i-1; 
        v[2] = hxdhy*(ds + gs) + hydhx*(dw + de + gw - ge);  col[2].j = row.j = j; col[2].i = row.i = i;
        v[3] = - hydhx*(de + ge);                            col[3].j = j;         col[3].i = i+1; 
        ierr = MatSetValuesStencil(jac,1,&row,4,col,v,INSERT_VALUES);CHKERRQ(ierr);
 
      }
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = DAVecRestoreArray((DA)dmmg->dm,localX,&x);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = DARestoreLocalVector((DA)dmmg->dm,&localX);CHKERRQ(ierr);

  ierr = PetscLogFlops((41.0 + 8.0*POWFLOP)*xm*ym);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

