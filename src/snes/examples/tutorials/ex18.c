/* $Id: ex18.c,v 1.7 2000/09/28 14:41:23 bsmith Exp bsmith $ */

#if !defined(PETSC_USE_COMPLEX)

static char help[] ="Solves nonlinear Radiative Transport PDE with multigrid.\n\
Uses 2-dimensional distributed arrays.\n\
A 2-dim simplified Radiative Transport test problem is used, with analytic Jacobian. \n\
\n\
  Solves the linear systems via multilevel methods \n\
\n\
The command line\n\
options are:\n\
  -tleft <tl>, where <tl> indicates the left Diriclet BC \n\
  -tright <tr>, where <tr> indicates the right Diriclet BC \n\
  -mx <xv>, where <xv> = number of coarse control volumes in the x-direction\n\
  -my <yv>, where <yv> = number of coarse control volumes in the y-direction\n";

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

/* User-defined application context */

typedef struct {
   double      tleft,tright;  /* Dirichlet boundary conditions */
   double      beta,bm1,coef; /* nonlinear diffusivity parameterizations */
} AppCtx;

#define POWFLOP 5 /* assume a pow() takes five flops */

extern int FormInitialGuess(SNES,Vec,void*);
extern int FormFunction(SNES,Vec,Vec,void*);
extern int FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  DMMG          *dmmg;
  SNES          snes;                      
  AppCtx        user;
  int           nlevels,ierr,its,lits,mx,my,ratio;
  double	litspit;

  PetscInitialize(&argc,&argv,PETSC_NULL,help);

  /* set problem parameters */
  user.tleft  = 1.0; 
  user.tright = 0.1;
  user.beta   = 2.5; 
  user.bm1    = 1.5; 
  user.coef   = 1.25;
  ierr = OptionsGetDouble(PETSC_NULL,"-tleft",&user.tleft,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-tright",&user.tright,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-beta",&user.beta,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-bm1",&user.bm1,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetDouble(PETSC_NULL,"-coef",&user.coef,PETSC_NULL);CHKERRA(ierr);

  /* set number of levels and grid size on coarsest level */
  mx              = 5; 
  my              = 5; 
  ratio           = 2;
  nlevels         = 3;
  ierr = OptionsGetInt(PETSC_NULL,"-mx",&mx,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-my",&my,PETSC_NULL);CHKERRA(ierr);

  /*
      Create the multilevel DA data structure 
  */
  ierr = DMMGCreate(PETSC_COMM_WORLD,nlevels,&user,&dmmg);CHKERRQ(ierr);

  /*
      Set the DA (grid structure) for the grids.
  */
  ierr = DMMGSetGrid(dmmg,2,DA_NONPERIODIC,DA_STENCIL_STAR,mx,my,0,1,1);CHKERRQ(ierr);

  /*
     Create the nonlinear solver, and tell the DMMG structure to use it
  */
  ierr = DMMGSetSNES(dmmg,FormFunction,FormJacobian);CHKERRQ(ierr);


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
  ierr = SNESGetIterationNumber(snes,&its);CHKERRA(ierr);
  ierr = SNESGetNumberLinearIterations(snes,&lits);CHKERRA(ierr);
  litspit = ((double)lits)/((double)its);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of Newton iterations = %d\n",its);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of Linear iterations = %d\n",lits);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Average Linear its / Newton = %e\n",litspit);CHKERRA(ierr);

  ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
  PetscFinalize();


  return 0;
}
/* --------------------  Form initial approximation ----------------- */
#undef __FUNC__
#define __FUNC__ "FormInitialGuess"
int FormInitialGuess(SNES snes,Vec X,void *ptr)
{
  DMMG    dmmg = (DMMG)ptr;
  AppCtx  *user = (AppCtx*)dmmg->user;
  int     i,j,row,ierr,xs,ys,xm,ym,Xm,Ym,Xs,Ys;
  double  tleft = user->tleft;
  Scalar  *x;
  Vec     localX;

  PetscFunctionBegin;
  ierr = DAGetLocalVector((DA)dmmg->dm,&localX);CHKERRQ(ierr);

  /* Get ghost points */
  ierr = DAGetCorners((DA)dmmg->dm,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners((DA)dmmg->dm,&Xs,&Ys,0,&Xm,&Ym,0);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);

  /* Compute initial guess */
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      row = i - Xs + (j - Ys)*Xm; 
      x[row] = tleft;
    }
  }
  ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);

  /* Insert values into global vector */
  ierr = DALocalToGlobal((DA)dmmg->dm,localX,INSERT_VALUES,X);CHKERRQ(ierr);
  ierr = DARestoreLocalVector((DA)dmmg->dm,&localX);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* --------------------  Evaluate Function F(x) --------------------- */
#undef __FUNC__
#define __FUNC__ "FormFunction"
int FormFunction(SNES snes,Vec X,Vec F,void* ptr)
{
  DMMG    dmmg = (DMMG)ptr;
  AppCtx  *user = (AppCtx*)dmmg->user;
  int     ierr,i,j,row,mx,my,xs,ys,xm,ym,Xs,Ys,Xm,Ym;
  double  zero = 0.0,one = 1.0;
  double  hx,hy,hxdhy,hydhx;
  double  t0,tn,ts,te,tw,an,as,ae,aw,dn,ds,de,dw,fn = 0.0,fs = 0.0,fe =0.0,fw = 0.0;
  double  tleft,tright,beta;
  Scalar  *x,*f;
  Vec     localX,localF;

  PetscFunctionBegin;
  ierr = DAGetLocalVector((DA)dmmg->dm,&localX);CHKERRQ(ierr);
  ierr = DAGetLocalVector((DA)dmmg->dm,&localF);CHKERRQ(ierr);
  ierr = DAGetInfo((DA)dmmg->dm,PETSC_NULL,&mx,&my,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  hx = one/(double)(mx-1);  hy = one/(double)(my-1);
  hxdhy = hx/hy;            hydhx = hy/hx;
  tleft = user->tleft;      tright = user->tright;
  beta = user->beta;
 
  /* Get ghost points */
  ierr = DAGlobalToLocalBegin((DA)dmmg->dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd((DA)dmmg->dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGetCorners((DA)dmmg->dm,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners((DA)dmmg->dm,&Xs,&Ys,0,&Xm,&Ym,0);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);

  /* Evaluate function */
  for (j=ys; j<ys+ym; j++) {
    row = (j - Ys)*Xm + xs - Xs - 1; 
    for (i=xs; i<xs+xm; i++) {
      row++;
      t0 = x[row];

      if (i > 0 && i < mx-1 && j > 0 && j < my-1) {

	/* general interior volume */

        tw = x[row - 1];   
        aw = 0.5*(t0 + tw);                 
        dw = pow(aw,beta);
        fw = dw*(t0 - tw);

	te = x[row + 1];
        ae = 0.5*(t0 + te);
        de = pow(ae,beta);
        fe = de*(te - t0);

	ts = x[row - Xm];
        as = 0.5*(t0 + ts);
        ds = pow(as,beta);
        fs = ds*(t0 - ts);
  
        tn = x[row + Xm];  
        an = 0.5*(t0 + tn);
        dn = pow(an,beta);
        fn = dn*(tn - t0);

      } else if (i == 0) {

	/* left-hand boundary */
        tw = tleft;   
        aw = 0.5*(t0 + tw);                 
        dw = pow(aw,beta);
        fw = dw*(t0 - tw);

	te = x[row + 1];
        ae = 0.5*(t0 + te);
        de = pow(ae,beta);
        fe = de*(te - t0);

	if (j > 0) {
	  ts = x[row - Xm];
          as = 0.5*(t0 + ts);
          ds = pow(as,beta);
          fs = ds*(t0 - ts);
	} else {
 	  fs = zero;
	}

	if (j < my-1) { 
          tn = x[row + Xm];  
          an = 0.5*(t0 + tn);
          dn = pow(an,beta);
	  fn = dn*(tn - t0);
	} else {
	  fn = zero; 
	}

      } else if (i == mx-1) {

        /* right-hand boundary */ 
        tw = x[row - 1];   
        aw = 0.5*(t0 + tw);
        dw = pow(aw,beta);
        fw = dw*(t0 - tw);
 
        te = tright;
        ae = 0.5*(t0 + te);
        de = pow(ae,beta);
        fe = de*(te - t0);
 
        if (j > 0) { 
          ts = x[row - Xm];
          as = 0.5*(t0 + ts);
          ds = pow(as,beta);
          fs = ds*(t0 - ts);
        } else {
          fs = zero;
        }
 
        if (j < my-1) {
          tn = x[row + Xm];
          an = 0.5*(t0 + tn);
          dn = pow(an,beta);
          fn = dn*(tn - t0); 
        } else {   
          fn = zero; 
        }

      } else if (j == 0) {

	/* bottom boundary,and i <> 0 or mx-1 */
        tw = x[row - 1];
        aw = 0.5*(t0 + tw);
        dw = pow(aw,beta);
        fw = dw*(t0 - tw);

        te = x[row + 1];
        ae = 0.5*(t0 + te);
        de = pow(ae,beta);
        fe = de*(te - t0);

        fs = zero;

        tn = x[row + Xm];
        an = 0.5*(t0 + tn);
        dn = pow(an,beta);
        fn = dn*(tn - t0);

      } else if (j == my-1) {

	/* top boundary,and i <> 0 or mx-1 */ 
        tw = x[row - 1];
        aw = 0.5*(t0 + tw);
        dw = pow(aw,beta);
        fw = dw*(t0 - tw);

        te = x[row + 1];
        ae = 0.5*(t0 + te);
        de = pow(ae,beta);
        fe = de*(te - t0);

        ts = x[row - Xm];
        as = 0.5*(t0 + ts);
        ds = pow(as,beta);
        fs = ds*(t0 - ts);

        fn = zero;

      }

      f[row] = - hydhx*(fe-fw) - hxdhy*(fn-fs); 

    }
  }
  ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);

  /* Insert values into global vector */
  ierr = DALocalToGlobal((DA)dmmg->dm,localF,INSERT_VALUES,F);CHKERRQ(ierr);
  ierr = DARestoreLocalVector((DA)dmmg->dm,&localX);CHKERRQ(ierr);
  ierr = DARestoreLocalVector((DA)dmmg->dm,&localF);CHKERRQ(ierr);
  ierr = PLogFlops((22 + 4*POWFLOP)*ym*xm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
} 
/* --------------------  Evaluate Jacobian F(x) --------------------- */
#undef __FUNC__
#define __FUNC__ "FormJacobian"
int FormJacobian(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flg,void *ptr)
{
  DMMG    dmmg = (DMMG)ptr;
  AppCtx  *user = (AppCtx*)dmmg->user;
  Mat     jac = *J;
  int     ierr,i,j,row,mx,my,xs,ys,xm,ym,Xs,Ys,Xm,Ym,col[5],nloc,*ltog,grow;
  double  one = 1.0,hx,hy,hxdhy,hydhx,t0,tn,ts,te,tw; 
  double  dn,ds,de,dw,an,as,ae,aw,bn,bs,be,bw,gn,gs,ge,gw;
  double  tleft,tright,beta,bm1,coef;
  Scalar  v[5],*x;
  Vec     localX;

  PetscFunctionBegin;
  ierr = DAGetLocalVector((DA)dmmg->dm,&localX);CHKERRQ(ierr);
  *flg = SAME_NONZERO_PATTERN;
  ierr = DAGetInfo((DA)dmmg->dm,PETSC_NULL,&mx,&my,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  hx = one/(double)(mx-1);  hy = one/(double)(my-1);
  hxdhy = hx/hy;            hydhx = hy/hx;
  tleft = user->tleft;      tright = user->tright;
  beta = user->beta;	    bm1 = user->bm1;		coef = user->coef;

  /* Get ghost points */
  ierr = DAGlobalToLocalBegin((DA)dmmg->dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd((DA)dmmg->dm,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGetCorners((DA)dmmg->dm,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners((DA)dmmg->dm,&Xs,&Ys,0,&Xm,&Ym,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices((DA)dmmg->dm,&nloc,&ltog);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);

  /* Evaluate Jacobian of function */
  for (j=ys; j<ys+ym; j++) {
    row = (j - Ys)*Xm + xs - Xs - 1; 
    for (i=xs; i<xs+xm; i++) {
      row++;
      grow = ltog[row];
      t0 = x[row];

      if (i > 0 && i < mx-1 && j > 0 && j < my-1) {

        /* general interior volume */

        tw = x[row - 1];   
        aw = 0.5*(t0 + tw);                 
        bw = pow(aw,bm1);
	/* dw = bw * aw */
	dw = pow(aw,beta); 
	gw = coef*bw*(t0 - tw);

        te = x[row + 1];
        ae = 0.5*(t0 + te);
        be = pow(ae,bm1);
	/* de = be * ae; */
	de = pow(ae,beta);
        ge = coef*be*(te - t0);

        ts = x[row - Xm];
        as = 0.5*(t0 + ts);
        bs = pow(as,bm1);
	/* ds = bs * as; */
	ds = pow(as,beta);
        gs = coef*bs*(t0 - ts);
  
        tn = x[row + Xm];  
        an = 0.5*(t0 + tn);
        bn = pow(an,bm1);
	/* dn = bn * an; */
	dn = pow(an,beta);
        gn = coef*bn*(tn - t0);

	col[0] = ltog[row - Xm];
        v[0] = - hxdhy*(ds - gs); 
	col[1] = ltog[row - 1];
        v[1] = - hydhx*(dw - gw); 
        col[2] = grow;
        v[2] = hxdhy*(ds + dn + gs - gn) + hydhx*(dw + de + gw - ge); 
	col[3] = ltog[row + 1];
        v[3] = - hydhx*(de + ge); 
	col[4] = ltog[row + Xm];
        v[4] = - hxdhy*(dn + gn); 
        ierr = MatSetValues(jac,1,&grow,5,col,v,INSERT_VALUES);CHKERRQ(ierr);

      } else if (i == 0) {

        /* left-hand boundary */
        tw = tleft;
        aw = 0.5*(t0 + tw);                  
        bw = pow(aw,bm1); 
	/* dw = bw * aw */
	dw = pow(aw,beta); 
        gw = coef*bw*(t0 - tw); 
 
        te = x[row + 1]; 
        ae = 0.5*(t0 + te); 
        be = pow(ae,bm1); 
	/* de = be * ae; */
	de = pow(ae,beta);
        ge = coef*be*(te - t0);
 
	/* left-hand bottom boundary */
	if (j == 0) {

          tn = x[row + Xm];   
          an = 0.5*(t0 + tn); 
          bn = pow(an,bm1); 
	  /* dn = bn * an; */
	  dn = pow(an,beta);
          gn = coef*bn*(tn - t0); 
          
          col[0] = grow;
          v[0] = hxdhy*(dn - gn) + hydhx*(dw + de + gw - ge); 
          col[1] = ltog[row + 1];
          v[1] = - hydhx*(de + ge); 
          col[2] = ltog[row + Xm];
          v[2] = - hxdhy*(dn + gn); 
          ierr = MatSetValues(jac,1,&grow,3,col,v,INSERT_VALUES);CHKERRQ(ierr);
 
	/* left-hand interior boundary */
	} else if (j < my-1) {

          ts = x[row - Xm];    
          as = 0.5*(t0 + ts); 
          bs = pow(as,bm1);  
	  /* ds = bs * as; */
	  ds = pow(as,beta);
          gs = coef*bs*(t0 - ts);  
          
          tn = x[row + Xm];    
          an = 0.5*(t0 + tn); 
          bn = pow(an,bm1);  
	  /* dn = bn * an; */
	  dn = pow(an,beta);
          gn = coef*bn*(tn - t0);  
          
          col[0] = ltog[row - Xm];
          v[0] = - hxdhy*(ds - gs); 
          col[1] = grow; 
          v[1] = hxdhy*(ds + dn + gs - gn) + hydhx*(dw + de + gw - ge);  
          col[2] = ltog[row + 1]; 
          v[2] = - hydhx*(de + ge);  
          col[3] = ltog[row + Xm]; 
          v[3] = - hxdhy*(dn + gn);  
          ierr = MatSetValues(jac,1,&grow,4,col,v,INSERT_VALUES);CHKERRQ(ierr);  
	/* left-hand top boundary */
	} else {

          ts = x[row - Xm];
          as = 0.5*(t0 + ts);
          bs = pow(as,bm1);
	  /* ds = bs * as; */
	  ds = pow(as,beta);
          gs = coef*bs*(t0 - ts);
          
          col[0] = ltog[row - Xm]; 
          v[0] = - hxdhy*(ds - gs);  
          col[1] = grow; 
          v[1] = hxdhy*(ds + gs) + hydhx*(dw + de + gw - ge);  
          col[2] = ltog[row + 1];  
          v[2] = - hydhx*(de + ge); 
          ierr = MatSetValues(jac,1,&grow,3,col,v,INSERT_VALUES);CHKERRQ(ierr); 
	}

      } else if (i == mx-1) {
 
        /* right-hand boundary */
        tw = x[row - 1];
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
 
	/* right-hand bottom boundary */
	if (j == 0) {

          tn = x[row + Xm];   
          an = 0.5*(t0 + tn); 
          bn = pow(an,bm1); 
	  /* dn = bn * an; */
	  dn = pow(an,beta);
          gn = coef*bn*(tn - t0); 
          
          col[0] = ltog[row - 1];
          v[0] = - hydhx*(dw - gw); 
          col[1] = grow;
          v[1] = hxdhy*(dn - gn) + hydhx*(dw + de + gw - ge); 
          col[2] = ltog[row + Xm];
          v[2] = - hxdhy*(dn + gn); 
          ierr = MatSetValues(jac,1,&grow,3,col,v,INSERT_VALUES);CHKERRQ(ierr);
 
	/* right-hand interior boundary */
	} else if (j < my-1) {

          ts = x[row - Xm];    
          as = 0.5*(t0 + ts); 
          bs = pow(as,bm1);  
	  /* ds = bs * as; */
	  ds = pow(as,beta);
          gs = coef*bs*(t0 - ts);  
          
          tn = x[row + Xm];    
          an = 0.5*(t0 + tn); 
          bn = pow(an,bm1);  
	  /* dn = bn * an; */
	  dn = pow(an,beta);
          gn = coef*bn*(tn - t0);  
          
          col[0] = ltog[row - Xm];
          v[0] = - hxdhy*(ds - gs); 
          col[1] = ltog[row - 1]; 
          v[1] = - hydhx*(dw - gw);  
          col[2] = grow; 
          v[2] = hxdhy*(ds + dn + gs - gn) + hydhx*(dw + de + gw - ge);  
          col[3] = ltog[row + Xm]; 
          v[3] = - hxdhy*(dn + gn);  
          ierr = MatSetValues(jac,1,&grow,4,col,v,INSERT_VALUES);CHKERRQ(ierr);  
	/* right-hand top boundary */
	} else {

          ts = x[row - Xm];
          as = 0.5*(t0 + ts);
          bs = pow(as,bm1);
	  /* ds = bs * as; */
	  ds = pow(as,beta);
          gs = coef*bs*(t0 - ts);
          
          col[0] = ltog[row - Xm]; 
          v[0] = - hxdhy*(ds - gs);  
          col[1] = ltog[row - 1];  
          v[1] = - hydhx*(dw - gw); 
          col[2] = grow; 
          v[2] = hxdhy*(ds + gs) + hydhx*(dw + de + gw - ge);  
          ierr = MatSetValues(jac,1,&grow,3,col,v,INSERT_VALUES);CHKERRQ(ierr); 
	}

      /* bottom boundary,and i <> 0 or mx-1 */
      } else if (j == 0) {

        tw = x[row - 1];
        aw = 0.5*(t0 + tw);
        bw = pow(aw,bm1);
	/* dw = bw * aw */
	dw = pow(aw,beta); 
        gw = coef*bw*(t0 - tw);

        te = x[row + 1];
        ae = 0.5*(t0 + te);
        be = pow(ae,bm1);
	/* de = be * ae; */
	de = pow(ae,beta);
        ge = coef*be*(te - t0);

        tn = x[row + Xm];
        an = 0.5*(t0 + tn);
        bn = pow(an,bm1);
	/* dn = bn * an; */
	dn = pow(an,beta);
        gn = coef*bn*(tn - t0);
 
        col[0] = ltog[row - 1];
        v[0] = - hydhx*(dw - gw);
        col[1] = grow;
        v[1] = hxdhy*(dn - gn) + hydhx*(dw + de + gw - ge);
        col[2] = ltog[row + 1];
        v[2] = - hydhx*(de + ge);
        col[3] = ltog[row + Xm];
        v[3] = - hxdhy*(dn + gn);
        ierr = MatSetValues(jac,1,&grow,4,col,v,INSERT_VALUES);CHKERRQ(ierr);
 
      /* top boundary,and i <> 0 or mx-1 */
      } else if (j == my-1) {
 
        tw = x[row - 1];
        aw = 0.5*(t0 + tw);
        bw = pow(aw,bm1);
	/* dw = bw * aw */
	dw = pow(aw,beta); 
        gw = coef*bw*(t0 - tw);

        te = x[row + 1];
        ae = 0.5*(t0 + te);
        be = pow(ae,bm1);
	/* de = be * ae; */
	de = pow(ae,beta);
        ge = coef*be*(te - t0);

        ts = x[row - Xm];
        as = 0.5*(t0 + ts);
        bs = pow(as,bm1);
 	/* ds = bs * as; */
	ds = pow(as,beta);
        gs = coef*bs*(t0 - ts);

        col[0] = ltog[row - Xm];
        v[0] = - hxdhy*(ds - gs);
        col[1] = ltog[row - 1];
        v[1] = - hydhx*(dw - gw);
        col[2] = grow;
        v[2] = hxdhy*(ds + gs) + hydhx*(dw + de + gw - ge);
        col[3] = ltog[row + 1];
        v[3] = - hydhx*(de + ge);
        ierr = MatSetValues(jac,1,&grow,4,col,v,INSERT_VALUES);CHKERRQ(ierr);
 
      }
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = DARestoreLocalVector((DA)dmmg->dm,&localX);CHKERRQ(ierr);

  ierr = PLogFlops((41 + 8*POWFLOP)*xm*ym);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#else

int main(int argc,char **argv)
{
  return 0;
}
#endif
