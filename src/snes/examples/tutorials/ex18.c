/* $Id: ex16.c,v 1.7 2000/05/05 22:18:34 balay Exp $ */

#if !defined(PETSC_USE_COMPLEX)

static char help[] =
"This program demonstrates use of the SNES package to solve systems of\n\
nonlinear equations in parallel, using 2-dimensional distributed arrays.\n\
A 2-dim simplified RT test problem is used, with analytic Jacobian. \n\
\n\
  Solves the linear systems via multilevel methods \n\
\n\
The command line\n\
options are:\n\
  -tleft <tl>, where <tl> indicates the left Diriclet BC \n\
  -tright <tr>, where <tr> indicates the right Diriclet BC \n\
  -Mx <xv>, where <xv> = number of coarse control volumes in the x-direction\n\
  -My <yv>, where <yv> = number of coarse control volumes in the y-direction\n\
  -ratio <r>, where <r> = ratio of fine volumes in each coarse in both x,y\n\
  -Nx <npx>, where <npx> = number of processors in the x-direction\n\
  -Ny <npy>, where <npy> = number of processors in the y-direction\n\n";

/*T
   Concepts: SNES^Solving a system of nonlinear equations (parallel example);
   Concepts: DA^Using distributed arrays
   Concepts: Multigrid;
   Routines: SNESCreate(); SNESSetFunction(); SNESSetJacobian();
   Routines: SNESSolve(); SNESSetFromOptions(); DAView();
   Routines: DACreate2d(); DADestroy(); DACreateGlobalVector(); DACreateLocalVector();
   Routines: DAGetCorners(); DAGetGhostCorners(); DALocalToGlobal();
   Routines: DAGlobalToLocalBegin(); DAGlobalToLocalEnd(); DAGetISLocalToGlobalMapping();
   Routines: DAGetInterpolation(); MatRestrict(); PetscTypeCompare();
   Routines: MGSetLevels(); MGSetType(); MGGetSmoother(); MGSetX(); MGSetRhs(); MGSetR();
   Routines: MGSetInterpolate(); MGSetRestriction(); MGSetResidual()
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

/* User-defined application contexts */

typedef struct {
   Vec        localX,localF;    /* local vectors with ghost region */
   DA         da;
   Vec        x,b,r;            /* global vectors */
   Mat        J;                /* Jacobian on grid */
   SLES       sles;
   Mat        R;                /* R and Rscale are not set on the coarsest grid */
   Vec        Rscale;
} GridCtx;

#define MAX_LEVELS 12

typedef struct {
   GridCtx     grid[MAX_LEVELS];
   int         ratio;            /* ratio of grid lines between grid levels */
   int         nlevels;
   double      tleft,tright;  /* Dirichlet boundary conditions */
   double      beta,bm1,coef;/* nonlinear diffusivity parameterizations */
} AppCtx;

#define POWFLOP 5 /* assume a pow() takes five flops */

extern int FormFunction(SNES,Vec,Vec,void*),FormInitialGuess1(AppCtx*,Vec);
extern int FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  SNES          snes;                      
  AppCtx        user;
  GridCtx       *finegrid;                      
  int           ierr,its,lits,n,Nx = PETSC_DECIDE,Ny = PETSC_DECIDE,nlocal,i,maxit,maxf,mx,my;
  PetscTruth    flag;
  double	atol,rtol,stol,litspit;
  SLES          sles;
  PC            pc;

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
  user.ratio      = 2;
  user.nlevels    = 2;
  mx              = 5; 
  my              = 5; 
  ierr = OptionsGetInt(PETSC_NULL,"-ratio",&user.ratio,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-nlevels",&user.nlevels,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-mx",&mx,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-my",&my,&flag);CHKERRA(ierr);
  if (!flag) { my = mx;}

  /* Set grid size for all finer levels */
  for (i=1; i<user.nlevels; i++) {
  }

  /* set partitioning of domains accross processors */
  ierr = OptionsGetInt(PETSC_NULL,"-Nx",&Nx,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-Ny",&Ny,PETSC_NULL);CHKERRA(ierr);

  /* Set up distributed array for  each level */
  for (i=0; i<user.nlevels; i++) {
    ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,mx,
                      my,Nx,Ny,1,1,PETSC_NULL,PETSC_NULL,&user.grid[i].da);CHKERRA(ierr);
    ierr = DACreateGlobalVector(user.grid[i].da,&user.grid[i].x);CHKERRA(ierr);
    ierr = VecDuplicate(user.grid[i].x,&user.grid[i].r);CHKERRA(ierr);
    ierr = VecDuplicate(user.grid[i].x,&user.grid[i].b);CHKERRA(ierr);
    ierr = DACreateLocalVector(user.grid[i].da,&user.grid[i].localX);CHKERRA(ierr);
    ierr = VecDuplicate(user.grid[i].localX,&user.grid[i].localF);CHKERRA(ierr);
    ierr = VecGetLocalSize(user.grid[i].x,&nlocal);CHKERRA(ierr);
    ierr = VecGetSize(user.grid[i].x,&n);CHKERRA(ierr);
    ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD,nlocal,nlocal,n,n,5,PETSC_NULL,3,PETSC_NULL,&user.grid[i].J);CHKERRA(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Grid %d size %d by %d\n",i,mx,my);CHKERRA(ierr);
    mx = user.ratio*(mx-1)+1; 
    my = user.ratio*(my-1)+1;
  }

  /* Create nonlinear solver */
  ierr = SNESCreate(PETSC_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes);CHKERRA(ierr);

  /* provide user function and Jacobian */
  finegrid = &user.grid[user.nlevels-1];
  ierr = SNESSetFunction(snes,finegrid->b,FormFunction,&user);CHKERRA(ierr);
  ierr = SNESSetJacobian(snes,finegrid->J,finegrid->J,FormJacobian,&user);CHKERRA(ierr);

  /* set multilevel (Schwarz) preconditioner */
  ierr = SNESGetSLES(snes,&sles);CHKERRA(ierr);
  ierr = SLESGetPC(sles,&pc);CHKERRA(ierr);
  ierr = PCSetType(pc,PCMG);CHKERRA(ierr);
  ierr = MGSetLevels(pc,user.nlevels);CHKERRA(ierr);
  ierr = MGSetType(pc,MGADDITIVE);CHKERRA(ierr);

  /* set the work vectors and SLES options for all the levels */
  for (i=0; i<user.nlevels; i++) {
    ierr = MGGetSmoother(pc,i,&user.grid[i].sles);CHKERRA(ierr);
    ierr = SLESSetFromOptions(user.grid[i].sles);CHKERRA(ierr);
    ierr = SLESSetOperators(user.grid[i].sles,user.grid[i].J,user.grid[i].J,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
    ierr = MGSetX(pc,i,user.grid[i].x);CHKERRA(ierr); 
    ierr = MGSetRhs(pc,i,user.grid[i].b);CHKERRA(ierr); 
    ierr = MGSetR(pc,i,user.grid[i].r);CHKERRA(ierr); 
    ierr = MGSetResidual(pc,i,MGDefaultResidual,user.grid[i].J);CHKERRA(ierr);
  }

  /* Create interpolation between the levels */
  for (i=1; i<user.nlevels; i++) {
    ierr = DAGetInterpolation(user.grid[i-1].da,user.grid[i].da,&user.grid[i].R,&user.grid[i].Rscale);CHKERRA(ierr);
    ierr = MGSetInterpolate(pc,i,user.grid[i].R);CHKERRA(ierr);
    ierr = MGSetRestriction(pc,i,user.grid[i].R);CHKERRA(ierr);
  }

  /* Solve 1 Newton iteration of nonlinear system 
     - to preload executable so next solve has accurate timing */
  ierr = SNESSetFromOptions(snes);CHKERRA(ierr);
  ierr = SNESGetTolerances(snes,&atol,&rtol,&stol,&maxit,&maxf);CHKERRA(ierr);
  ierr = SNESSetTolerances(snes,atol,rtol,stol,1,maxf);CHKERRA(ierr);
  ierr = FormInitialGuess1(&user,finegrid->x);CHKERRA(ierr);
  ierr = SNESSolve(snes,finegrid->x,&its);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Pre-load Newton iterations = %d\n",its);CHKERRA(ierr);

  /* Reset options, then solve nonlinear system */
  ierr = SNESSetTolerances(snes,atol,rtol,stol,maxit,maxf);CHKERRA(ierr);
  ierr = FormInitialGuess1(&user,finegrid->x);CHKERRA(ierr);
  ierr = PLogStagePush(1);CHKERRA(ierr);
  ierr = SNESSolve(snes,finegrid->x,&its);CHKERRA(ierr);
  ierr = SNESView(snes,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  ierr = PLogStagePop();CHKERRA(ierr);
  ierr = SNESGetNumberLinearIterations(snes,&lits);CHKERRA(ierr);
  litspit = ((double)lits)/((double)its);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of Newton iterations = %d\n",its);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of Linear iterations = %d\n",lits);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Average Linear its / Newton = %e\n",litspit);CHKERRA(ierr);

  /* Free data structures on the levels */
  for (i=0; i<user.nlevels; i++) {
    ierr = MatDestroy(user.grid[i].J);CHKERRA(ierr);
    ierr = VecDestroy(user.grid[i].x);CHKERRA(ierr);
    ierr = VecDestroy(user.grid[i].r);CHKERRA(ierr);
    ierr = VecDestroy(user.grid[i].b);CHKERRA(ierr);
    ierr = DADestroy(user.grid[i].da);CHKERRA(ierr);
    ierr = VecDestroy(user.grid[i].localX);CHKERRA(ierr);
    ierr = VecDestroy(user.grid[i].localF);CHKERRA(ierr);
  }

  /* Free interpolations between levels */
  for (i=1; i<user.nlevels; i++) {
    ierr = MatDestroy(user.grid[i].R);CHKERRA(ierr); 
    ierr = VecDestroy(user.grid[i].Rscale);CHKERRA(ierr); 
  }

  /* free nonlinear solver object */
  ierr = SNESDestroy(snes);CHKERRA(ierr);
  PetscFinalize();


  return 0;
}
/* --------------------  Form initial approximation ----------------- */
#undef __FUNC__
#define __FUNC__ "FormInitialGuess1"
int FormInitialGuess1(AppCtx *user,Vec X)
{
  int     i,j,row,ierr,xs,ys,xm,ym,Xm,Ym,Xs,Ys;
  double  tleft = user->tleft;
  Scalar  *x;
  GridCtx *finegrid = &user->grid[user->nlevels-1];
  Vec     localX = finegrid->localX;

  PetscFunctionBegin;

  /* Get ghost points */
  ierr = DAGetCorners(finegrid->da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(finegrid->da,&Xs,&Ys,0,&Xm,&Ym,0);CHKERRQ(ierr);
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
  ierr = DALocalToGlobal(finegrid->da,localX,INSERT_VALUES,X);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/* --------------------  Evaluate Function F(x) --------------------- */
/*
       This ONLY works on the finest grid
*/
#undef __FUNC__
#define __FUNC__ "FormFunction"
int FormFunction(SNES snes,Vec X,Vec F,void *ptr)
{
  AppCtx  *user = (AppCtx*)ptr;
  int     ierr,i,j,row,mx,my,xs,ys,xm,ym,Xs,Ys,Xm,Ym;
  double  zero = 0.0,one = 1.0;
  double  hx,hy,hxdhy,hydhx;
  double  t0,tn,ts,te,tw,an,as,ae,aw,dn,ds,de,dw,fn = 0.0,fs = 0.0,fe =0.0,fw = 0.0;
  double  tleft,tright,beta;
  Scalar  *x,*f;
  GridCtx *finegrid = &user->grid[user->nlevels-1];
  Vec     localX = finegrid->localX,localF = finegrid->localF; 

  PetscFunctionBegin;
  ierr = DAGetInfo(finegrid->da,PETSC_NULL,&mx,&my,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  hx = one/(double)(mx-1);  hy = one/(double)(my-1);
  hxdhy = hx/hy;            hydhx = hy/hx;
  tleft = user->tleft;      tright = user->tright;
  beta = user->beta;
 
  /* Get ghost points */
  ierr = DAGlobalToLocalBegin(finegrid->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(finegrid->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGetCorners(finegrid->da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(finegrid->da,&Xs,&Ys,0,&Xm,&Ym,0);CHKERRQ(ierr);
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
  ierr = DALocalToGlobal(finegrid->da,localF,INSERT_VALUES,F);CHKERRQ(ierr);
  PLogFlops((22 + 4*POWFLOP)*ym*xm);
  PetscFunctionReturn(0);
} 
/* --------------------  Evaluate Jacobian F(x) --------------------- */
/*
      This works on ANY grid
*/
#undef __FUNC__
#define __FUNC__ "FormJacobian_Grid"
int FormJacobian_Grid(AppCtx *user,GridCtx *grid,Vec X,Mat *J,Mat *B)
{
  Mat     jac = *J;
  int     ierr,i,j,row,mx,my,xs,ys,xm,ym,Xs,Ys,Xm,Ym,col[5],nloc,*ltog,grow;
  double  one = 1.0,hx,hy,hxdhy,hydhx,t0,tn,ts,te,tw; 
  double  dn,ds,de,dw,an,as,ae,aw,bn,bs,be,bw,gn,gs,ge,gw;
  double  tleft,tright,beta,bm1,coef;
  Scalar  v[5],*x;
  Vec     localX = grid->localX;

  PetscFunctionBegin;
  ierr = DAGetInfo(grid->da,PETSC_NULL,&mx,&my,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  hx = one/(double)(mx-1);  hy = one/(double)(my-1);
  hxdhy = hx/hy;            hydhx = hy/hx;
  tleft = user->tleft;      tright = user->tright;
  beta = user->beta;	    bm1 = user->bm1;		coef = user->coef;

  /* Get ghost points */
  ierr = DAGlobalToLocalBegin(grid->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(grid->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGetCorners(grid->da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(grid->da,&Xs,&Ys,0,&Xm,&Ym,0);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(grid->da,&nloc,&ltog);CHKERRQ(ierr);
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

  PLogFlops((41 + 8*POWFLOP)*xm*ym);
  PetscFunctionReturn(0);
}

/* --------------------  Evaluate Jacobian F'(x) --------------------- */
/*
      This evaluates the Jacobian on all of the grids. 
      This routine is called from the SNESSolve() code
*/
#undef __FUNC__
#define __FUNC__ "FormJacobian"
int FormJacobian(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  AppCtx     *user = (AppCtx*)ptr;
  int        ierr,i;
  SLES       sles;
  PC         pc;
  GridCtx    *finegrid = &user->grid[user->nlevels-1];
  PetscTruth ismg;

  PetscFunctionBegin;
  *flag = SAME_NONZERO_PATTERN;
  ierr = FormJacobian_Grid(user,finegrid,X,J,B);CHKERRQ(ierr);

  /* create coarse grid jacobian for preconditioner if multigrid is the preconditioner */
  ierr = SNESGetSLES(snes,&sles);CHKERRQ(ierr);
  ierr = SLESGetPC(sles,&pc);CHKERRQ(ierr);
  ierr = PetscTypeCompare((PetscObject)pc,PCMG,&ismg);CHKERRQ(ierr);
  if (ismg) {

    ierr = SLESSetOperators(finegrid->sles,finegrid->J,finegrid->J,SAME_NONZERO_PATTERN);CHKERRA(ierr);

    for (i=user->nlevels-1; i>0; i--) {

      /* restrict X to coarse grid */
      ierr = MatRestrict(user->grid[i].R,X,user->grid[i-1].x);CHKERRQ(ierr);
      X    = user->grid[i-1].x;      

      /* scale to "natural" scaling for that grid */
      ierr = VecPointwiseMult(user->grid[i].Rscale,X,X);CHKERRQ(ierr);

      /* form Jacobian on coarse grid */
      ierr = FormJacobian_Grid(user,&user->grid[i-1],X,&user->grid[i-1].J,&user->grid[i-1].J);CHKERRQ(ierr);
    
      ierr = SLESSetOperators(user->grid[i-1].sles,user->grid[i-1].J,user->grid[i-1].J,SAME_NONZERO_PATTERN);CHKERRA(ierr);
    }
  }
  PetscFunctionReturn(0);
}

#else

int main(int argc,char **argv)
{
  return 0;
}
#endif
