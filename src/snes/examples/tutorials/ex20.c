/* $Id: ex20.c,v 1.7 2001/01/15 21:48:06 bsmith Exp bsmith $ */

#if !defined(PETSC_USE_COMPLEX)

static char help[] =
"This program demonstrates use of the SNES package to solve systems of\n\
nonlinear equations in parallel, using 3-dimensional distributed arrays.\n\
A 3-dimensional simplified RT test problem is used, with analytic Jacobian. \n\
\n\
  Solves the linear systems via multilevel methods \n\
\n\
The command line\n\
options are:\n\
  -tleft <tl>, where <tl> indicates the left Dirichlet BC \n\
  -tright <tr>, where <tr> indicates the right Dirichlet BC \n\
  -mx <xv>, where <xv> = number of coarse control volumes in the x-direction\n\
  -my <yv>, where <yv> = number of coarse control volumes in the y-direction\n\
  -mz <zv>, where <zv> = number of coarse control volumes in the z-direction\n\
  -nlevels <nl>, where <nl> = number of multigrid levels\n\
  -ratio <r>, where <r> = ratio of fine volumes in each coarse in x,y,z\n\
  -Nx <npx>, where <npx> = number of processors in the x-direction\n\
  -Ny <npy>, where <npy> = number of processors in the y-direction\n\
  -Nz <npz>, where <npz> = number of processors in the z-direction\n\
  -beta <b>, where <b> = beta\n\
  -bm1 <bminus1>, where <bminus1> = beta - 1\n\
  -coef <c>, where <c> = beta / 2\n\n";

/*T
   Concepts: SNES^solving a system of nonlinear equations (parallel example);
   Concepts: DA^using distributed arrays
   Concepts: multigrid;
   Processors: n
T*/

/*  
  
    This example models the partial differential equation 
   
         - Div(alpha* T^beta (GRAD T)) = 0.
       
    where beta = 2.5 and alpha = 1.0
 
    BC: T_left = 1.0, T_right = 0.1, dT/dn_top = dT/dn_bottom = dT/dn_up = dT/dn_down = 0.
    
    in the unit cube, which is uniformly discretized in each of x, y, and 
    z in this simple encoding.  The degrees of freedom are cell centered.
 
    A finite volume approximation with the usual 7-point stencil 
    is used to discretize the boundary value problem to obtain a 
    nonlinear system of equations. 

    This code was contributed by David Keyes (see tutorial ex16.c) and extended
    to three dimensions by Nickolas Jovanovic.
 
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
  int           ierr,its,lits,n,Nx = PETSC_DECIDE,Ny = PETSC_DECIDE,Nz = PETSC_DECIDE,nlocal,i,maxit,maxf,mx,my,mz;
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
  /* set number of levels and grid size on coarsest level */
  user.ratio      = 2;
  user.nlevels    = 2;
  mx              = 5; 
  my              = 5; 
  mz              = 5; 

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,PETSC_NULL,"Application PetscOptions","None");
  ierr = PetscOptionsDouble("-tleft","left value","Manualpage",user.tleft,&user.tleft,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsDouble("-tright","right value","Manualpage",user.tright,&user.tright,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsDouble("-beta","beta","Manualpage",user.beta,&user.beta,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsDouble("-bm1","bm1","Manualpage",user.bm1,&user.bm1,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsDouble("-coef","coefficient","Manualpage",user.coef,&user.coef,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscOptionsInt("-ratio","grid ration","Manualpage",user.ratio,&user.ratio,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nlevels","number levels","Manualpage",user.nlevels,&user.nlevels,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mx","grid points in x","Manualpage",mx,&mx,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-my","grid points in y","Manualpage",my,&my,&flag);CHKERRQ(ierr);
  if (!flag) { my = mx;}
  ierr = PetscOptionsInt("-mz","grid points in z","Manualpage",mz,&mz,&flag);CHKERRQ(ierr);
  if (!flag) { mz = mx;}

  /* Set grid size for all finer levels */
  for (i=1; i<user.nlevels; i++) {
  }

  /* set partitioning of domains accross processors */
  ierr = PetscOptionsInt("-Nx","Nx","Manualpage",Nx,&Nx,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Ny","Ny","Manualpage",Ny,&Ny,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-Nz","Nz","Manualpage",Nz,&Nz,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  /* Set up distributed array for each level */
  for (i=0; i<user.nlevels; i++) {
    ierr = DACreate3d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,mx,
                      my,mz,Nx,Ny,Nz,1,1,PETSC_NULL,PETSC_NULL,PETSC_NULL,&user.grid[i].da);CHKERRQ(ierr);
    ierr = DACreateGlobalVector(user.grid[i].da,&user.grid[i].x);CHKERRQ(ierr);
    ierr = VecDuplicate(user.grid[i].x,&user.grid[i].r);CHKERRQ(ierr);
    ierr = VecDuplicate(user.grid[i].x,&user.grid[i].b);CHKERRQ(ierr);
    ierr = DACreateLocalVector(user.grid[i].da,&user.grid[i].localX);CHKERRQ(ierr);
    ierr = VecDuplicate(user.grid[i].localX,&user.grid[i].localF);CHKERRQ(ierr);
    ierr = VecGetLocalSize(user.grid[i].x,&nlocal);CHKERRQ(ierr);
    ierr = VecGetSize(user.grid[i].x,&n);CHKERRQ(ierr);
    ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD,nlocal,nlocal,n,n,7,PETSC_NULL,5,PETSC_NULL,&user.grid[i].J);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Grid %d size %d by %d by %d\n",i,mx,my,mz);CHKERRQ(ierr);
    mx = user.ratio*(mx-1)+1; 
    my = user.ratio*(my-1)+1;
    mz = user.ratio*(mz-1)+1;
  }

  /* Create nonlinear solver */
  ierr = SNESCreate(PETSC_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes);CHKERRQ(ierr);

  /* provide user function and Jacobian */
  finegrid = &user.grid[user.nlevels-1];
  ierr = SNESSetFunction(snes,finegrid->b,FormFunction,&user);CHKERRQ(ierr);
  ierr = SNESSetJacobian(snes,finegrid->J,finegrid->J,FormJacobian,&user);CHKERRQ(ierr);

  /* set multilevel (Schwarz) preconditioner */
  ierr = SNESGetSLES(snes,&sles);CHKERRQ(ierr);
  ierr = SLESGetPC(sles,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCMG);CHKERRQ(ierr);
  ierr = MGSetLevels(pc,user.nlevels,PETSC_NULL);CHKERRQ(ierr);
  ierr = MGSetType(pc,MGADDITIVE);CHKERRQ(ierr);

  /* set the work vectors and SLES options for all the levels */
  for (i=0; i<user.nlevels; i++) {
    ierr = MGGetSmoother(pc,i,&user.grid[i].sles);CHKERRQ(ierr);
    ierr = SLESSetFromOptions(user.grid[i].sles);CHKERRQ(ierr);
    ierr = SLESSetOperators(user.grid[i].sles,user.grid[i].J,user.grid[i].J,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MGSetX(pc,i,user.grid[i].x);CHKERRQ(ierr); 
    ierr = MGSetRhs(pc,i,user.grid[i].b);CHKERRQ(ierr); 
    ierr = MGSetR(pc,i,user.grid[i].r);CHKERRQ(ierr); 
    ierr = MGSetResidual(pc,i,MGDefaultResidual,user.grid[i].J);CHKERRQ(ierr);
  }

  /* Create interpolation between the levels */
  for (i=1; i<user.nlevels; i++) {
    ierr = DAGetInterpolation(user.grid[i-1].da,user.grid[i].da,&user.grid[i].R,&user.grid[i].Rscale);CHKERRQ(ierr);
    ierr = MGSetInterpolate(pc,i,user.grid[i].R);CHKERRQ(ierr);
    ierr = MGSetRestriction(pc,i,user.grid[i].R);CHKERRQ(ierr);
  }

  /* Solve 1 Newton iteration of nonlinear system 
     - to preload executable so next solve has accurate timing */
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  ierr = SNESGetTolerances(snes,&atol,&rtol,&stol,&maxit,&maxf);CHKERRQ(ierr);
  ierr = SNESSetTolerances(snes,atol,rtol,stol,1,maxf);CHKERRQ(ierr);
  ierr = FormInitialGuess1(&user,finegrid->x);CHKERRQ(ierr);

  ierr = SNESSolve(snes,finegrid->x,&its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Pre-load Newton iterations = %d\n",its);CHKERRQ(ierr);

  /* Reset options, then solve nonlinear system */
  ierr = SNESSetTolerances(snes,atol,rtol,stol,maxit,maxf);CHKERRQ(ierr);
  ierr = FormInitialGuess1(&user,finegrid->x);CHKERRQ(ierr);
  ierr = PetscLogStagePush(1);CHKERRQ(ierr);
  ierr = SNESSolve(snes,finegrid->x,&its);CHKERRQ(ierr);
  ierr = SNESView(snes,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscLogStagePop();CHKERRQ(ierr);
  ierr = SNESGetNumberLinearIterations(snes,&lits);CHKERRQ(ierr);
  litspit = ((double)lits)/((double)its);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of Newton iterations = %d\n",its);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of Linear iterations = %d\n",lits);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Average Linear its / Newton = %e\n",litspit);CHKERRQ(ierr);

  /* Free data structures on the levels */
  for (i=0; i<user.nlevels; i++) {
    ierr = MatDestroy(user.grid[i].J);CHKERRQ(ierr);
    ierr = VecDestroy(user.grid[i].x);CHKERRQ(ierr);
    ierr = VecDestroy(user.grid[i].r);CHKERRQ(ierr);
    ierr = VecDestroy(user.grid[i].b);CHKERRQ(ierr);
    ierr = DADestroy(user.grid[i].da);CHKERRQ(ierr);
    ierr = VecDestroy(user.grid[i].localX);CHKERRQ(ierr);
    ierr = VecDestroy(user.grid[i].localF);CHKERRQ(ierr);
  }

  /* Free interpolations between levels */
  for (i=1; i<user.nlevels; i++) {
    ierr = MatDestroy(user.grid[i].R);CHKERRQ(ierr); 
    ierr = VecDestroy(user.grid[i].Rscale);CHKERRQ(ierr); 
  }

  /* free nonlinear solver object */
  ierr = SNESDestroy(snes);CHKERRQ(ierr);
  PetscFinalize();


  return 0;
}
/* --------------------  Form initial approximation ----------------- */
#undef __FUNC__
#define __FUNC__ "FormInitialGuess1"
int FormInitialGuess1(AppCtx *user,Vec X)
{
  int     i,j,k,row,ierr,xs,ys,zs,xm,ym,zm,Xm,Ym,Zm,Xs,Ys,Zs;
  double  tleft = user->tleft;
  Scalar  *x;
  GridCtx *finegrid = &user->grid[user->nlevels-1];
  Vec     localX = finegrid->localX;

  PetscFunctionBegin;

  /* Get ghost points */
  ierr = DAGetCorners(finegrid->da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(finegrid->da,&Xs,&Ys,&Zs,&Xm,&Ym,&Zm);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);

  /* Compute initial guess */
  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        row = (i - Xs) + (j - Ys)*Xm + (k - Zs)*Ym*Xm; 
        x[row] = tleft;
      }
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
  int     ierr,i,j,k,row,mx,my,mz,xs,ys,zs,xm,ym,zm,Xs,Ys,Zs,Xm,Ym,Zm;
  double  zero = 0.0,one = 1.0;
  double  h_x,h_y,h_z,hxhydhz,hyhzdhx,hzhxdhy;
  double  t0,tn,ts,te,tw,tu,td,an,as,ae,aw,au,ad,dn,ds,de,dw,du,dd,fn=0.0,fs=0.0,fe=0.0,fw=0.0,fu=0.0,fd=0.0;
  double  tleft,tright,beta;
  Scalar  *x,*f;
  GridCtx *finegrid = &user->grid[user->nlevels-1];
  Vec     localX = finegrid->localX,localF = finegrid->localF; 

  PetscFunctionBegin;
  ierr = DAGetInfo(finegrid->da,PETSC_NULL,&mx,&my,&mz,0,0,0,0,0,0,0);CHKERRQ(ierr);
  h_x = one/(double)(mx-1);  h_y = one/(double)(my-1);  h_z = one/(double)(mz-1);
  hxhydhz = h_x*h_y/h_z;   hyhzdhx = h_y*h_z/h_x;   hzhxdhy = h_z*h_x/h_y;
  tleft = user->tleft;      tright = user->tright;
  beta = user->beta;
 
  /* Get ghost points */
  ierr = DAGlobalToLocalBegin(finegrid->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(finegrid->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGetCorners(finegrid->da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(finegrid->da,&Xs,&Ys,&Zs,&Xm,&Ym,&Zm);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);
  ierr = VecGetArray(localF,&f);CHKERRQ(ierr);

  /* Evaluate function */
  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        row = (i - Xs) + (j - Ys)*Xm + (k - Zs)*Ym*Xm; 
        t0 = x[row];

        if (i > 0 && i < mx-1 && j > 0 && j < my-1 && k > 0 && k < mz-1) {

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

          td = x[row - Xm*Ym];  
          ad = 0.5*(t0 + td);
          dd = pow(ad,beta);
          fd = dd*(t0 - td);

          tu = x[row + Xm*Ym];  
          au = 0.5*(t0 + tu);
          du = pow(au,beta);
          fu = du*(tu - t0);

        } else if (i == 0) {

 	  /* left-hand (west) boundary */
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

	  if (k > 0) {
            td = x[row - Xm*Ym];  
            ad = 0.5*(t0 + td);
            dd = pow(ad,beta);
            fd = dd*(t0 - td);
	  } else {
 	    fd = zero;
	  }

	  if (k < mz-1) { 
            tu = x[row + Xm*Ym];  
            au = 0.5*(t0 + tu);
            du = pow(au,beta);
            fu = du*(tu - t0);
	  } else {
 	    fu = zero;
	  }

        } else if (i == mx-1) {

          /* right-hand (east) boundary */ 
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

	  if (k > 0) {
            td = x[row - Xm*Ym];  
            ad = 0.5*(t0 + td);
            dd = pow(ad,beta);
            fd = dd*(t0 - td);
	  } else {
 	    fd = zero;
	  }

	  if (k < mz-1) { 
            tu = x[row + Xm*Ym];  
            au = 0.5*(t0 + tu);
            du = pow(au,beta);
            fu = du*(tu - t0);
	  } else {
 	    fu = zero;
	  }

        } else if (j == 0) {

	  /* bottom (south) boundary, and i <> 0 or mx-1 */
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

	  if (k > 0) {
            td = x[row - Xm*Ym];  
            ad = 0.5*(t0 + td);
            dd = pow(ad,beta);
            fd = dd*(t0 - td);
	  } else {
 	    fd = zero;
	  }

	  if (k < mz-1) { 
            tu = x[row + Xm*Ym];  
            au = 0.5*(t0 + tu);
            du = pow(au,beta);
            fu = du*(tu - t0);
	  } else {
 	    fu = zero;
	  }

        } else if (j == my-1) {

	  /* top (north) boundary, and i <> 0 or mx-1 */ 
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

	  if (k > 0) {
            td = x[row - Xm*Ym];  
            ad = 0.5*(t0 + td);
            dd = pow(ad,beta);
            fd = dd*(t0 - td);
	  } else {
 	    fd = zero;
	  }

	  if (k < mz-1) { 
            tu = x[row + Xm*Ym];  
            au = 0.5*(t0 + tu);
            du = pow(au,beta);
            fu = du*(tu - t0);
	  } else {
 	    fu = zero;
	  }

        } else if (k == 0) {

	  /* down boundary (interior only) */
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

 	  fd = zero;

          tu = x[row + Xm*Ym];  
          au = 0.5*(t0 + tu);
          du = pow(au,beta);
          fu = du*(tu - t0);
	
        } else if (k == mz-1) {

	  /* up boundary (interior only) */
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

          td = x[row - Xm*Ym];  
          ad = 0.5*(t0 + td);
          dd = pow(ad,beta);
          fd = dd*(t0 - td);

          fu = zero;
	}

        f[row] = - hyhzdhx*(fe-fw) - hzhxdhy*(fn-fs) - hxhydhz*(fu-fd); 

      }
    }
  } 
  ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);
  
  /* Insert values into global vector */ 
  ierr = DALocalToGlobal(finegrid->da,localF,INSERT_VALUES,F);CHKERRQ(ierr); 
  ierr = PetscLogFlops((33 + 6*POWFLOP)*ym*xm*zm);CHKERRQ(ierr);
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
  int     ierr,i,j,k,row,mx,my,mz,xs,ys,zs,xm,ym,zm,Xs,Ys,Zs,Xm,Ym,Zm,col[7],nloc,*ltog,grow;
  double  one = 1.0,h_x,h_y,h_z,hzhxdhy,hyhzdhx,hxhydhz,t0,tn,ts,te,tw,tu,td; 
  double  dn,ds,de,dw,du,dd,an,as,ae,aw,au,ad,bn,bs,be,bw,bu,bd,gn,gs,ge,gw,gu,gd;
  double  tleft,tright,beta,bm1,coef;
  Scalar  v[7],*x;
  Vec     localX = grid->localX;

  PetscFunctionBegin;
  ierr = DAGetInfo(grid->da,PETSC_NULL,&mx,&my,&mz,0,0,0,0,0,0,0);CHKERRQ(ierr);
  h_x = one/(double)(mx-1);  h_y = one/(double)(my-1);  h_z = one/(double)(mz-1);
  hzhxdhy = h_z*h_x/h_y;       hyhzdhx = h_y*h_z/h_x;       hxhydhz = h_x*h_y/h_z;
  tleft = user->tleft;      tright = user->tright;
  beta = user->beta;	    bm1 = user->bm1;		coef = user->coef;

  /* Get ghost points */
  ierr = DAGlobalToLocalBegin(grid->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(grid->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DAGetCorners(grid->da,&xs,&ys,&zs,&xm,&ym,&zm);CHKERRQ(ierr);
  ierr = DAGetGhostCorners(grid->da,&Xs,&Ys,&Zs,&Xm,&Ym,&Zm);CHKERRQ(ierr);
  ierr = DAGetGlobalIndices(grid->da,&nloc,&ltog);CHKERRQ(ierr);
  ierr = VecGetArray(localX,&x);CHKERRQ(ierr);

  /* Evaluate Jacobian of function */
  for (k=zs; k<zs+zm; k++) {
    for (j=ys; j<ys+ym; j++) {
      for (i=xs; i<xs+xm; i++) {
        row = (i - Xs) + (j - Ys)*Xm + (k - Zs)*Ym*Xm; 
        grow = ltog[row];
        t0 = x[row];

        if (i > 0 && i < mx-1 && j > 0 && j < my-1 && k > 0 && k < mz-1) {

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

          td = x[row - Xm*Ym];
          ad = 0.5*(t0 + td);
          bd = pow(ad,bm1);
	  /* dd = bd * ad; */
	  dd = pow(ad,beta);
          gd = coef*bd*(t0 - td);
  
          tu = x[row + Xm*Ym];  
          au = 0.5*(t0 + tu);
          bu = pow(au,bm1);
	  /* du = bu * au; */
	  du = pow(au,beta);
          gu = coef*bu*(tu - t0);

	  col[0] =   ltog[row - Xm*Ym];
          v[0]   = - hxhydhz*(dd - gd); 
	  col[1] =   ltog[row - Xm];
          v[1]   = - hzhxdhy*(ds - gs); 
	  col[2] =   ltog[row - 1];
          v[2]   = - hyhzdhx*(dw - gw); 
          col[3] =   grow;
	  v[3]   =   hzhxdhy*(ds + dn + gs - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + du + gd - gu);
	  col[4] =   ltog[row + 1];
          v[4]   = - hyhzdhx*(de + ge); 
	  col[5] =   ltog[row + Xm];
          v[5]   = - hzhxdhy*(dn + gn); 
	  col[6] =   ltog[row + Xm*Ym];
	  v[6]   = - hxhydhz*(du + gu);
          ierr   =   MatSetValues(jac,1,&grow,7,col,v,INSERT_VALUES);CHKERRQ(ierr);

        } else if (i == 0) {

          /* left-hand plane boundary */
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
 
	  /* left-hand bottom edge */
	  if (j == 0) {

            tn = x[row + Xm];   
            an = 0.5*(t0 + tn); 
            bn = pow(an,bm1); 
      	    /* dn = bn * an; */
	    dn = pow(an,beta);
            gn = coef*bn*(tn - t0); 
          
	    /* left-hand bottom down corner */
	    if (k == 0) {

              tu = x[row + Xm*Ym];   
              au = 0.5*(t0 + tu); 
              bu = pow(au,bm1); 
      	      /* du = bu * au; */
	      du = pow(au,beta);
              gu = coef*bu*(tu - t0); 
          
              col[0] =   grow;
              v[0]   =   hzhxdhy*(dn - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(du - gu); 
              col[1] =   ltog[row + 1];
              v[1]   = - hyhzdhx*(de + ge); 
              col[2] =   ltog[row + Xm];
              v[2]   = - hzhxdhy*(dn + gn); 
	      col[3] =   ltog[row + Xm*Ym];
	      v[3]   = - hxhydhz*(du + gu);
              ierr   =   MatSetValues(jac,1,&grow,4,col,v,INSERT_VALUES);CHKERRQ(ierr);

	    /* left-hand bottom interior edge */
            } else if (k < mz-1) {

              tu = x[row + Xm*Ym];   
              au = 0.5*(t0 + tu); 
              bu = pow(au,bm1); 
      	      /* du = bu * au; */
	      du = pow(au,beta);
              gu = coef*bu*(tu - t0); 
          
              td = x[row - Xm*Ym];   
              ad = 0.5*(t0 + td); 
              bd = pow(ad,bm1); 
      	      /* dd = bd * ad; */
	      dd = pow(ad,beta);
              gd = coef*bd*(td - t0); 
          
              col[0] =   ltog[row - Xm*Ym];
              v[0]   = - hxhydhz*(dd - gd); 
              col[1] =   grow;
              v[1]   =   hzhxdhy*(dn - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + du + gd - gu); 
              col[2] =   ltog[row + 1];
              v[2]   = - hyhzdhx*(de + ge); 
              col[3] =   ltog[row + Xm];
              v[3]   = - hzhxdhy*(dn + gn); 
	      col[4] =   ltog[row + Xm*Ym];
	      v[4]   = - hxhydhz*(du + gu);
              ierr   =   MatSetValues(jac,1,&grow,5,col,v,INSERT_VALUES);CHKERRQ(ierr);

	    /* left-hand bottom up corner */
            } else {

              td = x[row - Xm*Ym];   
              ad = 0.5*(t0 + td); 
              bd = pow(ad,bm1); 
      	      /* dd = bd * ad; */
	      dd = pow(ad,beta);
              gd = coef*bd*(td - t0); 
          
              col[0] =   ltog[row - Xm*Ym];
              v[0]   = - hxhydhz*(dd - gd); 
              col[1] =   grow;
              v[1]   =   hzhxdhy*(dn - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + gd); 
              col[2] =   ltog[row + 1];
              v[2]   = - hyhzdhx*(de + ge); 
              col[3] =   ltog[row + Xm];
              v[3]   = - hzhxdhy*(dn + gn); 
              ierr   =   MatSetValues(jac,1,&grow,4,col,v,INSERT_VALUES);CHKERRQ(ierr);
	    }

	  /* left-hand top edge */
	  } else if (j == my-1) {

            ts = x[row - Xm];   
            as = 0.5*(t0 + ts); 
            bs = pow(as,bm1); 
      	    /* ds = bs * as; */
	    ds = pow(as,beta);
            gs = coef*bs*(ts - t0); 
          
	    /* left-hand top down corner */
	    if (k == 0) {

              tu = x[row + Xm*Ym];   
              au = 0.5*(t0 + tu); 
              bu = pow(au,bm1); 
      	      /* du = bu * au; */
	      du = pow(au,beta);
              gu = coef*bu*(tu - t0); 
           
              col[0] =   ltog[row - Xm];
              v[0]   = - hzhxdhy*(ds - gs); 
              col[1] =   grow;
              v[1]   =   hzhxdhy*(ds + gs) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(du - gu); 
              col[2] =   ltog[row + 1];
              v[2]   = - hyhzdhx*(de + ge); 
	      col[3] =   ltog[row + Xm*Ym];
	      v[3]   = - hxhydhz*(du + gu);
              ierr   =   MatSetValues(jac,1,&grow,4,col,v,INSERT_VALUES);CHKERRQ(ierr);

	    /* left-hand top interior edge */
            } else if (k < mz-1) {

              tu = x[row + Xm*Ym];   
              au = 0.5*(t0 + tu); 
              bu = pow(au,bm1); 
      	      /* du = bu * au; */
	      du = pow(au,beta);
              gu = coef*bu*(tu - t0); 
          
              td = x[row - Xm*Ym];   
              ad = 0.5*(t0 + td); 
              bd = pow(ad,bm1); 
      	      /* dd = bd * ad; */
	      dd = pow(ad,beta);
              gd = coef*bd*(td - t0); 
          
              col[0] =   ltog[row - Xm*Ym];
              v[0]   = - hxhydhz*(dd - gd); 
              col[1] =   ltog[row - Xm];
              v[1]   = - hzhxdhy*(ds - gs); 
              col[2] =   grow;
              v[2]   =   hzhxdhy*(ds + gs) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + du + gd - gu); 
              col[3] =   ltog[row + 1];
              v[3]   = - hyhzdhx*(de + ge); 
	      col[4] =   ltog[row + Xm*Ym];
	      v[4]   = - hxhydhz*(du + gu);
              ierr   =   MatSetValues(jac,1,&grow,5,col,v,INSERT_VALUES);CHKERRQ(ierr);

	    /* left-hand top up corner */
            } else {

              td = x[row - Xm*Ym];   
              ad = 0.5*(t0 + td); 
              bd = pow(ad,bm1); 
      	      /* dd = bd * ad; */
	      dd = pow(ad,beta);
              gd = coef*bd*(td - t0); 
          
              col[0] =   ltog[row - Xm*Ym];
              v[0]   = - hxhydhz*(dd - gd); 
              col[1] =   ltog[row - Xm];
              v[1]   = - hzhxdhy*(ds - gs); 
              col[2] =   grow;
              v[2]   =   hzhxdhy*(ds + gs) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + gd); 
              col[3] =   ltog[row + 1];
              v[3]   = - hyhzdhx*(de + ge); 
              ierr   =   MatSetValues(jac,1,&grow,4,col,v,INSERT_VALUES);CHKERRQ(ierr);
	    }

          } else {

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

	    /* left-hand down interior edge */
            if (k == 0) {

              tu = x[row + Xm*Ym];  
              au = 0.5*(t0 + tu);
              bu = pow(au,bm1);
	      /* du = bu * au; */
	      du = pow(au,beta);
              gu = coef*bu*(tu - t0);

	      col[0] =   ltog[row - Xm];
              v[0]   = - hzhxdhy*(ds - gs); 
              col[1] =   grow;
	      v[1]   =   hzhxdhy*(ds + dn + gs - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(du - gu);
	      col[2] =   ltog[row + 1];
              v[2]   = - hyhzdhx*(de + ge); 
	      col[3] =   ltog[row + Xm];
              v[3]   = - hzhxdhy*(dn + gn); 
	      col[4] =   ltog[row + Xm*Ym];
	      v[4]   = - hxhydhz*(du + gu);
              ierr   =   MatSetValues(jac,1,&grow,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
            }

	    /* left-hand up interior edge */
	    else if (k == mz-1) {

              td = x[row - Xm*Ym];
              ad = 0.5*(t0 + td);
              bd = pow(ad,bm1);
	      /* dd = bd * ad; */
	      dd = pow(ad,beta);
              gd = coef*bd*(t0 - td);
  
	      col[0] =   ltog[row - Xm*Ym];
              v[0]   = - hxhydhz*(dd - gd); 
	      col[1] =   ltog[row - Xm];
              v[1]   = - hzhxdhy*(ds - gs); 
              col[2] =   grow;
	      v[2]   =   hzhxdhy*(ds + dn + gs - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + gd);
	      col[3] =   ltog[row + 1];
              v[3]   = - hyhzdhx*(de + ge); 
	      col[4] =   ltog[row + Xm];
              v[4]   = - hzhxdhy*(dn + gn); 
              ierr   =   MatSetValues(jac,1,&grow,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
	    }

	    /* left-hand interior plane */
	    else {

              td = x[row - Xm*Ym];
              ad = 0.5*(t0 + td);
              bd = pow(ad,bm1);
	      /* dd = bd * ad; */
	      dd = pow(ad,beta);
              gd = coef*bd*(t0 - td);
  
              tu = x[row + Xm*Ym];  
              au = 0.5*(t0 + tu);
              bu = pow(au,bm1);
	      /* du = bu * au; */
	      du = pow(au,beta);
              gu = coef*bu*(tu - t0);

	      col[0] =   ltog[row - Xm*Ym];
              v[0]   = - hxhydhz*(dd - gd); 
	      col[1] =   ltog[row - Xm];
              v[1]   = - hzhxdhy*(ds - gs); 
              col[2] =   grow;
	      v[2]   =   hzhxdhy*(ds + dn + gs - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + du + gd - gu);
	      col[3] =   ltog[row + 1];
              v[3]   = - hyhzdhx*(de + ge); 
	      col[4] =   ltog[row + Xm];
              v[4]   = - hzhxdhy*(dn + gn); 
	      col[5] =   ltog[row + Xm*Ym];
	      v[5]   = - hxhydhz*(du + gu);
              ierr   =   MatSetValues(jac,1,&grow,6,col,v,INSERT_VALUES);CHKERRQ(ierr);
	    }
	  }

        } else if (i == mx-1) {

          /* right-hand plane boundary */
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
 
	  /* right-hand bottom edge */
	  if (j == 0) {

            tn = x[row + Xm];   
            an = 0.5*(t0 + tn); 
            bn = pow(an,bm1); 
      	    /* dn = bn * an; */
	    dn = pow(an,beta);
            gn = coef*bn*(tn - t0); 
          
	    /* right-hand bottom down corner */
	    if (k == 0) {

              tu = x[row + Xm*Ym];   
              au = 0.5*(t0 + tu); 
              bu = pow(au,bm1); 
      	      /* du = bu * au; */
	      du = pow(au,beta);
              gu = coef*bu*(tu - t0); 
          
              col[0] =   ltog[row - 1];
              v[0]   = - hyhzdhx*(dw - gw); 
              col[1] =   grow;
              v[1]   =   hzhxdhy*(dn - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(du - gu); 
              col[2] =   ltog[row + Xm];
              v[2]   = - hzhxdhy*(dn + gn); 
	      col[3] =   ltog[row + Xm*Ym];
	      v[3]   = - hxhydhz*(du + gu);
              ierr   =   MatSetValues(jac,1,&grow,4,col,v,INSERT_VALUES);CHKERRQ(ierr);

	    /* right-hand bottom interior edge */
            } else if (k < mz-1) {

              tu = x[row + Xm*Ym];   
              au = 0.5*(t0 + tu); 
              bu = pow(au,bm1); 
      	      /* du = bu * au; */
	      du = pow(au,beta);
              gu = coef*bu*(tu - t0); 
          
              td = x[row - Xm*Ym];   
              ad = 0.5*(t0 + td); 
              bd = pow(ad,bm1); 
      	      /* dd = bd * ad; */
	      dd = pow(ad,beta);
              gd = coef*bd*(td - t0); 
          
              col[0] =   ltog[row - Xm*Ym];
              v[0]   = - hxhydhz*(dd - gd); 
              col[1] =   ltog[row - 1];
              v[1]   = - hyhzdhx*(dw - gw); 
              col[2] =   grow;
              v[2]   =   hzhxdhy*(dn - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + du + gd - gu); 
              col[3] =   ltog[row + Xm];
              v[3]   = - hzhxdhy*(dn + gn); 
	      col[4] =   ltog[row + Xm*Ym];
	      v[4]   = - hxhydhz*(du + gu);
              ierr   =   MatSetValues(jac,1,&grow,5,col,v,INSERT_VALUES);CHKERRQ(ierr);

	    /* right-hand bottom up corner */
            } else {

              td = x[row - Xm*Ym];   
              ad = 0.5*(t0 + td); 
              bd = pow(ad,bm1); 
      	      /* dd = bd * ad; */
	      dd = pow(ad,beta);
              gd = coef*bd*(td - t0); 
          
              col[0] =   ltog[row - Xm*Ym];
              v[0]   = - hxhydhz*(dd - gd); 
              col[1] =   ltog[row - 1];
              v[1]   = - hyhzdhx*(dw - gw); 
              col[2] =   grow;
              v[2]   =   hzhxdhy*(dn - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + gd); 
              col[3] =   ltog[row + Xm];
              v[3]   = - hzhxdhy*(dn + gn); 
              ierr   =   MatSetValues(jac,1,&grow,4,col,v,INSERT_VALUES);CHKERRQ(ierr);
	    }

	  /* right-hand top edge */
	  } else if (j == my-1) {

            ts = x[row - Xm];   
            as = 0.5*(t0 + ts); 
            bs = pow(as,bm1); 
      	    /* ds = bs * as; */
	    ds = pow(as,beta);
            gs = coef*bs*(ts - t0); 
          
	    /* right-hand top down corner */
	    if (k == 0) {

              tu = x[row + Xm*Ym];   
              au = 0.5*(t0 + tu); 
              bu = pow(au,bm1); 
      	      /* du = bu * au; */
	      du = pow(au,beta);
              gu = coef*bu*(tu - t0); 
           
              col[0] =   ltog[row - Xm];
              v[0]   = - hzhxdhy*(ds - gs); 
              col[1] =   ltog[row - 1];
              v[1]   = - hyhzdhx*(dw - gw); 
              col[2] =   grow;
              v[2]   =   hzhxdhy*(ds + gs) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(du - gu); 
	      col[3] =   ltog[row + Xm*Ym];
	      v[3]   = - hxhydhz*(du + gu);
              ierr   =   MatSetValues(jac,1,&grow,4,col,v,INSERT_VALUES);CHKERRQ(ierr);

	    /* right-hand top interior edge */
            } else if (k < mz-1) {

              tu = x[row + Xm*Ym];   
              au = 0.5*(t0 + tu); 
              bu = pow(au,bm1); 
      	      /* du = bu * au; */
	      du = pow(au,beta);
              gu = coef*bu*(tu - t0); 
          
              td = x[row - Xm*Ym];   
              ad = 0.5*(t0 + td); 
              bd = pow(ad,bm1); 
      	      /* dd = bd * ad; */
	      dd = pow(ad,beta);
              gd = coef*bd*(td - t0); 
          
              col[0] =   ltog[row - Xm*Ym];
              v[0]   = - hxhydhz*(dd - gd); 
              col[1] =   ltog[row - Xm];
              v[1]   = - hzhxdhy*(ds - gs); 
              col[2] =   ltog[row - 1];
              v[2]   = - hyhzdhx*(dw - gw); 
              col[3] =   grow;
              v[3]   =   hzhxdhy*(ds + gs) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + du + gd - gu); 
	      col[4] =   ltog[row + Xm*Ym];
	      v[4]   = - hxhydhz*(du + gu);
              ierr   =   MatSetValues(jac,1,&grow,5,col,v,INSERT_VALUES);CHKERRQ(ierr);

	    /* right-hand top up corner */
            } else {

              td = x[row - Xm*Ym];   
              ad = 0.5*(t0 + td); 
              bd = pow(ad,bm1); 
      	      /* dd = bd * ad; */
	      dd = pow(ad,beta);
              gd = coef*bd*(td - t0); 
          
              col[0] =   ltog[row - Xm*Ym];
              v[0]   = - hxhydhz*(dd - gd); 
              col[1] =   ltog[row - Xm];
              v[1]   = - hzhxdhy*(ds - gs); 
              col[2] =   ltog[row - 1];
              v[2]   = - hyhzdhx*(dw - gw); 
              col[3] =   grow;
              v[3]   =   hzhxdhy*(ds + gs) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + gd); 
              ierr   =   MatSetValues(jac,1,&grow,4,col,v,INSERT_VALUES);CHKERRQ(ierr);
	    }

          } else {

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

	    /* right-hand down interior edge */
            if (k == 0) {

              tu = x[row + Xm*Ym];  
              au = 0.5*(t0 + tu);
              bu = pow(au,bm1);
	      /* du = bu * au; */
	      du = pow(au,beta);
              gu = coef*bu*(tu - t0);

	      col[0] =   ltog[row - Xm];
              v[0]   = - hzhxdhy*(ds - gs); 
	      col[1] =   ltog[row - 1];
              v[1]   = - hyhzdhx*(dw - gw); 
              col[2] =   grow;
	      v[2]   =   hzhxdhy*(ds + dn + gs - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(du - gu);
	      col[3] =   ltog[row + Xm];
              v[3]   = - hzhxdhy*(dn + gn); 
	      col[4] =   ltog[row + Xm*Ym];
	      v[4]   = - hxhydhz*(du + gu);
              ierr   =   MatSetValues(jac,1,&grow,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
            }

	    /* right-hand up interior edge */
	    else if (k == mz-1) {

              td = x[row - Xm*Ym];
              ad = 0.5*(t0 + td);
              bd = pow(ad,bm1);
	      /* dd = bd * ad; */
	      dd = pow(ad,beta);
              gd = coef*bd*(t0 - td);
  
	      col[0] =   ltog[row - Xm*Ym];
              v[0]   = - hxhydhz*(dd - gd); 
	      col[1] =   ltog[row - Xm];
              v[1]   = - hzhxdhy*(ds - gs); 
	      col[2] =   ltog[row - 1];
              v[2]   = - hyhzdhx*(dw - gw); 
              col[3] =   grow;
	      v[3]   =   hzhxdhy*(ds + dn + gs - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + gd);
	      col[4] =   ltog[row + Xm];
              v[4]   = - hzhxdhy*(dn + gn); 
              ierr   =   MatSetValues(jac,1,&grow,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
	    }

	    /* right-hand interior plane */
	    else {

              td = x[row - Xm*Ym];
              ad = 0.5*(t0 + td);
              bd = pow(ad,bm1);
	      /* dd = bd * ad; */
	      dd = pow(ad,beta);
              gd = coef*bd*(t0 - td);
  
              tu = x[row + Xm*Ym];  
              au = 0.5*(t0 + tu);
              bu = pow(au,bm1);
	      /* du = bu * au; */
	      du = pow(au,beta);
              gu = coef*bu*(tu - t0);

	      col[0] =   ltog[row - Xm*Ym];
              v[0]   = - hxhydhz*(dd - gd); 
	      col[1] =   ltog[row - Xm];
              v[1]   = - hzhxdhy*(ds - gs); 
	      col[2] =   ltog[row - 1];
              v[2]   = - hyhzdhx*(dw - gw); 
              col[3] =   grow;
	      v[3]   =   hzhxdhy*(ds + dn + gs - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + du + gd - gu);
	      col[4] =   ltog[row + Xm];
              v[4]   = - hzhxdhy*(dn + gn); 
	      col[5] =   ltog[row + Xm*Ym];
	      v[5]   = - hxhydhz*(du + gu);
              ierr   =   MatSetValues(jac,1,&grow,6,col,v,INSERT_VALUES);CHKERRQ(ierr);
	    }
	  }

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


          /* bottom down interior edge */
	  if (k == 0) {

            tu = x[row + Xm*Ym];  
            au = 0.5*(t0 + tu);
            bu = pow(au,bm1);
	    /* du = bu * au; */
	    du = pow(au,beta);
            gu = coef*bu*(tu - t0);

	    col[0] =   ltog[row - 1];
            v[0]   = - hyhzdhx*(dw - gw); 
            col[1] =   grow;
	    v[1]   =   hzhxdhy*(dn - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(du - gu);
	    col[2] =   ltog[row + 1];
            v[2]   = - hyhzdhx*(de + ge); 
	    col[3] =   ltog[row + Xm];
            v[3]   = - hzhxdhy*(dn + gn); 
	    col[4] =   ltog[row + Xm*Ym];
	    v[4]   = - hxhydhz*(du + gu);
            ierr   =   MatSetValues(jac,1,&grow,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
          }

	  /* bottom up interior edge */
	  else if (k == mz-1) {

            td = x[row - Xm*Ym];  
            ad = 0.5*(t0 + td);
            bd = pow(ad,bm1);
	    /* dd = bd * ad; */
	    dd = pow(ad,beta);
            gd = coef*bd*(td - t0);

	    col[0] =   ltog[row - Xm*Ym];
	    v[0]   = - hxhydhz*(dd - gd);
	    col[1] =   ltog[row - 1];
            v[1]   = - hyhzdhx*(dw - gw); 
            col[2] =   grow;
	    v[2]   =   hzhxdhy*(dn - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + gd);
	    col[3] =   ltog[row + 1];
            v[3]   = - hyhzdhx*(de + ge); 
	    col[4] =   ltog[row + Xm];
            v[4]   = - hzhxdhy*(dn + gn); 
            ierr   =   MatSetValues(jac,1,&grow,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
          } 

	  /* bottom interior plane */
          else {

            tu = x[row + Xm*Ym];  
            au = 0.5*(t0 + tu);
            bu = pow(au,bm1);
	    /* du = bu * au; */
	    du = pow(au,beta);
            gu = coef*bu*(tu - t0);

            td = x[row - Xm*Ym];  
            ad = 0.5*(t0 + td);
            bd = pow(ad,bm1);
	    /* dd = bd * ad; */
	    dd = pow(ad,beta);
            gd = coef*bd*(td - t0);

	    col[0] =   ltog[row - Xm*Ym];
	    v[0]   = - hxhydhz*(dd - gd);
	    col[1] =   ltog[row - 1];
            v[1]   = - hyhzdhx*(dw - gw); 
            col[2] =   grow;
	    v[2]   =   hzhxdhy*(dn - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + du + gd - gu);
	    col[3] =   ltog[row + 1];
            v[3]   = - hyhzdhx*(de + ge); 
	    col[4] =   ltog[row + Xm];
            v[4]   = - hzhxdhy*(dn + gn); 
	    col[5] =   ltog[row + Xm*Ym];
	    v[5]   = - hxhydhz*(du + gu);
            ierr   =   MatSetValues(jac,1,&grow,6,col,v,INSERT_VALUES);CHKERRQ(ierr);
          } 

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
  
          /* top down interior edge */
	  if (k == 0) {

            tu = x[row + Xm*Ym];  
            au = 0.5*(t0 + tu);
            bu = pow(au,bm1);
	    /* du = bu * au; */
	    du = pow(au,beta);
            gu = coef*bu*(tu - t0);

	    col[0] =   ltog[row - Xm];
            v[0]   = - hzhxdhy*(ds - gs); 
	    col[1] =   ltog[row - 1];
            v[1]   = - hyhzdhx*(dw - gw); 
            col[2] =   grow;
	    v[2]   =   hzhxdhy*(ds + gs) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(du - gu);
	    col[3] =   ltog[row + 1];
            v[3]   = - hyhzdhx*(de + ge); 
	    col[4] =   ltog[row + Xm*Ym];
	    v[4]   = - hxhydhz*(du + gu);
            ierr   =   MatSetValues(jac,1,&grow,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
          }

	  /* top up interior edge */
	  else if (k == mz-1) {

            td = x[row - Xm*Ym];  
            ad = 0.5*(t0 + td);
            bd = pow(ad,bm1);
	    /* dd = bd * ad; */
	    dd = pow(ad,beta);
            gd = coef*bd*(td - t0);

	    col[0] =   ltog[row - Xm*Ym];
	    v[0]   = - hxhydhz*(dd - gd);
	    col[1] =   ltog[row - Xm];
            v[1]   = - hzhxdhy*(ds - gs); 
	    col[2] =   ltog[row - 1];
            v[2]   = - hyhzdhx*(dw - gw); 
            col[3] =   grow;
	    v[3]   =   hzhxdhy*(ds + gs) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + gd);
	    col[4] =   ltog[row + 1];
            v[4]   = - hyhzdhx*(de + ge); 
            ierr   =   MatSetValues(jac,1,&grow,5,col,v,INSERT_VALUES);CHKERRQ(ierr);
          }

	  /* top interior plane */
          else {

            tu = x[row + Xm*Ym];  
            au = 0.5*(t0 + tu);
            bu = pow(au,bm1);
	    /* du = bu * au; */
	    du = pow(au,beta);
            gu = coef*bu*(tu - t0);

            td = x[row - Xm*Ym];  
            ad = 0.5*(t0 + td);
            bd = pow(ad,bm1);
	    /* dd = bd * ad; */
	    dd = pow(ad,beta);
            gd = coef*bd*(td - t0);

	    col[0] =   ltog[row - Xm*Ym];
	    v[0]   = - hxhydhz*(dd - gd);
	    col[1] =   ltog[row - Xm];
            v[1]   = - hzhxdhy*(ds - gs); 
	    col[2] =   ltog[row - 1];
            v[2]   = - hyhzdhx*(dw - gw); 
            col[3] =   grow;
	    v[3]   =   hzhxdhy*(ds + gs) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + du + gd - gu);
	    col[4] =   ltog[row + 1];
            v[4]   = - hyhzdhx*(de + ge); 
	    col[5] =   ltog[row + Xm*Ym];
	    v[5]   = - hxhydhz*(du + gu);
            ierr   =   MatSetValues(jac,1,&grow,6,col,v,INSERT_VALUES);CHKERRQ(ierr);
          } 

        } else if (k == 0) {

          /* down interior plane */

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
 
          tu = x[row + Xm*Ym];  
          au = 0.5*(t0 + tu);
          bu = pow(au,bm1);
	  /* du = bu * au; */
	  du = pow(au,beta);
          gu = coef*bu*(tu - t0);

	  col[0] =   ltog[row - Xm];
          v[0]   = - hzhxdhy*(ds - gs); 
	  col[1] =   ltog[row - 1];
          v[1]   = - hyhzdhx*(dw - gw); 
          col[2] =   grow;
	  v[2]   =   hzhxdhy*(ds + dn + gs - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(du - gu);
	  col[3] =   ltog[row + 1];
          v[3]   = - hyhzdhx*(de + ge); 
	  col[4] =   ltog[row + Xm];
          v[4]   = - hzhxdhy*(dn + gn); 
	  col[5] =   ltog[row + Xm*Ym];
	  v[5]   = - hxhydhz*(du + gu);
          ierr   =   MatSetValues(jac,1,&grow,6,col,v,INSERT_VALUES);CHKERRQ(ierr);
        } 
	
	else if (k == mz-1) {

          /* up interior plane */

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
 
          td = x[row - Xm*Ym];
          ad = 0.5*(t0 + td);
          bd = pow(ad,bm1);
	  /* dd = bd * ad; */
	  dd = pow(ad,beta);
          gd = coef*bd*(t0 - td);
  
	  col[0] =   ltog[row - Xm*Ym];
          v[0]   = - hxhydhz*(dd - gd); 
	  col[1] =   ltog[row - Xm];
          v[1]   = - hzhxdhy*(ds - gs); 
	  col[2] =   ltog[row - 1];
          v[2]   = - hyhzdhx*(dw - gw); 
          col[3] =   grow;
	  v[3]   =   hzhxdhy*(ds + dn + gs - gn) + hyhzdhx*(dw + de + gw - ge) + hxhydhz*(dd + gd);
	  col[4] =   ltog[row + 1];
          v[4]   = - hyhzdhx*(de + ge); 
	  col[5] =   ltog[row + Xm];
          v[5]   = - hzhxdhy*(dn + gn); 
          ierr   =   MatSetValues(jac,1,&grow,6,col,v,INSERT_VALUES);CHKERRQ(ierr);
        }

      }
    }
  }
  ierr = MatAssemblyBegin(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(jac,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscLogFlops((61 + 12*POWFLOP)*xm*ym*zm);CHKERRQ(ierr);
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

    ierr = SLESSetOperators(finegrid->sles,finegrid->J,finegrid->J,SAME_NONZERO_PATTERN);CHKERRQ(ierr);

    for (i=user->nlevels-1; i>0; i--) {

      /* restrict X to coarse grid */
      ierr = MatRestrict(user->grid[i].R,X,user->grid[i-1].x);CHKERRQ(ierr);
      X    = user->grid[i-1].x;      

      /* scale to "natural" scaling for that grid */
      ierr = VecPointwiseMult(user->grid[i].Rscale,X,X);CHKERRQ(ierr);

      /* form Jacobian on coarse grid */
      ierr = FormJacobian_Grid(user,&user->grid[i-1],X,&user->grid[i-1].J,&user->grid[i-1].J);CHKERRQ(ierr);
    
      ierr = SLESSetOperators(user->grid[i-1].sles,user->grid[i-1].J,user->grid[i-1].J,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
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
