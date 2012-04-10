#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex15.c,v 1.11 1999/10/07 19:09:21 bsmith Exp bsmith $";
#endif

static char help[] =
"This program demonstrates use of the SNES package to solve systems of\n\
nonlinear equations in parallel, using 2-dimensional distributed arrays.\n\
A 2-dim simplified RT test problem is used, with analytic Jacobian. \n\
\n\
  Solves the linear systems via 2 level additive Schwarz \n\
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

/*  
  
    This example models the partial differential equation 
   
         - Div(alpha* T^beta (GRAD T)) = 0.
       
    where beta = 2.5 and alpha = 1.0
 
    BC: T_left = 1.0 , T_right = 0.1, dT/dn_top = dTdn_bottom = 0.
    
    in the unit square, which is uniformly discretized in each of x and 
    y in this simple encoding.  The degrees of freedom are cell centered.
 
    A finite volume approximation with the usual 5-point stencil 
    is used to discretize the boundary value problem to obtain a 
    nonlinear system of equations. 
 
*/

#include "snes.h"
#include "da.h"
#include "mg.h"

/* User-defined application contexts */

typedef struct {
   int        mx,my;            /* number grid points in x and y direction */
   Vec        localX,localF;    /* local vectors with ghost region */
   DM         da;
   Vec        x,b,r;            /* global vectors */
   Mat        J;                /* Jacobian on grid */
   SLES       sles;
   Mat        R;                /* R and Rscale are not set on the coarsest grid */
   Vec        Rscale;
} GridCtx;

#define MAX_LEVELS 10

typedef struct {
   GridCtx     grid[MAX_LEVELS];
   int         ratio;
   int         nlevels;
   double      tleft, tright;  /* Dirichlet boundary conditions */
   double      beta, bm1, coef;/* nonlinear diffusivity parameterizations */
} AppCtx;

#define POWFLOP 5 /* assume a pow() takes five flops */

extern int FormFunction(SNES,Vec,Vec,void*), FormInitialGuess1(AppCtx*,Vec);
extern int FormJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);
extern int FormInterpolation(AppCtx *,GridCtx*,GridCtx*);

/*
      Mm_ratio - ration of grid lines between grid levels
*/
#undef __FUNC__
#define __FUNC__ "main"
int main( int argc, char **argv )
{
  SNES          snes;                      
  AppCtx        user;
  GridCtx       *finegrid;                      
  int           ierr, its, lits, n, Nx = PETSC_DECIDE, Ny = PETSC_DECIDE;
  int           nlocal,i;
  int	        maxit, maxf,flag;
  double	atol, rtol, stol, litspit;
  SLES          sles;
  PC            pc;
  PLogDouble    v1, v2, elapsed;

  PetscInitialize( &argc, &argv,PETSC_NULL,help );

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
  user.grid[0].mx = 5; 
  user.grid[0].my = 5; 
  ierr = OptionsGetInt(PETSC_NULL,"-ratio",&user.ratio,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-nlevels",&user.nlevels,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-mx",&user.grid[0].mx,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-my",&user.grid[0].my,&flag);CHKERRA(ierr);
  if (!flag) { user.grid[0].my = user.grid[0].mx;}
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Coarse grid size %d by %d\n",user.grid[0].mx,user.grid[0].my);CHKERRA(ierr);

  /* Set grid size for all finer levels */
  for (i=1; i<user.nlevels; i++) {
    user.grid[i].mx = user.ratio*(user.grid[i-1].mx-1)+1; 
    user.grid[i].my = user.ratio*(user.grid[i-1].my-1)+1;
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Fine grid %d size %d by %d\n",i,user.grid[i].mx,user.grid[i].my);CHKERRA(ierr);
  }

  /* set partitioning of domains accross processors */
  ierr = OptionsGetInt(PETSC_NULL,"-Nx",&Nx,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-Ny",&Ny,PETSC_NULL);CHKERRA(ierr);

  /* Set up distributed array for  each level */
  for (i=0; i<user.nlevels; i++) {
    ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR,user.grid[i].mx,
                      user.grid[i].my,Nx,Ny,1,1,PETSC_NULL,PETSC_NULL,&user.grid[i].da);CHKERRA(ierr);
    ierr = DMCreateGlobalVector(user.grid[i].da,&user.grid[i].x);CHKERRA(ierr);
    ierr = VecDuplicate(user.grid[i].x,&user.grid[i].r);CHKERRA(ierr);
    ierr = VecDuplicate(user.grid[i].x,&user.grid[i].b);CHKERRA(ierr);
    ierr = DMCreateLocalVector(user.grid[i].da,&user.grid[i].localX);CHKERRA(ierr);
    ierr = VecDuplicate(user.grid[i].localX,&user.grid[i].localF);CHKERRA(ierr);
    ierr = VecGetLocalSize(user.grid[i].x,&nlocal);CHKERRA(ierr);
    ierr = VecGetSize(user.grid[i].x,&n);CHKERRA(ierr);
    ierr = MatCreateAIJ(PETSC_COMM_WORLD,nlocal,nlocal,n,n,5,PETSC_NULL,3,PETSC_NULL,&user.grid[i].J);CHKERRA(ierr);
  }

  /* Create nonlinear solver */
  ierr = SNESCreate(PETSC_COMM_WORLD,SNES_NONLINEAR_EQUATIONS,&snes);CHKERRA(ierr);

  /* provide user function and Jacobian */
  finegrid = &user.grid[user.nlevels-1];
  ierr = SNESSetFunction(snes,finegrid->b,FormFunction,&user);CHKERRA(ierr);
  ierr = SNESSetJacobian(snes,finegrid->J,finegrid->J,FormJacobian,&user);CHKERRA(ierr);

  /* set two level additive Schwarz preconditioner */
  ierr = SNESGetSLES(snes,&sles);CHKERRA(ierr);
  ierr = SLESGetPC(sles,&pc);CHKERRA(ierr);
  ierr = PCSetType(pc,PCMG);CHKERRA(ierr);
  ierr = PCMGSetLevels(pc,user.nlevels);CHKERRA(ierr);
  ierr = PCMGSetType(pc,PC_MG_ADDITIVE);CHKERRA(ierr);

  /* set the work vectors and SLES options for all the levels */
  for (i=0; i<user.nlevels; i++) {
    ierr = PCMGGetSmoother(pc,i,&user.grid[i].sles);CHKERRA(ierr);
    ierr = SLESSetFromOptions(user.grid[i].sles);CHKERRA(ierr);
    ierr = SLESSetOperators(user.grid[i].sles,user.grid[i].J,user.grid[i].J,DIFFERENT_NONZERO_PATTERN);CHKERRA(ierr);
    ierr = PCMGSetX(pc,i,user.grid[i].x);CHKERRA(ierr); 
    ierr = PCMGSetRhs(pc,i,user.grid[i].b);CHKERRA(ierr); 
    ierr = PCMGSetR(pc,i,user.grid[i].r);CHKERRA(ierr); 
    ierr = PCMGSetResidual(pc,i,PCMGDefaultResidual,user.grid[i].J);CHKERRA(ierr);
  }

  /* Create interpolation between the levels */
  for (i=1; i<user.nlevels; i++) {
    ierr = FormInterpolation(&user,&user.grid[i],&user.grid[i-1]);CHKERRA(ierr);
    ierr = PCMGSetInterpolation(pc,i,user.grid[i].R);CHKERRA(ierr);
    ierr = PCMGSetRestriction(pc,i,user.grid[i].R);CHKERRA(ierr);
  }

  /* Solve 1 Newton iteration of nonlinear system (to load all arrays) */
  ierr = SNESSetFromOptions(snes);CHKERRA(ierr);
  ierr = SNESGetTolerances(snes,&atol,&rtol,&stol,&maxit,&maxf);CHKERRA(ierr);
  ierr = SNESSetTolerances(snes,atol,rtol,stol,1,maxf);CHKERRA(ierr);
  ierr = FormInitialGuess1(&user,finegrid->x);CHKERRA(ierr);
  ierr = SNESSolve(snes,PETSC_NULL,finegrid->x);CHKERRA(ierr);
  ierr = SNESGetIterationNumber(snes, &its);CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Pre-load SNES iterations = %d\n", its );CHKERRA(ierr);

  /* Reset options, start timer, then solve nonlinear system */
  ierr = SNESSetTolerances(snes,atol,rtol,stol,maxit,maxf);CHKERRA(ierr);
  ierr = FormInitialGuess1(&user,finegrid->x);CHKERRA(ierr);
  ierr = PLogStagePush(1);CHKERRA(ierr);
  ierr = PetscGetTime(&v1);CHKERRA(ierr);
  ierr = SNESSolve(snes,PETSC_NULL,finegrid->x);CHKERRA(ierr);
  ierr = SNESGetIterationNumber(snes, &its);CHKERRA(ierr);
  ierr = SNESView(snes,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  ierr = PetscGetTime(&v2);CHKERRA(ierr);
  ierr = PLogStagePop();CHKERRA(ierr);
  elapsed = v2 - v1;
  ierr = SNESGetLinearSolveIterations(snes,&lits);CHKERRA(ierr);
  litspit = ((double)lits)/((double)its);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Elapsed Time = %e\n", elapsed );CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of SNES iterations = %d\n", its );CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of Linear iterations = %d\n", lits );CHKERRA(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Average Linear its / Newton = %e\n", litspit );CHKERRA(ierr);

  /* Free data structures on the levels */
  for (i=0; i<user.nlevels; i++) {
    ierr = MatDestroy(&user.grid[i].J);CHKERRA(ierr);
    ierr = VecDestroy(&user.grid[i].x);CHKERRA(ierr);
    ierr = VecDestroy(&user.grid[i].r);CHKERRA(ierr);
    ierr = VecDestroy(&user.grid[i].b);CHKERRA(ierr);
    ierr = DMDestroy(&user.grid[i].da);CHKERRA(ierr);
    ierr = VecDestroy(&user.grid[i].localX);CHKERRA(ierr);
    ierr = VecDestroy(&user.grid[i].localF);CHKERRA(ierr);
  }

  /* Free interpolations between levels */
  for (i=1; i<user.nlevels; i++) {
    ierr = MatDestroy(&user.grid[i].R);CHKERRA(ierr); 
    ierr = VecDestroy(&user.grid[i].Rscale);CHKERRA(ierr); 
  }

  /* free nonlinear solver object */
  ierr = SNESDestroy(&snes);CHKERRA(ierr);
  PetscFinalize();


  return 0;
}
/* --------------------  Form initial approximation ----------------- */
#undef __FUNC__
#define __FUNC__ "FormInitialGuess1"
int FormInitialGuess1(AppCtx *user,Vec X)
{
  int     i, j, row, mx, my, ierr, xs, ys, xm, ym, Xm, Ym, Xs, Ys;
  double  one = 1.0, hx, hy, hxdhy, hydhx, tleft, tright;
  Scalar  *x;
  GridCtx *finegrid = &user->grid[user->nlevels-1];
  Vec     localX = finegrid->localX;

  PetscFunctionBegin;
  mx = finegrid->mx;        my = finegrid->my;            
  hx = one/(double)(mx-1);  hy = one/(double)(my-1);
  hxdhy = hx/hy;            hydhx = hy/hx;
  tleft = user->tleft;      tright = user->tright;

  /* Get ghost points */
  ierr = DMDAGetCorners(finegrid->da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(finegrid->da,&Xs,&Ys,0,&Xm,&Ym,0);CHKERRQ(ierr);
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
  ierr = DMLocalToGlobalBegin(finegrid->da,localX,INSERT_VALUES,X);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(finegrid->da,localX,INSERT_VALUES,X);CHKERRQ(ierr);
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
  AppCtx  *user = (AppCtx *) ptr;
  int     ierr, i, j, row, mx, my, xs, ys, xm, ym, Xs, Ys, Xm, Ym;
  double  hx, hy, hxdhy, hydhx;
  double  t0, tn, ts, te, tw, an, as, ae, aw, dn, ds, de, dw, fn, fs, fe, fw;
  double  tleft, tright, beta;
  Scalar  *x,*f;
  GridCtx *finegrid = &user->grid[user->nlevels-1];
  Vec     localX = finegrid->localX, localF = finegrid->localF; 

  PetscFunctionBegin;
  mx = finegrid->mx;        my = finegrid->my;       
  hx = one/(double)(mx-1);  hy = one/(double)(my-1);
  hxdhy = hx/hy;            hydhx = hy/hx;
  tleft = user->tleft;      tright = user->tright;
  beta = user->beta;
 
  /* Get ghost points */
  ierr = DMGlobalToLocalBegin(finegrid->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(finegrid->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMDAGetCorners(finegrid->da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(finegrid->da,&Xs,&Ys,0,&Xm,&Ym,0);CHKERRQ(ierr);
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
        dw = pow(aw, beta);
        fw = dw*(t0 - tw);

	te = x[row + 1];
        ae = 0.5*(t0 + te);
        de = pow(ae, beta);
        fe = de*(te - t0);

	ts = x[row - Xm];
        as = 0.5*(t0 + ts);
        ds = pow(as, beta);
        fs = ds*(t0 - ts);
  
        tn = x[row + Xm];  
        an = 0.5*(t0 + tn);
        dn = pow(an, beta);
        fn = dn*(tn - t0);

      } else if (i == 0) {

	/* left-hand boundary */
        tw = tleft;   
        aw = 0.5*(t0 + tw);                 
        dw = pow(aw, beta);
        fw = dw*(t0 - tw);

	te = x[row + 1];
        ae = 0.5*(t0 + te);
        de = pow(ae, beta);
        fe = de*(te - t0);

	if (j > 0) {
	  ts = x[row - Xm];
          as = 0.5*(t0 + ts);
          ds = pow(as, beta);
          fs = ds*(t0 - ts);
	} else {
 	  fs = 0.0;
	}

	if (j < my-1) { 
          tn = x[row + Xm];  
          an = 0.5*(t0 + tn);
          dn = pow(an, beta);
	  fn = dn*(tn - t0);
	} else {
	  fn = 0.0; 
	}

      } else if (i == mx-1) {

        /* right-hand boundary */ 
        tw = x[row - 1];   
        aw = 0.5*(t0 + tw);
        dw = pow(aw, beta);
        fw = dw*(t0 - tw);
 
        te = tright;
        ae = 0.5*(t0 + te);
        de = pow(ae, beta);
        fe = de*(te - t0);
 
        if (j > 0) { 
          ts = x[row - Xm];
          as = 0.5*(t0 + ts);
          ds = pow(as, beta);
          fs = ds*(t0 - ts);
        } else {
          fs = 0.0;
        }
 
        if (j < my-1) {
          tn = x[row + Xm];
          an = 0.5*(t0 + tn);
          dn = pow(an, beta);
          fn = dn*(tn - t0); 
        } else {   
          fn = 0.0; 
        }

      } else if (j == 0) {

	/* bottom boundary, and i <> 0 or mx-1 */
        tw = x[row - 1];
        aw = 0.5*(t0 + tw);
        dw = pow(aw, beta);
        fw = dw*(t0 - tw);

        te = x[row + 1];
        ae = 0.5*(t0 + te);
        de = pow(ae, beta);
        fe = de*(te - t0);

        fs = 0.0;

        tn = x[row + Xm];
        an = 0.5*(t0 + tn);
        dn = pow(an, beta);
        fn = dn*(tn - t0);

      } else if (j == my-1) {

	/* top boundary, and i <> 0 or mx-1 */ 
        tw = x[row - 1];
        aw = 0.5*(t0 + tw);
        dw = pow(aw, beta);
        fw = dw*(t0 - tw);

        te = x[row + 1];
        ae = 0.5*(t0 + te);
        de = pow(ae, beta);
        fe = de*(te - t0);

        ts = x[row - Xm];
        as = 0.5*(t0 + ts);
        ds = pow(as, beta);
        fs = ds*(t0 - ts);

        fn = 0.0;

      }

      f[row] = - hydhx*(fe-fw) - hxdhy*(fn-fs); 

    }
  }
  ierr = VecRestoreArray(localX,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(localF,&f);CHKERRQ(ierr);

  /* Insert values into global vector */
  ierr = DMLocalToGlobalBegin(finegrid->da,localF,INSERT_VALUES,F);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(finegrid->da,localF,INSERT_VALUES,F);CHKERRQ(ierr);
  PLogFlops((22 + 4*POWFLOP)*ym*xm);
  PetscFunctionReturn(0);
} 
/* --------------------  Evaluate Jacobian F(x) --------------------- */
/*
      This works on ANY grid
*/
#undef __FUNC__
#define __FUNC__ "FormJacobian_Grid"
int FormJacobian_Grid(AppCtx *user,GridCtx *grid,Vec X, Mat *J,Mat *B)
{
  Mat     jac = *J;
  int     ierr, i, j, row, mx, my, xs, ys, xm, ym, Xs, Ys, Xm, Ym, col[5];
  int     nloc, *ltog, grow;
  double  hx, hy, hxdhy, hydhx, value;
  double  t0, tn, ts, te, tw; 
  double  dn, ds, de, dw, an, as, ae, aw, bn, bs, be, bw, gn, gs, ge, gw;
  double  tleft, tright, beta, bm1, coef;
  Scalar  v[5], *x;
  Vec     localX = grid->localX;

  PetscFunctionBegin;
  mx = grid->mx;            my = grid->my; 
  hx = one/(double)(mx-1);  hy = one/(double)(my-1);
  hxdhy = hx/hy;            hydhx = hy/hx;
  tleft = user->tleft;      tright = user->tright;
  beta = user->beta;	    bm1 = user->bm1;		coef = user->coef;

  /* Get ghost points */
  ierr = DMGlobalToLocalBegin(grid->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(grid->da,X,INSERT_VALUES,localX);CHKERRQ(ierr);
  ierr = DMDAGetCorners(grid->da,&xs,&ys,0,&xm,&ym,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(grid->da,&Xs,&Ys,0,&Xm,&Ym,0);CHKERRQ(ierr);
  ierr = DMDAGetGlobalIndices(grid->da,&nloc,&ltog);CHKERRQ(ierr);
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
        bw = pow(aw, bm1);
	/* dw = bw * aw */
	dw = pow(aw, beta); 
	gw = coef*bw*(t0 - tw);

        te = x[row + 1];
        ae = 0.5*(t0 + te);
        be = pow(ae, bm1);
	/* de = be * ae; */
	de = pow(ae, beta);
        ge = coef*be*(te - t0);

        ts = x[row - Xm];
        as = 0.5*(t0 + ts);
        bs = pow(as, bm1);
	/* ds = bs * as; */
	ds = pow(as, beta);
        gs = coef*bs*(t0 - ts);
  
        tn = x[row + Xm];  
        an = 0.5*(t0 + tn);
        bn = pow(an, bm1);
	/* dn = bn * an; */
	dn = pow(an, beta);
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
        bw = pow(aw, bm1); 
	/* dw = bw * aw */
	dw = pow(aw, beta); 
        gw = coef*bw*(t0 - tw); 
 
        te = x[row + 1]; 
        ae = 0.5*(t0 + te); 
        be = pow(ae, bm1); 
	/* de = be * ae; */
	de = pow(ae, beta);
        ge = coef*be*(te - t0);
 
	/* left-hand bottom boundary */
	if (j == 0) {

          tn = x[row + Xm];   
          an = 0.5*(t0 + tn); 
          bn = pow(an, bm1); 
	  /* dn = bn * an; */
	  dn = pow(an, beta);
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
          bs = pow(as, bm1);  
	  /* ds = bs * as; */
	  ds = pow(as, beta);
          gs = coef*bs*(t0 - ts);  
          
          tn = x[row + Xm];    
          an = 0.5*(t0 + tn); 
          bn = pow(an, bm1);  
	  /* dn = bn * an; */
	  dn = pow(an, beta);
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
          bs = pow(as, bm1);
	  /* ds = bs * as; */
	  ds = pow(as, beta);
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
        bw = pow(aw, bm1); 
	/* dw = bw * aw */
	dw = pow(aw, beta); 
        gw = coef*bw*(t0 - tw); 
 
        te = tright; 
        ae = 0.5*(t0 + te); 
        be = pow(ae, bm1); 
	/* de = be * ae; */
	de = pow(ae, beta);
        ge = coef*be*(te - t0);
 
	/* right-hand bottom boundary */
	if (j == 0) {

          tn = x[row + Xm];   
          an = 0.5*(t0 + tn); 
          bn = pow(an, bm1); 
	  /* dn = bn * an; */
	  dn = pow(an, beta);
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
          bs = pow(as, bm1);  
	  /* ds = bs * as; */
	  ds = pow(as, beta);
          gs = coef*bs*(t0 - ts);  
          
          tn = x[row + Xm];    
          an = 0.5*(t0 + tn); 
          bn = pow(an, bm1);  
	  /* dn = bn * an; */
	  dn = pow(an, beta);
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
          bs = pow(as, bm1);
	  /* ds = bs * as; */
	  ds = pow(as, beta);
          gs = coef*bs*(t0 - ts);
          
          col[0] = ltog[row - Xm]; 
          v[0] = - hxdhy*(ds - gs);  
          col[1] = ltog[row - 1];  
          v[1] = - hydhx*(dw - gw); 
          col[2] = grow; 
          v[2] = hxdhy*(ds + gs) + hydhx*(dw + de + gw - ge);  
          ierr = MatSetValues(jac,1,&grow,3,col,v,INSERT_VALUES);CHKERRQ(ierr); 
	}

      /* bottom boundary, and i <> 0 or mx-1 */
      } else if (j == 0) {

        tw = x[row - 1];
        aw = 0.5*(t0 + tw);
        bw = pow(aw, bm1);
	/* dw = bw * aw */
	dw = pow(aw, beta); 
        gw = coef*bw*(t0 - tw);

        te = x[row + 1];
        ae = 0.5*(t0 + te);
        be = pow(ae, bm1);
	/* de = be * ae; */
	de = pow(ae, beta);
        ge = coef*be*(te - t0);

        tn = x[row + Xm];
        an = 0.5*(t0 + tn);
        bn = pow(an, bm1);
	/* dn = bn * an; */
	dn = pow(an, beta);
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
 
      /* top boundary, and i <> 0 or mx-1 */
      } else if (j == my-1) {
 
        tw = x[row - 1];
        aw = 0.5*(t0 + tw);
        bw = pow(aw, bm1);
	/* dw = bw * aw */
	dw = pow(aw, beta); 
        gw = coef*bw*(t0 - tw);

        te = x[row + 1];
        ae = 0.5*(t0 + te);
        be = pow(ae, bm1);
	/* de = be * ae; */
	de = pow(ae, beta);
        ge = coef*be*(te - t0);

        ts = x[row - Xm];
        as = 0.5*(t0 + ts);
        bs = pow(as, bm1);
 	/* ds = bs * as; */
	ds = pow(as, beta);
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
      This evaluates the Jacobian on all of the grids 
*/
#undef __FUNC__
#define __FUNC__ "FormJacobian"
int FormJacobian(SNES snes,Vec X,Mat *J,Mat *B,MatStructure *flag,void *ptr)
{
  AppCtx  *user = (AppCtx *) ptr;
  int     ierr,i;
  SLES    sles;
  PC      pc;
  GridCtx *finegrid = &user->grid[user->nlevels-1];

  PetscFunctionBegin;
  *flag = SAME_NONZERO_PATTERN;
  ierr = FormJacobian_Grid(user,finegrid,X,J,B);CHKERRQ(ierr);

  /* create coarse grid jacobian for preconditioner */
  ierr = SNESGetSLES(snes,&sles);CHKERRQ(ierr);
  ierr = SLESGetPC(sles,&pc);CHKERRQ(ierr);
  if (PetscTypeCompare(pc,PCMG)) {

    ierr = SLESSetOperators(finegrid->sles,finegrid->J,finegrid->J,SAME_NONZERO_PATTERN);CHKERRA(ierr);

    for (i=user->nlevels-1; i>0; i--) {

      /* restrict X to coarse grid */
      ierr = MGRestrict(user->grid[i].R,X,user->grid[i-1].x);CHKERRQ(ierr);
      X    = user->grid[i-1].x;      

      /* scale to "natural" scaling for that grid */
      ierr = VecPointwiseMult(X,X,user->grid[i].Rscale);CHKERRQ(ierr);

      /* form Jacobian on coarse grid */
      ierr = FormJacobian_Grid(user,&user->grid[i-1],X,&user->grid[i-1].J,&user->grid[i-1].J);CHKERRQ(ierr);
    
      ierr = SLESSetOperators(user->grid[i-1].sles,user->grid[i-1].J,user->grid[i-1].J,SAME_NONZERO_PATTERN);CHKERRA(ierr);
    }
  }
  PetscFunctionReturn(0);;
}

#undef __FUNC__
#define __FUNC__ "FormInterpolation"
/*
      Forms the interpolation (and restriction) operator from a coarse grid g_c to a fine grid g_f
*/
int FormInterpolation(AppCtx *user,GridCtx *g_f,GridCtx *g_c)
{
  int      ierr,i,j,i_start,m_f,j_start,m,n,M,Mx = g_c->mx,My = g_c->my,*idx;
  int      m_ghost,n_ghost,*idx_c,m_ghost_c,n_ghost_c;
  int      row,col,i_start_ghost,j_start_ghost,cols[4],mx = g_f->mx, m_c,my = g_f->my;
  int      c0,c1,c2,c3,nc,ratio = user->ratio,i_end,i_end_ghost,m_c_local,m_f_local;
  int      i_c,j_c,i_start_c,j_start_c,n_c,i_start_ghost_c,j_start_ghost_c;
  Scalar   v[4],x,y, one = 1.0;
  Mat      mat;
  Vec	   Rscale; 

  PetscFunctionBegin;
  ierr = DMDAGetCorners(g_f->da,&i_start,&j_start,0,&m,&n,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(g_f->da,&i_start_ghost,&j_start_ghost,0,&m_ghost,&n_ghost,0);CHKERRQ(ierr);
  ierr = DMDAGetGlobalIndices(g_f->da,PETSC_NULL,&idx);CHKERRQ(ierr);

  ierr = DMDAGetCorners(g_c->da,&i_start_c,&j_start_c,0,&m_c,&n_c,0);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(g_c->da,&i_start_ghost_c,&j_start_ghost_c,0,&m_ghost_c,&n_ghost_c,0);CHKERRQ(ierr);
  ierr = DMDAGetGlobalIndices(g_c->da,PETSC_NULL,&idx_c);CHKERRQ(ierr);

  /* create interpolation matrix */
  ierr = VecGetLocalSize(g_f->x,&m_f_local);CHKERRQ(ierr);
  ierr = VecGetLocalSize(g_c->x,&m_c_local);CHKERRQ(ierr);
  ierr = VecGetSize(g_f->x,&m_f);CHKERRQ(ierr);
  ierr = VecGetSize(g_c->x,&m_c);CHKERRQ(ierr);
  ierr = MatCreateAIJ(PETSC_COMM_WORLD,m_f_local,m_c_local,m_f,m_c,5,0,3,0,&mat);CHKERRQ(ierr);

  /* loop over local fine grid nodes setting interpolation for those*/
  for ( j=j_start; j<j_start+n; j++ ) {
    for ( i=i_start; i<i_start+m; i++ ) {
      /* convert to local "natural" numbering and 
         then to PETSc global numbering */
      row    = idx[m_ghost*(j-j_start_ghost) + (i-i_start_ghost)];

      i_c = (i/ratio);    /* coarse grid node to left of fine grid node */
      j_c = (j/ratio);    /* coarse grid node below fine grid node */

      /* 
         Only include those interpolation points that are truly 
         nonzero. Note this is very important for final grid lines
         in x and y directions; since they have no right/top neighbors
      */
      x  = ((double)(i - i_c*ratio))/((double)ratio);
      y  = ((double)(j - j_c*ratio))/((double)ratio);
      /* printf("i j %d %d %g %g\n",i,j,x,y); */
      nc = 0;
      /* one left and below; or we are right on it */
      if (j_c < j_start_ghost_c || j_c > j_start_ghost_c+n_ghost_c) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL,"Sorry j %d %d %d",j_c,j_start_ghost_c,j_start_ghost_c+n_ghost_c);
      if (i_c < i_start_ghost_c || i_c > i_start_ghost_c+m_ghost_c) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL,"Sorry i %d %d %d",i_c,i_start_ghost_c,i_start_ghost_c+m_ghost_c);
      col      = m_ghost_c*(j_c-j_start_ghost_c) + (i_c-i_start_ghost_c);
      cols[nc] = idx_c[col]; 
      v[nc++]  = x*y - x - y + 1.0;
      /* one right and below */
      if (i_c*ratio != i) { 
        cols[nc] = idx_c[col+1];
        v[nc++]  = -x*y + x;
      }
      /* one left and above */
      if (j_c*ratio != j) { 
        cols[nc] = idx_c[col+m_ghost_c];
        v[nc++]  = -x*y + y;
      }
      /* one right and above */
      if (j_c*ratio != j && i_c*ratio != i) { 
        cols[nc] = idx_c[col+m_ghost_c+1];
        v[nc++]  = x*y;
      }
      ierr = MatSetValues(mat,1,&row,nc,cols,v,INSERT_VALUES);CHKERRQ(ierr); 
    }
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = VecDuplicate(g_c->x,&Rscale);CHKERRQ(ierr);
  ierr = VecSet(g_f->x,one);CHKERRQ(ierr);
  ierr = MGRestrict(mat,g_f->x,Rscale);CHKERRQ(ierr);
  ierr = VecReciprocal(Rscale);CHKERRQ(ierr);
  g_f->Rscale = Rscale;
  g_f->R      = mat;

  PLogFlops(13*m*n);
  PetscFunctionReturn(0);;
}

