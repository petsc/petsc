/*$Id: ex19.c,v 1.30 2001/08/07 21:31:17 bsmith Exp $*/

static char help[] = "Steady-state 2D isoviscous subduction flow, pressure and temperature solver.\n\
  \n\
The flow is driven by the subducting slab.\n\
  -width <#> = width of domain in KM.\n\
  -depth <#> = depth of domain in KM.\n\
  -lid_depth <#> = depth to the base of the lithosphere in KM.\n\
  -slab_dip <#> = dip of the subducting slab in RADIANS.\n\
  -slab_velocity <#> = velocity of slab in CM/YEAR.\n\
  -slab_age <#> = age of slab in MILLIONS OF YEARS. \n\
  -kappa <#> = thermal diffusivity in M^2/SEC. \n\
  -contours : draw contour plots of solution\n\n";

/*T
   Concepts: SNES^solving a system of nonlinear equations (parallel multicomponent example);
   Concepts: DA^using distributed arrays;
   Concepts: multicomponent
   Processors: n
T*/

/* ------------------------------------------------------------------------

    EXAMPLE CALL (200x200 grid, 45 degree dip angle): 
    ex30 -snes_monitor -ksp_truemonitor -pc_type lu -preload off -da_grid_x 20 -da_grid_y 20 -slab_dip 0.78539816339745
    -pc_lu_mat_ordering_type <nd,rcm,...>  -Sets ordering routine (natural)
    -dmmg_jacobian_fd
    -snes_max_it 0

    This problem is modeled by the partial differential equation system
  
        -dP/dx         + Lap(U)     = 0
	-dP/dz         + Lap(W)     = 0
 	 Div(U,W)                   = 0

    which is uniformly discretized on a staggered mesh:
                       ------w_ij------
                      /               /
                  u_i-1j    P_ij    u_ij
                    /               /
 		    ------w_ij-1----
    

    Boundary conditions:
      On i=0 all j: u=0, w=1, p=analytic corner flow;
      On j=0 all i: u=0, w=0, p=analytic corner flow;
      On other bounds: u,w,p=analytic corner flow;
    boundary conditions are enforced using buffer points.

    The Jacobian can be either
      * formed via finite differencing using coloring (the default), or
      * applied matrix-free via the option -snes_mf 
        (for larger grid problems this variant may not converge 
        without a preconditioner due to ill-conditioning).

  ------------------------------------------------------------------------- */

/* 
   Include "petscda.h" so that we can use distributed arrays (DAs).
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - system routines       petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscsles.h   - linear solvers 
*/
#include "petscsnes.h"
#include "petscda.h"

/* 
   User-defined routines and data structures
*/

/*
    The next two structures are essentially the same. The difference is that
  the first does not contain derivative information (as used by ADIC) while the
  second does. The first is used only to contain the solution at the previous time-step
  which is a constant in the computation of the current time step and hence passive 
  (meaning not active in terms of derivatives).
*/
typedef struct {
  PetscScalar u,w,p,T;
} PetscField;

typedef struct {
  PetscScalar u,w,p,T;
} Field;

typedef struct {
  int          mglevels;
  PetscReal    width, depth, scaled_width, scaled_depth, peclet;
  PetscReal    slab_dip, slab_age, slab_velocity, lid_depth, kappa;
  int          icorner, jcorner;
  PetscTruth   draw_contours;                /* indicates drawing contours of solution */
  PetscTruth   PreLoading;
} Parameter;

typedef struct {
  Vec          Xold,func;
  Parameter    *param;
  PetscReal    fnorm_ini, fnorm;
} AppCtx;

PetscReal HALFPI = 3.14159265358979323846/2.0;

extern int FormInitialGuess(SNES,Vec,void*);
extern int FormFunctionLocal(DALocalInfo*,Field**,Field**,void*);
extern PassiveScalar HorizVelocity(PassiveScalar x, PassiveScalar z, PassiveScalar c, PassiveScalar d);
extern PassiveScalar VertVelocity(PassiveScalar x, PassiveScalar z, PassiveScalar c, PassiveScalar d);
extern PassiveScalar Pressure(PassiveScalar x, PassiveScalar z, PassiveScalar c, PassiveScalar d);
extern PetscScalar UInterp(Field **x, int i, int j, PetscScalar fr);
extern PetscScalar WInterp(Field **x, int i, int j, PetscScalar fr);
extern PetscScalar PInterp(Field **x, int i, int j, PetscScalar fr);
extern PetscScalar TInterp(Field **x, int i, int j, PetscScalar fr);
extern void CalcJunk(PassiveScalar,PassiveScalar,PassiveScalar,PassiveScalar*,PassiveScalar*,PassiveScalar*,PassiveScalar*,PassiveScalar*);
extern PassiveScalar erf(PassiveScalar x);
int Initialize(DMMG*);

/*-----------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
/*-----------------------------------------------------------------------*/
{
  DMMG       *dmmg;               /* multilevel grid structure */
  AppCtx     *user;                /* user-defined work context */
  Parameter  param;
  int        mx,mz,its;
  int        i,ierr;
  MPI_Comm   comm;
  SNES       snes;
  DA         da;
  Vec        res;
  PetscReal  SEC_PER_YR = 3600.00*24.00*356.2500;

  PetscInitialize(&argc,&argv,(char *)0,help);
  comm = PETSC_COMM_WORLD;
  PreLoadBegin(PETSC_TRUE,"SetUp");

    param.PreLoading = PreLoading;  //where is PreLoading defined?
    ierr = DMMGCreate(comm,1,&user,&dmmg);CHKERRQ(ierr); 
    param.mglevels = DMMGGetLevels(dmmg);

    /*
      Create distributed array multigrid object (DMMG) to manage parallel grid and vectors
      for principal unknowns (x) and governing residuals (f)
    */
    ierr = DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_BOX,-4,-4,PETSC_DECIDE,PETSC_DECIDE,4,1,0,0,&da);CHKERRQ(ierr);
    ierr = DMMGSetDM(dmmg,(DM)da);CHKERRQ(ierr);
    ierr = DADestroy(da);CHKERRQ(ierr);
    ierr = DAGetInfo(DMMGGetDA(dmmg),0,&mx,&mz,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                     PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

    /* 
      Problem parameters
    */
    param.icorner       = (int)(mx/6+1);    /* gridpoints */
    param.width         = 1200.0;           /* km */
    param.depth         = 600.0;            /* km */
    param.slab_dip      = HALFPI/2.0;       /* 45 degrees */
    param.lid_depth     = 50.0;             /* km */
    param.slab_velocity = 5.0;              /* cm/yr */
    param.slab_age      = 50.0;             /* Ma */
    param.kappa         = 0.7272e-6;        /* m^2/sec */
    ierr = PetscOptionsGetInt(PETSC_NULL,"-icorner",&param.icorner,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(PETSC_NULL,"-width",&param.width,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(PETSC_NULL,"-depth",&param.depth,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(PETSC_NULL,"-slab_dip",&param.slab_dip,PETSC_NULL);CHKERRQ(ierr); 
    ierr = PetscOptionsGetReal(PETSC_NULL,"-slab_velocity",&param.slab_velocity,PETSC_NULL);CHKERRQ(ierr); 
    ierr = PetscOptionsGetReal(PETSC_NULL,"-slab_age",&param.slab_age,PETSC_NULL);CHKERRQ(ierr);   
    ierr = PetscOptionsGetReal(PETSC_NULL,"-lid_depth",&param.lid_depth,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetReal(PETSC_NULL,"-kappa",&param.kappa,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscOptionsHasName(PETSC_NULL,"-contours",&param.draw_contours);CHKERRQ(ierr);
    param.scaled_width = param.width/param.lid_depth;
    param.scaled_depth = param.depth/param.lid_depth;
    param.peclet =  param.slab_velocity/100.0/SEC_PER_YR /* m/sec */
                  * param.lid_depth*1000.0               /* m */
                  / param.kappa;                         /* m^2/sec */
    ierr = PetscOptionsGetReal(PETSC_NULL,"-peclet",&param.peclet,PETSC_NULL);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Peclet number = %g\n",param.peclet);CHKERRQ(ierr);

    ierr = DASetFieldName(DMMGGetDA(dmmg),0,"x-velocity");CHKERRQ(ierr);
    ierr = DASetFieldName(DMMGGetDA(dmmg),1,"y-velocity");CHKERRQ(ierr);
    ierr = DASetFieldName(DMMGGetDA(dmmg),2,"pressure");CHKERRQ(ierr);
    ierr = DASetFieldName(DMMGGetDA(dmmg),3,"temperature");CHKERRQ(ierr);

    /*======================================================================*/
    
    ierr = PetscMalloc(param.mglevels*sizeof(AppCtx),&user); CHKERRQ(ierr);
    for (i=0; i<param.mglevels; i++) {
      ierr = VecDuplicate(dmmg[i]->x, &(user[i].Xold)); CHKERRQ(ierr);
      ierr = VecDuplicate(dmmg[i]->x, &(user[i].func)); CHKERRQ(ierr);
      user[i].param = &param;
      dmmg[i]->user = &user[i]; 
    }
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create user context, set problem data, create vector data structures.
       Also, compute the initial guess.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create nonlinear solver context
       Process adiC(40): WInterp UInterp PInterp TInterp FormFunctionLocal
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /* ierr = DMMGSetSNESLocal(dmmg,FormFunctionLocal,0,ad_FormFunctionLocal,admf_FormFunctionLocal);CHKERRQ(ierr); */
    ierr = DMMGSetSNESLocal(dmmg,FormFunctionLocal,0,0,0);CHKERRQ(ierr); 
    ierr = DMMGSetInitialGuess(dmmg,FormInitialGuess);CHKERRQ(ierr);
        
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Solve the nonlinear system
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    PreLoadStage("Solve"); 
    ierr = Initialize(dmmg); CHKERRQ(ierr);

    ierr = DMMGSolve(dmmg);CHKERRQ(ierr); 
    snes = DMMGGetSNES(dmmg);
    ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of Newton iterations = %d\n", its);CHKERRQ(ierr);

    /*
      Visualize solution
    */
    if (param.draw_contours) {
      ierr = VecView(DMMGGetx(dmmg),PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
    }    
    
    /*
      Output stuff to Matlab socket.
    */
    ierr = SNESGetFunction(snes, &res, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
    ierr = VecView(res, PETSC_VIEWER_SOCKET_WORLD);CHKERRQ(ierr); 
    ierr = VecView(DMMGGetx(dmmg),PETSC_VIEWER_SOCKET_WORLD);CHKERRQ(ierr);  
    ierr = PetscViewerSocketPutInt(PETSC_VIEWER_SOCKET_WORLD,1,&mx);CHKERRQ(ierr); 
    ierr = PetscViewerSocketPutInt(PETSC_VIEWER_SOCKET_WORLD,1,&mz);CHKERRQ(ierr); 
    ierr = PetscViewerSocketPutReal(PETSC_VIEWER_SOCKET_WORLD,1,1,&(param.slab_dip));CHKERRQ(ierr); 
    ierr = PetscViewerSocketPutReal(PETSC_VIEWER_SOCKET_WORLD,1,1,&(param.scaled_width));CHKERRQ(ierr); 
    ierr = PetscViewerSocketPutReal(PETSC_VIEWER_SOCKET_WORLD,1,1,&(param.scaled_depth));CHKERRQ(ierr);
    ierr = PetscViewerSocketPutInt(PETSC_VIEWER_SOCKET_WORLD,1,&(param.icorner));CHKERRQ(ierr); 
    ierr = PetscViewerSocketPutInt(PETSC_VIEWER_SOCKET_WORLD,1,&(param.jcorner));CHKERRQ(ierr);  
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    for (i=0; i<param.mglevels; i++) {
      ierr = VecDestroy(user[i].Xold); CHKERRQ(ierr);
      ierr = VecDestroy(user[i].func); CHKERRQ(ierr);
    }
    ierr = PetscFree(user); CHKERRQ(ierr);
    ierr = DMMGDestroy(dmmg); CHKERRQ(ierr);
    PreLoadEnd();
    
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "Initialize"
/* ------------------------------------------------------------------- */
int Initialize(DMMG *dmmg)
{
  AppCtx    *user = (AppCtx*)dmmg[dmmg[0]->nlevels-1]->user;
  Parameter *param = user->param;
  DA        da;
  int       i,j,mx,mz,ierr,xs,ys,xm,ym,iw,jw;
  int       mglevel,ic,jc;
  PetscReal dx, dz, c, d, beta, itb, cb, sb, sbi, skt;
  PetscReal xp, zp, r, st, ct, th, fr, xPrimeSlab;
  Field     **x;

  da = (DA)(dmmg[param->mglevels-1]->dm); /* getting the fine grid */

  beta = param->slab_dip;
  CalcJunk(param->kappa,param->slab_age,beta,&skt,&cb,&sb,&itb,&sbi);
  c = beta*sb/(beta*beta-sb*sb);
  d = (beta*cb-sb)/(beta*beta-sb*sb); 

  ierr = DAGetInfo(da,0,&mx,&mz,0,0,0,0,0,0,0,0);CHKERRQ(ierr);
  dx  = param->scaled_width/((PetscReal)(mx-2)); 
  dz  = param->scaled_depth/((PetscReal)(mz-2)); 

  param->jcorner = (int)ceil(1.0/dz + 1.0); ic = param->icorner; jc = param->jcorner;
  PetscPrintf(PETSC_COMM_WORLD,"float jcorner = %f, int jcorner = %i, int icorner = %i\n", 1.0/dz + 1.0, jc, ic);
  xPrimeSlab = (ic-1)*dx;

  fr = (1.0-dz/dx*itb)/2.0;
  PetscPrintf(PETSC_COMM_WORLD,"interpolation fraction = %g\n", fr);
  if (fr<0.0)
    SETERRQ(1," USER ERROR: Grid shear exceeds limit! Decrease dz/dx!");

  /*
     Get local grid boundaries (for 2-dimensional DA):
       xs, ys   - starting grid indices (no ghost points)
       xm, ym   - scaled_widths of local grid (no ghost points)
  */
  ierr = DAGetCorners(da,&xs,&ys,PETSC_NULL,&xm,&ym,PETSC_NULL);CHKERRQ(ierr);

  /*
     Get a pointer to vector data.
       - For default PETSc vectors, VecGetArray() returns a pointer to
         the data array.  Otherwise, the routine is implementation dependent.
       - You MUST call VecRestoreArray() when you no longer need access to
         the array.
  */ 
  ierr = DAVecGetArray(da,((AppCtx*)dmmg[param->mglevels-1]->user)->Xold,(void**)&x);CHKERRQ(ierr);

  /*
     Compute initial guess (analytic soln) over the locally owned part of the grid
     Initial condition is isoviscous corner flow and uniform temperature in interior
  */
  for (j=ys; j<ys+ym; j++) {
    jw = j - jc + 1;
    for (i=xs; i<xs+xm; i++) {
      iw = i - ic + 1;

      if (i<ic) { /* slab */
	x[j][i].p   = 0.0;
	x[j][i].u   = cb;
	x[j][i].w   = sb;
	xp = (i-0.5)*dx; zp = (xPrimeSlab - xp)*tan(beta);
	x[j][i].T   = erf(zp*param->lid_depth/2.0/skt);
      } else if (j<jc) { /* lid */
	x[j][i].p   = 0.0;
	x[j][i].u   = 0.0;
	x[j][i].w   = 0.0;
	zp = (j-0.5)*dz;
	x[j][i].T   = zp;
      } else { /* wedge */
      /* horizontal velocity */
      zp = (jw-0.5)*dz; xp = iw*dx+zp*itb; 
      x[j][i].u =  HorizVelocity(xp,zp,c,d);
      /* vertical velocity */
      zp = jw*dz; xp = (iw-0.5)*dx+zp*itb; 
      x[j][i].w =  VertVelocity(xp,zp,c,d);
      /* pressure */
      zp = (jw-0.5)*dz; xp = (iw-0.5)*dx+zp*itb; 
      x[j][i].p =  Pressure(xp,zp,c,d); 
      /* temperature */
      x[j][i].T = 1.0;

      }
    }
  }

  /* Trash initial guess */ 
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      /*x[j][i].u = 0.0;
      x[j][i].w = 0.0;
      x[j][i].p = 0.0;*/
    }
  }

  /* Restore x to Xold */
  ierr = DAVecRestoreArray(da,((AppCtx*)dmmg[param->mglevels-1]->user)->Xold,(void**)&x);CHKERRQ(ierr);
  
  /* Restrict Xold to coarser levels */
  for (mglevel=param->mglevels-1; mglevel>0; mglevel--) {
    ierr = MatRestrict(dmmg[mglevel]->R, ((AppCtx*)dmmg[mglevel]->user)->Xold, ((AppCtx*)dmmg[mglevel-1]->user)->Xold);CHKERRQ(ierr);
    ierr = VecPointwiseMult(dmmg[mglevel]->Rscale,((AppCtx*)dmmg[mglevel-1]->user)->Xold,((AppCtx*)dmmg[mglevel-1]->user)->Xold);CHKERRQ(ierr);
  }
  
  return 0;
} 

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocal"
int FormFunctionLocal(DALocalInfo *info,Field **x,Field **f,void *ptr)
/*---------------------------------------------------------------------*/
{
  AppCtx         *user = (AppCtx*)ptr;
  Parameter      *param = user->param;
  int            ierr,i,j,mx,mz,sh,ish,ic,jc,iw,jw;
  int            xints,xinte,yints,yinte;
  PetscScalar    dudxN,dudxS,dudxW,dudxE,dudzN,dudzS,fr;
  PetscScalar    dwdzE,dwdzW,dwdxW,dwdxE,dwdzN,dwdzS;
  PetscScalar    pN,pS,pE,pW,uE,uW,wN,wS,wE,wW,vNE,vNW,vSE,vSW;
  PetscScalar    TN,TS,TE,TW,dTdxW,dTdxE,dTdzN,dTdzS,TNE,TNW,TSE,TSW,dTdxN,dTdxS;
  PassiveScalar  dx, dz, dxp, dzp, c, d, beta, itb, sbi, pe;
  PassiveScalar  xp, zp,  cb, sb, skt, xtrp;
  PassiveScalar  eps = 0.000000001;


  PetscFunctionBegin;

  /* 
     Define geometric and numeric parameters
  */
  pe = param->peclet;  beta = param->slab_dip;
  ic = param->icorner;   jc = param->jcorner; 
  CalcJunk(param->kappa,param->slab_age,beta,&skt,&cb,&sb,&itb,&sbi);
  c = beta*sb/(beta*beta-sb*sb);
  d = (beta*cb-sb)/(beta*beta-sb*sb); 
  
  /* 
     Define global and local grid parameters
  */
  mx  = info->mx;                          mz  = info->my;
  dx  = param->scaled_width/((double)(mx-2));     
  dz  = param->scaled_depth/((double)(mz-2));
  dxp = dx;                                dzp = dz*sbi;  
  xints = info->xs;                      xinte = info->xs+info->xm; 
  yints = info->ys;                      yinte = info->ys+info->ym;
  fr = (1.0-dz/dx*itb)/2.0;
  xtrp = (xinte - ic - 0.5)*dx; 

  /* 
     Stokes equation, no buoyancy terms, constant viscosity
     Steady state advection-diffusion equation for temperature
  */

  for (j=yints; j<yinte; j++) { 
    jw = j-jc+1;
    for (i=xints; i<xinte; i++) {
      iw = i-ic+1;

      /* MOMENTUM AND CONTINUITY EQUATIONS */
      if (i<ic) {                                           /* Buffer Points */
	f[j][i].p = x[j][i].p - 0.0;
	f[j][i].u = x[j][i].u - cb;
	f[j][i].w = x[j][i].w - sb;
      } else if (j<jc) {
	f[j][i].p = x[j][i].p - 0.0;
	f[j][i].u = x[j][i].u - 0.0;
	f[j][i].w = x[j][i].w - 0.0;
      } else if ( (i==xinte-1)||(j==yinte-1) /*|| (i>0)||(j>0) */ /*||((iw==1)&&(jw==1))*/ ) {
	/* horizontal velocity */
	zp = (jw-0.5)*dz; xp = iw*dx+zp*itb;
	f[j][i].u = x[j][i].u - HorizVelocity(xp,zp,c,d);
	/* vertical velocity */
	zp = jw*dz; xp = (iw-0.5)*dx+zp*itb;
	f[j][i].w = x[j][i].w - VertVelocity(xp,zp,c,d);
	/* pressure */
	zp = (jw-0.5)*dz; xp = (iw-0.5)*dx+zp*itb;
	f[j][i].p = x[j][i].p - Pressure(xp,zp,c,d); 
      } else {                                              /* Interior Points */
	/* Horizontal velocity */
	pW = x[j][i].p; pE = x[j][i+1].p;
	vNE = UInterp(x,i,j,fr); vNW = UInterp(x,i-1,j,fr);
	dudxE = ( x[j][i+1].u  - x[j][i].u   )/dxp;
	dudxW = ( x[j][i].u    - x[j][i-1].u )/dxp;
	dudzN = ( x[j+1][i].u  - x[j][i].u   )/dzp;
	dudxN = ( vNE - vNW )/dxp;
	if (j==jc) {
	  xp = iw*dxp;
	  dudzS = sb*(HorizVelocity(xp,eps,c,d)-HorizVelocity(xp,-eps,c,d))/eps/2.0;
	  dudxS = 0.0;
	} else { 
	  dudzS = ( x[j][i].u  - x[j-1][i].u )/dzp;
	  vSE = UInterp(x,i,j-1,fr); vSW = UInterp(x,i-1,j-1,fr);
	  dudxS = ( vSE - vSW )/dxp;
	}
	f[j][i].u = -( pE - pW )/dxp +  /* X-MOMENTUM */
	             ( dudxE - dudxW )/dxp * (1.0+itb*itb) +
	             ( dudzN - dudzS )/dzp * sbi*sbi -
	             ( dudxN - dudxS )/dzp * 2.0*itb*sbi;

	/* Vertical velocity */
	pE = PInterp(x,i,j,fr);	pS = x[j][i].p; pN = x[j+1][i].p;
	vNE =  WInterp(x,i,j,fr); vSE = WInterp(x,i,j-1,fr); 
	dwdzN = ( x[j+1][i].w - x[j][i].w   )/dzp;
	dwdzS = ( x[j][i].w   - x[j-1][i].w )/dzp;
	dwdxE = ( x[j][i+1].w - x[j][i].w   )/dxp;
	dwdzE = ( vNE - vSE )/dzp;
	if (i==ic) {
	  zp = jw*dz; xp = zp*itb;
	  pW = Pressure(xp,zp,c,d); 
	  dwdxW = ( VertVelocity(xp+eps,zp,c,d) - VertVelocity(xp,zp,c,d) )/eps;
	  dwdzW = 0.0;
	} else { 
	  pW = PInterp(x,i-1,j,fr);
	  dwdxW = ( x[j][i].w - x[j][i-1].w )/dxp;
	  vNW = WInterp(x,i-1,j,fr); vSW = WInterp(x,i-1,j-1,fr); 
	  dwdzW = ( vNW - vSW )/dzp;
	}	
	f[j][i].w =  ( pE - pW )/dxp * itb - /* Z-MOMENTUM */
	             ( pN - pS )/dzp * sbi +
	             ( dwdzN - dwdzS )/dzp * sbi*sbi +
	             ( dwdxE - dwdxW )/dxp * (1.0+itb*itb) - 
	             ( dwdzE - dwdzW )/dxp * 2.0*itb*sbi;
	
	/* pressure */
	uW = x[j][i-1].u; uE = x[j][i].u;
	wS = x[j-1][i].w; wN = x[j][i].w; wE = WInterp(x,i,j-1,fr);
	if (i==ic)
	  wW = sb;
	else
	  wW = WInterp(x,i-1,j-1,fr);
	f[j][i].p = ( uE - uW )/dxp -  /* CONTINUITY */
	            ( wE - wW )/dxp * itb +
	            ( wN - wS )/dzp * sbi;
	
      }
      
      /* TEMPERATURE EQUATION */
      zp = (j-0.5)*dz; xp = (iw-0.5)*dx+zp*itb;
      if (i==0) {    /* dirichlet on boundary along slab side */
	f[j][i].T = x[j][i].T - 1.0;
      } else if (j==0) {       /* force T=0 on surface */
	f[j][i].T = x[j][i].T + x[j+1][i].T;
      }else if ( xp<=0.0 ) {   /* slab inflow dirichlet */
	f[j][i].T = x[j][i].T - erf(zp*param->lid_depth/2.0/skt);
      } else if (j==yinte-1) { /* neumann on outflow boundary */
	f[j][i].T = x[j][i].T - x[j-1][i].T;
      } else if (xp>=20.0) {  /* THIS IS FOR THE BENCHMARK ONLY */ 
	if (j<jc) { /* linear profile on lid bound */
	  f[j][i].T = x[j][i].T - zp;
	} else {    /* dirichlet on inflow boundary */
	  f[j][i].T = x[j][i].T - 1.0;
	}
      } else {                 /* interior of the domain */

	uW = x[j][i-1].u; uE = x[j][i].u;
	wS = x[j-1][i].w; wN = x[j][i].w; 
	wE = WInterp(x,i,j-1,fr); wW = WInterp(x,i-1,j-1,fr);

	if (i==ic) { /* Just east of slab-wedge interface */
	  if (j<jc) {
	    wW = wE = wN = wS = 0.0; 
	    uW = uE = 0.0;
	  } else {
	    wW = sb;
	  }
	}

	if (i==ic-1) { /* Just west of slab-wedge interface */
	    wE = wW = sb;
	}

	TN  = ( x[j][i].T + x[j+1][i].T )/2.0;  TS  = ( x[j][i].T + x[j-1][i].T )/2.0;
	TE  = ( x[j][i].T + x[j][i+1].T )/2.0;  TW  = ( x[j][i].T + x[j][i-1].T )/2.0;
	TNE =   TInterp(x,i,j,fr);              TNW =   TInterp(x,i-1,j,fr);
	TSE =   TInterp(x,i,j-1,fr);            TSW =   TInterp(x,i-1,j-1,fr);
	dTdxN = ( TNE - TNW )/dxp;
	dTdxS = ( TSE - TSW )/dxp;
	dTdzN = ( x[j+1][i].T - x[j][i].T   )/dzp;
	dTdzS = ( x[j][i].T   - x[j-1][i].T )/dzp;
	dTdxE = ( x[j][i+1].T - x[j][i].T   )/dxp;
	dTdxW = ( x[j][i].T   - x[j][i-1].T )/dxp;
	
	f[j][i].T = ( ( dTdzN - dTdzS )/dzp * sbi*sbi + /* diffusion term */
		      ( dTdxE - dTdxW )/dxp * (1.0+itb*itb) -
		      ( dTdxN - dTdxS )/dzp * 2.0*itb*sbi  )*dx*dz/pe -

     	            ( ( wN*TN - wS*TS )*dxp +           /* advection term */
		      ( uE*sb - wE*cb )*TE    *dzp +
		      (-uW*sb + wW*cb )*TW    *dzp );
      }
    }
  }
  
  PetscFunctionReturn(0);
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
/* ------------------------------------------------------------------- */
int FormInitialGuess(SNES snes,Vec X,void *ptr)
{
  DMMG      dmmg = (DMMG)ptr;
  AppCtx    *user = (AppCtx*)dmmg->user;
  int       ierr;

  ierr = VecCopy(user->Xold, X); CHKERRQ(ierr);

  /* calculate the residual on fine mesh, but only the first time this is called */
  if (user->fnorm_ini == 0.0) {
    ierr = SNESComputeFunction(snes,user->Xold,user->func);
    ierr = VecNorm(user->func,NORM_2,&user->fnorm_ini);CHKERRQ(ierr);
  } 
  
  return 0;
} 

/*--------------------------UTILITY FUNCTION BELOW THIS LINE-----------------------------*/ 

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "UInterp"
PetscScalar UInterp(Field **x, int i, int j, PetscScalar fr)
/*---------------------------------------------------------------------*/
{
  PetscScalar p,m;
  p = (1.0-fr)*x[j+1][i].u + fr*x[j+1][i+1].u;
  m = (1.0-fr)*x[j][i+1].u + fr*x[j][i].u;
  return (p + m)/2.0;
}

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "WInterp"
PetscScalar WInterp(Field **x, int i, int j, PetscScalar fr)
/*---------------------------------------------------------------------*/
{
  PetscScalar p,m;
  p = (1.0-fr)*x[j+1][i].w + fr*x[j+1][i+1].w;
  m = (1.0-fr)*x[j][i+1].w + fr*x[j][i].w;
  return (p + m)/2.0;
}

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "PInterp"
PetscScalar PInterp(Field **x, int i, int j, PetscScalar fr)
/*---------------------------------------------------------------------*/
{
  PetscScalar p,m;
  p = (1.0-fr)*x[j+1][i].p + fr*x[j+1][i+1].p;
  m = (1.0-fr)*x[j][i+1].p + fr*x[j][i].p;
  return (p + m)/2.0;
}

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "TInterp"
PetscScalar TInterp(Field **x, int i, int j, PetscScalar fr)
/*---------------------------------------------------------------------*/
{
  PetscScalar p,m;
  p = (1.0-fr)*x[j+1][i].T + fr*x[j+1][i+1].T;
  m = (1.0-fr)*x[j][i+1].T + fr*x[j][i].T;
  return (p + m)/2.0;
}

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "HorizVelocity"
PetscScalar HorizVelocity(PetscScalar x, PetscScalar z, PetscScalar c, PetscScalar d)
/*---------------------------------------------------------------------*/
 {
   PetscScalar r, st, ct, th;
   r = sqrt(x*x+z*z);
   st = z/r;  ct = x/r;  th = atan(z/x); 
   return ct*(c*th*st+d*(st+th*ct)) + st*(c*(st-th*ct)+d*th*st);
}

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "VertVelocity"
PetscScalar VertVelocity(PetscScalar x, PetscScalar z, PetscScalar c, PetscScalar d)
/*---------------------------------------------------------------------*/
 {
   PetscScalar r, st, ct, th;
   r = sqrt(x*x+z*z);
   st = z/r;  ct = x/r;  th = atan(z/x); 
   return st*(c*th*st+d*(st+th*ct)) - ct*(c*(st-th*ct)+d*th*st);
}

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "Pressure"
PetscScalar Pressure(PetscScalar x, PetscScalar z, PetscScalar c, PetscScalar d)
/*---------------------------------------------------------------------*/
{
  PetscScalar r, st, ct;
  r = sqrt(x*x+z*z);
  st = z/r;  ct = x/r;  
  return (-2.0*(c*ct-d*st)/r);
}

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "CalcJunk"
void CalcJunk(double kappa,double slab_age,double beta,double *skt,double *cb,double *sb, double *itb,
              double *sbi)
/*---------------------------------------------------------------------*/
{
  PetscReal SEC_PER_YR = 3600.00*24.00*356.2500;

  *skt = sqrt(kappa*slab_age*SEC_PER_YR);
  *cb  = cos(beta); *sb = sin(beta);
  *itb = 1.0/tan(beta); *sbi = 1.0/(*sb);
}
