
static char help[] = "Steady-state 2D subduction flow, pressure and temperature solver.\n\\n\
The flow is driven by the subducting slab.\n\
  -ivisc <#> = rheology option.\n\
      0 --- constant viscosity.\n\
      1 --- olivine diffusion creep rheology (T-dependent, newtonian).\n\
      2 --- weak temperature dependent rheology (1200/T, newtonian).\n\
  -ibound <#> = boundary condition \n\
      0 --- isoviscous analytic.\n\
      1 --- stress free. \n\
      2 --- stress is von neumann. \n\
  -icorner <#> = i index of wedge corner point.\n\
  -jcorner <#> = j index of wedge corner point.\n\
  -slab_dip <#> = dip of the subducting slab in DEGREES.\n\
  -back_arc <#> = distance from trench to back-arc in KM.(if unspecified then no back-arc). \n\
  -u_back_arcocity <#> = full spreading rate of back arc as a factor of slab velocity. \n\
  -width <#> = width of domain in KM.\n\
  -depth <#> = depth of domain in KM.\n\
  -lid_depth <#> = depth to the base of the lithosphere in KM.\n\
  -slab_dip <#> = dip of the subducting slab in DEGREES.\n\
  -slab_velocity <#> = velocity of slab in CM/YEAR.\n\
  -slab_age <#> = age of slab in MILLIONS OF YEARS. \n\
  -potentialT <#> = mantle potential temperature in degrees CENTIGRADE.\n\
  -kappa <#> = thermal diffusivity in M^2/SEC. \n\
  -peclet <#> = dimensionless Peclet number (default 111.691)\n\\n";

/*T
   Concepts: SNES^solving a system of nonlinear equations (parallel multicomponent example);
   Concepts: DA^using distributed arrays;
   Concepts: multicomponent
   Processors: n
T*/

/* ------------------------------------------------------------------------

    This problem is modeled by the partial differential equation system
  
         -Grad(P) + Div[eta (Grad(v) + Grad(v)^T)] = 0
 	 Div(U,W) = 0

    which is uniformly discretized on a staggered mesh:
                       ------w_ij------
                      /               /
                  u_i-1j    P_ij    u_ij
                    /               /
 		    ------w_ij-1----

  ------------------------------------------------------------------------- */

/* 
   Include "petscda.h" so that we can use distributed arrays (DAs).
   Include "petscsnes.h" so that we can use SNES solvers.  Note that this
   file automatically includes:
     petsc.h       - base PETSc routines   petscvec.h - vectors
     petscsys.h    - system routines       petscmat.h - matrices
     petscis.h     - index sets            petscksp.h - Krylov subspace methods
     petscviewer.h - viewers               petscpc.h  - preconditioners
     petscksp.h   - linear solvers 
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
} Field;

typedef struct {
  int          mglevels;
  PetscReal    width, depth, scaled_width, scaled_depth, peclet, potentialT;
  PetscReal    slab_dip, slab_age, slab_velocity, lid_depth, kappa;
  PetscReal    u_back_arc, x_back_arc;
  int          icorner, jcorner, ivisc, ifromm, ibound;
  PetscTruth   PreLoading, back_arc;
} Parameter;

typedef struct {
  Vec          Xold,func;
  Parameter    *param;
  PetscReal    fnorm_ini, fnorm;
} AppCtx;

PetscReal HALFPI = 3.14159265358979323846/2.0;

int SetParams(Parameter *param, int mx, int mz);
extern int FormInitialGuess(SNES,Vec,void*);
extern int FormFunctionLocal(DALocalInfo*,Field**,Field**,void*);
extern PassiveScalar HorizVelocity(PassiveScalar x, PassiveScalar z, PassiveScalar c, PassiveScalar d);
extern PassiveScalar VertVelocity(PassiveScalar x, PassiveScalar z, PassiveScalar c, PassiveScalar d);
extern PassiveScalar Pressure(PassiveScalar x, PassiveScalar z, PassiveScalar c, PassiveScalar d);
extern PassiveScalar LidVelocity(PassiveScalar x, PassiveScalar xBA, PetscTruth BA);
extern PetscScalar Viscosity(PetscScalar T, int iVisc);
extern PetscScalar UInterp(Field **x, int i, int j, PetscScalar fr);
extern PetscScalar WInterp(Field **x, int i, int j, PetscScalar fr);
extern PetscScalar PInterp(Field **x, int i, int j, PetscScalar fr);
extern PetscScalar TInterp(Field **x, int i, int j, PetscScalar fr);
extern void CalcJunk(PassiveScalar,PassiveScalar,PassiveScalar,PassiveScalar*,PassiveScalar*,PassiveScalar*,PassiveScalar*,PassiveScalar*);
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
  int        i,ierr,tmpVisc,tmpBound;
  MPI_Comm   comm;
  SNES       snes;
  DA         da;
  Vec        res;
  PetscReal  SEC_PER_YR = 3600.00*24.00*356.2500;

  PetscInitialize(&argc,&argv,(char *)0,help);
  PetscOptionsSetValue("-preload","off");
  PetscOptionsSetValue("-mat_type","seqaij");
  PetscOptionsSetValue("-snes_monitor",PETSC_NULL);
  PetscOptionsSetValue("-ksp_truemonitor",PETSC_NULL);
  PetscOptionsSetValue("-pc_type","lu");
  PetscOptionsInsert(&argc,&argv,PETSC_NULL);

  comm = PETSC_COMM_WORLD;
  PreLoadBegin(PETSC_TRUE,"SetUp");

    param.PreLoading = PreLoading;  //where is PreLoading defined?
    ierr = DMMGCreate(comm,1,&user,&dmmg);CHKERRQ(ierr); 
    param.mglevels = DMMGGetLevels(dmmg);

    /*
      Create distributed array multigrid object (DMMG) to manage parallel grid and vectors
      for principal unknowns (x) and governing residuals (f)
    */
    ierr = DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_BOX,-4,-4,PETSC_DECIDE,PETSC_DECIDE,4,2,0,0,&da);CHKERRQ(ierr);
    ierr = DMMGSetDM(dmmg,(DM)da);CHKERRQ(ierr);
    ierr = DADestroy(da);CHKERRQ(ierr);
    ierr = DAGetInfo(DMMGGetDA(dmmg),0,&mx,&mz,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,PETSC_IGNORE,
                     PETSC_IGNORE,PETSC_IGNORE);CHKERRQ(ierr);

    /* 
      Problem parameters
    */
    ierr = SetParams(&param,mx,mz);CHKERRQ(ierr);

    ierr = DASetFieldName(DMMGGetDA(dmmg),0,"x-velocity");CHKERRQ(ierr);
    ierr = DASetFieldName(DMMGGetDA(dmmg),1,"y-velocity");CHKERRQ(ierr);
    ierr = DASetFieldName(DMMGGetDA(dmmg),2,"pressure");CHKERRQ(ierr);
    ierr = DASetFieldName(DMMGGetDA(dmmg),3,"temperature");CHKERRQ(ierr);

    /*======================================================================*/
    
    ierr = PetscMalloc(param.mglevels*sizeof(AppCtx),&user);CHKERRQ(ierr);
    for (i=0; i<param.mglevels; i++) {
      ierr = VecDuplicate(dmmg[i]->x, &(user[i].Xold));CHKERRQ(ierr);
      ierr = VecDuplicate(dmmg[i]->x, &(user[i].func));CHKERRQ(ierr);
      user[i].param = &param;
      dmmg[i]->user = &user[i]; 
    }
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create user context, set problem data, create vector data structures.
       Also, compute the initial guess.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Create nonlinear solver context
       Process NOT adiC(100): WInterp UInterp PInterp TInterp FormFunctionLocal Viscosity
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /* ierr = DMMGSetSNESLocal(dmmg,FormFunctionLocal,0,ad_FormFunctionLocal,admf_FormFunctionLocal);CHKERRQ(ierr);*/
    ierr = DMMGSetSNESLocal(dmmg,FormFunctionLocal,0,0,0);CHKERRQ(ierr); 
    ierr = DMMGSetInitialGuess(dmmg,FormInitialGuess);CHKERRQ(ierr);
        
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Solve the nonlinear system
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PreLoadStage("Solve"); 
    ierr = Initialize(dmmg);CHKERRQ(ierr);

    if (param.ivisc>0) { 
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Doing Constant Viscosity Solve\n", its);CHKERRQ(ierr);
      tmpVisc = param.ivisc; param.ivisc=0; tmpBound = param.ibound; param.ibound=0; 
      snes = DMMGGetSNES(dmmg);
      ierr = SNESSetTolerances(snes,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,2,PETSC_DEFAULT);CHKERRQ(ierr);
      ierr = DMMGSolve(dmmg);CHKERRQ(ierr);
      ierr = SNESSetTolerances(snes,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,50,PETSC_DEFAULT);CHKERRQ(ierr);
      ierr = VecCopy(DMMGGetx(dmmg),user->Xold);CHKERRQ(ierr);
      param.ivisc = tmpVisc; param.ibound=tmpBound; 
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Doing Variable Viscosity Solve\n", its);CHKERRQ(ierr);
    }
    if (param.ifromm==1)
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Using Fromm advection scheme\n", its);CHKERRQ(ierr);
    ierr = DMMGSolve(dmmg);CHKERRQ(ierr); 
    snes = DMMGGetSNES(dmmg);
    ierr = SNESGetIterationNumber(snes,&its);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Number of Newton iterations = %d\n", its);CHKERRQ(ierr);
    
    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
      Output stuff to Matlab socket.
      - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = SNESGetFunction(snes, &res, PETSC_NULL, PETSC_NULL);CHKERRQ(ierr);
    ierr = VecView(res, PETSC_VIEWER_SOCKET_WORLD);CHKERRQ(ierr); 
    ierr = VecView(DMMGGetx(dmmg),PETSC_VIEWER_SOCKET_WORLD);CHKERRQ(ierr);  
    ierr = PetscViewerSocketPutInt(PETSC_VIEWER_SOCKET_WORLD,1,&mx);CHKERRQ(ierr); 
    ierr = PetscViewerSocketPutInt(PETSC_VIEWER_SOCKET_WORLD,1,&mz);CHKERRQ(ierr); 
    ierr = PetscViewerSocketPutReal(PETSC_VIEWER_SOCKET_WORLD,1,1,&(param.slab_dip));CHKERRQ(ierr); 
    ierr = PetscViewerSocketPutReal(PETSC_VIEWER_SOCKET_WORLD,1,1,&(param.scaled_width));CHKERRQ(ierr); 
    ierr = PetscViewerSocketPutReal(PETSC_VIEWER_SOCKET_WORLD,1,1,&(param.scaled_depth));CHKERRQ(ierr); 
    ierr = PetscViewerSocketPutReal(PETSC_VIEWER_SOCKET_WORLD,1,1,&(param.lid_depth));CHKERRQ(ierr); 
    ierr = PetscViewerSocketPutReal(PETSC_VIEWER_SOCKET_WORLD,1,1,&(param.potentialT));CHKERRQ(ierr);
    ierr = PetscViewerSocketPutReal(PETSC_VIEWER_SOCKET_WORLD,1,1,&(param.x_back_arc));CHKERRQ(ierr);
    ierr = PetscViewerSocketPutInt(PETSC_VIEWER_SOCKET_WORLD,1,&(param.icorner));CHKERRQ(ierr); 
    ierr = PetscViewerSocketPutInt(PETSC_VIEWER_SOCKET_WORLD,1,&(param.jcorner));CHKERRQ(ierr);  
    ierr = PetscViewerSocketPutInt(PETSC_VIEWER_SOCKET_WORLD,1,&(param.ivisc));CHKERRQ(ierr);    
    ierr = PetscViewerSocketPutInt(PETSC_VIEWER_SOCKET_WORLD,1,&(param.ibound));CHKERRQ(ierr);  
    ierr = PetscViewerSocketPutInt(PETSC_VIEWER_SOCKET_WORLD,1,&(its));CHKERRQ(ierr);  

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    for (i=0; i<param.mglevels; i++) {
      ierr = VecDestroy(user[i].Xold);CHKERRQ(ierr);
      ierr = VecDestroy(user[i].func);CHKERRQ(ierr);
    }
    ierr = PetscFree(user);CHKERRQ(ierr);
    ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
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

  ic = param->icorner; jc = param->jcorner;
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

      if        (i<ic) { /* slab */
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
  /* 
  for (j=ys; j<ys+ym; j++) {
    for (i=xs; i<xs+xm; i++) {
      x[j][i].u = 0.0;
      x[j][i].w = 0.0;
      x[j][i].p = 0.0;
      x[j][i].T = 0.0;
    }
  }
  */

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
  PetscTruth     back_arc;
  int            ierr,i,j,mx,mz,sh,ish,ic,jc,iw,jw,ilim,jlim;
  int            xints,xinte,yints,yinte,ivisc,ifromm,ibound;
  PetscScalar    dudxN,dudxS,dudxW,dudxE,dudzN,dudzS,dudzE,dudzW,dudxC,dudzC;
  PetscScalar    dwdzE,dwdzW,dwdxW,dwdxE,dwdzN,dwdzS,dwdxN,dwdxS,dwdxC,dwdzC;
  PetscScalar    pN,pS,pE,pW,uE,uW,uC,uN,uS,wN,wS,wE,wW,wC;
  PetscScalar    uNE,uNW,uSE,uSW,wNE,wNW,wSE,wSW;
  PetscScalar    vE, vN, vS, vW, TE, TN, TS, TW, TC, pC;
  PetscScalar    fN,fS,fE,fW,dTdxW,dTdxE,dTdzN,dTdzS,TNE,TNW,TSE,TSW,dTdxN,dTdxS;
  PetscScalar    etaN,etaS,etaE,etaW,etaC;
  PassiveScalar  dx, dz, dxp, dzp, c, d, beta, itb, sbi, pe;
  PassiveScalar  xp, zp,  cb, sb, skt, z_scale, fr, x_back_arc, u_back_arc;
  PassiveScalar  eps=0.000000001, alpha_g_on_cp_units_inverse_km=4.0e-5*9.8;


  PetscFunctionBegin;

  /* 
     Define geometric and numeric parameters
  */
  back_arc = param->back_arc; x_back_arc = param->x_back_arc; u_back_arc=param->u_back_arc;
  pe = param->peclet;  beta = param->slab_dip; 
  ivisc = param->ivisc; ifromm = param->ifromm;   ibound = param->ibound;
  z_scale = param->lid_depth * alpha_g_on_cp_units_inverse_km;
  ic = param->icorner;   jc = param->jcorner; 
  CalcJunk(param->kappa,param->slab_age,beta,&skt,&cb,&sb,&itb,&sbi);
  c = beta*sb/(beta*beta-sb*sb);
  d = (beta*cb-sb)/(beta*beta-sb*sb); 
  
  /* 
     Define global and local grid parameters
  */
  mx   = info->mx;                         mz   = info->my;
  ilim = mx-1;                             jlim = mz-1;
  dx   = param->scaled_width/((double)(mx-2));     
  dz   = param->scaled_depth/((double)(mz-2));
  dxp  = dx;                                dzp = dz*sbi;  
  xints = info->xs;                       xinte = info->xs+info->xm; 
  yints = info->ys;                       yinte = info->ys+info->ym;
  fr   = (1.0-dz/dx*itb)/2.0; 

  /* 
     Stokes equation, no buoyancy terms
     Steady state advection-diffusion equation for temperature
  */

  for (j=yints; j<yinte; j++) { 
    jw = j-jc+1;
    for (i=xints; i<xinte; i++) {
      iw = i-ic+1;

      /* X-MOMENTUM/VELOCITY */
      if        (i<ic) { /* within the slab */     
	f[j][i].u = x[j][i].u - cb;
      } else if (j<jc) { /* within the lid */
	f[j][i].u = x[j][i].u - u_back_arc*LidVelocity(xp,x_back_arc,back_arc);
      } else if (i==ilim) { /* On the INFLOW boundary */
	if (ibound==0) { /* isoviscous analytic soln */
	  zp = (jw-0.5)*dz; xp = iw*dx+zp*itb;
	  f[j][i].u = x[j][i].u - HorizVelocity(xp,zp,c,d);
	} else { /* normal stress = 0 boundary condition */
	  uE = x[j][i].u; uW = x[j][i-1].u; pC = x[j][i].p;
	  TC = param->potentialT * x[j][i].T * exp( (j-0.5)*dz*z_scale );
	  etaC = Viscosity(TC,ivisc);
	  f[j][i].u = 2.0*etaC*( uE - uW )/dxp - pC;
       	}
      } else if (j==jlim) { /* On the OUTFLOW boundary */
	if (ibound==0) { /* isoviscous analytic soln */
	  zp = (jw-0.5)*dz; xp = iw*dx+zp*itb;
	  f[j][i].u = x[j][i].u - HorizVelocity(xp,zp,c,d);
	} else { /* shear stress = 0 boundary condition */
	  uN = x[j][i].u;   wE = x[j-1][i+1].w;   
	  uS = x[j-1][i].u; wW = x[j-1][i].w;
	  uW = UInterp(x,i-1,j-1,fr); uE = UInterp(x,i,j-1,fr);
	  f[j][i].u =  sbi*( uN - uS )/dzp 
	             - itb*( uE - uW )/dxp
                     +     ( wE - wW )/dxp;
	}
      } else {     /* Mantle wedge, horizontal velocity */

	pW = x[j][i].p; pE = x[j][i+1].p;

	TN = param->potentialT * TInterp(x,i,j,fr)   * exp(  j     *dz*z_scale );
	TS = param->potentialT * TInterp(x,i,j-1,fr) * exp( (j-1.0)*dz*z_scale );
	TE = param->potentialT * x[j][i+1].T         * exp( (j-0.5)*dz*z_scale );
	TW = param->potentialT * x[j][i].T           * exp( (j-0.5)*dz*z_scale );
	etaN = Viscosity(TN,ivisc); etaS = Viscosity(TS,ivisc);
	etaW = Viscosity(TW,ivisc); etaE = Viscosity(TE,ivisc);
	/*if (j==jc) etaS = 1.0;*/

	/* ------ BEGIN VAR VISCOSITY USE ONLY ------- */
	dwdxN = etaN * ( x[j][i+1].w   - x[j][i].w   )/dxp;
	dwdxS = etaS * ( x[j-1][i+1].w - x[j-1][i].w )/dxp;
	if (i<ilim-1) { wE = WInterp(x,i+1,j-1,fr); }
	else { wE = ( x[j][i].w + x[j-1][i].w )/2.0; }
	wC = WInterp(x,i,j-1,fr); 
	wW = WInterp(x,i-1,j-1,fr);   if (i==ic) wW = sb;
	dwdxE = etaE * ( wE - wC )/dxp; 
	dwdxW = etaW * ( wC - wW )/dxp;
	/* ------ END VAR VISCOSITY USE ONLY ------- */

	/* ------ BGN ISOVISC BETA != 0 USE ONLY ------- */
	uNE = UInterp(x,i,j,fr);   uNW = UInterp(x,i-1,j,fr);   
	uSE = UInterp(x,i,j-1,fr); uSW = UInterp(x,i-1,j-1,fr);
	if (j==jc) { 
	  xp = (iw+0.5)*dx+(j-1)*dz*itb; uSE = u_back_arc*LidVelocity(xp,x_back_arc,back_arc);
	  xp = (iw-0.5)*dx+(j-1)*dz*itb; uSW = u_back_arc*LidVelocity(xp,x_back_arc,back_arc);
	}
	dudxN = etaN * ( uNE - uNW )/dxp; dudxS = etaS * ( uSE - uSW )/dxp;
	dudzE = etaE * ( uNE - uSE )/dzp; dudzW = etaW * ( uNW - uSW )/dzp;
	/* ------ END ISOVISC BETA != 0 USE ONLY ------- */

	dudzN = etaN * ( x[j+1][i].u  - x[j][i].u   )/dzp;
	dudzS = etaS * ( x[j][i].u    - x[j-1][i].u )/dzp; 
	dudxE = etaE * ( x[j][i+1].u  - x[j][i].u   )/dxp;
	dudxW = etaW * ( x[j][i].u    - x[j][i-1].u )/dxp;
	if (j==jc) {
	  if (ibound==0) { /* apply isoviscous boundary condition */
	    xp = iw*dx; 
	    dudzS = etaS * sb*(HorizVelocity(xp,eps,c,d)-HorizVelocity(xp,-eps,c,d))/eps/2.0; 
	  } else  /* force u=0 on the lid-wedge interface (off-grid point) */
	    dudzS = etaS * ( 2.0*x[j][i].u - 2.0*x[j-1][i].u )/dzp; 
	}

	f[j][i].u = -( pE - pW )/dxp                         /* X-MOMENTUM EQUATION*/
	            +( dudxE - dudxW )/dxp * (1.0+itb*itb)
	            +( dudzN - dudzS )/dzp * sbi*sbi
                    -( ( dudxN - dudxS )/dzp 
		      +( dudzE - dudzW )/dxp ) * itb*sbi; 

	if (ivisc>0) {
	  f[j][i].u = f[j][i].u + ( dudxE - dudxW )/dxp
	                        + ( dwdxN - dwdxS )/dzp * sbi
	                        - ( dwdxE - dwdxW )/dxp * itb;
	}
      }

      /* Z-MOMENTUM/VELOCITY */
      if        (i<ic) {  /* within the slab */      
	f[j][i].w = x[j][i].w - sb;
      } else if (j<jc) {  /* within the lid */
	f[j][i].w = x[j][i].w - 0.0;
      } else if (j==jlim) { /* On the OUTFLOW boundary */
	if (ibound==0) { /* isoviscous analytic soln */
	  zp = jw*dz; xp = (iw-0.5)*dx+zp*itb;
	  f[j][i].w = x[j][i].w - VertVelocity(xp,zp,c,d);
	} else { /* normal stress = 0 boundary condition */
	  wN = x[j][i].w; wS = x[j-1][i].w; pC = x[j][i].p;
	  wW = WInterp(x,i-1,j-1,fr); if (i==ic) wW = sb;
	  if (i==ilim) wE = ( x[j][i].w + x[j-1][i].w )/2.0;
	  else wE = WInterp(x,i,j-1,fr);
	  TC = param->potentialT * x[j][i].T * exp( (j-0.5)*dz*z_scale );
	  etaC = Viscosity(TC,ivisc);
	  f[j][i].w = 2.0*etaC*( sbi*( wN - wS )/dzp  
                                -itb*( wE - wW )/dxp ) - pC;
	}
      } else if (i==ilim) { /* On the INFLOW boundary */
	if (ibound==0) { /* isoviscous analytic soln */
	  zp = jw*dz; xp = (iw-0.5)*dx+zp*itb;
	  f[j][i].w = x[j][i].w - VertVelocity(xp,zp,c,d);
	} else { /* shear stress = 0 boundary condition */
	  uN = x[j+1][i-1].u; wE = x[j][i].w;   
	  uS = x[j][i-1].u;   wW = x[j][i-1].w;
	  uW = UInterp(x,i-2,j,fr); uE = UInterp(x,i-1,j,fr);
	  f[j][i].w =  sbi*( uN - uS )/dzp 
	             - itb*( uE - uW )/dxp
                     +     ( wE - wW )/dxp;
	  if (j==jlim-1) {
	    f[j][i].w = x[j][i].w - x[j-1][i].w;
	  }
	}
      } else {   /* Mantle wedge, vertical velocity */
	
	pE = PInterp(x,i,j,fr); pW = PInterp(x,i-1,j,fr); pS = x[j][i].p; pN = x[j+1][i].p;
      	if ( (i==ic) && (ibound==0) ) { zp = jw*dz; xp = zp*itb; pW = Pressure(xp,zp,c,d); } 
	
	TN = param->potentialT * x[j+1][i].T         * exp( (j+0.5)*dz*z_scale );
	TS = param->potentialT * x[j][i].T           * exp( (j-0.5)*dz*z_scale );
	TE = param->potentialT * TInterp(x,i,j,fr)   * exp(  j     *dz*z_scale );
	TW = param->potentialT * TInterp(x,i-1,j,fr) * exp(  j     *dz*z_scale );
	etaN = Viscosity(TN,ivisc); etaS = Viscosity(TS,ivisc);
	etaW = Viscosity(TW,ivisc); etaE = Viscosity(TE,ivisc);

 	/* ------ BGN VAR VISCOSITY USE ONLY ------- */
	dudzE = etaE * ( x[j+1][i].u   - x[j][i].u   )/dzp;
	dudzW = etaW * ( x[j+1][i-1].u - x[j][i-1].u )/dzp;
	uE = UInterp(x,i,j,fr);   uC = UInterp(x,i-1,j,fr); 
	uW = UInterp(x,i-2,j,fr); if (i==ic) uW = 2.0*cb - uC;
	dudxE = etaE * ( uE - uC )/dxp; 
	dudxW = etaW * ( uC - uW )/dxp;
	/* ------ END VAR VISCOSITY USE ONLY ------- */

	/* ------ BGN ISOVISC BETA != 0 USE ONLY ------- */
	wNE = WInterp(x,i,j,fr);   wSE = WInterp(x,i,j-1,fr);   
	wNW = WInterp(x,i-1,j,fr); wSW = WInterp(x,i-1,j-1,fr); 
	if (i==ic) { wNW = wSW = sb; }
	dwdzE = etaE * ( wNE - wSE )/dzp; dwdzW = etaW * ( wNW - wSW )/dzp;
	dwdxN = etaN * ( wNE - wNW )/dxp; dwdxS = etaS * ( wSE - wSW )/dxp;
	/* ------ END ISOVISC BETA != 0 USE ONLY ------- */

	dwdzN = etaN * ( x[j+1][i].w - x[j][i].w   )/dzp;
	dwdzS = etaS * ( x[j][i].w   - x[j-1][i].w )/dzp;
	dwdxE = etaE * ( x[j][i+1].w - x[j][i].w   )/dxp;
	dwdxW = etaW * ( x[j][i].w - x[j][i-1].w   )/dxp;
	if (i==ic) { 
	  if (ibound==0) { /* apply isoviscous boundary condition */
	    zp = jw*dz; xp = itb*zp;
	    dwdxW = etaW * ( VertVelocity(xp+eps,zp,c,d) - VertVelocity(xp,zp,c,d) )/eps;
	  } else /*  force w=sin(beta) on the slab-wedge interface (off-grid point) */
	    dwdxW = etaW * ( 2.0*x[j][i].w - 2.0*sb )/dxp; 
	}

	 /* Z-MOMENTUM */
	f[j][i].w =  ( pE - pW )/dxp * itb                 /* constant viscosity terms */                  
	            -( pN - pS )/dzp * sbi 
	            +( dwdzN - dwdzS )/dzp * sbi*sbi
	            +( dwdxE - dwdxW )/dxp * (itb*itb+1.0)
                    -( ( dwdzE - dwdzW )/dxp               
		      +( dwdxN - dwdxS )/dzp ) * itb*sbi;

	if (ivisc>0) {
	  f[j][i].w = f[j][i].w + ( dwdxE - dwdxW )/dxp * itb*itb
	                        + ( dwdzN - dwdzS )/dzp * sbi*sbi
                                -( ( dwdzE - dwdzW )/dxp               
		                  +( dwdxN - dwdxS )/dzp ) * itb*sbi
	                        - ( dudxE - dudxW )/dxp * itb
	                        + ( dudzE - dudzW )/dxp * sbi;
	}
      }
      
      /* CONTINUITY/PRESSURE */
      if ( (j<jc) || (i<ic-1) ) { /* within slab or lid */
	f[j][i].p = x[j][i].p - 0.0;

      } else if ( (j==jlim)&&(i==ic-1) ) {
	f[j][i].p = x[j][i].p - 0.0;
	
      } else if ( (ibound==0) && ((i==ilim)||(j==jlim)) ) { /* isoviscous inflow/outflow BC */
	zp = (jw-0.5)*dz; xp = (iw-0.5)*dx+zp*itb;
	f[j][i].p = x[j][i].p - Pressure(xp,zp,c,d); 

      } else if (i==ic-1) { /* just west of the slab-wedge interface constrain 
			       pressure using the x-momentum equation */

	pE = x[j][i+1].p; pW = x[j][i].p; /* pW IS THE UNKNOWN */
	
	TN = param->potentialT * TInterp(x,i,j,fr)   * exp(  j     *dz*z_scale );
	TS = param->potentialT * TInterp(x,i,j-1,fr) * exp( (j-1.0)*dz*z_scale );
	TE = param->potentialT * x[j][i+1].T         * exp( (j-0.5)*dz*z_scale );
	TW = param->potentialT * x[j][i].T           * exp( (j-0.5)*dz*z_scale );
	etaN = Viscosity(TN,ivisc); etaS = Viscosity(TS,ivisc);
	etaW = Viscosity(TW,ivisc); etaE = Viscosity(TE,ivisc);

	/* ------ BGN VAR VISCOSITY USE ONLY ------- */
	dwdxN = etaN * ( x[j][i+1].w   - (2*sb - x[j][i+1].w)   )/dxp;
	dwdxS = etaS * ( x[j-1][i+1].w - (2*sb - x[j-1][i+1].w) )/dxp;
	wE = WInterp(x,i+1,j-1,fr);   wC = sb; wW = 2*sb - wE;
	dwdxE = etaE * ( wE - wC )/dxp; 
	dwdxW = etaW * ( wC - wW )/dxp;
	/* ------ END VAR VISCOSITY USE ONLY ------- */

	/* ------ BGN BETA != 0 USE ONLY ------- */
	uNE = UInterp(x,i,j,fr);   uNW = 2*cb - uNE;
	uSE = UInterp(x,i,j-1,fr); uSW = 2*cb - uSE;
	if (j==jc) uSE = 0.0;
	dudxN = etaN * ( uNE - uNW )/dxp; dudxS = etaS * ( uSE - uSW )/dxp;
	dudzE = etaE * ( uNE - uSE )/dzp; dudzW = 0.0;
	if (j==jc) dudxS = 0.0;
	/* ------ BGN BETA != 0 USE ONLY ------- */

	dudxE = etaE * ( x[j][i+1].u  - x[j][i].u   )/dxp;
	dudxW = 0.0;

	f[j][i].p = -( pE - pW )/dxp                                 /* X-MOMENTUM */
	            +( dudxE - dudxW )/dxp * (1.0+itb*itb)
                    -( ( dudxN - dudxS )/dzp 
		      +( dudzE - dudzW )/dxp ) * itb*sbi; 

	if (ivisc>0) {
	  f[j][i].p = f[j][i].p + ( dudxE - dudxW )/dxp
	                        + ( dwdxN - dwdxS )/dzp * sbi
	                        - ( dwdxE - dwdxW )/dxp * itb;
	}
      } else { /* interior of the domain */
	uW = x[j][i-1].u; uE = x[j][i].u;
	wS = x[j-1][i].w; wN = x[j][i].w; 
	wW = WInterp(x,i-1,j-1,fr); if (i==ic) wW = sb;
	if (i==ilim) wE = ( x[j][i].w + x[j-1][i].w )/2.0;
	else wE = WInterp(x,i,j-1,fr);

	f[j][i].p = ( uE - uW )/dxp   /* CONTINUITY ON PRESSURE POINTS */
	           -( wE - wW )/dxp * itb
	           +( wN - wS )/dzp * sbi;

	if ( (ivisc==10)&&(i<ilim)&&(j<jlim) ) {
	  wW = x[j][i].w;   wN = WInterp(x,i,j,fr);
	  wE = x[j][i+1].w; wS = WInterp(x,i,j-1,fr);
	  uW = UInterp(x,i-1,j,fr);  uE = UInterp(x,i,j,fr);
	
	  f[j][i].p = f[j][i].p + 
	              ( uE - uW )/dxp -  /* CONTINUITY ON VACANT POINTS */
	              ( wE - wW )/dxp * itb +
	              ( wN - wS )/dzp * sbi;
	}
 
      }
      
      /* TEMPERATURE EQUATION */
      zp = (j-0.5)*dz; xp = (iw-0.5)*dx+zp*itb;
      if (i<=1) {    /* dirichlet on boundary along slab side */
	f[j][i].T = x[j][i].T - 1.0;
      } else if ( (j<=2) && (i<ic) ) {   /* slab inflow dirichlet */
	f[j][i].T = x[j][i].T - erf(-xp*sb*param->lid_depth/2.0/skt);
      } else if (j==0) {   /* force T=0 on surface */
	f[j][i].T = x[j][i].T + x[j+1][i].T;
      } else if (j>=jlim-1) { /* neumann on outflow boundary */
	if (x[j][i].w<0.0) 
	  f[j][i].T = x[j][i].T - 1.0; 
	else               
	  f[j][i].T = x[j][i].T - x[j-1][i].T;
      } else if (i>=ilim-1) /* (xp>=20.0) */ {  
	if (back_arc && (x[j][i].u>0)) 
	  f[j][i].T = x[j][i].T - x[j][i-1].T;
	else                           
	  f[j][i].T = x[j][i].T - PetscMin(zp,1.0);
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

	if (i==ic-1) wE = sb; /* Just west of slab-wedge interface */

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
		      ( dTdxN - dTdxS )/dzp * 2.0*itb*sbi  )*dx*dz/pe;

	if ( (j<jc) && (i>=ic) ) { /* don't advect in lid */
	    fE = fW = fN = fS = 0.0;

	} else if ( (ifromm==0) ||(i>=ilim-2)||(j>=jlim-2)||(i<=2) ) { /* finite volume advection */
	  TN  = ( x[j][i].T + x[j+1][i].T )/2.0;  TS  = ( x[j][i].T + x[j-1][i].T )/2.0;
	  TE  = ( x[j][i].T + x[j][i+1].T )/2.0;  TW  = ( x[j][i].T + x[j][i-1].T )/2.0;
	  fN = wN*TN*dxp; fS = wS*TS*dxp;           
	  fE = ( uE*sb - wE*cb )*TE*dzp;
	  fW = ( uW*sb - wW*cb )*TW*dzp;
	  
	} else {         /* Fromm advection scheme */
	  vN = wN; vS = wS; vE = uE*sb - wE*cb; vW = uW*sb - wW*cb;
	  fE =     ( vE *(-x[j][i+2].T + 5.0*(x[j][i+1].T+x[j][i].T)-x[j][i-1].T)/8.0 
		     - fabs(vE)*(-x[j][i+2].T + 3.0*(x[j][i+1].T-x[j][i].T)+x[j][i-1].T)/8.0 )*dzp;
	  fW =     ( vW *(-x[j][i+1].T + 5.0*(x[j][i].T+x[j][i-1].T)-x[j][i-2].T)/8.0 
		     - fabs(vW)*(-x[j][i+1].T + 3.0*(x[j][i].T-x[j][i-1].T)+x[j][i-2].T)/8.0 )*dzp;
	  fN =     ( vN *(-x[j+2][i].T + 5.0*(x[j+1][i].T+x[j][i].T)-x[j-1][i].T)/8.0 
		     - fabs(vN)*(-x[j+2][i].T + 3.0*(x[j+1][i].T-x[j][i].T)+x[j-1][i].T)/8.0 )*dxp;
	  fS =     ( vS *(-x[j+1][i].T + 5.0*(x[j][i].T+x[j-1][i].T)-x[j-2][i].T)/8.0 
		     - fabs(vS)*(-x[j+1][i].T + 3.0*(x[j][i].T-x[j-1][i].T)+x[j-2][i].T)/8.0 )*dxp;
	}
	  
	  f[j][i].T = f[j][i].T -
	    ( fE - fW + fN - fS );          
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

  ierr = VecCopy(user->Xold, X);CHKERRQ(ierr);

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
#define __FUNCT__ "SetParams"
int SetParams(Parameter *param, int mx, int mz)
/*---------------------------------------------------------------------*/
{
  int ierr;
  PetscReal  SEC_PER_YR = 3600.00*24.00*356.2500;

  /* domain geometry */
  param->icorner       = (int)(mx/6+1);     /* gridpoints */
  param->jcorner       = (int)((mz-2)/12+1);/* gridpoints */
  param->width         = 1200.0;            /* km */
  param->depth         = 600.0;             /* km */
  param->slab_dip      = HALFPI;            /* 90 degrees */
  ierr = PetscOptionsGetInt(PETSC_NULL, "-icorner",&(param->icorner),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL, "-jcorner",&(param->jcorner),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-width",&(param->width),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-depth",&(param->depth),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-slab_dip",&(param->slab_dip),PETSC_NULL);CHKERRQ(ierr);

  /* back-arc */
  param->back_arc      = PETSC_FALSE;       /* no back arc spreading */
  param->x_back_arc    = 0.0;               /* km */
  param->u_back_arc    = 1.0;               /* full spreading at velocity of slab */
  ierr = PetscOptionsHasName(PETSC_NULL,"-back_arc",&(param->back_arc));CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-back_arc",&(param->x_back_arc),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-back_arc_velocity",&(param->u_back_arc),PETSC_NULL);CHKERRQ(ierr);
  if (param->back_arc) {
    PetscPrintf(PETSC_COMM_WORLD,"Dist to back arc = %g km, ",param->x_back_arc);
    PetscPrintf(PETSC_COMM_WORLD,"Full spreading rate of back arc (scaled) = %g \n",param->u_back_arc);
  }

  /* physics parameters */
  param->slab_velocity = 5.0;               /* cm/yr */
  param->slab_age      = 50.0;              /* Ma */
  param->kappa         = 0.7272e-6;         /* m^2/sec */
  param->potentialT    = 1300.0;            /* degrees C */
  param->ivisc         = 0;                 /* 0=constant, 1=diffusion creep, 2=simple T dependent */
  ierr = PetscOptionsGetReal(PETSC_NULL,"-slab_velocity",&(param->slab_velocity),PETSC_NULL);CHKERRQ(ierr); 
  ierr = PetscOptionsGetReal(PETSC_NULL,"-slab_age",&(param->slab_age),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-kappa",&(param->kappa),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(PETSC_NULL,"-potentialT",&(param->potentialT),PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL, "-ivisc",&(param->ivisc),PETSC_NULL);CHKERRQ(ierr);

  /* boundaries */
  param->ibound = param->ivisc;       /* 0=isovisc analytic, 1,2,...= stress free */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-ibound",&(param->ibound),PETSC_NULL);CHKERRQ(ierr);

  /* misc */
  param->ifromm = 1;                 /* advection scheme: 0=finite vol, 1=Fromm */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-ifromm",&(param->ifromm),PETSC_NULL);CHKERRQ(ierr);

  /* unit conversions and derived parameters */
  param->lid_depth = (param->jcorner - 1) * param->depth/((double)(mz-2));
  ierr = PetscOptionsGetReal(PETSC_NULL,"-lid_depth",&(param->lid_depth),PETSC_NULL);CHKERRQ(ierr);
  param->slab_dip =     param->slab_dip*HALFPI/90.0;
  param->scaled_width = param->width/param->lid_depth;
  param->scaled_depth = param->depth/param->lid_depth;
  param->x_back_arc     = param->x_back_arc/param->lid_depth;
  param->peclet =  param->slab_velocity/100.0/SEC_PER_YR /* m/sec */
                 * param->lid_depth*1000.0               /* m */
                 / param->kappa;                         /* m^2/sec */
  ierr = PetscOptionsGetReal(PETSC_NULL,"-peclet",&(param->peclet),PETSC_NULL);CHKERRQ(ierr);  
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Lid depth = %g km, ",param->lid_depth);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Peclet number = %g\n",param->peclet);CHKERRQ(ierr);
  if ( (param->ibound==0) && (param->ivisc>0) ) 
    PetscPrintf(PETSC_COMM_WORLD,"Warning: isoviscous BC may be inconsistent w/ var viscosity!!\n");

  return 0;
}


/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "Viscosity"
PetscScalar Viscosity(PetscScalar T, int iVisc)
/*---------------------------------------------------------------------*/
{
  PetscScalar  result;
  double       p1 = 1.32047792e-12, p2 = 335.0e3/8.314510;
  double       t1 = 1200.0;
  /*
    p1 = exp( -p2/(1473 K) ) so the cutoff for high viscosity 
    occurs at 1200 C.  Below this cutoff, all eta( T<1200 ) = 1.0;
    p2=Ea/R is from van Keken's subroutine
  */

  if (iVisc==0) {        /* constant viscosity */
    result = 1.0;
  } else if (iVisc==1) { /* diffusion creep rheology */
    result = p1*PetscExpScalar(p2/(T+273.0)); 
  } else if (iVisc==2) { /* ad hoc T-dependent rheology */
    result = t1/T;
  } else if (iVisc==3) {
    result = 1.0;
  }

  if (result<1.0)
    return result;
  else
    return 1.0;
}

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "LidVelocity"
PassiveScalar LidVelocity(PassiveScalar x, PassiveScalar xBA, PetscTruth BA)
/*---------------------------------------------------------------------*/
{
  PassiveScalar localize = 10.0;

   if (BA)
     return ( 1.0 + tanh( (x - xBA)*localize ) )/2.0;
   else
     return 0.0;
}

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
