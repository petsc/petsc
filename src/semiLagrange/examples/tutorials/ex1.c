static char help[] = "\n";
/* - - - - - - - - - - - - - - - - - - - - - - - - 
   ex1.c
   Simple example with 3 dof da: u,w,phi
   where u & w are time-independent & analytically prescribed. 
   phi is advected explicitly.
   - - - - - - - - - - - - - - - - - - - - - - - - */
#include "petscsnes.h"
#include "petscda.h"
#include "petscdmmg.h"
#include "petscbag.h"
#include "characteristic.h"

#define SHEAR_CELL     0
#define SOLID_BODY     1
#define FNAME_LENGTH   60

typedef struct field_s {
  PetscReal      u,w,phi;
} Field;

typedef struct parameter_s {
  int            ni, nj, pi, pj;
  PetscReal      sigma,xctr,zctr,L1,L2,LINF;
  int            verify_result, flow_type, sl_event;
  PetscTruth     verify, param_test, output_to_file;
  char           output_filename[FNAME_LENGTH];
  /* timestep stuff */
  PetscReal      t; /* the time */
  int            n; /* the time step */
  PetscReal      t_max, dt, cfl, t_output_interval;
  int            N_steps;
} Parameter;

typedef struct gridinfo_s {
  DAPeriodicType periodic;
  DAStencilType  stencil;
  int            ni,nj,dof,stencil_width,mglevels;
  PetscReal      dx,dz;
} GridInfo;

typedef struct appctx_s {
  Vec          Xold;
  PetscBag     bag;
  GridInfo     *grid;
} AppCtx;

/* Main routines */
int SetParams            (AppCtx*);
int ReportParams         (AppCtx*);
int Initialize           (DMMG*);
int DoSolve              (DMMG*);
int DoOutput             (DMMG*, int);
int CalcSolnNorms        (DMMG*, PetscReal*);
int DoVerification       (DMMG*, AppCtx*);
int DASetFieldNames      (const char*, const char*, const char*, DA);
PetscReal BiCubicInterp  (Field**, PetscReal, PetscReal);
PetscReal CubicInterp    (PetscReal, PetscReal, PetscReal, PetscReal, PetscReal);
PetscTruth OptionsHasName(const char*);

/* characteristic call-backs (static interface) */
PetscErrorCode InterpVelocity2D(void*, PetscReal[], PetscInt, PetscInt[], PetscReal[], void*);
PetscErrorCode InterpFields2D  (void*, PetscReal[], PetscInt, PetscInt[], PetscReal[], void*);

/* a few macros for convenience */
#define REG_REAL(A,B,C,D,E)   ierr=PetscBagRegisterReal(A,B,C,D,E);CHKERRQ(ierr)
#define REG_INTG(A,B,C,D,E)   ierr=PetscBagRegisterInt(A,B,C,D,E);CHKERRQ(ierr)
#define REG_TRUE(A,B,C,D,E)   ierr=PetscBagRegisterTruth(A,B,C,D,E);CHKERRQ(ierr)
#define REG_STRG(A,B,C,D,E,F) ierr=PetscBagRegisterString(A,B,C,D,E,F);CHKERRQ(ierr)
#define REG_ENUM(A,B,C,D,E,F) ierr=PetscBagRegisterEnum(A,B,C,D,E,F);CHKERRQ(ierr)

/*-----------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
/*-----------------------------------------------------------------------*/
{
  DMMG           *dmmg;               /* multilevel grid structure */
  AppCtx         *user;               /* user-defined work context */
  Parameter      *param;
  GridInfo       grid;
  int            ierr,result;
  MPI_Comm       comm;
  DA             da;

  PetscInitialize(&argc,&argv,(char *)0,help);
  comm = PETSC_COMM_WORLD;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set up the problem parameters.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */ 
  ierr = PetscMalloc(sizeof(AppCtx),&user);CHKERRQ(ierr);
  ierr = PetscBagCreate(comm,sizeof(Parameter),&(user->bag));CHKERRQ(ierr);
  user->grid    = &grid;
  ierr = SetParams(user);CHKERRQ(ierr);
  ierr = ReportParams(user);CHKERRQ(ierr);
  ierr = PetscBagGetData(user->bag,(void**)&param);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array multigrid object (DMMG) to manage parallel grid and vectors
     for principal unknowns (x) and governing residuals (f)
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */ 
  ierr = DMMGCreate(comm,grid.mglevels,user,&dmmg);CHKERRQ(ierr); 
  ierr = DACreate2d(comm,grid.periodic,grid.stencil,grid.ni,grid.nj,PETSC_DECIDE,PETSC_DECIDE,grid.dof,grid.stencil_width,0,0,&da);CHKERRQ(ierr);
  ierr = DMMGSetDM(dmmg,(DM)da);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);
  ierr = DAGetInfo(da,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,&(param->pi),&(param->pj),PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  REG_INTG(user->bag,&param->pi,param->pi ,"procs_x","<DO NOT SET> Processors in the x-direction");
  REG_INTG(user->bag,&param->pj,param->pj ,"procs_y","<DO NOT SET> Processors in the y-direction");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create user context, set problem data, create vector data structures.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */   
  ierr = DAGetGlobalVector(da, &(user->Xold));CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize and solve the nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = Initialize(dmmg);CHKERRQ(ierr);
  ierr = DoSolve(dmmg);CHKERRQ(ierr);
  if (param->verify) result = param->verify_result;
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space. 
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DARestoreGlobalVector(da, &(user->Xold));CHKERRQ(ierr);
  ierr = PetscBagDestroy(user->bag);CHKERRQ(ierr); 
  ierr = PetscFree(user);CHKERRQ(ierr);
  ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return result;
}

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "SetParams"
int SetParams(AppCtx *user)
/*---------------------------------------------------------------------*/
{
  PetscBag   bag = user->bag;
  Parameter  *p;
  GridInfo   *grid = user->grid;
  int        ierr, ierr_out=0;
  ierr = PetscBagGetData(bag,(void**)&p);CHKERRQ(ierr);

  /* give the bag a name */
  ierr = PetscBagSetName(bag,"ex1_params","Parameter bag for ex1.c");CHKERRQ(ierr);

  /* verification */
  REG_TRUE(bag,&p->verify,PETSC_FALSE  ,"verify","Do verification run (T/F)");

  /* domain geometry & grid size */
  REG_INTG(bag,&p->ni,40                  ,"ni","Grid points in x-dir");
  REG_INTG(bag,&p->nj,40                  ,"nj","Grid points in y-dir");
  grid->dx = 1.0/((double)(p->ni - 1));
  grid->dz = 1.0/((double)(p->nj - 1));

  /* initial conditions */
  REG_INTG(bag,&p->flow_type,SHEAR_CELL   ,"flow_type","Flow field mode: 0=shear cell, 1=translation");
  REG_REAL(bag,&p->sigma,0.07             ,"sigma","Standard deviation of the gaussian IC");
  REG_REAL(bag,&p->xctr,0.5               ,"xctr","x-position of the center of the gaussian IC");
  REG_REAL(bag,&p->zctr,0.75              ,"zctr","z-position of the center of the gaussian IC");

  /* time stepping */  
  REG_REAL(bag,&p->t_max,1                ,"t_max","Maximum dimensionless time");
  REG_REAL(bag,&p->cfl,5                  ,"cfl","Courant number");
  REG_REAL(bag,&p->t_output_interval,0.1  ,"t_output","Dimensionless time interval for output");
  REG_INTG(bag,&p->N_steps,1000           ,"nsteps","Maximum time-steps");
  REG_INTG(bag,&p->n,1                    ,"nsteps","<DO NOT SET> current time-step");
  REG_REAL(bag,&p->t,0.0                  ,"time","<DO NOT SET> initial time");
  REG_REAL(bag,&p->dt,p->cfl*PetscMin(grid->dx,grid->dz),"dt","<DO NOT SET> time-step size");

  /* output options */
  REG_TRUE(bag,&p->param_test     ,PETSC_FALSE  ,"test","Run parameter test only (T/F)");
  REG_STRG(bag,&p->output_filename,FNAME_LENGTH ,"null","output_file","Name base for output files, set with: -output_file <filename>");
  REG_TRUE(bag,&p->output_to_file,PETSC_FALSE   ,"do_output","<DO NOT SET> flag will be true if you specify an output file name");
  p->output_to_file = OptionsHasName("-output_file");

  if (p->verify) {
    REG_INTG(bag,&p->verify_result,0           ,"ver_result","<DO NOT SET> Result of verification test");
    REG_REAL(bag,&p->L1  ,4924.42              ,"L1","<DO NOT SET> L1");
    REG_REAL(bag,&p->L2  ,496.287              ,"L2","<DO NOT SET> L2");
    REG_REAL(bag,&p->LINF,100                  ,"L3","<DO NOT SET> L3");
    
    p->verify_result = 0; p->L1 = 4924.42; p->L2 = 496.287; p->LINF = 100;
    grid->ni = grid->nj = 40; grid->dx = 1.0/((double)(grid->ni)); grid->dz = 1.0/((double)(grid->nj)); 
    p->flow_type = SHEAR_CELL; p->sigma = 0.07; p->xctr = 0.5; p->zctr = 0.75;   
    p->t_max = 0.5; p->cfl = 5; p->t_output_interval = 0.1; p->dt = p->cfl*PetscMin(grid->dx,grid->dz);
  }

  grid->ni            = p->ni;
  grid->nj            = p->nj;
  grid->periodic      = DA_XYPERIODIC;
  grid->stencil       = DA_STENCIL_BOX;
  grid->dof           = 3;
  grid->stencil_width = 2;
  grid->mglevels      = 1;
  return ierr_out;
}

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "ReportParams"
int ReportParams(AppCtx *user)
/*---------------------------------------------------------------------*/
{
  Parameter  *param;
  GridInfo   *grid = user->grid;
  int        ierr, ierr_out=0;
  ierr = PetscBagGetData(user->bag,(void**)&param);CHKERRQ(ierr);

  PetscPrintf(PETSC_COMM_WORLD,"---------------MOC test 1----------------\n");
  PetscPrintf(PETSC_COMM_WORLD,"Prescribed wind, method of\n"); 
  PetscPrintf(PETSC_COMM_WORLD,"characteristics advection, explicit time-\n");
  PetscPrintf(PETSC_COMM_WORLD,"stepping.\n\n"); 
  if (param->flow_type == 0) {
    PetscPrintf(PETSC_COMM_WORLD,"Flow_type: %d (shear cell).\n\n",param->flow_type); 
  }
  if (param->flow_type == 1) {
    PetscPrintf(PETSC_COMM_WORLD,"Flow_type: %d (rigid body rotation).\n\n",param->flow_type); 
  }
  if (param->verify) {
    PetscPrintf(PETSC_COMM_WORLD," ** VERIFICATION RUN ** \n\n");  
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"  [ni,nj] = %d, %d   [dx,dz] = %5.4g, %5.4g\n",grid->ni,grid->nj,grid->dx,grid->dz);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"  t_max = %g, cfl = %g, dt = %5.4g,",param->t_max,param->cfl,param->dt);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," t_output = %g\n",param->t_output_interval);CHKERRQ(ierr);
  if (param->output_to_file) {
    PetscPrintf(PETSC_COMM_WORLD,"Output File:       Binary file \"%s\"\n",param->output_filename);
  }
  if (!param->output_to_file)
    PetscPrintf(PETSC_COMM_WORLD,"Output File:       NO OUTPUT!\n");
  ierr = PetscPrintf(PETSC_COMM_WORLD,"----------------------------------------\n");CHKERRQ(ierr);
  if (param->param_test) PetscEnd();
  return ierr_out;
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "Initialize"
int Initialize(DMMG *dmmg)
/* ------------------------------------------------------------------- */
{
  AppCtx    *user  = (AppCtx*)dmmg[0]->user;
  Parameter *param;
  DA        da;
  PetscReal PI = 3.14159265358979323846;
  PetscReal sigma,xc,zc;
  PetscReal dx=user->grid->dx,dz=user->grid->dz;
  int       i,j,ierr,is,js,im,jm;
  Field     **x;
  ierr = PetscBagGetData(user->bag,(void**)&param);CHKERRQ(ierr);
  sigma=param->sigma; xc=param->xctr; zc=param->zctr;

  /* Get the DA and grid */
  da = (DA)(dmmg[0]->dm); 
  ierr = DAGetCorners(da,&is,&js,PETSC_NULL,&im,&jm,PETSC_NULL);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,user->Xold,(void**)&x);CHKERRQ(ierr);

  for (j=js; j<js+jm; j++) {
    for (i=is; i<is+im; i++) {
      if (param->flow_type == SHEAR_CELL) {
        x[j][i].u = -sin(PI*i*dx)*cos(PI*j*dz)/dx;
        x[j][i].w =  sin(PI*j*dz)*cos(PI*i*dx)/dz;
      } else {
	x[j][i].u =  0.0;
	x[j][i].w = -1.0/dz; 
      }
      x[j][i].phi = 100*exp(-0.5*((i*dx-xc)*(i*dx-xc)+(j*dz-zc)*(j*dz-zc))/sigma/sigma);
    }
  }
  
  /* restore the grid to it's vector */
  ierr = DAVecRestoreArray(da,user->Xold,(void**)&x);CHKERRQ(ierr);
  ierr = VecCopy(user->Xold, DMMGGetx(dmmg));CHKERRQ(ierr);
  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "DoSolve"
int DoSolve(DMMG *dmmg)
/* ------------------------------------------------------------------- */
{
  AppCtx         *user  = (AppCtx*)dmmg[0]->user;
  Parameter      *param;
  PetscReal      t_output = 0.0;
  int            ierr, n_plot = 0, Ncomponents, components[3];
  DA             da = DMMGGetDA(dmmg);
  Vec            Xstar;
  Characteristic c;
  ierr = PetscBagGetData(user->bag,(void**)&param);CHKERRQ(ierr);

  ierr = DAGetGlobalVector(da, &Xstar);CHKERRQ(ierr);

  /*------------ BEGIN CHARACTERISTIC SETUP ---------------*/
  ierr = CharacteristicCreate(PETSC_COMM_WORLD, &c);CHKERRQ(ierr);
  /* set up the velocity interpolation system */
  Ncomponents = 2; components[0] = 0; components[1] = 1;
  ierr = CharacteristicSetVelocityInterpolationLocal(c, da, DMMGGetx(dmmg), user->Xold, Ncomponents, components, InterpVelocity2D, user);CHKERRQ(ierr);
  /* set up the fields interpolation system */
  Ncomponents = 1; components[0] = 2;
  ierr = CharacteristicSetFieldInterpolationLocal(c, da, user->Xold, Ncomponents, components, InterpFields2D, user);CHKERRQ(ierr);
  /*------------ END CHARACTERISTIC SETUP ----------------*/

  /* output initial data */
  PetscPrintf(PETSC_COMM_WORLD," Initialization, Time: %5.4g\n", param->t);
  if (param->verify) { ierr = DoVerification(dmmg,user);CHKERRQ(ierr); }
  ierr = DoOutput(dmmg,n_plot);CHKERRQ(ierr); 
  t_output += param->t_output_interval; n_plot++;

  /* timestep loop */
  for (param->t=param->dt; param->t<=param->t_max; param->t+=param->dt) {
    if (param->n > param->N_steps) {
      PetscPrintf(PETSC_COMM_WORLD,"EXCEEDED MAX NUMBER OF TIMESTEPS! EXITING SOLVE!\n");
      return 0;
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Solve at time t & copy solution into solution vector.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    /* Copy in the velocities to Xstar */
    ierr = VecCopy(DMMGGetx(dmmg), Xstar);CHKERRQ(ierr);
    /* Put \phi_* into Xstar */
    ierr = CharacteristicSolve(c, param->dt, Xstar);CHKERRQ(ierr);
    /* Copy the advected field into the solution \phi_t = \phi_* */
    ierr = VecCopy(Xstar, DMMGGetx(dmmg));CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Copy new solution to old solution in prep for the next timestep.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    ierr = VecCopy(DMMGGetx(dmmg), user->Xold);CHKERRQ(ierr);


    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Timestep complete, report and update counter.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscPrintf(PETSC_COMM_WORLD," Step: %d, Time: %5.4g\n", param->n, param->t);
    param->n++;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Verify and make output.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    if (param->verify) { ierr = DoVerification(dmmg,user);CHKERRQ(ierr); }
    if (param->t >= t_output) {
      ierr = DoOutput(dmmg,n_plot);CHKERRQ(ierr); 
      t_output += param->t_output_interval; n_plot++;
    }
  }
  ierr = DARestoreGlobalVector(da, &Xstar);CHKERRQ(ierr);
  ierr = CharacteristicDestroy(c);CHKERRQ(ierr);
  return 0; 
}

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "InterpVelocity2D"
/* uses analytic velocity fields */
PetscErrorCode InterpVelocity2D(void *f, PetscReal ij_real[], PetscInt numComp, 
				PetscInt components[], PetscReal velocity[], 
				void *ctx)
/*---------------------------------------------------------------------*/
{
  AppCtx    *user = (AppCtx *) ctx;
  Parameter *param;
  PetscReal dx=user->grid->dx, dz=user->grid->dz;
  PetscReal PI = 3.14159265358979323846;
  int       ierr;
  ierr = PetscBagGetData(user->bag,(void**)&param);CHKERRQ(ierr);

  if (param->flow_type == SHEAR_CELL) {
    velocity[0] = -sin(PI*ij_real[0]*dx)*cos(PI*ij_real[1]*dz)/dx;
    velocity[1] =  sin(PI*ij_real[1]*dz)*cos(PI*ij_real[0]*dx)/dz;
  } else {
    velocity[0] = 0.;
    velocity[1] = -1./dz;
  }
  return 0;
}

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "InterpFields2D"
/* uses bicubic interpolation */
PetscErrorCode InterpFields2D(void *f, PetscReal ij_real[], PetscInt numComp, 
			      PetscInt components[], PetscReal field[], 
			      void *ctx)
/*---------------------------------------------------------------------*/
{
  AppCtx    *user = (AppCtx*)ctx;
  Field     **x   = (Field**)f;
  int       ni=user->grid->ni, nj=user->grid->nj;
  PetscReal ir=ij_real[0], jr=ij_real[1];

  /* boundary condition: set to zero if characteristic begins outside the domain */
  if ( ir < 0 || ir > ni-1 || jr < 0 || jr> nj-1 ) { 
    field[0] = 0.0;
  }  else {
    field[0] = BiCubicInterp(x, ir, jr);
  }
  return 0;
}

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "BiCubicInterp"
PetscReal BiCubicInterp(Field **x, PetscReal ir, PetscReal jr)
/*---------------------------------------------------------------------*/
{
  int        im, jm, imm,jmm,ip,jp,ipp,jpp;
  PetscReal  il, jl, row1, row2, row3, row4;
  im = (int)floor(ir); jm = (int)floor(jr);
  il = ir - im + 1.0; jl = jr - jm + 1.0;
  imm = im-1; ip = im+1; ipp = im+2;
  jmm = jm-1; jp = jm+1; jpp = jm+2;
  row1 = CubicInterp(il,x[jmm][imm].phi,x[jmm][im].phi,x[jmm][ip].phi,x[jmm][ipp].phi);
  row2 = CubicInterp(il,x[jm] [imm].phi,x[jm] [im].phi,x[jm] [ip].phi,x[jm] [ipp].phi);
  row3 = CubicInterp(il,x[jp] [imm].phi,x[jp] [im].phi,x[jp] [ip].phi,x[jp] [ipp].phi);
  row4 = CubicInterp(il,x[jpp][imm].phi,x[jpp][im].phi,x[jpp][ip].phi,x[jpp][ipp].phi);
  return CubicInterp(jl,row1,row2,row3,row4);
}

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "CubicInterp"
PetscReal CubicInterp(PetscReal x, PetscReal y_1, PetscReal y_2, 
		      PetscReal y_3, PetscReal y_4)
/*---------------------------------------------------------------------*/
{
  PetscReal  sxth=0.16666666666667, retval;
  retval = - y_1*(x-1.0)*(x-2.0)*(x-3.0)*sxth + y_2*(x-0.0)*(x-2.0)*(x-3.0)*0.5
           - y_3*(x-0.0)*(x-1.0)*(x-3.0)*0.5  + y_4*(x-0.0)*(x-1.0)*(x-2.0)*sxth;
  return retval;
}

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "DoVerification"
int DoVerification(DMMG *dmmg, AppCtx *user)
/*---------------------------------------------------------------------*/
{
  Parameter *param;
  PetscReal t1,t2,t3,norms[3];
  int       ierr;
  ierr = PetscBagGetData(user->bag,(void**)&param);CHKERRQ(ierr);
  
  ierr = CalcSolnNorms(dmmg, norms);CHKERRQ(ierr);
  t1 = (norms[0]-param->L1)/param->L1*100.0;
  t2 = (norms[1]-param->L2)/param->L2*100.0;
  t3 = (norms[2]-param->LINF)/param->LINF*100.0;
  if ((fabs(t1)>1.0) || (fabs(t2)>1.0) || (fabs(t3)>5.0)) {
    param->verify_result = 1;
  }
  PetscPrintf(PETSC_COMM_WORLD," Step: %d, Soln norms: %g (L1) %g (L2) %g (LINF)\n",param->n,norms[0],norms[1],norms[2]);
  PetscPrintf(PETSC_COMM_WORLD," Step: %d, Soln norms %%err: %5.2g (L1) %5.2g (L2) %5.2g (LINF)\n",param->n,t1,t2,t3);
  return 0;
}

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "CalcSolnNorms"
int CalcSolnNorms(DMMG *dmmg, PetscReal norms[])
/*---------------------------------------------------------------------*/
{
  Vec           x;
  int           ierr;                                                  
  x = DMMGGetx(dmmg);
  ierr = VecNorm(x, NORM_1, &(norms[0]));
  ierr = VecNorm(x, NORM_2, &(norms[1]));
  ierr = VecNorm(x, NORM_INFINITY, &(norms[2]));
  return 0;
}

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "DoOutput"
int DoOutput(DMMG *dmmg, int n_plot)
/*---------------------------------------------------------------------*/
{
  AppCtx      *user = (AppCtx*)dmmg[0]->user;
  Parameter   *param;
  int         ierr;
  char        filename[FNAME_LENGTH];
  PetscViewer viewer;
  DA          da;
  ierr = PetscBagGetData(user->bag,(void**)&param);CHKERRQ(ierr);
  da = DMMGGetDA(dmmg);

  if (param->output_to_file) { /* send output to binary file */
    /* generate filename for time t */
    sprintf(filename,"%s_%3.3d",param->output_filename,n_plot);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Generating output: time t = %g, ",param->t);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"file = \"%s\"\n",filename);

    /* make output files */
    ierr = PetscViewerBinaryMatlabOpen(PETSC_COMM_WORLD,filename,&viewer);CHKERRQ(ierr);
    ierr = PetscViewerBinaryMatlabOutputBag(viewer,"par",user->bag);CHKERRQ(ierr);
    ierr = DASetFieldNames("u","v","phi",da);CHKERRQ(ierr);
    ierr = PetscViewerBinaryMatlabOutputVecDA(viewer,"field",DMMGGetx(dmmg),da);CHKERRQ(ierr);
    ierr = PetscViewerBinaryMatlabDestroy(viewer);CHKERRQ(ierr);
  }  
  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "DASetFieldNames"
int DASetFieldNames(const char n0[], const char n1[], const char n2[], 
		    DA da)
/* ------------------------------------------------------------------- */
{
  int ierr;
  ierr = DASetFieldName(da,0,n0);CHKERRQ(ierr);
  ierr = DASetFieldName(da,1,n1);CHKERRQ(ierr);
  ierr = DASetFieldName(da,2,n2);CHKERRQ(ierr);
  return 0;
}

/* ------------------------------------------------------------------- */
#undef __FUNCT__
#define __FUNCT__ "OptionsHasName"
PetscTruth OptionsHasName(const char name[])
/* ------------------------------------------------------------------- */
{
  PetscTruth retval; 
  int ierr;
  ierr = PetscOptionsHasName(PETSC_NULL,name,&retval);
  return retval;
}

