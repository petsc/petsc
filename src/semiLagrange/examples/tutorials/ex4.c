static char help[] = "\n";
/* - - - - - - - - - - - - - - - - - - - - - - - - 
   ex4.c
   Advection-diffusion equation

   Du
   -- - \nabla^2 u = 0
   dt 

which we discretize using Crank-Nicholson

   u^+ - u^*  1 /                                 \
   -------- - - | (\nabla^2 u)^+ + (\nabla^2 u)^* | = 0
   \Delta t   2 \                                 /

which gives, collecting terms

         \Delta t                        \Delta t
   u^+ - -------- (\nabla^2 u)^+ = u^* + -------- (\nabla^2 u)^*
             2                              2

where u^- is the solution at the old time, u^+ is
the solution at the new time, and u^* is the solution
at the old time advected to the new time.
- - - - - - - - - - - - - - - - - - - - - - - - */
#include "petscsnes.h"
#include "petscda.h"
#include "petscdmmg.h"
#include "petscbag.h"
#include "characteristic.h"

#define EXAMPLE_NUMBER 1
#define SHEAR_CELL     0
#define SOLID_BODY     1
#define FNAME_LENGTH   60

typedef struct field_s {
  PetscReal phi;
} Field;

typedef struct parameter_s {
  int            ni, nj,pi,pj;          /* number of grid points, number of procs */ 
  PassiveScalar  amp,sigma,xctr,zctr,L1,L2,LINF; /* parameters for gaussian Initial Condition */
  PassiveScalar  Pe, theta, ct, st, diffScale;   /* parameters for velocity field and diffusion length */
  int            flow_type, sl_event;
  PetscTruth     param_test, output_to_file;
  char           output_filename[FNAME_LENGTH];
  /* timestep stuff */
  PassiveScalar  t; /* the time */
  int            n; /* the time step */
  PassiveScalar  dtAdvection, dtDiffusion; /* minimal advection and diffusion time steps */
  PassiveScalar  t_max, dt, cfl, t_output_interval;
  int            N_steps;
} Parameter;

typedef struct gridinfo_s {
  DAPeriodicType periodic;
  DAStencilType  stencil;
  int            ni,nj,dof,stencil_width,mglevels;
  PassiveScalar  dx,dz;
} GridInfo;

typedef struct appctx_s {
  DMMG        *dmmg;
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
PetscReal BiCubicInterp  (Field**, PetscReal, PetscReal);
PetscReal CubicInterp    (PetscReal, PetscReal, PetscReal, PetscReal, PetscReal);
PetscTruth OptionsHasName(const char*);

/* characteristic call-backs (static interface) */
PetscErrorCode InterpVelocity2D(void*, PetscReal[], PetscInt, PetscInt[], PetscReal[], void*);
PetscErrorCode InterpFields2D  (void*, PetscReal[], PetscInt, PetscInt[], PetscReal[], void*);

PetscErrorCode FormOldTimeFunctionLocal(DALocalInfo *, PetscScalar **, PetscScalar **, AppCtx *);
PetscErrorCode FormNewTimeFunctionLocal(DALocalInfo *, PetscScalar **, PetscScalar **, AppCtx *);

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
  AppCtx         *user;               /* user-defined work context */
  Parameter      *param;
  DA              da;
  GridInfo        grid;
  MPI_Comm        comm;
  PetscErrorCode  ierr;

  PetscInitialize(&argc,&argv,(char *)0,help);
  comm = PETSC_COMM_WORLD;

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set up the problem parameters.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */ 
  ierr = PetscMalloc(sizeof(AppCtx),&user);CHKERRQ(ierr);
  ierr = PetscBagCreate(comm,sizeof(Parameter),&(user->bag));CHKERRQ(ierr);
  user->grid = &grid;
  ierr = SetParams(user);CHKERRQ(ierr);
  ierr = ReportParams(user);CHKERRQ(ierr);
  ierr = PetscBagGetData(user->bag,(void**)&param);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create distributed array multigrid object (DMMG) to manage parallel grid and vectors
     for principal unknowns (x) and governing residuals (f)
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */ 
  ierr = DMMGCreate(comm,grid.mglevels,user,&user->dmmg);CHKERRQ(ierr); 
  ierr = DACreate2d(comm,grid.periodic,grid.stencil,grid.ni,grid.nj,PETSC_DECIDE,PETSC_DECIDE,grid.dof,grid.stencil_width,0,0,&da);CHKERRQ(ierr);
  ierr = DMMGSetDM(user->dmmg,(DM) da);CHKERRQ(ierr);
  ierr = DADestroy(da);CHKERRQ(ierr);
  ierr = DMMGSetSNESLocal(user->dmmg,FormNewTimeFunctionLocal,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMMGSetFromOptions(user->dmmg);CHKERRQ(ierr);
  ierr = DAGetInfo(DMMGGetDA(user->dmmg),PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,&(param->pi),&(param->pj),PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  REG_INTG(user->bag,&param->pi,param->pi ,"procs_x","<DO NOT SET> Processors in the x-direction");
  REG_INTG(user->bag,&param->pj,param->pj ,"procs_y","<DO NOT SET> Processors in the y-direction");

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create user context, set problem data, create vector data structures.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */   
  ierr = DAGetGlobalVector(DMMGGetDA(user->dmmg), &(user->Xold));CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Initialize and solve the nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = Initialize(user->dmmg);CHKERRQ(ierr);
  ierr = DoSolve(user->dmmg);CHKERRQ(ierr);
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space. 
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DARestoreGlobalVector(DMMGGetDA(user->dmmg), &(user->Xold));CHKERRQ(ierr);
  ierr = PetscBagDestroy(user->bag);CHKERRQ(ierr); 
  ierr = DMMGDestroy(user->dmmg);CHKERRQ(ierr);
  ierr = PetscFree(user);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
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
  PetscReal  PI = 3.14159265358979323846;

  ierr = PetscBagGetData(bag,(void**)&p);CHKERRQ(ierr);

  /* domain geometry & grid size */
  REG_INTG(bag,&p->ni,40                  ,"ni","Grid points in x-dir");
  REG_INTG(bag,&p->nj,40                  ,"nj","Grid points in y-dir");
  grid->ni = p->ni;
  grid->nj = p->nj;
  grid->dx = 1.0/((double)(grid->ni - 1));
  grid->dz = 1.0/((double)(grid->nj - 1));

  /* initial conditions */
  REG_INTG(bag,&p->flow_type,SOLID_BODY   ,"flow_type","Flow field mode: 0=shear cell, 1=translation");
  REG_REAL(bag,&p->amp,1                  ,"amp","Amplitude of the gaussian IC");
  REG_REAL(bag,&p->sigma,0.07             ,"sigma","Standard deviation of the gaussian IC");
  REG_REAL(bag,&p->xctr,0.5               ,"xctr","x-position of the center of the gaussian IC");
  REG_REAL(bag,&p->zctr,0.5               ,"zctr","z-position of the center of the gaussian IC");

  /* Velocity field */
  REG_REAL(bag,&p->Pe,10                   ,"Pe","Peclet number");
  REG_REAL(bag,&p->theta,-90                ,"theta","velocity angle (degrees)");
  REG_REAL(bag,&p->ct,cos(p->theta*PI/180),"cosTheta","<DO NOT SET> cosine velocity angle");
  REG_REAL(bag,&p->st,sin(p->theta*PI/180),"sinTheta","<DO NOT SET> sine velocity angle");
  
  /* diffusion LengthScale for time stepping */
  REG_REAL(bag,&p->diffScale,2,            "diffScale","diffusion length scale (number of grid points for stable diffusion)");

  /* time stepping */  
  REG_REAL(bag,&p->t_max,1                ,"t_max","Maximum dimensionless time");
  REG_REAL(bag,&p->cfl,5                  ,"cfl","Courant number");
  REG_REAL(bag,&p->t_output_interval,0.1  ,"t_output","Dimensionless time interval for output");
  REG_INTG(bag,&p->N_steps,1000           ,"nsteps","Maximum time-steps");
  REG_INTG(bag,&p->n,1                    ,"nsteps","<DO NOT SET> current time-step");
  REG_REAL(bag,&p->t,0.0                  ,"time","<DO NOT SET> initial time");
  REG_REAL(bag,&p->dtAdvection,1./p->Pe/(sqrt(pow(p->ct/grid->dx,2)+pow(p->st/grid->dz,2))),"dtAdvection","<DO NOT SET> CFL limited advection time step");
  REG_REAL(bag,&p->dtDiffusion,1./(1./grid->dx/grid->dx+1./grid->dz/grid->dz),"dtDiffusion","<DO NOT SET> grid-space limited diffusion time step");
  REG_REAL(bag,&p->dt,PetscMin(p->cfl*p->dtAdvection,pow(p->diffScale,2)*p->dtDiffusion),"dt","<DO NOT SET> time-step size");

  /* output options */
  REG_TRUE(bag,&p->param_test     ,PETSC_FALSE  ,"test","Run parameter test only (T/F)");
  REG_STRG(bag,&p->output_filename,FNAME_LENGTH ,"null","output_file","Name base for output files, set with: -output_file <filename>");
  REG_TRUE(bag,&p->output_to_file,PETSC_FALSE   ,"do_output","<DO NOT SET> flag will be true if you specify an output file name");
  p->output_to_file = OptionsHasName("-output_file");

  grid->ni            = p->ni;
  grid->nj            = p->nj;
  grid->periodic      = DA_XYPERIODIC;
  grid->stencil       = DA_STENCIL_BOX;
  grid->dof           = 1;
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
    PetscPrintf(PETSC_COMM_WORLD,"Flow_type: %d (translation).\n\n",param->flow_type); 
  }
  ierr = PetscPrintf(PETSC_COMM_WORLD,"  [ni,nj] = %d, %d   [dx,dz] = %5.4g, %5.4g\n",grid->ni,grid->nj,grid->dx,grid->dz);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"  t_max = %g, cfl = %g, dt = %5.4g,",param->t_max,param->cfl,param->dt);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," t_output = %g\n",param->t_output_interval);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," dt_advection= %g, dt_diffusion= %g\n",param->dtAdvection,param->dtDiffusion);CHKERRQ(ierr);
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
  PetscReal amp, sigma, xc, zc ;
  PetscReal dx=user->grid->dx,dz=user->grid->dz;
  int       i,j,ierr,is,js,im,jm;
  Field     **x;
  ierr = PetscBagGetData(user->bag,(void**)&param);CHKERRQ(ierr);
  
  amp = param->amp;
  sigma = param->sigma;
  xc = param->xctr; zc = param->zctr;

  /* Get the DA and grid */
  da = DMMGGetDA(dmmg); 
  ierr = DAGetCorners(da,&is,&js,PETSC_NULL,&im,&jm,PETSC_NULL);CHKERRQ(ierr);
  ierr = DAVecGetArray(da,user->Xold,(void**)&x);CHKERRQ(ierr);

  for (j=js; j<js+jm; j++) {
    for (i=is; i<is+im; i++) {
      x[j][i].phi = param->amp*exp(-0.5*((i*dx-xc)*(i*dx-xc)+(j*dz-zc)*(j*dz-zc))/sigma/sigma);
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
  Ncomponents = 2; components[0] = 0; components[1] = 0;
  ierr = CharacteristicSetVelocityInterpolationLocal(c, da, DMMGGetx(dmmg), user->Xold, Ncomponents, components, InterpVelocity2D, user);CHKERRQ(ierr);
  /* set up the fields interpolation system */
  Ncomponents = 1; components[0] = 0;
  ierr = CharacteristicSetFieldInterpolationLocal(c, da, user->Xold, Ncomponents, components, InterpFields2D, user);CHKERRQ(ierr);
  /*------------ END CHARACTERISTIC SETUP ----------------*/

  /* output initial data */
  PetscPrintf(PETSC_COMM_WORLD," Initialization, Time: %5.4g\n", param->t);
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
    /* Evaluate operator (I + \Delta t/2 L) u^-  = X^- */
    ierr = DAFormFunctionLocal(da, (DALocalFunction1) FormOldTimeFunctionLocal, DMMGGetx(dmmg), user->Xold, user);CHKERRQ(ierr);
    /* Advect Xold into Xstar */
    ierr = CharacteristicSolve(c, param->dt, Xstar);CHKERRQ(ierr);
    /* Xstar -> Xold */
    ierr = VecCopy(Xstar, user->Xold);CHKERRQ(ierr);
    /* Solve u^+ = (I - \Delta t/2 L)^-1 Xstar which could be F(u^+) = Xstar */
    ierr = DMMGSolve(dmmg);CHKERRQ(ierr);

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       report step and update counter.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
    PetscPrintf(PETSC_COMM_WORLD," Step: %d, Time: %5.4g\n", param->n, param->t);
    param->n++;

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Output variables.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
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
/* linear interpolation, ir: [0, ni], jr: [0, nj] */
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

  /* remember: must be coordinate velocities not true velocities */
  if (param->flow_type == SHEAR_CELL) {
    velocity[0] = -sin(PI*ij_real[0]*dx)*cos(PI*ij_real[1]*dz)/dx;
    velocity[1] =  sin(PI*ij_real[1]*dz)*cos(PI*ij_real[0]*dx)/dz;
  } else {
    velocity[0] = param->Pe*param->ct/dx; 
    velocity[1] = param->Pe*param->st/dz;
  }
  return 0;
}

/*---------------------------------------------------------------------*/
#undef __FUNCT__
#define __FUNCT__ "InterpFields2D"
PetscErrorCode InterpFields2D(void *f, PetscReal ij_real[], PetscInt numComp, 
			      PetscInt components[], PetscReal field[], 
			      void *ctx)
/*---------------------------------------------------------------------*/
{
  AppCtx    *user = (AppCtx*)ctx;
  Field     **x   = (Field**)f;
  int       ni=user->grid->ni, nj=user->grid->nj;
  int       ierr;
  PetscReal ir=ij_real[0], jr=ij_real[1];

  /* map back to periodic domain if out of bounds */
  if ( ir < 0 || ir > ni-1 || jr < 0 || jr> nj-1 ) { 
    ierr = DAMapCoordsToPeriodicDomain(DMMGGetDA(user->dmmg), &ir, &jr);CHKERRQ(ierr);
  } 
  field[0] = BiCubicInterp(x, ir, jr);
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
    ierr = DASetFieldName(da,0,"phi");CHKERRQ(ierr);
    ierr = PetscViewerBinaryMatlabOutputVecDA(viewer,"field",DMMGGetx(dmmg),da);CHKERRQ(ierr);
    ierr = PetscViewerBinaryMatlabDestroy(viewer);CHKERRQ(ierr);
  }  
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

#undef __FUNCT__
#define __FUNCT__ "FormNewTimeFunctionLocal"
/* 
  FormNewTimeFunctionLocal - Evaluates f = (I - \Delta t/2 L) x - f(u^-)^*.

  Note: We get f(u^-)^* from Xold in the user context.

  Process adiC(36): FormNewTimeFunctionLocal
*/
PetscErrorCode FormNewTimeFunctionLocal(DALocalInfo *info, PetscScalar **x, PetscScalar **f, AppCtx *user)
{
  DA             da = DMMGGetDA(user->dmmg);
  PetscScalar  **fold;
  Parameter     *param;
  PetscScalar    u,uxx,uyy;
  PetscReal      hx,hy,hxdhy,hydhx;
  PetscInt       i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  hx     = 1.0/(PetscReal)(info->mx-1);
  hy     = 1.0/(PetscReal)(info->my-1);
  hxdhy  = hx/hy; 
  hydhx  = hy/hx;

  ierr = PetscBagGetData(user->bag, (void**) &param);CHKERRQ(ierr);
  ierr = DAVecGetArray(da, user->Xold, &fold);CHKERRQ(ierr);
  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {
      u       = x[j][i];
      uxx     = (2.0*u - x[j][i-1] - x[j][i+1])*hydhx;
      uyy     = (2.0*u - x[j-1][i] - x[j+1][i])*hxdhy;
      f[j][i] = u*hx*hy + param->dt*0.5*(uxx + uyy) - fold[j][i];
    }
  }
  ierr = DAVecRestoreArray(da, user->Xold, &fold);CHKERRQ(ierr);

  ierr = PetscLogFlops(13.0*info->ym*info->xm);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
} 

#undef __FUNCT__
#define __FUNCT__ "FormOldTimeFunctionLocal"
/* 
  FormOldTimeFunctionLocal - Evaluates f = (I + \Delta t/2 L) x.

  Process adiC(36): FormOldTimeFunctionLocal
*/
PetscErrorCode FormOldTimeFunctionLocal(DALocalInfo *info, PetscScalar **x, PetscScalar **f, AppCtx *user)
{
  Parameter     *param;
  PetscScalar    u,uxx,uyy;
  PetscReal      hx,hy,hxdhy,hydhx;
  PetscInt       i,j;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  hx     = 1.0/(PetscReal)(info->mx-1);
  hy     = 1.0/(PetscReal)(info->my-1);
  hxdhy  = hx/hy; 
  hydhx  = hy/hx;

  ierr = PetscBagGetData(user->bag, (void**) &param);CHKERRQ(ierr);
  for (j=info->ys; j<info->ys+info->ym; j++) {
    for (i=info->xs; i<info->xs+info->xm; i++) {
      u       = x[j][i];
      uxx     = (2.0*u - x[j][i-1] - x[j][i+1])*hydhx;
      uyy     = (2.0*u - x[j-1][i] - x[j+1][i])*hxdhy;
      f[j][i] = u*hx*hy - param->dt*0.5*(uxx + uyy);
    }
  }

  ierr = PetscLogFlops(12.0*info->ym*info->xm);CHKERRQ(ierr);
  PetscFunctionReturn(0); 
} 
