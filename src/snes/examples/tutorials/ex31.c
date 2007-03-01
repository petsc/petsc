
static char help[] = "Model multi-physics solver\n\n";

/*
     A model "multi-physics" solver based on the Vincent Mousseau's reactor core pilot code.

     There are three grids:

            --------------------- DA1        

                    nyv  -->       --------------------- DA2
                                   |                    | 
                                   |                    | 
                                   |                    |   
                                   |                    | 
                    nyvf-1 -->     |                    |         --------------------- DA3
                                   |                    |        |                    |
                                   |                    |        |                    |
                                   |                    |        |                    |
                                   |                    |        |                    |
                         0 -->     ---------------------          ---------------------

                   nxv                     nxv                          nxv

            Fluid                     Thermal conduction              Fission (core)
                                    (cladding and core)

    Notes:
     * The discretization approach used is to have ghost nodes OUTSIDE the physical domain
      that are used to apply the stencil near the boundary; in order to implement this with
      PETSc DAs we simply define the DAs to have periodic boundary conditions and use those
      periodic ghost points to store the needed extra variables (which do not equations associated
      with them). Note that these periodic ghost nodes have NOTHING to do with the ghost nodes
      used for parallel computing.

*/

#include "petscdmmg.h"

typedef struct {                  
  PetscScalar pri,ugi,ufi,agi,vgi,vfi;              /* initial conditions for fluid variables */
  PetscScalar prin,ugin,ufin,agin,vgin,vfin;        /* inflow boundary conditions for fluid */
  PetscScalar prout,ugout,ufout,agout,vgout;        /* outflow boundary conditions for fluid */

  PetscScalar twi;                                  /* initial condition for tempature */

  PetscScalar phii;                                 /* initial conditions for fuel */
  PetscScalar prei;

  PetscInt    nxv,nyv,nyvf;

  PetscViewer v1,v2,v3;

  DMComposite pack;
} AppCtx;

typedef struct {                 /* Fluid unknowns */
  PetscScalar prss;
  PetscScalar ergg;
  PetscScalar ergf;
  PetscScalar alfg;
  PetscScalar velg;
  PetscScalar velf;
} FluidField;

typedef struct {                 /* Fuel unknowns */
  PetscScalar phii;
  PetscScalar prei;
} FuelField;

extern PetscErrorCode FormInitialGuess(DMMG,Vec);
extern PetscErrorCode FormFunction(SNES,Vec,Vec,void*);
extern PetscErrorCode MyVecView(AppCtx*,Vec);
extern PetscErrorCode MyPCApply(void*,Vec,Vec);

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  DMMG           *dmmg;               /* multilevel grid structure */
  PetscErrorCode ierr;
  MPI_Comm       comm;
  DA             da;
  AppCtx         app;
  PC             pc;
  KSP            ksp;
  PetscTruth     isshell;

  PetscInitialize(&argc,&argv,(char *)0,help);
  comm = PETSC_COMM_WORLD;

  PreLoadBegin(PETSC_TRUE,"SetUp");

    app.nxv  = 6;
    app.nyvf = 3;
    app.nyv  = app.nyvf + 2;

    ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,PETSC_NULL,"Fluid",-1,-1,-1,-1,&app.v1);CHKERRQ(ierr);
    ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,PETSC_NULL,"Thermal",-1,-1,-1,-1,&app.v2);CHKERRQ(ierr);
    ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,PETSC_NULL,"Fuel",-1,-1,-1,-1,&app.v3);CHKERRQ(ierr);

    /*
       Create the DMComposite object to manage the three grids/physics. 
       We use a 1d decomposition along the y direction (since one of the grids is 1d).

    */
    ierr = DMCompositeCreate(comm,&app.pack);CHKERRQ(ierr);

    /* 6 fluid unknowns, 3 ghost points on each end for either periodicity or simply boundary conditions */
    ierr = DACreate1d(comm,DA_XPERIODIC,app.nxv,6,3,0,&da);CHKERRQ(ierr);
    ierr = DASetFieldName(da,0,"prss");CHKERRQ(ierr);
    ierr = DASetFieldName(da,1,"ergg");CHKERRQ(ierr);
    ierr = DASetFieldName(da,2,"ergf");CHKERRQ(ierr);
    ierr = DASetFieldName(da,3,"alfg");CHKERRQ(ierr);
    ierr = DASetFieldName(da,4,"velg");CHKERRQ(ierr);
    ierr = DASetFieldName(da,5,"velf");CHKERRQ(ierr);
    ierr = DMCompositeAddDA(app.pack,da);CHKERRQ(ierr);
    ierr = DADestroy(da);CHKERRQ(ierr);

    ierr = DACreate2d(comm,DA_YPERIODIC,DA_STENCIL_STAR,app.nxv,app.nyv,PETSC_DETERMINE,1,1,1,0,0,&da);CHKERRQ(ierr);
    ierr = DASetFieldName(da,0,"Tempature");CHKERRQ(ierr);
    ierr = DMCompositeAddDA(app.pack,da);CHKERRQ(ierr);
    ierr = DADestroy(da);CHKERRQ(ierr);

    ierr = DACreate2d(comm,DA_XYPERIODIC,DA_STENCIL_STAR,app.nxv,app.nyvf,PETSC_DETERMINE,1,2,1,0,0,&da);CHKERRQ(ierr);
    ierr = DASetFieldName(da,0,"Phi");CHKERRQ(ierr);
    ierr = DASetFieldName(da,1,"Pre");CHKERRQ(ierr);
    ierr = DMCompositeAddDA(app.pack,da);CHKERRQ(ierr);
    ierr = DADestroy(da);CHKERRQ(ierr);
   
    app.pri = 1.0135e+5;
    app.ugi = 2.5065e+6;
    app.ufi = 4.1894e+5;
    app.agi = 1.00e-1;
    app.vgi = 1.0e-1 ;
    app.vfi = 1.0e-1;

    app.prin = 1.0135e+5;
    app.ugin = 2.5065e+6;
    app.ufin = 4.1894e+5;
    app.agin = 1.00e-1;
    app.vgin = 1.0e-1 ;
    app.vfin = 1.0e-1;

    app.prout = 1.0135e+5;
    app.ugout = 2.5065e+6;
    app.ufout = 4.1894e+5;
    app.agout = 3.0e-1;

    app.twi = 373.15e+0;

    app.phii = 1.0e+0;
    app.prei = 1.0e-5;

    /*
       Create the solver object and attach the grid/physics info 
    */
    ierr = DMMGCreate(comm,1,0,&dmmg);CHKERRQ(ierr);
    ierr = DMMGSetDM(dmmg,(DM)app.pack);CHKERRQ(ierr);
    ierr = DMMGSetUser(dmmg,0,&app);CHKERRQ(ierr);
    ierr = DMMGSetISColoringType(dmmg,IS_COLORING_GLOBAL);CHKERRQ(ierr);
    CHKMEMQ;


    ierr = DMMGSetInitialGuess(dmmg,FormInitialGuess);CHKERRQ(ierr);
    ierr = DMMGSetSNES(dmmg,FormFunction,0);CHKERRQ(ierr);

    /* Supply custom shell preconditioner if requested */
    ierr = SNESGetKSP(DMMGGetSNES(dmmg),&ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
    ierr = PetscTypeCompare((PetscObject)pc,PCSHELL,&isshell);CHKERRQ(ierr);
    if (isshell) {
      ierr = PCShellSetContext(pc,&app);CHKERRQ(ierr);
      ierr = PCShellSetApply(pc,MyPCApply);CHKERRQ(ierr);
    }

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Solve the nonlinear system
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  PreLoadStage("Solve");
    ierr = DMMGSolve(dmmg);CHKERRQ(ierr); 


    ierr = MyVecView(&app,DMMGGetx(dmmg));CHKERRQ(ierr); 

    /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
       Free work space.  All PETSc objects should be destroyed when they
       are no longer needed.
       - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

    ierr = PetscViewerDestroy(app.v1);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(app.v2);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(app.v3);CHKERRQ(ierr); 
    ierr = DMCompositeDestroy(app.pack);CHKERRQ(ierr);
    ierr = DMMGDestroy(dmmg);CHKERRQ(ierr);
  PreLoadEnd();

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

/* ------------------------------------------------------------------- */


/* 
   FormInitialGuessLocal* Forms the initial SOLUTION for the fluid, cladding and fuel

 */
#undef __FUNCT__
#define __FUNCT__ "FormInitialGuessLocalFluid"
PetscErrorCode FormInitialGuessLocalFluid(AppCtx *app,DALocalInfo *info,FluidField *f)
{
  PetscInt       i;

  PetscFunctionBegin;
  for (i=info->xs; i<info->xs+info->xm; i++) {
    f[i].prss = app->pri;
    f[i].ergg = app->ugi;
    f[i].ergf = app->ufi;
    f[i].alfg = app->agi;
    f[i].velg = app->vgi;
    f[i].velf = app->vfi;
  }

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormInitialGuessLocalThermal"
PetscErrorCode FormInitialGuessLocalThermal(AppCtx *app,DALocalInfo *info2,PetscScalar **T)
{
  PetscInt i,j;

  PetscFunctionBegin;
  for (i=info2->xs; i<info2->xs+info2->xm; i++) {
    for (j=info2->ys;j<info2->ys+info2->ym; j++) {
      T[j][i] = app->twi;
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormInitialGuessLocalFuel"
PetscErrorCode FormInitialGuessLocalFuel(AppCtx *app,DALocalInfo *info2,FuelField **F)
{
  PetscInt i,j;

  PetscFunctionBegin;
  for (i=info2->xs; i<info2->xs+info2->xm; i++) {
    for (j=info2->ys;j<info2->ys+info2->ym; j++) {
      F[j][i].phii = app->phii;
      F[j][i].prei = app->prei;
    }
  }
  PetscFunctionReturn(0);
}

/* 
   FormFunctionLocal* - Forms user provided function

*/
#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocalFluid"
PetscErrorCode FormFunctionLocalFluid(DALocalInfo *info,FluidField *u,FluidField *f)
{
  PetscInt       i;

  PetscFunctionBegin;
  for (i=info->xs; i<info->xs+info->xm; i++) {
    f[i].prss = u[i].prss;
    f[i].ergg = u[i].ergg;
    f[i].ergf = u[i].ergf;
    f[i].alfg = u[i].alfg;
    f[i].velg = u[i].velg;
    f[i].velf = u[i].velf;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocalThermal"
PetscErrorCode FormFunctionLocalThermal(DALocalInfo *info,PetscScalar **T,PetscScalar **f)
{
  PetscInt i,j;

  PetscFunctionBegin;
  for (i=info->xs; i<info->xs+info->xm; i++) {
    for (j=info->ys;j<info->ys+info->ym; j++) {
      f[j][i] = T[j][i];
    }
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunctionLocalFuel"
PetscErrorCode FormFunctionLocalFuel(DALocalInfo *info,FuelField **U,FuelField **F)
{
  PetscInt i,j;

  PetscFunctionBegin;
  for (i=info->xs; i<info->xs+info->xm; i++) {
    for (j=info->ys;j<info->ys+info->ym; j++) {
      F[j][i].phii = U[j][i].phii;
      F[j][i].prei = U[j][i].prei;
    }
  }
  PetscFunctionReturn(0);
}

 
#undef __FUNCT__
#define __FUNCT__ "FormInitialGuess"
/* 
   FormInitialGuess  - Unwraps the global solution vector and passes its local pieces into the user function

 */
PetscErrorCode FormInitialGuess(DMMG dmmg,Vec X)
{
  DMComposite    dm = (DMComposite)dmmg->dm;
  DALocalInfo    info1,info2,info3;
  DA             da1,da2,da3;
  FluidField     *x1;
  PetscScalar    **x2;
  FuelField      **x3;
  Vec            X1,X2,X3;
  PetscErrorCode ierr;
  AppCtx         *app = (AppCtx*)dmmg->user;

  PetscFunctionBegin;
  ierr = DMCompositeGetEntries(dm,&da1,&da2,&da3);CHKERRQ(ierr);
  ierr = DAGetLocalInfo(da1,&info1);CHKERRQ(ierr);
  ierr = DAGetLocalInfo(da2,&info2);CHKERRQ(ierr);
  ierr = DAGetLocalInfo(da3,&info3);CHKERRQ(ierr);

  /* Access the three subvectors in X */
  ierr = DMCompositeGetAccess(dm,X,&X1,&X2,&X3);CHKERRQ(ierr);

  /* Access the arrays inside the subvectors of X */
  ierr = DAVecGetArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DAVecGetArray(da2,X2,(void**)&x2);CHKERRQ(ierr);
  ierr = DAVecGetArray(da3,X3,(void**)&x3);CHKERRQ(ierr);

  /* Evaluate local user provided function */
  ierr = FormInitialGuessLocalFluid(app,&info1,x1);CHKERRQ(ierr);
  ierr = FormInitialGuessLocalThermal(app,&info2,x2);CHKERRQ(ierr);
  ierr = FormInitialGuessLocalFuel(app,&info3,x3);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da2,X2,(void**)&x2);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da3,X3,(void**)&x3);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(dm,X,&X1,&X2,&X3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormFunction"
/* 
   FormFunction  - Unwraps the input vector and passes its local ghosted pieces into the user function

 */
PetscErrorCode FormFunction(SNES snes,Vec X,Vec F,void *ctx)
{
  DMMG           dmmg = (DMMG)ctx;
  DMComposite    dm = (DMComposite)dmmg->dm;
  DALocalInfo    info1,info2,info3;
  DA             da1,da2,da3;
  FluidField     *x1,*f1;
  PetscScalar    **x2,**f2;
  FuelField      **x3,**f3;
  Vec            X1,X2,X3,F1,F2,F3;
  PetscErrorCode ierr;
  PetscInt       i,j;
  AppCtx         *app = (AppCtx*)dmmg->user;

  PetscFunctionBegin;
  ierr = DMCompositeGetEntries(dm,&da1,&da2,&da3);CHKERRQ(ierr);
  ierr = DAGetLocalInfo(da1,&info1);CHKERRQ(ierr);
  ierr = DAGetLocalInfo(da2,&info2);CHKERRQ(ierr);
  ierr = DAGetLocalInfo(da3,&info3);CHKERRQ(ierr);

  /* Get local vectors to hold ghosted parts of X */
  ierr = DMCompositeGetLocalVectors(dm,&X1,&X2,&X3);CHKERRQ(ierr);
  ierr = DMCompositeScatter(dm,X,X1,X2,X3);CHKERRQ(ierr);

  /* Access the arrays inside the subvectors of X */
  ierr = DAVecGetArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DAVecGetArray(da2,X2,(void**)&x2);CHKERRQ(ierr);
  ierr = DAVecGetArray(da3,X3,(void**)&x3);CHKERRQ(ierr);

   /*
    Ghost points for periodicity are used to "force" inflow/outflow fluid boundary conditions 
    Note that there is no periodicity; we define periodicity to "cheat" and have ghost spaces to store "exterior to boundary" values

  */
  /* FLUID */
  if (info1.gxs == -3) {                   /* 3 points at left end of line */
    for (i=-3; i<0; i++) {
      x1[i].prss = app->prin;
      x1[i].ergg = app->ugin;
      x1[i].ergf = app->ufin;
      x1[i].alfg = app->agin;
      x1[i].velg = app->vgin;
      x1[i].velf = app->vfin;
    }
  }
  if (info1.gxs+info1.gxm == info1.mx+3) { /* 3 points at right end of line */
    for (i=info1.mx; i<info1.mx+3; i++) {
      x1[i].prss = app->prout;
      x1[i].ergg = app->ugout;
      x1[i].ergf = app->ufout;
      x1[i].alfg = app->agout;
      x1[i].velg = app->vgi;
      x1[i].velf = app->vfi;
    }
  }

  /* Thermal */
  if (info2.gxs == -1) {                                      /* left side of domain */
    for (j=info2.gys;j<info2.gys+info2.gym; j++) {
      x2[j][-1] = app->twi;
    }
  }
  if (info2.gxs+info2.gxm == info2.mx+1) {                   /* right side of domain */
    for (j=info2.gys;j<info2.gys+info2.gym; j++) {
      x2[j][info2.mx] = app->twi;
    }
  }

  /* FUEL */
  if (info3.gxs == -1) {                                      /* left side of domain */
    for (j=info3.gys;j<info3.gys+info3.gym; j++) {
      x3[j][-1].phii = app->phii;
      x3[j][-1].prei = app->prei;
    }
  }
  if (info3.gxs+info3.gxm == info3.mx+1) {                   /* right side of domain */
    for (j=info3.gys;j<info3.gys+info3.gym; j++) {
      x3[j][info3.mx].phii = app->phii;
      x3[j][info3.mx].prei = app->prei;
    }
  }
  if (info3.gys == -1) {                                      /* bottom of domain */
    for (i=info3.gxs;i<info3.gxs+info3.gxm; i++) {
      x3[-1][i].phii = app->phii;
      x3[-1][i].prei = app->prei;
    }
  }
  if (info3.gys+info3.gym == info3.my+1) {                   /* top of domain */
    for (i=info3.gxs;i<info3.gxs+info3.gxm; i++) {
      x3[info3.my][i].phii = app->phii;
      x3[info3.my][i].prei = app->prei;
    }
  }

  /* Access the three subvectors in F */
  ierr = DMCompositeGetAccess(dm,F,&F1,&F2,&F3);CHKERRQ(ierr);

  /* Access the arrays inside the subvectors of F */
  ierr = DAVecGetArray(da1,F1,(void**)&f1);CHKERRQ(ierr);
  ierr = DAVecGetArray(da2,F2,(void**)&f2);CHKERRQ(ierr);
  ierr = DAVecGetArray(da3,F3,(void**)&f3);CHKERRQ(ierr);

  /* Evaluate local user provided function */
  ierr = FormFunctionLocalFluid(&info1,x1,f1);CHKERRQ(ierr);
  ierr = FormFunctionLocalThermal(&info2,x2,f2);CHKERRQ(ierr);
  ierr = FormFunctionLocalFuel(&info3,x3,f3);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(da1,X1,(void**)&x1);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da2,X2,(void**)&x2);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da3,X3,(void**)&x3);CHKERRQ(ierr);
  ierr = DMCompositeRestoreLocalVectors(dm,&X1,&X2,&X3);CHKERRQ(ierr);

  ierr = DAVecRestoreArray(da1,F1,(void**)&f1);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da2,F2,(void**)&f2);CHKERRQ(ierr);
  ierr = DAVecRestoreArray(da3,F3,(void**)&f3);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(dm,F,&F1,&F2,&F3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* 
   Here is my custom preconditioner

 */
PetscErrorCode MyPCApply(void* ctx,Vec X,Vec Y)
{
  AppCtx         *app = (AppCtx*)ctx;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(X,Y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MyVecView"
PetscErrorCode MyVecView(AppCtx *app,Vec X)
{
  PetscErrorCode ierr;
  DA             DA1,DA2,DA3;
  Vec            X1,X2,X3;

  PetscFunctionBegin;
  ierr = DMCompositeGetEntries(app->pack,&DA1,&DA2,&DA3);CHKERRQ(ierr);
  ierr = DMCompositeGetAccess(app->pack,X,&X1,&X2,&X3);CHKERRQ(ierr);
  ierr = VecView(X1,app->v1);CHKERRQ(ierr);
  ierr = VecView(X2,app->v2);CHKERRQ(ierr);
  ierr = VecView(X3,app->v3);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccess(app->pack,X,&X1,&X2,&X3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
