static const char help[] = "Integrate chemistry using TChem.\n";

#include <petscts.h>
#include <petscdmda.h>

#if defined(PETSC_HAVE_TCHEM)
#if defined(MAX)
#undef MAX
#endif
#if defined(MIN)
#undef MIN
#endif
#  include <TC_params.h>
#  include <TC_interface.h>
#else
#  error TChem is required for this example.  Reconfigure PETSc using --download-tchem.
#endif
/*

    This is an extension of extchem.c to solve the reaction equations independently in each cell of a one dimensional field

    Obtain the three files into this directory

       curl http://combustion.berkeley.edu/gri_mech/version30/files30/grimech30.dat > chem.inp
       curl http://combustion.berkeley.edu/gri_mech/version30/files30/thermo30.dat > therm.dat
       cp $PETSC_DIR/$PETSC_ARCH/externalpackages/tchem/data/periodictable.dat .

    Run with
     ./extchemfield  -ts_arkimex_fully_implicit -ts_max_snes_failures -1 -ts_adapt_monitor -ts_adapt_dt_max 1e-4 -ts_arkimex_type 4 -ts_final_time .005 -draw_pause -2 -lg_indicate_data_points false -ts_lg_monitor_solution_variables H2,O2,H2O,CH4,CO,CO2,C2H2,N2

    The solution for component i = 0 is the temperature.

    The solution, i > 0, is the mass fraction, massf[i], of species i, i.e. mass of species i/ total mass of all species

    The mole fraction molef[i], i > 0, is the number of moles of a species/ total number of moles of all species
        Define M[i] = mass per mole of species i then
        molef[i] = massf[i]/(M[i]*(sum_j massf[j]/M[j]))

    FormMoleFraction(User,massf,molef) converts the mass fraction solution of each species to the mole fraction of each species.

*/
typedef struct _User *User;
struct _User {
  PetscReal pressure;
  int       Nspec;
  int       Nreac;
  PetscReal Tini;
  DM        dm;
  double    *tchemwork;
  double    *Jdense;        /* Dense array workspace where Tchem computes the Jacobian */ 
  PetscInt  *rows;
};

static PetscErrorCode MonitorCell(TS,User,PetscInt);
static PetscErrorCode FormRHSFunction(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FormRHSJacobian(TS,PetscReal,Vec,Mat,Mat,void*);
static PetscErrorCode FormInitialSolution(TS,Vec,void*);

#define TCCHKERRQ(ierr) do {if (ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in TChem library, return code %d",ierr);} while (0)

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  TS                ts;         /* time integrator */
  TSAdapt           adapt;
  Vec               X;          /* solution vector */
  Mat               J;          /* Jacobian matrix */
  PetscInt          steps,maxsteps;
  PetscErrorCode    ierr;
  PetscReal         ftime,dt;
  char              chemfile[PETSC_MAX_PATH_LEN] = "chem.inp",thermofile[PETSC_MAX_PATH_LEN] = "therm.dat";
  struct _User      user;
  TSConvergedReason reason;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Chemistry solver options","");CHKERRQ(ierr);
  ierr = PetscOptionsString("-chem","CHEMKIN input file","",chemfile,chemfile,sizeof(chemfile),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-thermo","NASA thermo input file","",thermofile,thermofile,sizeof(thermofile),NULL);CHKERRQ(ierr);
  user.pressure = 1.01325e5;    /* Pascal */
  ierr = PetscOptionsReal("-pressure","Pressure of reaction [Pa]","",user.pressure,&user.pressure,NULL);CHKERRQ(ierr);
  user.Tini = 1500;
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = TC_initChem(chemfile, thermofile, 0, 1.0);TCCHKERRQ(ierr);
  user.Nspec = TC_getNspec();
  user.Nreac = TC_getNreac();

  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,-2,user.Nspec+1,1,NULL,&user.dm);CHKERRQ(ierr);
  ierr = DMCreateMatrix(user.dm,&J);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(user.dm,&X);CHKERRQ(ierr);

  ierr = PetscMalloc3(user.Nspec+1,&user.tchemwork,PetscSqr(user.Nspec+1),&user.Jdense,user.Nspec+1,&user.rows);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetDM(ts,user.dm);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
  ierr = TSARKIMEXSetFullyImplicit(ts,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSARKIMEXSetType(ts,TSARKIMEX4);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,FormRHSFunction,&user);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,J,J,FormRHSJacobian,&user);CHKERRQ(ierr);

  ftime    = 1.0;
  maxsteps = 10000;
  ierr     = TSSetDuration(ts,maxsteps,ftime);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = FormInitialSolution(ts,X,&user);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  dt   = 1e-10;                 /* Initial time step */
  ierr = TSSetInitialTimeStep(ts,0.0,dt);CHKERRQ(ierr);
  ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
  ierr = TSAdaptSetStepLimits(adapt,1e-12,1e-4);CHKERRQ(ierr); /* Also available with -ts_adapt_dt_min/-ts_adapt_dt_max */
  ierr = TSSetMaxSNESFailures(ts,-1);CHKERRQ(ierr);            /* Retry step an unlimited number of times */


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Pass information to graphical monitoring routine
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = MonitorCell(ts,&user,0);CHKERRQ(ierr);
  ierr = MonitorCell(ts,&user,1);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve ODE
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,X);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts,&reason);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s at time %g after %D steps\n",TSConvergedReasons[reason],(double)ftime,steps);CHKERRQ(ierr);

  {
    Vec                max;
    const char * const *names;
    PetscInt           i;
    const PetscReal    *bmax;

    ierr = TSMonitorEnvelopeGetBounds(ts,&max,NULL);CHKERRQ(ierr);
    if (max) {
      ierr = TSMonitorLGGetVariableNames(ts,&names);CHKERRQ(ierr);
      ierr = VecGetArrayRead(max,&bmax);CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_SELF,"Species - maximum mass fraction\n");CHKERRQ(ierr);
      for (i=1; i<user.Nspec; i++) {
        if (bmax[i] > .01) {ierr = PetscPrintf(PETSC_COMM_SELF,"%s %g\n",names[i],bmax[i]);CHKERRQ(ierr);}
      }
      ierr = VecRestoreArrayRead(max,&bmax);CHKERRQ(ierr);
    }
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  TC_reset();
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFree3(user.tchemwork,user.Jdense,user.rows);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "FormRHSFunction"
static PetscErrorCode FormRHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ptr)
{
  User              user = (User)ptr;
  PetscErrorCode    ierr;
  PetscScalar       **f;
  const PetscScalar **x;
  DM                dm;
  PetscInt          i,xs,xm;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOFRead(dm,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(dm,F,&f);CHKERRQ(ierr);
  ierr = DMDAGetCorners(dm,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  for (i=xs; i<xm; i++) {
    ierr = PetscMemcpy(user->tchemwork,x[i],(user->Nspec+1)*sizeof(x[xs][0]));CHKERRQ(ierr);
    user->tchemwork[0] *= user->Tini; /* Dimensionalize */
    ierr = TC_getSrc(user->tchemwork,user->Nspec+1,f[i]);TCCHKERRQ(ierr);
    f[i][0] /= user->Tini;           /* Non-dimensionalize */
  }

  ierr = DMDAVecRestoreArrayDOFRead(dm,X,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOF(dm,F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormRHSJacobian"
static PetscErrorCode FormRHSJacobian(TS ts,PetscReal t,Vec X,Mat Amat,Mat Pmat,void *ptr)
{
  User              user = (User)ptr;
  PetscErrorCode    ierr;
  const PetscScalar **x;
  PetscInt          M = user->Nspec+1,i,j,xs,xm;;
  DM                dm;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = MatZeroEntries(Pmat);CHKERRQ(ierr);
  ierr = MatSetOption(Pmat,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatSetOption(Pmat,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOFRead(dm,X,&x);CHKERRQ(ierr);
  ierr = DMDAGetCorners(dm,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  for (i=xs; i<xm; i++) {
    ierr = PetscMemcpy(user->tchemwork,x[i],(user->Nspec+1)*sizeof(x[xs][0]));CHKERRQ(ierr);
    user->tchemwork[0] *= user->Tini;  /* Dimensionalize temperature (first row) because that is what Tchem wants */
    ierr = TC_getJacTYN(user->tchemwork,user->Nspec,user->Jdense,1);CHKERRQ(ierr);

    for (j=0; j<M; j++) user->Jdense[j + 0*M] /= user->Tini; /* Non-dimensionalize first column */
    for (j=0; j<M; j++) user->Jdense[0 + j*M] /= user->Tini; /* Non-dimensionalize first row */
    for (j=0; j<M; j++) user->rows[j] = i*M+j;
    ierr = MatSetValues(Pmat,M,user->rows,M,user->rows,user->Jdense,INSERT_VALUES);CHKERRQ(ierr);
  }

  ierr = DMDAVecRestoreArrayDOFRead(dm,X,&x);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Pmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Pmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (Amat != Pmat) {
    ierr = MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormInitialSolution"
PetscErrorCode FormInitialSolution(TS ts,Vec X,void *ctx)
{
  PetscScalar    **x;
  PetscErrorCode ierr;
  struct {const char *name; PetscReal massfrac;} initial[] = {
    {"CH4", 0.0948178320887},
    {"O2", 0.189635664177},
    {"N2", 0.706766236705},
    {"AR", 0.00878026702874}
  };
  PetscInt       i,j,xs,xm;
  DM             dm;

  PetscFunctionBeginUser;
  ierr = VecZeroEntries(X);CHKERRQ(ierr);
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(dm,X,&x);CHKERRQ(ierr);
  ierr = DMDAGetCorners(dm,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  for (i=xs; i<xs+xm; i++) {
    x[i][0] = 1.0 + i;  /* Non-dimensionalized by user->Tini */
    for (j=0; j<sizeof(initial)/sizeof(initial[0]); j++) {
      int ispec = TC_getSpos(initial[j].name, strlen(initial[j].name));
      if (ispec < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Could not find species %s",initial[j].name);
      ierr = PetscPrintf(PETSC_COMM_SELF,"Species %d: %s %g\n",j,initial[j].name,initial[j].massfrac);CHKERRQ(ierr);
      x[i][1+ispec] = initial[j].massfrac;
    }
  }
  ierr = DMDAVecRestoreArrayDOF(dm,X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Routines for displaying the solutions
*/
typedef struct {
  PetscInt cell;
  User     user;
} UserLGCtx;

#undef __FUNCT__
#define __FUNCT__ "FormMoleFraction"
static PetscErrorCode FormMoleFraction(UserLGCtx *ctx,Vec massf,Vec *molef)
{
  User              user = ctx->user;
  PetscErrorCode    ierr;
  PetscReal         *M,tM=0;
  PetscInt          i,n = user->Nspec+1;
  PetscScalar       *mof;
  const PetscScalar **maf;

  PetscFunctionBegin;
  ierr = VecCreateSeq(PETSC_COMM_SELF,n,molef);CHKERRQ(ierr);
  ierr = PetscMalloc1(user->Nspec,&M);CHKERRQ(ierr);
  TC_getSmass(user->Nspec, M);
  ierr = DMDAVecGetArrayDOFRead(user->dm,massf,&maf);CHKERRQ(ierr);
  ierr = VecGetArray(*molef,&mof);CHKERRQ(ierr);
  mof[0] = maf[ctx->cell][0]; /* copy over temperature */
  for (i=1; i<n; i++) tM += maf[ctx->cell][i]/M[i-1];
  for (i=1; i<n; i++) {
    mof[i] = maf[ctx->cell][i]/(M[i-1]*tM);
  }
  ierr = DMDAVecRestoreArrayDOFRead(user->dm,massf,&maf);CHKERRQ(ierr);
  ierr = VecRestoreArray(*molef,&mof);CHKERRQ(ierr);
  ierr = PetscFree(M);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MonitorCellDestroy"
static PetscErrorCode MonitorCellDestroy(UserLGCtx *uctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(uctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MonitorCell"
/*
   Use TSMonitorLG to monitor the reactions in a particular cell
*/
static PetscErrorCode MonitorCell(TS ts,User user,PetscInt cell)
{
  PetscErrorCode ierr;
  TSMonitorLGCtx ctx;
  char           **snames,*names;
  PetscInt       i;
  UserLGCtx      *uctx;

  PetscFunctionBegin;
  ierr = TSMonitorLGCtxCreate(PETSC_COMM_SELF,0,0,PETSC_DECIDE,PETSC_DECIDE,600,400,1,&ctx);CHKERRQ(ierr);
  ierr = PetscMalloc1((user->Nspec+1)*LENGTHOFSPECNAME,&names);CHKERRQ(ierr);
  ierr = PetscStrcpy(names,"Temp");CHKERRQ(ierr);
  TC_getSnames(user->Nspec,names+LENGTHOFSPECNAME);CHKERRQ(ierr);
  ierr = PetscMalloc1((user->Nspec+2),&snames);CHKERRQ(ierr);
  for (i=0; i<user->Nspec+1; i++) snames[i] = names+i*LENGTHOFSPECNAME;
  snames[user->Nspec+1] = NULL;
  ierr = TSMonitorLGCtxSetVariableNames(ctx,(const char * const *)snames);CHKERRQ(ierr);
  ierr = PetscFree(snames);CHKERRQ(ierr);
  ierr = PetscFree(names);CHKERRQ(ierr);
  ierr = PetscNew(&uctx);CHKERRQ(ierr);
  uctx->cell = cell;
  uctx->user = user;
  ierr = TSMonitorLGCtxSetTransform(ctx,(PetscErrorCode (*)(void*,Vec,Vec*))FormMoleFraction,(PetscErrorCode (*)(void*))MonitorCellDestroy,uctx);CHKERRQ(ierr);
  ierr = TSMonitorSet(ts,TSMonitorLGSolution,ctx,(PetscErrorCode (*)(void**))TSMonitorLGCtxDestroy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
