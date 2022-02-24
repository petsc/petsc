static const char help[] = "Integrate chemistry using TChem.\n";

#include <petscts.h>

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
    See extchem.example.1 for how to run an example

    See also h2_10sp.inp for another example

    Determine sensitivity of final temperature on each variables initial conditions
    -ts_dt 1.e-5 -ts_type cn -ts_adjoint_solve -ts_adjoint_view_solution draw

    The solution for component i = 0 is the temperature.

    The solution, i > 0, is the mass fraction, massf[i], of species i, i.e. mass of species i/ total mass of all species

    The mole fraction molef[i], i > 0, is the number of moles of a species/ total number of moles of all species
        Define M[i] = mass per mole of species i then
        molef[i] = massf[i]/(M[i]*(sum_j massf[j]/M[j]))

    FormMoleFraction(User,massf,molef) converts the mass fraction solution of each species to the mole fraction of each species.

    These are other data sets for other possible runs
       https://www-pls.llnl.gov/data/docs/science_and_technology/chemistry/combustion/n_heptane_v3.1_therm.dat
       https://www-pls.llnl.gov/data/docs/science_and_technology/chemistry/combustion/nc7_ver3.1_mech.txt

*/
typedef struct _User *User;
struct _User {
  PetscReal pressure;
  int       Nspec;
  int       Nreac;
  PetscReal Tini;
  double    *tchemwork;
  double    *Jdense;        /* Dense array workspace where Tchem computes the Jacobian */
  PetscInt  *rows;
  char      **snames;
};

static PetscErrorCode PrintSpecies(User,Vec);
static PetscErrorCode MassFractionToMoleFraction(User,Vec,Vec*);
static PetscErrorCode MoleFractionToMassFraction(User,Vec,Vec*);
static PetscErrorCode FormRHSFunction(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FormRHSJacobian(TS,PetscReal,Vec,Mat,Mat,void*);
static PetscErrorCode FormInitialSolution(TS,Vec,void*);
static PetscErrorCode ComputeMassConservation(Vec,PetscReal*,void*);
static PetscErrorCode MonitorMassConservation(TS,PetscInt,PetscReal,Vec,void*);
static PetscErrorCode MonitorTempature(TS,PetscInt,PetscReal,Vec,void*);

#define CHKERRTC(ierr) do {PetscCheck(!ierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in TChem library, return code %d",ierr);} while (0)

int main(int argc,char **argv)
{
  TS                ts;         /* time integrator */
  TSAdapt           adapt;
  Vec               X,lambda;          /* solution vector */
  Mat               J;          /* Jacobian matrix */
  PetscInt          steps;
  PetscErrorCode    ierr;
  PetscReal         ftime,dt;
  char              chemfile[PETSC_MAX_PATH_LEN],thermofile[PETSC_MAX_PATH_LEN],lchemfile[PETSC_MAX_PATH_LEN],lthermofile[PETSC_MAX_PATH_LEN],lperiodic[PETSC_MAX_PATH_LEN];
  const char        *periodic = "file://${PETSC_DIR}/${PETSC_ARCH}/share/periodictable.dat";
  struct _User      user;       /* user-defined work context */
  TSConvergedReason reason;
  char              **snames,*names;
  PetscInt          i;
  TSTrajectory      tj;
  PetscBool         flg = PETSC_FALSE,tflg = PETSC_FALSE,found;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Chemistry solver options","");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsString("-chem","CHEMKIN input file","",chemfile,chemfile,sizeof(chemfile),NULL));
  CHKERRQ(PetscFileRetrieve(PETSC_COMM_WORLD,chemfile,lchemfile,PETSC_MAX_PATH_LEN,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_FILE_OPEN,"Cannot download %s and no local version %s",chemfile,lchemfile);
  CHKERRQ(PetscOptionsString("-thermo","NASA thermo input file","",thermofile,thermofile,sizeof(thermofile),NULL));
  CHKERRQ(PetscFileRetrieve(PETSC_COMM_WORLD,thermofile,lthermofile,PETSC_MAX_PATH_LEN,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_FILE_OPEN,"Cannot download %s and no local version %s",thermofile,lthermofile);
  user.pressure = 1.01325e5;    /* Pascal */
  CHKERRQ(PetscOptionsReal("-pressure","Pressure of reaction [Pa]","",user.pressure,&user.pressure,NULL));
  user.Tini = 1000;             /* Kelvin */
  CHKERRQ(PetscOptionsReal("-Tini","Initial temperature [K]","",user.Tini,&user.Tini,NULL));
  CHKERRQ(PetscOptionsBool("-monitor_mass","Monitor the total mass at each timestep","",flg,&flg,NULL));
  CHKERRQ(PetscOptionsBool("-monitor_temp","Monitor the temperature each timestep","",tflg,&tflg,NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* tchem requires periodic table in current directory */
  CHKERRQ(PetscFileRetrieve(PETSC_COMM_WORLD,periodic,lperiodic,PETSC_MAX_PATH_LEN,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_FILE_OPEN,"Cannot located required periodic table %s or local version %s",periodic,lperiodic);

  CHKERRTC(TC_initChem(lchemfile, lthermofile, 0, 1.0));
  TC_setThermoPres(user.pressure);
  user.Nspec = TC_getNspec();
  user.Nreac = TC_getNreac();
  /*
      Get names of all species in easy to use array
  */
  CHKERRQ(PetscMalloc1((user.Nspec+1)*LENGTHOFSPECNAME,&names));
  CHKERRQ(PetscStrcpy(names,"Temp"));
  TC_getSnames(user.Nspec,names+LENGTHOFSPECNAME);
  CHKERRQ(PetscMalloc1((user.Nspec+2),&snames));
  for (i=0; i<user.Nspec+1; i++) snames[i] = names+i*LENGTHOFSPECNAME;
  snames[user.Nspec+1] = NULL;
  CHKERRQ(PetscStrArrayallocpy((const char *const *)snames,&user.snames));
  CHKERRQ(PetscFree(snames));
  CHKERRQ(PetscFree(names));

  CHKERRQ(PetscMalloc3(user.Nspec+1,&user.tchemwork,PetscSqr(user.Nspec+1),&user.Jdense,user.Nspec+1,&user.rows));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,user.Nspec+1,&X));

  /* CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_SELF,user.Nspec+1,user.Nspec+1,PETSC_DECIDE,NULL,&J)); */
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_SELF,user.Nspec+1,user.Nspec+1,NULL,&J));
  CHKERRQ(MatSetFromOptions(J));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSCreate(PETSC_COMM_WORLD,&ts));
  CHKERRQ(TSSetType(ts,TSARKIMEX));
  CHKERRQ(TSARKIMEXSetFullyImplicit(ts,PETSC_TRUE));
  CHKERRQ(TSARKIMEXSetType(ts,TSARKIMEX4));
  CHKERRQ(TSSetRHSFunction(ts,NULL,FormRHSFunction,&user));
  CHKERRQ(TSSetRHSJacobian(ts,J,J,FormRHSJacobian,&user));

  if (flg) {
    CHKERRQ(TSMonitorSet(ts,MonitorMassConservation,NULL,NULL));
  }
  if (tflg) {
    CHKERRQ(TSMonitorSet(ts,MonitorTempature,&user,NULL));
  }

  ftime = 1.0;
  CHKERRQ(TSSetMaxTime(ts,ftime));
  CHKERRQ(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(FormInitialSolution(ts,X,&user));
  CHKERRQ(TSSetSolution(ts,X));
  dt   = 1e-10;                 /* Initial time step */
  CHKERRQ(TSSetTimeStep(ts,dt));
  CHKERRQ(TSGetAdapt(ts,&adapt));
  CHKERRQ(TSAdaptSetStepLimits(adapt,1e-12,1e-4)); /* Also available with -ts_adapt_dt_min/-ts_adapt_dt_max */
  CHKERRQ(TSSetMaxSNESFailures(ts,-1));            /* Retry step an unlimited number of times */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set final conditions for sensitivities
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(VecDuplicate(X,&lambda));
  CHKERRQ(TSSetCostGradients(ts,1,&lambda,NULL));
  CHKERRQ(VecSetValue(lambda,0,1.0,INSERT_VALUES));
  CHKERRQ(VecAssemblyBegin(lambda));
  CHKERRQ(VecAssemblyEnd(lambda));

  CHKERRQ(TSGetTrajectory(ts,&tj));
  if (tj) {
    CHKERRQ(TSTrajectorySetVariableNames(tj,(const char * const *)user.snames));
    CHKERRQ(TSTrajectorySetTransform(tj,(PetscErrorCode (*)(void*,Vec,Vec*))MassFractionToMoleFraction,NULL,&user));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Pass information to graphical monitoring routine
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSMonitorLGSetVariableNames(ts,(const char * const *)user.snames));
  CHKERRQ(TSMonitorLGSetTransform(ts,(PetscErrorCode (*)(void*,Vec,Vec*))MassFractionToMoleFraction,NULL,&user));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve ODE
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  CHKERRQ(TSSolve(ts,X));
  CHKERRQ(TSGetSolveTime(ts,&ftime));
  CHKERRQ(TSGetStepNumber(ts,&steps));
  CHKERRQ(TSGetConvergedReason(ts,&reason));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%s at time %g after %D steps\n",TSConvergedReasons[reason],(double)ftime,steps));

  /* {
    Vec                max;
    PetscInt           i;
    const PetscReal    *bmax;

    CHKERRQ(TSMonitorEnvelopeGetBounds(ts,&max,NULL));
    if (max) {
      CHKERRQ(VecGetArrayRead(max,&bmax));
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Species - maximum mass fraction\n"));
      for (i=1; i<user.Nspec; i++) {
        if (bmax[i] > .01) CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"%s %g\n",user.snames[i],(double)bmax[i]));
      }
      CHKERRQ(VecRestoreArrayRead(max,&bmax));
    }
  }

  Vec y;
  CHKERRQ(MassFractionToMoleFraction(&user,X,&y));
  CHKERRQ(PrintSpecies(&user,y));
  CHKERRQ(VecDestroy(&y)); */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  TC_reset();
  CHKERRQ(PetscStrArrayDestroy(&user.snames));
  CHKERRQ(MatDestroy(&J));
  CHKERRQ(VecDestroy(&X));
  CHKERRQ(VecDestroy(&lambda));
  CHKERRQ(TSDestroy(&ts));
  CHKERRQ(PetscFree3(user.tchemwork,user.Jdense,user.rows));
  ierr = PetscFinalize();
  return ierr;
}

static PetscErrorCode FormRHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ptr)
{
  User              user = (User)ptr;
  PetscErrorCode    ierr;
  PetscScalar       *f;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(VecGetArray(F,&f));

  CHKERRQ(PetscArraycpy(user->tchemwork,x,user->Nspec+1));
  user->tchemwork[0] *= user->Tini; /* Dimensionalize */
  CHKERRTC(TC_getSrc(user->tchemwork,user->Nspec+1,f));
  f[0] /= user->Tini;           /* Non-dimensionalize */

  CHKERRQ(VecRestoreArrayRead(X,&x));
  CHKERRQ(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode FormRHSJacobian(TS ts,PetscReal t,Vec X,Mat Amat,Mat Pmat,void *ptr)
{
  User              user = (User)ptr;
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscInt          M = user->Nspec+1,i;

  PetscFunctionBeginUser;
  CHKERRQ(VecGetArrayRead(X,&x));
  CHKERRQ(PetscArraycpy(user->tchemwork,x,user->Nspec+1));
  CHKERRQ(VecRestoreArrayRead(X,&x));
  user->tchemwork[0] *= user->Tini;  /* Dimensionalize temperature (first row) because that is what Tchem wants */
  CHKERRQ(TC_getJacTYN(user->tchemwork,user->Nspec,user->Jdense,1));

  for (i=0; i<M; i++) user->Jdense[i + 0*M] /= user->Tini; /* Non-dimensionalize first column */
  for (i=0; i<M; i++) user->Jdense[0 + i*M] /= user->Tini; /* Non-dimensionalize first row */
  for (i=0; i<M; i++) user->rows[i] = i;
  CHKERRQ(MatSetOption(Pmat,MAT_ROW_ORIENTED,PETSC_FALSE));
  CHKERRQ(MatSetOption(Pmat,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE));
  CHKERRQ(MatZeroEntries(Pmat));
  CHKERRQ(MatSetValues(Pmat,M,user->rows,M,user->rows,user->Jdense,INSERT_VALUES));

  CHKERRQ(MatAssemblyBegin(Pmat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(Pmat,MAT_FINAL_ASSEMBLY));
  if (Amat != Pmat) {
    CHKERRQ(MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FormInitialSolution(TS ts,Vec X,void *ctx)
{
  PetscScalar    *x;
  PetscErrorCode ierr;
  PetscInt       i;
  Vec            y;
  const PetscInt maxspecies = 10;
  PetscInt       smax = maxspecies,mmax = maxspecies;
  char           *names[maxspecies];
  PetscReal      molefracs[maxspecies],sum;
  PetscBool      flg;

  PetscFunctionBeginUser;
  CHKERRQ(VecZeroEntries(X));
  CHKERRQ(VecGetArray(X,&x));
  x[0] = 1.0;  /* Non-dimensionalized by user->Tini */

  CHKERRQ(PetscOptionsGetStringArray(NULL,NULL,"-initial_species",names,&smax,&flg));
  PetscCheck(smax >= 2,PETSC_COMM_SELF,PETSC_ERR_USER,"Must provide at least two initial species");
  CHKERRQ(PetscOptionsGetRealArray(NULL,NULL,"-initial_mole",molefracs,&mmax,&flg));
  PetscCheck(smax == mmax,PETSC_COMM_SELF,PETSC_ERR_USER,"Must provide same number of initial species %D as initial moles %D",smax,mmax);
  sum = 0;
  for (i=0; i<smax; i++) sum += molefracs[i];
  for (i=0; i<smax; i++) molefracs[i] = molefracs[i]/sum;
  for (i=0; i<smax; i++) {
    int ispec = TC_getSpos(names[i], strlen(names[i]));
    PetscCheck(ispec >= 0,PETSC_COMM_SELF,PETSC_ERR_USER,"Could not find species %s",names[i]);
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Species %d: %s %g\n",i,names[i],molefracs[i]));
    x[1+ispec] = molefracs[i];
  }
  for (i=0; i<smax; i++) {
    CHKERRQ(PetscFree(names[i]));
  }
  CHKERRQ(VecRestoreArray(X,&x));
  /* CHKERRQ(PrintSpecies((User)ctx,X)); */
  CHKERRQ(MoleFractionToMassFraction((User)ctx,X,&y));
  CHKERRQ(VecCopy(y,X));
  CHKERRQ(VecDestroy(&y));
  PetscFunctionReturn(0);
}

/*
   Converts the input vector which is in mass fractions (used by tchem) to mole fractions
*/
PetscErrorCode MassFractionToMoleFraction(User user,Vec massf,Vec *molef)
{
  PetscErrorCode    ierr;
  PetscScalar       *mof;
  const PetscScalar *maf;

  PetscFunctionBegin;
  CHKERRQ(VecDuplicate(massf,molef));
  CHKERRQ(VecGetArrayRead(massf,&maf));
  CHKERRQ(VecGetArray(*molef,&mof));
  mof[0] = maf[0]; /* copy over temperature */
  TC_getMs2Ml((double*)maf+1,user->Nspec,mof+1);
  CHKERRQ(VecRestoreArray(*molef,&mof));
  CHKERRQ(VecRestoreArrayRead(massf,&maf));
  PetscFunctionReturn(0);
}

/*
   Converts the input vector which is in mole fractions to mass fractions (used by tchem)
*/
PetscErrorCode MoleFractionToMassFraction(User user,Vec molef,Vec *massf)
{
  PetscErrorCode    ierr;
  const PetscScalar *mof;
  PetscScalar       *maf;

  PetscFunctionBegin;
  CHKERRQ(VecDuplicate(molef,massf));
  CHKERRQ(VecGetArrayRead(molef,&mof));
  CHKERRQ(VecGetArray(*massf,&maf));
  maf[0] = mof[0]; /* copy over temperature */
  TC_getMl2Ms((double*)mof+1,user->Nspec,maf+1);
  CHKERRQ(VecRestoreArrayRead(molef,&mof));
  CHKERRQ(VecRestoreArray(*massf,&maf));
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeMassConservation(Vec x,PetscReal *mass,void* ctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  CHKERRQ(VecSum(x,mass));
  PetscFunctionReturn(0);
}

PetscErrorCode MonitorMassConservation(TS ts,PetscInt step,PetscReal time,Vec x,void* ctx)
{
  const PetscScalar  *T;
  PetscReal          mass;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  CHKERRQ(ComputeMassConservation(x,&mass,ctx));
  CHKERRQ(VecGetArrayRead(x,&T));
  mass -= PetscAbsScalar(T[0]);
  CHKERRQ(VecRestoreArrayRead(x,&T));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Timestep %D time %g percent mass lost or gained %g\n",step,(double)time,(double)100.*(1.0 - mass)));
  PetscFunctionReturn(0);
}

PetscErrorCode MonitorTempature(TS ts,PetscInt step,PetscReal time,Vec x,void* ctx)
{
  User               user = (User) ctx;
  const PetscScalar  *T;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  CHKERRQ(VecGetArrayRead(x,&T));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Timestep %D time %g temperature %g\n",step,(double)time,(double)T[0]*user->Tini));
  CHKERRQ(VecRestoreArrayRead(x,&T));
  PetscFunctionReturn(0);
}

/*
   Prints out each species with its name
*/
PETSC_UNUSED PetscErrorCode PrintSpecies(User user,Vec molef)
{
  PetscErrorCode    ierr;
  const PetscScalar *mof;
  PetscInt          i,*idx,n = user->Nspec+1;

  PetscFunctionBegin;
  CHKERRQ(PetscMalloc1(n,&idx));
  for (i=0; i<n;i++) idx[i] = i;
  CHKERRQ(VecGetArrayRead(molef,&mof));
  CHKERRQ(PetscSortRealWithPermutation(n,mof,idx));
  for (i=0; i<n; i++) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"%6s %g\n",user->snames[idx[n-i-1]],mof[idx[n-i-1]]));
  }
  CHKERRQ(PetscFree(idx));
  CHKERRQ(VecRestoreArrayRead(molef,&mof));
  PetscFunctionReturn(0);
}

/*TEST
    build:
      requires: tchem

    test:
      args: -chem http://combustion.berkeley.edu/gri_mech/version30/files30/grimech30.dat -thermo http://combustion.berkeley.edu/gri_mech/version30/files30/thermo30.dat -initial_species CH4,O2,N2,AR -initial_mole 0.0948178320887,0.189635664177,0.706766236705,0.00878026702874 -Tini 1500 -Tini 1500 -ts_arkimex_fully_implicit -ts_max_snes_failures -1  -ts_adapt_dt_max 1e-4 -ts_arkimex_type 4 -ts_max_time .005
      requires: !single
      filter: grep -v iterations

TEST*/
