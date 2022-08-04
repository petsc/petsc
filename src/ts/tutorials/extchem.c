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

#define PetscCallTC(ierr) do {PetscCheck(!ierr,PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in TChem library, return code %d",ierr);} while (0)

int main(int argc,char **argv)
{
  TS                ts;         /* time integrator */
  TSAdapt           adapt;
  Vec               X,lambda;          /* solution vector */
  Mat               J;          /* Jacobian matrix */
  PetscInt          steps;
  PetscReal         ftime,dt;
  char              chemfile[PETSC_MAX_PATH_LEN],thermofile[PETSC_MAX_PATH_LEN],lchemfile[PETSC_MAX_PATH_LEN],lthermofile[PETSC_MAX_PATH_LEN],lperiodic[PETSC_MAX_PATH_LEN];
  const char        *periodic = "file://${PETSC_DIR}/${PETSC_ARCH}/share/periodictable.dat";
  struct _User      user;       /* user-defined work context */
  TSConvergedReason reason;
  char              **snames,*names;
  PetscInt          i;
  TSTrajectory      tj;
  PetscBool         flg = PETSC_FALSE,tflg = PETSC_FALSE,found;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Chemistry solver options","");
  PetscCall(PetscOptionsString("-chem","CHEMKIN input file","",chemfile,chemfile,sizeof(chemfile),NULL));
  PetscCall(PetscFileRetrieve(PETSC_COMM_WORLD,chemfile,lchemfile,PETSC_MAX_PATH_LEN,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_FILE_OPEN,"Cannot download %s and no local version %s",chemfile,lchemfile);
  PetscCall(PetscOptionsString("-thermo","NASA thermo input file","",thermofile,thermofile,sizeof(thermofile),NULL));
  PetscCall(PetscFileRetrieve(PETSC_COMM_WORLD,thermofile,lthermofile,PETSC_MAX_PATH_LEN,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_FILE_OPEN,"Cannot download %s and no local version %s",thermofile,lthermofile);
  user.pressure = 1.01325e5;    /* Pascal */
  PetscCall(PetscOptionsReal("-pressure","Pressure of reaction [Pa]","",user.pressure,&user.pressure,NULL));
  user.Tini = 1000;             /* Kelvin */
  PetscCall(PetscOptionsReal("-Tini","Initial temperature [K]","",user.Tini,&user.Tini,NULL));
  PetscCall(PetscOptionsBool("-monitor_mass","Monitor the total mass at each timestep","",flg,&flg,NULL));
  PetscCall(PetscOptionsBool("-monitor_temp","Monitor the temperature each timestep","",tflg,&tflg,NULL));
  PetscOptionsEnd();

  /* tchem requires periodic table in current directory */
  PetscCall(PetscFileRetrieve(PETSC_COMM_WORLD,periodic,lperiodic,PETSC_MAX_PATH_LEN,&found));
  PetscCheck(found,PETSC_COMM_WORLD,PETSC_ERR_FILE_OPEN,"Cannot located required periodic table %s or local version %s",periodic,lperiodic);

  PetscCallTC(TC_initChem(lchemfile, lthermofile, 0, 1.0));
  TC_setThermoPres(user.pressure);
  user.Nspec = TC_getNspec();
  user.Nreac = TC_getNreac();
  /*
      Get names of all species in easy to use array
  */
  PetscCall(PetscMalloc1((user.Nspec+1)*LENGTHOFSPECNAME,&names));
  PetscCall(PetscStrcpy(names,"Temp"));
  TC_getSnames(user.Nspec,names+LENGTHOFSPECNAME);
  PetscCall(PetscMalloc1((user.Nspec+2),&snames));
  for (i=0; i<user.Nspec+1; i++) snames[i] = names+i*LENGTHOFSPECNAME;
  snames[user.Nspec+1] = NULL;
  PetscCall(PetscStrArrayallocpy((const char *const *)snames,&user.snames));
  PetscCall(PetscFree(snames));
  PetscCall(PetscFree(names));

  PetscCall(PetscMalloc3(user.Nspec+1,&user.tchemwork,PetscSqr(user.Nspec+1),&user.Jdense,user.Nspec+1,&user.rows));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,user.Nspec+1,&X));

  /* PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,user.Nspec+1,user.Nspec+1,PETSC_DECIDE,NULL,&J)); */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF,user.Nspec+1,user.Nspec+1,NULL,&J));
  PetscCall(MatSetFromOptions(J));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD,&ts));
  PetscCall(TSSetType(ts,TSARKIMEX));
  PetscCall(TSARKIMEXSetFullyImplicit(ts,PETSC_TRUE));
  PetscCall(TSARKIMEXSetType(ts,TSARKIMEX4));
  PetscCall(TSSetRHSFunction(ts,NULL,FormRHSFunction,&user));
  PetscCall(TSSetRHSJacobian(ts,J,J,FormRHSJacobian,&user));

  if (flg) PetscCall(TSMonitorSet(ts,MonitorMassConservation,NULL,NULL));
  if (tflg) {
    PetscCall(TSMonitorSet(ts,MonitorTempature,&user,NULL));
  }

  ftime = 1.0;
  PetscCall(TSSetMaxTime(ts,ftime));
  PetscCall(TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(FormInitialSolution(ts,X,&user));
  PetscCall(TSSetSolution(ts,X));
  dt   = 1e-10;                 /* Initial time step */
  PetscCall(TSSetTimeStep(ts,dt));
  PetscCall(TSGetAdapt(ts,&adapt));
  PetscCall(TSAdaptSetStepLimits(adapt,1e-12,1e-4)); /* Also available with -ts_adapt_dt_min/-ts_adapt_dt_max */
  PetscCall(TSSetMaxSNESFailures(ts,-1));            /* Retry step an unlimited number of times */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set final conditions for sensitivities
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(VecDuplicate(X,&lambda));
  PetscCall(TSSetCostGradients(ts,1,&lambda,NULL));
  PetscCall(VecSetValue(lambda,0,1.0,INSERT_VALUES));
  PetscCall(VecAssemblyBegin(lambda));
  PetscCall(VecAssemblyEnd(lambda));

  PetscCall(TSGetTrajectory(ts,&tj));
  if (tj) {
    PetscCall(TSTrajectorySetVariableNames(tj,(const char * const *)user.snames));
    PetscCall(TSTrajectorySetTransform(tj,(PetscErrorCode (*)(void*,Vec,Vec*))MassFractionToMoleFraction,NULL,&user));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Pass information to graphical monitoring routine
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSMonitorLGSetVariableNames(ts,(const char * const *)user.snames));
  PetscCall(TSMonitorLGSetTransform(ts,(PetscErrorCode (*)(void*,Vec,Vec*))MassFractionToMoleFraction,NULL,&user));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve ODE
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts,X));
  PetscCall(TSGetSolveTime(ts,&ftime));
  PetscCall(TSGetStepNumber(ts,&steps));
  PetscCall(TSGetConvergedReason(ts,&reason));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%s at time %g after %" PetscInt_FMT " steps\n",TSConvergedReasons[reason],(double)ftime,steps));

  /* {
    Vec                max;
    PetscInt           i;
    const PetscReal    *bmax;

    PetscCall(TSMonitorEnvelopeGetBounds(ts,&max,NULL));
    if (max) {
      PetscCall(VecGetArrayRead(max,&bmax));
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"Species - maximum mass fraction\n"));
      for (i=1; i<user.Nspec; i++) {
        if (bmax[i] > .01) PetscCall(PetscPrintf(PETSC_COMM_SELF,"%s %g\n",user.snames[i],(double)bmax[i]));
      }
      PetscCall(VecRestoreArrayRead(max,&bmax));
    }
  }

  Vec y;
  PetscCall(MassFractionToMoleFraction(&user,X,&y));
  PetscCall(PrintSpecies(&user,y));
  PetscCall(VecDestroy(&y)); */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  TC_reset();
  PetscCall(PetscStrArrayDestroy(&user.snames));
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&lambda));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFree3(user.tchemwork,user.Jdense,user.rows));
  PetscCall(PetscFinalize());
  return 0;
}

static PetscErrorCode FormRHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ptr)
{
  User              user = (User)ptr;
  PetscScalar       *f;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X,&x));
  PetscCall(VecGetArray(F,&f));

  PetscCall(PetscArraycpy(user->tchemwork,x,user->Nspec+1));
  user->tchemwork[0] *= user->Tini; /* Dimensionalize */
  PetscCallTC(TC_getSrc(user->tchemwork,user->Nspec+1,f));
  f[0] /= user->Tini;           /* Non-dimensionalize */

  PetscCall(VecRestoreArrayRead(X,&x));
  PetscCall(VecRestoreArray(F,&f));
  PetscFunctionReturn(0);
}

static PetscErrorCode FormRHSJacobian(TS ts,PetscReal t,Vec X,Mat Amat,Mat Pmat,void *ptr)
{
  User              user = (User)ptr;
  const PetscScalar *x;
  PetscInt          M = user->Nspec+1,i;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(X,&x));
  PetscCall(PetscArraycpy(user->tchemwork,x,user->Nspec+1));
  PetscCall(VecRestoreArrayRead(X,&x));
  user->tchemwork[0] *= user->Tini;  /* Dimensionalize temperature (first row) because that is what Tchem wants */
  PetscCall(TC_getJacTYN(user->tchemwork,user->Nspec,user->Jdense,1));

  for (i=0; i<M; i++) user->Jdense[i + 0*M] /= user->Tini; /* Non-dimensionalize first column */
  for (i=0; i<M; i++) user->Jdense[0 + i*M] /= user->Tini; /* Non-dimensionalize first row */
  for (i=0; i<M; i++) user->rows[i] = i;
  PetscCall(MatSetOption(Pmat,MAT_ROW_ORIENTED,PETSC_FALSE));
  PetscCall(MatSetOption(Pmat,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE));
  PetscCall(MatZeroEntries(Pmat));
  PetscCall(MatSetValues(Pmat,M,user->rows,M,user->rows,user->Jdense,INSERT_VALUES));

  PetscCall(MatAssemblyBegin(Pmat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Pmat,MAT_FINAL_ASSEMBLY));
  if (Amat != Pmat) {
    PetscCall(MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FormInitialSolution(TS ts,Vec X,void *ctx)
{
  PetscScalar    *x;
  PetscInt       i;
  Vec            y;
  const PetscInt maxspecies = 10;
  PetscInt       smax = maxspecies,mmax = maxspecies;
  char           *names[maxspecies];
  PetscReal      molefracs[maxspecies],sum;
  PetscBool      flg;

  PetscFunctionBeginUser;
  PetscCall(VecZeroEntries(X));
  PetscCall(VecGetArray(X,&x));
  x[0] = 1.0;  /* Non-dimensionalized by user->Tini */

  PetscCall(PetscOptionsGetStringArray(NULL,NULL,"-initial_species",names,&smax,&flg));
  PetscCheck(smax >= 2,PETSC_COMM_SELF,PETSC_ERR_USER,"Must provide at least two initial species");
  PetscCall(PetscOptionsGetRealArray(NULL,NULL,"-initial_mole",molefracs,&mmax,&flg));
  PetscCheck(smax == mmax,PETSC_COMM_SELF,PETSC_ERR_USER,"Must provide same number of initial species %" PetscInt_FMT " as initial moles %" PetscInt_FMT,smax,mmax);
  sum = 0;
  for (i=0; i<smax; i++) sum += molefracs[i];
  for (i=0; i<smax; i++) molefracs[i] = molefracs[i]/sum;
  for (i=0; i<smax; i++) {
    int ispec = TC_getSpos(names[i], strlen(names[i]));
    PetscCheck(ispec >= 0,PETSC_COMM_SELF,PETSC_ERR_USER,"Could not find species %s",names[i]);
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"Species %" PetscInt_FMT ": %s %g\n",i,names[i],(double)molefracs[i]));
    x[1+ispec] = molefracs[i];
  }
  for (i=0; i<smax; i++) {
    PetscCall(PetscFree(names[i]));
  }
  PetscCall(VecRestoreArray(X,&x));
  /* PetscCall(PrintSpecies((User)ctx,X)); */
  PetscCall(MoleFractionToMassFraction((User)ctx,X,&y));
  PetscCall(VecCopy(y,X));
  PetscCall(VecDestroy(&y));
  PetscFunctionReturn(0);
}

/*
   Converts the input vector which is in mass fractions (used by tchem) to mole fractions
*/
PetscErrorCode MassFractionToMoleFraction(User user,Vec massf,Vec *molef)
{
  PetscScalar       *mof;
  const PetscScalar *maf;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(massf,molef));
  PetscCall(VecGetArrayRead(massf,&maf));
  PetscCall(VecGetArray(*molef,&mof));
  mof[0] = maf[0]; /* copy over temperature */
  TC_getMs2Ml((double*)maf+1,user->Nspec,mof+1);
  PetscCall(VecRestoreArray(*molef,&mof));
  PetscCall(VecRestoreArrayRead(massf,&maf));
  PetscFunctionReturn(0);
}

/*
   Converts the input vector which is in mole fractions to mass fractions (used by tchem)
*/
PetscErrorCode MoleFractionToMassFraction(User user,Vec molef,Vec *massf)
{
  const PetscScalar *mof;
  PetscScalar       *maf;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(molef,massf));
  PetscCall(VecGetArrayRead(molef,&mof));
  PetscCall(VecGetArray(*massf,&maf));
  maf[0] = mof[0]; /* copy over temperature */
  TC_getMl2Ms((double*)mof+1,user->Nspec,maf+1);
  PetscCall(VecRestoreArrayRead(molef,&mof));
  PetscCall(VecRestoreArray(*massf,&maf));
  PetscFunctionReturn(0);
}

PetscErrorCode ComputeMassConservation(Vec x,PetscReal *mass,void* ctx)
{
  PetscFunctionBegin;
  PetscCall(VecSum(x,mass));
  PetscFunctionReturn(0);
}

PetscErrorCode MonitorMassConservation(TS ts,PetscInt step,PetscReal time,Vec x,void* ctx)
{
  const PetscScalar  *T;
  PetscReal          mass;

  PetscFunctionBegin;
  PetscCall(ComputeMassConservation(x,&mass,ctx));
  PetscCall(VecGetArrayRead(x,&T));
  mass -= PetscAbsScalar(T[0]);
  PetscCall(VecRestoreArrayRead(x,&T));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Timestep %" PetscInt_FMT " time %g percent mass lost or gained %g\n",step,(double)time,(double)(100.*(1.0 - mass))));
  PetscFunctionReturn(0);
}

PetscErrorCode MonitorTempature(TS ts,PetscInt step,PetscReal time,Vec x,void* ctx)
{
  User               user = (User) ctx;
  const PetscScalar  *T;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x,&T));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Timestep %" PetscInt_FMT " time %g temperature %g\n",step,(double)time,(double)(T[0]*user->Tini)));
  PetscCall(VecRestoreArrayRead(x,&T));
  PetscFunctionReturn(0);
}

/*
   Prints out each species with its name
*/
PETSC_UNUSED PetscErrorCode PrintSpecies(User user,Vec molef)
{
  const PetscScalar *mof;
  PetscInt          i,*idx,n = user->Nspec+1;

  PetscFunctionBegin;
  PetscCall(PetscMalloc1(n,&idx));
  for (i=0; i<n;i++) idx[i] = i;
  PetscCall(VecGetArrayRead(molef,&mof));
  PetscCall(PetscSortRealWithPermutation(n,mof,idx));
  for (i=0; i<n; i++) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"%6s %g\n",user->snames[idx[n-i-1]],(double)PetscRealPart(mof[idx[n-i-1]])));
  }
  PetscCall(PetscFree(idx));
  PetscCall(VecRestoreArrayRead(molef,&mof));
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
