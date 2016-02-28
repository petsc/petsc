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
    Obtain the three files into this directory

       curl http://combustion.berkeley.edu/gri_mech/version30/files30/grimech30.dat > chem.inp
       curl http://combustion.berkeley.edu/gri_mech/version30/files30/thermo30.dat > therm.dat

       https://www-pls.llnl.gov/data/docs/science_and_technology/chemistry/combustion/n_heptane_v3.1_therm.dat
       https://www-pls.llnl.gov/data/docs/science_and_technology/chemistry/combustion/nc7_ver3.1_mech.txt

       cp $PETSC_DIR/$PETSC_ARCH/externalpackages/tchem/data/periodictable.dat .

    Run with
   ./extchem -Tini 1500 -ts_arkimex_fully_implicit -ts_max_snes_failures -1 -ts_adapt_monitor -ts_adapt_dt_max 1e-4 -ts_arkimex_type 4 -ts_monitor_lg_solution -ts_final_time .005 -draw_pause -2 -lg_use_markers false -ts_monitor_lg_solution_variables H2,O2,H2O,CH4,CO,CO2,C2H2,N2  -ts_monitor_envelope

    Determine sensitivity of final tempature on each variables initial conditions
    -ts_dt 1.e-5 -ts_type cn -ts_adjoint_solve -ts_adjoint_view_solution draw

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

#define TCCHKERRQ(ierr) do {if (ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in TChem library, return code %d",ierr);} while (0)

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  TS                ts;         /* time integrator */
  TSAdapt           adapt;
  Vec               X,lambda;          /* solution vector */
  Mat               J;          /* Jacobian matrix */
  PetscInt          steps,maxsteps;
  PetscErrorCode    ierr;
  PetscReal         ftime,dt;
  char              chemfile[PETSC_MAX_PATH_LEN] = "chem.inp",thermofile[PETSC_MAX_PATH_LEN] = "therm.dat";
  struct _User      user;       /* user-defined work context */
  TSConvergedReason reason;
  char              **snames,*names;
  PetscInt          i;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Chemistry solver options","");CHKERRQ(ierr);
  ierr = PetscOptionsString("-chem","CHEMKIN input file","",chemfile,chemfile,sizeof(chemfile),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-thermo","NASA thermo input file","",thermofile,thermofile,sizeof(thermofile),NULL);CHKERRQ(ierr);
  user.pressure = 1.01325e5;    /* Pascal */
  ierr = PetscOptionsReal("-pressure","Pressure of reaction [Pa]","",user.pressure,&user.pressure,NULL);CHKERRQ(ierr);
  user.Tini = 1000;             /* Kelvin */
  ierr = PetscOptionsReal("-Tini","Initial temperature [K]","",user.Tini,&user.Tini,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = TC_initChem(chemfile, thermofile, 0, 1.0);TCCHKERRQ(ierr);
  user.Nspec = TC_getNspec();
  user.Nreac = TC_getNreac();
  /*
      Get names of all species in easy to use array
  */
  ierr = PetscMalloc1((user.Nspec+1)*LENGTHOFSPECNAME,&names);CHKERRQ(ierr);
  ierr = PetscStrcpy(names,"Temp");CHKERRQ(ierr);
  TC_getSnames(user.Nspec,names+LENGTHOFSPECNAME);CHKERRQ(ierr);
  ierr = PetscMalloc1((user.Nspec+2),&snames);CHKERRQ(ierr);
  for (i=0; i<user.Nspec+1; i++) snames[i] = names+i*LENGTHOFSPECNAME;
  snames[user.Nspec+1] = NULL;
  ierr = PetscStrArrayallocpy((const char *const *)snames,&user.snames);CHKERRQ(ierr);
  ierr = PetscFree(snames);CHKERRQ(ierr);
  ierr = PetscFree(names);CHKERRQ(ierr);

  ierr = PetscMalloc3(user.Nspec+1,&user.tchemwork,PetscSqr(user.Nspec+1),&user.Jdense,user.Nspec+1,&user.rows);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,user.Nspec+1,&X);CHKERRQ(ierr);

  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,user.Nspec+1,user.Nspec+1,PETSC_DECIDE,NULL,&J);CHKERRQ(ierr);
  /*ierr = MatCreateSeqDense(PETSC_COMM_SELF,user.Nspec+1,user.Nspec+1,NULL,&J);CHKERRQ(ierr);*/
  ierr = MatSetFromOptions(J);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
  ierr = TSARKIMEXSetFullyImplicit(ts,PETSC_TRUE);CHKERRQ(ierr);
  ierr = TSARKIMEXSetType(ts,TSARKIMEX4);CHKERRQ(ierr);
  ierr = TSSetRHSFunction(ts,NULL,FormRHSFunction,&user);CHKERRQ(ierr);
  ierr = TSSetRHSJacobian(ts,J,J,FormRHSJacobian,&user);CHKERRQ(ierr);

  ftime    = 1.0;
  maxsteps = 10000;
  ierr     = TSSetDuration(ts,maxsteps,ftime);CHKERRQ(ierr);
  ierr     = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);

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
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set final conditions for sensitivities
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = VecDuplicate(X,&lambda);CHKERRQ(ierr);
  ierr = TSSetCostGradients(ts,1,&lambda,NULL);CHKERRQ(ierr);
  ierr = VecSetValue(lambda,0,1.0,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(lambda);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(lambda);CHKERRQ(ierr);



  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Pass information to graphical monitoring routine
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSMonitorLGSetVariableNames(ts,(const char * const *)user.snames);CHKERRQ(ierr);
  ierr = TSMonitorLGSetTransform(ts,(PetscErrorCode (*)(void*,Vec,Vec*))MassFractionToMoleFraction,NULL,&user);CHKERRQ(ierr);

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

  Vec y;
  MassFractionToMoleFraction(&user,X,&y);CHKERRQ(ierr);
  PrintSpecies(&user,y);CHKERRQ(ierr);
  VecDestroy(&y);
  
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  TC_reset();
  ierr = PetscStrArrayDestroy(&user.snames);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&lambda);CHKERRQ(ierr);
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
  PetscScalar       *f;
  const PetscScalar *x;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecGetArray(F,&f);CHKERRQ(ierr);

  ierr = PetscMemcpy(user->tchemwork,x,(user->Nspec+1)*sizeof(x[0]));CHKERRQ(ierr);
  user->tchemwork[0] *= user->Tini; /* Dimensionalize */
  ierr = TC_getSrc(user->tchemwork,user->Nspec+1,f);TCCHKERRQ(ierr);
  f[0] /= user->Tini;           /* Non-dimensionalize */

  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  ierr = VecRestoreArray(F,&f);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormRHSJacobian"
static PetscErrorCode FormRHSJacobian(TS ts,PetscReal t,Vec X,Mat Amat,Mat Pmat,void *ptr)
{
  User              user = (User)ptr;
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscInt          M = user->Nspec+1,i;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = PetscMemcpy(user->tchemwork,x,(user->Nspec+1)*sizeof(x[0]));CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(X,&x);CHKERRQ(ierr);
  user->tchemwork[0] *= user->Tini;  /* Dimensionalize temperature (first row) because that is what Tchem wants */
  ierr = TC_getJacTYN(user->tchemwork,user->Nspec,user->Jdense,1);CHKERRQ(ierr);

  for (i=0; i<M; i++) user->Jdense[i + 0*M] /= user->Tini; /* Non-dimensionalize first column */
  for (i=0; i<M; i++) user->Jdense[0 + i*M] /= user->Tini; /* Non-dimensionalize first row */
  for (i=0; i<M; i++) user->rows[i] = i;
  ierr = MatSetOption(Pmat,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
  ierr = MatSetOption(Pmat,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
  ierr = MatZeroEntries(Pmat);CHKERRQ(ierr);
  ierr = MatSetValues(Pmat,M,user->rows,M,user->rows,user->Jdense,INSERT_VALUES);CHKERRQ(ierr);

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
  PetscScalar    *x;
  PetscErrorCode ierr;
  struct {const char *name; PetscReal molefrac;} initial[] = {
    {"CH4", 0.0948178320887},
    {"O2", 0.189635664177},
    {"N2", 0.706766236705},
    {"AR", 0.00878026702874}
  };
  PetscInt i;
  Vec      y;

  PetscFunctionBeginUser;
  ierr = VecZeroEntries(X);CHKERRQ(ierr);
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  x[0] = 1.0;  /* Non-dimensionalized by user->Tini */

  for (i=0; i<sizeof(initial)/sizeof(initial[0]); i++) {
    int ispec = TC_getSpos(initial[i].name, strlen(initial[i].name));
    if (ispec < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Could not find species %s",initial[i].name);
    ierr = PetscPrintf(PETSC_COMM_SELF,"Species %d: %s %g\n",i,initial[i].name,initial[i].molefrac);CHKERRQ(ierr);
    x[1+ispec] = initial[i].molefrac;
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  PrintSpecies((User)ctx,X);CHKERRQ(ierr);
  ierr = MoleFractionToMassFraction((User)ctx,X,&y);CHKERRQ(ierr);
  ierr = VecCopy(y,X);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MassFractionToMoleFraction"
/*
   Converts the input vector which is in mass fractions (used by tchem) to mole fractions
*/
PetscErrorCode MassFractionToMoleFraction(User user,Vec massf,Vec *molef)
{
  PetscErrorCode    ierr;
  PetscScalar       *mof;
  const PetscScalar *maf;

  PetscFunctionBegin;
  ierr = VecDuplicate(massf,molef);CHKERRQ(ierr);
  ierr = VecGetArrayRead(massf,&maf);CHKERRQ(ierr);
  ierr = VecGetArray(*molef,&mof);CHKERRQ(ierr);
  mof[0] = maf[0]; /* copy over temperature */
  TC_getMs2Ml((double*)maf+1,user->Nspec,mof+1);
  ierr = VecRestoreArray(*molef,&mof);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(massf,&maf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MoleFractionToMassFraction"
/*
   Converts the input vector which is in mole fractions to mass fractions (used by tchem)
*/
PetscErrorCode MoleFractionToMassFraction(User user,Vec molef,Vec *massf)
{
  PetscErrorCode    ierr;
  const PetscScalar *mof;
  PetscScalar       *maf;

  PetscFunctionBegin;
  ierr = VecDuplicate(molef,massf);CHKERRQ(ierr);
  ierr = VecGetArrayRead(molef,&mof);CHKERRQ(ierr);
  ierr = VecGetArray(*massf,&maf);CHKERRQ(ierr);
  maf[0] = mof[0]; /* copy over temperature */
  TC_getMl2Ms((double*)mof+1,user->Nspec,maf+1);
  ierr = VecRestoreArrayRead(molef,&mof);CHKERRQ(ierr);
  ierr = VecRestoreArray(*massf,&maf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PrintSpecies"
/*
   Prints out each species with its name
*/
PetscErrorCode PrintSpecies(User user,Vec molef)
{
  PetscErrorCode    ierr;
  const PetscScalar *mof;
  PetscInt          i,*idx,n = user->Nspec+1;

  PetscFunctionBegin;
  ierr = PetscMalloc1(n,&idx);CHKERRQ(ierr);
  for (i=0; i<n;i++) idx[i] = i;
  ierr = VecGetArrayRead(molef,&mof);CHKERRQ(ierr);
  ierr = PetscSortRealWithPermutation(n,mof,idx);CHKERRQ(ierr);
  for (i=0; i<n; i++) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"%6s %g\n",user->snames[idx[n-i-1]],mof[idx[n-i-1]]);CHKERRQ(ierr);
  }
  ierr = PetscFree(idx);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(molef,&mof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
