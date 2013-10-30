static const char help[] = "Integrate chemistry using TChem.\n";

#include <petscts.h>

#if defined(PETSC_HAVE_TCHEM)
#  include <TC_interface.h>
#else
#  error TChem is required for this example.  Reconfigure PETSc using --download-tchem.
#endif

typedef struct _User *User;
struct _User {
  PetscReal pressure;
  int       Nspec;
  int       Nreac;
  PetscReal Tini;
  double    *tchemwork;
  double    *Jdense;
  PetscInt  *rows;
};

static PetscErrorCode FormRHSFunction(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FormRHSJacobian(TS,PetscReal,Vec,Mat*,Mat*,MatStructure*,void*);
static PetscErrorCode FormInitialSolution(TS,Vec,void*);

#define TCCHKERRQ(ierr) do {if (ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in TChem library, return code %d",ierr);} while (0)

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  TS                ts;         /* time integrator */
  Vec               X;          /* solution, residual vectors */
  Mat               J;          /* Jacobian matrix */
  PetscInt          steps,maxsteps;
  PetscErrorCode    ierr;
  PetscReal         ftime,dt;
  char              chemfile[PETSC_MAX_PATH_LEN] = "chem.inp",thermofile[PETSC_MAX_PATH_LEN] = "therm.dat";
  struct _User      user;       /* user-defined work context */
  TSConvergedReason reason;

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

  ierr = PetscMalloc3(user.Nspec+1,double,&user.tchemwork,PetscSqr(user.Nspec+1),double,&user.Jdense,user.Nspec+1,PetscInt,&user.rows);CHKERRQ(ierr);
  ierr = VecCreateSeq(PETSC_COMM_SELF,user.Nspec+1,&X);CHKERRQ(ierr);

  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,user.Nspec+1,user.Nspec+1,PETSC_DECIDE,NULL,&J);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSCreate(PETSC_COMM_WORLD,&ts);CHKERRQ(ierr);
  ierr = TSSetType(ts,TSARKIMEX);CHKERRQ(ierr);
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

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve nonlinear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,X);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetTimeStepNumber(ts,&steps);CHKERRQ(ierr);
  ierr = TSGetConvergedReason(ts,&reason);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"%s at time %G after %D steps\n",TSConvergedReasons[reason],ftime,steps);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  TC_reset();
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
static PetscErrorCode FormRHSJacobian(TS ts,PetscReal t,Vec X,Mat *Amat,Mat *Pmat,MatStructure *flag,void *ptr)
{
  User              user = (User)ptr;
  PetscErrorCode    ierr;
  const PetscScalar *x;
  PetscInt          M = user->Nspec+1,i;

  PetscFunctionBeginUser;
  ierr = VecGetArrayRead(X,&x);CHKERRQ(ierr);
  ierr = PetscMemcpy(user->tchemwork,x,(user->Nspec+1)*sizeof(x[0]));CHKERRQ(ierr);
  user->tchemwork[0] *= user->Tini;
  ierr = TC_getJacTYN(user->tchemwork,user->Nspec,user->Jdense,1);CHKERRQ(ierr);

  for (i=0; i<M; i++) user->Jdense[i + 0*M] /= user->Tini; /* Non-dimensionalize first column */
  for (i=0; i<M; i++) user->Jdense[0 + i*M] /= user->Tini; /* Non-dimensionalize first row */
  for (i=0; i<M; i++) user->rows[i] = i;
  for (i=0; i<M; i++) {
    ierr = MatSetValues(*Pmat,M,user->rows,1,&i,&user->Jdense[0+i*M],INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*Pmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*Pmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (*Amat != *Pmat) {
    ierr = MatAssemblyBegin(*Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(*Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  *flag = DIFFERENT_NONZERO_PATTERN;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "FormInitialSolution"
PetscErrorCode FormInitialSolution(TS ts,Vec X,void *ctx)
{
  PetscScalar    *x;
  PetscErrorCode ierr;
  struct {const char *name; PetscReal massfrac;} initial[] = {
    {"CH4", 0.0948178320887},
    {"O2", 0.189635664177},
    {"N2", 0.706766236705},
    {"AR", 0.00878026702874}
  };
  PetscInt i;

  PetscFunctionBeginUser;
  ierr = VecZeroEntries(X);CHKERRQ(ierr);
  ierr = VecGetArray(X,&x);CHKERRQ(ierr);
  x[0] = 1.0;  /* Non-dimensionalized by user->Tini */

  for (i=0; i<sizeof(initial)/sizeof(initial[0]); i++) {
    int ispec = TC_getSpos(initial[i].name, strlen(initial[i].name));
    if (ispec < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Could not find species %s",initial[i].name);
    ierr = PetscPrintf(PETSC_COMM_SELF,"Species %d: %s\n",i,initial[i].name);CHKERRQ(ierr);
    x[1+ispec] = initial[i].massfrac;
  }
  ierr = VecRestoreArray(X,&x);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
