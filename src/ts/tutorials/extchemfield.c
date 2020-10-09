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
     ./extchemfield  -ts_arkimex_fully_implicit -ts_max_snes_failures -1 -ts_adapt_monitor -ts_adapt_dt_max 1e-4 -ts_arkimex_type 4 -ts_max_time .005

     Options for visualizing the solution:
        Watch certain variables in each cell evolve with time
        -draw_solution 1 -ts_monitor_lg_solution_variables Temp,H2,O2,H2O,CH4,CO,CO2,C2H2,N2 -lg_use_markers false  -draw_pause -2

        Watch certain variables in all cells evolve with time
        -da_refine 4 -ts_monitor_draw_solution -draw_fields_by_name Temp,H2 -draw_vec_mark_points  -draw_pause -2

        Keep the initial temperature distribution as one monitors the current temperature distribution
        -ts_monitor_draw_solution_initial -draw_bounds .9,1.7 -draw_fields_by_name Temp

        Save the images in a .gif (movie) file
        -draw_save -draw_save_single_file

        Compute the sensitivies of the solution of the first tempature on the initial conditions
        -ts_adjoint_solve  -ts_dt 1.e-5 -ts_type cn -ts_adjoint_view_solution draw

        Turn off diffusion
        -diffusion no

        Turn off reactions
        -reactions no


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
  PetscReal Tini,dx;
  PetscReal diffus;
  DM        dm;
  PetscBool diffusion,reactions;
  double    *tchemwork;
  double    *Jdense;        /* Dense array workspace where Tchem computes the Jacobian */
  PetscInt  *rows;
};

static PetscErrorCode MonitorCell(TS,User,PetscInt);
static PetscErrorCode FormRHSFunction(TS,PetscReal,Vec,Vec,void*);
static PetscErrorCode FormRHSJacobian(TS,PetscReal,Vec,Mat,Mat,void*);
static PetscErrorCode FormInitialSolution(TS,Vec,void*);

#define TCCHKERRQ(ierr) do {if (ierr) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in TChem library, return code %d",ierr);} while (0)

int main(int argc,char **argv)
{
  TS                ts;         /* time integrator */
  TSAdapt           adapt;
  Vec               X;          /* solution vector */
  Mat               J;          /* Jacobian matrix */
  PetscInt          steps,ncells,xs,xm,i;
  PetscErrorCode    ierr;
  PetscReal         ftime,dt;
  char              chemfile[PETSC_MAX_PATH_LEN] = "chem.inp",thermofile[PETSC_MAX_PATH_LEN] = "therm.dat";
  struct _User      user;
  TSConvergedReason reason;
  PetscBool         showsolutions = PETSC_FALSE;
  char              **snames,*names;
  Vec               lambda;     /* used with TSAdjoint for sensitivities */

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Chemistry solver options","");CHKERRQ(ierr);
  ierr = PetscOptionsString("-chem","CHEMKIN input file","",chemfile,chemfile,sizeof(chemfile),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-thermo","NASA thermo input file","",thermofile,thermofile,sizeof(thermofile),NULL);CHKERRQ(ierr);
  user.pressure = 1.01325e5;    /* Pascal */
  ierr = PetscOptionsReal("-pressure","Pressure of reaction [Pa]","",user.pressure,&user.pressure,NULL);CHKERRQ(ierr);
  user.Tini   = 1550;
  ierr = PetscOptionsReal("-Tini","Initial temperature [K]","",user.Tini,&user.Tini,NULL);CHKERRQ(ierr);
  user.diffus = 100;
  ierr = PetscOptionsReal("-diffus","Diffusion constant","",user.diffus,&user.diffus,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-draw_solution","Plot the solution for each cell","",showsolutions,&showsolutions,NULL);CHKERRQ(ierr);
  user.diffusion = PETSC_TRUE;
  ierr = PetscOptionsBool("-diffusion","Have diffusion","",user.diffusion,&user.diffusion,NULL);CHKERRQ(ierr);
  user.reactions = PETSC_TRUE;
  ierr = PetscOptionsBool("-reactions","Have reactions","",user.reactions,&user.reactions,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = TC_initChem(chemfile, thermofile, 0, 1.0);TCCHKERRQ(ierr);
  user.Nspec = TC_getNspec();
  user.Nreac = TC_getNreac();

  ierr    = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,10,user.Nspec+1,1,NULL,&user.dm);CHKERRQ(ierr);
  ierr    = DMSetFromOptions(user.dm);CHKERRQ(ierr);
  ierr    = DMSetUp(user.dm);CHKERRQ(ierr);
  ierr    = DMDAGetInfo(user.dm,NULL,&ncells,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  user.dx = 1.0/ncells;  /* Set the coordinates of the cell centers; note final ghost cell is at x coordinate 1.0 */
  ierr    = DMDASetUniformCoordinates(user.dm,0.0,1.0,0.0,1.0,0.0,1.0);CHKERRQ(ierr);

  /* set the names of each field in the DMDA based on the species name */
  ierr = PetscMalloc1((user.Nspec+1)*LENGTHOFSPECNAME,&names);CHKERRQ(ierr);
  ierr = PetscStrcpy(names,"Temp");CHKERRQ(ierr);
  TC_getSnames(user.Nspec,names+LENGTHOFSPECNAME);CHKERRQ(ierr);
  ierr = PetscMalloc1((user.Nspec+2),&snames);CHKERRQ(ierr);
  for (i=0; i<user.Nspec+1; i++) snames[i] = names+i*LENGTHOFSPECNAME;
  snames[user.Nspec+1] = NULL;
  ierr = DMDASetFieldNames(user.dm,(const char * const *)snames);CHKERRQ(ierr);
  ierr = PetscFree(snames);CHKERRQ(ierr);
  ierr = PetscFree(names);CHKERRQ(ierr);


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

  ftime = 1.0;
  ierr = TSSetMaxTime(ts,ftime);CHKERRQ(ierr);
  ierr = TSSetExactFinalTime(ts,TS_EXACTFINALTIME_STEPOVER);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = FormInitialSolution(ts,X,&user);CHKERRQ(ierr);
  ierr = TSSetSolution(ts,X);CHKERRQ(ierr);
  dt   = 1e-10;                 /* Initial time step */
  ierr = TSSetTimeStep(ts,dt);CHKERRQ(ierr);
  ierr = TSGetAdapt(ts,&adapt);CHKERRQ(ierr);
  ierr = TSAdaptSetStepLimits(adapt,1e-12,1e-4);CHKERRQ(ierr); /* Also available with -ts_adapt_dt_min/-ts_adapt_dt_max */
  ierr = TSSetMaxSNESFailures(ts,-1);CHKERRQ(ierr);            /* Retry step an unlimited number of times */


  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Pass information to graphical monitoring routine
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (showsolutions) {
    ierr = DMDAGetCorners(user.dm,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);
    for (i=xs;i<xs+xm;i++) {
      ierr = MonitorCell(ts,&user,i);CHKERRQ(ierr);
    }
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSetFromOptions(ts);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set final conditions for sensitivities
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = DMCreateGlobalVector(user.dm,&lambda);CHKERRQ(ierr);
  ierr = TSSetCostGradients(ts,1,&lambda,NULL);CHKERRQ(ierr);
  ierr = VecSetValue(lambda,0,1.0,INSERT_VALUES);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(lambda);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(lambda);CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve ODE
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = TSSolve(ts,X);CHKERRQ(ierr);
  ierr = TSGetSolveTime(ts,&ftime);CHKERRQ(ierr);
  ierr = TSGetStepNumber(ts,&steps);CHKERRQ(ierr);
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
      if (names) {
        ierr = VecGetArrayRead(max,&bmax);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_SELF,"Species - maximum mass fraction\n");CHKERRQ(ierr);
        for (i=1; i<user.Nspec; i++) {
          if (bmax[i] > .01) {ierr = PetscPrintf(PETSC_COMM_SELF,"%s %g\n",names[i],bmax[i]);CHKERRQ(ierr);}
        }
        ierr = VecRestoreArrayRead(max,&bmax);CHKERRQ(ierr);
      }
    }
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  TC_reset();
  ierr = DMDestroy(&user.dm);CHKERRQ(ierr);
  ierr = MatDestroy(&J);CHKERRQ(ierr);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&lambda);CHKERRQ(ierr);
  ierr = TSDestroy(&ts);CHKERRQ(ierr);
  ierr = PetscFree3(user.tchemwork,user.Jdense,user.rows);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*
   Applies the second order centered difference diffusion operator on a one dimensional periodic domain
*/
static PetscErrorCode FormDiffusionFunction(TS ts,PetscReal t,Vec X,Vec F,void *ptr)
{
  User              user = (User)ptr;
  PetscErrorCode    ierr;
  PetscScalar       **f;
  const PetscScalar **x;
  DM                dm;
  PetscInt          i,xs,xm,j,dof;
  Vec               Xlocal;
  PetscReal         idx;

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMDAGetInfo(dm,NULL,NULL,NULL,NULL,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMGetLocalVector(dm,&Xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(dm,X,INSERT_VALUES,Xlocal);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(dm,X,INSERT_VALUES,Xlocal);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOFRead(dm,Xlocal,&x);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(dm,F,&f);CHKERRQ(ierr);
  ierr = DMDAGetCorners(dm,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  idx = 1.0*user->diffus/user->dx;
  for (i=xs; i<xs+xm; i++) {
    for (j=0; j<dof; j++) {
      f[i][j] += idx*(x[i+1][j] - 2.0*x[i][j] + x[i-1][j]);
    }
  }
  ierr = DMDAVecRestoreArrayDOFRead(dm,Xlocal,&x);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOF(dm,F,&f);CHKERRQ(ierr);
  ierr = DMRestoreLocalVector(dm,&Xlocal);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Produces the second order centered difference diffusion operator on a one dimensional periodic domain
*/
static PetscErrorCode FormDiffusionJacobian(TS ts,PetscReal t,Vec X,Mat Amat,Mat Pmat,void *ptr)
{
  User              user = (User)ptr;
  PetscErrorCode    ierr;
  DM                dm;
  PetscInt          i,xs,xm,j,dof;
  PetscReal         idx,values[3];
  MatStencil        row,col[3];

  PetscFunctionBeginUser;
  ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
  ierr = DMDAGetInfo(dm,NULL,NULL,NULL,NULL,NULL,NULL,NULL,&dof,NULL,NULL,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = DMDAGetCorners(dm,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  idx = 1.0*user->diffus/user->dx;
  values[0] = idx;
  values[1] = -2.0*idx;
  values[2] = idx;
  for (i=xs; i<xs+xm; i++) {
    for (j=0; j<dof; j++) {
      row.i = i;      row.c = j;
      col[0].i = i-1; col[0].c = j;
      col[1].i = i;   col[1].c = j;
      col[2].i = i+1; col[2].c = j;
      ierr = MatSetValuesStencil(Pmat,1,&row,3,col,values,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(Pmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Pmat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode FormRHSFunction(TS ts,PetscReal t,Vec X,Vec F,void *ptr)
{
  User              user = (User)ptr;
  PetscErrorCode    ierr;
  PetscScalar       **f;
  const PetscScalar **x;
  DM                dm;
  PetscInt          i,xs,xm;

  PetscFunctionBeginUser;
  if (user->reactions) {
    ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
    ierr = DMDAVecGetArrayDOFRead(dm,X,&x);CHKERRQ(ierr);
    ierr = DMDAVecGetArrayDOF(dm,F,&f);CHKERRQ(ierr);
    ierr = DMDAGetCorners(dm,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

    for (i=xs; i<xs+xm; i++) {
      ierr = PetscArraycpy(user->tchemwork,x[i],user->Nspec+1);CHKERRQ(ierr);
      user->tchemwork[0] *= user->Tini; /* Dimensionalize */
      ierr = TC_getSrc(user->tchemwork,user->Nspec+1,f[i]);TCCHKERRQ(ierr);
      f[i][0] /= user->Tini;           /* Non-dimensionalize */
    }

    ierr = DMDAVecRestoreArrayDOFRead(dm,X,&x);CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayDOF(dm,F,&f);CHKERRQ(ierr);
  } else {
    ierr = VecZeroEntries(F);CHKERRQ(ierr);
  }
  if (user->diffusion) {
    ierr = FormDiffusionFunction(ts,t,X,F,ptr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode FormRHSJacobian(TS ts,PetscReal t,Vec X,Mat Amat,Mat Pmat,void *ptr)
{
  User              user = (User)ptr;
  PetscErrorCode    ierr;
  const PetscScalar **x;
  PetscInt          M = user->Nspec+1,i,j,xs,xm;
  DM                dm;

  PetscFunctionBeginUser;
  if (user->reactions) {
    ierr = TSGetDM(ts,&dm);CHKERRQ(ierr);
    ierr = MatZeroEntries(Pmat);CHKERRQ(ierr);
    ierr = MatSetOption(Pmat,MAT_ROW_ORIENTED,PETSC_FALSE);CHKERRQ(ierr);
    ierr = MatSetOption(Pmat,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMDAVecGetArrayDOFRead(dm,X,&x);CHKERRQ(ierr);
    ierr = DMDAGetCorners(dm,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

    for (i=xs; i<xs+xm; i++) {
      ierr = PetscArraycpy(user->tchemwork,x[i],user->Nspec+1);CHKERRQ(ierr);
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
  } else {
    ierr = MatZeroEntries(Pmat);CHKERRQ(ierr);
  }
  if (user->diffusion) {
    ierr = FormDiffusionJacobian(ts,t,X,Amat,Pmat,ptr);CHKERRQ(ierr);
  }
  if (Amat != Pmat) {
    ierr = MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode FormInitialSolution(TS ts,Vec X,void *ctx)
{
  PetscScalar    **x,*xc;
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
  ierr = DMDAGetCorners(dm,&xs,NULL,NULL,&xm,NULL,NULL);CHKERRQ(ierr);

  ierr = DMDAGetCoordinateArray(dm,&xc);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(dm,X,&x);CHKERRQ(ierr);
  for (i=xs; i<xs+xm; i++) {
    x[i][0] = 1.0 + .05*PetscSinScalar(2.*PETSC_PI*xc[i]);  /* Non-dimensionalized by user->Tini */
    for (j=0; j<sizeof(initial)/sizeof(initial[0]); j++) {
      int ispec = TC_getSpos(initial[j].name, strlen(initial[j].name));
      if (ispec < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Could not find species %s",initial[j].name);
      ierr = PetscPrintf(PETSC_COMM_SELF,"Species %d: %s %g\n",j,initial[j].name,initial[j].massfrac);CHKERRQ(ierr);
      x[i][1+ispec] = initial[j].massfrac;
    }
  }
  ierr = DMDAVecRestoreArrayDOF(dm,X,&x);CHKERRQ(ierr);
  ierr = DMDARestoreCoordinateArray(dm,&xc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    Routines for displaying the solutions
*/
typedef struct {
  PetscInt cell;
  User     user;
} UserLGCtx;

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

static PetscErrorCode MonitorCellDestroy(UserLGCtx *uctx)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(uctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
   Use TSMonitorLG to monitor the reactions in a particular cell
*/
static PetscErrorCode MonitorCell(TS ts,User user,PetscInt cell)
{
  PetscErrorCode ierr;
  TSMonitorLGCtx ctx;
  char           **snames;
  UserLGCtx      *uctx;
  char           label[128];
  PetscReal      temp,*xc;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  ierr = DMDAGetCoordinateArray(user->dm,&xc);CHKERRQ(ierr);
  temp = 1.0 + .05*PetscSinScalar(2.*PETSC_PI*xc[cell]);  /* Non-dimensionalized by user->Tini */
  ierr = DMDARestoreCoordinateArray(user->dm,&xc);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscSNPrintf(label,sizeof(label),"Initial Temperature %g Cell %d Rank %d",(double)user->Tini*temp,(int)cell,rank);CHKERRQ(ierr);
  ierr = TSMonitorLGCtxCreate(PETSC_COMM_SELF,NULL,label,PETSC_DECIDE,PETSC_DECIDE,600,400,1,&ctx);CHKERRQ(ierr);
  ierr = DMDAGetFieldNames(user->dm,(const char * const **)&snames);CHKERRQ(ierr);
  ierr = TSMonitorLGCtxSetVariableNames(ctx,(const char * const *)snames);CHKERRQ(ierr);
  ierr = PetscNew(&uctx);CHKERRQ(ierr);
  uctx->cell = cell;
  uctx->user = user;
  ierr = TSMonitorLGCtxSetTransform(ctx,(PetscErrorCode (*)(void*,Vec,Vec*))FormMoleFraction,(PetscErrorCode (*)(void*))MonitorCellDestroy,uctx);CHKERRQ(ierr);
  ierr = TSMonitorSet(ts,TSMonitorLGSolution,ctx,(PetscErrorCode (*)(void**))TSMonitorLGCtxDestroy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
