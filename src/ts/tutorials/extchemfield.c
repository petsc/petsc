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
  #include <TC_params.h>
  #include <TC_interface.h>
#else
  #error TChem is required for this example.  Reconfigure PETSc using --download-tchem.
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

        Compute the sensitivies of the solution of the first temperature on the initial conditions
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
  PetscReal Tini, dx;
  PetscReal diffus;
  DM        dm;
  PetscBool diffusion, reactions;
  double   *tchemwork;
  double   *Jdense; /* Dense array workspace where Tchem computes the Jacobian */
  PetscInt *rows;
};

static PetscErrorCode MonitorCell(TS, User, PetscInt);
static PetscErrorCode FormRHSFunction(TS, PetscReal, Vec, Vec, void *);
static PetscErrorCode FormRHSJacobian(TS, PetscReal, Vec, Mat, Mat, void *);
static PetscErrorCode FormInitialSolution(TS, Vec, void *);

#define PetscCallTC(ierr) \
  do { \
    PetscCheck(!ierr, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in TChem library, return code %d", ierr); \
  } while (0)

int main(int argc, char **argv)
{
  TS                ts; /* time integrator */
  TSAdapt           adapt;
  Vec               X; /* solution vector */
  Mat               J; /* Jacobian matrix */
  PetscInt          steps, ncells, xs, xm, i;
  PetscReal         ftime, dt;
  char              chemfile[PETSC_MAX_PATH_LEN] = "chem.inp", thermofile[PETSC_MAX_PATH_LEN] = "therm.dat";
  struct _User      user;
  TSConvergedReason reason;
  PetscBool         showsolutions = PETSC_FALSE;
  char            **snames, *names;
  Vec               lambda; /* used with TSAdjoint for sensitivities */

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscOptionsBegin(PETSC_COMM_WORLD, NULL, "Chemistry solver options", "");
  PetscCall(PetscOptionsString("-chem", "CHEMKIN input file", "", chemfile, chemfile, sizeof(chemfile), NULL));
  PetscCall(PetscOptionsString("-thermo", "NASA thermo input file", "", thermofile, thermofile, sizeof(thermofile), NULL));
  user.pressure = 1.01325e5; /* Pascal */
  PetscCall(PetscOptionsReal("-pressure", "Pressure of reaction [Pa]", "", user.pressure, &user.pressure, NULL));
  user.Tini = 1550;
  PetscCall(PetscOptionsReal("-Tini", "Initial temperature [K]", "", user.Tini, &user.Tini, NULL));
  user.diffus = 100;
  PetscCall(PetscOptionsReal("-diffus", "Diffusion constant", "", user.diffus, &user.diffus, NULL));
  PetscCall(PetscOptionsBool("-draw_solution", "Plot the solution for each cell", "", showsolutions, &showsolutions, NULL));
  user.diffusion = PETSC_TRUE;
  PetscCall(PetscOptionsBool("-diffusion", "Have diffusion", "", user.diffusion, &user.diffusion, NULL));
  user.reactions = PETSC_TRUE;
  PetscCall(PetscOptionsBool("-reactions", "Have reactions", "", user.reactions, &user.reactions, NULL));
  PetscOptionsEnd();

  PetscCallTC(TC_initChem(chemfile, thermofile, 0, 1.0));
  user.Nspec = TC_getNspec();
  user.Nreac = TC_getNreac();

  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_PERIODIC, 10, user.Nspec + 1, 1, NULL, &user.dm));
  PetscCall(DMSetFromOptions(user.dm));
  PetscCall(DMSetUp(user.dm));
  PetscCall(DMDAGetInfo(user.dm, NULL, &ncells, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  user.dx = 1.0 / ncells; /* Set the coordinates of the cell centers; note final ghost cell is at x coordinate 1.0 */
  PetscCall(DMDASetUniformCoordinates(user.dm, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0));

  /* set the names of each field in the DMDA based on the species name */
  PetscCall(PetscMalloc1((user.Nspec + 1) * LENGTHOFSPECNAME, &names));
  PetscCall(PetscStrncpy(names, "Temp", (user.Nspec + 1) * LENGTHOFSPECNAME);
  TC_getSnames(user.Nspec, names + LENGTHOFSPECNAME);
  PetscCall(PetscMalloc1((user.Nspec + 2), &snames));
  for (i = 0; i < user.Nspec + 1; i++) snames[i] = names + i * LENGTHOFSPECNAME;
  snames[user.Nspec + 1] = NULL;
  PetscCall(DMDASetFieldNames(user.dm, (const char *const *)snames));
  PetscCall(PetscFree(snames));
  PetscCall(PetscFree(names));

  PetscCall(DMCreateMatrix(user.dm, &J));
  PetscCall(DMCreateGlobalVector(user.dm, &X));

  PetscCall(PetscMalloc3(user.Nspec + 1, &user.tchemwork, PetscSqr(user.Nspec + 1), &user.Jdense, user.Nspec + 1, &user.rows));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Create timestepping solver context
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetDM(ts, user.dm));
  PetscCall(TSSetType(ts, TSARKIMEX));
  PetscCall(TSARKIMEXSetFullyImplicit(ts, PETSC_TRUE));
  PetscCall(TSARKIMEXSetType(ts, TSARKIMEX4));
  PetscCall(TSSetRHSFunction(ts, NULL, FormRHSFunction, &user));
  PetscCall(TSSetRHSJacobian(ts, J, J, FormRHSJacobian, &user));

  ftime = 1.0;
  PetscCall(TSSetMaxTime(ts, ftime));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_STEPOVER));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set initial conditions
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(FormInitialSolution(ts, X, &user));
  PetscCall(TSSetSolution(ts, X));
  dt = 1e-10; /* Initial time step */
  PetscCall(TSSetTimeStep(ts, dt));
  PetscCall(TSGetAdapt(ts, &adapt));
  PetscCall(TSAdaptSetStepLimits(adapt, 1e-12, 1e-4)); /* Also available with -ts_adapt_dt_min/-ts_adapt_dt_max */
  PetscCall(TSSetMaxSNESFailures(ts, -1));             /* Retry step an unlimited number of times */

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Pass information to graphical monitoring routine
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  if (showsolutions) {
    PetscCall(DMDAGetCorners(user.dm, &xs, NULL, NULL, &xm, NULL, NULL));
    for (i = xs; i < xs + xm; i++) PetscCall(MonitorCell(ts, &user, i));
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set runtime options
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSetFromOptions(ts));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Set final conditions for sensitivities
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(DMCreateGlobalVector(user.dm, &lambda));
  PetscCall(TSSetCostGradients(ts, 1, &lambda, NULL));
  PetscCall(VecSetValue(lambda, 0, 1.0, INSERT_VALUES));
  PetscCall(VecAssemblyBegin(lambda));
  PetscCall(VecAssemblyEnd(lambda));

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Solve ODE
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  PetscCall(TSSolve(ts, X));
  PetscCall(TSGetSolveTime(ts, &ftime));
  PetscCall(TSGetStepNumber(ts, &steps));
  PetscCall(TSGetConvergedReason(ts, &reason));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s at time %g after %" PetscInt_FMT " steps\n", TSConvergedReasons[reason], (double)ftime, steps));

  {
    Vec                max;
    const char *const *names;
    PetscInt           i;
    const PetscReal   *bmax;

    PetscCall(TSMonitorEnvelopeGetBounds(ts, &max, NULL));
    if (max) {
      PetscCall(TSMonitorLGGetVariableNames(ts, &names));
      if (names) {
        PetscCall(VecGetArrayRead(max, &bmax));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "Species - maximum mass fraction\n"));
        for (i = 1; i < user.Nspec; i++) {
          if (bmax[i] > .01) PetscCall(PetscPrintf(PETSC_COMM_SELF, "%s %g\n", names[i], (double)bmax[i]));
        }
        PetscCall(VecRestoreArrayRead(max, &bmax));
      }
    }
  }

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
     Free work space.
   - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  TC_reset();
  PetscCall(DMDestroy(&user.dm));
  PetscCall(MatDestroy(&J));
  PetscCall(VecDestroy(&X));
  PetscCall(VecDestroy(&lambda));
  PetscCall(TSDestroy(&ts));
  PetscCall(PetscFree3(user.tchemwork, user.Jdense, user.rows));
  PetscCall(PetscFinalize());
  return 0;
}

/*
   Applies the second order centered difference diffusion operator on a one dimensional periodic domain
*/
static PetscErrorCode FormDiffusionFunction(TS ts, PetscReal t, Vec X, Vec F, void *ptr)
{
  User                user = (User)ptr;
  PetscScalar       **f;
  const PetscScalar **x;
  DM                  dm;
  PetscInt            i, xs, xm, j, dof;
  Vec                 Xlocal;
  PetscReal           idx;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMDAGetInfo(dm, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &dof, NULL, NULL, NULL, NULL, NULL));
  PetscCall(DMGetLocalVector(dm, &Xlocal));
  PetscCall(DMGlobalToLocalBegin(dm, X, INSERT_VALUES, Xlocal));
  PetscCall(DMGlobalToLocalEnd(dm, X, INSERT_VALUES, Xlocal));
  PetscCall(DMDAVecGetArrayDOFRead(dm, Xlocal, &x));
  PetscCall(DMDAVecGetArrayDOF(dm, F, &f));
  PetscCall(DMDAGetCorners(dm, &xs, NULL, NULL, &xm, NULL, NULL));

  idx = 1.0 * user->diffus / user->dx;
  for (i = xs; i < xs + xm; i++) {
    for (j = 0; j < dof; j++) f[i][j] += idx * (x[i + 1][j] - 2.0 * x[i][j] + x[i - 1][j]);
  }
  PetscCall(DMDAVecRestoreArrayDOFRead(dm, Xlocal, &x));
  PetscCall(DMDAVecRestoreArrayDOF(dm, F, &f));
  PetscCall(DMRestoreLocalVector(dm, &Xlocal));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Produces the second order centered difference diffusion operator on a one dimensional periodic domain
*/
static PetscErrorCode FormDiffusionJacobian(TS ts, PetscReal t, Vec X, Mat Amat, Mat Pmat, void *ptr)
{
  User       user = (User)ptr;
  DM         dm;
  PetscInt   i, xs, xm, j, dof;
  PetscReal  idx, values[3];
  MatStencil row, col[3];

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMDAGetInfo(dm, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &dof, NULL, NULL, NULL, NULL, NULL));
  PetscCall(DMDAGetCorners(dm, &xs, NULL, NULL, &xm, NULL, NULL));

  idx       = 1.0 * user->diffus / user->dx;
  values[0] = idx;
  values[1] = -2.0 * idx;
  values[2] = idx;
  for (i = xs; i < xs + xm; i++) {
    for (j = 0; j < dof; j++) {
      row.i    = i;
      row.c    = j;
      col[0].i = i - 1;
      col[0].c = j;
      col[1].i = i;
      col[1].c = j;
      col[2].i = i + 1;
      col[2].c = j;
      PetscCall(MatSetValuesStencil(Pmat, 1, &row, 3, col, values, ADD_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(Pmat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Pmat, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormRHSFunction(TS ts, PetscReal t, Vec X, Vec F, void *ptr)
{
  User                user = (User)ptr;
  PetscScalar       **f;
  const PetscScalar **x;
  DM                  dm;
  PetscInt            i, xs, xm;

  PetscFunctionBeginUser;
  if (user->reactions) {
    PetscCall(TSGetDM(ts, &dm));
    PetscCall(DMDAVecGetArrayDOFRead(dm, X, &x));
    PetscCall(DMDAVecGetArrayDOF(dm, F, &f));
    PetscCall(DMDAGetCorners(dm, &xs, NULL, NULL, &xm, NULL, NULL));

    for (i = xs; i < xs + xm; i++) {
      PetscCall(PetscArraycpy(user->tchemwork, x[i], user->Nspec + 1));
      user->tchemwork[0] *= user->Tini; /* Dimensionalize */
      PetscCallTC(TC_getSrc(user->tchemwork, user->Nspec + 1, f[i]));
      f[i][0] /= user->Tini; /* Non-dimensionalize */
    }

    PetscCall(DMDAVecRestoreArrayDOFRead(dm, X, &x));
    PetscCall(DMDAVecRestoreArrayDOF(dm, F, &f));
  } else {
    PetscCall(VecZeroEntries(F));
  }
  if (user->diffusion) PetscCall(FormDiffusionFunction(ts, t, X, F, ptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormRHSJacobian(TS ts, PetscReal t, Vec X, Mat Amat, Mat Pmat, void *ptr)
{
  User                user = (User)ptr;
  const PetscScalar **x;
  PetscInt            M = user->Nspec + 1, i, j, xs, xm;
  DM                  dm;

  PetscFunctionBeginUser;
  if (user->reactions) {
    PetscCall(TSGetDM(ts, &dm));
    PetscCall(MatZeroEntries(Pmat));
    PetscCall(MatSetOption(Pmat, MAT_ROW_ORIENTED, PETSC_FALSE));
    PetscCall(MatSetOption(Pmat, MAT_IGNORE_ZERO_ENTRIES, PETSC_TRUE));
    PetscCall(DMDAVecGetArrayDOFRead(dm, X, &x));
    PetscCall(DMDAGetCorners(dm, &xs, NULL, NULL, &xm, NULL, NULL));

    for (i = xs; i < xs + xm; i++) {
      PetscCall(PetscArraycpy(user->tchemwork, x[i], user->Nspec + 1));
      user->tchemwork[0] *= user->Tini; /* Dimensionalize temperature (first row) because that is what Tchem wants */
      PetscCall(TC_getJacTYN(user->tchemwork, user->Nspec, user->Jdense, 1));

      for (j = 0; j < M; j++) user->Jdense[j + 0 * M] /= user->Tini; /* Non-dimensionalize first column */
      for (j = 0; j < M; j++) user->Jdense[0 + j * M] /= user->Tini; /* Non-dimensionalize first row */
      for (j = 0; j < M; j++) user->rows[j] = i * M + j;
      PetscCall(MatSetValues(Pmat, M, user->rows, M, user->rows, user->Jdense, INSERT_VALUES));
    }
    PetscCall(DMDAVecRestoreArrayDOFRead(dm, X, &x));
    PetscCall(MatAssemblyBegin(Pmat, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Pmat, MAT_FINAL_ASSEMBLY));
  } else {
    PetscCall(MatZeroEntries(Pmat));
  }
  if (user->diffusion) PetscCall(FormDiffusionJacobian(ts, t, X, Amat, Pmat, ptr));
  if (Amat != Pmat) {
    PetscCall(MatAssemblyBegin(Amat, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Amat, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormInitialSolution(TS ts, Vec X, void *ctx)
{
  PetscScalar **x, *xc;
  struct {
    const char *name;
    PetscReal   massfrac;
  } initial[] = {
    {"CH4", 0.0948178320887 },
    {"O2",  0.189635664177  },
    {"N2",  0.706766236705  },
    {"AR",  0.00878026702874}
  };
  PetscInt i, j, xs, xm;
  DM       dm;

  PetscFunctionBeginUser;
  PetscCall(VecZeroEntries(X));
  PetscCall(TSGetDM(ts, &dm));
  PetscCall(DMDAGetCorners(dm, &xs, NULL, NULL, &xm, NULL, NULL));

  PetscCall(DMDAGetCoordinateArray(dm, &xc));
  PetscCall(DMDAVecGetArrayDOF(dm, X, &x));
  for (i = xs; i < xs + xm; i++) {
    x[i][0] = 1.0 + .05 * PetscSinScalar(2. * PETSC_PI * xc[i]); /* Non-dimensionalized by user->Tini */
    for (j = 0; j < PETSC_STATIC_ARRAY_LENGTH(initial); j++) {
      int ispec = TC_getSpos(initial[j].name, strlen(initial[j].name));
      PetscCheck(ispec >= 0, PETSC_COMM_SELF, PETSC_ERR_USER, "Could not find species %s", initial[j].name);
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "Species %d: %s %g\n", j, initial[j].name, (double)initial[j].massfrac));
      x[i][1 + ispec] = initial[j].massfrac;
    }
  }
  PetscCall(DMDAVecRestoreArrayDOF(dm, X, &x));
  PetscCall(DMDARestoreCoordinateArray(dm, &xc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
    Routines for displaying the solutions
*/
typedef struct {
  PetscInt cell;
  User     user;
} UserLGCtx;

static PetscErrorCode FormMoleFraction(UserLGCtx *ctx, Vec massf, Vec *molef)
{
  User                user = ctx->user;
  PetscReal          *M, tM = 0;
  PetscInt            i, n  = user->Nspec + 1;
  PetscScalar        *mof;
  const PetscScalar **maf;

  PetscFunctionBeginUser;
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, n, molef));
  PetscCall(PetscMalloc1(user->Nspec, &M));
  TC_getSmass(user->Nspec, M);
  PetscCall(DMDAVecGetArrayDOFRead(user->dm, massf, &maf));
  PetscCall(VecGetArray(*molef, &mof));
  mof[0] = maf[ctx->cell][0]; /* copy over temperature */
  for (i = 1; i < n; i++) tM += maf[ctx->cell][i] / M[i - 1];
  for (i = 1; i < n; i++) mof[i] = maf[ctx->cell][i] / (M[i - 1] * tM);
  PetscCall(DMDAVecRestoreArrayDOFRead(user->dm, massf, &maf));
  PetscCall(VecRestoreArray(*molef, &mof));
  PetscCall(PetscFree(M));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MonitorCellDestroy(UserLGCtx *uctx)
{
  PetscFunctionBeginUser;
  PetscCall(PetscFree(uctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Use TSMonitorLG to monitor the reactions in a particular cell
*/
static PetscErrorCode MonitorCell(TS ts, User user, PetscInt cell)
{
  TSMonitorLGCtx ctx;
  char         **snames;
  UserLGCtx     *uctx;
  char           label[128];
  PetscReal      temp, *xc;
  PetscMPIInt    rank;

  PetscFunctionBeginUser;
  PetscCall(DMDAGetCoordinateArray(user->dm, &xc));
  temp = 1.0 + .05 * PetscSinScalar(2. * PETSC_PI * xc[cell]); /* Non-dimensionalized by user->Tini */
  PetscCall(DMDARestoreCoordinateArray(user->dm, &xc));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(PetscSNPrintf(label, sizeof(label), "Initial Temperature %g Cell %d Rank %d", (double)user->Tini * temp, (int)cell, rank));
  PetscCall(TSMonitorLGCtxCreate(PETSC_COMM_SELF, NULL, label, PETSC_DECIDE, PETSC_DECIDE, 600, 400, 1, &ctx));
  PetscCall(DMDAGetFieldNames(user->dm, (const char *const **)&snames));
  PetscCall(TSMonitorLGCtxSetVariableNames(ctx, (const char *const *)snames));
  PetscCall(PetscNew(&uctx));
  uctx->cell = cell;
  uctx->user = user;
  PetscCall(TSMonitorLGCtxSetTransform(ctx, (PetscErrorCode(*)(void *, Vec, Vec *))FormMoleFraction, (PetscErrorCode(*)(void *))MonitorCellDestroy, uctx));
  PetscCall(TSMonitorSet(ts, TSMonitorLGSolution, ctx, (PetscErrorCode(*)(void **))TSMonitorLGCtxDestroy));
  PetscFunctionReturn(PETSC_SUCCESS);
}
