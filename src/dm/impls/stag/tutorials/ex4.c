static char help[] = "Solves the incompressible, variable-viscosity Stokes equation in 2D or 3D, driven by buoyancy variations.\n"
                     "-dim: set dimension (2 or 3)\n"
                     "-nondimensional: replace dimensional domain and coefficients with nondimensional ones\n"
                     "-isoviscous: use constant viscosity\n"
                     "-levels: number of grids to create, by coarsening\n"
                     "-rediscretize: create operators for all grids and set up a FieldSplit/MG solver\n"
                     "-dump_solution: dump VTK files\n\n";

#include <petscdm.h>
#include <petscksp.h>
#include <petscdmstag.h>
#include <petscdmda.h>

/* Application context - grid-based data*/
typedef struct LevelCtxData_ {
  DM          dm_stokes,dm_coefficients,dm_faces;
  Vec         coeff;
  PetscInt    cells_x,cells_y,cells_z; /* redundant with DMs */
  PetscReal   hx_characteristic,hy_characteristic,hz_characteristic;
  PetscScalar K_bound,K_cont;
} LevelCtxData;
typedef LevelCtxData* LevelCtx;

/* Application context - problem and grid(s) (but not solver-specific data) */
typedef struct CtxData_ {
  MPI_Comm    comm;
  PetscInt    dim;                       /* redundant with DMs */
  PetscInt    cells_x, cells_y, cells_z; /* Redundant with finest DMs */
  PetscReal   xmax,ymax,xmin,ymin,zmin,zmax;
  PetscScalar eta1,eta2,rho1,rho2,gy,eta_characteristic;
  PetscBool   pin_pressure;
  PetscScalar (*GetEta)(struct CtxData_*,PetscScalar,PetscScalar,PetscScalar);
  PetscScalar (*GetRho)(struct CtxData_*,PetscScalar,PetscScalar,PetscScalar);
  PetscInt    n_levels;
  LevelCtx    *levels;
} CtxData;
typedef CtxData* Ctx;

/* Helper to pass system-creation parameters */
typedef struct SystemParameters_ {
  Ctx       ctx;
  PetscInt  level;
  PetscBool include_inverse_visc, faces_only;
} SystemParametersData;
typedef SystemParametersData* SystemParameters;

/* Main logic */
static PetscErrorCode AttachNullspace(DM,Mat);
static PetscErrorCode CreateAuxiliaryOperator(Ctx,PetscInt,Mat*);
static PetscErrorCode CreateSystem(SystemParameters,Mat*,Vec*);
static PetscErrorCode CtxCreateAndSetFromOptions(Ctx*);
static PetscErrorCode CtxDestroy(Ctx*);
static PetscErrorCode DumpSolution(Ctx,PetscInt,Vec);
static PetscErrorCode OperatorInsertInverseViscosityPressureTerms(DM,DM,Vec,PetscScalar,Mat);
static PetscErrorCode PopulateCoefficientData(Ctx,PetscInt);
static PetscErrorCode SystemParametersCreate(SystemParameters*,Ctx);
static PetscErrorCode SystemParametersDestroy(SystemParameters*);

int main(int argc,char **argv)
{
  Ctx            ctx;
  Mat            A,*A_faces = NULL,S_hat = NULL, P = NULL;
  Vec            x,b;
  KSP            ksp;
  DM             dm_stokes,dm_coefficients;
  PetscBool      dump_solution, build_auxiliary_operator, rediscretize, custom_pc_mat;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  /* Accept options for program behavior */
  dump_solution = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-dump_solution",&dump_solution,NULL));
  rediscretize = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-rediscretize",&rediscretize,NULL));
  build_auxiliary_operator = rediscretize;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-build_auxiliary_operator",&build_auxiliary_operator,NULL));
  custom_pc_mat = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-custom_pc_mat",&custom_pc_mat,NULL));

  /* Populate application context */
  PetscCall(CtxCreateAndSetFromOptions(&ctx));

  /* Create two DMStag objects, corresponding to the same domain and parallel
     decomposition ("topology"). Each defines a different set of fields on
     the domain ("section"); the first the solution to the Stokes equations
     (x- and y-velocities and scalar pressure), and the second holds coefficients
     (viscosities on elements and viscosities+densities on corners/edges in 2d/3d) */
  if (ctx->dim == 2) {
  PetscCall(DMStagCreate2d(
      ctx->comm,
      DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
      ctx->cells_x,ctx->cells_y,         /* Global element counts */
      PETSC_DECIDE,PETSC_DECIDE,         /* Determine parallel decomposition automatically */
      0,1,1,                             /* dof: 0 per vertex, 1 per edge, 1 per face/element */
      DMSTAG_STENCIL_BOX,
      1,                                 /* elementwise stencil width */
      NULL,NULL,
      &ctx->levels[ctx->n_levels-1]->dm_stokes));
  } else if (ctx->dim == 3) {
    PetscCall(DMStagCreate3d(
        ctx->comm,
        DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,
        ctx->cells_x,ctx->cells_y,ctx->cells_z,
        PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,
        0,0,1,1,                        /* dof: 1 per face, 1 per element */
        DMSTAG_STENCIL_BOX,
        1,
        NULL,NULL,NULL,
        &ctx->levels[ctx->n_levels-1]->dm_stokes));
  } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported dimension: %" PetscInt_FMT,ctx->dim);
  dm_stokes = ctx->levels[ctx->n_levels-1]->dm_stokes;
  PetscCall(DMSetFromOptions(dm_stokes));
  PetscCall(DMStagGetGlobalSizes(dm_stokes,&ctx->cells_x,&ctx->cells_y, &ctx->cells_z));
  PetscCall(DMSetUp(dm_stokes));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm_stokes,ctx->xmin,ctx->xmax,ctx->ymin,ctx->ymax,ctx->zmin,ctx->zmax));

  if (ctx->dim == 2) PetscCall(DMStagCreateCompatibleDMStag(dm_stokes,2,0,1,0,&ctx->levels[ctx->n_levels-1]->dm_coefficients));
  else if (ctx->dim == 3) PetscCall(DMStagCreateCompatibleDMStag(dm_stokes,0,2,0,1,&ctx->levels[ctx->n_levels-1]->dm_coefficients));
  else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported dimension: %" PetscInt_FMT,ctx->dim);
  dm_coefficients = ctx->levels[ctx->n_levels-1]->dm_coefficients;
  PetscCall(DMSetUp(dm_coefficients));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm_coefficients,ctx->xmin,ctx->xmax,ctx->ymin,ctx->ymax,ctx->zmin,ctx->zmax));

  /* Create additional DMs by coarsening. 0 is the coarsest level */
  for (PetscInt level=ctx->n_levels-1; level>0; --level) {
    PetscCall(DMCoarsen(ctx->levels[level]->dm_stokes,MPI_COMM_NULL,&ctx->levels[level-1]->dm_stokes));
    PetscCall(DMCoarsen(ctx->levels[level]->dm_coefficients,MPI_COMM_NULL,&ctx->levels[level-1]->dm_coefficients));
  }

  /* Compute scaling constants, knowing grid spacing */
  ctx->eta_characteristic = PetscMin(PetscRealPart(ctx->eta1),PetscRealPart(ctx->eta2));
  for (PetscInt level=0; level<ctx->n_levels; ++level) {
    PetscInt  N[3];
    PetscReal hx_avg_inv;

    PetscCall(DMStagGetGlobalSizes(ctx->levels[level]->dm_stokes,&N[0],&N[1],&N[2]));
    ctx->levels[level]->hx_characteristic = (ctx->xmax-ctx->xmin)/N[0];
    ctx->levels[level]->hy_characteristic = (ctx->ymax-ctx->ymin)/N[1];
    ctx->levels[level]->hz_characteristic = (ctx->zmax-ctx->zmin)/N[2];
    if (ctx->dim == 2) {
      hx_avg_inv = 2.0/(ctx->levels[level]->hx_characteristic + ctx->levels[level]->hy_characteristic);
    } else if (ctx->dim == 3) {
      hx_avg_inv = 3.0/(ctx->levels[level]->hx_characteristic + ctx->levels[level]->hy_characteristic + ctx->levels[level]->hz_characteristic);
    } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not Implemented for dimension %" PetscInt_FMT,ctx->dim);
    ctx->levels[level]->K_cont = ctx->eta_characteristic*hx_avg_inv;
    ctx->levels[level]->K_bound = ctx->eta_characteristic*hx_avg_inv*hx_avg_inv;
  }

  /* Populate coefficient data */
  for (PetscInt level=0; level<ctx->n_levels; ++level) PetscCall(PopulateCoefficientData(ctx,level));

  /* Construct main system */
  {
    SystemParameters system_parameters;

    PetscCall(SystemParametersCreate(&system_parameters,ctx));
    PetscCall(CreateSystem(system_parameters,&A,&b));
    PetscCall(SystemParametersDestroy(&system_parameters));
  }

  /* Attach a constant-pressure nullspace to the fine-level operator */
  if (!ctx->pin_pressure) PetscCall(AttachNullspace(dm_stokes,A));

  /* Set up solver */
  PetscCall(KSPCreate(ctx->comm,&ksp));
  PetscCall(KSPSetType(ksp,KSPFGMRES));
  {
    /* Default to a direct solver, if a package is available */
    PetscMPIInt size;

    PetscCall(MPI_Comm_size(ctx->comm,&size));
    if (PetscDefined(HAVE_SUITESPARSE) && size == 1) {
      PC pc;

      PetscCall(KSPGetPC(ksp,&pc));
      PetscCall(PCSetType(pc,PCLU));
      PetscCall(PCFactorSetMatSolverType(pc,MATSOLVERUMFPACK)); /* A default, requires SuiteSparse */
    }
    if (PetscDefined(HAVE_MUMPS) && size > 1) {
      PC pc;

      PetscCall(KSPGetPC(ksp,&pc));
      PetscCall(PCSetType(pc,PCLU));
      PetscCall(PCFactorSetMatSolverType(pc,MATSOLVERMUMPS)); /* A default, requires MUMPS */
    }
  }

  /* Create and set a custom preconditioning matrix */
  if (custom_pc_mat) {
      SystemParameters system_parameters;

      PetscCall(SystemParametersCreate(&system_parameters,ctx));
      system_parameters->include_inverse_visc = PETSC_TRUE;
      PetscCall(CreateSystem(system_parameters,&P,NULL));
      PetscCall(SystemParametersDestroy(&system_parameters));
    PetscCall(KSPSetOperators(ksp,A,P));
  } else {
    PetscCall(KSPSetOperators(ksp,A,A));
  }

  PetscCall(KSPSetDM(ksp,dm_stokes));
  PetscCall(KSPSetDMActive(ksp,PETSC_FALSE));

  /* Finish setting up solver (can override options set above) */
  PetscCall(KSPSetFromOptions(ksp));

  /* Additional solver configuration that can involve setting up and CANNOT be
     overridden from the command line */

  /* Construct an auxiliary operator for use a Schur complement preconditioner,
     and tell PCFieldSplit to use it (which has no effect if not using that PC) */
  if (build_auxiliary_operator && !rediscretize) {
    PC pc;

    PetscCall(KSPGetPC(ksp,&pc));
    PetscCall(CreateAuxiliaryOperator(ctx,ctx->n_levels-1,&S_hat));
    PetscCall(PCFieldSplitSetSchurPre(pc,PC_FIELDSPLIT_SCHUR_PRE_USER,S_hat));
    PetscCall(KSPSetFromOptions(ksp));
  }

  if (rediscretize) {
    /* Set up an ABF solver with rediscretized geometric multigrid on the velocity-velocity block */
    PC  pc,pc_faces;
    KSP ksp_faces;

    if (ctx->n_levels < 2) {
      PetscCall(PetscPrintf(ctx->comm,"Warning: not using multiple levels!\n"));
    }

    PetscCall(KSPGetPC(ksp,&pc));
    PetscCall(PCSetType(pc,PCFIELDSPLIT));
    PetscCall(PCFieldSplitSetType(pc,PC_COMPOSITE_SCHUR));
    PetscCall(PCFieldSplitSetSchurFactType(pc,PC_FIELDSPLIT_SCHUR_FACT_UPPER));
    if (build_auxiliary_operator) {
      PetscCall(CreateAuxiliaryOperator(ctx,ctx->n_levels-1,&S_hat));
      PetscCall(PCFieldSplitSetSchurPre(pc,PC_FIELDSPLIT_SCHUR_PRE_USER,S_hat));
    }

    /* Create rediscretized velocity-only DMs and operators */
    PetscCall(PetscMalloc1(ctx->n_levels,&A_faces));
    for (PetscInt level=0; level<ctx->n_levels; ++level) {
      if (ctx->dim == 2) {
        PetscCall(DMStagCreateCompatibleDMStag(ctx->levels[level]->dm_stokes,0,1,0,0,&ctx->levels[level]->dm_faces));
      } else if (ctx->dim == 3) {
        PetscCall(DMStagCreateCompatibleDMStag(ctx->levels[level]->dm_stokes,0,0,1,0,&ctx->levels[level]->dm_faces));
      } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not Implemented for dimension %" PetscInt_FMT,ctx->dim);
      {
        SystemParameters system_parameters;

        PetscCall(SystemParametersCreate(&system_parameters,ctx));
        system_parameters->faces_only = PETSC_TRUE;
        system_parameters->level = level;
        PetscCall(CreateSystem(system_parameters,&A_faces[level],NULL));
        PetscCall(SystemParametersDestroy(&system_parameters));
      }
    }

    /* Set up to populate enough to define the sub-solver */
    PetscCall(KSPSetUp(ksp));

    /* Set multigrid components and settings */
    {
      KSP *sub_ksp;

      PetscCall(PCFieldSplitSchurGetSubKSP(pc,NULL,&sub_ksp));
      ksp_faces = sub_ksp[0];
      PetscCall(PetscFree(sub_ksp));
    }
    PetscCall(KSPSetType(ksp_faces,KSPGCR));
    PetscCall(KSPGetPC(ksp_faces,&pc_faces));
    PetscCall(PCSetType(pc_faces,PCMG));
    PetscCall(PCMGSetLevels(pc_faces,ctx->n_levels,NULL));
    for (PetscInt level=0; level<ctx->n_levels; ++level) {
      KSP ksp_level;
      PC  pc_level;

      /* Smoothers */
      PetscCall(PCMGGetSmoother(pc_faces,level,&ksp_level));
      PetscCall(KSPGetPC(ksp_level,&pc_level));
      PetscCall(KSPSetOperators(ksp_level,A_faces[level],A_faces[level]));
      if (level > 0) {
        PetscCall(PCSetType(pc_level,PCJACOBI));
      }

      /* Transfer Operators */
      if (level > 0) {
        Mat restriction,interpolation;
        DM  dm_level=ctx->levels[level]->dm_faces;
        DM  dm_coarser=ctx->levels[level-1]->dm_faces;

        PetscCall(DMCreateInterpolation(dm_coarser,dm_level,&interpolation,NULL));
        PetscCall(PCMGSetInterpolation(pc_faces,level,interpolation));
        PetscCall(MatDestroy(&interpolation));
        PetscCall(DMCreateRestriction(dm_coarser,dm_level,&restriction));
        PetscCall(PCMGSetRestriction(pc_faces,level,restriction));
        PetscCall(MatDestroy(&restriction));
      }
    }
  }

  /* Solve */
  PetscCall(VecDuplicate(b,&x));
  PetscCall(KSPSolve(ksp,b,x));
  {
    KSPConvergedReason reason;
    PetscCall(KSPGetConvergedReason(ksp,&reason));
    PetscCheck(reason >= 0,PETSC_COMM_WORLD,PETSC_ERR_CONV_FAILED,"Linear solve failed");
  }

  /* Dump solution by converting to DMDAs and dumping */
  if (dump_solution) PetscCall(DumpSolution(ctx,ctx->n_levels-1,x));

  /* Destroy PETSc objects and finalize */
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFree(A));
  if (A_faces) {
    for (PetscInt level=0; level<ctx->n_levels; ++level) {
      if (A_faces[level]) {
        PetscCall(MatDestroy(&A_faces[level]));
      }
    }
    PetscCall(PetscFree(A_faces));
  }
  if (P) {
    PetscCall(MatDestroy(&P));
  }
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&S_hat));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(CtxDestroy(&ctx));
  PetscCall(PetscFinalize());
  return 0;
}

static PetscScalar GetEta_constant(Ctx ctx, PetscScalar x, PetscScalar y, PetscScalar z)
{
  (void) ctx;
  (void) x;
  (void) y;
  (void) z;
  return 1.0;
}

static PetscScalar GetRho_layers(Ctx ctx, PetscScalar x, PetscScalar y, PetscScalar z)
{
  (void) y;
  (void) z;
  return PetscRealPart(x) < (ctx->xmax - ctx->xmin) / 2.0 ? ctx->rho1 : ctx->rho2;
}

static PetscScalar GetEta_layers(Ctx ctx, PetscScalar x, PetscScalar y, PetscScalar z)
{
  (void) y;
  (void) z;
  return PetscRealPart(x) < (ctx->xmax - ctx->xmin) / 2.0 ? ctx->eta1 : ctx->eta2;
}

static PetscScalar GetRho_sinker_box2(Ctx ctx, PetscScalar x, PetscScalar y, PetscScalar z) {
  (void) z;
  const PetscReal d = ctx->xmax - ctx->xmin;
  const PetscReal xx = PetscRealPart(x)/d - 0.5;
  const PetscReal yy = PetscRealPart(y)/d - 0.5;
  return (xx*xx > 0.15*0.15 || yy*yy > 0.15*0.15) ? ctx->rho1 : ctx->rho2;
}

static PetscScalar GetEta_sinker_box2(Ctx ctx, PetscScalar x, PetscScalar y, PetscScalar z) {
  (void) z;
  const PetscReal d = ctx->xmax - ctx->xmin;
  const PetscReal xx = PetscRealPart(x)/d - 0.5;
  const PetscReal yy = PetscRealPart(y)/d - 0.5;
  return (xx*xx > 0.15*0.15 || yy*yy > 0.15*0.15) ? ctx->eta1 : ctx->eta2;
}

static PetscScalar GetRho_sinker_box3(Ctx ctx, PetscScalar x, PetscScalar y, PetscScalar z) {
  const PetscReal d = ctx->xmax - ctx->xmin;
  const PetscReal xx = PetscRealPart(x)/d - 0.5;
  const PetscReal yy = PetscRealPart(y)/d - 0.5;
  const PetscReal zz = PetscRealPart(z)/d - 0.5;
  const PetscReal half_width =  0.15;
  return (PetscAbsReal(xx) > half_width || PetscAbsReal(yy) > half_width || PetscAbsReal(zz) > half_width) ? ctx->rho1 : ctx->rho2;
}

static PetscScalar GetEta_sinker_box3(Ctx ctx, PetscScalar x, PetscScalar y, PetscScalar z) {
  const PetscReal d = ctx->xmax - ctx->xmin;
  const PetscReal xx = PetscRealPart(x)/d - 0.5;
  const PetscReal yy = PetscRealPart(y)/d - 0.5;
  const PetscReal zz = PetscRealPart(z)/d - 0.5;
  const PetscReal half_width = 0.15;
  return (PetscAbsReal(xx) > half_width || PetscAbsReal(yy) > half_width || PetscAbsReal(zz) > half_width) ? ctx->eta1 : ctx->eta2;
}

static PetscScalar GetRho_sinker_sphere3(Ctx ctx, PetscScalar x, PetscScalar y, PetscScalar z) {
  const PetscReal d = ctx->xmax - ctx->xmin;
  const PetscReal xx = PetscRealPart(x)/d - 0.5;
  const PetscReal yy = PetscRealPart(y)/d - 0.5;
  const PetscReal zz = PetscRealPart(z)/d - 0.5;
  const PetscReal half_width =  0.3;
  return (xx*xx + yy*yy + zz*zz > half_width*half_width) ? ctx->rho1 : ctx->rho2;
}

static PetscScalar GetEta_sinker_sphere3(Ctx ctx, PetscScalar x, PetscScalar y, PetscScalar z) {
  const PetscReal d = ctx->xmax - ctx->xmin;
  const PetscReal xx = PetscRealPart(x)/d - 0.5;
  const PetscReal yy = PetscRealPart(y)/d - 0.5;
  const PetscReal zz = PetscRealPart(z)/d - 0.5;
  const PetscReal half_width = 0.3;
  return (xx*xx + yy*yy + zz*zz > half_width*half_width) ? ctx->eta1 : ctx->eta2;
}

static PetscScalar GetEta_blob3(Ctx ctx, PetscScalar x, PetscScalar y, PetscScalar z) {
  const PetscReal d = ctx->xmax - ctx->xmin;
  const PetscReal xx = PetscRealPart(x)/d - 0.5;
  const PetscReal yy = PetscRealPart(y)/d - 0.5;
  const PetscReal zz = PetscRealPart(z)/d - 0.5;
  return ctx->eta1 + ctx->eta2 * PetscExpScalar(-20.0 * (xx*xx + yy*yy + zz*zz));
}

static PetscScalar GetRho_blob3(Ctx ctx, PetscScalar x, PetscScalar y, PetscScalar z) {
  const PetscReal d = ctx->xmax - ctx->xmin;
  const PetscReal xx = PetscRealPart(x)/d - 0.5;
  const PetscReal yy = PetscRealPart(y)/d - 0.5;
  const PetscReal zz = PetscRealPart(z)/d - 0.5;
  return ctx->rho1 + ctx->rho2 * PetscExpScalar(-20.0 * (xx*xx + yy*yy + zz*zz));
}

static PetscErrorCode LevelCtxCreate(LevelCtx *p_level_ctx)
{
  LevelCtx       level_ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc1(1,p_level_ctx));
  level_ctx = *p_level_ctx;
  level_ctx->dm_stokes = NULL;
  level_ctx->dm_coefficients = NULL;
  level_ctx->dm_faces = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode LevelCtxDestroy(LevelCtx *p_level_ctx)
{
  LevelCtx       level_ctx;

  PetscFunctionBeginUser;
  level_ctx = *p_level_ctx;
  if (level_ctx->dm_stokes) {
    PetscCall(DMDestroy(&level_ctx->dm_stokes));
  }
  if (level_ctx->dm_coefficients) {
    PetscCall(DMDestroy(&level_ctx->dm_coefficients));
  }
  if (level_ctx->dm_faces) {
    PetscCall(DMDestroy(&level_ctx->dm_faces));
  }
  if (level_ctx->coeff) {
    PetscCall(VecDestroy(&level_ctx->coeff));
  }
  PetscCall(PetscFree(*p_level_ctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode CtxCreateAndSetFromOptions(Ctx *p_ctx)
{
  Ctx            ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc1(1,p_ctx));
  ctx = *p_ctx;

  ctx->comm = PETSC_COMM_WORLD;
  ctx->pin_pressure = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-pin_pressure",&ctx->pin_pressure,NULL));
  ctx->dim = 3;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-dim",&ctx->dim,NULL));
  if (ctx->dim <= 2) {
    ctx->cells_x = 32;
  } else {
    ctx->cells_x = 16;
  }
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-s",&ctx->cells_x,NULL));  /* shortcut. Usually, use -stag_grid_x etc. */
  ctx->cells_z = ctx->cells_y = ctx->cells_x;
  ctx->xmin = ctx->ymin = ctx->zmin = 0.0;
  {
    PetscBool nondimensional = PETSC_TRUE;

    PetscCall(PetscOptionsGetBool(NULL,NULL,"-nondimensional",&nondimensional,NULL));
    if (nondimensional) {
      ctx->xmax = ctx->ymax = ctx->zmax = 1.0;
      ctx->rho1 = 0.0;
      ctx->rho2 = 1.0;
      ctx->eta1 = 1.0;
      ctx->eta2 = 1e2;
      ctx->gy   = -1.0; /* downwards */
    } else {
      ctx->xmax = 1e6;
      ctx->ymax = 1.5e6;
      ctx->zmax = 1e6;
      ctx->rho1 = 3200;
      ctx->rho2 = 3300;
      ctx->eta1 = 1e20;
      ctx->eta2 = 1e22;
      ctx->gy   = -10.0; /* downwards */
    }
  }
  {
    PetscBool isoviscous;

    isoviscous = PETSC_FALSE;
    PetscCall(PetscOptionsGetScalar(NULL,NULL,"-eta1",&ctx->eta1,NULL));
    PetscCall(PetscOptionsGetBool(NULL,NULL,"-isoviscous",&isoviscous,NULL));
    if (isoviscous) {
      ctx->eta2 = ctx->eta1;
      ctx->GetEta = GetEta_constant; /* override */
    } else {
      PetscCall(PetscOptionsGetScalar(NULL,NULL,"-eta2",&ctx->eta2,NULL));
    }
  }
  {
    char      mode[1024] = "sinker";
    PetscBool is_layers,is_blob,is_sinker_box,is_sinker_sphere;

    PetscCall(PetscOptionsGetString(NULL,NULL,"-coefficients",mode,sizeof(mode),NULL));
    PetscCall(PetscStrncmp(mode,"layers",sizeof(mode),&is_layers));
    PetscCall(PetscStrncmp(mode,"sinker",sizeof(mode),&is_sinker_box));
    if (!is_sinker_box) {
      PetscCall(PetscStrncmp(mode,"sinker_box",sizeof(mode),&is_sinker_box));
    }
    PetscCall(PetscStrncmp(mode,"sinker_sphere",sizeof(mode),&is_sinker_sphere));
    PetscCall(PetscStrncmp(mode,"blob",sizeof(mode),&is_blob));

    if (is_layers) {
        ctx->GetRho = GetRho_layers;
        ctx->GetEta = GetEta_layers;
    }
    if (is_blob) {
        if (ctx->dim == 3) {
          ctx->GetRho = GetRho_blob3;
          ctx->GetEta = GetEta_blob3;
        } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not implemented for dimension %" PetscInt_FMT,ctx->dim);
    }
    if (is_sinker_box) {
        if (ctx->dim == 2) {
          ctx->GetRho = GetRho_sinker_box2;
          ctx->GetEta = GetEta_sinker_box2;
        } else if (ctx->dim == 3) {
          ctx->GetRho = GetRho_sinker_box3;
          ctx->GetEta = GetEta_sinker_box3;
        } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not implemented for dimension %" PetscInt_FMT,ctx->dim);
      }
    if (is_sinker_sphere) {
        if (ctx->dim == 3) {
          ctx->GetRho = GetRho_sinker_sphere3;
          ctx->GetEta = GetEta_sinker_sphere3;
        } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not implemented for dimension %" PetscInt_FMT,ctx->dim);
      }
  }

  /* Per-level data */
  ctx->n_levels = 1;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-levels",&ctx->n_levels,NULL));
  PetscCall(PetscMalloc1(ctx->n_levels,&ctx->levels));
  for (PetscInt i=0; i<ctx->n_levels; ++i) {
    PetscCall(LevelCtxCreate(&ctx->levels[i]));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode CtxDestroy(Ctx *p_ctx)
{
  Ctx            ctx;

  PetscFunctionBeginUser;
  ctx = *p_ctx;
  for (PetscInt i=0; i<ctx->n_levels; ++i) {
    PetscCall(LevelCtxDestroy(&ctx->levels[i]));
  }
  PetscCall(PetscFree(ctx->levels));
  PetscCall(PetscFree(*p_ctx));
  PetscFunctionReturn(0);
}

static PetscErrorCode SystemParametersCreate(SystemParameters* parameters,Ctx ctx)
{
  PetscFunctionBeginUser;
  PetscCall(PetscMalloc1(1,parameters));
  (*parameters)->ctx = ctx;
  (*parameters)->level = ctx->n_levels-1;
  (*parameters)->include_inverse_visc = PETSC_FALSE;
  (*parameters)->faces_only = PETSC_FALSE;
  PetscFunctionReturn(0);
}

static PetscErrorCode SystemParametersDestroy(SystemParameters* parameters)
{
  PetscFunctionBeginUser;
  PetscCall(PetscFree(*parameters));
  *parameters = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateSystem2d(SystemParameters parameters, Mat *pA,Vec *pRhs)
{
  PetscInt       N[2];
  PetscInt       ex,ey,startx,starty,nx,ny;
  Mat            A;
  Vec            rhs;
  PetscReal      hx,hy,dv;
  Vec            coefficients_local;
  PetscBool      build_rhs;
  DM             dm_main, dm_coefficients;
  PetscScalar    K_cont, K_bound;
  Ctx            ctx = parameters->ctx;
  PetscInt       level = parameters->level;

  PetscFunctionBeginUser;
  if (parameters->faces_only) {
    dm_main = ctx->levels[level]->dm_faces;
  } else {
    dm_main = ctx->levels[level]->dm_stokes;
  }
  dm_coefficients = ctx->levels[level]->dm_coefficients;
  K_cont = ctx->levels[level]->K_cont;
  K_bound = ctx->levels[level]->K_bound;
  PetscCall(DMCreateMatrix(dm_main,pA));
  A = *pA;
  build_rhs = (PetscBool)(pRhs != NULL);
  PetscCheck(!(parameters->faces_only && build_rhs),PetscObjectComm((PetscObject)dm_main),PETSC_ERR_SUP,"RHS for faces-only not supported");
  if (build_rhs) {
    PetscCall(DMCreateGlobalVector(dm_main,pRhs));
    rhs = *pRhs;
  } else {
    rhs = NULL;
  }
  PetscCall(DMStagGetCorners(dm_main,&startx,&starty,NULL,&nx,&ny,NULL,NULL,NULL,NULL));
  PetscCall(DMStagGetGlobalSizes(dm_main,&N[0],&N[1],NULL));
  hx = ctx->levels[level]->hx_characteristic;
  hy = ctx->levels[level]->hy_characteristic;
  dv = hx*hy;
  PetscCall(DMGetLocalVector(dm_coefficients,&coefficients_local));
  PetscCall(DMGlobalToLocal(dm_coefficients,ctx->levels[level]->coeff,INSERT_VALUES,coefficients_local));

  /* Loop over all local elements */
  for (ey = starty; ey<starty+ny; ++ey) { /* With DMStag, always iterate x fastest, y second fastest, z slowest */
    for (ex = startx; ex<startx+nx; ++ex) {
      const PetscBool left_boundary   = (PetscBool) (ex == 0);
      const PetscBool right_boundary  = (PetscBool) (ex == N[0]-1);
      const PetscBool bottom_boundary = (PetscBool) (ey == 0);
      const PetscBool top_boundary    = (PetscBool) (ey == N[1]-1);

      if (ey == N[1]-1) {
        /* Top boundary velocity Dirichlet */
        DMStagStencil     row;
        const PetscScalar val_rhs = 0.0;
        const PetscScalar val_A = K_bound;

        row.i = ex; row.j = ey; row.loc = DMSTAG_UP; row.c = 0;
        PetscCall(DMStagMatSetValuesStencil(dm_main,A,1,&row,1,&row,&val_A,INSERT_VALUES));
        if (build_rhs) {
          PetscCall(DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES));
        }
      }

      if (ey == 0) {
        /* Bottom boundary velocity Dirichlet */
        DMStagStencil     row;
        const PetscScalar val_rhs = 0.0;
        const PetscScalar val_A = K_bound;

        row.i = ex; row.j = ey; row.loc = DMSTAG_DOWN; row.c = 0;
        PetscCall(DMStagMatSetValuesStencil(dm_main,A,1,&row,1,&row,&val_A,INSERT_VALUES));
        if (build_rhs) {
          PetscCall(DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES));
        }
      } else {
        /* Y-momentum equation : (u_xx + u_yy) - p_y = f^y
           includes non-zero forcing and free-slip boundary conditions */
        PetscInt      count;
        DMStagStencil row,col[11];
        PetscScalar   val_A[11];
        DMStagStencil rhoPoint[2];
        PetscScalar   rho[2],val_rhs;
        DMStagStencil etaPoint[4];
        PetscScalar   eta[4],eta_left,eta_right,eta_up,eta_down;

        row.i = ex; row.j = ey; row.loc = DMSTAG_DOWN; row.c = 0;

        /* get rho values  and compute rhs value*/
        rhoPoint[0].i = ex; rhoPoint[0].j = ey; rhoPoint[0].loc = DMSTAG_DOWN_LEFT;  rhoPoint[0].c = 1;
        rhoPoint[1].i = ex; rhoPoint[1].j = ey; rhoPoint[1].loc = DMSTAG_DOWN_RIGHT; rhoPoint[1].c = 1;
        PetscCall(DMStagVecGetValuesStencil(dm_coefficients,coefficients_local,2,rhoPoint,rho));
        val_rhs = -ctx->gy * dv * 0.5 * (rho[0] + rho[1]);

        /* Get eta values */
        etaPoint[0].i = ex; etaPoint[0].j = ey;   etaPoint[0].loc = DMSTAG_DOWN_LEFT;  etaPoint[0].c = 0; /* Left  */
        etaPoint[1].i = ex; etaPoint[1].j = ey;   etaPoint[1].loc = DMSTAG_DOWN_RIGHT; etaPoint[1].c = 0; /* Right */
        etaPoint[2].i = ex; etaPoint[2].j = ey;   etaPoint[2].loc = DMSTAG_ELEMENT;    etaPoint[2].c = 0; /* Up    */
        etaPoint[3].i = ex; etaPoint[3].j = ey-1; etaPoint[3].loc = DMSTAG_ELEMENT;    etaPoint[3].c = 0; /* Down  */
        PetscCall(DMStagVecGetValuesStencil(dm_coefficients,coefficients_local,4,etaPoint,eta));
        eta_left = eta[0]; eta_right = eta[1]; eta_up = eta[2]; eta_down = eta[3];

        count = 0;

        col[count] = row;
        val_A[count] = -2.0 * dv * (eta_down + eta_up) / (hy*hy);
        if (!left_boundary)  val_A[count] += -1.0 * dv * eta_left  / (hx*hx);
        if (!right_boundary) val_A[count] += -1.0 * dv * eta_right / (hx*hx);
        ++count;

        col[count].i   = ex;   col[count].j = ey-1; col[count].loc = DMSTAG_DOWN;    col[count].c  = 0; val_A[count] =  2.0 * dv * eta_down  / (hy*hy); ++count;
        col[count].i   = ex;   col[count].j = ey+1; col[count].loc = DMSTAG_DOWN;    col[count].c  = 0; val_A[count] =  2.0 * dv * eta_up    / (hy*hy); ++count;
        if (!left_boundary) {
          col[count].i = ex-1; col[count].j = ey;   col[count].loc = DMSTAG_DOWN;    col[count].c  = 0; val_A[count] =        dv * eta_left  / (hx*hx); ++count;
        }
        if (!right_boundary) {
          col[count].i = ex+1; col[count].j = ey;   col[count].loc = DMSTAG_DOWN;    col[count].c  = 0; val_A[count] =        dv * eta_right / (hx*hx); ++count;
        }
        col[count].i   = ex;   col[count].j = ey-1; col[count].loc = DMSTAG_LEFT;    col[count].c  = 0; val_A[count] =        dv * eta_left  / (hx*hy); ++count; /* down left x edge */
        col[count].i   = ex;   col[count].j = ey-1; col[count].loc = DMSTAG_RIGHT;   col[count].c  = 0; val_A[count] = -1.0 * dv * eta_right / (hx*hy); ++count; /* down right x edge */
        col[count].i   = ex;   col[count].j = ey;   col[count].loc = DMSTAG_LEFT;    col[count].c  = 0; val_A[count] = -1.0 * dv * eta_left  / (hx*hy); ++count; /* up left x edge */
        col[count].i   = ex;   col[count].j = ey;   col[count].loc = DMSTAG_RIGHT;   col[count].c  = 0; val_A[count] =        dv * eta_right / (hx*hy); ++count; /* up right x edge */
        if (!parameters->faces_only) {
          col[count].i = ex;   col[count].j = ey-1; col[count].loc = DMSTAG_ELEMENT; col[count].c  = 0; val_A[count] =        K_cont * dv / hy;         ++count;
          col[count].i = ex;   col[count].j = ey;   col[count].loc = DMSTAG_ELEMENT; col[count].c  = 0; val_A[count] = -1.0 * K_cont * dv / hy;         ++count;
        }

        PetscCall(DMStagMatSetValuesStencil(dm_main,A,1,&row,count,col,val_A,INSERT_VALUES));
        if (build_rhs) {
          PetscCall(DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES));
        }
      }

      if (ex == N[0]-1) {
        /* Right Boundary velocity Dirichlet */
        /* Redundant in the corner */
        DMStagStencil     row;
        const PetscScalar val_rhs = 0.0;
        const PetscScalar val_A = K_bound;

        row.i = ex; row.j = ey; row.loc = DMSTAG_RIGHT; row.c = 0;
        PetscCall(DMStagMatSetValuesStencil(dm_main,A,1,&row,1,&row,&val_A,INSERT_VALUES));
        if (build_rhs) {
          PetscCall(DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES));
        }
      }
      if (ex == 0) {
        /* Left velocity Dirichlet */
        DMStagStencil row;
        const PetscScalar val_rhs = 0.0;
        const PetscScalar val_A = K_bound;

        row.i = ex; row.j = ey; row.loc = DMSTAG_LEFT; row.c = 0;
        PetscCall(DMStagMatSetValuesStencil(dm_main,A,1,&row,1,&row,&val_A,INSERT_VALUES));
        if (build_rhs) {
          PetscCall(DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES));
        }
      } else {
        /* X-momentum equation : (u_xx + u_yy) - p_x = f^x
          zero RHS, including free-slip boundary conditions */
        PetscInt          count;
        DMStagStencil     row,col[11];
        PetscScalar       val_A[11];
        DMStagStencil     etaPoint[4];
        PetscScalar       eta[4],eta_left,eta_right,eta_up,eta_down;
        const PetscScalar val_rhs = 0.0;

        row.i = ex; row.j = ey; row.loc = DMSTAG_LEFT; row.c = 0;

        /* Get eta values */
        etaPoint[0].i = ex-1; etaPoint[0].j = ey; etaPoint[0].loc = DMSTAG_ELEMENT;   etaPoint[0].c = 0; /* Left  */
        etaPoint[1].i = ex;   etaPoint[1].j = ey; etaPoint[1].loc = DMSTAG_ELEMENT;   etaPoint[1].c = 0; /* Right */
        etaPoint[2].i = ex;   etaPoint[2].j = ey; etaPoint[2].loc = DMSTAG_UP_LEFT;   etaPoint[2].c = 0; /* Up    */
        etaPoint[3].i = ex;   etaPoint[3].j = ey; etaPoint[3].loc = DMSTAG_DOWN_LEFT; etaPoint[3].c = 0; /* Down  */
        PetscCall(DMStagVecGetValuesStencil(dm_coefficients,coefficients_local,4,etaPoint,eta));
        eta_left = eta[0]; eta_right = eta[1]; eta_up = eta[2]; eta_down = eta[3];

        count = 0;
        col[count] = row;
        val_A[count] = -2.0 * dv * (eta_left + eta_right) / (hx*hx);
        if (!bottom_boundary) val_A[count] += -1.0 * dv * eta_down / (hy*hy);
        if (!top_boundary)    val_A[count] += -1.0 * dv * eta_up   / (hy*hy);
        ++count;

        if (!bottom_boundary) {
          col[count].i  = ex;   col[count].j = ey-1; col[count].loc = DMSTAG_LEFT;    col[count].c  = 0; val_A[count] =        dv * eta_down  / (hy*hy); ++count;
        }
        if (!top_boundary) {
          col[count].i  = ex;   col[count].j = ey+1; col[count].loc = DMSTAG_LEFT;    col[count].c  = 0; val_A[count] =        dv * eta_up    / (hy*hy); ++count;
        }
        col[count].i    = ex-1; col[count].j = ey;   col[count].loc = DMSTAG_LEFT;    col[count].c  = 0; val_A[count] =  2.0 * dv * eta_left  / (hx*hx); ++count;
        col[count].i    = ex+1; col[count].j = ey;   col[count].loc = DMSTAG_LEFT;    col[count].c  = 0; val_A[count] =  2.0 * dv * eta_right / (hx*hx); ++count;
        col[count].i    = ex-1; col[count].j = ey;   col[count].loc = DMSTAG_DOWN;    col[count].c  = 0; val_A[count] =        dv * eta_down  / (hx*hy); ++count; /* down left */
        col[count].i    = ex;   col[count].j = ey;   col[count].loc = DMSTAG_DOWN;    col[count].c  = 0; val_A[count] = -1.0 * dv * eta_down  / (hx*hy); ++count; /* down right */
        col[count].i    = ex-1; col[count].j = ey;   col[count].loc = DMSTAG_UP;      col[count].c  = 0; val_A[count] = -1.0 * dv * eta_up    / (hx*hy); ++count; /* up left */
        col[count].i    = ex;   col[count].j = ey;   col[count].loc = DMSTAG_UP;      col[count].c  = 0; val_A[count] =        dv * eta_up    / (hx*hy); ++count; /* up right */
        if (!parameters->faces_only) {
          col[count].i  = ex-1; col[count].j = ey;   col[count].loc = DMSTAG_ELEMENT; col[count].c  = 0; val_A[count] =        K_cont * dv / hx;         ++count;
          col[count].i  = ex;   col[count].j = ey;   col[count].loc = DMSTAG_ELEMENT; col[count].c  = 0; val_A[count] = -1.0 * K_cont * dv / hx;         ++count;
        }

        PetscCall(DMStagMatSetValuesStencil(dm_main,A,1,&row,count,col,val_A,INSERT_VALUES));
        if (build_rhs) {
          PetscCall(DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES));
        }
      }

      /* P equation : u_x + v_y = 0

         Note that this includes an explicit zero on the diagonal. This is only needed for
         direct solvers (not required if using an iterative solver and setting the constant-pressure nullspace)

        Note: the scaling by dv is not chosen in a principled way and is likely sub-optimal
       */
      if (!parameters->faces_only) {
        if (ctx->pin_pressure && ex == 0 && ey == 0) { /* Pin the first pressure node to zero, if requested */
          DMStagStencil     row;
          const PetscScalar val_A = K_bound;
          const PetscScalar val_rhs = 0.0;

          row.i = ex; row.j = ey; row.loc = DMSTAG_ELEMENT; row.c = 0;
          PetscCall(DMStagMatSetValuesStencil(dm_main,A,1,&row,1,&row,&val_A,INSERT_VALUES));
          if (build_rhs) {
            PetscCall(DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES));
          }
        } else {
          DMStagStencil     row,col[5];
          PetscScalar       val_A[5];
          const PetscScalar val_rhs = 0.0;

          row.i    = ex; row.j    = ey; row.loc    = DMSTAG_ELEMENT; row.c    = 0;
          col[0].i = ex; col[0].j = ey; col[0].loc = DMSTAG_LEFT;    col[0].c = 0; val_A[0] = -1.0 * K_cont * dv / hx;
          col[1].i = ex; col[1].j = ey; col[1].loc = DMSTAG_RIGHT;   col[1].c = 0; val_A[1] =        K_cont * dv / hx;
          col[2].i = ex; col[2].j = ey; col[2].loc = DMSTAG_DOWN;    col[2].c = 0; val_A[2] = -1.0 * K_cont * dv / hy;
          col[3].i = ex; col[3].j = ey; col[3].loc = DMSTAG_UP;      col[3].c = 0; val_A[3] =        K_cont * dv / hy;
          col[4] = row;                                                            val_A[4] =  0.0;
          PetscCall(DMStagMatSetValuesStencil(dm_main,A,1,&row,5,col,val_A,INSERT_VALUES));
          if (build_rhs) {
            PetscCall(DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES));
          }
        }
      }
    }
  }
  PetscCall(DMRestoreLocalVector(dm_coefficients,&coefficients_local));

  /* Add additional inverse viscosity terms (for use in building a preconditioning matrix) */
  if (parameters->include_inverse_visc) {
    PetscCheck(!parameters->faces_only,PetscObjectComm((PetscObject)dm_main),PETSC_ERR_SUP,"Does not make sense with faces only");
    PetscCall(OperatorInsertInverseViscosityPressureTerms(dm_main, dm_coefficients, ctx->levels[level]->coeff, 1.0, A));
  }

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  if (build_rhs) PetscCall(VecAssemblyBegin(rhs));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  if (build_rhs) PetscCall(VecAssemblyEnd(rhs));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateSystem3d(SystemParameters parameters,Mat *pA,Vec *pRhs)
{
  PetscInt        N[3];
  PetscInt        ex,ey,ez,startx,starty,startz,nx,ny,nz;
  Mat             A;
  PetscReal       hx,hy,hz,dv;
  PetscInt        pinx,piny,pinz;
  Vec             coeff_local,rhs;
  PetscBool       build_rhs;
  DM              dm_main, dm_coefficients;
  PetscScalar     K_cont, K_bound;
  Ctx             ctx = parameters->ctx;
  PetscInt        level = parameters->level;

  PetscFunctionBeginUser;
  if (parameters->faces_only) {
    dm_main = ctx->levels[level]->dm_faces;
  } else {
    dm_main = ctx->levels[level]->dm_stokes;
  }
  dm_coefficients = ctx->levels[level]->dm_coefficients;
  K_cont = ctx->levels[level]->K_cont;
  K_bound = ctx->levels[level]->K_bound;
  PetscCall(DMCreateMatrix(dm_main,pA));
  A = *pA;
  build_rhs = (PetscBool) (pRhs != NULL);
  if (build_rhs) {
    PetscCall(DMCreateGlobalVector(dm_main,pRhs));
    rhs = *pRhs;
  } else {
    rhs = NULL;
  }
  PetscCall(DMStagGetCorners(dm_main,&startx,&starty,&startz,&nx,&ny,&nz,NULL,NULL,NULL));
  PetscCall(DMStagGetGlobalSizes(dm_main,&N[0],&N[1],&N[2]));
  hx = ctx->levels[level]->hx_characteristic;
  hy = ctx->levels[level]->hy_characteristic;
  hz = ctx->levels[level]->hz_characteristic;
  dv = hx*hy*hz;
  PetscCall(DMGetLocalVector(dm_coefficients,&coeff_local));
  PetscCall(DMGlobalToLocal(dm_coefficients,ctx->levels[level]->coeff,INSERT_VALUES,coeff_local));

  PetscCheck(N[0] >= 2 && N[1] >= 2 && N[2] >= 2,PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not implemented for less than 2 elements in any direction");
  pinx = 1; piny = 0; pinz = 0; /* Depends on assertion above that there are at least two element in the x direction */

  /* Loop over all local elements.

     For each element, fill 4-7 rows of the matrix, corresponding to
     - the pressure degree of freedom (dof), centered on the element
     - the 3 velocity dofs on left, bottom, and back faces of the element
     - velocity dof on the right, top, and front faces of the element (only on domain boundaries)

   */
  for (ez = startz; ez < startz + nz; ++ez) {
    for (ey = starty; ey < starty + ny; ++ey) {
      for (ex = startx; ex < startx + nx; ++ex) {
        const PetscBool left_boundary   = (PetscBool) (ex == 0);
        const PetscBool right_boundary  = (PetscBool) (ex == N[0]-1);
        const PetscBool bottom_boundary = (PetscBool) (ey == 0);
        const PetscBool top_boundary    = (PetscBool) (ey == N[1]-1);
        const PetscBool back_boundary   = (PetscBool) (ez == 0);
        const PetscBool front_boundary  = (PetscBool) (ez == N[2]-1);

        /* Note that below, we depend on the check above that there is never one
           element (globally) in a given direction.  Thus, for example, an
           element is never both on the left and right boundary */

        /* X-faces - right boundary */
        if (right_boundary) {
          /* Right x-velocity Dirichlet */
          DMStagStencil     row;
          const PetscScalar val_rhs = 0.0;
          const PetscScalar val_A = K_bound;

          row.i = ex; row.j = ey; row.k = ez; row.loc = DMSTAG_RIGHT; row.c = 0;
          PetscCall(DMStagMatSetValuesStencil(dm_main,A,1,&row,1,&row,&val_A,INSERT_VALUES));
          if (build_rhs) {
            PetscCall(DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES));
          }
        }

        /* X faces - left*/
        {
          DMStagStencil row;

          row.i = ex; row.j = ey; row.k = ez; row.loc = DMSTAG_LEFT; row.c = 0;

          if (left_boundary) {
            /* Left x-velocity Dirichlet */
            const PetscScalar val_rhs = 0.0;
            const PetscScalar val_A = K_bound;

            PetscCall(DMStagMatSetValuesStencil(dm_main,A,1,&row,1,&row,&val_A,INSERT_VALUES));
            if (build_rhs) {
              PetscCall(DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES));
            }
          } else {
            /* X-momentum equation */
            PetscInt      count;
            DMStagStencil col[17];
            PetscScalar   val_A[17];
            DMStagStencil eta_point[6];
            PetscScalar   eta[6],eta_left,eta_right,eta_up,eta_down,eta_back,eta_front; /* relative to the left face */
            const PetscScalar val_rhs = 0.0;

            /* Get eta values */
            eta_point[0].i = ex-1; eta_point[0].j = ey; eta_point[0].k = ez; eta_point[0].loc = DMSTAG_ELEMENT;    eta_point[0].c = 0; /* Left  */
            eta_point[1].i = ex;   eta_point[1].j = ey; eta_point[1].k = ez; eta_point[1].loc = DMSTAG_ELEMENT;    eta_point[1].c = 0; /* Right */
            eta_point[2].i = ex;   eta_point[2].j = ey; eta_point[2].k = ez; eta_point[2].loc = DMSTAG_DOWN_LEFT;  eta_point[2].c = 0; /* Down  */
            eta_point[3].i = ex;   eta_point[3].j = ey; eta_point[3].k = ez; eta_point[3].loc = DMSTAG_UP_LEFT;    eta_point[3].c = 0; /* Up    */
            eta_point[4].i = ex;   eta_point[4].j = ey; eta_point[4].k = ez; eta_point[4].loc = DMSTAG_BACK_LEFT;  eta_point[4].c = 0; /* Back  */
            eta_point[5].i = ex;   eta_point[5].j = ey; eta_point[5].k = ez; eta_point[5].loc = DMSTAG_FRONT_LEFT; eta_point[5].c = 0; /* Front  */
            PetscCall(DMStagVecGetValuesStencil(dm_coefficients,coeff_local,6,eta_point,eta));
            eta_left = eta[0]; eta_right = eta[1]; eta_down = eta[2]; eta_up = eta[3]; eta_back = eta[4]; eta_front = eta[5];

            count = 0;

            col[count] = row;
            val_A[count] = -2.0 * dv * (eta_left + eta_right) / (hx*hx);
            if (!top_boundary)    val_A[count] += -1.0 * dv * eta_up    / (hy*hy);
            if (!bottom_boundary) val_A[count] += -1.0 * dv * eta_down  / (hy*hy);
            if (!back_boundary)   val_A[count] += -1.0 * dv * eta_back  / (hz*hz);
            if (!front_boundary)  val_A[count] += -1.0 * dv * eta_front / (hz*hz);
            ++count;

            col[count].i = ex-1; col[count].j = ey; col[count].k = ez; col[count].loc = DMSTAG_LEFT; col[count].c = 0;
            val_A[count] = 2.0 * dv * eta_left  / (hx*hx); ++count;
            col[count].i = ex+1; col[count].j = ey; col[count].k = ez; col[count].loc = DMSTAG_LEFT; col[count].c = 0;
            val_A[count] = 2.0 * dv * eta_right  / (hx*hx); ++count;
            if (!bottom_boundary) {
              col[count].i = ex; col[count].j = ey-1; col[count].k = ez; col[count].loc = DMSTAG_LEFT; col[count].c = 0;
              val_A[count] = dv * eta_down / (hy*hy); ++count;
            }
            if (!top_boundary) {
              col[count].i = ex; col[count].j = ey+1; col[count].k = ez; col[count].loc = DMSTAG_LEFT; col[count].c = 0;
              val_A[count] = dv * eta_up / (hy*hy); ++count;
            }
            if (!back_boundary) {
              col[count].i = ex; col[count].j = ey; col[count].k = ez-1; col[count].loc = DMSTAG_LEFT; col[count].c = 0;
              val_A[count] = dv * eta_back / (hz*hz); ++count;
            }
            if (!front_boundary) {
              col[count].i = ex; col[count].j = ey; col[count].k = ez+1; col[count].loc = DMSTAG_LEFT; col[count].c = 0;
              val_A[count] = dv * eta_front / (hz*hz); ++count;
            }

            col[count].i  = ex-1; col[count].j  = ey; col[count].k = ez; col[count].loc  = DMSTAG_DOWN;  col[count].c = 0;
            val_A[count]  =        dv * eta_down  / (hx*hy); ++count; /* down left */
            col[count].i  = ex  ; col[count].j  = ey; col[count].k = ez; col[count].loc  = DMSTAG_DOWN;  col[count].c = 0;
            val_A[count]  = -1.0 * dv * eta_down  / (hx*hy); ++count; /* down right */

            col[count].i  = ex-1; col[count].j  = ey; col[count].k = ez; col[count].loc  = DMSTAG_UP;    col[count].c = 0;
            val_A[count]  = -1.0 * dv * eta_up    / (hx*hy); ++count; /* up left */
            col[count].i  = ex  ; col[count].j  = ey; col[count].k = ez; col[count].loc  = DMSTAG_UP;    col[count].c = 0;
            val_A[count]  =        dv * eta_up    / (hx*hy); ++count; /* up right */

            col[count].i  = ex-1; col[count].j  = ey; col[count].k = ez; col[count].loc  = DMSTAG_BACK;  col[count].c = 0;
            val_A[count]  =        dv * eta_back  / (hx*hz); ++count; /* back left */
            col[count].i  = ex  ; col[count].j  = ey; col[count].k = ez; col[count].loc  = DMSTAG_BACK;  col[count].c = 0;
            val_A[count]  = -1.0 * dv * eta_back  / (hx*hz); ++count; /* back right */

            col[count].i  = ex-1; col[count].j  = ey; col[count].k = ez; col[count].loc  = DMSTAG_FRONT; col[count].c = 0;
            val_A[count]  = -1.0 * dv * eta_front / (hx*hz); ++count; /* front left */
            col[count].i  = ex  ; col[count].j  = ey; col[count].k = ez; col[count].loc  = DMSTAG_FRONT; col[count].c = 0;
            val_A[count]  =        dv * eta_front / (hx*hz); ++count; /* front right */

            if (!parameters->faces_only) {
              col[count].i = ex-1; col[count].j = ey; col[count].k = ez; col[count].loc = DMSTAG_ELEMENT; col[count].c  = 0;
              val_A[count] =             K_cont * dv / hx; ++count;
              col[count].i = ex;   col[count].j = ey; col[count].k = ez; col[count].loc = DMSTAG_ELEMENT; col[count].c  = 0;
              val_A[count] = -1.0 *      K_cont * dv / hx; ++count;
            }

            PetscCall(DMStagMatSetValuesStencil(dm_main,A,1,&row,count,col,val_A,INSERT_VALUES));
            if (build_rhs) {
              PetscCall(DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES));
            }
          }
        }

        /* Y faces - top boundary */
        if (top_boundary) {
          /* Top y-velocity Dirichlet */
          DMStagStencil     row;
          const PetscScalar val_rhs = 0.0;
          const PetscScalar val_A = K_bound;

          row.i = ex; row.j = ey; row.k = ez; row.loc = DMSTAG_UP; row.c = 0;
          PetscCall(DMStagMatSetValuesStencil(dm_main,A,1,&row,1,&row,&val_A,INSERT_VALUES));
          if (build_rhs) {
            PetscCall(DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES));
          }
        }

        /* Y faces - down */
        {
          DMStagStencil row;

          row.i = ex; row.j = ey; row.k = ez; row.loc = DMSTAG_DOWN; row.c = 0;

          if (bottom_boundary) {
            /* Bottom y-velocity Dirichlet */
            const PetscScalar val_rhs = 0.0;
            const PetscScalar val_A = K_bound;

            PetscCall(DMStagMatSetValuesStencil(dm_main,A,1,&row,1,&row,&val_A,INSERT_VALUES));
            if (build_rhs) {
              PetscCall(DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES));
            }
          } else {
            /* Y-momentum equation (including non-zero forcing) */
            PetscInt      count;
            DMStagStencil col[17];
            PetscScalar   val_rhs,val_A[17];
            DMStagStencil eta_point[6],rho_point[4];
            PetscScalar   eta[6],rho[4],eta_left,eta_right,eta_up,eta_down,eta_back,eta_front; /* relative to the bottom face */

            if (build_rhs) {
              /* get rho values  (note .c = 1) */
              /* Note that we have rho at perhaps strange points (edges not corners) */
              rho_point[0].i = ex; rho_point[0].j = ey; rho_point[0].k = ez; rho_point[0].loc = DMSTAG_DOWN_LEFT;  rho_point[0].c = 1;
              rho_point[1].i = ex; rho_point[1].j = ey; rho_point[1].k = ez; rho_point[1].loc = DMSTAG_DOWN_RIGHT; rho_point[1].c = 1;
              rho_point[2].i = ex; rho_point[2].j = ey; rho_point[2].k = ez; rho_point[2].loc = DMSTAG_BACK_DOWN;  rho_point[2].c = 1;
              rho_point[3].i = ex; rho_point[3].j = ey; rho_point[3].k = ez; rho_point[3].loc = DMSTAG_FRONT_DOWN; rho_point[3].c = 1;
              PetscCall(DMStagVecGetValuesStencil(dm_coefficients,coeff_local,4,rho_point,rho));

              /* Compute forcing */
              val_rhs = ctx->gy * dv * (0.25 * (rho[0] + rho[1] + rho[2] + rho[3]));
            }

            /* Get eta values */
            eta_point[0].i = ex; eta_point[0].j = ey;   eta_point[0].k = ez; eta_point[0].loc = DMSTAG_DOWN_LEFT;  eta_point[0].c = 0; /* Left  */
            eta_point[1].i = ex; eta_point[1].j = ey;   eta_point[1].k = ez; eta_point[1].loc = DMSTAG_DOWN_RIGHT; eta_point[1].c = 0; /* Right */
            eta_point[2].i = ex; eta_point[2].j = ey-1; eta_point[2].k = ez; eta_point[2].loc = DMSTAG_ELEMENT;    eta_point[2].c = 0; /* Down  */
            eta_point[3].i = ex; eta_point[3].j = ey;   eta_point[3].k = ez; eta_point[3].loc = DMSTAG_ELEMENT;    eta_point[3].c = 0; /* Up    */
            eta_point[4].i = ex; eta_point[4].j = ey;   eta_point[4].k = ez; eta_point[4].loc = DMSTAG_BACK_DOWN;  eta_point[4].c = 0; /* Back  */
            eta_point[5].i = ex; eta_point[5].j = ey;   eta_point[5].k = ez; eta_point[5].loc = DMSTAG_FRONT_DOWN; eta_point[5].c = 0; /* Front  */
            PetscCall(DMStagVecGetValuesStencil(dm_coefficients,coeff_local,6,eta_point,eta));
            eta_left = eta[0]; eta_right = eta[1]; eta_down = eta[2]; eta_up = eta[3]; eta_back = eta[4]; eta_front = eta[5];

            count = 0;

            col[count] = row;
            val_A[count] = -2.0 * dv * (eta_up + eta_down) / (hy*hy);
            if (!left_boundary)  val_A[count] += -1.0 * dv * eta_left  / (hx*hx);
            if (!right_boundary) val_A[count] += -1.0 * dv * eta_right / (hx*hx);
            if (!back_boundary)  val_A[count] += -1.0 * dv * eta_back  / (hz*hz);
            if (!front_boundary) val_A[count] += -1.0 * dv * eta_front / (hz*hz);
            ++count;

            col[count].i = ex; col[count].j = ey-1; col[count].k = ez; col[count].loc = DMSTAG_DOWN; col[count].c = 0;
            val_A[count] = 2.0 * dv * eta_down / (hy*hy); ++count;
            col[count].i = ex; col[count].j = ey+1; col[count].k = ez; col[count].loc = DMSTAG_DOWN; col[count].c = 0;
            val_A[count] = 2.0 * dv * eta_up   / (hy*hy); ++count;

            if (!left_boundary) {
              col[count].i = ex-1; col[count].j = ey; col[count].k = ez; col[count].loc = DMSTAG_DOWN; col[count].c = 0;
              val_A[count] = dv * eta_left / (hx*hx); ++count;
            }
            if (!right_boundary) {
              col[count].i = ex+1; col[count].j = ey; col[count].k = ez; col[count].loc = DMSTAG_DOWN; col[count].c = 0;
              val_A[count] = dv * eta_right / (hx*hx); ++count;
            }
            if (!back_boundary) {
              col[count].i = ex; col[count].j = ey; col[count].k = ez-1; col[count].loc = DMSTAG_DOWN; col[count].c = 0;
              val_A[count] = dv * eta_back / (hz*hz); ++count;
            }
            if (!front_boundary) {
              col[count].i = ex; col[count].j = ey; col[count].k = ez+1; col[count].loc = DMSTAG_DOWN; col[count].c = 0;
              val_A[count] = dv * eta_front / (hz*hz); ++count;
            }

            col[count].i  = ex; col[count].j  = ey-1; col[count].k = ez; col[count].loc = DMSTAG_LEFT;  col[count].c = 0;
            val_A[count]  =        dv * eta_left  / (hx*hy); ++count; /* down left*/
            col[count].i  = ex; col[count].j  = ey;   col[count].k = ez; col[count].loc = DMSTAG_LEFT;  col[count].c = 0;
            val_A[count]  = -1.0 * dv * eta_left  / (hx*hy); ++count; /* up left*/

            col[count].i  = ex; col[count].j  = ey-1; col[count].k = ez; col[count].loc = DMSTAG_RIGHT; col[count].c = 0;
            val_A[count]  = -1.0 * dv * eta_right / (hx*hy); ++count; /* down right*/
            col[count].i  = ex; col[count].j  = ey;   col[count].k = ez; col[count].loc = DMSTAG_RIGHT; col[count].c = 0;
            val_A[count]  =        dv * eta_right / (hx*hy); ++count; /* up right*/

            col[count].i  = ex; col[count].j  = ey-1; col[count].k = ez; col[count].loc  = DMSTAG_BACK;  col[count].c = 0;
            val_A[count]  =        dv * eta_back  / (hy*hz); ++count; /* back down */
            col[count].i  = ex; col[count].j  = ey;   col[count].k = ez; col[count].loc  = DMSTAG_BACK;  col[count].c = 0;
            val_A[count]  = -1.0 * dv * eta_back  / (hy*hz); ++count;/* back up */

            col[count].i  = ex; col[count].j  = ey-1; col[count].k = ez; col[count].loc  = DMSTAG_FRONT; col[count].c = 0;
            val_A[count]  = -1.0 * dv * eta_front / (hy*hz); ++count; /* front down */
            col[count].i  = ex; col[count].j  = ey;   col[count].k = ez; col[count].loc  = DMSTAG_FRONT; col[count].c = 0;
            val_A[count]  =        dv * eta_front / (hy*hz); ++count;/* front up */

            if (!parameters->faces_only) {
              col[count].i = ex; col[count].j = ey-1; col[count].k = ez; col[count].loc = DMSTAG_ELEMENT; col[count].c  = 0;
              val_A[count] =             K_cont * dv / hy; ++count;
              col[count].i = ex; col[count].j = ey;   col[count].k = ez; col[count].loc = DMSTAG_ELEMENT; col[count].c  = 0;
              val_A[count] = -1.0 *      K_cont * dv / hy; ++count;
            }

            PetscCall(DMStagMatSetValuesStencil(dm_main,A,1,&row,count,col,val_A,INSERT_VALUES));
            if (build_rhs) {
              PetscCall(DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES));
            }
          }
        }

        if (front_boundary) {
          /* Front z-velocity Dirichlet */
          DMStagStencil     row;
          const PetscScalar val_rhs = 0.0;
          const PetscScalar val_A = K_bound;

          row.i = ex; row.j = ey; row.k = ez; row.loc = DMSTAG_FRONT; row.c = 0;
          PetscCall(DMStagMatSetValuesStencil(dm_main,A,1,&row,1,&row,&val_A,INSERT_VALUES));
          if (build_rhs) {
            PetscCall(DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES));
          }
        }

        /* Z faces - back */
        {
          DMStagStencil row;

          row.i = ex; row.j = ey; row.k = ez; row.loc = DMSTAG_BACK; row.c = 0;

          if (back_boundary) {
            /* Back z-velocity Dirichlet */
            const PetscScalar val_rhs = 0.0;
            const PetscScalar val_A = K_bound;

            PetscCall(DMStagMatSetValuesStencil(dm_main,A,1,&row,1,&row,&val_A,INSERT_VALUES));
            if (build_rhs) {
              PetscCall(DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES));
            }
          } else {
            /* Z-momentum equation */
            PetscInt          count;
            DMStagStencil     col[17];
            PetscScalar       val_A[17];
            DMStagStencil     eta_point[6];
            PetscScalar       eta[6],eta_left,eta_right,eta_up,eta_down,eta_back,eta_front; /* relative to the back face */
            const PetscScalar val_rhs = 0.0;

            /* Get eta values */
            eta_point[0].i = ex; eta_point[0].j = ey; eta_point[0].k = ez;   eta_point[0].loc = DMSTAG_BACK_LEFT;  eta_point[0].c = 0; /* Left  */
            eta_point[1].i = ex; eta_point[1].j = ey; eta_point[1].k = ez;   eta_point[1].loc = DMSTAG_BACK_RIGHT; eta_point[1].c = 0; /* Right */
            eta_point[2].i = ex; eta_point[2].j = ey; eta_point[2].k = ez;   eta_point[2].loc = DMSTAG_BACK_DOWN;  eta_point[2].c = 0; /* Down  */
            eta_point[3].i = ex; eta_point[3].j = ey; eta_point[3].k = ez;   eta_point[3].loc = DMSTAG_BACK_UP;    eta_point[3].c = 0; /* Up    */
            eta_point[4].i = ex; eta_point[4].j = ey; eta_point[4].k = ez-1; eta_point[4].loc = DMSTAG_ELEMENT;    eta_point[4].c = 0; /* Back  */
            eta_point[5].i = ex; eta_point[5].j = ey; eta_point[5].k = ez;   eta_point[5].loc = DMSTAG_ELEMENT;    eta_point[5].c = 0; /* Front  */
            PetscCall(DMStagVecGetValuesStencil(dm_coefficients,coeff_local,6,eta_point,eta));
            eta_left = eta[0]; eta_right = eta[1]; eta_down = eta[2]; eta_up = eta[3]; eta_back = eta[4]; eta_front = eta[5];

            count = 0;

            col[count] = row;
            val_A[count] = -2.0 * dv * (eta_back + eta_front) / (hz*hz);
            if (!left_boundary)   val_A[count] += -1.0 * dv * eta_left  / (hx*hx);
            if (!right_boundary)  val_A[count] += -1.0 * dv * eta_right / (hx*hx);
            if (!top_boundary)    val_A[count] += -1.0 * dv * eta_up    / (hy*hy);
            if (!bottom_boundary) val_A[count] += -1.0 * dv * eta_down  / (hy*hy);
            ++count;

            col[count].i = ex; col[count].j = ey; col[count].k = ez-1; col[count].loc = DMSTAG_BACK; col[count].c = 0;
            val_A[count] = 2.0 * dv * eta_back  / (hz*hz); ++count;
            col[count].i = ex; col[count].j = ey; col[count].k = ez+1; col[count].loc = DMSTAG_BACK; col[count].c = 0;
            val_A[count] = 2.0 * dv * eta_front / (hz*hz); ++count;

            if (!left_boundary) {
              col[count].i = ex-1; col[count].j = ey; col[count].k = ez; col[count].loc = DMSTAG_BACK; col[count].c = 0;
              val_A[count] = dv * eta_left / (hx*hx); ++count;
            }
            if (!right_boundary) {
              col[count].i = ex+1; col[count].j = ey; col[count].k = ez; col[count].loc = DMSTAG_BACK; col[count].c = 0;
              val_A[count] = dv * eta_right / (hx*hx); ++count;
            }
            if (!bottom_boundary) {
              col[count].i = ex; col[count].j = ey-1; col[count].k = ez; col[count].loc = DMSTAG_BACK; col[count].c = 0;
              val_A[count] = dv * eta_down / (hy*hy); ++count;
            }
            if (!top_boundary) {
              col[count].i = ex; col[count].j = ey+1; col[count].k = ez; col[count].loc = DMSTAG_BACK; col[count].c = 0;
              val_A[count] = dv * eta_up  / (hy*hy); ++count;
            }

            col[count].i  = ex; col[count].j  = ey; col[count].k = ez-1; col[count].loc = DMSTAG_LEFT; col[count].c = 0;
            val_A[count]  =        dv * eta_left  / (hx*hz); ++count; /* back left*/
            col[count].i  = ex; col[count].j  = ey; col[count].k = ez;   col[count].loc = DMSTAG_LEFT; col[count].c = 0;
            val_A[count]  = -1.0 * dv * eta_left  / (hx*hz); ++count; /* front left*/

            col[count].i  = ex; col[count].j  = ey; col[count].k = ez-1; col[count].loc = DMSTAG_RIGHT; col[count].c = 0;
            val_A[count]  = -1.0 * dv * eta_right / (hx*hz); ++count; /* back right */
            col[count].i  = ex; col[count].j  = ey; col[count].k = ez;   col[count].loc = DMSTAG_RIGHT; col[count].c = 0;
            val_A[count]  =        dv * eta_right / (hx*hz); ++count; /* front right*/

            col[count].i  = ex; col[count].j  = ey; col[count].k = ez-1; col[count].loc = DMSTAG_DOWN; col[count].c = 0;
            val_A[count]  =        dv * eta_down  / (hy*hz); ++count; /* back down */
            col[count].i  = ex; col[count].j  = ey; col[count].k = ez;   col[count].loc = DMSTAG_DOWN; col[count].c = 0;
            val_A[count]  = -1.0 * dv * eta_down  / (hy*hz); ++count; /* back down */

            col[count].i  = ex; col[count].j  = ey; col[count].k = ez-1; col[count].loc = DMSTAG_UP; col[count].c = 0;
            val_A[count]  = -1.0 * dv * eta_up    / (hy*hz); ++count; /* back up */
            col[count].i  = ex; col[count].j  = ey; col[count].k = ez;   col[count].loc = DMSTAG_UP; col[count].c = 0;
            val_A[count]  =        dv * eta_up    / (hy*hz); ++count; /* back up */

            if (!parameters->faces_only) {
              col[count].i = ex; col[count].j = ey; col[count].k = ez-1; col[count].loc = DMSTAG_ELEMENT; col[count].c  = 0;
              val_A[count] =             K_cont * dv / hz; ++count;
              col[count].i = ex; col[count].j = ey;   col[count].k = ez; col[count].loc = DMSTAG_ELEMENT; col[count].c  = 0;
              val_A[count] = -1.0 *      K_cont * dv / hz; ++count;
            }

            PetscCall(DMStagMatSetValuesStencil(dm_main,A,1,&row,count,col,val_A,INSERT_VALUES));
            if (build_rhs) {
              PetscCall(DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES));
            }
          }
        }

        /* Elements */
        if (!parameters->faces_only) {
          DMStagStencil row;

          row.i = ex; row.j = ey; row.k = ez; row.loc = DMSTAG_ELEMENT; row.c = 0;

          if (ctx->pin_pressure && ex == pinx && ey == piny && ez == pinz) {
            /* Pin a pressure node to zero, if requested */
            const PetscScalar val_A = K_bound;
            const PetscScalar val_rhs = 0.0;

            PetscCall(DMStagMatSetValuesStencil(dm_main,A,1,&row,1,&row,&val_A,INSERT_VALUES));
            if (build_rhs) {
              PetscCall(DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES));
            }
          } else {
            /* Continuity equation */
            /* Note that this includes an explicit zero on the diagonal. This is only needed for
               some direct solvers (not required if using an iterative solver and setting a constant-pressure nullspace) */
            /* Note: the scaling by dv is not chosen in a principled way and is likely sub-optimal */
            DMStagStencil     col[7];
            PetscScalar       val_A[7];
            const PetscScalar val_rhs = 0.0;

            col[0].i = ex; col[0].j = ey; col[0].k = ez; col[0].loc = DMSTAG_LEFT;    col[0].c = 0; val_A[0] = - 1.0 * K_cont * dv / hx;
            col[1].i = ex; col[1].j = ey; col[1].k = ez; col[1].loc = DMSTAG_RIGHT;   col[1].c = 0; val_A[1] =         K_cont * dv / hx;
            col[2].i = ex; col[2].j = ey; col[2].k = ez; col[2].loc = DMSTAG_DOWN;    col[2].c = 0; val_A[2] = - 1.0 * K_cont * dv / hy;
            col[3].i = ex; col[3].j = ey; col[3].k = ez; col[3].loc = DMSTAG_UP;      col[3].c = 0; val_A[3] =         K_cont * dv / hy;
            col[4].i = ex; col[4].j = ey; col[4].k = ez; col[4].loc = DMSTAG_BACK;    col[4].c = 0; val_A[4] = - 1.0 * K_cont * dv / hz;
            col[5].i = ex; col[5].j = ey; col[5].k = ez; col[5].loc = DMSTAG_FRONT;   col[5].c = 0; val_A[5] =         K_cont * dv / hz;
            col[6] = row;                                                                           val_A[6] = 0.0;
            PetscCall(DMStagMatSetValuesStencil(dm_main,A,1,&row,7,col,val_A,INSERT_VALUES));
            if (build_rhs) {
              PetscCall(DMStagVecSetValuesStencil(dm_main,rhs,1,&row,&val_rhs,INSERT_VALUES));
            }
          }
        }
      }
    }
  }
  PetscCall(DMRestoreLocalVector(dm_coefficients,&coeff_local));

  /* Add additional inverse viscosity terms (for use in building a preconditioning matrix) */
  if (parameters->include_inverse_visc) {
    PetscCheck(!parameters->faces_only,PetscObjectComm((PetscObject)dm_main),PETSC_ERR_SUP,"Does not make sense with faces only");
    PetscCall(OperatorInsertInverseViscosityPressureTerms(dm_main, dm_coefficients, ctx->levels[level]->coeff, 1.0, A));
  }

  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  if (build_rhs) PetscCall(VecAssemblyBegin(rhs));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  if (build_rhs) PetscCall(VecAssemblyEnd(rhs));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateSystem(SystemParameters parameters,Mat *pA,Vec *pRhs)
{
  PetscFunctionBeginUser;
  if (parameters->ctx->dim == 2) {
    PetscCall(CreateSystem2d(parameters,pA,pRhs));
  } else if (parameters->ctx->dim == 3) {
    PetscCall(CreateSystem3d(parameters,pA,pRhs));
  } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Unsupported dimension %" PetscInt_FMT,parameters->ctx->dim);
  PetscFunctionReturn(0);
}

PetscErrorCode PopulateCoefficientData(Ctx ctx,PetscInt level)
{
  PetscInt       dim;
  PetscInt       N[3];
  PetscInt       ex,ey,ez,startx,starty,startz,nx,ny,nz;
  PetscInt       slot_prev,slot_center;
  PetscInt       slot_rho_downleft,slot_rho_backleft,slot_rho_backdown,slot_eta_element,slot_eta_downleft,slot_eta_backleft,slot_eta_backdown;
  Vec            coeff_local;
  PetscReal      **arr_coordinates_x,**arr_coordinates_y,**arr_coordinates_z;
  DM             dm_coefficients;
  Vec            coeff;

  PetscFunctionBeginUser;
  dm_coefficients = ctx->levels[level]->dm_coefficients;
  PetscCall(DMGetDimension(dm_coefficients,&dim));

  /* Create global coefficient vector */
  PetscCall(DMCreateGlobalVector(dm_coefficients,&ctx->levels[level]->coeff));
  coeff = ctx->levels[level]->coeff;

  /* Get temporary access to a local representation of the coefficient data */
  PetscCall(DMGetLocalVector(dm_coefficients,&coeff_local));
  PetscCall(DMGlobalToLocal(dm_coefficients,coeff,INSERT_VALUES,coeff_local));

  /* Use direct array acccess to coefficient and coordinate arrays, to popoulate coefficient data */
  PetscCall(DMStagGetGhostCorners(dm_coefficients,&startx,&starty,&startz,&nx,&ny,&nz));
  PetscCall(DMStagGetGlobalSizes(dm_coefficients,&N[0],&N[1],&N[2]));
  PetscCall(DMStagGetProductCoordinateArraysRead(dm_coefficients,&arr_coordinates_x,&arr_coordinates_y,&arr_coordinates_z));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm_coefficients,DMSTAG_ELEMENT,&slot_center));
  PetscCall(DMStagGetProductCoordinateLocationSlot(dm_coefficients,DMSTAG_LEFT,&slot_prev));
  PetscCall(DMStagGetLocationSlot(dm_coefficients,DMSTAG_ELEMENT,  0,&slot_eta_element));
  PetscCall(DMStagGetLocationSlot(dm_coefficients,DMSTAG_DOWN_LEFT,0,&slot_eta_downleft));
  PetscCall(DMStagGetLocationSlot(dm_coefficients,DMSTAG_DOWN_LEFT,1,&slot_rho_downleft));
  if (dim == 2) {
    PetscScalar ***arr_coefficients;

    PetscCall(DMStagVecGetArray(dm_coefficients,coeff_local,&arr_coefficients));
    /* Note that these ranges are with respect to the local representation */
    for (ey =starty; ey<starty+ny; ++ey) {
      for (ex = startx; ex<startx+nx; ++ex) {
        arr_coefficients[ey][ex][slot_eta_element]  = ctx->GetEta(ctx,arr_coordinates_x[ex][slot_center],arr_coordinates_y[ey][slot_center],0.0);
        arr_coefficients[ey][ex][slot_eta_downleft] = ctx->GetEta(ctx,arr_coordinates_x[ex][slot_prev],  arr_coordinates_y[ey][slot_prev],  0.0);
        arr_coefficients[ey][ex][slot_rho_downleft] = ctx->GetRho(ctx,arr_coordinates_x[ex][slot_prev],  arr_coordinates_y[ey][slot_prev],  0.0);
      }
    }
    PetscCall(DMStagVecRestoreArray(dm_coefficients,coeff_local,&arr_coefficients));
  } else if (dim == 3) {
    PetscScalar ****arr_coefficients;

    PetscCall(DMStagGetLocationSlot(dm_coefficients,DMSTAG_BACK_LEFT,0,&slot_eta_backleft));
    PetscCall(DMStagGetLocationSlot(dm_coefficients,DMSTAG_BACK_LEFT,1,&slot_rho_backleft));
    PetscCall(DMStagGetLocationSlot(dm_coefficients,DMSTAG_BACK_DOWN,0,&slot_eta_backdown));
    PetscCall(DMStagGetLocationSlot(dm_coefficients,DMSTAG_BACK_DOWN,1,&slot_rho_backdown));
    PetscCall(DMStagVecGetArray(dm_coefficients,coeff_local,&arr_coefficients));
    /* Note that these are with respect to the entire local representation, including ghosts */
    for (ez = startz; ez<startz+nz; ++ez) {
      for (ey = starty; ey<starty+ny; ++ey) {
        for (ex = startx; ex<startx+nx; ++ex) {
          const PetscScalar x_prev = arr_coordinates_x[ex][slot_prev];
          const PetscScalar y_prev = arr_coordinates_y[ey][slot_prev];
          const PetscScalar z_prev = arr_coordinates_z[ez][slot_prev];
          const PetscScalar x_center = arr_coordinates_x[ex][slot_center];
          const PetscScalar y_center = arr_coordinates_y[ey][slot_center];
          const PetscScalar z_center = arr_coordinates_z[ez][slot_center];

          arr_coefficients[ez][ey][ex][slot_eta_element]  = ctx->GetEta(ctx,x_center,y_center,z_center);
          arr_coefficients[ez][ey][ex][slot_eta_downleft] = ctx->GetEta(ctx,x_prev,y_prev,z_center);
          arr_coefficients[ez][ey][ex][slot_rho_downleft] = ctx->GetRho(ctx,x_prev,y_prev,z_center);
          arr_coefficients[ez][ey][ex][slot_eta_backleft] = ctx->GetEta(ctx,x_prev,y_center,z_prev);
          arr_coefficients[ez][ey][ex][slot_rho_backleft] = ctx->GetRho(ctx,x_prev,y_center,z_prev);
          arr_coefficients[ez][ey][ex][slot_eta_backdown] = ctx->GetEta(ctx,x_center,y_prev,z_prev);
          arr_coefficients[ez][ey][ex][slot_rho_backdown] = ctx->GetRho(ctx,x_center,y_prev,z_prev);
        }
      }
    }
    PetscCall(DMStagVecRestoreArray(dm_coefficients,coeff_local,&arr_coefficients));
  } else SETERRQ(PetscObjectComm((PetscObject)dm_coefficients),PETSC_ERR_SUP,"Unsupported dimension %" PetscInt_FMT,dim);
  PetscCall(DMStagRestoreProductCoordinateArraysRead(dm_coefficients,&arr_coordinates_x,&arr_coordinates_y,&arr_coordinates_z));
  PetscCall(DMLocalToGlobal(dm_coefficients,coeff_local,INSERT_VALUES,coeff));
  PetscCall(DMRestoreLocalVector(dm_coefficients,&coeff_local));
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateAuxiliaryOperator(Ctx ctx,PetscInt level, Mat *p_S_hat)
{
  DM             dm_element;
  Mat            S_hat;
  DM             dm_stokes,dm_coefficients;
  Vec            coeff;

  PetscFunctionBeginUser;
  dm_stokes = ctx->levels[level]->dm_stokes;
  dm_coefficients = ctx->levels[level]->dm_coefficients;
  coeff = ctx->levels[level]->coeff;
  if (ctx->dim == 2) {
    PetscCall(DMStagCreateCompatibleDMStag(dm_stokes,0,0,1,0,&dm_element));
  } else if (ctx->dim == 3) {
    PetscCall(DMStagCreateCompatibleDMStag(dm_stokes,0,0,0,1,&dm_element));
  } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not implemented for dimension %" PetscInt_FMT,ctx->dim);
  PetscCall(DMCreateMatrix(dm_element,p_S_hat));
  S_hat = *p_S_hat;
  PetscCall(OperatorInsertInverseViscosityPressureTerms(dm_element,dm_coefficients,coeff,1.0,S_hat));
  PetscCall(MatAssemblyBegin(S_hat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(S_hat,MAT_FINAL_ASSEMBLY));
  PetscCall(DMDestroy(&dm_element));
  PetscFunctionReturn(0);
}

static PetscErrorCode OperatorInsertInverseViscosityPressureTerms(DM dm, DM dm_coefficients, Vec coefficients, PetscScalar scale, Mat mat)
{
  PetscInt       dim,ex,ey,ez,startx,starty,startz,nx,ny,nz;
  Vec            coeff_local;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm,&dim));
  PetscCall(DMGetLocalVector(dm_coefficients,&coeff_local));
  PetscCall(DMGlobalToLocal(dm_coefficients,coefficients,INSERT_VALUES,coeff_local));
  PetscCall(DMStagGetCorners(dm,&startx,&starty,&startz,&nx,&ny,&nz,NULL,NULL,NULL));
  if (dim == 2) { /* Trick to have one loop nest */
    startz = 0;
    nz = 1;
  }
  for (ez = startz; ez<startz+nz; ++ez) {
    for (ey = starty; ey<starty+ny; ++ey) {
      for (ex = startx; ex<startx+nx; ++ex) {
        DMStagStencil  from,to;
        PetscScalar    val;

        /* component 0 on element is viscosity */
        from.i = ex; from.j = ey; from.k = ez; from.c = 0; from.loc = DMSTAG_ELEMENT;
        PetscCall(DMStagVecGetValuesStencil(dm_coefficients,coeff_local,1,&from,&val));
        val = scale/val; /* inverse viscosity, scaled */
        to = from;
        PetscCall(DMStagMatSetValuesStencil(dm,mat,1,&to,1,&to,&val,INSERT_VALUES));
      }
    }
  }
  PetscCall(DMRestoreLocalVector(dm_coefficients,&coeff_local));
  /* Note that this function does not call MatAssembly{Begin,End} */
  PetscFunctionReturn(0);
}

/* Create a pressure-only DMStag and use it to generate a nullspace vector
   - Create a compatible DMStag with one dof per element (and nothing else).
   - Create a constant vector and normalize it
   - Migrate it to a vector on the original dmSol (making use of the fact
   that this will fill in zeros for "extra" dof)
   - Set the nullspace for the operator
   - Destroy everything (the operator keeps the references it needs) */
static PetscErrorCode AttachNullspace(DM dmSol,Mat A)
{
  DM             dmPressure;
  Vec            constantPressure,basis;
  PetscReal      nrm;
  MatNullSpace   matNullSpace;

  PetscFunctionBeginUser;
  PetscCall(DMStagCreateCompatibleDMStag(dmSol,0,0,1,0,&dmPressure));
  PetscCall(DMGetGlobalVector(dmPressure,&constantPressure));
  PetscCall(VecSet(constantPressure,1.0));
  PetscCall(VecNorm(constantPressure,NORM_2,&nrm));
  PetscCall(VecScale(constantPressure,1.0/nrm));
  PetscCall(DMCreateGlobalVector(dmSol,&basis));
  PetscCall(DMStagMigrateVec(dmPressure,constantPressure,dmSol,basis));
  PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)dmSol),PETSC_FALSE,1,&basis,&matNullSpace));
  PetscCall(VecDestroy(&basis));
  PetscCall(MatSetNullSpace(A,matNullSpace));
  PetscCall(MatNullSpaceDestroy(&matNullSpace));
  PetscCall(DMRestoreGlobalVector(dmPressure,&constantPressure));
  PetscCall(DMDestroy(&dmPressure));
  PetscFunctionReturn(0);
}

static PetscErrorCode DumpSolution(Ctx ctx,PetscInt level, Vec x)
{
  DM             dm_stokes,dm_coefficients;
  Vec            coeff;
  DM             dm_vel_avg;
  Vec            vel_avg;
  DM             da_vel_avg,da_p,da_eta_element;
  Vec            vec_vel_avg,vec_p,vec_eta_element;
  DM             da_eta_down_left,da_rho_down_left,da_eta_back_left,da_rho_back_left,da_eta_back_down,da_rho_back_down;
  Vec            vec_eta_down_left,vec_rho_down_left,vec_eta_back_left,vec_rho_back_left,vec_eta_back_down,vec_rho_back_down;
  PetscInt       ex,ey,ez,startx,starty,startz,nx,ny,nz;
  Vec            stokesLocal;

  PetscFunctionBeginUser;
  dm_stokes = ctx->levels[level]->dm_stokes;
  dm_coefficients = ctx->levels[level]->dm_coefficients;
  coeff = ctx->levels[level]->coeff;

  /* For convenience, create a new DM and Vec which will hold averaged velocities
     Note that this could also be accomplished with direct array access, using
     DMStagVecGetArray() and related functions */
  if (ctx->dim == 2) {
    PetscCall(DMStagCreateCompatibleDMStag(dm_stokes,0,0,2,0,&dm_vel_avg)); /* 2 dof per element */
  } else if (ctx->dim == 3) {
    PetscCall(DMStagCreateCompatibleDMStag(dm_stokes,0,0,0,3,&dm_vel_avg)); /* 3 dof per element */
  } else SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not Implemented for dimension %" PetscInt_FMT,ctx->dim);
  PetscCall(DMSetUp(dm_vel_avg));
  PetscCall(DMStagSetUniformCoordinatesProduct(dm_vel_avg,ctx->xmin,ctx->xmax,ctx->ymin,ctx->ymax,ctx->zmin,ctx->zmax));
  PetscCall(DMCreateGlobalVector(dm_vel_avg,&vel_avg));
    PetscCall(DMGetLocalVector(dm_stokes,&stokesLocal));
    PetscCall(DMGlobalToLocal(dm_stokes,x,INSERT_VALUES,stokesLocal));
    PetscCall(DMStagGetCorners(dm_vel_avg,&startx,&starty,&startz,&nx,&ny,&nz,NULL,NULL,NULL));
  if (ctx->dim == 2) {
    for (ey = starty; ey<starty+ny; ++ey) {
      for (ex = startx; ex<startx+nx; ++ex) {
        DMStagStencil from[4],to[2];
        PetscScalar   valFrom[4],valTo[2];

        from[0].i = ex; from[0].j = ey; from[0].loc = DMSTAG_UP;    from[0].c = 0;
        from[1].i = ex; from[1].j = ey; from[1].loc = DMSTAG_DOWN;  from[1].c = 0;
        from[2].i = ex; from[2].j = ey; from[2].loc = DMSTAG_LEFT;  from[2].c = 0;
        from[3].i = ex; from[3].j = ey; from[3].loc = DMSTAG_RIGHT; from[3].c = 0;
        PetscCall(DMStagVecGetValuesStencil(dm_stokes,stokesLocal,4,from,valFrom));
        to[0].i = ex; to[0].j = ey; to[0].loc = DMSTAG_ELEMENT;    to[0].c = 0; valTo[0] = 0.5 * (valFrom[2] + valFrom[3]);
        to[1].i = ex; to[1].j = ey; to[1].loc = DMSTAG_ELEMENT;    to[1].c = 1; valTo[1] = 0.5 * (valFrom[0] + valFrom[1]);
        PetscCall(DMStagVecSetValuesStencil(dm_vel_avg,vel_avg,2,to,valTo,INSERT_VALUES));
      }
    }
  } else if (ctx->dim == 3) {
    for (ez = startz; ez<startz+nz; ++ez) {
      for (ey = starty; ey<starty+ny; ++ey) {
        for (ex = startx; ex<startx+nx; ++ex) {
          DMStagStencil from[6],to[3];
          PetscScalar   valFrom[6],valTo[3];

          from[0].i = ex; from[0].j = ey; from[0].k = ez; from[0].loc = DMSTAG_UP;    from[0].c = 0;
          from[1].i = ex; from[1].j = ey; from[1].k = ez; from[1].loc = DMSTAG_DOWN;  from[1].c = 0;
          from[2].i = ex; from[2].j = ey; from[2].k = ez; from[2].loc = DMSTAG_LEFT;  from[2].c = 0;
          from[3].i = ex; from[3].j = ey; from[3].k = ez; from[3].loc = DMSTAG_RIGHT; from[3].c = 0;
          from[4].i = ex; from[4].j = ey; from[4].k = ez; from[4].loc = DMSTAG_BACK;  from[4].c = 0;
          from[5].i = ex; from[5].j = ey; from[5].k = ez; from[5].loc = DMSTAG_FRONT; from[5].c = 0;
          PetscCall(DMStagVecGetValuesStencil(dm_stokes,stokesLocal,6,from,valFrom));
          to[0].i = ex; to[0].j = ey; to[0].k = ez; to[0].loc = DMSTAG_ELEMENT;    to[0].c = 0; valTo[0] = 0.5 * (valFrom[2] + valFrom[3]);
          to[1].i = ex; to[1].j = ey; to[1].k = ez; to[1].loc = DMSTAG_ELEMENT;    to[1].c = 1; valTo[1] = 0.5 * (valFrom[0] + valFrom[1]);
          to[2].i = ex; to[2].j = ey; to[2].k = ez; to[2].loc = DMSTAG_ELEMENT;    to[2].c = 2; valTo[2] = 0.5 * (valFrom[4] + valFrom[5]);
          PetscCall(DMStagVecSetValuesStencil(dm_vel_avg,vel_avg,3,to,valTo,INSERT_VALUES));
        }
      }
    }
  }
  PetscCall(VecAssemblyBegin(vel_avg));
  PetscCall(VecAssemblyEnd(vel_avg));
  PetscCall(DMRestoreLocalVector(dm_stokes,&stokesLocal));

  /* Create individual DMDAs for sub-grids of our DMStag objects. This is
     somewhat inefficient, but allows use of the DMDA API without re-implementing
     all utilities for DMStag */

  PetscCall(DMStagVecSplitToDMDA(dm_stokes,x,DMSTAG_ELEMENT,0,&da_p,&vec_p));
  PetscCall(PetscObjectSetName((PetscObject)vec_p,"p (scaled)"));

  PetscCall(DMStagVecSplitToDMDA(dm_coefficients,coeff,DMSTAG_ELEMENT,0, &da_eta_element,&vec_eta_element));
  PetscCall(PetscObjectSetName((PetscObject)vec_eta_element,"eta"));

  PetscCall(DMStagVecSplitToDMDA(dm_coefficients,coeff,DMSTAG_DOWN_LEFT,0,&da_eta_down_left,&vec_eta_down_left));
  PetscCall(PetscObjectSetName((PetscObject)vec_eta_down_left,"eta"));

  PetscCall(DMStagVecSplitToDMDA(dm_coefficients,coeff,DMSTAG_DOWN_LEFT,1,&da_rho_down_left,&vec_rho_down_left));
  PetscCall(PetscObjectSetName((PetscObject)vec_rho_down_left,"density"));

  if (ctx->dim == 3) {
    PetscCall(DMStagVecSplitToDMDA(dm_coefficients,coeff,DMSTAG_BACK_LEFT,0,&da_eta_back_left,&vec_eta_back_left));
    PetscCall(PetscObjectSetName((PetscObject)vec_eta_back_left,"eta"));

    PetscCall(DMStagVecSplitToDMDA(dm_coefficients,coeff,DMSTAG_BACK_LEFT,1,&da_rho_back_left,&vec_rho_back_left));
    PetscCall(PetscObjectSetName((PetscObject)vec_rho_back_left,"rho"));

    PetscCall(DMStagVecSplitToDMDA(dm_coefficients,coeff,DMSTAG_BACK_DOWN,0,&da_eta_back_down,&vec_eta_back_down));
    PetscCall(PetscObjectSetName((PetscObject)vec_eta_back_down,"eta"));

    PetscCall(DMStagVecSplitToDMDA(dm_coefficients,coeff,DMSTAG_BACK_DOWN,1,&da_rho_back_down,&vec_rho_back_down));
    PetscCall(PetscObjectSetName((PetscObject)vec_rho_back_down,"rho"));
  }

  PetscCall(DMStagVecSplitToDMDA(dm_vel_avg,vel_avg,DMSTAG_ELEMENT,-3,&da_vel_avg,&vec_vel_avg)); /* note -3 : pad with zero */
  PetscCall(PetscObjectSetName((PetscObject)vec_vel_avg,"Velocity (Averaged)"));

  /* Dump element-based fields to a .vtr file */
  {
    PetscViewer viewer;

    PetscCall(PetscViewerVTKOpen(PetscObjectComm((PetscObject)da_vel_avg),"ex4_element.vtr",FILE_MODE_WRITE,&viewer));
    PetscCall(VecView(vec_vel_avg,viewer));
    PetscCall(VecView(vec_p,viewer));
    PetscCall(VecView(vec_eta_element,viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  /* Dump vertex- or edge-based fields to a second .vtr file */
  {
    PetscViewer viewer;

    PetscCall(PetscViewerVTKOpen(PetscObjectComm((PetscObject)da_eta_down_left),"ex4_down_left.vtr",FILE_MODE_WRITE,&viewer));
    PetscCall(VecView(vec_eta_down_left,viewer));
    PetscCall(VecView(vec_rho_down_left,viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  if (ctx->dim == 3) {
    PetscViewer viewer;

    PetscCall(PetscViewerVTKOpen(PetscObjectComm((PetscObject)da_eta_back_left),"ex4_back_left.vtr",FILE_MODE_WRITE,&viewer));
    PetscCall(VecView(vec_eta_back_left,viewer));
    PetscCall(VecView(vec_rho_back_left,viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }
  if (ctx->dim == 3) {
    PetscViewer viewer;

    PetscCall(PetscViewerVTKOpen(PetscObjectComm((PetscObject)da_eta_back_down),"ex4_back_down.vtr",FILE_MODE_WRITE,&viewer));
    PetscCall(VecView(vec_eta_back_down,viewer));
    PetscCall(VecView(vec_rho_back_down,viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  /* Destroy DMDAs and Vecs */
  PetscCall(VecDestroy(&vec_vel_avg));
  PetscCall(VecDestroy(&vec_p));
  PetscCall(VecDestroy(&vec_eta_element));
  PetscCall(VecDestroy(&vec_eta_down_left));
  if (ctx->dim == 3) {
    PetscCall(VecDestroy(&vec_eta_back_left));
    PetscCall(VecDestroy(&vec_eta_back_down));
  }
  PetscCall(VecDestroy(&vec_rho_down_left));
  if (ctx->dim == 3) {
    PetscCall(VecDestroy(&vec_rho_back_left));
    PetscCall(VecDestroy(&vec_rho_back_down));
  }
  PetscCall(DMDestroy(&da_vel_avg));
  PetscCall(DMDestroy(&da_p));
  PetscCall(DMDestroy(&da_eta_element));
  PetscCall(DMDestroy(&da_eta_down_left));
  if (ctx->dim == 3) {
    PetscCall(DMDestroy(&da_eta_back_left));
    PetscCall(DMDestroy(&da_eta_back_down));
  }
  PetscCall(DMDestroy(&da_rho_down_left));
  if (ctx->dim == 3) {
    PetscCall(DMDestroy(&da_rho_back_left));
    PetscCall(DMDestroy(&da_rho_back_down));
  }
  PetscCall(VecDestroy(&vel_avg));
  PetscCall(DMDestroy(&dm_vel_avg));
  PetscFunctionReturn(0);
}

/*TEST

   test:
      suffix: direct_umfpack
      requires: suitesparse !complex
      nsize: 1
      args: -dim 2 -coefficients layers -nondimensional 0 -stag_grid_x 12 -stag_grid_y 7 -pc_type lu -pc_factor_mat_solver_type umfpack -ksp_converged_reason

   test:
      suffix: direct_mumps
      requires: mumps !complex
      nsize: 9
      args: -dim 2 -coefficients layers -nondimensional 0 -stag_grid_x 13 -stag_grid_y 8 -pc_type lu -pc_factor_mat_solver_type mumps -ksp_converged_reason

   test:
      suffix: isovisc_nondim_abf_mg
      nsize: 1
      args: -dim 2 -coefficients layers -nondimensional 1 -pc_type fieldsplit -pc_fieldsplit_type schur -ksp_converged_reason -fieldsplit_element_ksp_type preonly  -pc_fieldsplit_detect_saddle_point false -fieldsplit_face_pc_type mg -fieldsplit_face_pc_mg_levels 3 -stag_grid_x 24 -stag_grid_y 24 -fieldsplit_face_pc_mg_galerkin -fieldsplit_face_ksp_converged_reason -ksp_type fgmres -fieldsplit_element_pc_type none -fieldsplit_face_mg_levels_ksp_max_it 6 -pc_fieldsplit_schur_fact_type upper -isoviscous

   test:
      suffix: isovisc_nondim_abf_mg_2
      nsize: 1
      args: -dim 2 -coefficients layers -nondimensional -isoviscous -eta1 1.0 -stag_grid_x 32 -stag_grid_y 32 -ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type upper -build_auxiliary_operator -fieldsplit_element_ksp_type preonly -fieldsplit_element_pc_type jacobi -fieldsplit_face_pc_type mg -fieldsplit_face_pc_mg_levels 3 -fieldsplit_face_pc_mg_galerkin -fieldsplit_face_mg_levels_pc_type jacobi -fieldsplit_face_mg_levels_ksp_type chebyshev -ksp_converged_reason

   test:
      suffix: nondim_abf_mg
      requires: suitesparse !complex
      nsize: 4
      args: -dim 2 -coefficients layers -pc_type fieldsplit -pc_fieldsplit_type schur -ksp_converged_reason -fieldsplit_element_ksp_type preonly  -pc_fieldsplit_detect_saddle_point false -fieldsplit_face_pc_type mg -fieldsplit_face_pc_mg_levels 3 -fieldsplit_face_pc_mg_galerkin -fieldsplit_face_mg_coarse_pc_type redundant -fieldsplit_face_mg_coarse_redundant_pc_type lu -fieldsplit_face_mg_coarse_redundant_pc_factor_mat_solver_type umfpack  -ksp_type fgmres -fieldsplit_element_pc_type none -fieldsplit_face_mg_levels_ksp_max_it 6 -pc_fieldsplit_schur_fact_type upper -nondimensional -eta1 1e-2 -eta2 1.0 -ksp_monitor -fieldsplit_face_mg_levels_pc_type jacobi -fieldsplit_face_mg_levels_ksp_type gmres -fieldsplit_element_pc_type jacobi -pc_fieldsplit_schur_precondition selfp   -stag_grid_x 32 -stag_grid_y 32 -fieldsplit_face_ksp_monitor

   test:
      suffix: nondim_abf_lu
      requires: suitesparse !complex
      nsize: 1
      args: -dim 2 -coefficients layers -pc_type fieldsplit -pc_fieldsplit_type schur -ksp_converged_reason -fieldsplit_element_ksp_type preonly  -pc_fieldsplit_detect_saddle_point false -ksp_type fgmres -fieldsplit_element_pc_type none -pc_fieldsplit_schur_fact_type upper -nondimensional -eta1 1e-2 -eta2 1.0 -isoviscous 0 -ksp_monitor -fieldsplit_element_pc_type jacobi -build_auxiliary_operator -fieldsplit_face_pc_type lu -fieldsplit_face_pc_factor_mat_solver_type umfpack -stag_grid_x 32 -stag_grid_y 32

   test:
      suffix: 3d_nondim_isovisc_abf_mg
      requires: !single
      nsize: 1
      args: -dim 3 -coefficients layers -isoviscous -nondimensional -build_auxiliary_operator -pc_type fieldsplit -pc_fieldsplit_type schur -ksp_converged_reason -fieldsplit_element_ksp_type preonly  -pc_fieldsplit_detect_saddle_point false -fieldsplit_face_pc_type mg -fieldsplit_face_pc_mg_levels 3 -s 16 -fieldsplit_face_pc_mg_galerkin -fieldsplit_face_ksp_converged_reason -ksp_type fgmres -fieldsplit_element_pc_type none -fieldsplit_face_mg_levels_ksp_max_it 6 -pc_fieldsplit_schur_fact_type upper

   test:
      TODO: unstable across systems
      suffix: monolithic
      nsize: 1
      requires: double !complex
      args: -dim {{2 3}separate output} -s 16 -custom_pc_mat -pc_type mg -pc_mg_levels 3 -pc_mg_galerkin -mg_levels_ksp_type gmres -mg_levels_ksp_norm_type unpreconditioned -mg_levels_ksp_max_it 10 -mg_levels_pc_type jacobi -ksp_converged_reason

   test:
      suffix: 3d_nondim_isovisc_sinker_abf_mg
      requires: !complex !single
      nsize: 1
      args: -dim 3 -coefficients sinker -isoviscous -nondimensional -pc_type fieldsplit -pc_fieldsplit_type schur -ksp_converged_reason -fieldsplit_element_ksp_type preonly  -pc_fieldsplit_detect_saddle_point false -fieldsplit_face_pc_type mg -fieldsplit_face_pc_mg_levels 3 -s 16 -fieldsplit_face_pc_mg_galerkin -fieldsplit_face_ksp_converged_reason -ksp_type fgmres -fieldsplit_element_pc_type none -fieldsplit_face_mg_levels_ksp_max_it 6 -pc_fieldsplit_schur_fact_type upper

   test:
      TODO: unstable across systems
      suffix: 3d_nondim_mono_mg_lamemstyle
      nsize: 1
      requires: suitesparse
      args: -dim 3 -coefficients layers -nondimensional -s 16 -custom_pc_mat -pc_type mg -pc_mg_galerkin -pc_mg_levels 2 -mg_levels_ksp_type richardson -mg_levels_pc_type jacobi -mg_levels_ksp_richardson_scale 0.5 -mg_levels_ksp_max_it 20 -mg_coarse_pc_type lu -mg_coarse_pc_factor_mat_solver_type umfpack -ksp_converged_reason

TEST*/
