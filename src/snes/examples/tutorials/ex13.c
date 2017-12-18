static char help[] = "Poisson Problem in 2d and 3d with finite elements.\n\
We solve the Poisson problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\
This example supports automatic convergence estimation\n\
and eventually adaptivity.\n\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscconvest.h>

typedef struct {
  /* Domain and mesh definition */
  PetscInt  dim;               /* The topological mesh dimension */
  PetscBool simplex;           /* Simplicial mesh */
  PetscInt  cells[3];          /* The initial domain division */
  PetscBool adjoint;           /* Solve the adjoint problem */
} AppCtx;

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  *u = 0.0;
  return 0;
}

static PetscErrorCode trig_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  *u = 0.0;
  for (d = 0; d < dim; ++d) *u += PetscSinReal(2.0*PETSC_PI*x[d]);
  return 0;
}

/* Compute integral of (residual of solution)*(adjoint solution - projection of adjoint solution) */
static void obj_error_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                        const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                        const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                        PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar obj[])
{
  obj[0] = a[aOff[0]]*(u[0] - a[aOff[1]]);
}

static void f0_trig_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f0[0] += -4.0*PetscSqr(PETSC_PI)*PetscSinReal(2.0*PETSC_PI*x[d]);
}

static void f0_unity_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                       const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                       const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                       PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 1.0;
}

static void f0_identityaux_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                             const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                             const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                             PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = a[0];
}

static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d*dim+d] = 1.0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       n = 3;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim      = 2;
  options->cells[0] = 1;
  options->cells[1] = 1;
  options->cells[2] = 1;
  options->simplex  = PETSC_TRUE;
  options->adjoint  = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Poisson Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex13.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-cells", "The initial mesh division", "ex13.c", options->cells, &n, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "Simplicial (true) or tensor (false) mesh", "ex13.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-adjoint", "Solve the adjoint problem", "ex13.c", options->adjoint, &options->adjoint, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* Create box mesh */
  ierr = DMPlexCreateBoxMesh(comm, user->dim, user->simplex, user->cells, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  /* Distribute mesh over processes */
  {
    DM               dmDist = NULL;
    PetscPartitioner part;

    ierr = DMPlexGetPartitioner(*dm, &part);CHKERRQ(ierr);
    ierr = PetscPartitionerSetFromOptions(part);CHKERRQ(ierr);
    ierr = DMPlexDistribute(*dm, 0, NULL, &dmDist);CHKERRQ(ierr);
    if (dmDist) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = dmDist;
    }
  }
  /* TODO: This should be pulled into the library */
  {
    char      convType[256];
    PetscBool flg;

    ierr = PetscOptionsBegin(comm, "", "Mesh conversion options", "DMPLEX");CHKERRQ(ierr);
    ierr = PetscOptionsFList("-dm_plex_convert_type","Convert DMPlex to another format","ex12",DMList,DMPLEX,convType,256,&flg);CHKERRQ(ierr);
    ierr = PetscOptionsEnd();
    if (flg) {
      DM dmConv;

      ierr = DMConvert(*dm,convType,&dmConv);CHKERRQ(ierr);
      if (dmConv) {
        ierr = DMDestroy(dm);CHKERRQ(ierr);
        *dm  = dmConv;
      }
    }
  }
  /* TODO: This should be pulled into the library */
  ierr = DMLocalizeCoordinates(*dm);CHKERRQ(ierr);

  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMSetApplicationContext(*dm, user);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  /* TODO: Add a hierachical viewer */
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupPrimalProblem(PetscDS prob, AppCtx *user)
{
  const PetscInt id = 1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscDSSetResidual(prob, 0, f0_trig_u, f1_u);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "wall", "marker", 0, 0, NULL, (void (*)(void)) trig_u, 1, &id, user);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(prob, 0, trig_u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupAdjointProblem(PetscDS prob, AppCtx *user)
{
  const PetscInt id = 1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscDSSetResidual(prob, 0, f0_unity_u, f1_u);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
  ierr = PetscDSAddBoundary(prob, DM_BC_ESSENTIAL, "wall", "marker", 0, 0, NULL, (void (*)(void)) zero, 1, &id, user);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupErrorProblem(PetscDS prob, AppCtx *user)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscDSSetObjective(prob, 0, obj_error_u);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, const char name[], PetscErrorCode (*setup)(PetscDS, AppCtx *), AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  PetscDS        prob;
  char           prefix[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* Create finite element */
  ierr = PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(dm, user->dim, 1, user->simplex, name ? prefix : NULL, -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, name);CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe);CHKERRQ(ierr);
  ierr = (*setup)(prob, user);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMSetDS(cdm, prob);CHKERRQ(ierr);
    /* TODO: Check whether the boundary of coarse meshes is marked */
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;   /* Problem specification */
  SNES           snes; /* Nonlinear solver */
  Vec            u;    /* Solutions */
  AppCtx         user; /* User-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  /* Primal system */
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = SetupDiscretization(dm, "potential", SetupPrimalProblem, &user);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = VecSet(u, 0.0);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "potential");CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(dm, &user, &user, &user);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);
  ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
  ierr = SNESGetSolution(snes, &u);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-potential_view");CHKERRQ(ierr);
  /* Adjoint system */
  if (user.adjoint) {
    DM   dmAdj;
    SNES snesAdj;
    Vec  uAdj;

    ierr = SNESCreate(PETSC_COMM_WORLD, &snesAdj);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject) snesAdj, "adjoint_");CHKERRQ(ierr);
    ierr = DMClone(dm, &dmAdj);CHKERRQ(ierr);
    ierr = SNESSetDM(snesAdj, dmAdj);CHKERRQ(ierr);
    ierr = SetupDiscretization(dmAdj, "adjoint", SetupAdjointProblem, &user);CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(dmAdj, &uAdj);CHKERRQ(ierr);
    ierr = VecSet(uAdj, 0.0);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) uAdj, "adjoint");CHKERRQ(ierr);
    ierr = DMPlexSetSNESLocalFEM(dmAdj, &user, &user, &user);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(snesAdj);CHKERRQ(ierr);
    ierr = SNESSolve(snesAdj, NULL, uAdj);CHKERRQ(ierr);
    ierr = SNESGetSolution(snesAdj, &uAdj);CHKERRQ(ierr);
    ierr = VecViewFromOptions(uAdj, NULL, "-adjoint_view");CHKERRQ(ierr);
    /* Error representation */
    {
      DM        dmErr, dmErrAux, dms[2];
      Vec       errorEst, errorL2, uErr, uErrLoc, uAdjLoc, uAdjProj;
      IS       *subis;
      PetscReal errorEstTot, errorL2Norm, errorL2Tot;
      PetscInt  N, i;
      PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *) = {trig_u};
      void (*identity[1])(PetscInt, PetscInt, PetscInt,
                          const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                          const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                          PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]) = {f0_identityaux_u};
      void            *ctxs[1] = {0};

      ctxs[0] = &user;
      ierr = DMClone(dm, &dmErr);CHKERRQ(ierr);
      ierr = SetupDiscretization(dmErr, "error", SetupErrorProblem, &user);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(dmErr, &errorEst);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(dmErr, &errorL2);CHKERRQ(ierr);
      /*   Compute auxiliary data (solution and projection of adjoint solution) */
      ierr = DMGetLocalVector(dmAdj, &uAdjLoc);CHKERRQ(ierr);
      ierr = DMGlobalToLocalBegin(dmAdj, uAdj, INSERT_VALUES, uAdjLoc);CHKERRQ(ierr);
      ierr = DMGlobalToLocalEnd(dmAdj, uAdj, INSERT_VALUES, uAdjLoc);CHKERRQ(ierr);
      ierr = DMGetGlobalVector(dm, &uAdjProj);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) dm, "dmAux", (PetscObject) dmAdj);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) dm, "A", (PetscObject) uAdjLoc);CHKERRQ(ierr);
      ierr = DMProjectField(dm, 0.0, u, identity, INSERT_VALUES, uAdjProj);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) dm, "dmAux", NULL);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) dm, "A", NULL);CHKERRQ(ierr);
      ierr = DMRestoreLocalVector(dmAdj, &uAdjLoc);CHKERRQ(ierr);
      /*   Attach auxiliary data */
      dms[0] = dm; dms[1] = dm;
      ierr = DMCreateSuperDM(dms, 2, &subis, &dmErrAux);CHKERRQ(ierr);
      if (0) {
        PetscSection sec;

        ierr = DMGetDefaultSection(dms[0], &sec);CHKERRQ(ierr);
        ierr = PetscSectionView(sec, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        ierr = DMGetDefaultSection(dms[1], &sec);CHKERRQ(ierr);
        ierr = PetscSectionView(sec, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
        ierr = DMGetDefaultSection(dmErrAux, &sec);CHKERRQ(ierr);
        ierr = PetscSectionView(sec, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      }
      ierr = DMViewFromOptions(dmErrAux, NULL, "-dm_err_view");CHKERRQ(ierr);
      ierr = ISViewFromOptions(subis[0], NULL, "-super_is_view");CHKERRQ(ierr);
      ierr = ISViewFromOptions(subis[1], NULL, "-super_is_view");CHKERRQ(ierr);
      ierr = DMGetGlobalVector(dmErrAux, &uErr);CHKERRQ(ierr);
      ierr = VecViewFromOptions(u, NULL, "-map_vec_view");CHKERRQ(ierr);
      ierr = VecViewFromOptions(uAdjProj, NULL, "-map_vec_view");CHKERRQ(ierr);
      ierr = VecViewFromOptions(uErr, NULL, "-map_vec_view");CHKERRQ(ierr);
      ierr = VecISCopy(uErr, subis[0], SCATTER_FORWARD, u);CHKERRQ(ierr);
      ierr = VecISCopy(uErr, subis[1], SCATTER_FORWARD, uAdjProj);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(dm, &uAdjProj);CHKERRQ(ierr);
      for (i = 0; i < 2; ++i) {ierr = ISDestroy(&subis[i]);CHKERRQ(ierr);}
      ierr = PetscFree(subis);CHKERRQ(ierr);
      ierr = DMGetLocalVector(dmErrAux, &uErrLoc);CHKERRQ(ierr);
      ierr = DMGlobalToLocalBegin(dm, uErr, INSERT_VALUES, uErrLoc);CHKERRQ(ierr);
      ierr = DMGlobalToLocalEnd(dm, uErr, INSERT_VALUES, uErrLoc);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(dmErrAux, &uErr);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) dmErr, "dmAux", (PetscObject) dmErrAux);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) dmErr, "A", (PetscObject) uErrLoc);CHKERRQ(ierr);
      /*   Compute cellwise error estimate */
      ierr = VecSet(errorEst, 0.0);CHKERRQ(ierr);
      ierr = DMPlexComputeCellwiseIntegralFEM(dmErr, uAdj, errorEst, &user);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) dmErr, "dmAux", NULL);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject) dmErr, "A", NULL);CHKERRQ(ierr);
      ierr = DMRestoreLocalVector(dmErrAux, &uErrLoc);CHKERRQ(ierr);
      ierr = DMDestroy(&dmErrAux);CHKERRQ(ierr);
      /*   Plot cellwise error vector */
      ierr = VecViewFromOptions(errorEst, NULL, "-error_view");CHKERRQ(ierr);
      /*   Compute ratio of estimate (sum over cells) with actual L_2 error */
      ierr = DMComputeL2Diff(dm, 0.0, funcs, ctxs, u, &errorL2Norm);CHKERRQ(ierr);
      ierr = DMPlexComputeL2DiffVec(dm, 0.0, funcs, ctxs, u, errorL2);CHKERRQ(ierr);
      ierr = VecViewFromOptions(errorL2, NULL, "-l2_error_view");CHKERRQ(ierr);
      ierr = VecNorm(errorL2,  NORM_INFINITY, &errorL2Tot);CHKERRQ(ierr);
      ierr = VecNorm(errorEst, NORM_INFINITY, &errorEstTot);CHKERRQ(ierr);
      ierr = VecGetSize(errorEst, &N);CHKERRQ(ierr);
      ierr = VecPointwiseDivide(errorEst, errorEst, errorL2);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) errorEst, "Error ratio");CHKERRQ(ierr);
      ierr = VecViewFromOptions(errorEst, NULL, "-error_ratio_view");CHKERRQ(ierr);
      ierr = PetscPrintf(PETSC_COMM_WORLD, "N: %D L2 error: %g Error Ratio: %g/%g = %g\n", N, (double) errorL2Norm, (double) errorEstTot, (double) PetscSqrtReal(errorL2Tot), (double) errorEstTot/PetscSqrtReal(errorL2Tot));CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(dmErr, &errorEst);CHKERRQ(ierr);
      ierr = DMRestoreGlobalVector(dmErr, &errorL2);CHKERRQ(ierr);
      ierr = DMDestroy(&dmErr);CHKERRQ(ierr);
    }
    ierr = DMDestroy(&dmAdj);CHKERRQ(ierr);
    ierr = VecDestroy(&uAdj);CHKERRQ(ierr);
    ierr = SNESDestroy(&snesAdj);CHKERRQ(ierr);
  }
  /* Cleanup */
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 2d_p1_0
    requires: triangle
    args: -potential_petscspace_order 1 -dm_refine 2 -num_refine 4 -snes_convergence_estimate
  test:
    suffix: 2d_p2_0
    requires: triangle
    args: -potential_petscspace_order 2 -dm_refine 2 -num_refine 4 -snes_convergence_estimate
  test:
    suffix: 2d_p3_0
    requires: triangle
    args: -potential_petscspace_order 3 -dm_refine 2 -num_refine 4 -snes_convergence_estimate
  test:
    suffix: 2d_q1_0
    args: -simplex 0 -potential_petscspace_order 1 -dm_refine 2 -num_refine 4 -snes_convergence_estimate
  test:
    suffix: 2d_q2_0
    args: -simplex 0 -potential_petscspace_order 2 -dm_refine 2 -num_refine 4 -snes_convergence_estimate
  test:
    suffix: 2d_q3_0
    args: -simplex 0 -potential_petscspace_order 3 -dm_refine 2 -num_refine 4 -snes_convergence_estimate
  test:
    suffix: 3d_p1_0
    requires: ctetgen
    args: -dim 3 -potential_petscspace_order 1 -num_refine 4 -snes_convergence_estimate
  test:
    suffix: 3d_p2_0
    requires: ctetgen
    args: -dim 3 -potential_petscspace_order 2 -num_refine 4 -snes_convergence_estimate
  test:
    suffix: 3d_p3_0
    requires: ctetgen
    args: -dim 3 -potential_petscspace_order 3 -dm_refine 1 -num_refine 3 -snes_convergence_estimate
  test:
    suffix: 3d_q1_0
    args: -dim 3 -simplex 0 -potential_petscspace_order 1 -dm_refine 1 -num_refine 3 -snes_convergence_estimate
  test:
    suffix: 3d_q2_0
    args: -dim 3 -simplex 0 -potential_petscspace_order 2 -dm_refine 2 -num_refine 2 -snes_convergence_estimate
  test:
    suffix: 3d_q3_0
    args: -dim 3 -simplex 0 -potential_petscspace_order 3 -num_refine 2 -snes_convergence_estimate
  test:
    suffix: 2d_p1_adj_0
    requires: triangle
    args: -potential_petscspace_order 1 -dm_refine 2 -adjoint -adjoint_petscspace_order 1 -error_petscspace_order 0

TEST*/
