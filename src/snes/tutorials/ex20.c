static char help[] = "Poisson Problem with finite elements.\n\
This example supports automatic convergence estimation for multilevel solvers\n\
and solver adaptivity.\n\n\n";

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscconvest.h>

/* Next steps:

- Show lowest eigenmodes using SLEPc code from my ex6

- Run CR example from Brannick's slides that looks like semicoarsening
  - Show lowest modes
  - Show CR convergence rate
  - Show CR solution to show non-convergence
  - Refine coarse grid around non-converged dofs
    - Maybe use Barry's "more then Z% above the average" monitor to label bad dofs
    - Mark coarse cells that contain bad dofs
    - Run SBR on coarse grid

- Run Helmholtz example from Gander's writeup

- Run Low Mach example?

- Run subduction example?
*/

typedef struct {
  PetscBool cr;  /* Use compatible relaxation */
} AppCtx;

static PetscErrorCode trig_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  u[0] = 0.0;
  for (d = 0; d < dim; ++d) u[0] += PetscSinReal(2.0*PETSC_PI*x[d]);
  return 0;
}

static void f0_trig_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f0[0] += -4.0*PetscSqr(PETSC_PI)*PetscSinReal(2.0*PETSC_PI*x[d]);
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
  PetscFunctionBeginUser;
  options->cr = PETSC_FALSE;
  PetscOptionsBegin(comm, "", "Poisson Problem Options", "DMPLEX");
  PetscCall(PetscOptionsBool("-cr", "Use compatible relaxarion", "ex20.c", options->cr, &options->cr, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMSetApplicationContext(*dm, user));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupPrimalProblem(DM dm, AppCtx *user)
{
  PetscDS        ds;
  DMLabel        label;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(DMGetLabel(dm, "marker", &label));
  PetscCall(PetscDSSetResidual(ds, 0, f0_trig_u, f1_u));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu));
  PetscCall(PetscDSSetExactSolution(ds, 0, trig_u, user));
  PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) trig_u, NULL, user, NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupDiscretization(DM dm, const char name[], PetscErrorCode (*setup)(DM, AppCtx *), AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe;
  DMPolytopeType ct;
  PetscBool      simplex;
  PetscInt       dim, cStart;
  char           prefix[PETSC_MAX_PATH_LEN];

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;

  PetscCall(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name));
  PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, simplex, name ? prefix : NULL, -1, &fe));
  PetscCall(PetscObjectSetName((PetscObject) fe, name));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject) fe));
  PetscCall(DMCreateDS(dm));
  PetscCall((*setup)(dm, user));
  while (cdm) {
    PetscCall(DMCopyDisc(dm,cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(0);
}

/*
  How to do CR in PETSc:

Loop over PCMG levels, coarse to fine:
  Run smoother for 5 iterates
    At each iterate, solve Inj u_f = u_c with LSQR to 1e-15
    Suppose that e_k = c^k e_0, which means log e_k = log e_0 + k log c
      Fit log of error to look at log c, the slope
      Check R^2 for linearity (1 - square residual / variance)
  Solve exactly
  Prolong to next level
*/

int main(int argc, char **argv)
{
  DM             dm;   /* Problem specification */
  SNES           snes; /* Nonlinear solver */
  Vec            u;    /* Solutions */
  AppCtx         user; /* User-defined work context */

  PetscCall(PetscInitialize(&argc, &argv, NULL,help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  /* Primal system */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(SetupDiscretization(dm, "potential", SetupPrimalProblem, &user));
  PetscCall(DMCreateGlobalVector(dm, &u));
  PetscCall(VecSet(u, 0.0));
  PetscCall(PetscObjectSetName((PetscObject) u, "potential"));
  PetscCall(DMPlexSetSNESLocalFEM(dm, &user, &user, &user));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(SNESSolve(snes, NULL, u));
  PetscCall(SNESGetSolution(snes, &u));
  PetscCall(VecViewFromOptions(u, NULL, "-potential_view"));
  /* Cleanup */
  PetscCall(VecDestroy(&u));
  PetscCall(SNESDestroy(&snes));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 2d_p1_gmg_vcycle_rate
    requires: triangle
    args: -potential_petscspace_degree 1 -dm_plex_box_faces 2,2 -dm_refine_hierarchy 3 \
          -ksp_rtol 5e-10 -ksp_converged_rate -pc_type mg \
            -mg_levels_ksp_max_it 5 -mg_levels_ksp_norm_type preconditioned -mg_levels_ksp_converged_rate \
            -mg_levels_esteig_ksp_type cg \
            -mg_levels_esteig_ksp_max_it 10 \
            -mg_levels_ksp_chebyshev_esteig 0,0.05,0,1.05 \
            -mg_levels_pc_type jacobi

  test:
    suffix: 2d_p1_gmg_vcycle_cr
    TODO: broken
    # cannot MatShift() a MATNORMAL until this MatType inherits from MATSHELL, cf. https://gitlab.com/petsc/petsc/-/issues/972
    requires: triangle
    args: -potential_petscspace_degree 1 -dm_plex_box_faces 2,2 -dm_refine_hierarchy 3 \
          -ksp_rtol 5e-10 -pc_type mg  -pc_mg_adapt_cr \
            -mg_levels_ksp_max_it 5 -mg_levels_ksp_norm_type preconditioned \
            -mg_levels_esteig_ksp_type cg \
            -mg_levels_esteig_ksp_max_it 10 \
            -mg_levels_ksp_chebyshev_esteig 0,0.05,0,1.05 \
            -mg_levels_cr_ksp_max_it 5 -mg_levels_cr_ksp_converged_rate -mg_levels_cr_ksp_converged_rate_type error

  test:
    suffix: 2d_p1_gmg_fcycle_rate
    requires: triangle
    args: -potential_petscspace_degree 1 -dm_plex_box_faces 2,2 -dm_refine_hierarchy 3 \
          -ksp_rtol 5e-10 -ksp_converged_rate -pc_type mg -pc_mg_type full \
            -mg_levels_ksp_max_it 5 -mg_levels_ksp_norm_type preconditioned -mg_levels_ksp_converged_rate \
            -mg_levels_esteig_ksp_type cg \
            -mg_levels_esteig_ksp_max_it 10 \
            -mg_levels_ksp_chebyshev_esteig 0,0.05,0,1.05 \
            -mg_levels_pc_type jacobi
  test:
    suffix: 2d_p1_gmg_vcycle_adapt_rate
    requires: triangle
    args: -petscpartitioner_type simple -potential_petscspace_degree 1 -dm_plex_box_faces 2,2 -dm_refine_hierarchy 3 \
          -ksp_rtol 5e-10 -ksp_converged_rate -pc_type mg \
            -pc_mg_galerkin -pc_mg_adapt_interp_coarse_space harmonic -pc_mg_adapt_interp_n 8 \
            -mg_levels_ksp_max_it 5 -mg_levels_ksp_norm_type preconditioned -mg_levels_ksp_converged_rate \
            -mg_levels_esteig_ksp_type cg \
            -mg_levels_esteig_ksp_max_it 10 \
            -mg_levels_ksp_chebyshev_esteig 0,0.05,0,1.05 \
            -mg_levels_pc_type jacobi
  test:
    suffix: 2d_p1_scalable_rate
    requires: triangle
    args: -potential_petscspace_degree 1 -dm_refine 3 \
      -ksp_type cg -ksp_rtol 1.e-11 -ksp_norm_type unpreconditioned -ksp_converged_rate \
      -pc_type gamg -pc_gamg_esteig_ksp_max_it 10 -pc_gamg_esteig_ksp_type cg \
        -pc_gamg_type agg -pc_gamg_agg_nsmooths 1 \
        -pc_gamg_coarse_eq_limit 1000 \
        -pc_gamg_threshold 0.05 \
        -pc_gamg_threshold_scale .0 \
        -mg_levels_ksp_type chebyshev -mg_levels_ksp_norm_type preconditioned -mg_levels_ksp_converged_rate \
        -mg_levels_ksp_max_it 5                                                \
      -matptap_via scalable

TEST*/
