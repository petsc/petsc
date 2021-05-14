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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->cr = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Poisson Problem Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsBool("-cr", "Use compatible relaxarion", "ex20.c", options->cr, &options->cr, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(*dm, user);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode SetupPrimalProblem(DM dm, AppCtx *user)
{
  PetscDS        ds;
  DMLabel        label;
  const PetscInt id = 1;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "marker", &label);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(ds, 0, f0_trig_u, f1_u);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, g3_uu);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(ds, 0, trig_u, user);CHKERRQ(ierr);
  ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) trig_u, NULL, user, NULL);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetCellType(dm, cStart, &ct);CHKERRQ(ierr);
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;

  ierr = PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, simplex, name ? prefix : NULL, -1, &fe);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe, name);CHKERRQ(ierr);
  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = (*setup)(dm, user);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMCopyDisc(dm,cdm);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
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
  /* Cleanup */
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
    requires: triangle bamg
    args: -potential_petscspace_degree 1 -dm_plex_box_faces 2,2 -dm_refine_hierarchy 3 \
          -ksp_rtol 5e-10 -ksp_converged_rate -pc_type mg \
            -pc_mg_galerkin -pc_mg_adapt_interp -pc_mg_adapt_interp_coarse_space harmonic -pc_mg_adapt_interp_n 8 \
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
      -pc_type gamg \
        -pc_gamg_type agg -pc_gamg_agg_nsmooths 1 \
        -pc_gamg_coarse_eq_limit 1000 \
        -pc_gamg_square_graph 1 \
        -pc_gamg_threshold 0.05 \
        -pc_gamg_threshold_scale .0 \
      -mg_levels_ksp_type chebyshev -mg_levels_ksp_norm_type preconditioned -mg_levels_ksp_converged_rate \
        -mg_levels_ksp_max_it 5 \
        -mg_levels_esteig_ksp_type cg \
        -mg_levels_esteig_ksp_max_it 10 \
        -mg_levels_ksp_chebyshev_esteig 0,0.05,0,1.05 \
        -mg_levels_pc_type jacobi \
      -matptap_via scalable

TEST*/
