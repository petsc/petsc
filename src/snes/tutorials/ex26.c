static char help[] = "'Good Cop' Helmholtz Problem in 2d and 3d with finite elements.\n\
We solve the 'Good Cop' Helmholtz problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\
This example supports automatic convergence estimation\n\
and coarse space adaptivity.\n\n\n";

/*
   The model problem:
      Solve "Good Cop" Helmholtz equation on the unit square: (0,1) x (0,1)
          - \Delta u + u = f,
           where \Delta = Laplace operator
      Dirichlet b.c.'s on all sides

*/

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscconvest.h>

typedef struct {
  PetscBool trig; /* Use trig function as exact solution */
} AppCtx;

/*For Primal Problem*/
static void g0_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                   const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                   const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                   PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g0[0] = 1.0;
}

static void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d*dim+d] = 1.0;
}

static PetscErrorCode trig_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  *u = 0.0;
  for (d = 0; d < dim; ++d) *u += PetscSinReal(2.0*PETSC_PI*x[d]);
  return 0;
}

static PetscErrorCode quad_u(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  *u = 1.0;
  for (d = 0; d < dim; ++d) *u += (d+1)*PetscSqr(x[d]);
  return 0;
}

static void f0_trig_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  f0[0] += u[0];
  for (d = 0; d < dim; ++d) f0[0] -= 4.0*PetscSqr(PETSC_PI)*PetscSinReal(2.0*PETSC_PI*x[d]) + PetscSinReal(2.0*PETSC_PI*x[d]);
}

static void f0_quad_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                      const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                      const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                      PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
    PetscInt d;
    switch (dim) {
        case 1:
            f0[0] = 1.0;
            break;
        case 2:
            f0[0] = 5.0;
            break;
        case 3:
            f0[0] = 11.0;
            break;
        default:
            f0[0] = 5.0;
            break;
    }
    f0[0] += u[0];
    for (d = 0; d < dim; ++d) f0[0] -= (d+1)*PetscSqr(x[d]);
}

static void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static PetscErrorCode ProcessOptions(DM dm, AppCtx *options)
{
  MPI_Comm       comm;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  CHKERRQ(DMGetDimension(dm, &dim));
  options->trig = PETSC_FALSE;

  ierr = PetscOptionsBegin(comm, "", "Helmholtz Problem Options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsBool("-exact_trig", "Use trigonometric exact solution (better for more complex finite elements)", "ex26.c", options->trig, &options->trig, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));

  CHKERRQ(PetscObjectSetName((PetscObject) *dm, "Mesh"));
  CHKERRQ(DMSetApplicationContext(*dm, user));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));

  PetscFunctionReturn(0);
}

static PetscErrorCode SetupPrimalProblem(DM dm, AppCtx *user)
{
  PetscDS        ds;
  DMLabel        label;
  const PetscInt id = 1;

  PetscFunctionBeginUser;
  CHKERRQ(DMGetDS(dm, &ds));
  CHKERRQ(DMGetLabel(dm, "marker", &label));
  if (user->trig) {
    CHKERRQ(PetscDSSetResidual(ds, 0, f0_trig_u, f1_u));
    CHKERRQ(PetscDSSetJacobian(ds, 0, 0, g0_uu, NULL, NULL, g3_uu));
    CHKERRQ(PetscDSSetExactSolution(ds, 0, trig_u, user));
    CHKERRQ(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) trig_u, NULL, user, NULL));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Trig Exact Solution\n"));
  } else {
    CHKERRQ(PetscDSSetResidual(ds, 0, f0_quad_u, f1_u));
    CHKERRQ(PetscDSSetJacobian(ds, 0, 0, g0_uu, NULL, NULL, g3_uu));
    CHKERRQ(PetscDSSetExactSolution(ds, 0, quad_u, user));
    CHKERRQ(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) quad_u, NULL, user, NULL));
  }
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
  CHKERRQ(DMGetDimension(dm, &dim));

  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  CHKERRQ(DMPlexGetCellType(dm, cStart, &ct));
  simplex = DMPolytopeTypeGetNumVertices(ct) == DMPolytopeTypeGetDim(ct)+1 ? PETSC_TRUE : PETSC_FALSE;
  /* Create finite element */
  CHKERRQ(PetscSNPrintf(prefix, PETSC_MAX_PATH_LEN, "%s_", name));
  CHKERRQ(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, name ? prefix : NULL, -1, &fe));
  CHKERRQ(PetscObjectSetName((PetscObject) fe, name));
  /* Set discretization and boundary conditions for each mesh */
  CHKERRQ(DMSetField(dm, 0, NULL, (PetscObject) fe));
  CHKERRQ(DMCreateDS(dm));
  CHKERRQ((*setup)(dm, user));
  while (cdm) {
    CHKERRQ(DMCopyDisc(dm,cdm));
    CHKERRQ(DMGetCoarseDM(cdm, &cdm));
  }
  CHKERRQ(PetscFEDestroy(&fe));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
    DM             dm;   /* Problem specification */
    PetscDS        ds;
    SNES           snes; /* Nonlinear solver */
    Vec            u;    /* Solutions */
    AppCtx         user; /* User-defined work context */

    CHKERRQ(PetscInitialize(&argc, &argv, NULL, help));
    /* Primal system */
    CHKERRQ(SNESCreate(PETSC_COMM_WORLD, &snes));
    CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
    CHKERRQ(ProcessOptions(dm, &user));
    CHKERRQ(SNESSetDM(snes, dm));
    CHKERRQ(SetupDiscretization(dm, "potential", SetupPrimalProblem, &user));
    CHKERRQ(DMCreateGlobalVector(dm, &u));
    CHKERRQ(VecSet(u, 0.0));
    CHKERRQ(PetscObjectSetName((PetscObject) u, "potential"));
    CHKERRQ(DMPlexSetSNESLocalFEM(dm, &user, &user, &user));
    CHKERRQ(SNESSetFromOptions(snes));
    CHKERRQ(DMSNESCheckFromOptions(snes, u));

    /*Looking for field error*/
    PetscInt Nfields;
    CHKERRQ(DMGetDS(dm, &ds));
    CHKERRQ(PetscDSGetNumFields(ds, &Nfields));
    CHKERRQ(SNESSolve(snes, NULL, u));
    CHKERRQ(SNESGetSolution(snes, &u));
    CHKERRQ(VecViewFromOptions(u, NULL, "-potential_view"));

    /* Cleanup */
    CHKERRQ(VecDestroy(&u));
    CHKERRQ(SNESDestroy(&snes));
    CHKERRQ(DMDestroy(&dm));
    CHKERRQ(PetscFinalize());
    return 0;
}

/*TEST
test:
  # L_2 convergence rate: 1.9
  suffix: 2d_p1_conv
  requires: triangle
  args: -potential_petscspace_degree 1 -snes_convergence_estimate -dm_refine 2 -convest_num_refine 3 -pc_type lu
test:
  # L_2 convergence rate: 1.9
  suffix: 2d_q1_conv
  args: -dm_plex_simplex 0 -potential_petscspace_degree 1 -snes_convergence_estimate -dm_refine 2 -convest_num_refine 3 -pc_type lu
test:
  # Using -convest_num_refine 3 we get L_2 convergence rate: -1.5
  suffix: 3d_p1_conv
  requires: ctetgen
  args: -dm_plex_dim 3 -dm_refine 2 -potential_petscspace_degree 1 -snes_convergence_estimate -convest_num_refine 1 -pc_type lu
test:
  # Using -dm_refine 2 -convest_num_refine 3 we get L_2 convergence rate: -1.2
  suffix: 3d_q1_conv
  args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_refine 2 -potential_petscspace_degree 1 -snes_convergence_estimate -convest_num_refine 1 -pc_type lu
test:
  # L_2 convergence rate: 1.9
  suffix: 2d_p1_trig_conv
  requires: triangle
  args: -potential_petscspace_degree 1 -snes_convergence_estimate -dm_refine 2 -convest_num_refine 3 -pc_type lu -exact_trig
test:
  # L_2 convergence rate: 1.9
  suffix: 2d_q1_trig_conv
  args: -dm_plex_simplex 0 -potential_petscspace_degree 1 -snes_convergence_estimate -dm_refine 2 -convest_num_refine 3 -pc_type lu -exact_trig
test:
  # Using -convest_num_refine 3 we get L_2 convergence rate: -1.5
  suffix: 3d_p1_trig_conv
  requires: ctetgen
  args: -dm_plex_dim 3 -dm_refine 2 -potential_petscspace_degree 1 -snes_convergence_estimate -convest_num_refine 1 -pc_type lu -exact_trig
test:
  # Using -dm_refine 2 -convest_num_refine 3 we get L_2 convergence rate: -1.2
  suffix: 3d_q1_trig_conv
  args: -dm_plex_dim 3 -dm_plex_simplex 0 -dm_refine 2 -potential_petscspace_degree 1 -snes_convergence_estimate -convest_num_refine 1 -pc_type lu -exact_trig
test:
  suffix: 2d_p1_gmg_vcycle
  requires: triangle
  args: -potential_petscspace_degree 1 -dm_plex_box_faces 2,2 -dm_refine_hierarchy 3 \
    -ksp_type cg -ksp_rtol 1e-10 -pc_type mg \
    -mg_levels_ksp_max_it 1 \
    -mg_levels_pc_type jacobi -snes_monitor -ksp_monitor
test:
  suffix: 2d_p1_gmg_fcycle
  requires: triangle
  args: -potential_petscspace_degree 1 -dm_plex_box_faces 2,2 -dm_refine_hierarchy 3 \
    -ksp_type cg -ksp_rtol 1e-10 -pc_type mg -pc_mg_type full \
    -mg_levels_ksp_max_it 2 \
    -mg_levels_pc_type jacobi -snes_monitor -ksp_monitor
test:
  suffix: 2d_p1_gmg_vcycle_trig
  requires: triangle
  args: -exact_trig -potential_petscspace_degree 1 -dm_plex_box_faces 2,2 -dm_refine_hierarchy 3 \
    -ksp_type cg -ksp_rtol 1e-10 -pc_type mg \
    -mg_levels_ksp_max_it 1 \
    -mg_levels_pc_type jacobi -snes_monitor -ksp_monitor
TEST*/
