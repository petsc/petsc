static char help[] = "Poiseuille Flow in 2d and 3d channels with finite elements.\n\
We solve the Poiseuille flow problem in a rectangular\n\
domain, using a parallel unstructured mesh (DMPLEX) to discretize it.\n\n\n";

/*F
A Poiseuille flow is a steady-state isoviscous Stokes flow in a pipe of constant cross-section. We discretize using the
finite element method on an unstructured mesh. The weak form equations are
\begin{align*}
  < \nabla v, \nu (\nabla u + {\nabla u}^T) > - < \nabla\cdot v, p > + < v, \Delta \hat n >_{\Gamma_o} = 0
  < q, \nabla\cdot u >                                                                                 = 0
\end{align*}
where $\nu$ is the kinematic viscosity, $\Delta$ is the pressure drop per unit length, assuming that pressure is 0 on
the left edge, and $\Gamma_o$ is the outlet boundary at the right edge of the pipe. The normal velocity will be zero at
the wall, but we will allow a fixed tangential velocity $u_0$.

In order to test our global to local basis transformation, we will allow the pipe to be at an angle $\alpha$ to the
coordinate axes.

For visualization, use

  -dm_view hdf5:$PWD/sol.h5 -sol_vec_view hdf5:$PWD/sol.h5::append -exact_vec_view hdf5:$PWD/sol.h5::append
F*/

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>
#include <petscbag.h>

typedef struct {
  PetscReal Delta; /* Pressure drop per unit length */
  PetscReal nu;    /* Kinematic viscosity */
  PetscReal u_0;   /* Tangential velocity at the wall */
  PetscReal alpha; /* Angle of pipe wall to x-axis */
} Parameter;

typedef struct {
  PetscBag bag;    /* Holds problem parameters */
} AppCtx;

/*
  In 2D, plane Poiseuille flow has exact solution:

    u = \Delta/(2 \nu) y (1 - y) + u_0
    v = 0
    p = -\Delta x
    f = 0

  so that

    -\nu \Delta u + \nabla p + f = <\Delta, 0> + <-\Delta, 0> + <0, 0> = 0
    \nabla \cdot u               = 0 + 0                               = 0

  In 3D we use exact solution:

    u = \Delta/(4 \nu) (y (1 - y) + z (1 - z)) + u_0
    v = 0
    w = 0
    p = -\Delta x
    f = 0

  so that

    -\nu \Delta u + \nabla p + f = <Delta, 0, 0> + <-Delta, 0, 0> + <0, 0, 0> = 0
    \nabla \cdot u               = 0 + 0 + 0                                  = 0

  Note that these functions use coordinates X in the global (rotated) frame
*/
PetscErrorCode quadratic_u(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  Parameter *param = (Parameter *) ctx;
  PetscReal  Delta = param->Delta;
  PetscReal  nu    = param->nu;
  PetscReal  u_0   = param->u_0;
  PetscReal  fac   = (PetscReal) (dim - 1);
  PetscInt   d;

  u[0] = u_0;
  for (d = 1; d < dim; ++d) u[0] += Delta/(fac * 2.0*nu) * X[d] * (1.0 - X[d]);
  for (d = 1; d < dim; ++d) u[d]  = 0.0;
  return 0;
}

PetscErrorCode linear_p(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *p, void *ctx)
{
  Parameter *param = (Parameter *) ctx;
  PetscReal  Delta = param->Delta;

  p[0] = -Delta * X[0];
  return 0;
}

PetscErrorCode wall_velocity(PetscInt dim, PetscReal time, const PetscReal X[], PetscInt Nf, PetscScalar *u, void *ctx)
{
  Parameter *param = (Parameter *) ctx;
  PetscReal  u_0   = param->u_0;
  PetscInt   d;

  u[0] = u_0;
  for (d = 1; d < dim; ++d) u[d] = 0.0;
  return 0;
}

/* gradU[comp*dim+d] = {u_x, u_y, v_x, v_y} or {u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z}
   u[Ncomp]          = {p} */
void f1_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  const PetscReal nu = PetscRealPart(constants[1]);
  const PetscInt  Nc = dim;
  PetscInt        c, d;

  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) {
      /* f1[c*dim+d] = 0.5*nu*(u_x[c*dim+d] + u_x[d*dim+c]); */
      f1[c*dim+d] = nu*u_x[c*dim+d];
    }
    f1[c*dim+c] -= u[uOff[1]];
  }
}

/* gradU[comp*dim+d] = {u_x, u_y, v_x, v_y} or {u_x, u_y, u_z, v_x, v_y, v_z, w_x, w_y, w_z} */
void f0_p(PetscInt dim, PetscInt Nf, PetscInt NfAux,
          const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
          const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
          PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;
  for (d = 0, f0[0] = 0.0; d < dim; ++d) f0[0] += u_x[d*dim+d];
}

/* Residual functions are in reference coordinates */
static void f0_bd_u(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                    PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscReal Delta = PetscRealPart(constants[0]);
  PetscReal       alpha = PetscRealPart(constants[3]);
  PetscReal       X     = PetscCosReal(alpha)*x[0] + PetscSinReal(alpha)*x[1];
  PetscInt        d;

  for (d = 0; d < dim; ++d) {
    f0[d] = -Delta * X * n[d];
  }
}

/* < q, \nabla\cdot u >
   NcompI = 1, NcompJ = dim */
void g1_pu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g1[d*dim+d] = 1.0; /* \frac{\partial\phi^{u_d}}{\partial x_d} */
}

/* -< \nabla\cdot v, p >
    NcompI = dim, NcompJ = 1 */
void g2_up(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g2[d*dim+d] = -1.0; /* \frac{\partial\psi^{u_d}}{\partial x_d} */
}

/* < \nabla v, \nabla u + {\nabla u}^T >
   This just gives \nabla u, give the perdiagonal for the transpose */
void g3_uu(PetscInt dim, PetscInt Nf, PetscInt NfAux,
           const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
           const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
           PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  const PetscReal nu = PetscRealPart(constants[1]);
  const PetscInt  Nc = dim;
  PetscInt        c, d;

  for (c = 0; c < Nc; ++c) {
    for (d = 0; d < dim; ++d) {
      g3[((c*Nc+c)*dim+d)*dim+d] = nu;
    }
  }
}

static PetscErrorCode SetupParameters(AppCtx *user)
{
  PetscBag       bag;
  Parameter     *p;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  /* setup PETSc parameter bag */
  ierr = PetscBagGetData(user->bag, (void **) &p);CHKERRQ(ierr);
  ierr = PetscBagSetName(user->bag, "par", "Poiseuille flow parameters");CHKERRQ(ierr);
  bag  = user->bag;
  ierr = PetscBagRegisterReal(bag, &p->Delta, 1.0, "Delta", "Pressure drop per unit length");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &p->nu,    1.0, "nu",    "Kinematic viscosity");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &p->u_0,   0.0, "u_0",   "Tangential velocity at the wall");CHKERRQ(ierr);
  ierr = PetscBagRegisterReal(bag, &p->alpha, 0.0, "alpha", "Angle of pipe wall to x-axis");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  {
    Parameter   *param;
    Vec          coordinates;
    PetscScalar *coords;
    PetscReal    alpha;
    PetscInt     cdim, N, bs, i;

    ierr = DMGetCoordinateDim(*dm, &cdim);CHKERRQ(ierr);
    ierr = DMGetCoordinates(*dm, &coordinates);CHKERRQ(ierr);
    ierr = VecGetLocalSize(coordinates, &N);CHKERRQ(ierr);
    ierr = VecGetBlockSize(coordinates, &bs);CHKERRQ(ierr);
    if (bs != cdim) SETERRQ2(comm, PETSC_ERR_ARG_WRONG, "Invalid coordinate blocksize %D != embedding dimension %D", bs, cdim);
    ierr = VecGetArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);
    alpha = param->alpha;
    for (i = 0; i < N; i += cdim) {
      PetscScalar x = coords[i+0];
      PetscScalar y = coords[i+1];

      coords[i+0] = PetscCosReal(alpha)*x - PetscSinReal(alpha)*y;
      coords[i+1] = PetscSinReal(alpha)*x + PetscCosReal(alpha)*y;
    }
    ierr = VecRestoreArray(coordinates, &coords);CHKERRQ(ierr);
    ierr = DMSetCoordinates(*dm, coordinates);CHKERRQ(ierr);
  }
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetupProblem(DM dm, AppCtx *user)
{
  PetscDS        ds;
  PetscWeakForm  wf;
  DMLabel        label;
  Parameter     *ctx;
  PetscInt       id, bd;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscBagGetData(user->bag, (void **) &ctx);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(ds, 0, NULL, f1_u);CHKERRQ(ierr);
  ierr = PetscDSSetResidual(ds, 1, f0_p, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(ds, 0, 0, NULL, NULL,  NULL,  g3_uu);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(ds, 0, 1, NULL, NULL,  g2_up, NULL);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(ds, 1, 0, NULL, g1_pu, NULL,  NULL);CHKERRQ(ierr);

  id   = 2;
  ierr = DMGetLabel(dm, "marker", &label);CHKERRQ(ierr);
  ierr = DMAddBoundary(dm, DM_BC_NATURAL, "right wall", label, 1, &id, 0, 0, NULL, NULL, NULL, ctx, &bd);CHKERRQ(ierr);
  ierr = PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = PetscWeakFormSetIndexBdResidual(wf, label, id, 0, 0, f0_bd_u, 0, NULL);CHKERRQ(ierr);
  /* Setup constants */
  {
    Parameter  *param;
    PetscScalar constants[4];

    ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);

    constants[0] = param->Delta;
    constants[1] = param->nu;
    constants[2] = param->u_0;
    constants[3] = param->alpha;
    ierr = PetscDSSetConstants(ds, 4, constants);CHKERRQ(ierr);
  }
  /* Setup Boundary Conditions */
  id   = 3;
  ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "top wall",    label, 1, &id, 0, 0, NULL, (void (*)(void)) wall_velocity, NULL, ctx, NULL);CHKERRQ(ierr);
  id   = 1;
  ierr = DMAddBoundary(dm, DM_BC_ESSENTIAL, "bottom wall", label, 1, &id, 0, 0, NULL, (void (*)(void)) wall_velocity, NULL, ctx, NULL);CHKERRQ(ierr);
  /* Setup exact solution */
  ierr = PetscDSSetExactSolution(ds, 0, quadratic_u, ctx);CHKERRQ(ierr);
  ierr = PetscDSSetExactSolution(ds, 1, linear_p, ctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM             cdm = dm;
  PetscFE        fe[2];
  Parameter     *param;
  PetscBool      simplex;
  PetscInt       dim;
  MPI_Comm       comm;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMPlexIsSimplex(dm, &simplex);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, dim, simplex, "vel_", PETSC_DEFAULT, &fe[0]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[0], "velocity");CHKERRQ(ierr);
  ierr = PetscFECreateDefault(comm, dim, 1, simplex, "pres_", PETSC_DEFAULT, &fe[1]);CHKERRQ(ierr);
  ierr = PetscFECopyQuadrature(fe[0], fe[1]);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) fe[1], "pressure");CHKERRQ(ierr);
  /* Set discretization and boundary conditions for each mesh */
  ierr = DMSetField(dm, 0, NULL, (PetscObject) fe[0]);CHKERRQ(ierr);
  ierr = DMSetField(dm, 1, NULL, (PetscObject) fe[1]);CHKERRQ(ierr);
  ierr = DMCreateDS(dm);CHKERRQ(ierr);
  ierr = SetupProblem(dm, user);CHKERRQ(ierr);
  ierr = PetscBagGetData(user->bag, (void **) &param);CHKERRQ(ierr);
  while (cdm) {
    ierr = DMCopyDisc(dm, cdm);CHKERRQ(ierr);
    ierr = DMPlexCreateBasisRotation(cdm, param->alpha, 0.0, 0.0);CHKERRQ(ierr);
    ierr = DMGetCoarseDM(cdm, &cdm);CHKERRQ(ierr);
  }
  ierr = PetscFEDestroy(&fe[0]);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe[1]);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  SNES           snes; /* nonlinear solver */
  DM             dm;   /* problem definition */
  Vec            u, r; /* solution and residual */
  AppCtx         user; /* user-defined work context */
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = PetscBagCreate(PETSC_COMM_WORLD, sizeof(Parameter), &user.bag);CHKERRQ(ierr);
  ierr = SetupParameters(&user);CHKERRQ(ierr);
  ierr = PetscBagSetFromOptions(user.bag);CHKERRQ(ierr);
  ierr = SNESCreate(PETSC_COMM_WORLD, &snes);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  ierr = SNESSetDM(snes, dm);CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dm, &user);CHKERRQ(ierr);
  /* Setup problem */
  ierr = SetupDiscretization(dm, &user);CHKERRQ(ierr);
  ierr = DMPlexCreateClosureIndex(dm, NULL);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dm, &u);CHKERRQ(ierr);
  ierr = VecDuplicate(u, &r);CHKERRQ(ierr);

  ierr = DMPlexSetSNESLocalFEM(dm,&user,&user,&user);CHKERRQ(ierr);

  ierr = SNESSetFromOptions(snes);CHKERRQ(ierr);

  {
    PetscDS              ds;
    PetscSimplePointFunc exactFuncs[2];
    void                *ctxs[2];

    ierr = DMGetDS(dm, &ds);CHKERRQ(ierr);
    ierr = PetscDSGetExactSolution(ds, 0, &exactFuncs[0], &ctxs[0]);CHKERRQ(ierr);
    ierr = PetscDSGetExactSolution(ds, 1, &exactFuncs[1], &ctxs[1]);CHKERRQ(ierr);
    ierr = DMProjectFunction(dm, 0.0, exactFuncs, ctxs, INSERT_ALL_VALUES, u);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) u, "Exact Solution");CHKERRQ(ierr);
    ierr = VecViewFromOptions(u, NULL, "-exact_vec_view");CHKERRQ(ierr);
  }
  ierr = DMSNESCheckFromOptions(snes, u);CHKERRQ(ierr);
  ierr = VecSet(u, 0.0);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) u, "Solution");CHKERRQ(ierr);
  ierr = SNESSolve(snes, NULL, u);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-sol_vec_view");CHKERRQ(ierr);

  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = SNESDestroy(&snes);CHKERRQ(ierr);
  ierr = PetscBagDestroy(&user.bag);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  # Convergence
  test:
    suffix: 2d_quad_q1_p0_conv
    requires: !single
    args: -dm_plex_simplex 0 -dm_plex_separate_marker -dm_refine 1 \
      -vel_petscspace_degree 1 -pres_petscspace_degree 0 \
      -snes_convergence_estimate -convest_num_refine 2 -snes_error_if_not_converged \
      -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full \
        -fieldsplit_velocity_pc_type lu \
        -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi
  test:
    suffix: 2d_quad_q1_p0_conv_u0
    requires: !single
    args: -dm_plex_simplex 0 -dm_plex_separate_marker -dm_refine 1 -u_0 0.125 \
      -vel_petscspace_degree 1 -pres_petscspace_degree 0 \
      -snes_convergence_estimate -convest_num_refine 2 -snes_error_if_not_converged \
      -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full \
        -fieldsplit_velocity_pc_type lu \
        -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi
  test:
    suffix: 2d_quad_q1_p0_conv_u0_alpha
    requires: !single
    args: -dm_plex_simplex 0 -dm_plex_separate_marker -dm_refine 1 -u_0 0.125 -alpha 0.3927 \
      -vel_petscspace_degree 1 -pres_petscspace_degree 0 \
      -snes_convergence_estimate -convest_num_refine 2 -snes_error_if_not_converged \
      -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full \
        -fieldsplit_velocity_pc_type lu \
        -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi
  test:
    suffix: 2d_quad_q1_p0_conv_gmg_vanka
    requires: !single long_runtime
    args: -dm_plex_simplex 0 -dm_plex_separate_marker -dm_plex_box_faces 2,2 -dm_refine_hierarchy 1 \
      -vel_petscspace_degree 1 -pres_petscspace_degree 0 \
      -snes_convergence_estimate -convest_num_refine 1 -snes_error_if_not_converged \
      -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full \
        -fieldsplit_velocity_pc_type mg \
          -fieldsplit_velocity_mg_levels_pc_type patch -fieldsplit_velocity_mg_levels_pc_patch_exclude_subspaces 1 \
          -fieldsplit_velocity_mg_levels_pc_patch_construct_codim 0 -fieldsplit_velocity_mg_levels_pc_patch_construct_type vanka \
        -fieldsplit_pressure_ksp_rtol 1e-5 -fieldsplit_pressure_pc_type jacobi
  test:
    suffix: 2d_tri_p2_p1_conv
    requires: triangle !single
    args: -dm_plex_separate_marker -dm_refine 1 \
      -vel_petscspace_degree 2 -pres_petscspace_degree 1 \
      -dmsnes_check .001 -snes_error_if_not_converged \
      -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full \
        -fieldsplit_velocity_pc_type lu \
        -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi
  test:
    suffix: 2d_tri_p2_p1_conv_u0_alpha
    requires: triangle !single
    args: -dm_plex_separate_marker -dm_refine 0 -u_0 0.125 -alpha 0.3927 \
      -vel_petscspace_degree 2 -pres_petscspace_degree 1 \
      -dmsnes_check .001 -snes_error_if_not_converged \
      -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full \
        -fieldsplit_velocity_pc_type lu \
        -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi
  test:
    suffix: 2d_tri_p2_p1_conv_gmg_vcycle
    requires: triangle !single
    args: -dm_plex_separate_marker -dm_plex_box_faces 2,2 -dm_refine_hierarchy 1 \
      -vel_petscspace_degree 2 -pres_petscspace_degree 1 \
      -dmsnes_check .001 -snes_error_if_not_converged \
      -ksp_type fgmres -ksp_gmres_restart 10 -ksp_rtol 1.0e-9 -ksp_error_if_not_converged \
      -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_factorization_type full \
        -fieldsplit_velocity_pc_type mg \
        -fieldsplit_pressure_ksp_rtol 1e-10 -fieldsplit_pressure_pc_type jacobi
TEST*/
