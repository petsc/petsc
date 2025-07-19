static char help[] = "Tests for Gauss' Law\n\n";

/* We want to check the weak version of Gauss' Law, namely that

  \int_\Omega v div q - \int_\Gamma v (q \cdot n) = 0

*/

#include <petscdmplex.h>
#include <petscsnes.h>
#include <petscds.h>

typedef struct {
  PetscInt  degree;  // The degree of the discretization
  PetscBool divFree; // True if the solution is divergence-free
} AppCtx;

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return PETSC_SUCCESS;
}

// div <x^d y^{d-1}, -x^{d-1} y^d> = 0
static void solenoidal_2d(PetscInt n, const PetscReal x[], PetscScalar u[])
{
  u[0] = PetscPowRealInt(x[0], n) * PetscPowRealInt(x[1], n - 1);
  u[1] = -PetscPowRealInt(x[0], n - 1) * PetscPowRealInt(x[1], n);
}
// div <x^d y^{d-1} z^{d-1}, -2 x^{d-1} y^d z^{d-1}, x^{d-1} y^{d-1} z^d> = 0
static void solenoidal_3d(PetscInt n, const PetscReal x[], PetscScalar u[])
{
  u[0] = PetscPowRealInt(x[0], n) * PetscPowRealInt(x[1], n - 1) * PetscPowRealInt(x[2], n - 1);
  u[1] = -2. * PetscPowRealInt(x[0], n - 1) * PetscPowRealInt(x[1], n) * PetscPowRealInt(x[2], n - 1);
  u[2] = PetscPowRealInt(x[0], n - 1) * PetscPowRealInt(x[1], n - 1) * PetscPowRealInt(x[2], n);
}

static PetscErrorCode solenoidal_totaldeg_2d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  const PetscInt deg = *(PetscInt *)ctx;
  const PetscInt n   = deg / 2 + deg % 2;

  solenoidal_2d(n, x, u);
  return PETSC_SUCCESS;
}

static PetscErrorCode solenoidal_totaldeg_3d(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  const PetscInt deg = *(PetscInt *)ctx;
  const PetscInt n   = deg / 3 + (deg % 3 ? 1 : 0);

  solenoidal_3d(n, x, u);
  return PETSC_SUCCESS;
}

// This is in P_n^{-}
static PetscErrorCode source_totaldeg(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  const PetscInt n = *(PetscInt *)ctx;

  for (PetscInt d = 0; d < dim; ++d) u[d] = PetscPowRealInt(x[d], n + 1);
  return PETSC_SUCCESS;
}

static void identity(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscInt deg = (PetscInt)PetscRealPart(constants[0]);
  PetscScalar    p[3];

  if (dim == 2) PetscCallVoid(solenoidal_totaldeg_2d(dim, t, x, uOff[1] - uOff[0], p, (void *)&deg));
  else PetscCallVoid(solenoidal_totaldeg_3d(dim, t, x, uOff[1] - uOff[0], p, (void *)&deg));
  for (PetscInt c = 0; c < dim; ++c) f0[c] = -u[c] + p[c];
}

static void zero_bd(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  for (PetscInt d = 0; d < dim; ++d) f0[0] = 0.;
}

static void flux(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], const PetscReal n[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  for (PetscInt d = 0; d < dim; ++d) f0[0] -= u[d] * n[d];
}

static void divergence(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  for (PetscInt d = 0; d < dim; ++d) f0[0] += u_x[d * dim + d];
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  options->degree = -1;

  PetscFunctionBeginUser;
  PetscOptionsBegin(comm, "", "Gauss' Law Test Options", "DMPLEX");
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateDiscretization(DM dm, AppCtx *user)
{
  PetscFE         feq, fep;
  PetscSpace      sp;
  PetscQuadrature quad, fquad;
  PetscDS         ds;
  DMLabel         label;
  DMPolytopeType  ct;
  PetscInt        dim, cStart, minDeg, maxDeg;
  PetscBool       isTrimmed, isSum;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, dim, ct, "field_", -1, &feq));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)feq));
  PetscCall(PetscFEGetQuadrature(feq, &quad));
  PetscCall(PetscFEGetFaceQuadrature(feq, &fquad));
  PetscCall(PetscFEGetBasisSpace(feq, &sp));
  PetscCall(PetscSpaceGetDegree(sp, &minDeg, &maxDeg));
  PetscCall(PetscObjectTypeCompare((PetscObject)sp, PETSCSPACEPTRIMMED, &isTrimmed));
  PetscCall(PetscObjectTypeCompare((PetscObject)sp, PETSCSPACESUM, &isSum));
  if (isSum) {
    PetscSpace subsp, xsp, ysp;
    PetscInt   xdeg, ydeg;
    PetscBool  isTensor;

    PetscCall(PetscSpaceSumGetSubspace(sp, 0, &subsp));
    PetscCall(PetscObjectTypeCompare((PetscObject)subsp, PETSCSPACETENSOR, &isTensor));
    if (isTensor) {
      PetscCall(PetscSpaceTensorGetSubspace(subsp, 0, &xsp));
      PetscCall(PetscSpaceTensorGetSubspace(subsp, 1, &ysp));
      PetscCall(PetscSpaceGetDegree(xsp, &xdeg, NULL));
      PetscCall(PetscSpaceGetDegree(ysp, &ydeg, NULL));
      isTrimmed = xdeg != ydeg ? PETSC_TRUE : PETSC_FALSE;
    }
  }
  user->degree = minDeg;
  if (isTrimmed) user->divFree = PETSC_FALSE;
  else user->divFree = PETSC_TRUE;
  PetscCheck(!user->divFree || user->degree, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Degree 0 solution not available");
  PetscCall(PetscFEDestroy(&feq));
  PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, "pot_", -1, &fep));
  PetscCall(DMSetField(dm, 1, NULL, (PetscObject)fep));
  PetscCall(PetscFESetQuadrature(fep, quad));
  PetscCall(PetscFESetFaceQuadrature(fep, fquad));
  PetscCall(PetscFEDestroy(&fep));
  PetscCall(DMCreateDS(dm));

  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetResidual(ds, 0, identity, NULL));
  PetscCall(PetscDSSetResidual(ds, 1, divergence, NULL));
  if (user->divFree) {
    if (dim == 2) PetscCall(PetscDSSetExactSolution(ds, 0, solenoidal_totaldeg_2d, &user->degree));
    else PetscCall(PetscDSSetExactSolution(ds, 0, solenoidal_totaldeg_3d, &user->degree));
  } else {
    PetscCall(PetscDSSetExactSolution(ds, 0, source_totaldeg, &user->degree));
  }
  PetscCall(PetscDSSetExactSolution(ds, 1, zero, &user->degree));
  PetscCall(DMGetLabel(dm, "marker", &label));

  // TODO Can we also test the boundary residual integration?
  //PetscWeakForm wf;
  //PetscInt      bd, id = 1;
  //PetscCall(DMAddBoundary(dm, DM_BC_NATURAL, "boundary", label, 1, &id, 1, 0, NULL, NULL, NULL, user, &bd));
  //PetscCall(PetscDSGetBoundary(ds, bd, &wf, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL));
  //PetscCall(PetscWeakFormSetIndexBdResidual(wf, label, id, 1, 0, 0, flux, 0, NULL));

  {
    PetscScalar constants[1];

    constants[0] = user->degree;
    PetscCall(PetscDSSetConstants(ds, 1, constants));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM          dm;
  SNES        snes;
  Vec         u;
  PetscReal   error[2], residual;
  PetscScalar source[2], outflow[2];
  AppCtx      user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
  PetscCall(CreateDiscretization(dm, &user));
  PetscCall(DMGetGlobalVector(dm, &u));
  PetscCall(PetscObjectSetName((PetscObject)u, "solution"));
  PetscCall(DMComputeExactSolution(dm, 0., u, NULL));

  PetscCall(SNESCreate(PetscObjectComm((PetscObject)dm), &snes));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(DMPlexSetSNESLocalFEM(dm, PETSC_FALSE, &user));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(DMSNESCheckDiscretization(snes, dm, 0., u, PETSC_DETERMINE, error));
  PetscCheck(PetscAbsReal(error[0]) < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Exact solution does not fit into FEM space: %g should be zero", (double)error[0]);
  if (user.divFree) {
    PetscCall(DMSNESCheckResidual(snes, dm, u, PETSC_DETERMINE, &residual));
    PetscCheck(PetscAbsReal(residual) < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Exact solution is not divergence-free: %g should be zero", (double)residual);
  } else {
    PetscDS ds;

    PetscCall(DMGetDS(dm, &ds));
    PetscCall(PetscDSSetObjective(ds, 1, divergence));
    PetscCall(DMPlexComputeIntegralFEM(dm, u, source, &user));
  }
  PetscCall(SNESDestroy(&snes));

  PetscBdPointFn *funcs[] = {zero_bd, flux};
  DMLabel         label;
  PetscInt        id = 1;

  PetscCall(DMGetLabel(dm, "marker", &label));
  PetscCall(DMPlexComputeBdIntegral(dm, u, label, 1, &id, funcs, outflow, &user));
  if (user.divFree) PetscCheck(PetscAbsScalar(outflow[1]) < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Outflow %g should be zero for a divergence-free field", (double)PetscRealPart(outflow[1]));
  else PetscCheck(PetscAbsScalar(source[1] + outflow[1]) < PETSC_SMALL, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Outflow %g should oppose source %g", (double)PetscRealPart(outflow[1]), (double)PetscRealPart(source[1]));

  PetscCall(DMRestoreGlobalVector(dm, &u));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    suffix: p
    requires: triangle ctetgen
    args: -dm_plex_dim {{2 3}} -dm_plex_box_faces 2,2,2
    output_file: output/empty.out

    test:
      suffix: 1
      args: -field_petscspace_degree 1 -pot_petscspace_degree 1
    test:
      suffix: 2
      args: -field_petscspace_degree 2 -pot_petscspace_degree 2
    test:
      suffix: 3
      args: -field_petscspace_degree 3 -pot_petscspace_degree 3
    test:
      suffix: 4
      args: -field_petscspace_degree 4 -pot_petscspace_degree 4

  testset:
    suffix: q
    args: -dm_plex_dim {{2 3}} -dm_plex_simplex 0 -dm_plex_box_faces 2,2
    output_file: output/empty.out

    test:
      suffix: 1
      args: -field_petscspace_degree 1 -pot_petscspace_degree 1
    test:
      suffix: 2
      args: -field_petscspace_degree 2 -pot_petscspace_degree 2
    test:
      suffix: 3
      args: -field_petscspace_degree 3 -pot_petscspace_degree 3
    test:
      suffix: 4
      args: -field_petscspace_degree 4 -pot_petscspace_degree 4

  testset:
    suffix: bdm
    requires: triangle ctetgen
    args: -dm_plex_dim 2 -dm_plex_box_faces 2,2
    output_file: output/empty.out

    test:
      suffix: 1
      args: -pot_petscspace_degree 0 -pot_petscdualspace_lagrange_continuity 0 \
            -field_petscspace_degree 1 -field_petscdualspace_type bdm \
            -field_petscfe_default_quadrature_order 2

  testset:
    suffix: rt
    requires: triangle ctetgen
    args: -dm_plex_dim 2 -dm_plex_box_faces 2,2
    output_file: output/empty.out

    test:
      suffix: 1
      args: -pot_petscspace_degree 0 -pot_petscdualspace_lagrange_continuity 0 \
            -field_petscspace_type ptrimmed \
            -field_petscspace_components 2 \
            -field_petscspace_ptrimmed_form_degree -1 \
            -field_petscdualspace_order 1 \
            -field_petscdualspace_form_degree -1 \
            -field_petscdualspace_lagrange_trimmed true \
            -field_petscfe_default_quadrature_order 2

  testset:
    suffix: rtq
    requires: triangle ctetgen
    args: -dm_plex_dim 2 -dm_plex_simplex 0 -dm_plex_box_faces 2,2
    output_file: output/empty.out

    test:
      suffix: 1
      args: -pot_petscspace_degree 0 -pot_petscdualspace_lagrange_continuity 0 \
            -field_petscspace_degree 1 \
            -field_petscspace_type sum \
            -field_petscspace_variables 2 \
            -field_petscspace_components 2 \
            -field_petscspace_sum_spaces 2 \
            -field_petscspace_sum_concatenate true \
            -field_sumcomp_0_petscspace_variables 2 \
            -field_sumcomp_0_petscspace_type tensor \
            -field_sumcomp_0_petscspace_tensor_spaces 2 \
            -field_sumcomp_0_petscspace_tensor_uniform false \
            -field_sumcomp_0_tensorcomp_0_petscspace_degree 1 \
            -field_sumcomp_0_tensorcomp_1_petscspace_degree 0 \
            -field_sumcomp_1_petscspace_variables 2 \
            -field_sumcomp_1_petscspace_type tensor \
            -field_sumcomp_1_petscspace_tensor_spaces 2 \
            -field_sumcomp_1_petscspace_tensor_uniform false \
            -field_sumcomp_1_tensorcomp_0_petscspace_degree 0 \
            -field_sumcomp_1_tensorcomp_1_petscspace_degree 1 \
            -field_petscdualspace_order 1 \
            -field_petscdualspace_form_degree -1 \
            -field_petscdualspace_lagrange_trimmed true \
            -field_petscfe_default_quadrature_order 2

TEST*/
