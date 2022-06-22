static char help[] = "Tests projection with DMSwarm using general particle shapes.\n";

#include <petsc/private/dmswarmimpl.h>
#include <petsc/private/petscfeimpl.h>

#include <petscdmplex.h>
#include <petscds.h>
#include <petscksp.h>

typedef struct {
  PetscInt dummy;
} AppCtx;

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode linear(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  u[0] = 0.0;
  for (d = 0; d < dim; ++d) u[0] += x[d];
  return 0;
}

static void identity(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.0;
}

static PetscErrorCode CreateParticles(DM dm, DM *sw, AppCtx *user)
{
  PetscDS          prob;
  PetscFE          fe;
  PetscQuadrature  quad;
  PetscScalar     *vals;
  PetscReal       *v0, *J, *invJ, detJ, *coords, *xi0;
  PetscInt        *cellid;
  const PetscReal *qpoints;
  PetscInt         Ncell, c, Nq, q, dim;
  PetscBool        simplex;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, simplex, NULL, -1, &fe));
  PetscCall(DMGetDS(dm, &prob));
  PetscCall(PetscDSSetDiscretization(prob, 0, (PetscObject) fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(PetscDSSetJacobian(prob, 0, 0, identity, NULL, NULL, NULL));
  PetscCall(DMPlexGetHeightStratum(dm, 0, NULL, &Ncell));
  PetscCall(PetscDSGetDiscretization(prob, 0, (PetscObject *) &fe));
  PetscCall(PetscFEGetQuadrature(fe, &quad));
  PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &Nq, &qpoints, NULL));

  PetscCall(DMCreate(PetscObjectComm((PetscObject) dm), sw));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));

  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*sw, dm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "f_q", 1, PETSC_SCALAR));
  PetscCall(DMSwarmFinalizeFieldRegister(*sw));
  PetscCall(DMSwarmSetLocalSizes(*sw, Ncell * Nq, 0));
  PetscCall(DMSetFromOptions(*sw));

  PetscCall(PetscMalloc4(dim, &xi0, dim, &v0, dim*dim, &J, dim*dim, &invJ));
  for (c = 0; c < dim; c++) xi0[c] = -1.;
  PetscCall(DMSwarmGetField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  PetscCall(DMSwarmGetField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid));
  PetscCall(DMSwarmGetField(*sw, "f_q", NULL, NULL, (void **) &vals));
  for (c = 0; c < Ncell; ++c) {
    for (q = 0; q < Nq; ++q) {
      PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ));
      cellid[c*Nq + q] = c;
      CoordinatesRefToReal(dim, dim, xi0, v0, J, &qpoints[q*dim], &coords[(c*Nq + q)*dim]);
      linear(dim, 0.0, &coords[(c*Nq + q)*dim], 1, &vals[c*Nq + q], NULL);
    }
  }
  PetscCall(DMSwarmRestoreField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  PetscCall(DMSwarmRestoreField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid));
  PetscCall(DMSwarmRestoreField(*sw, "f_q", NULL, NULL, (void **) &vals));
  PetscCall(PetscFree4(xi0, v0, J, invJ));
  PetscFunctionReturn(0);
}

static PetscErrorCode TestL2Projection(DM dm, DM sw, AppCtx *user)
{
  PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *);
  KSP              ksp;
  Mat              mass;
  Vec              u, rhs, uproj;
  PetscReal        error;

  PetscFunctionBeginUser;
  funcs[0] = linear;

  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "f_q", &u));
  PetscCall(VecViewFromOptions(u, NULL, "-f_view"));
  PetscCall(DMGetGlobalVector(dm, &rhs));
  PetscCall(DMCreateMassMatrix(sw, dm, &mass));
  PetscCall(MatMult(mass, u, rhs));
  PetscCall(MatDestroy(&mass));
  PetscCall(VecDestroy(&u));

  PetscCall(DMGetGlobalVector(dm, &uproj));
  PetscCall(DMCreateMatrix(dm, &mass));
  PetscCall(DMPlexSNESComputeJacobianFEM(dm, uproj, mass, mass, user));
  PetscCall(MatViewFromOptions(mass, NULL, "-mass_mat_view"));
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, mass, mass));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, rhs, uproj));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(PetscObjectSetName((PetscObject) uproj, "Full Projection"));
  PetscCall(VecViewFromOptions(uproj, NULL, "-proj_vec_view"));
  PetscCall(DMComputeL2Diff(dm, 0.0, funcs, NULL, uproj, &error));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Projected L2 Error: %g\n", (double) error));
  PetscCall(MatDestroy(&mass));
  PetscCall(DMRestoreGlobalVector(dm, &rhs));
  PetscCall(DMRestoreGlobalVector(dm, &uproj));
  PetscFunctionReturn(0);
}

int main (int argc, char * argv[]) {
  MPI_Comm       comm;
  DM             dm, sw;
  AppCtx         user;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(CreateMesh(comm, &dm, &user));
  PetscCall(CreateParticles(dm, &sw, &user));
  PetscCall(PetscObjectSetName((PetscObject) dm, "Mesh"));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(DMViewFromOptions(sw, NULL, "-sw_view"));
  PetscCall(TestL2Projection(dm, sw, &user));
  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&sw));
  PetscFinalize();
  return 0;
}

/*TEST

  test:
    suffix: proj_0
    requires: pragmatic
    TODO: broken
    args: -dm_plex_separate_marker 0 -dm_view -sw_view -petscspace_degree 1 -petscfe_default_quadrature_order 1 -pc_type lu -dm_adaptor pragmatic
  test:
    suffix: proj_1
    requires: pragmatic
    TODO: broken
    args: -dm_plex_simplex 0 -dm_plex_separate_marker 0 -dm_view -sw_view -petscspace_degree 1 -petscfe_default_quadrature_order 1 -pc_type lu -dm_adaptor pragmatic
  test:
    suffix: proj_2
    requires: pragmatic
    TODO: broken
    args: -dm_plex_dim 3 -dm_plex_box_faces 2,2,2 -dm_view -sw_view -petscspace_degree 1 -petscfe_default_quadrature_order 1 -pc_type lu -dm_adaptor pragmatic
  test:
    suffix: proj_3
    requires: pragmatic
    TODO: broken
    args: -dm_plex_simplex 0 -dm_plex_separate_marker 0 -dm_view -sw_view -petscspace_degree 1 -petscfe_default_quadrature_order 1 -pc_type lu -dm_adaptor pragmatic

TEST*/
