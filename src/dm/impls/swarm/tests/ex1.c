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
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
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
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexIsSimplex(dm, &simplex));
  CHKERRQ(PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, simplex, NULL, -1, &fe));
  CHKERRQ(DMGetDS(dm, &prob));
  CHKERRQ(PetscDSSetDiscretization(prob, 0, (PetscObject) fe));
  CHKERRQ(PetscFEDestroy(&fe));
  CHKERRQ(PetscDSSetJacobian(prob, 0, 0, identity, NULL, NULL, NULL));
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, NULL, &Ncell));
  CHKERRQ(PetscDSGetDiscretization(prob, 0, (PetscObject *) &fe));
  CHKERRQ(PetscFEGetQuadrature(fe, &quad));
  CHKERRQ(PetscQuadratureGetData(quad, NULL, NULL, &Nq, &qpoints, NULL));

  CHKERRQ(DMCreate(PetscObjectComm((PetscObject) dm), sw));
  CHKERRQ(DMSetType(*sw, DMSWARM));
  CHKERRQ(DMSetDimension(*sw, dim));

  CHKERRQ(DMSwarmSetType(*sw, DMSWARM_PIC));
  CHKERRQ(DMSwarmSetCellDM(*sw, dm));
  CHKERRQ(DMSwarmRegisterPetscDatatypeField(*sw, "f_q", 1, PETSC_SCALAR));
  CHKERRQ(DMSwarmFinalizeFieldRegister(*sw));
  CHKERRQ(DMSwarmSetLocalSizes(*sw, Ncell * Nq, 0));
  CHKERRQ(DMSetFromOptions(*sw));

  CHKERRQ(PetscMalloc4(dim, &xi0, dim, &v0, dim*dim, &J, dim*dim, &invJ));
  for (c = 0; c < dim; c++) xi0[c] = -1.;
  CHKERRQ(DMSwarmGetField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  CHKERRQ(DMSwarmGetField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid));
  CHKERRQ(DMSwarmGetField(*sw, "f_q", NULL, NULL, (void **) &vals));
  for (c = 0; c < Ncell; ++c) {
    for (q = 0; q < Nq; ++q) {
      CHKERRQ(DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ));
      cellid[c*Nq + q] = c;
      CoordinatesRefToReal(dim, dim, xi0, v0, J, &qpoints[q*dim], &coords[(c*Nq + q)*dim]);
      linear(dim, 0.0, &coords[(c*Nq + q)*dim], 1, &vals[c*Nq + q], NULL);
    }
  }
  CHKERRQ(DMSwarmRestoreField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords));
  CHKERRQ(DMSwarmRestoreField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid));
  CHKERRQ(DMSwarmRestoreField(*sw, "f_q", NULL, NULL, (void **) &vals));
  CHKERRQ(PetscFree4(xi0, v0, J, invJ));
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

  CHKERRQ(DMSwarmCreateGlobalVectorFromField(sw, "f_q", &u));
  CHKERRQ(VecViewFromOptions(u, NULL, "-f_view"));
  CHKERRQ(DMGetGlobalVector(dm, &rhs));
  CHKERRQ(DMCreateMassMatrix(sw, dm, &mass));
  CHKERRQ(MatMult(mass, u, rhs));
  CHKERRQ(MatDestroy(&mass));
  CHKERRQ(VecDestroy(&u));

  CHKERRQ(DMGetGlobalVector(dm, &uproj));
  CHKERRQ(DMCreateMatrix(dm, &mass));
  CHKERRQ(DMPlexSNESComputeJacobianFEM(dm, uproj, mass, mass, user));
  CHKERRQ(MatViewFromOptions(mass, NULL, "-mass_mat_view"));
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD, &ksp));
  CHKERRQ(KSPSetOperators(ksp, mass, mass));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPSolve(ksp, rhs, uproj));
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(PetscObjectSetName((PetscObject) uproj, "Full Projection"));
  CHKERRQ(VecViewFromOptions(uproj, NULL, "-proj_vec_view"));
  CHKERRQ(DMComputeL2Diff(dm, 0.0, funcs, NULL, uproj, &error));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Projected L2 Error: %g\n", (double) error));
  CHKERRQ(MatDestroy(&mass));
  CHKERRQ(DMRestoreGlobalVector(dm, &rhs));
  CHKERRQ(DMRestoreGlobalVector(dm, &uproj));
  PetscFunctionReturn(0);
}

int main (int argc, char * argv[]) {
  MPI_Comm       comm;
  DM             dm, sw;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  CHKERRQ(CreateMesh(comm, &dm, &user));
  CHKERRQ(CreateParticles(dm, &sw, &user));
  CHKERRQ(PetscObjectSetName((PetscObject) dm, "Mesh"));
  CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));
  CHKERRQ(DMViewFromOptions(sw, NULL, "-sw_view"));
  CHKERRQ(TestL2Projection(dm, sw, &user));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(DMDestroy(&sw));
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
