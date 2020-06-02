static char help[] = "Tests projection with DMSwarm using general particle shapes.\n";

#include <petsc/private/dmswarmimpl.h>
#include <petsc/private/petscfeimpl.h>

#include <petscdmplex.h>
#include <petscds.h>
#include <petscksp.h>

typedef struct {
  PetscInt  dim;                        /* The topological mesh dimension */
  PetscBool simplex;                    /* Flag for simplices or tensor cells */
  char      mshNam[PETSC_MAX_PATH_LEN]; /* Name of the mesh filename if any */
  PetscInt  nbrVerEdge;                 /* Number of vertices per edge if unit square/cube generated */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim     = 2;
  options->simplex = PETSC_TRUE;
  ierr = PetscStrcpy(options->mshNam, "");CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm, "", "Meshing Adaptation Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex1.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "The flag for simplices or tensor cells", "ex1.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-msh", "Name of the mesh filename if any", "ex1.c", options->mshNam, options->mshNam, sizeof(options->mshNam), NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-nbrVerEdge", "Number of vertices per edge if unit square/cube generated", "ex1.c", options->nbrVerEdge, &options->nbrVerEdge, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscBool      flag;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscStrcmp(user->mshNam, "", &flag);CHKERRQ(ierr);
  if (flag) {
    PetscInt faces[3];

    faces[0] = user->nbrVerEdge-1; faces[1] = user->nbrVerEdge-1; faces[2] = user->nbrVerEdge-1;
    ierr = DMPlexCreateBoxMesh(comm, user->dim, user->simplex, faces, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateFromFile(comm, user->mshNam, PETSC_TRUE, dm);CHKERRQ(ierr);
    ierr = DMGetDimension(*dm, &user->dim);CHKERRQ(ierr);
  }
  {
    DM distributedMesh = NULL;

    /* Distribute mesh over processes */
    ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
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
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, user->simplex, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  ierr = PetscDSSetJacobian(prob, 0, 0, identity, NULL, NULL, NULL);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, NULL, &Ncell);CHKERRQ(ierr);
  ierr = PetscDSGetDiscretization(prob, 0, (PetscObject *) &fe);CHKERRQ(ierr);
  ierr = PetscFEGetQuadrature(fe, &quad);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad, NULL, NULL, &Nq, &qpoints, NULL);CHKERRQ(ierr);

  ierr = DMCreate(PetscObjectComm((PetscObject) dm), sw);CHKERRQ(ierr);
  ierr = DMSetType(*sw, DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(*sw, dim);CHKERRQ(ierr);

  ierr = DMSwarmSetType(*sw, DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(*sw, dm);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(*sw, "f_q", 1, PETSC_SCALAR);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(*sw);CHKERRQ(ierr);
  ierr = DMSwarmSetLocalSizes(*sw, Ncell * Nq, 0);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*sw);CHKERRQ(ierr);

  ierr = PetscMalloc4(dim, &xi0, dim, &v0, dim*dim, &J, dim*dim, &invJ);CHKERRQ(ierr);
  for (c = 0; c < dim; c++) xi0[c] = -1.;
  ierr = DMSwarmGetField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmGetField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
  ierr = DMSwarmGetField(*sw, "f_q", NULL, NULL, (void **) &vals);CHKERRQ(ierr);
  for (c = 0; c < Ncell; ++c) {
    for (q = 0; q < Nq; ++q) {
      ierr = DMPlexComputeCellGeometryFEM(dm, c, NULL, v0, J, invJ, &detJ);CHKERRQ(ierr);
      cellid[c*Nq + q] = c;
      CoordinatesRefToReal(dim, dim, xi0, v0, J, &qpoints[q*dim], &coords[(c*Nq + q)*dim]);
      linear(dim, 0.0, &coords[(c*Nq + q)*dim], 1, &vals[c*Nq + q], NULL);
    }
  }
  ierr = DMSwarmRestoreField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(*sw, "f_q", NULL, NULL, (void **) &vals);CHKERRQ(ierr);
  ierr = PetscFree4(xi0, v0, J, invJ);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestL2Projection(DM dm, DM sw, AppCtx *user)
{
  PetscErrorCode (*funcs[1])(PetscInt, PetscReal, const PetscReal [], PetscInt, PetscScalar *, void *);
  KSP              ksp;
  Mat              mass;
  Vec              u, rhs, uproj;
  PetscReal        error;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  funcs[0] = linear;

  ierr = DMSwarmCreateGlobalVectorFromField(sw, "f_q", &u);CHKERRQ(ierr);
  ierr = VecViewFromOptions(u, NULL, "-f_view");CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &rhs);CHKERRQ(ierr);
  ierr = DMCreateMassMatrix(sw, dm, &mass);CHKERRQ(ierr);
  ierr = MatMult(mass, u, rhs);CHKERRQ(ierr);
  ierr = MatDestroy(&mass);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);

  ierr = DMGetGlobalVector(dm, &uproj);CHKERRQ(ierr);
  ierr = DMCreateMatrix(dm, &mass);CHKERRQ(ierr);
  ierr = DMPlexSNESComputeJacobianFEM(dm, uproj, mass, mass, user);CHKERRQ(ierr);
  ierr = MatViewFromOptions(mass, NULL, "-mass_mat_view");CHKERRQ(ierr);
  ierr = KSPCreate(PETSC_COMM_WORLD, &ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, mass, mass);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp, rhs, uproj);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) uproj, "Full Projection");CHKERRQ(ierr);
  ierr = VecViewFromOptions(uproj, NULL, "-proj_vec_view");CHKERRQ(ierr);
  ierr = DMComputeL2Diff(dm, 0.0, funcs, NULL, uproj, &error);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD, "Projected L2 Error: %g\n", (double) error);CHKERRQ(ierr);
  ierr = MatDestroy(&mass);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &rhs);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &uproj);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main (int argc, char * argv[]) {
  MPI_Comm       comm;
  DM             dm, sw;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &dm, &user);CHKERRQ(ierr);
  ierr = CreateParticles(dm, &sw, &user);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) dm, "Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(dm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = DMViewFromOptions(sw, NULL, "-sw_view");CHKERRQ(ierr);
  ierr = TestL2Projection(dm, sw, &user);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&sw);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}

/*TEST

  test:
    suffix: proj_0
    requires: pragmatic
    TODO: broken
    args: -dim 2 -nbrVerEdge 3 -dm_plex_separate_marker 0 -dm_view -sw_view -petscspace_degree 1 -petscfe_default_quadrature_order 1 -pc_type lu
  test:
    suffix: proj_1
    requires: pragmatic
    TODO: broken
    args: -dim 2 -simplex 0 -nbrVerEdge 3 -dm_plex_separate_marker 0 -dm_view -sw_view -petscspace_degree 1 -petscfe_default_quadrature_order 1 -pc_type lu
  test:
    suffix: proj_2
    requires: pragmatic
    TODO: broken
    args: -dim 3 -nbrVerEdge 3 -dm_view -sw_view -petscspace_degree 1 -petscfe_default_quadrature_order 1 -pc_type lu
  test:
    suffix: proj_3
    requires: pragmatic
    TODO: broken
    args: -dim 2 -simplex 0 -nbrVerEdge 3 -dm_plex_separate_marker 0 -dm_view -sw_view -petscspace_degree 1 -petscfe_default_quadrature_order 1 -pc_type lu

TEST*/
