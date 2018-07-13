static char help[] = "Tests L2 projection with DMSwarm.\n";

#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscds.h>
#include <petscksp.h>

typedef struct {
  PetscInt  dim;                              /* The topological mesh dimension */
  PetscBool simplex;                          /* Flag for simplices or tensor cells */
  char      meshFilename[PETSC_MAX_PATH_LEN]; /* Name of the mesh filename if any */
  PetscInt  faces;                            /* Number of faces per edge if unit square/cube generated */
} AppCtx;

static PetscErrorCode linear(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt d;
  u[0] = 0.0;
  for (d = 0; d < dim; ++d) u[0] += x[d];
  return 0;
}

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->dim     = 2;
  options->simplex = PETSC_TRUE;
  options->faces   = 1;
  ierr = PetscStrcpy(options->meshFilename, "");CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm, "", "L2 Projection Options", "DMPLEX");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-dim", "The topological mesh dimension", "ex2.c", options->dim, &options->dim, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-simplex", "The flag for simplices or tensor cells", "ex2.c", options->simplex, &options->simplex, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsString("-mesh", "Name of the mesh filename if any", "ex2.c", options->meshFilename, options->meshFilename, PETSC_MAX_PATH_LEN, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-faces", "Number of faces per edge if unit square/cube generated", "ex2.c", options->faces, &options->faces, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();

  PetscFunctionReturn(0);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm, AppCtx *user)
{
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscStrcmp(user->meshFilename, "", &flg);CHKERRQ(ierr);
  if (flg) {
    PetscInt faces[3];

    faces[0] = user->faces; faces[1] = user->faces; faces[2] = user->faces;
    ierr = DMPlexCreateBoxMesh(comm, user->dim, user->simplex, faces, NULL, NULL, NULL, PETSC_TRUE, dm);CHKERRQ(ierr);
  } else {
    ierr = DMPlexCreateFromFile(comm, user->meshFilename, PETSC_TRUE, dm);CHKERRQ(ierr);
    ierr = DMGetDimension(*dm, &user->dim);CHKERRQ(ierr);
  }
  {
    DM distributedMesh = NULL;

    ierr = DMPlexDistribute(*dm, 0, NULL, &distributedMesh);CHKERRQ(ierr);
    if (distributedMesh) {
      ierr = DMDestroy(dm);CHKERRQ(ierr);
      *dm  = distributedMesh;
    }
  }
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *dm, "Mesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "identity"
static void identity(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                     const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                     const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                     PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  g0[0] = 1.0;
}

static PetscErrorCode CreateFEM(DM dm, AppCtx *user)
{
  PetscDS        prob;
  PetscFE        fe;
  PetscInt       dim;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscFECreateDefault(PetscObjectComm((PetscObject) dm), dim, 1, user->simplex, NULL, -1, &fe);CHKERRQ(ierr);
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetDiscretization(prob, 0, (PetscObject) fe);CHKERRQ(ierr);
  ierr = PetscFEDestroy(&fe);CHKERRQ(ierr);
  /* Setup to form mass matrix */
  ierr = PetscDSSetJacobian(prob, 0, 0, identity, NULL, NULL, NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CreateParticles(DM dm, DM *sw, AppCtx *user)
{
  PetscScalar   *vals;
  PetscReal     *centroid, *coords;
  PetscInt      *cellid;
  PetscInt       Ncell, c, dim, d;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = DMCreate(PetscObjectComm((PetscObject) dm), sw);CHKERRQ(ierr);
  ierr = DMSetType(*sw, DMSWARM);CHKERRQ(ierr);
  ierr = DMSetDimension(*sw, dim);CHKERRQ(ierr);

  ierr = DMSwarmSetType(*sw, DMSWARM_PIC);CHKERRQ(ierr);
  ierr = DMSwarmSetCellDM(*sw, dm);CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(*sw, "w_q", 1, PETSC_SCALAR);CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(*sw);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, NULL, &Ncell);CHKERRQ(ierr);
  ierr = DMSwarmSetLocalSizes(*sw, Ncell, 0);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*sw);CHKERRQ(ierr);

  ierr = PetscMalloc1(dim, &centroid);CHKERRQ(ierr);
  ierr = DMSwarmGetField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmGetField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
  ierr = DMSwarmGetField(*sw, "w_q", NULL, NULL, (void **) &vals);CHKERRQ(ierr);
  for (c = 0; c < Ncell; ++c) {
    ierr = DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL);CHKERRQ(ierr);
    cellid[c] = c;
    for (d = 0; d < dim; ++d) coords[c*dim+d] = centroid[d];
    linear(dim, 0.0, &coords[c*dim], 1, &vals[c], NULL);
  }
  ierr = DMSwarmRestoreField(*sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(*sw, DMSwarmPICField_cellid, NULL, NULL, (void **) &cellid);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(*sw, "w_q", NULL, NULL, (void **) &vals);CHKERRQ(ierr);
  ierr = PetscFree(centroid);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *sw, "Particles");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*sw, NULL, "-sw_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode computeParticleMoments(DM sw, PetscReal moments[3], AppCtx *user)
{
  DM                 dm;
  const PetscReal   *coords;
  const PetscScalar *w;
  PetscReal          mom[3] = {0.0, 0.0, 0.0};
  PetscInt           cell, cStart, cEnd, dim;
  PetscErrorCode     ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDimension(sw, &dim);CHKERRQ(ierr);
  ierr = DMSwarmGetCellDM(sw, &dm);CHKERRQ(ierr);
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMSwarmSortGetAccess(sw);CHKERRQ(ierr);
  ierr = DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmGetField(sw, "w_q", NULL, NULL, (void **) &w);CHKERRQ(ierr);
  for (cell = cStart; cell < cEnd; ++cell) {
    PetscInt *pidx;
    PetscInt  Np, p, d;

    ierr = DMSwarmSortGetPointsPerCell(sw, cell, &Np, &pidx);CHKERRQ(ierr);
    for (p = 0; p < Np; ++p) {
      const PetscInt   idx = pidx[p];
      const PetscReal *c   = &coords[idx*dim];

      mom[0] += w[idx];
      mom[1] += w[idx] * c[0];
      for (d = 0; d < dim; ++d) mom[2] += w[idx] * c[d]*c[d];
    }
    ierr = PetscFree(pidx);CHKERRQ(ierr);
  }
  ierr = DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **) &coords);CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **) &w);CHKERRQ(ierr);
  ierr = DMSwarmSortRestoreAccess(sw);CHKERRQ(ierr);
  ierr = MPI_Allreduce(mom, moments, 3, MPIU_REAL, MPI_SUM, PetscObjectComm((PetscObject) sw));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "f0_1"
static void f0_1(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                 const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                 const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                 PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = u[0];
}

#undef __FUNCT__
#define __FUNCT__ "f0_x"
static void f0_x(PetscInt dim, PetscInt Nf, PetscInt NfAux,
		    const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
		    const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
		    PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = x[0]*u[0];
}

#undef __FUNCT__
#define __FUNCT__ "f0_r2"
static void f0_r2(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                  const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                  const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                  PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;

  for (d = 0; d < dim; ++d) f0[0] += PetscSqr(x[d])*u[0];
}

static PetscErrorCode computeFEMMoments(DM dm, Vec u, PetscReal moments[3], AppCtx *user)
{
  PetscDS        prob;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
  ierr = PetscDSSetObjective(prob, 0, &f0_1);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegralFEM(dm, u, &moments[0], user);CHKERRQ(ierr);
  ierr = PetscDSSetObjective(prob, 0, &f0_x);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegralFEM(dm, u, &moments[1], user);CHKERRQ(ierr);
  ierr = PetscDSSetObjective(prob, 0, &f0_r2);CHKERRQ(ierr);
  ierr = DMPlexComputeIntegralFEM(dm, u, &moments[2], user);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode TestL2Projection(DM dm, DM sw, AppCtx *user)
{
  MPI_Comm       comm;
  KSP            ksp;
  Mat            M;            /* FEM mass matrix */
  Mat            M_p;          /* Particle mass matrix */
  Vec            f, rhs, fhat; /* Particle field f, \int phi_i f, FEM field */
  PetscReal      pmoments[3];  /* \int f, \int x f, \int r^2 f */
  PetscReal      fmoments[3];  /* \int \hat f, \int x \hat f, \int r^2 \hat f */
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = KSPCreate(comm, &ksp);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &fhat);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(dm, &rhs);CHKERRQ(ierr);

  //ierr = DMSwarmCreateInterpolationMatrix(sw, dm, &M_p);CHKERRQ(ierr);
  ierr = DMCreateMassMatrix(sw, dm, &M_p);CHKERRQ(ierr);
  ierr = DMSwarmCreateGlobalVectorFromField(sw, "w_q", &f);CHKERRQ(ierr);
  ierr = MatMult(M_p, f, rhs);CHKERRQ(ierr);
  ierr = DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &f);CHKERRQ(ierr);

  ierr = DMCreateMatrix(dm, &M);CHKERRQ(ierr);
  ierr = DMPlexSNESComputeJacobianFEM(dm, fhat, M, M, user);CHKERRQ(ierr);
  ierr = MatViewFromOptions(M, NULL, "-fe_mass_mat_view");CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, M, M);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp, rhs, fhat);CHKERRQ(ierr);
  /* Check moments of field */
  ierr = computeParticleMoments(sw, pmoments, user);CHKERRQ(ierr);
  ierr = computeFEMMoments(dm, fhat, fmoments, user);CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "L2 projection m ([m - m_p]/m) density: %20.13e (%11.4e), x-momentum: %20.13e (%11.4e), energy: %20.13e (%11.4e).\n", fmoments[0], (fmoments[0] - pmoments[0])/fmoments[0],
                     fmoments[1], (fmoments[1] - pmoments[1])/fmoments[1], fmoments[2], (fmoments[2] - pmoments[2])/fmoments[2]);CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatDestroy(&M);CHKERRQ(ierr);
  ierr = MatDestroy(&M_p);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &fhat);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(dm, &rhs);CHKERRQ(ierr);
#if 0
  /* get FE moments */
  {
    PetscScalar momentum, energy, density, tt[0];
    PetscDS     prob;
    Vec         vecs[3];
    ierr = MatMultTranspose(QinterpT, f_q, rhs);CHKERRQ(ierr); /* interpolate particles to grid: Q * w_p */
    ierr = KSPSolve(ksp, rhs, uproj);CHKERRQ(ierr);
    ierr = DMGetDS(dm, &prob);CHKERRQ(ierr);
    ierr = PetscDSSetObjective(prob, 0, &f0_1);CHKERRQ(ierr);
    ierr = DMPlexComputeIntegralFEM(dm,uproj,tt,user);CHKERRQ(ierr);
    density = tt[0];
    ierr = PetscDSSetObjective(prob, 0, &f0_momx);CHKERRQ(ierr);
    ierr = DMPlexComputeIntegralFEM(dm,uproj,tt,user);CHKERRQ(ierr);
    momentum = tt[0];
    ierr = PetscDSSetObjective(prob, 0, &f0_energy);CHKERRQ(ierr);
    ierr = DMPlexComputeIntegralFEM(dm,uproj,tt,user);CHKERRQ(ierr);
    energy = tt[0];
    PetscPrintf(comm, "\t[%D] L2 projection x_m ([x_m-x_p]/x_m) rho: %20.13e (%11.4e), momentum_x: %20.13e (%11.4e), energy: %20.13e (%11.4e).\n",rank,density,(density-den0tot)/density,momentum,(momentum-mom0tot)/momentum,energy,(energy-energy0tot)/energy);
    /* compute coordinate vectos x and moments x' * Q * w_p */
    ierr = computeMomentVectors(dm, vecs, user);CHKERRQ(ierr);
    ierr = VecDot(vecs[0],rhs,&density);CHKERRQ(ierr);
    ierr = VecDot(vecs[1],rhs,&momentum);CHKERRQ(ierr);
    ierr = VecDot(vecs[2],rhs,&energy);CHKERRQ(ierr);
    PetscPrintf(comm, "\t[%D] x' * Q * w_p: x_m ([x_m-x_p]/x_m) rho: %20.13e (%11.4e), momentum_x: %20.13e (%11.4e), energy: %20.13e (%11.4e).\n",rank,density,(density-den0tot)/density,momentum,(momentum-mom0tot)/momentum,energy,(energy-energy0tot)/energy);
    ierr = VecDestroy(&vecs[0]);CHKERRQ(ierr);
    ierr = VecDestroy(&vecs[1]);CHKERRQ(ierr);
    ierr = VecDestroy(&vecs[2]);CHKERRQ(ierr);
  }
#endif
  PetscFunctionReturn(0);
}

int main (int argc, char * argv[]) {
  MPI_Comm       comm;
  DM             dm, sw;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);CHKERRQ(ierr);
  comm = PETSC_COMM_WORLD;
  ierr = ProcessOptions(comm, &user);CHKERRQ(ierr);
  ierr = CreateMesh(comm, &dm, &user);CHKERRQ(ierr);
  ierr = CreateFEM(dm, &user);CHKERRQ(ierr);
  ierr = CreateParticles(dm, &sw, &user);CHKERRQ(ierr);
  ierr = TestL2Projection(dm, sw, &user);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&sw);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: proj_0
    args: -dim 2 -faces 1 -dm_view -sw_view -petscspace_order 1 -petscfe_default_quadrature_order 1

TEST*/
