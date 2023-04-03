static char help[] = "Vlasov-Poisson example of central orbits\n";

/*
  To visualize the orbit, we can used

    -ts_monitor_sp_swarm -ts_monitor_sp_swarm_retain -1 -ts_monitor_sp_swarm_phase 0 -draw_size 500,500

  and we probably want it to run fast and not check convergence

    -convest_num_refine 0 -ts_dt 0.01 -ts_max_steps 100 -ts_max_time 100 -output_step 10
*/

#include <petscts.h>
#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petsc/private/dmpleximpl.h> /* For norm and dot */
#include <petscfe.h>
#include <petscds.h>
#include <petsc/private/petscfeimpl.h> /* For interpolation */
#include <petscksp.h>
#include "petscsnes.h"

PETSC_EXTERN PetscErrorCode circleSingleX(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);
PETSC_EXTERN PetscErrorCode circleSingleV(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);
PETSC_EXTERN PetscErrorCode circleMultipleX(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);
PETSC_EXTERN PetscErrorCode circleMultipleV(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);

const char *EMTypes[] = {"primal", "mixed", "coulomb", "none", "EMType", "EM_", NULL};
typedef enum {
  EM_PRIMAL,
  EM_MIXED,
  EM_COULOMB,
  EM_NONE
} EMType;

typedef struct {
  PetscBool error;     /* Flag for printing the error */
  PetscInt  ostep;     /* print the energy at each ostep time steps */
  PetscReal timeScale; /* Nondimensionalizing time scale */
  PetscReal sigma;     /* Linear charge per box length */
  EMType    em;        /* Type of electrostatic model */
  SNES      snes;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->error     = PETSC_FALSE;
  options->ostep     = 100;
  options->timeScale = 1.0e-6;
  options->sigma     = 1.;
  options->em        = EM_COULOMB;

  PetscOptionsBegin(comm, "", "Central Orbit Options", "DMSWARM");
  PetscCall(PetscOptionsBool("-error", "Flag to print the error", "ex6.c", options->error, &options->error, NULL));
  PetscCall(PetscOptionsInt("-output_step", "Number of time steps between output", "ex6.c", options->ostep, &options->ostep, NULL));
  PetscCall(PetscOptionsReal("-sigma", "Linear charge per box length", "ex6.c", options->sigma, &options->sigma, NULL));
  PetscCall(PetscOptionsReal("-timeScale", "Nondimensionalizing time scale", "ex6.c", options->timeScale, &options->timeScale, NULL));
  PetscCall(PetscOptionsEnum("-em_type", "Type of electrostatic solver", "ex6.c", EMTypes, (PetscEnum)options->em, (PetscEnum *)&options->em, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void laplacian_f1(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) f1[d] = u_x[d];
}

static void laplacian_g3(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g3[])
{
  PetscInt d;
  for (d = 0; d < dim; ++d) g3[d * dim + d] = 1.0;
}

/*
   /  I   grad\ /q\ = /0\
   \-div    0 / \u/   \f/
*/
static void f0_q(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  for (PetscInt c = 0; c < dim; ++c) f0[c] += u[uOff[0] + c];
}

static void f1_q(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  for (PetscInt d = 0; d < dim; ++d) f1[d * dim + d] += u[uOff[1]];
}

/* <t, q> */
static void g0_qq(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  PetscInt c;
  for (c = 0; c < dim; ++c) g0[c * dim + c] += 1.0;
}

static void g2_qu(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  for (PetscInt d = 0; d < dim; ++d) g2[d * dim + d] += 1.0;
}

static void g1_uq(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  for (PetscInt d = 0; d < dim; ++d) g1[d * dim + d] += 1.0;
}

static PetscErrorCode CreateFEM(DM dm, AppCtx *user)
{
  PetscFE   feu, feq;
  PetscDS   ds;
  PetscBool simplex;
  PetscInt  dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  if (user->em == EM_MIXED) {
    DMLabel label;

    PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, dim, simplex, "field_", PETSC_DETERMINE, &feq));
    PetscCall(PetscObjectSetName((PetscObject)feq, "field"));
    PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, "potential_", PETSC_DETERMINE, &feu));
    PetscCall(PetscObjectSetName((PetscObject)feu, "potential"));
    PetscCall(PetscFECopyQuadrature(feq, feu));
    PetscCall(DMSetField(dm, 0, NULL, (PetscObject)feq));
    PetscCall(DMSetField(dm, 1, NULL, (PetscObject)feu));
    PetscCall(DMCreateDS(dm));
    PetscCall(PetscFEDestroy(&feu));
    PetscCall(PetscFEDestroy(&feq));

    PetscCall(DMGetLabel(dm, "marker", &label));
    PetscCall(DMGetDS(dm, &ds));
    PetscCall(PetscDSSetResidual(ds, 0, f0_q, f1_q));
    PetscCall(PetscDSSetJacobian(ds, 0, 0, g0_qq, NULL, NULL, NULL));
    PetscCall(PetscDSSetJacobian(ds, 0, 1, NULL, NULL, g2_qu, NULL));
    PetscCall(PetscDSSetJacobian(ds, 1, 0, NULL, g1_uq, NULL, NULL));
  } else if (user->em == EM_PRIMAL) {
    PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, NULL, PETSC_DETERMINE, &feu));
    PetscCall(PetscObjectSetName((PetscObject)feu, "potential"));
    PetscCall(DMSetField(dm, 0, NULL, (PetscObject)feu));
    PetscCall(DMCreateDS(dm));
    PetscCall(PetscFEDestroy(&feu));
    PetscCall(DMGetDS(dm, &ds));
    PetscCall(PetscDSSetResidual(ds, 0, NULL, laplacian_f1));
    PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, laplacian_g3));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreatePoisson(DM dm, AppCtx *user)
{
  SNES         snes;
  Mat          J;
  MatNullSpace nullSpace;

  PetscFunctionBeginUser;
  PetscCall(CreateFEM(dm, user));
  PetscCall(SNESCreate(PetscObjectComm((PetscObject)dm), &snes));
  PetscCall(SNESSetOptionsPrefix(snes, "em_"));
  PetscCall(SNESSetDM(snes, dm));
  PetscCall(DMPlexSetSNESLocalFEM(dm, user, user, user));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(DMCreateMatrix(dm, &J));
  PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)dm), PETSC_TRUE, 0, NULL, &nullSpace));
  PetscCall(MatSetNullSpace(J, nullSpace));
  PetscCall(MatNullSpaceDestroy(&nullSpace));
  PetscCall(SNESSetJacobian(snes, J, J, NULL, NULL));
  PetscCall(MatDestroy(&J));
  user->snes = snes;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSwarm(DM dm, AppCtx *user, DM *sw)
{
  PetscReal v0[1] = {1.};
  PetscInt  dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject)dm), sw));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));
  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSwarmSetCellDM(*sw, dm));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "w_q", 1, PETSC_SCALAR));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "velocity", dim, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "species", 1, PETSC_INT));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "initCoordinates", dim, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "initVelocity", dim, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "E_field", dim, PETSC_REAL));
  PetscCall(DMSwarmFinalizeFieldRegister(*sw));
  PetscCall(DMSwarmComputeLocalSizeFromOptions(*sw));
  PetscCall(DMSwarmInitializeCoordinates(*sw));
  PetscCall(DMSwarmInitializeVelocitiesFromOptions(*sw, v0));
  PetscCall(DMSetFromOptions(*sw));
  PetscCall(DMSetApplicationContext(*sw, user));
  PetscCall(PetscObjectSetName((PetscObject)*sw, "Particles"));
  PetscCall(DMViewFromOptions(*sw, NULL, "-sw_view"));
  {
    Vec gc, gc0, gv, gv0;

    PetscCall(DMSwarmCreateGlobalVectorFromField(*sw, DMSwarmPICField_coor, &gc));
    PetscCall(DMSwarmCreateGlobalVectorFromField(*sw, "initCoordinates", &gc0));
    PetscCall(VecCopy(gc, gc0));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(*sw, DMSwarmPICField_coor, &gc));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(*sw, "initCoordinates", &gc0));
    PetscCall(DMSwarmCreateGlobalVectorFromField(*sw, "velocity", &gv));
    PetscCall(DMSwarmCreateGlobalVectorFromField(*sw, "initVelocity", &gv0));
    PetscCall(VecCopy(gv, gv0));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(*sw, "velocity", &gv));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(*sw, "initVelocity", &gv0));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFieldAtParticles_Coulomb(SNES snes, DM sw, PetscReal E[])
{
  PetscReal  *coords;
  PetscInt    dim, d, Np, p, q;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)snes), &size));
  PetscCheck(size == 1, PetscObjectComm((PetscObject)snes), PETSC_ERR_SUP, "Coulomb code only works in serial");
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));

  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  for (p = 0; p < Np; ++p) {
    PetscReal *pcoord = &coords[p * dim];
    PetscReal *pE     = &E[p * dim];
    /* Calculate field at particle p due to particle q */
    for (q = 0; q < Np; ++q) {
      PetscReal *qcoord = &coords[q * dim];
      PetscReal  rpq[3], r;

      if (p == q) continue;
      for (d = 0; d < dim; ++d) rpq[d] = pcoord[d] - qcoord[d];
      r = DMPlex_NormD_Internal(dim, rpq);
      for (d = 0; d < dim; ++d) pE[d] += rpq[d] / PetscPowRealInt(r, 3);
    }
  }
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFieldAtParticles_Primal(SNES snes, DM sw, PetscReal E[])
{
  DM         dm;
  PetscDS    ds;
  PetscFE    fe;
  Mat        M_p;
  Vec        phi, locPhi, rho, f;
  PetscReal *coords, chargeTol = 1e-13;
  PetscInt   dim, d, cStart, cEnd, c, Np;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));

  /* Create the charges rho */
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMCreateMassMatrix(sw, dm, &M_p));
  PetscCall(DMGetGlobalVector(dm, &rho));
  PetscCall(PetscObjectSetName((PetscObject)rho, "rho"));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &f));
  PetscCall(PetscObjectSetName((PetscObject)f, "particle weight"));
  PetscCall(MatMultTranspose(M_p, f, rho));
  PetscCall(MatViewFromOptions(M_p, NULL, "-mp_view"));
  PetscCall(VecViewFromOptions(f, NULL, "-weights_view"));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &f));
  PetscCall(MatDestroy(&M_p));
  {
    PetscScalar sum;
    PetscInt    n;
    PetscReal   phi_0 = 1.; /* (sigma*sigma*sigma)*(timeScale*timeScale)/(m_e*q_e*epsi_0)*/

    /* Remove constant from rho */
    PetscCall(VecGetSize(rho, &n));
    PetscCall(VecSum(rho, &sum));
    PetscCall(VecShift(rho, -sum / n));
    PetscCall(VecSum(rho, &sum));
    PetscCheck(PetscAbsScalar(sum) < chargeTol, PetscObjectComm((PetscObject)sw), PETSC_ERR_PLIB, "Charge should have no DC component: %g", (double)PetscRealPart(sum));
    /* Nondimensionalize rho */
    PetscCall(VecScale(rho, phi_0));
  }
  PetscCall(VecViewFromOptions(rho, NULL, "-poisson_rho_view"));

  PetscCall(DMGetGlobalVector(dm, &phi));
  PetscCall(PetscObjectSetName((PetscObject)phi, "potential"));
  PetscCall(VecSet(phi, 0.0));
  PetscCall(SNESSolve(snes, rho, phi));
  PetscCall(DMRestoreGlobalVector(dm, &rho));
  PetscCall(VecViewFromOptions(phi, NULL, "-phi_view"));

  PetscCall(DMGetLocalVector(dm, &locPhi));
  PetscCall(DMGlobalToLocalBegin(dm, phi, INSERT_VALUES, locPhi));
  PetscCall(DMGlobalToLocalEnd(dm, phi, INSERT_VALUES, locPhi));
  PetscCall(DMRestoreGlobalVector(dm, &phi));

  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject *)&fe));
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  for (c = cStart; c < cEnd; ++c) {
    PetscTabulation tab;
    PetscScalar    *clPhi = NULL;
    PetscReal      *pcoord, *refcoord;
    PetscReal       v[3], J[9], invJ[9], detJ;
    PetscInt       *points;
    PetscInt        Ncp, cp;

    PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Ncp, &points));
    PetscCall(DMGetWorkArray(dm, Ncp * dim, MPIU_REAL, &pcoord));
    PetscCall(DMGetWorkArray(dm, Ncp * dim, MPIU_REAL, &refcoord));
    for (cp = 0; cp < Ncp; ++cp)
      for (d = 0; d < dim; ++d) pcoord[cp * dim + d] = coords[points[cp] * dim + d];
    PetscCall(DMPlexCoordinatesToReference(dm, c, Ncp, pcoord, refcoord));
    PetscCall(PetscFECreateTabulation(fe, 1, Ncp, refcoord, 1, &tab));
    PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v, J, invJ, &detJ));
    PetscCall(DMPlexVecGetClosure(dm, NULL, locPhi, c, NULL, &clPhi));
    for (cp = 0; cp < Ncp; ++cp) {
      const PetscReal *basisDer = tab->T[1];
      const PetscInt   p        = points[cp];

      for (d = 0; d < dim; ++d) E[p * dim + d] = 0.;
      PetscCall(PetscFEFreeInterpolateGradient_Static(fe, basisDer, clPhi, dim, invJ, NULL, cp, &E[p * dim]));
      for (d = 0; d < dim; ++d) E[p * dim + d] *= -1.0;
    }
    PetscCall(DMPlexVecRestoreClosure(dm, NULL, locPhi, c, NULL, &clPhi));
    PetscCall(DMRestoreWorkArray(dm, Ncp * dim, MPIU_REAL, &pcoord));
    PetscCall(DMRestoreWorkArray(dm, Ncp * dim, MPIU_REAL, &refcoord));
    PetscCall(PetscTabulationDestroy(&tab));
    PetscCall(PetscFree(points));
  }
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmSortRestoreAccess(sw));
  PetscCall(DMRestoreLocalVector(dm, &locPhi));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFieldAtParticles_Mixed(SNES snes, DM sw, PetscReal E[])
{
  DM              dm, potential_dm;
  IS              potential_IS;
  PetscDS         ds;
  PetscFE         fe;
  PetscFEGeom     feGeometry;
  Mat             M_p;
  Vec             phi, locPhi, rho, f, temp_rho;
  PetscQuadrature q;
  PetscReal      *coords, chargeTol = 1e-13;
  PetscInt        dim, d, cStart, cEnd, c, Np, fields = 1;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));

  /* Create the charges rho */
  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMGetGlobalVector(dm, &rho));
  PetscCall(PetscObjectSetName((PetscObject)rho, "rho"));

  PetscCall(DMCreateSubDM(dm, 1, &fields, &potential_IS, &potential_dm));
  PetscCall(DMCreateMassMatrix(sw, potential_dm, &M_p));
  PetscCall(MatViewFromOptions(M_p, NULL, "-mp_view"));
  PetscCall(DMGetGlobalVector(potential_dm, &temp_rho));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &f));
  PetscCall(PetscObjectSetName((PetscObject)f, "particle weight"));
  PetscCall(VecViewFromOptions(f, NULL, "-weights_view"));
  PetscCall(MatMultTranspose(M_p, f, temp_rho));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &f));
  PetscCall(MatDestroy(&M_p));
  PetscCall(PetscObjectSetName((PetscObject)rho, "rho"));
  PetscCall(VecViewFromOptions(rho, NULL, "-poisson_rho_view"));
  PetscCall(VecISCopy(rho, potential_IS, SCATTER_FORWARD, temp_rho));
  PetscCall(DMRestoreGlobalVector(potential_dm, &temp_rho));
  PetscCall(DMDestroy(&potential_dm));
  PetscCall(ISDestroy(&potential_IS));
  {
    PetscScalar sum;
    PetscInt    n;
    PetscReal   phi_0 = 1.; /*(sigma*sigma*sigma)*(timeScale*timeScale)/(m_e*q_e*epsi_0);*/

    /*Remove constant from rho*/
    PetscCall(VecGetSize(rho, &n));
    PetscCall(VecSum(rho, &sum));
    PetscCall(VecShift(rho, -sum / n));
    PetscCall(VecSum(rho, &sum));
    PetscCheck(PetscAbsScalar(sum) < chargeTol, PetscObjectComm((PetscObject)sw), PETSC_ERR_PLIB, "Charge should have no DC component: %g", (double)PetscRealPart(sum));
    /* Nondimensionalize rho */
    PetscCall(VecScale(rho, phi_0));
  }
  PetscCall(DMGetGlobalVector(dm, &phi));
  PetscCall(PetscObjectSetName((PetscObject)phi, "potential"));
  PetscCall(VecSet(phi, 0.0));
  PetscCall(SNESSolve(snes, NULL, phi));
  PetscCall(DMRestoreGlobalVector(dm, &rho));
  PetscCall(VecViewFromOptions(phi, NULL, "-phi_view"));

  PetscCall(DMGetLocalVector(dm, &locPhi));
  PetscCall(DMGlobalToLocalBegin(dm, phi, INSERT_VALUES, locPhi));
  PetscCall(DMGlobalToLocalEnd(dm, phi, INSERT_VALUES, locPhi));
  PetscCall(DMRestoreGlobalVector(dm, &phi));

  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject *)&fe));
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  for (c = cStart; c < cEnd; ++c) {
    PetscTabulation tab;
    PetscScalar    *clPhi = NULL;
    PetscReal      *pcoord, *refcoord;
    PetscReal       v[3], J[9], invJ[9], detJ;
    PetscInt       *points;
    PetscInt        Ncp, cp;

    PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Ncp, &points));
    PetscCall(DMGetWorkArray(dm, Ncp * dim, MPIU_REAL, &pcoord));
    PetscCall(DMGetWorkArray(dm, Ncp * dim, MPIU_REAL, &refcoord));
    for (cp = 0; cp < Ncp; ++cp)
      for (d = 0; d < dim; ++d) pcoord[cp * dim + d] = coords[points[cp] * dim + d];
    PetscCall(DMPlexCoordinatesToReference(dm, c, Ncp, pcoord, refcoord));
    PetscCall(PetscFECreateTabulation(fe, 1, Ncp, refcoord, 1, &tab));
    PetscCall(DMPlexComputeCellGeometryFEM(dm, c, NULL, v, J, invJ, &detJ));
    PetscCall(DMPlexVecGetClosure(dm, NULL, locPhi, c, NULL, &clPhi));
    for (cp = 0; cp < Ncp; ++cp) {
      const PetscInt p = points[cp];

      for (d = 0; d < dim; ++d) E[p * dim + d] = 0.;
      PetscCall(PetscFEGetQuadrature(fe, &q));
      PetscCall(PetscFECreateCellGeometry(fe, q, &feGeometry));
      PetscCall(PetscFEInterpolateAtPoints_Static(fe, tab, clPhi, &feGeometry, cp, &E[p * dim]));
      PetscCall(PetscFEDestroyCellGeometry(fe, &feGeometry));
    }
    PetscCall(DMPlexVecRestoreClosure(dm, NULL, locPhi, c, NULL, &clPhi));
    PetscCall(DMRestoreWorkArray(dm, Ncp * dim, MPIU_REAL, &pcoord));
    PetscCall(DMRestoreWorkArray(dm, Ncp * dim, MPIU_REAL, &refcoord));
    PetscCall(PetscTabulationDestroy(&tab));
    PetscCall(PetscFree(points));
  }
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmSortRestoreAccess(sw));
  PetscCall(DMRestoreLocalVector(dm, &locPhi));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFieldAtParticles(SNES snes, DM sw, PetscReal E[])
{
  AppCtx  *ctx;
  PetscInt dim, Np;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(sw, DM_CLASSID, 2);
  PetscValidRealPointer(E, 3);
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMGetApplicationContext(sw, &ctx));
  PetscCall(PetscArrayzero(E, Np * dim));

  switch (ctx->em) {
  case EM_PRIMAL:
    PetscCall(ComputeFieldAtParticles_Primal(snes, sw, E));
    break;
  case EM_COULOMB:
    PetscCall(ComputeFieldAtParticles_Coulomb(snes, sw, E));
    break;
  case EM_MIXED:
    PetscCall(ComputeFieldAtParticles_Mixed(snes, sw, E));
    break;
  case EM_NONE:
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No solver for electrostatic model %s", EMTypes[ctx->em]);
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec G, void *ctx)
{
  DM                 sw;
  SNES               snes = ((AppCtx *)ctx)->snes;
  const PetscReal   *coords, *vel;
  const PetscScalar *u;
  PetscScalar       *g;
  PetscReal         *E;
  PetscInt           dim, d, Np, p;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  PetscCall(DMSwarmGetField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscCall(VecGetLocalSize(U, &Np));
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArray(G, &g));

  PetscCall(ComputeFieldAtParticles(snes, sw, E));

  Np /= 2 * dim;
  for (p = 0; p < Np; ++p) {
    const PetscReal x0    = coords[p * dim + 0];
    const PetscReal vy0   = vel[p * dim + 1];
    const PetscReal omega = vy0 / x0;

    for (d = 0; d < dim; ++d) {
      g[(p * 2 + 0) * dim + d] = u[(p * 2 + 1) * dim + d];
      g[(p * 2 + 1) * dim + d] = E[p * dim + d] - PetscSqr(omega) * u[(p * 2 + 0) * dim + d];
    }
  }
  PetscCall(DMSwarmRestoreField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  PetscCall(DMSwarmRestoreField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(G, &g));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* J_{ij} = dF_i/dx_j
   J_p = (  0   1)
         (-w^2  0)
   TODO Now there is another term with w^2 from the electric field. I think we will need to invert the operator.
        Perhaps we can approximate the Jacobian using only the cellwise P-P gradient from Coulomb
*/
static PetscErrorCode RHSJacobian(TS ts, PetscReal t, Vec U, Mat J, Mat P, void *ctx)
{
  DM               sw;
  const PetscReal *coords, *vel;
  PetscInt         dim, d, Np, p, rStart;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(VecGetLocalSize(U, &Np));
  PetscCall(MatGetOwnershipRange(J, &rStart, NULL));
  PetscCall(DMSwarmGetField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  Np /= 2 * dim;
  for (p = 0; p < Np; ++p) {
    const PetscReal x0      = coords[p * dim + 0];
    const PetscReal vy0     = vel[p * dim + 1];
    const PetscReal omega   = vy0 / x0;
    PetscScalar     vals[4] = {0., 1., -PetscSqr(omega), 0.};

    for (d = 0; d < dim; ++d) {
      const PetscInt rows[2] = {(p * 2 + 0) * dim + d + rStart, (p * 2 + 1) * dim + d + rStart};
      PetscCall(MatSetValues(J, 2, rows, 2, rows, vals, INSERT_VALUES));
    }
  }
  PetscCall(DMSwarmRestoreField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSFunctionX(TS ts, PetscReal t, Vec V, Vec Xres, void *ctx)
{
  DM                 sw;
  const PetscScalar *v;
  PetscScalar       *xres;
  PetscInt           Np, p, dim, d;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(VecGetLocalSize(Xres, &Np));
  Np /= dim;
  PetscCall(VecGetArrayRead(V, &v));
  PetscCall(VecGetArray(Xres, &xres));
  for (p = 0; p < Np; ++p) {
    for (d = 0; d < dim; ++d) xres[p * dim + d] = v[p * dim + d];
  }
  PetscCall(VecRestoreArrayRead(V, &v));
  PetscCall(VecRestoreArray(Xres, &xres));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSFunctionV(TS ts, PetscReal t, Vec X, Vec Vres, void *ctx)
{
  DM                 sw;
  SNES               snes = ((AppCtx *)ctx)->snes;
  const PetscScalar *x;
  const PetscReal   *coords, *vel;
  PetscScalar       *vres;
  PetscReal         *E;
  PetscInt           Np, p, dim, d;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  PetscCall(DMSwarmGetField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscCall(VecGetLocalSize(Vres, &Np));
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArray(Vres, &vres));
  PetscCheck(dim == 2, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Dimension must be 2");

  PetscCall(ComputeFieldAtParticles(snes, sw, E));

  Np /= dim;
  for (p = 0; p < Np; ++p) {
    const PetscReal x0    = coords[p * dim + 0];
    const PetscReal vy0   = vel[p * dim + 1];
    const PetscReal omega = vy0 / x0;

    for (d = 0; d < dim; ++d) vres[p * dim + d] = E[p * dim + d] - PetscSqr(omega) * x[p * dim + d];
  }
  PetscCall(VecRestoreArrayRead(X, &x));
  PetscCall(VecRestoreArray(Vres, &vres));
  PetscCall(DMSwarmRestoreField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  PetscCall(DMSwarmRestoreField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSolution(TS ts)
{
  DM       sw;
  Vec      u;
  PetscInt dim, Np;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
  PetscCall(VecSetBlockSize(u, dim));
  PetscCall(VecSetSizes(u, 2 * Np * dim, PETSC_DECIDE));
  PetscCall(VecSetUp(u));
  PetscCall(TSSetSolution(ts, u));
  PetscCall(VecDestroy(&u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetProblem(TS ts)
{
  AppCtx *user;
  DM      sw;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetApplicationContext(sw, (void **)&user));
  // Define unified system for (X, V)
  {
    Mat      J;
    PetscInt dim, Np;

    PetscCall(DMGetDimension(sw, &dim));
    PetscCall(DMSwarmGetLocalSize(sw, &Np));
    PetscCall(MatCreate(PETSC_COMM_WORLD, &J));
    PetscCall(MatSetSizes(J, 2 * Np * dim, 2 * Np * dim, PETSC_DECIDE, PETSC_DECIDE));
    PetscCall(MatSetBlockSize(J, 2 * dim));
    PetscCall(MatSetFromOptions(J));
    PetscCall(MatSetUp(J));
    PetscCall(TSSetRHSFunction(ts, NULL, RHSFunction, user));
    PetscCall(TSSetRHSJacobian(ts, J, J, RHSJacobian, user));
    PetscCall(MatDestroy(&J));
  }
  /* Define split system for X and V */
  {
    Vec             u;
    IS              isx, isv, istmp;
    const PetscInt *idx;
    PetscInt        dim, Np, rstart;

    PetscCall(TSGetSolution(ts, &u));
    PetscCall(DMGetDimension(sw, &dim));
    PetscCall(DMSwarmGetLocalSize(sw, &Np));
    PetscCall(VecGetOwnershipRange(u, &rstart, NULL));
    PetscCall(ISCreateStride(PETSC_COMM_WORLD, Np, (rstart / dim) + 0, 2, &istmp));
    PetscCall(ISGetIndices(istmp, &idx));
    PetscCall(ISCreateBlock(PETSC_COMM_WORLD, dim, Np, idx, PETSC_COPY_VALUES, &isx));
    PetscCall(ISRestoreIndices(istmp, &idx));
    PetscCall(ISDestroy(&istmp));
    PetscCall(ISCreateStride(PETSC_COMM_WORLD, Np, (rstart / dim) + 1, 2, &istmp));
    PetscCall(ISGetIndices(istmp, &idx));
    PetscCall(ISCreateBlock(PETSC_COMM_WORLD, dim, Np, idx, PETSC_COPY_VALUES, &isv));
    PetscCall(ISRestoreIndices(istmp, &idx));
    PetscCall(ISDestroy(&istmp));
    PetscCall(TSRHSSplitSetIS(ts, "position", isx));
    PetscCall(TSRHSSplitSetIS(ts, "momentum", isv));
    PetscCall(ISDestroy(&isx));
    PetscCall(ISDestroy(&isv));
    PetscCall(TSRHSSplitSetRHSFunction(ts, "position", NULL, RHSFunctionX, user));
    PetscCall(TSRHSSplitSetRHSFunction(ts, "momentum", NULL, RHSFunctionV, user));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmTSRedistribute(TS ts)
{
  DM        sw;
  Vec       u;
  PetscReal t, maxt, dt;
  PetscInt  n, maxn;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(TSGetTime(ts, &t));
  PetscCall(TSGetMaxTime(ts, &maxt));
  PetscCall(TSGetTimeStep(ts, &dt));
  PetscCall(TSGetStepNumber(ts, &n));
  PetscCall(TSGetMaxSteps(ts, &maxn));

  PetscCall(TSReset(ts));
  PetscCall(TSSetDM(ts, sw));
  /* TODO Check whether TS was set from options */
  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSSetTime(ts, t));
  PetscCall(TSSetMaxTime(ts, maxt));
  PetscCall(TSSetTimeStep(ts, dt));
  PetscCall(TSSetStepNumber(ts, n));
  PetscCall(TSSetMaxSteps(ts, maxn));

  PetscCall(CreateSolution(ts));
  PetscCall(SetProblem(ts));
  PetscCall(TSGetSolution(ts, &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode circleSingleX(PetscInt dim, PetscReal time, const PetscReal dummy[], PetscInt p, PetscScalar x[], void *ctx)
{
  x[0] = p + 1.;
  x[1] = 0.;
  return PETSC_SUCCESS;
}

PetscErrorCode circleSingleV(PetscInt dim, PetscReal time, const PetscReal dummy[], PetscInt p, PetscScalar v[], void *ctx)
{
  v[0] = 0.;
  v[1] = PetscSqrtReal(1000. / (p + 1.));
  return PETSC_SUCCESS;
}

/* Put 5 particles into each circle */
PetscErrorCode circleMultipleX(PetscInt dim, PetscReal time, const PetscReal dummy[], PetscInt p, PetscScalar x[], void *ctx)
{
  const PetscInt  n   = 5;
  const PetscReal r0  = (p / n) + 1.;
  const PetscReal th0 = (2. * PETSC_PI * (p % n)) / n;

  x[0] = r0 * PetscCosReal(th0);
  x[1] = r0 * PetscSinReal(th0);
  return PETSC_SUCCESS;
}

/* Put 5 particles into each circle */
PetscErrorCode circleMultipleV(PetscInt dim, PetscReal time, const PetscReal dummy[], PetscInt p, PetscScalar v[], void *ctx)
{
  const PetscInt  n     = 5;
  const PetscReal r0    = (p / n) + 1.;
  const PetscReal th0   = (2. * PETSC_PI * (p % n)) / n;
  const PetscReal omega = PetscSqrtReal(1000. / r0) / r0;

  v[0] = -r0 * omega * PetscSinReal(th0);
  v[1] = r0 * omega * PetscCosReal(th0);
  return PETSC_SUCCESS;
}

/*
  InitializeSolveAndSwarm - Set the solution values to the swarm coordinates and velocities, and also possibly set the initial values.

  Input Parameters:
+ ts         - The TS
- useInitial - Flag to also set the initial conditions to the current coordinates and velocities and setup the problem

  Output Parameter:
. u - The initialized solution vector

  Level: advanced

.seealso: InitializeSolve()
*/
static PetscErrorCode InitializeSolveAndSwarm(TS ts, PetscBool useInitial)
{
  DM      sw;
  Vec     u, gc, gv, gc0, gv0;
  IS      isx, isv;
  AppCtx *user;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetApplicationContext(sw, &user));
  if (useInitial) {
    PetscReal v0[1] = {1.};

    PetscCall(DMSwarmInitializeCoordinates(sw));
    PetscCall(DMSwarmInitializeVelocitiesFromOptions(sw, v0));
    PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));
    PetscCall(DMSwarmTSRedistribute(ts));
  }
  PetscCall(TSGetSolution(ts, &u));
  PetscCall(TSRHSSplitGetIS(ts, "position", &isx));
  PetscCall(TSRHSSplitGetIS(ts, "momentum", &isv));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "initCoordinates", &gc0));
  if (useInitial) PetscCall(VecCopy(gc, gc0));
  PetscCall(VecISCopy(u, isx, SCATTER_FORWARD, gc));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "initCoordinates", &gc0));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "velocity", &gv));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "initVelocity", &gv0));
  if (useInitial) PetscCall(VecCopy(gv, gv0));
  PetscCall(VecISCopy(u, isv, SCATTER_FORWARD, gv));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "velocity", &gv));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "initVelocity", &gv0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitializeSolve(TS ts, Vec u)
{
  PetscFunctionBegin;
  PetscCall(TSSetSolution(ts, u));
  PetscCall(InitializeSolveAndSwarm(ts, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeError(TS ts, Vec U, Vec E)
{
  MPI_Comm           comm;
  DM                 sw;
  AppCtx            *user;
  const PetscScalar *u;
  const PetscReal   *coords, *vel;
  PetscScalar       *e;
  PetscReal          t;
  PetscInt           dim, Np, p;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)ts, &comm));
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetApplicationContext(sw, &user));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(TSGetSolveTime(ts, &t));
  PetscCall(VecGetArray(E, &e));
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetLocalSize(U, &Np));
  PetscCall(DMSwarmGetField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  Np /= 2 * dim;
  for (p = 0; p < Np; ++p) {
    /* TODO generalize initial conditions and project into plane instead of assuming x-y */
    const PetscReal    r0    = DMPlex_NormD_Internal(dim, &coords[p * dim]);
    const PetscReal    th0   = PetscAtan2Real(coords[p * dim + 1], coords[p * dim + 0]);
    const PetscReal    v0    = DMPlex_NormD_Internal(dim, &vel[p * dim]);
    const PetscReal    omega = v0 / r0;
    const PetscReal    ct    = PetscCosReal(omega * t + th0);
    const PetscReal    st    = PetscSinReal(omega * t + th0);
    const PetscScalar *x     = &u[(p * 2 + 0) * dim];
    const PetscScalar *v     = &u[(p * 2 + 1) * dim];
    const PetscReal    xe[3] = {r0 * ct, r0 * st, 0.0};
    const PetscReal    ve[3] = {-v0 * st, v0 * ct, 0.0};
    PetscInt           d;

    for (d = 0; d < dim; ++d) {
      e[(p * 2 + 0) * dim + d] = x[d] - xe[d];
      e[(p * 2 + 1) * dim + d] = v[d] - ve[d];
    }
    if (user->error) {
      const PetscReal en   = 0.5 * DMPlex_DotRealD_Internal(dim, v, v);
      const PetscReal exen = 0.5 * PetscSqr(v0);
      PetscCall(PetscPrintf(comm, "t %.4g: p%" PetscInt_FMT " error [%.2g %.2g] sol [(%.6lf %.6lf) (%.6lf %.6lf)] exact [(%.6lf %.6lf) (%.6lf %.6lf)] energy/exact energy %g / %g (%.10lf%%)\n", (double)t, p, (double)DMPlex_NormD_Internal(dim, &e[(p * 2 + 0) * dim]), (double)DMPlex_NormD_Internal(dim, &e[(p * 2 + 1) * dim]), (double)x[0], (double)x[1], (double)v[0], (double)v[1], (double)xe[0], (double)xe[1], (double)ve[0], (double)ve[1], (double)en, (double)exen, (double)(PetscAbsReal(exen - en) * 100. / exen)));
    }
  }
  PetscCall(DMSwarmRestoreField(sw, "initCoordinates", NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(sw, "initVelocity", NULL, NULL, (void **)&vel));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(E, &e));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode EnergyMonitor(TS ts, PetscInt step, PetscReal t, Vec U, void *ctx)
{
  const PetscInt     ostep = ((AppCtx *)ctx)->ostep;
  const EMType       em    = ((AppCtx *)ctx)->em;
  DM                 sw;
  const PetscScalar *u;
  PetscReal         *coords, *E;
  PetscReal          enKin = 0., enEM = 0.;
  PetscInt           dim, d, Np, p, q;

  PetscFunctionBeginUser;
  if (step % ostep == 0) {
    PetscCall(TSGetDM(ts, &sw));
    PetscCall(DMGetDimension(sw, &dim));
    PetscCall(VecGetArrayRead(U, &u));
    PetscCall(VecGetLocalSize(U, &Np));
    Np /= 2 * dim;
    PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
    PetscCall(DMSwarmGetField(sw, "E_field", NULL, NULL, (void **)&E));
    if (!step) PetscCall(PetscPrintf(PetscObjectComm((PetscObject)ts), "Time     Step Part     Energy\n"));
    for (p = 0; p < Np; ++p) {
      const PetscReal v2     = DMPlex_DotRealD_Internal(dim, &u[(p * 2 + 1) * dim], &u[(p * 2 + 1) * dim]);
      PetscReal      *pcoord = &coords[p * dim];

      PetscCall(PetscSynchronizedPrintf(PetscObjectComm((PetscObject)ts), "%.6lf %4" PetscInt_FMT " %5" PetscInt_FMT " %10.4lf\n", (double)t, step, p, (double)(0.5 * v2)));
      enKin += 0.5 * v2;
      if (em == EM_NONE) {
        continue;
      } else if (em == EM_COULOMB) {
        for (q = p + 1; q < Np; ++q) {
          PetscReal *qcoord = &coords[q * dim];
          PetscReal  rpq[3], r;
          for (d = 0; d < dim; ++d) rpq[d] = pcoord[d] - qcoord[d];
          r = DMPlex_NormD_Internal(dim, rpq);
          enEM += 1. / r;
        }
      } else if (em == EM_PRIMAL || em == EM_MIXED) {
        for (d = 0; d < dim; ++d) enEM += E[p * dim + d];
      }
    }
    PetscCall(PetscSynchronizedPrintf(PetscObjectComm((PetscObject)ts), "%.6lf %4" PetscInt_FMT " KE\t    %10.4lf\n", (double)t, step, (double)enKin));
    PetscCall(PetscSynchronizedPrintf(PetscObjectComm((PetscObject)ts), "%.6lf %4" PetscInt_FMT " PE\t    %1.10g\n", (double)t, step, (double)enEM));
    PetscCall(PetscSynchronizedPrintf(PetscObjectComm((PetscObject)ts), "%.6lf %4" PetscInt_FMT " E\t    %10.4lf\n", (double)t, step, (double)(enKin + enEM)));
    PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
    PetscCall(DMSwarmRestoreField(sw, "E_field", NULL, NULL, (void **)&E));
    PetscCall(PetscSynchronizedFlush(PetscObjectComm((PetscObject)ts), NULL));
    PetscCall(VecRestoreArrayRead(U, &u));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MigrateParticles(TS ts)
{
  DM sw;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMViewFromOptions(sw, NULL, "-migrate_view_pre"));
  {
    Vec u, gc, gv;
    IS  isx, isv;

    PetscCall(TSGetSolution(ts, &u));
    PetscCall(TSRHSSplitGetIS(ts, "position", &isx));
    PetscCall(TSRHSSplitGetIS(ts, "momentum", &isv));
    PetscCall(DMSwarmCreateGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
    PetscCall(VecISCopy(u, isx, SCATTER_REVERSE, gc));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
    PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "velocity", &gv));
    PetscCall(VecISCopy(u, isv, SCATTER_REVERSE, gv));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "velocity", &gv));
  }
  PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));
  PetscCall(DMSwarmTSRedistribute(ts));
  PetscCall(InitializeSolveAndSwarm(ts, PETSC_FALSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM     dm, sw;
  TS     ts;
  Vec    u;
  AppCtx user;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(CreatePoisson(dm, &user));
  PetscCall(CreateSwarm(dm, &user, &sw));
  PetscCall(DMSetApplicationContext(sw, &user));

  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetDM(ts, sw));
  PetscCall(TSSetMaxTime(ts, 0.1));
  PetscCall(TSSetTimeStep(ts, 0.00001));
  PetscCall(TSSetMaxSteps(ts, 100));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));
  PetscCall(TSMonitorSet(ts, EnergyMonitor, &user, NULL));
  PetscCall(TSSetFromOptions(ts));
  PetscCall(DMSwarmVectorDefineField(sw, "velocity"));
  PetscCall(TSSetComputeInitialCondition(ts, InitializeSolve));
  PetscCall(TSSetComputeExactError(ts, ComputeError));
  PetscCall(TSSetPostStep(ts, MigrateParticles));

  PetscCall(CreateSolution(ts));
  PetscCall(TSGetSolution(ts, &u));
  PetscCall(TSComputeInitialCondition(ts, u));
  PetscCall(TSSolve(ts, NULL));

  PetscCall(SNESDestroy(&user.snes));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&sw));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
     requires: double !complex

   testset:
     requires: defined(PETSC_HAVE_EXECUTABLE_EXPORT)
     args: -dm_plex_dim 2 -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -dm_plex_box_lower -5,-5 -dm_plex_box_upper 5,5 \
           -dm_swarm_num_particles 2 -dm_swarm_coordinate_function circleSingleX -dm_swarm_velocity_function circleSingleV \
           -ts_type basicsymplectic\
           -dm_view -output_step 50 -ts_dt 0.01 -ts_max_time 10.0 -ts_max_steps 10
     test:
       suffix: none_bsi_2d_1
       args: -ts_basicsymplectic_type 1 -em_type none -error
     test:
       suffix: none_bsi_2d_2
       args: -ts_basicsymplectic_type 2 -em_type none -error
     test:
       suffix: none_bsi_2d_3
       args: -ts_basicsymplectic_type 3 -em_type none -error
     test:
       suffix: none_bsi_2d_4
       args: -ts_basicsymplectic_type 4 -em_type none -error
     test:
       suffix: coulomb_bsi_2d_1
       args: -ts_basicsymplectic_type 1
     test:
       suffix: coulomb_bsi_2d_2
       args: -ts_basicsymplectic_type 2
     test:
       suffix: coulomb_bsi_2d_3
       args: -ts_basicsymplectic_type 3
     test:
       suffix: coulomb_bsi_2d_4
       args: -ts_basicsymplectic_type 4

   testset:
     requires: defined(PETSC_HAVE_EXECUTABLE_EXPORT)
     args: -dm_plex_dim 2 -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -dm_plex_box_lower -5,-5 -dm_plex_box_upper 5,5 \
           -dm_swarm_num_particles 2 -dm_swarm_coordinate_function circleSingleX -dm_swarm_velocity_function circleSingleV \
           -ts_type basicsymplectic\
           -em_type primal -em_pc_type svd\
           -dm_view -output_step 50 -error -ts_dt 0.01 -ts_max_time 10.0 -ts_max_steps 10\
           -petscspace_degree 2 -petscfe_default_quadrature_order 3 -sigma 1.0e-8 -timeScale 2.0e-14
     test:
       suffix: poisson_bsi_2d_1
       args: -ts_basicsymplectic_type 1
     test:
       suffix: poisson_bsi_2d_2
       args: -ts_basicsymplectic_type 2
     test:
       suffix: poisson_bsi_2d_3
       args: -ts_basicsymplectic_type 3
     test:
       suffix: poisson_bsi_2d_4
       args: -ts_basicsymplectic_type 4

   testset:
     requires: defined(PETSC_HAVE_EXECUTABLE_EXPORT)
     args: -dm_swarm_num_particles 2 -dm_swarm_coordinate_function circleSingleX -dm_swarm_velocity_function circleSingleV \
           -ts_type theta -ts_theta_theta 0.5 -ts_convergence_estimate -convest_num_refine 2 \
             -mat_type baij -ksp_error_if_not_converged -em_pc_type svd\
           -dm_view -output_step 50 -error -ts_dt 0.01 -ts_max_time 10.0 -ts_max_steps 10\
           -pc_type svd -sigma 1.0e-8 -timeScale 2.0e-14
     test:
       suffix: im_2d_0
       args: -dm_plex_dim 2 -dm_plex_simplex 0 -dm_plex_box_faces 1,1 -dm_plex_box_lower -5,-5 -dm_plex_box_upper 5,5

   testset:
     requires: defined(PETSC_HAVE_EXECUTABLE_EXPORT)
     args: -dm_plex_dim 2 -dm_plex_simplex 0 -dm_plex_box_faces 10,10 -dm_plex_box_lower -5,-5 -dm_plex_box_upper 5,5 -petscpartitioner_type simple \
           -dm_swarm_num_particles 2 -dm_swarm_coordinate_function circleSingleX -dm_swarm_velocity_function circleSingleV -dm_swarm_num_species 1\
           -ts_type basicsymplectic -ts_convergence_estimate -convest_num_refine 2 \
           -em_snes_type ksponly -em_pc_type svd -em_type primal -petscspace_degree 1\
           -dm_view -output_step 50\
           -pc_type svd -sigma 1.0e-8 -timeScale 2.0e-14 -ts_dt 0.01 -ts_max_time 1.0 -ts_max_steps 10
     test:
       suffix: bsi_2d_mesh_1
       args: -ts_basicsymplectic_type 4
     test:
       suffix: bsi_2d_mesh_1_par_2
       nsize: 2
       args: -ts_basicsymplectic_type 4
     test:
       suffix: bsi_2d_mesh_1_par_3
       nsize: 3
       args: -ts_basicsymplectic_type 4
     test:
       suffix: bsi_2d_mesh_1_par_4
       nsize: 4
       args: -ts_basicsymplectic_type 4 -dm_swarm_num_particles 0,0,2,0

   testset:
     requires: defined(PETSC_HAVE_EXECUTABLE_EXPORT)
     args: -dm_plex_dim 2 -dm_plex_simplex 0 -dm_plex_box_faces 10,10 -dm_plex_box_lower -5,-5 -dm_plex_box_upper 5,5 \
           -dm_swarm_num_particles 10 -dm_swarm_coordinate_function circleMultipleX -dm_swarm_velocity_function circleMultipleV \
           -ts_convergence_estimate -convest_num_refine 2 \
             -em_pc_type lu\
           -dm_view -output_step 50 -error\
           -pc_type svd -sigma 1.0e-8 -timeScale 2.0e-14 -ts_dt 0.01 -ts_max_time 10.0 -ts_max_steps 10
     test:
       suffix: bsi_2d_multiple_1
       args: -ts_type basicsymplectic -ts_basicsymplectic_type 1
     test:
       suffix: bsi_2d_multiple_2
       args: -ts_type basicsymplectic -ts_basicsymplectic_type 2
     test:
       suffix: bsi_2d_multiple_3
       args: -ts_type basicsymplectic -ts_basicsymplectic_type 3 -ts_dt 0.001
     test:
       suffix: im_2d_multiple_0
       args: -ts_type theta -ts_theta_theta 0.5 \
               -mat_type baij -ksp_error_if_not_converged -em_pc_type lu

   testset:
     requires: defined(PETSC_HAVE_EXECUTABLE_EXPORT)
     args: -dm_plex_dim 2 -dm_plex_simplex 0 -dm_plex_box_faces 2,2 -dm_plex_box_lower -5,-5 -dm_plex_box_upper 5,5 \
           -dm_swarm_num_particles 2 -dm_swarm_coordinate_function circleSingleX -dm_swarm_velocity_function circleSingleV \
           -em_pc_type fieldsplit -ksp_rtol 1e-10 -em_ksp_type preonly -em_type mixed -em_ksp_error_if_not_converged\
           -dm_view -output_step 50 -error -dm_refine 0\
           -pc_type svd -sigma 1.0e-8 -timeScale 2.0e-14 -ts_dt 0.01 -ts_max_time 10.0 -ts_max_steps 10
     test:
       suffix: bsi_4_rt_1
       args: -ts_type basicsymplectic -ts_basicsymplectic_type 4\
             -pc_fieldsplit_detect_saddle_point\
             -pc_type fieldsplit\
             -pc_fieldsplit_type schur\
             -pc_fieldsplit_schur_precondition full \
             -field_petscspace_degree 2\
             -field_petscfe_default_quadrature_order 1\
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
             -field_petscdualspace_form_degree -1 \
             -field_petscdualspace_order 1 \
             -field_petscdualspace_lagrange_trimmed true\
             -ksp_gmres_restart 500

TEST*/
