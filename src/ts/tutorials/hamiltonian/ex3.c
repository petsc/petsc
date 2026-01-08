static char help[] = "Landau Damping test using Vlasov-Poisson equations\n";

/*
  This example solves the Vlasov-Poisson system for Landau damping (6X + 6V).
  The system is solved using a Particle-In-Cell (PIC) method with DMSwarm for particles and DMPlex/PetscFE for the field.
  This particular example uses the velocity mesh from DMPlexLandauCreateVelocitySpace for 3D velocity space.

  Options:
  -particle_monitor [prefix] : Output particle data (x, v, w, E) to binary files at each output step.
                               Optional prefix for filenames (default: "particles").

*/
#include <petscts.h>
#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscfe.h>
#include <petscds.h>
#include <petscbag.h>
#include <petscdraw.h>
#include <petscviewer.h>
#include <petsclandau.h>
#include <petscdmcomposite.h>
#include <petsc/private/dmpleximpl.h>  /* For norm and dot */
#include <petsc/private/petscfeimpl.h> /* For interpolation */
#include <petsc/private/dmswarmimpl.h> /* For swarm debugging */
#include "petscdm.h"
#include "petscdmlabel.h"

PETSC_EXTERN PetscErrorCode stream(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);
PETSC_EXTERN PetscErrorCode line(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);

typedef enum {
  V0,
  X0,
  T0,
  M0,
  Q0,
  PHI0,
  POISSON,
  VLASOV,
  SIGMA,
  NUM_CONSTANTS
} ConstantType;
typedef struct {
  PetscScalar v0; /* Velocity scale, often the thermal velocity */
  PetscScalar t0; /* Time scale */
  PetscScalar x0; /* Space scale */
  PetscScalar m0; /* Mass scale */
  PetscScalar q0; /* Charge scale */
  PetscScalar kb;
  PetscScalar epsi0;
  PetscScalar phi0;          /* Potential scale */
  PetscScalar poissonNumber; /* Non-Dimensional Poisson Number */
  PetscScalar vlasovNumber;  /* Non-Dimensional Vlasov Number */
  PetscReal   sigma;         /* Nondimensional charge per length in x */
} Parameter;

typedef struct {
  PetscBag      bag;            // Problem parameters
  PetscBool     error;          // Flag for printing the error
  PetscBool     efield_monitor; // Flag to show electric field monitor
  PetscBool     moment_monitor; // Flag to show distribution moment monitor
  char          particle_monitor_prefix[PETSC_MAX_PATH_LEN];
  PetscBool     particle_monitor; // Flag to output particle data
  PetscInt      ostep;            // Print the energy at each ostep time steps
  PetscInt      numParticles;
  PetscReal     timeScale;              /* Nondimensionalizing time scale */
  PetscReal     charges[2];             /* The charges of each species */
  PetscReal     masses[2];              /* The masses of each species */
  PetscReal     thermal_energy[2];      /* Thermal Energy (used to get other constants)*/
  PetscReal     cosine_coefficients[2]; /*(alpha, k)*/
  PetscReal     totalWeight;
  PetscReal     stepSize;
  PetscInt      steps;
  PetscReal     initVel;
  SNES          snes;       // EM solver
  DM            dmPot;      // The DM for potential
  Mat           M;          // The finite element mass matrix for potential
  PetscFEGeom  *fegeom;     // Geometric information for the DM cells
  PetscBool     validE;     // Flag to indicate E-field in swarm is valid
  PetscReal     drawlgEmin; // The minimum lg(E) to plot
  PetscDrawLG   drawlgE;    // Logarithm of maximum electric field
  DM            swarm;
  PetscBool     checkweights;
  PetscInt      checkVRes; // Flag to check/output velocity residuals for nightly tests
  DM            landau_pack;
  PetscBool     use_landau_velocity_space;
  PetscLogEvent RhsXEvent, RhsVEvent, ESolveEvent, ETabEvent;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  PetscInt d                         = 2;
  PetscInt maxSpecies                = 2;
  options->error                     = PETSC_FALSE;
  options->efield_monitor            = PETSC_FALSE;
  options->moment_monitor            = PETSC_FALSE;
  options->particle_monitor          = PETSC_FALSE;
  options->ostep                     = 100;
  options->timeScale                 = 2.0e-14;
  options->charges[0]                = -1.0;
  options->charges[1]                = 1.0;
  options->masses[0]                 = 1.0;
  options->masses[1]                 = 1000.0;
  options->thermal_energy[0]         = 1.0;
  options->thermal_energy[1]         = 1.0;
  options->cosine_coefficients[0]    = 0.01;
  options->cosine_coefficients[1]    = 0.5;
  options->initVel                   = 1;
  options->totalWeight               = 1.0;
  options->validE                    = PETSC_FALSE;
  options->drawlgEmin                = -6;
  options->drawlgE                   = NULL;
  options->snes                      = NULL;
  options->dmPot                     = NULL;
  options->M                         = NULL;
  options->numParticles              = 32768;
  options->checkweights              = PETSC_FALSE;
  options->checkVRes                 = 0;
  options->landau_pack               = NULL;
  options->use_landau_velocity_space = PETSC_FALSE;

  PetscOptionsBegin(comm, "", "Landau Damping options", "DMSWARM");
  PetscCall(PetscOptionsBool("-error", "Flag to print the error", "ex3.c", options->error, &options->error, NULL));
  PetscCall(PetscOptionsBool("-efield_monitor", "Flag to plot log(max E) over time", "ex3.c", options->efield_monitor, &options->efield_monitor, NULL));
  PetscCall(PetscOptionsReal("-efield_min_monitor", "Minimum E field to plot", "ex3.c", options->drawlgEmin, &options->drawlgEmin, NULL));
  PetscCall(PetscOptionsBool("-moments_monitor", "Flag to show moments table", "ex3.c", options->moment_monitor, &options->moment_monitor, NULL));
  PetscCall(PetscOptionsString("-particle_monitor", "Prefix for particle data files", "ex3.c", options->particle_monitor_prefix, options->particle_monitor_prefix, sizeof(options->particle_monitor_prefix), &options->particle_monitor));
  PetscCall(PetscOptionsBool("-check_weights", "Ensure all particle weights are positive", "ex3.c", options->checkweights, &options->checkweights, NULL));
  PetscCall(PetscOptionsInt("-check_vel_res", "Check particle velocity residuals for nightly tests", "ex3.c", options->checkVRes, &options->checkVRes, NULL));
  PetscCall(PetscOptionsInt("-output_step", "Number of time steps between output", "ex3.c", options->ostep, &options->ostep, NULL));
  PetscCall(PetscOptionsReal("-timeScale", "Nondimensionalizing time scale", "ex3.c", options->timeScale, &options->timeScale, NULL));
  PetscCall(PetscOptionsReal("-initial_velocity", "Initial velocity of perturbed particle", "ex3.c", options->initVel, &options->initVel, NULL));
  PetscCall(PetscOptionsReal("-total_weight", "Total weight of all particles", "ex3.c", options->totalWeight, &options->totalWeight, NULL));
  PetscCall(PetscOptionsRealArray("-cosine_coefficients", "Amplitude and frequency of cosine equation used in initialization", "ex3.c", options->cosine_coefficients, &d, NULL));
  PetscCall(PetscOptionsRealArray("-charges", "Species charges", "ex3.c", options->charges, &maxSpecies, NULL));
  PetscCall(PetscOptionsBool("-use_landau_velocity_space", "Use Landau velocity space", "ex3.c", options->use_landau_velocity_space, &options->use_landau_velocity_space, NULL));
  PetscOptionsEnd();

  PetscCall(PetscLogEventRegister("RhsX", TS_CLASSID, &options->RhsXEvent));
  PetscCall(PetscLogEventRegister("RhsV", TS_CLASSID, &options->RhsVEvent));
  PetscCall(PetscLogEventRegister("ESolve", TS_CLASSID, &options->ESolveEvent));
  PetscCall(PetscLogEventRegister("ETab", TS_CLASSID, &options->ETabEvent));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupContext(DM dm, DM sw, AppCtx *user)
{
  MPI_Comm comm;

  PetscFunctionBeginUser;
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  if (user->efield_monitor) {
    PetscDraw     draw;
    PetscDrawAxis axis;

    PetscCall(PetscDrawCreate(comm, NULL, "Max Electric Field", 0, 300, 400, 300, &draw));
    PetscCall(PetscDrawSetSave(draw, "ex3_Efield"));
    PetscCall(PetscDrawSetFromOptions(draw));
    PetscCall(PetscDrawLGCreate(draw, 1, &user->drawlgE));
    PetscCall(PetscDrawDestroy(&draw));
    PetscCall(PetscDrawLGGetAxis(user->drawlgE, &axis));
    PetscCall(PetscDrawAxisSetLabels(axis, "Electron Electric Field", "time", "E_max"));
    PetscCall(PetscDrawLGSetLimits(user->drawlgE, 0., user->steps * user->stepSize, user->drawlgEmin, 0.));
    PetscCall(PetscDrawLGSetUseMarkers(user->drawlgE, PETSC_FALSE));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroyContext(AppCtx *user)
{
  PetscFunctionBeginUser;
  if (user->landau_pack) PetscCall(DMPlexLandauDestroyVelocitySpace(&user->landau_pack));
  PetscCall(PetscDrawLGDestroy(&user->drawlgE));
  PetscCall(PetscBagDestroy(&user->bag));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckNonNegativeWeights(DM sw, AppCtx *user)
{
  const PetscScalar *w;
  PetscInt           Np;

  PetscFunctionBeginUser;
  if (!user->checkweights) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&w));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  for (PetscInt p = 0; p < Np; ++p) PetscCheck(w[p] >= 0.0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Particle %" PetscInt_FMT " has negative weight %g", p, w[p]);
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&w));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void f0_Dirichlet(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  for (PetscInt d = 0; d < dim; ++d) f0[0] += 0.5 * PetscSqr(u_x[d]);
}

static PetscErrorCode computeFieldEnergy(DM dm, Vec u, PetscReal *En)
{
  PetscDS        ds;
  const PetscInt field = 0;
  PetscInt       Nf;
  void          *ctx;

  PetscFunctionBegin;
  PetscCall(DMGetApplicationContext(dm, &ctx));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSGetNumFields(ds, &Nf));
  PetscCheck(Nf == 1, PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_WRONG, "We currently only support 1 field, not %" PetscInt_FMT, Nf);
  PetscCall(PetscDSSetObjective(ds, field, &f0_Dirichlet));
  PetscCall(DMPlexComputeIntegralFEM(dm, u, En, ctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void f0_grad_phi2(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 0.;
  for (PetscInt d = 0; d < dim; ++d) f0[0] += PetscSqr(u_x[uOff_x[0] + d]); // + d * dim  cause segfault
}

static PetscErrorCode MonitorEField(TS ts, PetscInt step, PetscReal t, Vec U, void *ctx)
{
  AppCtx     *user = (AppCtx *)ctx;
  DM          sw;
  PetscScalar intESq;
  PetscReal  *E, *x, *weight;
  PetscReal   Enorm = 0., lgEnorm, lgEmax, sum = 0., Emax = 0., chargesum = 0.;
  PetscReal   pmoments[16]; /* \int f, \int v f, \int v^2 f */
  PetscInt   *species, dim, Np, gNp;
  MPI_Comm    comm;

  PetscFunctionBeginUser;
  if (step < 0 || !user->validE) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectGetComm((PetscObject)ts, &comm));
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMSwarmGetSize(sw, &gNp));
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
  PetscCall(DMSwarmGetField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void **)&species));
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weight));

  for (PetscInt p = 0; p < Np; ++p) {
    PetscReal Emag = 0.;
    for (PetscInt d = 0; d < dim; ++d) {
      PetscReal temp = PetscAbsReal(E[p * dim + d]);
      if (temp > Emax) Emax = temp;
      Emag += PetscSqr(E[p * dim + d]);
    }
    Enorm += PetscSqrtReal(Emag);
    sum += E[p * dim];
    chargesum += user->charges[0] * weight[p];
  }
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &Emax, 1, MPIU_REAL, MPIU_MAX, comm));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &Enorm, 1, MPIU_REAL, MPIU_SUM, comm));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &sum, 1, MPIU_REAL, MPIU_SUM, comm));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &chargesum, 1, MPIU_REAL, MPIU_SUM, comm));
  lgEnorm = Enorm != 0 ? PetscLog10Real(Enorm) : -16.;
  lgEmax  = Emax != 0 ? PetscLog10Real(Emax) : user->drawlgEmin;
  if (lgEmax < user->drawlgEmin) lgEmax = user->drawlgEmin;

  PetscDS ds;
  Vec     phi;

  PetscCall(DMGetNamedGlobalVector(user->dmPot, "phi", &phi));
  PetscCall(DMGetDS(user->dmPot, &ds));
  PetscCall(PetscDSSetObjective(ds, 0, &f0_grad_phi2));
  PetscCall(DMPlexComputeIntegralFEM(user->dmPot, phi, &intESq, user));
  PetscCall(DMRestoreNamedGlobalVector(user->dmPot, "phi", &phi));

  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weight));
  PetscCall(DMSwarmRestoreField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **)&species));
  PetscCall(PetscDrawLGAddPoint(user->drawlgE, &t, &lgEmax));
  PetscCall(PetscDrawLGDraw(user->drawlgE));
  PetscDraw draw;
  PetscCall(PetscDrawLGGetDraw(user->drawlgE, &draw));
  PetscCall(PetscDrawSave(draw));

  PetscCall(DMSwarmComputeMoments(sw, "velocity", "w_q", pmoments));
  PetscCall(PetscPrintf(comm, "E: %f\t%+e\t%e\t%f\t%20.15e\t%f\t%f\t%f\t%20.15e\t%20.15e\t%20.15e\t%" PetscInt_FMT "\t(%" PetscInt_FMT ")\n", (double)t, (double)sum, (double)Enorm, (double)lgEnorm, (double)Emax, (double)lgEmax, (double)chargesum, (double)pmoments[0], (double)pmoments[1], (double)pmoments[1 + dim], (double)PetscSqrtReal(intESq), gNp, step));
  PetscCall(DMViewFromOptions(sw, NULL, "-sw_efield_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MonitorMoments(TS ts, PetscInt step, PetscReal t, Vec U, void *ctx)
{
  DM        sw;
  PetscReal pmoments[16]; /* \int f, \int v f, \int v^2 f */

  PetscFunctionBeginUser;
  if (step < 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TSGetDM(ts, &sw));

  PetscCall(DMSwarmComputeMoments(sw, "velocity", "w_q", pmoments));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%f\t%f\t%f\t%f\n", (double)t, (double)pmoments[0], (double)pmoments[1], (double)pmoments[3]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MonitorParticles(TS ts, PetscInt step, PetscReal t, Vec U, void *ctx)
{
  AppCtx            *user = (AppCtx *)ctx;
  DM                 sw;
  PetscInt           Np, dim;
  const PetscReal   *x, *v, *E;
  const PetscScalar *w;
  PetscViewer        viewer;
  char               filename[64];
  MPI_Comm           comm;

  PetscFunctionBeginUser;
  if (step < 0 || step % user->ostep != 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(PetscObjectGetComm((PetscObject)ts, &comm));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));

  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&v));
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&w));
  PetscCall(DMSwarmGetField(sw, "E_field", NULL, NULL, (void **)&E));

  if (user->particle_monitor_prefix[0]) {
    PetscCall(PetscSNPrintf(filename, sizeof(filename), "%s_step_%d.bin", user->particle_monitor_prefix, (int)step));
  } else {
    PetscCall(PetscSNPrintf(filename, sizeof(filename), "particles_step_%d.bin", (int)step));
  }
  PetscCall(PetscViewerBinaryOpen(comm, filename, FILE_MODE_WRITE, &viewer));

  {
    Vec vx, vv, vw, vE;
    PetscCall(VecCreateMPIWithArray(comm, dim, Np * dim, PETSC_DECIDE, x, &vx));
    PetscCall(VecCreateMPIWithArray(comm, dim, Np * dim, PETSC_DECIDE, v, &vv));
    PetscCall(VecCreateMPIWithArray(comm, 1, Np, PETSC_DECIDE, w, &vw));
    PetscCall(VecCreateMPIWithArray(comm, dim, Np * dim, PETSC_DECIDE, E, &vE));

    PetscCall(VecView(vx, viewer));
    PetscCall(VecView(vv, viewer));
    PetscCall(VecView(vw, viewer));
    PetscCall(VecView(vE, viewer));

    PetscCall(VecDestroy(&vx));
    PetscCall(VecDestroy(&vv));
    PetscCall(VecDestroy(&vw));
    PetscCall(VecDestroy(&vE));
  }
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&v));
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&w));
  PetscCall(DMSwarmRestoreField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscPDFPertubedConstant2D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = (1. + 0.01 * PetscCosReal(0.5 * x[0])) / (4. * PETSC_PI * 4. * PETSC_PI);
  return PETSC_SUCCESS;
}
PetscErrorCode PetscPDFPertubedConstant1D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = (1. + 0.01 * PetscCosReal(0.5 * x[0])) / (2 * PETSC_PI);
  return PETSC_SUCCESS;
}

PetscErrorCode PetscPDFCosine1D(const PetscReal x[], const PetscReal scale[], PetscReal p[])
{
  const PetscReal alpha = scale ? scale[0] : 0.0;
  const PetscReal k     = scale ? scale[1] : 1.;
  p[0]                  = (1 + alpha * PetscCosReal(k * x[0]));
  return PETSC_SUCCESS;
}

PetscErrorCode PetscPDFCosine2D(const PetscReal x[], const PetscReal scale[], PetscReal p[])
{
  const PetscReal alpha = scale ? scale[0] : 0.;
  const PetscReal k     = scale ? scale[1] : 1.;
  p[0]                  = (1 + alpha * PetscCosReal(k * x[0]));
  return PETSC_SUCCESS;
}

PetscErrorCode PetscPDFCosine3D(const PetscReal x[], const PetscReal scale[], PetscReal p[])
{
  const PetscReal alpha = scale ? scale[0] : 0.;
  const PetscReal k     = scale ? scale[1] : 1.;
  p[0]                  = (1 + alpha * PetscCosReal(k * x[0]));
  return PETSC_SUCCESS;
}

static PetscErrorCode CreateVelocityDM(DM sw, DM *vdm)
{
  PetscFE        fe;
  DMPolytopeType ct;
  PetscInt       dim, cStart;
  const char    *prefix = "v";
  AppCtx        *user;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMGetApplicationContext(sw, &user));
  if (dim == 3 && user->use_landau_velocity_space) {
    LandauCtx *ctx;
    Vec        X;
    Mat        J;

    PetscCall(DMPlexLandauCreateVelocitySpace(PETSC_COMM_SELF, dim, prefix, &X, &J, &user->landau_pack));
    PetscCall(DMGetApplicationContext(user->landau_pack, &ctx));
    *vdm = ctx->plex[0];
    PetscCall(PetscObjectReference((PetscObject)*vdm));
    PetscCall(VecDestroy(&X));
    PetscCall(PetscObjectSetName((PetscObject)*vdm, "velocity"));
  } else {
    PetscCall(DMCreate(PETSC_COMM_SELF, vdm));
    PetscCall(DMSetType(*vdm, DMPLEX));
    PetscCall(DMPlexSetOptionsPrefix(*vdm, prefix));
    PetscCall(DMSetFromOptions(*vdm));
    PetscCall(PetscObjectSetName((PetscObject)*vdm, "velocity"));
  }
  PetscCall(DMViewFromOptions(*vdm, NULL, "-dm_view"));

  PetscCall(DMGetDimension(*vdm, &dim));
  PetscCall(DMPlexGetHeightStratum(*vdm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(*vdm, cStart, &ct));
  PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, prefix, PETSC_DETERMINE, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "distribution"));
  PetscCall(DMSetField(*vdm, 0, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(*vdm));
  PetscCall(PetscFEDestroy(&fe));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  InitializeParticles_Centroid - Initialize a regular grid of particles.

  Input Parameters:
+ sw      - The `DMSWARM`
- force1D - Treat the spatial domain as 1D

  Notes:
  This functions sets the species, cellid, spatial coordinate, and velocity fields for all particles.

  It places one particle in the centroid of each cell in the implicit tensor product of the spatial
  and velocity meshes.
*/
static PetscErrorCode InitializeParticles_Centroid(DM sw)
{
  DM_Swarm     *swarm = (DM_Swarm *)sw->data;
  DMSwarmCellDM celldm;
  DM            xdm, vdm;
  PetscReal     vmin[3], vmax[3];
  PetscReal    *x, *v;
  PetscInt     *species, *cellid;
  PetscInt      dim, xcStart, xcEnd, vcStart, vcEnd, Ns, Np, Npc, debug;
  PetscBool     flg;
  MPI_Comm      comm;
  const char   *cellidname;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)sw, &comm));

  PetscOptionsBegin(comm, "", "DMSwarm Options", "DMSWARM");
  PetscCall(DMSwarmGetNumSpecies(sw, &Ns));
  PetscCall(PetscOptionsInt("-dm_swarm_num_species", "The number of species", "DMSwarmSetNumSpecies", Ns, &Ns, &flg));
  if (flg) PetscCall(DMSwarmSetNumSpecies(sw, Ns));
  PetscCall(PetscOptionsBoundedInt("-dm_swarm_print_coords", "Debug output level for particle coordinate computations", "InitializeParticles", 0, &swarm->printCoords, NULL, 0));
  PetscCall(PetscOptionsBoundedInt("-dm_swarm_print_weights", "Debug output level for particle weight computations", "InitializeWeights", 0, &swarm->printWeights, NULL, 0));
  PetscOptionsEnd();
  debug = swarm->printCoords;

  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetCellDM(sw, &xdm));
  PetscCall(DMPlexGetHeightStratum(xdm, 0, &xcStart, &xcEnd));

  PetscCall(DMSwarmGetCellDMByName(sw, "velocity", &celldm));
  PetscCall(DMSwarmCellDMGetDM(celldm, &vdm));
  PetscCall(DMPlexGetHeightStratum(vdm, 0, &vcStart, &vcEnd));

  // One particle per centroid on the tensor product grid
  Npc = (vcEnd - vcStart) * Ns;
  Np  = (xcEnd - xcStart) * Npc;
  PetscCall(DMSwarmSetLocalSizes(sw, Np, 0));
  if (debug) {
    PetscInt gNp, gNc, Nc = xcEnd - xcStart;
    PetscCallMPI(MPIU_Allreduce(&Np, &gNp, 1, MPIU_INT, MPIU_SUM, comm));
    PetscCall(PetscPrintf(comm, "Global Np = %" PetscInt_FMT "\n", gNp));
    PetscCallMPI(MPIU_Allreduce(&Nc, &gNc, 1, MPIU_INT, MPIU_SUM, comm));
    PetscCall(PetscPrintf(comm, "Global X-cells = %" PetscInt_FMT "\n", gNc));
    PetscCall(PetscPrintf(comm, "Global V-cells = %" PetscInt_FMT "\n", vcEnd - vcStart));
  }

  // Set species and cellid
  PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellidname));
  PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void **)&species));
  PetscCall(DMSwarmGetField(sw, cellidname, NULL, NULL, (void **)&cellid));
  for (PetscInt c = 0, p = 0; c < xcEnd - xcStart; ++c) {
    for (PetscInt s = 0; s < Ns; ++s) {
      for (PetscInt q = 0; q < Npc / Ns; ++q, ++p) {
        species[p] = s;
        cellid[p]  = c;
      }
    }
  }
  PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **)&species));
  PetscCall(DMSwarmRestoreField(sw, cellidname, NULL, NULL, (void **)&cellid));

  // Set particle coordinates
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&v));
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(DMGetBoundingBox(vdm, vmin, vmax));
  PetscCall(DMGetCoordinatesLocalSetUp(xdm));
  PetscCall(DMGetCoordinatesLocalSetUp(vdm));
  for (PetscInt c = 0; c < xcEnd - xcStart; ++c) {
    const PetscInt cell = c + xcStart;
    PetscInt      *pidx, Npc;
    PetscReal      centroid[3], volume;

    PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Npc, &pidx));
    PetscCall(DMPlexComputeCellGeometryFVM(xdm, cell, &volume, centroid, NULL));
    for (PetscInt s = 0; s < Ns; ++s) {
      for (PetscInt q = 0; q < Npc / Ns; ++q) {
        const PetscInt p  = pidx[q * Ns + s];
        const PetscInt vc = vcStart + q;
        PetscReal      vcentroid[3], vvolume;

        PetscCall(DMPlexComputeCellGeometryFVM(vdm, vc, &vvolume, vcentroid, NULL));
        for (PetscInt d = 0; d < dim; ++d) {
          x[p * dim + d] = centroid[d];
          v[p * dim + d] = vcentroid[d];
        }

        if (debug > 1) {
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "Particle %4" PetscInt_FMT " ", p));
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "  x: ("));
          for (PetscInt d = 0; d < dim; ++d) {
            if (d > 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, ", "));
            PetscCall(PetscPrintf(PETSC_COMM_SELF, "%g", x[p * dim + d]));
          }
          PetscCall(PetscPrintf(PETSC_COMM_SELF, ") v:("));
          for (PetscInt d = 0; d < dim; ++d) {
            if (d > 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, ", "));
            PetscCall(PetscPrintf(PETSC_COMM_SELF, "%g", v[p * dim + d]));
          }
          PetscCall(PetscPrintf(PETSC_COMM_SELF, ")\n"));
        }
      }
    }
    PetscCall(DMSwarmSortRestorePointsPerCell(sw, c, &Npc, &pidx));
  }
  PetscCall(DMSwarmSortRestoreAccess(sw));
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  InitializeWeights - Compute weight for each local particle

  Input Parameters:
+ sw          - The `DMSwarm`
. totalWeight - The sum of all particle weights
. func        - The PDF for the particle spatial distribution
- param       - The PDF parameters

  Notes:
  The PDF for velocity is assumed to be a Gaussian

  The particle weights are returned in the `w_q` field of `sw`.
*/
static PetscErrorCode InitializeWeights(DM sw, PetscReal totalWeight, PetscProbFn *func, const PetscReal param[])
{
  DM               xdm, vdm;
  DMSwarmCellDM    celldm;
  PetscScalar     *weight;
  PetscQuadrature  xquad;
  const PetscReal *xq, *xwq;
  const PetscInt   order = 5;
  PetscReal        xi0[3];
  PetscReal        xwtot = 0., pwtot = 0.;
  PetscInt         xNq;
  PetscInt         dim, Ns, xcStart, xcEnd, vcStart, vcEnd, debug = ((DM_Swarm *)sw->data)->printWeights;
  MPI_Comm         comm;
  PetscMPIInt      rank;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)sw, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetCellDM(sw, &xdm));
  PetscCall(DMSwarmGetNumSpecies(sw, &Ns));
  PetscCall(DMPlexGetHeightStratum(xdm, 0, &xcStart, &xcEnd));
  PetscCall(DMSwarmGetCellDMByName(sw, "velocity", &celldm));
  PetscCall(DMSwarmCellDMGetDM(celldm, &vdm));
  PetscCall(DMPlexGetHeightStratum(vdm, 0, &vcStart, &vcEnd));

  // Setup Quadrature for spatial and velocity weight calculations
  PetscCall(PetscDTGaussTensorQuadrature(dim, 1, order, -1.0, 1.0, &xquad));
  PetscCall(PetscQuadratureGetData(xquad, NULL, NULL, &xNq, &xq, &xwq));
  for (PetscInt d = 0; d < dim; ++d) xi0[d] = -1.0;

  // Integrate the density function to get the weights of particles in each cell
  PetscCall(DMGetCoordinatesLocalSetUp(vdm));
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weight));
  for (PetscInt c = xcStart; c < xcEnd; ++c) {
    PetscReal          xv0[3], xJ[9], xinvJ[9], xdetJ, xqr[3], xden, xw = 0.;
    PetscInt          *pidx, Npc;
    PetscInt           xNc;
    const PetscScalar *xarray;
    PetscScalar       *xcoords = NULL;
    PetscBool          xisDG;

    PetscCall(DMPlexGetCellCoordinates(xdm, c, &xisDG, &xNc, &xarray, &xcoords));
    PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Npc, &pidx));
    PetscCheck(Npc == (vcEnd - vcStart) * Ns, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of particles %" PetscInt_FMT " in cell (rank %d) != %" PetscInt_FMT " number of velocity vertices", Npc, rank, (vcEnd - vcStart) * Ns);
    PetscCall(DMPlexComputeCellGeometryFEM(xdm, c, NULL, xv0, xJ, xinvJ, &xdetJ));
    for (PetscInt q = 0; q < xNq; ++q) {
      // Transform quadrature points from ref space to real space
      CoordinatesRefToReal(dim, dim, xi0, xv0, xJ, &xq[q * dim], xqr);
      // Get probability density at quad point
      //   No need to scale xqr since PDF will be periodic
      PetscCall((*func)(xqr, param, &xden));
      xw += xden * (xwq[q] * xdetJ);
    }
    xwtot += xw;
    if (debug) {
      IS              globalOrdering;
      const PetscInt *ordering;

      PetscCall(DMPlexGetCellNumbering(xdm, &globalOrdering));
      PetscCall(ISGetIndices(globalOrdering, &ordering));
      PetscCall(PetscSynchronizedPrintf(comm, "c:%" PetscInt_FMT " [x_a,x_b] = %1.15f,%1.15f -> cell weight = %1.15f\n", ordering[c], (double)PetscRealPart(xcoords[0]), (double)PetscRealPart(xcoords[0 + dim]), (double)xw));
      PetscCall(ISRestoreIndices(globalOrdering, &ordering));
    }
    // Set weights to be Gaussian in velocity cells
    for (PetscInt vc = vcStart; vc < vcEnd; ++vc) {
      const PetscInt     p  = pidx[vc * Ns + 0];
      PetscReal          vw = 0.;
      PetscInt           vNc;
      const PetscScalar *varray;
      PetscScalar       *vcoords = NULL;
      PetscBool          visDG;

      PetscCall(DMPlexGetCellCoordinates(vdm, vc, &visDG, &vNc, &varray, &vcoords));
      PetscCheck(vNc > 0 && vNc % dim == 0, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Velocity cell %" PetscInt_FMT " has invalid coordinates (vNc=%" PetscInt_FMT ", dim=%" PetscInt_FMT ")", vc, vNc, dim);
      {
        PetscReal vmin[3] = {PETSC_MAX_REAL, PETSC_MAX_REAL, PETSC_MAX_REAL};
        PetscReal vmax[3] = {-PETSC_MAX_REAL, -PETSC_MAX_REAL, -PETSC_MAX_REAL};
        PetscInt  numVert = vNc / dim;
        for (PetscInt i = 0; i < numVert; ++i) {
          for (PetscInt d = 0; d < dim; ++d) {
            PetscReal coord = PetscRealPart(vcoords[i * dim + d]);
            vmin[d]         = PetscMin(vmin[d], coord);
            vmax[d]         = PetscMax(vmax[d], coord);
          }
        }
        vw = 1.0;
        for (PetscInt d = 0; d < dim; ++d) vw *= 0.5 * (PetscErfReal(vmax[d] / PetscSqrtReal(2.)) - PetscErfReal(vmin[d] / PetscSqrtReal(2.)));
        PetscCheck(PetscIsNormalReal(vw), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Particle %" PetscInt_FMT " velocity weight is not normal: vw=%g, vmin=(%g,%g,%g), vmax=(%g,%g,%g)", p, vw, vmin[0], vmin[1], vmin[2], vmax[0], vmax[1], vmax[2]);
      }

      weight[p] = totalWeight * vw * xw;
      pwtot += weight[p];
      PetscCheck(weight[p] <= 10., PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Particle %" PetscInt_FMT " weight exceeded 10: weight=%g, xw=%g, vw=%g, totalWeight=%g", p, weight[p], xw, vw, totalWeight);
      PetscCall(DMPlexRestoreCellCoordinates(vdm, vc, &visDG, &vNc, &varray, &vcoords));
      if (debug > 1) PetscCall(PetscPrintf(comm, "particle %" PetscInt_FMT ": %g, vw: %g xw: %g\n", p, weight[p], vw, xw));
    }
    PetscCall(DMPlexRestoreCellCoordinates(xdm, c, &xisDG, &xNc, &xarray, &xcoords));
    PetscCall(DMSwarmSortRestorePointsPerCell(sw, c, &Npc, &pidx));
  }
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weight));
  PetscCall(DMSwarmSortRestoreAccess(sw));
  PetscCall(PetscQuadratureDestroy(&xquad));

  if (debug) {
    PetscReal wtot[2] = {pwtot, xwtot}, gwtot[2];

    PetscCall(PetscSynchronizedFlush(comm, NULL));
    PetscCallMPI(MPIU_Allreduce(wtot, gwtot, 2, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD));
    PetscCall(PetscPrintf(comm, "particle weight sum = %1.10f cell weight sum = %1.10f\n", (double)gwtot[0], (double)gwtot[1]));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitializeParticles_PerturbedWeights(DM sw, AppCtx *user)
{
  PetscReal scale[2] = {user->cosine_coefficients[0], user->cosine_coefficients[1]};
  PetscInt  dim;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(InitializeParticles_Centroid(sw));
  PetscCall(InitializeWeights(sw, user->totalWeight, dim == 1 ? PetscPDFCosine1D : (dim == 2 ? PetscPDFCosine2D : PetscPDFCosine3D), scale));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitializeConstants(DM sw, AppCtx *user)
{
  DM         dm;
  PetscInt  *species;
  PetscReal *weight, totalCharge = 0., totalWeight = 0., gmin[3], gmax[3], global_charge, global_weight;
  PetscInt   Np, dim;

  PetscFunctionBegin;
  PetscCall(DMSwarmGetCellDM(sw, &dm));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMGetBoundingBox(dm, gmin, gmax));
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weight));
  PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void **)&species));
  for (PetscInt p = 0; p < Np; ++p) {
    totalWeight += weight[p];
    totalCharge += user->charges[species[p]] * weight[p];
  }
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weight));
  PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **)&species));
  {
    Parameter *param;
    PetscReal  Area;

    PetscCall(PetscBagGetData(user->bag, (void **)&param));
    switch (dim) {
    case 1:
      Area = (gmax[0] - gmin[0]);
      break;
    case 2:
      Area = (gmax[0] - gmin[0]) * (gmax[1] - gmin[1]);
      break;
    case 3:
      Area = (gmax[0] - gmin[0]) * (gmax[1] - gmin[1]) * (gmax[2] - gmin[2]);
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Dimension %" PetscInt_FMT " not supported", dim);
    }
    PetscCallMPI(MPIU_Allreduce(&totalWeight, &global_weight, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD));
    PetscCallMPI(MPIU_Allreduce(&totalCharge, &global_charge, 1, MPIU_REAL, MPIU_SUM, PETSC_COMM_WORLD));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "dim = %" PetscInt_FMT "\ttotalWeight = %f, user->charges[species[0]] = %f\ttotalCharge = %f, Total Area = %f\n", dim, (double)global_weight, (double)user->charges[0], (double)global_charge, (double)Area));
    param->sigma = PetscAbsReal(global_charge / (Area));

    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "sigma: %g\n", (double)param->sigma));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "(x0,v0,t0,m0,q0,phi0): (%e, %e, %e, %e, %e, %e) - (P, V) = (%e, %e)\n", (double)param->x0, (double)param->v0, (double)param->t0, (double)param->m0, (double)param->q0, (double)param->phi0, (double)param->poissonNumber,
                          (double)param->vlasovNumber));
  }
  /* Setup Constants */
  {
    PetscDS    ds;
    Parameter *param;
    PetscCall(PetscBagGetData(user->bag, (void **)&param));
    PetscScalar constants[NUM_CONSTANTS];
    constants[SIGMA]   = param->sigma;
    constants[V0]      = param->v0;
    constants[T0]      = param->t0;
    constants[X0]      = param->x0;
    constants[M0]      = param->m0;
    constants[Q0]      = param->q0;
    constants[PHI0]    = param->phi0;
    constants[POISSON] = param->poissonNumber;
    constants[VLASOV]  = param->vlasovNumber;
    PetscCall(DMGetDS(dm, &ds));
    PetscCall(PetscDSSetConstants(ds, NUM_CONSTANTS, constants));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupParameters(MPI_Comm comm, AppCtx *ctx)
{
  PetscBag   bag;
  Parameter *p;

  PetscFunctionBeginUser;
  /* setup PETSc parameter bag */
  PetscCall(PetscBagGetData(ctx->bag, (void **)&p));
  PetscCall(PetscBagSetName(ctx->bag, "par", "Vlasov-Poisson Parameters"));
  bag = ctx->bag;
  PetscCall(PetscBagRegisterScalar(bag, &p->v0, 1.0, "v0", "Velocity scale, m/s"));
  PetscCall(PetscBagRegisterScalar(bag, &p->t0, 1.0, "t0", "Time scale, s"));
  PetscCall(PetscBagRegisterScalar(bag, &p->x0, 1.0, "x0", "Space scale, m"));
  PetscCall(PetscBagRegisterScalar(bag, &p->phi0, 1.0, "phi0", "Potential scale, kg*m^2/A*s^3"));
  PetscCall(PetscBagRegisterScalar(bag, &p->q0, 1.0, "q0", "Charge Scale, A*s"));
  PetscCall(PetscBagRegisterScalar(bag, &p->m0, 1.0, "m0", "Mass Scale, kg"));
  PetscCall(PetscBagRegisterScalar(bag, &p->epsi0, 1.0, "epsi0", "Permittivity of Free Space, kg"));
  PetscCall(PetscBagRegisterScalar(bag, &p->kb, 1.0, "kb", "Boltzmann Constant, m^2 kg/s^2 K^1"));

  PetscCall(PetscBagRegisterScalar(bag, &p->sigma, 1.0, "sigma", "Charge per unit area, C/m^3"));
  PetscCall(PetscBagRegisterScalar(bag, &p->poissonNumber, 1.0, "poissonNumber", "Non-Dimensional Poisson Number"));
  PetscCall(PetscBagRegisterScalar(bag, &p->vlasovNumber, 1.0, "vlasovNumber", "Non-Dimensional Vlasov Number"));
  PetscCall(PetscBagSetFromOptions(bag));
  {
    PetscViewer       viewer;
    PetscViewerFormat format;
    PetscBool         flg;

    PetscCall(PetscOptionsCreateViewer(comm, NULL, NULL, "-param_view", &viewer, &format, &flg));
    if (flg) {
      PetscCall(PetscViewerPushFormat(viewer, format));
      PetscCall(PetscBagView(bag, viewer));
      PetscCall(PetscViewerFlush(viewer));
      PetscCall(PetscViewerPopFormat(viewer));
      PetscCall(PetscViewerDestroy(&viewer));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  const char *prefix = "x";

  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMPlexSetOptionsPrefix(*dm, prefix));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(PetscObjectSetName((PetscObject)*dm, "space"));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));

  // Cache the mesh geometry
  DMField         coordField;
  IS              cellIS;
  PetscQuadrature quad;
  PetscReal      *wt, *pt;
  PetscInt        cdim, cStart, cEnd;

  PetscCall(DMGetCoordinateField(*dm, &coordField));
  PetscCheck(coordField, comm, PETSC_ERR_USER, "DM must have a coordinate field");
  PetscCall(DMGetCoordinateDim(*dm, &cdim));
  PetscCall(DMPlexGetHeightStratum(*dm, 0, &cStart, &cEnd));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, cEnd - cStart, cStart, 1, &cellIS));
  PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, &quad));
  PetscCall(PetscMalloc1(1, &wt));
  PetscCall(PetscMalloc1(cdim, &pt));
  wt[0] = 1.;
  for (PetscInt d = 0; d < cdim; ++d) pt[d] = -1.;
  PetscCall(PetscQuadratureSetData(quad, cdim, 1, 1, pt, wt));
  PetscCall(DMFieldCreateFEGeom(coordField, cellIS, quad, PETSC_FEGEOM_BASIC, &user->fegeom));
  PetscCall(PetscQuadratureDestroy(&quad));
  PetscCall(ISDestroy(&cellIS));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void ion_f0(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = -constants[SIGMA];
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

static PetscErrorCode CreateFEM(DM dm, AppCtx *user)
{
  PetscFE      fephi;
  PetscDS      ds;
  PetscBool    simplex;
  PetscInt     dim;
  MatNullSpace nullsp;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexIsSimplex(dm, &simplex));

  PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, NULL, PETSC_DETERMINE, &fephi));
  PetscCall(PetscObjectSetName((PetscObject)fephi, "potential"));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fephi));
  PetscCall(DMCreateDS(dm));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetResidual(ds, 0, ion_f0, laplacian_f1));
  PetscCall(PetscDSSetJacobian(ds, 0, 0, NULL, NULL, NULL, laplacian_g3));
  PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)dm), PETSC_TRUE, 0, NULL, &nullsp));
  PetscCall(PetscObjectCompose((PetscObject)fephi, "nullspace", (PetscObject)nullsp));
  PetscCall(MatNullSpaceDestroy(&nullsp));
  PetscCall(PetscFEDestroy(&fephi));
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
  PetscCall(DMPlexSetSNESLocalFEM(dm, PETSC_FALSE, user));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(DMCreateMatrix(dm, &J));
  PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)dm), PETSC_TRUE, 0, NULL, &nullSpace));
  PetscCall(MatSetNullSpace(J, nullSpace));
  PetscCall(MatNullSpaceDestroy(&nullSpace));
  PetscCall(SNESSetJacobian(snes, J, J, NULL, NULL));
  PetscCall(MatDestroy(&J));

  user->dmPot = dm;
  PetscCall(PetscObjectReference((PetscObject)user->dmPot));

  PetscCall(DMCreateMassMatrix(user->dmPot, user->dmPot, &user->M));
  PetscCall(DMPlexCreateClosureIndex(dm, NULL));
  user->snes = snes;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateSwarm(DM dm, AppCtx *user, DM *sw)
{
  DMSwarmCellDM celldm;
  PetscInt      dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMCreate(PetscObjectComm((PetscObject)dm), sw));
  PetscCall(DMSetType(*sw, DMSWARM));
  PetscCall(DMSetDimension(*sw, dim));
  PetscCall(DMSwarmSetType(*sw, DMSWARM_PIC));
  PetscCall(DMSetApplicationContext(*sw, user));

  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "w_q", 1, PETSC_SCALAR));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "velocity", dim, PETSC_REAL));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "species", 1, PETSC_INT));
  PetscCall(DMSwarmRegisterPetscDatatypeField(*sw, "E_field", dim, PETSC_REAL));

  const char *fieldnames[2] = {DMSwarmPICField_coor, "velocity"};

  PetscCall(DMSwarmCellDMCreate(dm, 2, fieldnames, 1, fieldnames, &celldm));
  PetscCall(DMSwarmAddCellDM(*sw, celldm));
  PetscCall(DMSwarmCellDMDestroy(&celldm));

  const char *vfieldnames[2] = {"w_q"};
  DM          vdm;

  PetscCall(CreateVelocityDM(*sw, &vdm));
  PetscCall(DMSwarmCellDMCreate(vdm, 1, vfieldnames, 1, &fieldnames[1], &celldm));
  PetscCall(DMSwarmAddCellDM(*sw, celldm));
  PetscCall(DMSwarmCellDMDestroy(&celldm));
  PetscCall(DMDestroy(&vdm));

  DM mdm;

  PetscCall(DMClone(dm, &mdm));
  PetscCall(PetscObjectSetName((PetscObject)mdm, "moments"));
  PetscCall(DMCopyDisc(dm, mdm));
  PetscCall(DMSwarmCellDMCreate(mdm, 1, vfieldnames, 1, fieldnames, &celldm));
  PetscCall(DMDestroy(&mdm));
  PetscCall(DMSwarmAddCellDM(*sw, celldm));
  PetscCall(DMSwarmCellDMDestroy(&celldm));

  PetscCall(DMSetFromOptions(*sw));
  PetscCall(DMSetUp(*sw));

  PetscCall(DMSwarmSetCellDMActive(*sw, "space"));
  user->swarm = *sw;
  // TODO: This is redundant init since it is done in InitializeSolveAndSwarm, however DMSetUp() requires the local size be set
  PetscCall(InitializeParticles_PerturbedWeights(*sw, user));
  PetscCall(PetscObjectSetName((PetscObject)*sw, "Particles"));
  PetscCall(DMViewFromOptions(*sw, NULL, "-sw_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFieldAtParticles_Primal(SNES snes, DM sw, Mat M_p, PetscReal E[])
{
  DM         dm;
  AppCtx    *user;
  PetscDS    ds;
  PetscFE    fe;
  KSP        ksp;
  Vec        rhoRhs;      // Weak charge density, \int phi_i rho
  Vec        rho;         // Charge density, M^{-1} rhoRhs
  Vec        phi, locPhi; // Potential
  Vec        f;           // Particle weights
  PetscReal *coords;
  PetscInt   dim, cStart, cEnd, Np;

  PetscFunctionBegin;
  PetscCall(DMGetApplicationContext(sw, (void *)&user));
  PetscCall(PetscLogEventBegin(user->ESolveEvent, snes, sw, 0, 0));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));

  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMGetGlobalVector(dm, &rhoRhs));
  PetscCall(PetscObjectSetName((PetscObject)rhoRhs, "Weak charge density"));
  PetscCall(DMGetNamedGlobalVector(user->dmPot, "rho", &rho));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &f));
  PetscCall(PetscObjectSetName((PetscObject)f, "particle weight"));

  PetscCall(MatViewFromOptions(M_p, NULL, "-mp_view"));
  PetscCall(MatViewFromOptions(user->M, NULL, "-m_view"));
  PetscCall(VecViewFromOptions(f, NULL, "-weights_view"));

  PetscCall(MatMultTranspose(M_p, f, rhoRhs));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &f));

  // Low-pass filter rhoRhs
  PetscInt window = 0;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-rho_average", &window, NULL));
  if (window) {
    PetscScalar *a;
    PetscInt     n;
    PetscReal    width = 2. * window + 1.;

    // This will only work for P_1
    //   This works because integration against a test function is linear
    //   Do a real integral against weight function for higher order
    PetscCall(VecGetLocalSize(rhoRhs, &n));
    PetscCall(VecGetArrayWrite(rhoRhs, &a));
    for (PetscInt i = 0; i < n; ++i) {
      PetscScalar avg = a[i];
      for (PetscInt j = 1; j <= window; ++j) avg += a[(i - j + n) % n] + a[(i + j) % n];
      a[i] = avg / width;
      //a[i] = (a[(i - 1 + n) % n] + a[i] + a[(i + 1) % n]) / 3.;
    }
    PetscCall(VecRestoreArrayWrite(rhoRhs, &a));
  }

  PetscCall(KSPCreate(PetscObjectComm((PetscObject)dm), &ksp));
  PetscCall(KSPSetOptionsPrefix(ksp, "em_proj_"));
  PetscCall(KSPSetOperators(ksp, user->M, user->M));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, rhoRhs, rho));

  PetscCall(VecScale(rhoRhs, -1.0));

  PetscCall(VecViewFromOptions(rhoRhs, NULL, "-rho_view"));
  PetscCall(DMRestoreNamedGlobalVector(user->dmPot, "rho", &rho));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(DMGetNamedGlobalVector(user->dmPot, "phi", &phi));
  PetscCall(VecSet(phi, 0.0));
  PetscCall(SNESSolve(snes, rhoRhs, phi));
  PetscCall(DMRestoreGlobalVector(dm, &rhoRhs));
  PetscCall(VecViewFromOptions(phi, NULL, "-phi_view"));

  PetscCall(DMGetLocalVector(dm, &locPhi));
  PetscCall(DMGlobalToLocalBegin(dm, phi, INSERT_VALUES, locPhi));
  PetscCall(DMGlobalToLocalEnd(dm, phi, INSERT_VALUES, locPhi));
  PetscCall(DMRestoreNamedGlobalVector(user->dmPot, "phi", &phi));
  PetscCall(PetscLogEventEnd(user->ESolveEvent, snes, sw, 0, 0));

  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSGetDiscretization(ds, 0, (PetscObject *)&fe));
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));

  PetscCall(PetscLogEventBegin(user->ETabEvent, snes, sw, 0, 0));
  PetscTabulation tab;
  PetscReal      *pcoord, *refcoord;
  PetscFEGeom    *chunkgeom = NULL;
  PetscInt        maxNcp    = 0;

  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscInt Ncp;

    PetscCall(DMSwarmSortGetNumberOfPointsPerCell(sw, c, &Ncp));
    maxNcp = PetscMax(maxNcp, Ncp);
  }
  PetscCall(DMGetWorkArray(dm, maxNcp * dim, MPIU_REAL, &refcoord));
  PetscCall(PetscArrayzero(refcoord, maxNcp * dim));
  PetscCall(DMGetWorkArray(dm, maxNcp * dim, MPIU_REAL, &pcoord));
  PetscCall(PetscFECreateTabulation(fe, 1, maxNcp, refcoord, 1, &tab));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscScalar *clPhi = NULL;
    PetscInt    *points;
    PetscInt     Ncp;

    PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Ncp, &points));
    for (PetscInt cp = 0; cp < Ncp; ++cp)
      for (PetscInt d = 0; d < dim; ++d) pcoord[cp * dim + d] = coords[points[cp] * dim + d];
    {
      PetscCall(PetscFEGeomGetChunk(user->fegeom, c - cStart, c - cStart + 1, &chunkgeom));
      for (PetscInt i = 0; i < Ncp; ++i) {
        const PetscReal x0[3] = {-1., -1., -1.};
        CoordinatesRealToRef(dim, dim, x0, chunkgeom->v, chunkgeom->invJ, &pcoord[dim * i], &refcoord[dim * i]);
      }
    }
    PetscCall(PetscFEComputeTabulation(fe, Ncp, refcoord, 1, tab));
    PetscCall(DMPlexVecGetClosure(dm, NULL, locPhi, c, NULL, &clPhi));
    for (PetscInt cp = 0; cp < Ncp; ++cp) {
      const PetscReal *basisDer = tab->T[1];
      const PetscInt   p        = points[cp];

      for (PetscInt d = 0; d < dim; ++d) E[p * dim + d] = 0.;
      PetscCall(PetscFEFreeInterpolateGradient_Static(fe, basisDer, clPhi, dim, chunkgeom->invJ, NULL, cp, &E[p * dim]));
      for (PetscInt d = 0; d < dim; ++d) E[p * dim + d] *= -1.0;
    }
    PetscCall(DMPlexVecRestoreClosure(dm, NULL, locPhi, c, NULL, &clPhi));
    PetscCall(DMSwarmSortRestorePointsPerCell(sw, c, &Ncp, &points));
  }
  PetscCall(DMRestoreWorkArray(dm, maxNcp * dim, MPIU_REAL, &pcoord));
  PetscCall(DMRestoreWorkArray(dm, maxNcp * dim, MPIU_REAL, &refcoord));
  PetscCall(PetscTabulationDestroy(&tab));
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmSortRestoreAccess(sw));
  PetscCall(DMRestoreLocalVector(dm, &locPhi));
  PetscCall(PetscFEGeomRestoreChunk(user->fegeom, 0, 1, &chunkgeom));
  PetscCall(PetscLogEventEnd(user->ETabEvent, snes, sw, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFieldAtParticles(SNES snes, DM sw)
{
  AppCtx    *user;
  Mat        M_p;
  PetscReal *E;
  PetscInt   dim, Np;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 1);
  PetscValidHeaderSpecific(sw, DM_CLASSID, 2);
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMGetApplicationContext(sw, &user));

  PetscCall(DMSwarmSetCellDMActive(sw, "moments"));
  // TODO: Could share sort context with space cellDM
  PetscCall(DMSwarmMigrate(sw, PETSC_FALSE));
  PetscCall(DMCreateMassMatrix(sw, user->dmPot, &M_p));
  PetscCall(DMSwarmSetCellDMActive(sw, "space"));

  PetscCall(DMSwarmGetField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscCall(PetscArrayzero(E, Np * dim));
  user->validE = PETSC_TRUE;

  PetscCall(ComputeFieldAtParticles_Primal(snes, sw, M_p, E));
  PetscCall(DMSwarmRestoreField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscCall(MatDestroy(&M_p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSFunction(TS ts, PetscReal t, Vec U, Vec G, void *ctx)
{
  DM                 sw;
  SNES               snes = ((AppCtx *)ctx)->snes;
  const PetscScalar *u;
  PetscScalar       *g;
  PetscReal         *E, m_p = 1., q_p = -1.;
  PetscInt           dim, d, Np, p;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(ComputeFieldAtParticles(snes, sw));

  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMSwarmGetField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArray(G, &g));
  Np /= 2 * dim;
  for (p = 0; p < Np; ++p) {
    for (d = 0; d < dim; ++d) {
      g[(p * 2 + 0) * dim + d] = u[(p * 2 + 1) * dim + d];
      g[(p * 2 + 1) * dim + d] = q_p * E[p * dim + d] / m_p;
    }
  }
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
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(MatGetOwnershipRange(J, &rStart, NULL));
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&vel));
  Np /= 2 * dim;
  for (p = 0; p < Np; ++p) {
    // TODO This is not right because dv/dx has the electric field in it
    PetscScalar vals[4] = {0., 1., -1, 0.};

    for (d = 0; d < dim; ++d) {
      const PetscInt rows[2] = {(p * 2 + 0) * dim + d + rStart, (p * 2 + 1) * dim + d + rStart};
      PetscCall(MatSetValues(J, 2, rows, 2, rows, vals, INSERT_VALUES));
    }
  }
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&vel));
  PetscCall(MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSFunctionX(TS ts, PetscReal t, Vec V, Vec Xres, void *ctx)
{
  AppCtx            *user = (AppCtx *)ctx;
  DM                 sw;
  const PetscScalar *v;
  PetscScalar       *xres;
  PetscInt           Np, p, d, dim;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(user->RhsXEvent, ts, 0, 0, 0));
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(VecGetLocalSize(Xres, &Np));
  PetscCall(VecGetArrayRead(V, &v));
  PetscCall(VecGetArray(Xres, &xres));
  Np /= dim;
  for (p = 0; p < Np; ++p) {
    for (d = 0; d < dim; ++d) xres[p * dim + d] = v[p * dim + d];
  }
  PetscCall(VecRestoreArrayRead(V, &v));
  PetscCall(VecRestoreArray(Xres, &xres));
  PetscCall(PetscLogEventEnd(user->RhsXEvent, ts, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode RHSFunctionV(TS ts, PetscReal t, Vec X, Vec Vres, void *ctx)
{
  DM                 sw;
  AppCtx            *user = (AppCtx *)ctx;
  SNES               snes = ((AppCtx *)ctx)->snes;
  const PetscScalar *x;
  PetscScalar       *vres;
  PetscReal         *E, m_p, q_p;
  PetscInt           Np, p, dim, d;
  Parameter         *param;

  PetscFunctionBeginUser;
  PetscCall(PetscLogEventBegin(user->RhsVEvent, ts, 0, 0, 0));
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(ComputeFieldAtParticles(snes, sw));

  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscCall(PetscBagGetData(user->bag, (void **)&param));
  m_p = user->masses[0] * param->m0;
  q_p = user->charges[0] * param->q0;
  PetscCall(VecGetLocalSize(Vres, &Np));
  PetscCall(VecGetArrayRead(X, &x));
  PetscCall(VecGetArray(Vres, &vres));
  Np /= dim;
  for (p = 0; p < Np; ++p) {
    for (d = 0; d < dim; ++d) vres[p * dim + d] = q_p * E[p * dim + d] / m_p;
  }
  PetscCall(VecRestoreArrayRead(X, &x));
  /*
    Synchronized, ordered output for parallel/sequential test cases.
    In the 1D (on the 2D mesh) case, every y component should be zero.
  */
  if (user->checkVRes) {
    PetscBool pr = user->checkVRes > 1 ? PETSC_TRUE : PETSC_FALSE;
    PetscInt  step;

    PetscCall(TSGetStepNumber(ts, &step));
    if (pr) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "step: %" PetscInt_FMT "\n", step));
    for (PetscInt p = 0; p < Np; ++p) {
      if (pr) PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "Residual: %.12g %.12g\n", (double)PetscRealPart(vres[p * dim + 0]), (double)PetscRealPart(vres[p * dim + 1])));
      PetscCheck(PetscAbsScalar(vres[p * dim + 1]) < PETSC_SMALL, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Y velocity should be 0., not %g", (double)PetscRealPart(vres[p * dim + 1]));
    }
    if (pr) PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, PETSC_STDOUT));
  }
  PetscCall(VecRestoreArray(Vres, &vres));
  PetscCall(DMSwarmRestoreField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscCall(PetscLogEventEnd(user->RhsVEvent, ts, 0, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Discrete Gradients Formulation: S, F, gradF (G) */
PetscErrorCode RHSJacobianS(TS ts, PetscReal t, Vec U, Mat S, void *ctx)
{
  PetscScalar vals[4] = {0., 1., -1., 0.};
  DM          sw;
  PetscInt    dim, d, Np, p, rStart;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(VecGetLocalSize(U, &Np));
  PetscCall(MatGetOwnershipRange(S, &rStart, NULL));
  Np /= 2 * dim;
  for (p = 0; p < Np; ++p) {
    for (d = 0; d < dim; ++d) {
      const PetscInt rows[2] = {(p * 2 + 0) * dim + d + rStart, (p * 2 + 1) * dim + d + rStart};
      PetscCall(MatSetValues(S, 2, rows, 2, rows, vals, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode RHSObjectiveF(TS ts, PetscReal t, Vec U, PetscScalar *F, void *ctx)
{
  AppCtx            *user = (AppCtx *)ctx;
  DM                 sw;
  Vec                phi;
  const PetscScalar *u;
  PetscInt           dim, Np, cStart, cEnd;
  PetscReal         *vel, *coords, m_p = 1.;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMPlexGetHeightStratum(user->dmPot, 0, &cStart, &cEnd));

  PetscCall(DMGetNamedGlobalVector(user->dmPot, "phi", &phi));
  PetscCall(VecViewFromOptions(phi, NULL, "-phi_view_dg"));
  PetscCall(computeFieldEnergy(user->dmPot, phi, F));
  PetscCall(DMRestoreNamedGlobalVector(user->dmPot, "phi", &phi));

  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&vel));
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetLocalSize(U, &Np));
  Np /= 2 * dim;
  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscInt *points;
    PetscInt  Ncp;

    PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Ncp, &points));
    for (PetscInt cp = 0; cp < Ncp; ++cp) {
      const PetscInt  p  = points[cp];
      const PetscReal v2 = DMPlex_DotRealD_Internal(dim, &u[(p * 2 + 1) * dim], &u[(p * 2 + 1) * dim]);

      *F += 0.5 * m_p * v2;
    }
    PetscCall(DMSwarmSortRestorePointsPerCell(sw, c, &Ncp, &points));
  }
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(DMSwarmSortRestoreAccess(sw));
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&vel));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* dF/dx = q E   dF/dv = v */
PetscErrorCode RHSFunctionG(TS ts, PetscReal t, Vec U, Vec G, void *ctx)
{
  DM                 sw;
  SNES               snes = ((AppCtx *)ctx)->snes;
  const PetscReal   *coords, *vel, *E;
  const PetscScalar *u;
  PetscScalar       *g;
  PetscReal          m_p = 1., q_p = -1.;
  PetscInt           dim, d, Np, p;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(VecGetArrayRead(U, &u));
  PetscCall(VecGetArray(G, &g));

  PetscLogEvent COMPUTEFIELD;
  PetscCall(PetscLogEventRegister("COMPFIELDATPART", TS_CLASSID, &COMPUTEFIELD));
  PetscCall(PetscLogEventBegin(COMPUTEFIELD, 0, 0, 0, 0));
  PetscCall(ComputeFieldAtParticles(snes, sw));
  PetscCall(PetscLogEventEnd(COMPUTEFIELD, 0, 0, 0, 0));
  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&vel));
  PetscCall(DMSwarmGetField(sw, "E_field", NULL, NULL, (void **)&E));
  for (p = 0; p < Np; ++p) {
    for (d = 0; d < dim; ++d) {
      g[(p * 2 + 0) * dim + d] = -(q_p / m_p) * E[p * dim + d];
      g[(p * 2 + 1) * dim + d] = u[(p * 2 + 1) * dim + d];
    }
  }
  PetscCall(DMSwarmRestoreField(sw, "E_field", NULL, NULL, (void **)&E));
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&vel));
  PetscCall(VecRestoreArrayRead(U, &u));
  PetscCall(VecRestoreArray(G, &g));
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
  // Define symplectic formulation U_t = S . G, where G = grad F
  {
    PetscCall(TSDiscGradSetFormulation(ts, RHSJacobianS, RHSObjectiveF, RHSFunctionG, user));
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

PetscErrorCode line(PetscInt dim, PetscReal time, const PetscReal dummy[], PetscInt p, PetscScalar x[], void *ctx)
{
  DM        sw, cdm;
  PetscInt  Np;
  PetscReal low[2], high[2];
  AppCtx   *user = (AppCtx *)ctx;

  sw = user->swarm;
  PetscCall(DMSwarmGetCellDM(sw, &cdm));
  // Get the bounding box so we can equally space the particles
  PetscCall(DMGetLocalBoundingBox(cdm, low, high));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  // shift it by h/2 so nothing is initialized directly on a boundary
  x[0] = ((high[0] - low[0]) / Np) * (p + 0.5);
  x[1] = 0.;
  return PETSC_SUCCESS;
}

/*
  InitializeSolveAndSwarm - Set the solution values to the swarm coordinates and velocities, and also possibly set the initial values.

  Input Parameters:
+ ts         - The TS
- useInitial - Flag to also set the initial conditions to the current coordinates and velocities and setup the problem

  Output Parameters:
. u - The initialized solution vector

  Level: advanced

.seealso: InitializeSolve()
*/
static PetscErrorCode InitializeSolveAndSwarm(TS ts, PetscBool useInitial)
{
  DM       sw;
  Vec      u, gc, gv;
  IS       isx, isv;
  PetscInt dim;
  AppCtx  *user;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetApplicationContext(sw, &user));
  PetscCall(DMGetDimension(sw, &dim));
  if (useInitial) {
    PetscCall(InitializeParticles_PerturbedWeights(sw, user));
    PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));
    PetscCall(DMSwarmTSRedistribute(ts));
  }
  PetscCall(DMSetUp(sw));
  PetscCall(TSGetSolution(ts, &u));
  PetscCall(TSRHSSplitGetIS(ts, "position", &isx));
  PetscCall(TSRHSSplitGetIS(ts, "momentum", &isv));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "velocity", &gv));
  PetscCall(VecISCopy(u, isx, SCATTER_FORWARD, gc));
  PetscCall(VecISCopy(u, isv, SCATTER_FORWARD, gv));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "velocity", &gv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode InitializeSolve(TS ts, Vec u)
{
  PetscFunctionBegin;
  PetscCall(TSSetSolution(ts, u));
  PetscCall(InitializeSolveAndSwarm(ts, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MigrateParticles(TS ts)
{
  DM               sw, cdm;
  const PetscReal *L;
  AppCtx          *ctx;

  PetscFunctionBeginUser;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetApplicationContext(sw, &ctx));
  PetscCall(DMViewFromOptions(sw, NULL, "-migrate_view_pre"));
  {
    Vec        u, gc, gv, position, momentum;
    IS         isx, isv;
    PetscReal *pos, *mom;

    PetscCall(TSGetSolution(ts, &u));
    PetscCall(TSRHSSplitGetIS(ts, "position", &isx));
    PetscCall(TSRHSSplitGetIS(ts, "momentum", &isv));
    PetscCall(VecGetSubVector(u, isx, &position));
    PetscCall(VecGetSubVector(u, isv, &momentum));
    PetscCall(VecGetArray(position, &pos));
    PetscCall(VecGetArray(momentum, &mom));
    PetscCall(DMSwarmCreateGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
    PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "velocity", &gv));
    PetscCall(VecISCopy(u, isx, SCATTER_REVERSE, gc));
    PetscCall(VecISCopy(u, isv, SCATTER_REVERSE, gv));

    PetscCall(DMSwarmGetCellDM(sw, &cdm));
    PetscCall(DMGetPeriodicity(cdm, NULL, NULL, &L));
    if ((L[0] || L[1]) >= 0.) {
      PetscReal *x, *v, upper[3], lower[3];
      PetscInt   Np, dim;

      PetscCall(DMSwarmGetLocalSize(sw, &Np));
      PetscCall(DMGetDimension(cdm, &dim));
      PetscCall(DMGetBoundingBox(cdm, lower, upper));
      PetscCall(VecGetArray(gc, &x));
      PetscCall(VecGetArray(gv, &v));
      for (PetscInt p = 0; p < Np; ++p) {
        for (PetscInt d = 0; d < dim; ++d) {
          if (pos[p * dim + d] < lower[d]) {
            x[p * dim + d] = pos[p * dim + d] + (upper[d] - lower[d]);
          } else if (pos[p * dim + d] > upper[d]) {
            x[p * dim + d] = pos[p * dim + d] - (upper[d] - lower[d]);
          } else {
            x[p * dim + d] = pos[p * dim + d];
          }
          PetscCheck(x[p * dim + d] >= lower[d] && x[p * dim + d] <= upper[d], PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "p: %" PetscInt_FMT "x[%" PetscInt_FMT "] %g", p, d, (double)x[p * dim + d]);
          v[p * dim + d] = mom[p * dim + d];
        }
      }
      PetscCall(VecRestoreArray(gc, &x));
      PetscCall(VecRestoreArray(gv, &v));
    }
    PetscCall(VecRestoreArray(position, &pos));
    PetscCall(VecRestoreArray(momentum, &mom));
    PetscCall(VecRestoreSubVector(u, isx, &position));
    PetscCall(VecRestoreSubVector(u, isv, &momentum));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "velocity", &gv));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, DMSwarmPICField_coor, &gc));
  }
  PetscCall(DMSwarmMigrate(sw, PETSC_TRUE));
  // This MUST come last, since it recreates the subswarms and they must DMClone() the new swarm
  PetscCall(DMSwarmTSRedistribute(ts));
  PetscCall(InitializeSolveAndSwarm(ts, PETSC_FALSE));
  {
    const char *fieldnames[2] = {DMSwarmPICField_coor, "velocity"};
    PetscCall(DMSwarmVectorDefineFields(sw, 2, fieldnames));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        dm, sw;
  TS        ts;
  Vec       u;
  PetscReal dt;
  PetscInt  maxn;
  AppCtx    user;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(PetscBagCreate(PETSC_COMM_SELF, sizeof(Parameter), &user.bag));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(CreatePoisson(dm, &user));
  PetscCall(CreateSwarm(dm, &user, &sw));
  PetscCall(SetupParameters(PETSC_COMM_WORLD, &user));
  PetscCall(InitializeConstants(sw, &user));
  PetscCall(DMSetApplicationContext(sw, &user));

  PetscCall(TSCreate(PETSC_COMM_WORLD, &ts));
  PetscCall(TSSetProblemType(ts, TS_NONLINEAR));
  PetscCall(TSSetDM(ts, sw));
  PetscCall(TSSetMaxTime(ts, 0.1));
  PetscCall(TSSetTimeStep(ts, 0.00001));
  PetscCall(TSSetMaxSteps(ts, 100));
  PetscCall(TSSetExactFinalTime(ts, TS_EXACTFINALTIME_MATCHSTEP));

  if (user.efield_monitor) PetscCall(TSMonitorSet(ts, MonitorEField, &user, NULL));
  if (user.moment_monitor) PetscCall(TSMonitorSet(ts, MonitorMoments, &user, NULL));
  if (user.particle_monitor) PetscCall(TSMonitorSet(ts, MonitorParticles, &user, NULL));

  PetscCall(TSSetFromOptions(ts));
  PetscCall(TSGetTimeStep(ts, &dt));
  PetscCall(TSGetMaxSteps(ts, &maxn));
  user.steps    = maxn;
  user.stepSize = dt;
  PetscCall(SetupContext(dm, sw, &user));
  PetscCall(TSSetComputeInitialCondition(ts, InitializeSolve));
  PetscCall(TSSetPostStep(ts, MigrateParticles));
  PetscCall(CreateSolution(ts));
  PetscCall(TSGetSolution(ts, &u));
  PetscCall(TSComputeInitialCondition(ts, u));
  PetscCall(CheckNonNegativeWeights(sw, &user));
  PetscCall(TSSolve(ts, NULL));

  PetscCall(SNESDestroy(&user.snes));
  PetscCall(DMDestroy(&user.dmPot));
  PetscCall(MatDestroy(&user.M));
  PetscCall(PetscFEGeomDestroy(&user.fegeom));
  PetscCall(TSDestroy(&ts));
  PetscCall(DMDestroy(&sw));
  PetscCall(DMDestroy(&dm));
  PetscCall(DestroyContext(&user));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
    requires: !complex double

   testset:
     nsize: 2
     args: -cosine_coefficients 0.01 -charges -1. -total_weight 1. -xdm_plex_hash_location -vpetscspace_degree 2 -petscspace_degree 1 -em_snes_atol 1.e-12 \
     -em_snes_error_if_not_converged -em_ksp_error_if_not_converged -em_ksp_type cg -em_pc_type gamg -em_mg_coarse_ksp_type preonly -em_mg_coarse_pc_type svd -em_proj_ksp_type cg \
     -em_proj_pc_type gamg -em_proj_mg_coarse_ksp_type preonly -em_proj_mg_coarse_pc_type svd -ts_time_step 0.03 -xdm_plex_simplex 0 \
     -ts_max_time 100 -output_step 1 -ts_type basicsymplectic -ts_basicsymplectic_type 1 -check_weights -ts_max_steps 60

     test:
       suffix: landau_damping_1d
       args: -xdm_plex_dim 1 -xdm_plex_box_faces 60 -xdm_plex_box_lower 0. -xdm_plex_box_upper 12.5664 -xdm_plex_box_bd periodic -vdm_plex_dim 1 -vdm_plex_box_faces 60 \
       -vdm_plex_box_lower -6 -vdm_plex_box_upper 6 -vdm_plex_hash_location -vdm_plex_simplex 0 -vdm_plex_box_bd none

     test:
       suffix: landau_damping_2d
       args: -xdm_plex_dim 2 -xdm_plex_box_bd periodic,periodic -vdm_plex_dim 2 -xdm_plex_box_lower 0.,-.5 -vdm_plex_box_lower -6,-6 -vdm_plex_box_upper 6,6 -xdm_plex_box_faces 6,3 \
       -xdm_plex_box_upper 12.5664,.5 -vdm_plex_box_faces 15,15 -vdm_plex_box_bd none,none -vdm_plex_hash_location -vdm_plex_simplex 0

     test:
       suffix: landau_damping_3d
       args: -xdm_plex_dim 3 -xdm_plex_box_faces 6,3,3 -xdm_plex_box_lower 0.,-1,-1 -xdm_plex_box_upper 12.5664,1,1 -xdm_plex_box_bd periodic,periodic,periodic -vdm_plex_dim 3 -vdm_plex_box_faces 4,4,4 \
       -vdm_plex_box_lower -6,-6,-6 -vdm_plex_box_upper 6,6,6 -vdm_plex_hash_location -vdm_plex_simplex 0 -vdm_plex_box_bd none,none,none

     test:
       requires: !defined(PETSC_USE_DMLANDAU_2D)
       suffix: sphere_3d
       nsize: 1
       args: -use_landau_velocity_space -xdm_plex_dim 3 -vdm_landau_thermal_temps 1 -vdm_landau_device_type cpu -xdm_plex_box_faces 6,3,3 -xdm_plex_box_lower 0.,-1,-1 \
       -xdm_plex_box_upper 12.5664,1,1 -xdm_plex_box_bd periodic,periodic,periodic -vdm_landau_verbose 2 -vdm_landau_sphere -vdm_landau_map_sphere \
       -vdm_landau_domain_radius 6,6,6 -vdm_landau_sphere_inner_radius_90degree_scale .35 -vdm_refine 1

TEST*/
