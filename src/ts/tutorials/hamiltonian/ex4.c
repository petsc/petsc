static char help[] = "Two-level system for Landau Damping using Vlasov-Poisson equations\n";

/*
  Moment Equations:

    We will discretize the moment equations using finite elements, and we will project the moments into the finite element space We will use the PFAK method, which guarantees that our FE approximation is weakly equivalent to the true moment. The first moment, number density, is given by

      \int dx \phi_i n_f = \int dx \phi_i n_p
      \int dx \phi_i n_f = \int dx \phi_i \int dv f
      \int dx \phi_i n_f = \int dx \phi_i \int dv \sum_p w_p \delta(x - x_p) \delta(v - v_p)
      \int dx \phi_i n_f = \int dx \phi_i \sum_p w_p \delta(x - x_p)
                   M n_F = M_p w_p

    where

      (M_p){ip} = \phi_i(x_p)

    which is just a scaled version of the charge density. The second moment, momentum density, is given by

      \int dx \phi_i p_f = m \int dx \phi_i \int dv v f
      \int dx \phi_i p_f = m \int dx \phi_i \sum_p w_p \delta(x - x_p) v_p
                   M p_F = M_p v_p w_p

    And finally the third moment, pressure, is given by

      \int dx \phi_i pr_f = m \int dx \phi_i \int dv (v - u)^2 f
      \int dx \phi_i pr_f = m \int dx \phi_i \sum_p w_p \delta(x - x_p) (v_p - u)^2
                   M pr_F = M_p (v_p - u)^2 w_p
                          = M_p (v_p - p_F(x_p) / m n_F(x_p))^2 w_p
                          = M_p (v_p - (\sum_j p_F \phi_j(x_p)) / m (\sum_k n_F \phi_k(x_p)))^2 w_p

    Here we need all FEM basis functions \phi_i that see that particle p.

  To run the code with particles sinusoidally perturbed in x space use the test "pp_poisson_bsi_1d_4" or "pp_poisson_bsi_2d_4"
  According to Lukas, good damping results come at ~16k particles per cell

  Swarm CellDMs
  =============
  Name: "space"
  Fields: DMSwarmPICField_coor, "velocity"
  Coordinates: DMSwarmPICField_coor

  Name: "velocity"
  Fields: "w_q"
  Coordinates: "velocity"

  Name: "moments"
  Fields: "w_q"
  Coordinates: DMSwarmPICField_coor

  Name: "moment fields"
  Fields: "velocity"
  Coordinates: DMSwarmPICField_coor

  To visualize the maximum electric field use

    -efield_monitor

  To monitor velocity moments of the distribution use

    -ptof_pc_type lu -moments_monitor

  To monitor the particle positions in phase space use

    -positions_monitor

  To monitor the charge density, E field, and potential use

    -poisson_monitor

  To monitor the remapping field use

    -remap_uf_view draw

  To visualize the swarm distribution use

    -ts_monitor_hg_swarm

  To visualize the particles, we can use

    -ts_monitor_sp_swarm -ts_monitor_sp_swarm_retain 0 -ts_monitor_sp_swarm_phase 1 -draw_size 500,500
*/
#include <petsctao.h>
#include <petscts.h>
#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscfe.h>
#include <petscds.h>
#include <petscbag.h>
#include <petscdraw.h>
#include <petsc/private/petscfeimpl.h> /* For interpolation */
#include <petsc/private/dmswarmimpl.h> /* For swarm debugging */
#include "petscdm.h"
#include "petscdmlabel.h"

PETSC_EXTERN PetscErrorCode stream(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);
PETSC_EXTERN PetscErrorCode line(PetscInt, PetscReal, const PetscReal[], PetscInt, PetscScalar[], void *);

const char *EMTypes[] = {"primal", "mixed", "coulomb", "none", "EMType", "EM_", NULL};
typedef enum {
  EM_PRIMAL,
  EM_MIXED,
  EM_COULOMB,
  EM_NONE
} EMType;

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

typedef enum {
  E_MONITOR_NONE,
  E_MONITOR_FULL,
  E_MONITOR_QUIET
} EMonitorType;
const char *const EMonitorTypes[] = {"NONE", "FULL", "QUIET", "EMonitorType", "E_MONITOR_", NULL};

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
  PetscInt         s;    // Starting sample (we ignore some in the beginning)
  PetscInt         e;    // Ending sample
  PetscInt         per;  // Period of fitting
  const PetscReal *t;    // Time for each sample
  const PetscReal *Emax; // Emax for each sample
} EmaxCtx;

typedef struct {
  PetscBag     bag;                  // Problem parameters
  PetscBool    error;                // Flag for printing the error
  PetscInt     remapFreq;            // Number of timesteps between remapping
  EMonitorType efield_monitor;       // Flag to show electric field monitor
  PetscBool    moment_monitor;       // Flag to show distribution moment monitor
  PetscBool    moment_field_monitor; // Flag to show moment field monitor
  PetscBool    positions_monitor;    // Flag to show particle positins at each time step
  PetscBool    poisson_monitor;      // Flag to display charge, E field, and potential at each solve
  PetscBool    initial_monitor;      // Flag to monitor the initial conditions
  PetscInt     velocity_monitor;     // Cell to monitor the velocity distribution for
  PetscBool    perturbed_weights;    // Uniformly sample x,v space with gaussian weights
  PetscInt     ostep;                // Print the energy at each ostep time steps
  PetscInt     numParticles;
  PetscReal    timeScale;              /* Nondimensionalizing time scale */
  PetscReal    charges[2];             /* The charges of each species */
  PetscReal    masses[2];              /* The masses of each species */
  PetscReal    thermal_energy[2];      /* Thermal Energy (used to get other constants)*/
  PetscReal    cosine_coefficients[2]; /*(alpha, k)*/
  PetscReal    totalWeight;
  PetscReal    stepSize;
  PetscInt     steps;
  PetscReal    initVel;
  EMType       em;           // Type of electrostatic model
  SNES         snes;         // EM solver
  DM           dmMom;        // The DM for moment fields
  DM           dmN;          // The DM for number density fields
  IS           isN;          // The IS mapping dmN into dmMom
  Mat          MN;           // The finite element mass matrix for number density
  DM           dmP;          // The DM for momentum density fields
  IS           isP;          // The IS mapping dmP into dmMom
  Mat          MP;           // The finite element mass matrix for momentum density
  DM           dmE;          // The DM for energy density (pressure) fields
  IS           isE;          // The IS mapping dmE into dmMom
  Mat          ME;           // The finite element mass matrix for energy density (pressure)
  DM           dmPot;        // The DM for potential
  Mat          fftPot;       // Fourier Transform operator for the potential
  Vec          fftX, fftY;   //   FFT vectors with phases added (complex parts)
  IS           fftReal;      //   The indices for real parts
  IS           isPot;        // The IS for potential, or NULL in primal
  Mat          M;            // The finite element mass matrix for potential
  PetscFEGeom *fegeom;       // Geometric information for the DM cells
  PetscDrawHG  drawhgic_x;   // Histogram of the particle weight in each X cell
  PetscDrawHG  drawhgic_v;   // Histogram of the particle weight in each X cell
  PetscDrawHG  drawhgcell_v; // Histogram of the particle weight in a given cell
  PetscBool    validE;       // Flag to indicate E-field in swarm is valid
  PetscReal    drawlgEmin;   // The minimum lg(E) to plot
  PetscDrawLG  drawlgE;      // Logarithm of maximum electric field
  PetscDrawSP  drawspE;      // Electric field at particle positions
  PetscDrawSP  drawspX;      // Particle positions
  PetscViewer  viewerRho;    // Charge density viewer
  PetscViewer  viewerRhoHat; // Charge density Fourier Transform viewer
  PetscViewer  viewerPhi;    // Potential viewer
  PetscViewer  viewerN;      // Number density viewer
  PetscViewer  viewerP;      // Momentum density viewer
  PetscViewer  viewerE;      // Energy density (pressure) viewer
  PetscViewer  viewerNRes;   // Number density residual viewer
  PetscViewer  viewerPRes;   // Momentum density residual viewer
  PetscViewer  viewerERes;   // Energy density (pressure) residual viewer
  PetscDrawLG  drawlgMomRes; // Residuals for the moment equations
  DM           swarm;        // The particle swarm
  PetscRandom  random;       // Used for particle perturbations
  PetscBool    twostream;    // Flag for activating 2-stream setup
  PetscBool    checkweights; // Check weight normalization
  PetscInt     checkVRes;    // Flag to check/output velocity residuals for nightly tests
  PetscBool    checkLandau;  // Check the Landau damping result
  EmaxCtx      emaxCtx;      // Information for fit to decay profile
  PetscReal    gamma;        // The damping rate for Landau damping
  PetscReal    omega;        // The perturbed oscillation frequency for Landau damping

  PetscLogEvent RhsXEvent, RhsVEvent, ESolveEvent, ETabEvent;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  PetscInt d                      = 2;
  PetscInt maxSpecies             = 2;
  options->error                  = PETSC_FALSE;
  options->remapFreq              = 1;
  options->efield_monitor         = E_MONITOR_NONE;
  options->moment_monitor         = PETSC_FALSE;
  options->moment_field_monitor   = PETSC_FALSE;
  options->initial_monitor        = PETSC_FALSE;
  options->perturbed_weights      = PETSC_FALSE;
  options->poisson_monitor        = PETSC_FALSE;
  options->positions_monitor      = PETSC_FALSE;
  options->velocity_monitor       = -1;
  options->ostep                  = 100;
  options->timeScale              = 2.0e-14;
  options->charges[0]             = -1.0;
  options->charges[1]             = 1.0;
  options->masses[0]              = 1.0;
  options->masses[1]              = 1000.0;
  options->thermal_energy[0]      = 1.0;
  options->thermal_energy[1]      = 1.0;
  options->cosine_coefficients[0] = 0.01;
  options->cosine_coefficients[1] = 0.5;
  options->initVel                = 1;
  options->totalWeight            = 1.0;
  options->drawhgic_x             = NULL;
  options->drawhgic_v             = NULL;
  options->drawhgcell_v           = NULL;
  options->validE                 = PETSC_FALSE;
  options->drawlgEmin             = -6;
  options->drawlgE                = NULL;
  options->drawspE                = NULL;
  options->drawspX                = NULL;
  options->viewerRho              = NULL;
  options->viewerRhoHat           = NULL;
  options->viewerPhi              = NULL;
  options->viewerN                = NULL;
  options->viewerP                = NULL;
  options->viewerE                = NULL;
  options->viewerNRes             = NULL;
  options->viewerPRes             = NULL;
  options->viewerERes             = NULL;
  options->drawlgMomRes           = NULL;
  options->em                     = EM_COULOMB;
  options->snes                   = NULL;
  options->dmMom                  = NULL;
  options->dmN                    = NULL;
  options->MN                     = NULL;
  options->dmP                    = NULL;
  options->MP                     = NULL;
  options->dmE                    = NULL;
  options->ME                     = NULL;
  options->dmPot                  = NULL;
  options->fftPot                 = NULL;
  options->fftX                   = NULL;
  options->fftY                   = NULL;
  options->fftReal                = NULL;
  options->isPot                  = NULL;
  options->M                      = NULL;
  options->numParticles           = 32768;
  options->twostream              = PETSC_FALSE;
  options->checkweights           = PETSC_FALSE;
  options->checkVRes              = 0;
  options->checkLandau            = PETSC_FALSE;
  options->emaxCtx.s              = 50;
  options->emaxCtx.per            = 100;

  PetscOptionsBegin(comm, "", "Landau Damping and Two Stream options", "DMSWARM");
  PetscCall(PetscOptionsBool("-error", "Flag to print the error", __FILE__, options->error, &options->error, NULL));
  PetscCall(PetscOptionsInt("-remap_freq", "Number", __FILE__, options->remapFreq, &options->remapFreq, NULL));
  PetscCall(PetscOptionsEnum("-efield_monitor", "Flag to record and plot log(max E) over time", __FILE__, EMonitorTypes, (PetscEnum)options->efield_monitor, (PetscEnum *)&options->efield_monitor, NULL));
  PetscCall(PetscOptionsReal("-efield_min_monitor", "Minimum E field to plot", __FILE__, options->drawlgEmin, &options->drawlgEmin, NULL));
  PetscCall(PetscOptionsBool("-moments_monitor", "Flag to show moments table", __FILE__, options->moment_monitor, &options->moment_monitor, NULL));
  PetscCall(PetscOptionsBool("-moment_field_monitor", "Flag to show moment fields", __FILE__, options->moment_field_monitor, &options->moment_field_monitor, NULL));
  PetscCall(PetscOptionsBool("-ics_monitor", "Flag to show initial condition histograms", __FILE__, options->initial_monitor, &options->initial_monitor, NULL));
  PetscCall(PetscOptionsBool("-positions_monitor", "The flag to show particle positions", __FILE__, options->positions_monitor, &options->positions_monitor, NULL));
  PetscCall(PetscOptionsBool("-poisson_monitor", "The flag to show charges, Efield and potential solve", __FILE__, options->poisson_monitor, &options->poisson_monitor, NULL));
  PetscCall(PetscOptionsInt("-velocity_monitor", "Cell to show velocity histograms", __FILE__, options->velocity_monitor, &options->velocity_monitor, NULL));
  PetscCall(PetscOptionsBool("-twostream", "Run two stream instability", __FILE__, options->twostream, &options->twostream, NULL));
  PetscCall(PetscOptionsBool("-perturbed_weights", "Flag to run uniform sampling with perturbed weights", __FILE__, options->perturbed_weights, &options->perturbed_weights, NULL));
  PetscCall(PetscOptionsBool("-check_weights", "Ensure all particle weights are positive", __FILE__, options->checkweights, &options->checkweights, NULL));
  PetscCall(PetscOptionsBool("-check_landau", "Check the decay from Landau damping", __FILE__, options->checkLandau, &options->checkLandau, NULL));
  PetscCall(PetscOptionsInt("-output_step", "Number of time steps between output", __FILE__, options->ostep, &options->ostep, NULL));
  PetscCall(PetscOptionsReal("-timeScale", "Nondimensionalizing time scale", __FILE__, options->timeScale, &options->timeScale, NULL));
  PetscCall(PetscOptionsInt("-check_vel_res", "Check particle velocity residuals for nightly tests", __FILE__, options->checkVRes, &options->checkVRes, NULL));
  PetscCall(PetscOptionsReal("-initial_velocity", "Initial velocity of perturbed particle", __FILE__, options->initVel, &options->initVel, NULL));
  PetscCall(PetscOptionsReal("-total_weight", "Total weight of all particles", __FILE__, options->totalWeight, &options->totalWeight, NULL));
  PetscCall(PetscOptionsRealArray("-cosine_coefficients", "Amplitude and frequency of cosine equation used in initialization", __FILE__, options->cosine_coefficients, &d, NULL));
  PetscCall(PetscOptionsRealArray("-charges", "Species charges", __FILE__, options->charges, &maxSpecies, NULL));
  PetscCall(PetscOptionsEnum("-em_type", "Type of electrostatic solver", __FILE__, EMTypes, (PetscEnum)options->em, (PetscEnum *)&options->em, NULL));
  PetscCall(PetscOptionsInt("-emax_start_step", "First time step to use for Emax fits", __FILE__, options->emaxCtx.s, &options->emaxCtx.s, NULL));
  PetscCall(PetscOptionsInt("-emax_solve_step", "Number of time steps between Emax fits", __FILE__, options->emaxCtx.per, &options->emaxCtx.per, NULL));
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

    if (user->efield_monitor == E_MONITOR_FULL) {
      PetscCall(PetscDrawCreate(comm, NULL, "Max Electric Field", 0, 0, 400, 300, &draw));
      PetscCall(PetscDrawSetSave(draw, "ex2_Efield"));
      PetscCall(PetscDrawSetFromOptions(draw));
    } else {
      PetscCall(PetscDrawOpenNull(comm, &draw));
    }
    PetscCall(PetscDrawLGCreate(draw, 1, &user->drawlgE));
    PetscCall(PetscDrawDestroy(&draw));
    PetscCall(PetscDrawLGGetAxis(user->drawlgE, &axis));
    PetscCall(PetscDrawAxisSetLabels(axis, "Max Electric Field", "time", "E_max"));
    PetscCall(PetscDrawLGSetLimits(user->drawlgE, 0., user->steps * user->stepSize, user->drawlgEmin, 0.));
  }

  if (user->initial_monitor) {
    PetscDraw     drawic_x, drawic_v;
    PetscDrawAxis axis1, axis2;
    PetscReal     dmboxlower[2], dmboxupper[2];
    PetscInt      dim, cStart, cEnd;

    PetscCall(DMGetDimension(sw, &dim));
    PetscCall(DMGetBoundingBox(dm, dmboxlower, dmboxupper));
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));

    PetscCall(PetscDrawCreate(comm, NULL, "monitor_initial_conditions_x", 0, 300, 400, 300, &drawic_x));
    PetscCall(PetscDrawSetSave(drawic_x, "ex2_ic_x"));
    PetscCall(PetscDrawSetFromOptions(drawic_x));
    PetscCall(PetscDrawHGCreate(drawic_x, (int)dim, &user->drawhgic_x));
    PetscCall(PetscDrawHGCalcStats(user->drawhgic_x, PETSC_TRUE));
    PetscCall(PetscDrawHGGetAxis(user->drawhgic_x, &axis1));
    PetscCall(PetscDrawHGSetNumberBins(user->drawhgic_x, (int)(cEnd - cStart)));
    PetscCall(PetscDrawAxisSetLabels(axis1, "Initial X Distribution", "X", "weight"));
    PetscCall(PetscDrawAxisSetLimits(axis1, dmboxlower[0], dmboxupper[0], 0, 0));
    PetscCall(PetscDrawDestroy(&drawic_x));

    PetscCall(PetscDrawCreate(comm, NULL, "monitor_initial_conditions_v", 400, 300, 400, 300, &drawic_v));
    PetscCall(PetscDrawSetSave(drawic_v, "ex9_ic_v"));
    PetscCall(PetscDrawSetFromOptions(drawic_v));
    PetscCall(PetscDrawHGCreate(drawic_v, (int)dim, &user->drawhgic_v));
    PetscCall(PetscDrawHGCalcStats(user->drawhgic_v, PETSC_TRUE));
    PetscCall(PetscDrawHGGetAxis(user->drawhgic_v, &axis2));
    PetscCall(PetscDrawHGSetNumberBins(user->drawhgic_v, 21));
    PetscCall(PetscDrawAxisSetLabels(axis2, "Initial V_x Distribution", "V", "weight"));
    PetscCall(PetscDrawAxisSetLimits(axis2, -6, 6, 0, 0));
    PetscCall(PetscDrawDestroy(&drawic_v));
  }

  if (user->velocity_monitor >= 0) {
    DM            vdm;
    DMSwarmCellDM celldm;
    PetscDraw     drawcell_v;
    PetscDrawAxis axis;
    PetscReal     dmboxlower[2], dmboxupper[2];
    PetscInt      dim;
    char          title[PETSC_MAX_PATH_LEN];

    PetscCall(DMSwarmGetCellDMByName(sw, "velocity", &celldm));
    PetscCall(DMSwarmCellDMGetDM(celldm, &vdm));
    PetscCall(DMGetDimension(vdm, &dim));
    PetscCall(DMGetBoundingBox(vdm, dmboxlower, dmboxupper));

    PetscCall(PetscSNPrintf(title, PETSC_MAX_PATH_LEN, "Cell %" PetscInt_FMT ": Velocity Distribution", user->velocity_monitor));
    PetscCall(PetscDrawCreate(comm, NULL, title, 400, 300, 400, 300, &drawcell_v));
    PetscCall(PetscDrawSetSave(drawcell_v, "ex2_cell_v"));
    PetscCall(PetscDrawSetFromOptions(drawcell_v));
    PetscCall(PetscDrawHGCreate(drawcell_v, (int)dim, &user->drawhgcell_v));
    PetscCall(PetscDrawHGCalcStats(user->drawhgcell_v, PETSC_TRUE));
    PetscCall(PetscDrawHGGetAxis(user->drawhgcell_v, &axis));
    PetscCall(PetscDrawHGSetNumberBins(user->drawhgcell_v, 21));
    PetscCall(PetscDrawAxisSetLabels(axis, "V_x Distribution", "V", "weight"));
    PetscCall(PetscDrawAxisSetLimits(axis, dmboxlower[0], dmboxupper[0], 0, 0));
    PetscCall(PetscDrawDestroy(&drawcell_v));
  }

  if (user->positions_monitor) {
    PetscDraw     draw;
    PetscDrawAxis axis;

    PetscCall(PetscDrawCreate(comm, NULL, "Particle Position", 0, 0, 400, 300, &draw));
    PetscCall(PetscDrawSetSave(draw, "ex9_pos"));
    PetscCall(PetscDrawSetFromOptions(draw));
    PetscCall(PetscDrawSPCreate(draw, 10, &user->drawspX));
    PetscCall(PetscDrawDestroy(&draw));
    PetscCall(PetscDrawSPSetDimension(user->drawspX, 1));
    PetscCall(PetscDrawSPGetAxis(user->drawspX, &axis));
    PetscCall(PetscDrawAxisSetLabels(axis, "Particles", "x", "v"));
    PetscCall(PetscDrawSPReset(user->drawspX));
  }
  if (user->poisson_monitor) {
    Vec           rho, rhohat, phi;
    PetscDraw     draw;
    PetscDrawAxis axis;

    PetscCall(PetscDrawCreate(comm, NULL, "Electric_Field", 0, 0, 400, 300, &draw));
    PetscCall(PetscDrawSetFromOptions(draw));
    PetscCall(PetscDrawSetSave(draw, "ex9_E_spatial"));
    PetscCall(PetscDrawSPCreate(draw, 10, &user->drawspE));
    PetscCall(PetscDrawDestroy(&draw));
    PetscCall(PetscDrawSPSetDimension(user->drawspE, 1));
    PetscCall(PetscDrawSPGetAxis(user->drawspE, &axis));
    PetscCall(PetscDrawAxisSetLabels(axis, "Particles", "x", "E"));
    PetscCall(PetscDrawSPReset(user->drawspE));

    PetscCall(PetscViewerDrawOpen(comm, NULL, "Charge Density", 0, 0, 400, 300, &user->viewerRho));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)user->viewerRho, "rho_"));
    PetscCall(PetscViewerDrawGetDraw(user->viewerRho, 0, &draw));
    PetscCall(PetscDrawSetSave(draw, "ex9_rho_spatial"));
    PetscCall(PetscViewerSetFromOptions(user->viewerRho));
    PetscCall(DMGetNamedGlobalVector(user->dmPot, "rho", &rho));
    PetscCall(PetscObjectSetName((PetscObject)rho, "charge_density"));
    PetscCall(DMRestoreNamedGlobalVector(user->dmPot, "rho", &rho));

    PetscInt dim, N;

    PetscCall(DMGetDimension(user->dmPot, &dim));
    if (dim == 1) {
      PetscCall(DMGetNamedGlobalVector(user->dmPot, "rhohat", &rhohat));
      PetscCall(VecGetSize(rhohat, &N));
      PetscCall(MatCreateFFT(comm, dim, &N, MATFFTW, &user->fftPot));
      PetscCall(DMRestoreNamedGlobalVector(user->dmPot, "rhohat", &rhohat));
      PetscCall(MatCreateVecs(user->fftPot, &user->fftX, &user->fftY));
      PetscCall(ISCreateStride(PETSC_COMM_SELF, N, 0, 1, &user->fftReal));
    }

    PetscCall(PetscViewerDrawOpen(comm, NULL, "rhohat: Charge Density FT", 0, 0, 400, 300, &user->viewerRhoHat));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)user->viewerRhoHat, "rhohat_"));
    PetscCall(PetscViewerDrawGetDraw(user->viewerRhoHat, 0, &draw));
    PetscCall(PetscDrawSetSave(draw, "ex9_rho_ft"));
    PetscCall(PetscViewerSetFromOptions(user->viewerRhoHat));
    PetscCall(DMGetNamedGlobalVector(user->dmPot, "rhohat", &rhohat));
    PetscCall(PetscObjectSetName((PetscObject)rhohat, "charge_density_ft"));
    PetscCall(DMRestoreNamedGlobalVector(user->dmPot, "rhohat", &rhohat));

    PetscCall(PetscViewerDrawOpen(comm, NULL, "Potential", 400, 0, 400, 300, &user->viewerPhi));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)user->viewerPhi, "phi_"));
    PetscCall(PetscViewerDrawGetDraw(user->viewerPhi, 0, &draw));
    PetscCall(PetscDrawSetSave(draw, "ex9_phi_spatial"));
    PetscCall(PetscViewerSetFromOptions(user->viewerPhi));
    PetscCall(DMGetNamedGlobalVector(user->dmPot, "phi", &phi));
    PetscCall(PetscObjectSetName((PetscObject)phi, "potential"));
    PetscCall(DMRestoreNamedGlobalVector(user->dmPot, "phi", &phi));
  }
  if (user->moment_field_monitor) {
    Vec       n, p, e;
    Vec       nres, pres, eres;
    PetscDraw draw;

    PetscCall(PetscViewerDrawOpen(comm, NULL, "Number Density", 400, 0, 400, 300, &user->viewerN));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)user->viewerN, "n_"));
    PetscCall(PetscViewerDrawGetDraw(user->viewerN, 0, &draw));
    PetscCall(PetscDrawSetSave(draw, "ex4_n_spatial"));
    PetscCall(PetscViewerSetFromOptions(user->viewerN));
    PetscCall(DMGetNamedGlobalVector(user->dmN, "n", &n));
    PetscCall(PetscObjectSetName((PetscObject)n, "Number Density"));
    PetscCall(DMRestoreNamedGlobalVector(user->dmN, "n", &n));

    PetscCall(PetscViewerDrawOpen(comm, NULL, "Momentum Density", 800, 0, 400, 300, &user->viewerP));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)user->viewerP, "p_"));
    PetscCall(PetscViewerDrawGetDraw(user->viewerP, 0, &draw));
    PetscCall(PetscDrawSetSave(draw, "ex4_p_spatial"));
    PetscCall(PetscViewerSetFromOptions(user->viewerP));
    PetscCall(DMGetNamedGlobalVector(user->dmP, "p", &p));
    PetscCall(PetscObjectSetName((PetscObject)p, "Momentum Density"));
    PetscCall(DMRestoreNamedGlobalVector(user->dmP, "p", &p));

    PetscCall(PetscViewerDrawOpen(comm, NULL, "Emergy Density (Pressure)", 1200, 0, 400, 300, &user->viewerE));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)user->viewerE, "e_"));
    PetscCall(PetscViewerDrawGetDraw(user->viewerE, 0, &draw));
    PetscCall(PetscDrawSetSave(draw, "ex4_e_spatial"));
    PetscCall(PetscViewerSetFromOptions(user->viewerE));
    PetscCall(DMGetNamedGlobalVector(user->dmE, "e", &e));
    PetscCall(PetscObjectSetName((PetscObject)e, "Energy Density (Pressure)"));
    PetscCall(DMRestoreNamedGlobalVector(user->dmE, "e", &e));

    PetscDrawAxis axis;

    PetscCall(PetscDrawCreate(comm, NULL, "Moment Residual", 0, 320, 400, 300, &draw));
    PetscCall(PetscDrawSetSave(draw, "ex4_moment_res"));
    PetscCall(PetscDrawSetFromOptions(draw));
    PetscCall(PetscDrawLGCreate(draw, 3, &user->drawlgMomRes));
    PetscCall(PetscDrawDestroy(&draw));
    PetscCall(PetscDrawLGGetAxis(user->drawlgMomRes, &axis));
    PetscCall(PetscDrawAxisSetLabels(axis, "Moment Residial", "time", "Residual Norm"));
    PetscCall(PetscDrawLGSetLimits(user->drawlgMomRes, 0., user->steps * user->stepSize, -8, 0));

    PetscCall(PetscViewerDrawOpen(comm, NULL, "Number Density Residual", 400, 300, 400, 300, &user->viewerNRes));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)user->viewerNRes, "nres_"));
    PetscCall(PetscViewerDrawGetDraw(user->viewerNRes, 0, &draw));
    PetscCall(PetscDrawSetSave(draw, "ex4_nres_spatial"));
    PetscCall(PetscViewerSetFromOptions(user->viewerNRes));
    PetscCall(DMGetNamedGlobalVector(user->dmN, "nres", &nres));
    PetscCall(PetscObjectSetName((PetscObject)nres, "Number Density Residual"));
    PetscCall(DMRestoreNamedGlobalVector(user->dmN, "nres", &nres));

    PetscCall(PetscViewerDrawOpen(comm, NULL, "Momentum Density Residual", 800, 300, 400, 300, &user->viewerPRes));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)user->viewerPRes, "pres_"));
    PetscCall(PetscViewerDrawGetDraw(user->viewerPRes, 0, &draw));
    PetscCall(PetscDrawSetSave(draw, "ex4_pres_spatial"));
    PetscCall(PetscViewerSetFromOptions(user->viewerPRes));
    PetscCall(DMGetNamedGlobalVector(user->dmP, "pres", &pres));
    PetscCall(PetscObjectSetName((PetscObject)pres, "Momentum Density Residual"));
    PetscCall(DMRestoreNamedGlobalVector(user->dmP, "pres", &pres));

    PetscCall(PetscViewerDrawOpen(comm, NULL, "Energy Density Residual", 1200, 300, 400, 300, &user->viewerERes));
    PetscCall(PetscObjectSetOptionsPrefix((PetscObject)user->viewerERes, "eres_"));
    PetscCall(PetscViewerDrawGetDraw(user->viewerERes, 0, &draw));
    PetscCall(PetscDrawSetSave(draw, "ex4_eres_spatial"));
    PetscCall(PetscViewerSetFromOptions(user->viewerERes));
    PetscCall(DMGetNamedGlobalVector(user->dmE, "eres", &eres));
    PetscCall(PetscObjectSetName((PetscObject)eres, "Energy Density Residual"));
    PetscCall(DMRestoreNamedGlobalVector(user->dmE, "eres", &eres));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DestroyContext(AppCtx *user)
{
  PetscFunctionBeginUser;
  PetscCall(PetscDrawHGDestroy(&user->drawhgic_x));
  PetscCall(PetscDrawHGDestroy(&user->drawhgic_v));
  PetscCall(PetscDrawHGDestroy(&user->drawhgcell_v));

  PetscCall(PetscDrawLGDestroy(&user->drawlgE));
  PetscCall(PetscDrawSPDestroy(&user->drawspE));
  PetscCall(PetscDrawSPDestroy(&user->drawspX));
  PetscCall(PetscViewerDestroy(&user->viewerRho));
  PetscCall(PetscViewerDestroy(&user->viewerRhoHat));
  PetscCall(MatDestroy(&user->fftPot));
  PetscCall(VecDestroy(&user->fftX));
  PetscCall(VecDestroy(&user->fftY));
  PetscCall(ISDestroy(&user->fftReal));
  PetscCall(PetscViewerDestroy(&user->viewerPhi));
  PetscCall(PetscViewerDestroy(&user->viewerN));
  PetscCall(PetscViewerDestroy(&user->viewerP));
  PetscCall(PetscViewerDestroy(&user->viewerE));
  PetscCall(PetscViewerDestroy(&user->viewerNRes));
  PetscCall(PetscViewerDestroy(&user->viewerPRes));
  PetscCall(PetscViewerDestroy(&user->viewerERes));
  PetscCall(PetscDrawLGDestroy(&user->drawlgMomRes));

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

static PetscErrorCode computeVelocityFEMMoments(DM sw, PetscReal moments[], AppCtx *user)
{
  DMSwarmCellDM celldm;
  DM            vdm;
  Vec           u[1];
  const char   *fields[1] = {"w_q"};

  PetscFunctionBegin;
  PetscCall(DMSwarmSetCellDMActive(sw, "velocity"));
  PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
  PetscCall(DMSwarmCellDMGetDM(celldm, &vdm));
  PetscCall(DMGetGlobalVector(vdm, &u[0]));
  PetscCall(DMSwarmProjectFields(sw, vdm, 1, fields, u, SCATTER_FORWARD));
  PetscCall(DMPlexComputeMoments(vdm, u[0], moments));
  PetscCall(DMRestoreGlobalVector(vdm, &u[0]));
  PetscCall(DMSwarmSetCellDMActive(sw, "space"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void f0_grad_phi2(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] = 0.;
  for (PetscInt d = 0; d < dim; ++d) f0[0] += PetscSqr(u_x[uOff_x[0] + d * dim + d]);
}

// Our model is E_max(t) = C e^{-gamma t} |cos(omega t - phi)|
static PetscErrorCode ComputeEmaxResidual(Tao tao, Vec x, Vec res, void *user)
{
  EmaxCtx           *ctx = (EmaxCtx *)user;
  const PetscScalar *a;
  PetscScalar       *F;
  PetscReal          C, gamma, omega, phi;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x, &a));
  PetscCall(VecGetArray(res, &F));
  C     = PetscRealPart(a[0]);
  gamma = PetscRealPart(a[1]);
  omega = PetscRealPart(a[2]);
  phi   = PetscRealPart(a[3]);
  PetscCall(VecRestoreArrayRead(x, &a));
  for (PetscInt i = ctx->s; i < ctx->e; ++i) F[i - ctx->s] = PetscPowReal(10., ctx->Emax[i]) - C * PetscExpReal(-gamma * ctx->t[i]) * PetscAbsReal(PetscCosReal(omega * ctx->t[i] - phi));
  PetscCall(VecRestoreArray(res, &F));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// The Jacobian of the residual J = dr(x)/dx
static PetscErrorCode ComputeEmaxJacobian(Tao tao, Vec x, Mat J, Mat Jpre, void *user)
{
  EmaxCtx           *ctx = (EmaxCtx *)user;
  const PetscScalar *a;
  PetscScalar       *jac;
  PetscReal          C, gamma, omega, phi;
  const PetscInt     n = ctx->e - ctx->s;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x, &a));
  C     = PetscRealPart(a[0]);
  gamma = PetscRealPart(a[1]);
  omega = PetscRealPart(a[2]);
  phi   = PetscRealPart(a[3]);
  PetscCall(VecRestoreArrayRead(x, &a));
  PetscCall(MatDenseGetArray(J, &jac));
  for (PetscInt i = 0; i < n; ++i) {
    const PetscInt k = i + ctx->s;

    jac[i * 4 + 0] = -PetscExpReal(-gamma * ctx->t[k]) * PetscAbsReal(PetscCosReal(omega * ctx->t[k] - phi));
    jac[i * 4 + 1] = C * ctx->t[k] * PetscExpReal(-gamma * ctx->t[k]) * PetscAbsReal(PetscCosReal(omega * ctx->t[k] - phi));
    jac[i * 4 + 2] = C * ctx->t[k] * PetscExpReal(-gamma * ctx->t[k]) * (PetscCosReal(omega * ctx->t[k] - phi) < 0. ? -1. : 1.) * PetscSinReal(omega * ctx->t[k] - phi);
    jac[i * 4 + 3] = -C * PetscExpReal(-gamma * ctx->t[k]) * (PetscCosReal(omega * ctx->t[k] - phi) < 0. ? -1. : 1.) * PetscSinReal(omega * ctx->t[k] - phi);
  }
  PetscCall(MatDenseRestoreArray(J, &jac));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Our model is log_10 E_max(t) = log_10 C - gamma t log_10 e + log_10 |cos(omega t - phi)|
static PetscErrorCode ComputeLogEmaxResidual(Tao tao, Vec x, Vec res, void *user)
{
  EmaxCtx           *ctx = (EmaxCtx *)user;
  const PetscScalar *a;
  PetscScalar       *F;
  PetscReal          C, gamma, omega, phi;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x, &a));
  PetscCall(VecGetArray(res, &F));
  C     = PetscRealPart(a[0]);
  gamma = PetscRealPart(a[1]);
  omega = PetscRealPart(a[2]);
  phi   = PetscRealPart(a[3]);
  PetscCall(VecRestoreArrayRead(x, &a));
  for (PetscInt i = ctx->s; i < ctx->e; ++i) {
    if (C < 0) {
      F[i - ctx->s] = 1e10;
      continue;
    }
    F[i - ctx->s] = ctx->Emax[i] - (PetscLog10Real(C) - gamma * ctx->t[i] * PetscLog10Real(PETSC_E) + PetscLog10Real(PetscAbsReal(PetscCosReal(omega * ctx->t[i] - phi))));
  }
  PetscCall(VecRestoreArray(res, &F));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// The Jacobian of the residual J = dr(x)/dx
static PetscErrorCode ComputeLogEmaxJacobian(Tao tao, Vec x, Mat J, Mat Jpre, void *user)
{
  EmaxCtx           *ctx = (EmaxCtx *)user;
  const PetscScalar *a;
  PetscScalar       *jac;
  PetscReal          C, omega, phi;
  const PetscInt     n = ctx->e - ctx->s;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x, &a));
  C     = PetscRealPart(a[0]);
  omega = PetscRealPart(a[2]);
  phi   = PetscRealPart(a[3]);
  PetscCall(VecRestoreArrayRead(x, &a));
  PetscCall(MatDenseGetArray(J, &jac));
  for (PetscInt i = 0; i < n; ++i) {
    const PetscInt k = i + ctx->s;

    jac[0 * n + i] = -1. / (PetscLog10Real(PETSC_E) * C);
    jac[1 * n + i] = ctx->t[k] * PetscLog10Real(PETSC_E);
    jac[2 * n + i] = (PetscCosReal(omega * ctx->t[k] - phi) < 0. ? -1. : 1.) * ctx->t[k] * PetscSinReal(omega * ctx->t[k] - phi) / (PetscLog10Real(PETSC_E) * PetscAbsReal(PetscCosReal(omega * ctx->t[k] - phi)));
    jac[3 * n + i] = -(PetscCosReal(omega * ctx->t[k] - phi) < 0. ? -1. : 1.) * PetscSinReal(omega * ctx->t[k] - phi) / (PetscLog10Real(PETSC_E) * PetscAbsReal(PetscCosReal(omega * ctx->t[k] - phi)));
  }
  PetscCall(MatDenseRestoreArray(J, &jac));
  PetscCall(MatViewFromOptions(J, NULL, "-emax_jac_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MonitorEField(TS ts, PetscInt step, PetscReal t, Vec U, void *ctx)
{
  AppCtx     *user = (AppCtx *)ctx;
  DM          sw;
  PetscScalar intESq;
  PetscReal  *E, *x, *weight;
  PetscReal   Enorm = 0., lgEnorm, lgEmax, sum = 0., Emax = 0., chargesum = 0.;
  PetscReal   pmoments[4]; /* \int f, \int v f, \int v^2 f */
  PetscInt   *species, dim, Np, gNp;
  MPI_Comm    comm;
  PetscMPIInt rank;

  PetscFunctionBeginUser;
  if (step < 0 || !user->validE) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(PetscObjectGetComm((PetscObject)ts, &comm));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
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
    for (PetscInt d = 0; d < 1; ++d) {
      PetscReal temp = PetscAbsReal(E[p * dim + d]);
      if (temp > Emax) Emax = temp;
    }
    Enorm += PetscSqrtReal(E[p * dim] * E[p * dim]);
    sum += E[p * dim];
    chargesum += user->charges[0] * weight[p];
  }
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &Emax, 1, MPIU_REAL, MPIU_MAX, comm));
  lgEnorm = Enorm != 0 ? PetscLog10Real(Enorm) : -16.;
  lgEmax  = Emax != 0 ? PetscLog10Real(Emax) : user->drawlgEmin;

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
  if (user->efield_monitor == E_MONITOR_FULL) {
    PetscDraw draw;

    PetscCall(PetscDrawLGDraw(user->drawlgE));
    PetscCall(PetscDrawLGGetDraw(user->drawlgE, &draw));
    PetscCall(PetscDrawSave(draw));

    PetscCall(DMSwarmComputeMoments(sw, "velocity", "w_q", pmoments));
    PetscCall(PetscPrintf(comm, "E: %f\t%+e\t%e\t%f\t%20.15e\t%f\t%f\t%f\t%20.15e\t%20.15e\t%20.15e\t%" PetscInt_FMT "\t(%" PetscInt_FMT ")\n", (double)t, (double)sum, (double)Enorm, (double)lgEnorm, (double)Emax, (double)lgEmax, (double)chargesum, (double)pmoments[0], (double)pmoments[1], (double)pmoments[1 + dim], (double)PetscSqrtReal(intESq), gNp, step));
    PetscCall(DMViewFromOptions(sw, NULL, "-sw_efield_view"));
  }

  // Compute decay rate and frequency
  PetscCall(PetscDrawLGGetData(user->drawlgE, NULL, &user->emaxCtx.e, &user->emaxCtx.t, &user->emaxCtx.Emax));
  if (!rank && !(user->emaxCtx.e % user->emaxCtx.per)) {
    Tao          tao;
    Mat          J;
    Vec          x, r;
    PetscScalar *a;
    PetscBool    fitLog = PETSC_TRUE, debug = PETSC_FALSE;

    PetscCall(TaoCreate(PETSC_COMM_SELF, &tao));
    PetscCall(TaoSetOptionsPrefix(tao, "emax_"));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, 4, &x));
    PetscCall(TaoSetSolution(tao, x));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, user->emaxCtx.e - user->emaxCtx.s, &r));
    if (fitLog) PetscCall(TaoSetResidualRoutine(tao, r, ComputeLogEmaxResidual, &user->emaxCtx));
    else PetscCall(TaoSetResidualRoutine(tao, r, ComputeEmaxResidual, &user->emaxCtx));
    PetscCall(VecDestroy(&r));
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, user->emaxCtx.e - user->emaxCtx.s, 4, NULL, &J));
    if (fitLog) PetscCall(TaoSetJacobianResidualRoutine(tao, J, J, ComputeLogEmaxJacobian, &user->emaxCtx));
    else PetscCall(TaoSetJacobianResidualRoutine(tao, J, J, ComputeEmaxJacobian, &user->emaxCtx));
    PetscCall(MatDestroy(&J));
    PetscCall(TaoSetFromOptions(tao));
    PetscCall(VecGetArray(x, &a));
    a[0] = 0.02;
    a[1] = 0.15;
    a[2] = 1.4;
    a[3] = 0.45;
    PetscCall(VecRestoreArray(x, &a));
    PetscCall(TaoSolve(tao));
    if (debug) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "t = ["));
      for (PetscInt i = 0; i < user->emaxCtx.e; ++i) {
        if (i > 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, ", "));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "%g", user->emaxCtx.t[i]));
      }
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "]\n"));
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "Emax = ["));
      for (PetscInt i = 0; i < user->emaxCtx.e; ++i) {
        if (i > 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, ", "));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "%g", user->emaxCtx.Emax[i]));
      }
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "]\n"));
    }
    PetscDraw     draw;
    PetscDrawAxis axis;
    char          title[PETSC_MAX_PATH_LEN];

    PetscCall(VecGetArray(x, &a));
    user->gamma = a[1];
    user->omega = a[2];
    if (user->efield_monitor == E_MONITOR_FULL) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "Emax Fit: gamma %g omega %g C %g phi %g\n", a[1], a[2], a[0], a[3]));
      PetscCall(PetscDrawLGGetDraw(user->drawlgE, &draw));
      PetscCall(PetscSNPrintf(title, PETSC_MAX_PATH_LEN, "Max Electric Field gamma %.4g omega %.4g", a[1], a[2]));
      PetscCall(PetscDrawSetTitle(draw, title));
      PetscCall(PetscDrawLGGetAxis(user->drawlgE, &axis));
      PetscCall(PetscDrawAxisSetLabels(axis, title, "time", "E_max"));
    }
    PetscCall(VecRestoreArray(x, &a));
    PetscCall(VecDestroy(&x));
    PetscCall(TaoDestroy(&tao));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MonitorMoments(TS ts, PetscInt step, PetscReal t, Vec U, void *ctx)
{
  AppCtx   *user = (AppCtx *)ctx;
  DM        sw;
  PetscReal pmoments[4], fmoments[4]; /* \int f, \int v f, \int v^2 f */

  PetscFunctionBeginUser;
  if (step < 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(TSGetDM(ts, &sw));

  PetscCall(DMSwarmComputeMoments(sw, "velocity", "w_q", pmoments));
  PetscCall(computeVelocityFEMMoments(sw, fmoments, user));

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%f\t%f\t%f\t%f\t%f\t%f\t%f\n", (double)t, (double)pmoments[0], (double)pmoments[1], (double)pmoments[3], (double)fmoments[0], (double)fmoments[1], (double)fmoments[2]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  u[0] = 0.0;
  return PETSC_SUCCESS;
}

/*
    M_p w_p
      - Make M_p with "moments"
      - Get w_p from Swarm
    M_p v_p w_p
      - Get v_p from Swarm
      - pointwise multiply v_p and w_p
    M_p (v_p - (\sum_j p_F \phi_j(x_p)) / m (\sum_k n_F \phi_k(x_p)))^2 w_p
      - ProjectField(sw, {n, p} U, {v_p} A, tmp_p)
      - pointwise multiply tmp_p and w_p

  Projection works fpr swarms
    Fields are FE from the CellDM, and aux fields are the swarm fields
*/
static PetscErrorCode ComputeMomentFields(TS ts)
{
  AppCtx   *user;
  DM        sw;
  KSP       ksp;
  Mat       M_p, D_p;
  Vec       f, v, E, tmpMom;
  Vec       m, mold, mfluxold, mres, n, nrhs, nflux, nres, p, prhs, pflux, pres, e, erhs, eflux, eres;
  PetscReal dt, t;
  PetscInt  Nts;

  PetscFunctionBegin;
  PetscCall(TSGetStepNumber(ts, &Nts));
  PetscCall(TSGetTimeStep(ts, &dt));
  PetscCall(TSGetTime(ts, &t));
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetApplicationContext(sw, (void *)&user));
  PetscCall(DMSwarmSetCellDMActive(sw, "moment fields"));
  PetscCall(DMSwarmMigrate(sw, PETSC_FALSE));
  // TODO In higher dimensions, we will have to create different M_p and D_p for each field
  PetscCall(DMCreateMassMatrix(sw, user->dmN, &M_p));
  PetscCall(DMCreateGradientMatrix(sw, user->dmN, &D_p));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &f));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "velocity", &v));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "E_field", &E));
  PetscCall(PetscObjectSetName((PetscObject)f, "particle weight"));

  PetscCall(MatViewFromOptions(user->MN, NULL, "-mn_view"));
  PetscCall(MatViewFromOptions(user->MP, NULL, "-mp_view"));
  PetscCall(MatViewFromOptions(user->ME, NULL, "-me_view"));
  PetscCall(VecViewFromOptions(f, NULL, "-weights_view"));

  PetscCall(DMGetGlobalVector(user->dmN, &nrhs));
  PetscCall(DMGetGlobalVector(user->dmN, &nflux));
  PetscCall(PetscObjectSetName((PetscObject)nrhs, "Weak number density"));
  PetscCall(DMGetNamedGlobalVector(user->dmN, "n", &n));
  PetscCall(DMGetGlobalVector(user->dmP, &prhs));
  PetscCall(DMGetGlobalVector(user->dmP, &pflux));
  PetscCall(PetscObjectSetName((PetscObject)prhs, "Weak momentum density"));
  PetscCall(DMGetNamedGlobalVector(user->dmP, "p", &p));
  PetscCall(DMGetGlobalVector(user->dmE, &erhs));
  PetscCall(DMGetGlobalVector(user->dmE, &eflux));
  PetscCall(PetscObjectSetName((PetscObject)erhs, "Weak energy density (pressure)"));
  PetscCall(DMGetNamedGlobalVector(user->dmE, "e", &e));

  // Compute moments and fluxes
  PetscCall(VecDuplicate(f, &tmpMom));

  PetscCall(MatMultTranspose(M_p, f, nrhs));

  PetscCall(VecPointwiseMult(tmpMom, f, v));
  PetscCall(MatMultTranspose(M_p, tmpMom, prhs));
  PetscCall(MatMultTranspose(D_p, tmpMom, nflux));

  PetscCall(VecPointwiseMult(tmpMom, tmpMom, v));
  PetscCall(MatMultTranspose(M_p, tmpMom, erhs));
  PetscCall(MatMultTranspose(D_p, tmpMom, pflux));

  PetscCall(VecPointwiseMult(tmpMom, tmpMom, v));
  PetscCall(MatMultTranspose(D_p, tmpMom, eflux));

  PetscCall(VecPointwiseMult(tmpMom, f, E));
  PetscCall(MatMultTransposeAdd(M_p, tmpMom, pflux, pflux));

  PetscCall(VecPointwiseMult(tmpMom, v, E));
  PetscCall(VecScale(tmpMom, 2.));
  PetscCall(MatMultTransposeAdd(M_p, tmpMom, eflux, eflux));

  PetscCall(VecDestroy(&tmpMom));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "velocity", &v));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &f));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "E_field", &E));

  PetscCall(MatDestroy(&M_p));
  PetscCall(MatDestroy(&D_p));

  PetscCall(KSPCreate(PetscObjectComm((PetscObject)sw), &ksp));
  PetscCall(KSPSetOptionsPrefix(ksp, "mom_proj_"));
  PetscCall(KSPSetOperators(ksp, user->MN, user->MN));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, nrhs, n));
  PetscCall(KSPSetOperators(ksp, user->MP, user->MP));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, prhs, p));
  PetscCall(KSPSetOperators(ksp, user->ME, user->ME));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, erhs, e));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(DMRestoreGlobalVector(user->dmN, &nrhs));
  PetscCall(DMRestoreGlobalVector(user->dmP, &prhs));
  PetscCall(DMRestoreGlobalVector(user->dmE, &erhs));

  // Check moment residual
  // TODO Fix global2local here
  PetscReal res[3], logres[3];

  PetscCall(DMGetGlobalVector(user->dmMom, &m));
  PetscCall(VecISCopy(m, user->isN, SCATTER_FORWARD, n));
  PetscCall(VecISCopy(m, user->isP, SCATTER_FORWARD, p));
  PetscCall(VecISCopy(m, user->isE, SCATTER_FORWARD, e));
  PetscCall(DMGetNamedGlobalVector(user->dmMom, "mold", &mold));
  PetscCall(DMGetNamedGlobalVector(user->dmMom, "mfluxold", &mfluxold));
  if (!Nts) goto end;

  // e = \Tr{\tau}
  // M_p w^{k+1} - M_p w^k - \Delta t D_p (w^k \vb{v}^k) = 0
  // M_p \vb{p}^{k+1} - M_p \vb{p}^k - \Delta t D_p \tau - e \Delta t M_p \left( n \vb{E} \right) = 0
  // M_p e^{k+1} - M_p e^k - \Delta t D_p \vb{Q} - 2 e \Delta t M_p \left( \vb{p} \cdot \vb{E} \right) = 0
  PetscCall(DMGetGlobalVector(user->dmMom, &mres));
  PetscCall(VecCopy(mfluxold, mres));
  PetscCall(VecAXPBYPCZ(mres, 1. / dt, -1. / dt, -1., m, mold));

  PetscCall(DMGetNamedGlobalVector(user->dmN, "nres", &nres));
  PetscCall(DMGetNamedGlobalVector(user->dmP, "pres", &pres));
  PetscCall(DMGetNamedGlobalVector(user->dmE, "eres", &eres));
  PetscCall(VecISCopy(mres, user->isN, SCATTER_REVERSE, nres));
  PetscCall(VecISCopy(mres, user->isP, SCATTER_REVERSE, pres));
  PetscCall(VecISCopy(mres, user->isE, SCATTER_REVERSE, eres));
  PetscCall(VecNorm(nres, NORM_2, &res[0]));
  PetscCall(VecNorm(pres, NORM_2, &res[1]));
  PetscCall(VecNorm(eres, NORM_2, &res[2]));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)sw), "Mass Residual: %g\n", (double)res[0]));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)sw), "Momentum Residual: %g\n", (double)res[1]));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)sw), "Energy Residual: %g\n", (double)res[2]));
  PetscCall(DMRestoreNamedGlobalVector(user->dmN, "nres", &nres));
  PetscCall(DMRestoreNamedGlobalVector(user->dmP, "pres", &pres));
  PetscCall(DMRestoreNamedGlobalVector(user->dmE, "eres", &eres));
  PetscCall(DMRestoreGlobalVector(user->dmMom, &mres));

  for (PetscInt i = 0; i < 3; ++i) logres[i] = PetscLog10Real(res[i]);
  PetscCall(PetscDrawLGAddCommonPoint(user->drawlgMomRes, t, logres));
  PetscCall(PetscDrawLGDraw(user->drawlgMomRes));
  {
    PetscDraw draw;

    PetscCall(PetscDrawLGGetDraw(user->drawlgMomRes, &draw));
    PetscCall(PetscDrawSave(draw));
  }

end:
  PetscCall(VecCopy(m, mold));
  PetscCall(DMRestoreGlobalVector(user->dmMom, &m));
  PetscCall(DMRestoreNamedGlobalVector(user->dmMom, "mold", &mold));
  PetscCall(VecISCopy(mfluxold, user->isN, SCATTER_FORWARD, nflux));
  PetscCall(VecISCopy(mfluxold, user->isP, SCATTER_FORWARD, pflux));
  PetscCall(VecISCopy(mfluxold, user->isE, SCATTER_FORWARD, eflux));
  PetscCall(DMRestoreNamedGlobalVector(user->dmMom, "mfluxold", &mfluxold));

  PetscCall(DMRestoreGlobalVector(user->dmN, &nflux));
  PetscCall(DMRestoreGlobalVector(user->dmP, &pflux));
  PetscCall(DMRestoreGlobalVector(user->dmE, &eflux));
  PetscCall(DMRestoreNamedGlobalVector(user->dmN, "n", &n));
  PetscCall(DMRestoreNamedGlobalVector(user->dmP, "p", &p));
  PetscCall(DMRestoreNamedGlobalVector(user->dmE, "e", &e));
  PetscCall(DMSwarmSetCellDMActive(sw, "space"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MonitorMomentFields(TS ts, PetscInt step, PetscReal t, Vec U, void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  Vec     n, p, e;
  Vec     nres, pres, eres;

  PetscFunctionBeginUser;
  if (step < 0) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(ComputeMomentFields(ts));

  PetscCall(DMGetNamedGlobalVector(user->dmN, "n", &n));
  PetscCall(VecView(n, user->viewerN));
  PetscCall(DMRestoreNamedGlobalVector(user->dmN, "n", &n));

  PetscCall(DMGetNamedGlobalVector(user->dmP, "p", &p));
  PetscCall(VecView(p, user->viewerP));
  PetscCall(DMRestoreNamedGlobalVector(user->dmP, "p", &p));

  PetscCall(DMGetNamedGlobalVector(user->dmE, "e", &e));
  PetscCall(VecView(e, user->viewerE));
  PetscCall(DMRestoreNamedGlobalVector(user->dmE, "e", &e));

  PetscCall(DMGetNamedGlobalVector(user->dmN, "nres", &nres));
  PetscCall(VecView(nres, user->viewerNRes));
  PetscCall(DMRestoreNamedGlobalVector(user->dmN, "nres", &nres));

  PetscCall(DMGetNamedGlobalVector(user->dmP, "pres", &pres));
  PetscCall(VecView(pres, user->viewerPRes));
  PetscCall(DMRestoreNamedGlobalVector(user->dmP, "pres", &pres));

  PetscCall(DMGetNamedGlobalVector(user->dmE, "eres", &eres));
  PetscCall(VecView(eres, user->viewerERes));
  PetscCall(DMRestoreNamedGlobalVector(user->dmE, "eres", &eres));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MonitorInitialConditions(TS ts, PetscInt step, PetscReal t, Vec U, void *ctx)
{
  AppCtx    *user = (AppCtx *)ctx;
  DM         sw;
  PetscDraw  drawic_x, drawic_v;
  PetscReal *weight, *pos, *vel;
  PetscInt   dim, Np;

  PetscFunctionBegin;
  if (step < 0) PetscFunctionReturn(PETSC_SUCCESS); /* -1 indicates interpolated solution */
  if (step == 0) {
    PetscCall(TSGetDM(ts, &sw));
    PetscCall(DMGetDimension(sw, &dim));
    PetscCall(DMSwarmGetLocalSize(sw, &Np));

    PetscCall(PetscDrawHGReset(user->drawhgic_x));
    PetscCall(PetscDrawHGGetDraw(user->drawhgic_x, &drawic_x));
    PetscCall(PetscDrawClear(drawic_x));
    PetscCall(PetscDrawFlush(drawic_x));

    PetscCall(PetscDrawHGReset(user->drawhgic_v));
    PetscCall(PetscDrawHGGetDraw(user->drawhgic_v, &drawic_v));
    PetscCall(PetscDrawClear(drawic_v));
    PetscCall(PetscDrawFlush(drawic_v));

    PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&pos));
    PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&vel));
    PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weight));
    for (PetscInt p = 0; p < Np; ++p) {
      PetscCall(PetscDrawHGAddWeightedValue(user->drawhgic_x, pos[p * dim], weight[p]));
      PetscCall(PetscDrawHGAddWeightedValue(user->drawhgic_v, vel[p * dim], weight[p]));
    }
    PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&pos));
    PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&vel));
    PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weight));

    PetscCall(PetscDrawHGDraw(user->drawhgic_x));
    PetscCall(PetscDrawHGSave(user->drawhgic_x));
    PetscCall(PetscDrawHGDraw(user->drawhgic_v));
    PetscCall(PetscDrawHGSave(user->drawhgic_v));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Right now, make the complete velocity histogram
PetscErrorCode MonitorVelocity(TS ts, PetscInt step, PetscReal t, Vec U, void *ctx)
{
  AppCtx      *user = (AppCtx *)ctx;
  DM           sw, dm;
  Vec          ks;
  PetscProbFn *cdf;
  PetscDraw    drawcell_v;
  PetscScalar *ksa;
  PetscReal   *weight, *vel;
  PetscInt    *pidx;
  PetscInt     dim, Npc, cStart, cEnd, cell = user->velocity_monitor;

  PetscFunctionBegin;
  PetscCall(TSGetDM(ts, &sw));
  PetscCall(DMGetDimension(sw, &dim));

  PetscCall(DMSwarmGetCellDM(sw, &dm));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(VecCreate(PetscObjectComm((PetscObject)dm), &ks));
  PetscCall(PetscObjectSetName((PetscObject)ks, "KS Statistic by Cell"));
  PetscCall(VecSetSizes(ks, cEnd - cStart, PETSC_DETERMINE));
  PetscCall(VecSetFromOptions(ks));
  switch (dim) {
  case 1:
    //cdf = PetscCDFMaxwellBoltzmann1D;
    cdf = PetscCDFGaussian1D;
    break;
  case 2:
    cdf = PetscCDFMaxwellBoltzmann2D;
    break;
  case 3:
    cdf = PetscCDFMaxwellBoltzmann3D;
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_ARG_OUTOFRANGE, "Dimension %" PetscInt_FMT " not supported", dim);
  }

  PetscCall(PetscDrawHGReset(user->drawhgcell_v));
  PetscCall(PetscDrawHGGetDraw(user->drawhgcell_v, &drawcell_v));
  PetscCall(PetscDrawClear(drawcell_v));
  PetscCall(PetscDrawFlush(drawcell_v));

  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&vel));
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weight));
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(VecGetArrayWrite(ks, &ksa));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    Vec          cellv, cellw;
    PetscScalar *cella, *cellaw;
    PetscReal    totWgt = 0.;

    PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Npc, &pidx));
    PetscCall(VecCreate(PETSC_COMM_SELF, &cellv));
    PetscCall(VecSetBlockSize(cellv, dim));
    PetscCall(VecSetSizes(cellv, Npc * dim, Npc));
    PetscCall(VecSetFromOptions(cellv));
    PetscCall(VecCreate(PETSC_COMM_SELF, &cellw));
    PetscCall(VecSetSizes(cellw, Npc, Npc));
    PetscCall(VecSetFromOptions(cellw));
    PetscCall(VecGetArrayWrite(cellv, &cella));
    PetscCall(VecGetArrayWrite(cellw, &cellaw));
    for (PetscInt q = 0; q < Npc; ++q) {
      const PetscInt p = pidx[q];
      if (c == cell) PetscCall(PetscDrawHGAddWeightedValue(user->drawhgcell_v, vel[p * dim], weight[p]));
      for (PetscInt d = 0; d < dim; ++d) cella[q * dim + d] = vel[p * dim + d];
      cellaw[q] = weight[p];
      totWgt += weight[p];
    }
    PetscCall(VecRestoreArrayWrite(cellv, &cella));
    PetscCall(VecRestoreArrayWrite(cellw, &cellaw));
    PetscCall(VecScale(cellw, 1. / totWgt));
    PetscCall(PetscProbComputeKSStatisticWeighted(cellv, cellw, cdf, &ksa[c - cStart]));
    PetscCall(VecDestroy(&cellv));
    PetscCall(VecDestroy(&cellw));
    PetscCall(DMSwarmSortRestorePointsPerCell(sw, c, &Npc, &pidx));
  }
  PetscCall(VecRestoreArrayWrite(ks, &ksa));
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&vel));
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weight));
  PetscCall(DMSwarmSortRestoreAccess(sw));

  PetscReal minalpha, maxalpha;
  PetscInt  mincell, maxcell;

  PetscCall(VecFilter(ks, PETSC_SMALL));
  PetscCall(VecMin(ks, &mincell, &minalpha));
  PetscCall(VecMax(ks, &maxcell, &maxalpha));
  PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "Step %" PetscInt_FMT ": Min/Max KS statistic %g/%g in cell %" PetscInt_FMT "/%" PetscInt_FMT "\n", step, minalpha, maxalpha, mincell, maxcell));
  PetscCall(VecViewFromOptions(ks, NULL, "-ks_view"));
  PetscCall(VecDestroy(&ks));

  PetscCall(PetscDrawHGDraw(user->drawhgcell_v));
  PetscCall(PetscDrawHGSave(user->drawhgcell_v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MonitorPositions_2D(TS ts, PetscInt step, PetscReal t, Vec U, void *ctx)
{
  AppCtx         *user = (AppCtx *)ctx;
  DM              dm, sw;
  PetscDrawAxis   axis;
  char            title[1024];
  PetscScalar    *x, *v, *weight;
  PetscReal       lower[3], upper[3], speed;
  const PetscInt *s;
  PetscInt        dim, cStart, cEnd, c;

  PetscFunctionBeginUser;
  if (step > 0 && step % user->ostep == 0) {
    PetscCall(TSGetDM(ts, &sw));
    PetscCall(DMSwarmGetCellDM(sw, &dm));
    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(DMGetBoundingBox(dm, lower, upper));
    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
    PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&v));
    PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weight));
    PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void **)&s));
    PetscCall(DMSwarmSortGetAccess(sw));
    PetscCall(PetscDrawSPReset(user->drawspX));
    PetscCall(PetscDrawSPGetAxis(user->drawspX, &axis));
    PetscCall(PetscSNPrintf(title, 1024, "Step %" PetscInt_FMT " Time: %g", step, (double)t));
    PetscCall(PetscDrawAxisSetLabels(axis, title, "x", "v"));
    PetscCall(PetscDrawSPSetLimits(user->drawspX, lower[0], upper[0], lower[1], upper[1]));
    PetscCall(PetscDrawSPSetLimits(user->drawspX, lower[0], upper[0], -12, 12));
    for (c = 0; c < cEnd - cStart; ++c) {
      PetscInt *pidx, Npc, q;
      PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Npc, &pidx));
      for (q = 0; q < Npc; ++q) {
        const PetscInt p = pidx[q];
        if (s[p] == 0) {
          speed = 0.;
          for (PetscInt d = 0; d < dim; ++d) speed += PetscSqr(v[p * dim + d]);
          speed = PetscSqrtReal(speed);
          if (dim == 1) {
            PetscCall(PetscDrawSPAddPointColorized(user->drawspX, &x[p * dim], &v[p * dim], &speed));
          } else {
            PetscCall(PetscDrawSPAddPointColorized(user->drawspX, &x[p * dim], &x[p * dim + 1], &speed));
          }
        } else if (s[p] == 1) {
          PetscCall(PetscDrawSPAddPoint(user->drawspX, &x[p * dim], &v[p * dim]));
        }
      }
      PetscCall(DMSwarmSortRestorePointsPerCell(sw, c, &Npc, &pidx));
    }
    PetscCall(PetscDrawSPDraw(user->drawspX, PETSC_TRUE));
    PetscDraw draw;
    PetscCall(PetscDrawSPGetDraw(user->drawspX, &draw));
    PetscCall(PetscDrawSave(draw));
    PetscCall(DMSwarmSortRestoreAccess(sw));
    PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
    PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weight));
    PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&v));
    PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **)&s));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MonitorPoisson(TS ts, PetscInt step, PetscReal t, Vec U, void *ctx)
{
  AppCtx *user = (AppCtx *)ctx;
  DM      dm, sw;

  PetscFunctionBeginUser;
  if (step > 0 && step % user->ostep == 0) {
    PetscCall(TSGetDM(ts, &sw));
    PetscCall(DMSwarmGetCellDM(sw, &dm));

    if (user->validE) {
      PetscScalar *x, *E, *weight;
      PetscReal    lower[3], upper[3], xval;
      PetscDraw    draw;
      PetscInt     dim, cStart, cEnd;

      PetscCall(DMGetDimension(dm, &dim));
      PetscCall(DMGetBoundingBox(dm, lower, upper));
      PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));

      PetscCall(PetscDrawSPReset(user->drawspE));
      PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
      PetscCall(DMSwarmGetField(sw, "E_field", NULL, NULL, (void **)&E));
      PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&weight));

      PetscCall(DMSwarmSortGetAccess(sw));
      for (PetscInt c = 0; c < cEnd - cStart; ++c) {
        PetscReal Eavg = 0.0;
        PetscInt *pidx, Npc;

        PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Npc, &pidx));
        for (PetscInt q = 0; q < Npc; ++q) {
          const PetscInt p = pidx[q];
          Eavg += E[p * dim];
        }
        Eavg /= Npc;
        xval = (c + 0.5) * ((upper[0] - lower[0]) / (cEnd - cStart));
        PetscCall(PetscDrawSPAddPoint(user->drawspE, &xval, &Eavg));
        PetscCall(DMSwarmSortRestorePointsPerCell(sw, c, &Npc, &pidx));
      }
      PetscCall(PetscDrawSPDraw(user->drawspE, PETSC_TRUE));
      PetscCall(PetscDrawSPGetDraw(user->drawspE, &draw));
      PetscCall(PetscDrawSave(draw));
      PetscCall(DMSwarmSortRestoreAccess(sw));
      PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&x));
      PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&weight));
      PetscCall(DMSwarmRestoreField(sw, "E_field", NULL, NULL, (void **)&E));
    }

    Vec rho, rhohat, phi;

    PetscCall(DMGetNamedGlobalVector(user->dmPot, "rho", &rho));
    PetscCall(DMGetNamedGlobalVector(user->dmPot, "rhohat", &rhohat));
    PetscCall(VecView(rho, user->viewerRho));
    PetscCall(VecISCopy(user->fftX, user->fftReal, SCATTER_FORWARD, rho));
    PetscCall(MatMult(user->fftPot, user->fftX, user->fftY));
    PetscCall(VecFilter(user->fftY, PETSC_SMALL));
    PetscCall(VecViewFromOptions(user->fftX, NULL, "-real_view"));
    PetscCall(VecViewFromOptions(user->fftY, NULL, "-fft_view"));
    PetscCall(VecISCopy(user->fftY, user->fftReal, SCATTER_REVERSE, rhohat));
    PetscCall(VecSetValue(rhohat, 0, 0., INSERT_VALUES)); // Remove large DC component
    PetscCall(VecView(rhohat, user->viewerRhoHat));
    PetscCall(DMRestoreNamedGlobalVector(user->dmPot, "rho", &rho));
    PetscCall(DMRestoreNamedGlobalVector(user->dmPot, "rhohat", &rhohat));

    PetscCall(DMGetNamedGlobalVector(user->dmPot, "phi", &phi));
    PetscCall(VecView(phi, user->viewerPhi));
    PetscCall(DMRestoreNamedGlobalVector(user->dmPot, "phi", &phi));
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
  PetscCall(PetscBagRegisterScalar(bag, &p->v0, 1.0, "phi0", "Potential scale, kg*m^2/A*s^3"));
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
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
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
  PetscCall(PetscMalloc1(2, &pt));
  wt[0] = 1.;
  pt[0] = -1.;
  pt[1] = -1.;
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

/*
   /  I   -grad\ / q \ = /0\
   \-div    0  / \phi/   \f/
*/
static void f0_q(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  for (PetscInt d = 0; d < dim; ++d) f0[d] += u[uOff[0] + d];
}

static void f1_q(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f1[])
{
  for (PetscInt d = 0; d < dim; ++d) f1[d * dim + d] = u[uOff[1]];
}

static void f0_phi_backgroundCharge(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  f0[0] += constants[SIGMA];
  for (PetscInt d = 0; d < dim; ++d) f0[0] += u_x[uOff_x[0] + d * dim + d];
}

/* Boundary residual. Dirichlet boundary for u means u_bdy=p*n */
static void g0_qq(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g0[])
{
  for (PetscInt d = 0; d < dim; ++d) g0[d * dim + d] = 1.0;
}

static void g2_qphi(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g2[])
{
  for (PetscInt d = 0; d < dim; ++d) g2[d * dim + d] = 1.0;
}

static void g1_phiq(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, PetscReal u_tShift, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar g1[])
{
  for (PetscInt d = 0; d < dim; ++d) g1[d * dim + d] = 1.0;
}

static PetscErrorCode CreateFEM(DM dm, AppCtx *user)
{
  PetscFE   fephi, feq;
  PetscDS   ds;
  PetscBool simplex;
  PetscInt  dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexIsSimplex(dm, &simplex));
  if (user->em == EM_MIXED) {
    DMLabel        label;
    const PetscInt id = 1;

    PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, dim, simplex, "field_", PETSC_DETERMINE, &feq));
    PetscCall(PetscObjectSetName((PetscObject)feq, "field"));
    PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, simplex, "potential_", PETSC_DETERMINE, &fephi));
    PetscCall(PetscObjectSetName((PetscObject)fephi, "potential"));
    PetscCall(PetscFECopyQuadrature(feq, fephi));
    PetscCall(DMSetField(dm, 0, NULL, (PetscObject)feq));
    PetscCall(DMSetField(dm, 1, NULL, (PetscObject)fephi));
    PetscCall(DMCreateDS(dm));
    PetscCall(PetscFEDestroy(&fephi));
    PetscCall(PetscFEDestroy(&feq));

    PetscCall(DMGetLabel(dm, "marker", &label));
    PetscCall(DMGetDS(dm, &ds));

    PetscCall(PetscDSSetResidual(ds, 0, f0_q, f1_q));
    PetscCall(PetscDSSetResidual(ds, 1, f0_phi_backgroundCharge, NULL));
    PetscCall(PetscDSSetJacobian(ds, 0, 0, g0_qq, NULL, NULL, NULL));
    PetscCall(PetscDSSetJacobian(ds, 0, 1, NULL, NULL, g2_qphi, NULL));
    PetscCall(PetscDSSetJacobian(ds, 1, 0, NULL, g1_phiq, NULL, NULL));

    PetscCall(DMAddBoundary(dm, DM_BC_ESSENTIAL, "wall", label, 1, &id, 0, 0, NULL, (PetscVoidFn *)zero, NULL, NULL, NULL));

  } else {
    MatNullSpace nullsp;
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
  PetscCall(DMPlexSetSNESLocalFEM(dm, PETSC_FALSE, user));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(DMCreateMatrix(dm, &J));
  PetscCall(MatNullSpaceCreate(PetscObjectComm((PetscObject)dm), PETSC_TRUE, 0, NULL, &nullSpace));
  PetscCall(MatSetNullSpace(J, nullSpace));
  PetscCall(MatNullSpaceDestroy(&nullSpace));
  PetscCall(SNESSetJacobian(snes, J, J, NULL, NULL));
  PetscCall(MatDestroy(&J));
  if (user->em == EM_MIXED) {
    const PetscInt potential = 1;

    PetscCall(DMCreateSubDM(dm, 1, &potential, &user->isPot, &user->dmPot));
  } else {
    user->dmPot = dm;
    PetscCall(PetscObjectReference((PetscObject)user->dmPot));
  }
  PetscCall(DMCreateMassMatrix(user->dmPot, user->dmPot, &user->M));
  PetscCall(DMPlexCreateClosureIndex(dm, NULL));
  user->snes = snes;
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Conservation of mass (m = 1.0)
// n_t + 1/ m p_x = 0
static void f0_mass(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  for (PetscInt d = 0; d < dim; ++d) f0[0] += u_t[uOff[0]] + u_x[uOff_x[1] + d * dim + d];
}

// Conservation of momentum (m = 1, e = 1)
// p_t + (u p)_x = -pr_x + e n E
// p_t + (div u) p + u . grad p = -pr_x + e n E
// p_t + (div p) p / n - (p . grad n) p / n^2 + p / n . grad p = -pr_x + e n E
static void f0_momentum(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscScalar n = u[uOff[0]];

  for (PetscInt d = 0; d < dim; ++d) {
    PetscReal divp = 0.;

    f0[d] += u_t[uOff[1] + d];
    for (PetscInt e = 0; e < dim; ++e) {
      f0[d] += u[uOff[1] + e] * u_x[uOff_x[1] + d * dim + e] / n;                    // p / n . grad p
      f0[d] -= (u[uOff[1] + e] * u_x[uOff_x[0] + e]) * u[uOff[1] + d] / PetscSqr(n); // -(p . grad n) p / n^2
      divp += u_x[uOff_x[1] + e * dim + e];
    }
    f0[d] += divp * u[uOff[1] + d] / n; // (div p) p / n
    f0[d] += u_x[uOff_x[2] + d];        // pr_x
    f0[d] -= n * a[d];                  // -e n E
  }
}

// Conservation of energy
// pr_t + (u pr)_x = -3 pr u_x - q_x
// pr_t + (div u) pr + u . grad pr = -3 pr (div u) - q_x
// pr_t + 4 (div u) pr + u . grad pr = -q_x
// pr_t + 4 div p pr / n - 4 (p . grad n) pr / n^2 + p . grad pr / n = -q_x
static void f0_energy(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  const PetscScalar n    = u[uOff[0]];
  const PetscScalar pr   = u[uOff[2]];
  PetscReal         divp = 0.;

  f0[0] += u_t[uOff[2]];
  for (PetscInt d = 0; d < dim; ++d) {
    f0[0] += u[uOff[1] + d] * u_x[uOff_x[2] + d] / n;                     // p . grad pr / n
    f0[0] -= 4. * u[uOff[1] + d] * u_x[uOff_x[0] + d] * pr / PetscSqr(n); // -4 (p . grad n) pr / n^2
    divp += u_x[uOff_x[1] + d * dim + d];
  }
  f0[0] += 4. * divp * pr / n; // 4 div p pr / n
}

static PetscErrorCode SetupMomentProblem(DM dm, AppCtx *ctx)
{
  PetscDS ds;

  PetscFunctionBegin;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSSetResidual(ds, 0, f0_mass, NULL));
  PetscCall(PetscDSSetResidual(ds, 1, f0_momentum, NULL));
  PetscCall(PetscDSSetResidual(ds, 2, f0_energy, NULL));
  //PetscCall(PetscDSSetJacobian(ds, 0, 0, g0_mass_uu, NULL, NULL, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMomentFields(DM odm, AppCtx *user)
{
  DM             dm;
  PetscFE        fe;
  DMPolytopeType ct;
  PetscInt       dim, cStart;

  PetscFunctionBeginUser;
  PetscCall(DMClone(odm, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, NULL));
  PetscCall(DMPlexGetCellType(dm, cStart, &ct));
  PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, NULL, PETSC_DETERMINE, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "number density"));
  PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, dim, ct, NULL, PETSC_DETERMINE, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "momentum density"));
  PetscCall(DMSetField(dm, 1, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(PetscFECreateByCell(PETSC_COMM_SELF, dim, 1, ct, NULL, PETSC_DETERMINE, &fe));
  PetscCall(PetscObjectSetName((PetscObject)fe, "energy density"));
  PetscCall(DMSetField(dm, 2, NULL, (PetscObject)fe));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(DMCreateDS(dm));
  PetscCall(SetupMomentProblem(dm, user));
  user->dmMom = dm;
  PetscInt field;

  field = 0;
  PetscCall(DMCreateSubDM(user->dmMom, 1, &field, &user->isN, &user->dmN));
  PetscCall(DMCreateMassMatrix(user->dmN, user->dmN, &user->MN));
  field = 1;
  PetscCall(DMCreateSubDM(user->dmMom, 1, &field, &user->isP, &user->dmP));
  PetscCall(DMCreateMassMatrix(user->dmP, user->dmP, &user->MP));
  field = 2;
  PetscCall(DMCreateSubDM(user->dmMom, 1, &field, &user->isE, &user->dmE));
  PetscCall(DMCreateMassMatrix(user->dmE, user->dmE, &user->ME));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscPDFPertubedConstant2D(const PetscReal x[], const PetscReal dummy[], PetscReal p[])
{
  p[0] = (1 + 0.01 * PetscCosReal(0.5 * x[0])) / (2 * PETSC_PI);
  p[1] = (1 + 0.01 * PetscCosReal(0.5 * x[1])) / (2 * PETSC_PI);
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
  const PetscReal k     = scale ? scale[0] : 1.;
  p[0]                  = (1 + alpha * PetscCosReal(k * (x[0] + x[1])));
  return PETSC_SUCCESS;
}

static PetscErrorCode CreateVelocityDM(DM sw, DM *vdm)
{
  PetscFE        fe;
  DMPolytopeType ct;
  PetscInt       dim, cStart;
  const char    *prefix = "v";

  PetscFunctionBegin;
  PetscCall(DMCreate(PETSC_COMM_SELF, vdm));
  PetscCall(DMSetType(*vdm, DMPLEX));
  PetscCall(DMPlexSetOptionsPrefix(*vdm, prefix));
  PetscCall(DMSetFromOptions(*vdm));
  PetscCall(PetscObjectSetName((PetscObject)*vdm, "velocity"));
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
  for (PetscInt c = 0; c < xcEnd - xcStart; ++c) {
    const PetscInt cell = c + xcStart;
    PetscInt      *pidx, Npc;
    PetscReal      centroid[3], volume;

    PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Npc, &pidx));
    PetscCall(DMPlexComputeCellGeometryFVM(xdm, cell, &volume, centroid, NULL));
    for (PetscInt s = 0; s < Ns; ++s) {
      for (PetscInt q = 0; q < Npc / Ns; ++q) {
        const PetscInt p = pidx[q * Ns + s];

        for (PetscInt d = 0; d < dim; ++d) {
          x[p * dim + d] = centroid[d];
          v[p * dim + d] = vmin[0] + (q + 0.5) * ((vmax[0] - vmin[0]) / (Npc / Ns));
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
      // TODO: Fix 2 stream Ask Joe
      //   Two stream function from 1/2pi v^2 e^(-v^2/2)
      //   vw = 1. / (PetscSqrtReal(2 * PETSC_PI)) * (((coords_v[0] * PetscExpReal(-PetscSqr(coords_v[0]) / 2.)) - (coords_v[1] * PetscExpReal(-PetscSqr(coords_v[1]) / 2.)))) - 0.5 * PetscErfReal(coords_v[0] / PetscSqrtReal(2.)) + 0.5 * (PetscErfReal(coords_v[1] / PetscSqrtReal(2.)));
      vw = 0.5 * (PetscErfReal(vcoords[1] / PetscSqrtReal(2.)) - PetscErfReal(vcoords[0] / PetscSqrtReal(2.)));

      weight[p] = totalWeight * vw * xw;
      pwtot += weight[p];
      PetscCheck(weight[p] <= 10., PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Particle %" PetscInt_FMT " weight exceeded 1: %g, %g, %g", p, xw, vw, totalWeight);
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
  PetscCall(InitializeWeights(sw, user->totalWeight, dim == 1 ? PetscPDFCosine1D : PetscPDFCosine2D, scale));
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

static PetscErrorCode CreateSwarm(DM dm, AppCtx *user, DM *sw)
{
  DMSwarmCellDM celldm;
  DM            vdm;
  PetscReal     v0[2] = {1., 0.};
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

  DM mfdm;

  PetscCall(DMClone(dm, &mfdm));
  PetscCall(PetscObjectSetName((PetscObject)mfdm, "moment fields"));
  PetscCall(DMCopyDisc(dm, mfdm));
  PetscCall(DMSwarmCellDMCreate(mfdm, 1, &fieldnames[1], 1, fieldnames, &celldm));
  PetscCall(DMDestroy(&mfdm));
  PetscCall(DMSwarmAddCellDM(*sw, celldm));
  PetscCall(DMSwarmCellDMDestroy(&celldm));

  PetscCall(DMSetFromOptions(*sw));
  PetscCall(DMSetUp(*sw));

  PetscCall(DMSwarmSetCellDMActive(*sw, "space"));
  user->swarm = *sw;
  // TODO: This is redundant init since it is done in InitializeSolveAndSwarm, however DMSetUp() requires the local size be set
  if (user->perturbed_weights) {
    PetscCall(InitializeParticles_PerturbedWeights(*sw, user));
  } else {
    PetscCall(DMSwarmComputeLocalSizeFromOptions(*sw));
    PetscCall(DMSwarmInitializeCoordinates(*sw));
    PetscCall(DMSwarmInitializeVelocitiesFromOptions(*sw, v0));
  }
  PetscCall(PetscObjectSetName((PetscObject)*sw, "Particles"));
  PetscCall(DMViewFromOptions(*sw, NULL, "-sw_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode ComputeFieldAtParticles_Coulomb(SNES snes, DM sw, PetscReal E[])
{
  AppCtx     *user;
  PetscReal  *coords;
  PetscInt   *species, dim, Np, Ns;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)snes), &size));
  PetscCheck(size == 1, PetscObjectComm((PetscObject)snes), PETSC_ERR_SUP, "Coulomb code only works in serial");
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMSwarmGetNumSpecies(sw, &Ns));
  PetscCall(DMGetApplicationContext(sw, (void *)&user));

  PetscCall(DMSwarmGetField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
  PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void **)&species));
  for (PetscInt p = 0; p < Np; ++p) {
    PetscReal *pcoord = &coords[p * dim];
    PetscReal  pE[3]  = {0., 0., 0.};

    /* Calculate field at particle p due to particle q */
    for (PetscInt q = 0; q < Np; ++q) {
      PetscReal *qcoord = &coords[q * dim];
      PetscReal  rpq[3], r, r3, q_q;

      if (p == q) continue;
      q_q = user->charges[species[q]] * 1.;
      for (PetscInt d = 0; d < dim; ++d) rpq[d] = pcoord[d] - qcoord[d];
      r = DMPlex_NormD_Internal(dim, rpq);
      if (r < PETSC_SQRT_MACHINE_EPSILON) continue;
      r3 = PetscPowRealInt(r, 3);
      for (PetscInt d = 0; d < dim; ++d) pE[d] += q_q * rpq[d] / r3;
    }
    for (PetscInt d = 0; d < dim; ++d) E[p * dim + d] = pE[d];
  }
  PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **)&species));
  PetscCall(DMSwarmRestoreField(sw, DMSwarmPICField_coor, NULL, NULL, (void **)&coords));
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
  PetscCall(DMGetWorkArray(dm, maxNcp * dim, MPIU_REAL, &pcoord));
  // This can raise an FP_INEXACT in the dgemm inside
  PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF));
  PetscCall(PetscFECreateTabulation(fe, 1, maxNcp, refcoord, 1, &tab));
  PetscCall(PetscFPTrapPop());
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

static PetscErrorCode ComputeFieldAtParticles_Mixed(SNES snes, DM sw, Mat M_p, PetscReal E[])
{
  DM         dm;
  AppCtx    *user;
  PetscDS    ds;
  PetscFE    fe;
  KSP        ksp;
  Vec        rhoRhs, rhoRhsFull;   // Weak charge density, \int phi_i rho, and embedding in mixed problem
  Vec        rho;                  // Charge density, M^{-1} rhoRhs
  Vec        phi, locPhi, phiFull; // Potential and embedding in mixed problem
  Vec        f;                    // Particle weights
  PetscReal *coords;
  PetscInt   dim, cStart, cEnd, Np;

  PetscFunctionBegin;
  PetscCall(DMGetApplicationContext(sw, &user));
  PetscCall(PetscLogEventBegin(user->ESolveEvent, snes, sw, 0, 0));
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));

  PetscCall(SNESGetDM(snes, &dm));
  PetscCall(DMGetGlobalVector(user->dmPot, &rhoRhs));
  PetscCall(PetscObjectSetName((PetscObject)rhoRhs, "Weak charge density"));
  PetscCall(DMGetGlobalVector(dm, &rhoRhsFull));
  PetscCall(PetscObjectSetName((PetscObject)rhoRhsFull, "Weak charge density"));
  PetscCall(DMGetNamedGlobalVector(user->dmPot, "rho", &rho));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &f));
  PetscCall(PetscObjectSetName((PetscObject)f, "particle weight"));

  PetscCall(MatViewFromOptions(M_p, NULL, "-mp_view"));
  PetscCall(MatViewFromOptions(user->M, NULL, "-m_view"));
  PetscCall(VecViewFromOptions(f, NULL, "-weights_view"));

  PetscCall(MatMultTranspose(M_p, f, rhoRhs));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &f));

  PetscCall(KSPCreate(PetscObjectComm((PetscObject)dm), &ksp));
  PetscCall(KSPSetOptionsPrefix(ksp, "em_proj"));
  PetscCall(KSPSetOperators(ksp, user->M, user->M));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, rhoRhs, rho));

  PetscCall(VecISCopy(rhoRhsFull, user->isPot, SCATTER_FORWARD, rhoRhs));
  //PetscCall(VecScale(rhoRhsFull, -1.0));

  PetscCall(VecViewFromOptions(rhoRhs, NULL, "-rho_view"));
  PetscCall(VecViewFromOptions(rhoRhsFull, NULL, "-rho_full_view"));
  PetscCall(DMRestoreNamedGlobalVector(user->dmPot, "rho", &rho));
  PetscCall(DMRestoreGlobalVector(user->dmPot, &rhoRhs));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(DMGetGlobalVector(dm, &phiFull));
  PetscCall(DMGetNamedGlobalVector(user->dmPot, "phi", &phi));
  PetscCall(VecSet(phiFull, 0.0));
  PetscCall(SNESSolve(snes, rhoRhsFull, phiFull));
  PetscCall(DMRestoreGlobalVector(dm, &rhoRhsFull));
  PetscCall(VecViewFromOptions(phi, NULL, "-phi_view"));

  PetscCall(VecISCopy(phiFull, user->isPot, SCATTER_REVERSE, phi));
  PetscCall(DMRestoreNamedGlobalVector(user->dmPot, "phi", &phi));

  PetscCall(DMGetLocalVector(dm, &locPhi));
  PetscCall(DMGlobalToLocalBegin(dm, phiFull, INSERT_VALUES, locPhi));
  PetscCall(DMGlobalToLocalEnd(dm, phiFull, INSERT_VALUES, locPhi));
  PetscCall(DMRestoreGlobalVector(dm, &phiFull));
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
      const PetscInt p = points[cp];

      for (PetscInt d = 0; d < dim; ++d) E[p * dim + d] = 0.;
      PetscCall(PetscFEInterpolateAtPoints_Static(fe, tab, clPhi, chunkgeom, cp, &E[p * dim]));
      PetscCall(PetscFEPushforward(fe, chunkgeom, 1, &E[p * dim]));
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

  switch (user->em) {
  case EM_COULOMB:
    PetscCall(ComputeFieldAtParticles_Coulomb(snes, sw, E));
    break;
  case EM_PRIMAL:
    PetscCall(ComputeFieldAtParticles_Primal(snes, sw, M_p, E));
    break;
  case EM_MIXED:
    PetscCall(ComputeFieldAtParticles_Mixed(snes, sw, M_p, E));
    break;
  case EM_NONE:
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No solver for electrostatic model %s", EMTypes[user->em]);
  }
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
    PetscScalar vals[4] = {0., 1., -1., 0.};

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
      g[(p * 2 + 1) * dim + d] = m_p * u[(p * 2 + 1) * dim + d];
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
    PetscReal v0[2] = {1., 0.};
    if (user->perturbed_weights) {
      PetscCall(InitializeParticles_PerturbedWeights(sw, user));
    } else {
      PetscCall(DMSwarmComputeLocalSizeFromOptions(sw));
      PetscCall(DMSwarmInitializeCoordinates(sw));
      PetscCall(DMSwarmInitializeVelocitiesFromOptions(sw, v0));
    }
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
    PetscCheck(L, PetscObjectComm((PetscObject)cdm), PETSC_ERR_ARG_WRONG, "Mesh must be periodic");
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
  PetscInt step;

  PetscCall(TSGetStepNumber(ts, &step));
  if (!(step % ctx->remapFreq)) {
    // Monitor electric field before we destroy it
    PetscReal ptime;
    PetscInt  step;

    PetscCall(TSGetStepNumber(ts, &step));
    PetscCall(TSGetTime(ts, &ptime));
    if (ctx->efield_monitor) PetscCall(MonitorEField(ts, step, ptime, NULL, ctx));
    if (ctx->poisson_monitor) PetscCall(MonitorPoisson(ts, step, ptime, NULL, ctx));
    PetscCall(DMSwarmRemap(sw));
    ctx->validE = PETSC_FALSE;
  }
  // This MUST come last, since it recreates the subswarms and they must DMClone() the new swarm
  PetscCall(DMSwarmTSRedistribute(ts));
  PetscCall(InitializeSolveAndSwarm(ts, PETSC_FALSE));
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
  PetscCall(CreateMomentFields(dm, &user));
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
  if (user.moment_field_monitor) PetscCall(TSMonitorSet(ts, MonitorMomentFields, &user, NULL));
  if (user.initial_monitor) PetscCall(TSMonitorSet(ts, MonitorInitialConditions, &user, NULL));
  if (user.positions_monitor) PetscCall(TSMonitorSet(ts, MonitorPositions_2D, &user, NULL));
  if (user.poisson_monitor) PetscCall(TSMonitorSet(ts, MonitorPoisson, &user, NULL));
  if (user.velocity_monitor >= 0) PetscCall(TSMonitorSet(ts, MonitorVelocity, &user, NULL));

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

  if (user.checkLandau) {
    // We should get a lookup table based on charge density and \hat k
    const PetscReal gammaEx = -0.15336;
    const PetscReal omegaEx = 1.4156;
    const PetscReal tol     = 1e-2;

    PetscCheck(PetscAbsReal((user.gamma - gammaEx) / gammaEx) < tol, PETSC_COMM_WORLD, PETSC_ERR_LIB, "Invalid Landau gamma %g != %g", user.gamma, gammaEx);
    PetscCheck(PetscAbsReal((user.omega - omegaEx) / omegaEx) < tol, PETSC_COMM_WORLD, PETSC_ERR_LIB, "Invalid Landau omega %g != %g", user.omega, omegaEx);
  }

  PetscCall(SNESDestroy(&user.snes));
  PetscCall(DMDestroy(&user.dmN));
  PetscCall(ISDestroy(&user.isN));
  PetscCall(MatDestroy(&user.MN));
  PetscCall(DMDestroy(&user.dmP));
  PetscCall(ISDestroy(&user.isP));
  PetscCall(MatDestroy(&user.MP));
  PetscCall(DMDestroy(&user.dmE));
  PetscCall(ISDestroy(&user.isE));
  PetscCall(MatDestroy(&user.ME));
  PetscCall(DMDestroy(&user.dmMom));
  PetscCall(DMDestroy(&user.dmPot));
  PetscCall(ISDestroy(&user.isPot));
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

  # This tests that we can compute the correct decay rate and frequency
  #   For gold runs, use -dm_plex_box_faces 160 -vdm_plex_box_faces 450 -remap_dm_plex_box_faces 80,150 -ts_max_steps 1000
  #                      -remap_freq 100 -emax_start_step 50 -emax_solve_step 100
  testset:
    args: -cosine_coefficients 0.01 -charges -1. -perturbed_weights -total_weight 1. \
          -dm_plex_dim 1 -dm_plex_box_faces 80 -dm_plex_box_lower 0. -dm_plex_box_upper 12.5664 \
            -dm_plex_box_bd periodic -dm_plex_hash_location \
          -vdm_plex_dim 1 -vdm_plex_box_faces 220 -vdm_plex_box_lower -6 -vdm_plex_box_upper 6 \
            -vpetscspace_degree 2 -vdm_plex_hash_location \
          -remap_freq 1 -dm_swarm_remap_type pfak -remap_dm_plex_dim 2 -remap_dm_plex_simplex 0 \
            -remap_dm_plex_box_faces 40,110 -remap_dm_plex_box_bd periodic,none \
            -remap_dm_plex_box_lower 0.,-6. -remap_dm_plex_box_upper 12.5664,6. \
            -remap_petscspace_degree 1 -remap_dm_plex_hash_location \
            -ftop_ksp_type lsqr -ftop_pc_type none -ftop_ksp_rtol 1.e-14 -ptof_pc_type lu \
          -em_type primal -petscspace_degree 1 -em_snes_atol 1.e-12 -em_snes_error_if_not_converged \
            -em_ksp_error_if_not_converged -em_pc_type svd -em_proj_pc_type lu \
          -ts_dt 0.03 -ts_max_steps 2 -ts_max_time 100 \
          -emax_tao_type brgn -emax_tao_max_it 100 -emax_tao_brgn_regularization_type l2pure \
            -emax_tao_brgn_regularizer_weight 1e-5 -tao_brgn_subsolver_tao_bnk_ksp_rtol 1e-12 \
            -emax_start_step 1 -emax_solve_step 1 \
          -output_step 1 -efield_monitor quiet

    test:
      suffix: landau_damping_1d_bs
      args: -ts_type basicsymplectic -ts_basicsymplectic_type 1

    test:
      suffix: landau_damping_1d_dg
      args: -ts_type discgrad -ts_discgrad_type average -snes_type qn

TEST*/
