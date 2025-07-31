static char help[] = "Landau Damping/Two Stream instability test using Vlasov-Poisson equations\n";

/*
  TODO:
  - Cache mesh geometry
  - Move electrostatic solver to MG+SVD

  To run the code with particles sinusoidally perturbed in x space use the test "pp_poisson_bsi_1d_4" or "pp_poisson_bsi_2d_4"
  According to Lukas, good damping results come at ~16k particles per cell

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

For a Landau Damping verification run, we use

  # Physics
  -cosine_coefficients 0.01 -dm_swarm_num_species 1 -charges -1. -perturbed_weights -total_weight 1.
  # Spatial Mesh
  -dm_plex_dim 1 -dm_plex_box_faces 40 -dm_plex_box_lower 0. -dm_plex_box_upper 12.5664 -dm_plex_box_bd periodic  -dm_plex_hash_location
  # Velocity Mesh
  -vdm_plex_dim 1 -vdm_plex_box_faces 40 -vdm_plex_box_lower -6 -vdm_plex_box_upper 6 -vpetscspace_degree 2 -vdm_plex_hash_location
  # Remap Space
  -dm_swarm_remap_type pfak -remap_freq 100
  -remap_dm_plex_dim 2 -remap_dm_plex_simplex 0 -remap_dm_plex_box_faces 20,20 -remap_dm_plex_box_bd periodic,none -remap_dm_plex_box_lower 0.,-6.
    -remap_dm_plex_box_upper 12.5664,6. -remap_petscspace_degree 1 -remap_dm_plex_hash_location
  # Remap Solve
  -ftop_ksp_type lsqr -ftop_pc_type none -ftop_ksp_rtol 1.e-14 -ptof_pc_type lu
  # EM Solve
  -em_type primal -petscspace_degree 1 -em_snes_atol 1.e-12 -em_snes_error_if_not_converged -em_ksp_error_if_not_converged -em_pc_type svd -em_proj_pc_type lu
  # Timestepping
  -ts_type basicsymplectic -ts_basicsymplectic_type 1 -ts_dt 0.03 -ts_max_steps 1500 -ts_max_time 100
  # Monitoring
  -output_step 1 -check_vel_res -efield_monitor -poisson_monitor -positions_monitor -dm_swarm_print_coords 0 -remap_uf_view draw
    -ftop_ksp_lsqr_monitor -ftop_ksp_converged_reason

*/
#include <petscts.h>
#include <petscdmplex.h>
#include <petscdmswarm.h>
#include <petscfe.h>
#include <petscds.h>
#include <petscbag.h>
#include <petscdraw.h>
#include <petsc/private/dmpleximpl.h>  /* For norm and dot */
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
  PetscBag     bag;               // Problem parameters
  PetscBool    error;             // Flag for printing the error
  PetscInt     remapFreq;         // Number of timesteps between remapping
  PetscBool    efield_monitor;    // Flag to show electric field monitor
  PetscBool    moment_monitor;    // Flag to show distribution moment monitor
  PetscBool    positions_monitor; // Flag to show particle positins at each time step
  PetscBool    poisson_monitor;   // Flag to display charge, E field, and potential at each solve
  PetscBool    initial_monitor;   // Flag to monitor the initial conditions
  PetscInt     velocity_monitor;  // Cell to monitor the velocity distribution for
  PetscBool    perturbed_weights; // Uniformly sample x,v space with gaussian weights
  PetscInt     ostep;             // Print the energy at each ostep time steps
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
  DM           swarm;
  PetscRandom  random;
  PetscBool    twostream;
  PetscBool    checkweights;
  PetscInt     checkVRes; // Flag to check/output velocity residuals for nightly tests

  PetscLogEvent RhsXEvent, RhsVEvent, ESolveEvent, ETabEvent;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  PetscInt d                      = 2;
  PetscInt maxSpecies             = 2;
  options->error                  = PETSC_FALSE;
  options->remapFreq              = 1;
  options->efield_monitor         = PETSC_FALSE;
  options->moment_monitor         = PETSC_FALSE;
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
  options->em                     = EM_COULOMB;
  options->snes                   = NULL;
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

  PetscOptionsBegin(comm, "", "Landau Damping and Two Stream options", "DMSWARM");
  PetscCall(PetscOptionsBool("-error", "Flag to print the error", "ex2.c", options->error, &options->error, NULL));
  PetscCall(PetscOptionsInt("-remap_freq", "Number", "ex2.c", options->remapFreq, &options->remapFreq, NULL));
  PetscCall(PetscOptionsBool("-efield_monitor", "Flag to plot log(max E) over time", "ex2.c", options->efield_monitor, &options->efield_monitor, NULL));
  PetscCall(PetscOptionsReal("-efield_min_monitor", "Minimum E field to plot", "ex2.c", options->drawlgEmin, &options->drawlgEmin, NULL));
  PetscCall(PetscOptionsBool("-moments_monitor", "Flag to show moments table", "ex2.c", options->moment_monitor, &options->moment_monitor, NULL));
  PetscCall(PetscOptionsBool("-ics_monitor", "Flag to show initial condition histograms", "ex2.c", options->initial_monitor, &options->initial_monitor, NULL));
  PetscCall(PetscOptionsBool("-positions_monitor", "The flag to show particle positions", "ex2.c", options->positions_monitor, &options->positions_monitor, NULL));
  PetscCall(PetscOptionsBool("-poisson_monitor", "The flag to show charges, Efield and potential solve", "ex2.c", options->poisson_monitor, &options->poisson_monitor, NULL));
  PetscCall(PetscOptionsInt("-velocity_monitor", "Cell to show velocity histograms", "ex2.c", options->velocity_monitor, &options->velocity_monitor, NULL));
  PetscCall(PetscOptionsBool("-twostream", "Run two stream instability", "ex2.c", options->twostream, &options->twostream, NULL));
  PetscCall(PetscOptionsBool("-perturbed_weights", "Flag to run uniform sampling with perturbed weights", "ex2.c", options->perturbed_weights, &options->perturbed_weights, NULL));
  PetscCall(PetscOptionsBool("-check_weights", "Ensure all particle weights are positive", "ex2.c", options->checkweights, &options->checkweights, NULL));
  PetscCall(PetscOptionsInt("-check_vel_res", "Check particle velocity residuals for nightly tests", "ex2.c", options->checkVRes, &options->checkVRes, NULL));
  PetscCall(PetscOptionsInt("-output_step", "Number of time steps between output", "ex2.c", options->ostep, &options->ostep, NULL));
  PetscCall(PetscOptionsReal("-timeScale", "Nondimensionalizing time scale", "ex2.c", options->timeScale, &options->timeScale, NULL));
  PetscCall(PetscOptionsReal("-initial_velocity", "Initial velocity of perturbed particle", "ex2.c", options->initVel, &options->initVel, NULL));
  PetscCall(PetscOptionsReal("-total_weight", "Total weight of all particles", "ex2.c", options->totalWeight, &options->totalWeight, NULL));
  PetscCall(PetscOptionsRealArray("-cosine_coefficients", "Amplitude and frequency of cosine equation used in initialization", "ex2.c", options->cosine_coefficients, &d, NULL));
  PetscCall(PetscOptionsRealArray("-charges", "Species charges", "ex2.c", options->charges, &maxSpecies, NULL));
  PetscCall(PetscOptionsEnum("-em_type", "Type of electrostatic solver", "ex2.c", EMTypes, (PetscEnum)options->em, (PetscEnum *)&options->em, NULL));
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
    PetscCall(PetscDrawSetSave(draw, "ex2_Efield"));
    PetscCall(PetscDrawSetFromOptions(draw));
    PetscCall(PetscDrawLGCreate(draw, 1, &user->drawlgE));
    PetscCall(PetscDrawDestroy(&draw));
    PetscCall(PetscDrawLGGetAxis(user->drawlgE, &axis));
    PetscCall(PetscDrawAxisSetLabels(axis, "Electron Electric Field", "time", "E_max"));
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
#if 0
  PetscReal  *v, pvmin[3] = {0., 0., 0.}, pvmax[3] = {0., 0., 0.}, vmin[3], vmax[3], fact = 1.;
  PetscInt    dim, Np;

  PetscCall(PetscObjectQuery((PetscObject)sw, "__vdm__", (PetscObject *)&vdm));
  // Check for particles outside the velocity grid
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetLocalSize(sw, &Np));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&v));
  for (PetscInt p = 0; p < Np; ++p) {
    for (PetscInt d = 0; d < dim; ++d) {
      pvmin[d] = PetscMin(pvmax[d], v[p * dim + d]);
      pvmax[d] = PetscMax(pvmax[d], v[p * dim + d]);
    }
  }
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&v));
  PetscCall(DMGetBoundingBox(vdm, vmin, vmax));
  // To avoid particle loss, we enlarge the velocity grid if necessary
  for (PetscInt d = 0; d < dim; ++d) {
    fact = PetscMax(fact, pvmax[d] / vmax[d]);
    fact = PetscMax(fact, pvmin[d] / vmin[d]);
  }
  if (fact > 1.) {
    Vec coordinates, coordinatesLocal;

    fact *= 1.1;
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Expanding velocity grid by %g\n", fact));
    PetscCall(DMGetCoordinatesLocal(vdm, &coordinatesLocal));
    PetscCall(DMGetCoordinates(vdm, &coordinates));
    PetscCall(VecScale(coordinatesLocal, fact));
    PetscCall(VecScale(coordinates, fact));
    PetscCall(PetscGridHashDestroy(&((DM_Plex *)vdm->data)->lbox));
  }
#endif
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

static PetscErrorCode zero(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  *u = 0.0;
  return PETSC_SUCCESS;
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

  const char *vfieldnames[1] = {"w_q"};

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

  // Low-pass filter rhoRhs
  PetscInt window = 0;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-rho_average", &window, NULL));
  if (window) {
    PetscScalar *a;
    PetscInt     n;
    PetscReal    width = 2. * window + 1.;

    // This will only work for P_1
    //   This works because integration against a test function is linear
    //   This only works in serial since I need the periodic values (maybe use FFT)
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

  PetscCall(SNESDestroy(&user.snes));
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

  # This tests that we can put particles in a box and compute the Coulomb force
  # Recommend -draw_size 500,500
   testset:
     requires: defined(PETSC_HAVE_EXECUTABLE_EXPORT)
     args: -dm_plex_dim 1 -dm_plex_simplex 0 -dm_plex_box_faces 20 \
             -dm_plex_box_lower 0 -dm_plex_box_upper 12.5664 -dm_plex_box_bd periodic \
           -vdm_plex_simplex 0 \
           -dm_swarm_coordinate_density constant -dm_swarm_num_particles 100 \
           -sigma 1.0e-8 -timeScale 2.0e-14 \
           -ts_type basicsymplectic -ts_basicsymplectic_type 1 \
           -output_step 50 -check_vel_res
     test:
       suffix: none_1d
       requires:
       args: -em_type none -error
     test:
       suffix: coulomb_1d
       args: -em_type coulomb

   # for viewers
   #-ts_monitor_sp_swarm_phase -ts_monitor_sp_swarm -em_snes_monitor -ts_monitor_sp_swarm_multi_species 0 -ts_monitor_sp_swarm_retain 0
   testset:
     nsize: {{1 2}}
     requires: defined(PETSC_HAVE_EXECUTABLE_EXPORT)
     args: -dm_plex_dim 1 -dm_plex_simplex 0 -dm_plex_box_faces 36 \
             -dm_plex_box_lower 0. -dm_plex_box_upper 12.5664 -dm_plex_box_bd periodic \
           -vdm_plex_dim 1 -vdm_plex_simplex 0 -vdm_plex_box_faces 10 \
             -vdm_plex_box_lower -3 -vdm_plex_box_upper 3 \
           -dm_swarm_num_species 1 -twostream -charges -1.,1. -sigma 1.0e-8 \
             -cosine_coefficients 0.01,0.5 -perturbed_weights -total_weight 1. \
           -ts_type basicsymplectic -ts_basicsymplectic_type 2 \
             -ts_dt 0.01 -ts_max_time 5 -ts_max_steps 10 \
           -em_snes_atol 1.e-15 -em_snes_error_if_not_converged -em_ksp_error_if_not_converged \
           -output_step 1 -check_vel_res -dm_swarm_print_coords 1 -dm_swarm_print_weights 1
     test:
       suffix: two_stream_c0
       args: -em_type primal -petscfe_default_quadrature_order 2 -petscspace_degree 2 -em_pc_type svd
     test:
       suffix: two_stream_rt
       requires: superlu_dist
       args: -em_type mixed \
               -potential_petscspace_degree 0 \
               -potential_petscdualspace_lagrange_use_moments \
               -potential_petscdualspace_lagrange_moment_order 2 \
               -field_petscspace_degree 1 -field_petscfe_default_quadrature_order 1 \
             -em_snes_error_if_not_converged \
             -em_ksp_type preonly -em_ksp_error_if_not_converged \
             -em_pc_type fieldsplit -em_pc_fieldsplit_type schur \
               -em_pc_fieldsplit_schur_fact_type full -em_pc_fieldsplit_schur_precondition full \
               -em_fieldsplit_field_pc_type lu \
                 -em_fieldsplit_field_pc_factor_mat_solver_type superlu_dist \
               -em_fieldsplit_potential_pc_type svd

   # For an eyeball check, we use
   # -ts_max_steps 1000 -dm_plex_box_faces 10,1 -vdm_plex_box_faces 2000 -efield_monitor
   # For verification, we use
   # -ts_max_steps 1000 -dm_plex_box_faces 100,1 -vdm_plex_box_faces 8000 -dm_swarm_num_particles 800000 -monitor_efield
   # -ts_monitor_sp_swarm_multi_species 0 -ts_monitor_sp_swarm_retain 0 -ts_monitor_sp_swarm_phase 1 -draw_size 500,500
   testset:
     nsize: {{1 2}}
     args: -dm_plex_dim 1 -dm_plex_simplex 0 -dm_plex_box_faces 10 \
             -dm_plex_box_lower 0. -dm_plex_box_upper 12.5664 -dm_plex_box_bd periodic \
           -vdm_plex_dim 1 -vdm_plex_simplex 0 -vdm_plex_box_faces 10 \
             -vdm_plex_box_lower -10 -vdm_plex_box_upper 10 \
             -vpetscspace_degree 2 -vdm_plex_hash_location \
           -dm_swarm_num_species 1 -charges -1.,1. \
             -cosine_coefficients 0.01,0.5 -perturbed_weights -total_weight 1. \
           -ts_type basicsymplectic -ts_basicsymplectic_type 1 \
             -ts_dt 0.03 -ts_max_time 500 -ts_max_steps 1 \
           -em_snes_atol 1.e-12 -em_snes_error_if_not_converged -em_ksp_error_if_not_converged \
           -output_step 1 -check_vel_res -dm_swarm_print_coords 1 -dm_swarm_print_weights 1

     test:
       suffix: uniform_equilibrium_1d
       args: -cosine_coefficients 0.0,0.5 -em_type primal -petscspace_degree 1 -em_pc_type svd
     test:
       suffix: uniform_equilibrium_1d_real
       args: -dm_plex_dim 1 -dm_plex_simplex 1 -dm_plex_box_faces 10 \
               -dm_plex_box_lower 0. -dm_plex_box_upper 12.5664 -dm_plex_box_bd periodic \
             -cosine_coefficients 0.0 -em_type primal -petscspace_degree 1 -em_pc_type svd
     test:
       suffix: landau_damping_1d_c0
       args: -em_type primal -petscspace_degree 1 -em_pc_type svd
     test:
       suffix: uniform_primal_1d_real
       args: -dm_plex_dim 1 -dm_plex_simplex 1 -dm_plex_box_faces 10 \
               -dm_plex_box_lower 0. -dm_plex_box_upper 12.5664 -dm_plex_box_bd periodic \
             -cosine_coefficients 0.01 -em_type primal -petscspace_degree 1 -em_pc_type svd
     # NOT WORKING -ftop_pc_type bjacobi -ftop_sub_pc_type lu -ftop_sub_pc_factor_shift_type nonzero
     test:
       suffix: uniform_primal_1d_real_pfak
       nsize: 1
       args: -dm_plex_dim 1 -dm_plex_simplex 1 -dm_plex_box_faces 10 \
               -dm_plex_box_lower 0. -dm_plex_box_upper 12.5664 -dm_plex_box_bd periodic \
             -remap_dm_plex_dim 2 -remap_dm_plex_simplex 0 -remap_dm_plex_box_faces 10,10 -remap_dm_plex_box_bd periodic,none \
               -remap_dm_plex_box_lower 0.,-10. -remap_dm_plex_box_upper 12.5664,10. \
               -remap_petscspace_degree 2 -remap_dm_plex_hash_location \
             -remap_freq 1 -dm_swarm_remap_type pfak \
               -ftop_ksp_type lsqr -ftop_pc_type none -ftop_ksp_rtol 1.e-14 \
               -ptof_pc_type lu \
             -cosine_coefficients 0.01 -em_type primal -petscspace_degree 1 -em_pc_type svd -em_proj_pc_type lu
     test:
       requires: superlu_dist
       suffix: landau_damping_1d_mixed
       args: -em_type mixed \
               -potential_petscspace_degree 0 \
               -potential_petscdualspace_lagrange_use_moments \
               -potential_petscdualspace_lagrange_moment_order 2 \
               -field_petscspace_degree 1 \
             -em_snes_error_if_not_converged \
             -em_ksp_type preonly -em_ksp_error_if_not_converged \
             -em_pc_type fieldsplit -em_pc_fieldsplit_type schur \
               -em_pc_fieldsplit_schur_fact_type full -em_pc_fieldsplit_schur_precondition full \
               -em_fieldsplit_field_pc_type lu \
                 -em_fieldsplit_field_pc_factor_mat_solver_type superlu_dist \
               -em_fieldsplit_potential_pc_type svd

   # Same as above, with different timestepping
   testset:
     nsize: {{1 2}}
     args: -dm_plex_dim 1 -dm_plex_simplex 0 -dm_plex_box_faces 10 \
             -dm_plex_box_lower 0. -dm_plex_box_upper 12.5664 -dm_plex_box_bd periodic \
           -vdm_plex_dim 1 -vdm_plex_simplex 0 -vdm_plex_box_faces 10 \
             -vdm_plex_box_lower -10 -vdm_plex_box_upper 10 \
             -vpetscspace_degree 2 -vdm_plex_hash_location \
           -dm_swarm_num_species 1 -charges -1.,1. \
             -cosine_coefficients 0.01,0.5 -perturbed_weights -total_weight 1. \
           -ts_type discgrad -ts_discgrad_type average \
             -ts_dt 0.03 -ts_max_time 500 -ts_max_steps 1 \
           -snes_type qn \
           -em_snes_atol 1.e-12 -em_snes_error_if_not_converged -em_ksp_error_if_not_converged \
           -output_step 1 -check_vel_res -dm_swarm_print_coords 1 -dm_swarm_print_weights 1

     test:
       suffix: landau_damping_1d_dg
       args: -em_type primal -petscspace_degree 1 -em_pc_type svd

TEST*/
