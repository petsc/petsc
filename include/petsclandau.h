#pragma once

#include <petscdmplex.h> /*I      "petscdmplex.h"    I*/
#include <petscts.h>

/* MANSEC = TS */
/* SUBMANSEC = LANDAU */

PETSC_EXTERN PetscErrorCode DMPlexLandauPrintNorms(Vec, PetscInt);
PETSC_EXTERN PetscErrorCode DMPlexLandauCreateVelocitySpace(MPI_Comm, PetscInt, const char[], Vec *, Mat *, DM *);
PETSC_EXTERN PetscErrorCode DMPlexLandauDestroyVelocitySpace(DM *);
PETSC_EXTERN PetscErrorCode DMPlexLandauAccess(DM, Vec, PetscErrorCode (*)(DM, Vec, PetscInt, PetscInt, PetscInt, void *), void *);
PETSC_EXTERN PetscErrorCode DMPlexLandauAddMaxwellians(DM, Vec, PetscReal, PetscReal[], PetscReal[], PetscInt, PetscInt, PetscInt, void *);
PETSC_EXTERN PetscErrorCode DMPlexLandauCreateMassMatrix(DM dm, Mat *Amat);
PETSC_EXTERN PetscErrorCode DMPlexLandauIFunction(TS, PetscReal, Vec, Vec, Vec, void *);
PETSC_EXTERN PetscErrorCode DMPlexLandauIJacobian(TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void *);

/*MC
   LandauIdx - Integer type used to index entries in the `DMPLEX` Landau collision-operator data structures, such as the COO matrix workspaces and the `P4estVertexMaps` reduced-quadrature maps

   Level: developer

   Note:
   `LandauIdx` is a `PetscInt`; it is named separately so the device-side data structures used by the Landau collision operator can be sized independently from the rest of PETSc if needed.

.seealso: `DMPlexLandauCreateVelocitySpace()`, `LandauStaticData`, `LandauCtx`, `P4estVertexMaps`
M*/
typedef PetscInt LandauIdx;

/* the Fokker-Planck-Landau context */
#if !defined(LANDAU_MAX_SPECIES)
  #if PetscDefined(USE_DMLANDAU_2D)
    #define LANDAU_MAX_SPECIES 10
    #define LANDAU_MAX_GRIDS   3
  #else
    #define LANDAU_MAX_SPECIES 10
    #define LANDAU_MAX_GRIDS   3
  #endif
#else
  #define LANDAU_MAX_GRIDS 3
#endif

#if !defined(LANDAU_MAX_Q)
  #if defined(LANDAU_MAX_NQND)
    #error "LANDAU_MAX_NQND but not LANDAU_MAX_Q. Use -DLANDAU_MAX_Q=4 for Q3 elements"
  #endif
  #if PetscDefined(USE_DMLANDAU_2D)
    #define LANDAU_MAX_Q 6
  #else
    #define LANDAU_MAX_Q 6
  #endif
#else
  #undef LANDAU_MAX_NQND
#endif

#if PetscDefined(USE_DMLANDAU_2D)
  #define LANDAU_MAX_Q_FACE   LANDAU_MAX_Q
  #define LANDAU_MAX_NQND     (LANDAU_MAX_Q * LANDAU_MAX_Q)
  #define LANDAU_MAX_BATCH_SZ 1024
  #define LANDAU_DIM          2
#else
  #define LANDAU_MAX_Q_FACE   (LANDAU_MAX_Q * LANDAU_MAX_Q)
  #define LANDAU_MAX_NQND     (LANDAU_MAX_Q * LANDAU_MAX_Q * LANDAU_MAX_Q)
  #define LANDAU_MAX_BATCH_SZ 64
  #define LANDAU_DIM          3
#endif

/*E
   LandauDeviceType - Selects the backend used to evaluate the Landau collision-operator Jacobian and to hold its workspace data

   Values:
+   `LANDAU_KOKKOS` - run on the device with the Kokkos backend (requires PETSc configured `--with-kokkos`)
-   `LANDAU_CPU`    - run on the host (default when Kokkos is not enabled)

   Level: intermediate

.seealso: `DMPlexLandauCreateVelocitySpace()`, `LandauCtx`, `LandauStaticData`
E*/
typedef enum {
  LANDAU_KOKKOS,
  LANDAU_CPU
} LandauDeviceType;

/*S
   LandauStaticData - Workspace of pre-computed quadrature, geometry, and physics data that is shared by every Jacobian assembly of the `DMPLEX` Landau collision operator

   Level: developer

   Note:
   The fields are typed as `void *` so that the same struct can hold either host arrays or device (Kokkos/CUDA/HIP) arrays depending on `LandauDeviceType`.
   The contents are managed by `DMPlexLandauCreateVelocitySpace()` and friends and are not intended to be inspected by user code.

.seealso: `DMPlexLandauCreateVelocitySpace()`, `LandauCtx`, `LandauDeviceType`, `LandauIdx`
S*/
typedef struct {
  void *invJ;    // nip*dim*dim
  void *D;       // nq*nb*dim
  void *B;       // nq*nb
  void *alpha;   // ns
  void *beta;    // ns
  void *invMass; // ns
  void *w;       // nip
  void *x;       // nip
  void *y;       // nip
  void *z;       // nip
  void *Eq_m;    // ns - dynamic
  void *f;       //  nip*Nf - dynamic (IP)
  void *dfdx;    // nip*Nf - dynamic (IP)
  void *dfdy;    // nip*Nf - dynamic (IP)
  void *dfdz;    // nip*Nf - dynamic (IP)
  int   dim_, ns_, nip_, nq_, nb_;
  void *NCells;         // remove and use elem_offset - TODO
  void *species_offset; // for each grid, but same for all batched vertices
  void *mat_offset;     // for each grid, but same for all batched vertices
  void *elem_offset;    // for each grid, but same for all batched vertices
  void *ip_offset;      // for each grid, but same for all batched vertices
  void *ipf_offset;     // for each grid, but same for all batched vertices
  void *ipfdf_data;     // for each grid, but same for all batched vertices
  void *maps;           // for each grid, but same for all batched vertices
  // COO
  void      *coo_elem_offsets;
  void      *coo_elem_point_offsets;
  void      *coo_elem_fullNb;
  void      *coo_vals;
  void      *lambdas;
  LandauIdx  coo_n_cellsTot;
  PetscCount coo_size;
  LandauIdx  coo_max_fullnb;
} LandauStaticData;

/*E
   LandauOMPTimers - Identifiers for the timing slots kept in `LandauCtx.times[]` for the `DMPLEX` Landau collision operator

   Values:
+   `LANDAU_EX2_TSSOLVE`    - total `TSSolve()` time of the Landau example
.   `LANDAU_MATRIX_TOTAL`   - total time spent constructing the Jacobian and mass matrices
.   `LANDAU_OPERATOR`       - time inside the Landau operator evaluation
.   `LANDAU_JACOBIAN_COUNT` - number of Jacobian evaluations (stored as a count, reused as a timer slot)
.   `LANDAU_JACOBIAN`       - time inside Jacobian construction
.   `LANDAU_MASS`           - time inside mass-matrix construction
.   `LANDAU_F_DF`           - time evaluating the distribution function and its derivatives
.   `LANDAU_KERNEL`         - time inside the Landau collision kernel
.   `KSP_FACTOR`            - time inside the KSP factor stage when using a direct solver
.   `KSP_SOLVE`             - time inside `KSPSolve()`
-   `LANDAU_NUM_TIMERS`     - sentinel; equals the number of timer slots allocated in `LandauCtx`

   Level: developer

.seealso: `LandauCtx`, `DMPlexLandauCreateVelocitySpace()`
E*/
typedef enum {
  LANDAU_EX2_TSSOLVE,
  LANDAU_MATRIX_TOTAL,
  LANDAU_OPERATOR,
  LANDAU_JACOBIAN_COUNT,
  LANDAU_JACOBIAN,
  LANDAU_MASS,
  LANDAU_F_DF,
  LANDAU_KERNEL,
  KSP_FACTOR,
  KSP_SOLVE,
  LANDAU_NUM_TIMERS
} LandauOMPTimers;

/*S
   LandauCtx - Application context for the `DMPLEX` Landau collision operator that records the species data, mesh configuration, AMR settings, batching parameters, and pre-computed static data needed to evaluate the operator

   Level: intermediate

   Note:
   The context is created and managed by `DMPlexLandauCreateVelocitySpace()` and is attached to the returned `DM` as its application context. User code normally obtains it with `DMGetApplicationContext()` rather than constructing it directly.

.seealso: `DMPlexLandauCreateVelocitySpace()`, `DMPlexLandauDestroyVelocitySpace()`, `DMPlexLandauIFunction()`, `DMPlexLandauIJacobian()`,
          `LandauStaticData`, `LandauDeviceType`, `LandauOMPTimers`
S*/
typedef struct {
  PetscBool interpolate; /* Generate intermediate mesh elements */
  PetscBool gpu_assembly;
  MPI_Comm  comm; /* global communicator to use for errors and diagnostics */
  double    times[LANDAU_NUM_TIMERS];
  PetscBool use_matrix_mass;
  /* FE */
  PetscFE fe[LANDAU_MAX_SPECIES];
  /* geometry  */
  PetscReal radius[LANDAU_MAX_GRIDS];
  PetscReal radius_par[LANDAU_MAX_GRIDS];
  PetscReal radius_perp[LANDAU_MAX_GRIDS];
  PetscReal re_radius;      /* RE: radius of refinement along v_perp=0, z>0 */
  PetscReal vperp0_radius1; /* RE: radius of refinement along v_perp=0 */
  PetscReal vperp0_radius2; /* RE: radius of refinement along v_perp=0 after origin AMR refinement */
  PetscBool sphere;
  PetscBool map_sphere;
  PetscReal sphere_inner_radius_90degree[LANDAU_MAX_GRIDS];
  PetscReal sphere_inner_radius_45degree[LANDAU_MAX_GRIDS];
  PetscInt  cells0[3];
  /* AMR */
  PetscBool use_p4est;
  PetscInt  numRERefine;                     /* RE: refinement along v_perp=0, z > 0 */
  PetscInt  nZRefine1;                       /* RE: origin refinement after v_perp=0 refinement */
  PetscInt  nZRefine2;                       /* RE: origin refinement after origin AMR refinement */
  PetscInt  numAMRRefine[LANDAU_MAX_GRIDS];  /* normal AMR - refine from origin */
  PetscInt  postAMRRefine[LANDAU_MAX_GRIDS]; /* uniform refinement of AMR */
  PetscBool simplex;
  char      filename[PETSC_MAX_PATH_LEN];
  PetscReal thermal_speed[LANDAU_MAX_GRIDS];
  PetscBool sphere_uniform_normal;
  /* relativistic */
  PetscBool use_energy_tensor_trick;
  PetscBool use_relativistic_corrections;
  /* physics */
  PetscReal thermal_temps[LANDAU_MAX_SPECIES];
  PetscReal masses[LANDAU_MAX_SPECIES];  /* mass of each species  */
  PetscReal charges[LANDAU_MAX_SPECIES]; /* charge of each species  */
  PetscReal n[LANDAU_MAX_SPECIES];       /* number density of each species  */
  PetscReal m_0;                         /* reference mass */
  PetscReal v_0;                         /* reference velocity */
  PetscReal n_0;                         /* reference number density */
  PetscReal t_0;                         /* reference time */
  PetscReal Ez;
  PetscReal epsilon0;
  PetscReal k;
  PetscReal lambdas[LANDAU_MAX_GRIDS][LANDAU_MAX_GRIDS];
  PetscReal electronShift;
  PetscInt  num_species;
  PetscInt  num_grids;
  PetscInt  species_offset[LANDAU_MAX_GRIDS + 1]; // for each grid, but same for all batched vertices
  PetscInt  mat_offset[LANDAU_MAX_GRIDS + 1];     // for each grid, but same for all batched vertices
  // batching
  PetscBool  jacobian_field_major_order; // this could be a type but lets not get pedantic
  VecScatter plex_batch;
  Vec        work_vec;
  IS         batch_is;
  PetscErrorCode (*seqaij_mult)(Mat, Vec, Vec);
  PetscErrorCode (*seqaij_multtranspose)(Mat, Vec, Vec);
  PetscErrorCode (*seqaij_solve)(Mat, Vec, Vec);
  PetscErrorCode (*seqaij_getdiagonal)(Mat, Vec);
  /* COO */
  Mat J;
  Mat M;
  Vec X;
  /* derived type */
  void *data;
  /* computing */
  LandauDeviceType deviceType;
  DM               pack;
  DM               plex[LANDAU_MAX_GRIDS];
  LandauStaticData SData_d; /* static geometric data on device */
  /* diagnostics */
  PetscInt         verbose;
  PetscLogEvent    events[20];
  PetscLogStage    stage;
  PetscObjectState norm_state;
  PetscInt         batch_sz;
  PetscInt         batch_view_idx;
} LandauCtx;

#define LANDAU_SPECIES_MAJOR
#if !defined(LANDAU_SPECIES_MAJOR)
  #define LAND_PACK_IDX(_b, _g)                         (_b * ctx->num_grids + _g)
  #define LAND_MOFFSET(_b, _g, _nbch, _ngrid, _mat_off) (_b * _mat_off[_ngrid] + _mat_off[_g])
#else
  #define LAND_PACK_IDX(_b, _g)                         (_g * ctx->batch_sz + _b)
  #define LAND_MOFFSET(_b, _g, _nbch, _ngrid, _mat_off) (_nbch * _mat_off[_g] + _b * (_mat_off[_g + 1] - _mat_off[_g]))
#endif

/*S
   pointInterpolationP4est - One entry in the reduced quadrature-point map used by the `DMPLEX` Landau collision operator; pairs a global Landau matrix index with the weight that scales the contribution of the corresponding quadrature point

   Level: developer

   Note:
   These records are arrays inside `P4estVertexMaps` and describe how multiple coincident quadrature points produced by the p4est-based AMR mesh are combined into a single entry of the Landau Jacobian.

.seealso: `LandauIdx`, `LandauCtx`, `LandauStaticData`, `DMPlexLandauCreateVelocitySpace()`
S*/
typedef struct {
  PetscReal scale;
  LandauIdx gid; // Landau matrix index (<10,000)
} pointInterpolationP4est;
typedef struct _lP4estVertexMaps {
  LandauIdx (*gIdx)[LANDAU_MAX_SPECIES][LANDAU_MAX_NQND]; // #elems *  LANDAU_MAX_NQND
  LandauIdx        num_elements;
  LandauIdx        num_reduced;
  LandauIdx        num_face; // (Q or Q^2 for 3D)
  LandauDeviceType deviceType;
  PetscInt         Nf;
  pointInterpolationP4est (*c_maps)[LANDAU_MAX_Q_FACE];
  struct _lP4estVertexMaps *d_self;
  void                     *vp1, *vp2, *vp3;
  PetscInt                  numgrids;
} P4estVertexMaps;

#if PetscDefined(HAVE_KOKKOS)
PETSC_EXTERN PetscErrorCode LandauKokkosJacobian(DM[], const PetscInt, const PetscInt, const PetscInt, const PetscInt, const PetscInt[], PetscReal[], PetscScalar[], const PetscScalar[], const LandauStaticData *, const PetscReal, const PetscLogEvent[], const PetscInt[], const PetscInt[], Mat[], Mat);
PETSC_EXTERN PetscErrorCode LandauKokkosCreateMatMaps(P4estVertexMaps *, pointInterpolationP4est (*)[LANDAU_MAX_Q_FACE], PetscInt[], PetscInt);
PETSC_EXTERN PetscErrorCode LandauKokkosDestroyMatMaps(P4estVertexMaps *, PetscInt);
PETSC_EXTERN PetscErrorCode LandauKokkosStaticDataSet(DM, const PetscInt, const PetscInt, const PetscInt, const PetscInt, PetscInt[], PetscInt[], PetscInt[], PetscReal[], PetscReal[], PetscReal[], PetscReal[], PetscReal[], PetscReal[], PetscReal[], PetscReal[], PetscReal[], LandauStaticData *);
PETSC_EXTERN PetscErrorCode LandauKokkosStaticDataClear(LandauStaticData *);
#endif
