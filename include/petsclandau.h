#if !defined(PETSCLANDAU_H)
#define PETSCLANDAU_H

#include <petscdmplex.h> /*I      "petscdmplex.h"    I*/
#include <petscts.h>

PETSC_EXTERN PetscErrorCode LandauPrintNorms(Vec, PetscInt);
PETSC_EXTERN PetscErrorCode LandauCreateVelocitySpace(MPI_Comm,PetscInt,const char[],Vec*,Mat*,DM*);
PETSC_EXTERN PetscErrorCode LandauDestroyVelocitySpace(DM*);
PETSC_EXTERN PetscErrorCode LandauAddMaxwellians(DM, Vec, PetscReal, PetscReal[], PetscReal[], PetscInt, PetscInt, void *);
PETSC_EXTERN PetscErrorCode LandauCreateMassMatrix(DM dm, Mat *Amat);
PETSC_EXTERN PetscErrorCode LandauIFunction(TS, PetscReal,Vec,Vec,Vec,void *);
PETSC_EXTERN PetscErrorCode LandauIJacobian(TS, PetscReal,Vec,Vec,PetscReal,Mat,Mat,void *);

/* the Fokker-Planck-Landau context */
#if !defined(LANDAU_DIM)
#define LANDAU_DIM 2
#endif

#if !defined(LANDAU_MAX_SPECIES)
#if LANDAU_DIM==2
#define LANDAU_MAX_SPECIES 10
#define LANDAU_MAX_GRIDS 3
#else
#define LANDAU_MAX_SPECIES 3
#define LANDAU_MAX_GRIDS 2
#endif
#else
#define LANDAU_MAX_GRIDS 3
#endif

#define LANDAU_MAX_BATCH_SZ 320

#if !defined(LANDAU_MAX_Q)
#if defined(LANDAU_MAX_NQ)
#error"LANDAU_MAX_NQ but not LANDAU_MAX_Q. Use -DLANDAU_MAX_Q=4 for Q3 elements"
#endif
#if LANDAU_DIM==2
#define LANDAU_MAX_Q 5
#else
#define LANDAU_MAX_Q 3
#endif
#else
#undef LANDAU_MAX_NQ
#endif

#if LANDAU_DIM==2
#define LANDAU_MAX_Q_FACE LANDAU_MAX_Q
#define LANDAU_MAX_NQ (LANDAU_MAX_Q*LANDAU_MAX_Q)
#else
#define LANDAU_MAX_Q_FACE (LANDAU_MAX_Q*LANDAU_MAX_Q)
#define LANDAU_MAX_NQ (LANDAU_MAX_Q*LANDAU_MAX_Q*LANDAU_MAX_Q)
#endif

typedef enum {LANDAU_CUDA, LANDAU_KOKKOS, LANDAU_CPU} LandauDeviceType;

// static data - will be "device" data
typedef struct {
  void  *invJ;  // nip*dim*dim
  void  *D;     // nq*nb*dim
  void  *B;     // nq*nb
  void  *alpha; // ns
  void  *beta;  // ns
  void  *invMass; // ns
  void  *w; // nip
  void  *x; // nip
  void  *y; // nip
  void  *z; // nip
  void  *Eq_m; // ns - dynamic
  void  *f; //  nip*Nf - dynamic (IP)
  void  *dfdx; // nip*Nf - dynamic (IP)
  void  *dfdy; // nip*Nf - dynamic (IP)
  void  *dfdz; // nip*Nf - dynamic (IP)
  int   dim_,ns_,nip_,nq_,nb_;
  void  *NCells; // remove and ise elem_offset - TODO
  void  *species_offset; // for each grid, but same for all batched vertices
  void  *mat_offset; // for each grid, but same for all batched vertices
  void  *elem_offset; // for each grid, but same for all batched vertices
  void  *ip_offset; // for each grid, but same for all batched vertices
  void  *ipf_offset; // for each grid, but same for all batched vertices
  void  *ipfdf_data; // for each grid, but same for all batched vertices
  void  *maps; // for each grid, but same for all batched vertices
} LandauStaticData;

typedef enum {LANDAU_EX2_TSSOLVE, LANDAU_MATRIX_TOTAL, LANDAU_OPERATOR, LANDAU_JACOBIAN_COUNT, LANDAU_JACOBIAN, LANDAU_MASS, LANDAU_F_DF, LANDAU_KERNEL, KSP_FACTOR, KSP_SOLVE, LANDAU_NUM_TIMERS} LandauOMPTimers;

typedef struct {
  PetscBool      interpolate;                  /* Generate intermediate mesh elements */
  PetscBool      gpu_assembly;
  MPI_Comm       comm; /* global communicator to use for errors and diagnostics */
  double         times[LANDAU_NUM_TIMERS];
  PetscBool      use_matrix_mass;
  /* FE */
  PetscFE        fe[LANDAU_MAX_SPECIES];
  /* geometry  */
  PetscReal      radius[LANDAU_MAX_GRIDS];
  PetscReal      re_radius;           /* RE: radius of refinement along v_perp=0, z>0 */
  PetscReal      vperp0_radius1;      /* RE: radius of refinement along v_perp=0 */
  PetscReal      vperp0_radius2;      /* RE: radius of refinement along v_perp=0 after origin AMR refinement */
  PetscBool      sphere; // not used
  PetscBool      inflate; // not used
  PetscReal      i_radius[LANDAU_MAX_GRIDS]; // not used
  PetscReal      e_radius; // not used
  PetscInt       num_sections; // not used
  /* AMR */
  PetscBool      use_p4est;
  PetscInt       numRERefine;        /* RE: refinement along v_perp=0, z > 0 */
  PetscInt       nZRefine1;          /* RE: origin refinement after v_perp=0 refinement */
  PetscInt       nZRefine2;          /* RE: origin refinement after origin AMR refinement */
  PetscInt       numAMRRefine[LANDAU_MAX_GRIDS];  /* normal AMR - refine from origin */
  PetscInt       postAMRRefine[LANDAU_MAX_GRIDS]; /* uniform refinement of AMR */
  /* relativistic */
  PetscBool      use_energy_tensor_trick;
  PetscBool      use_relativistic_corrections;
  /* physics */
  PetscReal      thermal_temps[LANDAU_MAX_SPECIES];
  PetscReal      masses[LANDAU_MAX_SPECIES];  /* mass of each species  */
  PetscReal      charges[LANDAU_MAX_SPECIES]; /* charge of each species  */
  PetscReal      n[LANDAU_MAX_SPECIES];       /* number density of each species  */
  PetscReal      m_0;      /* reference mass */
  PetscReal      v_0;      /* reference velocity */
  PetscReal      n_0;      /* reference number density */
  PetscReal      t_0;      /* reference time */
  PetscReal      Ez;
  PetscReal      epsilon0;
  PetscReal      k;
  PetscReal      lnLam;
  PetscReal      electronShift;
  PetscInt       num_species;
  PetscInt       num_grids;
  PetscInt       species_offset[LANDAU_MAX_GRIDS+1]; // for each grid, but same for all batched vertices
  PetscInt       mat_offset[LANDAU_MAX_GRIDS+1]; // for each grid, but same for all batched vertices
  /* cache */
  Mat              J;
  Mat              M;
  Vec              X;
  /* derived type */
  void             *data;
  /* computing */
  LandauDeviceType deviceType;
  PetscInt         subThreadBlockSize;
  PetscInt         numConcurrency; /* number of SMs in Cuda to use */
  DM               pack;
  DM               plex[LANDAU_MAX_GRIDS];
  LandauStaticData SData_d; /* static geometric data on device */
  /* diagnostics */
  PetscInt         verbose;
  PetscLogEvent    events[20];
  PetscLogStage    stage;
  PetscBool        aux_bool;  /* helper */
  PetscInt         batch_sz;
  PetscInt         batch_view_idx;
} LandauCtx;

#define LANDAU_SPECIES_MAJOR
#if !defined(LANDAU_SPECIES_MAJOR)
#define LAND_PACK_IDX(_b,_g) (_b*ctx->num_grids + _g)
#define LAND_MOFFSET(_b,_g,_nbch,_ngrid,_mat_off) (_b*_mat_off[_ngrid] + _mat_off[_g])
#else
#define LAND_PACK_IDX(_b,_g) (_g*ctx->batch_sz + _b)
#define LAND_MOFFSET(_b,_g,_nbch,_ngrid,_mat_off) (_nbch*_mat_off[_g] + _b*(_mat_off[_g+1] - _mat_off[_g]))
#endif

typedef int LandauIdx;
typedef struct {
  PetscReal scale;
  LandauIdx gid;   // Lanadu matrix index (<10,000)
} pointInterpolationP4est;
typedef struct _lP4estVertexMaps {
  LandauIdx                (*gIdx)[LANDAU_MAX_SPECIES][LANDAU_MAX_NQ]; // #elems *  LANDAU_MAX_NQ (spoof for max , Nb) on device,
  LandauIdx                num_elements;
  LandauIdx                num_reduced;
  LandauIdx                num_face;  // (Q or Q^2 for 3D)
  LandauDeviceType         deviceType;
  PetscInt                 Nf;
  PetscInt                 Nq;
  pointInterpolationP4est (*c_maps)[LANDAU_MAX_Q_FACE];
  struct _lP4estVertexMaps*d_self;
  void                    *vp1,*vp2,*vp3;
  PetscInt                numgrids;
} P4estVertexMaps;

PETSC_EXTERN PetscErrorCode LandauCreateColoring(Mat, DM, PetscContainer *);
#if defined(PETSC_HAVE_CUDA)
PETSC_EXTERN PetscErrorCode LandauCUDAJacobian(DM[], const PetscInt, const PetscInt, const PetscInt, const PetscInt[], PetscReal[], PetscScalar[], const PetscScalar[],
                                               const LandauStaticData *, const PetscInt, const PetscReal, const PetscLogEvent[],const PetscInt[], const PetscInt[], Mat[], Mat);
PETSC_EXTERN PetscErrorCode LandauCUDACreateMatMaps(P4estVertexMaps *, pointInterpolationP4est (*)[LANDAU_MAX_Q_FACE], PetscInt[], PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode LandauCUDADestroyMatMaps(P4estVertexMaps *, PetscInt);
PETSC_EXTERN PetscErrorCode LandauCUDAStaticDataSet(DM, const PetscInt, const PetscInt, const PetscInt, PetscInt[], PetscInt[], PetscInt[], PetscReal [], PetscReal [], PetscReal[], PetscReal[], PetscReal[], PetscReal[], PetscReal[],
                                                    PetscReal[], LandauStaticData *);
PETSC_EXTERN PetscErrorCode LandauCUDAStaticDataClear(LandauStaticData *);
#endif
#if defined(PETSC_HAVE_KOKKOS)
PETSC_EXTERN PetscErrorCode LandauKokkosJacobian(DM[], const PetscInt, const PetscInt, const PetscInt, const PetscInt[], PetscReal[], PetscScalar[],  const PetscScalar[],
                                                 const LandauStaticData *, const PetscInt, const PetscReal, const PetscLogEvent[], const PetscInt[], const PetscInt[], Mat[], Mat);
PETSC_EXTERN PetscErrorCode LandauKokkosCreateMatMaps(P4estVertexMaps *, pointInterpolationP4est (*)[LANDAU_MAX_Q_FACE], PetscInt[], PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode LandauKokkosDestroyMatMaps(P4estVertexMaps *, PetscInt);
PETSC_EXTERN PetscErrorCode LandauKokkosStaticDataSet(DM, const PetscInt, const PetscInt, const PetscInt, PetscInt[], PetscInt[], PetscInt[], PetscReal [], PetscReal [], PetscReal[], PetscReal[], PetscReal[], PetscReal[], PetscReal[],
                                                      PetscReal[], LandauStaticData *);
PETSC_EXTERN PetscErrorCode LandauKokkosStaticDataClear(LandauStaticData*);
#endif

#endif /* PETSCLANDAU_H */
