#if !defined(PETSCLANDAU_H)
#define PETSCLANDAU_H

#include <petscdmplex.h> /*I      "petscdmplex.h"    I*/
#include <petscts.h>

PETSC_EXTERN PetscErrorCode LandauPrintNorms(Vec, PetscInt);
PETSC_EXTERN PetscErrorCode LandauCreateVelocitySpace(MPI_Comm,PetscInt,const char[],Vec*,Mat*,DM*);
PETSC_EXTERN PetscErrorCode LandauDestroyVelocitySpace(DM*);
PETSC_EXTERN PetscErrorCode LandauAddMaxwellians(DM, Vec, PetscReal, PetscReal[], PetscReal[], void *);
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
#else
#define LANDAU_MAX_SPECIES 3
#endif
#endif

#if !defined(LANDAU_MAX_Q)
#if defined(LANDAU_MAX_NQ)
#error"LANDAU_MAX_NQ but not LANDAU_MAX_Q. Use -DLANDAU_MAX_Q=4 for Q3 elements"
#endif
#if LANDAU_DIM==2
#define LANDAU_MAX_Q 5
#define LANDAU_MAX_Q_FACE LANDAU_MAX_Q
#else
#define LANDAU_MAX_Q 3
#define LANDAU_MAX_Q_FACE (LANDAU_MAX_Q*LANDAU_MAX_Q)
#endif
#else
#undef LANDAU_MAX_NQ
#if LANDAU_DIM==2
#define LANDAU_MAX_NQ (LANDAU_MAX_Q*LANDAU_MAX_Q)
#define LANDAU_MAX_Q_FACE LANDAU_MAX_Q
#else
#define LANDAU_MAX_NQ (LANDAU_MAX_Q*LANDAU_MAX_Q*LANDAU_MAX_Q)
#define LANDAU_MAX_Q_FACE (LANDAU_MAX_Q*LANDAU_MAX_Q)
#endif
#endif

#if LANDAU_DIM==2
#define LANDAU_MAX_NQ (LANDAU_MAX_Q*LANDAU_MAX_Q)
#else
#define LANDAU_MAX_NQ (LANDAU_MAX_Q*LANDAU_MAX_Q*LANDAU_MAX_Q)
#endif

typedef enum {LANDAU_CUDA, LANDAU_KOKKOS, LANDAU_CPU} LandauDeviceType;
typedef struct {
  PetscBool      interpolate;                  /* Generate intermediate mesh elements */
  PetscBool      gpu_assembly;
  PetscFE        fe[LANDAU_MAX_SPECIES];
  /* geometry  */
  PetscReal      i_radius;
  PetscReal      e_radius;
  PetscInt       num_sections;
  PetscReal      radius;
  PetscReal      re_radius;           /* radius of refinement along v_perp=0, z>0 */
  PetscReal      vperp0_radius1;      /* radius of refinement along v_perp=0 */
  PetscReal      vperp0_radius2;      /* radius of refinement along v_perp=0 after origin AMR refinement */
  PetscBool      sphere;
  PetscBool      inflate;
  PetscInt       numRERefine;       /* refinement along v_perp=0, z > 0 */
  PetscInt       nZRefine1;          /* origin refinement after v_perp=0 refinement */
  PetscInt       nZRefine2;          /* origin refinement after origin AMR refinement */
  PetscInt       maxRefIts;         /* normal AMR - refine from origin */
  PetscInt       postAMRRefine;     /* uniform refinement of AMR */
  /* discretization - AMR */
  PetscErrorCode (*errorIndicator)(PetscInt, PetscReal, PetscReal [], PetscInt, const PetscInt[], const PetscScalar[], const PetscScalar[], PetscReal *, void *);
  PetscReal      refineTol[LANDAU_MAX_SPECIES];
  PetscReal      coarsenTol[LANDAU_MAX_SPECIES];
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
  PetscReal      electronShift; /* for tests */
  PetscInt       num_species;
  /* diagnostics */
  PetscInt       verbose;
  PetscLogEvent  events[20];
  DM             dmv;
  /* cache */
  Mat            J;
  Mat            M;
  Vec            X;
  PetscReal      normJ; /* used to see if function changed */
  /* derived type */
  void          *data;
  PetscBool      aux_bool;  /* helper */
  /* computing */
  LandauDeviceType deviceType;
  PetscInt       subThreadBlockSize;
} LandauCtx;

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
  struct _lP4estVertexMaps*data;
  void                    *vp1,*vp2,*vp3;
} P4estVertexMaps;

typedef PetscReal LandauIPReal;
typedef struct {
  LandauIPReal  *w;
  LandauIPReal  *x;
  LandauIPReal  *y;
  LandauIPReal  *z;
  LandauIPReal  *coefs;
  int           dim_,ns_,nip_;
} LandauIPData;

PETSC_EXTERN PetscErrorCode LandauCreateColoring(Mat, DM, PetscContainer *);
PETSC_EXTERN PetscErrorCode LandauFormJacobian_Internal(Vec, Mat, const PetscInt, void *);
PETSC_EXTERN int LandauGetIPDataSize(const LandauIPData *const);
#if defined(PETSC_HAVE_CUDA)
PETSC_EXTERN PetscErrorCode LandauCUDAJacobian(DM, const PetscInt, const PetscReal [], const PetscReal [], const PetscReal[], const PetscReal[],
  const LandauIPData *const, const PetscReal [], const PetscLogEvent[], Mat);
PETSC_EXTERN PetscErrorCode LandauCUDACreateMatMaps(P4estVertexMaps *, pointInterpolationP4est (*)[LANDAU_MAX_Q_FACE], PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode LandauCUDADestroyMatMaps(P4estVertexMaps *);

#endif
#if defined(PETSC_HAVE_KOKKOS)
  /* TODO: this won't work if PETSc is built with C++ */
#if !defined(__cplusplus)
PETSC_EXTERN PetscErrorCode LandauKokkosJacobian(DM, const PetscInt, const PetscReal [], const PetscReal [], const PetscReal[], const PetscReal[],
                                                 const LandauIPData *const, const PetscReal [],const PetscInt, const PetscLogEvent[], Mat);
PETSC_EXTERN PetscErrorCode LandauKokkosCreateMatMaps(P4estVertexMaps *, pointInterpolationP4est (*)[LANDAU_MAX_Q_FACE], PetscInt, PetscInt);
PETSC_EXTERN PetscErrorCode LandauKokkosDestroyMatMaps(P4estVertexMaps *);
#endif
#endif

#endif /* PETSCLANDAU_H */
