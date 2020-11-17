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
#if !defined(LANDAU_MAX_NQ)
#if LANDAU_DIM==2
#define LANDAU_MAX_NQ 25
#else
#define LANDAU_MAX_NQ 27
#endif
#endif
typedef enum {LANDAU_CUDA, LANDAU_KOKKOS, LANDAU_CPU} LandauDeviceType;
typedef struct {
  PetscBool      interpolate;                  /* Generate intermediate mesh elements */
  PetscBool      simplex;
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

typedef PetscReal LandauIPReal;
typedef struct {
  LandauIPReal  *w_data;
  LandauIPReal  *x;
  LandauIPReal  *y;
  LandauIPReal  *z;
  LandauIPReal  *f;
  LandauIPReal  *dfx;
  LandauIPReal  *dfy;
  LandauIPReal  *dfz;
  int            dim_,ns_,nip_;
} LandauIPData;

PETSC_EXTERN PetscErrorCode LandauAssembleOpenMP(PetscInt cStart, PetscInt cEnd, PetscInt totDim, DM plex, PetscSection section, PetscSection globalSection, Mat JacP, PetscScalar elemMats[], PetscContainer container);
PETSC_EXTERN PetscErrorCode LandauCreateColoring(Mat, DM, PetscContainer *);
PETSC_EXTERN PetscErrorCode LandauFormJacobian_Internal(Vec, Mat, const PetscInt, void *);
PETSC_EXTERN int LandauGetIPDataSize(const LandauIPData *const);
#if defined(PETSC_HAVE_CUDA)
  PETSC_EXTERN PetscErrorCode LandauCUDAJacobian(DM, const PetscInt, const PetscReal [], const PetscReal [], const PetscReal[], const PetscReal[],
                                                 const LandauIPData *const, const PetscReal [],const PetscInt, const PetscLogEvent[], Mat);
#endif
#if defined(PETSC_HAVE_KOKKOS)
  /* TODO: this won't work if PETSc is built with C++ */
#if !defined(__cplusplus)
PETSC_EXTERN PetscErrorCode LandauKokkosJacobian(DM, const PetscInt, const PetscReal [], const PetscReal [], const PetscReal[], const PetscReal[],
                                                 const LandauIPData *const, const PetscReal [],const PetscInt, const PetscLogEvent[], Mat);
#endif
#endif

#endif /* PETSCLANDAU_H */
