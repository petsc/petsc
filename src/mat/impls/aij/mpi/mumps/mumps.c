/*
    Provides an interface to the MUMPS sparse solver
*/
#include <petscpkg_version.h>
#include <petscsf.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h> /*I  "petscmat.h"  I*/
#include <../src/mat/impls/sbaij/mpi/mpisbaij.h>
#include <../src/mat/impls/sell/mpi/mpisell.h>
#include <petsc/private/vecimpl.h>

#define MUMPS_MANUALS "(see users manual https://mumps-solver.org/index.php?page=doc \"Error and warning diagnostics\")"

EXTERN_C_BEGIN
#if defined(PETSC_HAVE_MUMPS_MIXED_PRECISION)
  #include <cmumps_c.h>
  #include <zmumps_c.h>
  #include <smumps_c.h>
  #include <dmumps_c.h>
#else
  #if defined(PETSC_USE_COMPLEX)
    #if defined(PETSC_USE_REAL_SINGLE)
      #include <cmumps_c.h>
      #define MUMPS_c     cmumps_c
      #define MumpsScalar CMUMPS_COMPLEX
    #else
      #include <zmumps_c.h>
      #define MUMPS_c     zmumps_c
      #define MumpsScalar ZMUMPS_COMPLEX
    #endif
  #else
    #if defined(PETSC_USE_REAL_SINGLE)
      #include <smumps_c.h>
      #define MUMPS_c     smumps_c
      #define MumpsScalar SMUMPS_REAL
    #else
      #include <dmumps_c.h>
      #define MUMPS_c     dmumps_c
      #define MumpsScalar DMUMPS_REAL
    #endif
  #endif
#endif
#if defined(PETSC_USE_COMPLEX)
  #if defined(PETSC_USE_REAL_SINGLE)
    #define MUMPS_STRUC_C CMUMPS_STRUC_C
  #else
    #define MUMPS_STRUC_C ZMUMPS_STRUC_C
  #endif
#else
  #if defined(PETSC_USE_REAL_SINGLE)
    #define MUMPS_STRUC_C SMUMPS_STRUC_C
  #else
    #define MUMPS_STRUC_C DMUMPS_STRUC_C
  #endif
#endif
EXTERN_C_END

#define JOB_INIT         -1
#define JOB_NULL         0
#define JOB_FACTSYMBOLIC 1
#define JOB_FACTNUMERIC  2
#define JOB_SOLVE        3
#define JOB_END          -2

/* MUMPS uses MUMPS_INT for nonzero indices such as irn/jcn, irn_loc/jcn_loc and uses int64_t for
   number of nonzeros such as nnz, nnz_loc. We typedef MUMPS_INT to PetscMUMPSInt to follow the
   naming convention in PetscMPIInt, PetscBLASInt etc.
*/
typedef MUMPS_INT PetscMUMPSInt;

#if PETSC_PKG_MUMPS_VERSION_GE(5, 3, 0)
  #if defined(MUMPS_INTSIZE64) /* MUMPS_INTSIZE64 is in MUMPS headers if it is built in full 64-bit mode, therefore the macro is more reliable */
    #error "PETSc has not been tested with full 64-bit MUMPS and we choose to error out"
  #endif
#else
  #if defined(INTSIZE64) /* INTSIZE64 is a command line macro one used to build MUMPS in full 64-bit mode */
    #error "PETSc has not been tested with full 64-bit MUMPS and we choose to error out"
  #endif
#endif

#define MPIU_MUMPSINT       MPI_INT
#define PETSC_MUMPS_INT_MAX 2147483647
#define PETSC_MUMPS_INT_MIN -2147483648

/* Cast PetscInt to PetscMUMPSInt. Usually there is no overflow since <a> is row/col indices or some small integers*/
static inline PetscErrorCode PetscMUMPSIntCast(PetscCount a, PetscMUMPSInt *b)
{
  PetscFunctionBegin;
#if PetscDefined(USE_64BIT_INDICES)
  PetscAssert(a <= PETSC_MUMPS_INT_MAX && a >= PETSC_MUMPS_INT_MIN, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "PetscInt too long for PetscMUMPSInt");
#endif
  *b = (PetscMUMPSInt)a;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Put these utility routines here since they are only used in this file */
static inline PetscErrorCode PetscOptionsMUMPSInt_Private(PetscOptionItems PetscOptionsObject, const char opt[], const char text[], const char man[], PetscMUMPSInt currentvalue, PetscMUMPSInt *value, PetscBool *set, PetscMUMPSInt lb, PetscMUMPSInt ub)
{
  PetscInt  myval;
  PetscBool myset;

  PetscFunctionBegin;
  /* PetscInt's size should be always >= PetscMUMPSInt's. It is safe to call PetscOptionsInt_Private to read a PetscMUMPSInt */
  PetscCall(PetscOptionsInt_Private(PetscOptionsObject, opt, text, man, (PetscInt)currentvalue, &myval, &myset, lb, ub));
  if (myset) PetscCall(PetscMUMPSIntCast(myval, value));
  if (set) *set = myset;
  PetscFunctionReturn(PETSC_SUCCESS);
}
#define PetscOptionsMUMPSInt(a, b, c, d, e, f) PetscOptionsMUMPSInt_Private(PetscOptionsObject, a, b, c, d, e, f, PETSC_MUMPS_INT_MIN, PETSC_MUMPS_INT_MAX)

// An abstract type for specific MUMPS types {S,D,C,Z}MUMPS_STRUC_C.
//
// With the abstract (outer) type, we can write shared code. We call MUMPS through a type-to-be-determined inner field within the abstract type.
// Before/after calling MUMPS, we need to copy in/out fields between the outer and the inner, which seems expensive. But note that the large fixed size
// arrays within the types are directly linked. At the end, we only need to copy ~20 intergers/pointers, which is doable. See PreMumpsCall()/PostMumpsCall().
//
// Not all fields in the specific types are exposed in the abstract type. We only need those used by the PETSc/MUMPS interface.
// Notably, DMUMPS_COMPLEX* and DMUMPS_REAL* fields are now declared as void *. Their type will be determined by the the actual precision to be used.
// Also note that we added some *_len fields not in specific types to track sizes of those MumpsScalar buffers.
typedef struct {
  PetscPrecision precision;   // precision used by MUMPS
  void          *internal_id; // the data structure passed to MUMPS, whose actual type {S,D,C,Z}MUMPS_STRUC_C is to be decided by precision and PETSc's use of complex

  // aliased fields from internal_id, so that we can use XMUMPS_STRUC_C to write shared code across different precisions.
  MUMPS_INT  sym, par, job;
  MUMPS_INT  comm_fortran; /* Fortran communicator */
  MUMPS_INT *icntl;
  void      *cntl; // MumpsReal, fixed size array
  MUMPS_INT  n;
  MUMPS_INT  nblk;

  /* Assembled entry */
  MUMPS_INT8 nnz;
  MUMPS_INT *irn;
  MUMPS_INT *jcn;
  void      *a; // MumpsScalar, centralized input
  PetscCount a_len;

  /* Distributed entry */
  MUMPS_INT8 nnz_loc;
  MUMPS_INT *irn_loc;
  MUMPS_INT *jcn_loc;
  void      *a_loc; // MumpsScalar, distributed input
  PetscCount a_loc_len;

  /* Matrix by blocks */
  MUMPS_INT *blkptr;
  MUMPS_INT *blkvar;

  /* Ordering, if given by user */
  MUMPS_INT *perm_in;

  /* RHS, solution, ouptput data and statistics */
  void      *rhs, *redrhs, *rhs_sparse, *sol_loc, *rhs_loc;                 // MumpsScalar buffers
  PetscCount rhs_len, redrhs_len, rhs_sparse_len, sol_loc_len, rhs_loc_len; // length of buffers (in MumpsScalar) IF allocated in a different precision than PetscScalar

  MUMPS_INT *irhs_sparse, *irhs_ptr, *isol_loc, *irhs_loc;
  MUMPS_INT  nrhs, lrhs, lredrhs, nz_rhs, lsol_loc, nloc_rhs, lrhs_loc;
  // MUMPS_INT  nsol_loc; // introduced in MUMPS-5.7, but PETSc doesn't use it; would cause compile errors with the widely used 5.6. If you add it, must also update PreMumpsCall() and guard this with #if PETSC_PKG_MUMPS_VERSION_GE(5, 7, 0)
  MUMPS_INT  schur_lld;
  MUMPS_INT *info, *infog;   // fixed size array
  void      *rinfo, *rinfog; // MumpsReal, fixed size array

  /* Null space */
  MUMPS_INT *pivnul_list; // allocated by MUMPS!
  MUMPS_INT *mapping;     // allocated by MUMPS!

  /* Schur */
  MUMPS_INT  size_schur;
  MUMPS_INT *listvar_schur;
  void      *schur; // MumpsScalar
  PetscCount schur_len;

  /* For out-of-core */
  char *ooc_tmpdir; // fixed size array
  char *ooc_prefix; // fixed size array
} XMUMPS_STRUC_C;

// Note: fixed-size arrays are allocated by MUMPS; redirect them to the outer struct
#define AllocateInternalID(MUMPS_STRUC_T, outer) \
  do { \
    MUMPS_STRUC_T *inner; \
    PetscCall(PetscNew(&inner)); \
    outer->icntl      = inner->icntl; \
    outer->cntl       = inner->cntl; \
    outer->info       = inner->info; \
    outer->infog      = inner->infog; \
    outer->rinfo      = inner->rinfo; \
    outer->rinfog     = inner->rinfog; \
    outer->ooc_tmpdir = inner->ooc_tmpdir; \
    outer->ooc_prefix = inner->ooc_prefix; \
    /* the three field should never change after init */ \
    inner->comm_fortran = outer->comm_fortran; \
    inner->par          = outer->par; \
    inner->sym          = outer->sym; \
    outer->internal_id  = inner; \
  } while (0)

// Allocate the internal [SDCZ]MUMPS_STRUC_C ID data structure in the given <precision>, and link fields of the outer and the inner
static inline PetscErrorCode MatMumpsAllocateInternalID(XMUMPS_STRUC_C *outer, PetscPrecision precision)
{
  PetscFunctionBegin;
  outer->precision = precision;
#if defined(PETSC_HAVE_MUMPS_MIXED_PRECISION)
  #if defined(PETSC_USE_COMPLEX)
  if (precision == PETSC_PRECISION_SINGLE) AllocateInternalID(CMUMPS_STRUC_C, outer);
  else AllocateInternalID(ZMUMPS_STRUC_C, outer);
  #else
  if (precision == PETSC_PRECISION_SINGLE) AllocateInternalID(SMUMPS_STRUC_C, outer);
  else AllocateInternalID(DMUMPS_STRUC_C, outer);
  #endif
#else
  AllocateInternalID(MUMPS_STRUC_C, outer);
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define FreeInternalIDFields(MUMPS_STRUC_T, outer) \
  do { \
    MUMPS_STRUC_T *inner = (MUMPS_STRUC_T *)(outer)->internal_id; \
    PetscCall(PetscFree(inner->a)); \
    PetscCall(PetscFree(inner->a_loc)); \
    PetscCall(PetscFree(inner->redrhs)); \
    PetscCall(PetscFree(inner->rhs)); \
    PetscCall(PetscFree(inner->rhs_sparse)); \
    PetscCall(PetscFree(inner->rhs_loc)); \
    PetscCall(PetscFree(inner->sol_loc)); \
    PetscCall(PetscFree(inner->schur)); \
  } while (0)

static inline PetscErrorCode MatMumpsFreeInternalID(XMUMPS_STRUC_C *outer)
{
  PetscFunctionBegin;
  if (outer->internal_id) { // sometimes, the inner is never created before we destroy the outer
#if defined(PETSC_HAVE_MUMPS_MIXED_PRECISION)
    const PetscPrecision mumps_precision = outer->precision;
    if (mumps_precision != PETSC_SCALAR_PRECISION) { // Free internal buffers if we used mixed precision
  #if defined(PETSC_USE_COMPLEX)
      if (mumps_precision == PETSC_PRECISION_SINGLE) FreeInternalIDFields(CMUMPS_STRUC_C, outer);
      else FreeInternalIDFields(ZMUMPS_STRUC_C, outer);
  #else
      if (mumps_precision == PETSC_PRECISION_SINGLE) FreeInternalIDFields(SMUMPS_STRUC_C, outer);
      else FreeInternalIDFields(DMUMPS_STRUC_C, outer);
  #endif
    }
#endif
    PetscCall(PetscFree(outer->internal_id));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Make a companion MumpsScalar array (with a given PetscScalar array), to hold at least <n> MumpsScalars in the given <precision> and return the address at <ma>.
// <convert> indicates if we need to convert PetscScalars to MumpsScalars after allocating the MumpsScalar array.
// (For bravity, we use <ma> for array address and <m> for its length in MumpsScalar, though in code they should be <*ma> and <*m>)
// If <ma> already points to a buffer/array, on input <m> should be its length. Note the buffer might be freed if it is not big enough for this request.
//
// The returned array is a companion, so how it is created depends on if PetscScalar and MumpsScalar are the same.
// 1) If they are different, a separate array will be made and its length and address will be provided at <m> and <ma> on output.
// 2) Otherwise, <pa> will be returned in <ma>, and <m> will be zero on output.
//
//
//   Input parameters:
// + convert   - whether to do PetscScalar to MumpsScalar conversion
// . n         - length of the PetscScalar array
// . pa        - [n]], points to the PetscScalar array
// . precision - precision of MumpsScalar
// . m         - on input, length of an existing MumpsScalar array <ma> if any, otherwise *m is just zero.
// - ma        - on input, an existing MumpsScalar array if any.
//
//   Output parameters:
// + m  - length of the MumpsScalar buffer at <ma> if MumpsScalar is different from PetscScalar, otherwise 0
// . ma - the MumpsScalar array, which could be an alias of <pa> when the two types are the same.
//
//   Note:
//    New memory, if allocated, is done via PetscMalloc1(), and is owned by caller.
static PetscErrorCode MatMumpsMakeMumpsScalarArray(PetscBool convert, PetscCount n, const PetscScalar *pa, PetscPrecision precision, PetscCount *m, void **ma)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_MUMPS_MIXED_PRECISION)
  const PetscPrecision mumps_precision = precision;
  PetscCheck(precision == PETSC_PRECISION_SINGLE || precision == PETSC_PRECISION_DOUBLE, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unsupported precicison (%d). Must be single or double", (int)precision);
  #if defined(PETSC_USE_COMPLEX)
  if (mumps_precision != PETSC_SCALAR_PRECISION) {
    if (mumps_precision == PETSC_PRECISION_SINGLE) {
      if (*m < n) {
        PetscCall(PetscFree(*ma));
        PetscCall(PetscMalloc1(n, (CMUMPS_COMPLEX **)ma));
        *m = n;
      }
      if (convert) {
        CMUMPS_COMPLEX *b = *(CMUMPS_COMPLEX **)ma;
        for (PetscCount i = 0; i < n; i++) {
          b[i].r = PetscRealPart(pa[i]);
          b[i].i = PetscImaginaryPart(pa[i]);
        }
      }
    } else {
      if (*m < n) {
        PetscCall(PetscFree(*ma));
        PetscCall(PetscMalloc1(n, (ZMUMPS_COMPLEX **)ma));
        *m = n;
      }
      if (convert) {
        ZMUMPS_COMPLEX *b = *(ZMUMPS_COMPLEX **)ma;
        for (PetscCount i = 0; i < n; i++) {
          b[i].r = PetscRealPart(pa[i]);
          b[i].i = PetscImaginaryPart(pa[i]);
        }
      }
    }
  }
  #else
  if (mumps_precision != PETSC_SCALAR_PRECISION) {
    if (mumps_precision == PETSC_PRECISION_SINGLE) {
      if (*m < n) {
        PetscCall(PetscFree(*ma));
        PetscCall(PetscMalloc1(n, (SMUMPS_REAL **)ma));
        *m = n;
      }
      if (convert) {
        SMUMPS_REAL *b = *(SMUMPS_REAL **)ma;
        for (PetscCount i = 0; i < n; i++) b[i] = pa[i];
      }
    } else {
      if (*m < n) {
        PetscCall(PetscFree(*ma));
        PetscCall(PetscMalloc1(n, (DMUMPS_REAL **)ma));
        *m = n;
      }
      if (convert) {
        DMUMPS_REAL *b = *(DMUMPS_REAL **)ma;
        for (PetscCount i = 0; i < n; i++) b[i] = pa[i];
      }
    }
  }
  #endif
  else
#endif
  {
    if (*m != 0) PetscCall(PetscFree(*ma)); // free existing buffer if any
    *ma = (void *)pa;                       // same precision, make them alias
    *m  = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Cast a MumpsScalar array <ma[n]> in <mumps_precision> to a PetscScalar array at address <pa>.
//
// 1) If the two types are different, cast array elements.
// 2) Otherwise, this works as a memcpy; of course, if the two addresses are equal, it is a no-op.
static PetscErrorCode MatMumpsCastMumpsScalarArray(PetscCount n, PetscPrecision mumps_precision, const void *ma, PetscScalar *pa)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_MUMPS_MIXED_PRECISION)
  if (mumps_precision != PETSC_SCALAR_PRECISION) {
  #if defined(PETSC_USE_COMPLEX)
    if (mumps_precision == PETSC_PRECISION_SINGLE) {
      PetscReal         *a = (PetscReal *)pa;
      const SMUMPS_REAL *b = (const SMUMPS_REAL *)ma;
      for (PetscCount i = 0; i < 2 * n; i++) a[i] = b[i];
    } else {
      PetscReal         *a = (PetscReal *)pa;
      const DMUMPS_REAL *b = (const DMUMPS_REAL *)ma;
      for (PetscCount i = 0; i < 2 * n; i++) a[i] = b[i];
    }
  #else
    if (mumps_precision == PETSC_PRECISION_SINGLE) {
      const SMUMPS_REAL *b = (const SMUMPS_REAL *)ma;
      for (PetscCount i = 0; i < n; i++) pa[i] = b[i];
    } else {
      const DMUMPS_REAL *b = (const DMUMPS_REAL *)ma;
      for (PetscCount i = 0; i < n; i++) pa[i] = b[i];
    }
  #endif
  } else
#endif
    PetscCall(PetscArraycpy((PetscScalar *)pa, (PetscScalar *)ma, n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Cast a PetscScalar array <pa[n]> to a MumpsScalar array in the given <mumps_precision> at address <ma>.
//
// 1) If the two types are different, cast array elements.
// 2) Otherwise, this works as a memcpy; of course, if the two addresses are equal, it is a no-op.
static PetscErrorCode MatMumpsCastPetscScalarArray(PetscCount n, const PetscScalar *pa, PetscPrecision mumps_precision, const void *ma)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_MUMPS_MIXED_PRECISION)
  if (mumps_precision != PETSC_SCALAR_PRECISION) {
  #if defined(PETSC_USE_COMPLEX)
    if (mumps_precision == PETSC_PRECISION_SINGLE) {
      CMUMPS_COMPLEX *b = (CMUMPS_COMPLEX *)ma;
      for (PetscCount i = 0; i < n; i++) {
        b[i].r = PetscRealPart(pa[i]);
        b[i].i = PetscImaginaryPart(pa[i]);
      }
    } else {
      ZMUMPS_COMPLEX *b = (ZMUMPS_COMPLEX *)ma;
      for (PetscCount i = 0; i < n; i++) {
        b[i].r = PetscRealPart(pa[i]);
        b[i].i = PetscImaginaryPart(pa[i]);
      }
    }
  #else
    if (mumps_precision == PETSC_PRECISION_SINGLE) {
      SMUMPS_REAL *b = (SMUMPS_REAL *)ma;
      for (PetscCount i = 0; i < n; i++) b[i] = pa[i];
    } else {
      DMUMPS_REAL *b = (DMUMPS_REAL *)ma;
      for (PetscCount i = 0; i < n; i++) b[i] = pa[i];
    }
  #endif
  } else
#endif
    PetscCall(PetscArraycpy((PetscScalar *)ma, (PetscScalar *)pa, n));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static inline MPI_Datatype MPIU_MUMPSREAL(const XMUMPS_STRUC_C *id)
{
  return id->precision == PETSC_PRECISION_DOUBLE ? MPI_DOUBLE : MPI_FLOAT;
}

#define PreMumpsCall(inner, outer, mumpsscalar) \
  do { \
    inner->job           = outer->job; \
    inner->n             = outer->n; \
    inner->nblk          = outer->nblk; \
    inner->nnz           = outer->nnz; \
    inner->irn           = outer->irn; \
    inner->jcn           = outer->jcn; \
    inner->a             = (mumpsscalar *)outer->a; \
    inner->nnz_loc       = outer->nnz_loc; \
    inner->irn_loc       = outer->irn_loc; \
    inner->jcn_loc       = outer->jcn_loc; \
    inner->a_loc         = (mumpsscalar *)outer->a_loc; \
    inner->blkptr        = outer->blkptr; \
    inner->blkvar        = outer->blkvar; \
    inner->perm_in       = outer->perm_in; \
    inner->rhs           = (mumpsscalar *)outer->rhs; \
    inner->redrhs        = (mumpsscalar *)outer->redrhs; \
    inner->rhs_sparse    = (mumpsscalar *)outer->rhs_sparse; \
    inner->sol_loc       = (mumpsscalar *)outer->sol_loc; \
    inner->rhs_loc       = (mumpsscalar *)outer->rhs_loc; \
    inner->irhs_sparse   = outer->irhs_sparse; \
    inner->irhs_ptr      = outer->irhs_ptr; \
    inner->isol_loc      = outer->isol_loc; \
    inner->irhs_loc      = outer->irhs_loc; \
    inner->nrhs          = outer->nrhs; \
    inner->lrhs          = outer->lrhs; \
    inner->lredrhs       = outer->lredrhs; \
    inner->nz_rhs        = outer->nz_rhs; \
    inner->lsol_loc      = outer->lsol_loc; \
    inner->nloc_rhs      = outer->nloc_rhs; \
    inner->lrhs_loc      = outer->lrhs_loc; \
    inner->schur_lld     = outer->schur_lld; \
    inner->size_schur    = outer->size_schur; \
    inner->listvar_schur = outer->listvar_schur; \
    inner->schur         = (mumpsscalar *)outer->schur; \
  } while (0)

#define PostMumpsCall(inner, outer) \
  do { \
    outer->pivnul_list = inner->pivnul_list; \
    outer->mapping     = inner->mapping; \
  } while (0)

// Entry for PETSc to call mumps
static inline PetscErrorCode PetscCallMumps_Private(XMUMPS_STRUC_C *outer)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_MUMPS_MIXED_PRECISION)
  #if defined(PETSC_USE_COMPLEX)
  if (outer->precision == PETSC_PRECISION_SINGLE) {
    CMUMPS_STRUC_C *inner = (CMUMPS_STRUC_C *)outer->internal_id;
    PreMumpsCall(inner, outer, CMUMPS_COMPLEX);
    PetscStackCallExternalVoid("cmumps_c", cmumps_c(inner));
    PostMumpsCall(inner, outer);
  } else {
    ZMUMPS_STRUC_C *inner = (ZMUMPS_STRUC_C *)outer->internal_id;
    PreMumpsCall(inner, outer, ZMUMPS_COMPLEX);
    PetscStackCallExternalVoid("zmumps_c", zmumps_c(inner));
    PostMumpsCall(inner, outer);
  }
  #else
  if (outer->precision == PETSC_PRECISION_SINGLE) {
    SMUMPS_STRUC_C *inner = (SMUMPS_STRUC_C *)outer->internal_id;
    PreMumpsCall(inner, outer, SMUMPS_REAL);
    PetscStackCallExternalVoid("smumps_c", smumps_c(inner));
    PostMumpsCall(inner, outer);
  } else {
    DMUMPS_STRUC_C *inner = (DMUMPS_STRUC_C *)outer->internal_id;
    PreMumpsCall(inner, outer, DMUMPS_REAL);
    PetscStackCallExternalVoid("dmumps_c", dmumps_c(inner));
    PostMumpsCall(inner, outer);
  }
  #endif
#else
  MUMPS_STRUC_C *inner = (MUMPS_STRUC_C *)outer->internal_id;
  PreMumpsCall(inner, outer, MumpsScalar);
  PetscStackCallExternalVoid(PetscStringize(MUMPS_c), MUMPS_c(inner));
  PostMumpsCall(inner, outer);
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* macros s.t. indices match MUMPS documentation */
#define ICNTL(I) icntl[(I) - 1]
#define INFOG(I) infog[(I) - 1]
#define INFO(I)  info[(I) - 1]

// Get a value from a MumpsScalar array, which is the <F> field in the struct of MUMPS_STRUC_C. The value is convertible to PetscScalar. Note no minus 1 on I!
#if defined(PETSC_USE_COMPLEX)
  #define ID_FIELD_GET(ID, F, I) ((ID).precision == PETSC_PRECISION_SINGLE ? ((CMUMPS_COMPLEX *)(ID).F)[I].r + PETSC_i * ((CMUMPS_COMPLEX *)(ID).F)[I].i : ((ZMUMPS_COMPLEX *)(ID).F)[I].r + PETSC_i * ((ZMUMPS_COMPLEX *)(ID).F)[I].i)
#else
  #define ID_FIELD_GET(ID, F, I) ((ID).precision == PETSC_PRECISION_SINGLE ? ((float *)(ID).F)[I] : ((double *)(ID).F)[I])
#endif

// Get a value from MumpsReal arrays. The value is convertible to PetscReal.
#define ID_CNTL_GET(ID, I)   ((ID).precision == PETSC_PRECISION_SINGLE ? ((float *)(ID).cntl)[(I) - 1] : ((double *)(ID).cntl)[(I) - 1])
#define ID_RINFOG_GET(ID, I) ((ID).precision == PETSC_PRECISION_SINGLE ? ((float *)(ID).rinfog)[(I) - 1] : ((double *)(ID).rinfog)[(I) - 1])
#define ID_RINFO_GET(ID, I)  ((ID).precision == PETSC_PRECISION_SINGLE ? ((float *)(ID).rinfo)[(I) - 1] : ((double *)(ID).rinfo)[(I) - 1])

// Set the I-th entry of the MumpsReal array id.cntl[] with a PetscReal <VAL>
#define ID_CNTL_SET(ID, I, VAL) \
  do { \
    if ((ID).precision == PETSC_PRECISION_SINGLE) ((float *)(ID).cntl)[(I) - 1] = (VAL); \
    else ((double *)(ID).cntl)[(I) - 1] = (VAL); \
  } while (0)

/* if using PETSc OpenMP support, we only call MUMPS on master ranks. Before/after the call, we change/restore CPUs the master ranks can run on */
#if defined(PETSC_HAVE_OPENMP_SUPPORT)
  #define PetscMUMPS_c(mumps) \
    do { \
      if (mumps->use_petsc_omp_support) { \
        if (mumps->is_omp_master) { \
          PetscCall(PetscOmpCtrlOmpRegionOnMasterBegin(mumps->omp_ctrl)); \
          PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF)); \
          PetscCall(PetscCallMumps_Private(&mumps->id)); \
          PetscCall(PetscFPTrapPop()); \
          PetscCall(PetscOmpCtrlOmpRegionOnMasterEnd(mumps->omp_ctrl)); \
        } \
        PetscCall(PetscOmpCtrlBarrier(mumps->omp_ctrl)); \
        /* Global info is same on all processes so we Bcast it within omp_comm. Local info is specific      \
         to processes, so we only Bcast info[1], an error code and leave others (since they do not have   \
         an easy translation between omp_comm and petsc_comm). See MUMPS-5.1.2 manual p82.                   \
         omp_comm is a small shared memory communicator, hence doing multiple Bcast as shown below is OK. \
      */ \
        MUMPS_STRUC_C tmp; /* All MUMPS_STRUC_C types have same lengths on these info arrays */ \
        PetscCallMPI(MPI_Bcast(mumps->id.infog, PETSC_STATIC_ARRAY_LENGTH(tmp.infog), MPIU_MUMPSINT, 0, mumps->omp_comm)); \
        PetscCallMPI(MPI_Bcast(mumps->id.info, PETSC_STATIC_ARRAY_LENGTH(tmp.info), MPIU_MUMPSINT, 0, mumps->omp_comm)); \
        PetscCallMPI(MPI_Bcast(mumps->id.rinfog, PETSC_STATIC_ARRAY_LENGTH(tmp.rinfog), MPIU_MUMPSREAL(&mumps->id), 0, mumps->omp_comm)); \
        PetscCallMPI(MPI_Bcast(mumps->id.rinfo, PETSC_STATIC_ARRAY_LENGTH(tmp.rinfo), MPIU_MUMPSREAL(&mumps->id), 0, mumps->omp_comm)); \
      } else { \
        PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF)); \
        PetscCall(PetscCallMumps_Private(&mumps->id)); \
        PetscCall(PetscFPTrapPop()); \
      } \
    } while (0)
#else
  #define PetscMUMPS_c(mumps) \
    do { \
      PetscCall(PetscFPTrapPush(PETSC_FP_TRAP_OFF)); \
      PetscCall(PetscCallMumps_Private(&mumps->id)); \
      PetscCall(PetscFPTrapPop()); \
    } while (0)
#endif

typedef struct Mat_MUMPS Mat_MUMPS;
struct Mat_MUMPS {
  XMUMPS_STRUC_C id;

  MatStructure   matstruc;
  PetscMPIInt    myid, petsc_size;
  PetscMUMPSInt *irn, *jcn;       /* the (i,j,v) triplets passed to mumps. */
  PetscScalar   *val, *val_alloc; /* For some matrices, we can directly access their data array without a buffer. For others, we need a buffer. So comes val_alloc. */
  PetscCount     nnz;             /* number of nonzeros. The type is called selective 64-bit in mumps */
  PetscMUMPSInt  sym;
  MPI_Comm       mumps_comm;
  PetscMUMPSInt *ICNTL_pre;
  PetscReal     *CNTL_pre;
  PetscMUMPSInt  ICNTL9_pre;         /* check if ICNTL(9) is changed from previous MatSolve */
  VecScatter     scat_rhs, scat_sol; /* used by MatSolve() */
  PetscMUMPSInt  ICNTL20;            /* use centralized (0) or distributed (10) dense RHS */
  PetscMUMPSInt  ICNTL26;
  PetscMUMPSInt  lrhs_loc, nloc_rhs, *irhs_loc;
#if defined(PETSC_HAVE_OPENMP_SUPPORT)
  PetscInt    *rhs_nrow, max_nrhs;
  PetscMPIInt *rhs_recvcounts, *rhs_disps;
  PetscScalar *rhs_loc, *rhs_recvbuf;
#endif
  Vec            b_seq, x_seq;
  PetscInt       ninfo, *info; /* which INFO to display */
  PetscInt       sizeredrhs;
  PetscScalar   *schur_sol;
  PetscInt       schur_sizesol;
  PetscScalar   *redrhs;              // buffer in PetscScalar in case MumpsScalar is in a different precision
  PetscMUMPSInt *ia_alloc, *ja_alloc; /* work arrays used for the CSR struct for sparse rhs */
  PetscCount     cur_ilen, cur_jlen;  /* current len of ia_alloc[], ja_alloc[] */
  PetscErrorCode (*ConvertToTriples)(Mat, PetscInt, MatReuse, Mat_MUMPS *);

  /* Support for MATNEST */
  PetscErrorCode (**nest_convert_to_triples)(Mat, PetscInt, MatReuse, Mat_MUMPS *);
  PetscCount  *nest_vals_start;
  PetscScalar *nest_vals;

  /* stuff used by petsc/mumps OpenMP support*/
  PetscBool    use_petsc_omp_support;
  PetscOmpCtrl omp_ctrl;             /* an OpenMP controller that blocked processes will release their CPU (MPI_Barrier does not have this guarantee) */
  MPI_Comm     petsc_comm, omp_comm; /* petsc_comm is PETSc matrix's comm */
  PetscCount  *recvcount;            /* a collection of nnz on omp_master */
  PetscMPIInt  tag, omp_comm_size;
  PetscBool    is_omp_master; /* is this rank the master of omp_comm */
  MPI_Request *reqs;
};

/* Cast a 1-based CSR represented by (nrow, ia, ja) of type PetscInt to a CSR of type PetscMUMPSInt.
   Here, nrow is number of rows, ia[] is row pointer and ja[] is column indices.
 */
static PetscErrorCode PetscMUMPSIntCSRCast(PETSC_UNUSED Mat_MUMPS *mumps, PetscInt nrow, PetscInt *ia, PetscInt *ja, PetscMUMPSInt **ia_mumps, PetscMUMPSInt **ja_mumps, PetscMUMPSInt *nnz_mumps)
{
  PetscInt nnz = ia[nrow] - 1; /* mumps uses 1-based indices. Uses PetscInt instead of PetscCount since mumps only uses PetscMUMPSInt for rhs */

  PetscFunctionBegin;
#if defined(PETSC_USE_64BIT_INDICES)
  {
    PetscInt i;
    if (nrow + 1 > mumps->cur_ilen) { /* realloc ia_alloc/ja_alloc to fit ia/ja */
      PetscCall(PetscFree(mumps->ia_alloc));
      PetscCall(PetscMalloc1(nrow + 1, &mumps->ia_alloc));
      mumps->cur_ilen = nrow + 1;
    }
    if (nnz > mumps->cur_jlen) {
      PetscCall(PetscFree(mumps->ja_alloc));
      PetscCall(PetscMalloc1(nnz, &mumps->ja_alloc));
      mumps->cur_jlen = nnz;
    }
    for (i = 0; i < nrow + 1; i++) PetscCall(PetscMUMPSIntCast(ia[i], &mumps->ia_alloc[i]));
    for (i = 0; i < nnz; i++) PetscCall(PetscMUMPSIntCast(ja[i], &mumps->ja_alloc[i]));
    *ia_mumps = mumps->ia_alloc;
    *ja_mumps = mumps->ja_alloc;
  }
#else
  *ia_mumps = ia;
  *ja_mumps = ja;
#endif
  PetscCall(PetscMUMPSIntCast(nnz, nnz_mumps));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMumpsResetSchur_Private(Mat_MUMPS *mumps)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(mumps->id.listvar_schur));
  PetscCall(PetscFree(mumps->redrhs)); // if needed, id.redrhs will be freed in MatMumpsFreeInternalID()
  PetscCall(PetscFree(mumps->schur_sol));
  mumps->id.size_schur = 0;
  mumps->id.schur_lld  = 0;
  if (mumps->id.internal_id) mumps->id.ICNTL(19) = 0; // sometimes, the inner id is yet built
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* solve with rhs in mumps->id.redrhs and return in the same location */
static PetscErrorCode MatMumpsSolveSchur_Private(Mat F)
{
  Mat_MUMPS           *mumps = (Mat_MUMPS *)F->data;
  Mat                  S, B, X; // solve S*X = B; all three matrices are dense
  MatFactorSchurStatus schurstatus;
  PetscInt             sizesol;
  const PetscScalar   *xarray;

  PetscFunctionBegin;
  PetscCall(MatFactorFactorizeSchurComplement(F));
  PetscCall(MatFactorGetSchurComplement(F, &S, &schurstatus));
  PetscCall(MatMumpsCastMumpsScalarArray(mumps->sizeredrhs, mumps->id.precision, mumps->id.redrhs, mumps->redrhs));

  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, mumps->id.size_schur, mumps->id.nrhs, mumps->redrhs, &B));
  PetscCall(MatSetType(B, ((PetscObject)S)->type_name));
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
  PetscCall(MatBindToCPU(B, S->boundtocpu));
#endif
  switch (schurstatus) {
  case MAT_FACTOR_SCHUR_FACTORED:
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, mumps->id.size_schur, mumps->id.nrhs, mumps->redrhs, &X));
    PetscCall(MatSetType(X, ((PetscObject)S)->type_name));
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
    PetscCall(MatBindToCPU(X, S->boundtocpu));
#endif
    if (!mumps->id.ICNTL(9)) { /* transpose solve */
      PetscCall(MatMatSolveTranspose(S, B, X));
    } else {
      PetscCall(MatMatSolve(S, B, X));
    }
    break;
  case MAT_FACTOR_SCHUR_INVERTED:
    sizesol = mumps->id.nrhs * mumps->id.size_schur;
    if (!mumps->schur_sol || sizesol > mumps->schur_sizesol) {
      PetscCall(PetscFree(mumps->schur_sol));
      PetscCall(PetscMalloc1(sizesol, &mumps->schur_sol));
      mumps->schur_sizesol = sizesol;
    }
    PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, mumps->id.size_schur, mumps->id.nrhs, mumps->schur_sol, &X));
    PetscCall(MatSetType(X, ((PetscObject)S)->type_name));
#if defined(PETSC_HAVE_VIENNACL) || defined(PETSC_HAVE_CUDA)
    PetscCall(MatBindToCPU(X, S->boundtocpu));
#endif
    PetscCall(MatProductCreateWithMat(S, B, NULL, X));
    if (!mumps->id.ICNTL(9)) { /* transpose solve */
      PetscCall(MatProductSetType(X, MATPRODUCT_AtB));
    } else {
      PetscCall(MatProductSetType(X, MATPRODUCT_AB));
    }
    PetscCall(MatProductSetFromOptions(X));
    PetscCall(MatProductSymbolic(X));
    PetscCall(MatProductNumeric(X));

    PetscCall(MatCopy(X, B, SAME_NONZERO_PATTERN));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)F), PETSC_ERR_SUP, "Unhandled MatFactorSchurStatus %d", F->schur_status);
  }
  // MUST get the array from X (not B), though they share the same host array. We can only guarantee X has the correct data on device.
  PetscCall(MatDenseGetArrayRead(X, &xarray)); // xarray should be mumps->redrhs, but using MatDenseGetArrayRead is safer with GPUs.
  PetscCall(MatMumpsCastPetscScalarArray(mumps->sizeredrhs, xarray, mumps->id.precision, mumps->id.redrhs));
  PetscCall(MatDenseRestoreArrayRead(X, &xarray));
  PetscCall(MatFactorRestoreSchurComplement(F, &S, schurstatus));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMumpsHandleSchur_Private(Mat F, PetscBool expansion)
{
  Mat_MUMPS *mumps = (Mat_MUMPS *)F->data;

  PetscFunctionBegin;
  if (!mumps->id.ICNTL(19)) { /* do nothing when Schur complement has not been computed */
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (!expansion) { /* prepare for the condensation step */
    PetscInt sizeredrhs = mumps->id.nrhs * mumps->id.size_schur;
    /* allocate MUMPS internal array to store reduced right-hand sides */
    if (!mumps->id.redrhs || sizeredrhs > mumps->sizeredrhs) {
      mumps->id.lredrhs = mumps->id.size_schur;
      mumps->sizeredrhs = mumps->id.nrhs * mumps->id.lredrhs;
      if (mumps->id.redrhs_len) PetscCall(PetscFree(mumps->id.redrhs));
      PetscCall(PetscFree(mumps->redrhs));
      PetscCall(PetscMalloc1(mumps->sizeredrhs, &mumps->redrhs));
      PetscCall(MatMumpsMakeMumpsScalarArray(PETSC_FALSE, mumps->sizeredrhs, mumps->redrhs, mumps->id.precision, &mumps->id.redrhs_len, &mumps->id.redrhs));
    }
  } else {                                    /* prepare for the expansion step */
    PetscCall(MatMumpsSolveSchur_Private(F)); /* solve Schur complement, put solution in id.redrhs (this has to be done by the MUMPS user, so basically us) */
    mumps->id.ICNTL(26) = 2;                  /* expansion phase */
    PetscMUMPS_c(mumps);
    PetscCheck(mumps->id.INFOG(1) >= 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "MUMPS error in solve: INFOG(1)=%d, INFO(2)=%d " MUMPS_MANUALS, mumps->id.INFOG(1), mumps->id.INFO(2));
    /* restore defaults */
    mumps->id.ICNTL(26) = -1;
    /* free MUMPS internal array for redrhs if we have solved for multiple rhs in order to save memory space */
    if (mumps->id.nrhs > 1) {
      if (mumps->id.redrhs_len) PetscCall(PetscFree(mumps->id.redrhs));
      PetscCall(PetscFree(mumps->redrhs));
      mumps->id.redrhs_len = 0;
      mumps->id.lredrhs    = 0;
      mumps->sizeredrhs    = 0;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatConvertToTriples_A_B - convert PETSc matrix to triples: row[nz], col[nz], val[nz]

  input:
    A       - matrix in aij,baij or sbaij format
    shift   - 0: C style output triple; 1: Fortran style output triple.
    reuse   - MAT_INITIAL_MATRIX: spaces are allocated and values are set for the triple
              MAT_REUSE_MATRIX:   only the values in v array are updated
  output:
    nnz     - dim of r, c, and v (number of local nonzero entries of A)
    r, c, v - row and col index, matrix values (matrix triples)

  The returned values r, c, and sometimes v are obtained in a single PetscMalloc(). Then in MatDestroy_MUMPS() it is
  freed with PetscFree(mumps->irn);  This is not ideal code, the fact that v is ONLY sometimes part of mumps->irn means
  that the PetscMalloc() cannot easily be replaced with a PetscMalloc3().

 */

static PetscErrorCode MatConvertToTriples_seqaij_seqaij(Mat A, PetscInt shift, MatReuse reuse, Mat_MUMPS *mumps)
{
  const PetscScalar *av;
  const PetscInt    *ai, *aj, *ajj, M = A->rmap->n;
  PetscCount         nz, rnz, k;
  PetscMUMPSInt     *row, *col;
  Mat_SeqAIJ        *aa = (Mat_SeqAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJGetArrayRead(A, &av));
  if (reuse == MAT_INITIAL_MATRIX) {
    nz = aa->nz;
    ai = aa->i;
    aj = aa->j;
    PetscCall(PetscMalloc2(nz, &row, nz, &col));
    for (PetscCount i = k = 0; i < M; i++) {
      rnz = ai[i + 1] - ai[i];
      ajj = aj + ai[i];
      for (PetscCount j = 0; j < rnz; j++) {
        PetscCall(PetscMUMPSIntCast(i + shift, &row[k]));
        PetscCall(PetscMUMPSIntCast(ajj[j] + shift, &col[k]));
        k++;
      }
    }
    mumps->val = (PetscScalar *)av;
    mumps->irn = row;
    mumps->jcn = col;
    mumps->nnz = nz;
  } else if (mumps->nest_vals) PetscCall(PetscArraycpy(mumps->val, av, aa->nz)); /* MatConvertToTriples_nest_xaij() allocates mumps->val outside of MatConvertToTriples_seqaij_seqaij(), so one needs to copy the memory */
  else mumps->val = (PetscScalar *)av;                                           /* in the default case, mumps->val is never allocated, one just needs to update the mumps->val pointer */
  PetscCall(MatSeqAIJRestoreArrayRead(A, &av));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatConvertToTriples_seqsell_seqaij(Mat A, PetscInt shift, MatReuse reuse, Mat_MUMPS *mumps)
{
  PetscCount     nz, i, j, k, r;
  Mat_SeqSELL   *a = (Mat_SeqSELL *)A->data;
  PetscMUMPSInt *row, *col;

  PetscFunctionBegin;
  nz = a->sliidx[a->totalslices];
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(PetscMalloc2(nz, &row, nz, &col));
    for (i = k = 0; i < a->totalslices; i++) {
      for (j = a->sliidx[i], r = 0; j < a->sliidx[i + 1]; j++, r = ((r + 1) & 0x07)) PetscCall(PetscMUMPSIntCast(8 * i + r + shift, &row[k++]));
    }
    for (i = 0; i < nz; i++) PetscCall(PetscMUMPSIntCast(a->colidx[i] + shift, &col[i]));
    mumps->irn = row;
    mumps->jcn = col;
    mumps->nnz = nz;
    mumps->val = a->val;
  } else if (mumps->nest_vals) PetscCall(PetscArraycpy(mumps->val, a->val, nz)); /* MatConvertToTriples_nest_xaij() allocates mumps->val outside of MatConvertToTriples_seqsell_seqaij(), so one needs to copy the memory */
  else mumps->val = a->val;                                                      /* in the default case, mumps->val is never allocated, one just needs to update the mumps->val pointer */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatConvertToTriples_seqbaij_seqaij(Mat A, PetscInt shift, MatReuse reuse, Mat_MUMPS *mumps)
{
  Mat_SeqBAIJ    *aa = (Mat_SeqBAIJ *)A->data;
  const PetscInt *ai, *aj, *ajj, bs2 = aa->bs2;
  PetscCount      M, nz = bs2 * aa->nz, idx = 0, rnz, i, j, k, m;
  PetscInt        bs;
  PetscMUMPSInt  *row, *col;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatGetBlockSize(A, &bs));
    M  = A->rmap->N / bs;
    ai = aa->i;
    aj = aa->j;
    PetscCall(PetscMalloc2(nz, &row, nz, &col));
    for (i = 0; i < M; i++) {
      ajj = aj + ai[i];
      rnz = ai[i + 1] - ai[i];
      for (k = 0; k < rnz; k++) {
        for (j = 0; j < bs; j++) {
          for (m = 0; m < bs; m++) {
            PetscCall(PetscMUMPSIntCast(i * bs + m + shift, &row[idx]));
            PetscCall(PetscMUMPSIntCast(bs * ajj[k] + j + shift, &col[idx]));
            idx++;
          }
        }
      }
    }
    mumps->irn = row;
    mumps->jcn = col;
    mumps->nnz = nz;
    mumps->val = aa->a;
  } else if (mumps->nest_vals) PetscCall(PetscArraycpy(mumps->val, aa->a, nz)); /* MatConvertToTriples_nest_xaij() allocates mumps->val outside of MatConvertToTriples_seqbaij_seqaij(), so one needs to copy the memory */
  else mumps->val = aa->a;                                                      /* in the default case, mumps->val is never allocated, one just needs to update the mumps->val pointer */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatConvertToTriples_seqsbaij_seqsbaij(Mat A, PetscInt shift, MatReuse reuse, Mat_MUMPS *mumps)
{
  const PetscInt *ai, *aj, *ajj;
  PetscInt        bs;
  PetscCount      nz, rnz, i, j, k, m;
  PetscMUMPSInt  *row, *col;
  PetscScalar    *val;
  Mat_SeqSBAIJ   *aa  = (Mat_SeqSBAIJ *)A->data;
  const PetscInt  bs2 = aa->bs2, mbs = aa->mbs;
#if defined(PETSC_USE_COMPLEX)
  PetscBool isset, hermitian;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  PetscCall(MatIsHermitianKnown(A, &isset, &hermitian));
  PetscCheck(!isset || !hermitian, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "MUMPS does not support Hermitian symmetric matrices for Choleksy");
#endif
  ai = aa->i;
  aj = aa->j;
  PetscCall(MatGetBlockSize(A, &bs));
  if (reuse == MAT_INITIAL_MATRIX) {
    const PetscCount alloc_size = aa->nz * bs2;

    PetscCall(PetscMalloc2(alloc_size, &row, alloc_size, &col));
    if (bs > 1) {
      PetscCall(PetscMalloc1(alloc_size, &mumps->val_alloc));
      mumps->val = mumps->val_alloc;
    } else {
      mumps->val = aa->a;
    }
    mumps->irn = row;
    mumps->jcn = col;
  } else {
    row = mumps->irn;
    col = mumps->jcn;
  }
  val = mumps->val;

  nz = 0;
  if (bs > 1) {
    for (i = 0; i < mbs; i++) {
      rnz = ai[i + 1] - ai[i];
      ajj = aj + ai[i];
      for (j = 0; j < rnz; j++) {
        for (k = 0; k < bs; k++) {
          for (m = 0; m < bs; m++) {
            if (ajj[j] > i || k >= m) {
              if (reuse == MAT_INITIAL_MATRIX) {
                PetscCall(PetscMUMPSIntCast(i * bs + m + shift, &row[nz]));
                PetscCall(PetscMUMPSIntCast(ajj[j] * bs + k + shift, &col[nz]));
              }
              val[nz++] = aa->a[(ai[i] + j) * bs2 + m + k * bs];
            }
          }
        }
      }
    }
  } else if (reuse == MAT_INITIAL_MATRIX) {
    for (i = 0; i < mbs; i++) {
      rnz = ai[i + 1] - ai[i];
      ajj = aj + ai[i];
      for (j = 0; j < rnz; j++) {
        PetscCall(PetscMUMPSIntCast(i + shift, &row[nz]));
        PetscCall(PetscMUMPSIntCast(ajj[j] + shift, &col[nz]));
        nz++;
      }
    }
    PetscCheck(nz == aa->nz, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Different numbers of nonzeros %" PetscCount_FMT " != %" PetscInt_FMT, nz, aa->nz);
  } else if (mumps->nest_vals)
    PetscCall(PetscArraycpy(mumps->val, aa->a, aa->nz)); /* bs == 1 and MAT_REUSE_MATRIX, MatConvertToTriples_nest_xaij() allocates mumps->val outside of MatConvertToTriples_seqsbaij_seqsbaij(), so one needs to copy the memory */
  else mumps->val = aa->a;                               /* in the default case, mumps->val is never allocated, one just needs to update the mumps->val pointer */
  if (reuse == MAT_INITIAL_MATRIX) mumps->nnz = nz;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatConvertToTriples_seqaij_seqsbaij(Mat A, PetscInt shift, MatReuse reuse, Mat_MUMPS *mumps)
{
  const PetscInt    *ai, *aj, *ajj, *adiag, M = A->rmap->n;
  PetscCount         nz, rnz, i, j;
  const PetscScalar *av, *v1;
  PetscScalar       *val;
  PetscMUMPSInt     *row, *col;
  Mat_SeqAIJ        *aa = (Mat_SeqAIJ *)A->data;
  PetscBool          diagDense;
#if defined(PETSC_USE_COMPLEX)
  PetscBool hermitian, isset;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  PetscCall(MatIsHermitianKnown(A, &isset, &hermitian));
  PetscCheck(!isset || !hermitian, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "MUMPS does not support Hermitian symmetric matrices for Choleksy");
#endif
  PetscCall(MatSeqAIJGetArrayRead(A, &av));
  ai = aa->i;
  aj = aa->j;
  PetscCall(MatGetDiagonalMarkers_SeqAIJ(A, &adiag, &diagDense));
  if (reuse == MAT_INITIAL_MATRIX) {
    /* count nz in the upper triangular part of A */
    nz = 0;
    if (!diagDense) {
      for (i = 0; i < M; i++) {
        if (PetscUnlikely(adiag[i] >= ai[i + 1])) {
          for (j = ai[i]; j < ai[i + 1]; j++) {
            if (aj[j] < i) continue;
            nz++;
          }
        } else {
          nz += ai[i + 1] - adiag[i];
        }
      }
    } else {
      for (i = 0; i < M; i++) nz += ai[i + 1] - adiag[i];
    }
    PetscCall(PetscMalloc2(nz, &row, nz, &col));
    PetscCall(PetscMalloc1(nz, &val));
    mumps->nnz = nz;
    mumps->irn = row;
    mumps->jcn = col;
    mumps->val = mumps->val_alloc = val;

    nz = 0;
    if (!diagDense) {
      for (i = 0; i < M; i++) {
        if (PetscUnlikely(adiag[i] >= ai[i + 1])) {
          for (j = ai[i]; j < ai[i + 1]; j++) {
            if (aj[j] < i) continue;
            PetscCall(PetscMUMPSIntCast(i + shift, &row[nz]));
            PetscCall(PetscMUMPSIntCast(aj[j] + shift, &col[nz]));
            val[nz] = av[j];
            nz++;
          }
        } else {
          rnz = ai[i + 1] - adiag[i];
          ajj = aj + adiag[i];
          v1  = av + adiag[i];
          for (j = 0; j < rnz; j++) {
            PetscCall(PetscMUMPSIntCast(i + shift, &row[nz]));
            PetscCall(PetscMUMPSIntCast(ajj[j] + shift, &col[nz]));
            val[nz++] = v1[j];
          }
        }
      }
    } else {
      for (i = 0; i < M; i++) {
        rnz = ai[i + 1] - adiag[i];
        ajj = aj + adiag[i];
        v1  = av + adiag[i];
        for (j = 0; j < rnz; j++) {
          PetscCall(PetscMUMPSIntCast(i + shift, &row[nz]));
          PetscCall(PetscMUMPSIntCast(ajj[j] + shift, &col[nz]));
          val[nz++] = v1[j];
        }
      }
    }
  } else {
    nz  = 0;
    val = mumps->val;
    if (!diagDense) {
      for (i = 0; i < M; i++) {
        if (PetscUnlikely(adiag[i] >= ai[i + 1])) {
          for (j = ai[i]; j < ai[i + 1]; j++) {
            if (aj[j] < i) continue;
            val[nz++] = av[j];
          }
        } else {
          rnz = ai[i + 1] - adiag[i];
          v1  = av + adiag[i];
          for (j = 0; j < rnz; j++) val[nz++] = v1[j];
        }
      }
    } else {
      for (i = 0; i < M; i++) {
        rnz = ai[i + 1] - adiag[i];
        v1  = av + adiag[i];
        for (j = 0; j < rnz; j++) val[nz++] = v1[j];
      }
    }
  }
  PetscCall(MatSeqAIJRestoreArrayRead(A, &av));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatConvertToTriples_mpisbaij_mpisbaij(Mat A, PetscInt shift, MatReuse reuse, Mat_MUMPS *mumps)
{
  const PetscInt    *ai, *aj, *bi, *bj, *garray, *ajj, *bjj;
  PetscInt           bs;
  PetscCount         rstart, nz, i, j, k, m, jj, irow, countA, countB;
  PetscMUMPSInt     *row, *col;
  const PetscScalar *av, *bv, *v1, *v2;
  PetscScalar       *val;
  Mat_MPISBAIJ      *mat = (Mat_MPISBAIJ *)A->data;
  Mat_SeqSBAIJ      *aa  = (Mat_SeqSBAIJ *)mat->A->data;
  Mat_SeqBAIJ       *bb  = (Mat_SeqBAIJ *)mat->B->data;
  const PetscInt     bs2 = aa->bs2, mbs = aa->mbs;
#if defined(PETSC_USE_COMPLEX)
  PetscBool hermitian, isset;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  PetscCall(MatIsHermitianKnown(A, &isset, &hermitian));
  PetscCheck(!isset || !hermitian, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "MUMPS does not support Hermitian symmetric matrices for Choleksy");
#endif
  PetscCall(MatGetBlockSize(A, &bs));
  rstart = A->rmap->rstart;
  ai     = aa->i;
  aj     = aa->j;
  bi     = bb->i;
  bj     = bb->j;
  av     = aa->a;
  bv     = bb->a;

  garray = mat->garray;

  if (reuse == MAT_INITIAL_MATRIX) {
    nz = (aa->nz + bb->nz) * bs2; /* just a conservative estimate */
    PetscCall(PetscMalloc2(nz, &row, nz, &col));
    PetscCall(PetscMalloc1(nz, &val));
    /* can not decide the exact mumps->nnz now because of the SBAIJ */
    mumps->irn = row;
    mumps->jcn = col;
    mumps->val = mumps->val_alloc = val;
  } else {
    val = mumps->val;
  }

  jj   = 0;
  irow = rstart;
  for (i = 0; i < mbs; i++) {
    ajj    = aj + ai[i]; /* ptr to the beginning of this row */
    countA = ai[i + 1] - ai[i];
    countB = bi[i + 1] - bi[i];
    bjj    = bj + bi[i];
    v1     = av + ai[i] * bs2;
    v2     = bv + bi[i] * bs2;

    if (bs > 1) {
      /* A-part */
      for (j = 0; j < countA; j++) {
        for (k = 0; k < bs; k++) {
          for (m = 0; m < bs; m++) {
            if (rstart + ajj[j] * bs > irow || k >= m) {
              if (reuse == MAT_INITIAL_MATRIX) {
                PetscCall(PetscMUMPSIntCast(irow + m + shift, &row[jj]));
                PetscCall(PetscMUMPSIntCast(rstart + ajj[j] * bs + k + shift, &col[jj]));
              }
              val[jj++] = v1[j * bs2 + m + k * bs];
            }
          }
        }
      }

      /* B-part */
      for (j = 0; j < countB; j++) {
        for (k = 0; k < bs; k++) {
          for (m = 0; m < bs; m++) {
            if (reuse == MAT_INITIAL_MATRIX) {
              PetscCall(PetscMUMPSIntCast(irow + m + shift, &row[jj]));
              PetscCall(PetscMUMPSIntCast(garray[bjj[j]] * bs + k + shift, &col[jj]));
            }
            val[jj++] = v2[j * bs2 + m + k * bs];
          }
        }
      }
    } else {
      /* A-part */
      for (j = 0; j < countA; j++) {
        if (reuse == MAT_INITIAL_MATRIX) {
          PetscCall(PetscMUMPSIntCast(irow + shift, &row[jj]));
          PetscCall(PetscMUMPSIntCast(rstart + ajj[j] + shift, &col[jj]));
        }
        val[jj++] = v1[j];
      }

      /* B-part */
      for (j = 0; j < countB; j++) {
        if (reuse == MAT_INITIAL_MATRIX) {
          PetscCall(PetscMUMPSIntCast(irow + shift, &row[jj]));
          PetscCall(PetscMUMPSIntCast(garray[bjj[j]] + shift, &col[jj]));
        }
        val[jj++] = v2[j];
      }
    }
    irow += bs;
  }
  if (reuse == MAT_INITIAL_MATRIX) mumps->nnz = jj;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatConvertToTriples_mpiaij_mpiaij(Mat A, PetscInt shift, MatReuse reuse, Mat_MUMPS *mumps)
{
  const PetscInt    *ai, *aj, *bi, *bj, *garray, m = A->rmap->n, *ajj, *bjj;
  PetscCount         rstart, cstart, nz, i, j, jj, irow, countA, countB;
  PetscMUMPSInt     *row, *col;
  const PetscScalar *av, *bv, *v1, *v2;
  PetscScalar       *val;
  Mat                Ad, Ao;
  Mat_SeqAIJ        *aa;
  Mat_SeqAIJ        *bb;

  PetscFunctionBegin;
  PetscCall(MatMPIAIJGetSeqAIJ(A, &Ad, &Ao, &garray));
  PetscCall(MatSeqAIJGetArrayRead(Ad, &av));
  PetscCall(MatSeqAIJGetArrayRead(Ao, &bv));

  aa = (Mat_SeqAIJ *)Ad->data;
  bb = (Mat_SeqAIJ *)Ao->data;
  ai = aa->i;
  aj = aa->j;
  bi = bb->i;
  bj = bb->j;

  rstart = A->rmap->rstart;
  cstart = A->cmap->rstart;

  if (reuse == MAT_INITIAL_MATRIX) {
    nz = (PetscCount)aa->nz + bb->nz; /* make sure the sum won't overflow PetscInt */
    PetscCall(PetscMalloc2(nz, &row, nz, &col));
    PetscCall(PetscMalloc1(nz, &val));
    mumps->nnz = nz;
    mumps->irn = row;
    mumps->jcn = col;
    mumps->val = mumps->val_alloc = val;
  } else {
    val = mumps->val;
  }

  jj   = 0;
  irow = rstart;
  for (i = 0; i < m; i++) {
    ajj    = aj + ai[i]; /* ptr to the beginning of this row */
    countA = ai[i + 1] - ai[i];
    countB = bi[i + 1] - bi[i];
    bjj    = bj + bi[i];
    v1     = av + ai[i];
    v2     = bv + bi[i];

    /* A-part */
    for (j = 0; j < countA; j++) {
      if (reuse == MAT_INITIAL_MATRIX) {
        PetscCall(PetscMUMPSIntCast(irow + shift, &row[jj]));
        PetscCall(PetscMUMPSIntCast(cstart + ajj[j] + shift, &col[jj]));
      }
      val[jj++] = v1[j];
    }

    /* B-part */
    for (j = 0; j < countB; j++) {
      if (reuse == MAT_INITIAL_MATRIX) {
        PetscCall(PetscMUMPSIntCast(irow + shift, &row[jj]));
        PetscCall(PetscMUMPSIntCast(garray[bjj[j]] + shift, &col[jj]));
      }
      val[jj++] = v2[j];
    }
    irow++;
  }
  PetscCall(MatSeqAIJRestoreArrayRead(Ad, &av));
  PetscCall(MatSeqAIJRestoreArrayRead(Ao, &bv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatConvertToTriples_mpibaij_mpiaij(Mat A, PetscInt shift, MatReuse reuse, Mat_MUMPS *mumps)
{
  Mat_MPIBAIJ       *mat = (Mat_MPIBAIJ *)A->data;
  Mat_SeqBAIJ       *aa  = (Mat_SeqBAIJ *)mat->A->data;
  Mat_SeqBAIJ       *bb  = (Mat_SeqBAIJ *)mat->B->data;
  const PetscInt    *ai = aa->i, *bi = bb->i, *aj = aa->j, *bj = bb->j, *ajj, *bjj;
  const PetscInt    *garray = mat->garray, mbs = mat->mbs, rstart = A->rmap->rstart, cstart = A->cmap->rstart;
  const PetscInt     bs2 = mat->bs2;
  PetscInt           bs;
  PetscCount         nz, i, j, k, n, jj, irow, countA, countB, idx;
  PetscMUMPSInt     *row, *col;
  const PetscScalar *av = aa->a, *bv = bb->a, *v1, *v2;
  PetscScalar       *val;

  PetscFunctionBegin;
  PetscCall(MatGetBlockSize(A, &bs));
  if (reuse == MAT_INITIAL_MATRIX) {
    nz = bs2 * (aa->nz + bb->nz);
    PetscCall(PetscMalloc2(nz, &row, nz, &col));
    PetscCall(PetscMalloc1(nz, &val));
    mumps->nnz = nz;
    mumps->irn = row;
    mumps->jcn = col;
    mumps->val = mumps->val_alloc = val;
  } else {
    val = mumps->val;
  }

  jj   = 0;
  irow = rstart;
  for (i = 0; i < mbs; i++) {
    countA = ai[i + 1] - ai[i];
    countB = bi[i + 1] - bi[i];
    ajj    = aj + ai[i];
    bjj    = bj + bi[i];
    v1     = av + bs2 * ai[i];
    v2     = bv + bs2 * bi[i];

    idx = 0;
    /* A-part */
    for (k = 0; k < countA; k++) {
      for (j = 0; j < bs; j++) {
        for (n = 0; n < bs; n++) {
          if (reuse == MAT_INITIAL_MATRIX) {
            PetscCall(PetscMUMPSIntCast(irow + n + shift, &row[jj]));
            PetscCall(PetscMUMPSIntCast(cstart + bs * ajj[k] + j + shift, &col[jj]));
          }
          val[jj++] = v1[idx++];
        }
      }
    }

    idx = 0;
    /* B-part */
    for (k = 0; k < countB; k++) {
      for (j = 0; j < bs; j++) {
        for (n = 0; n < bs; n++) {
          if (reuse == MAT_INITIAL_MATRIX) {
            PetscCall(PetscMUMPSIntCast(irow + n + shift, &row[jj]));
            PetscCall(PetscMUMPSIntCast(bs * garray[bjj[k]] + j + shift, &col[jj]));
          }
          val[jj++] = v2[idx++];
        }
      }
    }
    irow += bs;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatConvertToTriples_mpiaij_mpisbaij(Mat A, PetscInt shift, MatReuse reuse, Mat_MUMPS *mumps)
{
  const PetscInt    *ai, *aj, *adiag, *bi, *bj, *garray, m = A->rmap->n, *ajj, *bjj;
  PetscCount         rstart, nz, nza, nzb, i, j, jj, irow, countA, countB;
  PetscMUMPSInt     *row, *col;
  const PetscScalar *av, *bv, *v1, *v2;
  PetscScalar       *val;
  Mat                Ad, Ao;
  Mat_SeqAIJ        *aa;
  Mat_SeqAIJ        *bb;
#if defined(PETSC_USE_COMPLEX)
  PetscBool hermitian, isset;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  PetscCall(MatIsHermitianKnown(A, &isset, &hermitian));
  PetscCheck(!isset || !hermitian, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "MUMPS does not support Hermitian symmetric matrices for Choleksy");
#endif
  PetscCall(MatMPIAIJGetSeqAIJ(A, &Ad, &Ao, &garray));
  PetscCall(MatSeqAIJGetArrayRead(Ad, &av));
  PetscCall(MatSeqAIJGetArrayRead(Ao, &bv));

  aa = (Mat_SeqAIJ *)Ad->data;
  bb = (Mat_SeqAIJ *)Ao->data;
  ai = aa->i;
  aj = aa->j;
  bi = bb->i;
  bj = bb->j;
  PetscCall(MatGetDiagonalMarkers_SeqAIJ(Ad, &adiag, NULL));
  rstart = A->rmap->rstart;

  if (reuse == MAT_INITIAL_MATRIX) {
    nza = 0; /* num of upper triangular entries in mat->A, including diagonals */
    nzb = 0; /* num of upper triangular entries in mat->B */
    for (i = 0; i < m; i++) {
      nza += (ai[i + 1] - adiag[i]);
      countB = bi[i + 1] - bi[i];
      bjj    = bj + bi[i];
      for (j = 0; j < countB; j++) {
        if (garray[bjj[j]] > rstart) nzb++;
      }
    }

    nz = nza + nzb; /* total nz of upper triangular part of mat */
    PetscCall(PetscMalloc2(nz, &row, nz, &col));
    PetscCall(PetscMalloc1(nz, &val));
    mumps->nnz = nz;
    mumps->irn = row;
    mumps->jcn = col;
    mumps->val = mumps->val_alloc = val;
  } else {
    val = mumps->val;
  }

  jj   = 0;
  irow = rstart;
  for (i = 0; i < m; i++) {
    ajj    = aj + adiag[i]; /* ptr to the beginning of the diagonal of this row */
    v1     = av + adiag[i];
    countA = ai[i + 1] - adiag[i];
    countB = bi[i + 1] - bi[i];
    bjj    = bj + bi[i];
    v2     = bv + bi[i];

    /* A-part */
    for (j = 0; j < countA; j++) {
      if (reuse == MAT_INITIAL_MATRIX) {
        PetscCall(PetscMUMPSIntCast(irow + shift, &row[jj]));
        PetscCall(PetscMUMPSIntCast(rstart + ajj[j] + shift, &col[jj]));
      }
      val[jj++] = v1[j];
    }

    /* B-part */
    for (j = 0; j < countB; j++) {
      if (garray[bjj[j]] > rstart) {
        if (reuse == MAT_INITIAL_MATRIX) {
          PetscCall(PetscMUMPSIntCast(irow + shift, &row[jj]));
          PetscCall(PetscMUMPSIntCast(garray[bjj[j]] + shift, &col[jj]));
        }
        val[jj++] = v2[j];
      }
    }
    irow++;
  }
  PetscCall(MatSeqAIJRestoreArrayRead(Ad, &av));
  PetscCall(MatSeqAIJRestoreArrayRead(Ao, &bv));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatConvertToTriples_diagonal_xaij(Mat A, PETSC_UNUSED PetscInt shift, MatReuse reuse, Mat_MUMPS *mumps)
{
  const PetscScalar *av;
  const PetscInt     M = A->rmap->n;
  PetscCount         i;
  PetscMUMPSInt     *row, *col;
  Vec                v;

  PetscFunctionBegin;
  PetscCall(MatDiagonalGetDiagonal(A, &v));
  PetscCall(VecGetArrayRead(v, &av));
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(PetscMalloc2(M, &row, M, &col));
    for (i = 0; i < M; i++) {
      PetscCall(PetscMUMPSIntCast(i + A->rmap->rstart, &row[i]));
      col[i] = row[i];
    }
    mumps->val = (PetscScalar *)av;
    mumps->irn = row;
    mumps->jcn = col;
    mumps->nnz = M;
  } else if (mumps->nest_vals) PetscCall(PetscArraycpy(mumps->val, av, M)); /* MatConvertToTriples_nest_xaij() allocates mumps->val outside of MatConvertToTriples_diagonal_xaij(), so one needs to copy the memory */
  else mumps->val = (PetscScalar *)av;                                      /* in the default case, mumps->val is never allocated, one just needs to update the mumps->val pointer */
  PetscCall(VecRestoreArrayRead(v, &av));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatConvertToTriples_dense_xaij(Mat A, PETSC_UNUSED PetscInt shift, MatReuse reuse, Mat_MUMPS *mumps)
{
  PetscScalar   *v;
  const PetscInt m = A->rmap->n, N = A->cmap->N;
  PetscInt       lda;
  PetscCount     i, j;
  PetscMUMPSInt *row, *col;

  PetscFunctionBegin;
  PetscCall(MatDenseGetArray(A, &v));
  PetscCall(MatDenseGetLDA(A, &lda));
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(PetscMalloc2(m * N, &row, m * N, &col));
    for (i = 0; i < m; i++) {
      col[i] = 0;
      PetscCall(PetscMUMPSIntCast(i + A->rmap->rstart, &row[i]));
    }
    for (j = 1; j < N; j++) {
      for (i = 0; i < m; i++) PetscCall(PetscMUMPSIntCast(j, col + i + m * j));
      PetscCall(PetscArraycpy(row + m * j, row + m * (j - 1), m));
    }
    if (lda == m) mumps->val = v;
    else {
      PetscCall(PetscMalloc1(m * N, &mumps->val));
      mumps->val_alloc = mumps->val;
      for (j = 0; j < N; j++) PetscCall(PetscArraycpy(mumps->val + m * j, v + lda * j, m));
    }
    mumps->irn = row;
    mumps->jcn = col;
    mumps->nnz = m * N;
  } else {
    if (lda == m && !mumps->nest_vals) mumps->val = v;
    else {
      for (j = 0; j < N; j++) PetscCall(PetscArraycpy(mumps->val + m * j, v + lda * j, m));
    }
  }
  PetscCall(MatDenseRestoreArray(A, &v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// If the input Mat (sub) is either MATTRANSPOSEVIRTUAL or MATHERMITIANTRANSPOSEVIRTUAL, this function gets the parent Mat until it is not a
// MATTRANSPOSEVIRTUAL or MATHERMITIANTRANSPOSEVIRTUAL itself and returns the appropriate shift, scaling, and whether the parent Mat should be conjugated
// and its rows and columns permuted
// TODO FIXME: this should not be in this file and should instead be refactored where the same logic applies, e.g., MatAXPY_Dense_Nest()
static PetscErrorCode MatGetTranspose_TransposeVirtual(Mat *sub, PetscBool *conjugate, PetscScalar *vshift, PetscScalar *vscale, PetscBool *swap)
{
  Mat         A;
  PetscScalar s[2];
  PetscBool   isTrans, isHTrans, compare;

  PetscFunctionBegin;
  do {
    PetscCall(PetscObjectTypeCompare((PetscObject)*sub, MATTRANSPOSEVIRTUAL, &isTrans));
    if (isTrans) {
      PetscCall(MatTransposeGetMat(*sub, &A));
      isHTrans = PETSC_FALSE;
    } else {
      PetscCall(PetscObjectTypeCompare((PetscObject)*sub, MATHERMITIANTRANSPOSEVIRTUAL, &isHTrans));
      if (isHTrans) PetscCall(MatHermitianTransposeGetMat(*sub, &A));
    }
    compare = (PetscBool)(isTrans || isHTrans);
    if (compare) {
      if (vshift && vscale) {
        PetscCall(MatShellGetScalingShifts(*sub, s, s + 1, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
        if (!*conjugate) {
          *vshift += s[0] * *vscale;
          *vscale *= s[1];
        } else {
          *vshift += PetscConj(s[0]) * *vscale;
          *vscale *= PetscConj(s[1]);
        }
      }
      if (swap) *swap = (PetscBool)!*swap;
      if (isHTrans && conjugate) *conjugate = (PetscBool)!*conjugate;
      *sub = A;
    }
  } while (compare);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatConvertToTriples_nest_xaij(Mat A, PetscInt shift, MatReuse reuse, Mat_MUMPS *mumps)
{
  Mat     **mats;
  PetscInt  nr, nc;
  PetscBool chol = mumps->sym ? PETSC_TRUE : PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(MatNestGetSubMats(A, &nr, &nc, &mats));
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscMUMPSInt *irns, *jcns;
    PetscScalar   *vals;
    PetscCount     totnnz, cumnnz, maxnnz;
    PetscInt      *pjcns_w, Mbs = 0;
    IS            *rows, *cols;
    PetscInt     **rows_idx, **cols_idx;

    cumnnz = 0;
    maxnnz = 0;
    PetscCall(PetscMalloc2(nr * nc + 1, &mumps->nest_vals_start, nr * nc, &mumps->nest_convert_to_triples));
    for (PetscInt r = 0; r < nr; r++) {
      for (PetscInt c = 0; c < nc; c++) {
        Mat sub = mats[r][c];

        mumps->nest_convert_to_triples[r * nc + c] = NULL;
        if (chol && c < r) continue; /* skip lower-triangular block for Cholesky */
        if (sub) {
          PetscErrorCode (*convert_to_triples)(Mat, PetscInt, MatReuse, Mat_MUMPS *) = NULL;
          PetscBool isSeqAIJ, isMPIAIJ, isSeqBAIJ, isMPIBAIJ, isSeqSBAIJ, isMPISBAIJ, isDiag, isDense;
          MatInfo   info;

          PetscCall(MatGetTranspose_TransposeVirtual(&sub, NULL, NULL, NULL, NULL));
          PetscCall(PetscObjectBaseTypeCompare((PetscObject)sub, MATSEQAIJ, &isSeqAIJ));
          PetscCall(PetscObjectBaseTypeCompare((PetscObject)sub, MATMPIAIJ, &isMPIAIJ));
          PetscCall(PetscObjectBaseTypeCompare((PetscObject)sub, MATSEQBAIJ, &isSeqBAIJ));
          PetscCall(PetscObjectBaseTypeCompare((PetscObject)sub, MATMPIBAIJ, &isMPIBAIJ));
          PetscCall(PetscObjectBaseTypeCompare((PetscObject)sub, MATSEQSBAIJ, &isSeqSBAIJ));
          PetscCall(PetscObjectBaseTypeCompare((PetscObject)sub, MATMPISBAIJ, &isMPISBAIJ));
          PetscCall(PetscObjectTypeCompare((PetscObject)sub, MATDIAGONAL, &isDiag));
          PetscCall(PetscObjectTypeCompareAny((PetscObject)sub, &isDense, MATSEQDENSE, MATMPIDENSE, NULL));

          if (chol) {
            if (r == c) {
              if (isSeqAIJ) convert_to_triples = MatConvertToTriples_seqaij_seqsbaij;
              else if (isMPIAIJ) convert_to_triples = MatConvertToTriples_mpiaij_mpisbaij;
              else if (isSeqSBAIJ) convert_to_triples = MatConvertToTriples_seqsbaij_seqsbaij;
              else if (isMPISBAIJ) convert_to_triples = MatConvertToTriples_mpisbaij_mpisbaij;
              else if (isDiag) convert_to_triples = MatConvertToTriples_diagonal_xaij;
              else if (isDense) convert_to_triples = MatConvertToTriples_dense_xaij;
            } else {
              if (isSeqAIJ) convert_to_triples = MatConvertToTriples_seqaij_seqaij;
              else if (isMPIAIJ) convert_to_triples = MatConvertToTriples_mpiaij_mpiaij;
              else if (isSeqBAIJ) convert_to_triples = MatConvertToTriples_seqbaij_seqaij;
              else if (isMPIBAIJ) convert_to_triples = MatConvertToTriples_mpibaij_mpiaij;
              else if (isDiag) convert_to_triples = MatConvertToTriples_diagonal_xaij;
              else if (isDense) convert_to_triples = MatConvertToTriples_dense_xaij;
            }
          } else {
            if (isSeqAIJ) convert_to_triples = MatConvertToTriples_seqaij_seqaij;
            else if (isMPIAIJ) convert_to_triples = MatConvertToTriples_mpiaij_mpiaij;
            else if (isSeqBAIJ) convert_to_triples = MatConvertToTriples_seqbaij_seqaij;
            else if (isMPIBAIJ) convert_to_triples = MatConvertToTriples_mpibaij_mpiaij;
            else if (isDiag) convert_to_triples = MatConvertToTriples_diagonal_xaij;
            else if (isDense) convert_to_triples = MatConvertToTriples_dense_xaij;
          }
          PetscCheck(convert_to_triples, PetscObjectComm((PetscObject)sub), PETSC_ERR_SUP, "Not for block of type %s", ((PetscObject)sub)->type_name);
          mumps->nest_convert_to_triples[r * nc + c] = convert_to_triples;
          PetscCall(MatGetInfo(sub, MAT_LOCAL, &info));
          cumnnz += (PetscCount)info.nz_used; /* can be overestimated for Cholesky */
          maxnnz = PetscMax(maxnnz, info.nz_used);
        }
      }
    }

    /* Allocate total COO */
    totnnz = cumnnz;
    PetscCall(PetscMalloc2(totnnz, &irns, totnnz, &jcns));
    PetscCall(PetscMalloc1(totnnz, &vals));

    /* Handle rows and column maps
       We directly map rows and use an SF for the columns */
    PetscCall(PetscMalloc4(nr, &rows, nc, &cols, nr, &rows_idx, nc, &cols_idx));
    PetscCall(MatNestGetISs(A, rows, cols));
    for (PetscInt r = 0; r < nr; r++) PetscCall(ISGetIndices(rows[r], (const PetscInt **)&rows_idx[r]));
    for (PetscInt c = 0; c < nc; c++) PetscCall(ISGetIndices(cols[c], (const PetscInt **)&cols_idx[c]));
    if (PetscDefined(USE_64BIT_INDICES)) PetscCall(PetscMalloc1(maxnnz, &pjcns_w));
    else (void)maxnnz;

    cumnnz = 0;
    for (PetscInt r = 0; r < nr; r++) {
      for (PetscInt c = 0; c < nc; c++) {
        Mat             sub    = mats[r][c];
        const PetscInt *ridx   = rows_idx[r];
        const PetscInt *cidx   = cols_idx[c];
        PetscScalar     vscale = 1.0, vshift = 0.0;
        PetscInt        rst, size, bs;
        PetscSF         csf;
        PetscBool       conjugate = PETSC_FALSE, swap = PETSC_FALSE;
        PetscLayout     cmap;
        PetscInt        innz;

        mumps->nest_vals_start[r * nc + c] = cumnnz;
        if (c == r) {
          PetscCall(ISGetSize(rows[r], &size));
          if (!mumps->nest_convert_to_triples[r * nc + c]) {
            for (PetscInt c = 0; c < nc && !sub; ++c) sub = mats[r][c]; // diagonal Mat is NULL, so start over from the beginning of the current row
          }
          PetscCall(MatGetBlockSize(sub, &bs));
          Mbs += size / bs;
        }
        if (!mumps->nest_convert_to_triples[r * nc + c]) continue;

        /* Extract inner blocks if needed */
        PetscCall(MatGetTranspose_TransposeVirtual(&sub, &conjugate, &vshift, &vscale, &swap));
        PetscCheck(vshift == 0.0, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Nonzero shift in parent MatShell");

        /* Get column layout to map off-process columns */
        PetscCall(MatGetLayouts(sub, NULL, &cmap));

        /* Get row start to map on-process rows */
        PetscCall(MatGetOwnershipRange(sub, &rst, NULL));

        /* Directly use the mumps datastructure and use C ordering for now */
        PetscCall((*mumps->nest_convert_to_triples[r * nc + c])(sub, 0, MAT_INITIAL_MATRIX, mumps));

        /* Swap the role of rows and columns indices for transposed blocks
           since we need values with global final ordering */
        if (swap) {
          cidx = rows_idx[r];
          ridx = cols_idx[c];
        }

        /* Communicate column indices
           This could have been done with a single SF but it would have complicated the code a lot.
           But since we do it only once, we pay the price of setting up an SF for each block */
        if (PetscDefined(USE_64BIT_INDICES)) {
          for (PetscInt k = 0; k < mumps->nnz; k++) pjcns_w[k] = mumps->jcn[k];
        } else pjcns_w = (PetscInt *)mumps->jcn; /* This cast is needed only to silence warnings for 64bit integers builds */
        PetscCall(PetscSFCreate(PetscObjectComm((PetscObject)A), &csf));
        PetscCall(PetscIntCast(mumps->nnz, &innz));
        PetscCall(PetscSFSetGraphLayout(csf, cmap, innz, NULL, PETSC_OWN_POINTER, pjcns_w));
        PetscCall(PetscSFBcastBegin(csf, MPIU_INT, cidx, pjcns_w, MPI_REPLACE));
        PetscCall(PetscSFBcastEnd(csf, MPIU_INT, cidx, pjcns_w, MPI_REPLACE));
        PetscCall(PetscSFDestroy(&csf));

        /* Import indices: use direct map for rows and mapped indices for columns */
        if (swap) {
          for (PetscInt k = 0; k < mumps->nnz; k++) {
            PetscCall(PetscMUMPSIntCast(ridx[mumps->irn[k] - rst] + shift, &jcns[cumnnz + k]));
            PetscCall(PetscMUMPSIntCast(pjcns_w[k] + shift, &irns[cumnnz + k]));
          }
        } else {
          for (PetscInt k = 0; k < mumps->nnz; k++) {
            PetscCall(PetscMUMPSIntCast(ridx[mumps->irn[k] - rst] + shift, &irns[cumnnz + k]));
            PetscCall(PetscMUMPSIntCast(pjcns_w[k] + shift, &jcns[cumnnz + k]));
          }
        }

        /* Import values to full COO */
        if (conjugate) { /* conjugate the entries */
          PetscScalar *v = vals + cumnnz;
          for (PetscInt k = 0; k < mumps->nnz; k++) v[k] = vscale * PetscConj(mumps->val[k]);
        } else if (vscale != 1.0) {
          PetscScalar *v = vals + cumnnz;
          for (PetscInt k = 0; k < mumps->nnz; k++) v[k] = vscale * mumps->val[k];
        } else PetscCall(PetscArraycpy(vals + cumnnz, mumps->val, mumps->nnz));

        /* Shift new starting point and sanity check */
        cumnnz += mumps->nnz;
        PetscCheck(cumnnz <= totnnz, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unexpected number of nonzeros %" PetscCount_FMT " != %" PetscCount_FMT, cumnnz, totnnz);

        /* Free scratch memory */
        PetscCall(PetscFree2(mumps->irn, mumps->jcn));
        PetscCall(PetscFree(mumps->val_alloc));
        mumps->val = NULL;
        mumps->nnz = 0;
      }
    }
    if (mumps->id.ICNTL(15) == 1) {
      if (Mbs != A->rmap->N) {
        PetscMPIInt rank, size;

        PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
        PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A), &size));
        if (rank == 0) {
          PetscInt shift = 0;

          PetscCall(PetscMUMPSIntCast(Mbs, &mumps->id.nblk));
          PetscCall(PetscFree(mumps->id.blkptr));
          PetscCall(PetscMalloc1(Mbs + 1, &mumps->id.blkptr));
          mumps->id.blkptr[0] = 1;
          for (PetscInt i = 0; i < size; ++i) {
            for (PetscInt r = 0; r < nr; r++) {
              Mat             sub = mats[r][r];
              const PetscInt *ranges;
              PetscInt        bs;

              for (PetscInt c = 0; c < nc && !sub; ++c) sub = mats[r][c]; // diagonal Mat is NULL, so start over from the beginning of the current row
              PetscCall(MatGetOwnershipRanges(sub, &ranges));
              PetscCall(MatGetBlockSize(sub, &bs));
              for (PetscInt j = 0, start = mumps->id.blkptr[shift] + bs; j < ranges[i + 1] - ranges[i]; j += bs) PetscCall(PetscMUMPSIntCast(start + j, mumps->id.blkptr + shift + j / bs + 1));
              shift += (ranges[i + 1] - ranges[i]) / bs;
            }
          }
        }
      } else mumps->id.ICNTL(15) = 0;
    }
    if (PetscDefined(USE_64BIT_INDICES)) PetscCall(PetscFree(pjcns_w));
    for (PetscInt r = 0; r < nr; r++) PetscCall(ISRestoreIndices(rows[r], (const PetscInt **)&rows_idx[r]));
    for (PetscInt c = 0; c < nc; c++) PetscCall(ISRestoreIndices(cols[c], (const PetscInt **)&cols_idx[c]));
    PetscCall(PetscFree4(rows, cols, rows_idx, cols_idx));
    if (!chol) PetscCheck(cumnnz == totnnz, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Different number of nonzeros %" PetscCount_FMT " != %" PetscCount_FMT, cumnnz, totnnz);
    mumps->nest_vals_start[nr * nc] = cumnnz;

    /* Set pointers for final MUMPS data structure */
    mumps->nest_vals = vals;
    mumps->val_alloc = NULL; /* do not use val_alloc since it may be reallocated with the OMP callpath */
    mumps->val       = vals;
    mumps->irn       = irns;
    mumps->jcn       = jcns;
    mumps->nnz       = cumnnz;
  } else {
    PetscScalar *oval = mumps->nest_vals;
    for (PetscInt r = 0; r < nr; r++) {
      for (PetscInt c = 0; c < nc; c++) {
        PetscBool   conjugate = PETSC_FALSE;
        Mat         sub       = mats[r][c];
        PetscScalar vscale = 1.0, vshift = 0.0;
        PetscInt    midx = r * nc + c;

        if (!mumps->nest_convert_to_triples[midx]) continue;
        PetscCall(MatGetTranspose_TransposeVirtual(&sub, &conjugate, &vshift, &vscale, NULL));
        PetscCheck(vshift == 0.0, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Nonzero shift in parent MatShell");
        mumps->val = oval + mumps->nest_vals_start[midx];
        PetscCall((*mumps->nest_convert_to_triples[midx])(sub, shift, MAT_REUSE_MATRIX, mumps));
        if (conjugate) {
          PetscCount nnz = mumps->nest_vals_start[midx + 1] - mumps->nest_vals_start[midx];
          for (PetscCount k = 0; k < nnz; k++) mumps->val[k] = vscale * PetscConj(mumps->val[k]);
        } else if (vscale != 1.0) {
          PetscCount nnz = mumps->nest_vals_start[midx + 1] - mumps->nest_vals_start[midx];
          for (PetscCount k = 0; k < nnz; k++) mumps->val[k] *= vscale;
        }
      }
    }
    mumps->val = oval;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_MUMPS(Mat A)
{
  Mat_MUMPS *mumps = (Mat_MUMPS *)A->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(mumps->id.isol_loc));
  PetscCall(VecScatterDestroy(&mumps->scat_rhs));
  PetscCall(VecScatterDestroy(&mumps->scat_sol));
  PetscCall(VecDestroy(&mumps->b_seq));
  PetscCall(VecDestroy(&mumps->x_seq));
  PetscCall(PetscFree(mumps->id.perm_in));
  PetscCall(PetscFree(mumps->id.blkvar));
  PetscCall(PetscFree(mumps->id.blkptr));
  PetscCall(PetscFree2(mumps->irn, mumps->jcn));
  PetscCall(PetscFree(mumps->val_alloc));
  PetscCall(PetscFree(mumps->info));
  PetscCall(PetscFree(mumps->ICNTL_pre));
  PetscCall(PetscFree(mumps->CNTL_pre));
  PetscCall(MatMumpsResetSchur_Private(mumps));
  if (mumps->id.job != JOB_NULL) { /* cannot call PetscMUMPS_c() if JOB_INIT has never been called for this instance */
    mumps->id.job = JOB_END;
    PetscMUMPS_c(mumps);
    PetscCheck(mumps->id.INFOG(1) >= 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "MUMPS error in termination: INFOG(1)=%d " MUMPS_MANUALS, mumps->id.INFOG(1));
    if (mumps->mumps_comm != MPI_COMM_NULL) {
      if (PetscDefined(HAVE_OPENMP_SUPPORT) && mumps->use_petsc_omp_support) PetscCallMPI(MPI_Comm_free(&mumps->mumps_comm));
      else PetscCall(PetscCommRestoreComm(PetscObjectComm((PetscObject)A), &mumps->mumps_comm));
    }
  }
  PetscCall(MatMumpsFreeInternalID(&mumps->id));
#if defined(PETSC_HAVE_OPENMP_SUPPORT)
  if (mumps->use_petsc_omp_support) {
    PetscCall(PetscOmpCtrlDestroy(&mumps->omp_ctrl));
    PetscCall(PetscFree2(mumps->rhs_loc, mumps->rhs_recvbuf));
    PetscCall(PetscFree3(mumps->rhs_nrow, mumps->rhs_recvcounts, mumps->rhs_disps));
  }
#endif
  PetscCall(PetscFree(mumps->ia_alloc));
  PetscCall(PetscFree(mumps->ja_alloc));
  PetscCall(PetscFree(mumps->recvcount));
  PetscCall(PetscFree(mumps->reqs));
  PetscCall(PetscFree(mumps->irhs_loc));
  PetscCall(PetscFree2(mumps->nest_vals_start, mumps->nest_convert_to_triples));
  PetscCall(PetscFree(mumps->nest_vals));
  PetscCall(PetscFree(A->data));

  /* clear composed functions */
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatFactorGetSolverType_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatFactorSetSchurIS_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatFactorCreateSchurComplement_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMumpsSetIcntl_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMumpsGetIcntl_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMumpsSetCntl_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMumpsGetCntl_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMumpsGetInfo_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMumpsGetInfog_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMumpsGetRinfo_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMumpsGetRinfog_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMumpsGetNullPivots_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMumpsGetInverse_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMumpsGetInverseTranspose_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMumpsSetBlk_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Set up the distributed RHS info for MUMPS. <nrhs> is the number of RHS. <array> points to start of RHS on the local processor. */
static PetscErrorCode MatMumpsSetUpDistRHSInfo(Mat A, PetscInt nrhs, const PetscScalar *array)
{
  Mat_MUMPS        *mumps   = (Mat_MUMPS *)A->data;
  const PetscMPIInt ompsize = mumps->omp_comm_size;
  PetscInt          i, m, M, rstart;

  PetscFunctionBegin;
  PetscCall(MatGetSize(A, &M, NULL));
  PetscCall(MatGetLocalSize(A, &m, NULL));
  PetscCheck(M <= PETSC_MUMPS_INT_MAX, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "PetscInt too long for PetscMUMPSInt");
  if (ompsize == 1) {
    if (!mumps->irhs_loc) {
      mumps->nloc_rhs = (PetscMUMPSInt)m;
      PetscCall(PetscMalloc1(m, &mumps->irhs_loc));
      PetscCall(MatGetOwnershipRange(A, &rstart, NULL));
      for (i = 0; i < m; i++) PetscCall(PetscMUMPSIntCast(rstart + i + 1, &mumps->irhs_loc[i])); /* use 1-based indices */
    }
    PetscCall(MatMumpsMakeMumpsScalarArray(PETSC_TRUE, m * nrhs, array, mumps->id.precision, &mumps->id.rhs_loc_len, &mumps->id.rhs_loc));
  } else {
#if defined(PETSC_HAVE_OPENMP_SUPPORT)
    const PetscInt *ranges;
    PetscMPIInt     j, k, sendcount, *petsc_ranks, *omp_ranks;
    MPI_Group       petsc_group, omp_group;
    PetscScalar    *recvbuf = NULL;

    if (mumps->is_omp_master) {
      /* Lazily initialize the omp stuff for distributed rhs */
      if (!mumps->irhs_loc) {
        PetscCall(PetscMalloc2(ompsize, &omp_ranks, ompsize, &petsc_ranks));
        PetscCall(PetscMalloc3(ompsize, &mumps->rhs_nrow, ompsize, &mumps->rhs_recvcounts, ompsize, &mumps->rhs_disps));
        PetscCallMPI(MPI_Comm_group(mumps->petsc_comm, &petsc_group));
        PetscCallMPI(MPI_Comm_group(mumps->omp_comm, &omp_group));
        for (j = 0; j < ompsize; j++) omp_ranks[j] = j;
        PetscCallMPI(MPI_Group_translate_ranks(omp_group, ompsize, omp_ranks, petsc_group, petsc_ranks));

        /* Populate mumps->irhs_loc[], rhs_nrow[] */
        mumps->nloc_rhs = 0;
        PetscCall(MatGetOwnershipRanges(A, &ranges));
        for (j = 0; j < ompsize; j++) {
          mumps->rhs_nrow[j] = ranges[petsc_ranks[j] + 1] - ranges[petsc_ranks[j]];
          mumps->nloc_rhs += mumps->rhs_nrow[j];
        }
        PetscCall(PetscMalloc1(mumps->nloc_rhs, &mumps->irhs_loc));
        for (j = k = 0; j < ompsize; j++) {
          for (i = ranges[petsc_ranks[j]]; i < ranges[petsc_ranks[j] + 1]; i++, k++) PetscCall(PetscMUMPSIntCast(i + 1, &mumps->irhs_loc[k])); /* uses 1-based indices */
        }

        PetscCall(PetscFree2(omp_ranks, petsc_ranks));
        PetscCallMPI(MPI_Group_free(&petsc_group));
        PetscCallMPI(MPI_Group_free(&omp_group));
      }

      /* Realloc buffers when current nrhs is bigger than what we have met */
      if (nrhs > mumps->max_nrhs) {
        PetscCall(PetscFree2(mumps->rhs_loc, mumps->rhs_recvbuf));
        PetscCall(PetscMalloc2(mumps->nloc_rhs * nrhs, &mumps->rhs_loc, mumps->nloc_rhs * nrhs, &mumps->rhs_recvbuf));
        mumps->max_nrhs = nrhs;
      }

      /* Setup recvcounts[], disps[], recvbuf on omp rank 0 for the upcoming MPI_Gatherv */
      for (j = 0; j < ompsize; j++) PetscCall(PetscMPIIntCast(mumps->rhs_nrow[j] * nrhs, &mumps->rhs_recvcounts[j]));
      mumps->rhs_disps[0] = 0;
      for (j = 1; j < ompsize; j++) {
        mumps->rhs_disps[j] = mumps->rhs_disps[j - 1] + mumps->rhs_recvcounts[j - 1];
        PetscCheck(mumps->rhs_disps[j] >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "PetscMPIInt overflow!");
      }
      recvbuf = (nrhs == 1) ? mumps->rhs_loc : mumps->rhs_recvbuf; /* Directly use rhs_loc[] as recvbuf. Single rhs is common in Ax=b */
    }

    PetscCall(PetscMPIIntCast(m * nrhs, &sendcount));
    PetscCallMPI(MPI_Gatherv(array, sendcount, MPIU_SCALAR, recvbuf, mumps->rhs_recvcounts, mumps->rhs_disps, MPIU_SCALAR, 0, mumps->omp_comm));

    if (mumps->is_omp_master) {
      if (nrhs > 1) { /* Copy & re-arrange data from rhs_recvbuf[] to mumps->rhs_loc[] only when there are multiple rhs */
        PetscScalar *dst, *dstbase = mumps->rhs_loc;
        for (j = 0; j < ompsize; j++) {
          const PetscScalar *src = mumps->rhs_recvbuf + mumps->rhs_disps[j];
          dst                    = dstbase;
          for (i = 0; i < nrhs; i++) {
            PetscCall(PetscArraycpy(dst, src, mumps->rhs_nrow[j]));
            src += mumps->rhs_nrow[j];
            dst += mumps->nloc_rhs;
          }
          dstbase += mumps->rhs_nrow[j];
        }
      }
      PetscCall(MatMumpsMakeMumpsScalarArray(PETSC_TRUE, mumps->nloc_rhs * nrhs, mumps->rhs_loc, mumps->id.precision, &mumps->id.rhs_loc_len, &mumps->id.rhs_loc));
    }
#endif /* PETSC_HAVE_OPENMP_SUPPORT */
  }
  mumps->id.nrhs     = (PetscMUMPSInt)nrhs;
  mumps->id.nloc_rhs = (PetscMUMPSInt)mumps->nloc_rhs;
  mumps->id.lrhs_loc = mumps->nloc_rhs;
  mumps->id.irhs_loc = mumps->irhs_loc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolve_MUMPS(Mat A, Vec b, Vec x)
{
  Mat_MUMPS         *mumps  = (Mat_MUMPS *)A->data;
  const PetscScalar *barray = NULL;
  PetscScalar       *array;
  IS                 is_iden, is_petsc;
  PetscInt           i;
  PetscBool          second_solve = PETSC_FALSE;
  static PetscBool   cite1 = PETSC_FALSE, cite2 = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister("@article{MUMPS01,\n  author = {P.~R. Amestoy and I.~S. Duff and J.-Y. L'Excellent and J. Koster},\n  title = {A fully asynchronous multifrontal solver using distributed dynamic scheduling},\n  journal = {SIAM "
                                   "Journal on Matrix Analysis and Applications},\n  volume = {23},\n  number = {1},\n  pages = {15--41},\n  year = {2001}\n}\n",
                                   &cite1));
  PetscCall(PetscCitationsRegister("@article{MUMPS02,\n  author = {P.~R. Amestoy and A. Guermouche and J.-Y. L'Excellent and S. Pralet},\n  title = {Hybrid scheduling for the parallel solution of linear systems},\n  journal = {Parallel "
                                   "Computing},\n  volume = {32},\n  number = {2},\n  pages = {136--156},\n  year = {2006}\n}\n",
                                   &cite2));

  PetscCall(VecFlag(x, A->factorerrortype));
  if (A->factorerrortype) {
    PetscCall(PetscInfo(A, "MatSolve is called with singular matrix factor, INFOG(1)=%d, INFO(2)=%d\n", mumps->id.INFOG(1), mumps->id.INFO(2)));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  mumps->id.nrhs = 1;
  if (mumps->petsc_size > 1) {
    if (mumps->ICNTL20 == 10) {
      mumps->id.ICNTL(20) = 10; /* dense distributed RHS, need to set rhs_loc[], irhs_loc[] */
      PetscCall(VecGetArrayRead(b, &barray));
      PetscCall(MatMumpsSetUpDistRHSInfo(A, 1, barray));
    } else {
      mumps->id.ICNTL(20) = 0; /* dense centralized RHS; Scatter b into a sequential b_seq vector*/
      PetscCall(VecScatterBegin(mumps->scat_rhs, b, mumps->b_seq, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(mumps->scat_rhs, b, mumps->b_seq, INSERT_VALUES, SCATTER_FORWARD));
      if (!mumps->myid) {
        PetscCall(VecGetArray(mumps->b_seq, &array));
        PetscCall(MatMumpsMakeMumpsScalarArray(PETSC_TRUE, mumps->b_seq->map->n, array, mumps->id.precision, &mumps->id.rhs_len, &mumps->id.rhs));
      }
    }
  } else { /* petsc_size == 1, use MUMPS's dense centralized RHS feature, so that we don't need to bother with isol_loc[] to get the solution */
    mumps->id.ICNTL(20) = 0;
    PetscCall(VecCopy(b, x));
    PetscCall(VecGetArray(x, &array));
    PetscCall(MatMumpsMakeMumpsScalarArray(PETSC_TRUE, x->map->n, array, mumps->id.precision, &mumps->id.rhs_len, &mumps->id.rhs));
  }

  /*
     handle condensation step of Schur complement (if any)
     We set by default ICNTL(26) == -1 when Schur indices have been provided by the user.
     According to MUMPS (5.0.0) manual, any value should be harmful during the factorization phase
     Unless the user provides a valid value for ICNTL(26), MatSolve and MatMatSolve routines solve the full system.
     This requires an extra call to PetscMUMPS_c and the computation of the factors for S
  */
  if (mumps->id.size_schur > 0) {
    PetscCheck(mumps->petsc_size <= 1, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Parallel Schur complements not yet supported from PETSc");
    if (mumps->id.ICNTL(26) < 0 || mumps->id.ICNTL(26) > 2) {
      second_solve = PETSC_TRUE;
      PetscCall(MatMumpsHandleSchur_Private(A, PETSC_FALSE)); // allocate id.redrhs
      mumps->id.ICNTL(26) = 1;                                /* condensation phase */
    } else if (mumps->id.ICNTL(26) == 1) PetscCall(MatMumpsHandleSchur_Private(A, PETSC_FALSE));
  }

  mumps->id.job = JOB_SOLVE;
  PetscMUMPS_c(mumps); // reduced solve, put solution in id.redrhs
  PetscCheck(mumps->id.INFOG(1) >= 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "MUMPS error in solve: INFOG(1)=%d, INFO(2)=%d " MUMPS_MANUALS, mumps->id.INFOG(1), mumps->id.INFO(2));

  /* handle expansion step of Schur complement (if any) */
  if (second_solve) PetscCall(MatMumpsHandleSchur_Private(A, PETSC_TRUE));
  else if (mumps->id.ICNTL(26) == 1) { // condense the right hand side
    PetscCall(MatMumpsSolveSchur_Private(A));
    for (i = 0; i < mumps->id.size_schur; ++i) array[mumps->id.listvar_schur[i] - 1] = ID_FIELD_GET(mumps->id, redrhs, i);
  }

  if (mumps->petsc_size > 1) { /* convert mumps distributed solution to PETSc mpi x */
    if (mumps->scat_sol && mumps->ICNTL9_pre != mumps->id.ICNTL(9)) {
      /* when id.ICNTL(9) changes, the contents of ilsol_loc may change (not its size, lsol_loc), recreates scat_sol */
      PetscCall(VecScatterDestroy(&mumps->scat_sol));
    }
    if (!mumps->scat_sol) { /* create scatter scat_sol */
      PetscInt *isol2_loc = NULL;
      PetscCall(ISCreateStride(PETSC_COMM_SELF, mumps->id.lsol_loc, 0, 1, &is_iden)); /* from */
      PetscCall(PetscMalloc1(mumps->id.lsol_loc, &isol2_loc));
      for (i = 0; i < mumps->id.lsol_loc; i++) isol2_loc[i] = mumps->id.isol_loc[i] - 1;                        /* change Fortran style to C style */
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF, mumps->id.lsol_loc, isol2_loc, PETSC_OWN_POINTER, &is_petsc)); /* to */
      PetscCall(VecScatterCreate(mumps->x_seq, is_iden, x, is_petsc, &mumps->scat_sol));
      PetscCall(ISDestroy(&is_iden));
      PetscCall(ISDestroy(&is_petsc));
      mumps->ICNTL9_pre = mumps->id.ICNTL(9); /* save current value of id.ICNTL(9) */
    }

    PetscScalar *xarray;
    PetscCall(VecGetArray(mumps->x_seq, &xarray));
    PetscCall(MatMumpsCastMumpsScalarArray(mumps->id.lsol_loc, mumps->id.precision, mumps->id.sol_loc, xarray));
    PetscCall(VecRestoreArray(mumps->x_seq, &xarray));
    PetscCall(VecScatterBegin(mumps->scat_sol, mumps->x_seq, x, INSERT_VALUES, SCATTER_FORWARD));
    PetscCall(VecScatterEnd(mumps->scat_sol, mumps->x_seq, x, INSERT_VALUES, SCATTER_FORWARD));

    if (mumps->ICNTL20 == 10) { // distributed RHS
      PetscCall(VecRestoreArrayRead(b, &barray));
    } else if (!mumps->myid) { // centralized RHS
      PetscCall(VecRestoreArray(mumps->b_seq, &array));
    }
  } else {
    // id.rhs has the solution in mumps precision
    PetscCall(MatMumpsCastMumpsScalarArray(x->map->n, mumps->id.precision, mumps->id.rhs, array));
    PetscCall(VecRestoreArray(x, &array));
  }

  PetscCall(PetscLogFlops(2.0 * PetscMax(0, (mumps->id.INFO(28) >= 0 ? mumps->id.INFO(28) : -1000000 * mumps->id.INFO(28)) - A->cmap->n)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSolveTranspose_MUMPS(Mat A, Vec b, Vec x)
{
  Mat_MUMPS          *mumps = (Mat_MUMPS *)A->data;
  const PetscMUMPSInt value = mumps->id.ICNTL(9);

  PetscFunctionBegin;
  mumps->id.ICNTL(9) = 0;
  PetscCall(MatSolve_MUMPS(A, b, x));
  mumps->id.ICNTL(9) = value;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMatSolve_MUMPS(Mat A, Mat B, Mat X)
{
  Mat                Bt = NULL;
  PetscBool          denseX, denseB, flg, flgT;
  Mat_MUMPS         *mumps = (Mat_MUMPS *)A->data;
  PetscInt           i, nrhs, M, nrhsM;
  PetscScalar       *array;
  const PetscScalar *barray;
  PetscInt           lsol_loc, nlsol_loc, *idxx, iidx = 0;
  PetscMUMPSInt     *isol_loc, *isol_loc_save;
  PetscScalar       *sol_loc;
  void              *sol_loc_save;
  PetscCount         sol_loc_len_save;
  IS                 is_to, is_from;
  PetscInt           k, proc, j, m, myrstart;
  const PetscInt    *rstart;
  Vec                v_mpi, msol_loc;
  VecScatter         scat_sol;
  Vec                b_seq;
  VecScatter         scat_rhs;
  PetscScalar       *aa;
  PetscInt           spnr, *ia, *ja;
  Mat_MPIAIJ        *b = NULL;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)X, &denseX, MATSEQDENSE, MATMPIDENSE, NULL));
  PetscCheck(denseX, PetscObjectComm((PetscObject)X), PETSC_ERR_ARG_WRONG, "Matrix X must be MATDENSE matrix");

  PetscCall(PetscObjectTypeCompareAny((PetscObject)B, &denseB, MATSEQDENSE, MATMPIDENSE, NULL));

  if (denseB) {
    PetscCheck(B->rmap->n == X->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Matrix B and X must have same row distribution");
    mumps->id.ICNTL(20) = 0; /* dense RHS */
  } else {                   /* sparse B */
    PetscCheck(X != B, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_IDN, "X and B must be different matrices");
    PetscCall(PetscObjectTypeCompare((PetscObject)B, MATTRANSPOSEVIRTUAL, &flgT));
    PetscCheck(flgT, PetscObjectComm((PetscObject)B), PETSC_ERR_ARG_WRONG, "Matrix B must be MATTRANSPOSEVIRTUAL matrix");
    PetscCall(MatShellGetScalingShifts(B, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
    /* input B is transpose of actual RHS matrix,
     because mumps requires sparse compressed COLUMN storage! See MatMatTransposeSolve_MUMPS() */
    PetscCall(MatTransposeGetMat(B, &Bt));
    mumps->id.ICNTL(20) = 1; /* sparse RHS */
  }

  PetscCall(MatGetSize(B, &M, &nrhs));
  PetscCall(PetscIntMultError(nrhs, M, &nrhsM));
  mumps->id.nrhs = (PetscMUMPSInt)nrhs;
  mumps->id.lrhs = (PetscMUMPSInt)M;

  if (mumps->petsc_size == 1) { // handle this easy case specially and return early
    PetscScalar *aa;
    PetscInt     spnr, *ia, *ja;
    PetscBool    second_solve = PETSC_FALSE;

    PetscCall(MatDenseGetArray(X, &array));
    if (denseB) {
      /* copy B to X */
      PetscCall(MatDenseGetArrayRead(B, &barray));
      PetscCall(PetscArraycpy(array, barray, nrhsM));
      PetscCall(MatDenseRestoreArrayRead(B, &barray));
    } else { /* sparse B */
      PetscCall(MatSeqAIJGetArray(Bt, &aa));
      PetscCall(MatGetRowIJ(Bt, 1, PETSC_FALSE, PETSC_FALSE, &spnr, (const PetscInt **)&ia, (const PetscInt **)&ja, &flg));
      PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot get IJ structure");
      PetscCall(PetscMUMPSIntCSRCast(mumps, spnr, ia, ja, &mumps->id.irhs_ptr, &mumps->id.irhs_sparse, &mumps->id.nz_rhs));
      PetscCall(MatMumpsMakeMumpsScalarArray(PETSC_TRUE, mumps->id.nz_rhs, aa, mumps->id.precision, &mumps->id.rhs_sparse_len, &mumps->id.rhs_sparse));
    }
    PetscCall(MatMumpsMakeMumpsScalarArray(denseB, nrhsM, array, mumps->id.precision, &mumps->id.rhs_len, &mumps->id.rhs));

    /* handle condensation step of Schur complement (if any) */
    if (mumps->id.size_schur > 0) {
      if (mumps->id.ICNTL(26) < 0 || mumps->id.ICNTL(26) > 2) {
        second_solve = PETSC_TRUE;
        PetscCall(MatMumpsHandleSchur_Private(A, PETSC_FALSE)); // allocate id.redrhs
        mumps->id.ICNTL(26) = 1;                                /* condensation phase, i.e, to solve id.redrhs */
      } else if (mumps->id.ICNTL(26) == 1) PetscCall(MatMumpsHandleSchur_Private(A, PETSC_FALSE));
    }

    mumps->id.job = JOB_SOLVE;
    PetscMUMPS_c(mumps);
    PetscCheck(mumps->id.INFOG(1) >= 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "MUMPS error in solve: INFOG(1)=%d, INFO(2)=%d " MUMPS_MANUALS, mumps->id.INFOG(1), mumps->id.INFO(2));

    /* handle expansion step of Schur complement (if any) */
    if (second_solve) PetscCall(MatMumpsHandleSchur_Private(A, PETSC_TRUE));
    else if (mumps->id.ICNTL(26) == 1) { // condense the right hand side
      PetscCall(MatMumpsSolveSchur_Private(A));
      for (j = 0; j < nrhs; ++j)
        for (i = 0; i < mumps->id.size_schur; ++i) array[mumps->id.listvar_schur[i] - 1 + j * M] = ID_FIELD_GET(mumps->id, redrhs, i + j * mumps->id.lredrhs);
    }

    if (!denseB) { /* sparse B, restore ia, ja */
      PetscCall(MatSeqAIJRestoreArray(Bt, &aa));
      PetscCall(MatRestoreRowIJ(Bt, 1, PETSC_FALSE, PETSC_FALSE, &spnr, (const PetscInt **)&ia, (const PetscInt **)&ja, &flg));
      PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot restore IJ structure");
    }

    // no matter dense B or sparse B, solution is in id.rhs; convert it to array of X.
    PetscCall(MatMumpsCastMumpsScalarArray(nrhsM, mumps->id.precision, mumps->id.rhs, array));
    PetscCall(MatDenseRestoreArray(X, &array));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* parallel case: MUMPS requires rhs B to be centralized on the host! */
  PetscCheck(!mumps->id.ICNTL(19), PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Parallel Schur complements not yet supported from PETSc");

  /* create msol_loc to hold mumps local solution */
  isol_loc_save         = mumps->id.isol_loc; /* save these, as we want to reuse them in MatSolve() */
  sol_loc_save          = mumps->id.sol_loc;
  sol_loc_len_save      = mumps->id.sol_loc_len;
  mumps->id.isol_loc    = NULL; // an init state
  mumps->id.sol_loc     = NULL;
  mumps->id.sol_loc_len = 0;

  lsol_loc = mumps->id.lsol_loc;
  PetscCall(PetscIntMultError(nrhs, lsol_loc, &nlsol_loc)); /* length of sol_loc */
  PetscCall(PetscMalloc2(nlsol_loc, &sol_loc, lsol_loc, &isol_loc));
  PetscCall(MatMumpsMakeMumpsScalarArray(PETSC_FALSE, nlsol_loc, sol_loc, mumps->id.precision, &mumps->id.sol_loc_len, &mumps->id.sol_loc));
  mumps->id.isol_loc = isol_loc;

  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, nlsol_loc, (PetscScalar *)sol_loc, &msol_loc));

  if (denseB) {
    if (mumps->ICNTL20 == 10) {
      mumps->id.ICNTL(20) = 10; /* dense distributed RHS */
      PetscCall(MatDenseGetArrayRead(B, &barray));
      PetscCall(MatMumpsSetUpDistRHSInfo(A, nrhs, barray)); // put barray to rhs_loc
      PetscCall(MatDenseRestoreArrayRead(B, &barray));
      PetscCall(MatGetLocalSize(B, &m, NULL));
      PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)B), 1, nrhs * m, nrhsM, NULL, &v_mpi)); // will scatter the solution to v_mpi, which wraps X
    } else {
      mumps->id.ICNTL(20) = 0; /* dense centralized RHS */
      /* TODO: Because of non-contiguous indices, the created vecscatter scat_rhs is not done in MPI_Gather, resulting in
        very inefficient communication. An optimization is to use VecScatterCreateToZero to gather B to rank 0. Then on rank
        0, re-arrange B into desired order, which is a local operation.
      */

      /* scatter v_mpi to b_seq because MUMPS before 5.3.0 only supports centralized rhs */
      /* wrap dense rhs matrix B into a vector v_mpi */
      PetscCall(MatGetLocalSize(B, &m, NULL));
      PetscCall(MatDenseGetArrayRead(B, &barray));
      PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)B), 1, nrhs * m, nrhsM, barray, &v_mpi));
      PetscCall(MatDenseRestoreArrayRead(B, &barray));

      /* scatter v_mpi to b_seq in proc[0]. With ICNTL(20) = 0, MUMPS requires rhs to be centralized on the host! */
      if (!mumps->myid) {
        PetscInt *idx;
        /* idx: maps from k-th index of v_mpi to (i,j)-th global entry of B */
        PetscCall(PetscMalloc1(nrhsM, &idx));
        PetscCall(MatGetOwnershipRanges(B, &rstart));
        for (proc = 0, k = 0; proc < mumps->petsc_size; proc++) {
          for (j = 0; j < nrhs; j++) {
            for (i = rstart[proc]; i < rstart[proc + 1]; i++) idx[k++] = j * M + i;
          }
        }

        PetscCall(VecCreateSeq(PETSC_COMM_SELF, nrhsM, &b_seq));
        PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nrhsM, idx, PETSC_OWN_POINTER, &is_to));
        PetscCall(ISCreateStride(PETSC_COMM_SELF, nrhsM, 0, 1, &is_from));
      } else {
        PetscCall(VecCreateSeq(PETSC_COMM_SELF, 0, &b_seq));
        PetscCall(ISCreateStride(PETSC_COMM_SELF, 0, 0, 1, &is_to));
        PetscCall(ISCreateStride(PETSC_COMM_SELF, 0, 0, 1, &is_from));
      }

      PetscCall(VecScatterCreate(v_mpi, is_from, b_seq, is_to, &scat_rhs));
      PetscCall(VecScatterBegin(scat_rhs, v_mpi, b_seq, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(ISDestroy(&is_to));
      PetscCall(ISDestroy(&is_from));
      PetscCall(VecScatterEnd(scat_rhs, v_mpi, b_seq, INSERT_VALUES, SCATTER_FORWARD));

      if (!mumps->myid) { /* define rhs on the host */
        PetscCall(VecGetArrayRead(b_seq, &barray));
        PetscCall(MatMumpsMakeMumpsScalarArray(PETSC_TRUE, nrhsM, barray, mumps->id.precision, &mumps->id.rhs_len, &mumps->id.rhs));
        PetscCall(VecRestoreArrayRead(b_seq, &barray));
      }
    }
  } else { /* sparse B */
    b = (Mat_MPIAIJ *)Bt->data;

    /* wrap dense X into a vector v_mpi */
    PetscCall(MatGetLocalSize(X, &m, NULL));
    PetscCall(MatDenseGetArrayRead(X, &barray));
    PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)X), 1, nrhs * m, nrhsM, barray, &v_mpi));
    PetscCall(MatDenseRestoreArrayRead(X, &barray));

    if (!mumps->myid) {
      PetscCall(MatSeqAIJGetArray(b->A, &aa));
      PetscCall(MatGetRowIJ(b->A, 1, PETSC_FALSE, PETSC_FALSE, &spnr, (const PetscInt **)&ia, (const PetscInt **)&ja, &flg));
      PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot get IJ structure");
      PetscCall(PetscMUMPSIntCSRCast(mumps, spnr, ia, ja, &mumps->id.irhs_ptr, &mumps->id.irhs_sparse, &mumps->id.nz_rhs));
      PetscCall(MatMumpsMakeMumpsScalarArray(PETSC_TRUE, ((Mat_SeqAIJ *)b->A->data)->nz, aa, mumps->id.precision, &mumps->id.rhs_sparse_len, &mumps->id.rhs_sparse));
    } else {
      mumps->id.irhs_ptr    = NULL;
      mumps->id.irhs_sparse = NULL;
      mumps->id.nz_rhs      = 0;
      if (mumps->id.rhs_sparse_len) {
        PetscCall(PetscFree(mumps->id.rhs_sparse));
        mumps->id.rhs_sparse_len = 0;
      }
    }
  }

  /* solve phase */
  mumps->id.job = JOB_SOLVE;
  PetscMUMPS_c(mumps);
  PetscCheck(mumps->id.INFOG(1) >= 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "MUMPS error in solve: INFOG(1)=%d " MUMPS_MANUALS, mumps->id.INFOG(1));

  /* scatter mumps distributed solution to PETSc vector v_mpi, which shares local arrays with solution matrix X */
  PetscCall(MatDenseGetArray(X, &array));
  PetscCall(VecPlaceArray(v_mpi, array));

  /* create scatter scat_sol */
  PetscCall(MatGetOwnershipRanges(X, &rstart));
  /* iidx: index for scatter mumps solution to PETSc X */

  PetscCall(ISCreateStride(PETSC_COMM_SELF, nlsol_loc, 0, 1, &is_from));
  PetscCall(PetscMalloc1(nlsol_loc, &idxx));
  for (i = 0; i < lsol_loc; i++) {
    isol_loc[i] -= 1; /* change Fortran style to C style. isol_loc[i+j*lsol_loc] contains x[isol_loc[i]] in j-th vector */

    for (proc = 0; proc < mumps->petsc_size; proc++) {
      if (isol_loc[i] >= rstart[proc] && isol_loc[i] < rstart[proc + 1]) {
        myrstart = rstart[proc];
        k        = isol_loc[i] - myrstart;          /* local index on 1st column of PETSc vector X */
        iidx     = k + myrstart * nrhs;             /* maps mumps isol_loc[i] to PETSc index in X */
        m        = rstart[proc + 1] - rstart[proc]; /* rows of X for this proc */
        break;
      }
    }

    for (j = 0; j < nrhs; j++) idxx[i + j * lsol_loc] = iidx + j * m;
  }
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nlsol_loc, idxx, PETSC_COPY_VALUES, &is_to));
  PetscCall(MatMumpsCastMumpsScalarArray(nlsol_loc, mumps->id.precision, mumps->id.sol_loc, sol_loc)); // Vec msol_loc is created with sol_loc[]
  PetscCall(VecScatterCreate(msol_loc, is_from, v_mpi, is_to, &scat_sol));
  PetscCall(VecScatterBegin(scat_sol, msol_loc, v_mpi, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(ISDestroy(&is_from));
  PetscCall(ISDestroy(&is_to));
  PetscCall(VecScatterEnd(scat_sol, msol_loc, v_mpi, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(MatDenseRestoreArray(X, &array));

  if (mumps->id.sol_loc_len) { // in case we allocated intermediate buffers
    mumps->id.sol_loc_len = 0;
    PetscCall(PetscFree(mumps->id.sol_loc));
  }

  // restore old values
  mumps->id.sol_loc     = sol_loc_save;
  mumps->id.sol_loc_len = sol_loc_len_save;
  mumps->id.isol_loc    = isol_loc_save;

  PetscCall(PetscFree2(sol_loc, isol_loc));
  PetscCall(PetscFree(idxx));
  PetscCall(VecDestroy(&msol_loc));
  PetscCall(VecDestroy(&v_mpi));
  if (!denseB) {
    if (!mumps->myid) {
      b = (Mat_MPIAIJ *)Bt->data;
      PetscCall(MatSeqAIJRestoreArray(b->A, &aa));
      PetscCall(MatRestoreRowIJ(b->A, 1, PETSC_FALSE, PETSC_FALSE, &spnr, (const PetscInt **)&ia, (const PetscInt **)&ja, &flg));
      PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot restore IJ structure");
    }
  } else {
    if (mumps->ICNTL20 == 0) {
      PetscCall(VecDestroy(&b_seq));
      PetscCall(VecScatterDestroy(&scat_rhs));
    }
  }
  PetscCall(VecScatterDestroy(&scat_sol));
  PetscCall(PetscLogFlops(nrhs * PetscMax(0, 2.0 * (mumps->id.INFO(28) >= 0 ? mumps->id.INFO(28) : -1000000 * mumps->id.INFO(28)) - A->cmap->n)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMatSolveTranspose_MUMPS(Mat A, Mat B, Mat X)
{
  Mat_MUMPS          *mumps = (Mat_MUMPS *)A->data;
  const PetscMUMPSInt value = mumps->id.ICNTL(9);

  PetscFunctionBegin;
  mumps->id.ICNTL(9) = 0;
  PetscCall(MatMatSolve_MUMPS(A, B, X));
  mumps->id.ICNTL(9) = value;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMatTransposeSolve_MUMPS(Mat A, Mat Bt, Mat X)
{
  PetscBool flg;
  Mat       B;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)Bt, &flg, MATSEQAIJ, MATMPIAIJ, NULL));
  PetscCheck(flg, PetscObjectComm((PetscObject)Bt), PETSC_ERR_ARG_WRONG, "Matrix Bt must be MATAIJ matrix");

  /* Create B=Bt^T that uses Bt's data structure */
  PetscCall(MatCreateTranspose(Bt, &B));

  PetscCall(MatMatSolve_MUMPS(A, B, X));
  PetscCall(MatDestroy(&B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

#if !defined(PETSC_USE_COMPLEX)
/*
  input:
   F:        numeric factor
  output:
   nneg:     total number of negative pivots
   nzero:    total number of zero pivots
   npos:     (global dimension of F) - nneg - nzero
*/
static PetscErrorCode MatGetInertia_SBAIJMUMPS(Mat F, PetscInt *nneg, PetscInt *nzero, PetscInt *npos)
{
  Mat_MUMPS  *mumps = (Mat_MUMPS *)F->data;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)F), &size));
  /* MUMPS 4.3.1 calls ScaLAPACK when ICNTL(13)=0 (default), which does not offer the possibility to compute the inertia of a dense matrix. Set ICNTL(13)=1 to skip ScaLAPACK */
  PetscCheck(size <= 1 || mumps->id.ICNTL(13) == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "ICNTL(13)=%d. -mat_mumps_icntl_13 must be set as 1 for correct global matrix inertia", mumps->id.INFOG(13));

  if (nneg) *nneg = mumps->id.INFOG(12);
  if (nzero || npos) {
    PetscCheck(mumps->id.ICNTL(24) == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "-mat_mumps_icntl_24 must be set as 1 for null pivot row detection");
    if (nzero) *nzero = mumps->id.INFOG(28);
    if (npos) *npos = F->rmap->N - (mumps->id.INFOG(12) + mumps->id.INFOG(28));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

static PetscErrorCode MatMumpsGatherNonzerosOnMaster(MatReuse reuse, Mat_MUMPS *mumps)
{
  PetscMPIInt    nreqs;
  PetscMUMPSInt *irn, *jcn;
  PetscMPIInt    count;
  PetscCount     totnnz, remain;
  const PetscInt osize = mumps->omp_comm_size;
  PetscScalar   *val;

  PetscFunctionBegin;
  if (osize > 1) {
    if (reuse == MAT_INITIAL_MATRIX) {
      /* master first gathers counts of nonzeros to receive */
      if (mumps->is_omp_master) PetscCall(PetscMalloc1(osize, &mumps->recvcount));
      PetscCallMPI(MPI_Gather(&mumps->nnz, 1, MPIU_INT64, mumps->recvcount, 1, MPIU_INT64, 0 /*master*/, mumps->omp_comm));

      /* Then each computes number of send/recvs */
      if (mumps->is_omp_master) {
        /* Start from 1 since self communication is not done in MPI */
        nreqs = 0;
        for (PetscMPIInt i = 1; i < osize; i++) nreqs += (mumps->recvcount[i] + PETSC_MPI_INT_MAX - 1) / PETSC_MPI_INT_MAX;
      } else {
        nreqs = (PetscMPIInt)(((mumps->nnz + PETSC_MPI_INT_MAX - 1) / PETSC_MPI_INT_MAX));
      }
      PetscCall(PetscMalloc1(nreqs * 3, &mumps->reqs)); /* Triple the requests since we send irn, jcn and val separately */

      /* The following code is doing a very simple thing: omp_master rank gathers irn/jcn/val from others.
         MPI_Gatherv would be enough if it supports big counts > 2^31-1. Since it does not, and mumps->nnz
         might be a prime number > 2^31-1, we have to slice the message. Note omp_comm_size
         is very small, the current approach should have no extra overhead compared to MPI_Gatherv.
       */
      nreqs = 0; /* counter for actual send/recvs */
      if (mumps->is_omp_master) {
        totnnz = 0;

        for (PetscMPIInt i = 0; i < osize; i++) totnnz += mumps->recvcount[i]; /* totnnz = sum of nnz over omp_comm */
        PetscCall(PetscMalloc2(totnnz, &irn, totnnz, &jcn));
        PetscCall(PetscMalloc1(totnnz, &val));

        /* Self communication */
        PetscCall(PetscArraycpy(irn, mumps->irn, mumps->nnz));
        PetscCall(PetscArraycpy(jcn, mumps->jcn, mumps->nnz));
        PetscCall(PetscArraycpy(val, mumps->val, mumps->nnz));

        /* Replace mumps->irn/jcn etc on master with the newly allocated bigger arrays */
        PetscCall(PetscFree2(mumps->irn, mumps->jcn));
        PetscCall(PetscFree(mumps->val_alloc));
        mumps->nnz = totnnz;
        mumps->irn = irn;
        mumps->jcn = jcn;
        mumps->val = mumps->val_alloc = val;

        irn += mumps->recvcount[0]; /* recvcount[0] is old mumps->nnz on omp rank 0 */
        jcn += mumps->recvcount[0];
        val += mumps->recvcount[0];

        /* Remote communication */
        for (PetscMPIInt i = 1; i < osize; i++) {
          count  = (PetscMPIInt)PetscMin(mumps->recvcount[i], (PetscMPIInt)PETSC_MPI_INT_MAX);
          remain = mumps->recvcount[i] - count;
          while (count > 0) {
            PetscCallMPI(MPIU_Irecv(irn, count, MPIU_MUMPSINT, i, mumps->tag, mumps->omp_comm, &mumps->reqs[nreqs++]));
            PetscCallMPI(MPIU_Irecv(jcn, count, MPIU_MUMPSINT, i, mumps->tag, mumps->omp_comm, &mumps->reqs[nreqs++]));
            PetscCallMPI(MPIU_Irecv(val, count, MPIU_SCALAR, i, mumps->tag, mumps->omp_comm, &mumps->reqs[nreqs++]));
            irn += count;
            jcn += count;
            val += count;
            count = (PetscMPIInt)PetscMin(remain, (PetscMPIInt)PETSC_MPI_INT_MAX);
            remain -= count;
          }
        }
      } else {
        irn    = mumps->irn;
        jcn    = mumps->jcn;
        val    = mumps->val;
        count  = (PetscMPIInt)PetscMin(mumps->nnz, (PetscMPIInt)PETSC_MPI_INT_MAX);
        remain = mumps->nnz - count;
        while (count > 0) {
          PetscCallMPI(MPIU_Isend(irn, count, MPIU_MUMPSINT, 0, mumps->tag, mumps->omp_comm, &mumps->reqs[nreqs++]));
          PetscCallMPI(MPIU_Isend(jcn, count, MPIU_MUMPSINT, 0, mumps->tag, mumps->omp_comm, &mumps->reqs[nreqs++]));
          PetscCallMPI(MPIU_Isend(val, count, MPIU_SCALAR, 0, mumps->tag, mumps->omp_comm, &mumps->reqs[nreqs++]));
          irn += count;
          jcn += count;
          val += count;
          count = (PetscMPIInt)PetscMin(remain, (PetscMPIInt)PETSC_MPI_INT_MAX);
          remain -= count;
        }
      }
    } else {
      nreqs = 0;
      if (mumps->is_omp_master) {
        val = mumps->val + mumps->recvcount[0];
        for (PetscMPIInt i = 1; i < osize; i++) { /* Remote communication only since self data is already in place */
          count  = (PetscMPIInt)PetscMin(mumps->recvcount[i], (PetscMPIInt)PETSC_MPI_INT_MAX);
          remain = mumps->recvcount[i] - count;
          while (count > 0) {
            PetscCallMPI(MPIU_Irecv(val, count, MPIU_SCALAR, i, mumps->tag, mumps->omp_comm, &mumps->reqs[nreqs++]));
            val += count;
            count = (PetscMPIInt)PetscMin(remain, (PetscMPIInt)PETSC_MPI_INT_MAX);
            remain -= count;
          }
        }
      } else {
        val    = mumps->val;
        count  = (PetscMPIInt)PetscMin(mumps->nnz, (PetscMPIInt)PETSC_MPI_INT_MAX);
        remain = mumps->nnz - count;
        while (count > 0) {
          PetscCallMPI(MPIU_Isend(val, count, MPIU_SCALAR, 0, mumps->tag, mumps->omp_comm, &mumps->reqs[nreqs++]));
          val += count;
          count = (PetscMPIInt)PetscMin(remain, (PetscMPIInt)PETSC_MPI_INT_MAX);
          remain -= count;
        }
      }
    }
    PetscCallMPI(MPI_Waitall(nreqs, mumps->reqs, MPI_STATUSES_IGNORE));
    mumps->tag++; /* It is totally fine for above send/recvs to share one mpi tag */
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFactorNumeric_MUMPS(Mat F, Mat A, PETSC_UNUSED const MatFactorInfo *info)
{
  Mat_MUMPS *mumps = (Mat_MUMPS *)F->data;

  PetscFunctionBegin;
  if (mumps->id.INFOG(1) < 0 && !(mumps->id.INFOG(1) == -16 && mumps->id.INFOG(1) == 0)) {
    if (mumps->id.INFOG(1) == -6) PetscCall(PetscInfo(A, "MatFactorNumeric is called with singular matrix structure, INFOG(1)=%d, INFO(2)=%d\n", mumps->id.INFOG(1), mumps->id.INFO(2)));
    PetscCall(PetscInfo(A, "MatFactorNumeric is called after analysis phase fails, INFOG(1)=%d, INFO(2)=%d\n", mumps->id.INFOG(1), mumps->id.INFO(2)));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall((*mumps->ConvertToTriples)(A, 1, MAT_REUSE_MATRIX, mumps));
  PetscCall(MatMumpsGatherNonzerosOnMaster(MAT_REUSE_MATRIX, mumps));

  /* numerical factorization phase */
  mumps->id.job = JOB_FACTNUMERIC;
  if (!mumps->id.ICNTL(18)) { /* A is centralized */
    if (!mumps->myid) PetscCall(MatMumpsMakeMumpsScalarArray(PETSC_TRUE, mumps->nnz, mumps->val, mumps->id.precision, &mumps->id.a_len, &mumps->id.a));
  } else {
    PetscCall(MatMumpsMakeMumpsScalarArray(PETSC_TRUE, mumps->nnz, mumps->val, mumps->id.precision, &mumps->id.a_loc_len, &mumps->id.a_loc));
  }

  if (F->schur) {
    const PetscScalar *array;
    MUMPS_INT          size = mumps->id.size_schur;
    PetscCall(MatDenseGetArrayRead(F->schur, &array));
    PetscCall(MatMumpsMakeMumpsScalarArray(PETSC_FALSE, size * size, array, mumps->id.precision, &mumps->id.schur_len, &mumps->id.schur));
    PetscCall(MatDenseRestoreArrayRead(F->schur, &array));
  }

  PetscMUMPS_c(mumps);
  if (mumps->id.INFOG(1) < 0) {
    PetscCheck(!A->erroriffailure, PETSC_COMM_SELF, PETSC_ERR_LIB, "MUMPS error in numerical factorization: INFOG(1)=%d, INFO(2)=%d " MUMPS_MANUALS, mumps->id.INFOG(1), mumps->id.INFO(2));
    if (mumps->id.INFOG(1) == -10) {
      PetscCall(PetscInfo(F, "MUMPS error in numerical factorization: matrix is numerically singular, INFOG(1)=%d, INFO(2)=%d\n", mumps->id.INFOG(1), mumps->id.INFO(2)));
      F->factorerrortype = MAT_FACTOR_NUMERIC_ZEROPIVOT;
    } else if (mumps->id.INFOG(1) == -13) {
      PetscCall(PetscInfo(F, "MUMPS error in numerical factorization: INFOG(1)=%d, cannot allocate required memory %d megabytes\n", mumps->id.INFOG(1), mumps->id.INFO(2)));
      F->factorerrortype = MAT_FACTOR_OUTMEMORY;
    } else if (mumps->id.INFOG(1) == -8 || mumps->id.INFOG(1) == -9 || (-16 < mumps->id.INFOG(1) && mumps->id.INFOG(1) < -10)) {
      PetscCall(PetscInfo(F, "MUMPS error in numerical factorization: INFOG(1)=%d, INFO(2)=%d, problem with work array\n", mumps->id.INFOG(1), mumps->id.INFO(2)));
      F->factorerrortype = MAT_FACTOR_OUTMEMORY;
    } else {
      PetscCall(PetscInfo(F, "MUMPS error in numerical factorization: INFOG(1)=%d, INFO(2)=%d\n", mumps->id.INFOG(1), mumps->id.INFO(2)));
      F->factorerrortype = MAT_FACTOR_OTHER;
    }
  }
  PetscCheck(mumps->myid || mumps->id.ICNTL(16) <= 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "MUMPS error in numerical factorization: ICNTL(16)=%d " MUMPS_MANUALS, mumps->id.INFOG(16));

  F->assembled = PETSC_TRUE;

  if (F->schur) { /* reset Schur status to unfactored */
#if defined(PETSC_HAVE_CUDA)
    F->schur->offloadmask = PETSC_OFFLOAD_CPU;
#endif
    PetscScalar *array;
    PetscCall(MatDenseGetArray(F->schur, &array));
    PetscCall(MatMumpsCastMumpsScalarArray(mumps->id.size_schur * mumps->id.size_schur, mumps->id.precision, mumps->id.schur, array));
    PetscCall(MatDenseRestoreArray(F->schur, &array));
    if (mumps->id.ICNTL(19) == 1) { /* stored by rows */
      mumps->id.ICNTL(19) = 2;
      PetscCall(MatTranspose(F->schur, MAT_INPLACE_MATRIX, &F->schur));
    }
    PetscCall(MatFactorRestoreSchurComplement(F, NULL, MAT_FACTOR_SCHUR_UNFACTORED));
  }

  /* just to be sure that ICNTL(19) value returned by a call from MatMumpsGetIcntl is always consistent */
  if (!mumps->sym && mumps->id.ICNTL(19) && mumps->id.ICNTL(19) != 1) mumps->id.ICNTL(19) = 3;

  if (!mumps->is_omp_master) mumps->id.INFO(23) = 0;
  // MUMPS userguide: ISOL_loc should be allocated by the user between the factorization and the
  // solve phases. On exit from the solve phase, ISOL_loc(i) contains the index of the variables for
  // which the solution (in SOL_loc) is available on the local processor.
  // If successive calls to the solve phase (JOB= 3) are performed for a given matrix, ISOL_loc will
  // normally have the same contents for each of these calls. The only exception is the case of
  // unsymmetric matrices (SYM=1) when the transpose option is changed (see ICNTL(9)) and non
  // symmetric row/column exchanges (see ICNTL(6)) have occurred before the solve phase.
  if (mumps->petsc_size > 1) {
    PetscInt     lsol_loc;
    PetscScalar *array;

    /* distributed solution; Create x_seq=sol_loc for repeated use */
    if (mumps->x_seq) {
      PetscCall(VecScatterDestroy(&mumps->scat_sol));
      PetscCall(PetscFree(mumps->id.isol_loc));
      PetscCall(VecDestroy(&mumps->x_seq));
    }
    lsol_loc = mumps->id.INFO(23); /* length of sol_loc */
    PetscCall(PetscMalloc1(lsol_loc, &mumps->id.isol_loc));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, lsol_loc, &mumps->x_seq));
    PetscCall(VecGetArray(mumps->x_seq, &array));
    PetscCall(MatMumpsMakeMumpsScalarArray(PETSC_FALSE, lsol_loc, array, mumps->id.precision, &mumps->id.sol_loc_len, &mumps->id.sol_loc));
    PetscCall(VecRestoreArray(mumps->x_seq, &array));
    mumps->id.lsol_loc = (PetscMUMPSInt)lsol_loc;
  }
  PetscCall(PetscLogFlops((double)ID_RINFO_GET(mumps->id, 2)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Sets MUMPS options from the options database */
static PetscErrorCode MatSetFromOptions_MUMPS(Mat F, Mat A)
{
  Mat_MUMPS    *mumps = (Mat_MUMPS *)F->data;
  PetscReal     cntl;
  PetscMUMPSInt icntl = 0, size, *listvar_schur;
  PetscInt      info[80], i, ninfo = 80, rbs, cbs;
  PetscBool     flg   = PETSC_FALSE;
  PetscBool     schur = mumps->id.icntl ? (PetscBool)(mumps->id.ICNTL(26) == -1) : (PetscBool)(mumps->ICNTL26 == -1);
  void         *arr;

  PetscFunctionBegin;
  PetscOptionsBegin(PetscObjectComm((PetscObject)F), ((PetscObject)F)->prefix, "MUMPS Options", "Mat");
  if (mumps->id.job == JOB_NULL) { /* MatSetFromOptions_MUMPS() has never been called before */
    PetscPrecision precision  = PetscDefined(USE_REAL_SINGLE) ? PETSC_PRECISION_SINGLE : PETSC_PRECISION_DOUBLE;
    PetscInt       nthreads   = 0;
    PetscInt       nCNTL_pre  = mumps->CNTL_pre ? mumps->CNTL_pre[0] : 0;
    PetscInt       nICNTL_pre = mumps->ICNTL_pre ? mumps->ICNTL_pre[0] : 0;
    PetscMUMPSInt  nblk, *blkvar, *blkptr;

    mumps->petsc_comm = PetscObjectComm((PetscObject)A);
    PetscCallMPI(MPI_Comm_size(mumps->petsc_comm, &mumps->petsc_size));
    PetscCallMPI(MPI_Comm_rank(mumps->petsc_comm, &mumps->myid)); /* "if (!myid)" still works even if mumps_comm is different */

    PetscCall(PetscOptionsName("-mat_mumps_use_omp_threads", "Convert MPI processes into OpenMP threads", "None", &mumps->use_petsc_omp_support));
    if (mumps->use_petsc_omp_support) nthreads = -1; /* -1 will let PetscOmpCtrlCreate() guess a proper value when user did not supply one */
    /* do not use PetscOptionsInt() so that the option -mat_mumps_use_omp_threads is not displayed twice in the help */
    PetscCall(PetscOptionsGetInt(NULL, ((PetscObject)F)->prefix, "-mat_mumps_use_omp_threads", &nthreads, NULL));
    if (mumps->use_petsc_omp_support) {
      PetscCheck(!schur, PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot use -%smat_mumps_use_omp_threads with the Schur complement feature", ((PetscObject)F)->prefix ? ((PetscObject)F)->prefix : "");
#if defined(PETSC_HAVE_OPENMP_SUPPORT)
      PetscCall(PetscOmpCtrlCreate(mumps->petsc_comm, nthreads, &mumps->omp_ctrl));
      PetscCall(PetscOmpCtrlGetOmpComms(mumps->omp_ctrl, &mumps->omp_comm, &mumps->mumps_comm, &mumps->is_omp_master));
#else
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP_SYS, "The system does not have PETSc OpenMP support but you added the -%smat_mumps_use_omp_threads option. Configure PETSc with --with-openmp --download-hwloc (or --with-hwloc) to enable it, see more in MATSOLVERMUMPS manual",
              ((PetscObject)F)->prefix ? ((PetscObject)F)->prefix : "");
#endif
    } else {
      mumps->omp_comm      = PETSC_COMM_SELF;
      mumps->mumps_comm    = mumps->petsc_comm;
      mumps->is_omp_master = PETSC_TRUE;
    }
    PetscCallMPI(MPI_Comm_size(mumps->omp_comm, &mumps->omp_comm_size));
    mumps->reqs = NULL;
    mumps->tag  = 0;

    if (mumps->mumps_comm != MPI_COMM_NULL) {
      if (PetscDefined(HAVE_OPENMP_SUPPORT) && mumps->use_petsc_omp_support) {
        /* It looks like MUMPS does not dup the input comm. Dup a new comm for MUMPS to avoid any tag mismatches. */
        MPI_Comm comm;
        PetscCallMPI(MPI_Comm_dup(mumps->mumps_comm, &comm));
        mumps->mumps_comm = comm;
      } else PetscCall(PetscCommGetComm(mumps->petsc_comm, &mumps->mumps_comm));
    }

    mumps->id.comm_fortran = MPI_Comm_c2f(mumps->mumps_comm);
    mumps->id.job          = JOB_INIT;
    mumps->id.par          = 1; /* host participates factorizaton and solve */
    mumps->id.sym          = mumps->sym;

    size          = mumps->id.size_schur;
    arr           = mumps->id.schur;
    listvar_schur = mumps->id.listvar_schur;
    nblk          = mumps->id.nblk;
    blkvar        = mumps->id.blkvar;
    blkptr        = mumps->id.blkptr;
    if (PetscDefined(USE_DEBUG)) {
      for (PetscInt i = 0; i < size; i++)
        PetscCheck(listvar_schur[i] - 1 >= 0 && listvar_schur[i] - 1 < A->rmap->N, PETSC_COMM_SELF, PETSC_ERR_USER, "Invalid Schur index at position %" PetscInt_FMT "! %" PetscInt_FMT " must be in [0, %" PetscInt_FMT ")", i, (PetscInt)listvar_schur[i] - 1,
                   A->rmap->N);
    }

    PetscCall(PetscOptionsEnum("-pc_precision", "Precision used by MUMPS", "MATSOLVERMUMPS", PetscPrecisionTypes, (PetscEnum)precision, (PetscEnum *)&precision, NULL));
    PetscCheck(precision == PETSC_PRECISION_SINGLE || precision == PETSC_PRECISION_DOUBLE, PetscObjectComm((PetscObject)F), PETSC_ERR_SUP, "MUMPS does not support %s precision", PetscPrecisionTypes[precision]);
    PetscCheck(precision == PETSC_SCALAR_PRECISION || PetscDefined(HAVE_MUMPS_MIXED_PRECISION), PetscObjectComm((PetscObject)F), PETSC_ERR_USER, "Your MUMPS library does not support mixed precision, but which is needed with your specified PetscScalar");
    PetscCall(MatMumpsAllocateInternalID(&mumps->id, precision));

    PetscMUMPS_c(mumps);
    PetscCheck(mumps->id.INFOG(1) >= 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "MUMPS error: INFOG(1)=%d " MUMPS_MANUALS, mumps->id.INFOG(1));

    /* set PETSc-MUMPS default options - override MUMPS default */
    mumps->id.ICNTL(3) = 0;
    mumps->id.ICNTL(4) = 0;
    if (mumps->petsc_size == 1) {
      mumps->id.ICNTL(18) = 0; /* centralized assembled matrix input */
      mumps->id.ICNTL(7)  = 7; /* automatic choice of ordering done by the package */
    } else {
      mumps->id.ICNTL(18) = 3; /* distributed assembled matrix input */
      mumps->id.ICNTL(21) = 1; /* distributed solution */
    }
    if (nblk && blkptr) {
      mumps->id.ICNTL(15) = 1;
      mumps->id.nblk      = nblk;
      mumps->id.blkvar    = blkvar;
      mumps->id.blkptr    = blkptr;
    } else mumps->id.ICNTL(15) = 0;

    /* restore cached ICNTL and CNTL values */
    for (icntl = 0; icntl < nICNTL_pre; ++icntl) mumps->id.ICNTL(mumps->ICNTL_pre[1 + 2 * icntl]) = mumps->ICNTL_pre[2 + 2 * icntl];
    for (icntl = 0; icntl < nCNTL_pre; ++icntl) ID_CNTL_SET(mumps->id, (PetscInt)mumps->CNTL_pre[1 + 2 * icntl], mumps->CNTL_pre[2 + 2 * icntl]);

    PetscCall(PetscFree(mumps->ICNTL_pre));
    PetscCall(PetscFree(mumps->CNTL_pre));

    if (schur) {
      mumps->id.size_schur    = size;
      mumps->id.schur_lld     = size;
      mumps->id.schur         = arr;
      mumps->id.listvar_schur = listvar_schur;
      if (mumps->petsc_size > 1) {
        PetscBool gs; /* gs is false if any rank other than root has non-empty IS */

        mumps->id.ICNTL(19) = 1;                                                                            /* MUMPS returns Schur centralized on the host */
        gs                  = mumps->myid ? (mumps->id.size_schur ? PETSC_FALSE : PETSC_TRUE) : PETSC_TRUE; /* always true on root; false on others if their size != 0 */
        PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &gs, 1, MPI_C_BOOL, MPI_LAND, mumps->petsc_comm));
        PetscCheck(gs, PETSC_COMM_SELF, PETSC_ERR_SUP, "MUMPS distributed parallel Schur complements not yet supported from PETSc");
      } else {
        if (F->factortype == MAT_FACTOR_LU) {
          mumps->id.ICNTL(19) = 3; /* MUMPS returns full matrix */
        } else {
          mumps->id.ICNTL(19) = 2; /* MUMPS returns lower triangular part */
        }
      }
      mumps->id.ICNTL(26) = -1;
    }

    /* copy MUMPS default control values from master to slaves. Although slaves do not call MUMPS, they may access these values in code.
       For example, ICNTL(9) is initialized to 1 by MUMPS and slaves check ICNTL(9) in MatSolve_MUMPS.
     */
    PetscCallMPI(MPI_Bcast(mumps->id.icntl, 40, MPI_INT, 0, mumps->omp_comm));
    PetscCallMPI(MPI_Bcast(mumps->id.cntl, 15, MPIU_MUMPSREAL(&mumps->id), 0, mumps->omp_comm));

    mumps->scat_rhs = NULL;
    mumps->scat_sol = NULL;
  }
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_1", "ICNTL(1): output stream for error messages", "None", mumps->id.ICNTL(1), &icntl, &flg));
  if (flg) mumps->id.ICNTL(1) = icntl;
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_2", "ICNTL(2): output stream for diagnostic printing, statistics, and warning", "None", mumps->id.ICNTL(2), &icntl, &flg));
  if (flg) mumps->id.ICNTL(2) = icntl;
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_3", "ICNTL(3): output stream for global information, collected on the host", "None", mumps->id.ICNTL(3), &icntl, &flg));
  if (flg) mumps->id.ICNTL(3) = icntl;

  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_4", "ICNTL(4): level of printing (0 to 4)", "None", mumps->id.ICNTL(4), &icntl, &flg));
  if (flg) mumps->id.ICNTL(4) = icntl;
  if (mumps->id.ICNTL(4) || PetscLogPrintInfo) mumps->id.ICNTL(3) = 6; /* resume MUMPS default id.ICNTL(3) = 6 */

  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_6", "ICNTL(6): permutes to a zero-free diagonal and/or scale the matrix (0 to 7)", "None", mumps->id.ICNTL(6), &icntl, &flg));
  if (flg) mumps->id.ICNTL(6) = icntl;

  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_7", "ICNTL(7): computes a symmetric permutation in sequential analysis. 0=AMD, 2=AMF, 3=Scotch, 4=PORD, 5=Metis, 6=QAMD, and 7=auto(default)", "None", mumps->id.ICNTL(7), &icntl, &flg));
  if (flg) {
    PetscCheck(icntl != 1 && icntl >= 0 && icntl <= 7, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Valid values are 0=AMD, 2=AMF, 3=Scotch, 4=PORD, 5=Metis, 6=QAMD, and 7=auto");
    mumps->id.ICNTL(7) = icntl;
  }

  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_8", "ICNTL(8): scaling strategy (-2 to 8 or 77)", "None", mumps->id.ICNTL(8), &mumps->id.ICNTL(8), NULL));
  /* PetscCall(PetscOptionsInt("-mat_mumps_icntl_9","ICNTL(9): computes the solution using A or A^T","None",mumps->id.ICNTL(9),&mumps->id.ICNTL(9),NULL)); handled by MatSolveTranspose_MUMPS() */
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_10", "ICNTL(10): max num of refinements", "None", mumps->id.ICNTL(10), &mumps->id.ICNTL(10), NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_11", "ICNTL(11): statistics related to an error analysis (via -ksp_view)", "None", mumps->id.ICNTL(11), &mumps->id.ICNTL(11), NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_12", "ICNTL(12): an ordering strategy for symmetric matrices (0 to 3)", "None", mumps->id.ICNTL(12), &mumps->id.ICNTL(12), NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_13", "ICNTL(13): parallelism of the root node (enable ScaLAPACK) and its splitting", "None", mumps->id.ICNTL(13), &mumps->id.ICNTL(13), NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_14", "ICNTL(14): percentage increase in the estimated working space", "None", mumps->id.ICNTL(14), &mumps->id.ICNTL(14), NULL));
  PetscCall(MatGetBlockSizes(A, &rbs, &cbs));
  if (rbs == cbs && rbs > 1) mumps->id.ICNTL(15) = (PetscMUMPSInt)-rbs;
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_15", "ICNTL(15): compression of the input matrix resulting from a block format", "None", mumps->id.ICNTL(15), &mumps->id.ICNTL(15), &flg));
  if (flg) {
    if (mumps->id.ICNTL(15) < 0) PetscCheck((-mumps->id.ICNTL(15) % cbs == 0) && (-mumps->id.ICNTL(15) % rbs == 0), PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "The opposite of -mat_mumps_icntl_15 must be a multiple of the column and row blocksizes");
    else if (mumps->id.ICNTL(15) > 0) {
      const PetscInt *bsizes;
      PetscInt        nblocks, p, *blkptr = NULL;
      PetscMPIInt    *recvcounts, *displs, n;
      PetscMPIInt     rank, size = 0;

      PetscCall(MatGetVariableBlockSizes(A, &nblocks, &bsizes));
      flg = PETSC_TRUE;
      for (p = 0; p < nblocks; ++p) {
        if (bsizes[p] > 1) break;
      }
      if (p == nblocks) flg = PETSC_FALSE;
      PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, &flg, 1, MPI_C_BOOL, MPI_LOR, PetscObjectComm((PetscObject)A)));
      if (flg) { // if at least one process supplies variable block sizes and they are not all set to 1
        PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
        if (rank == 0) PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A), &size));
        PetscCall(PetscCalloc2(size, &recvcounts, size + 1, &displs));
        PetscCall(PetscMPIIntCast(nblocks, &n));
        PetscCallMPI(MPI_Gather(&n, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, PetscObjectComm((PetscObject)A)));
        for (PetscInt p = 0; p < size; ++p) displs[p + 1] = displs[p] + recvcounts[p];
        PetscCall(PetscMalloc1(displs[size] + 1, &blkptr));
        PetscCallMPI(MPI_Bcast(displs + size, 1, MPIU_INT, 0, PetscObjectComm((PetscObject)A)));
        PetscCallMPI(MPI_Gatherv(bsizes, n, MPIU_INT, blkptr + 1, recvcounts, displs, MPIU_INT, 0, PetscObjectComm((PetscObject)A)));
        if (rank == 0) {
          blkptr[0] = 1;
          for (PetscInt p = 0; p < n; ++p) blkptr[p + 1] += blkptr[p];
          PetscCall(MatMumpsSetBlk(F, displs[size], NULL, blkptr));
        }
        PetscCall(PetscFree2(recvcounts, displs));
        PetscCall(PetscFree(blkptr));
      }
    }
  }
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_19", "ICNTL(19): computes the Schur complement", "None", mumps->id.ICNTL(19), &mumps->id.ICNTL(19), NULL));
  if (mumps->id.ICNTL(19) <= 0 || mumps->id.ICNTL(19) > 3) { /* reset any schur data (if any) */
    PetscCall(MatDestroy(&F->schur));
    PetscCall(MatMumpsResetSchur_Private(mumps));
  }

  /* Two MPICH Fortran MPI_IN_PLACE binding bugs prevented the use of 'mpich + mumps'. One happened with "mpi4py + mpich + mumps",
     and was reported by Firedrake. See https://bitbucket.org/mpi4py/mpi4py/issues/162/mpi4py-initialization-breaks-fortran
     and a petsc-maint mailing list thread with subject 'MUMPS segfaults in parallel because of ...'
     This bug was fixed by https://github.com/pmodels/mpich/pull/4149. But the fix brought a new bug,
     see https://github.com/pmodels/mpich/issues/5589. This bug was fixed by https://github.com/pmodels/mpich/pull/5590.
     In short, we could not use distributed RHS until with MPICH v4.0b1 or we enabled a workaround in mumps-5.6.2+
   */
  mumps->ICNTL20 = 10; /* Distributed dense RHS, by default */
#if PETSC_PKG_MUMPS_VERSION_LT(5, 3, 0) || (PetscDefined(HAVE_MPICH) && MPICH_NUMVERSION < 40000101) || PetscDefined(HAVE_MSMPI)
  mumps->ICNTL20 = 0; /* Centralized dense RHS, if need be */
#endif
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_20", "ICNTL(20): give mumps centralized (0) or distributed (10) dense right-hand sides", "None", mumps->ICNTL20, &mumps->ICNTL20, &flg));
  PetscCheck(!flg || mumps->ICNTL20 == 10 || mumps->ICNTL20 == 0, PETSC_COMM_SELF, PETSC_ERR_SUP, "ICNTL(20)=%d is not supported by the PETSc/MUMPS interface. Allowed values are 0, 10", (int)mumps->ICNTL20);
#if PETSC_PKG_MUMPS_VERSION_LT(5, 3, 0)
  PetscCheck(!flg || mumps->ICNTL20 != 10, PETSC_COMM_SELF, PETSC_ERR_SUP, "ICNTL(20)=10 is not supported before MUMPS-5.3.0");
#endif
  /* PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_21","ICNTL(21): the distribution (centralized or distributed) of the solution vectors","None",mumps->id.ICNTL(21),&mumps->id.ICNTL(21),NULL)); we only use distributed solution vector */

  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_22", "ICNTL(22): in-core/out-of-core factorization and solve (0 or 1)", "None", mumps->id.ICNTL(22), &mumps->id.ICNTL(22), NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_23", "ICNTL(23): max size of the working memory (MB) that can allocate per processor", "None", mumps->id.ICNTL(23), &mumps->id.ICNTL(23), NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_24", "ICNTL(24): detection of null pivot rows (0 or 1)", "None", mumps->id.ICNTL(24), &mumps->id.ICNTL(24), NULL));
  if (mumps->id.ICNTL(24)) mumps->id.ICNTL(13) = 1; /* turn-off ScaLAPACK to help with the correct detection of null pivots */

  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_25", "ICNTL(25): computes a solution of a deficient matrix and a null space basis", "None", mumps->id.ICNTL(25), &mumps->id.ICNTL(25), NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_26", "ICNTL(26): drives the solution phase if a Schur complement matrix", "None", mumps->id.ICNTL(26), &mumps->id.ICNTL(26), NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_27", "ICNTL(27): controls the blocking size for multiple right-hand sides", "None", mumps->id.ICNTL(27), &mumps->id.ICNTL(27), NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_28", "ICNTL(28): use 1 for sequential analysis and ICNTL(7) ordering, or 2 for parallel analysis and ICNTL(29) ordering", "None", mumps->id.ICNTL(28), &mumps->id.ICNTL(28), NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_29", "ICNTL(29): parallel ordering 1 = ptscotch, 2 = parmetis", "None", mumps->id.ICNTL(29), &mumps->id.ICNTL(29), NULL));
  /* PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_30","ICNTL(30): compute user-specified set of entries in inv(A)","None",mumps->id.ICNTL(30),&mumps->id.ICNTL(30),NULL)); */ /* call MatMumpsGetInverse() directly */
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_31", "ICNTL(31): indicates which factors may be discarded during factorization", "None", mumps->id.ICNTL(31), &mumps->id.ICNTL(31), NULL));
  /* PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_32","ICNTL(32): performs the forward elimination of the right-hand sides during factorization","None",mumps->id.ICNTL(32),&mumps->id.ICNTL(32),NULL));  -- not supported by PETSc API */
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_33", "ICNTL(33): compute determinant", "None", mumps->id.ICNTL(33), &mumps->id.ICNTL(33), NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_35", "ICNTL(35): activates Block Low Rank (BLR) based factorization", "None", mumps->id.ICNTL(35), &mumps->id.ICNTL(35), NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_36", "ICNTL(36): choice of BLR factorization variant", "None", mumps->id.ICNTL(36), &mumps->id.ICNTL(36), NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_37", "ICNTL(37): compression of the contribution blocks (CB)", "None", mumps->id.ICNTL(37), &mumps->id.ICNTL(37), NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_38", "ICNTL(38): estimated compression rate of LU factors with BLR", "None", mumps->id.ICNTL(38), &mumps->id.ICNTL(38), NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_48", "ICNTL(48): multithreading with tree parallelism", "None", mumps->id.ICNTL(48), &mumps->id.ICNTL(48), NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_49", "ICNTL(49): compact workarray at the end of factorization phase", "None", mumps->id.ICNTL(49), &mumps->id.ICNTL(49), NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_56", "ICNTL(56): postponing and rank-revealing factorization", "None", mumps->id.ICNTL(56), &mumps->id.ICNTL(56), NULL));
  PetscCall(PetscOptionsMUMPSInt("-mat_mumps_icntl_58", "ICNTL(58): defines options for symbolic factorization", "None", mumps->id.ICNTL(58), &mumps->id.ICNTL(58), NULL));

  PetscCall(PetscOptionsReal("-mat_mumps_cntl_1", "CNTL(1): relative pivoting threshold", "None", (PetscReal)ID_CNTL_GET(mumps->id, 1), &cntl, &flg));
  if (flg) ID_CNTL_SET(mumps->id, 1, cntl);
  PetscCall(PetscOptionsReal("-mat_mumps_cntl_2", "CNTL(2): stopping criterion of refinement", "None", (PetscReal)ID_CNTL_GET(mumps->id, 2), &cntl, &flg));
  if (flg) ID_CNTL_SET(mumps->id, 2, cntl);
  PetscCall(PetscOptionsReal("-mat_mumps_cntl_3", "CNTL(3): absolute pivoting threshold", "None", (PetscReal)ID_CNTL_GET(mumps->id, 3), &cntl, &flg));
  if (flg) ID_CNTL_SET(mumps->id, 3, cntl);
  PetscCall(PetscOptionsReal("-mat_mumps_cntl_4", "CNTL(4): value for static pivoting", "None", (PetscReal)ID_CNTL_GET(mumps->id, 4), &cntl, &flg));
  if (flg) ID_CNTL_SET(mumps->id, 4, cntl);
  PetscCall(PetscOptionsReal("-mat_mumps_cntl_5", "CNTL(5): fixation for null pivots", "None", (PetscReal)ID_CNTL_GET(mumps->id, 5), &cntl, &flg));
  if (flg) ID_CNTL_SET(mumps->id, 5, cntl);
  PetscCall(PetscOptionsReal("-mat_mumps_cntl_7", "CNTL(7): dropping parameter used during BLR", "None", (PetscReal)ID_CNTL_GET(mumps->id, 7), &cntl, &flg));
  if (flg) ID_CNTL_SET(mumps->id, 7, cntl);

  PetscCall(PetscOptionsString("-mat_mumps_ooc_tmpdir", "out of core directory", "None", mumps->id.ooc_tmpdir, mumps->id.ooc_tmpdir, sizeof(mumps->id.ooc_tmpdir), NULL));

  PetscCall(PetscOptionsIntArray("-mat_mumps_view_info", "request INFO local to each processor", "", info, &ninfo, NULL));
  if (ninfo) {
    PetscCheck(ninfo <= 80, PETSC_COMM_SELF, PETSC_ERR_USER, "number of INFO %" PetscInt_FMT " must <= 80", ninfo);
    PetscCall(PetscMalloc1(ninfo, &mumps->info));
    mumps->ninfo = ninfo;
    for (i = 0; i < ninfo; i++) {
      PetscCheck(info[i] >= 0 && info[i] <= 80, PETSC_COMM_SELF, PETSC_ERR_USER, "index of INFO %" PetscInt_FMT " must between 1 and 80", ninfo);
      mumps->info[i] = info[i];
    }
  }
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFactorSymbolic_MUMPS_ReportIfError(Mat F, Mat A, PETSC_UNUSED const MatFactorInfo *info, Mat_MUMPS *mumps)
{
  PetscFunctionBegin;
  if (mumps->id.INFOG(1) < 0) {
    PetscCheck(!A->erroriffailure, PETSC_COMM_SELF, PETSC_ERR_LIB, "MUMPS error in analysis: INFOG(1)=%d " MUMPS_MANUALS, mumps->id.INFOG(1));
    if (mumps->id.INFOG(1) == -6) {
      PetscCall(PetscInfo(F, "MUMPS error in analysis: matrix is singular, INFOG(1)=%d, INFO(2)=%d\n", mumps->id.INFOG(1), mumps->id.INFO(2)));
      F->factorerrortype = MAT_FACTOR_STRUCT_ZEROPIVOT;
    } else if (mumps->id.INFOG(1) == -5 || mumps->id.INFOG(1) == -7) {
      PetscCall(PetscInfo(F, "MUMPS error in analysis: problem with work array, INFOG(1)=%d, INFO(2)=%d\n", mumps->id.INFOG(1), mumps->id.INFO(2)));
      F->factorerrortype = MAT_FACTOR_OUTMEMORY;
    } else {
      PetscCall(PetscInfo(F, "MUMPS error in analysis: INFOG(1)=%d, INFO(2)=%d " MUMPS_MANUALS "\n", mumps->id.INFOG(1), mumps->id.INFO(2)));
      F->factorerrortype = MAT_FACTOR_OTHER;
    }
  }
  if (!mumps->id.n) F->factorerrortype = MAT_FACTOR_NOERROR;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLUFactorSymbolic_AIJMUMPS(Mat F, Mat A, IS r, PETSC_UNUSED IS c, const MatFactorInfo *info)
{
  Mat_MUMPS     *mumps = (Mat_MUMPS *)F->data;
  Vec            b;
  const PetscInt M = A->rmap->N;

  PetscFunctionBegin;
  if (mumps->matstruc == SAME_NONZERO_PATTERN) {
    /* F is assembled by a previous call of MatLUFactorSymbolic_AIJMUMPS() */
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* Set MUMPS options from the options database */
  PetscCall(MatSetFromOptions_MUMPS(F, A));

  PetscCall((*mumps->ConvertToTriples)(A, 1, MAT_INITIAL_MATRIX, mumps));
  PetscCall(MatMumpsGatherNonzerosOnMaster(MAT_INITIAL_MATRIX, mumps));

  /* analysis phase */
  mumps->id.job = JOB_FACTSYMBOLIC;
  PetscCall(PetscMUMPSIntCast(M, &mumps->id.n));
  switch (mumps->id.ICNTL(18)) {
  case 0: /* centralized assembled matrix input */
    if (!mumps->myid) {
      mumps->id.nnz = mumps->nnz;
      mumps->id.irn = mumps->irn;
      mumps->id.jcn = mumps->jcn;
      if (1 < mumps->id.ICNTL(6) && mumps->id.ICNTL(6) < 7) PetscCall(MatMumpsMakeMumpsScalarArray(PETSC_TRUE, mumps->nnz, mumps->val, mumps->id.precision, &mumps->id.a_len, &mumps->id.a));
      if (r && mumps->id.ICNTL(7) == 7) {
        mumps->id.ICNTL(7) = 1;
        if (!mumps->myid) {
          const PetscInt *idx;
          PetscInt        i;

          PetscCall(PetscMalloc1(M, &mumps->id.perm_in));
          PetscCall(ISGetIndices(r, &idx));
          for (i = 0; i < M; i++) PetscCall(PetscMUMPSIntCast(idx[i] + 1, &mumps->id.perm_in[i])); /* perm_in[]: start from 1, not 0! */
          PetscCall(ISRestoreIndices(r, &idx));
        }
      }
    }
    break;
  case 3: /* distributed assembled matrix input (size>1) */
    mumps->id.nnz_loc = mumps->nnz;
    mumps->id.irn_loc = mumps->irn;
    mumps->id.jcn_loc = mumps->jcn;
    if (1 < mumps->id.ICNTL(6) && mumps->id.ICNTL(6) < 7) PetscCall(MatMumpsMakeMumpsScalarArray(PETSC_TRUE, mumps->nnz, mumps->val, mumps->id.precision, &mumps->id.a_loc_len, &mumps->id.a_loc));
    if (mumps->ICNTL20 == 0) { /* Centralized rhs. Create scatter scat_rhs for repeated use in MatSolve() */
      PetscCall(MatCreateVecs(A, NULL, &b));
      PetscCall(VecScatterCreateToZero(b, &mumps->scat_rhs, &mumps->b_seq));
      PetscCall(VecDestroy(&b));
    }
    break;
  }
  PetscMUMPS_c(mumps);
  PetscCall(MatFactorSymbolic_MUMPS_ReportIfError(F, A, info, mumps));

  F->ops->lufactornumeric   = MatFactorNumeric_MUMPS;
  F->ops->solve             = MatSolve_MUMPS;
  F->ops->solvetranspose    = MatSolveTranspose_MUMPS;
  F->ops->matsolve          = MatMatSolve_MUMPS;
  F->ops->mattransposesolve = MatMatTransposeSolve_MUMPS;
  F->ops->matsolvetranspose = MatMatSolveTranspose_MUMPS;

  mumps->matstruc = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Note the PETSc r and c permutations are ignored */
static PetscErrorCode MatLUFactorSymbolic_BAIJMUMPS(Mat F, Mat A, PETSC_UNUSED IS r, PETSC_UNUSED IS c, const MatFactorInfo *info)
{
  Mat_MUMPS     *mumps = (Mat_MUMPS *)F->data;
  Vec            b;
  const PetscInt M = A->rmap->N;

  PetscFunctionBegin;
  if (mumps->matstruc == SAME_NONZERO_PATTERN) {
    /* F is assembled by a previous call of MatLUFactorSymbolic_BAIJMUMPS() */
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* Set MUMPS options from the options database */
  PetscCall(MatSetFromOptions_MUMPS(F, A));

  PetscCall((*mumps->ConvertToTriples)(A, 1, MAT_INITIAL_MATRIX, mumps));
  PetscCall(MatMumpsGatherNonzerosOnMaster(MAT_INITIAL_MATRIX, mumps));

  /* analysis phase */
  mumps->id.job = JOB_FACTSYMBOLIC;
  PetscCall(PetscMUMPSIntCast(M, &mumps->id.n));
  switch (mumps->id.ICNTL(18)) {
  case 0: /* centralized assembled matrix input */
    if (!mumps->myid) {
      mumps->id.nnz = mumps->nnz;
      mumps->id.irn = mumps->irn;
      mumps->id.jcn = mumps->jcn;
      if (1 < mumps->id.ICNTL(6) && mumps->id.ICNTL(6) < 7) PetscCall(MatMumpsMakeMumpsScalarArray(PETSC_TRUE, mumps->nnz, mumps->val, mumps->id.precision, &mumps->id.a_len, &mumps->id.a));
    }
    break;
  case 3: /* distributed assembled matrix input (size>1) */
    mumps->id.nnz_loc = mumps->nnz;
    mumps->id.irn_loc = mumps->irn;
    mumps->id.jcn_loc = mumps->jcn;
    if (1 < mumps->id.ICNTL(6) && mumps->id.ICNTL(6) < 7) PetscCall(MatMumpsMakeMumpsScalarArray(PETSC_TRUE, mumps->nnz, mumps->val, mumps->id.precision, &mumps->id.a_loc_len, &mumps->id.a_loc));
    if (mumps->ICNTL20 == 0) { /* Centralized rhs. Create scatter scat_rhs for repeated use in MatSolve() */
      PetscCall(MatCreateVecs(A, NULL, &b));
      PetscCall(VecScatterCreateToZero(b, &mumps->scat_rhs, &mumps->b_seq));
      PetscCall(VecDestroy(&b));
    }
    break;
  }
  PetscMUMPS_c(mumps);
  PetscCall(MatFactorSymbolic_MUMPS_ReportIfError(F, A, info, mumps));

  F->ops->lufactornumeric   = MatFactorNumeric_MUMPS;
  F->ops->solve             = MatSolve_MUMPS;
  F->ops->solvetranspose    = MatSolveTranspose_MUMPS;
  F->ops->matsolvetranspose = MatMatSolveTranspose_MUMPS;

  mumps->matstruc = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Note the PETSc r permutation and factor info are ignored */
static PetscErrorCode MatCholeskyFactorSymbolic_MUMPS(Mat F, Mat A, PETSC_UNUSED IS r, const MatFactorInfo *info)
{
  Mat_MUMPS     *mumps = (Mat_MUMPS *)F->data;
  Vec            b;
  const PetscInt M = A->rmap->N;

  PetscFunctionBegin;
  if (mumps->matstruc == SAME_NONZERO_PATTERN) {
    /* F is assembled by a previous call of MatCholeskyFactorSymbolic_MUMPS() */
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  /* Set MUMPS options from the options database */
  PetscCall(MatSetFromOptions_MUMPS(F, A));

  PetscCall((*mumps->ConvertToTriples)(A, 1, MAT_INITIAL_MATRIX, mumps));
  PetscCall(MatMumpsGatherNonzerosOnMaster(MAT_INITIAL_MATRIX, mumps));

  /* analysis phase */
  mumps->id.job = JOB_FACTSYMBOLIC;
  PetscCall(PetscMUMPSIntCast(M, &mumps->id.n));
  switch (mumps->id.ICNTL(18)) {
  case 0: /* centralized assembled matrix input */
    if (!mumps->myid) {
      mumps->id.nnz = mumps->nnz;
      mumps->id.irn = mumps->irn;
      mumps->id.jcn = mumps->jcn;
      if (1 < mumps->id.ICNTL(6) && mumps->id.ICNTL(6) < 7) PetscCall(MatMumpsMakeMumpsScalarArray(PETSC_TRUE, mumps->nnz, mumps->val, mumps->id.precision, &mumps->id.a_len, &mumps->id.a));
    }
    break;
  case 3: /* distributed assembled matrix input (size>1) */
    mumps->id.nnz_loc = mumps->nnz;
    mumps->id.irn_loc = mumps->irn;
    mumps->id.jcn_loc = mumps->jcn;
    if (1 < mumps->id.ICNTL(6) && mumps->id.ICNTL(6) < 7) PetscCall(MatMumpsMakeMumpsScalarArray(PETSC_TRUE, mumps->nnz, mumps->val, mumps->id.precision, &mumps->id.a_loc_len, &mumps->id.a_loc));
    if (mumps->ICNTL20 == 0) { /* Centralized rhs. Create scatter scat_rhs for repeated use in MatSolve() */
      PetscCall(MatCreateVecs(A, NULL, &b));
      PetscCall(VecScatterCreateToZero(b, &mumps->scat_rhs, &mumps->b_seq));
      PetscCall(VecDestroy(&b));
    }
    break;
  }
  PetscMUMPS_c(mumps);
  PetscCall(MatFactorSymbolic_MUMPS_ReportIfError(F, A, info, mumps));

  F->ops->choleskyfactornumeric = MatFactorNumeric_MUMPS;
  F->ops->solve                 = MatSolve_MUMPS;
  F->ops->solvetranspose        = MatSolve_MUMPS;
  F->ops->matsolve              = MatMatSolve_MUMPS;
  F->ops->mattransposesolve     = MatMatTransposeSolve_MUMPS;
  F->ops->matsolvetranspose     = MatMatSolveTranspose_MUMPS;
#if defined(PETSC_USE_COMPLEX)
  F->ops->getinertia = NULL;
#else
  F->ops->getinertia = MatGetInertia_SBAIJMUMPS;
#endif

  mumps->matstruc = SAME_NONZERO_PATTERN;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatView_MUMPS(Mat A, PetscViewer viewer)
{
  PetscBool         isascii;
  PetscViewerFormat format;
  Mat_MUMPS        *mumps = (Mat_MUMPS *)A->data;

  PetscFunctionBegin;
  /* check if matrix is mumps type */
  if (A->ops->solve != MatSolve_MUMPS) PetscFunctionReturn(PETSC_SUCCESS);

  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_INFO || format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "MUMPS run parameters:\n"));
      if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
        PetscCall(PetscViewerASCIIPrintf(viewer, "  SYM (matrix type):                   %d\n", mumps->id.sym));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  PAR (host participation):            %d\n", mumps->id.par));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(1) (output for error):         %d\n", mumps->id.ICNTL(1)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(2) (output of diagnostic msg): %d\n", mumps->id.ICNTL(2)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(3) (output for global info):   %d\n", mumps->id.ICNTL(3)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(4) (level of printing):        %d\n", mumps->id.ICNTL(4)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(5) (input mat struct):         %d\n", mumps->id.ICNTL(5)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(6) (matrix prescaling):        %d\n", mumps->id.ICNTL(6)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(7) (sequential matrix ordering):%d\n", mumps->id.ICNTL(7)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(8) (scaling strategy):         %d\n", mumps->id.ICNTL(8)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(10) (max num of refinements):  %d\n", mumps->id.ICNTL(10)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(11) (error analysis):          %d\n", mumps->id.ICNTL(11)));
        if (mumps->id.ICNTL(11) > 0) {
          PetscCall(PetscViewerASCIIPrintf(viewer, "    RINFOG(4) (inf norm of input mat):        %g\n", (double)ID_RINFOG_GET(mumps->id, 4)));
          PetscCall(PetscViewerASCIIPrintf(viewer, "    RINFOG(5) (inf norm of solution):         %g\n", (double)ID_RINFOG_GET(mumps->id, 5)));
          PetscCall(PetscViewerASCIIPrintf(viewer, "    RINFOG(6) (inf norm of residual):         %g\n", (double)ID_RINFOG_GET(mumps->id, 6)));
          PetscCall(PetscViewerASCIIPrintf(viewer, "    RINFOG(7),RINFOG(8) (backward error est): %g, %g\n", (double)ID_RINFOG_GET(mumps->id, 7), (double)ID_RINFOG_GET(mumps->id, 8)));
          PetscCall(PetscViewerASCIIPrintf(viewer, "    RINFOG(9) (error estimate):               %g\n", (double)ID_RINFOG_GET(mumps->id, 9)));
          PetscCall(PetscViewerASCIIPrintf(viewer, "    RINFOG(10),RINFOG(11)(condition numbers): %g, %g\n", (double)ID_RINFOG_GET(mumps->id, 10), (double)ID_RINFOG_GET(mumps->id, 11)));
        }
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(12) (efficiency control):                         %d\n", mumps->id.ICNTL(12)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(13) (sequential factorization of the root node):  %d\n", mumps->id.ICNTL(13)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(14) (percentage of estimated workspace increase): %d\n", mumps->id.ICNTL(14)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(15) (compression of the input matrix):            %d\n", mumps->id.ICNTL(15)));
        /* ICNTL(15-17) not used */
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(18) (input mat struct):                           %d\n", mumps->id.ICNTL(18)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(19) (Schur complement info):                      %d\n", mumps->id.ICNTL(19)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(20) (RHS sparse pattern):                         %d\n", mumps->id.ICNTL(20)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(21) (solution struct):                            %d\n", mumps->id.ICNTL(21)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(22) (in-core/out-of-core facility):               %d\n", mumps->id.ICNTL(22)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(23) (max size of memory can be allocated locally):%d\n", mumps->id.ICNTL(23)));

        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(24) (detection of null pivot rows):               %d\n", mumps->id.ICNTL(24)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(25) (computation of a null space basis):          %d\n", mumps->id.ICNTL(25)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(26) (Schur options for RHS or solution):          %d\n", mumps->id.ICNTL(26)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(27) (blocking size for multiple RHS):             %d\n", mumps->id.ICNTL(27)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(28) (use parallel or sequential ordering):        %d\n", mumps->id.ICNTL(28)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(29) (parallel ordering):                          %d\n", mumps->id.ICNTL(29)));

        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(30) (user-specified set of entries in inv(A)):    %d\n", mumps->id.ICNTL(30)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(31) (factors is discarded in the solve phase):    %d\n", mumps->id.ICNTL(31)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(33) (compute determinant):                        %d\n", mumps->id.ICNTL(33)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(35) (activate BLR based factorization):           %d\n", mumps->id.ICNTL(35)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(36) (choice of BLR factorization variant):        %d\n", mumps->id.ICNTL(36)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(37) (compression of the contribution blocks):     %d\n", mumps->id.ICNTL(37)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(38) (estimated compression rate of LU factors):   %d\n", mumps->id.ICNTL(38)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(48) (multithreading with tree parallelism):       %d\n", mumps->id.ICNTL(48)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(49) (compact workarray at the end of factorization phase):%d\n", mumps->id.ICNTL(49)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(56) (postponing and rank-revealing factorization):%d\n", mumps->id.ICNTL(56)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  ICNTL(58) (options for symbolic factorization):         %d\n", mumps->id.ICNTL(58)));

        PetscCall(PetscViewerASCIIPrintf(viewer, "  CNTL(1) (relative pivoting threshold):      %g\n", (double)ID_CNTL_GET(mumps->id, 1)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  CNTL(2) (stopping criterion of refinement): %g\n", (double)ID_CNTL_GET(mumps->id, 2)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  CNTL(3) (absolute pivoting threshold):      %g\n", (double)ID_CNTL_GET(mumps->id, 3)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  CNTL(4) (value of static pivoting):         %g\n", (double)ID_CNTL_GET(mumps->id, 4)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  CNTL(5) (fixation for null pivots):         %g\n", (double)ID_CNTL_GET(mumps->id, 5)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  CNTL(7) (dropping parameter for BLR):       %g\n", (double)ID_CNTL_GET(mumps->id, 7)));

        /* information local to each processor */
        PetscCall(PetscViewerASCIIPrintf(viewer, "  RINFO(1) (local estimated flops for the elimination after analysis):\n"));
        PetscCall(PetscViewerASCIIPushSynchronized(viewer));
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "    [%d] %g\n", mumps->myid, (double)ID_RINFO_GET(mumps->id, 1)));
        PetscCall(PetscViewerFlush(viewer));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  RINFO(2) (local estimated flops for the assembly after factorization):\n"));
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "    [%d] %g\n", mumps->myid, (double)ID_RINFO_GET(mumps->id, 2)));
        PetscCall(PetscViewerFlush(viewer));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  RINFO(3) (local estimated flops for the elimination after factorization):\n"));
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "    [%d] %g\n", mumps->myid, (double)ID_RINFO_GET(mumps->id, 3)));
        PetscCall(PetscViewerFlush(viewer));

        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFO(15) (estimated size of (in MB) MUMPS internal data for running numerical factorization):\n"));
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "    [%d] %d\n", mumps->myid, mumps->id.INFO(15)));
        PetscCall(PetscViewerFlush(viewer));

        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFO(16) (size of (in MB) MUMPS internal data used during numerical factorization):\n"));
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "    [%d] %d\n", mumps->myid, mumps->id.INFO(16)));
        PetscCall(PetscViewerFlush(viewer));

        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFO(23) (num of pivots eliminated on this processor after factorization):\n"));
        PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "    [%d] %d\n", mumps->myid, mumps->id.INFO(23)));
        PetscCall(PetscViewerFlush(viewer));

        if (mumps->ninfo && mumps->ninfo <= 80) {
          PetscInt i;
          for (i = 0; i < mumps->ninfo; i++) {
            PetscCall(PetscViewerASCIIPrintf(viewer, "  INFO(%" PetscInt_FMT "):\n", mumps->info[i]));
            PetscCall(PetscViewerASCIISynchronizedPrintf(viewer, "    [%d] %d\n", mumps->myid, mumps->id.INFO(mumps->info[i])));
            PetscCall(PetscViewerFlush(viewer));
          }
        }
        PetscCall(PetscViewerASCIIPopSynchronized(viewer));
      } else PetscCall(PetscViewerASCIIPrintf(viewer, "  Use -%sksp_view ::ascii_info_detail to display information for all processes\n", ((PetscObject)A)->prefix ? ((PetscObject)A)->prefix : ""));

      if (mumps->myid == 0) { /* information from the host */
        PetscCall(PetscViewerASCIIPrintf(viewer, "  RINFOG(1) (global estimated flops for the elimination after analysis): %g\n", (double)ID_RINFOG_GET(mumps->id, 1)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  RINFOG(2) (global estimated flops for the assembly after factorization): %g\n", (double)ID_RINFOG_GET(mumps->id, 2)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  RINFOG(3) (global estimated flops for the elimination after factorization): %g\n", (double)ID_RINFOG_GET(mumps->id, 3)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  (RINFOG(12) RINFOG(13))*2^INFOG(34) (determinant): (%g,%g)*(2^%d)\n", (double)ID_RINFOG_GET(mumps->id, 12), (double)ID_RINFOG_GET(mumps->id, 13), mumps->id.INFOG(34)));

        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(3) (estimated real workspace for factors on all processors after analysis): %d\n", mumps->id.INFOG(3)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(4) (estimated integer workspace for factors on all processors after analysis): %d\n", mumps->id.INFOG(4)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(5) (estimated maximum front size in the complete tree): %d\n", mumps->id.INFOG(5)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(6) (number of nodes in the complete tree): %d\n", mumps->id.INFOG(6)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(7) (ordering option effectively used after analysis): %d\n", mumps->id.INFOG(7)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(8) (structural symmetry in percent of the permuted matrix after analysis): %d\n", mumps->id.INFOG(8)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(9) (total real/complex workspace to store the matrix factors after factorization): %d\n", mumps->id.INFOG(9)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(10) (total integer space store the matrix factors after factorization): %d\n", mumps->id.INFOG(10)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(11) (order of largest frontal matrix after factorization): %d\n", mumps->id.INFOG(11)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(12) (number of off-diagonal pivots): %d\n", mumps->id.INFOG(12)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(13) (number of delayed pivots after factorization): %d\n", mumps->id.INFOG(13)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(14) (number of memory compress after factorization): %d\n", mumps->id.INFOG(14)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(15) (number of steps of iterative refinement after solution): %d\n", mumps->id.INFOG(15)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(16) (estimated size (in MB) of all MUMPS internal data for factorization after analysis: value on the most memory consuming processor): %d\n", mumps->id.INFOG(16)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(17) (estimated size of all MUMPS internal data for factorization after analysis: sum over all processors): %d\n", mumps->id.INFOG(17)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(18) (size of all MUMPS internal data allocated during factorization: value on the most memory consuming processor): %d\n", mumps->id.INFOG(18)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(19) (size of all MUMPS internal data allocated during factorization: sum over all processors): %d\n", mumps->id.INFOG(19)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(20) (estimated number of entries in the factors): %d\n", mumps->id.INFOG(20)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(21) (size in MB of memory effectively used during factorization - value on the most memory consuming processor): %d\n", mumps->id.INFOG(21)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(22) (size in MB of memory effectively used during factorization - sum over all processors): %d\n", mumps->id.INFOG(22)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(23) (after analysis: value of ICNTL(6) effectively used): %d\n", mumps->id.INFOG(23)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(24) (after analysis: value of ICNTL(12) effectively used): %d\n", mumps->id.INFOG(24)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(25) (after factorization: number of pivots modified by static pivoting): %d\n", mumps->id.INFOG(25)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(28) (after factorization: number of null pivots encountered): %d\n", mumps->id.INFOG(28)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(29) (after factorization: effective number of entries in the factors (sum over all processors)): %d\n", mumps->id.INFOG(29)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(30, 31) (after solution: size in Mbytes of memory used during solution phase): %d, %d\n", mumps->id.INFOG(30), mumps->id.INFOG(31)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(32) (after analysis: type of analysis done): %d\n", mumps->id.INFOG(32)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(33) (value used for ICNTL(8)): %d\n", mumps->id.INFOG(33)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(34) (exponent of the determinant if determinant is requested): %d\n", mumps->id.INFOG(34)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(35) (after factorization: number of entries taking into account BLR factor compression - sum over all processors): %d\n", mumps->id.INFOG(35)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(36) (after analysis: estimated size of all MUMPS internal data for running BLR in-core - value on the most memory consuming processor): %d\n", mumps->id.INFOG(36)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(37) (after analysis: estimated size of all MUMPS internal data for running BLR in-core - sum over all processors): %d\n", mumps->id.INFOG(37)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(38) (after analysis: estimated size of all MUMPS internal data for running BLR out-of-core - value on the most memory consuming processor): %d\n", mumps->id.INFOG(38)));
        PetscCall(PetscViewerASCIIPrintf(viewer, "  INFOG(39) (after analysis: estimated size of all MUMPS internal data for running BLR out-of-core - sum over all processors): %d\n", mumps->id.INFOG(39)));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetInfo_MUMPS(Mat A, PETSC_UNUSED MatInfoType flag, MatInfo *info)
{
  Mat_MUMPS *mumps = (Mat_MUMPS *)A->data;

  PetscFunctionBegin;
  info->block_size        = 1.0;
  info->nz_allocated      = mumps->id.INFOG(20) >= 0 ? mumps->id.INFOG(20) : -1000000 * mumps->id.INFOG(20);
  info->nz_used           = mumps->id.INFOG(20) >= 0 ? mumps->id.INFOG(20) : -1000000 * mumps->id.INFOG(20);
  info->nz_unneeded       = 0.0;
  info->assemblies        = 0.0;
  info->mallocs           = 0.0;
  info->memory            = 0.0;
  info->fill_ratio_given  = 0;
  info->fill_ratio_needed = 0;
  info->factor_mallocs    = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFactorSetSchurIS_MUMPS(Mat F, IS is)
{
  Mat_MUMPS         *mumps = (Mat_MUMPS *)F->data;
  const PetscScalar *arr;
  const PetscInt    *idxs;
  PetscInt           size, i;

  PetscFunctionBegin;
  PetscCall(ISGetLocalSize(is, &size));
  /* Schur complement matrix */
  PetscCall(MatDestroy(&F->schur));
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, size, size, NULL, &F->schur));
  PetscCall(MatDenseGetArrayRead(F->schur, &arr));
  // don't allocate mumps->id.schur[] now as its precision is yet to know
  PetscCall(PetscMUMPSIntCast(size, &mumps->id.size_schur));
  PetscCall(PetscMUMPSIntCast(size, &mumps->id.schur_lld));
  PetscCall(MatDenseRestoreArrayRead(F->schur, &arr));
  if (mumps->sym == 1) PetscCall(MatSetOption(F->schur, MAT_SPD, PETSC_TRUE));

  /* MUMPS expects Fortran style indices */
  PetscCall(PetscFree(mumps->id.listvar_schur));
  PetscCall(PetscMalloc1(size, &mumps->id.listvar_schur));
  PetscCall(ISGetIndices(is, &idxs));
  for (i = 0; i < size; i++) PetscCall(PetscMUMPSIntCast(idxs[i] + 1, &mumps->id.listvar_schur[i]));
  PetscCall(ISRestoreIndices(is, &idxs));
  /* set a special value of ICNTL (not handled my MUMPS) to be used in the solve phase by PETSc */
  if (mumps->id.icntl) mumps->id.ICNTL(26) = -1;
  else mumps->ICNTL26 = -1;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFactorCreateSchurComplement_MUMPS(Mat F, Mat *S)
{
  Mat          St;
  Mat_MUMPS   *mumps = (Mat_MUMPS *)F->data;
  PetscScalar *array;
  PetscInt     i, j, N = mumps->id.size_schur;

  PetscFunctionBegin;
  PetscCheck(mumps->id.ICNTL(19), PetscObjectComm((PetscObject)F), PETSC_ERR_ORDER, "Schur complement mode not selected! Call MatFactorSetSchurIS() to enable it");
  PetscCall(MatCreate(PETSC_COMM_SELF, &St));
  PetscCall(MatSetSizes(St, PETSC_DECIDE, PETSC_DECIDE, mumps->id.size_schur, mumps->id.size_schur));
  PetscCall(MatSetType(St, MATDENSE));
  PetscCall(MatSetUp(St));
  PetscCall(MatDenseGetArray(St, &array));
  if (!mumps->sym) {                /* MUMPS always return a full matrix */
    if (mumps->id.ICNTL(19) == 1) { /* stored by rows */
      for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) array[j * N + i] = ID_FIELD_GET(mumps->id, schur, i * N + j);
      }
    } else { /* stored by columns */
      PetscCall(MatMumpsCastMumpsScalarArray(N * N, mumps->id.precision, mumps->id.schur, array));
    }
  } else {                          /* either full or lower-triangular (not packed) */
    if (mumps->id.ICNTL(19) == 2) { /* lower triangular stored by columns */
      for (i = 0; i < N; i++) {
        for (j = i; j < N; j++) array[i * N + j] = array[j * N + i] = ID_FIELD_GET(mumps->id, schur, i * N + j);
      }
    } else if (mumps->id.ICNTL(19) == 3) { /* full matrix */
      PetscCall(MatMumpsCastMumpsScalarArray(N * N, mumps->id.precision, mumps->id.schur, array));
    } else { /* ICNTL(19) == 1 lower triangular stored by rows */
      for (i = 0; i < N; i++) {
        for (j = 0; j < i + 1; j++) array[i * N + j] = array[j * N + i] = ID_FIELD_GET(mumps->id, schur, i * N + j);
      }
    }
  }
  PetscCall(MatDenseRestoreArray(St, &array));
  *S = St;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMumpsSetIcntl_MUMPS(Mat F, PetscInt icntl, PetscInt ival)
{
  Mat_MUMPS *mumps = (Mat_MUMPS *)F->data;

  PetscFunctionBegin;
  if (mumps->id.job == JOB_NULL) {                                            /* need to cache icntl and ival since PetscMUMPS_c() has never been called */
    PetscMUMPSInt i, nICNTL_pre = mumps->ICNTL_pre ? mumps->ICNTL_pre[0] : 0; /* number of already cached ICNTL */
    for (i = 0; i < nICNTL_pre; ++i)
      if (mumps->ICNTL_pre[1 + 2 * i] == icntl) break; /* is this ICNTL already cached? */
    if (i == nICNTL_pre) {                             /* not already cached */
      if (i > 0) PetscCall(PetscRealloc(sizeof(PetscMUMPSInt) * (2 * nICNTL_pre + 3), &mumps->ICNTL_pre));
      else PetscCall(PetscCalloc(sizeof(PetscMUMPSInt) * 3, &mumps->ICNTL_pre));
      mumps->ICNTL_pre[0]++;
    }
    mumps->ICNTL_pre[1 + 2 * i] = (PetscMUMPSInt)icntl;
    PetscCall(PetscMUMPSIntCast(ival, mumps->ICNTL_pre + 2 + 2 * i));
  } else PetscCall(PetscMUMPSIntCast(ival, &mumps->id.ICNTL(icntl)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMumpsGetIcntl_MUMPS(Mat F, PetscInt icntl, PetscInt *ival)
{
  Mat_MUMPS *mumps = (Mat_MUMPS *)F->data;

  PetscFunctionBegin;
  if (mumps->id.job == JOB_NULL) {
    PetscInt i, nICNTL_pre = mumps->ICNTL_pre ? mumps->ICNTL_pre[0] : 0;
    *ival = 0;
    for (i = 0; i < nICNTL_pre; ++i) {
      if (mumps->ICNTL_pre[1 + 2 * i] == icntl) *ival = mumps->ICNTL_pre[2 + 2 * i];
    }
  } else *ival = mumps->id.ICNTL(icntl);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMumpsSetIcntl - Set MUMPS parameter ICNTL() <https://mumps-solver.org/index.php?page=doc>

  Logically Collective

  Input Parameters:
+ F     - the factored matrix obtained by calling `MatGetFactor()` with a `MatSolverType` of `MATSOLVERMUMPS` and a `MatFactorType` of `MAT_FACTOR_LU` or `MAT_FACTOR_CHOLESKY`
. icntl - index of MUMPS parameter array `ICNTL()`
- ival  - value of MUMPS `ICNTL(icntl)`

  Options Database Key:
. -mat_mumps_icntl_<icntl> <ival> - change the option numbered `icntl` to `ival`

  Level: beginner

  Note:
  Ignored if MUMPS is not installed or `F` is not a MUMPS matrix

.seealso: [](ch_matrices), `Mat`, `MatGetFactor()`, `MatMumpsGetIcntl()`, `MatMumpsSetCntl()`, `MatMumpsGetCntl()`, `MatMumpsGetInfo()`, `MatMumpsGetInfog()`, `MatMumpsGetRinfo()`, `MatMumpsGetRinfog()`
@*/
PetscErrorCode MatMumpsSetIcntl(Mat F, PetscInt icntl, PetscInt ival)
{
  PetscFunctionBegin;
  PetscValidType(F, 1);
  PetscCheck(F->factortype, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONGSTATE, "Only for factored matrix");
  PetscValidLogicalCollectiveInt(F, icntl, 2);
  PetscValidLogicalCollectiveInt(F, ival, 3);
  PetscCheck((icntl >= 1 && icntl <= 38) || icntl == 48 || icntl == 49 || icntl == 56 || icntl == 58, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONG, "Unsupported ICNTL value %" PetscInt_FMT, icntl);
  PetscTryMethod(F, "MatMumpsSetIcntl_C", (Mat, PetscInt, PetscInt), (F, icntl, ival));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMumpsGetIcntl - Get MUMPS parameter ICNTL() <https://mumps-solver.org/index.php?page=doc>

  Logically Collective

  Input Parameters:
+ F     - the factored matrix obtained by calling `MatGetFactor()` with a `MatSolverType` of `MATSOLVERMUMPS` and a `MatFactorType` of `MAT_FACTOR_LU` or `MAT_FACTOR_CHOLESKY`
- icntl - index of MUMPS parameter array ICNTL()

  Output Parameter:
. ival - value of MUMPS ICNTL(icntl)

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MatGetFactor()`, `MatMumpsSetIcntl()`, `MatMumpsSetCntl()`, `MatMumpsGetCntl()`, `MatMumpsGetInfo()`, `MatMumpsGetInfog()`, `MatMumpsGetRinfo()`, `MatMumpsGetRinfog()`
@*/
PetscErrorCode MatMumpsGetIcntl(Mat F, PetscInt icntl, PetscInt *ival)
{
  PetscFunctionBegin;
  PetscValidType(F, 1);
  PetscCheck(F->factortype, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONGSTATE, "Only for factored matrix");
  PetscValidLogicalCollectiveInt(F, icntl, 2);
  PetscAssertPointer(ival, 3);
  PetscCheck((icntl >= 1 && icntl <= 38) || icntl == 48 || icntl == 49 || icntl == 56 || icntl == 58, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONG, "Unsupported ICNTL value %" PetscInt_FMT, icntl);
  PetscUseMethod(F, "MatMumpsGetIcntl_C", (Mat, PetscInt, PetscInt *), (F, icntl, ival));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMumpsSetCntl_MUMPS(Mat F, PetscInt icntl, PetscReal val)
{
  Mat_MUMPS *mumps = (Mat_MUMPS *)F->data;

  PetscFunctionBegin;
  if (mumps->id.job == JOB_NULL) {
    PetscInt i, nCNTL_pre = mumps->CNTL_pre ? mumps->CNTL_pre[0] : 0;
    for (i = 0; i < nCNTL_pre; ++i)
      if (mumps->CNTL_pre[1 + 2 * i] == icntl) break;
    if (i == nCNTL_pre) {
      if (i > 0) PetscCall(PetscRealloc(sizeof(PetscReal) * (2 * nCNTL_pre + 3), &mumps->CNTL_pre));
      else PetscCall(PetscCalloc(sizeof(PetscReal) * 3, &mumps->CNTL_pre));
      mumps->CNTL_pre[0]++;
    }
    mumps->CNTL_pre[1 + 2 * i] = icntl;
    mumps->CNTL_pre[2 + 2 * i] = val;
  } else ID_CNTL_SET(mumps->id, icntl, val);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMumpsGetCntl_MUMPS(Mat F, PetscInt icntl, PetscReal *val)
{
  Mat_MUMPS *mumps = (Mat_MUMPS *)F->data;

  PetscFunctionBegin;
  if (mumps->id.job == JOB_NULL) {
    PetscInt i, nCNTL_pre = mumps->CNTL_pre ? mumps->CNTL_pre[0] : 0;
    *val = 0.0;
    for (i = 0; i < nCNTL_pre; ++i) {
      if (mumps->CNTL_pre[1 + 2 * i] == icntl) *val = mumps->CNTL_pre[2 + 2 * i];
    }
  } else *val = ID_CNTL_GET(mumps->id, icntl);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMumpsSetCntl - Set MUMPS parameter CNTL() <https://mumps-solver.org/index.php?page=doc>

  Logically Collective

  Input Parameters:
+ F     - the factored matrix obtained by calling `MatGetFactor()` with a `MatSolverType` of `MATSOLVERMUMPS` and a `MatFactorType` of `MAT_FACTOR_LU` or `MAT_FACTOR_CHOLESKY`
. icntl - index of MUMPS parameter array `CNTL()`
- val   - value of MUMPS `CNTL(icntl)`

  Options Database Key:
. -mat_mumps_cntl_<icntl> <val> - change the option numbered icntl to ival

  Level: beginner

  Note:
  Ignored if MUMPS is not installed or `F` is not a MUMPS matrix

.seealso: [](ch_matrices), `Mat`, `MatGetFactor()`, `MatMumpsSetIcntl()`, `MatMumpsGetIcntl()`, `MatMumpsGetCntl()`, `MatMumpsGetInfo()`, `MatMumpsGetInfog()`, `MatMumpsGetRinfo()`, `MatMumpsGetRinfog()`
@*/
PetscErrorCode MatMumpsSetCntl(Mat F, PetscInt icntl, PetscReal val)
{
  PetscFunctionBegin;
  PetscValidType(F, 1);
  PetscCheck(F->factortype, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONGSTATE, "Only for factored matrix");
  PetscValidLogicalCollectiveInt(F, icntl, 2);
  PetscValidLogicalCollectiveReal(F, val, 3);
  PetscCheck(icntl >= 1 && icntl <= 7, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONG, "Unsupported CNTL value %" PetscInt_FMT, icntl);
  PetscTryMethod(F, "MatMumpsSetCntl_C", (Mat, PetscInt, PetscReal), (F, icntl, val));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMumpsGetCntl - Get MUMPS parameter CNTL() <https://mumps-solver.org/index.php?page=doc>

  Logically Collective

  Input Parameters:
+ F     - the factored matrix obtained by calling `MatGetFactor()` with a `MatSolverType` of `MATSOLVERMUMPS` and a `MatFactorType` of `MAT_FACTOR_LU` or `MAT_FACTOR_CHOLESKY`
- icntl - index of MUMPS parameter array CNTL()

  Output Parameter:
. val - value of MUMPS CNTL(icntl)

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MatGetFactor()`, `MatMumpsSetIcntl()`, `MatMumpsGetIcntl()`, `MatMumpsSetCntl()`, `MatMumpsGetInfo()`, `MatMumpsGetInfog()`, `MatMumpsGetRinfo()`, `MatMumpsGetRinfog()`
@*/
PetscErrorCode MatMumpsGetCntl(Mat F, PetscInt icntl, PetscReal *val)
{
  PetscFunctionBegin;
  PetscValidType(F, 1);
  PetscCheck(F->factortype, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONGSTATE, "Only for factored matrix");
  PetscValidLogicalCollectiveInt(F, icntl, 2);
  PetscAssertPointer(val, 3);
  PetscCheck(icntl >= 1 && icntl <= 7, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONG, "Unsupported CNTL value %" PetscInt_FMT, icntl);
  PetscUseMethod(F, "MatMumpsGetCntl_C", (Mat, PetscInt, PetscReal *), (F, icntl, val));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMumpsGetInfo_MUMPS(Mat F, PetscInt icntl, PetscInt *info)
{
  Mat_MUMPS *mumps = (Mat_MUMPS *)F->data;

  PetscFunctionBegin;
  *info = mumps->id.INFO(icntl);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMumpsGetInfog_MUMPS(Mat F, PetscInt icntl, PetscInt *infog)
{
  Mat_MUMPS *mumps = (Mat_MUMPS *)F->data;

  PetscFunctionBegin;
  *infog = mumps->id.INFOG(icntl);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMumpsGetRinfo_MUMPS(Mat F, PetscInt icntl, PetscReal *rinfo)
{
  Mat_MUMPS *mumps = (Mat_MUMPS *)F->data;

  PetscFunctionBegin;
  *rinfo = ID_RINFO_GET(mumps->id, icntl);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMumpsGetRinfog_MUMPS(Mat F, PetscInt icntl, PetscReal *rinfog)
{
  Mat_MUMPS *mumps = (Mat_MUMPS *)F->data;

  PetscFunctionBegin;
  *rinfog = ID_RINFOG_GET(mumps->id, icntl);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMumpsGetNullPivots_MUMPS(Mat F, PetscInt *size, PetscInt **array)
{
  Mat_MUMPS *mumps = (Mat_MUMPS *)F->data;

  PetscFunctionBegin;
  PetscCheck(mumps->id.ICNTL(24) == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "-mat_mumps_icntl_24 must be set as 1 for null pivot row detection");
  *size  = 0;
  *array = NULL;
  if (!mumps->myid) {
    *size = mumps->id.INFOG(28);
    PetscCall(PetscMalloc1(*size, array));
    for (int i = 0; i < *size; i++) (*array)[i] = mumps->id.pivnul_list[i] - 1;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMumpsGetInverse_MUMPS(Mat F, Mat spRHS)
{
  Mat          Bt = NULL, Btseq = NULL;
  PetscBool    flg;
  Mat_MUMPS   *mumps = (Mat_MUMPS *)F->data;
  PetscScalar *aa;
  PetscInt     spnr, *ia, *ja, M, nrhs;

  PetscFunctionBegin;
  PetscAssertPointer(spRHS, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)spRHS, MATTRANSPOSEVIRTUAL, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)spRHS), PETSC_ERR_ARG_WRONG, "Matrix spRHS must be type MATTRANSPOSEVIRTUAL matrix");
  PetscCall(MatShellGetScalingShifts(spRHS, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
  PetscCall(MatTransposeGetMat(spRHS, &Bt));

  PetscCall(MatMumpsSetIcntl(F, 30, 1));

  if (mumps->petsc_size > 1) {
    Mat_MPIAIJ *b = (Mat_MPIAIJ *)Bt->data;
    Btseq         = b->A;
  } else {
    Btseq = Bt;
  }

  PetscCall(MatGetSize(spRHS, &M, &nrhs));
  mumps->id.nrhs = (PetscMUMPSInt)nrhs;
  PetscCall(PetscMUMPSIntCast(M, &mumps->id.lrhs));
  mumps->id.rhs = NULL;

  if (!mumps->myid) {
    PetscCall(MatSeqAIJGetArray(Btseq, &aa));
    PetscCall(MatGetRowIJ(Btseq, 1, PETSC_FALSE, PETSC_FALSE, &spnr, (const PetscInt **)&ia, (const PetscInt **)&ja, &flg));
    PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot get IJ structure");
    PetscCall(PetscMUMPSIntCSRCast(mumps, spnr, ia, ja, &mumps->id.irhs_ptr, &mumps->id.irhs_sparse, &mumps->id.nz_rhs));
    PetscCall(MatMumpsMakeMumpsScalarArray(PETSC_TRUE, ((Mat_SeqAIJ *)Btseq->data)->nz, aa, mumps->id.precision, &mumps->id.rhs_sparse_len, &mumps->id.rhs_sparse));
  } else {
    mumps->id.irhs_ptr    = NULL;
    mumps->id.irhs_sparse = NULL;
    mumps->id.nz_rhs      = 0;
    if (mumps->id.rhs_sparse_len) {
      PetscCall(PetscFree(mumps->id.rhs_sparse));
      mumps->id.rhs_sparse_len = 0;
    }
  }
  mumps->id.ICNTL(20) = 1; /* rhs is sparse */
  mumps->id.ICNTL(21) = 0; /* solution is in assembled centralized format */

  /* solve phase */
  mumps->id.job = JOB_SOLVE;
  PetscMUMPS_c(mumps);
  PetscCheck(mumps->id.INFOG(1) >= 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "MUMPS error in solve: INFOG(1)=%d INFO(2)=%d " MUMPS_MANUALS, mumps->id.INFOG(1), mumps->id.INFO(2));

  if (!mumps->myid) {
    PetscCall(MatSeqAIJRestoreArray(Btseq, &aa));
    PetscCall(MatRestoreRowIJ(Btseq, 1, PETSC_FALSE, PETSC_FALSE, &spnr, (const PetscInt **)&ia, (const PetscInt **)&ja, &flg));
    PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cannot get IJ structure");
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMumpsGetInverse - Get user-specified set of entries in inverse of `A` <https://mumps-solver.org/index.php?page=doc>

  Logically Collective

  Input Parameter:
. F - the factored matrix obtained by calling `MatGetFactor()` with a `MatSolverType` of `MATSOLVERMUMPS` and a `MatFactorType` of `MAT_FACTOR_LU` or `MAT_FACTOR_CHOLESKY`

  Output Parameter:
. spRHS - sequential sparse matrix in `MATTRANSPOSEVIRTUAL` format with requested entries of inverse of `A`

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MatGetFactor()`, `MatCreateTranspose()`
@*/
PetscErrorCode MatMumpsGetInverse(Mat F, Mat spRHS)
{
  PetscFunctionBegin;
  PetscValidType(F, 1);
  PetscCheck(F->factortype, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONGSTATE, "Only for factored matrix");
  PetscUseMethod(F, "MatMumpsGetInverse_C", (Mat, Mat), (F, spRHS));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMumpsGetInverseTranspose_MUMPS(Mat F, Mat spRHST)
{
  Mat spRHS;

  PetscFunctionBegin;
  PetscCall(MatCreateTranspose(spRHST, &spRHS));
  PetscCall(MatMumpsGetInverse_MUMPS(F, spRHS));
  PetscCall(MatDestroy(&spRHS));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMumpsGetInverseTranspose - Get user-specified set of entries in inverse of matrix $A^T $ <https://mumps-solver.org/index.php?page=doc>

  Logically Collective

  Input Parameter:
. F - the factored matrix of A obtained by calling `MatGetFactor()` with a `MatSolverType` of `MATSOLVERMUMPS` and a `MatFactorType` of `MAT_FACTOR_LU` or `MAT_FACTOR_CHOLESKY`

  Output Parameter:
. spRHST - sequential sparse matrix in `MATAIJ` format containing the requested entries of inverse of `A`^T

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MatGetFactor()`, `MatCreateTranspose()`, `MatMumpsGetInverse()`
@*/
PetscErrorCode MatMumpsGetInverseTranspose(Mat F, Mat spRHST)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidType(F, 1);
  PetscCheck(F->factortype, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONGSTATE, "Only for factored matrix");
  PetscCall(PetscObjectTypeCompareAny((PetscObject)spRHST, &flg, MATSEQAIJ, MATMPIAIJ, NULL));
  PetscCheck(flg, PetscObjectComm((PetscObject)spRHST), PETSC_ERR_ARG_WRONG, "Matrix spRHST must be MATAIJ matrix");
  PetscUseMethod(F, "MatMumpsGetInverseTranspose_C", (Mat, Mat), (F, spRHST));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMumpsSetBlk_MUMPS(Mat F, PetscInt nblk, const PetscInt blkvar[], const PetscInt blkptr[])
{
  Mat_MUMPS *mumps = (Mat_MUMPS *)F->data;

  PetscFunctionBegin;
  if (nblk) {
    PetscAssertPointer(blkptr, 4);
    PetscCall(PetscMUMPSIntCast(nblk, &mumps->id.nblk));
    PetscCall(PetscFree(mumps->id.blkptr));
    PetscCall(PetscMalloc1(nblk + 1, &mumps->id.blkptr));
    for (PetscInt i = 0; i < nblk + 1; ++i) PetscCall(PetscMUMPSIntCast(blkptr[i], mumps->id.blkptr + i));
    // mumps->id.icntl[] might have not been allocated, which is done in MatSetFromOptions_MUMPS(). So we don't assign ICNTL(15).
    // We use id.nblk and id.blkptr to know what values to set to ICNTL(15) in MatSetFromOptions_MUMPS().
    // mumps->id.ICNTL(15) = 1;
    if (blkvar) {
      PetscCall(PetscFree(mumps->id.blkvar));
      PetscCall(PetscMalloc1(F->rmap->N, &mumps->id.blkvar));
      for (PetscInt i = 0; i < F->rmap->N; ++i) PetscCall(PetscMUMPSIntCast(blkvar[i], mumps->id.blkvar + i));
    }
  } else {
    PetscCall(PetscFree(mumps->id.blkptr));
    PetscCall(PetscFree(mumps->id.blkvar));
    // mumps->id.ICNTL(15) = 0;
    mumps->id.nblk = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMumpsSetBlk - Set user-specified variable block sizes to be used with `-mat_mumps_icntl_15 1`

  Not collective, only relevant on the first process of the MPI communicator

  Input Parameters:
+ F      - the factored matrix of A obtained by calling `MatGetFactor()` with a `MatSolverType` of `MATSOLVERMUMPS` and a `MatFactorType` of `MAT_FACTOR_LU` or `MAT_FACTOR_CHOLESKY`
. nblk   - the number of blocks
. blkvar - see MUMPS documentation, `blkvar(blkptr(iblk):blkptr(iblk+1)-1)`, (`iblk=1, nblk`) holds the variables associated to block `iblk`
- blkptr - array starting at 1 and of size `nblk + 1` storing the prefix sum of all blocks

  Level: advanced

.seealso: [](ch_matrices), `MATSOLVERMUMPS`, `Mat`, `MatGetFactor()`, `MatMumpsSetIcntl()`, `MatSetVariableBlockSizes()`
@*/
PetscErrorCode MatMumpsSetBlk(Mat F, PetscInt nblk, const PetscInt blkvar[], const PetscInt blkptr[])
{
  PetscFunctionBegin;
  PetscValidType(F, 1);
  PetscCheck(F->factortype, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONGSTATE, "Only for factored matrix");
  PetscUseMethod(F, "MatMumpsSetBlk_C", (Mat, PetscInt, const PetscInt[], const PetscInt[]), (F, nblk, blkvar, blkptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMumpsGetInfo - Get MUMPS parameter INFO() <https://mumps-solver.org/index.php?page=doc>

  Logically Collective

  Input Parameters:
+ F     - the factored matrix obtained by calling `MatGetFactor()` with a `MatSolverType` of `MATSOLVERMUMPS` and a `MatFactorType` of `MAT_FACTOR_LU` or `MAT_FACTOR_CHOLESKY`
- icntl - index of MUMPS parameter array INFO()

  Output Parameter:
. ival - value of MUMPS INFO(icntl)

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MatGetFactor()`, `MatMumpsSetIcntl()`, `MatMumpsGetIcntl()`, `MatMumpsSetCntl()`, `MatMumpsGetCntl()`, `MatMumpsGetInfog()`, `MatMumpsGetRinfo()`, `MatMumpsGetRinfog()`
@*/
PetscErrorCode MatMumpsGetInfo(Mat F, PetscInt icntl, PetscInt *ival)
{
  PetscFunctionBegin;
  PetscValidType(F, 1);
  PetscCheck(F->factortype, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONGSTATE, "Only for factored matrix");
  PetscAssertPointer(ival, 3);
  PetscUseMethod(F, "MatMumpsGetInfo_C", (Mat, PetscInt, PetscInt *), (F, icntl, ival));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMumpsGetInfog - Get MUMPS parameter INFOG() <https://mumps-solver.org/index.php?page=doc>

  Logically Collective

  Input Parameters:
+ F     - the factored matrix obtained by calling `MatGetFactor()` with a `MatSolverType` of `MATSOLVERMUMPS` and a `MatFactorType` of `MAT_FACTOR_LU` or `MAT_FACTOR_CHOLESKY`
- icntl - index of MUMPS parameter array INFOG()

  Output Parameter:
. ival - value of MUMPS INFOG(icntl)

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MatGetFactor()`, `MatMumpsSetIcntl()`, `MatMumpsGetIcntl()`, `MatMumpsSetCntl()`, `MatMumpsGetCntl()`, `MatMumpsGetInfo()`, `MatMumpsGetRinfo()`, `MatMumpsGetRinfog()`
@*/
PetscErrorCode MatMumpsGetInfog(Mat F, PetscInt icntl, PetscInt *ival)
{
  PetscFunctionBegin;
  PetscValidType(F, 1);
  PetscCheck(F->factortype, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONGSTATE, "Only for factored matrix");
  PetscAssertPointer(ival, 3);
  PetscUseMethod(F, "MatMumpsGetInfog_C", (Mat, PetscInt, PetscInt *), (F, icntl, ival));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMumpsGetRinfo - Get MUMPS parameter RINFO() <https://mumps-solver.org/index.php?page=doc>

  Logically Collective

  Input Parameters:
+ F     - the factored matrix obtained by calling `MatGetFactor()` with a `MatSolverType` of `MATSOLVERMUMPS` and a `MatFactorType` of `MAT_FACTOR_LU` or `MAT_FACTOR_CHOLESKY`
- icntl - index of MUMPS parameter array RINFO()

  Output Parameter:
. val - value of MUMPS RINFO(icntl)

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MatGetFactor()`, `MatMumpsSetIcntl()`, `MatMumpsGetIcntl()`, `MatMumpsSetCntl()`, `MatMumpsGetCntl()`, `MatMumpsGetInfo()`, `MatMumpsGetInfog()`, `MatMumpsGetRinfog()`
@*/
PetscErrorCode MatMumpsGetRinfo(Mat F, PetscInt icntl, PetscReal *val)
{
  PetscFunctionBegin;
  PetscValidType(F, 1);
  PetscCheck(F->factortype, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONGSTATE, "Only for factored matrix");
  PetscAssertPointer(val, 3);
  PetscUseMethod(F, "MatMumpsGetRinfo_C", (Mat, PetscInt, PetscReal *), (F, icntl, val));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMumpsGetRinfog - Get MUMPS parameter RINFOG() <https://mumps-solver.org/index.php?page=doc>

  Logically Collective

  Input Parameters:
+ F     - the factored matrix obtained by calling `MatGetFactor()` with a `MatSolverType` of `MATSOLVERMUMPS` and a `MatFactorType` of `MAT_FACTOR_LU` or `MAT_FACTOR_CHOLESKY`
- icntl - index of MUMPS parameter array RINFOG()

  Output Parameter:
. val - value of MUMPS RINFOG(icntl)

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MatGetFactor()`, `MatMumpsSetIcntl()`, `MatMumpsGetIcntl()`, `MatMumpsSetCntl()`, `MatMumpsGetCntl()`, `MatMumpsGetInfo()`, `MatMumpsGetInfog()`, `MatMumpsGetRinfo()`
@*/
PetscErrorCode MatMumpsGetRinfog(Mat F, PetscInt icntl, PetscReal *val)
{
  PetscFunctionBegin;
  PetscValidType(F, 1);
  PetscCheck(F->factortype, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONGSTATE, "Only for factored matrix");
  PetscAssertPointer(val, 3);
  PetscUseMethod(F, "MatMumpsGetRinfog_C", (Mat, PetscInt, PetscReal *), (F, icntl, val));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatMumpsGetNullPivots - Get MUMPS parameter PIVNUL_LIST() <https://mumps-solver.org/index.php?page=doc>

  Logically Collective

  Input Parameter:
. F - the factored matrix obtained by calling `MatGetFactor()` with a `MatSolverType` of `MATSOLVERMUMPS` and a `MatFactorType` of `MAT_FACTOR_LU` or `MAT_FACTOR_CHOLESKY`

  Output Parameters:
+ size  - local size of the array. The size of the array is non-zero only on MPI rank 0
- array - array of rows with null pivot, these rows follow 0-based indexing. The array gets allocated within the function and the user is responsible
          for freeing this array.

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MatGetFactor()`, `MatMumpsSetIcntl()`, `MatMumpsGetIcntl()`, `MatMumpsSetCntl()`, `MatMumpsGetCntl()`, `MatMumpsGetInfo()`, `MatMumpsGetInfog()`, `MatMumpsGetRinfo()`
@*/
PetscErrorCode MatMumpsGetNullPivots(Mat F, PetscInt *size, PetscInt **array)
{
  PetscFunctionBegin;
  PetscValidType(F, 1);
  PetscCheck(F->factortype, PetscObjectComm((PetscObject)F), PETSC_ERR_ARG_WRONGSTATE, "Only for factored matrix");
  PetscAssertPointer(size, 2);
  PetscAssertPointer(array, 3);
  PetscUseMethod(F, "MatMumpsGetNullPivots_C", (Mat, PetscInt *, PetscInt **), (F, size, array));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
  MATSOLVERMUMPS -  A matrix type providing direct solvers (LU and Cholesky) for
  MPI distributed and sequential matrices via the external package MUMPS <https://mumps-solver.org/index.php?page=doc>

  Works with `MATAIJ` and `MATSBAIJ` matrices

  Use ./configure --download-mumps --download-scalapack --download-parmetis --download-metis --download-ptscotch to have PETSc installed with MUMPS

  Use ./configure --with-openmp --download-hwloc (or --with-hwloc) to enable running MUMPS in MPI+OpenMP hybrid mode and non-MUMPS in flat-MPI mode.
  See details below.

  Use `-pc_type cholesky` or `lu` `-pc_factor_mat_solver_type mumps` to use this direct solver

  Options Database Keys:
+  -mat_mumps_icntl_1  - ICNTL(1): output stream for error messages
.  -mat_mumps_icntl_2  - ICNTL(2): output stream for diagnostic printing, statistics, and warning
.  -mat_mumps_icntl_3  - ICNTL(3): output stream for global information, collected on the host
.  -mat_mumps_icntl_4  - ICNTL(4): level of printing (0 to 4)
.  -mat_mumps_icntl_6  - ICNTL(6): permutes to a zero-free diagonal and/or scale the matrix (0 to 7)
.  -mat_mumps_icntl_7  - ICNTL(7): computes a symmetric permutation in sequential analysis, 0=AMD, 2=AMF, 3=Scotch, 4=PORD, 5=Metis, 6=QAMD, and 7=auto
                          Use -pc_factor_mat_ordering_type <type> to have PETSc perform the ordering (sequential only)
.  -mat_mumps_icntl_8  - ICNTL(8): scaling strategy (-2 to 8 or 77)
.  -mat_mumps_icntl_10 - ICNTL(10): max num of refinements
.  -mat_mumps_icntl_11 - ICNTL(11): statistics related to an error analysis (via -ksp_view)
.  -mat_mumps_icntl_12 - ICNTL(12): an ordering strategy for symmetric matrices (0 to 3)
.  -mat_mumps_icntl_13 - ICNTL(13): parallelism of the root node (enable ScaLAPACK) and its splitting
.  -mat_mumps_icntl_14 - ICNTL(14): percentage increase in the estimated working space
.  -mat_mumps_icntl_15 - ICNTL(15): compression of the input matrix resulting from a block format
.  -mat_mumps_icntl_19 - ICNTL(19): computes the Schur complement
.  -mat_mumps_icntl_20 - ICNTL(20): give MUMPS centralized (0) or distributed (10) dense RHS
.  -mat_mumps_icntl_22 - ICNTL(22): in-core/out-of-core factorization and solve (0 or 1)
.  -mat_mumps_icntl_23 - ICNTL(23): max size of the working memory (MB) that can allocate per processor
.  -mat_mumps_icntl_24 - ICNTL(24): detection of null pivot rows (0 or 1)
.  -mat_mumps_icntl_25 - ICNTL(25): compute a solution of a deficient matrix and a null space basis
.  -mat_mumps_icntl_26 - ICNTL(26): drives the solution phase if a Schur complement matrix
.  -mat_mumps_icntl_28 - ICNTL(28): use 1 for sequential analysis and ICNTL(7) ordering, or 2 for parallel analysis and ICNTL(29) ordering
.  -mat_mumps_icntl_29 - ICNTL(29): parallel ordering 1 = ptscotch, 2 = parmetis
.  -mat_mumps_icntl_30 - ICNTL(30): compute user-specified set of entries in inv(A)
.  -mat_mumps_icntl_31 - ICNTL(31): indicates which factors may be discarded during factorization
.  -mat_mumps_icntl_33 - ICNTL(33): compute determinant
.  -mat_mumps_icntl_35 - ICNTL(35): level of activation of BLR (Block Low-Rank) feature
.  -mat_mumps_icntl_36 - ICNTL(36): controls the choice of BLR factorization variant
.  -mat_mumps_icntl_37 - ICNTL(37): compression of the contribution blocks (CB)
.  -mat_mumps_icntl_38 - ICNTL(38): sets the estimated compression rate of LU factors with BLR
.  -mat_mumps_icntl_48 - ICNTL(48): multithreading with tree parallelism
.  -mat_mumps_icntl_49 - ICNTL(49): compact workarray at the end of factorization phase
.  -mat_mumps_icntl_58 - ICNTL(58): options for symbolic factorization
.  -mat_mumps_cntl_1   - CNTL(1): relative pivoting threshold
.  -mat_mumps_cntl_2   - CNTL(2): stopping criterion of refinement
.  -mat_mumps_cntl_3   - CNTL(3): absolute pivoting threshold
.  -mat_mumps_cntl_4   - CNTL(4): value for static pivoting
.  -mat_mumps_cntl_5   - CNTL(5): fixation for null pivots
.  -mat_mumps_cntl_7   - CNTL(7): precision of the dropping parameter used during BLR factorization
-  -mat_mumps_use_omp_threads [m] - run MUMPS in MPI+OpenMP hybrid mode as if omp_set_num_threads(m) is called before calling MUMPS.
                                    Default might be the number of cores per CPU package (socket) as reported by hwloc and suggested by the MUMPS manual.

  Level: beginner

  Notes:
  MUMPS Cholesky does not handle (complex) Hermitian matrices (see User's Guide at <https://mumps-solver.org/index.php?page=doc>) so using it will
  error if the matrix is Hermitian.

  When used within a `KSP`/`PC` solve the options are prefixed with that of the `PC`. Otherwise one can set the options prefix by calling
  `MatSetOptionsPrefixFactor()` on the matrix from which the factor was obtained or `MatSetOptionsPrefix()` on the factor matrix.

  When a MUMPS factorization fails inside a KSP solve, for example with a `KSP_DIVERGED_PC_FAILED`, one can find the MUMPS information about
  the failure with
.vb
          KSPGetPC(ksp,&pc);
          PCFactorGetMatrix(pc,&mat);
          MatMumpsGetInfo(mat,....);
          MatMumpsGetInfog(mat,....); etc.
.ve
  Or run with `-ksp_error_if_not_converged` and the program will be stopped and the information printed in the error message.

  MUMPS provides 64-bit integer support in two build modes:
  full 64-bit: here MUMPS is built with C preprocessing flag -DINTSIZE64 and Fortran compiler option -i8, -fdefault-integer-8 or equivalent, and
  requires all dependent libraries MPI, ScaLAPACK, LAPACK and BLAS built the same way with 64-bit integers (for example ILP64 Intel MKL and MPI).

  selective 64-bit: with the default MUMPS build, 64-bit integers have been introduced where needed. In compressed sparse row (CSR) storage of matrices,
  MUMPS stores column indices in 32-bit, but row offsets in 64-bit, so you can have a huge number of non-zeros, but must have less than 2^31 rows and
  columns. This can lead to significant memory and performance gains with respect to a full 64-bit integer MUMPS version. This requires a regular (32-bit
  integer) build of all dependent libraries MPI, ScaLAPACK, LAPACK and BLAS.

  With --download-mumps=1, PETSc always build MUMPS in selective 64-bit mode, which can be used by both --with-64-bit-indices=0/1 variants of PETSc.

  Two modes to run MUMPS/PETSc with OpenMP
.vb
   Set `OMP_NUM_THREADS` and run with fewer MPI ranks than cores. For example, if you want to have 16 OpenMP
   threads per rank, then you may use "export `OMP_NUM_THREADS` = 16 && mpirun -n 4 ./test".
.ve

.vb
   `-mat_mumps_use_omp_threads` [m] and run your code with as many MPI ranks as the number of cores. For example,
   if a compute node has 32 cores and you run on two nodes, you may use "mpirun -n 64 ./test -mat_mumps_use_omp_threads 16"
.ve

   To run MUMPS in MPI+OpenMP hybrid mode (i.e., enable multithreading in MUMPS), but still run the non-MUMPS part
   (i.e., PETSc part) of your code in the so-called flat-MPI (aka pure-MPI) mode, you need to configure PETSc with `--with-openmp` `--download-hwloc`
   (or `--with-hwloc`), and have an MPI that supports MPI-3.0's process shared memory (which is usually available). Since MUMPS calls BLAS
   libraries, to really get performance, you should have multithreaded BLAS libraries such as Intel MKL, AMD ACML, Cray libSci or OpenBLAS
   (PETSc will automatically try to utilized a threaded BLAS if `--with-openmp` is provided).

   If you run your code through a job submission system, there are caveats in MPI rank mapping. We use MPI_Comm_split_type() to obtain MPI
   processes on each compute node. Listing the processes in rank ascending order, we split processes on a node into consecutive groups of
   size m and create a communicator called omp_comm for each group. Rank 0 in an omp_comm is called the master rank, and others in the omp_comm
   are called slave ranks (or slaves). Only master ranks are seen to MUMPS and slaves are not. We will free CPUs assigned to slaves (might be set
   by CPU binding policies in job scripts) and make the CPUs available to the master so that OMP threads spawned by MUMPS can run on the CPUs.
   In a multi-socket compute node, MPI rank mapping is an issue. Still use the above example and suppose your compute node has two sockets,
   if you interleave MPI ranks on the two sockets, in other words, even ranks are placed on socket 0, and odd ranks are on socket 1, and bind
   MPI ranks to cores, then with `-mat_mumps_use_omp_threads` 16, a master rank (and threads it spawns) will use half cores in socket 0, and half
   cores in socket 1, that definitely hurts locality. On the other hand, if you map MPI ranks consecutively on the two sockets, then the
   problem will not happen. Therefore, when you use `-mat_mumps_use_omp_threads`, you need to keep an eye on your MPI rank mapping and CPU binding.
   For example, with the Slurm job scheduler, one can use srun `--cpu-bind`=verbose -m block:block to map consecutive MPI ranks to sockets and
   examine the mapping result.

   PETSc does not control thread binding in MUMPS. So to get best performance, one still has to set `OMP_PROC_BIND` and `OMP_PLACES` in job scripts,
   for example, export `OMP_PLACES`=threads and export `OMP_PROC_BIND`=spread. One does not need to export `OMP_NUM_THREADS`=m in job scripts as PETSc
   calls `omp_set_num_threads`(m) internally before calling MUMPS.

   See {cite}`heroux2011bi` and {cite}`gutierrez2017accommodating`

.seealso: [](ch_matrices), `Mat`, `PCFactorSetMatSolverType()`, `MatSolverType`, `MatMumpsSetIcntl()`, `MatMumpsGetIcntl()`, `MatMumpsSetCntl()`, `MatMumpsGetCntl()`, `MatMumpsGetInfo()`, `MatMumpsGetInfog()`, `MatMumpsGetRinfo()`, `MatMumpsGetRinfog()`, `MatMumpsSetBlk()`, `KSPGetPC()`, `PCFactorGetMatrix()`
M*/

static PetscErrorCode MatFactorGetSolverType_mumps(PETSC_UNUSED Mat A, MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERMUMPS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* MatGetFactor for Seq and MPI AIJ matrices */
static PetscErrorCode MatGetFactor_aij_mumps(Mat A, MatFactorType ftype, Mat *F)
{
  Mat         B;
  Mat_MUMPS  *mumps;
  PetscBool   isSeqAIJ, isDiag, isDense;
  PetscMPIInt size;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  if (ftype == MAT_FACTOR_CHOLESKY && A->hermitian == PETSC_BOOL3_TRUE && A->symmetric != PETSC_BOOL3_TRUE) {
    PetscCall(PetscInfo(A, "Hermitian MAT_FACTOR_CHOLESKY is not supported. Use MAT_FACTOR_LU instead.\n"));
    *F = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
#endif
  /* Create the factorization matrix */
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A, MATSEQAIJ, &isSeqAIJ));
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)A, MATDIAGONAL, &isDiag));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &isDense, MATSEQDENSE, MATMPIDENSE, NULL));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
  PetscCall(MatSetSizes(B, A->rmap->n, A->cmap->n, A->rmap->N, A->cmap->N));
  PetscCall(PetscStrallocpy(MATSOLVERMUMPS, &((PetscObject)B)->type_name));
  PetscCall(MatSetUp(B));

  PetscCall(PetscNew(&mumps));

  B->ops->view    = MatView_MUMPS;
  B->ops->getinfo = MatGetInfo_MUMPS;

  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatFactorGetSolverType_C", MatFactorGetSolverType_mumps));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatFactorSetSchurIS_C", MatFactorSetSchurIS_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatFactorCreateSchurComplement_C", MatFactorCreateSchurComplement_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsSetIcntl_C", MatMumpsSetIcntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetIcntl_C", MatMumpsGetIcntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsSetCntl_C", MatMumpsSetCntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetCntl_C", MatMumpsGetCntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetInfo_C", MatMumpsGetInfo_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetInfog_C", MatMumpsGetInfog_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetRinfo_C", MatMumpsGetRinfo_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetRinfog_C", MatMumpsGetRinfog_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetNullPivots_C", MatMumpsGetNullPivots_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetInverse_C", MatMumpsGetInverse_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetInverseTranspose_C", MatMumpsGetInverseTranspose_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsSetBlk_C", MatMumpsSetBlk_MUMPS));

  if (ftype == MAT_FACTOR_LU) {
    B->ops->lufactorsymbolic = MatLUFactorSymbolic_AIJMUMPS;
    B->factortype            = MAT_FACTOR_LU;
    if (isSeqAIJ) mumps->ConvertToTriples = MatConvertToTriples_seqaij_seqaij;
    else if (isDiag) mumps->ConvertToTriples = MatConvertToTriples_diagonal_xaij;
    else if (isDense) mumps->ConvertToTriples = MatConvertToTriples_dense_xaij;
    else mumps->ConvertToTriples = MatConvertToTriples_mpiaij_mpiaij;
    PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL, (char **)&B->preferredordering[MAT_FACTOR_LU]));
    mumps->sym = 0;
  } else {
    B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_MUMPS;
    B->factortype                  = MAT_FACTOR_CHOLESKY;
    if (isSeqAIJ) mumps->ConvertToTriples = MatConvertToTriples_seqaij_seqsbaij;
    else if (isDiag) mumps->ConvertToTriples = MatConvertToTriples_diagonal_xaij;
    else if (isDense) mumps->ConvertToTriples = MatConvertToTriples_dense_xaij;
    else mumps->ConvertToTriples = MatConvertToTriples_mpiaij_mpisbaij;
    PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL, (char **)&B->preferredordering[MAT_FACTOR_CHOLESKY]));
#if defined(PETSC_USE_COMPLEX)
    mumps->sym = 2;
#else
    if (A->spd == PETSC_BOOL3_TRUE) mumps->sym = 1;
    else mumps->sym = 2;
#endif
  }

  /* set solvertype */
  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERMUMPS, &B->solvertype));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A), &size));
  if (size == 1) {
    /* MUMPS option -mat_mumps_icntl_7 1 is automatically set if PETSc ordering is passed into symbolic factorization */
    B->canuseordering = PETSC_TRUE;
  }
  B->ops->destroy = MatDestroy_MUMPS;
  B->data         = (void *)mumps;

  *F               = B;
  mumps->id.job    = JOB_NULL;
  mumps->ICNTL_pre = NULL;
  mumps->CNTL_pre  = NULL;
  mumps->matstruc  = DIFFERENT_NONZERO_PATTERN;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* MatGetFactor for Seq and MPI SBAIJ matrices */
static PetscErrorCode MatGetFactor_sbaij_mumps(Mat A, PETSC_UNUSED MatFactorType ftype, Mat *F)
{
  Mat         B;
  Mat_MUMPS  *mumps;
  PetscBool   isSeqSBAIJ;
  PetscMPIInt size;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  if (ftype == MAT_FACTOR_CHOLESKY && A->hermitian == PETSC_BOOL3_TRUE && A->symmetric != PETSC_BOOL3_TRUE) {
    PetscCall(PetscInfo(A, "Hermitian MAT_FACTOR_CHOLESKY is not supported. Use MAT_FACTOR_LU instead.\n"));
    *F = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
#endif
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
  PetscCall(MatSetSizes(B, A->rmap->n, A->cmap->n, A->rmap->N, A->cmap->N));
  PetscCall(PetscStrallocpy(MATSOLVERMUMPS, &((PetscObject)B)->type_name));
  PetscCall(MatSetUp(B));

  PetscCall(PetscNew(&mumps));
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQSBAIJ, &isSeqSBAIJ));
  if (isSeqSBAIJ) {
    mumps->ConvertToTriples = MatConvertToTriples_seqsbaij_seqsbaij;
  } else {
    mumps->ConvertToTriples = MatConvertToTriples_mpisbaij_mpisbaij;
  }

  B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_MUMPS;
  B->ops->view                   = MatView_MUMPS;
  B->ops->getinfo                = MatGetInfo_MUMPS;

  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatFactorGetSolverType_C", MatFactorGetSolverType_mumps));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatFactorSetSchurIS_C", MatFactorSetSchurIS_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatFactorCreateSchurComplement_C", MatFactorCreateSchurComplement_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsSetIcntl_C", MatMumpsSetIcntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetIcntl_C", MatMumpsGetIcntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsSetCntl_C", MatMumpsSetCntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetCntl_C", MatMumpsGetCntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetInfo_C", MatMumpsGetInfo_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetInfog_C", MatMumpsGetInfog_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetRinfo_C", MatMumpsGetRinfo_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetRinfog_C", MatMumpsGetRinfog_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetNullPivots_C", MatMumpsGetNullPivots_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetInverse_C", MatMumpsGetInverse_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetInverseTranspose_C", MatMumpsGetInverseTranspose_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsSetBlk_C", MatMumpsSetBlk_MUMPS));

  B->factortype = MAT_FACTOR_CHOLESKY;
#if defined(PETSC_USE_COMPLEX)
  mumps->sym = 2;
#else
  if (A->spd == PETSC_BOOL3_TRUE) mumps->sym = 1;
  else mumps->sym = 2;
#endif

  /* set solvertype */
  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERMUMPS, &B->solvertype));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A), &size));
  if (size == 1) {
    /* MUMPS option -mat_mumps_icntl_7 1 is automatically set if PETSc ordering is passed into symbolic factorization */
    B->canuseordering = PETSC_TRUE;
  }
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL, (char **)&B->preferredordering[MAT_FACTOR_CHOLESKY]));
  B->ops->destroy = MatDestroy_MUMPS;
  B->data         = (void *)mumps;

  *F               = B;
  mumps->id.job    = JOB_NULL;
  mumps->ICNTL_pre = NULL;
  mumps->CNTL_pre  = NULL;
  mumps->matstruc  = DIFFERENT_NONZERO_PATTERN;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetFactor_baij_mumps(Mat A, MatFactorType ftype, Mat *F)
{
  Mat         B;
  Mat_MUMPS  *mumps;
  PetscBool   isSeqBAIJ;
  PetscMPIInt size;

  PetscFunctionBegin;
  /* Create the factorization matrix */
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQBAIJ, &isSeqBAIJ));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
  PetscCall(MatSetSizes(B, A->rmap->n, A->cmap->n, A->rmap->N, A->cmap->N));
  PetscCall(PetscStrallocpy(MATSOLVERMUMPS, &((PetscObject)B)->type_name));
  PetscCall(MatSetUp(B));

  PetscCall(PetscNew(&mumps));
  PetscCheck(ftype == MAT_FACTOR_LU, PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot use PETSc BAIJ matrices with MUMPS Cholesky, use SBAIJ or AIJ matrix instead");
  B->ops->lufactorsymbolic = MatLUFactorSymbolic_BAIJMUMPS;
  B->factortype            = MAT_FACTOR_LU;
  if (isSeqBAIJ) mumps->ConvertToTriples = MatConvertToTriples_seqbaij_seqaij;
  else mumps->ConvertToTriples = MatConvertToTriples_mpibaij_mpiaij;
  mumps->sym = 0;
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL, (char **)&B->preferredordering[MAT_FACTOR_LU]));

  B->ops->view    = MatView_MUMPS;
  B->ops->getinfo = MatGetInfo_MUMPS;

  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatFactorGetSolverType_C", MatFactorGetSolverType_mumps));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatFactorSetSchurIS_C", MatFactorSetSchurIS_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatFactorCreateSchurComplement_C", MatFactorCreateSchurComplement_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsSetIcntl_C", MatMumpsSetIcntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetIcntl_C", MatMumpsGetIcntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsSetCntl_C", MatMumpsSetCntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetCntl_C", MatMumpsGetCntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetInfo_C", MatMumpsGetInfo_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetInfog_C", MatMumpsGetInfog_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetRinfo_C", MatMumpsGetRinfo_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetRinfog_C", MatMumpsGetRinfog_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetNullPivots_C", MatMumpsGetNullPivots_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetInverse_C", MatMumpsGetInverse_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetInverseTranspose_C", MatMumpsGetInverseTranspose_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsSetBlk_C", MatMumpsSetBlk_MUMPS));

  /* set solvertype */
  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERMUMPS, &B->solvertype));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A), &size));
  if (size == 1) {
    /* MUMPS option -mat_mumps_icntl_7 1 is automatically set if PETSc ordering is passed into symbolic factorization */
    B->canuseordering = PETSC_TRUE;
  }
  B->ops->destroy = MatDestroy_MUMPS;
  B->data         = (void *)mumps;

  *F               = B;
  mumps->id.job    = JOB_NULL;
  mumps->ICNTL_pre = NULL;
  mumps->CNTL_pre  = NULL;
  mumps->matstruc  = DIFFERENT_NONZERO_PATTERN;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* MatGetFactor for Seq and MPI SELL matrices */
static PetscErrorCode MatGetFactor_sell_mumps(Mat A, MatFactorType ftype, Mat *F)
{
  Mat         B;
  Mat_MUMPS  *mumps;
  PetscBool   isSeqSELL;
  PetscMPIInt size;

  PetscFunctionBegin;
  /* Create the factorization matrix */
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATSEQSELL, &isSeqSELL));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
  PetscCall(MatSetSizes(B, A->rmap->n, A->cmap->n, A->rmap->N, A->cmap->N));
  PetscCall(PetscStrallocpy(MATSOLVERMUMPS, &((PetscObject)B)->type_name));
  PetscCall(MatSetUp(B));

  PetscCall(PetscNew(&mumps));

  B->ops->view    = MatView_MUMPS;
  B->ops->getinfo = MatGetInfo_MUMPS;

  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatFactorGetSolverType_C", MatFactorGetSolverType_mumps));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatFactorSetSchurIS_C", MatFactorSetSchurIS_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatFactorCreateSchurComplement_C", MatFactorCreateSchurComplement_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsSetIcntl_C", MatMumpsSetIcntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetIcntl_C", MatMumpsGetIcntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsSetCntl_C", MatMumpsSetCntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetCntl_C", MatMumpsGetCntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetInfo_C", MatMumpsGetInfo_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetInfog_C", MatMumpsGetInfog_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetRinfo_C", MatMumpsGetRinfo_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetRinfog_C", MatMumpsGetRinfog_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetNullPivots_C", MatMumpsGetNullPivots_MUMPS));

  PetscCheck(ftype == MAT_FACTOR_LU, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "To be implemented");
  B->ops->lufactorsymbolic = MatLUFactorSymbolic_AIJMUMPS;
  B->factortype            = MAT_FACTOR_LU;
  PetscCheck(isSeqSELL, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "To be implemented");
  mumps->ConvertToTriples = MatConvertToTriples_seqsell_seqaij;
  mumps->sym              = 0;
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL, (char **)&B->preferredordering[MAT_FACTOR_LU]));

  /* set solvertype */
  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERMUMPS, &B->solvertype));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A), &size));
  if (size == 1) {
    /* MUMPS option -mat_mumps_icntl_7 1 is automatically set if PETSc ordering is passed into symbolic factorization  */
    B->canuseordering = PETSC_TRUE;
  }
  B->ops->destroy = MatDestroy_MUMPS;
  B->data         = (void *)mumps;

  *F               = B;
  mumps->id.job    = JOB_NULL;
  mumps->ICNTL_pre = NULL;
  mumps->CNTL_pre  = NULL;
  mumps->matstruc  = DIFFERENT_NONZERO_PATTERN;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* MatGetFactor for MATNEST matrices */
static PetscErrorCode MatGetFactor_nest_mumps(Mat A, MatFactorType ftype, Mat *F)
{
  Mat         B, **mats;
  Mat_MUMPS  *mumps;
  PetscInt    nr, nc;
  PetscMPIInt size;
  PetscBool   flg = PETSC_TRUE;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  if (ftype == MAT_FACTOR_CHOLESKY && A->hermitian == PETSC_BOOL3_TRUE && A->symmetric != PETSC_BOOL3_TRUE) {
    PetscCall(PetscInfo(A, "Hermitian MAT_FACTOR_CHOLESKY is not supported. Use MAT_FACTOR_LU instead.\n"));
    *F = NULL;
    PetscFunctionReturn(PETSC_SUCCESS);
  }
#endif

  /* Return if some condition is not satisfied */
  *F = NULL;
  PetscCall(MatNestGetSubMats(A, &nr, &nc, &mats));
  if (ftype == MAT_FACTOR_CHOLESKY) {
    IS       *rows, *cols;
    PetscInt *m, *M;

    PetscCheck(nr == nc, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "MAT_FACTOR_CHOLESKY not supported for nest sizes %" PetscInt_FMT " != %" PetscInt_FMT ". Use MAT_FACTOR_LU.", nr, nc);
    PetscCall(PetscMalloc2(nr, &rows, nc, &cols));
    PetscCall(MatNestGetISs(A, rows, cols));
    for (PetscInt r = 0; flg && r < nr; r++) PetscCall(ISEqualUnsorted(rows[r], cols[r], &flg));
    if (!flg) {
      PetscCall(PetscFree2(rows, cols));
      PetscCall(PetscInfo(A, "MAT_FACTOR_CHOLESKY not supported for unequal row and column maps. Use MAT_FACTOR_LU.\n"));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
    PetscCall(PetscMalloc2(nr, &m, nr, &M));
    for (PetscInt r = 0; r < nr; r++) PetscCall(ISGetMinMax(rows[r], &m[r], &M[r]));
    for (PetscInt r = 0; flg && r < nr; r++)
      for (PetscInt k = r + 1; flg && k < nr; k++)
        if ((m[k] <= m[r] && m[r] <= M[k]) || (m[k] <= M[r] && M[r] <= M[k])) flg = PETSC_FALSE;
    PetscCall(PetscFree2(m, M));
    PetscCall(PetscFree2(rows, cols));
    if (!flg) {
      PetscCall(PetscInfo(A, "MAT_FACTOR_CHOLESKY not supported for intersecting row maps. Use MAT_FACTOR_LU.\n"));
      PetscFunctionReturn(PETSC_SUCCESS);
    }
  }

  for (PetscInt r = 0; r < nr; r++) {
    for (PetscInt c = 0; c < nc; c++) {
      Mat       sub = mats[r][c];
      PetscBool isSeqAIJ, isMPIAIJ, isSeqBAIJ, isMPIBAIJ, isSeqSBAIJ, isMPISBAIJ, isDiag, isDense;

      if (!sub || (ftype == MAT_FACTOR_CHOLESKY && c < r)) continue;
      PetscCall(MatGetTranspose_TransposeVirtual(&sub, NULL, NULL, NULL, NULL));
      PetscCall(PetscObjectBaseTypeCompare((PetscObject)sub, MATSEQAIJ, &isSeqAIJ));
      PetscCall(PetscObjectBaseTypeCompare((PetscObject)sub, MATMPIAIJ, &isMPIAIJ));
      PetscCall(PetscObjectBaseTypeCompare((PetscObject)sub, MATSEQBAIJ, &isSeqBAIJ));
      PetscCall(PetscObjectBaseTypeCompare((PetscObject)sub, MATMPIBAIJ, &isMPIBAIJ));
      PetscCall(PetscObjectBaseTypeCompare((PetscObject)sub, MATSEQSBAIJ, &isSeqSBAIJ));
      PetscCall(PetscObjectBaseTypeCompare((PetscObject)sub, MATMPISBAIJ, &isMPISBAIJ));
      PetscCall(PetscObjectTypeCompare((PetscObject)sub, MATDIAGONAL, &isDiag));
      PetscCall(PetscObjectTypeCompareAny((PetscObject)sub, &isDense, MATSEQDENSE, MATMPIDENSE, NULL));
      if (ftype == MAT_FACTOR_CHOLESKY) {
        if (r == c) {
          if (!isSeqAIJ && !isMPIAIJ && !isSeqBAIJ && !isMPIBAIJ && !isSeqSBAIJ && !isMPISBAIJ && !isDiag && !isDense) {
            PetscCall(PetscInfo(sub, "MAT_FACTOR_CHOLESKY not supported for diagonal block of type %s.\n", ((PetscObject)sub)->type_name));
            flg = PETSC_FALSE;
          }
        } else if (!isSeqAIJ && !isMPIAIJ && !isSeqBAIJ && !isMPIBAIJ && !isDiag && !isDense) {
          PetscCall(PetscInfo(sub, "MAT_FACTOR_CHOLESKY not supported for off-diagonal block of type %s.\n", ((PetscObject)sub)->type_name));
          flg = PETSC_FALSE;
        }
      } else if (!isSeqAIJ && !isMPIAIJ && !isSeqBAIJ && !isMPIBAIJ && !isDiag && !isDense) {
        PetscCall(PetscInfo(sub, "MAT_FACTOR_LU not supported for block of type %s.\n", ((PetscObject)sub)->type_name));
        flg = PETSC_FALSE;
      }
    }
  }
  if (!flg) PetscFunctionReturn(PETSC_SUCCESS);

  /* Create the factorization matrix */
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
  PetscCall(MatSetSizes(B, A->rmap->n, A->cmap->n, A->rmap->N, A->cmap->N));
  PetscCall(PetscStrallocpy(MATSOLVERMUMPS, &((PetscObject)B)->type_name));
  PetscCall(MatSetUp(B));

  PetscCall(PetscNew(&mumps));

  B->ops->view    = MatView_MUMPS;
  B->ops->getinfo = MatGetInfo_MUMPS;

  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatFactorGetSolverType_C", MatFactorGetSolverType_mumps));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatFactorSetSchurIS_C", MatFactorSetSchurIS_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatFactorCreateSchurComplement_C", MatFactorCreateSchurComplement_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsSetIcntl_C", MatMumpsSetIcntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetIcntl_C", MatMumpsGetIcntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsSetCntl_C", MatMumpsSetCntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetCntl_C", MatMumpsGetCntl_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetInfo_C", MatMumpsGetInfo_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetInfog_C", MatMumpsGetInfog_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetRinfo_C", MatMumpsGetRinfo_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetRinfog_C", MatMumpsGetRinfog_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetNullPivots_C", MatMumpsGetNullPivots_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetInverse_C", MatMumpsGetInverse_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsGetInverseTranspose_C", MatMumpsGetInverseTranspose_MUMPS));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMumpsSetBlk_C", MatMumpsSetBlk_MUMPS));

  if (ftype == MAT_FACTOR_LU) {
    B->ops->lufactorsymbolic = MatLUFactorSymbolic_AIJMUMPS;
    B->factortype            = MAT_FACTOR_LU;
    mumps->sym               = 0;
  } else {
    B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_MUMPS;
    B->factortype                  = MAT_FACTOR_CHOLESKY;
#if defined(PETSC_USE_COMPLEX)
    mumps->sym = 2;
#else
    if (A->spd == PETSC_BOOL3_TRUE) mumps->sym = 1;
    else mumps->sym = 2;
#endif
  }
  mumps->ConvertToTriples = MatConvertToTriples_nest_xaij;
  PetscCall(PetscStrallocpy(MATORDERINGEXTERNAL, (char **)&B->preferredordering[ftype]));

  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A), &size));
  if (size == 1) {
    /* MUMPS option -mat_mumps_icntl_7 1 is automatically set if PETSc ordering is passed into symbolic factorization */
    B->canuseordering = PETSC_TRUE;
  }

  /* set solvertype */
  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERMUMPS, &B->solvertype));
  B->ops->destroy = MatDestroy_MUMPS;
  B->data         = (void *)mumps;

  *F               = B;
  mumps->id.job    = JOB_NULL;
  mumps->ICNTL_pre = NULL;
  mumps->CNTL_pre  = NULL;
  mumps->matstruc  = DIFFERENT_NONZERO_PATTERN;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatSolverTypeRegister_MUMPS(void)
{
  PetscFunctionBegin;
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS, MATMPIAIJ, MAT_FACTOR_LU, MatGetFactor_aij_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS, MATMPIAIJ, MAT_FACTOR_CHOLESKY, MatGetFactor_aij_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS, MATMPIBAIJ, MAT_FACTOR_LU, MatGetFactor_baij_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS, MATMPIBAIJ, MAT_FACTOR_CHOLESKY, MatGetFactor_baij_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS, MATMPISBAIJ, MAT_FACTOR_CHOLESKY, MatGetFactor_sbaij_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS, MATSEQAIJ, MAT_FACTOR_LU, MatGetFactor_aij_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS, MATSEQAIJ, MAT_FACTOR_CHOLESKY, MatGetFactor_aij_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS, MATSEQBAIJ, MAT_FACTOR_LU, MatGetFactor_baij_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS, MATSEQBAIJ, MAT_FACTOR_CHOLESKY, MatGetFactor_baij_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS, MATSEQSBAIJ, MAT_FACTOR_CHOLESKY, MatGetFactor_sbaij_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS, MATSEQSELL, MAT_FACTOR_LU, MatGetFactor_sell_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS, MATDIAGONAL, MAT_FACTOR_LU, MatGetFactor_aij_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS, MATDIAGONAL, MAT_FACTOR_CHOLESKY, MatGetFactor_aij_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS, MATSEQDENSE, MAT_FACTOR_LU, MatGetFactor_aij_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS, MATSEQDENSE, MAT_FACTOR_CHOLESKY, MatGetFactor_aij_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS, MATMPIDENSE, MAT_FACTOR_LU, MatGetFactor_aij_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS, MATMPIDENSE, MAT_FACTOR_CHOLESKY, MatGetFactor_aij_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS, MATNEST, MAT_FACTOR_LU, MatGetFactor_nest_mumps));
  PetscCall(MatSolverTypeRegister(MATSOLVERMUMPS, MATNEST, MAT_FACTOR_CHOLESKY, MatGetFactor_nest_mumps));
  PetscFunctionReturn(PETSC_SUCCESS);
}
