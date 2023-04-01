#include <petscvec_kokkos.hpp>
#include <petsc/private/deviceimpl.h>
#include <petsc/private/pcimpl.h>
#include <petsc/private/kspimpl.h>
#include <petscksp.h> /*I "petscksp.h" I*/
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/mat/impls/aij/mpi/kokkos/mpiaijkok.hpp>
#include "petscsection.h"
#include <petscdmcomposite.h>
#include "Kokkos_Core.hpp"

#include <../src/mat/impls/aij/seq/aij.h>
#include <../src/mat/impls/aij/seq/kokkos/aijkok.hpp>

#if defined(PETSC_HAVE_CUDA)
  #include <nvToolsExt.h>
#endif

#include <petscdevice_cupm.h>

#define PCBJKOKKOS_SHARED_LEVEL 1 // 0 is shared, 1 is global
#define PCBJKOKKOS_VEC_SIZE     16
#define PCBJKOKKOS_TEAM_SIZE    16

#define PCBJKOKKOS_VERBOSE_LEVEL 1

typedef Kokkos::DefaultExecutionSpace exec_space;
using layout           = Kokkos::LayoutRight;
using IntView          = Kokkos::View<PetscInt **, layout, exec_space>;
using AMatrixValueView = const Kokkos::View<PetscScalar **, layout, exec_space>;
using XYType           = const Kokkos::View<PetscScalar **, layout, exec_space>;

typedef enum {
  BATCH_KSP_BICG_IDX,
  BATCH_KSP_TFQMR_IDX,
  BATCH_KSP_GMRES_IDX,
  NUM_BATCH_TYPES
} KSPIndex;
typedef struct {
  Vec                                               vec_diag;
  PetscInt                                          nBlocks; /* total number of blocks */
  PetscInt                                          n;       // cache host version of d_bid_eqOffset_k[nBlocks]
  KSP                                               ksp;     // Used just for options. Should have one for each block
  Kokkos::View<PetscInt *, Kokkos::LayoutRight>    *d_bid_eqOffset_k;
  Kokkos::View<PetscScalar *, Kokkos::LayoutRight> *d_idiag_k;
  Kokkos::View<PetscInt *>                         *d_isrow_k;
  Kokkos::View<PetscInt *>                         *d_isicol_k;
  KSPIndex                                          ksp_type_idx;
  PetscInt                                          nwork;
  PetscInt                                          const_block_size; // used to decide to use shared memory for work vectors
  PetscInt                                         *dm_Nf;            // Number of fields in each DM
  PetscInt                                          num_dms;
  // diagnostics
  PetscBool reason;
  PetscBool monitor;
  PetscInt  batch_target;
  PetscInt  nsolves_team;
  PetscInt  max_nits;
  // caches
  IntView          *rowOffsets;
  IntView          *colIndices;
  XYType           *batch_b;
  XYType           *batch_x;
  AMatrixValueView *batch_values;
} PC_PCBJKOKKOS;

#if defined(PETSC_HAVE_KOKKOS_KERNELS_GMRES)
  #include <fstream>

  #include "Kokkos_Timer.hpp"
  #include "Kokkos_Random.hpp"
  #include "Kokkos_UnorderedMap.hpp"
  #include "Kokkos_Sort.hpp"

  /// KokkosKernels headers
  #include "KokkosBatched_Util.hpp"
  #include "KokkosBatched_Vector.hpp"

  #include <Kokkos_ArithTraits.hpp>
  #include <KokkosBatched_Util.hpp>
  #include <KokkosBatched_Vector.hpp>
  #include <KokkosBatched_Copy_Decl.hpp>
  #include <KokkosBatched_Copy_Impl.hpp>
  #include <KokkosBatched_AddRadial_Decl.hpp>
  #include <KokkosBatched_AddRadial_Impl.hpp>
  #include <KokkosBatched_Gemm_Decl.hpp>
  #include <KokkosBatched_Gemm_Serial_Impl.hpp>
  #include <KokkosBatched_Gemm_Team_Impl.hpp>
  #include <KokkosBatched_Gemv_Decl.hpp>
  #include <KokkosBatched_Gemv_Serial_Impl.hpp>
  #include <KokkosBatched_Gemv_Team_Impl.hpp>
  #include <KokkosBatched_Trsm_Decl.hpp>
  #include <KokkosBatched_Trsm_Serial_Impl.hpp>
  #include <KokkosBatched_Trsm_Team_Impl.hpp>
  #include <KokkosBatched_Trsv_Decl.hpp>
  #include <KokkosBatched_Trsv_Serial_Impl.hpp>
  #include <KokkosBatched_Trsv_Team_Impl.hpp>
  #include <KokkosBatched_LU_Decl.hpp>
  #include <KokkosBatched_LU_Serial_Impl.hpp>
  #include <KokkosBatched_LU_Team_Impl.hpp>
  #include <KokkosSparse_CrsMatrix.hpp>
  #include "KokkosBatched_Spmv.hpp"
  #include "KokkosBatched_CrsMatrix.hpp"
  #include "KokkosBatched_Krylov_Handle.hpp"
  #include "KokkosBatched_GMRES.hpp"
  #include "KokkosBatched_JacobiPrec.hpp"

template <typename DeviceType, typename ValuesViewType, typename IntView, typename VectorViewType, typename KrylovHandleType>
struct Functor_TestBatchedTeamVectorGMRES {
  const ValuesViewType _D;
  const ValuesViewType _diag;
  const IntView        _r;
  const IntView        _c;
  const VectorViewType _X;
  const VectorViewType _B;
  const int            _N_team, _team_size, _vector_length;
  const int            _N_iteration;
  const double         _tol;
  const int            _ortho_strategy;
  const int            _scratch_pad_level;
  KrylovHandleType     _handle;

  KOKKOS_INLINE_FUNCTION
  Functor_TestBatchedTeamVectorGMRES(const ValuesViewType &D, const IntView &r, const IntView &c, const VectorViewType &X, const VectorViewType &B, const int N_team, const int team_size, const int vector_length, const int N_iteration, const double tol, const int ortho_strategy, const int scratch_pad_level, KrylovHandleType &handle) :
    _D(D), _r(r), _c(c), _X(X), _B(B), _N_team(N_team), _team_size(team_size), _vector_length(vector_length), _N_iteration(N_iteration), _tol(tol), _ortho_strategy(ortho_strategy), _scratch_pad_level(scratch_pad_level), _handle(handle)
  {
  }

  KOKKOS_INLINE_FUNCTION
  Functor_TestBatchedTeamVectorGMRES(const ValuesViewType &D, const ValuesViewType &diag, const IntView &r, const IntView &c, const VectorViewType &X, const VectorViewType &B, const int N_team, const int team_size, const int vector_length, const int N_iteration, const double tol, int ortho_strategy, const int scratch_pad_level, KrylovHandleType &handle) :
    _D(D), _diag(diag), _r(r), _c(c), _X(X), _B(B), _N_team(N_team), _team_size(team_size), _vector_length(vector_length), _N_iteration(N_iteration), _tol(tol), _ortho_strategy(ortho_strategy), _scratch_pad_level(scratch_pad_level), _handle(handle)
  {
  }

  template <typename MemberType>
  KOKKOS_INLINE_FUNCTION void operator()(const MemberType &member) const
  {
    const int first_matrix = static_cast<int>(member.league_rank()) * _N_team;
    const int N            = _D.extent(0);
    const int last_matrix  = (static_cast<int>(member.league_rank() + 1) * _N_team < N ? static_cast<int>(member.league_rank() + 1) * _N_team : N);
    const int graphID      = static_cast<int>(member.league_rank());
    using TeamVectorCopy1D = KokkosBatched::TeamVectorCopy<MemberType, KokkosBatched::Trans::NoTranspose, 1>;

    auto d                         = Kokkos::subview(_D, Kokkos::make_pair(first_matrix, last_matrix), Kokkos::ALL);
    auto x                         = Kokkos::subview(_X, Kokkos::make_pair(first_matrix, last_matrix), Kokkos::ALL);
    auto b                         = Kokkos::subview(_B, Kokkos::make_pair(first_matrix, last_matrix), Kokkos::ALL);
    using ScratchPadIntViewType    = Kokkos::View<typename IntView::non_const_value_type *, typename IntView::array_layout, typename IntView::execution_space::scratch_memory_space>;
    using ScratchPadValuesViewType = Kokkos::View<typename ValuesViewType::non_const_value_type **, typename ValuesViewType::array_layout, typename ValuesViewType::execution_space::scratch_memory_space>;

    using Operator = KokkosBatched::CrsMatrix<ValuesViewType, ScratchPadIntViewType>;
    ScratchPadIntViewType r(member.team_scratch(1), _r.extent(1));
    ScratchPadIntViewType c(member.team_scratch(1), _c.extent(1));

    TeamVectorCopy1D::invoke(member, Kokkos::subview(_r, graphID, Kokkos::ALL), r);
    TeamVectorCopy1D::invoke(member, Kokkos::subview(_c, graphID, Kokkos::ALL), c);
    Operator A(d, r, c);

    ScratchPadValuesViewType diag(member.team_scratch(1), last_matrix - first_matrix, _diag.extent(1));
    using PrecOperator = KokkosBatched::JacobiPrec<ScratchPadValuesViewType>;

    KokkosBatched::TeamVectorCopy<MemberType>::invoke(member, Kokkos::subview(_diag, Kokkos::make_pair(first_matrix, last_matrix), Kokkos::ALL), diag);
    PrecOperator P(diag);
    P.setComputedInverse();

    KokkosBatched::TeamVectorGMRES<MemberType>::template invoke<Operator, VectorViewType, PrecOperator, KrylovHandleType>(member, A, b, x, P, _handle);
  }
  inline double run(PC pc)
  {
    typedef typename ValuesViewType::value_type value_type;
    std::string                                 name("KokkosBatched::Test::TeamVectorGMRES");
    Kokkos::Timer                               timer;
    Kokkos::Profiling::pushRegion(name.c_str());

    Kokkos::TeamPolicy<DeviceType> auto_policy(ceil(1. * _D.extent(0) / _N_team), Kokkos::AUTO(), Kokkos::AUTO());
    Kokkos::TeamPolicy<DeviceType> tuned_policy(ceil(1. * _D.extent(0) / _N_team), _team_size, _vector_length);
    Kokkos::TeamPolicy<DeviceType> policy;

    if (_team_size < 1) policy = auto_policy;
    else policy = tuned_policy;

    _handle.set_max_iteration(_N_iteration);
    _handle.set_tolerance(_tol);
    _handle.set_ortho_strategy(_ortho_strategy);
    _handle.set_scratch_pad_level(_scratch_pad_level);
    _handle.set_compute_last_residual(true);

    int maximum_iteration = _handle.get_max_iteration();

    using ScalarType = typename ValuesViewType::non_const_value_type;
    using Layout     = typename ValuesViewType::array_layout;
    using EXSP       = typename ValuesViewType::execution_space;

    using MagnitudeType = typename Kokkos::Details::ArithTraits<ScalarType>::mag_type;

    using ViewType1D    = Kokkos::View<MagnitudeType *, Layout, EXSP>;
    using ViewType2D    = Kokkos::View<ScalarType **, Layout, EXSP>;
    using ViewType3D    = Kokkos::View<ScalarType ***, Layout, EXSP>;
    using IntViewType1D = Kokkos::View<PetscInt *, Layout, EXSP>;

    size_t bytes_1D      = ViewType2D::shmem_size(_N_team, 1);
    size_t bytes_row_ptr = IntViewType1D::shmem_size(_r.extent(1));
    size_t bytes_col_idc = IntViewType1D::shmem_size(_c.extent(1));
    size_t bytes_2D_1    = ViewType2D::shmem_size(_N_team, _X.extent(1));
    size_t bytes_2D_2    = ViewType2D::shmem_size(_N_team, maximum_iteration + 1);

    size_t bytes_diag = bytes_2D_1;
    size_t bytes_tmp  = 2 * bytes_2D_1 + 2 * bytes_1D + bytes_2D_2;

    policy.set_scratch_size(0, Kokkos::PerTeam(bytes_tmp));
    policy.set_scratch_size(1, Kokkos::PerTeam(bytes_col_idc + bytes_row_ptr + bytes_diag));
    PetscCall(PetscInfo(pc, "%d scratch memory(0) = %d + %d + %d bytes_diag=%d; %d scratch memory(1); %d maximum_iterations\n", (int)(bytes_tmp), 2 * (int)bytes_2D_1, 2 * (int)bytes_1D, (int)bytes_2D_2, (int)bytes_diag, (int)(bytes_row_ptr + bytes_col_idc + bytes_diag), (int)maximum_iteration));
    exec_space().fence();
    timer.reset();
    Kokkos::parallel_for(name.c_str(), policy, *this);
    exec_space().fence();
    double sec = timer.seconds();

    return sec;
  }
};
#endif // KK GMRES

typedef Kokkos::TeamPolicy<>::member_type team_member;

static PetscErrorCode PCBJKOKKOSCreateKSP_BJKOKKOS(PC pc)
{
  const char    *prefix;
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS *)pc->data;
  DM             dm;

  PetscFunctionBegin;
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)pc), &jac->ksp));
  PetscCall(KSPSetErrorIfNotConverged(jac->ksp, pc->erroriffailure));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject)jac->ksp, (PetscObject)pc, 1));
  PetscCall(PCGetOptionsPrefix(pc, &prefix));
  PetscCall(KSPSetOptionsPrefix(jac->ksp, prefix));
  PetscCall(KSPAppendOptionsPrefix(jac->ksp, "pc_bjkokkos_"));
  PetscCall(PCGetDM(pc, &dm));
  if (dm) {
    PetscCall(KSPSetDM(jac->ksp, dm));
    PetscCall(KSPSetDMActive(jac->ksp, PETSC_FALSE));
  }
  jac->reason       = PETSC_FALSE;
  jac->monitor      = PETSC_FALSE;
  jac->batch_target = -1;
  jac->nsolves_team = 1;
  jac->ksp->max_it  = 50; // this is really for GMRES w/o restarts
  PetscFunctionReturn(PETSC_SUCCESS);
}

// y <-- Ax
KOKKOS_INLINE_FUNCTION PetscErrorCode MatMult(const team_member team, const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, const PetscInt *r, const PetscInt *ic, const PetscInt start, const PetscInt end, const PetscScalar *x_loc, PetscScalar *y_loc)
{
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, start, end), [=](const int rowb) {
    int                rowa = ic[rowb];
    int                n    = glb_Aai[rowa + 1] - glb_Aai[rowa];
    const PetscInt    *aj   = glb_Aaj + glb_Aai[rowa]; // global
    const PetscScalar *aa   = glb_Aaa + glb_Aai[rowa];
    PetscScalar        sum;
    Kokkos::parallel_reduce(
      Kokkos::ThreadVectorRange(team, n), [=](const int i, PetscScalar &lsum) { lsum += aa[i] * x_loc[r[aj[i]] - start]; }, sum);
    Kokkos::single(Kokkos::PerThread(team), [=]() { y_loc[rowb - start] = sum; });
  });
  team.team_barrier();
  return PETSC_SUCCESS;
}

// temp buffer per thread with reduction at end?
KOKKOS_INLINE_FUNCTION PetscErrorCode MatMultTranspose(const team_member team, const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, const PetscInt *r, const PetscInt *ic, const PetscInt start, const PetscInt end, const PetscScalar *x_loc, PetscScalar *y_loc)
{
  Kokkos::parallel_for(Kokkos::TeamVectorRange(team, end - start), [=](int i) { y_loc[i] = 0; });
  team.team_barrier();
  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, start, end), [=](const int rowb) {
    int                rowa = ic[rowb];
    int                n    = glb_Aai[rowa + 1] - glb_Aai[rowa];
    const PetscInt    *aj   = glb_Aaj + glb_Aai[rowa]; // global
    const PetscScalar *aa   = glb_Aaa + glb_Aai[rowa];
    const PetscScalar  xx   = x_loc[rowb - start]; // rowb = ic[rowa] = ic[r[rowb]]
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, n), [=](const int &i) {
      PetscScalar val = aa[i] * xx;
      Kokkos::atomic_fetch_add(&y_loc[r[aj[i]] - start], val);
    });
  });
  team.team_barrier();
  return PETSC_SUCCESS;
}

typedef struct Batch_MetaData_TAG {
  PetscInt           flops;
  PetscInt           its;
  KSPConvergedReason reason;
} Batch_MetaData;

// Solve A(BB^-1)x = y with TFQMR. Right preconditioned to get un-preconditioned residual
KOKKOS_INLINE_FUNCTION PetscErrorCode BJSolve_TFQMR(const team_member team, const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, const PetscInt *r, const PetscInt *ic, PetscScalar *work_space_global, const int stride_global, const int nShareVec, PetscScalar *work_space_shared, const int stride_shared, PetscReal rtol, PetscReal atol, PetscReal dtol, PetscInt maxit, Batch_MetaData *metad, const PetscInt start, const PetscInt end, const PetscScalar glb_idiag[], const PetscScalar *glb_b, PetscScalar *glb_x, bool monitor)
{
  using Kokkos::parallel_for;
  using Kokkos::parallel_reduce;
  int                Nblk = end - start, i, m, stride = stride_shared, idx = 0;
  PetscReal          dp, dpold, w, dpest, tau, psi, cm, r0;
  const PetscScalar *Diag = &glb_idiag[start];
  PetscScalar       *ptr  = work_space_shared, rho, rhoold, a, s, b, eta, etaold, psiold, cf, dpi;

  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *XX = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *R = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *RP = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *V = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *T = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Q = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *P = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *U = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *D = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *AUQ = V;

  // init: get b, zero x
  parallel_for(Kokkos::TeamVectorRange(team, start, end), [=](int rowb) {
    int rowa         = ic[rowb];
    R[rowb - start]  = glb_b[rowa];
    XX[rowb - start] = 0;
  });
  team.team_barrier();
  parallel_reduce(
    Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &lsum) { lsum += R[idx] * PetscConj(R[idx]); }, dpi);
  team.team_barrier();
  r0 = dp = PetscSqrtReal(PetscRealPart(dpi));
// diagnostics
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
  if (monitor) Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("%3d KSP Residual norm %14.12e \n", 0, (double)dp); });
#endif
  if (dp < atol) {
    metad->reason = KSP_CONVERGED_ATOL_NORMAL;
    return PETSC_SUCCESS;
  }
  if (0 == maxit) {
    metad->reason = KSP_DIVERGED_ITS;
    return PETSC_SUCCESS;
  }

  /* Make the initial Rp = R */
  parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) { RP[idx] = R[idx]; });
  team.team_barrier();
  /* Set the initial conditions */
  etaold = 0.0;
  psiold = 0.0;
  tau    = dp;
  dpold  = dp;

  /* rhoold = (r,rp)     */
  parallel_reduce(
    Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &dot) { dot += R[idx] * PetscConj(RP[idx]); }, rhoold);
  team.team_barrier();
  parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) {
    U[idx] = R[idx];
    P[idx] = R[idx];
    T[idx] = Diag[idx] * P[idx];
    D[idx] = 0;
  });
  team.team_barrier();
  static_cast<void>(MatMult(team, glb_Aai, glb_Aaj, glb_Aaa, r, ic, start, end, T, V));

  i = 0;
  do {
    /* s <- (v,rp)          */
    parallel_reduce(
      Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &dot) { dot += V[idx] * PetscConj(RP[idx]); }, s);
    team.team_barrier();
    a = rhoold / s; /* a <- rho / s         */
    /* q <- u - a v    VecWAXPY(w,alpha,x,y): w = alpha x + y.     */
    /* t <- u + q           */
    parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) {
      Q[idx] = U[idx] - a * V[idx];
      T[idx] = U[idx] + Q[idx];
    });
    team.team_barrier();
    // KSP_PCApplyBAorAB
    parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) { T[idx] = Diag[idx] * T[idx]; });
    team.team_barrier();
    static_cast<void>(MatMult(team, glb_Aai, glb_Aaj, glb_Aaa, r, ic, start, end, T, AUQ));
    /* r <- r - a K (u + q) */
    parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) { R[idx] = R[idx] - a * AUQ[idx]; });
    team.team_barrier();
    parallel_reduce(
      Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &lsum) { lsum += R[idx] * PetscConj(R[idx]); }, dpi);
    team.team_barrier();
    dp = PetscSqrtReal(PetscRealPart(dpi));
    for (m = 0; m < 2; m++) {
      if (!m) w = PetscSqrtReal(dp * dpold);
      else w = dp;
      psi = w / tau;
      cm  = 1.0 / PetscSqrtReal(1.0 + psi * psi);
      tau = tau * psi * cm;
      eta = cm * cm * a;
      cf  = psiold * psiold * etaold / a;
      if (!m) {
        /* D = U + cf D */
        parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) { D[idx] = U[idx] + cf * D[idx]; });
      } else {
        /* D = Q + cf D */
        parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) { D[idx] = Q[idx] + cf * D[idx]; });
      }
      team.team_barrier();
      parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) { XX[idx] = XX[idx] + eta * D[idx]; });
      team.team_barrier();
      dpest = PetscSqrtReal(2 * i + m + 2.0) * tau;
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
      if (monitor && m == 1) Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("%3d KSP Residual norm %14.12e \n", i + 1, (double)dpest); });
#endif
      if (dpest < atol) {
        metad->reason = KSP_CONVERGED_ATOL_NORMAL;
        goto done;
      }
      if (dpest / r0 < rtol) {
        metad->reason = KSP_CONVERGED_RTOL_NORMAL;
        goto done;
      }
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
      if (dpest / r0 > dtol) {
        metad->reason = KSP_DIVERGED_DTOL;
        Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("ERROR block %d diverged: %d it, res=%e, r_0=%e\n", team.league_rank(), i, dpest, r0); });
        goto done;
      }
#else
      if (dpest / r0 > dtol) {
        metad->reason = KSP_DIVERGED_DTOL;
        goto done;
      }
#endif
      if (i + 1 == maxit) {
        metad->reason = KSP_DIVERGED_ITS;
        goto done;
      }

      etaold = eta;
      psiold = psi;
    }

    /* rho <- (r,rp)       */
    parallel_reduce(
      Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &dot) { dot += R[idx] * PetscConj(RP[idx]); }, rho);
    team.team_barrier();
    b = rho / rhoold; /* b <- rho / rhoold   */
    /* u <- r + b q        */
    /* p <- u + b(q + b p) */
    parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) {
      U[idx] = R[idx] + b * Q[idx];
      Q[idx] = Q[idx] + b * P[idx];
      P[idx] = U[idx] + b * Q[idx];
    });
    /* v <- K p  */
    team.team_barrier();
    parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) { T[idx] = Diag[idx] * P[idx]; });
    team.team_barrier();
    static_cast<void>(MatMult(team, glb_Aai, glb_Aaj, glb_Aaa, r, ic, start, end, T, V));

    rhoold = rho;
    dpold  = dp;

    i++;
  } while (i < maxit);
done:
  // KSPUnwindPreconditioner
  parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) { XX[idx] = Diag[idx] * XX[idx]; });
  team.team_barrier();
  // put x into Plex order
  parallel_for(Kokkos::TeamVectorRange(team, start, end), [=](int rowb) {
    int rowa    = ic[rowb];
    glb_x[rowa] = XX[rowb - start];
  });
  metad->its = i + 1;
  if (1) {
    int nnz;
    parallel_reduce(
      Kokkos::TeamVectorRange(team, start, end), [=](const int idx, int &lsum) { lsum += (glb_Aai[idx + 1] - glb_Aai[idx]); }, nnz);
    metad->flops = 2 * (metad->its * (10 * Nblk + 2 * nnz) + 5 * Nblk);
  } else {
    metad->flops = 2 * (metad->its * (10 * Nblk + 2 * 50 * Nblk) + 5 * Nblk); // guess
  }
  return PETSC_SUCCESS;
}

// Solve Ax = y with biCG
KOKKOS_INLINE_FUNCTION PetscErrorCode BJSolve_BICG(const team_member team, const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, const PetscInt *r, const PetscInt *ic, PetscScalar *work_space_global, const int stride_global, const int nShareVec, PetscScalar *work_space_shared, const int stride_shared, PetscReal rtol, PetscReal atol, PetscReal dtol, PetscInt maxit, Batch_MetaData *metad, const PetscInt start, const PetscInt end, const PetscScalar glb_idiag[], const PetscScalar *glb_b, PetscScalar *glb_x, bool monitor)
{
  using Kokkos::parallel_for;
  using Kokkos::parallel_reduce;
  int                Nblk = end - start, i, stride = stride_shared, idx = 0; // start in shared mem
  PetscReal          dp, r0;
  const PetscScalar *Di  = &glb_idiag[start];
  PetscScalar       *ptr = work_space_shared, dpi, a = 1.0, beta, betaold = 1.0, b, b2, ma, mac;

  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *XX = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Rl = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Zl = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Pl = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Rr = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Zr = ptr;
  ptr += stride;
  if (idx++ == nShareVec) {
    ptr    = work_space_global;
    stride = stride_global;
  }
  PetscScalar *Pr = ptr;
  ptr += stride;

  /*     r <- b (x is 0) */
  parallel_for(Kokkos::TeamVectorRange(team, start, end), [=](int rowb) {
    int rowa         = ic[rowb];
    Rl[rowb - start] = Rr[rowb - start] = glb_b[rowa];
    XX[rowb - start]                    = 0;
  });
  team.team_barrier();
  /*     z <- Br         */
  parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) {
    Zr[idx] = Di[idx] * Rr[idx];
    Zl[idx] = Di[idx] * Rl[idx];
  });
  team.team_barrier();
  /*    dp <- r'*r       */
  parallel_reduce(
    Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &lsum) { lsum += Rr[idx] * PetscConj(Rr[idx]); }, dpi);
  team.team_barrier();
  r0 = dp = PetscSqrtReal(PetscRealPart(dpi));
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
  if (monitor) Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("%3d KSP Residual norm %14.12e \n", 0, (double)dp); });
#endif
  if (dp < atol) {
    metad->reason = KSP_CONVERGED_ATOL_NORMAL;
    return PETSC_SUCCESS;
  }
  if (0 == maxit) {
    metad->reason = KSP_DIVERGED_ITS;
    return PETSC_SUCCESS;
  }
  i = 0;
  do {
    /*     beta <- r'z     */
    parallel_reduce(
      Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &dot) { dot += Zr[idx] * PetscConj(Rl[idx]); }, beta);
    team.team_barrier();
#if PCBJKOKKOS_VERBOSE_LEVEL >= 6
  #if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
    Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("%7d beta = Z.R = %22.14e \n", i, (double)beta); });
  #endif
#endif
    if (!i) {
      if (beta == 0.0) {
        metad->reason = KSP_DIVERGED_BREAKDOWN_BICG;
        goto done;
      }
      /*     p <- z          */
      parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) {
        Pr[idx] = Zr[idx];
        Pl[idx] = Zl[idx];
      });
    } else {
      b = beta / betaold;
      /*     p <- z + b* p   */
      b2 = PetscConj(b);
      parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) {
        Pr[idx] = b * Pr[idx] + Zr[idx];
        Pl[idx] = b2 * Pl[idx] + Zl[idx];
      });
    }
    team.team_barrier();
    betaold = beta;
    /*     z <- Kp         */
    static_cast<void>(MatMult(team, glb_Aai, glb_Aaj, glb_Aaa, r, ic, start, end, Pr, Zr));
    static_cast<void>(MatMultTranspose(team, glb_Aai, glb_Aaj, glb_Aaa, r, ic, start, end, Pl, Zl));
    /*     dpi <- z'p      */
    parallel_reduce(
      Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &lsum) { lsum += Zr[idx] * PetscConj(Pl[idx]); }, dpi);
    team.team_barrier();
    //
    a   = beta / dpi; /*     a = beta/p'z    */
    ma  = -a;
    mac = PetscConj(ma);
    /*     x <- x + ap     */
    parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) {
      XX[idx] = XX[idx] + a * Pr[idx];
      Rr[idx] = Rr[idx] + ma * Zr[idx];
      Rl[idx] = Rl[idx] + mac * Zl[idx];
    });
    team.team_barrier();
    team.team_barrier();
    /*    dp <- r'*r       */
    parallel_reduce(
      Kokkos::TeamVectorRange(team, Nblk), [=](const int idx, PetscScalar &lsum) { lsum += Rr[idx] * PetscConj(Rr[idx]); }, dpi);
    team.team_barrier();
    dp = PetscSqrtReal(PetscRealPart(dpi));
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
    if (monitor) Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("%3d KSP Residual norm %14.12e \n", i + 1, (double)dp); });
#endif
    if (dp < atol) {
      metad->reason = KSP_CONVERGED_ATOL_NORMAL;
      goto done;
    }
    if (dp / r0 < rtol) {
      metad->reason = KSP_CONVERGED_RTOL_NORMAL;
      goto done;
    }
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
    if (dp / r0 > dtol) {
      metad->reason = KSP_DIVERGED_DTOL;
      Kokkos::single(Kokkos::PerTeam(team), [=]() { printf("ERROR block %d diverged: %d it, res=%e, r_0=%e\n", team.league_rank(), i, dp, r0); });
      goto done;
    }
#else
    if (dp / r0 > dtol) {
      metad->reason = KSP_DIVERGED_DTOL;
      goto done;
    }
#endif
    if (i + 1 == maxit) {
      metad->reason = KSP_DIVERGED_ITS;
      goto done;
    }
    /* z <- Br  */
    parallel_for(Kokkos::TeamVectorRange(team, Nblk), [=](int idx) {
      Zr[idx] = Di[idx] * Rr[idx];
      Zl[idx] = Di[idx] * Rl[idx];
    });
    i++;
    team.team_barrier();
  } while (i < maxit);
done:
  // put x back into Plex order
  parallel_for(Kokkos::TeamVectorRange(team, start, end), [=](int rowb) {
    int rowa    = ic[rowb];
    glb_x[rowa] = XX[rowb - start];
  });
  metad->its = i + 1;
  if (1) {
    int nnz;
    parallel_reduce(
      Kokkos::TeamVectorRange(team, start, end), [=](const int idx, int &lsum) { lsum += (glb_Aai[idx + 1] - glb_Aai[idx]); }, nnz);
    metad->flops = 2 * (metad->its * (10 * Nblk + 2 * nnz) + 5 * Nblk);
  } else {
    metad->flops = 2 * (metad->its * (10 * Nblk + 2 * 50 * Nblk) + 5 * Nblk); // guess
  }
  return PETSC_SUCCESS;
}

// KSP solver solve Ax = b; xout is output, bin is input
static PetscErrorCode PCApply_BJKOKKOS(PC pc, Vec bin, Vec xout)
{
  PC_PCBJKOKKOS    *jac = (PC_PCBJKOKKOS *)pc->data;
  Mat               A   = pc->pmat;
  Mat_SeqAIJKokkos *aijkok;

  PetscFunctionBegin;
  aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);
  if (!aijkok) aijkok = static_cast<Mat_SeqAIJKokkos *>(((Mat_MPIAIJ *)A->data)->A->spptr);
  PetscCheck(aijkok, PetscObjectComm((PetscObject)pc), PETSC_ERR_USER, "No aijkok");
  {
    PetscInt           maxit = jac->ksp->max_it;
    const PetscInt     conc = Kokkos::DefaultExecutionSpace().concurrency(), openmp = !!(conc < 1000), team_size = (openmp == 0 && PCBJKOKKOS_VEC_SIZE != 1) ? PCBJKOKKOS_TEAM_SIZE : 1;
    const PetscInt     nwork = jac->nwork, nBlk = jac->nBlocks;
    PetscScalar       *glb_xdata = NULL;
    PetscReal          rtol = jac->ksp->rtol, atol = jac->ksp->abstol, dtol = jac->ksp->divtol;
    const PetscScalar *glb_idiag = jac->d_idiag_k->data(), *glb_bdata = NULL;
    const PetscInt    *glb_Aai = aijkok->i_device_data(), *glb_Aaj = aijkok->j_device_data(), *d_bid_eqOffset = jac->d_bid_eqOffset_k->data();
    const PetscScalar *glb_Aaa  = aijkok->a_device_data();
    const PetscInt    *d_isicol = jac->d_isicol_k->data(), *d_isrow = jac->d_isrow_k->data();
    PCFailedReason     pcreason;
    KSPIndex           ksp_type_idx = jac->ksp_type_idx;
    PetscMemType       mtype;
    PetscContainer     container;
    PetscInt           batch_sz;                // the number of repeated DMs, [DM_e_1, DM_e_2, DM_e_batch_sz, DM_i_1, ...]
    VecScatter         plex_batch = NULL;       // not used
    Vec                bvec;                    // a copy of b for scatter (just alias to bin now)
    PetscBool          monitor  = jac->monitor; // captured
    PetscInt           view_bid = jac->batch_target;
    MatInfo            info;
    //PetscCall(MatGetOwnershipRange(A, &Istart, NULL));
    jac->max_nits = 0;
    if (view_bid < 0) view_bid = 0;
    PetscCall(MatGetInfo(A, MAT_LOCAL, &info));
    // get field major is to map plex IO to/from block/field major
    PetscCall(PetscObjectQuery((PetscObject)A, "plex_batch_is", (PetscObject *)&container));
    if (container) {
      PetscCall(VecDuplicate(bin, &bvec));
      PetscCall(PetscContainerGetPointer(container, (void **)&plex_batch));
      PetscCall(VecScatterBegin(plex_batch, bin, bvec, INSERT_VALUES, SCATTER_FORWARD));
      PetscCall(VecScatterEnd(plex_batch, bin, bvec, INSERT_VALUES, SCATTER_FORWARD));
      SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_USER, "No plex_batch_is -- require NO field major ordering for now");
    } else {
      bvec = bin;
    }
    // get x
    PetscCall(VecGetArrayAndMemType(xout, &glb_xdata, &mtype));
#if defined(PETSC_HAVE_CUDA)
    PetscCheck(PetscMemTypeDevice(mtype), PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "No GPU data for x %" PetscInt_FMT " != %" PetscInt_FMT, mtype, PETSC_MEMTYPE_DEVICE);
#endif
    PetscCall(VecGetArrayReadAndMemType(bvec, &glb_bdata, &mtype));
#if defined(PETSC_HAVE_CUDA)
    PetscCheck(PetscMemTypeDevice(mtype), PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "No GPU data for b");
#endif
    // get batch size
    PetscCall(PetscObjectQuery((PetscObject)A, "batch size", (PetscObject *)&container));
    if (container) {
      PetscInt *pNf = NULL;
      PetscCall(PetscContainerGetPointer(container, (void **)&pNf));
      batch_sz = *pNf; // number of times to repeat the DMs
    } else batch_sz = 1;
    PetscCheck(nBlk % batch_sz == 0, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "batch_sz = %" PetscInt_FMT ", nBlk = %" PetscInt_FMT, batch_sz, nBlk);
    if (ksp_type_idx == BATCH_KSP_GMRES_IDX) { // KK solver - move PETSc data into Kokkos Views, setup solver, solve, move data out of Kokkos, process metadata (convergence tests, etc.)
#if defined(PETSC_HAVE_KOKKOS_KERNELS_GMRES)
      int       Nsolves_team = jac->nsolves_team, fill_idx = 0;
      int       Nloc    = jac->const_block_size; // same grids
      const int Nsolves = nBlk;
      const int nnz     = (int)info.nz_used / Nsolves;      // fix for variable grid size
      if (Nsolves_team > batch_sz) Nsolves_team = batch_sz; // silently fix this
      PetscCheck(jac->const_block_size, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "Kokkos (GMRES) solver requires constant block size (but can be made to work with species ordering or N_team==1)");
      PetscCheck(Nsolves % Nsolves_team == 0, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "Nsolves.mod(Nsolves_team) != 0: Nsolves = %d, Nsolves_team = %d", Nsolves, Nsolves_team);
      PetscCheck(((int)info.nz_used) % Nsolves == 0, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "info.nz_used.mod(Nsolves) != 0: info.nz_used = %g, Nsolves = %d", info.nz_used, Nsolves);
  #if defined(PETSC_HAVE_CUDA)
      nvtxRangePushA("gmres-kk");
  #endif
      Kokkos::View<PetscScalar **, layout, exec_space, Kokkos::MemoryTraits<Kokkos::Unmanaged>> inv_diag((PetscScalar *)glb_idiag, Nsolves, Nloc); // in correct order
      if (!jac->rowOffsets) {
        jac->rowOffsets   = new IntView("rowOffsets", Nsolves / Nsolves_team, Nloc + 1); // same grids
        jac->colIndices   = new IntView("colIndices", Nsolves / Nsolves_team, nnz);
        jac->batch_b      = new XYType("batch rhs", Nsolves, Nloc);
        jac->batch_x      = new XYType("batch sol", Nsolves, Nloc);
        jac->batch_values = new AMatrixValueView("batch values", Nsolves, nnz);
        fill_idx          = 1;
        PetscCall(PetscInfo(pc, "Setup indices Nloc=%d, nnz=%d\n", Nloc, nnz));
      }
      IntView          &rowOffsets   = *jac->rowOffsets;
      IntView          &colIndices   = *jac->colIndices;
      XYType           &batch_b      = *jac->batch_b;
      XYType           &batch_x      = *jac->batch_x;
      AMatrixValueView &batch_values = *jac->batch_values;

      Kokkos::deep_copy(batch_x, 0.);
      PetscCall(PetscInfo(pc, "\tjac->n = %" PetscInt_FMT ", Nloc = %d, Nsolves = %d, nnz = %d, Nsolves_team = %d, league size = %d, maxit = %" PetscInt_FMT "\n", jac->n, Nloc, Nsolves, nnz, Nsolves_team, Nsolves / Nsolves_team, maxit));
      Kokkos::parallel_for(
        "rowOffsets+map", Kokkos::TeamPolicy<>(Nsolves, team_size, PCBJKOKKOS_VEC_SIZE), KOKKOS_LAMBDA(const team_member team) {
          const int blkID = team.league_rank(), start = d_bid_eqOffset[blkID], end = d_bid_eqOffset[blkID + 1];
          if (fill_idx) {
            if (blkID % Nsolves_team == 0) {                                                        // first matrix on this member
              Kokkos::parallel_for(Kokkos::TeamVectorRange(team, start, end), [=](const int rowb) { // Nloc
                int rowa                                           = d_isicol[rowb];
                int n                                              = glb_Aai[rowa + 1] - glb_Aai[rowa];
                rowOffsets(blkID / Nsolves_team, rowb + 1 - start) = n; // save sizes
              });
            }
          }
          // map b into field major space
          Kokkos::parallel_for(Kokkos::TeamVectorRange(team, start, end), [=](int rowb) {
            int rowa                     = d_isicol[rowb];
            batch_b(blkID, rowb - start) = glb_bdata[rowa];
          });
        });
      Kokkos::fence();
      if (fill_idx) {
        Kokkos::parallel_for(
          "prefix sum", Kokkos::TeamPolicy<>(Nsolves / Nsolves_team, 1, 1), KOKKOS_LAMBDA(const team_member team) {
            const int graphID      = team.league_rank();
            rowOffsets(graphID, 0) = 0;
            for (size_t i = 0; i < Nloc; ++i) rowOffsets(graphID, i + 1) += rowOffsets(graphID, i);
          });
        Kokkos::fence();
      }
      Kokkos::parallel_for(
        "copy matrix", Kokkos::TeamPolicy<>(Nsolves /* /batch_sz */, team_size, PCBJKOKKOS_VEC_SIZE), KOKKOS_LAMBDA(const team_member team) {
          const int blkID = team.league_rank(), start = d_bid_eqOffset[blkID], end = d_bid_eqOffset[blkID + 1], graphID = blkID / Nsolves_team;
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, start, end), [=](const int rowb) {
            int                rowa = d_isicol[rowb];
            int                n    = glb_Aai[rowa + 1] - glb_Aai[rowa];
            const PetscInt    *aj   = glb_Aaj + glb_Aai[rowa]; // global index
            const PetscScalar *aa   = glb_Aaa + glb_Aai[rowa];
            Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, n), [=](const int &i) {
              PetscScalar val = aa[i];
              if (fill_idx && blkID % Nsolves_team == 0) colIndices(graphID, rowOffsets(graphID, rowb - start) + i) = d_isrow[aj[i]] - blkID * Nloc; // local" global - block start
              batch_values(blkID, rowOffsets(graphID, rowb - start) + i) = val;
            });
          });
        });
      Kokkos::fence();
      // setup solver
      using ScalarType                = typename AMatrixValueView::non_const_value_type;
      using MagnitudeType             = typename Kokkos::Details::ArithTraits<ScalarType>::mag_type;
      using NormViewType              = Kokkos::View<MagnitudeType *, layout, exec_space>;
      using Norm2DViewType            = Kokkos::View<MagnitudeType **, layout, exec_space>;
      using Scalar3DViewType          = Kokkos::View<ScalarType ***, layout, exec_space>;
      using IntViewType               = Kokkos::View<int *, layout, exec_space>;
      using KrylovHandleType          = KokkosBatched::KrylovHandle<Norm2DViewType, IntViewType, Scalar3DViewType>;
      const int        n_iterations   = maxit;
      const int        team_size      = -1;
      const int        vector_length  = -1;
      const double     tol            = rtol;
      const int        ortho_strategy = 0;
      KrylovHandleType handle(Nsolves, Nsolves_team, n_iterations, true);
      handle.Arnoldi_view = Scalar3DViewType("", Nsolves, n_iterations, Nloc + n_iterations + 3);
      // solve
      double time = Functor_TestBatchedTeamVectorGMRES<exec_space, AMatrixValueView, IntView, XYType, KrylovHandleType>(batch_values, inv_diag, rowOffsets, colIndices, batch_x, batch_b, Nsolves_team, team_size, vector_length, n_iterations, tol, ortho_strategy, 0, handle)
                      .run(pc);
      Kokkos::fence();
      // get data back
      Kokkos::parallel_for(
        "map", Kokkos::TeamPolicy<>(Nsolves /* /batch_sz */, team_size, PCBJKOKKOS_VEC_SIZE), KOKKOS_LAMBDA(const team_member team) {
          const int blkID = team.league_rank(), start = d_bid_eqOffset[blkID], end = d_bid_eqOffset[blkID + 1]; // 0
          // map x into Plex/PETSc
          Kokkos::parallel_for(Kokkos::TeamVectorRange(team, start, end), [=](int rowb) {
            int rowa        = d_isicol[rowb];
            glb_xdata[rowa] = batch_x(blkID, rowb - start);
          });
        });
      // output assume species major - clone from Kokkos solvers
  #if PCBJKOKKOS_VERBOSE_LEVEL >= 3
    #if PCBJKOKKOS_VERBOSE_LEVEL >= 4
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "Iterations\n"));
    #else
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "max iterations per species (gmres) :"));
    #endif
      for (PetscInt dmIdx = 0, s = 0, head = 0; dmIdx < jac->num_dms; dmIdx += batch_sz) {
        for (PetscInt f = 0, idx = head; f < jac->dm_Nf[dmIdx]; f++, s++, idx++) {
    #if PCBJKOKKOS_VERBOSE_LEVEL >= 4
          PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "%2D:", s));
          for (int bid = 0; bid < batch_sz; bid++) PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "%3D ", handle.get_iteration_host(idx + bid * jac->dm_Nf[dmIdx])));
          PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "\n"));
    #else
          int count = 0, ii;
          for (int bid = 0; bid < batch_sz; bid++) {
            if ((ii = handle.get_iteration_host(idx + bid * jac->dm_Nf[dmIdx])) > count) count = ii;
          }
          PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "%3d", count));
    #endif
        }
        head += batch_sz * jac->dm_Nf[dmIdx];
      }
    #if PCBJKOKKOS_VERBOSE_LEVEL == 3
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "\n"));
    #endif
  #endif
      // return error code, get max it
      PetscInt count = 0, mbid = 0;
      if (handle.is_converged_host()) {
        pcreason = PC_NOERROR;
        if (!jac->max_nits) {
          for (int blkID = 0; blkID < nBlk; blkID++) {
            if (handle.get_iteration_host(blkID) > jac->max_nits) {
              jac->max_nits = handle.get_iteration_host(blkID);
              mbid          = blkID;
            }
          }
        }
      } else {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "There is at least one system that did not converge."));
        pcreason = PC_SUBPC_ERROR;
      }
      // output - assume species major order
      for (int blkID = 0; blkID < nBlk; blkID++) {
        if (jac->reason) { // -pc_bjkokkos_ksp_converged_reason
          if (jac->batch_target == blkID) {
            if (batch_sz != 1)
              PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "    Linear solve %s in %d iterations, batch %" PetscInt_FMT ", species %" PetscInt_FMT "\n", handle.is_converged_host(blkID) ? "converged" : "diverged", handle.get_iteration_host(blkID), blkID % batch_sz, blkID / batch_sz));
            else PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "    Linear solve %s in %d iterations, block %" PetscInt_FMT "\n", handle.is_converged_host(blkID) ? "converged" : "diverged", handle.get_iteration_host(blkID), blkID));
          } else if (jac->batch_target == -1 && handle.get_iteration_host(blkID) >= count) {
            jac->max_nits = count = handle.get_iteration_host(blkID);
            mbid                  = blkID;
          }
          if (!handle.is_converged_host(blkID)) PetscCall(PetscPrintf(PETSC_COMM_SELF, "ERROR species %d, batch %d did not converge with %d iterations\n", (int)(blkID / batch_sz), (int)blkID % batch_sz, handle.get_iteration_host(blkID)));
        }
      }
      if (jac->batch_target == -1 && jac->reason) {
        if (batch_sz != 1)
          PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "    Linear solve %s in %d iteration, batch %" PetscInt_FMT ", specie %" PetscInt_FMT "\n", handle.is_converged_host(mbid) ? "converged" : "diverged", jac->max_nits, mbid % batch_sz, mbid / batch_sz));
        else PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "    Linear solve %s in %d iteration, block %" PetscInt_FMT "\n", handle.is_converged_host(mbid) ? "converged" : "diverged", jac->max_nits, mbid));
      }
  #if defined(PETSC_HAVE_CUDA)
      nvtxRangePop();
  #endif
#else
      SETERRQ(PetscObjectComm((PetscObject)A), PETSC_ERR_USER, "batch GMRES not supported");
#endif
    } else { // Kokkos Krylov
      using scr_mem_t    = Kokkos::DefaultExecutionSpace::scratch_memory_space;
      using vect2D_scr_t = Kokkos::View<PetscScalar **, Kokkos::LayoutLeft, scr_mem_t>;
      Kokkos::View<Batch_MetaData *, Kokkos::DefaultExecutionSpace> d_metadata("solver meta data", nBlk);
      int                                                           stride_shared, stride_global, global_buff_words;
      d_bid_eqOffset = jac->d_bid_eqOffset_k->data();
      // solve each block independently
      int scr_bytes_team_shared = 0, nShareVec = 0, nGlobBVec = 0;
      if (jac->const_block_size) { // use shared memory for work vectors only if constant block size - todo: test efficiency loss
        int         maximum_shared_mem_size = 64000;
        PetscDevice device;
        PetscCall(PetscDeviceGetDefault_Internal(&device));
        PetscCall(PetscDeviceGetAttribute(device, PETSC_DEVICE_ATTR_SIZE_T_SHARED_MEM_PER_BLOCK, &maximum_shared_mem_size));
        stride_shared = jac->const_block_size;                                                   // captured
        nShareVec     = maximum_shared_mem_size / (jac->const_block_size * sizeof(PetscScalar)); // integer floor, number of vectors that fit in shared
        if (nShareVec > nwork) nShareVec = nwork;
        else nGlobBVec = nwork - nShareVec;
        global_buff_words     = jac->n * nGlobBVec;
        scr_bytes_team_shared = jac->const_block_size * nShareVec * sizeof(PetscScalar);
      } else {
        scr_bytes_team_shared = 0;
        stride_shared         = 0;
        global_buff_words     = jac->n * nwork;
        nGlobBVec             = nwork; // not needed == fix
      }
      stride_global = jac->n; // captured
#if defined(PETSC_HAVE_CUDA)
      nvtxRangePushA("batch-kokkos-solve");
#endif
      Kokkos::View<PetscScalar *, Kokkos::DefaultExecutionSpace> d_work_vecs_k("workvectors", global_buff_words); // global work vectors
      PetscCall(PetscInfo(pc, "\tn = %d. %d shared bytes/team, %d global mem bytes, rtol=%e, num blocks %d, team_size=%d, %d vector threads, %d shared vectors, %d global vectors\n", (int)jac->n, scr_bytes_team_shared, global_buff_words, rtol, (int)nBlk, (int)team_size, PCBJKOKKOS_VEC_SIZE, nShareVec, nGlobBVec));
      PetscScalar *d_work_vecs = d_work_vecs_k.data();
      Kokkos::parallel_for(
        "Solve", Kokkos::TeamPolicy<Kokkos::LaunchBounds<256, 4>>(nBlk, team_size, PCBJKOKKOS_VEC_SIZE).set_scratch_size(PCBJKOKKOS_SHARED_LEVEL, Kokkos::PerTeam(scr_bytes_team_shared)), KOKKOS_LAMBDA(const team_member team) {
          const int    blkID = team.league_rank(), start = d_bid_eqOffset[blkID], end = d_bid_eqOffset[blkID + 1];
          vect2D_scr_t work_vecs_shared(team.team_scratch(PCBJKOKKOS_SHARED_LEVEL), end - start, nShareVec);
          PetscScalar *work_buff_shared = work_vecs_shared.data();
          PetscScalar *work_buff_global = &d_work_vecs[start]; // start inc'ed in
          bool         print            = monitor && (blkID == view_bid);
          switch (ksp_type_idx) {
          case BATCH_KSP_BICG_IDX:
            static_cast<void>(BJSolve_BICG(team, glb_Aai, glb_Aaj, glb_Aaa, d_isrow, d_isicol, work_buff_global, stride_global, nShareVec, work_buff_shared, stride_shared, rtol, atol, dtol, maxit, &d_metadata[blkID], start, end, glb_idiag, glb_bdata, glb_xdata, print));
            break;
          case BATCH_KSP_TFQMR_IDX:
            static_cast<void>(BJSolve_TFQMR(team, glb_Aai, glb_Aaj, glb_Aaa, d_isrow, d_isicol, work_buff_global, stride_global, nShareVec, work_buff_shared, stride_shared, rtol, atol, dtol, maxit, &d_metadata[blkID], start, end, glb_idiag, glb_bdata, glb_xdata, print));
            break;
          case BATCH_KSP_GMRES_IDX:
            //BJSolve_GMRES();
            break;
          default:
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
            printf("Unknown KSP type %d\n", ksp_type_idx);
#else
            /* void */;
#endif
          }
        });
      Kokkos::fence();
#if defined(PETSC_HAVE_CUDA)
      nvtxRangePop();
      nvtxRangePushA("Post-solve-metadata");
#endif
      auto h_metadata = Kokkos::create_mirror(Kokkos::HostSpace::memory_space(), d_metadata);
      Kokkos::deep_copy(h_metadata, d_metadata);
      if (jac->reason) { // -pc_bjkokkos_ksp_converged_reason
#if PCBJKOKKOS_VERBOSE_LEVEL >= 3
  #if PCBJKOKKOS_VERBOSE_LEVEL >= 4
        PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Iterations\n"));
  #endif
        // assume species major
  #if PCBJKOKKOS_VERBOSE_LEVEL < 4
        if (batch_sz != 1) PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "%s: max iterations per species:", ksp_type_idx == BATCH_KSP_BICG_IDX ? "bicg" : "tfqmr"));
        else PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "    Linear solve converged due to %s iterations ", ksp_type_idx == BATCH_KSP_BICG_IDX ? "bicg" : "tfqmr"));
  #endif
        for (PetscInt dmIdx = 0, s = 0, head = 0; dmIdx < jac->num_dms; dmIdx += batch_sz) {
          for (PetscInt f = 0, idx = head; f < jac->dm_Nf[dmIdx]; f++, s++, idx++) {
  #if PCBJKOKKOS_VERBOSE_LEVEL >= 4
            PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "%2" PetscInt_FMT ":", s));
            for (int bid = 0; bid < batch_sz; bid++) PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "%3" PetscInt_FMT " ", h_metadata[idx + bid * jac->dm_Nf[dmIdx]].its));
            PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "\n"));
  #else
            PetscInt count = 0;
            for (int bid = 0; bid < batch_sz; bid++) {
              if (h_metadata[idx + bid * jac->dm_Nf[dmIdx]].its > count) count = h_metadata[idx + bid * jac->dm_Nf[dmIdx]].its;
            }
            PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "%3" PetscInt_FMT " ", count));
  #endif
          }
          head += batch_sz * jac->dm_Nf[dmIdx];
        }
  #if PCBJKOKKOS_VERBOSE_LEVEL == 3
        PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "\n"));
  #endif
#endif
        PetscInt count = 0, mbid = 0;
        for (int blkID = 0; blkID < nBlk; blkID++) {
          PetscCall(PetscLogGpuFlops((PetscLogDouble)h_metadata[blkID].flops));
#if PCBJKOKKOS_VERBOSE_LEVEL < 3
          if (jac->batch_target == blkID) {
            if (batch_sz != 1)
              PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "    Linear solve converged due to %s iterations %d, batch %" PetscInt_FMT ", species %" PetscInt_FMT "\n", KSPConvergedReasons[h_metadata[blkID].reason], h_metadata[blkID].its, blkID % batch_sz, blkID / batch_sz));
            else PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "    Linear solve converged due to %s iterations %d, block %" PetscInt_FMT "\n", KSPConvergedReasons[h_metadata[blkID].reason], h_metadata[blkID].its, blkID));
          } else if (jac->batch_target == -1 && h_metadata[blkID].its >= count) {
            jac->max_nits = count = h_metadata[blkID].its;
            mbid                  = blkID;
          }
#endif
#if PCBJKOKKOS_VERBOSE_LEVEL > 0
          if (h_metadata[blkID].reason < 0) {
            PetscCall(PetscPrintf(PETSC_COMM_SELF, "ERROR reason=%s, its=%" PetscInt_FMT ". species %" PetscInt_FMT ", batch %" PetscInt_FMT "\n", KSPConvergedReasons[h_metadata[blkID].reason], h_metadata[blkID].its, blkID / batch_sz, blkID % batch_sz));
          }
#endif
        }
        if (jac->batch_target == -1) {
          if (batch_sz != 1)
            PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "    Linear solve converged due to %s iterations %d, batch %" PetscInt_FMT ", species %" PetscInt_FMT "\n", KSPConvergedReasons[h_metadata[mbid].reason], h_metadata[mbid].its, mbid % batch_sz, mbid / batch_sz));
          else PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "    Linear solve converged due to %s iterations %d, block %" PetscInt_FMT "\n", KSPConvergedReasons[h_metadata[mbid].reason], h_metadata[mbid].its, mbid));
        }
      }
      for (int blkID = 0; blkID < nBlk; blkID++) {
        PetscCall(PetscLogGpuFlops((PetscLogDouble)h_metadata[blkID].flops));
#if PCBJKOKKOS_VERBOSE_LEVEL > 0
        if (h_metadata[blkID].reason < 0) {
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "ERROR reason=%s, its=%" PetscInt_FMT ". species %" PetscInt_FMT ", batch %" PetscInt_FMT "\n", KSPConvergedReasons[h_metadata[blkID].reason], h_metadata[blkID].its, blkID / batch_sz, blkID % batch_sz));
        }
#endif
      }
      {
        int errsum;
        Kokkos::parallel_reduce(
          nBlk,
          KOKKOS_LAMBDA(const int idx, int &lsum) {
            if (d_metadata[idx].reason < 0) ++lsum;
          },
          errsum);
        pcreason = errsum ? PC_SUBPC_ERROR : PC_NOERROR;
        if (!errsum && !jac->max_nits) { // set max its to give back to top KSP
          for (int blkID = 0; blkID < nBlk; blkID++) {
            if (h_metadata[blkID].its > jac->max_nits) jac->max_nits = h_metadata[blkID].its;
          }
        } else if (errsum) {
          PetscCall(PetscPrintf(PETSC_COMM_SELF, "ERROR Kokkos batch solver did not converge in all solves\n"));
        }
      }
#if defined(PETSC_HAVE_CUDA)
      nvtxRangePop();
#endif
    } // end of Kokkos (not Kernels) solvers block
    PetscCall(VecRestoreArrayAndMemType(xout, &glb_xdata));
    PetscCall(VecRestoreArrayReadAndMemType(bvec, &glb_bdata));
    PetscCall(PCSetFailedReason(pc, pcreason));
    // map back to Plex space - not used
    if (plex_batch) {
      PetscCall(VecCopy(xout, bvec));
      PetscCall(VecScatterBegin(plex_batch, bvec, xout, INSERT_VALUES, SCATTER_REVERSE));
      PetscCall(VecScatterEnd(plex_batch, bvec, xout, INSERT_VALUES, SCATTER_REVERSE));
      PetscCall(VecDestroy(&bvec));
    }
  } // whole 'have aijkok' block
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetUp_BJKOKKOS(PC pc)
{
  PC_PCBJKOKKOS    *jac = (PC_PCBJKOKKOS *)pc->data;
  Mat               A = pc->pmat, Aseq = A; // use filtered block matrix, really "P"
  Mat_SeqAIJKokkos *aijkok;
  PetscBool         flg;

  PetscFunctionBegin;
  PetscCheck(!pc->useAmat, PetscObjectComm((PetscObject)pc), PETSC_ERR_SUP, "No support for using 'use_amat'");
  PetscCheck(A, PetscObjectComm((PetscObject)A), PETSC_ERR_USER, "No matrix - A is used above");
  PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &flg, MATSEQAIJKOKKOS, MATMPIAIJKOKKOS, MATAIJKOKKOS, ""));
  PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "must use '-[dm_]mat_type aijkokkos -[dm_]vec_type kokkos' for -pc_type bjkokkos");
  aijkok = static_cast<Mat_SeqAIJKokkos *>(Aseq->spptr);
  if (!aijkok) {
    PetscCall(MatMPIAIJGetSeqAIJ(A, &Aseq, NULL, NULL));
    aijkok = static_cast<Mat_SeqAIJKokkos *>(Aseq->spptr);
  }
  PetscCheck((aijkok), PetscObjectComm((PetscObject)A), PETSC_ERR_USER, "No aijkok");
  {
    PetscInt    Istart, Iend;
    PetscMPIInt rank;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));
    if (!jac->vec_diag) {
      Vec     *subX = NULL;
      DM       pack, *subDM = NULL;
      PetscInt nDMs, n, *block_sizes = NULL;
      IS       isrow, isicol;
      { // Permute the matrix to get a block diagonal system: d_isrow_k, d_isicol_k
        MatOrderingType rtype;
        const PetscInt *rowindices, *icolindices;
        rtype = MATORDERINGRCM;
        // get permutation. And invert. should we convert to local indices?
        PetscCall(MatGetOrdering(Aseq, rtype, &isrow, &isicol)); // only seems to work for seq matrix
        PetscCall(ISDestroy(&isrow));
        PetscCall(ISInvertPermutation(isicol, PETSC_DECIDE, &isrow)); // THIS IS BACKWARD -- isrow is inverse
        // if (rank==1) PetscCall(ISView(isicol, PETSC_VIEWER_STDOUT_SELF));
        if (1) {
          Mat mat_block_order; // debug
          PetscCall(ISShift(isicol, Istart, isicol));
          PetscCall(MatCreateSubMatrix(A, isicol, isicol, MAT_INITIAL_MATRIX, &mat_block_order));
          PetscCall(ISShift(isicol, -Istart, isicol));
          PetscCall(MatViewFromOptions(mat_block_order, NULL, "-ksp_batch_reorder_view"));
          PetscCall(MatDestroy(&mat_block_order));
        }
        PetscCall(ISGetIndices(isrow, &rowindices)); // local idx
        PetscCall(ISGetIndices(isicol, &icolindices));
        const Kokkos::View<PetscInt *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_isrow_k((PetscInt *)rowindices, A->rmap->n);
        const Kokkos::View<PetscInt *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_isicol_k((PetscInt *)icolindices, A->rmap->n);
        jac->d_isrow_k  = new Kokkos::View<PetscInt *>(Kokkos::create_mirror(DefaultMemorySpace(), h_isrow_k));
        jac->d_isicol_k = new Kokkos::View<PetscInt *>(Kokkos::create_mirror(DefaultMemorySpace(), h_isicol_k));
        Kokkos::deep_copy(*jac->d_isrow_k, h_isrow_k);
        Kokkos::deep_copy(*jac->d_isicol_k, h_isicol_k);
        PetscCall(ISRestoreIndices(isrow, &rowindices));
        PetscCall(ISRestoreIndices(isicol, &icolindices));
        // if (rank==1) PetscCall(ISView(isicol, PETSC_VIEWER_STDOUT_SELF));
      }
      // get block sizes & allocate vec_diag
      PetscCall(PCGetDM(pc, &pack));
      if (pack) {
        PetscCall(PetscObjectTypeCompare((PetscObject)pack, DMCOMPOSITE, &flg));
        if (flg) {
          PetscCall(DMCompositeGetNumberDM(pack, &nDMs));
          PetscCall(DMCreateGlobalVector(pack, &jac->vec_diag));
        } else pack = NULL; // flag for no DM
      }
      if (!jac->vec_diag) { // get 'nDMs' and sizes 'block_sizes' w/o DMComposite. User could provide ISs (todo)
        PetscInt        bsrt, bend, ncols, ntot = 0;
        const PetscInt *colsA, nloc = Iend - Istart;
        const PetscInt *rowindices, *icolindices;
        PetscCall(PetscMalloc1(nloc, &block_sizes)); // very inefficient, to big
        PetscCall(ISGetIndices(isrow, &rowindices));
        PetscCall(ISGetIndices(isicol, &icolindices));
        nDMs = 0;
        bsrt = 0;
        bend = 1;
        for (PetscInt row_B = 0; row_B < nloc; row_B++) { // for all rows in block diagonal space
          PetscInt rowA = icolindices[row_B], minj = PETSC_MAX_INT, maxj = 0;
          //PetscCall(PetscPrintf(PETSC_COMM_SELF, "\t[%d] rowA = %d\n",rank,rowA));
          PetscCall(MatGetRow(Aseq, rowA, &ncols, &colsA, NULL)); // not sorted in permutation
          PetscCheck(ncols, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "Empty row not supported: %" PetscInt_FMT "\n", row_B);
          for (PetscInt colj = 0; colj < ncols; colj++) {
            PetscInt colB = rowindices[colsA[colj]]; // use local idx
            //PetscCall(PetscPrintf(PETSC_COMM_SELF, "\t\t[%d] colB = %d\n",rank,colB));
            PetscCheck(colB >= 0 && colB < nloc, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "colB < 0: %" PetscInt_FMT "\n", colB);
            if (colB > maxj) maxj = colB;
            if (colB < minj) minj = colB;
          }
          PetscCall(MatRestoreRow(Aseq, rowA, &ncols, &colsA, NULL));
          if (minj >= bend) { // first column is > max of last block -- new block or last block
            //PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A), "\t\t finish block %d, N loc = %d (%d,%d)\n", nDMs+1, bend - bsrt,bsrt,bend));
            block_sizes[nDMs] = bend - bsrt;
            ntot += block_sizes[nDMs];
            PetscCheck(minj == bend, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "minj != bend: %" PetscInt_FMT " != %" PetscInt_FMT "\n", minj, bend);
            bsrt = bend;
            bend++; // start with size 1 in new block
            nDMs++;
          }
          if (maxj + 1 > bend) bend = maxj + 1;
          PetscCheck(minj >= bsrt || row_B == Iend - 1, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "%" PetscInt_FMT ") minj < bsrt: %" PetscInt_FMT " != %" PetscInt_FMT "\n", rowA, minj, bsrt);
          //PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] %d) row %d.%d) cols %d : %d ; bsrt = %d, bend = %d\n",rank,row_B,nDMs,rowA,minj,maxj,bsrt,bend));
        }
        // do last block
        //PetscCall(PetscPrintf(PETSC_COMM_SELF, "\t\t\t [%d] finish block %d, N loc = %d (%d,%d)\n", rank, nDMs+1, bend - bsrt,bsrt,bend));
        block_sizes[nDMs] = bend - bsrt;
        ntot += block_sizes[nDMs];
        nDMs++;
        // cleanup
        PetscCheck(ntot == nloc, PetscObjectComm((PetscObject)pc), PETSC_ERR_ARG_WRONG, "n total != n local: %" PetscInt_FMT " != %" PetscInt_FMT "\n", ntot, nloc);
        PetscCall(ISRestoreIndices(isrow, &rowindices));
        PetscCall(ISRestoreIndices(isicol, &icolindices));
        PetscCall(PetscRealloc(sizeof(PetscInt) * nDMs, &block_sizes));
        PetscCall(MatCreateVecs(A, &jac->vec_diag, NULL));
        PetscCall(PetscInfo(pc, "Setup Matrix based meta data (not DMComposite not attached to PC) %" PetscInt_FMT " sub domains\n", nDMs));
      }
      PetscCall(ISDestroy(&isrow));
      PetscCall(ISDestroy(&isicol));
      jac->num_dms = nDMs;
      PetscCall(VecGetLocalSize(jac->vec_diag, &n));
      jac->n         = n;
      jac->d_idiag_k = new Kokkos::View<PetscScalar *, Kokkos::LayoutRight>("idiag", n);
      // options
      PetscCall(PCBJKOKKOSCreateKSP_BJKOKKOS(pc));
      PetscCall(KSPSetFromOptions(jac->ksp));
      PetscCall(PetscObjectTypeCompareAny((PetscObject)jac->ksp, &flg, KSPBICG, ""));
      if (flg) {
        jac->ksp_type_idx = BATCH_KSP_BICG_IDX;
        jac->nwork        = 7;
      } else {
        PetscCall(PetscObjectTypeCompareAny((PetscObject)jac->ksp, &flg, KSPTFQMR, ""));
        if (flg) {
          jac->ksp_type_idx = BATCH_KSP_TFQMR_IDX;
          jac->nwork        = 10;
        } else {
          PetscCall(PetscObjectTypeCompareAny((PetscObject)jac->ksp, &flg, KSPGMRES, ""));
          PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONG, "Unsupported batch ksp type");
          jac->ksp_type_idx = BATCH_KSP_GMRES_IDX;
          jac->nwork        = 0;
        }
      }
      PetscOptionsBegin(PetscObjectComm((PetscObject)jac->ksp), ((PetscObject)jac->ksp)->prefix, "Options for Kokkos batch solver", "none");
      PetscCall(PetscOptionsBool("-ksp_converged_reason", "", "bjkokkos.kokkos.cxx.c", jac->reason, &jac->reason, NULL));
      PetscCall(PetscOptionsBool("-ksp_monitor", "", "bjkokkos.kokkos.cxx.c", jac->monitor, &jac->monitor, NULL));
      PetscCall(PetscOptionsInt("-ksp_batch_target", "", "bjkokkos.kokkos.cxx.c", jac->batch_target, &jac->batch_target, NULL));
      PetscCall(PetscOptionsInt("-ksp_batch_nsolves_team", "", "bjkokkos.kokkos.cxx.c", jac->nsolves_team, &jac->nsolves_team, NULL));
      PetscCheck(jac->batch_target < jac->num_dms, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "-ksp_batch_target (%" PetscInt_FMT ") >= number of DMs (%" PetscInt_FMT ")", jac->batch_target, jac->num_dms);
      PetscOptionsEnd();
      // get blocks - jac->d_bid_eqOffset_k
      if (pack) {
        PetscCall(PetscMalloc(sizeof(*subX) * nDMs, &subX));
        PetscCall(PetscMalloc(sizeof(*subDM) * nDMs, &subDM));
      }
      PetscCall(PetscMalloc(sizeof(*jac->dm_Nf) * nDMs, &jac->dm_Nf));
      PetscCall(PetscInfo(pc, "Have %" PetscInt_FMT " blocks, n=%" PetscInt_FMT " rtol=%g type = %s\n", nDMs, n, (double)jac->ksp->rtol, ((PetscObject)jac->ksp)->type_name));
      if (pack) PetscCall(DMCompositeGetEntriesArray(pack, subDM));
      jac->nBlocks = 0;
      for (PetscInt ii = 0; ii < nDMs; ii++) {
        PetscInt Nf;
        if (subDM) {
          DM           dm = subDM[ii];
          PetscSection section;
          PetscCall(DMGetLocalSection(dm, &section));
          PetscCall(PetscSectionGetNumFields(section, &Nf));
        } else Nf = 1;
        jac->nBlocks += Nf;
#if PCBJKOKKOS_VERBOSE_LEVEL <= 2
        if (ii == 0) PetscCall(PetscInfo(pc, "%" PetscInt_FMT ") %" PetscInt_FMT " blocks (%" PetscInt_FMT " total)\n", ii, Nf, jac->nBlocks));
#else
        PetscCall(PetscInfo(pc, "%" PetscInt_FMT ") %" PetscInt_FMT " blocks (%" PetscInt_FMT " total)\n", ii, Nf, jac->nBlocks));
#endif
        jac->dm_Nf[ii] = Nf;
      }
      { // d_bid_eqOffset_k
        Kokkos::View<PetscInt *, Kokkos::LayoutRight, Kokkos::HostSpace> h_block_offsets("block_offsets", jac->nBlocks + 1);
        if (pack) PetscCall(DMCompositeGetAccessArray(pack, jac->vec_diag, nDMs, NULL, subX));
        h_block_offsets[0]    = 0;
        jac->const_block_size = -1;
        for (PetscInt ii = 0, idx = 0; ii < nDMs; ii++) {
          PetscInt nloc, nblk;
          if (pack) PetscCall(VecGetSize(subX[ii], &nloc));
          else nloc = block_sizes[ii];
          nblk = nloc / jac->dm_Nf[ii];
          PetscCheck(nloc % jac->dm_Nf[ii] == 0, PetscObjectComm((PetscObject)pc), PETSC_ERR_USER, "nloc%%jac->dm_Nf[ii] (%" PetscInt_FMT ") != 0 DMs", nloc % jac->dm_Nf[ii]);
          for (PetscInt jj = 0; jj < jac->dm_Nf[ii]; jj++, idx++) {
            h_block_offsets[idx + 1] = h_block_offsets[idx] + nblk;
#if PCBJKOKKOS_VERBOSE_LEVEL <= 2
            if (idx == 0) PetscCall(PetscInfo(pc, "\t%" PetscInt_FMT ") Add block with %" PetscInt_FMT " equations of %" PetscInt_FMT "\n", idx + 1, nblk, jac->nBlocks));
#else
            PetscCall(PetscInfo(pc, "\t%" PetscInt_FMT ") Add block with %" PetscInt_FMT " equations of %" PetscInt_FMT "\n", idx + 1, nblk, jac->nBlocks));
#endif
            if (jac->const_block_size == -1) jac->const_block_size = nblk;
            else if (jac->const_block_size > 0 && jac->const_block_size != nblk) jac->const_block_size = 0;
          }
        }
        if (pack) {
          PetscCall(DMCompositeRestoreAccessArray(pack, jac->vec_diag, jac->nBlocks, NULL, subX));
          PetscCall(PetscFree(subX));
          PetscCall(PetscFree(subDM));
        }
        jac->d_bid_eqOffset_k = new Kokkos::View<PetscInt *, Kokkos::LayoutRight>(Kokkos::create_mirror(Kokkos::DefaultExecutionSpace::memory_space(), h_block_offsets));
        Kokkos::deep_copy(*jac->d_bid_eqOffset_k, h_block_offsets);
      }
      if (!pack) PetscCall(PetscFree(block_sizes));
    }
    { // get jac->d_idiag_k (PC setup),
      const PetscInt    *d_ai = aijkok->i_device_data(), *d_aj = aijkok->j_device_data();
      const PetscScalar *d_aa = aijkok->a_device_data();
      const PetscInt     conc = Kokkos::DefaultExecutionSpace().concurrency(), openmp = !!(conc < 1000), team_size = (openmp == 0 && PCBJKOKKOS_VEC_SIZE != 1) ? PCBJKOKKOS_TEAM_SIZE : 1;
      const PetscInt    *d_bid_eqOffset = jac->d_bid_eqOffset_k->data(), *r = jac->d_isrow_k->data(), *ic = jac->d_isicol_k->data();
      PetscScalar       *d_idiag = jac->d_idiag_k->data();
      Kokkos::parallel_for(
        "Diag", Kokkos::TeamPolicy<>(jac->nBlocks, team_size, PCBJKOKKOS_VEC_SIZE), KOKKOS_LAMBDA(const team_member team) {
          const PetscInt blkID = team.league_rank();
          Kokkos::parallel_for(Kokkos::TeamThreadRange(team, d_bid_eqOffset[blkID], d_bid_eqOffset[blkID + 1]), [=](const int rowb) {
            const PetscInt     rowa = ic[rowb], ai = d_ai[rowa], *aj = d_aj + ai; // grab original data
            const PetscScalar *aa   = d_aa + ai;
            const PetscInt     nrow = d_ai[rowa + 1] - ai;
            int                found;
            Kokkos::parallel_reduce(
              Kokkos::ThreadVectorRange(team, nrow),
              [=](const int &j, int &count) {
                const PetscInt colb = r[aj[j]];
                if (colb == rowb) {
                  d_idiag[rowb] = 1. / aa[j];
                  count++;
                }
              },
              found);
#if defined(PETSC_USE_DEBUG) && !defined(PETSC_HAVE_SYCL)
            if (found != 1) Kokkos::single(Kokkos::PerThread(team), [=]() { printf("ERRORrow %d) found = %d\n", rowb, found); });
#endif
          });
        });
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Default destroy, if it has never been setup */
static PetscErrorCode PCReset_BJKOKKOS(PC pc)
{
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS *)pc->data;

  PetscFunctionBegin;
  PetscCall(KSPDestroy(&jac->ksp));
  PetscCall(VecDestroy(&jac->vec_diag));
  if (jac->d_bid_eqOffset_k) delete jac->d_bid_eqOffset_k;
  if (jac->d_idiag_k) delete jac->d_idiag_k;
  if (jac->d_isrow_k) delete jac->d_isrow_k;
  if (jac->d_isicol_k) delete jac->d_isicol_k;
  jac->d_bid_eqOffset_k = NULL;
  jac->d_idiag_k        = NULL;
  jac->d_isrow_k        = NULL;
  jac->d_isicol_k       = NULL;
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCBJKOKKOSGetKSP_C", NULL)); // not published now (causes configure errors)
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCBJKOKKOSSetKSP_C", NULL));
  PetscCall(PetscFree(jac->dm_Nf));
  jac->dm_Nf = NULL;
  if (jac->rowOffsets) delete jac->rowOffsets;
  if (jac->colIndices) delete jac->colIndices;
  if (jac->batch_b) delete jac->batch_b;
  if (jac->batch_x) delete jac->batch_x;
  if (jac->batch_values) delete jac->batch_values;
  jac->rowOffsets   = NULL;
  jac->colIndices   = NULL;
  jac->batch_b      = NULL;
  jac->batch_x      = NULL;
  jac->batch_values = NULL;

  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCDestroy_BJKOKKOS(PC pc)
{
  PetscFunctionBegin;
  PetscCall(PCReset_BJKOKKOS(pc));
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCView_BJKOKKOS(PC pc, PetscViewer viewer)
{
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS *)pc->data;
  PetscBool      iascii;

  PetscFunctionBegin;
  if (!jac->ksp) PetscCall(PCBJKOKKOSCreateKSP_BJKOKKOS(pc));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  Batched device linear solver: Krylov (KSP) method with Jacobi preconditioning\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "\t\tnwork = %" PetscInt_FMT ", rel tol = %e, abs tol = %e, div tol = %e, max it =%" PetscInt_FMT ", type = %s\n", jac->nwork, jac->ksp->rtol, jac->ksp->abstol, jac->ksp->divtol, jac->ksp->max_it,
                                     ((PetscObject)jac->ksp)->type_name));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCSetFromOptions_BJKOKKOS(PC pc, PetscOptionItems *PetscOptionsObject)
{
  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "PC BJKOKKOS options");
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCBJKOKKOSSetKSP_BJKOKKOS(PC pc, KSP ksp)
{
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS *)pc->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectReference((PetscObject)ksp));
  PetscCall(KSPDestroy(&jac->ksp));
  jac->ksp = ksp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PCBJKOKKOSSetKSP - Sets the `KSP` context for `PCBJKOKKOS`

   Collective

   Input Parameters:
+  pc - the `PCBJKOKKOS` preconditioner context
-  ksp - the `KSP` solver

   Level: advanced

   Notes:
   The `PC` and the `KSP` must have the same communicator

   If the `PC` is not `PCBJKOKKOS` this function returns without doing anything

,seealso: `PCBJKOKKOSGetKSP()`, `PCBJKOKKOS`
@*/
PetscErrorCode PCBJKOKKOSSetKSP(PC pc, KSP ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 2);
  PetscCheckSameComm(pc, 1, ksp, 2);
  PetscTryMethod(pc, "PCBJKOKKOSSetKSP_C", (PC, KSP), (pc, ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode PCBJKOKKOSGetKSP_BJKOKKOS(PC pc, KSP *ksp)
{
  PC_PCBJKOKKOS *jac = (PC_PCBJKOKKOS *)pc->data;

  PetscFunctionBegin;
  if (!jac->ksp) PetscCall(PCBJKOKKOSCreateKSP_BJKOKKOS(pc));
  *ksp = jac->ksp;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   PCBJKOKKOSGetKSP - Gets the `KSP` context for the `PCBJKOKKOS` preconditioner

   Not Collective but `KSP` returned is parallel if `PC` was parallel

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  ksp - the `KSP` solver

   Level: advanced

   Notes:
   You must call `KSPSetUp()` before calling `PCBJKOKKOSGetKSP()`.

   If the `PC` is not a `PCBJKOKKOS` object it raises an error

.seealso: `PCBJKOKKOS`, `PCBJKOKKOSSetKSP()`
@*/
PetscErrorCode PCBJKOKKOSGetKSP(PC pc, KSP *ksp)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc, PC_CLASSID, 1);
  PetscValidPointer(ksp, 2);
  PetscUseMethod(pc, "PCBJKOKKOSGetKSP_C", (PC, KSP *), (pc, ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     PCBJKOKKOS -  Defines a preconditioner that applies a Krylov solver and preconditioner to the blocks in a `MATSEQAIJ` matrix on the GPU using Kokkos

   Options Database Key:
.     -pc_bjkokkos_ - options prefix for its `KSP` options

   Level: intermediate

   Note:
    For use with -ksp_type preonly to bypass any computation on the CPU

   Developer Notes:
   The documentation is incomplete. Is this a block Jacobi preconditioner?

   Why does it have its own `KSP`? Where is the `KSP` run if used with -ksp_type preonly?

.seealso: `PCCreate()`, `PCSetType()`, `PCType`, `PC`, `PCBJACOBI`,
          `PCSHELL`, `PCCOMPOSITE`, `PCSetUseAmat()`, `PCBJKOKKOSGetKSP()`
M*/

PETSC_EXTERN PetscErrorCode PCCreate_BJKOKKOS(PC pc)
{
  PC_PCBJKOKKOS *jac;

  PetscFunctionBegin;
  PetscCall(PetscNew(&jac));
  pc->data = (void *)jac;

  jac->ksp              = NULL;
  jac->vec_diag         = NULL;
  jac->d_bid_eqOffset_k = NULL;
  jac->d_idiag_k        = NULL;
  jac->d_isrow_k        = NULL;
  jac->d_isicol_k       = NULL;
  jac->nBlocks          = 1;
  jac->max_nits         = 0;

  PetscCall(PetscMemzero(pc->ops, sizeof(struct _PCOps)));
  pc->ops->apply          = PCApply_BJKOKKOS;
  pc->ops->applytranspose = NULL;
  pc->ops->setup          = PCSetUp_BJKOKKOS;
  pc->ops->reset          = PCReset_BJKOKKOS;
  pc->ops->destroy        = PCDestroy_BJKOKKOS;
  pc->ops->setfromoptions = PCSetFromOptions_BJKOKKOS;
  pc->ops->view           = PCView_BJKOKKOS;

  jac->rowOffsets   = NULL;
  jac->colIndices   = NULL;
  jac->batch_b      = NULL;
  jac->batch_x      = NULL;
  jac->batch_values = NULL;

  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCBJKOKKOSGetKSP_C", PCBJKOKKOSGetKSP_BJKOKKOS));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc, "PCBJKOKKOSSetKSP_C", PCBJKOKKOSSetKSP_BJKOKKOS));
  PetscFunctionReturn(PETSC_SUCCESS);
}
