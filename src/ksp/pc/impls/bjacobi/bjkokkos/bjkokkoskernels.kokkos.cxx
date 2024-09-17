#include <petsc/private/pcbjkokkosimpl.h>

#if defined(PETSC_HAVE_KOKKOS_KERNELS_BATCH)
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
  // #include <KokkosBatched_Gemv_Serial_Impl.hpp>
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
    //typedef typename ValuesViewType::value_type value_type;
    std::string   name("KokkosBatched::Test::TeamVectorGMRES");
    Kokkos::Timer timer;
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

    using ViewType2D    = Kokkos::View<ScalarType **, Layout, EXSP>;
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
    PetscCall(PetscInfo(pc, "%d scratch memory(0) = %d + %d + %d bytes_diag=%d; %d scratch memory(1); %d maximum_iterations\n", (int)bytes_tmp, 2 * (int)bytes_2D_1, 2 * (int)bytes_1D, (int)bytes_2D_2, (int)bytes_diag, (int)(bytes_row_ptr + bytes_col_idc + bytes_diag), (int)maximum_iteration));
    exec_space().fence();
    timer.reset();
    Kokkos::parallel_for(name.c_str(), policy, *this);
    exec_space().fence();
    double sec = timer.seconds();

    return sec;
  }
};

PETSC_INTERN PetscErrorCode PCApply_BJKOKKOSKERNELS(PC pc, const PetscScalar *glb_bdata, PetscScalar *glb_xdata, const PetscInt *glb_Aai, const PetscInt *glb_Aaj, const PetscScalar *glb_Aaa, const PetscInt team_size, MatInfo info, const PetscInt batch_sz, PCFailedReason *pcreason)
{
  PC_PCBJKOKKOS     *jac   = (PC_PCBJKOKKOS *)pc->data;
  Mat                A     = pc->pmat;
  const PetscInt     maxit = jac->ksp->max_it, nBlk = jac->nBlocks;
  const int          Nsolves      = nBlk;
  int                Nsolves_team = jac->nsolves_team, fill_idx = 0;
  int                Nloc           = jac->const_block_size;       // same grids
  const int          nnz            = (int)info.nz_used / Nsolves; // fix for variable grid size
  PetscReal          rtol           = jac->ksp->rtol;
  const PetscScalar *glb_idiag      = jac->d_idiag_k->data();
  const PetscInt    *d_bid_eqOffset = jac->d_bid_eqOffset_k->data();
  const PetscInt    *d_isicol = jac->d_isicol_k->data(), *d_isrow = jac->d_isrow_k->data();

  PetscFunctionBegin;
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
  PetscCall(PetscInfo(pc, "\tjac->n = %d, Nloc = %d, Nsolves = %d, nnz = %d, Nsolves_team = %d, league size = %d, maxit = %d\n", (int)jac->n, Nloc, Nsolves, nnz, Nsolves_team, Nsolves / Nsolves_team, (int)maxit));
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
        for (int i = 0; i < Nloc; ++i) rowOffsets(graphID, i + 1) += rowOffsets(graphID, i);
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
  using ScalarType    = typename AMatrixValueView::non_const_value_type;
  using MagnitudeType = typename Kokkos::Details::ArithTraits<ScalarType>::mag_type;
  //using NormViewType              = Kokkos::View<MagnitudeType *, layout, exec_space>;
  using Norm2DViewType   = Kokkos::View<MagnitudeType **, layout, exec_space>;
  using Scalar3DViewType = Kokkos::View<ScalarType ***, layout, exec_space>;
  using IntViewType      = Kokkos::View<int *, layout, exec_space>;
  using KrylovHandleType = KokkosBatched::KrylovHandle<Norm2DViewType, IntViewType, Scalar3DViewType>;
  const int n_iterations = maxit;
  //const int        team_size      = -1;
  const int        vector_length  = -1;
  const double     tol            = rtol;
  const int        ortho_strategy = 0;
  KrylovHandleType handle(Nsolves, Nsolves_team, n_iterations, true);
  handle.Arnoldi_view = Scalar3DViewType("", Nsolves, n_iterations, Nloc + n_iterations + 3);
  // solve
  Functor_TestBatchedTeamVectorGMRES<exec_space, AMatrixValueView, IntView, XYType, KrylovHandleType>(batch_values, inv_diag, rowOffsets, colIndices, batch_x, batch_b, Nsolves_team, -1, vector_length, n_iterations, tol, ortho_strategy, 0, handle).run(pc);
  Kokkos::fence();
  // get data back
  Kokkos::parallel_for(
    "map", Kokkos::TeamPolicy<>(Nsolves /* /batch_sz */, -1, PCBJKOKKOS_VEC_SIZE), KOKKOS_LAMBDA(const team_member team) {
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
    *pcreason = PC_NOERROR;
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
    *pcreason = PC_SUBPC_ERROR;
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

  return PETSC_SUCCESS;
}
#endif
