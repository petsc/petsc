#include <petscvec_kokkos.hpp>
#include <petscpkg_version.h>
#include <petscsf.h>
#include <petsc/private/sfimpl.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <../src/mat/impls/aij/mpi/kokkos/mpiaijkok.hpp>
#include <KokkosSparse_spadd.hpp>
#include <KokkosSparse_spgemm.hpp>

PetscErrorCode MatAssemblyEnd_MPIAIJKokkos(Mat A, MatAssemblyType mode)
{
  Mat_SeqAIJKokkos *aijkok;
  Mat_MPIAIJ       *mpiaij = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(MatAssemblyEnd_MPIAIJ(A, mode));
  /* E.g., MatCreateSubMatrix() calls MatCreateMPIAIJWithSeqAIJ(comm,A,B,..), which creates Bnew of SEQAIJ and destroys B of SEQAIJKOKKOS.
     Thus we finalize A/B/lvec's type in MatAssemblyEnd() to handle various cases.
   */
  if (mode == MAT_FINAL_ASSEMBLY) {
    PetscCall(MatSetType(mpiaij->A, MATSEQAIJKOKKOS));
    PetscCall(MatSetType(mpiaij->B, MATSEQAIJKOKKOS));
    PetscCall(VecSetType(mpiaij->lvec, VECSEQKOKKOS));
  }
  aijkok = static_cast<Mat_SeqAIJKokkos *>(((Mat_MPIAIJ *)A->data)->A->spptr); /* Access spptr after MatAssemblyEnd_MPIAIJ(), which might have deleted old spptr */
  if (aijkok && aijkok->device_mat_d.data()) {
    A->offloadmask = PETSC_OFFLOAD_GPU; // in GPU mode, no going back. MatSetValues checks this
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMPIAIJSetPreallocation_MPIAIJKokkos(Mat mat, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[])
{
  Mat_MPIAIJ *mpiaij = (Mat_MPIAIJ *)mat->data;

  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(mat->rmap));
  PetscCall(PetscLayoutSetUp(mat->cmap));
#if defined(PETSC_USE_DEBUG)
  if (d_nnz) {
    PetscInt i;
    for (i = 0; i < mat->rmap->n; i++) PetscCheck(d_nnz[i] >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "d_nnz cannot be less than 0: local row %" PetscInt_FMT " value %" PetscInt_FMT, i, d_nnz[i]);
  }
  if (o_nnz) {
    PetscInt i;
    for (i = 0; i < mat->rmap->n; i++) PetscCheck(o_nnz[i] >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "o_nnz cannot be less than 0: local row %" PetscInt_FMT " value %" PetscInt_FMT, i, o_nnz[i]);
  }
#endif
#if defined(PETSC_USE_CTABLE)
  PetscCall(PetscHMapIDestroy(&mpiaij->colmap));
#else
  PetscCall(PetscFree(mpiaij->colmap));
#endif
  PetscCall(PetscFree(mpiaij->garray));
  PetscCall(VecDestroy(&mpiaij->lvec));
  PetscCall(VecScatterDestroy(&mpiaij->Mvctx));
  /* Because the B will have been resized we simply destroy it and create a new one each time */
  PetscCall(MatDestroy(&mpiaij->B));

  if (!mpiaij->A) {
    PetscCall(MatCreate(PETSC_COMM_SELF, &mpiaij->A));
    PetscCall(MatSetSizes(mpiaij->A, mat->rmap->n, mat->cmap->n, mat->rmap->n, mat->cmap->n));
  }
  if (!mpiaij->B) {
    PetscMPIInt size;
    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)mat), &size));
    PetscCall(MatCreate(PETSC_COMM_SELF, &mpiaij->B));
    PetscCall(MatSetSizes(mpiaij->B, mat->rmap->n, size > 1 ? mat->cmap->N : 0, mat->rmap->n, size > 1 ? mat->cmap->N : 0));
  }
  PetscCall(MatSetType(mpiaij->A, MATSEQAIJKOKKOS));
  PetscCall(MatSetType(mpiaij->B, MATSEQAIJKOKKOS));
  PetscCall(MatSeqAIJSetPreallocation(mpiaij->A, d_nz, d_nnz));
  PetscCall(MatSeqAIJSetPreallocation(mpiaij->B, o_nz, o_nnz));
  mat->preallocated = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMult_MPIAIJKokkos(Mat mat, Vec xx, Vec yy)
{
  Mat_MPIAIJ *mpiaij = (Mat_MPIAIJ *)mat->data;
  PetscInt    nt;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(xx, &nt));
  PetscCheck(nt == mat->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible partition of mat (%" PetscInt_FMT ") and xx (%" PetscInt_FMT ")", mat->cmap->n, nt);
  PetscCall(VecScatterBegin(mpiaij->Mvctx, xx, mpiaij->lvec, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall((*mpiaij->A->ops->mult)(mpiaij->A, xx, yy));
  PetscCall(VecScatterEnd(mpiaij->Mvctx, xx, mpiaij->lvec, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall((*mpiaij->B->ops->multadd)(mpiaij->B, mpiaij->lvec, yy, yy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMultAdd_MPIAIJKokkos(Mat mat, Vec xx, Vec yy, Vec zz)
{
  Mat_MPIAIJ *mpiaij = (Mat_MPIAIJ *)mat->data;
  PetscInt    nt;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(xx, &nt));
  PetscCheck(nt == mat->cmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible partition of mat (%" PetscInt_FMT ") and xx (%" PetscInt_FMT ")", mat->cmap->n, nt);
  PetscCall(VecScatterBegin(mpiaij->Mvctx, xx, mpiaij->lvec, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall((*mpiaij->A->ops->multadd)(mpiaij->A, xx, yy, zz));
  PetscCall(VecScatterEnd(mpiaij->Mvctx, xx, mpiaij->lvec, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall((*mpiaij->B->ops->multadd)(mpiaij->B, mpiaij->lvec, zz, zz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatMultTranspose_MPIAIJKokkos(Mat mat, Vec xx, Vec yy)
{
  Mat_MPIAIJ *mpiaij = (Mat_MPIAIJ *)mat->data;
  PetscInt    nt;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(xx, &nt));
  PetscCheck(nt == mat->rmap->n, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible partition of mat (%" PetscInt_FMT ") and xx (%" PetscInt_FMT ")", mat->rmap->n, nt);
  PetscCall((*mpiaij->B->ops->multtranspose)(mpiaij->B, xx, mpiaij->lvec));
  PetscCall((*mpiaij->A->ops->multtranspose)(mpiaij->A, xx, yy));
  PetscCall(VecScatterBegin(mpiaij->Mvctx, mpiaij->lvec, yy, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(mpiaij->Mvctx, mpiaij->lvec, yy, ADD_VALUES, SCATTER_REVERSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Merge the "A, B" matrices of mat into a matrix C.  mat's type is MPIAIJKOKKOS. C's type is MATSEQAIJKOKKOS.
   A is put before B. C's size would be A->rmap->n by (A->cmap->n + B->cmap->n).
   C still uses local column ids. Their corresponding global column ids are returned in glob.
*/
PetscErrorCode MatMPIAIJGetLocalMatMerge_MPIAIJKokkos(Mat mat, MatReuse reuse, IS *glob, Mat *C)
{
  Mat             Ad, Ao;
  const PetscInt *cmap;

  PetscFunctionBegin;
  PetscCall(MatMPIAIJGetSeqAIJ(mat, &Ad, &Ao, &cmap));
  PetscCall(MatSeqAIJKokkosMergeMats(Ad, Ao, reuse, C));
  if (glob) {
    PetscInt cst, i, dn, on, *gidx;
    PetscCall(MatGetLocalSize(Ad, NULL, &dn));
    PetscCall(MatGetLocalSize(Ao, NULL, &on));
    PetscCall(MatGetOwnershipRangeColumn(mat, &cst, NULL));
    PetscCall(PetscMalloc1(dn + on, &gidx));
    for (i = 0; i < dn; i++) gidx[i] = cst + i;
    for (i = 0; i < on; i++) gidx[i + dn] = cmap[i];
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)Ad), dn + on, gidx, PETSC_OWN_POINTER, glob));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Structs used in matrix products of type C=AB, C=A^tB and C=B^tAB */
struct MatMatStruct {
  PetscInt            n, *garray;     // C's garray and its size.
  KokkosCsrMatrix     Cd, Co;         // C is in split form matrices (all in local column indcies)
  KokkosCsrMatrix     C1, C2, C3, C4; // intermediate mat products
  KokkosCsrMatrix     C2_mid, C4_mid; // alias of C2, C4; share their a[], i[], but with different j[] (hence column size)
  PetscIntKokkosView  E_NzLeft;
  PetscSF             sf = nullptr; // SF to bcast or reduce matrices E to F
  MatScalarKokkosView rootBuf, leafBuf;
  KokkosCsrMatrix     Fd, Fo; // F in split form

  KernelHandle kh1; // compute C1, add C1+C3 or C1+Fd
  KernelHandle kh2; // compute C2, add C2+C4 or C2+Fo
  KernelHandle kh3; // compute C3
  KernelHandle kh4; // compute C4

  PetscInt E_TeamSize; // kernel launching parameters in merging E or splitting F
  PetscInt E_VectorLength;
  PetscInt E_RowsPerTeam;
  PetscInt F_TeamSize;
  PetscInt F_VectorLength;
  PetscInt F_RowsPerTeam;

  ~MatMatStruct()
  {
    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF, PetscSFDestroy(&sf));
    PetscFunctionReturnVoid();
  }
};

struct MatMatStruct_AB : public MatMatStruct {
  PetscIntKokkosView F_NzLeft; // plans to split F (in leafbuf) into Fd, Fo
  PetscIntKokkosView irootloc; // plans to put E (i.e., Bd, Bo) into rootBuf
  PetscIntKokkosView rowoffset;
};

struct MatMatStruct_AtB : public MatMatStruct {
  MatColIdxKokkosView Fdjmap; // plans to reduce data in rootBuf to Fd, Fo
  MatColIdxKokkosView Fdjperm;
  MatColIdxKokkosView Fojmap;
  MatColIdxKokkosView Fojperm;
};

struct MatProductData_MPIAIJKokkos {
  MatMatStruct_AB  *mmAB     = nullptr;
  MatMatStruct_AtB *mmAtB    = nullptr;
  PetscBool         reusesym = PETSC_FALSE;
  Mat               Z        = nullptr; // store Z=AB in computing BtAB

  ~MatProductData_MPIAIJKokkos()
  {
    delete mmAB;
    delete mmAtB;
    PetscCallAbort(PETSC_COMM_SELF, MatDestroy(&Z));
  }
};

static PetscErrorCode MatProductDataDestroy_MPIAIJKokkos(void *data)
{
  PetscFunctionBegin;
  PetscCallCXX(delete static_cast<MatProductData_MPIAIJKokkos *>(data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* MatSetMPIAIJKokkosWithSplitSeqAIJKokkosMatrices - Set the diag and offdiag matrices of a MATMPIAIJKOKKOS matrix.
   It is similar to MatCreateMPIAIJWithSplitArrays.

  Input Parameters:
+  mat   - the MATMPIAIJKOKKOS matrix, which should have its type and layout set, but should not have its diag, offdiag matrices set
.  A     - the diag matrix using local col ids
-  B     - the offdiag matrix using global col ids

  Output Parameter:
.  mat   - the updated MATMPIAIJKOKKOS matrix
*/
static PetscErrorCode MatSetMPIAIJKokkosWithSplitSeqAIJKokkosMatrices(Mat mat, Mat A, Mat B, PetscInt *garray)
{
  Mat_MPIAIJ *mpiaij = static_cast<Mat_MPIAIJ *>(mat->data);
  PetscInt    m, n, M, N, Am, An, Bm, Bn;

  PetscFunctionBegin;
  PetscCall(MatGetSize(mat, &M, &N));
  PetscCall(MatGetLocalSize(mat, &m, &n));
  PetscCall(MatGetLocalSize(A, &Am, &An));
  PetscCall(MatGetLocalSize(B, &Bm, &Bn));

  PetscCheck(m == Am && m == Bm, PETSC_COMM_SELF, PETSC_ERR_PLIB, "local number of rows do not match");
  PetscCheck(n == An, PETSC_COMM_SELF, PETSC_ERR_PLIB, "local number of columns do not match");
  // PetscCheck(N == Bn, PETSC_COMM_SELF, PETSC_ERR_PLIB, "global number of columns do not match");
  PetscCheck(!mpiaij->A && !mpiaij->B, PETSC_COMM_SELF, PETSC_ERR_PLIB, "A, B of the MPIAIJ matrix are not empty");
  mpiaij->A      = A;
  mpiaij->B      = B;
  mpiaij->garray = garray;

  mat->preallocated     = PETSC_TRUE;
  mat->nooffprocentries = PETSC_TRUE; /* See MatAssemblyBegin_MPIAIJ. In effect, making MatAssemblyBegin a nop */

  PetscCall(MatSetOption(mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  /* MatAssemblyEnd is critical here. It sets mat->offloadmask according to A and B's, and
    also gets mpiaij->B compacted, with its col ids and size reduced
  */
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOption(mat, MAT_NO_OFF_PROC_ENTRIES, PETSC_FALSE));
  PetscCall(MatSetOption(mat, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_TRUE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Adapted from Kokkos-Kernels spmv_launch_parameters(), to get parameters in Kokkos nested loops which we used to merge or
// split csr matrices. The rule is to have "vector_length * team_size" be around 256 on GPUs (e.g., for a CUDA thread block)
template <class ExecutionSpace>
static PetscErrorCode MatMergeGetLaunchParameters(PetscInt numRows, PetscInt nnz, PetscInt rows_per_thread, PetscInt &team_size, PetscInt &vector_length, PetscInt &rows_per_team)
{
  Kokkos::TeamPolicy<ExecutionSpace> teamPolicy(128, Kokkos::AUTO);

  PetscFunctionBegin;
  PetscInt nnz_per_row = numRows ? (nnz / numRows) : 0; // we might meet empty matrices

  if (nnz_per_row < 1) nnz_per_row = 1;

  int max_vector_length = teamPolicy.vector_length_max();

  if (vector_length < 1) {
    vector_length = 1;
    while (vector_length < max_vector_length && vector_length * 6 < nnz_per_row) vector_length *= 2;
  }

  // Determine rows per thread
  if (rows_per_thread < 1) {
    if (KokkosKernels::Impl::kk_is_gpu_exec_space<ExecutionSpace>()) rows_per_thread = 1;
    else {
      if (nnz_per_row < 20 && nnz > 5000000) {
        rows_per_thread = 256;
      } else rows_per_thread = 64;
    }
  }

  if (team_size < 1) {
    if (KokkosKernels::Impl::kk_is_gpu_exec_space<ExecutionSpace>()) {
      team_size = 256 / vector_length;
    } else {
      team_size = 1;
    }
  }

  rows_per_team = rows_per_thread * team_size;

  if (rows_per_team < 0) {
    PetscInt nnz_per_team = 4096;
    PetscInt conc         = ExecutionSpace().concurrency();
    while ((conc * nnz_per_team * 4 > nnz) && (nnz_per_team > 256)) nnz_per_team /= 2;
    rows_per_team = (nnz_per_team + nnz_per_row - 1) / nnz_per_row;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Reduce two sets of global indices into local ones

  Input Parameters:
+  n1          - size of garray1[], the first set
.  garray1[n1] - a sorted global index array (without duplicates)
.  m           - size of indices[], the second set
-  indices[m]  - a unsorted global index array (might have duplicates), which will be updated on output into local ones

  Output Parameters:
+  n2          - size of garray2[], the merged set, which combines garray1[] and indices[]
.  garray2[n2] - allocated by callee using PetscMalloc1(). Contains sorted unique global indices (without duplicates). Caller needs to free it.
.  map[n1]     - allocated by caller. It gives garray1[i] = garray2[map[i]]
-  indices[m]  - on output, global indices in this array are rewritten with local ones, i.e, indices_input[i] = garray2[indices_output[i]]

   Example, say
    n1         = 5
    garray1[5] = {1, 4, 7, 8, 10}
    m          = 4
    indices[4] = {2, 4, 8, 9}

   Combining them together, we have 7 global indices in garray2[]
    n2         = 7
    garray2[7] = {1, 2, 4, 7, 8, 9, 10}

   And we have map[] to connect "garray1[i] = garray2[map[i]], i=[0,n1)"
    map[5] = {0, 2, 3, 4, 6}

   On output, indices[] is updated with local indices
    indices[4] = {1, 2, 4, 5}
*/
static PetscErrorCode ReduceTwoSetsOfGlobalIndices(PetscInt n1, const PetscInt *garray1, PetscInt m, PetscInt *indices, PetscInt *n2_, PetscInt **garray2_, PetscInt *map)
{
  PetscHMapI    g2l = nullptr;
  PetscHashIter iter;
  PetscInt      tot, key, val; // total unique global indices. key is global id; val is local id
  PetscInt      n2, *garray2;

  PetscFunctionBegin;
  tot = 0;
  PetscCall(PetscHMapICreateWithSize(n1, &g2l));
  for (PetscInt i = 0; i < m; i++) {                                // insert those in indices[]
    PetscCall(PetscHMapIGetWithDefault(g2l, indices[i], -1, &val)); // if not exist, val is set with -1
    if (val < 0) PetscCall(PetscHMapISet(g2l, indices[i], tot++));  // val < 0 means gid is not in the hash table yet
  }

  for (PetscInt i = 0; i < n1; i++) { // insert those in garray1[]
    PetscCall(PetscHMapIGetWithDefault(g2l, garray1[i], -1, &val));
    if (val < 0) PetscCall(PetscHMapISet(g2l, garray1[i], tot++));
  }

  // Pull out (unique) globals in the hash table and put them in garray2[]
  n2 = tot;
  PetscCall(PetscMalloc1(n2, &garray2));
  tot = 0;
  PetscHashIterBegin(g2l, iter);
  while (!PetscHashIterAtEnd(g2l, iter)) {
    PetscHashIterGetKey(g2l, iter, key);
    PetscHashIterNext(g2l, iter);
    garray2[tot++] = key;
  }

  // Sort garray2[] and then map them to local indices starting from 0
  PetscCall(PetscSortInt(n2, garray2));
  PetscCall(PetscHMapIClear(g2l));
  for (PetscInt i = 0; i < tot; i++) PetscCall(PetscHMapISet(g2l, garray2[i], i)); // i is the local id

  // Rewrite indices[] with local indices
  for (PetscInt i = 0; i < m; i++) {
    PetscCall(PetscHMapIGetWithDefault(g2l, indices[i], -1, &val));
    PetscAssert(val >= 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Met a negative local column index");
    indices[i] = val;
  }
  // Record the map that maps garray1[i] to garray2[map[i]]
  for (PetscInt i = 0; i < n1; i++) PetscCall(PetscHMapIGetWithDefault(g2l, garray1[i], -1, &map[i]));
  PetscCall(PetscHMapIDestroy(&g2l));
  *n2_      = n2;
  *garray2_ = garray2;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatMPIAIJKokkosReduce - Reduce rows of a MPIAIJKOKKOS matrix (E, in split form) to produce another matrix (F, also in split form, stored in mm)

  It is the reverse of MatMPIAIJKokkosBcast() in some sense, but with a different signature since we do not really need a fully populated MPIAIJKOKKOS E.

  Think each row of E as a leaf, then the given ownerSF specifies roots for the leaves. Roots may connect to multiple leaves.
  In this routine, we sparse-merge leaves (rows) at their roots to form potentially longer rows in F. F's number of rows will be nroots of ownerSF.

  Input Parameters:
+  comm       - MPI communicator of E
.  A          - diag block of E, using local column indices
.  B          - off-diag block of E, using local column indices
.  cstart      - (global) start column of Ed
.  cend        - (global) end column + 1 of Ed.  In other words, E's column ownership is in range of [cstart, cend)
.  garray1[n1] - global column indices of Eo. Here n1 is Eo's column size.
.  ownerSF     - the SF specifies ownership (root) of rows in E
.  reuse       - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX
-  mm          - to stash intermediate data structures for reuse

  Output Parameters:
+  map[n1]  - allocated by caller. It maps garray1[] to garray2[]. See more at ReduceTwoSetsOfGlobalIndices().
-  mm       - contains various info, such as garray2[], F (Fd, Fo) etc.

  Notes:
  When reuse = MAT_REUSE_MATRIX, cstart, cend, garray1, ownerSF, map are not significant.

 */
static PetscErrorCode MatMPIAIJKokkosReduceBegin(MPI_Comm comm, KokkosCsrMatrix A, KokkosCsrMatrix B, PetscInt cstart, PetscInt cend, const PetscInt *garray1, PetscSF ownerSF, MatReuse reuse, PetscInt *map, MatMatStruct_AtB *mm)
{
  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscInt Em = A.numRows(), Fm;
    PetscInt n1 = B.numCols();

    PetscCall(PetscSFGetGraph(ownerSF, &Fm, NULL, NULL, NULL)); // Fm = #rows of F = nroots of ownerSF

    // Do the analysis on host
    auto                 Ai_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.graph.row_map);
    auto                 Aj_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.graph.entries);
    auto                 Bi_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), B.graph.row_map);
    auto                 Bj_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), B.graph.entries);
    const MatRowMapType *Ai = Ai_h.data(), *Bi = Bi_h.data();
    const MatColIdxType *Aj = Aj_h.data(), *Bj = Bj_h.data();

    // Count how many nonzeros of each row in E are in the left of the diag block [cstart,cend)
    PetscIntKokkosViewHost E_NzLeft_h("E_NzLeft_h", Em), E_RowLen_h("E_RowLen_h", Em);
    PetscInt              *E_NzLeft = E_NzLeft_h.data(), *E_RowLen = E_RowLen_h.data();
    for (PetscInt i = 0; i < Em; i++) {
      const PetscInt *first, *last, *it;
      PetscInt        count, step;
      // std::lower_bound(first,last,cstart), but need to use global column indices
      first = Bj + Bi[i];
      last  = Bj + Bi[i + 1];
      count = last - first;
      while (count > 0) {
        it   = first;
        step = count / 2;
        it += step;
        if (garray1[*it] < cstart) { // map local to global
          first = ++it;
          count -= step + 1;
        } else count = step;
      }
      E_NzLeft[i] = first - (Bj + Bi[i]);
      E_RowLen[i] = (Ai[i + 1] - Ai[i]) + (Bi[i + 1] - Bi[i]);
    }

    // Get length of rows (i.e., sizes of leaves) that contribute to my roots
    const PetscMPIInt *iranks, *ranks;
    const PetscInt    *ioffset, *irootloc, *roffset, *rmine;
    PetscInt           niranks, nranks;
    MPI_Request       *reqs;
    PetscMPIInt        tag;
    PetscSF            reduceSF;
    PetscInt          *sdisp, *rdisp;

    PetscCall(PetscCommGetNewTag(comm, &tag));
    PetscCall(PetscSFGetLeafRanks(ownerSF, &niranks, &iranks, &ioffset, &irootloc));  // get leaf ranks connecting to roots on this process (I'll recv from them)
    PetscCall(PetscSFGetRootRanks(ownerSF, &nranks, &ranks, &roffset, &rmine, NULL)); // get root ranks this process connects (I'll send to them)

    // Find out length of each row I will receive. Even for the same row index, when they are from
    // different senders, they might have different lengths (and sparsity patterns)
    PetscInt  sendRowCnt = roffset[nranks], recvRowCnt = ioffset[niranks];
    PetscInt *sendRowLen, *recvRowLen; // lengths of rows of I need to send/recv per process

    PetscCall(PetscMalloc5(sendRowCnt, &sendRowLen, recvRowCnt + 1, &recvRowLen, nranks, &sdisp, niranks + 1, &rdisp, nranks + niranks, &reqs));

    for (PetscInt i = 0; i < sendRowCnt; i++) sendRowLen[i] = E_RowLen[rmine[i]];
    recvRowLen[0] = 0; // since we will make it in CSR format later
    recvRowLen++;      // advance the pointer now
    for (PetscInt i = 0; i < niranks; i++) { MPI_Irecv(&recvRowLen[ioffset[i]], ioffset[i + 1] - ioffset[i], MPIU_INT, iranks[i], tag, comm, &reqs[nranks + i]); }
    for (PetscInt i = 0; i < nranks; i++) { MPI_Isend(&sendRowLen[roffset[i]], roffset[i + 1] - roffset[i], MPIU_INT, ranks[i], tag, comm, &reqs[i]); }
    PetscCallMPI(MPI_Waitall(nranks + niranks, reqs, MPI_STATUSES_IGNORE));

    // Build the real PetscSF for reducing E rows (buffer to buffer)
    rdisp[0] = 0;
    for (PetscInt i = 0; i < niranks; i++) {
      rdisp[i + 1] = rdisp[i];
      for (PetscInt j = ioffset[i]; j < ioffset[i + 1]; j++) { rdisp[i + 1] += recvRowLen[j]; }
    }
    recvRowLen--; // put it back into csr format
    for (PetscInt i = 0; i < recvRowCnt; i++) recvRowLen[i + 1] += recvRowLen[i];

    for (PetscInt i = 0; i < nranks; i++) { MPI_Irecv(&sdisp[i], 1, MPIU_INT, ranks[i], tag, comm, &reqs[i]); }
    for (PetscInt i = 0; i < niranks; i++) { MPI_Isend(&rdisp[i], 1, MPIU_INT, iranks[i], tag, comm, &reqs[nranks + i]); }
    PetscCallMPI(MPI_Waitall(nranks + niranks, reqs, MPI_STATUSES_IGNORE));

    PetscInt     nleaves = 0, Enz = 0;    // leaves are nonzeros I will send
    PetscInt     nroots = rdisp[niranks]; // roots are nonzeros I will recv
    PetscSFNode *iremote;

    for (PetscInt i = 0; i < Em; i++) Enz += E_RowLen[i];
    PetscAssert(A.nnz() + B.nnz() == Enz, comm, PETSC_ERR_PLIB, "Enz should be equal to sum of nnz of A and B");
    PetscCallMPI(PetscMalloc1(Enz, &iremote)); // no free, since we give ownership to reduceSF

    for (PetscInt i = 0; i < nranks; i++) {
      PetscInt count = 0;
      for (PetscInt j = roffset[i]; j < roffset[i + 1]; j++) count += E_RowLen[rmine[j]];
      for (PetscInt j = 0; j < count; j++) {
        iremote[nleaves + j].rank  = ranks[i];
        iremote[nleaves + j].index = sdisp[i] + j;
      }
      nleaves += count;
    }
    PetscCheck(nleaves == Enz, comm, PETSC_ERR_PLIB, "nleaves should be equal to Enz");

    PetscCall(PetscSFCreate(comm, &reduceSF));
    PetscCall(PetscSFSetGraph(reduceSF, nroots, nleaves, NULL, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));

    // Copy (global) column indices of the needed rows in E to sendCol[], and then PetscSFReduce to recvCol[]
    PetscInt *sendCol, *recvCol;
    PetscCall(PetscMalloc2(nleaves, &sendCol, nroots, &recvCol));
    for (PetscInt k = 0; k < roffset[nranks]; k++) {
      PetscInt  i      = rmine[k]; // row to be copied
      PetscInt *buf    = &sendCol[Ai[i] + Bi[i]];
      PetscInt  nzLeft = E_NzLeft[i];
      PetscInt  alen = Ai[i + 1] - Ai[i], blen = Bi[i + 1] - Bi[i];
      for (PetscInt j = 0; j < alen + blen; j++) {
        if (j < nzLeft) {
          buf[j] = garray1[Bj[Bi[i] + j]]; // left B, in global
        } else if (j < nzLeft + alen) {
          buf[j] = Aj[Ai[i] + j - nzLeft] + cstart; // diag A, also in global
        } else {
          buf[j] = garray1[Bj[Bi[i] + j - alen]]; // right B, in global
        }
      }
    }
    PetscCall(PetscSFReduceWithMemTypeBegin(reduceSF, MPIU_INT, PETSC_MEMTYPE_HOST, sendCol, PETSC_MEMTYPE_HOST, recvCol, MPI_REPLACE));
    PetscCall(PetscSFReduceEnd(reduceSF, MPIU_INT, sendCol, recvCol, MPI_REPLACE));

    // With recvCol[], we do a series of analysis to get i, j of Fd, Fo, and build plans to reduce nonzeros in recv buffers to Fd and Fo
    PetscInt *recvRowPerm, *recvColSorted;
    PetscInt *recvNzPerm, *recvNzPermSorted;
    PetscCall(PetscMalloc4(recvRowCnt, &recvRowPerm, nroots, &recvColSorted, nroots, &recvNzPerm, nroots, &recvNzPermSorted));

    for (PetscInt i = 0; i < nroots; i++) recvNzPerm[i] = i;                   // numbering all received nonzeros
    for (PetscInt i = 0; i < recvRowCnt; i++) recvRowPerm[i] = i;              // put up a permutation array, so that after sorting we know where to get a row in recvCol[]
    PetscCall(PetscSortIntWithPermutation(recvRowCnt, irootloc, recvRowPerm)); // irootloc[] (owned by ownerSF) won't be changed

    // i[] array, nz are always easiest to compute
    MatRowMapKokkosViewHost Fdi_h("Fdi_h", Fm + 1), Foi_h("Foi_h", Fm + 1);
    MatRowMapType          *Fdi, *Foi;
    PetscInt                FnzDups = 0, Fdnz = 0, FdnzDups = 0, Fonz = 0, FonzDups = 0; // nz (with or without dups) in F, Fd, Fo
    PetscInt                iter;

    Kokkos::deep_copy(Fdi_h, 0); // zero, as we will do 'val++' on them
    Kokkos::deep_copy(Foi_h, 0);
    Fdi  = Fdi_h.data() + 1; // +1 for easy indexing in code below
    Foi  = Foi_h.data() + 1;
    iter = 0;
    while (iter < recvRowCnt) { // iter over received rows
      PetscInt curRowIdx = irootloc[recvRowPerm[iter]];
      PetscInt dupRows   = 1; // current row has this many contributing rows (of various sparsity patterns)

      while (iter + dupRows < recvRowCnt && irootloc[recvRowPerm[iter + dupRows]] == curRowIdx) dupRows++;

      // Copy column indices (and their permutation) of these rows into recvColSorted & recvNzPermSorted
      PetscInt  nz    = 0; // nz (with dups) in the current row
      PetscInt *jbuf  = recvColSorted + FnzDups;
      PetscInt *pbuf  = recvNzPermSorted + FnzDups;
      PetscInt *jbuf2 = jbuf; // temp pointers
      PetscInt *pbuf2 = pbuf;
      for (PetscInt d = 0; d < dupRows; d++) {
        PetscInt i   = recvRowPerm[iter + d];
        PetscInt len = recvRowLen[i + 1] - recvRowLen[i];
        PetscCall(PetscArraycpy(jbuf2, &recvCol[recvRowLen[i]], len));
        PetscCall(PetscArraycpy(pbuf2, &recvNzPerm[recvRowLen[i]], len));
        jbuf2 += len;
        pbuf2 += len;
        nz += len;
      }
      PetscCall(PetscIntSortSemiOrderedWithArray(nz, jbuf, pbuf)); // It could be improved with k-way merge sort, since the rows are already sorted

      // Scan column indices (in jbuf[0,nz), might have dups) of this row, and see how many go to Fd and how many go to Fo
      PetscInt cur = 0;
      while (cur < nz) {
        PetscInt curColIdx = jbuf[cur];
        PetscInt dups      = 1;

        while (cur + dups < nz && jbuf[cur + dups] == curColIdx) dups++;
        if (curColIdx >= cstart && curColIdx < cend) {
          Fdi[curRowIdx]++;
          FdnzDups += dups;
        } else {
          Foi[curRowIdx]++;
          FonzDups += dups;
        }
        cur += dups;
      }

      FnzDups += nz;
      iter += dupRows; // Move to next unique row
    }

    Fdi = Fdi_h.data(); // restore Fdi, Foi and make them CSR
    Foi = Foi_h.data();
    for (PetscInt i = 0; i < Fm; i++) {
      Fdi[i + 1] += Fdi[i];
      Foi[i + 1] += Foi[i];
    }
    Fdnz = Fdi[Fm];
    Fonz = Foi[Fm];
    PetscCall(PetscFree2(sendCol, recvCol));

    // Allocate j, jmap, jperm for Fd and Fo
    MatColIdxKokkosViewHost Fdj_h("Fdj_h", Fdnz), Foj_h("Foj_h", Fonz);
    MatRowMapKokkosViewHost Fdjmap_h("Fdjmap_h", Fdnz + 1), Fojmap_h("Fojmap_h", Fonz + 1); // +1 to make csr
    MatRowMapKokkosViewHost Fdjperm_h("Fdjperm_h", FdnzDups), Fojperm_h("Fojperm_h", FonzDups);
    MatColIdxType          *Fdj = Fdj_h.data(), *Foj = Foj_h.data();
    MatRowMapType          *Fdjmap = Fdjmap_h.data(), *Fojmap = Fojmap_h.data();
    MatRowMapType          *Fdjperm = Fdjperm_h.data(), *Fojperm = Fojperm_h.data();

    // Scan recvColSorted[] again, and fill j, jmap, jperm for Fd and Fo
    Fdjmap[0] = 0;
    Fojmap[0] = 0;
    FnzDups   = 0;
    Fdnz      = 0;
    Fonz      = 0;
    iter      = 0; // iter over received rows
    while (iter < recvRowCnt) {
      PetscInt curRowIdx = irootloc[recvRowPerm[iter]]; // current row idx
      PetscInt dupRows   = 1;                           // It has this many contributing rows (of various lengths)
      PetscInt nz        = 0;                           // nz (with dups) in the current row

      while (iter + dupRows < recvRowCnt && irootloc[recvRowPerm[iter + dupRows]] == curRowIdx) dupRows++;
      for (PetscInt d = 0; d < dupRows; d++) {
        PetscInt i = recvRowPerm[iter + d];
        nz += recvRowLen[i + 1] - recvRowLen[i];
      }

      PetscInt *jbuf = recvColSorted + FnzDups;
      // Scan columns (in jbuf[0,nz) of this row, copy them and their permutation to j[] and jperm[] of Fd and Fo
      PetscInt cur = 0;
      while (cur < nz) {
        PetscInt curColIdx = jbuf[cur];
        PetscInt dups      = 1;

        while (cur + dups < nz && jbuf[cur + dups] == curColIdx) dups++;
        if (curColIdx >= cstart && curColIdx < cend) {
          Fdj[Fdnz]        = curColIdx - cstart; // easily convert to local
          Fdjmap[Fdnz + 1] = Fdjmap[Fdnz] + dups;
          for (PetscInt j = 0; j < dups; j++) Fdjperm[Fdjmap[Fdnz] + j] = recvNzPermSorted[FnzDups + j];
          FdnzDups += dups;
          Fdnz++;
        } else {
          Foj[Fonz]        = curColIdx; // in global
          Fojmap[Fonz + 1] = Fojmap[Fonz] + dups;
          for (PetscInt j = 0; j < dups; j++) Fojperm[Fojmap[Fonz] + j] = recvNzPermSorted[FnzDups + j];
          FonzDups += dups;
          Fonz++;
        }
        cur += dups;
        FnzDups += dups;
      }
      iter += dupRows; // Move to next unique row
    }
    PetscCall(PetscFree4(recvRowPerm, recvColSorted, recvNzPerm, recvNzPermSorted));
    PetscCall(PetscFree5(sendRowLen, recvRowLen, sdisp, rdisp, reqs));

    // Combine global column indices in garray1[] and Foj[]
    PetscInt n2, *garray2;

    PetscCall(ReduceTwoSetsOfGlobalIndices(n1, garray1, Fonz, Foj, &n2, &garray2, map));
    mm->sf       = reduceSF;
    mm->leafBuf  = MatScalarKokkosView("leafBuf", nleaves);
    mm->rootBuf  = MatScalarKokkosView("rootBuf", nroots);
    mm->garray   = garray2; // give ownership, so no free
    mm->n        = n2;
    mm->E_NzLeft = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), E_NzLeft_h);
    mm->Fdjmap   = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), Fdjmap_h);
    mm->Fdjperm  = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), Fdjperm_h);
    mm->Fojmap   = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), Fojmap_h);
    mm->Fojperm  = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), Fojperm_h);

    // Output Fd and Fo in KokkosCsrMatrix format
    MatScalarKokkosView Fda_d("Fda_d", Fdnz);
    MatRowMapKokkosView Fdi_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), Fdi_h);
    MatColIdxKokkosView Fdj_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), Fdj_h);
    MatScalarKokkosView Foa_d("Foa_d", Fonz);
    MatRowMapKokkosView Foi_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), Foi_h);
    MatColIdxKokkosView Foj_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), Foj_h);

    PetscCallCXX(mm->Fd = KokkosCsrMatrix("Fd", Fm, cend - cstart, Fdnz, Fda_d, Fdi_d, Fdj_d));
    PetscCallCXX(mm->Fo = KokkosCsrMatrix("Fo", Fm, n2, Fonz, Foa_d, Foi_d, Foj_d)); // Fo's column size is n2, length of garray2[]

    // Compute kernel launch parameters in merging E
    PetscInt teamSize, vectorLength, rowsPerTeam;

    teamSize = vectorLength = rowsPerTeam = -1;
    PetscCall(MatMergeGetLaunchParameters<DefaultExecutionSpace>(Em, Enz, -1, teamSize, vectorLength, rowsPerTeam));
    mm->E_TeamSize     = teamSize;
    mm->E_VectorLength = vectorLength;
    mm->E_RowsPerTeam  = rowsPerTeam;
  } else PetscCheck(reuse == MAT_REUSE_MATRIX, comm, PETSC_ERR_PLIB, "Unsupported MatReuse enum %d", reuse);

  // Handy aliases
  auto       &Aa           = A.values;
  auto       &Ba           = B.values;
  const auto &Ai           = A.graph.row_map;
  const auto &Bi           = B.graph.row_map;
  const auto &E_NzLeft     = mm->E_NzLeft;
  auto       &leafBuf      = mm->leafBuf;
  auto       &rootBuf      = mm->rootBuf;
  PetscSF     reduceSF     = mm->sf;
  PetscInt    Em           = A.numRows();
  PetscInt    teamSize     = mm->E_TeamSize;
  PetscInt    vectorLength = mm->E_VectorLength;
  PetscInt    rowsPerTeam  = mm->E_RowsPerTeam;
  PetscInt    workSets     = (Em + rowsPerTeam - 1) / rowsPerTeam;

  // Copy rows in A/B of E to leafBuf, then pass it to rootBuf
  PetscCallCXX(Kokkos::parallel_for(
    Kokkos::TeamPolicy<>(workSets, teamSize, vectorLength), KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, rowsPerTeam), [&](PetscInt k) {
        PetscInt i = t.league_rank() * rowsPerTeam + k; // i-th row in F
        if (i < Em) {
          PetscInt disp   = Ai(i) + Bi(i);
          PetscInt alen   = Ai(i + 1) - Ai(i);
          PetscInt blen   = Bi(i + 1) - Bi(i);
          PetscInt nzleft = E_NzLeft(i);

          Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, alen + blen), [&](PetscInt j) {
            MatScalar &val = leafBuf(disp + j);
            if (j < nzleft) { // B left
              val = Ba(Bi(i) + j);
            } else if (j < nzleft + alen) { // diag A
              val = Aa(Ai(i) + j - nzleft);
            } else { // B right
              val = Ba(Bi(i) + j - alen);
            }
          });
        }
      });
    }));
  PetscCall(PetscSFReduceWithMemTypeBegin(reduceSF, MPIU_SCALAR, PETSC_MEMTYPE_KOKKOS, leafBuf.data(), PETSC_MEMTYPE_KOKKOS, rootBuf.data(), MPI_REPLACE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// To finish MatMPIAIJKokkosReduce.
static PetscErrorCode MatMPIAIJKokkosReduceEnd(MPI_Comm comm, KokkosCsrMatrix A, KokkosCsrMatrix B, PetscInt cstart, PetscInt cend, const PetscInt *garray1, PetscSF ownerSF, MatReuse reuse, PetscInt *map, MatMatStruct_AtB *mm)
{
  PetscFunctionBegin;
  auto       &leafBuf  = mm->leafBuf;
  auto       &rootBuf  = mm->rootBuf;
  auto       &Fda      = mm->Fd.values;
  const auto &Fdjmap   = mm->Fdjmap;
  const auto &Fdjperm  = mm->Fdjperm;
  auto        Fdnz     = mm->Fd.nnz();
  auto       &Foa      = mm->Fo.values;
  const auto &Fojmap   = mm->Fojmap;
  const auto &Fojperm  = mm->Fojperm;
  auto        Fonz     = mm->Fo.nnz();
  PetscSF     reduceSF = mm->sf;

  PetscCall(PetscSFReduceEnd(reduceSF, MPIU_SCALAR, leafBuf.data(), rootBuf.data(), MPI_REPLACE));

  // Reduce data in rootBuf to Fd and Fo
  PetscCallCXX(Kokkos::parallel_for(
    Fdnz, KOKKOS_LAMBDA(const MatRowMapType i) {
      PetscScalar sum = 0.0;
      for (MatRowMapType k = Fdjmap(i); k < Fdjmap(i + 1); k++) sum += rootBuf(Fdjperm(k));
      Fda(i) = sum;
    }));

  PetscCallCXX(Kokkos::parallel_for(
    Fonz, KOKKOS_LAMBDA(const MatRowMapType i) {
      PetscScalar sum = 0.0;
      for (MatRowMapType k = Fojmap(i); k < Fojmap(i + 1); k++) sum += rootBuf(Fojperm(k));
      Foa(i) = sum;
    }));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  MatMPIAIJKokkosBcast - Bcast local rows of a MPIAIJKOKKOS matrix (E) to produce a local matrix (F, stored in mm) in split form

  This is a complex routine. It is essentially the MPIAIJKOKKOS counterpart of MatGetBrowsOfAoCols_MPIAIJ, but supports
  device and involves various index mapping.

  In the given ownerSF, leaves correspond to rows in F, and roots correspond to rows in E. Roots may connect to multiple leaves.
  Suppose F's j-th row is connected to a root identified by PetscSFNode (k,i), it means we need to bcast the i-th row of E on rank k
  to j-th row of F. ownerSF is not an arbitrary SF, instead it is the Mvctx of another MPIAIJ matrix A that is able to perform A*E.
  F has the same column layout as E.

  Conceptually F has global column indices. In this routine, we spit F into diagonal Fd and off-diagonal Fo.
  Fd uses local column indices, which are easy to compute. We just need to subtract the "local column range start" from the global indices.
  Fo had global column indices at first. We will reduce them into local ones. In doing that, we also take into account the global
  column indices that E's off-diag block has. Let's say there are n1 such indices stored in garray1[]. We will reduce them along with
  column indices in Fo and update Fo with local indices.

   Input Parameters:
+   E       - the MPIAIJKOKKOS matrix
.   ownerSF - the ownership SF (nonsignificant in MAT_REUSE_MATRIX)
.   reuse   - either MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX
-   mm      - to stash matproduct intermediate data structures

    Output Parameters:
+   map[n1] - allocated by caller. It maps garray1[] to garray2[]. See more at ReduceTwoSetsOfGlobalIndices.
-   mm      - contains various info, such as garray2[], Fd, Fo, etc.

    Notes:
    When reuse = MAT_REUSE_MATRIX, ownerSF, map are not significant.
    The routine is provide in split-phase form MatMPIAIJKokkosBcastBegin/End() to provide computation/communication opportunities.
*/
static PetscErrorCode MatMPIAIJKokkosBcastBegin(Mat E, PetscSF ownerSF, MatReuse reuse, PetscInt *map, MatMatStruct_AB *mm)
{
  Mat_MPIAIJ       *empi = static_cast<Mat_MPIAIJ *>(E->data);
  Mat               A = empi->A, B = empi->B; // diag and off-diag
  Mat_SeqAIJKokkos *akok = static_cast<Mat_SeqAIJKokkos *>(A->spptr), *bkok = static_cast<Mat_SeqAIJKokkos *>(B->spptr);
  PetscInt          Em = E->rmap->n; // #local rows
  MPI_Comm          comm;

  PetscFunctionBegin;
  PetscCallMPI(PetscObjectGetComm((PetscObject)E, &comm));
  if (reuse == MAT_INITIAL_MATRIX) {
    Mat_SeqAIJ     *aseq = static_cast<Mat_SeqAIJ *>(A->data), *bseq = static_cast<Mat_SeqAIJ *>(B->data);
    PetscInt        n1 = B->cmap->n, *Ai = aseq->i, *Aj = aseq->j, *Bi = bseq->i, *Bj = bseq->j;
    const PetscInt *garray1 = empi->garray; // its size is n1
    PetscInt        cstart, cend;
    PetscSF         bcastSF;

    PetscCall(MatGetOwnershipRangeColumn(E, &cstart, &cend));

    // Count how many nonzeros of each row in E are in the left of the diag block [cstart,cend)
    PetscIntKokkosViewHost E_NzLeft_h("E_NzLeft_h", Em), E_RowLen_h("E_RowLen_h", Em);
    PetscInt              *E_NzLeft = E_NzLeft_h.data(), *E_RowLen = E_RowLen_h.data();
    for (PetscInt i = 0; i < Em; i++) {
      const PetscInt *first, *last, *it;
      PetscInt        count, step;
      // std::lower_bound(first,last,cstart), but need to use global column indices
      first = Bj + Bi[i];
      last  = Bj + Bi[i + 1];
      count = last - first;
      while (count > 0) {
        it   = first;
        step = count / 2;
        it += step;
        if (empi->garray[*it] < cstart) { // map local to global
          first = ++it;
          count -= step + 1;
        } else count = step;
      }
      E_NzLeft[i] = first - (Bj + Bi[i]);
      E_RowLen[i] = (Ai[i + 1] - Ai[i]) + (Bi[i + 1] - Bi[i]);
    }

    // Compute row pointer Fi of F
    PetscInt *Fi, Fm, Fnz;
    PetscCall(PetscSFGetGraph(ownerSF, NULL, &Fm, NULL, NULL)); // Fm = #rows of F = nleaves of ownerSF
    PetscCall(PetscMalloc1(Fm + 1, &Fi));
    Fi[0] = 0;
    PetscCall(PetscSFBcastWithMemTypeBegin(ownerSF, MPIU_INT, PETSC_MEMTYPE_HOST, E_RowLen, PETSC_MEMTYPE_HOST, &Fi[1], MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(ownerSF, MPIU_INT, E_RowLen, &Fi[1], MPI_REPLACE));
    for (PetscInt i = 0; i < Fm; i++) Fi[i + 1] += Fi[i];
    Fnz = Fi[Fm];

    // Build the real PetscSF for bcasting E rows (buffer to buffer)
    const PetscMPIInt *iranks, *ranks;
    const PetscInt    *ioffset, *irootloc, *roffset;
    PetscInt           niranks, nranks, *sdisp, *rdisp;
    MPI_Request       *reqs;
    PetscMPIInt        tag;

    PetscCall(PetscSFGetLeafRanks(ownerSF, &niranks, &iranks, &ioffset, &irootloc)); // get leaf ranks referencing roots on this process
    PetscCall(PetscSFGetRootRanks(ownerSF, &nranks, &ranks, &roffset, NULL, NULL));  // recv info
    PetscCall(PetscMalloc3(niranks + 1, &sdisp, nranks, &rdisp, niranks + nranks, &reqs));

    sdisp[0] = 0; // send displacement
    for (PetscInt i = 0; i < niranks; i++) {
      sdisp[i + 1] = sdisp[i];
      for (PetscInt j = ioffset[i]; j < ioffset[i + 1]; j++) {
        PetscInt r = irootloc[j]; // row to be sent
        sdisp[i + 1] += E_RowLen[r];
      }
    }

    PetscCallMPI(PetscCommGetNewTag(comm, &tag));
    for (PetscInt i = 0; i < nranks; i++) PetscCallMPI(MPI_Irecv(&rdisp[i], 1, MPIU_INT, ranks[i], tag, comm, &reqs[i]));
    for (PetscInt i = 0; i < niranks; i++) PetscCallMPI(MPI_Isend(&sdisp[i], 1, MPIU_INT, iranks[i], tag, comm, &reqs[nranks + i]));
    PetscCallMPI(MPI_Waitall(niranks + nranks, reqs, MPI_STATUSES_IGNORE));

    PetscInt     nleaves = Fnz;            // leaves are nonzeros I will receive
    PetscInt     nroots  = sdisp[niranks]; // roots are nonzeros I will send
    PetscSFNode *iremote;                  // give ownership to bcastSF
    PetscCall(PetscMalloc1(nleaves, &iremote));
    for (PetscInt i = 0; i < nranks; i++) { // for each sender rank
      PetscInt k = 0;
      for (PetscInt j = Fi[roffset[i]]; j < Fi[roffset[i + 1]]; j++) { // I will receive rows [roffset[i], roffset[i+1]) of F from ranks[i]
        iremote[j].rank  = ranks[i];
        iremote[j].index = rdisp[i] + k; // their root location
        k++;
      }
    }
    PetscCall(PetscSFCreate(comm, &bcastSF));
    PetscCall(PetscSFSetGraph(bcastSF, nroots, nleaves, NULL, PETSC_OWN_POINTER, iremote, PETSC_OWN_POINTER));
    PetscCall(PetscFree3(sdisp, rdisp, reqs));

    // Build a plan (rowoffset, irootloc, E_NzLeft) to copy rows in E to rootdata of bcastSF in parallel
    PetscIntKokkosViewHost rowoffset_h("rowoffset_h", ioffset[niranks] + 1);
    PetscInt              *rowoffset = rowoffset_h.data(); // for each entry (row) indicated in irootloc[], we calculate its destinate offset in copying
    rowoffset[0]                     = 0;
    for (PetscInt i = 0; i < ioffset[niranks]; i++) { rowoffset[i + 1] += rowoffset[i] + E_RowLen[irootloc[i]]; }

    // Copy (global) column indices of the needed rows in E to a buffer, and then bcast to Fj[]
    PetscInt *jbuf, *Fj;
    PetscCall(PetscMalloc2(nroots, &jbuf, Fnz, &Fj));
    for (PetscInt k = 0; k < ioffset[niranks]; k++) {
      PetscInt  i      = irootloc[k]; // row to be copied
      PetscInt *buf    = &jbuf[rowoffset[k]];
      PetscInt  nzLeft = E_NzLeft[i];
      PetscInt  alen = Ai[i + 1] - Ai[i], blen = Bi[i + 1] - Bi[i];
      for (PetscInt j = 0; j < alen + blen; j++) {
        if (j < nzLeft) {
          buf[j] = empi->garray[Bj[Bi[i] + j]]; // left B, in global
        } else if (j < nzLeft + alen) {
          buf[j] = Aj[Ai[i] + j - nzLeft] + cstart; // diag A, also in global
        } else {
          buf[j] = empi->garray[Bj[Bi[i] + j - alen]]; // right B, in global
        }
      }
    }
    PetscCall(PetscSFBcastWithMemTypeBegin(bcastSF, MPIU_INT, PETSC_MEMTYPE_HOST, jbuf, PETSC_MEMTYPE_HOST, Fj, MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(bcastSF, MPIU_INT, jbuf, Fj, MPI_REPLACE));

    // Build a plan (i.e., F_NzLeft) to split F into Fd and Fo
    MatRowMapKokkosViewHost Fdi_h("Fdi_h", Fm + 1), Foi_h("Foi_h", Fm + 1); // row pointer of Fd, Fo
    MatColIdxKokkosViewHost F_NzLeft_h("F_NzLeft_h", Fm);                   // split each row of F into Left, Diag, Right. We only need to record #nz in Left and Diag.
    MatRowMapType          *Fdi = Fdi_h.data(), *Foi = Foi_h.data();
    MatColIdxType          *F_NzLeft = F_NzLeft_h.data();

    Fdi[0] = Foi[0] = 0;
    for (PetscInt i = 0; i < Fm; i++) {
      PetscInt *first, *last, *lb1, *lb2;
      // cut the row into: Left, [cstart, cend), Right
      first       = Fj + Fi[i];
      last        = Fj + Fi[i + 1];
      lb1         = std::lower_bound(first, last, cstart);
      F_NzLeft[i] = lb1 - first;
      lb2         = std::lower_bound(first, last, cend);
      Fdi[i + 1]  = lb2 - lb1;                        // row i length in Fdi
      Foi[i + 1]  = (Fi[i + 1] - Fi[i]) - Fdi[i + 1]; // row i length in Foi
    }
    for (PetscInt i = 0; i < Fm; i++) {
      Fdi[i + 1] += Fdi[i];
      Foi[i + 1] += Foi[i];
    }

    // Fill Fdj[] and Foj[], i.e., columns of Fd and Fo. Fdj[] are local, but Foj[] are not yet.
    PetscInt                Fdnz = Fdi[Fm], Fonz = Foi[Fm];
    MatColIdxKokkosViewHost Fdj_h("Fdj_h", Fdnz), Foj_h("Foj_h", Fonz);
    MatColIdxType          *Fdj = Fdj_h.data(), *Foj = Foj_h.data(), gid;

    for (PetscInt i = 0; i < Fm; i++) {
      PetscInt nzLeft = F_NzLeft[i];
      PetscInt len    = Fdi[i + 1] - Fdi[i]; // diag row len
      for (PetscInt j = 0; j < Fi[i + 1] - Fi[i]; j++) {
        gid = Fj[Fi[i] + j];
        if (j < nzLeft) { // left, in global
          Foj[Foi[i] + j] = gid;
        } else if (j < nzLeft + len) { // diag, in local
          Fdj[Fdi[i] + j - nzLeft] = gid - cstart;
        } else { // right, in global
          Foj[Foi[i] + j - len] = gid;
        }
      }
    }
    PetscCall(PetscFree2(jbuf, Fj));
    PetscCall(PetscFree(Fi));

    // Reduce global indices in Foj[] and garray1[] into local ones
    PetscInt n2, *garray2;
    PetscCall(ReduceTwoSetsOfGlobalIndices(n1, garray1, Fonz, Foj, &n2, &garray2, map));

    // Record the plans built above, for reuse
    PetscIntKokkosViewHost tmp(const_cast<PetscInt *>(irootloc), ioffset[niranks]); // irootloc[] is owned by ownerSF. We create a copy for safety
    PetscIntKokkosViewHost irootloc_h("irootloc_h", ioffset[niranks]);
    Kokkos::deep_copy(irootloc_h, tmp);
    mm->sf        = bcastSF;
    mm->E_NzLeft  = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), E_NzLeft_h);
    mm->F_NzLeft  = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), F_NzLeft_h);
    mm->irootloc  = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), irootloc_h);
    mm->rowoffset = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), rowoffset_h);
    mm->rootBuf   = MatScalarKokkosView("rootBuf", nroots);
    mm->leafBuf   = MatScalarKokkosView("leafBuf", nleaves);
    mm->garray    = garray2;
    mm->n         = n2;

    // Output Fd and Fo in KokkosCsrMatrix format
    MatScalarKokkosView Fda_d("Fda_d", Fdnz), Foa_d("Foa_d", Fonz);
    MatRowMapKokkosView Fdi_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), Fdi_h);
    MatColIdxKokkosView Fdj_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), Fdj_h);
    MatRowMapKokkosView Foi_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), Foi_h);
    MatColIdxKokkosView Foj_d = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), Foj_h);

    PetscCallCXX(mm->Fd = KokkosCsrMatrix("Fd", Fm, cend - cstart, Fdnz, Fda_d, Fdi_d, Fdj_d));
    PetscCallCXX(mm->Fo = KokkosCsrMatrix("Fo", Fm, n2, Fonz, Foa_d, Foi_d, Foj_d));

    // Compute kernel launch parameters in merging E or splitting F
    PetscInt teamSize, vectorLength, rowsPerTeam;

    teamSize = vectorLength = rowsPerTeam = -1;
    PetscCall(MatMergeGetLaunchParameters<DefaultExecutionSpace>(mm->irootloc.extent(0), mm->rootBuf.extent(0), -1, teamSize, vectorLength, rowsPerTeam));
    mm->E_TeamSize     = teamSize;
    mm->E_VectorLength = vectorLength;
    mm->E_RowsPerTeam  = rowsPerTeam;

    teamSize = vectorLength = rowsPerTeam = -1;
    PetscCall(MatMergeGetLaunchParameters<DefaultExecutionSpace>(Fm, Fnz, -1, teamSize, vectorLength, rowsPerTeam));
    mm->F_TeamSize     = teamSize;
    mm->F_VectorLength = vectorLength;
    mm->F_RowsPerTeam  = rowsPerTeam;
  } else PetscCheck(reuse == MAT_REUSE_MATRIX, comm, PETSC_ERR_PLIB, "Unsupported MatReuse enum %d", reuse);

  // Sync E's value to device
  akok->a_dual.sync_device();
  bkok->a_dual.sync_device();

  // Handy aliases
  const auto &Aa = akok->a_dual.view_device();
  const auto &Ba = bkok->a_dual.view_device();
  const auto &Ai = akok->i_dual.view_device();
  const auto &Bi = bkok->i_dual.view_device();

  // Fetch the plans
  PetscIntKokkosView  &E_NzLeft  = mm->E_NzLeft;
  PetscSF             &bcastSF   = mm->sf;
  MatScalarKokkosView &rootBuf   = mm->rootBuf;
  MatScalarKokkosView &leafBuf   = mm->leafBuf;
  PetscIntKokkosView  &irootloc  = mm->irootloc;
  PetscIntKokkosView  &rowoffset = mm->rowoffset;

  PetscInt teamSize     = mm->E_TeamSize;
  PetscInt vectorLength = mm->E_VectorLength;
  PetscInt rowsPerTeam  = mm->E_RowsPerTeam;
  PetscInt workSets     = (irootloc.extent(0) + rowsPerTeam - 1) / rowsPerTeam;

  // Copy rows in A/B of E to rootBuf, then bcast it to leafBuf
  PetscCallCXX(Kokkos::parallel_for(
    Kokkos::TeamPolicy<>(workSets, teamSize, vectorLength), KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, rowsPerTeam), [&](PetscInt k) {
        size_t r = t.league_rank() * rowsPerTeam + k; // r-th entry in irootloc[]
        if (r < irootloc.extent(0)) {
          PetscInt i      = irootloc(r); // row i of E
          PetscInt disp   = rowoffset(r);
          PetscInt alen   = Ai(i + 1) - Ai(i);
          PetscInt blen   = Bi(i + 1) - Bi(i);
          PetscInt nzleft = E_NzLeft(i);

          Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, alen + blen), [&](PetscInt j) {
            if (j < nzleft) { // B left
              rootBuf(disp + j) = Ba(Bi(i) + j);
            } else if (j < nzleft + alen) { // diag A
              rootBuf(disp + j) = Aa(Ai(i) + j - nzleft);
            } else { // B right
              rootBuf(disp + j) = Ba(Bi(i) + j - alen);
            }
          });
        }
      });
    }));
  PetscCall(PetscSFBcastWithMemTypeBegin(bcastSF, MPIU_SCALAR, PETSC_MEMTYPE_KOKKOS, rootBuf.data(), PETSC_MEMTYPE_KOKKOS, leafBuf.data(), MPI_REPLACE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// To finish MatMPIAIJKokkosBcast.
static PetscErrorCode MatMPIAIJKokkosBcastEnd(Mat E, PetscSF ownerSF, MatReuse reuse, PetscInt *map, MatMatStruct_AB *mm)
{
  PetscFunctionBegin;
  const auto &Fd  = mm->Fd;
  const auto &Fo  = mm->Fo;
  const auto &Fdi = Fd.graph.row_map;
  const auto &Foi = Fo.graph.row_map;
  auto       &Fda = Fd.values;
  auto       &Foa = Fo.values;
  auto        Fm  = Fd.numRows();

  PetscIntKokkosView  &F_NzLeft     = mm->F_NzLeft;
  PetscSF             &bcastSF      = mm->sf;
  MatScalarKokkosView &rootBuf      = mm->rootBuf;
  MatScalarKokkosView &leafBuf      = mm->leafBuf;
  PetscInt             teamSize     = mm->F_TeamSize;
  PetscInt             vectorLength = mm->F_VectorLength;
  PetscInt             rowsPerTeam  = mm->F_RowsPerTeam;
  PetscInt             workSets     = (Fm + rowsPerTeam - 1) / rowsPerTeam;

  PetscCall(PetscSFBcastEnd(bcastSF, MPIU_SCALAR, rootBuf.data(), leafBuf.data(), MPI_REPLACE));

  // Update Fda and Foa with new data in leafBuf (as if it is Fa)
  PetscCallCXX(Kokkos::parallel_for(
    Kokkos::TeamPolicy<>(workSets, teamSize, vectorLength), KOKKOS_LAMBDA(const KokkosTeamMemberType &t) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(t, 0, rowsPerTeam), [&](PetscInt k) {
        PetscInt i = t.league_rank() * rowsPerTeam + k; // i-th row in F
        if (i < Fm) {
          PetscInt nzLeft = F_NzLeft(i);
          PetscInt alen   = Fdi(i + 1) - Fdi(i);
          PetscInt blen   = Foi(i + 1) - Foi(i);
          PetscInt Fii    = Fdi(i) + Foi(i);

          Kokkos::parallel_for(Kokkos::ThreadVectorRange(t, alen + blen), [&](PetscInt j) {
            PetscScalar val = leafBuf(Fii + j);
            if (j < nzLeft) { // left
              Foa(Foi(i) + j) = val;
            } else if (j < nzLeft + alen) { // diag
              Fda(Fdi(i) + j - nzLeft) = val;
            } else { // right
              Foa(Foi(i) + j - alen) = val;
            }
          });
        }
      });
    }));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSymbolic_MPIAIJKokkos_AtB(Mat_Product *product, Mat A, Mat B, MatMatStruct_AtB *mm)
{
  Mat_MPIAIJ     *ampi = static_cast<Mat_MPIAIJ *>(A->data);
  Mat_MPIAIJ     *bmpi = static_cast<Mat_MPIAIJ *>(B->data);
  KokkosCsrMatrix Adt, Aot, Ad, Ao, Bd, Bo;
  PetscInt        cstart, cend;
  MPI_Comm        comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)B, &comm));
  PetscCall(MatSeqAIJKokkosGenerateTranspose_Private(ampi->A, &Adt));
  PetscCall(MatSeqAIJKokkosGenerateTranspose_Private(ampi->B, &Aot));
  PetscCall(MatSeqAIJKokkosGetKokkosCsrMatrix(ampi->A, &Ad));
  PetscCall(MatSeqAIJKokkosGetKokkosCsrMatrix(ampi->B, &Ao));
  PetscCall(MatSeqAIJKokkosGetKokkosCsrMatrix(bmpi->A, &Bd));
  PetscCall(MatSeqAIJKokkosGetKokkosCsrMatrix(bmpi->B, &Bo));

  // TODO: add command line options to select spgemm algorithms
  auto spgemm_alg = KokkosSparse::SPGEMMAlgorithm::SPGEMM_DEFAULT; // default is TPL if enabled, otherwise KK

  // CUDA-10.2's spgemm has bugs. We prefer the SpGEMMreuse APIs introduced in cuda-11.4
#if defined(KOKKOSKERNELS_ENABLE_TPL_CUSPARSE)
  #if PETSC_PKG_CUDA_VERSION_LT(11, 4, 0)
  spgemm_alg = KokkosSparse::SPGEMMAlgorithm::SPGEMM_KK;
  #endif
#endif

  PetscCallCXX(mm->kh1.create_spgemm_handle(spgemm_alg));
  PetscCallCXX(mm->kh2.create_spgemm_handle(spgemm_alg));
  PetscCallCXX(mm->kh3.create_spgemm_handle(spgemm_alg));
  PetscCallCXX(mm->kh4.create_spgemm_handle(spgemm_alg));

  // Aot * (B's diag + B's off-diag)
  PetscCallCXX(KokkosSparse::spgemm_symbolic(mm->kh3, Aot, false, Bd, false, mm->C3));
  PetscCallCXX(KokkosSparse::spgemm_symbolic(mm->kh4, Aot, false, Bo, false, mm->C4));
  // KK spgemm_symbolic() only populates the result's row map, but not its columns.
  // TODO: Remove the fake spgemm_numeric() after KK fixed this problem.
  PetscCallCXX(KokkosSparse::spgemm_numeric(mm->kh3, Aot, false, Bd, false, mm->C3));
  PetscCallCXX(KokkosSparse::spgemm_numeric(mm->kh4, Aot, false, Bo, false, mm->C4));
#if PETSC_PKG_KOKKOS_KERNELS_VERSION_LT(4, 0, 0)
  PetscCallCXX(sort_crs_matrix(mm->C3));
  PetscCallCXX(sort_crs_matrix(mm->C4));
#endif

  // Reduce E (i.e., C3 and C4)'s rows to form F, and overlap the communication
  PetscIntKokkosViewHost map_h("map_h", bmpi->B->cmap->n);
  PetscCall(MatGetOwnershipRangeColumn(B, &cstart, &cend));
  PetscCall(MatMPIAIJKokkosReduceBegin(comm, mm->C3, mm->C4, cstart, cend, bmpi->garray, ampi->Mvctx, MAT_INITIAL_MATRIX, map_h.data(), mm));

  // Adt * (B's diag + B's off-diag)
  PetscCallCXX(KokkosSparse::spgemm_symbolic(mm->kh1, Adt, false, Bd, false, mm->C1));
  PetscCallCXX(KokkosSparse::spgemm_symbolic(mm->kh2, Adt, false, Bo, false, mm->C2_mid));
  PetscCallCXX(KokkosSparse::spgemm_numeric(mm->kh1, Adt, false, Bd, false, mm->C1));
  PetscCallCXX(KokkosSparse::spgemm_numeric(mm->kh2, Adt, false, Bo, false, mm->C2_mid));
#if PETSC_PKG_KOKKOS_KERNELS_VERSION_LT(4, 0, 0)
  PetscCallCXX(sort_crs_matrix(mm->C1));
  PetscCallCXX(sort_crs_matrix(mm->C2_mid));
#endif

  PetscCall(MatMPIAIJKokkosReduceEnd(comm, mm->C3, mm->C4, cstart, cend, bmpi->garray, ampi->Mvctx, MAT_INITIAL_MATRIX, map_h.data(), mm));

  // Create C2, which shares a, i arrays with C2_mid, but with new column indices and potentially larger column size
  MatColIdxKokkosView oldj = mm->C2_mid.graph.entries, newj("j", oldj.extent(0));
  PetscIntKokkosView  map  = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), map_h);
  PetscCallCXX(Kokkos::parallel_for(
    oldj.extent(0), KOKKOS_LAMBDA(const PetscInt i) { newj(i) = map(oldj(i)); }));
  PetscCallCXX(mm->C2 = KokkosCsrMatrix("C2", mm->C2_mid.numRows(), mm->n /*new column size*/, mm->C2_mid.nnz(), mm->C2_mid.values, mm->C2_mid.graph.row_map, newj));

  // C = (C1+Fd, C2+Fo)
  PetscCallCXX(mm->kh1.create_spadd_handle(true)); // C1, Fd are sorted
  PetscCallCXX(mm->kh2.create_spadd_handle(true)); // C2, Fo are sorted
  PetscCallCXX(KokkosSparse::spadd_symbolic(&mm->kh1, mm->C1, mm->Fd, mm->Cd));
  PetscCallCXX(KokkosSparse::spadd_symbolic(&mm->kh2, mm->C2, mm->Fo, mm->Co));
  PetscCallCXX(KokkosSparse::spadd_numeric(&mm->kh1, 1.0, mm->C1, 1.0, mm->Fd, mm->Cd));
  PetscCallCXX(KokkosSparse::spadd_numeric(&mm->kh2, 1.0, mm->C2, 1.0, mm->Fo, mm->Co));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductNumeric_MPIAIJKokkos_AtB(Mat_Product *product, Mat A, Mat B, MatMatStruct_AtB *mm)
{
  Mat_MPIAIJ     *ampi = static_cast<Mat_MPIAIJ *>(A->data);
  Mat_MPIAIJ     *bmpi = static_cast<Mat_MPIAIJ *>(B->data);
  KokkosCsrMatrix Adt, Aot, Bd, Bo;
  MPI_Comm        comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)B, &comm));
  PetscCall(MatSeqAIJKokkosGenerateTranspose_Private(ampi->A, &Adt));
  PetscCall(MatSeqAIJKokkosGenerateTranspose_Private(ampi->B, &Aot));
  PetscCall(MatSeqAIJKokkosGetKokkosCsrMatrix(bmpi->A, &Bd));
  PetscCall(MatSeqAIJKokkosGetKokkosCsrMatrix(bmpi->B, &Bo));

  // Aot * (B's diag + B's off-diag)
  PetscCallCXX(KokkosSparse::spgemm_numeric(mm->kh3, Aot, false, Bd, false, mm->C3));
  PetscCallCXX(KokkosSparse::spgemm_numeric(mm->kh4, Aot, false, Bo, false, mm->C4));

  // Reduce E (i.e., C3 and C4)'s rows to form F, and overlap the communication
  PetscCall(MatMPIAIJKokkosReduceBegin(comm, mm->C3, mm->C4, 0, 0, NULL, NULL, MAT_REUSE_MATRIX, NULL, mm));

  // Adt * (B's diag + B's off-diag)
  PetscCallCXX(KokkosSparse::spgemm_numeric(mm->kh1, Adt, false, Bd, false, mm->C1));
  PetscCallCXX(KokkosSparse::spgemm_numeric(mm->kh2, Adt, false, Bo, false, mm->C2_mid));

  PetscCall(MatMPIAIJKokkosReduceEnd(comm, mm->C3, mm->C4, 0, 0, NULL, NULL, MAT_REUSE_MATRIX, NULL, mm));

  // C = (C1+Fd, C2+Fo)
  PetscCallCXX(KokkosSparse::spadd_numeric(&mm->kh1, 1.0, mm->C1, 1.0, mm->Fd, mm->Cd));
  PetscCallCXX(KokkosSparse::spadd_numeric(&mm->kh2, 1.0, mm->C2, 1.0, mm->Fo, mm->Co));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* MatProductSymbolic_MPIAIJKokkos_AB - AB flavor of MatProductSymbolic_MPIAIJKokkos

  Input Parameters:
+  product  - Mat_Product which carried out the computation. Passed in to access info about this mat product.
.  A        - an MPIAIJKOKKOS matrix
.  B        - an MPIAIJKOKKOS matrix
-  mm       - a struct used to stash intermediate data when computing AB. Persist from symbolic to numeric operations.
*/
static PetscErrorCode MatProductSymbolic_MPIAIJKokkos_AB(Mat_Product *product, Mat A, Mat B, MatMatStruct_AB *mm)
{
  Mat_MPIAIJ     *ampi = static_cast<Mat_MPIAIJ *>(A->data);
  Mat_MPIAIJ     *bmpi = static_cast<Mat_MPIAIJ *>(B->data);
  KokkosCsrMatrix Ad, Ao, Bd, Bo;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJKokkosGetKokkosCsrMatrix(ampi->A, &Ad));
  PetscCall(MatSeqAIJKokkosGetKokkosCsrMatrix(ampi->B, &Ao));
  PetscCall(MatSeqAIJKokkosGetKokkosCsrMatrix(bmpi->A, &Bd));
  PetscCall(MatSeqAIJKokkosGetKokkosCsrMatrix(bmpi->B, &Bo));

  // TODO: add command line options to select spgemm algorithms
  auto spgemm_alg = KokkosSparse::SPGEMMAlgorithm::SPGEMM_DEFAULT; // default is TPL if enabled, otherwise KK

  // CUDA-10.2's spgemm has bugs. We prefer the SpGEMMreuse APIs introduced in cuda-11.4
#if defined(KOKKOSKERNELS_ENABLE_TPL_CUSPARSE)
  #if PETSC_PKG_CUDA_VERSION_LT(11, 4, 0)
  spgemm_alg = KokkosSparse::SPGEMMAlgorithm::SPGEMM_KK;
  #endif
#endif

  mm->kh1.create_spgemm_handle(spgemm_alg);
  mm->kh2.create_spgemm_handle(spgemm_alg);
  mm->kh3.create_spgemm_handle(spgemm_alg);
  mm->kh4.create_spgemm_handle(spgemm_alg);

  // Bcast B's rows to form F, and overlap the communication
  PetscIntKokkosViewHost map_h("map_h", bmpi->B->cmap->n);
  PetscCall(MatMPIAIJKokkosBcastBegin(B, ampi->Mvctx, MAT_INITIAL_MATRIX, map_h.data(), mm));

  // A's diag * (B's diag + B's off-diag)
  PetscCallCXX(KokkosSparse::spgemm_symbolic(mm->kh1, Ad, false, Bd, false, mm->C1));
  PetscCallCXX(KokkosSparse::spgemm_symbolic(mm->kh2, Ad, false, Bo, false, mm->C2_mid)); // C2 aliases with C2_mid, except with new column indices
  // KK spgemm_symbolic() only populates the result's row map, but not its columns.
  // TODO: Remove the fake spgemm_numeric() after KK fixed this problem.
  PetscCallCXX(KokkosSparse::spgemm_numeric(mm->kh1, Ad, false, Bd, false, mm->C1));
  PetscCallCXX(KokkosSparse::spgemm_numeric(mm->kh2, Ad, false, Bo, false, mm->C2_mid));
#if PETSC_PKG_KOKKOS_KERNELS_VERSION_LT(4, 0, 0)
  PetscCallCXX(sort_crs_matrix(mm->C1));
  PetscCallCXX(sort_crs_matrix(mm->C2_mid));
#endif

  PetscCall(MatMPIAIJKokkosBcastEnd(B, ampi->Mvctx, MAT_INITIAL_MATRIX, map_h.data(), mm));

  // A's off-diag * (F's diag + F's off-diag)
  PetscCallCXX(KokkosSparse::spgemm_symbolic(mm->kh3, Ao, false, mm->Fd, false, mm->C3));
  PetscCallCXX(KokkosSparse::spgemm_numeric(mm->kh3, Ao, false, mm->Fd, false, mm->C3));
  PetscCallCXX(KokkosSparse::spgemm_symbolic(mm->kh4, Ao, false, mm->Fo, false, mm->C4));
  PetscCallCXX(KokkosSparse::spgemm_numeric(mm->kh4, Ao, false, mm->Fo, false, mm->C4));
#if PETSC_PKG_KOKKOS_KERNELS_VERSION_LT(4, 0, 0)
  PetscCallCXX(sort_crs_matrix(mm->C3));
  PetscCallCXX(sort_crs_matrix(mm->C4));
#endif

  // Create C2, which shares a, i arrays with C2_mid, but with new column indices and potentially larger column size
  MatColIdxKokkosView oldj = mm->C2_mid.graph.entries, newj("j", oldj.extent(0));
  PetscIntKokkosView  map  = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), map_h);
  PetscCallCXX(Kokkos::parallel_for(
    oldj.extent(0), KOKKOS_LAMBDA(const PetscInt i) { newj(i) = map(oldj(i)); }));
  mm->C2 = KokkosCsrMatrix("C2", mm->C2_mid.numRows(), mm->n /*new column size*/, mm->C2_mid.nnz(), mm->C2_mid.values, mm->C2_mid.graph.row_map, newj);

  // C = (Cd, Co) = (C1+C3, C2+C4)
  mm->kh1.create_spadd_handle(true); // C1, C3 are sorted
  mm->kh2.create_spadd_handle(true); // C2, C4 are sorted
  PetscCallCXX(KokkosSparse::spadd_symbolic(&mm->kh1, mm->C1, mm->C3, mm->Cd));
  PetscCallCXX(KokkosSparse::spadd_symbolic(&mm->kh2, mm->C2, mm->C4, mm->Co));
  PetscCallCXX(KokkosSparse::spadd_numeric(&mm->kh1, 1.0, mm->C1, 1.0, mm->C3, mm->Cd));
  PetscCallCXX(KokkosSparse::spadd_numeric(&mm->kh2, 1.0, mm->C2, 1.0, mm->C4, mm->Co));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductNumeric_MPIAIJKokkos_AB(Mat_Product *product, Mat A, Mat B, MatMatStruct_AB *mm)
{
  Mat_MPIAIJ     *ampi = static_cast<Mat_MPIAIJ *>(A->data);
  Mat_MPIAIJ     *bmpi = static_cast<Mat_MPIAIJ *>(B->data);
  KokkosCsrMatrix Ad, Ao, Bd, Bo;

  PetscFunctionBegin;
  PetscCall(MatSeqAIJKokkosGetKokkosCsrMatrix(ampi->A, &Ad));
  PetscCall(MatSeqAIJKokkosGetKokkosCsrMatrix(ampi->B, &Ao));
  PetscCall(MatSeqAIJKokkosGetKokkosCsrMatrix(bmpi->A, &Bd));
  PetscCall(MatSeqAIJKokkosGetKokkosCsrMatrix(bmpi->B, &Bo));

  // Bcast B's rows to form F, and overlap the communication
  PetscCall(MatMPIAIJKokkosBcastBegin(B, NULL, MAT_REUSE_MATRIX, NULL, mm));

  // A's diag * (B's diag + B's off-diag)
  PetscCallCXX(KokkosSparse::spgemm_numeric(mm->kh1, Ad, false, Bd, false, mm->C1));
  PetscCallCXX(KokkosSparse::spgemm_numeric(mm->kh2, Ad, false, Bo, false, mm->C2_mid));

  PetscCall(MatMPIAIJKokkosBcastEnd(B, NULL, MAT_REUSE_MATRIX, NULL, mm));

  // A's off-diag * (F's diag + F's off-diag)
  PetscCallCXX(KokkosSparse::spgemm_numeric(mm->kh3, Ao, false, mm->Fd, false, mm->C3));
  PetscCallCXX(KokkosSparse::spgemm_numeric(mm->kh4, Ao, false, mm->Fo, false, mm->C4));

  // C = (Cd, Co) = (C1+C3, C2+C4)
  PetscCallCXX(KokkosSparse::spadd_numeric(&mm->kh1, 1.0, mm->C1, 1.0, mm->C3, mm->Cd));
  PetscCallCXX(KokkosSparse::spadd_numeric(&mm->kh2, 1.0, mm->C2, 1.0, mm->C4, mm->Co));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatProductNumeric_MPIAIJKokkos(Mat C)
{
  Mat_MPIAIJ                  *cmpi = static_cast<Mat_MPIAIJ *>(C->data);
  Mat_Product                 *product;
  MatProductData_MPIAIJKokkos *pdata;
  MatProductType               ptype;
  Mat                          A, B;

  PetscFunctionBegin;
  MatCheckProduct(C, 1); // make sure C is a product
  product = C->product;
  pdata   = static_cast<MatProductData_MPIAIJKokkos *>(product->data);
  ptype   = product->type;
  A       = product->A;
  B       = product->B;

  // See if numeric has already been done in symbolic (e.g., user calls MatMatMult(A,B,MAT_INITIAL_MATRIX,..,C)).
  // If yes, skip the numeric, but reset the flag so that next time when user calls MatMatMult(E,F,MAT_REUSE_MATRIX,..,C),
  // we still do numeric.
  if (pdata->reusesym) { // numeric reuses results from symbolic
    pdata->reusesym = PETSC_FALSE;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if (ptype == MATPRODUCT_AB) {
    PetscCall(MatProductNumeric_MPIAIJKokkos_AB(product, A, B, pdata->mmAB));
  } else if (ptype == MATPRODUCT_AtB) {
    PetscCall(MatProductNumeric_MPIAIJKokkos_AtB(product, A, B, pdata->mmAtB));
  } else if (ptype == MATPRODUCT_PtAP) { // BtAB, computed by Z = AB; C= BtZ
    PetscCall(MatProductNumeric_MPIAIJKokkos_AB(product, A, B, pdata->mmAB));
    PetscCall(MatProductNumeric_MPIAIJKokkos_AtB(product, B, pdata->Z, pdata->mmAtB));
  }

  PetscCall(MatSeqAIJKokkosModifyDevice(cmpi->A)); // mark that A, B on device are modified
  PetscCall(MatSeqAIJKokkosModifyDevice(cmpi->B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatProductSymbolic_MPIAIJKokkos(Mat C)
{
  Mat                          A, B;
  Mat_Product                 *product;
  MatProductType               ptype;
  MatProductData_MPIAIJKokkos *pdata;
  MatMatStruct                *mm = NULL;
  PetscInt                     m, n, M, N;
  Mat                          Cd, Co;
  MPI_Comm                     comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)C, &comm));
  MatCheckProduct(C, 1);
  product = C->product;
  PetscCheck(!product->data, comm, PETSC_ERR_PLIB, "Product data not empty");
  ptype = product->type;
  A     = product->A;
  B     = product->B;

  switch (ptype) {
  case MATPRODUCT_AB:
    m = A->rmap->n;
    n = B->cmap->n;
    M = A->rmap->N;
    N = B->cmap->N;
    break;
  case MATPRODUCT_AtB:
    m = A->cmap->n;
    n = B->cmap->n;
    M = A->cmap->N;
    N = B->cmap->N;
    break;
  case MATPRODUCT_PtAP:
    m = B->cmap->n;
    n = B->cmap->n;
    M = B->cmap->N;
    N = B->cmap->N;
    break; /* BtAB */
  default:
    SETERRQ(comm, PETSC_ERR_PLIB, "Not for product type %s", MatProductTypes[ptype]);
  }

  PetscCall(MatSetSizes(C, m, n, M, N));
  PetscCall(PetscLayoutSetUp(C->rmap));
  PetscCall(PetscLayoutSetUp(C->cmap));
  PetscCall(MatSetType(C, ((PetscObject)A)->type_name));

  pdata           = new MatProductData_MPIAIJKokkos();
  pdata->reusesym = product->api_user;

  if (ptype == MATPRODUCT_AB) {
    auto mmAB = new MatMatStruct_AB();
    PetscCall(MatProductSymbolic_MPIAIJKokkos_AB(product, A, B, mmAB));
    mm = pdata->mmAB = mmAB;
  } else if (ptype == MATPRODUCT_AtB) {
    auto mmAtB = new MatMatStruct_AtB();
    PetscCall(MatProductSymbolic_MPIAIJKokkos_AtB(product, A, B, mmAtB));
    mm = pdata->mmAtB = mmAtB;
  } else if (ptype == MATPRODUCT_PtAP) { // C = BtAB, computed as Z = AB; C= BtZ
    Mat Zd, Zo, Z;                       // Zd, Zo are owned by pdata->Z

    auto mmAB = new MatMatStruct_AB();
    PetscCall(MatProductSymbolic_MPIAIJKokkos_AB(product, A, B, mmAB)); // Z stored as mmAB->{Cd, Co}
    PetscCall(MatCreateSeqAIJKokkosWithKokkosCsrMatrix(PETSC_COMM_SELF, mmAB->Cd, &Zd));
    PetscCall(MatCreateSeqAIJKokkosWithKokkosCsrMatrix(PETSC_COMM_SELF, mmAB->Co, &Zo));
    pdata->mmAB = mmAB;

    m = A->rmap->n; // Z's layout
    n = B->cmap->n;
    M = A->rmap->N;
    N = B->cmap->N;
    PetscCall(MatCreate(comm, &Z));
    PetscCall(MatSetSizes(Z, m, n, M, N));
    PetscCall(PetscLayoutSetUp(Z->rmap));
    PetscCall(PetscLayoutSetUp(Z->cmap));
    PetscCall(MatSetType(Z, MATMPIAIJKOKKOS));
    PetscCall(MatSetMPIAIJKokkosWithSplitSeqAIJKokkosMatrices(Z, Zd, Zo, mmAB->garray));

    auto mmAtB = new MatMatStruct_AtB();
    PetscCall(MatProductSymbolic_MPIAIJKokkos_AtB(product, B, Z, mmAtB)); // final result C stored as mmAtB->{Cd, Co}

    pdata->Z = Z; // give ownership to pdata
    mm = pdata->mmAtB = mmAtB;
  }

  PetscCall(MatCreateSeqAIJKokkosWithKokkosCsrMatrix(PETSC_COMM_SELF, mm->Cd, &Cd));
  PetscCall(MatCreateSeqAIJKokkosWithKokkosCsrMatrix(PETSC_COMM_SELF, mm->Co, &Co));
  PetscCall(MatSetMPIAIJKokkosWithSplitSeqAIJKokkosMatrices(C, Cd, Co, mm->garray));

  C->product->data       = pdata;
  C->product->destroy    = MatProductDataDestroy_MPIAIJKokkos;
  C->ops->productnumeric = MatProductNumeric_MPIAIJKokkos;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_MPIAIJKokkos(Mat mat)
{
  Mat_Product *product = mat->product;
  PetscBool    match   = PETSC_FALSE;
  PetscBool    usecpu  = PETSC_FALSE;

  PetscFunctionBegin;
  MatCheckProduct(mat, 1);
  if (!product->A->boundtocpu && !product->B->boundtocpu) PetscCall(PetscObjectTypeCompare((PetscObject)product->B, ((PetscObject)product->A)->type_name, &match));
  if (match) { /* we can always fallback to the CPU if requested */
    switch (product->type) {
    case MATPRODUCT_AB:
      if (product->api_user) {
        PetscOptionsBegin(PetscObjectComm((PetscObject)mat), ((PetscObject)mat)->prefix, "MatMatMult", "Mat");
        PetscCall(PetscOptionsBool("-matmatmult_backend_cpu", "Use CPU code", "MatMatMult", usecpu, &usecpu, NULL));
        PetscOptionsEnd();
      } else {
        PetscOptionsBegin(PetscObjectComm((PetscObject)mat), ((PetscObject)mat)->prefix, "MatProduct_AB", "Mat");
        PetscCall(PetscOptionsBool("-mat_product_algorithm_backend_cpu", "Use CPU code", "MatMatMult", usecpu, &usecpu, NULL));
        PetscOptionsEnd();
      }
      break;
    case MATPRODUCT_AtB:
      if (product->api_user) {
        PetscOptionsBegin(PetscObjectComm((PetscObject)mat), ((PetscObject)mat)->prefix, "MatTransposeMatMult", "Mat");
        PetscCall(PetscOptionsBool("-mattransposematmult_backend_cpu", "Use CPU code", "MatTransposeMatMult", usecpu, &usecpu, NULL));
        PetscOptionsEnd();
      } else {
        PetscOptionsBegin(PetscObjectComm((PetscObject)mat), ((PetscObject)mat)->prefix, "MatProduct_AtB", "Mat");
        PetscCall(PetscOptionsBool("-mat_product_algorithm_backend_cpu", "Use CPU code", "MatTransposeMatMult", usecpu, &usecpu, NULL));
        PetscOptionsEnd();
      }
      break;
    case MATPRODUCT_PtAP:
      if (product->api_user) {
        PetscOptionsBegin(PetscObjectComm((PetscObject)mat), ((PetscObject)mat)->prefix, "MatPtAP", "Mat");
        PetscCall(PetscOptionsBool("-matptap_backend_cpu", "Use CPU code", "MatPtAP", usecpu, &usecpu, NULL));
        PetscOptionsEnd();
      } else {
        PetscOptionsBegin(PetscObjectComm((PetscObject)mat), ((PetscObject)mat)->prefix, "MatProduct_PtAP", "Mat");
        PetscCall(PetscOptionsBool("-mat_product_algorithm_backend_cpu", "Use CPU code", "MatPtAP", usecpu, &usecpu, NULL));
        PetscOptionsEnd();
      }
      break;
    default:
      break;
    }
    match = (PetscBool)!usecpu;
  }
  if (match) {
    switch (product->type) {
    case MATPRODUCT_AB:
    case MATPRODUCT_AtB:
    case MATPRODUCT_PtAP:
      mat->ops->productsymbolic = MatProductSymbolic_MPIAIJKokkos;
      break;
    default:
      break;
    }
  }
  /* fallback to MPIAIJ ops */
  if (!mat->ops->productsymbolic) PetscCall(MatProductSetFromOptions_MPIAIJ(mat));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetPreallocationCOO_MPIAIJKokkos(Mat mat, PetscCount coo_n, PetscInt coo_i[], PetscInt coo_j[])
{
  Mat_MPIAIJ       *mpiaij = (Mat_MPIAIJ *)mat->data;
  Mat_MPIAIJKokkos *mpikok;

  PetscFunctionBegin;
  PetscCall(MatSetPreallocationCOO_MPIAIJ(mat, coo_n, coo_i, coo_j)); /* mpiaij->A,B's type is set to seqaijkokkos */
  mat->preallocated = PETSC_TRUE;
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatZeroEntries(mat));
  mpikok = static_cast<Mat_MPIAIJKokkos *>(mpiaij->spptr);
  delete mpikok;
  mpiaij->spptr = new Mat_MPIAIJKokkos(mpiaij);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetValuesCOO_MPIAIJKokkos(Mat mat, const PetscScalar v[], InsertMode imode)
{
  Mat_MPIAIJ                 *mpiaij = static_cast<Mat_MPIAIJ *>(mat->data);
  Mat_MPIAIJKokkos           *mpikok = static_cast<Mat_MPIAIJKokkos *>(mpiaij->spptr);
  Mat                         A = mpiaij->A, B = mpiaij->B;
  PetscCount                  Annz = mpiaij->Annz, Annz2 = mpiaij->Annz2, Bnnz = mpiaij->Bnnz, Bnnz2 = mpiaij->Bnnz2;
  MatScalarKokkosView         Aa, Ba;
  MatScalarKokkosView         v1;
  MatScalarKokkosView        &vsend  = mpikok->sendbuf_d;
  const MatScalarKokkosView  &v2     = mpikok->recvbuf_d;
  const PetscCountKokkosView &Ajmap1 = mpikok->Ajmap1_d, Ajmap2 = mpikok->Ajmap2_d, Aimap2 = mpikok->Aimap2_d;
  const PetscCountKokkosView &Bjmap1 = mpikok->Bjmap1_d, Bjmap2 = mpikok->Bjmap2_d, Bimap2 = mpikok->Bimap2_d;
  const PetscCountKokkosView &Aperm1 = mpikok->Aperm1_d, Aperm2 = mpikok->Aperm2_d, Bperm1 = mpikok->Bperm1_d, Bperm2 = mpikok->Bperm2_d;
  const PetscCountKokkosView &Cperm1 = mpikok->Cperm1_d;
  PetscMemType                memtype;

  PetscFunctionBegin;
  PetscCall(PetscGetMemType(v, &memtype)); /* Return PETSC_MEMTYPE_HOST when v is NULL */
  if (PetscMemTypeHost(memtype)) {         /* If user gave v[] in host, we need to copy it to device if any */
    v1 = Kokkos::create_mirror_view_and_copy(DefaultMemorySpace(), MatScalarKokkosViewHost((PetscScalar *)v, mpiaij->coo_n));
  } else {
    v1 = MatScalarKokkosView((PetscScalar *)v, mpiaij->coo_n); /* Directly use v[]'s memory */
  }

  if (imode == INSERT_VALUES) {
    PetscCall(MatSeqAIJGetKokkosViewWrite(A, &Aa)); /* write matrix values */
    PetscCall(MatSeqAIJGetKokkosViewWrite(B, &Ba));
  } else {
    PetscCall(MatSeqAIJGetKokkosView(A, &Aa)); /* read & write matrix values */
    PetscCall(MatSeqAIJGetKokkosView(B, &Ba));
  }

  /* Pack entries to be sent to remote */
  Kokkos::parallel_for(
    vsend.extent(0), KOKKOS_LAMBDA(const PetscCount i) { vsend(i) = v1(Cperm1(i)); });

  /* Send remote entries to their owner and overlap the communication with local computation */
  PetscCall(PetscSFReduceWithMemTypeBegin(mpiaij->coo_sf, MPIU_SCALAR, PETSC_MEMTYPE_KOKKOS, vsend.data(), PETSC_MEMTYPE_KOKKOS, v2.data(), MPI_REPLACE));
  /* Add local entries to A and B in one kernel */
  Kokkos::parallel_for(
    Annz + Bnnz, KOKKOS_LAMBDA(PetscCount i) {
      PetscScalar sum = 0.0;
      if (i < Annz) {
        for (PetscCount k = Ajmap1(i); k < Ajmap1(i + 1); k++) sum += v1(Aperm1(k));
        Aa(i) = (imode == INSERT_VALUES ? 0.0 : Aa(i)) + sum;
      } else {
        i -= Annz;
        for (PetscCount k = Bjmap1(i); k < Bjmap1(i + 1); k++) sum += v1(Bperm1(k));
        Ba(i) = (imode == INSERT_VALUES ? 0.0 : Ba(i)) + sum;
      }
    });
  PetscCall(PetscSFReduceEnd(mpiaij->coo_sf, MPIU_SCALAR, vsend.data(), v2.data(), MPI_REPLACE));

  /* Add received remote entries to A and B in one kernel */
  Kokkos::parallel_for(
    Annz2 + Bnnz2, KOKKOS_LAMBDA(PetscCount i) {
      if (i < Annz2) {
        for (PetscCount k = Ajmap2(i); k < Ajmap2(i + 1); k++) Aa(Aimap2(i)) += v2(Aperm2(k));
      } else {
        i -= Annz2;
        for (PetscCount k = Bjmap2(i); k < Bjmap2(i + 1); k++) Ba(Bimap2(i)) += v2(Bperm2(k));
      }
    });

  if (imode == INSERT_VALUES) {
    PetscCall(MatSeqAIJRestoreKokkosViewWrite(A, &Aa)); /* Increase A & B's state etc. */
    PetscCall(MatSeqAIJRestoreKokkosViewWrite(B, &Ba));
  } else {
    PetscCall(MatSeqAIJRestoreKokkosView(A, &Aa));
    PetscCall(MatSeqAIJRestoreKokkosView(B, &Ba));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatDestroy_MPIAIJKokkos(Mat A)
{
  Mat_MPIAIJ *mpiaij = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMPIAIJSetPreallocation_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMPIAIJGetLocalMatMerge_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetPreallocationCOO_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetValuesCOO_C", NULL));
  delete (Mat_MPIAIJKokkos *)mpiaij->spptr;
  PetscCall(MatDestroy_MPIAIJ(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIAIJKokkos(Mat A, MatType mtype, MatReuse reuse, Mat *newmat)
{
  Mat         B;
  Mat_MPIAIJ *a;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, newmat));
  } else if (reuse == MAT_REUSE_MATRIX) {
    PetscCall(MatCopy(A, *newmat, SAME_NONZERO_PATTERN));
  }
  B = *newmat;

  B->boundtocpu = PETSC_FALSE;
  PetscCall(PetscFree(B->defaultvectype));
  PetscCall(PetscStrallocpy(VECKOKKOS, &B->defaultvectype));
  PetscCall(PetscObjectChangeTypeName((PetscObject)B, MATMPIAIJKOKKOS));

  a = static_cast<Mat_MPIAIJ *>(A->data);
  if (a->A) PetscCall(MatSetType(a->A, MATSEQAIJKOKKOS));
  if (a->B) PetscCall(MatSetType(a->B, MATSEQAIJKOKKOS));
  if (a->lvec) PetscCall(VecSetType(a->lvec, VECSEQKOKKOS));

  B->ops->assemblyend           = MatAssemblyEnd_MPIAIJKokkos;
  B->ops->mult                  = MatMult_MPIAIJKokkos;
  B->ops->multadd               = MatMultAdd_MPIAIJKokkos;
  B->ops->multtranspose         = MatMultTranspose_MPIAIJKokkos;
  B->ops->productsetfromoptions = MatProductSetFromOptions_MPIAIJKokkos;
  B->ops->destroy               = MatDestroy_MPIAIJKokkos;

  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMPIAIJSetPreallocation_C", MatMPIAIJSetPreallocation_MPIAIJKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatMPIAIJGetLocalMatMerge_C", MatMPIAIJGetLocalMatMerge_MPIAIJKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSetPreallocationCOO_C", MatSetPreallocationCOO_MPIAIJKokkos));
  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatSetValuesCOO_C", MatSetValuesCOO_MPIAIJKokkos));
  PetscFunctionReturn(PETSC_SUCCESS);
}
/*MC
   MATAIJKOKKOS - "mpiaijkokkos", a matrix type to be used for CSR sparse matrices with Kokkos

   A matrix type type using Kokkos-Kernels CrsMatrix type for portability across different device types

   Options Database Key:
.  -mat_type aijkokkos - sets the matrix type to `MATAIJKOKKOS`

  Level: beginner

.seealso: [](chapter_matrices), `Mat`, `MatCreateAIJKokkos()`, `MATSEQAIJKOKKOS`, `MATSEQAIJ`, `MATMPIAIJ`
M*/
PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJKokkos(Mat A)
{
  PetscFunctionBegin;
  PetscCall(PetscKokkosInitializeCheck());
  PetscCall(MatCreate_MPIAIJ(A));
  PetscCall(MatConvert_MPIAIJ_MPIAIJKokkos(A, MATMPIAIJKOKKOS, MAT_INPLACE_MATRIX, &A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
   MatCreateAIJKokkos - Creates a sparse matrix in `MATAIJKOKOS` (compressed row) format
   (the default parallel PETSc format).  This matrix will ultimately pushed down
   to Kokkos for calculations.

   Collective

   Input Parameters:
+  comm - MPI communicator, set to `PETSC_COMM_SELF`
.  m - number of local rows (or `PETSC_DECIDE` to have calculated if `M` is given)
           This value should be the same as the local size used in creating the
           y vector for the matrix-vector product y = Ax.
.  n - This value should be the same as the local size used in creating the
       x vector for the matrix-vector product y = Ax. (or `PETSC_DECIDE` to have
       calculated if N is given) For square matrices n is almost always `m`.
.  M - number of global rows (or `PETSC_DETERMINE` to have calculated if `m` is given)
.  N - number of global columns (or `PETSC_DETERMINE` to have calculated if `n` is given)
.  d_nz  - number of nonzeros per row in DIAGONAL portion of local submatrix
           (same value is used for all local rows)
.  d_nnz - array containing the number of nonzeros in the various rows of the
           DIAGONAL portion of the local submatrix (possibly different for each row)
           or `NULL`, if `d_nz` is used to specify the nonzero structure.
           The size of this array is equal to the number of local rows, i.e `m`.
           For matrices you plan to factor you must leave room for the diagonal entry and
           put in the entry even if it is zero.
.  o_nz  - number of nonzeros per row in the OFF-DIAGONAL portion of local
           submatrix (same value is used for all local rows).
-  o_nnz - array containing the number of nonzeros in the various rows of the
           OFF-DIAGONAL portion of the local submatrix (possibly different for
           each row) or `NULL`, if `o_nz` is used to specify the nonzero
           structure. The size of this array is equal to the number
           of local rows, i.e `m`.

   Output Parameter:
.  A - the matrix

   Level: intermediate

   Notes:
   It is recommended that one use the `MatCreate()`, `MatSetType()` and/or `MatSetFromOptions()`,
   MatXXXXSetPreallocation() paradigm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, `MatSeqAIJSetPreallocation()`]

   The AIJ format, also called compressed row storage), is fully compatible with standard Fortran
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.

.seealso: [](chapter_matrices), `Mat`, `MATAIJKOKOS`, `MATSEQAIJKOKOS`, `MATMPIAIJKOKOS`, `MatCreate()`, `MatCreateAIJ()`, `MatSetValues()`,
          `MatSeqAIJSetColumnIndices()`, `MatCreateSeqAIJWithArrays()`, `MatCreateAIJ()`, `MATMPIAIJKOKKOS`, `MATAIJKOKKOS`
@*/
PetscErrorCode MatCreateAIJKokkos(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[], Mat *A)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSizes(*A, m, n, M, N));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size > 1) {
    PetscCall(MatSetType(*A, MATMPIAIJKOKKOS));
    PetscCall(MatMPIAIJSetPreallocation(*A, d_nz, d_nnz, o_nz, o_nnz));
  } else {
    PetscCall(MatSetType(*A, MATSEQAIJKOKKOS));
    PetscCall(MatSeqAIJSetPreallocation(*A, d_nz, d_nnz));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// get GPU pointer to stripped down Mat. For both Seq and MPI Mat.
PetscErrorCode MatKokkosGetDeviceMatWrite(Mat A, PetscSplitCSRDataStructure *B)
{
  PetscMPIInt                size, rank;
  MPI_Comm                   comm;
  PetscSplitCSRDataStructure d_mat = NULL;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A, &comm));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  if (size == 1) {
    PetscCall(MatSeqAIJKokkosGetDeviceMat(A, &d_mat));
    PetscCall(MatSeqAIJKokkosModifyDevice(A)); /* Since we are going to modify matrix values on device */
  } else {
    Mat_MPIAIJ *aij = (Mat_MPIAIJ *)A->data;
    PetscCall(MatSeqAIJKokkosGetDeviceMat(aij->A, &d_mat));
    PetscCall(MatSeqAIJKokkosModifyDevice(aij->A));
    PetscCall(MatSeqAIJKokkosModifyDevice(aij->B));
    PetscCheck(A->nooffprocentries || aij->donotstash, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Device assembly does not currently support offproc values insertion. Use MatSetOption(A,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE) or MatSetOption(A,MAT_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE)");
  }
  // act like MatSetValues because not called on host
  if (A->assembled) {
    if (A->was_assembled) PetscCall(PetscInfo(A, "Assemble more than once already\n"));
    A->was_assembled = PETSC_TRUE; // this is done (lazy) in MatAssemble but we are not calling it anymore - done in AIJ AssemblyEnd, need here?
  } else {
    PetscCall(PetscInfo(A, "Warning !assemble ??? assembled=%" PetscInt_FMT "\n", A->assembled));
  }
  if (!d_mat) {
    struct _n_SplitCSRMat h_mat; /* host container */
    Mat_SeqAIJKokkos     *aijkokA;
    Mat_SeqAIJ           *jaca;
    PetscInt              n = A->rmap->n, nnz;
    Mat                   Amat;
    PetscInt             *colmap;

    /* create and copy h_mat */
    h_mat.M = A->cmap->N; // use for debug build
    PetscCall(PetscInfo(A, "Create device matrix in Kokkos\n"));
    if (size == 1) {
      Amat            = A;
      jaca            = (Mat_SeqAIJ *)A->data;
      h_mat.rstart    = 0;
      h_mat.rend      = A->rmap->n;
      h_mat.cstart    = 0;
      h_mat.cend      = A->cmap->n;
      h_mat.offdiag.i = h_mat.offdiag.j = NULL;
      h_mat.offdiag.a                   = NULL;
      aijkokA                           = static_cast<Mat_SeqAIJKokkos *>(A->spptr);
    } else {
      Mat_MPIAIJ       *aij  = (Mat_MPIAIJ *)A->data;
      Mat_SeqAIJ       *jacb = (Mat_SeqAIJ *)aij->B->data;
      PetscInt          ii;
      Mat_SeqAIJKokkos *aijkokB;

      Amat    = aij->A;
      aijkokA = static_cast<Mat_SeqAIJKokkos *>(aij->A->spptr);
      aijkokB = static_cast<Mat_SeqAIJKokkos *>(aij->B->spptr);
      jaca    = (Mat_SeqAIJ *)aij->A->data;
      PetscCheck(!aij->B->cmap->n || aij->garray, comm, PETSC_ERR_PLIB, "MPIAIJ Matrix was assembled but is missing garray");
      PetscCheck(aij->B->rmap->n == aij->A->rmap->n, comm, PETSC_ERR_SUP, "Only support aij->B->rmap->n == aij->A->rmap->n");
      aij->donotstash          = PETSC_TRUE;
      aij->A->nooffprocentries = aij->B->nooffprocentries = A->nooffprocentries = PETSC_TRUE;
      jaca->nonew = jacb->nonew = PETSC_TRUE; // no more disassembly
      PetscCall(PetscCalloc1(A->cmap->N, &colmap));
      for (ii = 0; ii < aij->B->cmap->n; ii++) colmap[aij->garray[ii]] = ii + 1;
      // allocate B copy data
      h_mat.rstart = A->rmap->rstart;
      h_mat.rend   = A->rmap->rend;
      h_mat.cstart = A->cmap->rstart;
      h_mat.cend   = A->cmap->rend;
      nnz          = jacb->i[n];
      if (jacb->compressedrow.use) {
        const Kokkos::View<PetscInt *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_i_k(jacb->i, n + 1);
        aijkokB->i_uncompressed_d = Kokkos::View<PetscInt *>(Kokkos::create_mirror(DefaultMemorySpace(), h_i_k));
        Kokkos::deep_copy(aijkokB->i_uncompressed_d, h_i_k);
        h_mat.offdiag.i = aijkokB->i_uncompressed_d.data();
      } else {
        h_mat.offdiag.i = aijkokB->i_device_data();
      }
      h_mat.offdiag.j = aijkokB->j_device_data();
      h_mat.offdiag.a = aijkokB->a_device_data();
      {
        Kokkos::View<PetscInt *, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> h_colmap_k(colmap, A->cmap->N);
        aijkokB->colmap_d = Kokkos::View<PetscInt *>(Kokkos::create_mirror(DefaultMemorySpace(), h_colmap_k));
        Kokkos::deep_copy(aijkokB->colmap_d, h_colmap_k);
        h_mat.colmap = aijkokB->colmap_d.data();
        PetscCall(PetscFree(colmap));
      }
      h_mat.offdiag.ignorezeroentries = jacb->ignorezeroentries;
      h_mat.offdiag.n                 = n;
    }
    // allocate A copy data
    nnz                          = jaca->i[n];
    h_mat.diag.n                 = n;
    h_mat.diag.ignorezeroentries = jaca->ignorezeroentries;
    PetscCallMPI(MPI_Comm_rank(comm, &h_mat.rank));
    PetscCheck(!jaca->compressedrow.use, PETSC_COMM_SELF, PETSC_ERR_PLIB, "A does not support compressed row (todo)");
    h_mat.diag.i = aijkokA->i_device_data();
    h_mat.diag.j = aijkokA->j_device_data();
    h_mat.diag.a = aijkokA->a_device_data();
    // copy pointers and metadata to device
    PetscCall(MatSeqAIJKokkosSetDeviceMat(Amat, &h_mat));
    PetscCall(MatSeqAIJKokkosGetDeviceMat(Amat, &d_mat));
    PetscCall(PetscInfo(A, "Create device Mat n=%" PetscInt_FMT " nnz=%" PetscInt_FMT "\n", h_mat.diag.n, nnz));
  }
  *B           = d_mat;       // return it, set it in Mat, and set it up
  A->assembled = PETSC_FALSE; // ready to write with matsetvalues - this done (lazy) in normal MatSetValues
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatSeqAIJKokkosGetOffloadMask(Mat A, const char **mask)
{
  Mat_SeqAIJKokkos *aijkok = static_cast<Mat_SeqAIJKokkos *>(A->spptr);

  PetscFunctionBegin;
  if (!aijkok) *mask = "AIJKOK_UNALLOCATED";
  else if (aijkok->a_dual.need_sync_host()) *mask = "PETSC_OFFLOAD_GPU";
  else if (aijkok->a_dual.need_sync_device()) *mask = "PETSC_OFFLOAD_CPU";
  else *mask = "PETSC_OFFLOAD_BOTH";
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatAIJKokkosPrintOffloadMask(Mat A)
{
  PetscMPIInt size;
  Mat         Ad, Ao;
  const char *amask, *bmask;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A), &size));

  if (size == 1) {
    PetscCall(MatSeqAIJKokkosGetOffloadMask(A, &amask));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "%s\n", amask));
  } else {
    Ad = ((Mat_MPIAIJ *)A->data)->A;
    Ao = ((Mat_MPIAIJ *)A->data)->B;
    PetscCall(MatSeqAIJKokkosGetOffloadMask(Ad, &amask));
    PetscCall(MatSeqAIJKokkosGetOffloadMask(Ao, &bmask));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Diag : Off-diag = %s : %s\n", amask, bmask));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
