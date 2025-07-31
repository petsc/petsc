#include <../src/mat/impls/htool/htool.hpp> /*I "petscmat.h" I*/
#include <set>

const char *const MatHtoolCompressorTypes[] = {"sympartialACA", "fullACA", "SVD"};
const char *const MatHtoolClusteringTypes[] = {"PCARegular", "PCAGeometric", "BoundingBox1Regular", "BoundingBox1Geometric"};
const char        HtoolCitation[]           = "@article{marchand2020two,\n"
                                              "  Author = {Marchand, Pierre and Claeys, Xavier and Jolivet, Pierre and Nataf, Fr\\'ed\\'eric and Tournier, Pierre-Henri},\n"
                                              "  Title = {Two-level preconditioning for $h$-version boundary element approximation of hypersingular operator with {GenEO}},\n"
                                              "  Year = {2020},\n"
                                              "  Publisher = {Elsevier},\n"
                                              "  Journal = {Numerische Mathematik},\n"
                                              "  Volume = {146},\n"
                                              "  Pages = {597--628},\n"
                                              "  Url = {https://github.com/htool-ddm/htool}\n"
                                              "}\n";
static PetscBool  HtoolCite                 = PETSC_FALSE;

static PetscErrorCode MatGetDiagonal_Htool(Mat A, Vec v)
{
  Mat_Htool   *a;
  PetscScalar *x;
  PetscBool    flg;

  PetscFunctionBegin;
  PetscCall(MatHasCongruentLayouts(A, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Only congruent layouts supported");
  PetscCall(MatShellGetContext(A, &a));
  PetscCall(VecGetArrayWrite(v, &x));
  PetscStackCallExternalVoid("copy_diagonal_in_user_numbering", htool::copy_diagonal_in_user_numbering(a->distributed_operator_holder->hmatrix, x));
  PetscCall(VecRestoreArrayWrite(v, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetDiagonalBlock_Htool(Mat A, Mat *b)
{
  Mat_Htool                 *a;
  Mat                        B;
  PetscScalar               *ptr, shift, scale;
  PetscBool                  flg;
  PetscMPIInt                rank;
  htool::Cluster<PetscReal> *source_cluster = nullptr;

  PetscFunctionBegin;
  PetscCall(MatHasCongruentLayouts(A, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Only congruent layouts supported");
  PetscCall(MatShellGetContext(A, &a));
  PetscCall(PetscObjectQuery((PetscObject)A, "DiagonalBlock", (PetscObject *)&B)); /* same logic as in MatGetDiagonalBlock_MPIDense() */
  if (!B) {
    PetscCall(MatShellGetScalingShifts(A, &shift, &scale, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
    PetscCall(MatCreateDense(PETSC_COMM_SELF, A->rmap->n, A->rmap->n, A->rmap->n, A->rmap->n, nullptr, &B));
    PetscCall(MatDenseGetArrayWrite(B, &ptr));
    PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
    source_cluster = a->source_cluster ? a->source_cluster.get() : a->target_cluster.get();
    PetscStackCallExternalVoid("copy_to_dense_in_user_numbering", htool::copy_to_dense_in_user_numbering(*a->distributed_operator_holder->hmatrix.get_sub_hmatrix(a->target_cluster->get_cluster_on_partition(rank), source_cluster->get_cluster_on_partition(rank)), ptr));
    PetscCall(MatDenseRestoreArrayWrite(B, &ptr));
    PetscCall(MatPropagateSymmetryOptions(A, B));
    PetscCall(PetscObjectCompose((PetscObject)A, "DiagonalBlock", (PetscObject)B));
    *b = B;
    PetscCall(MatDestroy(&B));
    PetscCall(MatShift(*b, shift));
    PetscCall(MatScale(*b, scale));
  } else {
    PetscCall(MatShellGetScalingShifts(A, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
    *b = B;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_Htool(Mat A, Vec x, Vec y)
{
  Mat_Htool         *a;
  const PetscScalar *in;
  PetscScalar       *out;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  PetscCall(VecGetArrayRead(x, &in));
  PetscCall(VecGetArrayWrite(y, &out));
  if (a->permutation == PETSC_TRUE) htool::add_distributed_operator_vector_product_local_to_local<PetscScalar>('N', 1.0, a->distributed_operator_holder->distributed_operator, in, 0.0, out, nullptr);
  else htool::internal_add_distributed_operator_vector_product_local_to_local<PetscScalar>('N', 1.0, a->distributed_operator_holder->distributed_operator, in, 0.0, out, nullptr);
  PetscCall(VecRestoreArrayRead(x, &in));
  PetscCall(VecRestoreArrayWrite(y, &out));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultTranspose_Htool(Mat A, Vec x, Vec y)
{
  Mat_Htool         *a;
  const PetscScalar *in;
  PetscScalar       *out;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  PetscCall(VecGetArrayRead(x, &in));
  PetscCall(VecGetArrayWrite(y, &out));
  if (a->permutation == PETSC_TRUE) htool::add_distributed_operator_vector_product_local_to_local<PetscScalar>('T', 1.0, a->distributed_operator_holder->distributed_operator, in, 0.0, out, nullptr);
  else htool::internal_add_distributed_operator_vector_product_local_to_local<PetscScalar>('T', 1.0, a->distributed_operator_holder->distributed_operator, in, 0.0, out, nullptr);
  PetscCall(VecRestoreArrayRead(x, &in));
  PetscCall(VecRestoreArrayWrite(y, &out));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatIncreaseOverlap_Htool(Mat A, PetscInt is_max, IS is[], PetscInt ov)
{
  std::set<PetscInt> set;
  const PetscInt    *idx;
  PetscInt          *oidx, size, bs[2];
  PetscMPIInt        csize;

  PetscFunctionBegin;
  PetscCall(MatGetBlockSizes(A, bs, bs + 1));
  if (bs[0] != bs[1]) bs[0] = 1;
  for (PetscInt i = 0; i < is_max; ++i) {
    /* basic implementation that adds indices by shifting an IS by -ov, -ov+1..., -1, 1..., ov-1, ov */
    /* needed to avoid subdomain matrices to replicate A since it is dense                           */
    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)is[i]), &csize));
    PetscCheck(csize == 1, PETSC_COMM_SELF, PETSC_ERR_WRONG_MPI_SIZE, "Unsupported parallel IS");
    PetscCall(ISGetSize(is[i], &size));
    PetscCall(ISGetIndices(is[i], &idx));
    for (PetscInt j = 0; j < size; ++j) {
      set.insert(idx[j]);
      for (PetscInt k = 1; k <= ov; ++k) {                                              /* for each layer of overlap      */
        if (idx[j] - k >= 0) set.insert(idx[j] - k);                                    /* do not insert negative indices */
        if (idx[j] + k < A->rmap->N && idx[j] + k < A->cmap->N) set.insert(idx[j] + k); /* do not insert indices greater than the dimension of A */
      }
    }
    PetscCall(ISRestoreIndices(is[i], &idx));
    PetscCall(ISDestroy(is + i));
    if (bs[0] > 1) {
      for (std::set<PetscInt>::iterator it = set.cbegin(); it != set.cend(); it++) {
        std::vector<PetscInt> block(bs[0]);
        std::iota(block.begin(), block.end(), (*it / bs[0]) * bs[0]);
        set.insert(block.cbegin(), block.cend());
      }
    }
    size = set.size(); /* size with overlap */
    PetscCall(PetscMalloc1(size, &oidx));
    for (const PetscInt j : set) *oidx++ = j;
    oidx -= size;
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, size, oidx, PETSC_OWN_POINTER, is + i));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCreateSubMatrices_Htool(Mat A, PetscInt n, const IS irow[], const IS icol[], MatReuse scall, Mat *submat[])
{
  Mat_Htool         *a;
  Mat                D, B, BT;
  const PetscScalar *copy;
  PetscScalar       *ptr, shift, scale;
  const PetscInt    *idxr, *idxc, *it;
  PetscInt           nrow, m, i;
  PetscBool          flg;

  PetscFunctionBegin;
  PetscCall(MatShellGetScalingShifts(A, &shift, &scale, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
  PetscCall(MatShellGetContext(A, &a));
  if (scall != MAT_REUSE_MATRIX) PetscCall(PetscCalloc1(n, submat));
  for (i = 0; i < n; ++i) {
    PetscCall(ISGetLocalSize(irow[i], &nrow));
    PetscCall(ISGetLocalSize(icol[i], &m));
    PetscCall(ISGetIndices(irow[i], &idxr));
    PetscCall(ISGetIndices(icol[i], &idxc));
    if (scall != MAT_REUSE_MATRIX) PetscCall(MatCreateDense(PETSC_COMM_SELF, nrow, m, nrow, m, nullptr, (*submat) + i));
    PetscCall(MatDenseGetArrayWrite((*submat)[i], &ptr));
    if (irow[i] == icol[i]) { /* same row and column IS? */
      PetscCall(MatHasCongruentLayouts(A, &flg));
      if (flg) {
        PetscCall(ISSorted(irow[i], &flg));
        if (flg) { /* sorted IS? */
          it = std::lower_bound(idxr, idxr + nrow, A->rmap->rstart);
          if (it != idxr + nrow && *it == A->rmap->rstart) {    /* rmap->rstart in IS? */
            if (std::distance(idxr, it) + A->rmap->n <= nrow) { /* long enough IS to store the local diagonal block? */
              for (PetscInt j = 0; j < A->rmap->n && flg; ++j)
                if (PetscUnlikely(it[j] != A->rmap->rstart + j)) flg = PETSC_FALSE;
              if (flg) { /* complete local diagonal block in IS? */
                /* fast extraction when the local diagonal block is part of the submatrix, e.g., for PCASM or PCHPDDM
                 *      [   B   C   E   ]
                 *  A = [   B   D   E   ]
                 *      [   B   F   E   ]
                 */
                m = std::distance(idxr, it); /* shift of the coefficient (0,0) of block D from above */
                PetscCall(MatGetDiagonalBlock(A, &D));
                PetscCall(MatDenseGetArrayRead(D, &copy));
                for (PetscInt k = 0; k < A->rmap->n; ++k) PetscCall(PetscArraycpy(ptr + (m + k) * nrow + m, copy + k * A->rmap->n, A->rmap->n)); /* block D from above */
                PetscCall(MatDenseRestoreArrayRead(D, &copy));
                if (m) {
                  a->wrapper->copy_submatrix(nrow, m, idxr, idxc, ptr); /* vertical block B from above */
                  /* entry-wise assembly may be costly, so transpose already-computed entries when possible */
                  if (A->symmetric == PETSC_BOOL3_TRUE || A->hermitian == PETSC_BOOL3_TRUE) {
                    PetscCall(MatCreateDense(PETSC_COMM_SELF, A->rmap->n, m, A->rmap->n, m, ptr + m, &B));
                    PetscCall(MatDenseSetLDA(B, nrow));
                    PetscCall(MatCreateDense(PETSC_COMM_SELF, m, A->rmap->n, m, A->rmap->n, ptr + m * nrow, &BT));
                    PetscCall(MatDenseSetLDA(BT, nrow));
                    if (A->hermitian == PETSC_BOOL3_TRUE && PetscDefined(USE_COMPLEX)) {
                      PetscCall(MatHermitianTranspose(B, MAT_REUSE_MATRIX, &BT));
                    } else {
                      PetscCall(MatTransposeSetPrecursor(B, BT));
                      PetscCall(MatTranspose(B, MAT_REUSE_MATRIX, &BT));
                    }
                    PetscCall(MatDestroy(&B));
                    PetscCall(MatDestroy(&BT));
                  } else {
                    for (PetscInt k = 0; k < A->rmap->n; ++k) { /* block C from above */
                      a->wrapper->copy_submatrix(m, 1, idxr, idxc + m + k, ptr + (m + k) * nrow);
                    }
                  }
                }
                if (m + A->rmap->n != nrow) {
                  a->wrapper->copy_submatrix(nrow, std::distance(it + A->rmap->n, idxr + nrow), idxr, idxc + m + A->rmap->n, ptr + (m + A->rmap->n) * nrow); /* vertical block E from above */
                  /* entry-wise assembly may be costly, so transpose already-computed entries when possible */
                  if (A->symmetric == PETSC_BOOL3_TRUE || A->hermitian == PETSC_BOOL3_TRUE) {
                    PetscCall(MatCreateDense(PETSC_COMM_SELF, A->rmap->n, nrow - (m + A->rmap->n), A->rmap->n, nrow - (m + A->rmap->n), ptr + (m + A->rmap->n) * nrow + m, &B));
                    PetscCall(MatDenseSetLDA(B, nrow));
                    PetscCall(MatCreateDense(PETSC_COMM_SELF, nrow - (m + A->rmap->n), A->rmap->n, nrow - (m + A->rmap->n), A->rmap->n, ptr + m * nrow + m + A->rmap->n, &BT));
                    PetscCall(MatDenseSetLDA(BT, nrow));
                    if (A->hermitian == PETSC_BOOL3_TRUE && PetscDefined(USE_COMPLEX)) {
                      PetscCall(MatHermitianTranspose(B, MAT_REUSE_MATRIX, &BT));
                    } else {
                      PetscCall(MatTransposeSetPrecursor(B, BT));
                      PetscCall(MatTranspose(B, MAT_REUSE_MATRIX, &BT));
                    }
                    PetscCall(MatDestroy(&B));
                    PetscCall(MatDestroy(&BT));
                  } else {
                    for (PetscInt k = 0; k < A->rmap->n; ++k) { /* block F from above */
                      a->wrapper->copy_submatrix(std::distance(it + A->rmap->n, idxr + nrow), 1, it + A->rmap->n, idxc + m + k, ptr + (m + k) * nrow + m + A->rmap->n);
                    }
                  }
                }
              } /* complete local diagonal block not in IS */
            } else flg = PETSC_FALSE; /* IS not long enough to store the local diagonal block */
          } else flg = PETSC_FALSE;   /* rmap->rstart not in IS */
        } /* unsorted IS */
      }
    } else flg = PETSC_FALSE;                                       /* different row and column IS */
    if (!flg) a->wrapper->copy_submatrix(nrow, m, idxr, idxc, ptr); /* reassemble everything */
    PetscCall(ISRestoreIndices(irow[i], &idxr));
    PetscCall(ISRestoreIndices(icol[i], &idxc));
    PetscCall(MatDenseRestoreArrayWrite((*submat)[i], &ptr));
    PetscCall(MatShift((*submat)[i], shift));
    PetscCall(MatScale((*submat)[i], scale));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_Htool(Mat A)
{
  Mat_Htool               *a;
  PetscContainer           container;
  MatHtoolKernelTranspose *kernelt;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_htool_seqdense_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_htool_mpidense_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_htool_seqdense_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_htool_mpidense_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolGetHierarchicalMat_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolSetKernel_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolGetPermutationSource_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolGetPermutationTarget_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolUsePermutation_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolUseRecompression_C", nullptr));
  PetscCall(PetscObjectQuery((PetscObject)A, "KernelTranspose", (PetscObject *)&container));
  if (container) { /* created in MatTranspose_Htool() */
    PetscCall(PetscContainerGetPointer(container, (void **)&kernelt));
    PetscCall(MatDestroy(&kernelt->A));
    PetscCall(PetscObjectCompose((PetscObject)A, "KernelTranspose", nullptr));
  }
  if (a->gcoords_source != a->gcoords_target) PetscCall(PetscFree(a->gcoords_source));
  PetscCall(PetscFree(a->gcoords_target));
  delete a->wrapper;
  a->target_cluster.reset();
  a->source_cluster.reset();
  a->distributed_operator_holder.reset();
  PetscCall(PetscFree(a));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatShellSetContext_C", nullptr)); // needed to avoid a call to MatShellSetContext_Immutable()
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatView_Htool(Mat A, PetscViewer pv)
{
  Mat_Htool                         *a;
  PetscScalar                        shift, scale;
  PetscBool                          flg;
  std::map<std::string, std::string> hmatrix_information;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  hmatrix_information = htool::get_distributed_hmatrix_information(a->distributed_operator_holder->hmatrix, PetscObjectComm((PetscObject)A));
  PetscCall(PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &flg));
  if (flg) {
    PetscCall(MatShellGetScalingShifts(A, &shift, &scale, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
    PetscCall(PetscViewerASCIIPrintf(pv, "symmetry: %c\n", a->distributed_operator_holder->block_diagonal_hmatrix->get_symmetry()));
    if (PetscAbsScalar(scale - 1.0) > PETSC_MACHINE_EPSILON) {
#if defined(PETSC_USE_COMPLEX)
      PetscCall(PetscViewerASCIIPrintf(pv, "scaling: %g+%gi\n", (double)PetscRealPart(scale), (double)PetscImaginaryPart(scale)));
#else
      PetscCall(PetscViewerASCIIPrintf(pv, "scaling: %g\n", (double)scale));
#endif
    }
    if (PetscAbsScalar(shift) > PETSC_MACHINE_EPSILON) {
#if defined(PETSC_USE_COMPLEX)
      PetscCall(PetscViewerASCIIPrintf(pv, "shift: %g+%gi\n", (double)PetscRealPart(shift), (double)PetscImaginaryPart(shift)));
#else
      PetscCall(PetscViewerASCIIPrintf(pv, "shift: %g\n", (double)shift));
#endif
    }
    PetscCall(PetscViewerASCIIPrintf(pv, "maximal cluster leaf size: %" PetscInt_FMT "\n", a->max_cluster_leaf_size));
    PetscCall(PetscViewerASCIIPrintf(pv, "epsilon: %g\n", (double)a->epsilon));
    PetscCall(PetscViewerASCIIPrintf(pv, "eta: %g\n", (double)a->eta));
    PetscCall(PetscViewerASCIIPrintf(pv, "minimum target depth: %" PetscInt_FMT "\n", a->depth[0]));
    PetscCall(PetscViewerASCIIPrintf(pv, "minimum source depth: %" PetscInt_FMT "\n", a->depth[1]));
    PetscCall(PetscViewerASCIIPrintf(pv, "compressor: %s\n", MatHtoolCompressorTypes[a->compressor]));
    PetscCall(PetscViewerASCIIPrintf(pv, "clustering: %s\n", MatHtoolClusteringTypes[a->clustering]));
    PetscCall(PetscViewerASCIIPrintf(pv, "compression ratio: %s\n", hmatrix_information["Compression_ratio"].c_str()));
    PetscCall(PetscViewerASCIIPrintf(pv, "space saving: %s\n", hmatrix_information["Space_saving"].c_str()));
    PetscCall(PetscViewerASCIIPrintf(pv, "block tree consistency: %s\n", PetscBools[a->distributed_operator_holder->hmatrix.is_block_tree_consistent()]));
    PetscCall(PetscViewerASCIIPrintf(pv, "recompression: %s\n", PetscBools[a->recompression]));
    PetscCall(PetscViewerASCIIPrintf(pv, "number of dense (resp. low rank) matrices: %s (resp. %s)\n", hmatrix_information["Number_of_dense_blocks"].c_str(), hmatrix_information["Number_of_low_rank_blocks"].c_str()));
    PetscCall(
      PetscViewerASCIIPrintf(pv, "(minimum, mean, maximum) dense block sizes: (%s, %s, %s)\n", hmatrix_information["Dense_block_size_min"].c_str(), hmatrix_information["Dense_block_size_mean"].c_str(), hmatrix_information["Dense_block_size_max"].c_str()));
    PetscCall(PetscViewerASCIIPrintf(pv, "(minimum, mean, maximum) low rank block sizes: (%s, %s, %s)\n", hmatrix_information["Low_rank_block_size_min"].c_str(), hmatrix_information["Low_rank_block_size_mean"].c_str(),
                                     hmatrix_information["Low_rank_block_size_max"].c_str()));
    PetscCall(PetscViewerASCIIPrintf(pv, "(minimum, mean, maximum) ranks: (%s, %s, %s)\n", hmatrix_information["Rank_min"].c_str(), hmatrix_information["Rank_mean"].c_str(), hmatrix_information["Rank_max"].c_str()));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* naive implementation of MatGetRow() needed for MatConvert_Nest_AIJ() */
static PetscErrorCode MatGetRow_Htool(Mat A, PetscInt row, PetscInt *nz, PetscInt **idx, PetscScalar **v)
{
  Mat_Htool   *a;
  PetscScalar  shift, scale;
  PetscInt    *idxc;
  PetscBLASInt one = 1, bn;

  PetscFunctionBegin;
  PetscCall(MatShellGetScalingShifts(A, &shift, &scale, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
  PetscCall(MatShellGetContext(A, &a));
  if (nz) *nz = A->cmap->N;
  if (idx || v) { /* even if !idx, need to set idxc for htool::copy_submatrix() */
    PetscCall(PetscMalloc1(A->cmap->N, &idxc));
    for (PetscInt i = 0; i < A->cmap->N; ++i) idxc[i] = i;
  }
  if (idx) *idx = idxc;
  if (v) {
    PetscCall(PetscMalloc1(A->cmap->N, v));
    if (a->wrapper) a->wrapper->copy_submatrix(1, A->cmap->N, &row, idxc, *v);
    else reinterpret_cast<htool::VirtualGenerator<PetscScalar> *>(a->kernelctx)->copy_submatrix(1, A->cmap->N, &row, idxc, *v);
    PetscCall(PetscBLASIntCast(A->cmap->N, &bn));
    PetscCallCXX(htool::Blas<PetscScalar>::scal(&bn, &scale, *v, &one));
    if (row < A->cmap->N) (*v)[row] += shift;
  }
  if (!idx) PetscCall(PetscFree(idxc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatRestoreRow_Htool(Mat, PetscInt, PetscInt *, PetscInt **idx, PetscScalar **v)
{
  PetscFunctionBegin;
  if (idx) PetscCall(PetscFree(*idx));
  if (v) PetscCall(PetscFree(*v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetFromOptions_Htool(Mat A, PetscOptionItems PetscOptionsObject)
{
  Mat_Htool *a;
  PetscInt   n;
  PetscBool  flg;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  PetscOptionsHeadBegin(PetscOptionsObject, "Htool options");
  PetscCall(PetscOptionsBoundedInt("-mat_htool_max_cluster_leaf_size", "Maximal leaf size in cluster tree", nullptr, a->max_cluster_leaf_size, &a->max_cluster_leaf_size, nullptr, 0));
  PetscCall(PetscOptionsBoundedReal("-mat_htool_epsilon", "Relative error in Frobenius norm when approximating a block", nullptr, a->epsilon, &a->epsilon, nullptr, 0.0));
  PetscCall(PetscOptionsReal("-mat_htool_eta", "Admissibility condition tolerance", nullptr, a->eta, &a->eta, nullptr));
  PetscCall(PetscOptionsBoundedInt("-mat_htool_min_target_depth", "Minimal cluster tree depth associated with the rows", nullptr, a->depth[0], a->depth, nullptr, 0));
  PetscCall(PetscOptionsBoundedInt("-mat_htool_min_source_depth", "Minimal cluster tree depth associated with the columns", nullptr, a->depth[1], a->depth + 1, nullptr, 0));
  PetscCall(PetscOptionsBool("-mat_htool_block_tree_consistency", "Block tree consistency", nullptr, a->block_tree_consistency, &a->block_tree_consistency, nullptr));
  PetscCall(PetscOptionsBool("-mat_htool_recompression", "Use recompression", nullptr, a->recompression, &a->recompression, nullptr));

  n = 0;
  PetscCall(PetscOptionsEList("-mat_htool_compressor", "Type of compression", "MatHtoolCompressorType", MatHtoolCompressorTypes, PETSC_STATIC_ARRAY_LENGTH(MatHtoolCompressorTypes), MatHtoolCompressorTypes[MAT_HTOOL_COMPRESSOR_SYMPARTIAL_ACA], &n, &flg));
  if (flg) a->compressor = MatHtoolCompressorType(n);
  n = 0;
  PetscCall(PetscOptionsEList("-mat_htool_clustering", "Type of clustering", "MatHtoolClusteringType", MatHtoolClusteringTypes, PETSC_STATIC_ARRAY_LENGTH(MatHtoolClusteringTypes), MatHtoolClusteringTypes[MAT_HTOOL_CLUSTERING_PCA_REGULAR], &n, &flg));
  if (flg) a->clustering = MatHtoolClusteringType(n);
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatAssemblyEnd_Htool(Mat A, MatAssemblyType)
{
  Mat_Htool                                                           *a;
  const PetscInt                                                      *ranges;
  PetscInt                                                            *offset;
  PetscMPIInt                                                          size, rank;
  char                                                                 S = PetscDefined(USE_COMPLEX) && A->hermitian == PETSC_BOOL3_TRUE ? 'H' : (A->symmetric == PETSC_BOOL3_TRUE ? 'S' : 'N'), uplo = S == 'N' ? 'N' : 'U';
  htool::VirtualGenerator<PetscScalar>                                *generator = nullptr;
  htool::ClusterTreeBuilder<PetscReal>                                 recursive_build_strategy;
  htool::Cluster<PetscReal>                                           *source_cluster;
  std::shared_ptr<htool::VirtualInternalLowRankGenerator<PetscScalar>> compressor;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(HtoolCitation, &HtoolCite));
  PetscCall(MatShellGetContext(A, &a));
  delete a->wrapper;
  a->target_cluster.reset();
  a->source_cluster.reset();
  a->distributed_operator_holder.reset();
  // clustering
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A), &size));
  PetscCall(PetscMalloc1(2 * size, &offset));
  PetscCall(MatGetOwnershipRanges(A, &ranges));
  for (PetscInt i = 0; i < size; ++i) {
    offset[2 * i]     = ranges[i];
    offset[2 * i + 1] = ranges[i + 1] - ranges[i];
  }
  switch (a->clustering) {
  case MAT_HTOOL_CLUSTERING_PCA_GEOMETRIC:
    recursive_build_strategy.set_partitioning_strategy(std::make_shared<htool::Partitioning<PetscReal, htool::ComputeLargestExtent<PetscReal>, htool::GeometricSplitting<PetscReal>>>());
    break;
  case MAT_HTOOL_CLUSTERING_BOUNDING_BOX_1_GEOMETRIC:
    recursive_build_strategy.set_partitioning_strategy(std::make_shared<htool::Partitioning<PetscReal, htool::ComputeBoundingBox<PetscReal>, htool::GeometricSplitting<PetscReal>>>());
    break;
  case MAT_HTOOL_CLUSTERING_BOUNDING_BOX_1_REGULAR:
    recursive_build_strategy.set_partitioning_strategy(std::make_shared<htool::Partitioning<PetscReal, htool::ComputeBoundingBox<PetscReal>, htool::RegularSplitting<PetscReal>>>());
    break;
  default:
    recursive_build_strategy.set_partitioning_strategy(std::make_shared<htool::Partitioning<PetscReal, htool::ComputeLargestExtent<PetscReal>, htool::RegularSplitting<PetscReal>>>());
  }
  recursive_build_strategy.set_maximal_leaf_size(a->max_cluster_leaf_size);
  a->target_cluster = std::make_unique<htool::Cluster<PetscReal>>(recursive_build_strategy.create_cluster_tree_from_local_partition(A->rmap->N, a->dim, a->gcoords_target, 2, size, offset));
  if (a->gcoords_target != a->gcoords_source) {
    PetscCall(MatGetOwnershipRangesColumn(A, &ranges));
    for (PetscInt i = 0; i < size; ++i) {
      offset[2 * i]     = ranges[i];
      offset[2 * i + 1] = ranges[i + 1] - ranges[i];
    }
    switch (a->clustering) {
    case MAT_HTOOL_CLUSTERING_PCA_GEOMETRIC:
      recursive_build_strategy.set_partitioning_strategy(std::make_shared<htool::Partitioning<PetscReal, htool::ComputeLargestExtent<PetscReal>, htool::GeometricSplitting<PetscReal>>>());
      break;
    case MAT_HTOOL_CLUSTERING_BOUNDING_BOX_1_GEOMETRIC:
      recursive_build_strategy.set_partitioning_strategy(std::make_shared<htool::Partitioning<PetscReal, htool::ComputeBoundingBox<PetscReal>, htool::GeometricSplitting<PetscReal>>>());
      break;
    case MAT_HTOOL_CLUSTERING_BOUNDING_BOX_1_REGULAR:
      recursive_build_strategy.set_partitioning_strategy(std::make_shared<htool::Partitioning<PetscReal, htool::ComputeBoundingBox<PetscReal>, htool::RegularSplitting<PetscReal>>>());
      break;
    default:
      recursive_build_strategy.set_partitioning_strategy(std::make_shared<htool::Partitioning<PetscReal, htool::ComputeLargestExtent<PetscReal>, htool::RegularSplitting<PetscReal>>>());
    }
    recursive_build_strategy.set_maximal_leaf_size(a->max_cluster_leaf_size);
    a->source_cluster = std::make_unique<htool::Cluster<PetscReal>>(recursive_build_strategy.create_cluster_tree_from_local_partition(A->cmap->N, a->dim, a->gcoords_source, 2, size, offset));
    S = uplo       = 'N';
    source_cluster = a->source_cluster.get();
  } else source_cluster = a->target_cluster.get();
  PetscCall(PetscFree(offset));
  // generator
  if (a->kernel) a->wrapper = new WrapperHtool(a->dim, a->kernel, a->kernelctx);
  else {
    a->wrapper = nullptr;
    generator  = reinterpret_cast<htool::VirtualGenerator<PetscScalar> *>(a->kernelctx);
  }
  // compressor
  switch (a->compressor) {
  case MAT_HTOOL_COMPRESSOR_FULL_ACA:
    compressor = std::make_shared<htool::fullACA<PetscScalar>>(a->wrapper ? *a->wrapper : *generator, a->target_cluster->get_permutation().data(), source_cluster->get_permutation().data());
    break;
  case MAT_HTOOL_COMPRESSOR_SVD:
    compressor = std::make_shared<htool::SVD<PetscScalar>>(a->wrapper ? *a->wrapper : *generator, a->target_cluster->get_permutation().data(), source_cluster->get_permutation().data());
    break;
  default:
    compressor = std::make_shared<htool::sympartialACA<PetscScalar>>(a->wrapper ? *a->wrapper : *generator, a->target_cluster->get_permutation().data(), source_cluster->get_permutation().data());
  }
  // local hierarchical matrix
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
  auto hmatrix_builder = htool::HMatrixTreeBuilder<PetscScalar>(a->epsilon, a->eta, S, uplo);
  if (a->recompression) {
    std::shared_ptr<htool::VirtualInternalLowRankGenerator<PetscScalar>> RecompressedLowRankGenerator = std::make_shared<htool::RecompressedLowRankGenerator<PetscScalar>>(*compressor, std::function<void(htool::LowRankMatrix<PetscScalar> &)>(htool::SVD_recompression<PetscScalar>));
    hmatrix_builder.set_low_rank_generator(RecompressedLowRankGenerator);
  } else {
    hmatrix_builder.set_low_rank_generator(compressor);
  }
  hmatrix_builder.set_minimal_target_depth(a->depth[0]);
  hmatrix_builder.set_minimal_source_depth(a->depth[1]);
  PetscCheck(a->block_tree_consistency || (!a->block_tree_consistency && !(A->symmetric == PETSC_BOOL3_TRUE || A->hermitian == PETSC_BOOL3_TRUE)), PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Cannot have a MatHtool with inconsistent block tree which is either symmetric or Hermitian");
  hmatrix_builder.set_block_tree_consistency(a->block_tree_consistency);
  a->distributed_operator_holder = std::make_unique<htool::DefaultApproximationBuilder<PetscScalar>>(a->wrapper ? *a->wrapper : *generator, *a->target_cluster, *source_cluster, hmatrix_builder, PetscObjectComm((PetscObject)A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductNumeric_Htool(Mat C)
{
  Mat_Product       *product = C->product;
  Mat_Htool         *a;
  const PetscScalar *in;
  PetscScalar       *out;
  PetscInt           N, lda;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  PetscCall(MatGetSize(C, nullptr, &N));
  PetscCall(MatDenseGetLDA(C, &lda));
  PetscCheck(lda == C->rmap->n, PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported leading dimension (%" PetscInt_FMT " != %" PetscInt_FMT ")", lda, C->rmap->n);
  PetscCall(MatDenseGetArrayRead(product->B, &in));
  PetscCall(MatDenseGetArrayWrite(C, &out));
  PetscCall(MatShellGetContext(product->A, &a));
  switch (product->type) {
  case MATPRODUCT_AB:
    if (a->permutation == PETSC_TRUE) htool::add_distributed_operator_matrix_product_local_to_local<PetscScalar>('N', 1.0, a->distributed_operator_holder->distributed_operator, in, 0.0, out, N, nullptr);
    else htool::internal_add_distributed_operator_matrix_product_local_to_local<PetscScalar>('N', 1.0, a->distributed_operator_holder->distributed_operator, in, 0.0, out, N, nullptr);
    break;
  case MATPRODUCT_AtB:
    if (a->permutation == PETSC_TRUE) htool::add_distributed_operator_matrix_product_local_to_local<PetscScalar>('T', 1.0, a->distributed_operator_holder->distributed_operator, in, 0.0, out, N, nullptr);
    else htool::internal_add_distributed_operator_matrix_product_local_to_local<PetscScalar>('T', 1.0, a->distributed_operator_holder->distributed_operator, in, 0.0, out, N, nullptr);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "MatProductType %s is not supported", MatProductTypes[product->type]);
  }
  PetscCall(MatDenseRestoreArrayWrite(C, &out));
  PetscCall(MatDenseRestoreArrayRead(product->B, &in));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSymbolic_Htool(Mat C)
{
  Mat_Product *product = C->product;
  Mat          A, B;
  PetscBool    flg;

  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  A = product->A;
  B = product->B;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)B, &flg, MATSEQDENSE, MATMPIDENSE, ""));
  PetscCheck(flg && (product->type == MATPRODUCT_AB || product->type == MATPRODUCT_AtB), PetscObjectComm((PetscObject)B), PETSC_ERR_SUP, "ProductType %s not supported for %s", MatProductTypes[product->type], ((PetscObject)product->B)->type_name);
  if (C->rmap->n == PETSC_DECIDE || C->cmap->n == PETSC_DECIDE || C->rmap->N == PETSC_DECIDE || C->cmap->N == PETSC_DECIDE) {
    if (product->type == MATPRODUCT_AB) PetscCall(MatSetSizes(C, A->rmap->n, B->cmap->n, A->rmap->N, B->cmap->N));
    else PetscCall(MatSetSizes(C, A->cmap->n, B->cmap->n, A->cmap->N, B->cmap->N));
  }
  PetscCall(MatSetType(C, MATDENSE));
  PetscCall(MatSetUp(C));
  PetscCall(MatSetOption(C, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));
  C->ops->productsymbolic = nullptr;
  C->ops->productnumeric  = MatProductNumeric_Htool;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductSetFromOptions_Htool(Mat C)
{
  PetscFunctionBegin;
  MatCheckProduct(C, 1);
  if (C->product->type == MATPRODUCT_AB || C->product->type == MATPRODUCT_AtB) C->ops->productsymbolic = MatProductSymbolic_Htool;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatHtoolGetHierarchicalMat_Htool(Mat A, const htool::DistributedOperator<PetscScalar> **distributed_operator)
{
  Mat_Htool *a;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  *distributed_operator = &a->distributed_operator_holder->distributed_operator;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatHtoolGetHierarchicalMat - Retrieves the opaque pointer to a Htool virtual matrix stored in a `MATHTOOL`.

  No Fortran Support, No C Support

  Input Parameter:
. A - hierarchical matrix

  Output Parameter:
. distributed_operator - opaque pointer to a Htool virtual matrix

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`
@*/
PETSC_EXTERN PetscErrorCode MatHtoolGetHierarchicalMat(Mat A, const htool::DistributedOperator<PetscScalar> **distributed_operator)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscAssertPointer(distributed_operator, 2);
  PetscTryMethod(A, "MatHtoolGetHierarchicalMat_C", (Mat, const htool::DistributedOperator<PetscScalar> **), (A, distributed_operator));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatHtoolSetKernel_Htool(Mat A, MatHtoolKernelFn *kernel, void *kernelctx)
{
  Mat_Htool *a;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  a->kernel    = kernel;
  a->kernelctx = kernelctx;
  delete a->wrapper;
  if (a->kernel) a->wrapper = new WrapperHtool(a->dim, a->kernel, a->kernelctx);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatHtoolSetKernel - Sets the kernel and context used for the assembly of a `MATHTOOL`.

  Collective, No Fortran Support

  Input Parameters:
+ A         - hierarchical matrix
. kernel    - computational kernel (or `NULL`)
- kernelctx - kernel context (if kernel is `NULL`, the pointer must be of type htool::VirtualGenerator<PetscScalar>*)

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`, `MatCreateHtoolFromKernel()`
@*/
PetscErrorCode MatHtoolSetKernel(Mat A, MatHtoolKernelFn *kernel, void *kernelctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  if (!kernelctx) PetscValidFunction(kernel, 2);
  if (!kernel) PetscAssertPointer(kernelctx, 3);
  PetscTryMethod(A, "MatHtoolSetKernel_C", (Mat, MatHtoolKernelFn *, void *), (A, kernel, kernelctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatHtoolGetPermutationSource_Htool(Mat A, IS *is)
{
  Mat_Htool                       *a;
  PetscMPIInt                      rank;
  const std::vector<PetscInt>     *source;
  const htool::Cluster<PetscReal> *local_source_cluster;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
  local_source_cluster = a->source_cluster ? &a->source_cluster->get_cluster_on_partition(rank) : &a->target_cluster->get_cluster_on_partition(rank);
  source               = &local_source_cluster->get_permutation();
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)A), local_source_cluster->get_size(), source->data() + local_source_cluster->get_offset(), PETSC_COPY_VALUES, is));
  PetscCall(ISSetPermutation(*is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatHtoolGetPermutationSource - Gets the permutation associated to the source cluster for a `MATHTOOL` matrix.

  Input Parameter:
. A - hierarchical matrix

  Output Parameter:
. is - permutation

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`, `MatHtoolGetPermutationTarget()`, `MatHtoolUsePermutation()`
@*/
PetscErrorCode MatHtoolGetPermutationSource(Mat A, IS *is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  if (!is) PetscAssertPointer(is, 2);
  PetscTryMethod(A, "MatHtoolGetPermutationSource_C", (Mat, IS *), (A, is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatHtoolGetPermutationTarget_Htool(Mat A, IS *is)
{
  Mat_Htool                   *a;
  const std::vector<PetscInt> *target;
  PetscMPIInt                  rank;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
  target = &a->target_cluster->get_permutation();
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)A), a->target_cluster->get_cluster_on_partition(rank).get_size(), target->data() + a->target_cluster->get_cluster_on_partition(rank).get_offset(), PETSC_COPY_VALUES, is));
  PetscCall(ISSetPermutation(*is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatHtoolGetPermutationTarget - Gets the permutation associated to the target cluster for a `MATHTOOL` matrix.

  Input Parameter:
. A - hierarchical matrix

  Output Parameter:
. is - permutation

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`, `MatHtoolGetPermutationSource()`, `MatHtoolUsePermutation()`
@*/
PetscErrorCode MatHtoolGetPermutationTarget(Mat A, IS *is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  if (!is) PetscAssertPointer(is, 2);
  PetscTryMethod(A, "MatHtoolGetPermutationTarget_C", (Mat, IS *), (A, is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatHtoolUsePermutation_Htool(Mat A, PetscBool use)
{
  Mat_Htool *a;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  a->permutation = use;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatHtoolUsePermutation - Sets whether a `MATHTOOL` matrix should permute input (resp. output) vectors following its internal source (resp. target) permutation.

  Input Parameters:
+ A   - hierarchical matrix
- use - Boolean value

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`, `MatHtoolGetPermutationSource()`, `MatHtoolGetPermutationTarget()`
@*/
PetscErrorCode MatHtoolUsePermutation(Mat A, PetscBool use)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveBool(A, use, 2);
  PetscTryMethod(A, "MatHtoolUsePermutation_C", (Mat, PetscBool), (A, use));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatHtoolUseRecompression_Htool(Mat A, PetscBool use)
{
  Mat_Htool *a;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  a->recompression = use;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatHtoolUseRecompression - Sets whether a `MATHTOOL` matrix should use recompression.

  Input Parameters:
+ A   - hierarchical matrix
- use - Boolean value

  Level: advanced

.seealso: [](ch_matrices), `Mat`, `MATHTOOL`
@*/
PetscErrorCode MatHtoolUseRecompression(Mat A, PetscBool use)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveBool(A, use, 2);
  PetscTryMethod(A, "MatHtoolUseRecompression_C", (Mat, PetscBool), (A, use));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatConvert_Htool_Dense(Mat A, MatType, MatReuse reuse, Mat *B)
{
  Mat          C;
  Mat_Htool   *a;
  PetscScalar *array, shift, scale;
  PetscInt     lda;

  PetscFunctionBegin;
  PetscCall(MatShellGetScalingShifts(A, &shift, &scale, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
  PetscCall(MatShellGetContext(A, &a));
  if (reuse == MAT_REUSE_MATRIX) {
    C = *B;
    PetscCheck(C->rmap->n == A->rmap->n && C->cmap->N == A->cmap->N, PETSC_COMM_SELF, PETSC_ERR_ARG_SIZ, "Incompatible dimensions");
    PetscCall(MatDenseGetLDA(C, &lda));
    PetscCheck(lda == C->rmap->n, PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported leading dimension (%" PetscInt_FMT " != %" PetscInt_FMT ")", lda, C->rmap->n);
  } else {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &C));
    PetscCall(MatSetSizes(C, A->rmap->n, A->cmap->n, A->rmap->N, A->cmap->N));
    PetscCall(MatSetType(C, MATDENSE));
    PetscCall(MatSetUp(C));
    PetscCall(MatSetOption(C, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE));
  }
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatDenseGetArrayWrite(C, &array));
  htool::copy_to_dense_in_user_numbering(a->distributed_operator_holder->hmatrix, array);
  PetscCall(MatDenseRestoreArrayWrite(C, &array));
  PetscCall(MatShift(C, shift));
  PetscCall(MatScale(C, scale));
  if (reuse == MAT_INPLACE_MATRIX) PetscCall(MatHeaderReplace(A, &C));
  else *B = C;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GenEntriesTranspose(PetscInt sdim, PetscInt M, PetscInt N, const PetscInt *rows, const PetscInt *cols, PetscScalar *ptr, void *ctx)
{
  MatHtoolKernelTranspose *generator = (MatHtoolKernelTranspose *)ctx;
  PetscScalar             *tmp;

  PetscFunctionBegin;
  PetscCall(generator->kernel(sdim, N, M, cols, rows, ptr, generator->kernelctx));
  PetscCall(PetscMalloc1(M * N, &tmp));
  PetscCall(PetscArraycpy(tmp, ptr, M * N));
  for (PetscInt i = 0; i < M; ++i) {
    for (PetscInt j = 0; j < N; ++j) ptr[i + j * M] = tmp[j + i * N];
  }
  PetscCall(PetscFree(tmp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* naive implementation which keeps a reference to the original Mat */
static PetscErrorCode MatTranspose_Htool(Mat A, MatReuse reuse, Mat *B)
{
  Mat                      C;
  Mat_Htool               *a, *c;
  PetscScalar              shift, scale;
  PetscInt                 M = A->rmap->N, N = A->cmap->N, m = A->rmap->n, n = A->cmap->n;
  PetscContainer           container;
  MatHtoolKernelTranspose *kernelt;

  PetscFunctionBegin;
  PetscCall(MatShellGetScalingShifts(A, &shift, &scale, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
  PetscCall(MatShellGetContext(A, &a));
  if (reuse == MAT_REUSE_MATRIX) PetscCall(MatTransposeCheckNonzeroState_Private(A, *B));
  PetscCheck(reuse != MAT_INPLACE_MATRIX, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "MatTranspose() with MAT_INPLACE_MATRIX not supported");
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &C));
    PetscCall(MatSetSizes(C, n, m, N, M));
    PetscCall(MatSetType(C, ((PetscObject)A)->type_name));
    PetscCall(MatSetUp(C));
    PetscCall(PetscNew(&kernelt));
    PetscCall(PetscObjectContainerCompose((PetscObject)C, "KernelTranspose", kernelt, PetscCtxDestroyDefault));
  } else {
    C = *B;
    PetscCall(PetscObjectQuery((PetscObject)C, "KernelTranspose", (PetscObject *)&container));
    PetscCheck(container, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call MatTranspose() with MAT_INITIAL_MATRIX first");
    PetscCall(PetscContainerGetPointer(container, (void **)&kernelt));
  }
  PetscCall(MatShellGetContext(C, &c));
  c->dim = a->dim;
  PetscCall(MatShift(C, shift));
  PetscCall(MatScale(C, scale));
  c->kernel = GenEntriesTranspose;
  if (kernelt->A != A) {
    PetscCall(MatDestroy(&kernelt->A));
    kernelt->A = A;
    PetscCall(PetscObjectReference((PetscObject)A));
  }
  kernelt->kernel           = a->kernel;
  kernelt->kernelctx        = a->kernelctx;
  c->kernelctx              = kernelt;
  c->max_cluster_leaf_size  = a->max_cluster_leaf_size;
  c->epsilon                = a->epsilon;
  c->eta                    = a->eta;
  c->block_tree_consistency = a->block_tree_consistency;
  c->permutation            = a->permutation;
  c->recompression          = a->recompression;
  c->compressor             = a->compressor;
  c->clustering             = a->clustering;
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(PetscMalloc1(N * c->dim, &c->gcoords_target));
    PetscCall(PetscArraycpy(c->gcoords_target, a->gcoords_source, N * c->dim));
    if (a->gcoords_target != a->gcoords_source) {
      PetscCall(PetscMalloc1(M * c->dim, &c->gcoords_source));
      PetscCall(PetscArraycpy(c->gcoords_source, a->gcoords_target, M * c->dim));
    } else c->gcoords_source = c->gcoords_target;
  }
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));
  if (reuse == MAT_INITIAL_MATRIX) *B = C;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_Factor(Mat F)
{
  PetscContainer               container;
  htool::HMatrix<PetscScalar> *A;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)F, "HMatrix", (PetscObject *)&container));
  if (container) {
    PetscCall(PetscContainerGetPointer(container, (void **)&A));
    delete A;
    PetscCall(PetscObjectCompose((PetscObject)F, "HMatrix", nullptr));
  }
  PetscCall(PetscObjectComposeFunction((PetscObject)F, "MatFactorGetSolverType_C", nullptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatFactorGetSolverType_Htool(Mat, MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERHTOOL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <char trans>
static inline PetscErrorCode MatSolve_Private(Mat A, htool::Matrix<PetscScalar> &X)
{
  PetscContainer               container;
  htool::HMatrix<PetscScalar> *B;

  PetscFunctionBegin;
  PetscCheck(A->factortype == MAT_FACTOR_LU || A->factortype == MAT_FACTOR_CHOLESKY, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_UNKNOWN_TYPE, "Only MAT_LU_FACTOR and MAT_CHOLESKY_FACTOR are supported");
  PetscCall(PetscObjectQuery((PetscObject)A, "HMatrix", (PetscObject *)&container));
  PetscCheck(container, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call Mat%sFactorNumeric() before Mat%sSolve%s()", A->factortype == MAT_FACTOR_LU ? "LU" : "Cholesky", X.nb_cols() == 1 ? "" : "Mat", trans == 'N' ? "" : "Transpose");
  PetscCall(PetscContainerGetPointer(container, (void **)&B));
  if (A->factortype == MAT_FACTOR_LU) htool::lu_solve(trans, *B, X);
  else htool::cholesky_solve('L', *B, X);
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <char trans, class Type, typename std::enable_if<std::is_same<Type, Vec>::value>::type * = nullptr>
static PetscErrorCode MatSolve_Htool(Mat A, Type b, Type x)
{
  PetscInt                   n;
  htool::Matrix<PetscScalar> v;
  PetscScalar               *array;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(b, &n));
  PetscCall(VecCopy(b, x));
  PetscCall(VecGetArrayWrite(x, &array));
  v.assign(n, 1, array, false);
  PetscCall(VecRestoreArrayWrite(x, &array));
  PetscCall(MatSolve_Private<trans>(A, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <char trans, class Type, typename std::enable_if<std::is_same<Type, Mat>::value>::type * = nullptr>
static PetscErrorCode MatSolve_Htool(Mat A, Type B, Type X)
{
  PetscInt                   m, N;
  htool::Matrix<PetscScalar> v;
  PetscScalar               *array;

  PetscFunctionBegin;
  PetscCall(MatGetLocalSize(B, &m, nullptr));
  PetscCall(MatGetLocalSize(B, nullptr, &N));
  PetscCall(MatCopy(B, X, SAME_NONZERO_PATTERN));
  PetscCall(MatDenseGetArrayWrite(X, &array));
  v.assign(m, N, array, false);
  PetscCall(MatDenseRestoreArrayWrite(X, &array));
  PetscCall(MatSolve_Private<trans>(A, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <MatFactorType ftype>
static PetscErrorCode MatFactorNumeric_Htool(Mat F, Mat A, const MatFactorInfo *)
{
  Mat_Htool                   *a;
  htool::HMatrix<PetscScalar> *B;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  B = new htool::HMatrix<PetscScalar>(a->distributed_operator_holder->hmatrix);
  if (ftype == MAT_FACTOR_LU) htool::sequential_lu_factorization(*B);
  else htool::sequential_cholesky_factorization('L', *B);
  PetscCall(PetscObjectContainerCompose((PetscObject)F, "HMatrix", B, nullptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <MatFactorType ftype>
PetscErrorCode MatFactorSymbolic_Htool(Mat F, Mat)
{
  PetscFunctionBegin;
  F->preallocated  = PETSC_TRUE;
  F->assembled     = PETSC_TRUE;
  F->ops->solve    = MatSolve_Htool<'N', Vec>;
  F->ops->matsolve = MatSolve_Htool<'N', Mat>;
  if (!PetscDefined(USE_COMPLEX) || ftype == MAT_FACTOR_LU) {
    F->ops->solvetranspose    = MatSolve_Htool<'T', Vec>;
    F->ops->matsolvetranspose = MatSolve_Htool<'T', Mat>;
  }
  F->ops->destroy = MatDestroy_Factor;
  if (ftype == MAT_FACTOR_LU) F->ops->lufactornumeric = MatFactorNumeric_Htool<MAT_FACTOR_LU>;
  else F->ops->choleskyfactornumeric = MatFactorNumeric_Htool<MAT_FACTOR_CHOLESKY>;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatLUFactorSymbolic_Htool(Mat F, Mat A, IS, IS, const MatFactorInfo *)
{
  PetscFunctionBegin;
  PetscCall(MatFactorSymbolic_Htool<MAT_FACTOR_LU>(F, A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCholeskyFactorSymbolic_Htool(Mat F, Mat A, IS, const MatFactorInfo *)
{
  PetscFunctionBegin;
  PetscCall(MatFactorSymbolic_Htool<MAT_FACTOR_CHOLESKY>(F, A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetFactor_htool_htool(Mat A, MatFactorType ftype, Mat *F)
{
  Mat         B;
  Mat_Htool  *a;
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A), &size));
  PetscCall(MatShellGetContext(A, &a));
  PetscCheck(size == 1, PetscObjectComm((PetscObject)A), PETSC_ERR_WRONG_MPI_SIZE, "Unsupported parallel MatGetFactor()");
  PetscCheck(a->block_tree_consistency, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Cannot factor a MatHtool with inconsistent block tree");
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &B));
  PetscCall(MatSetSizes(B, A->rmap->n, A->cmap->n, A->rmap->N, A->cmap->N));
  PetscCall(PetscStrallocpy(MATSOLVERHTOOL, &((PetscObject)B)->type_name));
  PetscCall(MatSetUp(B));

  B->ops->getinfo    = MatGetInfo_External;
  B->factortype      = ftype;
  B->trivialsymbolic = PETSC_TRUE;

  if (ftype == MAT_FACTOR_LU) B->ops->lufactorsymbolic = MatLUFactorSymbolic_Htool;
  else if (ftype == MAT_FACTOR_CHOLESKY) B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_Htool;

  PetscCall(PetscFree(B->solvertype));
  PetscCall(PetscStrallocpy(MATSOLVERHTOOL, &B->solvertype));

  PetscCall(PetscObjectComposeFunction((PetscObject)B, "MatFactorGetSolverType_C", MatFactorGetSolverType_Htool));
  *F = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatSolverTypeRegister_Htool(void)
{
  PetscFunctionBegin;
  PetscCall(MatSolverTypeRegister(MATSOLVERHTOOL, MATHTOOL, MAT_FACTOR_LU, MatGetFactor_htool_htool));
  PetscCall(MatSolverTypeRegister(MATSOLVERHTOOL, MATHTOOL, MAT_FACTOR_CHOLESKY, MatGetFactor_htool_htool));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  MatCreateHtoolFromKernel - Creates a `MATHTOOL` from a user-supplied kernel.

  Collective, No Fortran Support

  Input Parameters:
+ comm          - MPI communicator
. m             - number of local rows (or `PETSC_DECIDE` to have calculated if `M` is given)
. n             - number of local columns (or `PETSC_DECIDE` to have calculated if `N` is given)
. M             - number of global rows (or `PETSC_DETERMINE` to have calculated if `m` is given)
. N             - number of global columns (or `PETSC_DETERMINE` to have calculated if `n` is given)
. spacedim      - dimension of the space coordinates
. coords_target - coordinates of the target
. coords_source - coordinates of the source
. kernel        - computational kernel (or `NULL`)
- kernelctx     - kernel context (if kernel is `NULL`, the pointer must be of type htool::VirtualGenerator<PetscScalar>*)

  Output Parameter:
. B - matrix

  Options Database Keys:
+ -mat_htool_max_cluster_leaf_size <`PetscInt`>                                                - maximal leaf size in cluster tree
. -mat_htool_epsilon <`PetscReal`>                                                             - relative error in Frobenius norm when approximating a block
. -mat_htool_eta <`PetscReal`>                                                                 - admissibility condition tolerance
. -mat_htool_min_target_depth <`PetscInt`>                                                     - minimal cluster tree depth associated with the rows
. -mat_htool_min_source_depth <`PetscInt`>                                                     - minimal cluster tree depth associated with the columns
. -mat_htool_block_tree_consistency <`PetscBool`>                                              - block tree consistency
. -mat_htool_recompression <`PetscBool`>                                                       - use recompression
. -mat_htool_compressor <sympartialACA, fullACA, SVD>                                          - type of compression
- -mat_htool_clustering <PCARegular, PCAGeometric, BounbingBox1Regular, BoundingBox1Geometric> - type of clustering

  Level: intermediate

.seealso: [](ch_matrices), `Mat`, `MatCreate()`, `MATHTOOL`, `PCSetCoordinates()`, `MatHtoolSetKernel()`, `MatHtoolCompressorType`, `MATH2OPUS`, `MatCreateH2OpusFromKernel()`
@*/
PetscErrorCode MatCreateHtoolFromKernel(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt spacedim, const PetscReal coords_target[], const PetscReal coords_source[], MatHtoolKernelFn *kernel, void *kernelctx, Mat *B)
{
  Mat        A;
  Mat_Htool *a;

  PetscFunctionBegin;
  PetscCall(MatCreate(comm, &A));
  PetscValidLogicalCollectiveInt(A, spacedim, 6);
  PetscAssertPointer(coords_target, 7);
  PetscAssertPointer(coords_source, 8);
  if (!kernelctx) PetscValidFunction(kernel, 9);
  if (!kernel) PetscAssertPointer(kernelctx, 10);
  PetscCall(MatSetSizes(A, m, n, M, N));
  PetscCall(MatSetType(A, MATHTOOL));
  PetscCall(MatSetUp(A));
  PetscCall(MatShellGetContext(A, &a));
  a->dim       = spacedim;
  a->kernel    = kernel;
  a->kernelctx = kernelctx;
  PetscCall(PetscCalloc1(A->rmap->N * spacedim, &a->gcoords_target));
  PetscCall(PetscArraycpy(a->gcoords_target + A->rmap->rstart * spacedim, coords_target, A->rmap->n * spacedim));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, a->gcoords_target, A->rmap->N * spacedim, MPIU_REAL, MPI_SUM, PetscObjectComm((PetscObject)A))); /* global target coordinates */
  if (coords_target != coords_source) {
    PetscCall(PetscCalloc1(A->cmap->N * spacedim, &a->gcoords_source));
    PetscCall(PetscArraycpy(a->gcoords_source + A->cmap->rstart * spacedim, coords_source, A->cmap->n * spacedim));
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, a->gcoords_source, A->cmap->N * spacedim, MPIU_REAL, MPI_SUM, PetscObjectComm((PetscObject)A))); /* global source coordinates */
  } else a->gcoords_source = a->gcoords_target;
  *B = A;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     MATHTOOL = "htool" - A matrix type for hierarchical matrices using the Htool package.

  Use `./configure --download-htool` to install PETSc to use Htool.

   Options Database Key:
.     -mat_type htool - matrix type to `MATHTOOL`

   Level: beginner

.seealso: [](ch_matrices), `Mat`, `MATH2OPUS`, `MATDENSE`, `MatCreateHtoolFromKernel()`, `MatHtoolSetKernel()`
M*/
PETSC_EXTERN PetscErrorCode MatCreate_Htool(Mat A)
{
  Mat_Htool *a;

  PetscFunctionBegin;
  PetscCall(MatSetType(A, MATSHELL));
  PetscCall(PetscNew(&a));
  PetscCall(MatShellSetContext(A, a));
  PetscCall(MatShellSetOperation(A, MATOP_GET_DIAGONAL, (PetscErrorCodeFn *)MatGetDiagonal_Htool));
  PetscCall(MatShellSetOperation(A, MATOP_GET_DIAGONAL_BLOCK, (PetscErrorCodeFn *)MatGetDiagonalBlock_Htool));
  PetscCall(MatShellSetOperation(A, MATOP_MULT, (PetscErrorCodeFn *)MatMult_Htool));
  PetscCall(MatShellSetOperation(A, MATOP_MULT_TRANSPOSE, (PetscErrorCodeFn *)MatMultTranspose_Htool));
  if (!PetscDefined(USE_COMPLEX)) PetscCall(MatShellSetOperation(A, MATOP_MULT_HERMITIAN_TRANSPOSE, (PetscErrorCodeFn *)MatMultTranspose_Htool));
  A->ops->increaseoverlap   = MatIncreaseOverlap_Htool;
  A->ops->createsubmatrices = MatCreateSubMatrices_Htool;
  PetscCall(MatShellSetOperation(A, MATOP_VIEW, (PetscErrorCodeFn *)MatView_Htool));
  PetscCall(MatShellSetOperation(A, MATOP_SET_FROM_OPTIONS, (PetscErrorCodeFn *)MatSetFromOptions_Htool));
  PetscCall(MatShellSetOperation(A, MATOP_GET_ROW, (PetscErrorCodeFn *)MatGetRow_Htool));
  PetscCall(MatShellSetOperation(A, MATOP_RESTORE_ROW, (PetscErrorCodeFn *)MatRestoreRow_Htool));
  PetscCall(MatShellSetOperation(A, MATOP_ASSEMBLY_END, (PetscErrorCodeFn *)MatAssemblyEnd_Htool));
  PetscCall(MatShellSetOperation(A, MATOP_TRANSPOSE, (PetscErrorCodeFn *)MatTranspose_Htool));
  PetscCall(MatShellSetOperation(A, MATOP_DESTROY, (PetscErrorCodeFn *)MatDestroy_Htool));
  a->dim                    = 0;
  a->gcoords_target         = nullptr;
  a->gcoords_source         = nullptr;
  a->max_cluster_leaf_size  = 10;
  a->epsilon                = PetscSqrtReal(PETSC_SMALL);
  a->eta                    = 10.0;
  a->depth[0]               = 0;
  a->depth[1]               = 0;
  a->block_tree_consistency = PETSC_TRUE;
  a->permutation            = PETSC_TRUE;
  a->recompression          = PETSC_FALSE;
  a->compressor             = MAT_HTOOL_COMPRESSOR_SYMPARTIAL_ACA;
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_htool_seqdense_C", MatProductSetFromOptions_Htool));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_htool_mpidense_C", MatProductSetFromOptions_Htool));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_htool_seqdense_C", MatConvert_Htool_Dense));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_htool_mpidense_C", MatConvert_Htool_Dense));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolGetHierarchicalMat_C", MatHtoolGetHierarchicalMat_Htool));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolSetKernel_C", MatHtoolSetKernel_Htool));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolGetPermutationSource_C", MatHtoolGetPermutationSource_Htool));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolGetPermutationTarget_C", MatHtoolGetPermutationTarget_Htool));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolUsePermutation_C", MatHtoolUsePermutation_Htool));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolUseRecompression_C", MatHtoolUseRecompression_Htool));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatShellSetContext_C", MatShellSetContext_Immutable));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatShellSetContextDestroy_C", MatShellSetContextDestroy_Immutable));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatShellSetManageScalingShifts_C", MatShellSetManageScalingShifts_Immutable));
  PetscCall(PetscObjectChangeTypeName((PetscObject)A, MATHTOOL));
  PetscFunctionReturn(PETSC_SUCCESS);
}
