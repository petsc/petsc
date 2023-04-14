#include <../src/mat/impls/htool/htool.hpp> /*I "petscmat.h" I*/
#include <petscblaslapack.h>
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
  Mat_Htool   *a = (Mat_Htool *)A->data;
  PetscScalar *x;
  PetscBool    flg;

  PetscFunctionBegin;
  PetscCall(MatHasCongruentLayouts(A, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Only congruent layouts supported");
  PetscCall(VecGetArrayWrite(v, &x));
  a->hmatrix->copy_local_diagonal(x);
  PetscCall(VecRestoreArrayWrite(v, &x));
  PetscCall(VecScale(v, a->s));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetDiagonalBlock_Htool(Mat A, Mat *b)
{
  Mat_Htool   *a = (Mat_Htool *)A->data;
  Mat          B;
  PetscScalar *ptr;
  PetscBool    flg;

  PetscFunctionBegin;
  PetscCall(MatHasCongruentLayouts(A, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Only congruent layouts supported");
  PetscCall(PetscObjectQuery((PetscObject)A, "DiagonalBlock", (PetscObject *)&B)); /* same logic as in MatGetDiagonalBlock_MPIDense() */
  if (!B) {
    PetscCall(MatCreateDense(PETSC_COMM_SELF, A->rmap->n, A->rmap->n, A->rmap->n, A->rmap->n, nullptr, &B));
    PetscCall(MatDenseGetArrayWrite(B, &ptr));
    a->hmatrix->copy_local_diagonal_block(ptr);
    PetscCall(MatDenseRestoreArrayWrite(B, &ptr));
    PetscCall(MatPropagateSymmetryOptions(A, B));
    PetscCall(MatScale(B, a->s));
    PetscCall(PetscObjectCompose((PetscObject)A, "DiagonalBlock", (PetscObject)B));
    *b = B;
    PetscCall(MatDestroy(&B));
  } else *b = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_Htool(Mat A, Vec x, Vec y)
{
  Mat_Htool         *a = (Mat_Htool *)A->data;
  const PetscScalar *in;
  PetscScalar       *out;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x, &in));
  PetscCall(VecGetArrayWrite(y, &out));
  a->hmatrix->mvprod_local_to_local(in, out);
  PetscCall(VecRestoreArrayRead(x, &in));
  PetscCall(VecRestoreArrayWrite(y, &out));
  PetscCall(VecScale(y, a->s));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* naive implementation of MatMultAdd() needed for FEM-BEM coupling via MATNEST */
static PetscErrorCode MatMultAdd_Htool(Mat A, Vec v1, Vec v2, Vec v3)
{
  Mat_Htool        *a = (Mat_Htool *)A->data;
  Vec               tmp;
  const PetscScalar scale = a->s;

  PetscFunctionBegin;
  PetscCall(VecDuplicate(v2, &tmp));
  PetscCall(VecCopy(v2, v3)); /* no-op in MatMultAdd(bA->m[i][j],bx[j],by[i],by[i]) since VecCopy() checks for x == y */
  a->s = 1.0;                 /* set s to 1.0 since VecAXPY() may be used to scale the MatMult() output Vec */
  PetscCall(MatMult_Htool(A, v1, tmp));
  PetscCall(VecAXPY(v3, scale, tmp));
  PetscCall(VecDestroy(&tmp));
  a->s = scale; /* set s back to its original value */
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultTranspose_Htool(Mat A, Vec x, Vec y)
{
  Mat_Htool         *a = (Mat_Htool *)A->data;
  const PetscScalar *in;
  PetscScalar       *out;

  PetscFunctionBegin;
  PetscCall(VecGetArrayRead(x, &in));
  PetscCall(VecGetArrayWrite(y, &out));
  a->hmatrix->mvprod_transp_local_to_local(in, out);
  PetscCall(VecRestoreArrayRead(x, &in));
  PetscCall(VecRestoreArrayWrite(y, &out));
  PetscCall(VecScale(y, a->s));
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
    PetscCheck(csize == 1, PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported parallel IS");
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
  Mat_Htool         *a = (Mat_Htool *)A->data;
  Mat                D, B, BT;
  const PetscScalar *copy;
  PetscScalar       *ptr;
  const PetscInt    *idxr, *idxc, *it;
  PetscInt           nrow, m, i;
  PetscBool          flg;

  PetscFunctionBegin;
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
                PetscCall(MatGetDiagonalBlock_Htool(A, &D));
                PetscCall(MatDenseGetArrayRead(D, &copy));
                for (PetscInt k = 0; k < A->rmap->n; ++k) { PetscCall(PetscArraycpy(ptr + (m + k) * nrow + m, copy + k * A->rmap->n, A->rmap->n)); /* block D from above */ }
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
              }                       /* complete local diagonal block not in IS */
            } else flg = PETSC_FALSE; /* IS not long enough to store the local diagonal block */
          } else flg = PETSC_FALSE;   /* rmap->rstart not in IS */
        }                             /* unsorted IS */
      }
    } else flg = PETSC_FALSE;                                       /* different row and column IS */
    if (!flg) a->wrapper->copy_submatrix(nrow, m, idxr, idxc, ptr); /* reassemble everything */
    PetscCall(ISRestoreIndices(irow[i], &idxr));
    PetscCall(ISRestoreIndices(icol[i], &idxc));
    PetscCall(MatDenseRestoreArrayWrite((*submat)[i], &ptr));
    PetscCall(MatScale((*submat)[i], a->s));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_Htool(Mat A)
{
  Mat_Htool               *a = (Mat_Htool *)A->data;
  PetscContainer           container;
  MatHtoolKernelTranspose *kernelt;

  PetscFunctionBegin;
  PetscCall(PetscObjectChangeTypeName((PetscObject)A, nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_htool_seqdense_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatProductSetFromOptions_htool_mpidense_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_htool_seqdense_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_htool_mpidense_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolGetHierarchicalMat_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolSetKernel_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolGetPermutationSource_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolGetPermutationTarget_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolUsePermutation_C", nullptr));
  PetscCall(PetscObjectQuery((PetscObject)A, "KernelTranspose", (PetscObject *)&container));
  if (container) { /* created in MatTranspose_Htool() */
    PetscCall(PetscContainerGetPointer(container, (void **)&kernelt));
    PetscCall(MatDestroy(&kernelt->A));
    PetscCall(PetscFree(kernelt));
    PetscCall(PetscContainerDestroy(&container));
    PetscCall(PetscObjectCompose((PetscObject)A, "KernelTranspose", nullptr));
  }
  if (a->gcoords_source != a->gcoords_target) PetscCall(PetscFree(a->gcoords_source));
  PetscCall(PetscFree(a->gcoords_target));
  PetscCall(PetscFree2(a->work_source, a->work_target));
  delete a->wrapper;
  delete a->hmatrix;
  PetscCall(PetscFree(A->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatView_Htool(Mat A, PetscViewer pv)
{
  Mat_Htool *a = (Mat_Htool *)A->data;
  PetscBool  flg;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &flg));
  if (flg) {
    PetscCall(PetscViewerASCIIPrintf(pv, "symmetry: %c\n", a->hmatrix->get_symmetry_type()));
    if (PetscAbsScalar(a->s - 1.0) > PETSC_MACHINE_EPSILON) {
#if defined(PETSC_USE_COMPLEX)
      PetscCall(PetscViewerASCIIPrintf(pv, "scaling: %g+%gi\n", (double)PetscRealPart(a->s), (double)PetscImaginaryPart(a->s)));
#else
      PetscCall(PetscViewerASCIIPrintf(pv, "scaling: %g\n", (double)a->s));
#endif
    }
    PetscCall(PetscViewerASCIIPrintf(pv, "minimum cluster size: %" PetscInt_FMT "\n", a->bs[0]));
    PetscCall(PetscViewerASCIIPrintf(pv, "maximum block size: %" PetscInt_FMT "\n", a->bs[1]));
    PetscCall(PetscViewerASCIIPrintf(pv, "epsilon: %g\n", (double)a->epsilon));
    PetscCall(PetscViewerASCIIPrintf(pv, "eta: %g\n", (double)a->eta));
    PetscCall(PetscViewerASCIIPrintf(pv, "minimum target depth: %" PetscInt_FMT "\n", a->depth[0]));
    PetscCall(PetscViewerASCIIPrintf(pv, "minimum source depth: %" PetscInt_FMT "\n", a->depth[1]));
    PetscCall(PetscViewerASCIIPrintf(pv, "compressor: %s\n", MatHtoolCompressorTypes[a->compressor]));
    PetscCall(PetscViewerASCIIPrintf(pv, "clustering: %s\n", MatHtoolClusteringTypes[a->clustering]));
    PetscCall(PetscViewerASCIIPrintf(pv, "compression ratio: %s\n", a->hmatrix->get_infos("Compression_ratio").c_str()));
    PetscCall(PetscViewerASCIIPrintf(pv, "space saving: %s\n", a->hmatrix->get_infos("Space_saving").c_str()));
    PetscCall(PetscViewerASCIIPrintf(pv, "number of dense (resp. low rank) matrices: %s (resp. %s)\n", a->hmatrix->get_infos("Number_of_dmat").c_str(), a->hmatrix->get_infos("Number_of_lrmat").c_str()));
    PetscCall(PetscViewerASCIIPrintf(pv, "(minimum, mean, maximum) dense block sizes: (%s, %s, %s)\n", a->hmatrix->get_infos("Dense_block_size_min").c_str(), a->hmatrix->get_infos("Dense_block_size_mean").c_str(),
                                     a->hmatrix->get_infos("Dense_block_size_max").c_str()));
    PetscCall(PetscViewerASCIIPrintf(pv, "(minimum, mean, maximum) low rank block sizes: (%s, %s, %s)\n", a->hmatrix->get_infos("Low_rank_block_size_min").c_str(), a->hmatrix->get_infos("Low_rank_block_size_mean").c_str(),
                                     a->hmatrix->get_infos("Low_rank_block_size_max").c_str()));
    PetscCall(PetscViewerASCIIPrintf(pv, "(minimum, mean, maximum) ranks: (%s, %s, %s)\n", a->hmatrix->get_infos("Rank_min").c_str(), a->hmatrix->get_infos("Rank_mean").c_str(), a->hmatrix->get_infos("Rank_max").c_str()));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatScale_Htool(Mat A, PetscScalar s)
{
  Mat_Htool *a = (Mat_Htool *)A->data;

  PetscFunctionBegin;
  a->s *= s;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* naive implementation of MatGetRow() needed for MatConvert_Nest_AIJ() */
static PetscErrorCode MatGetRow_Htool(Mat A, PetscInt row, PetscInt *nz, PetscInt **idx, PetscScalar **v)
{
  Mat_Htool   *a = (Mat_Htool *)A->data;
  PetscInt    *idxc;
  PetscBLASInt one = 1, bn;

  PetscFunctionBegin;
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
    PetscCallBLAS("BLASscal", BLASscal_(&bn, &a->s, *v, &one));
  }
  if (!idx) PetscCall(PetscFree(idxc));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatRestoreRow_Htool(Mat, PetscInt, PetscInt *nz, PetscInt **idx, PetscScalar **v)
{
  PetscFunctionBegin;
  if (nz) *nz = 0;
  if (idx) PetscCall(PetscFree(*idx));
  if (v) PetscCall(PetscFree(*v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetFromOptions_Htool(Mat A, PetscOptionItems *PetscOptionsObject)
{
  Mat_Htool *a = (Mat_Htool *)A->data;
  PetscInt   n;
  PetscBool  flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "Htool options");
  PetscCall(PetscOptionsInt("-mat_htool_min_cluster_size", "Minimal leaf size in cluster tree", nullptr, a->bs[0], a->bs, nullptr));
  PetscCall(PetscOptionsInt("-mat_htool_max_block_size", "Maximal number of coefficients in a dense block", nullptr, a->bs[1], a->bs + 1, nullptr));
  PetscCall(PetscOptionsReal("-mat_htool_epsilon", "Relative error in Frobenius norm when approximating a block", nullptr, a->epsilon, &a->epsilon, nullptr));
  PetscCall(PetscOptionsReal("-mat_htool_eta", "Admissibility condition tolerance", nullptr, a->eta, &a->eta, nullptr));
  PetscCall(PetscOptionsInt("-mat_htool_min_target_depth", "Minimal cluster tree depth associated with the rows", nullptr, a->depth[0], a->depth, nullptr));
  PetscCall(PetscOptionsInt("-mat_htool_min_source_depth", "Minimal cluster tree depth associated with the columns", nullptr, a->depth[1], a->depth + 1, nullptr));
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
  Mat_Htool                                                   *a = (Mat_Htool *)A->data;
  const PetscInt                                              *ranges;
  PetscInt                                                    *offset;
  PetscMPIInt                                                  size;
  char                                                         S = PetscDefined(USE_COMPLEX) && A->hermitian == PETSC_BOOL3_TRUE ? 'H' : (A->symmetric == PETSC_BOOL3_TRUE ? 'S' : 'N'), uplo = S == 'N' ? 'N' : 'U';
  htool::VirtualGenerator<PetscScalar>                        *generator = nullptr;
  std::shared_ptr<htool::VirtualCluster>                       t, s = nullptr;
  std::shared_ptr<htool::VirtualLowRankGenerator<PetscScalar>> compressor = nullptr;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(HtoolCitation, &HtoolCite));
  delete a->wrapper;
  delete a->hmatrix;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A), &size));
  PetscCall(PetscMalloc1(2 * size, &offset));
  PetscCall(MatGetOwnershipRanges(A, &ranges));
  for (PetscInt i = 0; i < size; ++i) {
    offset[2 * i]     = ranges[i];
    offset[2 * i + 1] = ranges[i + 1] - ranges[i];
  }
  switch (a->clustering) {
  case MAT_HTOOL_CLUSTERING_PCA_GEOMETRIC:
    t = std::make_shared<htool::Cluster<htool::PCA<htool::SplittingTypes::GeometricSplitting>>>(a->dim);
    break;
  case MAT_HTOOL_CLUSTERING_BOUNDING_BOX_1_GEOMETRIC:
    t = std::make_shared<htool::Cluster<htool::BoundingBox1<htool::SplittingTypes::GeometricSplitting>>>(a->dim);
    break;
  case MAT_HTOOL_CLUSTERING_BOUNDING_BOX_1_REGULAR:
    t = std::make_shared<htool::Cluster<htool::BoundingBox1<htool::SplittingTypes::RegularSplitting>>>(a->dim);
    break;
  default:
    t = std::make_shared<htool::Cluster<htool::PCA<htool::SplittingTypes::RegularSplitting>>>(a->dim);
  }
  t->set_minclustersize(a->bs[0]);
  t->build(A->rmap->N, a->gcoords_target, offset, -1, PetscObjectComm((PetscObject)A));
  if (a->kernel) a->wrapper = new WrapperHtool(A->rmap->N, A->cmap->N, a->dim, a->kernel, a->kernelctx);
  else {
    a->wrapper = nullptr;
    generator  = reinterpret_cast<htool::VirtualGenerator<PetscScalar> *>(a->kernelctx);
  }
  if (a->gcoords_target != a->gcoords_source) {
    PetscCall(MatGetOwnershipRangesColumn(A, &ranges));
    for (PetscInt i = 0; i < size; ++i) {
      offset[2 * i]     = ranges[i];
      offset[2 * i + 1] = ranges[i + 1] - ranges[i];
    }
    switch (a->clustering) {
    case MAT_HTOOL_CLUSTERING_PCA_GEOMETRIC:
      s = std::make_shared<htool::Cluster<htool::PCA<htool::SplittingTypes::GeometricSplitting>>>(a->dim);
      break;
    case MAT_HTOOL_CLUSTERING_BOUNDING_BOX_1_GEOMETRIC:
      s = std::make_shared<htool::Cluster<htool::BoundingBox1<htool::SplittingTypes::GeometricSplitting>>>(a->dim);
      break;
    case MAT_HTOOL_CLUSTERING_BOUNDING_BOX_1_REGULAR:
      s = std::make_shared<htool::Cluster<htool::BoundingBox1<htool::SplittingTypes::RegularSplitting>>>(a->dim);
      break;
    default:
      s = std::make_shared<htool::Cluster<htool::PCA<htool::SplittingTypes::RegularSplitting>>>(a->dim);
    }
    s->set_minclustersize(a->bs[0]);
    s->build(A->cmap->N, a->gcoords_source, offset, -1, PetscObjectComm((PetscObject)A));
    S = uplo = 'N';
  }
  PetscCall(PetscFree(offset));
  switch (a->compressor) {
  case MAT_HTOOL_COMPRESSOR_FULL_ACA:
    compressor = std::make_shared<htool::fullACA<PetscScalar>>();
    break;
  case MAT_HTOOL_COMPRESSOR_SVD:
    compressor = std::make_shared<htool::SVD<PetscScalar>>();
    break;
  default:
    compressor = std::make_shared<htool::sympartialACA<PetscScalar>>();
  }
  a->hmatrix = dynamic_cast<htool::VirtualHMatrix<PetscScalar> *>(new htool::HMatrix<PetscScalar>(t, s ? s : t, a->epsilon, a->eta, S, uplo, -1, PetscObjectComm((PetscObject)A)));
  a->hmatrix->set_compression(compressor);
  a->hmatrix->set_maxblocksize(a->bs[1]);
  a->hmatrix->set_mintargetdepth(a->depth[0]);
  a->hmatrix->set_minsourcedepth(a->depth[1]);
  if (s) a->hmatrix->build(a->wrapper ? *a->wrapper : *generator, a->gcoords_target, a->gcoords_source);
  else a->hmatrix->build(a->wrapper ? *a->wrapper : *generator, a->gcoords_target);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatProductNumeric_Htool(Mat C)
{
  Mat_Product       *product = C->product;
  Mat_Htool         *a       = (Mat_Htool *)product->A->data;
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
  switch (product->type) {
  case MATPRODUCT_AB:
    a->hmatrix->mvprod_local_to_local(in, out, N);
    break;
  case MATPRODUCT_AtB:
    a->hmatrix->mvprod_transp_local_to_local(in, out, N);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "MatProductType %s is not supported", MatProductTypes[product->type]);
  }
  PetscCall(MatDenseRestoreArrayWrite(C, &out));
  PetscCall(MatDenseRestoreArrayRead(product->B, &in));
  PetscCall(MatScale(C, a->s));
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
  PetscCheck(flg, PetscObjectComm((PetscObject)B), PETSC_ERR_SUP, "MatProduct_AB not supported for %s", ((PetscObject)product->B)->type_name);
  switch (product->type) {
  case MATPRODUCT_AB:
    if (C->rmap->n == PETSC_DECIDE || C->cmap->n == PETSC_DECIDE || C->rmap->N == PETSC_DECIDE || C->cmap->N == PETSC_DECIDE) PetscCall(MatSetSizes(C, A->rmap->n, B->cmap->n, A->rmap->N, B->cmap->N));
    break;
  case MATPRODUCT_AtB:
    if (C->rmap->n == PETSC_DECIDE || C->cmap->n == PETSC_DECIDE || C->rmap->N == PETSC_DECIDE || C->cmap->N == PETSC_DECIDE) PetscCall(MatSetSizes(C, A->cmap->n, B->cmap->n, A->cmap->N, B->cmap->N));
    break;
  default:
    SETERRQ(PetscObjectComm((PetscObject)B), PETSC_ERR_SUP, "ProductType %s is not supported", MatProductTypes[product->type]);
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

static PetscErrorCode MatHtoolGetHierarchicalMat_Htool(Mat A, const htool::VirtualHMatrix<PetscScalar> **hmatrix)
{
  Mat_Htool *a = (Mat_Htool *)A->data;

  PetscFunctionBegin;
  *hmatrix = a->hmatrix;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
     MatHtoolGetHierarchicalMat - Retrieves the opaque pointer to a Htool virtual matrix stored in a `MATHTOOL`.

   Input Parameter:
.     A - hierarchical matrix

   Output Parameter:
.     hmatrix - opaque pointer to a Htool virtual matrix

   Level: advanced

.seealso: [](chapter_matrices), `Mat`, `MATHTOOL`
@*/
PETSC_EXTERN PetscErrorCode MatHtoolGetHierarchicalMat(Mat A, const htool::VirtualHMatrix<PetscScalar> **hmatrix)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidPointer(hmatrix, 2);
  PetscTryMethod(A, "MatHtoolGetHierarchicalMat_C", (Mat, const htool::VirtualHMatrix<PetscScalar> **), (A, hmatrix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatHtoolSetKernel_Htool(Mat A, MatHtoolKernel kernel, void *kernelctx)
{
  Mat_Htool *a = (Mat_Htool *)A->data;

  PetscFunctionBegin;
  a->kernel    = kernel;
  a->kernelctx = kernelctx;
  delete a->wrapper;
  if (a->kernel) a->wrapper = new WrapperHtool(A->rmap->N, A->cmap->N, a->dim, a->kernel, a->kernelctx);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
     MatHtoolSetKernel - Sets the kernel and context used for the assembly of a `MATHTOOL`.

   Input Parameters:
+     A - hierarchical matrix
.     kernel - computational kernel (or `NULL`)
-     kernelctx - kernel context (if kernel is `NULL`, the pointer must be of type htool::VirtualGenerator<PetscScalar>*)

   Level: advanced

.seealso: [](chapter_matrices), `Mat`, `MATHTOOL`, `MatCreateHtoolFromKernel()`
@*/
PETSC_EXTERN PetscErrorCode MatHtoolSetKernel(Mat A, MatHtoolKernel kernel, void *kernelctx)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  if (!kernelctx) PetscValidFunction(kernel, 2);
  if (!kernel) PetscValidPointer(kernelctx, 3);
  PetscTryMethod(A, "MatHtoolSetKernel_C", (Mat, MatHtoolKernel, void *), (A, kernel, kernelctx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatHtoolGetPermutationSource_Htool(Mat A, IS *is)
{
  Mat_Htool            *a = (Mat_Htool *)A->data;
  std::vector<PetscInt> source;

  PetscFunctionBegin;
  source = a->hmatrix->get_source_cluster()->get_local_perm();
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)A), source.size(), source.data(), PETSC_COPY_VALUES, is));
  PetscCall(ISSetPermutation(*is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
     MatHtoolGetPermutationSource - Gets the permutation associated to the source cluster for a `MATHTOOL` matrix.

   Input Parameter:
.     A - hierarchical matrix

   Output Parameter:
.     is - permutation

   Level: advanced

.seealso: [](chapter_matrices), `Mat`, `MATHTOOL`, `MatHtoolGetPermutationTarget()`, `MatHtoolUsePermutation()`
@*/
PETSC_EXTERN PetscErrorCode MatHtoolGetPermutationSource(Mat A, IS *is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  if (!is) PetscValidPointer(is, 2);
  PetscTryMethod(A, "MatHtoolGetPermutationSource_C", (Mat, IS *), (A, is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatHtoolGetPermutationTarget_Htool(Mat A, IS *is)
{
  Mat_Htool            *a = (Mat_Htool *)A->data;
  std::vector<PetscInt> target;

  PetscFunctionBegin;
  target = a->hmatrix->get_target_cluster()->get_local_perm();
  PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)A), target.size(), target.data(), PETSC_COPY_VALUES, is));
  PetscCall(ISSetPermutation(*is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
     MatHtoolGetPermutationTarget - Gets the permutation associated to the target cluster for a `MATHTOOL` matrix.

   Input Parameter:
.     A - hierarchical matrix

   Output Parameter:
.     is - permutation

   Level: advanced

.seealso: [](chapter_matrices), `Mat`, `MATHTOOL`, `MatHtoolGetPermutationSource()`, `MatHtoolUsePermutation()`
@*/
PETSC_EXTERN PetscErrorCode MatHtoolGetPermutationTarget(Mat A, IS *is)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  if (!is) PetscValidPointer(is, 2);
  PetscTryMethod(A, "MatHtoolGetPermutationTarget_C", (Mat, IS *), (A, is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatHtoolUsePermutation_Htool(Mat A, PetscBool use)
{
  Mat_Htool *a = (Mat_Htool *)A->data;

  PetscFunctionBegin;
  a->hmatrix->set_use_permutation(use);
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
     MatHtoolUsePermutation - Sets whether a `MATHTOOL` matrix should permute input (resp. output) vectors following its internal source (resp. target) permutation.

   Input Parameters:
+     A - hierarchical matrix
-     use - Boolean value

   Level: advanced

.seealso: [](chapter_matrices), `Mat`, `MATHTOOL`, `MatHtoolGetPermutationSource()`, `MatHtoolGetPermutationTarget()`
@*/
PETSC_EXTERN PetscErrorCode MatHtoolUsePermutation(Mat A, PetscBool use)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidLogicalCollectiveBool(A, use, 2);
  PetscTryMethod(A, "MatHtoolUsePermutation_C", (Mat, PetscBool), (A, use));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatConvert_Htool_Dense(Mat A, MatType, MatReuse reuse, Mat *B)
{
  Mat          C;
  Mat_Htool   *a = (Mat_Htool *)A->data;
  PetscInt     lda;
  PetscScalar *array;

  PetscFunctionBegin;
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
  a->hmatrix->copy_local_dense_perm(array);
  PetscCall(MatDenseRestoreArrayWrite(C, &array));
  PetscCall(MatScale(C, a->s));
  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A, &C));
  } else *B = C;
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
  Mat_Htool               *a = (Mat_Htool *)A->data, *c;
  PetscInt                 M = A->rmap->N, N = A->cmap->N, m = A->rmap->n, n = A->cmap->n;
  PetscContainer           container;
  MatHtoolKernelTranspose *kernelt;

  PetscFunctionBegin;
  if (reuse == MAT_REUSE_MATRIX) PetscCall(MatTransposeCheckNonzeroState_Private(A, *B));
  PetscCheck(reuse != MAT_INPLACE_MATRIX, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "MatTranspose() with MAT_INPLACE_MATRIX not supported");
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &C));
    PetscCall(MatSetSizes(C, n, m, N, M));
    PetscCall(MatSetType(C, ((PetscObject)A)->type_name));
    PetscCall(MatSetUp(C));
    PetscCall(PetscContainerCreate(PetscObjectComm((PetscObject)C), &container));
    PetscCall(PetscNew(&kernelt));
    PetscCall(PetscContainerSetPointer(container, kernelt));
    PetscCall(PetscObjectCompose((PetscObject)C, "KernelTranspose", (PetscObject)container));
  } else {
    C = *B;
    PetscCall(PetscObjectQuery((PetscObject)C, "KernelTranspose", (PetscObject *)&container));
    PetscCheck(container, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call MatTranspose() with MAT_INITIAL_MATRIX first");
    PetscCall(PetscContainerGetPointer(container, (void **)&kernelt));
  }
  c         = (Mat_Htool *)C->data;
  c->dim    = a->dim;
  c->s      = a->s;
  c->kernel = GenEntriesTranspose;
  if (kernelt->A != A) {
    PetscCall(MatDestroy(&kernelt->A));
    kernelt->A = A;
    PetscCall(PetscObjectReference((PetscObject)A));
  }
  kernelt->kernel    = a->kernel;
  kernelt->kernelctx = a->kernelctx;
  c->kernelctx       = kernelt;
  if (reuse == MAT_INITIAL_MATRIX) {
    PetscCall(PetscMalloc1(N * c->dim, &c->gcoords_target));
    PetscCall(PetscArraycpy(c->gcoords_target, a->gcoords_source, N * c->dim));
    if (a->gcoords_target != a->gcoords_source) {
      PetscCall(PetscMalloc1(M * c->dim, &c->gcoords_source));
      PetscCall(PetscArraycpy(c->gcoords_source, a->gcoords_target, M * c->dim));
    } else c->gcoords_source = c->gcoords_target;
    PetscCall(PetscCalloc2(M, &c->work_source, N, &c->work_target));
  }
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));
  if (reuse == MAT_INITIAL_MATRIX) *B = C;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
     MatCreateHtoolFromKernel - Creates a `MATHTOOL` from a user-supplied kernel.

   Input Parameters:
+     comm - MPI communicator
.     m - number of local rows (or `PETSC_DECIDE` to have calculated if `M` is given)
.     n - number of local columns (or `PETSC_DECIDE` to have calculated if `N` is given)
.     M - number of global rows (or `PETSC_DETERMINE` to have calculated if `m` is given)
.     N - number of global columns (or `PETSC_DETERMINE` to have calculated if `n` is given)
.     spacedim - dimension of the space coordinates
.     coords_target - coordinates of the target
.     coords_source - coordinates of the source
.     kernel - computational kernel (or `NULL`)
-     kernelctx - kernel context (if kernel is `NULL`, the pointer must be of type htool::VirtualGenerator<PetscScalar>*)

   Output Parameter:
.     B - matrix

   Options Database Keys:
+     -mat_htool_min_cluster_size <`PetscInt`> - minimal leaf size in cluster tree
.     -mat_htool_max_block_size <`PetscInt`> - maximal number of coefficients in a dense block
.     -mat_htool_epsilon <`PetscReal`> - relative error in Frobenius norm when approximating a block
.     -mat_htool_eta <`PetscReal`> - admissibility condition tolerance
.     -mat_htool_min_target_depth <`PetscInt`> - minimal cluster tree depth associated with the rows
.     -mat_htool_min_source_depth <`PetscInt`> - minimal cluster tree depth associated with the columns
.     -mat_htool_compressor <sympartialACA, fullACA, SVD> - type of compression
-     -mat_htool_clustering <PCARegular, PCAGeometric, BounbingBox1Regular, BoundingBox1Geometric> - type of clustering

   Level: intermediate

.seealso: [](chapter_matrices), `Mat`, `MatCreate()`, `MATHTOOL`, `PCSetCoordinates()`, `MatHtoolSetKernel()`, `MatHtoolCompressorType`, `MATH2OPUS`, `MatCreateH2OpusFromKernel()`
@*/
PetscErrorCode MatCreateHtoolFromKernel(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt spacedim, const PetscReal coords_target[], const PetscReal coords_source[], MatHtoolKernel kernel, void *kernelctx, Mat *B)
{
  Mat        A;
  Mat_Htool *a;

  PetscFunctionBegin;
  PetscCall(MatCreate(comm, &A));
  PetscValidLogicalCollectiveInt(A, spacedim, 6);
  PetscValidRealPointer(coords_target, 7);
  PetscValidRealPointer(coords_source, 8);
  if (!kernelctx) PetscValidFunction(kernel, 9);
  if (!kernel) PetscValidPointer(kernelctx, 10);
  PetscCall(MatSetSizes(A, m, n, M, N));
  PetscCall(MatSetType(A, MATHTOOL));
  PetscCall(MatSetUp(A));
  a            = (Mat_Htool *)A->data;
  a->dim       = spacedim;
  a->s         = 1.0;
  a->kernel    = kernel;
  a->kernelctx = kernelctx;
  PetscCall(PetscCalloc1(A->rmap->N * spacedim, &a->gcoords_target));
  PetscCall(PetscArraycpy(a->gcoords_target + A->rmap->rstart * spacedim, coords_target, A->rmap->n * spacedim));
  PetscCall(MPIU_Allreduce(MPI_IN_PLACE, a->gcoords_target, A->rmap->N * spacedim, MPIU_REAL, MPI_SUM, PetscObjectComm((PetscObject)A))); /* global target coordinates */
  if (coords_target != coords_source) {
    PetscCall(PetscCalloc1(A->cmap->N * spacedim, &a->gcoords_source));
    PetscCall(PetscArraycpy(a->gcoords_source + A->cmap->rstart * spacedim, coords_source, A->cmap->n * spacedim));
    PetscCall(MPIU_Allreduce(MPI_IN_PLACE, a->gcoords_source, A->cmap->N * spacedim, MPIU_REAL, MPI_SUM, PetscObjectComm((PetscObject)A))); /* global source coordinates */
  } else a->gcoords_source = a->gcoords_target;
  PetscCall(PetscCalloc2(A->cmap->N, &a->work_source, A->rmap->N, &a->work_target));
  *B = A;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     MATHTOOL = "htool" - A matrix type for hierarchical matrices using the Htool package.

  Use `./configure --download-htool` to install PETSc to use Htool.

   Options Database Key:
.     -mat_type htool - matrix type to `MATHTOOL`

   Level: beginner

.seealso: [](chapter_matrices), `Mat`, `MATH2OPUS`, `MATDENSE`, `MatCreateHtoolFromKernel()`, `MatHtoolSetKernel()`
M*/
PETSC_EXTERN PetscErrorCode MatCreate_Htool(Mat A)
{
  Mat_Htool *a;

  PetscFunctionBegin;
  PetscCall(PetscNew(&a));
  A->data = (void *)a;
  PetscCall(PetscObjectChangeTypeName((PetscObject)A, MATHTOOL));
  PetscCall(PetscMemzero(A->ops, sizeof(struct _MatOps)));
  A->ops->getdiagonal      = MatGetDiagonal_Htool;
  A->ops->getdiagonalblock = MatGetDiagonalBlock_Htool;
  A->ops->mult             = MatMult_Htool;
  A->ops->multadd          = MatMultAdd_Htool;
  A->ops->multtranspose    = MatMultTranspose_Htool;
  if (!PetscDefined(USE_COMPLEX)) A->ops->multhermitiantranspose = MatMultTranspose_Htool;
  A->ops->increaseoverlap   = MatIncreaseOverlap_Htool;
  A->ops->createsubmatrices = MatCreateSubMatrices_Htool;
  A->ops->transpose         = MatTranspose_Htool;
  A->ops->destroy           = MatDestroy_Htool;
  A->ops->view              = MatView_Htool;
  A->ops->setfromoptions    = MatSetFromOptions_Htool;
  A->ops->scale             = MatScale_Htool;
  A->ops->getrow            = MatGetRow_Htool;
  A->ops->restorerow        = MatRestoreRow_Htool;
  A->ops->assemblyend       = MatAssemblyEnd_Htool;
  a->dim                    = 0;
  a->gcoords_target         = nullptr;
  a->gcoords_source         = nullptr;
  a->s                      = 1.0;
  a->bs[0]                  = 10;
  a->bs[1]                  = 1000000;
  a->epsilon                = PetscSqrtReal(PETSC_SMALL);
  a->eta                    = 10.0;
  a->depth[0]               = 0;
  a->depth[1]               = 0;
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
  PetscFunctionReturn(PETSC_SUCCESS);
}
