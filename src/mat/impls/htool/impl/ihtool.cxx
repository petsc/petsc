#include <../src/mat/impls/htool/htool.hpp> /*I "petscmat.h" I*/
#include <petscdraw.h>
#include <set>

const char *const MatHtoolCompressorTypes[] = {"sympartialACA", "fullACA", "SVD"};
const char *const MatHtoolClusteringTypes[] = {"PCARegular", "PCAGeometric", "BoundingBox1Regular", "BoundingBox1Geometric"};
const char       *HtoolCitations[2]         = {"@article{marchand2020two,\n"
                                               "  Author = {Marchand, Pierre and Claeys, Xavier and Jolivet, Pierre and Nataf, Fr\\'ed\\'eric and Tournier, Pierre-Henri},\n"
                                               "  Title = {Two-level preconditioning for $h$-version boundary element approximation of hypersingular operator with {GenEO}},\n"
                                               "  Year = {2020},\n"
                                               "  Publisher = {Elsevier},\n"
                                               "  Journal = {Numerische Mathematik},\n"
                                               "  Volume = {146},\n"
                                               "  Pages = {597--628},\n"
                                               "  Url = {https://github.com/htool-ddm/htool}\n"
                                               "}\n",
                                               "@article{Marchand2026,\n"
                                               "  Author = {Marchand, Pierre and Tournier, Pierre-Henri and Jolivet, Pierre},\n"
                                               "  Title = {{Htool-DDM}: A {C++} library for parallel solvers and compressed linear systems},\n"
                                               "  Year = {2026},\n"
                                               "  Publisher = {The Open Journal},\n"
                                               "  Journal = {Journal of Open Source Software},\n"
                                               "  Volume = {11},\n"
                                               "  Number = {118},\n"
                                               "  Pages = {9279},\n"
                                               "  Url = {https://doi.org/10.21105/joss.09279}\n"
                                               "}\n"};
static PetscBool  HtoolCite[2]              = {PETSC_FALSE, PETSC_FALSE};

static PetscErrorCode MatGetDiagonal_Htool(Mat A, Vec v)
{
  Mat_Htool   *a;
  PetscScalar *x;
  PetscBool    flg;

  PetscFunctionBegin;
  PetscCall(MatHasCongruentLayouts(A, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Only congruent layouts supported");
  PetscCall(MatShellGetContext(A, &a));
  PetscCheck(a->block_diagonal_hmatrix, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONGSTATE, "Block diagonal htool::HMatrix not found");
  PetscCall(VecGetArrayWrite(v, &x));
  PetscCallExternalVoid("copy_diagonal_in_user_numbering", htool::copy_diagonal_in_user_numbering(*a->block_diagonal_hmatrix, x));
  PetscCall(VecRestoreArrayWrite(v, &x));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetDiagonalBlock_Htool(Mat A, Mat *b)
{
  Mat_Htool  *a, *c;
  Mat         B;
  PetscScalar shift[2], scale[2];
  PetscBool   flg;

  PetscFunctionBegin;
  PetscCall(MatHasCongruentLayouts(A, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Only congruent layouts supported");
  PetscCall(MatShellGetContext(A, &a));
  PetscCall(PetscObjectQuery((PetscObject)A, "DiagonalBlock", (PetscObject *)&B)); /* same logic as in MatGetDiagonalBlock_MPIDense() */
  PetscCall(MatShellGetScalingShifts(A, shift, scale, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
  if (B) {
    PetscCall(MatShellGetScalingShifts(B, shift + 1, scale + 1, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
    /* invalidate cache when scale or shift changed; PetscObjectCompose() below releases the old entry */
    if (scale[0] != scale[1] || shift[0] != shift[1]) B = nullptr;
  }
  if (!B) {
    PetscCheck(a->block_diagonal_hmatrix, PetscObjectComm((PetscObject)A), PETSC_ERR_ARG_WRONGSTATE, "Block diagonal htool::HMatrix not found");
    PetscCall(MatCreate(PETSC_COMM_SELF, &B));
    PetscCall(MatSetSizes(B, A->rmap->n, A->cmap->n, A->rmap->n, A->cmap->n));
    PetscCall(MatSetType(B, MATHTOOL));
    PetscCall(MatSetUp(B));
    PetscCall(MatShellGetContext(B, &c));
    c->dim                     = a->dim;
    c->max_cluster_leaf_size   = a->max_cluster_leaf_size;
    c->epsilon                 = a->epsilon;
    c->eta                     = a->eta;
    c->depth[0]                = a->depth[0];
    c->depth[1]                = a->depth[1];
    c->block_tree_consistency  = a->block_tree_consistency;
    c->permutation             = a->permutation;
    c->recompression           = a->recompression;
    c->compressor              = a->compressor;
    c->clustering              = a->clustering;
    c->kernel                  = a->kernel;
    c->kernelctx               = a->kernelctx;
    c->local_to_local_operator = std::make_unique<htool::LocalToLocalHMatrix<PetscScalar>>(*a->block_diagonal_hmatrix);
    c->distributed_operator_holder = std::make_unique<htool::CustomApproximationBuilder<PetscScalar>>(a->block_diagonal_hmatrix->get_target_cluster(), a->block_diagonal_hmatrix->get_source_cluster(), PetscObjectComm((PetscObject)A), *c->local_to_local_operator);
    c->distributed_operator   = &c->distributed_operator_holder->distributed_operator;
    c->block_diagonal_hmatrix = a->block_diagonal_hmatrix;
    c->local_hmatrix_view     = a->block_diagonal_hmatrix;
    B->assembled              = PETSC_TRUE;
    PetscCall(MatPropagateSymmetryOptions(A, B));
    PetscCall(PetscObjectCompose((PetscObject)A, "DiagonalBlock", (PetscObject)B));
    *b = B;
    PetscCall(MatDestroy(&B));
    PetscCall(MatScale(*b, *scale));
    PetscCall(MatShift(*b, *shift));
  } else *b = B;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDuplicate_Htool(Mat A, MatDuplicateOption op, Mat *B)
{
  Mat                        C;
  Mat_Htool                 *a, *c;
  PetscMPIInt                rank;
  PetscScalar                shift, scale;
  htool::Cluster<PetscReal> *source_cluster;

  PetscFunctionBegin;
  PetscCall(MatShellGetScalingShifts(A, &shift, &scale, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
  PetscCall(MatShellGetContext(A, &a));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &C));
  PetscCall(MatSetSizes(C, A->rmap->n, A->cmap->n, A->rmap->N, A->cmap->N));
  PetscCall(MatSetBlockSizesFromMats(C, A, A));
  PetscCall(MatSetType(C, MATHTOOL));
  PetscCall(MatSetUp(C));
  PetscCall(MatPropagateSymmetryOptions(A, C));
  PetscCall(MatShellGetContext(C, &c));
  c->dim = a->dim;
  if (a->gcoords_target) {
    PetscCall(PetscMalloc1(A->rmap->N * c->dim, &c->gcoords_target));
    PetscCall(PetscArraycpy(c->gcoords_target, a->gcoords_target, A->rmap->N * c->dim));
  }
  if (a->gcoords_source == a->gcoords_target) c->gcoords_source = c->gcoords_target;
  else if (a->gcoords_source) {
    PetscCall(PetscMalloc1(A->cmap->N * c->dim, &c->gcoords_source));
    PetscCall(PetscArraycpy(c->gcoords_source, a->gcoords_source, A->cmap->N * c->dim));
  }
  c->max_cluster_leaf_size  = a->max_cluster_leaf_size;
  c->epsilon                = a->epsilon;
  c->eta                    = a->eta;
  c->depth[0]               = a->depth[0];
  c->depth[1]               = a->depth[1];
  c->block_tree_consistency = a->block_tree_consistency;
  c->permutation            = a->permutation;
  c->recompression          = a->recompression;
  c->compressor             = a->compressor;
  c->clustering             = a->clustering;
  c->kernel                 = a->kernel;
  c->kernelctx              = a->kernelctx;
  // no copy of wrapper because it can be created within MatAssemblyEnd() or useless
  if (a->local_hmatrix) {
    c->target_cluster = a->target_cluster;
    if (a->source_cluster) {
      c->source_cluster = a->source_cluster;
      source_cluster    = c->source_cluster.get();
    } else source_cluster = c->target_cluster.get();
    c->local_hmatrix = std::make_unique<htool::HMatrix<PetscScalar>>(*a->local_hmatrix);
    if (a->global_to_local_operator) {
      PetscCallMPI(MPI_Comm_rank(PetscObjectComm((PetscObject)A), &rank));
      c->global_to_local_operator    = std::make_unique<htool::RestrictedGlobalToLocalHMatrix<PetscScalar>>(*c->local_hmatrix, c->local_hmatrix->get_target_cluster(), c->local_hmatrix->get_source_cluster(), false, false);
      c->distributed_operator_holder = std::make_unique<htool::CustomApproximationBuilder<PetscScalar>>(*c->target_cluster, *source_cluster, PetscObjectComm((PetscObject)C), *c->global_to_local_operator);
      c->distributed_operator        = &c->distributed_operator_holder->distributed_operator;
      c->block_diagonal_hmatrix      = c->local_hmatrix->get_sub_hmatrix(c->target_cluster->get_cluster_on_partition(rank), source_cluster->get_cluster_on_partition(rank));
      c->local_hmatrix_view          = c->local_hmatrix.get();
    } else if (a->local_to_local_operator) {
      c->local_to_local_operator = std::make_unique<htool::LocalToLocalHMatrix<PetscScalar>>(*a->block_diagonal_hmatrix);
      c->distributed_operator_holder = std::make_unique<htool::CustomApproximationBuilder<PetscScalar>>(a->block_diagonal_hmatrix->get_target_cluster(), a->block_diagonal_hmatrix->get_source_cluster(), PetscObjectComm((PetscObject)A), *c->local_to_local_operator);
      c->distributed_operator   = &c->distributed_operator_holder->distributed_operator;
      c->block_diagonal_hmatrix = c->local_hmatrix.get();
      c->local_hmatrix_view     = c->local_hmatrix.get();
    }
    C->assembled = PETSC_TRUE;
  } else if (op != MAT_DO_NOT_COPY_VALUES) {
    PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));
  }
  if (C->assembled == PETSC_TRUE) {
    PetscCall(MatScale(C, scale));
    PetscCall(MatShift(C, shift));
  }
  *B = C;
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
  if (a->permutation) PetscCallExternalVoid("add_distributed_operator_vector_product_local_to_local", htool::add_distributed_operator_vector_product_local_to_local<PetscScalar>('N', 1.0, *a->distributed_operator, in, 0.0, out, nullptr));
  else PetscCallExternalVoid("internal_add_distributed_operator_vector_product_local_to_local", htool::internal_add_distributed_operator_vector_product_local_to_local<PetscScalar>('N', 1.0, *a->distributed_operator, in, 0.0, out, nullptr));
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
  if (a->permutation) PetscCallExternalVoid("add_distributed_operator_vector_product_local_to_local", htool::add_distributed_operator_vector_product_local_to_local<PetscScalar>('T', 1.0, *a->distributed_operator, in, 0.0, out, nullptr));
  else PetscCallExternalVoid("internal_add_distributed_operator_vector_product_local_to_local", htool::internal_add_distributed_operator_vector_product_local_to_local<PetscScalar>('T', 1.0, *a->distributed_operator, in, 0.0, out, nullptr));
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
  Mat_Htool      *a, *d;
  PetscScalar    *ptr;
  PetscScalar     shift, scale;
  const PetscInt *idxr, *idxc, *it;
  PetscInt        nrow, m;
  PetscBool       flg;

  PetscFunctionBegin;
  PetscCall(MatShellGetScalingShifts(A, &shift, &scale, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
  PetscCall(MatShellGetContext(A, &a));
  if (scall != MAT_REUSE_MATRIX) PetscCall(PetscCalloc1(n, submat));
  for (PetscInt i = 0; i < n; ++i) {
    PetscCall(ISGetLocalSize(irow[i], &nrow));
    PetscCall(ISGetLocalSize(icol[i], &m));
    PetscCall(ISGetIndices(irow[i], &idxr));
    PetscCall(ISGetIndices(icol[i], &idxc));
    flg = PETSC_FALSE;
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
                PetscInt  nb, nd, na, nblk, didx, bidx = -1, aidx = -1;
                PetscInt  blk_sz[3], blk_off[3];
                Mat       submats[9] = {}, D, B, BT;
                PetscBool sym        = (PetscBool)(A->symmetric == PETSC_BOOL3_TRUE || A->hermitian == PETSC_BOOL3_TRUE);

                /* fast extraction when the local diagonal block is part of the submatrix, e.g., for PCASM or PCHPDDM:
                 * returns a MATNEST with the local MATHTOOL block D and dense off-diagonal blocks
                 *      [   B   C   E   ]
                 *  A = [   B   D   E   ]
                 *      [   B   F   E   ]
                 */
                nb   = (PetscInt)std::distance(idxr, it); /* size of "before" partition (may be 0) */
                nd   = A->rmap->n;                        /* size of local diagonal block */
                na   = nrow - nb - nd;                    /* size of "after" partition (may be 0) */
                nblk = (nb > 0 ? 1 : 0) + 1 + (na > 0 ? 1 : 0);
                didx = (nb > 0 ? 1 : 0); /* row/column index of D in the MATNEST */
                if (nb > 0) bidx = 0;
                if (na > 0) aidx = nblk - 1;

                /* block sizes and offsets indexed by MATNEST position */
                if (nb > 0) {
                  blk_sz[bidx]  = nb;
                  blk_off[bidx] = 0;
                }
                blk_sz[didx]  = nd;
                blk_off[didx] = nb;
                if (na > 0) {
                  blk_sz[aidx]  = na;
                  blk_off[aidx] = nb + nd;
                }

                PetscCall(MatCreate(PETSC_COMM_SELF, &D));
                PetscCall(MatSetSizes(D, nd, nd, nd, nd));
                PetscCall(MatSetType(D, MATHTOOL));
                PetscCall(MatSetUp(D));
                PetscCall(MatPropagateSymmetryOptions(A, D));
                PetscCall(MatShellGetContext(D, &d));
                d->dim = a->dim;
                if (a->gcoords_target) {
                  PetscCall(PetscMalloc1(A->rmap->N * d->dim, &d->gcoords_target));
                  PetscCall(PetscArraycpy(d->gcoords_target, a->gcoords_target, A->rmap->N * d->dim));
                }
                if (a->gcoords_source == a->gcoords_target) d->gcoords_source = d->gcoords_target;
                else if (a->gcoords_source) {
                  PetscCall(PetscMalloc1(A->cmap->N * d->dim, &d->gcoords_source));
                  PetscCall(PetscArraycpy(d->gcoords_source, a->gcoords_source, A->cmap->N * d->dim));
                }
                d->max_cluster_leaf_size  = a->max_cluster_leaf_size;
                d->epsilon                = a->epsilon;
                d->eta                    = a->eta;
                d->depth[0]               = a->depth[0];
                d->depth[1]               = a->depth[1];
                d->block_tree_consistency = a->block_tree_consistency;
                d->permutation            = a->permutation;
                d->recompression          = a->recompression;
                d->compressor             = a->compressor;
                d->clustering             = a->clustering;
                d->kernel                 = a->kernel;
                d->kernelctx              = a->kernelctx;
                d->target_cluster         = a->target_cluster;
                if (a->source_cluster) d->source_cluster = a->source_cluster;
                d->local_hmatrix           = std::make_unique<htool::HMatrix<PetscScalar>>(*a->block_diagonal_hmatrix);
                d->local_to_local_operator = std::make_unique<htool::LocalToLocalHMatrix<PetscScalar>>(*d->local_hmatrix);
                d->distributed_operator_holder = std::make_unique<htool::CustomApproximationBuilder<PetscScalar>>(a->block_diagonal_hmatrix->get_target_cluster(), a->block_diagonal_hmatrix->get_source_cluster(), PetscObjectComm((PetscObject)A), *d->local_to_local_operator);
                d->distributed_operator   = &d->distributed_operator_holder->distributed_operator;
                d->block_diagonal_hmatrix = d->local_hmatrix.get();
                d->local_hmatrix_view     = d->local_hmatrix.get();
                D->assembled              = PETSC_TRUE;

                if (scall != MAT_REUSE_MATRIX) {
                  /* diagonal MATHTOOL block and dense off-diagonal blocks */
                  submats[didx * nblk + didx] = D;
                  PetscCall(PetscObjectReference((PetscObject)D));
                  for (PetscInt kr = 0; kr < nblk; kr++) {
                    for (PetscInt kc = (sym ? kr : 0); kc < nblk; kc++) {
                      if (kr == didx && kc == didx) continue;
                      PetscCall(MatCreateDense(PETSC_COMM_SELF, blk_sz[kr], blk_sz[kc], blk_sz[kr], blk_sz[kc], nullptr, &submats[kr * nblk + kc]));
                    }
                  }
                  PetscCall(MatCreateNest(PETSC_COMM_SELF, nblk, nullptr, nblk, nullptr, submats, (*submat) + i));
                  for (PetscInt k = 0; k < nblk * nblk; k++) PetscCall(MatDestroy(&submats[k]));
                } else PetscCall(MatNestSetSubMat((*submat)[i], didx, didx, D));
                PetscCall(MatDestroy(&D));

                /* fill MATDENSE off-diagonal blocks; upper-triangle (kc >= kr) first, then exploit symmetry */
                for (PetscInt kr = 0; kr < nblk; kr++) {
                  for (PetscInt kc = (sym ? kr : 0); kc < nblk; kc++) {
                    if (kr == didx && kc == didx) continue; /* MATHTOOL diagonal, skip */
                    PetscCall(MatNestGetSubMat((*submat)[i], kr, kc, &B));
                    PetscCall(MatDenseGetArrayWrite(B, &ptr));
                    a->wrapper->copy_submatrix(blk_sz[kr], blk_sz[kc], idxr + blk_off[kr], idxc + blk_off[kc], ptr);
                    PetscCall(MatDenseRestoreArrayWrite(B, &ptr));
                  }
                }
                if (sym && scall == MAT_REUSE_MATRIX) { /* need to reset the lower-triangular blocks to avoid having them being MatScale() and MatShift() twice */
                  for (PetscInt kr = 0; kr < nblk; kr++) {
                    for (PetscInt kc = 0; kc < kr; kc++) PetscCall(MatNestSetSubMat((*submat)[i], kr, kc, nullptr));
                  }
                }
                PetscCall(MatScale((*submat)[i], scale));
                PetscCall(MatShift((*submat)[i], shift)); /* MatScale() and MatShift() before filling the lower-triangular blocks */
                /* exploit symmetry: lower-triangular blocks are (conjugate) transposes of the upper ones */
                if (sym) {
                  for (PetscInt kr = 0; kr < nblk; kr++) {
                    for (PetscInt kc = 0; kc < kr; kc++) {
                      PetscCall(MatNestGetSubMat((*submat)[i], kc, kr, &B)); /* upper block (already filled) */
                      if (A->hermitian == PETSC_BOOL3_TRUE && PetscDefined(USE_COMPLEX)) PetscCall(MatCreateHermitianTranspose(B, &BT));
                      else PetscCall(MatCreateTranspose(B, &BT));
                      PetscCall(MatNestSetSubMat((*submat)[i], kr, kc, BT));
                      PetscCall(MatDestroy(&BT));
                    }
                  }
                }
              } /* complete local diagonal block in IS */
            } else flg = PETSC_FALSE; /* IS not long enough to store the local diagonal block */
          } else flg = PETSC_FALSE;   /* rmap->rstart not in IS */
        } /* unsorted IS */
      }
    } else flg = PETSC_FALSE; /* different row and column IS */
    if (!flg) {               /* dense fallback: reassemble everything */
      if (scall != MAT_REUSE_MATRIX) PetscCall(MatCreateDense(PETSC_COMM_SELF, nrow, m, nrow, m, nullptr, (*submat) + i));
      PetscCall(MatDenseGetArrayWrite((*submat)[i], &ptr));
      a->wrapper->copy_submatrix(nrow, m, idxr, idxc, ptr);
      PetscCall(MatDenseRestoreArrayWrite((*submat)[i], &ptr));
    }
    PetscCall(ISRestoreIndices(irow[i], &idxr));
    PetscCall(ISRestoreIndices(icol[i], &idxc));
    if (!flg) {
      PetscCall(MatScale((*submat)[i], scale));
      PetscCall(MatShift((*submat)[i], shift));
    }
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
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolGetEpsilon_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolSetEpsilon_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolGetEta_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolSetEta_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolGetMaxClusterLeafSize_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolSetMaxClusterLeafSize_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolGetMinTargetDepth_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolSetMinTargetDepth_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolGetMinSourceDepth_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolSetMinSourceDepth_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolGetBlockTreeConsistency_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolSetBlockTreeConsistency_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolGetCompressorType_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolSetCompressorType_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolGetClusteringType_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolSetClusteringType_C", nullptr));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolCreateFromKernel_C", nullptr));
  PetscCall(PetscObjectQuery((PetscObject)A, "KernelTranspose", (PetscObject *)&container));
  if (container) { /* created in MatTranspose_Htool() */
    PetscCall(PetscContainerGetPointer(container, &kernelt));
    PetscCall(MatDestroy(&kernelt->A));
    PetscCall(PetscObjectCompose((PetscObject)A, "KernelTranspose", nullptr));
  }
  if (a->gcoords_source != a->gcoords_target) PetscCall(PetscFree(a->gcoords_source));
  PetscCall(PetscFree(a->gcoords_target));
  a->distributed_operator_holder.reset();
  a->global_to_local_operator.reset();
  a->local_to_local_operator.reset();
  a->local_hmatrix.reset();
  a->source_cluster.reset();
  a->target_cluster.reset();
  delete a->wrapper;
  PetscCall(PetscFree(a));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatShellSetContext_C", nullptr)); // needed to avoid a call to MatShellSetContext_Immutable()
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatView_Htool_Draw_Zoom(PetscDraw draw, void *ptr)
{
  Mat                                                         A = (Mat)ptr;
  Mat_Htool                                                  *a;
  PetscReal                                                   x_r, y_r, x_l, y_l, w, h;
  PetscInt                                                    min_max[2] = {PETSC_INT_MAX, 0};
  const int                                                   greens[]   = {PETSC_DRAW_LIMEGREEN, PETSC_DRAW_FORESTGREEN, PETSC_DRAW_DARKGREEN};
  int                                                         color;
  char                                                        str[16];
  std::vector<const htool::HMatrix<PetscScalar, PetscReal> *> dense_blocks, low_rank_blocks;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  PetscCallExternalVoid("get_leaves", htool::get_leaves(*a->local_hmatrix_view, dense_blocks, low_rank_blocks));
  for (const htool::HMatrix<PetscScalar, PetscReal> *block : low_rank_blocks) {
    const PetscInt rank = block->get_rank();

    if (rank < min_max[0]) min_max[0] = rank;
    if (rank > min_max[1]) min_max[1] = rank;
  }
  PetscCall(PetscGlobalMinMaxInt(PetscObjectComm((PetscObject)A), min_max, min_max));
  if (min_max[0] == PETSC_INT_MAX) min_max[0] = min_max[1];
  PetscCall(PetscDrawStringGetSize(draw, &w, &h));
  PetscDrawCollectiveBegin(draw);
  for (const htool::HMatrix<PetscScalar, PetscReal> *block : dense_blocks) {
    x_l = x_r = (PetscReal)block->get_source_cluster().get_offset();
    x_r += (PetscReal)block->get_source_cluster().get_size();
    y_l = y_r = (PetscReal)(A->rmap->N - block->get_target_cluster().get_offset());
    y_l -= (PetscReal)block->get_target_cluster().get_size();
    PetscCall(PetscDrawRectangle(draw, x_l, y_l, x_r, y_r, PETSC_DRAW_RED, PETSC_DRAW_RED, PETSC_DRAW_RED, PETSC_DRAW_RED));
    PetscCall(PetscDrawLine(draw, x_l, y_l, x_r, y_l, PETSC_DRAW_BLACK));
    PetscCall(PetscDrawLine(draw, x_r, y_l, x_r, y_r, PETSC_DRAW_BLACK));
    PetscCall(PetscDrawLine(draw, x_r, y_r, x_l, y_r, PETSC_DRAW_BLACK));
    PetscCall(PetscDrawLine(draw, x_l, y_r, x_l, y_l, PETSC_DRAW_BLACK));
  }
  for (const htool::HMatrix<PetscScalar, PetscReal> *block : low_rank_blocks) {
    PetscReal      th;
    const PetscInt rank = block->get_rank();

    x_l = x_r = (PetscReal)block->get_source_cluster().get_offset();
    x_r += (PetscReal)block->get_source_cluster().get_size();
    y_l = y_r = (PetscReal)(A->rmap->N - block->get_target_cluster().get_offset());
    y_l -= (PetscReal)block->get_target_cluster().get_size();
    if (min_max[1] > min_max[0]) color = greens[(int)((PetscReal)(rank - min_max[0]) / (PetscReal)(min_max[1] - min_max[0]) * (PETSC_STATIC_ARRAY_LENGTH(greens) - 1) + 0.5)];
    else color = greens[PETSC_STATIC_ARRAY_LENGTH(greens) - 1];
    PetscCall(PetscDrawRectangle(draw, x_l, y_l, x_r, y_r, color, color, color, color));
    PetscCall(PetscDrawLine(draw, x_l, y_l, x_r, y_l, PETSC_DRAW_BLACK));
    PetscCall(PetscDrawLine(draw, x_r, y_l, x_r, y_r, PETSC_DRAW_BLACK));
    PetscCall(PetscDrawLine(draw, x_r, y_r, x_l, y_r, PETSC_DRAW_BLACK));
    PetscCall(PetscDrawLine(draw, x_l, y_r, x_l, y_l, PETSC_DRAW_BLACK));
    PetscCall(PetscSNPrintf(str, sizeof(str), "%d", rank));
    PetscCall(PetscDrawStringGetSize(draw, nullptr, &th));
    if (x_r - x_l > 4 * w && y_r - y_l > 4 * h) PetscCall(PetscDrawStringCentered(draw, 0.5 * (x_l + x_r), 0.5 * (y_l + y_r) - th / 2, PETSC_DRAW_BLACK, str));
  }
  PetscDrawCollectiveEnd(draw);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatView_Htool_Draw(Mat A, PetscViewer viewer)
{
  PetscDraw draw;
  PetscReal x_r = (PetscReal)A->cmap->N, y_r = (PetscReal)A->rmap->N, w, h;
  PetscBool flg;

  PetscFunctionBegin;
  PetscCall(PetscViewerDrawGetDraw(viewer, 0, &draw));
  PetscCall(PetscDrawIsNull(draw, &flg));
  if (flg) PetscFunctionReturn(PETSC_SUCCESS);
  w = x_r / 10.0;
  h = y_r / 10.0;
  PetscCall(PetscDrawSetCoordinates(draw, -w, -h, x_r + w, y_r + h));
  PetscCall(PetscObjectCompose((PetscObject)A, "Zoomviewer", (PetscObject)viewer));
  PetscCall(PetscDrawZoom(draw, MatView_Htool_Draw_Zoom, A));
  PetscCall(PetscObjectCompose((PetscObject)A, "Zoomviewer", nullptr));
  PetscCall(PetscDrawSave(draw));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatView_Htool(Mat A, PetscViewer pv)
{
  Mat_Htool                         *a;
  PetscScalar                        shift, scale;
  PetscBool                          flg;
  std::map<std::string, std::string> hmatrix_information;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERDRAW, &flg));
  if (flg) PetscCall(MatView_Htool_Draw(A, pv));
  else {
    PetscCall(MatShellGetContext(A, &a));
    hmatrix_information = htool::get_distributed_hmatrix_information(*a->local_hmatrix_view, PetscObjectComm((PetscObject)A));
    PetscCall(PetscObjectTypeCompare((PetscObject)pv, PETSCVIEWERASCII, &flg));
    if (flg) {
      PetscCall(MatShellGetScalingShifts(A, &shift, &scale, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
      PetscCall(PetscViewerASCIIPrintf(pv, "symmetry: %c\n", a->block_diagonal_hmatrix ? a->block_diagonal_hmatrix->get_symmetry() : 'N'));
      if (PetscAbsScalar(scale - 1.0) > PETSC_MACHINE_EPSILON) {
#if PetscDefined(USE_COMPLEX)
        PetscCall(PetscViewerASCIIPrintf(pv, "scaling: %g+%gi\n", (double)PetscRealPart(scale), (double)PetscImaginaryPart(scale)));
#else
        PetscCall(PetscViewerASCIIPrintf(pv, "scaling: %g\n", (double)scale));
#endif
      }
      if (PetscAbsScalar(shift) > PETSC_MACHINE_EPSILON) {
#if PetscDefined(USE_COMPLEX)
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
      PetscCall(PetscViewerASCIIPrintf(pv, "block tree consistency: %s\n", PetscBools[a->local_hmatrix_view->is_block_tree_consistent()]));
      PetscCall(PetscViewerASCIIPrintf(pv, "recompression: %s\n", PetscBools[a->recompression]));
      PetscCall(PetscViewerASCIIPrintf(pv, "number of dense (resp. low rank) matrices: %s (resp. %s)\n", hmatrix_information["Number_of_dense_blocks"].c_str(), hmatrix_information["Number_of_low_rank_blocks"].c_str()));
      PetscCall(
        PetscViewerASCIIPrintf(pv, "(minimum, mean, maximum) dense block sizes: (%s, %s, %s)\n", hmatrix_information["Dense_block_size_min"].c_str(), hmatrix_information["Dense_block_size_mean"].c_str(), hmatrix_information["Dense_block_size_max"].c_str()));
      PetscCall(PetscViewerASCIIPrintf(pv, "(minimum, mean, maximum) low rank block sizes: (%s, %s, %s)\n", hmatrix_information["Low_rank_block_size_min"].c_str(), hmatrix_information["Low_rank_block_size_mean"].c_str(),
                                       hmatrix_information["Low_rank_block_size_max"].c_str()));
      PetscCall(PetscViewerASCIIPrintf(pv, "(minimum, mean, maximum) ranks: (%s, %s, %s)\n", hmatrix_information["Rank_min"].c_str(), hmatrix_information["Rank_mean"].c_str(), hmatrix_information["Rank_max"].c_str()));
    }
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
    PetscCallExternalVoid("scal", htool::Blas<PetscScalar>::scal(&bn, &scale, *v, &one));
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
  PetscReal  r;
  PetscInt   n;
  PetscBool  b, flg, changed = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  PetscOptionsHeadBegin(PetscOptionsObject, "Htool options");
  n = a->max_cluster_leaf_size;
  PetscCall(PetscOptionsBoundedInt("-mat_htool_max_cluster_leaf_size", "Maximal leaf size in cluster tree", nullptr, a->max_cluster_leaf_size, &n, &flg, 0));
  if (flg) {
    if (n != a->max_cluster_leaf_size) changed = PETSC_TRUE;
    a->max_cluster_leaf_size = n;
  }
  r = a->epsilon;
  PetscCall(PetscOptionsBoundedReal("-mat_htool_epsilon", "Relative error in Frobenius norm when approximating a block", nullptr, a->epsilon, &r, &flg, 0.0));
  if (flg) {
    if (r != a->epsilon) changed = PETSC_TRUE;
    a->epsilon = r;
  }
  r = a->eta;
  PetscCall(PetscOptionsReal("-mat_htool_eta", "Admissibility condition tolerance", nullptr, a->eta, &r, &flg));
  if (flg) {
    if (r != a->eta) changed = PETSC_TRUE;
    a->eta = r;
  }
  n = a->depth[0];
  PetscCall(PetscOptionsBoundedInt("-mat_htool_min_target_depth", "Minimal cluster tree depth associated with the rows", nullptr, a->depth[0], &n, &flg, 0));
  if (flg) {
    if (n != a->depth[0]) changed = PETSC_TRUE;
    a->depth[0] = n;
  }
  n = a->depth[1];
  PetscCall(PetscOptionsBoundedInt("-mat_htool_min_source_depth", "Minimal cluster tree depth associated with the columns", nullptr, a->depth[1], &n, &flg, 0));
  if (flg) {
    if (n != a->depth[1]) changed = PETSC_TRUE;
    a->depth[1] = n;
  }
  b = a->block_tree_consistency;
  PetscCall(PetscOptionsBool("-mat_htool_block_tree_consistency", "Block tree consistency", nullptr, a->block_tree_consistency, &b, &flg));
  if (flg) {
    if (b != a->block_tree_consistency) changed = PETSC_TRUE;
    a->block_tree_consistency = b;
  }
  b = a->recompression;
  PetscCall(PetscOptionsBool("-mat_htool_recompression", "Use recompression", nullptr, a->recompression, &b, &flg));
  if (flg) {
    if (b != a->recompression) changed = PETSC_TRUE;
    a->recompression = b;
  }
  n = static_cast<PetscInt>(a->compressor);
  PetscCall(PetscOptionsEList("-mat_htool_compressor", "Type of compression", "MatHtoolCompressorType", MatHtoolCompressorTypes, PETSC_STATIC_ARRAY_LENGTH(MatHtoolCompressorTypes), MatHtoolCompressorTypes[a->compressor], &n, &flg));
  if (flg) {
    if (n != static_cast<PetscInt>(a->compressor)) changed = PETSC_TRUE;
    a->compressor = MatHtoolCompressorType(n);
  }
  n = static_cast<PetscInt>(a->clustering);
  PetscCall(PetscOptionsEList("-mat_htool_clustering", "Type of clustering", "MatHtoolClusteringType", MatHtoolClusteringTypes, PETSC_STATIC_ARRAY_LENGTH(MatHtoolClusteringTypes), MatHtoolClusteringTypes[a->clustering], &n, &flg));
  if (flg) {
    if (n != static_cast<PetscInt>(a->clustering)) changed = PETSC_TRUE;
    a->clustering = MatHtoolClusteringType(n);
  }
  PetscOptionsHeadEnd();
  if (changed) A->assembled = PETSC_FALSE;
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
  for (size_t i = 0; i < PETSC_STATIC_ARRAY_LENGTH(HtoolCite); ++i) PetscCall(PetscCitationsRegister(HtoolCitations[i], HtoolCite + i));
  PetscCall(MatShellGetContext(A, &a));
  if (A->was_assembled != PETSC_TRUE) {
    delete a->wrapper;
    a->target_cluster.reset();
    a->source_cluster.reset();
    a->local_hmatrix.reset();
    a->local_to_local_operator.reset();
    a->global_to_local_operator.reset();
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
    a->target_cluster = std::make_shared<htool::Cluster<PetscReal>>(recursive_build_strategy.create_cluster_tree_from_local_partition(A->rmap->N, a->dim, a->gcoords_target, 2, size, offset));
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
      a->source_cluster = std::make_shared<htool::Cluster<PetscReal>>(recursive_build_strategy.create_cluster_tree_from_local_partition(A->cmap->N, a->dim, a->gcoords_source, 2, size, offset));
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
    } else hmatrix_builder.set_low_rank_generator(compressor);
    hmatrix_builder.set_minimal_target_depth(a->depth[0]);
    hmatrix_builder.set_minimal_source_depth(a->depth[1]);
    PetscCheck(a->block_tree_consistency || (!a->block_tree_consistency && !(A->symmetric == PETSC_BOOL3_TRUE || A->hermitian == PETSC_BOOL3_TRUE)), PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Cannot have a MatHtool with inconsistent block tree which is either symmetric or Hermitian");
    hmatrix_builder.set_block_tree_consistency(a->block_tree_consistency);
    a->local_hmatrix               = std::make_unique<htool::HMatrix<PetscScalar>>(hmatrix_builder.build(a->wrapper ? *a->wrapper : *generator, *a->target_cluster, *source_cluster, rank, rank));
    a->global_to_local_operator    = std::make_unique<htool::RestrictedGlobalToLocalHMatrix<PetscScalar>>(*a->local_hmatrix, a->local_hmatrix->get_target_cluster(), a->local_hmatrix->get_source_cluster(), false, false);
    a->distributed_operator_holder = std::make_unique<htool::CustomApproximationBuilder<PetscScalar>>(*a->target_cluster, *source_cluster, PetscObjectComm((PetscObject)A), *a->global_to_local_operator);
    a->distributed_operator        = &a->distributed_operator_holder->distributed_operator;
    a->block_diagonal_hmatrix      = a->local_hmatrix->get_sub_hmatrix(a->target_cluster->get_cluster_on_partition(rank), source_cluster->get_cluster_on_partition(rank));
    a->local_hmatrix_view          = a->local_hmatrix.get();
  }
  A->was_assembled = PETSC_FALSE;
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
    if (a->permutation) PetscCallExternalVoid("add_distributed_operator_matrix_product_local_to_local", htool::add_distributed_operator_matrix_product_local_to_local<PetscScalar>('N', 1.0, *a->distributed_operator, in, 0.0, out, N, nullptr));
    else PetscCallExternalVoid("internal_add_distributed_operator_matrix_product_local_to_local", htool::internal_add_distributed_operator_matrix_product_local_to_local<PetscScalar>('N', 1.0, *a->distributed_operator, in, 0.0, out, N, nullptr));
    break;
  case MATPRODUCT_AtB:
    if (a->permutation) PetscCallExternalVoid("add_distributed_operator_matrix_product_local_to_local", htool::add_distributed_operator_matrix_product_local_to_local<PetscScalar>('T', 1.0, *a->distributed_operator, in, 0.0, out, N, nullptr));
    else PetscCallExternalVoid("internal_add_distributed_operator_matrix_product_local_to_local", htool::internal_add_distributed_operator_matrix_product_local_to_local<PetscScalar>('T', 1.0, *a->distributed_operator, in, 0.0, out, N, nullptr));
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

static PetscErrorCode MatHtoolGetHierarchicalMat_Htool(Mat A, void *distributed_operator)
{
  Mat_Htool *a;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  *(const void **)distributed_operator = static_cast<const void *>(a->distributed_operator);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatHtoolSetKernel_Htool(Mat A, MatHtoolKernelFn *kernel, void *kernelctx)
{
  Mat_Htool *a;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  if (kernel != a->kernel || a->kernelctx != kernelctx) A->assembled = PETSC_FALSE;
  a->kernel    = kernel;
  a->kernelctx = kernelctx;
  delete a->wrapper;
  if (a->kernel) a->wrapper = new WrapperHtool(a->dim, a->kernel, a->kernelctx);
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

static PetscErrorCode MatHtoolUsePermutation_Htool(Mat A, PetscBool use)
{
  Mat_Htool *a;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  a->permutation = use;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatHtoolUseRecompression_Htool(Mat A, PetscBool use)
{
  Mat_Htool *a;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  if (a->recompression != use) A->assembled = PETSC_FALSE;
  a->recompression = use;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#define PETSC_HTOOL_PARAMETER(Type, Name, member) \
  static PetscErrorCode MatHtoolGet##Name##_Htool(Mat A, Type *v) \
  { \
    Mat_Htool *a; \
    PetscFunctionBegin; \
    PetscCall(MatShellGetContext(A, &a)); \
    *v = a->member; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  } \
  static PetscErrorCode MatHtoolSet##Name##_Htool(Mat A, Type v) \
  { \
    Mat_Htool *a; \
    PetscFunctionBegin; \
    PetscCall(MatShellGetContext(A, &a)); \
    if (a->member != v) A->assembled = PETSC_FALSE; \
    a->member = v; \
    PetscFunctionReturn(PETSC_SUCCESS); \
  }

PETSC_HTOOL_PARAMETER(PetscReal, Epsilon, epsilon)
PETSC_HTOOL_PARAMETER(PetscReal, Eta, eta)
PETSC_HTOOL_PARAMETER(PetscInt, MaxClusterLeafSize, max_cluster_leaf_size)
PETSC_HTOOL_PARAMETER(PetscInt, MinTargetDepth, depth[0])
PETSC_HTOOL_PARAMETER(PetscInt, MinSourceDepth, depth[1])
PETSC_HTOOL_PARAMETER(PetscBool, BlockTreeConsistency, block_tree_consistency)
PETSC_HTOOL_PARAMETER(MatHtoolCompressorType, CompressorType, compressor)
PETSC_HTOOL_PARAMETER(MatHtoolClusteringType, ClusteringType, clustering)

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
  PetscCallExternalVoid("copy_to_dense_in_user_numbering", htool::copy_to_dense_in_user_numbering(*a->local_hmatrix_view, array));
  PetscCall(MatDenseRestoreArrayWrite(C, &array));
  PetscCall(MatScale(C, scale));
  PetscCall(MatShift(C, shift));
  if (reuse == MAT_INPLACE_MATRIX) PetscCall(MatHeaderReplace(A, &C));
  else *B = C;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GenEntriesTranspose(PetscInt sdim, PetscInt M, PetscInt N, const PetscInt *rows, const PetscInt *cols, PetscScalar *ptr, PetscCtx ctx)
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
    PetscCall(MatSetType(C, MATHTOOL));
    PetscCall(MatSetUp(C));
    PetscCall(PetscNew(&kernelt));
    PetscCall(PetscObjectContainerCompose((PetscObject)C, "KernelTranspose", kernelt, PetscCtxDestroyDefault));
  } else {
    C = *B;
    PetscCall(PetscObjectQuery((PetscObject)C, "KernelTranspose", (PetscObject *)&container));
    PetscCheck(container, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call MatTranspose() with MAT_INITIAL_MATRIX first");
    PetscCall(PetscContainerGetPointer(container, &kernelt));
  }
  PetscCall(MatShellGetContext(C, &c));
  c->dim    = a->dim;
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
  if (reuse != MAT_INITIAL_MATRIX) C->assembled = PETSC_FALSE; // so that C->was_assembled is not set to PETSC_TRUE
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatScale(C, scale));
  PetscCall(MatShift(C, shift));
  if (reuse == MAT_INITIAL_MATRIX) *B = C;
  PetscFunctionReturn(PETSC_SUCCESS);
}

struct MatFactorCtx {
  htool::HMatrix<PetscScalar> *hmatrix; /* factorized HMatrix filled by MatFactorNumeric_Htool() */
  PetscScalar                  scale;   /* scaling factor from MatShellGetScalingShifts(), applied as inverse scaling in Mat[Mat]Solve() */
};

static PetscErrorCode MatFactorCtxDestroy(PetscCtxRt ctx)
{
  MatFactorCtx *data = *reinterpret_cast<MatFactorCtx **>(ctx);

  PetscFunctionBegin;
  delete data->hmatrix;
  PetscCall(PetscFree(data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_Factor(Mat F)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectCompose((PetscObject)F, "HMatrix", nullptr));
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
  PetscContainer container;
  MatFactorCtx  *data;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)A, "HMatrix", (PetscObject *)&container));
  PetscCheck(container, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Must call Mat%sFactorNumeric() before Mat%sSolve%s()", A->factortype == MAT_FACTOR_LU ? "LU" : "Cholesky", X.nb_cols() == 1 ? "" : "Mat", trans == 'N' ? "" : "Transpose");
  PetscCall(PetscContainerGetPointer(container, &data));
  if (A->factortype == MAT_FACTOR_LU) PetscCallExternalVoid("lu_solve", htool::lu_solve(trans, *data->hmatrix, X));
  else PetscCallExternalVoid("cholesky_solve", htool::cholesky_solve('U', *data->hmatrix, X));
  PetscCallExternalVoid("scale", htool::scale(1.0 / data->scale, X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <char trans, class Type, typename std::enable_if<std::is_same<Type, Vec>::value>::type * = nullptr>
static PetscErrorCode MatSolve_Htool(Mat A, Type b, Type x)
{
  htool::Matrix<PetscScalar> v;
  PetscScalar               *array;
  PetscInt                   n;

  PetscFunctionBegin;
  PetscCall(VecGetLocalSize(b, &n));
  PetscCall(VecCopy(b, x));
  PetscCall(VecGetArrayWrite(x, &array));
  PetscCallCXX(v.assign(n, 1, array, false));
  PetscCall(VecRestoreArrayWrite(x, &array));
  PetscCall(MatSolve_Private<trans>(A, v));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <char trans, class Type, typename std::enable_if<std::is_same<Type, Mat>::value>::type * = nullptr>
static PetscErrorCode MatSolve_Htool(Mat A, Type B, Type X)
{
  htool::Matrix<PetscScalar> v;
  PetscScalar               *array;
  PetscInt                   m, N, lda;

  PetscFunctionBegin;
  PetscCall(MatGetLocalSize(B, &m, nullptr));
  PetscCall(MatGetLocalSize(B, nullptr, &N));
  PetscCall(MatDenseGetLDA(X, &lda));
  PetscCheck(lda == X->rmap->n, PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported leading dimension (%" PetscInt_FMT " != %" PetscInt_FMT ")", lda, X->rmap->n);
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
  Mat_Htool     *a;
  PetscContainer container;
  MatFactorCtx  *data;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  PetscCall(PetscObjectQuery((PetscObject)F, "HMatrix", (PetscObject *)&container));
  PetscCheck(container, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Mat%sFactorSymbolic() must be called before Mat%sFactorNumeric()", ftype == MAT_FACTOR_LU ? "LU" : "Cholesky", ftype == MAT_FACTOR_LU ? "LU" : "Cholesky");
  PetscCall(PetscContainerGetPointer(container, &data));
  if (ftype == MAT_FACTOR_LU) PetscCheck(a->local_hmatrix_view->get_UPLO() == 'N', PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "LU factorization requires a MATHTOOL with full storage");
  delete data->hmatrix;
  data->hmatrix = new htool::HMatrix<PetscScalar>(*a->local_hmatrix_view);
  PetscCall(MatShellGetScalingShifts(A, (PetscScalar *)MAT_SHELL_NOT_ALLOWED, &data->scale, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Vec *)MAT_SHELL_NOT_ALLOWED, (Mat *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED, (IS *)MAT_SHELL_NOT_ALLOWED));
  if (ftype == MAT_FACTOR_LU) PetscCallExternalVoid("sequential_lu_factorization", htool::sequential_lu_factorization(*data->hmatrix));
  else PetscCallExternalVoid("sequential_cholesky_factorization", htool::sequential_cholesky_factorization('U', *data->hmatrix));
  PetscFunctionReturn(PETSC_SUCCESS);
}

template <MatFactorType ftype>
PetscErrorCode MatFactorSymbolic_Htool(Mat F, Mat)
{
  PetscContainer container;
  MatFactorCtx  *data;

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
  PetscCall(PetscObjectQuery((PetscObject)F, "HMatrix", (PetscObject *)&container));
  if (!container) {
    PetscCall(PetscNew(&data));
    PetscCall(PetscObjectContainerCompose((PetscObject)F, "HMatrix", data, MatFactorCtxDestroy));
  }
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

  PetscCheck(ftype == MAT_FACTOR_LU || ftype == MAT_FACTOR_CHOLESKY, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "Only MAT_FACTOR_LU and MAT_FACTOR_CHOLESKY are supported");
  if (ftype == MAT_FACTOR_LU) B->ops->lufactorsymbolic = MatLUFactorSymbolic_Htool;
  else B->ops->choleskyfactorsymbolic = MatCholeskyFactorSymbolic_Htool;

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

static PetscErrorCode MatHtoolCreateFromKernel_Htool(Mat A, PetscInt spacedim, const PetscReal coords_target[], const PetscReal coords_source[], MatHtoolKernelFn *kernel, void *kernelctx)
{
  Mat_Htool *a;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &a));
  a->dim       = spacedim;
  a->kernel    = kernel;
  a->kernelctx = kernelctx;
  PetscCall(PetscCalloc1(A->rmap->N * spacedim, &a->gcoords_target));
  PetscCall(PetscArraycpy(a->gcoords_target + A->rmap->rstart * spacedim, coords_target, A->rmap->n * spacedim));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, a->gcoords_target, A->rmap->N * spacedim, MPIU_REAL, MPI_SUM, PetscObjectComm((PetscObject)A)));
  if (coords_target != coords_source) {
    PetscCall(PetscCalloc1(A->cmap->N * spacedim, &a->gcoords_source));
    PetscCall(PetscArraycpy(a->gcoords_source + A->cmap->rstart * spacedim, coords_source, A->cmap->n * spacedim));
    PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, a->gcoords_source, A->cmap->N * spacedim, MPIU_REAL, MPI_SUM, PetscObjectComm((PetscObject)A)));
  } else a->gcoords_source = a->gcoords_target;
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
  PetscCall(MatShellSetOperation(A, MATOP_DUPLICATE, (PetscErrorCodeFn *)MatDuplicate_Htool));
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
  A->assembled              = PETSC_FALSE; // MatCreate_Shell() forces this value to PETSC_TRUE
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
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolGetEpsilon_C", MatHtoolGetEpsilon_Htool));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolSetEpsilon_C", MatHtoolSetEpsilon_Htool));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolGetEta_C", MatHtoolGetEta_Htool));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolSetEta_C", MatHtoolSetEta_Htool));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolGetMaxClusterLeafSize_C", MatHtoolGetMaxClusterLeafSize_Htool));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolSetMaxClusterLeafSize_C", MatHtoolSetMaxClusterLeafSize_Htool));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolGetMinTargetDepth_C", MatHtoolGetMinTargetDepth_Htool));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolSetMinTargetDepth_C", MatHtoolSetMinTargetDepth_Htool));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolGetMinSourceDepth_C", MatHtoolGetMinSourceDepth_Htool));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolSetMinSourceDepth_C", MatHtoolSetMinSourceDepth_Htool));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolGetBlockTreeConsistency_C", MatHtoolGetBlockTreeConsistency_Htool));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolSetBlockTreeConsistency_C", MatHtoolSetBlockTreeConsistency_Htool));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolGetCompressorType_C", MatHtoolGetCompressorType_Htool));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolSetCompressorType_C", MatHtoolSetCompressorType_Htool));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolGetClusteringType_C", MatHtoolGetClusteringType_Htool));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolSetClusteringType_C", MatHtoolSetClusteringType_Htool));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatHtoolCreateFromKernel_C", MatHtoolCreateFromKernel_Htool));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatShellSetContext_C", MatShellSetContext_Immutable));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatShellSetContextDestroy_C", MatShellSetContextDestroy_Immutable));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatShellSetManageScalingShifts_C", MatShellSetManageScalingShifts_Immutable));
  PetscCall(PetscObjectChangeTypeName((PetscObject)A, MATHTOOL));
  PetscFunctionReturn(PETSC_SUCCESS);
}
