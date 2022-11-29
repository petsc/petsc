/*
  AIJHIPSPARSE methods implemented with HIP kernels. Uses hipSparse/Thrust maps from AIJHIPSPARSE
  Portions of this code are under:
  Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
*/
#define PETSC_SKIP_SPINLOCK

#include <petscconf.h>
#include <../src/mat/impls/aij/seq/aij.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#undef VecType
#include <../src/mat/impls/aij/seq/seqhipsparse/hipsparsematimpl.h>
#define AIJBANDUSEGROUPS 1
#if defined(AIJBANDUSEGROUPS)
  #include <hip/hip_cooperative_groups.h>
#endif

/*
  LU BAND factorization with optimization for block diagonal (Nf blocks) in natural order (-mat_no_inode -pc_factor_mat_ordering_type rcm with Nf>1 fields)

  requires:
     structurally symmetric: fix with transpose/column meta data
*/

static PetscErrorCode MatLUFactorSymbolic_SeqAIJHIPSPARSEBAND(Mat, Mat, IS, IS, const MatFactorInfo *);
static PetscErrorCode MatLUFactorNumeric_SeqAIJHIPSPARSEBAND(Mat, Mat, const MatFactorInfo *);
static PetscErrorCode MatSolve_SeqAIJHIPSPARSEBAND(Mat, Vec, Vec);

/*
  The GPU LU factor kernel
*/
__global__ void __launch_bounds__(1024, 1) mat_lu_factor_band_init_set_i(const PetscInt n, const int bw, int bi_csr[])
{
  const PetscInt Nf = gridDim.x, Nblk = gridDim.y, nloc = n / Nf;
  const PetscInt field = blockIdx.x, blkIdx = blockIdx.y;
  const PetscInt nloc_i = (nloc / Nblk + !!(nloc % Nblk)), start_i = field * nloc + blkIdx * nloc_i, end_i = (start_i + nloc_i) > (field + 1) * nloc ? (field + 1) * nloc : (start_i + nloc_i);

  // set i (row+1)
  if (threadIdx.x + threadIdx.y + blockIdx.x + blockIdx.y == 0) bi_csr[0] = 0; // dummy at zero
  for (int rowb = start_i + threadIdx.y; rowb < end_i; rowb += blockDim.y) {   // rows in block by thread y
    if (rowb < end_i && threadIdx.x == 0) {
      PetscInt i = rowb + 1, ni = (rowb > bw) ? bw + 1 : i, n1L = ni * (ni - 1) / 2, nug = i * bw, n2L = bw * ((rowb > bw) ? (rowb - bw) : 0), mi = bw + rowb + 1 - n, clip = (mi > 0) ? mi * (mi - 1) / 2 + mi : 0;
      bi_csr[rowb + 1] = n1L + nug - clip + n2L + i;
    }
  }
}
// copy AIJ to AIJ_BAND
__global__ void __launch_bounds__(1024, 1) mat_lu_factor_band_copy_aij_aij(const PetscInt n, const int bw, const PetscInt r[], const PetscInt ic[], const int ai_d[], const int aj_d[], const PetscScalar aa_d[], const int bi_csr[], PetscScalar ba_csr[])
{
  const PetscInt Nf = gridDim.x, Nblk = gridDim.y, nloc = n / Nf;
  const PetscInt field = blockIdx.x, blkIdx = blockIdx.y;
  const PetscInt nloc_i = (nloc / Nblk + !!(nloc % Nblk)), start_i = field * nloc + blkIdx * nloc_i, end_i = (start_i + nloc_i) > (field + 1) * nloc ? (field + 1) * nloc : (start_i + nloc_i);

  // zero B
  if (threadIdx.x + threadIdx.y + blockIdx.x + blockIdx.y == 0) ba_csr[bi_csr[n]] = 0; // flop count at end
  for (int rowb = start_i + threadIdx.y; rowb < end_i; rowb += blockDim.y) {           // rows in block by thread y
    if (rowb < end_i) {
      PetscScalar   *batmp = ba_csr + bi_csr[rowb];
      const PetscInt nzb   = bi_csr[rowb + 1] - bi_csr[rowb];
      for (int j = threadIdx.x; j < nzb; j += blockDim.x) {
        if (j < nzb) { batmp[j] = 0; }
      }
    }
  }

  // copy A into B with CSR format -- these two loops can be fused
  for (int rowb = start_i + threadIdx.y; rowb < end_i; rowb += blockDim.y) { // rows in block by thread y
    if (rowb < end_i) {
      const PetscInt     rowa = r[rowb], nza = ai_d[rowa + 1] - ai_d[rowa];
      const int         *ajtmp = aj_d + ai_d[rowa], bjStart = (rowb > bw) ? rowb - bw : 0;
      const PetscScalar *av    = aa_d + ai_d[rowa];
      PetscScalar       *batmp = ba_csr + bi_csr[rowb];
      /* load in initial (unfactored row) */
      for (int j = threadIdx.x; j < nza; j += blockDim.x) {
        if (j < nza) {
          PetscInt    colb = ic[ajtmp[j]], idx = colb - bjStart;
          PetscScalar vala = av[j];
          batmp[idx]       = vala;
        }
      }
    }
  }
}
// print AIJ_BAND
__global__ void print_mat_aij_band(const PetscInt n, const int bi_csr[], const PetscScalar ba_csr[])
{
  // debug
  if (threadIdx.x + threadIdx.y + blockIdx.x + blockIdx.y == 0) {
    printf("B (AIJ) n=%d:\n", (int)n);
    for (int rowb = 0; rowb < n; rowb++) {
      const PetscInt     nz    = bi_csr[rowb + 1] - bi_csr[rowb];
      const PetscScalar *batmp = ba_csr + bi_csr[rowb];
      for (int j = 0; j < nz; j++) printf("(%13.6e) ", PetscRealPart(batmp[j]));
      printf(" bi=%d\n", bi_csr[rowb + 1]);
    }
  }
}
// Band LU kernel ---  ba_csr bi_csr
__global__ void __launch_bounds__(1024, 1) mat_lu_factor_band(const PetscInt n, const PetscInt bw, const int bi_csr[], PetscScalar ba_csr[], int *use_group_sync)
{
  const PetscInt Nf = gridDim.x, Nblk = gridDim.y, nloc = n / Nf;
  const PetscInt field = blockIdx.x, blkIdx = blockIdx.y;
  const PetscInt start = field * nloc, end = start + nloc;
#if defined(AIJBANDUSEGROUPS)
  auto g = cooperative_groups::this_grid();
#endif
  // A22 panel update for each row A(1,:) and col A(:,1)
  for (int glbDD = start, locDD = 0; glbDD < end; glbDD++, locDD++) {
    PetscInt           tnzUd = bw, maxU = end - 1 - glbDD;                                        // we are chopping off the inter ears
    const PetscInt     nzUd = (tnzUd > maxU) ? maxU : tnzUd, dOffset = (glbDD > bw) ? bw : glbDD; // global to go past ears after first
    PetscScalar       *pBdd   = ba_csr + bi_csr[glbDD] + dOffset;
    const PetscScalar *baUd   = pBdd + 1; // vector of data  U(i,i+1:end)
    const PetscScalar  Bdd    = *pBdd;
    const PetscInt     offset = blkIdx * blockDim.y + threadIdx.y, inc = Nblk * blockDim.y;
    if (threadIdx.x == 0) {
      for (int idx = offset, myi = glbDD + offset + 1; idx < nzUd; idx += inc, myi += inc) { /* assuming symmetric structure */
        const PetscInt bwi = myi > bw ? bw : myi, kIdx = bwi - (myi - glbDD);                // cuts off just the first (global) block
        PetscScalar   *Aid = ba_csr + bi_csr[myi] + kIdx;
        *Aid               = *Aid / Bdd;
      }
    }
    __syncthreads(); // synch on threadIdx.x only
    for (int idx = offset, myi = glbDD + offset + 1; idx < nzUd; idx += inc, myi += inc) {
      const PetscInt    bwi = myi > bw ? bw : myi, kIdx = bwi - (myi - glbDD); // cuts off just the first (global) block
      PetscScalar      *Aid = ba_csr + bi_csr[myi] + kIdx;
      PetscScalar      *Aij = Aid + 1;
      const PetscScalar Lid = *Aid;
      for (int jIdx = threadIdx.x; jIdx < nzUd; jIdx += blockDim.x) { Aij[jIdx] -= Lid * baUd[jIdx]; }
    }
#if defined(AIJBANDUSEGROUPS)
    if (use_group_sync) {
      g.sync();
    } else {
      __syncthreads();
    }
#else
    __syncthreads();
#endif
  } /* endof for (i=0; i<n; i++) { */
}

static PetscErrorCode MatLUFactorNumeric_SeqAIJHIPSPARSEBAND(Mat B, Mat A, const MatFactorInfo *info)
{
  Mat_SeqAIJ                    *b                   = (Mat_SeqAIJ *)B->data;
  Mat_SeqAIJHIPSPARSETriFactors *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors *)B->spptr;
  PetscCheck(hipsparseTriFactors, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing hipsparseTriFactors");

  Mat_SeqAIJHIPSPARSE           *hipsparsestructA = (Mat_SeqAIJHIPSPARSE *)A->spptr;
  Mat_SeqAIJHIPSPARSEMultStruct *matstructA;
  CsrMatrix                     *matrixA;
  const PetscInt                 n = A->rmap->n, *ic, *r;
  const int                     *ai_d, *aj_d;
  const PetscScalar             *aa_d;
  PetscScalar                   *ba_t = hipsparseTriFactors->a_band_d;
  int                           *bi_t = hipsparseTriFactors->i_band_d;
  int                            Ni = 10, team_size = 9, Nf = 1, nVec = 56, nconcurrent = 1, nsm = -1; // Nf is batch size - not used

  PetscFunctionBegin;
  if (A->rmap->n == 0) PetscFunctionReturn(0);
  // hipsparse setup
  PetscCheck(hipsparsestructA, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing hipsparsestructA");
  matstructA = (Mat_SeqAIJHIPSPARSEMultStruct *)hipsparsestructA->mat; //  matstruct->cprowIndices
  PetscCheck(matstructA, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing mat struct");
  matrixA = (CsrMatrix *)matstructA->mat;
  PetscCheck(matrixA, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Missing matrix hipsparsestructA->mat->mat");

  // get data
  ic   = thrust::raw_pointer_cast(hipsparseTriFactors->cpermIndices->data());
  ai_d = thrust::raw_pointer_cast(matrixA->row_offsets->data());
  aj_d = thrust::raw_pointer_cast(matrixA->column_indices->data());
  aa_d = thrust::raw_pointer_cast(matrixA->values->data().get());
  r    = thrust::raw_pointer_cast(hipsparseTriFactors->rpermIndices->data());

  PetscCallHIP(WaitForHIP());
  PetscCall(PetscLogGpuTimeBegin());
  {
    int bw = (int)(2. * (double)n - 1. - (double)(PetscSqrtReal(1. + 4. * ((double)n * (double)n - (double)b->nz)) + PETSC_MACHINE_EPSILON)) / 2, bm1 = bw - 1, nl = n / Nf;
#if !defined(AIJBANDUSEGROUPS)
    Ni = 1 / nconcurrent;
    Ni = 1;
#else
    if (!hipsparseTriFactors->init_dev_prop) {
      int gpuid;
      hipsparseTriFactors->init_dev_prop = PETSC_TRUE;
      hipGetDevice(&gpuid);
      hipGetDeviceProperties(&hipsparseTriFactors->dev_prop, gpuid);
    }
    nsm = hipsparseTriFactors->dev_prop.multiProcessorCount;
    Ni  = nsm / Nf / nconcurrent;
#endif
    team_size = bw / Ni + !!(bw % Ni);
    nVec      = PetscMin(bw, 1024 / team_size);
    PetscCall(PetscInfo(A, "Matrix Bandwidth = %d, number SMs/block = %d, num concurency = %d, num fields = %d, numSMs/GPU = %d, thread group size = %d,%d\n", bw, Ni, nconcurrent, Nf, nsm, team_size, nVec));
    {
      dim3 dimBlockTeam(nVec, team_size);
      dim3 dimBlockLeague(Nf, Ni);
      hipLaunchKernelGGL(mat_lu_factor_band_copy_aij_aij, dim3(dimBlockLeague), dim3(dimBlockTeam), 0, 0, n, bw, r, ic, ai_d, aj_d, aa_d, bi_t, ba_t);
      PetscHIPCheckLaunch; // does a sync
#if defined(AIJBANDUSEGROUPS)
      if (Ni > 1) {
        void *kernelArgs[] = {(void *)&n, (void *)&bw, (void *)&bi_t, (void *)&ba_t, (void *)&nsm};
        hipLaunchCooperativeKernel((void *)mat_lu_factor_band, dimBlockLeague, dimBlockTeam, kernelArgs, 0, NULL);
      } else {
        hipLaunchKernelGGL(mat_lu_factor_band, dim3(dimBlockLeague), dim3(dimBlockTeam), 0, 0, n, bw, bi_t, ba_t, NULL);
      }
#else
      hipLaunchKernelGGL(mat_lu_factor_band, dim3(dimBlockLeague), dim3(dimBlockTeam), 0, 0, n, bw, bi_t, ba_t, NULL);
#endif
      PetscHIPCheckLaunch; // does a sync
#if defined(PETSC_USE_LOG)
      PetscCall(PetscLogGpuFlops((PetscLogDouble)Nf * (bm1 * (bm1 + 1) * (PetscLogDouble)(2 * bm1 + 1) / 3 + (PetscLogDouble)2 * (nl - bw) * bw * bw + (PetscLogDouble)nl * (nl + 1) / 2)));
#endif
    }
  }
  PetscCall(PetscLogGpuTimeEnd());
  /* determine which version of MatSolve needs to be used. from MatLUFactorNumeric_AIJ_SeqAIJHIPSPARSE */
  B->ops->solve             = MatSolve_SeqAIJHIPSPARSEBAND;
  B->ops->solvetranspose    = NULL; /* need transpose */
  B->ops->matsolve          = NULL;
  B->ops->matsolvetranspose = NULL;
  PetscFunctionReturn(0);
}

PetscErrorCode MatLUFactorSymbolic_SeqAIJHIPSPARSEBAND(Mat B, Mat A, IS isrow, IS iscol, const MatFactorInfo *info)
{
  Mat_SeqAIJ                    *a = (Mat_SeqAIJ *)A->data, *b;
  IS                             isicol;
  const PetscInt                *ic, *ai = a->i, *aj = a->j;
  PetscScalar                   *ba_t;
  int                           *bi_t;
  PetscInt                       i, n = A->rmap->n, Nf = 1; /* Nf batch size - not used */
  PetscInt                       nzBcsr, bwL, bwU;
  PetscBool                      missing;
  Mat_SeqAIJHIPSPARSETriFactors *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors *)B->spptr;

  PetscFunctionBegin;
  PetscCheck(A->rmap->N == A->cmap->N, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "matrix must be square");
  PetscCall(MatMissingDiagonal(A, &missing, &i));
  PetscCheck(!missing, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Matrix is missing diagonal entry %" PetscInt_FMT, i);
  PetscCheck(hipsparseTriFactors, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "!hipsparseTriFactors");
  PetscCall(MatGetOption(A, MAT_STRUCTURALLY_SYMMETRIC, &missing));
  PetscCheck(missing, PetscObjectComm((PetscObject)A), PETSC_ERR_SUP, "only structrally symmetric matrices supported");
  PetscCall(ISInvertPermutation(iscol, PETSC_DECIDE, &isicol));
  PetscCall(ISGetIndices(isicol, &ic));
  PetscCall(MatSeqAIJSetPreallocation_SeqAIJ(B, MAT_SKIP_ALLOCATION, NULL));
  PetscCall(PetscLogObjectParent((PetscObject)B, (PetscObject)isicol));
  b = (Mat_SeqAIJ *)(B)->data;

  /* get band widths, MatComputeBandwidth should take a reordering ic and do this */
  bwL = bwU = 0;
  for (int rwb = 0; rwb < n; rwb++) {
    const PetscInt rwa = ic[rwb], anz = ai[rwb + 1] - ai[rwb], *ajtmp = aj + ai[rwb];
    for (int j = 0; j < anz; j++) {
      PetscInt colb = ic[ajtmp[j]];
      if (colb < rwa) { // L
        if (rwa - colb > bwL) bwL = rwa - colb;
      } else {
        if (colb - rwa > bwU) bwU = colb - rwa;
      }
    }
  }
  PetscCall(ISRestoreIndices(isicol, &ic));
  /* only support structurally symmetric, but it might work */
  PetscCheck(bwL == bwU, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Only symmetric structure supported (now) W_L=%" PetscInt_FMT " W_U=%" PetscInt_FMT, bwL, bwU);
  PetscCall(MatSeqAIJHIPSPARSETriFactors_Reset(&hipsparseTriFactors));
  nzBcsr   = n + (2 * n - 1) * bwU - bwU * bwU;
  b->maxnz = b->nz         = nzBcsr;
  hipsparseTriFactors->nnz = b->nz; // only meta data needed: n & nz
  PetscCall(PetscInfo(A, "Matrix Bandwidth = %" PetscInt_FMT ", nnz = %" PetscInt_FMT "\n", bwL, b->nz));
  if (!hipsparseTriFactors->workVector) hipsparseTriFactors->workVector = new THRUSTARRAY(n);
  PetscCallHIP(hipMalloc(&ba_t, (b->nz + 1) * sizeof(PetscScalar))); // include a place for flops
  PetscCallHIP(hipMalloc(&bi_t, (n + 1) * sizeof(int)));
  hipsparseTriFactors->a_band_d = ba_t;
  hipsparseTriFactors->i_band_d = bi_t;
  /* In b structure:  Free imax, ilen, old a, old j.  Allocate solve_work, new a, new j */
  PetscCall(PetscLogObjectMemory((PetscObject)B, (nzBcsr + 1) * (sizeof(PetscInt) + sizeof(PetscScalar))));
  {
    dim3 dimBlockTeam(1, 128);
    dim3 dimBlockLeague(Nf, 1);
    hipLaunchKernelGGL(mat_lu_factor_band_init_set_i, dim3(dimBlockLeague), dim3(dimBlockTeam), 0, 0, n, bwU, bi_t);
  }
  PetscHIPCheckLaunch; // does a sync

  // setup data
  if (!hipsparseTriFactors->rpermIndices) {
    const PetscInt *r;
    PetscCall(ISGetIndices(isrow, &r));
    hipsparseTriFactors->rpermIndices = new THRUSTINTARRAY(n);
    hipsparseTriFactors->rpermIndices->assign(r, r + n);
    PetscCall(ISRestoreIndices(isrow, &r));
    PetscCall(PetscLogCpuToGpu(n * sizeof(PetscInt)));
  }
  /* upper triangular indices */
  if (!hipsparseTriFactors->cpermIndices) {
    const PetscInt *c;
    PetscCall(ISGetIndices(isicol, &c));
    hipsparseTriFactors->cpermIndices = new THRUSTINTARRAY(n);
    hipsparseTriFactors->cpermIndices->assign(c, c + n);
    PetscCall(ISRestoreIndices(isicol, &c));
    PetscCall(PetscLogCpuToGpu(n * sizeof(PetscInt)));
  }

  /* put together the new matrix */
  b->free_a       = PETSC_FALSE;
  b->free_ij      = PETSC_FALSE;
  b->singlemalloc = PETSC_FALSE;
  b->ilen         = NULL;
  b->imax         = NULL;
  b->row          = isrow;
  b->col          = iscol;
  PetscCall(PetscObjectReference((PetscObject)isrow));
  PetscCall(PetscObjectReference((PetscObject)iscol));
  b->icol = isicol;
  PetscCall(PetscMalloc1(n + 1, &b->solve_work));

  B->factortype            = MAT_FACTOR_LU;
  B->info.factor_mallocs   = 0;
  B->info.fill_ratio_given = 0;

  if (ai[n]) B->info.fill_ratio_needed = ((PetscReal)(nzBcsr)) / ((PetscReal)ai[n]);
  else B->info.fill_ratio_needed = 0.0;
#if defined(PETSC_USE_INFO)
  if (ai[n] != 0) {
    PetscReal af = B->info.fill_ratio_needed;
    PetscCall(PetscInfo(A, "Band fill ratio %g\n", (double)af));
  } else PetscCall(PetscInfo(A, "Empty matrix\n"));
#endif
  if (a->inode.size) PetscCall(PetscInfo(A, "Warning: using inodes in band solver.\n"));
  PetscCall(MatSeqAIJCheckInode_FactorLU(B));
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJHIPSPARSEBAND;
  B->offloadmask          = PETSC_OFFLOAD_GPU;

  PetscFunctionReturn(0);
}

/* Use -pc_factor_mat_solver_type hipsparseband */
PetscErrorCode MatFactorGetSolverType_seqaij_hipsparse_band(Mat A, MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERHIPSPARSEBAND;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatGetFactor_seqaijhipsparse_hipsparse_band(Mat A, MatFactorType ftype, Mat *B)
{
  PetscInt n = A->rmap->n;

  PetscFunctionBegin;
  PetscCall(MatCreate(PetscObjectComm((PetscObject)A), B));
  PetscCall(MatSetSizes(*B, n, n, n, n));
  (*B)->factortype     = ftype;
  (*B)->canuseordering = PETSC_TRUE;
  PetscCall(MatSetType(*B, MATSEQAIJHIPSPARSE));

  if (ftype == MAT_FACTOR_LU) {
    PetscCall(MatSetBlockSizesFromMats(*B, A, A));
    (*B)->ops->ilufactorsymbolic = NULL; // MatILUFactorSymbolic_SeqAIJHIPSPARSE;
    (*B)->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqAIJHIPSPARSEBAND;
    PetscCall(PetscStrallocpy(MATORDERINGRCM, (char **)&(*B)->preferredordering[MAT_FACTOR_LU]));
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Factor type not supported for HIPSPARSEBAND Matrix Types");

  PetscCall(MatSeqAIJSetPreallocation(*B, MAT_SKIP_ALLOCATION, NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)(*B), "MatFactorGetSolverType_C", MatFactorGetSolverType_seqaij_hipsparse_band));
  PetscFunctionReturn(0);
}

#define WARP_SIZE 32 // to be consistent with Nvidia terminology. WARP == Wavefront
template <typename T>
__forceinline__ __device__ T wreduce(T a)
{
  T b;
#pragma unroll
  for (int i = WARP_SIZE / 2; i >= 1; i = i >> 1) {
    b = __shfl_down(0xffffffff, a, i);
    a += b;
  }
  return a;
}
// reduce in a block, returns result in thread 0
template <typename T, int BLOCK_SIZE>
__device__ T breduce(T a)
{
  constexpr int     NWARP = BLOCK_SIZE / WARP_SIZE;
  __shared__ double buf[NWARP];
  int               wid    = threadIdx.x / WARP_SIZE;
  int               laneid = threadIdx.x % WARP_SIZE;
  T                 b      = wreduce<T>(a);
  if (laneid == 0) buf[wid] = b;
  __syncthreads();
  if (wid == 0) {
    if (threadIdx.x < NWARP) a = buf[threadIdx.x];
    else a = 0;
    for (int i = (NWARP + 1) / 2; i >= 1; i = i >> 1) { a += __shfl_down(0xffffffff, a, i); }
  }
  return a;
}

// Band LU kernel ---  ba_csr bi_csr
template <int BLOCK_SIZE>
__global__ void __launch_bounds__(256, 1) mat_solve_band(const PetscInt n, const PetscInt bw, const PetscScalar ba_csr[], PetscScalar x[])
{
  const PetscInt     Nf = gridDim.x, nloc = n / Nf, field = blockIdx.x, start = field * nloc, end = start + nloc, chopnz = bw * (bw + 1) / 2, blocknz = (2 * bw + 1) * nloc, blocknz_0 = blocknz - chopnz;
  const PetscScalar *pLi;
  const int          tid = threadIdx.x;

  /* Next, solve L */
  pLi = ba_csr + (field == 0 ? 0 : blocknz_0 + (field - 1) * blocknz + bw); // diagonal (0,0) in field
  for (int glbDD = start, locDD = 0; glbDD < end; glbDD++, locDD++) {
    const PetscInt col = locDD < bw ? start : (glbDD - bw);
    PetscScalar    t   = 0;
    for (int j = col + tid, idx = tid; j < glbDD; j += blockDim.x, idx += blockDim.x) { t += pLi[idx] * x[j]; }
#if defined(PETSC_USE_COMPLEX)
    PetscReal   tr = PetscRealPartComplex(t), ti = PetscImaginaryPartComplex(t);
    PetscScalar tt(breduce<PetscReal, BLOCK_SIZE>(tr), breduce<PetscReal, BLOCK_SIZE>(ti));
    t = tt;
#else
    t = breduce<PetscReal, BLOCK_SIZE>(t);
#endif
    if (threadIdx.x == 0) x[glbDD] -= t; // /1.0
    __syncthreads();
    // inc
    pLi += glbDD - col;                           // get to diagonal
    if (glbDD > n - 1 - bw) pLi += n - 1 - glbDD; // skip over U, only last block has funny offset
    else pLi += bw;
    pLi += 1;                                                   // skip to next row
    if (field > 0 && (locDD + 1) < bw) pLi += bw - (locDD + 1); // skip padding at beginning (ear)
  }
  /* Then, solve U */
  pLi = ba_csr + Nf * blocknz - 2 * chopnz - 1;                            // end of real data on block (diagonal)
  if (field != Nf - 1) pLi -= blocknz_0 + (Nf - 2 - field) * blocknz + bw; // diagonal of last local row

  for (int glbDD = end - 1, locDD = 0; glbDD >= start; glbDD--, locDD++) {
    const PetscInt col = (locDD < bw) ? end - 1 : glbDD + bw; // end of row in U
    PetscScalar    t   = 0;
    for (int j = col - tid, idx = tid; j > glbDD; j -= blockDim.x, idx += blockDim.x) { t += pLi[-idx] * x[j]; }
#if defined(PETSC_USE_COMPLEX)
    PetscReal   tr = PetscRealPartComplex(t), ti = PetscImaginaryPartComplex(t);
    PetscScalar tt(breduce<PetscReal, BLOCK_SIZE>(tr), breduce<PetscReal, BLOCK_SIZE>(ti));
    t = tt;
#else
    t = breduce<PetscReal, BLOCK_SIZE>(PetscRealPart(t));
#endif
    pLi -= col - glbDD; // diagonal
    if (threadIdx.x == 0) {
      x[glbDD] -= t;
      x[glbDD] /= pLi[0];
    }
    __syncthreads();
    // inc past L to start of previous U
    pLi -= bw + 1;
    if (glbDD < bw) pLi += bw - glbDD;                                    // overshot in top left corner
    if (((locDD + 1) < bw) && field != Nf - 1) pLi -= (bw - (locDD + 1)); // skip past right corner
  }
}

static PetscErrorCode MatSolve_SeqAIJHIPSPARSEBAND(Mat A, Vec bb, Vec xx)
{
  const PetscScalar                    *barray;
  PetscScalar                          *xarray;
  thrust::device_ptr<const PetscScalar> bGPU;
  thrust::device_ptr<PetscScalar>       xGPU;
  Mat_SeqAIJHIPSPARSETriFactors        *hipsparseTriFactors = (Mat_SeqAIJHIPSPARSETriFactors *)A->spptr;
  THRUSTARRAY                          *tempGPU             = (THRUSTARRAY *)hipsparseTriFactors->workVector;
  PetscInt                              n = A->rmap->n, nz = hipsparseTriFactors->nnz, Nf = 1;                                                                                 // Nf is batch size - not used
  PetscInt                              bw = (int)(2. * (double)n - 1. - (double)(PetscSqrtReal(1. + 4. * ((double)n * (double)n - (double)nz)) + PETSC_MACHINE_EPSILON)) / 2; // quadric formula for bandwidth

  PetscFunctionBegin;
  if (A->rmap->n == 0) PetscFunctionReturn(0);

  /* Get the GPU pointers */
  PetscCall(VecHIPGetArrayWrite(xx, &xarray));
  PetscCall(VecHIPGetArrayRead(bb, &barray));
  xGPU = thrust::device_pointer_cast(xarray);
  bGPU = thrust::device_pointer_cast(barray);

  PetscCall(PetscLogGpuTimeBegin());
  /* First, reorder with the row permutation */
  thrust::copy(thrust::hip::par.on(PetscDefaultHipStream), thrust::make_permutation_iterator(bGPU, hipsparseTriFactors->rpermIndices->begin()), thrust::make_permutation_iterator(bGPU, hipsparseTriFactors->rpermIndices->end()), tempGPU->begin());
  constexpr int block = 128;
  hipLaunchKernelGGL(HIP_KERNEL_NAME(mat_solve_band<block>), dim3(Nf), dim3(block), 0, 0, n, bw, hipsparseTriFactors->a_band_d, tempGPU->data().get());
  PetscHIPCheckLaunch; // does a sync

  /* Last, reorder with the column permutation */
  thrust::copy(thrust::hip::par.on(PetscDefaultHipStream), thrust::make_permutation_iterator(tempGPU->begin(), hipsparseTriFactors->cpermIndices->begin()), thrust::make_permutation_iterator(tempGPU->begin(), hipsparseTriFactors->cpermIndices->end()), xGPU);

  PetscCall(VecHIPRestoreArrayRead(bb, &barray));
  PetscCall(VecHIPRestoreArrayWrite(xx, &xarray));
  PetscCall(PetscLogGpuFlops(2.0 * hipsparseTriFactors->nnz - A->cmap->n));
  PetscCall(PetscLogGpuTimeEnd());

  PetscFunctionReturn(0);
}
