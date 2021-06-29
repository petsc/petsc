/*
  AIJCUSPARSE methods implemented with Cuda kernels. Uses cuSparse/Thrust maps from AIJCUSPARSE
*/
#define PETSC_SKIP_SPINLOCK
#define PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND 1

#include <petscconf.h>
#include <../src/mat/impls/aij/seq/aij.h>          /*I "petscmat.h" I*/
#include <../src/mat/impls/sbaij/seq/sbaij.h>
#undef VecType
#include <../src/mat/impls/aij/seq/seqcusparse/cusparsematimpl.h>
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
#include <cooperative_groups.h>
#endif

#define CHECK_LAUNCH_ERROR()                                                             \
do {                                                                                     \
  /* Check synchronous errors, i.e. pre-launch */                                        \
  cudaError_t err = cudaGetLastError();                                                  \
  if (cudaSuccess != err) {                                                              \
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cuda error: %s",cudaGetErrorString(err)); \
  }                                                                                      \
  /* Check asynchronous errors, i.e. kernel failed (ULF) */                              \
  err = cudaDeviceSynchronize();                                                         \
  if (cudaSuccess != err) {                                                              \
    SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cuda error: %s",cudaGetErrorString(err)); \
  }                                                                                      \
 } while (0)

/*
  LU BAND factorization with optimization for block diagonal (Nf blocks) in natural order (-mat_no_inode -pc_factor_mat_ordering_type rcm with Nf>1 fields)

  requires:
     structurally symmetric: fix with transpose/column meta data
*/

static PetscErrorCode MatLUFactorSymbolic_SeqAIJCUSPARSEBAND(Mat,Mat,IS,IS,const MatFactorInfo*);
static PetscErrorCode MatLUFactorNumeric_SeqAIJCUSPARSEBAND(Mat,Mat,const MatFactorInfo*);

/*
  The GPU LU factor kernel
*/
__global__
void __launch_bounds__(1024,1)
mat_lu_factor_band_init_set_i(const PetscInt n, const int bw, int bi_csr[])
{
  const PetscInt  Nf = gridDim.x, Nblk = gridDim.y, nloc = n/Nf;
  const PetscInt  field = blockIdx.x, blkIdx = blockIdx.y;
  const PetscInt  nloc_i =  (nloc/Nblk + !!(nloc%Nblk)), start_i = field*nloc + blkIdx*nloc_i, end_i = (start_i + nloc_i) > (field+1)*nloc ? (field+1)*nloc : (start_i + nloc_i);

  // set i (row+1)
  if (threadIdx.x + threadIdx.y + blockIdx.x + blockIdx.y == 0) bi_csr[0] = 0; // dummy at zero
  for (int rowb = start_i + threadIdx.y; rowb < end_i; rowb += blockDim.y) { // rows in block by thread y
    if (rowb < end_i && threadIdx.x==0) {
      PetscInt i=rowb+1, ni = (rowb>bw) ? bw+1 : i, n1L = ni*(ni-1)/2, nug= i*bw, n2L = bw*((rowb>bw) ? (rowb-bw) : 0), mi = bw + rowb + 1 - n, clip = (mi>0) ? mi*(mi-1)/2 + mi: 0;
      bi_csr[rowb+1] = n1L + nug - clip + n2L + i;
    }
  }
}
// copy AIJ to AIJ_BAND
__global__
void __launch_bounds__(1024,1)
mat_lu_factor_band_copy_aij_aij(const PetscInt n, const int bw, const PetscInt r[], const PetscInt ic[],
                                const int ai_d[], const int aj_d[], const PetscScalar aa_d[],
                                const int bi_csr[], PetscScalar ba_csr[])
{
  const PetscInt  Nf = gridDim.x, Nblk = gridDim.y, nloc = n/Nf;
  const PetscInt  field = blockIdx.x, blkIdx = blockIdx.y;
  const PetscInt  nloc_i =  (nloc/Nblk + !!(nloc%Nblk)), start_i = field*nloc + blkIdx*nloc_i, end_i = (start_i + nloc_i) > (field+1)*nloc ? (field+1)*nloc : (start_i + nloc_i);

  // zero B
  if (threadIdx.x + threadIdx.y + blockIdx.x + blockIdx.y == 0) ba_csr[bi_csr[n]] = 0; // flop count at end
  for (int rowb = start_i + threadIdx.y; rowb < end_i; rowb += blockDim.y) { // rows in block by thread y
    if (rowb < end_i) {
      PetscScalar    *batmp = ba_csr + bi_csr[rowb];
      const PetscInt nzb = bi_csr[rowb+1] - bi_csr[rowb];
      for (int j=threadIdx.x ; j<nzb ; j += blockDim.x) {
        if (j<nzb) {
          batmp[j] = 0;
        }
      }
    }
  }

  // copy A into B with CSR format -- these two loops can be fused
  for (int rowb = start_i + threadIdx.y; rowb < end_i; rowb += blockDim.y) { // rows in block by thread y
    if (rowb < end_i) {
      const PetscInt    rowa = r[rowb], nza = ai_d[rowa+1] - ai_d[rowa];
      const int         *ajtmp = aj_d + ai_d[rowa], bjStart = (rowb>bw) ? rowb-bw : 0;
      const PetscScalar *av    = aa_d + ai_d[rowa];
      PetscScalar       *batmp = ba_csr + bi_csr[rowb];
      /* load in initial (unfactored row) */
      for (int j=threadIdx.x ; j<nza ; j += blockDim.x) {
        if (j<nza) {
          PetscInt    colb = ic[ajtmp[j]], idx = colb - bjStart;
          PetscScalar vala = av[j];
          batmp[idx] = vala;
        }
      }
    }
  }
}
// print AIJ_BAND
__global__
void print_mat_aij_band(const PetscInt n, const int bi_csr[], const PetscScalar ba_csr[])
{
  // debug
  if (threadIdx.x + threadIdx.y + blockIdx.x + blockIdx.y == 0) {
    printf("B (AIJ) n=%d:\n",(int)n);
    for (int rowb=0;rowb<n;rowb++) {
      const PetscInt    nz = bi_csr[rowb+1] - bi_csr[rowb];
      const PetscScalar *batmp = ba_csr + bi_csr[rowb];
      for (int j=0; j<nz; j++) printf("(%13.6e) ",PetscRealPart(batmp[j]));
      printf(" bi=%d\n",bi_csr[rowb+1]);
    }
  }
}
// Band LU kernel ---  ba_csr bi_csr
__global__
void __launch_bounds__(1024,1)
  mat_lu_factor_band(const PetscInt n, const PetscInt bw, const int bi_csr[], PetscScalar ba_csr[], int *use_group_sync)
{
  const PetscInt  Nf = gridDim.x, Nblk = gridDim.y, nloc = n/Nf;
  const PetscInt  field = blockIdx.x, blkIdx = blockIdx.y;
  const PetscInt  start = field*nloc, end = start + nloc;
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
  auto g = cooperative_groups::this_grid();
#endif
  // A22 panel update for each row A(1,:) and col A(:,1)
  for (int glbDD=start, locDD = 0; glbDD<end; glbDD++, locDD++) {
    PetscInt          tnzUd = bw, maxU = end-1 - glbDD; // we are chopping off the inter ears
    const PetscInt    nzUd  = (tnzUd>maxU) ? maxU : tnzUd, dOffset = (glbDD > bw) ? bw : glbDD; // global to go past ears after first
    PetscScalar       *pBdd = ba_csr + bi_csr[glbDD] + dOffset;
    const PetscScalar *baUd = pBdd + 1; // vector of data  U(i,i+1:end)
    const PetscScalar Bdd = *pBdd;
    const PetscInt    offset = blkIdx*blockDim.y + threadIdx.y, inc = Nblk*blockDim.y;
    if (threadIdx.x==0) {
      for (int idx = offset, myi = glbDD + offset + 1; idx < nzUd; idx += inc, myi += inc) { /* assuming symmetric structure */
        const PetscInt bwi = myi > bw ? bw : myi, kIdx = bwi - (myi-glbDD); // cuts off just the first (global) block
        PetscScalar    *Aid = ba_csr + bi_csr[myi] + kIdx;
        *Aid = *Aid/Bdd;
      }
    }
    __syncthreads(); // synch on threadIdx.x only
    for (int idx = offset, myi = glbDD + offset + 1; idx < nzUd; idx += inc, myi += inc) {
      const PetscInt    bwi = myi > bw ? bw : myi, kIdx = bwi - (myi-glbDD); // cuts off just the first (global) block
      PetscScalar       *Aid = ba_csr + bi_csr[myi] + kIdx;
      PetscScalar       *Aij =  Aid + 1;
      const PetscScalar Lid  = *Aid;
      for (int jIdx=threadIdx.x ; jIdx<nzUd; jIdx += blockDim.x) {
        Aij[jIdx] -= Lid*baUd[jIdx];
      }
    }
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
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

static PetscErrorCode MatSolve_SeqAIJCUSPARSEBAND(Mat,Vec,Vec);
static PetscErrorCode MatLUFactorNumeric_SeqAIJCUSPARSEBAND(Mat B,Mat A,const MatFactorInfo *info)
{
  Mat_SeqAIJ                   *b = (Mat_SeqAIJ*)B->data;
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)B->spptr;
  if (!cusparseTriFactors) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing cusparseTriFactors");
  Mat_SeqAIJCUSPARSE           *cusparsestructA = (Mat_SeqAIJCUSPARSE*)A->spptr;
  Mat_SeqAIJCUSPARSEMultStruct *matstructA;
  CsrMatrix                    *matrixA;
  PetscErrorCode               ierr;
  cudaError_t                  cerr;
  const PetscInt               n=A->rmap->n, *ic, *r;
  const int                    *ai_d, *aj_d;
  const PetscScalar            *aa_d;
  PetscScalar                  *ba_t = cusparseTriFactors->a_band_d;
  int                          *bi_t = cusparseTriFactors->i_band_d;
  PetscContainer               container;
  int                          Ni = 10, team_size=9, Nf, nVec=56, nconcurrent = 1, nsm = -1;

  PetscFunctionBegin;
  if (A->rmap->n == 0) {
    PetscFunctionReturn(0);
  }
  // cusparse setup
  if (!cusparsestructA) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing cusparsestructA");
  matstructA = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestructA->mat; //  matstruct->cprowIndices
  if (!matstructA) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing mat struct");
  matrixA = (CsrMatrix*)matstructA->mat;
  if (!matrixA) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing matrix cusparsestructA->mat->mat");

  // factor: get Nf if available
  ierr = PetscObjectQuery((PetscObject) A, "Nf", (PetscObject *) &container);CHKERRQ(ierr);
  if (container) {
    PetscInt *pNf=NULL;
    ierr = PetscContainerGetPointer(container, (void **) &pNf);CHKERRQ(ierr);
    Nf = (*pNf)%1000;
    if ((*pNf)/1000>0) nconcurrent = (*pNf)/1000; // number of SMs to use
  } else Nf = 1;
  if (n%Nf) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"n % Nf != 0 %D %D",n,Nf);

  // get data
  ic      = thrust::raw_pointer_cast(cusparseTriFactors->cpermIndices->data());
  ai_d    = thrust::raw_pointer_cast(matrixA->row_offsets->data());
  aj_d    = thrust::raw_pointer_cast(matrixA->column_indices->data());
  aa_d    = thrust::raw_pointer_cast(matrixA->values->data().get());
  r       = thrust::raw_pointer_cast(cusparseTriFactors->rpermIndices->data());

  cerr = WaitForCUDA();CHKERRCUDA(cerr);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  {
    int bw = (int)(2.*(double)n-1. - (double)(PetscSqrtReal(1.+4.*((double)n*(double)n-(double)b->nz))+PETSC_MACHINE_EPSILON))/2, bm1=bw-1,nl=n/Nf;
#if PETSC_PKG_CUDA_VERSION_LT(11,0,0)
    Ni = 1/nconcurrent;
    Ni = 1;
#else
    if (!cusparseTriFactors->init_dev_prop) {
      int gpuid;
      cusparseTriFactors->init_dev_prop = PETSC_TRUE;
      cudaGetDevice(&gpuid);
      cudaGetDeviceProperties(&cusparseTriFactors->dev_prop, gpuid);
    }
    nsm = cusparseTriFactors->dev_prop.multiProcessorCount;
    Ni = nsm/Nf/nconcurrent;
#endif
    team_size = bw/Ni + !!(bw%Ni);
    nVec = PetscMin(bw, 1024/team_size);
    ierr = PetscInfo7(A,"Matrix Bandwidth = %d, number SMs/block = %d, num concurency = %d, num fields = %d, numSMs/GPU = %d, thread group size = %d,%d\n",bw,Ni,nconcurrent,Nf,nsm,team_size,nVec);CHKERRQ(ierr);
    {
      dim3 dimBlockTeam(nVec,team_size);
      dim3 dimBlockLeague(Nf,Ni);
      mat_lu_factor_band_copy_aij_aij<<<dimBlockLeague,dimBlockTeam>>>(n, bw, r, ic, ai_d, aj_d, aa_d, bi_t, ba_t);
      CHECK_LAUNCH_ERROR(); // does a sync
#if PETSC_PKG_CUDA_VERSION_GE(11,0,0)
      if (Ni > 1) {
        void *kernelArgs[] = { (void*)&n, (void*)&bw, (void*)&bi_t, (void*)&ba_t, (void*)&nsm };
        cudaLaunchCooperativeKernel((void*)mat_lu_factor_band, dimBlockLeague, dimBlockTeam, kernelArgs, 0, NULL);
      } else {
        mat_lu_factor_band<<<dimBlockLeague,dimBlockTeam>>>(n, bw, bi_t, ba_t, NULL);
      }
#else
      mat_lu_factor_band<<<dimBlockLeague,dimBlockTeam>>>(n, bw, bi_t, ba_t, NULL);
#endif
      CHECK_LAUNCH_ERROR(); // does a sync
#if defined(PETSC_USE_LOG)
      ierr = PetscLogGpuFlops((PetscLogDouble)Nf*(bm1*(bm1 + 1)*(PetscLogDouble)(2*bm1 + 1)/3 + (PetscLogDouble)2*(nl-bw)*bw*bw + (PetscLogDouble)nl*(nl+1)/2));CHKERRQ(ierr);
#endif
    }
  }
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  /* determine which version of MatSolve needs to be used. from MatLUFactorNumeric_AIJ_SeqAIJCUSPARSE */
  B->ops->solve = MatSolve_SeqAIJCUSPARSEBAND;
  B->ops->solvetranspose = NULL; // need transpose
  B->ops->matsolve = NULL;
  B->ops->matsolvetranspose = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatrixNfDestroy(void *ptr)
{
  PetscInt *nf = (PetscInt *)ptr;
  PetscErrorCode  ierr;
  PetscFunctionBegin;
  ierr = PetscFree(nf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatLUFactorSymbolic_SeqAIJCUSPARSEBAND(Mat B,Mat A,IS isrow,IS iscol,const MatFactorInfo *info)
{
  Mat_SeqAIJ         *a = (Mat_SeqAIJ*)A->data,*b;
  IS                 isicol;
  PetscErrorCode     ierr;
  cudaError_t        cerr;
  const PetscInt     *ic,*ai=a->i,*aj=a->j;
  PetscScalar        *ba_t;
  int                *bi_t;
  PetscInt           i,n=A->rmap->n,Nf;
  PetscInt           nzBcsr,bwL,bwU;
  PetscBool          missing;
  Mat_SeqAIJCUSPARSETriFactors *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)B->spptr;
  PetscContainer               container;

  PetscFunctionBegin;
  if (A->rmap->N != A->cmap->N) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"matrix must be square");
  ierr = MatMissingDiagonal(A,&missing,&i);CHKERRQ(ierr);
  if (missing) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Matrix is missing diagonal entry %D",i);
  if (!cusparseTriFactors) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"!cusparseTriFactors");
  ierr = MatGetOption(A,MAT_STRUCTURALLY_SYMMETRIC,&missing);CHKERRQ(ierr);
  if (!missing) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"only structrally symmetric matrices supported");

   // factor: get Nf if available
  ierr = PetscObjectQuery((PetscObject) A, "Nf", (PetscObject *) &container);CHKERRQ(ierr);
  if (container) {
    PetscInt *pNf=NULL;
    ierr = PetscContainerGetPointer(container, (void **) &pNf);CHKERRQ(ierr);
    Nf = (*pNf)%1000;
    ierr = PetscContainerCreate(PETSC_COMM_SELF, &container);CHKERRQ(ierr);
    ierr = PetscMalloc(sizeof(PetscInt), &pNf);CHKERRQ(ierr);
    *pNf = Nf;
    ierr = PetscContainerSetPointer(container, (void *)pNf);CHKERRQ(ierr);
    ierr = PetscContainerSetUserDestroy(container, MatrixNfDestroy);CHKERRQ(ierr);
    ierr = PetscObjectCompose((PetscObject)B, "Nf", (PetscObject) container);CHKERRQ(ierr);
    ierr = PetscContainerDestroy(&container);CHKERRQ(ierr);
  } else Nf = 1;
  if (n%Nf) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"n % Nf != 0 %D %D",n,Nf);

  ierr = ISInvertPermutation(iscol,PETSC_DECIDE,&isicol);CHKERRQ(ierr);
  ierr = ISGetIndices(isicol,&ic);CHKERRQ(ierr);

  ierr = MatSeqAIJSetPreallocation_SeqAIJ(B,MAT_SKIP_ALLOCATION,NULL);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)B,(PetscObject)isicol);CHKERRQ(ierr);
  b    = (Mat_SeqAIJ*)(B)->data;

  /* get band widths, MatComputeBandwidth should take a reordering ic and do this */
  bwL = bwU = 0;
  for (int rwb=0; rwb<n; rwb++) {
    const PetscInt rwa = ic[rwb], anz = ai[rwb+1] - ai[rwb], *ajtmp = aj + ai[rwb];
    for (int j=0;j<anz;j++) {
      PetscInt colb = ic[ajtmp[j]];
      if (colb<rwa) { // L
        if (rwa-colb > bwL) bwL = rwa-colb;
      } else {
        if (colb-rwa > bwU) bwU = colb-rwa;
      }
    }
  }
  ierr = ISRestoreIndices(isicol,&ic);CHKERRQ(ierr);
  /* only support structurally symmetric, but it might work */
  if (bwL!=bwU) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Only symmetric structure supported (now) W_L=%D W_U=%D",bwL,bwU);
  ierr = MatSeqAIJCUSPARSETriFactors_Reset(&cusparseTriFactors);CHKERRQ(ierr);
  nzBcsr = n + (2*n-1)*bwU - bwU*bwU;
  b->maxnz = b->nz = nzBcsr;
  cusparseTriFactors->nnz = b->nz; // only meta data needed: n & nz
  ierr = PetscInfo2(A,"Matrix Bandwidth = %D, nnz = %D\n",bwL,b->nz);CHKERRQ(ierr);
  if (!cusparseTriFactors->workVector) { cusparseTriFactors->workVector = new THRUSTARRAY(n); }
  cerr = cudaMalloc(&ba_t,(b->nz+1)*sizeof(PetscScalar));CHKERRCUDA(cerr); // incude a place for flops
  cerr = cudaMalloc(&bi_t,(n+1)*sizeof(int));CHKERRCUDA(cerr);
  cusparseTriFactors->a_band_d = ba_t;
  cusparseTriFactors->i_band_d = bi_t;
  /* In b structure:  Free imax, ilen, old a, old j.  Allocate solve_work, new a, new j */
  ierr = PetscLogObjectMemory((PetscObject)B,(nzBcsr+1)*(sizeof(PetscInt)+sizeof(PetscScalar)));CHKERRQ(ierr);
  {
    dim3 dimBlockTeam(1,128);
    dim3 dimBlockLeague(Nf,1);
    mat_lu_factor_band_init_set_i<<<dimBlockLeague,dimBlockTeam>>>(n, bwU, bi_t);
  }
  CHECK_LAUNCH_ERROR(); // does a sync

  // setup data
  if (!cusparseTriFactors->rpermIndices) {
    const PetscInt *r;

    ierr = ISGetIndices(isrow,&r);CHKERRQ(ierr);
    cusparseTriFactors->rpermIndices = new THRUSTINTARRAY(n);
    cusparseTriFactors->rpermIndices->assign(r, r+n);
    ierr = ISRestoreIndices(isrow,&r);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpu(n*sizeof(PetscInt));CHKERRQ(ierr);
  }
  /* upper triangular indices */
  if (!cusparseTriFactors->cpermIndices) {
    const PetscInt *c;

    ierr = ISGetIndices(isicol,&c);CHKERRQ(ierr);
    cusparseTriFactors->cpermIndices = new THRUSTINTARRAY(n);
    cusparseTriFactors->cpermIndices->assign(c, c+n);
    ierr = ISRestoreIndices(isicol,&c);CHKERRQ(ierr);
    ierr = PetscLogCpuToGpu(n*sizeof(PetscInt));CHKERRQ(ierr);
  }

  /* put together the new matrix */
  b->free_a       = PETSC_FALSE;
  b->free_ij      = PETSC_FALSE;
  b->singlemalloc = PETSC_FALSE;
  b->ilen = NULL;
  b->imax = NULL;
  b->row  = isrow;
  b->col  = iscol;
  ierr    = PetscObjectReference((PetscObject)isrow);CHKERRQ(ierr);
  ierr    = PetscObjectReference((PetscObject)iscol);CHKERRQ(ierr);
  b->icol = isicol;
  ierr    = PetscMalloc1(n+1,&b->solve_work);CHKERRQ(ierr);

  B->factortype            = MAT_FACTOR_LU;
  B->info.factor_mallocs   = 0;
  B->info.fill_ratio_given = 0;

  if (ai[n]) {
    B->info.fill_ratio_needed = ((PetscReal)(nzBcsr))/((PetscReal)ai[n]);
  } else {
    B->info.fill_ratio_needed = 0.0;
  }
#if defined(PETSC_USE_INFO)
  if (ai[n] != 0) {
    PetscReal af = B->info.fill_ratio_needed;
    ierr = PetscInfo1(A,"Band fill ratio %g\n",(double)af);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(A,"Empty matrix\n");CHKERRQ(ierr);
  }
#endif
  if (a->inode.size) {
    ierr = PetscInfo(A,"Warning: using inodes in band solver.\n");CHKERRQ(ierr);
  }
  ierr = MatSeqAIJCheckInode_FactorLU(B);CHKERRQ(ierr);
  B->ops->lufactornumeric = MatLUFactorNumeric_SeqAIJCUSPARSEBAND;
  B->offloadmask = PETSC_OFFLOAD_GPU;

  PetscFunctionReturn(0);
}

/* Use -pc_factor_mat_solver_type cusparseband */
PetscErrorCode MatFactorGetSolverType_seqaij_cusparse_band(Mat A,MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERCUSPARSEBAND;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatGetFactor_seqaijcusparse_cusparse_band(Mat A,MatFactorType ftype,Mat *B)
{
  PetscErrorCode ierr;
  PetscInt       n = A->rmap->n;

  PetscFunctionBegin;
  ierr = MatCreate(PetscObjectComm((PetscObject)A),B);CHKERRQ(ierr);
  ierr = MatSetSizes(*B,n,n,n,n);CHKERRQ(ierr);
  (*B)->factortype = ftype;
  (*B)->canuseordering = PETSC_TRUE;
  ierr = MatSetType(*B,MATSEQAIJCUSPARSE);CHKERRQ(ierr);

  if (ftype == MAT_FACTOR_LU) {
    ierr = MatSetBlockSizesFromMats(*B,A,A);CHKERRQ(ierr);
    (*B)->ops->ilufactorsymbolic = NULL; // MatILUFactorSymbolic_SeqAIJCUSPARSE;
    (*B)->ops->lufactorsymbolic  = MatLUFactorSymbolic_SeqAIJCUSPARSEBAND;
    ierr = PetscStrallocpy(MATORDERINGRCM,(char**)&(*B)->preferredordering[MAT_FACTOR_LU]);CHKERRQ(ierr);
  } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Factor type not supported for CUSPARSEBAND Matrix Types");

  ierr = MatSeqAIJSetPreallocation(*B,MAT_SKIP_ALLOCATION,NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)(*B),"MatFactorGetSolverType_C",MatFactorGetSolverType_seqaij_cusparse_band);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#define WARP_SIZE 32
template <typename T>
__forceinline__ __device__
T wreduce(T a)
{
  T b;
  #pragma unroll
  for (int i = WARP_SIZE/2; i >= 1; i = i >> 1) {
    b = __shfl_down_sync(0xffffffff, a, i);
    a += b;
  }
  return a;
}
// reduce in a block, returns result in thread 0
template <typename T, int BLOCK_SIZE>
__device__
T breduce(T a)
{
  constexpr int NWARP = BLOCK_SIZE/WARP_SIZE;
  __shared__ double buf[NWARP];
  int wid = threadIdx.x / WARP_SIZE;
  int laneid = threadIdx.x % WARP_SIZE;
  T b = wreduce<T>(a);
  if (laneid == 0)
    buf[wid] = b;
  __syncthreads();
  if (wid == 0) {
    if (threadIdx.x < NWARP)
      a = buf[threadIdx.x];
    else
      a = 0;
    for (int i = (NWARP+1)/2; i >= 1; i = i >> 1) {
      a += __shfl_down_sync(0xffffffff, a, i);
    }
  }
  return a;
}

// Band LU kernel ---  ba_csr bi_csr
template <int BLOCK_SIZE>
__global__
void __launch_bounds__(256,1)
mat_solve_band(const PetscInt n, const PetscInt bw, const PetscScalar ba_csr[], PetscScalar x[])
{
  const PetscInt    Nf = gridDim.x, nloc = n/Nf, field = blockIdx.x, start = field*nloc, end = start + nloc, chopnz = bw*(bw+1)/2, blocknz=(2*bw+1)*nloc, blocknz_0 = blocknz-chopnz;
  const PetscScalar *pLi;
  const int tid = threadIdx.x;

  /* Next, solve L */
  pLi = ba_csr + (field==0 ? 0 : blocknz_0 + (field-1)*blocknz + bw); // diagonal (0,0) in field
  for (int glbDD=start, locDD = 0; glbDD<end; glbDD++, locDD++) {
    const PetscInt col = locDD<bw ? start : (glbDD-bw);
    PetscScalar t = 0;
    for (int j=col+tid,idx=tid;j<glbDD;j+=blockDim.x,idx+=blockDim.x) {
      t += pLi[idx]*x[j];
    }
#if defined(PETSC_USE_COMPLEX)
    PetscReal tr = PetscRealPartComplex(t), ti = PetscImaginaryPartComplex(t);
    PetscScalar tt(breduce<PetscReal,BLOCK_SIZE>(tr), breduce<PetscReal,BLOCK_SIZE>(ti));
    t = tt;
#else
    t = breduce<PetscReal,BLOCK_SIZE>(t);
#endif
    if (threadIdx.x == 0)
      x[glbDD] -= t; // /1.0
    __syncthreads();
    // inc
    pLi += glbDD-col; // get to diagonal
    if (glbDD > n-1-bw) pLi += n-1-glbDD; // skip over U, only last block has funny offset
    else pLi += bw;
    pLi += 1; // skip to next row
    if (field>0 && (locDD+1)<bw) pLi += bw-(locDD+1); // skip padding at beginning (ear)
  }
  /* Then, solve U */
  pLi = ba_csr + Nf*blocknz - 2*chopnz - 1; // end of real data on block (diagonal)
  if (field != Nf-1) pLi -= blocknz_0 + (Nf-2-field)*blocknz + bw; // diagonal of last local row

  for (int glbDD=end-1, locDD = 0; glbDD >= start; glbDD--, locDD++) {
    const PetscInt col = (locDD<bw) ? end-1 : glbDD+bw; // end of row in U
    PetscScalar t = 0;
    for (int j=col-tid,idx=tid;j>glbDD;j-=blockDim.x,idx+=blockDim.x) {
      t += pLi[-idx]*x[j];
    }
#if defined(PETSC_USE_COMPLEX)
    PetscReal tr = PetscRealPartComplex(t), ti = PetscImaginaryPartComplex(t);
    PetscScalar tt(breduce<PetscReal,BLOCK_SIZE>(tr), breduce<PetscReal,BLOCK_SIZE>(ti));
    t = tt;
#else
    t = breduce<PetscReal,BLOCK_SIZE>(PetscRealPart(t));
#endif
    pLi -= col-glbDD; // diagonal
    if (threadIdx.x == 0) {
      x[glbDD] -= t;
      x[glbDD] /= pLi[0];
    }
    __syncthreads();
    // inc past L to start of previous U
    pLi -= bw+1;
    if (glbDD<bw) pLi += bw-glbDD; // overshot in top left corner
    if (((locDD+1) < bw) && field != Nf-1) pLi -= (bw - (locDD+1)); // skip past right corner
  }
}

static PetscErrorCode MatSolve_SeqAIJCUSPARSEBAND(Mat A,Vec bb,Vec xx)
{
  const PetscScalar                     *barray;
  PetscScalar                           *xarray;
  thrust::device_ptr<const PetscScalar> bGPU;
  thrust::device_ptr<PetscScalar>       xGPU;
  Mat_SeqAIJCUSPARSETriFactors          *cusparseTriFactors = (Mat_SeqAIJCUSPARSETriFactors*)A->spptr;
  THRUSTARRAY                           *tempGPU = (THRUSTARRAY*)cusparseTriFactors->workVector;
  PetscInt                              n=A->rmap->n, nz=cusparseTriFactors->nnz, Nf;
  PetscInt                              bw = (int)(2.*(double)n-1.-(double)(PetscSqrtReal(1.+4.*((double)n*(double)n-(double)nz))+PETSC_MACHINE_EPSILON))/2; // quadric formula for bandwidth
  PetscErrorCode                        ierr;
  PetscContainer                        container;

  PetscFunctionBegin;
  if (A->rmap->n == 0) {
    PetscFunctionReturn(0);
  }
  // factor: get Nf if available
  ierr = PetscObjectQuery((PetscObject) A, "Nf", (PetscObject *) &container);CHKERRQ(ierr);
  if (container) {
    PetscInt *pNf=NULL;
    ierr = PetscContainerGetPointer(container, (void **) &pNf);CHKERRQ(ierr);
    Nf = (*pNf)%1000;
  } else Nf = 1;
  if (n%Nf) SETERRQ2(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"n(%D) % Nf(%D) != 0",n,Nf);

  /* Get the GPU pointers */
  ierr = VecCUDAGetArrayWrite(xx,&xarray);CHKERRQ(ierr);
  ierr = VecCUDAGetArrayRead(bb,&barray);CHKERRQ(ierr);
  xGPU = thrust::device_pointer_cast(xarray);
  bGPU = thrust::device_pointer_cast(barray);

  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  /* First, reorder with the row permutation */
  thrust::copy(thrust::cuda::par.on(PetscDefaultCudaStream),thrust::make_permutation_iterator(bGPU, cusparseTriFactors->rpermIndices->begin()),
               thrust::make_permutation_iterator(bGPU, cusparseTriFactors->rpermIndices->end()),
               tempGPU->begin());
  constexpr int block = 128;
  mat_solve_band<block><<<Nf,block>>>(n,bw,cusparseTriFactors->a_band_d,tempGPU->data().get());
  CHECK_LAUNCH_ERROR(); // does a sync

  /* Last, reorder with the column permutation */
  thrust::copy(thrust::cuda::par.on(PetscDefaultCudaStream),thrust::make_permutation_iterator(tempGPU->begin(), cusparseTriFactors->cpermIndices->begin()),
               thrust::make_permutation_iterator(tempGPU->begin(), cusparseTriFactors->cpermIndices->end()),
               xGPU);

  ierr = VecCUDARestoreArrayRead(bb,&barray);CHKERRQ(ierr);
  ierr = VecCUDARestoreArrayWrite(xx,&xarray);CHKERRQ(ierr);
  ierr = PetscLogGpuFlops(2.0*cusparseTriFactors->nnz - A->cmap->n);CHKERRQ(ierr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
