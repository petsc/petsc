#define PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND 1

#include <petscconf.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/seqcusparse/cusparsematimpl.h>
#include <../src/mat/impls/aij/mpi/mpicusparse/mpicusparsematimpl.h>
#include <thrust/advance.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <petscsf.h>

struct VecCUDAEquals {
  template <typename Tuple>
  __host__ __device__ void operator()(Tuple t)
  {
    thrust::get<1>(t) = thrust::get<0>(t);
  }
};

static PetscErrorCode MatCOOStructDestroy_MPIAIJCUSPARSE(void **data)
{
  MatCOOStruct_MPIAIJ *coo = (MatCOOStruct_MPIAIJ *)*data;

  PetscFunctionBegin;
  PetscCall(PetscSFDestroy(&coo->sf));
  PetscCallCUDA(cudaFree(coo->Ajmap1));
  PetscCallCUDA(cudaFree(coo->Aperm1));
  PetscCallCUDA(cudaFree(coo->Bjmap1));
  PetscCallCUDA(cudaFree(coo->Bperm1));
  PetscCallCUDA(cudaFree(coo->Aimap2));
  PetscCallCUDA(cudaFree(coo->Ajmap2));
  PetscCallCUDA(cudaFree(coo->Aperm2));
  PetscCallCUDA(cudaFree(coo->Bimap2));
  PetscCallCUDA(cudaFree(coo->Bjmap2));
  PetscCallCUDA(cudaFree(coo->Bperm2));
  PetscCallCUDA(cudaFree(coo->Cperm1));
  PetscCallCUDA(cudaFree(coo->sendbuf));
  PetscCallCUDA(cudaFree(coo->recvbuf));
  PetscCall(PetscFree(coo));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetPreallocationCOO_MPIAIJCUSPARSE(Mat mat, PetscCount coo_n, PetscInt coo_i[], PetscInt coo_j[])
{
  Mat_MPIAIJ          *mpiaij = (Mat_MPIAIJ *)mat->data;
  PetscBool            dev_ij = PETSC_FALSE;
  PetscMemType         mtype  = PETSC_MEMTYPE_HOST;
  PetscInt            *i, *j;
  PetscContainer       container_h;
  MatCOOStruct_MPIAIJ *coo_h, *coo_d;

  PetscFunctionBegin;
  PetscCall(PetscFree(mpiaij->garray));
  PetscCall(VecDestroy(&mpiaij->lvec));
#if defined(PETSC_USE_CTABLE)
  PetscCall(PetscHMapIDestroy(&mpiaij->colmap));
#else
  PetscCall(PetscFree(mpiaij->colmap));
#endif
  PetscCall(VecScatterDestroy(&mpiaij->Mvctx));
  mat->assembled     = PETSC_FALSE;
  mat->was_assembled = PETSC_FALSE;
  PetscCall(PetscGetMemType(coo_i, &mtype));
  if (PetscMemTypeDevice(mtype)) {
    dev_ij = PETSC_TRUE;
    PetscCall(PetscMalloc2(coo_n, &i, coo_n, &j));
    PetscCallCUDA(cudaMemcpy(i, coo_i, coo_n * sizeof(PetscInt), cudaMemcpyDeviceToHost));
    PetscCallCUDA(cudaMemcpy(j, coo_j, coo_n * sizeof(PetscInt), cudaMemcpyDeviceToHost));
  } else {
    i = coo_i;
    j = coo_j;
  }

  PetscCall(MatSetPreallocationCOO_MPIAIJ(mat, coo_n, i, j));
  if (dev_ij) PetscCall(PetscFree2(i, j));
  mat->offloadmask = PETSC_OFFLOAD_CPU;
  // Create the GPU memory
  PetscCall(MatSeqAIJCUSPARSECopyToGPU(mpiaij->A));
  PetscCall(MatSeqAIJCUSPARSECopyToGPU(mpiaij->B));

  // Copy the COO struct to device
  PetscCall(PetscObjectQuery((PetscObject)mat, "__PETSc_MatCOOStruct_Host", (PetscObject *)&container_h));
  PetscCall(PetscContainerGetPointer(container_h, (void **)&coo_h));
  PetscCall(PetscMalloc1(1, &coo_d));
  *coo_d = *coo_h; // do a shallow copy and then amend fields in coo_d

  PetscCall(PetscObjectReference((PetscObject)coo_d->sf)); // Since we destroy the sf in both coo_h and coo_d
  PetscCallCUDA(cudaMalloc((void **)&coo_d->Ajmap1, (coo_h->Annz + 1) * sizeof(PetscCount)));
  PetscCallCUDA(cudaMalloc((void **)&coo_d->Aperm1, coo_h->Atot1 * sizeof(PetscCount)));
  PetscCallCUDA(cudaMalloc((void **)&coo_d->Bjmap1, (coo_h->Bnnz + 1) * sizeof(PetscCount)));
  PetscCallCUDA(cudaMalloc((void **)&coo_d->Bperm1, coo_h->Btot1 * sizeof(PetscCount)));
  PetscCallCUDA(cudaMalloc((void **)&coo_d->Aimap2, coo_h->Annz2 * sizeof(PetscCount)));
  PetscCallCUDA(cudaMalloc((void **)&coo_d->Ajmap2, (coo_h->Annz2 + 1) * sizeof(PetscCount)));
  PetscCallCUDA(cudaMalloc((void **)&coo_d->Aperm2, coo_h->Atot2 * sizeof(PetscCount)));
  PetscCallCUDA(cudaMalloc((void **)&coo_d->Bimap2, coo_h->Bnnz2 * sizeof(PetscCount)));
  PetscCallCUDA(cudaMalloc((void **)&coo_d->Bjmap2, (coo_h->Bnnz2 + 1) * sizeof(PetscCount)));
  PetscCallCUDA(cudaMalloc((void **)&coo_d->Bperm2, coo_h->Btot2 * sizeof(PetscCount)));
  PetscCallCUDA(cudaMalloc((void **)&coo_d->Cperm1, coo_h->sendlen * sizeof(PetscCount)));
  PetscCallCUDA(cudaMalloc((void **)&coo_d->sendbuf, coo_h->sendlen * sizeof(PetscScalar)));
  PetscCallCUDA(cudaMalloc((void **)&coo_d->recvbuf, coo_h->recvlen * sizeof(PetscScalar)));

  PetscCallCUDA(cudaMemcpy(coo_d->Ajmap1, coo_h->Ajmap1, (coo_h->Annz + 1) * sizeof(PetscCount), cudaMemcpyHostToDevice));
  PetscCallCUDA(cudaMemcpy(coo_d->Aperm1, coo_h->Aperm1, coo_h->Atot1 * sizeof(PetscCount), cudaMemcpyHostToDevice));
  PetscCallCUDA(cudaMemcpy(coo_d->Bjmap1, coo_h->Bjmap1, (coo_h->Bnnz + 1) * sizeof(PetscCount), cudaMemcpyHostToDevice));
  PetscCallCUDA(cudaMemcpy(coo_d->Bperm1, coo_h->Bperm1, coo_h->Btot1 * sizeof(PetscCount), cudaMemcpyHostToDevice));
  PetscCallCUDA(cudaMemcpy(coo_d->Aimap2, coo_h->Aimap2, coo_h->Annz2 * sizeof(PetscCount), cudaMemcpyHostToDevice));
  PetscCallCUDA(cudaMemcpy(coo_d->Ajmap2, coo_h->Ajmap2, (coo_h->Annz2 + 1) * sizeof(PetscCount), cudaMemcpyHostToDevice));
  PetscCallCUDA(cudaMemcpy(coo_d->Aperm2, coo_h->Aperm2, coo_h->Atot2 * sizeof(PetscCount), cudaMemcpyHostToDevice));
  PetscCallCUDA(cudaMemcpy(coo_d->Bimap2, coo_h->Bimap2, coo_h->Bnnz2 * sizeof(PetscCount), cudaMemcpyHostToDevice));
  PetscCallCUDA(cudaMemcpy(coo_d->Bjmap2, coo_h->Bjmap2, (coo_h->Bnnz2 + 1) * sizeof(PetscCount), cudaMemcpyHostToDevice));
  PetscCallCUDA(cudaMemcpy(coo_d->Bperm2, coo_h->Bperm2, coo_h->Btot2 * sizeof(PetscCount), cudaMemcpyHostToDevice));
  PetscCallCUDA(cudaMemcpy(coo_d->Cperm1, coo_h->Cperm1, coo_h->sendlen * sizeof(PetscCount), cudaMemcpyHostToDevice));

  // Put the COO struct in a container and then attach that to the matrix
  PetscCall(PetscObjectContainerCompose((PetscObject)mat, "__PETSc_MatCOOStruct_Device", coo_d, MatCOOStructDestroy_MPIAIJCUSPARSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

__global__ static void MatPackCOOValues(const PetscScalar kv[], PetscCount nnz, const PetscCount perm[], PetscScalar buf[])
{
  PetscCount       i         = blockIdx.x * blockDim.x + threadIdx.x;
  const PetscCount grid_size = gridDim.x * blockDim.x;
  for (; i < nnz; i += grid_size) buf[i] = kv[perm[i]];
}

__global__ static void MatAddLocalCOOValues(const PetscScalar kv[], InsertMode imode, PetscCount Annz, const PetscCount Ajmap1[], const PetscCount Aperm1[], PetscScalar Aa[], PetscCount Bnnz, const PetscCount Bjmap1[], const PetscCount Bperm1[], PetscScalar Ba[])
{
  PetscCount       i         = blockIdx.x * blockDim.x + threadIdx.x;
  const PetscCount grid_size = gridDim.x * blockDim.x;
  for (; i < Annz + Bnnz; i += grid_size) {
    PetscScalar sum = 0.0;
    if (i < Annz) {
      for (PetscCount k = Ajmap1[i]; k < Ajmap1[i + 1]; k++) sum += kv[Aperm1[k]];
      Aa[i] = (imode == INSERT_VALUES ? 0.0 : Aa[i]) + sum;
    } else {
      i -= Annz;
      for (PetscCount k = Bjmap1[i]; k < Bjmap1[i + 1]; k++) sum += kv[Bperm1[k]];
      Ba[i] = (imode == INSERT_VALUES ? 0.0 : Ba[i]) + sum;
    }
  }
}

__global__ static void MatAddRemoteCOOValues(const PetscScalar kv[], PetscCount Annz2, const PetscCount Aimap2[], const PetscCount Ajmap2[], const PetscCount Aperm2[], PetscScalar Aa[], PetscCount Bnnz2, const PetscCount Bimap2[], const PetscCount Bjmap2[], const PetscCount Bperm2[], PetscScalar Ba[])
{
  PetscCount       i         = blockIdx.x * blockDim.x + threadIdx.x;
  const PetscCount grid_size = gridDim.x * blockDim.x;
  for (; i < Annz2 + Bnnz2; i += grid_size) {
    if (i < Annz2) {
      for (PetscCount k = Ajmap2[i]; k < Ajmap2[i + 1]; k++) Aa[Aimap2[i]] += kv[Aperm2[k]];
    } else {
      i -= Annz2;
      for (PetscCount k = Bjmap2[i]; k < Bjmap2[i + 1]; k++) Ba[Bimap2[i]] += kv[Bperm2[k]];
    }
  }
}

static PetscErrorCode MatSetValuesCOO_MPIAIJCUSPARSE(Mat mat, const PetscScalar v[], InsertMode imode)
{
  Mat_MPIAIJ          *mpiaij = static_cast<Mat_MPIAIJ *>(mat->data);
  Mat                  A = mpiaij->A, B = mpiaij->B;
  PetscScalar         *Aa, *Ba;
  const PetscScalar   *v1 = v;
  PetscMemType         memtype;
  PetscContainer       container;
  MatCOOStruct_MPIAIJ *coo;

  PetscFunctionBegin;
  PetscCall(PetscObjectQuery((PetscObject)mat, "__PETSc_MatCOOStruct_Device", (PetscObject *)&container));
  PetscCheck(container, PetscObjectComm((PetscObject)mat), PETSC_ERR_PLIB, "Not found MatCOOStruct on this matrix");
  PetscCall(PetscContainerGetPointer(container, (void **)&coo));

  const auto &Annz   = coo->Annz;
  const auto &Annz2  = coo->Annz2;
  const auto &Bnnz   = coo->Bnnz;
  const auto &Bnnz2  = coo->Bnnz2;
  const auto &vsend  = coo->sendbuf;
  const auto &v2     = coo->recvbuf;
  const auto &Ajmap1 = coo->Ajmap1;
  const auto &Ajmap2 = coo->Ajmap2;
  const auto &Aimap2 = coo->Aimap2;
  const auto &Bjmap1 = coo->Bjmap1;
  const auto &Bjmap2 = coo->Bjmap2;
  const auto &Bimap2 = coo->Bimap2;
  const auto &Aperm1 = coo->Aperm1;
  const auto &Aperm2 = coo->Aperm2;
  const auto &Bperm1 = coo->Bperm1;
  const auto &Bperm2 = coo->Bperm2;
  const auto &Cperm1 = coo->Cperm1;

  PetscCall(PetscGetMemType(v, &memtype));
  if (PetscMemTypeHost(memtype)) { /* If user gave v[] in host, we need to copy it to device */
    PetscCallCUDA(cudaMalloc((void **)&v1, coo->n * sizeof(PetscScalar)));
    PetscCallCUDA(cudaMemcpy((void *)v1, v, coo->n * sizeof(PetscScalar), cudaMemcpyHostToDevice));
  }

  if (imode == INSERT_VALUES) {
    PetscCall(MatSeqAIJCUSPARSEGetArrayWrite(A, &Aa)); /* write matrix values */
    PetscCall(MatSeqAIJCUSPARSEGetArrayWrite(B, &Ba));
  } else {
    PetscCall(MatSeqAIJCUSPARSEGetArray(A, &Aa)); /* read & write matrix values */
    PetscCall(MatSeqAIJCUSPARSEGetArray(B, &Ba));
  }

  PetscCall(PetscLogGpuTimeBegin());
  /* Pack entries to be sent to remote */
  if (coo->sendlen) {
    MatPackCOOValues<<<(coo->sendlen + 255) / 256, 256>>>(v1, coo->sendlen, Cperm1, vsend);
    PetscCallCUDA(cudaPeekAtLastError());
  }

  /* Send remote entries to their owner and overlap the communication with local computation */
  PetscCall(PetscSFReduceWithMemTypeBegin(coo->sf, MPIU_SCALAR, PETSC_MEMTYPE_CUDA, vsend, PETSC_MEMTYPE_CUDA, v2, MPI_REPLACE));
  /* Add local entries to A and B */
  if (Annz + Bnnz > 0) {
    MatAddLocalCOOValues<<<(int)((Annz + Bnnz + 255) / 256), 256>>>(v1, imode, Annz, Ajmap1, Aperm1, Aa, Bnnz, Bjmap1, Bperm1, Ba);
    PetscCallCUDA(cudaPeekAtLastError());
  }
  PetscCall(PetscSFReduceEnd(coo->sf, MPIU_SCALAR, vsend, v2, MPI_REPLACE));

  /* Add received remote entries to A and B */
  if (Annz2 + Bnnz2 > 0) {
    MatAddRemoteCOOValues<<<(int)((Annz2 + Bnnz2 + 255) / 256), 256>>>(v2, Annz2, Aimap2, Ajmap2, Aperm2, Aa, Bnnz2, Bimap2, Bjmap2, Bperm2, Ba);
    PetscCallCUDA(cudaPeekAtLastError());
  }
  PetscCall(PetscLogGpuTimeEnd());

  if (imode == INSERT_VALUES) {
    PetscCall(MatSeqAIJCUSPARSERestoreArrayWrite(A, &Aa));
    PetscCall(MatSeqAIJCUSPARSERestoreArrayWrite(B, &Ba));
  } else {
    PetscCall(MatSeqAIJCUSPARSERestoreArray(A, &Aa));
    PetscCall(MatSeqAIJCUSPARSERestoreArray(B, &Ba));
  }
  if (PetscMemTypeHost(memtype)) PetscCallCUDA(cudaFree((void *)v1));
  mat->offloadmask = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMPIAIJGetLocalMatMerge_MPIAIJCUSPARSE(Mat A, MatReuse scall, IS *glob, Mat *A_loc)
{
  Mat             Ad, Ao;
  const PetscInt *cmap;

  PetscFunctionBegin;
  PetscCall(MatMPIAIJGetSeqAIJ(A, &Ad, &Ao, &cmap));
  PetscCall(MatSeqAIJCUSPARSEMergeMats(Ad, Ao, scall, A_loc));
  if (glob) {
    PetscInt cst, i, dn, on, *gidx;

    PetscCall(MatGetLocalSize(Ad, NULL, &dn));
    PetscCall(MatGetLocalSize(Ao, NULL, &on));
    PetscCall(MatGetOwnershipRangeColumn(A, &cst, NULL));
    PetscCall(PetscMalloc1(dn + on, &gidx));
    for (i = 0; i < dn; i++) gidx[i] = cst + i;
    for (i = 0; i < on; i++) gidx[i + dn] = cmap[i];
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)Ad), dn + on, gidx, PETSC_OWN_POINTER, glob));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMPIAIJSetPreallocation_MPIAIJCUSPARSE(Mat B, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[])
{
  Mat_MPIAIJ         *b              = (Mat_MPIAIJ *)B->data;
  Mat_MPIAIJCUSPARSE *cusparseStruct = (Mat_MPIAIJCUSPARSE *)b->spptr;
  PetscInt            i;

  PetscFunctionBegin;
  if (B->hash_active) {
    B->ops[0]      = b->cops;
    B->hash_active = PETSC_FALSE;
  }
  PetscCall(PetscLayoutSetUp(B->rmap));
  PetscCall(PetscLayoutSetUp(B->cmap));
  if (PetscDefined(USE_DEBUG) && d_nnz) {
    for (i = 0; i < B->rmap->n; i++) PetscCheck(d_nnz[i] >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "d_nnz cannot be less than 0: local row %" PetscInt_FMT " value %" PetscInt_FMT, i, d_nnz[i]);
  }
  if (PetscDefined(USE_DEBUG) && o_nnz) {
    for (i = 0; i < B->rmap->n; i++) PetscCheck(o_nnz[i] >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "o_nnz cannot be less than 0: local row %" PetscInt_FMT " value %" PetscInt_FMT, i, o_nnz[i]);
  }
#if defined(PETSC_USE_CTABLE)
  PetscCall(PetscHMapIDestroy(&b->colmap));
#else
  PetscCall(PetscFree(b->colmap));
#endif
  PetscCall(PetscFree(b->garray));
  PetscCall(VecDestroy(&b->lvec));
  PetscCall(VecScatterDestroy(&b->Mvctx));
  /* Because the B will have been resized we simply destroy it and create a new one each time */
  PetscCall(MatDestroy(&b->B));
  if (!b->A) {
    PetscCall(MatCreate(PETSC_COMM_SELF, &b->A));
    PetscCall(MatSetSizes(b->A, B->rmap->n, B->cmap->n, B->rmap->n, B->cmap->n));
  }
  if (!b->B) {
    PetscMPIInt size;
    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)B), &size));
    PetscCall(MatCreate(PETSC_COMM_SELF, &b->B));
    PetscCall(MatSetSizes(b->B, B->rmap->n, size > 1 ? B->cmap->N : 0, B->rmap->n, size > 1 ? B->cmap->N : 0));
  }
  PetscCall(MatSetType(b->A, MATSEQAIJCUSPARSE));
  PetscCall(MatSetType(b->B, MATSEQAIJCUSPARSE));
  PetscCall(MatBindToCPU(b->A, B->boundtocpu));
  PetscCall(MatBindToCPU(b->B, B->boundtocpu));
  PetscCall(MatSeqAIJSetPreallocation(b->A, d_nz, d_nnz));
  PetscCall(MatSeqAIJSetPreallocation(b->B, o_nz, o_nnz));
  PetscCall(MatCUSPARSESetFormat(b->A, MAT_CUSPARSE_MULT, cusparseStruct->diagGPUMatFormat));
  PetscCall(MatCUSPARSESetFormat(b->B, MAT_CUSPARSE_MULT, cusparseStruct->offdiagGPUMatFormat));
  B->preallocated  = PETSC_TRUE;
  B->was_assembled = PETSC_FALSE;
  B->assembled     = PETSC_FALSE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMult_MPIAIJCUSPARSE(Mat A, Vec xx, Vec yy)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(VecScatterBegin(a->Mvctx, xx, a->lvec, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall((*a->A->ops->mult)(a->A, xx, yy));
  PetscCall(VecScatterEnd(a->Mvctx, xx, a->lvec, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall((*a->B->ops->multadd)(a->B, a->lvec, yy, yy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatZeroEntries_MPIAIJCUSPARSE(Mat A)
{
  Mat_MPIAIJ *l = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(MatZeroEntries(l->A));
  PetscCall(MatZeroEntries(l->B));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultAdd_MPIAIJCUSPARSE(Mat A, Vec xx, Vec yy, Vec zz)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(VecScatterBegin(a->Mvctx, xx, a->lvec, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall((*a->A->ops->multadd)(a->A, xx, yy, zz));
  PetscCall(VecScatterEnd(a->Mvctx, xx, a->lvec, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall((*a->B->ops->multadd)(a->B, a->lvec, zz, zz));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatMultTranspose_MPIAIJCUSPARSE(Mat A, Vec xx, Vec yy)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall((*a->B->ops->multtranspose)(a->B, xx, a->lvec));
  PetscCall((*a->A->ops->multtranspose)(a->A, xx, yy));
  PetscCall(VecScatterBegin(a->Mvctx, a->lvec, yy, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(a->Mvctx, a->lvec, yy, ADD_VALUES, SCATTER_REVERSE));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatCUSPARSESetFormat_MPIAIJCUSPARSE(Mat A, MatCUSPARSEFormatOperation op, MatCUSPARSEStorageFormat format)
{
  Mat_MPIAIJ         *a              = (Mat_MPIAIJ *)A->data;
  Mat_MPIAIJCUSPARSE *cusparseStruct = (Mat_MPIAIJCUSPARSE *)a->spptr;

  PetscFunctionBegin;
  switch (op) {
  case MAT_CUSPARSE_MULT_DIAG:
    cusparseStruct->diagGPUMatFormat = format;
    break;
  case MAT_CUSPARSE_MULT_OFFDIAG:
    cusparseStruct->offdiagGPUMatFormat = format;
    break;
  case MAT_CUSPARSE_ALL:
    cusparseStruct->diagGPUMatFormat    = format;
    cusparseStruct->offdiagGPUMatFormat = format;
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "unsupported operation %d for MatCUSPARSEFormatOperation. Only MAT_CUSPARSE_MULT_DIAG, MAT_CUSPARSE_MULT_DIAG, and MAT_CUSPARSE_MULT_ALL are currently supported.", op);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatSetFromOptions_MPIAIJCUSPARSE(Mat A, PetscOptionItems PetscOptionsObject)
{
  MatCUSPARSEStorageFormat format;
  PetscBool                flg;
  Mat_MPIAIJ              *a              = (Mat_MPIAIJ *)A->data;
  Mat_MPIAIJCUSPARSE      *cusparseStruct = (Mat_MPIAIJCUSPARSE *)a->spptr;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "MPIAIJCUSPARSE options");
  if (A->factortype == MAT_FACTOR_NONE) {
    PetscCall(PetscOptionsEnum("-mat_cusparse_mult_diag_storage_format", "sets storage format of the diagonal blocks of (mpi)aijcusparse gpu matrices for SpMV", "MatCUSPARSESetFormat", MatCUSPARSEStorageFormats, (PetscEnum)cusparseStruct->diagGPUMatFormat, (PetscEnum *)&format, &flg));
    if (flg) PetscCall(MatCUSPARSESetFormat(A, MAT_CUSPARSE_MULT_DIAG, format));
    PetscCall(PetscOptionsEnum("-mat_cusparse_mult_offdiag_storage_format", "sets storage format of the off-diagonal blocks (mpi)aijcusparse gpu matrices for SpMV", "MatCUSPARSESetFormat", MatCUSPARSEStorageFormats, (PetscEnum)cusparseStruct->offdiagGPUMatFormat, (PetscEnum *)&format, &flg));
    if (flg) PetscCall(MatCUSPARSESetFormat(A, MAT_CUSPARSE_MULT_OFFDIAG, format));
    PetscCall(PetscOptionsEnum("-mat_cusparse_storage_format", "sets storage format of the diagonal and off-diagonal blocks (mpi)aijcusparse gpu matrices for SpMV", "MatCUSPARSESetFormat", MatCUSPARSEStorageFormats, (PetscEnum)cusparseStruct->diagGPUMatFormat, (PetscEnum *)&format, &flg));
    if (flg) PetscCall(MatCUSPARSESetFormat(A, MAT_CUSPARSE_ALL, format));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatAssemblyEnd_MPIAIJCUSPARSE(Mat A, MatAssemblyType mode)
{
  Mat_MPIAIJ *mpiaij = (Mat_MPIAIJ *)A->data;

  PetscFunctionBegin;
  PetscCall(MatAssemblyEnd_MPIAIJ(A, mode));
  if (mpiaij->lvec) PetscCall(VecSetType(mpiaij->lvec, VECSEQCUDA));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatDestroy_MPIAIJCUSPARSE(Mat A)
{
  Mat_MPIAIJ         *aij            = (Mat_MPIAIJ *)A->data;
  Mat_MPIAIJCUSPARSE *cusparseStruct = (Mat_MPIAIJCUSPARSE *)aij->spptr;

  PetscFunctionBegin;
  PetscCheck(cusparseStruct, PETSC_COMM_SELF, PETSC_ERR_COR, "Missing spptr");
  PetscCallCXX(delete cusparseStruct);
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMPIAIJSetPreallocation_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMPIAIJGetLocalMatMerge_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetPreallocationCOO_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetValuesCOO_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatCUSPARSESetFormat_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_mpiaijcusparse_hypre_C", NULL));
  PetscCall(MatDestroy_MPIAIJ(A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* defines MatSetValues_MPICUSPARSE_Hash() */
#define TYPE AIJ
#define TYPE_AIJ
#define SUB_TYPE_CUSPARSE
#include "../src/mat/impls/aij/mpi/mpihashmat.h"
#undef TYPE
#undef TYPE_AIJ
#undef SUB_TYPE_CUSPARSE

static PetscErrorCode MatSetUp_MPI_HASH_CUSPARSE(Mat A)
{
  Mat_MPIAIJ         *b              = (Mat_MPIAIJ *)A->data;
  Mat_MPIAIJCUSPARSE *cusparseStruct = (Mat_MPIAIJCUSPARSE *)b->spptr;

  PetscFunctionBegin;
  PetscCall(MatSetUp_MPI_Hash(A));
  PetscCall(MatCUSPARSESetFormat(b->A, MAT_CUSPARSE_MULT, cusparseStruct->diagGPUMatFormat));
  PetscCall(MatCUSPARSESetFormat(b->B, MAT_CUSPARSE_MULT, cusparseStruct->offdiagGPUMatFormat));
  A->preallocated = PETSC_TRUE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIAIJCUSPARSE(Mat B, MatType, MatReuse reuse, Mat *newmat)
{
  Mat_MPIAIJ *a;
  Mat         A;

  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
  if (reuse == MAT_INITIAL_MATRIX) PetscCall(MatDuplicate(B, MAT_COPY_VALUES, newmat));
  else if (reuse == MAT_REUSE_MATRIX) PetscCall(MatCopy(B, *newmat, SAME_NONZERO_PATTERN));
  A             = *newmat;
  A->boundtocpu = PETSC_FALSE;
  PetscCall(PetscFree(A->defaultvectype));
  PetscCall(PetscStrallocpy(VECCUDA, &A->defaultvectype));

  a = (Mat_MPIAIJ *)A->data;
  if (a->A) PetscCall(MatSetType(a->A, MATSEQAIJCUSPARSE));
  if (a->B) PetscCall(MatSetType(a->B, MATSEQAIJCUSPARSE));
  if (a->lvec) PetscCall(VecSetType(a->lvec, VECSEQCUDA));

  if (reuse != MAT_REUSE_MATRIX && !a->spptr) PetscCallCXX(a->spptr = new Mat_MPIAIJCUSPARSE);

  A->ops->assemblyend           = MatAssemblyEnd_MPIAIJCUSPARSE;
  A->ops->mult                  = MatMult_MPIAIJCUSPARSE;
  A->ops->multadd               = MatMultAdd_MPIAIJCUSPARSE;
  A->ops->multtranspose         = MatMultTranspose_MPIAIJCUSPARSE;
  A->ops->setfromoptions        = MatSetFromOptions_MPIAIJCUSPARSE;
  A->ops->destroy               = MatDestroy_MPIAIJCUSPARSE;
  A->ops->zeroentries           = MatZeroEntries_MPIAIJCUSPARSE;
  A->ops->productsetfromoptions = MatProductSetFromOptions_MPIAIJBACKEND;
  A->ops->setup                 = MatSetUp_MPI_HASH_CUSPARSE;
  A->ops->getcurrentmemtype     = MatGetCurrentMemType_MPIAIJ;

  PetscCall(PetscObjectChangeTypeName((PetscObject)A, MATMPIAIJCUSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMPIAIJGetLocalMatMerge_C", MatMPIAIJGetLocalMatMerge_MPIAIJCUSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatMPIAIJSetPreallocation_C", MatMPIAIJSetPreallocation_MPIAIJCUSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatCUSPARSESetFormat_C", MatCUSPARSESetFormat_MPIAIJCUSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetPreallocationCOO_C", MatSetPreallocationCOO_MPIAIJCUSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatSetValuesCOO_C", MatSetValuesCOO_MPIAIJCUSPARSE));
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatConvert_mpiaijcusparse_hypre_C", MatConvert_AIJ_HYPRE));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJCUSPARSE(Mat A)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
  PetscCall(MatCreate_MPIAIJ(A));
  PetscCall(MatConvert_MPIAIJ_MPIAIJCUSPARSE(A, MATMPIAIJCUSPARSE, MAT_INPLACE_MATRIX, &A));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  MatCreateAIJCUSPARSE - Creates a sparse matrix in `MATAIJCUSPARSE` (compressed row) format. This matrix will ultimately be pushed down
  to NVIDIA GPUs and use the CuSPARSE library for calculations.

  Collective

  Input Parameters:
+ comm  - MPI communicator, set to `PETSC_COMM_SELF`
. m     - number of local rows (or `PETSC_DECIDE` to have calculated if `M` is given)
          This value should be the same as the local size used in creating the
          $y$ vector for the matrix-vector product $y = Ax$.
. n     - This value should be the same as the local size used in creating the
          $x$ vector for the matrix-vector product $y = Ax$. (or `PETSC_DECIDE` to have
          calculated if `N` is given) For square matrices `n` is almost always `m`.
. M     - number of global rows (or `PETSC_DETERMINE` to have calculated if `m` is given)
. N     - number of global columns (or `PETSC_DETERMINE` to have calculated if `n` is given)
. d_nz  - number of nonzeros per row in DIAGONAL portion of local submatrix
          (same value is used for all local rows)
. d_nnz - array containing the number of nonzeros in the various rows of the
          DIAGONAL portion of the local submatrix (possibly different for each row)
          or `NULL`, if `d_nz` is used to specify the nonzero structure.
          The size of this array is equal to the number of local rows, i.e `m`.
          For matrices you plan to factor you must leave room for the diagonal entry and
          put in the entry even if it is zero.
. o_nz  - number of nonzeros per row in the OFF-DIAGONAL portion of local
          submatrix (same value is used for all local rows).
- o_nnz - array containing the number of nonzeros in the various rows of the
          OFF-DIAGONAL portion of the local submatrix (possibly different for
          each row) or `NULL`, if `o_nz` is used to specify the nonzero
          structure. The size of this array is equal to the number
          of local rows, i.e `m`.

  Output Parameter:
. A - the matrix

  Level: intermediate

  Notes:
  It is recommended that one use the `MatCreate()`, `MatSetType()` and/or `MatSetFromOptions()`,
  MatXXXXSetPreallocation() paradigm instead of this routine directly.
  [MatXXXXSetPreallocation() is, for example, `MatSeqAIJSetPreallocation()`]

  The AIJ format, also called the
  compressed row storage), is fully compatible with standard Fortran
  storage.  That is, the stored row and column indices can begin at
  either one (as in Fortran) or zero.

  When working with matrices for GPUs, it is often better to use the `MatSetPreallocationCOO()` and `MatSetValuesCOO()` paradigm rather than using this routine and `MatSetValues()`

.seealso: [](ch_matrices), `Mat`, `MATAIJCUSPARSE`, `MatCreate()`, `MatCreateAIJ()`, `MatSetValues()`, `MatSeqAIJSetColumnIndices()`, `MatCreateSeqAIJWithArrays()`, `MATMPIAIJCUSPARSE`
@*/
PetscErrorCode MatCreateAIJCUSPARSE(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt d_nz, const PetscInt d_nnz[], PetscInt o_nz, const PetscInt o_nnz[], Mat *A)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCall(MatCreate(comm, A));
  PetscCall(MatSetSizes(*A, m, n, M, N));
  PetscCallMPI(MPI_Comm_size(comm, &size));
  if (size > 1) {
    PetscCall(MatSetType(*A, MATMPIAIJCUSPARSE));
    PetscCall(MatMPIAIJSetPreallocation(*A, d_nz, d_nnz, o_nz, o_nnz));
  } else {
    PetscCall(MatSetType(*A, MATSEQAIJCUSPARSE));
    PetscCall(MatSeqAIJSetPreallocation(*A, d_nz, d_nnz));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
   MATMPIAIJCUSPARSE - A matrix type to be used for sparse matrices on NVIDIA GPUs.

   Options Database Keys:
+  -mat_type mpiaijcusparse                      - sets the matrix type to `MATMPIAIJCUSPARSE`
.  -mat_cusparse_storage_format csr              - sets the storage format of diagonal and off-diagonal matrices. Other options include ell (ellpack) or hyb (hybrid).
.  -mat_cusparse_mult_diag_storage_format csr    - sets the storage format of diagonal matrix. Other options include ell (ellpack) or hyb (hybrid).
-  -mat_cusparse_mult_offdiag_storage_format csr - sets the storage format of off-diagonal matrix. Other options include ell (ellpack) or hyb (hybrid).

  Level: beginner

  Notes:
  These matrices can be in either CSR, ELL, or HYB format. The ELL and HYB formats require CUDA 4.2 or later.

  All matrix calculations are performed on NVIDIA GPUs using the cuSPARSE library.

  Uses 32-bit integers internally. If PETSc is configured `--with-64-bit-indices`, the integer row and column indices are stored on the GPU with `int`. It is unclear what happens
  if some integer values passed in do not fit in `int`.

.seealso: [](ch_matrices), `Mat`, `MatCreateAIJCUSPARSE()`, `MATSEQAIJCUSPARSE`, `MATMPIAIJCUSPARSE`, `MatCreateSeqAIJCUSPARSE()`, `MatCUSPARSESetFormat()`, `MatCUSPARSEStorageFormat`, `MatCUSPARSEFormatOperation`
M*/

/*MC
   MATAIJCUSPARSE - A matrix type to be used for sparse matrices on NVIDIA GPUs; it is as same as `MATSEQAIJCUSPARSE` on one MPI process and `MATMPIAIJCUSPARSE` on multiple processes.

  Level: beginner

.seealso: [](ch_matrices), `Mat`, `MATAIJCUSPARSE`, `MATSEQAIJCUSPARSE`
M*/
