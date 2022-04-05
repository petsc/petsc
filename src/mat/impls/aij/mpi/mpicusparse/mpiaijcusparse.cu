#define PETSC_SKIP_SPINLOCK
#define PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND 1

#include <petscconf.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>   /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/seqcusparse/cusparsematimpl.h>
#include <../src/mat/impls/aij/mpi/mpicusparse/mpicusparsematimpl.h>
#include <thrust/advance.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <petscsf.h>

struct VecCUDAEquals
{
  template <typename Tuple>
  __host__ __device__
  void operator()(Tuple t)
  {
    thrust::get<1>(t) = thrust::get<0>(t);
  }
};

static PetscErrorCode MatResetPreallocationCOO_MPIAIJCUSPARSE(Mat mat)
{
  Mat_MPIAIJ         *aij = (Mat_MPIAIJ*)mat->data;
  Mat_MPIAIJCUSPARSE *cusparseStruct = (Mat_MPIAIJCUSPARSE*)aij->spptr;

  PetscFunctionBegin;
  if (!cusparseStruct) PetscFunctionReturn(0);
  if (cusparseStruct->use_extended_coo) {
    PetscCallCUDA(cudaFree(cusparseStruct->Aimap1_d));
    PetscCallCUDA(cudaFree(cusparseStruct->Ajmap1_d));
    PetscCallCUDA(cudaFree(cusparseStruct->Aperm1_d));
    PetscCallCUDA(cudaFree(cusparseStruct->Bimap1_d));
    PetscCallCUDA(cudaFree(cusparseStruct->Bjmap1_d));
    PetscCallCUDA(cudaFree(cusparseStruct->Bperm1_d));
    PetscCallCUDA(cudaFree(cusparseStruct->Aimap2_d));
    PetscCallCUDA(cudaFree(cusparseStruct->Ajmap2_d));
    PetscCallCUDA(cudaFree(cusparseStruct->Aperm2_d));
    PetscCallCUDA(cudaFree(cusparseStruct->Bimap2_d));
    PetscCallCUDA(cudaFree(cusparseStruct->Bjmap2_d));
    PetscCallCUDA(cudaFree(cusparseStruct->Bperm2_d));
    PetscCallCUDA(cudaFree(cusparseStruct->Cperm1_d));
    PetscCallCUDA(cudaFree(cusparseStruct->sendbuf_d));
    PetscCallCUDA(cudaFree(cusparseStruct->recvbuf_d));
  }
  cusparseStruct->use_extended_coo = PETSC_FALSE;
  delete cusparseStruct->coo_p;
  delete cusparseStruct->coo_pw;
  cusparseStruct->coo_p  = NULL;
  cusparseStruct->coo_pw = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValuesCOO_MPIAIJCUSPARSE_Basic(Mat A, const PetscScalar v[], InsertMode imode)
{
  Mat_MPIAIJ         *a = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJCUSPARSE *cusp = (Mat_MPIAIJCUSPARSE*)a->spptr;
  PetscInt           n = cusp->coo_nd + cusp->coo_no;

  PetscFunctionBegin;
  if (cusp->coo_p && v) {
    thrust::device_ptr<const PetscScalar> d_v;
    THRUSTARRAY                           *w = NULL;

    if (isCudaMem(v)) {
      d_v = thrust::device_pointer_cast(v);
    } else {
      w = new THRUSTARRAY(n);
      w->assign(v,v+n);
      PetscCall(PetscLogCpuToGpu(n*sizeof(PetscScalar)));
      d_v = w->data();
    }

    auto zibit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(d_v,cusp->coo_p->begin()),
                                                              cusp->coo_pw->begin()));
    auto zieit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(d_v,cusp->coo_p->end()),
                                                              cusp->coo_pw->end()));
    PetscCall(PetscLogGpuTimeBegin());
    thrust::for_each(zibit,zieit,VecCUDAEquals());
    PetscCall(PetscLogGpuTimeEnd());
    delete w;
    PetscCall(MatSetValuesCOO_SeqAIJCUSPARSE_Basic(a->A,cusp->coo_pw->data().get(),imode));
    PetscCall(MatSetValuesCOO_SeqAIJCUSPARSE_Basic(a->B,cusp->coo_pw->data().get()+cusp->coo_nd,imode));
  } else {
    PetscCall(MatSetValuesCOO_SeqAIJCUSPARSE_Basic(a->A,v,imode));
    PetscCall(MatSetValuesCOO_SeqAIJCUSPARSE_Basic(a->B,v ? v+cusp->coo_nd : NULL,imode));
  }
  PetscFunctionReturn(0);
}

template <typename Tuple>
struct IsNotOffDiagT
{
  PetscInt _cstart,_cend;

  IsNotOffDiagT(PetscInt cstart, PetscInt cend) : _cstart(cstart), _cend(cend) {}
  __host__ __device__
  inline bool operator()(Tuple t)
  {
    return !(thrust::get<1>(t) < _cstart || thrust::get<1>(t) >= _cend);
  }
};

struct IsOffDiag
{
  PetscInt _cstart,_cend;

  IsOffDiag(PetscInt cstart, PetscInt cend) : _cstart(cstart), _cend(cend) {}
  __host__ __device__
  inline bool operator() (const PetscInt &c)
  {
    return c < _cstart || c >= _cend;
  }
};

struct GlobToLoc
{
  PetscInt _start;

  GlobToLoc(PetscInt start) : _start(start) {}
  __host__ __device__
  inline PetscInt operator() (const PetscInt &c)
  {
    return c - _start;
  }
};

static PetscErrorCode MatSetPreallocationCOO_MPIAIJCUSPARSE_Basic(Mat B, PetscCount n, const PetscInt coo_i[], const PetscInt coo_j[])
{
  Mat_MPIAIJ             *b = (Mat_MPIAIJ*)B->data;
  Mat_MPIAIJCUSPARSE     *cusp = (Mat_MPIAIJCUSPARSE*)b->spptr;
  PetscInt               N,*jj;
  size_t                 noff = 0;
  THRUSTINTARRAY         d_i(n); /* on device, storing partitioned coo_i with diagonal first, and off-diag next */
  THRUSTINTARRAY         d_j(n);
  ISLocalToGlobalMapping l2g;

  PetscFunctionBegin;
  PetscCall(MatDestroy(&b->A));
  PetscCall(MatDestroy(&b->B));

  PetscCall(PetscLogCpuToGpu(2.*n*sizeof(PetscInt)));
  d_i.assign(coo_i,coo_i+n);
  d_j.assign(coo_j,coo_j+n);
  PetscCall(PetscLogGpuTimeBegin());
  auto firstoffd = thrust::find_if(thrust::device,d_j.begin(),d_j.end(),IsOffDiag(B->cmap->rstart,B->cmap->rend));
  auto firstdiag = thrust::find_if_not(thrust::device,firstoffd,d_j.end(),IsOffDiag(B->cmap->rstart,B->cmap->rend));
  if (firstoffd != d_j.end() && firstdiag != d_j.end()) {
    cusp->coo_p = new THRUSTINTARRAY(n);
    cusp->coo_pw = new THRUSTARRAY(n);
    thrust::sequence(thrust::device,cusp->coo_p->begin(),cusp->coo_p->end(),0);
    auto fzipp = thrust::make_zip_iterator(thrust::make_tuple(d_i.begin(),d_j.begin(),cusp->coo_p->begin()));
    auto ezipp = thrust::make_zip_iterator(thrust::make_tuple(d_i.end(),d_j.end(),cusp->coo_p->end()));
    auto mzipp = thrust::partition(thrust::device,fzipp,ezipp,IsNotOffDiagT<thrust::tuple<PetscInt,PetscInt,PetscInt> >(B->cmap->rstart,B->cmap->rend));
    firstoffd = mzipp.get_iterator_tuple().get<1>();
  }
  cusp->coo_nd = thrust::distance(d_j.begin(),firstoffd);
  cusp->coo_no = thrust::distance(firstoffd,d_j.end());

  /* from global to local */
  thrust::transform(thrust::device,d_i.begin(),d_i.end(),d_i.begin(),GlobToLoc(B->rmap->rstart));
  thrust::transform(thrust::device,d_j.begin(),firstoffd,d_j.begin(),GlobToLoc(B->cmap->rstart));
  PetscCall(PetscLogGpuTimeEnd());

  /* copy offdiag column indices to map on the CPU */
  PetscCall(PetscMalloc1(cusp->coo_no,&jj)); /* jj[] will store compacted col ids of the offdiag part */
  PetscCallCUDA(cudaMemcpy(jj,d_j.data().get()+cusp->coo_nd,cusp->coo_no*sizeof(PetscInt),cudaMemcpyDeviceToHost));
  auto o_j = d_j.begin();
  PetscCall(PetscLogGpuTimeBegin());
  thrust::advance(o_j,cusp->coo_nd); /* sort and unique offdiag col ids */
  thrust::sort(thrust::device,o_j,d_j.end());
  auto wit = thrust::unique(thrust::device,o_j,d_j.end()); /* return end iter of the unique range */
  PetscCall(PetscLogGpuTimeEnd());
  noff = thrust::distance(o_j,wit);
  /* allocate the garray, the columns of B will not be mapped in the multiply setup */
  PetscCall(PetscMalloc1(noff,&b->garray));
  PetscCallCUDA(cudaMemcpy(b->garray,d_j.data().get()+cusp->coo_nd,noff*sizeof(PetscInt),cudaMemcpyDeviceToHost));
  PetscCall(PetscLogGpuToCpu((noff+cusp->coo_no)*sizeof(PetscInt)));
  PetscCall(ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,1,noff,b->garray,PETSC_COPY_VALUES,&l2g));
  PetscCall(ISLocalToGlobalMappingSetType(l2g,ISLOCALTOGLOBALMAPPINGHASH));
  PetscCall(ISGlobalToLocalMappingApply(l2g,IS_GTOLM_DROP,cusp->coo_no,jj,&N,jj));
  PetscCheck(N == cusp->coo_no,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected is size %" PetscInt_FMT " != %" PetscInt_FMT " coo size",N,cusp->coo_no);
  PetscCall(ISLocalToGlobalMappingDestroy(&l2g));

  PetscCall(MatCreate(PETSC_COMM_SELF,&b->A));
  PetscCall(MatSetSizes(b->A,B->rmap->n,B->cmap->n,B->rmap->n,B->cmap->n));
  PetscCall(MatSetType(b->A,MATSEQAIJCUSPARSE));
  PetscCall(PetscLogObjectParent((PetscObject)B,(PetscObject)b->A));
  PetscCall(MatCreate(PETSC_COMM_SELF,&b->B));
  PetscCall(MatSetSizes(b->B,B->rmap->n,noff,B->rmap->n,noff));
  PetscCall(MatSetType(b->B,MATSEQAIJCUSPARSE));
  PetscCall(PetscLogObjectParent((PetscObject)B,(PetscObject)b->B));

  /* GPU memory, cusparse specific call handles it internally */
  PetscCall(MatSetPreallocationCOO_SeqAIJCUSPARSE_Basic(b->A,cusp->coo_nd,d_i.data().get(),d_j.data().get()));
  PetscCall(MatSetPreallocationCOO_SeqAIJCUSPARSE_Basic(b->B,cusp->coo_no,d_i.data().get()+cusp->coo_nd,jj));
  PetscCall(PetscFree(jj));

  PetscCall(MatCUSPARSESetFormat(b->A,MAT_CUSPARSE_MULT,cusp->diagGPUMatFormat));
  PetscCall(MatCUSPARSESetFormat(b->B,MAT_CUSPARSE_MULT,cusp->offdiagGPUMatFormat));

  PetscCall(MatBindToCPU(b->A,B->boundtocpu));
  PetscCall(MatBindToCPU(b->B,B->boundtocpu));
  PetscCall(MatSetUpMultiply_MPIAIJ(B));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetPreallocationCOO_MPIAIJCUSPARSE(Mat mat, PetscCount coo_n, const PetscInt coo_i[], const PetscInt coo_j[])
{
  Mat_MPIAIJ         *mpiaij    = (Mat_MPIAIJ*)mat->data;
  Mat_MPIAIJCUSPARSE *mpidev;
  PetscBool           coo_basic = PETSC_TRUE;
  PetscMemType        mtype     = PETSC_MEMTYPE_DEVICE;
  PetscInt            rstart,rend;

  PetscFunctionBegin;
  PetscCall(PetscFree(mpiaij->garray));
  PetscCall(VecDestroy(&mpiaij->lvec));
#if defined(PETSC_USE_CTABLE)
  PetscCall(PetscTableDestroy(&mpiaij->colmap));
#else
  PetscCall(PetscFree(mpiaij->colmap));
#endif
  PetscCall(VecScatterDestroy(&mpiaij->Mvctx));
  mat->assembled                                                                      = PETSC_FALSE;
  mat->was_assembled                                                                  = PETSC_FALSE;
  PetscCall(MatResetPreallocationCOO_MPIAIJ(mat));
  PetscCall(MatResetPreallocationCOO_MPIAIJCUSPARSE(mat));
  if (coo_i) {
    PetscCall(PetscLayoutGetRange(mat->rmap,&rstart,&rend));
    PetscCall(PetscGetMemType(coo_i,&mtype));
    if (PetscMemTypeHost(mtype)) {
      for (PetscCount k=0; k<coo_n; k++) { /* Are there negative indices or remote entries? */
        if (coo_i[k]<0 || coo_i[k]<rstart || coo_i[k]>=rend || coo_j[k]<0) {coo_basic = PETSC_FALSE; break;}
      }
    }
  }
  /* All ranks must agree on the value of coo_basic */
  PetscCallMPI(MPI_Allreduce(MPI_IN_PLACE,&coo_basic,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)mat)));
  if (coo_basic) {
    PetscCall(MatSetPreallocationCOO_MPIAIJCUSPARSE_Basic(mat,coo_n,coo_i,coo_j));
  } else {
    PetscCall(MatSetPreallocationCOO_MPIAIJ(mat,coo_n,coo_i,coo_j));
    mat->offloadmask = PETSC_OFFLOAD_CPU;
    /* creates the GPU memory */
    PetscCall(MatSeqAIJCUSPARSECopyToGPU(mpiaij->A));
    PetscCall(MatSeqAIJCUSPARSECopyToGPU(mpiaij->B));
    mpidev = static_cast<Mat_MPIAIJCUSPARSE*>(mpiaij->spptr);
    mpidev->use_extended_coo = PETSC_TRUE;

    PetscCallCUDA(cudaMalloc((void**)&mpidev->Aimap1_d,mpiaij->Annz1*sizeof(PetscCount)));
    PetscCallCUDA(cudaMalloc((void**)&mpidev->Ajmap1_d,(mpiaij->Annz1+1)*sizeof(PetscCount)));
    PetscCallCUDA(cudaMalloc((void**)&mpidev->Aperm1_d,mpiaij->Atot1*sizeof(PetscCount)));

    PetscCallCUDA(cudaMalloc((void**)&mpidev->Bimap1_d,mpiaij->Bnnz1*sizeof(PetscCount)));
    PetscCallCUDA(cudaMalloc((void**)&mpidev->Bjmap1_d,(mpiaij->Bnnz1+1)*sizeof(PetscCount)));
    PetscCallCUDA(cudaMalloc((void**)&mpidev->Bperm1_d,mpiaij->Btot1*sizeof(PetscCount)));

    PetscCallCUDA(cudaMalloc((void**)&mpidev->Aimap2_d,mpiaij->Annz2*sizeof(PetscCount)));
    PetscCallCUDA(cudaMalloc((void**)&mpidev->Ajmap2_d,(mpiaij->Annz2+1)*sizeof(PetscCount)));
    PetscCallCUDA(cudaMalloc((void**)&mpidev->Aperm2_d,mpiaij->Atot2*sizeof(PetscCount)));

    PetscCallCUDA(cudaMalloc((void**)&mpidev->Bimap2_d,mpiaij->Bnnz2*sizeof(PetscCount)));
    PetscCallCUDA(cudaMalloc((void**)&mpidev->Bjmap2_d,(mpiaij->Bnnz2+1)*sizeof(PetscCount)));
    PetscCallCUDA(cudaMalloc((void**)&mpidev->Bperm2_d,mpiaij->Btot2*sizeof(PetscCount)));

    PetscCallCUDA(cudaMalloc((void**)&mpidev->Cperm1_d,mpiaij->sendlen*sizeof(PetscCount)));
    PetscCallCUDA(cudaMalloc((void**)&mpidev->sendbuf_d,mpiaij->sendlen*sizeof(PetscScalar)));
    PetscCallCUDA(cudaMalloc((void**)&mpidev->recvbuf_d,mpiaij->recvlen*sizeof(PetscScalar)));

    PetscCallCUDA(cudaMemcpy(mpidev->Aimap1_d,mpiaij->Aimap1,mpiaij->Annz1*sizeof(PetscCount),cudaMemcpyHostToDevice));
    PetscCallCUDA(cudaMemcpy(mpidev->Ajmap1_d,mpiaij->Ajmap1,(mpiaij->Annz1+1)*sizeof(PetscCount),cudaMemcpyHostToDevice));
    PetscCallCUDA(cudaMemcpy(mpidev->Aperm1_d,mpiaij->Aperm1,mpiaij->Atot1*sizeof(PetscCount),cudaMemcpyHostToDevice));

    PetscCallCUDA(cudaMemcpy(mpidev->Bimap1_d,mpiaij->Bimap1,mpiaij->Bnnz1*sizeof(PetscCount),cudaMemcpyHostToDevice));
    PetscCallCUDA(cudaMemcpy(mpidev->Bjmap1_d,mpiaij->Bjmap1,(mpiaij->Bnnz1+1)*sizeof(PetscCount),cudaMemcpyHostToDevice));
    PetscCallCUDA(cudaMemcpy(mpidev->Bperm1_d,mpiaij->Bperm1,mpiaij->Btot1*sizeof(PetscCount),cudaMemcpyHostToDevice));

    PetscCallCUDA(cudaMemcpy(mpidev->Aimap2_d,mpiaij->Aimap2,mpiaij->Annz2*sizeof(PetscCount),cudaMemcpyHostToDevice));
    PetscCallCUDA(cudaMemcpy(mpidev->Ajmap2_d,mpiaij->Ajmap2,(mpiaij->Annz2+1)*sizeof(PetscCount),cudaMemcpyHostToDevice));
    PetscCallCUDA(cudaMemcpy(mpidev->Aperm2_d,mpiaij->Aperm2,mpiaij->Atot2*sizeof(PetscCount),cudaMemcpyHostToDevice));

    PetscCallCUDA(cudaMemcpy(mpidev->Bimap2_d,mpiaij->Bimap2,mpiaij->Bnnz2*sizeof(PetscCount),cudaMemcpyHostToDevice));
    PetscCallCUDA(cudaMemcpy(mpidev->Bjmap2_d,mpiaij->Bjmap2,(mpiaij->Bnnz2+1)*sizeof(PetscCount),cudaMemcpyHostToDevice));
    PetscCallCUDA(cudaMemcpy(mpidev->Bperm2_d,mpiaij->Bperm2,mpiaij->Btot2*sizeof(PetscCount),cudaMemcpyHostToDevice));

    PetscCallCUDA(cudaMemcpy(mpidev->Cperm1_d,mpiaij->Cperm1,mpiaij->sendlen*sizeof(PetscCount),cudaMemcpyHostToDevice));
  }
  PetscFunctionReturn(0);
}

__global__ void MatPackCOOValues(const PetscScalar kv[],PetscCount nnz,const PetscCount perm[],PetscScalar buf[])
{
  PetscCount       i = blockIdx.x*blockDim.x + threadIdx.x;
  const PetscCount grid_size = gridDim.x * blockDim.x;
  for (; i<nnz; i+= grid_size) buf[i] = kv[perm[i]];
}

__global__ void MatAddCOOValues(const PetscScalar kv[],PetscCount nnz,const PetscCount imap[],const PetscCount jmap[],const PetscCount perm[],PetscScalar a[])
{
  PetscCount       i = blockIdx.x*blockDim.x + threadIdx.x;
  const PetscCount grid_size  = gridDim.x * blockDim.x;
  for (; i<nnz; i            += grid_size) {for (PetscCount k=jmap[i]; k<jmap[i+1]; k++) a[imap[i]] += kv[perm[k]];}
}

static PetscErrorCode MatSetValuesCOO_MPIAIJCUSPARSE(Mat mat,const PetscScalar v[],InsertMode imode)
{
  Mat_MPIAIJ         *mpiaij = static_cast<Mat_MPIAIJ*>(mat->data);
  Mat_MPIAIJCUSPARSE *mpidev = static_cast<Mat_MPIAIJCUSPARSE*>(mpiaij->spptr);
  Mat                 A      = mpiaij->A,B = mpiaij->B;
  PetscCount          Annz1  = mpiaij->Annz1,Annz2 = mpiaij->Annz2,Bnnz1 = mpiaij->Bnnz1,Bnnz2 = mpiaij->Bnnz2;
  PetscScalar        *Aa,*Ba = NULL;
  PetscScalar        *vsend  = mpidev->sendbuf_d,*v2 = mpidev->recvbuf_d;
  const PetscScalar  *v1     = v;
  const PetscCount   *Ajmap1 = mpidev->Ajmap1_d,*Ajmap2 = mpidev->Ajmap2_d,*Aimap1 = mpidev->Aimap1_d,*Aimap2 = mpidev->Aimap2_d;
  const PetscCount   *Bjmap1 = mpidev->Bjmap1_d,*Bjmap2 = mpidev->Bjmap2_d,*Bimap1 = mpidev->Bimap1_d,*Bimap2 = mpidev->Bimap2_d;
  const PetscCount   *Aperm1 = mpidev->Aperm1_d,*Aperm2 = mpidev->Aperm2_d,*Bperm1 = mpidev->Bperm1_d,*Bperm2 = mpidev->Bperm2_d;
  const PetscCount   *Cperm1 = mpidev->Cperm1_d;
  PetscMemType        memtype;

  PetscFunctionBegin;
  if (mpidev->use_extended_coo) {
    PetscMPIInt size;

    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)mat),&size));
    PetscCall(PetscGetMemType(v,&memtype));
    if (PetscMemTypeHost(memtype)) { /* If user gave v[] in host, we need to copy it to device */
      PetscCallCUDA(cudaMalloc((void**)&v1,mpiaij->coo_n*sizeof(PetscScalar)));
      PetscCallCUDA(cudaMemcpy((void*)v1,v,mpiaij->coo_n*sizeof(PetscScalar),cudaMemcpyHostToDevice));
    }

    if (imode == INSERT_VALUES) {
      PetscCall(MatSeqAIJCUSPARSEGetArrayWrite(A,&Aa)); /* write matrix values */
      PetscCallCUDA(cudaMemset(Aa,0,((Mat_SeqAIJ*)A->data)->nz*sizeof(PetscScalar)));
      if (size > 1) {
        PetscCall(MatSeqAIJCUSPARSEGetArrayWrite(B,&Ba));
        PetscCallCUDA(cudaMemset(Ba,0,((Mat_SeqAIJ*)B->data)->nz*sizeof(PetscScalar)));
      }
    } else {
      PetscCall(MatSeqAIJCUSPARSEGetArray(A,&Aa)); /* read & write matrix values */
      if (size > 1) PetscCall(MatSeqAIJCUSPARSEGetArray(B,&Ba));
    }

    /* Pack entries to be sent to remote */
    if (mpiaij->sendlen) {
      MatPackCOOValues<<<(mpiaij->sendlen+255)/256,256>>>(v1,mpiaij->sendlen,Cperm1,vsend);
      PetscCallCUDA(cudaPeekAtLastError());
    }

    /* Send remote entries to their owner and overlap the communication with local computation */
    if (size > 1) PetscCall(PetscSFReduceWithMemTypeBegin(mpiaij->coo_sf,MPIU_SCALAR,PETSC_MEMTYPE_CUDA,vsend,PETSC_MEMTYPE_CUDA,v2,MPI_REPLACE));
    /* Add local entries to A and B */
    if (Annz1) {
      MatAddCOOValues<<<(Annz1+255)/256,256>>>(v1,Annz1,Aimap1,Ajmap1,Aperm1,Aa);
      PetscCallCUDA(cudaPeekAtLastError());
    }
    if (Bnnz1) {
      MatAddCOOValues<<<(Bnnz1+255)/256,256>>>(v1,Bnnz1,Bimap1,Bjmap1,Bperm1,Ba);
      PetscCallCUDA(cudaPeekAtLastError());
    }
    if (size > 1) PetscCall(PetscSFReduceEnd(mpiaij->coo_sf,MPIU_SCALAR,vsend,v2,MPI_REPLACE));

    /* Add received remote entries to A and B */
    if (Annz2) {
      MatAddCOOValues<<<(Annz2+255)/256,256>>>(v2,Annz2,Aimap2,Ajmap2,Aperm2,Aa);
      PetscCallCUDA(cudaPeekAtLastError());
    }
    if (Bnnz2) {
      MatAddCOOValues<<<(Bnnz2+255)/256,256>>>(v2,Bnnz2,Bimap2,Bjmap2,Bperm2,Ba);
      PetscCallCUDA(cudaPeekAtLastError());
    }

    if (imode == INSERT_VALUES) {
      PetscCall(MatSeqAIJCUSPARSERestoreArrayWrite(A,&Aa));
      if (size > 1) PetscCall(MatSeqAIJCUSPARSERestoreArrayWrite(B,&Ba));
    } else {
      PetscCall(MatSeqAIJCUSPARSERestoreArray(A,&Aa));
      if (size > 1) PetscCall(MatSeqAIJCUSPARSERestoreArray(B,&Ba));
    }
    if (PetscMemTypeHost(memtype)) PetscCallCUDA(cudaFree((void*)v1));
  } else {
    PetscCall(MatSetValuesCOO_MPIAIJCUSPARSE_Basic(mat,v,imode));
  }
  mat->offloadmask = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMPIAIJGetLocalMatMerge_MPIAIJCUSPARSE(Mat A,MatReuse scall,IS *glob,Mat *A_loc)
{
  Mat             Ad,Ao;
  const PetscInt *cmap;

  PetscFunctionBegin;
  PetscCall(MatMPIAIJGetSeqAIJ(A,&Ad,&Ao,&cmap));
  PetscCall(MatSeqAIJCUSPARSEMergeMats(Ad,Ao,scall,A_loc));
  if (glob) {
    PetscInt cst, i, dn, on, *gidx;

    PetscCall(MatGetLocalSize(Ad,NULL,&dn));
    PetscCall(MatGetLocalSize(Ao,NULL,&on));
    PetscCall(MatGetOwnershipRangeColumn(A,&cst,NULL));
    PetscCall(PetscMalloc1(dn+on,&gidx));
    for (i = 0; i<dn; i++) gidx[i]    = cst + i;
    for (i = 0; i<on; i++) gidx[i+dn] = cmap[i];
    PetscCall(ISCreateGeneral(PetscObjectComm((PetscObject)Ad),dn+on,gidx,PETSC_OWN_POINTER,glob));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMPIAIJSetPreallocation_MPIAIJCUSPARSE(Mat B,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  Mat_MPIAIJ *b = (Mat_MPIAIJ*)B->data;
  Mat_MPIAIJCUSPARSE *cusparseStruct = (Mat_MPIAIJCUSPARSE*)b->spptr;
  PetscInt           i;

  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(B->rmap));
  PetscCall(PetscLayoutSetUp(B->cmap));
  if (PetscDefined(USE_DEBUG) && d_nnz) {
    for (i=0; i<B->rmap->n; i++) {
      PetscCheck(d_nnz[i] >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"d_nnz cannot be less than 0: local row %" PetscInt_FMT " value %" PetscInt_FMT,i,d_nnz[i]);
    }
  }
  if (PetscDefined(USE_DEBUG) && o_nnz) {
    for (i=0; i<B->rmap->n; i++) {
      PetscCheck(o_nnz[i] >= 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"o_nnz cannot be less than 0: local row %" PetscInt_FMT " value %" PetscInt_FMT,i,o_nnz[i]);
    }
  }
#if defined(PETSC_USE_CTABLE)
  PetscCall(PetscTableDestroy(&b->colmap));
#else
  PetscCall(PetscFree(b->colmap));
#endif
  PetscCall(PetscFree(b->garray));
  PetscCall(VecDestroy(&b->lvec));
  PetscCall(VecScatterDestroy(&b->Mvctx));
  /* Because the B will have been resized we simply destroy it and create a new one each time */
  PetscCall(MatDestroy(&b->B));
  if (!b->A) {
    PetscCall(MatCreate(PETSC_COMM_SELF,&b->A));
    PetscCall(MatSetSizes(b->A,B->rmap->n,B->cmap->n,B->rmap->n,B->cmap->n));
    PetscCall(PetscLogObjectParent((PetscObject)B,(PetscObject)b->A));
  }
  if (!b->B) {
    PetscMPIInt size;
    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)B),&size));
    PetscCall(MatCreate(PETSC_COMM_SELF,&b->B));
    PetscCall(MatSetSizes(b->B,B->rmap->n,size > 1 ? B->cmap->N : 0,B->rmap->n,size > 1 ? B->cmap->N : 0));
    PetscCall(PetscLogObjectParent((PetscObject)B,(PetscObject)b->B));
  }
  PetscCall(MatSetType(b->A,MATSEQAIJCUSPARSE));
  PetscCall(MatSetType(b->B,MATSEQAIJCUSPARSE));
  PetscCall(MatBindToCPU(b->A,B->boundtocpu));
  PetscCall(MatBindToCPU(b->B,B->boundtocpu));
  PetscCall(MatSeqAIJSetPreallocation(b->A,d_nz,d_nnz));
  PetscCall(MatSeqAIJSetPreallocation(b->B,o_nz,o_nnz));
  PetscCall(MatCUSPARSESetFormat(b->A,MAT_CUSPARSE_MULT,cusparseStruct->diagGPUMatFormat));
  PetscCall(MatCUSPARSESetFormat(b->B,MAT_CUSPARSE_MULT,cusparseStruct->offdiagGPUMatFormat));
  B->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_MPIAIJCUSPARSE(Mat A,Vec xx,Vec yy)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ*)A->data;

  PetscFunctionBegin;
  PetscCall(VecScatterBegin(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall((*a->A->ops->mult)(a->A,xx,yy));
  PetscCall(VecScatterEnd(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall((*a->B->ops->multadd)(a->B,a->lvec,yy,yy));
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroEntries_MPIAIJCUSPARSE(Mat A)
{
  Mat_MPIAIJ     *l = (Mat_MPIAIJ*)A->data;

  PetscFunctionBegin;
  PetscCall(MatZeroEntries(l->A));
  PetscCall(MatZeroEntries(l->B));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_MPIAIJCUSPARSE(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ*)A->data;

  PetscFunctionBegin;
  PetscCall(VecScatterBegin(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall((*a->A->ops->multadd)(a->A,xx,yy,zz));
  PetscCall(VecScatterEnd(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall((*a->B->ops->multadd)(a->B,a->lvec,zz,zz));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_MPIAIJCUSPARSE(Mat A,Vec xx,Vec yy)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ*)A->data;

  PetscFunctionBegin;
  PetscCall((*a->B->ops->multtranspose)(a->B,xx,a->lvec));
  PetscCall((*a->A->ops->multtranspose)(a->A,xx,yy));
  PetscCall(VecScatterBegin(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE));
  PetscCall(VecScatterEnd(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE));
  PetscFunctionReturn(0);
}

PetscErrorCode MatCUSPARSESetFormat_MPIAIJCUSPARSE(Mat A,MatCUSPARSEFormatOperation op,MatCUSPARSEStorageFormat format)
{
  Mat_MPIAIJ         *a               = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJCUSPARSE * cusparseStruct = (Mat_MPIAIJCUSPARSE*)a->spptr;

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
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"unsupported operation %d for MatCUSPARSEFormatOperation. Only MAT_CUSPARSE_MULT_DIAG, MAT_CUSPARSE_MULT_DIAG, and MAT_CUSPARSE_MULT_ALL are currently supported.",op);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatSetFromOptions_MPIAIJCUSPARSE(PetscOptionItems *PetscOptionsObject,Mat A)
{
  MatCUSPARSEStorageFormat format;
  PetscBool                flg;
  Mat_MPIAIJ               *a = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJCUSPARSE       *cusparseStruct = (Mat_MPIAIJCUSPARSE*)a->spptr;

  PetscFunctionBegin;
  PetscCall(PetscOptionsHead(PetscOptionsObject,"MPIAIJCUSPARSE options"));
  if (A->factortype==MAT_FACTOR_NONE) {
    PetscCall(PetscOptionsEnum("-mat_cusparse_mult_diag_storage_format","sets storage format of the diagonal blocks of (mpi)aijcusparse gpu matrices for SpMV",
                             "MatCUSPARSESetFormat",MatCUSPARSEStorageFormats,(PetscEnum)cusparseStruct->diagGPUMatFormat,(PetscEnum*)&format,&flg));
    if (flg) PetscCall(MatCUSPARSESetFormat(A,MAT_CUSPARSE_MULT_DIAG,format));
    PetscCall(PetscOptionsEnum("-mat_cusparse_mult_offdiag_storage_format","sets storage format of the off-diagonal blocks (mpi)aijcusparse gpu matrices for SpMV",
                             "MatCUSPARSESetFormat",MatCUSPARSEStorageFormats,(PetscEnum)cusparseStruct->offdiagGPUMatFormat,(PetscEnum*)&format,&flg));
    if (flg) PetscCall(MatCUSPARSESetFormat(A,MAT_CUSPARSE_MULT_OFFDIAG,format));
    PetscCall(PetscOptionsEnum("-mat_cusparse_storage_format","sets storage format of the diagonal and off-diagonal blocks (mpi)aijcusparse gpu matrices for SpMV",
                             "MatCUSPARSESetFormat",MatCUSPARSEStorageFormats,(PetscEnum)cusparseStruct->diagGPUMatFormat,(PetscEnum*)&format,&flg));
    if (flg) PetscCall(MatCUSPARSESetFormat(A,MAT_CUSPARSE_ALL,format));
  }
  PetscCall(PetscOptionsTail());
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_MPIAIJCUSPARSE(Mat A,MatAssemblyType mode)
{
  Mat_MPIAIJ         *mpiaij = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJCUSPARSE *cusp = (Mat_MPIAIJCUSPARSE*)mpiaij->spptr;
  PetscObjectState   onnz = A->nonzerostate;

  PetscFunctionBegin;
  PetscCall(MatAssemblyEnd_MPIAIJ(A,mode));
  if (mpiaij->lvec) PetscCall(VecSetType(mpiaij->lvec,VECSEQCUDA));
  if (onnz != A->nonzerostate && cusp->deviceMat) {
    PetscSplitCSRDataStructure d_mat = cusp->deviceMat, h_mat;

    PetscCall(PetscInfo(A,"Destroy device mat since nonzerostate changed\n"));
    PetscCall(PetscNew(&h_mat));
    PetscCallCUDA(cudaMemcpy(h_mat,d_mat,sizeof(*d_mat),cudaMemcpyDeviceToHost));
    PetscCallCUDA(cudaFree(h_mat->colmap));
    if (h_mat->allocated_indices) {
      PetscCallCUDA(cudaFree(h_mat->diag.i));
      PetscCallCUDA(cudaFree(h_mat->diag.j));
      if (h_mat->offdiag.j) {
        PetscCallCUDA(cudaFree(h_mat->offdiag.i));
        PetscCallCUDA(cudaFree(h_mat->offdiag.j));
      }
    }
    PetscCallCUDA(cudaFree(d_mat));
    PetscCall(PetscFree(h_mat));
    cusp->deviceMat = NULL;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MPIAIJCUSPARSE(Mat A)
{
  Mat_MPIAIJ         *aij            = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJCUSPARSE *cusparseStruct = (Mat_MPIAIJCUSPARSE*)aij->spptr;

  PetscFunctionBegin;
  PetscCheck(cusparseStruct,PETSC_COMM_SELF,PETSC_ERR_COR,"Missing spptr");
  if (cusparseStruct->deviceMat) {
    PetscSplitCSRDataStructure d_mat = cusparseStruct->deviceMat, h_mat;

    PetscCall(PetscInfo(A,"Have device matrix\n"));
    PetscCall(PetscNew(&h_mat));
    PetscCallCUDA(cudaMemcpy(h_mat,d_mat,sizeof(*d_mat),cudaMemcpyDeviceToHost));
    PetscCallCUDA(cudaFree(h_mat->colmap));
    if (h_mat->allocated_indices) {
      PetscCallCUDA(cudaFree(h_mat->diag.i));
      PetscCallCUDA(cudaFree(h_mat->diag.j));
      if (h_mat->offdiag.j) {
        PetscCallCUDA(cudaFree(h_mat->offdiag.i));
        PetscCallCUDA(cudaFree(h_mat->offdiag.j));
      }
    }
    PetscCallCUDA(cudaFree(d_mat));
    PetscCall(PetscFree(h_mat));
  }
  /* Free COO */
  PetscCall(MatResetPreallocationCOO_MPIAIJCUSPARSE(A));
  PetscCallCXX(delete cusparseStruct);
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJGetLocalMatMerge_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOO_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatCUSPARSESetFormat_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_mpiaijcusparse_hypre_C",NULL));
  PetscCall(MatDestroy_MPIAIJ(A));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIAIJCUSPARSE(Mat B, MatType mtype, MatReuse reuse, Mat* newmat)
{
  Mat_MPIAIJ *a;
  Mat         A;

  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
  if (reuse == MAT_INITIAL_MATRIX) PetscCall(MatDuplicate(B,MAT_COPY_VALUES,newmat));
  else if (reuse == MAT_REUSE_MATRIX) PetscCall(MatCopy(B,*newmat,SAME_NONZERO_PATTERN));
  A = *newmat;
  A->boundtocpu = PETSC_FALSE;
  PetscCall(PetscFree(A->defaultvectype));
  PetscCall(PetscStrallocpy(VECCUDA,&A->defaultvectype));

  a = (Mat_MPIAIJ*)A->data;
  if (a->A) PetscCall(MatSetType(a->A,MATSEQAIJCUSPARSE));
  if (a->B) PetscCall(MatSetType(a->B,MATSEQAIJCUSPARSE));
  if (a->lvec) PetscCall(VecSetType(a->lvec,VECSEQCUDA));

  if (reuse != MAT_REUSE_MATRIX && !a->spptr) {
    PetscCallCXX(a->spptr = new Mat_MPIAIJCUSPARSE);
  }

  A->ops->assemblyend           = MatAssemblyEnd_MPIAIJCUSPARSE;
  A->ops->mult                  = MatMult_MPIAIJCUSPARSE;
  A->ops->multadd               = MatMultAdd_MPIAIJCUSPARSE;
  A->ops->multtranspose         = MatMultTranspose_MPIAIJCUSPARSE;
  A->ops->setfromoptions        = MatSetFromOptions_MPIAIJCUSPARSE;
  A->ops->destroy               = MatDestroy_MPIAIJCUSPARSE;
  A->ops->zeroentries           = MatZeroEntries_MPIAIJCUSPARSE;
  A->ops->productsetfromoptions = MatProductSetFromOptions_MPIAIJBACKEND;

  PetscCall(PetscObjectChangeTypeName((PetscObject)A,MATMPIAIJCUSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJGetLocalMatMerge_C",MatMPIAIJGetLocalMatMerge_MPIAIJCUSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",MatMPIAIJSetPreallocation_MPIAIJCUSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatCUSPARSESetFormat_C",MatCUSPARSESetFormat_MPIAIJCUSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOO_C",MatSetPreallocationCOO_MPIAIJCUSPARSE));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",MatSetValuesCOO_MPIAIJCUSPARSE));
#if defined(PETSC_HAVE_HYPRE)
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatConvert_mpiaijcusparse_hypre_C",MatConvert_AIJ_HYPRE));
#endif
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJCUSPARSE(Mat A)
{
  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
  PetscCall(MatCreate_MPIAIJ(A));
  PetscCall(MatConvert_MPIAIJ_MPIAIJCUSPARSE(A,MATMPIAIJCUSPARSE,MAT_INPLACE_MATRIX,&A));
  PetscFunctionReturn(0);
}

/*@
   MatCreateAIJCUSPARSE - Creates a sparse matrix in AIJ (compressed row) format
   (the default parallel PETSc format).  This matrix will ultimately pushed down
   to NVidia GPUs and use the CUSPARSE library for calculations. For good matrix
   assembly performance the user should preallocate the matrix storage by setting
   the parameter nz (or the array nnz).  By setting these parameters accurately,
   performance during matrix assembly can be increased by more than a factor of 50.

   Collective

   Input Parameters:
+  comm - MPI communicator, set to PETSC_COMM_SELF
.  m - number of rows
.  n - number of columns
.  nz - number of nonzeros per row (same for all rows)
-  nnz - array containing the number of nonzeros in the various rows
         (possibly different for each row) or NULL

   Output Parameter:
.  A - the matrix

   It is recommended that one use the MatCreate(), MatSetType() and/or MatSetFromOptions(),
   MatXXXXSetPreallocation() paradigm instead of this routine directly.
   [MatXXXXSetPreallocation() is, for example, MatSeqAIJSetPreallocation]

   Notes:
   If nnz is given then nz is ignored

   The AIJ format (also called the Yale sparse matrix format or
   compressed row storage), is fully compatible with standard Fortran 77
   storage.  That is, the stored row and column indices can begin at
   either one (as in Fortran) or zero.  See the users' manual for details.

   Specify the preallocated storage with either nz or nnz (not both).
   Set nz=PETSC_DEFAULT and nnz=NULL for PETSc to control dynamic memory
   allocation.  For large problems you MUST preallocate memory or you
   will get TERRIBLE performance, see the users' manual chapter on matrices.

   By default, this format uses inodes (identical nodes) when possible, to
   improve numerical efficiency of matrix-vector products and solves. We
   search for consecutive rows with the same nonzero structure, thereby
   reusing matrix information to achieve increased efficiency.

   Level: intermediate

.seealso: MatCreate(), MatCreateAIJ(), MatSetValues(), MatSeqAIJSetColumnIndices(), MatCreateSeqAIJWithArrays(), MatCreateAIJ(), MATMPIAIJCUSPARSE, MATAIJCUSPARSE
@*/
PetscErrorCode  MatCreateAIJCUSPARSE(MPI_Comm comm,PetscInt m,PetscInt n,PetscInt M,PetscInt N,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[],Mat *A)
{
  PetscMPIInt size;

  PetscFunctionBegin;
  PetscCall(MatCreate(comm,A));
  PetscCall(MatSetSizes(*A,m,n,M,N));
  PetscCallMPI(MPI_Comm_size(comm,&size));
  if (size > 1) {
    PetscCall(MatSetType(*A,MATMPIAIJCUSPARSE));
    PetscCall(MatMPIAIJSetPreallocation(*A,d_nz,d_nnz,o_nz,o_nnz));
  } else {
    PetscCall(MatSetType(*A,MATSEQAIJCUSPARSE));
    PetscCall(MatSeqAIJSetPreallocation(*A,d_nz,d_nnz));
  }
  PetscFunctionReturn(0);
}

/*MC
   MATAIJCUSPARSE - A matrix type to be used for sparse matrices; it is as same as MATMPIAIJCUSPARSE.

   A matrix type type whose data resides on Nvidia GPUs. These matrices can be in either
   CSR, ELL, or Hybrid format. The ELL and HYB formats require CUDA 4.2 or later.
   All matrix calculations are performed on Nvidia GPUs using the CUSPARSE library.

   This matrix type is identical to MATSEQAIJCUSPARSE when constructed with a single process communicator,
   and MATMPIAIJCUSPARSE otherwise.  As a result, for single process communicators,
   MatSeqAIJSetPreallocation is supported, and similarly MatMPIAIJSetPreallocation is supported
   for communicators controlling multiple processes.  It is recommended that you call both of
   the above preallocation routines for simplicity.

   Options Database Keys:
+  -mat_type mpiaijcusparse - sets the matrix type to "mpiaijcusparse" during a call to MatSetFromOptions()
.  -mat_cusparse_storage_format csr - sets the storage format of diagonal and off-diagonal matrices during a call to MatSetFromOptions(). Other options include ell (ellpack) or hyb (hybrid).
.  -mat_cusparse_mult_diag_storage_format csr - sets the storage format of diagonal matrix during a call to MatSetFromOptions(). Other options include ell (ellpack) or hyb (hybrid).
-  -mat_cusparse_mult_offdiag_storage_format csr - sets the storage format of off-diagonal matrix during a call to MatSetFromOptions(). Other options include ell (ellpack) or hyb (hybrid).

  Level: beginner

 .seealso: MatCreateAIJCUSPARSE(), MATSEQAIJCUSPARSE, MATMPIAIJCUSPARSE, MatCreateSeqAIJCUSPARSE(), MatCUSPARSESetFormat(), MatCUSPARSEStorageFormat, MatCUSPARSEFormatOperation
M*/

/*MC
   MATMPIAIJCUSPARSE - A matrix type to be used for sparse matrices; it is as same as MATAIJCUSPARSE.

  Level: beginner

 .seealso: MATAIJCUSPARSE, MATSEQAIJCUSPARSE
M*/

// get GPU pointers to stripped down Mat. For both seq and MPI Mat.
PetscErrorCode MatCUSPARSEGetDeviceMatWrite(Mat A, PetscSplitCSRDataStructure *B)
{
  PetscSplitCSRDataStructure d_mat;
  PetscMPIInt                size;
  int                        *ai = NULL,*bi = NULL,*aj = NULL,*bj = NULL;
  PetscScalar                *aa = NULL,*ba = NULL;
  Mat_SeqAIJ                 *jaca = NULL, *jacb = NULL;
  Mat_SeqAIJCUSPARSE         *cusparsestructA = NULL;
  CsrMatrix                  *matrixA = NULL,*matrixB = NULL;

  PetscFunctionBegin;
  PetscCheck(A->assembled,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Need already assembled matrix");
  if (A->factortype != MAT_FACTOR_NONE) {
    *B = NULL;
    PetscFunctionReturn(0);
  }
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  // get jaca
  if (size == 1) {
    PetscBool isseqaij;

    PetscCall(PetscObjectBaseTypeCompare((PetscObject)A,MATSEQAIJ,&isseqaij));
    if (isseqaij) {
      jaca = (Mat_SeqAIJ*)A->data;
      PetscCheck(jaca->roworiented,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Device assembly does not currently support column oriented values insertion");
      cusparsestructA = (Mat_SeqAIJCUSPARSE*)A->spptr;
      d_mat = cusparsestructA->deviceMat;
      PetscCall(MatSeqAIJCUSPARSECopyToGPU(A));
    } else {
      Mat_MPIAIJ *aij = (Mat_MPIAIJ*)A->data;
      PetscCheck(aij->roworiented,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Device assembly does not currently support column oriented values insertion");
      Mat_MPIAIJCUSPARSE *spptr = (Mat_MPIAIJCUSPARSE*)aij->spptr;
      jaca = (Mat_SeqAIJ*)aij->A->data;
      cusparsestructA = (Mat_SeqAIJCUSPARSE*)aij->A->spptr;
      d_mat = spptr->deviceMat;
      PetscCall(MatSeqAIJCUSPARSECopyToGPU(aij->A));
    }
    if (cusparsestructA->format==MAT_CUSPARSE_CSR) {
      Mat_SeqAIJCUSPARSEMultStruct *matstruct = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestructA->mat;
      PetscCheck(matstruct,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing Mat_SeqAIJCUSPARSEMultStruct for A");
      matrixA = (CsrMatrix*)matstruct->mat;
      bi = NULL;
      bj = NULL;
      ba = NULL;
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Device Mat needs MAT_CUSPARSE_CSR");
  } else {
    Mat_MPIAIJ *aij = (Mat_MPIAIJ*)A->data;
    PetscCheck(aij->roworiented,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Device assembly does not currently support column oriented values insertion");
    jaca = (Mat_SeqAIJ*)aij->A->data;
    jacb = (Mat_SeqAIJ*)aij->B->data;
    Mat_MPIAIJCUSPARSE *spptr = (Mat_MPIAIJCUSPARSE*)aij->spptr;

    PetscCheck(A->nooffprocentries || aij->donotstash,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Device assembly does not currently support offproc values insertion. Use MatSetOption(A,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE) or MatSetOption(A,MAT_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE)");
    cusparsestructA = (Mat_SeqAIJCUSPARSE*)aij->A->spptr;
    Mat_SeqAIJCUSPARSE *cusparsestructB = (Mat_SeqAIJCUSPARSE*)aij->B->spptr;
    PetscCheck(cusparsestructA->format==MAT_CUSPARSE_CSR,PETSC_COMM_SELF,PETSC_ERR_SUP,"Device Mat A needs MAT_CUSPARSE_CSR");
    if (cusparsestructB->format==MAT_CUSPARSE_CSR) {
      PetscCall(MatSeqAIJCUSPARSECopyToGPU(aij->A));
      PetscCall(MatSeqAIJCUSPARSECopyToGPU(aij->B));
      Mat_SeqAIJCUSPARSEMultStruct *matstructA = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestructA->mat;
      Mat_SeqAIJCUSPARSEMultStruct *matstructB = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestructB->mat;
      PetscCheck(matstructA,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing Mat_SeqAIJCUSPARSEMultStruct for A");
      PetscCheck(matstructB,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing Mat_SeqAIJCUSPARSEMultStruct for B");
      matrixA = (CsrMatrix*)matstructA->mat;
      matrixB = (CsrMatrix*)matstructB->mat;
      if (jacb->compressedrow.use) {
        if (!cusparsestructB->rowoffsets_gpu) {
          cusparsestructB->rowoffsets_gpu = new THRUSTINTARRAY32(A->rmap->n+1);
          cusparsestructB->rowoffsets_gpu->assign(jacb->i,jacb->i+A->rmap->n+1);
        }
        bi = thrust::raw_pointer_cast(cusparsestructB->rowoffsets_gpu->data());
      } else {
        bi = thrust::raw_pointer_cast(matrixB->row_offsets->data());
      }
      bj = thrust::raw_pointer_cast(matrixB->column_indices->data());
      ba = thrust::raw_pointer_cast(matrixB->values->data());
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Device Mat B needs MAT_CUSPARSE_CSR");
    d_mat = spptr->deviceMat;
  }
  if (jaca->compressedrow.use) {
    if (!cusparsestructA->rowoffsets_gpu) {
      cusparsestructA->rowoffsets_gpu = new THRUSTINTARRAY32(A->rmap->n+1);
      cusparsestructA->rowoffsets_gpu->assign(jaca->i,jaca->i+A->rmap->n+1);
    }
    ai = thrust::raw_pointer_cast(cusparsestructA->rowoffsets_gpu->data());
  } else {
    ai = thrust::raw_pointer_cast(matrixA->row_offsets->data());
  }
  aj = thrust::raw_pointer_cast(matrixA->column_indices->data());
  aa = thrust::raw_pointer_cast(matrixA->values->data());

  if (!d_mat) {
    PetscSplitCSRDataStructure h_mat;

    // create and populate strucy on host and copy on device
    PetscCall(PetscInfo(A,"Create device matrix\n"));
    PetscCall(PetscNew(&h_mat));
    PetscCallCUDA(cudaMalloc((void**)&d_mat,sizeof(*d_mat)));
    if (size > 1) { /* need the colmap array */
      Mat_MPIAIJ *aij = (Mat_MPIAIJ*)A->data;
      PetscInt   *colmap;
      PetscInt   ii,n = aij->B->cmap->n,N = A->cmap->N;

      PetscCheck(!n || aij->garray,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MPIAIJ Matrix was assembled but is missing garray");

      PetscCall(PetscCalloc1(N+1,&colmap));
      for (ii=0; ii<n; ii++) colmap[aij->garray[ii]] = (int)(ii+1);
#if defined(PETSC_USE_64BIT_INDICES)
      { // have to make a long version of these
        int        *h_bi32, *h_bj32;
        PetscInt   *h_bi64, *h_bj64, *d_bi64, *d_bj64;
        PetscCall(PetscCalloc4(A->rmap->n+1,&h_bi32,jacb->nz,&h_bj32,A->rmap->n+1,&h_bi64,jacb->nz,&h_bj64));
        PetscCallCUDA(cudaMemcpy(h_bi32, bi, (A->rmap->n+1)*sizeof(*h_bi32),cudaMemcpyDeviceToHost));
        for (int i=0;i<A->rmap->n+1;i++) h_bi64[i] = h_bi32[i];
        PetscCallCUDA(cudaMemcpy(h_bj32, bj, jacb->nz*sizeof(*h_bj32),cudaMemcpyDeviceToHost));
        for (int i=0;i<jacb->nz;i++) h_bj64[i] = h_bj32[i];

        PetscCallCUDA(cudaMalloc((void**)&d_bi64,(A->rmap->n+1)*sizeof(*d_bi64)));
        PetscCallCUDA(cudaMemcpy(d_bi64, h_bi64,(A->rmap->n+1)*sizeof(*d_bi64),cudaMemcpyHostToDevice));
        PetscCallCUDA(cudaMalloc((void**)&d_bj64,jacb->nz*sizeof(*d_bj64)));
        PetscCallCUDA(cudaMemcpy(d_bj64, h_bj64,jacb->nz*sizeof(*d_bj64),cudaMemcpyHostToDevice));

        h_mat->offdiag.i = d_bi64;
        h_mat->offdiag.j = d_bj64;
        h_mat->allocated_indices = PETSC_TRUE;

        PetscCall(PetscFree4(h_bi32,h_bj32,h_bi64,h_bj64));
      }
#else
      h_mat->offdiag.i = (PetscInt*)bi;
      h_mat->offdiag.j = (PetscInt*)bj;
      h_mat->allocated_indices = PETSC_FALSE;
#endif
      h_mat->offdiag.a = ba;
      h_mat->offdiag.n = A->rmap->n;

      PetscCallCUDA(cudaMalloc((void**)&h_mat->colmap,(N+1)*sizeof(*h_mat->colmap)));
      PetscCallCUDA(cudaMemcpy(h_mat->colmap,colmap,(N+1)*sizeof(*h_mat->colmap),cudaMemcpyHostToDevice));
      PetscCall(PetscFree(colmap));
    }
    h_mat->rstart = A->rmap->rstart;
    h_mat->rend   = A->rmap->rend;
    h_mat->cstart = A->cmap->rstart;
    h_mat->cend   = A->cmap->rend;
    h_mat->M      = A->cmap->N;
#if defined(PETSC_USE_64BIT_INDICES)
    {
      PetscCheck(sizeof(PetscInt) == 8,PETSC_COMM_SELF,PETSC_ERR_PLIB,"size pof PetscInt = %d",sizeof(PetscInt));
      int        *h_ai32, *h_aj32;
      PetscInt   *h_ai64, *h_aj64, *d_ai64, *d_aj64;
      PetscCall(PetscCalloc4(A->rmap->n+1,&h_ai32,jaca->nz,&h_aj32,A->rmap->n+1,&h_ai64,jaca->nz,&h_aj64));
      PetscCallCUDA(cudaMemcpy(h_ai32, ai, (A->rmap->n+1)*sizeof(*h_ai32),cudaMemcpyDeviceToHost));
      for (int i=0;i<A->rmap->n+1;i++) h_ai64[i] = h_ai32[i];
      PetscCallCUDA(cudaMemcpy(h_aj32, aj, jaca->nz*sizeof(*h_aj32),cudaMemcpyDeviceToHost));
      for (int i=0;i<jaca->nz;i++) h_aj64[i] = h_aj32[i];

      PetscCallCUDA(cudaMalloc((void**)&d_ai64,(A->rmap->n+1)*sizeof(*d_ai64)));
      PetscCallCUDA(cudaMemcpy(d_ai64, h_ai64,(A->rmap->n+1)*sizeof(*d_ai64),cudaMemcpyHostToDevice));
      PetscCallCUDA(cudaMalloc((void**)&d_aj64,jaca->nz*sizeof(*d_aj64)));
      PetscCallCUDA(cudaMemcpy(d_aj64, h_aj64,jaca->nz*sizeof(*d_aj64),cudaMemcpyHostToDevice));

      h_mat->diag.i = d_ai64;
      h_mat->diag.j = d_aj64;
      h_mat->allocated_indices = PETSC_TRUE;

      PetscCall(PetscFree4(h_ai32,h_aj32,h_ai64,h_aj64));
    }
#else
    h_mat->diag.i = (PetscInt*)ai;
    h_mat->diag.j = (PetscInt*)aj;
    h_mat->allocated_indices = PETSC_FALSE;
#endif
    h_mat->diag.a = aa;
    h_mat->diag.n = A->rmap->n;
    h_mat->rank   = PetscGlobalRank;
    // copy pointers and metadata to device
    PetscCallCUDA(cudaMemcpy(d_mat,h_mat,sizeof(*d_mat),cudaMemcpyHostToDevice));
    PetscCall(PetscFree(h_mat));
  } else {
    PetscCall(PetscInfo(A,"Reusing device matrix\n"));
  }
  *B = d_mat;
  A->offloadmask = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(0);
}
