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
    CHKERRCUDA(cudaFree(cusparseStruct->Aimap1_d));
    CHKERRCUDA(cudaFree(cusparseStruct->Ajmap1_d));
    CHKERRCUDA(cudaFree(cusparseStruct->Aperm1_d));
    CHKERRCUDA(cudaFree(cusparseStruct->Bimap1_d));
    CHKERRCUDA(cudaFree(cusparseStruct->Bjmap1_d));
    CHKERRCUDA(cudaFree(cusparseStruct->Bperm1_d));
    CHKERRCUDA(cudaFree(cusparseStruct->Aimap2_d));
    CHKERRCUDA(cudaFree(cusparseStruct->Ajmap2_d));
    CHKERRCUDA(cudaFree(cusparseStruct->Aperm2_d));
    CHKERRCUDA(cudaFree(cusparseStruct->Bimap2_d));
    CHKERRCUDA(cudaFree(cusparseStruct->Bjmap2_d));
    CHKERRCUDA(cudaFree(cusparseStruct->Bperm2_d));
    CHKERRCUDA(cudaFree(cusparseStruct->Cperm1_d));
    CHKERRCUDA(cudaFree(cusparseStruct->sendbuf_d));
    CHKERRCUDA(cudaFree(cusparseStruct->recvbuf_d));
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
      CHKERRQ(PetscLogCpuToGpu(n*sizeof(PetscScalar)));
      d_v = w->data();
    }

    auto zibit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(d_v,cusp->coo_p->begin()),
                                                              cusp->coo_pw->begin()));
    auto zieit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(d_v,cusp->coo_p->end()),
                                                              cusp->coo_pw->end()));
    CHKERRQ(PetscLogGpuTimeBegin());
    thrust::for_each(zibit,zieit,VecCUDAEquals());
    CHKERRQ(PetscLogGpuTimeEnd());
    delete w;
    CHKERRQ(MatSetValuesCOO_SeqAIJCUSPARSE_Basic(a->A,cusp->coo_pw->data().get(),imode));
    CHKERRQ(MatSetValuesCOO_SeqAIJCUSPARSE_Basic(a->B,cusp->coo_pw->data().get()+cusp->coo_nd,imode));
  } else {
    CHKERRQ(MatSetValuesCOO_SeqAIJCUSPARSE_Basic(a->A,v,imode));
    CHKERRQ(MatSetValuesCOO_SeqAIJCUSPARSE_Basic(a->B,v ? v+cusp->coo_nd : NULL,imode));
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
  if (b->A) CHKERRQ(MatCUSPARSEClearHandle(b->A));
  if (b->B) CHKERRQ(MatCUSPARSEClearHandle(b->B));
  CHKERRQ(MatDestroy(&b->A));
  CHKERRQ(MatDestroy(&b->B));

  CHKERRQ(PetscLogCpuToGpu(2.*n*sizeof(PetscInt)));
  d_i.assign(coo_i,coo_i+n);
  d_j.assign(coo_j,coo_j+n);
  CHKERRQ(PetscLogGpuTimeBegin());
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
  CHKERRQ(PetscLogGpuTimeEnd());

  /* copy offdiag column indices to map on the CPU */
  CHKERRQ(PetscMalloc1(cusp->coo_no,&jj)); /* jj[] will store compacted col ids of the offdiag part */
  CHKERRCUDA(cudaMemcpy(jj,d_j.data().get()+cusp->coo_nd,cusp->coo_no*sizeof(PetscInt),cudaMemcpyDeviceToHost));
  auto o_j = d_j.begin();
  CHKERRQ(PetscLogGpuTimeBegin());
  thrust::advance(o_j,cusp->coo_nd); /* sort and unique offdiag col ids */
  thrust::sort(thrust::device,o_j,d_j.end());
  auto wit = thrust::unique(thrust::device,o_j,d_j.end()); /* return end iter of the unique range */
  CHKERRQ(PetscLogGpuTimeEnd());
  noff = thrust::distance(o_j,wit);
  /* allocate the garray, the columns of B will not be mapped in the multiply setup */
  CHKERRQ(PetscMalloc1(noff,&b->garray));
  CHKERRCUDA(cudaMemcpy(b->garray,d_j.data().get()+cusp->coo_nd,noff*sizeof(PetscInt),cudaMemcpyDeviceToHost));
  CHKERRQ(PetscLogGpuToCpu((noff+cusp->coo_no)*sizeof(PetscInt)));
  CHKERRQ(ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,1,noff,b->garray,PETSC_COPY_VALUES,&l2g));
  CHKERRQ(ISLocalToGlobalMappingSetType(l2g,ISLOCALTOGLOBALMAPPINGHASH));
  CHKERRQ(ISGlobalToLocalMappingApply(l2g,IS_GTOLM_DROP,cusp->coo_no,jj,&N,jj));
  PetscCheck(N == cusp->coo_no,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected is size %" PetscInt_FMT " != %" PetscInt_FMT " coo size",N,cusp->coo_no);
  CHKERRQ(ISLocalToGlobalMappingDestroy(&l2g));

  CHKERRQ(MatCreate(PETSC_COMM_SELF,&b->A));
  CHKERRQ(MatSetSizes(b->A,B->rmap->n,B->cmap->n,B->rmap->n,B->cmap->n));
  CHKERRQ(MatSetType(b->A,MATSEQAIJCUSPARSE));
  CHKERRQ(PetscLogObjectParent((PetscObject)B,(PetscObject)b->A));
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&b->B));
  CHKERRQ(MatSetSizes(b->B,B->rmap->n,noff,B->rmap->n,noff));
  CHKERRQ(MatSetType(b->B,MATSEQAIJCUSPARSE));
  CHKERRQ(PetscLogObjectParent((PetscObject)B,(PetscObject)b->B));

  /* GPU memory, cusparse specific call handles it internally */
  CHKERRQ(MatSetPreallocationCOO_SeqAIJCUSPARSE_Basic(b->A,cusp->coo_nd,d_i.data().get(),d_j.data().get()));
  CHKERRQ(MatSetPreallocationCOO_SeqAIJCUSPARSE_Basic(b->B,cusp->coo_no,d_i.data().get()+cusp->coo_nd,jj));
  CHKERRQ(PetscFree(jj));

  CHKERRQ(MatCUSPARSESetFormat(b->A,MAT_CUSPARSE_MULT,cusp->diagGPUMatFormat));
  CHKERRQ(MatCUSPARSESetFormat(b->B,MAT_CUSPARSE_MULT,cusp->offdiagGPUMatFormat));
  CHKERRQ(MatCUSPARSESetHandle(b->A,cusp->handle));
  CHKERRQ(MatCUSPARSESetHandle(b->B,cusp->handle));
  /*
  CHKERRQ(MatCUSPARSESetStream(b->A,cusp->stream));
  CHKERRQ(MatCUSPARSESetStream(b->B,cusp->stream));
  */

  CHKERRQ(MatBindToCPU(b->A,B->boundtocpu));
  CHKERRQ(MatBindToCPU(b->B,B->boundtocpu));
  CHKERRQ(MatSetUpMultiply_MPIAIJ(B));
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
  CHKERRQ(PetscFree(mpiaij->garray));
  CHKERRQ(VecDestroy(&mpiaij->lvec));
#if defined(PETSC_USE_CTABLE)
  CHKERRQ(PetscTableDestroy(&mpiaij->colmap));
#else
  CHKERRQ(PetscFree(mpiaij->colmap));
#endif
  CHKERRQ(VecScatterDestroy(&mpiaij->Mvctx));
  mat->assembled                                                                      = PETSC_FALSE;
  mat->was_assembled                                                                  = PETSC_FALSE;
  CHKERRQ(MatResetPreallocationCOO_MPIAIJ(mat));
  CHKERRQ(MatResetPreallocationCOO_MPIAIJCUSPARSE(mat));
  if (coo_i) {
    CHKERRQ(PetscLayoutGetRange(mat->rmap,&rstart,&rend));
    CHKERRQ(PetscGetMemType(coo_i,&mtype));
    if (PetscMemTypeHost(mtype)) {
      for (PetscCount k=0; k<coo_n; k++) { /* Are there negative indices or remote entries? */
        if (coo_i[k]<0 || coo_i[k]<rstart || coo_i[k]>=rend || coo_j[k]<0) {coo_basic = PETSC_FALSE; break;}
      }
    }
  }
  /* All ranks must agree on the value of coo_basic */
  CHKERRMPI(MPI_Allreduce(MPI_IN_PLACE,&coo_basic,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)mat)));
  if (coo_basic) {
    CHKERRQ(MatSetPreallocationCOO_MPIAIJCUSPARSE_Basic(mat,coo_n,coo_i,coo_j));
  } else {
    CHKERRQ(MatSetPreallocationCOO_MPIAIJ(mat,coo_n,coo_i,coo_j));
    mat->offloadmask = PETSC_OFFLOAD_CPU;
    /* creates the GPU memory */
    CHKERRQ(MatSeqAIJCUSPARSECopyToGPU(mpiaij->A));
    CHKERRQ(MatSeqAIJCUSPARSECopyToGPU(mpiaij->B));
    mpidev = static_cast<Mat_MPIAIJCUSPARSE*>(mpiaij->spptr);
    mpidev->use_extended_coo = PETSC_TRUE;

    CHKERRCUDA(cudaMalloc((void**)&mpidev->Aimap1_d,mpiaij->Annz1*sizeof(PetscCount)));
    CHKERRCUDA(cudaMalloc((void**)&mpidev->Ajmap1_d,(mpiaij->Annz1+1)*sizeof(PetscCount)));
    CHKERRCUDA(cudaMalloc((void**)&mpidev->Aperm1_d,mpiaij->Atot1*sizeof(PetscCount)));

    CHKERRCUDA(cudaMalloc((void**)&mpidev->Bimap1_d,mpiaij->Bnnz1*sizeof(PetscCount)));
    CHKERRCUDA(cudaMalloc((void**)&mpidev->Bjmap1_d,(mpiaij->Bnnz1+1)*sizeof(PetscCount)));
    CHKERRCUDA(cudaMalloc((void**)&mpidev->Bperm1_d,mpiaij->Btot1*sizeof(PetscCount)));

    CHKERRCUDA(cudaMalloc((void**)&mpidev->Aimap2_d,mpiaij->Annz2*sizeof(PetscCount)));
    CHKERRCUDA(cudaMalloc((void**)&mpidev->Ajmap2_d,(mpiaij->Annz2+1)*sizeof(PetscCount)));
    CHKERRCUDA(cudaMalloc((void**)&mpidev->Aperm2_d,mpiaij->Atot2*sizeof(PetscCount)));

    CHKERRCUDA(cudaMalloc((void**)&mpidev->Bimap2_d,mpiaij->Bnnz2*sizeof(PetscCount)));
    CHKERRCUDA(cudaMalloc((void**)&mpidev->Bjmap2_d,(mpiaij->Bnnz2+1)*sizeof(PetscCount)));
    CHKERRCUDA(cudaMalloc((void**)&mpidev->Bperm2_d,mpiaij->Btot2*sizeof(PetscCount)));

    CHKERRCUDA(cudaMalloc((void**)&mpidev->Cperm1_d,mpiaij->sendlen*sizeof(PetscCount)));
    CHKERRCUDA(cudaMalloc((void**)&mpidev->sendbuf_d,mpiaij->sendlen*sizeof(PetscScalar)));
    CHKERRCUDA(cudaMalloc((void**)&mpidev->recvbuf_d,mpiaij->recvlen*sizeof(PetscScalar)));

    CHKERRCUDA(cudaMemcpy(mpidev->Aimap1_d,mpiaij->Aimap1,mpiaij->Annz1*sizeof(PetscCount),cudaMemcpyHostToDevice));
    CHKERRCUDA(cudaMemcpy(mpidev->Ajmap1_d,mpiaij->Ajmap1,(mpiaij->Annz1+1)*sizeof(PetscCount),cudaMemcpyHostToDevice));
    CHKERRCUDA(cudaMemcpy(mpidev->Aperm1_d,mpiaij->Aperm1,mpiaij->Atot1*sizeof(PetscCount),cudaMemcpyHostToDevice));

    CHKERRCUDA(cudaMemcpy(mpidev->Bimap1_d,mpiaij->Bimap1,mpiaij->Bnnz1*sizeof(PetscCount),cudaMemcpyHostToDevice));
    CHKERRCUDA(cudaMemcpy(mpidev->Bjmap1_d,mpiaij->Bjmap1,(mpiaij->Bnnz1+1)*sizeof(PetscCount),cudaMemcpyHostToDevice));
    CHKERRCUDA(cudaMemcpy(mpidev->Bperm1_d,mpiaij->Bperm1,mpiaij->Btot1*sizeof(PetscCount),cudaMemcpyHostToDevice));

    CHKERRCUDA(cudaMemcpy(mpidev->Aimap2_d,mpiaij->Aimap2,mpiaij->Annz2*sizeof(PetscCount),cudaMemcpyHostToDevice));
    CHKERRCUDA(cudaMemcpy(mpidev->Ajmap2_d,mpiaij->Ajmap2,(mpiaij->Annz2+1)*sizeof(PetscCount),cudaMemcpyHostToDevice));
    CHKERRCUDA(cudaMemcpy(mpidev->Aperm2_d,mpiaij->Aperm2,mpiaij->Atot2*sizeof(PetscCount),cudaMemcpyHostToDevice));

    CHKERRCUDA(cudaMemcpy(mpidev->Bimap2_d,mpiaij->Bimap2,mpiaij->Bnnz2*sizeof(PetscCount),cudaMemcpyHostToDevice));
    CHKERRCUDA(cudaMemcpy(mpidev->Bjmap2_d,mpiaij->Bjmap2,(mpiaij->Bnnz2+1)*sizeof(PetscCount),cudaMemcpyHostToDevice));
    CHKERRCUDA(cudaMemcpy(mpidev->Bperm2_d,mpiaij->Bperm2,mpiaij->Btot2*sizeof(PetscCount),cudaMemcpyHostToDevice));

    CHKERRCUDA(cudaMemcpy(mpidev->Cperm1_d,mpiaij->Cperm1,mpiaij->sendlen*sizeof(PetscCount),cudaMemcpyHostToDevice));
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

    CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)mat),&size));
    CHKERRQ(PetscGetMemType(v,&memtype));
    if (PetscMemTypeHost(memtype)) { /* If user gave v[] in host, we need to copy it to device */
      CHKERRCUDA(cudaMalloc((void**)&v1,mpiaij->coo_n*sizeof(PetscScalar)));
      CHKERRCUDA(cudaMemcpy((void*)v1,v,mpiaij->coo_n*sizeof(PetscScalar),cudaMemcpyHostToDevice));
    }

    if (imode == INSERT_VALUES) {
      CHKERRQ(MatSeqAIJCUSPARSEGetArrayWrite(A,&Aa)); /* write matrix values */
      CHKERRCUDA(cudaMemset(Aa,0,((Mat_SeqAIJ*)A->data)->nz*sizeof(PetscScalar)));
      if (size > 1) {
        CHKERRQ(MatSeqAIJCUSPARSEGetArrayWrite(B,&Ba));
        CHKERRCUDA(cudaMemset(Ba,0,((Mat_SeqAIJ*)B->data)->nz*sizeof(PetscScalar)));
      }
    } else {
      CHKERRQ(MatSeqAIJCUSPARSEGetArray(A,&Aa)); /* read & write matrix values */
      if (size > 1) CHKERRQ(MatSeqAIJCUSPARSEGetArray(B,&Ba));
    }

    /* Pack entries to be sent to remote */
    if (mpiaij->sendlen) {
      MatPackCOOValues<<<(mpiaij->sendlen+255)/256,256>>>(v1,mpiaij->sendlen,Cperm1,vsend);
      CHKERRCUDA(cudaPeekAtLastError());
    }

    /* Send remote entries to their owner and overlap the communication with local computation */
    if (size > 1) CHKERRQ(PetscSFReduceWithMemTypeBegin(mpiaij->coo_sf,MPIU_SCALAR,PETSC_MEMTYPE_CUDA,vsend,PETSC_MEMTYPE_CUDA,v2,MPI_REPLACE));
    /* Add local entries to A and B */
    if (Annz1) {
      MatAddCOOValues<<<(Annz1+255)/256,256>>>(v1,Annz1,Aimap1,Ajmap1,Aperm1,Aa);
      CHKERRCUDA(cudaPeekAtLastError());
    }
    if (Bnnz1) {
      MatAddCOOValues<<<(Bnnz1+255)/256,256>>>(v1,Bnnz1,Bimap1,Bjmap1,Bperm1,Ba);
      CHKERRCUDA(cudaPeekAtLastError());
    }
    if (size > 1) CHKERRQ(PetscSFReduceEnd(mpiaij->coo_sf,MPIU_SCALAR,vsend,v2,MPI_REPLACE));

    /* Add received remote entries to A and B */
    if (Annz2) {
      MatAddCOOValues<<<(Annz2+255)/256,256>>>(v2,Annz2,Aimap2,Ajmap2,Aperm2,Aa);
      CHKERRCUDA(cudaPeekAtLastError());
    }
    if (Bnnz2) {
      MatAddCOOValues<<<(Bnnz2+255)/256,256>>>(v2,Bnnz2,Bimap2,Bjmap2,Bperm2,Ba);
      CHKERRCUDA(cudaPeekAtLastError());
    }

    if (imode == INSERT_VALUES) {
      CHKERRQ(MatSeqAIJCUSPARSERestoreArrayWrite(A,&Aa));
      if (size > 1) CHKERRQ(MatSeqAIJCUSPARSERestoreArrayWrite(B,&Ba));
    } else {
      CHKERRQ(MatSeqAIJCUSPARSERestoreArray(A,&Aa));
      if (size > 1) CHKERRQ(MatSeqAIJCUSPARSERestoreArray(B,&Ba));
    }
    if (PetscMemTypeHost(memtype)) CHKERRCUDA(cudaFree((void*)v1));
  } else {
    CHKERRQ(MatSetValuesCOO_MPIAIJCUSPARSE_Basic(mat,v,imode));
  }
  mat->offloadmask = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMPIAIJGetLocalMatMerge_MPIAIJCUSPARSE(Mat A,MatReuse scall,IS *glob,Mat *A_loc)
{
  Mat             Ad,Ao;
  const PetscInt *cmap;

  PetscFunctionBegin;
  CHKERRQ(MatMPIAIJGetSeqAIJ(A,&Ad,&Ao,&cmap));
  CHKERRQ(MatSeqAIJCUSPARSEMergeMats(Ad,Ao,scall,A_loc));
  if (glob) {
    PetscInt cst, i, dn, on, *gidx;

    CHKERRQ(MatGetLocalSize(Ad,NULL,&dn));
    CHKERRQ(MatGetLocalSize(Ao,NULL,&on));
    CHKERRQ(MatGetOwnershipRangeColumn(A,&cst,NULL));
    CHKERRQ(PetscMalloc1(dn+on,&gidx));
    for (i = 0; i<dn; i++) gidx[i]    = cst + i;
    for (i = 0; i<on; i++) gidx[i+dn] = cmap[i];
    CHKERRQ(ISCreateGeneral(PetscObjectComm((PetscObject)Ad),dn+on,gidx,PETSC_OWN_POINTER,glob));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMPIAIJSetPreallocation_MPIAIJCUSPARSE(Mat B,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  Mat_MPIAIJ *b = (Mat_MPIAIJ*)B->data;
  Mat_MPIAIJCUSPARSE *cusparseStruct = (Mat_MPIAIJCUSPARSE*)b->spptr;
  PetscInt           i;

  PetscFunctionBegin;
  CHKERRQ(PetscLayoutSetUp(B->rmap));
  CHKERRQ(PetscLayoutSetUp(B->cmap));
  if (PetscDefined(USE_DEBUG) && d_nnz) {
    for (i=0; i<B->rmap->n; i++) {
      PetscCheckFalse(d_nnz[i] < 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"d_nnz cannot be less than 0: local row %" PetscInt_FMT " value %" PetscInt_FMT,i,d_nnz[i]);
    }
  }
  if (PetscDefined(USE_DEBUG) && o_nnz) {
    for (i=0; i<B->rmap->n; i++) {
      PetscCheckFalse(o_nnz[i] < 0,PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"o_nnz cannot be less than 0: local row %" PetscInt_FMT " value %" PetscInt_FMT,i,o_nnz[i]);
    }
  }
#if defined(PETSC_USE_CTABLE)
  CHKERRQ(PetscTableDestroy(&b->colmap));
#else
  CHKERRQ(PetscFree(b->colmap));
#endif
  CHKERRQ(PetscFree(b->garray));
  CHKERRQ(VecDestroy(&b->lvec));
  CHKERRQ(VecScatterDestroy(&b->Mvctx));
  /* Because the B will have been resized we simply destroy it and create a new one each time */
  CHKERRQ(MatDestroy(&b->B));
  if (!b->A) {
    CHKERRQ(MatCreate(PETSC_COMM_SELF,&b->A));
    CHKERRQ(MatSetSizes(b->A,B->rmap->n,B->cmap->n,B->rmap->n,B->cmap->n));
    CHKERRQ(PetscLogObjectParent((PetscObject)B,(PetscObject)b->A));
  }
  if (!b->B) {
    PetscMPIInt size;
    CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)B),&size));
    CHKERRQ(MatCreate(PETSC_COMM_SELF,&b->B));
    CHKERRQ(MatSetSizes(b->B,B->rmap->n,size > 1 ? B->cmap->N : 0,B->rmap->n,size > 1 ? B->cmap->N : 0));
    CHKERRQ(PetscLogObjectParent((PetscObject)B,(PetscObject)b->B));
  }
  CHKERRQ(MatSetType(b->A,MATSEQAIJCUSPARSE));
  CHKERRQ(MatSetType(b->B,MATSEQAIJCUSPARSE));
  CHKERRQ(MatBindToCPU(b->A,B->boundtocpu));
  CHKERRQ(MatBindToCPU(b->B,B->boundtocpu));
  CHKERRQ(MatSeqAIJSetPreallocation(b->A,d_nz,d_nnz));
  CHKERRQ(MatSeqAIJSetPreallocation(b->B,o_nz,o_nnz));
  CHKERRQ(MatCUSPARSESetFormat(b->A,MAT_CUSPARSE_MULT,cusparseStruct->diagGPUMatFormat));
  CHKERRQ(MatCUSPARSESetFormat(b->B,MAT_CUSPARSE_MULT,cusparseStruct->offdiagGPUMatFormat));
  CHKERRQ(MatCUSPARSESetHandle(b->A,cusparseStruct->handle));
  CHKERRQ(MatCUSPARSESetHandle(b->B,cusparseStruct->handle));
  B->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_MPIAIJCUSPARSE(Mat A,Vec xx,Vec yy)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ*)A->data;

  PetscFunctionBegin;
  CHKERRQ(VecScatterBegin(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ((*a->A->ops->mult)(a->A,xx,yy));
  CHKERRQ(VecScatterEnd(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ((*a->B->ops->multadd)(a->B,a->lvec,yy,yy));
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroEntries_MPIAIJCUSPARSE(Mat A)
{
  Mat_MPIAIJ     *l = (Mat_MPIAIJ*)A->data;

  PetscFunctionBegin;
  CHKERRQ(MatZeroEntries(l->A));
  CHKERRQ(MatZeroEntries(l->B));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_MPIAIJCUSPARSE(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ*)A->data;

  PetscFunctionBegin;
  CHKERRQ(VecScatterBegin(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ((*a->A->ops->multadd)(a->A,xx,yy,zz));
  CHKERRQ(VecScatterEnd(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ((*a->B->ops->multadd)(a->B,a->lvec,zz,zz));
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_MPIAIJCUSPARSE(Mat A,Vec xx,Vec yy)
{
  Mat_MPIAIJ *a = (Mat_MPIAIJ*)A->data;

  PetscFunctionBegin;
  CHKERRQ((*a->B->ops->multtranspose)(a->B,xx,a->lvec));
  CHKERRQ((*a->A->ops->multtranspose)(a->A,xx,yy));
  CHKERRQ(VecScatterBegin(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE));
  CHKERRQ(VecScatterEnd(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE));
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
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"MPIAIJCUSPARSE options"));
  if (A->factortype==MAT_FACTOR_NONE) {
    CHKERRQ(PetscOptionsEnum("-mat_cusparse_mult_diag_storage_format","sets storage format of the diagonal blocks of (mpi)aijcusparse gpu matrices for SpMV",
                             "MatCUSPARSESetFormat",MatCUSPARSEStorageFormats,(PetscEnum)cusparseStruct->diagGPUMatFormat,(PetscEnum*)&format,&flg));
    if (flg) CHKERRQ(MatCUSPARSESetFormat(A,MAT_CUSPARSE_MULT_DIAG,format));
    CHKERRQ(PetscOptionsEnum("-mat_cusparse_mult_offdiag_storage_format","sets storage format of the off-diagonal blocks (mpi)aijcusparse gpu matrices for SpMV",
                             "MatCUSPARSESetFormat",MatCUSPARSEStorageFormats,(PetscEnum)cusparseStruct->offdiagGPUMatFormat,(PetscEnum*)&format,&flg));
    if (flg) CHKERRQ(MatCUSPARSESetFormat(A,MAT_CUSPARSE_MULT_OFFDIAG,format));
    CHKERRQ(PetscOptionsEnum("-mat_cusparse_storage_format","sets storage format of the diagonal and off-diagonal blocks (mpi)aijcusparse gpu matrices for SpMV",
                             "MatCUSPARSESetFormat",MatCUSPARSEStorageFormats,(PetscEnum)cusparseStruct->diagGPUMatFormat,(PetscEnum*)&format,&flg));
    if (flg) CHKERRQ(MatCUSPARSESetFormat(A,MAT_CUSPARSE_ALL,format));
  }
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_MPIAIJCUSPARSE(Mat A,MatAssemblyType mode)
{
  Mat_MPIAIJ         *mpiaij = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJCUSPARSE *cusp = (Mat_MPIAIJCUSPARSE*)mpiaij->spptr;
  PetscObjectState   onnz = A->nonzerostate;

  PetscFunctionBegin;
  CHKERRQ(MatAssemblyEnd_MPIAIJ(A,mode));
  if (mpiaij->lvec) CHKERRQ(VecSetType(mpiaij->lvec,VECSEQCUDA));
  if (onnz != A->nonzerostate && cusp->deviceMat) {
    PetscSplitCSRDataStructure d_mat = cusp->deviceMat, h_mat;

    CHKERRQ(PetscInfo(A,"Destroy device mat since nonzerostate changed\n"));
    CHKERRQ(PetscNew(&h_mat));
    CHKERRCUDA(cudaMemcpy(h_mat,d_mat,sizeof(*d_mat),cudaMemcpyDeviceToHost));
    CHKERRCUDA(cudaFree(h_mat->colmap));
    if (h_mat->allocated_indices) {
      CHKERRCUDA(cudaFree(h_mat->diag.i));
      CHKERRCUDA(cudaFree(h_mat->diag.j));
      if (h_mat->offdiag.j) {
        CHKERRCUDA(cudaFree(h_mat->offdiag.i));
        CHKERRCUDA(cudaFree(h_mat->offdiag.j));
      }
    }
    CHKERRCUDA(cudaFree(d_mat));
    CHKERRQ(PetscFree(h_mat));
    cusp->deviceMat = NULL;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MPIAIJCUSPARSE(Mat A)
{
  Mat_MPIAIJ         *aij            = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJCUSPARSE *cusparseStruct = (Mat_MPIAIJCUSPARSE*)aij->spptr;

  PetscFunctionBegin;
  PetscCheckFalse(!cusparseStruct,PETSC_COMM_SELF,PETSC_ERR_COR,"Missing spptr");
  if (cusparseStruct->deviceMat) {
    PetscSplitCSRDataStructure d_mat = cusparseStruct->deviceMat, h_mat;

    CHKERRQ(PetscInfo(A,"Have device matrix\n"));
    CHKERRQ(PetscNew(&h_mat));
    CHKERRCUDA(cudaMemcpy(h_mat,d_mat,sizeof(*d_mat),cudaMemcpyDeviceToHost));
    CHKERRCUDA(cudaFree(h_mat->colmap));
    if (h_mat->allocated_indices) {
      CHKERRCUDA(cudaFree(h_mat->diag.i));
      CHKERRCUDA(cudaFree(h_mat->diag.j));
      if (h_mat->offdiag.j) {
        CHKERRCUDA(cudaFree(h_mat->offdiag.i));
        CHKERRCUDA(cudaFree(h_mat->offdiag.j));
      }
    }
    CHKERRCUDA(cudaFree(d_mat));
    CHKERRQ(PetscFree(h_mat));
  }
  try {
    if (aij->A) CHKERRQ(MatCUSPARSEClearHandle(aij->A));
    if (aij->B) CHKERRQ(MatCUSPARSEClearHandle(aij->B));
    CHKERRCUSPARSE(cusparseDestroy(cusparseStruct->handle));
    /* We want cusparseStruct to use PetscDefaultCudaStream
    if (cusparseStruct->stream) {
      CHKERRCUDA(cudaStreamDestroy(cusparseStruct->stream));
    }
    */
    /* Free COO */
    CHKERRQ(MatResetPreallocationCOO_MPIAIJCUSPARSE(A));
    delete cusparseStruct;
  } catch(char *ex) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Mat_MPIAIJCUSPARSE error: %s", ex);
  }
  CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJGetLocalMatMerge_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOO_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatCUSPARSESetFormat_C",NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatConvert_mpiaijcusparse_hypre_C",NULL));
  CHKERRQ(MatDestroy_MPIAIJ(A));
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIAIJCUSPARSE(Mat B, MatType mtype, MatReuse reuse, Mat* newmat)
{
  Mat_MPIAIJ *a;
  Mat         A;

  PetscFunctionBegin;
  CHKERRQ(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
  if (reuse == MAT_INITIAL_MATRIX) CHKERRQ(MatDuplicate(B,MAT_COPY_VALUES,newmat));
  else if (reuse == MAT_REUSE_MATRIX) CHKERRQ(MatCopy(B,*newmat,SAME_NONZERO_PATTERN));
  A = *newmat;
  A->boundtocpu = PETSC_FALSE;
  CHKERRQ(PetscFree(A->defaultvectype));
  CHKERRQ(PetscStrallocpy(VECCUDA,&A->defaultvectype));

  a = (Mat_MPIAIJ*)A->data;
  if (a->A) CHKERRQ(MatSetType(a->A,MATSEQAIJCUSPARSE));
  if (a->B) CHKERRQ(MatSetType(a->B,MATSEQAIJCUSPARSE));
  if (a->lvec) CHKERRQ(VecSetType(a->lvec,VECSEQCUDA));

  if (reuse != MAT_REUSE_MATRIX && !a->spptr) {
    Mat_MPIAIJCUSPARSE *cusp = new Mat_MPIAIJCUSPARSE;
    CHKERRCUSPARSE(cusparseCreate(&(cusp->handle)));
    a->spptr = cusp;
  }

  A->ops->assemblyend           = MatAssemblyEnd_MPIAIJCUSPARSE;
  A->ops->mult                  = MatMult_MPIAIJCUSPARSE;
  A->ops->multadd               = MatMultAdd_MPIAIJCUSPARSE;
  A->ops->multtranspose         = MatMultTranspose_MPIAIJCUSPARSE;
  A->ops->setfromoptions        = MatSetFromOptions_MPIAIJCUSPARSE;
  A->ops->destroy               = MatDestroy_MPIAIJCUSPARSE;
  A->ops->zeroentries           = MatZeroEntries_MPIAIJCUSPARSE;
  A->ops->productsetfromoptions = MatProductSetFromOptions_MPIAIJBACKEND;

  CHKERRQ(PetscObjectChangeTypeName((PetscObject)A,MATMPIAIJCUSPARSE));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJGetLocalMatMerge_C",MatMPIAIJGetLocalMatMerge_MPIAIJCUSPARSE));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",MatMPIAIJSetPreallocation_MPIAIJCUSPARSE));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatCUSPARSESetFormat_C",MatCUSPARSESetFormat_MPIAIJCUSPARSE));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOO_C",MatSetPreallocationCOO_MPIAIJCUSPARSE));
  CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",MatSetValuesCOO_MPIAIJCUSPARSE));
#if defined(PETSC_HAVE_HYPRE)
  CHKERRQ(PetscObjectComposeFunction((PetscObject)A,"MatConvert_mpiaijcusparse_hypre_C",MatConvert_AIJ_HYPRE));
#endif
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJCUSPARSE(Mat A)
{
  PetscFunctionBegin;
  CHKERRQ(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
  CHKERRQ(MatCreate_MPIAIJ(A));
  CHKERRQ(MatConvert_MPIAIJ_MPIAIJCUSPARSE(A,MATMPIAIJCUSPARSE,MAT_INPLACE_MATRIX,&A));
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
  CHKERRQ(MatCreate(comm,A));
  CHKERRQ(MatSetSizes(*A,m,n,M,N));
  CHKERRMPI(MPI_Comm_size(comm,&size));
  if (size > 1) {
    CHKERRQ(MatSetType(*A,MATMPIAIJCUSPARSE));
    CHKERRQ(MatMPIAIJSetPreallocation(*A,d_nz,d_nnz,o_nz,o_nnz));
  } else {
    CHKERRQ(MatSetType(*A,MATSEQAIJCUSPARSE));
    CHKERRQ(MatSeqAIJSetPreallocation(*A,d_nz,d_nnz));
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
  PetscCheckFalse(!A->assembled,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Need already assembled matrix");
  if (A->factortype != MAT_FACTOR_NONE) {
    *B = NULL;
    PetscFunctionReturn(0);
  }
  CHKERRMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  // get jaca
  if (size == 1) {
    PetscBool isseqaij;

    CHKERRQ(PetscObjectBaseTypeCompare((PetscObject)A,MATSEQAIJ,&isseqaij));
    if (isseqaij) {
      jaca = (Mat_SeqAIJ*)A->data;
      PetscCheckFalse(!jaca->roworiented,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Device assembly does not currently support column oriented values insertion");
      cusparsestructA = (Mat_SeqAIJCUSPARSE*)A->spptr;
      d_mat = cusparsestructA->deviceMat;
      CHKERRQ(MatSeqAIJCUSPARSECopyToGPU(A));
    } else {
      Mat_MPIAIJ *aij = (Mat_MPIAIJ*)A->data;
      PetscCheckFalse(!aij->roworiented,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Device assembly does not currently support column oriented values insertion");
      Mat_MPIAIJCUSPARSE *spptr = (Mat_MPIAIJCUSPARSE*)aij->spptr;
      jaca = (Mat_SeqAIJ*)aij->A->data;
      cusparsestructA = (Mat_SeqAIJCUSPARSE*)aij->A->spptr;
      d_mat = spptr->deviceMat;
      CHKERRQ(MatSeqAIJCUSPARSECopyToGPU(aij->A));
    }
    if (cusparsestructA->format==MAT_CUSPARSE_CSR) {
      Mat_SeqAIJCUSPARSEMultStruct *matstruct = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestructA->mat;
      PetscCheckFalse(!matstruct,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing Mat_SeqAIJCUSPARSEMultStruct for A");
      matrixA = (CsrMatrix*)matstruct->mat;
      bi = NULL;
      bj = NULL;
      ba = NULL;
    } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Device Mat needs MAT_CUSPARSE_CSR");
  } else {
    Mat_MPIAIJ *aij = (Mat_MPIAIJ*)A->data;
    PetscCheckFalse(!aij->roworiented,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Device assembly does not currently support column oriented values insertion");
    jaca = (Mat_SeqAIJ*)aij->A->data;
    jacb = (Mat_SeqAIJ*)aij->B->data;
    Mat_MPIAIJCUSPARSE *spptr = (Mat_MPIAIJCUSPARSE*)aij->spptr;

    PetscCheckFalse(!A->nooffprocentries && !aij->donotstash,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Device assembly does not currently support offproc values insertion. Use MatSetOption(A,MAT_NO_OFF_PROC_ENTRIES,PETSC_TRUE) or MatSetOption(A,MAT_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE)");
    cusparsestructA = (Mat_SeqAIJCUSPARSE*)aij->A->spptr;
    Mat_SeqAIJCUSPARSE *cusparsestructB = (Mat_SeqAIJCUSPARSE*)aij->B->spptr;
    PetscCheckFalse(cusparsestructA->format!=MAT_CUSPARSE_CSR,PETSC_COMM_SELF,PETSC_ERR_SUP,"Device Mat A needs MAT_CUSPARSE_CSR");
    if (cusparsestructB->format==MAT_CUSPARSE_CSR) {
      CHKERRQ(MatSeqAIJCUSPARSECopyToGPU(aij->A));
      CHKERRQ(MatSeqAIJCUSPARSECopyToGPU(aij->B));
      Mat_SeqAIJCUSPARSEMultStruct *matstructA = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestructA->mat;
      Mat_SeqAIJCUSPARSEMultStruct *matstructB = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestructB->mat;
      PetscCheckFalse(!matstructA,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing Mat_SeqAIJCUSPARSEMultStruct for A");
      PetscCheckFalse(!matstructB,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing Mat_SeqAIJCUSPARSEMultStruct for B");
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
    CHKERRQ(PetscInfo(A,"Create device matrix\n"));
    CHKERRQ(PetscNew(&h_mat));
    CHKERRCUDA(cudaMalloc((void**)&d_mat,sizeof(*d_mat)));
    if (size > 1) { /* need the colmap array */
      Mat_MPIAIJ *aij = (Mat_MPIAIJ*)A->data;
      PetscInt   *colmap;
      PetscInt   ii,n = aij->B->cmap->n,N = A->cmap->N;

      PetscCheckFalse(n && !aij->garray,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MPIAIJ Matrix was assembled but is missing garray");

      CHKERRQ(PetscCalloc1(N+1,&colmap));
      for (ii=0; ii<n; ii++) colmap[aij->garray[ii]] = (int)(ii+1);
#if defined(PETSC_USE_64BIT_INDICES)
      { // have to make a long version of these
        int        *h_bi32, *h_bj32;
        PetscInt   *h_bi64, *h_bj64, *d_bi64, *d_bj64;
        CHKERRQ(PetscCalloc4(A->rmap->n+1,&h_bi32,jacb->nz,&h_bj32,A->rmap->n+1,&h_bi64,jacb->nz,&h_bj64));
        CHKERRCUDA(cudaMemcpy(h_bi32, bi, (A->rmap->n+1)*sizeof(*h_bi32),cudaMemcpyDeviceToHost));
        for (int i=0;i<A->rmap->n+1;i++) h_bi64[i] = h_bi32[i];
        CHKERRCUDA(cudaMemcpy(h_bj32, bj, jacb->nz*sizeof(*h_bj32),cudaMemcpyDeviceToHost));
        for (int i=0;i<jacb->nz;i++) h_bj64[i] = h_bj32[i];

        CHKERRCUDA(cudaMalloc((void**)&d_bi64,(A->rmap->n+1)*sizeof(*d_bi64)));
        CHKERRCUDA(cudaMemcpy(d_bi64, h_bi64,(A->rmap->n+1)*sizeof(*d_bi64),cudaMemcpyHostToDevice));
        CHKERRCUDA(cudaMalloc((void**)&d_bj64,jacb->nz*sizeof(*d_bj64)));
        CHKERRCUDA(cudaMemcpy(d_bj64, h_bj64,jacb->nz*sizeof(*d_bj64),cudaMemcpyHostToDevice));

        h_mat->offdiag.i = d_bi64;
        h_mat->offdiag.j = d_bj64;
        h_mat->allocated_indices = PETSC_TRUE;

        CHKERRQ(PetscFree4(h_bi32,h_bj32,h_bi64,h_bj64));
      }
#else
      h_mat->offdiag.i = (PetscInt*)bi;
      h_mat->offdiag.j = (PetscInt*)bj;
      h_mat->allocated_indices = PETSC_FALSE;
#endif
      h_mat->offdiag.a = ba;
      h_mat->offdiag.n = A->rmap->n;

      CHKERRCUDA(cudaMalloc((void**)&h_mat->colmap,(N+1)*sizeof(*h_mat->colmap)));
      CHKERRCUDA(cudaMemcpy(h_mat->colmap,colmap,(N+1)*sizeof(*h_mat->colmap),cudaMemcpyHostToDevice));
      CHKERRQ(PetscFree(colmap));
    }
    h_mat->rstart = A->rmap->rstart;
    h_mat->rend   = A->rmap->rend;
    h_mat->cstart = A->cmap->rstart;
    h_mat->cend   = A->cmap->rend;
    h_mat->M      = A->cmap->N;
#if defined(PETSC_USE_64BIT_INDICES)
    {
      PetscCheckFalse(sizeof(PetscInt) != 8,PETSC_COMM_SELF,PETSC_ERR_PLIB,"size pof PetscInt = %d",sizeof(PetscInt));
      int        *h_ai32, *h_aj32;
      PetscInt   *h_ai64, *h_aj64, *d_ai64, *d_aj64;
      CHKERRQ(PetscCalloc4(A->rmap->n+1,&h_ai32,jaca->nz,&h_aj32,A->rmap->n+1,&h_ai64,jaca->nz,&h_aj64));
      CHKERRCUDA(cudaMemcpy(h_ai32, ai, (A->rmap->n+1)*sizeof(*h_ai32),cudaMemcpyDeviceToHost));
      for (int i=0;i<A->rmap->n+1;i++) h_ai64[i] = h_ai32[i];
      CHKERRCUDA(cudaMemcpy(h_aj32, aj, jaca->nz*sizeof(*h_aj32),cudaMemcpyDeviceToHost));
      for (int i=0;i<jaca->nz;i++) h_aj64[i] = h_aj32[i];

      CHKERRCUDA(cudaMalloc((void**)&d_ai64,(A->rmap->n+1)*sizeof(*d_ai64)));
      CHKERRCUDA(cudaMemcpy(d_ai64, h_ai64,(A->rmap->n+1)*sizeof(*d_ai64),cudaMemcpyHostToDevice));
      CHKERRCUDA(cudaMalloc((void**)&d_aj64,jaca->nz*sizeof(*d_aj64)));
      CHKERRCUDA(cudaMemcpy(d_aj64, h_aj64,jaca->nz*sizeof(*d_aj64),cudaMemcpyHostToDevice));

      h_mat->diag.i = d_ai64;
      h_mat->diag.j = d_aj64;
      h_mat->allocated_indices = PETSC_TRUE;

      CHKERRQ(PetscFree4(h_ai32,h_aj32,h_ai64,h_aj64));
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
    CHKERRCUDA(cudaMemcpy(d_mat,h_mat,sizeof(*d_mat),cudaMemcpyHostToDevice));
    CHKERRQ(PetscFree(h_mat));
  } else {
    CHKERRQ(PetscInfo(A,"Reusing device matrix\n"));
  }
  *B = d_mat;
  A->offloadmask = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(0);
}
