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
  cudaError_t        cerr;
  Mat_MPIAIJ         *aij = (Mat_MPIAIJ*)mat->data;
  Mat_MPIAIJCUSPARSE *cusparseStruct = (Mat_MPIAIJCUSPARSE*)aij->spptr;

  PetscFunctionBegin;
  if (!cusparseStruct) PetscFunctionReturn(0);
  if (cusparseStruct->use_extended_coo) {
    cerr = cudaFree(cusparseStruct->Aimap1_d);CHKERRCUDA(cerr);
    cerr = cudaFree(cusparseStruct->Ajmap1_d);CHKERRCUDA(cerr);
    cerr = cudaFree(cusparseStruct->Aperm1_d);CHKERRCUDA(cerr);
    cerr = cudaFree(cusparseStruct->Bimap1_d);CHKERRCUDA(cerr);
    cerr = cudaFree(cusparseStruct->Bjmap1_d);CHKERRCUDA(cerr);
    cerr = cudaFree(cusparseStruct->Bperm1_d);CHKERRCUDA(cerr);
    cerr = cudaFree(cusparseStruct->Aimap2_d);CHKERRCUDA(cerr);
    cerr = cudaFree(cusparseStruct->Ajmap2_d);CHKERRCUDA(cerr);
    cerr = cudaFree(cusparseStruct->Aperm2_d);CHKERRCUDA(cerr);
    cerr = cudaFree(cusparseStruct->Bimap2_d);CHKERRCUDA(cerr);
    cerr = cudaFree(cusparseStruct->Bjmap2_d);CHKERRCUDA(cerr);
    cerr = cudaFree(cusparseStruct->Bperm2_d);CHKERRCUDA(cerr);
    cerr = cudaFree(cusparseStruct->Cperm1_d);CHKERRCUDA(cerr);
    cerr = cudaFree(cusparseStruct->sendbuf_d);CHKERRCUDA(cerr);
    cerr = cudaFree(cusparseStruct->recvbuf_d);CHKERRCUDA(cerr);
  }
  cusparseStruct->use_extended_coo = PETSC_FALSE;
  delete cusparseStruct->coo_p;
  delete cusparseStruct->coo_pw;
  cusparseStruct->coo_p = NULL;
  cusparseStruct->coo_pw = NULL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetValuesCOO_MPIAIJCUSPARSE_Basic(Mat A, const PetscScalar v[], InsertMode imode)
{
  Mat_MPIAIJ         *a = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJCUSPARSE *cusp = (Mat_MPIAIJCUSPARSE*)a->spptr;
  PetscInt           n = cusp->coo_nd + cusp->coo_no;
  PetscErrorCode     ierr;

  PetscFunctionBegin;
  if (cusp->coo_p && v) {
    thrust::device_ptr<const PetscScalar> d_v;
    THRUSTARRAY                           *w = NULL;

    if (isCudaMem(v)) {
      d_v = thrust::device_pointer_cast(v);
    } else {
      w = new THRUSTARRAY(n);
      w->assign(v,v+n);
      ierr = PetscLogCpuToGpu(n*sizeof(PetscScalar));CHKERRQ(ierr);
      d_v = w->data();
    }

    auto zibit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(d_v,cusp->coo_p->begin()),
                                                              cusp->coo_pw->begin()));
    auto zieit = thrust::make_zip_iterator(thrust::make_tuple(thrust::make_permutation_iterator(d_v,cusp->coo_p->end()),
                                                              cusp->coo_pw->end()));
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    thrust::for_each(zibit,zieit,VecCUDAEquals());
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    delete w;
    ierr = MatSetValuesCOO_SeqAIJCUSPARSE_Basic(a->A,cusp->coo_pw->data().get(),imode);CHKERRQ(ierr);
    ierr = MatSetValuesCOO_SeqAIJCUSPARSE_Basic(a->B,cusp->coo_pw->data().get()+cusp->coo_nd,imode);CHKERRQ(ierr);
  } else {
    ierr = MatSetValuesCOO_SeqAIJCUSPARSE_Basic(a->A,v,imode);CHKERRQ(ierr);
    ierr = MatSetValuesCOO_SeqAIJCUSPARSE_Basic(a->B,v ? v+cusp->coo_nd : NULL,imode);CHKERRQ(ierr);
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
  PetscErrorCode         ierr;
  PetscInt               N,*jj;
  size_t                 noff = 0;
  THRUSTINTARRAY         d_i(n); /* on device, storing partitioned coo_i with diagonal first, and off-diag next */
  THRUSTINTARRAY         d_j(n);
  ISLocalToGlobalMapping l2g;
  cudaError_t            cerr;

  PetscFunctionBegin;
  if (b->A) { ierr = MatCUSPARSEClearHandle(b->A);CHKERRQ(ierr); }
  if (b->B) { ierr = MatCUSPARSEClearHandle(b->B);CHKERRQ(ierr); }
  ierr = MatDestroy(&b->A);CHKERRQ(ierr);
  ierr = MatDestroy(&b->B);CHKERRQ(ierr);

  ierr = PetscLogCpuToGpu(2.*n*sizeof(PetscInt));CHKERRQ(ierr);
  d_i.assign(coo_i,coo_i+n);
  d_j.assign(coo_j,coo_j+n);
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
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
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);

  /* copy offdiag column indices to map on the CPU */
  ierr = PetscMalloc1(cusp->coo_no,&jj);CHKERRQ(ierr); /* jj[] will store compacted col ids of the offdiag part */
  cerr = cudaMemcpy(jj,d_j.data().get()+cusp->coo_nd,cusp->coo_no*sizeof(PetscInt),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
  auto o_j = d_j.begin();
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  thrust::advance(o_j,cusp->coo_nd); /* sort and unique offdiag col ids */
  thrust::sort(thrust::device,o_j,d_j.end());
  auto wit = thrust::unique(thrust::device,o_j,d_j.end()); /* return end iter of the unique range */
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  noff = thrust::distance(o_j,wit);
  /* allocate the garray, the columns of B will not be mapped in the multiply setup */
  ierr = PetscMalloc1(noff,&b->garray);CHKERRQ(ierr);
  cerr = cudaMemcpy(b->garray,d_j.data().get()+cusp->coo_nd,noff*sizeof(PetscInt),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
  ierr = PetscLogGpuToCpu((noff+cusp->coo_no)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,1,noff,b->garray,PETSC_COPY_VALUES,&l2g);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingSetType(l2g,ISLOCALTOGLOBALMAPPINGHASH);CHKERRQ(ierr);
  ierr = ISGlobalToLocalMappingApply(l2g,IS_GTOLM_DROP,cusp->coo_no,jj,&N,jj);CHKERRQ(ierr);
  PetscCheck(N == cusp->coo_no,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected is size %" PetscInt_FMT " != %" PetscInt_FMT " coo size",N,cusp->coo_no);
  ierr = ISLocalToGlobalMappingDestroy(&l2g);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_SELF,&b->A);CHKERRQ(ierr);
  ierr = MatSetSizes(b->A,B->rmap->n,B->cmap->n,B->rmap->n,B->cmap->n);CHKERRQ(ierr);
  ierr = MatSetType(b->A,MATSEQAIJCUSPARSE);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)B,(PetscObject)b->A);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_SELF,&b->B);CHKERRQ(ierr);
  ierr = MatSetSizes(b->B,B->rmap->n,noff,B->rmap->n,noff);CHKERRQ(ierr);
  ierr = MatSetType(b->B,MATSEQAIJCUSPARSE);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject)B,(PetscObject)b->B);CHKERRQ(ierr);

  /* GPU memory, cusparse specific call handles it internally */
  ierr = MatSetPreallocationCOO_SeqAIJCUSPARSE_Basic(b->A,cusp->coo_nd,d_i.data().get(),d_j.data().get());CHKERRQ(ierr);
  ierr = MatSetPreallocationCOO_SeqAIJCUSPARSE_Basic(b->B,cusp->coo_no,d_i.data().get()+cusp->coo_nd,jj);CHKERRQ(ierr);
  ierr = PetscFree(jj);CHKERRQ(ierr);

  ierr = MatCUSPARSESetFormat(b->A,MAT_CUSPARSE_MULT,cusp->diagGPUMatFormat);CHKERRQ(ierr);
  ierr = MatCUSPARSESetFormat(b->B,MAT_CUSPARSE_MULT,cusp->offdiagGPUMatFormat);CHKERRQ(ierr);
  ierr = MatCUSPARSESetHandle(b->A,cusp->handle);CHKERRQ(ierr);
  ierr = MatCUSPARSESetHandle(b->B,cusp->handle);CHKERRQ(ierr);
  /*
  ierr = MatCUSPARSESetStream(b->A,cusp->stream);CHKERRQ(ierr);
  ierr = MatCUSPARSESetStream(b->B,cusp->stream);CHKERRQ(ierr);
  */

  ierr = MatBindToCPU(b->A,B->boundtocpu);CHKERRQ(ierr);
  ierr = MatBindToCPU(b->B,B->boundtocpu);CHKERRQ(ierr);
  ierr = MatSetUpMultiply_MPIAIJ(B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetPreallocationCOO_MPIAIJCUSPARSE(Mat mat, PetscCount coo_n, const PetscInt coo_i[], const PetscInt coo_j[])
{
  PetscErrorCode         ierr;
  Mat_MPIAIJ             *mpiaij = (Mat_MPIAIJ*)mat->data;
  Mat_MPIAIJCUSPARSE     *mpidev;
  PetscBool              coo_basic = PETSC_TRUE;
  PetscMemType           mtype = PETSC_MEMTYPE_DEVICE;
  PetscInt               rstart,rend;
  cudaError_t            cerr;

  PetscFunctionBegin;
  ierr = PetscFree(mpiaij->garray);CHKERRQ(ierr);
  ierr = VecDestroy(&mpiaij->lvec);CHKERRQ(ierr);
#if defined(PETSC_USE_CTABLE)
  ierr = PetscTableDestroy(&mpiaij->colmap);CHKERRQ(ierr);
#else
  ierr = PetscFree(mpiaij->colmap);CHKERRQ(ierr);
#endif
  ierr = VecScatterDestroy(&mpiaij->Mvctx);CHKERRQ(ierr);
  mat->assembled = PETSC_FALSE;
  mat->was_assembled = PETSC_FALSE;
  ierr = MatResetPreallocationCOO_MPIAIJ(mat);CHKERRQ(ierr);
  ierr = MatResetPreallocationCOO_MPIAIJCUSPARSE(mat);CHKERRQ(ierr);
  if (coo_i) {
    ierr = PetscLayoutGetRange(mat->rmap,&rstart,&rend);CHKERRQ(ierr);
    ierr = PetscGetMemType(coo_i,&mtype);CHKERRQ(ierr);
    if (PetscMemTypeHost(mtype)) {
      for (PetscCount k=0; k<coo_n; k++) { /* Are there negative indices or remote entries? */
        if (coo_i[k]<0 || coo_i[k]<rstart || coo_i[k]>=rend || coo_j[k]<0) {coo_basic = PETSC_FALSE; break;}
      }
    }
  }
  /* All ranks must agree on the value of coo_basic */
  ierr = MPI_Allreduce(MPI_IN_PLACE,&coo_basic,1,MPIU_BOOL,MPI_LAND,PetscObjectComm((PetscObject)mat));CHKERRMPI(ierr);
  if (coo_basic) {
    ierr = MatSetPreallocationCOO_MPIAIJCUSPARSE_Basic(mat,coo_n,coo_i,coo_j);CHKERRQ(ierr);
  } else {
    ierr = MatSetPreallocationCOO_MPIAIJ(mat,coo_n,coo_i,coo_j);CHKERRQ(ierr);
    mat->offloadmask = PETSC_OFFLOAD_CPU;
    /* creates the GPU memory */
    ierr = MatSeqAIJCUSPARSECopyToGPU(mpiaij->A);CHKERRQ(ierr);
    ierr = MatSeqAIJCUSPARSECopyToGPU(mpiaij->B);CHKERRQ(ierr);
    mpidev = static_cast<Mat_MPIAIJCUSPARSE*>(mpiaij->spptr);
    mpidev->use_extended_coo = PETSC_TRUE;

    cerr = cudaMalloc((void**)&mpidev->Aimap1_d,mpiaij->Annz1*sizeof(PetscCount));CHKERRCUDA(cerr);
    cerr = cudaMalloc((void**)&mpidev->Ajmap1_d,(mpiaij->Annz1+1)*sizeof(PetscCount));CHKERRCUDA(cerr);
    cerr = cudaMalloc((void**)&mpidev->Aperm1_d,mpiaij->Atot1*sizeof(PetscCount));CHKERRCUDA(cerr);

    cerr = cudaMalloc((void**)&mpidev->Bimap1_d,mpiaij->Bnnz1*sizeof(PetscCount));CHKERRCUDA(cerr);
    cerr = cudaMalloc((void**)&mpidev->Bjmap1_d,(mpiaij->Bnnz1+1)*sizeof(PetscCount));CHKERRCUDA(cerr);
    cerr = cudaMalloc((void**)&mpidev->Bperm1_d,mpiaij->Btot1*sizeof(PetscCount));CHKERRCUDA(cerr);

    cerr = cudaMalloc((void**)&mpidev->Aimap2_d,mpiaij->Annz2*sizeof(PetscCount));CHKERRCUDA(cerr);
    cerr = cudaMalloc((void**)&mpidev->Ajmap2_d,(mpiaij->Annz2+1)*sizeof(PetscCount));CHKERRCUDA(cerr);
    cerr = cudaMalloc((void**)&mpidev->Aperm2_d,mpiaij->Atot2*sizeof(PetscCount));CHKERRCUDA(cerr);

    cerr = cudaMalloc((void**)&mpidev->Bimap2_d,mpiaij->Bnnz2*sizeof(PetscCount));CHKERRCUDA(cerr);
    cerr = cudaMalloc((void**)&mpidev->Bjmap2_d,(mpiaij->Bnnz2+1)*sizeof(PetscCount));CHKERRCUDA(cerr);
    cerr = cudaMalloc((void**)&mpidev->Bperm2_d,mpiaij->Btot2*sizeof(PetscCount));CHKERRCUDA(cerr);

    cerr = cudaMalloc((void**)&mpidev->Cperm1_d,mpiaij->sendlen*sizeof(PetscCount));CHKERRCUDA(cerr);
    cerr = cudaMalloc((void**)&mpidev->sendbuf_d,mpiaij->sendlen*sizeof(PetscScalar));CHKERRCUDA(cerr);
    cerr = cudaMalloc((void**)&mpidev->recvbuf_d,mpiaij->recvlen*sizeof(PetscScalar));CHKERRCUDA(cerr);

    cerr = cudaMemcpy(mpidev->Aimap1_d,mpiaij->Aimap1,mpiaij->Annz1*sizeof(PetscCount),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    cerr = cudaMemcpy(mpidev->Ajmap1_d,mpiaij->Ajmap1,(mpiaij->Annz1+1)*sizeof(PetscCount),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    cerr = cudaMemcpy(mpidev->Aperm1_d,mpiaij->Aperm1,mpiaij->Atot1*sizeof(PetscCount),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);

    cerr = cudaMemcpy(mpidev->Bimap1_d,mpiaij->Bimap1,mpiaij->Bnnz1*sizeof(PetscCount),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    cerr = cudaMemcpy(mpidev->Bjmap1_d,mpiaij->Bjmap1,(mpiaij->Bnnz1+1)*sizeof(PetscCount),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    cerr = cudaMemcpy(mpidev->Bperm1_d,mpiaij->Bperm1,mpiaij->Btot1*sizeof(PetscCount),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);

    cerr = cudaMemcpy(mpidev->Aimap2_d,mpiaij->Aimap2,mpiaij->Annz2*sizeof(PetscCount),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    cerr = cudaMemcpy(mpidev->Ajmap2_d,mpiaij->Ajmap2,(mpiaij->Annz2+1)*sizeof(PetscCount),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    cerr = cudaMemcpy(mpidev->Aperm2_d,mpiaij->Aperm2,mpiaij->Atot2*sizeof(PetscCount),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);

    cerr = cudaMemcpy(mpidev->Bimap2_d,mpiaij->Bimap2,mpiaij->Bnnz2*sizeof(PetscCount),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    cerr = cudaMemcpy(mpidev->Bjmap2_d,mpiaij->Bjmap2,(mpiaij->Bnnz2+1)*sizeof(PetscCount),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    cerr = cudaMemcpy(mpidev->Bperm2_d,mpiaij->Bperm2,mpiaij->Btot2*sizeof(PetscCount),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);

    cerr = cudaMemcpy(mpidev->Cperm1_d,mpiaij->Cperm1,mpiaij->sendlen*sizeof(PetscCount),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
  }
  PetscFunctionReturn(0);
}

__global__ void MatPackCOOValues(const PetscScalar kv[],PetscCount nnz,const PetscCount perm[],PetscScalar buf[])
{
  PetscCount        i = blockIdx.x*blockDim.x + threadIdx.x;
  const PetscCount  grid_size = gridDim.x * blockDim.x;
  for (; i<nnz; i+= grid_size) buf[i] = kv[perm[i]];
}

__global__ void MatAddCOOValues(const PetscScalar kv[],PetscCount nnz,const PetscCount imap[],const PetscCount jmap[],const PetscCount perm[],PetscScalar a[])
{
  PetscCount        i = blockIdx.x*blockDim.x + threadIdx.x;
  const PetscCount  grid_size = gridDim.x * blockDim.x;
  for (; i<nnz; i+= grid_size) {for (PetscCount k=jmap[i]; k<jmap[i+1]; k++) a[imap[i]] += kv[perm[k]];}
}

static PetscErrorCode MatSetValuesCOO_MPIAIJCUSPARSE(Mat mat,const PetscScalar v[],InsertMode imode)
{
  PetscErrorCode                 ierr;
  cudaError_t                    cerr;
  Mat_MPIAIJ                     *mpiaij = static_cast<Mat_MPIAIJ*>(mat->data);
  Mat_MPIAIJCUSPARSE             *mpidev = static_cast<Mat_MPIAIJCUSPARSE*>(mpiaij->spptr);
  Mat                            A = mpiaij->A,B = mpiaij->B;
  PetscCount                     Annz1 = mpiaij->Annz1,Annz2 = mpiaij->Annz2,Bnnz1 = mpiaij->Bnnz1,Bnnz2 = mpiaij->Bnnz2;
  PetscScalar                    *Aa,*Ba = NULL;
  PetscScalar                    *vsend = mpidev->sendbuf_d,*v2 = mpidev->recvbuf_d;
  const PetscScalar              *v1 = v;
  const PetscCount               *Ajmap1 = mpidev->Ajmap1_d,*Ajmap2 = mpidev->Ajmap2_d,*Aimap1 = mpidev->Aimap1_d,*Aimap2 = mpidev->Aimap2_d;
  const PetscCount               *Bjmap1 = mpidev->Bjmap1_d,*Bjmap2 = mpidev->Bjmap2_d,*Bimap1 = mpidev->Bimap1_d,*Bimap2 = mpidev->Bimap2_d;
  const PetscCount               *Aperm1 = mpidev->Aperm1_d,*Aperm2 = mpidev->Aperm2_d,*Bperm1 = mpidev->Bperm1_d,*Bperm2 = mpidev->Bperm2_d;
  const PetscCount               *Cperm1 = mpidev->Cperm1_d;
  PetscMemType                   memtype;

  PetscFunctionBegin;
  if (mpidev->use_extended_coo) {
    PetscMPIInt size;

    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)mat),&size);CHKERRMPI(ierr);
    ierr = PetscGetMemType(v,&memtype);CHKERRQ(ierr);
    if (PetscMemTypeHost(memtype)) { /* If user gave v[] in host, we need to copy it to device */
      cerr = cudaMalloc((void**)&v1,mpiaij->coo_n*sizeof(PetscScalar));CHKERRCUDA(cerr);
      cerr = cudaMemcpy((void*)v1,v,mpiaij->coo_n*sizeof(PetscScalar),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    }

    if (imode == INSERT_VALUES) {
      ierr = MatSeqAIJCUSPARSEGetArrayWrite(A,&Aa);CHKERRQ(ierr); /* write matrix values */
      cerr = cudaMemset(Aa,0,((Mat_SeqAIJ*)A->data)->nz*sizeof(PetscScalar));CHKERRCUDA(cerr);
      if (size > 1) {
        ierr = MatSeqAIJCUSPARSEGetArrayWrite(B,&Ba);CHKERRQ(ierr);
        cerr = cudaMemset(Ba,0,((Mat_SeqAIJ*)B->data)->nz*sizeof(PetscScalar));CHKERRCUDA(cerr);
      }
    } else {
      ierr = MatSeqAIJCUSPARSEGetArray(A,&Aa);CHKERRQ(ierr); /* read & write matrix values */
      if (size > 1) { ierr = MatSeqAIJCUSPARSEGetArray(B,&Ba);CHKERRQ(ierr); }
    }

    /* Pack entries to be sent to remote */
    if (mpiaij->sendlen) {
      MatPackCOOValues<<<(mpiaij->sendlen+255)/256,256>>>(v1,mpiaij->sendlen,Cperm1,vsend);
      CHKERRCUDA(cudaPeekAtLastError());
    }

    /* Send remote entries to their owner and overlap the communication with local computation */
    if (size > 1) { ierr = PetscSFReduceWithMemTypeBegin(mpiaij->coo_sf,MPIU_SCALAR,PETSC_MEMTYPE_CUDA,vsend,PETSC_MEMTYPE_CUDA,v2,MPI_REPLACE);CHKERRQ(ierr); }
    /* Add local entries to A and B */
    if (Annz1) {
      MatAddCOOValues<<<(Annz1+255)/256,256>>>(v1,Annz1,Aimap1,Ajmap1,Aperm1,Aa);
      CHKERRCUDA(cudaPeekAtLastError());
    }
    if (Bnnz1) {
      MatAddCOOValues<<<(Bnnz1+255)/256,256>>>(v1,Bnnz1,Bimap1,Bjmap1,Bperm1,Ba);
      CHKERRCUDA(cudaPeekAtLastError());
    }
    if (size > 1) { ierr = PetscSFReduceEnd(mpiaij->coo_sf,MPIU_SCALAR,vsend,v2,MPI_REPLACE);CHKERRQ(ierr); }

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
      ierr = MatSeqAIJCUSPARSERestoreArrayWrite(A,&Aa);CHKERRQ(ierr);
      if (size > 1) { ierr = MatSeqAIJCUSPARSERestoreArrayWrite(B,&Ba);CHKERRQ(ierr); }
    } else {
      ierr = MatSeqAIJCUSPARSERestoreArray(A,&Aa);CHKERRQ(ierr);
      if (size > 1) { ierr = MatSeqAIJCUSPARSERestoreArray(B,&Ba);CHKERRQ(ierr); }
    }
    if (PetscMemTypeHost(memtype)) {cerr = cudaFree((void*)v1);CHKERRCUDA(cerr);}
  } else {
    ierr = MatSetValuesCOO_MPIAIJCUSPARSE_Basic(mat,v,imode);CHKERRQ(ierr);
  }
  mat->offloadmask = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMPIAIJGetLocalMatMerge_MPIAIJCUSPARSE(Mat A,MatReuse scall,IS *glob,Mat *A_loc)
{
  Mat            Ad,Ao;
  const PetscInt *cmap;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMPIAIJGetSeqAIJ(A,&Ad,&Ao,&cmap);CHKERRQ(ierr);
  ierr = MatSeqAIJCUSPARSEMergeMats(Ad,Ao,scall,A_loc);CHKERRQ(ierr);
  if (glob) {
    PetscInt cst, i, dn, on, *gidx;

    ierr = MatGetLocalSize(Ad,NULL,&dn);CHKERRQ(ierr);
    ierr = MatGetLocalSize(Ao,NULL,&on);CHKERRQ(ierr);
    ierr = MatGetOwnershipRangeColumn(A,&cst,NULL);CHKERRQ(ierr);
    ierr = PetscMalloc1(dn+on,&gidx);CHKERRQ(ierr);
    for (i=0; i<dn; i++) gidx[i]    = cst + i;
    for (i=0; i<on; i++) gidx[i+dn] = cmap[i];
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)Ad),dn+on,gidx,PETSC_OWN_POINTER,glob);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMPIAIJSetPreallocation_MPIAIJCUSPARSE(Mat B,PetscInt d_nz,const PetscInt d_nnz[],PetscInt o_nz,const PetscInt o_nnz[])
{
  Mat_MPIAIJ         *b = (Mat_MPIAIJ*)B->data;
  Mat_MPIAIJCUSPARSE *cusparseStruct = (Mat_MPIAIJCUSPARSE*)b->spptr;
  PetscErrorCode     ierr;
  PetscInt           i;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
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
  ierr = PetscTableDestroy(&b->colmap);CHKERRQ(ierr);
#else
  ierr = PetscFree(b->colmap);CHKERRQ(ierr);
#endif
  ierr = PetscFree(b->garray);CHKERRQ(ierr);
  ierr = VecDestroy(&b->lvec);CHKERRQ(ierr);
  ierr = VecScatterDestroy(&b->Mvctx);CHKERRQ(ierr);
  /* Because the B will have been resized we simply destroy it and create a new one each time */
  ierr = MatDestroy(&b->B);CHKERRQ(ierr);
  if (!b->A) {
    ierr = MatCreate(PETSC_COMM_SELF,&b->A);CHKERRQ(ierr);
    ierr = MatSetSizes(b->A,B->rmap->n,B->cmap->n,B->rmap->n,B->cmap->n);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)B,(PetscObject)b->A);CHKERRQ(ierr);
  }
  if (!b->B) {
    PetscMPIInt size;
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)B),&size);CHKERRMPI(ierr);
    ierr = MatCreate(PETSC_COMM_SELF,&b->B);CHKERRQ(ierr);
    ierr = MatSetSizes(b->B,B->rmap->n,size > 1 ? B->cmap->N : 0,B->rmap->n,size > 1 ? B->cmap->N : 0);CHKERRQ(ierr);
    ierr = PetscLogObjectParent((PetscObject)B,(PetscObject)b->B);CHKERRQ(ierr);
  }
  ierr = MatSetType(b->A,MATSEQAIJCUSPARSE);CHKERRQ(ierr);
  ierr = MatSetType(b->B,MATSEQAIJCUSPARSE);CHKERRQ(ierr);
  ierr = MatBindToCPU(b->A,B->boundtocpu);CHKERRQ(ierr);
  ierr = MatBindToCPU(b->B,B->boundtocpu);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(b->A,d_nz,d_nnz);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(b->B,o_nz,o_nnz);CHKERRQ(ierr);
  ierr = MatCUSPARSESetFormat(b->A,MAT_CUSPARSE_MULT,cusparseStruct->diagGPUMatFormat);CHKERRQ(ierr);
  ierr = MatCUSPARSESetFormat(b->B,MAT_CUSPARSE_MULT,cusparseStruct->offdiagGPUMatFormat);CHKERRQ(ierr);
  ierr = MatCUSPARSESetHandle(b->A,cusparseStruct->handle);CHKERRQ(ierr);
  ierr = MatCUSPARSESetHandle(b->B,cusparseStruct->handle);CHKERRQ(ierr);
  B->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_MPIAIJCUSPARSE(Mat A,Vec xx,Vec yy)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->A->ops->mult)(a->A,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,yy,yy);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatZeroEntries_MPIAIJCUSPARSE(Mat A)
{
  Mat_MPIAIJ     *l = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatZeroEntries(l->A);CHKERRQ(ierr);
  ierr = MatZeroEntries(l->B);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultAdd_MPIAIJCUSPARSE(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecScatterBegin(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->A->ops->multadd)(a->A,xx,yy,zz);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,xx,a->lvec,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
  ierr = (*a->B->ops->multadd)(a->B,a->lvec,zz,zz);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatMultTranspose_MPIAIJCUSPARSE(Mat A,Vec xx,Vec yy)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = (*a->B->ops->multtranspose)(a->B,xx,a->lvec);CHKERRQ(ierr);
  ierr = (*a->A->ops->multtranspose)(a->A,xx,yy);CHKERRQ(ierr);
  ierr = VecScatterBegin(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
  ierr = VecScatterEnd(a->Mvctx,a->lvec,yy,ADD_VALUES,SCATTER_REVERSE);CHKERRQ(ierr);
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
  PetscErrorCode           ierr;
  PetscBool                flg;
  Mat_MPIAIJ               *a = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJCUSPARSE       *cusparseStruct = (Mat_MPIAIJCUSPARSE*)a->spptr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"MPIAIJCUSPARSE options");CHKERRQ(ierr);
  if (A->factortype==MAT_FACTOR_NONE) {
    ierr = PetscOptionsEnum("-mat_cusparse_mult_diag_storage_format","sets storage format of the diagonal blocks of (mpi)aijcusparse gpu matrices for SpMV",
                            "MatCUSPARSESetFormat",MatCUSPARSEStorageFormats,(PetscEnum)cusparseStruct->diagGPUMatFormat,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatCUSPARSESetFormat(A,MAT_CUSPARSE_MULT_DIAG,format);CHKERRQ(ierr);
    }
    ierr = PetscOptionsEnum("-mat_cusparse_mult_offdiag_storage_format","sets storage format of the off-diagonal blocks (mpi)aijcusparse gpu matrices for SpMV",
                            "MatCUSPARSESetFormat",MatCUSPARSEStorageFormats,(PetscEnum)cusparseStruct->offdiagGPUMatFormat,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatCUSPARSESetFormat(A,MAT_CUSPARSE_MULT_OFFDIAG,format);CHKERRQ(ierr);
    }
    ierr = PetscOptionsEnum("-mat_cusparse_storage_format","sets storage format of the diagonal and off-diagonal blocks (mpi)aijcusparse gpu matrices for SpMV",
                            "MatCUSPARSESetFormat",MatCUSPARSEStorageFormats,(PetscEnum)cusparseStruct->diagGPUMatFormat,(PetscEnum*)&format,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = MatCUSPARSESetFormat(A,MAT_CUSPARSE_ALL,format);CHKERRQ(ierr);
    }
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatAssemblyEnd_MPIAIJCUSPARSE(Mat A,MatAssemblyType mode)
{
  PetscErrorCode     ierr;
  Mat_MPIAIJ         *mpiaij = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJCUSPARSE *cusp = (Mat_MPIAIJCUSPARSE*)mpiaij->spptr;
  PetscObjectState   onnz = A->nonzerostate;

  PetscFunctionBegin;
  ierr = MatAssemblyEnd_MPIAIJ(A,mode);CHKERRQ(ierr);
  if (mpiaij->lvec) { ierr = VecSetType(mpiaij->lvec,VECSEQCUDA);CHKERRQ(ierr); }
  if (onnz != A->nonzerostate && cusp->deviceMat) {
    PetscSplitCSRDataStructure d_mat = cusp->deviceMat, h_mat;
    cudaError_t                cerr;

    ierr = PetscInfo(A,"Destroy device mat since nonzerostate changed\n");CHKERRQ(ierr);
    ierr = PetscNew(&h_mat);CHKERRQ(ierr);
    cerr = cudaMemcpy(h_mat,d_mat,sizeof(*d_mat),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
    cerr = cudaFree(h_mat->colmap);CHKERRCUDA(cerr);
    if (h_mat->allocated_indices) {
      cerr = cudaFree(h_mat->diag.i);CHKERRCUDA(cerr);
      cerr = cudaFree(h_mat->diag.j);CHKERRCUDA(cerr);
      if (h_mat->offdiag.j) {
        cerr = cudaFree(h_mat->offdiag.i);CHKERRCUDA(cerr);
        cerr = cudaFree(h_mat->offdiag.j);CHKERRCUDA(cerr);
      }
    }
    cerr = cudaFree(d_mat);CHKERRCUDA(cerr);
    ierr = PetscFree(h_mat);CHKERRQ(ierr);
    cusp->deviceMat = NULL;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MPIAIJCUSPARSE(Mat A)
{
  PetscErrorCode     ierr;
  cudaError_t        cerr;
  Mat_MPIAIJ         *aij            = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJCUSPARSE *cusparseStruct = (Mat_MPIAIJCUSPARSE*)aij->spptr;
  cusparseStatus_t   stat;

  PetscFunctionBegin;
  PetscCheckFalse(!cusparseStruct,PETSC_COMM_SELF,PETSC_ERR_COR,"Missing spptr");
  if (cusparseStruct->deviceMat) {
    PetscSplitCSRDataStructure d_mat = cusparseStruct->deviceMat, h_mat;

    ierr = PetscInfo(A,"Have device matrix\n");CHKERRQ(ierr);
    ierr = PetscNew(&h_mat);CHKERRQ(ierr);
    cerr = cudaMemcpy(h_mat,d_mat,sizeof(*d_mat),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
    cerr = cudaFree(h_mat->colmap);CHKERRCUDA(cerr);
    if (h_mat->allocated_indices) {
      cerr = cudaFree(h_mat->diag.i);CHKERRCUDA(cerr);
      cerr = cudaFree(h_mat->diag.j);CHKERRCUDA(cerr);
      if (h_mat->offdiag.j) {
        cerr = cudaFree(h_mat->offdiag.i);CHKERRCUDA(cerr);
        cerr = cudaFree(h_mat->offdiag.j);CHKERRCUDA(cerr);
      }
    }
    cerr = cudaFree(d_mat);CHKERRCUDA(cerr);
    ierr = PetscFree(h_mat);CHKERRQ(ierr);
  }
  try {
    if (aij->A) { ierr = MatCUSPARSEClearHandle(aij->A);CHKERRQ(ierr); }
    if (aij->B) { ierr = MatCUSPARSEClearHandle(aij->B);CHKERRQ(ierr); }
    stat = cusparseDestroy(cusparseStruct->handle);CHKERRCUSPARSE(stat);
    /* We want cusparseStruct to use PetscDefaultCudaStream
    if (cusparseStruct->stream) {
      cudaError_t err = cudaStreamDestroy(cusparseStruct->stream);CHKERRCUDA(err);
    }
    */
    /* Free COO */
    ierr = MatResetPreallocationCOO_MPIAIJCUSPARSE(A);CHKERRQ(ierr);
    delete cusparseStruct;
  } catch(char *ex) {
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Mat_MPIAIJCUSPARSE error: %s", ex);
  }
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJGetLocalMatMerge_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOO_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatCUSPARSESetFormat_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_mpiaijcusparse_hypre_C",NULL);CHKERRQ(ierr);
  ierr = MatDestroy_MPIAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIAIJCUSPARSE(Mat B, MatType mtype, MatReuse reuse, Mat* newmat)
{
  PetscErrorCode     ierr;
  Mat_MPIAIJ         *a;
  cusparseStatus_t   stat;
  Mat                A;

  PetscFunctionBegin;
  ierr = PetscDeviceInitialize(PETSC_DEVICE_CUDA);CHKERRQ(ierr);
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(B,MAT_COPY_VALUES,newmat);CHKERRQ(ierr);
  } else if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatCopy(B,*newmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  A = *newmat;
  A->boundtocpu = PETSC_FALSE;
  ierr = PetscFree(A->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECCUDA,&A->defaultvectype);CHKERRQ(ierr);

  a = (Mat_MPIAIJ*)A->data;
  if (a->A) { ierr = MatSetType(a->A,MATSEQAIJCUSPARSE);CHKERRQ(ierr); }
  if (a->B) { ierr = MatSetType(a->B,MATSEQAIJCUSPARSE);CHKERRQ(ierr); }
  if (a->lvec) {
    ierr = VecSetType(a->lvec,VECSEQCUDA);CHKERRQ(ierr);
  }

  if (reuse != MAT_REUSE_MATRIX && !a->spptr) {
    Mat_MPIAIJCUSPARSE *cusp = new Mat_MPIAIJCUSPARSE;
    stat     = cusparseCreate(&(cusp->handle));CHKERRCUSPARSE(stat);
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

  ierr = PetscObjectChangeTypeName((PetscObject)A,MATMPIAIJCUSPARSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJGetLocalMatMerge_C",MatMPIAIJGetLocalMatMerge_MPIAIJCUSPARSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",MatMPIAIJSetPreallocation_MPIAIJCUSPARSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatCUSPARSESetFormat_C",MatCUSPARSESetFormat_MPIAIJCUSPARSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOO_C",MatSetPreallocationCOO_MPIAIJCUSPARSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",MatSetValuesCOO_MPIAIJCUSPARSE);CHKERRQ(ierr);
#if defined(PETSC_HAVE_HYPRE)
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatConvert_mpiaijcusparse_hypre_C",MatConvert_AIJ_HYPRE);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJCUSPARSE(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDeviceInitialize(PETSC_DEVICE_CUDA);CHKERRQ(ierr);
  ierr = MatCreate_MPIAIJ(A);CHKERRQ(ierr);
  ierr = MatConvert_MPIAIJ_MPIAIJCUSPARSE(A,MATMPIAIJCUSPARSE,MAT_INPLACE_MATRIX,&A);CHKERRQ(ierr);
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
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
  ierr = MatCreate(comm,A);CHKERRQ(ierr);
  ierr = MatSetSizes(*A,m,n,M,N);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size > 1) {
    ierr = MatSetType(*A,MATMPIAIJCUSPARSE);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(*A,d_nz,d_nnz,o_nz,o_nnz);CHKERRQ(ierr);
  } else {
    ierr = MatSetType(*A,MATSEQAIJCUSPARSE);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(*A,d_nz,d_nnz);CHKERRQ(ierr);
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
  PetscErrorCode             ierr;
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
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRMPI(ierr);
  // get jaca
  if (size == 1) {
    PetscBool isseqaij;

    ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATSEQAIJ,&isseqaij);CHKERRQ(ierr);
    if (isseqaij) {
      jaca = (Mat_SeqAIJ*)A->data;
      PetscCheckFalse(!jaca->roworiented,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Device assembly does not currently support column oriented values insertion");
      cusparsestructA = (Mat_SeqAIJCUSPARSE*)A->spptr;
      d_mat = cusparsestructA->deviceMat;
      ierr = MatSeqAIJCUSPARSECopyToGPU(A);CHKERRQ(ierr);
    } else {
      Mat_MPIAIJ *aij = (Mat_MPIAIJ*)A->data;
      PetscCheckFalse(!aij->roworiented,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Device assembly does not currently support column oriented values insertion");
      Mat_MPIAIJCUSPARSE *spptr = (Mat_MPIAIJCUSPARSE*)aij->spptr;
      jaca = (Mat_SeqAIJ*)aij->A->data;
      cusparsestructA = (Mat_SeqAIJCUSPARSE*)aij->A->spptr;
      d_mat = spptr->deviceMat;
      ierr = MatSeqAIJCUSPARSECopyToGPU(aij->A);CHKERRQ(ierr);
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
      ierr = MatSeqAIJCUSPARSECopyToGPU(aij->A);CHKERRQ(ierr);
      ierr = MatSeqAIJCUSPARSECopyToGPU(aij->B);CHKERRQ(ierr);
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
    cudaError_t                cerr;
    PetscSplitCSRDataStructure h_mat;

    // create and populate strucy on host and copy on device
    ierr = PetscInfo(A,"Create device matrix\n");CHKERRQ(ierr);
    ierr = PetscNew(&h_mat);CHKERRQ(ierr);
    cerr = cudaMalloc((void**)&d_mat,sizeof(*d_mat));CHKERRCUDA(cerr);
    if (size > 1) { /* need the colmap array */
      Mat_MPIAIJ *aij = (Mat_MPIAIJ*)A->data;
      PetscInt   *colmap;
      PetscInt   ii,n = aij->B->cmap->n,N = A->cmap->N;

      PetscCheckFalse(n && !aij->garray,PETSC_COMM_SELF,PETSC_ERR_PLIB,"MPIAIJ Matrix was assembled but is missing garray");

      ierr = PetscCalloc1(N+1,&colmap);CHKERRQ(ierr);
      for (ii=0; ii<n; ii++) colmap[aij->garray[ii]] = (int)(ii+1);
#if defined(PETSC_USE_64BIT_INDICES)
      { // have to make a long version of these
        int        *h_bi32, *h_bj32;
        PetscInt   *h_bi64, *h_bj64, *d_bi64, *d_bj64;
        ierr = PetscCalloc4(A->rmap->n+1,&h_bi32,jacb->nz,&h_bj32,A->rmap->n+1,&h_bi64,jacb->nz,&h_bj64);CHKERRQ(ierr);
        cerr = cudaMemcpy(h_bi32, bi, (A->rmap->n+1)*sizeof(*h_bi32),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
        for (int i=0;i<A->rmap->n+1;i++) h_bi64[i] = h_bi32[i];
        cerr = cudaMemcpy(h_bj32, bj, jacb->nz*sizeof(*h_bj32),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
        for (int i=0;i<jacb->nz;i++) h_bj64[i] = h_bj32[i];

        cerr = cudaMalloc((void**)&d_bi64,(A->rmap->n+1)*sizeof(*d_bi64));CHKERRCUDA(cerr);
        cerr = cudaMemcpy(d_bi64, h_bi64,(A->rmap->n+1)*sizeof(*d_bi64),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
        cerr = cudaMalloc((void**)&d_bj64,jacb->nz*sizeof(*d_bj64));CHKERRCUDA(cerr);
        cerr = cudaMemcpy(d_bj64, h_bj64,jacb->nz*sizeof(*d_bj64),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);

        h_mat->offdiag.i = d_bi64;
        h_mat->offdiag.j = d_bj64;
        h_mat->allocated_indices = PETSC_TRUE;

        ierr = PetscFree4(h_bi32,h_bj32,h_bi64,h_bj64);CHKERRQ(ierr);
      }
#else
      h_mat->offdiag.i = (PetscInt*)bi;
      h_mat->offdiag.j = (PetscInt*)bj;
      h_mat->allocated_indices = PETSC_FALSE;
#endif
      h_mat->offdiag.a = ba;
      h_mat->offdiag.n = A->rmap->n;

      cerr = cudaMalloc((void**)&h_mat->colmap,(N+1)*sizeof(*h_mat->colmap));CHKERRCUDA(cerr);
      cerr = cudaMemcpy(h_mat->colmap,colmap,(N+1)*sizeof(*h_mat->colmap),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
      ierr = PetscFree(colmap);CHKERRQ(ierr);
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
      ierr = PetscCalloc4(A->rmap->n+1,&h_ai32,jaca->nz,&h_aj32,A->rmap->n+1,&h_ai64,jaca->nz,&h_aj64);CHKERRQ(ierr);
      cerr = cudaMemcpy(h_ai32, ai, (A->rmap->n+1)*sizeof(*h_ai32),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
      for (int i=0;i<A->rmap->n+1;i++) h_ai64[i] = h_ai32[i];
      cerr = cudaMemcpy(h_aj32, aj, jaca->nz*sizeof(*h_aj32),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
      for (int i=0;i<jaca->nz;i++) h_aj64[i] = h_aj32[i];

      cerr = cudaMalloc((void**)&d_ai64,(A->rmap->n+1)*sizeof(*d_ai64));CHKERRCUDA(cerr);
      cerr = cudaMemcpy(d_ai64, h_ai64,(A->rmap->n+1)*sizeof(*d_ai64),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
      cerr = cudaMalloc((void**)&d_aj64,jaca->nz*sizeof(*d_aj64));CHKERRCUDA(cerr);
      cerr = cudaMemcpy(d_aj64, h_aj64,jaca->nz*sizeof(*d_aj64),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);

      h_mat->diag.i = d_ai64;
      h_mat->diag.j = d_aj64;
      h_mat->allocated_indices = PETSC_TRUE;

      ierr = PetscFree4(h_ai32,h_aj32,h_ai64,h_aj64);CHKERRQ(ierr);
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
    cerr = cudaMemcpy(d_mat,h_mat,sizeof(*d_mat),cudaMemcpyHostToDevice);CHKERRCUDA(cerr);
    ierr = PetscFree(h_mat);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(A,"Reusing device matrix\n");CHKERRQ(ierr);
  }
  *B = d_mat;
  A->offloadmask = PETSC_OFFLOAD_GPU;
  PetscFunctionReturn(0);
}
