#define PETSC_SKIP_SPINLOCK
#define PETSC_SKIP_CXX_COMPLEX_FIX
#define PETSC_SKIP_IMMINTRIN_H_CUDAWORKAROUND 1

#include <petscconf.h>
#include <../src/mat/impls/aij/mpi/mpiaij.h>   /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/seq/seqcusparse/cusparsematimpl.h>
#include <../src/mat/impls/aij/mpi/mpicusparse/mpicusparsematimpl.h>
#include <thrust/advance.h>
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

static PetscErrorCode MatSetValuesCOO_MPIAIJCUSPARSE(Mat A, const PetscScalar v[], InsertMode imode)
{
  Mat_MPIAIJ         *a = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJCUSPARSE *cusp = (Mat_MPIAIJCUSPARSE*)a->spptr;
  PetscInt           n = cusp->coo_nd + cusp->coo_no;
  PetscErrorCode     ierr;
  cudaError_t        cerr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(MAT_CUSPARSESetVCOO,A,0,0,0);CHKERRQ(ierr);
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
    cerr = WaitForCUDA();CHKERRCUDA(cerr);
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    delete w;
    ierr = MatSetValuesCOO_SeqAIJCUSPARSE(a->A,cusp->coo_pw->data().get(),imode);CHKERRQ(ierr);
    ierr = MatSetValuesCOO_SeqAIJCUSPARSE(a->B,cusp->coo_pw->data().get()+cusp->coo_nd,imode);CHKERRQ(ierr);
  } else {
    ierr = MatSetValuesCOO_SeqAIJCUSPARSE(a->A,v,imode);CHKERRQ(ierr);
    ierr = MatSetValuesCOO_SeqAIJCUSPARSE(a->B,v ? v+cusp->coo_nd : NULL,imode);CHKERRQ(ierr);
  }
  ierr = PetscLogEventEnd(MAT_CUSPARSESetVCOO,A,0,0,0);CHKERRQ(ierr);
  ierr = PetscObjectStateIncrease((PetscObject)A);CHKERRQ(ierr);
  A->num_ass++;
  A->assembled        = PETSC_TRUE;
  A->ass_nonzerostate = A->nonzerostate;
  A->offloadmask      = PETSC_OFFLOAD_GPU;
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

static PetscErrorCode MatSetPreallocationCOO_MPIAIJCUSPARSE(Mat B, PetscInt n, const PetscInt coo_i[], const PetscInt coo_j[])
{
  Mat_MPIAIJ             *b = (Mat_MPIAIJ*)B->data;
  Mat_MPIAIJCUSPARSE     *cusp = (Mat_MPIAIJCUSPARSE*)b->spptr;
  PetscErrorCode         ierr;
  PetscInt               *jj;
  size_t                 noff = 0;
  THRUSTINTARRAY         d_i(n);
  THRUSTINTARRAY         d_j(n);
  ISLocalToGlobalMapping l2g;
  cudaError_t            cerr;

  PetscFunctionBegin;
  ierr = PetscLogEventBegin(MAT_CUSPARSEPreallCOO,B,0,0,0);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(B->cmap);CHKERRQ(ierr);
  if (b->A) { ierr = MatCUSPARSEClearHandle(b->A);CHKERRQ(ierr); }
  if (b->B) { ierr = MatCUSPARSEClearHandle(b->B);CHKERRQ(ierr); }
  ierr = PetscFree(b->garray);CHKERRQ(ierr);
  ierr = VecDestroy(&b->lvec);CHKERRQ(ierr);
  ierr = MatDestroy(&b->A);CHKERRQ(ierr);
  ierr = MatDestroy(&b->B);CHKERRQ(ierr);

  ierr = PetscLogCpuToGpu(2.*n*sizeof(PetscInt));CHKERRQ(ierr);
  d_i.assign(coo_i,coo_i+n);
  d_j.assign(coo_j,coo_j+n);
  delete cusp->coo_p;
  delete cusp->coo_pw;
  cusp->coo_p = NULL;
  cusp->coo_pw = NULL;
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
  cerr = WaitForCUDA();CHKERRCUDA(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);

  /* copy offdiag column indices to map on the CPU */
  ierr = PetscMalloc1(cusp->coo_no,&jj);CHKERRQ(ierr);
  cerr = cudaMemcpy(jj,d_j.data().get()+cusp->coo_nd,cusp->coo_no*sizeof(PetscInt),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
  auto o_j = d_j.begin();
  ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
  thrust::advance(o_j,cusp->coo_nd);
  thrust::sort(thrust::device,o_j,d_j.end());
  auto wit = thrust::unique(thrust::device,o_j,d_j.end());
  cerr = WaitForCUDA();CHKERRCUDA(cerr);
  ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
  noff = thrust::distance(o_j,wit);
  ierr = PetscMalloc1(noff+1,&b->garray);CHKERRQ(ierr);
  cerr = cudaMemcpy(b->garray,d_j.data().get()+cusp->coo_nd,noff*sizeof(PetscInt),cudaMemcpyDeviceToHost);CHKERRCUDA(cerr);
  ierr = PetscLogGpuToCpu((noff+cusp->coo_no)*sizeof(PetscInt));CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingCreate(PETSC_COMM_SELF,1,noff,b->garray,PETSC_COPY_VALUES,&l2g);CHKERRQ(ierr);
  ierr = ISLocalToGlobalMappingSetType(l2g,ISLOCALTOGLOBALMAPPINGHASH);CHKERRQ(ierr);
  ierr = ISGlobalToLocalMappingApply(l2g,IS_GTOLM_DROP,cusp->coo_no,jj,&n,jj);CHKERRQ(ierr);
  if (n != cusp->coo_no) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unexpected is size %D != %D coo size",n,cusp->coo_no);
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
  ierr = MatSetPreallocationCOO_SeqAIJCUSPARSE(b->A,cusp->coo_nd,d_i.data().get(),d_j.data().get());CHKERRQ(ierr);
  ierr = MatSetPreallocationCOO_SeqAIJCUSPARSE(b->B,cusp->coo_no,d_i.data().get()+cusp->coo_nd,jj);CHKERRQ(ierr);
  ierr = PetscFree(jj);CHKERRQ(ierr);

  ierr = MatCUSPARSESetFormat(b->A,MAT_CUSPARSE_MULT,cusp->diagGPUMatFormat);CHKERRQ(ierr);
  ierr = MatCUSPARSESetFormat(b->B,MAT_CUSPARSE_MULT,cusp->offdiagGPUMatFormat);CHKERRQ(ierr);
  ierr = MatCUSPARSESetHandle(b->A,cusp->handle);CHKERRQ(ierr);
  ierr = MatCUSPARSESetHandle(b->B,cusp->handle);CHKERRQ(ierr);
  ierr = MatCUSPARSESetStream(b->A,cusp->stream);CHKERRQ(ierr);
  ierr = MatCUSPARSESetStream(b->B,cusp->stream);CHKERRQ(ierr);
  ierr = MatSetUpMultiply_MPIAIJ(B);CHKERRQ(ierr);
  B->preallocated = PETSC_TRUE;
  B->nonzerostate++;
  ierr = PetscLogEventEnd(MAT_CUSPARSEPreallCOO,B,0,0,0);CHKERRQ(ierr);

  ierr = MatBindToCPU(b->A,B->boundtocpu);CHKERRQ(ierr);
  ierr = MatBindToCPU(b->B,B->boundtocpu);CHKERRQ(ierr);
  B->offloadmask = PETSC_OFFLOAD_CPU;
  B->assembled = PETSC_FALSE;
  B->was_assembled = PETSC_FALSE;
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
      if (d_nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"d_nnz cannot be less than 0: local row %D value %D",i,d_nnz[i]);
    }
  }
  if (PetscDefined(USE_DEBUG) && o_nnz) {
    for (i=0; i<B->rmap->n; i++) {
      if (o_nnz[i] < 0) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"o_nnz cannot be less than 0: local row %D value %D",i,o_nnz[i]);
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
    ierr = MPI_Comm_size(PetscObjectComm((PetscObject)B),&size);CHKERRQ(ierr);
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
  ierr = MatCUSPARSESetStream(b->A,cusparseStruct->stream);CHKERRQ(ierr);
  ierr = MatCUSPARSESetStream(b->B,cusparseStruct->stream);CHKERRQ(ierr);

  B->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

typedef struct {
  Mat         *mp;    /* intermediate products */
  PetscBool   *mptmp; /* is the intermediate product temporary ? */
  PetscInt    cp;     /* number of intermediate products */

  /* support for MatGetBrowsOfAoCols_MPIAIJ for P_oth */
  PetscInt    *startsj_s,*startsj_r;
  PetscScalar *bufa;
  Mat         P_oth;

  /* may take advantage of merging product->B */
  Mat Bloc;

  /* cusparse does not have support to split between symbolic and numeric phases
     When api_user is true, we don't need to update the numerical values
     of the temporary storage */
  PetscBool reusesym;

  /* support for COO values insertion */
  PetscScalar *coo_v,*coo_w;
  PetscSF     sf; /* if present, non-local values insertion (i.e. AtB or PtAP) */
} MatMatMPIAIJCUSPARSE;

PetscErrorCode MatDestroy_MatMatMPIAIJCUSPARSE(void *data)
{
  MatMatMPIAIJCUSPARSE *mmdata = (MatMatMPIAIJCUSPARSE*)data;
  PetscInt             i;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  ierr = PetscFree2(mmdata->startsj_s,mmdata->startsj_r);CHKERRQ(ierr);
  ierr = PetscFree(mmdata->bufa);CHKERRQ(ierr);
  ierr = PetscFree(mmdata->coo_v);CHKERRQ(ierr);
  ierr = PetscFree(mmdata->coo_w);CHKERRQ(ierr);
  ierr = MatDestroy(&mmdata->P_oth);CHKERRQ(ierr);
  ierr = MatDestroy(&mmdata->Bloc);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&mmdata->sf);CHKERRQ(ierr);
  for (i = 0; i < mmdata->cp; i++) {
    ierr = MatDestroy(&mmdata->mp[i]);CHKERRQ(ierr);
  }
  ierr = PetscFree(mmdata->mp);CHKERRQ(ierr);
  ierr = PetscFree(mmdata->mptmp);CHKERRQ(ierr);
  ierr = PetscFree(mmdata);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductNumeric_MPIAIJCUSPARSE_MPIAIJCUSPARSE(Mat C)
{
  MatMatMPIAIJCUSPARSE *mmdata;
  PetscScalar          *tmp;
  PetscInt             i,n;
  PetscErrorCode       ierr;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  if (!C->product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data empty");
  mmdata = (MatMatMPIAIJCUSPARSE*)C->product->data;
  tmp = mmdata->sf ? mmdata->coo_w : mmdata->coo_v;
  if (!mmdata->reusesym) { /* update temporary matrices */
    if (mmdata->P_oth) {
      ierr = MatGetBrowsOfAoCols_MPIAIJ(C->product->A,C->product->B,MAT_REUSE_MATRIX,&mmdata->startsj_s,&mmdata->startsj_r,&mmdata->bufa,&mmdata->P_oth);CHKERRQ(ierr);
    }
    if (mmdata->Bloc) {
      ierr = MatMPIAIJGetLocalMatMerge(C->product->B,MAT_REUSE_MATRIX,NULL,&mmdata->Bloc);CHKERRQ(ierr);
    }
  }
  mmdata->reusesym = PETSC_FALSE;
  for (i = 0, n = 0; i < mmdata->cp; i++) {
    Mat_SeqAIJ        *mm = (Mat_SeqAIJ*)mmdata->mp[i]->data;
    const PetscScalar *vv;

    if (!mmdata->mp[i]->ops->productnumeric) SETERRQ1(PetscObjectComm((PetscObject)mmdata->mp[i]),PETSC_ERR_PLIB,"Missing numeric op for %s",MatProductTypes[mmdata->mp[i]->product->type]);
    ierr = (*mmdata->mp[i]->ops->productnumeric)(mmdata->mp[i]);CHKERRQ(ierr);
    if (mmdata->mptmp[i]) continue;
    /* TODO: add support for using GPU data directly */
    ierr = MatSeqAIJGetArrayRead(mmdata->mp[i],&vv);CHKERRQ(ierr);
    ierr = PetscArraycpy(tmp + n,vv,mm->nz);CHKERRQ(ierr);
    ierr = MatSeqAIJRestoreArrayRead(mmdata->mp[i],&vv);CHKERRQ(ierr);
    n   += mm->nz;
  }
  if (mmdata->sf) { /* offprocess insertion */
    ierr = PetscSFGatherBegin(mmdata->sf,MPIU_SCALAR,tmp,mmdata->coo_v);CHKERRQ(ierr);
    ierr = PetscSFGatherEnd(mmdata->sf,MPIU_SCALAR,tmp,mmdata->coo_v);CHKERRQ(ierr);
  }
  ierr = MatSetValuesCOO(C,mmdata->coo_v,INSERT_VALUES);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Pt * A or A * P */
#define MAX_NUMBER_INTERMEDIATE 4
static PetscErrorCode MatProductSymbolic_MPIAIJCUSPARSE_MPIAIJCUSPARSE(Mat C)
{
  Mat_Product            *product = C->product;
  Mat                    A,P,mp[MAX_NUMBER_INTERMEDIATE];
  Mat_MPIAIJ             *a,*p;
  MatMatMPIAIJCUSPARSE   *mmdata;
  ISLocalToGlobalMapping P_oth_l2g = NULL;
  IS                     glob = NULL;
  const PetscInt         *globidx,*P_oth_idx;
  const PetscInt         *cmapa[MAX_NUMBER_INTERMEDIATE],*rmapa[MAX_NUMBER_INTERMEDIATE];
  PetscInt               cp = 0,m,n,M,N,ncoo,*coo_i,*coo_j,cmapt[MAX_NUMBER_INTERMEDIATE],rmapt[MAX_NUMBER_INTERMEDIATE],i,j;
  MatProductType         ptype;
  PetscBool              mptmp[MAX_NUMBER_INTERMEDIATE],hasoffproc = PETSC_FALSE;
  PetscMPIInt            size;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  if (product->data) SETERRQ(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Product data not empty");
  ptype = product->type;
  if (product->A->symmetric && ptype == MATPRODUCT_AtB) ptype = MATPRODUCT_AB;
  switch (ptype) {
  case MATPRODUCT_AB:
    A = product->A;
    P = product->B;
    m = A->rmap->n;
    n = P->cmap->n;
    M = A->rmap->N;
    N = P->cmap->N;
    break;
  case MATPRODUCT_AtB:
    P = product->A;
    A = product->B;
    m = P->cmap->n;
    n = A->cmap->n;
    M = P->cmap->N;
    N = A->cmap->N;
    hasoffproc = PETSC_TRUE;
    break;
  case MATPRODUCT_PtAP:
    A = product->A;
    P = product->B;
    m = P->cmap->n;
    n = P->cmap->n;
    M = P->cmap->N;
    N = P->cmap->N;
    hasoffproc = PETSC_TRUE;
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Not for product type %s",MatProductTypes[ptype]);
  }
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)C),&size);CHKERRQ(ierr);
  if (size == 1) hasoffproc = PETSC_FALSE;

  for (i=0;i<MAX_NUMBER_INTERMEDIATE;i++) {
    mp[i]    = NULL;
    mptmp[i] = PETSC_FALSE;
    rmapt[i] = 0;
    cmapt[i] = 0;
    rmapa[i] = NULL;
    cmapa[i] = NULL;
  }

  a = (Mat_MPIAIJ*)A->data;
  p = (Mat_MPIAIJ*)P->data;
  ierr = MatSetSizes(C,m,n,M,N);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(C->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(C->cmap);CHKERRQ(ierr);
  ierr = MatSetType(C,MATMPIAIJCUSPARSE);CHKERRQ(ierr);
  ierr = PetscNew(&mmdata);CHKERRQ(ierr);
  mmdata->reusesym = product->api_user;
  switch (ptype) {
  case MATPRODUCT_AB: /* A * P */
    ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_INITIAL_MATRIX,&mmdata->startsj_s,&mmdata->startsj_r,&mmdata->bufa,&mmdata->P_oth);CHKERRQ(ierr);

    if (1) { /* A_diag * P_loc and A_off * P_oth TODO: add customization for this */
      /* P is product->B */
      ierr = MatMPIAIJGetLocalMatMerge(P,MAT_INITIAL_MATRIX,&glob,&mmdata->Bloc);CHKERRQ(ierr);
      ierr = MatProductCreate(a->A,mmdata->Bloc,NULL,&mp[cp]);CHKERRQ(ierr);
      ierr = MatProductSetType(mp[cp],MATPRODUCT_AB);CHKERRQ(ierr);
      ierr = MatProductSetFill(mp[cp],product->fill);CHKERRQ(ierr);
      mp[cp]->product->api_user = product->api_user;
      ierr = MatProductSetFromOptions(mp[cp]);CHKERRQ(ierr);
      if (!mp[cp]->ops->productsymbolic) SETERRQ1(PetscObjectComm((PetscObject)mp[cp]),PETSC_ERR_PLIB,"Missing symbolic op for %s",MatProductTypes[mp[cp]->product->type]);
      ierr = (*mp[cp]->ops->productsymbolic)(mp[cp]);CHKERRQ(ierr);
      ierr = ISGetIndices(glob,&globidx);CHKERRQ(ierr);
      rmapt[cp] = 1;
      cmapt[cp] = 2;
      cmapa[cp] = globidx;
      mptmp[cp] = PETSC_FALSE;
      cp++;
    } else { /* A_diag * P_diag and A_diag * P_off and A_off * P_oth */
      ierr = MatProductCreate(a->A,p->A,NULL,&mp[cp]);CHKERRQ(ierr);
      ierr = MatProductSetType(mp[cp],MATPRODUCT_AB);CHKERRQ(ierr);
      ierr = MatProductSetFill(mp[cp],product->fill);CHKERRQ(ierr);
      mp[cp]->product->api_user = product->api_user;
      ierr = MatProductSetFromOptions(mp[cp]);CHKERRQ(ierr);
      if (!mp[cp]->ops->productsymbolic) SETERRQ1(PetscObjectComm((PetscObject)mp[cp]),PETSC_ERR_PLIB,"Missing symbolic op for %s",MatProductTypes[mp[cp]->product->type]);
      ierr = (*mp[cp]->ops->productsymbolic)(mp[cp]);CHKERRQ(ierr);
      rmapt[cp] = 1;
      cmapt[cp] = 1;
      mptmp[cp] = PETSC_FALSE;
      cp++;
      ierr = MatProductCreate(a->A,p->B,NULL,&mp[cp]);CHKERRQ(ierr);
      ierr = MatProductSetType(mp[cp],MATPRODUCT_AB);CHKERRQ(ierr);
      ierr = MatProductSetFill(mp[cp],product->fill);CHKERRQ(ierr);
      mp[cp]->product->api_user = product->api_user;
      ierr = MatProductSetFromOptions(mp[cp]);CHKERRQ(ierr);
      if (!mp[cp]->ops->productsymbolic) SETERRQ1(PetscObjectComm((PetscObject)mp[cp]),PETSC_ERR_PLIB,"Missing symbolic op for %s",MatProductTypes[mp[cp]->product->type]);
      ierr = (*mp[cp]->ops->productsymbolic)(mp[cp]);CHKERRQ(ierr);
      rmapt[cp] = 1;
      cmapt[cp] = 2;
      cmapa[cp] = p->garray;
      mptmp[cp] = PETSC_FALSE;
      cp++;
    }
    if (mmdata->P_oth) {
      ierr = MatSeqAIJCompactOutExtraColumns_SeqAIJ(mmdata->P_oth,&P_oth_l2g);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingGetIndices(P_oth_l2g,&P_oth_idx);CHKERRQ(ierr);
      ierr = MatSetType(mmdata->P_oth,((PetscObject)(a->B))->type_name);CHKERRQ(ierr);
      ierr = MatProductCreate(a->B,mmdata->P_oth,NULL,&mp[cp]);CHKERRQ(ierr);
      ierr = MatProductSetType(mp[cp],MATPRODUCT_AB);CHKERRQ(ierr);
      ierr = MatProductSetFill(mp[cp],product->fill);CHKERRQ(ierr);
      mp[cp]->product->api_user = product->api_user;
      ierr = MatProductSetFromOptions(mp[cp]);CHKERRQ(ierr);
      if (!mp[cp]->ops->productsymbolic) SETERRQ1(PetscObjectComm((PetscObject)mp[cp]),PETSC_ERR_PLIB,"Missing symbolic op for %s",MatProductTypes[mp[cp]->product->type]);
      ierr = (*mp[cp]->ops->productsymbolic)(mp[cp]);CHKERRQ(ierr);
      rmapt[cp] = 1;
      cmapt[cp] = 2;
      cmapa[cp] = P_oth_idx;
      mptmp[cp] = PETSC_FALSE;
      cp++;
    }
    break;
  case MATPRODUCT_AtB: /* (P^t * A): P_diag * A_loc + P_off * A_loc */
    /* A is product->B */
    ierr = MatMPIAIJGetLocalMatMerge(A,MAT_INITIAL_MATRIX,&glob,&mmdata->Bloc);CHKERRQ(ierr);
    ierr = MatProductCreate(p->A,mmdata->Bloc,NULL,&mp[cp]);CHKERRQ(ierr);
    ierr = MatProductSetType(mp[cp],MATPRODUCT_AtB);CHKERRQ(ierr);
    ierr = MatProductSetFill(mp[cp],product->fill);CHKERRQ(ierr);
    mp[cp]->product->api_user = product->api_user;
    ierr = MatProductSetFromOptions(mp[cp]);CHKERRQ(ierr);
    if (!mp[cp]->ops->productsymbolic) SETERRQ1(PetscObjectComm((PetscObject)mp[cp]),PETSC_ERR_PLIB,"Missing symbolic op for %s",MatProductTypes[mp[cp]->product->type]);
    ierr = (*mp[cp]->ops->productsymbolic)(mp[cp]);CHKERRQ(ierr);
    ierr = ISGetIndices(glob,&globidx);CHKERRQ(ierr);
    rmapt[cp] = 1;
    cmapt[cp] = 2;
    cmapa[cp] = globidx;
    mptmp[cp] = PETSC_FALSE;
    cp++;
    ierr = MatProductCreate(p->B,mmdata->Bloc,NULL,&mp[cp]);CHKERRQ(ierr);
    ierr = MatProductSetType(mp[cp],MATPRODUCT_AtB);CHKERRQ(ierr);
    ierr = MatProductSetFill(mp[cp],product->fill);CHKERRQ(ierr);
    mp[cp]->product->api_user = product->api_user;
    ierr = MatProductSetFromOptions(mp[cp]);CHKERRQ(ierr);
    if (!mp[cp]->ops->productsymbolic) SETERRQ1(PetscObjectComm((PetscObject)mp[cp]),PETSC_ERR_PLIB,"Missing symbolic op for %s",MatProductTypes[mp[cp]->product->type]);
    ierr = (*mp[cp]->ops->productsymbolic)(mp[cp]);CHKERRQ(ierr);
    rmapt[cp] = 2;
    rmapa[cp] = p->garray;
    cmapt[cp] = 2;
    cmapa[cp] = globidx;
    mptmp[cp] = PETSC_FALSE;
    cp++;
    break;
  case MATPRODUCT_PtAP:
    ierr = MatGetBrowsOfAoCols_MPIAIJ(A,P,MAT_INITIAL_MATRIX,&mmdata->startsj_s,&mmdata->startsj_r,&mmdata->bufa,&mmdata->P_oth);CHKERRQ(ierr);
    /* P is product->B */
    ierr = MatMPIAIJGetLocalMatMerge(P,MAT_INITIAL_MATRIX,&glob,&mmdata->Bloc);CHKERRQ(ierr);
    ierr = MatProductCreate(a->A,mmdata->Bloc,NULL,&mp[cp]);CHKERRQ(ierr);
    ierr = MatProductSetType(mp[cp],MATPRODUCT_PtAP);CHKERRQ(ierr);
    ierr = MatProductSetFill(mp[cp],product->fill);CHKERRQ(ierr);
    mp[cp]->product->api_user = product->api_user;
    ierr = MatProductSetFromOptions(mp[cp]);CHKERRQ(ierr);
    if (!mp[cp]->ops->productsymbolic) SETERRQ1(PetscObjectComm((PetscObject)mp[cp]),PETSC_ERR_PLIB,"Missing symbolic op for %s",MatProductTypes[mp[cp]->product->type]);
    ierr = (*mp[cp]->ops->productsymbolic)(mp[cp]);CHKERRQ(ierr);
    ierr = ISGetIndices(glob,&globidx);CHKERRQ(ierr);
    rmapt[cp] = 2;
    rmapa[cp] = globidx;
    cmapt[cp] = 2;
    cmapa[cp] = globidx;
    mptmp[cp] = PETSC_FALSE;
    cp++;
    if (mmdata->P_oth) {
      ierr = MatSeqAIJCompactOutExtraColumns_SeqAIJ(mmdata->P_oth,&P_oth_l2g);CHKERRQ(ierr);
      ierr = MatSetType(mmdata->P_oth,((PetscObject)(a->B))->type_name);CHKERRQ(ierr);
      ierr = ISLocalToGlobalMappingGetIndices(P_oth_l2g,&P_oth_idx);CHKERRQ(ierr);
      ierr = MatProductCreate(a->B,mmdata->P_oth,NULL,&mp[cp]);CHKERRQ(ierr);
      ierr = MatProductSetType(mp[cp],MATPRODUCT_AB);CHKERRQ(ierr);
      ierr = MatProductSetFill(mp[cp],product->fill);CHKERRQ(ierr);
      mp[cp]->product->api_user = product->api_user;
      ierr = MatProductSetFromOptions(mp[cp]);CHKERRQ(ierr);
      if (!mp[cp]->ops->productsymbolic) SETERRQ1(PetscObjectComm((PetscObject)mp[cp]),PETSC_ERR_PLIB,"Missing symbolic op for %s",MatProductTypes[mp[cp]->product->type]);
      ierr = (*mp[cp]->ops->productsymbolic)(mp[cp]);CHKERRQ(ierr);
      mptmp[cp] = PETSC_TRUE;
      cp++;
      ierr = MatProductCreate(mmdata->Bloc,mp[1],NULL,&mp[cp]);CHKERRQ(ierr);
      ierr = MatProductSetType(mp[cp],MATPRODUCT_AtB);CHKERRQ(ierr);
      ierr = MatProductSetFill(mp[cp],product->fill);CHKERRQ(ierr);
      mp[cp]->product->api_user = product->api_user;
      ierr = MatProductSetFromOptions(mp[cp]);CHKERRQ(ierr);
      if (!mp[cp]->ops->productsymbolic) SETERRQ1(PetscObjectComm((PetscObject)mp[cp]),PETSC_ERR_PLIB,"Missing symbolic op for %s",MatProductTypes[mp[cp]->product->type]);
      ierr = (*mp[cp]->ops->productsymbolic)(mp[cp]);CHKERRQ(ierr);
      rmapt[cp] = 2;
      rmapa[cp] = globidx;
      cmapt[cp] = 2;
      cmapa[cp] = P_oth_idx;
      mptmp[cp] = PETSC_FALSE;
      cp++;
    }
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)C),PETSC_ERR_PLIB,"Not for product type %s",MatProductTypes[ptype]);
  }
  ierr = PetscMalloc1(cp,&mmdata->mp);CHKERRQ(ierr);
  for (i = 0; i < cp; i++) mmdata->mp[i] = mp[i];
  ierr = PetscMalloc1(cp,&mmdata->mptmp);CHKERRQ(ierr);
  for (i = 0; i < cp; i++) mmdata->mptmp[i] = mptmp[i];
  mmdata->cp = cp;
  C->product->data       = mmdata;
  C->product->destroy    = MatDestroy_MatMatMPIAIJCUSPARSE;
  C->ops->productnumeric = MatProductNumeric_MPIAIJCUSPARSE_MPIAIJCUSPARSE;

  /* prepare coo coordinates for values insertion */
  ncoo = 0;
  for (cp = 0; cp < mmdata->cp; cp++) {
    Mat_SeqAIJ *mm = (Mat_SeqAIJ*)mp[cp]->data;
    if (mptmp[cp]) continue;
    ncoo += mm->nz;
  }
  ierr = PetscMalloc2(ncoo,&coo_i,ncoo,&coo_j);CHKERRQ(ierr);
  ncoo = 0;
  for (cp = 0; cp < mmdata->cp; cp++) {
    Mat_SeqAIJ     *mm = (Mat_SeqAIJ*)mp[cp]->data;
    PetscInt       *coi = coo_i + ncoo;
    PetscInt       *coj = coo_j + ncoo;
    const PetscInt mr = mp[cp]->rmap->n;
    const PetscInt *jj  = mm->j;
    const PetscInt *ii  = mm->i;

    if (mptmp[cp]) continue;
    /* rows coo */
    if (!rmapt[cp]) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"This should not happen");
    else if (rmapt[cp] == 1) { /* local to global for owned rows of  */
      const PetscInt rs = C->rmap->rstart;
      for (i = 0; i < mr; i++) {
        const PetscInt gr = i + rs;
        for (j = ii[i]; j < ii[i+1]; j++) {
          coi[j] = gr;
        }
      }
    } else { /* offprocess */
      const PetscInt *rmap = rmapa[cp];
      for (i = 0; i < mr; i++) {
        const PetscInt gr = rmap[i];
        for (j = ii[i]; j < ii[i+1]; j++) {
          coi[j] = gr;
        }
      }
    }
    /* columns coo */
    if (!cmapt[cp]) {
      ierr = PetscArraycpy(coj,jj,mm->nz);CHKERRQ(ierr);
    } else if (cmapt[cp] == 1) { /* local to global for owned columns of P */
      const PetscInt cs = P->cmap->rstart;
      for (j = 0; j < mm->nz; j++) {
        coj[j] = jj[j] + cs;
      }
    } else { /* offdiag */
      const PetscInt *cmap = cmapa[cp];
      for (j = 0; j < mm->nz; j++) {
        coj[j] = cmap[jj[j]];
      }
    }
    ncoo += mm->nz;
  }
  if (glob) {
    ierr = ISRestoreIndices(glob,&globidx);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&glob);CHKERRQ(ierr);
  if (P_oth_l2g) {
    ierr = ISLocalToGlobalMappingRestoreIndices(P_oth_l2g,&P_oth_idx);CHKERRQ(ierr);
  }
  ierr = ISLocalToGlobalMappingDestroy(&P_oth_l2g);CHKERRQ(ierr);

  if (hasoffproc) { /* offproc values insertion */
    const PetscInt *sfdeg;
    const PetscInt n = P->cmap->n;
    PetscInt ncoo2,*coo_i2,*coo_j2;

    ierr = PetscSFCreate(PetscObjectComm((PetscObject)C),&mmdata->sf);CHKERRQ(ierr);
    ierr = PetscSFSetGraphLayout(mmdata->sf,P->cmap,ncoo,NULL,PETSC_OWN_POINTER,coo_i);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeBegin(mmdata->sf,&sfdeg);CHKERRQ(ierr);
    ierr = PetscSFComputeDegreeEnd(mmdata->sf,&sfdeg);CHKERRQ(ierr);
    for (i = 0, ncoo2 = 0; i < n; i++) ncoo2 += sfdeg[i];
    ierr = PetscMalloc2(ncoo2,&coo_i2,ncoo2,&coo_j2);CHKERRQ(ierr);
    ierr = PetscSFGatherBegin(mmdata->sf,MPIU_INT,coo_i,coo_i2);CHKERRQ(ierr);
    ierr = PetscSFGatherEnd(mmdata->sf,MPIU_INT,coo_i,coo_i2);CHKERRQ(ierr);
    ierr = PetscSFGatherBegin(mmdata->sf,MPIU_INT,coo_j,coo_j2);CHKERRQ(ierr);
    ierr = PetscSFGatherEnd(mmdata->sf,MPIU_INT,coo_j,coo_j2);CHKERRQ(ierr);
    ierr = PetscFree2(coo_i,coo_j);CHKERRQ(ierr);
    ierr = PetscMalloc1(ncoo,&mmdata->coo_w);CHKERRQ(ierr);
    coo_i = coo_i2;
    coo_j = coo_j2;
    ncoo  = ncoo2;
  }

  /* preallocate with COO data */
  ierr = MatSetPreallocationCOO(C,ncoo,coo_i,coo_j);CHKERRQ(ierr);
  ierr = PetscMalloc1(ncoo,&mmdata->coo_v);CHKERRQ(ierr);
  ierr = PetscFree2(coo_i,coo_j);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_MPIAIJCUSPARSE(Mat mat)
{
  Mat_Product    *product = mat->product;
  PetscErrorCode ierr;
  PetscBool      Biscusp = PETSC_FALSE;

  PetscFunctionBegin;
  MatCheckProduct(mat,1);
  if (!product->B->boundtocpu) {
    ierr = PetscObjectTypeCompare((PetscObject)product->B,MATMPIAIJCUSPARSE,&Biscusp);CHKERRQ(ierr);
  }
  if (Biscusp) {
    switch (product->type) {
    case MATPRODUCT_AB:
    case MATPRODUCT_AtB:
    case MATPRODUCT_PtAP:
      mat->ops->productsymbolic = MatProductSymbolic_MPIAIJCUSPARSE_MPIAIJCUSPARSE;
      break;
    default:
      break;
    }
  }
  if (!mat->ops->productsymbolic) {
    ierr = MatProductSetFromOptions_MPIAIJ(mat);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
   MatAIJCUSPARSESetGenerateTranspose - Sets the flag to explicitly generate the transpose matrix before calling MatMultTranspose

   Not collective

   Input Parameters:
+  A - Matrix of type SEQAIJCUSPARSE or MPIAIJCUSPARSE
-  gen - the boolean flag

   Level: intermediate

.seealso: MATSEQAIJCUSPARSE, MATMPIAIJCUSPARSE
@*/
PetscErrorCode  MatAIJCUSPARSESetGenerateTranspose(Mat A, PetscBool gen)
{
  PetscErrorCode ierr;
  PetscBool      ismpiaij;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  MatCheckPreallocated(A,1);
  ierr = PetscObjectBaseTypeCompare((PetscObject)A,MATMPIAIJ,&ismpiaij);CHKERRQ(ierr);
  if (ismpiaij) {
    Mat A_d,A_o;

    ierr = MatMPIAIJGetSeqAIJ(A,&A_d,&A_o,NULL);CHKERRQ(ierr);
    ierr = MatSeqAIJCUSPARSESetGenerateTranspose(A_d,gen);CHKERRQ(ierr);
    ierr = MatSeqAIJCUSPARSESetGenerateTranspose(A_o,gen);CHKERRQ(ierr);
  } else {
    ierr = MatSeqAIJCUSPARSESetGenerateTranspose(A,gen);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatMult_MPIAIJCUSPARSE(Mat A,Vec xx,Vec yy)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of A (%D) and xx (%D)",A->cmap->n,nt);
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
  if (A->factortype == MAT_FACTOR_NONE) {
    Mat_MPIAIJCUSPARSE *spptr = (Mat_MPIAIJCUSPARSE*)l->spptr;
    PetscSplitCSRDataStructure *d_mat = spptr->deviceMat;
    if (d_mat) {
      Mat_SeqAIJ   *a = (Mat_SeqAIJ*)l->A->data;
      Mat_SeqAIJ   *b = (Mat_SeqAIJ*)l->B->data;
      PetscInt     n = A->rmap->n, nnza = a->i[n], nnzb = b->i[n];
      cudaError_t  err;
      PetscScalar  *vals;
      ierr = PetscInfo(A,"Zero device matrix diag and offfdiag\n");CHKERRQ(ierr);
      err = cudaMemcpy( &vals, &d_mat->diag.a, sizeof(PetscScalar*), cudaMemcpyDeviceToHost);CHKERRCUDA(err);
      err = cudaMemset( vals, 0, (nnza)*sizeof(PetscScalar));CHKERRCUDA(err);
      err = cudaMemcpy( &vals, &d_mat->offdiag.a, sizeof(PetscScalar*), cudaMemcpyDeviceToHost);CHKERRCUDA(err);
      err = cudaMemset( vals, 0, (nnzb)*sizeof(PetscScalar));CHKERRCUDA(err);
    }
  }
  ierr = MatZeroEntries(l->A);CHKERRQ(ierr);
  ierr = MatZeroEntries(l->B);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
PetscErrorCode MatMultAdd_MPIAIJCUSPARSE(Mat A,Vec xx,Vec yy,Vec zz)
{
  Mat_MPIAIJ     *a = (Mat_MPIAIJ*)A->data;
  PetscErrorCode ierr;
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->cmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of A (%D) and xx (%D)",A->cmap->n,nt);
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
  PetscInt       nt;

  PetscFunctionBegin;
  ierr = VecGetLocalSize(xx,&nt);CHKERRQ(ierr);
  if (nt != A->rmap->n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Incompatible partition of A (%D) and xx (%D)",A->rmap->n,nt);
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
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"unsupported operation %d for MatCUSPARSEFormatOperation. Only MAT_CUSPARSE_MULT_DIAG, MAT_CUSPARSE_MULT_DIAG, and MAT_CUSPARSE_MULT_ALL are currently supported.",op);
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
  PetscErrorCode             ierr;
  Mat_MPIAIJ                 *mpiaij = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJCUSPARSE         *cusparseStruct = (Mat_MPIAIJCUSPARSE*)mpiaij->spptr;
  PetscSplitCSRDataStructure *d_mat = cusparseStruct->deviceMat;
  PetscFunctionBegin;
  ierr = MatAssemblyEnd_MPIAIJ(A,mode);CHKERRQ(ierr);
  if (!A->was_assembled && mode == MAT_FINAL_ASSEMBLY) {
    ierr = VecSetType(mpiaij->lvec,VECSEQCUDA);CHKERRQ(ierr);
  }
  if (d_mat) {
    A->offloadmask = PETSC_OFFLOAD_GPU; // if we assembled on the device
  }

  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_MPIAIJCUSPARSE(Mat A)
{
  PetscErrorCode     ierr;
  Mat_MPIAIJ         *aij            = (Mat_MPIAIJ*)A->data;
  Mat_MPIAIJCUSPARSE *cusparseStruct = (Mat_MPIAIJCUSPARSE*)aij->spptr;
  cudaError_t        err;
  cusparseStatus_t   stat;

  PetscFunctionBegin;
  if (!cusparseStruct) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_COR,"Missing spptr");
  if (cusparseStruct->deviceMat) {
    Mat_SeqAIJ                 *jaca = (Mat_SeqAIJ*)aij->A->data;
    Mat_SeqAIJ                 *jacb = (Mat_SeqAIJ*)aij->B->data;
    PetscSplitCSRDataStructure *d_mat = cusparseStruct->deviceMat, h_mat;
    ierr = PetscInfo(A,"Have device matrix\n");CHKERRQ(ierr);
    err = cudaMemcpy( &h_mat, d_mat, sizeof(PetscSplitCSRDataStructure), cudaMemcpyDeviceToHost);CHKERRCUDA(err);
    if (jaca->compressedrow.use) {
      err = cudaFree(h_mat.diag.i);CHKERRCUDA(err);
    }
    if (jacb->compressedrow.use) {
      err = cudaFree(h_mat.offdiag.i);CHKERRCUDA(err);
    }
    err = cudaFree(h_mat.colmap);CHKERRCUDA(err);
    err = cudaFree(d_mat);CHKERRCUDA(err);
  }
  try {
    if (aij->A) { ierr = MatCUSPARSEClearHandle(aij->A);CHKERRQ(ierr); }
    if (aij->B) { ierr = MatCUSPARSEClearHandle(aij->B);CHKERRQ(ierr); }
    stat = cusparseDestroy(cusparseStruct->handle);CHKERRCUSPARSE(stat);
    if (cusparseStruct->stream) {
      err = cudaStreamDestroy(cusparseStruct->stream);CHKERRCUDA(err);
    }
    delete cusparseStruct->coo_p;
    delete cusparseStruct->coo_pw;
    delete cusparseStruct;
  } catch(char *ex) {
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_LIB,"Mat_MPIAIJCUSPARSE error: %s", ex);
  }
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJGetLocalMatMerge_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOO_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatCUSPARSESetFormat_C",NULL);CHKERRQ(ierr);
  ierr = MatDestroy_MPIAIJ(A);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIAIJCUSPARSE(Mat B, MatType mtype, MatReuse reuse, Mat* newmat)
{
  PetscErrorCode     ierr;
  Mat_MPIAIJ         *a;
  Mat_MPIAIJCUSPARSE *cusparseStruct;
  cusparseStatus_t   stat;
  Mat                A;

  PetscFunctionBegin;
  if (reuse == MAT_INITIAL_MATRIX) {
    ierr = MatDuplicate(B,MAT_COPY_VALUES,newmat);CHKERRQ(ierr);
  } else if (reuse == MAT_REUSE_MATRIX) {
    ierr = MatCopy(B,*newmat,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  }
  A = *newmat;

  ierr = PetscFree(A->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECCUDA,&A->defaultvectype);CHKERRQ(ierr);

  a = (Mat_MPIAIJ*)A->data;
  if (a->A) { ierr = MatSetType(a->A,MATSEQAIJCUSPARSE);CHKERRQ(ierr); }
  if (a->B) { ierr = MatSetType(a->B,MATSEQAIJCUSPARSE);CHKERRQ(ierr); }
  if (a->lvec) {
    ierr = VecSetType(a->lvec,VECSEQCUDA);CHKERRQ(ierr);
  }

  if (reuse != MAT_REUSE_MATRIX && !a->spptr) {
    a->spptr = new Mat_MPIAIJCUSPARSE;

    cusparseStruct                      = (Mat_MPIAIJCUSPARSE*)a->spptr;
    cusparseStruct->diagGPUMatFormat    = MAT_CUSPARSE_CSR;
    cusparseStruct->offdiagGPUMatFormat = MAT_CUSPARSE_CSR;
    cusparseStruct->coo_p               = NULL;
    cusparseStruct->coo_pw              = NULL;
    cusparseStruct->stream              = 0;
    stat = cusparseCreate(&(cusparseStruct->handle));CHKERRCUSPARSE(stat);
    cusparseStruct->deviceMat = NULL;
  }

  A->ops->assemblyend           = MatAssemblyEnd_MPIAIJCUSPARSE;
  A->ops->mult                  = MatMult_MPIAIJCUSPARSE;
  A->ops->multadd               = MatMultAdd_MPIAIJCUSPARSE;
  A->ops->multtranspose         = MatMultTranspose_MPIAIJCUSPARSE;
  A->ops->setfromoptions        = MatSetFromOptions_MPIAIJCUSPARSE;
  A->ops->destroy               = MatDestroy_MPIAIJCUSPARSE;
  A->ops->zeroentries           = MatZeroEntries_MPIAIJCUSPARSE;
  A->ops->productsetfromoptions = MatProductSetFromOptions_MPIAIJCUSPARSE;

  ierr = PetscObjectChangeTypeName((PetscObject)A,MATMPIAIJCUSPARSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJGetLocalMatMerge_C",MatMPIAIJGetLocalMatMerge_MPIAIJCUSPARSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatMPIAIJSetPreallocation_C",MatMPIAIJSetPreallocation_MPIAIJCUSPARSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatCUSPARSESetFormat_C",MatCUSPARSESetFormat_MPIAIJCUSPARSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetPreallocationCOO_C",MatSetPreallocationCOO_MPIAIJCUSPARSE);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatSetValuesCOO_C",MatSetValuesCOO_MPIAIJCUSPARSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_MPIAIJCUSPARSE(Mat A)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscCUDAInitializeCheck();CHKERRQ(ierr);
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
   MATAIJCUSPARSE - MATMPIAIJCUSPARSE = "aijcusparse" = "mpiaijcusparse" - A matrix type to be used for sparse matrices.

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

 .seealso: MatCreateAIJCUSPARSE(), MATSEQAIJCUSPARSE, MatCreateSeqAIJCUSPARSE(), MatCUSPARSESetFormat(), MatCUSPARSEStorageFormat, MatCUSPARSEFormatOperation
M
M*/

// get GPU pointer to stripped down Mat. For both Seq and MPI Mat.
PetscErrorCode MatCUSPARSEGetDeviceMatWrite(Mat A, PetscSplitCSRDataStructure **B)
{
#if defined(PETSC_USE_CTABLE)
  SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Device metadata does not support ctable (--with-ctable=0)");
#else
  PetscSplitCSRDataStructure **p_d_mat;
  PetscMPIInt                size,rank;
  MPI_Comm                   comm;
  PetscErrorCode             ierr;
  int                        *ai,*bi,*aj,*bj;
  PetscScalar                *aa,*ba;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  if (A->factortype == MAT_FACTOR_NONE) {
    CsrMatrix *matrixA,*matrixB=NULL;
    if (size == 1) {
      Mat_SeqAIJCUSPARSE *cusparsestruct = (Mat_SeqAIJCUSPARSE*)A->spptr;
      p_d_mat = &cusparsestruct->deviceMat;
      Mat_SeqAIJCUSPARSEMultStruct *matstruct = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestruct->mat;
      if (cusparsestruct->format==MAT_CUSPARSE_CSR) {
        matrixA = (CsrMatrix*)matstruct->mat;
        bi = bj = NULL; ba = NULL;
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Device Mat needs MAT_CUSPARSE_CSR");
    } else {
      Mat_MPIAIJ         *aij = (Mat_MPIAIJ*)A->data;
      Mat_MPIAIJCUSPARSE *spptr = (Mat_MPIAIJCUSPARSE*)aij->spptr;
      p_d_mat = &spptr->deviceMat;
      Mat_SeqAIJCUSPARSE *cusparsestructA = (Mat_SeqAIJCUSPARSE*)aij->A->spptr;
      Mat_SeqAIJCUSPARSE *cusparsestructB = (Mat_SeqAIJCUSPARSE*)aij->B->spptr;
      Mat_SeqAIJCUSPARSEMultStruct *matstructA = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestructA->mat;
      Mat_SeqAIJCUSPARSEMultStruct *matstructB = (Mat_SeqAIJCUSPARSEMultStruct*)cusparsestructB->mat;
      if (cusparsestructA->format==MAT_CUSPARSE_CSR) {
        if (cusparsestructB->format!=MAT_CUSPARSE_CSR) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Device Mat B needs MAT_CUSPARSE_CSR");
        matrixA = (CsrMatrix*)matstructA->mat;
        matrixB = (CsrMatrix*)matstructB->mat;
        bi = thrust::raw_pointer_cast(matrixB->row_offsets->data());
        bj = thrust::raw_pointer_cast(matrixB->column_indices->data());
        ba = thrust::raw_pointer_cast(matrixB->values->data());
      } else SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Device Mat A needs MAT_CUSPARSE_CSR");
    }
    ai = thrust::raw_pointer_cast(matrixA->row_offsets->data());
    aj = thrust::raw_pointer_cast(matrixA->column_indices->data());
    aa = thrust::raw_pointer_cast(matrixA->values->data());
  } else {
    *B = NULL;
    PetscFunctionReturn(0);
  }
  // act like MatSetValues because not called on host
  if (A->assembled) {
    if (A->was_assembled) {
      ierr = PetscInfo(A,"Assemble more than once already\n");CHKERRQ(ierr);
    }
    A->was_assembled = PETSC_TRUE; // this is done (lazy) in MatAssemble but we are not calling it anymore - done in AIJ AssemblyEnd, need here?
  } else {
    SETERRQ(comm,PETSC_ERR_SUP,"Need assemble matrix");
  }
  if (!*p_d_mat) {
    cudaError_t                 err;
    PetscSplitCSRDataStructure  *d_mat, h_mat;
    Mat_SeqAIJ                  *jaca;
    PetscInt                    n = A->rmap->n, nnz;
    // create and copy
    ierr = PetscInfo(A,"Create device matrix\n");CHKERRQ(ierr);
    err = cudaMalloc((void **)&d_mat, sizeof(PetscSplitCSRDataStructure));CHKERRCUDA(err);
    err = cudaMemset( d_mat, 0,       sizeof(PetscSplitCSRDataStructure));CHKERRCUDA(err);
    *B = *p_d_mat = d_mat; // return it, set it in Mat, and set it up
    if (size == 1) {
      jaca = (Mat_SeqAIJ*)A->data;
      h_mat.rstart = 0; h_mat.rend = A->rmap->n;
      h_mat.cstart = 0; h_mat.cend = A->cmap->n;
      h_mat.offdiag.i = h_mat.offdiag.j = NULL;
      h_mat.offdiag.a = NULL;
    } else {
      Mat_MPIAIJ  *aij = (Mat_MPIAIJ*)A->data;
      Mat_SeqAIJ  *jacb;
      jaca = (Mat_SeqAIJ*)aij->A->data;
      jacb = (Mat_SeqAIJ*)aij->B->data;
      if (!aij->garray) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"MPIAIJ Matrix was assembled but is missing garray");
      if (aij->B->rmap->n != aij->A->rmap->n) SETERRQ(comm,PETSC_ERR_SUP,"Only support aij->B->rmap->n == aij->A->rmap->n");
      // create colmap - this is ussually done (lazy) in MatSetValues
      aij->donotstash = PETSC_TRUE;
      aij->A->nooffprocentries = aij->B->nooffprocentries = A->nooffprocentries = PETSC_TRUE;
      jaca->nonew = jacb->nonew = PETSC_TRUE; // no more dissassembly
      ierr = PetscCalloc1(A->cmap->N+1,&aij->colmap);CHKERRQ(ierr);
      aij->colmap[A->cmap->N] = -9;
      ierr = PetscLogObjectMemory((PetscObject)A,(A->cmap->N+1)*sizeof(PetscInt));CHKERRQ(ierr);
      {
        PetscInt ii;
        for (ii=0; ii<aij->B->cmap->n; ii++) aij->colmap[aij->garray[ii]] = ii+1;
      }
      if (aij->colmap[A->cmap->N] != -9) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"aij->colmap[A->cmap->N] != -9");
      // allocate B copy data
      h_mat.rstart = A->rmap->rstart; h_mat.rend = A->rmap->rend;
      h_mat.cstart = A->cmap->rstart; h_mat.cend = A->cmap->rend;
      nnz = jacb->i[n];

      if (jacb->compressedrow.use) {
        err = cudaMalloc((void **)&h_mat.offdiag.i,               (n+1)*sizeof(int));CHKERRCUDA(err); // kernel input
        err = cudaMemcpy(          h_mat.offdiag.i,    jacb->i,   (n+1)*sizeof(int), cudaMemcpyHostToDevice);CHKERRCUDA(err);
      } else h_mat.offdiag.i = bi;
      h_mat.offdiag.j = bj;
      h_mat.offdiag.a = ba;

      err = cudaMalloc((void **)&h_mat.colmap,                  (A->cmap->N+1)*sizeof(PetscInt));CHKERRCUDA(err); // kernel output
      err = cudaMemcpy(          h_mat.colmap,    aij->colmap,  (A->cmap->N+1)*sizeof(PetscInt), cudaMemcpyHostToDevice);CHKERRCUDA(err);
      h_mat.offdiag.ignorezeroentries = jacb->ignorezeroentries;
      h_mat.offdiag.n = n;
    }
    // allocate A copy data
    nnz = jaca->i[n];
    h_mat.diag.n = n;
    h_mat.diag.ignorezeroentries = jaca->ignorezeroentries;
    ierr = MPI_Comm_rank(comm,&h_mat.rank);CHKERRMPI(ierr);
    if (jaca->compressedrow.use) {
      err = cudaMalloc((void **)&h_mat.diag.i,               (n+1)*sizeof(int));CHKERRCUDA(err); // kernel input
      err = cudaMemcpy(          h_mat.diag.i,    jaca->i,   (n+1)*sizeof(int), cudaMemcpyHostToDevice);CHKERRCUDA(err);
    } else {
      h_mat.diag.i = ai;
    }
    h_mat.diag.j = aj;
    h_mat.diag.a = aa;
    // copy pointers and metdata to device
    err = cudaMemcpy(          d_mat, &h_mat, sizeof(PetscSplitCSRDataStructure), cudaMemcpyHostToDevice);CHKERRCUDA(err);
    ierr = PetscInfo2(A,"Create device Mat n=%D nnz=%D\n",h_mat.diag.n, nnz);CHKERRQ(ierr);
  } else {
    *B = *p_d_mat;
  }
  A->assembled = PETSC_FALSE; // ready to write with matsetvalues - this done (lazy) in normal MatSetValues
  PetscFunctionReturn(0);
#endif
}
