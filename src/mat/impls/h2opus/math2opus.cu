#include <h2opusconf.h>
/* skip compilation of this .cu file if H2OPUS is CPU only while PETSc has GPU support */
#if !defined(__CUDACC__) || defined(H2OPUS_USE_GPU)
#include <h2opus.h>
#if defined(H2OPUS_USE_MPI)
#include <h2opus/distributed/distributed_h2opus_handle.h>
#include <h2opus/distributed/distributed_geometric_construction.h>
#include <h2opus/distributed/distributed_hgemv.h>
#include <h2opus/distributed/distributed_horthog.h>
#include <h2opus/distributed/distributed_hcompress.h>
#endif
#include <h2opus/util/boxentrygen.h>
#include <petsc/private/matimpl.h>
#include <petsc/private/vecimpl.h>
#include <petsc/private/deviceimpl.h>
#include <petscsf.h>

/* math2opusutils */
PETSC_INTERN PetscErrorCode PetscSFGetVectorSF(PetscSF,PetscInt,PetscInt,PetscInt,PetscSF*);
PETSC_INTERN PetscErrorCode VecSign(Vec,Vec);
PETSC_INTERN PetscErrorCode VecSetDelta(Vec,PetscInt);
PETSC_INTERN PetscErrorCode MatApproximateNorm_Private(Mat,NormType,PetscInt,PetscReal*);

#define MatH2OpusGetThrustPointer(v) thrust::raw_pointer_cast((v).data())

/* Use GPU only if H2OPUS is configured for GPU */
#if defined(PETSC_HAVE_CUDA) && defined(H2OPUS_USE_GPU)
#define PETSC_H2OPUS_USE_GPU
#endif
#if defined(PETSC_H2OPUS_USE_GPU)
#define MatH2OpusUpdateIfNeeded(A,B) MatBindToCPU(A,(PetscBool)((A)->boundtocpu || (B)))
#else
#define MatH2OpusUpdateIfNeeded(A,B) 0
#endif

// TODO H2OPUS:
// DistributedHMatrix
//   unsymmetric ?
//   transpose for distributed_hgemv?
//   clearData()
// Unify interface for sequential and parallel?
// Reuse geometric construction (almost possible, only the unsymmetric case is explicitly handled)
//
template <class T> class PetscPointCloud : public H2OpusDataSet<T>
{
  private:
    int dimension;
    size_t num_points;
    std::vector<T> pts;

  public:
    PetscPointCloud(int dim, size_t num_pts, const T coords[])
    {
      dim = dim > 0 ? dim : 1;
      this->dimension = dim;
      this->num_points = num_pts;

      pts.resize(num_pts*dim);
      if (coords) {
        for (size_t n = 0; n < num_points; n++)
          for (int i = 0; i < dim; i++)
            pts[n*dim + i] = coords[n*dim + i];
      } else {
        PetscReal h = 1./(num_points - 1);
        for (size_t n = 0; n < num_points; n++)
          for (int i = 0; i < dim; i++)
            pts[n*dim + i] = i*h;
      }
    }

    PetscPointCloud(const PetscPointCloud<T>& other)
    {
      size_t N = other.dimension * other.num_points;
      this->dimension = other.dimension;
      this->num_points = other.num_points;
      this->pts.resize(N);
      for (size_t i = 0; i < N; i++)
        this->pts[i] = other.pts[i];
    }

    int getDimension() const
    {
        return dimension;
    }

    size_t getDataSetSize() const
    {
        return num_points;
    }

    T getDataPoint(size_t idx, int dim) const
    {
        assert(dim < dimension && idx < num_points);
        return pts[idx*dimension + dim];
    }

    void Print(std::ostream& out = std::cout)
    {
       out << "Dimension: " << dimension << std::endl;
       out << "NumPoints: " << num_points << std::endl;
       for (size_t n = 0; n < num_points; n++) {
         for (int d = 0; d < dimension; d++)
           out << pts[n*dimension + d] << " ";
         out << std::endl;
       }
    }
};

template<class T> class PetscFunctionGenerator
{
private:
  MatH2OpusKernel k;
  int             dim;
  void            *ctx;

public:
    PetscFunctionGenerator(MatH2OpusKernel k, int dim, void* ctx) { this->k = k; this->dim = dim; this->ctx = ctx; }
    PetscFunctionGenerator(PetscFunctionGenerator& other) { this->k = other.k; this->dim = other.dim; this->ctx = other.ctx; }
    T operator()(PetscReal *pt1, PetscReal *pt2)
    {
        return (T)((*this->k)(this->dim,pt1,pt2,this->ctx));
    }
};

#include <../src/mat/impls/h2opus/math2opussampler.hpp>

/* just to not clutter the code */
#if !defined(H2OPUS_USE_GPU)
typedef HMatrix HMatrix_GPU;
#if defined(H2OPUS_USE_MPI)
typedef DistributedHMatrix DistributedHMatrix_GPU;
#endif
#endif

typedef struct {
#if defined(H2OPUS_USE_MPI)
  distributedH2OpusHandle_t handle;
#else
  h2opusHandle_t handle;
#endif
  /* Sequential and parallel matrices are two different classes at the moment */
  HMatrix *hmatrix;
#if defined(H2OPUS_USE_MPI)
  DistributedHMatrix *dist_hmatrix;
#else
  HMatrix *dist_hmatrix; /* just to not clutter the code */
#endif
  /* May use permutations */
  PetscSF sf;
  PetscLayout h2opus_rmap, h2opus_cmap;
  IS h2opus_indexmap;
  thrust::host_vector<PetscScalar> *xx,*yy;
  PetscInt xxs,yys;
  PetscBool multsetup;

  /* GPU */
  HMatrix_GPU *hmatrix_gpu;
#if defined(H2OPUS_USE_MPI)
  DistributedHMatrix_GPU *dist_hmatrix_gpu;
#else
  HMatrix_GPU *dist_hmatrix_gpu; /* just to not clutter the code */
#endif
#if defined(PETSC_H2OPUS_USE_GPU)
  thrust::device_vector<PetscScalar> *xx_gpu,*yy_gpu;
  PetscInt xxs_gpu,yys_gpu;
#endif

  /* construction from matvecs */
  PetscMatrixSampler* sampler;
  PetscBool nativemult;

  /* Admissibility */
  PetscReal eta;
  PetscInt  leafsize;

  /* for dof reordering */
  PetscPointCloud<PetscReal> *ptcloud;

  /* kernel for generating matrix entries */
  PetscFunctionGenerator<PetscScalar> *kernel;

  /* basis orthogonalized? */
  PetscBool orthogonal;

  /* customization */
  PetscInt  basisord;
  PetscInt  max_rank;
  PetscInt  bs;
  PetscReal rtol;
  PetscInt  norm_max_samples;
  PetscBool check_construction;
  PetscBool hara_verbose;

  /* keeps track of MatScale values */
  PetscScalar s;
} Mat_H2OPUS;

static PetscErrorCode MatDestroy_H2OPUS(Mat A)
{
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(H2OPUS_USE_MPI)
  h2opusDestroyDistributedHandle(a->handle);
#else
  h2opusDestroyHandle(a->handle);
#endif
  delete a->dist_hmatrix;
  delete a->hmatrix;
  ierr = PetscSFDestroy(&a->sf);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&a->h2opus_rmap);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&a->h2opus_cmap);CHKERRQ(ierr);
  ierr = ISDestroy(&a->h2opus_indexmap);CHKERRQ(ierr);
  delete a->xx;
  delete a->yy;
  delete a->hmatrix_gpu;
  delete a->dist_hmatrix_gpu;
#if defined(PETSC_H2OPUS_USE_GPU)
  delete a->xx_gpu;
  delete a->yy_gpu;
#endif
  delete a->sampler;
  delete a->ptcloud;
  delete a->kernel;
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_h2opus_seqdense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_h2opus_seqdensecuda_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_h2opus_mpidense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_h2opus_mpidensecuda_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,NULL);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode MatH2OpusSetNativeMult(Mat A, PetscBool nm)
{
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;
  PetscBool      ish2opus;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidLogicalCollectiveBool(A,nm,2);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATH2OPUS,&ish2opus);CHKERRQ(ierr);
  if (ish2opus) {
    if (a->h2opus_rmap) { /* need to swap layouts for vector creation */
      if ((!a->nativemult && nm) || (a->nativemult && !nm)) {
        PetscLayout t;
        t = A->rmap;
        A->rmap = a->h2opus_rmap;
        a->h2opus_rmap = t;
        t = A->cmap;
        A->cmap = a->h2opus_cmap;
        a->h2opus_cmap = t;
      }
    }
    a->nativemult = nm;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatH2OpusGetNativeMult(Mat A, PetscBool *nm)
{
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;
  PetscBool      ish2opus;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(nm,2);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATH2OPUS,&ish2opus);CHKERRQ(ierr);
  if (!ish2opus) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not for type %s",((PetscObject)A)->type_name);
  *nm = a->nativemult;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatNorm_H2OPUS(Mat A, NormType normtype, PetscReal* n)
{
  PetscErrorCode ierr;
  PetscBool      ish2opus;
  PetscInt       nmax = PETSC_DECIDE;
  Mat_H2OPUS     *a = NULL;
  PetscBool      mult = PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATH2OPUS,&ish2opus);CHKERRQ(ierr);
  if (ish2opus) { /* set userdefine number of samples and fastpath for mult (norms are order independent) */
    a = (Mat_H2OPUS*)A->data;

    nmax = a->norm_max_samples;
    mult = a->nativemult;
    ierr = MatH2OpusSetNativeMult(A,PETSC_TRUE);CHKERRQ(ierr);
  } else {
    ierr = PetscOptionsGetInt(((PetscObject)A)->options,((PetscObject)A)->prefix,"-mat_approximate_norm_samples",&nmax,NULL);CHKERRQ(ierr);
  }
  ierr = MatApproximateNorm_Private(A,normtype,nmax,n);CHKERRQ(ierr);
  if (a) { ierr = MatH2OpusSetNativeMult(A,mult);CHKERRQ(ierr); }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultNKernel_H2OPUS(Mat A, PetscBool transA, Mat B, Mat C)
{
  Mat_H2OPUS     *h2opus = (Mat_H2OPUS*)A->data;
#if defined(H2OPUS_USE_MPI)
  h2opusHandle_t handle = h2opus->handle->handle;
#else
  h2opusHandle_t handle = h2opus->handle;
#endif
  PetscBool      boundtocpu = PETSC_TRUE;
  PetscScalar    *xx,*yy,*uxx,*uyy;
  PetscInt       blda,clda;
  PetscMPIInt    size;
  PetscSF        bsf,csf;
  PetscBool      usesf = (PetscBool)(h2opus->sf && !h2opus->nativemult);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  HLibProfile::clear();
#if defined(PETSC_H2OPUS_USE_GPU)
  boundtocpu = A->boundtocpu;
#endif
  ierr = MatDenseGetLDA(B,&blda);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(C,&clda);CHKERRQ(ierr);
  if (usesf) {
    PetscInt n;

    ierr = PetscSFGetGraph(h2opus->sf,NULL,&n,NULL,NULL);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)B,"_math2opus_vectorsf",(PetscObject*)&bsf);CHKERRQ(ierr);
    if (!bsf) {
      ierr = PetscSFGetVectorSF(h2opus->sf,B->cmap->N,blda,PETSC_DECIDE,&bsf);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)B,"_math2opus_vectorsf",(PetscObject)bsf);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)bsf);CHKERRQ(ierr);
    }
    ierr = PetscObjectQuery((PetscObject)C,"_math2opus_vectorsf",(PetscObject*)&csf);CHKERRQ(ierr);
    if (!csf) {
      ierr = PetscSFGetVectorSF(h2opus->sf,B->cmap->N,clda,PETSC_DECIDE,&csf);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)C,"_math2opus_vectorsf",(PetscObject)csf);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)csf);CHKERRQ(ierr);
    }
    blda = n;
    clda = n;
  }
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRMPI(ierr);
  if (boundtocpu) {
    if (usesf) {
      PetscInt n;

      ierr = PetscSFGetGraph(h2opus->sf,NULL,&n,NULL,NULL);CHKERRQ(ierr);
      if (h2opus->xxs < B->cmap->n) { h2opus->xx->resize(n*B->cmap->N); h2opus->xxs = B->cmap->N; }
      if (h2opus->yys < B->cmap->n) { h2opus->yy->resize(n*B->cmap->N); h2opus->yys = B->cmap->N; }
    }
    ierr = MatDenseGetArrayRead(B,(const PetscScalar**)&xx);CHKERRQ(ierr);
    ierr = MatDenseGetArrayWrite(C,&yy);CHKERRQ(ierr);
    if (usesf) {
      uxx  = MatH2OpusGetThrustPointer(*h2opus->xx);
      uyy  = MatH2OpusGetThrustPointer(*h2opus->yy);
      ierr = PetscSFBcastBegin(bsf,MPIU_SCALAR,xx,uxx,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(bsf,MPIU_SCALAR,xx,uxx,MPI_REPLACE);CHKERRQ(ierr);
    } else {
      uxx = xx;
      uyy = yy;
    }
    if (size > 1) {
      if (!h2opus->dist_hmatrix) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing distributed CPU matrix");
      if (transA && !A->symmetric) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MatMultTranspose not yet coded in parallel");
#if defined(H2OPUS_USE_MPI)
      distributed_hgemv(/*transA ? H2Opus_Trans : H2Opus_NoTrans, */h2opus->s, *h2opus->dist_hmatrix, uxx, blda, 0.0, uyy, clda, B->cmap->N, h2opus->handle);
#endif
    } else {
      if (!h2opus->hmatrix) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
      hgemv(transA ? H2Opus_Trans : H2Opus_NoTrans, h2opus->s, *h2opus->hmatrix, uxx, blda, 0.0, uyy, clda, B->cmap->N, handle);
    }
    ierr = MatDenseRestoreArrayRead(B,(const PetscScalar**)&xx);CHKERRQ(ierr);
    if (usesf) {
      ierr = PetscSFReduceBegin(csf,MPIU_SCALAR,uyy,yy,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(csf,MPIU_SCALAR,uyy,yy,MPI_REPLACE);CHKERRQ(ierr);
    }
    ierr = MatDenseRestoreArrayWrite(C,&yy);CHKERRQ(ierr);
#if defined(PETSC_H2OPUS_USE_GPU)
  } else {
    PetscBool ciscuda,biscuda;

    if (usesf) {
      PetscInt n;

      ierr = PetscSFGetGraph(h2opus->sf,NULL,&n,NULL,NULL);CHKERRQ(ierr);
      if (h2opus->xxs_gpu < B->cmap->n) { h2opus->xx_gpu->resize(n*B->cmap->N); h2opus->xxs_gpu = B->cmap->N; }
      if (h2opus->yys_gpu < B->cmap->n) { h2opus->yy_gpu->resize(n*B->cmap->N); h2opus->yys_gpu = B->cmap->N; }
    }
    /* If not of type seqdensecuda, convert on the fly (i.e. allocate GPU memory) */
    ierr = PetscObjectTypeCompareAny((PetscObject)B,&biscuda,MATSEQDENSECUDA,MATMPIDENSECUDA,"");CHKERRQ(ierr);
    if (!biscuda) {
      ierr = MatConvert(B,MATDENSECUDA,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
    }
    ierr = PetscObjectTypeCompareAny((PetscObject)C,&ciscuda,MATSEQDENSECUDA,MATMPIDENSECUDA,"");CHKERRQ(ierr);
    if (!ciscuda) {
      C->assembled = PETSC_TRUE;
      ierr = MatConvert(C,MATDENSECUDA,MAT_INPLACE_MATRIX,&C);CHKERRQ(ierr);
    }
    ierr = MatDenseCUDAGetArrayRead(B,(const PetscScalar**)&xx);CHKERRQ(ierr);
    ierr = MatDenseCUDAGetArrayWrite(C,&yy);CHKERRQ(ierr);
    if (usesf) {
      uxx  = MatH2OpusGetThrustPointer(*h2opus->xx_gpu);
      uyy  = MatH2OpusGetThrustPointer(*h2opus->yy_gpu);
      ierr = PetscSFBcastBegin(bsf,MPIU_SCALAR,xx,uxx,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(bsf,MPIU_SCALAR,xx,uxx,MPI_REPLACE);CHKERRQ(ierr);
    } else {
      uxx = xx;
      uyy = yy;
    }
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    if (size > 1) {
      if (!h2opus->dist_hmatrix_gpu) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing distributed GPU matrix");
      if (transA && !A->symmetric) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MatMultTranspose not yet coded in parallel");
#if defined(H2OPUS_USE_MPI)
      distributed_hgemv(/* transA ? H2Opus_Trans : H2Opus_NoTrans, */h2opus->s, *h2opus->dist_hmatrix_gpu, uxx, blda, 0.0, uyy, clda, B->cmap->N, h2opus->handle);
#endif
    } else {
      if (!h2opus->hmatrix_gpu) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
      hgemv(transA ? H2Opus_Trans : H2Opus_NoTrans, h2opus->s, *h2opus->hmatrix_gpu, uxx, blda, 0.0, uyy, clda, B->cmap->N, handle);
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = MatDenseCUDARestoreArrayRead(B,(const PetscScalar**)&xx);CHKERRQ(ierr);
    if (usesf) {
      ierr = PetscSFReduceBegin(csf,MPIU_SCALAR,uyy,yy,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(csf,MPIU_SCALAR,uyy,yy,MPI_REPLACE);CHKERRQ(ierr);
    }
    ierr = MatDenseCUDARestoreArrayWrite(C,&yy);CHKERRQ(ierr);
    if (!biscuda) {
      ierr = MatConvert(B,MATDENSE,MAT_INPLACE_MATRIX,&B);CHKERRQ(ierr);
    }
    if (!ciscuda) {
      ierr = MatConvert(C,MATDENSE,MAT_INPLACE_MATRIX,&C);CHKERRQ(ierr);
    }
#endif
  }
  { /* log flops */
    double gops,time,perf,dev;
    HLibProfile::getHgemvPerf(gops,time,perf,dev);
#if defined(PETSC_H2OPUS_USE_GPU)
    if (boundtocpu) {
      ierr = PetscLogFlops(1e9*gops);CHKERRQ(ierr);
    } else {
      ierr = PetscLogGpuFlops(1e9*gops);CHKERRQ(ierr);
    }
#else
    ierr = PetscLogFlops(1e9*gops);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductNumeric_H2OPUS(Mat C)
{
  Mat_Product    *product = C->product;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  switch (product->type) {
  case MATPRODUCT_AB:
    ierr = MatMultNKernel_H2OPUS(product->A,PETSC_FALSE,product->B,C);CHKERRQ(ierr);
    break;
  case MATPRODUCT_AtB:
    ierr = MatMultNKernel_H2OPUS(product->A,PETSC_TRUE,product->B,C);CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatProduct type %s is not supported",MatProductTypes[product->type]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSymbolic_H2OPUS(Mat C)
{
  PetscErrorCode ierr;
  Mat_Product    *product = C->product;
  PetscBool      cisdense;
  Mat            A,B;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  A = product->A;
  B = product->B;
  switch (product->type) {
  case MATPRODUCT_AB:
    ierr = MatSetSizes(C,A->rmap->n,B->cmap->n,A->rmap->N,B->cmap->N);CHKERRQ(ierr);
    ierr = MatSetBlockSizesFromMats(C,product->A,product->B);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompareAny((PetscObject)C,&cisdense,MATSEQDENSE,MATMPIDENSE,MATSEQDENSECUDA,MATMPIDENSECUDA,"");CHKERRQ(ierr);
    if (!cisdense) { ierr = MatSetType(C,((PetscObject)product->B)->type_name);CHKERRQ(ierr); }
    ierr = MatSetUp(C);CHKERRQ(ierr);
    break;
  case MATPRODUCT_AtB:
    ierr = MatSetSizes(C,A->cmap->n,B->cmap->n,A->cmap->N,B->cmap->N);CHKERRQ(ierr);
    ierr = MatSetBlockSizesFromMats(C,product->A,product->B);CHKERRQ(ierr);
    ierr = PetscObjectTypeCompareAny((PetscObject)C,&cisdense,MATSEQDENSE,MATMPIDENSE,MATSEQDENSECUDA,MATMPIDENSECUDA,"");CHKERRQ(ierr);
    if (!cisdense) { ierr = MatSetType(C,((PetscObject)product->B)->type_name);CHKERRQ(ierr); }
    ierr = MatSetUp(C);CHKERRQ(ierr);
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatProduct type %s is not supported",MatProductTypes[product->type]);
  }
  C->ops->productsymbolic = NULL;
  C->ops->productnumeric = MatProductNumeric_H2OPUS;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_H2OPUS(Mat C)
{
  PetscFunctionBegin;
  MatCheckProduct(C,1);
  if (C->product->type == MATPRODUCT_AB || C->product->type == MATPRODUCT_AtB) {
    C->ops->productsymbolic = MatProductSymbolic_H2OPUS;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultKernel_H2OPUS(Mat A, Vec x, PetscScalar sy, Vec y, PetscBool trans)
{
  Mat_H2OPUS     *h2opus = (Mat_H2OPUS*)A->data;
#if defined(H2OPUS_USE_MPI)
  h2opusHandle_t handle = h2opus->handle->handle;
#else
  h2opusHandle_t handle = h2opus->handle;
#endif
  PetscBool      boundtocpu = PETSC_TRUE;
  PetscInt       n;
  PetscScalar    *xx,*yy,*uxx,*uyy;
  PetscMPIInt    size;
  PetscBool      usesf = (PetscBool)(h2opus->sf && !h2opus->nativemult);
  PetscErrorCode ierr;

  PetscFunctionBegin;
  HLibProfile::clear();
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRMPI(ierr);
#if defined(PETSC_H2OPUS_USE_GPU)
  boundtocpu = A->boundtocpu;
#endif
  if (usesf) {
    ierr = PetscSFGetGraph(h2opus->sf,NULL,&n,NULL,NULL);CHKERRQ(ierr);
  } else n = A->rmap->n;
  if (boundtocpu) {
    ierr = VecGetArrayRead(x,(const PetscScalar**)&xx);CHKERRQ(ierr);
    if (sy == 0.0) {
      ierr = VecGetArrayWrite(y,&yy);CHKERRQ(ierr);
    } else {
      ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
    }
    if (usesf) {
      uxx = MatH2OpusGetThrustPointer(*h2opus->xx);
      uyy = MatH2OpusGetThrustPointer(*h2opus->yy);

      ierr = PetscSFBcastBegin(h2opus->sf,MPIU_SCALAR,xx,uxx,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(h2opus->sf,MPIU_SCALAR,xx,uxx,MPI_REPLACE);CHKERRQ(ierr);
      if (sy != 0.0) {
        ierr = PetscSFBcastBegin(h2opus->sf,MPIU_SCALAR,yy,uyy,MPI_REPLACE);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(h2opus->sf,MPIU_SCALAR,yy,uyy,MPI_REPLACE);CHKERRQ(ierr);
      }
    } else {
      uxx = xx;
      uyy = yy;
    }
    if (size > 1) {
      if (!h2opus->dist_hmatrix) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing distributed CPU matrix");
      if (trans && !A->symmetric) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MatMultTranspose not yet coded in parallel");
#if defined(H2OPUS_USE_MPI)
      distributed_hgemv(/*trans ? H2Opus_Trans : H2Opus_NoTrans, */h2opus->s, *h2opus->dist_hmatrix, uxx, n, sy, uyy, n, 1, h2opus->handle);
#endif
    } else {
      if (!h2opus->hmatrix) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
      hgemv(trans ? H2Opus_Trans : H2Opus_NoTrans, h2opus->s, *h2opus->hmatrix, uxx, n, sy, uyy, n, 1, handle);
    }
    ierr = VecRestoreArrayRead(x,(const PetscScalar**)&xx);CHKERRQ(ierr);
    if (usesf) {
      ierr = PetscSFReduceBegin(h2opus->sf,MPIU_SCALAR,uyy,yy,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(h2opus->sf,MPIU_SCALAR,uyy,yy,MPI_REPLACE);CHKERRQ(ierr);
    }
    if (sy == 0.0) {
      ierr = VecRestoreArrayWrite(y,&yy);CHKERRQ(ierr);
    } else {
      ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
    }
#if defined(PETSC_H2OPUS_USE_GPU)
  } else {
    ierr = VecCUDAGetArrayRead(x,(const PetscScalar**)&xx);CHKERRQ(ierr);
    if (sy == 0.0) {
      ierr = VecCUDAGetArrayWrite(y,&yy);CHKERRQ(ierr);
    } else {
      ierr = VecCUDAGetArray(y,&yy);CHKERRQ(ierr);
    }
    if (usesf) {
      uxx = MatH2OpusGetThrustPointer(*h2opus->xx_gpu);
      uyy = MatH2OpusGetThrustPointer(*h2opus->yy_gpu);

      ierr = PetscSFBcastBegin(h2opus->sf,MPIU_SCALAR,xx,uxx,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(h2opus->sf,MPIU_SCALAR,xx,uxx,MPI_REPLACE);CHKERRQ(ierr);
      if (sy != 0.0) {
        ierr = PetscSFBcastBegin(h2opus->sf,MPIU_SCALAR,yy,uyy,MPI_REPLACE);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(h2opus->sf,MPIU_SCALAR,yy,uyy,MPI_REPLACE);CHKERRQ(ierr);
      }
    } else {
      uxx = xx;
      uyy = yy;
    }
    ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
    if (size > 1) {
      if (!h2opus->dist_hmatrix_gpu) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing distributed GPU matrix");
      if (trans && !A->symmetric) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MatMultTranspose not yet coded in parallel");
#if defined(H2OPUS_USE_MPI)
      distributed_hgemv(/*trans ? H2Opus_Trans : H2Opus_NoTrans, */h2opus->s, *h2opus->dist_hmatrix_gpu, uxx, n, sy, uyy, n, 1, h2opus->handle);
#endif
    } else {
      if (!h2opus->hmatrix_gpu) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
      hgemv(trans ? H2Opus_Trans : H2Opus_NoTrans, h2opus->s, *h2opus->hmatrix_gpu, uxx, n, sy, uyy, n, 1, handle);
    }
    ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
    ierr = VecCUDARestoreArrayRead(x,(const PetscScalar**)&xx);CHKERRQ(ierr);
    if (usesf) {
      ierr = PetscSFReduceBegin(h2opus->sf,MPIU_SCALAR,uyy,yy,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(h2opus->sf,MPIU_SCALAR,uyy,yy,MPI_REPLACE);CHKERRQ(ierr);
    }
    if (sy == 0.0) {
      ierr = VecCUDARestoreArrayWrite(y,&yy);CHKERRQ(ierr);
    } else {
      ierr = VecCUDARestoreArray(y,&yy);CHKERRQ(ierr);
    }
#endif
  }
  { /* log flops */
    double gops,time,perf,dev;
    HLibProfile::getHgemvPerf(gops,time,perf,dev);
#if defined(PETSC_H2OPUS_USE_GPU)
    if (boundtocpu) {
      ierr = PetscLogFlops(1e9*gops);CHKERRQ(ierr);
    } else {
      ierr = PetscLogGpuFlops(1e9*gops);CHKERRQ(ierr);
    }
#else
    ierr = PetscLogFlops(1e9*gops);CHKERRQ(ierr);
#endif
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_H2OPUS(Mat A, Vec x, Vec y)
{
  PetscBool      xiscuda,yiscuda;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompareAny((PetscObject)x,&xiscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)y,&yiscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
  ierr = MatH2OpusUpdateIfNeeded(A,!xiscuda || !yiscuda);CHKERRQ(ierr);
  ierr = MatMultKernel_H2OPUS(A,x,0.0,y,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_H2OPUS(Mat A, Vec x, Vec y)
{
  PetscBool      xiscuda,yiscuda;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompareAny((PetscObject)x,&xiscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)y,&yiscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
  ierr = MatH2OpusUpdateIfNeeded(A,!xiscuda || !yiscuda);CHKERRQ(ierr);
  ierr = MatMultKernel_H2OPUS(A,x,0.0,y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTransposeAdd_H2OPUS(Mat A, Vec x, Vec y, Vec z)
{
  PetscBool      xiscuda,ziscuda;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(y,z);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)x,&xiscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)z,&ziscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
  ierr = MatH2OpusUpdateIfNeeded(A,!xiscuda || !ziscuda);CHKERRQ(ierr);
  ierr = MatMultKernel_H2OPUS(A,x,1.0,z,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultAdd_H2OPUS(Mat A, Vec x, Vec y, Vec z)
{
  PetscBool      xiscuda,ziscuda;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(y,z);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)x,&xiscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
  ierr = PetscObjectTypeCompareAny((PetscObject)z,&ziscuda,VECSEQCUDA,VECMPICUDA,"");CHKERRQ(ierr);
  ierr = MatH2OpusUpdateIfNeeded(A,!xiscuda || !ziscuda);CHKERRQ(ierr);
  ierr = MatMultKernel_H2OPUS(A,x,1.0,z,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScale_H2OPUS(Mat A, PetscScalar s)
{
  Mat_H2OPUS *a = (Mat_H2OPUS*)A->data;

  PetscFunctionBegin;
  a->s *= s;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetFromOptions_H2OPUS(PetscOptionItems *PetscOptionsObject,Mat A)
{
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"H2OPUS options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_h2opus_leafsize","Leaf size of cluster tree",NULL,a->leafsize,&a->leafsize,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_h2opus_eta","Admissibility condition tolerance",NULL,a->eta,&a->eta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_h2opus_order","Basis order for off-diagonal sampling when constructed from kernel",NULL,a->basisord,&a->basisord,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_h2opus_maxrank","Maximum rank when constructed from matvecs",NULL,a->max_rank,&a->max_rank,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_h2opus_samples","Maximum number of samples to be taken concurrently when constructing from matvecs",NULL,a->bs,&a->bs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_h2opus_normsamples","Maximum bumber of samples to be when estimating norms",NULL,a->norm_max_samples,&a->norm_max_samples,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_h2opus_rtol","Relative tolerance for construction from sampling",NULL,a->rtol,&a->rtol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-mat_h2opus_check","Check error when constructing from sampling during MatAssemblyEnd()",NULL,a->check_construction,&a->check_construction,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-mat_h2opus_hara_verbose","Verbose output from hara construction",NULL,a->hara_verbose,&a->hara_verbose,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatH2OpusSetCoords_H2OPUS(Mat,PetscInt,const PetscReal[],PetscBool,MatH2OpusKernel,void*);

static PetscErrorCode MatH2OpusInferCoordinates_Private(Mat A)
{
  Mat_H2OPUS        *a = (Mat_H2OPUS*)A->data;
  Vec               c;
  PetscInt          spacedim;
  const PetscScalar *coords;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (a->ptcloud) PetscFunctionReturn(0);
  ierr = PetscObjectQuery((PetscObject)A,"__math2opus_coords",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c && a->sampler) {
    Mat S = a->sampler->GetSamplingMat();

    ierr = PetscObjectQuery((PetscObject)S,"__math2opus_coords",(PetscObject*)&c);CHKERRQ(ierr);
  }
  if (!c) {
    ierr = MatH2OpusSetCoords_H2OPUS(A,-1,NULL,PETSC_FALSE,NULL,NULL);CHKERRQ(ierr);
  } else {
    ierr = VecGetArrayRead(c,&coords);CHKERRQ(ierr);
    ierr = VecGetBlockSize(c,&spacedim);CHKERRQ(ierr);
    ierr = MatH2OpusSetCoords_H2OPUS(A,spacedim,coords,PETSC_FALSE,NULL,NULL);CHKERRQ(ierr);
    ierr = VecRestoreArrayRead(c,&coords);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetUpMultiply_H2OPUS(Mat A)
{
  MPI_Comm       comm;
  PetscMPIInt    size;
  PetscErrorCode ierr;
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;
  PetscInt       n = 0,*idx = NULL;
  int            *iidx = NULL;
  PetscCopyMode  own;
  PetscBool      rid;

  PetscFunctionBegin;
  if (a->multsetup) PetscFunctionReturn(0);
  if (a->sf) { /* MatDuplicate_H2OPUS takes reference to the SF */
    ierr = PetscSFGetGraph(a->sf,NULL,&n,NULL,NULL);CHKERRQ(ierr);
#if defined(PETSC_H2OPUS_USE_GPU)
    a->xx_gpu  = new thrust::device_vector<PetscScalar>(n);
    a->yy_gpu  = new thrust::device_vector<PetscScalar>(n);
    a->xxs_gpu = 1;
    a->yys_gpu = 1;
#endif
    a->xx  = new thrust::host_vector<PetscScalar>(n);
    a->yy  = new thrust::host_vector<PetscScalar>(n);
    a->xxs = 1;
    a->yys = 1;
  } else {
    IS is;
    ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
    if (!a->h2opus_indexmap) {
      if (size > 1) {
        if (!a->dist_hmatrix) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing distributed CPU matrix");
#if defined(H2OPUS_USE_MPI)
        iidx = MatH2OpusGetThrustPointer(a->dist_hmatrix->basis_tree.basis_branch.index_map);
        n    = a->dist_hmatrix->basis_tree.basis_branch.index_map.size();
#endif
      } else {
        iidx = MatH2OpusGetThrustPointer(a->hmatrix->u_basis_tree.index_map);
        n    = a->hmatrix->u_basis_tree.index_map.size();
      }

      if (PetscDefined(USE_64BIT_INDICES)) {
        PetscInt i;

        own  = PETSC_OWN_POINTER;
        ierr = PetscMalloc1(n,&idx);CHKERRQ(ierr);
        for (i=0;i<n;i++) idx[i] = iidx[i];
      } else {
        own  = PETSC_COPY_VALUES;
        idx  = (PetscInt*)iidx;
      }
      ierr = ISCreateGeneral(comm,n,idx,own,&is);CHKERRQ(ierr);
      ierr = ISSetPermutation(is);CHKERRQ(ierr);
      ierr = ISViewFromOptions(is,(PetscObject)A,"-mat_h2opus_indexmap_view");CHKERRQ(ierr);
      a->h2opus_indexmap = is;
    }
    ierr = ISGetLocalSize(a->h2opus_indexmap,&n);CHKERRQ(ierr);
    ierr = ISGetIndices(a->h2opus_indexmap,(const PetscInt **)&idx);CHKERRQ(ierr);
    rid  = (PetscBool)(n == A->rmap->n);
    ierr = MPIU_Allreduce(MPI_IN_PLACE,&rid,1,MPIU_BOOL,MPI_LAND,comm);CHKERRMPI(ierr);
    if (rid) {
      ierr = ISIdentity(a->h2opus_indexmap,&rid);CHKERRQ(ierr);
    }
    if (!rid) {
      if (size > 1) { /* Parallel distribution may be different, save it here for fast path in MatMult (see MatH2OpusSetNativeMult) */
        ierr = PetscLayoutCreate(comm,&a->h2opus_rmap);CHKERRQ(ierr);
        ierr = PetscLayoutSetLocalSize(a->h2opus_rmap,n);CHKERRQ(ierr);
        ierr = PetscLayoutSetUp(a->h2opus_rmap);CHKERRQ(ierr);
        ierr = PetscLayoutReference(a->h2opus_rmap,&a->h2opus_cmap);CHKERRQ(ierr);
      }
      ierr = PetscSFCreate(comm,&a->sf);CHKERRQ(ierr);
      ierr = PetscSFSetGraphLayout(a->sf,A->rmap,n,NULL,PETSC_OWN_POINTER,idx);CHKERRQ(ierr);
      ierr = PetscSFViewFromOptions(a->sf,(PetscObject)A,"-mat_h2opus_sf_view");CHKERRQ(ierr);
#if defined(PETSC_H2OPUS_USE_GPU)
      a->xx_gpu  = new thrust::device_vector<PetscScalar>(n);
      a->yy_gpu  = new thrust::device_vector<PetscScalar>(n);
      a->xxs_gpu = 1;
      a->yys_gpu = 1;
#endif
      a->xx  = new thrust::host_vector<PetscScalar>(n);
      a->yy  = new thrust::host_vector<PetscScalar>(n);
      a->xxs = 1;
      a->yys = 1;
    }
    ierr = ISRestoreIndices(a->h2opus_indexmap,(const PetscInt **)&idx);CHKERRQ(ierr);
  }
  a->multsetup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyEnd_H2OPUS(Mat A, MatAssemblyType assemblytype)
{
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;
#if defined(H2OPUS_USE_MPI)
  h2opusHandle_t handle = a->handle->handle;
#else
  h2opusHandle_t handle = a->handle;
#endif
  PetscBool      kernel = PETSC_FALSE;
  PetscBool      boundtocpu = PETSC_TRUE;
  PetscBool      samplingdone = PETSC_FALSE;
  MPI_Comm       comm;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  if (A->rmap->n != A->cmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Different row and column local sizes are not supported");
  if (A->rmap->N != A->cmap->N) SETERRQ(comm,PETSC_ERR_SUP,"Rectangular matrices are not supported");
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  /* TODO REUSABILITY of geometric construction */
  delete a->hmatrix;
  delete a->dist_hmatrix;
#if defined(PETSC_H2OPUS_USE_GPU)
  delete a->hmatrix_gpu;
  delete a->dist_hmatrix_gpu;
#endif
  a->orthogonal = PETSC_FALSE;

  /* TODO: other? */
  H2OpusBoxCenterAdmissibility adm(a->eta);

  ierr = PetscLogEventBegin(MAT_H2Opus_Build,A,0,0,0);CHKERRQ(ierr);
  if (size > 1) {
#if defined(H2OPUS_USE_MPI)
    a->dist_hmatrix = new DistributedHMatrix(A->rmap->n/*,A->symmetric*/);
#else
    a->dist_hmatrix = NULL;
#endif
  } else {
    a->hmatrix = new HMatrix(A->rmap->n,A->symmetric);
  }
  ierr = MatH2OpusInferCoordinates_Private(A);CHKERRQ(ierr);
  if (!a->ptcloud) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing pointcloud");
  if (a->kernel) {
    BoxEntryGen<PetscScalar, H2OPUS_HWTYPE_CPU, PetscFunctionGenerator<PetscScalar>> entry_gen(*a->kernel);
    if (size > 1) {
      if (!a->dist_hmatrix) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing distributed CPU matrix");
#if defined(H2OPUS_USE_MPI)
      buildDistributedHMatrix(*a->dist_hmatrix,a->ptcloud,adm,entry_gen,a->leafsize,a->basisord,a->handle);
#endif
    } else {
      buildHMatrix(*a->hmatrix,a->ptcloud,adm,entry_gen,a->leafsize,a->basisord);
    }
    kernel = PETSC_TRUE;
  } else {
    if (size > 1) SETERRQ(comm,PETSC_ERR_SUP,"Construction from sampling not supported in parallel");
    buildHMatrixStructure(*a->hmatrix,a->ptcloud,a->leafsize,adm);
  }
  ierr = MatSetUpMultiply_H2OPUS(A);CHKERRQ(ierr);

#if defined(PETSC_H2OPUS_USE_GPU)
  boundtocpu = A->boundtocpu;
  if (!boundtocpu) {
    if (size > 1) {
      if (!a->dist_hmatrix) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing distributed CPU matrix");
#if defined(H2OPUS_USE_MPI)
      a->dist_hmatrix_gpu = new DistributedHMatrix_GPU(*a->dist_hmatrix);
#endif
    } else {
      a->hmatrix_gpu = new HMatrix_GPU(*a->hmatrix);
    }
  }
#endif
  if (size == 1) {
    if (!kernel && a->sampler && a->sampler->GetSamplingMat()) {
      PetscReal Anorm;
      bool      verbose;

      ierr = PetscOptionsGetBool(((PetscObject)A)->options,((PetscObject)A)->prefix,"-mat_h2opus_hara_verbose",&a->hara_verbose,NULL);CHKERRQ(ierr);
      verbose = a->hara_verbose;
      ierr = MatApproximateNorm_Private(a->sampler->GetSamplingMat(),NORM_2,a->norm_max_samples,&Anorm);CHKERRQ(ierr);
      if (a->hara_verbose) { ierr = PetscPrintf(PETSC_COMM_SELF,"Sampling uses max rank %d, tol %g (%g*%g), %s samples %d\n",a->max_rank,a->rtol*Anorm,a->rtol,Anorm,boundtocpu ? "CPU" : "GPU",a->bs);CHKERRQ(ierr); }
      if (a->sf && !a->nativemult) {
        a->sampler->SetIndexMap(a->hmatrix->u_basis_tree.index_map.size(),a->hmatrix->u_basis_tree.index_map.data());
      }
      a->sampler->SetStream(handle->getMainStream());
      if (boundtocpu) {
        a->sampler->SetGPUSampling(false);
        hara(a->sampler, *a->hmatrix, a->max_rank, 10 /* TODO */,a->rtol*Anorm,a->bs,handle,verbose);
#if defined(PETSC_H2OPUS_USE_GPU)
      } else {
        a->sampler->SetGPUSampling(true);
        hara(a->sampler, *a->hmatrix_gpu, a->max_rank, 10 /* TODO */,a->rtol*Anorm,a->bs,handle,verbose);
#endif
      }
      samplingdone = PETSC_TRUE;
    }
  }
#if defined(PETSC_H2OPUS_USE_GPU)
  if (!boundtocpu) {
    delete a->hmatrix;
    delete a->dist_hmatrix;
    a->hmatrix = NULL;
    a->dist_hmatrix = NULL;
  }
  A->offloadmask = boundtocpu ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_GPU;
#endif
  ierr = PetscLogEventEnd(MAT_H2Opus_Build,A,0,0,0);CHKERRQ(ierr);

  if (!a->s) a->s = 1.0;
  A->assembled = PETSC_TRUE;

  if (samplingdone) {
    PetscBool check = a->check_construction;
    PetscBool checke = PETSC_FALSE;

    ierr = PetscOptionsGetBool(((PetscObject)A)->options,((PetscObject)A)->prefix,"-mat_h2opus_check",&check,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsGetBool(((PetscObject)A)->options,((PetscObject)A)->prefix,"-mat_h2opus_check_explicit",&checke,NULL);CHKERRQ(ierr);
    if (check) {
      Mat       E,Ae;
      PetscReal n1,ni,n2;
      PetscReal n1A,niA,n2A;
      void      (*normfunc)(void);

      Ae   = a->sampler->GetSamplingMat();
      ierr = MatConvert(A,MATSHELL,MAT_INITIAL_MATRIX,&E);CHKERRQ(ierr);
      ierr = MatShellSetOperation(E,MATOP_NORM,(void (*)(void))MatNorm_H2OPUS);CHKERRQ(ierr);
      ierr = MatAXPY(E,-1.0,Ae,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatNorm(E,NORM_1,&n1);CHKERRQ(ierr);
      ierr = MatNorm(E,NORM_INFINITY,&ni);CHKERRQ(ierr);
      ierr = MatNorm(E,NORM_2,&n2);CHKERRQ(ierr);
      if (checke) {
        Mat eA,eE,eAe;

        ierr = MatComputeOperator(A,MATAIJ,&eA);CHKERRQ(ierr);
        ierr = MatComputeOperator(E,MATAIJ,&eE);CHKERRQ(ierr);
        ierr = MatComputeOperator(Ae,MATAIJ,&eAe);CHKERRQ(ierr);
        ierr = MatChop(eA,PETSC_SMALL);CHKERRQ(ierr);
        ierr = MatChop(eE,PETSC_SMALL);CHKERRQ(ierr);
        ierr = MatChop(eAe,PETSC_SMALL);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)eA,"H2Mat");CHKERRQ(ierr);
        ierr = MatView(eA,NULL);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)eAe,"S");CHKERRQ(ierr);
        ierr = MatView(eAe,NULL);CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)eE,"H2Mat - S");CHKERRQ(ierr);
        ierr = MatView(eE,NULL);CHKERRQ(ierr);
        ierr = MatDestroy(&eA);CHKERRQ(ierr);
        ierr = MatDestroy(&eE);CHKERRQ(ierr);
        ierr = MatDestroy(&eAe);CHKERRQ(ierr);
      }

      ierr = MatGetOperation(Ae,MATOP_NORM,&normfunc);CHKERRQ(ierr);
      ierr = MatSetOperation(Ae,MATOP_NORM,(void (*)(void))MatNorm_H2OPUS);CHKERRQ(ierr);
      ierr = MatNorm(Ae,NORM_1,&n1A);CHKERRQ(ierr);
      ierr = MatNorm(Ae,NORM_INFINITY,&niA);CHKERRQ(ierr);
      ierr = MatNorm(Ae,NORM_2,&n2A);CHKERRQ(ierr);
      n1A  = PetscMax(n1A,PETSC_SMALL);
      n2A  = PetscMax(n2A,PETSC_SMALL);
      niA  = PetscMax(niA,PETSC_SMALL);
      ierr = MatSetOperation(Ae,MATOP_NORM,normfunc);CHKERRQ(ierr);
      ierr = PetscPrintf(PetscObjectComm((PetscObject)A),"MATH2OPUS construction errors: NORM_1 %g, NORM_INFINITY %g, NORM_2 %g (%g %g %g)\n",(double)n1,(double)ni,(double)n2,(double)(n1/n1A),(double)(ni/niA),(double)(n2/n2A));
      ierr = MatDestroy(&E);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_H2OPUS(Mat A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not yet supported");
  else {
    a->hmatrix->clearData();
#if defined(PETSC_H2OPUS_USE_GPU)
    if (a->hmatrix_gpu) a->hmatrix_gpu->clearData();
#endif
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_H2OPUS(Mat B, MatDuplicateOption op, Mat *nA)
{
  Mat            A;
  Mat_H2OPUS     *a, *b = (Mat_H2OPUS*)B->data;
#if defined(PETSC_H2OPUS_USE_GPU)
  PetscBool      iscpu = PETSC_FALSE;
#else
  PetscBool      iscpu = PETSC_TRUE;
#endif
  PetscErrorCode ierr;
  MPI_Comm       comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)B,&comm);CHKERRQ(ierr);
  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,B->rmap->n,B->cmap->n,B->rmap->N,B->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(A,MATH2OPUS);CHKERRQ(ierr);
  ierr = MatPropagateSymmetryOptions(B,A);CHKERRQ(ierr);
  a = (Mat_H2OPUS*)A->data;

  a->eta              = b->eta;
  a->leafsize         = b->leafsize;
  a->basisord         = b->basisord;
  a->max_rank         = b->max_rank;
  a->bs               = b->bs;
  a->rtol             = b->rtol;
  a->norm_max_samples = b->norm_max_samples;
  if (op == MAT_COPY_VALUES) a->s = b->s;

  a->ptcloud = new PetscPointCloud<PetscReal>(*b->ptcloud);
  if (op == MAT_COPY_VALUES && b->kernel) a->kernel = new PetscFunctionGenerator<PetscScalar>(*b->kernel);

#if defined(H2OPUS_USE_MPI)
  if (b->dist_hmatrix) { a->dist_hmatrix = new DistributedHMatrix(*b->dist_hmatrix); }
#if defined(PETSC_H2OPUS_USE_GPU)
  if (b->dist_hmatrix_gpu) { a->dist_hmatrix_gpu = new DistributedHMatrix_GPU(*b->dist_hmatrix_gpu); }
#endif
#endif
  if (b->hmatrix) {
    a->hmatrix = new HMatrix(*b->hmatrix);
    if (op == MAT_DO_NOT_COPY_VALUES) a->hmatrix->clearData();
  }
#if defined(PETSC_H2OPUS_USE_GPU)
  if (b->hmatrix_gpu) {
    a->hmatrix_gpu = new HMatrix_GPU(*b->hmatrix_gpu);
    if (op == MAT_DO_NOT_COPY_VALUES) a->hmatrix_gpu->clearData();
  }
#endif
  if (b->sf) {
    ierr = PetscObjectReference((PetscObject)b->sf);CHKERRQ(ierr);
    a->sf = b->sf;
  }
  if (b->h2opus_indexmap) {
    ierr = PetscObjectReference((PetscObject)b->h2opus_indexmap);CHKERRQ(ierr);
    a->h2opus_indexmap = b->h2opus_indexmap;
  }

  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatSetUpMultiply_H2OPUS(A);CHKERRQ(ierr);
  if (op == MAT_COPY_VALUES) {
    A->assembled = PETSC_TRUE;
    a->orthogonal = b->orthogonal;
#if defined(PETSC_H2OPUS_USE_GPU)
    A->offloadmask = B->offloadmask;
#endif
  }
#if defined(PETSC_H2OPUS_USE_GPU)
  iscpu = B->boundtocpu;
#endif
  ierr = MatBindToCPU(A,iscpu);CHKERRQ(ierr);

  *nA = A;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_H2OPUS(Mat A, PetscViewer view)
{
  Mat_H2OPUS        *h2opus = (Mat_H2OPUS*)A->data;
  PetscBool         isascii;
  PetscErrorCode    ierr;
  PetscMPIInt       size;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)view,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(view,&format);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRMPI(ierr);
  if (isascii) {
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      if (size == 1) {
        FILE *fp;
        ierr = PetscViewerASCIIGetPointer(view,&fp);CHKERRQ(ierr);
        dumpHMatrix(*h2opus->hmatrix,6,fp);
      }
    } else {
      ierr = PetscViewerASCIIPrintf(view,"  H-Matrix constructed from %s\n",h2opus->kernel ? "Kernel" : "Mat");CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(view,"  PointCloud dim %" PetscInt_FMT "\n",h2opus->ptcloud ? h2opus->ptcloud->getDimension() : 0);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(view,"  Admissibility parameters: leaf size %" PetscInt_FMT ", eta %g\n",h2opus->leafsize,(double)h2opus->eta);CHKERRQ(ierr);
      if (!h2opus->kernel) {
        ierr = PetscViewerASCIIPrintf(view,"  Sampling parameters: max_rank %" PetscInt_FMT ", samples %" PetscInt_FMT ", tolerance %g\n",h2opus->max_rank,h2opus->bs,(double)h2opus->rtol);CHKERRQ(ierr);
      } else {
        ierr = PetscViewerASCIIPrintf(view,"  Offdiagonal blocks approximation order %" PetscInt_FMT "\n",h2opus->basisord);CHKERRQ(ierr);
      }
      ierr = PetscViewerASCIIPrintf(view,"  Number of samples for norms %" PetscInt_FMT "\n",h2opus->norm_max_samples);CHKERRQ(ierr);
      if (size == 1) {
        double dense_mem_cpu = h2opus->hmatrix ? h2opus->hmatrix->getDenseMemoryUsage() : 0;
        double low_rank_cpu = h2opus->hmatrix ? h2opus->hmatrix->getLowRankMemoryUsage() : 0;
#if defined(PETSC_HAVE_CUDA)
        double dense_mem_gpu = h2opus->hmatrix_gpu ? h2opus->hmatrix_gpu->getDenseMemoryUsage() : 0;
        double low_rank_gpu = h2opus->hmatrix_gpu ? h2opus->hmatrix_gpu->getLowRankMemoryUsage() : 0;
#endif
        ierr = PetscViewerASCIIPrintf(view,"  Memory consumption GB (CPU): %g (dense) %g (low rank) %g (total)\n", dense_mem_cpu, low_rank_cpu, low_rank_cpu + dense_mem_cpu);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
        ierr = PetscViewerASCIIPrintf(view,"  Memory consumption GB (GPU): %g (dense) %g (low rank) %g (total)\n", dense_mem_gpu, low_rank_gpu, low_rank_gpu + dense_mem_gpu);CHKERRQ(ierr);
#endif
      } else {
#if defined(PETSC_HAVE_CUDA)
        double      matrix_mem[4] = {0.,0.,0.,0.};
        PetscMPIInt rsize = 4;
#else
        double      matrix_mem[2] = {0.,0.};
        PetscMPIInt rsize = 2;
#endif
#if defined(H2OPUS_USE_MPI)
        matrix_mem[0] = h2opus->dist_hmatrix ? h2opus->dist_hmatrix->getLocalDenseMemoryUsage() : 0;
        matrix_mem[1] = h2opus->dist_hmatrix ? h2opus->dist_hmatrix->getLocalLowRankMemoryUsage() : 0;
#if defined(PETSC_HAVE_CUDA)
        matrix_mem[2] = h2opus->dist_hmatrix_gpu ? h2opus->dist_hmatrix_gpu->getLocalDenseMemoryUsage() : 0;
        matrix_mem[3] = h2opus->dist_hmatrix_gpu ? h2opus->dist_hmatrix_gpu->getLocalLowRankMemoryUsage() : 0;
#endif
#endif
        ierr = MPIU_Allreduce(MPI_IN_PLACE,matrix_mem,rsize,MPI_DOUBLE_PRECISION,MPI_SUM,PetscObjectComm((PetscObject)A));CHKERRMPI(ierr);
        ierr = PetscViewerASCIIPrintf(view,"  Memory consumption GB (CPU): %g (dense) %g (low rank) %g (total)\n", matrix_mem[0], matrix_mem[1], matrix_mem[0] + matrix_mem[1]);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
        ierr = PetscViewerASCIIPrintf(view,"  Memory consumption GB (GPU): %g (dense) %g (low rank) %g (total)\n", matrix_mem[2], matrix_mem[3], matrix_mem[2] + matrix_mem[3]);CHKERRQ(ierr);
#endif
      }
    }
  }
#if 0
  if (size == 1) {
    char filename[256];
    const char *name;

    ierr = PetscObjectGetName((PetscObject)A,&name);CHKERRQ(ierr);
    ierr = PetscSNPrintf(filename,sizeof(filename),"%s_structure.eps",name);CHKERRQ(ierr);
    outputEps(*h2opus->hmatrix,filename);
  }
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode MatH2OpusSetCoords_H2OPUS(Mat A, PetscInt spacedim, const PetscReal coords[], PetscBool cdist, MatH2OpusKernel kernel, void *kernelctx)
{
  Mat_H2OPUS     *h2opus = (Mat_H2OPUS*)A->data;
  PetscReal      *gcoords;
  PetscInt       N;
  MPI_Comm       comm;
  PetscMPIInt    size;
  PetscBool      cong;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscLayoutSetUp(A->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp(A->cmap);CHKERRQ(ierr);
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MatHasCongruentLayouts(A,&cong);CHKERRQ(ierr);
  if (!cong) SETERRQ(comm,PETSC_ERR_SUP,"Only for square matrices with congruent layouts");
  N    = A->rmap->N;
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (spacedim > 0 && size > 1 && cdist) {
    PetscSF      sf;
    MPI_Datatype dtype;

    ierr = MPI_Type_contiguous(spacedim,MPIU_REAL,&dtype);CHKERRMPI(ierr);
    ierr = MPI_Type_commit(&dtype);CHKERRMPI(ierr);

    ierr = PetscSFCreate(comm,&sf);CHKERRQ(ierr);
    ierr = PetscSFSetGraphWithPattern(sf,A->rmap,PETSCSF_PATTERN_ALLGATHER);CHKERRQ(ierr);
    ierr = PetscMalloc1(spacedim*N,&gcoords);CHKERRQ(ierr);
    ierr = PetscSFBcastBegin(sf,dtype,coords,gcoords,MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(sf,dtype,coords,gcoords,MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&sf);CHKERRQ(ierr);
    ierr = MPI_Type_free(&dtype);CHKERRMPI(ierr);
  } else gcoords = (PetscReal*)coords;

  delete h2opus->ptcloud;
  delete h2opus->kernel;
  h2opus->ptcloud = new PetscPointCloud<PetscReal>(spacedim,N,gcoords);
  if (kernel) h2opus->kernel = new PetscFunctionGenerator<PetscScalar>(kernel,spacedim,kernelctx);
  if (gcoords != coords) { ierr = PetscFree(gcoords);CHKERRQ(ierr); }
  A->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#if defined(PETSC_H2OPUS_USE_GPU)
static PetscErrorCode MatBindToCPU_H2OPUS(Mat A, PetscBool flg)
{
  PetscMPIInt    size;
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRMPI(ierr);
  if (flg && A->offloadmask == PETSC_OFFLOAD_GPU) {
    if (size > 1) {
      if (!a->dist_hmatrix_gpu) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
#if defined(H2OPUS_USE_MPI)
      if (!a->dist_hmatrix) a->dist_hmatrix = new DistributedHMatrix(*a->dist_hmatrix_gpu);
      else *a->dist_hmatrix = *a->dist_hmatrix_gpu;
#endif
    } else {
      if (!a->hmatrix_gpu) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
      if (!a->hmatrix) a->hmatrix = new HMatrix(*a->hmatrix_gpu);
      else *a->hmatrix = *a->hmatrix_gpu;
    }
    delete a->hmatrix_gpu;
    delete a->dist_hmatrix_gpu;
    a->hmatrix_gpu = NULL;
    a->dist_hmatrix_gpu = NULL;
    A->offloadmask = PETSC_OFFLOAD_CPU;
  } else if (!flg && A->offloadmask == PETSC_OFFLOAD_CPU) {
    if (size > 1) {
      if (!a->dist_hmatrix) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
#if defined(H2OPUS_USE_MPI)
      if (!a->dist_hmatrix_gpu) a->dist_hmatrix_gpu = new DistributedHMatrix_GPU(*a->dist_hmatrix);
      else *a->dist_hmatrix_gpu = *a->dist_hmatrix;
#endif
    } else {
      if (!a->hmatrix) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
      if (!a->hmatrix_gpu) a->hmatrix_gpu = new HMatrix_GPU(*a->hmatrix);
      else *a->hmatrix_gpu = *a->hmatrix;
    }
    delete a->hmatrix;
    delete a->dist_hmatrix;
    a->hmatrix = NULL;
    a->dist_hmatrix = NULL;
    A->offloadmask = PETSC_OFFLOAD_GPU;
  }
  ierr = PetscFree(A->defaultvectype);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscStrallocpy(VECCUDA,&A->defaultvectype);CHKERRQ(ierr);
  } else {
    ierr = PetscStrallocpy(VECSTANDARD,&A->defaultvectype);CHKERRQ(ierr);
  }
  A->boundtocpu = flg;
  PetscFunctionReturn(0);
}
#endif

/*MC
     MATH2OPUS = "h2opus" - A matrix type for hierarchical matrices using the H2Opus package.

   Options Database Keys:
.     -mat_type h2opus - matrix type to "h2opus" during a call to MatSetFromOptions()

   Notes:
     H2Opus implements hierarchical matrices in the H^2 flavour.
     It supports CPU or NVIDIA GPUs.
     For CPU only builds, use ./configure --download-h2opus --download-thrust to install PETSc to use H2Opus.
     In order to run on NVIDIA GPUs, use ./configure --download-h2opus --download-magma --download-kblas.
     For details and additional references, see
       "H2Opus: A distributed-memory multi-GPU software package for non-local operators",
     available at https://arxiv.org/abs/2109.05451.

   Level: beginner

.seealso: MATHTOOL, MATDENSE, MatCreateH2OpusFromKernel(), MatCreateH2OpusFromMat()
M*/
PETSC_EXTERN PetscErrorCode MatCreate_H2OPUS(Mat A)
{
  Mat_H2OPUS     *a;
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
#if defined(PETSC_H2OPUS_USE_GPU)
  ierr = PetscDeviceInitialize(PETSC_DEVICE_CUDA);CHKERRQ(ierr);
#endif
  ierr = PetscNewLog(A,&a);CHKERRQ(ierr);
  A->data = (void*)a;

  a->eta              = 0.9;
  a->leafsize         = 32;
  a->basisord         = 4;
  a->max_rank         = 64;
  a->bs               = 32;
  a->rtol             = 1.e-4;
  a->s                = 1.0;
  a->norm_max_samples = 10;
#if defined(H2OPUS_USE_MPI)
  h2opusCreateDistributedHandleComm(&a->handle,PetscObjectComm((PetscObject)A));
#else
  h2opusCreateHandle(&a->handle);
#endif
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRMPI(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATH2OPUS);CHKERRQ(ierr);
  ierr = PetscMemzero(A->ops,sizeof(struct _MatOps));CHKERRQ(ierr);

  A->ops->destroy          = MatDestroy_H2OPUS;
  A->ops->view             = MatView_H2OPUS;
  A->ops->assemblyend      = MatAssemblyEnd_H2OPUS;
  A->ops->mult             = MatMult_H2OPUS;
  A->ops->multtranspose    = MatMultTranspose_H2OPUS;
  A->ops->multadd          = MatMultAdd_H2OPUS;
  A->ops->multtransposeadd = MatMultTransposeAdd_H2OPUS;
  A->ops->scale            = MatScale_H2OPUS;
  A->ops->duplicate        = MatDuplicate_H2OPUS;
  A->ops->setfromoptions   = MatSetFromOptions_H2OPUS;
  A->ops->norm             = MatNorm_H2OPUS;
  A->ops->zeroentries      = MatZeroEntries_H2OPUS;
#if defined(PETSC_H2OPUS_USE_GPU)
  A->ops->bindtocpu        = MatBindToCPU_H2OPUS;
#endif

  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_h2opus_seqdense_C",MatProductSetFromOptions_H2OPUS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_h2opus_seqdensecuda_C",MatProductSetFromOptions_H2OPUS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_h2opus_mpidense_C",MatProductSetFromOptions_H2OPUS);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_h2opus_mpidensecuda_C",MatProductSetFromOptions_H2OPUS);CHKERRQ(ierr);
#if defined(PETSC_H2OPUS_USE_GPU)
  ierr = PetscFree(A->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECCUDA,&A->defaultvectype);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/*@C
     MatH2OpusOrthogonalize - Orthogonalize the basis tree of a hierarchical matrix.

   Input Parameter:
.     A - the matrix

   Level: intermediate

.seealso:  MatCreate(), MATH2OPUS, MatCreateH2OpusFromMat(), MatCreateH2OpusFromKernel(), MatH2OpusCompress()
*/
PetscErrorCode MatH2OpusOrthogonalize(Mat A)
{
  PetscErrorCode ierr;
  PetscBool      ish2opus;
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;
  PetscMPIInt    size;
  PetscBool      boundtocpu = PETSC_TRUE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATH2OPUS,&ish2opus);CHKERRQ(ierr);
  if (!ish2opus) PetscFunctionReturn(0);
  if (a->orthogonal) PetscFunctionReturn(0);
  HLibProfile::clear();
  ierr = PetscLogEventBegin(MAT_H2Opus_Orthog,A,0,0,0);CHKERRQ(ierr);
#if defined(PETSC_H2OPUS_USE_GPU)
  boundtocpu = A->boundtocpu;
#endif
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRMPI(ierr);
  if (size > 1) {
    if (boundtocpu) {
      if (!a->dist_hmatrix) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
#if defined(H2OPUS_USE_MPI)
      distributed_horthog(*a->dist_hmatrix, a->handle);
#endif
#if defined(PETSC_H2OPUS_USE_GPU)
      A->offloadmask = PETSC_OFFLOAD_CPU;
    } else {
      if (!a->dist_hmatrix_gpu) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
#if defined(H2OPUS_USE_MPI)
      distributed_horthog(*a->dist_hmatrix_gpu, a->handle);
#endif
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
#endif
    }
  } else {
#if defined(H2OPUS_USE_MPI)
    h2opusHandle_t handle = a->handle->handle;
#else
    h2opusHandle_t handle = a->handle;
#endif
    if (boundtocpu) {
      if (!a->hmatrix) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
      horthog(*a->hmatrix, handle);
#if defined(PETSC_H2OPUS_USE_GPU)
      A->offloadmask = PETSC_OFFLOAD_CPU;
    } else {
      if (!a->hmatrix_gpu) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      horthog(*a->hmatrix_gpu, handle);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
#endif
    }
  }
  a->orthogonal = PETSC_TRUE;
  { /* log flops */
    double gops,time,perf,dev;
    HLibProfile::getHorthogPerf(gops,time,perf,dev);
#if defined(PETSC_H2OPUS_USE_GPU)
    if (boundtocpu) {
      ierr = PetscLogFlops(1e9*gops);CHKERRQ(ierr);
    } else {
      ierr = PetscLogGpuFlops(1e9*gops);CHKERRQ(ierr);
    }
#else
    ierr = PetscLogFlops(1e9*gops);CHKERRQ(ierr);
#endif
  }
  ierr = PetscLogEventEnd(MAT_H2Opus_Orthog,A,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
     MatH2OpusCompress - Compress a hierarchical matrix.

   Input Parameters:
+     A - the matrix
-     tol - the absolute truncation threshold

   Level: intermediate

.seealso:  MatCreate(), MATH2OPUS, MatCreateH2OpusFromMat(), MatCreateH2OpusFromKernel(), MatH2OpusOrthogonalize()
*/
PetscErrorCode MatH2OpusCompress(Mat A, PetscReal tol)
{
  PetscErrorCode ierr;
  PetscBool      ish2opus;
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;
  PetscMPIInt    size;
  PetscBool      boundtocpu = PETSC_TRUE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATH2OPUS,&ish2opus);CHKERRQ(ierr);
  if (!ish2opus) PetscFunctionReturn(0);
  ierr = MatH2OpusOrthogonalize(A);CHKERRQ(ierr);
  HLibProfile::clear();
  ierr = PetscLogEventBegin(MAT_H2Opus_Compress,A,0,0,0);CHKERRQ(ierr);
#if defined(PETSC_H2OPUS_USE_GPU)
  boundtocpu = A->boundtocpu;
#endif
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRMPI(ierr);
  if (size > 1) {
    if (boundtocpu) {
      if (!a->dist_hmatrix) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
#if defined(H2OPUS_USE_MPI)
      distributed_hcompress(*a->dist_hmatrix, tol, a->handle);
#endif
#if defined(PETSC_H2OPUS_USE_GPU)
      A->offloadmask = PETSC_OFFLOAD_CPU;
    } else {
      if (!a->dist_hmatrix_gpu) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
#if defined(H2OPUS_USE_MPI)
      distributed_hcompress(*a->dist_hmatrix_gpu, tol, a->handle);
#endif
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
#endif
    }
  } else {
#if defined(H2OPUS_USE_MPI)
    h2opusHandle_t handle = a->handle->handle;
#else
    h2opusHandle_t handle = a->handle;
#endif
    if (boundtocpu) {
      if (!a->hmatrix) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
      hcompress(*a->hmatrix, tol, handle);
#if defined(PETSC_H2OPUS_USE_GPU)
      A->offloadmask = PETSC_OFFLOAD_CPU;
    } else {
      if (!a->hmatrix_gpu) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
      ierr = PetscLogGpuTimeBegin();CHKERRQ(ierr);
      hcompress(*a->hmatrix_gpu, tol, handle);
      ierr = PetscLogGpuTimeEnd();CHKERRQ(ierr);
#endif
    }
  }
  { /* log flops */
    double gops,time,perf,dev;
    HLibProfile::getHcompressPerf(gops,time,perf,dev);
#if defined(PETSC_H2OPUS_USE_GPU)
    if (boundtocpu) {
      ierr = PetscLogFlops(1e9*gops);CHKERRQ(ierr);
    } else {
      ierr = PetscLogGpuFlops(1e9*gops);CHKERRQ(ierr);
    }
#else
    ierr = PetscLogFlops(1e9*gops);CHKERRQ(ierr);
#endif
  }
  ierr = PetscLogEventEnd(MAT_H2Opus_Compress,A,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
     MatH2OpusSetSamplingMat - Set a matrix to be sampled from matrix vector product to construct a hierarchical matrix.

   Input Parameters:
+     A - the hierarchical matrix
.     B - the matrix to be sampled
.     bs - maximum number of samples to be taken concurrently
-     tol - relative tolerance for construction

   Notes: Need to call MatAssemblyBegin/End() to update the hierarchical matrix.

   Level: intermediate

.seealso:  MatCreate(), MATH2OPUS, MatCreateH2OpusFromMat(), MatCreateH2OpusFromKernel(), MatH2OpusCompress(), MatH2OpusOrthogonalize()
*/
PetscErrorCode MatH2OpusSetSamplingMat(Mat A, Mat B, PetscInt bs, PetscReal tol)
{
  PetscBool      ish2opus;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  if (B) PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATH2OPUS,&ish2opus);CHKERRQ(ierr);
  if (ish2opus) {
    Mat_H2OPUS *a = (Mat_H2OPUS*)A->data;

    if (!a->sampler) a->sampler = new PetscMatrixSampler();
    a->sampler->SetSamplingMat(B);
    if (bs > 0) a->bs = bs;
    if (tol > 0.) a->rtol = tol;
    delete a->kernel;
  }
  PetscFunctionReturn(0);
}

/*@C
     MatCreateH2OpusFromKernel - Creates a MATH2OPUS from a user-supplied kernel.

   Input Parameters:
+     comm - MPI communicator
.     m - number of local rows (or PETSC_DECIDE to have calculated if M is given)
.     n - number of local columns (or PETSC_DECIDE to have calculated if N is given)
.     M - number of global rows (or PETSC_DETERMINE to have calculated if m is given)
.     N - number of global columns (or PETSC_DETERMINE to have calculated if n is given)
.     spacedim - dimension of the space coordinates
.     coords - coordinates of the points
.     cdist - whether or not coordinates are distributed
.     kernel - computational kernel (or NULL)
.     kernelctx - kernel context
.     eta - admissibility condition tolerance
.     leafsize - leaf size in cluster tree
-     basisord - approximation order for Chebychev interpolation of low-rank blocks

   Output Parameter:
.     nA - matrix

   Options Database Keys:
+     -mat_h2opus_leafsize <PetscInt>
.     -mat_h2opus_eta <PetscReal>
.     -mat_h2opus_order <PetscInt> - Chebychev approximation order
-     -mat_h2opus_normsamples <PetscInt> - Maximum bumber of samples to be when estimating norms

   Level: intermediate

.seealso:  MatCreate(), MATH2OPUS, MatCreateH2OpusFromMat()
@*/
PetscErrorCode MatCreateH2OpusFromKernel(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt spacedim, const PetscReal coords[], PetscBool cdist, MatH2OpusKernel kernel, void *kernelctx, PetscReal eta, PetscInt leafsize, PetscInt basisord, Mat* nA)
{
  Mat            A;
  Mat_H2OPUS     *h2opus;
#if defined(PETSC_H2OPUS_USE_GPU)
  PetscBool      iscpu = PETSC_FALSE;
#else
  PetscBool      iscpu = PETSC_TRUE;
#endif
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (m != n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Different row and column local sizes are not supported");
  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,m,n,M,N);CHKERRQ(ierr);
  if (M != N) SETERRQ(comm,PETSC_ERR_SUP,"Rectangular matrices are not supported");
  ierr = MatSetType(A,MATH2OPUS);CHKERRQ(ierr);
  ierr = MatBindToCPU(A,iscpu);CHKERRQ(ierr);
  ierr = MatH2OpusSetCoords_H2OPUS(A,spacedim,coords,cdist,kernel,kernelctx);CHKERRQ(ierr);

  h2opus = (Mat_H2OPUS*)A->data;
  if (eta > 0.) h2opus->eta = eta;
  if (leafsize > 0) h2opus->leafsize = leafsize;
  if (basisord > 0) h2opus->basisord = basisord;

  *nA = A;
  PetscFunctionReturn(0);
}

/*@C
     MatCreateH2OpusFromMat - Creates a MATH2OPUS sampling from a user-supplied operator.

   Input Parameters:
+     B - the matrix to be sampled
.     spacedim - dimension of the space coordinates
.     coords - coordinates of the points
.     cdist - whether or not coordinates are distributed
.     eta - admissibility condition tolerance
.     leafsize - leaf size in cluster tree
.     maxrank - maximum rank allowed
.     bs - maximum number of samples to be taken concurrently
-     rtol - relative tolerance for construction

   Output Parameter:
.     nA - matrix

   Options Database Keys:
+     -mat_h2opus_leafsize <PetscInt>
.     -mat_h2opus_eta <PetscReal>
.     -mat_h2opus_maxrank <PetscInt>
.     -mat_h2opus_samples <PetscInt>
.     -mat_h2opus_rtol <PetscReal>
.     -mat_h2opus_check <PetscBool> - Check error when constructing from sampling during MatAssemblyEnd()
.     -mat_h2opus_hara_verbose <PetscBool> - Verbose output from hara construction
-     -mat_h2opus_normsamples <PetscInt> - Maximum bumber of samples to be when estimating norms

   Notes: not available in parallel

   Level: intermediate

.seealso:  MatCreate(), MATH2OPUS, MatCreateH2OpusFromKernel()
@*/
PetscErrorCode MatCreateH2OpusFromMat(Mat B, PetscInt spacedim, const PetscReal coords[], PetscBool cdist, PetscReal eta, PetscInt leafsize, PetscInt maxrank, PetscInt bs, PetscReal rtol, Mat *nA)
{
  Mat            A;
  Mat_H2OPUS     *h2opus;
  MPI_Comm       comm;
  PetscBool      boundtocpu = PETSC_TRUE;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(B,spacedim,2);
  PetscValidLogicalCollectiveBool(B,cdist,4);
  PetscValidLogicalCollectiveReal(B,eta,5);
  PetscValidLogicalCollectiveInt(B,leafsize,6);
  PetscValidLogicalCollectiveInt(B,maxrank,7);
  PetscValidLogicalCollectiveInt(B,bs,8);
  PetscValidLogicalCollectiveReal(B,rtol,9);
  PetscValidPointer(nA,10);
  ierr = PetscObjectGetComm((PetscObject)B,&comm);CHKERRQ(ierr);
  if (B->rmap->n != B->cmap->n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Different row and column local sizes are not supported");
  if (B->rmap->N != B->cmap->N) SETERRQ(comm,PETSC_ERR_SUP,"Rectangular matrices are not supported");
  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,B->rmap->n,B->cmap->n,B->rmap->N,B->cmap->N);CHKERRQ(ierr);
#if defined(PETSC_H2OPUS_USE_GPU)
  {
    PetscBool iscuda;
    VecType   vtype;

    ierr = MatGetVecType(B,&vtype);CHKERRQ(ierr);
    ierr = PetscStrcmp(vtype,VECCUDA,&iscuda);CHKERRQ(ierr);
    if (!iscuda) {
      ierr = PetscStrcmp(vtype,VECSEQCUDA,&iscuda);CHKERRQ(ierr);
      if (!iscuda) {
        ierr = PetscStrcmp(vtype,VECMPICUDA,&iscuda);CHKERRQ(ierr);
      }
    }
    if (iscuda && !B->boundtocpu) boundtocpu = PETSC_FALSE;
  }
#endif
  ierr = MatSetType(A,MATH2OPUS);CHKERRQ(ierr);
  ierr = MatBindToCPU(A,boundtocpu);CHKERRQ(ierr);
  if (spacedim) {
    ierr = MatH2OpusSetCoords_H2OPUS(A,spacedim,coords,cdist,NULL,NULL);CHKERRQ(ierr);
  }
  ierr = MatPropagateSymmetryOptions(B,A);CHKERRQ(ierr);
  /* if (!A->symmetric) SETERRQ(comm,PETSC_ERR_SUP,"Unsymmetric sampling does not work"); */

  h2opus = (Mat_H2OPUS*)A->data;
  h2opus->sampler = new PetscMatrixSampler(B);
  if (eta > 0.) h2opus->eta = eta;
  if (leafsize > 0) h2opus->leafsize = leafsize;
  if (maxrank > 0) h2opus->max_rank = maxrank;
  if (bs > 0) h2opus->bs = bs;
  if (rtol > 0.) h2opus->rtol = rtol;
  *nA = A;
  A->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

/*@C
     MatH2OpusGetIndexMap - Access reordering index set.

   Input Parameters:
.     A - the matrix

   Output Parameter:
.     indexmap - the index set for the reordering

   Level: intermediate

.seealso:  MatCreate(), MATH2OPUS, MatCreateH2OpusFromMat(), MatCreateH2OpusFromKernel()
@*/
PetscErrorCode MatH2OpusGetIndexMap(Mat A, IS *indexmap)
{
  PetscBool      ish2opus;
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidPointer(indexmap,2);
  if (!A->assembled) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  ierr = PetscObjectTypeCompare((PetscObject)A,MATH2OPUS,&ish2opus);CHKERRQ(ierr);
  if (!ish2opus) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not for type %s",((PetscObject)A)->type_name);
  *indexmap = a->h2opus_indexmap;
  PetscFunctionReturn(0);
}

/*@C
     MatH2OpusMapVec - Maps a vector between PETSc and H2Opus ordering

   Input Parameters:
+     A - the matrix
.     nativetopetsc - if true, maps from H2Opus ordering to PETSc ordering. If false, applies the reverse map
-     in - the vector to be mapped

   Output Parameter:
.     out - the newly created mapped vector

   Level: intermediate

.seealso:  MatCreate(), MATH2OPUS, MatCreateH2OpusFromMat(), MatCreateH2OpusFromKernel()
*/
PetscErrorCode MatH2OpusMapVec(Mat A, PetscBool nativetopetsc, Vec in, Vec* out)
{
  PetscBool      ish2opus;
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;
  PetscScalar    *xin,*xout;
  PetscBool      nm;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidLogicalCollectiveBool(A,nativetopetsc,2);
  PetscValidHeaderSpecific(in,VEC_CLASSID,3);
  PetscValidPointer(out,4);
  if (!A->assembled) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  ierr = PetscObjectTypeCompare((PetscObject)A,MATH2OPUS,&ish2opus);CHKERRQ(ierr);
  if (!ish2opus) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not for type %s",((PetscObject)A)->type_name);
  nm   = a->nativemult;
  ierr = MatH2OpusSetNativeMult(A,(PetscBool)!nativetopetsc);CHKERRQ(ierr);
  ierr = MatCreateVecs(A,out,NULL);CHKERRQ(ierr);
  ierr = MatH2OpusSetNativeMult(A,nm);CHKERRQ(ierr);
  if (!a->sf) { /* same ordering */
    ierr = VecCopy(in,*out);CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }
  ierr = VecGetArrayRead(in,(const PetscScalar**)&xin);CHKERRQ(ierr);
  ierr = VecGetArrayWrite(*out,&xout);CHKERRQ(ierr);
  if (nativetopetsc) {
    ierr = PetscSFReduceBegin(a->sf,MPIU_SCALAR,xin,xout,MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(a->sf,MPIU_SCALAR,xin,xout,MPI_REPLACE);CHKERRQ(ierr);
  } else {
    ierr = PetscSFBcastBegin(a->sf,MPIU_SCALAR,xin,xout,MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(a->sf,MPIU_SCALAR,xin,xout,MPI_REPLACE);CHKERRQ(ierr);
  }
  ierr = VecRestoreArrayRead(in,(const PetscScalar**)&xin);CHKERRQ(ierr);
  ierr = VecRestoreArrayWrite(*out,&xout);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif
