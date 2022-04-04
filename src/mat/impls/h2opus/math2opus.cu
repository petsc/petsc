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
PETSC_INTERN PetscErrorCode MatDenseGetH2OpusVectorSF(Mat,PetscSF,PetscSF*);
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

  PetscFunctionBegin;
#if defined(H2OPUS_USE_MPI)
  h2opusDestroyDistributedHandle(a->handle);
#else
  h2opusDestroyHandle(a->handle);
#endif
  delete a->dist_hmatrix;
  delete a->hmatrix;
  PetscCall(PetscSFDestroy(&a->sf));
  PetscCall(PetscLayoutDestroy(&a->h2opus_rmap));
  PetscCall(PetscLayoutDestroy(&a->h2opus_cmap));
  PetscCall(ISDestroy(&a->h2opus_indexmap));
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
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_h2opus_seqdense_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_h2opus_seqdensecuda_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_h2opus_mpidense_C",NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_h2opus_mpidensecuda_C",NULL));
  PetscCall(PetscObjectChangeTypeName((PetscObject)A,NULL));
  PetscCall(PetscFree(A->data));
  PetscFunctionReturn(0);
}

PetscErrorCode MatH2OpusSetNativeMult(Mat A, PetscBool nm)
{
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;
  PetscBool      ish2opus;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidLogicalCollectiveBool(A,nm,2);
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATH2OPUS,&ish2opus));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidPointer(nm,2);
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATH2OPUS,&ish2opus));
  PetscCheck(ish2opus,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not for type %s",((PetscObject)A)->type_name);
  *nm = a->nativemult;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatNorm_H2OPUS(Mat A, NormType normtype, PetscReal* n)
{
  PetscBool      ish2opus;
  PetscInt       nmax = PETSC_DECIDE;
  Mat_H2OPUS     *a = NULL;
  PetscBool      mult = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATH2OPUS,&ish2opus));
  if (ish2opus) { /* set userdefine number of samples and fastpath for mult (norms are order independent) */
    a = (Mat_H2OPUS*)A->data;

    nmax = a->norm_max_samples;
    mult = a->nativemult;
    PetscCall(MatH2OpusSetNativeMult(A,PETSC_TRUE));
  } else {
    PetscCall(PetscOptionsGetInt(((PetscObject)A)->options,((PetscObject)A)->prefix,"-mat_approximate_norm_samples",&nmax,NULL));
  }
  PetscCall(MatApproximateNorm_Private(A,normtype,nmax,n));
  if (a) PetscCall(MatH2OpusSetNativeMult(A,mult));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatH2OpusResizeBuffers_Private(Mat A, PetscInt xN, PetscInt yN)
{
  Mat_H2OPUS     *h2opus = (Mat_H2OPUS*)A->data;
  PetscInt       n;
  PetscBool      boundtocpu = PETSC_TRUE;

  PetscFunctionBegin;
#if defined(PETSC_H2OPUS_USE_GPU)
  boundtocpu = A->boundtocpu;
#endif
  PetscCall(PetscSFGetGraph(h2opus->sf,NULL,&n,NULL,NULL));
  if (boundtocpu) {
    if (h2opus->xxs < xN) { h2opus->xx->resize(n*xN); h2opus->xxs = xN; }
    if (h2opus->yys < yN) { h2opus->yy->resize(n*yN); h2opus->yys = yN; }
  }
#if defined(PETSC_H2OPUS_USE_GPU)
  if (!boundtocpu) {
    if (h2opus->xxs_gpu < xN) { h2opus->xx_gpu->resize(n*xN); h2opus->xxs_gpu = xN; }
    if (h2opus->yys_gpu < yN) { h2opus->yy_gpu->resize(n*yN); h2opus->yys_gpu = yN; }
  }
#endif
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

  PetscFunctionBegin;
  HLibProfile::clear();
#if defined(PETSC_H2OPUS_USE_GPU)
  boundtocpu = A->boundtocpu;
#endif
  PetscCall(MatDenseGetLDA(B,&blda));
  PetscCall(MatDenseGetLDA(C,&clda));
  if (usesf) {
    PetscInt n;

    PetscCall(MatDenseGetH2OpusVectorSF(B,h2opus->sf,&bsf));
    PetscCall(MatDenseGetH2OpusVectorSF(C,h2opus->sf,&csf));

    PetscCall(MatH2OpusResizeBuffers_Private(A,B->cmap->N,C->cmap->N));
    PetscCall(PetscSFGetGraph(h2opus->sf,NULL,&n,NULL,NULL));
    blda = n;
    clda = n;
  }
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  if (boundtocpu) {
    PetscCall(MatDenseGetArrayRead(B,(const PetscScalar**)&xx));
    PetscCall(MatDenseGetArrayWrite(C,&yy));
    if (usesf) {
      uxx  = MatH2OpusGetThrustPointer(*h2opus->xx);
      uyy  = MatH2OpusGetThrustPointer(*h2opus->yy);
      PetscCall(PetscSFBcastBegin(bsf,MPIU_SCALAR,xx,uxx,MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(bsf,MPIU_SCALAR,xx,uxx,MPI_REPLACE));
    } else {
      uxx = xx;
      uyy = yy;
    }
    if (size > 1) {
      PetscCheck(h2opus->dist_hmatrix,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing distributed CPU matrix");
      PetscCheck(!transA || A->symmetric,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MatMultTranspose not yet coded in parallel");
#if defined(H2OPUS_USE_MPI)
      distributed_hgemv(/* transA ? H2Opus_Trans : H2Opus_NoTrans, */h2opus->s, *h2opus->dist_hmatrix, uxx, blda, 0.0, uyy, clda, B->cmap->N, h2opus->handle);
#endif
    } else {
      PetscCheck(h2opus->hmatrix,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
      hgemv(transA ? H2Opus_Trans : H2Opus_NoTrans, h2opus->s, *h2opus->hmatrix, uxx, blda, 0.0, uyy, clda, B->cmap->N, handle);
    }
    PetscCall(MatDenseRestoreArrayRead(B,(const PetscScalar**)&xx));
    if (usesf) {
      PetscCall(PetscSFReduceBegin(csf,MPIU_SCALAR,uyy,yy,MPI_REPLACE));
      PetscCall(PetscSFReduceEnd(csf,MPIU_SCALAR,uyy,yy,MPI_REPLACE));
    }
    PetscCall(MatDenseRestoreArrayWrite(C,&yy));
#if defined(PETSC_H2OPUS_USE_GPU)
  } else {
    PetscBool ciscuda,biscuda;

    /* If not of type seqdensecuda, convert on the fly (i.e. allocate GPU memory) */
    PetscCall(PetscObjectTypeCompareAny((PetscObject)B,&biscuda,MATSEQDENSECUDA,MATMPIDENSECUDA,""));
    if (!biscuda) {
      PetscCall(MatConvert(B,MATDENSECUDA,MAT_INPLACE_MATRIX,&B));
    }
    PetscCall(PetscObjectTypeCompareAny((PetscObject)C,&ciscuda,MATSEQDENSECUDA,MATMPIDENSECUDA,""));
    if (!ciscuda) {
      C->assembled = PETSC_TRUE;
      PetscCall(MatConvert(C,MATDENSECUDA,MAT_INPLACE_MATRIX,&C));
    }
    PetscCall(MatDenseCUDAGetArrayRead(B,(const PetscScalar**)&xx));
    PetscCall(MatDenseCUDAGetArrayWrite(C,&yy));
    if (usesf) {
      uxx  = MatH2OpusGetThrustPointer(*h2opus->xx_gpu);
      uyy  = MatH2OpusGetThrustPointer(*h2opus->yy_gpu);
      PetscCall(PetscSFBcastBegin(bsf,MPIU_SCALAR,xx,uxx,MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(bsf,MPIU_SCALAR,xx,uxx,MPI_REPLACE));
    } else {
      uxx = xx;
      uyy = yy;
    }
    PetscCall(PetscLogGpuTimeBegin());
    if (size > 1) {
      PetscCheck(h2opus->dist_hmatrix_gpu,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing distributed GPU matrix");
      PetscCheck(!transA || A->symmetric,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MatMultTranspose not yet coded in parallel");
#if defined(H2OPUS_USE_MPI)
      distributed_hgemv(/* transA ? H2Opus_Trans : H2Opus_NoTrans, */h2opus->s, *h2opus->dist_hmatrix_gpu, uxx, blda, 0.0, uyy, clda, B->cmap->N, h2opus->handle);
#endif
    } else {
      PetscCheck(h2opus->hmatrix_gpu,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
      hgemv(transA ? H2Opus_Trans : H2Opus_NoTrans, h2opus->s, *h2opus->hmatrix_gpu, uxx, blda, 0.0, uyy, clda, B->cmap->N, handle);
    }
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(MatDenseCUDARestoreArrayRead(B,(const PetscScalar**)&xx));
    if (usesf) {
      PetscCall(PetscSFReduceBegin(csf,MPIU_SCALAR,uyy,yy,MPI_REPLACE));
      PetscCall(PetscSFReduceEnd(csf,MPIU_SCALAR,uyy,yy,MPI_REPLACE));
    }
    PetscCall(MatDenseCUDARestoreArrayWrite(C,&yy));
    if (!biscuda) {
      PetscCall(MatConvert(B,MATDENSE,MAT_INPLACE_MATRIX,&B));
    }
    if (!ciscuda) {
      PetscCall(MatConvert(C,MATDENSE,MAT_INPLACE_MATRIX,&C));
    }
#endif
  }
  { /* log flops */
    double gops,time,perf,dev;
    HLibProfile::getHgemvPerf(gops,time,perf,dev);
#if defined(PETSC_H2OPUS_USE_GPU)
    if (boundtocpu) {
      PetscCall(PetscLogFlops(1e9*gops));
    } else {
      PetscCall(PetscLogGpuFlops(1e9*gops));
    }
#else
    PetscCall(PetscLogFlops(1e9*gops));
#endif
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductNumeric_H2OPUS(Mat C)
{
  Mat_Product    *product = C->product;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  switch (product->type) {
  case MATPRODUCT_AB:
    PetscCall(MatMultNKernel_H2OPUS(product->A,PETSC_FALSE,product->B,C));
    break;
  case MATPRODUCT_AtB:
    PetscCall(MatMultNKernel_H2OPUS(product->A,PETSC_TRUE,product->B,C));
    break;
  default:
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatProduct type %s is not supported",MatProductTypes[product->type]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSymbolic_H2OPUS(Mat C)
{
  Mat_Product    *product = C->product;
  PetscBool      cisdense;
  Mat            A,B;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  A = product->A;
  B = product->B;
  switch (product->type) {
  case MATPRODUCT_AB:
    PetscCall(MatSetSizes(C,A->rmap->n,B->cmap->n,A->rmap->N,B->cmap->N));
    PetscCall(MatSetBlockSizesFromMats(C,product->A,product->B));
    PetscCall(PetscObjectTypeCompareAny((PetscObject)C,&cisdense,MATSEQDENSE,MATMPIDENSE,MATSEQDENSECUDA,MATMPIDENSECUDA,""));
    if (!cisdense) PetscCall(MatSetType(C,((PetscObject)product->B)->type_name));
    PetscCall(MatSetUp(C));
    break;
  case MATPRODUCT_AtB:
    PetscCall(MatSetSizes(C,A->cmap->n,B->cmap->n,A->cmap->N,B->cmap->N));
    PetscCall(MatSetBlockSizesFromMats(C,product->A,product->B));
    PetscCall(PetscObjectTypeCompareAny((PetscObject)C,&cisdense,MATSEQDENSE,MATMPIDENSE,MATSEQDENSECUDA,MATMPIDENSECUDA,""));
    if (!cisdense) PetscCall(MatSetType(C,((PetscObject)product->B)->type_name));
    PetscCall(MatSetUp(C));
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

  PetscFunctionBegin;
  HLibProfile::clear();
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
#if defined(PETSC_H2OPUS_USE_GPU)
  boundtocpu = A->boundtocpu;
#endif
  if (usesf) {
    PetscCall(PetscSFGetGraph(h2opus->sf,NULL,&n,NULL,NULL));
  } else n = A->rmap->n;
  if (boundtocpu) {
    PetscCall(VecGetArrayRead(x,(const PetscScalar**)&xx));
    if (sy == 0.0) {
      PetscCall(VecGetArrayWrite(y,&yy));
    } else {
      PetscCall(VecGetArray(y,&yy));
    }
    if (usesf) {
      uxx = MatH2OpusGetThrustPointer(*h2opus->xx);
      uyy = MatH2OpusGetThrustPointer(*h2opus->yy);

      PetscCall(PetscSFBcastBegin(h2opus->sf,MPIU_SCALAR,xx,uxx,MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(h2opus->sf,MPIU_SCALAR,xx,uxx,MPI_REPLACE));
      if (sy != 0.0) {
        PetscCall(PetscSFBcastBegin(h2opus->sf,MPIU_SCALAR,yy,uyy,MPI_REPLACE));
        PetscCall(PetscSFBcastEnd(h2opus->sf,MPIU_SCALAR,yy,uyy,MPI_REPLACE));
      }
    } else {
      uxx = xx;
      uyy = yy;
    }
    if (size > 1) {
      PetscCheck(h2opus->dist_hmatrix,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing distributed CPU matrix");
      PetscCheck(!trans || A->symmetric,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MatMultTranspose not yet coded in parallel");
#if defined(H2OPUS_USE_MPI)
      distributed_hgemv(/*trans ? H2Opus_Trans : H2Opus_NoTrans, */h2opus->s, *h2opus->dist_hmatrix, uxx, n, sy, uyy, n, 1, h2opus->handle);
#endif
    } else {
      PetscCheck(h2opus->hmatrix,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
      hgemv(trans ? H2Opus_Trans : H2Opus_NoTrans, h2opus->s, *h2opus->hmatrix, uxx, n, sy, uyy, n, 1, handle);
    }
    PetscCall(VecRestoreArrayRead(x,(const PetscScalar**)&xx));
    if (usesf) {
      PetscCall(PetscSFReduceBegin(h2opus->sf,MPIU_SCALAR,uyy,yy,MPI_REPLACE));
      PetscCall(PetscSFReduceEnd(h2opus->sf,MPIU_SCALAR,uyy,yy,MPI_REPLACE));
    }
    if (sy == 0.0) {
      PetscCall(VecRestoreArrayWrite(y,&yy));
    } else {
      PetscCall(VecRestoreArray(y,&yy));
    }
#if defined(PETSC_H2OPUS_USE_GPU)
  } else {
    PetscCall(VecCUDAGetArrayRead(x,(const PetscScalar**)&xx));
    if (sy == 0.0) {
      PetscCall(VecCUDAGetArrayWrite(y,&yy));
    } else {
      PetscCall(VecCUDAGetArray(y,&yy));
    }
    if (usesf) {
      uxx = MatH2OpusGetThrustPointer(*h2opus->xx_gpu);
      uyy = MatH2OpusGetThrustPointer(*h2opus->yy_gpu);

      PetscCall(PetscSFBcastBegin(h2opus->sf,MPIU_SCALAR,xx,uxx,MPI_REPLACE));
      PetscCall(PetscSFBcastEnd(h2opus->sf,MPIU_SCALAR,xx,uxx,MPI_REPLACE));
      if (sy != 0.0) {
        PetscCall(PetscSFBcastBegin(h2opus->sf,MPIU_SCALAR,yy,uyy,MPI_REPLACE));
        PetscCall(PetscSFBcastEnd(h2opus->sf,MPIU_SCALAR,yy,uyy,MPI_REPLACE));
      }
    } else {
      uxx = xx;
      uyy = yy;
    }
    PetscCall(PetscLogGpuTimeBegin());
    if (size > 1) {
      PetscCheck(h2opus->dist_hmatrix_gpu,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing distributed GPU matrix");
      PetscCheck(!trans || A->symmetric,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MatMultTranspose not yet coded in parallel");
#if defined(H2OPUS_USE_MPI)
      distributed_hgemv(/*trans ? H2Opus_Trans : H2Opus_NoTrans, */h2opus->s, *h2opus->dist_hmatrix_gpu, uxx, n, sy, uyy, n, 1, h2opus->handle);
#endif
    } else {
      PetscCheck(h2opus->hmatrix_gpu,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
      hgemv(trans ? H2Opus_Trans : H2Opus_NoTrans, h2opus->s, *h2opus->hmatrix_gpu, uxx, n, sy, uyy, n, 1, handle);
    }
    PetscCall(PetscLogGpuTimeEnd());
    PetscCall(VecCUDARestoreArrayRead(x,(const PetscScalar**)&xx));
    if (usesf) {
      PetscCall(PetscSFReduceBegin(h2opus->sf,MPIU_SCALAR,uyy,yy,MPI_REPLACE));
      PetscCall(PetscSFReduceEnd(h2opus->sf,MPIU_SCALAR,uyy,yy,MPI_REPLACE));
    }
    if (sy == 0.0) {
      PetscCall(VecCUDARestoreArrayWrite(y,&yy));
    } else {
      PetscCall(VecCUDARestoreArray(y,&yy));
    }
#endif
  }
  { /* log flops */
    double gops,time,perf,dev;
    HLibProfile::getHgemvPerf(gops,time,perf,dev);
#if defined(PETSC_H2OPUS_USE_GPU)
    if (boundtocpu) {
      PetscCall(PetscLogFlops(1e9*gops));
    } else {
      PetscCall(PetscLogGpuFlops(1e9*gops));
    }
#else
    PetscCall(PetscLogFlops(1e9*gops));
#endif
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_H2OPUS(Mat A, Vec x, Vec y)
{
  PetscBool      xiscuda,yiscuda;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)x,&xiscuda,VECSEQCUDA,VECMPICUDA,""));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)y,&yiscuda,VECSEQCUDA,VECMPICUDA,""));
  PetscCall(MatH2OpusUpdateIfNeeded(A,!xiscuda || !yiscuda));
  PetscCall(MatMultKernel_H2OPUS(A,x,0.0,y,PETSC_TRUE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_H2OPUS(Mat A, Vec x, Vec y)
{
  PetscBool      xiscuda,yiscuda;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompareAny((PetscObject)x,&xiscuda,VECSEQCUDA,VECMPICUDA,""));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)y,&yiscuda,VECSEQCUDA,VECMPICUDA,""));
  PetscCall(MatH2OpusUpdateIfNeeded(A,!xiscuda || !yiscuda));
  PetscCall(MatMultKernel_H2OPUS(A,x,0.0,y,PETSC_FALSE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTransposeAdd_H2OPUS(Mat A, Vec x, Vec y, Vec z)
{
  PetscBool      xiscuda,ziscuda;

  PetscFunctionBegin;
  PetscCall(VecCopy(y,z));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)x,&xiscuda,VECSEQCUDA,VECMPICUDA,""));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)z,&ziscuda,VECSEQCUDA,VECMPICUDA,""));
  PetscCall(MatH2OpusUpdateIfNeeded(A,!xiscuda || !ziscuda));
  PetscCall(MatMultKernel_H2OPUS(A,x,1.0,z,PETSC_TRUE));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultAdd_H2OPUS(Mat A, Vec x, Vec y, Vec z)
{
  PetscBool      xiscuda,ziscuda;

  PetscFunctionBegin;
  PetscCall(VecCopy(y,z));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)x,&xiscuda,VECSEQCUDA,VECMPICUDA,""));
  PetscCall(PetscObjectTypeCompareAny((PetscObject)z,&ziscuda,VECSEQCUDA,VECMPICUDA,""));
  PetscCall(MatH2OpusUpdateIfNeeded(A,!xiscuda || !ziscuda));
  PetscCall(MatMultKernel_H2OPUS(A,x,1.0,z,PETSC_FALSE));
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

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"H2OPUS options");
  PetscCall(PetscOptionsInt("-mat_h2opus_leafsize","Leaf size of cluster tree",NULL,a->leafsize,&a->leafsize,NULL));
  PetscCall(PetscOptionsReal("-mat_h2opus_eta","Admissibility condition tolerance",NULL,a->eta,&a->eta,NULL));
  PetscCall(PetscOptionsInt("-mat_h2opus_order","Basis order for off-diagonal sampling when constructed from kernel",NULL,a->basisord,&a->basisord,NULL));
  PetscCall(PetscOptionsInt("-mat_h2opus_maxrank","Maximum rank when constructed from matvecs",NULL,a->max_rank,&a->max_rank,NULL));
  PetscCall(PetscOptionsInt("-mat_h2opus_samples","Maximum number of samples to be taken concurrently when constructing from matvecs",NULL,a->bs,&a->bs,NULL));
  PetscCall(PetscOptionsInt("-mat_h2opus_normsamples","Maximum bumber of samples to be when estimating norms",NULL,a->norm_max_samples,&a->norm_max_samples,NULL));
  PetscCall(PetscOptionsReal("-mat_h2opus_rtol","Relative tolerance for construction from sampling",NULL,a->rtol,&a->rtol,NULL));
  PetscCall(PetscOptionsBool("-mat_h2opus_check","Check error when constructing from sampling during MatAssemblyEnd()",NULL,a->check_construction,&a->check_construction,NULL));
  PetscCall(PetscOptionsBool("-mat_h2opus_hara_verbose","Verbose output from hara construction",NULL,a->hara_verbose,&a->hara_verbose,NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode MatH2OpusSetCoords_H2OPUS(Mat,PetscInt,const PetscReal[],PetscBool,MatH2OpusKernel,void*);

static PetscErrorCode MatH2OpusInferCoordinates_Private(Mat A)
{
  Mat_H2OPUS        *a = (Mat_H2OPUS*)A->data;
  Vec               c;
  PetscInt          spacedim;
  const PetscScalar *coords;

  PetscFunctionBegin;
  if (a->ptcloud) PetscFunctionReturn(0);
  PetscCall(PetscObjectQuery((PetscObject)A,"__math2opus_coords",(PetscObject*)&c));
  if (!c && a->sampler) {
    Mat S = a->sampler->GetSamplingMat();

    PetscCall(PetscObjectQuery((PetscObject)S,"__math2opus_coords",(PetscObject*)&c));
  }
  if (!c) {
    PetscCall(MatH2OpusSetCoords_H2OPUS(A,-1,NULL,PETSC_FALSE,NULL,NULL));
  } else {
    PetscCall(VecGetArrayRead(c,&coords));
    PetscCall(VecGetBlockSize(c,&spacedim));
    PetscCall(MatH2OpusSetCoords_H2OPUS(A,spacedim,coords,PETSC_FALSE,NULL,NULL));
    PetscCall(VecRestoreArrayRead(c,&coords));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetUpMultiply_H2OPUS(Mat A)
{
  MPI_Comm       comm;
  PetscMPIInt    size;
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;
  PetscInt       n = 0,*idx = NULL;
  int            *iidx = NULL;
  PetscCopyMode  own;
  PetscBool      rid;

  PetscFunctionBegin;
  if (a->multsetup) PetscFunctionReturn(0);
  if (a->sf) { /* MatDuplicate_H2OPUS takes reference to the SF */
    PetscCall(PetscSFGetGraph(a->sf,NULL,&n,NULL,NULL));
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
    PetscCall(PetscObjectGetComm((PetscObject)A,&comm));
    PetscCallMPI(MPI_Comm_size(comm,&size));
    if (!a->h2opus_indexmap) {
      if (size > 1) {
        PetscCheck(a->dist_hmatrix,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing distributed CPU matrix");
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
        PetscCall(PetscMalloc1(n,&idx));
        for (i=0;i<n;i++) idx[i] = iidx[i];
      } else {
        own  = PETSC_COPY_VALUES;
        idx  = (PetscInt*)iidx;
      }
      PetscCall(ISCreateGeneral(comm,n,idx,own,&is));
      PetscCall(ISSetPermutation(is));
      PetscCall(ISViewFromOptions(is,(PetscObject)A,"-mat_h2opus_indexmap_view"));
      a->h2opus_indexmap = is;
    }
    PetscCall(ISGetLocalSize(a->h2opus_indexmap,&n));
    PetscCall(ISGetIndices(a->h2opus_indexmap,(const PetscInt **)&idx));
    rid  = (PetscBool)(n == A->rmap->n);
    PetscCall(MPIU_Allreduce(MPI_IN_PLACE,&rid,1,MPIU_BOOL,MPI_LAND,comm));
    if (rid) {
      PetscCall(ISIdentity(a->h2opus_indexmap,&rid));
    }
    if (!rid) {
      if (size > 1) { /* Parallel distribution may be different, save it here for fast path in MatMult (see MatH2OpusSetNativeMult) */
        PetscCall(PetscLayoutCreate(comm,&a->h2opus_rmap));
        PetscCall(PetscLayoutSetLocalSize(a->h2opus_rmap,n));
        PetscCall(PetscLayoutSetUp(a->h2opus_rmap));
        PetscCall(PetscLayoutReference(a->h2opus_rmap,&a->h2opus_cmap));
      }
      PetscCall(PetscSFCreate(comm,&a->sf));
      PetscCall(PetscSFSetGraphLayout(a->sf,A->rmap,n,NULL,PETSC_OWN_POINTER,idx));
      PetscCall(PetscSFViewFromOptions(a->sf,(PetscObject)A,"-mat_h2opus_sf_view"));
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
    PetscCall(ISRestoreIndices(a->h2opus_indexmap,(const PetscInt **)&idx));
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

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));
  PetscCheck(A->rmap->n == A->cmap->n,PETSC_COMM_SELF,PETSC_ERR_SUP,"Different row and column local sizes are not supported");
  PetscCheck(A->rmap->N == A->cmap->N,comm,PETSC_ERR_SUP,"Rectangular matrices are not supported");

  /* XXX */
  a->leafsize = PetscMin(a->leafsize, PetscMin(A->rmap->N, A->cmap->N));

  PetscCallMPI(MPI_Comm_size(comm,&size));
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

  PetscCall(PetscLogEventBegin(MAT_H2Opus_Build,A,0,0,0));
  if (size > 1) {
#if defined(H2OPUS_USE_MPI)
    a->dist_hmatrix = new DistributedHMatrix(A->rmap->n/* ,A->symmetric */);
#else
    a->dist_hmatrix = NULL;
#endif
  } else {
    a->hmatrix = new HMatrix(A->rmap->n,A->symmetric);
  }
  PetscCall(MatH2OpusInferCoordinates_Private(A));
  PetscCheck(a->ptcloud,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing pointcloud");
  if (a->kernel) {
    BoxEntryGen<PetscScalar, H2OPUS_HWTYPE_CPU, PetscFunctionGenerator<PetscScalar>> entry_gen(*a->kernel);
    if (size > 1) {
      PetscCheck(a->dist_hmatrix,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing distributed CPU matrix");
#if defined(H2OPUS_USE_MPI)
      buildDistributedHMatrix(*a->dist_hmatrix,a->ptcloud,adm,entry_gen,a->leafsize,a->basisord,a->handle);
#endif
    } else {
      buildHMatrix(*a->hmatrix,a->ptcloud,adm,entry_gen,a->leafsize,a->basisord);
    }
    kernel = PETSC_TRUE;
  } else {
    PetscCheck(size <= 1,comm,PETSC_ERR_SUP,"Construction from sampling not supported in parallel");
    buildHMatrixStructure(*a->hmatrix,a->ptcloud,a->leafsize,adm);
  }
  PetscCall(MatSetUpMultiply_H2OPUS(A));

#if defined(PETSC_H2OPUS_USE_GPU)
  boundtocpu = A->boundtocpu;
  if (!boundtocpu) {
    if (size > 1) {
      PetscCheck(a->dist_hmatrix,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing distributed CPU matrix");
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

      PetscCall(PetscOptionsGetBool(((PetscObject)A)->options,((PetscObject)A)->prefix,"-mat_h2opus_hara_verbose",&a->hara_verbose,NULL));
      verbose = a->hara_verbose;
      PetscCall(MatApproximateNorm_Private(a->sampler->GetSamplingMat(),NORM_2,a->norm_max_samples,&Anorm));
      if (a->hara_verbose) PetscCall(PetscPrintf(PETSC_COMM_SELF,"Sampling uses max rank %d, tol %g (%g*%g), %s samples %d\n",a->max_rank,a->rtol*Anorm,a->rtol,Anorm,boundtocpu ? "CPU" : "GPU",a->bs));
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
  PetscCall(PetscLogEventEnd(MAT_H2Opus_Build,A,0,0,0));

  if (!a->s) a->s = 1.0;
  A->assembled = PETSC_TRUE;

  if (samplingdone) {
    PetscBool check = a->check_construction;
    PetscBool checke = PETSC_FALSE;

    PetscCall(PetscOptionsGetBool(((PetscObject)A)->options,((PetscObject)A)->prefix,"-mat_h2opus_check",&check,NULL));
    PetscCall(PetscOptionsGetBool(((PetscObject)A)->options,((PetscObject)A)->prefix,"-mat_h2opus_check_explicit",&checke,NULL));
    if (check) {
      Mat       E,Ae;
      PetscReal n1,ni,n2;
      PetscReal n1A,niA,n2A;
      void      (*normfunc)(void);

      Ae   = a->sampler->GetSamplingMat();
      PetscCall(MatConvert(A,MATSHELL,MAT_INITIAL_MATRIX,&E));
      PetscCall(MatShellSetOperation(E,MATOP_NORM,(void (*)(void))MatNorm_H2OPUS));
      PetscCall(MatAXPY(E,-1.0,Ae,DIFFERENT_NONZERO_PATTERN));
      PetscCall(MatNorm(E,NORM_1,&n1));
      PetscCall(MatNorm(E,NORM_INFINITY,&ni));
      PetscCall(MatNorm(E,NORM_2,&n2));
      if (checke) {
        Mat eA,eE,eAe;

        PetscCall(MatComputeOperator(A,MATAIJ,&eA));
        PetscCall(MatComputeOperator(E,MATAIJ,&eE));
        PetscCall(MatComputeOperator(Ae,MATAIJ,&eAe));
        PetscCall(MatChop(eA,PETSC_SMALL));
        PetscCall(MatChop(eE,PETSC_SMALL));
        PetscCall(MatChop(eAe,PETSC_SMALL));
        PetscCall(PetscObjectSetName((PetscObject)eA,"H2Mat"));
        PetscCall(MatView(eA,NULL));
        PetscCall(PetscObjectSetName((PetscObject)eAe,"S"));
        PetscCall(MatView(eAe,NULL));
        PetscCall(PetscObjectSetName((PetscObject)eE,"H2Mat - S"));
        PetscCall(MatView(eE,NULL));
        PetscCall(MatDestroy(&eA));
        PetscCall(MatDestroy(&eE));
        PetscCall(MatDestroy(&eAe));
      }

      PetscCall(MatGetOperation(Ae,MATOP_NORM,&normfunc));
      PetscCall(MatSetOperation(Ae,MATOP_NORM,(void (*)(void))MatNorm_H2OPUS));
      PetscCall(MatNorm(Ae,NORM_1,&n1A));
      PetscCall(MatNorm(Ae,NORM_INFINITY,&niA));
      PetscCall(MatNorm(Ae,NORM_2,&n2A));
      n1A  = PetscMax(n1A,PETSC_SMALL);
      n2A  = PetscMax(n2A,PETSC_SMALL);
      niA  = PetscMax(niA,PETSC_SMALL);
      PetscCall(MatSetOperation(Ae,MATOP_NORM,normfunc));
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)A),"MATH2OPUS construction errors: NORM_1 %g, NORM_INFINITY %g, NORM_2 %g (%g %g %g)\n",(double)n1,(double)ni,(double)n2,(double)(n1/n1A),(double)(ni/niA),(double)(n2/n2A)));
      PetscCall(MatDestroy(&E));
    }
    a->sampler->SetSamplingMat(NULL);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_H2OPUS(Mat A)
{
  PetscMPIInt    size;
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  PetscCheck(size <= 1,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not yet supported");
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
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)B,&comm));
  PetscCall(MatCreate(comm,&A));
  PetscCall(MatSetSizes(A,B->rmap->n,B->cmap->n,B->rmap->N,B->cmap->N));
  PetscCall(MatSetType(A,MATH2OPUS));
  PetscCall(MatPropagateSymmetryOptions(B,A));
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
    PetscCall(PetscObjectReference((PetscObject)b->sf));
    a->sf = b->sf;
  }
  if (b->h2opus_indexmap) {
    PetscCall(PetscObjectReference((PetscObject)b->h2opus_indexmap));
    a->h2opus_indexmap = b->h2opus_indexmap;
  }

  PetscCall(MatSetUp(A));
  PetscCall(MatSetUpMultiply_H2OPUS(A));
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
  PetscCall(MatBindToCPU(A,iscpu));

  *nA = A;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_H2OPUS(Mat A, PetscViewer view)
{
  Mat_H2OPUS        *h2opus = (Mat_H2OPUS*)A->data;
  PetscBool         isascii, vieweps;
  PetscMPIInt       size;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)view,PETSCVIEWERASCII,&isascii));
  PetscCall(PetscViewerGetFormat(view,&format));
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  if (isascii) {
    if (format == PETSC_VIEWER_ASCII_MATLAB) {
      if (size == 1) {
        FILE *fp;
        PetscCall(PetscViewerASCIIGetPointer(view,&fp));
        dumpHMatrix(*h2opus->hmatrix,6,fp);
      }
    } else {
      PetscCall(PetscViewerASCIIPrintf(view,"  H-Matrix constructed from %s\n",h2opus->kernel ? "Kernel" : "Mat"));
      PetscCall(PetscViewerASCIIPrintf(view,"  PointCloud dim %" PetscInt_FMT "\n",h2opus->ptcloud ? h2opus->ptcloud->getDimension() : 0));
      PetscCall(PetscViewerASCIIPrintf(view,"  Admissibility parameters: leaf size %" PetscInt_FMT ", eta %g\n",h2opus->leafsize,(double)h2opus->eta));
      if (!h2opus->kernel) {
        PetscCall(PetscViewerASCIIPrintf(view,"  Sampling parameters: max_rank %" PetscInt_FMT ", samples %" PetscInt_FMT ", tolerance %g\n",h2opus->max_rank,h2opus->bs,(double)h2opus->rtol));
      } else {
        PetscCall(PetscViewerASCIIPrintf(view,"  Offdiagonal blocks approximation order %" PetscInt_FMT "\n",h2opus->basisord));
      }
      PetscCall(PetscViewerASCIIPrintf(view,"  Number of samples for norms %" PetscInt_FMT "\n",h2opus->norm_max_samples));
      if (size == 1) {
        double dense_mem_cpu = h2opus->hmatrix ? h2opus->hmatrix->getDenseMemoryUsage() : 0;
        double low_rank_cpu = h2opus->hmatrix ? h2opus->hmatrix->getLowRankMemoryUsage() : 0;
#if defined(PETSC_HAVE_CUDA)
        double dense_mem_gpu = h2opus->hmatrix_gpu ? h2opus->hmatrix_gpu->getDenseMemoryUsage() : 0;
        double low_rank_gpu = h2opus->hmatrix_gpu ? h2opus->hmatrix_gpu->getLowRankMemoryUsage() : 0;
#endif
        PetscCall(PetscViewerASCIIPrintf(view,"  Memory consumption GB (CPU): %g (dense) %g (low rank) %g (total)\n", dense_mem_cpu, low_rank_cpu, low_rank_cpu + dense_mem_cpu));
#if defined(PETSC_HAVE_CUDA)
        PetscCall(PetscViewerASCIIPrintf(view,"  Memory consumption GB (GPU): %g (dense) %g (low rank) %g (total)\n", dense_mem_gpu, low_rank_gpu, low_rank_gpu + dense_mem_gpu));
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
        PetscCall(MPIU_Allreduce(MPI_IN_PLACE,matrix_mem,rsize,MPI_DOUBLE_PRECISION,MPI_SUM,PetscObjectComm((PetscObject)A)));
        PetscCall(PetscViewerASCIIPrintf(view,"  Memory consumption GB (CPU): %g (dense) %g (low rank) %g (total)\n", matrix_mem[0], matrix_mem[1], matrix_mem[0] + matrix_mem[1]));
#if defined(PETSC_HAVE_CUDA)
        PetscCall(PetscViewerASCIIPrintf(view,"  Memory consumption GB (GPU): %g (dense) %g (low rank) %g (total)\n", matrix_mem[2], matrix_mem[3], matrix_mem[2] + matrix_mem[3]));
#endif
      }
    }
  }
  vieweps = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(((PetscObject)A)->options,((PetscObject)A)->prefix,"-mat_h2opus_vieweps",&vieweps,NULL));
  if (vieweps) {
    char filename[256];
    const char *name;

    PetscCall(PetscObjectGetName((PetscObject)A,&name));
    PetscCall(PetscSNPrintf(filename,sizeof(filename),"%s_structure.eps",name));
    PetscCall(PetscOptionsGetString(((PetscObject)A)->options,((PetscObject)A)->prefix,"-mat_h2opus_vieweps_filename",filename,sizeof(filename),NULL));
    outputEps(*h2opus->hmatrix,filename);
  }
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

  PetscFunctionBegin;
  PetscCall(PetscLayoutSetUp(A->rmap));
  PetscCall(PetscLayoutSetUp(A->cmap));
  PetscCall(PetscObjectGetComm((PetscObject)A,&comm));
  PetscCall(MatHasCongruentLayouts(A,&cong));
  PetscCheck(cong,comm,PETSC_ERR_SUP,"Only for square matrices with congruent layouts");
  N    = A->rmap->N;
  PetscCallMPI(MPI_Comm_size(comm,&size));
  if (spacedim > 0 && size > 1 && cdist) {
    PetscSF      sf;
    MPI_Datatype dtype;

    PetscCallMPI(MPI_Type_contiguous(spacedim,MPIU_REAL,&dtype));
    PetscCallMPI(MPI_Type_commit(&dtype));

    PetscCall(PetscSFCreate(comm,&sf));
    PetscCall(PetscSFSetGraphWithPattern(sf,A->rmap,PETSCSF_PATTERN_ALLGATHER));
    PetscCall(PetscMalloc1(spacedim*N,&gcoords));
    PetscCall(PetscSFBcastBegin(sf,dtype,coords,gcoords,MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(sf,dtype,coords,gcoords,MPI_REPLACE));
    PetscCall(PetscSFDestroy(&sf));
    PetscCallMPI(MPI_Type_free(&dtype));
  } else gcoords = (PetscReal*)coords;

  delete h2opus->ptcloud;
  delete h2opus->kernel;
  h2opus->ptcloud = new PetscPointCloud<PetscReal>(spacedim,N,gcoords);
  if (kernel) h2opus->kernel = new PetscFunctionGenerator<PetscScalar>(kernel,spacedim,kernelctx);
  if (gcoords != coords) PetscCall(PetscFree(gcoords));
  A->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#if defined(PETSC_H2OPUS_USE_GPU)
static PetscErrorCode MatBindToCPU_H2OPUS(Mat A, PetscBool flg)
{
  PetscMPIInt    size;
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;

  PetscFunctionBegin;
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  if (flg && A->offloadmask == PETSC_OFFLOAD_GPU) {
    if (size > 1) {
      PetscCheck(a->dist_hmatrix_gpu,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
#if defined(H2OPUS_USE_MPI)
      if (!a->dist_hmatrix) a->dist_hmatrix = new DistributedHMatrix(*a->dist_hmatrix_gpu);
      else *a->dist_hmatrix = *a->dist_hmatrix_gpu;
#endif
    } else {
      PetscCheck(a->hmatrix_gpu,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
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
      PetscCheck(a->dist_hmatrix,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
#if defined(H2OPUS_USE_MPI)
      if (!a->dist_hmatrix_gpu) a->dist_hmatrix_gpu = new DistributedHMatrix_GPU(*a->dist_hmatrix);
      else *a->dist_hmatrix_gpu = *a->dist_hmatrix;
#endif
    } else {
      PetscCheck(a->hmatrix,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
      if (!a->hmatrix_gpu) a->hmatrix_gpu = new HMatrix_GPU(*a->hmatrix);
      else *a->hmatrix_gpu = *a->hmatrix;
    }
    delete a->hmatrix;
    delete a->dist_hmatrix;
    a->hmatrix = NULL;
    a->dist_hmatrix = NULL;
    A->offloadmask = PETSC_OFFLOAD_GPU;
  }
  PetscCall(PetscFree(A->defaultvectype));
  if (!flg) {
    PetscCall(PetscStrallocpy(VECCUDA,&A->defaultvectype));
  } else {
    PetscCall(PetscStrallocpy(VECSTANDARD,&A->defaultvectype));
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

.seealso: `MATHTOOL`, `MATDENSE`, `MatCreateH2OpusFromKernel()`, `MatCreateH2OpusFromMat()`
M*/
PETSC_EXTERN PetscErrorCode MatCreate_H2OPUS(Mat A)
{
  Mat_H2OPUS     *a;
  PetscMPIInt    size;

  PetscFunctionBegin;
#if defined(PETSC_H2OPUS_USE_GPU)
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
#endif
  PetscCall(PetscNewLog(A,&a));
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
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  PetscCall(PetscObjectChangeTypeName((PetscObject)A,MATH2OPUS));
  PetscCall(PetscMemzero(A->ops,sizeof(struct _MatOps)));

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

  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_h2opus_seqdense_C",MatProductSetFromOptions_H2OPUS));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_h2opus_seqdensecuda_C",MatProductSetFromOptions_H2OPUS));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_h2opus_mpidense_C",MatProductSetFromOptions_H2OPUS));
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_h2opus_mpidensecuda_C",MatProductSetFromOptions_H2OPUS));
#if defined(PETSC_H2OPUS_USE_GPU)
  PetscCall(PetscFree(A->defaultvectype));
  PetscCall(PetscStrallocpy(VECCUDA,&A->defaultvectype));
#endif
  PetscFunctionReturn(0);
}

/*@C
     MatH2OpusOrthogonalize - Orthogonalize the basis tree of a hierarchical matrix.

   Input Parameter:
.     A - the matrix

   Level: intermediate

.seealso: `MatCreate()`, `MATH2OPUS`, `MatCreateH2OpusFromMat()`, `MatCreateH2OpusFromKernel()`, `MatH2OpusCompress()`
@*/
PetscErrorCode MatH2OpusOrthogonalize(Mat A)
{
  PetscBool      ish2opus;
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;
  PetscMPIInt    size;
  PetscBool      boundtocpu = PETSC_TRUE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATH2OPUS,&ish2opus));
  if (!ish2opus) PetscFunctionReturn(0);
  if (a->orthogonal) PetscFunctionReturn(0);
  HLibProfile::clear();
  PetscCall(PetscLogEventBegin(MAT_H2Opus_Orthog,A,0,0,0));
#if defined(PETSC_H2OPUS_USE_GPU)
  boundtocpu = A->boundtocpu;
#endif
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  if (size > 1) {
    if (boundtocpu) {
      PetscCheck(a->dist_hmatrix,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
#if defined(H2OPUS_USE_MPI)
      distributed_horthog(*a->dist_hmatrix, a->handle);
#endif
#if defined(PETSC_H2OPUS_USE_GPU)
      A->offloadmask = PETSC_OFFLOAD_CPU;
    } else {
      PetscCheck(a->dist_hmatrix_gpu,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
      PetscCall(PetscLogGpuTimeBegin());
#if defined(H2OPUS_USE_MPI)
      distributed_horthog(*a->dist_hmatrix_gpu, a->handle);
#endif
      PetscCall(PetscLogGpuTimeEnd());
#endif
    }
  } else {
#if defined(H2OPUS_USE_MPI)
    h2opusHandle_t handle = a->handle->handle;
#else
    h2opusHandle_t handle = a->handle;
#endif
    if (boundtocpu) {
      PetscCheck(a->hmatrix,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
      horthog(*a->hmatrix, handle);
#if defined(PETSC_H2OPUS_USE_GPU)
      A->offloadmask = PETSC_OFFLOAD_CPU;
    } else {
      PetscCheck(a->hmatrix_gpu,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
      PetscCall(PetscLogGpuTimeBegin());
      horthog(*a->hmatrix_gpu, handle);
      PetscCall(PetscLogGpuTimeEnd());
#endif
    }
  }
  a->orthogonal = PETSC_TRUE;
  { /* log flops */
    double gops,time,perf,dev;
    HLibProfile::getHorthogPerf(gops,time,perf,dev);
#if defined(PETSC_H2OPUS_USE_GPU)
    if (boundtocpu) {
      PetscCall(PetscLogFlops(1e9*gops));
    } else {
      PetscCall(PetscLogGpuFlops(1e9*gops));
    }
#else
    PetscCall(PetscLogFlops(1e9*gops));
#endif
  }
  PetscCall(PetscLogEventEnd(MAT_H2Opus_Orthog,A,0,0,0));
  PetscFunctionReturn(0);
}

/*@C
     MatH2OpusCompress - Compress a hierarchical matrix.

   Input Parameters:
+     A - the matrix
-     tol - the absolute truncation threshold

   Level: intermediate

.seealso: `MatCreate()`, `MATH2OPUS`, `MatCreateH2OpusFromMat()`, `MatCreateH2OpusFromKernel()`, `MatH2OpusOrthogonalize()`
@*/
PetscErrorCode MatH2OpusCompress(Mat A, PetscReal tol)
{
  PetscBool      ish2opus;
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;
  PetscMPIInt    size;
  PetscBool      boundtocpu = PETSC_TRUE;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidLogicalCollectiveReal(A,tol,2);
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATH2OPUS,&ish2opus));
  if (!ish2opus || tol <= 0.0) PetscFunctionReturn(0);
  PetscCall(MatH2OpusOrthogonalize(A));
  HLibProfile::clear();
  PetscCall(PetscLogEventBegin(MAT_H2Opus_Compress,A,0,0,0));
#if defined(PETSC_H2OPUS_USE_GPU)
  boundtocpu = A->boundtocpu;
#endif
  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
  if (size > 1) {
    if (boundtocpu) {
      PetscCheck(a->dist_hmatrix,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
#if defined(H2OPUS_USE_MPI)
      distributed_hcompress(*a->dist_hmatrix, tol, a->handle);
#endif
#if defined(PETSC_H2OPUS_USE_GPU)
      A->offloadmask = PETSC_OFFLOAD_CPU;
    } else {
      PetscCheck(a->dist_hmatrix_gpu,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
      PetscCall(PetscLogGpuTimeBegin());
#if defined(H2OPUS_USE_MPI)
      distributed_hcompress(*a->dist_hmatrix_gpu, tol, a->handle);
#endif
      PetscCall(PetscLogGpuTimeEnd());
#endif
    }
  } else {
#if defined(H2OPUS_USE_MPI)
    h2opusHandle_t handle = a->handle->handle;
#else
    h2opusHandle_t handle = a->handle;
#endif
    if (boundtocpu) {
      PetscCheck(a->hmatrix,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
      hcompress(*a->hmatrix, tol, handle);
#if defined(PETSC_H2OPUS_USE_GPU)
      A->offloadmask = PETSC_OFFLOAD_CPU;
    } else {
      PetscCheck(a->hmatrix_gpu,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
      PetscCall(PetscLogGpuTimeBegin());
      hcompress(*a->hmatrix_gpu, tol, handle);
      PetscCall(PetscLogGpuTimeEnd());
#endif
    }
  }
  { /* log flops */
    double gops,time,perf,dev;
    HLibProfile::getHcompressPerf(gops,time,perf,dev);
#if defined(PETSC_H2OPUS_USE_GPU)
    if (boundtocpu) {
      PetscCall(PetscLogFlops(1e9*gops));
    } else {
      PetscCall(PetscLogGpuFlops(1e9*gops));
    }
#else
    PetscCall(PetscLogFlops(1e9*gops));
#endif
  }
  PetscCall(PetscLogEventEnd(MAT_H2Opus_Compress,A,0,0,0));
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

.seealso: `MatCreate()`, `MATH2OPUS`, `MatCreateH2OpusFromMat()`, `MatCreateH2OpusFromKernel()`, `MatH2OpusCompress()`, `MatH2OpusOrthogonalize()`
@*/
PetscErrorCode MatH2OpusSetSamplingMat(Mat A, Mat B, PetscInt bs, PetscReal tol)
{
  PetscBool      ish2opus;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  if (B) PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  PetscValidLogicalCollectiveInt(A,bs,3);
  PetscValidLogicalCollectiveReal(A,tol,3);
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATH2OPUS,&ish2opus));
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
+     -mat_h2opus_leafsize <PetscInt> - Leaf size of cluster tree
.     -mat_h2opus_eta <PetscReal> - Admissibility condition tolerance
.     -mat_h2opus_order <PetscInt> - Chebychev approximation order
-     -mat_h2opus_normsamples <PetscInt> - Maximum number of samples to be used when estimating norms

   Level: intermediate

.seealso: `MatCreate()`, `MATH2OPUS`, `MatCreateH2OpusFromMat()`
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

  PetscFunctionBegin;
  PetscCheck(m == n,PETSC_COMM_SELF,PETSC_ERR_SUP,"Different row and column local sizes are not supported");
  PetscCall(MatCreate(comm,&A));
  PetscCall(MatSetSizes(A,m,n,M,N));
  PetscCheck(M == N,comm,PETSC_ERR_SUP,"Rectangular matrices are not supported");
  PetscCall(MatSetType(A,MATH2OPUS));
  PetscCall(MatBindToCPU(A,iscpu));
  PetscCall(MatH2OpusSetCoords_H2OPUS(A,spacedim,coords,cdist,kernel,kernelctx));

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
+     -mat_h2opus_leafsize <PetscInt> - Leaf size of cluster tree
.     -mat_h2opus_eta <PetscReal> - Admissibility condition tolerance
.     -mat_h2opus_maxrank <PetscInt> - Maximum rank when constructed from matvecs
.     -mat_h2opus_samples <PetscInt> - Maximum number of samples to be taken concurrently when constructing from matvecs
.     -mat_h2opus_rtol <PetscReal> - Relative tolerance for construction from sampling
.     -mat_h2opus_check <PetscBool> - Check error when constructing from sampling during MatAssemblyEnd()
.     -mat_h2opus_hara_verbose <PetscBool> - Verbose output from hara construction
-     -mat_h2opus_normsamples <PetscInt> - Maximum bumber of samples to be when estimating norms

   Notes: not available in parallel

   Level: intermediate

.seealso: `MatCreate()`, `MATH2OPUS`, `MatCreateH2OpusFromKernel()`
@*/
PetscErrorCode MatCreateH2OpusFromMat(Mat B, PetscInt spacedim, const PetscReal coords[], PetscBool cdist, PetscReal eta, PetscInt leafsize, PetscInt maxrank, PetscInt bs, PetscReal rtol, Mat *nA)
{
  Mat            A;
  Mat_H2OPUS     *h2opus;
  MPI_Comm       comm;
  PetscBool      boundtocpu = PETSC_TRUE;

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
  PetscCall(PetscObjectGetComm((PetscObject)B,&comm));
  PetscCheck(B->rmap->n == B->cmap->n,PETSC_COMM_SELF,PETSC_ERR_SUP,"Different row and column local sizes are not supported");
  PetscCheck(B->rmap->N == B->cmap->N,comm,PETSC_ERR_SUP,"Rectangular matrices are not supported");
  PetscCall(MatCreate(comm,&A));
  PetscCall(MatSetSizes(A,B->rmap->n,B->cmap->n,B->rmap->N,B->cmap->N));
#if defined(PETSC_H2OPUS_USE_GPU)
  {
    PetscBool iscuda;
    VecType   vtype;

    PetscCall(MatGetVecType(B,&vtype));
    PetscCall(PetscStrcmp(vtype,VECCUDA,&iscuda));
    if (!iscuda) {
      PetscCall(PetscStrcmp(vtype,VECSEQCUDA,&iscuda));
      if (!iscuda) {
        PetscCall(PetscStrcmp(vtype,VECMPICUDA,&iscuda));
      }
    }
    if (iscuda && !B->boundtocpu) boundtocpu = PETSC_FALSE;
  }
#endif
  PetscCall(MatSetType(A,MATH2OPUS));
  PetscCall(MatBindToCPU(A,boundtocpu));
  if (spacedim) {
    PetscCall(MatH2OpusSetCoords_H2OPUS(A,spacedim,coords,cdist,NULL,NULL));
  }
  PetscCall(MatPropagateSymmetryOptions(B,A));
  /* PetscCheck(A->symmetric,comm,PETSC_ERR_SUP,"Unsymmetric sampling does not work"); */

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

.seealso: `MatCreate()`, `MATH2OPUS`, `MatCreateH2OpusFromMat()`, `MatCreateH2OpusFromKernel()`
@*/
PetscErrorCode MatH2OpusGetIndexMap(Mat A, IS *indexmap)
{
  PetscBool      ish2opus;
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidPointer(indexmap,2);
  PetscCheck(A->assembled,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATH2OPUS,&ish2opus));
  PetscCheck(ish2opus,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not for type %s",((PetscObject)A)->type_name);
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

.seealso: `MatCreate()`, `MATH2OPUS`, `MatCreateH2OpusFromMat()`, `MatCreateH2OpusFromKernel()`
@*/
PetscErrorCode MatH2OpusMapVec(Mat A, PetscBool nativetopetsc, Vec in, Vec* out)
{
  PetscBool      ish2opus;
  Mat_H2OPUS     *a = (Mat_H2OPUS*)A->data;
  PetscScalar    *xin,*xout;
  PetscBool      nm;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidLogicalCollectiveBool(A,nativetopetsc,2);
  PetscValidHeaderSpecific(in,VEC_CLASSID,3);
  PetscValidPointer(out,4);
  PetscCheck(A->assembled,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATH2OPUS,&ish2opus));
  PetscCheck(ish2opus,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not for type %s",((PetscObject)A)->type_name);
  nm   = a->nativemult;
  PetscCall(MatH2OpusSetNativeMult(A,(PetscBool)!nativetopetsc));
  PetscCall(MatCreateVecs(A,out,NULL));
  PetscCall(MatH2OpusSetNativeMult(A,nm));
  if (!a->sf) { /* same ordering */
    PetscCall(VecCopy(in,*out));
    PetscFunctionReturn(0);
  }
  PetscCall(VecGetArrayRead(in,(const PetscScalar**)&xin));
  PetscCall(VecGetArrayWrite(*out,&xout));
  if (nativetopetsc) {
    PetscCall(PetscSFReduceBegin(a->sf,MPIU_SCALAR,xin,xout,MPI_REPLACE));
    PetscCall(PetscSFReduceEnd(a->sf,MPIU_SCALAR,xin,xout,MPI_REPLACE));
  } else {
    PetscCall(PetscSFBcastBegin(a->sf,MPIU_SCALAR,xin,xout,MPI_REPLACE));
    PetscCall(PetscSFBcastEnd(a->sf,MPIU_SCALAR,xin,xout,MPI_REPLACE));
  }
  PetscCall(VecRestoreArrayRead(in,(const PetscScalar**)&xin));
  PetscCall(VecRestoreArrayWrite(*out,&xout));
  PetscFunctionReturn(0);
}

/*@C
     MatH2OpusLowRankUpdate - Perform a low-rank update of the form A = A + s * U * V^T

   Input Parameters:
+     A - the hierarchical matrix
.     s - the scaling factor
.     U - the dense low-rank update matrix
-     V - (optional) the dense low-rank update matrix (if NULL, then V = U is assumed)

   Notes: The U and V matrices must be in dense format

   Level: intermediate

.seealso: `MatCreate()`, `MATH2OPUS`, `MatCreateH2OpusFromMat()`, `MatCreateH2OpusFromKernel()`, `MatH2OpusCompress()`, `MatH2OpusOrthogonalize()`, `MATDENSE`
@*/
PetscErrorCode MatH2OpusLowRankUpdate(Mat A, Mat U, Mat V, PetscScalar s)
{
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscCheck(A->assembled,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Not for unassembled matrix");
  PetscValidHeaderSpecific(U,MAT_CLASSID,2);
  PetscCheckSameComm(A,1,U,2);
  if (V) {
    PetscValidHeaderSpecific(V,MAT_CLASSID,3);
    PetscCheckSameComm(A,1,V,3);
  }
  PetscValidLogicalCollectiveScalar(A,s,4);

  if (!V) V = U;
  PetscCheck(U->cmap->N == V->cmap->N,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"Non matching rank update %" PetscInt_FMT " != %" PetscInt_FMT,U->cmap->N,V->cmap->N);
  if (!U->cmap->N) PetscFunctionReturn(0);
  PetscCall(PetscLayoutCompare(U->rmap,A->rmap,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"A and U must have the same row layout");
  PetscCall(PetscLayoutCompare(V->rmap,A->cmap,&flg));
  PetscCheck(flg,PetscObjectComm((PetscObject)A),PETSC_ERR_ARG_WRONGSTATE,"A column layout must match V row column layout");
  PetscCall(PetscObjectTypeCompare((PetscObject)A,MATH2OPUS,&flg));
  if (flg) {
    Mat_H2OPUS        *a = (Mat_H2OPUS*)A->data;
    const PetscScalar *u,*v,*uu,*vv;
    PetscInt          ldu,ldv;
    PetscMPIInt       size;
#if defined(H2OPUS_USE_MPI)
    h2opusHandle_t    handle = a->handle->handle;
#else
    h2opusHandle_t    handle = a->handle;
#endif
    PetscBool         usesf = (PetscBool)(a->sf && !a->nativemult);
    PetscSF           usf,vsf;

    PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)A),&size));
    PetscCheck(size <= 1,PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not yet implemented in parallel");
    PetscCall(PetscLogEventBegin(MAT_H2Opus_LR,A,0,0,0));
    PetscCall(PetscObjectBaseTypeCompareAny((PetscObject)U,&flg,MATSEQDENSE,MATMPIDENSE,""));
    PetscCheck(flg,PetscObjectComm((PetscObject)U),PETSC_ERR_SUP,"Not for U of type %s",((PetscObject)U)->type_name);
    PetscCall(PetscObjectBaseTypeCompareAny((PetscObject)V,&flg,MATSEQDENSE,MATMPIDENSE,""));
    PetscCheck(flg,PetscObjectComm((PetscObject)V),PETSC_ERR_SUP,"Not for V of type %s",((PetscObject)V)->type_name);
    PetscCall(MatDenseGetLDA(U,&ldu));
    PetscCall(MatDenseGetLDA(V,&ldv));
    PetscCall(MatBoundToCPU(A,&flg));
    if (usesf) {
      PetscInt n;

      PetscCall(MatDenseGetH2OpusVectorSF(U,a->sf,&usf));
      PetscCall(MatDenseGetH2OpusVectorSF(V,a->sf,&vsf));
      PetscCall(MatH2OpusResizeBuffers_Private(A,U->cmap->N,V->cmap->N));
      PetscCall(PetscSFGetGraph(a->sf,NULL,&n,NULL,NULL));
      ldu = n;
      ldv = n;
    }
    if (flg) {
      PetscCheck(a->hmatrix,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing CPU matrix");
      PetscCall(MatDenseGetArrayRead(U,&u));
      PetscCall(MatDenseGetArrayRead(V,&v));
      if (usesf) {
        vv = MatH2OpusGetThrustPointer(*a->yy);
        PetscCall(PetscSFBcastBegin(vsf,MPIU_SCALAR,v,(PetscScalar*)vv,MPI_REPLACE));
        PetscCall(PetscSFBcastEnd(vsf,MPIU_SCALAR,v,(PetscScalar*)vv,MPI_REPLACE));
        if (U != V) {
          uu = MatH2OpusGetThrustPointer(*a->xx);
          PetscCall(PetscSFBcastBegin(usf,MPIU_SCALAR,u,(PetscScalar*)uu,MPI_REPLACE));
          PetscCall(PetscSFBcastEnd(usf,MPIU_SCALAR,u,(PetscScalar*)uu,MPI_REPLACE));
        } else uu = vv;
      } else { uu = u; vv = v; }
      hlru_global(*a->hmatrix,uu,ldu,vv,ldv,U->cmap->N,s,handle);
      PetscCall(MatDenseRestoreArrayRead(U,&u));
      PetscCall(MatDenseRestoreArrayRead(V,&v));
    } else {
#if defined(PETSC_H2OPUS_USE_GPU)
      PetscBool flgU, flgV;

      PetscCheck(a->hmatrix_gpu,PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"Missing GPU matrix");
      PetscCall(PetscObjectTypeCompareAny((PetscObject)U,&flgU,MATSEQDENSE,MATMPIDENSE,""));
      if (flgU) PetscCall(MatConvert(U,MATDENSECUDA,MAT_INPLACE_MATRIX,&U));
      PetscCall(PetscObjectTypeCompareAny((PetscObject)V,&flgV,MATSEQDENSE,MATMPIDENSE,""));
      if (flgV) PetscCall(MatConvert(V,MATDENSECUDA,MAT_INPLACE_MATRIX,&V));
      PetscCall(MatDenseCUDAGetArrayRead(U,&u));
      PetscCall(MatDenseCUDAGetArrayRead(V,&v));
      if (usesf) {
        vv = MatH2OpusGetThrustPointer(*a->yy_gpu);
        PetscCall(PetscSFBcastBegin(vsf,MPIU_SCALAR,v,(PetscScalar*)vv,MPI_REPLACE));
        PetscCall(PetscSFBcastEnd(vsf,MPIU_SCALAR,v,(PetscScalar*)vv,MPI_REPLACE));
        if (U != V) {
          uu = MatH2OpusGetThrustPointer(*a->xx_gpu);
          PetscCall(PetscSFBcastBegin(usf,MPIU_SCALAR,u,(PetscScalar*)uu,MPI_REPLACE));
          PetscCall(PetscSFBcastEnd(usf,MPIU_SCALAR,u,(PetscScalar*)uu,MPI_REPLACE));
        } else uu = vv;
      } else { uu = u; vv = v; }
#else
      SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_PLIB,"This should not happen");
#endif
      hlru_global(*a->hmatrix_gpu,uu,ldu,vv,ldv,U->cmap->N,s,handle);
#if defined(PETSC_H2OPUS_USE_GPU)
      PetscCall(MatDenseCUDARestoreArrayRead(U,&u));
      PetscCall(MatDenseCUDARestoreArrayRead(V,&v));
      if (flgU) PetscCall(MatConvert(U,MATDENSE,MAT_INPLACE_MATRIX,&U));
      if (flgV) PetscCall(MatConvert(V,MATDENSE,MAT_INPLACE_MATRIX,&V));
#endif
    }
    PetscCall(PetscLogEventEnd(MAT_H2Opus_LR,A,0,0,0));
    a->orthogonal = PETSC_FALSE;
  }
  PetscFunctionReturn(0);
}
#endif
