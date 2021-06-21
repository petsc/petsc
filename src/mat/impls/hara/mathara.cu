#include <hara.h>
#include <distributed/distributed_hara_handle.h>
#include <distributed/distributed_hmatrix.h>
#include <distributed/distributed_geometric_construction.h>
#include <distributed/distributed_hgemv.h>
#include <petsc/private/matimpl.h>
#include <petscsf.h>

#define MatHaraGetThrustPointer(v) thrust::raw_pointer_cast((v).data())

// TODO HARA:
// MatDuplicate buggy with commsize > 1
// kernel needs (global?) id of points (issues with Chebyshev points and coupling matrix computation)
// unsymmetrix DistributedHMatrix (transpose for distributed_hgemv?)
// Unify interface for sequential and parallel?
// Reuse geometric construction (almost possible, only the unsymmetric case is explicitly handled)
// Fix includes:
// - everything under hara/ dir (except for hara.h)
// - fix kblas includes
// - namespaces?
// Fix naming convention DistributedHMatrix_GPU vs GPU_HMatrix
// Diagnostics? FLOPS, MEMORY USAGE IN PARALLEL
// Why do we need to template the admissibility condition in the hmatrix construction?
//
template<typename T, int D>
struct PetscPointCloud
{
  static const int Dim = D;
  typedef T ElementType;

  struct Point
  {
    T x[D];
    Point() {
      for (int i = 0; i < D; i++) x[i] = 0;
    }
    Point(const T xx[]) {
      for (int i = 0; i < D; i++) x[i] = xx[i];
    }
  };

  std::vector<Point> pts;

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const { return pts.size(); }

  // Returns the dim'th component of the idx'th point in the class:
  inline T kdtree_get_pt(const size_t idx, int dim) const
  {
    return pts[idx].x[dim];
  }

  inline T get_pt(const size_t idx, int dim) const
  {
    return kdtree_get_pt(idx, dim);
  }

  inline bool kdtree_ignore_point(const size_t idx) const { return false; }

  // Optional bounding-box computation: return false to default to a standard bbox computation loop.
  //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
  //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

  PetscPointCloud(PetscInt,const T[]);
  PetscPointCloud(PetscPointCloud&);
};

template<typename T, int D>
PetscPointCloud<T,D>::PetscPointCloud(PetscInt n, const T coords[])
{
  this->pts.resize(n);
  for (PetscInt i = 0; i < n; i++) {
    Point p(coords);
    this->pts[i] = p;
    coords += D;
  }
}

template<typename T, int D>
PetscPointCloud<T,D>::PetscPointCloud(PetscPointCloud<T,D>& other)
{
  this->pts = other.pts;
}

template <typename T>
using PetscPointCloud3D = PetscPointCloud<T,3>;
template <typename T>
using PetscPointCloud2D = PetscPointCloud<T,2>;
template <typename T>
using PetscPointCloud1D = PetscPointCloud<T,1>;

template<typename T, int Dim>
struct PetscFunctionGenerator
{
private:
  MatHaraKernel k;
  void          *ctx;

public:
    PetscFunctionGenerator(MatHaraKernel k, void* ctx) { this->k = k; this->ctx = ctx; }
    PetscFunctionGenerator(PetscFunctionGenerator& other) { this->k = other.k; this->ctx = other.ctx; }
    T operator()(PetscReal pt1[Dim], PetscReal pt2[Dim])
    {
        return (T)(this->k ? (*this->k)(Dim,pt1,pt2,this->ctx) : 0);
    }
};

template <typename T>
using PetscFunctionGenerator3D = PetscFunctionGenerator<T,3>;
template <typename T>
using PetscFunctionGenerator2D = PetscFunctionGenerator<T,2>;
template <typename T>
using PetscFunctionGenerator1D = PetscFunctionGenerator<T,1>;

#include <../src/mat/impls/hara/matharasampler.hpp>

typedef struct {
  distributedHaraHandle_t handle;

  /* two different classes at the moment */
  HMatrix *hmatrix;
  DistributedHMatrix *dist_hmatrix;

  /* May use permutations */
  PetscSF sf;
  PetscLayout hara_rmap;
  thrust::host_vector<PetscScalar> *xx,*yy;
  PetscInt xxs,yys;
  PetscBool multsetup;

  /* GPU */
#if defined(HARA_USE_GPU)
  GPU_HMatrix *hmatrix_gpu;
  DistributedHMatrix_GPU *dist_hmatrix_gpu;
  thrust::device_vector<PetscScalar> *xx_gpu,*yy_gpu;
  PetscInt xxs_gpu,yys_gpu;
#endif

  /* construction from matvecs */
  PetscMatrixSampler* sampler;

  /* Admissibility */
  PetscReal eta;
  PetscInt  leafsize;

  /* for dof reordering */
  PetscInt spacedim;
  PetscPointCloud1D<PetscReal> *ptcloud1;
  PetscPointCloud2D<PetscReal> *ptcloud2;
  PetscPointCloud3D<PetscReal> *ptcloud3;

  /* kernel for generating matrix entries */
  PetscFunctionGenerator1D<PetscScalar> *kernel1;
  PetscFunctionGenerator2D<PetscScalar> *kernel2;
  PetscFunctionGenerator3D<PetscScalar> *kernel3;

  /* customization */
  PetscInt  basisord;
  PetscInt  max_rank;
  PetscInt  bs;
  PetscReal rtol;
  PetscInt  norm_max_samples;
  PetscBool check_construction;

  /* keeps track of MatScale values */
  PetscScalar s;
} Mat_HARA;

static PetscErrorCode MatDestroy_HARA(Mat A)
{
  Mat_HARA       *a = (Mat_HARA*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  haraDestroyDistributedHandle(a->handle);
  delete a->hmatrix;
  delete a->dist_hmatrix;
  ierr = PetscSFDestroy(&a->sf);CHKERRQ(ierr);
  ierr = PetscLayoutDestroy(&a->hara_rmap);CHKERRQ(ierr);
  delete a->xx;
  delete a->yy;
#if defined(HARA_USE_GPU)
  delete a->hmatrix_gpu;
  delete a->dist_hmatrix_gpu;
  delete a->xx_gpu;
  delete a->yy_gpu;
#endif
  delete a->sampler;
  delete a->ptcloud1;
  delete a->ptcloud2;
  delete a->ptcloud3;
  delete a->kernel1;
  delete a->kernel2;
  delete a->kernel3;
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_hara_seqdense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_hara_seqdensecuda_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_hara_mpidense_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_hara_mpidensecuda_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,NULL);CHKERRQ(ierr);
  ierr = PetscFree(A->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSFGetVectorSF(PetscSF sf, PetscInt nv, PetscInt ldr, PetscInt ldl, PetscSF *vsf)
{
  PetscSF           rankssf;
  const PetscSFNode *iremote;
  PetscSFNode       *viremote,*rremotes;
  const PetscInt    *ilocal;
  PetscInt          *vilocal = NULL,*ldrs;
  const PetscMPIInt *ranks;
  PetscMPIInt       *sranks;
  PetscInt          nranks,nr,nl,vnr,vnl,i,v,j,maxl;
  MPI_Comm          comm;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (nv == 1) {
    ierr = PetscObjectReference((PetscObject)sf);CHKERRQ(ierr);
    *vsf = sf;
    PetscFunctionReturn(0);
  }
  ierr = PetscObjectGetComm((PetscObject)sf,&comm);CHKERRQ(ierr);
  ierr = PetscSFGetGraph(sf,&nr,&nl,&ilocal,&iremote);CHKERRQ(ierr);
  ierr = PetscSFGetLeafRange(sf,NULL,&maxl);CHKERRQ(ierr);
  maxl += 1;
  if (ldl == PETSC_DECIDE) ldl = maxl;
  if (ldr == PETSC_DECIDE) ldr = nr;
  if (ldr < nr) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid leading dimension %D < %D",ldr,nr);
  if (ldl < maxl) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid leading dimension %D < %D",ldl,maxl);
  vnr  = nr*nv;
  vnl  = nl*nv;
  ierr = PetscMalloc1(vnl,&viremote);CHKERRQ(ierr);
  if (ilocal) {
    ierr = PetscMalloc1(vnl,&vilocal);CHKERRQ(ierr);
  }

  /* TODO: Should this special SF be available, e.g.
     PetscSFGetRanksSF or similar? */
  ierr = PetscSFGetRootRanks(sf,&nranks,&ranks,NULL,NULL,NULL);CHKERRQ(ierr);
  ierr = PetscMalloc1(nranks,&sranks);CHKERRQ(ierr);
  ierr = PetscArraycpy(sranks,ranks,nranks);CHKERRQ(ierr);
  ierr = PetscSortMPIInt(nranks,sranks);CHKERRQ(ierr);
  ierr = PetscMalloc1(nranks,&rremotes);CHKERRQ(ierr);
  for (i=0;i<nranks;i++) {
    rremotes[i].rank  = sranks[i];
    rremotes[i].index = 0;
  }
  ierr = PetscSFDuplicate(sf,PETSCSF_DUPLICATE_CONFONLY,&rankssf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(rankssf,1,nranks,NULL,PETSC_OWN_POINTER,rremotes,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscMalloc1(nranks,&ldrs);CHKERRQ(ierr);
  ierr = PetscSFBcastBegin(rankssf,MPIU_INT,&ldr,ldrs,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFBcastEnd(rankssf,MPIU_INT,&ldr,ldrs,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&rankssf);CHKERRQ(ierr);

  j = -1;
  for (i=0;i<nl;i++) {
    const PetscInt r  = iremote[i].rank;
    const PetscInt ii = iremote[i].index;

    if (j < 0 || sranks[j] != r) {
      ierr = PetscFindMPIInt(r,nranks,sranks,&j);CHKERRQ(ierr);
    }
    if (j < 0) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Unable to locate neighbor rank %D",r);
    for (v=0;v<nv;v++) {
      viremote[v*nl + i].rank  = r;
      viremote[v*nl + i].index = v*ldrs[j] + ii;
      if (ilocal) vilocal[v*nl + i] = v*ldl + ilocal[i];
    }
  }
  ierr = PetscFree(sranks);CHKERRQ(ierr);
  ierr = PetscFree(ldrs);CHKERRQ(ierr);
  ierr = PetscSFCreate(comm,vsf);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(*vsf,vnr,vnl,vilocal,PETSC_OWN_POINTER,viremote,PETSC_OWN_POINTER);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode VecSign(Vec v, Vec s)
{
  const PetscScalar *av;
  PetscScalar       *as;
  PetscInt          i,n;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,VEC_CLASSID,1);
  PetscValidHeaderSpecific(s,VEC_CLASSID,2);
  ierr = VecGetArrayRead(v,&av);CHKERRQ(ierr);
  ierr = VecGetArrayWrite(s,&as);CHKERRQ(ierr);
  ierr = VecGetLocalSize(s,&n);CHKERRQ(ierr);
  ierr = VecGetLocalSize(v,&i);CHKERRQ(ierr);
  if (i != n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_SUP,"Invalid local sizes %D != %D",i,n);
  for (i=0;i<n;i++) as[i] = PetscAbsScalar(av[i]) < 0 ? -1. : 1.;
  ierr = VecRestoreArrayWrite(s,&as);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(v,&av);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* these are approximate norms */
/* NORM_2: Estimating the matrix p-norm Nicholas J. Higham
   NORM_1/NORM_INFINITY: A block algorithm for matrix 1-norm estimation, with an application to 1-norm pseudospectra Higham, Nicholas J. and Tisseur, Francoise */
static PetscErrorCode MatApproximateNorm_Private(Mat A, NormType normtype, PetscInt normsamples, PetscReal* n)
{
  Vec            x,y,w,z;
  PetscReal      normz,adot;
  PetscScalar    dot;
  PetscInt       i,j,N,jold = -1;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  switch (normtype) {
  case NORM_INFINITY:
  case NORM_1:
    if (normsamples < 0) normsamples = 10; /* pure guess */
    if (normtype == NORM_INFINITY) {
      Mat B;
      ierr = MatCreateTranspose(A,&B);CHKERRQ(ierr);
      A = B;
    } else {
      ierr = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
    }
    ierr = MatCreateVecs(A,&x,&y);CHKERRQ(ierr);
    ierr = MatCreateVecs(A,&z,&w);CHKERRQ(ierr);
    ierr = VecGetSize(x,&N);CHKERRQ(ierr);
    ierr = VecSet(x,1./N);CHKERRQ(ierr);
    ierr = VecSetOption(x,VEC_IGNORE_OFF_PROC_ENTRIES,PETSC_TRUE);CHKERRQ(ierr);
    *n   = 0.0;
    for (i = 0; i < normsamples; i++) {
      ierr = MatMult(A,x,y);CHKERRQ(ierr);
      ierr = VecSign(y,w);CHKERRQ(ierr);
      ierr = MatMultTranspose(A,w,z);CHKERRQ(ierr);
      ierr = VecNorm(z,NORM_INFINITY,&normz);CHKERRQ(ierr);
      ierr = VecDot(x,z,&dot);CHKERRQ(ierr);
      adot = PetscAbsScalar(dot);
      ierr = PetscInfo4(A,"%s norm it %D -> (%g %g)\n",NormTypes[normtype],i,(double)normz,(double)adot);CHKERRQ(ierr);
      if (normz <= adot && i > 0) {
        ierr = VecNorm(y,NORM_1,n);CHKERRQ(ierr);
        break;
      }
      ierr = VecSet(x,0.);CHKERRQ(ierr);
      ierr = VecMax(z,&j,&normz);CHKERRQ(ierr);
      if (j == jold) {
        ierr = VecNorm(y,NORM_1,n);CHKERRQ(ierr);
        ierr = PetscInfo2(A,"%s norm it %D -> breakdown (j==jold)\n",NormTypes[normtype],i);CHKERRQ(ierr);
        break;
      }
      jold = j;
      ierr = VecSetValue(x,j,1.0,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
      ierr = VecAssemblyEnd(x);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&w);CHKERRQ(ierr);
    ierr = VecDestroy(&y);CHKERRQ(ierr);
    ierr = VecDestroy(&z);CHKERRQ(ierr);
    break;
  case NORM_2:
    if (normsamples < 0) normsamples = 20; /* pure guess */
    ierr = MatCreateVecs(A,&x,&y);CHKERRQ(ierr);
    ierr = MatCreateVecs(A,&z,NULL);CHKERRQ(ierr);
    ierr = VecSetRandom(x,NULL);CHKERRQ(ierr);
    ierr = VecNormalize(x,NULL);CHKERRQ(ierr);
    *n   = 0.0;
    for (i = 0; i < normsamples; i++) {
      ierr = MatMult(A,x,y);CHKERRQ(ierr);
      ierr = VecNormalize(y,n);CHKERRQ(ierr);
      ierr = MatMultTranspose(A,y,z);CHKERRQ(ierr);
      ierr = VecNorm(z,NORM_2,&normz);CHKERRQ(ierr);
      ierr = VecDot(x,z,&dot);CHKERRQ(ierr);
      adot = PetscAbsScalar(dot);
      ierr = PetscInfo5(A,"%s norm it %D -> %g (%g %g)\n",NormTypes[normtype],i,(double)*n,(double)normz,(double)adot);CHKERRQ(ierr);
      if (normz <= adot) break;
      if (i < normsamples - 1) {
        Vec t;

        ierr = VecNormalize(z,NULL);CHKERRQ(ierr);
        t = x;
        x = z;
        z = t;
      }
    }
    ierr = VecDestroy(&x);CHKERRQ(ierr);
    ierr = VecDestroy(&y);CHKERRQ(ierr);
    ierr = VecDestroy(&z);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"%s norm not supported",NormTypes[normtype]);
  }
  ierr = PetscInfo3(A,"%s norm %g computed in %D iterations\n",NormTypes[normtype],(double)*n,i);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatNorm_HARA(Mat A, NormType normtype, PetscReal* n)
{
  PetscErrorCode ierr;
  PetscBool      ishara;
  PetscInt       nmax = PETSC_DECIDE;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)A,MATHARA,&ishara);CHKERRQ(ierr);
  if (ishara) {
    Mat_HARA *a = (Mat_HARA*)A->data;

    nmax = a->norm_max_samples;
  }
  ierr = MatApproximateNorm_Private(A,normtype,nmax,n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultNKernel_HARA(Mat A, PetscBool transA, Mat B, Mat C)
{
  Mat_HARA       *hara = (Mat_HARA*)A->data;
  haraHandle_t   handle = hara->handle->handle;
  PetscBool      boundtocpu = PETSC_TRUE;
  PetscScalar    *xx,*yy,*uxx,*uyy;
  PetscInt       blda,clda;
  PetscSF        bsf,csf;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(HARA_USE_GPU)
  boundtocpu = A->boundtocpu;
#endif
  ierr = MatDenseGetLDA(B,&blda);CHKERRQ(ierr);
  ierr = MatDenseGetLDA(C,&clda);CHKERRQ(ierr);
  if (hara->sf) {
    PetscInt n;

    ierr = PetscSFGetGraph(hara->sf,NULL,&n,NULL,NULL);CHKERRQ(ierr);
    ierr = PetscObjectQuery((PetscObject)B,"_mathara_vectorsf",(PetscObject*)&bsf);CHKERRQ(ierr);
    if (!bsf) {
      ierr = PetscSFGetVectorSF(hara->sf,B->cmap->N,blda,PETSC_DECIDE,&bsf);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)B,"_mathara_vectorsf",(PetscObject)bsf);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)bsf);CHKERRQ(ierr);
    }
    ierr = PetscObjectQuery((PetscObject)C,"_mathara_vectorsf",(PetscObject*)&csf);CHKERRQ(ierr);
    if (!csf) {
      ierr = PetscSFGetVectorSF(hara->sf,B->cmap->N,clda,PETSC_DECIDE,&csf);CHKERRQ(ierr);
      ierr = PetscObjectCompose((PetscObject)C,"_mathara_vectorsf",(PetscObject)csf);CHKERRQ(ierr);
      ierr = PetscObjectDereference((PetscObject)csf);CHKERRQ(ierr);
    }
    blda = n;
    clda = n;
  }
  if (!boundtocpu) {
#if defined(HARA_USE_GPU)
    PetscBool ciscuda,biscuda;

    if (hara->sf) {
      PetscInt n;

      ierr = PetscSFGetGraph(hara->sf,NULL,&n,NULL,NULL);CHKERRQ(ierr);
      if (hara->xxs_gpu < B->cmap->n) { hara->xx_gpu->resize(n*B->cmap->N); hara->xxs_gpu = B->cmap->N; }
      if (hara->yys_gpu < B->cmap->n) { hara->yy_gpu->resize(n*B->cmap->N); hara->yys_gpu = B->cmap->N; }
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
    if (hara->sf) {
      uxx  = MatHaraGetThrustPointer(*hara->xx_gpu);
      uyy  = MatHaraGetThrustPointer(*hara->yy_gpu);
      ierr = PetscSFBcastBegin(bsf,MPIU_SCALAR,xx,uxx,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(bsf,MPIU_SCALAR,xx,uxx,MPI_REPLACE);CHKERRQ(ierr);
    } else {
      uxx = xx;
      uyy = yy;
    }
    if (hara->dist_hmatrix_gpu) {
      if (transA && !A->symmetric) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MatMultTranspose not yet coded in parallel");
      distributed_hgemv(/* transA ? HARA_Trans : HARA_NoTrans, */hara->s, *hara->dist_hmatrix_gpu, uxx, blda, 0.0, uyy, clda, B->cmap->N, hara->handle);
    } else {
      hgemv(transA ? HARA_Trans : HARA_NoTrans, hara->s, *hara->hmatrix_gpu, uxx, blda, 0.0, uyy, clda, B->cmap->N, handle);
    }
    ierr = MatDenseCUDARestoreArrayRead(B,(const PetscScalar**)&xx);CHKERRQ(ierr);
    if (hara->sf) {
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
  } else {
    if (hara->sf) {
      PetscInt n;

      ierr = PetscSFGetGraph(hara->sf,NULL,&n,NULL,NULL);CHKERRQ(ierr);
      if (hara->xxs < B->cmap->n) { hara->xx->resize(n*B->cmap->N); hara->xxs = B->cmap->N; }
      if (hara->yys < B->cmap->n) { hara->yy->resize(n*B->cmap->N); hara->yys = B->cmap->N; }
    }
    ierr = MatDenseGetArrayRead(B,(const PetscScalar**)&xx);CHKERRQ(ierr);
    ierr = MatDenseGetArrayWrite(C,&yy);CHKERRQ(ierr);
    if (hara->sf) {
      uxx  = MatHaraGetThrustPointer(*hara->xx);
      uyy  = MatHaraGetThrustPointer(*hara->yy);
      ierr = PetscSFBcastBegin(bsf,MPIU_SCALAR,xx,uxx,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(bsf,MPIU_SCALAR,xx,uxx,MPI_REPLACE);CHKERRQ(ierr);
    } else {
      uxx = xx;
      uyy = yy;
    }
    if (hara->dist_hmatrix) {
      if (transA && !A->symmetric) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MatMultTranspose not yet coded in parallel");
      distributed_hgemv(/*transA ? HARA_Trans : HARA_NoTrans, */hara->s, *hara->dist_hmatrix, uxx, blda, 0.0, uyy, clda, B->cmap->N, hara->handle);
    } else {
      hgemv(transA ? HARA_Trans : HARA_NoTrans, hara->s, *hara->hmatrix, uxx, blda, 0.0, uyy, clda, B->cmap->N, handle);
    }
    ierr = MatDenseRestoreArrayRead(B,(const PetscScalar**)&xx);CHKERRQ(ierr);
    if (hara->sf) {
      ierr = PetscSFReduceBegin(csf,MPIU_SCALAR,uyy,yy,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(csf,MPIU_SCALAR,uyy,yy,MPI_REPLACE);CHKERRQ(ierr);
    }
    ierr = MatDenseRestoreArrayWrite(C,&yy);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductNumeric_HARA(Mat C)
{
  Mat_Product    *product = C->product;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  MatCheckProduct(C,1);
  switch (product->type) {
  case MATPRODUCT_AB:
    ierr = MatMultNKernel_HARA(product->A,PETSC_FALSE,product->B,C);CHKERRQ(ierr);
    break;
  case MATPRODUCT_AtB:
    ierr = MatMultNKernel_HARA(product->A,PETSC_TRUE,product->B,C);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatProduct type %s is not supported",MatProductTypes[product->type]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSymbolic_HARA(Mat C)
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
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_SUP,"MatProduct type %s is not supported",MatProductTypes[product->type]);
  }
  C->ops->productsymbolic = NULL;
  C->ops->productnumeric = MatProductNumeric_HARA;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatProductSetFromOptions_HARA(Mat C)
{
  PetscFunctionBegin;
  MatCheckProduct(C,1);
  if (C->product->type == MATPRODUCT_AB || C->product->type == MATPRODUCT_AtB) {
    C->ops->productsymbolic = MatProductSymbolic_HARA;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultKernel_HARA(Mat A, Vec x, PetscScalar sy, Vec y, PetscBool trans)
{
  Mat_HARA       *hara = (Mat_HARA*)A->data;
  haraHandle_t   handle = hara->handle->handle;
  PetscBool      boundtocpu = PETSC_TRUE;
  PetscInt       n;
  PetscScalar    *xx,*yy,*uxx,*uyy;
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(HARA_USE_GPU)
  boundtocpu = A->boundtocpu;
#endif
  if (hara->sf) {
    ierr = PetscSFGetGraph(hara->sf,NULL,&n,NULL,NULL);CHKERRQ(ierr);
  } else n = A->rmap->n;
  if (!boundtocpu) {
#if defined(HARA_USE_GPU)
    ierr = VecCUDAGetArrayRead(x,(const PetscScalar**)&xx);CHKERRQ(ierr);
    if (sy == 0.0) {
      ierr = VecCUDAGetArrayWrite(y,&yy);CHKERRQ(ierr);
    } else {
      ierr = VecCUDAGetArray(y,&yy);CHKERRQ(ierr);
    }
    if (hara->sf) {
      uxx = MatHaraGetThrustPointer(*hara->xx_gpu);
      uyy = MatHaraGetThrustPointer(*hara->yy_gpu);

      ierr = PetscSFBcastBegin(hara->sf,MPIU_SCALAR,xx,uxx,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(hara->sf,MPIU_SCALAR,xx,uxx,MPI_REPLACE);CHKERRQ(ierr);
      if (sy != 0.0) {
        ierr = PetscSFBcastBegin(hara->sf,MPIU_SCALAR,yy,uyy,MPI_REPLACE);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(hara->sf,MPIU_SCALAR,yy,uyy,MPI_REPLACE);CHKERRQ(ierr);
      }
    } else {
      uxx = xx;
      uyy = yy;
    }
    if (hara->dist_hmatrix_gpu) {
      if (trans && !A->symmetric) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MatMultTranspose not yet coded in parallel");
      distributed_hgemv(/*trans ? HARA_Trans : HARA_NoTrans, */hara->s, *hara->dist_hmatrix_gpu, uxx, n, sy, uyy, n, 1, hara->handle);
    } else {
      hgemv(trans ? HARA_Trans : HARA_NoTrans, hara->s, *hara->hmatrix_gpu, uxx, n, sy, uyy, n, 1, handle);
    }
    ierr = VecCUDARestoreArrayRead(x,(const PetscScalar**)&xx);CHKERRQ(ierr);
    if (hara->sf) {
      ierr = PetscSFReduceBegin(hara->sf,MPIU_SCALAR,uyy,yy,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(hara->sf,MPIU_SCALAR,uyy,yy,MPI_REPLACE);CHKERRQ(ierr);
    }
    if (sy == 0.0) {
      ierr = VecCUDARestoreArrayWrite(y,&yy);CHKERRQ(ierr);
    } else {
      ierr = VecCUDARestoreArray(y,&yy);CHKERRQ(ierr);
    }
#endif
  } else {
    ierr = VecGetArrayRead(x,(const PetscScalar**)&xx);CHKERRQ(ierr);
    if (sy == 0.0) {
      ierr = VecGetArrayWrite(y,&yy);CHKERRQ(ierr);
    } else {
      ierr = VecGetArray(y,&yy);CHKERRQ(ierr);
    }
    if (hara->sf) {
      uxx = MatHaraGetThrustPointer(*hara->xx);
      uyy = MatHaraGetThrustPointer(*hara->yy);

      ierr = PetscSFBcastBegin(hara->sf,MPIU_SCALAR,xx,uxx,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFBcastEnd(hara->sf,MPIU_SCALAR,xx,uxx,MPI_REPLACE);CHKERRQ(ierr);
      if (sy != 0.0) {
        ierr = PetscSFBcastBegin(hara->sf,MPIU_SCALAR,yy,uyy,MPI_REPLACE);CHKERRQ(ierr);
        ierr = PetscSFBcastEnd(hara->sf,MPIU_SCALAR,yy,uyy,MPI_REPLACE);CHKERRQ(ierr);
      }
    } else {
      uxx = xx;
      uyy = yy;
    }
    if (hara->dist_hmatrix) {
      if (trans && !A->symmetric) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"MatMultTranspose not yet coded in parallel");
      distributed_hgemv(/*trans ? HARA_Trans : HARA_NoTrans, */hara->s, *hara->dist_hmatrix, uxx, n, sy, uyy, n, 1, hara->handle);
    } else {
      hgemv(trans ? HARA_Trans : HARA_NoTrans, hara->s, *hara->hmatrix, uxx, n, sy, uyy, n, 1, handle);
    }
    ierr = VecRestoreArrayRead(x,(const PetscScalar**)&xx);CHKERRQ(ierr);
    if (hara->sf) {
      ierr = PetscSFReduceBegin(hara->sf,MPIU_SCALAR,uyy,yy,MPI_REPLACE);CHKERRQ(ierr);
      ierr = PetscSFReduceEnd(hara->sf,MPIU_SCALAR,uyy,yy,MPI_REPLACE);CHKERRQ(ierr);
    }
    if (sy == 0.0) {
      ierr = VecRestoreArrayWrite(y,&yy);CHKERRQ(ierr);
    } else {
      ierr = VecRestoreArray(y,&yy);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_HARA(Mat A, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultKernel_HARA(A,x,0.0,y,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_HARA(Mat A, Vec x, Vec y)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultKernel_HARA(A,x,0.0,y,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTransposeAdd_HARA(Mat A, Vec x, Vec y, Vec z)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(y,z);CHKERRQ(ierr);
  ierr = MatMultKernel_HARA(A,x,1.0,z,PETSC_TRUE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultAdd_HARA(Mat A, Vec x, Vec y, Vec z)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(y,z);CHKERRQ(ierr);
  ierr = MatMultKernel_HARA(A,x,1.0,z,PETSC_FALSE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatScale_HARA(Mat A, PetscScalar s)
{
  Mat_HARA *a = (Mat_HARA*)A->data;

  PetscFunctionBegin;
  a->s *= s;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetFromOptions_HARA(PetscOptionItems *PetscOptionsObject,Mat A)
{
  Mat_HARA       *a = (Mat_HARA*)A->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"HARA options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_hara_leafsize","Leaf size when constructed from kernel",NULL,a->leafsize,&a->leafsize,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_hara_eta","Admissibility condition tolerance",NULL,a->eta,&a->eta,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_hara_order","Basis order for off-diagonal sampling when constructed from kernel",NULL,a->basisord,&a->basisord,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_hara_maxrank","Maximum rank when constructed from matvecs",NULL,a->max_rank,&a->max_rank,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-mat_hara_samples","Number of samples to be taken concurrently when constructing from matvecs",NULL,a->bs,&a->bs,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsReal("-mat_hara_rtol","Relative tolerance for construction from sampling",NULL,a->rtol,&a->rtol,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-mat_hara_check","Check error when constructing from sampling during MatAssemblyEnd()",NULL,a->check_construction,&a->check_construction,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatHaraSetCoords_HARA(Mat,PetscInt,const PetscReal[],MatHaraKernel,void*);

static PetscErrorCode MatHaraInferCoordinates_Private(Mat A)
{
  Mat_HARA          *a = (Mat_HARA*)A->data;
  Vec               c;
  PetscInt          spacedim;
  const PetscScalar *coords;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (a->spacedim) PetscFunctionReturn(0);
  ierr = PetscObjectQuery((PetscObject)A,"__mathara_coords",(PetscObject*)&c);CHKERRQ(ierr);
  if (!c && a->sampler) {
    Mat S = a->sampler->GetSamplingMat();

    ierr = PetscObjectQuery((PetscObject)S,"__mathara_coords",(PetscObject*)&c);CHKERRQ(ierr);
    if (!c) {
      PetscBool ishara;

      ierr = PetscObjectTypeCompare((PetscObject)S,MATHARA,&ishara);CHKERRQ(ierr);
      if (ishara) {
        Mat_HARA *s = (Mat_HARA*)S->data;

        a->spacedim = s->spacedim;
        if (s->ptcloud1) {
          a->ptcloud1 = new PetscPointCloud1D<PetscReal>(*s->ptcloud1);
        } else if (s->ptcloud2) {
          a->ptcloud2 = new PetscPointCloud2D<PetscReal>(*s->ptcloud2);
        } else if (s->ptcloud3) {
          a->ptcloud3 = new PetscPointCloud3D<PetscReal>(*s->ptcloud3);
        }
      }
    }
  }
  if (!c) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Missing coordinates");
  ierr = VecGetArrayRead(c,&coords);CHKERRQ(ierr);
  ierr = VecGetBlockSize(c,&spacedim);CHKERRQ(ierr);
  ierr = MatHaraSetCoords_HARA(A,spacedim,coords,NULL,NULL);CHKERRQ(ierr);
  ierr = VecRestoreArrayRead(c,&coords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode MatSetUpMultiply_HARA(Mat A)
{
  MPI_Comm       comm;
  PetscMPIInt    size;
  PetscErrorCode ierr;
  Mat_HARA       *a = (Mat_HARA*)A->data;
  IS             is;
  PetscInt       n,*idx;
  int            *iidx;
  PetscCopyMode  own;
  PetscBool      rid;

  PetscFunctionBegin;
  if (a->multsetup) PetscFunctionReturn(0);
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  if (size > 1) {
    iidx = MatHaraGetThrustPointer(a->dist_hmatrix->basis_tree.basis_branch.index_map);
    n    = a->dist_hmatrix->basis_tree.basis_branch.index_map.size();
  } else {
    n    = a->hmatrix->u_basis_tree.index_map.size();
    iidx = MatHaraGetThrustPointer(a->hmatrix->u_basis_tree.index_map);
  }
  if (PetscDefined(USE_64BIT_INDICES)) {
    PetscInt i;

    own  = PETSC_OWN_POINTER;
    ierr = PetscMalloc1(n,&idx);CHKERRQ(ierr);
    for (i=0;i<n;i++) idx[i] = iidx[i];
  } else {
    own  = PETSC_USE_POINTER;
    idx  = iidx;
  }
  ierr = ISCreateGeneral(comm,n,idx,own,&is);CHKERRQ(ierr);
  ierr = ISIdentity(is,&rid);CHKERRQ(ierr);
  if (!rid) {
    ierr = PetscSFCreate(comm,&a->sf);CHKERRQ(ierr);
    ierr = PetscSFSetGraphLayout(a->sf,A->rmap,n,NULL,PETSC_OWN_POINTER,idx);CHKERRQ(ierr);
#if defined(HARA_USE_GPU)
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
  ierr = ISDestroy(&is);CHKERRQ(ierr);
  a->multsetup = PETSC_TRUE;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatAssemblyEnd_HARA(Mat A, MatAssemblyType asstype)
{
  Mat_HARA       *a = (Mat_HARA*)A->data;
  haraHandle_t   handle = a->handle->handle;
  PetscBool      kernel = PETSC_FALSE;
  PetscBool      boundtocpu = PETSC_TRUE;
  MPI_Comm       comm;
  PetscMPIInt    size;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  /* TODO REUSABILITY of geometric construction */
  delete a->hmatrix;
  delete a->dist_hmatrix;
#if defined(HARA_USE_GPU)
  delete a->hmatrix_gpu;
  delete a->dist_hmatrix_gpu;
#endif
  /* TODO: other? */
  BoxCenterAdmissibility<Hara_Real,1> adm1(a->eta,a->leafsize);
  BoxCenterAdmissibility<Hara_Real,2> adm2(a->eta,a->leafsize);
  BoxCenterAdmissibility<Hara_Real,3> adm3(a->eta,a->leafsize);
  if (size > 1) {
    a->dist_hmatrix = new DistributedHMatrix(A->rmap->n/*,A->symmetric*/);
  } else {
    a->hmatrix = new HMatrix(A->rmap->n,A->symmetric);
  }
  ierr = MatHaraInferCoordinates_Private(A);CHKERRQ(ierr);
  switch (a->spacedim) {
  case 1:
    if (!a->ptcloud1) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing pointcloud");
    if (a->kernel1) {
      kernel = PETSC_TRUE;
      if (size > 1) {
        buildDistributedHMatrix(*a->dist_hmatrix,*a->ptcloud1,adm1,*a->kernel1,a->leafsize,a->basisord/*,a->basisord*/,a->handle);
      } else {
        buildHMatrix(*a->hmatrix,*a->ptcloud1,adm1,*a->kernel1,a->leafsize,a->basisord,a->basisord);
      }
    } else {
      if (size > 1) SETERRQ(comm,PETSC_ERR_SUP,"Construction from sampling not supported in parallel");
      buildHMatrixStructure(*a->hmatrix,*a->ptcloud1,adm1,a->leafsize,0,0);
    }
    break;
  case 2:
    if (!a->ptcloud2) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing pointcloud");
    if (a->kernel2) {
      kernel = PETSC_TRUE;
      if (size > 1) {
        buildDistributedHMatrix(*a->dist_hmatrix,*a->ptcloud2,adm2,*a->kernel2,a->leafsize,a->basisord/*,a->basisord*/,a->handle);
      } else {
        buildHMatrix(*a->hmatrix,*a->ptcloud2,adm2,*a->kernel2,a->leafsize,a->basisord,a->basisord);
      }
    } else {
      if (size > 1) SETERRQ(comm,PETSC_ERR_SUP,"Construction from sampling not supported in parallel");
      buildHMatrixStructure(*a->hmatrix,*a->ptcloud2,adm2,a->leafsize,0,0);
    }
    break;
  case 3:
    if (!a->ptcloud3) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Missing pointcloud");
    if (a->kernel3) {
      kernel = PETSC_TRUE;
      if (size > 1) {
        buildDistributedHMatrix(*a->dist_hmatrix,*a->ptcloud3,adm3,*a->kernel3,a->leafsize,a->basisord/*,a->basisord*/,a->handle);
      } else {
        buildHMatrix(*a->hmatrix,*a->ptcloud3,adm3,*a->kernel3,a->leafsize,a->basisord,a->basisord);
      }
    } else {
      if (size > 1) SETERRQ(comm,PETSC_ERR_SUP,"Construction from sampling not supported in parallel");
      buildHMatrixStructure(*a->hmatrix,*a->ptcloud3,adm3,a->leafsize,0,0);
    }
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Unhandled dimension %D",a->spacedim);
  }

  ierr = MatSetUpMultiply_HARA(A);CHKERRQ(ierr);

#if defined(HARA_USE_GPU)
  boundtocpu = A->boundtocpu;
  if (!boundtocpu) {
    if (size > 1) {
      a->dist_hmatrix_gpu = new DistributedHMatrix_GPU(*a->dist_hmatrix);
    } else {
      a->hmatrix_gpu = new GPU_HMatrix(*a->hmatrix);
    }
  }
#endif
  if (size == 1) {
    if (!kernel && a->sampler) {
      PetscReal Anorm;
      bool      verbose = false;

      ierr = MatApproximateNorm_Private(a->sampler->GetSamplingMat(),NORM_2,PETSC_DECIDE,&Anorm);CHKERRQ(ierr);
      if (boundtocpu) {
        a->sampler->SetGPUSampling(false);
        hara(a->sampler, *a->hmatrix, a->max_rank, 10 /* TODO */,a->rtol*Anorm,a->bs,handle,verbose);
#if defined(HARA_USE_GPU)
      } else {
        a->sampler->SetGPUSampling(true);
        hara(a->sampler, *a->hmatrix_gpu, a->max_rank, 10 /* TODO */,a->rtol*Anorm,a->bs,handle,verbose);
#endif
      }
    }
  }
#if defined(HARA_USE_GPU)
  if (kernel) A->offloadmask = PETSC_OFFLOAD_BOTH;
  else A->offloadmask = boundtocpu ? PETSC_OFFLOAD_CPU : PETSC_OFFLOAD_GPU;
#endif

  if (!a->s) a->s = 1.0;
  A->assembled = PETSC_TRUE;

  if (a->sampler) {
    PetscBool check = a->check_construction;

    ierr = PetscOptionsGetBool(((PetscObject)A)->options,((PetscObject)A)->prefix,"-mat_hara_check",&check,NULL);CHKERRQ(ierr);
    if (check) {
      Mat       E,Ae;
      PetscReal n1,ni,n2;
      PetscReal n1A,niA,n2A;
      void      (*normfunc)(void);

      Ae   = a->sampler->GetSamplingMat();
      ierr = MatConvert(A,MATSHELL,MAT_INITIAL_MATRIX,&E);CHKERRQ(ierr);
      ierr = MatShellSetOperation(E,MATOP_NORM,(void (*)(void))MatNorm_HARA);CHKERRQ(ierr);
      ierr = MatAXPY(E,-1.0,Ae,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
      ierr = MatNorm(E,NORM_1,&n1);CHKERRQ(ierr);
      ierr = MatNorm(E,NORM_INFINITY,&ni);CHKERRQ(ierr);
      ierr = MatNorm(E,NORM_2,&n2);CHKERRQ(ierr);

      ierr = MatGetOperation(Ae,MATOP_NORM,&normfunc);CHKERRQ(ierr);
      ierr = MatSetOperation(Ae,MATOP_NORM,(void (*)(void))MatNorm_HARA);CHKERRQ(ierr);
      ierr = MatNorm(Ae,NORM_1,&n1A);CHKERRQ(ierr);
      ierr = MatNorm(Ae,NORM_INFINITY,&niA);CHKERRQ(ierr);
      ierr = MatNorm(Ae,NORM_2,&n2A);CHKERRQ(ierr);
      n1A  = PetscMax(n1A,PETSC_SMALL);
      n2A  = PetscMax(n2A,PETSC_SMALL);
      niA  = PetscMax(niA,PETSC_SMALL);
      ierr = MatSetOperation(Ae,MATOP_NORM,normfunc);CHKERRQ(ierr);
      ierr = PetscPrintf(PetscObjectComm((PetscObject)A),"MATHARA construction errors: NORM_1 %g, NORM_INFINITY %g, NORM_2 %g (%g %g %g)\n",(double)n1,(double)ni,(double)n2,(double)(n1/n1A),(double)(ni/niA),(double)(n2/n2A));
      ierr = MatDestroy(&E);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatZeroEntries_HARA(Mat A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  Mat_HARA       *a = (Mat_HARA*)A->data;

  PetscFunctionBegin;
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRMPI(ierr);
  if (size > 1) SETERRQ(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Not yet supported");
  else {
    a->hmatrix->clearData();
#if defined(HARA_USE_GPU)
    if (a->hmatrix_gpu) a->hmatrix_gpu->clearData();
#endif
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDuplicate_HARA(Mat B, MatDuplicateOption op, Mat *nA)
{
  Mat            A;
  Mat_HARA       *a, *b = (Mat_HARA*)B->data;
#if defined(PETSC_HAVE_CUDA)
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
  ierr = MatSetType(A,MATHARA);CHKERRQ(ierr);
  ierr = MatPropagateSymmetryOptions(B,A);CHKERRQ(ierr);

  a = (Mat_HARA*)A->data;
  a->s = b->s;
  a->spacedim = b->spacedim;
  if (b->ptcloud1) {
    a->ptcloud1 = new PetscPointCloud1D<PetscReal>(*b->ptcloud1);
  } else if (b->ptcloud2) {
    a->ptcloud2 = new PetscPointCloud2D<PetscReal>(*b->ptcloud2);
  } else if (b->ptcloud3) {
    a->ptcloud3 = new PetscPointCloud3D<PetscReal>(*b->ptcloud3);
  }
  if (op == MAT_COPY_VALUES) {
    if (b->kernel1) {
      a->kernel1 = new PetscFunctionGenerator1D<PetscScalar>(*b->kernel1);
    } else if (b->kernel2) {
      a->kernel2 = new PetscFunctionGenerator2D<PetscScalar>(*b->kernel2);
    } else if (b->kernel3) {
      a->kernel3 = new PetscFunctionGenerator3D<PetscScalar>(*b->kernel3);
    }
  }

  if (b->dist_hmatrix) { a->dist_hmatrix = new DistributedHMatrix(*b->dist_hmatrix); }
#if defined(HARA_USE_GPU)
  if (b->dist_hmatrix_gpu) { a->dist_hmatrix_gpu = new DistributedHMatrix_GPU(*b->dist_hmatrix_gpu); }
#endif
  if (b->hmatrix) {
    a->hmatrix = new HMatrix(*b->hmatrix);
    if (op == MAT_DO_NOT_COPY_VALUES) a->hmatrix->clearData();
  }
#if defined(HARA_USE_GPU)
  if (b->hmatrix_gpu) {
    a->hmatrix_gpu = new GPU_HMatrix(*b->hmatrix_gpu);
    if (op == MAT_DO_NOT_COPY_VALUES) a->hmatrix_gpu->clearData();
  }
#endif

  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatSetUpMultiply_HARA(A);CHKERRQ(ierr);
  if (op == MAT_COPY_VALUES) {
    A->assembled = PETSC_TRUE;
#if defined(PETSC_HAVE_CUDA)
    iscpu = B->boundtocpu;
#endif
  }
  ierr = MatBindToCPU(A,iscpu);CHKERRQ(ierr);

  *nA = A;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatView_HARA(Mat A, PetscViewer view)
{
  Mat_HARA          *hara = (Mat_HARA*)A->data;
  PetscBool         isascii;
  PetscErrorCode    ierr;
  PetscMPIInt       size;
  PetscViewerFormat format;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)view,PETSCVIEWERASCII,&isascii);CHKERRQ(ierr);
  ierr = PetscViewerGetFormat(view,&format);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRMPI(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(view,"  H-Matrix constructed from %s\n",hara->sampler ? "Mat" : (hara->spacedim ? "Kernel" : "None"));CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(view,"  PointCloud dim %D\n",hara->spacedim);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(view,"  Admissibility parameters: leaf size %D, eta %g\n",hara->leafsize,(double)hara->eta);CHKERRQ(ierr);
    if (hara->sampler) {
      ierr = PetscViewerASCIIPrintf(view,"  Sampling parameters: max_rank %D, samples %D, tolerance %g\n",hara->max_rank,hara->bs,(double)hara->rtol);CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(view,"  Offdiagonal blocks approximation order %D\n",hara->basisord);CHKERRQ(ierr);
    }
    if (size == 1) {
      double dense_mem_cpu = hara->hmatrix ? hara->hmatrix->getDenseMemoryUsage() : 0;
      double low_rank_cpu = hara->hmatrix ? hara->hmatrix->getLowRankMemoryUsage() : 0;
#if defined(HARA_USE_GPU)
      double dense_mem_gpu = hara->hmatrix_gpu ? hara->hmatrix_gpu->getDenseMemoryUsage() : 0;
      double low_rank_gpu = hara->hmatrix_gpu ? hara->hmatrix_gpu->getLowRankMemoryUsage() : 0;
#endif
      ierr = PetscViewerASCIIPrintf(view,"  Memory consumption (CPU): %g (dense) %g (low rank) %g GB (total)\n", dense_mem_cpu, low_rank_cpu, low_rank_cpu + dense_mem_cpu);CHKERRQ(ierr);
#if defined(HARA_USE_GPU)
      ierr = PetscViewerASCIIPrintf(view,"  Memory consumption (GPU): %g (dense) %g (low rank) %g GB (total)\n", dense_mem_gpu, low_rank_gpu, low_rank_gpu + dense_mem_gpu);CHKERRQ(ierr);
#endif
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode MatHaraSetCoords_HARA(Mat A, PetscInt spacedim, const PetscReal coords[], MatHaraKernel kernel, void *kernelctx)
{
  Mat_HARA       *hara = (Mat_HARA*)A->data;
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
  if (size > 1) {
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

  delete hara->ptcloud1;
  delete hara->ptcloud2;
  delete hara->ptcloud3;
  delete hara->kernel1;
  delete hara->kernel2;
  delete hara->kernel3;
  hara->spacedim = spacedim;
  switch (spacedim) {
  case 1:
    hara->ptcloud1 = new PetscPointCloud1D<PetscReal>(N,gcoords);
    if (kernel) hara->kernel1 = new PetscFunctionGenerator1D<PetscScalar>(kernel,kernelctx);
    break;
  case 2:
    hara->ptcloud2 = new PetscPointCloud2D<PetscReal>(N,gcoords);
    if (kernel) hara->kernel2 = new PetscFunctionGenerator2D<PetscScalar>(kernel,kernelctx);
    break;
  case 3:
    hara->ptcloud3 = new PetscPointCloud3D<PetscReal>(N,gcoords);
    if (kernel) hara->kernel3 = new PetscFunctionGenerator3D<PetscScalar>(kernel,kernelctx);
    break;
  default:
    SETERRQ1(PetscObjectComm((PetscObject)A),PETSC_ERR_SUP,"Unhandled dimension %D",hara->spacedim);
  }
  if (gcoords != coords) { ierr = PetscFree(gcoords);CHKERRQ(ierr); }
  A->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatCreate_HARA(Mat A)
{
  Mat_HARA       *a;
  PetscErrorCode ierr;
  PetscMPIInt    size;

  PetscFunctionBegin;
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
  haraCreateDistributedHandleComm(&a->handle,PetscObjectComm((PetscObject)A));

  ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRMPI(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)A,MATHARA);CHKERRQ(ierr);
  ierr = PetscMemzero(A->ops,sizeof(struct _MatOps));CHKERRQ(ierr);

  A->ops->destroy          = MatDestroy_HARA;
  A->ops->view             = MatView_HARA;
  A->ops->assemblyend      = MatAssemblyEnd_HARA;
  A->ops->mult             = MatMult_HARA;
  A->ops->multtranspose    = MatMultTranspose_HARA;
  A->ops->multadd          = MatMultAdd_HARA;
  A->ops->multtransposeadd = MatMultTransposeAdd_HARA;
  A->ops->scale            = MatScale_HARA;
  A->ops->duplicate        = MatDuplicate_HARA;
  A->ops->setfromoptions   = MatSetFromOptions_HARA;
  A->ops->norm             = MatNorm_HARA;
  A->ops->zeroentries      = MatZeroEntries_HARA;

  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_hara_seqdense_C",MatProductSetFromOptions_HARA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_hara_seqdensecuda_C",MatProductSetFromOptions_HARA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_hara_mpidense_C",MatProductSetFromOptions_HARA);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)A,"MatProductSetFromOptions_hara_mpidensecuda_C",MatProductSetFromOptions_HARA);CHKERRQ(ierr);
#if defined(PETSC_HAVE_CUDA)
  ierr = PetscFree(A->defaultvectype);CHKERRQ(ierr);
  ierr = PetscStrallocpy(VECCUDA,&A->defaultvectype);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode MatHaraSetSamplingMat(Mat A, Mat B, PetscInt bs, PetscReal tol)
{
  PetscBool      ishara;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A,MAT_CLASSID,1);
  PetscValidType(A,1);
  PetscValidHeaderSpecific(B,MAT_CLASSID,2);
  ierr = PetscObjectTypeCompare((PetscObject)A,MATHARA,&ishara);CHKERRQ(ierr);
  if (ishara) {
    Mat_HARA *a = (Mat_HARA*)A->data;

    if (!a->sampler) a->sampler = new PetscMatrixSampler();
    a->sampler->SetSamplingMat(B);
    if (bs > 0) a->bs = bs;
    if (tol > 0.) a->rtol = tol;
    delete a->kernel1;
    delete a->kernel2;
    delete a->kernel3;
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateHaraFromKernel(MPI_Comm comm, PetscInt m, PetscInt n, PetscInt M, PetscInt N, PetscInt spacedim, const PetscReal coords[], MatHaraKernel kernel,void *kernelctx, PetscReal eta, PetscInt leafsize, PetscInt basisord, Mat* nA)
{
  Mat            A;
  Mat_HARA       *hara;
#if defined(PETSC_HAVE_CUDA)
  PetscBool      iscpu = PETSC_FALSE;
#else
  PetscBool      iscpu = PETSC_TRUE;
#endif
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,m,n,M,N);CHKERRQ(ierr);
  ierr = MatSetType(A,MATHARA);CHKERRQ(ierr);
  ierr = MatBindToCPU(A,iscpu);CHKERRQ(ierr);
  ierr = MatHaraSetCoords_HARA(A,spacedim,coords,kernel,kernelctx);CHKERRQ(ierr);

  hara = (Mat_HARA*)A->data;
  if (eta > 0.) hara->eta = eta;
  if (leafsize > 0) hara->leafsize = leafsize;
  if (basisord > 0) hara->basisord = basisord;

  *nA = A;
  PetscFunctionReturn(0);
}

PetscErrorCode MatCreateHaraFromMat(Mat B, PetscInt spacedim, const PetscReal coords[], PetscReal eta, PetscInt leafsize, PetscInt maxrank, PetscInt bs, PetscReal rtol, Mat *nA)
{
  Mat            A;
  Mat_HARA       *hara;
  MPI_Comm       comm;
#if defined(PETSC_HAVE_CUDA)
  PetscBool      iscpu = PETSC_FALSE;
#else
  PetscBool      iscpu = PETSC_TRUE;
#endif
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(B,MAT_CLASSID,1);
  PetscValidLogicalCollectiveInt(B,spacedim,2);
  PetscValidPointer(nA,4);
  ierr = PetscObjectGetComm((PetscObject)B,&comm);CHKERRQ(ierr);
  ierr = MatCreate(comm,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,B->rmap->n,B->cmap->n,B->rmap->N,B->cmap->N);CHKERRQ(ierr);
  ierr = MatSetType(A,MATHARA);CHKERRQ(ierr);
  ierr = MatBindToCPU(A,iscpu);CHKERRQ(ierr);
  if (spacedim) {
    ierr = MatHaraSetCoords_HARA(A,spacedim,coords,NULL,NULL);CHKERRQ(ierr);
  }
  ierr = MatPropagateSymmetryOptions(B,A);CHKERRQ(ierr);
  /* if (!A->symmetric) SETERRQ(comm,PETSC_ERR_SUP,"Unsymmetric sampling does not work"); */

  hara = (Mat_HARA*)A->data;
  hara->sampler = new PetscMatrixSampler(B);
  if (eta > 0.) hara->eta = eta;
  if (leafsize > 0) hara->leafsize = leafsize;
  if (maxrank > 0) hara->max_rank = maxrank;
  if (bs > 0) hara->bs = bs;
  if (rtol > 0.) hara->rtol = rtol;

  *nA = A;
  A->preallocated = PETSC_TRUE;
  PetscFunctionReturn(0);
}
