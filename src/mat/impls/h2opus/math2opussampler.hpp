#include <petscmat.h>
#include <h2opus.h>

#ifndef __MATH2OPUS_HPP
#define __MATH2OPUS_HPP

class PetscMatrixSampler : public HMatrixSampler
{
protected:
  Mat  A;
  typedef typename VectorContainer<H2OPUS_HWTYPE_CPU, H2Opus_Real>::type HRealVector;
  typedef typename VectorContainer<H2OPUS_HWTYPE_CPU, int>::type HIntVector;
  HIntVector hindexmap;
  HRealVector hbuffer_in,hbuffer_out;
#if defined(PETSC_HAVE_CUDA) && defined(H2OPUS_USE_GPU)
  H2OpusDeviceVector<int> dindexmap;
  H2OpusDeviceVector<H2Opus_Real> dbuffer_in,dbuffer_out;
#endif
  bool gpusampling;
  h2opusComputeStream_t stream;

private:
  void Init();
  void VerifyBuffers(int);
  void PermuteBuffersIn(int,H2Opus_Real*,H2Opus_Real**,H2Opus_Real*,H2Opus_Real**);
  void PermuteBuffersOut(int,H2Opus_Real*);

public:
  PetscMatrixSampler();
  PetscMatrixSampler(Mat);
  ~PetscMatrixSampler();
  void SetSamplingMat(Mat);
  void SetIndexMap(int,int*);
  void SetGPUSampling(bool);
  void SetStream(h2opusComputeStream_t);
  virtual void sample(H2Opus_Real*,H2Opus_Real*,int);
  Mat GetSamplingMat() { return A; }
};

void PetscMatrixSampler::Init()
{
  this->A = NULL;
  this->gpusampling = false;
  this->stream = NULL;
}

PetscMatrixSampler::PetscMatrixSampler()
{
  Init();
}

PetscMatrixSampler::PetscMatrixSampler(Mat A)
{
  Init();
  SetSamplingMat(A);
}

void PetscMatrixSampler::SetSamplingMat(Mat A)
{
  PetscErrorCode ierr;
  PetscMPIInt    size = 1;

  if (A) { ierr = MPI_Comm_size(PetscObjectComm((PetscObject)A),&size);CHKERRV(ierr); }
  if (size > 1) CHKERRV(PETSC_ERR_SUP);
  ierr = PetscObjectReference((PetscObject)A);CHKERRV(ierr);
  ierr = MatDestroy(&this->A);CHKERRV(ierr);
  this->A = A;
}

void PetscMatrixSampler::SetStream(h2opusComputeStream_t stream)
{
  this->stream = stream;
}

void PetscMatrixSampler::SetIndexMap(int n,int *indexmap)
{
  copyVector(this->hindexmap, indexmap, n, H2OPUS_HWTYPE_CPU);
#if defined(PETSC_HAVE_CUDA) && defined(H2OPUS_USE_GPU)
  copyVector(this->dindexmap, indexmap, n, H2OPUS_HWTYPE_CPU);
#endif
}

void PetscMatrixSampler::VerifyBuffers(int nv)
{
  if (this->hindexmap.size()) {
    size_t n = this->hindexmap.size();
    if (!this->gpusampling) {
      if (hbuffer_in.size() < (size_t)n * nv)
          hbuffer_in.resize(n * nv);
      if (hbuffer_out.size() < (size_t)n * nv)
          hbuffer_out.resize(n * nv);
    } else {
#if defined(PETSC_HAVE_CUDA) && defined(H2OPUS_USE_GPU)
      if (dbuffer_in.size() < (size_t)n * nv)
          dbuffer_in.resize(n * nv);
      if (dbuffer_out.size() < (size_t)n * nv)
          dbuffer_out.resize(n * nv);
#endif
    }
  }
}

void PetscMatrixSampler::PermuteBuffersIn(int nv, H2Opus_Real *v, H2Opus_Real **w, H2Opus_Real *ov, H2Opus_Real **ow)
{
  *w = v;
  *ow = ov;
  VerifyBuffers(nv);
  if (this->hindexmap.size()) {
    size_t n = this->hindexmap.size();
    if (!this->gpusampling) {
      permute_vectors(v, this->hbuffer_in.data(), n, nv, this->hindexmap.data(), 1, H2OPUS_HWTYPE_CPU,
                      this->stream);
      *w = this->hbuffer_in.data();
      *ow = this->hbuffer_out.data();
    } else {
#if defined(PETSC_HAVE_CUDA) && defined(H2OPUS_USE_GPU)
      permute_vectors(v, this->dbuffer_in.data(), n, nv, this->dindexmap.data(), 1, H2OPUS_HWTYPE_GPU,
                      this->stream);
      *w = this->dbuffer_in.data();
      *ow = this->dbuffer_out.data();
#endif
    }
  }
}

void PetscMatrixSampler::PermuteBuffersOut(int nv, H2Opus_Real *v)
{
  VerifyBuffers(nv);
  if (this->hindexmap.size()) {
    size_t n = this->hindexmap.size();
    if (!this->gpusampling) {
      permute_vectors(this->hbuffer_out.data(), v, n, nv, this->hindexmap.data(), 0, H2OPUS_HWTYPE_CPU,
                      this->stream);
    } else {
#if defined(PETSC_HAVE_CUDA) && defined(H2OPUS_USE_GPU)
      permute_vectors(this->dbuffer_out.data(), v, n, nv, this->dindexmap.data(), 0, H2OPUS_HWTYPE_GPU,
                      this->stream);
#endif
    }
  }
}

void PetscMatrixSampler::SetGPUSampling(bool gpusampling)
{
  this->gpusampling = gpusampling;
}

PetscMatrixSampler::~PetscMatrixSampler()
{
  PetscErrorCode ierr;

  ierr = MatDestroy(&A);CHKERRV(ierr);
}

void PetscMatrixSampler::sample(H2Opus_Real *x, H2Opus_Real *y, int samples)
{
  PetscErrorCode ierr;
  MPI_Comm       comm = PetscObjectComm((PetscObject)this->A);
  Mat            X = NULL,Y = NULL;
  PetscInt       M,N,m,n;
  H2Opus_Real    *px,*py;

  if (!this->A) CHKERRV(PETSC_ERR_PLIB);
  ierr = MatGetSize(this->A,&M,&N);CHKERRV(ierr);
  ierr = MatGetLocalSize(this->A,&m,&n);CHKERRV(ierr);
  ierr = PetscObjectGetComm((PetscObject)A,&comm);CHKERRV(ierr);
  PermuteBuffersIn(samples,x,&px,y,&py);
  if (!this->gpusampling) {
    ierr = MatCreateDense(comm,n,PETSC_DECIDE,N,samples,px,&X);CHKERRV(ierr);
    ierr = MatCreateDense(comm,m,PETSC_DECIDE,M,samples,py,&Y);CHKERRV(ierr);
  } else {
#if defined(PETSC_HAVE_CUDA)
    ierr = MatCreateDenseCUDA(comm,n,PETSC_DECIDE,N,samples,px,&X);CHKERRV(ierr);
    ierr = MatCreateDenseCUDA(comm,m,PETSC_DECIDE,M,samples,py,&Y);CHKERRV(ierr);
#endif
  }
  ierr = PetscLogObjectParent((PetscObject)this->A,(PetscObject)X);CHKERRV(ierr);
  ierr = PetscLogObjectParent((PetscObject)this->A,(PetscObject)Y);CHKERRV(ierr);
  ierr = MatMatMult(this->A,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&Y);CHKERRV(ierr);
#if defined(PETSC_HAVE_CUDA)
  if (this->gpusampling) {
    const PetscScalar *dummy;
    ierr = MatDenseCUDAGetArrayRead(Y,&dummy);CHKERRV(ierr);
    ierr = MatDenseCUDARestoreArrayRead(Y,&dummy);CHKERRV(ierr);
  }
#endif
  PermuteBuffersOut(samples,y);
  ierr = MatDestroy(&X);CHKERRV(ierr);
  ierr = MatDestroy(&Y);CHKERRV(ierr);
}

#endif
