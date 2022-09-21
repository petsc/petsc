#include <petscmat.h>
#include <h2opus.h>

#ifndef __MATH2OPUS_HPP
  #define __MATH2OPUS_HPP

class PetscMatrixSampler : public HMatrixSampler {
protected:
  Mat                                                                    A;
  typedef typename VectorContainer<H2OPUS_HWTYPE_CPU, H2Opus_Real>::type HRealVector;
  typedef typename VectorContainer<H2OPUS_HWTYPE_CPU, int>::type         HIntVector;
  HIntVector                                                             hindexmap;
  HRealVector                                                            hbuffer_in, hbuffer_out;
  #if defined(PETSC_HAVE_CUDA) && defined(H2OPUS_USE_GPU)
  H2OpusDeviceVector<int>         dindexmap;
  H2OpusDeviceVector<H2Opus_Real> dbuffer_in, dbuffer_out;
  #endif
  bool                  gpusampling;
  h2opusComputeStream_t stream;

private:
  void Init();
  void VerifyBuffers(int);
  void PermuteBuffersIn(int, H2Opus_Real *, H2Opus_Real **, H2Opus_Real *, H2Opus_Real **);
  void PermuteBuffersOut(int, H2Opus_Real *);

public:
  PetscMatrixSampler();
  PetscMatrixSampler(Mat);
  ~PetscMatrixSampler();
  void         SetSamplingMat(Mat);
  void         SetIndexMap(int, int *);
  void         SetGPUSampling(bool);
  void         SetStream(h2opusComputeStream_t);
  virtual void sample(H2Opus_Real *, H2Opus_Real *, int);
  Mat          GetSamplingMat() { return A; }
};

void PetscMatrixSampler::Init()
{
  this->A           = NULL;
  this->gpusampling = false;
  this->stream      = NULL;
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
  PetscMPIInt size = 1;

  if (A) PetscCallVoid(MPI_Comm_size(PetscObjectComm((PetscObject)A), &size));
  if (size > 1) PetscCallVoid(PETSC_ERR_SUP);
  PetscCallVoid(PetscObjectReference((PetscObject)A));
  PetscCallVoid(MatDestroy(&this->A));
  this->A = A;
}

void PetscMatrixSampler::SetStream(h2opusComputeStream_t stream)
{
  this->stream = stream;
}

void PetscMatrixSampler::SetIndexMap(int n, int *indexmap)
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
      if (hbuffer_in.size() < (size_t)n * nv) hbuffer_in.resize(n * nv);
      if (hbuffer_out.size() < (size_t)n * nv) hbuffer_out.resize(n * nv);
    } else {
  #if defined(PETSC_HAVE_CUDA) && defined(H2OPUS_USE_GPU)
      if (dbuffer_in.size() < (size_t)n * nv) dbuffer_in.resize(n * nv);
      if (dbuffer_out.size() < (size_t)n * nv) dbuffer_out.resize(n * nv);
  #endif
    }
  }
}

void PetscMatrixSampler::PermuteBuffersIn(int nv, H2Opus_Real *v, H2Opus_Real **w, H2Opus_Real *ov, H2Opus_Real **ow)
{
  *w  = v;
  *ow = ov;
  VerifyBuffers(nv);
  if (this->hindexmap.size()) {
    size_t n = this->hindexmap.size();
    if (!this->gpusampling) {
      permute_vectors(v, this->hbuffer_in.data(), n, nv, this->hindexmap.data(), 1, H2OPUS_HWTYPE_CPU, this->stream);
      *w  = this->hbuffer_in.data();
      *ow = this->hbuffer_out.data();
    } else {
  #if defined(PETSC_HAVE_CUDA) && defined(H2OPUS_USE_GPU)
      permute_vectors(v, this->dbuffer_in.data(), n, nv, this->dindexmap.data(), 1, H2OPUS_HWTYPE_GPU, this->stream);
      *w  = this->dbuffer_in.data();
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
      permute_vectors(this->hbuffer_out.data(), v, n, nv, this->hindexmap.data(), 0, H2OPUS_HWTYPE_CPU, this->stream);
    } else {
  #if defined(PETSC_HAVE_CUDA) && defined(H2OPUS_USE_GPU)
      permute_vectors(this->dbuffer_out.data(), v, n, nv, this->dindexmap.data(), 0, H2OPUS_HWTYPE_GPU, this->stream);
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
  PetscCallVoid(MatDestroy(&A));
}

void PetscMatrixSampler::sample(H2Opus_Real *x, H2Opus_Real *y, int samples)
{
  MPI_Comm     comm = PetscObjectComm((PetscObject)this->A);
  Mat          X = NULL, Y = NULL;
  PetscInt     M, N, m, n;
  H2Opus_Real *px, *py;

  if (!this->A) PetscCallVoid(PETSC_ERR_PLIB);
  PetscCallVoid(MatGetSize(this->A, &M, &N));
  PetscCallVoid(MatGetLocalSize(this->A, &m, &n));
  PetscCallVoid(PetscObjectGetComm((PetscObject)A, &comm));
  PermuteBuffersIn(samples, x, &px, y, &py);
  if (!this->gpusampling) {
    PetscCallVoid(MatCreateDense(comm, n, PETSC_DECIDE, N, samples, px, &X));
    PetscCallVoid(MatCreateDense(comm, m, PETSC_DECIDE, M, samples, py, &Y));
  } else {
  #if defined(PETSC_HAVE_CUDA)
    PetscCallVoid(MatCreateDenseCUDA(comm, n, PETSC_DECIDE, N, samples, px, &X));
    PetscCallVoid(MatCreateDenseCUDA(comm, m, PETSC_DECIDE, M, samples, py, &Y));
  #endif
  }
  PetscCallVoid(MatMatMult(this->A, X, MAT_REUSE_MATRIX, PETSC_DEFAULT, &Y));
  #if defined(PETSC_HAVE_CUDA)
  if (this->gpusampling) {
    const PetscScalar *dummy;
    PetscCallVoid(MatDenseCUDAGetArrayRead(Y, &dummy));
    PetscCallVoid(MatDenseCUDARestoreArrayRead(Y, &dummy));
  }
  #endif
  PermuteBuffersOut(samples, y);
  PetscCallVoid(MatDestroy(&X));
  PetscCallVoid(MatDestroy(&Y));
}

#endif
