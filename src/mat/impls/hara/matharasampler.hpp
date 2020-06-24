#include <petscmat.h>
#include <hara.h>

#ifndef __MATHARA_HPP
#define __MATHARA_HPP

class PetscMatrixSampler : public HMatrixSampler
{
protected:
  Mat  A;
  bool gpusampling;

private:
  void Init();

public:
  PetscMatrixSampler();
  PetscMatrixSampler(Mat);
  ~PetscMatrixSampler();
  void SetSamplingMat(Mat);
  void SetGPUSampling(bool);
  virtual void sample(Hara_Real*,Hara_Real*,int);
  Mat GetSamplingMat() { return A; }
};

void PetscMatrixSampler::Init()
{
  this->A = NULL;
  this->gpusampling = false;
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

  ierr = PetscObjectReference((PetscObject)A);CHKERRCONTINUE(ierr);
  ierr = MatDestroy(&this->A);CHKERRCONTINUE(ierr);
  this->A = A;
}

void PetscMatrixSampler::SetGPUSampling(bool gpusampling)
{
  this->gpusampling = gpusampling;
}

PetscMatrixSampler::~PetscMatrixSampler()
{
  PetscErrorCode ierr;

  ierr = MatDestroy(&A);CHKERRCONTINUE(ierr);
}

void PetscMatrixSampler::sample(Hara_Real *x, Hara_Real *y, int samples)
{
  PetscErrorCode ierr;
  MPI_Comm       comm = PetscObjectComm((PetscObject)this->A);
  Mat            X,Y;
  PetscInt       M,N,m,n;

  ierr = MatGetLocalSize(this->A,&m,&n);CHKERRCONTINUE(ierr);
  ierr = MatGetSize(this->A,&M,&N);CHKERRCONTINUE(ierr);
  if (!this->gpusampling) {
    ierr = MatCreateDense(comm,n,PETSC_DECIDE,N,samples,x,&X);CHKERRCONTINUE(ierr);
    ierr = MatCreateDense(comm,m,PETSC_DECIDE,M,samples,y,&Y);CHKERRCONTINUE(ierr);
  } else {
#if defined(PETSC_HAVE_CUDA)
    ierr = MatCreateDenseCUDA(comm,n,PETSC_DECIDE,N,samples,x,&X);CHKERRCONTINUE(ierr);
    ierr = MatCreateDenseCUDA(comm,m,PETSC_DECIDE,M,samples,y,&Y);CHKERRCONTINUE(ierr);
#endif
  }
  ierr = PetscLogObjectParent((PetscObject)A,(PetscObject)X);CHKERRCONTINUE(ierr);
  ierr = PetscLogObjectParent((PetscObject)A,(PetscObject)Y);CHKERRCONTINUE(ierr);
  ierr = MatMatMult(this->A,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&Y);CHKERRCONTINUE(ierr);
  ierr = MatDestroy(&X);CHKERRCONTINUE(ierr);
  ierr = MatDestroy(&Y);CHKERRCONTINUE(ierr);
}

#endif
