#include <petsc/private/randomimpl.h>
#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

#if defined(PETSC_USE_COMPLEX)
struct complexscalelw
  #if PETSC_PKG_CUDA_VERSION_LT(12, 8, 0)
  :
  public thrust::unary_function<thrust::tuple<PetscReal, size_t>, PetscReal>
  #endif
{
  PetscReal rl, rw;
  PetscReal il, iw;

  complexscalelw(PetscScalar low, PetscScalar width)
  {
    rl = PetscRealPart(low);
    il = PetscImaginaryPart(low);
    rw = PetscRealPart(width);
    iw = PetscImaginaryPart(width);
  }

  __host__ __device__ PetscReal operator()(thrust::tuple<PetscReal, size_t> x) { return thrust::get<1>(x) % 2 ? thrust::get<0>(x) * iw + il : thrust::get<0>(x) * rw + rl; }
};
#endif

struct realscalelw
#if PETSC_PKG_CUDA_VERSION_LT(12, 8, 0) // To suppress the warning "thrust::THRUST_200700_860_NS::unary_function is deprecated"
  :
  public thrust::unary_function<PetscReal, PetscReal>
#endif
{
  PetscReal l, w;

  realscalelw(PetscReal low, PetscReal width) : l(low), w(width) { }

  __host__ __device__ PetscReal operator()(PetscReal x) { return x * w + l; }
};

PETSC_INTERN PetscErrorCode PetscRandomCurandScale_Private(PetscRandom r, size_t n, PetscReal *val, PetscBool isneg)
{
  PetscFunctionBegin;
  if (!r->iset) PetscFunctionReturn(PETSC_SUCCESS);
  if (isneg) { /* complex case, need to scale differently */
#if defined(PETSC_USE_COMPLEX)
    thrust::device_ptr<PetscReal> pval  = thrust::device_pointer_cast(val);
    auto                          zibit = thrust::make_zip_iterator(thrust::make_tuple(pval, thrust::counting_iterator<size_t>(0)));
    thrust::transform(zibit, zibit + n, pval, complexscalelw(r->low, r->width));
#else
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Negative array size %" PetscInt_FMT, (PetscInt)n);
#endif
  } else {
    PetscReal                     rl   = PetscRealPart(r->low);
    PetscReal                     rw   = PetscRealPart(r->width);
    thrust::device_ptr<PetscReal> pval = thrust::device_pointer_cast(val);
    thrust::transform(pval, pval + n, pval, realscalelw(rl, rw));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
