#include <../src/sys/classes/random/randomimpl.h>
#include <thrust/transform.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/counting_iterator.h>

#if defined(PETSC_USE_COMPLEX)
struct complexscalelw : public thrust::unary_function<thrust::tuple<PetscReal, size_t>,PetscReal>
{
  PetscReal rl,rw;
  PetscReal il,iw;

  complexscalelw(PetscScalar low, PetscScalar width) {
    rl = PetscRealPart(low);
    il = PetscImaginaryPart(low);
    rw = PetscRealPart(width);
    iw = PetscImaginaryPart(width);
  }

  __host__ __device__
  PetscReal operator()(thrust::tuple<PetscReal, size_t> x) {
    return x.get<1>()%2 ? x.get<0>()*iw + il : x.get<0>()*rw + rl;
  }
};
#endif

struct realscalelw : public thrust::unary_function<PetscReal,PetscReal>
{
  PetscReal l,w;

  realscalelw(PetscReal low, PetscReal width) : l(low), w(width) {}

  __host__ __device__
  PetscReal operator()(PetscReal x) {
    return x*w + l;
  }
};

PETSC_INTERN PetscErrorCode PetscRandomCurandScale_Private(PetscRandom r, size_t n, PetscReal *val, PetscBool isneg)
{
  PetscFunctionBegin;
  if (!r->iset) PetscFunctionReturn(0);
  if (isneg) { /* complex case, need to scale differently */
#if defined(PETSC_USE_COMPLEX)
    thrust::device_ptr<PetscReal> pval = thrust::device_pointer_cast(val);
    auto zibit = thrust::make_zip_iterator(thrust::make_tuple(pval,thrust::counting_iterator<size_t>(0)));
    thrust::transform(zibit,zibit+n,pval,complexscalelw(r->low,r->width));
#else
    SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Negative array size %D",(PetscInt)n);
#endif
  } else {
    PetscReal rl = PetscRealPart(r->low);
    PetscReal rw = PetscRealPart(r->width);
    thrust::device_ptr<PetscReal> pval = thrust::device_pointer_cast(val);
    thrust::transform(pval,pval+n,pval,realscalelw(rl,rw));
  }
  PetscFunctionReturn(0);
}
