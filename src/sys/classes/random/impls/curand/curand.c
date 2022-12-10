#include <petsc/private/deviceimpl.h>
#include <petsc/private/randomimpl.h>
#include <petscdevice_cuda.h>
#include <curand.h>

typedef struct {
  curandGenerator_t gen;
} PetscRandom_CURAND;

PetscErrorCode PetscRandomSeed_CURAND(PetscRandom r)
{
  PetscRandom_CURAND *curand = (PetscRandom_CURAND *)r->data;

  PetscFunctionBegin;
  PetscCallCURAND(curandSetPseudoRandomGeneratorSeed(curand->gen, r->seed));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode PetscRandomCurandScale_Private(PetscRandom, size_t, PetscReal *, PetscBool);

PetscErrorCode PetscRandomGetValuesReal_CURAND(PetscRandom r, PetscInt n, PetscReal *val)
{
  PetscRandom_CURAND *curand = (PetscRandom_CURAND *)r->data;
  size_t              nn     = n < 0 ? (size_t)(-2 * n) : n; /* handle complex case */

  PetscFunctionBegin;
#if defined(PETSC_USE_REAL_SINGLE)
  PetscCallCURAND(curandGenerateUniform(curand->gen, val, nn));
#else
  PetscCallCURAND(curandGenerateUniformDouble(curand->gen, val, nn));
#endif
  if (r->iset) PetscCall(PetscRandomCurandScale_Private(r, nn, val, (PetscBool)(n < 0)));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscRandomGetValues_CURAND(PetscRandom r, PetscInt n, PetscScalar *val)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  /* pass negative size to flag complex scaling (if needed) */
  PetscCall(PetscRandomGetValuesReal_CURAND(r, -n, (PetscReal *)val));
#else
  PetscCall(PetscRandomGetValuesReal_CURAND(r, n, val));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscRandomDestroy_CURAND(PetscRandom r)
{
  PetscRandom_CURAND *curand = (PetscRandom_CURAND *)r->data;

  PetscFunctionBegin;
  PetscCallCURAND(curandDestroyGenerator(curand->gen));
  PetscCall(PetscFree(r->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static struct _PetscRandomOps PetscRandomOps_Values = {
  PetscDesignatedInitializer(seed, PetscRandomSeed_CURAND),
  PetscDesignatedInitializer(getvalue, NULL),
  PetscDesignatedInitializer(getvaluereal, NULL),
  PetscDesignatedInitializer(getvalues, PetscRandomGetValues_CURAND),
  PetscDesignatedInitializer(getvaluesreal, PetscRandomGetValuesReal_CURAND),
  PetscDesignatedInitializer(destroy, PetscRandomDestroy_CURAND),
};

/*MC
   PETSCCURAND - access to the CUDA random number generator from a `PetscRandom` object

  PETSc must be ./configure with the option --with-cuda to use this random number generator.

  Level: beginner

.seealso: `PetscRandomCreate()`, `PetscRandomSetType()`, `PetscRandomType`
M*/

PETSC_EXTERN PetscErrorCode PetscRandomCreate_CURAND(PetscRandom r)
{
  PetscRandom_CURAND *curand;

  PetscFunctionBegin;
  PetscCall(PetscDeviceInitialize(PETSC_DEVICE_CUDA));
  PetscCall(PetscNew(&curand));
  PetscCallCURAND(curandCreateGenerator(&curand->gen, CURAND_RNG_PSEUDO_DEFAULT));
  /* https://docs.nvidia.com/cuda/curand/host-api-overview.html#performance-notes2 */
  PetscCallCURAND(curandSetGeneratorOrdering(curand->gen, CURAND_ORDERING_PSEUDO_SEEDED));
  PetscCall(PetscMemcpy(r->ops, &PetscRandomOps_Values, sizeof(PetscRandomOps_Values)));
  PetscCall(PetscObjectChangeTypeName((PetscObject)r, PETSCCURAND));
  r->data = curand;
  PetscCall(PetscRandomSeed_CURAND(r));
  PetscFunctionReturn(PETSC_SUCCESS);
}
