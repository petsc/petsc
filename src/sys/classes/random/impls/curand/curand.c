#include <petsc/private/randomimpl.h>
#include <curand.h>

#define CHKERRCURAND(stat) \
do { \
   if (PetscUnlikely(stat != CURAND_STATUS_SUCCESS)) { \
     if (((stat == CURAND_STATUS_INITIALIZATION_FAILED) || (stat == CURAND_STATUS_ALLOCATION_FAILED)) && PetscCUDAInitialized) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_GPU_RESOURCE,"cuRAND error %d. Reports not initialized or alloc failed; this indicates the GPU has run out resources",(int)stat); \
     else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_GPU,"cuRand error %d",(int)stat); \
   } \
} while (0)

typedef struct {
  curandGenerator_t gen;
} PetscRandom_CURAND;

PetscErrorCode PetscRandomSeed_CURAND(PetscRandom r)
{
  curandStatus_t     cerr;
  PetscRandom_CURAND *curand = (PetscRandom_CURAND*)r->data;

  PetscFunctionBegin;
  cerr = curandSetPseudoRandomGeneratorSeed(curand->gen,r->seed);CHKERRCURAND(cerr);
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode PetscRandomCurandScale_Private(PetscRandom,size_t,PetscReal*,PetscBool);

PetscErrorCode  PetscRandomGetValuesReal_CURAND(PetscRandom r, PetscInt n, PetscReal *val)
{
  curandStatus_t     cerr;
  PetscRandom_CURAND *curand = (PetscRandom_CURAND*)r->data;
  size_t             nn = n < 0 ? (size_t)(-2*n) : n; /* handle complex case */

  PetscFunctionBegin;
#if defined(PETSC_USE_REAL_SINGLE)
  cerr = curandGenerateUniform(curand->gen,val,nn);CHKERRCURAND(cerr);
#else
  cerr = curandGenerateUniformDouble(curand->gen,val,nn);CHKERRCURAND(cerr);
#endif
  if (r->iset) {
    PetscErrorCode ierr = PetscRandomCurandScale_Private(r,nn,val,(PetscBool)(n<0));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscRandomGetValues_CURAND(PetscRandom r, PetscInt n, PetscScalar *val)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  /* pass negative size to flag complex scaling (if needed) */
  ierr = PetscRandomGetValuesReal_CURAND(r,-n,(PetscReal*)val);CHKERRQ(ierr);
#else
  ierr = PetscRandomGetValuesReal_CURAND(r,n,val);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

PetscErrorCode PetscRandomDestroy_CURAND(PetscRandom r)
{
  PetscErrorCode     ierr;
  curandStatus_t     cerr;
  PetscRandom_CURAND *curand = (PetscRandom_CURAND*)r->data;

  PetscFunctionBegin;
  cerr = curandDestroyGenerator(curand->gen);CHKERRCURAND(cerr);
  ierr = PetscFree(r->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static struct _PetscRandomOps PetscRandomOps_Values = {
  PetscRandomSeed_CURAND,
  NULL,
  NULL,
  PetscRandomGetValues_CURAND,
  PetscRandomGetValuesReal_CURAND,
  PetscRandomDestroy_CURAND,
  NULL
};

/*MC
   PETSCCURAND - access to the CUDA random number generator

  Level: beginner

.seealso: PetscRandomCreate(), PetscRandomSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscRandomCreate_CURAND(PetscRandom r)
{
  PetscErrorCode     ierr;
  curandStatus_t     cerr;
  PetscRandom_CURAND *curand;

  PetscFunctionBegin;
  ierr = PetscCUDAInitializeCheck();CHKERRQ(ierr);
  ierr = PetscNewLog(r,&curand);CHKERRQ(ierr);
  cerr = curandCreateGenerator(&curand->gen,CURAND_RNG_PSEUDO_DEFAULT);CHKERRCURAND(cerr);
  /* https://docs.nvidia.com/cuda/curand/host-api-overview.html#performance-notes2 */
  cerr = curandSetGeneratorOrdering(curand->gen,CURAND_ORDERING_PSEUDO_SEEDED);CHKERRCURAND(cerr);
  ierr = PetscMemcpy(r->ops,&PetscRandomOps_Values,sizeof(PetscRandomOps_Values));CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)r,PETSCCURAND);CHKERRQ(ierr);
  r->data = curand;
  r->seed = 1234ULL; /* taken from example */
  ierr = PetscRandomSeed_CURAND(r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
