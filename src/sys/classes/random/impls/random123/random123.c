#include <petsc/private/randomimpl.h>
#include <Random123/threefry.h>

/* The structure of the Random123 methods are similar enough that templates could be used to make the other CBRNGs in
 * the package (aes, ars, philox) available, as well as different block sizes.  But threefry4x64 is a good default,
 * and I'd rather get a simple implementation up and working and come back if there's interest. */
typedef struct _n_PetscRandom123
{
  threefry4x64_ctr_t  counter;
  threefry4x64_key_t  key;
  threefry4x64_ctr_t  result;
  PetscInt            count;
}
PetscRandom123;

R123_ULONG_LONG PETSCR123_SEED_0 = R123_64BIT(0x615D333D2655FE14);
R123_ULONG_LONG PETSCR123_SEED_1 = R123_64BIT(0xAFF6369B3EE9FE96);
R123_ULONG_LONG PETSCR123_SEED_2 = R123_64BIT(0x5956EBC717B60E07);
R123_ULONG_LONG PETSCR123_SEED_3 = R123_64BIT(0xEE8612A0CBEABFF1);

PetscErrorCode PetscRandomSeed_Random123(PetscRandom r)
{
  threefry4x64_ukey_t  ukey;
  PetscRandom123      *r123 = (PetscRandom123 *) r->data;

  PetscFunctionBegin;
  ukey.v[0] = (R123_ULONG_LONG) r->seed;
  ukey.v[1] = PETSCR123_SEED_1;
  ukey.v[2] = PETSCR123_SEED_2;
  ukey.v[3] = PETSCR123_SEED_3;
  /* The point of seeding should be that every time the sequence is seeded you get the same output.  In this CBRNG,
   * that means we have to initialize the key and reset the counts */
  r123->key = threefry4x64keyinit(ukey);
  r123->counter.v[0] = 0;
  r123->counter.v[1] = 1;
  r123->counter.v[2] = 2;
  r123->counter.v[3] = 3;
  r123->result = threefry4x64(r123->counter,r123->key);
  r123->count = 0;
  PetscFunctionReturn(0);
}

static PetscReal PetscRandom123Step(PetscRandom123 *r123)
{
  PetscReal scale = ((PetscReal) 1.) / (UINT64_MAX + ((PetscReal) 1.));
  PetscReal shift = .5 * scale;
  PetscInt  mod   = (r123->count++) % 4;
  PetscReal ret;

  ret = r123->result.v[mod] * scale + shift;

  if (mod == 3) {
    r123->counter.v[0] += 4;
    r123->counter.v[1] += 4;
    r123->counter.v[2] += 4;
    r123->counter.v[3] += 4;
    r123->result = threefry4x64(r123->counter,r123->key);
  }

  return ret;
}

PetscErrorCode PetscRandomGetValue_Random123(PetscRandom r,PetscScalar *val)
{
  PetscRandom123 *r123 = (PetscRandom123 *) r->data;
  PetscScalar     rscal;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  {
    PetscReal re = PetscRandom123Step(r123);
    PetscReal im = PetscRandom123Step(r123);

    if (r->iset) {
      re = re * PetscRealPart(r->width) + PetscRealPart(r->low);
      im = im * PetscImaginaryPart(r->width) + PetscImaginaryPart(r->low);
    }

    rscal = PetscCMPLX(re,im);
  }
#else
  rscal = PetscRandom123Step(r123);
  if (r->iset) rscal = rscal * r->width + r->low;
#endif
  *val = rscal;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscRandomGetValueReal_Random123(PetscRandom r,PetscReal *val)
{
  PetscRandom123 *r123 = (PetscRandom123 *) r->data;
  PetscReal       rreal;

  PetscFunctionBegin;
  rreal = PetscRandom123Step(r123);
  if (r->iset) rreal = rreal * PetscRealPart(r->width) + PetscRealPart(r->low);
  *val = rreal;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscRandomDestroy_Random123(PetscRandom r)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(r->data));
  PetscFunctionReturn(0);
}

static struct _PetscRandomOps PetscRandomOps_Values = {
  PetscDesignatedInitializer(seed,PetscRandomSeed_Random123),
  PetscDesignatedInitializer(getvalue,PetscRandomGetValue_Random123),
  PetscDesignatedInitializer(getvaluereal,PetscRandomGetValueReal_Random123),
  PetscDesignatedInitializer(getvalues,NULL),
  PetscDesignatedInitializer(getvaluesreal,NULL),
  PetscDesignatedInitializer(destroy,PetscRandomDestroy_Random123),
};

/*MC
   PETSCRANDOM123- access to Random123 counter based pseudorandom number generators (currently threefry4x64)

   Options Database Keys:
. -random_type <rand,rand48,sprng,random123>

  Level: beginner

   PETSc must have been ./configure with the option --download-random123 to use
   this random number generator.

.seealso: RandomCreate(), RandomSetType(), PETSCRAND, PETSCRAND48, PETSCSPRNG
M*/

PETSC_EXTERN PetscErrorCode PetscRandomCreate_Random123(PetscRandom r)
{
  PetscRandom123 *r123;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(r,&r123));
  r->data = r123;
  PetscCall(PetscMemcpy(r->ops,&PetscRandomOps_Values,sizeof(PetscRandomOps_Values)));
  PetscCall(PetscObjectChangeTypeName((PetscObject)r,PETSCRANDOM123));
  PetscCall(PetscRandomSeed(r));
  PetscFunctionReturn(0);
}
