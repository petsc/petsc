#include <../src/sys/classes/random/randomimpl.h>

#define RANDER48_SEED_0 (0x330e)
#define RANDER48_SEED_1 (0xabcd)
#define RANDER48_SEED_2 (0x1234)
#define RANDER48_MULT_0 (0xe66d)
#define RANDER48_MULT_1 (0xdeec)
#define RANDER48_MULT_2 (0x0005)
#define RANDER48_ADD    (0x000b)

unsigned short _rander48_seed[3] = {
  RANDER48_SEED_0,
  RANDER48_SEED_1,
  RANDER48_SEED_2
};
unsigned short _rander48_mult[3] = {
  RANDER48_MULT_0,
  RANDER48_MULT_1,
  RANDER48_MULT_2
};
unsigned short _rander48_add = RANDER48_ADD;

void _dorander48(unsigned short xseed[3])
{
  unsigned long accu;
  unsigned short temp[2];

  accu     = (unsigned long) _rander48_mult[0] * (unsigned long) xseed[0] + (unsigned long)_rander48_add;
  temp[0]  = (unsigned short) accu;        /* lower 16 bits */
  accu   >>= sizeof(unsigned short) * 8;
  accu    += (unsigned long) _rander48_mult[0] * (unsigned long) xseed[1] + (unsigned long) _rander48_mult[1] * (unsigned long) xseed[0];
  temp[1]  = (unsigned short)accu;        /* middle 16 bits */
  accu   >>= sizeof(unsigned short) * 8;
  accu    += _rander48_mult[0] * xseed[2] + _rander48_mult[1] * xseed[1] + _rander48_mult[2] * xseed[0];
  xseed[0] = temp[0];
  xseed[1] = temp[1];
  xseed[2] = (unsigned short) accu;
}

double erander48(unsigned short xseed[3])
{
  _dorander48(xseed);
  return ldexp((double) xseed[0], -48) + ldexp((double) xseed[1], -32) + ldexp((double) xseed[2], -16);
}

double drander48() {
  return erander48(_rander48_seed);
}

void srander48(long seed) {
  _rander48_seed[0] = RANDER48_SEED_0;
  _rander48_seed[1] = (unsigned short) seed;
  _rander48_seed[2] = (unsigned short) (seed >> 16);
  _rander48_mult[0] = RANDER48_MULT_0;
  _rander48_mult[1] = RANDER48_MULT_1;
  _rander48_mult[2] = RANDER48_MULT_2;
  _rander48_add     = RANDER48_ADD;
}

#undef __FUNCT__
#define __FUNCT__ "PetscRandomSeed_Rander48"
PetscErrorCode  PetscRandomSeed_Rander48(PetscRandom r)
{
  PetscFunctionBegin;
  srander48(r->seed);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscRandomGetValue_Rander48"
PetscErrorCode  PetscRandomGetValue_Rander48(PetscRandom r, PetscScalar *val)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  if (r->iset) {
    *val = PetscRealPart(r->low) + PetscImaginaryPart(r->low) * PETSC_i;
    if (PetscRealPart(r->width)) {
      *val += PetscRealPart(r->width)* drander48();
    }
    if (PetscImaginaryPart(r->width)) {
      *val += PetscImaginaryPart(r->width)* drander48() * PETSC_i;
    }
  } else {
    *val = drander48() +  drander48()*PETSC_i;
  }
#else
  if (r->iset) *val = r->width * drander48() + r->low;
  else         *val = drander48();
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscRandomGetValueReal_Rander48"
PetscErrorCode  PetscRandomGetValueReal_Rander48(PetscRandom r, PetscReal *val)
{
  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  if (r->iset) *val = PetscRealPart(r->width)*drander48() + PetscRealPart(r->low);
  else         *val = drander48();
#else
  if (r->iset) *val = r->width * drander48() + r->low;
  else         *val = drander48();
#endif
  PetscFunctionReturn(0);
}

static struct _PetscRandomOps PetscRandomOps_Values = {
  /* 0 */
  PetscRandomSeed_Rander48,
  PetscRandomGetValue_Rander48,
  PetscRandomGetValueReal_Rander48,
  0,
  /* 5 */
  0
};

/*MC
   PETSCRANDER48 - simple portable reimplementation of basic Unix drand48() random number generator that should generate the
        exact same random numbers on any system.

   Options Database Keys:
. -random_type <rand,rand48,rander48,sprng>

  Level: beginner

.seealso: RandomCreate(), RandomSetType(), PETSCRAND, PETSCRAND48, PETSCRANDER48, PETSCSPRNG
M*/

#undef __FUNCT__
#define __FUNCT__ "PetscRandomCreate_Rander48"
PETSC_EXTERN PetscErrorCode PetscRandomCreate_Rander48(PetscRandom r)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemcpy(r->ops, &PetscRandomOps_Values, sizeof(PetscRandomOps_Values));CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) r, PETSCRANDER48);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
