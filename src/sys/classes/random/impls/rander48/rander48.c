#include <../src/sys/classes/random/randomimpl.h>

typedef struct {
  unsigned short seed[3];
  unsigned short mult[3];
  unsigned short add;
} PetscRandom_Rander48;

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
  PetscRandom_Rander48 *r48 = (PetscRandom_Rander48*)r->data;

  PetscFunctionBegin;
  r48->seed[0] = RANDER48_SEED_0;
  r48->seed[1] = (unsigned short) r->seed;
  r48->seed[2] = (unsigned short) (r->seed >> 16);
  r48->mult[0] = RANDER48_MULT_0;
  r48->mult[1] = RANDER48_MULT_1;
  r48->mult[2] = RANDER48_MULT_2;
  r48->add     = RANDER48_ADD;
  srander48(r->seed);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscRandomGetValue_Rander48"
PetscErrorCode  PetscRandomGetValue_Rander48(PetscRandom r, PetscScalar *val)
{
  PetscRandom_Rander48 *r48 = (PetscRandom_Rander48*)r->data;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  if (r->iset) {
    *val = PetscRealPart(r->low) + PetscImaginaryPart(r->low) * PETSC_i;
    if (PetscRealPart(r->width)) {
      *val += PetscRealPart(r->width)* erander48(r48->seed);
    }
    if (PetscImaginaryPart(r->width)) {
      *val += PetscImaginaryPart(r->width)* erander48(r48->seed) * PETSC_i;
    }
  } else {
    *val = erander48(r48->seed) +  erander48(r48->seed)*PETSC_i;
  }
#else
  if (r->iset) *val = r->width * erander48(r48->seed) + r->low;
  else         *val = erander48(r48->seed);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscRandomGetValueReal_Rander48"
PetscErrorCode  PetscRandomGetValueReal_Rander48(PetscRandom r, PetscReal *val)
{
  PetscRandom_Rander48 *r48 = (PetscRandom_Rander48*)r->data;

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  if (r->iset) *val = PetscRealPart(r->width)*erander48(r48->seed) + PetscRealPart(r->low);
  else         *val = erander48(r48->seed);
#else
  if (r->iset) *val = r->width * erander48(r48->seed) + r->low;
  else         *val = erander48(r48->seed);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscRandomDestroy_Rander48"
PetscErrorCode  PetscRandomDestroy_Rander48(PetscRandom r)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(r->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static struct _PetscRandomOps PetscRandomOps_Values = {
  /* 0 */
  PetscRandomSeed_Rander48,
  PetscRandomGetValue_Rander48,
  PetscRandomGetValueReal_Rander48,
  PetscRandomDestroy_Rander48,
  /* 5 */
  0
};

/*MC
   PETSCRANDER48 - simple portable reimplementation of basic Unix drand48() random number generator that should generate the
        exact same random numbers on any system.

   Options Database Keys:
. -random_type <rand,rand48,rander48,sprng>

  Notes: This is the default random number generate provided by PetscRandomCreate() if you do not set a particular implementation.

  Level: beginner

.seealso: RandomCreate(), RandomSetType(), PETSCRAND, PETSCRAND48, PETSCRANDER48, PETSCSPRNG
M*/

#undef __FUNCT__
#define __FUNCT__ "PetscRandomCreate_Rander48"
PETSC_EXTERN PetscErrorCode PetscRandomCreate_Rander48(PetscRandom r)
{
  PetscErrorCode       ierr;
  PetscRandom_Rander48 *r48;

  PetscFunctionBegin;
  ierr = PetscNewLog(r,&r48);CHKERRQ(ierr);
  r->data = r48;
  ierr = PetscMemcpy(r->ops, &PetscRandomOps_Values, sizeof(PetscRandomOps_Values));CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject) r, PETSCRANDER48);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
