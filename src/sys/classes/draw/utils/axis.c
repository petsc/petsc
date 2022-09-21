
#include <petsc/private/drawimpl.h> /*I   "petscdraw.h"  I*/

/*
   val is the label value.  sep is the separation to the next (or previous)
   label; this is useful in determining how many significant figures to
   keep.
 */
PetscErrorCode PetscADefLabel(PetscReal val, PetscReal sep, char **p)
{
  static char buf[40];

  PetscFunctionBegin;
  /* Find the string */
  if (PetscAbsReal(val) / sep < 1.e-4) {
    buf[0] = '0';
    buf[1] = 0;
  } else {
    sprintf(buf, "%0.1e", (double)val);
    PetscCall(PetscStripZerosPlus(buf));
    PetscCall(PetscStripe0(buf));
    PetscCall(PetscStripInitialZero(buf));
    PetscCall(PetscStripAllZeros(buf));
    PetscCall(PetscStripTrailingZeros(buf));
  }
  *p = buf;
  PetscFunctionReturn(0);
}

/* Finds "nice" locations for the ticks */
PetscErrorCode PetscADefTicks(PetscReal low, PetscReal high, int num, int *ntick, PetscReal *tickloc, int maxtick)
{
  int       i, power;
  PetscReal x = 0.0, base = 0.0, eps;

  PetscFunctionBegin;
  PetscCall(PetscAGetBase(low, high, num, &base, &power));
  PetscCall(PetscAGetNice(low, base, -1, &x));

  /* Values are of the form j * base */
  /* Find the starting value */
  if (x < low) x += base;

  i   = 0;
  eps = base / 10;
  while (i < maxtick && x <= high + eps) {
    tickloc[i++] = x;
    x += base;
  }
  *ntick         = i;
  tickloc[i - 1] = PetscMin(tickloc[i - 1], high);

  if (i < 2 && num < 10) PetscCall(PetscADefTicks(low, high, num + 1, ntick, tickloc, maxtick));
  PetscFunctionReturn(0);
}

#define EPS 1.e-6

PetscErrorCode PetscExp10(PetscReal d, PetscReal *result)
{
  PetscFunctionBegin;
  *result = PetscPowReal((PetscReal)10.0, d);
  PetscFunctionReturn(0);
}

PetscErrorCode PetscMod(PetscReal x, PetscReal y, PetscReal *result)
{
  int i;

  PetscFunctionBegin;
  if (y == 1) {
    *result = 0.0;
    PetscFunctionReturn(0);
  }
  i = ((int)x) / ((int)y);
  x = x - i * y;
  while (x > y) x -= y;
  *result = x;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscCopysign(PetscReal a, PetscReal b, PetscReal *result)
{
  PetscFunctionBegin;
  if (b >= 0) *result = a;
  else *result = -a;
  PetscFunctionReturn(0);
}

/*
    Given a value "in" and a "base", return a nice value.
    based on "sign", extend up (+1) or down (-1)
 */
PetscErrorCode PetscAGetNice(PetscReal in, PetscReal base, int sign, PetscReal *result)
{
  PetscReal etmp, s, s2, m;

  PetscFunctionBegin;
  PetscCall(PetscCopysign(0.5, (double)sign, &s));
  etmp = in / base + 0.5 + s;
  PetscCall(PetscCopysign(0.5, etmp, &s));
  PetscCall(PetscCopysign(EPS * etmp, (double)sign, &s2));
  etmp = etmp - 0.5 + s - s2;
  PetscCall(PetscMod(etmp, 1.0, &m));
  etmp    = base * (etmp - m);
  *result = etmp;
  PetscFunctionReturn(0);
}

PetscErrorCode PetscAGetBase(PetscReal vmin, PetscReal vmax, int num, PetscReal *Base, int *power)
{
  PetscReal        base, ftemp, e10;
  static PetscReal base_try[5] = {10.0, 5.0, 2.0, 1.0, 0.5};
  int              i;

  PetscFunctionBegin;
  /* labels of the form n * BASE */
  /* get an approximate value for BASE */
  base = (vmax - vmin) / (double)(num + 1);

  /* make it of form   m x 10^power,  m in [1.0, 10) */
  if (base <= 0.0) {
    base = PetscAbsReal(vmin);
    if (base < 1.0) base = 1.0;
  }
  ftemp = PetscLog10Real((1.0 + EPS) * base);
  if (ftemp < 0.0) ftemp -= 1.0;
  *power = (int)ftemp;
  PetscCall(PetscExp10((double)-*power, &e10));
  base = base * e10;
  if (base < 1.0) base = 1.0;
  /* now reduce it to one of 1, 2, or 5 */
  for (i = 1; i < 5; i++) {
    if (base >= base_try[i]) {
      PetscCall(PetscExp10((double)*power, &e10));
      base = base_try[i - 1] * e10;
      if (i == 1) *power = *power + 1;
      break;
    }
  }
  *Base = base;
  PetscFunctionReturn(0);
}
