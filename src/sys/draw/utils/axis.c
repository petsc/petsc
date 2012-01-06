
#include <../src/sys/draw/utils/axisimpl.h>

#undef __FUNCT__  
#define __FUNCT__ "PetscRint" 
static PetscErrorCode PetscRint(PetscReal x,PetscReal *result)
{
  PetscFunctionBegin;
  if (x > 0) *result = floor(x + 0.5);
  else       *result = floor(x - 0.5);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawAxisSetLimits" 
/*@
    PetscDrawAxisSetLimits -  Sets the limits (in user coords) of the axis
    
    Not Collective (ignored on all processors except processor 0 of PetscDrawAxis)

    Input Parameters:
+   axis - the axis
.   xmin,xmax - limits in x
-   ymin,ymax - limits in y

    Level: advanced

.seealso:  PetscDrawAxisSetHoldLimits()

@*/
PetscErrorCode  PetscDrawAxisSetLimits(PetscDrawAxis axis,PetscReal xmin,PetscReal xmax,PetscReal ymin,PetscReal ymax)
{
  PetscFunctionBegin;
  if (!axis) PetscFunctionReturn(0);
  if (axis->hold) PetscFunctionReturn(0);
  axis->xlow = xmin;
  axis->xhigh= xmax;
  axis->ylow = ymin;
  axis->yhigh= ymax;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscADefLabel" 
/*
   val is the label value.  sep is the separation to the next (or previous)
   label; this is useful in determining how many significant figures to   
   keep.
 */
PetscErrorCode PetscADefLabel(PetscReal val,PetscReal sep,char **p)
{
  static char    buf[40];
  char           fmat[10];
  PetscErrorCode ierr;
  int            w,d;
  PetscReal      rval;

  PetscFunctionBegin;
  /* Find the string */
  if (PetscAbsReal(val)/sep <  1.e-6) {
    buf[0] = '0'; buf[1] = 0;
  } else if (PetscAbsReal(val) < 1.0e6 && PetscAbsReal(val) > 1.e-4) {
    /* Compute the number of digits */
    w = 0;
    d = 0;
    if (sep > 0.0) {
	d = (int)ceil(- log10 (sep));
	if (d < 0) d = 0;
	if (PetscAbsReal(val) < 1.0e-6*sep) {
	    /* This is the case where we are near zero and less than a small
	       fraction of the sep.  In this case, we use 0 as the value */
	    val = 0.0;
	    w   = d;
        }
	else if (val == 0.0) w   = d;
	else w = (int)(ceil(log10(PetscAbsReal(val))) + d);
	if (w < 1)   w ++;
	if (val < 0) w ++;
    }

    ierr = PetscRint(val,&rval);CHKERRQ(ierr);
    if (rval == val) {
	if (w > 0) sprintf(fmat,"%%%dd",w);
	else {ierr = PetscStrcpy(fmat,"%d");CHKERRQ(ierr);}
	sprintf(buf,fmat,(int)val);
        ierr = PetscStripInitialZero(buf);CHKERRQ(ierr);
        ierr = PetscStripAllZeros(buf);CHKERRQ(ierr);
        ierr = PetscStripTrailingZeros(buf);CHKERRQ(ierr);
    } else {
	/* The code used here is inappropriate for a val of 0, which
	   tends to print with an excessive numer of digits.  In this
	   case, we should look at the next/previous values and 
	   use those widths */
	if (w > 0) sprintf(fmat,"%%%d.%dlf",w + 1,d);
	else {ierr = PetscStrcpy(fmat,"%lf");CHKERRQ(ierr);}
	sprintf(buf,fmat,(double)val);
        ierr = PetscStripInitialZero(buf);CHKERRQ(ierr);
        ierr = PetscStripAllZeros(buf);CHKERRQ(ierr);
        ierr = PetscStripTrailingZeros(buf);CHKERRQ(ierr);
    }
  } else {
    ierr = PetscSNPrintf(buf,40,"%g",(double)val);
    /* remove the extraneous 0 before the e */
    ierr = PetscStripZeros(buf);CHKERRQ(ierr);
    ierr = PetscStripZerosPlus(buf);CHKERRQ(ierr);
    ierr = PetscStripInitialZero(buf);CHKERRQ(ierr);
    ierr = PetscStripAllZeros(buf);CHKERRQ(ierr);
    ierr = PetscStripTrailingZeros(buf);CHKERRQ(ierr);
  }
  *p =buf;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscADefTicks" 
/* Finds "nice" locations for the ticks */
PetscErrorCode PetscADefTicks(PetscReal low,PetscReal high,int num,int *ntick,PetscReal * tickloc,int  maxtick)
{
  PetscErrorCode ierr;
  int            i,power;
  PetscReal      x = 0.0,base=0.0;

  PetscFunctionBegin;
  /* patch if low == high */
  if (low == high) {
    low  -= .01;
    high += .01;
  }

  /*  if (PetscAbsReal(low-high) < 1.e-8) {
    low  -= .01;
    high += .01;
  } */

  ierr = PetscAGetBase(low,high,num,&base,&power);CHKERRQ(ierr);
  ierr = PetscAGetNice(low,base,-1,&x);CHKERRQ(ierr);

  /* Values are of the form j * base */
  /* Find the starting value */
  if (x < low) x += base;

  i = 0;
  while (i < maxtick && x <= high) {
    tickloc[i++] = x;
    x += base;
  }
  *ntick = i;

  if (i < 2 && num < 10) {
    ierr = PetscADefTicks(low,high,num+1,ntick,tickloc,maxtick);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#define EPS 1.e-6

#undef __FUNCT__  
#define __FUNCT__ "PetscExp10" 
PetscErrorCode PetscExp10(PetscReal d,PetscReal *result)
{
  PetscFunctionBegin;
  *result = pow((PetscReal)10.0,d);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMod" 
PetscErrorCode PetscMod(PetscReal x,PetscReal y,PetscReal *result)
{
  int     i;

  PetscFunctionBegin;
  if (y == 1) {
    *result = 0.0;
    PetscFunctionReturn(0);
  }
  i   = ((int)x) / ((int)y);
  x   = x - i * y;
  while (x > y) x -= y;
  *result = x;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscCopysign" 
PetscErrorCode PetscCopysign(PetscReal a,PetscReal b,PetscReal *result)
{
  PetscFunctionBegin;
  if (b >= 0) *result = a;
  else        *result = -a;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscAGetNice" 
/*
    Given a value "in" and a "base", return a nice value.
    based on "sign", extend up (+1) or down (-1)
 */
PetscErrorCode PetscAGetNice(PetscReal in,PetscReal base,int sign,PetscReal *result)
{
  PetscReal      etmp,s,s2,m;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr    = PetscCopysign (0.5,(double)sign,&s);CHKERRQ(ierr);
  etmp    = in / base + 0.5 + s;
  ierr    = PetscCopysign (0.5,etmp,&s);CHKERRQ(ierr);
  ierr    = PetscCopysign (EPS * etmp,(double)sign,&s2);CHKERRQ(ierr);
  etmp    = etmp - 0.5 + s - s2;
  ierr    = PetscMod(etmp,1.0,&m);CHKERRQ(ierr);
  etmp    = base * (etmp -  m);
  *result = etmp;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscAGetBase" 
PetscErrorCode PetscAGetBase(PetscReal vmin,PetscReal vmax,int num,PetscReal*Base,int*power)
{
  PetscReal        base,ftemp,e10;
  static PetscReal base_try[5] = {10.0,5.0,2.0,1.0,0.5};
  PetscErrorCode   ierr;
  int              i;

  PetscFunctionBegin;
  /* labels of the form n * BASE */
  /* get an approximate value for BASE */
  base    = (vmax - vmin) / (double)(num + 1);

  /* make it of form   m x 10^power,  m in [1.0, 10) */
  if (base <= 0.0) {
    base    = PetscAbsReal(vmin);
    if (base < 1.0) base = 1.0;
  }
  ftemp   = log10((1.0 + EPS) * base);
  if (ftemp < 0.0)  ftemp   -= 1.0;
  *power  = (int)ftemp;
  ierr = PetscExp10((double)- *power,&e10);CHKERRQ(ierr);
  base    = base * e10;
  if (base < 1.0) base    = 1.0;
  /* now reduce it to one of 1, 2, or 5 */
  for (i=1; i<5; i++) {
    if (base >= base_try[i]) {
      ierr = PetscExp10((double)*power,&e10);CHKERRQ(ierr);
      base = base_try[i-1] * e10;
      if (i == 1) *power    = *power + 1;
      break;
    }
  }
  *Base   = base;
  PetscFunctionReturn(0);
}






