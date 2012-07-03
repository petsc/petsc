
#include <petscsys.h>              /*I "petscsys.h" I*/

/*
    Set up a color map, using uniform separation in hue space.
    Map entries are Red, Green, Blue.
    Values are "gamma" corrected.
 */

/*  
   Gamma is a monitor dependent value.  The value here is an 
   approximate that gives somewhat better results than Gamma = 1.
 */
static PetscReal Gamma = 2.0;

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawUtilitySetGamma"  
PetscErrorCode  PetscDrawUtilitySetGamma(PetscReal g)
{
  PetscFunctionBegin;
  Gamma = g;
  PetscFunctionReturn(0);
}


/*
 * This algorithm is from Foley and van Dam, page 616
 * given
 *   (0:359, 0:100, 0:100).
 *      h       l      s
 * set
 *   (0:255, 0:255, 0:255)
 *      r       g      b
 */
#undef __FUNCT__  
#define __FUNCT__ "PetscDrawUtilityHlsHelper" 
static PetscErrorCode PetscDrawUtilityHlsHelper(int h,int n1,int n2)
{
  PetscFunctionBegin;
  while (h > 360) h = h - 360;
  while (h < 0)   h = h + 360;
  if (h < 60)  PetscFunctionReturn(n1 + (n2-n1)*h/60);
  if (h < 180) PetscFunctionReturn(n2);
  if (h < 240) PetscFunctionReturn(n1 + (n2-n1)*(240-h)/60);
  PetscFunctionReturn(n1);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawUtilityHlsToRgb" 
static PetscErrorCode PetscDrawUtilityHlsToRgb(int h,int l,int s,unsigned char *r,unsigned char *g,unsigned char *b)
{
  int m1,m2;         /* in 0 to 100 */

  PetscFunctionBegin;
  if (l <= 50) m2 = l * (100 + s) / 100 ;           /* not sure of "/100" */
  else         m2 = l + s - l*s/100;

  m1  = 2*l - m2;
  if (!s) {
    /* ignore h */
    *r  = 255 * l / 100;
    *g  = 255 * l / 100;
    *b  = 255 * l / 100;
  } else {
    *r  = (255 * PetscDrawUtilityHlsHelper(h+120,m1,m2)) / 100;
    *g  = (255 * PetscDrawUtilityHlsHelper(h,m1,m2))     / 100;
    *b  = (255 * PetscDrawUtilityHlsHelper(h-120,m1,m2)) / 100;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDrawUtilitySetCmapHue" 
PetscErrorCode  PetscDrawUtilitySetCmapHue(unsigned char *red,unsigned char *green,unsigned char * blue,int mapsize)
{
  PetscErrorCode ierr;
  int        i,hue,lightness,saturation;
  PetscReal  igamma = 1.0 / Gamma;

  PetscFunctionBegin;
  red[0]      = 0;
  green[0]    = 0;
  blue[0]     = 0;
  hue         = 0;        /* in 0:359 */
  lightness   = 50;       /* in 0:100 */
  saturation  = 100;      /* in 0:100 */
  for (i = 0; i < mapsize; i++) {
    ierr     = PetscDrawUtilityHlsToRgb(hue,lightness,saturation,red + i,green + i,blue + i);CHKERRQ(ierr);
    red[i]   = (int)floor(255.999 * pow(((PetscReal) red[i])/(PetscReal)255.0,igamma));
    blue[i]  = (int)floor(255.999 * pow(((PetscReal)blue[i])/(PetscReal)255.0,igamma));
    green[i] = (int)floor(255.999 * pow(((PetscReal)green[i])/(PetscReal)255.0,igamma));
    hue     += (359/(mapsize-2));
  }
  PetscFunctionReturn(0);
}
