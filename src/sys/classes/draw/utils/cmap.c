#include <petscsys.h>              /*I "petscsys.h" I*/
#include <petscdraw.h>

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

PetscErrorCode  PetscDrawUtilitySetGamma(PetscReal g)
{
  PetscFunctionBegin;
  Gamma = g;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE double PetscHlsHelper(double m1,double m2,double h)
{
  while (h > 1.0) h -= 1.0;
  while (h < 0.0) h += 1.0;
  if (h < 1/6.0) return m1 + (m2-m1)*h*6;
  if (h < 1/2.0) return m2;
  if (h < 2/3.0) return m1 + (m2-m1)*(2/3.0-h)*6;
  return m1;
}

PETSC_STATIC_INLINE void PetscHlsToRgb(double h,double l,double s,double *r,double *g,double *b)
{
  if (s > 0.0) {
    double m2 = l <= 0.5 ? l * (1.0+s) : l+s-(l*s);
    double m1 = 2*l - m2;
    *r = PetscHlsHelper(m1,m2,h+1/3.);
    *g = PetscHlsHelper(m1,m2,h);
    *b = PetscHlsHelper(m1,m2,h-1/3.);
  } else {
    /* ignore hue */
    *r = *g = *b = l;
  }
}

PETSC_STATIC_INLINE void PetscGammaCorrect(double *r,double *g,double *b)
{
  PetscReal igamma = 1/Gamma;
  *r = (double)PetscPowReal((PetscReal)*r,igamma);
  *g = (double)PetscPowReal((PetscReal)*g,igamma);
  *b = (double)PetscPowReal((PetscReal)*b,igamma);
}

static PetscErrorCode PetscDrawCmap_Hue(int mapsize, unsigned char R[],unsigned char G[],unsigned char B[])
{
  int    i;
  double maxhue = 212.0/360,lightness = 0.5,saturation = 1.0;

  PetscFunctionBegin;
  for (i=0; i<mapsize; i++) {
    double hue = maxhue*(double)i/(mapsize-1),r,g,b;
    PetscHlsToRgb(hue,lightness,saturation,&r,&g,&b);
    PetscGammaCorrect(&r,&g,&b);
    R[i] = (unsigned char)(255*PetscMin(r,1.0));
    G[i] = (unsigned char)(255*PetscMin(g,1.0));
    B[i] = (unsigned char)(255*PetscMin(b,1.0));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawCmap_Gray(int mapsize,unsigned char R[],unsigned char G[],unsigned char B[])
{
  int i;
  PetscFunctionBegin;
  for (i=0; i<mapsize; i++) R[i] = G[i] = B[i] = (unsigned char)((255.0*i)/(mapsize-1));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawCmap_Jet(int mapsize,unsigned char R[],unsigned char G[],unsigned char B[])
{
  int          i;
  const double knots[] =  {0, 1/8., 3/8., 5/8., 7/8., 1};

  PetscFunctionBegin;
  for (i=0; i<mapsize; i++) {
    double u = (double)i/(mapsize-1);
    double m, r=0, g=0, b=0; int k = 0;
    while (k < 4 && u > knots[k+1]) k++;
    m = (u-knots[k])/(knots[k+1]-knots[k]);
    switch(k) {
    case 0: r = 0;     g = 0;   b = (m+1)/2; break;
    case 1: r = 0;     g = m;   b = 1;       break;
    case 2: r = m;     g = 1;   b = 1-m;     break;
    case 3: r = 1;     g = 1-m; b = 0;       break;
    case 4: r = 1-m/2; g = 0;   b = 0;       break;
    }
    R[i] = (unsigned char)(255*PetscMin(r,1.0));
    G[i] = (unsigned char)(255*PetscMin(g,1.0));
    B[i] = (unsigned char)(255*PetscMin(b,1.0));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawCmap_Hot(int mapsize,unsigned char R[],unsigned char G[],unsigned char B[])
{
  int          i;
  const double knots[] =  {0, 3/8., 3/4., 1};

  PetscFunctionBegin;
  for (i=0; i<mapsize; i++) {
    double u = (double)i/(mapsize-1);
    double m, r=0, g=0, b=0; int k = 0;
    while (k < 2 && u > knots[k+1]) k++;
    m = (u-knots[k])/(knots[k+1]-knots[k]);
    switch(k) {
    case 0: r = m; g = 0; b = 0; break;
    case 1: r = 1; g = m; b = 0; break;
    case 2: r = 1; g = 1; b = m; break;
    }
    R[i] = (unsigned char)(255*PetscMin(r,1.0));
    G[i] = (unsigned char)(255*PetscMin(g,1.0));
    B[i] = (unsigned char)(255*PetscMin(b,1.0));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDrawCmap_Bone(int mapsize,unsigned char R[],unsigned char G[],unsigned char B[])
{
  int i;
  PetscFunctionBegin;
  (void)PetscDrawCmap_Hot(mapsize,R,G,B);
  for (i=0; i<mapsize; i++) {
    double u = (double)i/(mapsize-1);
    double r = (7*u + B[i]/255.0)/8;
    double g = (7*u + G[i]/255.0)/8;
    double b = (7*u + R[i]/255.0)/8;
    R[i] = (unsigned char)(255*PetscMin(r,1.0));
    G[i] = (unsigned char)(255*PetscMin(g,1.0));
    B[i] = (unsigned char)(255*PetscMin(b,1.0));
  }
  PetscFunctionReturn(0);
}

#include "../src/sys/classes/draw/utils/cmap/coolwarm.h"
#include "../src/sys/classes/draw/utils/cmap/parula.h"
#include "../src/sys/classes/draw/utils/cmap/viridis.h"
#include "../src/sys/classes/draw/utils/cmap/plasma.h"
#include "../src/sys/classes/draw/utils/cmap/inferno.h"
#include "../src/sys/classes/draw/utils/cmap/magma.h"

static struct {
  const char           *name;
  const unsigned char (*data)[3];
  PetscErrorCode      (*cmap)(int,unsigned char[],unsigned char[],unsigned char[]);
} PetscDrawCmapTable[] = {
  {"hue",      NULL, PetscDrawCmap_Hue },     /* varying hue with constant lightness and saturation */
  {"gray",     NULL, PetscDrawCmap_Gray},     /* black to white with shades of gray */
  {"bone",     NULL, PetscDrawCmap_Bone},     /* black to white with gray-blue shades */
  {"jet",      NULL, PetscDrawCmap_Jet },     /* rainbow-like colormap from NCSA, University of Illinois */
  {"hot",      NULL, PetscDrawCmap_Hot },     /* black-body radiation */
  {"coolwarm", PetscDrawCmap_coolwarm, NULL}, /* ParaView default (Cool To Warm with Diverging interpolation) */
  {"parula",   PetscDrawCmap_parula,   NULL}, /* MATLAB (default since R2014b) */
  {"viridis",  PetscDrawCmap_viridis,  NULL}, /* matplotlib 1.5 (default since 2.0) */
  {"plasma",   PetscDrawCmap_plasma,   NULL}, /* matplotlib 1.5 */
  {"inferno",  PetscDrawCmap_inferno,  NULL}, /* matplotlib 1.5 */
  {"magma",    PetscDrawCmap_magma,    NULL}, /* matplotlib 1.5 */
};

PetscErrorCode  PetscDrawUtilitySetCmap(const char colormap[],int mapsize,unsigned char R[],unsigned char G[],unsigned char B[])
{
  int             i,j;
  const char      *cmap_name_list[sizeof(PetscDrawCmapTable)/sizeof(PetscDrawCmapTable[0])];
  PetscInt        id = 0, count = (PetscInt)(sizeof(cmap_name_list)/sizeof(char*));
  PetscBool       reverse = PETSC_FALSE, brighten = PETSC_FALSE;
  PetscReal       beta = 0;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  for (i=0; i<count; i++) cmap_name_list[i] = PetscDrawCmapTable[i].name;
  if (colormap && colormap[0]) {
    PetscBool match = PETSC_FALSE;
    for (id=0; !match && id<count; id++) {ierr = PetscStrcasecmp(colormap,cmap_name_list[id],&match);CHKERRQ(ierr);}
    PetscAssertFalse(!match,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Colormap '%s' not found",colormap);
  }
  ierr = PetscOptionsGetEList(NULL,NULL,"-draw_cmap",cmap_name_list,count,&id,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetBool(NULL,NULL,"-draw_cmap_reverse",&reverse,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetReal(NULL,NULL,"-draw_cmap_brighten",&beta,&brighten);CHKERRQ(ierr);
  PetscAssertFalse(brighten && (beta <= (PetscReal)-1 || beta >= (PetscReal)+1),PETSC_COMM_SELF,PETSC_ERR_ARG_OUTOFRANGE,"brighten parameter %g must be in the range (-1,1)",(double)beta);

  if (PetscDrawCmapTable[id].cmap) {
    ierr = PetscDrawCmapTable[id].cmap(mapsize,R,G,B);CHKERRQ(ierr);
  } else {
    const unsigned char (*rgb)[3] = PetscDrawCmapTable[id].data;
    PetscAssertFalse(mapsize != 256-PETSC_DRAW_BASIC_COLORS,PETSC_COMM_SELF,PETSC_ERR_SUP,"Colormap '%s' with size %d not supported",cmap_name_list[id],mapsize);
    for (i=0; i<mapsize; i++) {R[i] = rgb[i][0]; G[i] = rgb[i][1]; B[i] = rgb[i][2];}
  }

  if (reverse) {
    i = 0; j = mapsize-1;
    while (i < j) {
#define SWAP(a,i,j) do { unsigned char t = a[i]; a[i] = a[j]; a[j] = t; } while (0)
      SWAP(R,i,j);
      SWAP(G,i,j);
      SWAP(B,i,j);
#undef SWAP
      i++; j--;
    }
  }

  if (brighten) {
    PetscReal gamma = (beta > 0.0) ? (1 - beta) : (1 / (1 + beta));
    for (i=0; i<mapsize; i++) {
      PetscReal r = PetscPowReal((PetscReal)R[i]/255,gamma);
      PetscReal g = PetscPowReal((PetscReal)G[i]/255,gamma);
      PetscReal b = PetscPowReal((PetscReal)B[i]/255,gamma);
      R[i] = (unsigned char)(255*PetscMin(r,(PetscReal)1.0));
      G[i] = (unsigned char)(255*PetscMin(g,(PetscReal)1.0));
      B[i] = (unsigned char)(255*PetscMin(b,(PetscReal)1.0));
    }
  }
  PetscFunctionReturn(0);
}
