
#if !defined(__ESMFGRID_H)
#define __ESMFGRID_H

#include "esmf.h"
#include "petscda.h"

typedef struct _p_ESMFGrid *ESMFGrid;
struct _p_ESMFGrid {
  struct _p_ESMFBase base;
  int  dimension;
};

typedef struct _p_ESMFGridStructuredRectangular *ESMFGridStructuredRectangular;
struct _p_ESMFGridStructuredRectangular {
  struct _p_ESMFGrid grid;
  int                nc,sw;
  int                localrange[2][2];
  int                ghostrange[2][2];
  int                globalsizes[2];
  int                localsizes[2];
  int                ghostsizes[2];
  DA                 da;
};

typedef struct _p_ESMFGridUnStructuredQuads *ESMFGridUnStructuredQuads;
struct _p_ESMFGridUnStructuredQuads {
  struct _p_ESMFGrid grid;
  int                nlocalcells,nghostcells;
  /*  F90Array2d         cells;
      F90Array2d         vertices;*/
};

#endif
