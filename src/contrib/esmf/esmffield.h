
#include "petscda.h"

typedef struct _p_ESMFGrid *ESMFGrid;
struct _p_ESMFGrid {
  char typename[64];
  int  dimension;
  int  fcomm;
  int  refcount;
};

typedef struct _p_ESMFGridStructuredRectangular *ESMFGridStructuredRectangular;
struct _p_ESMFGridStructuredRectangular {
  struct _p_ESMFGrid grid;
  int                localrange[2][2];
  int                ghostrange[2][2];
  int                globalsizes[2];
  int                localsizes[2];
  int                ghostsizes[2];
  DA                 privateda;
};

typedef struct _p_ESMFGridUnStructuredQuads *ESMFGridUnStructuredQuads;
struct _p_ESMFGridUnStructuredQuads {
  struct _p_ESMFGrid grid;
  int                nlocalcells,nghostcells;
  F90Array2d         cells;
  F90Array2d         vertices;
};

