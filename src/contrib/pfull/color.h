
/* 
    Coloring structure and routines.
    Only temporarily defined here; should eventually move to PETSc proper
 */

#include "is.h"

typedef struct {
  int    nis;              /* number of index sets */
  IS     *isa;             /* array of index sets */
  Scalar *wscale, *scale;  /* arrays to hold scaling parameters */
} Coloring;

extern int MatCreateColoring(int,int,IS*,Coloring**);
extern int MatDestroyColoring(Coloring*);
extern int SNESSparseComputeJacobian(SNES,Vec,Mat*,Mat*,MatStructure*,void*);

