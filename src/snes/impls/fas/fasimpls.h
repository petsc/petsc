#ifndef _SNES_FASIMPLS
#define _SNES_FASIMPLS

#include <private/snesimpl.h>

typedef struct {

  /* flags for knowing the global place of this FAS object */
  PetscInt       level;                        /* level = 0 coarsest level */
  PetscInt       levels;                       /* if level + 1 = levels; we're the last turtle */


  /* smoothing objects */
  SNES           presmooth;                    /* the SNES for presmoothing */
  SNES           postsmooth;                   /* the SNES for postsmoothing */

  /* coarse grid correction objects */
  SNES           next;                         /* the SNES instance for the next level in the hierarchy */
  Mat            interpolate;                  /* interpolation */
  Mat            restrct;                      /* restriction operator */
  Vec            rscale;                       /* the pointwise scaling of the restriction operator */

  /* method specific */
  PetscInt cycles;

} SNES_FAS;

#endif
