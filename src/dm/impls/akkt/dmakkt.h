#if !defined(_dmakktimpl_h)
#define _dmakktimpl_h

#include <petscdmakkt.h>
#define DMAKKT_DECOMPOSITION_NAME_LEN 1024
typedef struct {
  Mat       Aff;      /* Mat encoding the underlying fine geometry. */
  DM        dm;       /* Optional DM encapsulating the KKT system matrix and (optionally) its decomposition. */
  char      dname[DMAKKT_DECOMPOSITION_NAME_LEN+1];    /* Optional name of the decomposition defining the split into primal and dual variables. */
  char*     names[2]; /* Optional names of the decomposition parts. */
  DM        dmf[2];   /* DMs of the primal-dual split; dual subDM is optional. */
  IS        isf[2];   /* ISs of the primal-dual split; either is optional, but at least one must be set -- the other is then the complement. */
  PetscBool transposeP;/* Whether the primal prolongator needs to be transpose to be a prolongator (i.e., to map from coarse to fine). */
  DM        cdm;      /* If this DM has been coarsened, cache the result for use in DMCreateInterpolation() */
  Mat       Pfc;      /* Prolongator for the combined system. */
  PetscBool duplicate_mat;
  PetscBool detect_saddle_point;
} DM_AKKT;


#endif
