#if !defined(_SNES_FASIMPLS)
#define _SNES_FASIMPLS

#include <petsc/private/snesimpl.h>
#include <petsc/private/linesearchimpl.h>
#include <petscdm.h>

typedef struct {

  /* flags for knowing the global place of this FAS object */
  PetscInt level;                              /* level = 0 coarsest level */
  PetscInt levels;                             /* if level + 1 = levels; we're the last turtle */

  /* smoothing objects */
  SNES smoothu;                                /* the SNES for presmoothing */
  SNES smoothd;                                /* the SNES for postsmoothing */

  /* coarse grid correction objects */
  SNES next;                                   /* the SNES instance for the next coarser level in the hierarchy */
  SNES fine;                                   /* the finest SNES instance; used as a reference for prefixes */
  SNES previous;                               /* the SNES instance for the next finer level in the hierarchy */
  Mat  interpolate;                            /* interpolation */
  Mat  inject;                                 /* injection operator (unscaled) */
  Mat  restrct;                                /* restriction operator */
  Vec  rscale;                                 /* the pointwise scaling of the restriction operator */

  /* method parameters */
  PetscInt    n_cycles;                        /* number of cycles on this level */
  SNESFASType fastype;                         /* FAS type */
  PetscInt    max_up_it;                       /* number of pre-smooths */
  PetscInt    max_down_it;                     /* number of post-smooth cycles */
  PetscBool   usedmfornumberoflevels;          /* uses a DM to generate a number of the levels */
  PetscBool   full_downsweep;                  /* smooth on the initial full downsweep */
  PetscBool   full_total;                      /* use total residual restriction and total solution interpolation on the initial downsweep and upsweep */
  PetscBool   continuation;                    /* sets the setup to default to continuation */
  PetscInt    full_stage;                      /* stage of the full cycle -- 0 is the upswing, 1 is the downsweep and final V-cycle */

  /* Galerkin FAS state */
  PetscBool galerkin;                          /* use Galerkin formation of the coarse problem */
  Vec       Xg;                                /* Galerkin solution projection */
  Vec       Fg;                                /* Galerkin function projection */

  /* if logging times for each level */
  PetscLogEvent eventsmoothsetup;              /* level setup */
  PetscLogEvent eventsmoothsolve;              /* level smoother solves */
  PetscLogEvent eventresidual;                 /* level residual evaluation */
  PetscLogEvent eventinterprestrict;           /* level interpolation and restriction */
} SNES_FAS;

PETSC_INTERN PetscErrorCode SNESFASCycleCreateSmoother_Private(SNES,SNES*);

#endif
