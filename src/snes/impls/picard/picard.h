/* 
   Private context for Picard iteration
*/

#ifndef __SNES_PICARD_H
#define __SNES_PICARD_H
#include "private/snesimpl.h"

typedef struct {
  int type; /* Type of Picard iteration */
  /* Line Search */
  PetscErrorCode (*LineSearch)(SNES,void*,Vec,Vec,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal*,PetscReal*,PetscTruth*);
  /* Line Search Parameters */
  PetscReal        alpha;		                                                   /* used to determine sufficient reduction */
  PetscReal        maxstep;                                                        /* maximum step size */
  PetscReal        steptol;                                                        /* step convergence tolerance */
  PetscErrorCode (*precheckstep)(SNES,Vec,Vec,void*,PetscTruth*);                  /* step-checking routine (optional) */
  void            *precheck;                                                       /* user-defined step-checking context (optional) */
  PetscErrorCode (*postcheckstep)(SNES,Vec,Vec,Vec,void*,PetscTruth*,PetscTruth*); /* step-checking routine (optional) */
  void            *postcheck;                                                      /* user-defined step-checking context (optional) */
  void            *lsP;                                                            /* user-defined line-search context (optional) */
} SNES_Picard;

#endif

