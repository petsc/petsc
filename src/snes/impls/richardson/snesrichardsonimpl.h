/* 
   Private context for Richardson iteration
*/

#ifndef __SNES_RICHARDSON_H
#define __SNES_RICHARDSON_H
#include <private/snesimpl.h>

typedef struct {
  SNESLineSearchType type; 
  PetscErrorCode     (*LineSearch)(SNES,void*,Vec,Vec,Vec,PetscReal,PetscReal,Vec,Vec,PetscReal*,PetscReal*,PetscBool *);
  /* Line Search Parameters */
  PetscReal          damping;		                                             
  PetscReal          maxstep;                                                        /* maximum step size */
  PetscReal          steptol;                                                        /* step convergence tolerance */
  PetscErrorCode     (*precheckstep)(SNES,Vec,Vec,void*,PetscBool *);                  /* step-checking routine (optional) */
  void               *precheck;                                                       /* user-defined step-checking context (optional) */
  PetscErrorCode     (*postcheckstep)(SNES,Vec,Vec,Vec,void*,PetscBool *,PetscBool *); /* step-checking routine (optional) */
  void               *postcheck;                                                      /* user-defined step-checking context (optional) */
  void               *lsP;                                                            /* user-defined line-search context (optional) */
  PetscViewer        monitor;
} SNES_Richardson;

#endif

