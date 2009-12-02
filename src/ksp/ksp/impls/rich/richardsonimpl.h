/*
      Private data structure for Richardson Iteration 
*/

#if !defined(__RICHARDSONIMPL_H)
#define __RICHARDSONIMPL_H

typedef struct {
  PetscReal scale;               /* scaling on preconditioner */
} KSP_Richardson;

#endif
