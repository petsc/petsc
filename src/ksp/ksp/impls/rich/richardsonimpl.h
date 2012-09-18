/*
      Private data structure for Richardson Iteration
*/

#if !defined(__RICHARDSONIMPL_H)
#define __RICHARDSONIMPL_H

typedef struct {
  PetscReal  scale;               /* scaling on preconditioner */
  PetscBool  selfscale;           /* determine optimimal scaling each iteration to minimize 2-norm of resulting residual */
} KSP_Richardson;

#endif
