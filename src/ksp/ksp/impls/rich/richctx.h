/*
      Private data structure for Richardson Iteration 
*/

#if !defined(__RICH_H)
#define __RICH_H

typedef struct {
  PetscReal scale;               /* scaling on preconditioner */
} KSP_Richardson;

#endif
