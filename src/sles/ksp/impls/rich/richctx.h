/*$Id: richctx.h,v 1.4 1999/11/24 21:54:54 bsmith Exp bsmith $*/
/*
      Private data structure for Richardson Iteration 
*/

#if !defined(__RICH_H)
#define __RICH_H

typedef struct {
  PetscReal scale;               /* scaling on preconditioner */
} KSP_Richardson;

#endif
