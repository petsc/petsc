/*$Id: richctx.h,v 1.5 2001/08/06 21:16:46 bsmith Exp $*/
/*
      Private data structure for Richardson Iteration 
*/

#if !defined(__RICH_H)
#define __RICH_H

typedef struct {
  PetscReal scale;               /* scaling on preconditioner */
} KSP_Richardson;

#endif
