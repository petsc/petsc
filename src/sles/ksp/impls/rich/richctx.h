/*
      Private context for Richardson Iteration 
*/

#if !defined(__RICH)
#define __RICH

typedef struct {
  double scale;               /* scaling on preconditioner */
} KSPRichardsonCntx;

#endif
