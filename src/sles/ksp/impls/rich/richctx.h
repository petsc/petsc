/*$Id: rich.c,v 1.85 1999/11/05 14:46:46 bsmith Exp bsmith $*/
/*
      Private data structure for Richardson Iteration 
*/

#if !defined(__RICH_H)
#define __RICH_H

typedef struct {
  double scale;               /* scaling on preconditioner */
} KSP_Richardson;

#endif
