/* "$Id: general.h,v 1.4 1999/01/31 16:03:26 bsmith Exp bsmith $"; */

#if !defined(__GENERAL_H)
#define __GENERAL_H

/*
    Defines the data structure used for the general index set
*/
#include "src/vec/is/isimpl.h"
#include "sys.h"

typedef struct {
  int        n;         /* number of indices */ 
  PetscTruth sorted;    /* indicates the indices are sorted */ 
  int        *idx;
} IS_General;

#endif
