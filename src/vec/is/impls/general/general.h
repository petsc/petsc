/* "$Id: general.h,v 1.3 1998/12/03 03:56:18 bsmith Exp bsmith $"; */

#if !defined(__GENERAL_H)
#define __GENERAL_H

/*
    Defines the data structure used for the general index set
*/
#include "src/vec/is/isimpl.h"
#include "sys.h"

typedef struct {
  int n;         /* number of indices */ 
  int sorted;    /* indicates the indices are sorted */ 
  int *idx;
} IS_General;

#endif
