/* "$Id: general.h,v 1.2 1998/06/11 19:54:35 bsmith Exp bsmith $"; */

#if !defined(__GENERAL_H)
#define __GENERAL_H

/*
    Defines the data structure used for the general index set
*/
#include "src/is/isimpl.h"
#include "sys.h"

typedef struct {
  int n;         /* number of indices */ 
  int sorted;    /* indicates the indices are sorted */ 
  int *idx;
} IS_General;

#endif
