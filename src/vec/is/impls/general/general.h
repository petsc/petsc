/* "$Id: general.h,v 1.5 2000/05/04 16:25:02 bsmith Exp balay $"; */

#if !defined(__GENERAL_H)
#define __GENERAL_H

/*
    Defines the data structure used for the general index set
*/
#include "src/vec/is/isimpl.h"
#include "petscsys.h"

typedef struct {
  int        n;         /* number of indices */ 
  PetscTruth sorted;    /* indicates the indices are sorted */ 
  int        *idx;
} IS_General;

#endif
