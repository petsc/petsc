#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: general.h,v 1.1 1998/06/03 16:07:08 bsmith Exp bsmith $";
#endif
/*
    Defines the data structure used for the general index set
*/
#include "src/is/isimpl.h"
#include "pinclude/pviewer.h"
#include "sys.h"

typedef struct {
  int n;         /* number of indices */ 
  int sorted;    /* indicates the indices are sorted */ 
  int *idx;
} IS_General;

