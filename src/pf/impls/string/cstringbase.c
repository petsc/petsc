/*$Id: cstring.c,v 1.1 2000/02/02 21:28:09 bsmith Exp bsmith $*/
#include "src/pf/pfimpl.h"            /*I "pf.h" I*/

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PFApply_String_1"
int PFApply_String_1(void *value,int n,Scalar *x,Scalar *y)
{
  int    i;

  PetscFunctionBegin;
  for (i=0; i<n; i++) {
    y[i] = 3.0;
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END
