#ifndef lint
static char vcid[] = "$Id: zda.c,v 1.1 1995/08/27 00:35:57 bsmith Exp bsmith $";
#endif

#include "zpetsc.h"
#include "is.h"
#ifdef FORTRANCAPS
#define isdestroy_ ISDESTROY
#elif !defined(FORTRANUNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define isdestroy_ isdestroy
#endif

void isdestroy_(IS is, int *__ierr ){
  *__ierr = ISDestroy((IS)MPIR_ToPointer( *(int*)(is) ));
  MPIR_RmPointer(*(int*)(is) ));
}
