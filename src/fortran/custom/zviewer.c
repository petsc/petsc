
#ifndef lint
static char vcid[] = "$Id: zoptions.c,v 1.1 1995/08/21 19:56:20 bsmith Exp bsmith $";
#endif

#include "zpetsc.h"
#include "petsc.h"

#ifdef FORTRANCAPS
#define viewerdestroy_ VIEWERDESTROY
#elif !defined(FORTRANUNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define viewerdestroy_ viewerdestroy
#endif

void viewerdestroy_(Viewer v, int *__ierr ){
  *__ierr = ViewerDestroy((Viewer)MPIR_ToPointer( *(int*)(v) ));
  MPIR_RmPointer(*(int*)(v) );
}
