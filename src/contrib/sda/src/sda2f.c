/* sda2.c */
/* Fortran interface file */

/*
 * This file was generated automatically by bfort from the C source
 * file.  
 */

#ifdef HAVE_64BITS
extern void *MPIR_ToPointer();
extern int MPIR_FromPointer();
extern void MPIR_RmPointer();
#else
#define MPIR_ToPointer(a) (a)
#define MPIR_FromPointer(a) (int)(a)
#define MPIR_RmPointer(a)
#endif

#include "da.h"
#ifdef HAVE_FORTRAN_CAPS
#define sdadestroy_ SDADESTROY
#elif !defined(HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define sdadestroy_ sdadestroy
#endif
#ifdef HAVE_FORTRAN_CAPS
#define sdalocaltolocalbegin_ SDALOCALTOLOCALBEGIN
#elif !defined(HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define sdalocaltolocalbegin_ sdalocaltolocalbegin
#endif
#ifdef HAVE_FORTRAN_CAPS
#define sdalocaltolocalend_ SDALOCALTOLOCALEND
#elif !defined(HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define sdalocaltolocalend_ sdalocaltolocalend
#endif


/* Definitions of Fortran Wrapper routines */
#if defined(__cplusplus)
extern "C" {
#endif
void sdadestroy_(SDA *sda, int *__ierr ){
*__ierr = SDADestroy(*sda);
}
void sdalocaltolocalbegin_(SDA *sda,Scalar *g,InsertMode *mode,Scalar *l, int *__ierr ){
*__ierr = SDALocalToLocalBegin(*sda,g,*mode,l);
}
void sdalocaltolocalend_(SDA *sda,Scalar *g,InsertMode *mode,Scalar *l, int *__ierr ){
*__ierr = SDALocalToLocalEnd(*sda,g,*mode,l);
}
#if defined(__cplusplus)
}
#endif
