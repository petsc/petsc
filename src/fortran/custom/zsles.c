#ifdef HAVE_64BITS
extern void *MPIR_ToPointer(int);
extern int MPIR_FromPointer(void*);
extern void MPIR_RmPointer(int);
#else
#define MPIR_ToPointer(a) (a)
#define MPIR_FromPointer(a) (int)(a)
#define MPIR_RmPointer(a)
#endif

#include "mat.h"
#ifdef FORTRANCAPS
#define matreorderingregisterall_ MATREORDERINGREGISTERALL
#define matdestroy_               MATDESTROY
#elif !defined(FORTRANUNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define matreorderingregisterall_ matreorderingregisterall
#define matdestroy_               matdestroy
#endif

void matdestroy_(Mat mat, int *__ierr )
{
  *__ierr = MatDestroy((Mat)MPIR_ToPointer( *(int*)(mat) ));
   MPIR_RmPointer(*(int*)(mat)); 
}

void matreorderingregisterall_(int *MPIR_ierr)
{
  *__ierr = MatReorderingRegisterAll();
}
