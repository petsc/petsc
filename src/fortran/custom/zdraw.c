#ifdef HAVE_64BITS
extern void *MPIR_ToPointer(int);
extern int MPIR_FromPointer(void*);
extern void MPIR_RmPointer(int);
#else
#define MPIR_ToPointer(a) (a)
#define MPIR_FromPointer(a) (int)(a)
#define MPIR_RmPointer(a)
#endif

#include "sles.h"

#ifdef FORTRANCAPS
#define slesdestroy_ SLESDESTROY
#define slescreate_  SLESCREATE
#elif !defined(FORTRANUNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define slesdestroy_ slesdestroy
#define slescreate_  slescreate
#endif

void slesdestroy_(SLES sles, int *__ierr )
{
  *__ierr = SLESDestroy((SLES)MPIR_ToPointer( *(int*)(sles) ));
  MPIR_RmPointer( *(int*)(sles) );
}

void slescreate_(MPI_Comm comm,SLES *outsles, int *__ierr )
{
  SLES sles;
  *__ierr = SLESCreate((MPI_Comm)MPIR_ToPointer( *(int*)(comm) ),&sles);
  *(int*) outsles = MPIR_FromPointer(sles);

}
