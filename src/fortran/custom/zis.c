#ifndef lint
static char vcid[] = "$Id: zis.c,v 1.1 1995/09/04 04:38:47 bsmith Exp bsmith $";
#endif

#include "zpetsc.h"
#include "is.h"
#ifdef FORTRANCAPS
#define isdestroy_ ISDESTROY
#define iscreatestrideseq_ ISCREATESTRIDESEQ
#define iscreateseq_ ISCREATESEQ
#define isgetindices        ISGETINDICES
#define isrestoreindices        ISRESTOREINDICES
#elif !defined(FORTRANUNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define isdestroy_ isdestroy
#define iscreatestrideseq_ iscreatestrideseq
#define iscreateseq_ iscreateseq
#define isgetindices        isgetindices
#define isrestoreindices    isrestoreindices
#endif

void isgetindices_(IS x,int *a,int *__ierr)
{
  IS    xin = (IS)MPIR_ToPointer( *(int*)(x) );
  int   *lx;
  int   n,i;

  *__ierr = ISGetIndices(xin,&lx); if (*__ierr) return;
  *__ierr = ISGetLocalSize(xin,&n); if (*__ierr) return;
  for ( i=0; i<n; i++ ) {
    a[i] = lx[i];
  }
  *__ierr = ISRestoreIndices(xin,&lx);
}

void isrestoreindices_(IS x,int *a,int *__ierr)
{
  return;
}


void iscreateseq_(MPI_Comm comm,int *n,int *idx,IS *is, int *__ierr ){
  IS ii;
  *__ierr = ISCreateSeq(
	(MPI_Comm)MPIR_ToPointer( *(int*)(comm) ),*n,idx,&ii);
  *(int*) is = MPIR_FromPointer(ii);
}

void iscreatestrideseq_(MPI_Comm comm,int *n,int *first,int *step,
                               IS *is, int *__ierr ){
  IS ii;
  *__ierr = ISCreateStrideSeq(
	(MPI_Comm)MPIR_ToPointer( *(int*)(comm) ),*n,*first,*step,&ii);
  *(int*) is = MPIR_FromPointer(ii);
}

void isdestroy_(IS is, int *__ierr ){
  *__ierr = ISDestroy((IS)MPIR_ToPointer( *(int*)(is) ));
  MPIR_RmPointer(*(int*)(is) );
}
