#ifndef lint
static char vcid[] = "$Id: zoptions.c,v 1.1 1995/08/21 19:56:20 bsmith Exp bsmith $";
#endif

#include "zpetsc.h"
#include "vec.h"
#ifdef FORTRANCAPS
#define veccreatesequential_  VECCREATESEQUENTIAL
#define veccreate_            VECCREATE
#define vecduplicate_         VECDUPLICATE
#define veccreatempi_         VECCREATEMPI
#define vecscatterctxcreate_  VECSCATTERCTXCREATE
#define vecscatterctxcopy_    VECSCATTERCTXCOPY
#define vecdestroy_           VECDESTROY
#define vecscatterctxdestroy_ VECSCATTERCTXDESTROY
#define vecrestorearray_      VECRESTOREARRAY
#define vecgetarray_          VECGETARRAY
#elif !defined(FORTRANUNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define veccreatesequential_  veccreatesequential
#define veccreate_            veccreate
#define vecduplicate_         vecduplicate
#define veccreatempi_         veccreatempi
#define vecscatterctxcreate_  vecscatterctxcreate
#define vecscatterctxcopy_    vecscatterctxcopy
#define vecdestroy_           vecdestroy
#define vecscatterctxdestroy_ vecscatterctxdestroy
#define vecrestorearray_      vecrestorearray
#define vecgetarray_          vecgetarray
#endif

void vecrestorearray_(Vec x,Scalar *a,int *__ierr)
{
  Vec    xin = (Vec)MPIR_ToPointer( *(int*)(x) );
  Scalar *lx;
  int    n,i;

  *__ierr = VecGetArray(xin,&lx); if (*__ierr) return;
  *__ierr = VecGetLocalSize(xin,&n); if (*__ierr) return;
  for ( i=0; i<n; i++ ) {
    lx[i] = a[i];
  }
  *__ierr = VecRestoreArray(xin,&lx);
}

void vecgetarray_(Vec x,Scalar *a,int *__ierr)
{
  Vec    xin = (Vec)MPIR_ToPointer( *(int*)(x) );
  Scalar *lx;
  int    n,i;

  *__ierr = VecGetArray(xin,&lx); if (*__ierr) return;
  *__ierr = VecGetLocalSize(xin,&n); if (*__ierr) return;
  for ( i=0; i<n; i++ ) {
    a[i] = lx[i];
  }
  *__ierr = VecRestoreArray(xin,&lx);
}
   

void vecscatterctxdestroy_(VecScatterCtx ctx, int *__ierr ){
*__ierr = VecScatterCtxDestroy(
	(VecScatterCtx)MPIR_ToPointer( *(int*)(ctx) ));
   MPIR_RmPointer(*(int*)(ctx)); 
}

void vecdestroy_(Vec v, int *__ierr ){
*__ierr = VecDestroy(
	(Vec)MPIR_ToPointer( *(int*)(v) ));
   MPIR_RmPointer(*(int*)(v)); 
}

void vecscatterctxcreate_(Vec xin,IS ix,Vec yin,IS iy,VecScatterCtx *newctx, int *__ierr ){
  VecScatterCtx lV;
*__ierr = VecScatterCtxCreate(
	(Vec)MPIR_ToPointer( *(int*)(xin) ),
	(IS)MPIR_ToPointer( *(int*)(ix) ),
	(Vec)MPIR_ToPointer( *(int*)(yin) ),
	(IS)MPIR_ToPointer( *(int*)(iy) ),&lV);
  *(int*) newctx = MPIR_FromPointer(lV);
}
void vecscatterctxcopy_(VecScatterCtx sctx,VecScatterCtx *ctx, int *__ierr ){
  VecScatterCtx lV;
*__ierr = VecScatterCtxCopy(
	(VecScatterCtx)MPIR_ToPointer( *(int*)(sctx) ),&lV);
   *(int*) ctx = MPIR_FromPointer(lV); 
}


void veccreatempi_(MPI_Comm comm,int *n,int *N,Vec *vv, int *__ierr )
{
  Vec lV;
  *__ierr = VecCreateMPI((MPI_Comm)MPIR_ToPointer( *(int*)(comm) ),*n,*N,&lV);
  *(int*)vv = MPIR_FromPointer(lV);
}
void veccreatesequential_(MPI_Comm comm,int *n,Vec *V, int *__ierr )
{
  Vec lV;
  *__ierr = VecCreateSequential(
	(MPI_Comm)MPIR_ToPointer( *(int*)(comm) ),*n,&lV);
  *(int*)V = MPIR_FromPointer(lV);
}

void veccreate_(MPI_Comm comm,int *n,Vec *V, int *__ierr ){
  Vec lV;
  *__ierr = VecCreate(
	(MPI_Comm)MPIR_ToPointer( *(int*)(comm) ),*n,&lV);
  *(int*)V = MPIR_FromPointer(lV);
}

void vecduplicate_(Vec v,Vec *newv, int *__ierr )
{
  Vec lV;
  *__ierr = VecDuplicate((Vec)MPIR_ToPointer( *(int*)(v) ),&lV);
  *(int*)newv = MPIR_FromPointer(lV);
}
