#ifndef lint
static char vcid[] = "$Id: zvec.c,v 1.5 1995/11/27 19:02:45 bsmith Exp curfman $";
#endif

#include "zpetsc.h"
#include "vec.h"
#ifdef FORTRANCAPS
#define veccreateseq_         VECCREATESEQ
#define veccreate_            VECCREATE
#define vecduplicate_         VECDUPLICATE
#define veccreatempi_         VECCREATEMPI
#define VecScattercreate_  VecScatterCREATE
#define VecScattercopy_    VecScatterCOPY
#define vecdestroy_           VECDESTROY
#define VecScatterdestroy_ VecScatterDESTROY
#define vecrestorearray_      VECRESTOREARRAY
#define vecgetarray_          VECGETARRAY
#define vecload_              VECLOAD
#elif !defined(FORTRANUNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define veccreateseq_         veccreateseq
#define veccreate_            veccreate
#define vecduplicate_         vecduplicate
#define veccreatempi_         veccreatempi
#define VecScattercreate_  VecScattercreate
#define VecScattercopy_    VecScattercopy
#define vecdestroy_           vecdestroy
#define VecScatterdestroy_ VecScatterdestroy
#define vecrestorearray_      vecrestorearray
#define vecgetarray_          vecgetarray
#define vecload_              vecload
#endif

void vecload_(Viewer bview,Vec *newvec, int *__ierr )
{ 
  Vec vv;
  *__ierr = VecLoad((Viewer)MPIR_ToPointer( *(int*)(bview) ),&vv);
  *(int *) newvec = MPIR_FromPointer(vv);
}

/* 
     When you fix these to pass back integers rather then 
  copy the array make sure you fix vec/examples/ex21.F and 
  snes/examples/ex12.F 
*/
void vecrestorearray_(Vec x,double *fa,int *ia,int *__ierr)
{
  Vec    xin = (Vec)MPIR_ToPointer( *(int*)(x) );
  Scalar *lx = PetscDoubleAddressFromFortran(fa,*ia);

  *__ierr = VecRestoreArray(xin,&lx);
}

void vecgetarray_(Vec x,double *fa,int *ia,int *__ierr)
{
  Vec    xin = (Vec)MPIR_ToPointer( *(int*)(x) );
  Scalar *lx;

  *__ierr = VecGetArray(xin,&lx); if (*__ierr) return;
  *ia      = PetscDoubleAddressToFortran(fa,lx);
}
   

void VecScatterdestroy_(VecScatter ctx, int *__ierr )
{
  *__ierr = VecScatterDestroy((VecScatter)MPIR_ToPointer( *(int*)(ctx) ));
   MPIR_RmPointer(*(int*)(ctx)); 
}

void vecdestroy_(Vec v, int *__ierr )
{
  *__ierr = VecDestroy((Vec)MPIR_ToPointer( *(int*)(v) ));
   MPIR_RmPointer(*(int*)(v)); 
}

void VecScattercreate_(Vec xin,IS ix,Vec yin,IS iy,VecScatter *newctx, int *__ierr )
{
  VecScatter lV;
  *__ierr = VecScatterCreate(
	(Vec)MPIR_ToPointer( *(int*)(xin) ),
	(IS)MPIR_ToPointer( *(int*)(ix) ),
	(Vec)MPIR_ToPointer( *(int*)(yin) ),
	(IS)MPIR_ToPointer( *(int*)(iy) ),&lV);
  *(int*) newctx = MPIR_FromPointer(lV);
}
void VecScattercopy_(VecScatter sctx,VecScatter *ctx, int *__ierr )
{
  VecScatter lV;
  *__ierr = VecScatterCopy((VecScatter)MPIR_ToPointer( *(int*)(sctx) ),&lV);
   *(int*) ctx = MPIR_FromPointer(lV); 
}


void veccreatempi_(MPI_Comm comm,int *n,int *N,Vec *vv, int *__ierr )
{
  Vec lV;
  *__ierr = VecCreateMPI((MPI_Comm)MPIR_ToPointer( *(int*)(comm) ),*n,*N,&lV);
  *(int*)vv = MPIR_FromPointer(lV);
}
void veccreateseq_(MPI_Comm comm,int *n,Vec *V, int *__ierr )
{
  Vec lV;
  *__ierr = VecCreateSeq((MPI_Comm)MPIR_ToPointer( *(int*)(comm) ),*n,&lV);
  *(int*)V = MPIR_FromPointer(lV);
}

void veccreate_(MPI_Comm comm,int *n,Vec *V, int *__ierr ){
  Vec lV;
  *__ierr = VecCreate((MPI_Comm)MPIR_ToPointer( *(int*)(comm) ),*n,&lV);
  *(int*)V = MPIR_FromPointer(lV);
}

void vecduplicate_(Vec v,Vec *newv, int *__ierr )
{
  Vec lV;
  *__ierr = VecDuplicate((Vec)MPIR_ToPointer( *(int*)(v) ),&lV);
  *(int*)newv = MPIR_FromPointer(lV);
}
