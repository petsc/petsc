#ifndef lint
static char vcid[] = "$Id: zvec.c,v 1.12 1996/03/23 16:56:55 bsmith Exp bsmith $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "vec.h"
#ifdef HAVE_FORTRAN_CAPS
#define veccreateseq_         VECCREATESEQ
#define veccreate_            VECCREATE
#define vecduplicate_         VECDUPLICATE
#define veccreatempi_         VECCREATEMPI
#define vecscattercreate_     VECSCATTERCREATE
#define vecscattercopy_       VECSCATTERCOPY
#define vecdestroy_           VECDESTROY
#define vecscatterdestroy_    VECSCATTERDESTROY
#define vecrestorearray_      VECRESTOREARRAY
#define vecgetarray_          VECGETARRAY
#define vecload_              VECLOAD
#define vecgettype_           VECGETTYPE
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define veccreateseq_         veccreateseq
#define veccreate_            veccreate
#define vecduplicate_         vecduplicate
#define veccreatempi_         veccreatempi
#define vecscattercreate_     vecscattercreate
#define vecscattercopy_       vecscattercopy
#define vecdestroy_           vecdestroy
#define vecscatterdestroy_    vecscatterdestroy
#define vecrestorearray_      vecrestorearray
#define vecgetarray_          vecgetarray
#define vecload_              vecload
#define vecgettype_           vecgettype
#endif

#if defined(__cplusplus)
extern "C" {
#endif



void vecgettype_(Vec vv,VecType *type,CHAR name,int *__ierr,int len)
{
  char *tname;
  if (FORTRANNULL(type)) type = PETSC_NULL;
  *__ierr = VecGetType((Vec)PetscToPointer(*(int*)vv),type,&tname);
#if defined(USES_CPTOFCD)
  {
  char *t = _fcdtocp(name); int len1 = _fcdlen(name);
  if (t != PETSC_NULL_CHARACTER_Fortran) PetscStrncpy(t,tname,len1);
  }
#else
  if (name != PETSC_NULL_CHARACTER_Fortran) PetscStrncpy(name,tname,len);
#endif

}

void vecload_(Viewer viewer,Vec *newvec, int *__ierr )
{ 
  Vec vv;
  *__ierr = VecLoad((Viewer)PetscToPointer( *(int*)(viewer) ),&vv);
  *(int *) newvec = PetscFromPointer(vv);
}

/* Be to keep vec/examples/ex21.F and snes/examples/ex12.F up to date */
void vecrestorearray_(Vec x,Scalar *fa,int *ia,int *__ierr)
{
  Vec    xin = (Vec)PetscToPointer( *(int*)(x) );
  Scalar *lx = PetscScalarAddressFromFortran(fa,*ia);

  *__ierr = VecRestoreArray(xin,&lx);
}

void vecgetarray_(Vec x,Scalar *fa,int *ia,int *__ierr)
{
  Vec    xin = (Vec)PetscToPointer( *(int*)(x) );
  Scalar *lx;

  *__ierr = VecGetArray(xin,&lx); if (*__ierr) return;
  *ia      = PetscScalarAddressToFortran(fa,lx);
}

void vecscatterdestroy_(VecScatter ctx, int *__ierr )
{
  *__ierr = VecScatterDestroy((VecScatter)PetscToPointer( *(int*)(ctx) ));
   PetscRmPointer(*(int*)(ctx)); 
}

void vecdestroy_(Vec v, int *__ierr )
{
  *__ierr = VecDestroy((Vec)PetscToPointer( *(int*)(v) ));
   PetscRmPointer(*(int*)(v)); 
}

void vecscattercreate_(Vec xin,IS ix,Vec yin,IS iy,VecScatter *newctx, int *__ierr )
{
  VecScatter lV;
  *__ierr = VecScatterCreate(
	(Vec)PetscToPointer( *(int*)(xin) ),
	(IS)PetscToPointer( *(int*)(ix) ),
	(Vec)PetscToPointer( *(int*)(yin) ),
	(IS)PetscToPointer( *(int*)(iy) ),&lV);
  *(int*) newctx = PetscFromPointer(lV);
}
void vecscattercopy_(VecScatter sctx,VecScatter *ctx, int *__ierr )
{
  VecScatter lV;
  *__ierr = VecScatterCopy((VecScatter)PetscToPointer( *(int*)(sctx) ),&lV);
   *(int*) ctx = PetscFromPointer(lV); 
}


void veccreatempi_(MPI_Comm comm,int *n,int *N,Vec *vv, int *__ierr )
{
  Vec lV;
  *__ierr = VecCreateMPI((MPI_Comm)PetscToPointerComm( *(int*)(comm) ),*n,*N,&lV);
  *(int*)vv = PetscFromPointer(lV);
}

void veccreateseq_(MPI_Comm comm,int *n,Vec *V, int *__ierr )
{
  Vec lV;
  *__ierr = VecCreateSeq((MPI_Comm)PetscToPointerComm( *(int*)(comm)),*n,&lV);
  *(int*)V = PetscFromPointer(lV);
}

void veccreate_(MPI_Comm comm,int *n,Vec *V, int *__ierr ){
  Vec lV;
  *__ierr = VecCreate((MPI_Comm)PetscToPointerComm( *(int*)(comm) ),*n,&lV);
  *(int*)V = PetscFromPointer(lV);
}

void vecduplicate_(Vec v,Vec *newv, int *__ierr )
{
  Vec lV;
  *__ierr = VecDuplicate((Vec)PetscToPointer( *(int*)(v) ),&lV);
  *(int*)newv = PetscFromPointer(lV);
}

#if defined(__cplusplus)
}
#endif
