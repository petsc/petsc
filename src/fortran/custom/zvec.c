#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zvec.c,v 1.32 1998/03/30 22:23:23 balay Exp bsmith $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "vec.h"
#ifdef HAVE_FORTRAN_CAPS
#define vecsetvalue_           VECSETVALUE
#define vecmaxpy_              VECMAXPY
#define vecmdot_               VECMDOT
#define veccreateseq_          VECCREATESEQ
#define veccreateseqwitharray_ VECCREATESEQWITHARRAY
#define veccreatempiwitharray_ VECCREATEMPIWITHARRAY
#define veccreate_             VECCREATE
#define vecduplicate_          VECDUPLICATE
#define veccreatempi_          VECCREATEMPI
#define veccreateshared_       VECCREATESHARED
#define vecscattercreate_      VECSCATTERCREATE
#define vecscattercopy_        VECSCATTERCOPY
#define vecdestroy_            VECDESTROY
#define vecdestroyvecs_        VECDESTROYVECS
#define vecscatterdestroy_     VECSCATTERDESTROY
#define vecrestorearray_       VECRESTOREARRAY
#define vecgetarray_           VECGETARRAY
#define vecload_               VECLOAD
#define vecgettype_            VECGETTYPE
#define vecduplicatevecs_      VECDUPLICATEVECS
#define vecview_               VECVIEW
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
#define vecsetvalue_           vecsetvalue
#define vecview_               vecview
#define vecmaxpy_              vecmaxpy
#define vecmdot_               vecmdot
#define veccreateseq_          veccreateseq
#define veccreateseqwitharray_ veccreateseqwitharray
#define veccreatempiwitharray_ veccreatempiwitharray
#define veccreate_             veccreate
#define vecduplicate_          vecduplicate
#define veccreatempi_          veccreatempi
#define veccreateshared_       veccreateshared
#define vecscattercreate_      vecscattercreate
#define vecscattercopy_        vecscattercopy
#define vecdestroy_            vecdestroy
#define vecdestroyvecs_        vecdestroyvecs
#define vecscatterdestroy_     vecscatterdestroy
#define vecrestorearray_       vecrestorearray
#define vecgetarray_           vecgetarray
#define vecload_               vecload
#define vecgettype_            vecgettype
#define vecduplicatevecs_      vecduplicatevecs
#endif

#if defined(__cplusplus)
extern "C" {
#endif

void vecsetvalue_(Vec v,int *i,Scalar *va,InsertMode *mode)
{
  /* cannot use VecSetValue() here since that uses CHKERRQ() which has a return in it */
  VecSetValues((Vec)PetscToPointer(v),1,i,va,*mode);
}

void vecview_(Vec v,Viewer viewer, int *__ierr )
{
  PetscPatchDefaultViewers_Fortran(viewer);
  *__ierr = VecView((Vec)PetscToPointer(v),viewer);
}

void vecgettype_(Vec vv,VecType *type,CHAR name,int *__ierr,int len)
{
  char *tname;
  if (FORTRANNULL(type)) type = PETSC_NULL;
  *__ierr = VecGetType((Vec)PetscToPointer(vv),type,&tname);
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
  *__ierr = VecLoad((Viewer)PetscToPointer(viewer),&vv);
  *(PetscFortranAddr*) newvec = PetscFromPointer(vv);
}

/* Be to keep vec/examples/ex21.F and snes/examples/ex12.F up to date */
void vecrestorearray_(Vec x,Scalar *fa,long *ia,int *__ierr)
{
  Vec    xin = (Vec)PetscToPointer(x);
  Scalar *lx = PetscScalarAddressFromFortran(fa,*ia);

  *__ierr = VecRestoreArray(xin,&lx);
}

void vecgetarray_(Vec x,Scalar *fa,long *ia,int *__ierr)
{
  Vec    xin = (Vec)PetscToPointer(x);
  Scalar *lx;
  int    shift;

  *__ierr = VecGetArray(xin,&lx); if (*__ierr) return;
  PetscScalarAddressToFortran(fa,lx,ia,&shift);
  if (shift) {
    unsigned long tmp1 = (unsigned long) lx;
    unsigned long tmp2 = (unsigned long) fa;
    /* Address conversion is messed up */
    (*PetscErrorPrintf)("PetscScalarAddressToFortran:C and Fortran arrays are\n");
    (*PetscErrorPrintf)("not commonly aligned.\n");
    (*PetscErrorPrintf)("Locations/sizeof(Scalar): C %f Fortran %f\n",
                          ((double) tmp1)/sizeof(Scalar),((double) tmp2)/sizeof(Scalar));
    MPI_Abort(PETSC_COMM_WORLD,1);
  }
}

void vecscatterdestroy_(VecScatter ctx, int *__ierr )
{
  *__ierr = VecScatterDestroy((VecScatter)PetscToPointer(ctx));
   PetscRmPointer(ctx); 
}

void vecdestroy_(Vec v, int *__ierr )
{
  *__ierr = VecDestroy((Vec)PetscToPointer(v));
   PetscRmPointer(v); 
}

void vecscattercreate_(Vec xin,IS ix,Vec yin,IS iy,VecScatter *newctx, int *__ierr )
{
  VecScatter lV;
  *__ierr = VecScatterCreate((Vec)PetscToPointer(xin),
                             (IS)PetscToPointer(ix),
                             (Vec)PetscToPointer(yin),
                             (IS)PetscToPointer(iy),&lV);
  *(PetscFortranAddr*) newctx = PetscFromPointer(lV);
}

void vecscattercopy_(VecScatter sctx,VecScatter *ctx, int *__ierr )
{
  VecScatter lV;
  *__ierr = VecScatterCopy((VecScatter)PetscToPointer(sctx),&lV);
   *(PetscFortranAddr*) ctx = PetscFromPointer(lV); 
}

void veccreatempi_(MPI_Comm *comm,int *n,int *N,Vec *vv, int *__ierr )
{
  Vec lV;
  *__ierr = VecCreateMPI((MPI_Comm)PetscToPointerComm( *comm ),*n,*N,&lV);
  *(PetscFortranAddr*)vv = PetscFromPointer(lV);
}

void veccreateshared_(MPI_Comm *comm,int *n,int *N,Vec *vv, int *__ierr )
{
  Vec lV;
  *__ierr = VecCreateShared((MPI_Comm)PetscToPointerComm( *comm ),*n,*N,&lV);
  *(PetscFortranAddr*)vv = PetscFromPointer(lV);
}

void veccreateseq_(MPI_Comm *comm,int *n,Vec *V, int *__ierr )
{
  Vec lV;
  *__ierr = VecCreateSeq((MPI_Comm)PetscToPointerComm( *comm),*n,&lV);
  *(PetscFortranAddr*)V = PetscFromPointer(lV);
}

void veccreateseqwitharray_(MPI_Comm *comm,int *n,Scalar *s,Vec *V, int *__ierr )
{
  Vec lV;
  *__ierr = VecCreateSeqWithArray((MPI_Comm)PetscToPointerComm( *comm),*n,s,&lV);
  *(PetscFortranAddr*)V = PetscFromPointer(lV);
}

void veccreatempiwitharray_(MPI_Comm *comm,int *n,int *N,Scalar *s,Vec *V, int *__ierr )
{
  Vec lV;
  *__ierr = VecCreateMPIWithArray((MPI_Comm)PetscToPointerComm( *comm),*n,*N,s,&lV);
  *(PetscFortranAddr*)V = PetscFromPointer(lV);
}

void veccreate_(MPI_Comm *comm,int *n,int *N,Vec *V, int *__ierr )
{
  Vec lV;
  *__ierr = VecCreate((MPI_Comm)PetscToPointerComm( *comm ),*n,*N,&lV);
  *(PetscFortranAddr*)V = PetscFromPointer(lV);
}

void vecduplicate_(Vec v,Vec *newv, int *__ierr )
{
  Vec lV;
  *__ierr = VecDuplicate((Vec)PetscToPointer(v),&lV);
  *(PetscFortranAddr*)newv = PetscFromPointer(lV);
}

void vecduplicatevecs_(Vec v,int *m,PetscFortranAddr *newv, int *__ierr )
{
  Vec *lV;
  int i;
  *__ierr = VecDuplicateVecs((Vec)PetscToPointer(v),*m,&lV);
  for (i=0; i<*m; i++) {
    newv[i] = PetscFromPointer(lV[i]);
  }
  PetscFree(lV); 
}

void vecdestroyvecs_(PetscFortranAddr *vecs,int *m,int *__ierr )
{
  int i;
  for (i=0; i<*m; i++) {
    *__ierr = VecDestroy((Vec)PetscToPointer(&vecs[i]));
    PetscRmPointer(&vecs[i]); 
  }
}

void vecmtdot_(int *nv,Vec x,PetscFortranAddr *y,Scalar *val, int *__ierr )
{
  int i;
  Vec *yV = (Vec *) PetscMalloc( *nv * sizeof(Vec *));
  if (!(yV)) {
     *__ierr = PetscError(__LINE__,"VecMTDot_Fortran",__FILE__,__SDIR__,PETSC_ERR_MEM,0,(char*)0);
     return;
  }
  for (i=0; i<*nv; i++) yV[i] = ((Vec)PetscToPointer(&y[i]));
  *__ierr = VecMTDot(*nv,(Vec)PetscToPointer(x),yV,val);
  PetscFree(yV);
}

void vecmdot_(int *nv,Vec x,PetscFortranAddr *y,Scalar *val, int *__ierr )
{
  int i;
  Vec *yV = (Vec *) PetscMalloc( *nv * sizeof(Vec *));
  if (!(yV)) {
     *__ierr = PetscError(__LINE__,"VecMDot_Fortran",__FILE__,__SDIR__,PETSC_ERR_MEM,0,(char*)0);
     return;
  }
  for (i=0; i<*nv; i++) yV[i] = ((Vec)PetscToPointer(&y[i]));
  *__ierr = VecMDot(*nv,(Vec)PetscToPointer(x),yV,val);
  PetscFree(yV);
}

void vecmaxpy_(int *nv,Scalar *alpha,Vec x,PetscFortranAddr *y, int *__ierr )
{
  int i;
  Vec *yV = (Vec *) PetscMalloc( *nv * sizeof(Vec *));
  if (!(yV)) {
     *__ierr = PetscError(__LINE__,"VecMAXPY_Fortran",__FILE__,__SDIR__,PETSC_ERR_MEM,0,(char*)0);
     return;
  }
  for (i=0; i<*nv; i++) yV[i] = ((Vec)PetscToPointer(&y[i]));
  *__ierr = VecMAXPY(*nv,alpha,(Vec)PetscToPointer(x),yV);
  PetscFree(yV);
}

#if defined(__cplusplus)
}
#endif
