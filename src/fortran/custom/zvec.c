/*$Id: zvec.c,v 1.61 2000/06/24 03:59:07 balay Exp balay $*/

#include "src/fortran/custom/zpetsc.h"
#include "petscvec.h"
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define vecsetfromoptions_        VECSETFROMOPTIONS
#define vecsettype_               VECSETTYPE
#define vecsetvalue_              VECSETVALUE
#define vecmaxpy_                 VECMAXPY
#define vecmdot_                  VECMDOT
#define veccreateseq_             VECCREATESEQ
#define veccreateseqwitharray_    VECCREATESEQWITHARRAY
#define veccreatempiwitharray_    VECCREATEMPIWITHARRAY
#define veccreate_                VECCREATE
#define vecduplicate_             VECDUPLICATE
#define veccreatempi_             VECCREATEMPI
#define veccreateshared_          VECCREATESHARED
#define vecscattercreate_         VECSCATTERCREATE
#define vecscattercopy_           VECSCATTERCOPY
#define vecdestroy_               VECDESTROY
#define vecdestroyvecs_           VECDESTROYVECS
#define vecscatterdestroy_        VECSCATTERDESTROY
#define vecrestorearray_          VECRESTOREARRAY
#define vecgetarray_              VECGETARRAY
#define vecload_                  VECLOAD
#define vecgettype_               VECGETTYPE
#define vecduplicatevecs_         VECDUPLICATEVECS
#define vecview_                  VECVIEW
#define mapgetlocalsize_          MAPGETLOCALSIZE
#define mapgetsize_               MAPGETSIZE
#define mapgetlocalrange_         MAPGETLOCALRANGE
#define mapgetglobalrange_        MAPGETGLOBALRANGE
#define mapdestroy_               MAPDESTROY
#define mapcreatempi_             MAPCREATEMPI
#define vecgetmap_                VECGETMAP
#define vecghostgetlocalform_     VECGHOSTGETLOCALFORM
#define vecghostrestorelocalform_ VECGHOSTRESTORELOCALFORM
#define veccreateghostwitharray_  VECCREATEGHOSTWITHARRAY
#define veccreateghost_           VECCREATEGHOST
#define vecstridenorm_            VECSTRIDENORM
#define vecmax_                   VECMAX
#define drawtensorcontour_        DRAWTENSORCONTOUR
#define vecsetrandom_              VECSETRANDOM
#define veccreateghostblockwitharray_ VECCREATEGHOSTBLOCKWITHARRAY
#define veccreateghostblock_          VECCREATEGHOSTBLOCK
#define vecloadintovector_            VECLOADINTOVECTOR   
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define vecloadintovector_            vecloadintovector
#define veccreateghostblockwitharray_ veccreateghostblockwitharray
#define veccreateghostblock_      veccreateghostblock
#define drawtensorcontour_        drawtensorcontour
#define vecsetfromoptions_        vecsetfromoptions
#define vecsettype_               vecsettype
#define vecstridenorm_            vecstridenorm
#define vecghostrestorelocalform_ vecghostrestorelocalform
#define vecghostgetlocalform_     vecghostgetlocalform
#define veccreateghostwitharray_  veccreateghostwitharray
#define veccreateghost_           veccreateghost
#define vecgetmap_                vecgetmap
#define mapcreatempi_             mapcreatempi
#define mapgetglobalrange_        mapgetglobalrange
#define mapgetsize_               mapgetsize
#define mapgetlocalsize_          mapgetlocalsize
#define mapgetlocalrange_         mapgetlocalrange
#define mapdestroy_               mapdestroy
#define vecsetvalue_              vecsetvalue
#define vecview_                  vecview
#define vecmaxpy_                 vecmaxpy
#define vecmdot_                  vecmdot
#define veccreateseq_             veccreateseq
#define veccreateseqwitharray_    veccreateseqwitharray
#define veccreatempiwitharray_    veccreatempiwitharray
#define veccreate_                veccreate
#define vecduplicate_             vecduplicate
#define veccreatempi_             veccreatempi
#define veccreateshared_          veccreateshared
#define vecscattercreate_         vecscattercreate
#define vecscattercopy_           vecscattercopy
#define vecdestroy_               vecdestroy
#define vecdestroyvecs_           vecdestroyvecs
#define vecscatterdestroy_        vecscatterdestroy
#define vecrestorearray_          vecrestorearray
#define vecgetarray_              vecgetarray
#define vecload_                  vecload
#define vecgettype_               vecgettype
#define vecduplicatevecs_         vecduplicatevecs
#define vecmax_                   vecmax
#define vecsetrandom_              vecsetrandom
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL vecloadintovector_(Viewer *viewer,Vec *vec,int *ierr)
{
  *ierr = VecLoadIntoVector(*viewer,*vec);
}

void PETSC_STDCALL vecsetrandom_(PetscRandom *r,Vec *x,int *ierr)
{
  *ierr = VecSetRandom(*r,*x);
}

void PETSC_STDCALL drawtensorcontour_(Draw *win,int *m,int *n,double *x,double *y,Scalar *V,int *ierr)
{
  double *xx,*yy;
  if (FORTRANNULLDOUBLE(x)) xx = PETSC_NULL; 
  else xx = x;
  if (FORTRANNULLDOUBLE(y)) yy = PETSC_NULL; 
  else yy = y;

  *ierr = DrawTensorContour(*win,*m,*n,xx,yy,V);
}

void PETSC_STDCALL vecsetfromoptions_(Vec *x,int *ierr)
{
  *ierr = VecSetFromOptions(*x);
}

void PETSC_STDCALL vecsettype_(Vec *x,CHAR type_name PETSC_MIXED_LEN(len),int *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type_name,len,t);
  *ierr = VecSetType(*x,t);
  FREECHAR(type_name,t);
}

void PETSC_STDCALL vecgetmap_(Vec *x,Map *map,int *ierr)
{
  *ierr = VecGetMap(*x,map);
}

void PETSC_STDCALL mapgetlocalsize_(Map *m,int *n,int *ierr)
{
  *ierr = MapGetLocalSize(*m,n);
}

void PETSC_STDCALL mapgetsize_(Map *m,int *N,int *ierr)
{
  *ierr = MapGetSize(*m,N);
}

void PETSC_STDCALL mapgetlocalrange_(Map *m,int *rstart,int *rend,int *ierr)
{
  *ierr = MapGetLocalRange(*m,rstart,rend);
}

void PETSC_STDCALL mapgetglobalrange_(Map *m,int **range,int *ierr)
{
  *ierr = MapGetGlobalRange(*m,range);
}

void PETSC_STDCALL mapdestroy_(Map *m,int *ierr)
{
  *ierr = MapDestroy(*m);
}

void PETSC_STDCALL vecsetvalue_(Vec *v,int *i,Scalar *va,InsertMode *mode)
{
  /* cannot use VecSetValue() here since that usesCHKERRQ() which has a return in it */
  VecSetValues(*v,1,i,va,*mode);
}

void PETSC_STDCALL vecview_(Vec *x,Viewer *vin,int *ierr)
{
  Viewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = VecView(*x,v);
}

void PETSC_STDCALL vecgettype_(Vec *vv,CHAR name PETSC_MIXED_LEN(len),int *ierr PETSC_END_LEN(len))
{
  char *tname;
  *ierr = VecGetType(*vv,&tname);
#if defined(PETSC_USES_CPTOFCD)
  {
  char *t = _fcdtocp(name); int len1 = _fcdlen(name);
  *ierr = PetscStrncpy(t,tname,len1);
  }
#else
  *ierr = PetscStrncpy(name,tname,len);
#endif

}

void PETSC_STDCALL vecload_(Viewer *viewer,Vec *newvec,int *ierr)
{ 
  *ierr = VecLoad(*viewer,newvec);
}

/* Be to keep vec/examples/ex21.F and snes/examples/ex12.F up to date */
void PETSC_STDCALL vecrestorearray_(Vec *x,Scalar *fa,long *ia,int *ierr)
{
  int    m;
  Scalar *lx;

  *ierr = VecGetLocalSize(*x,&m);if (*ierr) return;
  *ierr = PetscScalarAddressFromFortran((PetscObject)*x,fa,*ia,m,&lx);if (*ierr) return;
  *ierr = VecRestoreArray(*x,&lx);if (*ierr) return;
}

void PETSC_STDCALL vecgetarray_(Vec *x,Scalar *fa,long *ia,int *ierr)
{
  Scalar *lx;
  int    m;

  *ierr = VecGetArray(*x,&lx); if (*ierr) return;
  *ierr = VecGetLocalSize(*x,&m);if (*ierr) return;
  *ierr = PetscScalarAddressToFortran((PetscObject)*x,fa,lx,m,ia);
}

void PETSC_STDCALL vecscatterdestroy_(VecScatter *ctx,int *ierr)
{
  *ierr = VecScatterDestroy(*ctx);
}

void PETSC_STDCALL vecdestroy_(Vec *v,int *ierr)
{
  *ierr = VecDestroy(*v);
}

void PETSC_STDCALL vecscattercreate_(Vec *xin,IS *ix,Vec *yin,IS *iy,VecScatter *newctx,int *ierr)
{
  *ierr = VecScatterCreate(*xin,*ix,*yin,*iy,newctx);
}

void PETSC_STDCALL vecscattercopy_(VecScatter *sctx,VecScatter *ctx,int *ierr)
{
  *ierr = VecScatterCopy(*sctx,ctx);
}

void PETSC_STDCALL mapcreatempi_(MPI_Comm *comm,int *n,int *N,Map *vv,int *ierr)
{
  *ierr = MapCreateMPI((MPI_Comm)PetscToPointerComm(*comm),*n,*N,vv);
}

void PETSC_STDCALL veccreatempi_(MPI_Comm *comm,int *n,int *N,Vec *vv,int *ierr)
{
  *ierr = VecCreateMPI((MPI_Comm)PetscToPointerComm(*comm),*n,*N,vv);
}

void PETSC_STDCALL veccreateshared_(MPI_Comm *comm,int *n,int *N,Vec *vv,int *ierr)
{
  *ierr = VecCreateShared((MPI_Comm)PetscToPointerComm(*comm),*n,*N,vv);
}

void PETSC_STDCALL veccreateseq_(MPI_Comm *comm,int *n,Vec *V,int *ierr)
{
  *ierr = VecCreateSeq((MPI_Comm)PetscToPointerComm(*comm),*n,V);
}

void PETSC_STDCALL veccreateseqwitharray_(MPI_Comm *comm,int *n,Scalar *s,Vec *V,int *ierr)
{
  *ierr = VecCreateSeqWithArray((MPI_Comm)PetscToPointerComm(*comm),*n,s,V);
}

void PETSC_STDCALL veccreatempiwitharray_(MPI_Comm *comm,int *n,int *N,Scalar *s,Vec *V,int *ierr)
{
  *ierr = VecCreateMPIWithArray((MPI_Comm)PetscToPointerComm(*comm),*n,*N,s,V);
}

void PETSC_STDCALL veccreate_(MPI_Comm *comm,int *n,int *N,Vec *V,int *ierr)
{
  *ierr = VecCreate((MPI_Comm)PetscToPointerComm(*comm),*n,*N,V);
}

void PETSC_STDCALL vecduplicate_(Vec *v,Vec *newv,int *ierr)
{
  *ierr = VecDuplicate(*v,newv);
}

/*
      vecduplicatevecs() and vecdestroyvecs() are slightly different from C since the 
    Fortran provides the array to hold the vector objects,while in C that 
    array is allocated by the VecDuplicateVecs()
*/
void PETSC_STDCALL vecduplicatevecs_(Vec *v,int *m,Vec *newv,int *ierr)
{
  Vec *lV;
  int i;
  *ierr = VecDuplicateVecs(*v,*m,&lV);
  for (i=0; i<*m; i++) {
    newv[i] = lV[i];
  }
  PetscFree(lV); 
}

void PETSC_STDCALL vecdestroyvecs_(Vec *vecs,int *m,int *ierr)
{
  int i;
  for (i=0; i<*m; i++) {
    *ierr = VecDestroy(vecs[i]);if (*ierr) return;
  }
}

void PETSC_STDCALL vecmtdot_(int *nv,Vec *x,Vec *y,Scalar *val,int *ierr)
{
  *ierr = VecMTDot(*nv,*x,y,val);
}

void PETSC_STDCALL vecmdot_(int *nv,Vec *x,Vec *y,Scalar *val,int *ierr)
{
  *ierr = VecMDot(*nv,*x,y,val);
}

void PETSC_STDCALL vecmaxpy_(int *nv,Scalar *alpha,Vec *x,Vec *y,int *ierr)
{
  *ierr = VecMAXPY(*nv,alpha,*x,y);
}

void PETSC_STDCALL vecstridenorm_(Vec *x,int *start,NormType *type,double *val,int *ierr)
{
  *ierr = VecStrideNorm(*x,*start,*type,val);
}

/* ----------------------------------------------------------------------------------------------*/
void PETSC_STDCALL veccreateghostblockwitharray_(MPI_Comm *comm,int *bs,int *n,int *N,int *nghost,int *ghosts,
                              Scalar *array,Vec *vv,int *ierr)
{
  *ierr = VecCreateGhostBlockWithArray((MPI_Comm)PetscToPointerComm(*comm),*bs,*n,*N,*nghost,
                                    ghosts,array,vv);
}

void PETSC_STDCALL veccreateghostblock_(MPI_Comm *comm,int *bs,int *n,int *N,int *nghost,int *ghosts,Vec *vv,
                          int *ierr)
{
  *ierr = VecCreateGhostBlock((MPI_Comm)PetscToPointerComm(*comm),*bs,*n,*N,*nghost,ghosts,vv);
}

void PETSC_STDCALL veccreateghostwitharray_(MPI_Comm *comm,int *n,int *N,int *nghost,int *ghosts,Scalar *array,
                              Vec *vv,int *ierr)
{
  *ierr = VecCreateGhostWithArray((MPI_Comm)PetscToPointerComm(*comm),*n,*N,*nghost,
                                    ghosts,array,vv);
}

void PETSC_STDCALL veccreateghost_(MPI_Comm *comm,int *n,int *N,int *nghost,int *ghosts,Vec *vv,int *ierr)
{
  *ierr = VecCreateGhost((MPI_Comm)PetscToPointerComm(*comm),*n,*N,*nghost,ghosts,vv);
}

void PETSC_STDCALL vecghostgetlocalform_(Vec *g,Vec *l,int *ierr)
{
  *ierr = VecGhostGetLocalForm(*g,l);
}

void PETSC_STDCALL vecghostrestorelocalform_(Vec *g,Vec *l,int *ierr)
{
  *ierr = VecGhostRestoreLocalForm(*g,l);
}

void PETSC_STDCALL vecmax_(Vec *x,int *p,double *val,int *ierr)
{
  if (FORTRANNULLINTEGER(p)) p = PETSC_NULL;
  *ierr = VecMax(*x,p,val);
}

EXTERN_C_END
