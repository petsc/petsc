#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: zvec.c,v 1.49 1999/01/31 16:12:02 bsmith Exp bsmith $";
#endif

#include "src/fortran/custom/zpetsc.h"
#include "vec.h"
#ifdef HAVE_FORTRAN_CAPS
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
#elif !defined(HAVE_FORTRAN_UNDERSCORE)
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

void vecsetrandom_(PetscRandom *r,Vec *x, int *__ierr )
{
  *__ierr = VecSetRandom(*r,*x);
}

void drawtensorcontour_(Draw *win,int *m,int *n,double *x,double *y,Scalar *V, int *__ierr )
{
  double *xx,*yy;
  if (FORTRANNULLDOUBLE(x)) xx = PETSC_NULL; 
  else xx = x;
  if (FORTRANNULLDOUBLE(y)) yy = PETSC_NULL; 
  else yy = y;

  *__ierr = DrawTensorContour(*win,*m,*n,xx,yy,V);
}

void vecsetfromoptions_(Vec *x,int *__ierr)
{
  *__ierr = VecSetFromOptions(*x);
}

void vecsettype_(Vec *x,char *type_name,int *__ierr,int len)
{
  char *t;

  FIXCHAR(type_name,len,t);
  *__ierr = VecSetType(*x,t);
  FREECHAR(type_name,t);
}

void vecgetmap_(Vec *x,Map *map, int *__ierr )
{
  *__ierr = VecGetMap(*x,map);
}

void mapgetlocalsize_(Map *m,int *n, int *__ierr )
{
  *__ierr = MapGetLocalSize(*m,n);
}

void mapgetsize_(Map *m,int *N, int *__ierr )
{
  *__ierr = MapGetSize(*m,N);
}

void mapgetlocalrange_(Map *m,int *rstart,int *rend, int *__ierr )
{
  *__ierr = MapGetLocalRange(*m,rstart,rend);
}

void mapgetglobalrange_(Map *m,int **range, int *__ierr )
{
  *__ierr = MapGetGlobalRange(*m,range);
}

void mapdestroy_(Map *m, int *__ierr )
{
  *__ierr = MapDestroy(*m);
}

void vecsetvalue_(Vec *v,int *i,Scalar *va,InsertMode *mode)
{
  /* cannot use VecSetValue() here since that uses CHKERRQ() which has a return in it */
  VecSetValues(*v,1,i,va,*mode);
}

void vecview_(Vec *x,Viewer *vin, int *__ierr )
{
  Viewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *__ierr = VecView(*x,v);
}

void vecgettype_(Vec *vv,CHAR name,int *__ierr,int len)
{
  char *tname;
  *__ierr = VecGetType(*vv,&tname);
#if defined(USES_CPTOFCD)
  {
  char *t = _fcdtocp(name); int len1 = _fcdlen(name);
  PetscStrncpy(t,tname,len1);
  }
#else
  PetscStrncpy(name,tname,len);
#endif

}

void vecload_(Viewer *viewer,Vec *newvec, int *__ierr )
{ 
  *__ierr = VecLoad(*viewer,newvec);
}

/* Be to keep vec/examples/ex21.F and snes/examples/ex12.F up to date */
void vecrestorearray_(Vec *x,Scalar *fa,long *ia,int *__ierr)
{
  int    m;
  Scalar *lx;

  *__ierr = VecGetLocalSize(*x,&m);if (*__ierr) return;
  *__ierr = PetscScalarAddressFromFortran((PetscObject)*x,fa,*ia,m,&lx);if (*__ierr) return;
  *__ierr = VecRestoreArray(*x,&lx);if (*__ierr) return;
}

void vecgetarray_(Vec *x,Scalar *fa,long *ia,int *__ierr)
{
  Scalar *lx;
  int    m;

  *__ierr = VecGetArray(*x,&lx); if (*__ierr) return;
  *__ierr = VecGetLocalSize(*x,&m);if (*__ierr) return;
  *__ierr = PetscScalarAddressToFortran((PetscObject)*x,fa,lx,m,ia);
}

void vecscatterdestroy_(VecScatter *ctx, int *__ierr )
{
  *__ierr = VecScatterDestroy(*ctx);
}

void vecdestroy_(Vec *v, int *__ierr )
{
  *__ierr = VecDestroy(*v);
}

void vecscattercreate_(Vec *xin,IS *ix,Vec *yin,IS *iy,VecScatter *newctx, int *__ierr )
{
  *__ierr = VecScatterCreate(*xin,*ix,*yin,*iy,newctx);
}

void vecscattercopy_(VecScatter *sctx,VecScatter *ctx, int *__ierr )
{
  *__ierr = VecScatterCopy(*sctx,ctx);
}

void mapcreatempi_(MPI_Comm *comm,int *n,int *N,Map *vv, int *__ierr )
{
  *__ierr = MapCreateMPI((MPI_Comm)PetscToPointerComm( *comm ),*n,*N,vv);
}

void veccreatempi_(MPI_Comm *comm,int *n,int *N,Vec *vv, int *__ierr )
{
  *__ierr = VecCreateMPI((MPI_Comm)PetscToPointerComm( *comm ),*n,*N,vv);
}

void veccreateshared_(MPI_Comm *comm,int *n,int *N,Vec *vv, int *__ierr )
{
  *__ierr = VecCreateShared((MPI_Comm)PetscToPointerComm( *comm ),*n,*N,vv);
}

void veccreateseq_(MPI_Comm *comm,int *n,Vec *V, int *__ierr )
{
  *__ierr = VecCreateSeq((MPI_Comm)PetscToPointerComm( *comm),*n,V);
}

void veccreateseqwitharray_(MPI_Comm *comm,int *n,Scalar *s,Vec *V, int *__ierr )
{
  *__ierr = VecCreateSeqWithArray((MPI_Comm)PetscToPointerComm( *comm),*n,s,V);
}

void veccreatempiwitharray_(MPI_Comm *comm,int *n,int *N,Scalar *s,Vec *V, int *__ierr )
{
  *__ierr = VecCreateMPIWithArray((MPI_Comm)PetscToPointerComm( *comm),*n,*N,s,V);
}

void veccreate_(MPI_Comm *comm,int *n,int *N,Vec *V, int *__ierr )
{
  *__ierr = VecCreate((MPI_Comm)PetscToPointerComm( *comm ),*n,*N,V);
}

void vecduplicate_(Vec *v,Vec *newv, int *__ierr )
{
  *__ierr = VecDuplicate(*v,newv);
}

/*
      vecduplicatevecs() and vecdestroyvecs() are slightly different from C since the 
    Fortran provides the array to hold the vector objects, while in C that 
    array is allocated by the VecDuplicateVecs()
*/
void vecduplicatevecs_(Vec *v,int *m,Vec *newv, int *__ierr )
{
  Vec *lV;
  int i;
  *__ierr = VecDuplicateVecs(*v,*m,&lV);
  for (i=0; i<*m; i++) {
    newv[i] = lV[i];
  }
  PetscFree(lV); 
}

void vecdestroyvecs_(Vec *vecs,int *m,int *__ierr )
{
  int i;
  for (i=0; i<*m; i++) {
    *__ierr = VecDestroy(vecs[i]);if (*__ierr) return;
  }
}

void vecmtdot_(int *nv,Vec *x,Vec *y,Scalar *val, int *__ierr )
{
  *__ierr = VecMTDot(*nv,*x,y,val);
}

void vecmdot_(int *nv,Vec *x,Vec *y,Scalar *val, int *__ierr )
{
  *__ierr = VecMDot(*nv,*x,y,val);
}

void vecmaxpy_(int *nv,Scalar *alpha,Vec *x,Vec *y, int *__ierr )
{
  *__ierr = VecMAXPY(*nv,alpha,*x,y);
}

void vecstridenorm_(Vec *x,int *start,NormType *type,double *val, int *__ierr )
{
  *__ierr = VecStrideNorm(*x,*start,*type,val);
}

/* ----------------------------------------------------------------------------------------------*/
void veccreateghostwitharray_(MPI_Comm *comm,int *n,int *N,int *nghost,int *ghosts,Scalar *array,
                              Vec *vv, int *__ierr )
{
  *__ierr = VecCreateGhostWithArray((MPI_Comm)PetscToPointerComm( *comm) ,*n,*N,*nghost,
                                    ghosts,array,vv);
}

void veccreateghost_(MPI_Comm *comm,int *n,int *N,int *nghost,int *ghosts,Vec *vv, int *__ierr )
{
  *__ierr = VecCreateGhost((MPI_Comm)PetscToPointerComm( *comm),*n,*N,*nghost,ghosts,vv);
}

void vecghostgetlocalform_(Vec *g,Vec *l, int *__ierr )
{
  *__ierr = VecGhostGetLocalForm(*g,l);
}

void vecghostrestorelocalform_(Vec *g,Vec *l, int *__ierr )
{
  *__ierr = VecGhostRestoreLocalForm(*g,l);
}

void vecmax_(Vec *x,int *p,double *val, int *__ierr )
{
  if (FORTRANNULLINTEGER(p)) p = PETSC_NULL;
  *__ierr = VecMax(*x,p,val);
}

EXTERN_C_END
