
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
#define vecgetpetscmap_           VECGETPETSCMAP
#define vecghostgetlocalform_     VECGHOSTGETLOCALFORM
#define vecghostrestorelocalform_ VECGHOSTRESTORELOCALFORM
#define veccreateghostwitharray_  VECCREATEGHOSTWITHARRAY
#define veccreateghost_           VECCREATEGHOST
#define vecstridenorm_            VECSTRIDENORM
#define vecmax_                   VECMAX
#define petscdrawtensorcontour_   PETSCDRAWTENSORCONTOUR
#define vecsetrandom_             VECSETRANDOM
#define veccreateghostblockwitharray_ VECCREATEGHOSTBLOCKWITHARRAY
#define veccreateghostblock_          VECCREATEGHOSTBLOCK
#define vecloadintovector_            VECLOADINTOVECTOR  
#define vecscattercreatetoall_        VECSCATTERCREATETOALL
#define vecscattercreatetozero_       VECSCATTERCREATETOZERO
#define vecgetownershiprange_         VECGETOWNERSHIPRANGE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define vecloadintovector_            vecloadintovector
#define veccreateghostblockwitharray_ veccreateghostblockwitharray
#define veccreateghostblock_      veccreateghostblock
#define petscdrawtensorcontour_   petscdrawtensorcontour
#define vecsetfromoptions_        vecsetfromoptions
#define vecsettype_               vecsettype
#define vecstridenorm_            vecstridenorm
#define vecghostrestorelocalform_ vecghostrestorelocalform
#define vecghostgetlocalform_     vecghostgetlocalform
#define veccreateghostwitharray_  veccreateghostwitharray
#define veccreateghost_           veccreateghost
#define vecgetpetscmap_           vecgetpetscmap
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
#define vecsetrandom_             vecsetrandom
#define vecscattercreatetoall_    vecscattercreatetoall
#define vecscattercreatetozero_   vecscattercreatetozero
#define vecgetownershiprange_     vecgetownershiprange
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL vecloadintovector_(PetscViewer *viewer,Vec *vec,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = VecLoadIntoVector(v,*vec);
}

void PETSC_STDCALL vecsetrandom_(Vec *x,PetscRandom *r,PetscErrorCode *ierr)
{
  *ierr = VecSetRandom(*x,*r);
}
void PETSC_STDCALL petscdrawtensorcontour_(PetscDraw *win,int *m,int *n,PetscReal *x,PetscReal *y,PetscReal *V,PetscErrorCode *ierr)
{
  CHKFORTRANNULLDOUBLE(x);
  CHKFORTRANNULLDOUBLE(y);
  *ierr = PetscDrawTensorContour(*win,*m,*n,x,y,V);
}

void PETSC_STDCALL vecsetfromoptions_(Vec *x,PetscErrorCode *ierr)
{
  *ierr = VecSetFromOptions(*x);
}

void PETSC_STDCALL vecsettype_(Vec *x,CHAR type_name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type_name,len,t);
  *ierr = VecSetType(*x,t);
  FREECHAR(type_name,t);
}

void PETSC_STDCALL vecgetpetscmap_(Vec *x,PetscMap *map,PetscErrorCode *ierr)
{
  *ierr = VecGetPetscMap(*x,map);
}

void PETSC_STDCALL mapgetlocalsize_(PetscMap *m,PetscInt *n,PetscErrorCode *ierr)
{
  *ierr = PetscMapGetLocalSize(*m,n);
}

void PETSC_STDCALL mapgetsize_(PetscMap *m,PetscInt *N,PetscErrorCode *ierr)
{
  *ierr = PetscMapGetSize(*m,N);
}

void PETSC_STDCALL mapgetlocalrange_(PetscMap *m,PetscInt *rstart,PetscInt *rend,PetscErrorCode *ierr)
{
  *ierr = PetscMapGetLocalRange(*m,rstart,rend);
}

void PETSC_STDCALL mapgetglobalrange_(PetscMap *m,PetscInt **range,PetscErrorCode *ierr)
{
  *ierr = PetscMapGetGlobalRange(*m,range);
}

void PETSC_STDCALL mapdestroy_(PetscMap *m,PetscErrorCode *ierr)
{
  *ierr = PetscMapDestroy(*m);
}

void PETSC_STDCALL vecsetvalue_(Vec *v,PetscInt *i,PetscScalar *va,InsertMode *mode,PetscErrorCode *ierr)
{
  /* cannot use VecSetValue() here since that usesCHKERRQ() which has a return in it */
  *ierr = VecSetValues(*v,1,i,va,*mode);
}

void PETSC_STDCALL vecview_(Vec *x,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = VecView(*x,v);
}

void PETSC_STDCALL vecgettype_(Vec *vv,CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  const char *tname;

  *ierr = VecGetType(*vv,&tname);
#if defined(PETSC_USES_CPTOFCD)
  {
  char *t = _fcdtocp(name); int len1 = _fcdlen(name);
  *ierr = PetscStrncpy(t,tname,len1);
  }
#else
  *ierr = PetscStrncpy(name,tname,len);
#endif
  FIXRETURNCHAR(name,len);
}

void PETSC_STDCALL vecload_(PetscViewer *viewer,CHAR outtype PETSC_MIXED_LEN(len),Vec *newvec,PetscErrorCode *ierr PETSC_END_LEN(len))
{ 
  char *t;
  PetscViewer v;
  FIXCHAR(outtype,len,t);
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = VecLoad(v,t,newvec);
}

/* Be to keep vec/examples/ex21.F and snes/examples/ex12.F up to date */
void PETSC_STDCALL vecrestorearray_(Vec *x,PetscScalar *fa,size_t *ia,PetscErrorCode *ierr)
{
  PetscInt    m;
  PetscScalar *lx;

  *ierr = VecGetLocalSize(*x,&m);if (*ierr) return;
  *ierr = PetscScalarAddressFromFortran((PetscObject)*x,fa,*ia,m,&lx);if (*ierr) return;
  *ierr = VecRestoreArray(*x,&lx);if (*ierr) return;
}

void PETSC_STDCALL vecgetarray_(Vec *x,PetscScalar *fa,size_t *ia,PetscErrorCode *ierr)
{
  PetscScalar *lx;
  PetscInt    m;

  *ierr = VecGetArray(*x,&lx); if (*ierr) return;
  *ierr = VecGetLocalSize(*x,&m);if (*ierr) return;
  *ierr = PetscScalarAddressToFortran((PetscObject)*x,fa,lx,m,ia);
}

void PETSC_STDCALL vecscatterdestroy_(VecScatter *ctx,PetscErrorCode *ierr)
{
  *ierr = VecScatterDestroy(*ctx);
}

void PETSC_STDCALL vecdestroy_(Vec *v,PetscErrorCode *ierr)
{
  *ierr = VecDestroy(*v);
}

void PETSC_STDCALL vecscattercreate_(Vec *xin,IS *ix,Vec *yin,IS *iy,VecScatter *newctx,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECTDEREFERENCE(ix);
  CHKFORTRANNULLOBJECTDEREFERENCE(iy);
  *ierr = VecScatterCreate(*xin,*ix,*yin,*iy,newctx);
}

void PETSC_STDCALL vecscattercopy_(VecScatter *sctx,VecScatter *ctx,PetscErrorCode *ierr)
{
  *ierr = VecScatterCopy(*sctx,ctx);
}

void PETSC_STDCALL mapcreatempi_(MPI_Comm *comm,PetscInt *n,PetscInt *N,PetscMap *vv,PetscErrorCode *ierr)
{
  *ierr = PetscMapCreateMPI((MPI_Comm)PetscToPointerComm(*comm),*n,*N,vv);
}

void PETSC_STDCALL veccreatempi_(MPI_Comm *comm,PetscInt *n,PetscInt *N,Vec *vv,PetscErrorCode *ierr)
{
  *ierr = VecCreateMPI((MPI_Comm)PetscToPointerComm(*comm),*n,*N,vv);
}

void PETSC_STDCALL veccreateshared_(MPI_Comm *comm,PetscInt *n,PetscInt *N,Vec *vv,PetscErrorCode *ierr)
{
  *ierr = VecCreateShared((MPI_Comm)PetscToPointerComm(*comm),*n,*N,vv);
}

void PETSC_STDCALL veccreateseq_(MPI_Comm *comm,PetscInt *n,Vec *V,PetscErrorCode *ierr)
{
  *ierr = VecCreateSeq((MPI_Comm)PetscToPointerComm(*comm),*n,V);
}

void PETSC_STDCALL veccreateseqwitharray_(MPI_Comm *comm,PetscInt *n,PetscScalar *s,Vec *V,PetscErrorCode *ierr)
{
  CHKFORTRANNULLSCALAR(s);
  *ierr = VecCreateSeqWithArray((MPI_Comm)PetscToPointerComm(*comm),*n,s,V);
}

void PETSC_STDCALL veccreatempiwitharray_(MPI_Comm *comm,PetscInt *n,PetscInt *N,PetscScalar *s,Vec *V,PetscErrorCode *ierr)
{
  CHKFORTRANNULLSCALAR(s);
  *ierr = VecCreateMPIWithArray((MPI_Comm)PetscToPointerComm(*comm),*n,*N,s,V);
}

void PETSC_STDCALL veccreate_(MPI_Comm *comm,Vec *V,PetscErrorCode *ierr)
{
  *ierr = VecCreate((MPI_Comm)PetscToPointerComm(*comm),V);
}

void PETSC_STDCALL vecduplicate_(Vec *v,Vec *newv,PetscErrorCode *ierr)
{
  *ierr = VecDuplicate(*v,newv);
}

/*
      vecduplicatevecs() and vecdestroyvecs() are slightly different from C since the 
    Fortran provides the array to hold the vector objects,while in C that 
    array is allocated by the VecDuplicateVecs()
*/
void PETSC_STDCALL vecduplicatevecs_(Vec *v,PetscInt *m,Vec *newv,PetscErrorCode *ierr)
{
  Vec *lV;
  PetscInt i;
  *ierr = VecDuplicateVecs(*v,*m,&lV); if (*ierr) return;
  for (i=0; i<*m; i++) {
    newv[i] = lV[i];
  }
  *ierr = PetscFree(lV); 
}

void PETSC_STDCALL vecdestroyvecs_(Vec *vecs,PetscInt *m,PetscErrorCode *ierr)
{
  PetscInt i;
  for (i=0; i<*m; i++) {
    *ierr = VecDestroy(vecs[i]);if (*ierr) return;
  }
}

void PETSC_STDCALL vecmtdot_(PetscInt *nv,Vec *x,Vec *y,PetscScalar *val,PetscErrorCode *ierr)
{
  *ierr = VecMTDot(*nv,*x,y,val);
}

void PETSC_STDCALL vecmdot_(PetscInt *nv,Vec *x,Vec *y,PetscScalar *val,PetscErrorCode *ierr)
{
  *ierr = VecMDot(*nv,*x,y,val);
}

void PETSC_STDCALL vecmaxpy_(Vec *y,PetscInt *nv,PetscScalar *alpha,Vec *x,PetscErrorCode *ierr)
{
  *ierr = VecMAXPY(*y,*nv,alpha,x);
}

void PETSC_STDCALL vecstridenorm_(Vec *x,PetscInt *start,NormType *type,PetscReal *val,PetscErrorCode *ierr)
{
  *ierr = VecStrideNorm(*x,*start,*type,val);
}

/* ----------------------------------------------------------------------------------------------*/
void PETSC_STDCALL veccreateghostblockwitharray_(MPI_Comm *comm,PetscInt *bs,PetscInt *n,PetscInt *N,PetscInt *nghost,PetscInt *ghosts,
                              PetscScalar *array,Vec *vv,PetscErrorCode *ierr)
{
  CHKFORTRANNULLSCALAR(array);
  *ierr = VecCreateGhostBlockWithArray((MPI_Comm)PetscToPointerComm(*comm),*bs,*n,*N,*nghost,
                                    ghosts,array,vv);
}

void PETSC_STDCALL veccreateghostblock_(MPI_Comm *comm,PetscInt *bs,PetscInt *n,PetscInt *N,PetscInt *nghost,PetscInt *ghosts,Vec *vv,
                          PetscErrorCode *ierr)
{
  *ierr = VecCreateGhostBlock((MPI_Comm)PetscToPointerComm(*comm),*bs,*n,*N,*nghost,ghosts,vv);
}

void PETSC_STDCALL veccreateghostwitharray_(MPI_Comm *comm,PetscInt *n,PetscInt *N,PetscInt *nghost,PetscInt *ghosts,PetscScalar *array,
                              Vec *vv,PetscErrorCode *ierr)
{
  CHKFORTRANNULLSCALAR(array);
  *ierr = VecCreateGhostWithArray((MPI_Comm)PetscToPointerComm(*comm),*n,*N,*nghost,
                                    ghosts,array,vv);
}

void PETSC_STDCALL veccreateghost_(MPI_Comm *comm,PetscInt *n,PetscInt *N,PetscInt *nghost,PetscInt *ghosts,Vec *vv,PetscErrorCode *ierr)
{
  *ierr = VecCreateGhost((MPI_Comm)PetscToPointerComm(*comm),*n,*N,*nghost,ghosts,vv);
}

void PETSC_STDCALL vecghostgetlocalform_(Vec *g,Vec *l,PetscErrorCode *ierr)
{
  *ierr = VecGhostGetLocalForm(*g,l);
}

void PETSC_STDCALL vecghostrestorelocalform_(Vec *g,Vec *l,PetscErrorCode *ierr)
{
  *ierr = VecGhostRestoreLocalForm(*g,l);
}

void PETSC_STDCALL vecmax_(Vec *x,PetscInt *p,PetscReal *val,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(p);
  *ierr = VecMax(*x,p,val);
}

void PETSC_STDCALL vecscattercreatetoall_(Vec *v,VecScatter *ctx,Vec *newv,PetscErrorCode *ierr)
{
  *ierr = VecScatterCreateToAll(*v,ctx,newv);
}

void PETSC_STDCALL vecscattercreatetozero_(Vec *v,VecScatter *ctx,Vec *newv,PetscErrorCode *ierr)
{
  *ierr = VecScatterCreateToZero(*v,ctx,newv);
}

void PETSC_STDCALL vecgetownershiprange_(Vec *x,PetscInt *low,PetscInt *high, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(low);
  CHKFORTRANNULLINTEGER(high);
  *ierr = VecGetOwnershipRange(*x,low,high);
}

EXTERN_C_END


