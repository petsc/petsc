
#include "zpetsc.h"
#include "petscvec.h"
#ifdef PETSC_HAVE_FORTRAN_CAPS
#define vecsettype_               VECSETTYPE
#define vecsetvalue_              VECSETVALUE
#define veccreateseqwitharray_    VECCREATESEQWITHARRAY
#define veccreatempiwitharray_    VECCREATEMPIWITHARRAY
#define vecscattercreate_         VECSCATTERCREATE
#define vecdestroyvecs_           VECDESTROYVECS
#define vecrestorearray_          VECRESTOREARRAY
#define vecgetarray_              VECGETARRAY
#define vecload_                  VECLOAD
#define vecgettype_               VECGETTYPE
#define vecduplicatevecs_         VECDUPLICATEVECS
#define vecview_                  VECVIEW
#define veccreateghostwitharray_  VECCREATEGHOSTWITHARRAY
#define vecmax_                   VECMAX
#define petscdrawtensorcontour_   PETSCDRAWTENSORCONTOUR
#define veccreateghostblockwitharray_ VECCREATEGHOSTBLOCKWITHARRAY
#define vecloadintovector_            VECLOADINTOVECTOR  
#define vecgetownershiprange_         VECGETOWNERSHIPRANGE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define vecloadintovector_            vecloadintovector
#define veccreateghostblockwitharray_ veccreateghostblockwitharray
#define petscdrawtensorcontour_   petscdrawtensorcontour
#define vecsettype_               vecsettype
#define veccreateghostwitharray_  veccreateghostwitharray
#define vecsetvalue_              vecsetvalue
#define vecview_                  vecview
#define veccreateseqwitharray_    veccreateseqwitharray
#define veccreatempiwitharray_    veccreatempiwitharray
#define vecscattercreate_         vecscattercreate
#define vecdestroyvecs_           vecdestroyvecs
#define vecrestorearray_          vecrestorearray
#define vecgetarray_              vecgetarray
#define vecload_                  vecload
#define vecgettype_               vecgettype
#define vecduplicatevecs_         vecduplicatevecs
#define vecmax_                   vecmax
#define vecgetownershiprange_     vecgetownershiprange
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL vecloadintovector_(PetscViewer *viewer,Vec *vec,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = VecLoadIntoVector(v,*vec);
}

void PETSC_STDCALL petscdrawtensorcontour_(PetscDraw *win,int *m,int *n,PetscReal *x,PetscReal *y,PetscReal *V,PetscErrorCode *ierr)
{
  CHKFORTRANNULLDOUBLE(x);
  CHKFORTRANNULLDOUBLE(y);
  *ierr = PetscDrawTensorContour(*win,*m,*n,x,y,V);
}

void PETSC_STDCALL vecsettype_(Vec *x,CHAR type_name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type_name,len,t);
  *ierr = VecSetType(*x,t);
  FREECHAR(type_name,t);
}

void PETSC_STDCALL vecsetvalue_(Vec *v,PetscInt *i,PetscScalar *va,InsertMode *mode,PetscErrorCode *ierr)
{
  /* cannot use VecSetValue() here since that uses CHKERRQ() which has a return in it */
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
void PETSC_STDCALL vecrestorearray_(Vec *x,PetscScalar *fa,PetscInt *ia,PetscErrorCode *ierr)
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

void PETSC_STDCALL vecscattercreate_(Vec *xin,IS *ix,Vec *yin,IS *iy,VecScatter *newctx,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECTDEREFERENCE(ix);
  CHKFORTRANNULLOBJECTDEREFERENCE(iy);
  *ierr = VecScatterCreate(*xin,*ix,*yin,*iy,newctx);
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

/* ----------------------------------------------------------------------------------------------*/
void PETSC_STDCALL veccreateghostblockwitharray_(MPI_Comm *comm,PetscInt *bs,PetscInt *n,PetscInt *N,PetscInt *nghost,PetscInt *ghosts,
                              PetscScalar *array,Vec *vv,PetscErrorCode *ierr)
{
  CHKFORTRANNULLSCALAR(array);
  *ierr = VecCreateGhostBlockWithArray((MPI_Comm)PetscToPointerComm(*comm),*bs,*n,*N,*nghost,
                                    ghosts,array,vv);
}

void PETSC_STDCALL veccreateghostwitharray_(MPI_Comm *comm,PetscInt *n,PetscInt *N,PetscInt *nghost,PetscInt *ghosts,PetscScalar *array,
                              Vec *vv,PetscErrorCode *ierr)
{
  CHKFORTRANNULLSCALAR(array);
  *ierr = VecCreateGhostWithArray((MPI_Comm)PetscToPointerComm(*comm),*n,*N,*nghost,
                                    ghosts,array,vv);
}

void PETSC_STDCALL vecmax_(Vec *x,PetscInt *p,PetscReal *val,PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(p);
  *ierr = VecMax(*x,p,val);
}

void PETSC_STDCALL vecgetownershiprange_(Vec *x,PetscInt *low,PetscInt *high, PetscErrorCode *ierr)
{
  CHKFORTRANNULLINTEGER(low);
  CHKFORTRANNULLINTEGER(high);
  *ierr = VecGetOwnershipRange(*x,low,high);
}

EXTERN_C_END


