#include <petsc/private/f90impl.h>
#include <petsc/private/fortranimpl.h>
#include <petscbag.h>
#include <petsc/private/bagimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscbagdestroy_ PETSCBAGDESTROY
#define petscbagview_ PETSCBAGVIEW
#define petscbagload_ PETSCBAGLOAD
#define petscbaggetdata_ PETSCBAGGETDATA
#define petscbagregisterint_ PETSCBAGREGISTERINT
#define petscbagregisterint64_ PETSCBAGREGISTERINT64
#define petscbagregisterintarray_ PETSCBAGREGISTERINTARRAY
#define petscbagregisterscalar_ PETSCBAGREGISTERSCALAR
#define petscbagregisterstring_ PETSCBAGREGISTERSTRING
#define petscbagregisterreal_ PETSCBAGREGISTERREAL
#define petscbagregisterrealarray_ PETSCBAGREGISTERREALARRAY
#define petscbagregisterbool_ PETSCBAGREGISTERBOOL
#define petscbagregisterboolarray_ PETSCBAGREGISTERBOOLARRAY
#define petscbagsetname_ PETSCBAGSETNAME
#define petscbagsetoptionsprefix_ PETSCBAGSETOPTIONSPREFIX
#define petscbagcreate_ PETSCBAGCREATE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscbagdestroy_ petscbagdestroy
#define petscbagview_ petscbagview
#define petscbagload_ petscbagload
#define petscbaggetdata_ petscbaggetdata
#define petscbagregisterint_ petscbagregisterint
#define petscbagregisterint64_ petscbagregisterint64
#define petscbagregisterintarray_ petscbagregisterintarray
#define petscbagregisterscalar_ petscbagregisterscalar
#define petscbagregisterstring_ petscbagregisterstring
#define petscbagregisterreal_ petscbagregisterreal
#define petscbagregisterrealarray_ petscbagregisterrealarray
#define petscbagregisterbool_ petscbagregisterbool
#define petscbagregisterboolarray_ petscbagregisterboolarray
#define petscbagsetname_ petscbagsetname
#define petscbagsetoptionsprefix_ petscbagsetoptionsprefix
#define petscbagcreate_ petscbagcreate
#endif

PETSC_EXTERN void  petscbagcreate_(MPI_Fint * comm,size_t *bagsize,PetscBag *bag, PetscErrorCode *ierr )
{
  *ierr = PetscBagCreate(MPI_Comm_f2c(*(comm)),*bagsize,bag);
}

PETSC_EXTERN void petscbagdestroy_(PetscBag *bag,PetscErrorCode *ierr)
{
  *ierr = PetscBagDestroy(bag);
}

PETSC_EXTERN void petscbagview_(PetscBag *bag,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscBagView(*bag,v);
}

PETSC_EXTERN void petscbagload_(PetscViewer *viewer,PetscBag *bag,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscBagLoad(v,*bag);
}

PETSC_EXTERN void petscbagregisterint_(PetscBag *bag,void *ptr,PetscInt *def,char* s1,
                                        char* s2,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T l1,PETSC_FORTRAN_CHARLEN_T l2)
{
  char *t1,*t2;
  FIXCHAR(s1,l1,t1);
  FIXCHAR(s2,l2,t2);
  *ierr = PetscBagRegisterInt(*bag,ptr,*def,t1,t2);if (*ierr) return;
  FREECHAR(s1,t1);
  FREECHAR(s2,t2);
}

PETSC_EXTERN void petscbagregisterint64_(PetscBag *bag,void *ptr,PetscInt64 *def,char* s1,
                                        char* s2,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T l1,PETSC_FORTRAN_CHARLEN_T l2)
{
  char *t1,*t2;
  FIXCHAR(s1,l1,t1);
  FIXCHAR(s2,l2,t2);
  *ierr = PetscBagRegisterInt64(*bag,ptr,*def,t1,t2);if (*ierr) return;
  FREECHAR(s1,t1);
  FREECHAR(s2,t2);
}

PETSC_EXTERN void petscbagregisterintarray_(PetscBag *bag,void *ptr,PetscInt *msize,char* s1,
                                             char* s2,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T l1,PETSC_FORTRAN_CHARLEN_T l2)
{
  char *t1,*t2;
  FIXCHAR(s1,l1,t1);
  FIXCHAR(s2,l2,t2);
  *ierr = PetscBagRegisterIntArray(*bag,ptr,*msize,t1,t2);if (*ierr) return;
  FREECHAR(s1,t1);
  FREECHAR(s2,t2);
}

PETSC_EXTERN void petscbagregisterscalar_(PetscBag *bag,void *ptr,PetscScalar *def,char* s1,
                                           char* s2,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T l1,PETSC_FORTRAN_CHARLEN_T l2)
{
  char *t1,*t2;
  FIXCHAR(s1,l1,t1);
  FIXCHAR(s2,l2,t2);
  *ierr = PetscBagRegisterScalar(*bag,ptr,*def,t1,t2);if (*ierr) return;
  FREECHAR(s1,t1);
  FREECHAR(s2,t2);
}

PETSC_EXTERN void petscbagregisterreal_(PetscBag *bag,void *ptr,PetscReal *def,char* s1,
                                         char* s2,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T l1,PETSC_FORTRAN_CHARLEN_T l2)
{
  char *t1,*t2;
  FIXCHAR(s1,l1,t1);
  FIXCHAR(s2,l2,t2);
  *ierr = PetscBagRegisterReal(*bag,ptr,*def,t1,t2);if (*ierr) return;
  FREECHAR(s1,t1);
  FREECHAR(s2,t2);
}

PETSC_EXTERN void petscbagregisterrealarray_(PetscBag *bag,void *ptr,PetscInt *msize,char* s1,
                                              char* s2,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T l1,PETSC_FORTRAN_CHARLEN_T l2)
{
  char *t1,*t2;
  FIXCHAR(s1,l1,t1);
  FIXCHAR(s2,l2,t2);
  *ierr = PetscBagRegisterRealArray(*bag,ptr,*msize,t1,t2);if (*ierr) return;
  FREECHAR(s1,t1);
  FREECHAR(s2,t2);
}

PETSC_EXTERN void petscbagregisterbool_(PetscBag *bag,void *ptr,PetscBool *def,char* s1,
                                         char* s2,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T l1,PETSC_FORTRAN_CHARLEN_T l2)
{
  char       *t1,*t2;
  PetscBool  flg = PETSC_FALSE;

  /* some Fortran compilers use -1 as boolean */
  if (*def) flg = PETSC_TRUE;
  FIXCHAR(s1,l1,t1);
  FIXCHAR(s2,l2,t2);
  *ierr = PetscBagRegisterBool(*bag,ptr,flg,t1,t2);if (*ierr) return;
  FREECHAR(s1,t1);
  FREECHAR(s2,t2);
}

PETSC_EXTERN void petscbagregisterboolarray_(PetscBag *bag,void *ptr,PetscInt *msize,char* s1,
                                              char* s2,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T l1,PETSC_FORTRAN_CHARLEN_T l2)
{
  char       *t1,*t2;

  /* some Fortran compilers use -1 as boolean */
  FIXCHAR(s1,l1,t1);
  FIXCHAR(s2,l2,t2);
  *ierr = PetscBagRegisterBoolArray(*bag,ptr,*msize,t1,t2);if (*ierr) return;
  FREECHAR(s1,t1);
  FREECHAR(s2,t2);
}

PETSC_EXTERN void petscbagregisterstring_(PetscBag *bag,char* p,char* cs1,char* s1,
                                                        char* s2,PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T pl,PETSC_FORTRAN_CHARLEN_T cl1,PETSC_FORTRAN_CHARLEN_T l1,PETSC_FORTRAN_CHARLEN_T l2)
{
  char *t1,*t2,*ct1;
  FIXCHAR(s1,l1,t1);
  FIXCHAR(cs1,cl1,ct1);
  FIXCHAR(s2,l2,t2);
  *ierr = PetscBagRegisterString(*bag,p,pl,ct1,t1,t2);if (*ierr) return;
  FREECHAR(cs1,ct1);
  FREECHAR(s1,t1);
  FREECHAR(s2,t2);
}


PETSC_EXTERN void petscbaggetdata_(PetscBag *bag,void **data,PetscErrorCode *ierr)
{
  *ierr = PetscBagGetData(*bag,data);
}

PETSC_EXTERN void petscbagsetname_(PetscBag *bag,char* ns,char* hs, PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T nl,PETSC_FORTRAN_CHARLEN_T hl)
{
  char *nt,*ht;
  FIXCHAR(ns,nl,nt);
  FIXCHAR(hs,hl,ht);
  *ierr = PetscBagSetName(*bag,nt,ht);if (*ierr) return;
  FREECHAR(ns,nt);
  FREECHAR(hs,ht);
}

PETSC_EXTERN void petscbagsetoptionsprefix_(PetscBag *bag,char* pre, PetscErrorCode *ierr,PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;
  FIXCHAR(pre,len,t);
  *ierr = PetscBagSetOptionsPrefix(*bag,t);if (*ierr) return;
  FREECHAR(pre,t);
}
