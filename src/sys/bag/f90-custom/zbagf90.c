#include "../src/sys/f90-src/f90impl.h"
#include "private/fortranimpl.h"
#include "petscbag.h"
#include "../src/sys/bag/bagimpl.h"

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscbagdestroy_ PETSCBAGDESTROY
#define petscbagview_ PETSCBAGVIEW
#define petscbagload_ PETSCBAGLOAD
#define petscbaggetdata_ PETSCBAGGETDATA
#define petscbagregisterint_ PETSCBAGREGISTERINT
#define petscbagregisterscalar_ PETSCBAGREGISTERSCALAR
#define petscbagregisterstring_ PETSCBAGREGISTERSTRING
#define petscbagregisterreal_ PETSCBAGREGISTERREAL
#define petscbagregistertruth_ PETSCBAGREGISTERTRUTH
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscbagdestroy_ petscbagdestroy
#define petscbagview_ petscbagview
#define petscbagload_ petscbagload
#define petscbaggetdata_ petscbaggetdata
#define petscbagregisterint_ petscbagregisterint
#define petscbagregisterscalar_ petscbagregisterscalar
#define petscbagregisterstring_ petscbagregisterstring
#define petscbagregisterreal_ petscbagregisterreal
#define petscbagregistertruth_ petscbagregistertruth
#endif

EXTERN_C_BEGIN


void PETSC_STDCALL petscbagdestroy_(PetscBag *bag,PetscErrorCode *ierr)
{
  *ierr = PetscBagDestroy(*bag);
}

void PETSC_STDCALL petscbagview_(PetscBag *bag,PetscViewer *viewer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscBagView(*bag,v);
}

void PETSC_STDCALL petscbagload_(PetscViewer *viewer,PetscBag *bag,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscBagLoad(v,bag);
}

void PETSC_STDCALL petscbagregisterint_(PetscBag *bag,void *ptr,PetscInt *def,CHAR s1 PETSC_MIXED_LEN(l1),
					CHAR s2 PETSC_MIXED_LEN(l2),PetscErrorCode *ierr PETSC_END_LEN(l1) PETSC_END_LEN(l2))
{
  char *t1,*t2;
  FIXCHAR(s1,l1,t1);
  FIXCHAR(s2,l2,t2);
  *ierr = PetscBagRegisterInt(*bag,ptr,*def,t1,t2);
  FREECHAR(s1,t1);
  FREECHAR(s2,t2);
}

void PETSC_STDCALL petscbagregisterscalar_(PetscBag *bag,void *ptr,PetscScalar *def,CHAR s1 PETSC_MIXED_LEN(l1),
					CHAR s2 PETSC_MIXED_LEN(l2),PetscErrorCode *ierr PETSC_END_LEN(l1) PETSC_END_LEN(l2))
{
  char *t1,*t2;
  FIXCHAR(s1,l1,t1);
  FIXCHAR(s2,l2,t2);
  *ierr = PetscBagRegisterScalar(*bag,ptr,*def,t1,t2);
  FREECHAR(s1,t1);
  FREECHAR(s2,t2);
}

void PETSC_STDCALL petscbagregisterreal_(PetscBag *bag,void *ptr,PetscReal *def,CHAR s1 PETSC_MIXED_LEN(l1),
					CHAR s2 PETSC_MIXED_LEN(l2),PetscErrorCode *ierr PETSC_END_LEN(l1) PETSC_END_LEN(l2))
{
  char *t1,*t2;
  FIXCHAR(s1,l1,t1);
  FIXCHAR(s2,l2,t2);
  *ierr = PetscBagRegisterReal(*bag,ptr,*def,t1,t2);
  FREECHAR(s1,t1);
  FREECHAR(s2,t2);
}

void PETSC_STDCALL petscbagregistertruth_(PetscBag *bag,void *ptr,PetscTruth *def,CHAR s1 PETSC_MIXED_LEN(l1),
					CHAR s2 PETSC_MIXED_LEN(l2),PetscErrorCode *ierr PETSC_END_LEN(l1) PETSC_END_LEN(l2))
{
  char       *t1,*t2;
  PetscTruth flg = PETSC_FALSE;

  /* some Fortran compilers use -1 as boolean */
  if (*def) flg = PETSC_TRUE;
  FIXCHAR(s1,l1,t1);
  FIXCHAR(s2,l2,t2);
  *ierr = PetscBagRegisterTruth(*bag,ptr,flg,t1,t2);
  FREECHAR(s1,t1);
  FREECHAR(s2,t2);
}

void PETSC_STDCALL petscbagregisterstring_(PetscBag *bag,CHAR p PETSC_MIXED_LEN(pl),CHAR cs1 PETSC_MIXED_LEN(cl1),CHAR s1 PETSC_MIXED_LEN(l1),
					   CHAR s2 PETSC_MIXED_LEN(l2),PetscErrorCode *ierr PETSC_END_LEN(pl) PETSC_END_LEN(cl1) PETSC_END_LEN(l1) PETSC_END_LEN(l2))
{
  char *t1,*t2,*ct1;
  FIXCHAR(s1,l1,t1);
  FIXCHAR(cs1,cl1,ct1);
  FIXCHAR(s2,l2,t2);
  *ierr = PetscBagRegisterString(*bag,p,pl,ct1,t1,t2);
  FREECHAR(cs1,ct1);
  FREECHAR(s1,t1);
  FREECHAR(s2,t2);
}


void PETSC_STDCALL petscbaggetdata_(PetscBag *bag,void **data,PetscErrorCode *ierr)
{
  *ierr = PetscBagGetData(*bag,data);
}

EXTERN_C_END
