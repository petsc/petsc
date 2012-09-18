/*
  This file contains Fortran stubs for Options routines.
  These are not generated automatically since they require passing strings
  between Fortran and C.
*/

#include <petsc-private/fortranimpl.h>

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petscoptionsgetenumprivate_        PETSCOPTIONSGETENUMPRIVATE
#define petscoptionsgetbool_               PETSCOPTIONSGETBOOL
#define petscoptionsgetintarray_           PETSCOPTIONSGETINTARRAY
#define petscoptionssetvalue_              PETSCOPTIONSSETVALUE
#define petscoptionsclearvalue_            PETSCOPTIONSCLEARVALUE
#define petscoptionshasname_               PETSCOPTIONSHASNAME
#define petscoptionsgetint_                PETSCOPTIONSGETINT
#define petscoptionsgetreal_               PETSCOPTIONSGETREAL
#define petscoptionsgetrealarray_          PETSCOPTIONSGETREALARRAY
#define petscoptionsgetstring_             PETSCOPTIONSGETSTRING
#define petscgetprogramname                PETSCGETPROGRAMNAME
#define petscoptionsinsertfile_            PETSCOPTIONSINSERTFILE
#define petscoptionsclear_                 PETSCOPTIONSCLEAR
#define petscoptionsinsertstring_          PETSCOPTIONSINSERTSTRING
#define petscoptionsview_                  PETSCOPTIONSVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscoptionsgetenumprivate_        petscoptionsgetenumprivate
#define petscoptionsgetbool_               petscoptionsgetbool
#define petscoptionssetvalue_              petscoptionssetvalue
#define petscoptionsclearvalue_            petscoptionsclearvalue
#define petscoptionshasname_               petscoptionshasname
#define petscoptionsgetint_                petscoptionsgetint
#define petscoptionsgetreal_               petscoptionsgetreal
#define petscoptionsgetrealarray_          petscoptionsgetrealarray
#define petscoptionsgetstring_             petscoptionsgetstring
#define petscoptionsgetintarray_           petscoptionsgetintarray
#define petscgetprogramname_               petscgetprogramname
#define petscoptionsinsertfile_            petscoptionsinsertfile
#define petscoptionsclear_                 petscoptionsclear
#define petscoptionsinsertstring_          petscoptionsinsertstring
#define petscoptionsview_                  petscoptionsview
#endif

EXTERN_C_BEGIN

/* ---------------------------------------------------------------------*/

void PETSC_STDCALL petscoptionsinsertstring_(CHAR file PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;

  FIXCHAR(file,len,c1);
  *ierr = PetscOptionsInsertString(c1);
  FREECHAR(file,c1);
}

void PETSC_STDCALL petscoptionsinsertfile_(MPI_Fint *comm,CHAR file PETSC_MIXED_LEN(len),PetscBool  *require,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;

  FIXCHAR(file,len,c1);
  *ierr = PetscOptionsInsertFile(MPI_Comm_f2c(*comm),c1,*require);
  FREECHAR(file,c1);
}

void PETSC_STDCALL petscoptionssetvalue_(CHAR name PETSC_MIXED_LEN(len1),CHAR value PETSC_MIXED_LEN(len2),
                   PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *c1,*c2;

  FIXCHAR(name,len1,c1);
  FIXCHAR(value,len2,c2);
  *ierr = PetscOptionsSetValue(c1,c2);
  FREECHAR(name,c1);
  FREECHAR(value,c2);
}

void PETSC_STDCALL petscoptionsclear_(PetscErrorCode *ierr)
{
  *ierr = PetscOptionsClear();
}

void PETSC_STDCALL petscoptionsclearvalue_(CHAR name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;

  FIXCHAR(name,len,c1);
  *ierr = PetscOptionsClearValue(c1);
  FREECHAR(name,c1);
}

void PETSC_STDCALL petscoptionshasname_(CHAR pre PETSC_MIXED_LEN(len1),CHAR name PETSC_MIXED_LEN(len2),
                    PetscBool  *flg,PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *c1,*c2;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  *ierr = PetscOptionsHasName(c1,c2,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

void PETSC_STDCALL petscoptionsgetint_(CHAR pre PETSC_MIXED_LEN(len1),CHAR name PETSC_MIXED_LEN(len2),
                    PetscInt *ivalue,PetscBool  *flg,PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *c1,*c2;
  PetscBool  flag;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  *ierr = PetscOptionsGetInt(c1,c2,ivalue,&flag);
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

void PETSC_STDCALL petscoptionsgetenumprivate_(CHAR pre PETSC_MIXED_LEN(len1),CHAR name PETSC_MIXED_LEN(len2),const char *const*list,
                    PetscEnum *ivalue,PetscBool  *flg,PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *c1,*c2;
  PetscBool  flag;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  *ierr = PetscOptionsGetEnum(c1,c2,list,ivalue,&flag);
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

void PETSC_STDCALL petscoptionsgetbool_(CHAR pre PETSC_MIXED_LEN(len1),CHAR name PETSC_MIXED_LEN(len2),
                    PetscBool  *ivalue,PetscBool  *flg,PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *c1,*c2;
  PetscBool  flag;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  *ierr = PetscOptionsGetBool(c1,c2,ivalue,&flag);
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

void PETSC_STDCALL petscoptionsgetreal_(CHAR pre PETSC_MIXED_LEN(len1),CHAR name PETSC_MIXED_LEN(len2),
                    PetscReal *dvalue,PetscBool  *flg,PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *c1,*c2;
  PetscBool  flag;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  *ierr = PetscOptionsGetReal(c1,c2,dvalue,&flag);
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

void PETSC_STDCALL petscoptionsgetrealarray_(CHAR pre PETSC_MIXED_LEN(len1),CHAR name PETSC_MIXED_LEN(len2),
                PetscReal *dvalue,PetscInt *nmax,PetscBool  *flg,PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *c1,*c2;
  PetscBool  flag;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  *ierr = PetscOptionsGetRealArray(c1,c2,dvalue,nmax,&flag);
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

void PETSC_STDCALL petscoptionsgetintarray_(CHAR pre PETSC_MIXED_LEN(len1),CHAR name PETSC_MIXED_LEN(len2),
                   PetscInt *dvalue,PetscInt *nmax,PetscBool  *flg,PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *c1,*c2;
  PetscBool  flag;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  *ierr = PetscOptionsGetIntArray(c1,c2,dvalue,nmax,&flag);
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

void PETSC_STDCALL petscoptionsgetstring_(CHAR pre PETSC_MIXED_LEN(len1),CHAR name PETSC_MIXED_LEN(len2),
                    CHAR string PETSC_MIXED_LEN(len),PetscBool  *flg,
                    PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2) PETSC_END_LEN(len))
{
  char *c1,*c2,*c3;
  size_t len3;
  PetscBool  flag;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  c3   = string;
  len3 = len - 1;

  *ierr = PetscOptionsGetString(c1,c2,c3,len3,&flag);
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
  FIXRETURNCHAR(flag,string,len);
}

void PETSC_STDCALL petscgetprogramname_(CHAR name PETSC_MIXED_LEN(len_in),PetscErrorCode *ierr PETSC_END_LEN(len_in))
{
  char *tmp;
  size_t len;
  tmp = name;
  len = len_in - 1;
  *ierr = PetscGetProgramName(tmp,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len_in);
}

void PETSC_STDCALL petscoptionsview_(PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = PetscOptionsView(v);
}

EXTERN_C_END

