/*
  This file contains Fortran stubs for Options routines.
  These are not generated automatically since they require passing strings
  between Fortran and C.
*/

#include <petsc/private/fortranimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscoptionsgetenumprivate_        PETSCOPTIONSGETENUMPRIVATE
#define petscoptionsgetbool_               PETSCOPTIONSGETBOOL
#define petscoptionsgetintarray_           PETSCOPTIONSGETINTARRAY
#define petscoptionssetvalue_              PETSCOPTIONSSETVALUE
#define petscoptionsclearvalue_            PETSCOPTIONSCLEARVALUE
#define petscoptionshasname_               PETSCOPTIONSHASNAME
#define petscoptionsgetint_                PETSCOPTIONSGETINT
#define petscoptionsgetreal_               PETSCOPTIONSGETREAL
#define petscoptionsgetscalar_             PETSCOPTIONSGETSCALAR
#define petscoptionsgetrealarray_          PETSCOPTIONSGETREALARRAY
#define petscoptionsgetstring_             PETSCOPTIONSGETSTRING
#define petscgetprogramname                PETSCGETPROGRAMNAME
#define petscoptionsinsertfile_            PETSCOPTIONSINSERTFILE
#define petscoptionsclear_                 PETSCOPTIONSCLEAR
#define petscoptionsinsertstring_          PETSCOPTIONSINSERTSTRING
#define petscoptionsview_                  PETSCOPTIONSVIEW
#define petscoptionsleft_                  PETSCOPTIONSLEFT
#define petscobjectviewfromoptions_        PETSCOBJECTVIEWFROMOPTIONS
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscoptionsgetenumprivate_        petscoptionsgetenumprivate
#define petscoptionsgetbool_               petscoptionsgetbool
#define petscoptionssetvalue_              petscoptionssetvalue
#define petscoptionsclearvalue_            petscoptionsclearvalue
#define petscoptionshasname_               petscoptionshasname
#define petscoptionsgetint_                petscoptionsgetint
#define petscoptionsgetreal_               petscoptionsgetreal
#define petscoptionsgetscalar_             petscoptionsgetscalar
#define petscoptionsgetrealarray_          petscoptionsgetrealarray
#define petscoptionsgetstring_             petscoptionsgetstring
#define petscoptionsgetintarray_           petscoptionsgetintarray
#define petscgetprogramname_               petscgetprogramname
#define petscoptionsinsertfile_            petscoptionsinsertfile
#define petscoptionsclear_                 petscoptionsclear
#define petscoptionsinsertstring_          petscoptionsinsertstring
#define petscoptionsview_                  petscoptionsview
#define petscoptionsleft_                  petscoptionsleft
#define petscobjectviewfromoptions_        petscobjectviewfromoptions
#endif

/* ---------------------------------------------------------------------*/

PETSC_EXTERN void PETSC_STDCALL petscoptionsinsertstring_(PetscOptions *options,char* file PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;

  FIXCHAR(file,len,c1);
  CHKFORTRANNULLOBJECTDEREFERENCE(options);
  *ierr = PetscOptionsInsertString(*options,c1);
  FREECHAR(file,c1);
}

PETSC_EXTERN void PETSC_STDCALL petscoptionsinsertfile_(MPI_Fint *comm,PetscOptions *options,char* file PETSC_MIXED_LEN(len),PetscBool *require,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;

  FIXCHAR(file,len,c1);
  CHKFORTRANNULLOBJECTDEREFERENCE(options);
  *ierr = PetscOptionsInsertFile(MPI_Comm_f2c(*comm),*options,c1,*require);
  FREECHAR(file,c1);
}

PETSC_EXTERN void PETSC_STDCALL petscoptionssetvalue_(PetscOptions *options,char* name PETSC_MIXED_LEN(len1),char* value PETSC_MIXED_LEN(len2),
                   PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *c1,*c2;

  FIXCHAR(name,len1,c1);
  FIXCHAR(value,len2,c2);
  CHKFORTRANNULLOBJECTDEREFERENCE(options);
  *ierr = PetscOptionsSetValue(*options,c1,c2);
  FREECHAR(name,c1);
  FREECHAR(value,c2);
}

PETSC_EXTERN void PETSC_STDCALL petscoptionsclear_(PetscOptions *options,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECTDEREFERENCE(options);
  *ierr = PetscOptionsClear(*options);
}

PETSC_EXTERN void PETSC_STDCALL petscoptionsclearvalue_(PetscOptions *options,char* name PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *c1;

  FIXCHAR(name,len,c1);
  CHKFORTRANNULLOBJECTDEREFERENCE(options);
  *ierr = PetscOptionsClearValue(*options,c1);
  FREECHAR(name,c1);
}

PETSC_EXTERN void PETSC_STDCALL petscoptionshasname_(PetscOptions *options,char* pre PETSC_MIXED_LEN(len1),char* name PETSC_MIXED_LEN(len2),
                    PetscBool  *flg,PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *c1,*c2;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  CHKFORTRANNULLOBJECTDEREFERENCE(options);
  *ierr = PetscOptionsHasName(*options,c1,c2,flg);
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

PETSC_EXTERN void PETSC_STDCALL petscoptionsgetint_(PetscOptions *opt,char* pre PETSC_MIXED_LEN(len1),char* name PETSC_MIXED_LEN(len2),
                    PetscInt *ivalue,PetscBool  *flg,PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char      *c1,*c2;
  PetscBool flag;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  CHKFORTRANNULLOBJECTDEREFERENCE(opt);
  *ierr = PetscOptionsGetInt(*opt,c1,c2,ivalue,&flag);
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

PETSC_EXTERN void PETSC_STDCALL petscoptionsgetenumprivate_(PetscOptions *options,char* pre PETSC_MIXED_LEN(len1),char* name PETSC_MIXED_LEN(len2),const char *const*list,
                    PetscEnum *ivalue,PetscBool  *flg,PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char      *c1,*c2;
  PetscBool flag;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  CHKFORTRANNULLOBJECTDEREFERENCE(options);
  *ierr = PetscOptionsGetEnum(*options,c1,c2,list,ivalue,&flag);
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

PETSC_EXTERN void PETSC_STDCALL petscoptionsgetbool_(PetscOptions *options,char* pre PETSC_MIXED_LEN(len1),char* name PETSC_MIXED_LEN(len2),
                    PetscBool  *ivalue,PetscBool  *flg,PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char      *c1,*c2;
  PetscBool flag;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  CHKFORTRANNULLOBJECTDEREFERENCE(options);
  *ierr = PetscOptionsGetBool(*options,c1,c2,ivalue,&flag);
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

PETSC_EXTERN void PETSC_STDCALL petscoptionsgetreal_(PetscOptions *options,char* pre PETSC_MIXED_LEN(len1),char* name PETSC_MIXED_LEN(len2),
                    PetscReal *dvalue,PetscBool  *flg,PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *c1,*c2;
  PetscBool  flag;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  CHKFORTRANNULLOBJECTDEREFERENCE(options);
  *ierr = PetscOptionsGetReal(*options,c1,c2,dvalue,&flag);
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

PETSC_EXTERN void PETSC_STDCALL petscoptionsgetscalar_(PetscOptions *options,char* pre PETSC_MIXED_LEN(len1),char* name PETSC_MIXED_LEN(len2),
                    PetscScalar *dvalue,PetscBool  *flg,PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char *c1,*c2;
  PetscBool  flag;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  CHKFORTRANNULLOBJECTDEREFERENCE(options);
  *ierr = PetscOptionsGetScalar(*options,c1,c2,dvalue,&flag);
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

PETSC_EXTERN void PETSC_STDCALL petscoptionsgetrealarray_(PetscOptions *options,char* pre PETSC_MIXED_LEN(len1),char* name PETSC_MIXED_LEN(len2),
                PetscReal *dvalue,PetscInt *nmax,PetscBool  *flg,PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char      *c1,*c2;
  PetscBool flag;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  CHKFORTRANNULLOBJECTDEREFERENCE(options);
  *ierr = PetscOptionsGetRealArray(*options,c1,c2,dvalue,nmax,&flag);
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

PETSC_EXTERN void PETSC_STDCALL petscoptionsgetintarray_(PetscOptions *options,char* pre PETSC_MIXED_LEN(len1),char* name PETSC_MIXED_LEN(len2),
                   PetscInt *dvalue,PetscInt *nmax,PetscBool  *flg,PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char      *c1,*c2;
  PetscBool flag;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  CHKFORTRANNULLOBJECTDEREFERENCE(options);
  *ierr = PetscOptionsGetIntArray(*options,c1,c2,dvalue,nmax,&flag);
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
}

PETSC_EXTERN void PETSC_STDCALL petscoptionsgetstring_(PetscOptions *options,char* pre PETSC_MIXED_LEN(len1),char* name PETSC_MIXED_LEN(len2),
                    char* string PETSC_MIXED_LEN(len),PetscBool  *flg,
                    PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2) PETSC_END_LEN(len))
{
  char      *c1,*c2,*c3;
  size_t    len3;
  PetscBool flag;

  FIXCHAR(pre,len1,c1);
  FIXCHAR(name,len2,c2);
  c3   = string;
  len3 = len - 1;

  CHKFORTRANNULLOBJECTDEREFERENCE(options);
  *ierr = PetscOptionsGetString(*options,c1,c2,c3,len3,&flag);
  if (!FORTRANNULLBOOL(flg)) *flg = flag;
  FREECHAR(pre,c1);
  FREECHAR(name,c2);
  FIXRETURNCHAR(flag,string,len);
}

PETSC_EXTERN void PETSC_STDCALL petscgetprogramname_(char* name PETSC_MIXED_LEN(len_in),PetscErrorCode *ierr PETSC_END_LEN(len_in))
{
  char   *tmp;
  size_t len;
  tmp   = name;
  len   = len_in - 1;
  *ierr = PetscGetProgramName(tmp,len);
  FIXRETURNCHAR(PETSC_TRUE,name,len_in);
}

PETSC_EXTERN void PETSC_STDCALL petscoptionsview_(PetscOptions *options,PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;

  PetscPatchDefaultViewers_Fortran(vin,v);
  CHKFORTRANNULLOBJECTDEREFERENCE(options);
  *ierr = PetscOptionsView(*options,v);
}

PETSC_EXTERN void PETSC_STDCALL petscoptionsleft_(PetscOptions *options,PetscErrorCode *ierr)
{
  CHKFORTRANNULLOBJECTDEREFERENCE(options);
  *ierr = PetscOptionsLeft(*options);
}


PETSC_EXTERN void PETSC_STDCALL petscobjectviewfromoptions_(PetscObject *obj,PetscObject *bobj,char* option PETSC_MIXED_LEN(loption),PetscErrorCode *ierr  PETSC_END_LEN(loption))
{
  char *o;

  FIXCHAR(option, loption, o);
  CHKFORTRANNULLOBJECTDEREFERENCE(bobj);
  *ierr = PetscObjectViewFromOptions(*obj, *bobj, o);
  FREECHAR(option, o);
}
