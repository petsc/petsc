#include <petsc/private/fortranimpl.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscviewersetformat_        PETSCVIEWERSETFORMAT
#define petscviewersettype_          PETSCVIEWERSETTYPE
#define petscviewergettype_          PETSCVIEWERGETTYPE
#define petscviewerpushformat_       PETSCVIEWERPUSHFORMAT
#define petscviewerpopformat_        PETSCVIEWERPOPFORMAT
#define petscviewerandformatcreate_  PETSCVIEWERANDFORMATCREATE
#define petscviewerandformatdestroy_ PETSCVIEWERANDFORMATDESTROY
#define petscviewergetsubviewer_     PETSCVIEWERGETSUBVIEWER
#define petscviewerrestoresubviewer_ PETSCVIEWERRESTORESUBVIEWER
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscviewersetformat_        petscviewersetformat
#define petscviewersettype_          petscviewersettype
#define petscviewergettype_          petscviewergettype
#define petscviewerpushformat_       petscviewerpushformat
#define petscviewerpopformat_        petscviewerpopformat
#define petscviewerandformatcreate_  petscviewerandformatcreate
#define petscviewerandformatdestroy_ petscviewerandformatdestroy
#define petscviewergetsubviewer_     petscviewergetsubviewer
#define petscviewerrestoresubviewer_ petscviewerrestoresubviewer
#endif

PETSC_EXTERN void PETSC_STDCALL  petscviewergetsubviewer_(PetscViewer *vin,MPI_Fint * comm,PetscViewer *outviewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = PetscViewerGetSubViewer(v,MPI_Comm_f2c(*(comm)),outviewer);
}

PETSC_EXTERN void PETSC_STDCALL  petscviewerrestoresubviewer_(PetscViewer *vin,MPI_Fint * comm,PetscViewer *outviewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = PetscViewerRestoreSubViewer(v,MPI_Comm_f2c(*(comm)),outviewer);
}

PETSC_EXTERN PetscErrorCode PetscViewerSetFormatDeprecated(PetscViewer, PetscViewerFormat);

PETSC_EXTERN void PETSC_STDCALL petscviewerandformatcreate_(PetscViewer *vin, PetscViewerFormat *format, PetscViewerAndFormat **vf, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = PetscViewerAndFormatCreate(v, *format, vf);
}

PETSC_EXTERN void petscviewerandformatdestroy_(PetscViewerAndFormat **vf, PetscErrorCode *ierr)
{
  *ierr = PetscViewerAndFormatDestroy(vf);
}

PETSC_EXTERN void PETSC_STDCALL petscviewersetformat_(PetscViewer *vin, PetscViewerFormat *format, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = PetscViewerSetFormatDeprecated(v, *format);
}

PETSC_EXTERN void PETSC_STDCALL petscviewersettype_(PetscViewer *x, char* type_name PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t;

  FIXCHAR(type_name, len, t);
  *ierr = PetscViewerSetType(*x, t);
  FREECHAR(type_name, t);
}

PETSC_EXTERN void PETSC_STDCALL petscviewergettype_(PetscViewer *viewer, char* type PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))
{
   const char *c1;

   *ierr = PetscViewerGetType(*viewer, &c1);
   *ierr = PetscStrncpy(type, c1, len);
   FIXRETURNCHAR(PETSC_TRUE, type, len);
}

PETSC_EXTERN void PETSC_STDCALL petscviewerpushformat_(PetscViewer *vin, PetscViewerFormat *format, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = PetscViewerPushFormat(v, *format);
}

PETSC_EXTERN void PETSC_STDCALL petscviewerpopformat_(PetscViewer *vin, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = PetscViewerPopFormat(v);
}
