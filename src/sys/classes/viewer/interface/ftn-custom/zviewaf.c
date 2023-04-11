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
  #define petscviewierview_            PETSCVIEWERVIEW
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
  #define petscviewierview_            petscviewerview
#endif

PETSC_EXTERN void petscviewergetsubviewer_(PetscViewer *vin, MPI_Fint *comm, PetscViewer *outviewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = PetscViewerGetSubViewer(v, MPI_Comm_f2c(*(comm)), outviewer);
}

PETSC_EXTERN void petscviewerrestoresubviewer_(PetscViewer *vin, MPI_Fint *comm, PetscViewer *outviewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = PetscViewerRestoreSubViewer(v, MPI_Comm_f2c(*(comm)), outviewer);
}

PETSC_EXTERN PetscErrorCode PetscViewerSetFormatDeprecated(PetscViewer, PetscViewerFormat);

PETSC_EXTERN void petscviewerandformatcreate_(PetscViewer *vin, PetscViewerFormat *format, PetscViewerAndFormat **vf, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = PetscViewerAndFormatCreate(v, *format, vf);
}

PETSC_EXTERN void petscviewerandformatdestroy_(PetscViewerAndFormat **vf, PetscErrorCode *ierr)
{
  *ierr = PetscViewerAndFormatDestroy(vf);
}

PETSC_EXTERN void petscviewersetformat_(PetscViewer *vin, PetscViewerFormat *format, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = PetscViewerSetFormatDeprecated(v, *format);
}

PETSC_EXTERN void petscviewersettype_(PetscViewer *x, char *type_name, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  char *t;

  FIXCHAR(type_name, len, t);
  *ierr = PetscViewerSetType(*x, t);
  if (*ierr) return;
  FREECHAR(type_name, t);
}

PETSC_EXTERN void petscviewergettype_(PetscViewer *viewer, char *type, PetscErrorCode *ierr, PETSC_FORTRAN_CHARLEN_T len)
{
  const char *c1;

  *ierr = PetscViewerGetType(*viewer, &c1);
  *ierr = PetscStrncpy(type, c1, len);
  FIXRETURNCHAR(PETSC_TRUE, type, len);
}

PETSC_EXTERN void petscviewerpushformat_(PetscViewer *vin, PetscViewerFormat *format, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = PetscViewerPushFormat(v, *format);
}

PETSC_EXTERN void petscviewerpopformat_(PetscViewer *vin, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin, v);
  *ierr = PetscViewerPopFormat(v);
}

PETSC_EXTERN void petscviewerview_(PetscViewer *vin, PetscViewer *viewerin, PetscErrorCode *ierr)
{
  PetscViewer v, viewer;
  PetscPatchDefaultViewers_Fortran(vin, v);
  PetscPatchDefaultViewers_Fortran(viewerin, viewer);
  *ierr = PetscViewerView(v, viewer);
}
