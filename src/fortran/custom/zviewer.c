
#include "src/fortran/custom/zpetsc.h"
#include "petsc.h"

#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petscviewerdestroy_        PETSCVIEWERDESTROY
#define petscviewerasciiopen_      PETSCVIEWERASCIIOPEN
#define petscviewersetformat_      PETSCVIEWERSETFORMAT
#define petscviewerpushformat_     PETSCVIEWERPUSHFORMAT
#define petscviewerpopformat_      PETSCVIEWERPOPFORMAT
#define petscviewerbinaryopen_     PETSCVIEWERBINARYOPEN
#define petscviewermatlabopen_     PETSCVIEWERMATLABOPEN
#define petscviewersocketopen_     PETSCVIEWERSOCKETOPEN
#define petscviewerstringopen_     PETSCVIEWERSTRINGOPEN
#define petscviewerdrawopen_       PETSCVIEWERDRAWOPEN
#define petscviewersetfiletype_    PETSCVIEWERSETFILETYPE
#define petscviewersetfilename_    PETSCVIEWERSETFILENAME
#define petscviewersocketputscalar_ PETSCVIEWERSOCKETPUTSCALAR
#define petscviewersocketputint_    PETSCVIEWERSOCKETPUTINT
#define petscviewersocketputreal_   PETSCVIEWERSOCKETPUTREAL
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscviewersocketputscalar_ petscviewersocketputscalar
#define petscviewersocketputint_    petscviewersocketputint
#define petscviewersocketputreal_   petscviewersocketputreal
#define petscviewerdestroy_        petscviewerdestroy
#define petscviewerasciiopen_      petscviewerasciiopen
#define petscviewersetformat_      petscviewersetformat
#define petscviewerpushformat_     petscviewerpushformat
#define petscviewerpopformat_      petscviewerpopformat
#define petscviewerbinaryopen_     petscviewerbinaryopen
#define petscviewermatlabopen_     petscviewermatlabopen
#define petscviewersocketopen_     petscviewersocketopen
#define petscviewerstringopen_     petscviewerstringopen
#define petscviewerdrawopen_       petscviewerdrawopen
#define petscviewersetfiletype_    petscviewersetfiletype
#define petscviewersetfilename_    petscviewersetfilename
#endif

EXTERN_C_BEGIN

#if defined(PETSC_HAVE_SOCKET)
void PETSC_STDCALL petscviewersocketputscalar(PetscViewer *viewer,PetscInt *m,PetscInt *n,PetscScalar *s,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerSocketPutScalar(v,*m,*n,s);
}

void PETSC_STDCALL petscviewersocketputreal(PetscViewer *viewer,PetscInt *m,PetscInt *n,PetscReal *s,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerSocketPutReal(v,*m,*n,s);
}

void PETSC_STDCALL petscviewersocketputint(PetscViewer *viewer,PetscInt *m,PetscInt *s,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerSocketPutInt(v,*m,s);
}
#endif

void PETSC_STDCALL petscviewersetfilename_(PetscViewer *viewer,CHAR name PETSC_MIXED_LEN(len),
                                      PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char   *c1;
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  FIXCHAR(name,len,c1);
  *ierr = PetscViewerSetFilename(v,c1);
  FREECHAR(name,c1);
}

void PETSC_STDCALL  petscviewersetfiletype_(PetscViewer *viewer,PetscViewerFileType *type,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer,v);
  *ierr = PetscViewerSetFileType(v,*type);
}

#if defined(PETSC_HAVE_SOCKET)
void PETSC_STDCALL petscviewersocketopen_(MPI_Comm *comm,CHAR name PETSC_MIXED_LEN(len),int *port,PetscViewer *lab,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char   *c1;
  FIXCHAR(name,len,c1);
  *ierr = PetscViewerSocketOpen((MPI_Comm)PetscToPointerComm(*comm),c1,*port,lab);
  FREECHAR(name,c1);
}
#endif

void PETSC_STDCALL petscviewerbinaryopen_(MPI_Comm *comm,CHAR name PETSC_MIXED_LEN(len),PetscViewerFileType *type,
                           PetscViewer *binv,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char   *c1;
  FIXCHAR(name,len,c1);
  *ierr = PetscViewerBinaryOpen((MPI_Comm)PetscToPointerComm(*comm),c1,*type,binv);
  FREECHAR(name,c1);
}

#if defined(PETSC_HAVE_MATLAB)
void PETSC_STDCALL petscviewermatlabopen_(MPI_Comm *comm,CHAR name PETSC_MIXED_LEN(len),PetscViewerFileType *type,
                           PetscViewer *binv,PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char   *c1;
  FIXCHAR(name,len,c1);
  *ierr = PetscViewerMatlabOpen((MPI_Comm)PetscToPointerComm(*comm),c1,*type,binv);
  FREECHAR(name,c1);
}
#endif

void PETSC_STDCALL petscviewerasciiopen_(MPI_Comm *comm,CHAR name PETSC_MIXED_LEN(len),PetscViewer *lab,
                                    PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char   *c1;
  FIXCHAR(name,len,c1);
  *ierr = PetscViewerASCIIOpen((MPI_Comm)PetscToPointerComm(*comm),c1,lab);
  FREECHAR(name,c1);
}

void PETSC_STDCALL petscviewersetformat_(PetscViewer *vin,PetscViewerFormat *format,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = PetscViewerSetFormat(v,*format);
}

void PETSC_STDCALL petscviewerpushformat_(PetscViewer *vin,PetscViewerFormat *format,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = PetscViewerPushFormat(v,*format);
}

void PETSC_STDCALL petscviewerpopformat_(PetscViewer *vin,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = PetscViewerPopFormat(v);
}

void PETSC_STDCALL petscviewerdestroy_(PetscViewer *v,PetscErrorCode *ierr)
{
  *ierr = PetscViewerDestroy(*v);
}

void PETSC_STDCALL petscviewerstringopen_(MPI_Comm *comm,CHAR name PETSC_MIXED_LEN(len1),PetscInt *len,PetscViewer *str,
                                     PetscErrorCode *ierr PETSC_END_LEN(len1))
{
#if defined(PETSC_USES_CPTOFCD)
  *ierr = PetscViewerStringOpen((MPI_Comm)PetscToPointerComm(*comm),_fcdtocp(name),*len,str);
#else
  *ierr = PetscViewerStringOpen((MPI_Comm)PetscToPointerComm(*comm),name,*len,str);
#endif
}
  
void PETSC_STDCALL petscviewerdrawopen_(MPI_Comm *comm,CHAR display PETSC_MIXED_LEN(len1),
                   CHAR title PETSC_MIXED_LEN(len2),int *x,int*y,int*w,int*h,PetscViewer *v,
                   PetscErrorCode *ierr PETSC_END_LEN(len1) PETSC_END_LEN(len2))
{
  char   *c1,*c2;

  FIXCHAR(display,len1,c1);
  FIXCHAR(title,len2,c2);
  *ierr = PetscViewerDrawOpen((MPI_Comm)PetscToPointerComm(*comm),c1,c2,*x,*y,*w,*h,v);
  FREECHAR(display,c1);
  FREECHAR(title,c2);
}

EXTERN_C_END


