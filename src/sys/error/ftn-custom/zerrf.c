#include <petsc-private/fortranimpl.h>
#include <petscsys.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscpusherrorhandler_        PETSCPUSHERRORHANDLER
#define petsctracebackerrorhandler_   PETSCTRACEBACKERRORHANDLER
#define petscaborterrorhandler_       PETSCABORTERRORHANDLER
#define petscignoreerrorhandler_      PETSCIGNOREERRORHANDLER
#define petscemacsclienterrorhandler_ PETSCEMACSCLIENTERRORHANDLER
#define petscattachdebuggererrorhandler_   PETSCATTACHDEBUGGERERRORHANDLER
#define petscerror_                PETSCERROR
#define petscrealview_             PETSCREALVIEW
#define petscintview_              PETSCINTVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscpusherrorhandler_   petscpusherrorhandler
#define petsctracebackerrorhandler_   petsctracebackerrorhandler
#define petscaborterrorhandler_       petscaborterrorhandler
#define petscignoreerrorhandler_      petscignoreerrorhandler
#define petscemacsclienterrorhandler_ petscemacsclienterrorhandler
#define petscattachdebuggererrorhandler_   petscattachdebuggererrorhandler
#define petscerror_                petscerror
#define petscrealview_             petscrealview
#define petscintview_              petscintview
#endif

EXTERN_C_BEGIN
static void (PETSC_STDCALL *f2)(MPI_Comm *comm,int*,const CHAR PETSC_MIXED_LEN(len1),const CHAR PETSC_MIXED_LEN(len2),const CHAR PETSC_MIXED_LEN(len3),PetscErrorCode*,PetscErrorType*,const CHAR PETSC_MIXED_LEN(len4),void*,PetscErrorCode* PETSC_END_LEN(len1) PETSC_END_LEN(len2) PETSC_END_LEN(len3) PETSC_END_LEN(len4));
EXTERN_C_END

/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode ourerrorhandler(MPI_Comm comm,int line,const char *fun,const char *file,const char *dir,PetscErrorCode n,PetscErrorType p,const char *mess,void *ctx)
{
  PetscErrorCode ierr = 0;
  size_t         len1,len2,len3,len4;
  int            l1,l2,l3,l4;

  PetscStrlen(fun,&len1); l1 = (int)len1;
  PetscStrlen(file,&len2);l2 = (int)len2;
  PetscStrlen(dir,&len3);l3 = (int)len3;
  PetscStrlen(mess,&len4);l4 = (int)len4;

#if defined(PETSC_HAVE_FORTRAN_MIXED_STR_ARG)
  (*f2)(&comm,&line,fun,l1,file,l2,dir,l3,&n,&p,mess,l4,ctx,&ierr);
#else
  (*f2)(&comm,&line,fun,file,dir,&n,&p,mess,ctx,&ierr,l1,l2,l3,l4);
#endif
  return ierr;
}

EXTERN_C_BEGIN

/*
        These are not usually called from Fortran but allow Fortran users
   to transparently set these monitors from .F code

   functions, hence no STDCALL
*/
void petsctracebackerrorhandler_(MPI_Comm *comm,int *line,const char *fun,const char *file,const char *dir,PetscErrorCode *n,PetscErrorType *p,const char *mess,void *ctx,PetscErrorCode *ierr)
{
  *ierr = PetscTraceBackErrorHandler(*comm,*line,fun,file,dir,*n,*p,mess,ctx);
}

void petscaborterrorhandler_(MPI_Comm *comm,int *line,const char *fun,const char *file,const char *dir,PetscErrorCode *n,PetscErrorType *p,const char *mess,void *ctx,PetscErrorCode *ierr)
{
  *ierr = PetscAbortErrorHandler(*comm,*line,fun,file,dir,*n,*p,mess,ctx);
}

void petscattachdebuggererrorhandler_(MPI_Comm *comm,int *line,const char *fun,const char *file,const char *dir,PetscErrorCode *n,PetscErrorType *p,const char *mess,void *ctx,PetscErrorCode *ierr)
{
  *ierr = PetscAttachDebuggerErrorHandler(*comm,*line,fun,file,dir,*n,*p,mess,ctx);
}

void petscemacsclienterrorhandler_(MPI_Comm *comm,int *line,const char *fun,const char *file,const char *dir,PetscErrorCode *n,PetscErrorType *p,const char *mess,void *ctx,PetscErrorCode *ierr)
{
  *ierr = PetscEmacsClientErrorHandler(*comm,*line,fun,file,dir,*n,*p,mess,ctx);
}

void petscignoreerrorhandler_(MPI_Comm *comm,int *line,const char *fun,const char *file,const char *dir,PetscErrorCode *n,PetscErrorType *p,const char *mess,void *ctx,PetscErrorCode *ierr)
{
  *ierr = PetscIgnoreErrorHandler(*comm,*line,fun,file,dir,*n,*p,mess,ctx);
}

void PETSC_STDCALL petscpusherrorhandler_(void (PETSC_STDCALL *handler)(MPI_Comm *comm,int*,const CHAR PETSC_MIXED_LEN(len1),const CHAR PETSC_MIXED_LEN(len2),const CHAR PETSC_MIXED_LEN(len3),PetscErrorCode*,PetscErrorType*,const CHAR PETSC_MIXED_LEN(len4),void*,PetscErrorCode* PETSC_END_LEN(len1) PETSC_END_LEN(len2) PETSC_END_LEN(len3) PETSC_END_LEN(len4)),void *ctx,PetscErrorCode *ierr)
{
  if ((void(*)(void))handler == (void(*)(void))petsctracebackerrorhandler_) {
    *ierr = PetscPushErrorHandler(PetscTraceBackErrorHandler,0);
  } else {
    f2    = handler;
    *ierr = PetscPushErrorHandler(ourerrorhandler,ctx);
  }
}

void PETSC_STDCALL petscerror_(MPI_Comm *comm,PetscErrorCode *number,int *line,PetscErrorType *p,CHAR message PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *t1;
  FIXCHAR(message,len,t1);
  *ierr = PetscError(*comm,*line,0,0,0,*number,*p,t1);
  FREECHAR(message,t1);
}

void PETSC_STDCALL petscrealview_(PetscInt *n,PetscReal *d,PetscViewer *viwer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viwer,v);
  *ierr = PetscRealView(*n,d,v);
}

void PETSC_STDCALL petscintview_(PetscInt *n,PetscInt *d,PetscViewer *viwer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viwer,v);
  *ierr = PetscIntView(*n,d,v);
}

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscscalarview_             PETSCSCALARVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscscalarview_             petscscalarview
#endif

void PETSC_STDCALL petscscalarview_(PetscInt *n,PetscScalar *d,PetscViewer *viwer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viwer,v);
  *ierr = PetscScalarView(*n,d,v);
}

EXTERN_C_END
