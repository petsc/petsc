#include <petsc/private/fortranimpl.h>
#include <petscsys.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscpusherrorhandler_        PETSCPUSHERRORHANDLER
#define petsctracebackerrorhandler_   PETSCTRACEBACKERRORHANDLER
#define petscaborterrorhandler_       PETSCABORTERRORHANDLER
#define petscignoreerrorhandler_      PETSCIGNOREERRORHANDLER
#define petscemacsclienterrorhandler_ PETSCEMACSCLIENTERRORHANDLER
#define petscattachdebuggererrorhandler_   PETSCATTACHDEBUGGERERRORHANDLER
#define petscerror_                PETSCERROR
#define petscerrorf_                PETSCERRORF
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
#define petscerrorf_                petscerrorf
#define petscrealview_             petscrealview
#define petscintview_              petscintview
#endif

static void (PETSC_STDCALL *f2)(MPI_Comm *comm,int*,const char* PETSC_MIXED_LEN(len1),const char* PETSC_MIXED_LEN(len2),PetscErrorCode*,PetscErrorType*,const char* PETSC_MIXED_LEN(len3),void*,PetscErrorCode* PETSC_END_LEN(len1) PETSC_END_LEN(len2) PETSC_END_LEN(len3));

/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode ourerrorhandler(MPI_Comm comm,int line,const char *fun,const char *file,PetscErrorCode n,PetscErrorType p,const char *mess,void *ctx)
{
  PetscErrorCode ierr = 0;
  size_t         len1,len2,len3;
  int            l1,l2,l3;

  PetscStrlen(fun,&len1); l1 = (int)len1;
  PetscStrlen(file,&len2);l2 = (int)len2;
  PetscStrlen(mess,&len3);l3 = (int)len3;

#if defined(PETSC_HAVE_FORTRAN_MIXED_STR_ARG)
  (*f2)(&comm,&line,fun,l1,file,l2,&n,&p,mess,l3,ctx,&ierr);
#else
  (*f2)(&comm,&line,fun,file,&n,&p,mess,ctx,&ierr,l1,l2,l3);
#endif
  return ierr;
}

/*
        These are not usually called from Fortran but allow Fortran users
   to transparently set these monitors from .F code

   functions, hence no STDCALL
*/
PETSC_EXTERN void petsctracebackerrorhandler_(MPI_Comm *comm,int *line,const char *fun,const char *file,PetscErrorCode *n,PetscErrorType *p,const char *mess,void *ctx,PetscErrorCode *ierr)
{
  *ierr = PetscTraceBackErrorHandler(*comm,*line,fun,file,*n,*p,mess,ctx);
}

PETSC_EXTERN void petscaborterrorhandler_(MPI_Comm *comm,int *line,const char *fun,const char *file,PetscErrorCode *n,PetscErrorType *p,const char *mess,void *ctx,PetscErrorCode *ierr)
{
  *ierr = PetscAbortErrorHandler(*comm,*line,fun,file,*n,*p,mess,ctx);
}

PETSC_EXTERN void petscattachdebuggererrorhandler_(MPI_Comm *comm,int *line,const char *fun,const char *file,PetscErrorCode *n,PetscErrorType *p,const char *mess,void *ctx,PetscErrorCode *ierr)
{
  *ierr = PetscAttachDebuggerErrorHandler(*comm,*line,fun,file,*n,*p,mess,ctx);
}

PETSC_EXTERN void petscemacsclienterrorhandler_(MPI_Comm *comm,int *line,const char *fun,const char *file,PetscErrorCode *n,PetscErrorType *p,const char *mess,void *ctx,PetscErrorCode *ierr)
{
  *ierr = PetscEmacsClientErrorHandler(*comm,*line,fun,file,*n,*p,mess,ctx);
}

PETSC_EXTERN void petscignoreerrorhandler_(MPI_Comm *comm,int *line,const char *fun,const char *file,PetscErrorCode *n,PetscErrorType *p,const char *mess,void *ctx,PetscErrorCode *ierr)
{
  *ierr = PetscIgnoreErrorHandler(*comm,*line,fun,file,*n,*p,mess,ctx);
}

PETSC_EXTERN void PETSC_STDCALL petscpusherrorhandler_(void (PETSC_STDCALL *handler)(MPI_Comm *comm,int*,const char* PETSC_MIXED_LEN(len1),const char* PETSC_MIXED_LEN(len2),PetscErrorCode*,PetscErrorType*,const char* PETSC_MIXED_LEN(len3),void*,PetscErrorCode* PETSC_END_LEN(len1) PETSC_END_LEN(len2) PETSC_END_LEN(len3)),void *ctx,PetscErrorCode *ierr)
{
  if ((void(*)(void))handler == (void(*)(void))petsctracebackerrorhandler_) *ierr = PetscPushErrorHandler(PetscTraceBackErrorHandler,0);
  else {
    f2    = handler;
    *ierr = PetscPushErrorHandler(ourerrorhandler,ctx);
  }
}

PETSC_EXTERN void PETSC_STDCALL petscerror_(MPI_Fint *comm,PetscErrorCode *number,PetscErrorType *p,char* message PETSC_MIXED_LEN(len) PETSC_END_LEN(len))
{
  PetscErrorCode nierr,*ierr = &nierr;
  char *t1;
  FIXCHAR(message,len,t1);
  nierr = PetscError(MPI_Comm_f2c(*(comm)),0,NULL,NULL,*number,*p,t1);
  FREECHAR(message,t1);
}

/* helper routine for CHKERRQ and CHKERRABORT macros on the fortran side */
PETSC_EXTERN void PETSC_STDCALL petscerrorf_(PetscErrorCode *number)
{
  PetscError(PETSC_COMM_SELF,0,NULL,NULL,*number,PETSC_ERROR_REPEAT,NULL);
}

PETSC_EXTERN void PETSC_STDCALL petscrealview_(PetscInt *n,PetscReal *d,PetscViewer *viwer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viwer,v);
  *ierr = PetscRealView(*n,d,v);
}

PETSC_EXTERN void PETSC_STDCALL petscintview_(PetscInt *n,PetscInt *d,PetscViewer *viwer,PetscErrorCode *ierr)
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

PETSC_EXTERN void PETSC_STDCALL petscscalarview_(PetscInt *n,PetscScalar *d,PetscViewer *viwer,PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viwer,v);
  *ierr = PetscScalarView(*n,d,v);
}
