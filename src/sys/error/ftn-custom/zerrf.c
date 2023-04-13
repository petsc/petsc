#include <petsc/private/fortranimpl.h>
#include <petscsys.h>
#include <petscviewer.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscpusherrorhandler_           PETSCPUSHERRORHANDLER
  #define petsctracebackerrorhandler_      PETSCTRACEBACKERRORHANDLER
  #define petscaborterrorhandler_          PETSCABORTERRORHANDLER
  #define petscignoreerrorhandler_         PETSCIGNOREERRORHANDLER
  #define petscemacsclienterrorhandler_    PETSCEMACSCLIENTERRORHANDLER
  #define petscattachdebuggererrorhandler_ PETSCATTACHDEBUGGERERRORHANDLER
  #define petscerror_                      PETSCERROR
  #define petscerrorf_                     PETSCERRORF
  #define petscerrormpi_                   PETSCERRORMPI
  #define petscrealview_                   PETSCREALVIEW
  #define petscintview_                    PETSCINTVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscpusherrorhandler_           petscpusherrorhandler
  #define petsctracebackerrorhandler_      petsctracebackerrorhandler
  #define petscaborterrorhandler_          petscaborterrorhandler
  #define petscignoreerrorhandler_         petscignoreerrorhandler
  #define petscemacsclienterrorhandler_    petscemacsclienterrorhandler
  #define petscattachdebuggererrorhandler_ petscattachdebuggererrorhandler
  #define petscerror_                      petscerror
  #define petscerrorf_                     petscerrorf
  #define petscerrormpi_                   petscerrormpi
  #define petscrealview_                   petscrealview
  #define petscintview_                    petscintview
#endif

static void (*f2)(MPI_Comm *comm, int *, const char *, const char *, PetscErrorCode *, PetscErrorType *, const char *, void *, PetscErrorCode *, PETSC_FORTRAN_CHARLEN_T len1, PETSC_FORTRAN_CHARLEN_T len2, PETSC_FORTRAN_CHARLEN_T len3);

/* These are not extern C because they are passed into non-extern C user level functions */
static PetscErrorCode ourerrorhandler(MPI_Comm comm, int line, const char *fun, const char *file, PetscErrorCode n, PetscErrorType p, const char *mess, void *ctx)
{
  PetscErrorCode ierr = PETSC_SUCCESS;
  size_t         len1, len2, len3;

  ierr = PetscStrlen(fun, &len1);
  ierr = PetscStrlen(file, &len2);
  ierr = PetscStrlen(mess, &len3);

  ierr = PETSC_SUCCESS;
  (*f2)(&comm, &line, fun, file, &n, &p, mess, ctx, &ierr, ((PETSC_FORTRAN_CHARLEN_T)(len1)), ((PETSC_FORTRAN_CHARLEN_T)(len2)), ((PETSC_FORTRAN_CHARLEN_T)(len3)));
  return ierr;
}

/*
        These are not usually called from Fortran but allow Fortran users
   to transparently set these monitors from .F code
*/
PETSC_EXTERN void petsctracebackerrorhandler_(MPI_Comm *comm, int *line, const char *fun, const char *file, PetscErrorCode *n, PetscErrorType *p, const char *mess, void *ctx, PetscErrorCode *ierr)
{
  *ierr = PetscTraceBackErrorHandler(*comm, *line, fun, file, *n, *p, mess, ctx);
}

PETSC_EXTERN void petscaborterrorhandler_(MPI_Comm *comm, int *line, const char *fun, const char *file, PetscErrorCode *n, PetscErrorType *p, const char *mess, void *ctx, PetscErrorCode *ierr)
{
  *ierr = PetscAbortErrorHandler(*comm, *line, fun, file, *n, *p, mess, ctx);
}

PETSC_EXTERN void petscattachdebuggererrorhandler_(MPI_Comm *comm, int *line, const char *fun, const char *file, PetscErrorCode *n, PetscErrorType *p, const char *mess, void *ctx, PetscErrorCode *ierr)
{
  *ierr = PetscAttachDebuggerErrorHandler(*comm, *line, fun, file, *n, *p, mess, ctx);
}

PETSC_EXTERN void petscemacsclienterrorhandler_(MPI_Comm *comm, int *line, const char *fun, const char *file, PetscErrorCode *n, PetscErrorType *p, const char *mess, void *ctx, PetscErrorCode *ierr)
{
  *ierr = PetscEmacsClientErrorHandler(*comm, *line, fun, file, *n, *p, mess, ctx);
}

PETSC_EXTERN void petscignoreerrorhandler_(MPI_Comm *comm, int *line, const char *fun, const char *file, PetscErrorCode *n, PetscErrorType *p, const char *mess, void *ctx, PetscErrorCode *ierr)
{
  *ierr = PetscIgnoreErrorHandler(*comm, *line, fun, file, *n, *p, mess, ctx);
}

PETSC_EXTERN void petscpusherrorhandler_(void (*handler)(MPI_Comm *comm, int *, const char *, const char *, PetscErrorCode *, PetscErrorType *, const char *, void *, PetscErrorCode *, PETSC_FORTRAN_CHARLEN_T len1, PETSC_FORTRAN_CHARLEN_T len2, PETSC_FORTRAN_CHARLEN_T len3), void *ctx, PetscErrorCode *ierr)
{
  if ((void (*)(void))handler == (void (*)(void))petsctracebackerrorhandler_) *ierr = PetscPushErrorHandler(PetscTraceBackErrorHandler, 0);
  else {
    f2    = handler;
    *ierr = PetscPushErrorHandler(ourerrorhandler, ctx);
  }
}

PETSC_EXTERN void petscerror_(MPI_Fint *comm, PetscErrorCode *number, PetscErrorType *p, char *message, PETSC_FORTRAN_CHARLEN_T len)
{
  PetscErrorCode nierr, *ierr = &nierr;
  char          *t1;
  FIXCHAR(message, len, t1);
  nierr = PetscError(MPI_Comm_f2c(*(comm)), 0, NULL, NULL, *number, *p, "%s", t1);
  FREECHAR(message, t1);
}

#if defined(PETSC_HAVE_FORTRAN_FREE_LINE_LENGTH_NONE)
PETSC_EXTERN void petscerrorf_(PetscErrorCode *err, int *line, char *file, PETSC_FORTRAN_CHARLEN_T len)
{
  char          *tfile;
  PetscErrorCode ierr[] = {PETSC_SUCCESS}; /* needed by FIXCHAR */

  FIXCHAR(file, len, tfile);
  *err = PetscError(PETSC_COMM_SELF, *line, NULL, tfile, *err, PETSC_ERROR_REPEAT, NULL);
  FREECHAR(file, tfile);
}

PETSC_EXTERN void petscerrormpi_(PetscErrorCode *err, int *line, char *file, PETSC_FORTRAN_CHARLEN_T len)
{
  char           errorstring[2 * MPI_MAX_ERROR_STRING];
  char          *tfile;
  PetscErrorCode ierr[] = {PETSC_SUCCESS}; /* needed by FIXCHAR */

  FIXCHAR(file, len, tfile);
  PetscMPIErrorString(*err, errorstring);
  *err = PetscError(PETSC_COMM_SELF, *line, NULL, file, PETSC_ERR_MPI, PETSC_ERROR_INITIAL, "MPI error %d %s", *err, errorstring);
  FREECHAR(file, tfile);
  *err = PETSC_ERR_MPI;
}
#else
PETSC_EXTERN void petscerrorf_(PetscErrorCode *err)
{
  *err = PetscError(PETSC_COMM_SELF, 0, NULL, NULL, *err, PETSC_ERROR_REPEAT, NULL);
}

PETSC_EXTERN void petscerrormpi_(PetscErrorCode *err)
{
  char errorstring[2 * MPI_MAX_ERROR_STRING];

  PetscMPIErrorString(*err, errorstring);
  *err = PetscError(PETSC_COMM_SELF, 0, NULL, NULL, PETSC_ERR_MPI, PETSC_ERROR_INITIAL, "MPI error %d %s", *err, errorstring);
  *err = PETSC_ERR_MPI;
}
#endif

PETSC_EXTERN void petscrealview_(PetscInt *n, PetscReal *d, PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer, v);
  *ierr = PetscRealView(*n, d, v);
}

PETSC_EXTERN void petscintview_(PetscInt *n, PetscInt *d, PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer, v);
  *ierr = PetscIntView(*n, d, v);
}

#if defined(PETSC_HAVE_FORTRAN_CAPS)
  #define petscscalarview_ PETSCSCALARVIEW
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
  #define petscscalarview_ petscscalarview
#endif

PETSC_EXTERN void petscscalarview_(PetscInt *n, PetscScalar *d, PetscViewer *viewer, PetscErrorCode *ierr)
{
  PetscViewer v;
  PetscPatchDefaultViewers_Fortran(viewer, v);
  *ierr = PetscScalarView(*n, d, v);
}
