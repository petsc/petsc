#include <petsc-private/fortranimpl.h>
#include <petscmatlab.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscmatlabenginecreate_      PETSCMATLABENGINECREATE
#define petscmatlabengineevaluate_    PETSCMATLABENGINEEVALUATE
#define petscmatlabenginegetoutput_   PETSCMATLABENGINEGETOUTPUT
#define petscmatlabengineprintoutput_ PETSCMATLABENGINEPRINTOUTPUT
#define petscmatlabengineputarray_    PETSCMATLABENGINEPUTARRAY
#define petscmatlabenginegetarray_    PETSCMATLABENGINEGETARRAY
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscmatlabenginecreate_      petscmatlabenginecreate
#define petscmatlabengineevaluate_    petscmatlabengineevaluate
#define petscmatlabenginegetoutput_   petscmatlabenginegetoutput
#define petscmatlabengineprintoutput_ petscmatlabengineprintoutput
#define petscmatlabengineputarray_    petscmatlabengineputarray
#define petscmatlabenginegetarray_    petscmatlabenginegetarray
#endif

EXTERN_C_BEGIN

void PETSC_STDCALL petscmatlabenginecreate_(MPI_Comm *comm,CHAR m PETSC_MIXED_LEN(len),PetscMatlabEngine *e,
                                            PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *ms;

  FIXCHAR(m,len,ms);
  *ierr = PetscMatlabEngineCreate(MPI_Comm_f2c(*(MPI_Fint *)&*comm),ms,e);
  FREECHAR(m,ms);
}

void PETSC_STDCALL petscmatlabengineevaluate_(PetscMatlabEngine *e,CHAR m PETSC_MIXED_LEN(len),
                                              PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *ms;
  FIXCHAR(m,len,ms);
  *ierr = PetscMatlabEngineEvaluate(*e,ms);
  FREECHAR(m,ms);
}

void PETSC_STDCALL petscmatlabengineputarray_(PetscMatlabEngine *e,PetscInt *m,PetscInt *n,PetscScalar *a,
                                              CHAR s PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *ms;
  FIXCHAR(s,len,ms);
  *ierr = PetscMatlabEnginePutArray(*e,*m,*n,a,ms);
  FREECHAR(s,ms);
}

void PETSC_STDCALL petscmatlabenginegetarray_(PetscMatlabEngine *e,PetscInt *m,PetscInt *n,PetscScalar *a,
                                              CHAR s PETSC_MIXED_LEN(len),PetscErrorCode *ierr PETSC_END_LEN(len))
{
  char *ms;
  FIXCHAR(s,len,ms);
  *ierr = PetscMatlabEngineGetArray(*e,*m,*n,a,ms);
  FREECHAR(s,ms);
}

/*
extern int PetscMatlabEngineGetOutput(PetscMatlabEngine,char **);
extern int PetscMatlabEnginePrintOutput(PetscMatlabEngine,FILE*);
*/

EXTERN_C_END
