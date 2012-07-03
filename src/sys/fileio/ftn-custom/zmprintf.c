#include <petsc-private/fortranimpl.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscfprintf_              PETSCFPRINTF
#define petscprintf_               PETSCPRINTF
#define petscsynchronizedfprintf_  PETSCSYNCHRONIZEDFPRINTF
#define petscsynchronizedprintf_   PETSCSYNCHRONIZEDPRINTF
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscfprintf_                 petscfprintf
#define petscprintf_                  petscprintf
#define petscsynchronizedfprintf_     petscsynchronizedfprintf
#define petscsynchronizedprintf_      petscsynchronizedprintf
#endif

EXTERN_C_BEGIN

#undef __FUNCT__  
#define __FUNCT__ "PetscFixSlashN"
static PetscErrorCode PetscFixSlashN(const char *in, char **out)
{
  PetscErrorCode ierr;
  PetscInt       i;
  size_t         len;

  PetscFunctionBegin;
  ierr = PetscStrallocpy(in,out);CHKERRQ(ierr);
  ierr = PetscStrlen(*out,&len);CHKERRQ(ierr);
  for (i=0; i<(int)len-1; i++) {
    if ((*out)[i] == '\\' && (*out)[i+1] == 'n') {(*out)[i] = ' '; (*out)[i+1] = '\n';}
  }
  PetscFunctionReturn(0);
}

void PETSC_STDCALL petscfprintf_(MPI_Comm *comm,FILE **file,CHAR fname PETSC_MIXED_LEN(len1),PetscErrorCode *ierr PETSC_END_LEN(len1))
{
  char     *c1,*tmp;

  FIXCHAR(fname,len1,c1);
  *ierr = PetscFixSlashN(c1,&tmp);if (*ierr) return;
  *ierr = PetscFPrintf(MPI_Comm_f2c(*(MPI_Fint *)&*comm),*file,tmp);if (*ierr) return;
  *ierr = PetscFree(tmp);if (*ierr) return;
  FREECHAR(fname,c1);
}

void PETSC_STDCALL petscprintf_(MPI_Comm *comm,CHAR fname PETSC_MIXED_LEN(len1),PetscErrorCode *ierr PETSC_END_LEN(len1))
{
  char *c1,*tmp;

  FIXCHAR(fname,len1,c1);
  *ierr = PetscFixSlashN(c1,&tmp);if (*ierr) return;
  *ierr = PetscPrintf(MPI_Comm_f2c(*(MPI_Fint *)&*comm),tmp);if (*ierr) return;
  *ierr = PetscFree(tmp);if (*ierr) return;
  FREECHAR(fname,c1);
}

void PETSC_STDCALL petscsynchronizedfprintf_(MPI_Comm *comm,FILE **file,CHAR fname PETSC_MIXED_LEN(len1),PetscErrorCode *ierr PETSC_END_LEN(len1))
{
  char *c1,*tmp;

  FIXCHAR(fname,len1,c1);
  *ierr = PetscFixSlashN(c1,&tmp);if (*ierr) return;
  *ierr = PetscSynchronizedFPrintf(MPI_Comm_f2c(*(MPI_Fint *)&*comm),*file,tmp);if (*ierr) return;
  *ierr = PetscFree(tmp);if (*ierr) return;
  FREECHAR(fname,c1);
}

void PETSC_STDCALL petscsynchronizedprintf_(MPI_Comm *comm,CHAR fname PETSC_MIXED_LEN(len1),PetscErrorCode *ierr PETSC_END_LEN(len1))
{
  char *c1,*tmp;

  FIXCHAR(fname,len1,c1);
  *ierr = PetscFixSlashN(c1,&tmp);if (*ierr) return;
  *ierr = PetscSynchronizedPrintf(MPI_Comm_f2c(*(MPI_Fint *)&*comm),tmp);if (*ierr) return;
  *ierr = PetscFree(tmp);if (*ierr) return;
  FREECHAR(fname,c1);
}

EXTERN_C_END
