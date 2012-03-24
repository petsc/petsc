#include <petsc-private/fortranimpl.h>
#include <petscsys.h>

#if defined(PETSC_HAVE_FORTRAN_CAPS)
#define petscmallocdump_           PETSCMALLOCDUMP
#define petscmallocdumplog_        PETSCMALLOCDUMPLOG
#define petscmallocvalidate_       PETSCMALLOCVALIDATE
#define petscmemoryshowusage_      PETSCMEMORYSHOWUSAGE
#define petscmemorysetgetmaximumusage_ PETSCMEMORYSETGETMAXIMUMUSAGE
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE)
#define petscmallocdump_           petscmallocdump
#define petscmallocdumplog_        petscmallocdumplog
#define petscmallocvalidate_       petscmallocvalidate
#define petscmemoryshowusage_      petscmemoryshowusage
#define petscmemorysetgetmaximumusage_ petscmemorysetgetmaximumusage
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

void PETSC_STDCALL  petscmallocdump_(PetscErrorCode *ierr)
{
  *ierr = PetscMallocDump(stdout);
}
void PETSC_STDCALL petscmallocdumplog_(PetscErrorCode *ierr)
{
  *ierr = PetscMallocDumpLog(stdout);
}

void PETSC_STDCALL petscmallocvalidate_(PetscErrorCode *ierr)
{
  *ierr = PetscMallocValidate(0,"Unknown Fortran",0,0);
}

void PETSC_STDCALL petscmemorysetgetmaximumusage_(PetscErrorCode *ierr)
{
  *ierr = PetscMemorySetGetMaximumUsage();
}

void PETSC_STDCALL petscmemoryshowusage_(PetscViewer *vin, CHAR message PETSC_MIXED_LEN(len), PetscErrorCode *ierr PETSC_END_LEN(len))
{
  PetscViewer v;
  char *msg, *tmp;

  FIXCHAR(message,len,msg);
  *ierr = PetscFixSlashN(msg,&tmp);if (*ierr) return;
  PetscPatchDefaultViewers_Fortran(vin,v);
  *ierr = PetscMemoryShowUsage(v,tmp);
  FREECHAR(message,msg);
}

EXTERN_C_END
