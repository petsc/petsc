#include <petsc-private/fortranimpl.h>


#ifdef PETSC_HAVE_FORTRAN_CAPS
#define petscinfo_ PETSCINFO
#elif !defined(PETSC_HAVE_FORTRAN_UNDERSCORE) && !defined(FORTRANDOUBLEUNDERSCORE)
#define petscinfo_ petscinfo
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

void PETSC_STDCALL petscinfo_(CHAR text PETSC_MIXED_LEN(len1),PetscErrorCode *ierr PETSC_END_LEN(len1))
{
  char *c1,*tmp;

  FIXCHAR(text,len1,c1);
  *ierr = PetscFixSlashN(c1,&tmp);if (*ierr) return;
  *ierr = PetscInfo(PETSC_NULL,tmp);if (*ierr) return;
  *ierr = PetscFree(tmp);if (*ierr) return;
  FREECHAR(text,c1);
}

EXTERN_C_END
