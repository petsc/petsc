#define PETSC_SKIP_COMPLEX
#include <petscsys.h>

#if defined(PETSC_HAVE_CXXABI_H)
#include <cxxabi.h>
#endif

PetscErrorCode PetscDemangleSymbol(const char mangledName[], char **name)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
#if defined(PETSC_HAVE_CXXABI_H)
  char *newname;
  int   status;

  newname = __cxxabiv1::__cxa_demangle(mangledName, NULL, NULL, &status);
  if (status) {
    PetscAssertFalse(status == -1,PETSC_COMM_SELF, PETSC_ERR_MEM, "Failed to allocate memory for symbol %s", mangledName);
    else if (status == -2) {
      /* Mangled name is not a valid name under the C++ ABI mangling rules */
      ierr = PetscStrallocpy(mangledName, name);CHKERRQ(ierr);
      PetscFunctionReturn(0);
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Demangling failed for symbol %s", mangledName);
  }
  ierr = PetscStrallocpy(newname, name);CHKERRQ(ierr);
  free(newname);
#else
  ierr = PetscStrallocpy(mangledName, name);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
