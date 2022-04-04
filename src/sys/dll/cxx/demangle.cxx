#define PETSC_SKIP_COMPLEX
#include <petscsys.h>

#if defined(PETSC_HAVE_CXXABI_H)
#include <cxxabi.h>
#endif

PetscErrorCode PetscDemangleSymbol(const char mangledName[], char **name)
{
  PetscFunctionBegin;
#if defined(PETSC_HAVE_CXXABI_H)
  char *newname;
  int   status;

  newname = __cxxabiv1::__cxa_demangle(mangledName, NULL, NULL, &status);
  if (status) {
    PetscCheck(status != -1,PETSC_COMM_SELF, PETSC_ERR_MEM, "Failed to allocate memory for symbol %s", mangledName);
    else if (status == -2) {
      /* Mangled name is not a valid name under the C++ ABI mangling rules */
      PetscCall(PetscStrallocpy(mangledName, name));
      PetscFunctionReturn(0);
    } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_LIB, "Demangling failed for symbol %s", mangledName);
  }
  PetscCall(PetscStrallocpy(newname, name));
  free(newname);
#else
  PetscCall(PetscStrallocpy(mangledName, name));
#endif
  PetscFunctionReturn(0);
}
