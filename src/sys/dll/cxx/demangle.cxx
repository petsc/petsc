#if !defined(PETSC_SKIP_COMPLEX)
  #define PETSC_SKIP_COMPLEX
#endif
#include <petscsys.h>
#include <petsc/private/petscimpl.h>

#if defined(PETSC_HAVE_CXXABI_H)
  #include <cxxabi.h>
#endif

PetscErrorCode PetscDemangleSymbol(const char mangledName[], char **name)
{
  PetscFunctionBegin;
  if (mangledName) PetscAssertPointer(mangledName, 1);
  PetscAssertPointer(name, 2);

  *name = nullptr;
  if (!mangledName) PetscFunctionReturn(PETSC_SUCCESS);
#if defined(PETSC_HAVE_CXXABI_H)
  char *newname;
  int   status;

  newname = __cxxabiv1::__cxa_demangle(mangledName, nullptr, nullptr, &status);
  if (status) {
    PetscCheck(status != -1, PETSC_COMM_SELF, PETSC_ERR_MEM, "Failed to allocate memory for symbol %s", mangledName);
    PetscCheck(status == -2, PETSC_COMM_SELF, PETSC_ERR_LIB, "Demangling failed for symbol %s", mangledName);
    /* Mangled name is not a valid name under the C++ ABI mangling rules */
    PetscCall(PetscStrallocpy(mangledName, name));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscStrallocpy(newname, name));
  free(newname);
#else
  PetscCall(PetscStrallocpy(mangledName, name));
#endif
  PetscFunctionReturn(PETSC_SUCCESS);
}
