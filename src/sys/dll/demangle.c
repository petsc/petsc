#define PETSC_DESIRE_FEATURE_TEST_MACROS /* for and RTLD_* */
#include <petsc/private/petscimpl.h>

#if defined(PETSC_HAVE_DLFCN_H)
  #include <dlfcn.h>
#endif

#if defined(__cplusplus) && defined(PETSC_HAVE_CXXABI_H)
  #include <cxxabi.h>
#endif

PetscErrorCode PetscDemangleSymbol(const char mangledName[], char **name)
{
  char *(*cxa_demangle)(const char *, char *, size_t *, int *) = PETSC_NULLPTR;
  char *newname;
  int   status;

  PetscFunctionBegin;
  if (mangledName) PetscAssertPointer(mangledName, 1);
  PetscAssertPointer(name, 2);

  *name = PETSC_NULLPTR;
  if (!mangledName) PetscFunctionReturn(PETSC_SUCCESS);

#if defined(__cplusplus) && defined(PETSC_HAVE_CXXABI_H)
  cxa_demangle = __cxxabiv1::__cxa_demangle;
#endif

#if defined(PETSC_HAVE_DLFCN_H) && defined(PETSC_HAVE_DLOPEN)
  if (!cxa_demangle) {
    void *symbol = PETSC_NULLPTR;
  #if defined(PETSC_HAVE_RTLD_DEFAULT)
    symbol = dlsym(RTLD_DEFAULT, "__cxa_demangle");
  #endif
    if (!symbol) {
      int   mode   = 0;
      void *handle = PETSC_NULLPTR;
  #if defined(PETSC_HAVE_RTLD_LAZY)
      mode |= RTLD_LAZY;
  #endif
  #if defined(PETSC_HAVE_RTLD_LOCAL)
      mode |= RTLD_LOCAL;
  #endif
  #if defined(PETSC_HAVE_RTLD_NOLOAD)
      mode |= RTLD_NOLOAD;
  #endif
  #ifdef __APPLE__
      if (!handle) handle = dlopen("libc++.1.dylib", mode);
  #else
      if (!handle) handle = dlopen("libstdc++.so.6", mode);
  #endif
      if (handle) {
        symbol = dlsym(handle, "__cxa_demangle");
        dlclose(handle);
      }
    }
    *(void **)(&cxa_demangle) = symbol;
  }
#endif

  if (!cxa_demangle) {
    PetscCall(PetscStrallocpy(mangledName, name));
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  newname = cxa_demangle(mangledName, PETSC_NULLPTR, PETSC_NULLPTR, &status);
  if (status) {
    PetscCheck(status != -1, PETSC_COMM_SELF, PETSC_ERR_MEM, "Failed to allocate memory for symbol %s", mangledName);
    PetscCheck(status == -2, PETSC_COMM_SELF, PETSC_ERR_LIB, "Demangling failed for symbol %s", mangledName);
    /* Mangled name is not a valid name under the C++ ABI mangling rules */
    PetscCall(PetscStrallocpy(mangledName, name));
    PetscFunctionReturn(PETSC_SUCCESS);
  }
  PetscCall(PetscStrallocpy(newname, name));
  free(newname);
  PetscFunctionReturn(PETSC_SUCCESS);
}
