
#include <petsc/private/petscimpl.h> /*I    "petscsys.h"   I*/

/*@C
   PetscHasExternalPackage - Determine whether PETSc has been configured with the given package

   Not Collective

   Input Parameters:
.  pkg - external package name

   Output Parameters:
.  has - `PETSC_TRUE` if PETSc is configured with the given package, else `PETSC_FALSE`.

   Level: intermediate

   Notes:
   This is basically an alternative for PETSC_HAVE_XXX whenever a preprocessor macro is not available/desirable, e.g. in Python.

   The external package name pkg is e.g. "hdf5", "yaml", "parmetis".
   It should correspond to the name listed in  ./configure --help  or e.g. in PetscViewerType, MatPartitioningType, MatSolverType.

   The lookup is case insensitive, i.e. looking for "HDF5" or "hdf5" is the same.

.seealso: `PetscViewerType`, `MatPartitioningType`, `MatSolverType`
@*/
PetscErrorCode PetscHasExternalPackage(const char pkg[], PetscBool *has)
{
  char   pkgstr[128], *loc;
  size_t cnt;

  PetscFunctionBegin;
  PetscValidCharPointer(pkg, 1);
  PetscValidBoolPointer(has, 2);
  PetscCall(PetscSNPrintfCount(pkgstr, sizeof(pkgstr), ":%s:", &cnt, pkg));
  PetscCheck(cnt < sizeof(pkgstr), PETSC_COMM_SELF, PETSC_ERR_SUP, "Package name is too long: \"%s\"", pkg);
  PetscCall(PetscStrtolower(pkgstr));
#if defined(PETSC_HAVE_PACKAGES)
  PetscCall(PetscStrstr(PETSC_HAVE_PACKAGES, pkgstr, &loc));
#else
  #error "PETSC_HAVE_PACKAGES macro undefined. Please reconfigure"
#endif
  *has = loc ? PETSC_TRUE : PETSC_FALSE;
  PetscFunctionReturn(0);
}
