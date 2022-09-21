#define PETSC_DESIRE_FEATURE_TEST_MACROS /* for getdomainname() */
/*
      Code for manipulating files.
*/
#include <petscsys.h>
#if defined(PETSC_HAVE_SYS_UTSNAME_H)
  #include <sys/utsname.h>
#endif
#if defined(PETSC_HAVE_WINDOWS_H)
  #include <windows.h>
#endif
#if defined(PETSC_HAVE_SYS_SYSTEMINFO_H)
  #include <sys/systeminfo.h>
#endif
#if defined(PETSC_HAVE_UNISTD_H)
  #include <unistd.h>
#endif
#if defined(PETSC_HAVE_NETDB_H)
  #include <netdb.h>
#endif

/*@C
    PetscGetHostName - Returns the name of the host. This attempts to
    return the entire Internet name. It may not return the same name
    as `MPI_Get_processor_name()`.

    Not Collective

    Input Parameter:
.   nlen - length of name

    Output Parameter:
.   name - contains host name.  Must be long enough to hold the name
           This is the fully qualified name, including the domain.

    Level: developer

   Fortran Note:
   In Fortran this routine has the format

$       character*(64) name
$       call PetscGetHostName(name,ierr)

.seealso: `PetscGetUserName()`, `PetscGetArchType()`
@*/
PetscErrorCode PetscGetHostName(char name[], size_t nlen)
{
  char *domain;
#if defined(PETSC_HAVE_UNAME) && !defined(PETSC_HAVE_GETCOMPUTERNAME)
  struct utsname utname;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_GETCOMPUTERNAME)
  {
    size_t nnlen = nlen;
    GetComputerName((LPTSTR)name, (LPDWORD)(&nnlen));
  }
#elif defined(PETSC_HAVE_UNAME)
  uname(&utname);
  PetscCall(PetscStrncpy(name, utname.nodename, nlen));
#elif defined(PETSC_HAVE_GETHOSTNAME)
  gethostname(name, nlen);
#elif defined(PETSC_HAVE_SYSINFO_3ARG)
  sysinfo(SI_HOSTNAME, name, nlen);
#endif
  /* if there was not enough room then system call will not null terminate name */
  name[nlen - 1] = 0;

  /* See if this name includes the domain */
  PetscCall(PetscStrchr(name, '.', &domain));
  if (!domain) {
    size_t l, ll;
    PetscCall(PetscStrlen(name, &l));
    if (l == nlen - 1) PetscFunctionReturn(0);
    name[l++] = '.';
    name[l]   = 0;
#if defined(PETSC_HAVE_SYSINFO_3ARG)
    sysinfo(SI_SRPC_DOMAIN, name + l, nlen - l);
#elif defined(PETSC_HAVE_GETDOMAINNAME)
    PetscCheck(!getdomainname(name + l, nlen - l), PETSC_COMM_SELF, PETSC_ERR_SYS, "getdomainname()");
#endif
    /* check if domain name is not a dnsdomainname and nuke it */
    PetscCall(PetscStrlen(name, &ll));
    if (ll > 4) {
      const char *suffixes[] = {".edu", ".com", ".net", ".org", ".mil", NULL};
      PetscInt    index;
      PetscCall(PetscStrendswithwhich(name, suffixes, &index));
      if (!suffixes[index]) {
        PetscCall(PetscInfo(NULL, "Rejecting domainname, likely is NIS %s\n", name));
        name[l - 1] = 0;
      }
    }
  }
  PetscFunctionReturn(0);
}
