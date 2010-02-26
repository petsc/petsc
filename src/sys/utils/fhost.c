#define PETSC_DLL
/*
      Code for manipulating files.
*/
#include "petscsys.h"
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif
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

#undef __FUNCT__  
#define __FUNCT__ "PetscGetHostName"
/*@C
    PetscGetHostName - Returns the name of the host. This attempts to
    return the entire Internet name. It may not return the same name
    as MPI_Get_processor_name().

    Not Collective

    Input Parameter:
.   nlen - length of name

    Output Parameter:
.   name - contains host name.  Must be long enough to hold the name
           This is the fully qualified name, including the domain.

    Level: developer

    Concepts: machine name
    Concepts: host name

.seealso: PetscGetUserName()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscGetHostName(char name[],size_t nlen)
{
  char           *domain;
  PetscErrorCode ierr;
  PetscTruth     flag;
#if defined(PETSC_HAVE_UNAME) && !defined(PETSC_HAVE_GETCOMPUTERNAME)
  struct utsname utname;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_GETCOMPUTERNAME)
 {
    size_t nnlen = nlen;
    GetComputerName((LPTSTR)name,(LPDWORD)(&nnlen));
 }
#elif defined(PETSC_HAVE_UNAME)
  uname(&utname); 
  ierr = PetscStrncpy(name,utname.nodename,nlen);CHKERRQ(ierr);
#elif defined(PETSC_HAVE_GETHOSTNAME)
  gethostname(name,nlen);
#elif defined(PETSC_HAVE_SYSINFO_3ARG)
  sysinfo(SI_HOSTNAME,name,nlen);
#endif
  /* if there was not enough room then system call will not null terminate name */
  name[nlen-1] = 0;

  /* See if this name includes the domain */
  ierr = PetscStrchr(name,'.',&domain);CHKERRQ(ierr);
  if (!domain) {
    size_t  l,ll;
    ierr = PetscStrlen(name,&l);CHKERRQ(ierr);
    if (l == nlen-1) PetscFunctionReturn(0);
    name[l++] = '.';
#if defined(PETSC_HAVE_SYSINFO_3ARG)
    sysinfo(SI_SRPC_DOMAIN,name+l,nlen-l);
#elif defined(PETSC_HAVE_GETDOMAINNAME)
    getdomainname(name+l,nlen - l);
#endif
    /* check if domain name is not a dnsdomainname and nuke it */
    ierr = PetscStrlen(name,&ll);CHKERRQ(ierr);
    if (ll > 4) {
      ierr = PetscStrcmp(name + ll - 4,".edu",&flag);CHKERRQ(ierr);
      if (!flag) {
        ierr = PetscStrcmp(name + ll - 4,".com",&flag);CHKERRQ(ierr);
        if (!flag) {
          ierr = PetscStrcmp(name + ll - 4,".net",&flag);CHKERRQ(ierr);
          if (!flag) {
            ierr = PetscStrcmp(name + ll - 4,".org",&flag);CHKERRQ(ierr);
            if (!flag) {
              ierr = PetscStrcmp(name + ll - 4,".mil",&flag);CHKERRQ(ierr);
              if (!flag) {
                ierr = PetscInfo1(0,"Rejecting domainname, likely is NIS %s\n",name);CHKERRQ(ierr);
                name[l-1] = 0;
              }
            }
          }
        }
      }
    }
  }
  PetscFunctionReturn(0);
}
