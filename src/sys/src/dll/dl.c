


#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dl.c,v 1.42 1999/03/16 16:42:39 bsmith Exp balay $";
#endif
/*
      Routines for opening dynamic link libraries (DLLs), keeping a searchable
   path of DLLs, obtaining remote DLLs via a URL and opening them locally.
*/

#include "petsc.h"
#include "sys.h"
#include "pinclude/ptime.h"
#if defined(HAVE_PWD_H)
#include <pwd.h>
#endif
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#if defined(HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#if !defined(PARCH_win32)
#include <sys/utsname.h>
#endif
#if defined(PARCH_win32)
#include <windows.h>
#include <io.h>
#include <direct.h>
#endif
#if defined (PARCH_win32_gnu)
#include <windows.h>
#endif
#include <fcntl.h>
#include <time.h>  
#if defined(HAVE_SYS_SYSTEMINFO_H)
#include <sys/systeminfo.h>
#endif
#include "pinclude/petscfix.h"

#ifndef MAXPATHLEN
#define MAXPATHLEN 1024
#endif

/* ------------------------------------------------------------------------------*/
/*
      Code to maintain a list of opened dynamic libraries and load symbols
*/
#if defined(USE_DYNAMIC_LIBRARIES)
#include <dlfcn.h>

struct _DLLibraryList {
  DLLibraryList next;
  void          *handle;
  char          libname[1024];
};

extern int Petsc_DelTag(MPI_Comm,int,void*,void*);

int DLLibraryPrintPath(void)
{
  DLLibraryList libs;

  PetscFunctionBegin;
  PetscErrorPrintf("Unable to find function. Search path:\n");
  libs = DLLibrariesLoaded;
  while (libs) {
    PetscErrorPrintf("  %s\n",libs->libname);
    libs = libs->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DLLibraryGetInfo"
/*@C
   DLLibraryGetInfo - Gets the text information from a PETSc
       dynamic library

     Not Collective

   Input Parameters:
.   handle - library handle returned by DLLibraryOpen()

   Level: developer

@*/
int DLLibraryGetInfo(void *handle,char *type,char **mess)
{
  int  ierr, (*sfunc)(const char *,const char*,char **);

  PetscFunctionBegin;
  sfunc   = (int (*)(const char *,const char*,char **)) dlsym(handle,"DLLibraryInfo");
  if (!sfunc) {
    *mess = "No library information in the file\n";
  } else {
    ierr = (*sfunc)(0,type,mess);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DLLibraryRetrieve"
/*@C
   DLLibraryRetrieve - Copies a PETSc dynamic library from a remote location
     (if it is remote), indicates if it exits and its local name.

     Collective on MPI_Comm

   Input Parameters:
+   comm - processors that are opening the library
-   libname - name of the library, can be relative or absolute

   Output Parameter:
.   handle - library handle 

   Level: developer

   Notes:
   [[<http,ftp>://hostname]/directoryname/]filename[.so.1.0]

   $PETSC_ARCH and $BOPT occuring in directoryname and filename 
   will be replaced with appropriate values.
@*/
int DLLibraryRetrieve(MPI_Comm comm,const char libname[],char *lname,int llen,PetscTruth *found)
{
  char       *par2,ierr,len,*par3,arch[10],buff[10],*en,*gz;
  int        flg;

  PetscFunctionBegin;

  /* 
     make copy of library name and replace $PETSC_ARCH and $BOPT and 
     so we can add to the end of it to look for something like .so.1.0 etc.
  */
  len   = PetscStrlen(libname);
  par2  = (char *) PetscMalloc((1024)*sizeof(char));CHKPTRQ(par2);
  ierr  = PetscStrcpy(par2,libname);CHKERRQ(ierr);
  
  par3 = PetscStrstr(par2,"$PETSC_ARCH");
  while (par3) {
    *par3  =  0;
    par3  += 11;
    ierr   = PetscGetArchType(arch,10);
    ierr = PetscStrcat(par2,arch);CHKERRQ(ierr);
    ierr = PetscStrcat(par2,par3);CHKERRQ(ierr);
    par3 = PetscStrstr(par2,"$PETSC_ARCH");
  }

  par3 = PetscStrstr(par2,"$BOPT");
  while (par3) {
    *par3  =  0;
    par3  += 5;
    ierr = PetscStrcat(par2,PETSC_BOPT);CHKERRQ(ierr);
    ierr = PetscStrcat(par2,par3);CHKERRQ(ierr);
    par3 = PetscStrstr(par2,"$BOPT");
  }

  /* 
     Remove any file: header
  */
  flg = !PetscStrncmp(par2,"file:",5);
  if (flg) {
    ierr = PetscStrcpy(par2,par2+5);CHKERRQ(ierr);
  }

  /* strip out .a from it if user put it in by mistake */
  len    = PetscStrlen(par2);
  if (par2[len-1] == 'a' && par2[len-2] == '.') par2[len-2] = 0;

  /* remove .gz if it ends library name */
  if ((gz = PetscStrstr(par2,".gz")) && (PetscStrlen(gz) == 3)) {
    *gz = 0;
  }

  /* see if library name does already not have suffix attached */
  ierr = PetscStrcpy(buff,".");CHKERRQ(ierr);
  ierr = PetscStrcat(buff,PETSC_SLSUFFIX);CHKERRQ(ierr);
  if (!(en = PetscStrstr(par2,buff)) || (PetscStrlen(en) != 1+PetscStrlen(PETSC_SLSUFFIX))) {
    ierr = PetscStrcat(par2,".");CHKERRQ(ierr);
    ierr = PetscStrcat(par2,PETSC_SLSUFFIX);CHKERRQ(ierr);
  }

  /* put the .gz back on if it was there */
  if (gz) {
    ierr = PetscStrcat(par2,".gz");CHKERRQ(ierr);
  }

  ierr = PetscFileRetrieve(comm,par2,lname,llen,found);CHKERRQ(ierr);
  PetscFree(par2);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DLLibraryOpen"
/*@C
   DLLibraryOpen - Opens a dynamic link library

     Collective on MPI_Comm

   Input Parameters:
+   comm - processors that are opening the library
-   libname - name of the library, can be relative or absolute

   Output Parameter:
.   handle - library handle 

   Level: developer

   Notes:
   [[<http,ftp>://hostname]/directoryname/]filename[.so.1.0]

   $PETSC_ARCH and $BOPT occuring in directoryname and filename 
   will be replaced with appropriate values.
@*/
int DLLibraryOpen(MPI_Comm comm,const char libname[],void **handle)
{
  char       *par2,ierr;
  PetscTruth foundlibrary;
  int        (*func)(const char*);

  PetscFunctionBegin;

  par2 = (char *) PetscMalloc(1024*sizeof(char));CHKPTRQ(par2);
  ierr = DLLibraryRetrieve(comm,libname,par2,1024,&foundlibrary);CHKERRQ(ierr);
  if (!foundlibrary) {
    SETERRQ1(1,1,"Unable to locate dynamic library:\n  %s\n",libname);
  }

#if !defined(USE_NONEXECUTABLE_SO)
  ierr  = PetscTestFile(par2,'x',&foundlibrary);CHKERRQ(ierr);
  if (!foundlibrary) {
    SETERRQ2(1,1,"Dynamic library is not executable:\n  %s\n  %s\n",libname,par2);
  }
#endif

  /*
    Under linux open the executable itself in the hope it will
    resolve some symbols; doesn't seem to matter.
  */
#if defined(PARCH_linux)
  *handle = dlopen(0,RTLD_LAZY  |  RTLD_GLOBAL);
#endif

  /*
      Mode indicates symbols required by symbol loaded with dlsym() 
     are only loaded when required (not all together) also indicates
     symbols required can be contained in other libraries also opened
     with dlopen()
  */
  PLogInfo(0,"DLLibraryOpen:Opening %s\n",libname);
#if defined(HAVE_RTLD_GLOBAL)
  *handle = dlopen(par2,RTLD_LAZY  |  RTLD_GLOBAL); 
#else
  *handle = dlopen(par2,RTLD_LAZY); 
#endif

  if (!*handle) {
    SETERRQ3(1,1,"Unable to open dynamic library:\n  %s\n  %s\n  Error message from dlopen() %s\n",
             libname,par2,dlerror());
  }

  /* run the function FListAdd() if it is in the library */
  func  = (int (*)(const char *)) dlsym(*handle,"DLLibraryRegister");
  if (func) {
    ierr = (*func)(libname);CHKERRQ(ierr);
    PLogInfo(0,"DLLibraryOpen:Loading registered routines from %s\n",libname);
  }
  if (PLogPrintInfo) {
    int  (*sfunc)(const char *,const char*,char **);
    char *mess;

    sfunc   = (int (*)(const char *,const char*,char **)) dlsym(*handle,"DLLibraryInfo");
    if (sfunc) {
      ierr = (*sfunc)(libname,"Contents",&mess);CHKERRQ(ierr);
      if (mess) {
        PLogInfo(0,"Contents:\n %s",mess);
      }
      ierr = (*sfunc)(libname,"Authors",&mess);CHKERRQ(ierr);
      if (mess) {
        PLogInfo(0,"Authors:\n %s",mess);
      }
      ierr = (*sfunc)(libname,"Version",&mess);CHKERRQ(ierr);
      if (mess) {
        PLogInfo(0,"Version:\n %s\n",mess);
      }
    }
  }

  PetscFree(par2);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DLLibrarySym"
/*@C
   DLLibrarySym - Load a symbol from the dynamic link libraries.

   Collective on MPI_Comm

   Input Parameter:
+  path     - optional complete library name
-  insymbol - name of symbol

   Output Parameter:
.  value 

   Level: developer

   Notes: Symbol can be of the form
        [/path/libname[.so.1.0]:]functionname[()] where items in [] denote optional 

        Will attempt to (retrieve and) open the library if it is not yet been opened.

@*/
int DLLibrarySym(MPI_Comm comm,DLLibraryList *inlist,const char path[],
                 const char insymbol[], void **value)
{
  char          *par1,*symbol;
  int           ierr,len;
  DLLibraryList nlist,prev,list = *inlist;

  PetscFunctionBegin;
  *value = 0;

  /* make copy of symbol so we can edit it in place */
  len    = PetscStrlen(insymbol);
  symbol = (char *) PetscMalloc((len+1)*sizeof(char));CHKPTRQ(symbol);
  ierr   = PetscStrcpy(symbol,insymbol);CHKERRQ(ierr);

  /* 
      If symbol contains () then replace with a NULL, to support functionname() 
  */
  par1 = PetscStrchr(symbol,'(');
  if (par1) *par1 = 0;


  /*
       Function name does include library 
       -------------------------------------
  */
  if (path) {
    void *handle;
    
    /*   
        Look if library is already opened and in path
    */
    nlist = list;
    prev  = 0;
    while (nlist) {
      if (!PetscStrcmp(nlist->libname,path)) {
        handle = nlist->handle;
        goto done;
      }
      prev = nlist;
      nlist = nlist->next;
    }
    ierr = DLLibraryOpen(comm,path,&handle);CHKERRQ(ierr);

    nlist = (DLLibraryList) PetscMalloc(sizeof(struct _DLLibraryList));CHKPTRQ(list);
    nlist->next   = 0;
    nlist->handle = handle;
    ierr = PetscStrcpy(nlist->libname,path);CHKERRQ(ierr);

    if (prev) {
      prev->next = nlist;
    } else {
      *inlist    = list;
    }
    PLogInfo(0,"DLLibraryAppend:Appending %s to dynamic library search path\n",symbol);

    done:; 
    *value   = dlsym(handle,symbol);
    if (!*value) {
      SETERRQ2(1,1,"Unable to locate function %s in dynamic library %s",insymbol,path);
    }
    PLogInfo(0,"DLLibrarySym:Loading function %s from dynamic library %s\n",insymbol,path);

  /*
       Function name does not include library so search path
       -----------------------------------------------------
  */
  } else {
    while (list) {
      *value =  dlsym(list->handle,symbol);
      if (*value) {
        PLogInfo(0,"DLLibrarySym:Loading function %s from dynamic library %s\n",symbol,list->libname);
        break;
      }
      list = list->next;
    }
  }

  PetscFree(symbol);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DLLibraryAppend"
/*@C
     DLLibraryAppend - Appends another dynamic link library to the seach list, to the end
                of the search path.

     Collective on MPI_Comm

     Input Parameters:
+     comm - MPI communicator
-     libname - name of the library

     Output Parameter:
.     outlist - list of libraries

     Level: developer

     Notes: if library is already in path will not add it.
@*/
int DLLibraryAppend(MPI_Comm comm,DLLibraryList *outlist,const char libname[])
{
  DLLibraryList list,prev;
  void*         handle;
  int           ierr;

  PetscFunctionBegin;

  /* see if library was already open then we are done */
  list = prev = *outlist;
  while (list) {
    if (!PetscStrcmp(list->libname,libname)) {
      PetscFunctionReturn(0);
    }
    prev = list;
    list = list->next;
  }

  ierr = DLLibraryOpen(comm,libname,&handle);CHKERRQ(ierr);

  list = (DLLibraryList) PetscMalloc(sizeof(struct _DLLibraryList));CHKPTRQ(list);
  list->next   = 0;
  list->handle = handle;
  ierr = PetscStrcpy(list->libname,libname);CHKERRQ(ierr);

  if (!*outlist) {
    *outlist   = list;
  } else {
    prev->next = list;
  }
  PLogInfo(0,"DLLibraryAppend:Appending %s to dynamic library search path\n",libname);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DLLibraryPrepend"
/*@C
     DLLibraryPrepend - Add another dynamic library to search for symbols to the beginning of
                 the search path.

     Collective on MPI_Comm

     Input Parameters:
+     comm - MPI communicator
-     libname - name of the library

     Output Parameter:
.     outlist - list of libraries

     Level: developer

     Notes: If library is already in path will remove old reference.

@*/
int DLLibraryPrepend(MPI_Comm comm,DLLibraryList *outlist,const char libname[])
{
  DLLibraryList list,prev;
  void*         handle;
  int           ierr;

  PetscFunctionBegin;
 
  /* see if library was already open and move it to the front */
  list = *outlist;
  prev = 0;
  while (list) {
    if (!PetscStrcmp(list->libname,libname)) {
      if (prev) prev->next = list->next;
      list->next = *outlist;
      *outlist   = list;
      PetscFunctionReturn(0);
    }
    prev = list;
    list = list->next;
  }

  /* open the library and add to front of list */
  ierr = DLLibraryOpen(comm,libname,&handle);CHKERRQ(ierr);

  PLogInfo(0,"DLLibraryPrepend:Prepending %s to dynamic library search path\n",libname);

  list         = (DLLibraryList) PetscMalloc(sizeof(struct _DLLibraryList));CHKPTRQ(list);
  list->handle = handle;
  list->next   = *outlist;
  ierr = PetscStrcpy(list->libname,libname);CHKERRQ(ierr);
  *outlist     = list;

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DLLibraryClose"
/*@C
     DLLibraryClose - Destroys the search path of dynamic libraries and closes the libraries.

    Collective on DLLibrary

    Input Parameter:
.     next - library list

     Level: developer

@*/
int DLLibraryClose(DLLibraryList next)
{
  DLLibraryList prev;

  PetscFunctionBegin;

  while (next) {
    prev = next;
    next = next->next;
    /* free the space in the prev data-structure */
    PetscFree(prev);
  }
  PetscFunctionReturn(0);
}

#endif


