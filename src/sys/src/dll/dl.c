
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: dl.c,v 1.26 1998/07/23 14:46:29 balay Exp balay $";
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
#if !defined(PARCH_nt)
#include <sys/utsname.h>
#endif
#if defined(PARCH_nt)
#include <windows.h>
#include <io.h>
#include <direct.h>
#endif
#if defined (PARCH_nt_gnu)
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

int DLLibraryPrintPath()
{
  DLLibraryList libs;

  PetscFunctionBegin;
  PetscErrorPrintf("Unable to find function. Search path:\n");
  libs = DLLibrariesLoaded;
  while (libs) {
    PetscErrorPrintf("%s\n",libs->libname);
    libs = libs->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DLLibrarySharedTmp"
/*
      Assumes that all processors in a communicator either
       1) have a common /tmp or
       2) each has a seperate /tmp

      Stores the status as a MPI attribute so it does not have
    to be redetermined each time.
*/
int DLLibrarySharedTmp(MPI_Comm comm,int *shared)
{
  int        ierr,size,rank,flag;
  FILE       *fd;
  static int Petsc_Tmp_keyval = MPI_KEYVAL_INVALID;
  int        *tagvalp;

  PetscFunctionBegin;
  MPI_Comm_size(comm,&size);
  if (size == 1) {
    *shared = 1;
    PetscFunctionReturn(0);
  }

  if (Petsc_Tmp_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,Petsc_DelTag,&Petsc_Tmp_keyval,0);CHKERRQ(ierr);
  }

  ierr = MPI_Attr_get(comm,Petsc_Tmp_keyval,(void**)&tagvalp,&flag);CHKERRQ(ierr);
  if (!flag) {
    /* This communicator does not yet have a shared tmp attribute */
    tagvalp    = (int *) PetscMalloc( sizeof(int) ); CHKPTRQ(tagvalp);
    ierr       = MPI_Attr_put(comm,Petsc_Tmp_keyval, tagvalp);CHKERRQ(ierr);

    ierr = MPI_Comm_rank(comm,&rank);
    if (rank == 0) {
      fd = fopen("/tmp/petsctestshared","w");
      if (!fd) {
        SETERRQ(1,1,"Unable to open test file in /tmp directory");
      }
      fclose(fd);
    }
    ierr = MPI_Barrier(comm);CHKERRQ(ierr);
    if (rank == 1) {
      fd = fopen("/tmp/petsctestshared","r");
      if (fd) *shared = 1; else *shared = 0;
      if (fd) {
        fclose(fd);
      }
    }
    ierr = MPI_Bcast(shared,1,MPI_INT,1,comm);CHKERRQ(ierr);
    if (rank == 0) {
      unlink("/tmp/petsctestshared");
    }
    *tagvalp = *shared;
  } else {
    *shared = *tagvalp;
  }
  PLogInfo(0,"DLLibrarySharedTmp:1 indicates detected shared /tmp, 0 not shared %d\n",*shared);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "DLLibraryObtain"
/*@C
    DLLibraryObtain - Obtains a library from a URL and copies into local disk space.

    Collective on MPI_Comm

    Input Parameter:
+   comm - processors accessing the library
.   libname - name of library, including entire URL
-   llen     - length of llibname

    Output Parameter:
.   llibname - name of local copy of library

@*/
int DLLibraryObtain(MPI_Comm comm,char *libname,char *llibname,int llen)
{
  char       *par4,buf[1024];
  FILE       *fp;
  int        i,rank,ierr,sharedtmp;

  PetscFunctionBegin;

  /* Determine if all processors share a common /tmp */
  ierr = DLLibrarySharedTmp(comm,&sharedtmp);CHKERRQ(ierr);

  if (PetscStrncmp(libname,"ftp://",6) && PetscStrncmp(libname,"http://",7)) {
    SETERRQ(1,1,"Only support for ftp/http DLL retrieval with python");
  }
    
  MPI_Comm_rank(comm,&rank);
  if (!rank || !sharedtmp) {
  
    /* Construct the Python script to get URL file */
    par4 = (char *) PetscMalloc(1024*sizeof(char));CHKPTRQ(par4);
    PetscStrcpy(par4,"python1.5 ");
    PetscStrcat(par4,PETSC_DIR);
    PetscStrcat(par4,"/bin/urlget.py ");
    PetscStrcat(par4,libname);

    PLogInfo(0,"DLLibraryObtain: Running python script:%s\n",par4);
    if ((fp = popen(par4,"r")) == 0) {
      SETERRQ(1,1,"Cannot Execute python1.5 on ${PETSC_DIR}/bin/urlget.py\n\
        Check if python1.5 is in your path");
    }
    if (fgets(buf,1024,fp) == 0) {
      SETERRQ(1,1,"No output from ${PETSC_DIR}/bin/urlget.py");
    }
    /* Check for \n and make it 0 */
    for ( i=0; i<1024; i++ ) {
      if ( buf[i] == '\n') {
        buf[i] = 0;
        break;
      }
    }
    if (!PetscStrncmp(buf,"Error",5) ||!PetscStrncmp(buf,"Traceback",9)) {SETERRQ(1,1,buf);}
    PetscStrncpy(llibname,buf,llen);
    PetscFree(par4);
  }
  if (sharedtmp) { /* send library name to all processors */
    MPI_Bcast(llibname,llen,MPI_CHAR,0,comm);
  }

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

   Notes:
   [[<http,ftp>://hostname]/directoryname/]filename[.so.1.0]

   $PETSC_ARCH and $BOPT occuring in directoryname and filename 
   will be replaced with appropriate values.
@*/
int DLLibraryOpen(MPI_Comm comm,const char libname[],void **handle)
{
  char       *par2,ierr,len,*par3,arch[10];
  PetscTruth foundlibrary;
  int        flg;
  int        (*func)(const char*);

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
    PetscStrcat(par2,arch);
    PetscStrcat(par2,par3);
    par3 = PetscStrstr(par2,"$PETSC_ARCH");
  }

  par3 = PetscStrstr(par2,"$BOPT");
  while (par3) {
    *par3  =  0;
    par3  += 5;
    PetscStrcat(par2,PETSC_BOPT);
    PetscStrcat(par2,par3);
    par3 = PetscStrstr(par2,"$BOPT");
  }

  /* 
     Remove any file: header
  */
  flg = !PetscStrncmp(par2,"file:",5);
  if (flg) {
    PetscStrcpy(par2,par2+5);
  }

  /* strip out .a from it if user put it in by mistake */
  len    = PetscStrlen(par2);
  if (par2[len-1] == 'a' && par2[len-2] == '.') par2[len-2] = 0;

  /*
     Is it an ftp or http library?
  */
  flg = !PetscStrncmp(par2,"ftp://",6) || !PetscStrncmp(par2,"http://",7);
  if (flg) {

    /* see if library name does not have suffix attached */
    if (par2 == PetscStrrchr(par2,'.')) { /* no period in name, so append prefix */
      PetscStrcat(par2,".");
      PetscStrcat(par2,PETSC_SLSUFFIX);
    } else if (par2[0] != 's' || par2[1] != 'o') {
      PetscStrcat(par2,".");
      PetscStrcat(par2,PETSC_SLSUFFIX);
    }

    par3 = (char *) PetscMalloc(1024*sizeof(char));CHKPTRQ(par3);
    ierr = DLLibraryObtain(comm,par2,par3,1024);CHKERRQ(ierr);
    PetscStrcpy(par2,par3);
    PetscFree(par3);
  }

  /* first check original given name */
  ierr  = PetscTestFile(par2,'x',&foundlibrary);CHKERRQ(ierr);
  if (!foundlibrary) {
    /* try appending .so suffix */
    PetscStrcat(par2,".");
    PetscStrcat(par2,PETSC_SLSUFFIX);
    ierr  = PetscTestFile(par2,'x',&foundlibrary);CHKERRQ(ierr);
    if (!foundlibrary) {
      PetscErrorPrintf("Library name %s\n  %s\n",libname,par2);
      SETERRQ(1,1,"Unable to locate dynamic library");
    }
  }

  *handle = dlopen(par2,1);    
  if (!*handle) {
    PetscErrorPrintf("Error message from dlopen() %s\n",dlerror());    
    PetscErrorPrintf("Library name %s\n  %s\n",libname,par2);
    SETERRQ(1,1,"Unable to open dynamic library");
  }

  /* run the function FListAdd() if it is in the library */
  func  = (int (*)(const char *)) dlsym(*handle,"DLLibraryRegister");
  if (func) {
    ierr = (*func)(libname);CHKERRQ(ierr);
    PLogInfo(0,"DLLibraryOpen:Loading registered routines from %s\n",libname);
  }
  if (PLogPrintInfo) {
    char *(*sfunc)(const char *,const char*),*mess;

    sfunc   = (char *(*)(const char *,const char*)) dlsym(*handle,"DLLibraryInfo");
    if (sfunc) {
      mess = (*sfunc)(libname,"Contents");
      if (mess) {
        PLogInfo(0,"Contents:\n %s",mess);
      }
      mess = (*sfunc)(libname,"Authors");
      if (mess) {
        PLogInfo(0,"Authors:\n %s",mess);
      }
      mess = (*sfunc)(libname,"Version");
      if (mess) {
        PLogInfo(0,"Version:\n %s\n",mess);
      }
    }
  }

  PetscFree(par2);
  PetscFunctionReturn(0);
}

/*@C
   DLLibrarySym - Load a symbol from the dynamic link libraries.

   Collective on MPI_Comm

   Input Parameter:
+  path     - optional complete library name
-  insymbol - name of symbol

   Output Parameter:
.  value 

   Notes: Symbol can be of the form
        [/path/libname[.so.1.0]:]functionname[()] where items in [] denote optional 

        Will attempt to (retrieve and) open the library if it is not yet been opened.

@*/
#undef __FUNC__  
#define __FUNC__ "DLLibrarySym"
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
    PetscStrcpy(nlist->libname,path);

    if (prev) {
      prev->next = nlist;
    } else {
      *inlist    = list;
    }
    PLogInfo(0,"DLLibraryAppend:Appending %s to dynamic library search path\n",symbol);

    done:; 
    *value   = dlsym(handle,symbol);
    if (!*value) {
      PetscErrorPrintf("Library path %s and function name %s\n",path,insymbol);
      SETERRQ(1,1,"Unable to locate function in dynamic library");
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

/*@C
     DLLibraryAppend - Appends another dynamic link library to the seach list, to the end
                of the search path.

     Collective on MPI_Comm

     Input Parameters:
+     comm - MPI communicator
-     libname - name of the library

     Output Parameter:
.     outlist - list of libraries

     Notes: if library is already in path will not add it.
@*/
#undef __FUNC__  
#define __FUNC__ "DLLibraryAppend"
int DLLibraryAppend(MPI_Comm comm,DLLibraryList *outlist,const char libname[])
{
  DLLibraryList list,prev;
  void*         handle;
  int           ierr;

  PetscFunctionBegin;

  /* see if library was already open then we are done */
  list = *outlist;
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
  PetscStrcpy(list->libname,libname);

  if (!*outlist) {
    *outlist   = list;
  } else {
    prev->next = list;
  }
  PLogInfo(0,"DLLibraryAppend:Appending %s to dynamic library search path\n",libname);
  PetscFunctionReturn(0);
}

/*@C
     DLLibraryPrepend - Add another dynamic library to search for symbols to the beginning of
                 the search path.

     Collective on MPI_Comm

     Input Parameters:
+     comm - MPI communicator
-     libname - name of the library

     Output Parameter:
.     outlist - list of libraries

     Notes: If library is already in path will remove old reference.

@*/
#undef __FUNC__  
#define __FUNC__ "DLLibraryPrepend"
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
  PetscStrcpy(list->libname,libname);
  *outlist     = list;

  PetscFunctionReturn(0);
}

/*@C
     DLLibraryClose - Destroys the search path of dynamic libraries and closes the libraries.

    Collective on DLLibrary

    Input Parameter:
.     next - library list

@*/
#undef __FUNC__  
#define __FUNC__ "DLLibraryClose"
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


