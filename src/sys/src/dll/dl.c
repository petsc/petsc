
/*$Id: dl.c,v 1.71 2001/04/10 19:34:28 bsmith Exp $*/
/*
      Routines for opening dynamic link libraries (DLLs), keeping a searchable
   path of DLLs, obtaining remote DLLs via a URL and opening them locally.
*/

#include "petsc.h"
#include "petscsys.h"
#include "petscfix.h"
#if defined(PETSC_HAVE_PWD_H)
#include <pwd.h>
#endif
#include <ctype.h>
#include <sys/types.h>
#include <sys/stat.h>
#if defined(PETSC_HAVE_UNISTD_H)
#include <unistd.h>
#endif
#if defined(PETSC_HAVE_STDLIB_H)
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
#if defined(PETSC_HAVE_SYS_SYSTEMINFO_H)
#include <sys/systeminfo.h>
#endif
#include "petscfix.h"


/*
   Contains the list of registered CCA components
*/
PetscFList CCAList = 0;


/* ------------------------------------------------------------------------------*/
/*
      Code to maintain a list of opened dynamic libraries and load symbols
*/
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
#include <dlfcn.h>

struct _PetscDLLibraryList {
  PetscDLLibraryList next;
  void          *handle;
  char          libname[PETSC_MAX_PATH_LEN];
};

EXTERN_C_BEGIN
extern int Petsc_DelTag(MPI_Comm,int,void*,void*);
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryPrintPath"
int PetscDLLibraryPrintPath(void)
{
  PetscDLLibraryList libs;

  PetscFunctionBegin;
  libs = DLLibrariesLoaded;
  while (libs) {
    PetscErrorPrintf("  %s\n",libs->libname);
    libs = libs->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryGetInfo"
/*@C
   PetscDLLibraryGetInfo - Gets the text information from a PETSc
       dynamic library

     Not Collective

   Input Parameters:
.   handle - library handle returned by PetscDLLibraryOpen()

   Level: developer

@*/
int PetscDLLibraryGetInfo(void *handle,const char type[],const char *mess[])
{
  int  ierr,(*sfunc)(const char *,const char*,const char *[]);

  PetscFunctionBegin;
  sfunc   = (int (*)(const char *,const char*,const char *[])) dlsym(handle,"PetscDLLibraryInfo");
  if (!sfunc) {
    *mess = "No library information in the file\n";
  } else {
    ierr = (*sfunc)(0,type,mess);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryRetrieve"
/*@C
   PetscDLLibraryRetrieve - Copies a PETSc dynamic library from a remote location
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

   ${PETSC_ARCH}, ${PETSC_DIR}, ${PETSC_LIB_DIR}, ${BOPT}, or ${any environmental variable}
   occuring in directoryname and filename will be replaced with appropriate values.
@*/
int PetscDLLibraryRetrieve(MPI_Comm comm,const char libname[],char *lname,int llen,PetscTruth *found)
{
  char       *par2,buff[10],*en,*gz;
  int        ierr,len1,len2,len;
  PetscTruth tflg,flg;

  PetscFunctionBegin;

  /* 
     make copy of library name and replace $PETSC_ARCH and $BOPT and 
     so we can add to the end of it to look for something like .so.1.0 etc.
  */
  ierr = PetscStrlen(libname,&len);CHKERRQ(ierr);
  len  = PetscMax(4*len,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscMalloc(len*sizeof(char),&par2);CHKERRQ(ierr);
  ierr = PetscStrreplace(comm,libname,par2,len);CHKERRQ(ierr);

  /* 
     Remove any file: header
  */
  ierr = PetscStrncmp(par2,"file:",5,&tflg);CHKERRQ(ierr);
  if (tflg) {
    ierr = PetscStrcpy(par2,par2+5);CHKERRQ(ierr);
  }

  /* strip out .a from it if user put it in by mistake */
  ierr    = PetscStrlen(par2,&len);CHKERRQ(ierr);
  if (par2[len-1] == 'a' && par2[len-2] == '.') par2[len-2] = 0;

  /* remove .gz if it ends library name */
  ierr = PetscStrstr(par2,".gz",&gz);CHKERRQ(ierr);
  if (gz) {
    ierr = PetscStrlen(gz,&len);CHKERRQ(ierr);
    if (len == 3) {
      *gz = 0;
    }
  }

  /* see if library name does already not have suffix attached */
  ierr = PetscStrcpy(buff,".");CHKERRQ(ierr);
  ierr = PetscStrcat(buff,PETSC_SLSUFFIX);CHKERRQ(ierr);
  ierr = PetscStrstr(par2,buff,&en);CHKERRQ(ierr);
  if (en) {
    ierr = PetscStrlen(en,&len1);CHKERRQ(ierr);
    ierr = PetscStrlen(PETSC_SLSUFFIX,&len2);CHKERRQ(ierr); 
    flg = (PetscTruth) (len1 != 1 + len2);
  } else {
    flg = PETSC_TRUE;
  }
  if (flg) {
    ierr = PetscStrcat(par2,".");CHKERRQ(ierr);
    ierr = PetscStrcat(par2,PETSC_SLSUFFIX);CHKERRQ(ierr);
  }

  /* put the .gz back on if it was there */
  if (gz) {
    ierr = PetscStrcat(par2,".gz");CHKERRQ(ierr);
  }

  ierr = PetscFileRetrieve(comm,par2,lname,llen,found);CHKERRQ(ierr);
  ierr = PetscFree(par2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryOpen"
/*@C
   PetscDLLibraryOpen - Opens a dynamic link library

     Collective on MPI_Comm

   Input Parameters:
+   comm - processors that are opening the library
-   libname - name of the library, can be relative or absolute

   Output Parameter:
.   handle - library handle 

   Level: developer

   Notes:
   [[<http,ftp>://hostname]/directoryname/]filename[.so.1.0]

   ${PETSC_ARCH} and ${BOPT} occuring in directoryname and filename 
   will be replaced with appropriate values.
@*/
int PetscDLLibraryOpen(MPI_Comm comm,const char libname[],void **handle)
{
  char       *par2,ierr;
  PetscTruth foundlibrary;
  int        (*func)(const char*);

  PetscFunctionBegin;

  ierr = PetscMalloc(1024*sizeof(char),&par2);CHKERRQ(ierr);
  ierr = PetscDLLibraryRetrieve(comm,libname,par2,1024,&foundlibrary);CHKERRQ(ierr);
  if (!foundlibrary) {
    SETERRQ1(1,"Unable to locate dynamic library:\n  %s\n",libname);
  }

#if !defined(PETSC_USE_NONEXECUTABLE_SO)
  ierr  = PetscTestFile(par2,'x',&foundlibrary);CHKERRQ(ierr);
  if (!foundlibrary) {
    SETERRQ2(1,"Dynamic library is not executable:\n  %s\n  %s\n",libname,par2);
  }
#endif

  /*
      Mode indicates symbols required by symbol loaded with dlsym() 
     are only loaded when required (not all together) also indicates
     symbols required can be contained in other libraries also opened
     with dlopen()
  */
  PetscLogInfo(0,"PetscDLLibraryOpen:Opening %s\n",libname);
#if defined(PETSC_HAVE_RTLD_GLOBAL)
  *handle = dlopen(par2,RTLD_LAZY  |  RTLD_GLOBAL); 
#else
  *handle = dlopen(par2,RTLD_LAZY); 
#endif

  if (!*handle) {
    SETERRQ3(1,"Unable to open dynamic library:\n  %s\n  %s\n  Error message from dlopen() %s\n",libname,par2,dlerror());
  }

  /* run the function PetscFListAddDynamic() if it is in the library */
  func  = (int (*)(const char *)) dlsym(*handle,"PetscDLLibraryRegister");
  if (func) {
    ierr = (*func)(libname);CHKERRQ(ierr);
    PetscLogInfo(0,"PetscDLLibraryOpen:Loading registered routines from %s\n",libname);
  }
  if (PetscLogPrintInfo) {
    int  (*sfunc)(const char *,const char*,char **);
    char *mess;

    sfunc   = (int (*)(const char *,const char*,char **)) dlsym(*handle,"PetscDLLibraryInfo");
    if (sfunc) {
      ierr = (*sfunc)(libname,"Contents",&mess);CHKERRQ(ierr);
      if (mess) {
        PetscLogInfo(0,"Contents:\n %s",mess);
      }
      ierr = (*sfunc)(libname,"Authors",&mess);CHKERRQ(ierr);
      if (mess) {
        PetscLogInfo(0,"Authors:\n %s",mess);
      }
      ierr = (*sfunc)(libname,"Version",&mess);CHKERRQ(ierr);
      if (mess) {
        PetscLogInfo(0,"Version:\n %s\n",mess);
      }
    }
  }

  /* Look for CCA components in the library */
#if defined(__cplusplus) && !defined(PETSC_USE_COMPLEX)
  char **(*gcl)(void) = (char **(*)(void)) dlsym(*handle,"getESIFactoryList");
  if (gcl) {
    char       **list = (*gcl)(),*sname,*rname;
    int        i = 0;
    PetscToken *token;

    while (list[i]) {
      PetscLogInfo(0,"ESI factory function and name: %s from %s\n",list[i],libname);
      ierr = PetscTokenCreate(list[i],' ',&token);CHKERRQ(ierr);
      ierr = PetscTokenFind(token,&rname);CHKERRQ(ierr);
      ierr = PetscTokenFind(token,&sname);CHKERRQ(ierr);
      ierr = PetscFListAdd(&CCAList,sname,rname,PETSC_NULL);CHKERRQ(ierr);
      ierr = PetscTokenDestroy(token);CHKERRQ(ierr);
      i++;
    }
  }
#endif

  ierr = PetscFree(par2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibrarySym"
/*@C
   PetscDLLibrarySym - Load a symbol from the dynamic link libraries.

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
int PetscDLLibrarySym(MPI_Comm comm,PetscDLLibraryList *inlist,const char path[],const char insymbol[],void **value)
{
  char               *par1,*symbol;
  int                ierr,len;
  PetscDLLibraryList nlist,prev,list;

  PetscFunctionBegin;
  if (inlist) list = *inlist; else list = PETSC_NULL;
  *value = 0;

  /* make copy of symbol so we can edit it in place */
  ierr = PetscStrlen(insymbol,&len);CHKERRQ(ierr);
  ierr = PetscMalloc((len+1)*sizeof(char),&symbol);CHKERRQ(ierr);
  ierr = PetscStrcpy(symbol,insymbol);CHKERRQ(ierr);

  /* 
      If symbol contains () then replace with a NULL, to support functionname() 
  */
  ierr = PetscStrchr(symbol,'(',&par1);CHKERRQ(ierr);
  if (par1) *par1 = 0;


  /*
       Function name does include library 
       -------------------------------------
  */
  if (path && path[0] != '\0') {
    void *handle;
    
    /*   
        Look if library is already opened and in path
    */
    nlist = list;
    prev  = 0;
    while (nlist) {
      PetscTruth match;

      ierr = PetscStrcmp(nlist->libname,path,&match);CHKERRQ(ierr);
      if (match) {
        handle = nlist->handle;
        goto done;
      }
      prev  = nlist;
      nlist = nlist->next;
    }
    ierr = PetscDLLibraryOpen(comm,path,&handle);CHKERRQ(ierr);

    ierr          = PetscNew(struct _PetscDLLibraryList,&nlist);CHKERRQ(ierr);
    nlist->next   = 0;
    nlist->handle = handle;
    ierr = PetscStrcpy(nlist->libname,path);CHKERRQ(ierr);

    if (prev) {
      prev->next = nlist;
    } else {
      if (inlist) *inlist = nlist;
      else {ierr = PetscDLLibraryClose(nlist);CHKERRQ(ierr);}
    }
    PetscLogInfo(0,"PetscDLLibraryAppend:Appending %s to dynamic library search path\n",path);

    done:; 
    *value   = dlsym(handle,symbol);
    if (!*value) {
      SETERRQ2(1,"Unable to locate function %s in dynamic library %s",insymbol,path);
    }
    PetscLogInfo(0,"PetscDLLibrarySym:Loading function %s from dynamic library %s\n",insymbol,path);

  /*
       Function name does not include library so search path
       -----------------------------------------------------
  */
  } else {
    while (list) {
      *value =  dlsym(list->handle,symbol);
      if (*value) {
        PetscLogInfo(0,"PetscDLLibrarySym:Loading function %s from dynamic library %s\n",symbol,list->libname);
        break;
      }
      list = list->next;
    }
    if (!*value) {
      *value =  dlsym(0,symbol);
      if (*value) {
        PetscLogInfo(0,"PetscDLLibrarySym:Loading function %s from object code\n",symbol);
      }
    }
  }

  ierr = PetscFree(symbol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryAppend"
/*@C
     PetscDLLibraryAppend - Appends another dynamic link library to the seach list, to the end
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
int PetscDLLibraryAppend(MPI_Comm comm,PetscDLLibraryList *outlist,const char libname[])
{
  PetscDLLibraryList list,prev;
  void*              handle;
  int                ierr,len;
  PetscTruth         match,dir;
  char               program[PETSC_MAX_PATH_LEN],buf[8*PETSC_MAX_PATH_LEN],*found,*libname1,suffix[16],*s;
  PetscToken         *token;

  PetscFunctionBegin;

  /* is libname a directory? */
  ierr = PetscTestDirectory(libname,'r',&dir);CHKERRQ(ierr);
  if (dir) {
    PetscLogInfo(0,"Checking directory %s for dynamic libraries\n",libname);
    ierr  = PetscStrcpy(program,libname);CHKERRQ(ierr);
    ierr  = PetscStrlen(program,&len);CHKERRQ(ierr);
    if (program[len-1] == '/') {
      ierr  = PetscStrcat(program,"*.");CHKERRQ(ierr);
    } else {
      ierr  = PetscStrcat(program,"/*.");CHKERRQ(ierr);
    }
    ierr  = PetscStrcat(program,PETSC_SLSUFFIX);CHKERRQ(ierr);

    ierr = PetscLs(comm,program,buf,8*PETSC_MAX_PATH_LEN,&dir);CHKERRQ(ierr);
    if (!dir) PetscFunctionReturn(0);
    found = buf;
  } else {
    found = (char*)libname;
  }
  ierr = PetscStrcpy(suffix,".");CHKERRQ(ierr);
  ierr = PetscStrcat(suffix,PETSC_SLSUFFIX);CHKERRQ(ierr);

  ierr = PetscTokenCreate(found,'\n',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&libname1);CHKERRQ(ierr);
  ierr = PetscStrstr(libname1,suffix,&s);CHKERRQ(ierr);
  if (s) s[0] = 0;
  while (libname1) {

    /* see if library was already open then we are done */
    list  = prev = *outlist;
    match = PETSC_FALSE;
    while (list) {

      ierr = PetscStrcmp(list->libname,libname1,&match);CHKERRQ(ierr);
      if (match) break;
      prev = list;
      list = list->next;
    }
    if (!match) {

      ierr = PetscDLLibraryOpen(comm,libname1,&handle);CHKERRQ(ierr);

      ierr         = PetscNew(struct _PetscDLLibraryList,&list);CHKERRQ(ierr);
      list->next   = 0;
      list->handle = handle;
      ierr = PetscStrcpy(list->libname,libname1);CHKERRQ(ierr);

      if (!*outlist) {
	*outlist   = list;
      } else {
	prev->next = list;
      }
      PetscLogInfo(0,"PetscDLLibraryAppend:Appending %s to dynamic library search path\n",libname1);
    }
    ierr = PetscTokenFind(token,&libname1);CHKERRQ(ierr);
    if (libname1) {
      ierr = PetscStrstr(libname1,suffix,&s);CHKERRQ(ierr);
      if (s) s[0] = 0;
    }
  }
  ierr = PetscTokenDestroy(token);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryPrepend"
/*@C
     PetscDLLibraryPrepend - Add another dynamic library to search for symbols to the beginning of
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
int PetscDLLibraryPrepend(MPI_Comm comm,PetscDLLibraryList *outlist,const char libname[])
{
  PetscDLLibraryList list,prev;
  void*              handle;
  int                ierr,len;
  PetscTruth         match,dir;
  char               program[PETSC_MAX_PATH_LEN],buf[8*PETSC_MAX_PATH_LEN],*found,*libname1,suffix[16],*s;
  PetscToken         *token;

  PetscFunctionBegin;
 
  /* is libname a directory? */
  ierr = PetscTestDirectory(libname,'r',&dir);CHKERRQ(ierr);
  if (dir) {
    PetscLogInfo(0,"Checking directory %s for dynamic libraries\n",libname);
    ierr  = PetscStrcpy(program,libname);CHKERRQ(ierr);
    ierr  = PetscStrlen(program,&len);CHKERRQ(ierr);
    if (program[len-1] == '/') {
      ierr  = PetscStrcat(program,"*.");CHKERRQ(ierr);
    } else {
      ierr  = PetscStrcat(program,"/*.");CHKERRQ(ierr);
    }
    ierr  = PetscStrcat(program,PETSC_SLSUFFIX);CHKERRQ(ierr);

    ierr = PetscLs(comm,program,buf,8*PETSC_MAX_PATH_LEN,&dir);CHKERRQ(ierr);
    if (!dir) PetscFunctionReturn(0);
    found = buf;
  } else {
    found = (char*)libname;
  }

  ierr = PetscStrcpy(suffix,".");CHKERRQ(ierr);
  ierr = PetscStrcat(suffix,PETSC_SLSUFFIX);CHKERRQ(ierr);

  ierr = PetscTokenCreate(found,'\n',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&libname1);CHKERRQ(ierr);
  ierr = PetscStrstr(libname1,suffix,&s);CHKERRQ(ierr);
  if (s) s[0] = 0;
  while (libname1) {
    /* see if library was already open and move it to the front */
    list  = *outlist;
    prev  = 0;
    match = PETSC_FALSE;
    while (list) {

      ierr = PetscStrcmp(list->libname,libname1,&match);CHKERRQ(ierr);
      if (match) {
	if (prev) prev->next = list->next;
	list->next = *outlist;
	*outlist   = list;
	break;
      }
      prev = list;
      list = list->next;
    }
    if (!match) {
      /* open the library and add to front of list */
      ierr = PetscDLLibraryOpen(comm,libname1,&handle);CHKERRQ(ierr);
      
      PetscLogInfo(0,"PetscDLLibraryPrepend:Prepending %s to dynamic library search path\n",libname1);

      ierr         = PetscNew(struct _PetscDLLibraryList,&list);CHKERRQ(ierr);
      list->handle = handle;
      list->next   = *outlist;
      ierr = PetscStrcpy(list->libname,libname1);CHKERRQ(ierr);
      *outlist     = list;
    }
    ierr = PetscTokenFind(token,&libname1);CHKERRQ(ierr);
    if (libname1) {
      ierr = PetscStrstr(libname1,suffix,&s);CHKERRQ(ierr);
      if (s) s[0] = 0;
    }
  }
  ierr = PetscTokenDestroy(token);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryClose"
/*@C
     PetscDLLibraryClose - Destroys the search path of dynamic libraries and closes the libraries.

    Collective on PetscDLLibrary

    Input Parameter:
.     next - library list

     Level: developer

@*/
int PetscDLLibraryClose(PetscDLLibraryList next)
{
  PetscDLLibraryList prev;
  int           ierr;

  PetscFunctionBegin;

  while (next) {
    prev = next;
    next = next->next;
    /* free the space in the prev data-structure */
    ierr = PetscFree(prev);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryCCAAppend"
/*@C
     PetscDLLibraryCCAAppend - Appends another CCA dynamic link library to the seach list, to the end
                of the search path.

     Collective on MPI_Comm

     Input Parameters:
+     comm - MPI communicator
-     libname - name of directory to check

     Output Parameter:
.     outlist - list of libraries

     Level: developer

     Notes: if library is already in path will not add it.
@*/
int PetscDLLibraryCCAAppend(MPI_Comm comm,PetscDLLibraryList *outlist,const char dirname[])
{
  int                ierr,l;
  PetscTruth         dir;
  char               program[PETSC_MAX_PATH_LEN],buf[8*PETSC_MAX_PATH_LEN],*libname1,fbuf[PETSC_MAX_PATH_LEN],*found,suffix[16],*f2;
  char               *func,*funcname,libname[PETSC_MAX_PATH_LEN],*lib;
  FILE               *fp;
  PetscToken         *token1,*token2;

  PetscFunctionBegin;

  /* is dirname a directory? */
  ierr = PetscTestDirectory(dirname,'r',&dir);CHKERRQ(ierr);
  if (!dir) PetscFunctionReturn(0);

  PetscLogInfo(0,"Checking directory %s for CCA components\n",dirname);
  ierr  = PetscStrcpy(program,dirname);CHKERRQ(ierr);
  ierr  = PetscStrcat(program,"/*.cca");CHKERRQ(ierr);

  ierr = PetscLs(comm,program,buf,8*PETSC_MAX_PATH_LEN,&dir);CHKERRQ(ierr);
  if (!dir) PetscFunctionReturn(0);

  ierr = PetscStrcpy(suffix,".");CHKERRQ(ierr);
  ierr = PetscStrcat(suffix,PETSC_SLSUFFIX);CHKERRQ(ierr);
  ierr = PetscTokenCreate(buf,'\n',&token1);CHKERRQ(ierr);
  ierr = PetscTokenFind(token1,&libname1);CHKERRQ(ierr);
  while (libname1) {
    fp    = fopen(libname1,"r"); if (!fp) continue;
    while ((found = fgets(fbuf,1024,fp))) {
      if (found[0] == '!') continue;
      ierr = PetscStrstr(found,suffix,&f2);CHKERRQ(ierr);
      if (f2) { /* found library name */
        if (found[0] == '/') {
          lib = found;
        } else {
          ierr = PetscStrcpy(libname,dirname);CHKERRQ(ierr); 
          ierr = PetscStrlen(libname,&l);CHKERRQ(ierr);
          if (libname[l-1] != '/') {ierr = PetscStrcat(libname,"/");CHKERRQ(ierr);}
          ierr = PetscStrcat(libname,found);CHKERRQ(ierr); 
          lib  = libname;
        }
        ierr = PetscDLLibraryAppend(comm,outlist,lib);CHKERRQ(ierr);
      } else {
        PetscLogInfo(0,"CCA Component function and name: %s from %s\n",found,libname1);
        ierr = PetscTokenCreate(found,' ',&token2);CHKERRQ(ierr);
        ierr = PetscTokenFind(token2,&func);CHKERRQ(ierr);
        ierr = PetscTokenFind(token2,&funcname);CHKERRQ(ierr);
        ierr = PetscFListAdd(&CCAList,funcname,func,PETSC_NULL);CHKERRQ(ierr);
        ierr = PetscTokenDestroy(token2);CHKERRQ(ierr);
      }
    }
    fclose(fp);
    ierr = PetscTokenFind(token1,&libname1);CHKERRQ(ierr);
  }
  ierr = PetscTokenDestroy(token1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#endif


