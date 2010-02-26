#define PETSC_DLL
/*
      Routines for opening dynamic link libraries (DLLs), keeping a searchable
   path of DLLs, obtaining remote DLLs via a URL and opening them locally.
*/

#include "petscsys.h"
#include "../src/sys/dll/dlimpl.h"

/* ------------------------------------------------------------------------------*/
/*
      Code to maintain a list of opened dynamic libraries and load symbols
*/
struct _n_PetscDLLibrary {
  PetscDLLibrary next;
  PetscDLHandle  handle;
  char           libname[PETSC_MAX_PATH_LEN];
};

#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryPrintPath"
PetscErrorCode PETSC_DLLEXPORT PetscDLLibraryPrintPath(PetscDLLibrary libs)
{
  PetscFunctionBegin;
  while (libs) {
    PetscErrorPrintf("  %s\n",libs->libname);
    libs = libs->next;
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

   ${PETSC_ARCH}, ${PETSC_DIR}, ${PETSC_LIB_DIR}, or ${any environmental variable}
   occuring in directoryname and filename will be replaced with appropriate values.
@*/
PetscErrorCode PETSC_DLLEXPORT PetscDLLibraryRetrieve(MPI_Comm comm,const char libname[],char *lname,size_t llen,PetscTruth *found)
{
  char           *buf,*par2,suffix[16],*gz,*so;
  size_t         len;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* 
     make copy of library name and replace $PETSC_ARCH etc
     so we can add to the end of it to look for something like .so.1.0 etc.
  */
  ierr = PetscStrlen(libname,&len);CHKERRQ(ierr);
  len  = PetscMax(4*len,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscMalloc(len*sizeof(char),&buf);CHKERRQ(ierr);
  par2 = buf;
  ierr = PetscStrreplace(comm,libname,par2,len);CHKERRQ(ierr);

  /* temporarily remove .gz if it ends library name */
  ierr = PetscStrrstr(par2,".gz",&gz);CHKERRQ(ierr);
  if (gz) {
    ierr = PetscStrlen(gz,&len);CHKERRQ(ierr);
    if (len != 3) gz  = 0; /* do not end (exactly) with .gz */
    else          *gz = 0; /* ends with .gz, so remove it   */
  }
  /* strip out .a from it if user put it in by mistake */
  ierr = PetscStrlen(par2,&len);CHKERRQ(ierr);
  if (par2[len-1] == 'a' && par2[len-2] == '.') par2[len-2] = 0;


  /* see if library name does already not have suffix attached */
  ierr = PetscStrcpy(suffix,".");CHKERRQ(ierr);
  ierr = PetscStrcat(suffix,PETSC_SLSUFFIX);CHKERRQ(ierr);
  ierr = PetscStrrstr(par2,suffix,&so);CHKERRQ(ierr);
  /* and attach the suffix if it is not there */
  if (!so) { ierr = PetscStrcat(par2,suffix);CHKERRQ(ierr); }

  /* restore the .gz suffix if it was there */
  if (gz) { ierr = PetscStrcat(par2,".gz");CHKERRQ(ierr); }

  /* and finally retrieve the file */
  ierr = PetscFileRetrieve(comm,par2,lname,llen,found);CHKERRQ(ierr);

  ierr = PetscFree(buf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryOpen"
/*@C
   PetscDLLibraryOpen - Opens a PETSc dynamic link library

     Collective on MPI_Comm

   Input Parameters:
+   comm - processors that are opening the library
-   path - name of the library, can be relative or absolute

   Output Parameter:
.   entry - a PETSc dynamic link library entry

   Level: developer

   Notes:
   [[<http,ftp>://hostname]/directoryname/]libbasename[.so.1.0]

   If the library has the symbol PetscDLLibraryRegister_basename() in it then that function is automatically run
   when the library is opened.

   ${PETSC_ARCH} occuring in directoryname and filename 
   will be replaced with the appropriate value.

.seealso: PetscLoadDynamicLibrary(), PetscDLLibraryAppend()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscDLLibraryOpen(MPI_Comm comm,const char path[],PetscDLLibrary *entry)
{
  PetscErrorCode ierr;
  PetscTruth     foundlibrary,match;
  char           libname[PETSC_MAX_PATH_LEN],par2[PETSC_MAX_PATH_LEN],suffix[16],*s;
  char           *basename,registername[128];
  PetscDLHandle  handle;
  PetscErrorCode (*func)(const char*) = NULL;
  size_t         len;

  PetscFunctionBegin;
  PetscValidCharPointer(path,2);
  PetscValidPointer(entry,3);

  *entry = PETSC_NULL;
  
  /* retrieve the library */
  ierr = PetscInfo1(0,"Retrieving %s\n",path);CHKERRQ(ierr);
  ierr = PetscDLLibraryRetrieve(comm,path,par2,PETSC_MAX_PATH_LEN,&foundlibrary);CHKERRQ(ierr);
  if (!foundlibrary) SETERRQ1(PETSC_ERR_FILE_OPEN,"Unable to locate dynamic library:\n  %s\n",path);
  /* Eventually config/configure.py should determine if the system needs an executable dynamic library */
#define PETSC_USE_NONEXECUTABLE_SO
#if !defined(PETSC_USE_NONEXECUTABLE_SO)
  ierr  = PetscTestFile(par2,'x',&foundlibrary);CHKERRQ(ierr);
  if (!foundlibrary) SETERRQ2(PETSC_ERR_FILE_OPEN,"Dynamic library is not executable:\n  %s\n  %s\n",path,par2);
#endif

  /* copy path and setup shared library suffix  */
  ierr = PetscStrncpy(libname,path,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  ierr = PetscStrcpy(suffix,".");CHKERRQ(ierr);
  ierr = PetscStrcat(suffix,PETSC_SLSUFFIX);CHKERRQ(ierr);
  /* remove wrong suffixes from libname */
  ierr = PetscStrrstr(libname,".gz",&s);CHKERRQ(ierr);
  if (s && s[3] == 0) s[0] = 0;
  ierr = PetscStrrstr(libname,".a",&s);CHKERRQ(ierr);
  if (s && s[2] == 0) s[0] = 0;
  /* remove shared suffix from libname */
  ierr = PetscStrrstr(libname,suffix,&s);CHKERRQ(ierr);
  if (s) s[0] = 0;

  /* open the dynamic library */
  ierr = PetscInfo1(0,"Opening dynamic library %s\n",libname);CHKERRQ(ierr);
  ierr = PetscDLOpen(par2,PETSC_DL_DECIDE,&handle);CHKERRQ(ierr);

  /* look for [path/]libXXXXX.YYY and extract out the XXXXXX */
  ierr = PetscStrrchr(libname,'/',&basename);CHKERRQ(ierr); /* XXX Windows ??? */
  if (!basename) basename = libname;
  ierr = PetscStrncmp(basename,"lib",3,&match);CHKERRQ(ierr);
  if (match) { 
    basename = basename + 3; 
  } else { 
    ierr = PetscInfo1(0,"Dynamic library %s does not have lib prefix\n",libname);CHKERRQ(ierr); 
  }
  ierr = PetscStrlen(basename,&len);CHKERRQ(ierr);
  ierr = PetscStrcpy(registername,"PetscDLLibraryRegister_");CHKERRQ(ierr);
  ierr = PetscStrncat(registername,basename,len);CHKERRQ(ierr);
  ierr = PetscDLSym(handle,registername,(void**)&func);CHKERRQ(ierr);
  if (func) {
    ierr = PetscInfo1(0,"Loading registered routines from %s\n",libname);CHKERRQ(ierr);
    ierr = (*func)(libname);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo2(0,"Dynamic library %s does not have symbol %s\n",libname,registername);CHKERRQ(ierr);
  }
  
  ierr = PetscNew(struct _n_PetscDLLibrary,entry);CHKERRQ(ierr);
  (*entry)->next   = 0;
  (*entry)->handle = handle;
  ierr = PetscStrcpy((*entry)->libname,libname);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibrarySym"
/*@C
   PetscDLLibrarySym - Load a symbol from the dynamic link libraries.

   Collective on MPI_Comm

   Input Parameter:
+  comm - communicator that will open the library
.  outlist - list of already open libraries that may contain symbol (checks here before path)
.  path     - optional complete library name
-  insymbol - name of symbol

   Output Parameter:
.  value 

   Level: developer

   Notes: Symbol can be of the form
        [/path/libname[.so.1.0]:]functionname[()] where items in [] denote optional 

        Will attempt to (retrieve and) open the library if it is not yet been opened.

@*/
PetscErrorCode PETSC_DLLEXPORT PetscDLLibrarySym(MPI_Comm comm,PetscDLLibrary *outlist,const char path[],const char insymbol[],void **value)
{
  char           libname[PETSC_MAX_PATH_LEN],suffix[16],*symbol,*s;
  size_t         len;
  PetscDLLibrary nlist,prev,list;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(outlist,2);
  if (path) PetscValidCharPointer(path,3);
  PetscValidCharPointer(insymbol,4);
  PetscValidPointer(value,5);

  list   = *outlist;
  *value = 0;

  /* make copy of symbol so we can edit it in place */
  ierr = PetscStrlen(insymbol,&len);CHKERRQ(ierr);
  ierr = PetscMalloc((len+1)*sizeof(char),&symbol);CHKERRQ(ierr);
  ierr = PetscStrcpy(symbol,insymbol);CHKERRQ(ierr);
  /* If symbol contains () then replace with a NULL, to support functionname() */
  ierr = PetscStrchr(symbol,'(',&s);CHKERRQ(ierr);
  if (s) s[0] = 0;

  /*
       Function name does include library 
       -------------------------------------
  */
  if (path && path[0] != '\0') {
    /* copy path and remove suffix from libname */
    ierr = PetscStrncpy(libname,path,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
    ierr = PetscStrcpy(suffix,".");CHKERRQ(ierr);
    ierr = PetscStrcat(suffix,PETSC_SLSUFFIX);CHKERRQ(ierr);
    ierr = PetscStrrstr(libname,suffix,&s);CHKERRQ(ierr);
    if (s) s[0] = 0;
    /* Look if library is already opened and in path */
    prev  = 0;
    nlist = list;
    while (nlist) {
      PetscTruth match;
      ierr = PetscStrcmp(nlist->libname,libname,&match);CHKERRQ(ierr);
      if (match) goto done;
      prev  = nlist;
      nlist = nlist->next;
    }
    /* open the library and append it to path */
    ierr = PetscDLLibraryOpen(comm,path,&nlist);CHKERRQ(ierr);
    ierr = PetscInfo1(0,"Appending %s to dynamic library search path\n",path);CHKERRQ(ierr);
    if (prev) { prev->next = nlist; }
    else      { *outlist   = nlist; }

  done:;
    ierr = PetscDLSym(nlist->handle,symbol,value);CHKERRQ(ierr);
    if (!*value) {
      SETERRQ2(PETSC_ERR_PLIB,"Unable to locate function %s in dynamic library %s",insymbol,path);
    }
    ierr = PetscInfo2(0,"Loading function %s from dynamic library %s\n",insymbol,path);CHKERRQ(ierr);

  /*
       Function name does not include library so search path
       -----------------------------------------------------
  */
  } else {
    while (list) {
      ierr = PetscDLSym(list->handle,symbol,value);CHKERRQ(ierr);
      if (*value) {
        ierr = PetscInfo2(0,"Loading symbol %s from dynamic library %s\n",symbol,list->libname);CHKERRQ(ierr);
        break;
      }
      list = list->next;
    }
    if (!*value) {
      ierr = PetscDLSym(PETSC_NULL,symbol,value);CHKERRQ(ierr);
      if (*value) {
        ierr = PetscInfo1(0,"Loading symbol %s from object code\n",symbol);CHKERRQ(ierr);
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
-     path - name of the library

     Output Parameter:
.     outlist - list of libraries

     Level: developer

     Notes: if library is already in path will not add it.

  If the library has the symbol PetscDLLibraryRegister_basename() in it then that function is automatically run
      when the library is opened.

.seealso: PetscDLLibraryOpen()
@*/
PetscErrorCode PETSC_DLLEXPORT PetscDLLibraryAppend(MPI_Comm comm,PetscDLLibrary *outlist,const char path[])
{
  PetscDLLibrary list,prev;
  PetscErrorCode ierr;
  size_t         len;
  PetscTruth     match,dir;
  char           program[PETSC_MAX_PATH_LEN],found[8*PETSC_MAX_PATH_LEN];
  char           *libname,suffix[16],*s;
  PetscToken     token;

  PetscFunctionBegin;
  PetscValidPointer(outlist,2);
  
  /* is path a directory? */
  ierr = PetscTestDirectory(path,'r',&dir);CHKERRQ(ierr);
  if (dir) {
    ierr = PetscInfo1(0,"Checking directory %s for dynamic libraries\n",path);CHKERRQ(ierr);
    ierr  = PetscStrcpy(program,path);CHKERRQ(ierr);
    ierr  = PetscStrlen(program,&len);CHKERRQ(ierr);
    if (program[len-1] == '/') {
      ierr  = PetscStrcat(program,"*.");CHKERRQ(ierr);
    } else {
      ierr  = PetscStrcat(program,"/*.");CHKERRQ(ierr);
    }
    ierr  = PetscStrcat(program,PETSC_SLSUFFIX);CHKERRQ(ierr);

    ierr = PetscLs(comm,program,found,8*PETSC_MAX_PATH_LEN,&dir);CHKERRQ(ierr);
    if (!dir) PetscFunctionReturn(0);
  } else {
    ierr = PetscStrncpy(found,path,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  }
  ierr = PetscStrcpy(suffix,".");CHKERRQ(ierr);
  ierr = PetscStrcat(suffix,PETSC_SLSUFFIX);CHKERRQ(ierr);

  ierr = PetscTokenCreate(found,'\n',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&libname);CHKERRQ(ierr);
  while (libname) {
    /* remove suffix from libname */
    ierr = PetscStrrstr(libname,suffix,&s);CHKERRQ(ierr);
    if (s) s[0] = 0;
    /* see if library was already open then we are done */
    list  = prev = *outlist;
    match = PETSC_FALSE;
    while (list) {
      ierr = PetscStrcmp(list->libname,libname,&match);CHKERRQ(ierr);
      if (match) break;
      prev = list;
      list = list->next;
    }
    /* restore suffix from libname */
    if (s) s[0] = '.';
    if (!match) {
      /* open the library and add to end of list */
      ierr = PetscDLLibraryOpen(comm,libname,&list);CHKERRQ(ierr);
      ierr = PetscInfo1(0,"Appending %s to dynamic library search path\n",libname);CHKERRQ(ierr);
      if (!*outlist) {
	*outlist   = list;
      } else {
	prev->next = list;
      }
    }
    ierr = PetscTokenFind(token,&libname);CHKERRQ(ierr);
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
-     path - name of the library

     Output Parameter:
.     outlist - list of libraries

     Level: developer

     Notes: If library is already in path will remove old reference.

@*/
PetscErrorCode PETSC_DLLEXPORT PetscDLLibraryPrepend(MPI_Comm comm,PetscDLLibrary *outlist,const char path[])
{
  PetscDLLibrary list,prev;
  PetscErrorCode ierr;
  size_t         len;
  PetscTruth     match,dir;
  char           program[PETSC_MAX_PATH_LEN],found[8*PETSC_MAX_PATH_LEN];
  char           *libname,suffix[16],*s;
  PetscToken     token;

  PetscFunctionBegin;
  PetscValidPointer(outlist,2);
 
  /* is path a directory? */
  ierr = PetscTestDirectory(path,'r',&dir);CHKERRQ(ierr);
  if (dir) {
    ierr = PetscInfo1(0,"Checking directory %s for dynamic libraries\n",path);CHKERRQ(ierr);
    ierr  = PetscStrcpy(program,path);CHKERRQ(ierr);
    ierr  = PetscStrlen(program,&len);CHKERRQ(ierr);
    if (program[len-1] == '/') {
      ierr  = PetscStrcat(program,"*.");CHKERRQ(ierr);
    } else {
      ierr  = PetscStrcat(program,"/*.");CHKERRQ(ierr);
    }
    ierr  = PetscStrcat(program,PETSC_SLSUFFIX);CHKERRQ(ierr);

    ierr = PetscLs(comm,program,found,8*PETSC_MAX_PATH_LEN,&dir);CHKERRQ(ierr);
    if (!dir) PetscFunctionReturn(0);
  } else {
    ierr = PetscStrncpy(found,path,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
  }

  ierr = PetscStrcpy(suffix,".");CHKERRQ(ierr);
  ierr = PetscStrcat(suffix,PETSC_SLSUFFIX);CHKERRQ(ierr);

  ierr = PetscTokenCreate(found,'\n',&token);CHKERRQ(ierr);
  ierr = PetscTokenFind(token,&libname);CHKERRQ(ierr);
  while (libname) {
    /* remove suffix from libname */
    ierr = PetscStrstr(libname,suffix,&s);CHKERRQ(ierr);
    if (s) s[0] = 0;
    /* see if library was already open and move it to the front */
    prev  = 0;
    list  = *outlist;
    match = PETSC_FALSE;
    while (list) {
      ierr = PetscStrcmp(list->libname,libname,&match);CHKERRQ(ierr);
      if (match) {
	ierr = PetscInfo1(0,"Moving %s to begin of dynamic library search path\n",libname);CHKERRQ(ierr);
	if (prev) prev->next = list->next;
	if (prev) list->next = *outlist;
	*outlist = list;
	break;
      }
      prev = list;
      list = list->next;
    }
    /* restore suffix from libname */
    if (s) s[0] = '.';
    if (!match) {
      /* open the library and add to front of list */
      ierr = PetscDLLibraryOpen(comm,libname,&list);CHKERRQ(ierr);
      ierr = PetscInfo1(0,"Prepending %s to dynamic library search path\n",libname);CHKERRQ(ierr);
      list->next = *outlist;
      *outlist   = list;
    }
    ierr = PetscTokenFind(token,&libname);CHKERRQ(ierr);
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
.     head - library list

     Level: developer

@*/
PetscErrorCode PETSC_DLLEXPORT PetscDLLibraryClose(PetscDLLibrary list)
{
  PetscTruth     done = PETSC_FALSE;
  PetscDLLibrary prev,tail;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!list) PetscFunctionReturn(0);
  /* traverse the list in reverse order */
  while (!done) {
    if (!list->next) done = PETSC_TRUE;
    prev = tail = list;
    while (tail->next) {
      prev = tail;
      tail = tail->next;
    }
    prev->next = 0;
    /* close the dynamic library and free the space in entry data-structure*/
    ierr = PetscInfo1(0,"Closing dynamic library %s\n",tail->libname);CHKERRQ(ierr);
    ierr = PetscDLClose(&tail->handle);CHKERRQ(ierr);
    ierr = PetscFree(tail);CHKERRQ(ierr);
  };
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------*/

/*
   Contains the list of registered CCA components
*/
PetscFList CCAList = 0;

#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryCCAAppend"
/*@C
     PetscDLLibraryCCAAppend - Appends another CCA dynamic link library to the seach list, to the end
                of the search path.

     Collective on MPI_Comm

     Input Parameters:
+     comm - MPI communicator
-     dirname - name of directory to check

     Output Parameter:
.     outlist - list of libraries

     Level: developer

     Notes: if library is already in path will not add it.
@*/
PetscErrorCode PETSC_DLLEXPORT PetscDLLibraryCCAAppend(MPI_Comm comm,PetscDLLibrary *outlist,const char dirname[])
{
  PetscErrorCode ierr;
  size_t         l;
  PetscTruth     dir;
  char           program[PETSC_MAX_PATH_LEN],buf[8*PETSC_MAX_PATH_LEN],*libname1,fbuf[PETSC_MAX_PATH_LEN],*found,suffix[16],*f2;
  char           *func,*funcname,libname[PETSC_MAX_PATH_LEN],*lib;
  FILE           *fp;
  PetscToken     token1, token2;
  int            err;

  PetscFunctionBegin;
  /* is dirname a directory? */
  ierr = PetscTestDirectory(dirname,'r',&dir);CHKERRQ(ierr);
  if (!dir) PetscFunctionReturn(0);

  ierr = PetscInfo1(0,"Checking directory %s for CCA components\n",dirname);CHKERRQ(ierr);
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
    while ((found = fgets(fbuf,PETSC_MAX_PATH_LEN,fp))) {
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
        ierr = PetscInfo2(0,"CCA Component function and name: %s from %s\n",found,libname1);CHKERRQ(ierr);
        ierr = PetscTokenCreate(found,' ',&token2);CHKERRQ(ierr);
        ierr = PetscTokenFind(token2,&func);CHKERRQ(ierr);
        ierr = PetscTokenFind(token2,&funcname);CHKERRQ(ierr);
        ierr = PetscFListAdd(&CCAList,funcname,func,PETSC_NULL);CHKERRQ(ierr);
        ierr = PetscTokenDestroy(token2);CHKERRQ(ierr);
      }
    }
    err = fclose(fp);
    if (err) SETERRQ(PETSC_ERR_SYS,"fclose() failed on file");    
    ierr = PetscTokenFind(token1,&libname1);CHKERRQ(ierr);
  }
  ierr = PetscTokenDestroy(token1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
