
#include "src/pf/pfimpl.h"            /*I "petscpf.h" I*/

/*
        Ths PF generates a function on the fly and loads it into the running 
   program.
*/

#undef __FUNCT__  
#define __FUNCT__ "PFView_String"
int PFView_String(void *value,PetscViewer viewer)
{
  int        ierr;
  PetscTruth isascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"String = %s\n",(char*)value);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PFDestroy_String"
int PFDestroy_String(void *value)
{
  int       ierr;

  PetscFunctionBegin;
  ierr = PetscStrfree((char*)value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PFStringCreateFunction"
/*
    PFStringCreateFunction - Creates a function from a string

   Collective over PF

  Input Parameters:
+    pf - the function object
-    string - the string that defines the function

  Output Parameter:
.    f - the function pointer.

.seealso: PFSetFromOptions()

*/
int PFStringCreateFunction(PF pf,char *string,void **f)
{
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
  int        ierr;
  char       task[1024],tmp[256],lib[256],username[64];
  FILE       *fd;
  PetscTruth tmpshared,wdshared,keeptmpfiles = PETSC_FALSE;
  MPI_Comm   comm;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
  ierr = PetscStrfree((char*)pf->data);CHKERRQ(ierr);
  ierr = PetscStrallocpy(string,(char**)&pf->data);CHKERRQ(ierr);

  /* create the new C function and compile it */
  ierr = PetscSharedTmp(pf->comm,&tmpshared);CHKERRQ(ierr);
  ierr = PetscSharedWorkingDirectory(pf->comm,&wdshared);CHKERRQ(ierr);
  if (tmpshared) {  /* do it in /tmp since everyone has one */
    ierr = PetscGetTmp(pf->comm,tmp,256);CHKERRQ(ierr);
    comm = pf->comm;
  } else if (!wdshared) {  /* each one does in private /tmp */
    ierr = PetscGetTmp(pf->comm,tmp,256);CHKERRQ(ierr);
    comm = PETSC_COMM_SELF;
  } else { /* do it in current directory */
    ierr = PetscStrcpy(tmp,".");CHKERRQ(ierr);
    comm = pf->comm;
  } 
  ierr = PetscOptionsHasName(pf->prefix,"-pf_string_keep_files",&keeptmpfiles);CHKERRQ(ierr);
  if (keeptmpfiles) {
    sprintf(task,"cd %s ; mkdir ${USERNAME} ; cd ${USERNAME} ; \\cp -f ${PETSC_DIR}/src/pf/impls/string/makefile ./makefile ; make BOPT=${BOPT} MIN=%d NOUT=%d petscdlib STRINGFUNCTION=\"%s\" ; sync\n",tmp,pf->dimin,pf->dimout,string);
  } else {
    sprintf(task,"cd %s ; mkdir ${USERNAME} ;cd ${USERNAME} ; \\cp -f ${PETSC_DIR}/src/pf/impls/string/makefile ./makefile ; make BOPT=${BOPT} MIN=%d NOUT=%d -f makefile petscdlib STRINGFUNCTION=\"%s\" ; \\rm -f makefile petscdlib.c libpetscdlib.a ;  sync\n",tmp,pf->dimin,pf->dimout,string);
  }
  ierr = PetscPOpen(comm,PETSC_NULL,task,"r",&fd);CHKERRQ(ierr);
  ierr = PetscPClose(comm,fd);CHKERRQ(ierr);

  ierr = MPI_Barrier(comm);CHKERRQ(ierr);

  /* load the apply function from the dynamic library */
  ierr = PetscGetUserName(username,64);CHKERRQ(ierr);
  sprintf(lib,"%s/%s/libpetscdlib",tmp,username);
  ierr = PetscDLLibrarySym(comm,PETSC_NULL,lib,"PFApply_String",f);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);    
}

#undef __FUNCT__  
#define __FUNCT__ "PFSetFromOptions_String"
int PFSetFromOptions_String(PF pf)
{
  int        ierr;
  PetscTruth flag;
  char       value[256];
  int        (*f)(void *,int,PetscScalar*,PetscScalar*) = 0;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("String function options");CHKERRQ(ierr);
    ierr = PetscOptionsString("-pf_string","Enter the function","PFStringCreateFunction","",value,256,&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = PFStringCreateFunction(pf,value,(void**)&f);CHKERRQ(ierr);
      pf->ops->apply = f;
    }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);    
}

typedef int (*FCN)(void*,int,PetscScalar*,PetscScalar*); /* force argument to next function to not be extern C*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PFCreate_String"
int PFCreate_String(PF pf,void *value)
{
  int        ierr;
  FCN        f = 0;

  PetscFunctionBegin;
  
  if (value) {
    ierr = PFStringCreateFunction(pf,(char*)value,(void**)&f);CHKERRQ(ierr);
  }
  ierr   = PFSet(pf,f,PETSC_NULL,PFView_String,PFDestroy_String,PETSC_NULL);CHKERRQ(ierr);

  pf->ops->setfromoptions = PFSetFromOptions_String;
  PetscFunctionReturn(0);
}
EXTERN_C_END





