
#include <../src/vec/pf/pfimpl.h>            /*I "petscpf.h" I*/

/*
        Ths PF generates a function on the fly and loads it into the running
   program.
*/

#undef __FUNCT__
#define __FUNCT__ "PFView_String"
PetscErrorCode PFView_String(void *value,PetscViewer viewer)
{
  PetscErrorCode ierr;
  PetscBool  iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"String = %s\n",(char*)value);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PFDestroy_String"
PetscErrorCode PFDestroy_String(void *value)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(value);CHKERRQ(ierr);
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
PetscErrorCode  PFStringCreateFunction(PF pf,char *string,void **f)
{
#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
  PetscErrorCode ierr;
  char       task[1024],tmp[256],lib[PETSC_MAX_PATH_LEN],username[64];
  FILE       *fd;
  PetscBool  tmpshared,wdshared,keeptmpfiles = PETSC_FALSE;
  MPI_Comm   comm;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
  ierr = PetscFree(pf->data);CHKERRQ(ierr);
  ierr = PetscStrallocpy(string,(char**)&pf->data);CHKERRQ(ierr);

  /* create the new C function and compile it */
  ierr = PetscSharedTmp(((PetscObject)pf)->comm,&tmpshared);CHKERRQ(ierr);
  ierr = PetscSharedWorkingDirectory(((PetscObject)pf)->comm,&wdshared);CHKERRQ(ierr);
  if (tmpshared) {  /* do it in /tmp since everyone has one */
    ierr = PetscGetTmp(((PetscObject)pf)->comm,tmp,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
    comm = ((PetscObject)pf)->comm;
  } else if (!wdshared) {  /* each one does in private /tmp */
    ierr = PetscGetTmp(((PetscObject)pf)->comm,tmp,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);
    comm = PETSC_COMM_SELF;
  } else { /* do it in current directory */
    ierr = PetscStrcpy(tmp,".");CHKERRQ(ierr);
    comm = ((PetscObject)pf)->comm;
  }
  ierr = PetscOptionsGetBool(((PetscObject)pf)->prefix,"-pf_string_keep_files",&keeptmpfiles,PETSC_NULL);CHKERRQ(ierr);
  if (keeptmpfiles) {
    sprintf(task,"cd %s ; mkdir ${USERNAME} ; cd ${USERNAME} ; \\cp -f ${PETSC_DIR}/src/pf/impls/string/makefile ./makefile ; ke  MIN=%d NOUT=%d petscdlib STRINGFUNCTION=\"%s\" ; sync\n",tmp,(int)pf->dimin,(int)pf->dimout,string);
  } else {
    sprintf(task,"cd %s ; mkdir ${USERNAME} ;cd ${USERNAME} ; \\cp -f ${PETSC_DIR}/src/pf/impls/string/makefile ./makefile ; make  MIN=%d NOUT=%d -f makefile petscdlib STRINGFUNCTION=\"%s\" ; \\rm -f makefile petscdlib.c libpetscdlib.a ;  sync\n",tmp,(int)pf->dimin,(int)pf->dimout,string);
  }
#if defined(PETSC_HAVE_POPEN)
  ierr = PetscPOpen(comm,PETSC_NULL,task,"r",&fd);CHKERRQ(ierr);
  ierr = PetscPClose(comm,fd,PETSC_NULL);CHKERRQ(ierr);
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Cannot run external programs on this machine");
#endif

  ierr = MPI_Barrier(comm);CHKERRQ(ierr);

  /* load the apply function from the dynamic library */
  ierr = PetscGetUserName(username,64);CHKERRQ(ierr);
  sprintf(lib,"%s/%s/libpetscdlib",tmp,username);
  ierr = PetscDLLibrarySym(comm,PETSC_NULL,lib,"PFApply_String",f);CHKERRQ(ierr);
  if (!f) SETERRQ1(((PetscObject)pf)->comm,PETSC_ERR_ARG_WRONGSTATE,"Cannot find function %s",lib);
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PFSetFromOptions_String"
PetscErrorCode PFSetFromOptions_String(PF pf)
{
  PetscErrorCode ierr;
  PetscBool  flag;
  char       value[PETSC_MAX_PATH_LEN];
  PetscErrorCode (*f)(void*,PetscInt,const PetscScalar*,PetscScalar*) = 0;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("String function options");CHKERRQ(ierr);
    ierr = PetscOptionsString("-pf_string","Enter the function","PFStringCreateFunction","",value,PETSC_MAX_PATH_LEN,&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = PFStringCreateFunction(pf,value,(void**)&f);CHKERRQ(ierr);
      pf->ops->apply = f;
    }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

typedef PetscErrorCode (*FCN)(void*,PetscInt,const PetscScalar*,PetscScalar*); /* force argument to next function to not be extern C*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "PFCreate_String"
PetscErrorCode  PFCreate_String(PF pf,void *value)
{
  PetscErrorCode ierr;
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





