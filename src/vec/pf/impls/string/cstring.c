
#include <../src/vec/pf/pfimpl.h>            /*I "petscpf.h" I*/

/*
        This PF generates a function on the fly and loads it into the running
   program.
*/

static PetscErrorCode PFView_String(void *value,PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"String = %s\n",(char*)value));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PFDestroy_String(void *value)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(value));
  PetscFunctionReturn(0);
}

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
  char           task[1024],tmp[256],lib[PETSC_MAX_PATH_LEN],username[64];
  FILE           *fd;
  PetscBool      tmpshared,wdshared,keeptmpfiles = PETSC_FALSE;
  MPI_Comm       comm;
#endif

  PetscFunctionBegin;
#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
  CHKERRQ(PetscFree(pf->data));
  CHKERRQ(PetscStrallocpy(string,(char**)&pf->data));

  /* create the new C function and compile it */
  CHKERRQ(PetscSharedTmp(PetscObjectComm((PetscObject)pf),&tmpshared));
  CHKERRQ(PetscSharedWorkingDirectory(PetscObjectComm((PetscObject)pf),&wdshared));
  if (tmpshared) {  /* do it in /tmp since everyone has one */
    CHKERRQ(PetscGetTmp(PetscObjectComm((PetscObject)pf),tmp,PETSC_MAX_PATH_LEN));
    CHKERRQ(PetscObjectGetComm((PetscObject)pf,&comm));
  } else if (!wdshared) {  /* each one does in private /tmp */
    CHKERRQ(PetscGetTmp(PetscObjectComm((PetscObject)pf),tmp,PETSC_MAX_PATH_LEN));
    comm = PETSC_COMM_SELF;
  } else { /* do it in current directory */
    CHKERRQ(PetscStrcpy(tmp,"."));
    CHKERRQ(PetscObjectGetComm((PetscObject)pf,&comm));
  }
  CHKERRQ(PetscOptionsGetBool(((PetscObject)pf)->options,((PetscObject)pf)->prefix,"-pf_string_keep_files",&keeptmpfiles,NULL));
  if (keeptmpfiles) sprintf(task,"cd %s ; mkdir ${USERNAME} ; cd ${USERNAME} ; \\cp -f ${PETSC_DIR}/src/pf/impls/string/makefile ./makefile ; ke  MIN=%d NOUT=%d petscdlib STRINGFUNCTION=\"%s\" ; sync\n",tmp,(int)pf->dimin,(int)pf->dimout,string);
  else              sprintf(task,"cd %s ; mkdir ${USERNAME} ; cd ${USERNAME} ; \\cp -f ${PETSC_DIR}/src/pf/impls/string/makefile ./makefile ; make  MIN=%d NOUT=%d -f makefile petscdlib STRINGFUNCTION=\"%s\" ; \\rm -f makefile petscdlib.c libpetscdlib.a ;  sync\n",tmp,(int)pf->dimin,(int)pf->dimout,string);

#if defined(PETSC_HAVE_POPEN)
  CHKERRQ(PetscPOpen(comm,NULL,task,"r",&fd));
  CHKERRQ(PetscPClose(comm,fd));
#else
  SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP_SYS,"Cannot run external programs on this machine");
#endif

  CHKERRMPI(MPI_Barrier(comm));

  /* load the apply function from the dynamic library */
  CHKERRQ(PetscGetUserName(username,64));
  sprintf(lib,"%s/%s/libpetscdlib",tmp,username);
  CHKERRQ(PetscDLLibrarySym(comm,NULL,lib,"PFApply_String",f));
  PetscCheck(f,PetscObjectComm((PetscObject)pf),PETSC_ERR_ARG_WRONGSTATE,"Cannot find function %s",lib);
#endif
  PetscFunctionReturn(0);
}

static PetscErrorCode PFSetFromOptions_String(PetscOptionItems *PetscOptionsObject,PF pf)
{
  PetscBool      flag;
  char           value[PETSC_MAX_PATH_LEN];
  PetscErrorCode (*f)(void*,PetscInt,const PetscScalar*,PetscScalar*) = NULL;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"String function options"));
  CHKERRQ(PetscOptionsString("-pf_string","Enter the function","PFStringCreateFunction","",value,sizeof(value),&flag));
  if (flag) {
    CHKERRQ(PFStringCreateFunction(pf,value,(void**)&f));
    pf->ops->apply = f;
  }
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

typedef PetscErrorCode (*FCN)(void*,PetscInt,const PetscScalar*,PetscScalar*); /* force argument to next function to not be extern C*/

PETSC_EXTERN PetscErrorCode PFCreate_String(PF pf,void *value)
{
  FCN            f = NULL;

  PetscFunctionBegin;
  if (value) {
    CHKERRQ(PFStringCreateFunction(pf,(char*)value,(void**)&f));
  }
  CHKERRQ(PFSet(pf,f,NULL,PFView_String,PFDestroy_String,NULL));
  pf->ops->setfromoptions = PFSetFromOptions_String;
  PetscFunctionReturn(0);
}
