/*$Id: cstring.c,v 1.4 2000/04/09 04:40:41 bsmith Exp bsmith $*/
#include "src/pf/pfimpl.h"            /*I "pf.h" I*/

/*
        Ths PF generates a function on the fly and loads it into the running 
   program.
*/

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PFView_String"
int PFView_String(void *value,Viewer viewer)
{
  int        ierr;
  PetscTruth isascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  if (isascii) {
    ierr = ViewerASCIIPrintf(viewer,"String = %s\n",(char*)value);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PFDestroy_String"
int PFDestroy_String(void *value)
{
  int       ierr;

  PetscFunctionBegin;
  ierr = PetscStrfree((char*)value);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PFStringCreateFunction"
int PFStringCreateFunction(PF pf,char *string,void **f)
{
#if defined(PETSC_USE_DYNAMIC_LIBRARIES)
  int        ierr;
  char       task[1024],tmp[256],lib[256];
  FILE       *fd;
  PetscTruth tmpshared,wdshared;
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
  sprintf(task,"cd %s ; \\cp -f ${PETSC_DIR}/src/pf/impls/string/makefile ./petscmakefile ; make BOPT=${BOPT} MIN=%d NOUT=%d -f petscmakefile petscdlib STRINGFUNCTION=\"%s\" ; \\rm -f petscmakefile petscdlib.c libpetscdlib.a ;  sync\n",tmp,pf->dimin,pf->dimout,string);
  ierr = PetscPOpen(comm,PETSC_NULL,task,"r",&fd);CHKERRQ(ierr);
  ierr = PetscPClose(comm,fd);CHKERRQ(ierr);

  ierr = MPI_Barrier(comm);CHKERRQ(ierr);

  /* load the apply function from the dynamic library */
  sprintf(lib,"%s/libpetscdlib",tmp);
  ierr = DLLibrarySym(comm,PETSC_NULL,lib,"PFApply_String",f);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);    
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PFSetFromOptions_String"
int PFSetFromOptions_String(PF pf)
{
  int        ierr;
  PetscTruth flag;
  char       value[256];
  int        (*f)(void *,int,Scalar*,Scalar*) = 0;

  PetscFunctionBegin;
  ierr = OptionsGetString(pf->prefix,"-pf_string",value,256,&flag);CHKERRQ(ierr);
  if (flag) {
    ierr = PFStringCreateFunction(pf,value,(void**)&f);CHKERRQ(ierr);
    pf->ops->apply = f;
  }
  PetscFunctionReturn(0);    
}


EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PFCreate_String"
int PFCreate_String(PF pf,void *value)
{
  int        ierr;
  int        (*f)(void *,int,Scalar*,Scalar*) = 0;

  PetscFunctionBegin;
  
  if (value) {
    ierr = PFStringCreateFunction(pf,(char*)value,(void**)&f);CHKERRQ(ierr);
  }
  ierr   = PFSet(pf,f,PETSC_NULL,PFView_String,PFDestroy_String,PETSC_NULL);CHKERRQ(ierr);

  pf->ops->setfromoptions = PFSetFromOptions_String;
  PetscFunctionReturn(0);
}
EXTERN_C_END





