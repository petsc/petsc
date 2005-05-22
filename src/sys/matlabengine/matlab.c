#define PETSC_DLL

#include "engine.h"   /* Matlab include file */
#include "petsc.h" 
#include <stdarg.h>

struct  _p_PetscMatlabEngine {
  PETSCHEADER(int);
  Engine   *ep;
  char     buffer[1024];
};

PetscCookie MATLABENGINE_COOKIE = -1;

#undef __FUNCT__  
#define __FUNCT__ "PetscMatlabEngineCreate"
/*@C
    PetscMatlabEngineCreate - Creates a Matlab engine object 

    Not Collective

    Input Parameters:
+   comm - a seperate Matlab engine is started for each process in the communicator
-   machine - name of machine where Matlab engine is to be run (usually PETSC_NULL)

    Output Parameter:
.   mengine - the resulting object

   Level: advanced

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEnginePut(), PetscMatlabEngineGet(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineGetOutput(), PetscMatlabEnginePrintOutput(),
          PETSC_MATLAB_ENGINE_(), PetscMatlabEnginePutArray(), PetscMatlabEngineGetArray(), PetscMatlabEngine
@*/
PetscErrorCode PETSC_DLLEXPORT PetscMatlabEngineCreate(MPI_Comm comm,const char machine[],PetscMatlabEngine *mengine)
{
  PetscErrorCode    ierr;
  PetscMPIInt       rank,size;
  char              buffer[256];
  PetscMatlabEngine e;

  PetscFunctionBegin;
  if (MATLABENGINE_COOKIE == -1) {
    ierr = PetscLogClassRegister(&MATLABENGINE_COOKIE,"Matlab Engine");CHKERRQ(ierr);
  }
  ierr = PetscHeaderCreate(e,_p_PetscMatlabEngine,int,MATLABENGINE_COOKIE,0,"MatlabEngine",comm,PetscMatlabEngineDestroy,0);CHKERRQ(ierr);

  if (!machine) machine = "\0";
  ierr = PetscLogInfo((0,"PetscMatlabEngineCreate:Starting Matlab engine on %s\n",machine));CHKERRQ(ierr);
  e->ep = engOpen(machine);
  if (!e->ep) SETERRQ1(PETSC_ERR_LIB,"Unable to start Matlab engine on %s\n",machine);
  engOutputBuffer(e->ep,e->buffer,1024);

  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  sprintf(buffer,"MPI_Comm_rank = %d; MPI_Comm_size = %d;\n",rank,size);
  engEvalString(e->ep, buffer);
  ierr = PetscLogInfo((0,"PetscMatlabEngineCreate:Started Matlab engine on %s\n",machine));CHKERRQ(ierr);
  
  *mengine = e;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMatlabEngineDestroy"
/*@C
   PetscMatlabEngineDestroy - Destroys a vector.

   Collective on PetscMatlabEngine

   Input Parameters:
.  e  - the engine

   Level: advanced

.seealso: PetscMatlabEnginCreate(), PetscMatlabEnginePut(), PetscMatlabEngineGet(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineGetOutput(), PetscMatlabEnginePrintOutput(),
          PETSC_MATLAB_ENGINE_(), PetscMatlabEnginePutArray(), PetscMatlabEngineGetArray(), PetscMatlabEngine
@*/
PetscErrorCode PETSC_DLLEXPORT PetscMatlabEngineDestroy(PetscMatlabEngine v)
{
  PetscErrorCode ierr;
  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,MATLABENGINE_COOKIE,1);
  if (--v->refct > 0) PetscFunctionReturn(0);
  ierr = PetscHeaderDestroy(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMatlabEngineEvaluate"
/*@C
    PetscMatlabEngineEvaluate - Evaluates a string in Matlab

    Not Collective

    Input Parameters:
+   mengine - the Matlab engine
-   string - format as in a printf()

   Level: advanced

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEnginePut(), PetscMatlabEngineGet(),
          PetscMatlabEngineCreate(), PetscMatlabEngineGetOutput(), PetscMatlabEnginePrintOutput(),
          PETSC_MATLAB_ENGINE_(), PetscMatlabEnginePutArray(), PetscMatlabEngineGetArray(), PetscMatlabEngine
@*/
PetscErrorCode PETSC_DLLEXPORT PetscMatlabEngineEvaluate(PetscMatlabEngine mengine,const char string[],...)
{
  va_list           Argp;
  char              buffer[1024];
  PetscErrorCode ierr;
  int               flops;
  size_t            len;

  PetscFunctionBegin;  
  ierr = PetscStrcpy(buffer,"flops(0);");
  va_start(Argp,string);
  ierr = PetscVSNPrintf(buffer+9,1024-9-5,string,Argp);CHKERRQ(ierr);
  va_end(Argp);
  ierr = PetscStrcat(buffer,",flops");

  ierr = PetscLogInfo((0,"PetscMatlabEngineEvaluate:Evaluating Matlab string: %s\n",buffer));CHKERRQ(ierr);
  engEvalString(mengine->ep, buffer);

  /* 
     Check for error in Matlab: indicated by ? as first character in engine->buffer
  */

  if (mengine->buffer[4] == '?') {
    SETERRQ2(PETSC_ERR_LIB,"Error in evaluating Matlab command:%s\n%s",string,mengine->buffer);
  }

  /*
     Get flop number back from Matlab output
  */
  ierr = PetscStrlen(mengine->buffer,&len);CHKERRQ(ierr);
  len -= 2;
  while (len > 0) {
    len--;
    if (mengine->buffer[len] == ' ') break;
    if (mengine->buffer[len] == '\n') break;
    if (mengine->buffer[len] == '\t') break;
  }
  sscanf(mengine->buffer+len," %d\n",&flops);
  ierr = PetscLogFlops(flops);CHKERRQ(ierr);
  /* strip out of engine->buffer the end part about flops */
  if (len < 14) SETERRQ1(PETSC_ERR_LIB,"Error from Matlab %s",mengine->buffer);
  len -= 14;
  mengine->buffer[len] = 0;

  ierr = PetscLogInfo((0,"PetscMatlabEngineEvaluate:Done evaluating Matlab string: %s\n",buffer));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMatlabEngineGetOutput"
/*@C
    PetscMatlabEngineGetOutput - Gets a string buffer where the Matlab output is 
          printed

    Not Collective

    Input Parameter:
.   mengine - the Matlab engine

    Output Parameter:
.   string - buffer where Matlab output is printed

   Level: advanced

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEnginePut(), PetscMatlabEngineGet(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineCreate(), PetscMatlabEnginePrintOutput(),
          PETSC_MATLAB_ENGINE_(), PetscMatlabEnginePutArray(), PetscMatlabEngineGetArray(), PetscMatlabEngine
@*/
PetscErrorCode PETSC_DLLEXPORT PetscMatlabEngineGetOutput(PetscMatlabEngine mengine,char **string)
{
  PetscFunctionBegin;  
  *string = mengine->buffer;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMatlabEnginePrintOutput"
/*@C
    PetscMatlabEnginePrintOutput - prints the output from Matlab

    Collective on PetscMatlabEngine

    Input Parameters:
.    mengine - the Matlab engine

   Level: advanced

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEnginePut(), PetscMatlabEngineGet(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineGetOutput(), PetscMatlabEngineCreate(),
          PETSC_MATLAB_ENGINE_(), PetscMatlabEnginePutArray(), PetscMatlabEngineGetArray(), PetscMatlabEngine
@*/
PetscErrorCode PETSC_DLLEXPORT PetscMatlabEnginePrintOutput(PetscMatlabEngine mengine,FILE *fd)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;  
  ierr = MPI_Comm_rank(mengine->comm,&rank);CHKERRQ(ierr);
  ierr = PetscSynchronizedFPrintf(mengine->comm,fd,"[%d]%s",rank,mengine->buffer);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(mengine->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMatlabEnginePut"
/*@C
    PetscMatlabEnginePut - Puts a Petsc object into the Matlab space. For parallel objects,
      each processors part is put in a seperate  Matlab process.

    Collective on PetscObject

    Input Parameters:
+    mengine - the Matlab engine
-    object - the PETSc object, for example Vec

   Level: advanced

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEngineCreate(), PetscMatlabEngineGet(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineGetOutput(), PetscMatlabEnginePrintOutput(),
          PETSC_MATLAB_ENGINE_(), PetscMatlabEnginePutArray(), MatlabEngineGetArray(), PetscMatlabEngine
@*/
PetscErrorCode PETSC_DLLEXPORT PetscMatlabEnginePut(PetscMatlabEngine mengine,PetscObject obj)
{
  PetscErrorCode ierr,(*put)(PetscObject,void*);
  
  PetscFunctionBegin;  
  ierr = PetscObjectQueryFunction(obj,"PetscMatlabEnginePut_C",(void (**)(void))&put);CHKERRQ(ierr);
  if (!put) {
    SETERRQ1(PETSC_ERR_SUP,"Object %s cannot be put into Matlab engine",obj->class_name);
  }
  ierr = PetscLogInfo((0,"PetscMatlabEnginePut:Putting Matlab object\n"));CHKERRQ(ierr);
  ierr = (*put)(obj,mengine->ep);CHKERRQ(ierr);
  ierr = PetscLogInfo((0,"PetscMatlabEnginePut:Put Matlab object: %s\n",obj->name));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMatlabEngineGet"
/*@C
    PetscMatlabEngineGet - Gets a variable from Matlab into a PETSc object.

    Collective on PetscObject

    Input Parameters:
+    mengine - the Matlab engine
-    object - the PETSc object, for example Vec

   Level: advanced

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEnginePut(), PetscMatlabEngineCreate(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineGetOutput(), PetscMatlabEnginePrintOutput(),
          PETSC_MATLAB_ENGINE_(), PetscMatlabEnginePutArray(), MatlabEngineGetArray(), PetscMatlabEngine
@*/
PetscErrorCode PETSC_DLLEXPORT PetscMatlabEngineGet(PetscMatlabEngine mengine,PetscObject obj)
{
  PetscErrorCode ierr,(*get)(PetscObject,void*);
  
  PetscFunctionBegin;  
  if (!obj->name) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Cannot get object that has no name");
  }
  ierr = PetscObjectQueryFunction(obj,"PetscMatlabEngineGet_C",(void (**)(void))&get);CHKERRQ(ierr);
  if (!get) {
    SETERRQ1(PETSC_ERR_SUP,"Object %s cannot be gotten from Matlab engine",obj->class_name);
  }
  ierr = PetscLogInfo((0,"PetscMatlabEngineGet:Getting Matlab object\n"));CHKERRQ(ierr);
  ierr = (*get)(obj,mengine->ep);CHKERRQ(ierr);
  ierr = PetscLogInfo((0,"PetscMatlabEngineGet:Got Matlab object: %s\n",obj->name));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    The variable Petsc_Matlab_Engine_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a PetscMatlabEngine
*/
static PetscMPIInt Petsc_Matlab_Engine_keyval = MPI_KEYVAL_INVALID;


#undef __FUNCT__  
#define __FUNCT__ "PETSC_MATLAB_ENGINE_"  
/*@C
   PETSC_MATLAB_ENGINE_ - Creates a matlab engine shared by all processors 
                    in a communicator.

   Not Collective

   Input Parameter:
.  comm - the MPI communicator to share the engine

   Level: developer

   Notes: 
   Unlike almost all other PETSc routines, this does not return 
   an error code. Usually used in the form
$      PetscMatlabEngineYYY(XXX object,PETSC_MATLAB_ENGINE_(comm));

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEnginePut(), PetscMatlabEngineGet(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineGetOutput(), PetscMatlabEnginePrintOutput(),
          PetscMatlabEngineCreate(), PetscMatlabEnginePutArray(), PetscMatlabEngineGetArray(), PetscMatlabEngine,
          PETSC_MATLAB_ENGINE_WORLD, PETSC_MATLAB_ENGINE_SELF
 
@*/
PetscMatlabEngine PETSC_DLLEXPORT PETSC_MATLAB_ENGINE_(MPI_Comm comm)
{
  PetscErrorCode ierr;
  PetscTruth        flg;
  PetscMatlabEngine mengine;

  PetscFunctionBegin;
  if (Petsc_Matlab_Engine_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Matlab_Engine_keyval,0);
    if (ierr) {PetscError(__LINE__,"PETSC_MATLAB_ENGINE_",__FILE__,__SDIR__,1,1," "); mengine = 0;}
  }
  ierr = MPI_Attr_get(comm,Petsc_Matlab_Engine_keyval,(void **)&mengine,(int*)&flg);
  if (ierr) {PetscError(__LINE__,"PETSC_MATLAB_ENGINE_",__FILE__,__SDIR__,1,1," "); mengine = 0;}
  if (!flg) { /* viewer not yet created */
    char *machinename = 0,machine[64];

    ierr = PetscOptionsGetString(PETSC_NULL,"-matlab_engine_machine",machine,64,&flg);
    if (ierr) {PetscError(__LINE__,"PETSC_MATLAB_ENGINE_",__FILE__,__SDIR__,1,1," "); mengine = 0;}
    if (flg) machinename = machine;
    ierr = PetscMatlabEngineCreate(comm,machinename,&mengine);
    if (ierr) {PetscError(__LINE__,"PETSC_MATLAB_ENGINE_",__FILE__,__SDIR__,1,1," "); mengine = 0;}
    ierr = PetscObjectRegisterDestroy((PetscObject)mengine);
    if (ierr) {PetscError(__LINE__,"PETSC_MATLAB_ENGINE_",__FILE__,__SDIR__,1,1," "); mengine = 0;}
    ierr = MPI_Attr_put(comm,Petsc_Matlab_Engine_keyval,mengine);
    if (ierr) {PetscError(__LINE__,"PETSC_MATLAB_ENGINE_",__FILE__,__SDIR__,1,1," "); mengine = 0;}
  } 
  PetscFunctionReturn(mengine);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMatlabEnginePutArray"
/*@C
    PetscMatlabEnginePutArray - Puts a Petsc object into the Matlab space. For parallel objects,
      each processors part is put in a seperate  Matlab process.

    Collective on PetscObject

    Input Parameters:
+    mengine - the Matlab engine
.    m,n - the dimensions of the array
.    array - the array (represented in one dimension)
-    name - the name of the array

   Level: advanced

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEngineCreate(), PetscMatlabEngineGet(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineGetOutput(), PetscMatlabEnginePrintOutput(),
          PETSC_MATLAB_ENGINE_(), PetscMatlabEnginePut(), MatlabEngineGetArray(), PetscMatlabEngine
@*/
PetscErrorCode PETSC_DLLEXPORT PetscMatlabEnginePutArray(PetscMatlabEngine mengine,int m,int n,PetscScalar *array,const char name[])
{
  PetscErrorCode ierr;
  mxArray *mat;
  
  PetscFunctionBegin;  
  ierr = PetscLogInfo((0,"PetscMatlabEnginePutArray:Putting Matlab array %s\n",name));CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  mat  = mxCreateDoubleMatrix(m,n,mxREAL);
#else
  mat  = mxCreateDoubleMatrix(m,n,mxCOMPLEX);
#endif
  ierr = PetscMemcpy(mxGetPr(mat),array,m*n*sizeof(PetscScalar));CHKERRQ(ierr);
  engPutVariable(mengine->ep,name,mat);

  ierr = PetscLogInfo((0,"PetscMatlabEnginePutArray:Put Matlab array %s\n",name));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMatlabEngineGetArray"
/*@C
    PetscMatlabEngineGetArray - Gets a variable from Matlab into an array

    Not Collective

    Input Parameters:
+    mengine - the Matlab engine
.    m,n - the dimensions of the array
.    array - the array (represented in one dimension)
-    name - the name of the array

   Level: advanced

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEnginePut(), PetscMatlabEngineCreate(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineGetOutput(), PetscMatlabEnginePrintOutput(),
          PETSC_MATLAB_ENGINE_(), PetscMatlabEnginePutArray(), PetscMatlabEngineGet(), PetscMatlabEngine
@*/
PetscErrorCode PETSC_DLLEXPORT PetscMatlabEngineGetArray(PetscMatlabEngine mengine,int m,int n,PetscScalar *array,const char name[])
{
  PetscErrorCode ierr;
  mxArray *mat;
  
  PetscFunctionBegin;  
  ierr = PetscLogInfo((0,"PetscMatlabEngineGetArray:Getting Matlab array %s\n",name));CHKERRQ(ierr);
  mat  = engGetVariable(mengine->ep,name);
  if (!mat) SETERRQ1(PETSC_ERR_LIB,"Unable to get array %s from matlab",name);
  ierr = PetscMemcpy(array,mxGetPr(mat),m*n*sizeof(PetscScalar));CHKERRQ(ierr);
  ierr = PetscLogInfo((0,"PetscMatlabEngineGetArray:Got Matlab array %s\n",name));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}




