/* $Id: matlab.c,v 1.17 2001/08/06 21:14:26 bsmith Exp $ #include "petsc.h" */

#include "engine.h"   /* Matlab include file */
#include "petsc.h" 
#include <stdarg.h>

struct  _p_PetscMatlabEngine {
  PETSCHEADER(int)
  Engine   *ep;
  char     buffer[1024];
};

#undef __FUNCT__  
#define __FUNCT__ "PetscMatlabEngineCreate"
/*@C
    PetscMatlabEngineCreate - Creates a Matlab engine object 

    Not Collective

    Input Parameters:
+   comm - a seperate Matlab engine is started for each process in the communicator
-   machine - name of machine where Matlab engine is to be run (usually PETSC_NULL)

    Output Parameter:
.   engine - the resulting object

   Level: advanced

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEnginePut(), PetscMatlabEngineGet(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineGetOutput(), PetscMatlabEnginePrintOutput(),
          PETSC_MATLAB_ENGINE_(), PetscMatlabEnginePutScalar()
@*/
int PetscMatlabEngineCreate(MPI_Comm comm,char *machine,PetscMatlabEngine *engine)
{
  int               ierr,rank,size;
  char              buffer[128];
  PetscMatlabEngine e;

  PetscFunctionBegin;
  PetscHeaderCreate(e,_p_PetscMatlabEngine,int,MATLABENGINE_COOKIE,0,"MatlabEngine",comm,PetscMatlabEngineDestroy,0);
  PetscLogObjectCreate(e);

  if (!machine) machine = "\0";
  PetscLogInfo(0,"Starting Matlab engine on %s\n",machine);
  e->ep = engOpen(machine);
  if (!e->ep) SETERRQ1(1,"Unable to start Matlab engine on %s\n",machine);
  engOutputBuffer(e->ep,e->buffer,1024);

  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  sprintf(buffer,"MPI_Comm_rank = %d; MPI_Comm_size = %d;\n",rank,size);
  engEvalString(e->ep, buffer);
  PetscLogInfo(0,"Started Matlab engine on %s\n",machine);
  
  *engine = e;
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
          PETSC_MATLAB_ENGINE_(), PetscMatlabEnginePutScalar()
@*/
int PetscMatlabEngineDestroy(PetscMatlabEngine v)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,MATLABENGINE_COOKIE);
  if (--v->refct > 0) PetscFunctionReturn(0);
  PetscLogObjectDestroy(v);
  PetscHeaderDestroy(v); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMatlabEngineEvaluate"
/*@C
    PetscMatlabEngineEvaluate - Evaluates a string in Matlab

    Not Collective

    Input Parameters:
+   engine - the Matlab engine
-   string - format as in a printf()

   Level: advanced

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEnginePut(), PetscMatlabEngineGet(),
          PetscMatlabEngineCreate(), PetscMatlabEngineGetOutput(), PetscMatlabEnginePrintOutput(),
          PETSC_MATLAB_ENGINE_(), PetscMatlabEnginePutScalar()
@*/
int PetscMatlabEngineEvaluate(PetscMatlabEngine engine,char *string,...)
{
  va_list           Argp;
  char              buffer[1024];
  int               ierr,len,flops;

  PetscFunctionBegin;  
  ierr = PetscStrcpy(buffer,"flops(0);");
  va_start(Argp,string);
#if defined(PETSC_HAVE_VPRINTF_CHAR)
  vsprintf(buffer+9,string,(char *)Argp);
#else
  vsprintf(buffer+9,string,Argp);
#endif
  va_end(Argp);
  ierr = PetscStrcat(buffer,",flops");

  PetscLogInfo(0,"Evaluating Matlab string: %s\n",buffer);
  engEvalString(engine->ep, buffer);

  /* 
     Check for error in Matlab: indicated by ? as first character in engine->buffer
  */

  if (engine->buffer[4] == '?') {
    SETERRQ2(1,"Error in evaluating Matlab command:%s\n%s",string,engine->buffer);
  }

  /*
     Get flop number back from Matlab output
  */
  ierr = PetscStrlen(engine->buffer,&len);CHKERRQ(ierr);
  len -= 2;
  while (len >= 0) {
    len--;
    if (engine->buffer[len] == ' ') break;
    if (engine->buffer[len] == '\n') break;
    if (engine->buffer[len] == '\t') break;
  }
  sscanf(engine->buffer+len," %d\n",&flops);
  PetscLogFlops(flops);
  /* strip out of engine->buffer the end part about flops */
  if (len < 14) SETERRQ1(1,"Error from Matlab %s",engine->buffer);
  len -= 14;
  engine->buffer[len] = 0;

  PetscLogInfo(0,"Done evaluating Matlab string: %s\n",buffer);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMatlabEngineGetOutput"
/*@C
    PetscMatlabEngineGetOutput - Gets a string buffer where the Matlab output is 
          printed

    Not Collective

    Input Parameter:
.   engine - the Matlab engine

    Output Parameter:
.   string - buffer where Matlab output is printed

   Level: advanced

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEnginePut(), PetscMatlabEngineGet(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineCreate(), PetscMatlabEnginePrintOutput(),
          PETSC_MATLAB_ENGINE_(), PetscMatlabEnginePutScalar()
@*/
int PetscMatlabEngineGetOutput(PetscMatlabEngine engine,char **string)
{
  PetscFunctionBegin;  
  *string = engine->buffer;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMatlabEnginePrintOutput"
/*@C
    PetscMatlabEnginePrintOutput - prints the output from Matlab

    Collective on PetscMatlabEngine

    Input Parameters:
.    engine - the Matlab engine

   Level: advanced

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEnginePut(), PetscMatlabEngineGet(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineGetOutput(), PetscMatlabEngineCreate(),
          PETSC_MATLAB_ENGINE_(), PetscMatlabEnginePutScalar()
@*/
int PetscMatlabEnginePrintOutput(PetscMatlabEngine engine,FILE *fd)
{
  int               ierr,rank;

  PetscFunctionBegin;  
  ierr = MPI_Comm_rank(engine->comm,&rank);CHKERRQ(ierr);
  ierr = PetscSynchronizedFPrintf(engine->comm,fd,"[%d]%s",rank,engine->buffer);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(engine->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMatlabEnginePut"
/*@C
    PetscMatlabEnginePut - Puts a Petsc object into the Matlab space. For parallel objects,
      each processors part is put in a seperate  Matlab process.

    Collective on PetscObject

    Input Parameters:
+    engine - the Matlab engine
-    object - the PETSc object, for example Vec

   Level: advanced

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEngineCreate(), PetscMatlabEngineGet(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineGetOutput(), PetscMatlabEnginePrintOutput(),
          PETSC_MATLAB_ENGINE_(), , PetscMatlabEnginePutArray(), MatlabEngineGetArray()
@*/
int PetscMatlabEnginePut(PetscMatlabEngine engine,PetscObject obj)
{
  int ierr,(*put)(PetscObject,void*);
  
  PetscFunctionBegin;  
  ierr = PetscObjectQueryFunction(obj,"PetscMatlabEnginePut_C",(void (**)(void))&put);CHKERRQ(ierr);
  if (!put) {
    SETERRQ1(1,"Object %s cannot be put into Matlab engine",obj->class_name);
  }
  PetscLogInfo(0,"Putting Matlab object\n");
  ierr = (*put)(obj,engine->ep);CHKERRQ(ierr);
  PetscLogInfo(0,"Put Matlab object: %s\n",obj->name);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMatlabEngineGet"
/*@C
    PetscMatlabEngineGet - Gets a variable from Matlab into a PETSc object.

    Collective on PetscObject

    Input Parameters:
+    engine - the Matlab engine
-    object - the PETSc object, for example Vec

   Level: advanced

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEnginePut(), PetscMatlabEngineCreate(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineGetOutput(), PetscMatlabEnginePrintOutput(),
          PETSC_MATLAB_ENGINE_(), PetscMatlabEnginePutArray(), MatlabEngineGetArray()
@*/
int PetscMatlabEngineGet(PetscMatlabEngine engine,PetscObject obj)
{
  int ierr,(*get)(PetscObject,void*);
  
  PetscFunctionBegin;  
  if (!obj->name) {
    SETERRQ(1,"Cannot get object that has no name");
  }
  ierr = PetscObjectQueryFunction(obj,"PetscMatlabEngineGet_C",(void (**)(void))&get);CHKERRQ(ierr);
  if (!get) {
    SETERRQ1(1,"Object %s cannot be get into Matlab engine",obj->class_name);
  }
  PetscLogInfo(0,"Getting Matlab object\n");
  ierr = (*get)(obj,engine->ep);CHKERRQ(ierr);
  PetscLogInfo(0,"Got Matlab object: %s\n",obj->name);
  PetscFunctionReturn(0);
}

/*
    The variable Petsc_Matlab_Engine_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a PetscMatlabEngine
*/
static int Petsc_Matlab_Engine_keyval = MPI_KEYVAL_INVALID;

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
          PetscMatlabEngineCreate(), PetscMatlabEnginePutScalar()

@*/
PetscMatlabEngine PETSC_MATLAB_ENGINE_(MPI_Comm comm)
{
  int               ierr;
  PetscTruth        flg;
  PetscMatlabEngine engine;

  PetscFunctionBegin;
  if (Petsc_Matlab_Engine_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Matlab_Engine_keyval,0);
    if (ierr) {PetscError(__LINE__,"PETSC_MATLAB_ENGINE_",__FILE__,__SDIR__,1,1,0); engine = 0;}
  }
  ierr = MPI_Attr_get(comm,Petsc_Matlab_Engine_keyval,(void **)&engine,(int*)&flg);
  if (ierr) {PetscError(__LINE__,"PETSC_MATLAB_ENGINE_",__FILE__,__SDIR__,1,1,0); engine = 0;}
  if (!flg) { /* viewer not yet created */
    char *machinename = 0,machine[64];

    ierr = PetscOptionsGetString(PETSC_NULL,"-matlab_engine_machine",machine,64,&flg);
    if (ierr) {PetscError(__LINE__,"PETSC_MATLAB_ENGINE_",__FILE__,__SDIR__,1,1,0); engine = 0;}
    if (flg) machinename = machine;
    ierr = PetscMatlabEngineCreate(comm,machinename,&engine);
    if (ierr) {PetscError(__LINE__,"PETSC_MATLAB_ENGINE_",__FILE__,__SDIR__,1,1,0); engine = 0;}
    ierr = PetscObjectRegisterDestroy((PetscObject)engine);
    if (ierr) {PetscError(__LINE__,"PETSC_MATLAB_ENGINE_",__FILE__,__SDIR__,1,1,0); engine = 0;}
    ierr = MPI_Attr_put(comm,Petsc_Matlab_Engine_keyval,engine);
    if (ierr) {PetscError(__LINE__,"PETSC_MATLAB_ENGINE_",__FILE__,__SDIR__,1,1,0); engine = 0;}
  } 
  PetscFunctionReturn(engine);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMatlabEnginePutArray"
/*@C
    PetscMatlabEnginePutArray - Puts a Petsc object into the Matlab space. For parallel objects,
      each processors part is put in a seperate  Matlab process.

    Collective on PetscObject

    Input Parameters:
+    engine - the Matlab engine
.    m,n - the dimensions of the array
.    array - the array (represented in one dimension)
-    name - the name of the array

   Level: advanced

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEngineCreate(), PetscMatlabEngineGet(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineGetOutput(), PetscMatlabEnginePrintOutput(),
          PETSC_MATLAB_ENGINE_(), PetscMatlabEnginePut(), MatlabEngineGetArray()
@*/
int PetscMatlabEnginePutArray(PetscMatlabEngine engine,int m,int n,PetscScalar *array,char *name)
{
  int     ierr;
  mxArray *mat;
  
  PetscFunctionBegin;  
  PetscLogInfo(0,"Putting Matlab array %s\n",name);
#if !defined(PETSC_USE_COMPLEX)
  mat  = mxCreateDoubleMatrix(m,n,mxREAL);
#else
  mat  = mxCreateDoubleMatrix(m,n,mxCOMPLEX);
#endif
  ierr = PetscMemcpy(mxGetPr(mat),array,m*n*sizeof(PetscScalar));CHKERRQ(ierr);
  mxSetName(mat,name);
  engPutArray(engine->ep,mat);

  PetscLogInfo(0,"Put Matlab array %s\n",name);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscMatlabEngineGetArray"
/*@C
    PetscMatlabEngineGetArray - Gets a variable from Matlab into an array

    Not Collective

    Input Parameters:
+    engine - the Matlab engine
.    m,n - the dimensions of the array
.    array - the array (represented in one dimension)
-    name - the name of the array

   Level: advanced

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEnginePut(), PetscMatlabEngineCreate(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineGetOutput(), PetscMatlabEnginePrintOutput(),
          PETSC_MATLAB_ENGINE_(), PetscMatlabEnginePutArray(), PetscMatlabEngineGet()
@*/
int PetscMatlabEngineGetArray(PetscMatlabEngine engine,int m,int n,PetscScalar *array,char *name)
{
  int     ierr;
  mxArray *mat;
  
  PetscFunctionBegin;  
  PetscLogInfo(0,"Getting Matlab array %s\n",name);
  mat  = engGetArray(engine->ep,name);
  if (!mat) SETERRQ1(1,"Unable to get array %s from matlab",name);
  ierr = PetscMemcpy(array,mxGetPr(mat),m*n*sizeof(PetscScalar));CHKERRQ(ierr);
  PetscLogInfo(0,"Got Matlab array %s\n",name);
  PetscFunctionReturn(0);
}




