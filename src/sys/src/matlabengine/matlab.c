/* $Id: matlab.c,v 1.6 2000/05/07 17:02:26 bsmith Exp bsmith $ #include "petsc.h" */

#include "engine.h"   /* Matlab include file */
#include "petsc.h" 
#include <stdarg.h>

struct  _p_PetscMatlabEngine {
  PETSCHEADER(int)
  Engine   *ep;
  char     buffer[1024];
};

#undef __FUNC__  
#define __FUNC__ /*<a name="PetscMatlabEngineCreate"></a>*/"PetscMatlabEngineCreate"
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
          MATLAB_ENGINE_(), PetscMatlabEnginePutScalar()
@*/
int PetscMatlabEngineCreate(MPI_Comm comm,char *machine,PetscMatlabEngine *engine)
{
  int               ierr,rank,size;
  char              buffer[128];
  PetscMatlabEngine e;

  PetscFunctionBegin;
  PetscHeaderCreate(e,_p_PetscMatlabEngine,int,MATLABENGINE_COOKIE,0,"MatlabEngine",comm,PetscMatlabEngineDestroy,0);
  PLogObjectCreate(e);

  if (!machine) machine = "\0";
  e->ep = engOpen(machine);
  if (!e->ep) SETERRQ1(1,1,"Unable to start Matlab engine on %s\n",machine);
  engOutputBuffer(e->ep,e->buffer,1024);

  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  sprintf(buffer,"MPI_Comm_rank = %d; MPI_Comm_size = %d;\n",rank,size);
  engEvalString(e->ep, buffer);
  
  *engine = e;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscMatlabEngineDestroy"
/*@C
   PetscMatlabEngineDestroy - Destroys a vector.

   Collective on PetscMatlabEngine

   Input Parameters:
.  e  - the engine

   Level: advanced

.seealso: PetscMatlabEnginCreate(), PetscMatlabEnginePut(), PetscMatlabEngineGet(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineGetOutput(), PetscMatlabEnginePrintOutput(),
          MATLAB_ENGINE_(), PetscMatlabEnginePutScalar()
@*/
int PetscMatlabEngineDestroy(PetscMatlabEngine v)
{
  int ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(v,MATLABENGINE_COOKIE);
  if (--v->refct > 0) PetscFunctionReturn(0);
  PLogObjectDestroy(v);
  PetscHeaderDestroy(v); 
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PetscMatlabEngineEvaluate"></a>*/"PetscMatlabEngineEvaluate"
/*@C
    PetscMatlabEngineCreate - Evaluates a string in Matlab

    Not Collective

    Input Parameters:
+   engine - the Matlab engine
-   string - format as in a printf()

   Level: advanced

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEnginePut(), PetscMatlabEngineGet(),
          PetscMatlabEngineCreate(), PetscMatlabEngineGetOutput(), PetscMatlabEnginePrintOutput(),
          MATLAB_ENGINE_(), PetscMatlabEnginePutScalar()
@*/
int PetscMatlabEngineEvaluate(PetscMatlabEngine engine,char *string,...)
{
  va_list           Argp;
  char              buffer[1024];

  PetscFunctionBegin;  
  va_start(Argp,string);
  vsprintf(buffer,string,(char *)Argp);
  va_end(Argp);

  PLogInfo(0,"Evaluating Matlab string: %s\n",buffer);
  engEvalString(engine->ep, buffer);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PetscMatlabEngineGetOutput"></a>*/"PetscMatlabEngineGetOutput"
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
          MATLAB_ENGINE_(), PetscMatlabEnginePutScalar()
@*/
int PetscMatlabEngineGetOutput(PetscMatlabEngine engine,char **string)
{
  PetscFunctionBegin;  
  *string = engine->buffer + 2;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PetscMatlabEnginePrintOutput"></a>*/"PetscMatlabEnginePrintOutput"
/*@C
    PetscMatlabEnginePrintOutput - prints the output from Matlab

    Collective on PetscMatlabEngine

    Input Parameters:
.    engine - the Matlab engine

   Level: advanced

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEnginePut(), PetscMatlabEngineGet(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineGetOutput(), PetscMatlabEngineCreate(),
          MATLAB_ENGINE_(), PetscMatlabEnginePutScalar()
@*/
int PetscMatlabEnginePrintOutput(PetscMatlabEngine engine,FILE *fd)
{
  int               ierr,rank;

  PetscFunctionBegin;  
  ierr = MPI_Comm_rank(engine->comm,&rank);CHKERRQ(ierr);
  ierr = PetscSynchronizedFPrintf(engine->comm,fd,"[%d]%s",rank,engine->buffer + 2);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(engine->comm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name="PetscMatlabEnginePut"></a>*/"PetscMatlabEnginePut"
/*@C
    PetscMatlabEnginPut - Puts a Petsc object into the Matlab space. For parallel objects,
      each processors part is put in a seperate  Matlab process.

    Collective on PetscObject

    Input Parameters:
+    engine - the Matlab engine
-    object - the PETSc object, for example Vec

   Level: advanced

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEngineCreate(), PetscMatlabEngineGet(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineGetOutput(), PetscMatlabEnginePrintOutput(),
          MATLAB_ENGINE_(), PetscMatlabEnginePutScalar()
@*/
int PetscMatlabEnginePut(PetscMatlabEngine engine,PetscObject obj)
{
  int ierr,(*put)(PetscObject,void*);
  
  PetscFunctionBegin;  
  ierr = PetscObjectQueryFunction(obj,"PetscMatlabEnginePut_C",(void**)&put);CHKERRQ(ierr);
  if (!put) {
    SETERRQ1(1,1,"Object %s cannot be put into Matlab engine",obj->class_name);
  }
  ierr = (*put)(obj,engine->ep);CHKERRQ(ierr);
  PLogInfo(0,"Putting Matlab object: %s\n",obj->name);

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PetscMatlabEngineGet"
/*@C
    PetscMatlabEngineGet - Gets a variable from Matlab into a PETSc object.

    Collective on PetscObject

    Input Parameters:
+    engine - the Matlab engine
-    object - the PETSc object, for example Vec

   Level: advanced

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEnginePut(), PetscMatlabEngineCreate(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineGetOutput(), PetscMatlabEnginePrintOutput(),
          MATLAB_ENGINE_(), PetscMatlabEnginePutScalar()
@*/
int PetscMatlabEngineGet(PetscMatlabEngine engine,PetscObject obj)
{
  int ierr,(*get)(PetscObject,void*);
  
  PetscFunctionBegin;  
  ierr = PetscObjectQueryFunction(obj,"PetscMatlabEngineGet_C",(void**)&get);CHKERRQ(ierr);
  if (!get) {
    SETERRQ1(1,1,"Object %s cannot be get into Matlab engine",obj->class_name);
  }
  ierr = (*get)(obj,engine->ep);CHKERRQ(ierr);
  PLogInfo(0,"Getting Matlab object: %s\n",obj->name);

  PetscFunctionReturn(0);
}

/*
    The variable Petsc_Matlab_Engine_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a PetscMatlabEngine
*/
static int Petsc_Matlab_Engine_keyval = MPI_KEYVAL_INVALID;

#undef __FUNC__  
#define __FUNC__ /*<a name="MATLAB_ENGINE_"></a>*/"MATLAB_ENGINE_"  
/*@C
   MATLAB_ENGINE_ - Creates a matlab engine shared by all processors 
                    in a communicator.

   Not Collective

   Input Parameter:
.  comm - the MPI communicator to share the engine

   Level: developer

   Notes: 
   Unlike almost all other PETSc routines, this does not return 
   an error code. Usually used in the form
$      PetscMatlabEngineYYY(XXX object,MATLAB_ENGINE_(comm));

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEnginePut(), PetscMatlabEngineGet(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineGetOutput(), PetscMatlabEnginePrintOutput(),
          PetscMatlabEngineCreate(), PetscMatlabEnginePutScalar()

@*/
PetscMatlabEngine MATLAB_ENGINE_(MPI_Comm comm)
{
  int               ierr;
  PetscTruth        flg;
  PetscMatlabEngine engine;

  PetscFunctionBegin;
  if (Petsc_Matlab_Engine_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Matlab_Engine_keyval,0);
    if (ierr) {PetscError(__LINE__,"MATLAB_ENGINE_",__FILE__,__SDIR__,1,1,0); engine = 0;}
  }
  ierr = MPI_Attr_get(comm,Petsc_Matlab_Engine_keyval,(void **)&engine,(int*)&flg);
  if (ierr) {PetscError(__LINE__,"MATLAB_ENGINE_",__FILE__,__SDIR__,1,1,0); engine = 0;}
  if (!flg) { /* viewer not yet created */
    ierr = PetscMatlabEngineCreate(comm,PETSC_NULL,&engine);
    if (ierr) {PetscError(__LINE__,"MATLAB_ENGINE_",__FILE__,__SDIR__,1,1,0); engine = 0;}
    ierr = PetscObjectRegisterDestroy((PetscObject)engine);
    if (ierr) {PetscError(__LINE__,"MATLAB_ENGINE_",__FILE__,__SDIR__,1,1,0); engine = 0;}
    ierr = MPI_Attr_put(comm,Petsc_Matlab_Engine_keyval,engine);
    if (ierr) {PetscError(__LINE__,"MATLAB_ENGINE_",__FILE__,__SDIR__,1,1,0); engine = 0;}
  } 
  PetscFunctionReturn(engine);
}





