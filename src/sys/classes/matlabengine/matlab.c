
#include <engine.h>   /* MATLAB include file */
#include <petscsys.h>
#include <petscmatlab.h>               /*I   "petscmatlab.h"  I*/
#include <petsc/private/petscimpl.h>

struct  _p_PetscMatlabEngine {
  PETSCHEADER(int);
  Engine *ep;
  char   buffer[1024];
};

PetscClassId MATLABENGINE_CLASSID = -1;

/*@C
    PetscMatlabEngineCreate - Creates a MATLAB engine object

    Not Collective

    Input Parameters:
+   comm - a separate MATLAB engine is started for each process in the communicator
-   host - name of machine where MATLAB engine is to be run (usually NULL)

    Output Parameter:
.   mengine - the resulting object

   Options Database:
+    -matlab_engine_graphics - allow the MATLAB engine to display graphics
.    -matlab_engine_host - hostname, machine to run the MATLAB engine on
-    -info - print out all requests to MATLAB and all if its responses (for debugging)

   Level: advanced

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEnginePut(), PetscMatlabEngineGet(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineGetOutput(), PetscMatlabEnginePrintOutput(),
          PETSC_MATLAB_ENGINE_(), PetscMatlabEnginePutArray(), PetscMatlabEngineGetArray(), PetscMatlabEngine
@*/
PetscErrorCode  PetscMatlabEngineCreate(MPI_Comm comm,const char host[],PetscMatlabEngine *mengine)
{
  PetscErrorCode    ierr;
  PetscMPIInt       rank,size;
  char              buffer[256];
  PetscMatlabEngine e;
  PetscBool         flg = PETSC_FALSE;

  PetscFunctionBegin;
  if (MATLABENGINE_CLASSID == -1) {
    ierr = PetscClassIdRegister("MATLAB Engine",&MATLABENGINE_CLASSID);CHKERRQ(ierr);
  }
  ierr = PetscHeaderCreate(e,MATLABENGINE_CLASSID,"MatlabEngine","MATLAB Engine","Sys",comm,PetscMatlabEngineDestroy,NULL);CHKERRQ(ierr);

  if (!host) {
    char lhost[64];

    ierr = PetscOptionsGetString(NULL,NULL,"-matlab_engine_host",lhost,sizeof(lhost),&flg);CHKERRQ(ierr);
    if (flg) {host = lhost;}
  }
  flg = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-matlab_engine_graphics",&flg,NULL);CHKERRQ(ierr);

  if (host) {
    ierr  = PetscInfo1(0,"Starting MATLAB engine on %s\n",host);CHKERRQ(ierr);
  } else {

  }
  if (host) {
    ierr = PetscStrcpy(buffer,"ssh ");CHKERRQ(ierr);
    ierr = PetscStrcat(buffer,host);CHKERRQ(ierr);
    ierr = PetscStrcat(buffer," \"");CHKERRQ(ierr);
    ierr = PetscStrlcat(buffer,PETSC_MATLAB_COMMAND,sizeof(buffer));CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscStrlcat(buffer," -nodisplay ",sizeof(buffer));CHKERRQ(ierr);
    }
    ierr  = PetscStrlcat(buffer," -nosplash ",sizeof(buffer));CHKERRQ(ierr);
    ierr = PetscStrcat(buffer,"\"");CHKERRQ(ierr);
  } else {
    ierr = PetscStrncpy(buffer,PETSC_MATLAB_COMMAND,sizeof(buffer));CHKERRQ(ierr);
    if (!flg) {
      ierr = PetscStrlcat(buffer," -nodisplay ",sizeof(buffer));CHKERRQ(ierr);
    }
    ierr  = PetscStrlcat(buffer," -nosplash ",sizeof(buffer));CHKERRQ(ierr);
  }
  ierr  = PetscInfo1(0,"Starting MATLAB engine with command %s\n",buffer);CHKERRQ(ierr);
  e->ep = engOpen(buffer);
  if (!e->ep) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Unable to start MATLAB engine with %s",buffer);
  engOutputBuffer(e->ep,e->buffer,sizeof(e->buffer));
  if (host) {
    ierr = PetscInfo1(0,"Started MATLAB engine on %s\n",host);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(0,"Started MATLAB engine\n");CHKERRQ(ierr);
  }

  ierr = MPI_Comm_rank(comm,&rank);CHKERRMPI(ierr);
  ierr = MPI_Comm_size(comm,&size);CHKERRMPI(ierr);
  ierr = PetscMatlabEngineEvaluate(e,"MPI_Comm_rank = %d; MPI_Comm_size = %d;\n",rank,size);CHKERRQ(ierr);
  *mengine = e;
  PetscFunctionReturn(0);
}

/*@
   PetscMatlabEngineDestroy - Shuts down a MATLAB engine.

   Collective on PetscMatlabEngine

   Input Parameters:
.  e  - the engine

   Level: advanced

.seealso: PetscMatlabEngineCreate(), PetscMatlabEnginePut(), PetscMatlabEngineGet(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineGetOutput(), PetscMatlabEnginePrintOutput(),
          PETSC_MATLAB_ENGINE_(), PetscMatlabEnginePutArray(), PetscMatlabEngineGetArray(), PetscMatlabEngine
@*/
PetscErrorCode  PetscMatlabEngineDestroy(PetscMatlabEngine *v)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*v) PetscFunctionReturn(0);
  PetscValidHeaderSpecific(*v,MATLABENGINE_CLASSID,1);
  if (--((PetscObject)(*v))->refct > 0) PetscFunctionReturn(0);
  ierr = PetscInfo(0,"Stopping MATLAB engine\n");CHKERRQ(ierr);
  ierr = engClose((*v)->ep);
  if (ierr) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error closing Matlab engine");
  ierr = PetscInfo(0,"MATLAB engine stopped\n");CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(v);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
    PetscMatlabEngineEvaluate - Evaluates a string in MATLAB

    Not Collective

    Input Parameters:
+   mengine - the MATLAB engine
-   string - format as in a printf()

   Notes:
     Run the PETSc program with -info to always have printed back MATLAB's response to the string evaluation

   Level: advanced

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEnginePut(), PetscMatlabEngineGet(),
          PetscMatlabEngineCreate(), PetscMatlabEngineGetOutput(), PetscMatlabEnginePrintOutput(),
          PETSC_MATLAB_ENGINE_(), PetscMatlabEnginePutArray(), PetscMatlabEngineGetArray(), PetscMatlabEngine
@*/
PetscErrorCode  PetscMatlabEngineEvaluate(PetscMatlabEngine mengine,const char string[],...)
{
  va_list        Argp;
  char           buffer[1024];
  PetscErrorCode ierr;
  size_t         fullLength;

  PetscFunctionBegin;
  va_start(Argp,string);
  ierr = PetscVSNPrintf(buffer,sizeof(buffer)-9-5,string,&fullLength,Argp);CHKERRQ(ierr);
  va_end(Argp);

  ierr = PetscInfo1(0,"Evaluating MATLAB string: %s\n",buffer);CHKERRQ(ierr);
  engEvalString(mengine->ep, buffer);
  ierr = PetscInfo1(0,"Done evaluating MATLAB string: %s\n",buffer);CHKERRQ(ierr);
  ierr = PetscInfo1(0,"  MATLAB output message: %s\n",mengine->buffer);CHKERRQ(ierr);

  /*
     Check for error in MATLAB: indicated by ? as first character in engine->buffer
  */
  if (mengine->buffer[4] == '?') SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Error in evaluating MATLAB command:%s\n%s",string,mengine->buffer);
  PetscFunctionReturn(0);
}

/*@C
    PetscMatlabEngineGetOutput - Gets a string buffer where the MATLAB output is
          printed

    Not Collective

    Input Parameter:
.   mengine - the MATLAB engine

    Output Parameter:
.   string - buffer where MATLAB output is printed

   Level: advanced

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEnginePut(), PetscMatlabEngineGet(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineCreate(), PetscMatlabEnginePrintOutput(),
          PETSC_MATLAB_ENGINE_(), PetscMatlabEnginePutArray(), PetscMatlabEngineGetArray(), PetscMatlabEngine
@*/
PetscErrorCode  PetscMatlabEngineGetOutput(PetscMatlabEngine mengine,char **string)
{
  PetscFunctionBegin;
  if (!mengine) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Null argument: probably PETSC_MATLAB_ENGINE_() failed");
  *string = mengine->buffer;
  PetscFunctionReturn(0);
}

/*@C
    PetscMatlabEnginePrintOutput - prints the output from MATLAB

    Collective on PetscMatlabEngine

    Input Parameters:
.    mengine - the Matlab engine

   Level: advanced

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEnginePut(), PetscMatlabEngineGet(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineGetOutput(), PetscMatlabEngineCreate(),
          PETSC_MATLAB_ENGINE_(), PetscMatlabEnginePutArray(), PetscMatlabEngineGetArray(), PetscMatlabEngine
@*/
PetscErrorCode  PetscMatlabEnginePrintOutput(PetscMatlabEngine mengine,FILE *fd)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;

  PetscFunctionBegin;
  if (!mengine) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Null argument: probably PETSC_MATLAB_ENGINE_() failed");
  ierr = MPI_Comm_rank(PetscObjectComm((PetscObject)mengine),&rank);CHKERRMPI(ierr);
  ierr = PetscSynchronizedFPrintf(PetscObjectComm((PetscObject)mengine),fd,"[%d]%s",rank,mengine->buffer);CHKERRQ(ierr);
  ierr = PetscSynchronizedFlush(PetscObjectComm((PetscObject)mengine),fd);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    PetscMatlabEnginePut - Puts a Petsc object into the MATLAB space. For parallel objects,
      each processors part is put in a separate  MATLAB process.

    Collective on PetscObject

    Input Parameters:
+    mengine - the MATLAB engine
-    object - the PETSc object, for example Vec

   Level: advanced

   Note: Mats transferred between PETSc and MATLAB and vis versa are transposed in the other space
         (this is because MATLAB uses compressed column format and PETSc uses compressed row format)

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEngineCreate(), PetscMatlabEngineGet(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineGetOutput(), PetscMatlabEnginePrintOutput(),
          PETSC_MATLAB_ENGINE_(), PetscMatlabEnginePutArray(), PetscMatlabEngineGetArray(), PetscMatlabEngine
@*/
PetscErrorCode  PetscMatlabEnginePut(PetscMatlabEngine mengine,PetscObject obj)
{
  PetscErrorCode ierr,(*put)(PetscObject,void*);

  PetscFunctionBegin;
  if (!mengine) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Null argument: probably PETSC_MATLAB_ENGINE_() failed");
  ierr = PetscObjectQueryFunction(obj,"PetscMatlabEnginePut_C",&put);CHKERRQ(ierr);
  if (!put) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Object %s cannot be put into MATLAB engine",obj->class_name);
  ierr = PetscInfo(0,"Putting MATLAB object\n");CHKERRQ(ierr);
  ierr = (*put)(obj,mengine->ep);CHKERRQ(ierr);
  ierr = PetscInfo1(0,"Put MATLAB object: %s\n",obj->name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
    PetscMatlabEngineGet - Gets a variable from MATLAB into a PETSc object.

    Collective on PetscObject

    Input Parameters:
+    mengine - the MATLAB engine
-    object - the PETSc object, for example Vec

   Level: advanced

   Note: Mats transferred between PETSc and MATLAB and vis versa are transposed in the other space
         (this is because MATLAB uses compressed column format and PETSc uses compressed row format)

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEnginePut(), PetscMatlabEngineCreate(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineGetOutput(), PetscMatlabEnginePrintOutput(),
          PETSC_MATLAB_ENGINE_(), PetscMatlabEnginePutArray(), PetscMatlabEngineGetArray(), PetscMatlabEngine
@*/
PetscErrorCode  PetscMatlabEngineGet(PetscMatlabEngine mengine,PetscObject obj)
{
  PetscErrorCode ierr,(*get)(PetscObject,void*);

  PetscFunctionBegin;
  if (!mengine) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Null argument: probably PETSC_MATLAB_ENGINE_() failed");
  if (!obj->name) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Cannot get object that has no name");
  ierr = PetscObjectQueryFunction(obj,"PetscMatlabEngineGet_C",&get);CHKERRQ(ierr);
  if (!get) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"Object %s cannot be gotten from MATLAB engine",obj->class_name);
  ierr = PetscInfo(0,"Getting MATLAB object\n");CHKERRQ(ierr);
  ierr = (*get)(obj,mengine->ep);CHKERRQ(ierr);
  ierr = PetscInfo1(0,"Got MATLAB object: %s\n",obj->name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
    The variable Petsc_Matlab_Engine_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a PetscMatlabEngine
*/
static PetscMPIInt Petsc_Matlab_Engine_keyval = MPI_KEYVAL_INVALID;

/*@C
   PETSC_MATLAB_ENGINE_ - Creates a MATLAB engine on each process in a communicator.

   Not Collective

   Input Parameter:
.  comm - the MPI communicator to share the engine

   Options Database:
.  -matlab_engine_host - hostname

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
PetscMatlabEngine  PETSC_MATLAB_ENGINE_(MPI_Comm comm)
{
  PetscErrorCode    ierr;
  PetscBool         flg;
  PetscMatlabEngine mengine;

  PetscFunctionBegin;
  if (Petsc_Matlab_Engine_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Comm_create_keyval(MPI_COMM_NULL_COPY_FN,MPI_COMM_NULL_DELETE_FN,&Petsc_Matlab_Engine_keyval,0);
    if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_MATLAB_ENGINE_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(NULL);}
  }
  ierr = MPI_Comm_get_attr(comm,Petsc_Matlab_Engine_keyval,(void**)&mengine,(int*)&flg);
  if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_MATLAB_ENGINE_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(NULL);}
  if (!flg) { /* viewer not yet created */
    ierr = PetscMatlabEngineCreate(comm,NULL,&mengine);
    if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_MATLAB_ENGINE_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_REPEAT," ");PetscFunctionReturn(NULL);}
    ierr = PetscObjectRegisterDestroy((PetscObject)mengine);
    if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_MATLAB_ENGINE_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_REPEAT," ");PetscFunctionReturn(NULL);}
    ierr = MPI_Comm_set_attr(comm,Petsc_Matlab_Engine_keyval,mengine);
    if (ierr) {PetscError(PETSC_COMM_SELF,__LINE__,"PETSC_MATLAB_ENGINE_",__FILE__,PETSC_ERR_PLIB,PETSC_ERROR_INITIAL," ");PetscFunctionReturn(NULL);}
  }
  PetscFunctionReturn(mengine);
}

/*@C
    PetscMatlabEnginePutArray - Puts an array into the MATLAB space, treating it as a Fortran style (column major ordering) array. For parallel objects,
      each processors part is put in a separate  MATLAB process.

    Collective on PetscObject

    Input Parameters:
+    mengine - the MATLAB engine
.    m,n - the dimensions of the array
.    array - the array (represented in one dimension)
-    name - the name of the array

   Level: advanced

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEngineCreate(), PetscMatlabEngineGet(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineGetOutput(), PetscMatlabEnginePrintOutput(),
          PETSC_MATLAB_ENGINE_(), PetscMatlabEnginePut(), PetscMatlabEngineGetArray(), PetscMatlabEngine
@*/
PetscErrorCode  PetscMatlabEnginePutArray(PetscMatlabEngine mengine,int m,int n,const PetscScalar *array,const char name[])
{
  PetscErrorCode ierr;
  mxArray        *mat;

  PetscFunctionBegin;
  if (!mengine) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Null argument: probably PETSC_MATLAB_ENGINE_() failed");
  ierr = PetscInfo1(0,"Putting MATLAB array %s\n",name);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  mat = mxCreateDoubleMatrix(m,n,mxREAL);
#else
  mat = mxCreateDoubleMatrix(m,n,mxCOMPLEX);
#endif
  ierr = PetscArraycpy(mxGetPr(mat),array,m*n);CHKERRQ(ierr);
  engPutVariable(mengine->ep,name,mat);

  ierr = PetscInfo1(0,"Put MATLAB array %s\n",name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
    PetscMatlabEngineGetArray - Gets a variable from MATLAB into an array

    Not Collective

    Input Parameters:
+    mengine - the MATLAB engine
.    m,n - the dimensions of the array
.    array - the array (represented in one dimension)
-    name - the name of the array

   Level: advanced

.seealso: PetscMatlabEngineDestroy(), PetscMatlabEnginePut(), PetscMatlabEngineCreate(),
          PetscMatlabEngineEvaluate(), PetscMatlabEngineGetOutput(), PetscMatlabEnginePrintOutput(),
          PETSC_MATLAB_ENGINE_(), PetscMatlabEnginePutArray(), PetscMatlabEngineGet(), PetscMatlabEngine
@*/
PetscErrorCode  PetscMatlabEngineGetArray(PetscMatlabEngine mengine,int m,int n,PetscScalar *array,const char name[])
{
  PetscErrorCode ierr;
  mxArray        *mat;

  PetscFunctionBegin;
  if (!mengine) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_NULL,"Null argument: probably PETSC_MATLAB_ENGINE_() failed");
  ierr = PetscInfo1(0,"Getting MATLAB array %s\n",name);CHKERRQ(ierr);
  mat  = engGetVariable(mengine->ep,name);
  if (!mat) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Unable to get array %s from matlab",name);
  if (mxGetM(mat) != (size_t) m) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Array %s in MATLAB first dimension %d does not match requested size %d",name,(int)mxGetM(mat),m);
  if (mxGetN(mat) != (size_t) n) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_LIB,"Array %s in MATLAB second dimension %d does not match requested size %d",name,(int)mxGetN(mat),m);
  ierr = PetscArraycpy(array,mxGetPr(mat),m*n);CHKERRQ(ierr);
  ierr = PetscInfo1(0,"Got MATLAB array %s\n",name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

