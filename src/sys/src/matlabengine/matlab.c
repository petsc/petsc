/* $Id: matlab.c,v 1.4 2000/04/30 04:28:01 bsmith Exp bsmith $ #include "petsc.h" */

#include "engine.h"   /* Matlab include file */
#include "petsc.h" 

typedef struct {
  Engine   *ep;
  char     buffer[1024];
} PetscMatlabEngine;

/*
    The variable Petsc_Matlab_Engine_keyval is used to indicate an MPI attribute that
  is attached to a communicator, in this case the attribute is a Matlab Engine.
*/
static int Petsc_Matlab_Engine_keyval = MPI_KEYVAL_INVALID;

/*
   This routine is called by MPI when a communicator that has a Matlab Engine associated
 with it is freed.
*/
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "Petsc_DelMatlabEngine"
int Petsc_DelMatlabEngine(MPI_Comm comm,int keyval,void* attr_val,void* extra_state)
{
  PetscMatlabEngine *engine = (PetscMatlabEngine*)attr_val;
  int               ierr;
  
  PetscFunctionBegin;
  engClose(engine->ep);
  ierr = PetscFree(engine);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "PetscMatlabEngineInitialize"
int PetscMatlabEngineInitialize(MPI_Comm comm,char *machine)
{
  PetscMatlabEngine *engine;
  PetscTruth        flg;
  int               ierr,rank,size;
  char              buffer[128];

  PetscFunctionBegin;
  if (Petsc_Matlab_Engine_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,Petsc_DelMatlabEngine,&Petsc_Matlab_Engine_keyval,0);CHKERRQ(ierr);
  }
  ierr = MPI_Attr_get(comm,Petsc_Matlab_Engine_keyval,(void **)&engine,(int *)&flg);CHKERRQ(ierr);
  if (!flg) { /* engine not yet created */
    engine = PetscNew(PetscMatlabEngine);CHKPTRQ(engine);
    if (!machine) machine = "\0";
    engine->ep = engOpen(machine);
    if (!engine->ep) SETERRQ1(1,1,"Unable to start Matlab engine on %s\n",machine);
    engOutputBuffer(engine->ep,engine->buffer,1024);
    ierr = MPI_Attr_put(comm,Petsc_Matlab_Engine_keyval,engine);CHKERRQ(ierr);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
    ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
    sprintf(buffer,"MPI_Comm_rank = %d; MPI_Comm_size = %d;\n",rank,size);
    engEvalString(engine->ep, buffer);
  }

  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "PetscMatlabEngineEvaluate"
int PetscMatlabEngineEvaluate(MPI_Comm comm,char *string)
{
  PetscMatlabEngine *engine;
  int               ierr;
  PetscTruth        flg;

  PetscFunctionBegin;  
  if (Petsc_Matlab_Engine_keyval == MPI_KEYVAL_INVALID) {
    ierr = PetscMatlabEngineInitialize(comm,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = MPI_Attr_get(comm,Petsc_Matlab_Engine_keyval,(void **)&engine,(int *)&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscMatlabEngineInitialize(comm,PETSC_NULL);CHKERRQ(ierr);
    ierr = MPI_Attr_get(comm,Petsc_Matlab_Engine_keyval,(void **)&engine,(int *)&flg);CHKERRQ(ierr);
  } 
  PLogInfo(0,"Evaluating Matlab string: %s\n",string);
  engEvalString(engine->ep, string);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscMatlabEngineGetOutput"
int PetscMatlabEngineGetOutput(MPI_Comm comm,char **string)
{
  PetscMatlabEngine *engine;
  int               ierr;
  PetscTruth        flg;

  PetscFunctionBegin;  
  if (Petsc_Matlab_Engine_keyval == MPI_KEYVAL_INVALID) {
    ierr = PetscMatlabEngineInitialize(comm,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = MPI_Attr_get(comm,Petsc_Matlab_Engine_keyval,(void **)&engine,(int *)&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscMatlabEngineInitialize(comm,PETSC_NULL);CHKERRQ(ierr);
    ierr = MPI_Attr_get(comm,Petsc_Matlab_Engine_keyval,(void **)&engine,(int *)&flg);CHKERRQ(ierr);
  }
  *string = engine->buffer + 2;
  PetscFunctionReturn(0);
}

/*
    This is a hacked version. Should do somethine better
*/
#undef __FUNC__  
#define __FUNC__ "PetscMatlabEnginePut"
int PetscMatlabEnginePut(PetscObject obj)
{
  PetscMatlabEngine *engine;
  int               ierr,(*put)(PetscObject,void*);
  PetscTruth        flg;
  MPI_Comm          comm;
  
  PetscFunctionBegin;  
  if (Petsc_Matlab_Engine_keyval == MPI_KEYVAL_INVALID) {
    ierr = PetscMatlabEngineInitialize(comm,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscObjectGetComm(obj,&comm);CHKERRQ(ierr);
  ierr = MPI_Attr_get(comm,Petsc_Matlab_Engine_keyval,(void **)&engine,(int *)&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscMatlabEngineInitialize(comm,PETSC_NULL);CHKERRQ(ierr);
    ierr = MPI_Attr_get(comm,Petsc_Matlab_Engine_keyval,(void **)&engine,(int *)&flg);CHKERRQ(ierr);
  } 
  ierr = PetscObjectQueryFunction(obj,"PetscMatlabEnginePut_C",(void**)&put);CHKERRQ(ierr);
  if (!put) {
    SETERRQ1(1,1,"Object %s cannot be put into Matlab engine",obj->class_name);
  }
  ierr = (*put)(obj,engine->ep);CHKERRQ(ierr);
  PLogInfo(0,"Putting Matlab object: %s\n",obj->name);

  PetscFunctionReturn(0);
}

/*
    This is a hacked version. Should do somethine better
*/
#undef __FUNC__  
#define __FUNC__ "PetscMatlabEngineGet"
int PetscMatlabEngineGet(PetscObject obj)
{
  PetscMatlabEngine *engine;
  int               ierr,(*get)(PetscObject,void*);
  PetscTruth        flg;
  MPI_Comm          comm;
  
  PetscFunctionBegin;  
  if (Petsc_Matlab_Engine_keyval == MPI_KEYVAL_INVALID) {
    ierr = PetscMatlabEngineInitialize(comm,PETSC_NULL);CHKERRQ(ierr);
  }
  ierr = PetscObjectGetComm(obj,&comm);CHKERRQ(ierr);
  ierr = MPI_Attr_get(comm,Petsc_Matlab_Engine_keyval,(void **)&engine,(int *)&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscMatlabEngineInitialize(comm,PETSC_NULL);CHKERRQ(ierr);
    ierr = MPI_Attr_get(comm,Petsc_Matlab_Engine_keyval,(void **)&engine,(int *)&flg);CHKERRQ(ierr);
  } 
  ierr = PetscObjectQueryFunction(obj,"PetscMatlabEngineGet_C",(void**)&get);CHKERRQ(ierr);
  if (!get) {
    SETERRQ1(1,1,"Object %s cannot be get into Matlab engine",obj->class_name);
  }
  ierr = (*get)(obj,engine->ep);CHKERRQ(ierr);
  PLogInfo(0,"Getting Matlab object: %s\n",obj->name);

  PetscFunctionReturn(0);
}





