/* $Id: matlab.c,v 1.2 2000/04/04 03:33:30 bsmith Exp bsmith $ #include "petsc.h" */

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

#undef __FUNC__  
#define __FUNC__ "PetscMatlabEngineInitialize"
int PetscMatlabEngineInitialize(MPI_Comm comm,char *machine)
{
  PetscMatlabEngine *engine;
  PetscTruth        flg;
  int               ierr;

  PetscFunctionBegin;
  if (Petsc_Matlab_Engine_keyval == MPI_KEYVAL_INVALID) {
    ierr = MPI_Keyval_create(MPI_NULL_COPY_FN,MPI_NULL_DELETE_FN,&Petsc_Matlab_Engine_keyval,0);CHKERRQ(ierr);
  }
  ierr = MPI_Attr_get(comm,Petsc_Matlab_Engine_keyval,(void **)&engine,(int *)&flg);CHKERRQ(ierr);
  if (!flg) { /* engine not yet created */
    engine = PetscNew(PetscMatlabEngine);CHKPTRQ(engine);
    if (!machine) machine = "\0";
    engine->ep = engOpen(machine);
    if (!engine->ep) SETERRQ1(1,1,"Unable to start Matlab engine on %s\n",machine);
    engOutputBuffer(engine->ep,engine->buffer,1024);
    ierr = MPI_Attr_put(comm,Petsc_Matlab_Engine_keyval,(void *)&engine);CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PetscMatlabEngineFinalize"
int PetscMatlabEngineFinalize(MPI_Comm comm)
{
  PetscMatlabEngine *engine;
  int               ierr;
  PetscTruth        flg;
  
  PetscFunctionBegin;
  if (Petsc_Matlab_Engine_keyval == MPI_KEYVAL_INVALID) PetscFunctionReturn(0);
  ierr = MPI_Attr_get(comm,Petsc_Matlab_Engine_keyval,(void **)&engine,(int *)&flg);CHKERRQ(ierr);
  if (!flg) PetscFunctionReturn(0);
  engClose(engine->ep);
  ierr = PetscFree(engine);CHKERRQ(ierr);
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







