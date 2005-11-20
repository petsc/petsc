#include "petsc.h" 
#include "src/sys/time/ptime.h"

#undef __FUNCT__  
#define __FUNCT__ "PetscGetTime"
/*@
   PetscGetTime - Returns the current time of day in seconds. This 
   returns wall-clock time.  

   Not Collective

   Output Parameter:
.  v - time counter

   Usage: 
.vb
      PetscLogDouble v1,v2,elapsed_time;
      ierr = PetscGetTime(&v1);CHKERR(ierr);
      .... perform some calculation ...
      ierr = PetscGetTime(&v2);CHKERR(ierr);
      elapsed_time = v2 - v1;   
.ve

   Notes:
   Since the PETSc libraries incorporate timing of phases and operations, 
   PetscGetTime() is intended only for timing of application codes.  
   The options database commands -log, -log_summary, and -log_all activate
   PETSc library timing.  See the users manual for further details.

   Level: intermediate

.seealso: PetscLogEventRegister(), PetscLogEventBegin(), PetscLogEventEnd(),  PetscLogStagePush(), 
          PetscLogStagePop(), PetscLogStageRegister(), PetscGetFlops()

.keywords:  get, time
@*/
PetscErrorCode PETSC_DLLEXPORT PetscGetTime(PetscLogDouble *t)
{
  PetscFunctionBegin;
  PetscTime(*t);
  PetscFunctionReturn(0);
}

