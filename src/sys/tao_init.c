#include "tao.h"   /*I  "tao.h" I*/

PetscErrorCode TAOSOLVER_DLLEXPORT TaoRegisterEvents();

/* ------------------------Nasty global variables -------------------------------*/
int TaoInitializeCalled = 0;
//PetscClassId TAOSOLVER_CLASSID = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "TaoInitialize"
/*@C 
  TaoInitialize - Initializes the TAO component and many of the packages associated with it.

   Collective on MPI_COMM_WORLD

   Input Parameters:
+  argc - [optional] count of number of command line arguments
.  args - [optional] the command line arguments
.  file - [optional] PETSc database file, defaults to ~username/.petscrc
          (use TAO_NULL for default)
-  help - [optional] Help message to print, use TAO_NULL for no message

   Note:
   TaoInitialize() should always be called near the beginning of your 
   program.  However, this command should come after PetscInitialize()

   Note:
   The input arguments are required if the options database is to be used.

   Level: beginner

.keywords: TAO_SOLVER, initialize

.seealso: TaoInitializeFortran(), TaoFinalize(), PetscInitialize()
@*/
/*PetscErrorCode TaoInitialize(int *argc,char ***args,char file[],const char help[])
{
  PetscErrorCode info=0;

  PetscFunctionBegin;

  if (TaoInitializeCalled){ PetscFunctionReturn(0);}
  TaoInitializeCalled++;


  TAO_CLASSID = 0;
  info=TaoLogClassRegister(&TAO_CLASSID,"TAO"); CHKERRQ(info);

  info = TaoRegisterEvents(); CHKERRQ(info);
  info = TaoStandardRegisterAll();CHKERRQ(info);
  info = PetscInfo(0,"TaoInitialize:TAO successfully started\n"); CHKERRQ(info);
  PetscFunctionReturn(info);
}
*/
/*@C 
  TaoInitializeNoArguments - Initializes the TAO component and many of the packages associated with it.

   Collective on MPI_COMM_WORLD


   Note:
   TaoInitializeNoArguments() provides a way to initialize TAO when the
   command line arguments are not available.  TaoInitialize() is the preferred
   way to initialize TAO.

   Note:
   The input arguments are required if the options database is to be used.

   Level: beginner

.keywords: TAO_SOLVER, initialize

.seealso: TaoInitialize(), TaoFinalize(), PetscInitializeNoArguments()
@*/
 /*PetscErrorCode  PETSC_DLLEXPORT TaoInitializeNoArguments()
{
  int info=0;
  PetscFunctionBegin;
  info = TaoInitialize(0,0,0,0); CHKERRQ(info);
  PetscFunctionReturn(info);
}
 */
 
#undef __FUNCT__  
#define __FUNCT__ "TaoFinalize"
/*@C
   TaoFinalize - Checks for options at the end of the TAO program
   and finalizes interfaces with other packages.

   Collective on MPI_COMM_WORLD

   Level: beginner

.keywords: finalize, exit, end

.seealso: TaoInitialize(), PetscFinalize()
@*/
  /*int TaoFinalize(void)
{
  int info;
  
  PetscFunctionBegin;
  TaoInitializeCalled--;
  if (TaoInitializeCalled==0){
    info = PetscInfo(0,"TaoFinalize:Tao successfully ended!\n"); 
           CHKERRQ(info);
    info = TaoRegisterDestroy(); CHKERRQ(info);
  }
  PetscFunctionReturn(0);
}
  */

#undef __FUNCT__  
#define __FUNCT__ "TaoRegisterEvents"
// int Tao_Solve, Tao_LineSearch;
/*
   TaoRegisterEvents - Registers TAO events for use in performance logging.
*/
   /*
int TaoRegisterEvents()
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

   */
