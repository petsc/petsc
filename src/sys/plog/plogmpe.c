
/*
      PETSc code to log PETSc events using MPE
*/
#include <petscsys.h>        /*I    "petscsys.h"   I*/
#if defined(PETSC_USE_LOG) && defined (PETSC_HAVE_MPE)
#include <mpe.h>

PetscBool  UseMPE = PETSC_FALSE;
PetscBool  PetscBeganMPE = PETSC_FALSE;

#undef __FUNCT__  
#define __FUNCT__ "PetscLogMPEBegin"
/*@C
   PetscLogMPEBegin - Turns on MPE logging of events. This creates large log files 
   and slows the program down.

   Collective over PETSC_COMM_WORLD

   Options Database Keys:
. -log_mpe - Prints extensive log information (for code compiled
             with PETSC_USE_LOG)

   Notes:
   A related routine is PetscLogBegin (with the options key -log), which is 
   intended for production runs since it logs only flop rates and object
   creation (and should not significantly slow the programs).

   Level: advanced

   Concepts: logging^MPE
   Concepts: logging^message passing

.seealso: PetscLogDump(), PetscLogBegin(), PetscLogAllBegin(), PetscLogEventActivate(),
          PetscLogEventDeactivate()
@*/
PetscErrorCode  PetscLogMPEBegin(void)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
    
  PetscFunctionBegin;
  /* Do MPE initialization */
  if (!MPE_Initialized_logging()) { /* This function exists in mpich 1.1.2 and higher */
    ierr = PetscInfo(0,"Initializing MPE.\n");CHKERRQ(ierr);
    ierr = MPE_Init_log();CHKERRQ(ierr);
    PetscBeganMPE = PETSC_TRUE;
  } else {
    ierr = PetscInfo(0,"MPE already initialized. Not attempting to reinitialize.\n");CHKERRQ(ierr);
  }
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  UseMPE = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogMPEDump"
/*@C
   PetscLogMPEDump - Dumps the MPE logging info to file for later use with Upshot.

   Collective over PETSC_COMM_WORLD

   Level: advanced

.seealso: PetscLogDump(), PetscLogAllBegin(), PetscLogMPEBegin()
@*/
PetscErrorCode  PetscLogMPEDump(const char sname[])
{
  char           name[PETSC_MAX_PATH_LEN];
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (PetscBeganMPE) {
    ierr = PetscInfo(0,"Finalizing MPE.\n");CHKERRQ(ierr);
    if (sname) { ierr = PetscStrcpy(name,sname);CHKERRQ(ierr);}
    else { ierr = PetscGetProgramName(name,PETSC_MAX_PATH_LEN);CHKERRQ(ierr);}
    ierr = MPE_Finish_log(name);CHKERRQ(ierr);
  } else {
    ierr = PetscInfo(0,"Not finalizing MPE (not started by PETSc).\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#endif /* PETSC_USE_LOG && PETSC_HAVE_MPE */


/* Color function used by MPE */


#define PETSC_RGB_COLOR_MAX 39
const char *(PetscRGBColor[PETSC_RGB_COLOR_MAX]) = {
  "OliveDrab:      ",
  "BlueViolet:     ",
  "CadetBlue:      ",
  "CornflowerBlue: ",
  "DarkGoldenrod:  ",
  "DarkGreen:      ",
  "DarkKhaki:      ",
  "DarkOliveGreen: ",
  "DarkOrange:     ",
  "DarkOrchid:     ",
  "DarkSeaGreen:   ",
  "DarkSlateGray:  ",
  "DarkTurquoise:  ",
  "DeepPink:       ",
  "DarkKhaki:      ",
  "DimGray:        ", 
  "DodgerBlue:     ",
  "GreenYellow:    ",
  "HotPink:        ",
  "IndianRed:      ",
  "LavenderBlush:  ",
  "LawnGreen:      ",
  "LemonChiffon:   ", 
  "LightCoral:     ",
  "LightCyan:      ",
  "LightPink:      ",
  "LightSalmon:    ",
  "LightSlateGray: ",
  "LightYellow:    ",
  "LimeGreen:      ",
  "MediumPurple:   ",
  "MediumSeaGreen: ",
  "MediumSlateBlue:",
  "MidnightBlue:   ",
  "MintCream:      ",
  "MistyRose:      ",
  "NavajoWhite:    ",
  "NavyBlue:       ",
  "OliveDrab:      "
};

#undef __FUNCT__  
#define __FUNCT__ "PetscLogGetRGBColor"
/*@C
  PetscLogGetRGBColor - This routine returns a rgb color useable with PetscLogEventRegister()
  
  Not collective. Maybe it should be?

  Output Parameter
. str - charecter string representing the color

  Level: beginner

.keywords: log, mpe , color
.seealso: PetscLogEventRegister
@*/
PetscErrorCode  PetscLogGetRGBColor(const char *str[])
{
  static int idx = 0;

  PetscFunctionBegin;
  *str  = PetscRGBColor[idx];
  idx = (idx + 1)% PETSC_RGB_COLOR_MAX;
  PetscFunctionReturn(0);
}
