/*$Id: plogmpe.c,v 1.58 2001/05/29 16:34:36 bsmith Exp $*/
/*
      PETSc code to log PETSc events using MPE
*/
#include "petscconfig.h"
#include "petsc.h"        /*I    "petsc.h"   I*/
#if defined(PETSC_USE_LOG) && defined (PETSC_HAVE_MPE)
#include "petscsys.h"
#include "mpe.h"

/* 
   Make sure that all events used by PETSc have the
   corresponding flags set here: 
     1 - activated for MPE logging
     0 - not activated for MPE logging
 */
int PetscLogEventMPEFlags[] = {  1,1,1,1,1,  /* 0 - 24*/
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        0,1,1,1,1,  /* 25 -49 */
                        1,1,1,1,1,
                        1,1,0,0,0,
                        1,1,1,1,1,
                        1,1,1,1,1,
                        1,1,1,1,1, /* 50 - 74 */
                        1,1,1,1,1,
                        1,1,1,1,0,
                        0,0,0,0,0,
                        1,1,1,0,1,
                        1,1,1,1,1, /* 75 - 99 */
                        1,1,1,1,1,
                        1,1,0,0,0,
                        1,1,1,1,0,
                        0,0,0,0,0,
                        1,0,0,0,0, /* 100 - 124 */ 
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0, /* 125 - 149 */
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0, /* 150 - 174 */
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0, /* 175 - 199 */
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0,
                        0,0,0,0,0};

/* For Colors, check out the file  /usr/local/X11/lib/rgb.txt */

char *(PetscLogEventColor[]) = {"OliveDrab:      ",
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
                            "OliveDrab:      ",
                            "OrangeRed:      ",
                            "PaleGoldenrod:  ",
                            "PaleVioletRed:  ",
                            "PapayaWhip:     ",
                            "PeachPuff:      ",
                            "RosyBrown:      ",
                            "SaddleBrown:    ",
                            "OrangeRed:      ",
                            "SteelBlue:      ",
                            "VioletRed:      ",
                            "beige:          ",
                            "chocolate:      ",
                            "coral:          ",
                            "gold:           ",
                            "magenta:        ",
                            "maroon:         ",
                            "orchid:         ",
                            "pink:           ",
                            "plum:           ",
                            "red:            ",
                            "tan:            ",
                            "tomato:         ",
                            "violet:         ",
                            "wheat:          ",
                            "yellow:         ",
                            "AliceBlue:      ",
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
                            "OliveDrab:      ",
                            "OrangeRed:      ",
                            "PaleGoldenrod:  ",
                            "PaleVioletRed:  ",
                            "PapayaWhip:     ",
                            "PeachPuff:      ",
                            "RosyBrown:      ",
                            "SaddleBrown:    ",
                            "OrangeRed:      ",
                            "SteelBlue:      ",
                            "VioletRed:      ",
                            "beige:          ",
                            "chocolate:      ",
                            "coral:          ",
                            "gold:           ",
                            "magenta:        ",
                            "maroon:         ",
                            "orchid:         ",
                            "pink:           ",
                            "AliceBlue:      ",
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
                            "OliveDrab:      ",
                            "OrangeRed:      ",
                            "PaleGoldenrod:  ",
                            "PaleVioletRed:  ",
                            "PapayaWhip:     ",
                            "PeachPuff:      ",
                            "RosyBrown:      ",
                            "SaddleBrown:    ",
                            "OrangeRed:      ",
                            "SteelBlue:      ",
                            "VioletRed:      ",
                            "beige:          ",
                            "chocolate:      ",
                            "coral:          ",
                            "gold:           ",
                            "magenta:        ",
                            "maroon:         ",
                            "orchid:         ",
                            "pink:           ",
                            "plum:           ",
                            "red:            ",
                            "tan:            ",
                            "tomato:         ",
                            "violet:         ",
                            "wheat:          ",
                            "yellow:         ",
                            "AliceBlue:      ",
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
                            "DarkKhaki:      "};

/*
    Indicates if a color was malloced for each event, or if it is
  the default color. Used to ensure malloced space is properly freed.
*/
int PetscLogEventColorMalloced[] = {0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0,
                                0,0,0,0,0,0,0,0,0,0};

PetscTruth UseMPE = PETSC_FALSE;
PetscTruth PetscBeganMPE = PETSC_FALSE;
extern char *PetscLogEventName[];

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
int PetscLogMPEBegin(void)
{
  int        rank,ierr;
#if !defined (PETSC_HAVE_MPE_INITIALIZED_LOGGING)
  PetscTruth flg;
#endif
    
  PetscFunctionBegin;
  /* Do MPE initialization */
#if defined (PETSC_HAVE_MPE_INITIALIZED_LOGGING)
  if (!MPE_Initialized_logging()) { /* This function exists in mpich 1.1.2 and higher */
    PetscLogInfo(0,"PetscLogMPEBegin: Initializing MPE.\n");
    ierr = MPE_Init_log();CHKERRQ(ierr);
    PetscBeganMPE = PETSC_TRUE;
  } else {
    PetscLogInfo(0,"PetscLogMPEBegin: MPE already initialized. Not attempting to reinitialize.\n");
  }
#else
  ierr = PetscOptionsHasName(PETSC_NULL,"-log_mpe_avoid_init",&flg);CHKERRQ(ierr);
  if (flg) {
    PetscLogInfo(0,"PetscLogMPEBegin: Not initializing MPE.\n");
  } else {
    PetscLogInfo(0,"PetscLogMPEBegin: Initializing MPE.\n");
    ierr = MPE_Init_log();CHKERRQ(ierr);
    PetscBeganMPE = PETSC_TRUE;
  }
#endif
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (!rank) {
    /*    for (i=0; i < PETSC_LOG_USER_EVENT_HIGH; i++) {
      if (PetscLogEventMPEFlags[i]) {
        MPE_Describe_state(MPEBEGIN+2*i,MPEBEGIN+2*i+1,PetscLogEventName[i],PetscLogEventColor[i]);
      }
      } */
  }
  UseMPE = PETSC_TRUE;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogEventMPEDeactivate"
/*@
    PetscLogEventMPEDeactivate - Indicates that a particular event should not be
       logged using MPE. Note: the event may be either a pre-defined
       PETSc event (found in include/petsclog.h) or an event number obtained
       with PetscLogEventRegister().

   Not Collective

   Input Parameter:
.  event - integer indicating event

   Example of Usage:
.vb
     PetscLogEventMPEDeactivate(VEC_SetValues);
        [code where you do not want to log VecSetValues()]
     PetscLogEventMPEActivate(VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve

   Level: advanced

.seealso: PetscLogEventMPEActivate(), PetscLogEventActivate(), PetscLogEventDeactivate()
@*/
int PetscLogEventMPEDeactivate(int event)
{
  PetscFunctionBegin;
  PetscLogEventMPEFlags[event] = 0;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PetscLogEventMPEActivate"
/*@
    PetscLogEventMPEActivate - Indicates that a particular event should be
       logged using MPE. Note: the event may be either a pre-defined
       PETSc event (found in include/petsclog.h) or an event number obtained
       with PetscLogEventRegister().

   Not Collective

   Input Parameter:
.  event - integer indicating event

   Example of Usage:
.vb
     PetscLogEventMPEDeactivate(VEC_SetValues);
        [code where you do not want to log VecSetValues()]
     PetscLogEventMPEActivate(VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve

   Level: advanced

.seealso: PetscLogEventMPEDeactivate(), PetscLogEventActivate(), PetscLogEventDeactivate()
@*/
int PetscLogEventMPEActivate(int event)
{
  PetscFunctionBegin;
  PetscLogEventMPEFlags[event] = 1;
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
int PetscLogMPEDump(const char sname[])
{
  char name[256];
  int  ierr;

  PetscFunctionBegin;
  if (PetscBeganMPE) {
    PetscLogInfo(0,"PetscLogMPEDump: Finalizing MPE.\n");
    if (sname) { ierr = PetscStrcpy(name,sname);CHKERRQ(ierr);}
    else { ierr = PetscGetProgramName(name,256);CHKERRQ(ierr);}
    ierr = MPE_Finish_log(name);CHKERRQ(ierr);
  } else {
    PetscLogInfo(0,"PetscLogMPEDump: Not finalizing MPE.\n");
  }
  PetscFunctionReturn(0);
}

#else

/*
     Dummy function so that compilers will not complain about 
  empty files.
*/
int PETScMPEDummy(int dummy)
{
  return 0;
}

#endif

