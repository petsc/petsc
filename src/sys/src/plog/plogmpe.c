/*$Id: plogmpe.c,v 1.52 2000/05/05 22:14:18 balay Exp bsmith $*/
/*
      PETSc code to log PETSc events using MPE
*/
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
int PLogEventMPEFlags[] = {  1,1,1,1,1,  /* 0 - 24*/
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

char *(PLogEventColor[]) = {"OliveDrab:      ",
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
int PLogEventColorMalloced[] = {0,0,0,0,0,0,0,0,0,0,
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

int UseMPE = 0;
int PetscBeganMPE = 0;
extern char *PLogEventName[];

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogMPEBegin"
/*@C
   PLogMPEBegin - Turns on MPE logging of events. This creates large log files 
   and slows the program down.

   Collective over PETSC_COMM_WORLD

   Options Database Keys:
. -log_mpe - Prints extensive log information (for code compiled
             with PETSC_USE_LOG)

   Notes:
   A related routine is PLogBegin (with the options key -log), which is 
   intended for production runs since it logs only flop rates and object
   creation (and should not significantly slow the programs).

   Level: advanced

   Concepts: logging^MPE
   Concepts: logging^message passing

.seealso: PLogDump(), PLogBegin(), PLogAllBegin(), PLogEventActivate(),
          PLogEventDeactivate()
@*/
int PLogMPEBegin(void)
{
  int        i,rank,ierr;
  PetscTruth flg;
    
  PetscFunctionBegin;
  /* Do MPE initialization */
#if defined (PETSC_HAVE_MPE_INITIALIZED_LOGGING)
  if (!MPE_Initialized_logging()) { /* This function exists in mpich 1.1.2 and higher */
    PLogInfo(0,"PLogMPEBegin: Initializing MPE.\n");
    ierr = MPE_Init_log();CHKERRQ(ierr);
    PetscBeganMPE = 1;
  } else {
    PLogInfo(0,"PLogMPEBegin: MPE already initialized. Not attempting to reinitialize.\n");
  }
#else
  ierr = OptionsHasName(PETSC_NULL,"-log_mpe_avoid_init",&flg);CHKERRQ(ierr);
  if (flg) {
    PLogInfo(0,"PLogMPEBegin: Not initializing MPE.\n");
  } else {
    PLogInfo(0,"PLogMPEBegin: Initializing MPE.\n");
    ierr = MPE_Init_log();CHKERRQ(ierr);
    PetscBeganMPE = 1;
  }
#endif
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  if (!rank) {
    for (i=0; i < PLOG_USER_EVENT_HIGH; i++) {
      if (PLogEventMPEFlags[i]) {
        MPE_Describe_state(MPEBEGIN+2*i,MPEBEGIN+2*i+1,PLogEventName[i],PLogEventColor[i]);
      }
    }
  }
  UseMPE = 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogEventMPEDeactivate"
/*@
    PLogEventMPEDeactivate - Indicates that a particular event should not be
       logged using MPE. Note: the event may be either a pre-defined
       PETSc event (found in include/petsclog.h) or an event number obtained
       with PLogEventRegister().

   Not Collective

   Input Parameter:
.  event - integer indicating event

   Example of Usage:
.vb
     PLogEventMPEDeactivate(VEC_SetValues);
        [code where you do not want to log VecSetValues()]
     PLogEventMPEActivate(VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve

   Level: advanced

.seealso: PLogEventMPEActivate(), PlogEventActivate(), PlogEventDeactivate()
@*/
int PLogEventMPEDeactivate(int event)
{
  PetscFunctionBegin;
  PLogEventMPEFlags[event] = 0;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogEventMPEActivate"
/*@
    PLogEventMPEActivate - Indicates that a particular event should be
       logged using MPE. Note: the event may be either a pre-defined
       PETSc event (found in include/petsclog.h) or an event number obtained
       with PLogEventRegister().

   Not Collective

   Input Parameter:
.  event - integer indicating event

   Example of Usage:
.vb
     PLogEventMPEDeactivate(VEC_SetValues);
        [code where you do not want to log VecSetValues()]
     PLogEventMPEActivate(VEC_SetValues);
        [code where you do want to log VecSetValues()]
.ve

   Level: advanced

.seealso: PLogEventMPEDeactivate(), PLogEventActivate(), PLogEventDeactivate()
@*/
int PLogEventMPEActivate(int event)
{
  PetscFunctionBegin;
  PLogEventMPEFlags[event] = 1;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ /*<a name=""></a>*/"PLogMPEDump"
/*@C
   PLogMPEDump - Dumps the MPE logging info to file for later use with Upshot.

   Collective over PETSC_COMM_WORLD

   Level: advanced

.seealso: PLogDump(), PLogAllBegin(), PLogMPEBegin()
@*/
int PLogMPEDump(const char sname[])
{
  char name[256];
  int  ierr;

  PetscFunctionBegin;
  if (PetscBeganMPE == 1) {
    PLogInfo(0,"PLogMPEDump: Finalizing MPE.\n");
    if (sname) { ierr = PetscStrcpy(name,sname);CHKERRQ(ierr);}
    else { ierr = PetscGetProgramName(name,256);CHKERRQ(ierr);}
    ierr = MPE_Finish_log(name);CHKERRQ(ierr);
  } else {
    PLogInfo(0,"PLogMPEDump: Not finalizing MPE.\n");
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

