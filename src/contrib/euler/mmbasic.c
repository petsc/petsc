#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: mmbasic.c,v 1.8 1998/05/19 01:34:08 curfman Exp curfman $";
#endif

/*
    The MM (multi-model) interface routines, callable by users.
*/
#include "mmimpl.h"
#include "pinclude/pviewer.h"

#undef __FUNC__  
#define __FUNC__ "MMPrintHelp"
/*@
   MMPrintHelp - Prints all the options for the MM component.

   Input Parameter:
.  mm - the multi-model context

   Options Database Keys:
$  -help, -h

.seealso: MMSetFromOptions()
@*/
int MMPrintHelp(MM mm)
{
  char p[64]; 
  int  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mm,mm->MM_COOKIE);
  PetscStrcpy(p,"-");
  if (mm->prefix) PetscStrcat(p,mm->prefix);
  PetscPrintf(mm->comm,"MM options --------------------------------------------------\n");
  ierr = DLRegisterPrintTypes(mm->comm,stdout,mm->prefix,"mm_type",MMList); CHKERRQ(ierr);
  (*PetscHelpPrintf)(mm->comm,"Run program with -help %smm_type <method> for help on ",p);
  (*PetscHelpPrintf)(mm->comm,"a particular method\n");
  if (mm->printhelp) {ierr = (*mm->printhelp)(mm,p); CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MMDestroy"
/*@C
   MMDestroy - Destroys MM context that was created with MMCreate().

   Input Parameter:
.  mm - the multi-model context

.seealso: MMCreate(), MMSetUp()
@*/
int MMDestroy(MM mm)
{
  int ierr = 0;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(mm,mm->MM_COOKIE);
  if (--mm->refct > 0) return 0;

  if (mm->destroy) {ierr =  (*mm->destroy)(mm);CHKERRQ(ierr);}
  else {if (mm->data) PetscFree(mm->data);}
  PLogObjectDestroy(mm);
  PetscHeaderDestroy(mm);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MMView"
/*
   MMView - Prints the multi-model data structure.

   Input Parameters:
.  mm - the multi-model context
.  viewer - visualization context

   Note:
   The available visualization contexts include
$     VIEWER_STDOUT_SELF - standard output (default)
$     VIEWER_STDOUT_WORLD - synchronized standard
$       output where only the first processor opens
$       the file.  All other processors send their 
$       data to the first processor to print. 

   The user can open alternative visualization contexts with
$    ViewerFileOpenASCII() - output to a specified file

.seealso: ViewerFileOpenASCII()
*/
int MMView(MM mm,Viewer viewer)
{
  FILE        *fd;
  MMType      method;
  int         ierr;
  ViewerType  vtype;

  PetscFunctionBegin;
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  MMGetType(mm,&method);
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    PetscFPrintf(mm->comm,fd,"MM Object:\n");
    PetscFPrintf(mm->comm,fd,"  method: %s\n",method);
    if (mm->view) (*mm->view)(mm,viewer);
  } else if (vtype == STRING_VIEWER) {
    ierr = ViewerStringSPrintf(viewer," %-7.7s",method); CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MMGetNumberOfComponents"
/*
   MMGetNumberOfComponents - Gets the number of components in the multi-model data structure.

   Input Parameter:
.  mm - the multi-model context

   Output Parameter:
.  nc - number of components

.seealso: MMView()
*/
int MMGetNumberOfComponents(MM mm,int *nc)
{
  PetscFunctionBegin;
  *nc = mm->ncomponents;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "MMCreate"
/*@C
   MMCreate - Creates a multi-model context.

   Input Parameter:
.  comm - MPI communicator 

   Output Parameter:
.  mm - location to put the multi-model context

.seealso: MMSetUp(), MMApply(), MMDestroy()
@*/

int MMCreate(MPI_Comm comm,MM *newmm)
{
  MM  mm;
  int ierr, MM_COOKIE = 0;

  PetscFunctionBegin;
  *newmm          = 0;

  ierr = PetscRegisterCookie(&MM_COOKIE); CHKERRQ(ierr);
  PetscHeaderCreate(mm,_p_MM,int,MM_COOKIE,-1,comm,MMDestroy,MMView);
  PLogObjectCreate(mm);
  *newmm             = mm;
  mm->type           = -1;
  mm->view           = 0;
  mm->printhelp      = 0;
  mm->setfromoptions = 0;
  mm->setupcalled    = 0;
  mm->MM_COOKIE      = MM_COOKIE;
  PetscFunctionReturn(0);
}
