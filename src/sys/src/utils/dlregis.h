/* $Id: dlregis.h,v 1.8 2001/03/23 23:20:45 balay Exp $ */
/*
   This file is included by all the dlregis.c files to provide common information
   on the PETSC team.
*/

static char *authors = PETSC_AUTHOR_INFO;

EXTERN_C_BEGIN
/* --------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryInfo"
int PetscDLLibraryInfo(char *path,char *type,char **mess) 
{
  PetscTruth iscon,isaut,isver;
  int        ierr;

  PetscFunctionBegin; 

  ierr = PetscStrcmp(type,"Contents",&iscon);CHKERRQ(ierr);
  ierr = PetscStrcmp(type,"Authors",&isaut);CHKERRQ(ierr);
  ierr = PetscStrcmp(type,"Version",&isver);CHKERRQ(ierr);
  if (iscon)      *mess = contents;
  else if (isaut) *mess = authors;
  else if (isver) PetscGetVersion(mess);
  else            *mess = 0;

  PetscFunctionReturn(0);
}
EXTERN_C_END
