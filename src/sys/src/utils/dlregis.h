/* $Id: dlregis.h,v 1.4 1999/11/10 03:18:08 bsmith Exp bsmith $ */
/*
   This file is included by all the dlregis.c files to provide common information
   on the PETSC team.
*/

static char *authors = PETSC_AUTHOR_INFO;
static char *version = PETSC_VERSION_NUMBER;

EXTERN_C_BEGIN
/* --------------------------------------------------------------------------*/
#undef __FUNC__  
#define  __FUNC__ /*<a name=""></a>*/"DLLibraryInfo"
int DLLibraryInfo(char *path,char *type,char **mess) 
{
  PetscTruth iscon,isaut,isver;
  int        ierr;

  PetscFunctionBegin; 

  ierr = PetscStrcmp(type,"Contents",&iscon);CHKERRQ(ierr);
  ierr = PetscStrcmp(type,"Authors",&isaut);CHKERRQ(ierr);
  ierr = PetscStrcmp(type,"Version",&isver);CHKERRQ(ierr);
  if (iscon)      *mess = contents;
  else if (isaut) *mess = authors;
  else if (isver) *mess = version;
  else            *mess = 0;

  PetscFunctionReturn(0);
}
EXTERN_C_END
