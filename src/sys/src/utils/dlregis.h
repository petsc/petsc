/* $Id: dlregis.c,v 1.1 1999/10/22 01:10:33 bsmith Exp bsmith $ */
/*
   This file is included by all the dlregis.c files to provide common information
   on the PETSC team.
*/

static char *authors = PETSC_AUTHOR_INFO;
static char *version = PETSC_VERSION_NUMBER;

EXTERN_C_BEGIN
/* --------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "DLLibraryInfo"
int DLLibraryInfo(char *path,char *type,char **mess) 
{
  int iscon,isaut,isver;

  PetscFunctionBegin; 

  iscon = !PetscStrcmp(type,"Contents");
  isaut = !PetscStrcmp(type,"Authors");
  isver = !PetscStrcmp(type,"Version");
  if (iscon)      *mess = contents;
  else if (isaut) *mess = authors;
  else if (isver) *mess = version;
  else            *mess = 0;

  PetscFunctionReturn(0);
}
EXTERN_C_END
