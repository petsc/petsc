/*
   This file is included by all the dlregis.c files to provide common information
   on the PETSC team.
*/

static char version[256];

EXTERN_C_BEGIN
/* --------------------------------------------------------------------------*/
#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryInfo"
PetscErrorCode PetscDLLibraryInfo(char *path,char *type,const char *mess[]) 
{
  PetscTruth iscon,isaut,isver;
  PetscErrorCode ierr;

  PetscFunctionBegin; 

  ierr = PetscStrcmp(type,"Contents",&iscon);CHKERRQ(ierr);
  ierr = PetscStrcmp(type,"Authors",&isaut);CHKERRQ(ierr);
  ierr = PetscStrcmp(type,"Version",&isver);CHKERRQ(ierr);
  if (iscon)      *mess = contents;
  else if (isaut) *mess = authors;
  else if (isver) {ierr = PetscGetVersion(&version);CHKERRQ(ierr);*mess=version;}
  else            *mess = 0;

  PetscFunctionReturn(0);
}
EXTERN_C_END
