/*
   This file is included by all the dlregis.c files to provide common information
   on the PETSC team.
*/

static char version[256];

EXTERN_C_BEGIN
/* --------------------------------------------------------------------------*/
/* Note: This routine is exported by several different libraries, so we use DLL_EXPORT here. */
/* This is a problem on Windows, so we must generate the dll normally, then manually remove  */
/* the symbol from the generated import library, forcing this symbol to only be found via    */
/* GetProcAddress on the proper dll.  This is how PETSc calls this routine on every OS.  To  */
/* accomplish this, we must generate the import library with the symbol present, then build  */
/* a .def file from the dll and manually add the PRIVATE specifier to this symbol, and then  */
/* regenerate the import library using this .def.  UUUUGH!!!!  I'm fixing this later.        */

/* Maybe this isn't a problem if these symbols never get requested other than dynamically.   */
/* Hmmmmm....                                                                                */
#undef __FUNCT__  
#define __FUNCT__ "PetscDLLibraryInfo"
PetscErrorCode PETSC_DLL_EXPORT PetscDLLibraryInfo(char *path,char *type,const char *mess[]) 
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
