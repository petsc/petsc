#define PETSC_DLL

#include "petscsys.h"        
#include <stdarg.h>
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif

#undef __FUNCT__  
#define __FUNCT__ "PetscOptionsGetenv"
/*@C
     PetscOptionsGetenv - Gets an environmental variable, broadcasts to all
          processors in communicator from first.

     Collective on MPI_Comm

   Input Parameters:
+    comm - communicator to share variable
.    name - name of environmental variable
-    len - amount of space allocated to hold variable

   Output Parameters:
+    flag - if not PETSC_NULL tells if variable found or not
-    env - value of variable

  Level: advanced

   Notes:
    You can also "set" the environmental variable by setting the options database value
    -name "stringvalue" (with name in lower case). If name begins with PETSC_ this is 
    discarded before checking the database. For example, PETSC_VIEWER_SOCKET_PORT would 
    be given as -viewer_socket_port 9000

    If comm does not contain the 0th process in the MPIEXEC it is likely on
    many systems that the environmental variable will not be set unless you
    put it in a universal location like a .chsrc file

@*/
PetscErrorCode PETSC_DLLEXPORT PetscOptionsGetenv(MPI_Comm comm,const char name[],char env[],size_t len,PetscTruth *flag)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  char           *str,work[256];
  PetscTruth     flg = PETSC_FALSE,spetsc;
   
  PetscFunctionBegin;

  /* first check options database */
  ierr = PetscStrncmp(name,"PETSC_",6,&spetsc);CHKERRQ(ierr);
  
  ierr = PetscStrcpy(work,"-");CHKERRQ(ierr);
  if (spetsc) {
    ierr = PetscStrcat(work,name+6);CHKERRQ(ierr);
  } else {
    ierr = PetscStrcat(work,name);CHKERRQ(ierr);
  }
  ierr = PetscStrtolower(work);CHKERRQ(ierr);
  if (env) {
    ierr = PetscOptionsGetString(PETSC_NULL,work,env,len,&flg);CHKERRQ(ierr);
    if (flg) {
      if (flag) *flag = PETSC_TRUE;
    } else { /* now check environment */
      ierr = PetscMemzero(env,len*sizeof(char));CHKERRQ(ierr);

      ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
      if (!rank) {
	str = getenv(name);
	if (str) flg = PETSC_TRUE;
	if (str && env) {ierr = PetscStrncpy(env,str,len);CHKERRQ(ierr);}
      }
      ierr = MPI_Bcast(&flg,1,MPI_INT,0,comm);CHKERRQ(ierr);
      ierr = MPI_Bcast(env,len,MPI_CHAR,0,comm);CHKERRQ(ierr);
      if (flag) *flag = flg;
    } 
  } else {
    ierr = PetscOptionsHasName(PETSC_NULL,work,flag);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
     PetscSetDisplay - Tries to set the X windows display variable for all processors.
                       The variable PetscDisplay contains the X windows display variable.

*/
static char PetscDisplay[256]; 

#undef __FUNCT__  
#define __FUNCT__ "PetscSetDisplay" 
PetscErrorCode PETSC_DLLEXPORT PetscSetDisplay(void)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscTruth     flag;
  char           *str,display[256];

  PetscFunctionBegin;
  ierr = PetscMemzero(display,256*sizeof(char));CHKERRQ(ierr);
  ierr = PetscOptionsGetString(PETSC_NULL,"-display",PetscDisplay,256,&flag);CHKERRQ(ierr);
  if (flag) PetscFunctionReturn(0);

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr); 
  if (!rank) {
    str = getenv("DISPLAY");
    if (!str || (str[0] == ':' && size > 1)) {
      ierr = PetscGetHostName(display,255);CHKERRQ(ierr);
      ierr = PetscStrcat(display,":0.0");CHKERRQ(ierr);
    } else {
      ierr = PetscStrncpy(display,str,256);CHKERRQ(ierr);
    }
  }
  ierr = MPI_Bcast(display,256,MPI_CHAR,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  if (rank) {
    str = getenv("DISPLAY");
    /* assume that ssh port forwarding is working */
    if (str && (str[0] != ':')) {
      ierr = PetscStrcpy(display,str);CHKERRQ(ierr);
    }
  }
  ierr = PetscStrcpy(PetscDisplay,display);CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}

#undef __FUNCT__  
#define __FUNCT__ "PetscGetDisplay" 
/*
     PetscGetDisplay - Gets the display variable for all processors.

  Input Parameters:
.   n - length of string display

  Output Parameters:
.   display - the display string

*/
PetscErrorCode PETSC_DLLEXPORT PetscGetDisplay(char display[],size_t n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrncpy(display,PetscDisplay,n);CHKERRQ(ierr);
  PetscFunctionReturn(0);  
}
