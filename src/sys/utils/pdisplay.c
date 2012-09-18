
#include <petscsys.h>
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
PetscErrorCode  PetscOptionsGetenv(MPI_Comm comm,const char name[],char env[],size_t len,PetscBool  *flag)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  char           *str,work[256];
  PetscBool      flg = PETSC_FALSE,spetsc;

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
#define __FUNCT__ "PetscWorldIsSingleHost"
static PetscErrorCode PetscWorldIsSingleHost(PetscBool  *onehost)
{
  PetscErrorCode ierr;
  char           hostname[256],roothostname[256];
  PetscMPIInt    localmatch,allmatch;
  PetscBool      flag;

  PetscFunctionBegin;
  ierr = PetscGetHostName(hostname,256);CHKERRQ(ierr);
  ierr = PetscMemcpy(roothostname,hostname,256);CHKERRQ(ierr);
  ierr = MPI_Bcast(roothostname,256,MPI_CHAR,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = PetscStrcmp(hostname,roothostname,&flag);CHKERRQ(ierr);
  localmatch = (PetscMPIInt)flag;
  ierr = MPI_Allreduce(&localmatch,&allmatch,1,MPI_INT,MPI_LAND,PETSC_COMM_WORLD);CHKERRQ(ierr);
  *onehost = (PetscBool)allmatch;
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "PetscSetDisplay"
PetscErrorCode  PetscSetDisplay(void)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscBool      flag,singlehost=PETSC_FALSE;
  char           display[sizeof(PetscDisplay)];
  const char     *str;

  PetscFunctionBegin;
  ierr = PetscOptionsGetString(PETSC_NULL,"-display",PetscDisplay,sizeof(PetscDisplay),&flag);CHKERRQ(ierr);
  if (flag) PetscFunctionReturn(0);

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = PetscWorldIsSingleHost(&singlehost);CHKERRQ(ierr);

  str = getenv("DISPLAY");
  if (!str) str = ":0.0";
  if (str[0] != ':' || singlehost) {
    ierr = PetscStrncpy(display,str,sizeof(display));CHKERRQ(ierr);
  } else {
    if (!rank) {
      size_t len;
      ierr = PetscGetHostName(display,sizeof(display));CHKERRQ(ierr);
      ierr = PetscStrlen(display,&len);CHKERRQ(ierr);
      ierr = PetscStrncat(display,str,sizeof(display)-len-1);CHKERRQ(ierr);
    }
  }
  ierr = MPI_Bcast(display,sizeof(display),MPI_CHAR,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = PetscMemcpy(PetscDisplay,display,sizeof(PetscDisplay));CHKERRQ(ierr);
  PetscDisplay[sizeof(PetscDisplay)-1] = 0;
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
PetscErrorCode  PetscGetDisplay(char display[],size_t n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrncpy(display,PetscDisplay,n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
