
#include <petscsys.h>

/*@C
     PetscOptionsGetenv - Gets an environmental variable, broadcasts to all
          processors in communicator from first.

     Collective

   Input Parameters:
+    comm - communicator to share variable
.    name - name of environmental variable
-    len - amount of space allocated to hold variable

   Output Parameters:
+    flag - if not NULL tells if variable found or not
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
    ierr = PetscStrlcat(work,name+6,sizeof(work));CHKERRQ(ierr);
  } else {
    ierr = PetscStrlcat(work,name,sizeof(work));CHKERRQ(ierr);
  }
  ierr = PetscStrtolower(work);CHKERRQ(ierr);
  if (env) {
    ierr = PetscOptionsGetString(NULL,NULL,work,env,len,&flg);CHKERRQ(ierr);
    if (flg) {
      if (flag) *flag = PETSC_TRUE;
    } else { /* now check environment */
      ierr = PetscArrayzero(env,len);CHKERRQ(ierr);

      ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
      if (!rank) {
        str = getenv(name);
        if (str) flg = PETSC_TRUE;
        if (str && env) {ierr = PetscStrncpy(env,str,len);CHKERRQ(ierr);}
      }
      ierr = MPI_Bcast(&flg,1,MPIU_BOOL,0,comm);CHKERRQ(ierr);
      ierr = MPI_Bcast(env,len,MPI_CHAR,0,comm);CHKERRQ(ierr);
      if (flag) *flag = flg;
    }
  } else {
    ierr = PetscOptionsHasName(NULL,NULL,work,flag);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
     PetscSetDisplay - Tries to set the X windows display variable for all processors.
                       The variable PetscDisplay contains the X windows display variable.

*/
static char PetscDisplay[256];

static PetscErrorCode PetscWorldIsSingleHost(PetscBool  *onehost)
{
  PetscErrorCode ierr;
  char           hostname[256],roothostname[256];
  PetscMPIInt    localmatch,allmatch;
  PetscBool      flag;

  PetscFunctionBegin;
  ierr = PetscGetHostName(hostname,sizeof(hostname));CHKERRQ(ierr);
  ierr = PetscMemcpy(roothostname,hostname,sizeof(hostname));CHKERRQ(ierr);
  ierr = MPI_Bcast(roothostname,sizeof(roothostname),MPI_CHAR,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = PetscStrcmp(hostname,roothostname,&flag);CHKERRQ(ierr);

  localmatch = (PetscMPIInt)flag;

  ierr = MPIU_Allreduce(&localmatch,&allmatch,1,MPI_INT,MPI_LAND,PETSC_COMM_WORLD);CHKERRQ(ierr);

  *onehost = (PetscBool)allmatch;
  PetscFunctionReturn(0);
}


PetscErrorCode  PetscSetDisplay(void)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscBool      flag,singlehost=PETSC_FALSE;
  char           display[sizeof(PetscDisplay)];
  const char     *str;

  PetscFunctionBegin;
  ierr = PetscOptionsGetString(NULL,NULL,"-display",PetscDisplay,sizeof(PetscDisplay),&flag);CHKERRQ(ierr);
  if (flag) PetscFunctionReturn(0);

  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  ierr = PetscWorldIsSingleHost(&singlehost);CHKERRQ(ierr);

  str = getenv("DISPLAY");
  if (!str) str = ":0.0";
#if defined(PETSC_HAVE_X)
  flag = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-x_virtual",&flag,NULL);CHKERRQ(ierr);
  if (flag) {
    /*  this is a crude hack, but better than nothing */
    ierr = PetscPOpen(PETSC_COMM_WORLD,NULL,"pkill -9 Xvfb","r",NULL);CHKERRQ(ierr);
    ierr = PetscSleep(1);CHKERRQ(ierr);
    ierr = PetscPOpen(PETSC_COMM_WORLD,NULL,"Xvfb :15 -screen 0 1600x1200x24","r",NULL);CHKERRQ(ierr);
    ierr = PetscSleep(5);CHKERRQ(ierr);
    str  = ":15";
  }
#endif
  if (str[0] != ':' || singlehost) {
    ierr = PetscStrncpy(display,str,sizeof(display));CHKERRQ(ierr);
  } else if (!rank) {
    ierr = PetscGetHostName(display,sizeof(display));CHKERRQ(ierr);
    ierr = PetscStrlcat(display,str,sizeof(display));CHKERRQ(ierr);
  }
  ierr = MPI_Bcast(display,sizeof(display),MPI_CHAR,0,PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = PetscMemcpy(PetscDisplay,display,sizeof(PetscDisplay));CHKERRQ(ierr);

  PetscDisplay[sizeof(PetscDisplay)-1] = 0;
  PetscFunctionReturn(0);
}

/*
     PetscGetDisplay - Gets the display variable for all processors.

  Input Parameters:
.   n - length of string display

  Output Parameters:
.   display - the display string

  Options Database:
+  -display <display> - sets the display to use
-  -x_virtual - forces use of a X virtual display Xvfb that will not display anything but -draw_save will still work. Xvfb is automatically
                started up in PetscSetDisplay() with this option

*/
PetscErrorCode  PetscGetDisplay(char display[],size_t n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscStrncpy(display,PetscDisplay,n);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
