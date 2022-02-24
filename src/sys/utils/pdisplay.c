
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
  PetscMPIInt    rank;
  char           *str,work[256];
  PetscBool      flg = PETSC_FALSE,spetsc;

  PetscFunctionBegin;
  /* first check options database */
  CHKERRQ(PetscStrncmp(name,"PETSC_",6,&spetsc));

  CHKERRQ(PetscStrcpy(work,"-"));
  if (spetsc) {
    CHKERRQ(PetscStrlcat(work,name+6,sizeof(work)));
  } else {
    CHKERRQ(PetscStrlcat(work,name,sizeof(work)));
  }
  CHKERRQ(PetscStrtolower(work));
  if (env) {
    CHKERRQ(PetscOptionsGetString(NULL,NULL,work,env,len,&flg));
    if (flg) {
      if (flag) *flag = PETSC_TRUE;
    } else { /* now check environment */
      CHKERRQ(PetscArrayzero(env,len));

      CHKERRMPI(MPI_Comm_rank(comm,&rank));
      if (rank == 0) {
        str = getenv(name);
        if (str) flg = PETSC_TRUE;
        if (str && env) CHKERRQ(PetscStrncpy(env,str,len));
      }
      CHKERRMPI(MPI_Bcast(&flg,1,MPIU_BOOL,0,comm));
      CHKERRMPI(MPI_Bcast(env,len,MPI_CHAR,0,comm));
      if (flag) *flag = flg;
    }
  } else {
    CHKERRQ(PetscOptionsHasName(NULL,NULL,work,flag));
  }
  PetscFunctionReturn(0);
}

/*
     PetscSetDisplay - Tries to set the X Windows display variable for all processors.
                       The variable PetscDisplay contains the X Windows display variable.

*/
static char PetscDisplay[256];

static PetscErrorCode PetscWorldIsSingleHost(PetscBool  *onehost)
{
  char           hostname[256],roothostname[256];
  PetscMPIInt    localmatch,allmatch;
  PetscBool      flag;

  PetscFunctionBegin;
  CHKERRQ(PetscGetHostName(hostname,sizeof(hostname)));
  CHKERRQ(PetscMemcpy(roothostname,hostname,sizeof(hostname)));
  CHKERRMPI(MPI_Bcast(roothostname,sizeof(roothostname),MPI_CHAR,0,PETSC_COMM_WORLD));
  CHKERRQ(PetscStrcmp(hostname,roothostname,&flag));

  localmatch = (PetscMPIInt)flag;

  CHKERRMPI(MPIU_Allreduce(&localmatch,&allmatch,1,MPI_INT,MPI_LAND,PETSC_COMM_WORLD));

  *onehost = (PetscBool)allmatch;
  PetscFunctionReturn(0);
}

PetscErrorCode  PetscSetDisplay(void)
{
  PetscMPIInt    size,rank;
  PetscBool      flag,singlehost=PETSC_FALSE;
  char           display[sizeof(PetscDisplay)];
  const char     *str;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-display",PetscDisplay,sizeof(PetscDisplay),&flag));
  if (flag) PetscFunctionReturn(0);

  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  CHKERRQ(PetscWorldIsSingleHost(&singlehost));

  str = getenv("DISPLAY");
  if (!str) str = ":0.0";
#if defined(PETSC_HAVE_X)
  flag = PETSC_FALSE;
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-x_virtual",&flag,NULL));
  if (flag) {
    /*  this is a crude hack, but better than nothing */
    CHKERRQ(PetscPOpen(PETSC_COMM_WORLD,NULL,"pkill -9 Xvfb","r",NULL));
    CHKERRQ(PetscSleep(1));
    CHKERRQ(PetscPOpen(PETSC_COMM_WORLD,NULL,"Xvfb :15 -screen 0 1600x1200x24","r",NULL));
    CHKERRQ(PetscSleep(5));
    str  = ":15";
  }
#endif
  if (str[0] != ':' || singlehost) {
    CHKERRQ(PetscStrncpy(display,str,sizeof(display)));
  } else if (rank == 0) {
    CHKERRQ(PetscGetHostName(display,sizeof(display)));
    CHKERRQ(PetscStrlcat(display,str,sizeof(display)));
  }
  CHKERRMPI(MPI_Bcast(display,sizeof(display),MPI_CHAR,0,PETSC_COMM_WORLD));
  CHKERRQ(PetscMemcpy(PetscDisplay,display,sizeof(PetscDisplay)));

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
  PetscFunctionBegin;
  CHKERRQ(PetscStrncpy(display,PetscDisplay,n));
  PetscFunctionReturn(0);
}
