/*$Id: ex1.c,v 1.8 1999/10/01 21:22:58 bsmith Exp bsmith $*/

static char help[] = 
"Reads an AODatabase and displays the key and segment names. Runtime options include:\n\
    -f input_file : Specifies input file\n\
    -d : Dumps the entire database\n\
    -e : Allows addition of character string values to the database\n\
    -r : Allows removal of items from the database\n\n";

/*T
   Concepts: AOData^Using an AOData database for grid information;
   Routines: AODataLoadBasic(); AODataKeyExists(); AODataKeyAdd();
   Routines: AODataSegmentAdd(); AODataView();
   Processors: n
T*/

/* 
  Include "ao.h" so that we can use the various AO and AOData routines for
  manipulating simple parallel databases of grid (and related) information.
  Note that this file automatically includes:
     petsc.h  - base PETSc routines   
     sys.h    - system routines
     is.h     - index sets            
*/

#include "ao.h"

#undef __FUNC__
#define __FUNC__ "main"
int main( int argc, char **argv )
{
  int            ierr,flag,bs,zero = 0,edited = 0;
  char           filename[256],string[256],*segname,*value,keyname[256],*ikeyname;
  Viewer         binary;
  AOData         aodata;
  PetscTruth     keyexists;

  /* ---------------------------------------------------------------------
       Initialize PETSc
     --------------------------------------------------------------------- */

  PetscInitialize(&argc,&argv,(char *)0,help);

  /* 
      Load the grid database and initialize graphics 
  */
  /*
     Load in the grid database
  */
  ierr = OptionsGetString(PETSC_NULL,"-f",filename,256,&flag);CHKERRA(ierr);
  if (!flag) SETERRA(1,1,"Unable to open database, must run with: ex1 -f filename");
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,filename,BINARY_RDONLY,&binary);CHKERRA(ierr);
  ierr = AODataLoadBasic(binary,&aodata);CHKERRA(ierr);
  ierr = ViewerDestroy(binary);CHKERRQ(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-d",&flag);CHKERRA(ierr);
  if (!flag) {
    ierr = ViewerPushFormat(VIEWER_STDOUT_WORLD,VIEWER_FORMAT_ASCII_INFO,0);CHKERRA(ierr)
  }
  ierr = AODataView(aodata,VIEWER_STDOUT_WORLD);CHKERRA(ierr);


  /*
       Allow user to add text keys to database
  */
  ierr = OptionsHasName(PETSC_NULL,"-e",&flag);CHKERRA(ierr);
  if (flag) {
    edited = 1;
    printf("Enter keyname: (or return to end) ");
    gets(string);
    while (string[0] != 0) {
      ierr = AODataKeyExists(aodata,string,&keyexists);CHKERRA(ierr);
      if (!keyexists) {
        ierr = AODataKeyAdd(aodata,string,1,1);CHKERRA(ierr);
      }
      ierr = PetscStrcpy(keyname,string);CHKERRQ(ierr);
      printf("Enter segmentname: value (or return to end) ");
      gets(string);
      while (string[0] != 0) {
        ierr    = PetscStrtok(string," ",&segname);CHKERRA(ierr);
        ierr    = PetscStrtok(0," ",&value);CHKERRA(ierr);
        ierr     = PetscStrlen(value,&bs);CHKERRA(ierr);
        ierr = AODataSegmentAdd(aodata,keyname,segname,bs,1,&zero,value,PETSC_CHAR);CHKERRA(ierr);
        printf("Enter segmentname: value (or return to end) ");
        gets(string);
      }
      printf("Enter keyname: (or return to end) ");
      gets(string);
    } 
    ierr = ViewerPushFormat(VIEWER_STDOUT_WORLD,VIEWER_FORMAT_ASCII_INFO,0);CHKERRA(ierr)
    ierr = AODataView(aodata,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  }

  /*
      Allow user to remove keys and segements from database
  */
  ierr = OptionsHasName(PETSC_NULL,"-r",&flag);CHKERRA(ierr);
  if (flag) {
    edited = 1;
    printf("Enter keyname to remove: (or return to end) ");
    gets(string);
    while (string[0] != 0) {
      ierr = AODataKeyRemove(aodata,string);CHKERRA(ierr);
      printf("Enter keyname to remove: (or return to end) ");
      gets(string);
    }
    printf("Enter keyname segment name to remove: (or return to end) ");
    gets(string);
    while (string[0] != 0) {
      ierr = PetscStrtok(string," ",&ikeyname);CHKERRA(ierr);
      ierr = PetscStrtok(0," ",&segname);CHKERRA(ierr);
      ierr = AODataSegmentRemove(aodata,ikeyname,segname);CHKERRA(ierr);
      printf("Enter keyname segment name to remove: (or return to end) ");
      gets(string);
    }
    ierr = ViewerPushFormat(VIEWER_STDOUT_WORLD,VIEWER_FORMAT_ASCII_INFO,0);CHKERRA(ierr)
    ierr = AODataView(aodata,VIEWER_STDOUT_WORLD);CHKERRA(ierr);
  }

  if (edited) {
    PetscStrcat(filename,".new");
    ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,filename,BINARY_CREATE,&binary);CHKERRA(ierr);
    ierr = AODataView(aodata,binary);CHKERRA(ierr);
    ierr = ViewerDestroy(binary);CHKERRQ(ierr);
  }

  ierr = AODataDestroy(aodata);CHKERRA(ierr);


  PetscFinalize();

  PetscFunctionReturn(0);
}




