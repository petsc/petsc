
static char help[] = "Reads an AODatabase and displays the key and segment names. Runtime options include:\n\
    -f input_file : Specifies input file\n\
    -d : Dumps the entire database\n\
    -e : Allows addition of character string values to the database\n\
    -r : Allows removal of items from the database\n\n";

/*T
   Concepts: AOData^using an AOData database for grid information;
   Processors: n
T*/

/* 
  Include "petscao.h" so that we can use the various AO and AOData routines for
  manipulating simple parallel databases of grid (and related) information.
  Note that this file automatically includes:
     petsc.h       - base PETSc routines   
     petscsys.h    - system routines
     petscis.h     - index sets            
*/

#include "petscao.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  int            ierr,zero = 0,edited = 0;
  size_t         bs;
  char           filename[PETSC_MAX_PATH_LEN],string[256],*segname,*value,keyname[256],*ikeyname;
  PetscViewer    binary;
  AOData         aodata;
  PetscTruth     keyexists,flag;

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
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",filename,256,&flag);CHKERRQ(ierr);
  if (!flag) SETERRQ(1,"Unable to open database, must run with: ex1 -f filename");
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&binary);CHKERRQ(ierr);
  ierr = AODataLoadBasic(binary,&aodata);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(binary);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-d",&flag);CHKERRQ(ierr);
  if (!flag) {
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr)
  }
  ierr = AODataView(aodata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);


  /*
       Allow user to add text keys to database
  */
  ierr = PetscOptionsHasName(PETSC_NULL,"-e",&flag);CHKERRQ(ierr);
  if (flag) {
    edited = 1;
    printf("Enter keyname: (or return to end) ");
    fgets(string, 256, stdin);
    while (string[0] != 0) {
      ierr = AODataKeyExists(aodata,string,&keyexists);CHKERRQ(ierr);
      if (!keyexists) {
        ierr = AODataKeyAdd(aodata,string,1,1);CHKERRQ(ierr);
      }
      ierr = PetscStrcpy(keyname,string);CHKERRQ(ierr);
      printf("Enter segmentname: value (or return to end) ");
      fgets(string, 256, stdin);
      while (string[0] != 0) {
        PetscToken *token;
        ierr    = PetscTokenCreate(string,' ',&token);CHKERRQ(ierr);
        ierr    = PetscTokenFind(token,&segname);CHKERRQ(ierr);
        ierr    = PetscTokenFind(token,&value);CHKERRQ(ierr);
        ierr    = PetscTokenDestroy(token);CHKERRQ(ierr);
        ierr    = PetscStrlen(value,&bs);CHKERRQ(ierr);
        ierr = AODataSegmentAdd(aodata,keyname,segname,bs,1,&zero,value,PETSC_CHAR);CHKERRQ(ierr);
        printf("Enter segmentname: value (or return to end) ");
        fgets(string, 256, stdin);
      }
      printf("Enter keyname: (or return to end) ");
      fgets(string, 256, stdin);
    } 
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr)
    ierr = AODataView(aodata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /*
      Allow user to remove keys and segements from database
  */
  ierr = PetscOptionsHasName(PETSC_NULL,"-r",&flag);CHKERRQ(ierr);
  if (flag) {
    edited = 1;
    printf("Enter keyname to remove: (or return to end) ");
    fgets(string, 256, stdin);
    while (string[0] != 0) {
      ierr = AODataKeyRemove(aodata,string);CHKERRQ(ierr);
      printf("Enter keyname to remove: (or return to end) ");
      fgets(string, 256, stdin);
    }
    printf("Enter keyname segment name to remove: (or return to end) ");
    fgets(string, 256, stdin);
    while (string[0] != 0) {
      PetscToken *token;
      ierr = PetscTokenCreate(string,' ',&token);CHKERRQ(ierr);
      ierr = PetscTokenFind(token,&ikeyname);CHKERRQ(ierr);
      ierr = PetscTokenFind(token,&segname);CHKERRQ(ierr);
      ierr = PetscTokenDestroy(token);CHKERRQ(ierr);
      ierr = AODataSegmentRemove(aodata,ikeyname,segname);CHKERRQ(ierr);
      printf("Enter keyname segment name to remove: (or return to end) ");
      fgets(string, 256, stdin);
    }
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr)
    ierr = AODataView(aodata,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  if (edited) {
    PetscStrcat(filename,".new");
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&binary);CHKERRQ(ierr);
    ierr = AODataView(aodata,binary);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(binary);CHKERRQ(ierr);
  }

  ierr = AODataDestroy(aodata);CHKERRQ(ierr);


  ierr = PetscFinalize();CHKERRQ(ierr);

  PetscFunctionReturn(0);
}




