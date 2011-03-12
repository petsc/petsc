
#include "private/daimpl.h"     /*I  "petscdm.h"   I*/


#undef __FUNCT__  
#define __FUNCT__ "DMDALoad"
/*@C
      DMDALoad - Creates an appropriate DMDA and loads its global coordinate vector from a binary file saved with DAView()

   Input Parameter:
+    viewer - a binary viewer (created with PetscViewerBinaryOpen())
.    M - number of processors in x direction
.    N - number of processors in y direction
-    P - number of processors in z direction

   Output Parameter:
.    da - the DMDA object

   Notes: If DMView() is called with multiple different DA's each one writes into the info file WITHOUT using the DM options prefix hence
     loading them with DMDALoad() will likely crash. Need to change DMView() to use the options prefix and DMDALoad() to take a already 
     partly constructed DM (like VecLoad()) and use its prefix for reading info.

   Level: intermediate

@*/
PetscErrorCode  DMDALoad(PetscViewer viewer,PetscInt M,PetscInt N,PetscInt P,DM *da)
{
  PetscErrorCode ierr;
  PetscInt       info[10],nmax = 10,i;
  MPI_Comm       comm;
  char           fieldnametag[32],fieldname[64];
  PetscBool      isbinary,flag;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,1);
  PetscValidPointer(da,5);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERBINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_ARG_WRONG,"Must be binary viewer");

  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);

  ierr = PetscOptionsGetIntArray(PETSC_NULL,"-daload_info",info,&nmax,&flag);CHKERRQ(ierr);
  if (!flag) SETERRQ(((PetscObject)viewer)->comm,PETSC_ERR_FILE_UNEXPECTED,"No DMDA information in file");
  if (nmax != 10) SETERRQ1(((PetscObject)viewer)->comm,PETSC_ERR_FILE_UNEXPECTED,"Wrong number of items in DMDA information in file: %D",nmax);

  if (info[0] == 1) {
    ierr = DMDACreate1d(comm,(DMDABoundaryType) info[7],info[1],info[4],info[5],0,da);CHKERRQ(ierr);
  } else if (info[0] == 2) {
    ierr = DMDACreate2d(comm,(DMDABoundaryType) info[7],(DMDABoundaryType) info[8],(DMDAStencilType) info[6],info[1],info[2],M,N,info[4],info[5],0,0,da);CHKERRQ(ierr);
  } else if (info[0] == 3) {
    ierr = DMDACreate3d(comm,(DMDABoundaryType) info[7],(DMDABoundaryType) info[8],(DMDABoundaryType) info[9],(DMDAStencilType) info[6],info[1],info[2],info[3],M,N,P,info[4],info[5],0,0,0,da);CHKERRQ(ierr);
  } else SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_UNEXPECTED,"Dimension in info file is not 1, 2, or 3 it is %D",info[0]);

  for (i=0; i<info[4]; i++) {
    sprintf(fieldnametag,"-daload_fieldname_%d",(int)i);
    ierr = PetscOptionsGetString(PETSC_NULL,fieldnametag,fieldname,64,&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = DMDASetFieldName(*da,i,fieldname);CHKERRQ(ierr);
    }
  }

  /*
    Read in coordinate information if kept in file
  */
  flag = PETSC_FALSE;
  ierr = PetscOptionsGetBool(PETSC_NULL,"-daload_coordinates",&flag,PETSC_NULL);CHKERRQ(ierr);
  if (flag) {
    DM      dac;
    Vec     tmpglobal,global;
    PetscInt mlocal;

    if (info[0] == 1) {
      ierr = DMDACreate1d(comm,DMDA_BOUNDARY_NONE,info[1],1,0,0,&dac);CHKERRQ(ierr);
    } else if (info[0] == 2) {
      ierr = DMDACreate2d(comm,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_BOX,info[1],info[2],M,N,2,0,0,0,&dac);CHKERRQ(ierr);
    } else if (info[0] == 3) {
      ierr = DMDACreate3d(comm,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_BOUNDARY_NONE,DMDA_STENCIL_BOX,info[1],info[2],info[3],M,N,P,3,0,0,0,0,&dac);CHKERRQ(ierr);
    }

    /* this nonsense is so that the vector set to DMDASetCoordinates() does NOT have a DMDA */
    /* We should change the handling of coordinates so there is always a coordinate DMDA when there is a coordinate vector */
    ierr = DMCreateGlobalVector(dac,&tmpglobal);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject)tmpglobal,"coor_");CHKERRQ(ierr);
    ierr = VecLoad(tmpglobal,viewer);CHKERRQ(ierr);
    ierr = VecGetLocalSize(tmpglobal,&mlocal);CHKERRQ(ierr);
    ierr = VecCreateMPI(comm,mlocal,PETSC_DETERMINE,&global);CHKERRQ(ierr);
    ierr = VecCopy(tmpglobal,global);CHKERRQ(ierr);
    ierr = VecDestroy(tmpglobal);CHKERRQ(ierr); 
    ierr = DMDestroy(dac);CHKERRQ(ierr);
    ierr = DMDASetCoordinates(*da,global);CHKERRQ(ierr);
    ierr = VecDestroy(global);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
