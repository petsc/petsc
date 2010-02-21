#define PETSCDM_DLL

#include "private/daimpl.h"     /*I  "petscda.h"   I*/


#undef __FUNCT__  
#define __FUNCT__ "DALoad"
/*@C
      DALoad - Creates an appropriate DA and loads its global vector from a file.

   Input Parameter:
+    viewer - a binary viewer (created with PetscViewerBinaryOpen())
.    M - number of processors in x direction
.    N - number of processors in y direction
-    P - number of processors in z direction

   Output Parameter:
.    da - the DA object

   Level: intermediate

@*/
PetscErrorCode PETSCDM_DLLEXPORT DALoad(PetscViewer viewer,PetscInt M,PetscInt N,PetscInt P,DA *da)
{
  PetscErrorCode ierr;
  PetscInt       info[8],nmax = 8,i;
  MPI_Comm       comm;
  char           fieldnametag[32],fieldname[64];
  PetscTruth     isbinary,flag;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_COOKIE,1);
  PetscValidPointer(da,5);
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_BINARY,&isbinary);CHKERRQ(ierr);
  if (!isbinary) SETERRQ(PETSC_ERR_ARG_WRONG,"Must be binary viewer");

  ierr = PetscObjectGetComm((PetscObject)viewer,&comm);CHKERRQ(ierr);

  ierr = PetscOptionsGetIntArray(PETSC_NULL,"-daload_info",info,&nmax,&flag);CHKERRQ(ierr);
  if (!flag) {
    SETERRQ(PETSC_ERR_FILE_UNEXPECTED,"No DA information in file");
  }
  if (nmax != 8) {
    SETERRQ1(PETSC_ERR_FILE_UNEXPECTED,"Wrong number of items in DA information in file: %D",nmax);
  }
  if (info[0] == 1) {
    ierr = DACreate1d(comm,(DAPeriodicType) info[7],info[1],info[4],info[5],0,da);CHKERRQ(ierr);
  } else if (info[0] == 2) {
    ierr = DACreate2d(comm,(DAPeriodicType) info[7],(DAStencilType) info[6],info[1],info[2],M,N,info[4],
                      info[5],0,0,da);CHKERRQ(ierr);
  } else if (info[0] == 3) {
    ierr = DACreate3d(comm,(DAPeriodicType) info[7],(DAStencilType) info[6],info[1],info[2],info[3],M,N,P,
                      info[4],info[5],0,0,0,da);CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_FILE_UNEXPECTED,"Dimension in info file is not 1, 2, or 3 it is %D",info[0]);
  }
  for (i=0; i<info[4]; i++) {
    sprintf(fieldnametag,"-daload_fieldname_%d",(int)i);
    ierr = PetscOptionsGetString(PETSC_NULL,fieldnametag,fieldname,64,&flag);CHKERRQ(ierr);
    if (flag) {
      ierr = DASetFieldName(*da,i,fieldname);CHKERRQ(ierr);
    }
  }

  /*
    Read in coordinate information if kept in file
  */
  flag = PETSC_FALSE;
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-daload_coordinates",&flag,PETSC_NULL);CHKERRQ(ierr);
  if (flag) {
    DA      dac;
    Vec     tmpglobal,global;
    PetscInt mlocal;

    if (info[0] == 1) {
      ierr = DACreate1d(comm,DA_NONPERIODIC,info[1],1,0,0,&dac);CHKERRQ(ierr);
    } else if (info[0] == 2) {
      ierr = DACreate2d(comm,DA_NONPERIODIC,DA_STENCIL_BOX,info[1],info[2],M,N,2,
                      0,0,0,&dac);CHKERRQ(ierr);
    } else if (info[0] == 3) {
      ierr = DACreate3d(comm,DA_NONPERIODIC,DA_STENCIL_BOX,info[1],info[2],info[3],M,N,P,
                        3,0,0,0,0,&dac);CHKERRQ(ierr);
    }

    /* this nonsense is so that the vector set to DASetCoordinates() does NOT have a DA */
    /* We should change the handling of coordinates so there is always a coordinate DA when there is a coordinate vector */
    ierr = DACreateGlobalVector(dac,&tmpglobal);CHKERRQ(ierr);
    ierr = PetscObjectSetOptionsPrefix((PetscObject)tmpglobal,"coor_");CHKERRQ(ierr);
    ierr = VecLoadIntoVector(viewer,tmpglobal);CHKERRQ(ierr);
    ierr = VecGetLocalSize(tmpglobal,&mlocal);CHKERRQ(ierr);
    ierr = VecCreateMPI(comm,mlocal,PETSC_DETERMINE,&global);CHKERRQ(ierr);
    ierr = VecCopy(tmpglobal,global);CHKERRQ(ierr);
    ierr = VecDestroy(tmpglobal);CHKERRQ(ierr); 
    ierr = DADestroy(dac);CHKERRQ(ierr);
    ierr = DASetCoordinates(*da,global);CHKERRQ(ierr);
    ierr = VecDestroy(global);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
