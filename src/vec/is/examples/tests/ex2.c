
static char help[]= "Tests ISView() and ISLoad() \n\n";

#include <petscis.h>
#include <petscviewer.h>

int main(int argc,char **argv)
{
  PetscErrorCode         ierr;
  PetscInt               n = 3,ix[2][3] = {{3,5,4},{1,7,9}};
  IS                     isx,il;
  PetscMPIInt            size,rank;
  PetscViewer            vx,vl;
  PetscBool              equal;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size > 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_SIZ,"Example only works with one or two processes");
  ierr = ISCreateGeneral(PETSC_COMM_WORLD,n,ix[rank],PETSC_COPY_VALUES,&isx);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"testfile",FILE_MODE_WRITE,&vx);CHKERRQ(ierr);
  ierr = ISView(isx,vx);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vx);CHKERRQ(ierr);

  ierr = ISCreate(PETSC_COMM_WORLD,&il);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"testfile",FILE_MODE_READ,&vl);CHKERRQ(ierr);
  ierr = ISLoad(il,vl);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&vl);CHKERRQ(ierr);

  ierr = ISEqual(il,isx,&equal);CHKERRQ(ierr);
  if (!equal) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Index set loaded from file does not match index set to file");
  ierr = ISDestroy(&il);CHKERRQ(ierr);
  ierr = ISDestroy(&isx);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

