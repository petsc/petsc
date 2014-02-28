

#include <petscsys.h>
#include <../src/sys/webclient/authorization.h>
PETSC_EXTERN PetscErrorCode PetscGoogleDriveAuthorize(MPI_Comm,char[],char[],size_t);
PETSC_EXTERN PetscErrorCode PetscGoogleDriveUpload(MPI_Comm,const char[],const char []);
PETSC_EXTERN PetscErrorCode PetscGoogleDriveRefresh(MPI_Comm,const char[],char[],size_t);

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  char           access_token[512],refresh_token[512];

  PetscInitialize(&argc,&argv,NULL,NULL);
  /*
  ierr = PetscGoogleDriveAuthorize(PETSC_COMM_WORLD,access_token,refresh_token,sizeof(access_token));CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Access token: %s Refresh token %s\n",access_token,refresh_token);CHKERRQ(ierr);
  */

  ierr = PetscGoogleDriveRefresh(PETSC_COMM_WORLD,BARRYS_REFRESH_TOKEN,access_token,sizeof(access_token));CHKERRQ(ierr);
  ierr = PetscGoogleDriveUpload(PETSC_COMM_WORLD,access_token,"googledriveauthorize.c");CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}


