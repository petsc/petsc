
/*
    Run with -google_refresh_token XXX to allow access to your Google Drive or else it will prompt you to enter log in information for Google Drive.
*/

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  char           access_token[512];

  PetscInitialize(&argc,&argv,NULL,NULL);

  ierr = PetscGoogleDriveRefresh(PETSC_COMM_WORLD,NULL,access_token,sizeof(access_token));CHKERRQ(ierr);
  ierr = PetscGoogleDriveUpload(PETSC_COMM_WORLD,access_token,"googledriveupload.c");CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}


