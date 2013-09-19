static const char help[] = "EXAMPLE REMOVED.\n\n";

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char *argv[])
{
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc,&argv,0,help);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
