static char help[] = "REPLACE WITH AN ACTUAL EXAMPLE\n\n";

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  ierr = PetscInitialize(&argc, &argv, PETSC_NULL, help);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
