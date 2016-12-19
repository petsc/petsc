static char help[] = "REPLACE WITH AN ACTUAL EXAMPLE\n\n";

int main(int argc, char **argv)
{
  ierr = PetscInitialize(&argc, &argv, NULL,help);if (ierr) return ierr;
  ierr = PetscFinalize();
  return ierr;
}
