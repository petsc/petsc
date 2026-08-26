static char help[] = "Tests that DMSetLocalSection() and DMSetGlobalSection() keep a local-to-global mapping built by the DM implementation.\n\n";

#include <petscdmda.h>
#include <petscsection.h>

int main(int argc, char **argv)
{
  DM                     da;
  PetscSection           section;
  ISLocalToGlobalMapping ltog, ltognew;
  PetscInt               p, xs, xm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 8, 1, 1, NULL, &da));
  PetscCall(DMSetUp(da));
  /* DMDA builds its mapping in DMSetUp() and cannot rebuild it from a section, so the section setters must keep it.
     Hold a reference to the mapping so a wrongly rebuilt one cannot alias its address. */
  PetscCall(DMGetLocalToGlobalMapping(da, &ltog));
  PetscCall(PetscObjectReference((PetscObject)ltog));
  PetscCall(DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL));
  PetscCall(PetscSectionCreate(PETSC_COMM_WORLD, &section));
  PetscCall(PetscSectionSetChart(section, xs, xs + xm));
  for (p = xs; p < xs + xm; ++p) PetscCall(PetscSectionSetDof(section, p, 1));
  PetscCall(PetscSectionSetUp(section));
  PetscCall(DMSetLocalSection(da, section));
  PetscCall(PetscSectionDestroy(&section));
  PetscCall(DMGetLocalToGlobalMapping(da, &ltognew));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "l2g map %s by DMSetLocalSection()\n", ltognew == ltog ? "kept" : "NOT kept"));
  PetscCall(DMSetGlobalSection(da, NULL));
  PetscCall(DMGetLocalToGlobalMapping(da, &ltognew));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "l2g map %s by DMSetGlobalSection()\n", ltognew == ltog ? "kept" : "NOT kept"));
  PetscCall(ISLocalToGlobalMappingDestroy(&ltog));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # A DMDA builds its local-to-global mapping in DMSetUp() and registers no way to rebuild it from a
  # section, so the section setters must not destroy it (they invalidate only section-derived maps)
  test:
    suffix: keep_ltog

  test:
    suffix: keep_ltog_par
    nsize: 2
    output_file: output/ex55_keep_ltog.out

TEST*/
