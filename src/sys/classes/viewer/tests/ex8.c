static char help[] = "Tests CGNS viewers.\n\n";

#include <petscsys.h>
#include <petscviewer.h>
#include <petscdm.h>
#include <petscfe.h>

static PetscErrorCode TestOpen(PetscFileMode mode, PetscViewer *viewer)
{
  PetscFunctionBeginUser;
  PetscCall(PetscViewerCGNSOpen(PETSC_COMM_WORLD, "cgns.cgns", mode, viewer));
  PetscCall(PetscViewerSetUp(*viewer));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode GetLongDescription(PetscInt *len, char **desc)
{
  const PetscInt desc_len = 4096, desc_dup_len = 32;
  const char     desc_dup[] = "this is a far longer description";

  PetscFunctionBeginUser;
  *len = desc_len;
  PetscCall(PetscCalloc1(desc_len + 1, desc));
  for (PetscInt i = 0; i < desc_len; i += desc_dup_len) {
    PetscCall(PetscMemcpy(&(*desc)[i], desc_dup, desc_dup_len));
  }
  (*desc)[desc_len] = '\0';
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestWriteDescriptors(PetscViewer viewer)
{
  char    *desc;
  PetscInt desc_len;

  PetscFunctionBeginUser;
  PetscCall(PetscViewerCGNSSetDescriptor(viewer, "Help", "This will be overwritten"));

  PetscCall(GetLongDescription(&desc_len, &desc));
  PetscCall(PetscViewerCGNSSetDescriptor(viewer, "Long Description", desc));
  PetscCall(PetscFree(desc));

  PetscCall(PetscViewerCGNSSetDescriptor(viewer, "Resize", "This causes a resize operation"));

  PetscCall(PetscViewerCGNSSetDescriptor(viewer, "Help", help));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestReadDescriptors(PetscViewer viewer)
{
  char    **descriptors, **names, *expected_desc;
  PetscInt  num_descriptors, expected_desc_len;
  PetscBool is_same;

  PetscFunctionBeginUser;
  PetscCall(PetscViewerCGNSGetDescriptors(viewer, &num_descriptors, &names, &descriptors));
  PetscCheck(num_descriptors == 3, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONGSTATE, "Expected 3 descriptors, got %" PetscInt_FMT, num_descriptors);

  PetscCall(PetscStrcmp(names[0], "Help", &is_same));
  PetscCheck(is_same, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONGSTATE, "Wrong name for descriptor 0, expected 'Help' but got '%s'", names[0] ? names[0] : "(null)");

  PetscCall(PetscStrcmp(names[1], "Long Description", &is_same));
  PetscCheck(is_same, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONGSTATE, "Wrong name for descriptor 1, expected 'Long Description' but got '%s'", names[1] ? names[1] : "(null)");

  PetscCall(PetscStrcmp(names[2], "Resize", &is_same));
  PetscCheck(is_same, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONGSTATE, "Wrong name for descriptor 2, expected 'Resize' but got '%s'", names[2] ? names[2] : "(null)");

  PetscCall(PetscStrcmp(descriptors[0], help, &is_same));
  PetscCheck(is_same, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONGSTATE, "Wrong value for descriptor 0, expected '%s' but got '%s'", help, descriptors[0] ? descriptors[0] : "(null)");

  PetscCall(GetLongDescription(&expected_desc_len, &expected_desc));
  PetscCall(PetscStrcmp(descriptors[1], expected_desc, &is_same));
  PetscCheck(is_same, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONGSTATE, "Wrong value for descriptor 1");
  PetscCall(PetscFree(expected_desc));

  PetscCall(PetscStrcmp(descriptors[2], "This causes a resize operation", &is_same));
  PetscCheck(is_same, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONGSTATE, "Wrong value for descriptor 2, expected 'This causes a resize operation' but got '%s'", descriptors[2] ? descriptors[2] : "(null)");

  PetscCall(PetscViewerCGNSRestoreDescriptors(viewer, &num_descriptors, &names, &descriptors));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestReadDescriptorsEmpty(PetscViewer viewer)
{
  char   **descriptors, **names;
  PetscInt num_descriptors;

  PetscFunctionBeginUser;
  PetscCall(PetscViewerCGNSGetDescriptors(viewer, &num_descriptors, &names, &descriptors));
  PetscCheck(num_descriptors == 0, PetscObjectComm((PetscObject)viewer), PETSC_ERR_ARG_WRONGSTATE, "Descriptors should be empty, found %" PetscInt_FMT, num_descriptors);
  PetscCall(PetscViewerCGNSRestoreDescriptors(viewer, &num_descriptors, &names, &descriptors));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  PetscViewer viewer;
  DM          dm;
  Vec         v;
  PetscInt    dim;
  PetscFE     fe;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMGetCoordinateDim(dm, &dim));
  PetscCall(PetscFECreateDefault(PETSC_COMM_WORLD, dim, 1, PETSC_FALSE, NULL, 2, &fe));
  PetscCall(DMAddField(dm, NULL, (PetscObject)fe));
  PetscCall(DMCreateDS(dm));

  PetscCall(DMGetGlobalVector(dm, &v));
  PetscCall(VecZeroEntries(v));

  PetscCall(TestOpen(FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(v, viewer));
  PetscCall(TestWriteDescriptors(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(TestOpen(FILE_MODE_READ, &viewer));
  PetscCall(TestReadDescriptors(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(TestOpen(FILE_MODE_WRITE, &viewer));
  PetscCall(TestWriteDescriptors(viewer));
  PetscCall(VecView(v, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(TestOpen(FILE_MODE_READ, &viewer));
  PetscCall(TestReadDescriptors(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(TestOpen(FILE_MODE_WRITE, &viewer));
  PetscCall(TestWriteDescriptors(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(TestOpen(FILE_MODE_READ, &viewer));
  PetscCall(TestReadDescriptorsEmpty(viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(DMRestoreGlobalVector(dm, &v));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFEDestroy(&fe));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST
  build:
    requires: cgns
  test:
    nsize: 1 2
    output_file: output/empty.out
    args: -dm_plex_box_faces 3,3,3 -dm_plex_dim 3
TEST*/
