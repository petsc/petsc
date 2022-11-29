static char help[] = "Example for PetscOptionsInsertFileYAML\n";

#include <petscsys.h>
#include <petscviewer.h>

int main(int argc, char **argv)
{
  char      filename[PETSC_MAX_PATH_LEN];
  PetscBool flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscOptionsGetString(NULL, NULL, "-f", filename, sizeof(filename), &flg));
  if (flg) PetscCall(PetscOptionsInsertFileYAML(PETSC_COMM_WORLD, NULL, filename, PETSC_TRUE));

  PetscCall(PetscOptionsGetString(NULL, NULL, "-yaml", filename, sizeof(filename), &flg));
  if (flg) {
    PetscBool monitor = PETSC_FALSE;

    PetscCall(PetscOptionsGetBool(NULL, NULL, "-monitor", &monitor, NULL));
    if (monitor) PetscCall(PetscOptionsMonitorSet(PetscOptionsMonitorDefault, PETSC_VIEWER_STDOUT_WORLD, NULL));
    PetscCall(PetscOptionsClear(NULL));
    PetscCall(PetscOptionsInsertFileYAML(PETSC_COMM_WORLD, NULL, filename, PETSC_TRUE));
  }

  PetscCall(PetscOptionsGetString(NULL, NULL, "-yamlstr", filename, sizeof(filename), &flg));
  if (flg) {
    PetscBool monitor = PETSC_FALSE;

    PetscCall(PetscOptionsGetBool(NULL, NULL, "-monitor", &monitor, NULL));
    if (monitor) PetscCall(PetscOptionsMonitorSet(PetscOptionsMonitorDefault, NULL, NULL));
    PetscCall(PetscOptionsClear(NULL));
    PetscCall(PetscOptionsInsertStringYAML(NULL, filename));
  }

  PetscCall(PetscOptionsView(NULL, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscOptionsClear(NULL));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
     args: -options_left false
     filter:  grep -E -v "(options_left|options_monitor)"
     localrunfiles: petsc.yml

     test:
        suffix: 1
        args: -f petsc.yml

     test:
        suffix: 2_file
        output_file: output/ex47_2.out
        args: -options_file_yaml petsc.yml

     test:
        suffix: 2_string
        args: -options_string_yaml "`cat petsc.yml`"

     test:
        suffix: 2_auto
        args: -options_monitor
        args: -options_file ex47-yaml_tag
        args: -options_file ex47-yaml_doc
        localrunfiles: ex47-yaml_tag ex47-yaml_doc

     test:
        suffix: 2_prefix
        args: -options_monitor
        args: -options_file ex47-opt.txt
        args: -prefix_push p5_ -options_file ex47-opt.yml -prefix_pop
        args: -prefix_push p5_ -options_file ex47-opt.yml:yaml -prefix_pop
        args: -prefix_push p6_ -options_file_yaml ex47-opt.yml -prefix_pop
        args: -prefix_push p7_ -options_string_yaml "`cat ex47-opt.yml`" -prefix_pop
        args: -prefix_push p7_ -options_string_yaml "`cat ex47-opt.yml`" -prefix_pop
        args: -prefix_push p8_ -options_string_yaml "`cat ex47-opt.yml`" -prefix_pop
        args: -prefix_push p9_ -options_file ex47-opt.json -prefix_pop
        localrunfiles: ex47-opt.txt ex47-opt.yml ex47-opt.json

   testset:
     nsize: {{1 2}}

     test:
        suffix: 3_empty
        args: -yaml ex47-empty.yaml
        localrunfiles: ex47-empty.yaml

     test:
        suffix: 3_merge
        args: -yaml ex47-merge.yaml -monitor
        localrunfiles: ex47-merge.yaml

     test:
        suffix: 3_env
        env: PETSC_OPTIONS_YAML='"name: value"'
        filter: grep -E -v -e "(options_left)"
        args: -monitor

     test:
        suffix: 3_str
        args: -yamlstr "name: value" -monitor

     test:
        suffix: 3_options
        args: -yaml ex47-options.yaml
        localrunfiles: ex47-options.yaml

     test:
        suffix: 3_include
        args: -yaml ex47-include.yaml
        localrunfiles: ex47-include.yaml ex47-empty.yaml ex47-options.yaml

     test:
        suffix: 3_prefix
        args: -yaml ex47-prefix.yaml
        localrunfiles: ex47-prefix.yaml

     test:
        suffix: 3_multidoc
        args: -yaml ex47-multidoc.yaml
        localrunfiles: ex47-multidoc.yaml

TEST*/
