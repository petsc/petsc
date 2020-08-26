#include <petsc/private/petscimpl.h>        /*I  "petscsys.h"   I*/
#include <yaml.h>

enum storage_flags {VAR,VAL,SEQ};     /* "Store as" switch */

static PetscErrorCode PetscParseLayerYAML(PetscOptions options, yaml_document_t *doc, yaml_node_t *node)
{
  char            option[PETSC_MAX_PATH_LEN],prefix[PETSC_MAX_PATH_LEN];
  yaml_node_pair_t *start, *top;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscSNPrintf(option,PETSC_MAX_PATH_LEN,"%s"," ");CHKERRQ(ierr);
  if (node->type != YAML_MAPPING_NODE) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Unsupported yaml node type: expected mapping");
  start = node->data.mapping.pairs.start;
  top   = node->data.mapping.pairs.top;
  for (yaml_node_pair_t *pair = start; pair < top; pair++) {
    int key_id = pair->key;
    int value_id = pair->value;
    yaml_node_t *key = NULL;
    yaml_node_t *value = NULL;

    key = yaml_document_get_node(doc, key_id);
    if (!key) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_LIB, "Corrupt yaml document");
    value = yaml_document_get_node(doc, value_id);
    if (!value) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_LIB, "Corrupt yaml document");
    if (key->type != YAML_SCALAR_NODE) SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Unsupported yaml node type: expected scalar");
    if (value->type == YAML_SCALAR_NODE) {
      ierr = PetscSNPrintf(option,PETSC_MAX_PATH_LEN,"-%s %s",(char *)(key->data.scalar.value),(char*)(value->data.scalar.value));CHKERRQ(ierr);
      ierr = PetscOptionsInsertString(options, option);CHKERRQ(ierr);
    } else if (value->type == YAML_MAPPING_NODE) {
      PetscBool isMerge;

      /* "<<" is the merge key: don't increment the prefix */
      ierr = PetscStrcmp((char *)(key->data.scalar.value), "<<", &isMerge);CHKERRQ(ierr);
      if (!isMerge) {
        ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN,"%s_",(char *)(key->data.scalar.value));CHKERRQ(ierr);
        ierr = PetscOptionsPrefixPush(options, prefix);CHKERRQ(ierr);
      }
      ierr = PetscParseLayerYAML(options, doc, value);CHKERRQ(ierr);
      if (!isMerge) {
        ierr = PetscOptionsPrefixPop(options);CHKERRQ(ierr);
      }
    } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Unsupported yaml node tye: expected scalar or mapping");
  }
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsInsertStringYAML - Inserts YAML-formatted options into the database from a string

   Logically Collective

   Input Parameter:
+  options - options object
-  in_str - YAML-formatted string op options

   Level: intermediate

.seealso: PetscOptionsSetValue(), PetscOptionsView(), PetscOptionsHasName(), PetscOptionsGetInt(),
          PetscOptionsGetReal(), PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList(), PetscOptionsInsertFile(), PetscOptionsInsertFileYAML()
@*/
PetscErrorCode PetscOptionsInsertStringYAML(PetscOptions options,const char in_str[])
{
  PetscErrorCode ierr;
  size_t         yamlLength;
  yaml_parser_t  parser;
  yaml_document_t doc;
  yaml_node_t    *root = NULL;

  PetscFunctionBegin;
  ierr = PetscStrlen(in_str, &yamlLength);CHKERRQ(ierr);
  if (!yaml_parser_initialize(&parser)){
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_LIB,"YAML parser initialization error");
  }
  yaml_parser_set_input_string(&parser,(const unsigned char *) in_str,yamlLength);
  if (!yaml_parser_load(&parser, &doc)) {
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_LIB,"YAML parser loading error");
  }
  root = yaml_document_get_root_node(&doc);
  if (root) {
    ierr = PetscParseLayerYAML(options, &doc, root);CHKERRQ(ierr);
  }
  yaml_document_delete(&doc);
  yaml_parser_delete(&parser);
  PetscFunctionReturn(0);
}

/*@C

  PetscOptionsInsertFileYAML - Insert a YAML-formatted file in the option database

  Collective

  Input Parameter:
+   comm - the processes that will share the options (usually PETSC_COMM_WORLD)
.   file - name of file
-   require - if PETSC_TRUE will generate an error if the file does not exist

  Only a small subset of the YAML standard is implemented. Sequences are NOT supported;
  aliases and the merge key "<<" are.
  The algorithm recursively parses the yaml file, pushing and popping prefixes
  and inserting key + values pairs using PetscOptionsInsertString().

  PETSc will generate an error condition that stops the program if a YAML error
  is detected, hence the user should check that the YAML file is valid before
  supplying it, for instance at http://www.yamllint.com/ .

  Inspired by https://stackoverflow.com/a/621451

  Level: intermediate

.seealso: PetscOptionsSetValue(), PetscOptionsView(), PetscOptionsHasName(), PetscOptionsGetInt(),
          PetscOptionsGetReal(), PetscOptionsGetString(), PetscOptionsGetIntArray(), PetscOptionsBool(),
          PetscOptionsName(), PetscOptionsBegin(), PetscOptionsEnd(), PetscOptionsHead(),
          PetscOptionsStringArray(),PetscOptionsRealArray(), PetscOptionsScalar(),
          PetscOptionsBoolGroupBegin(), PetscOptionsBoolGroup(), PetscOptionsBoolGroupEnd(),
          PetscOptionsFList(), PetscOptionsEList(), PetscOptionsInsertFile(), PetscOptionsInsertStringYAML()
@*/
PetscErrorCode PetscOptionsInsertFileYAML(MPI_Comm comm,const char file[],PetscBool require)
{
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  char           fname[PETSC_MAX_PATH_LEN];
  unsigned char *optionsStr;
  int            yamlLength;
  FILE          *source;
  PetscInt       offset;
  size_t         rd;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    ierr = PetscFixFilename(file,fname);CHKERRQ(ierr);
    source = fopen(fname,"r");
    if (source) {
      fseek(source,0,SEEK_END);
      yamlLength = ftell(source);
      fseek(source,0,SEEK_SET);
      ierr = PetscMalloc1(yamlLength+1,&optionsStr);CHKERRQ(ierr);
      /* Read the content of the YAML file one char at a time; why does this read the file one byte at a time? */
      for (offset = 0; offset < yamlLength; offset++) {
        rd = fread(&(optionsStr[offset]), sizeof(unsigned char),1,source);
        if (rd != sizeof(unsigned char)) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to read entire YAML file: %s",file);
      }
      fclose(source);
      optionsStr[yamlLength] = '\0';
    } else if (require) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_FILE_OPEN,"Unable to open YAML option file %s\n",fname);
    ierr = MPI_Bcast(&yamlLength,1,MPI_INT,0,comm);CHKERRQ(ierr);
    ierr = MPI_Bcast(optionsStr,yamlLength+1,MPI_UNSIGNED_CHAR,0,comm);CHKERRQ(ierr);
  } else {
    ierr = MPI_Bcast(&yamlLength,1,MPI_INT,0,comm);CHKERRQ(ierr);
    ierr = PetscMalloc1(yamlLength+1,&optionsStr);CHKERRQ(ierr);
    ierr = MPI_Bcast(optionsStr,yamlLength+1,MPI_UNSIGNED_CHAR,0,comm);CHKERRQ(ierr);
  }
  ierr = PetscOptionsInsertStringYAML(NULL, (const char *) optionsStr);CHKERRQ(ierr);
  ierr = PetscFree(optionsStr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
