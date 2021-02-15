#include <petsc/private/petscimpl.h>        /*I  "petscsys.h"   I*/
#if defined(PETSC_HAVE_YAML)
#include <yaml.h>
#endif

static MPI_Comm petsc_yaml_comm = MPI_COMM_NULL; /* only used for parallel error handling */

PETSC_STATIC_INLINE MPI_Comm PetscYAMLGetComm(void)
{
  return PetscLikely(petsc_yaml_comm != MPI_COMM_NULL) ? petsc_yaml_comm : (petsc_yaml_comm = PETSC_COMM_SELF);
}

PETSC_STATIC_INLINE MPI_Comm PetscYAMLSetComm(MPI_Comm comm)
{
  MPI_Comm prev = PetscYAMLGetComm(); petsc_yaml_comm = comm; return prev;
}

#if defined(PETSC_HAVE_YAML)

#define MAXOPTNAME 512

#define TAG(node) ((const char *)((node)->tag))
#define STR(node) ((const char *)((node)->data.scalar.value))
#define SEQ(node) ((node)->data.sequence.items)
#define MAP(node) ((node)->data.mapping.pairs)

static PetscErrorCode PetscParseLayerYAML(PetscOptions options, yaml_document_t *doc, yaml_node_t *node)
{
  MPI_Comm         comm = PetscYAMLGetComm();
  char             name[MAXOPTNAME] = "", prefix[MAXOPTNAME] = "";
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (node->type != YAML_MAPPING_NODE) SETERRQ(comm, PETSC_ERR_SUP, "Unsupported YAML node type: expected mapping");
  for (yaml_node_pair_t *pair = MAP(node).start; pair < MAP(node).top; pair++) {
    yaml_node_t *keynode = yaml_document_get_node(doc, pair->key);
    yaml_node_t *valnode = yaml_document_get_node(doc, pair->value);
    PetscBool   isMergeKey,isDummyKey,isIncludeTag;

    if (!keynode) SETERRQ(comm, PETSC_ERR_LIB, "Corrupt YAML document");
    if (!valnode) SETERRQ(comm, PETSC_ERR_LIB, "Corrupt YAML document");
    if (keynode->type != YAML_SCALAR_NODE) SETERRQ(comm, PETSC_ERR_SUP, "Unsupported YAML node type: expected scalar");

    /* "<<" is the merge key: don't increment the prefix */
    ierr = PetscStrcmp(STR(keynode), "<<", &isMergeKey);CHKERRQ(ierr);
    if (isMergeKey) {
      if (valnode->type == YAML_SEQUENCE_NODE) {
        for (yaml_node_item_t *item = SEQ(valnode).start; item < SEQ(valnode).top; item++) {
          yaml_node_t *itemnode = yaml_document_get_node(doc, *item);
          if (!itemnode) SETERRQ(comm, PETSC_ERR_LIB, "Corrupt YAML document");
          if (itemnode->type != YAML_MAPPING_NODE) SETERRQ(comm, PETSC_ERR_SUP, "Unsupported YAML node type: expected mapping");
          ierr = PetscParseLayerYAML(options, doc, itemnode);CHKERRQ(ierr);
        }
      } else if (valnode->type == YAML_MAPPING_NODE) {
        ierr = PetscParseLayerYAML(options, doc, valnode);CHKERRQ(ierr);
      } else SETERRQ(comm, PETSC_ERR_SUP, "Unsupported YAML node type: expected sequence or mapping");
      continue; /* to next pair */
    }

    /* "$$*" are treated as dummy keys, we use them for !include tags and to define anchors */
    ierr = PetscStrbeginswith(STR(keynode), "$$", &isDummyKey);CHKERRQ(ierr);
    if (isDummyKey) {
      ierr = PetscStrendswith(TAG(valnode), "!include", &isIncludeTag);CHKERRQ(ierr);CHKERRQ(ierr);
      if (isIncludeTag) { /* TODO: add proper support relative paths */
        ierr = PetscOptionsInsertFileYAML(comm, options, STR(valnode), PETSC_TRUE);CHKERRQ(ierr);
      }
      continue; /* to next pair */
    }

    if (valnode->type == YAML_SCALAR_NODE) {
      ierr = PetscSNPrintf(name, sizeof(name), "-%s", STR(keynode));CHKERRQ(ierr);
      ierr = PetscOptionsSetValue(options, name, STR(valnode));CHKERRQ(ierr);

    } else if (valnode->type == YAML_SEQUENCE_NODE) {
      PetscSegBuffer seg;
      char           *buf, *strlist;
      PetscBool      addSep = PETSC_FALSE;

      ierr = PetscSegBufferCreate(sizeof(char), PETSC_MAX_PATH_LEN, &seg);CHKERRQ(ierr);
      for (yaml_node_item_t *item = SEQ(valnode).start; item < SEQ(valnode).top; item++) {
        yaml_node_t *itemnode = yaml_document_get_node(doc, *item);
        const char  *itemstr = NULL;
        size_t       itemlen;

        if (!itemnode) SETERRQ(comm, PETSC_ERR_LIB, "Corrupt YAML document");

        if (itemnode->type == YAML_SCALAR_NODE) {
          itemstr = STR(itemnode);

        } else if (itemnode->type == YAML_MAPPING_NODE) {
          yaml_node_pair_t *kvn = itemnode->data.mapping.pairs.start;
          yaml_node_pair_t *top = itemnode->data.mapping.pairs.top;

          if (top - kvn > 1) SETERRQ(comm, PETSC_ERR_SUP, "Unsupported YAML node value: expected a single key:value pair");
          if (top - kvn > 0) {
            yaml_node_t *kn = yaml_document_get_node(doc, kvn->key);
            yaml_node_t *vn = yaml_document_get_node(doc, kvn->value);

            if (!kn) SETERRQ(comm, PETSC_ERR_LIB, "Corrupt YAML document");
            if (!vn) SETERRQ(comm, PETSC_ERR_LIB, "Corrupt YAML document");
            if (kn->type != YAML_SCALAR_NODE) SETERRQ(comm, PETSC_ERR_SUP, "Unsupported YAML node type: expected scalar");

            ierr = PetscStrcmp(STR(kn), "<<", &isMergeKey);CHKERRQ(ierr);
            if (isMergeKey) SETERRQ(comm, PETSC_ERR_SUP, "Unsupported YAML node value: merge key '<<' not supported here");

            ierr = PetscStrbeginswith(STR(kn), "$$", &isDummyKey);CHKERRQ(ierr);
            if (isDummyKey) continue;
            itemstr = STR(kn);
          }

          ierr = PetscSNPrintf(prefix,sizeof(prefix), "%s_", STR(keynode));CHKERRQ(ierr);
          ierr = PetscOptionsPrefixPush(options, prefix);CHKERRQ(ierr);
          ierr = PetscParseLayerYAML(options, doc, itemnode);CHKERRQ(ierr);
          ierr = PetscOptionsPrefixPop(options);CHKERRQ(ierr);

        } else SETERRQ(comm, PETSC_ERR_SUP, "Unsupported YAML node type: expected scalar or mapping");

        ierr = PetscStrlen(itemstr, &itemlen);CHKERRQ(ierr);
        if (itemlen) {
          if (addSep) {
            ierr = PetscSegBufferGet(seg, 1, &buf);CHKERRQ(ierr);
            ierr = PetscArraycpy(buf, ",", 1);CHKERRQ(ierr);
          }
          ierr = PetscSegBufferGet(seg, itemlen, &buf);CHKERRQ(ierr);
          ierr = PetscArraycpy(buf, itemstr, itemlen);CHKERRQ(ierr);
          addSep = PETSC_TRUE;
        }
      }
      ierr = PetscSegBufferGet(seg, 1, &buf);CHKERRQ(ierr);
      ierr = PetscArrayzero(buf, 1);CHKERRQ(ierr);
      ierr = PetscSegBufferExtractAlloc(seg, &strlist);CHKERRQ(ierr);
      ierr = PetscSegBufferDestroy(&seg);CHKERRQ(ierr);

      ierr = PetscSNPrintf(name, sizeof(name), "-%s", STR(keynode));CHKERRQ(ierr);
      ierr = PetscOptionsSetValue(options, name, strlist);CHKERRQ(ierr);
      ierr = PetscFree(strlist);CHKERRQ(ierr);

    } else if (valnode->type == YAML_MAPPING_NODE) {
      ierr = PetscSNPrintf(prefix,sizeof(prefix), "%s_", STR(keynode));CHKERRQ(ierr);
      ierr = PetscOptionsPrefixPush(options, prefix);CHKERRQ(ierr);
      ierr = PetscParseLayerYAML(options, doc, valnode);CHKERRQ(ierr);
      ierr = PetscOptionsPrefixPop(options);CHKERRQ(ierr);

    } else SETERRQ(comm, PETSC_ERR_SUP, "Unsupported YAML node type: expected scalar, sequence or mapping");
  }
  PetscFunctionReturn(0);
}

#endif

/*@C
   PetscOptionsInsertStringYAML - Inserts YAML-formatted options into the database from a string

   Logically Collective

   Input Parameter:
+  options - options database, use NULL for default global database
-  in_str - YAML-formatted string options

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
#if defined(PETSC_HAVE_YAML)
  MPI_Comm        comm = PetscYAMLGetComm();
  yaml_parser_t   parser;
  yaml_document_t doc;
  yaml_node_t     *root;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!in_str) in_str = "";
  ierr = !yaml_parser_initialize(&parser); if (ierr) SETERRQ(comm, PETSC_ERR_LIB, "YAML parser initialization error");
  yaml_parser_set_input_string(&parser, (const unsigned char *)in_str, strlen(in_str));
  do {
    ierr = !yaml_parser_load(&parser, &doc); if (ierr) SETERRQ(comm, PETSC_ERR_LIB, "YAML parser loading error");
    root = yaml_document_get_root_node(&doc);
    if (root) {
      ierr = PetscParseLayerYAML(options, &doc, root);CHKERRQ(ierr);
    }
    yaml_document_delete(&doc);
  } while (root);
  yaml_parser_delete(&parser);
  PetscFunctionReturn(0);
#else
  MPI_Comm comm = PetscYAMLGetComm();
  (void)options; (void)in_str; /* unused */
  SETERRQ(comm, PETSC_ERR_SUP, "YAML not supported in this build.\nPlease reconfigure using --download-yaml");
#endif
}

/*@C
  PetscOptionsInsertFileYAML - Insert a YAML-formatted file in the option database

  Collective

  Input Parameter:
+   comm - the processes that will share the options (usually PETSC_COMM_WORLD)
.   options - options database, use NULL for default global database
.   file - name of file
-   require - if PETSC_TRUE will generate an error if the file does not exist

  Only a small subset of the YAML standard is implemented. Non-scalar keys are NOT supported;
  aliases and the merge key "<<" are.
  The algorithm recursively parses the yaml file, pushing and popping prefixes
  and inserting key + values pairs using PetscOptionsSetValue().

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
PetscErrorCode PetscOptionsInsertFileYAML(MPI_Comm comm,PetscOptions options,const char file[],PetscBool require)
{
  int            yamlLength = -1;
  char          *yamlString = NULL;
  MPI_Comm       prev;
  PetscMPIInt    rank;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  if (!rank) {
    char   fpath[PETSC_MAX_PATH_LEN];
    char   fname[PETSC_MAX_PATH_LEN];
    FILE  *fd;
    size_t rd;

    ierr = PetscStrreplace(PETSC_COMM_SELF, file, fpath, sizeof(fpath));CHKERRQ(ierr);
    ierr = PetscFixFilename(fpath, fname);CHKERRQ(ierr);

    fd = fopen(fname, "r");
    if (fd) {
      fseek(fd, 0, SEEK_END);
      yamlLength = (int)ftell(fd);
      fseek(fd, 0, SEEK_SET);
      if (yamlLength < 0) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Unable to query size of YAML file: %s", fname);
      ierr = PetscMalloc1(yamlLength+1, &yamlString);CHKERRQ(ierr);
      rd = fread(yamlString, 1, (size_t)yamlLength, fd);
      if (rd != (size_t)yamlLength) SETERRQ1(PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Unable to read entire YAML file: %s", fname);
      yamlString[yamlLength] = 0;
      fclose(fd);
    }
  }

  ierr = MPI_Bcast(&yamlLength, 1, MPI_INT, 0, comm);CHKERRMPI(ierr);
  if (require && yamlLength < 0) SETERRQ1(comm, PETSC_ERR_FILE_OPEN, "Unable to open YAML option file: %s\n", file);
  if (yamlLength < 0) PetscFunctionReturn(0);

  if (rank) {ierr = PetscMalloc1(yamlLength+1, &yamlString);CHKERRQ(ierr);}
  ierr = MPI_Bcast(yamlString, yamlLength+1, MPI_CHAR, 0, comm);CHKERRMPI(ierr);

  prev = PetscYAMLSetComm(comm);
  ierr = PetscOptionsInsertStringYAML(options, yamlString);CHKERRQ(ierr);
  (void) PetscYAMLSetComm(prev);

  ierr = PetscFree(yamlString);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
