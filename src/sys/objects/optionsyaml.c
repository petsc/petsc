#define PETSC_DESIRE_FEATURE_TEST_MACROS /* for strdup() */
#include <petsc/private/petscimpl.h>     /*I  "petscsys.h"  I*/

#if defined(PETSC_HAVE_YAML)
#include <yaml.h>  /* use external LibYAML */
#else
#include <../src/sys/yaml/include/yaml.h>
#endif

static MPI_Comm petsc_yaml_comm = MPI_COMM_NULL; /* only used for parallel error handling */

static inline MPI_Comm PetscYAMLGetComm(void)
{
  return PetscLikely(petsc_yaml_comm != MPI_COMM_NULL) ? petsc_yaml_comm : (petsc_yaml_comm = PETSC_COMM_SELF);
}

static inline MPI_Comm PetscYAMLSetComm(MPI_Comm comm)
{
  MPI_Comm prev = PetscYAMLGetComm(); petsc_yaml_comm = comm; return prev;
}

#define TAG(node) ((const char *)((node)->tag))
#define STR(node) ((const char *)((node)->data.scalar.value))
#define SEQ(node) ((node)->data.sequence.items)
#define MAP(node) ((node)->data.mapping.pairs)

static PetscErrorCode PetscParseLayerYAML(PetscOptions options, yaml_document_t *doc, yaml_node_t *node)
{
  MPI_Comm         comm = PetscYAMLGetComm();
  char             name[PETSC_MAX_OPTION_NAME] = "", prefix[PETSC_MAX_OPTION_NAME] = "";
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  if (node->type == YAML_SCALAR_NODE && !STR(node)[0]) PetscFunctionReturn(0); /* empty */
  PetscAssertFalse(node->type != YAML_MAPPING_NODE,comm, PETSC_ERR_SUP, "Unsupported YAML node type: expected mapping");
  for (yaml_node_pair_t *pair = MAP(node).start; pair < MAP(node).top; pair++) {
    yaml_node_t *keynode = yaml_document_get_node(doc, pair->key);
    yaml_node_t *valnode = yaml_document_get_node(doc, pair->value);
    PetscBool   isMergeKey,isDummyKey,isIncludeTag;

    PetscAssertFalse(!keynode,comm, PETSC_ERR_LIB, "Corrupt YAML document");
    PetscAssertFalse(!valnode,comm, PETSC_ERR_LIB, "Corrupt YAML document");
    PetscAssertFalse(keynode->type != YAML_SCALAR_NODE,comm, PETSC_ERR_SUP, "Unsupported YAML node type: expected scalar");

    /* "<<" is the merge key: don't increment the prefix */
    ierr = PetscStrcmp(STR(keynode), "<<", &isMergeKey);CHKERRQ(ierr);
    if (isMergeKey) {
      if (valnode->type == YAML_SEQUENCE_NODE) {
        for (yaml_node_item_t *item = SEQ(valnode).start; item < SEQ(valnode).top; item++) {
          yaml_node_t *itemnode = yaml_document_get_node(doc, *item);
          PetscAssertFalse(!itemnode,comm, PETSC_ERR_LIB, "Corrupt YAML document");
          PetscAssertFalse(itemnode->type != YAML_MAPPING_NODE,comm, PETSC_ERR_SUP, "Unsupported YAML node type: expected mapping");
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

        PetscAssertFalse(!itemnode,comm, PETSC_ERR_LIB, "Corrupt YAML document");

        if (itemnode->type == YAML_SCALAR_NODE) {
          itemstr = STR(itemnode);

        } else if (itemnode->type == YAML_MAPPING_NODE) {
          yaml_node_pair_t *kvn = itemnode->data.mapping.pairs.start;
          yaml_node_pair_t *top = itemnode->data.mapping.pairs.top;

          PetscAssertFalse(top - kvn > 1,comm, PETSC_ERR_SUP, "Unsupported YAML node value: expected a single key:value pair");
          if (top - kvn > 0) {
            yaml_node_t *kn = yaml_document_get_node(doc, kvn->key);
            yaml_node_t *vn = yaml_document_get_node(doc, kvn->value);

            PetscAssertFalse(!kn,comm, PETSC_ERR_LIB, "Corrupt YAML document");
            PetscAssertFalse(!vn,comm, PETSC_ERR_LIB, "Corrupt YAML document");
            PetscAssertFalse(kn->type != YAML_SCALAR_NODE,comm, PETSC_ERR_SUP, "Unsupported YAML node type: expected scalar");

            ierr = PetscStrcmp(STR(kn), "<<", &isMergeKey);CHKERRQ(ierr);
            PetscAssertFalse(isMergeKey,comm, PETSC_ERR_SUP, "Unsupported YAML node value: merge key '<<' not supported here");

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

/*@C
   PetscOptionsInsertStringYAML - Inserts YAML-formatted options into the database from a string

   Logically Collective

   Input Parameters:
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
  MPI_Comm        comm = PetscYAMLGetComm();
  yaml_parser_t   parser;
  yaml_document_t doc;
  yaml_node_t     *root;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  if (!in_str) in_str = "";
  ierr = !yaml_parser_initialize(&parser); PetscAssertFalse(ierr,comm, PETSC_ERR_LIB, "YAML parser initialization error");
  yaml_parser_set_input_string(&parser, (const unsigned char *)in_str, strlen(in_str));
  do {
    ierr = !yaml_parser_load(&parser, &doc); PetscAssertFalse(ierr,comm, PETSC_ERR_LIB, "YAML parser loading error");
    root = yaml_document_get_root_node(&doc);
    if (root) {
      ierr = PetscParseLayerYAML(options, &doc, root);CHKERRQ(ierr);
    }
    yaml_document_delete(&doc);
  } while (root);
  yaml_parser_delete(&parser);
  PetscFunctionReturn(0);
}

/*@C
  PetscOptionsInsertFileYAML - Insert a YAML-formatted file in the options database

  Collective

  Input Parameters:
+   comm - the processes that will share the options (usually PETSC_COMM_WORLD)
.   options - options database, use NULL for default global database
.   file - name of file
-   require - if PETSC_TRUE will generate an error if the file does not exist

  PETSc will generate an error condition that stops the program if a YAML error
  is detected, hence the user should check that the YAML file is valid before
  supplying it, for instance at http://www.yamllint.com/ .

  Uses PetscOptionsInsertStringYAML().

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
  if (rank == 0) {
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
      PetscAssertFalse(yamlLength < 0,PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Unable to query size of YAML file: %s", fname);
      ierr = PetscMalloc1(yamlLength+1, &yamlString);CHKERRQ(ierr);
      rd = fread(yamlString, 1, (size_t)yamlLength, fd);
      PetscAssertFalse(rd != (size_t)yamlLength,PETSC_COMM_SELF, PETSC_ERR_FILE_READ, "Unable to read entire YAML file: %s", fname);
      yamlString[yamlLength] = 0;
      fclose(fd);
    }
  }

  ierr = MPI_Bcast(&yamlLength, 1, MPI_INT, 0, comm);CHKERRMPI(ierr);
  PetscAssertFalse(require && yamlLength < 0,comm, PETSC_ERR_FILE_OPEN, "Unable to open YAML option file: %s", file);
  if (yamlLength < 0) PetscFunctionReturn(0);

  if (rank) {ierr = PetscMalloc1(yamlLength+1, &yamlString);CHKERRQ(ierr);}
  ierr = MPI_Bcast(yamlString, yamlLength+1, MPI_CHAR, 0, comm);CHKERRMPI(ierr);

  prev = PetscYAMLSetComm(comm);
  ierr = PetscOptionsInsertStringYAML(options, yamlString);CHKERRQ(ierr);
  (void) PetscYAMLSetComm(prev);

  ierr = PetscFree(yamlString);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if !defined(PETSC_HAVE_YAML)

/*
#if !defined(PETSC_HAVE_STRDUP)
#define strdup(s) (char*)memcpy(malloc(strlen(s)+1),s,strlen(s)+1)
#endif
*/

/* Embed LibYAML in this compilation unit */
#include <../src/sys/yaml/src/api.c>
#include <../src/sys/yaml/src/loader.c>
#include <../src/sys/yaml/src/parser.c>
#include <../src/sys/yaml/src/reader.c>

/*
  Avoid compiler warnings like
    scanner.c, line 3181: warning: integer conversion resulted in a change of sign
                          *(string.pointer++) = '\xC2';

  Once yaml fixes them, we can remove the pragmas
*/
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#include <../src/sys/yaml/src/scanner.c>
#pragma GCC diagnostic pop

/* Silence a few unused-function warnings */
static PETSC_UNUSED void petsc_yaml_unused(void)
{
  (void)yaml_parser_scan;
  (void)yaml_document_get_node;
  (void)yaml_parser_set_encoding;
  (void)yaml_parser_set_input;
  (void)yaml_parser_set_input_file;
}

#endif
