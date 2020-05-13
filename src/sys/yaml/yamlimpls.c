#include <petsc/private/petscimpl.h>        /*I  "petscsys.h"   I*/
#include <yaml.h>

enum storage_flags {VAR,VAL,SEQ};     /* "Store as" switch */

static PetscErrorCode PetscParseLayerYAML(PetscOptions options, yaml_parser_t *parser,int *lvl)
{
  yaml_event_t    event;
  int             storage = VAR; /* mapping cannot start with VAL definition w/o VAR key */
  char            key[PETSC_MAX_PATH_LEN],option[PETSC_MAX_PATH_LEN],prefix[PETSC_MAX_PATH_LEN];
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = PetscSNPrintf(option,PETSC_MAX_PATH_LEN,"%s"," ");CHKERRQ(ierr);
  do {
    if(!yaml_parser_parse(parser,&event)){
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_LIB,"YAML parse error (for instance, improper indentation)");
    }
    /* Parse value either as a new leaf in the mapping */
    /*  or as a leaf value (one of them, in case it's a sequence) */
    switch (event.type) {
      case YAML_SCALAR_EVENT:
        if (storage) {
          ierr = PetscSNPrintf(option,PETSC_MAX_PATH_LEN,"-%s %s",key,(char*)event.data.scalar.value);CHKERRQ(ierr);
          ierr = PetscOptionsInsertString(options,option);CHKERRQ(ierr);
        } else {
          ierr = PetscStrncpy(key,(char*)event.data.scalar.value,event.data.scalar.length+1);CHKERRQ(ierr);
        }
        storage ^= VAL;           /* Flip VAR/VAL switch for the next event */
        yaml_event_delete(&event);
        break;
      case YAML_SEQUENCE_START_EVENT:
        /* Sequence - all the following scalars will be appended to the last_leaf */
        storage = SEQ;
        SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP ,"Unable to open YAML option file: sequences not supported");
        yaml_event_delete(&event);
        break;
      case YAML_SEQUENCE_END_EVENT:
        storage = VAR;
        yaml_event_delete(&event);
        break;
      case YAML_MAPPING_START_EVENT:
        if (*lvl > 0) {
          ierr = PetscSNPrintf(prefix,PETSC_MAX_PATH_LEN,"%s_",key);CHKERRQ(ierr);
          ierr = PetscOptionsPrefixPush(options,prefix);CHKERRQ(ierr);
        }
        (*lvl)++;
        ierr = PetscParseLayerYAML(options,parser,lvl);CHKERRQ(ierr);
        (*lvl)--;
        if (*lvl > 0) {
          ierr = PetscOptionsPrefixPop(options);CHKERRQ(ierr);
        }
        storage ^= VAL;           /* Flip VAR/VAL, w/o touching SEQ */
        yaml_event_delete(&event);
        break;
      default:
        break;
    }
  }
  while ((event.type != YAML_MAPPING_END_EVENT) && (event.type != YAML_STREAM_END_EVENT));
  PetscFunctionReturn(0);
}

/*@C
   PetscOptionsInsertStringYAML - Inserts YAML-formatted options into the database from a string

   Logically Collective

   Input Parameter:
.  in_str - YAML-formatted string op options

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
  int            lvl=0;

  PetscFunctionBegin;
  ierr = PetscStrlen(in_str, &yamlLength);CHKERRQ(ierr);
  if(!yaml_parser_initialize(&parser)){
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_LIB,"YAML parser initialization error");
  }
  yaml_parser_set_input_string(&parser,(const unsigned char *) in_str,yamlLength);
  ierr = PetscParseLayerYAML(options, &parser,&lvl);CHKERRQ(ierr);
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

  Only a small subset of the YAML standard is implemented. Sequences and alias
  are NOT supported.
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
