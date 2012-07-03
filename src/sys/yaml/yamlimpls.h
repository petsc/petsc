#include <petscsys.h>
#include <yaml.h>

/* The option structure */
typedef struct option_s {
  /* The option name */
  char *name;
  /* The group the option is in. Defaults to default */
  char *group;
  struct {
    /* The array of strings containing the arguments */
    char **args;
    /* The number of arguments in the list */
    int count;
  } arguments;
} option_t;

/* The options_list structure */
typedef struct options_list_s {
  /* The array containing the options */
  option_t *options;
  /* The length of the options list */
  int count;
} options_list_t;

/**
 * This is a generic function to call on the proper function to populate an options list.
 *
 * The application is responsible for freeing any buffers associated with the produced
 * options_list object using the options_list_delete function.
 *
 * @param[in]   filename       A string containing the filename
 * @param[out]  options_list   An empty options_list_t object.
 *
 * returns 1 if the function succeeded, 0 on error.
 */
int
options_list_populate(char *filename, options_list_t *options_list);

/**
 * Reads a YAML file from a string and produces an options_list.
 *
 * The application is responsible for freeing any buffers assiciated with the produced
 * options_list object using the options_list_delete function.
 *
 * @param[in]   str            A string containing the YAML file. 
 * @param[out]  options_list   An empty options_list object.
 *
 * returns 1 if the function succeeded, 0 on error.
 */
int
options_list_populate_yaml(char *str, options_list_t *options_list);

/**
 * Destroy an options_list
 *
 * @param[in,out]  options_list  An options_list object.
 */
void
options_list_delete(options_list_t *options_list);

/**
 * Reads data from a file and copies it to a string.
 *
 * The application is responsible for freeing the str buffer.
 *
 * @param[in]    filename    The name of the file to be read to string.
 * @param[out]   str         The address of a NULL char* object.
 *
 * returns 1 on success, 0 on error.
 */
PetscErrorCode
file_to_string(char *filename, char **str);

/* The grouping_stack_group structure */
typedef struct grouping_stack_group_s {
  /* The name of the group */
  char *name;
  /* The event index the group starts at */
  int start;
  /* The event index the group ends at */
  int end;
} grouping_stack_group_t;

/* The grouping_stack structure */
typedef struct grouping_stack_s {
  /* The array of groups in the stack */
  grouping_stack_group_t *groups;
  /* The number of elements in the string array */
  int count;
} grouping_stack_t;

/* The alias_key_value structure */
typedef struct alias_key_value_s {
  /* The string containing the alias name */
  char *alias;
  /* The YAML event corresponding with the name */
  yaml_event_t event;
} alias_key_value_t;

/* The alias_list structure */
typedef struct alias_list_s {
  /* The length of the list */
  int count;
  /* The list itself */
  alias_key_value_t *list;
} alias_list_t;

/**
 * Generic copy constructor for a YAML event.
 *
 * @param[out]  out  An uninitialized yaml_event_t object.
 * @param[in]   in   The yaml_event_t object to copy.
 *
 * returns 1 if the function succeeded, 0 on error.
 */
int
yaml_event_initialize(yaml_event_t *out, yaml_event_t *in);

/**
 * Populates a list of alias information from parsing a yaml file.
 * This is only called on by the options_list_populate_yaml function.
 *
 * The application is responsible for freeing any buffers associated
 * with the alias_list_t object by use of the alias_list_delete() function.
 *
 * @param[in]   str        A string containing the YAML document to be read.
 * @param[out]  list       An empty alias_list_t object.
 *
 * returns 1 on success.
 */
int
alias_list_populate_yaml(char *str, alias_list_t *list);

/**
 * Destroy an alias_list_t object.
 *
 * @param[in,out]   list   An alias_list_t object.
 */
void
alias_list_delete(alias_list_t *list);
