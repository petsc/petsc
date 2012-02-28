#include "yamlimpls.h"

void options_list_delete(options_list_t *options_list) {
  int i, j;

  for(i=0; i<(*options_list).count; i++)
  {
    for(j=0; j<(*options_list).options[i].arguments.count; j++) {
      if((*options_list).options[i].arguments.args[j]) {
        free((*options_list).options[i].arguments.args[j]);
      }
    }
    if((*options_list).options[i].arguments.args) {
      free((*options_list).options[i].arguments.args);
    }
    if((*options_list).options[i].name) {
      free((*options_list).options[i].name);
    }
    if((*options_list).options[i].group) {
      free((*options_list).options[i].group);
    }
  }
  if((*options_list).options) {
    free((*options_list).options);
  }
}

int options_list_populate_yaml(char *str, options_list_t *options_list) {
  yaml_parser_t parser;
  yaml_event_t event, *events=0;
  int i, j, k, ii; /* generic counters */
  int alias_count, events_length, sequence_stack, mapping_stack, mapping_end_index;; /* named counters */
  alias_list_t list;
  grouping_stack_t grouping_stack;
  const int MAX_NESTED_GROUPS = 10;
  /* This can be edited later as needed, this is for memory allocation purposes for the grouping_stack */

  /* Initialize objects and check for errors. */
  if (!yaml_parser_initialize(&parser)) {
    fprintf(stderr, "Failed to initialize parser. (%s:%d)\n", __FILE__, __LINE__-1);
    return 0;
  }
  yaml_parser_set_input_string(&parser, (unsigned char*) str, strlen(str));

  /* Counting things for memory allocation purposes */
  if(!yaml_parser_parse(&parser, &event)) {
    fprintf(stderr, "Parser error. (%s:%d)", __FILE__, __LINE__-1);
    return 0;
  }
  i=0;
  while(event.type != YAML_STREAM_END_EVENT) {
    if(event.type == YAML_DOCUMENT_START_EVENT) {
      yaml_event_delete(&event);
      if(!yaml_parser_parse(&parser, &event)) {
        fprintf(stderr, "Parser error. (%s:%d)", __FILE__, __LINE__-1);
        return 0;
      }
      if(event.type == YAML_MAPPING_START_EVENT) {
        yaml_event_delete(&event);
        if(!yaml_parser_parse(&parser, &event)) {
          fprintf(stderr, "Parser error. (%s:%d)", __FILE__, __LINE__-1);
          return 0;
        }
        if(event.type == YAML_SCALAR_EVENT) {
          if(strcmp((char*) event.data.scalar.value, "Options") == 0
          || strcmp((char*) event.data.scalar.value, "options") == 0) {
            i=3;alias_count=0;
            while(event.type != YAML_DOCUMENT_END_EVENT) {
              yaml_event_delete(&event);
              if(!yaml_parser_parse(&parser, &event)) {
                fprintf(stderr, "Parser error. (%s:%d)", __FILE__, __LINE__-1);
                return 0;
              }
              i++;
            }
          }
        }
      }
    }
    yaml_event_delete(&event);
    if(!yaml_parser_parse(&parser, &event)) {
      fprintf(stderr, "Parser error. (%s:%d)", __FILE__, __LINE__-1);
      return 0;
    }
  }
  yaml_event_delete(&event);
  yaml_parser_delete(&parser);

  /* Populating the alias list. */
  if(!alias_list_populate_yaml(str, &list)) {
    fprintf(stderr, "error alias_list_populate_yaml (%s:%d)", __FILE__, __LINE__-1);
    return 0;
  }

  /* Allocating memory based on counts from above */
  events = (yaml_event_t*) calloc((i+1)*4, sizeof(yaml_event_t));
  /* Multiplied by four because I am not counting the alias events this needs to be worked on later */
  /* We could overallocate by a lot and realloc once we get the actual number of events in the array later */

  /* Time to load the events to an array so I can better play with them */
  yaml_parser_initialize(&parser);
  yaml_parser_set_input_string(&parser, (unsigned char*) str, strlen(str));
  if(!yaml_parser_parse(&parser, &event)) {
    fprintf(stderr, "Parser error. (%s:%d)\n", __FILE__, __LINE__-1);
    return 0;
  }
  while(event.type != YAML_STREAM_END_EVENT) {
    i=0;j=0;sequence_stack=0;
    if(event.type == YAML_DOCUMENT_START_EVENT) {
      if(!yaml_event_initialize(&events[i], &event)) {
        fprintf(stderr, "error yaml_event_initialize (%s:%d)\n",__FILE__,__LINE__-1);
        return 0;
      }
      yaml_event_delete(&event);
      if(!yaml_parser_parse(&parser, &event)) {
        fprintf(stderr, "Parser error. (%s:%d)", __FILE__, __LINE__-1);
        return 0;
      }
      i++;
      if(event.type == YAML_MAPPING_START_EVENT) {
        if(!yaml_event_initialize(&events[i], &event)) {
          fprintf(stderr, "error, yamL_event_initialize (%s:%d)\n", __FILE__, __LINE__);
          return 0;
        }
        yaml_event_delete(&event);
        if(!yaml_parser_parse(&parser, &event)) {
          fprintf(stderr, "Parser error. (%s:%d)\n", __FILE__, __LINE__-1);
          return 0;
        }
        i++;
        if(event.type == YAML_SCALAR_EVENT) {
          if(!yaml_event_initialize(&events[i], &event)) {
            fprintf(stderr, "error yaml_event_initialize (%s:%d)\n", __FILE__, __LINE__-1);
            return 0;
          }
          i++;
          if(strcmp((char*) event.data.scalar.value, "options") == 0
          || strcmp((char*) event.data.scalar.value, "Options") == 0) {
            yaml_event_delete(&event);
            if(!yaml_parser_parse(&parser, &event)) {
              fprintf(stderr, "Parser error. (%s:%d)\n", __FILE__, __LINE__-1);
              return 0;
            }
            while(event.type != YAML_DOCUMENT_END_EVENT) {
              switch(event.type) {
                case YAML_ALIAS_EVENT:
                  /* Copy all of the alias event info from the alias list */
                  for(j=0; j<list.count; j++) {
                    if(strcmp(list.list[j].alias, (char*) event.data.alias.anchor) == 0) {
                      if(!yaml_event_initialize(&events[i], &list.list[j].event)) {
                        fprintf(stderr, "error yaml_event_initialize (%s:%d)\n", __FILE__, __LINE__-1);
                        return 0;
                      }
                      i++;
                    }
                  }
                  break;
                default:
                  yaml_event_initialize(&events[i], &event);
                  i++;
                  break;
              }
              yaml_event_delete(&event);
              if(!yaml_parser_parse(&parser, &event)) {
                fprintf(stderr, "Parser error. (%s:%d)\n", __FILE__, __LINE__-1);
                return 0;
              }
              if(event.type == YAML_DOCUMENT_END_EVENT) {
                if(!yaml_event_initialize(&events[i], &event)) {
                  fprintf(stderr, "error yaml_event_initialize (%s:%d)\n", __FILE__, __LINE__-1);
                  return 0;
                }
                i++;
              }
            }
            events_length = i;
            break;
          } else {
            for(i--; i>=0; i--) {
              yaml_event_delete(&events[i]);
            }
          }
        } else {
          for(i--; i>=0; i--) {
            yaml_event_delete(&events[i]);
          }
        }
      } else {
        for(i--; i>=0; i--) {
          yaml_event_delete(&events[i]);
        }
      }
    }
    yaml_event_delete(&event);
    if(!yaml_parser_parse(&parser, &event)) {
      fprintf(stderr, "Parser error. (%s:%d)", __FILE__, __LINE__-1);
      return 0;
    }
  }
  yaml_event_delete(&event);
  yaml_parser_delete(&parser);
  alias_list_delete(&list);

  /* Making sure the last block of code ran properly and my config file was written properly */
  if(events[0].type != YAML_DOCUMENT_START_EVENT
  || events[1].type != YAML_MAPPING_START_EVENT
  || events[2].type != YAML_SCALAR_EVENT
  ||(strcmp((char*) events[2].data.scalar.value, "options") != 0
  && strcmp((char*) events[2].data.scalar.value, "Options") != 0)
  || events[3].type != YAML_SEQUENCE_START_EVENT) {
    fprintf(stderr, "Events did not load properly. (%s:%d)\n", __FILE__, __LINE__);
    return 0;
  }
  for(i=0; i<events_length; i++) {
    if(events[i].type == YAML_NO_EVENT) {
      fprintf(stderr, "Events did not load properly. (%s:%d)\n", __FILE__, __LINE__);
      return 0;
    }
  }

  /* Getting the number of options */
  j=0;
  for(i=0; i<events_length; i++) {
    if(events[i].type == YAML_MAPPING_START_EVENT
    && events[i+1].type == YAML_SCALAR_EVENT
    &&(events[i+2].type == YAML_SCALAR_EVENT
    ||(events[i+2].type == YAML_SEQUENCE_START_EVENT
    && events[i+3].type == YAML_SCALAR_EVENT))) {
      j++;
    }
  }
  (*options_list).count = j;

  /* Allocating memory for the options_list options */
  (*options_list).options = (option_t*) calloc((*options_list).count+30, sizeof(option_t));

  /* Time to populate the options_list */
  /* Start out by putting a fork in the garbage disposal */
  /* Set up the grouping stack before use */
  grouping_stack.count = 0;
  grouping_stack.groups = (grouping_stack_group_t*) calloc(MAX_NESTED_GROUPS, sizeof(grouping_stack_group_t));
  for(i=0; i<MAX_NESTED_GROUPS; i++) {
    grouping_stack.groups[i].name = 0;
    grouping_stack.groups[i].start = 0;
    grouping_stack.groups[i].end = 0;
  }
  grouping_stack.groups[0].name = (char*) calloc(8, sizeof(char));
  strcpy(grouping_stack.groups[0].name, "default");
  grouping_stack.groups[0].start = 0;
  grouping_stack.groups[0].end = 0;
  grouping_stack.count = 1;

  j=0; mapping_end_index = 0;
  for(i=3; i<events_length; i++) {
    if(grouping_stack.groups[grouping_stack.count-1].end == i) {
      if(grouping_stack.groups[grouping_stack.count-1].name) {
        free(grouping_stack.groups[grouping_stack.count-1].name);
        grouping_stack.groups[grouping_stack.count-1].name = 0;
      }
      grouping_stack.groups[grouping_stack.count-1].end = 0;
      grouping_stack.groups[grouping_stack.count-1].start = 0;
      grouping_stack.count--;
      if(grouping_stack.count == 0) {
        grouping_stack.count = 1;
        grouping_stack.groups[0].name = (char*) calloc(8, sizeof(char));
        strcpy(grouping_stack.groups[0].name, "default");
        grouping_stack.groups[0].start = 0;
        grouping_stack.groups[0].end = 0;
      }
    }
    if(events[i].type == YAML_MAPPING_START_EVENT
    && events[i+1].type == YAML_SCALAR_EVENT) {
      if(events[i+2].type == YAML_SCALAR_EVENT
      && events[i+3].type == YAML_MAPPING_END_EVENT) {
        /* We have an option with only one arg */
        ii=0;
        for(k=0; k<grouping_stack.count; k++) {
          if(grouping_stack.groups[k].name) {
            ii+=strlen(grouping_stack.groups[k].name);
          }
        }
        (*options_list).options[j].name = (char*) calloc(events[i+1].data.scalar.length+1, sizeof(char));
        strcpy((*options_list).options[j].name, (char*)events[i+1].data.scalar.value);
        (*options_list).options[j].group = (char*) calloc(ii + grouping_stack.count, sizeof(char));
        strcpy((*options_list).options[j].group, grouping_stack.groups[0].name);
        for(k=1; k<grouping_stack.count; k++) {
          strcat((*options_list).options[j].group, "_");
          strcat((*options_list).options[j].group, grouping_stack.groups[k].name);
        }
        (*options_list).options[j].arguments.count = 1;
        (*options_list).options[j].arguments.args = (char**) calloc(
        (*options_list).options[j].arguments.count+1, sizeof(char*));
        (*options_list).options[j].arguments.args[0] = (char*) calloc(
        events[i+2].data.scalar.length+1, sizeof(char));
        strcpy((*options_list).options[j].arguments.args[0], (char*) events[i+2].data.scalar.value);
        j++;i+=2;
      } else if(events[i+2].type == YAML_SEQUENCE_START_EVENT) {
        if(events[i+3].type == YAML_SCALAR_EVENT) {
          /* We have an option that has a sequence of args */
          /* First lets do what we can before performing a count of the args */
          ii=0;
          for(k=0; k<grouping_stack.count; k++) {
            if(grouping_stack.groups[k].name) {
              ii+=strlen(grouping_stack.groups[k].name);
            }
          }
          (*options_list).options[j].name = (char*) calloc(events[i+1].data.scalar.length+1, sizeof(char));
          strcpy((*options_list).options[j].name, (char*) events[i+1].data.scalar.value);
          (*options_list).options[j].group = (char*) calloc(ii + grouping_stack.count, sizeof(char));
          strcpy((*options_list).options[j].group, grouping_stack.groups[0].name);
          for(k=1; k<grouping_stack.count; k++) {
            strcat((*options_list).options[j].group, "_");
            strcat((*options_list).options[j].group, grouping_stack.groups[k].name);
          }
          k=i+2+1;
          /* 2+1 for clear thought.  i+2 is the first sequence start event, so I will start at i+2+1 */
          sequence_stack=1;
          (*options_list).options[j].arguments.count = 0;
          while(sequence_stack != 0) {
            switch(events[k].type) {
              case YAML_SEQUENCE_START_EVENT:
                sequence_stack++;
                break;
              case YAML_SEQUENCE_END_EVENT:
                sequence_stack--;
                break;
              case YAML_SCALAR_EVENT:
                if(sequence_stack == 1) {
                  (*options_list).options[j].arguments.count++;
                }
                break;
              default: break;
            }
            k++;
          }
          (*options_list).options[j].arguments.args = (char**) calloc(
          (*options_list).options[j].arguments.count+1, sizeof(char*));
          for(ii=i+2+1; ii < k; ii++) {
            if(events[ii].type == YAML_SCALAR_EVENT) {
              (*options_list).options[j].arguments.args[ii-i-2-1] = (char*) calloc(
              events[ii].data.scalar.length+1, sizeof(char));
              strcpy((*options_list).options[j].arguments.args[ii-i-2-1],
              (char*) events[ii].data.scalar.value);
            }
          }
          j++;
        } else if(events[i+3].type == YAML_MAPPING_START_EVENT) {
          /* We have a group of options coming up. */
          if(grouping_stack.count == 1 && strcmp(grouping_stack.groups[0].name, "default") == 0) {
            grouping_stack.count--;
          }
          if(grouping_stack.groups[grouping_stack.count].name) {
            free(grouping_stack.groups[grouping_stack.count].name);
            grouping_stack.groups[grouping_stack.count].name = 0;
          }
          grouping_stack.groups[grouping_stack.count].name = (char*) calloc(
          events[i+1].data.scalar.length+1, sizeof(char));
          strcpy(grouping_stack.groups[grouping_stack.count].name, (char*) events[i+1].data.scalar.value);
          grouping_stack.groups[grouping_stack.count].start = i+3;
          k=i+1;
          mapping_stack=1;
          while(mapping_stack!=0) {
            switch(events[k].type) {
              case YAML_MAPPING_START_EVENT:
                mapping_stack++;
                break;
              case YAML_MAPPING_END_EVENT:
                mapping_stack--;
                break;
              default: break;
            }
            k++;
          }
          mapping_end_index = k-1;
          grouping_stack.groups[grouping_stack.count].end = k-1;
          grouping_stack.count++;
          i+=2;
        }
      }
    }
  }

  /* Cleanup */
  for(i=0; i<MAX_NESTED_GROUPS; i++) {
    if(grouping_stack.groups[i].name) free(grouping_stack.groups[i].name);
  }
  if(grouping_stack.groups) free(grouping_stack.groups);
  for(i=0; i<events_length; i++) {
    yaml_event_delete(&events[i]);
  }
  if(events) free(events);

  return 1;
}

int yaml_event_initialize(yaml_event_t *out, yaml_event_t *in) {
  switch((*in).type) {
    case YAML_STREAM_START_EVENT:
      if(!yaml_stream_start_event_initialize(&(*out), (*in).data.stream_start.encoding)) {
        fprintf(stderr, "error yaml_stream_start_event_initialize (%s:%d)\n", __FILE__, __LINE__-1);
        return 0;
      }
      break;
    case YAML_STREAM_END_EVENT:
      if(!yaml_stream_end_event_initialize(&(*out))) {
        fprintf(stderr, "error yaml_stream_end_event_initialize (%s:%d)\n", __FILE__, __LINE__-1);
        return 0;
      }
      break;
    case YAML_DOCUMENT_START_EVENT:
      if(!yaml_document_start_event_initialize(&(*out), (*in).data.document_start.version_directive,
      (*in).data.document_start.tag_directives.start, (*in).data.document_start.tag_directives.end,
      (*in).data.document_start.implicit)) {
        fprintf(stderr, "error yaml_document_start_event_initialize (%s:%d)\n", __FILE__, __LINE__-1);
        return 0;
      }
      break;
    case YAML_DOCUMENT_END_EVENT:
      if(!yaml_document_end_event_initialize(&(*out), (*in).data.document_end.implicit)) {
        fprintf(stderr, "error yaml_document_end_event_initialize (%s:%d)\n", __FILE__, __LINE__-1);
        return 0;
      }
      break;
    case YAML_ALIAS_EVENT:
      if(!yaml_alias_event_initialize(&(*out), (*in).data.alias.anchor)) {
        fprintf(stderr, "error yaml_alias_event_initialize (%s:%d)\n", __FILE__, __LINE__-1);
        return 0;
      }
      break;
    case YAML_SCALAR_EVENT:
      if(!yaml_scalar_event_initialize(&(*out), (*in).data.scalar.anchor,
      (*in).data.scalar.tag, (*in).data.scalar.value, (*in).data.scalar.length,
      (*in).data.scalar.plain_implicit, (*in).data.scalar.quoted_implicit,
      (*in).data.scalar.style)) {
        fprintf(stderr, "error yaml_scalar_event_initialize (%s:%d)\n", __FILE__, __LINE__-1);
        return 0;
      }
      break;
    case YAML_SEQUENCE_START_EVENT:
      if(!yaml_sequence_start_event_initialize(&(*out), (*in).data.sequence_start.anchor,
      (*in).data.sequence_start.tag, (*in).data.sequence_start.implicit,
      (*in).data.sequence_start.style)) {
        fprintf(stderr, "error yaml_sequence_start_event_initialize (%s:%d)\n", __FILE__, __LINE__-1);
        return 0;
      }
      break;
    case YAML_SEQUENCE_END_EVENT:
      if(!yaml_sequence_end_event_initialize(&(*out))) {
        fprintf(stderr, "error yaml_sequence_end_event_initialize (%s:%d)\n", __FILE__, __LINE__-1);
        return 0;
      }
      break;
    case YAML_MAPPING_START_EVENT:
      if(!yaml_mapping_start_event_initialize(&(*out), (*in).data.mapping_start.anchor,
      (*in).data.mapping_start.tag, (*in).data.mapping_start.implicit,
      (*in).data.mapping_start.style)) {
        fprintf(stderr, "error yaml_mapping_start_event_initialize (%s:%d)\n", __FILE__, __LINE__-1);
        return 0;
      }
      break;
    case YAML_MAPPING_END_EVENT:
      if(!yaml_mapping_end_event_initialize(&(*out))) {
        fprintf(stderr, "error yaml_mapping_end_event_initialize (%s:%d)\n", __FILE__, __LINE__-1);
        return 0;
      }
      break;
    default:
      fprintf(stderr, "unexpected event (%s:%d)\n", __FILE__, __LINE__);
      return 0;
      break;
  }

  return 1;
}

int alias_list_populate_yaml(char *str, alias_list_t *list) {
  yaml_parser_t parser;
  yaml_event_t event, *events=0;
  int i, j, k, stacknumber, events_length;

  if(!yaml_parser_initialize(&parser)) {
    fprintf(stderr, "error initializing parser (%s:%d)\n", __FILE__, __LINE__-1);
    return 0;
  }
  yaml_parser_set_input_string(&parser, (unsigned char*) str, strlen(str));

  /* Getting count to allocate memory for the events array. */
  i=0;
  if(!yaml_parser_parse(&parser, &event)) {
    fprintf(stderr, "error yaml_parser_parse (%s:%d)\n", __FILE__, __LINE__-1);
    return 0;
  }
  while(event.type != YAML_STREAM_END_EVENT) {
    i++;
    yaml_event_delete(&event);
    if(!yaml_parser_parse(&parser, &event)) {
      fprintf(stderr, "error yaml_parser_parse (%s:%d)\n", __FILE__, __LINE__-1);
      return 0;
    }
  }
  events_length = i;
  yaml_event_delete(&event);
  yaml_parser_delete(&parser);

  /* Allocate memory for the events array */
  events = (yaml_event_t*) calloc(events_length+1, sizeof(yaml_event_t));

  /* Now to copy everything to the events array */
  yaml_parser_initialize(&parser);
  yaml_parser_set_input_string(&parser, (unsigned char*) str, strlen(str));
  for(i=0; i<events_length; i++) {
    if(!yaml_parser_parse(&parser, &event)) {
      fprintf(stderr, "error yaml_parser_parse (%s:%d)\n", __FILE__, __LINE__-1);
      return 0;
    }
    if(!yaml_event_initialize(&events[i], &event)) {
      fprintf(stderr, "error yaml_event_initialize (%s:%d)\n", __FILE__, __LINE__-1);
      return 0;
    }
    yaml_event_delete(&event);
  }
  yaml_parser_delete(&parser);

  /* Lets count so I can later allocate memory for the alias list */
  stacknumber = 0;
  (*list).count = 0;
  for(i=0; i<events_length; i++) {
    switch(events[i].type) {
      case YAML_SCALAR_EVENT:
        if(events[i].data.scalar.anchor != NULL) {
          (*list).count++;
        }
        break;
      case YAML_SEQUENCE_START_EVENT:
        if(events[i].data.sequence_start.anchor != NULL) {
          (*list).count++;
          stacknumber = 1;
          j=i;j++;
          while(stacknumber != 0) {
            switch(events[j].type) {
              case YAML_SEQUENCE_START_EVENT:
                stacknumber++;
                break;
              case YAML_SEQUENCE_END_EVENT:
                stacknumber--;
                break;
              default: break;
            }
            j++;
            (*list).count++;
          }
        }
        break;
      case YAML_MAPPING_START_EVENT:
        if(events[i].data.mapping_start.anchor != NULL) {
          (*list).count++;
          stacknumber = 1;
          j=i;j++;
          while(stacknumber != 0) {
            switch(events[j].type) {
              case YAML_MAPPING_START_EVENT:
                stacknumber++;
                break;
              case YAML_MAPPING_END_EVENT:
                stacknumber--;
                break;
              default: break;
            }
            j++;
            (*list).count++;
          }
        }
        break;
      default: break;
    }
  }

  /* Lets allocate memory for the alias list */
  (*list).list = (alias_key_value_t*) calloc((*list).count+1, sizeof(alias_key_value_t));

  /* Now to run through the same algorithm to populate the list */
  j=0;
  for(i=0; i<events_length; i++) {
    switch(events[i].type) {
      case YAML_SCALAR_EVENT:
        if(events[i].data.scalar.anchor != NULL) {
          (*list).list[j].alias = (char*) calloc(
          strlen((char*) events[i].data.scalar.anchor)+1, sizeof(char));
          strcpy((*list).list[j].alias, (char*) events[i].data.scalar.anchor);
          if(!yaml_event_initialize(&(*list).list[j].event, &events[i])) {
            fprintf(stderr, "error yaml_event_initialize (%s:%d)\n", __FILE__, __LINE__-1);
            return 0;
          }
          j++;
        }
        break;
      case YAML_SEQUENCE_START_EVENT:
        if(events[i].data.sequence_start.anchor != NULL) {
          (*list).list[j].alias = (char*) calloc(
          strlen((char*) events[i].data.sequence_start.anchor)+1, sizeof(char));
          strcpy((*list).list[j].alias, (char*) events[i].data.sequence_start.anchor);
          if(!yaml_event_initialize(&(*list).list[j].event, &events[i])) {
            fprintf(stderr, "error yaml_event_initialize (%s:%d)\n", __FILE__, __LINE__-1);
            return 0;
          }
          stacknumber = 1;
          j++;
          k=i;k++;
          while(stacknumber != 0) {
            switch(events[k].type) {
              case YAML_SEQUENCE_START_EVENT:
                stacknumber++;
                break;
              case YAML_SEQUENCE_END_EVENT:
                stacknumber--;
                break;
              default: break;
            }
            (*list).list[j].alias = (char*) calloc(strlen((*list).list[j-1].alias)+1, sizeof(char));
            strcpy((*list).list[j].alias, (*list).list[j-1].alias);
            if(!yaml_event_initialize(&(*list).list[j].event, &events[k])) {
              fprintf(stderr, "error yaml_event_initialize (%s:%d)\n", __FILE__, __LINE__-1);
              return 0;
            }
            j++;k++;
          }
        }
        break;
      case YAML_MAPPING_START_EVENT:
        if(events[i].data.mapping_start.anchor != NULL) {
          (*list).list[j].alias = (char*) calloc(
          strlen((char*) events[i].data.mapping_start.anchor)+1, sizeof(char));
          strcpy((*list).list[j].alias, (char*) events[i].data.mapping_start.anchor);
          if(!yaml_event_initialize(&(*list).list[j].event, &events[i])) {
            fprintf(stderr, "error yaml_event_initialize (%s:%d)\n", __FILE__, __LINE__-1);
            return 0;
          }
          stacknumber = 1;
          j++;
          k=i;k++;
          while(stacknumber != 0) {
            switch(events[k].type) {
              case YAML_MAPPING_START_EVENT:
                stacknumber++;
                break;
              case YAML_SEQUENCE_END_EVENT:
                stacknumber--;
                break;
              default: break;
            }
            (*list).list[j].alias = (char*) calloc(strlen((*list).list[j-1].alias)+1, sizeof(char));
            strcpy((*list).list[j].alias, (*list).list[j-1].alias);
            if(!yaml_event_initialize(&(*list).list[j].event, &events[i])) {
              fprintf(stderr, "error yaml_event_initialize(%s:%d)\n", __FILE__, __LINE__-1);
              return 0;
            }
            j++;k++;
          }
        }
        break;
      default: break;
    }
  }

  /* Cleanup */
  for(i=0; i<events_length; i++) {
    yaml_event_delete(&events[i]);
  }
  if(events) free(events);

  return 1;
}

void alias_list_delete(alias_list_t *list) {
  int i;

  for(i=0; i<(*list).count; i++) {
    if((*list).list[i].alias) free((*list).list[i].alias);
    yaml_event_delete(&(*list).list[i].event);
  }
  if((*list).list) free((*list).list);
  (*list).count = 0;
}

#undef __FUNCT__
#define __FUNCT__ "file_to_string"
PetscErrorCode file_to_string(char* filename, char** str) {
  FILE *fh;
  char *line;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if((*str) != NULL) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SYS,"argument: str is not NULL");

  fh = fopen(filename, "r");
  if(!fh) PetscFunctionReturn(1); /* Return error code , and let calling function decide about the error */

  ierr = PetscMalloc(64000*sizeof(char), &line);CHKERRQ(ierr);
  ierr = PetscMalloc(128000*sizeof(char), str);CHKERRQ(ierr);
  /* might change to dynamically allocate this at a later time */

  while(fgets(line, 64000, fh) != NULL) strcat((*str), line);

  ierr = PetscFree(line);CHKERRQ(ierr);
  if(fh) fclose(fh);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PetscOptionsInsertFile_YAML"
PetscErrorCode PetscOptionsInsertFile_YAML(MPI_Comm comm, const char file[], PetscBool require)
{
  PetscErrorCode ierr, ierr_file;
  options_list_t options_list;
  PetscMPIInt    rank,cnt=0;
  char           *vstring = 0, fname[PETSC_MAX_PATH_LEN], *ostring = 0;
  size_t         i, len;
  PetscBool match;

  PetscFunctionBegin;
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  if (!rank) {
    /* Warning: assume a maximum size for all options in a string */
    ierr = PetscMalloc(128000*sizeof(char),&vstring);CHKERRQ(ierr);
    vstring[0] = 0;
    cnt = 0;

    ierr      = PetscFixFilename(file,fname);CHKERRQ(ierr);
    ierr_file = file_to_string(fname, &ostring);
    if (ierr_file && require) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Unable to open YAML Options File %s",fname);
    if (ierr_file) PetscFunctionReturn(0);

    if (options_list_populate_yaml(ostring,&options_list)) {
      ierr = PetscInfo1(0,"Read YAML options file %s\n",file);CHKERRQ(ierr);
      for (i=0;i<options_list.count;i++) {
        if (options_list.options[i].arguments.count == 1) {
          ierr = PetscStrcasecmp(options_list.options[i].arguments.args[0], "false", &match);CHKERRQ(ierr);
          if (!match) {
            /* The current option has one argument it is not false.  Something will have to be copied */
            ierr = PetscStrcat(vstring,"-");CHKERRQ(ierr);
            ierr = PetscStrcasecmp(options_list.options[i].group, "default", &match);CHKERRQ(ierr);
            if (!match) {
              /* The current option is not in the default group. The group name and option name needs to be copied. */
              ierr = PetscStrcat(vstring,options_list.options[i].group);CHKERRQ(ierr);
              ierr = PetscStrcat(vstring,"_");CHKERRQ(ierr);
            }
            ierr = PetscStrcat(vstring,options_list.options[i].name);CHKERRQ(ierr);
            ierr = PetscStrcat(vstring," ");CHKERRQ(ierr);
            ierr = PetscStrcasecmp(options_list.options[i].arguments.args[0], "true", &match);CHKERRQ(ierr);
            if (!match) {
              /*The argument needs to be copied. */
              ierr = PetscStrcat(vstring,options_list.options[i].arguments.args[0]);CHKERRQ(ierr);
              ierr = PetscStrcat(vstring," ");CHKERRQ(ierr);
            }
          }
        } else {
          SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"Invalid number of arguments (%s: %s)",options_list.options[i].group,options_list.options[i].name);
        }
      }
      options_list_delete(&options_list);
      ierr = PetscStrlen(vstring,&len);CHKERRQ(ierr);
      cnt  = PetscMPIIntCast(len);CHKERRQ(ierr);
    } else if (require) {
      SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_USER,"Unable to process YAML Options File %s",fname);
    }
  }

  ierr = MPI_Bcast(&cnt,1,MPI_INT,0,comm);CHKERRQ(ierr);
  if (cnt) {
    if (rank) {
      ierr = PetscMalloc((cnt+1)*sizeof(char),&vstring);CHKERRQ(ierr);
    }
    ierr = MPI_Bcast(vstring,cnt,MPI_CHAR,0,comm);CHKERRQ(ierr);
    vstring[cnt] = 0;
    ierr = PetscOptionsInsertString(vstring);CHKERRQ(ierr);
  }
  ierr = PetscFree(vstring);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
