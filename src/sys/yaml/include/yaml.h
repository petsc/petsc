/**
 * @file yaml.h
 * @brief Public interface for libyaml.
 * 
 * Include the header file with the code:
 * @code
 * #include <yaml.h>
 * @endcode
 */

#ifndef YAML_H
#define YAML_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

/**
 * @defgroup export Export Definitions
 * @{
 */

/** The public API declaration. */

#define YAML_DECLARE(type) static type

/** @} */

/**
 * @defgroup basic Basic Types
 * @{
 */

/** The character type (UTF-8 octet). */
typedef unsigned char yaml_char_t;

/** The version directive data. */
typedef struct yaml_version_directive_s {
    /** The major version number. */
    int major;
    /** The minor version number. */
    int minor;
} yaml_version_directive_t;

/** The tag directive data. */
typedef struct yaml_tag_directive_s {
    /** The tag handle. */
    yaml_char_t *handle;
    /** The tag prefix. */
    yaml_char_t *prefix;
} yaml_tag_directive_t;

/** The stream encoding. */
typedef enum yaml_encoding_e {
    /** Let the parser choose the encoding. */
    YAML_ANY_ENCODING,
    /** The default UTF-8 encoding. */
    YAML_UTF8_ENCODING,
    /** The UTF-16-LE encoding with BOM. */
    YAML_UTF16LE_ENCODING,
    /** The UTF-16-BE encoding with BOM. */
    YAML_UTF16BE_ENCODING
} yaml_encoding_t;

/** Line break types. */

typedef enum yaml_break_e {
    /** Let the parser choose the break type. */
    YAML_ANY_BREAK,
    /** Use CR for line breaks (Mac style). */
    YAML_CR_BREAK,
    /** Use LN for line breaks (Unix style). */
    YAML_LN_BREAK,
    /** Use CR LN for line breaks (DOS style). */
    YAML_CRLN_BREAK
} yaml_break_t;

/** Many bad things could happen with the parser and emitter. */
typedef enum yaml_error_type_e {
    /** No error is produced. */
    YAML_NO_ERROR,

    /** Cannot allocate or reallocate a block of memory. */
    YAML_MEMORY_ERROR,

    /** Cannot read or decode the input stream. */
    YAML_READER_ERROR,
    /** Cannot scan the input stream. */
    YAML_SCANNER_ERROR,
    /** Cannot parse the input stream. */
    YAML_PARSER_ERROR,
    /** Cannot compose a YAML document. */
    YAML_COMPOSER_ERROR,

    /** Cannot write to the output stream. */
    YAML_WRITER_ERROR,
    /** Cannot emit a YAML stream. */
    YAML_EMITTER_ERROR
} yaml_error_type_t;

/** The pointer position. */
typedef struct yaml_mark_s {
    /** The position index. */
    size_t index;

    /** The position line. */
    size_t line;

    /** The position column. */
    size_t column;
} yaml_mark_t;

/** @} */

/**
 * @defgroup styles Node Styles
 * @{
 */

/** Scalar styles. */
typedef enum yaml_scalar_style_e {
    /** Let the emitter choose the style. */
    YAML_ANY_SCALAR_STYLE,

    /** The plain scalar style. */
    YAML_PLAIN_SCALAR_STYLE,

    /** The single-quoted scalar style. */
    YAML_SINGLE_QUOTED_SCALAR_STYLE,
    /** The double-quoted scalar style. */
    YAML_DOUBLE_QUOTED_SCALAR_STYLE,

    /** The literal scalar style. */
    YAML_LITERAL_SCALAR_STYLE,
    /** The folded scalar style. */
    YAML_FOLDED_SCALAR_STYLE
} yaml_scalar_style_t;

/** Sequence styles. */
typedef enum yaml_sequence_style_e {
    /** Let the emitter choose the style. */
    YAML_ANY_SEQUENCE_STYLE,

    /** The block sequence style. */
    YAML_BLOCK_SEQUENCE_STYLE,
    /** The flow sequence style. */
    YAML_FLOW_SEQUENCE_STYLE
} yaml_sequence_style_t;

/** Mapping styles. */
typedef enum yaml_mapping_style_e {
    /** Let the emitter choose the style. */
    YAML_ANY_MAPPING_STYLE,

    /** The block mapping style. */
    YAML_BLOCK_MAPPING_STYLE,
    /** The flow mapping style. */
    YAML_FLOW_MAPPING_STYLE
/*    YAML_FLOW_SET_MAPPING_STYLE   */
} yaml_mapping_style_t;

/** @} */

/**
 * @defgroup tokens Tokens
 * @{
 */

/** Token types. */
typedef enum yaml_token_type_e {
    /** An empty token. */
    YAML_NO_TOKEN,

    /** A STREAM-START token. */
    YAML_STREAM_START_TOKEN,
    /** A STREAM-END token. */
    YAML_STREAM_END_TOKEN,

    /** A VERSION-DIRECTIVE token. */
    YAML_VERSION_DIRECTIVE_TOKEN,
    /** A TAG-DIRECTIVE token. */
    YAML_TAG_DIRECTIVE_TOKEN,
    /** A DOCUMENT-START token. */
    YAML_DOCUMENT_START_TOKEN,
    /** A DOCUMENT-END token. */
    YAML_DOCUMENT_END_TOKEN,

    /** A BLOCK-SEQUENCE-START token. */
    YAML_BLOCK_SEQUENCE_START_TOKEN,
    /** A BLOCK-MAPPING-START token. */
    YAML_BLOCK_MAPPING_START_TOKEN,
    /** A BLOCK-END token. */
    YAML_BLOCK_END_TOKEN,

    /** A FLOW-SEQUENCE-START token. */
    YAML_FLOW_SEQUENCE_START_TOKEN,
    /** A FLOW-SEQUENCE-END token. */
    YAML_FLOW_SEQUENCE_END_TOKEN,
    /** A FLOW-MAPPING-START token. */
    YAML_FLOW_MAPPING_START_TOKEN,
    /** A FLOW-MAPPING-END token. */
    YAML_FLOW_MAPPING_END_TOKEN,

    /** A BLOCK-ENTRY token. */
    YAML_BLOCK_ENTRY_TOKEN,
    /** A FLOW-ENTRY token. */
    YAML_FLOW_ENTRY_TOKEN,
    /** A KEY token. */
    YAML_KEY_TOKEN,
    /** A VALUE token. */
    YAML_VALUE_TOKEN,

    /** An ALIAS token. */
    YAML_ALIAS_TOKEN,
    /** An ANCHOR token. */
    YAML_ANCHOR_TOKEN,
    /** A TAG token. */
    YAML_TAG_TOKEN,
    /** A SCALAR token. */
    YAML_SCALAR_TOKEN
} yaml_token_type_t;

/** The token structure. */
typedef struct yaml_token_s {

    /** The token type. */
    yaml_token_type_t type;

    /** The token data. */
    union {

        /** The stream start (for @c YAML_STREAM_START_TOKEN). */
        struct {
            /** The stream encoding. */
            yaml_encoding_t encoding;
        } stream_start;

        /** The alias (for @c YAML_ALIAS_TOKEN). */
        struct {
            /** The alias value. */
            yaml_char_t *value;
        } alias;

        /** The anchor (for @c YAML_ANCHOR_TOKEN). */
        struct {
            /** The anchor value. */
            yaml_char_t *value;
        } anchor;

        /** The tag (for @c YAML_TAG_TOKEN). */
        struct {
            /** The tag handle. */
            yaml_char_t *handle;
            /** The tag suffix. */
            yaml_char_t *suffix;
        } tag;

        /** The scalar value (for @c YAML_SCALAR_TOKEN). */
        struct {
            /** The scalar value. */
            yaml_char_t *value;
            /** The length of the scalar value. */
            size_t length;
            /** The scalar style. */
            yaml_scalar_style_t style;
        } scalar;

        /** The version directive (for @c YAML_VERSION_DIRECTIVE_TOKEN). */
        struct {
            /** The major version number. */
            int major;
            /** The minor version number. */
            int minor;
        } version_directive;

        /** The tag directive (for @c YAML_TAG_DIRECTIVE_TOKEN). */
        struct {
            /** The tag handle. */
            yaml_char_t *handle;
            /** The tag prefix. */
            yaml_char_t *prefix;
        } tag_directive;

    } data;

    /** The beginning of the token. */
    yaml_mark_t start_mark;
    /** The end of the token. */
    yaml_mark_t end_mark;

} yaml_token_t;

/**
 * Free any memory allocated for a token object.
 *
 * @param[in,out]   token   A token object.
 */

YAML_DECLARE(void)
yaml_token_delete(yaml_token_t *token);

/** @} */

/**
 * @defgroup events Events
 * @{
 */

/** Event types. */
typedef enum yaml_event_type_e {
    /** An empty event. */
    YAML_NO_EVENT,

    /** A STREAM-START event. */
    YAML_STREAM_START_EVENT,
    /** A STREAM-END event. */
    YAML_STREAM_END_EVENT,

    /** A DOCUMENT-START event. */
    YAML_DOCUMENT_START_EVENT,
    /** A DOCUMENT-END event. */
    YAML_DOCUMENT_END_EVENT,

    /** An ALIAS event. */
    YAML_ALIAS_EVENT,
    /** A SCALAR event. */
    YAML_SCALAR_EVENT,

    /** A SEQUENCE-START event. */
    YAML_SEQUENCE_START_EVENT,
    /** A SEQUENCE-END event. */
    YAML_SEQUENCE_END_EVENT,

    /** A MAPPING-START event. */
    YAML_MAPPING_START_EVENT,
    /** A MAPPING-END event. */
    YAML_MAPPING_END_EVENT
} yaml_event_type_t;

/** The event structure. */
typedef struct yaml_event_s {

    /** The event type. */
    yaml_event_type_t type;

    /** The event data. */
    union {
        
        /** The stream parameters (for @c YAML_STREAM_START_EVENT). */
        struct {
            /** The document encoding. */
            yaml_encoding_t encoding;
        } stream_start;

        /** The document parameters (for @c YAML_DOCUMENT_START_EVENT). */
        struct {
            /** The version directive. */
            yaml_version_directive_t *version_directive;

            /** The list of tag directives. */
            struct {
                /** The beginning of the tag directives list. */
                yaml_tag_directive_t *start;
                /** The end of the tag directives list. */
                yaml_tag_directive_t *end;
            } tag_directives;

            /** Is the document indicator implicit? */
            int implicit;
        } document_start;

        /** The document end parameters (for @c YAML_DOCUMENT_END_EVENT). */
        struct {
            /** Is the document end indicator implicit? */
            int implicit;
        } document_end;

        /** The alias parameters (for @c YAML_ALIAS_EVENT). */
        struct {
            /** The anchor. */
            yaml_char_t *anchor;
        } alias;

        /** The scalar parameters (for @c YAML_SCALAR_EVENT). */
        struct {
            /** The anchor. */
            yaml_char_t *anchor;
            /** The tag. */
            yaml_char_t *tag;
            /** The scalar value. */
            yaml_char_t *value;
            /** The length of the scalar value. */
            size_t length;
            /** Is the tag optional for the plain style? */
            int plain_implicit;
            /** Is the tag optional for any non-plain style? */
            int quoted_implicit;
            /** The scalar style. */
            yaml_scalar_style_t style;
        } scalar;

        /** The sequence parameters (for @c YAML_SEQUENCE_START_EVENT). */
        struct {
            /** The anchor. */
            yaml_char_t *anchor;
            /** The tag. */
            yaml_char_t *tag;
            /** Is the tag optional? */
            int implicit;
            /** The sequence style. */
            yaml_sequence_style_t style;
        } sequence_start;

        /** The mapping parameters (for @c YAML_MAPPING_START_EVENT). */
        struct {
            /** The anchor. */
            yaml_char_t *anchor;
            /** The tag. */
            yaml_char_t *tag;
            /** Is the tag optional? */
            int implicit;
            /** The mapping style. */
            yaml_mapping_style_t style;
        } mapping_start;

    } data;

    /** The beginning of the event. */
    yaml_mark_t start_mark;
    /** The end of the event. */
    yaml_mark_t end_mark;

} yaml_event_t;

/** @} */

/**
 * @defgroup nodes Nodes
 * @{
 */

/** The tag @c !!null with the only possible value: @c null. */
#define YAML_NULL_TAG       "tag:yaml.org,2002:null"
/** The tag @c !!bool with the values: @c true and @c false. */
#define YAML_BOOL_TAG       "tag:yaml.org,2002:bool"
/** The tag @c !!str for string values. */
#define YAML_STR_TAG        "tag:yaml.org,2002:str"
/** The tag @c !!int for integer values. */
#define YAML_INT_TAG        "tag:yaml.org,2002:int"
/** The tag @c !!float for float values. */
#define YAML_FLOAT_TAG      "tag:yaml.org,2002:float"
/** The tag @c !!timestamp for date and time values. */
#define YAML_TIMESTAMP_TAG  "tag:yaml.org,2002:timestamp"

/** The tag @c !!seq is used to denote sequences. */
#define YAML_SEQ_TAG        "tag:yaml.org,2002:seq"
/** The tag @c !!map is used to denote mapping. */
#define YAML_MAP_TAG        "tag:yaml.org,2002:map"

/** The default scalar tag is @c !!str. */
#define YAML_DEFAULT_SCALAR_TAG     YAML_STR_TAG
/** The default sequence tag is @c !!seq. */
#define YAML_DEFAULT_SEQUENCE_TAG   YAML_SEQ_TAG
/** The default mapping tag is @c !!map. */
#define YAML_DEFAULT_MAPPING_TAG    YAML_MAP_TAG

/** Node types. */
typedef enum yaml_node_type_e {
    /** An empty node. */
    YAML_NO_NODE,

    /** A scalar node. */
    YAML_SCALAR_NODE,
    /** A sequence node. */
    YAML_SEQUENCE_NODE,
    /** A mapping node. */
    YAML_MAPPING_NODE
} yaml_node_type_t;

/** The forward definition of a document node structure. */
typedef struct yaml_node_s yaml_node_t;

/** An element of a sequence node. */
typedef int yaml_node_item_t;

/** An element of a mapping node. */
typedef struct yaml_node_pair_s {
    /** The key of the element. */
    int key;
    /** The value of the element. */
    int value;
} yaml_node_pair_t;

/** The node structure. */
struct yaml_node_s {

    /** The node type. */
    yaml_node_type_t type;

    /** The node tag. */
    yaml_char_t *tag;

    /** The node data. */
    union {
        
        /** The scalar parameters (for @c YAML_SCALAR_NODE). */
        struct {
            /** The scalar value. */
            yaml_char_t *value;
            /** The length of the scalar value. */
            size_t length;
            /** The scalar style. */
            yaml_scalar_style_t style;
        } scalar;

        /** The sequence parameters (for @c YAML_SEQUENCE_NODE). */
        struct {
            /** The stack of sequence items. */
            struct {
                /** The beginning of the stack. */
                yaml_node_item_t *start;
                /** The end of the stack. */
                yaml_node_item_t *end;
                /** The top of the stack. */
                yaml_node_item_t *top;
            } items;
            /** The sequence style. */
            yaml_sequence_style_t style;
        } sequence;

        /** The mapping parameters (for @c YAML_MAPPING_NODE). */
        struct {
            /** The stack of mapping pairs (key, value). */
            struct {
                /** The beginning of the stack. */
                yaml_node_pair_t *start;
                /** The end of the stack. */
                yaml_node_pair_t *end;
                /** The top of the stack. */
                yaml_node_pair_t *top;
            } pairs;
            /** The mapping style. */
            yaml_mapping_style_t style;
        } mapping;

    } data;

    /** The beginning of the node. */
    yaml_mark_t start_mark;
    /** The end of the node. */
    yaml_mark_t end_mark;

};

/** The document structure. */
typedef struct yaml_document_s {

    /** The document nodes. */
    struct {
        /** The beginning of the stack. */
        yaml_node_t *start;
        /** The end of the stack. */
        yaml_node_t *end;
        /** The top of the stack. */
        yaml_node_t *top;
    } nodes;

    /** The version directive. */
    yaml_version_directive_t *version_directive;

    /** The list of tag directives. */
    struct {
        /** The beginning of the tag directives list. */
        yaml_tag_directive_t *start;
        /** The end of the tag directives list. */
        yaml_tag_directive_t *end;
    } tag_directives;

    /** Is the document start indicator implicit? */
    int start_implicit;
    /** Is the document end indicator implicit? */
    int end_implicit;

    /** The beginning of the document. */
    yaml_mark_t start_mark;
    /** The end of the document. */
    yaml_mark_t end_mark;

} yaml_document_t;

/**
 * Delete a YAML document and all its nodes.
 *
 * @param[in,out]   document        A document object.
 */

YAML_DECLARE(void)
yaml_document_delete(yaml_document_t *document);

/**
 * Get a node of a YAML document.
 *
 * The pointer returned by this function is valid until any of the functions
 * modifying the documents are called.
 *
 * @param[in]       document        A document object.
 * @param[in]       index           The node id.
 *
 * @returns the node objct or @c NULL if @c node_id is out of range.
 */

YAML_DECLARE(yaml_node_t *)
yaml_document_get_node(yaml_document_t *document, int index);

/**
 * Get the root of a YAML document node.
 *
 * The root object is the first object added to the document.
 *
 * The pointer returned by this function is valid until any of the functions
 * modifying the documents are called.
 *
 * An empty document produced by the parser signifies the end of a YAML
 * stream.
 *
 * @param[in]       document        A document object.
 *
 * @returns the node object or @c NULL if the document is empty.
 */

YAML_DECLARE(yaml_node_t *)
yaml_document_get_root_node(yaml_document_t *document);

/**
 * Create a SCALAR node and attach it to the document.
 *
 * The @a style argument may be ignored by the emitter.
 *
 * @param[in,out]   document        A document object.
 * @param[in]       tag             The scalar tag.
 * @param[in]       value           The scalar value.
 * @param[in]       length          The length of the scalar value.
 * @param[in]       style           The scalar style.
 *
 * @returns the node id or @c 0 on error.
 */

/** @} */

/**
 * @defgroup parser Parser Definitions
 * @{
 */

/**
 * The prototype of a read handler.
 *
 * The read handler is called when the parser needs to read more bytes from the
 * source.  The handler should write not more than @a size bytes to the @a
 * buffer.  The number of written bytes should be set to the @a length variable.
 *
 * @param[in,out]   data        A pointer to an application data specified by
 *                              yaml_parser_set_input().
 * @param[out]      buffer      The buffer to write the data from the source.
 * @param[in]       size        The size of the buffer.
 * @param[out]      size_read   The actual number of bytes read from the source.
 *
 * @returns On success, the handler should return @c 1.  If the handler failed,
 * the returned value should be @c 0.  On EOF, the handler should set the
 * @a size_read to @c 0 and return @c 1.
 */

typedef int yaml_read_handler_t(void *data, unsigned char *buffer, size_t size,
        size_t *size_read);

/**
 * This structure holds information about a potential simple key.
 */

typedef struct yaml_simple_key_s {
    /** Is a simple key possible? */
    int possible;

    /** Is a simple key required? */
    int required;

    /** The number of the token. */
    size_t token_number;

    /** The position mark. */
    yaml_mark_t mark;
} yaml_simple_key_t;

/**
 * The states of the parser.
 */
typedef enum yaml_parser_state_e {
    /** Expect STREAM-START. */
    YAML_PARSE_STREAM_START_STATE,
    /** Expect the beginning of an implicit document. */
    YAML_PARSE_IMPLICIT_DOCUMENT_START_STATE,
    /** Expect DOCUMENT-START. */
    YAML_PARSE_DOCUMENT_START_STATE,
    /** Expect the content of a document. */
    YAML_PARSE_DOCUMENT_CONTENT_STATE,
    /** Expect DOCUMENT-END. */
    YAML_PARSE_DOCUMENT_END_STATE,

    /** Expect a block node. */
    YAML_PARSE_BLOCK_NODE_STATE,
    /** Expect a block node or indentless sequence. */
    YAML_PARSE_BLOCK_NODE_OR_INDENTLESS_SEQUENCE_STATE,
    /** Expect a flow node. */
    YAML_PARSE_FLOW_NODE_STATE,
    /** Expect the first entry of a block sequence. */
    YAML_PARSE_BLOCK_SEQUENCE_FIRST_ENTRY_STATE,
    /** Expect an entry of a block sequence. */
    YAML_PARSE_BLOCK_SEQUENCE_ENTRY_STATE,

    /** Expect an entry of an indentless sequence. */
    YAML_PARSE_INDENTLESS_SEQUENCE_ENTRY_STATE,
    /** Expect the first key of a block mapping. */
    YAML_PARSE_BLOCK_MAPPING_FIRST_KEY_STATE,
    /** Expect a block mapping key. */
    YAML_PARSE_BLOCK_MAPPING_KEY_STATE,
    /** Expect a block mapping value. */
    YAML_PARSE_BLOCK_MAPPING_VALUE_STATE,
    /** Expect the first entry of a flow sequence. */
    YAML_PARSE_FLOW_SEQUENCE_FIRST_ENTRY_STATE,

    /** Expect an entry of a flow sequence. */
    YAML_PARSE_FLOW_SEQUENCE_ENTRY_STATE,
    /** Expect a key of an ordered mapping. */
    YAML_PARSE_FLOW_SEQUENCE_ENTRY_MAPPING_KEY_STATE,
    /** Expect a value of an ordered mapping. */
    YAML_PARSE_FLOW_SEQUENCE_ENTRY_MAPPING_VALUE_STATE,
    /** Expect the and of an ordered mapping entry. */
    YAML_PARSE_FLOW_SEQUENCE_ENTRY_MAPPING_END_STATE,
    /** Expect the first key of a flow mapping. */
    YAML_PARSE_FLOW_MAPPING_FIRST_KEY_STATE,
    /** Expect a key of a flow mapping. */

    YAML_PARSE_FLOW_MAPPING_KEY_STATE,
    /** Expect a value of a flow mapping. */
    YAML_PARSE_FLOW_MAPPING_VALUE_STATE,
    /** Expect an empty value of a flow mapping. */
    YAML_PARSE_FLOW_MAPPING_EMPTY_VALUE_STATE,
    /** Expect nothing. */
    YAML_PARSE_END_STATE
} yaml_parser_state_t;

/**
 * This structure holds aliases data.
 */

typedef struct yaml_alias_data_s {
    /** The anchor. */
    yaml_char_t *anchor;
    /** The node id. */
    int index;
    /** The anchor mark. */
    yaml_mark_t mark;
} yaml_alias_data_t;

/**
 * The parser structure.
 *
 * All members are internal.  Manage the structure using the @c yaml_parser_
 * family of functions.
 */

typedef struct yaml_parser_s {

    /**
     * @name Error handling
     * @{
     */

    /** Error type. */
    yaml_error_type_t error;
    /** Error description. */
    const char *problem;
    /** The byte about which the problem occurred. */
    size_t problem_offset;
    /** The problematic value (@c -1 is none). */
    int problem_value;
    /** The problem position. */
    yaml_mark_t problem_mark;
    /** The error context. */
    const char *context;
    /** The context position. */
    yaml_mark_t context_mark;

    /**
     * @}
     */

    /**
     * @name Reader stuff
     * @{
     */

    /** Read handler. */
    yaml_read_handler_t *read_handler;

    /** A pointer for passing to the read handler. */
    void *read_handler_data;

    /** Standard (string or file) input data. */
    union {
        /** String input data. */
        struct {
            /** The string start pointer. */
            const unsigned char *start;
            /** The string end pointer. */
            const unsigned char *end;
            /** The string current position. */
            const unsigned char *current;
        } string;

        /** File input data. */
        FILE *file;
    } input;

    /** EOF flag */
    int eof;

    /** The working buffer. */
    struct {
        /** The beginning of the buffer. */
        yaml_char_t *start;
        /** The end of the buffer. */
        yaml_char_t *end;
        /** The current position of the buffer. */
        yaml_char_t *pointer;
        /** The last filled position of the buffer. */
        yaml_char_t *last;
    } buffer;

    /* The number of unread characters in the buffer. */
    size_t unread;

    /** The raw buffer. */
    struct {
        /** The beginning of the buffer. */
        unsigned char *start;
        /** The end of the buffer. */
        unsigned char *end;
        /** The current position of the buffer. */
        unsigned char *pointer;
        /** The last filled position of the buffer. */
        unsigned char *last;
    } raw_buffer;

    /** The input encoding. */
    yaml_encoding_t encoding;

    /** The offset of the current position (in bytes). */
    size_t offset;

    /** The mark of the current position. */
    yaml_mark_t mark;

    /**
     * @}
     */

    /**
     * @name Scanner stuff
     * @{
     */

    /** Have we started to scan the input stream? */
    int stream_start_produced;

    /** Have we reached the end of the input stream? */
    int stream_end_produced;

    /** The number of unclosed '[' and '{' indicators. */
    int flow_level;

    /** The tokens queue. */
    struct {
        /** The beginning of the tokens queue. */
        yaml_token_t *start;
        /** The end of the tokens queue. */
        yaml_token_t *end;
        /** The head of the tokens queue. */
        yaml_token_t *head;
        /** The tail of the tokens queue. */
        yaml_token_t *tail;
    } tokens;

    /** The number of tokens fetched from the queue. */
    size_t tokens_parsed;

    /** Does the tokens queue contain a token ready for dequeueing. */
    int token_available;

    /** The indentation levels stack. */
    struct {
        /** The beginning of the stack. */
        int *start;
        /** The end of the stack. */
        int *end;
        /** The top of the stack. */
        int *top;
    } indents;

    /** The current indentation level. */
    int indent;

    /** May a simple key occur at the current position? */
    int simple_key_allowed;

    /** The stack of simple keys. */
    struct {
        /** The beginning of the stack. */
        yaml_simple_key_t *start;
        /** The end of the stack. */
        yaml_simple_key_t *end;
        /** The top of the stack. */
        yaml_simple_key_t *top;
    } simple_keys;

    /**
     * @}
     */

    /**
     * @name Parser stuff
     * @{
     */

    /** The parser states stack. */
    struct {
        /** The beginning of the stack. */
        yaml_parser_state_t *start;
        /** The end of the stack. */
        yaml_parser_state_t *end;
        /** The top of the stack. */
        yaml_parser_state_t *top;
    } states;

    /** The current parser state. */
    yaml_parser_state_t state;

    /** The stack of marks. */
    struct {
        /** The beginning of the stack. */
        yaml_mark_t *start;
        /** The end of the stack. */
        yaml_mark_t *end;
        /** The top of the stack. */
        yaml_mark_t *top;
    } marks;

    /** The list of TAG directives. */
    struct {
        /** The beginning of the list. */
        yaml_tag_directive_t *start;
        /** The end of the list. */
        yaml_tag_directive_t *end;
        /** The top of the list. */
        yaml_tag_directive_t *top;
    } tag_directives;

    /**
     * @}
     */

    /**
     * @name Dumper stuff
     * @{
     */

    /** The alias data. */
    struct {
        /** The beginning of the list. */
        yaml_alias_data_t *start;
        /** The end of the list. */
        yaml_alias_data_t *end;
        /** The top of the list. */
        yaml_alias_data_t *top;
    } aliases;

    /** The currently parsed document. */
    yaml_document_t *document;

    /**
     * @}
     */

} yaml_parser_t;

/**
 * Initialize a parser.
 *
 * This function creates a new parser object.  An application is responsible
 * for destroying the object using the yaml_parser_delete() function.
 *
 * @param[out]      parser  An empty parser object.
 *
 * @returns @c 1 if the function succeeded, @c 0 on error.
 */

YAML_DECLARE(int)
yaml_parser_initialize(yaml_parser_t *parser);

/**
 * Destroy a parser.
 *
 * @param[in,out]   parser  A parser object.
 */

YAML_DECLARE(void)
yaml_parser_delete(yaml_parser_t *parser);

/**
 * Set a string input.
 *
 * Note that the @a input pointer must be valid while the @a parser object
 * exists.  The application is responsible for destroying @a input after
 * destroying the @a parser.
 *
 * @param[in,out]   parser  A parser object.
 * @param[in]       input   A source data.
 * @param[in]       size    The length of the source data in bytes.
 */

YAML_DECLARE(void)
yaml_parser_set_input_string(yaml_parser_t *parser,
        const unsigned char *input, size_t size);

/**
 * Set a file input.
 *
 * @a file should be a file object open for reading.  The application is
 * responsible for closing the @a file.
 *
 * @param[in,out]   parser  A parser object.
 * @param[in]       file    An open file.
 */

YAML_DECLARE(void)
yaml_parser_set_input_file(yaml_parser_t *parser, FILE *file);

/**
 * Set a generic input handler.
 *
 * @param[in,out]   parser  A parser object.
 * @param[in]       handler A read handler.
 * @param[in]       data    Any application data for passing to the read
 *                          handler.
 */

YAML_DECLARE(void)
yaml_parser_set_input(yaml_parser_t *parser,
        yaml_read_handler_t *handler, void *data);

/**
 * Set the source encoding.
 *
 * @param[in,out]   parser      A parser object.
 * @param[in]       encoding    The source encoding.
 */

YAML_DECLARE(void)
yaml_parser_set_encoding(yaml_parser_t *parser, yaml_encoding_t encoding);

/**
 * Scan the input stream and produce the next token.
 *
 * Call the function subsequently to produce a sequence of tokens corresponding
 * to the input stream.  The initial token has the type
 * @c YAML_STREAM_START_TOKEN while the ending token has the type
 * @c YAML_STREAM_END_TOKEN.
 *
 * An application is responsible for freeing any buffers associated with the
 * produced token object using the @c yaml_token_delete function.
 *
 * An application must not alternate the calls of yaml_parser_scan() with the
 * calls of yaml_parser_parse() or yaml_parser_load(). Doing this will break
 * the parser.
 *
 * @param[in,out]   parser      A parser object.
 * @param[out]      token       An empty token object.
 *
 * @returns @c 1 if the function succeeded, @c 0 on error.
 */

YAML_DECLARE(int)
yaml_parser_scan(yaml_parser_t *parser, yaml_token_t *token);

/**
 * Parse the input stream and produce the next parsing event.
 *
 * Call the function subsequently to produce a sequence of events corresponding
 * to the input stream.  The initial event has the type
 * @c YAML_STREAM_START_EVENT while the ending event has the type
 * @c YAML_STREAM_END_EVENT.
 *
 * An application is responsible for freeing any buffers associated with the
 * produced event object using the yaml_event_delete() function.
 *
 * An application must not alternate the calls of yaml_parser_parse() with the
 * calls of yaml_parser_scan() or yaml_parser_load(). Doing this will break the
 * parser.
 *
 * @param[in,out]   parser      A parser object.
 * @param[out]      event       An empty event object.
 *
 * @returns @c 1 if the function succeeded, @c 0 on error.
 */

YAML_DECLARE(int)
yaml_parser_parse(yaml_parser_t *parser, yaml_event_t *event);

/**
 * Parse the input stream and produce the next YAML document.
 *
 * Call this function subsequently to produce a sequence of documents
 * constituting the input stream.
 *
 * If the produced document has no root node, it means that the document
 * end has been reached.
 *
 * An application is responsible for freeing any data associated with the
 * produced document object using the yaml_document_delete() function.
 *
 * An application must not alternate the calls of yaml_parser_load() with the
 * calls of yaml_parser_scan() or yaml_parser_parse(). Doing this will break
 * the parser.
 *
 * @param[in,out]   parser      A parser object.
 * @param[out]      document    An empty document object.
 *
 * @returns @c 1 if the function succeeded, @c 0 on error.
 */

YAML_DECLARE(int)
yaml_parser_load(yaml_parser_t *parser, yaml_document_t *document);

/** @} */

#ifdef __cplusplus
}
#endif

#endif /* #ifndef YAML_H */

