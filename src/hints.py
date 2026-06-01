ARGUMENT_HINTS = {
    "fn_square_root": {
        "number": "The value given from the user query \
that the function will calculate the square root of."
    },
    "fn_substitute_string_with_regex": {
        "source_string": "The target string to be modified.\
 It is the text inside the quotes.",
        "regex": "The mathematical search pattern. \
'[0-9a-zA-Z]' for set of characters, or the exact word to be replaced.",
        "replacement": "The exact character or word that will replace \
the matched regex pattern. example: '*'."
    },
    "fn_read_file": {
        "path": "The absolute file path, including all directories and slashes\
 /home/user/file.txt."
    },
    "fn_format_template": {
        "template": "The template provided by the user, \
strictly including the curly braces {}. example: 'Say Hello to {name}'"
    }
}

#         "source_string": "The exact target text to be modified,\
#  strictly excluding instruction words like 'Replace' or 'Substitute'.\
#  It is the text inside the quotes.",
