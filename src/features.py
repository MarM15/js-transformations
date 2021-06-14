# Copyright (C) 2021 Marvin Moog, Markus Demmel
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import os
import sys
import signal
import logging
from scipy.stats import entropy
SRC_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.append("./src/Gibberish-Detector/")
from gib_detect_py3 import detect

logger = logging.getLogger('obf_analysis')


def ast_hunter(
        ast,
        cnt_literal,
        cnt_id,
        cnt_methods,
        cnt_new_expression,
        cnt_evil_methods,
        cnt_call_expression,
        count_arguments,
        count_calls,
        count_string,
        sum_length,
        breadth,
        s,
        s2,
        count_lit,
        sum_lit_length,
        filename,
        sc,
        count_s,
        count_i,
        c,
        m,
        n,
        string_list):
    """
        Collection of features based on AST (recursive).
        -------
        Parameter:
        - ast: node
            AST which should be traversed.
        - cnt_literal: integer
            Counts literals
        - cnt_id: integer
            Counts identifer
        - cnt_methods: integer
            Counts MemberExpressions
        - cnt_new_expressions: integer
            Counts NewExpressions
        - cnt_evil_methods: integer
            Counts CallExpression with eval or setTimeout or setInterval:
        - cnt_call_expressions: integer
            Counts cCallExpressions
        -------
        Returns:
        - same as parameter
    """
    cnt_strings = cnt_literal
    cnt_identifier = cnt_id
    cnt_member_expression = cnt_methods
    cnt_new_exp = cnt_new_expression
    cnt_evil = cnt_evil_methods
    cnt_calls = cnt_call_expression
    if ast.name == "Literal":
        cnt_strings += 1
    elif ast.name == "Identifier":
        cnt_identifier += 1
        if ast.parent.name == "CallExpression" and (ast.attributes.get("name") == "eval" or ast.attributes.get(
                "name") == "setTimeout" or ast.attributes.get("name") == "setInterval"):
            cnt_evil += 1
    elif ast.name == "MemberExpression":
        cnt_member_expression += 1
    elif ast.name == "NewExpression":
        cnt_new_exp += 1
    elif ast.name == "CallExpression":
        cnt_calls += 1

    # Average length of arguments that are strings
    c1 = count_arguments
    c2 = count_calls
    c3 = count_string
    c4 = sum_length
    if ast.name == "CallExpression":
        c2 += 1
        c1 += len(ast.children) - 1
    if 'raw' in ast.attributes:
        if (ast.attributes['raw'].startswith("\'")
                or ast.attributes['raw'].startswith("\"")):
            c4 += 1
            c3 += len(ast.attributes['raw'])

    # Breadth of AST
    if len(ast.children) == 0:
        breadth += 1

    # Counts exports statements
    size = s
    try:
        if "name" in ast.attributes:
            if ast.attributes["name"] == "exports" and ast.parent is not False:
                size += len(ast.parent.parent.children[1].attributes["value"])
    except Exception as e:
        logger.debug(
            "ast_hunter for exports statements encountered:\n %s",
            e)

    # Counts define statements
    size2 = s2
    try:
        if "name" in ast.attributes:
            if (ast.attributes["name"] ==
                    "define" and ast.parent is not False):
                size2 += sub_size(ast.parent.parent, 0)
    except Exception as e:
        logger.debug(
            "ast_hunter for define statements encountered:\n %s", e)

    # Average length of identifier
    cl = count_lit
    sll = sum_lit_length
    if ast.name == "Identifier":
        try:
            key = list(
                ast.attributes.keys())[
                list(
                    ast.attributes.values()).index(
                    next(
                        iter(
                            ast.attributes.values())))]
            value = ast.attributes.get(key)
            if not key == "range":
                sll += len(str(value))
                cl += 1
        except Exception as e:
            logger.debug(
                "ast_hunter for average length of identifier encountered:\n %s",
                e)

    # Counts switchCases
    switch_count = sc
    if ast.name == "SwitchCase":
        switch_count += 1

    # Counts If-Statements
    count_statement = count_s
    count_if = count_i
    if ast.name == "IfStatement":
        count_if += 1
    if "Statement" in ast.name:
        count_statement += 1

    # Counts ternary operator, max literal and identifier size, computed member expressions
    count_cond = c
    max_size = m
    number_comp = n
    # ternary
    if ast.name == "ConditionalExpression":
        count_cond += 1
    # max literal/identifier
    if ast.name == 'Literal':
        if len(ast.attributes["raw"]) > max_size:
            max_size = len(ast.attributes["raw"])
    if ast.name == "Identifier":
        if len(ast.attributes["name"]) > max_size:
            max_size = len(ast.attributes["name"])
    # computed
    if "computed" in ast.attributes:
        if ast.attributes["computed"]:
            number_comp += 1

    # Returns array of all raw strings
    try:
        if "raw" in ast.attributes:
            if ast.attributes["raw"].startswith(
                    "\"") or ast.attributes["raw"].startswith("\'"):
                content = ast.attributes["value"]
                string_list.append(content)
    except Exception as e:
        logger.debug(
            "ast_hunter for array of all raw strings encountered:\n %s",
            e)

    # Depth of AST
    lis = []

    for child in ast.children:
        (cnt_strings,
         cnt_identifier,
         cnt_member_expression,
         cnt_new_exp,
         cnt_evil,
         cnt_calls,
         c1,
         c2,
         c3,
         c4,
         breadth,
         size,
         size2,
         cl,
         sll,
         switch_count,
         count_statement,
         count_if,
         count_cond,
         max_size,
         number_comp,
         string_list,
         de) = ast_hunter(child,
                          cnt_strings,
                          cnt_identifier,
                          cnt_member_expression,
                          cnt_new_exp,
                          cnt_evil,
                          cnt_calls,
                          c1,
                          c2,
                          c3,
                          c4,
                          breadth,
                          size,
                          size2,
                          cl,
                          sll,
                          filename,
                          switch_count,
                          count_statement,
                          count_if,
                          count_cond,
                          max_size,
                          number_comp,
                          string_list)
        lis.append(de + 1)
    return (
        cnt_strings,
        cnt_identifier,
        cnt_member_expression,
        cnt_new_exp,
        cnt_evil,
        cnt_calls,
        c1,
        c2,
        c3,
        c4,
        breadth,
        size,
        size2,
        cl,
        sll,
        switch_count,
        count_statement,
        count_if,
        count_cond,
        max_size,
        number_comp,
        string_list,
        0 if len(lis) == 0 else max(lis))


# returns: (avg chars per line, % whitespaces, overall script
# length in chars, cnt hex, cnt oct, cnt words)
# note: really only the whitespaces are counted (so no \t\r\n)
# furthermore \n\t etc. is not counted as char
# where words = words, as separated by split(); i.e. words
# separated by whitespaces
def features_based_on_lines_and_words(filename):
    """
        Calculates features based on the raw file,

        -------
        Parameter:
        - filename: string
            Path + Name of the file to be inspected
        -------
        Returns:
        - Average characters per line
        - % of whitespaces
        - Overall script length in chars
        - Count hex-characters
        - Count oct-characters
        - Count words

    """

    with open(filename) as f:
        counter_hex = 0
        counter_oct = 0
        counter_bin = 0
        counter_words = 0
        list_of_words_for_avg_word_size = []
        max_word_size = 0
        counter_uni = 0
        lines = 0
        chars = 0
        whitespaces = 0
        for line in f:
            chars += len(line)
            lines += 1
            whitespaces += line.count(' ')
            for word in line.split():
                list_of_words_for_avg_word_size.append(word)
                if len(word) >= max_word_size:
                    max_word_size = len(word)
                try:
                    if "0b" in word or "\\b" in word or "0B" in word:
                        counter_bin += 1
                    if "0x" in word or "\\x" in word or "0X" in word:
                        counter_hex += 1
                    if "0o" in word or "\\o" in word or "0O" in word:
                        counter_oct += 1
                    if "\\u" in word or "\\U" in word:
                        counter_uni += 1
                except Exception as e:
                    logger.warning(
                        "features_based_on_lines_and_words failed for %s with:\n %s",
                        filename,
                        e)

                counter_words += 1
        lines_divider = lines
        chars_divider = chars
        tmp_counter = 0
        counter_words_divider = counter_words
        avg_word_size_divider = len(list_of_words_for_avg_word_size)
        for word in list_of_words_for_avg_word_size:
            tmp_counter += len(word)

        divider = [tmp_counter, lines, chars, counter_words_divider,
                   avg_word_size_divider]  # Prevent division by zero
        for i in range(len(divider)):
            if divider[i] == 0:
                divider[i] = 1
        tmp_counter, lines, chars, counter_words_divider, avg_word_size_divider = divider

        return (
            chars /
            lines_divider,
            whitespaces /
            chars_divider,
            counter_hex /
            counter_words_divider,
            counter_oct /
            counter_words_divider,
            counter_uni /
            counter_words_divider,
            counter_words /
            lines_divider,
            counter_words,
            lines,
            chars,
            whitespaces,
            max_word_size,
            tmp_counter /
            avg_word_size_divider,
            counter_bin /
            counter_words_divider)


def number_of_comments(comments):
    """ Returns number of comments seen in JS-file """

    return len(comments)


def counts_from_files(filename, characters):
    """ Counts various characaters in raw file and entropy """
    with open(filename, "r") as f:
        content = f.read()
        tabs = content.count("\t")
        brackets_eck = content.count("[") + content.count("]")
        brackets = content.count("(") + content.count(")")
        exclamation_mark = content.count("!")
        dollar = content.count("$")
        plus = content.count("+")
    ent = entropy(bytearray(content.encode()), base=2)

    return (
        tabs,
        (brackets +
         brackets_eck +
         exclamation_mark +
         plus +
         dollar) /
        characters,
        ent)


def counts_from_tokens(tokens):
    """ Counts various kinds of tokens, like "indexOf", "charAt"... """

    tokens_only = [x["value"] for x in tokens]

    indexOf = tokens_only.count("indexOf")
    charAt = tokens_only.count("cahrAt")
    substring = tokens_only.count("substring") + tokens_only.count("substr")
    fromCharCode = tokens_only.count("fromCharCode")
    sqrt = tokens_only.count("sqrt")
    RegExp = tokens_only.count("RegExp")
    Length = tokens_only.count("length")
    Document = tokens_only.count("document")
    Math = tokens_only.count("Math")
    replace = tokens_only.count("replace")
    BitOp = tokens_only.count("&") + tokens_only.count("^") + tokens_only.count(
        "~") + tokens_only.count("<<") + tokens_only.count(">>") + tokens_only.count(">>>")
    PlusOp = tokens_only.count("+")
    MultOp = tokens_only.count("*")
    encodeURIComponent = tokens_only.count("encodeURIComponent")
    decode = tokens_only.count("decode")
    encode = tokens_only.count("encode")
    Base64 = tokens_only.count("Base64")
    charCodeAt = tokens_only.count("charCodeAt")
    unescape = tokens_only.count("unescape")
    escape = tokens_only.count("escape")
    toString = tokens_only.count("toString")
    window = tokens_only.count("window")

    return (
        indexOf,
        charAt,
        substring,
        fromCharCode,
        sqrt,
        RegExp,
        Length,
        Document,
        Math,
        replace,
        encodeURIComponent,
        decode,
        encode,
        Base64,
        charCodeAt,
        unescape,
        escape,
        toString,
        window,
        BitOp,
        PlusOp,
        MultOp)


def fun_cmw(string):
    """ Counts multiple whitespaces (used by count_mult_whitespaces). """

    test = 0
    res = -1
    multiple = False
    for a in string:
        test += 1
        if multiple and a == " ":
            res = 1
            break
        elif a == " ":
            multiple = True
        else:
            multiple = False
    string = string[test::]
    new_string = ""
    for s in string:
        if s == " ":
            new_string += s
        else:
            break

    return res, string[len(new_string)::]


def assignments_on_dict(node, last_seen_nodes, seen_values):
    """ Average assignments on dictionaries
        (no dynamic calculation, just how often there is an operation on it in the raw code). """

    if node.id in last_seen_nodes:
        return 0
    change_of_size = 0
    try:
        for child in node.data_dep_children:
            child = child.extremity
            if child.parent.parent.name == "AssignmentExpression":
                if "value" in child.parent.children[1].attributes:
                    if not child.parent.children[1].attributes["value"] in seen_values:
                        change_of_size += 1
                        seen_values[child.parent.children[1].attributes["value"]] = 0
                elif "name" in child.parent.children[1].attributes:
                    if not child.parent.children[1].attributes["name"] in seen_values:
                        change_of_size += 1
                        seen_values[child.parent.children[1].attributes["name"]] = 0
            last_seen_nodes.add(node.id)
            change_of_size += assignments_on_dict(
                child, last_seen_nodes, seen_values)
    except Exception as e:
        logger.debug("assignments_on_dict encountered:\n %s", e)
    return change_of_size


def count_mult_whitespaces(test):
    """ Counts multiple whitespaces. """

    result = 0
    string = test
    counter = 0
    while result != -1:
        result, tmp = fun_cmw(string)
        string = tmp
        if result != -1:
            counter += 1
    return counter


def array_size_at_declaration(ast, array_dict, seen_count):
    """ Calculates average array size at declaration. """

    for child in ast.children:
        if child.name == "ArrayExpression" and child.parent is not False:  # new array
            if child.parent.children[0].name != "Identifier":
                continue
            name = child.parent.children[0].attributes["name"]

            # workaround that newly defined objects with same names do not
            # overwrite their predecessor
            if name in array_dict:
                array_dict["removeme" +
                           str(9999999 -
                               seen_count) +
                           name] = (len(child.children), child.parent.children[0])
                seen_count += 1
            else:
                array_dict[name] = (len(child.children),
                                    child.parent.children[0])
        array_dict, seen_count = array_size_at_declaration(
            child, array_dict, seen_count)
    return array_dict, seen_count


def string_size_at_declaration(ast, string_dict, seen_count):
    """ Calculates average string size at declaration. """

    try:
        for child in ast.children:
            if "raw" in child.attributes:
                if child.attributes["raw"].startswith(
                        "\"") or child.attributes["raw"].startswith("\'"):
                    name = child.attributes["value"]

                    # workaround that newly defined objects with same names do
                    # not overwrite their predecessor
                    if name in string_dict:
                        string_dict["removeme" +
                                    str(9999999 -
                                        seen_count) +
                                    name] = (len(name), child.parent.children[0])
                        seen_count += 1
                    else:
                        string_dict[name] = (
                            len(name), child.parent.children[0])

            string_dict, seen_count = string_size_at_declaration(
                child, string_dict, seen_count)
    except Exception as e:
        logger.debug("string_size_at_declaration encountered:\n %s", e)

    return string_dict, seen_count


def variables_declared(ast, var_dict, seen_count):
    """ Returns a dictionary of arrays with their initial number of elements. """

    try:
        for child in ast.children:
            if child.name == "VariableDeclaration" and child.parent is not False:
                name = child.children[0].children[0].attributes["name"]

                # workaround that newly defined objects with same names do not
                # overwrite their predecessor
                if name in var_dict:
                    var_dict["removeme" +
                             str(9999999 -
                                 seen_count) +
                             name] = (0, child.children[0].children[0])
                    seen_count += 1
                else:
                    var_dict[name] = (0, child.children[0].children[0])

            var_dict, seen_count = variables_declared(
                child, var_dict, seen_count)
    except Exception as e:
        logger.debug("variables_declared encountered:\n %s", e)

    return var_dict, seen_count


def dictionarys_declared(ast, var_dict, seen_count):
    """ Creates a dictionary of dictionaries with their initial number of elements. """

    try:
        for child in ast.children:
            try:

                if (child.name == "VariableDeclaration" and child.parent is not False
                        and child.children[0].children[1].name == "ObjectExpression"):
                    seen_values = {}
                    for child_property in child.children[0].children[1].children:
                        if "name" in child_property.children[0].attributes:
                            seen_values[child_property.children[0].attributes["name"]] = 0
                        elif "value" in child_property.children[0].attributes:
                            seen_values[child_property.children[0].attributes["value"]] = 0
                    name = child.children[0].children[0].attributes["name"]

                    # workaround that newly defined objects with same names do
                    # not overwrite their predecessor
                    if name in var_dict:
                        var_dict["removeme" + str(9999999 - seen_count) + name] = (
                            len(child.children[0].children[1].children),
                            child.children[0].children[0],
                            seen_values)
                        seen_count += 1
                    else:
                        var_dict[name] = (
                            len(child.children[0].children[1].children),
                            child.children[0].children[0],
                            seen_values)
            except Exception as e:
                logger.debug(
                    "dictionarys_declared inner encountered:\n %s", e)

            var_dict, seen_count = dictionarys_declared(
                child, var_dict, seen_count)
    except Exception as e:
        logger.debug(
            "dictionarys_declared outer encountered:\n %s", e)

    return var_dict, seen_count


def avg_string_length(string_dict):
    """ Average string length. """

    acc = 0
    for i in string_dict:
        acc += string_dict[i][0]

    if len(string_dict) == 0:
        return 0
    return acc / len(string_dict)


def avg_ops_on_strings(string_dict):
    """ Average operations on strings
        (no dynamic calculation, just how often there is an operation on it in the raw code). """

    acc = 0
    for i in string_dict:
        acc += operations_on_element(string_dict[i][1], set())
    if len(string_dict) == 0:
        return 0
    return acc / len(string_dict)


def operations_on_element(node, last_seen_nodes):
    """ Average operations on strings
        (no dynamic calculation, just how often there is an operation on it in the raw code). """

    if node.id in last_seen_nodes:
        return 0
    op_count = 0

    try:
        for child in node.data_dep_children:
            child = child.extremity
            op_count += 1

            last_seen_nodes.add(node.id)
            op_count += operations_on_element(child, last_seen_nodes)
    except Exception as e:
        logger.debug("operations_on_element encountered:\n %s", e)

    return op_count


def max_array_operations_on_variable(node, last_seen_nodes):
    """ Calculates operations on variabls. """

    if node.id in last_seen_nodes:
        return 0
    change_of_size = 0
    try:
        for child in node.data_dep_children:
            child = child.extremity
            operation = child.parent.children[1].attributes["name"]
            if operation == "push":
                change_of_size += 1
            elif operation == "unshift":
                change_of_size += 1
            elif operation == "splice":
                deletions = int(
                    child.parent.parent.children[2].attributes["raw"])
                insertions = len(child.parent.parent.children) - 3
                change = insertions - deletions
                if change >= 1:
                    change_of_size += change
            last_seen_nodes.add(node.id)
            change_of_size += max_array_operations_on_variable(
                child, last_seen_nodes)
    except Exception as e:
        logger.debug(
            "max_array_operations_on_variable encountered:\n %s", e)

    return change_of_size


def array_operations_on_variable(node, last_seen_nodes):
    """ Average operations on variables
        (no dynamic calculation, just how often there is an operation on it in the raw code). """

    if node.id in last_seen_nodes:  # prevent dataflow loops
        return 0

    change_of_size = 0
    try:
        for child in node.data_dep_children:
            child = child.extremity
            operation = child.parent.children[1].attributes["name"]
            if operation == "push":
                change_of_size += 1
            elif operation == "unshift":
                change_of_size += 1
            elif operation == "shift":
                change_of_size += -1
            elif operation == "pop":
                change_of_size += -1
            elif operation == "splice":
                deletions = int(
                    child.parent.parent.children[2].attributes["raw"])
                insertions = len(child.parent.parent.children) - 3
                change_of_size += insertions - deletions
            last_seen_nodes.add(node.id)
            change_of_size += array_operations_on_variable(
                child, last_seen_nodes)
    except Exception as e:
        logger.debug(
            "array_operations_on_variable encountered:\n %s", e)
    return change_of_size


class TimeoutError(Exception):
    pass


def handler(s, f):
    raise TimeoutError


def array_hunter(array_dict):
    """ Calculates multiple features based on arrays. """

    if len(array_dict) == 0:
        return 0, 0, 0

    sum_arraysize = 0
    sum_arraysize_2 = 0
    sum_ops = 0
    signal.signal(signal.SIGALRM, handler)
    for var in array_dict:
        try:
            # Calculates array sizes (avg. the maximum size at runtime,
            # operations like "pop" do not subtract anything from the size here).
            signal.alarm(1)
            sum_arraysize += max_array_operations_on_variable(
                array_dict[var][1], set()) + array_dict[var][0]

            sum_arraysize_2 += array_operations_on_variable(
                array_dict[var][1], set()) + array_dict[var][0]

            # Calculates average operations on array.
            sum_ops += operations_on_element(array_dict[var][1], set())
            signal.alarm(0)
        except TimeoutError:
            logger.warning("TIMEOUT in array_hunter")

    return sum_arraysize / len(array_dict), sum_arraysize_2 / \
        len(array_dict), sum_ops / len(array_dict)


def var_hunter(var_dict):
    """ Calculates multiple features based on variables. """

    if (len(var_dict)) == 0:
        return 0, 0

    sum_ops = 0
    sum_unused = 0

    for var in var_dict:
        # Calculates average operations on variables. """
        ops = operations_on_element(var_dict[var][1], set())
        sum_ops += ops

        # Counts unused variables -> variables without dataflow. """
        if ops == 0:
            sum_unused += 1

    return sum_ops / len(var_dict), sum_unused / len(var_dict)


def dict_hunter(dict_dict):
    """ Calculates multiple features based on dictionaries. """

    if len(dict_dict) == 0:
        return 0, 0, 0

    sum_ops = 0
    sum_ops_2 = 0
    acc = 0
    for dic in dict_dict:
        # Calculates average size of dictionaries at declaration.
        sum_ops += dict_dict[dic][0]
        # Calculates average size of dictionaries at runtime (not dynamic).
        sum_ops_2 += dict_dict[dic][0] + \
            assignments_on_dict(dict_dict[dic][1], set(), dict_dict[dic][2])
        # Average operations on dictionaries
        # (no dynamic calculation, just how often there is an operation on it in the raw code).
        acc += operations_on_element(dict_dict[dic][1], set())

    return sum_ops / len(dict_dict), sum_ops_2 / \
        len(dict_dict), acc / len(dict_dict)


def get_variable_function_names(ast, vari):
    """ Returns list of identifier names. """

    variables = vari
    if ast.name == "Identifier":
        variables.append(ast.attributes["name"])

    for child in ast.children:
        variables = get_variable_function_names(child, variables)
    return variables


def cnt_unique_identifier(variables):
    """ Returns list of unique identifier names. """

    return len(set(variables))


def human_readable(raws, variables):
    """ Calculates average of "human readable" variables and strings, based on some rules. """

    variables += [z[0].strip("/..") for z in [x.split(" ") for x in raws]]
    variables = list(filter(None, variables))
    human = 0
    for v in variables:
        skip = False
        vowels = 0
        filt = [c.lower() for c in v if c.isalpha()]

        if len(filt) / len(v) < 0.7:  # 70% alphabetically
            continue
        for w in v.lower():
            if w == "a" or w == "e" or w == "i" or w == "o" or w == "u":
                vowels += 1
        if vowels / len(v) < 0.2 or vowels / len(v) > 0.6:
            continue
        if len(v) > 15:
            continue
        if len(v) > 2:
            for i in range(len(v) - 2):
                if v[i] == v[i + 1] and v[i + 1] == v[i + 2]:
                    skip = True
        if skip is True:
            continue
        human += 1
    if len(variables) == 0:
        return 0
    return human / len(variables)


def human_readable_markov_chain(raws, variables):
    """ "humand_readable" that uses markov-chain by https://github.com/rrenaud/Gibberish-Detector by rrenaud.
        We adjusted the Gibberish-Detector to fit our purposes. """

    variables += [z[0].strip("/..") for z in [x.split(" ") for x in raws]]
    variables = list(filter(None, variables))

    true_acc = 0
    for v in variables:
        if detect(v):
            true_acc += 1
    if len(variables) == 0:
        variables.append("PREVENT_DIVISION_BY_ZERO")

    return true_acc / len(variables)


def human_readable_markov_chain_comments(comments):
    """ "humand_readable" on comments, uses markov-chain by https://github.com/rrenaud/Gibberish-Detector by rrenaud.
        We adjusted the Gibberish-Detector to fit our purposes. """

    true_acc = 0
    for comment in comments:
        if detect(comment["value"]):
            true_acc += 1
    if len(comments) == 0:
        comments.append("PREVENT_DIVISION_BY_ZERO")
    return true_acc / len(comments)


def sub_size(ast, s):
    """ Used by define_size. """

    size = s
    try:
        for child in ast.children:
            if "value" in child.attributes:
                size += len(child.attributes["value"])
            size = sub_size(child, size)
    except Exception as e:
        logger.debug("sub_size encountered:\n %s", e)
    return size


def count_jfogs_identifier(ast):
    """ Counts how many children
    (root)Expressionstatement -> Callexpression -> FunctionExpression has. """

    try:
        return len(ast.children[0].children[0].children[0].children)
    except Exception as e:
        logger.debug("count_jfogs_identifier encountered:\n %s", e)
        return 0


def cointains_debugger_string(raws):
    """ Counts the occurrence of "debugger" in various forms. """

    for var in raws:
        if var in ("deb", "debu", "ugger", "gger", "ger"):
            return 1
    return 0


def min_detected(file, file_Len):
    """ Calculates if biggest line is ~ length of whole code. """

    try:
        with open(file, "r") as f:
            for line in f.readlines():
                if line.startswith("*") or line.startswith("/*"):  # comments
                    file_Len -= len(line)
                # tolerance has to be bigger than 70 chars to exclude short
                # oneliners
                if len(line) - 5 < file_Len < len(line) + 5 and len(line) >= 80:
                    return 1
                return 0
    except Exception as e:
        logger.debug("min_detected encountered:\n %s", e)
