# Copyright (C) 2021 Aurore Fass
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

"""
    Builds a Code Dependency Graph.
"""

import copy

import node as _node
import js_reserved
import scope as _scope
from handle_json import *
import utility_df


DECLARATIONS = ['VariableDeclaration', 'FunctionDeclaration']
EXPRESSIONS = ['AssignmentExpression', 'ArrayExpression', 'ArrowFunctionExpression',
               'AwaitExpression', 'BinaryExpression', 'CallExpression', 'ClassExpression',
               'ConditionalExpression', 'FunctionExpression', 'LogicalExpression',
               'MemberExpression', 'NewExpression', 'ObjectExpression', 'SequenceExpression',
               'TaggedTemplateExpression', 'ThisExpression', 'UnaryExpression', 'UpdateExpression',
               'YieldExpression']


"""
In the following,
    - scopes: list of Scope
        Stores the variables currently declared and where they should be referred to.
    - id_list: list
        Stores the id of the node already handled.
     - entry: int
        Indicates if we are in the global scope (1) or not (0).
If not stated otherwise,
    - node: Node
        Current node.

If not stated otherwise, the defined functions return a list of Scope.
"""


def get_pos_identifier(identifier_node, scopes):
    """ Position of identifier_node in the corresponding scope. """

    for scope_index, scope in reversed(list(enumerate(scopes))):
        # Search from local scopes to the global one, if no match found
        var_index = scope.get_pos_identifier(identifier_node)
        if var_index is not None:
            return var_index, scope_index  # Variable position, corresponding scope index
    return None, None


def get_nearest_statement(node, answer=None, fun_expr=False):
    """
        Gets the statement node nearest to node (using CF).

        -------
        Parameters:
        - answer: Node
            Such that isinstance(answer, _node.Statement) = True. Used to force taking a statement
            node parent of the nearest node (use case: boolean DF). Default: None.
        - fun_expr: bool
            Specific to FunctionExpression nodes. Default: False.

        -------
        Returns:
        - Node:
            answer, if given, otherwise the statement node nearest to node.
    """

    if answer is not None:
        return answer
    else:
        if isinstance(node, _node.Statement) or (fun_expr and node.name == 'FunctionExpression'):
            # To also get the code back from a FunctionExpression node (which is no Statement)
            return node
        else:
            if len(node.statement_dep_parents) > 1:
                logging.warning('Several statement dependencies are joining on the same node %s',
                                node.name)
            # return get_nearest_statement(node.statement_dep_parents[0].extremity)
            return get_nearest_statement(node.parent, fun_expr=fun_expr)


def set_df(scope, var_index, identifier_node):
    """
        Sets the DD from the variable in scope at position var_index, to identifier_node.

        -------
        Parameters:
        - scope: Scope
            List of variables.
        - var_index: int
            Position of the variable considered in var.
        - identifier_node: Node
            End of the DF.
    """

    if not isinstance(scope, _scope.Scope):
        logging.error('The parameter given should be typed Scope. Got %s', str(scope))
    else:
        begin_df = get_nearest_statement(scope.var_list[var_index], scope.var_if2_list[var_index])
        begin_id_df = scope.var_list[var_index]
        if isinstance(begin_df, list):
            for i, _ in enumerate(begin_df):
                begin_df[i].set_data_dependency(extremity=identifier_node)
                identifier_node.set_value(begin_df[i].value)  # Initiates new node to previous value
                identifier_node.set_code(begin_df[i].code)  # Initiates new node to previous code
        else:
            begin_id_df.set_data_dependency(extremity=identifier_node, nearest_statement=begin_df)
            identifier_node.set_value(begin_id_df.value)  # Initiates new node to previous value
            identifier_node.set_code(begin_id_df.code)  # Initiates new node to previous code


def assignment_df(identifier_node, scopes):
    """ Add DD on Identifier nodes. """

    var_index, scope_index = get_pos_identifier(identifier_node, scopes)
    if var_index is not None:  # Position of identifier_node
        if scope_index == 0:  # Global scope
            logging.debug('The global variable %s was used', identifier_node.attributes['name'])
        else:
            logging.debug('The variable %s was used', identifier_node.attributes['name'])
        # Data dependency between last time variable used and now
        set_df(scopes[scope_index], var_index, identifier_node)

    elif identifier_node.attributes['name'].lower() not in js_reserved.RESERVED_WORDS_LOWER:
        scopes[0].add_unknown_var(identifier_node)  # TODO: handle scope of unknown var


def var_decl_df(node, scopes, entry, assignt=False, obj=False, let_const=False):
    """
        Handles the variables declared.

        -------
        Parameters:
        - node: Node
            Node whose name Identifier is.
        - assignt: Bool
            False if this is a variable declaration with var/let, True if with AssignmentExpression.
            Default: False.
        - obj: Bool
            True if node is an object, False if it is a variable. Default: False.
        - let_const: Bool
            Specific scope for variables declared with let/const keyword. Default: False.
    """

    if let_const:  # Specific scope for variables declared with let/const keyword
        current_scope = scopes[-1:]
    elif len(scopes) == 1 or entry == 1\
            or (assignt and get_pos_identifier(node, scopes[1:])[0] is None):
        # Only one scope or global scope or (directly assigned and not known as a local variable)
        current_scope = scopes[:1]  # Global scope
    else:
        current_scope = scopes[1:]  # Local scope

    var_index, scope_index = get_pos_identifier(node, current_scope)

    if var_index is None:
        current_scope[-1].add_var(node)  # Add variable in the list
        if not assignt:
            logging.debug('The variable %s was declared', node.attributes['name'])
        else:
            logging.debug('The global variable %s was declared', node.attributes['name'])
        # hoisting(node, scopes)  # Hoisting only for FunctionDeclaration

    else:
        if assignt:
            if obj:  # In the case of objects, we will always keep their AST order
                logging.debug('The object %s was used and modified', node.attributes['name'])
                # Data dependency between last time object used and now
                set_df(current_scope[scope_index], var_index, node)
            else:
                logging.debug('The variable %s was modified', node.attributes['name'])
        else:
            logging.debug('The variable %s was redefined', node.attributes['name'])

        current_scope[scope_index].update_var(var_index, node)  # Update last time with current


def var_declaration_df(node, scopes, id_list, entry, let_const=False):
    """ Handles the node VariableDeclarator: 1) Element0: id, 2) Element1: init. """

    if node.name == 'VariableDeclarator':
        identifiers = search_identifiers(node.children[0], id_list, tab=[])  # Variable definition

        if node.children[0].name != 'ObjectPattern':  # Traditional variable declaration
            for decl in identifiers:
                id_list.append(decl.id)
                var_decl_df(node=decl, scopes=scopes, entry=entry, let_const=let_const)
            if not identifiers:
                logging.warning('No identifier variable found')

        else:  # Specific case for ObjectPattern
            logging.debug('The node %s is an object pattern', node.name)
            scopes = obj_pattern_scope(node.children[0], scopes=scopes, id_list=id_list)

        if len(node.children) > 1:  # Variable initialized
            scopes = build_dfg(node.children[1], scopes, id_list=id_list, entry=entry)

        elif node.children[0].name != 'ObjectPattern':  # Var (so not objPattern) not initialized
            for decl in identifiers:
                logging.debug('The variable %s was not initialized', decl.attributes['name'])

        else:  # ObjectPattern not initialized
            logging.debug('The ObjectPattern %s was not initialized', node.children[0].attributes)

        if len(node.children) > 2:
            logging.warning('I did not expect a %s node to have more than 2 children', node.name)

    return scopes


def search_identifiers(node, id_list, tab, rec=True):
    """
        Searches the Identifier nodes children of node.

        -------
        Parameters:
        - tab: list
            To store the Identifier nodes found.
        - rec: Bool
            Indicates whether to go recursively in the node or not. Default: True (i.e. recursive).

        -------
        Returns:
        - list
            Stores the Identifier nodes found.
    """

    if node.name == 'ObjectExpression':  # Only consider the object name, no properties
        pass
    elif node.name == 'Identifier':
        """
        MemberExpression can be:
        - obj.prop[.prop.prop...]: we consider only obj;
        - this.something or window.something: we consider only something.
        """
        if node.parent.name == 'MemberExpression':
            if node.parent.children[0] == node:  # current = obj, this or window
                if node.attributes['name'] == 'this' or node.attributes['name'] == 'window':
                    id_list.append(node.id)  # As window an Identifier is
                    logging.debug('%s is not the variable\'s name', node.attributes['name'])
                    prop = node.parent.children[1]
                    if prop.name == 'Identifier':
                        tab.append(prop)  # We want the something after this/window
                else:
                    tab.append(node)  # otherwise current = obj, which we store
            elif node.parent.children[0].name == 'ThisExpression':  # Parent of this=ThisExpression
                tab.append(node)  # node is actually node.parent.children[1]
            else:
                if node.parent.attributes['computed']:  # Access through a table, could be an index
                    logging.debug('The variable %s was considered', node.attributes['name'])
                    tab.append(node)
        else:
            tab.append(node)  # Otherwise this is just a variable
    else:
        if rec:
            for child in node.children:
                search_identifiers(child, id_list, tab, rec)
    return tab


def assignment_expr_df(node, scopes, id_list, entry, call_expr=False):
    """ Handles the node AssignmentExpression: 1) Element0: assignee, 2) Element1: assignt. """

    identifiers = search_identifiers(node.children[0], id_list, tab=[])
    for assignee in identifiers:
        id_list.append(assignee.id)

        # 1) To draw DD from old assignee version
        if 'operator' in assignee.parent.attributes:
            if assignee.parent.attributes['operator'] != '=':  # Could be += where assignee is used
                assignment_df(identifier_node=assignee, scopes=scopes)

        # 2) The old assignee version can be replaced by the current one
        if (assignee.parent.name == 'MemberExpression'
                and assignee.parent.children[0].name != 'ThisExpression'
                and 'window' not in assignee.parent.children[0].attributes.values())\
                or (assignee.parent.name == 'MemberExpression'
                    and assignee.parent.parent.name == 'MemberExpression'):
            # assignee is an object, we excluded window/this.var, but not window/this.obj.prop
            # logging.warning(assignee.attributes['name'])
            if assignee.parent.attributes['computed']:  # Access through a table, could be an index
                # TODO Not sure why I made a difference here
                # assignment_df(identifier_node=assignee, scopes=scopes)
                var_decl_df(node=assignee, scopes=scopes, assignt=True, obj=True, entry=entry)
            else:
                if call_expr:
                    if get_pos_identifier(assignee, scopes)[0] is not None:
                        # Only if the obj assignee already defined, avoids DF on console.log
                        var_decl_df(node=assignee, scopes=scopes, assignt=True, obj=True,
                                    entry=entry)
                else:
                    var_decl_df(node=assignee, scopes=scopes, assignt=True, obj=True, entry=entry)
        else:  # assignee is a variable
            var_decl_df(node=assignee, scopes=scopes, assignt=True, entry=entry)

    if not identifiers:
        logging.warning('No identifier assignee found')

    for i in range(1, len(node.children)):
        scopes = build_dfg(node.children[i], scopes=scopes, id_list=id_list, entry=entry)

    # if len(node.children) > 2:  # Could be a comment
    # logging.warning('I did not expect a %s node to have more than 2 children', node.name)

    return scopes


def update_expr_df(node, scopes, id_list, entry):
    """ Handles the node UpdateExpression: Element0: argument. """

    arguments = search_identifiers(node.children[0], id_list, tab=[])
    for argument in arguments:
        # Variable used, modified, used to have 2 data dependencies, one on the original variable
        # and one of the variable modified that will be used after.
        assignment_df(identifier_node=argument, scopes=scopes)
        var_decl_df(node=argument, scopes=scopes, assignt=True, entry=entry)
        assignment_df(identifier_node=argument, scopes=scopes)

    if not arguments:
        logging.warning('No identifier assignee found')


def this_df():
    """ ThisExpression. """
    # TODO: this only


def identifier_update(node, scopes, id_list, entry):
    """ Adds data flow dependency to the considered node. """

    identifiers = search_identifiers(node, id_list, rec=False, tab=[])
    # rec=False so as to not get the same Identifier multiple times by going through its family.
    for identifier in identifiers:
        if identifier.parent.name == 'CatchClause':  # As an identifier can be used as a parameter
            # Ex: catch(err) {}, err has to be defined here
            var_decl_df(node=node, scopes=scopes, entry=entry)
        else:
            assignment_df(identifier_node=identifier, scopes=scopes)


def search_function_expression(node, tab):
    """ Searches the FunctionExpression nodes descendant of node. """

    if node.name == 'FunctionExpression':
        tab.append(node)
    else:
        for child in node.children:
            search_function_expression(child, tab)
    return tab


def link_fun_expr(node):
    """ Make the link between a function expression and the variable where it may be stored. """

    fun_expr_node = node

    while node.name != 'VariableDeclarator' and node.name != 'AssignmentExpression'\
            and node.name != 'Property' and node.name != 'Program':
        if node.name == 'CallExpression' and node.children[0].name != 'FunctionExpression':
            # Not in the condition, so that f3 assigned to a: a = function f3(){}()
            # var ex = fun(function(a) {return a});
            node = node.parent  # To assign e.g. ex to an anonymous function
            break  # To avoid e.g. assigning a to ex
        node = node.parent

    if node.name == 'VariableDeclarator' or node.name == 'AssignmentExpression'\
            or node.name == 'Property':
        var = search_identifiers(node.children[0], id_list=[], tab=[])

        functions = search_function_expression(node.children[1], tab=[])

        for i, _ in enumerate(functions):
            if fun_expr_node.id == functions[i].id:
                node_nb = i  # Position of the function expression name in the function_names list
                break

        if 'node_nb' in locals():
            if len(var) != len(functions):
                logging.warning('Trying to map %s FunctionExpression nodes to %s '
                                + 'VariableDecaration nodes', str(len(functions)), str(len(var)))
            else:
                fun_expr_def = var[node_nb]  # Variable storing the function expression
                anonym = True
                for child in fun_expr_node.children:
                    if child.body == 'id':
                        fun_expr_def.set_value(child)
                        logging.debug('The variable %s refers to the function expression %s',
                                      fun_expr_def.attributes['name'], child.attributes['name'])
                        anonym = False
                    elif anonym:
                        fun_expr_def.set_value('Anonymous function')
                        fun_expr_def.set_code(fun_expr_node)
                        logging.debug('The variable %s refers to an anonymous function ',
                                      fun_expr_def.attributes['name'])
                        anonym = False
                return fun_expr_def  # Variable referring to the function expression.
    return None


def hoisting(node, scopes):
    """ Checks if unknown variables are in fact function names which were hoisted. """

    for scope in scopes:
        unknown_var_copy = copy.copy(scope.unknown_var)
        for unknown in unknown_var_copy:
            if node.attributes['name'] == unknown.attributes['name']:
                logging.debug('Using hoisting, the function %s was first used, then defined',
                              node.attributes['name'])
                node.set_data_dependency(extremity=unknown)
                scope.remove_unknown_var(unknown)


def function_scope(node, scopes, id_list, fun_expr):
    """
        Function scope.

        -------
        Parameters:
        - fun_expr: bool
            Indicates if we handle a function declaration or expression. In the expression case,
            the function cannot be called from an outer scope.
    """

    scopes.append(_scope.Scope('Function'))  # Added function scope

    for child in node.children:
        if child.body == 'id' or child.body == 'params':
            identifiers = search_identifiers(child, id_list, tab=[])
            for param in identifiers:
                id_list.append(param.id)
                if child.body == 'id' and not fun_expr:
                    # Stores the function name, so that it can be used in the upper scope
                    var_decl_df(node=param, scopes=scopes[:-1], entry=0)
                    hoisting(param, scopes)
                else:
                    var_decl_df(node=param, scopes=scopes, entry=0)

        else:
            scopes = build_dfg(child, scopes=scopes, id_list=id_list, entry=0)

    if fun_expr:
        link_fun_expr(node)

    let_const_scope(node, scopes)  # Limit scope when going out of the block
    scopes.pop()  # Variables declared before entering the function + function name

    return scopes


def obj_expr_scope(node, scopes, id_list):
    """ ObjectExpression scope. """

    scopes.append(_scope.Scope('ObjectExpresion'))  # Added object scope

    for prop in node.children:
        for child in prop.children:
            if child.body == 'key':
                identifiers = search_identifiers(child, id_list, tab=[])
                for param in identifiers:
                    id_list.append(param.id)
                    var_decl_df(node=param, scopes=scopes, entry=0)
                    hoisting(param, scopes)

            else:
                scopes = build_dfg(child, scopes=scopes, id_list=id_list, entry=0)

    let_const_scope(node, scopes)  # Limit scope when going out of the block
    scopes.pop()  # Back to the initial scope when we are not in the object anymore

    return scopes


def obj_pattern_scope(node, scopes, id_list):
    """ ObjectPattern scope. """

    for prop in node.children:
        for child in prop.children:
            if child.body == 'value':  # Actual property name is somewhere here
                if not isinstance(child, _node.Identifier):
                    scopes = build_dfg(child, scopes=scopes, id_list=id_list, entry=0)
                else:  # Actual property name, considered as a variable
                    id_list.append(child.id)
                    var_decl_df(node=child, scopes=scopes, entry=0)

            elif child.body == 'key':  # Key, but very local to the object, not a variable
                pass
            else:
                logging.warning('The node %s had unexpected properties %s on %s', node.name,
                                child.body, child.name)

    let_const_scope(node, scopes)  # Limit scope when going out of the block

    return scopes


def get_var_branch(node_list, scopes, id_list, entry, scope_name):
    """
        Statement scope for boolean conditions.

        -------
        Parameters:
        - node_list: list of Nodes
            Current nodes to be handled.

        -------
        Returns:
        - initial_scope, and local_scope and global_scope from the considered branch
    """

    scopes.append(_scope.Scope(scope_name))  # scopes modified for the branch
    global_scope = scopes[0].copy_scope()

    for boolean_node in node_list:
        scopes = build_dfg(boolean_node, scopes=scopes, id_list=id_list, entry=entry)

    local_scope_cf = scopes.pop()
    global_scope_cf = scopes.pop(0)
    scopes.insert(0, global_scope)

    return scopes, local_scope_cf, global_scope_cf


def merge_var_boolean_cf(current_scope, scope_true, scope_false):
    """
        Merges in scope_true the variables declared in a true and false branches.

        -------
        Parameters:
        - current_scope: Scope
            Stores the variables declared before entering any conditions and where they should be
            referred to.
        - scope_true: Scope
            Stores the variables currently declared if cond = true and where they should be
            referred to.
        - scope_false: Scope
            Stores the variables currently declared if cond = false and where they should be
            referred to.

        -------
        Returns:
        - scope_true
    """

    for node_false in scope_false.var_list:
        if not any(node_false.attributes['name'] == node_true.attributes['name']
                   for node_true in scope_true.var_list):
            logging.debug('The variable %s was added to the list', node_false.attributes['name'])
            scope_true.add_var(node_false)

        for node_true in scope_true.var_list:
            if node_false.attributes['name'] == node_true.attributes['name']\
                    and node_false.id != node_true.id:  # The var was modified in >=1 branch
                var_index = scope_true.get_pos_identifier(node_true)
                if any(node_true.id == node.id for node in current_scope.var_list):
                    logging.debug('The variable %s has been modified in the branch False',
                                  node_false.attributes['name'])
                    scope_true.update_var(var_index, node_false)
                elif any(node_false.id == node.id for node in current_scope.var_list):
                    logging.debug('The variable %s has been modified in the branch True',
                                  node_true.attributes['name'])
                    # Already handled, as we work on var_list_true
                else:  # Both were modified, we refer to the nearest common statement
                    logging.debug('The variable %s has been modified in the branches True and '
                                  + 'False', node_false.attributes['name'])
                    scope_true.update_var_if2(var_index, [node_true, node_false])

    return scope_true


def handle_several_branches(todo_true, todo_false, scopes, id_list, entry):
    """
        Statement scope.

        -------
        Parameters:
        - todo_true: list of Node
            From the True branch.
        - todo_false: list of Node
            From the False branch.
    """

    if todo_true or todo_false:
        scopes, local_scope_true, global_scope_true = get_var_branch(todo_true, scopes=scopes,
                                                                     id_list=id_list, entry=entry,
                                                                     scope_name='Branch_true')

        scopes, local_scope_false, global_scope_false = get_var_branch(todo_false, scopes=scopes,
                                                                       id_list=id_list, entry=entry,
                                                                       scope_name='Branch_false')

        if not global_scope_true.is_equal(global_scope_false):  # Here we have
            global_scope = merge_var_boolean_cf(scopes[0], global_scope_true, global_scope_false)
            scopes.pop(0)
            scopes.insert(0, global_scope)

        if not local_scope_true.is_equal(local_scope_false):  # Here we have
            local_scope = merge_var_boolean_cf(scopes[-1], local_scope_true, local_scope_false)
            scopes.pop()
            scopes.append(local_scope)

        # Finally scopes contains all variables defined in the true + false branches

    return scopes


def statement_scope(node, scopes, id_list, entry):
    """ Statement scope. """

    todo_true = []
    todo_false = []

    # Statements that do belong after one another
    for child_statement_dep in node.statement_dep_children:
        child_statement = child_statement_dep.extremity
        logging.debug('The node %s has a statement dependency', child_statement.name)
        scopes = build_dfg(child_statement, scopes=scopes, id_list=id_list, entry=entry)

    for child_cf_dep in node.control_dep_children:  # Control flow statements
        child_cf = child_cf_dep.extremity
        if isinstance(child_cf_dep.label, bool):  # Several branches according to the cond
            logging.debug('The node %s has a boolean CF dependency', child_cf.name)
            if child_cf_dep.label:
                todo_true.append(child_cf)  # SwitchCase: several True possible
            else:
                todo_false.append(child_cf)

        else:  # Epsilon statements
            logging.debug('The node %s has an epsilon CF dependency', child_cf.name)
            scopes = build_dfg(child_cf, scopes=scopes, id_list=id_list, entry=entry)

    # Separate variables if separate true/false branches
    scopes = handle_several_branches(todo_true=todo_true, todo_false=todo_false, scopes=scopes,
                                     id_list=id_list, entry=entry)

    let_const_scope(node, scopes)  # Limit scope when going out of the block

    return scopes


def let_const_scope(node, scopes):
    """ Pops scope specific to variables defined with let or const. """

    if len(scopes) > 1 and scopes[-1].name == "let_const" + str(node.id):
        scopes.pop()
    elif len(scopes) > 2 and "Branch" in scopes[-1].name\
            and scopes[-2].name == "let_const" + str(node.id):  # As special scope for True branches
        scopes.pop(-2)


def build_df_variable_declaration(node, scopes, id_list, entry):
    """ VariableDeclaration data dependencies. """

    logging.debug('The node %s is a variable declaration', node.name)

    let_const = False
    if node.attributes['kind'] != 'var':  # let or const
        let_const = True
        let_const_scope_name = 'let_const' + str(node.parent.id)
        if scopes[-1].name != let_const_scope_name:  # New block scope if not already defined
            scopes.append(_scope.Scope(let_const_scope_name))

    for child in node.children:
        scopes = var_declaration_df(child, scopes=scopes, id_list=id_list, entry=entry,
                                    let_const=let_const)

    return scopes


def build_df_assignment(node, scopes, id_list, entry):
    """ AssignmentExpression data dependencies. """

    logging.debug('The node %s is an assignment expression', node.name)
    scopes = assignment_expr_df(node, scopes=scopes, id_list=id_list, entry=entry)
    return scopes


def build_df_call_expr(node, scopes, id_list, entry):
    """ CallExpression on object data dependencies. """

    logging.debug('The node %s is a call expression on an object', node.name)
    scopes = assignment_expr_df(node, scopes=scopes, id_list=id_list, entry=entry, call_expr=True)
    return scopes


def build_df_update(node, scopes, id_list, entry):
    """ UpdateExpression data dependencies. """

    logging.debug('The node %s is an update expression', node.name)
    update_expr_df(node, scopes=scopes, id_list=id_list, entry=entry)


def build_df_function(node, scopes, id_list, fun_expr=False):
    """ FunctionDeclaration and FunctionExpression data dependencies. """

    logging.debug('The node %s is a function declaration', node.name)
    return function_scope(node=node, scopes=scopes, id_list=id_list, fun_expr=fun_expr)


def build_df_statement(node, scopes, id_list, entry):
    """ Statement (statement, epsilon, boolean) data dependencies. """

    logging.debug('The node %s is a statement', node.name)
    return statement_scope(node=node, scopes=scopes, id_list=id_list, entry=entry)


def build_df_identifier(node, scopes, id_list, entry):
    """ Identifier data dependencies. """

    if node.id not in id_list:
        logging.debug('The variable %s has not been handled yet', node.attributes['name'])
        identifier_update(node, scopes=scopes, id_list=id_list, entry=entry)
    else:
        logging.debug('The variable %s has already been handled', node.attributes['name'])


def build_dfg_content(child, scopes, id_list, entry):
    """ Data dependency for a given node whatever it is. """

    if child.name == 'VariableDeclaration':
        scopes = build_df_variable_declaration(child, scopes=scopes, id_list=id_list, entry=entry)

    elif child.name == 'AssignmentExpression':
        scopes = build_df_assignment(child, scopes=scopes, id_list=id_list, entry=entry)

    elif (child.name == 'CallExpression' and child.children[0].name == 'MemberExpression'
          and child.children[0].children[0].name != 'ThisExpression'
          and 'window' not in child.children[0].children[0].attributes.values())\
            or (child.name == 'CallExpression' and child.children[0].name == 'MemberExpression'
                and child.children[0].parent.name == 'MemberExpression'):
        scopes = build_df_call_expr(child, scopes=scopes, id_list=id_list, entry=entry)

    elif child.name == 'UpdateExpression':
        build_df_update(child, scopes=scopes, id_list=id_list, entry=entry)

    elif child.name == 'FunctionDeclaration':
        scopes = build_df_function(child, scopes=scopes, id_list=id_list)

    elif child.name == 'FunctionExpression':
        scopes = build_df_function(child, scopes=scopes, id_list=id_list, fun_expr=True)

    elif isinstance(child, _node.Statement):
        scopes = build_df_statement(child, scopes=scopes, id_list=id_list, entry=entry)

    elif child.name == 'ObjectExpression':  # Only consider the object name, no properties
        logging.debug('The node %s is an object expression', child.name)
        scopes = obj_expr_scope(child, scopes=scopes, id_list=id_list)

    elif child.name == 'ObjectPattern':  # Only consider the object name, not the key or properties
        logging.debug('The node %s is an object pattern', child.name)
        scopes = obj_pattern_scope(child, scopes=scopes, id_list=id_list)

    elif child.name == 'Identifier':
        build_df_identifier(child, scopes=scopes, id_list=id_list, entry=entry)

    else:
        scopes = df_scoping(child, scopes=scopes, id_list=id_list)[1]

    return scopes


def build_dfg(child, scopes, id_list, entry):
    """ Cf build_dfg_content. Added try/catch to see code snippets leading to problems and
    performing the analysis to the end. """

    try:
        scopes = build_dfg_content(child, scopes, id_list, entry)

    except utility_df.Timeout.Timeout as e:
        raise e  # Will be caught in pdgs_generation

    except Exception as e:
        logging.critical('Something went wrong with the following code snippet, %s', '')

    return scopes


def df_scoping(cfg_nodes, scopes, id_list, entry=0):
    """
        Data dependency for a complete CFG.

        -------
        Parameters:
        - cfg_nodes: Node
            Output of produce_cfg(ast_to_ast_nodes(<ast>, ast_nodes=Node('Program'))).

        -------
        Returns:
        - Node
            With data flow dependencies added.
    """

    for child in cfg_nodes.children:
        scopes = build_dfg(child, scopes=scopes, id_list=id_list, entry=entry)
    return [cfg_nodes, scopes]
