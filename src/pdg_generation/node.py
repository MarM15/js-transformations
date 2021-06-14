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
    Definition of classes Dependence and Node with subclasses Identifier, Statement and Comment.
"""

import logging


EPSILON = ['BlockStatement', 'DebuggerStatement', 'EmptyStatement',
           'ExpressionStatement', 'LabeledStatement', 'ReturnStatement',
           'ThrowStatement', 'WithStatement', 'CatchClause', 'VariableDeclaration',
           'FunctionDeclaration', 'ClassDeclaration']
# TODO: FunctionDeclaration with epsilon statements? And not just children? Same for BlockStatement

CONDITIONAL = ['DoWhileStatement', 'ForStatement', 'ForOfStatement', 'ForInStatement',
               'IfStatement', 'SwitchCase', 'SwitchStatement', 'TryStatement',
               'WhileStatement', 'ConditionalExpression']

UNSTRUCTURED = ['BreakStatement', 'ContinueStatement']

STATEMENTS = EPSILON + CONDITIONAL + UNSTRUCTURED

COMMENTS = ['Line', 'Block']


class Dependence:
    """ For control, data, comment and statement dependencies. """

    def __init__(self, dependency_type, extremity, label, nearest_statement=None):
        self.type = dependency_type
        self.extremity = extremity
        self.nearest_statement = nearest_statement  # TODO: useful??
        self.label = label


class Node:
    id = 0

    def __init__(self, name, parent=None):
        self.name = name
        self.id = Node.id
        Node.id += 1
        self.attributes = {}
        self.body = None
        self.body_list = False
        self.parent = parent
        self.children = []
        self.statement_dep_parents = []
        self.statement_dep_children = []  # Between Statement and their non-Statement descendants

    def is_leaf(self):
        return not self.children

    def set_attribute(self, attribute_type, node_attribute):
        self.attributes[attribute_type] = node_attribute

    def set_body(self, body):
        self.body = body

    def set_body_list(self, bool_body_list):
        self.body_list = bool_body_list

    def set_parent(self, parent):
        self.parent = parent

    def set_child(self, child):
        self.children.append(child)

    def set_statement_dependency(self, extremity):
        self.statement_dep_children.append(Dependence('statement dependency', extremity, 's'))
        extremity.statement_dep_parents.append(Dependence('statement dependency', self, 's'))

    # def set_comment_dependency(self, extremity):
        # self.statement_dep_children.append(Dependence('comment dependency', extremity, 'c'))
        # extremity.statement_dep_parents.append(Dependence('comment dependency', self, 'c'))

    def is_comment(self):
        if self.name in COMMENTS:
            return True
        return False


class Identifier(Node):

    def __init__(self, name, parent):
        Node.__init__(self, name, parent)
        self.value = None
        self.update_value = True
        self.code = None
        self.data_dep_parents = []
        self.data_dep_children = []

    def set_value(self, value):
        self.value = value

    def set_update_value(self, update_value):
        self.update_value = update_value

    def set_code(self, code):
        self.code = code

    def set_data_dependency(self, extremity, nearest_statement=None):
        self.data_dep_children.append(Dependence('data dependency', extremity, 'data',
                                                 nearest_statement))
        extremity.data_dep_parents.append(Dependence('data dependency', self, 'data',
                                                     nearest_statement))


class LiteralArrayObject(Node):

    def __init__(self, name, parent):
        Node.__init__(self, name, parent)
        self.value = None

    def set_value(self, value):
        self.value = value


class Statement(Node):

    def __init__(self, name, parent):
        Node.__init__(self, name, parent)
        self.control_dep_parents = []
        self.control_dep_children = []

    def set_control_dependency(self, extremity, label):
        self.control_dep_children.append(Dependence('control dependency', extremity, label))
        try:
            extremity.control_dep_parents.append(Dependence('control dependency', self, label))
        except AttributeError as e:
            logging.debug('Unable to build a CF to go up the tree: %s', e)

    def remove_control_dependency(self, extremity):
        for i, _ in enumerate(self.control_dep_children):
            elt = self.control_dep_children[i]
            if elt.extremity.id == extremity.id:
                del self.control_dep_children[i]
                try:
                    del extremity.control_dep_parents[i]
                except AttributeError as e:
                    logging.debug('No CF going up the tree to delete: %s', e)
