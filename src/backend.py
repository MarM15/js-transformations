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

import sys
import os
import argparse
import pickle
import time
import queue
import copy
import shutil
import random
import subprocess
import logging
import numpy as np
from joblib import dump, load
from script import script
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from features import *
SRC_PATH = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(SRC_PATH, 'pdg_generation'))
from pdgs_generation import get_data_flow
sys.path.insert(0, os.path.join(SRC_PATH, '../JStap/classification'))
from features_space import features_vector

logger = logging.getLogger('obf_analysis')


def get_scripts_for_folder_with_labels(
        folderlist, file_queue, evaluation):
    """
        Reads given file with paths and labels (path;label 1,label 2...\n...)

        -------
        Parameter:
        - folderlist: string
            Path to the .txt file, specified with "-i" or "-p"
        - file_queue: multiprocessing.Queue
            Queue to put the scripts on
        - evaluation: boolean
            Decides whether labels are expected at the end of each line in the .txt
        -------
        Returns:
        - file_counter: integer
            Counts how many scripts were processed

    """

    file_counter = 0
    folder_counter = 0
    with open(folderlist, "r") as f:
        while True:
            folder = f.readline().strip("; \n \t")
            if folder == "":
                break
            if folder[0] == "#":  # Comment function
                logger.debug("! skipped %s", folder[1:])
                continue
            if evaluation:  # split format of path; label,label...
                try:
                    folder, label = folder.split(";")
                except Exception as e:
                    logger.critical("%s for %s \nYou either have to specify labels for the folders,"
                                    " or additionally use \"-p\" to classify unlabeled samples",
                                    e,
                                    folder)
                    sys.exit()
                label = label.strip(" \n \t")
                label = label.split(",")
            containingFiles = os.listdir(folder)
            for item in containingFiles:
                if item.endswith(".js"):
                    file_counter += 1
                    if evaluation:
                        resulting_labels = {}
                        for l in [k.strip(" ") for k in label]:
                            if l[0] in resulting_labels:
                                resulting_labels[l[0]
                                                 ] = resulting_labels[l[0]] + [int(l)]
                            else:
                                resulting_labels[l[0]] = [int(l)]
                        file_queue.put(
                            script(
                                folder.rstrip("/") +
                                "/" +
                                item,
                                resulting_labels))
                    else:
                        file_queue.put(
                            script(
                                folder.rstrip("; /") +
                                "/" +
                                item,
                                None))
            folder_counter += 1
        logger.info(
            "Read %s scripts from %s directories",
            file_counter,
            folder_counter)

        return


def write_error(filename, error_name, dst):
    """ Writes errors of scripts that can't be processed into one txt file. """

    with open(dst + "errors.txt", "a+") as e:
        e.write("{0} : {1} \n".format(filename, error_name))


def count_relevant_expressions(ast, x):
    """ Counts expressions listed below in the ast """

    expressions = [
        "ArrowFunctionExpression",
        "FunctionExpression",
        "FunctionDeclaration",
        "CallExpression",
        "TaggedTemplateExpression",
        "DoWhileStatement",
        "WhileStatement",
        "ForStatement",
        "ForOfStatement",
        "ForInStatement",
        "IfStatement",
        "ConditionalExpression",
        "TryStatement",
        "SwitchStatement"]

    for child in ast.children:
        for exp in expressions:
            if child.name == exp:
                x += 1
        x = count_relevant_expressions(child, x)
    return x


def worker(
        queue_in,
        queue_arrays_l1,
        queue_arrays_l2_trans,
        pred_queue,
        pickle_address,
        usecase,
        only_level_1,
        error,
        move):
    """
        Computes features for JavaScript files.

        -------
        Parameter:
        - queue_in: multiprocessing.Queue
            Queue containing all scripts, that should be computed
        - queue_arrays_l1, queue_arrays_l2_trans: multiprocessing.Queue
            Queue where the worker puts all his computed features at the end of his lifetime
        - pred_queue: multiprocessing.Queue
            Queue where the worker puts all his computed features, if usecase == "prediction"
        - pickle_address: string
            Path to models and multilabelbinarizer
        - usecase: string
            Defines what the worker is used for (learning, accuracy, prediction)
        - only_level_1: boolean
            Defines if only level 1 should be considered
        - error: string
            Defines if and where scripts that cause an error should be documented
        - error: string
            Defines if and where scripts that were processed successfully should be moved

    """

    logger.info("Worker started")

    # Computed features (and labels) get appended to these lists. if all scripts
    # are processed, these lists are put on a queue
    x_train_l1 = []
    y_train_l1 = []
    x_train_l2_obf = []
    y_train_l2_obf = []
    pred_array = []
    JStap_features_path = "./JStap/Analysis/Features/ngrams/ast_selected_features_99"
    JStap_selected_features = load(JStap_features_path)
    JStap_selected_features_amount = len(JStap_selected_features) + 1

    features2int_dict = pickle.load(open(JStap_features_path, 'rb'))

    while True:
        try:
            if queue_in.qsize() == 0:
                break
            script = queue_in.get(timeout=2)
            filename = script.filename

            # to store the pdg for JStap, which is deleted later
            # TODO: if PDGs are generated when learning, do we need this here again? relict?
            open(filename.rstrip(".js"), 'a').close()
            ast, comments, _, tokens = get_data_flow(
                filename, benchmarks=dict(), store_pdgs="/".join(filename.split("/")[:-1]))

            filename = script.filename
            logger.debug("Working on %s", filename)
            if queue_in.qsize() % 250 == 0:
                logger.info(
                    "Scripts left: %s, working on: %s, since: %s",
                    queue_in.qsize(),
                    filename,
                    time.time())

            if ast is None or comments is None or tokens is None:
                if error is not None:
                    write_error(filename, "Esprima returned None", error)
                continue

            if count_relevant_expressions(ast, 0) == 0:
                if error is not None:
                    write_error(filename, "No relevant expression", error)
                continue

            signal.alarm(240)  # Feature timeout = 4 minutes

            # Begin calculating features
            features_with_labels = []
            var_dict, _ = variables_declared(ast, {}, 0)
            variables = get_variable_function_names(ast, [])
            dict_dict, _ = dictionarys_declared(ast, {}, 0)
            string_dict, _ = string_size_at_declaration(ast, {}, 0)
            array_dict, _ = array_size_at_declaration(ast, {}, 0)
            token_counts = counts_from_tokens(tokens)

            ast_hunted = ast_hunter(
                ast,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                filename,
                0, 0, 0, 0, 0, 0,
                [])
            cnt_literale = ast_hunted[0]
            cnt_identifier = ast_hunted[1]
            cnt_member_expression = ast_hunted[2]
            cnt_new_expression = ast_hunted[3]
            cnt_evil_expression = ast_hunted[4]
            cnt_call_expression = ast_hunted[5]
            c1 = ast_hunted[6]
            c2 = ast_hunted[7]
            c3 = ast_hunted[8]
            c4 = ast_hunted[9]
            if c2 == 0 or c4 == 0:
                statements_length, string_length = (0, 0)
            else:
                statements_length, string_length = (c1 / c2, c3 / c4)
            lines_and_words_features = features_based_on_lines_and_words(
                filename)
            avg_whitespaces = lines_and_words_features[1]
            avg_hex = lines_and_words_features[2]
            avg_oct = lines_and_words_features[3]
            avg_uni = lines_and_words_features[4]
            words = lines_and_words_features[6]
            number_of_lines = lines_and_words_features[7]
            chars = lines_and_words_features[8]
            max_word_size = lines_and_words_features[10]
            avg_word_size = lines_and_words_features[11]
            avg_binary = lines_and_words_features[12]
            breadth = ast_hunted[10]
            max_array_size_, array_size_, array_ops = array_hunter(array_dict)
            var_ops, unused_var = var_hunter(var_dict)
            dict_size, dict_size_run, dict_ops = dict_hunter(dict_dict)
            count_tabs, avg_char_jsfuck, entropy = counts_from_files(
                filename, chars + 1)
            cnt_ternary_words = ast_hunted[18]
            max_lit_size = ast_hunted[19]
            cnt_computed_expression = ast_hunted[20]
            raws = ast_hunted[21]
            depth = ast_hunted[22] + 1
            unique_identifier = cnt_unique_identifier(variables)
            cl = ast_hunted[13]
            sll = ast_hunted[14]
            count_statements = ast_hunted[16]
            count_if_statements = ast_hunted[17]

            # Prevent division by zero
            divider = [
                count_statements,
                chars,
                words,
                cnt_call_expression,
                number_of_lines,
                unique_identifier,
                cl]
            for i in range(len(divider)):
                if divider[i] == 0:
                    divider[i] = 1
            count_statements, \
            chars, \
            words, \
            cnt_call_expression, \
            number_of_lines, \
            unique_identifier,\
            cl = divider

            features_with_labels.append(breadth / words)
            features_with_labels.append(depth / words)
            features_with_labels.append(count_tabs / chars)
            for item in token_counts[:19]:
                features_with_labels.append(item / cnt_call_expression)
            for item in token_counts[19:]:
                features_with_labels.append(item / chars)
            features_with_labels.append(avg_whitespaces)
            features_with_labels.append(avg_binary)
            features_with_labels.append(avg_hex)
            features_with_labels.append(avg_oct)
            features_with_labels.append(avg_uni)
            features_with_labels.append(max_word_size)
            features_with_labels.append(avg_word_size)
            features_with_labels.append(cnt_literale / words)
            features_with_labels.append(cnt_identifier / words)
            features_with_labels.append(
                cnt_member_expression / unique_identifier)
            features_with_labels.append(cnt_new_expression / words)
            features_with_labels.append(
                cnt_evil_expression / cnt_call_expression)
            features_with_labels.append(cnt_call_expression / words)
            features_with_labels.append(cnt_call_expression / words)
            features_with_labels.append(statements_length)
            features_with_labels.append(string_length)
            features_with_labels.append(entropy)
            features_with_labels.append(cnt_ternary_words / words)
            features_with_labels.append(max_lit_size)
            features_with_labels.append(
                cnt_computed_expression /
                cnt_call_expression)
            with open(filename, "r") as k:
                features_with_labels.append(
                    count_mult_whitespaces(
                        k.read()) / words)
            features_with_labels.append(array_size_)
            features_with_labels.append(max_array_size_)
            features_with_labels.append(array_ops)
            features_with_labels.append(human_readable(raws, variables))
            features_with_labels.append(
                avg_string_length(
                    copy.copy(string_dict)))
            features_with_labels.append(
                avg_ops_on_strings(
                    copy.copy(string_dict)))
            features_with_labels.append(var_ops)
            features_with_labels.append(dict_size)
            features_with_labels.append(dict_ops)
            features_with_labels.append(dict_size_run)
            features_with_labels.append(unused_var)
            features_with_labels.append(
                (ast_hunted[11] + ast_hunted[12]) / chars)
            features_with_labels.append(
                human_readable_markov_chain(
                    raws, variables))
            features_with_labels.append(sll / cl)
            features_with_labels.append(
                human_readable_markov_chain_comments(comments))
            features_with_labels.append(ast_hunted[15] / words)
            features_with_labels.append(
                count_jfogs_identifier(ast) /
                unique_identifier)
            features_with_labels.append(cointains_debugger_string(raws))
            features_with_labels.append(count_if_statements / count_statements)
            features_with_labels.append(min_detected(filename, chars))
            features_with_labels.append(avg_char_jsfuck)
            # If you wish to add features, add them here

            features_with_labels2 = copy.copy(features_with_labels)


            # JStap ast ngrams features
            features_vect = None
            try:
                features_vect = features_vector(
                    filename.rstrip(".js"), "ast", "ngrams", 4, features2int_dict)
            except Exception as e:
                logger.warning(
                    "JStap failed for %s with:\n %s",
                    filename,
                    e)
                if error is not None:
                    write_error(filename, e, error)

            os.remove(filename.rstrip(".js"))

            if features_vect is not None:
                features_vect = features_vect.toarray()
                features_vect = features_vect[0].tolist()
            else:
                features_vect = [[0] * JStap_selected_features_amount][0]

            # JStap features for level 1
            features_with_labels += features_vect
            # JStap features for level 2
            features_with_labels2 += features_vect

            # Stop timeout
            signal.alarm(0)

            # Append features (labels) to correct lists
            script.add_features(features_with_labels)
            if script.is_it_for_level1() or usecase == "accuracy":
                x_train_l1.append(features_with_labels)
                y_train_l1.append(script.get_labels_for_level1())

            # Append features (labels) to correct lists
            # (if they are not only for level 1, e.g. two-digit label)
            if not only_level_1 and not script.is_it_for_level1():
                if usecase == "learning":
                    labels_level1 = script.get_labels_for_level1()
                    if 2 in labels_level1:
                        x_train_l2_obf.append(features_with_labels2)
                        y_train_l2_obf.append(script.get_labels_for("2"))
                elif usecase == "accuracy":
                    labels_level1 = script.get_labels_for_level1()
                    rf = load(pickle_address + "L1_CC_rf.joblib")
                    prediction = rf.predict([features_with_labels2])
                    # If script is predicted to be transformed, give it to
                    # level 2
                    if 2 in labels_level1 and (
                            prediction[0][2] == 1 or prediction[0][1] == 1):
                        x_train_l2_obf.append(features_with_labels2)
                        y_train_l2_obf.append(script.get_labels_for("2"))
                elif usecase == "prediction":
                    pred_array.append(script)

            # If processed scripts should be moved to given folder
            if move is not None:
                shutil.move(filename, move + filename.split("/")[-1:][0])

        except queue.Empty:
            logger.debug(
                "queue empty exception in PID %s",
                os.getpid())
            break
        except Exception as e:
            logger.warning(
                "Worker failed for %s with:\n %s",
                filename,
                e)
            if error is not None:
                write_error(filename, e, error)

    if usecase == "prediction":
        pred_queue.put(pred_array)
    else:
        queue_arrays_l1.put((x_train_l1, y_train_l1))
        queue_arrays_l2_trans.put((x_train_l2_obf, y_train_l2_obf))

    logger.info("Worker terminated, PID: %s", os.getpid())


def ml_learner(pickle_address, queue_arrays_l1, num_workers):
    """
        Trains the model for level 1.

        -------
        Parameter:
        - pickle_address: string
            Path to models and multilabelbinarizer
        - queue_arrays_l2: multiprocessing.Queue
            Queue where the workers puts all their computed features at the end of their lifetime
        -num_workers: integer
            Specifies how many threads should be used in the learnung phase
        -------
    """
    x_train = []
    y_train = []

    # Reconstruct features and labels out of queue from worker into single
    # lists
    try:
        while True:
            x, y = queue_arrays_l1.get(timeout=2)
            x_train += x
            y_train += y

    except queue.Empty:
        logger.debug("reconstructed all features + labels")

    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform(y_train)
    dump(mlb, pickle_address + "L1_Mlb.joblib")
    cc_rf = ClassifierChain(
        RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=0,
            n_jobs=num_workers))
    cc_rf.fit(x_train, y_train)
    dump(cc_rf, pickle_address + "L1_CC_rf.joblib")


def ml_classifier(pickle_address, queue_arrays_l1):
    """
        Evaluates accuracy on level 1.

        -------
        Parameter:
        - pickle_address: string
            Path to models and multilabelbinarizer
        - queue_arrays_l1: multiprocessing.Queue
            Queue where the workers puts all their computed features at the end of their lifetime
        -------
    """
    x_test = []
    y_test_l1 = []

    # Reconstruct features and labels out of queue from worker into single
    # lists
    try:
        while True:
            x, y = queue_arrays_l1.get(timeout=2)
            x_test += x
            y_test_l1 += y

    except queue.Empty:
        logger.debug("reconstructed all features + labels")

    dump([y_test_l1, x_test], "ml_classifier_L1_data.joblib")

    f = open("./results.txt", "w+")
    mlb = load(pickle_address + "L1_Mlb.joblib")
    y_test_l1 = mlb.transform(y_test_l1)
    rf = load(pickle_address + "L1_CC_rf.joblib")

    # Transform prediction so that there is always at least one prediction
    # and label 1 & 2 are seen as label 2 ("transformed")
    y_pred = transforme(rf.predict(x_test), rf.predict_proba(x_test))
    f.write("### Level 1\nLevel 1 - RandomForest evaluation:\nAccuracy: {0}".format(
        str(metrics.accuracy_score(y_test_l1, y_pred)) + "\n"))
    f.close()


def arguments():
    """ Read cli arguments. """
    parser = argparse.ArgumentParser()
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    parser._action_groups.append(optional)
    required.add_argument(
        "-i",
        "--input",
        help="Points to a .txt with folders of JS-files (+ labels if not -p)",
        required=True)
    optional.add_argument(
        "-p",
        "--predict",
        help="Stores the prediction for each JS-file specified with -i in results.txt",
        action='store_true')
    optional.add_argument(
        "-m",
        "--model",
        help="The Path where the model is stored (default: ./model)",
        default="./model")
    optional.add_argument(
        "-t",
        "--threads",
        type=int,
        help="Number of threads (default: 1)",
        default=1)
    optional.add_argument(
        "-l1",
        "--level1",
        action='store_true',
        help="Only evaluate the classifiers at level one")
    optional.add_argument(
        "--translate",
        help="Points to a .txt to translate integer-labels to names, if -p is chosen")
    optional.add_argument(
        "--move",
        help="Moves successfully processed files to given folder")
    optional.add_argument(
        "--error",
        help="Creates an errors.txt in the specified path to log errors")
    optional.add_argument(
        "--log",
        choices=[
            'DEBUG',
            'INFO',
            'WARNING',
            'ERROR',
            'CRITICAL'],
        help="Sets the logging level",
        default='INFO')
    optional.add_argument(
        "--pdg_regenerate",
        action='store_true',
        help="Deletes already generated PDGs and generates them again. Use this if you changed your trainingset")
    args = parser.parse_args()

    num_workers = args.threads
    input_file = args.input
    evaluation = not args.predict
    ml_path = args.model
    if ml_path is not None:
        ml_path = ml_path.rstrip("/") + "/"
        if not os.path.isdir(ml_path) and ml_path == "./model/":
            os.mkdir("./model")
    only_level_1 = args.level1
    translation_file = args.translate
    error = args.error
    if error is not None:
        error = error.rstrip("/") + "/"
    move = args.move
    if move is not None:
        move = move.rstrip("/") + "/"
    logging_level = args.log
    pdg_regen = args.pdg_regenerate

    if input_file is None and "-help" not in sys.argv:
        print("Invalid or not enough arguments. Use --help")
        sys.exit()

    return (
        input_file,
        ml_path,
        num_workers,
        only_level_1,
        evaluation,
        translation_file,
        error,
        move,
        logging_level,
        pdg_regen)


def ml_learner_l2(pickle_address, queue_arrays_l2, num_workers):
    """
        Trains the model for level 2.

        -------
        Parameter:
        - pickle_address: string
            Path to models and multilabelbinarizer
        - queue_arrays_l2: multiprocessing.Queue
            Queue where the workers puts all their computed features at the end of their lifetime
        -num_workers: integer
            Specifies how many threads should be used in the learnung phase
        -------
    """
    x_train = []
    y_train = []
    mlb = MultiLabelBinarizer()

    # Reconstruct features and labels out of queue from worker into single
    # lists
    try:
        while True:
            x, y = queue_arrays_l2.get(timeout=2)
            x_train += x
            y_train += y

    except queue.Empty:
        logger.debug("reconstructed all features + labels")

    if len(x_train) == 0 or len(y_train) == 0:
        logger.error("There is no script to train on level 2")
        sys.exit(0)

    mlb.fit(y_train)
    y_train = mlb.transform(y_train)
    dump(mlb, "{0}binarizer-l2.joblib".format(pickle_address))
    cc_rf = ClassifierChain(
        RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            random_state=0,
            n_jobs=num_workers))
    cc_rf.fit(x_train, y_train)
    dump(cc_rf, "{0}CC-l2_rf.joblib".format(pickle_address))


def ml_classifier_L2(pickle_address, queue_arrays_l2):
    """
        Evaluates accuracy, k-score, multi_label_accuracy and accuracy_per_label on level 2.
        It also saves corresponding charts.

        -------
        Parameter:
        - pickle_address: string
            Path to models and multilabelbinarizer
        - queue_arrays_l2: multiprocessing.Queue
            Queue where the workers puts all their computed features at the end of their lifetime
        -------
    """
    y_test_l2 = []
    x_test = []

    # Reconstruct features and labels out of queue from worker into single
    # lists
    try:
        while True:
            x, y = queue_arrays_l2.get(timeout=2)
            x_test += x
            y_test_l2 += y

    except queue.Empty:
        logger.debug("reconstructed all features + labels")

    dump([y_test_l2, x_test], "ml_classifier_L2_data.joblib")

    if len(x_test) == 0 or len(y_test_l2) == 0:
        logger.error("There are no scripts to classify on level 2")
        sys.exit(0)

    mlb = load("{0}binarizer-l2.joblib".format(pickle_address))
    y_test_l2 = mlb.transform(y_test_l2)

    with open("./results.txt", "a+") as f:
        f.write("\n\n ### Level 2\n")

        # Create predictions for every threshold
        x = np.arange(5, 100, 5)
        rf = load("{0}/CC-l2_rf.joblib".format(pickle_address))
        y_pred_array = apply_threshold(rf.predict_proba(x_test), mlb)

        # Evaluation Fig. 1 for the thresholds 0%, 10%, 50%, 90%
        for th in [0.0, 0.1, 0.5, 0.9]:
            kscore, dealbreaker, dealbreaker_reverse = k_score(
                rf.predict_proba(x_test), y_test_l2, mlb.classes_, th)
            wrong_labels = []
            for _, value in dealbreaker.items():
                count = 0
                for _, value2 in value.items():
                    count += value2
                wrong_labels.append(count / len(x_test))

            wrong_labels_reverse = []
            for key, value in dealbreaker_reverse.items():
                count = 0
                for key2, value2 in value.items():
                    count += value2
                wrong_labels_reverse.append(count / len(x_test))

        threshold = 0.05
        acc = []
        for y_pred in y_pred_array:  # Iterate through all possible thresholds
            accuracy = metrics.accuracy_score(y_test_l2, y_pred)
            acc.append(accuracy * 100)
            f.write(
                "\nLevel 2 - RandomForest Accuracy (t = {0:.2f}):\t\t {1}\n".format(threshold, accuracy))
            f.write(
                "multi_label_accuracy = {0}\n".format(
                    multi_label_accuracy(
                        y_pred, y_test_l2)))
            f.write(
                "accuracy_per_label = {0}\n".format(
                    accuracy_per_label(
                        y_pred,
                        y_test_l2,
                        mlb)))
            threshold += 0.05



def translate_labels(translation_file):
    """ Read translation-file to beatify output of the prediction,
        returns a dirctionary for the translation.
        (replacing numeric labels with names)
    """

    if translation_file is None:
        return None
    translation = {}
    with open(translation_file, "r") as t:
        for line in t:
            name, label = line[:-1].split(";")
            translation[int(label)] = name
    return translation


def apply_threshold(y_pred, mlb):
    """ Lowers threshold for prediction in 5% steps. """

    classes = mlb.classes_
    y_pred_array = []
    for threshold in np.arange(0.05, 1, 0.05):
        y_new = []
        for i in range(len(y_pred)):
            y = []
            for z in range(len(y_pred[0])):
                if y_pred[i][z] >= threshold:
                    y.append(classes[z])
            y_new.append(y)
        y_pred_array.append(mlb.transform(y_new))
    return y_pred_array


def multi_label_accuracy(y_pred, y_test_l2):
    """
        Another way of calculating the accuracy for multilabel-predictions.
        X = total files with N number of techniques
        Y = label_count_predicted
        a = incorrectly detected
        b = not recognized

        -------
        Parameter:
        - y_pred: list
            Prediction of a model (rf.predict(x_test))
        - y_test_l2: list
            True labels of the scripts
        -------
        Returns:
        - accuracy: dictionary
            As described above

    """

    x_dict = {}
    accuracy = {}

    for i in range(len(y_pred)):
        label_count_test = np.count_nonzero(
            y_test_l2[i] == 1)  # Amount of labels on this sample
        if label_count_test not in x_dict:  # Adding it to the total of files with this many labels
            x_dict[label_count_test] = 1
        else:
            x_dict[label_count_test] += 1

        # Number of correctly predicted features | at y_test_l2 0s are set to
        # 2, so that 0==0 does not count in the comparison
        label_count_predicted = np.count_nonzero(
            np.equal(y_pred[i], [2 if j < 1 else j for j in y_test_l2[i]]))
        label_count_false_positive = np.count_nonzero(
            np.equal(y_pred[i], [j + 1 for j in y_test_l2[i]]))
        label_count_false_negative = np.count_nonzero(
            np.equal([j + 1 for j in y_pred[i]], y_test_l2[i]))
        if str(label_count_predicted) + "/" + \
                str(label_count_test) not in accuracy:
            accuracy[str(label_count_predicted) + "/" + str(label_count_test)] = [1,
                                                                                  None,
                                                                                  label_count_false_positive,
                                                                                  label_count_false_negative]
        else:
            accuracy[str(label_count_predicted) + "/" + str(label_count_test)
                     ][0] = accuracy[str(label_count_predicted) + "/" + str(label_count_test)][0] + 1
            accuracy[str(label_count_predicted) + "/" + str(label_count_test)][2] = accuracy[str(
                label_count_predicted) + "/" + str(label_count_test)][2] + label_count_false_positive
            accuracy[str(label_count_predicted) + "/" + str(label_count_test)][3] = accuracy[str(
                label_count_predicted) + "/" + str(label_count_test)][3] + label_count_false_negative

    for key in accuracy:  # x is added and avg of FN and FP is calculated
        accuracy[key][1] = x_dict[int(key[key.find("/") + 1:])]
        accuracy[key][2] = accuracy[key][2] / accuracy[key][0]
        accuracy[key][3] = accuracy[key][3] / accuracy[key][0]

    # Returns accuracy, sorted by size of the values in descending order
    return {
        k: v for k,
        v in sorted(
            accuracy.items(),
            key=lambda item: item[1],
            reverse=True)}


def accuracy_per_label(y_pred, y_test_l2, mlb):
    """ Calculates the accuracy for every label (|label in y_true| / |label in y_true and label in y_pred|). """

    accuracy = {}
    classes = mlb.classes_
    for entry in range(len(y_test_l2[0])):  # initializing
        accuracy[entry] = [0, 0]

    for row in range(len(y_pred)):
        for entry in range(len(y_pred[row])):
            if y_pred[row][entry] == 1 and y_test_l2[row][entry] == 1:  # predicts label
                accuracy[entry] = [
                    accuracy[entry][0] + 1,
                    accuracy[entry][1] + 1]
            elif y_test_l2[row][entry] == 1:  # was not predicted
                accuracy[entry][1] = accuracy[entry][1] + 1

    # calculates average per label
    accuracy = {
        k: float(
            v[0]) /
        float(
            v[1]) if v[1] > 0 else None for k,
        v in sorted(
                accuracy.items(),
                key=lambda item: item[1],
            reverse=True)}
    accuracy_new = {}
    for key in accuracy:  # translates position into label
        accuracy_new[classes[key]] = accuracy[key]
    return accuracy_new


def get_k_best(predict_proba, k, threshold):
    """ Returns k best predictions (with highest probability). """

    y_pred = [] * len(predict_proba[0])
    for i in range(len(predict_proba)):
        sample_dict = {}
        for z in range(len(predict_proba[0])):
            sample_dict[z] = predict_proba[i][z]
        y = [0] * len(predict_proba[0])
        for p in range(k):
            if sorted(
                    sample_dict.items(),
                    key=lambda item: item[1])[
                    ::-1][p][1] >= threshold:
                y[sorted(sample_dict.items(),
                         key=lambda item: item[1])[::-1][p][0]] = 1
        y_pred.append(y)
    return y_pred


def compare(y_pred, y_true, classes):
    """ Compares if the k-predicted labels are correct. """

    hits = 0
    count = 0
    dealbreaker = {}
    dealbreaker_reverse = {}
    for i in range(len(y_pred)):
        # Sets the indices of the selected labels
        p = set([i for i, x in enumerate(y_pred[i]) if x == 1])
        t = set([i for i, x in enumerate(y_true[i]) if x == 1])
        if len(p) > len(t):
            for d in p.difference(t):
                if classes[d] in dealbreaker:
                    x = dealbreaker[classes[d]]
                    dealbreaker[classes[d]] = x + 1
                else:
                    dealbreaker[classes[d]] = 1
            for d in t.difference(p):
                if classes[d] in dealbreaker_reverse:
                    x = dealbreaker_reverse[classes[d]]
                    dealbreaker_reverse[classes[d]] = x + 1
                else:
                    dealbreaker_reverse[classes[d]] = 1
            continue
        if len(p) == 0:
            continue
        if len(p.difference(t)) == 0:
            # Then the prediction is correct (the k best predictions)
            hits += 1
        else:
            for d in p.difference(t):
                if classes[d] in dealbreaker:
                    x = dealbreaker[classes[d]]
                    dealbreaker[classes[d]] = x + 1
                else:
                    dealbreaker[classes[d]] = 1
            for d in t.difference(p):
                if classes[d] in dealbreaker_reverse:
                    x = dealbreaker_reverse[classes[d]]
                    dealbreaker_reverse[classes[d]] = x + 1
                else:
                    dealbreaker_reverse[classes[d]] = 1
        count += 1
    return (0 if count == 0 else hits / count,
            {k: v for k,
             v in sorted(dealbreaker.items(),
                         key=lambda item: item[1],
                         reverse=True)},
            {k: v for k,
             v in sorted(dealbreaker_reverse.items(),
                         key=lambda item: item[1],
                         reverse=True)})


def k_score(predict_proba, y_true, classes, threshold):
    """
        Calculates the k-score.

        -------
        Parameter:
        - predict_proba: list
            Pobabilities of model (rf.predict_proba(x_test))
        - y_true: list
            True labels of the scripts
        - classes: list
            The classes the model is trained on (mlb.classes)
        - threshold: float
            The threshold that should be applied to the predictions
        -------
        Returns:
        - kscore: list
            The resulting k-score
        - dealbreaker: dictionary
            Counting the most falsely predicted labels
        - dealbreaker_reverse: dictionary
            Counting the most missing labels, that were not predicted

    """

    k_score = []
    dealbreaker = {}
    dealbreaker_reverse = {}
    # Calculate score for every k
    for k in range(1, len(y_true[0] + 1)):
        y_pred = get_k_best(predict_proba, k, threshold)
        score, dealbreaker[k], dealbreaker_reverse[k] = compare(
            y_pred, y_true, classes)
        k_score.append(score)
    return k_score, dealbreaker, dealbreaker_reverse


def predictor(pred_queue, pickle_address, translation):
    """
        Prints predictions and charts about the usage of transformation in the given files

        -------
        Parameter:
        - pred_queue: multiprocessing.Queue
            Queue containing arrays from the workers, containing script.onkects with their features
        - pickle_address: string
            Path to models and multilabelbinarizer
        - translation: dictionary
            Dictionary to translate integer-labels to names

    """

    scripts = []
    l1 = []
    trans = []

    names = [
        "Identifier obfuscation",
        "Self-defending",
        "String obfuscation",
        "Dead-code injection",
        "Debug protection",
        "No alphanumeric",
        "Control-flow flattening",
        "Global array",
        "Minification advanced",
        "Minification simple"]

    colors = [
        "aqua",
        "blue",
        "blueviolet",
        "chartreuse",
        "darkcyan",
        "darkgreen",
        "darkorange",
        "dimgray",
        "fuchsia",
        "goldenrod"]

    # Reconstruct features out of queue from worker into single lists
    logger.debug(
        "got %s processed scripts to predict",
        pred_queue.qsize())
    while True:
        try:
            x = pred_queue.get(timeout=2)
            scripts += x
            for y in x:
                l1.append(y.features)
                trans.append(y.get_features_obf())

        except queue.Empty:
            logger.debug("reconstructed all features + labels")
            break

    dump(scripts, "./src/scripts_features.joblib")  # dumps all script-objects

    for y in scripts:
        l1.append(y.features)
        trans.append(y.get_features_obf())

    rf = load(pickle_address + "L1_CC_rf.joblib")
    mlb = load(pickle_address + "L1_Mlb.joblib")
    trans_mlb = load(pickle_address + "binarizer-l2.joblib")
    trans_rf = load(pickle_address + "CC-l2_rf.joblib")
    trans_classes = trans_mlb.classes_
    l1_pred = rf.predict(l1)
    l1_proba = rf.predict_proba(l1)
    l1_pred = transforme(l1_pred, l1_proba)
    trans_pred = trans_rf.predict(trans)
    trans_proba = trans_rf.predict_proba(trans)
    counts_l1 = [0, 0]
    counts_l2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Over 50%
    counts_l2_6 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Over 6%
    avg_l2 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # Avg probability per label

    normal = []
    mini = []
    obf = []
    trans = []

    with open("prediction.txt", "w+") as r:
        for j in range(len(scripts)):
            r.write("\n{0}\n   --> LEVEL 1 \n   {1}\n".format(
                scripts[j].filename, mlb.inverse_transform(np.array([l1_pred[j]]))))
            for i in range(len(l1_proba[0])):
                if int(i) == 1 and l1_proba[j][i] >= 0.5:
                    mini.append(scripts[j].filename)
                if int(i) == 2 and l1_proba[j][i] >= 0.5:
                    obf.append(scripts[j].filename)
                if translation is None:
                    r.write(
                        "   Label {0} = {1}\n".format(
                            mlb.classes_[i],
                            l1_proba[j][i] * 100))
                else:
                    r.write("    " +
                            translation[trans_classes[mlb.classes_[i]]] +
                            " = " +
                            str(l1_proba[j][i] * 100) +
                            "%\n")
            if l1_pred[j][0] == 1:  # Predicted as normal
                r.write("    --> Normal\n")
                counts_l1[0] += 1
                normal.append(scripts[j].filename)

            if l1_pred[j][1] == 1 or l1_pred[j][2] == 1:  # Predicted as transformed
                trans.append(scripts[j].filename)

                r.write("    --> LEVEL 2 Transformed\n")
                r.write(
                    "    " + str(trans_mlb.inverse_transform(np.array([trans_pred[j]]))[0]) + "\n")
                for p in range(len(trans_proba[j])):
                    if translation is None:
                        r.write("    Label " +
                                str(trans_classes[p]) +
                                " = " +
                                str(float(trans_proba[j][p]) *
                                    100) +
                                "%\n")
                    else:
                        r.write("    " +
                                translation[trans_classes[p]] +
                                " = " +
                                str(float(trans_proba[j][p]) *
                                    100) +
                                "%\n")
                    if float(trans_proba[j][p]) >= 0.5:
                        counts_l2[p] += 1
                    if float(trans_proba[j][p]) >= 0.06:
                        counts_l2_6[p] += 1
                    avg_l2[p] += float(trans_proba[j][p])

                counts_l1[1] += 1


def transforme(y_pred, y_pred_proba):
    """
        Makes sure that always at least one label is predicted
        & transforms all min-predictions to "obfuscated" -> representing transformed!
        -------
        Parameter:
        - y_pred: list
            Prediction of a model (rf.predict(x_test))
        - y_pred_proba: list
            Pobabilities of model (rf.predict_proba(x_test))
        -------
        Returns:
        - list
            transformed prediction
    """

    for i in range(len(y_pred)):
        if no_prediction(y_pred[i]):
            x = {0}
            for j in range(len(y_pred_proba[i])):  # find highest probability
                ran = x.pop()
                if y_pred_proba[i][j] > y_pred_proba[i][ran]:
                    ran = j
                if y_pred_proba[i][j] == y_pred_proba[i][ran]:
                    x.add(j)
                x.add(ran)
            for z in x:
                y_pred[i][z] = 1

    for p in y_pred:
        if p[1] == 1:
            p[1] = 0
            p[2] = 1
    return y_pred


def no_prediction(y):
    """ Check if there was no label predicted. """

    for i in y:
        if i != 0:
            return False
    return True


def train_JStap(normal_pdg_path, normal_pdg_path_validate):
    """ Starts JStap to train a model

        -------
        Parameter:
        - normal_pdg_path: string
            Path to the PDGs of the training set
        - normal_pdg_path: string
            Path to the PDGs of the validation set
        -------
    """
    logger.info("JStap: start training JStap-model")
    try:
        subprocess.run(['python3',
                        'learner.py',
                        '--d',
                        normal_pdg_path,
                        '../../src/transformed_pdgs/Analysis/PDG',
                        '--l',
                        'benign',
                        'malicious',
                        '--vd',
                        normal_pdg_path_validate,
                        '../../src/transformed_pdgs/Analysis/PDG_validate',
                        '--vl',
                        'benign',
                        'malicious',
                        '--level',
                        'ast',
                        '--features',
                        'ngrams',
                        '--mn',
                        'FEATURES_LEVEL'],
                       cwd=os.path.join(SRC_PATH, '../JStap/classification'),
                       check=True)
    except Exception as e:
        logger.critical(
            "JStap training failed with:\n %s",
            e)
        sys.exit()

    set_JStap_worker(2)


def set_JStap_worker(num_workers):
    """ Sets the workers to be used when generating the PDGs """
    utility = open(
        os.path.join(
            SRC_PATH,
            "../JStap/pdg_generation/utility_df.py"),
        "r")
    utility_lines = utility.readlines()
    for i, line in enumerate(utility_lines):
        if line.startswith("NUM_WORKERS"):
            utility_lines[i] = "NUM_WORKERS = {0}\n".format(num_workers)

    utility = open(
        os.path.join(
            SRC_PATH,
            "../JStap/pdg_generation/utility_df.py"),
        "w")
    utility.writelines(utility_lines)
    utility.close()


def handle_JStap(folderlist, pdg_regen, num_workers):
    """ Handles the setup for JStap to build a model
        - creates validation set
        - generates PDGs
        - trains JStap

        -------
        Parameter:
        - folderlist: string
            Path to the .txt file, specified with "-i" or "-p"
        - pdg_regen: boolean
            Decides whether PDGs should be regenerated
        - num_workers: integer
            Specifies how many threads should be used in the learnung phase
        -------
    """
    set_JStap_worker(num_workers)
    transformed_path = os.path.join(SRC_PATH, "transformed_pdgs/")

    normal_path = None
    minified_path = None
    obfuscated_path = None

    with open(folderlist, "r") as f:
        while True:
            folder = f.readline().strip("; \n \t")
            if folder == "":
                break
            if folder[0] == "#":  # Comment function
                logger.debug("! skipped %s", folder[1:])
                continue

            folder, label = folder.split(";")
            label = label.strip(" \n \t")
            label = label.split(",")

            if label[0] == "0":
                normal_path = os.path.join(folder)
            elif label[0] == "1":
                minified_path = os.path.join(folder)
            elif label[0] == "2":
                obfuscated_path = os.path.join(folder)

    normal_pdg_path = os.path.join(normal_path, "Analysis/PDG/")
    normal_pdg_path_validate = os.path.join(
        normal_path, "Analysis/PDG_validate")
    transformed_pdg_path = os.path.join(transformed_path, "Analysis/PDG")
    transformed_pdg_validate_path = os.path.join(
        transformed_path, "Analysis/PDG_validate")

    # If PDGs were already generated and should not be regenerated, start
    # training
    if not pdg_regen and os.path.isdir(transformed_path):
        train_JStap(normal_pdg_path, normal_pdg_path_validate)
        return

    # Create folder for transformed PDGs (delete existing if they are to be
    # regenerated)
    if os.path.isdir(transformed_path):
        print(
            "{0} will be deleted. Do you want to continue? (y/n)".format(transformed_path))
        if input() != "y":
            sys.exit()
        shutil.rmtree(transformed_path)
    os.mkdir(transformed_path)

    # Create folder for non-transformed PDGs (delete existing if they are to
    # be regenerated)
    normal_path_analysis = os.path.join(normal_path, "Analysis/")
    if os.path.isdir(normal_path_analysis):
        print(
            "{0} will be deleted. Do you want to continue? (y/n)".format(normal_path_analysis))
        if input() != "y":
            sys.exit()
        shutil.rmtree(normal_path_analysis)
    else:
        os.mkdir(normal_path_analysis)

    # Generate PDGs for non-transformed JS-files
    logger.info("JStap: generating PDGs for %s", normal_path)
    try:
        subprocess.run(
            [
                'python3',
                '-c',
                "from pdgs_generation import *; store_pdg_folder('{0}')".format(normal_path)],
            cwd=os.path.join(
                SRC_PATH,
                '../JStap/pdg_generation'),
            check=True)
    except Exception as e:
        logger.critical(
            "PDG generation failed with:\n %s",
            e)
    # Create a set containing minified and obfuscated JS-files
    normal_files = []
    if normal_files is not None:
        normal_files = [
            os.path.join(
                normal_path,
                f) for f in os.listdir(normal_path) if os.path.isfile(
                os.path.join(
                    normal_path,
                    f))]
    minified_files = []
    if minified_files is not None:
        minified_files = [("1", os.path.join(minified_path, f)) for f in os.listdir(
            minified_path) if os.path.isfile(os.path.join(minified_path, f))]
    obfuscated_files = []
    if obfuscated_path is not None:
        obfuscated_files = [("2", os.path.join(obfuscated_path, f)) for f in os.listdir(
            obfuscated_path) if os.path.isfile(os.path.join(obfuscated_path, f))]

    transformed_files = minified_files + obfuscated_files
    # Get as much transformed files as normal files
    transformed_files = random.sample(transformed_files, len(normal_files))

    for label, tfile in transformed_files:
        _, name = os.path.split(tfile)
        shutil.copyfile(tfile, os.path.join(
            transformed_path, "_".join([label, name])))

    # Generate PDGs for transformed JS-files
    logger.info("JStap: generating PDGs for %s", transformed_path)
    try:
        subprocess.run(
            [
                'python3',
                '-c',
                "from pdgs_generation import *; store_pdg_folder('{0}')".format(transformed_path)],
            cwd=os.path.join(
                SRC_PATH,
                '../JStap/pdg_generation'),
            check=True)
    except Exception as e:
        logger.critical(
            "PDG generation failed with:\n %s",
            e)
    # Split PDGs in train & validate set
    os.mkdir(normal_pdg_path_validate)
    normal_pdg = [
        os.path.join(
            normal_pdg_path,
            f) for f in os.listdir(normal_pdg_path) if os.path.isfile(
            os.path.join(
                normal_pdg_path,
                f))]
    normal_pdg_validate = random.sample(normal_pdg, int(len(normal_pdg) / 2))
    for nfile in normal_pdg_validate:
        _, name = os.path.split(nfile)
        shutil.move(nfile, os.path.join(normal_pdg_path_validate, name))

    os.mkdir(transformed_pdg_validate_path)
    transformed_pdg = [
        os.path.join(
            transformed_pdg_path,
            f) for f in os.listdir(transformed_pdg_path) if os.path.isfile(
            os.path.join(
                transformed_pdg_path,
                f))]
    transformed_pdg_validate = random.sample(
        transformed_pdg, int(len(transformed_pdg) / 2))
    for tfile in transformed_pdg_validate:
        _, name = os.path.split(tfile)
        shutil.move(tfile, os.path.join(transformed_pdg_validate_path, name))

    # Start training the JStap-model
    train_JStap(normal_pdg_path, normal_pdg_path_validate)
