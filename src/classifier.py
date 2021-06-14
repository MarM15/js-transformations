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

import time
import logging
from multiprocessing import Process, Queue
from backend import worker, get_scripts_for_folder_with_labels, \
    ml_classifier, translate_labels, ml_classifier_L2, predictor, arguments


def main():
    """ Classifies / predicts unknown JS-files """

    (input_file,
    ml_path,
    num_workers,
    only_level_1,
    evaluation,
    translation_file,
    error,
    move,
    logging_level,
    pdg_regen) = arguments()

    logger = logging.getLogger('obf_analysis')
    logger.setLevel(logging_level)

    queue_in = Queue()
    queue_arrays_l1 = Queue()
    queue_arrays_l2_trans = Queue()
    pred_queue = Queue()

    # Read file with directories + labels and fill queue
    get_scripts_for_folder_with_labels(
        input_file,
        queue_in,
        evaluation)

    for _ in range(num_workers):
        proc = Process(
            target=worker,
            args=(
                queue_in,
                queue_arrays_l1,
                queue_arrays_l2_trans,
                pred_queue,
                ml_path,
                "accuracy" if evaluation else "prediction",
                only_level_1,
                error,
                move))
        proc.start()

    # Each worker returns an array of features at the end. If there are N
    # arrays in queue  -> all are terminated
    while queue_arrays_l1.qsize() != num_workers and pred_queue.qsize() != num_workers:
        time.sleep(2)

    logger.info("all worker terminated")

    translation = translate_labels(translation_file)
    if evaluation:
        ml_classifier(ml_path, queue_arrays_l1)
        if not only_level_1:
            ml_classifier_L2(ml_path, queue_arrays_l2_trans)
        logger.info("Done\nResults are written to results.txt")
    else:
        predictor(pred_queue, ml_path, translation)
        logger.info("Done\nResults are written to prediction.txt")


if __name__ == "__main__":
    main()
