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
from backend import worker, get_scripts_for_folder_with_labels, ml_learner, \
    ml_learner_l2, arguments, handle_JStap


def main():
    """ Calculates features and trains models """

    (input_file,
    ml_path,
    num_workers,
    only_level_1,
    evaluation,
    _,
    error,
    move,
    logging_level,
    pdg_regen) = arguments()

    logger = logging.getLogger('obf_analysis')
    logger.setLevel(logging_level)

    queue_in = Queue()
    queue_arrays_l1 = Queue()
    queue_arrays_l2_trans = Queue()
    # Read file with directories + labels and fill queue
    get_scripts_for_folder_with_labels(
        input_file,
        queue_in,
        evaluation)

    handle_JStap(input_file, pdg_regen, num_workers)

    for _ in range(num_workers):  # Start n threads to calculate features
        proc = Process(
            target=worker,
            args=(
                queue_in,
                queue_arrays_l1,
                queue_arrays_l2_trans,
                None,
                ml_path,
                "learning",
                only_level_1,
                error,
                move))
        proc.start()

    # Each worker returns an array of features at the end. If there are N
    # arrays in queue  -> all are terminated
    while queue_arrays_l1.qsize() != num_workers:
        time.sleep(2)

    logger.info("all worker terminated")

    # train models
    ml_learner(ml_path, queue_arrays_l1, num_workers)
    if not only_level_1:
        ml_learner_l2(ml_path, queue_arrays_l2_trans, num_workers)

    logger.info("Done\nModels are stored in {0}".format(ml_path))


if __name__ == "__main__":
    main()
