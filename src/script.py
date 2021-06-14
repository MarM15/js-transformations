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

class script:
    """ Class to represent a single JS file. """

    def __init__(self, filename, label):
        self.filename = filename
        # Label = {"0":[], "1":[11,..], "2":[21,22,..]}
        self.label = label
        self.features = None
        self.date = None
        self.prediction_l1 = None
        self.prediction_l2 = None
        self.prediction_l1_proba = None
        self.prediction_l2_proba = None

    def add_features(self, features):
        """ Add the corresponding features. """
        self.features = features

    def get_features_obf(self):
        """ Return the corresponding features. """
        return self.features

    def get_labels_for(self, over_level):
        """ Returns Subclasslabels for given "Overclass"-Label.
         returns None if Script is not from the "Overclass". """

        if over_level in self.label:
            return self.label[over_level]
        return None

    def get_labels_for_level1(self):
        """ Returns all label for Level 1 (the "Overcalss"-Labels) as integer. """

        if self.label is None:
            return None
        return [int(key) for key, value in self.label.items()]

    def is_it_for_level1(self):
        """ Checks if script is for level 1 (learning phase). """

        if self.label is None:
            return False
        if "0" in self.label:
            if len(self.label["0"]) == 1 and self.label["0"][0] == 0:
                return True
        if "1" in self.label:
            if len(self.label["1"]) == 1 and self.label["1"][0] == 1:
                return True
        if "2" in self.label:
            if len(self.label["2"]) == 1 and self.label["2"][0] == 2:
                return True
        return False

    def get_filename(self):
        """ Retuns Path + Filename. """
        return self.filename

    def get_labels(self):
        """ Returns labels of the script. """
        return self.label

    def set_date(self, date):
        """ Add date to the script. """
        self.date = date

    def get_date(self):
        """ Return date of the script. """
        return self.date

    def set_prediction_l1(self, prediction_1):
        """ Add prediction on Level 1. """
        self.prediction_l1 = prediction_1

    def set_prediction_l2(self, prediction_l2):
        """ Add prediction on Level 2. """
        self.prediction_l2 = prediction_l2

    def get_prediction_l1(self):
        """ Returns prediction on Level 1. """
        return self.prediction_l1

    def get_prediction_l2(self):
        """ Returns prediction on Level 2. """
        return self.prediction_l2

    def set_prediction_l1_proba(self, prediction_l1_proba):
        """ Add probabilities for every label on Level 1. """
        self.prediction_l1_proba = prediction_l1_proba

    def set_prediction_l2_proba(self, prediction_l2_proba, classes):
        """ Add probabilities for every label on Level 2 as a dictionary """
        dic = {}
        for i, value in enumerate(prediction_l2_proba):
            if classes[i] not in dic:
                dic[classes[i]] = value
        self.prediction_l2_proba = dic

    def get_prediction_l1_proba(self):
        """ Retuns robabilities for every label on Level 1. """
        return self.prediction_l1_proba

    def get_prediction_l2_proba(self):
        """ Returns probabilities for every label on Level 2. """
        return self.prediction_l2_proba
