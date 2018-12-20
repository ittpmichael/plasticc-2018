# -*- coding: utf-8 -*-
#
#    ICRAR - International Centre for Radio Astronomy Research
#    (c) UWA - The University of Western Australia
#    Copyright by UWA (in the framework of the ICRAR)
#    All rights reserved
#
#    This library is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 2.1 of the License, or (at your option) any later version.
#
#    This library is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with this library; if not, write to the Free Software
#    Foundation, Inc., 59 Temple Place, Suite 330, Boston,
#    MA 02111-1307  USA
#
"""

"""

from collections import OrderedDict

DATA_TRAINING_SET_METADATA_CSV = '../data/training_set_metadata.csv'
DATA_TRAINING_SET_METADATA_NORMED_CSV = '../data/training_set_metadata_normed.csv'
DATA_TRAINING_SET_METADATA_B_NORMED_CSV = '../data/training_set_metadata_b_normed.csv'
DATA_TRAINING_SET_CSV = '../data/training_set.csv'
DATA_TRAINING_SET_NORMED_CSV = '../data/training_set_normed.csv'
DATA_TRAINING_SET_B_NORMED_CSV = '../data/training_set_b_normed.csv'
DATA_TRAINING_SET_METADATA_STATS_CSV = '../data/training_set_metadata_stats.csv'
NUMBER_CHANNELS, NUMBER_OF_CLASSES = 6, 14
CLASSES_DICT = OrderedDict([(6, 0), (15, 1), (16, 2), (42, 3), (52, 4),
                            (53, 5), (62, 6), (64, 7), (65, 8), (67, 9),
                            (88, 10), (90, 11), (92, 12), (95, 13)])
passbands_dict = OrderedDict([(0, 'u'), (1, 'g'), (2, 'r'),
                              (3, 'i'), (4, 'z'), (5, 'y'),])