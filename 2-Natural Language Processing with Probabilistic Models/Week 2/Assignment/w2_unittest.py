from utils_pos import get_word_tag, preprocess
import pandas as pd
from collections import defaultdict
import math
import numpy as np
import pickle


def test_create_dictionaries(target, training_corpus, vocab):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_case",
            "input": {
                "training_corpus": training_corpus,
                "vocab": vocab,
                "verbose": False,
            },
            "expected": {
                "len_emission_counts": 31140,
                "len_transition_counts": 1421,
                "len_tag_counts": 46,
                "emission_counts": {
                    ("DT", "the"): 41098,
                    ("NNP", "--unk_upper--"): 4635,
                    ("NNS", "Arts"): 2,
                },
                "transition_counts": {
                    ("VBN", "TO"): 2142,
                    ("CC", "IN"): 1227,
                    ("VBN", "JJR"): 66,
                },
                "tag_counts": {"PRP": 17436, "UH": 97, ")": 1376,},
            },
        },
        {
            "name": "small_case",
            "input": {
                "training_corpus": training_corpus[:1000],
                "vocab": vocab,
                "verbose": False,
            },
            "expected": {
                "len_emission_counts": 442,
                "len_transition_counts": 272,
                "len_tag_counts": 38,
                "emission_counts": {
                    ("DT", "the"): 48,
                    ("NNP", "--unk_upper--"): 9,
                    ("NNS", "Arts"): 1,
                },
                "transition_counts": {
                    ("VBN", "TO"): 3,
                    ("CC", "IN"): 2,
                    ("VBN", "JJR"): 1,
                },
                "tag_counts": {"PRP": 11, "UH": 0, ")": 2,},
            },
        },
    ]

    for test_case in test_cases:
        result_emission, result_transition, result_tag = target(**test_case["input"])

        # emission dictionary
        try:
            assert isinstance(result_emission, defaultdict)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": defaultdict,
                    "got": type(result_emission),
                }
            )
            print(
                f"Wrong output type for emission_counts dictionary.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert len(result_emission) == test_case["expected"]["len_emission_counts"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["len_emission_counts"],
                    "got": len(result_emission),
                }
            )
            print(
                f"Wrong output length for emission_counts dictionary.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            for k, v in test_case["expected"]["emission_counts"].items():
                assert np.isclose(result_emission[k], v)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["emission_counts"],
                    "got": result_emission,
                }
            )
            print(
                f"Wrong output values for emission_counts dictionary.\n\t Expected: {failed_cases[-1].get('expected')}."
            )

        # transition dictionary
        try:
            assert isinstance(result_transition, defaultdict)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": defaultdict,
                    "got": type(result_transition),
                }
            )
            print(
                f"Wrong output type for transition_counts dictionary.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert (
                len(result_transition) == test_case["expected"]["len_transition_counts"]
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["len_transition_counts"],
                    "got": len(result_transition),
                }
            )
            print(
                f"Wrong output length for transition_counts dictionary.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            for k, v in test_case["expected"]["transition_counts"].items():
                assert np.isclose(result_transition[k], v)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["transition_counts"],
                    "got": result_transition,
                }
            )
            print(
                f"Wrong output values for transition_counts dictionary.\n\t Expected: {failed_cases[-1].get('expected')}."
            )

        # tags count
        try:
            assert isinstance(result_tag, defaultdict)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": defaultdict,
                    "got": type(result_transition),
                }
            )
            print(
                f"Wrong output type for tag_counts dictionary.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert len(result_tag) == test_case["expected"]["len_tag_counts"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["len_tag_counts"],
                    "got": len(result_tag),
                }
            )
            print(
                f"Wrong output length for tag_counts dictionary.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            for k, v in test_case["expected"]["tag_counts"].items():
                assert np.isclose(result_tag[k], v)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["tag_counts"],
                    "got": result_tag,
                }
            )
            print(
                f"Wrong output values for tag_counts dictionary.\n\t Expected: {failed_cases[-1].get('expected')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_predict_pos(target, prep, y, emission_counts, vocab, states):

    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "prep": prep,
                "y": y,
                "emission_counts": emission_counts,
                "vocab": vocab,
                "states": states,
            },
            "expected": 0.8888563993099213,
        },
        {
            "name": "small_check",
            "input": {
                "prep": prep[:1000],
                "y": y[:1000],
                "emission_counts": emission_counts,
                "vocab": vocab,
                "states": states,
            },
            "expected": 0.876,
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])
        try:
            assert np.isclose(result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"],
                    "got": result,
                }
            )
            print(
                f"Wrong output values for tag_counts dictionary.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_create_transition_matrix(target, tag_counts, transition_counts):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "alpha": 0.001,
                "tag_counts": tag_counts,
                "transition_counts": transition_counts,
            },
            "expected": {
                "0:5": np.array(
                    [
                        [
                            7.03997297e-06,
                            7.03997297e-06,
                            7.03997297e-06,
                            7.03997297e-06,
                            7.03997297e-06,
                        ],
                        [
                            1.35647553e-07,
                            1.35647553e-07,
                            1.35647553e-07,
                            1.35647553e-07,
                            1.35647553e-07,
                        ],
                        [
                            1.44528595e-07,
                            1.44673124e-04,
                            6.93751711e-03,
                            6.79298851e-03,
                            5.05864537e-03,
                        ],
                        [
                            7.32039770e-07,
                            1.69101919e-01,
                            7.32039770e-07,
                            7.32039770e-07,
                            7.32039770e-07,
                        ],
                        [
                            7.26719892e-07,
                            7.27446612e-04,
                            7.26719892e-07,
                            7.27446612e-04,
                            7.26719892e-07,
                        ],
                    ]
                ),
                "30:35": np.array(
                    [
                        [
                            2.21706877e-06,
                            2.21706877e-06,
                            2.21706877e-06,
                            8.87049214e-03,
                            2.21706877e-06,
                        ],
                        [
                            3.75650909e-07,
                            7.51677469e-04,
                            3.75650909e-07,
                            5.10888993e-02,
                            3.75650909e-07,
                        ],
                        [
                            1.72277159e-05,
                            1.72277159e-05,
                            1.72277159e-05,
                            1.72277159e-05,
                            1.72277159e-05,
                        ],
                        [
                            4.47733569e-05,
                            4.47286283e-08,
                            4.47286283e-08,
                            8.95019852e-05,
                            4.47733569e-05,
                        ],
                        [
                            1.03043917e-05,
                            1.03043917e-05,
                            1.03043917e-05,
                            6.18366548e-02,
                            3.09234796e-02,
                        ],
                    ]
                ),
            },
        },
        {
            "name": "alpha_check",
            "input": {
                "alpha": 0.05,
                "tag_counts": tag_counts,
                "transition_counts": transition_counts,
            },
            "expected": {
                "0:5": np.array(
                    [
                        [
                            3.46500347e-04,
                            3.46500347e-04,
                            3.46500347e-04,
                            3.46500347e-04,
                            3.46500347e-04,
                        ],
                        [
                            6.78030457e-06,
                            6.78030457e-06,
                            6.78030457e-06,
                            6.78030457e-06,
                            6.78030457e-06,
                        ],
                        [
                            7.22407640e-06,
                            1.51705604e-04,
                            6.94233742e-03,
                            6.79785589e-03,
                            5.06407756e-03,
                        ],
                        [
                            3.65416941e-05,
                            1.68859168e-01,
                            3.65416941e-05,
                            3.65416941e-05,
                            3.65416941e-05,
                        ],
                        [
                            3.62765726e-05,
                            7.61808024e-04,
                            3.62765726e-05,
                            7.61808024e-04,
                            3.62765726e-05,
                        ],
                    ]
                ),
                "30:35": np.array(
                    [
                        [
                            1.10302228e-04,
                            1.10302228e-04,
                            1.10302228e-04,
                            8.93448048e-03,
                            1.10302228e-04,
                        ],
                        [
                            1.87666554e-05,
                            7.69432872e-04,
                            1.87666554e-05,
                            5.10640694e-02,
                            1.87666554e-05,
                        ],
                        [
                            8.29187396e-04,
                            8.29187396e-04,
                            8.29187396e-04,
                            8.29187396e-04,
                            8.29187396e-04,
                        ],
                        [
                            4.69603252e-05,
                            2.23620596e-06,
                            2.23620596e-06,
                            9.16844445e-05,
                            4.69603252e-05,
                        ],
                        [
                            5.03524673e-04,
                            5.03524673e-04,
                            5.03524673e-04,
                            6.09264854e-02,
                            3.07150050e-02,
                        ],
                    ]
                ),
            },
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

    try:
        assert isinstance(result, np.ndarray)
        successful_cases += 1
    except:
        failed_cases.append(
            {"name": test_case["name"], "expected": np.ndarray, "got": type(result),}
        )
        print(
            f"Wrong output type .\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
        )

    try:
        assert np.allclose(result[0:5, 0:5], test_case["expected"]["0:5"])
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": test_case["name"],
                "expected": test_case["expected"]["0:5"],
                "got": result[0:5, 0:5],
            }
        )
        print(
            f"Wrong output values in rows and columns with indexes between 0 and 5.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
        )

    try:
        assert np.allclose(result[30:35, 30:35], test_case["expected"]["30:35"])
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": test_case["name"],
                "expected": test_case["expected"]["30:35"],
                "got": result[30:35, 30:35],
            }
        )
        print(
            f"Wrong output values in rows and columns with indexes between 30 and 35.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
        )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_create_emission_matrix(target, tag_counts, emission_counts, vocab):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "alpha": 0.001,
                "tag_counts": tag_counts,
                "emission_counts": emission_counts,
                "vocab": vocab,
            },
            "expected": {
                "0:5": np.array(
                    [
                        [
                            6.03219988e-06,
                            6.03219988e-06,
                            8.56578416e-01,
                            6.03219988e-06,
                            6.03219988e-06,
                        ],
                        [
                            1.35212298e-07,
                            1.35212298e-07,
                            1.35212298e-07,
                            9.71365280e-01,
                            1.35212298e-07,
                        ],
                        [
                            1.44034584e-07,
                            1.44034584e-07,
                            1.44034584e-07,
                            1.44034584e-07,
                            1.44034584e-07,
                        ],
                        [
                            7.19539897e-07,
                            7.19539897e-07,
                            7.19539897e-07,
                            7.19539897e-07,
                            7.19539897e-07,
                        ],
                        [
                            7.14399508e-07,
                            7.14399508e-07,
                            7.14399508e-07,
                            7.14399508e-07,
                            7.14399508e-07,
                        ],
                    ]
                ),
                "30:35": np.array(
                    [
                        [
                            2.10625199e-06,
                            2.10625199e-06,
                            2.10625199e-06,
                            2.10625199e-06,
                            2.10625199e-06,
                        ],
                        [
                            3.72331731e-07,
                            3.72331731e-07,
                            3.72331731e-07,
                            3.72331731e-07,
                            3.72331731e-07,
                        ],
                        [
                            1.22283772e-05,
                            1.22406055e-02,
                            1.22283772e-05,
                            1.22283772e-05,
                            1.22283772e-05,
                        ],
                        [
                            4.46812012e-08,
                            4.46812012e-08,
                            4.46812012e-08,
                            4.46812012e-08,
                            4.46812012e-08,
                        ],
                        [
                            8.27972213e-06,
                            4.96866125e-02,
                            8.27972213e-06,
                            8.27972213e-06,
                            8.27972213e-06,
                        ],
                    ]
                ),
            },
        },
        {
            "name": "alpha_check",
            "input": {
                "alpha": 0.05,
                "tag_counts": tag_counts,
                "emission_counts": emission_counts,
                "vocab": vocab,
            },
            "expected": {
                "0:5": np.array(
                    [
                        [
                            3.75699741e-05,
                            3.75699741e-05,
                            1.06736296e-01,
                            3.75699741e-05,
                            3.75699741e-05,
                        ],
                        [
                            5.84054154e-06,
                            5.84054154e-06,
                            5.84054154e-06,
                            8.39174848e-01,
                            5.84054154e-06,
                        ],
                        [
                            6.16686298e-06,
                            6.16686298e-06,
                            6.16686298e-06,
                            6.16686298e-06,
                            6.16686298e-06,
                        ],
                        [
                            1.95706206e-05,
                            1.95706206e-05,
                            1.95706206e-05,
                            1.95706206e-05,
                            1.95706206e-05,
                        ],
                        [
                            1.94943174e-05,
                            1.94943174e-05,
                            1.94943174e-05,
                            1.94943174e-05,
                            1.94943174e-05,
                        ],
                    ]
                ),
                "30:35": np.array(
                    [
                        [
                            3.04905937e-05,
                            3.04905937e-05,
                            3.04905937e-05,
                            3.04905937e-05,
                            3.04905937e-05,
                        ],
                        [
                            1.29841464e-05,
                            1.29841464e-05,
                            1.29841464e-05,
                            1.29841464e-05,
                            1.29841464e-05,
                        ],
                        [
                            4.01010547e-05,
                            8.42122148e-04,
                            4.01010547e-05,
                            4.01010547e-05,
                            4.01010547e-05,
                        ],
                        [
                            2.12351646e-06,
                            2.12351646e-06,
                            2.12351646e-06,
                            2.12351646e-06,
                            2.12351646e-06,
                        ],
                        [
                            3.88847844e-05,
                            4.70505891e-03,
                            3.88847844e-05,
                            3.88847844e-05,
                            3.88847844e-05,
                        ],
                    ]
                ),
            },
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert isinstance(result, np.ndarray)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": np.ndarray,
                    "got": type(result),
                }
            )
            print(
                f"Wrong output type .\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result[0:5, 0:5], test_case["expected"]["0:5"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["0:5"],
                    "got": result[0:5, 0:5],
                }
            )
            print(
                f"Wrong output values in rows and columns with indexes between 0 and 5.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result[30:35, 30:35], test_case["expected"]["30:35"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["30:35"],
                    "got": result[30:35, 30:35],
                }
            )
            print(
                f"Wrong output values in rows and columns with indexes between 30 and 35.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_initialize(target, states, tag_counts, A, B, corpus, vocab):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "states": states,
                "tag_counts": tag_counts,
                "A": A,
                "B": B,
                "corpus": corpus,
                "vocab": vocab,
            },
            "expected": {
                "best_probs_shape": (46, 34199),
                "best_paths_shape": (46, 34199),
                "best_probs_col0": np.array(
                    [
                        -22.60982633,
                        -23.07660654,
                        -23.57298822,
                        -19.76726066,
                        -24.74325104,
                        -35.20241402,
                        -35.00096024,
                        -34.99203854,
                        -21.35069072,
                        -19.85767814,
                        -21.92098414,
                        -4.01623741,
                        -19.16380593,
                        -21.1062242,
                        -20.47163973,
                        -21.10157273,
                        -21.49584851,
                        -20.4811853,
                        -18.25856307,
                        -23.39717471,
                        -21.92146798,
                        -9.41377777,
                        -21.03053445,
                        -21.08029591,
                        -20.10863677,
                        -33.48185979,
                        -19.47301382,
                        -20.77150242,
                        -20.11727696,
                        -20.56031676,
                        -20.57193964,
                        -32.30366295,
                        -18.07551522,
                        -22.58887909,
                        -19.1585905,
                        -16.02994331,
                        -24.30968545,
                        -20.92932218,
                        -21.96797222,
                        -24.29571895,
                        -23.45968569,
                        -22.43665883,
                        -20.46568904,
                        -22.75551606,
                        -19.6637215,
                        -18.36288463,
                    ]
                ),
            },
        }
    ]

    for test_case in test_cases:
        result_best_probs, result_best_paths = target(**test_case["input"])

        try:
            assert isinstance(result_best_probs, np.ndarray)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]) + "index 0",
                    "expected": np.ndarray,
                    "got": type(result_best_probs),
                }
            )
            print(
                f"Wrong output type .\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert isinstance(result_best_paths, np.ndarray)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]) + "index 1",
                    "expected": np.ndarray,
                    "got": type(result_best_paths),
                }
            )
            print(
                f"Wrong output type .\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert result_best_probs.shape == test_case["expected"]["best_probs_shape"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]),
                    "expected": test_case["expected"]["best_probs_shape"],
                    "got": result_best_probs.shape,
                }
            )
            print(
                f"Wrong output shape for best_probs.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert result_best_paths.shape == test_case["expected"]["best_paths_shape"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]),
                    "expected": test_case["expected"]["best_paths_shape"],
                    "got": result_best_paths.shape,
                }
            )
            print(
                f"Wrong output shape for best_paths.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                result_best_probs[:, 0], test_case["expected"]["best_probs_col0"]
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]),
                    "expected": test_case["expected"]["best_probs_col0"],
                    "got": result_best_probs[:, 0],
                }
            )
            print(
                f"Wrong non-zero values for best_probs.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.all((result_best_paths == 0))
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]),
                    "expected": "Array of zeros with shape (46, 34199)",
                }
            )
            print(
                f"Wrong values for best_paths.\n\t Expected: {failed_cases[-1].get('expected')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_viterbi_forward(target, A, B, test_corpus, vocab):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "A": A,
                "B": B,
                "test_corpus": test_corpus,
                "best_probs": pickle.load(
                    open("./support_files/best_probs_initilized.pkl", "rb")
                ),
                "best_paths": pickle.load(
                    open("./support_files/best_paths_initilized.pkl", "rb")
                ),
                "vocab": vocab,
                "verbose": False,
            },
            "expected": {
                "best_probs0:5": np.array(
                    [
                        [
                            -22.60982633,
                            -24.78215633,
                            -34.08246498,
                            -34.34107105,
                            -49.56012613,
                        ],
                        [
                            -23.07660654,
                            -24.51583896,
                            -35.04774303,
                            -35.28281026,
                            -50.52540418,
                        ],
                        [
                            -23.57298822,
                            -29.98305064,
                            -31.98004656,
                            -38.99187549,
                            -47.45770771,
                        ],
                        [
                            -19.76726066,
                            -25.7122143,
                            -31.54577612,
                            -37.38331695,
                            -47.02343727,
                        ],
                        [
                            -24.74325104,
                            -28.78696025,
                            -31.458494,
                            -36.00456711,
                            -46.93615515,
                        ],
                    ]
                ),
                "best_probs30:35": np.array(
                    [
                        [
                            -202.75618827,
                            -208.38838519,
                            -210.46938402,
                            -210.15943098,
                            -223.79223672,
                        ],
                        [
                            -202.58297597,
                            -217.72266765,
                            -207.23725672,
                            -215.529735,
                            -224.13957203,
                        ],
                        [
                            -202.00878092,
                            -214.23093833,
                            -217.41021623,
                            -220.73768708,
                            -222.03338753,
                        ],
                        [
                            -200.44016117,
                            -209.46937757,
                            -209.06951664,
                            -216.22297765,
                            -221.09669653,
                        ],
                        [
                            -208.74189499,
                            -214.62088817,
                            -209.79346523,
                            -213.52623459,
                            -228.70417526,
                        ],
                    ]
                ),
                "best_paths0:5": np.array(
                    [
                        [0, 11, 20, 25, 20],
                        [0, 11, 20, 25, 20],
                        [0, 11, 20, 25, 20],
                        [0, 11, 20, 25, 20],
                        [0, 11, 20, 25, 20],
                    ]
                ),
                "best_paths30:35": np.array(
                    [
                        [20, 19, 35, 11, 21],
                        [20, 19, 35, 11, 21],
                        [20, 19, 35, 11, 21],
                        [20, 19, 35, 11, 21],
                        [35, 19, 35, 11, 34],
                    ]
                ),
            },
        }
    ]

    for test_case in test_cases:
        result_best_probs, result_best_paths = target(**test_case["input"])

        try:
            assert isinstance(result_best_probs, np.ndarray)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]) + "index 0",
                    "expected": np.ndarray,
                    "got": type(result_best_probs),
                }
            )
            print(
                f"Wrong output type .\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert isinstance(result_best_paths, np.ndarray)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]) + "index 1",
                    "expected": np.ndarray,
                    "got": type(result_best_paths),
                }
            )
            print(
                f"Wrong output type .\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                result_best_probs[0:5, 0:5], test_case["expected"]["best_probs0:5"]
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]),
                    "expected": test_case["expected"]["best_probs0:5"],
                    "got": result_best_probs[0:5, 0:5],
                }
            )
            print(
                f"Wrong values for best_probs.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                result_best_probs[30:35, 30:35],
                test_case["expected"]["best_probs30:35"],
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]),
                    "expected": test_case["expected"]["best_probs30:35"],
                    "got": result_best_probs[:, 0],
                }
            )
            print(
                f"Wrong values for best_probs.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                result_best_paths[0:5, 0:5], test_case["expected"]["best_paths0:5"],
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]),
                    "expected": test_case["expected"]["best_paths0:5"],
                    "got": result_best_paths[0:5, 0:5],
                }
            )
            print(
                f"Wrong values for best_paths.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                result_best_paths[30:35, 30:35],
                test_case["expected"]["best_paths30:35"],
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]),
                    "expected": test_case["expected"]["best_paths30:35"],
                    "got": result_best_paths[30:35, 30:35],
                }
            )
            print(
                f"Wrong values for best_paths.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_viterbi_backward(target, corpus, states):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "corpus": corpus,
                "best_probs": pickle.load(
                    open("./support_files/best_probs_trained.pkl", "rb")
                ),
                "best_paths": pickle.load(
                    open("./support_files/best_paths_trained.pkl", "rb")
                ),
                "states": states,
            },
            "expected": {
                "pred_len": 34199,
                "pred_head": [
                    "DT",
                    "NN",
                    "POS",
                    "NN",
                    "MD",
                    "VB",
                    "VBN",
                    "IN",
                    "JJ",
                    "NN",
                ],
                "pred_tail": [
                    "PRP",
                    "MD",
                    "RB",
                    "VB",
                    "PRP",
                    "RB",
                    "IN",
                    "PRP",
                    ".",
                    "--s--",
                ],
            },
        }
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert isinstance(result, list)
            successful_cases += 1
        except:
            failed_cases.append(
                {"name": str(test_case["name"]), "expected": list, "got": type(result)}
            )
            print(
                f"Wrong output type .\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert len(result) == test_case["expected"]["pred_len"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]),
                    "expected": test_case["expected"]["pred_len"],
                    "got": len(result),
                }
            )
            print(
                f"Wrong output lenght.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert result[:10] == test_case["expected"]["pred_head"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]),
                    "expected": test_case["expected"]["pred_head"],
                    "got": result[:10],
                }
            )
            print(
                f"Wrong values for pred list.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert result[-10:] == test_case["expected"]["pred_tail"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]),
                    "expected": test_case["expected"]["pred_tail"],
                    "got": result[-10:],
                }
            )
            print(
                f"Wrong values for pred list.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_compute_accuracy(target, pred, y):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {"pred": pred, "y": y},
            "expected": 0.953063647155511,
        },
        {
            "name": "small_check",
            "input": {"pred": pred[:100], "y": y[:100]},
            "expected": 0.979381443298969,
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert isinstance(result, float)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]),
                    "expected": float,
                    "got": type(result),
                }
            )
            print(
                f"Wrong output type.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.isclose(result, test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": str(test_case["name"]),
                    "expected": float,
                    "got": type(result),
                }
            )
            print(
                f"Wrong output type.\n\t Expected: {failed_cases[-1].get('expected')}.\n\t Got: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases

