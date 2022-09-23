import pickle
from numpy.lib.shape_base import expand_dims
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
from collections import Counter
from utils2 import sigmoid, get_batches, compute_pca, get_dict


def test_initialize_model(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {"N": 4, "V": 10, "random_seed": 1},
            "expected": {
                "w1": np.array(
                    [
                        [
                            4.17022005e-01,
                            7.20324493e-01,
                            1.14374817e-04,
                            3.02332573e-01,
                            1.46755891e-01,
                            9.23385948e-02,
                            1.86260211e-01,
                            3.45560727e-01,
                            3.96767474e-01,
                            5.38816734e-01,
                        ],
                        [
                            4.19194514e-01,
                            6.85219500e-01,
                            2.04452250e-01,
                            8.78117436e-01,
                            2.73875932e-02,
                            6.70467510e-01,
                            4.17304802e-01,
                            5.58689828e-01,
                            1.40386939e-01,
                            1.98101489e-01,
                        ],
                        [
                            8.00744569e-01,
                            9.68261576e-01,
                            3.13424178e-01,
                            6.92322616e-01,
                            8.76389152e-01,
                            8.94606664e-01,
                            8.50442114e-02,
                            3.90547832e-02,
                            1.69830420e-01,
                            8.78142503e-01,
                        ],
                        [
                            9.83468338e-02,
                            4.21107625e-01,
                            9.57889530e-01,
                            5.33165285e-01,
                            6.91877114e-01,
                            3.15515631e-01,
                            6.86500928e-01,
                            8.34625672e-01,
                            1.82882773e-02,
                            7.50144315e-01,
                        ],
                    ]
                ),
                "w2": np.array(
                    [
                        [0.98886109, 0.74816565, 0.28044399, 0.78927933],
                        [0.10322601, 0.44789353, 0.9085955, 0.29361415],
                        [0.28777534, 0.13002857, 0.01936696, 0.67883553],
                        [0.21162812, 0.26554666, 0.49157316, 0.05336255],
                        [0.57411761, 0.14672857, 0.58930554, 0.69975836],
                        [0.10233443, 0.41405599, 0.69440016, 0.41417927],
                        [0.04995346, 0.53589641, 0.66379465, 0.51488911],
                        [0.94459476, 0.58655504, 0.90340192, 0.1374747],
                        [0.13927635, 0.80739129, 0.39767684, 0.1653542],
                        [0.92750858, 0.34776586, 0.7508121, 0.72599799],
                    ]
                ),
                "b1": np.array(
                    [[0.88330609], [0.62367221], [0.75094243], [0.34889834]]
                ),
                "b2": np.array(
                    [
                        [0.26992789],
                        [0.89588622],
                        [0.42809119],
                        [0.96484005],
                        [0.6634415],
                        [0.62169572],
                        [0.11474597],
                        [0.94948926],
                        [0.44991213],
                        [0.57838961],
                    ]
                ),
            },
        },
        {
            "name": "smaller_check",
            "input": {"N": 2, "V": 3, "random_seed": 1},
            "expected": {
                "w1": np.array(
                    [
                        [4.17022005e-01, 7.20324493e-01, 1.14374817e-04],
                        [3.02332573e-01, 1.46755891e-01, 9.23385948e-02],
                    ]
                ),
                "w2": np.array(
                    [
                        [0.18626021, 0.34556073],
                        [0.39676747, 0.53881673],
                        [0.41919451, 0.6852195],
                    ]
                ),
                "b1": np.array([[0.20445225], [0.87811744]]),
                "b2": np.array([[0.02738759], [0.67046751], [0.4173048]]),
            },
        },
        {
            "name": "small_check",
            "input": {"N": 3, "V": 5, "random_seed": 5},
            "expected": {
                "w1": np.array(
                    [
                        [0.22199317, 0.87073231, 0.20671916, 0.91861091, 0.48841119],
                        [0.61174386, 0.76590786, 0.51841799, 0.2968005, 0.18772123],
                        [0.08074127, 0.7384403, 0.44130922, 0.15830987, 0.87993703],
                    ]
                ),
                "w2": np.array(
                    [
                        [0.27408646, 0.41423502, 0.29607993],
                        [0.62878791, 0.57983781, 0.5999292],
                        [0.26581912, 0.28468588, 0.25358821],
                        [0.32756395, 0.1441643, 0.16561286],
                        [0.96393053, 0.96022672, 0.18841466],
                    ]
                ),
                "b1": np.array([[0.02430656], [0.20455555], [0.69984361]]),
                "b2": np.array(
                    [
                        [0.77951459],
                        [0.02293309],
                        [0.57766286],
                        [0.00164217],
                        [0.51547261],
                    ]
                ),
            },
        },
    ]

    for test_case in test_cases:
        tmp_W1, tmp_W2, tmp_b1, tmp_b2 = target(**test_case["input"])

        try:
            assert isinstance(tmp_W1, np.ndarray)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": type(test_case["expected"]["w1"]),
                    "got": type(tmp_W1),
                }
            )
            print(
                f"Wrong type for W1 matrix.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert isinstance(tmp_W2, np.ndarray)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": type(test_case["expected"]["w2"]),
                    "got": type(tmp_W2),
                }
            )
            print(
                f"Wrong type for W2 matrix.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert isinstance(tmp_b1, np.ndarray)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": type(test_case["expected"]["b1"]),
                    "got": type(tmp_b1),
                }
            )
            print(
                f"Wrong type for b1 vector.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert isinstance(tmp_b2, np.ndarray)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": type(test_case["expected"]["b2"]),
                    "got": type(tmp_b2),
                }
            )
            print(
                f"Wrong type for b2 vector.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        # Test initialization
        try:
            assert np.allclose(
                tmp_W1, test_case["expected"]["w1"], rtol=1e-05, atol=1e-08
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["w1"],
                    "got": tmp_W1,
                }
            )
            print(
                f"Wrong initialization for W1 matrix. Check the use of the random seed.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                tmp_W2, test_case["expected"]["w2"], rtol=1e-05, atol=1e-08
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["w2"],
                    "got": tmp_W2,
                }
            )
            print(
                f"Wrong initialization for W2 matrix. Check the use of the random seed.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                tmp_b1, test_case["expected"]["b1"], rtol=1e-05, atol=1e-08
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["b1"],
                    "got": tmp_b1,
                }
            )
            print(
                f"Wrong initialization for b1 vector. Check the use of the random seed.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(
                tmp_b2, test_case["expected"]["b2"], rtol=1e-05, atol=1e-08
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["b2"],
                    "got": tmp_b2,
                }
            )
            print(
                f"Wrong initialization for b2 vector. Check the use of the random seed.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        # Testing shapes
        try:
            assert tmp_W1.shape == test_case["expected"]["w1"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["w1"].shape,
                    "got": tmp_W1.shape,
                }
            )
            print(
                f"Wrong shape for W1 matrix.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert tmp_W2.shape == test_case["expected"]["w2"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["w2"].shape,
                    "got": tmp_W2.shape,
                }
            )
            print(
                f"Wrong shape for W2 matrix.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert tmp_b1.shape == test_case["expected"]["b1"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["b1"].shape,
                    "got": tmp_b1.shape,
                }
            )
            print(
                f"Wrong shape for b1 vector.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert tmp_b2.shape == test_case["expected"]["b2"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["b2"].shape,
                    "got": tmp_b2.shape,
                }
            )
            print(
                f"Wrong shape for b2 vector.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_softmax(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": np.array([[1, 2, 3], [1, 1, 1]]),
            "expected": np.array(
                [[0.5, 0.73105858, 0.88079708], [0.5, 0.26894142, 0.11920292]]
            ),
        },
        {
            "name": "larger_check",
            "input": np.array(
                [[1, 2, 3, 4], [1, 1, 1, 1], [-1, 0, 0, 3], [9, 8, 7, 6], [5, 4, 3, 2]]
            ),
            "expected": np.array(
                [
                    [3.29197356e-04, 2.42529448e-03, 1.76108202e-02, 1.11831081e-01],
                    [3.29197356e-04, 8.92215977e-04, 2.38336534e-03, 5.56774167e-03],
                    [4.45520174e-05, 3.28227915e-04, 8.76791109e-04, 4.11403556e-02],
                    [9.81323487e-01, 9.78433625e-01, 9.61518203e-01, 8.26326131e-01],
                    [1.79735666e-02, 1.79206369e-02, 1.76108202e-02, 1.51346910e-02],
                ]
            ),
        },
    ]

    for test_case in test_cases:
        result = target(test_case["input"])

        try:
            assert isinstance(result, np.ndarray)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": type(test_case["expected"]),
                    "got": type(result),
                }
            )
            print(
                f"Wrong output type for softmax function.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result, test_case["expected"])
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
                f"Wrong output for softmax function.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert result.shape, test_case["expected"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"].shape,
                    "got": result.shape,
                }
            )
            print(
                f"Wrong shape for softmax function. Check the axis when you perform the sum in the denominator.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_forward_prop(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "x": np.array([[0, 1, 0]]).T,
                "W1": np.array(
                    [
                        [4.17022005e-01, 7.20324493e-01, 1.14374817e-04],
                        [3.02332573e-01, 1.46755891e-01, 9.23385948e-02],
                    ]
                ),
                "W2": np.array(
                    [
                        [0.18626021, 0.34556073],
                        [0.39676747, 0.53881673],
                        [0.41919451, 0.6852195],
                    ]
                ),
                "b1": np.array([[0.20445225], [0.87811744]]),
                "b2": np.array([[0.02738759], [0.67046751], [0.4173048]]),
            },
            "expected": (
                np.array([[0.55379268], [1.58960774], [1.50722933]]),
                np.array([[0.92477674], [1.02487333]]),
            ),
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert isinstance(result[0], np.ndarray)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": type(test_case["expected"][0]),
                    "got": type(result[0]),
                }
            )
            print(
                f"Wrong type for z vector.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert isinstance(result[1], np.ndarray)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": type(test_case["expected"][1]),
                    "got": type(result[1]),
                }
            )
            print(
                f"Wrong type for h vector.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        # Test values
        try:
            assert np.allclose(result[0], test_case["expected"][0])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"][0],
                    "got": result[0],
                }
            )
            print(
                f"Wrong output values for z vector.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result[1], test_case["expected"][1])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"][1],
                    "got": result[1],
                }
            )
            print(
                f"Wrong type for h vector.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        # Test shapes
        try:
            assert result[0].shape == test_case["expected"][0].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"][0].shape,
                    "got": result[0].shape,
                }
            )
            print(
                f"Wrong output shape for z vector.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert result[1].shape == test_case["expected"][1].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"][1].shape,
                    "got": result[1].shape,
                }
            )
            print(
                f"Wrong output shape for h vector.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_back_prop(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "x": np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                "yhat": np.array(
                    [
                        [0.09488382, 0.09488382, 0.09488382, 0.09488382],
                        [0.07354783, 0.07354783, 0.07354783, 0.07354783],
                        [0.04061033, 0.04061033, 0.04061033, 0.04061033],
                        [0.11081115, 0.11081115, 0.11081115, 0.11081115],
                        [0.06029008, 0.06029008, 0.06029008, 0.06029008],
                        [0.15000559, 0.15000559, 0.15000559, 0.15000559],
                        [0.12740163, 0.12740163, 0.12740163, 0.12740163],
                        [0.14957542, 0.14957542, 0.14957542, 0.14957542],
                        [0.1007054, 0.1007054, 0.1007054, 0.1007054],
                        [0.09216876, 0.09216876, 0.09216876, 0.09216876],
                    ]
                ),
                "y": np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                "h": np.array(
                    [
                        [0.8118587, 0.8118587, 0.8118587, 0.8118587],
                        [0.87496164, 0.87496164, 0.87496164, 0.87496164],
                        [0.68841325, 0.68841325, 0.68841325, 0.68841325],
                        [0.56949441, 0.56949441, 0.56949441, 0.56949441],
                        [0.16097144, 0.16097144, 0.16097144, 0.16097144],
                        [0.46688002, 0.46688002, 0.46688002, 0.46688002],
                        [0.34517205, 0.34517205, 0.34517205, 0.34517205],
                        [0.22503996, 0.22503996, 0.22503996, 0.22503996],
                        [0.59251187, 0.59251187, 0.59251187, 0.59251187],
                        [0.31226984, 0.31226984, 0.31226984, 0.31226984],
                        [0.91630555, 0.91630555, 0.91630555, 0.91630555],
                        [0.90963552, 0.90963552, 0.90963552, 0.90963552],
                        [0.25711829, 0.25711829, 0.25711829, 0.25711829],
                        [0.1108913, 0.1108913, 0.1108913, 0.1108913],
                        [0.19296273, 0.19296273, 0.19296273, 0.19296273],
                    ]
                ),
                "W1": np.array(
                    [
                        [
                            4.17022005e-01,
                            7.20324493e-01,
                            1.14374817e-04,
                            3.02332573e-01,
                            1.46755891e-01,
                            9.23385948e-02,
                            1.86260211e-01,
                            3.45560727e-01,
                            3.96767474e-01,
                            5.38816734e-01,
                        ],
                        [
                            4.19194514e-01,
                            6.85219500e-01,
                            2.04452250e-01,
                            8.78117436e-01,
                            2.73875932e-02,
                            6.70467510e-01,
                            4.17304802e-01,
                            5.58689828e-01,
                            1.40386939e-01,
                            1.98101489e-01,
                        ],
                        [
                            8.00744569e-01,
                            9.68261576e-01,
                            3.13424178e-01,
                            6.92322616e-01,
                            8.76389152e-01,
                            8.94606664e-01,
                            8.50442114e-02,
                            3.90547832e-02,
                            1.69830420e-01,
                            8.78142503e-01,
                        ],
                        [
                            9.83468338e-02,
                            4.21107625e-01,
                            9.57889530e-01,
                            5.33165285e-01,
                            6.91877114e-01,
                            3.15515631e-01,
                            6.86500928e-01,
                            8.34625672e-01,
                            1.82882773e-02,
                            7.50144315e-01,
                        ],
                        [
                            9.88861089e-01,
                            7.48165654e-01,
                            2.80443992e-01,
                            7.89279328e-01,
                            1.03226007e-01,
                            4.47893526e-01,
                            9.08595503e-01,
                            2.93614148e-01,
                            2.87775339e-01,
                            1.30028572e-01,
                        ],
                        [
                            1.93669579e-02,
                            6.78835533e-01,
                            2.11628116e-01,
                            2.65546659e-01,
                            4.91573159e-01,
                            5.33625451e-02,
                            5.74117605e-01,
                            1.46728575e-01,
                            5.89305537e-01,
                            6.99758360e-01,
                        ],
                        [
                            1.02334429e-01,
                            4.14055988e-01,
                            6.94400158e-01,
                            4.14179270e-01,
                            4.99534589e-02,
                            5.35896406e-01,
                            6.63794645e-01,
                            5.14889112e-01,
                            9.44594756e-01,
                            5.86555041e-01,
                        ],
                        [
                            9.03401915e-01,
                            1.37474704e-01,
                            1.39276347e-01,
                            8.07391289e-01,
                            3.97676837e-01,
                            1.65354197e-01,
                            9.27508580e-01,
                            3.47765860e-01,
                            7.50812103e-01,
                            7.25997985e-01,
                        ],
                        [
                            8.83306091e-01,
                            6.23672207e-01,
                            7.50942434e-01,
                            3.48898342e-01,
                            2.69927892e-01,
                            8.95886218e-01,
                            4.28091190e-01,
                            9.64840047e-01,
                            6.63441498e-01,
                            6.21695720e-01,
                        ],
                        [
                            1.14745973e-01,
                            9.49489259e-01,
                            4.49912133e-01,
                            5.78389614e-01,
                            4.08136803e-01,
                            2.37026980e-01,
                            9.03379521e-01,
                            5.73679487e-01,
                            2.87032703e-03,
                            6.17144914e-01,
                        ],
                        [
                            3.26644902e-01,
                            5.27058102e-01,
                            8.85942099e-01,
                            3.57269760e-01,
                            9.08535151e-01,
                            6.23360116e-01,
                            1.58212428e-02,
                            9.29437234e-01,
                            6.90896918e-01,
                            9.97322850e-01,
                        ],
                        [
                            1.72340508e-01,
                            1.37135750e-01,
                            9.32595463e-01,
                            6.96818161e-01,
                            6.60001727e-02,
                            7.55463053e-01,
                            7.53876188e-01,
                            9.23024536e-01,
                            7.11524759e-01,
                            1.24270962e-01,
                        ],
                        [
                            1.98801338e-02,
                            2.62109869e-02,
                            2.83064880e-02,
                            2.46211068e-01,
                            8.60027949e-01,
                            5.38831064e-01,
                            5.52821979e-01,
                            8.42030892e-01,
                            1.24173315e-01,
                            2.79183679e-01,
                        ],
                        [
                            5.85759271e-01,
                            9.69595748e-01,
                            5.61030219e-01,
                            1.86472894e-02,
                            8.00632673e-01,
                            2.32974274e-01,
                            8.07105196e-01,
                            3.87860644e-01,
                            8.63541855e-01,
                            7.47121643e-01,
                        ],
                        [
                            5.56240234e-01,
                            1.36455226e-01,
                            5.99176895e-02,
                            1.21343456e-01,
                            4.45518785e-02,
                            1.07494129e-01,
                            2.25709339e-01,
                            7.12988980e-01,
                            5.59716982e-01,
                            1.25559802e-02,
                        ],
                    ]
                ),
                "W2": np.array(
                    [
                        [
                            7.19742797e-02,
                            9.67276330e-01,
                            5.68100462e-01,
                            2.03293235e-01,
                            2.52325745e-01,
                            7.43825854e-01,
                            1.95429481e-01,
                            5.81358927e-01,
                            9.70019989e-01,
                            8.46828801e-01,
                            2.39847759e-01,
                            4.93769714e-01,
                            6.19955718e-01,
                            8.28980900e-01,
                            1.56791395e-01,
                        ],
                        [
                            1.85762022e-02,
                            7.00221437e-02,
                            4.86345111e-01,
                            6.06329462e-01,
                            5.68851437e-01,
                            3.17362409e-01,
                            9.88616154e-01,
                            5.79745219e-01,
                            3.80141173e-01,
                            5.50948219e-01,
                            7.45334431e-01,
                            6.69232893e-01,
                            2.64919558e-01,
                            6.63348344e-02,
                            3.70084198e-01,
                        ],
                        [
                            6.29717507e-01,
                            2.10174010e-01,
                            7.52755554e-01,
                            6.65364814e-02,
                            2.60315099e-01,
                            8.04754564e-01,
                            1.93434283e-01,
                            6.39460881e-01,
                            5.24670309e-01,
                            9.24807970e-01,
                            2.63296770e-01,
                            6.59610907e-02,
                            7.35065963e-01,
                            7.72178030e-01,
                            9.07815853e-01,
                        ],
                        [
                            9.31972069e-01,
                            1.39515730e-02,
                            2.34362086e-01,
                            6.16778357e-01,
                            9.49016321e-01,
                            9.50176119e-01,
                            5.56653188e-01,
                            9.15606350e-01,
                            6.41566209e-01,
                            3.90007714e-01,
                            4.85990667e-01,
                            6.04310483e-01,
                            5.49547922e-01,
                            9.26181427e-01,
                            9.18733436e-01,
                        ],
                        [
                            3.94875613e-01,
                            9.63262528e-01,
                            1.73955667e-01,
                            1.26329519e-01,
                            1.35079158e-01,
                            5.05662166e-01,
                            2.15248053e-02,
                            9.47970211e-01,
                            8.27115471e-01,
                            1.50189807e-02,
                            1.76196256e-01,
                            3.32063574e-01,
                            1.30996845e-01,
                            8.09490692e-01,
                            3.44736653e-01,
                        ],
                        [
                            9.40107482e-01,
                            5.82014180e-01,
                            8.78831984e-01,
                            8.44734445e-01,
                            9.05392319e-01,
                            4.59880266e-01,
                            5.46346816e-01,
                            7.98603591e-01,
                            2.85718852e-01,
                            4.90253523e-01,
                            5.99110308e-01,
                            1.55332756e-02,
                            5.93481408e-01,
                            4.33676349e-01,
                            8.07360529e-01,
                        ],
                        [
                            3.15244803e-01,
                            8.92888709e-01,
                            5.77857215e-01,
                            1.84010202e-01,
                            7.87929234e-01,
                            6.12031177e-01,
                            5.39092721e-02,
                            4.20193680e-01,
                            6.79068837e-01,
                            9.18601778e-01,
                            4.02024891e-04,
                            9.76759149e-01,
                            3.76580315e-01,
                            9.73783538e-01,
                            6.04716101e-01,
                        ],
                        [
                            8.28845808e-01,
                            5.74711505e-01,
                            6.28076198e-01,
                            2.85576282e-01,
                            5.86833341e-01,
                            7.50021764e-01,
                            8.58313836e-01,
                            7.55082188e-01,
                            6.98057248e-01,
                            8.64479430e-01,
                            3.22680997e-01,
                            6.70788791e-01,
                            4.50873936e-01,
                            3.82102752e-01,
                            4.10811350e-01,
                        ],
                        [
                            4.01479583e-01,
                            3.17383946e-01,
                            6.21919368e-01,
                            4.30247271e-01,
                            9.73802078e-01,
                            6.77800891e-01,
                            1.98569888e-01,
                            4.26701009e-01,
                            3.43346240e-01,
                            7.97638804e-01,
                            8.79998289e-01,
                            9.03841956e-01,
                            6.62719812e-01,
                            2.70208262e-01,
                            2.52366702e-01,
                        ],
                        [
                            8.54897943e-01,
                            5.27714646e-01,
                            8.02161084e-01,
                            5.72488517e-01,
                            7.33142525e-01,
                            5.19011627e-01,
                            7.70883911e-01,
                            5.68857991e-01,
                            4.65709879e-01,
                            3.42688908e-01,
                            6.82093484e-02,
                            3.77924179e-01,
                            7.96260777e-02,
                            9.82817114e-01,
                            1.81612851e-01,
                        ],
                    ]
                ),
                "b1": np.array(
                    [
                        [0.8118587],
                        [0.87496164],
                        [0.68841325],
                        [0.56949441],
                        [0.16097144],
                        [0.46688002],
                        [0.34517205],
                        [0.22503996],
                        [0.59251187],
                        [0.31226984],
                        [0.91630555],
                        [0.90963552],
                        [0.25711829],
                        [0.1108913],
                        [0.19296273],
                    ]
                ),
                "b2": np.array(
                    [
                        [0.49958417],
                        [0.72858567],
                        [0.20819444],
                        [0.24803356],
                        [0.85167187],
                        [0.41584872],
                        [0.61668507],
                        [0.23366614],
                        [0.10196726],
                        [0.51585702],
                    ]
                ),
                "batch_size": 4,
            },
            "expected": {
                "grad_W1": np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                "grad_W2": np.array(
                    [
                        [
                            0.07703226,
                            0.0830197,
                            0.06531928,
                            0.05403581,
                            0.01527358,
                            0.04429936,
                            0.03275124,
                            0.02135265,
                            0.05621979,
                            0.02962936,
                            0.08694257,
                            0.08630969,
                            0.02439637,
                            0.01052179,
                            0.01830904,
                        ],
                        [
                            -0.75214825,
                            -0.81061012,
                            -0.63778195,
                            -0.52760933,
                            -0.14913234,
                            -0.43254201,
                            -0.3197854,
                            -0.20848876,
                            -0.54893391,
                            -0.28930307,
                            -0.84891327,
                            -0.84273381,
                            -0.2382078,
                            -0.10273549,
                            -0.17877074,
                        ],
                        [
                            0.03296985,
                            0.03553248,
                            0.02795669,
                            0.02312735,
                            0.0065371,
                            0.01896015,
                            0.01401755,
                            0.00913895,
                            0.0240621,
                            0.01268138,
                            0.03721147,
                            0.03694059,
                            0.01044166,
                            0.00450333,
                            0.00783628,
                        ],
                        [
                            0.08996299,
                            0.0969555,
                            0.07628386,
                            0.06310633,
                            0.01783743,
                            0.05173551,
                            0.03824891,
                            0.02493694,
                            0.06565692,
                            0.03460298,
                            0.10153687,
                            0.10079776,
                            0.02849157,
                            0.01228799,
                            0.02138242,
                        ],
                        [
                            0.04894702,
                            0.0527515,
                            0.04150449,
                            0.03433486,
                            0.00970498,
                            0.02814823,
                            0.02081045,
                            0.01356768,
                            0.03572259,
                            0.01882677,
                            0.05524413,
                            0.054842,
                            0.01550168,
                            0.00668564,
                            0.01163374,
                        ],
                        [
                            0.12178334,
                            0.13124914,
                            0.10326583,
                            0.08542734,
                            0.02414662,
                            0.07003461,
                            0.05177774,
                            0.03375725,
                            0.08888009,
                            0.04684222,
                            0.13745095,
                            0.13645041,
                            0.03856918,
                            0.01663431,
                            0.02894549,
                        ],
                        [
                            0.10343212,
                            0.11147154,
                            0.08770497,
                            0.07255452,
                            0.02050802,
                            0.05948128,
                            0.04397548,
                            0.02867046,
                            0.07548698,
                            0.03978369,
                            0.11673882,
                            0.11588905,
                            0.03275729,
                            0.01412773,
                            0.02458377,
                        ],
                        [
                            0.12143411,
                            0.13087276,
                            0.1029697,
                            0.08518237,
                            0.02407737,
                            0.06983378,
                            0.05162926,
                            0.03366045,
                            0.08862521,
                            0.04670789,
                            0.13705679,
                            0.13605912,
                            0.03845858,
                            0.01658661,
                            0.02886248,
                        ],
                        [
                            0.08175855,
                            0.08811336,
                            0.06932693,
                            0.05735116,
                            0.01621069,
                            0.04701734,
                            0.03476069,
                            0.02266274,
                            0.05966914,
                            0.03144726,
                            0.09227692,
                            0.09160521,
                            0.0258932,
                            0.01116735,
                            0.01943239,
                        ],
                        [
                            0.07482801,
                            0.08064413,
                            0.06345019,
                            0.05248959,
                            0.01483654,
                            0.04303175,
                            0.03181408,
                            0.02074165,
                            0.05461108,
                            0.02878152,
                            0.08445474,
                            0.08383998,
                            0.02369827,
                            0.01022071,
                            0.01778514,
                        ],
                    ]
                ),
                "grad_b1": np.array(
                    [
                        [0.56665732],
                        [0.46268776],
                        [0.10631469],
                        [0.0],
                        [0.11041817],
                        [0.32025188],
                        [0.0],
                        [0.08430877],
                        [0.19341],
                        [0.08339139],
                        [0.0],
                        [0.0],
                        [0.19055422],
                        [0.56405984],
                        [0.13321987],
                    ]
                ),
                "grad_b2": np.array(
                    [
                        [0.09488382],
                        [-0.92645217],
                        [0.04061033],
                        [0.11081115],
                        [0.06029008],
                        [0.15000559],
                        [0.12740163],
                        [0.14957542],
                        [0.1007054],
                        [0.09216876],
                    ]
                ),
            },
        },
        # {
        #     "name": "default_check",
        #     "input": {
        #         "x": np.array(
        #             [
        #                 [0.0, 0.0, 0.0, 0.0],
        #                 [0.0, 0.0, 0.0, 0.0],
        #                 [0.0, 0.0, 0.0, 0.0],
        #                 [0.0, 0.0, 0.0, 0.0],
        #                 [0.0, 0.0, 0.0, 0.0],
        #             ]
        #         ),
        #         "yhat": np.array(
        #             [
        #                 [0.22636192, 0.22636192, 0.22636192, 0.22636192],
        #                 [0.40947153, 0.40947153, 0.40947153, 0.40947153],
        #                 [0.12268763, 0.12268763, 0.12268763, 0.12268763],
        #                 [0.06530444, 0.06530444, 0.06530444, 0.06530444],
        #                 [0.17617448, 0.17617448, 0.17617448, 0.17617448],
        #             ]
        #         ),
        #         "y": np.array(
        #             [
        #                 [0.0, 0.0, 0.0, 0.0],
        #                 [1.0, 1.0, 1.0, 1.0],
        #                 [0.0, 0.0, 0.0, 0.0],
        #                 [0.0, 0.0, 0.0, 0.0],
        #                 [0.0, 0.0, 0.0, 0.0],
        #             ]
        #         ),
        #         "h": np.array(
        #             [
        #                 [0.9851368, 0.9851368, 0.9851368, 0.9851368],
        #                 [0.80140153, 0.80140153, 0.80140153, 0.80140153],
        #                 [0.0539621, 0.0539621, 0.0539621, 0.0539621],
        #                 [0.19047777, 0.19047777, 0.19047777, 0.19047777],
        #                 [0.45241885, 0.45241885, 0.45241885, 0.45241885],
        #                 [0.70294208, 0.70294208, 0.70294208, 0.70294208],
        #                 [0.33204815, 0.33204815, 0.33204815, 0.33204815],
        #                 [0.3599832, 0.3599832, 0.3599832, 0.3599832],
        #                 [0.92147057, 0.92147057, 0.92147057, 0.92147057],
        #                 [0.95363051, 0.95363051, 0.95363051, 0.95363051],
        #             ]
        #         ),
        #         "W1": np.array(
        #             [
        #                 [0.22199317, 0.87073231, 0.20671916, 0.91861091, 0.48841119],
        #                 [0.61174386, 0.76590786, 0.51841799, 0.2968005, 0.18772123],
        #                 [0.08074127, 0.7384403, 0.44130922, 0.15830987, 0.87993703],
        #                 [0.27408646, 0.41423502, 0.29607993, 0.62878791, 0.57983781],
        #                 [0.5999292, 0.26581912, 0.28468588, 0.25358821, 0.32756395],
        #                 [0.1441643, 0.16561286, 0.96393053, 0.96022672, 0.18841466],
        #                 [0.02430656, 0.20455555, 0.69984361, 0.77951459, 0.02293309],
        #                 [0.57766286, 0.00164217, 0.51547261, 0.63979518, 0.9856244],
        #                 [0.2590976, 0.80249689, 0.87048309, 0.92274961, 0.00221421],
        #                 [0.46948837, 0.98146874, 0.3989448, 0.81373248, 0.5464565],
        #             ]
        #         ),
        #         "W2": np.array(
        #             [
        #                 [
        #                     0.77085409,
        #                     0.48493107,
        #                     0.02911156,
        #                     0.08652569,
        #                     0.11145381,
        #                     0.25124511,
        #                     0.96491529,
        #                     0.63176605,
        #                     0.8166602,
        #                     0.566082,
        #                 ],
        #                 [
        #                     0.63535621,
        #                     0.81190239,
        #                     0.92668262,
        #                     0.91262676,
        #                     0.82481072,
        #                     0.09420273,
        #                     0.36104842,
        #                     0.03550903,
        #                     0.54635835,
        #                     0.79614272,
        #                 ],
        #                 [
        #                     0.0511428,
        #                     0.18866774,
        #                     0.36547777,
        #                     0.24429087,
        #                     0.79508747,
        #                     0.35209494,
        #                     0.63887768,
        #                     0.49341505,
        #                     0.58349974,
        #                     0.93929935,
        #                 ],
        #                 [
        #                     0.94354008,
        #                     0.11169243,
        #                     0.84355497,
        #                     0.34602815,
        #                     0.10082727,
        #                     0.38340907,
        #                     0.5103548,
        #                     0.96110308,
        #                     0.37151262,
        #                     0.01236941,
        #                 ],
        #                 [
        #                     0.85970689,
        #                     0.11111075,
        #                     0.47833904,
        #                     0.84998003,
        #                     0.51473797,
        #                     0.44660783,
        #                     0.80047642,
        #                     0.02039138,
        #                     0.57261865,
        #                     0.41138362,
        #                 ],
        #             ]
        #         ),
        #         "b1": np.array(
        #             [
        #                 [0.9851368],
        #                 [0.80140153],
        #                 [0.0539621],
        #                 [0.19047777],
        #                 [0.45241885],
        #                 [0.70294208],
        #                 [0.33204815],
        #                 [0.3599832],
        #                 [0.92147057],
        #                 [0.95363051],
        #             ]
        #         ),
        #         "b2": np.array(
        #             [
        #                 [0.40768573],
        #                 [0.89857115],
        #                 [0.33025325],
        #                 [0.08273857],
        #                 [0.52671757],
        #             ]
        #         ),
        #         "batch_size": 2,
        #     },
        #     "expected": {
        #         "grad_W1": np.array(
        #             [
        #                 [0.0, 0.0, 0.0, 0.0, 0.0],
        #                 [0.0, 0.0, 0.0, 0.0, 0.0],
        #                 [0.0, 0.0, 0.0, 0.0, 0.0],
        #                 [0.0, 0.0, 0.0, 0.0, 0.0],
        #                 [0.0, 0.0, 0.0, 0.0, 0.0],
        #                 [0.0, 0.0, 0.0, 0.0, 0.0],
        #                 [0.0, 0.0, 0.0, 0.0, 0.0],
        #                 [0.0, 0.0, 0.0, 0.0, 0.0],
        #                 [0.0, 0.0, 0.0, 0.0, 0.0],
        #                 [0.0, 0.0, 0.0, 0.0, 0.0],
        #             ]
        #         ),
        #         "grad_W2": np.array(
        #             [
        #                 [
        #                     0.44599491,
        #                     0.36281358,
        #                     0.02442993,
        #                     0.08623383,
        #                     0.2048208,
        #                     0.31823863,
        #                     0.15032611,
        #                     0.16297297,
        #                     0.41717169,
        #                     0.43173126,
        #                 ],
        #                 [
        #                     -1.16350265,
        #                     -0.94650084,
        #                     -0.06373232,
        #                     -0.2249651,
        #                     -0.53433242,
        #                     -0.83021462,
        #                     -0.39216777,
        #                     -0.42516065,
        #                     -1.0883092,
        #                     -1.12629192,
        #                 ],
        #                 [
        #                     0.2417282,
        #                     0.19664411,
        #                     0.01324097,
        #                     0.04673853,
        #                     0.11101239,
        #                     0.1724846,
        #                     0.0814764,
        #                     0.08833097,
        #                     0.22610608,
        #                     0.23399734,
        #                 ],
        #                 [
        #                     0.12866761,
        #                     0.10467015,
        #                     0.00704793,
        #                     0.02487809,
        #                     0.05908992,
        #                     0.09181047,
        #                     0.04336843,
        #                     0.047017,
        #                     0.12035223,
        #                     0.12455261,
        #                 ],
        #                 [
        #                     0.34711193,
        #                     0.282373,
        #                     0.01901349,
        #                     0.06711465,
        #                     0.15940931,
        #                     0.24768091,
        #                     0.11699682,
        #                     0.12683971,
        #                     0.3246792,
        #                     0.33601072,
        #                 ],
        #             ]
        #         ),
        #         "grad_b1": np.array(
        #             [
        #                 [0.03729288],
        #                 [0.0],
        #                 [0.0],
        #                 [0.0],
        #                 [0.0],
        #                 [0.29631968],
        #                 [0.5158901],
        #                 [0.49786268],
        #                 [0.11790206],
        #                 [0.0],
        #             ]
        #         ),
        #         "grad_b2": np.array(
        #             [
        #                 [0.45272384],
        #                 [-1.18105694],
        #                 [0.24537526],
        #                 [0.13060887],
        #                 [0.35234896],
        #             ]
        #         ),
        #     },
        # },
        #################
        {
            "name": "default_check",
            "input": {
                "x": np.array(
                    [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],]
                ),
                "yhat": np.array(
                    [
                        [0.22636192, 0.22636192],
                        [0.40947153, 0.40947153],
                        [0.12268763, 0.12268763],
                        [0.06530444, 0.06530444],
                        [0.17617448, 0.17617448],
                    ]
                ),
                "y": np.array(
                    [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0],]
                ),
                "h": np.array(
                    [
                        [0.9851368, 0.9851368],
                        [0.80140153, 0.80140153],
                        [0.0539621, 0.0539621],
                        [0.19047777, 0.19047777],
                        [0.45241885, 0.45241885],
                        [0.70294208, 0.70294208],
                        [0.33204815, 0.33204815],
                        [0.3599832, 0.3599832],
                        [0.92147057, 0.92147057],
                        [0.95363051, 0.95363051],
                    ]
                ),
                "W1": np.array(
                    [
                        [0.22199317, 0.87073231, 0.20671916, 0.91861091, 0.48841119],
                        [0.61174386, 0.76590786, 0.51841799, 0.2968005, 0.18772123],
                        [0.08074127, 0.7384403, 0.44130922, 0.15830987, 0.87993703],
                        [0.27408646, 0.41423502, 0.29607993, 0.62878791, 0.57983781],
                        [0.5999292, 0.26581912, 0.28468588, 0.25358821, 0.32756395],
                        [0.1441643, 0.16561286, 0.96393053, 0.96022672, 0.18841466],
                        [0.02430656, 0.20455555, 0.69984361, 0.77951459, 0.02293309],
                        [0.57766286, 0.00164217, 0.51547261, 0.63979518, 0.9856244],
                        [0.2590976, 0.80249689, 0.87048309, 0.92274961, 0.00221421],
                        [0.46948837, 0.98146874, 0.3989448, 0.81373248, 0.5464565],
                    ]
                ),
                "W2": np.array(
                    [
                        [
                            0.77085409,
                            0.48493107,
                            0.02911156,
                            0.08652569,
                            0.11145381,
                            0.25124511,
                            0.96491529,
                            0.63176605,
                            0.8166602,
                            0.566082,
                        ],
                        [
                            0.63535621,
                            0.81190239,
                            0.92668262,
                            0.91262676,
                            0.82481072,
                            0.09420273,
                            0.36104842,
                            0.03550903,
                            0.54635835,
                            0.79614272,
                        ],
                        [
                            0.0511428,
                            0.18866774,
                            0.36547777,
                            0.24429087,
                            0.79508747,
                            0.35209494,
                            0.63887768,
                            0.49341505,
                            0.58349974,
                            0.93929935,
                        ],
                        [
                            0.94354008,
                            0.11169243,
                            0.84355497,
                            0.34602815,
                            0.10082727,
                            0.38340907,
                            0.5103548,
                            0.96110308,
                            0.37151262,
                            0.01236941,
                        ],
                        [
                            0.85970689,
                            0.11111075,
                            0.47833904,
                            0.84998003,
                            0.51473797,
                            0.44660783,
                            0.80047642,
                            0.02039138,
                            0.57261865,
                            0.41138362,
                        ],
                    ]
                ),
                "b1": np.array(
                    [
                        [0.9851368],
                        [0.80140153],
                        [0.0539621],
                        [0.19047777],
                        [0.45241885],
                        [0.70294208],
                        [0.33204815],
                        [0.3599832],
                        [0.92147057],
                        [0.95363051],
                    ]
                ),
                "b2": np.array(
                    [
                        [0.40768573],
                        [0.89857115],
                        [0.33025325],
                        [0.08273857],
                        [0.52671757],
                    ]
                ),
                "batch_size": 2,
            },
            "expected": {
                "grad_W1": np.array(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ),
                "grad_W2": np.array(
                    [
                        [
                            0.22299746,
                            0.18140679,
                            0.01221496,
                            0.04311691,
                            0.1024104,
                            0.15911932,
                            0.07516306,
                            0.08148649,
                            0.20858585,
                            0.21586563,
                        ],
                        [
                            -0.58175133,
                            -0.47325042,
                            -0.03186616,
                            -0.11248255,
                            -0.26716621,
                            -0.41510731,
                            -0.19608389,
                            -0.21258033,
                            -0.54415461,
                            -0.56314597,
                        ],
                        [
                            0.1208641,
                            0.09832205,
                            0.00662048,
                            0.02336927,
                            0.0555062,
                            0.0862423,
                            0.0407382,
                            0.04416549,
                            0.11305304,
                            0.11699867,
                        ],
                        [
                            0.06433381,
                            0.05233508,
                            0.00352396,
                            0.01243904,
                            0.02954496,
                            0.04590524,
                            0.02168422,
                            0.0235085,
                            0.06017612,
                            0.06227631,
                        ],
                        [
                            0.17355596,
                            0.1411865,
                            0.00950674,
                            0.03355732,
                            0.07970466,
                            0.12384046,
                            0.05849841,
                            0.06341985,
                            0.1623396,
                            0.16800536,
                        ],
                    ]
                ),
                "grad_b1": np.array(
                    [
                        [0.01864644],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.0],
                        [0.14815984],
                        [0.25794505],
                        [0.24893135],
                        [0.05895103],
                        [0.0],
                    ]
                ),
                "grad_b2": np.array(
                    [
                        [0.22636192],
                        [-0.59052847],
                        [0.12268763],
                        [0.06530444],
                        [0.17617448],
                    ]
                ),
            },
        },
        #################
    ]

    for test_case in test_cases:
        tmp_grad_W1, tmp_grad_W2, tmp_grad_b1, tmp_grad_b2 = target(
            **test_case["input"]
        )

        try:
            assert tmp_grad_W1.shape == test_case["expected"]["grad_W1"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["grad_W1"].shape,
                    "got": tmp_grad_W1.shape,
                }
            )
            print(
                f"Wrong output shape for gradient of W1 matrix.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert tmp_grad_W2.shape == test_case["expected"]["grad_W2"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["grad_W2"].shape,
                    "got": tmp_grad_W2.shape,
                }
            )
            print(
                f"Wrong output shape for gradient of W2 matrix.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert tmp_grad_b1.shape == test_case["expected"]["grad_b1"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["grad_b1"].shape,
                    "got": tmp_grad_b1.shape,
                }
            )
            print(
                f"Wrong output shape for gradient of b1 vector.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert tmp_grad_b2.shape == test_case["expected"]["grad_b2"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["grad_b2"].shape,
                    "got": tmp_grad_b2.shape,
                }
            )
            print(
                f"Wrong output shape for gradient of b2 vector.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        # Testing values
        try:
            assert np.allclose(tmp_grad_W1, test_case["expected"]["grad_W1"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["grad_W1"],
                    "got": tmp_grad_W1,
                }
            )
            print(
                f"Wrong output values for gradient of W1 matrix.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(tmp_grad_W2, test_case["expected"]["grad_W2"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["grad_W2"],
                    "got": tmp_grad_W2,
                }
            )
            print(
                f"Wrong output values for gradient of W2 matrix.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(tmp_grad_b1, test_case["expected"]["grad_b1"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["grad_b1"],
                    "got": tmp_grad_b1,
                }
            )
            print(
                f"Wrong output values for gradient of b1 vector.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(tmp_grad_b2, test_case["expected"]["grad_b2"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["grad_b2"],
                    "got": tmp_grad_b2,
                }
            )
            print(
                f"Wrong output values for gradient of b2 vector.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_gradient_descent(target, data, word2Ind, N, V, num_iters):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "data": data,
                "word2Ind": word2Ind,
                "N": N,
                "V": V,
                "num_iters": num_iters,
                "alpha": 0.03,
                "random_seed": 282,
            },
            "expected": {
                "W1": pickle.load(
                    open("./support_files/gradient_descent/w1.pkl", "rb")
                ),
                "W2": pickle.load(
                    open("./support_files/gradient_descent/w2.pkl", "rb")
                ),
                "b1": pickle.load(
                    open("./support_files/gradient_descent/b1.pkl", "rb")
                ),
                "b2": pickle.load(
                    open("./support_files/gradient_descent/b2.pkl", "rb")
                ),
            },
        },
        {
            "name": "small_check",
            "input": {
                "data": data,
                "word2Ind": word2Ind,
                "N": 5,
                "V": V,
                "num_iters": num_iters,
                "alpha": 0.01,
                "random_seed": 5,
            },
            "expected": {
                "W1": pickle.load(
                    open("./support_files/gradient_descent/w1_exm2.pkl", "rb")
                ),
                "W2": pickle.load(
                    open("./support_files/gradient_descent/w2_exm2.pkl", "rb")
                ),
                "b1": pickle.load(
                    open("./support_files/gradient_descent/b1_exm2.pkl", "rb")
                ),
                "b2": pickle.load(
                    open("./support_files/gradient_descent/b2_exm2.pkl", "rb")
                ),
            },
        },
    ]

    for test_case in test_cases:
        print("name", test_case["name"])
        W1, W2, b1, b2 = target(**test_case["input"])

        try:
            assert W1.shape == test_case["expected"]["W1"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["W1"].shape,
                    "got": W1.shape,
                }
            )
            print(
                f"Wrong output shape for W1 matrix.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert W2.shape == test_case["expected"]["W2"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["W2"].shape,
                    "got": W2.shape,
                }
            )
            print(
                f"Wrong output shape for W2 matrix.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert b1.shape == test_case["expected"]["b1"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["b1"].shape,
                    "got": b1.shape,
                }
            )
            print(
                f"Wrong output shape for b1 vector.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert b2.shape == test_case["expected"]["b2"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["b2"].shape,
                    "got": b2.shape,
                }
            )
            print(
                f"Wrong output shape for b2 vector.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        # Testing values
        try:
            assert np.allclose(W1, test_case["expected"]["W1"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["W1"],
                    "got": W1,
                }
            )
            print(
                f"Wrong output values for W1 matrix.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(W2, test_case["expected"]["W2"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["W2"],
                    "got": W2,
                }
            )
            print(
                f"Wrong output values for W2 matrix.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(b1, test_case["expected"]["b1"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["b1"],
                    "got": b1,
                }
            )
            print(
                f"Wrong output values for b1 vector.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(b2, test_case["expected"]["b2"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["b2"],
                    "got": b2,
                }
            )
            print(
                f"Wrong output values for gradient of b2 vector.\n\t Expected: {failed_cases[-1].get('expected')} \n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases

