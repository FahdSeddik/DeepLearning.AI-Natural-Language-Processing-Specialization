import pickle
import numpy as np
import pandas as pd


def test_cosine_similarity(target):
    successful_cases = 0
    failed_cases = []

    word_embeddings = pickle.load(open("./data/word_embeddings_subset.p", "rb"))

    test_cases = [
        {
            "name": "cosine_score_1",
            "input": {"A": word_embeddings["king"], "B": word_embeddings["queen"]},
            "expected": [0.650, 0.6512, 0.6510957],
        },
        {
            "name": "cosine_score_2",
            "input": {"A": word_embeddings["Japan"], "B": word_embeddings["Tokyo"]},
            "expected": [0.699, 0.701, 0.70022535],
        },
        {
            "name": "cosine_score_3",
            "input": {"A": word_embeddings["Germany"], "B": word_embeddings["Beirut"]},
            "expected": [0.172, 0.174, 0.17339969],
        },
        {
            "name": "cosine_score_4_to_catch_alternate_solution",
            "input": {"A": word_embeddings["China"], "B": word_embeddings["Chile"]},
            "expected": [0.32, 0.381, 0.3801232],
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])
        try:
            # print(cosine_similarity(test_case['input'][0],test_case['input'][1]))
            assert np.isclose(result, test_case["expected"][2]) or (
                test_case["expected"][0] <= result <= test_case["expected"][1]
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"][2],
                    "got": result,
                }
            )
            print(
                f"Wrong output in cosine similarity function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_euclidean(target):
    successful_cases = 0
    failed_cases = []

    word_embeddings = pickle.load(open("./data/word_embeddings_subset.p", "rb"))

    test_cases = [
        {
            "name": "euclidean_score_1",
            "input": {"A": word_embeddings["king"], "B": word_embeddings["queen"]},
            "expected": [2.47, 2.48, 2.4796925],
        },
        {
            "name": "euclidean_score_2",
            "input": {"A": word_embeddings["Japan"], "B": word_embeddings["Tokyo"]},
            "expected": [2.43, 2.44, 2.4345345],
        },
        {
            "name": "euclidean_score_3",
            "input": {"A": word_embeddings["Germany"], "B": word_embeddings["Beirut"]},
            "expected": [4.0, 4.1, 4.0416517],
        },
        {
            "name": "euclidean_score_4",
            "input": {"A": word_embeddings["China"], "B": word_embeddings["Chile"]},
            "expected": [3.2, 3.3, 3.2326782],
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert np.isclose(
                result, test_case["expected"][2], rtol=1e-3, atol=1e-05
            ) or (test_case["expected"][0] <= result <= test_case["expected"][1])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"][2],
                    "got": result,
                }
            )
            print(
                f"Wrong output in the euclidean distance function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_get_country(target):
    successful_cases = 0
    failed_cases = []

    word_embeddings = pickle.load(open("./data/word_embeddings_subset.p", "rb"))

    test_cases = [
        {
            "name": "get_country_score_1",
            "input": {
                "city1": "Athens",
                "country1": "Greece",
                "city2": "Cairo",
                "embeddings": word_embeddings,
            },
            "expected": ("Egypt", 0.7626821),
        },
        {
            "name": "get_country_score_2_for_wrong_cosine_similarity",
            "input": {
                "city1": "oil",
                "country1": "gas",
                "city2": "town",
                "embeddings": word_embeddings,
            },
            "expected": ("village", 0.5611889),
        },
        {
            "name": "get_country_score_3",
            "input": {
                "city1": "Doha",
                "country1": "Qatar",
                "city2": "Jakarta",
                "embeddings": word_embeddings,
            },
            "expected": ("Indonesia", 0.6782036),
        },
        {
            "name": "get_country_score_4",
            "input": {
                "city1": "Tokyo",
                "country1": "Japan",
                "city2": "Canberra",
                "embeddings": word_embeddings,
            },
            "expected": ("Australia", 0.7139509),
        },
        {
            "name": "get_country_score_5_for_wrong_cosine_similarity",
            "input": {
                "city1": "joyful",
                "country1": "happy",
                "city2": "sad",
                "embeddings": word_embeddings,
            },
            "expected": ("king", 0.09570546),
        },
        {
            "name": "get_country_score_6_for_wrong_cosine_similarity",
            "input": {
                "city1": "happy",
                "country1": "joyful",
                "city2": "sad",
                "embeddings": word_embeddings,
            },
            "expected": ("Lebanon", 0.14527377),
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert isinstance(result, tuple)
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
                f"Wrong output type. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert result[0] == test_case["expected"][0]
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
                f"Wrong output word. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.isclose(result[1], test_case["expected"][1])
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
                f"Wrong output similarity. Maybe you should check your cosine_similarity implementation. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_get_accuracy(target, data):
    successful_cases = 0
    failed_cases = []

    word_embeddings = pickle.load(open("./data/word_embeddings_subset.p", "rb"))

    test_cases = [
        {
            "name": "default_check",
            "input": {"word_embeddings": word_embeddings, "data": data},
            "expected": 0.9192082407594425,
        },
        {
            "name": "smaller_check",
            "input": {
                "word_embeddings": word_embeddings,
                "data": data.sample(frac=0.15, random_state=3),
            },
            "expected": 0.9125168236877523,
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
                f"Wrong accuracy output. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_compute_pca(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "X": np.array(  # np.random.seed(1)
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
                    ]
                ),
                "n_components": 2,
            },
            "expected": np.array(
                [
                    [0.43437323, 0.49820384],
                    [0.42077249, -0.50351448],
                    [-0.85514571, 0.00531064],
                ]
            ),
        },
        {
            "name": "larger_check",
            "input": {
                "X": np.array(  # np.random.seed(2)
                    [
                        [
                            0.4359949,
                            0.02592623,
                            0.54966248,
                            0.43532239,
                            0.4203678,
                            0.33033482,
                            0.20464863,
                            0.61927097,
                            0.29965467,
                            0.26682728,
                            0.62113383,
                            0.52914209,
                            0.13457995,
                            0.51357812,
                            0.18443987,
                        ],
                        [
                            0.78533515,
                            0.85397529,
                            0.49423684,
                            0.84656149,
                            0.07964548,
                            0.50524609,
                            0.0652865,
                            0.42812233,
                            0.09653092,
                            0.12715997,
                            0.59674531,
                            0.226012,
                            0.10694568,
                            0.22030621,
                            0.34982629,
                        ],
                        [
                            0.46778748,
                            0.20174323,
                            0.64040673,
                            0.48306984,
                            0.50523672,
                            0.38689265,
                            0.79363745,
                            0.58000418,
                            0.1622986,
                            0.70075235,
                            0.96455108,
                            0.50000836,
                            0.88952006,
                            0.34161365,
                            0.56714413,
                        ],
                        [
                            0.42754596,
                            0.43674726,
                            0.77655918,
                            0.53560417,
                            0.95374223,
                            0.54420816,
                            0.08209492,
                            0.3663424,
                            0.8508505,
                            0.40627504,
                            0.02720237,
                            0.24717724,
                            0.06714437,
                            0.99385201,
                            0.97058031,
                        ],
                        [
                            0.80025835,
                            0.60181712,
                            0.76495986,
                            0.16922545,
                            0.29302323,
                            0.52406688,
                            0.35662428,
                            0.04567897,
                            0.98315345,
                            0.44135492,
                            0.50400044,
                            0.32354132,
                            0.25974475,
                            0.38688989,
                            0.8320169,
                        ],
                    ]
                ),
                "n_components": 3,
            },
            "expected": np.array(
                [
                    [-0.32462796, 0.01881248, -0.51389463],
                    [-0.36781354, 0.88364184, 0.05985815],
                    [-0.75767901, -0.69452194, 0.12223214],
                    [1.01698298, -0.17990871, -0.33555475],
                    [0.43313753, -0.02802368, 0.66735909],
                ]
            ),
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
                f"Wrong output type. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert result.shape == test_case["expected"].shape
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
                f"Wrong output shape. Check if you are taking the proper number of dimensions.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
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
                f"Wrong output shape. Check if you are taking the proper number of dimensions.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases

