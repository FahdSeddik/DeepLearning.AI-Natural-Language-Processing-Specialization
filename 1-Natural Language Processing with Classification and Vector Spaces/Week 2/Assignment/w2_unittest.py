# -*- coding: utf-8 -*-
from utils import process_tweet, lookup
import numpy as np
import pandas as pd
import pickle


def test_count_tweets(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "result": {},
                "tweets": [
                    "i am happy",
                    "i am tricked",
                    "i am sad",
                    "i am tired",
                    "i am tired",
                ],
                "ys": [1, 0, 0, 0, 0],
            },
            "expected": {
                ("happi", 1): 1,
                ("trick", 0): 1,
                ("sad", 0): 1,
                ("tire", 0): 2,
            },
        },
        {
            "name": "larger_check",
            "input": {
                "result": {},
                "tweets": [
                    "i am happy",
                    "i am tricked",
                    "i am sad",
                    "i am tired",
                    "i am tired but proud today",
                    "i am you are",
                    "you are happy",
                    "he was sad",
                ],
                "ys": [1, 0, 0, 0, 1, 0, 1, 0],
            },
            "expected": {
                ("happi", 1): 2,
                ("trick", 0): 1,
                ("sad", 0): 2,
                ("tire", 0): 1,
                ("tire", 1): 1,
                ("proud", 1): 1,
                ("today", 1): 1,
            },
        },
        {
            "name": "nonempyt_dict_check",
            "input": {
                "result": {
                    ("happi", 1): 3,
                    ("sad", 0): 1,
                    ("tire", 0): 2,
                    ("tire", 1): 1,
                },
                "tweets": [
                    "i am happy",
                    "i am tricked",
                    "i am sad",
                    "i am tired",
                    "i am tired but proud today",
                    "i am you are",
                    "you are happy",
                    "he was sad",
                ],
                "ys": [1, 0, 0, 0, 1, 0, 1, 0],
            },
            "expected": {
                ("happi", 1): 5,
                ("sad", 0): 3,
                ("tire", 0): 3,
                ("tire", 1): 2,
                ("trick", 0): 1,
                ("proud", 1): 1,
                ("today", 1): 1,
            },
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert isinstance(result, dict)
            successful_cases += 1
        except:
            failed_cases.append(
                {"name": test_case["name"], "expected": dict, "got": type(result),}
            )
            print(
                f"Wrong output type for count_tweets function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert result == test_case["expected"]
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
                f"Wrong output values in count_tweets function. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )
            if test_case["name"] == "nonempyt_dict_check":
                print(
                    f"Check the use of the result dictionary.\nRemember that it is passed as parameter and should not be initialized inside the function."
                )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_train_naive_bayes_bk(target, freqs, train_x, train_y):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {"freqs": freqs, "train_x": train_x, "train_y": train_y},
            "expected": {
                "logprior": 0.0,
                "loglikelihood": pickle.load(
                    open("./support_files/loglikelihood_test0.pkl", "rb")
                ),
            },
        },
        {
            "name": "smaller_check",
            "input": {
                "freqs": freqs,
                "train_x": train_x[:10] + train_x[-10:],
                "train_y": np.concatenate((train_y[:10], train_y[-10:]), axis=0),
            },
            "expected": {
                "logprior": 0.0,
                "loglikelihood": pickle.load(
                    open("./support_files/loglikelihood_test1.pkl", "rb")
                ),
            },
        },
        {
            "name": "smaller_unbalanced_check",
            "input": {
                "freqs": freqs,
                "train_x": train_x[:10] + train_x[-5:],
                "train_y": np.concatenate((train_y[:10], train_y[-5:]), axis=0),
            },
            "expected": {
                "logprior": 0.6931471805599456,
                "loglikelihood": pickle.load(
                    open("./support_files/loglikelihood_test2.pkl", "rb")
                ),
            },
        },
    ]

    for test_case in test_cases:
        result1, result2 = target(**test_case["input"])

        print("result2", result2["encor"])
        print("expected", test_case["expected"]["loglikelihood"]["encor"])

        try:
            assert isinstance(result1, np.float64) or isinstance(result1, float)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": np.float64,
                    "got": type(result1),
                }
            )
            print(
                f"Wrong output type for logprior. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.isclose(result1, test_case["expected"]["logprior"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["logprior"],
                    "got": result1,
                }
            )
            print(
                f"Wrong value for logprior. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert isinstance(result2, dict)
            successful_cases += 1
        except:
            failed_cases.append(
                {"name": test_case["name"], "expected": dict, "got": type(result2),}
            )
            print(
                f"Wrong output type for loglikelihood. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert len(result2) == len(test_case["expected"]["loglikelihood"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": len(test_case["expected"]["loglikelihood"]),
                    "got": len(result2),
                }
            )
            print(
                f"Wrong number of keys in loglikelihood dictionary. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.isclose(
                result2, test_case["expected"]["loglikelihood"], atol=1e-3
            )

            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": len(test_case["expected"]["loglikelihood"]),
                    "got": len(result2),
                }
            )
            print(
                f"Wrong values for loglikelihood dictionary. Please check your implementation for the loglikelihood dictionary."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_train_naive_bayes(target, freqs, train_x, train_y):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {"freqs": freqs, "train_x": train_x, "train_y": train_y},
            "expected": {
                "logprior": 0.0,
                "loglikelihood": pickle.load(
                    open("./support_files/loglikelihood_test.pkl", "rb")
                ),
            },
        },
        {
            "name": "smaller_check",
            "input": {
                "freqs": freqs,
                "train_x": train_x[:10] + train_x[-10:],
                "train_y": np.concatenate((train_y[:10], train_y[-10:]), axis=0),
            },
            "expected": {
                "logprior": 0.0,
                "loglikelihood": pickle.load(
                    open("./support_files/loglikelihood_test.pkl", "rb")
                ),
            },
        },
        {
            "name": "smaller_unbalanced_check",
            "input": {
                "freqs": freqs,
                "train_x": train_x[:10] + train_x[-5:],
                "train_y": np.concatenate((train_y[:10], train_y[-5:]), axis=0),
            },
            "expected": {
                "logprior": 0.6931471805599456,
                "loglikelihood": pickle.load(
                    open("./support_files/loglikelihood_test.pkl", "rb")
                ),
            },
        },
    ]

    for test_case in test_cases:
        result1, result2 = target(**test_case["input"])

        try:
            assert isinstance(result1, np.float64) or isinstance(result1, float)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": np.float64,
                    "got": type(result1),
                }
            )
            print(
                f"Wrong output type for logprior. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.isclose(result1, test_case["expected"]["logprior"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["logprior"],
                    "got": result1,
                }
            )
            print(
                f"Wrong value for logprior. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert isinstance(result2, dict)
            successful_cases += 1
        except:
            failed_cases.append(
                {"name": test_case["name"], "expected": dict, "got": type(result2),}
            )
            print(
                f"Wrong output type for loglikelihood. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert len(result2) == len(test_case["expected"]["loglikelihood"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": len(test_case["expected"]["loglikelihood"]),
                    "got": len(result2),
                }
            )
            print(
                f"Wrong number of keys in loglikelihood dictionary. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        count_good = 0
        for key, value in test_case["expected"]["loglikelihood"].items():

            if np.isclose(result2[key], value):
                count_good += 1

        try:
            assert count_good == len(test_case["expected"]["loglikelihood"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": len(test_case["expected"]["loglikelihood"]),
                    "got": len(result2),
                }
            )

            print(
                f"Wrong values for loglikelihood dictionary. Please check your implementation for the loglikelihood dictionary."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_naive_bayes_predict(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "tweet": "She smiled.",
                "logprior": 0.0,
                "loglikelihood": pickle.load(
                    open("./support_files/loglikelihood_test.pkl", "rb")
                ),
            },
            "expected": 1.5577981920239676,
        },
        {
            "name": "neutral_example_check",
            "input": {
                "tweet": "She did not answer my question.",
                "logprior": 0.0,
                "loglikelihood": pickle.load(
                    open("./support_files/loglikelihood_test.pkl", "rb")
                ),
            },
            "expected": 0.6025188572932385,
        },
        {
            "name": "negative_example_check",
            "input": {
                "tweet": "She did not answer my question :(",
                "logprior": 0.0,
                "loglikelihood": pickle.load(
                    open("./support_files/loglikelihood_test.pkl", "rb")
                ),
            },
            "expected": -6.924732171512272,
        },
        {
            "name": "positive_prior_check",
            "input": {
                "tweet": "She smiled.",
                "logprior": 0.6931471805599456,
                "loglikelihood": pickle.load(
                    open("./support_files/loglikelihood_test.pkl", "rb")
                ),
            },
            "expected": 2.2509453725839133,
        },
        {
            "name": "neutral_example_prior_check",
            "input": {
                "tweet": "She did not answer my question.",
                "logprior": 0.6931471805599456,
                "loglikelihood": pickle.load(
                    open("./support_files/loglikelihood_test.pkl", "rb")
                ),
            },
            "expected": 1.2956660378531841,
        },
        {
            "name": "negative_example_prior_check",
            "input": {
                "tweet": "She did not answer my question :(",
                "logprior": 0.6931471805599456,
                "loglikelihood": pickle.load(
                    open("./support_files/loglikelihood_test.pkl", "rb")
                ),
            },
            "expected": -6.231584990952325,
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
                f"Wrong output value. \n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def unittest_test_naive_bayes(target, test_x, test_y):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {
                "test_x": test_x,
                "test_y": test_y,
                "logprior": 0.0,
                "loglikelihood": pickle.load(
                    open("./support_files/loglikelihood_test.pkl", "rb")
                ),
            },
            "expected": 0.9955,
        },
        {
            "name": "smaller_check",
            "input": {
                "test_x": test_x[:100] + test_x[-100:],
                "test_y": np.concatenate((test_y[:100], test_y[-100:]), axis=0),
                "logprior": 0.0,
                "loglikelihood": pickle.load(
                    open("./support_files/loglikelihood_test.pkl", "rb")
                ),
            },
            "expected": 0.995,
        },
        {
            "name": "smaller_prior_check",
            "input": {
                "test_x": test_x[:100] + test_x[-100:],
                "test_y": np.concatenate((test_y[:100], test_y[-100:]), axis=0),
                "logprior": 0.6931471805599456,
                "loglikelihood": pickle.load(
                    open("./support_files/loglikelihood_test.pkl", "rb")
                ),
            },
            "expected": 0.995,
        },
        {
            "name": "small_check",
            "input": {
                "test_x": test_x[:150] + test_x[-100:],
                "test_y": np.concatenate((test_y[:150], test_y[-100:]), axis=0),
                "logprior": 0.0,
                "loglikelihood": pickle.load(
                    open("./support_files/loglikelihood_test.pkl", "rb")
                ),
            },
            "expected": 0.996,
        },
        {
            "name": "small_prior_check",
            "input": {
                "test_x": test_x[:150] + test_x[-100:],
                "test_y": np.concatenate((test_y[:150], test_y[-100:]), axis=0),
                "logprior": 0.6931471805599456,
                "loglikelihood": pickle.load(
                    open("./support_files/loglikelihood_test.pkl", "rb")
                ),
            },
            "expected": 0.996,
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
                f"Wrong output value for accuracy.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_get_ratio(target, freqs):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check",
            "input": {"freqs": freqs, "word": "happi"},
            "expected": {"positive": 162, "negative": 18, "ratio": 8.578947368421053},
        },
        {
            "name": "word_bad_check",
            "input": {"freqs": freqs, "word": "bad"},
            "expected": {"positive": 14, "negative": 54, "ratio": 0.2727272727272727},
        },
        {
            "name": "word_fun_check",
            "input": {"freqs": freqs, "word": "fun"},
            "expected": {"positive": 45, "negative": 20, "ratio": 2.1904761904761907},
        },
        {
            "name": "word_sad_check",
            "input": {"freqs": freqs, "word": "sad"},
            "expected": {"positive": 5, "negative": 100, "ratio": 0.0594059405940594},
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert isinstance(result, dict)
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
                f"Wrong output type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        for key, value in test_case["expected"].items():
            try:
                assert np.isclose(result[key], value)
                successful_cases += 1
            except:
                failed_cases.append(
                    {
                        "name": test_case["name"],
                        "expected": {key: value},
                        "got": {key: result[key]},
                    }
                )
                print(
                    f"Wrong output values.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
                )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")


def test_get_words_by_threshold(target, freqs):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "default_check1",
            "input": {"freqs": freqs, "label": 0, "threshold": 0.05},
            "expected": {
                ":(": {"positive": 1, "negative": 3675, "ratio": 0.000544069640914037},
                ":-(": {"positive": 0, "negative": 386, "ratio": 0.002583979328165375},
                "zayniscomingbackonjuli": {
                    "positive": 0,
                    "negative": 19,
                    "ratio": 0.05,
                },
                "26": {"positive": 0, "negative": 20, "ratio": 0.047619047619047616},
                ">:(": {"positive": 0, "negative": 43, "ratio": 0.022727272727272728},
                "lost": {"positive": 0, "negative": 19, "ratio": 0.05},
                "♛": {"positive": 0, "negative": 210, "ratio": 0.004739336492890996},
                "》": {"positive": 0, "negative": 210, "ratio": 0.004739336492890996},
                "beli̇ev": {
                    "positive": 0,
                    "negative": 35,
                    "ratio": 0.027777777777777776,
                },
                "wi̇ll": {"positive": 0, "negative": 35, "ratio": 0.027777777777777776},
                "justi̇n": {
                    "positive": 0,
                    "negative": 35,
                    "ratio": 0.027777777777777776,
                },
                "ｓｅｅ": {"positive": 0, "negative": 35, "ratio": 0.027777777777777776},
                "ｍｅ": {"positive": 0, "negative": 35, "ratio": 0.027777777777777776},
            },
        },
        {
            "name": "default_check2",
            "input": {"freqs": freqs, "label": 1, "threshold": 10},
            "expected": {
                "followfriday": {"positive": 23, "negative": 0, "ratio": 24.0},
                "commun": {"positive": 27, "negative": 1, "ratio": 14.0},
                ":)": {"positive": 2960, "negative": 2, "ratio": 987.0},
                "flipkartfashionfriday": {"positive": 16, "negative": 0, "ratio": 17.0},
                ":D": {"positive": 523, "negative": 0, "ratio": 524.0},
                ":p": {"positive": 104, "negative": 0, "ratio": 105.0},
                "influenc": {"positive": 16, "negative": 0, "ratio": 17.0},
                ":-)": {"positive": 552, "negative": 0, "ratio": 553.0},
                "here'": {"positive": 20, "negative": 0, "ratio": 21.0},
                "youth": {"positive": 14, "negative": 0, "ratio": 15.0},
                "bam": {"positive": 44, "negative": 0, "ratio": 45.0},
                "warsaw": {"positive": 44, "negative": 0, "ratio": 45.0},
                "shout": {"positive": 11, "negative": 0, "ratio": 12.0},
                ";)": {"positive": 22, "negative": 0, "ratio": 23.0},
                "stat": {"positive": 51, "negative": 0, "ratio": 52.0},
                "arriv": {"positive": 57, "negative": 4, "ratio": 11.6},
                "glad": {"positive": 41, "negative": 2, "ratio": 14.0},
                "blog": {"positive": 27, "negative": 0, "ratio": 28.0},
                "fav": {"positive": 11, "negative": 0, "ratio": 12.0},
                "fantast": {"positive": 9, "negative": 0, "ratio": 10.0},
                "fback": {"positive": 26, "negative": 0, "ratio": 27.0},
                "pleasur": {"positive": 10, "negative": 0, "ratio": 11.0},
                "←": {"positive": 9, "negative": 0, "ratio": 10.0},
                "aqui": {"positive": 9, "negative": 0, "ratio": 10.0},
            },
        },
        {
            "name": "low_threshold_check",
            "input": {"freqs": freqs, "label": 0, "threshold": 0.01},
            "expected": {
                ":(": {"positive": 1, "negative": 3675, "ratio": 0.000544069640914037},
                ":-(": {"positive": 0, "negative": 386, "ratio": 0.002583979328165375},
                "♛": {"positive": 0, "negative": 210, "ratio": 0.004739336492890996},
                "》": {"positive": 0, "negative": 210, "ratio": 0.004739336492890996},
            },
        },
        {
            "name": "high_threshold_check",
            "input": {"freqs": freqs, "label": 1, "threshold": 30},
            "expected": {
                ":)": {"positive": 2960, "negative": 2, "ratio": 987.0},
                ":D": {"positive": 523, "negative": 0, "ratio": 524.0},
                ":p": {"positive": 104, "negative": 0, "ratio": 105.0},
                ":-)": {"positive": 552, "negative": 0, "ratio": 553.0},
                "bam": {"positive": 44, "negative": 0, "ratio": 45.0},
                "warsaw": {"positive": 44, "negative": 0, "ratio": 45.0},
                "stat": {"positive": 51, "negative": 0, "ratio": 52.0},
            },
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert isinstance(result, dict)
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
                f"Wrong output type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert len(result) == len(test_case["expected"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": len(test_case["expected"]),
                    "got": len(result),
                }
            )
            print(
                f"Wrong number of elements in output dictionary.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        for key, value in test_case["expected"].items():
            for sec_key, sec_value in value.items():
                try:
                    assert np.isclose(result[key][sec_key], sec_value)
                    successful_cases += 1
                except:
                    failed_cases.append(
                        {
                            "name": test_case["name"],
                            "expected": {key: {sec_key: sec_value}},
                            "got": {key: result[key][sec_key]},
                        }
                    )
                    print(
                        f"Wrong output values.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
                    )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")
