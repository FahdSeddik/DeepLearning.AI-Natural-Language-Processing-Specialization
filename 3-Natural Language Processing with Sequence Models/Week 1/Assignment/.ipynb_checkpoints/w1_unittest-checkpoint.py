import re
import jax
import trax
import pickle
import inspect
import numpy as old_np
from trax import fastmath

random = fastmath.random


def test_tweet_to_tensor(target, Vocab):

    test_cases = [
        {
            "name": "simple_test_check1",
            "input": {
                "tweet": "@heyclaireee is back! thnx God!!! i'm so happy :)",
                "vocab_dict": Vocab,
                "unk_token": "__UNK__",
            },
            "expected": {
                "output_list": [444, 2, 304, 567, 56, 9],
                "output_type": type([]),
            },
        },
        {
            "name": "simple_test_check2",
            "input": {
                "tweet": "@heyclaireee is back! thnx God!!! i'm so happy :)",
                "vocab_dict": Vocab,
                "unk_token": "followfriday",
            },
            "expected": {
                "output_list": [444, 3, 304, 567, 56, 9],
                "output_type": type([]),
            },
        },
        {
            "name": "simple_test_check3",
            "input": {
                "tweet": "@BBCRadio3 thought it was my ears which were malfunctioning, thank goodness you cleared that one up with an apology :-)",
                "vocab_dict": Vocab,
                "unk_token": "__UNK__",
            },
            "expected": {
                "output_list": [60, 2992, 2, 22, 236, 1292, 45, 1354, 118],
                "output_type": type([]),
            },
        },
        {
            "name": "simple_test_check4",
            "input": {
                "tweet": "@BBCRadio3 thought it was my ears which were malfunctioning, thank goodness you cleared that one up with an apology :-)",
                "vocab_dict": Vocab,
                "unk_token": "__UNK__",
            },
            "expected": {
                "output_list": [-1, -10, 2, -22, 236, 1292, 45, -4531, 118],
                "output_type": type([]),
            },
        },
    ]

    failed_cases = []
    successful_cases = 0

    for test_case in test_cases:

        if test_case["name"] == "simple_test_check4":
            Vocab_not_global = {k: v for k, v in Vocab.items()}
            Vocab_not_global["thought"] = -1
            Vocab_not_global["ear"] = -10
            Vocab_not_global["thank"] = -22
            Vocab_not_global["apolog"] = -4531
            test_case["input"]["vocab_dict"] = Vocab_not_global

        result = target(**test_case["input"])

        try:
            assert result == test_case["expected"]["output_list"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "simple_test_check",
                    "expected": test_case["expected"]["output_list"],
                    "got": result,
                }
            )
            print(
                f"Output does not match with expected values. Maybe you can check the value you are using for unk_token variable. Also, try to avoid using the global dictionary Vocab.\n Expected: {failed_cases[-1].get('expected')}.\n Got: {failed_cases[-1].get('got')}."
            )

        try:

            assert isinstance(result, test_case["expected"]["output_type"])
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": "simple_test_check",
                    "expected": test_case["expected"]["output_list"],
                    "got": result,
                }
            )
            print(
                f"Output object does not have the correct type.\n Expected: {failed_cases[-1].get('expected')}.\n Got: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_data_generator(target):
    failed_cases = []
    successful_cases = 0

    result = next(target)

    # print(result)

    try:
        assert isinstance(result, tuple)
        successful_cases += 1
    except:
        failed_cases.append(
            {"name": "check_type", "expected": tuple, "got": type(result)}
        )
        print(
            f"Wrong output type from data_generator.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
        )

    try:
        assert len(result) == 3
        successful_cases += 1
    except:
        failed_cases.append({"name": "check_len", "expected": 3, "got": len(result)})
        print(
            f"Function data_generator returned a tuple with a different number of elements. Make sure you return inputs, targets, example_weights.\n\tExpected number of elements {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
        )

    try:
        assert result[0].shape == (4, 14)
        successful_cases += 1
    except:
        failed_cases.append(
            {"name": "check_shape0", "expected": (4, 14), "got": result[0].shape}
        )
        print(
            f"First output from data_generator (inputs) has the wrong shape.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
        )

    try:
        assert result[1].shape == (4,)
        successful_cases += 1
    except:
        failed_cases.append(
            {"name": "check_shape1", "expected": (4,), "got": result[0].shape}
        )
        print(
            f"Second output from data_generator (targets) has the wrong shape.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
        )

    try:
        assert result[2].shape == (4,)
        successful_cases += 1
    except:
        failed_cases.append(
            {"name": "check_shape2", "expected": (4,), "got": result[0].shape}
        )
        print(
            f"Third output from data_generator (example_weights) has the wrong shape.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
        )

    try:
        assert isinstance(result[0], jax.interpreters.xla.DeviceArray)
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "check_type_subarray0",
                "expected": jax.interpreters.xla.DeviceArray,
                "got": type(result[0]),
            }
        )
        print(
            f"First output from data_generator (inputs) has the wrong type.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
        )

    try:
        assert isinstance(result[1], jax.interpreters.xla.DeviceArray)
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "check_type_subarray1",
                "expected": jax.interpreters.xla.DeviceArray,
                "got": type(result[1]),
            }
        )
        print(
            f"Second output from data_generator (targets) has the wrong type.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
        )

    try:
        assert isinstance(result[2], jax.interpreters.xla.DeviceArray)
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "check_type_subarray2",
                "expected": jax.interpreters.xla.DeviceArray,
                "got": type(result[2]),
            }
        )
        print(
            f"Third output from data_generator (example_weights) has the wrong type.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
        )

    try:  # numpy.allclose
        assert old_np.allclose(
            result[0],
            jax.numpy.array(
                [
                    [3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0],
                    [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 9, 21, 22],
                    [5738, 2901, 3761, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [
                        858,
                        256,
                        3652,
                        5739,
                        307,
                        4458,
                        567,
                        1230,
                        2767,
                        328,
                        1202,
                        3761,
                        0,
                        0,
                    ],
                ]
            ),
        )
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "check_values_subarray0",
                "expected": jax.numpy.array(
                    [
                        [3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0],
                        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 9, 21, 22],
                        [5738, 2901, 3761, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [
                            858,
                            256,
                            3652,
                            5739,
                            307,
                            4458,
                            567,
                            1230,
                            2767,
                            328,
                            1202,
                            3761,
                            0,
                            0,
                        ],
                    ]
                ),
                "got": result[0],
            }
        )
        print(
            f"First output from data_generator (inputs) has the wrong type.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
        )

    try:
        assert old_np.allclose(result[1], jax.numpy.array([1, 1, 0, 0]),)
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "check_values_subarray1",
                "expected": jax.numpy.array([1, 1, 0, 0]),
                "got": result[1],
            }
        )
        print(
            f"Second output from data_generator (targets) has the wrong values.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
        )

    try:
        assert old_np.allclose(result[2], jax.numpy.array([1, 1, 1, 1]),)
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "check_values_subarray2",
                "expected": jax.numpy.array([1, 1, 1, 1]),
                "got": result[2],
            }
        )
        print(
            f"Third output from data_generator (example_weights) has the wrong values.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
        )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    return failed_cases, len(failed_cases) + successful_cases


def test_Relu(target):
    failed_cases = []
    successful_cases = 0

    test_cases = [
        {
            "name": "check_output1",
            "input": old_np.array([[-2.0, -1.0, 0.0], [0.0, 1.0, 2.0]], dtype=float),
            "expected": {
                "values": old_np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 2.0]]),
                "shape": (2, 3),
            },
        },
        {
            "name": "check_output2",
            "input": old_np.array(
                [
                    [-3.0, 1.0, -5.0, 4.0],
                    [-100.0, 3.0, -2.0, 0.0],
                    [-4.0, 0.0, 1.0, 5.0],
                ],
                dtype=float,
            ),
            "expected": {
                "values": old_np.array(
                    [[0.0, 1.0, 0.0, 4.0], [0.0, 3.0, 0.0, 0.0], [0.0, 0.0, 1.0, 5.0]]
                ),
                "shape": (3, 4),
            },
        },
    ]

    relu_layer = target()

    for test_case in test_cases:
        result = relu_layer(test_case["input"])
        try:
            assert result.shape == test_case["expected"]["shape"]
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["shape"],
                    "got": result.shape,
                }
            )
            print(
                f"Relu layer is modifying the input shape.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert old_np.allclose(result, test_case["expected"]["values"],)
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["values"],
                    "got": result,
                }
            )
            print(
                f"Output from Relu layer is incorrect.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_Dense(target):
    failed_cases = []
    successful_cases = 0

    test_cases = [
        {
            "name": "check_output1",
            "input": {"n_units": 10, "x": old_np.array([[2.0, 7.0, 25.0]])},
            "expected": {
                "output": jax.numpy.array(
                    [
                        [
                            -3.0395496,
                            0.9266802,
                            2.5414743,
                            -2.050473,
                            -1.9769388,
                            -2.582209,
                            -1.7952735,
                            0.94427425,
                            -0.8980402,
                            -3.7497487,
                        ]
                    ]
                ),
                "output_shape": (1, 10),
                "output_type": jax.interpreters.xla.DeviceArray,
                "weights_shape": (3, 10),
                "weights": jax.numpy.array(
                    [
                        [
                            -0.02837108,
                            0.09368162,
                            -0.10050076,
                            0.14165013,
                            0.10543301,
                            0.09108126,
                            -0.04265672,
                            0.0986188,
                            -0.05575325,
                            0.00153249,
                        ],
                        [
                            -0.20785688,
                            0.0554837,
                            0.09142365,
                            0.05744595,
                            0.07227863,
                            0.01210617,
                            -0.03237354,
                            0.16234995,
                            0.02450038,
                            -0.13809784,
                        ],
                        [
                            -0.06111237,
                            0.01403724,
                            0.08410042,
                            -0.1094358,
                            -0.10775021,
                            -0.11396459,
                            -0.05933381,
                            -0.01557652,
                            -0.03832145,
                            -0.11144515,
                        ],
                    ]
                ),
            },
        },
        {
            "name": "check_output_size",
            "input": {"n_units": 5, "x": old_np.array([[-1.0, 10.0, 0.0, 5.0]])},
            "expected": {
                "output": jax.numpy.array(
                    [[-1.4564116, -0.7315445, 0.14365998, 1.665177, 1.2354646]]
                ),
                "output_shape": (1, 5),
                "output_type": jax.interpreters.xla.DeviceArray,
                "weights_shape": (4, 5),
                "weights": jax.numpy.array(
                    [
                        [0.1054516, -0.09692889, -0.05946022, -0.00318858, 0.2410932],
                        [-0.18784496, -0.07847697, -0.03137085, 0.03337089, 0.17677031],
                        [-0.10277647, 0.14111717, -0.05084972, -0.05263776, 0.05031503],
                        [0.10549793, -0.00874074, 0.07958166, 0.2656559, -0.05822907],
                    ]
                ),
            },
        },
    ]

    random_key = random.get_prng(seed=0)

    for test_case in test_cases:
        dense_layer = target(n_units=test_case["input"]["n_units"])
        dense_layer.init(test_case["input"]["x"], random_key)
        result = dense_layer(test_case["input"]["x"])

        try:
            assert result.shape == test_case["expected"]["output_shape"]
            successful_cases += 1

        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["output_shape"],
                    "got": result.shape,
                }
            )
            print(
                f"Output from dense layer has the incorrect shape.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert old_np.allclose(result, test_case["expected"]["output"],)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["output"],
                    "got": result,
                }
            )
            print(
                f"Dense layer produced incorrect output. Check your weights or your output computation.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert isinstance(result, test_case["expected"]["output_type"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["output_type"],
                    "got": type(result),
                }
            )
            print(
                f"Output object has the incorrect type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert dense_layer.weights.shape == test_case["expected"]["weights_shape"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["weights_shape"],
                    "got": dense_layer.weights.shape,
                }
            )
            print(
                f"Weights matrix has the incorrect shape.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert old_np.allclose(
                dense_layer.weights, test_case["expected"]["weights"],
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["weights"],
                    "got": dense_layer.weights,
                }
            )
            print(
                f"Weights matrix values are incorrect.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_classifier(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "check_default_layer",
            "input": {"vocab_size": 9088, "embedding_dim": 256, "output_dim": 2},
            "expected": {
                "expected_str": "Serial[\n  Embedding_9088_256\n  Mean\n  Dense_2\n  LogSoftmax\n]",
                "expected_sublayers_types": [
                    trax.layers.core.Embedding,
                    trax.layers.base.PureLayer,
                    trax.layers.core.Dense,
                    trax.layers.base.PureLayer,
                ],
                "expected_type": trax.layers.combinators.Serial,
            },
        },
        {
            "name": "check_small_layer",
            "input": {"vocab_size": 100, "embedding_dim": 16, "output_dim": 4},
            "expected": {
                "expected_str": "Serial[\n  Embedding_100_16\n  Mean\n  Dense_4\n  LogSoftmax\n]",
                "expected_sublayers_types": [
                    trax.layers.core.Embedding,
                    trax.layers.base.PureLayer,
                    trax.layers.core.Dense,
                    trax.layers.base.PureLayer,
                ],
                "expected_type": trax.layers.combinators.Serial,
            },
        },
    ]

    for test_case in test_cases:
        # print(test_case["name"])
        classifier = target(**test_case["input"])
        proposed = str(classifier)

        try:
            assert proposed.replace(" ", "") == test_case["expected"][
                "expected_str"
            ].replace(" ", "")
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "str_rep_check",
                    "expected": test_case["expected"]["expected_str"],
                    "got": proposed,
                }
            )

            print(
                f"Wrong model.\n\tGot: {failed_cases[-1].get('got')},\n\tExpected:\n%s {failed_cases[-1].get('expected')}"
            )

        # Test the output type
        try:
            assert isinstance(classifier, test_case["expected"]["expected_type"])
            successful_cases += 1
            # Test the number of layers
        except:
            failed_cases.append(
                {
                    "name": "object_type_check",
                    "expected": test_case["expected"]["expected_type"],
                    "got": type(classifier),
                }
            )
            print(
                "The classifier is not an object of type",
                test_case["expected"]["expected_type"],
            )

        try:
            # Test
            assert len(classifier.sublayers) == len(
                test_case["expected"]["expected_sublayers_types"]
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "num_layers_check",
                    "expected": len(test_case["expected"]["expected_sublayers_types"]),
                    "got": len(classifier.sublayers),
                }
            )
            print(
                f"The number of sublayers does not match.\n\tGot: {len(classifier.sublayers)}.\n\tExpected: {len(test_case['expected']['expected_sublayers_types'])}"
            )

        try:
            sublayers_type = lambda x: list(map(type, x.sublayers))
            classifier_sublayers_type = sublayers_type(classifier)

            for i in range(len(test_case["expected"]["expected_sublayers_types"])):
                assert str(classifier_sublayers_type[i]) == str(
                    test_case["expected"]["expected_sublayers_types"][i]
                )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "sublayers_type_check",
                    "expected": test_case["expected"]["expected_sublayers_types"],
                    "got": classifier_sublayers_type,
                }
            )
            print(
                f"Classifier sublayers do not have the correct type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_train_model(target):

    successful_cases = 0
    failed_cases = []
    output_loop = target

    try:
        assert isinstance(output_loop, trax.supervised.training.Loop)
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "check_loop_type",
                "expected": trax.supervised.training.Loop,
                "got": type(output_loop),
            }
        )
        print(
            f"Wrong loss functions. CrossEntropyLoss_in3 was expected. Got {failed_cases[-1].get('got')}."
        )

    try:
        strlabel = str(output_loop._tasks[0]._loss_layer)
        assert strlabel == "CrossEntropyLoss_in3"
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "loss_layer_check",
                "expected": "CrossEntropyLoss_in3",
                "got": strlabel,
            }
        )
        print(
            f"Wrong loss functions. CrossEntropyLoss_in3 was expected. Got {failed_cases[-1].get('got')}."
        )

    # Test the optimizer parameter
    try:
        assert isinstance(output_loop._tasks[0].optimizer, trax.optimizers.adam.Adam)
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "optimizer_check",
                "expected": trax.optimizers.adam.Adam,
                "got": type(output_loop._tasks[0].optimizer),
            }
        )
        print(
            f"Wrong optimizer. Expected {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
        )

    opt_params_dict = {
        "weight_decay_rate": fastmath.numpy.array(1.0e-5),
        "b1": fastmath.numpy.array(0.9),
        "b2": fastmath.numpy.array(0.999),
        "eps": fastmath.numpy.array(1.0e-5),
        "learning_rate": fastmath.numpy.array(0.01),
    }

    try:
        assert output_loop._tasks[0].optimizer.opt_params == opt_params_dict
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "optimizer_parameters",
                "expected": opt_params_dict,
                "got": output_loop._tasks[0].optimizer.opt_params,
            }
        )

        print(
            f"Optimizer has the wrong parameters.\n\tExpected {opt_params_dict}.\n\tGot {output_loop._tasks[0].optimizer.opt_params}."
        )

    # Test the _n_steps_per_checkpoint parameter
    try:
        assert output_loop._tasks[0]._n_steps_per_checkpoint == 10
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "n_steps_per_checkpoint",
                "expected": 10,
                "got": output_loop._tasks[0]._n_steps_per_checkpoint,
            }
        )
        print(
            f"Wrong checkpoint step frequency.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot:{failed_cases[-1].get('got')}"
        )

    # Test the metrics in the evaluation task
    test_func = lambda x: list(map(str, x._eval_tasks[0]._metric_names))

    try:
        assert test_func(output_loop) == ["CrossEntropyLoss", "Accuracy"]
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "metrics_check",
                "expected": ["CrossEntropyLoss", "Accuracy"],
                "got": test_func(output_loop),
            }
        )
        print(
            f"Wrong metrics in evaluations task. Expected {failed_cases[-1].get('expected')}. Got {failed_cases[-1].get('got')}."
        )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_compute_accuracy(target):
    successful_cases = 0
    failed_cases = []

    test_cases = [
        {
            "name": "check_default_results",
            "input": {
                "preds": old_np.array(
                    [
                        [-2.34253144e00, -1.01018906e-01],
                        [-4.91725826e00, -7.34615326e-03],
                        [-5.30653572e00, -4.97150421e-03],
                        [-4.57595682e00, -1.03497505e-02],
                        [-5.12225914e00, -5.98025322e-03],
                        [-5.67220068e00, -3.44634056e-03],
                        [-4.69125652e00, -9.21750069e-03],
                        [-4.69832897e00, -9.15217400e-03],
                        [-6.24671125e00, -1.93881989e-03],
                        [-5.59627056e00, -3.71861458e-03],
                        [-5.43021059e00, -4.39167023e-03],
                        [-3.28867722e00, -3.80167961e-02],
                        [-6.39779758e00, -1.66654587e-03],
                        [-4.83748817e00, -7.95841217e-03],
                        [-4.93317604e00, -7.22980499e-03],
                        [-5.60852242e00, -3.67331505e-03],
                        [-5.01646233e00, -6.64997101e-03],
                        [-4.03559542e00, -1.78332329e-02],
                        [-4.00632906e00, -1.83677673e-02],
                        [-5.12225914e00, -5.98025322e-03],
                        [-5.24182701e00, -5.30457497e-03],
                        [-4.84393072e00, -7.90715218e-03],
                        [-4.75897694e00, -8.61144066e-03],
                        [-4.07698059e00, -1.71039104e-02],
                        [-4.07379913e00, -1.71589851e-02],
                        [-4.61745882e00, -9.92703438e-03],
                        [-4.71810913e00, -8.97216797e-03],
                        [-4.30195522e00, -1.36344433e-02],
                        [-3.99270916e00, -1.86219215e-02],
                        [-4.71889591e00, -8.96501541e-03],
                        [-5.84391356e00, -2.90155411e-03],
                        [-4.05747557e00, -1.74438953e-02],
                        [-7.06982613e-03, -4.95544052e00],
                        [-1.50053501e-02, -4.20684242e00],
                        [-1.14690661e-01, -2.22231436e00],
                        [-6.53460026e-02, -2.76055384e00],
                        [-2.26595640e-01, -1.59574771e00],
                        [-6.77399635e-02, -2.72575760e00],
                        [-5.98562956e-02, -2.84558773e00],
                        [-1.18734002e-01, -2.18964934e00],
                        [-5.94434738e-02, -2.85230422e00],
                        [-1.12637758e-01, -2.23936844e00],
                        [-4.58146334e-02, -3.10597134e00],
                        [-4.30107117e-02, -3.16773319e00],
                        [-9.14014578e-02, -2.43784523e00],
                        [-9.28543806e-02, -2.42279029e00],
                        [-5.39070368e-02, -2.94732571e00],
                        [-6.53460026e-02, -2.76055384e00],
                        [-1.11358166e-02, -4.50315619e00],
                        [-9.96928215e-02, -2.35509443e00],
                        [-5.43351173e-02, -2.93963003e00],
                        [-8.07831287e-02, -2.55610609e00],
                        [-6.53460026e-02, -2.76055384e00],
                        [-8.74786377e-02, -2.47978020e00],
                        [-8.04913044e-02, -2.55958152e00],
                        [-3.72824669e-02, -3.30781627e00],
                        [-6.28944635e-02, -2.79757881e00],
                        [-2.12790966e-02, -3.86065006e00],
                        [-3.54200602e-02, -3.35813284e00],
                        [-2.09717751e-02, -3.87504196e00],
                        [-5.10288477e-02, -3.00077057e00],
                        [-7.36353397e-02, -2.64522123e00],
                        [-1.10135078e-02, -4.51413631e00],
                        [-5.87701797e-04, -7.43953419e00],
                    ]
                ),
                "y": old_np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                    ]
                ),
                "y_weights": old_np.array(
                    [
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                        1,
                    ]
                ),
            },
            "expected": (
                jax.numpy.array(1.0),
                jax.numpy.array(64.0),
                jax.numpy.array(64.0),
            ),
            "error": "The compute_accuracy function's output is wrong.",
        },
        {
            "name": "check_results",
            "input": {
                "preds": old_np.array(
                    [
                        [-2.1831987, -0.11955023],
                        [-2.8474317, -0.05974269],
                        [-2.6502347, -0.07325327],
                        [-2.7246172, -0.06781995],
                        [-2.5021744, -0.08545625],
                        [-0.10777152, -2.2811432],
                        [-0.10801494, -2.279008],
                        [-0.17963344, -1.80531],
                        [-0.09246862, -2.4267645],
                        [-0.05025101, -3.0157456],
                    ],
                ),
                "y": old_np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0],),
                "y_weights": old_np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1],),
            },
            "expected": (
                jax.numpy.array(1.0),
                jax.numpy.array(10.0),
                jax.numpy.array(10.0),
            ),
            "error": "The compute_accuracy function's output is wrong.",
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert result[0] == test_case["expected"][0]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "check_results0",
                    "expected": test_case["expected"][0],
                    "got": result[0],
                }
            )

            print(
                f"{test_case['error']}.\nFirst element, corresponding to accuracy variable is wrong.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
            )

        try:
            assert result[1] == test_case["expected"][1]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "check_results0",
                    "expected": test_case["expected"][1],
                    "got": result[1],
                }
            )
            print(
                f"{test_case['error']}.\nSecond element, corresponding to weighted_num_correct variable is wrong.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
            )

        try:
            assert result[2] == test_case["expected"][2]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "check_results0",
                    "expected": test_case["expected"][2],
                    "got": result[2],
                }
            )
            print(
                f"{test_case['error']}.\nThird element, corresponding to sum_weights variable is wrong.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def unittest_test_model(target, generator, model):
    successful_cases = 0
    failed_cases = []
    
    result = target(generator, model)

    try:
        assert 0.99 <= result <= 1.0
    except:
        failed_cases.append({"name": "output_check", "expected": 0.9950, "got": result})
        print(f"Accuracy should be above 0.99.\n\tGot{failed_cases[-1].get('got')}")

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases

