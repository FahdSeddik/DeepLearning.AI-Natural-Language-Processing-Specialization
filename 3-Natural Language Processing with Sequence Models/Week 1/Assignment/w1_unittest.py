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
        assert issubclass(
            trax.fastmath.numpy.dtype(result[0]).type, trax.fastmath.numpy.integer
        )
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "check_type0",
                "expected": trax.fastmath.numpy.integer,
                "got": trax.fastmath.numpy.dtype(result[0]).type,
            }
        )
        print(
            f"First output from data_generator (inputs) has elements with the wrong type.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
        )

    try:
        assert issubclass(
            trax.fastmath.numpy.dtype(result[1]).type, trax.fastmath.numpy.integer
        )
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "check_type1",
                "expected": trax.fastmath.numpy.integer,
                "got": trax.fastmath.numpy.dtype(result[1]).type,
            }
        )
        print(
            f"Second output from data_generator (targets) has elements with the wrong type.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
        )

    try:
        assert issubclass(
            trax.fastmath.numpy.dtype(result[2]).type, trax.fastmath.numpy.integer
        )
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "check_type2",
                "expected": trax.fastmath.numpy.integer,
                "got": trax.fastmath.numpy.dtype(result[2]).type,
            }
        )
        print(
            f"Third output from data_generator (example_weights) has elements with the wrong type.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot {failed_cases[-1].get('got')}."
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
                
    try: 
        for i in range(10):    
            tmp_inputs, tmp_targets, tmp_example_weights = next(target)
    except Exception as e:
        failed_cases.append(
            {
                "name": "check_pos_neg_indexes",
                "expected": "",
                "got": e,
            }
        )
        print(
            f"Data generator is not able to run properly through several batches.\n\tGot the following error: {failed_cases[-1].get('got')}."
        )
        
    try:
        for i in range(10):    
            tmp_inputs, tmp_targets, tmp_example_weights = next(target)        
                    
        assert tmp_inputs.shape[0] == 4 # Must be equal to the batch size
    except:
        failed_cases.append(
            {
                "name": "input_size_several_iterations",
                "expected": 4,
                "got": tmp_inputs.shape[0],
            }
        )
        print(
            f"Data generator is not generating the same batch size after several iterations.\n\tExpected {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
        )
        

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


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
            f"Wrong object type. {failed_cases[-1].get('expected')} was expected. Got {failed_cases[-1].get('got')}."
        )

    try:
        strlabel = str(output_loop._tasks[0]._loss_layer)
        assert strlabel == "WeightedCategoryCrossEntropy_in3"
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "loss_layer_check",
                "expected": "WeightedCategoryCrossEntropy_in3",
                "got": strlabel,
            }
        )
        print(
            f"Wrong loss function. {failed_cases[-1].get('expected')} was expected. Got {failed_cases[-1].get('got')}."
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
        assert test_func(output_loop) == [
            "WeightedCategoryCrossEntropy",
            "WeightedCategoryAccuracy",
        ]  # ["CrossEntropyLoss", "Accuracy"]
        successful_cases += 1
    except:
        failed_cases.append(
            {
                "name": "metrics_check",
                "expected": [
                    "WeightedCategoryCrossEntropy",
                    "WeightedCategoryAccuracy",
                ],
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
        {
            "name": "check_results",
            "input": {
                "preds": old_np.array(
                    [
                        [-5.9710550e00, -2.5548935e-03],
                        [-1.6250032e-01, -1.8972251e00],
                        [-5.7942066e00, -3.0498505e-03],
                        [-3.8784173e00, -2.0900488e-02],
                        [-4.7201972e00, -8.9533329e-03],
                        [-4.4713774e00, -1.1497498e-02],
                        [-6.1483717e00, -2.1390915e-03],
                        [-4.3007736e00, -1.3650894e-02],
                        [-1.6250032e-01, -1.8972251e00],
                        [-8.0825253e00, -3.0899048e-04],
                        [-5.9081078e00, -2.7210712e-03],
                        [-5.9345040e00, -2.6500225e-03],
                        [-2.9287841e00, -5.4944158e-02],
                        [-7.0137210e00, -8.9979172e-04],
                        [-4.8264866e00, -8.0468655e-03],
                        [-5.9972801e00, -2.4886131e-03],
                        [-6.0228448e00, -2.4256706e-03],
                        [-8.6874599e00, -1.6880035e-04],
                        [-6.3576012e00, -1.7349720e-03],
                        [-3.8556619e00, -2.1386743e-02],
                        [-3.9758420e00, -1.8941760e-02],
                        [-7.4882288e00, -5.5980682e-04],
                        [-1.1433375e00, -3.8383099e-01],
                        [-3.7842584e00, -2.2987843e-02],
                        [-6.6745539e00, -1.2633801e-03],
                        [-2.0699925e00, -1.3488865e-01],
                        [-5.0829401e00, -6.2210560e-03],
                        [-4.5627761e-01, -1.0041330e00],
                        [-6.2781792e00, -1.8785000e-03],
                        [-5.5470734e00, -3.9064884e-03],
                        [-3.3517728e00, -3.5650134e-02],
                        [-7.1239138e00, -8.0585480e-04],
                        [-2.5272369e-05, -1.0586840e01],
                        [-1.4619827e-02, -4.2326741e00],
                        [-1.1920929e-05, -1.1328121e01],
                        [-6.6683292e-03, -5.0137224e00],
                        [-4.5704842e-04, -7.6912327e00],
                        [-1.1444092e-04, -9.0741482e00],
                        [-9.0122223e-05, -9.3118935e00],
                        [-8.9740753e-04, -7.0164309e00],
                        [-4.6300888e-04, -7.6779804e00],
                        [-1.0261536e-03, -6.8824816e00],
                        [-2.8095245e-03, -5.8761311e00],
                        [-4.7826767e-04, -7.6454802e00],
                        [-3.8766861e-04, -7.8551168e00],
                        [-1.0950565e-03, -6.8173752e00],
                        [-4.4965744e-04, -7.7072825e00],
                        [-1.6498566e-04, -8.7095642e00],
                        [-4.3435097e-03, -5.4412403e00],
                        [-9.1114044e-03, -4.7027726e00],
                        [-3.1089783e-04, -8.0764790e00],
                        [-1.9211769e-03, -6.2557869e00],
                        [-2.4127960e-04, -8.3293018e00],
                        [-1.6641617e-04, -8.7014294e00],
                        [-5.2452087e-05, -9.8545036e00],
                        [-1.4781952e-05, -1.1122052e01],
                        [-4.6868324e-03, -5.3653350e00],
                        [-1.0411739e-03, -6.8679361e00],
                        [-3.2813549e-03, -5.7211237e00],
                        [-4.4808388e-03, -5.4101715e00],
                        [-7.7366829e-04, -7.1647758e00],
                        [-8.0323219e-04, -7.1273246e00],
                        [-2.4318695e-04, -8.3220530e00],
                        [-2.0945787e-02, -3.8762670e00],
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
                jax.numpy.array(0.953125),
                jax.numpy.array(61.0),
                jax.numpy.array(64.0),
            ),
            "error": "The compute_accuracy function's output is wrong.",
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert result[0] == test_case["expected"][0] or old_np.isclose(
                result[0], test_case["expected"][0], atol=1e-5
            )
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
        assert 0.98 <= result <= 1.0
    except:
        failed_cases.append({"name": "output_check", "expected": 0.9881, "got": result})
        print(f"Accuracy should be above 0.98.\n\tGot{failed_cases[-1].get('got')}")

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases
