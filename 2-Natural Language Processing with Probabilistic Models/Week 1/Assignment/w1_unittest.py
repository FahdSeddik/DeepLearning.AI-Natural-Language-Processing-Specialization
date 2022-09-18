import re
import collections
from collections import Counter
import numpy as np
import pandas as pd


def test_process_data(target):

    failed_cases = []
    successful_cases = 0

    test_cases = [
        {
            "name": "default_check",
            "input": {"file_name": "./data/shakespeare.txt"},
            "expected": {
                "expected_output_head": [
                    "o",
                    "for",
                    "a",
                    "muse",
                    "of",
                    "fire",
                    "that",
                    "would",
                    "ascend",
                    "the",
                ],
                "expected_output_tail": [
                    "whilst",
                    "you",
                    "abide",
                    "here",
                    "enobarbus",
                    "humbly",
                    "sir",
                    "i",
                    "thank",
                    "you",
                ],
                "expected_n_words": 6116,
            },
        }
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert result[:10] == test_case["expected"]["expected_output_head"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "default_check",
                    "expected": test_case["expected"]["expected_output_head"],
                    "got": result[:10],
                }
            )
            print(
                f"Wrong output. Take a look at how you are reading the file.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert result[-10:] == test_case["expected"]["expected_output_tail"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "default_check",
                    "expected": test_case["expected"]["expected_output_tail"],
                    "got": result[:10],
                }
            )
            print(
                f"Wrong output. Take a look at how you are reading the file.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        n_words = len(set(result))
        try:
            assert n_words == test_case["expected"]["expected_n_words"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": "default_check",
                    "expected": test_case["expected"]["expected_n_words"],
                    "got": n_words,
                }
            )
            print(
                f"Wrong number of words. Take a look at how you are extracting every word in the list.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_get_count(target, word_l):
    failed_cases = []
    successful_cases = 0

    test_cases = [
        {
            "name": "default_check",
            "input": {"word_l": word_l},
            "expected": {
                "expected_key_value_pairs": 6116,
                "expected_count_thee": 240,
                "expected_count_esteemed": 3,
                "expected_count_your": 397,
            },
        },
        {
            "name": "small_head_set_check",
            "input": {"word_l": word_l[:1000]},
            "expected": {
                "expected_key_value_pairs": 487,
                "expected_count_thee": 0,
                "expected_count_esteemed": -1,
                "expected_count_your": 4,
            },
        },
        {
            "name": "medium_tail_set_check",
            "input": {"word_l": word_l[-10000:]},
            "expected": {
                "expected_key_value_pairs": 2080,
                "expected_count_thee": 18,
                "expected_count_esteemed": 1,
                "expected_count_your": 57,
            },
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert isinstance(result, collections.Counter) or isinstance(result, dict)
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": "dict or collections.Counter",
                    "got": str(type(result)),
                }
            )
            print(
                f"Wrong output type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert len(result) == test_case["expected"]["expected_key_value_pairs"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["expected_n_words"],
                    "got": n_words,
                }
            )
            print(
                f"Wrong number of key-value pairs. Check that you are not skipping any word in the loop.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert result.get("thee", 0) == test_case["expected"]["expected_count_thee"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["expected_count_thee"],
                    "got": result.get("thee", 0),
                }
            )
            print(
                f"Wrong word count for word 'thee'. Check your loop implementation.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert (
                result.get("esteemed", -1)
                == test_case["expected"]["expected_count_esteemed"]
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["expected_count_esteemed"],
                    "got": result.get("esteemed", -1),
                }
            )
            print(
                f"Wrong word count for word 'esteemed'. Check your loop implementation.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert result.get("your", 0) == test_case["expected"]["expected_count_your"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["expected_count_your"],
                    "got": result.get("your", 0),
                }
            )
            print(
                f"Wrong word count for word 'esteemed'. Check your loop implementation.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_get_probs(target, word_count_dict):
    failed_cases = []
    successful_cases = 0

    test_cases = [
        {
            "name": "default_check",
            "input": {"word_count_dict": word_count_dict},
            "expected": {
                "expected_prob_length": 6116,
                "expected_prob_thee": 0.0045,
                "expected_prob_esteemed": 0.0001,
                "expected_prob_your": 0.0074,
            },
        },
        {
            "name": "small_check",
            "input": {
                "word_count_dict": {
                    key: word_count_dict[key]
                    for key in [
                        "for",
                        "a",
                        "fire",
                        "ascend",
                        "esteemed",
                        "your",
                        "thee",
                    ]
                }
            },
            "expected": {
                "expected_prob_length": 7,
                "expected_prob_thee": 0.1267,
                "expected_prob_esteemed": 0.0016,
                "expected_prob_your": 0.2096,
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
                f"Wrong output type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert len(result) == test_case["expected"]["expected_prob_length"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["expected_prob_length"],
                    "got": len(result),
                }
            )
            print(
                f"Wrong number of probabilities in the dictionary.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.isclose(
                round(result["thee"], 4), test_case["expected"]["expected_prob_thee"],
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["expected_prob_thee"],
                    "got": result["thee"],
                }
            )
            print(
                f"Wrong probability for word 'thee'.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.isclose(
                round(result["esteemed"], 4),
                test_case["expected"]["expected_prob_esteemed"],
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["expected_prob_esteemed"],
                    "got": result["esteemed"],
                }
            )
            print(
                f"Wrong probability for word 'esteemed'.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.isclose(
                round(result["your"], 4), test_case["expected"]["expected_prob_your"],
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["expected_prob_your"],
                    "got": result["your"],
                }
            )
            print(
                f"Wrong probability for word 'your'.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_delete_letter(target):
    failed_cases = []
    successful_cases = 0

    test_cases = [
        {
            "name": "default_check",
            "input": {"word": "cans"},
            "expected": ["ans", "cns", "cas", "can"],
        },
        {
            "name": "small_default_check",
            "input": {"word": "at"},
            "expected": ["t", "a"],
        },
        {
            "name": "long_check",
            "input": {"word": "esteemed"},
            "expected": [
                "steemed",
                "eteemed",
                "eseemed",
                "estemed",
                "estemed",
                "esteeed",
                "esteemd",
                "esteeme",
            ],
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert isinstance(result, list)
            successful_cases += 1
        except:
            failed_cases.append(
                {"name": test_case["name"], "expected": list, "got": type(result)}
            )
            print(
                f"Wrong output type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert sorted(result) == sorted(test_case["expected"])
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
                f"Wrong output values.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_switch_letter(target):
    failed_cases = []
    successful_cases = 0

    test_cases = [
        {
            "name": "default_check",
            "input": {"word": "eta"},
            "expected": ["tea", "eat"],
        },
        {"name": "small_default_check", "input": {"word": "at"}, "expected": ["ta"],},
        {
            "name": "long_check",
            "input": {"word": "satr"},
            "expected": ["astr", "star", "sart"],
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert isinstance(result, list)
            successful_cases += 1
        except:
            failed_cases.append(
                {"name": test_case["name"], "expected": list, "got": type(result)}
            )
            print(
                f"Wrong output type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert sorted(result) == sorted(test_case["expected"])
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
                f"Wrong output values.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_replace_letter(target):
    failed_cases = []
    successful_cases = 0

    test_cases = [
        {
            "name": "default_check",
            "input": {"word": "can"},
            "expected": [
                "aan",
                "ban",
                "caa",
                "cab",
                "cac",
                "cad",
                "cae",
                "caf",
                "cag",
                "cah",
                "cai",
                "caj",
                "cak",
                "cal",
                "cam",
                "cao",
                "cap",
                "caq",
                "car",
                "cas",
                "cat",
                "cau",
                "cav",
                "caw",
                "cax",
                "cay",
                "caz",
                "cbn",
                "ccn",
                "cdn",
                "cen",
                "cfn",
                "cgn",
                "chn",
                "cin",
                "cjn",
                "ckn",
                "cln",
                "cmn",
                "cnn",
                "con",
                "cpn",
                "cqn",
                "crn",
                "csn",
                "ctn",
                "cun",
                "cvn",
                "cwn",
                "cxn",
                "cyn",
                "czn",
                "dan",
                "ean",
                "fan",
                "gan",
                "han",
                "ian",
                "jan",
                "kan",
                "lan",
                "man",
                "nan",
                "oan",
                "pan",
                "qan",
                "ran",
                "san",
                "tan",
                "uan",
                "van",
                "wan",
                "xan",
                "yan",
                "zan",
            ],
        },
        {
            "name": "small_default_check",
            "input": {"word": "at"},
            "expected": [
                "aa",
                "ab",
                "ac",
                "ad",
                "ae",
                "af",
                "ag",
                "ah",
                "ai",
                "aj",
                "ak",
                "al",
                "am",
                "an",
                "ao",
                "ap",
                "aq",
                "ar",
                "as",
                "au",
                "av",
                "aw",
                "ax",
                "ay",
                "az",
                "bt",
                "ct",
                "dt",
                "et",
                "ft",
                "gt",
                "ht",
                "it",
                "jt",
                "kt",
                "lt",
                "mt",
                "nt",
                "ot",
                "pt",
                "qt",
                "rt",
                "st",
                "tt",
                "ut",
                "vt",
                "wt",
                "xt",
                "yt",
                "zt",
            ],
        },
        {
            "name": "long_check",
            "input": {"word": "star"},
            "expected": [
                "atar",
                "btar",
                "ctar",
                "dtar",
                "etar",
                "ftar",
                "gtar",
                "htar",
                "itar",
                "jtar",
                "ktar",
                "ltar",
                "mtar",
                "ntar",
                "otar",
                "ptar",
                "qtar",
                "rtar",
                "saar",
                "sbar",
                "scar",
                "sdar",
                "sear",
                "sfar",
                "sgar",
                "shar",
                "siar",
                "sjar",
                "skar",
                "slar",
                "smar",
                "snar",
                "soar",
                "spar",
                "sqar",
                "srar",
                "ssar",
                "staa",
                "stab",
                "stac",
                "stad",
                "stae",
                "staf",
                "stag",
                "stah",
                "stai",
                "staj",
                "stak",
                "stal",
                "stam",
                "stan",
                "stao",
                "stap",
                "staq",
                "stas",
                "stat",
                "stau",
                "stav",
                "staw",
                "stax",
                "stay",
                "staz",
                "stbr",
                "stcr",
                "stdr",
                "ster",
                "stfr",
                "stgr",
                "sthr",
                "stir",
                "stjr",
                "stkr",
                "stlr",
                "stmr",
                "stnr",
                "stor",
                "stpr",
                "stqr",
                "strr",
                "stsr",
                "sttr",
                "stur",
                "stvr",
                "stwr",
                "stxr",
                "styr",
                "stzr",
                "suar",
                "svar",
                "swar",
                "sxar",
                "syar",
                "szar",
                "ttar",
                "utar",
                "vtar",
                "wtar",
                "xtar",
                "ytar",
                "ztar",
            ],
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert isinstance(result, list)
            successful_cases += 1
        except:
            failed_cases.append(
                {"name": test_case["name"], "expected": list, "got": type(result)}
            )
            print(
                f"Wrong output type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert len(set(map(len, result))) == 1
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": len(test_case["input"]["word"]),
                    "got": set(map(len, result)),
                }
            )
            print(
                f"The list has elements with a different length than the input length.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert sorted(result) == sorted(test_case["expected"])
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
                f"Wrong output values.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_insert_letter(target):
    failed_cases = []
    successful_cases = 0

    test_cases = [
        {
            "name": "default_check",
            "input": {"word": "at"},
            "expected": [
                "aat",
                "bat",
                "cat",
                "dat",
                "eat",
                "fat",
                "gat",
                "hat",
                "iat",
                "jat",
                "kat",
                "lat",
                "mat",
                "nat",
                "oat",
                "pat",
                "qat",
                "rat",
                "sat",
                "tat",
                "uat",
                "vat",
                "wat",
                "xat",
                "yat",
                "zat",
                "aat",
                "abt",
                "act",
                "adt",
                "aet",
                "aft",
                "agt",
                "aht",
                "ait",
                "ajt",
                "akt",
                "alt",
                "amt",
                "ant",
                "aot",
                "apt",
                "aqt",
                "art",
                "ast",
                "att",
                "aut",
                "avt",
                "awt",
                "axt",
                "ayt",
                "azt",
                "ata",
                "atb",
                "atc",
                "atd",
                "ate",
                "atf",
                "atg",
                "ath",
                "ati",
                "atj",
                "atk",
                "atl",
                "atm",
                "atn",
                "ato",
                "atp",
                "atq",
                "atr",
                "ats",
                "att",
                "atu",
                "atv",
                "atw",
                "atx",
                "aty",
                "atz",
            ],
        },
        {
            "name": "long_check",
            "input": {"word": "can"},
            "expected": [
                "acan",
                "bcan",
                "ccan",
                "dcan",
                "ecan",
                "fcan",
                "gcan",
                "hcan",
                "ican",
                "jcan",
                "kcan",
                "lcan",
                "mcan",
                "ncan",
                "ocan",
                "pcan",
                "qcan",
                "rcan",
                "scan",
                "tcan",
                "ucan",
                "vcan",
                "wcan",
                "xcan",
                "ycan",
                "zcan",
                "caan",
                "cban",
                "ccan",
                "cdan",
                "cean",
                "cfan",
                "cgan",
                "chan",
                "cian",
                "cjan",
                "ckan",
                "clan",
                "cman",
                "cnan",
                "coan",
                "cpan",
                "cqan",
                "cran",
                "csan",
                "ctan",
                "cuan",
                "cvan",
                "cwan",
                "cxan",
                "cyan",
                "czan",
                "caan",
                "cabn",
                "cacn",
                "cadn",
                "caen",
                "cafn",
                "cagn",
                "cahn",
                "cain",
                "cajn",
                "cakn",
                "caln",
                "camn",
                "cann",
                "caon",
                "capn",
                "caqn",
                "carn",
                "casn",
                "catn",
                "caun",
                "cavn",
                "cawn",
                "caxn",
                "cayn",
                "cazn",
                "cana",
                "canb",
                "canc",
                "cand",
                "cane",
                "canf",
                "cang",
                "canh",
                "cani",
                "canj",
                "cank",
                "canl",
                "canm",
                "cann",
                "cano",
                "canp",
                "canq",
                "canr",
                "cans",
                "cant",
                "canu",
                "canv",
                "canw",
                "canx",
                "cany",
                "canz",
            ],
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert isinstance(result, list)
            successful_cases += 1
        except:
            failed_cases.append(
                {"name": test_case["name"], "expected": list, "got": type(result)}
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
                f"Output list do not have the expected number of elements.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert sorted(result) == sorted(test_case["expected"])
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
                f"Wrong output values.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_edit_one_letter(target):
    failed_cases = []
    successful_cases = 0

    test_cases = [
        {
            "name": "default_check",
            "input": {"word": "at"},
            "expected": set(
                [
                    "a",
                    "aa",
                    "aat",
                    "ab",
                    "abt",
                    "ac",
                    "act",
                    "ad",
                    "adt",
                    "ae",
                    "aet",
                    "af",
                    "aft",
                    "ag",
                    "agt",
                    "ah",
                    "aht",
                    "ai",
                    "ait",
                    "aj",
                    "ajt",
                    "ak",
                    "akt",
                    "al",
                    "alt",
                    "am",
                    "amt",
                    "an",
                    "ant",
                    "ao",
                    "aot",
                    "ap",
                    "apt",
                    "aq",
                    "aqt",
                    "ar",
                    "art",
                    "as",
                    "ast",
                    "ata",
                    "atb",
                    "atc",
                    "atd",
                    "ate",
                    "atf",
                    "atg",
                    "ath",
                    "ati",
                    "atj",
                    "atk",
                    "atl",
                    "atm",
                    "atn",
                    "ato",
                    "atp",
                    "atq",
                    "atr",
                    "ats",
                    "att",
                    "atu",
                    "atv",
                    "atw",
                    "atx",
                    "aty",
                    "atz",
                    "au",
                    "aut",
                    "av",
                    "avt",
                    "aw",
                    "awt",
                    "ax",
                    "axt",
                    "ay",
                    "ayt",
                    "az",
                    "azt",
                    "bat",
                    "bt",
                    "cat",
                    "ct",
                    "dat",
                    "dt",
                    "eat",
                    "et",
                    "fat",
                    "ft",
                    "gat",
                    "gt",
                    "hat",
                    "ht",
                    "iat",
                    "it",
                    "jat",
                    "jt",
                    "kat",
                    "kt",
                    "lat",
                    "lt",
                    "mat",
                    "mt",
                    "nat",
                    "nt",
                    "oat",
                    "ot",
                    "pat",
                    "pt",
                    "qat",
                    "qt",
                    "rat",
                    "rt",
                    "sat",
                    "st",
                    "t",
                    "ta",
                    "tat",
                    "tt",
                    "uat",
                    "ut",
                    "vat",
                    "vt",
                    "wat",
                    "wt",
                    "xat",
                    "xt",
                    "yat",
                    "yt",
                    "zat",
                    "zt",
                ]
            ),
        },
        {
            "name": "long_check",
            "input": {"word": "can"},
            "expected": set(
                [
                    "aan",
                    "acan",
                    "acn",
                    "an",
                    "ban",
                    "bcan",
                    "ca",
                    "caa",
                    "caan",
                    "cab",
                    "cabn",
                    "cac",
                    "cacn",
                    "cad",
                    "cadn",
                    "cae",
                    "caen",
                    "caf",
                    "cafn",
                    "cag",
                    "cagn",
                    "cah",
                    "cahn",
                    "cai",
                    "cain",
                    "caj",
                    "cajn",
                    "cak",
                    "cakn",
                    "cal",
                    "caln",
                    "cam",
                    "camn",
                    "cana",
                    "canb",
                    "canc",
                    "cand",
                    "cane",
                    "canf",
                    "cang",
                    "canh",
                    "cani",
                    "canj",
                    "cank",
                    "canl",
                    "canm",
                    "cann",
                    "cano",
                    "canp",
                    "canq",
                    "canr",
                    "cans",
                    "cant",
                    "canu",
                    "canv",
                    "canw",
                    "canx",
                    "cany",
                    "canz",
                    "cao",
                    "caon",
                    "cap",
                    "capn",
                    "caq",
                    "caqn",
                    "car",
                    "carn",
                    "cas",
                    "casn",
                    "cat",
                    "catn",
                    "cau",
                    "caun",
                    "cav",
                    "cavn",
                    "caw",
                    "cawn",
                    "cax",
                    "caxn",
                    "cay",
                    "cayn",
                    "caz",
                    "cazn",
                    "cban",
                    "cbn",
                    "ccan",
                    "ccn",
                    "cdan",
                    "cdn",
                    "cean",
                    "cen",
                    "cfan",
                    "cfn",
                    "cgan",
                    "cgn",
                    "chan",
                    "chn",
                    "cian",
                    "cin",
                    "cjan",
                    "cjn",
                    "ckan",
                    "ckn",
                    "clan",
                    "cln",
                    "cman",
                    "cmn",
                    "cn",
                    "cna",
                    "cnan",
                    "cnn",
                    "coan",
                    "con",
                    "cpan",
                    "cpn",
                    "cqan",
                    "cqn",
                    "cran",
                    "crn",
                    "csan",
                    "csn",
                    "ctan",
                    "ctn",
                    "cuan",
                    "cun",
                    "cvan",
                    "cvn",
                    "cwan",
                    "cwn",
                    "cxan",
                    "cxn",
                    "cyan",
                    "cyn",
                    "czan",
                    "czn",
                    "dan",
                    "dcan",
                    "ean",
                    "ecan",
                    "fan",
                    "fcan",
                    "gan",
                    "gcan",
                    "han",
                    "hcan",
                    "ian",
                    "ican",
                    "jan",
                    "jcan",
                    "kan",
                    "kcan",
                    "lan",
                    "lcan",
                    "man",
                    "mcan",
                    "nan",
                    "ncan",
                    "oan",
                    "ocan",
                    "pan",
                    "pcan",
                    "qan",
                    "qcan",
                    "ran",
                    "rcan",
                    "san",
                    "scan",
                    "tan",
                    "tcan",
                    "uan",
                    "ucan",
                    "van",
                    "vcan",
                    "wan",
                    "wcan",
                    "xan",
                    "xcan",
                    "yan",
                    "ycan",
                    "zan",
                    "zcan",
                ]
            ),
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert isinstance(result, set)
            successful_cases += 1
        except:
            failed_cases.append(
                {"name": test_case["name"], "expected": set, "got": type(result)}
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
                f"Wrong number of elements in the returned set.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert sorted(result) == sorted(test_case["expected"])
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
                f"Wrong output values.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_edit_two_letters(target):
    failed_cases = []
    successful_cases = 0

    test_cases = [
        {
            "name": "default_check1",
            "input": {"word": "a"},
            "expected": {
                "expected_n_words_edit_dist": 2654,
                "expected_head": [
                    "",
                    "a",
                    "aa",
                    "aaa",
                    "aab",
                    "aac",
                    "aad",
                    "aae",
                    "aaf",
                    "aag",
                ],
                "expected_tail": [
                    "zv",
                    "zva",
                    "zw",
                    "zwa",
                    "zx",
                    "zxa",
                    "zy",
                    "zya",
                    "zz",
                    "zza",
                ],
            },
        },
        {
            "name": "default_check2",
            "input": {"word": "at"},
            "expected": {
                "expected_n_words_edit_dist": 7154,
                "expected_head": [
                    "",
                    "a",
                    "aa",
                    "aaa",
                    "aaat",
                    "aab",
                    "aabt",
                    "aac",
                    "aact",
                    "aad",
                ],
                "expected_tail": [
                    "zwt",
                    "zx",
                    "zxat",
                    "zxt",
                    "zy",
                    "zyat",
                    "zyt",
                    "zz",
                    "zzat",
                    "zzt",
                ],
            },
        },
        {
            "name": "switches_check",
            "input": {"word": "at", "allow_switches": False},
            "expected": {
                "expected_n_words_edit_dist": 7130,
                "expected_head": [
                    "",
                    "a",
                    "aa",
                    "aaa",
                    "aaat",
                    "aab",
                    "aabt",
                    "aac",
                    "aact",
                    "aad",
                ],
                "expected_tail": [
                    "zwt",
                    "zx",
                    "zxat",
                    "zxt",
                    "zy",
                    "zyat",
                    "zyt",
                    "zz",
                    "zzat",
                    "zzt",
                ],
            },
        },
        {
            "name": "long_check_no_switch",
            "input": {"word": "cat", "allow_switches": False},
            "expected": {
                "expected_n_words_edit_dist": 14206,
                "expected_head": [
                    "a",
                    "aa",
                    "aaa",
                    "aaat",
                    "aab",
                    "aabt",
                    "aac",
                    "aacat",
                    "aact",
                    "aad",
                ],
                "expected_tail": [
                    "zwt",
                    "zxat",
                    "zxcat",
                    "zxt",
                    "zyat",
                    "zycat",
                    "zyt",
                    "zzat",
                    "zzcat",
                    "zzt",
                ],
            },
        },
        {
            "name": "long_check_no_switch",
            "input": {"word": "cat"},
            "expected": {
                "expected_n_words_edit_dist": 14352,
                "expected_head": [
                    "a",
                    "aa",
                    "aaa",
                    "aaat",
                    "aab",
                    "aabt",
                    "aac",
                    "aacat",
                    "aact",
                    "aad",
                ],
                "expected_tail": [
                    "zwt",
                    "zxat",
                    "zxcat",
                    "zxt",
                    "zyat",
                    "zycat",
                    "zyt",
                    "zzat",
                    "zzcat",
                    "zzt",
                ],
            },
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert isinstance(result, set)
            successful_cases += 1
        except:
            failed_cases.append(
                {"name": test_case["name"], "expected": set, "got": type(result),}
            )
            print(
                f"Wrong output type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert len(result) == test_case["expected"]["expected_n_words_edit_dist"]
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["expected_n_words_edit_dist"],
                    "got": len(result),
                }
            )
            print(
                f"Wrong output type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert sorted(list(result))[:10] == sorted(
                test_case["expected"]["expected_head"]
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["expected_head"],
                    "got": result[:10],
                }
            )
            print(
                f"First ten output values are wrong.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert sorted(list(result))[-10:] == sorted(
                test_case["expected"]["expected_tail"]
            )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["expected_tail"],
                    "got": result[:10],
                }
            )
            print(
                f"Last ten output values are wrong.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases


def test_get_corrections(target, probs, vocab):
    failed_cases = []
    successful_cases = 0

    test_cases = [
        {
            "name": "default_check",
            "input": {"word": "dys", "probs": probs, "vocab": vocab, "n": 2},
            "expected": {
                "n_best": [
                    ("days", 0.0004103405826836274),
                    ("dye", 1.865184466743761e-5),
                ]
            },
        },
        {
            "name": "default_check",
            "input": {"word": "satr", "probs": probs, "vocab": vocab, "n": 2},
            "expected": {
                "n_best": [
                    ("star", 0.00013056291267206328),
                    ("sat", 5.595553400231283e-05),
                ]
            },
        },
        {
            "name": "default_check",
            "input": {"word": "san", "probs": probs, "vocab": vocab, "n": 5},
            "expected": {
                "n_best": [
                    ("say", 0.0019770955347483865),
                    ("can", 0.0019211400007460738),
                    ("an", 0.0017719252434065728),
                    ("man", 0.0013242809713880702),
                    ("son", 0.0007274219420300668),
                ]
            },
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert isinstance(result, list)
            successful_cases += 1
        except:
            failed_cases.append(
                {"name": test_case["name"], "expected": list, "got": type(result),}
            )
            print(
                f"Wrong output type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert len(result) == len(test_case["expected"]["n_best"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": len(test_case["expected"]["n_best"]),
                    "got": len(result),
                }
            )
            print(
                f"Wrong output type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            for index, elem in enumerate(sorted(result)):
                assert elem[0] == sorted(test_case["expected"]["n_best"])[index][0]
                assert np.isclose(
                    elem[1], sorted(test_case["expected"]["n_best"])[index][1]
                )
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["n_best"],
                    "got": result,
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

    # return failed_cases, len(failed_cases) + successful_cases


def test_min_edit_distance(target):
    failed_cases = []
    successful_cases = 0

    test_cases = [
        {
            "name": "default_check1",
            "input": {
                "source": "play",
                "target": "stay",
                "ins_cost": 1,
                "del_cost": 1,
                "rep_cost": 2,
            },
            "expected": {
                "D": np.array(
                    [
                        [0, 1, 2, 3, 4],
                        [1, 2, 3, 4, 5],
                        [2, 3, 4, 5, 6],
                        [3, 4, 5, 4, 5],
                        [4, 5, 6, 5, 4],
                    ]
                ),
                "med": 4,
            },
        },
        {
            "name": "default_check2",
            "input": {
                "source": "eer",
                "target": "near",
                "ins_cost": 1,
                "del_cost": 1,
                "rep_cost": 2,
            },
            "expected": {
                "D": np.array(
                    [[0, 1, 2, 3, 4], [1, 2, 1, 2, 3], [2, 3, 2, 3, 4], [3, 4, 3, 4, 3]]
                ),
                "med": 3,
            },
        },
        {
            "name": "nonmodified_costs_check",
            "input": {
                "source": "star",
                "target": "stack",
                "ins_cost": 1,
                "del_cost": 1,
                "rep_cost": 2,
            },
            "expected": {
                "D": np.array(
                    [
                        [0, 1, 2, 3, 4, 5],
                        [1, 0, 1, 2, 3, 4],
                        [2, 1, 0, 1, 2, 3],
                        [3, 2, 1, 0, 1, 2],
                        [4, 3, 2, 1, 2, 3],
                    ]
                ),
                "med": 3,
            },
        },
        {
            "name": "modified_costs_check",
            "input": {
                "source": "star",
                "target": "stack",
                "ins_cost": 2,
                "del_cost": 2,
                "rep_cost": 3,
            },
            "expected": {
                "D": np.array(
                    [
                        [0, 2, 4, 6, 8, 10],
                        [2, 0, 2, 4, 6, 8],
                        [4, 2, 0, 2, 4, 6],
                        [6, 4, 2, 0, 2, 4],
                        [8, 6, 4, 2, 3, 5],
                    ]
                ),
                "med": 5,
            },
        },
    ]

    for test_case in test_cases:
        result = target(**test_case["input"])

        try:
            assert isinstance(result[0], type(result[0]))
            assert isinstance(result[1], type(result[1]))
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": (type(result[0]), type(result[1])),
                    "got": (type(result[0]), type(result[1])),
                }
            )
            print(
                f"Wrong output type.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert result[0].shape == test_case["expected"]["D"].shape
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["D"].shape,
                    "got": result[0].shape,
                }
            )
            print(
                f"Distance matrix has wrong shape.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.allclose(result[0], test_case["expected"]["D"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["D"],
                    "got": result[0],
                }
            )
            print(
                f"Distance matrix has wrong values.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

        try:
            assert np.isclose(result[1], test_case["expected"]["med"])
            successful_cases += 1
        except:
            failed_cases.append(
                {
                    "name": test_case["name"],
                    "expected": test_case["expected"]["med"],
                    "got": result[1],
                }
            )
            print(
                f"Wrong value for minimum edit distance.\n\tExpected: {failed_cases[-1].get('expected')}.\n\tGot: {failed_cases[-1].get('got')}."
            )

    if len(failed_cases) == 0:
        print("\033[92m All tests passed")
    else:
        print("\033[92m", successful_cases, " Tests passed")
        print("\033[91m", len(failed_cases), " Tests failed")

    # return failed_cases, len(failed_cases) + successful_cases
