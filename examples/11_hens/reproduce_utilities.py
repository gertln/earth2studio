import hashlib

import numpy as np


def calculate_torch_seed(s):
    """
    calculates torch seed based on a given string.
    String s is used as input to sha256 hash algorithm.
    Output is converted to integer by taking the maximum integer size of torch seed into account.

    Parameters
    ----------
    s : str
        seed string

    Returns
    -------
    torch: np.int64
        integer value that can be used as random seed in torch

    """
    torch_seed = int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % (2**64) - 1
    return torch_seed


def create_base_seed_string(pkg, ic, base_random_seed):
    """
    concatenates information of model package name, initial condition time and and base_random seed into one base seed string

    Parameters
    ----------
    pkg : str
        model package name
    ic : str
        initial condition time
    base_random_seed : str
        base seed string

    Returns
    -------
    base_seed_string: str
        string that can be used as random seed

    """
    s0 = str(base_random_seed)
    s1 = "".join(
        e for e in pkg if e.isalnum()
    )  # remove all special characters from package name
    s2 = str(ic.astype("datetime64[s]"))
    base_seed_string = "_".join([s0, s1, s2])
    return base_seed_string


def calculate_all_torch_seeds(base_seed_string, batch_ids):
    """
    calculates all torch random seeds that will be used based on the base_seed_string and the batch_ids

    Parameters
    ----------
    base_seed_string : str
        base seed
    batch_ids : list[int]
        list of batch_ids that will be calculated

    Returns
    -------
    full_seed_strings: np.array
        contains all seed strings that will be used to calculate torch seeds
    torch_seeds: np.array
        contains all torch random seeds that will be used

    """
    sall = np.char.add(
        np.array(base_seed_string + "_"), np.array([str(x) for x in batch_ids])
    )
    torch_seeds = np.zeros((len(sall), 1), dtype=np.uint64)
    full_seed_strings = np.empty(np.shape(torch_seeds), dtype=object)
    for i, s in enumerate(sall):
        full_seed_strings[i] = s
        torch_seeds[i] = calculate_torch_seed(s)
    return full_seed_strings, torch_seeds


def check_uniquness_of_torch_seeds(torch_seeds):
    """
    checks if all torch seeds are unique

    Parameters
    ----------
    torch_seeds : np.array
        torch seeds

    Returns
    -------
    all_unique: bool
        True if no duplicates of torch seeds were found

    """
    num_runs = len(torch_seeds)
    num_unique_seeds = len(np.unique(torch_seeds))
    if num_unique_seeds == num_runs:
        all_unique = True
    else:
        all_unique = False
        raise ValueError(
            "Calculated torch seeds for every run must be unique! num_unique_seeds = %s, num_runs = %s"
            % (num_unique_seeds, num_runs)
        )
    return all_unique


def ensure_all_torch_seeds_are_unique(ensemble_configs, base_random_seed):
    """
    checks if all torch seeds based on ensemble_configs and base_random_seed are unique
    """
    torch_seeds_list = []
    full_seed_string_list = []
    for pkg, ic, _, batch_ids_produce in ensemble_configs:
        base_seed_string = create_base_seed_string(pkg, ic, base_random_seed)
        full_seed_strings, torch_seeds = calculate_all_torch_seeds(
            base_seed_string, batch_ids_produce
        )
        if check_uniquness_of_torch_seeds(torch_seeds):
            torch_seeds_list.append(torch_seeds)
            full_seed_string_list.append(full_seed_strings)
    if torch_seeds_list:
        check_uniquness_of_torch_seeds(np.concatenate(torch_seeds_list, axis=0))
    else:
        raise ValueError("Torch seeds could not be calculated.")
