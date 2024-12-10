"""NIfTI to MRD Conversion Utilities."""

import warnings

import numpy as np
import pydicom

from dcmstack.dcmstack import default_group_keys, default_close_keys, DicomStack
from dcmstack.utils import iteritems


def parse_and_group(
    dsets,
    group_by=default_group_keys,
    extractor=None,
    force=False,
    warn_on_except=False,
    close_tests=default_close_keys,
):
    """Parse the given dicom files and group them together.

    Each group is stored as a (list) value in a dict where the key is a tuple of values
    corresponding to the keys in 'group_by'.

    Parameters
    ----------
    src_paths : sequence
        A list of paths to the source DICOM files.

    group_by : tuple
        Meta data keys to group data sets with. Any data set with the same
        values for these keys will be grouped together. This tuple of values
        will also be the key in the result dictionary.

    extractor : callable
        Should take a pydicom.dataset.Dataset and return a dictionary of the
        extracted meta data.

    force : bool
        Force reading source files even if they do not appear to be DICOM.

    warn_on_except : bool
        Convert exceptions into warnings, possibly allowing some results to be
        returned.

    close_tests : sequence
        Any `group_by` key listed here is tested with `numpy.allclose` instead
        of straight equality when determining group membership.

    Returns
    -------
    groups : dict
        A dict mapping tuples of values (corresponding to 'group_by') to groups
        of data sets. Each element in the list is a tuple containing the dicom
        object, the parsed meta data, and the filename.
    """
    if extractor is None:
        from dcmstack.extract import default_extractor

        extractor = default_extractor

    results = {}
    for n in range(len(dsets)):
        dcm = dsets[n]

        # Extract the meta data and group
        meta = extractor(dcm)
        key_list = []  # Values from group_by elems with equality testing
        close_list = []  # Values from group_by elems with np.allclose testing
        for grp_key in group_by:
            key_elem = meta.get(grp_key)
            if isinstance(key_elem, list) or isinstance(
                key_elem, pydicom.multival.MultiValue
            ):
                key_elem = tuple(key_elem)
            if grp_key in close_tests:
                close_list.append(key_elem)
            else:
                key_list.append(key_elem)

        # Initially each key has multiple sub_results (corresponding to
        # different values of the "close" keys)
        key = tuple(key_list)
        if not key in results:
            results[key] = [(close_list, [(dcm, meta, str(n))])]
        else:
            # Look for a matching sub_result
            for c_list, sub_res in results[key]:
                for c_idx, c_val in enumerate(c_list):
                    if not (
                        (c_val is None and close_list[c_idx] is None)
                        or np.allclose(c_val, close_list[c_idx], atol=5e-5)
                    ):
                        break
                else:
                    sub_res.append((dcm, meta, str(n)))
                    break
            else:
                # No match found, append another sub result
                results[key].append((close_list, [(dcm, meta, str(n))]))

    # Unpack sub results, using the canonical value for the close keys
    full_results = {}
    for eq_key, sub_res_list in iteritems(results):
        for close_key, sub_res in sub_res_list:
            full_key = []
            eq_idx = 0
            close_idx = 0
            for grp_key in group_by:
                if grp_key in close_tests:
                    full_key.append(close_key[close_idx])
                    close_idx += 1
                else:
                    full_key.append(eq_key[eq_idx])
                    eq_idx += 1
            full_key = tuple(full_key)
            full_results[full_key] = sub_res

    return dict(sorted(full_results.items()))


def stack_group(group, warn_on_except=False, **stack_args):
    result = DicomStack(**stack_args)
    for dcm, meta, fn in group:
        try:
            result.add_dcm(dcm, meta)
        except Exception as e:
            if warn_on_except:
                warnings.warn("Error adding file %s to stack: %s" % (fn, str(e)))
            else:
                raise
    return result


def parse_and_stack(
    src_paths,
    group_by=default_group_keys,
    extractor=None,
    force=False,
    warn_on_except=False,
    **stack_args,
):
    """Parse the given dicom files into a dictionary containing one or more
    DicomStack objects.

    Parameters
    ----------
    src_paths : sequence
        A list of paths to the source DICOM files.

    group_by : tuple
        Meta data keys to group data sets with. Any data set with the same
        values for these keys will be grouped together. This tuple of values
        will also be the key in the result dictionary.

    extractor : callable
        Should take a pydicom.dataset.Dataset and return a dictionary of the
        extracted meta data.

    force : bool
        Force reading source files even if they do not appear to be DICOM.

    warn_on_except : bool
        Convert exceptions into warnings, possibly allowing some results to be
        returned.

    stack_args : kwargs
        Keyword arguments to pass to the DicomStack constructor.
    """
    results = parse_and_group(src_paths, group_by, extractor, force, warn_on_except)

    for key, group in iteritems(results):
        results[key] = stack_group(group, warn_on_except, **stack_args)

    return results
