"""Validate and rewrite CF variable relationships, without reading field data.

References
----------
CF Metadata Conventions 1.12, attributes and related variables:
https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html
MetPy's metadata adapter provides an organizational reference:
https://github.com/Unidata/MetPy/blob/v1.7.1/src/metpy/xarray.py

Climate-Base improvements
-------------------------
References are checked in both attrs and encoding because xarray's CF decoder
can move them. Renaming updates relationships rather than silently leaving
old names. This is a limited consistency check, not CF compliance certification.
No third-party implementation has been copied.
"""

import re

from .exceptions import MetadataError
from .validation import require_dataset

_SINGLE = {"bounds", "climatology"}
_LIST = {"coordinates", "ancillary_variables"}
_PAIRS = {"cell_measures", "formula_terms"}
_RELATIONS = _SINGLE | _LIST | _PAIRS | {"grid_mapping"}
_PAIR_PATTERN = re.compile(r"(?:^|\s)([^\s:]+)\s*:\s*([^\s:]+)")


def _without_cell_methods(value, axes):
    """Remove transformed-axis clauses, retaining other CF methods/comments."""
    searchable = re.sub(r"\([^)]*\)", lambda match: " " * len(match[0]), value)
    clauses = list(re.finditer(r"(?<!\S)(?:[^\s:()]+\s*:\s*)+", searchable))
    retained = []
    for index, clause in enumerate(clauses):
        labels = {axis.strip() for axis in clause[0].split(":")[:-1]}
        end = clauses[index + 1].start() if index + 1 < len(clauses) else len(value)
        if not labels.intersection(axes):
            retained.append(value[clause.start():end].strip())
    return " ".join(retained)


def _attribute(variable, attribute):
    left = variable.attrs.get(attribute)
    right = variable.encoding.get(attribute)
    for value in (left, right):
        if value is not None and not isinstance(value, str):
            raise MetadataError(f"{attribute} must be a string")
    if left is not None and right is not None and left.strip() != right.strip():
        raise MetadataError(f"Conflicting {attribute!r} in attrs and encoding")
    return left if left is not None else right


def _pairs(value, attribute):
    pairs = _PAIR_PATTERN.findall(value)
    if not pairs or _PAIR_PATTERN.sub("", value).strip():
        raise MetadataError(f"Invalid CF {attribute}: {value!r}")
    return pairs


def _references(attribute, value):
    if not value or not value.strip():
        raise MetadataError(f"CF {attribute} must not be empty")
    if attribute in _SINGLE:
        if len(value.split()) != 1 or ":" in value:
            raise MetadataError(f"CF {attribute} requires one variable name")
        return value.split()
    if attribute in _LIST:
        return value.split()
    if attribute in _PAIRS:
        return [name for _, name in _pairs(value, attribute)]
    # CF 1.12 permits either 'crs' or 'crs: x y other_crs: lat lon'.
    if ":" in value:
        parts = re.split(r"(?<!\S)([^\s:]+)\s*:\s*", value.strip())
        if parts[0] or len(parts) < 3 or len(parts) % 2 != 1:
            raise MetadataError(f"Invalid CF grid_mapping: {value!r}")
        references = []
        for mapping, coordinates in zip(parts[1::2], parts[2::2]):
            if not coordinates.strip() or ":" in coordinates:
                raise MetadataError(f"Expanded grid_mapping requires coordinate names: {value!r}")
            references.extend([mapping, *coordinates.split()])
        return references
    if len(value.split()) != 1:
        raise MetadataError("Unexpanded grid_mapping requires one variable name")
    return value.split()


def validate_metadata(dataset):
    """Check supported CF references in attrs/encoding; return None on success.

    No data values are read. This checks reference existence and basic grammar,
    not projection validity, formula physics or complete CF conformance.
    """
    require_dataset(dataset)
    for name, variable in dataset.variables.items():
        for attribute in _RELATIONS:
            value = _attribute(variable, attribute)
            if value is None:
                continue
            for reference in _references(attribute, value):
                if reference not in dataset.variables:
                    raise MetadataError(
                        f"Variable {name!r} has dangling {attribute} reference {reference!r}"
                    )


def rename_cf_references(dataset, mapping):
    """Return a metadata-only copy with references updated after a rename.

    Formula labels such as ``a:`` remain labels; only their variable values are
    changed. Cell-method axis labels are renamed without changing the method.
    The actual variables must already have been renamed by the caller.
    """
    require_dataset(dataset)
    result = dataset.copy(deep=False)
    for variable in result.variables.values():
        for container in (variable.attrs, variable.encoding):
            for attribute in _RELATIONS:
                if attribute not in container:
                    continue
                value = container[attribute]
                if not isinstance(value, str):
                    raise MetadataError(f"{attribute} must be a string")
                if attribute in _PAIRS:
                    container[attribute] = " ".join(
                        f"{label}: {mapping.get(name, name)}"
                        for label, name in _pairs(value, attribute)
                    )
                else:
                    container[attribute] = re.sub(
                        r"[^\s:]+", lambda match: mapping.get(match[0], match[0]), value
                    )
            if "cell_methods" in container:
                value = container["cell_methods"]
                if isinstance(value, str):
                    # CF comments may contain text such as 'time: original'.
                    # Only actual axis labels outside comments are renamed.
                    searchable = re.sub(r"\([^)]*\)", lambda match: " " * len(match[0]), value)
                    matches = list(re.finditer(r"(?<!\S)([^\s:()]+)(?=\s*:)", searchable))
                    for match in reversed(matches):
                        value = value[:match.start()] + mapping.get(match[0], match[0]) + value[match.end():]
                    container["cell_methods"] = value
    result.set_close(dataset.close)
    validate_metadata(result)
    return result
