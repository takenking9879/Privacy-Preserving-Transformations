"""Duck-typed synthesizer registry.

``SYNTHESIZERS`` maps a short name to a callable with signature::

    fn(df, n, seed, include_targets=True) -> pandas.DataFrame

Registration is best-effort.  If a synthesizer module is missing (or has
no compatible ``fit``/``sample`` / ``synthesize_*`` API) it is skipped
with a :class:`RuntimeWarning` instead of raising.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import warnings
from typing import Any, Callable, Iterable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

SynthesizerFn = Callable[..., pd.DataFrame]
SYNTHESIZERS: dict[str, SynthesizerFn] = {}

# Preferred import paths / attribute names for the three families we try.
_SPECS: dict[str, dict[str, tuple[str, ...]]] = {
    "copula": {
        "modules": (
            "src.synthesizers.gaussian_copula",
            "src.synthesizers.copula",
            "src.synthesizers.copulas",
            "src.synthesizers.gaussian_copulas",
        ),
        "functions": (
            "synthesize_copula",
            "synthesize_gaussian_copula",
            "generate_copula",
            "synthesize",
            "generate",
            "generate_synthetic",
            "fit_sample",
            "fit_and_sample",
        ),
        "classes": (
            "GaussianCopulaSynthesizer",
            "CopulaSynthesizer",
            "GaussianCopula",
            "Copula",
        ),
    },
    "cart": {
        "modules": (
            "src.synthesizers.cart_sequential",
            "src.synthesizers.sequential_cart",
            "src.synthesizers.cart",
            "src.synthesizers.synthpop",
        ),
        "functions": (
            "synthesize_cart",
            "synthesize_sequential_cart",
            "generate_cart",
            "synthesize",
            "generate",
            "generate_synthetic",
            "fit_sample",
            "fit_and_sample",
        ),
        "classes": (
            "CARTSequentialSynthesizer",
            "SequentialCARTSynthesizer",
            "CARTSynthesizer",
            "CartSynthesizer",
            "CART",
        ),
    },
    "hybrid": {
        "modules": (
            "src.synthesizers.hybrid",
            "src.synthesizers.copula_conditional",
        ),
        "functions": (
            "synthesize_hybrid",
            "synthesize",
            "generate",
            "generate_synthetic",
            "fit_sample",
            "fit_and_sample",
        ),
        "classes": (
            "HybridSynthesizer",
            "CopulaConditionalSynthesizer",
        ),
    },
    "mixture": {
        "modules": (
            "src.synthesizers.conditional_mixture",
            "src.synthesizers.mixture",
            "src.synthesizers.gmm",
            "src.synthesizers.gaussian_mixture",
        ),
        "functions": (
            "synthesize_mixture",
            "synthesize_conditional_mixture",
            "generate_mixture",
            "synthesize",
            "generate",
            "generate_synthetic",
            "fit_sample",
            "fit_and_sample",
        ),
        "classes": (
            "ConditionalMixtureSynthesizer",
            "MixtureSynthesizer",
            "GaussianMixtureSynthesizer",
            "Mixture",
        ),
    },
    "forest": {
        "modules": (
            "src.synthesizers.forest_sequential",
        ),
        "functions": (
            "synthesize_forest",
            "synthesize",
            "generate",
            "generate_synthetic",
            "fit_sample",
            "fit_and_sample",
        ),
        "classes": (
            "ForestSequentialSynthesizer",
        ),
    },
    "knn": {
        "modules": (
            "src.synthesizers.knn_conditional",
        ),
        "functions": (
            "synthesize_knn",
            "synthesize",
            "generate",
            "generate_synthetic",
            "fit_sample",
            "fit_and_sample",
        ),
        "classes": (
            "KNNConditionalSynthesizer",
        ),
    },
    "auto": {
        "modules": (
            "src.synthesizers.auto",
        ),
        "functions": (
            "synthesize_auto",
            "synthesize",
            "generate",
            "generate_synthetic",
            "fit_sample",
            "fit_and_sample",
        ),
        "classes": (
            "AutoSynthesizer",
        ),
    },
    "ensemble": {
        "modules": (
            "src.synthesizers.ensemble",
        ),
        "functions": (
            "synthesize_ensemble",
            "synthesize",
            "generate",
            "generate_synthetic",
            "fit_sample",
            "fit_and_sample",
        ),
        "classes": (
            "EnsembleSynthesizer",
        ),
    },
}


def _is_dataframe(obj: Any) -> bool:
    return isinstance(obj, pd.DataFrame)


def _filter_kwargs(fn: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs
    accepted = {
        name
        for name, p in sig.parameters.items()
        if p.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    return {k: v for k, v in kwargs.items() if k in accepted}


def _try_calls(attempts: Iterable[Callable[[], Any]]) -> Any:
    last_err: Optional[BaseException] = None
    for attempt in attempts:
        try:
            out = attempt()
        except TypeError as exc:
            last_err = exc
            continue
        if _is_dataframe(out):
            return out
        if out is not None and not _is_dataframe(out):
            # Fluent fit() returning self is not a failure for sample; skip.
            last_err = TypeError(f"call returned {type(out).__name__}, not DataFrame")
            continue
    if last_err is not None:
        raise last_err
    raise TypeError("no compatible call produced a DataFrame")


def _target_kwargs(include_targets: bool) -> dict[str, Any]:
    """Both polarities: some APIs take ``include_targets``, copula uses ``exclude_targets``."""
    return {
        "include_targets": include_targets,
        "exclude_targets": not include_targets,
        "with_targets": include_targets,
    }


def _invoke_function(
    fn: Callable[..., Any],
    df: pd.DataFrame,
    n: int,
    seed: int,
    include_targets: bool,
) -> pd.DataFrame:
    kw_full = {
        "n": n,
        "seed": seed,
        "n_samples": n,
        "random_state": seed,
        **_target_kwargs(include_targets),
    }
    # Never pass the include/exclude flag positionally: the 4th argument of
    # synthesize_copula is ``exclude_targets`` (inverted polarity).
    return _try_calls(
        (
            lambda: fn(df, n, seed, include_targets=include_targets),
            lambda: fn(df, n, seed, exclude_targets=not include_targets),
            lambda: fn(df, **_filter_kwargs(fn, kw_full)),
            lambda: fn(df, n, seed),
            lambda: fn(df, n),
            lambda: fn(real_data=df, **_filter_kwargs(fn, kw_full)),
        )
    )


def _construct(cls: type, seed: int) -> Any:
    attempts = (
        lambda: cls(seed=seed),
        lambda: cls(random_state=seed),
        lambda: cls(),
    )
    last_err: Optional[BaseException] = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_err = exc
    if last_err is not None:
        raise last_err
    return cls()


def _invoke_class(
    cls: type,
    df: pd.DataFrame,
    n: int,
    seed: int,
    include_targets: bool,
) -> pd.DataFrame:
    obj = _construct(cls, seed)

    if callable(obj) and not isinstance(obj, type):
        try:
            return _invoke_function(obj, df, n, seed, include_targets)
        except TypeError:
            pass

    fit = getattr(obj, "fit", None)
    if callable(fit):
        fit_kwargs_list = (
            {**_target_kwargs(include_targets), "seed": seed, "random_state": seed},
            _target_kwargs(include_targets),
            {"include_targets": include_targets},
            {"exclude_targets": not include_targets},
            {"seed": seed},
            {},
        )
        fitted = False
        last_err: Optional[BaseException] = None
        for kw in fit_kwargs_list:
            try:
                result = fit(df, **_filter_kwargs(fit, kw))
                fitted = True
                if result is not None and hasattr(result, "sample"):
                    obj = result
                break
            except TypeError as exc:
                last_err = exc
        if not fitted and last_err is not None:
            raise last_err

    for meth_name in ("sample", "generate", "synthesize"):
        meth = getattr(obj, meth_name, None)
        if not callable(meth):
            continue
        try:
            return _try_calls(
                (
                    lambda m=meth: m(n, seed=seed),
                    lambda m=meth: m(n, random_state=seed),
                    lambda m=meth: m(n, seed),
                    lambda m=meth: m(n),
                    lambda m=meth: m(n_samples=n, seed=seed),
                    lambda m=meth: m(**_filter_kwargs(m, {"n": n, "seed": seed, "random_state": seed})),
                )
            )
        except TypeError:
            continue
    raise TypeError(f"{cls.__name__} has no compatible sample/generate method")


def adapt(obj: Any, name: str) -> SynthesizerFn:
    """Wrap a function or ``fit``/``sample`` class into the registry callable."""

    if isinstance(obj, type):

        def _class_wrapper(
            df: pd.DataFrame,
            n: int,
            seed: int,
            include_targets: bool = True,
            _cls: type = obj,
            _name: str = name,
        ) -> pd.DataFrame:
            return _invoke_class(_cls, df, int(n), int(seed), bool(include_targets))

        _class_wrapper.__name__ = f"{name}_class_adapter"
        _class_wrapper.__doc__ = f"Adapted class {obj!r} as synthesizer '{name}'."
        return _class_wrapper

    if not callable(obj):
        raise TypeError(f"Cannot adapt {obj!r} for synthesizer '{name}'")

    def _fn_wrapper(
        df: pd.DataFrame,
        n: int,
        seed: int,
        include_targets: bool = True,
        _fn: Callable[..., Any] = obj,
        _name: str = name,
    ) -> pd.DataFrame:
        return _invoke_function(_fn, df, int(n), int(seed), bool(include_targets))

    _fn_wrapper.__name__ = f"{name}_fn_adapter"
    _fn_wrapper.__doc__ = f"Adapted callable {getattr(obj, '__name__', obj)!r} as synthesizer '{name}'."
    return _fn_wrapper


def _public_attrs(module: Any) -> list[tuple[str, Any]]:
    names = list(getattr(module, "__all__", []) or [])
    if not names:
        names = [n for n in dir(module) if not n.startswith("_")]
    out: list[tuple[str, Any]] = []
    for n in names:
        try:
            out.append((n, getattr(module, n)))
        except AttributeError:
            continue
    return out


def _pick_from_module(module: Any, spec: dict[str, tuple[str, ...]]) -> Optional[Any]:
    for attr in spec["functions"]:
        obj = getattr(module, attr, None)
        if callable(obj) and not isinstance(obj, type):
            return obj
    for attr in spec["classes"]:
        obj = getattr(module, attr, None)
        if isinstance(obj, type) and (
            hasattr(obj, "fit") or hasattr(obj, "sample") or callable(obj)
        ):
            return obj
    # Duck-type: first public function, then first class with fit+sample.
    functions: list[Any] = []
    classes: list[Any] = []
    for _name, obj in _public_attrs(module):
        if isinstance(obj, type):
            if hasattr(obj, "fit") and (
                hasattr(obj, "sample") or hasattr(obj, "generate")
            ):
                classes.append(obj)
        elif callable(obj):
            functions.append(obj)
    if functions:
        return functions[0]
    if classes:
        return classes[0]
    return None


def try_register(name: str) -> bool:
    """Import and register ``name`` if a compatible module exists.

    Returns True when ``SYNTHESIZERS[name]`` is set.
    """
    if name in SYNTHESIZERS:
        return True
    spec = _SPECS.get(name)
    if spec is None:
        warnings.warn(
            f"Unknown synthesizer family '{name}'; not registered.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False

    import_errors: list[str] = []
    for mod_path in spec["modules"]:
        try:
            module = importlib.import_module(mod_path)
        except ImportError as exc:
            import_errors.append(f"{mod_path} ({exc})")
            logger.debug("Could not import %s for '%s': %s", mod_path, name, exc)
            continue
        except Exception as exc:  # noqa: BLE001 — registry must not crash the runner
            warnings.warn(
                f"Importing {mod_path} for synthesizer '{name}' failed: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        picked = _pick_from_module(module, spec)
        if picked is None:
            warnings.warn(
                f"Module {mod_path} loaded but has no compatible synthesizer API "
                f"for '{name}'.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        try:
            SYNTHESIZERS[name] = adapt(picked, name)
        except Exception as exc:  # noqa: BLE001
            warnings.warn(
                f"Could not adapt {picked!r} from {mod_path} as '{name}': {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        logger.info("Registered synthesizer '%s' from %s (%s)", name, mod_path, picked)
        return True

    warnings.warn(
        f"Synthesizer '{name}' not registered (module missing or incompatible). "
        f"Tried: {', '.join(import_errors) if import_errors else ', '.join(spec['modules'])}.",
        RuntimeWarning,
        stacklevel=2,
    )
    return False


def register(name: str, fn: SynthesizerFn) -> SynthesizerFn:
    """Explicitly register a callable (useful in tests)."""
    SYNTHESIZERS[name] = adapt(fn, name)
    return SYNTHESIZERS[name]


def available() -> list[str]:
    return list(SYNTHESIZERS.keys())


def _register_defaults() -> None:
    for name in ("copula", "cart", "mixture", "hybrid"):
        try_register(name)
    # Optional families: register after the original four so a missing
    # module only skips that family (try_register warns, does not raise).
    for name in ("forest", "knn", "auto", "ensemble"):
        try_register(name)


_register_defaults()

__all__ = [
    "SYNTHESIZERS",
    "adapt",
    "available",
    "register",
    "try_register",
]
