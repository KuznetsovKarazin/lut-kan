# src/schemas/config.py
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, ConfigDict, model_validator


AutoInt = Union[Literal["auto"], int]
AutoStr = Literal["auto"]


class ExperimentConfig(BaseModel):
    name: str
    group: str
    description: str = ""
    timestamp: Optional[Literal["auto"]] = "auto"


class RuntimeConfig(BaseModel):
    seed: int = 42
    device: Literal["cpu", "cuda"] = "cpu"
    num_threads: Optional[int] = 1
    deterministic: bool = True


class DatasetPreprocessConfig(BaseModel):
    normalize: bool = True
    normalize_mode: Literal["standard", "minmax", "none"] = "standard"


class DatasetSplitConfig(BaseModel):
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15

    @model_validator(mode="after")
    def _sum_to_one(self):
        s = float(self.train) + float(self.val) + float(self.test)
        if abs(s - 1.0) > 1e-6:
            raise ValueError(f"dataset.split must sum to 1.0, got {s}")
        return self


class DatasetConfig(BaseModel):
    name: str
    path: str
    task: Literal["regression", "classification"] = "regression"
    split: DatasetSplitConfig = Field(default_factory=DatasetSplitConfig)
    preprocess: DatasetPreprocessConfig = Field(default_factory=DatasetPreprocessConfig)


class CalibrationConfig(BaseModel):
    source_split: Literal["train", "val"] = "train"
    num_samples: int = 1024
    batch_size: int = 32
    sampling: Literal["random", "first", "stratified"] = "random"
    seed: int = 42

    @model_validator(mode="after")
    def _positive(self):
        if self.num_samples <= 0:
            raise ValueError("calibration.num_samples must be > 0")
        if self.batch_size <= 0:
            raise ValueError("calibration.batch_size must be > 0")
        return self


class FloatModelConfig(BaseModel):
    backend: str = "pykan"
    checkpoint: str
    adapter: str
    arch: Optional[Dict[str, Any]] = None


class TrainingConfig(BaseModel):
    enabled: bool = False
    load_from_checkpoint: bool = True
    epochs: int = 100
    lr: float = 1e-3
    batch_size: int = 32

    @model_validator(mode="after")
    def _train_positive(self):
        if self.enabled:
            if self.epochs <= 0:
                raise ValueError("training.epochs must be > 0")
            if self.lr <= 0:
                raise ValueError("training.lr must be > 0")
            if self.batch_size <= 0:
                raise ValueError("training.batch_size must be > 0")
        return self


class BuildLUTConfig(BaseModel):
    L: int = 64
    knots_source: Literal["model"] = "model"
    x_range_mode: Literal["knots", "empirical", "margin"] = "knots"
    margin_std: float = 3.0

    @model_validator(mode="after")
    def _lut(self):
        if self.L < 2:
            raise ValueError("converter.build_lut.L must be >= 2")
        if self.x_range_mode == "margin" and self.margin_std <= 0:
            raise ValueError("converter.build_lut.margin_std must be > 0 for margin mode")
        return self


class OOBPolicyConfig(BaseModel):
    # allow extra fields such as boundary=half_open/closed (new contract)
    model_config = ConfigDict(extra="allow")

    mode: Literal["clip_x", "saturate_y", "zero_spline"] = "clip_x"


class InterpConfig(BaseModel):
    mode: Literal["nearest", "linear"] = "linear"


class YRangeConfig(BaseModel):
    method: Literal["minmax", "percentile"] = "minmax"
    lower_percentile: float = 0.1
    upper_percentile: float = 99.9

    @model_validator(mode="after")
    def _percentiles(self):
        if self.method == "percentile":
            if not (0.0 <= self.lower_percentile < self.upper_percentile <= 100.0):
                raise ValueError("converter.y_range percentiles must satisfy 0<=lower<upper<=100")
        return self


class QuantConfig(BaseModel):
    dtype: Literal["uint8", "int8"] = "uint8"
    scheme: Literal["asymmetric", "symmetric"] = "asymmetric"
    use_full_int8_range: bool = False
    qmin: AutoInt = "auto"
    qmax: AutoInt = "auto"
    zero_point: Union[Literal["auto"], int] = "auto"
    per_segment: bool = True
    meta_dtype: Literal["float16", "float32"] = "float16"
    store_float_lut: bool = False

    @model_validator(mode="after")
    def _quant_rules(self):
        # uint8 -> asymmetric only
        if self.dtype == "uint8" and self.scheme != "asymmetric":
            raise ValueError("converter.quant: uint8 requires asymmetric scheme")

        # symmetric -> zero_point must be 0
        if self.scheme == "symmetric":
            if self.zero_point != 0:
                raise ValueError("converter.quant: symmetric scheme requires zero_point = 0")

        # asymmetric -> zero_point must be 'auto' (offset-by-y_min design)
        if self.scheme == "asymmetric":
            if self.zero_point != "auto":
                raise ValueError(
                    "converter.quant: asymmetric scheme uses offset-y_min and does not accept zero_point (must be 'auto')"
                )

        # qmin/qmax constraints when explicitly set
        if isinstance(self.qmin, int) and isinstance(self.qmax, int):
            if self.qmin >= self.qmax:
                raise ValueError("converter.quant: qmin must be < qmax")
            if self.dtype == "uint8":
                if not (0 <= self.qmin <= 255 and 0 <= self.qmax <= 255):
                    raise ValueError("converter.quant: uint8 qmin/qmax must be within [0,255]")
            if self.dtype == "int8":
                if not (-128 <= self.qmin <= 127 and -128 <= self.qmax <= 127):
                    raise ValueError("converter.quant: int8 qmin/qmax must be within [-128,127]")
                if self.scheme == "symmetric":
                    if not (self.qmin == -self.qmax):
                        raise ValueError("converter.quant: int8 symmetric requires qmin == -qmax")
                    if self.qmin == -128 and not self.use_full_int8_range:
                        raise ValueError("converter.quant: qmin=-128 requires use_full_int8_range=true")

        # use_full_int8_range is meaningful only for int8 asymmetric
        if self.use_full_int8_range and not (self.dtype == "int8" and self.scheme == "asymmetric"):
            raise ValueError("converter.quant: use_full_int8_range is only valid for int8 asymmetric")

        return self


class ConverterConfig(BaseModel):
    # allow extra fields such as value_kind/value_representation, etc.
    model_config = ConfigDict(extra="allow")

    enabled: bool = True
    build_lut: BuildLUTConfig = Field(default_factory=BuildLUTConfig)
    oob_policy: OOBPolicyConfig = Field(default_factory=OOBPolicyConfig)
    interp: InterpConfig = Field(default_factory=InterpConfig)
    y_range: YRangeConfig = Field(default_factory=YRangeConfig)
    quant: QuantConfig = Field(default_factory=QuantConfig)


class InferenceConfig(BaseModel):
    batch_size: int = 1024
    enable_batching: bool = True
    accumulation: Literal["fp32"] = "fp32"
    fixed_point_interp: Optional[Dict[str, Any]] = None

    @model_validator(mode="after")
    def _infer(self):
        if self.batch_size <= 0:
            raise ValueError("inference.batch_size must be > 0")
        return self


class PhiErrorConfig(BaseModel):
    enable: bool = True
    metrics: List[Literal["mae", "rmse", "max_abs"]] = Field(default_factory=lambda: ["mae", "rmse", "max_abs"])
    per_segment: bool = True
    report_topk_functions: int = 3


class SpeedConfig(BaseModel):
    enable: bool = True
    warmup_iters: int = 10
    measure_iters: int = 100


class MemoryConfig(BaseModel):
    enable: bool = True
    breakdown: bool = True


class EvaluationExtraConfig(BaseModel):
    phi_error: PhiErrorConfig = Field(default_factory=PhiErrorConfig)
    speed: SpeedConfig = Field(default_factory=SpeedConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)


class EvaluationConfig(BaseModel):
    metrics: Dict[str, List[str]] = Field(
        default_factory=lambda: {"regression": ["mse", "mae", "r2"], "classification": ["accuracy", "f1"]}
    )
    extra: EvaluationExtraConfig = Field(default_factory=EvaluationExtraConfig)


class LoggingConfig(BaseModel):
    out_dir: str = "outputs/exp_runs"
    versioning: Literal["git_hash", "timestamp", "increment"] = "git_hash"
    artifact_format: Literal["npz", "torch"] = "npz"
    save_artifact: bool = True
    save_predictions: bool = False
    save_plots: bool = True
    save_intermediate: bool = False
    plot_worst_functions: int = 3
    verbose: bool = True


class DebugConfig(BaseModel):
    enable: bool = False
    dataset_limit_samples: Optional[int] = None
    save_all_luts: bool = False
    plot_every_function: bool = False
    dump_quant_stats: bool = False
    validate_numerics: bool = True
    epsilon: float = 1e-7

    @model_validator(mode="after")
    def _debug(self):
        if self.dataset_limit_samples is not None and self.dataset_limit_samples <= 0:
            raise ValueError("debug.dataset_limit_samples must be > 0 or null")
        if self.epsilon <= 0:
            raise ValueError("debug.epsilon must be > 0")
        return self


class ValidationConfig(BaseModel):
    strict: bool = True
    rules: List[str] = Field(default_factory=list)


class RootConfig(BaseModel):
    # allow extra keys at the root too (helps during refactors)
    model_config = ConfigDict(extra="allow")

    version: float = 1.2
    experiment: ExperimentConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    dataset: DatasetConfig
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    float_model: FloatModelConfig
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    converter: ConverterConfig = Field(default_factory=ConverterConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    @model_validator(mode="after")
    def _mvp_rules(self):
        if self.validation.strict:
            if self.converter.enabled and self.converter.build_lut.knots_source != "model":
                raise ValueError("MVP strict: converter.build_lut.knots_source must be 'model'")
        return self
