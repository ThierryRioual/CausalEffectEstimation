class EmptyConditionException(ZeroDivisionError):
    def __init__(self, message=None) -> None:
        self.message = "No matching instances found in the data for the "\
            "provided conditions.\nPlease ensure the conditions "\
            "are correctly specified or consider using a Pandas "\
            "DataFrame with these conditions containing intervention "\
            "and control instances for estimation purposes."
        super().__init__(self.message)

METHOD_TEXT = "\n(Use `use[estimator_name]()` to select an estimator.)"

class BackdoorException(ValueError):
    def __init__(self, estimator_name=None) -> None:
        self.message = (
            f"The specified estimator: '{estimator_name}' is not supported "\
            "by the backdoor criterion. Please choose a supported estimator:"
            "\n- CausalModelEstimator"
            "\n- SLearner"
            "\n- TLearner"
            "\n- XLearner"
            "\n- PStratification"
            "\n- IPW"
            + METHOD_TEXT
        )
        super().__init__(self.message)

class FrontdoorException(ValueError):
    def __init__(self, estimator_name=None) -> None:
        self.message = (
            f"The specified estimator: '{estimator_name}' is not supported "\
            "by the (genralized) frontdoor criterion. Please choose a "\
            "supported estimator:"
            "\n- CausalModelEstimator"\
            "\n- SimplePlugIn"\
            "\n- GeneralizedPlugIn"
            + METHOD_TEXT
        )
        super().__init__(self.message)

class IVException(ValueError):
    def __init__(self, estimator_name=None) -> None:
        self.message = (
            f"The specified estimator: '{estimator_name}' is not supported "\
            "by the (conditional) instrumental variable criterion. Please "\
            "choose a supported estimator:"
            "\n- CausalModelEstimator"\
            "\n- Wald"\
            "\n- WaldIPW"\
            "\n- NormalizedWaldIPW"\
            "\n- TSLS"
            + METHOD_TEXT
        )
        super().__init__(self.message)