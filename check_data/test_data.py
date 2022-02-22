import pandas as pd
import scipy.stats

# Non Deterministic Test
def test_kolmogorov_smirnov(data, ks_alpha):

    sample1, sample2 = data

    numerical_columns = [
        "Time","V1","V2","V3","V4","V5","V6","V7",
        "V8","V9","V10","V11","V12","V13","V14","V15",
        "V16","V17","V18","V19","V20","V21","V22","V23",
        "V24","V25","V26","V27","V28","Amount","Class"
    ]
    
    # Bonferroni correction for multiple hypothesis testing
    alpha_prime = 1 - (1 - ks_alpha)**(1 / len(numerical_columns))

    for col in numerical_columns:

        # two-sided: The null hypothesis is that the two distributions are identical
        # the alternative is that they are not identical.
        ts, p_value = scipy.stats.ks_2samp(
            sample1[col],
            sample2[col],
            alternative='two-sided'
        )

        # NOTE: as always, the p-value should be interpreted as the probability of
        # obtaining a test statistic (TS) equal or more extreme that the one we got
        # by chance, when the null hypothesis is true. If this probability is not
        # large enough, this dataset should be looked at carefully, hence we fail
        assert p_value > alpha_prime
        
# Determinstic Test
def test_column_presence_and_type(data):
    
    # Disregard the reference dataset
    _, df = data

    required_columns = {
        "Time": pd.api.types.is_float_dtype,
        "Class": pd.api.types.is_int64_dtype
    }

    # Check column presence
    assert set(df.columns.values).issuperset(set(required_columns.keys()))

    for col_name, format_verification_funct in required_columns.items():

        assert format_verification_funct(df[col_name]), f"Column {col_name} failed test {format_verification_funct}"

# Deterministic Test
def test_class_names(data):
    
    # Disregard the reference dataset
    _, df = data

    # Check that only the known classes are present
    known_classes = [
        1,
        0
    ]

    assert df["Class"].isin(known_classes).all()

# Deterministic Test
def test_column_ranges(data):
    
    # Disregard the reference dataset
    _, df = data

    ranges = {
        "Time": (0, 10000000.0),
        "Amount": (0, 100000.0)
    }

    for col_name, (minimum, maximum) in ranges.items():

        assert df[col_name].dropna().between(minimum, maximum).all(), (
            f"Column {col_name} failed the test. Should be between {minimum} and {maximum}, "
            f"instead min={df[col_name].min()} and max={df[col_name].max()}"
        )