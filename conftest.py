"""Project-wide pytest settings (loaded in addition to tests/conftest.py)."""


def pytest_configure(config):
    # matplotlib 3.9 calls pyparsing functions (oneOf, parseString, ...) that newer pyparsing
    # releases mark as deprecated. This is third-party noise, not our code, so only that
    # message pattern, raised from inside matplotlib, is silenced. Every other warning still shows.
    config.addinivalue_line("filterwarnings", r"ignore:.*deprecated - use:Warning:matplotlib\..*")
