# Execution-labeler image for the family-AB harder corpus (BigCodeBench lib-bearing problems).
#
# Built + invoked by prism.eval.container_sandbox:
#   docker build -t prism-labeler:latest -f eval/docker/labeler.Dockerfile .
#   docker run --rm --network none --read-only --tmpfs /tmp ... prism-labeler:latest python /work/runner.py
#
# The container runs the SAME fixed runner as the in-process sandbox (sandbox._RUNNER_SRC, mounted at
# /work) with reliability_guard active. Network is disabled at RUN time (--network none), so the libs
# here support IMPORT + local computation, not network calls.
FROM python:3.12-slim

# A curated subset of BigCodeBench's high-frequency libraries (the dataset spans ~140 libs; this covers
# the common ones). A problem importing a lib absent here raises ModuleNotFoundError -> familygen SKIPS
# it (never mislabeled buggy). Major-pinned for label reproducibility; refresh deliberately.
RUN pip install --no-cache-dir \
        "numpy<3" \
        "pandas<3" \
        "scipy<2" \
        "scikit-learn<2" \
        "matplotlib<4" \
        "seaborn<1" \
        "sympy<2" \
        "statsmodels<1" \
        "pillow<12" \
        "beautifulsoup4<5" \
        "lxml<6" \
        "openpyxl<4" \
        "python-dateutil<3" \
        "pytz<2026" \
        "regex" \
        "nltk<4" \
        "networkx<4" \
        "requests<3" \
    && useradd --create-home --uid 10001 labeler

USER labeler
WORKDIR /work
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
# The runner is mounted at /work/runner.py at run time; this is the default entry for a bare `docker run`.
CMD ["python", "/work/runner.py"]
