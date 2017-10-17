import setuptools as st


st.setup(
    name='pytrails',
    version='0.0.4',
    packages=st.find_packages(),

    # metadata for upload to PyPI
    author="Martin Becker",
    author_email="becker@informatik.uni-wuerzburg.de",
    description="A collection of Bayesian methods for hypothesis comparison on sequential data.",
    keywords="bayesian model comparison bayes factor Markov model sequence sequential data",
    url="https://github.com/mgbckr/pytrails",
)
