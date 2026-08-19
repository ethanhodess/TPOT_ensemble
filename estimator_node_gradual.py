# custom class for nodes
# try https://automl.github.io/ConfigSpace/main/api/hyperparameters.html

import numpy as np
from tpot2.search_spaces.base import SklearnIndividual, SearchSpace
from ConfigSpace import ConfigurationSpace
from typing import final
import ConfigSpace

NONE_SPECIAL_STRING = "<NONE>"
TRUE_SPECIAL_STRING = "<TRUE>"
FALSE_SPECIAL_STRING = "<FALSE>"

def default_hyperparameter_parser(params:dict) -> dict:
    return params

# NOTE: This is not the default, currently experimental
class EstimatorNodeIndividualGradual(SklearnIndividual):
    """
    Note that ConfigurationSpace does not support None as a parameter. Instead, use the special string "<NONE>". TPOT will automatically replace instances of this string with the Python None.

    Parameters
    ----------
    method : type
        The class of the estimator to be used

    space : ConfigurationSpace|dict
        The hyperparameter space to be used. If a dict is passed, hyperparameters are fixed and not learned.

    """
    def __init__(self, method: type,
                        space: ConfigurationSpace|dict,
                        hyperparameter_parser: callable = None,
                        rng=None) -> None:
        super().__init__()
        self.method = method
        self.space = space

        if hyperparameter_parser is None:
            self.hyperparameter_parser = default_hyperparameter_parser
        else:
            self.hyperparameter_parser = hyperparameter_parser

        if isinstance(space, dict):
            self.hyperparameters = space
        else:
            rng = np.random.default_rng(rng)
            self.space.seed(rng.integers(0, 2**32))
            self.hyperparameters = dict(self.space.sample_configuration())

        self.check_hyperparameters_for_None()

    def mutate(self, rng=None):
        if isinstance(self.space, dict):
            return False

        rng = np.random.default_rng(rng)

        # coin flip to determine if we skip mutation mutation
        if rng.choice([True, False]):
            return False

        # mutate the hyperparameters one of two ways

        # 1. randomly sample a new set of parameters with a 10% chance
        if rng.choice([True, False], p=[0.1, 0.9]):
            self.space.seed(rng.integers(0, 2**32))
            self.hyperparameters = dict(self.space.sample_configuration())
        else:
            # 2. gradually update the hyperparameters with a 90% chance
            self.hyperparameters = gradual_hyperparameter_update(params=self.hyperparameters, configspace=self.space, rng=rng)
        self.check_hyperparameters_for_None()
        return True

    def crossover(self, other, rng=None):
        if isinstance(self.space, dict):
            return False

        rng = np.random.default_rng(rng)
        if self.method != other.method:
            return False

        #loop through hyperparameters, randomly swap items in self.hyperparameters with items in other.hyperparameters
        for hyperparameter in self.space:
            if rng.choice([True, False]):
                if hyperparameter in other.hyperparameters:
                    self.hyperparameters[hyperparameter] = other.hyperparameters[hyperparameter]

        self.check_hyperparameters_for_None()

        return True

    def check_hyperparameters_for_None(self):
        for key, value in self.hyperparameters.items():
            #if string
            if isinstance(value, str):
                if value == NONE_SPECIAL_STRING:
                    self.hyperparameters[key] = None
                elif value == TRUE_SPECIAL_STRING:
                    self.hyperparameters[key] = True
                elif value == FALSE_SPECIAL_STRING:
                    self.hyperparameters[key] = False

    @final #this method should not be overridden, instead override hyperparameter_parser
    def export_pipeline(self, **kwargs):
        return self.method(**self.hyperparameter_parser(self.hyperparameters))

    def unique_id(self):
        #return a dictionary of the method and the hyperparameters
        method_str = self.method.__name__
        params = list(self.hyperparameters.keys())
        params = sorted(params)

        id_str = f"{method_str}({', '.join([f'{param}={self.hyperparameters[param]}' for param in params])})"

        return id_str

# function to gradually update numerical hyperparameters to explore the hyperparameter space more effectively
# will not update categorical hyperparameters, as we want to keep the same categories and tune the values paired them
def gradual_hyperparameter_update(params:dict, configspace:ConfigurationSpace, rng=None):
    # use the same rng for reproducibility
    rng = np.random.default_rng(rng)

    # will be used to store the new hyperparameters
    new_params = {}

    for param in list(params.keys()):
        try:
            # if parameter is float
            if issubclass(type(configspace[param]), ConfigSpace.hyperparameters.hyperparameter.FloatHyperparameter):

                # log scale or not?
                if configspace[param].log:
                    new_params[param] = params[param] * rng.lognormal(0, 1)
                else:
                    new_params[param] = params[param] * rng.normal(1.0, 0.1)

                # if check if above or below min and cap
                if new_params[param] < configspace[param].lower:
                    new_params[param] = configspace[param].lower
                elif new_params[param] > configspace[param].upper:
                    new_params[param] = configspace[param].upper

            #if parameter is integer, add normal distribution
            elif issubclass(type(configspace[param]), ConfigSpace.hyperparameters.hyperparameter.IntegerHyperparameter):
                new_params[param] = params[param] * rng.normal(1.0, 0.1)

                # if check if above or below min and cap
                if new_params[param] < configspace[param].lower:
                    new_params[param] = configspace[param].lower
                elif new_params[param] > configspace[param].upper:
                    new_params[param] = configspace[param].upper
                new_params[param] = int(new_params[param])
            # TODO : add support for categorical hyperparameters
            else:
                new_params[param] = params[param]
        except:
            pass

    return new_params

class EstimatorNodeGradual(SearchSpace):
    def __init__(self, method, space, hyperparameter_parser=default_hyperparameter_parser):
        self.method = method
        self.space = space
        self.hyperparameter_parser = hyperparameter_parser

    def generate(self, rng=None):
        return EstimatorNodeIndividualGradual(self.method, self.space, hyperparameter_parser=self.hyperparameter_parser, rng=rng)