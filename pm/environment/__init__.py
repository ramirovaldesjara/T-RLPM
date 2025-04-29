from gymnasium.envs.registration import register
from .pm_based_portfolio_value import EnvironmentPV
from .pm_based_portfolio_return import EnvironmentRET
from .wrapper import EnvironmentWrapper

register(id = "PortfolioManagement-v0", entry_point=EnvironmentWrapper)