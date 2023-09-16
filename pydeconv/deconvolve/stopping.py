class Callback:
    def __call__(self, history, iteration):
        """Evaluate the iteration and return True to stop, False to continue."""
        raise NotImplementedError


class EarlyStopping(Callback):
    pass


class NoStopping(EarlyStopping):
    def __call__(self, history, iteration):
        return False


class MaxIterationsStopping(EarlyStopping):
    def __init__(self, max_iterations):
        self.max_iterations = max_iterations

    def __call__(self, steps, iteration):
        return iteration >= self.max_iterations


class ThresholdStopping(EarlyStopping):
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, steps, iteration):
        return history[-1] < self.threshold


class DeltaStopping(EarlyStopping):
    def __init__(self, delta=1e-4):
        self.delta = delta

    def __call__(self, steps, iteration):
        return np.abs(history[-1] - history[-2]) < self.delta


def stopping_decorator(early_stopping_criteria):
    def decorator(func):
        def wrapper(*args, **kwargs):
            history = kwargs.get("steps", None)
            iteration = kwargs.get("iteration", None)

            # Assume the first argument (after self if a method) is 'steps'
            steps = args[1] if len(args) > 1 else None

            if early_stopping_criteria(steps, iteration):
                return steps[: iteration + 1]
            return func(*args, **kwargs)

        return wrapper

    return decorator
