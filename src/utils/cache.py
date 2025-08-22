from functools import wraps
import pickle
import os
import time


def cache_result(duration: int = 3600):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
            except Exception:
                return func(*args, **kwargs)

            cache_dir = 'cache'
            cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")

            try:
                if os.path.exists(cache_file):
                    if time.time() - os.path.getmtime(cache_file) < duration:
                        with open(cache_file, 'rb') as f:
                            return pickle.load(f)
            except Exception:
                pass

            result = func(*args, **kwargs)

            try:
                os.makedirs(cache_dir, exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
            except Exception:
                pass

            return result
        return wrapper
    return decorator


