from . import skeleton_registry
from .utils import assert_plausible_skeleton

from toolz.functoolz import identity, compose


class _ConversionRegistry:
    def __init__(self):
        self._functions = {}

    def register(self, from_skeleton_name: str, to_skeleton_name: str):
        def decorator(func):
            def augmented_func(joints):
                from_skeleton = skeleton_registry[from_skeleton_name]
                to_skeleton = skeleton_registry[to_skeleton_name]
                assert_plausible_skeleton(joints, from_skeleton)
                new_joints = func(joints, from_skeleton, to_skeleton)
                assert_plausible_skeleton(new_joints, to_skeleton)
                return new_joints
            self._functions.setdefault(from_skeleton_name, {})[to_skeleton_name] = augmented_func
            return func
        return decorator

    def _compose_conversion_functions(self, from_skeleton_name: str, to_skeleton_name: str):
        if from_skeleton_name == to_skeleton_name:
            return identity
        try:
            queue = list(self._functions[from_skeleton_name].items())
        except KeyError:
            queue = []
        seen = {from_skeleton_name}
        while len(queue) > 0:
            skeleton_name, func = queue.pop()
            if skeleton_name == to_skeleton_name:
                return func
            seen.add(skeleton_name)
            try:
                for next_skeleton_name, next_func in self._functions[skeleton_name].items():
                    if next_skeleton_name not in seen:
                        queue.insert(0, (next_skeleton_name, compose(next_func, func)))
            except KeyError:
                pass
        raise NotImplementedError(f'no skeleton conversion path defined from {from_skeleton_name} to {to_skeleton_name}')

    def convert(self, joints, from_skeleton_name: str, to_skeleton_name: str):
        return self._compose_conversion_functions(from_skeleton_name, to_skeleton_name)(joints)


skeleton_converter = _ConversionRegistry()
