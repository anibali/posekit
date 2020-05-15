import weakref
from typing import Optional, List, Callable, TypeVar, Generic


T = TypeVar('T')
class WeakStack(Generic[T]):
    def __init__(self, callback: Callable[[Optional[T], Optional[T]], None]):
        """Create a stack containing weak references to values.

        Items in the stack will be replaced with None when there are no other references to them.

        Args:
            callback: Function to be called when the top of the stack changes. It will be passed
                the old top value and new top value as arguments.
        """
        self._list: List[weakref.ReferenceType] = []
        self._callback = callback

    def empty(self):
        """Return True if and only if there are no items on the stack.
        """
        return len(self._list) == 0

    def peek(self) -> Optional[T]:
        """Return the top item of the stack.

        Returns:
            The top item of the stack if the stack is not empty, otherwise None.
        """
        if self.empty():
            return None
        return self._list[-1]()

    def _set_top(self, value: T):
        if self.empty():
            old_top = None
        else:
            old_top = self._pop()
        self._push(value)
        return old_top

    def _push(self, value: T):
        self._list.append(weakref.ref(value))

    def _pop(self) -> T:
        return self._list.pop()()

    def set_top(self, value: T):
        """Replace the top of the stack with a new value.

        Args:
            value: The new value for the top of the stack.

        Returns:
            The previous top, or None if the stack was empty.
        """
        old_top = self._set_top(value)
        self._callback(old_top, value)
        return old_top

    def push(self, value: T):
        """Push a new item onto the stack.

        Args:
            value: The new item.
        """
        old_top = self.peek()
        self._push(value)
        self._callback(old_top, value)

    def pop(self) -> T:
        """Remove the item on top of the stack.

        Returns:
            The removed item.
        """
        old_top = self._pop()
        top = self.peek()
        self._callback(old_top, top)
        return old_top
