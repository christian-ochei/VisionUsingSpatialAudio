import gc
import json
import marshal
import os
import pickle
import threading
import traceback
import zmq
import platform

ACTUALLY_STICKY = False

if ACTUALLY_STICKY:
    # Prepare our context and sockets
    context = zmq.Context()
    #  Socket to talk to server
    print("Connecting StickyMemory [STUNN_EDIT_AUTOMATICALLY]")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5556")
else:
    print('StickyMemory is Disabled ✔')

def b_jsonify(dict_: dict):
    return b'json' + json.dumps(dict_).encode('utf-8')

def b_code(function):
    return b'code' + marshal.dumps(function.__code__)


class MemoryManager:
    lock = threading.RLock()
    ALL_INITIALIZATIONS = []

    @staticmethod
    def delete_all_sticky_memory(force_delete=False):
        if os.name != 'nt':
            return

        if not ACTUALLY_STICKY:
            print(f'Deleting all sticky-memory | '
                  f'ALL_INITIALIZATIONS: {[a["new_items"] for a in MemoryManager.ALL_INITIALIZATIONS]}', flush=True)
            try:
                with StickyMemory.lock:
                    for class_data in MemoryManager.ALL_INITIALIZATIONS:
                        for key in class_data['new_items']:
                                total_refs = gc.get_referrers(getattr(class_data['class'], key))
                                total_refs = len(total_refs)
                                if total_refs <= 1:
                                    attr = getattr(class_data['class'], key)
                                    type_n = attr.__class__.__name__
                                    if isinstance(attr, StickyMemory.SafeDeletableWrapper):
                                        if attr.deletable_context_manager.deletable():
                                            ...
                                            # attr.__del__()
                                            # delattr(class_data['class'], key)
                                            # print(f'Deleted SAFE {class_data["class"].__name__}.{key} - type: {type_n}',
                                            #       flush=True)
                                        else:
                                            print(f'IGNORED {class_data["class"].__name__}.{key} - type: {type_n}',
                                                  flush=True)
                                    else:
                                        delattr(class_data['class'], key)
                                        print(f'Deleted {class_data["class"].__name__}.{key} - type: {type_n}', flush=True)
                                else:
                                    print(f'Skipping deletion of {class_data["class"].__name__}.{key} because {total_refs=}', flush=True)
                            # except AttributeError:
                            #     pass
                        class_data['class_dict']['initialized'] = False
                        class_data['class'].initialized = False
                        # We just initialized out models twice, because refrences exists,
                        # the model is not deleted but initialized is false
                    gc.collect()
                    MemoryManager.ALL_INITIALIZATIONS = []
                    print(f'Deleted all sticky-memory | ALL_INITIALIZATIONS: {MemoryManager.ALL_INITIALIZATIONS}', flush=True)
            except:
                traceback.print_exc()
        else:
            if ACTUALLY_STICKY:
                print('Skipping MemoryManager.delete_all_sticky_memory because ACTUALLY_STICKY', flush=True)
            else:
                print('Skipping MemoryManager.delete_all_sticky_memory because IS_RUNNING_ON_THE_CLOUD', flush=True)


class StickyMemory(type):
    lock = threading.RLock()
    classes = []

    def __init__(self, name, bases, class_dict):
        self.name = name
        self.bases = bases
        self.class_dict = class_dict
        self.class_dict['initialized'] = False
        self.initialized = False
        StickyMemory.classes.append(self)

    def initialize_function_wrapper(class_):
        with StickyMemory.lock:
            if not class_.initialized:
                initialize_function = class_.__sticky_initialize__
                socket.send(b_jsonify({'make_class_sticky': class_.__name__}))
                assert socket.recv() == b''
                socket.send(b_code(initialize_function))
                error = socket.recv()
                if error != b'':
                    raise RuntimeError(error[5:].decode())

                class_.initialized = True

            socket.send(b_jsonify({'fetch_all_keys': class_.__name__}))
            all_keys = json.loads(socket.recv())
            all_keys = set(all_keys) - set(class_.class_dict.keys())
            all_keys = [key for key in all_keys if not key.startswith('__')]

            for key in all_keys:
                class_.class_dict[key] = StickyMemory.ValueWrapper(key, [class_.__name__])

    def __getattr__(self, item):
        if ACTUALLY_STICKY:
            with StickyMemory.lock:
                if item in self.class_dict:
                    return self.class_dict[item]
                else:
                    print(f'StickyInitializing "{self.name}" because of "{item}"', flush=True)
                    self.initialize_function_wrapper()
                if item in self.class_dict:
                    return self.class_dict[item]
                raise AttributeError(f'{self.__name__} has no attribute {item}')
        else:
            # with StickyMemory.lock:
                if not self.class_dict['initialized']:
                    print(f'StickyInitializing "{self.name}" because of "{item}"', flush=True)
                    existing_items = set(self.__dict__.keys())
                    self.__sticky_initialize__()
                    self.class_dict['initialized'] = True
                    new_items = {
                        'new_items': set(self.__dict__.keys()) - existing_items,
                        'class': self,
                        'class_dict': self.class_dict
                    }
                    # for key in new_items['new_items']:
                        # we want to use the super of getattr(self, key)
                        # if callable(super(StickyMemory, self).__getattribute__(key)):
                        #     print(f'Wrapping {key} because it is callable')
                        #     setattr(self, key, StickyMemory.SafeDeletableWrapper(super(StickyMemory, self).__getattribute__(key)))

                    MemoryManager.ALL_INITIALIZATIONS.append(new_items)
                    assert hasattr(self, item)
        return getattr(self, item)

    class _CalledContextManager:
        def __init__(self):
            print('Creating context manager')
            self.instances = 0

        def deletable(self):
            return self.instances <= 0

        def __enter__(self):
            print('Entering context manager')
            self.instances += 1

        def __exit__(self, exc_type, exc_val, exc_tb):
            print('Exiting context manager')
            self.instances -= 1
            assert self.instances >= 0

    class SafeDeletableWrapper:
        def __init__(self, obj):
            self.obj = obj
            self.deletable_context_manager = StickyMemory._CalledContextManager()

        def __getattr__(self, item):
            return getattr(self.obj, item)

        def __del__(self):
            print(f'Deleting {self.obj}')
            del self.obj

        def __call__(self, *args, **kwargs):
            with self.deletable_context_manager:
                # How about multiple threads? We use instances count to take care of that
                return self.obj(*args, **kwargs)

        __str__ = lambda self: self.__getattr__('__str__')()
        __repr__ = lambda self: self.__getattr__('__repr__')()
        __len__ = lambda self: self.__getattr__('__len__')()
        __getitem__ = lambda self, item: self.__getattr__('__getitem__')(item)
        __setitem__ = lambda self, item, value: self.__getattr__('__setitem__')(item, value)
        __delitem__ = lambda self, item: self.__getattr__('__delitem__')(item)
        __iter__ = lambda self: self.__getattr__('__iter__')()
        __contains__ = lambda self, item: self.__getattr__('__contains__')(item)
        __add__ = lambda self, other: self.__getattr__('__add__')(other)
        __sub__ = lambda self, other: self.__getattr__('__sub__')(other)
        __mul__ = lambda self, other: self.__getattr__('__mul__')(other)
        __floordiv__ = lambda self, other: self.__getattr__('__floordiv__')(other)
        __mod__ = lambda self, other: self.__getattr__('__mod__')(other)
        __pow__ = lambda self, other: self.__getattr__('__pow__')(other)
        __lshift__ = lambda self, other: self.__getattr__('__lshift__')(other)
        __rshift__ = lambda self, other: self.__getattr__('__rshift__')(other)
        __and__ = lambda self, other: self.__getattr__('__and__')(other)
        __xor__ = lambda self, other: self.__getattr__('__xor__')(other)
        __or__ = lambda self, other: self.__getattr__('__or__')(other)
        __divmod__ = lambda self, other: self.__getattr__('__divmod__')(other)
        __radd__ = lambda self, other: self.__getattr__('__radd__')(other)
        __rsub__ = lambda self, other: self.__getattr__('__rsub__')(other)
        __rmul__ = lambda self, other: self.__getattr__('__rmul__')(other)
        __rtruediv__ = lambda self, other: self.__getattr__('__rtruediv__')(other)
        __rfloordiv__ = lambda self, other: self.__getattr__('__rfloordiv__')(other)
        __rmod__ = lambda self, other: self.__getattr__('__rmod__')(other)
        __rpow__ = lambda self, other: self.__getattr__('__rpow__')(other)
        __rlshift__ = lambda self, other: self.__getattr__('__rlshift__')(other)
        __rrshift__ = lambda self, other: self.__getattr__('__rrshift__')(other)
        __rand__ = lambda self, other: self.__getattr__('__rand__')(other)
        __rxor__ = lambda self, other: self.__getattr__('__rxor__')(other)
        __ror__ = lambda self, other: self.__getattr__('__ror__')(other)
        __iadd__ = lambda self, other: self.__getattr__('__iadd__')(other)
        __isub__ = lambda self, other: self.__getattr__('__isub__')(other)
        __imul__ = lambda self, other: self.__getattr__('__imul__')(other)
        __itruediv__ = lambda self, other: self.__getattr__('__itruediv__')(other)
        __ifloordiv__ = lambda self, other: self.__getattr__('__ifloordiv__')(other)
        __imod__ = lambda self, other: self.__getattr__('__imod__')(other)
        __ipow__ = lambda self, other: self.__getattr__('__ipow__')(other)
        __ilshift__ = lambda self, other: self.__getattr__('__ilshift__')(other)
        __irshift__ = lambda self, other: self.__getattr__('__irshift__')(other)
        __iand__ = lambda self, other: self.__getattr__('__iand__')(other)
        __ixor__ = lambda self, other: self.__getattr__('__ixor__')(other)
        __ior__ = lambda self, other: self.__getattr__('__ior__')(other)
        __neg__ = lambda self: self.__getattr__('__neg__')()
        __pos__ = lambda self: self.__getattr__('__pos__')()
        __abs__ = lambda self: self.__getattr__('__abs__')()
        __invert__ = lambda self: self.__getattr__('__invert__')()
        __complex__ = lambda self: self.__getattr__('__complex__')()
        __int__ = lambda self: self.__getattr__('__int__')()
        __float__ = lambda self: self.__getattr__('__float__')()
        __round__ = lambda self, n: self.__getattr__('__round__')(n)
        __index__ = lambda self: self.__getattr__('__index__')()
        __trunc__ = lambda self: self.__getattr__('__trunc__')()
        __floor__ = lambda self: self.__getattr__('__floor__')()
        __ceil__ = lambda self: self.__getattr__('__ceil__')()
        __enter__ = lambda self: self.__getattr__('__enter__')()
        __exit__ = lambda self, exc_type, exc_val, exc_tb: self.__getattr__('__exit__')(exc_type, exc_val, exc_tb)
        __hash__ = lambda self: self.__getattr__('__hash__')()
        __bool__ = lambda self: self.__getattr__('__bool__')()
        __str__ = lambda self: self.__getattr__('__str__')()
        __repr__ = lambda self: self.__getattr__('__repr__')()
        __len__ = lambda self: self.__getattr__('__len__')()
        __getitem__ = lambda self, item: self.__getattr__('__getitem__')(item)
        __setitem__ = lambda self, key, value: self.__getattr__('__setitem__')(key, value)
        __delitem__ = lambda self, key: self.__getattr__('__delitem__')(key)
        __iter__ = lambda self: self.__getattr__('__iter__')()
        __reversed__ = lambda self: self.__getattr__('__reversed__')()
        __contains__ = lambda self, item: self.__getattr__('__contains__')(item)
        __missing__ = lambda self, key: self.__getattr__('__missing__')(key)

    class ValueWrapper:
        def __init__(self, key, class_chain: [str]):
            self.key = key
            self.class_chain = [*class_chain, self.key]

        def __getattr__(self, item):
            # print(f'Getting {[*self.class_chain, self.key]}', item, '__getattr__')
            if item in ['key', 'class_chain', '__call__']:
                return super().__getattribute__(item)

            with StickyMemory.lock:
                socket.send(b_jsonify({'has_key': item, 'class_chain': self.class_chain}))
                has_key = socket.recv()

            if has_key != b'1':
                raise AttributeError(f'{self.key} has no attribute {item}')
            return StickyMemory.ValueWrapper(item, self.class_chain)

        def __call__(self, *args, **kwargs):
            with StickyMemory.lock:
                # print('call', self.class_chain, args, kwargs)
                packet = {
                    'args': args,
                    'kwargs': kwargs,
                }
                socket.send(b'call'+pickle.dumps({'class_chain': self.class_chain, 'packet': packet}))
                message = socket.recv()
                if message.startswith(b'error'):
                    raise RuntimeError(message[5:].decode())
                # print(f'Finished sending call packet for {self.class_chain}')
                return pickle.loads(message)

        __str__      = lambda self: self.__getattr__('__str__')()
        __repr__     = lambda self: self.__getattr__('__repr__')()
        __len__      = lambda self: self.__getattr__('__len__')()
        __getitem__  = lambda self, item: self.__getattr__('__getitem__')(item)
        __setitem__  = lambda self, item, value: self.__getattr__('__setitem__')(item, value)
        __delitem__  = lambda self, item: self.__getattr__('__delitem__')(item)
        __iter__     = lambda self: self.__getattr__('__iter__')()
        __contains__ = lambda self, item: self.__getattr__('__contains__')(item)
        __add__      = lambda self, other: self.__getattr__('__add__')(other)
        __sub__      = lambda self, other: self.__getattr__('__sub__')(other)
        __mul__      = lambda self, other: self.__getattr__('__mul__')(other)
        __floordiv__ = lambda self, other: self.__getattr__('__floordiv__')(other)
        __mod__      = lambda self, other: self.__getattr__('__mod__')(other)
        __pow__      = lambda self, other: self.__getattr__('__pow__')(other)
        __lshift__   = lambda self, other: self.__getattr__('__lshift__')(other)
        __rshift__   = lambda self, other: self.__getattr__('__rshift__')(other)
        __and__      = lambda self, other: self.__getattr__('__and__')(other)
        __xor__      = lambda self, other: self.__getattr__('__xor__')(other)
        __or__       = lambda self, other: self.__getattr__('__or__')(other)
        __divmod__   = lambda self, other: self.__getattr__('__divmod__')(other)
        __radd__     = lambda self, other: self.__getattr__('__radd__')(other)
        __rsub__     = lambda self, other: self.__getattr__('__rsub__')(other)
        __rmul__     = lambda self, other: self.__getattr__('__rmul__')(other)
        __rtruediv__ = lambda self, other: self.__getattr__('__rtruediv__')(other)
        __rfloordiv__= lambda self, other: self.__getattr__('__rfloordiv__')(other)
        __rmod__     = lambda self, other: self.__getattr__('__rmod__')(other)
        __rpow__     = lambda self, other: self.__getattr__('__rpow__')(other)
        __rlshift__  = lambda self, other: self.__getattr__('__rlshift__')(other)
        __rrshift__  = lambda self, other: self.__getattr__('__rrshift__')(other)
        __rand__     = lambda self, other: self.__getattr__('__rand__')(other)
        __rxor__     = lambda self, other: self.__getattr__('__rxor__')(other)
        __ror__      = lambda self, other: self.__getattr__('__ror__')(other)
        __iadd__     = lambda self, other: self.__getattr__('__iadd__')(other)
        __isub__     = lambda self, other: self.__getattr__('__isub__')(other)
        __imul__     = lambda self, other: self.__getattr__('__imul__')(other)
        __itruediv__ = lambda self, other: self.__getattr__('__itruediv__')(other)
        __ifloordiv__= lambda self, other: self.__getattr__('__ifloordiv__')(other)
        __imod__     = lambda self, other: self.__getattr__('__imod__')(other)
        __ipow__     = lambda self, other: self.__getattr__('__ipow__')(other)
        __ilshift__  = lambda self, other: self.__getattr__('__ilshift__')(other)
        __irshift__  = lambda self, other: self.__getattr__('__irshift__')(other)
        __iand__     = lambda self, other: self.__getattr__('__iand__')(other)
        __ixor__     = lambda self, other: self.__getattr__('__ixor__')(other)
        __ior__      = lambda self, other: self.__getattr__('__ior__')(other)
        __neg__      = lambda self: self.__getattr__('__neg__')()
        __pos__      = lambda self: self.__getattr__('__pos__')()
        __abs__      = lambda self: self.__getattr__('__abs__')()
        __invert__   = lambda self: self.__getattr__('__invert__')()
        __complex__  = lambda self: self.__getattr__('__complex__')()
        __int__      = lambda self: self.__getattr__('__int__')()
        __float__    = lambda self: self.__getattr__('__float__')()
        __round__    = lambda self, n: self.__getattr__('__round__')(n)
        __index__    = lambda self: self.__getattr__('__index__')()
        __trunc__    = lambda self: self.__getattr__('__trunc__')()
        __floor__    = lambda self: self.__getattr__('__floor__')()
        __ceil__     = lambda self: self.__getattr__('__ceil__')()
        __enter__    = lambda self: self.__getattr__('__enter__')()
        __exit__     = lambda self, exc_type, exc_val, exc_tb: self.__getattr__('__exit__')(exc_type, exc_val, exc_tb)
        __hash__     = lambda self: self.__getattr__('__hash__')()
        __bool__     = lambda self: self.__getattr__('__bool__')()
        __str__      = lambda self: self.__getattr__('__str__')()
        __repr__     = lambda self: self.__getattr__('__repr__')()
        __len__      = lambda self: self.__getattr__('__len__')()
        __getitem__  = lambda self, item: self.__getattr__('__getitem__')(item)
        __setitem__  = lambda self, key, value: self.__getattr__('__setitem__')(key, value)
        __delitem__  = lambda self, key: self.__getattr__('__delitem__')(key)
        __iter__     = lambda self: self.__getattr__('__iter__')()
        __reversed__ = lambda self: self.__getattr__('__reversed__')()
        __contains__ = lambda self, item: self.__getattr__('__contains__')(item)
        __missing__  = lambda self, key: self.__getattr__('__missing__')(key)