def fullname(obj):
    """ Resolve the full module-qualified name of an object. Typically used for logger naming. """
    return obj.__module__ + "." + obj.__class__.__name__
