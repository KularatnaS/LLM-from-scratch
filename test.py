ds = ['hello cat', 'ya ya to', 'well well']

def get_all():
    for i in ds:
        yield i

print(list(get_all()))