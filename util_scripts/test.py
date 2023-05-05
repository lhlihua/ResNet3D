lists = ['a','b','c']

def process(item):
    return item + ' pro'

def printlists(lists):
    for item in lists:
        x = process(item)
        y = process(item)
        yield x , y

g = printlists(lists=lists)
print(next(g))
print('-'*20)
print(next(g))
